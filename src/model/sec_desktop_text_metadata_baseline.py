from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("Paper 4")
ITEM_CODES = (
    "1.01",
    "1.02",
    "1.03",
    "2.01",
    "2.02",
    "2.03",
    "2.04",
    "3.01",
    "3.02",
    "3.03",
    "4.01",
    "4.02",
    "5.01",
    "5.02",
    "5.03",
    "5.07",
    "7.01",
    "8.01",
    "9.01",
)
KEYWORDS = (
    "bankruptcy",
    "receivership",
    "default",
    "acceleration",
    "delisting",
    "nasdaq",
    "nyse",
    "notice",
    "debt",
    "credit",
    "agreement",
    "earnings",
    "resignation",
    "director",
    "auditor",
    "amendment",
    "acquisition",
    "merger",
    "sale",
    "restructuring",
    "liquidation",
    "going concern",
    "impairment",
    "material",
    "rights",
)
WINDOWS = (90, 180, 365, 730)
HORIZONS = (365, 730)
MODELS = ("logistic", "random_forest")


def normalize_cik(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else ""


def split_items(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()}


def finite_float(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    return float(value) if isinstance(value, (np.floating, float)) else int(value)


def safe_metric(fn, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return finite_float(fn(y_true, y_score))
    except Exception:
        return None


def top_fraction_metrics(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.1) -> dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {"top_fraction": fraction, "top_n": 0, "precision": None, "recall": None, "lift": None}
    top_n = max(1, int(np.ceil(n * fraction)))
    order = np.argsort(-y_score)
    top = order[:top_n]
    base_rate = float(np.mean(y_true)) if n else 0.0
    precision = float(np.mean(y_true[top])) if top_n else None
    positives = float(np.sum(y_true))
    recall = float(np.sum(y_true[top]) / positives) if positives else None
    lift = float(precision / base_rate) if base_rate and precision is not None else None
    return {
        "top_fraction": fraction,
        "top_n": int(top_n),
        "precision": finite_float(precision),
        "recall": finite_float(recall),
        "lift": finite_float(lift),
    }


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    return {
        "roc_auc": safe_metric(roc_auc_score, y_true, y_score),
        "average_precision": safe_metric(average_precision_score, y_true, y_score),
        "brier_score": finite_float(brier_score_loss(y_true, y_score)),
        "top_10pct": top_fraction_metrics(y_true, y_score, 0.10),
        "top_5pct": top_fraction_metrics(y_true, y_score, 0.05),
    }


def load_base_panel(panel_path: Path, financial_panel_path: Path | None) -> tuple[pd.DataFrame, str]:
    if financial_panel_path and financial_panel_path.exists():
        panel = pd.read_parquet(financial_panel_path)
        source = str(financial_panel_path)
    else:
        panel = pd.read_parquet(panel_path)
        source = str(panel_path)
    panel = panel.copy()
    panel["firm_id"] = panel["firm_id"].map(normalize_cik)
    panel["anchor_date_ts"] = pd.to_datetime(panel["anchor_date"], errors="coerce")
    return panel, source


def load_filings(filings_path: Path) -> pd.DataFrame:
    cols = ["cik", "filing_date", "items", "primary_doc_description", "primary_document"]
    filings = pd.read_csv(filings_path, usecols=lambda col: col in cols, low_memory=False)
    filings["cik"] = filings["cik"].map(normalize_cik)
    filings["filing_date_ts"] = pd.to_datetime(filings["filing_date"], errors="coerce")
    filings = filings[filings["cik"].ne("") & filings["filing_date_ts"].notna()].copy()
    text = (
        filings["items"].fillna("").astype(str)
        + " "
        + filings["primary_doc_description"].fillna("").astype(str)
        + " "
        + filings["primary_document"].fillna("").astype(str)
    ).str.lower()
    item_sets = filings["items"].map(split_items)
    for item in ITEM_CODES:
        filings[f"item_{item.replace('.', '_')}"] = item_sets.map(lambda values, item=item: int(item in values))
    for keyword in KEYWORDS:
        filings[f"kw_{keyword.replace(' ', '_')}"] = text.str.contains(keyword, regex=False).astype(int)
    filings["desc_nonempty"] = filings["primary_doc_description"].notna().astype(int)
    filings["desc_len"] = filings["primary_doc_description"].fillna("").astype(str).str.len().astype(float)
    return filings.sort_values(["cik", "filing_date_ts", "items"]).reset_index(drop=True)


def build_lookup(filings: pd.DataFrame, base_cols: list[str]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for cik, group in filings.groupby("cik", sort=False):
        values = group[base_cols].to_numpy(dtype=float)
        prefix = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
        lookup[str(cik)] = {
            "dates": group["filing_date_ts"].to_numpy(dtype="datetime64[ns]"),
            "prefix": prefix,
        }
    return lookup


def build_text_metadata_features(panel: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    item_cols = [f"item_{item.replace('.', '_')}" for item in ITEM_CODES]
    keyword_cols = [f"kw_{keyword.replace(' ', '_')}" for keyword in KEYWORDS]
    base_cols = item_cols + keyword_cols + ["desc_nonempty", "desc_len"]
    lookup = build_lookup(filings, base_cols)
    anchors = panel[["task", "firm_id", "anchor_year", "anchor_date", "anchor_date_ts"]].drop_duplicates()
    rows: list[dict[str, Any]] = []
    for row in anchors.itertuples(index=False):
        rec: dict[str, Any] = {
            "task": row.task,
            "firm_id": row.firm_id,
            "anchor_year": row.anchor_year,
            "anchor_date": row.anchor_date,
        }
        firm = lookup.get(row.firm_id)
        anchor = np.datetime64(pd.Timestamp(row.anchor_date_ts).to_datetime64())
        if firm is None:
            for window in WINDOWS:
                rec[f"textmeta_w{window}_filing_count"] = 0
                for col in base_cols:
                    rec[f"textmeta_w{window}_{col}_count"] = 0.0
            rows.append(rec)
            continue
        dates = firm["dates"]
        prefix = firm["prefix"]
        end_idx = int(np.searchsorted(dates, anchor, side="right"))
        for window in WINDOWS:
            start = anchor - np.timedelta64(window, "D")
            start_idx = int(np.searchsorted(dates, start, side="left"))
            counts = prefix[end_idx] - prefix[start_idx]
            rec[f"textmeta_w{window}_filing_count"] = int(end_idx - start_idx)
            for idx, col in enumerate(base_cols):
                rec[f"textmeta_w{window}_{col}_count"] = float(counts[idx])
        rows.append(rec)
    features = pd.DataFrame(rows)
    return panel.drop(columns=["anchor_date_ts"], errors="ignore").merge(
        features,
        on=["task", "firm_id", "anchor_year", "anchor_date"],
        how="left",
    )


def textmeta_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("textmeta_")]


def sequence_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("seq_")]


def financial_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("fin_")]


def make_model(model_name: str) -> Pipeline:
    if model_name == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=150,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise ValueError(model_name)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], label: str, model_name: str) -> np.ndarray:
    y_train = train[label].astype(int).to_numpy()
    if not features:
        return np.repeat(float(y_train.mean()) if len(y_train) else 0.0, len(test))
    if len(np.unique(y_train)) < 2:
        return np.repeat(float(y_train.mean()) if len(y_train) else 0.0, len(test))
    model = make_model(model_name)
    model.fit(train[features], y_train)
    return model.predict_proba(test[features])[:, 1]


def evaluate_panel(panel: pd.DataFrame, train_end_year: int, test_start_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    text_features = textmeta_columns(panel)
    seq_features = sequence_columns(panel)
    fin_features = financial_columns(panel)
    feature_sets = {
        "metadata_text_only": text_features,
        "sequence_plus_metadata_text": seq_features + text_features,
    }
    if fin_features:
        feature_sets["sequence_financial_metadata_text"] = seq_features + fin_features + text_features
    metrics: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for task, task_panel in panel.groupby("task"):
        train = task_panel[task_panel["anchor_year"] <= train_end_year].copy()
        test = task_panel[task_panel["anchor_year"] >= test_start_year].copy()
        for horizon in HORIZONS:
            label = f"label_{horizon}d"
            y_test = test[label].astype(int).to_numpy()
            for feature_set, features in feature_sets.items():
                for model_name in MODELS:
                    score = fit_predict(train, test, features, label, model_name)
                    result = evaluate_predictions(y_test, score)
                    top10 = result["top_10pct"]
                    metrics.append(
                        {
                            "task": task,
                            "horizon_days": horizon,
                            "feature_set": feature_set,
                            "model": model_name,
                            "feature_count": int(len(features)),
                            "train_rows": int(len(train)),
                            "test_rows": int(len(test)),
                            "train_positives": int(train[label].sum()),
                            "test_positives": int(test[label].sum()),
                            "roc_auc": result["roc_auc"],
                            "average_precision": result["average_precision"],
                            "brier_score": result["brier_score"],
                            "top_10pct_precision": top10["precision"],
                            "top_10pct_recall": top10["recall"],
                            "top_10pct_lift": top10["lift"],
                        }
                    )
                    pred_cols = ["task", "firm_id", "firm_name", "anchor_year", "anchor_date", "outcome_date", "outcome_label", label]
                    pred = test[pred_cols].copy()
                    pred = pred.rename(columns={label: "label"})
                    pred["horizon_days"] = horizon
                    pred["feature_set"] = feature_set
                    pred["model"] = model_name
                    pred["score"] = score
                    prediction_frames.append(pred)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return pd.DataFrame(metrics), predictions


def write_markdown(summary: dict[str, Any], metrics: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Local SEC Metadata-Text Baseline",
        "",
        "This file evaluates text-like features from SEC submissions metadata fields (`items`, `primary_doc_description`, and `primary_document`). It is not a full filing-body text model.",
        "",
        "## Coverage",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Base panel source | `{summary['base_panel_source']}` |",
        f"| Filing metadata rows | {summary['filing_rows']} |",
        f"| Filing metadata firms | {summary['filing_firms']} |",
        f"| Panel rows | {summary['panel_rows']} |",
        f"| Metadata-text feature count | {summary['textmeta_feature_count']} |",
        f"| Sequence feature count | {summary['sequence_feature_count']} |",
        f"| Financial feature count | {summary['financial_feature_count']} |",
        "",
        "## Metrics",
        "",
        "| Task | Horizon | Feature set | Model | Test rows | Test positives | ROC AUC | Avg precision | Brier | Top 10% precision | Top 10% recall | Top 10% lift |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            "| {task} | {horizon} | {feature_set} | {model} | {test_rows} | {test_positives} | {roc} | {ap} | {brier} | {p10} | {r10} | {l10} |".format(
                task=row.task,
                horizon=row.horizon_days,
                feature_set=row.feature_set,
                model=row.model,
                test_rows=row.test_rows,
                test_positives=row.test_positives,
                roc=round(row.roc_auc, 4) if pd.notna(row.roc_auc) else "",
                ap=round(row.average_precision, 4) if pd.notna(row.average_precision) else "",
                brier=round(row.brier_score, 4) if pd.notna(row.brier_score) else "",
                p10=round(row.top_10pct_precision, 4) if pd.notna(row.top_10pct_precision) else "",
                r10=round(row.top_10pct_recall, 4) if pd.notna(row.top_10pct_recall) else "",
                l10=round(row.top_10pct_lift, 2) if pd.notna(row.top_10pct_lift) else "",
            )
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This is a metadata-text baseline, not a full SEC filing-body text model.",
            "- It uses local SEC submissions metadata and makes no SEC API calls.",
            "- Full filing text still requires SEC Archives download/caching and a separate text model.",
            "- Bankruptcy-only positives remain sparse; all-distress metrics are more stable.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-path", default=str(ROOT / "data" / "model_sec_desktop" / "sec_desktop_validation_panel.parquet"))
    parser.add_argument("--financial-panel-path", default=str(ROOT / "data" / "model_sec_desktop" / "sec_desktop_financial_panel.parquet"))
    parser.add_argument("--filings-path", default=str(ROOT / "data" / "raw" / "sec_desktop_population" / "sec_desktop_population_8k_filings.csv"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "model_sec_desktop"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    financial_panel_path = Path(args.financial_panel_path)
    panel, panel_source = load_base_panel(Path(args.panel_path), financial_panel_path if financial_panel_path.exists() else None)
    filings = load_filings(Path(args.filings_path))
    text_panel = build_text_metadata_features(panel, filings)
    text_panel.to_parquet(out_dir / "sec_desktop_text_metadata_panel.parquet", index=False)
    text_panel.to_csv(out_dir / "sec_desktop_text_metadata_panel.csv", index=False, encoding="utf-8-sig")

    metrics, predictions = evaluate_panel(text_panel, args.train_end_year, args.test_start_year)
    metrics.to_csv(results_dir / "sec_desktop_text_metadata_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(results_dir / "sec_desktop_text_metadata_baseline_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "created",
        "base_panel_source": panel_source,
        "filings_path": args.filings_path,
        "filing_rows": int(len(filings)),
        "filing_firms": int(filings["cik"].nunique()) if not filings.empty else 0,
        "panel_rows": int(len(text_panel)),
        "firms": int(text_panel["firm_id"].nunique()) if not text_panel.empty else 0,
        "textmeta_feature_count": int(len(textmeta_columns(text_panel))),
        "sequence_feature_count": int(len(sequence_columns(text_panel))),
        "financial_feature_count": int(len(financial_columns(text_panel))),
        "metrics": metrics.to_dict(orient="records"),
        "guardrail": "Metadata-text baseline only; full filing-body text baseline still requires SEC Archives text collection.",
    }
    (results_dir / "sec_desktop_text_metadata_baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(summary, metrics, results_dir / "sec_desktop_text_metadata_baseline_summary.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
