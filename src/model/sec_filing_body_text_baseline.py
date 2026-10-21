from __future__ import annotations

import argparse
import json
import re
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
WINDOWS = (90, 180, 365, 730)
HORIZONS = (365, 730)
MODELS = ("logistic", "random_forest")

KEYWORDS = (
    "bankruptcy",
    "chapter 11",
    "receivership",
    "default",
    "event of default",
    "acceleration",
    "delisting",
    "nasdaq",
    "nyse",
    "going concern",
    "substantial doubt",
    "liquidity",
    "debt",
    "credit agreement",
    "covenant",
    "waiver",
    "forbearance",
    "restructuring",
    "impairment",
    "material weakness",
    "auditor",
    "resignation",
)


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


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


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


def load_manifest(manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path, low_memory=False)
    manifest["cik"] = manifest["cik"].map(normalize_cik)
    manifest["filing_date_ts"] = pd.to_datetime(manifest["filing_date"], errors="coerce")
    manifest["local_exists_now"] = manifest["local_text_path"].fillna("").astype(str).map(lambda value: Path(value).exists())
    manifest["local_bytes_now"] = manifest["local_text_path"].fillna("").astype(str).map(
        lambda value: int(Path(value).stat().st_size) if Path(value).exists() else 0
    )
    cached = manifest[
        manifest["cik"].ne("")
        & manifest["filing_date_ts"].notna()
        & manifest["local_exists_now"]
        & manifest["local_text_path"].notna()
    ].copy()
    return manifest, cached.sort_values(["cik", "filing_date_ts", "accession_number"]).reset_index(drop=True)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def extract_text_features(cached: pd.DataFrame, max_files: int | None = None) -> pd.DataFrame:
    if max_files and max_files > 0:
        cached = cached.head(max_files).copy()
    rows: list[dict[str, Any]] = []
    for row in cached.itertuples(index=False):
        path = Path(str(row.local_text_path))
        raw = read_text(path)
        text = raw.lower()
        item_set = split_items(row.items)
        rec: dict[str, Any] = {
            "cik": str(row.cik),
            "accession_number": str(row.accession_number),
            "filing_date": str(row.filing_date),
            "filing_date_ts": row.filing_date_ts,
            "items": str(row.items),
            "body_char_count": float(len(raw)),
            "body_word_count": float(len(text.split())),
            "body_line_count": float(raw.count("\n") + 1 if raw else 0),
            "body_has_html": float(1 if "<html" in text or "<document>" in text else 0),
            "body_item_1_03": float(1 if "1.03" in item_set else 0),
            "body_item_2_04": float(1 if "2.04" in item_set else 0),
            "body_item_3_01": float(1 if "3.01" in item_set else 0),
        }
        for keyword in KEYWORDS:
            rec[f"body_kw_{safe_name(keyword)}"] = float(text.count(keyword))
        rows.append(rec)
    return pd.DataFrame(rows)


def body_base_columns() -> list[str]:
    return [
        "body_char_count",
        "body_word_count",
        "body_line_count",
        "body_has_html",
        "body_item_1_03",
        "body_item_2_04",
        "body_item_3_01",
    ] + [f"body_kw_{safe_name(keyword)}" for keyword in KEYWORDS]


def build_lookup(filings: pd.DataFrame, base_cols: list[str]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if filings.empty:
        return lookup
    for cik, group in filings.groupby("cik", sort=False):
        values = group[base_cols].to_numpy(dtype=float)
        prefix = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
        lookup[str(cik)] = {
            "dates": group["filing_date_ts"].to_numpy(dtype="datetime64[ns]"),
            "prefix": prefix,
        }
    return lookup


def build_body_text_features(panel: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    base_cols = body_base_columns()
    if filings.empty:
        filings = pd.DataFrame(columns=["cik", "filing_date_ts", "accession_number", *base_cols])
    for col in base_cols:
        if col not in filings:
            filings[col] = 0.0
    filings = filings.sort_values(["cik", "filing_date_ts", "accession_number"], na_position="last")
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
        if firm is None or pd.isna(row.anchor_date_ts):
            for window in WINDOWS:
                rec[f"textbody_w{window}_filing_count"] = 0
                for col in base_cols:
                    rec[f"textbody_w{window}_{col}_sum"] = 0.0
            rows.append(rec)
            continue
        dates = firm["dates"]
        prefix = firm["prefix"]
        end_idx = int(np.searchsorted(dates, anchor, side="right"))
        for window in WINDOWS:
            start = anchor - np.timedelta64(window, "D")
            start_idx = int(np.searchsorted(dates, start, side="left"))
            counts = prefix[end_idx] - prefix[start_idx]
            rec[f"textbody_w{window}_filing_count"] = int(end_idx - start_idx)
            for idx, col in enumerate(base_cols):
                rec[f"textbody_w{window}_{col}_sum"] = float(counts[idx])
        rows.append(rec)
    features = pd.DataFrame(rows)
    merged = panel.drop(columns=["anchor_date_ts"], errors="ignore").merge(
        features,
        on=["task", "firm_id", "anchor_year", "anchor_date"],
        how="left",
    )
    for col in textbody_columns(merged):
        merged[col] = merged[col].fillna(0.0)
    return merged


def textbody_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("textbody_")]


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
    if not features or len(np.unique(y_train)) < 2:
        return np.repeat(float(y_train.mean()) if len(y_train) else 0.0, len(test))
    model = make_model(model_name)
    model.fit(train[features], y_train)
    return model.predict_proba(test[features])[:, 1]


def evaluate_panel(panel: pd.DataFrame, train_end_year: int, test_start_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    body_features = textbody_columns(panel)
    seq_features = sequence_columns(panel)
    fin_features = financial_columns(panel)
    feature_sets = {
        "filing_body_text_only": body_features,
        "sequence_plus_filing_body_text": seq_features + body_features,
    }
    if fin_features:
        feature_sets["sequence_financial_filing_body_text"] = seq_features + fin_features + body_features
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
        "# SEC Filing-Body Text Baseline",
        "",
        "This file evaluates transparent keyword/count features from cached SEC filing-body text files. It is separate from the submissions metadata-text baseline.",
        "",
        "## Coverage",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Status | `{summary['status']}` |",
        f"| Base panel source | `{summary['base_panel_source']}` |",
        f"| Manifest rows | {summary['manifest_rows']} |",
        f"| Cached filing-body files | {summary['cached_text_files']} |",
        f"| Cached filing-body bytes | {summary['cached_text_bytes']} |",
        f"| Cached filing rows used | {summary['cached_filing_rows_used']} |",
        f"| Cache coverage rate | {summary['cache_coverage_rate']} |",
        f"| Panel rows | {summary['panel_rows']} |",
        f"| Filing-body feature count | {summary['textbody_feature_count']} |",
        f"| Sequence feature count | {summary['sequence_feature_count']} |",
        f"| Financial feature count | {summary['financial_feature_count']} |",
        "",
        "## Metrics",
        "",
    ]
    if metrics.empty:
        lines.extend(
            [
                "No metrics were estimated because no cached filing-body text files were available.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "| Task | Horizon | Feature set | Model | Test rows | Test positives | ROC AUC | Avg precision | Brier | Top 10% precision | Top 10% recall | Top 10% lift |",
                "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
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
        lines.append("")
    lines.extend(
        [
            "## Guardrails",
            "",
            "- The current cache covers the target-item SEC Archives manifest, not every SEC 8-K filing body in the population frame.",
            "- Features are transparent keyword/count features and should be treated as v2 support evidence.",
            "- Do not describe this as finalized U.S. predictive validation until full all-8-K filing-body coverage and final U.S. validation scope are completed.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-path", default=str(ROOT / "data" / "model_sec_desktop" / "sec_desktop_validation_panel.parquet"))
    parser.add_argument("--financial-panel-path", default=str(ROOT / "data" / "model_sec_desktop" / "sec_desktop_financial_panel.parquet"))
    parser.add_argument("--manifest-path", default=str(ROOT / "data" / "raw" / "sec_filing_text" / "sec_filing_text_manifest.csv"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "model_sec_desktop"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    financial_panel_path = Path(args.financial_panel_path)
    panel, panel_source = load_base_panel(Path(args.panel_path), financial_panel_path if financial_panel_path.exists() else None)
    manifest, cached = load_manifest(Path(args.manifest_path))
    body_filings = extract_text_features(cached, args.max_files if args.max_files > 0 else None)
    text_panel = build_body_text_features(panel, body_filings)
    text_panel.to_parquet(out_dir / "sec_filing_body_text_panel.parquet", index=False)
    text_panel.to_csv(out_dir / "sec_filing_body_text_panel.csv", index=False, encoding="utf-8-sig")

    if body_filings.empty:
        metrics = pd.DataFrame()
        predictions = pd.DataFrame()
        status = "insufficient_cached_text"
    else:
        metrics, predictions = evaluate_panel(text_panel, args.train_end_year, args.test_start_year)
        status = "target_item_cache_baseline_created" if len(body_filings) == len(manifest) else "sample_cache_baseline_created"
    metrics.to_csv(results_dir / "sec_filing_body_text_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(results_dir / "sec_filing_body_text_baseline_predictions.csv", index=False, encoding="utf-8-sig")

    cached_text_files = int(manifest["local_exists_now"].sum()) if not manifest.empty else 0
    cached_text_bytes = int(manifest["local_bytes_now"].sum()) if not manifest.empty else 0
    summary = {
        "status": status,
        "base_panel_source": panel_source,
        "manifest_path": args.manifest_path,
        "manifest_rows": int(len(manifest)),
        "cached_text_files": cached_text_files,
        "cached_text_bytes": cached_text_bytes,
        "cached_filing_rows_used": int(len(body_filings)),
        "cache_coverage_rate": finite_float(cached_text_files / len(manifest)) if len(manifest) else None,
        "panel_rows": int(len(text_panel)),
        "firms": int(text_panel["firm_id"].nunique()) if not text_panel.empty else 0,
        "textbody_feature_count": int(len(textbody_columns(text_panel))),
        "sequence_feature_count": int(len(sequence_columns(text_panel))),
        "financial_feature_count": int(len(financial_columns(text_panel))),
        "metrics": metrics.to_dict(orient="records"),
        "guardrail": "Cached target-item filing-body model; not a population-complete full SEC 8-K filing-body archive.",
    }
    (results_dir / "sec_filing_body_text_baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(summary, metrics, results_dir / "sec_filing_body_text_baseline_summary.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
