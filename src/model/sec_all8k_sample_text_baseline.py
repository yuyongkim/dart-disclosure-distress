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
DEFAULT_SAMPLE_STATUS = ROOT / "data" / "raw" / "sec_all8k_text" / "sec_all8k_text_sample_status.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "model_sec_desktop"
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_TEXT_ROOT = ROOT / "data" / "raw" / "sec_filing_text"
TRAIN_END_YEAR = 2021
TEST_START_YEAR = 2022

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
    "risk factor",
    "material adverse",
    "termination",
    "amendment",
    "merger",
    "acquisition",
    "disposition",
    "financing",
    "convertible",
)


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
    if len(y_true) == 0:
        return {"top_fraction": fraction, "top_n": 0, "precision": None, "recall": None, "lift": None}
    top_n = max(1, int(np.ceil(len(y_true) * fraction)))
    order = np.argsort(-y_score)
    top = order[:top_n]
    base_rate = float(np.mean(y_true))
    precision = float(np.mean(y_true[top]))
    positives = float(np.sum(y_true))
    recall = float(np.sum(y_true[top]) / positives) if positives else None
    lift = float(precision / base_rate) if base_rate else None
    return {
        "top_fraction": fraction,
        "top_n": int(top_n),
        "precision": finite_float(precision),
        "recall": finite_float(recall),
        "lift": finite_float(lift),
    }


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    top10 = top_fraction_metrics(y_true, y_score, 0.10)
    return {
        "roc_auc": safe_metric(roc_auc_score, y_true, y_score),
        "average_precision": safe_metric(average_precision_score, y_true, y_score),
        "brier_score": finite_float(brier_score_loss(y_true, y_score)) if len(y_true) else None,
        "top_10pct_precision": top10["precision"],
        "top_10pct_recall": top10["recall"],
        "top_10pct_lift": top10["lift"],
    }


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = "" if value is None or pd.isna(value) else str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def normalize_cik(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else ""


def accession_nodash(value: object) -> str:
    return str(value or "").replace("-", "").strip()


def canonical_text_path(text_root: Path, cik: object, accession: object) -> str:
    cik_norm = normalize_cik(cik)
    accession_key = accession_nodash(accession)
    if not cik_norm or not accession_key:
        return ""
    return str(text_root / cik_norm / f"{accession_key}.txt")


def add_canonical_text_paths(sample: pd.DataFrame, text_root: Path) -> pd.DataFrame:
    sample = sample.copy()
    original_path = (
        sample["local_text_path"].fillna("").astype(str)
        if "local_text_path" in sample
        else pd.Series([""] * len(sample), index=sample.index)
    )
    sample["sample_local_path_redacted"] = original_path.str.startswith("public_")
    sample["model_text_path"] = sample.apply(
        lambda row: canonical_text_path(text_root, row.get("cik"), row.get("accession_number")),
        axis=1,
    )
    missing_path = sample["model_text_path"].eq("")
    if missing_path.any() and "local_text_path" in sample:
        sample.loc[missing_path, "model_text_path"] = sample.loc[missing_path, "local_text_path"].fillna("").astype(str)
    return sample


def extract_features(sample: pd.DataFrame, text_root: Path, max_files: int | None = None) -> pd.DataFrame:
    sample = sample.copy()
    sample = add_canonical_text_paths(sample, text_root)
    sample["filing_year"] = pd.to_numeric(sample["filing_year"], errors="coerce")
    sample["cached_now"] = sample["model_text_path"].fillna("").map(lambda value: Path(str(value)).exists())
    sample = sample[sample["cached_now"]].sort_values(["sample_role", "filing_year", "item_bucket", "cik", "accession_number"])
    if max_files and max_files > 0:
        sample = sample.head(max_files).copy()

    rows: list[dict[str, Any]] = []
    for row in sample.itertuples(index=False):
        raw = read_text(Path(str(row.model_text_path)))
        text = raw.lower()
        has_target = as_bool(getattr(row, "has_target_distress_item", False))
        rec: dict[str, Any] = {
            "sample_id": str(row.sample_id),
            "sample_role": str(row.sample_role),
            "sample_stratum": str(row.sample_stratum),
            "filing_year": int(row.filing_year) if pd.notna(row.filing_year) else None,
            "item_bucket": str(row.item_bucket),
            "cik": str(row.cik),
            "entity_name": str(row.entity_name),
            "accession_number": str(row.accession_number),
            "filing_date": str(row.filing_date),
            "items": str(row.items),
            "has_target_distress_item": int(has_target),
            "body_char_count": float(len(raw)),
            "body_word_count": float(len(text.split())),
            "body_line_count": float(raw.count("\n") + 1 if raw else 0),
            "body_has_html": float(1 if "<html" in text or "<document>" in text else 0),
        }
        for keyword in KEYWORDS:
            rec[f"body_kw_{safe_name(keyword)}"] = float(text.count(keyword))
        rows.append(rec)
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col.startswith("body_") and col not in {"body_has_html"}
    ] + (["body_has_html"] if "body_has_html" in df.columns else [])


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
                        n_estimators=200,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise ValueError(model_name)


def evaluate_models(features: pd.DataFrame, train_end_year: int, test_start_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = feature_columns(features)
    train = features[features["filing_year"] <= train_end_year].copy()
    test = features[features["filing_year"] >= test_start_year].copy()
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for model_name in ("logistic", "random_forest"):
        y_train = train["has_target_distress_item"].astype(int).to_numpy()
        y_test = test["has_target_distress_item"].astype(int).to_numpy()
        if len(cols) == 0 or len(train) == 0 or len(test) == 0 or len(np.unique(y_train)) < 2:
            score = np.repeat(float(np.mean(y_train)) if len(y_train) else 0.0, len(test))
        else:
            model = make_model(model_name)
            model.fit(train[cols], y_train)
            score = model.predict_proba(test[cols])[:, 1]
        result = evaluate_predictions(y_test, score)
        metrics.append(
            {
                "task": "target_item_vs_non_target_sample",
                "model": model_name,
                "feature_set": "sample_filing_body_keyword_counts",
                "feature_count": int(len(cols)),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_positives": int(y_train.sum()) if len(y_train) else 0,
                "test_positives": int(y_test.sum()) if len(y_test) else 0,
                **result,
            }
        )
        pred = test[
            [
                "sample_id",
                "sample_role",
                "filing_year",
                "item_bucket",
                "cik",
                "entity_name",
                "accession_number",
                "filing_date",
                "items",
                "has_target_distress_item",
            ]
        ].copy()
        pred["model"] = model_name
        pred["score"] = score
        predictions.append(pred)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def write_markdown(summary: dict[str, Any], metrics: pd.DataFrame, path: Path) -> None:
    lines = [
        "# SEC All-8-K Sample Text Baseline",
        "",
        "This artifact evaluates the bounded all-8-K filing-body text sample. It is a sample-level text sensitivity check, not standalone U.S. distress-prediction validation.",
        "",
        "## Coverage",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Status | `{summary['status']}` |",
        f"| Sample rows | {summary['sample_rows']} |",
        f"| Path resolution mode | `{summary['path_resolution_mode']}` |",
        f"| Redacted sample path rows | {summary['sample_local_path_redacted_rows']} |",
        f"| Canonical text path rows | {summary['canonical_text_path_rows']} |",
        f"| Cached text rows used | {summary['cached_text_rows_used']} |",
        f"| Missing sample text rows | {summary['missing_sample_text_rows']} |",
        f"| Non-target rows | {summary['non_target_rows']} |",
        f"| Target-reference rows | {summary['target_reference_rows']} |",
        f"| Cached text bytes | {summary['cached_text_bytes']} |",
        f"| Feature count | {summary['feature_count']} |",
        f"| Train years | <= {summary['train_end_year']} |",
        f"| Test years | >= {summary['test_start_year']} |",
        "",
        "## Metrics",
        "",
    ]
    if metrics.empty:
        lines.append("No model metrics were estimated.")
    else:
        lines.extend(
            [
                "| Task | Model | Train rows | Test rows | Train positives | Test positives | ROC AUC | Avg precision | Brier | Top 10% precision | Top 10% recall | Top 10% lift |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in metrics.itertuples(index=False):
            lines.append(
                "| {task} | {model} | {train_rows} | {test_rows} | {train_pos} | {test_pos} | {roc} | {ap} | {brier} | {p10} | {r10} | {l10} |".format(
                    task=row.task,
                    model=row.model,
                    train_rows=row.train_rows,
                    test_rows=row.test_rows,
                    train_pos=row.train_positives,
                    test_pos=row.test_positives,
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
            "- This uses the bounded sample only: 960 non-target all-8-K filings plus 119 cached target-reference filings.",
            "- The prediction task is target-item versus non-target sample classification, not future corporate distress prediction.",
            "- Use this artifact to show that non-target filing-body sample text has been downloaded and modeled.",
            "- Do not call this full all-8-K text validation or standalone U.S. distress validation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-status-path", default=str(DEFAULT_SAMPLE_STATUS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--text-root", default=str(DEFAULT_TEXT_ROOT))
    parser.add_argument("--train-end-year", type=int, default=TRAIN_END_YEAR)
    parser.add_argument("--test-start-year", type=int, default=TEST_START_YEAR)
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    text_root = Path(args.text_root)
    sample = pd.read_csv(args.sample_status_path, low_memory=False)
    sample = add_canonical_text_paths(sample, text_root)
    features = extract_features(sample, text_root, args.max_files if args.max_files > 0 else None)
    metrics, predictions = evaluate_models(features, args.train_end_year, args.test_start_year)

    features_csv = out_dir / "sec_all8k_sample_text_features.csv"
    features_parquet = out_dir / "sec_all8k_sample_text_features.parquet"
    metrics_csv = results_dir / "sec_all8k_sample_text_baseline_metrics.csv"
    predictions_csv = results_dir / "sec_all8k_sample_text_baseline_predictions.csv"
    features.to_csv(features_csv, index=False, encoding="utf-8-sig")
    try:
        features.to_parquet(features_parquet, index=False)
    except Exception:
        features_parquet = Path("")
    metrics.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_csv, index=False, encoding="utf-8-sig")

    cached_text_rows = int(len(features))
    missing_rows = int(len(sample) - cached_text_rows)
    non_target_rows = int((features["sample_role"] == "non_target_text_scope_sample").sum()) if not features.empty else 0
    target_rows = int((features["sample_role"] == "target_cached_reference_sample").sum()) if not features.empty else 0
    bytes_sum = int(pd.to_numeric(sample.get("local_text_bytes_now", 0), errors="coerce").fillna(0).sum())
    status = "sample_text_baseline_created" if missing_rows == 0 and cached_text_rows else "sample_text_baseline_partial"
    summary = {
        "status": status,
        "sample_status_path": args.sample_status_path,
        "sample_rows": int(len(sample)),
        "path_resolution_mode": "canonical_cik_accession_text_root",
        "sample_local_path_redacted_rows": int(sample.get("sample_local_path_redacted", pd.Series(dtype=bool)).sum()),
        "canonical_text_path_rows": int(sample["model_text_path"].fillna("").astype(str).ne("").sum()),
        "cached_text_rows_used": cached_text_rows,
        "missing_sample_text_rows": missing_rows,
        "non_target_rows": non_target_rows,
        "target_reference_rows": target_rows,
        "cached_text_bytes": bytes_sum,
        "feature_count": int(len(feature_columns(features))) if not features.empty else 0,
        "train_end_year": int(args.train_end_year),
        "test_start_year": int(args.test_start_year),
        "features_csv": str(features_csv),
        "features_parquet": str(features_parquet) if str(features_parquet) else "",
        "metrics_csv": str(metrics_csv),
        "predictions_csv": str(predictions_csv),
        "metrics": metrics.to_dict(orient="records"),
        "guardrail": "Sample-level target-item vs non-target filing-body text sensitivity; not full all-8-K or standalone U.S. distress validation.",
    }
    (results_dir / "sec_all8k_sample_text_baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(summary, metrics, results_dir / "sec_all8k_sample_text_baseline_summary.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
