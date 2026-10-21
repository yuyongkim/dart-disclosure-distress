from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def stable_bucket(value: object, folds: int) -> int:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [col for col in panel.columns if col.startswith("seq_")]


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


def top_fraction_metrics(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.1) -> dict:
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


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "roc_auc": safe_metric(roc_auc_score, y_true, y_score),
        "average_precision": safe_metric(average_precision_score, y_true, y_score),
        "brier_score": finite_float(brier_score_loss(y_true, y_score)),
        "top_10pct": top_fraction_metrics(y_true, y_score, 0.10),
        "top_5pct": top_fraction_metrics(y_true, y_score, 0.05),
    }


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
    if len(np.unique(y_train)) < 2:
        return np.repeat(float(y_train.mean()) if len(y_train) else 0.0, len(test))
    model = make_model(model_name)
    model.fit(train[features], y_train)
    return model.predict_proba(test[features])[:, 1]


def run_robustness(panel: pd.DataFrame, train_end_year: int, test_start_year: int, folds: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = panel.copy()
    panel["bucket"] = panel["firm_id"].map(lambda value: stable_bucket(value, folds))
    features = feature_columns(panel)
    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []

    for task, task_panel in panel.groupby("task"):
        for horizon in (365, 730):
            label = f"label_{horizon}d"
            for model_name in ("logistic", "random_forest"):
                for fold in range(folds):
                    train = task_panel[(task_panel["anchor_year"] <= train_end_year) & (task_panel["bucket"] != fold)].copy()
                    test = task_panel[(task_panel["anchor_year"] >= test_start_year) & (task_panel["bucket"] == fold)].copy()
                    if test.empty:
                        continue
                    score = fit_predict(train, test, features, label, model_name)
                    y_test = test[label].astype(int).to_numpy()
                    metric = evaluate_predictions(y_test, score)
                    top10 = metric["top_10pct"]
                    fold_rows.append(
                        {
                            "task": task,
                            "horizon_days": horizon,
                            "model": model_name,
                            "fold": fold,
                            "train_rows": int(len(train)),
                            "test_rows": int(len(test)),
                            "train_positives": int(train[label].sum()),
                            "test_positives": int(test[label].sum()),
                            "roc_auc": metric["roc_auc"],
                            "average_precision": metric["average_precision"],
                            "brier_score": metric["brier_score"],
                            "top_10pct_precision": top10["precision"],
                            "top_10pct_recall": top10["recall"],
                            "top_10pct_lift": top10["lift"],
                        }
                    )
                    pred = test[["task", "firm_id", "firm_name", "anchor_year", "anchor_date", "outcome_date", "outcome_label", label, "bucket"]].copy()
                    pred = pred.rename(columns={label: "label"})
                    pred["horizon_days"] = horizon
                    pred["model"] = model_name
                    pred["fold"] = fold
                    pred["score"] = score
                    prediction_frames.append(pred)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    fold_metrics = pd.DataFrame(fold_rows)
    aggregate_rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(["task", "horizon_days", "model"]):
        task, horizon, model_name = keys
        y = group["label"].astype(int).to_numpy()
        score = group["score"].astype(float).to_numpy()
        metric = evaluate_predictions(y, score)
        top10 = metric["top_10pct"]
        aggregate_rows.append(
            {
                "task": task,
                "horizon_days": int(horizon),
                "model": model_name,
                "test_rows": int(len(group)),
                "test_firms": int(group["firm_id"].nunique()),
                "test_positives": int(group["label"].sum()),
                "roc_auc": metric["roc_auc"],
                "average_precision": metric["average_precision"],
                "brier_score": metric["brier_score"],
                "top_10pct_precision": top10["precision"],
                "top_10pct_recall": top10["recall"],
                "top_10pct_lift": top10["lift"],
            }
        )
    aggregate_metrics = pd.DataFrame(aggregate_rows).sort_values(["task", "horizon_days", "model"])
    return aggregate_metrics, fold_metrics, predictions


def write_markdown(summary: dict, metrics: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Local SEC Firm-Disjoint Robustness",
        "",
        "This file evaluates the preliminary SEC v2 sequence models under a stricter split: train anchors are <= 2021, test anchors are >= 2022, and test firms are excluded from each fold's training set using stable CIK hash buckets.",
        "",
        "## Split",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Folds | {summary['folds']} |",
        f"| Train anchors | <= {summary['train_end_year']} |",
        f"| Test anchors | >= {summary['test_start_year']} |",
        f"| Feature count | {summary['feature_count']} |",
        "",
        "## Aggregate Metrics",
        "",
        "| Task | Horizon | Model | Test rows | Test firms | Test positives | ROC AUC | Avg precision | Brier | Top 10% precision | Top 10% recall | Top 10% lift |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            "| {task} | {horizon} | {model} | {rows} | {firms} | {pos} | {auc} | {ap} | {brier} | {p10} | {r10} | {l10} |".format(
                task=row.task,
                horizon=row.horizon_days,
                model=row.model,
                rows=row.test_rows,
                firms=row.test_firms,
                pos=row.test_positives,
                auc=round(row.roc_auc, 4) if pd.notna(row.roc_auc) else "",
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
            "- This is still sequence-only and uses cached SEC submissions metadata with available archive JSON included.",
            "- Bankruptcy-only positives remain very sparse; report those metrics as suggestive only.",
            "- The split is firm-disjoint within each fold and temporally ordered by anchor year, but it is not a fully refreshed SEC sampling frame.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-path", default=str(Path("Paper 4") / "data" / "model_sec_desktop" / "sec_desktop_validation_panel.parquet"))
    parser.add_argument("--results-dir", default=str(Path("Paper 4") / "results"))
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel_path)
    aggregate_metrics, fold_metrics, predictions = run_robustness(panel, args.train_end_year, args.test_start_year, args.folds)
    aggregate_metrics.to_csv(results_dir / "sec_desktop_validation_firm_disjoint_metrics.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(results_dir / "sec_desktop_validation_firm_disjoint_fold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(results_dir / "sec_desktop_validation_firm_disjoint_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "folds": args.folds,
        "train_end_year": args.train_end_year,
        "test_start_year": args.test_start_year,
        "feature_count": len(feature_columns(panel)),
        "metrics": aggregate_metrics.to_dict(orient="records"),
        "guardrail": "Firm-disjoint temporal robustness for SEC v2 feasibility evidence.",
    }
    (results_dir / "sec_desktop_validation_firm_disjoint_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(summary, aggregate_metrics, results_dir / "sec_desktop_validation_firm_disjoint_summary.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
