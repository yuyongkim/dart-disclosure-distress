from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FINANCIAL_FEATURE_COLUMNS = [
    "fin_log_revenue",
    "fin_log_operating_income_signed",
    "fin_log_net_income_signed",
    "fin_log_total_assets",
    "fin_leverage",
    "fin_equity_ratio",
    "fin_roa",
    "fin_operating_roa",
    "fin_net_margin",
    "fin_operating_margin",
    "fin_loss_flag",
    "fin_negative_equity_flag",
    "fin_missing_record_flag",
]


def sequence_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("seq_")]


def market_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c.startswith("mkt_")
        and c != "mkt_last_trade_date"
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def finite_float(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    return float(value) if isinstance(value, (np.floating, float)) else int(value)


def safe_metric(fn, y_true: np.ndarray, y_score_or_pred: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return finite_float(fn(y_true, y_score_or_pred))
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
    recall = float(np.sum(y_true[top]) / np.sum(y_true)) if np.sum(y_true) else None
    lift = float(precision / base_rate) if base_rate and precision is not None else None
    return {
        "top_fraction": fraction,
        "top_n": int(top_n),
        "precision": finite_float(precision),
        "recall": finite_float(recall),
        "lift": finite_float(lift),
    }


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_pred = (y_score >= 0.5).astype(int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return {
        "roc_auc": safe_metric(roc_auc_score, y_true, y_score),
        "average_precision": safe_metric(average_precision_score, y_true, y_score),
        "brier_score": finite_float(brier_score_loss(y_true, y_score)),
        "threshold_0_5": {
            "precision": finite_float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": finite_float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": finite_float(f1_score(y_true, y_pred, zero_division=0)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },
        "top_10pct": top_fraction_metrics(y_true, y_score, 0.10),
        "top_5pct": top_fraction_metrics(y_true, y_score, 0.05),
    }


def bootstrap_confidence_intervals(y_true: np.ndarray, y_score: np.ndarray, n_boot: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    metrics: dict[str, list[float]] = {
        "roc_auc": [],
        "average_precision": [],
        "brier_score": [],
        "top_10pct_precision": [],
        "top_10pct_recall": [],
        "top_10pct_lift": [],
        "top_5pct_precision": [],
        "top_5pct_recall": [],
        "top_5pct_lift": [],
    }
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        score_b = y_score[idx]
        if len(np.unique(y_b)) >= 2:
            metrics["roc_auc"].append(float(roc_auc_score(y_b, score_b)))
            metrics["average_precision"].append(float(average_precision_score(y_b, score_b)))
        metrics["brier_score"].append(float(brier_score_loss(y_b, score_b)))
        for fraction, label in ((0.10, "top_10pct"), (0.05, "top_5pct")):
            top = top_fraction_metrics(y_b, score_b, fraction)
            for key in ("precision", "recall", "lift"):
                value = top.get(key)
                if value is not None:
                    metrics[f"{label}_{key}"].append(float(value))

    rows: list[dict] = []
    for metric, values in metrics.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            rows.append({"metric": metric, "mean": None, "ci_low": None, "ci_high": None, "n_boot_valid": 0})
            continue
        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(arr)),
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "n_boot_valid": int(arr.size),
            }
        )
    return rows


def calibration_rows(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> list[dict]:
    n = len(y_true)
    if n == 0:
        return []
    order = np.argsort(y_score)
    bins = np.array_split(order, n_bins)
    rows: list[dict] = []
    for i, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "risk_bin": i,
                "count": int(len(idx)),
                "mean_predicted_risk": float(np.mean(y_score[idx])),
                "observed_rate": float(np.mean(y_true[idx])),
                "positives": int(np.sum(y_true[idx])),
                "min_score": float(np.min(y_score[idx])),
                "max_score": float(np.max(y_score[idx])),
            }
        )
    return rows


def logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def forest_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def calibrated_forest_model() -> Pipeline:
    base = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                CalibratedClassifierCV(
                    estimator=base,
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )


def fit_predict(model: Pipeline, train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> tuple[np.ndarray, Pipeline]:
    x_train = train[features].replace([np.inf, -np.inf], np.nan)
    y_train = train[target].astype(int).to_numpy()
    x_test = test[features].replace([np.inf, -np.inf], np.nan)
    recency_cols = [c for c in features if c.startswith("seq_days_since_last")]
    if recency_cols:
        x_train.loc[:, recency_cols] = x_train.loc[:, recency_cols].fillna(9999.0)
        x_test.loc[:, recency_cols] = x_test.loc[:, recency_cols].fillna(9999.0)
    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    return y_score, model


def model_specs(df: pd.DataFrame) -> dict[str, list[str]]:
    seq = sequence_columns(df)
    fin = [c for c in FINANCIAL_FEATURE_COLUMNS if c in df.columns]
    mkt = market_columns(df)
    specs = {
        "sequence_logit": seq,
        "financial_logit": fin,
        "combined_logit": seq + fin,
        "combined_random_forest": seq + fin,
        "combined_calibrated_random_forest": seq + fin,
    }
    if mkt:
        specs.update(
            {
                "market_logit": mkt,
                "sequence_market_logit": seq + mkt,
                "financial_market_logit": fin + mkt,
                "combined_market_logit": seq + fin + mkt,
                "market_random_forest": mkt,
                "financial_market_random_forest": fin + mkt,
                "combined_market_random_forest": seq + fin + mkt,
                "combined_market_calibrated_random_forest": seq + fin + mkt,
            }
        )
    return specs


def prediction_rows(test: pd.DataFrame, horizon: int, model_name: str, y_score: np.ndarray) -> list[dict]:
    target = f"target_{horizon}d"
    cols = [
        "firm_id",
        "firm_name",
        "security_id",
        "anchor_year",
        "anchor_date",
        "future_outcome_date",
        "future_outcome_label",
        "days_to_outcome",
    ]
    rows: list[dict] = []
    for score, row in zip(y_score, test[cols + [target]].itertuples(index=False)):
        rows.append(
            {
                "horizon_days": horizon,
                "model": model_name,
                "firm_id": row.firm_id,
                "firm_name": row.firm_name,
                "security_id": row.security_id,
                "anchor_year": int(row.anchor_year),
                "anchor_date": row.anchor_date,
                "future_outcome_date": row.future_outcome_date,
                "future_outcome_label": row.future_outcome_label,
                "days_to_outcome": row.days_to_outcome,
                "target": bool(getattr(row, target)),
                "predicted_risk": float(score),
            }
        )
    return rows


def run_horizon(
    df: pd.DataFrame,
    horizon: int,
    observation_end: pd.Timestamp,
    n_boot: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    target = f"target_{horizon}d"
    usable = df["anchor_date"] + pd.to_timedelta(horizon, unit="D") <= observation_end
    data = df[usable].copy()
    train = data[data["anchor_year"].le(2021)].copy()
    test = data[data["anchor_year"].ge(2022)].copy()
    results: list[dict] = []
    coefficients: list[dict] = []
    importances: list[dict] = []
    predictions: list[dict] = []
    bootstrap_rows: list[dict] = []
    calibration: list[dict] = []
    y_test = test[target].astype(int).to_numpy()

    for name, features in model_specs(data).items():
        if not features:
            continue
        if train[target].nunique() < 2 or test[target].nunique() < 2:
            results.append(
                {
                    "horizon_days": horizon,
                    "model": name,
                    "status": "skipped_one_class_split",
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "train_positives": int(train[target].sum()),
                    "test_positives": int(test[target].sum()),
                }
            )
            continue

        if "calibrated_random_forest" in name:
            model = calibrated_forest_model()
        elif name.endswith("random_forest"):
            model = forest_model()
        else:
            model = logistic_model()
        y_score, fitted = fit_predict(model, train, test, features, target)
        result = {
            "horizon_days": horizon,
            "model": name,
            "status": "ok",
            "n_features": int(len(features)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_positives": int(train[target].sum()),
            "test_positives": int(test[target].sum()),
            "train_positive_rate": float(train[target].mean()),
            "test_positive_rate": float(test[target].mean()),
            **evaluate_predictions(y_test, y_score),
        }
        results.append(result)
        predictions.extend(prediction_rows(test, horizon, name, y_score))
        for row in bootstrap_confidence_intervals(y_test, y_score, n_boot=n_boot, seed=42 + horizon + len(results)):
            row["horizon_days"] = horizon
            row["model"] = name
            bootstrap_rows.append(row)
        for row in calibration_rows(y_test, y_score):
            row["horizon_days"] = horizon
            row["model"] = name
            calibration.append(row)

        if name.endswith("logit"):
            estimator = fitted.named_steps["model"]
            for feature, coef in zip(features, estimator.coef_[0]):
                coefficients.append(
                    {
                        "horizon_days": horizon,
                        "model": name,
                        "feature": feature,
                        "coefficient": float(coef),
                    }
                )
        if name == "combined_random_forest":
            estimator = fitted.named_steps["model"]
            for feature, importance in zip(features, estimator.feature_importances_):
                importances.append(
                    {
                        "horizon_days": horizon,
                        "model": name,
                        "feature": feature,
                        "importance": float(importance),
                    }
                )
    return results, coefficients, importances, predictions, bootstrap_rows, calibration


def best_by_average_precision(results: list[dict]) -> dict[int, dict]:
    best: dict[int, dict] = {}
    for item in results:
        if item.get("status") != "ok":
            continue
        horizon = int(item["horizon_days"])
        score = item.get("average_precision")
        if score is None:
            continue
        if horizon not in best or score > best[horizon].get("average_precision", -1):
            best[horizon] = item
    return best


def write_markdown(
    results: list[dict],
    coeffs: pd.DataFrame,
    importances: pd.DataFrame,
    bootstrap: pd.DataFrame,
    calibration: pd.DataFrame,
    path: Path,
) -> None:
    lines = ["# DART Early-Warning Baseline Results", ""]
    lines.append("Split: train anchor years <= 2021; test anchor years >= 2022. Rows whose full outcome horizon is not observable are excluded per horizon.")
    lines.append("")
    lines.append("| Horizon | Model | Train rows | Test rows | Test positives | ROC AUC | Avg precision | Brier | Top 10% precision | Top 10% recall | Top 10% lift |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in results:
        if item.get("status") != "ok":
            lines.append(
                f"| {item['horizon_days']} | {item['model']} | {item['n_train']} | {item['n_test']} | {item['test_positives']} | skipped | skipped | skipped | skipped | skipped | skipped |"
            )
            continue
        top = item["top_10pct"]
        lines.append(
            "| {h} | {m} | {tr} | {te} | {pos} | {auc} | {ap} | {brier} | {p10} | {r10} | {l10} |".format(
                h=item["horizon_days"],
                m=item["model"],
                tr=item["n_train"],
                te=item["n_test"],
                pos=item["test_positives"],
                auc=round(item["roc_auc"], 4) if item["roc_auc"] is not None else "",
                ap=round(item["average_precision"], 4) if item["average_precision"] is not None else "",
                brier=round(item["brier_score"], 4) if item["brier_score"] is not None else "",
                p10=round(top["precision"], 4) if top["precision"] is not None else "",
                r10=round(top["recall"], 4) if top["recall"] is not None else "",
                l10=round(top["lift"], 2) if top["lift"] is not None else "",
            )
        )
    lines.append("")
    best = best_by_average_precision(results)
    if best:
        lines.append("## Best Current Signal")
        lines.append("")
        for horizon, item in sorted(best.items()):
            top = item["top_10pct"]
            lines.append(
                f"- {horizon} days: `{item['model']}` has average precision {item['average_precision']:.4f}; top-decile precision {top['precision']:.4f}, recall {top['recall']:.4f}, lift {top['lift']:.2f}."
            )
        lines.append("")
    if not bootstrap.empty:
        lines.append("## Bootstrap Confidence Intervals")
        lines.append("")
        lines.append("Bootstrap intervals use row-level resampling of the out-of-time test set.")
        lines.append("")
        lines.append("| Horizon | Model | Metric | Mean | 95% CI low | 95% CI high | Valid resamples |")
        lines.append("|---:|---|---|---:|---:|---:|---:|")
        keep_metrics = ["roc_auc", "average_precision", "top_10pct_recall", "top_10pct_lift", "brier_score"]
        view = bootstrap[bootstrap["metric"].isin(keep_metrics)].copy()
        for row in view.itertuples(index=False):
            if row.mean is None or row.ci_low is None or row.ci_high is None:
                continue
            lines.append(
                f"| {row.horizon_days} | {row.model} | `{row.metric}` | {row.mean:.4f} | {row.ci_low:.4f} | {row.ci_high:.4f} | {row.n_boot_valid} |"
            )
        lines.append("")
    if not calibration.empty:
        lines.append("## Calibration Snapshot")
        lines.append("")
        lines.append("The table below shows the highest-risk decile only; full decile tables are saved in `dart_model_calibration.csv`.")
        lines.append("")
        lines.append("| Horizon | Model | Top bin count | Mean predicted risk | Observed rate | Positives |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        top_bins = calibration.sort_values(["horizon_days", "model", "risk_bin"]).groupby(["horizon_days", "model"]).tail(1)
        for row in top_bins.itertuples(index=False):
            lines.append(
                f"| {row.horizon_days} | {row.model} | {row.count} | {row.mean_predicted_risk:.4f} | {row.observed_rate:.4f} | {row.positives} |"
            )
        lines.append("")
    if not coeffs.empty:
        lines.append("## Strongest Combined Logistic Coefficients")
        lines.append("")
        combined = coeffs[coeffs["model"].eq("combined_logit")].copy()
        for horizon in sorted(combined["horizon_days"].unique()):
            h = combined[combined["horizon_days"].eq(horizon)].copy()
            h["abs_coef"] = h["coefficient"].abs()
            lines.append(f"### {horizon} days")
            lines.append("")
            lines.append("| Feature | Coefficient |")
            lines.append("|---|---:|")
            for row in h.sort_values("abs_coef", ascending=False).head(12).itertuples(index=False):
                lines.append(f"| `{row.feature}` | {row.coefficient:.4f} |")
            lines.append("")
    if not importances.empty:
        lines.append("## Strongest Random Forest Feature Importances")
        lines.append("")
        for horizon in sorted(importances["horizon_days"].unique()):
            h = importances[importances["horizon_days"].eq(horizon)].copy()
            lines.append(f"### {horizon} days")
            lines.append("")
            lines.append("| Feature | Importance |")
            lines.append("|---|---:|")
            for row in h.sort_values("importance", ascending=False).head(12).itertuples(index=False):
                lines.append(f"| `{row.feature}` | {row.importance:.4f} |")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-path", default=str(Path("Paper 4") / "data" / "model" / "dart_model_panel.parquet"))
    parser.add_argument("--summary-path", default=str(Path("Paper 4") / "data" / "model" / "dart_model_panel_summary.json"))
    parser.add_argument("--results-dir", default=str(Path("Paper 4") / "results"))
    parser.add_argument("--n-bootstrap", type=int, default=500)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel_path)
    panel["anchor_date"] = pd.to_datetime(panel["anchor_date"], errors="coerce")
    summary = json.loads(Path(args.summary_path).read_text(encoding="utf-8"))
    observation_end = pd.to_datetime(summary["outcome_observation_end"])

    results: list[dict] = []
    coeff_rows: list[dict] = []
    importance_rows: list[dict] = []
    prediction_row_data: list[dict] = []
    bootstrap_row_data: list[dict] = []
    calibration_row_data: list[dict] = []
    for horizon in (365, 730):
        h_results, h_coeffs, h_importances, h_predictions, h_bootstrap, h_calibration = run_horizon(
            panel,
            horizon,
            observation_end,
            args.n_bootstrap,
        )
        results.extend(h_results)
        coeff_rows.extend(h_coeffs)
        importance_rows.extend(h_importances)
        prediction_row_data.extend(h_predictions)
        bootstrap_row_data.extend(h_bootstrap)
        calibration_row_data.extend(h_calibration)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "dart_model_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    coeffs = pd.DataFrame(coeff_rows)
    if not coeffs.empty:
        coeffs.to_csv(results_dir / "dart_model_coefficients.csv", index=False, encoding="utf-8-sig")
    importances = pd.DataFrame(importance_rows)
    if not importances.empty:
        importances.to_csv(results_dir / "dart_model_feature_importance.csv", index=False, encoding="utf-8-sig")
    predictions = pd.DataFrame(prediction_row_data)
    if not predictions.empty:
        predictions.to_csv(results_dir / "dart_model_predictions.csv", index=False, encoding="utf-8-sig")
    bootstrap = pd.DataFrame(bootstrap_row_data)
    if not bootstrap.empty:
        bootstrap.to_csv(results_dir / "dart_model_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    calibration = pd.DataFrame(calibration_row_data)
    if not calibration.empty:
        calibration.to_csv(results_dir / "dart_model_calibration.csv", index=False, encoding="utf-8-sig")
    write_markdown(results, coeffs, importances, bootstrap, calibration, results_dir / "dart_model_metrics.md")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
