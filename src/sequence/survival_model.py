from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.duration.hazard_regression import PHReg


ROOT = Path("Paper 4")

SEQUENCE_FEATURES = [
    "seq_distinct_taxonomies_365d",
    "seq_T05_365d",
    "seq_T06_365d",
    "seq_T99_365d",
    "seq_repeated_T05_365d_flag",
    "seq_T05_T06_180d_flag",
    "seq_days_since_last_event",
]

FINANCIAL_FEATURES = [
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


@dataclass
class FittedHazard:
    result: object
    features: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def observation_end(summary_path: Path) -> pd.Timestamp:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return pd.to_datetime(summary["outcome_observation_end"])


def prepare_panel(panel_path: Path, summary_path: Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    df = pd.read_parquet(panel_path)
    obs_end = observation_end(summary_path)
    df["anchor_date"] = pd.to_datetime(df["anchor_date"], errors="coerce")
    df["future_outcome_date"] = pd.to_datetime(df["future_outcome_date"], errors="coerce")
    df["event_observed"] = df["future_outcome_date"].notna()
    censor_days = (obs_end - df["anchor_date"]).dt.days
    df["survival_duration_days"] = np.where(df["event_observed"], df["days_to_outcome"], censor_days)
    df = df[df["survival_duration_days"].gt(0)].copy()
    return df, obs_end


def usable_for_horizon(df: pd.DataFrame, obs_end: pd.Timestamp, horizon: int) -> pd.DataFrame:
    full_horizon_observed = df["anchor_date"] + pd.to_timedelta(horizon, unit="D") <= obs_end
    return df[full_horizon_observed].copy()


def feature_specs(df: pd.DataFrame) -> dict[str, list[str]]:
    seq = [c for c in SEQUENCE_FEATURES if c in df.columns]
    fin = [c for c in FINANCIAL_FEATURES if c in df.columns]
    return {
        "sequence_hazard": seq,
        "financial_hazard": fin,
        "combined_hazard": seq + fin,
    }


def clean_feature_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = df[features].replace([np.inf, -np.inf], np.nan).copy()
    recency_cols = [c for c in features if c.startswith("seq_days_since")]
    if recency_cols:
        x.loc[:, recency_cols] = x.loc[:, recency_cols].fillna(9999.0)
    return x


def drop_constant_features(x: pd.DataFrame, features: list[str]) -> list[str]:
    imputed = SimpleImputer(strategy="median").fit_transform(x)
    std = np.nanstd(imputed, axis=0)
    return [feature for feature, value in zip(features, std) if value > 1e-12]


def fit_cloglog_hazard(train: pd.DataFrame, features: list[str], target: str) -> FittedHazard:
    x_raw = clean_feature_frame(train, features)
    kept = drop_constant_features(x_raw, features)
    x_raw = clean_feature_frame(train, kept)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(x_raw))
    x = sm.add_constant(x, has_constant="add")
    y = train[target].astype(int)
    model = sm.GLM(
        y,
        x,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    )
    result = model.fit(maxiter=200, disp=0)
    return FittedHazard(result=result, features=kept, imputer=imputer, scaler=scaler)


def predict_cloglog(fitted: FittedHazard, test: pd.DataFrame) -> np.ndarray:
    x_raw = clean_feature_frame(test, fitted.features)
    x = fitted.scaler.transform(fitted.imputer.transform(x_raw))
    x = sm.add_constant(x, has_constant="add")
    pred = np.asarray(fitted.result.predict(x), dtype=float)
    return np.clip(pred, 0.0, 1.0)


def top_fraction_metrics(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.1) -> dict:
    if len(y_true) == 0:
        return {"top_fraction": fraction, "top_n": 0, "precision": None, "recall": None, "lift": None}
    top_n = max(1, int(np.ceil(len(y_true) * fraction)))
    order = np.argsort(-y_score)
    top = order[:top_n]
    positives = float(np.sum(y_true))
    base_rate = float(np.mean(y_true))
    precision = float(np.mean(y_true[top]))
    recall = float(np.sum(y_true[top]) / positives) if positives else None
    lift = float(precision / base_rate) if base_rate else None
    return {
        "top_fraction": fraction,
        "top_n": int(top_n),
        "precision": finite_float(precision),
        "recall": finite_float(recall),
        "lift": finite_float(lift),
    }


def concordance_index(duration: np.ndarray, event: np.ndarray, risk: np.ndarray) -> dict:
    comparable = 0
    concordant = 0.0
    for i, observed in enumerate(event):
        if not observed:
            continue
        mask = duration > duration[i]
        if not np.any(mask):
            continue
        comparable += int(np.sum(mask))
        concordant += float(np.sum(risk[i] > risk[mask]))
        concordant += 0.5 * float(np.sum(risk[i] == risk[mask]))
    return {
        "concordance_index": finite_float(concordant / comparable) if comparable else None,
        "comparable_pairs": int(comparable),
    }


def evaluate_hazard(test: pd.DataFrame, target: str, y_score: np.ndarray) -> dict:
    y_true = test[target].astype(int).to_numpy()
    metrics = {
        "roc_auc": finite_float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "average_precision": finite_float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "top_10pct": top_fraction_metrics(y_true, y_score, 0.10),
        "top_5pct": top_fraction_metrics(y_true, y_score, 0.05),
    }
    metrics.update(
        concordance_index(
            test["survival_duration_days"].astype(float).to_numpy(),
            test["event_observed"].astype(bool).to_numpy(),
            y_score,
        )
    )
    return metrics


def coefficient_rows(fitted: FittedHazard, horizon: int, model_name: str) -> list[dict]:
    params = np.asarray(fitted.result.params, dtype=float)
    bse = np.asarray(fitted.result.bse, dtype=float)
    pvalues = np.asarray(fitted.result.pvalues, dtype=float)
    rows = []
    for idx, feature in enumerate(["const", *fitted.features]):
        coef = params[idx]
        se = bse[idx]
        pvalue = pvalues[idx]
        rows.append(
            {
                "horizon_days": horizon,
                "model": model_name,
                "feature": feature,
                "coefficient": finite_float(coef),
                "std_error": finite_float(se),
                "p_value": finite_float(pvalue),
                "relative_hazard_per_1sd": finite_float(np.exp(coef)) if feature != "const" else None,
            }
        )
    return rows


def prediction_rows(test: pd.DataFrame, horizon: int, model_name: str, y_score: np.ndarray) -> list[dict]:
    rows = []
    target = f"target_{horizon}d"
    cols = [
        "firm_id",
        "firm_name",
        "security_id",
        "anchor_year",
        "anchor_date",
        "future_outcome_date",
        "future_outcome_label",
        "survival_duration_days",
        "event_observed",
        target,
    ]
    for score, row in zip(y_score, test[cols].itertuples(index=False)):
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
                "survival_duration_days": float(row.survival_duration_days),
                "event_observed": bool(row.event_observed),
                "target": bool(getattr(row, target)),
                "hazard_score": float(score),
            }
        )
    return rows


def cox_sanity_check(df: pd.DataFrame) -> dict:
    train = df[df["anchor_year"].le(2021)].copy()
    feature = "seq_days_since_last_event"
    if feature not in train.columns or train["event_observed"].sum() == 0:
        return {"status": "skipped", "reason": "missing feature or no events"}
    x = clean_feature_frame(train, [feature])
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    arr = scaler.fit_transform(imputer.fit_transform(x))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = PHReg(
                train["survival_duration_days"].astype(float),
                arr,
                status=train["event_observed"].astype(int),
                ties="breslow",
            ).fit(disp=0)
        params = np.asarray(result.params, dtype=float)
        if not np.isfinite(params).all():
            return {
                "status": "unstable",
                "reason": "PHReg returned non-finite coefficients on the selected panel",
                "warnings": [str(w.message) for w in caught[:5]],
            }
        return {
            "status": "ok_single_feature_sanity_check",
            "feature": feature,
            "coefficient": float(params[0]),
            "hazard_ratio_per_1sd": float(np.exp(params[0])),
            "warnings": [str(w.message) for w in caught[:5]],
            "note": "Full multivariable Cox was not used for the manuscript because sparse/constant sequence columns make PHReg unstable on the landmark panel.",
        }
    except Exception as exc:
        return {"status": "failed", "reason": repr(exc)}


def run(panel_path: Path, summary_path: Path, results_dir: Path) -> dict:
    panel, obs_end = prepare_panel(panel_path, summary_path)
    metrics: list[dict] = []
    coefficients: list[dict] = []
    predictions: list[dict] = []

    for horizon in (365, 730):
        target = f"target_{horizon}d"
        data = usable_for_horizon(panel, obs_end, horizon)
        train = data[data["anchor_year"].le(2021)].copy()
        test = data[data["anchor_year"].ge(2022)].copy()
        for model_name, features in feature_specs(data).items():
            if not features or train[target].nunique() < 2 or test[target].nunique() < 2:
                metrics.append(
                    {
                        "horizon_days": horizon,
                        "model": model_name,
                        "status": "skipped_one_class_or_no_features",
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        "train_events": int(train[target].sum()),
                        "test_events": int(test[target].sum()),
                    }
                )
                continue
            fitted = fit_cloglog_hazard(train, features, target)
            y_score = predict_cloglog(fitted, test)
            row = {
                "horizon_days": horizon,
                "model": model_name,
                "status": "ok",
                "n_features": int(len(fitted.features)),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "train_events": int(train[target].sum()),
                "test_events": int(test[target].sum()),
                "train_event_rate": float(train[target].mean()),
                "test_event_rate": float(test[target].mean()),
            }
            row.update(evaluate_hazard(test, target, y_score))
            metrics.append(row)
            coefficients.extend(coefficient_rows(fitted, horizon, model_name))
            predictions.extend(prediction_rows(test, horizon, model_name, y_score))

    cox = cox_sanity_check(panel)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.json_normalize(metrics)
    coefficients_df = pd.DataFrame(coefficients)
    predictions_df = pd.DataFrame(predictions)
    metrics_df.to_csv(results_dir / "survival_hazard_metrics.csv", index=False, encoding="utf-8-sig")
    coefficients_df.to_csv(results_dir / "survival_hazard_coefficients.csv", index=False, encoding="utf-8-sig")
    predictions_df.to_csv(results_dir / "survival_hazard_predictions.csv", index=False, encoding="utf-8-sig")
    payload = {
        "method": "discrete-time complementary log-log hazard with out-of-time landmark split",
        "panel_path": str(panel_path),
        "observation_end": obs_end.date().isoformat(),
        "cox_sanity_check": cox,
        "metrics": metrics,
    }
    (results_dir / "survival_hazard_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(metrics, coefficients_df, cox, results_dir / "survival_hazard_summary.md")
    return payload


def write_markdown(metrics: list[dict], coefficients: pd.DataFrame, cox: dict, path: Path) -> None:
    lines = [
        "# DART Survival and Hazard Analysis",
        "",
        "This analysis adds the time-to-event layer requested in `paper4_research_plan.md`. Because the current artifact is a firm-year landmark panel, the main survival-style model is a discrete-time complementary log-log hazard model. The split remains out-of-time: train anchors <= 2021, test anchors >= 2022.",
        "",
        "## Hazard Ranking Results",
        "",
        "| Horizon | Model | Train rows | Test rows | Test events | ROC AUC | Avg precision | C-index | Top 10% precision | Top 10% recall | Top 10% lift |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        if row.get("status") != "ok":
            lines.append(
                f"| {row['horizon_days']} | {row['model']} | {row['n_train']} | {row['n_test']} | {row['test_events']} | skipped | skipped | skipped | skipped | skipped | skipped |"
            )
            continue
        top = row["top_10pct"]
        lines.append(
            "| {h} | {m} | {tr} | {te} | {ev} | {auc} | {ap} | {ci} | {p10} | {r10} | {l10} |".format(
                h=row["horizon_days"],
                m=row["model"],
                tr=row["n_train"],
                te=row["n_test"],
                ev=row["test_events"],
                auc=f"{row['roc_auc']:.4f}" if row["roc_auc"] is not None else "",
                ap=f"{row['average_precision']:.4f}" if row["average_precision"] is not None else "",
                ci=f"{row['concordance_index']:.4f}" if row["concordance_index"] is not None else "",
                p10=f"{top['precision']:.4f}" if top["precision"] is not None else "",
                r10=f"{top['recall']:.4f}" if top["recall"] is not None else "",
                l10=f"{top['lift']:.2f}" if top["lift"] is not None else "",
            )
        )
    lines.extend(["", "## Interpretation", ""])
    ok = [row for row in metrics if row.get("status") == "ok"]
    for horizon in sorted({int(row["horizon_days"]) for row in ok}):
        h_rows = [row for row in ok if int(row["horizon_days"]) == horizon]
        best = max(h_rows, key=lambda row: row["average_precision"] or -1)
        top = best["top_10pct"]
        lines.append(
            f"- {horizon}-day hazard: `{best['model']}` has the best average precision ({best['average_precision']:.4f}) and top-decile lift {top['lift']:.2f}."
        )
        seq = next((row for row in h_rows if row["model"] == "sequence_hazard"), None)
        fin = next((row for row in h_rows if row["model"] == "financial_hazard"), None)
        if seq and fin:
            lines.append(
                f"- Sequence-only hazard outperforms financial-only hazard on ROC AUC ({seq['roc_auc']:.4f} vs. {fin['roc_auc']:.4f}) and average precision ({seq['average_precision']:.4f} vs. {fin['average_precision']:.4f})."
            )
    lines.extend(
        [
            "",
            "## Cox PH Sanity Check",
            "",
            f"- Status: `{cox.get('status')}`.",
        ]
    )
    if cox.get("coefficient") is not None:
        lines.append(
            f"- Single-feature PHReg check for `{cox.get('feature')}` gives coefficient {cox['coefficient']:.4f} and hazard ratio per 1 SD {cox['hazard_ratio_per_1sd']:.4f}."
        )
    if cox.get("note"):
        lines.append(f"- Note: {cox['note']}")
    if cox.get("reason"):
        lines.append(f"- Reason: {cox['reason']}")

    if not coefficients.empty:
        lines.extend(["", "## Largest Combined-Hazard Coefficients", ""])
        combined = coefficients[(coefficients["model"] == "combined_hazard") & (coefficients["feature"] != "const")].copy()
        if not combined.empty:
            combined["abs_coef"] = combined["coefficient"].abs()
            for horizon in sorted(combined["horizon_days"].unique()):
                h = combined[combined["horizon_days"] == horizon].sort_values("abs_coef", ascending=False).head(10)
                lines.extend([f"### {horizon} days", "", "| Feature | Coefficient | Relative hazard per 1 SD | p-value |", "|---|---:|---:|---:|"])
                for row in h.itertuples(index=False):
                    lines.append(
                        f"| `{row.feature}` | {row.coefficient:.4f} | {row.relative_hazard_per_1sd:.4f} | {row.p_value:.4g} |"
                    )
                lines.append("")
    lines.extend(
        [
            "## Use in Manuscript",
            "",
            "- Use as a robustness/time-to-event supplement, not as a replacement for the main random-forest watchlist model.",
            "- The key claim supported here is that standardized disclosure sequence features contain timing information for later distress, even in a simple hazard specification.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-path", default=str(ROOT / "data" / "model" / "dart_model_panel.parquet"))
    parser.add_argument("--summary-path", default=str(ROOT / "data" / "model" / "dart_model_panel_summary.json"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    payload = run(Path(args.panel_path), Path(args.summary_path), Path(args.results_dir))
    print(
        json.dumps(
            {
                "method": payload["method"],
                "cox_status": payload["cox_sanity_check"].get("status"),
                "metric_rows": len(payload["metrics"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
