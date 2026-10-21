from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from train_dart_baselines import FINANCIAL_FEATURE_COLUMNS, evaluate_predictions, fit_predict, forest_model, sequence_columns


def feature_groups(features: list[str]) -> dict[str, str]:
    groups: dict[str, str] = {}
    for feature in features:
        if feature.startswith("seq_"):
            groups[feature] = "sequence"
        elif feature.startswith("fin_"):
            groups[feature] = "financial"
        else:
            groups[feature] = "other"
    return groups


def prepare_x(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = df[features].replace([np.inf, -np.inf], np.nan)
    recency_cols = [c for c in features if c.startswith("seq_days_since_last")]
    if recency_cols:
        x.loc[:, recency_cols] = x.loc[:, recency_cols].fillna(9999.0)
    return x


def run_horizon(panel: pd.DataFrame, horizon: int, observation_end: pd.Timestamp, n_repeats: int) -> tuple[list[dict], dict]:
    target = f"target_{horizon}d"
    usable = panel["anchor_date"] + pd.to_timedelta(horizon, unit="D") <= observation_end
    data = panel[usable].copy()
    train = data[data["anchor_year"].le(2021)].copy()
    test = data[data["anchor_year"].ge(2022)].copy()
    features = sequence_columns(data) + [c for c in FINANCIAL_FEATURE_COLUMNS if c in data.columns]
    y_test = test[target].astype(int).to_numpy()

    y_score, fitted = fit_predict(forest_model(), train, test, features, target)
    baseline = evaluate_predictions(y_test, y_score)
    x_test = prepare_x(test, features)
    groups = feature_groups(features)

    rows: list[dict] = []
    for scoring in ("average_precision", "roc_auc"):
        result = permutation_importance(
            fitted,
            x_test,
            y_test,
            n_repeats=n_repeats,
            random_state=42 + horizon,
            n_jobs=-1,
            scoring=scoring,
        )
        order = np.argsort(-result.importances_mean)
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "horizon_days": horizon,
                    "model": "combined_random_forest",
                    "scoring": scoring,
                    "rank": rank,
                    "feature": features[int(idx)],
                    "feature_group": groups[features[int(idx)]],
                    "importance_mean": float(result.importances_mean[int(idx)]),
                    "importance_std": float(result.importances_std[int(idx)]),
                    "n_repeats": int(n_repeats),
                }
            )

    summary = {
        "horizon_days": horizon,
        "model": "combined_random_forest",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_positives": int(test[target].sum()),
        "baseline_roc_auc": baseline["roc_auc"],
        "baseline_average_precision": baseline["average_precision"],
        "baseline_top_10pct_lift": baseline["top_10pct"]["lift"],
    }
    return rows, summary


def write_markdown(rows: pd.DataFrame, summaries: list[dict], path: Path) -> None:
    lines = ["# Permutation Importance", ""]
    lines.append("Permutation importance is computed on the out-of-time test split for the combined random forest. It complements the built-in random-forest feature importance table because it measures score degradation after shuffling each feature.")
    lines.append("")
    lines.append("## Baseline Scores")
    lines.append("")
    lines.append("| Horizon | Test rows | Test positives | ROC AUC | Average precision | Top 10% lift |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for item in summaries:
        lines.append(
            f"| {item['horizon_days']} | {item['n_test']:,} | {item['test_positives']} | {item['baseline_roc_auc']:.4f} | {item['baseline_average_precision']:.4f} | {item['baseline_top_10pct_lift']:.2f} |"
        )
    lines.append("")
    for scoring in ("average_precision", "roc_auc"):
        lines.append(f"## Top Features by {scoring}")
        lines.append("")
        lines.append("| Horizon | Rank | Feature | Group | Importance mean | Std |")
        lines.append("|---:|---:|---|---|---:|---:|")
        view = rows[rows["scoring"].eq(scoring)].sort_values(["horizon_days", "rank"]).groupby("horizon_days").head(15)
        for row in view.itertuples(index=False):
            lines.append(
                f"| {row.horizon_days} | {row.rank} | `{row.feature}` | {row.feature_group} | {row.importance_mean:.5f} | {row.importance_std:.5f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-path", default=str(Path("Paper 4") / "data" / "model" / "dart_model_panel.parquet"))
    parser.add_argument("--summary-path", default=str(Path("Paper 4") / "data" / "model" / "dart_model_panel_summary.json"))
    parser.add_argument("--results-dir", default=str(Path("Paper 4") / "results"))
    parser.add_argument("--n-repeats", type=int, default=5)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel_path)
    panel["anchor_date"] = pd.to_datetime(panel["anchor_date"], errors="coerce")
    summary = json.loads(Path(args.summary_path).read_text(encoding="utf-8"))
    observation_end = pd.to_datetime(summary["outcome_observation_end"])

    rows: list[dict] = []
    summaries: list[dict] = []
    for horizon in (365, 730):
        h_rows, h_summary = run_horizon(panel, horizon, observation_end, args.n_repeats)
        rows.extend(h_rows)
        summaries.append(h_summary)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(results_dir / "dart_model_permutation_importance.csv", index=False, encoding="utf-8-sig")
    (results_dir / "dart_model_permutation_importance_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(rows_df, summaries, results_dir / "dart_model_permutation_importance.md")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
