from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY_MODELS = [
    "sequence_logit",
    "financial_logit",
    "combined_logit",
    "combined_random_forest",
    "combined_calibrated_random_forest",
]


def load_metrics(path: Path, scenario: str) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in data:
        if item.get("status") != "ok":
            continue
        rows.append(
            {
                "scenario": scenario,
                "horizon_days": int(item["horizon_days"]),
                "model": item["model"],
                "n_train": int(item["n_train"]),
                "n_test": int(item["n_test"]),
                "test_positives": int(item["test_positives"]),
                "test_positive_rate": float(item["test_positive_rate"]),
                "roc_auc": item.get("roc_auc"),
                "average_precision": item.get("average_precision"),
                "brier_score": item.get("brier_score"),
                "top_10_precision": item["top_10pct"].get("precision"),
                "top_10_recall": item["top_10pct"].get("recall"),
                "top_10_lift": item["top_10pct"].get("lift"),
                "top_5_precision": item["top_5pct"].get("precision"),
                "top_5_recall": item["top_5pct"].get("recall"),
                "top_5_lift": item["top_5pct"].get("lift"),
            }
        )
    return pd.DataFrame(rows)


def load_summary(path: Path, scenario: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "scenario": scenario,
        "rows": data.get("rows"),
        "firms": data.get("firms"),
        "outcome_rows": data.get("outcome_rows"),
        "outcome_firms": data.get("outcome_firms"),
        "financial_match_rate": data.get("financial_match_rate"),
        "target_365_positive_rows": data.get("targets", {}).get("365d", {}).get("positive_rows"),
        "target_730_positive_rows": data.get("targets", {}).get("730d", {}).get("positive_rows"),
        "outcome_filter": data.get("outcome_filter", {}),
    }


def best_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    candidates = metrics[metrics["model"].isin(KEY_MODELS)].copy()
    candidates["rank_metric"] = candidates["average_precision"].fillna(-1)
    idx = candidates.groupby(["scenario", "horizon_days"])["rank_metric"].idxmax()
    return candidates.loc[idx].sort_values(["scenario", "horizon_days"])


def write_markdown(metrics: pd.DataFrame, summaries: list[dict], path: Path) -> None:
    lines = ["# DART Robustness Summary", ""]
    lines.append("Robust outcomes exclude weak `T02` listing-status outcomes and SPAC-like dissolution cases. The main outcomes keep all DART-derived distress proxies.")
    lines.append("")
    lines.append("## Dataset Comparison")
    lines.append("")
    lines.append("| Scenario | Rows | Firms | Outcome rows | Outcome firms | 365d positives | 730d positives | Financial match |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for summary in summaries:
        lines.append(
            "| {scenario} | {rows} | {firms} | {outcome_rows} | {outcome_firms} | {t365} | {t730} | {match:.3f} |".format(
                scenario=summary["scenario"],
                rows=summary["rows"],
                firms=summary["firms"],
                outcome_rows=summary["outcome_rows"],
                outcome_firms=summary["outcome_firms"],
                t365=summary["target_365_positive_rows"],
                t730=summary["target_730_positive_rows"],
                match=summary["financial_match_rate"],
            )
        )
    lines.append("")
    robust_filter = next((s["outcome_filter"] for s in summaries if s["scenario"] == "robust"), {})
    if robust_filter:
        lines.append("Robust filter exclusions:")
        lines.append("")
        lines.append(f"- Excluded rows: {robust_filter.get('excluded_rows')} across {robust_filter.get('excluded_firms')} firms.")
        by_rule = robust_filter.get("excluded_by_rule", {})
        lines.append(f"- Weak T02 listing-status rows: {by_rule.get('weak_t02_rows')} rows, {by_rule.get('weak_t02_firms')} firms.")
        lines.append(f"- SPAC-like dissolution rows: {by_rule.get('spac_dissolution_rows')} rows, {by_rule.get('spac_dissolution_firms')} firms.")
        lines.append("")

    lines.append("## Model Comparison")
    lines.append("")
    lines.append("| Scenario | Horizon | Model | Test positives | ROC AUC | Avg precision | Brier | Top 10% recall | Top 10% lift |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    view = metrics[metrics["model"].isin(["combined_random_forest", "combined_calibrated_random_forest", "sequence_logit", "financial_logit"])].copy()
    view = view.sort_values(["scenario", "horizon_days", "model"])
    for row in view.itertuples(index=False):
        lines.append(
            f"| {row.scenario} | {row.horizon_days} | `{row.model}` | {row.test_positives} | {row.roc_auc:.4f} | {row.average_precision:.4f} | {row.brier_score:.4f} | {row.top_10_recall:.4f} | {row.top_10_lift:.2f} |"
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    robust_rf = metrics[
        metrics["scenario"].eq("robust")
        & metrics["model"].eq("combined_random_forest")
        & metrics["horizon_days"].eq(730)
    ].iloc[0]
    lines.append(
        f"- After excluding weak outcomes, the 730-day combined random forest still reaches ROC AUC {robust_rf.roc_auc:.4f} and top-decile lift {robust_rf.top_10_lift:.2f}."
    )
    robust_cal = metrics[
        metrics["scenario"].eq("robust")
        & metrics["model"].eq("combined_calibrated_random_forest")
        & metrics["horizon_days"].eq(730)
    ].iloc[0]
    lines.append(
        f"- The robust calibrated random forest has lower Brier score ({robust_cal.brier_score:.4f}) and ROC AUC {robust_cal.roc_auc:.4f}, but the raw random forest has slightly higher average precision."
    )
    lines.append("- This supports reporting the main model as a watchlist-ranking model, with calibrated RF as the probability-oriented sensitivity check.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-results", default=str(Path("Paper 4") / "results" / "dart_model_metrics.json"))
    parser.add_argument("--robust-results", default=str(Path("Paper 4") / "results" / "robust" / "dart_model_metrics.json"))
    parser.add_argument("--main-summary", default=str(Path("Paper 4") / "data" / "model" / "dart_model_panel_summary.json"))
    parser.add_argument("--robust-summary", default=str(Path("Paper 4") / "data" / "model_robust" / "dart_model_panel_summary.json"))
    parser.add_argument("--out-dir", default=str(Path("Paper 4") / "results"))
    args = parser.parse_args()

    metrics = pd.concat(
        [
            load_metrics(Path(args.main_results), "main"),
            load_metrics(Path(args.robust_results), "robust"),
        ],
        ignore_index=True,
    )
    summaries = [
        load_summary(Path(args.main_summary), "main"),
        load_summary(Path(args.robust_summary), "robust"),
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "dart_robustness_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_json(out_dir / "dart_robustness_dataset_summary.json", orient="records", indent=2, force_ascii=False)
    best_rows(metrics).to_csv(out_dir / "dart_robustness_best_models.csv", index=False, encoding="utf-8-sig")
    write_markdown(metrics, summaries, out_dir / "dart_robustness_summary.md")
    print(f"wrote {out_dir / 'dart_robustness_summary.md'}")


if __name__ == "__main__":
    main()
