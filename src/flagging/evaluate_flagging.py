from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


OUTCOME_SUBTYPES = {
    "DART": {"rehabilitation", "dissolution", "default", "listing_or_trading_status"},
    "SEC": {"bankruptcy_or_receivership", "delisting_notice", "debt_acceleration_trigger"},
}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    severity: str
    description: str
    mode: str
    window_days: int | None = None
    min_count: int | None = None
    code_a: str | None = None
    code_b: str | None = None


RULES = [
    Rule(
        "R1_REPEATED_T05_365D",
        "MEDIUM",
        "At least two non-outcome T05 capital-structure-stress events within 365 days.",
        "repeat_code",
        window_days=365,
        min_count=2,
        code_a="T05",
    ),
    Rule(
        "R2_T05_T06_180D",
        "HIGH",
        "At least one non-outcome T05 and one non-outcome T06 event within 180 days.",
        "mixed_codes",
        window_days=180,
        code_a="T05",
        code_b="T06",
    ),
    Rule(
        "R3_PRIOR_T03",
        "HIGH",
        "Any non-outcome T03 debt-default or acceleration signal before the outcome.",
        "any_code",
        code_a="T03",
    ),
    Rule(
        "R4_T06_THEN_T05_365D",
        "MEDIUM",
        "T06 governance/control change followed by T05 capital-structure stress within 365 days.",
        "ordered_pair",
        window_days=365,
        code_a="T06",
        code_b="T05",
    ),
]


def read_source(processed_dir: Path, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_l = source.lower()
    events = pd.read_parquet(processed_dir / f"events_{source_l}.parquet")
    outcomes = pd.read_parquet(processed_dir / f"outcomes_{source_l}.parquet")
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    outcomes["outcome_date"] = pd.to_datetime(outcomes["outcome_date"], errors="coerce")
    return events, outcomes


def apply_rule(events: pd.DataFrame, rule: Rule) -> pd.Timestamp | None:
    events = events.sort_values("event_date")
    if rule.mode == "repeat_code":
        subset = events[events["taxonomy_code"].eq(rule.code_a)]
        dates = list(subset["event_date"].dropna())
        for i, date in enumerate(dates):
            start = date - pd.Timedelta(days=rule.window_days or 0)
            if sum(1 for d in dates[: i + 1] if start <= d <= date) >= (rule.min_count or 1):
                return pd.Timestamp(date)
        return None

    if rule.mode == "mixed_codes":
        subset = events[events["taxonomy_code"].isin({rule.code_a, rule.code_b})]
        rows = [(str(r.taxonomy_code), pd.Timestamp(r.event_date)) for r in subset.itertuples(index=False) if pd.notna(r.event_date)]
        for i, (_, date) in enumerate(rows):
            start = date - pd.Timedelta(days=rule.window_days or 0)
            codes = {code for code, d in rows[: i + 1] if start <= d <= date}
            if rule.code_a in codes and rule.code_b in codes:
                return date
        return None

    if rule.mode == "any_code":
        subset = events[events["taxonomy_code"].eq(rule.code_a)]
        if subset.empty:
            return None
        return pd.Timestamp(subset["event_date"].min())

    if rule.mode == "ordered_pair":
        a_dates = list(events.loc[events["taxonomy_code"].eq(rule.code_a), "event_date"].dropna())
        b_dates = list(events.loc[events["taxonomy_code"].eq(rule.code_b), "event_date"].dropna())
        for a in a_dates:
            for b in b_dates:
                if a <= b <= a + pd.Timedelta(days=rule.window_days or 0):
                    return pd.Timestamp(b)
        return None

    raise ValueError(f"Unknown rule mode: {rule.mode}")


def build_rule_flags(events: pd.DataFrame, outcomes: pd.DataFrame, source: str) -> pd.DataFrame:
    outcome_by_firm = {firm_id: g.sort_values("outcome_date") for firm_id, g in outcomes.groupby("firm_id")}
    outcome_subtypes = OUTCOME_SUBTYPES[source]
    rows: list[dict] = []

    for firm_id, group in events.groupby("firm_id"):
        group = group.sort_values("event_date").copy()
        out = outcome_by_firm.get(firm_id)
        has_outcome = out is not None and not out.empty
        outcome_date = pd.Timestamp(out["outcome_date"].iloc[0]) if has_outcome else pd.NaT
        outcome_label = str(out["outcome_label"].iloc[0]) if has_outcome else ""
        name_col = "corp_name" if source == "DART" else "company_name"
        firm_name = str(group[name_col].dropna().iloc[0]) if name_col in group.columns and group[name_col].notna().any() else ""

        visible = group[group["event_date"].lt(outcome_date)].copy() if has_outcome else group.copy()
        visible = visible[~visible["event_subtype"].isin(outcome_subtypes)].copy()

        for rule in RULES:
            flag_date = apply_rule(visible, rule)
            lead_days = int((outcome_date - flag_date).days) if has_outcome and flag_date is not None else None
            rows.append(
                {
                    "source": source,
                    "firm_id": firm_id,
                    "firm_name": firm_name,
                    "rule_id": rule.rule_id,
                    "severity": rule.severity,
                    "description": rule.description,
                    "flagged": flag_date is not None,
                    "flag_date": flag_date,
                    "has_outcome": bool(has_outcome),
                    "outcome_date": outcome_date,
                    "outcome_label": outcome_label,
                    "lead_days": lead_days,
                    "early_true_positive": bool(has_outcome and lead_days is not None and lead_days > 0),
                }
            )
    return pd.DataFrame(rows)


def evaluate_flags(flags: pd.DataFrame, horizon_days: int | None) -> list[dict]:
    rows: list[dict] = []
    for (source, rule_id), group in flags.groupby(["source", "rule_id"]):
        if horizon_days is None:
            hit = group["early_true_positive"]
            horizon_label = "any_positive_lead"
        else:
            hit = group["early_true_positive"] & group["lead_days"].notna() & group["lead_days"].le(horizon_days)
            horizon_label = str(horizon_days)
        flagged = group["flagged"]
        outcome = group["has_outcome"]
        tp = int(hit.sum())
        fp = int((flagged & ~hit).sum())
        fn = int((outcome & ~hit).sum())
        tn = int((~outcome & ~flagged).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / int(outcome.sum()) if int(outcome.sum()) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        leads = group.loc[hit, "lead_days"].dropna()
        first = group.iloc[0]
        rows.append(
            {
                "evaluation_type": "rule",
                "source": source,
                "horizon_days": horizon_label,
                "method": rule_id,
                "severity": first["severity"],
                "description": first["description"],
                "firms": int(group["firm_id"].nunique()),
                "outcome_firms": int(outcome.sum()),
                "flagged": int(flagged.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "median_lead_days": float(leads.median()) if not leads.empty else None,
                "mean_lead_days": float(leads.mean()) if not leads.empty else None,
            }
        )
    return rows


def evaluate_model_watchlists(predictions_path: Path) -> list[dict]:
    if not predictions_path.exists():
        return []
    preds = pd.read_csv(predictions_path, low_memory=False)
    rows: list[dict] = []
    for (horizon, model), group in preds.groupby(["horizon_days", "model"]):
        group = group.copy()
        top_n = max(1, int(round(len(group) * 0.10)))
        top = group.sort_values("predicted_risk", ascending=False).head(top_n)
        target = group["target"].astype(bool)
        tp = int(top["target"].astype(bool).sum())
        fp = int(len(top) - tp)
        fn = int(target.sum() - tp)
        tn = int(len(group) - top_n - fn)
        precision = tp / top_n if top_n else 0.0
        recall = tp / int(target.sum()) if int(target.sum()) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        leads = pd.to_numeric(top.loc[top["target"].astype(bool), "days_to_outcome"], errors="coerce").dropna()
        rows.append(
            {
                "evaluation_type": "model_top10pct",
                "source": "DART",
                "horizon_days": str(int(horizon)),
                "method": str(model),
                "severity": "WATCHLIST",
                "description": "Top decile by out-of-time predicted risk.",
                "firms": int(len(group)),
                "outcome_firms": int(target.sum()),
                "flagged": int(top_n),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "median_lead_days": float(leads.median()) if not leads.empty else None,
                "mean_lead_days": float(leads.mean()) if not leads.empty else None,
            }
        )
    return rows


def write_markdown(results: pd.DataFrame, path: Path) -> None:
    lines = ["# SupTech Flagging Performance", ""]
    lines.append("Rule-based flags use only pre-outcome non-outcome event sequences. Model watchlists are top-decile out-of-time firm-year predictions from the DART model dataset.")
    lines.append("")
    lines.append("## Best DART Watchlist Rows")
    lines.append("")
    lines.append("| Type | Horizon | Method | Flagged | TP | FP | Precision | Recall | F1 | Median lead days |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    dart_view = results[results["source"].eq("DART")].copy()
    if not dart_view.empty:
        ranked = dart_view.sort_values(["horizon_days", "recall", "precision"], ascending=[True, False, False])
        for row in ranked.head(30).itertuples(index=False):
            median = "" if pd.isna(row.median_lead_days) else f"{row.median_lead_days:.1f}"
            lines.append(
                f"| {row.evaluation_type} | {row.horizon_days} | `{row.method}` | {row.flagged} | {row.tp} | {row.fp} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {median} |"
            )
    lines.append("")
    lines.append("## Full Rule and Model Table")
    lines.append("")
    lines.append("| Source | Type | Horizon | Method | Severity | Flagged | TP | FP | FN | Precision | Recall | F1 | Median lead days |")
    lines.append("|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in results.sort_values(["source", "evaluation_type", "horizon_days", "method"]).itertuples(index=False):
        median = "" if pd.isna(row.median_lead_days) else f"{row.median_lead_days:.1f}"
        lines.append(
            f"| {row.source} | {row.evaluation_type} | {row.horizon_days} | `{row.method}` | {row.severity} | {row.flagged} | {row.tp} | {row.fp} | {row.fn} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {median} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Broad hand-written sequence rules provide long lead times but low precision in DART. The combined model watchlist is the stronger v1 SupTech artifact because it converts the same sequence layer plus financial fundamentals into a ranked review queue.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=str(Path("Paper 4") / "data" / "processed"))
    parser.add_argument("--results-dir", default=str(Path("Paper 4") / "results"))
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_flags = []
    all_results = []
    for source in ("DART", "SEC"):
        events, outcomes = read_source(processed_dir, source)
        flags = build_rule_flags(events, outcomes, source)
        all_flags.append(flags)
        for horizon in (None, 180, 365, 730):
            all_results.extend(evaluate_flags(flags, horizon))

    flags_df = pd.concat(all_flags, ignore_index=True)
    model_rows = evaluate_model_watchlists(results_dir / "dart_model_predictions.csv")
    all_results.extend(model_rows)
    results_df = pd.DataFrame(all_results)

    flags_df.to_csv(results_dir / "flagging_rule_firm_flags.csv", index=False, encoding="utf-8-sig")
    results_df.to_csv(results_dir / "flagging_performance.csv", index=False, encoding="utf-8-sig")
    (results_dir / "flagging_performance.json").write_text(
        json.dumps(results_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(results_df, results_dir / "flagging_performance.md")
    print(json.dumps({"rows": int(len(results_df)), "flags": int(len(flags_df))}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
