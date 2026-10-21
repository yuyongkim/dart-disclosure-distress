from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path("Paper 4")

OUTCOME_SUBTYPES = {
    "DART": {"rehabilitation", "dissolution", "default", "listing_or_trading_status"},
    "SEC": {"bankruptcy_or_receivership", "delisting_notice", "debt_acceleration_trigger"},
}


@dataclass(frozen=True)
class FlaggingRule:
    rule_id: str
    severity: str
    action: str
    pattern: tuple[str, ...]
    window_days: int
    mode: str
    include_outcome_events: bool
    description: str


FLAGGING_RULES = [
    FlaggingRule(
        rule_id="CRITICAL_T01_IMMEDIATE",
        severity="CRITICAL",
        action="IMMEDIATE_REVIEW",
        pattern=("T01",),
        window_days=1,
        mode="any_code",
        include_outcome_events=True,
        description="Any insolvency, bankruptcy, rehabilitation, or receivership event.",
    ),
    FlaggingRule(
        rule_id="CRITICAL_T02_IMMEDIATE",
        severity="CRITICAL",
        action="IMMEDIATE_REVIEW",
        pattern=("T02",),
        window_days=1,
        mode="any_code",
        include_outcome_events=True,
        description="Any delisting or trading-status event.",
    ),
    FlaggingRule(
        rule_id="HIGH_T04_THEN_T03_90D",
        severity="HIGH",
        action="PRIORITY_REVIEW",
        pattern=("T04", "T03"),
        window_days=90,
        mode="ordered_pair",
        include_outcome_events=False,
        description="Going-concern/audit concern followed by debt default or acceleration within 90 days.",
    ),
    FlaggingRule(
        rule_id="HIGH_T05_T06_30D",
        severity="HIGH",
        action="PRIORITY_REVIEW",
        pattern=("T05", "T06"),
        window_days=30,
        mode="mixed_codes",
        include_outcome_events=False,
        description="Capital-structure stress and governance/control change occurring within 30 days.",
    ),
    FlaggingRule(
        rule_id="MEDIUM_T05_THEN_T04_365D",
        severity="MEDIUM",
        action="WATCHLIST",
        pattern=("T05", "T04"),
        window_days=365,
        mode="ordered_pair",
        include_outcome_events=False,
        description="Capital-structure stress followed by going-concern/audit concern within 365 days.",
    ),
    FlaggingRule(
        rule_id="MEDIUM_REPEATED_T05_365D",
        severity="MEDIUM",
        action="WATCHLIST",
        pattern=("T05", "T05"),
        window_days=365,
        mode="repeat_code",
        include_outcome_events=False,
        description="At least two capital-structure-stress events within 365 days.",
    ),
]


def read_source(processed_dir: Path, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_l = source.lower()
    events = pd.read_parquet(processed_dir / f"events_{source_l}.parquet")
    outcomes = pd.read_parquet(processed_dir / f"outcomes_{source_l}.parquet")
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    outcomes["outcome_date"] = pd.to_datetime(outcomes["outcome_date"], errors="coerce")
    return events, outcomes


def firm_name(events: pd.DataFrame, source: str) -> str:
    name_col = "corp_name" if source == "DART" else "company_name"
    if name_col not in events.columns or not events[name_col].notna().any():
        return ""
    return str(events[name_col].dropna().iloc[0])


def rule_event_pool(events: pd.DataFrame, source: str, rule: FlaggingRule, first_outcome_date: pd.Timestamp | pd.NaT) -> pd.DataFrame:
    pool = events.sort_values("event_date").copy()
    if not rule.include_outcome_events:
        pool = pool[~pool["event_subtype"].isin(OUTCOME_SUBTYPES[source])].copy()
    if pd.notna(first_outcome_date):
        if rule.include_outcome_events:
            pool = pool[pool["event_date"].le(first_outcome_date)].copy()
        else:
            pool = pool[pool["event_date"].lt(first_outcome_date)].copy()
    return pool


def first_any_code(events: pd.DataFrame, code: str) -> pd.Timestamp | None:
    subset = events[events["taxonomy_code"].eq(code)]
    if subset.empty:
        return None
    return pd.Timestamp(subset["event_date"].min())


def first_repeat_code(events: pd.DataFrame, code: str, min_count: int, window_days: int) -> pd.Timestamp | None:
    dates = [pd.Timestamp(value) for value in events.loc[events["taxonomy_code"].eq(code), "event_date"].dropna()]
    for i, date in enumerate(dates):
        start = date - pd.Timedelta(days=window_days)
        if sum(1 for value in dates[: i + 1] if start <= value <= date) >= min_count:
            return date
    return None


def first_mixed_codes(events: pd.DataFrame, code_a: str, code_b: str, window_days: int) -> pd.Timestamp | None:
    subset = events[events["taxonomy_code"].isin({code_a, code_b})].sort_values("event_date")
    rows = [(str(row.taxonomy_code), pd.Timestamp(row.event_date)) for row in subset.itertuples(index=False) if pd.notna(row.event_date)]
    for i, (_, date) in enumerate(rows):
        start = date - pd.Timedelta(days=window_days)
        codes = {code for code, value in rows[: i + 1] if start <= value <= date}
        if code_a in codes and code_b in codes:
            return date
    return None


def first_ordered_pair(events: pd.DataFrame, code_a: str, code_b: str, window_days: int) -> pd.Timestamp | None:
    a_dates = [pd.Timestamp(value) for value in events.loc[events["taxonomy_code"].eq(code_a), "event_date"].dropna()]
    b_dates = [pd.Timestamp(value) for value in events.loc[events["taxonomy_code"].eq(code_b), "event_date"].dropna()]
    best: pd.Timestamp | None = None
    for a_date in a_dates:
        for b_date in b_dates:
            if a_date <= b_date <= a_date + pd.Timedelta(days=window_days):
                if best is None or b_date < best:
                    best = b_date
    return best


def apply_rule(events: pd.DataFrame, rule: FlaggingRule) -> pd.Timestamp | None:
    if events.empty:
        return None
    if rule.mode == "any_code":
        return first_any_code(events, rule.pattern[0])
    if rule.mode == "repeat_code":
        return first_repeat_code(events, rule.pattern[0], min_count=len(rule.pattern), window_days=rule.window_days)
    if rule.mode == "mixed_codes":
        return first_mixed_codes(events, rule.pattern[0], rule.pattern[1], rule.window_days)
    if rule.mode == "ordered_pair":
        return first_ordered_pair(events, rule.pattern[0], rule.pattern[1], rule.window_days)
    raise ValueError(f"Unknown rule mode: {rule.mode}")


def build_flags(events: pd.DataFrame, outcomes: pd.DataFrame, source: str) -> pd.DataFrame:
    outcome_by_firm = {firm_id: group.sort_values("outcome_date") for firm_id, group in outcomes.groupby("firm_id")}
    rows: list[dict] = []

    for firm_id, group in events.groupby("firm_id"):
        group = group.sort_values("event_date").copy()
        outcome_group = outcome_by_firm.get(firm_id)
        has_outcome = outcome_group is not None and not outcome_group.empty
        first_outcome_date = pd.Timestamp(outcome_group["outcome_date"].iloc[0]) if has_outcome else pd.NaT
        outcome_label = str(outcome_group["outcome_label"].iloc[0]) if has_outcome else ""

        for rule in FLAGGING_RULES:
            pool = rule_event_pool(group, source, rule, first_outcome_date)
            flag_date = apply_rule(pool, rule)
            lead_days = int((first_outcome_date - flag_date).days) if has_outcome and flag_date is not None else None
            rows.append(
                {
                    "source": source,
                    "firm_id": firm_id,
                    "firm_name": firm_name(group, source),
                    "rule_id": rule.rule_id,
                    "severity": rule.severity,
                    "action": rule.action,
                    "pattern": ">".join(rule.pattern),
                    "window_days": int(rule.window_days),
                    "mode": rule.mode,
                    "include_outcome_events": bool(rule.include_outcome_events),
                    "description": rule.description,
                    "flagged": flag_date is not None,
                    "flag_date": flag_date,
                    "has_outcome": bool(has_outcome),
                    "outcome_date": first_outcome_date,
                    "outcome_label": outcome_label,
                    "lead_days": lead_days,
                    "early_true_positive": bool(has_outcome and lead_days is not None and lead_days > 0),
                    "same_day_or_reactive": bool(has_outcome and lead_days is not None and lead_days <= 0),
                }
            )
    return pd.DataFrame(rows)


def summarize_flags(flags: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (source, rule_id), group in flags.groupby(["source", "rule_id"], dropna=False):
        flagged = group["flagged"].astype(bool)
        has_outcome = group["has_outcome"].astype(bool)
        early_tp = group["early_true_positive"].astype(bool)
        same_day = group["same_day_or_reactive"].astype(bool)
        leads = pd.to_numeric(group.loc[early_tp, "lead_days"], errors="coerce").dropna()
        first = group.iloc[0]
        rows.append(
            {
                "source": source,
                "rule_id": rule_id,
                "severity": first["severity"],
                "action": first["action"],
                "pattern": first["pattern"],
                "window_days": int(first["window_days"]),
                "include_outcome_events": bool(first["include_outcome_events"]),
                "firms": int(group["firm_id"].nunique()),
                "outcome_firms": int(has_outcome.sum()),
                "flagged_firms": int(flagged.sum()),
                "early_true_positive_firms": int(early_tp.sum()),
                "same_day_or_reactive_firms": int(same_day.sum()),
                "false_positive_firms": int((flagged & ~early_tp & ~same_day).sum()),
                "early_precision": float(early_tp.sum() / flagged.sum()) if flagged.sum() else 0.0,
                "early_recall": float(early_tp.sum() / has_outcome.sum()) if has_outcome.sum() else 0.0,
                "median_early_lead_days": float(leads.median()) if not leads.empty else None,
                "mean_early_lead_days": float(leads.mean()) if not leads.empty else None,
                "description": first["description"],
            }
        )
    return pd.DataFrame(rows).sort_values(["source", "severity", "rule_id"])


def rule_registry_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule_id": rule.rule_id,
                "severity": rule.severity,
                "action": rule.action,
                "pattern": ">".join(rule.pattern),
                "window_days": rule.window_days,
                "mode": rule.mode,
                "include_outcome_events": rule.include_outcome_events,
                "description": rule.description,
            }
            for rule in FLAGGING_RULES
        ]
    )


def write_markdown(registry: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# SupTech Rule Engine",
        "",
        "This file documents the explicit supervisory flagging rules requested in `paper4_research_plan.md`. These rules are separate from the model-based watchlist and are best interpreted as auditable trigger definitions.",
        "",
        "## Rule Registry",
        "",
        "| Rule | Severity | Action | Pattern | Window days | Includes outcome events | Description |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in registry.itertuples(index=False):
        lines.append(
            f"| `{row.rule_id}` | {row.severity} | {row.action} | `{row.pattern}` | {row.window_days} | {row.include_outcome_events} | {str(row.description).replace('|', '\\|')} |"
        )
    lines.extend(
        [
            "",
            "## Rule Output Summary",
            "",
            "| Source | Rule | Flagged firms | Early TP | Same-day/reactive | Early precision | Early recall | Median early lead days |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        median = "" if pd.isna(row.median_early_lead_days) else f"{row.median_early_lead_days:.1f}"
        lines.append(
            f"| {row.source} | `{row.rule_id}` | {row.flagged_firms} | {row.early_true_positive_firms} | {row.same_day_or_reactive_firms} | {row.early_precision:.4f} | {row.early_recall:.4f} | {median} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `CRITICAL` rules are immediate-review triggers and may fire on the same date as an observed formal distress event.",
            "- `HIGH` and `MEDIUM` rules exclude outcome-like event subtypes when measuring early-warning behavior.",
            "- These rules remain interpretable but noisy. The main v1 paper should keep the ranked model watchlist as the headline SupTech artifact.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    flags = []
    for source in ("DART", "SEC"):
        events, outcomes = read_source(processed_dir, source)
        flags.append(build_flags(events, outcomes, source))
    flags_df = pd.concat(flags, ignore_index=True)
    summary_df = summarize_flags(flags_df)
    registry_df = rule_registry_frame()

    registry_df.to_csv(results_dir / "flagging_rule_registry.csv", index=False, encoding="utf-8-sig")
    flags_df.to_csv(results_dir / "flagging_rule_engine_flags.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(results_dir / "flagging_rule_engine_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "rules": registry_df.to_dict(orient="records"),
        "summary": summary_df.to_dict(orient="records"),
        "guardrail": "Rule-engine outputs are interpretable triggers. Use flagging_performance.md for comparison with model-based watchlists.",
    }
    (results_dir / "flagging_rule_engine_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(registry_df, summary_df, results_dir / "flagging_rule_engine_summary.md")
    print(json.dumps({"rules": len(registry_df), "flag_rows": len(flags_df), "summary_rows": len(summary_df)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
