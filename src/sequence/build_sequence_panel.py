from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


OUTCOME_SUBTYPES_DART = {
    "rehabilitation",
    "dissolution",
    "default",
    "listing_or_trading_status",
}

OUTCOME_SUBTYPES_SEC = {
    "bankruptcy_or_receivership",
    "delisting_notice",
    "debt_acceleration_trigger",
}


@dataclass(frozen=True)
class RuleFlag:
    rule: str
    date: pd.Timestamp


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def rolling_pair_flag(events: pd.DataFrame, codes: Iterable[str], window_days: int, min_count: int, rule_name: str) -> RuleFlag | None:
    subset = events[events["taxonomy_code"].isin(set(codes))].sort_values("event_date")
    dates = list(pd.to_datetime(subset["event_date"], errors="coerce").dropna())
    for i, date in enumerate(dates):
        start = date - pd.Timedelta(days=window_days)
        count = sum(1 for d in dates[: i + 1] if start <= d <= date)
        if count >= min_count:
            return RuleFlag(rule_name, date)
    return None


def mixed_codes_flag(events: pd.DataFrame, code_a: str, code_b: str, window_days: int, rule_name: str) -> RuleFlag | None:
    subset = events[events["taxonomy_code"].isin({code_a, code_b})].sort_values("event_date")
    rows = [(r.taxonomy_code, pd.Timestamp(r.event_date)) for r in subset.itertuples(index=False) if pd.notna(r.event_date)]
    for i, (code, date) in enumerate(rows):
        start = date - pd.Timedelta(days=window_days)
        window_codes = {c for c, d in rows[: i + 1] if start <= d <= date}
        if code_a in window_codes and code_b in window_codes:
            return RuleFlag(rule_name, date)
    return None


def first_flag(events: pd.DataFrame) -> RuleFlag | None:
    rules = [
        rolling_pair_flag(events, ["T05"], 365, 2, "R1_repeated_T05_365d"),
        mixed_codes_flag(events, "T05", "T06", 180, "R2_T05_T06_180d"),
    ]
    rules = [r for r in rules if r is not None]
    if not rules:
        return None
    return sorted(rules, key=lambda r: r.date)[0]


def build_panel(events: pd.DataFrame, outcomes: pd.DataFrame, source: str) -> pd.DataFrame:
    events = events.copy()
    outcomes = outcomes.copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    outcome_date_col = "outcome_date"
    outcomes[outcome_date_col] = pd.to_datetime(outcomes[outcome_date_col], errors="coerce")

    name_col = "corp_name" if source == "DART" else "company_name"
    id_cols = ["firm_id"]
    rows = []
    outcome_by_firm = {k: g.sort_values(outcome_date_col) for k, g in outcomes.groupby("firm_id")}

    for firm_id, g in events.groupby("firm_id"):
        g = g.sort_values("event_date")
        out = outcome_by_firm.get(firm_id)
        has_outcome = out is not None and not out.empty
        first_outcome_date = out[outcome_date_col].iloc[0] if has_outcome else pd.NaT
        first_outcome_label = out["outcome_label"].iloc[0] if has_outcome else None

        if source == "DART":
            non_outcome_events = g[~g["event_subtype"].isin(OUTCOME_SUBTYPES_DART)].copy()
        else:
            non_outcome_events = g[~g["event_subtype"].isin(OUTCOME_SUBTYPES_SEC)].copy()

        flag = first_flag(non_outcome_events)
        flag_date = flag.date if flag else pd.NaT
        flag_rule = flag.rule if flag else None

        if has_outcome and pd.notna(flag_date):
            lead_days = int((first_outcome_date - flag_date).days)
            early_tp = lead_days > 0
        else:
            lead_days = None
            early_tp = False

        before_outcome = g[g["event_date"] < first_outcome_date] if has_outcome else g
        features: dict[str, int] = {}
        for window in (90, 180, 365):
            if has_outcome:
                w = before_outcome[before_outcome["event_date"] >= first_outcome_date - pd.Timedelta(days=window)]
            else:
                w = before_outcome
            features[f"events_pre_{window}d"] = int(len(w))
            for code in ("T01", "T02", "T03", "T05", "T06", "T99"):
                features[f"{code}_pre_{window}d"] = int((w["taxonomy_code"] == code).sum())

        rows.append(
            {
                "source": source,
                "firm_id": firm_id,
                "firm_name": str(g[name_col].dropna().iloc[0]) if name_col in g.columns and g[name_col].notna().any() else None,
                "n_events": int(len(g)),
                "first_event_date": g["event_date"].min(),
                "last_event_date": g["event_date"].max(),
                "taxonomy_sequence": " ".join(g["taxonomy_code"].astype(str)),
                "subtype_sequence": " | ".join(g["event_subtype"].astype(str)),
                "has_outcome": bool(has_outcome),
                "first_outcome_date": first_outcome_date,
                "first_outcome_label": first_outcome_label,
                "flagged": bool(flag),
                "flag_date": flag_date,
                "flag_rule": flag_rule,
                "lead_days": lead_days,
                "early_true_positive": bool(early_tp),
                **features,
            }
        )

    return pd.DataFrame(rows).sort_values(["source", "firm_id"])


def evaluate_at_horizon(panel: pd.DataFrame, horizon_days: int | None) -> dict:
    n = len(panel)
    outcome = panel["has_outcome"]
    flagged = panel["flagged"]
    lead = panel["lead_days"]
    if horizon_days is None:
        early_hit = panel["early_true_positive"]
    else:
        early_hit = panel["early_true_positive"] & lead.notna() & (lead <= horizon_days)
    tp = int(early_hit.sum())
    fp = int((flagged & ~early_hit).sum())
    fn = int((outcome & ~early_hit).sum())
    tn = int((~outcome & ~flagged).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / int(outcome.sum()) if int(outcome.sum()) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lead_values = panel.loc[early_hit, "lead_days"].dropna()
    return {
        "horizon_days": horizon_days if horizon_days is not None else "any_positive_lead",
        "firms": int(n),
        "outcome_firms": int(outcome.sum()),
        "flagged_firms": int(flagged.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "median_lead_days": float(lead_values.median()) if not lead_values.empty else None,
        "mean_lead_days": float(lead_values.mean()) if not lead_values.empty else None,
    }


def evaluate(panel: pd.DataFrame) -> dict:
    return {
        "any_positive_lead": evaluate_at_horizon(panel, None),
        "180d": evaluate_at_horizon(panel, 180),
        "365d": evaluate_at_horizon(panel, 365),
        "730d": evaluate_at_horizon(panel, 730),
    }


def write_frame(df: pd.DataFrame, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path_base.with_suffix(".parquet"), index=False)
    df.to_csv(path_base.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def write_metrics_md(metrics: dict, path: Path) -> None:
    lines = ["# Preliminary Flagging Metrics", ""]
    lines.append("Rules evaluated:")
    lines.append("")
    lines.append("- `R1_repeated_T05_365d`: at least two non-outcome `T05` events within 365 days.")
    lines.append("- `R2_T05_T06_180d`: at least one non-outcome `T05` and one non-outcome `T06` within 180 days.")
    lines.append("")
    lines.append("| Source | Horizon | Firms | Outcome firms | Flagged | TP | FP | FN | Precision | Recall | F1 | Median lead days |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for source, source_metrics in metrics.items():
        for horizon, m in source_metrics.items():
            h = m["horizon_days"]
            lines.append(
                f"| {source} | {h} | {m['firms']} | {m['outcome_firms']} | {m['flagged_firms']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']} | {m['recall']} | {m['f1']} | {m['median_lead_days']} |"
            )
    lines.append("")
    lines.append("Interpretation: these are deliberately simple baseline rules. Low precision is expected if broad restructuring patterns are used as supervisory watchlist triggers rather than hard distress predictions.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=str(Path("Paper 4") / "data" / "processed"))
    parser.add_argument("--results-dir", default=str(Path("Paper 4") / "results"))
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)

    dart_panel = build_panel(
        read_frame(processed / "events_dart.parquet"),
        read_frame(processed / "outcomes_dart.parquet"),
        "DART",
    )
    sec_panel = build_panel(
        read_frame(processed / "events_sec.parquet"),
        read_frame(processed / "outcomes_sec.parquet"),
        "SEC",
    )

    write_frame(dart_panel, processed / "sequence_panel_dart")
    write_frame(sec_panel, processed / "sequence_panel_sec")

    metrics = {
        "DART": evaluate(dart_panel),
        "SEC": evaluate(sec_panel),
    }
    (results / "preliminary_flagging_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_metrics_md(metrics, results / "preliminary_flagging_metrics.md")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
