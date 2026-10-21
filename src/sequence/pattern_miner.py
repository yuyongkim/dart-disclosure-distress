from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


OUTCOME_SUBTYPES = {
    "DART": {"rehabilitation", "dissolution", "default", "listing_or_trading_status"},
    "SEC": {"bankruptcy_or_receivership", "delisting_notice", "debt_acceleration_trigger"},
}


def read_source(processed_dir: Path, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_l = source.lower()
    events = pd.read_parquet(processed_dir / f"events_{source_l}.parquet")
    outcomes = pd.read_parquet(processed_dir / f"outcomes_{source_l}.parquet")
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    outcomes["outcome_date"] = pd.to_datetime(outcomes["outcome_date"], errors="coerce")
    return events, outcomes


def compressed_codes(events: pd.DataFrame) -> list[tuple[str, pd.Timestamp]]:
    rows: list[tuple[str, pd.Timestamp]] = []
    last_code: str | None = None
    for row in events.sort_values("event_date").itertuples(index=False):
        code = str(row.taxonomy_code)
        if code == last_code:
            continue
        rows.append((code, pd.Timestamp(row.event_date)))
        last_code = code
    return rows


def ngrams(sequence: list[tuple[str, pd.Timestamp]], max_len: int) -> dict[tuple[str, ...], pd.Timestamp]:
    first_seen: dict[tuple[str, ...], pd.Timestamp] = {}
    codes = [code for code, _ in sequence]
    dates = [date for _, date in sequence]
    for length in range(1, max_len + 1):
        for i in range(0, len(codes) - length + 1):
            pattern = tuple(codes[i : i + length])
            if all(code == "T99" for code in pattern):
                continue
            first_seen.setdefault(pattern, dates[i + length - 1])
    return first_seen


def build_firm_patterns(events: pd.DataFrame, outcomes: pd.DataFrame, source: str, max_len: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    outcome_by_firm = {firm_id: g.sort_values("outcome_date") for firm_id, g in outcomes.groupby("firm_id")}
    source_outcomes = OUTCOME_SUBTYPES[source]

    for firm_id, group in events.groupby("firm_id"):
        group = group.sort_values("event_date").copy()
        out = outcome_by_firm.get(firm_id)
        has_outcome = out is not None and not out.empty
        outcome_date = pd.Timestamp(out["outcome_date"].iloc[0]) if has_outcome else pd.NaT
        outcome_label = str(out["outcome_label"].iloc[0]) if has_outcome else ""
        name_col = "corp_name" if source == "DART" else "company_name"
        firm_name = str(group[name_col].dropna().iloc[0]) if name_col in group.columns and group[name_col].notna().any() else ""

        if has_outcome:
            visible = group[group["event_date"].lt(outcome_date)].copy()
        else:
            visible = group.copy()
        visible = visible[~visible["event_subtype"].isin(source_outcomes)].copy()
        sequence = compressed_codes(visible)
        pattern_dates = ngrams(sequence, max_len=max_len)

        for pattern, pattern_date in pattern_dates.items():
            lead_days = int((outcome_date - pattern_date).days) if has_outcome and pd.notna(pattern_date) else None
            rows.append(
                {
                    "source": source,
                    "firm_id": firm_id,
                    "firm_name": firm_name,
                    "pattern": ">".join(pattern),
                    "pattern_length": len(pattern),
                    "first_pattern_date": pattern_date,
                    "has_outcome": bool(has_outcome),
                    "outcome_date": outcome_date,
                    "outcome_label": outcome_label,
                    "lead_days": lead_days,
                }
            )

    metadata = {
        "source": source,
        "firms": int(events["firm_id"].nunique()),
        "outcome_firms": int(outcomes["firm_id"].nunique()),
        "base_outcome_rate": float(outcomes["firm_id"].nunique() / events["firm_id"].nunique()) if events["firm_id"].nunique() else 0.0,
    }
    return rows, metadata


def summarize_patterns(firm_patterns: pd.DataFrame, metadata_by_source: dict[str, dict], min_support_firms: int) -> pd.DataFrame:
    rows: list[dict] = []
    for (source, pattern), group in firm_patterns.groupby(["source", "pattern"]):
        source_meta = metadata_by_source[source]
        support = int(group["firm_id"].nunique())
        if support < min_support_firms:
            continue
        pos = group[group["has_outcome"]].copy()
        outcome_support = int(pos["firm_id"].nunique())
        confidence = outcome_support / support if support else 0.0
        base_rate = source_meta["base_outcome_rate"]
        lift = confidence / base_rate if base_rate else None
        lead = pos["lead_days"].dropna()
        rows.append(
            {
                "source": source,
                "pattern": pattern,
                "pattern_length": int(str(pattern).count(">") + 1),
                "support_firms": support,
                "support_share": support / source_meta["firms"] if source_meta["firms"] else 0.0,
                "outcome_firms_with_pattern": outcome_support,
                "confidence": confidence,
                "base_outcome_rate": base_rate,
                "lift": lift,
                "recall_among_outcome_firms": outcome_support / source_meta["outcome_firms"] if source_meta["outcome_firms"] else 0.0,
                "median_lead_days": float(lead.median()) if not lead.empty else None,
                "mean_lead_days": float(lead.mean()) if not lead.empty else None,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["source", "lift", "confidence", "support_firms"], ascending=[True, False, False, False])


def write_markdown(patterns: pd.DataFrame, metadata: dict[str, dict], path: Path) -> None:
    lines = ["# Sequence Pattern Mining Results", ""]
    lines.append("Patterns are contiguous taxonomy-code n-grams after compressing consecutive duplicate codes. Outcome-like event subtypes are excluded from the pre-outcome sequence to reduce look-ahead leakage.")
    lines.append("")
    lines.append("## Source Coverage")
    lines.append("")
    lines.append("| Source | Firms | Outcome firms | Base outcome rate |")
    lines.append("|---|---:|---:|---:|")
    for source, item in metadata.items():
        lines.append(f"| {source} | {item['firms']:,} | {item['outcome_firms']:,} | {item['base_outcome_rate']:.4f} |")
    lines.append("")
    if patterns.empty:
        lines.append("No patterns met the support threshold.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for source in sorted(patterns["source"].unique()):
        lines.append(f"## {source} Top Patterns by Lift")
        lines.append("")
        lines.append("| Pattern | Support firms | Outcome firms | Confidence | Lift | Recall | Median lead days |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        view = patterns[patterns["source"].eq(source)].head(20)
        for row in view.itertuples(index=False):
            median_lead = "" if pd.isna(row.median_lead_days) else f"{row.median_lead_days:.1f}"
            lift = "" if pd.isna(row.lift) else f"{row.lift:.2f}"
            lines.append(
                f"| `{row.pattern}` | {row.support_firms} | {row.outcome_firms_with_pattern} | {row.confidence:.4f} | {lift} | {row.recall_among_outcome_firms:.4f} | {median_lead} |"
            )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("These patterns are screening candidates, not final causal claims. High lift with low support should be treated as a case-discovery signal; broad high-support patterns are more suitable for watchlist features.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=str(Path("Paper 4") / "data" / "processed"))
    parser.add_argument("--results-dir", default=str(Path("Paper 4") / "results"))
    parser.add_argument("--max-len", type=int, default=3)
    parser.add_argument("--min-support-firms", type=int, default=8)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    metadata: dict[str, dict] = {}
    for source in ("DART", "SEC"):
        events, outcomes = read_source(processed_dir, source)
        rows, meta = build_firm_patterns(events, outcomes, source, max_len=args.max_len)
        all_rows.extend(rows)
        metadata[source] = meta

    firm_patterns = pd.DataFrame(all_rows)
    firm_patterns.to_csv(results_dir / "pattern_firm_occurrences.csv", index=False, encoding="utf-8-sig")
    patterns = summarize_patterns(firm_patterns, metadata, min_support_firms=args.min_support_firms)
    patterns.to_csv(results_dir / "pattern_rules.csv", index=False, encoding="utf-8-sig")
    (results_dir / "pattern_rules.json").write_text(
        json.dumps(patterns.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (results_dir / "pattern_mining_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(patterns, metadata, results_dir / "pattern_rules.md")
    print(json.dumps({"metadata": metadata, "pattern_rows": int(len(patterns))}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
