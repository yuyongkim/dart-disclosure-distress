from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SRC = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC / "prepare"))
sys.path.append(str(SRC / "sequence"))

from build_phase1_events import sec_taxonomy  # noqa: E402
from build_sequence_panel import build_panel, evaluate, write_frame  # noqa: E402


TARGET_OUTCOME_SUBTYPES = {
    "bankruptcy_or_receivership": "BANKRUPTCY_OR_RECEIVERSHIP",
    "delisting_notice": "DELISTING_NOTICE",
    "debt_acceleration_trigger": "DEFAULT_OR_ACCELERATION",
}


def split_items(value: object) -> list[str]:
    if pd.isna(value):
        return ["NO_ITEM"]
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return items or ["NO_ITEM"]


def build_events(raw: pd.DataFrame, start_year: int, end_year: int, source_name: str) -> pd.DataFrame:
    raw = raw.copy()
    raw["event_date"] = pd.to_datetime(raw["filing_date"], errors="coerce")
    raw = raw[raw["event_date"].dt.year.between(start_year, end_year, inclusive="both")].copy()

    rows: list[dict[str, object]] = []
    for row in raw.itertuples(index=False):
        cik = str(getattr(row, "cik", "")).split(".")[0].zfill(10)
        for item in split_items(getattr(row, "items", "")):
            code, label, subtype, is_outcome = sec_taxonomy(item)
            rows.append(
                {
                    "source": source_name,
                    "firm_id": cik,
                    "cik": cik,
                    "company_name": getattr(row, "entity_name", ""),
                    "tickers": getattr(row, "tickers", ""),
                    "sic": getattr(row, "sic", ""),
                    "event_date": getattr(row, "event_date"),
                    "filing_date": getattr(row, "filing_date", ""),
                    "report_date": getattr(row, "report_date", ""),
                    "form": getattr(row, "form", ""),
                    "accession_number": getattr(row, "accession_number", ""),
                    "primary_document": getattr(row, "primary_document", ""),
                    "seed_case_label": getattr(row, "seed_case_label", ""),
                    "seed_expected_event_year": getattr(row, "seed_expected_event_year", ""),
                    "taxonomy_code": code,
                    "taxonomy_label": label,
                    "event_subtype": subtype,
                    "item": item,
                    "is_outcome_item": bool(is_outcome),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "firm_id",
                "cik",
                "company_name",
                "event_date",
                "taxonomy_code",
                "taxonomy_label",
                "event_subtype",
                "item",
                "is_outcome_item",
            ]
        )
    return pd.DataFrame(rows).sort_values(["firm_id", "event_date", "item"])


def build_outcomes(events: pd.DataFrame) -> pd.DataFrame:
    outcomes = events[events["event_subtype"].isin(TARGET_OUTCOME_SUBTYPES)].copy()
    outcomes["outcome_label"] = outcomes["event_subtype"].map(TARGET_OUTCOME_SUBTYPES)
    outcomes = outcomes.rename(columns={"event_date": "outcome_date"})
    return outcomes[
        [
            "source",
            "firm_id",
            "cik",
            "company_name",
            "sic",
            "outcome_date",
            "outcome_label",
            "taxonomy_code",
            "event_subtype",
            "item",
            "accession_number",
        ]
    ].sort_values(["firm_id", "outcome_date", "item"])


def summarize(events: pd.DataFrame, outcomes: pd.DataFrame, panel: pd.DataFrame, metrics: dict) -> dict:
    item_counts = events["item"].value_counts().head(25).to_dict() if not events.empty else {}
    taxonomy_counts = events["taxonomy_code"].value_counts().to_dict() if not events.empty else {}
    outcome_counts = outcomes["outcome_label"].value_counts().to_dict() if not outcomes.empty else {}
    sequence_lengths = panel["n_events"] if "n_events" in panel else pd.Series(dtype=float)
    return {
        "event_rows": int(len(events)),
        "event_firms": int(events["firm_id"].nunique()) if not events.empty else 0,
        "filing_rows_represented": int(events["accession_number"].nunique()) if "accession_number" in events else 0,
        "outcome_rows": int(len(outcomes)),
        "outcome_firms": int(outcomes["firm_id"].nunique()) if not outcomes.empty else 0,
        "taxonomy_counts": {str(k): int(v) for k, v in taxonomy_counts.items()},
        "outcome_counts": {str(k): int(v) for k, v in outcome_counts.items()},
        "top_item_counts": {str(k): int(v) for k, v in item_counts.items()},
        "sequence_panel_firms": int(len(panel)),
        "median_sequence_length": float(sequence_lengths.median()) if not sequence_lengths.empty else None,
        "max_sequence_length": int(sequence_lengths.max()) if not sequence_lengths.empty else 0,
        "flagging_metrics": metrics,
    }


def write_markdown(summary: dict, path: Path, title: str, description: str) -> None:
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "## Coverage",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Item-level event rows | {summary['event_rows']} |",
        f"| SEC firms with events | {summary['event_firms']} |",
        f"| 8-K filings represented | {summary['filing_rows_represented']} |",
        f"| Outcome rows, Items 1.03/2.04/3.01 | {summary['outcome_rows']} |",
        f"| Outcome firms | {summary['outcome_firms']} |",
        f"| Sequence-panel firms | {summary['sequence_panel_firms']} |",
        f"| Median sequence length | {summary['median_sequence_length']} |",
        f"| Max sequence length | {summary['max_sequence_length']} |",
        "",
        "## Taxonomy Counts",
        "",
        "| Code | Rows |",
        "|---|---:|",
    ]
    for code, count in summary["taxonomy_counts"].items():
        lines.append(f"| `{code}` | {count} |")
    lines.extend(["", "## Outcome Counts", "", "| Outcome | Rows |", "|---|---:|"])
    for label, count in summary["outcome_counts"].items():
        lines.append(f"| `{label}` | {count} |")
    lines.extend(["", "## Flagging Metrics", "", "| Horizon | Firms | Outcome firms | Flagged | TP | Precision | Recall | Median lead days |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for _, metric in summary["flagging_metrics"].items():
        lines.append(
            f"| {metric['horizon_days']} | {metric['firms']} | {metric['outcome_firms']} | {metric['flagged_firms']} | {metric['tp']} | {metric['precision']} | {metric['recall']} | {metric['median_lead_days']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if summary["outcome_counts"].get("BANKRUPTCY_OR_RECEIVERSHIP", 0):
        lines.append("- Item 1.03 bankruptcy/receivership rows are present, so this artifact can support U.S. bankruptcy sequence case examples.")
    else:
        lines.append("- Item 1.03 remains absent in this collected metadata, so this artifact does not support a bankruptcy-centered U.S. validation claim.")
    lines.extend(
        [
            "- Use this as a v2 support artifact unless it is explicitly merged into a population-representative U.S. validation design.",
            "- Items 2.04 and 3.01 provide debt-acceleration and listing-status outcome proxies; describe them separately from bankruptcy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", default=str(Path("Paper 4") / "data" / "raw" / "sec_expanded" / "sec_expanded_8k_filings.csv"))
    parser.add_argument("--processed-dir", default=str(Path("Paper 4") / "data" / "processed"))
    parser.add_argument("--results-dir", default=str(Path("Paper 4") / "results"))
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--prefix", default="sec_expanded")
    parser.add_argument("--source-name", default="SEC_EXPANDED")
    parser.add_argument("--title", default="SEC Expanded Sequence Summary")
    parser.add_argument(
        "--description",
        default="This file converts newly collected SEC submissions metadata into the same taxonomy-sequence structure used by Paper 4. It is a v2 support artifact, not a v1 standalone U.S. prediction claim.",
    )
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.raw_path, low_memory=False)
    events = build_events(raw, args.start_year, args.end_year, args.source_name)
    outcomes = build_outcomes(events)
    panel = build_panel(events, outcomes, "SEC")
    metrics = evaluate(panel)
    summary = summarize(events, outcomes, panel, metrics)

    write_frame(events, processed / f"{args.prefix}_events")
    write_frame(outcomes, processed / f"{args.prefix}_outcomes")
    write_frame(panel, processed / f"{args.prefix}_sequence_panel")
    (results / f"{args.prefix}_sequence_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, results / f"{args.prefix}_sequence_summary.md", args.title, args.description)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
