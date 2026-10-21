from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_examples(events: pd.DataFrame, window_days: int, max_events: int) -> pd.DataFrame:
    events = events.copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    rows: list[dict[str, object]] = []
    for firm_id, firm_events in events.groupby("firm_id"):
        firm_events = firm_events.sort_values(["event_date", "item"]).copy()
        item103 = firm_events[firm_events["item"].astype(str).eq("1.03")]
        if item103.empty:
            continue
        first_item103 = item103.iloc[0]
        first_date = pd.Timestamp(first_item103["event_date"])
        pre = firm_events[
            (firm_events["event_date"] < first_date)
            & (firm_events["event_date"] >= first_date - pd.Timedelta(days=window_days))
        ].copy()
        last_pre = pre.tail(max_events)
        salient = pre[pre["taxonomy_code"].astype(str).ne("T99")].copy()
        last_salient = salient.tail(max_events)
        sequence = " > ".join(last_pre["taxonomy_code"].astype(str))
        item_sequence = " > ".join(last_pre["item"].astype(str))
        event_trace = " | ".join(
            f"{pd.Timestamp(r.event_date).date()}:{r.taxonomy_code}/{r.item}"
            for r in last_pre.itertuples(index=False)
        )
        salient_sequence = " > ".join(last_salient["taxonomy_code"].astype(str))
        salient_item_sequence = " > ".join(last_salient["item"].astype(str))
        salient_trace = " | ".join(
            f"{pd.Timestamp(r.event_date).date()}:{r.taxonomy_code}/{r.item}"
            for r in last_salient.itertuples(index=False)
        )
        rows.append(
            {
                "firm_id": firm_id,
                "case_label": str(firm_events["seed_case_label"].dropna().iloc[0])
                if "seed_case_label" in firm_events.columns and firm_events["seed_case_label"].notna().any()
                else "",
                "company_name": str(firm_events["company_name"].dropna().iloc[0])
                if "company_name" in firm_events.columns and firm_events["company_name"].notna().any()
                else "",
                "first_item103_date": first_date.date().isoformat(),
                "first_item103_accession": first_item103.get("accession_number", ""),
                "pre_event_count_window": int(len(pre)),
                "pre_salient_event_count_window": int(len(salient)),
                "pre_taxonomy_sequence_tail": sequence,
                "pre_item_sequence_tail": item_sequence,
                "pre_event_trace_tail": event_trace,
                "pre_salient_taxonomy_sequence_tail": salient_sequence,
                "pre_salient_item_sequence_tail": salient_item_sequence,
                "pre_salient_event_trace_tail": salient_trace,
                "total_item103_rows": int(len(item103)),
            }
        )
    return pd.DataFrame(rows).sort_values(["first_item103_date", "case_label", "firm_id"])


def write_markdown(examples: pd.DataFrame, path: Path, window_days: int, title: str, description: str, interpretation: str) -> None:
    lines = [
        f"# {title}",
        "",
        description.format(window_days=window_days),
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Case firms | {len(examples)} |",
        f"| Median pre-event count | {float(examples['pre_event_count_window'].median()) if not examples.empty else 0} |",
        f"| Median salient pre-event count | {float(examples['pre_salient_event_count_window'].median()) if not examples.empty else 0} |",
        f"| Max pre-event count | {int(examples['pre_event_count_window'].max()) if not examples.empty else 0} |",
        "",
        "## Case Trace",
        "",
        "| Case | First Item 1.03 | Salient pre-events | Salient taxonomy tail | Salient item tail |",
        "|---|---|---:|---|---|",
    ]
    for row in examples.itertuples(index=False):
        seq = str(row.pre_salient_taxonomy_sequence_tail)
        if len(seq) > 90:
            seq = seq[-90:]
        items = str(row.pre_salient_item_sequence_tail)
        if len(items) > 90:
            items = items[-90:]
        lines.append(
            f"| {row.case_label or row.company_name} | {row.first_item103_date} | {row.pre_salient_event_count_window} | `{seq}` | `{items}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- {interpretation}",
            "- The table verifies that SEC Item 1.03 events can be embedded in the same T-code sequence layer as DART events.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-path", default=str(Path("Paper 4") / "data" / "processed" / "sec_bankruptcy_case_events.parquet"))
    parser.add_argument("--results-dir", default=str(Path("Paper 4") / "results"))
    parser.add_argument("--output-prefix", default="sec_bankruptcy_case_sequences")
    parser.add_argument("--title", default="SEC Bankruptcy Case Sequences")
    parser.add_argument(
        "--description",
        default="This file lists selected SEC Item 1.03 bankruptcy/receivership cases and the trailing regulatory-event sequence within {window_days} days before the first Item 1.03 filing.",
    )
    parser.add_argument(
        "--interpretation",
        default="These are selected U.S. bankruptcy cases, so they are useful for concrete sequence examples rather than population-level performance claims.",
    )
    parser.add_argument("--window-days", type=int, default=730)
    parser.add_argument("--max-events", type=int, default=12)
    args = parser.parse_args()

    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(args.events_path)
    examples = build_examples(events, args.window_days, args.max_events)
    examples.to_csv(results / f"{args.output_prefix}.csv", index=False, encoding="utf-8-sig")
    write_markdown(examples, results / f"{args.output_prefix}.md", args.window_days, args.title, args.description, args.interpretation)
    print(
        {
            "case_firms": int(len(examples)),
            "median_pre_event_count": float(examples["pre_event_count_window"].median()) if not examples.empty else 0,
            "median_salient_pre_event_count": float(examples["pre_salient_event_count_window"].median()) if not examples.empty else 0,
        }
    )


if __name__ == "__main__":
    main()
