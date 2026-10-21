from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TAXONOMY_ROWS = [
    {
        "taxonomy_code": "T01",
        "taxonomy_label": "insolvency_or_rehabilitation",
        "severity": 5,
        "dart_examples": "rehabilitation filing; dissolution",
        "sec_examples": "8-K Item 1.03 Bankruptcy or Receivership",
        "edinet_examples": "civil rehabilitation filing; bankruptcy-like proceeding",
    },
    {
        "taxonomy_code": "T02",
        "taxonomy_label": "delisting_or_trading_suspension",
        "severity": 5,
        "dart_examples": "delisting reason; trading suspension",
        "sec_examples": "8-K Item 3.01 delisting notice",
        "edinet_examples": "delisting; trading suspension",
    },
    {
        "taxonomy_code": "T03",
        "taxonomy_label": "debt_default_or_acceleration",
        "severity": 5,
        "dart_examples": "default occurrence",
        "sec_examples": "8-K Item 2.04 debt acceleration or trigger event",
        "edinet_examples": "loss of benefit of time; debt default",
    },
    {
        "taxonomy_code": "T04",
        "taxonomy_label": "audit_opinion_or_going_concern",
        "severity": 4,
        "dart_examples": "modified audit opinion; going-concern note",
        "sec_examples": "10-K going-concern disclosure",
        "edinet_examples": "going-concern note",
    },
    {
        "taxonomy_code": "T05",
        "taxonomy_label": "capital_structure_stress",
        "severity": 3,
        "dart_examples": "paid-in capital increase; capital reduction; convertible bond; asset transfer",
        "sec_examples": "8-K Item 1.01, 2.01, 3.02, 3.03",
        "edinet_examples": "capital reduction; net asset decline; financing stress",
    },
    {
        "taxonomy_code": "T06",
        "taxonomy_label": "governance_or_control_change",
        "severity": 2,
        "dart_examples": "merger; split; lawsuit",
        "sec_examples": "8-K Item 5.02 or 5.03",
        "edinet_examples": "director/officer change; governance change",
    },
    {
        "taxonomy_code": "T99",
        "taxonomy_label": "other_event",
        "severity": 1,
        "dart_examples": "other major event report",
        "sec_examples": "unmapped 8-K items or filings without parsed item",
        "edinet_examples": "other temporary report",
    },
]


SEC_ITEM_DESCRIPTIONS = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation",
    "3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "5.02": "Departure or Election of Directors or Certain Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "NO_ITEM": "8-K filing without parsed item in the local file",
}


def read_events(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dart = pd.read_parquet(processed_dir / "events_dart.parquet")
    sec = pd.read_parquet(processed_dir / "events_sec.parquet")
    dart["event_date"] = pd.to_datetime(dart["event_date"], errors="coerce")
    sec["event_date"] = pd.to_datetime(sec["event_date"], errors="coerce")
    return dart, sec


def taxonomy_frame() -> pd.DataFrame:
    return pd.DataFrame(TAXONOMY_ROWS)


def build_aligned_events(dart: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    dart_view = pd.DataFrame(
        {
            "source": "DART",
            "firm_id": dart["firm_id"],
            "firm_name": dart["corp_name"],
            "security_id": dart["security_id"],
            "event_date": dart["event_date"],
            "taxonomy_code": dart["taxonomy_code"],
            "taxonomy_label": dart["taxonomy_label"],
            "event_subtype": dart["event_subtype"],
            "raw_event": dart["report_nm"],
            "normalized_event": dart["clean_event_title"],
            "source_code": dart["rcept_no"],
        }
    )
    sec_view = pd.DataFrame(
        {
            "source": "SEC",
            "firm_id": sec["firm_id"],
            "firm_name": sec["company_name"],
            "security_id": sec["cik"].astype(str),
            "event_date": sec["event_date"],
            "taxonomy_code": sec["taxonomy_code"],
            "taxonomy_label": sec["taxonomy_label"],
            "event_subtype": sec["event_subtype"],
            "raw_event": "8-K Item " + sec["item"].astype(str),
            "normalized_event": sec["item"].astype(str).map(SEC_ITEM_DESCRIPTIONS).fillna(sec["item"].astype(str)),
            "source_code": sec["item"].astype(str),
        }
    )
    return pd.concat([dart_view, sec_view], ignore_index=True).sort_values(["source", "firm_id", "event_date"])


def event_dictionary(aligned: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        aligned.groupby(["source", "taxonomy_code", "taxonomy_label", "event_subtype", "normalized_event"], dropna=False)
        .agg(
            event_count=("normalized_event", "size"),
            firm_count=("firm_id", "nunique"),
            first_date=("event_date", "min"),
            last_date=("event_date", "max"),
        )
        .reset_index()
        .sort_values(["source", "taxonomy_code", "event_count"], ascending=[True, True, False])
    )
    grouped["first_date"] = grouped["first_date"].dt.date.astype(str)
    grouped["last_date"] = grouped["last_date"].dt.date.astype(str)
    return grouped


def build_alignment_candidates(dictionary: pd.DataFrame, target_rows: int = 100) -> pd.DataFrame:
    dart = dictionary[dictionary["source"].eq("DART")].copy()
    sec = dictionary[dictionary["source"].eq("SEC")].copy()
    rows: list[dict] = []

    for code in sorted(set(dart["taxonomy_code"]) | set(sec["taxonomy_code"])):
        d_part = dart[dart["taxonomy_code"].eq(code)].head(12)
        s_part = sec[sec["taxonomy_code"].eq(code)].head(8)
        for d_row in d_part.itertuples(index=False):
            for s_row in s_part.itertuples(index=False):
                rows.append(
                    {
                        "candidate_rank": len(rows) + 1,
                        "proposed_label": "MATCH",
                        "taxonomy_code": code,
                        "dart_event": d_row.normalized_event,
                        "dart_subtype": d_row.event_subtype,
                        "dart_event_count": int(d_row.event_count),
                        "sec_event": s_row.normalized_event,
                        "sec_subtype": s_row.event_subtype,
                        "sec_event_count": int(s_row.event_count),
                        "rationale": "Same supervisory taxonomy code in deterministic seed mapping.",
                        "human_label": "",
                        "human_notes": "",
                    }
                )
                if len(rows) >= target_rows:
                    return pd.DataFrame(rows)

    if len(rows) < target_rows:
        for d_row in dart.head(30).itertuples(index=False):
            for s_row in sec.head(30).itertuples(index=False):
                if d_row.taxonomy_code == s_row.taxonomy_code:
                    continue
                rows.append(
                    {
                        "candidate_rank": len(rows) + 1,
                        "proposed_label": "NO_MATCH",
                        "taxonomy_code": f"{d_row.taxonomy_code}|{s_row.taxonomy_code}",
                        "dart_event": d_row.normalized_event,
                        "dart_subtype": d_row.event_subtype,
                        "dart_event_count": int(d_row.event_count),
                        "sec_event": s_row.normalized_event,
                        "sec_subtype": s_row.event_subtype,
                        "sec_event_count": int(s_row.event_count),
                        "rationale": "Different taxonomy codes; included as negative annotation seed.",
                        "human_label": "",
                        "human_notes": "",
                    }
                )
                if len(rows) >= target_rows:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def coverage_summary(aligned: pd.DataFrame, dictionary: pd.DataFrame, seed: pd.DataFrame) -> dict:
    by_source_code = (
        aligned.groupby(["source", "taxonomy_code"], dropna=False)
        .agg(rows=("taxonomy_code", "size"), firms=("firm_id", "nunique"))
        .reset_index()
    )
    overlap_codes = sorted(
        set(by_source_code[by_source_code["source"].eq("DART")]["taxonomy_code"])
        & set(by_source_code[by_source_code["source"].eq("SEC")]["taxonomy_code"])
    )
    non_other = aligned[~aligned["taxonomy_code"].eq("T99")]
    return {
        "aligned_event_rows": int(len(aligned)),
        "sources": {
            source: {
                "rows": int(part.shape[0]),
                "firms": int(part["firm_id"].nunique()),
                "first_date": str(part["event_date"].min().date()) if not part.empty else None,
                "last_date": str(part["event_date"].max().date()) if not part.empty else None,
                "non_other_share": float((~part["taxonomy_code"].eq("T99")).mean()) if not part.empty else None,
            }
            for source, part in aligned.groupby("source")
        },
        "taxonomy_overlap_codes": overlap_codes,
        "taxonomy_overlap_count": int(len(overlap_codes)),
        "dictionary_rows": int(len(dictionary)),
        "manual_candidate_rows": int(len(seed)),
        "non_other_rows": int(len(non_other)),
        "note": "Precision@1 and MRR require human labels in manual_alignment_seed_100.csv; current outputs are deterministic candidate mappings, not a validated gold standard.",
    }


def write_markdown(summary: dict, taxonomy: pd.DataFrame, by_code: pd.DataFrame, path: Path) -> None:
    lines = ["# DART-SEC Alignment Summary", ""]
    lines.append("This file documents the deterministic seed alignment built from the existing processed DART and SEC event layers.")
    lines.append("It is not a human-validated Precision@1/MRR result yet. Human labels should be added to `manual_alignment_seed_100.csv` before claiming alignment accuracy.")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Aligned event rows: {summary['aligned_event_rows']:,}")
    lines.append(f"- Shared DART/SEC taxonomy codes: {', '.join(summary['taxonomy_overlap_codes'])}")
    lines.append(f"- Manual annotation candidate rows: {summary['manual_candidate_rows']}")
    lines.append("")
    lines.append("| Source | Rows | Firms | Date min | Date max | Non-other share |")
    lines.append("|---|---:|---:|---|---|---:|")
    for source, item in summary["sources"].items():
        non_other = item["non_other_share"]
        lines.append(
            f"| {source} | {item['rows']:,} | {item['firms']:,} | {item['first_date']} | {item['last_date']} | {non_other:.3f} |"
        )
    lines.append("")
    lines.append("## Taxonomy")
    lines.append("")
    lines.append("| Code | Label | Severity | DART examples | SEC examples | EDINET examples |")
    lines.append("|---|---|---:|---|---|---|")
    for row in taxonomy.itertuples(index=False):
        lines.append(
            f"| {row.taxonomy_code} | `{row.taxonomy_label}` | {row.severity} | {row.dart_examples} | {row.sec_examples} | {row.edinet_examples} |"
        )
    lines.append("")
    lines.append("## Event Counts by Source and Code")
    lines.append("")
    lines.append("| Source | Code | Rows | Firms |")
    lines.append("|---|---|---:|---:|")
    for row in by_code.itertuples(index=False):
        lines.append(f"| {row.source} | {row.taxonomy_code} | {row.rows:,} | {row.firms:,} |")
    lines.append("")
    lines.append("## Next Alignment Step")
    lines.append("")
    lines.append("Fill `human_label` in `manual_alignment_seed_100.csv` with `MATCH`, `PARTIAL`, or `NO_MATCH`, then run an evaluator to report Precision@1 and MRR.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(processed_dir: Path, results_dir: Path, target_rows: int) -> None:
    dart, sec = read_events(processed_dir)
    taxonomy = taxonomy_frame()
    aligned = build_aligned_events(dart, sec)
    dictionary = event_dictionary(aligned)
    seed = build_alignment_candidates(dictionary, target_rows=target_rows)
    by_code = (
        aligned.groupby(["source", "taxonomy_code"], dropna=False)
        .agg(rows=("taxonomy_code", "size"), firms=("firm_id", "nunique"))
        .reset_index()
        .sort_values(["source", "taxonomy_code"])
    )
    summary = coverage_summary(aligned, dictionary, seed)

    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    taxonomy.to_csv(processed_dir / "regulatory_event_taxonomy.csv", index=False, encoding="utf-8-sig")
    aligned.to_parquet(processed_dir / "events_aligned.parquet", index=False)
    aligned.to_csv(processed_dir / "events_aligned.csv", index=False, encoding="utf-8-sig")
    dictionary.to_csv(processed_dir / "event_dictionary.csv", index=False, encoding="utf-8-sig")
    seed.to_csv(processed_dir / "manual_alignment_seed_100.csv", index=False, encoding="utf-8-sig")
    (results_dir / "alignment_scores.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, taxonomy, by_code, results_dir / "alignment_summary.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=str(Path("Paper 4") / "data" / "processed"))
    parser.add_argument("--results-dir", default=str(Path("Paper 4") / "results"))
    parser.add_argument("--target-rows", type=int, default=100)
    args = parser.parse_args()
    write_outputs(Path(args.processed_dir), Path(args.results_dir), args.target_rows)


if __name__ == "__main__":
    main()
