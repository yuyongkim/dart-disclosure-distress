from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("Paper 4")

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

TARGET_CODE_ORDER = ["T01", "T02", "T03", "T05", "T06", "T99"]
MATCH_ROW_QUOTAS = {
    "T01": 12,
    "T02": 8,
    "T03": 8,
    "T05": 24,
    "T06": 18,
    "T99": 8,
}
LABEL_COLUMNS = [
    "human_label",
    "human_notes",
    "expert_confidence",
    "annotator1_label",
    "annotator1_notes",
    "annotator2_label",
    "annotator2_notes",
    "adjudicated_label",
    "adjudication_notes",
]


def read_events(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def dart_dictionary(dart: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        dart.groupby(["taxonomy_code", "taxonomy_label", "event_subtype", "clean_event_title"], dropna=False)
        .agg(
            event_count=("clean_event_title", "size"),
            firm_count=("firm_id", "nunique"),
            first_date=("event_date", "min"),
            last_date=("event_date", "max"),
        )
        .reset_index()
        .rename(columns={"clean_event_title": "normalized_event"})
    )
    grouped["source"] = "DART"
    return grouped.sort_values(["taxonomy_code", "event_count"], ascending=[True, False])


def sec_dictionary(sec: pd.DataFrame) -> pd.DataFrame:
    sec = sec.copy()
    sec["item"] = sec["item"].fillna("NO_ITEM").astype(str)
    sec["normalized_event"] = "8-K Item " + sec["item"] + ": " + sec["item"].map(SEC_ITEM_DESCRIPTIONS).fillna(sec["item"])
    grouped = (
        sec.groupby(["taxonomy_code", "taxonomy_label", "event_subtype", "item", "normalized_event"], dropna=False)
        .agg(
            event_count=("normalized_event", "size"),
            firm_count=("firm_id", "nunique"),
            first_date=("event_date", "min"),
            last_date=("event_date", "max"),
        )
        .reset_index()
    )
    grouped["source"] = "SEC_DESKTOP_POPULATION"
    return grouped.sort_values(["taxonomy_code", "event_count"], ascending=[True, False])


def ordered_codes(dart: pd.DataFrame, sec: pd.DataFrame) -> list[str]:
    codes = sorted(set(dart["taxonomy_code"]) | set(sec["taxonomy_code"]))
    return [code for code in TARGET_CODE_ORDER if code in codes] + [code for code in codes if code not in TARGET_CODE_ORDER]


def add_match_rows(rows: list[dict[str, Any]], dart_dict: pd.DataFrame, sec_dict: pd.DataFrame, per_code_limit: int) -> None:
    for code in ordered_codes(dart_dict, sec_dict):
        d_part = dart_dict[dart_dict["taxonomy_code"].eq(code)].head(per_code_limit)
        s_part = sec_dict[sec_dict["taxonomy_code"].eq(code)].head(per_code_limit)
        if d_part.empty or s_part.empty:
            continue
        quota = MATCH_ROW_QUOTAS.get(code, 10)
        code_rows = 0
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
                        "dart_firm_count": int(d_row.firm_count),
                        "sec_event": s_row.normalized_event,
                        "sec_subtype": s_row.event_subtype,
                        "sec_item": s_row.item,
                        "sec_event_count": int(s_row.event_count),
                        "sec_firm_count": int(s_row.firm_count),
                        "sec_source_scope": "local SEC population scan",
                        "rationale": "Same supervisory taxonomy code using the broader local SEC population scan.",
                        "human_label": "",
                        "human_notes": "",
                    }
                )
                code_rows += 1
                if code_rows >= quota:
                    break
            if code_rows >= quota:
                break


def add_negative_rows(rows: list[dict[str, Any]], dart_dict: pd.DataFrame, sec_dict: pd.DataFrame, target_rows: int) -> None:
    dart_top = dart_dict.sort_values("event_count", ascending=False).head(30)
    sec_top = sec_dict.sort_values("event_count", ascending=False).head(30)
    for d_row in dart_top.itertuples(index=False):
        for s_row in sec_top.itertuples(index=False):
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
                    "dart_firm_count": int(d_row.firm_count),
                    "sec_event": s_row.normalized_event,
                    "sec_subtype": s_row.event_subtype,
                    "sec_item": s_row.item,
                    "sec_event_count": int(s_row.event_count),
                    "sec_firm_count": int(s_row.firm_count),
                    "sec_source_scope": "local SEC population scan",
                    "rationale": "Different taxonomy codes; included as a negative annotation seed.",
                    "human_label": "",
                    "human_notes": "",
                }
            )
            if len(rows) >= target_rows:
                return


def build_seed(dart_dict: pd.DataFrame, sec_dict: pd.DataFrame, target_rows: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    add_match_rows(rows, dart_dict, sec_dict, per_code_limit=8)
    if len(rows) < target_rows:
        add_negative_rows(rows, dart_dict, sec_dict, target_rows=target_rows)
    seed = pd.DataFrame(rows).head(target_rows).copy()
    seed["candidate_rank"] = range(1, len(seed) + 1)
    return seed


def preserve_existing_labels(seed: pd.DataFrame, existing_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    seed = seed.copy()
    for column in LABEL_COLUMNS:
        if column not in seed.columns:
            seed[column] = ""
        seed[column] = seed[column].fillna("").astype(str)
    if not existing_path.exists():
        return seed, {"preserved_rows": 0, "preserved_cells": 0, "source": str(existing_path), "source_exists": False}

    existing = pd.read_csv(existing_path, low_memory=False).copy()
    for column in LABEL_COLUMNS:
        if column not in existing.columns:
            existing[column] = ""
        existing[column] = existing[column].fillna("").astype(str)
    key_cols = ["candidate_rank", "dart_event", "sec_event", "taxonomy_code"]
    available_keys = [column for column in key_cols if column in existing.columns and column in seed.columns]
    if not available_keys:
        return seed, {"preserved_rows": 0, "preserved_cells": 0, "source": str(existing_path), "source_exists": True}

    existing = existing[available_keys + LABEL_COLUMNS].drop_duplicates(available_keys, keep="last")
    merged = seed.merge(existing, on=available_keys, how="left", suffixes=("", "_existing"))
    preserved_cells = 0
    preserved_rows: set[int] = set()
    for column in LABEL_COLUMNS:
        existing_col = f"{column}_existing"
        if existing_col not in merged.columns:
            continue
        mask = merged[existing_col].fillna("").astype(str).str.strip().ne("")
        if mask.any():
            merged.loc[mask, column] = merged.loc[mask, existing_col]
            preserved_cells += int(mask.sum())
            preserved_rows.update(int(idx) for idx in merged.index[mask].tolist())
    merged = merged.drop(columns=[column for column in merged.columns if column.endswith("_existing")])
    return merged, {
        "preserved_rows": len(preserved_rows),
        "preserved_cells": preserved_cells,
        "source": str(existing_path),
        "source_exists": True,
    }


def summarize(
    seed: pd.DataFrame,
    dart: pd.DataFrame,
    sec: pd.DataFrame,
    dart_dict: pd.DataFrame,
    sec_dict: pd.DataFrame,
    preservation: dict[str, Any],
) -> dict[str, Any]:
    match_seed = seed[seed["proposed_label"].eq("MATCH")]
    codes_in_seed = sorted(set(match_seed["taxonomy_code"]))
    source_overlap = sorted(set(dart["taxonomy_code"]) & set(sec["taxonomy_code"]))
    t01_seed = match_seed[match_seed["taxonomy_code"].eq("T01")]
    return {
        "status": "created_sec_desktop_alignment_seed",
        "candidate_rows": int(len(seed)),
        "match_rows": int(seed["proposed_label"].eq("MATCH").sum()),
        "negative_rows": int(seed["proposed_label"].eq("NO_MATCH").sum()),
        "taxonomy_codes_in_match_seed": codes_in_seed,
        "taxonomy_codes_in_source_overlap": source_overlap,
        "missing_overlap_codes_from_match_seed": sorted(set(source_overlap) - set(codes_in_seed)),
        "t01_candidate_rows": int(len(t01_seed)),
        "dart_t01_event_rows": int(dart["taxonomy_code"].eq("T01").sum()),
        "sec_desktop_t01_event_rows": int(sec["taxonomy_code"].eq("T01").sum()),
        "dart_dictionary_rows": int(len(dart_dict)),
        "sec_desktop_dictionary_rows": int(len(sec_dict)),
        "human_labeled_rows": int(seed["human_label"].astype(str).str.strip().ne("").sum()),
        "preservation": preservation,
        "guardrail": "This supplemental seed uses the broader local SEC population scan to represent T01. It is annotation-ready evidence only; do not report Precision@1/MRR until human labels are filled and evaluated.",
    }


def write_markdown(summary: dict[str, Any], seed: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Local SEC Supplemental Alignment Seed",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Candidate rows | {summary['candidate_rows']} |",
        f"| Match rows | {summary['match_rows']} |",
        f"| Negative rows | {summary['negative_rows']} |",
        f"| T01 candidate rows | {summary['t01_candidate_rows']} |",
        f"| DART T01 event rows | {summary['dart_t01_event_rows']} |",
        f"| Local SEC T01 event rows | {summary['sec_desktop_t01_event_rows']} |",
        f"| Human-labeled rows | {summary['human_labeled_rows']} |",
        f"| Preserved rows | {summary['preservation']['preserved_rows']} |",
        f"| Preserved cells | {summary['preservation']['preserved_cells']} |",
        "",
        "## Taxonomy Coverage",
        "",
        f"- Match-seed codes: {', '.join(summary['taxonomy_codes_in_match_seed'])}",
        f"- Source-overlap codes: {', '.join(summary['taxonomy_codes_in_source_overlap'])}",
        f"- Missing overlap codes from match seed: {', '.join(summary['missing_overlap_codes_from_match_seed']) or 'none'}",
        "",
        "## T01 Examples",
        "",
        "| Rank | DART event | SEC event | Proposed label |",
        "|---:|---|---|---|",
    ]
    t01_rows = seed[seed["taxonomy_code"].eq("T01")].head(12)
    if t01_rows.empty:
        lines.append("| - | - | - | - |")
    else:
        for row in t01_rows.itertuples(index=False):
            lines.append(
                f"| {row.candidate_rank} | {escape(row.dart_event)} | {escape(row.sec_event)} | `{row.proposed_label}` |"
            )
    lines.extend(["", "## Guardrail", "", summary["guardrail"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_guide(path: Path) -> None:
    lines = [
        "# Local SEC Supplemental Alignment Labeling Guide",
        "",
        "This guide covers `manual_alignment_seed_sec_desktop_100.csv/xlsx`, the supplemental T01-inclusive DART-SEC annotation seed built from the broad local SEC population scan.",
        "",
        "## Files",
        "",
        "- CSV: `Paper 4\\data\\processed\\manual_alignment_seed_sec_desktop_100.csv`",
        "- Excel: `Paper 4\\data\\processed\\manual_alignment_seed_sec_desktop_100.xlsx`",
        "- Evaluation output: `Paper 4\\results\\alignment_sec_desktop_evaluation.md`",
        "",
        "## Label Set",
        "",
        "| Label | Use when |",
        "|---|---|",
        "| `MATCH` | The DART and SEC events describe substantially the same legal/economic disclosure event. |",
        "| `PARTIAL` | The events share a risk domain but differ in legal trigger, timing, or specificity. |",
        "| `NO_MATCH` | The relationship is too broad or unrelated. |",
        "",
        "## Workflow",
        "",
        "1. Fill `human_label` for the supplemental seed, or use `annotator1_label`, `annotator2_label`, and `adjudicated_label` for the two-annotator route.",
        "2. Run `python \"Paper 4\\src\\align\\evaluate_alignment.py\" --seed-path \"Paper 4\\data\\processed\\manual_alignment_seed_sec_desktop_100.csv\" --out-json \"Paper 4\\results\\alignment_sec_desktop_evaluation.json\" --out-md \"Paper 4\\results\\alignment_sec_desktop_evaluation.md\"`.",
        "3. Report Precision@1/MRR only after labels are filled. Report Cohen's kappa only after both annotator columns are filled.",
        "",
        "## Guardrail",
        "",
        "This supplemental seed fixes T01 coverage for annotation prep. It does not by itself validate DART-SEC alignment accuracy, and Precision@1/MRR remain unavailable until human labels are filled and evaluated.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_excel(seed: pd.DataFrame, path: Path) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            seed.to_excel(writer, index=False, sheet_name="seed")
            ws = writer.book["seed"]
            widths = {
                "A": 14,
                "B": 14,
                "C": 14,
                "D": 34,
                "E": 24,
                "H": 46,
                "I": 30,
                "J": 12,
                "N": 28,
                "O": 70,
                "P": 16,
                "Q": 34,
            }
            for column, width in widths.items():
                ws.column_dimensions[column].width = width
    except Exception:
        # The CSV remains the canonical annotation file when optional Excel support is unavailable.
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dart-events", default=str(ROOT / "data" / "processed" / "events_dart.csv"))
    parser.add_argument("--sec-events", default=str(ROOT / "data" / "processed" / "sec_desktop_population_events.csv"))
    parser.add_argument("--target-rows", type=int, default=100)
    parser.add_argument("--out-csv", default=str(ROOT / "data" / "processed" / "manual_alignment_seed_sec_desktop_100.csv"))
    parser.add_argument("--out-xlsx", default=str(ROOT / "data" / "processed" / "manual_alignment_seed_sec_desktop_100.xlsx"))
    parser.add_argument("--out-md", default=str(ROOT / "results" / "alignment_sec_desktop_seed_summary.md"))
    parser.add_argument("--out-json", default=str(ROOT / "results" / "alignment_sec_desktop_seed_summary.json"))
    parser.add_argument("--out-guide", default=str(ROOT / "manuscript" / "alignment_sec_desktop_labeling_guide.md"))
    args = parser.parse_args()

    dart = read_events(Path(args.dart_events))
    sec = read_events(Path(args.sec_events))
    dart_dict = dart_dictionary(dart)
    sec_dict = sec_dictionary(sec)
    seed = build_seed(dart_dict, sec_dict, target_rows=args.target_rows)
    seed, preservation = preserve_existing_labels(seed, Path(args.out_csv))
    summary = summarize(seed, dart, sec, dart_dict, sec_dict, preservation)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    seed.to_csv(out_csv, index=False, encoding="utf-8-sig")
    write_excel(seed, Path(args.out_xlsx))

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, seed, Path(args.out_md))
    write_guide(Path(args.out_guide))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
