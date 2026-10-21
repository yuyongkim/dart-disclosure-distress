from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("Paper 4")
RAW_ROOT = Path("Paper3") / "dart_apriori" / "data" / "raw"
RESULTS_DIR = ROOT / "results"

OUTCOME_ITEMS = {
    "1.03": "bankruptcy_or_receivership",
    "2.04": "debt_acceleration_or_default_trigger",
    "3.01": "delisting_notice",
    "3.03": "security_holder_rights_modification",
}
CONTEXT_ITEMS = ["1.01", "2.01", "2.03", "3.02", "5.02", "5.03"]
ITEMS_OF_INTEREST = list(OUTCOME_ITEMS) + CONTEXT_ITEMS


def split_items(value: object) -> list[str]:
    if pd.isna(value):
        return ["NO_ITEM"]
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return items or ["NO_ITEM"]


def period_mask(dates: pd.Series, start_year: int, end_year: int) -> pd.Series:
    return dates.dt.year.between(start_year, end_year, inclusive="both")


def normalize_cik(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(10) if text else ""


def safe_int(value: object) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def load_sec_files(raw_root: Path) -> list[Path]:
    return sorted(raw_root.glob("edgar_8k_items*.csv"))


def summarize_file(path: Path) -> tuple[dict, list[dict], pd.DataFrame]:
    df = pd.read_csv(path, low_memory=False)
    required = {"cik", "company_name", "filed_date", "items"}
    if not required.issubset(df.columns):
        return {
            "source_file": str(path),
            "usable": False,
            "reason": f"missing required columns: {sorted(required - set(df.columns))}",
        }, [], pd.DataFrame()

    frame = df.copy()
    frame["filed_date"] = pd.to_datetime(frame["filed_date"], errors="coerce")
    frame = frame.dropna(subset=["filed_date"])
    frame["cik_norm"] = frame["cik"].map(normalize_cik)
    frame["item"] = frame["items"].map(split_items)
    exploded = frame.explode("item").copy()
    exploded["item"] = exploded["item"].astype(str).str.strip()
    exploded = exploded[exploded["item"] != ""]

    mask_2015_2024 = period_mask(exploded["filed_date"], 2015, 2024)
    mask_2015_2025 = period_mask(exploded["filed_date"], 2015, 2025)

    summary: dict[str, object] = {
        "source_file": str(path),
        "usable": True,
        "rows": int(len(frame)),
        "exploded_item_rows": int(len(exploded)),
        "firms": int(frame["cik_norm"].nunique()),
        "date_min": frame["filed_date"].min().date().isoformat() if not frame.empty else "",
        "date_max": frame["filed_date"].max().date().isoformat() if not frame.empty else "",
        "rows_2015_2024": int(mask_2015_2024.sum()),
        "rows_2015_2025": int(mask_2015_2025.sum()),
        "firms_2015_2024": int(exploded.loc[mask_2015_2024, "cik_norm"].nunique()),
        "firms_2015_2025": int(exploded.loc[mask_2015_2025, "cik_norm"].nunique()),
    }

    for item in ITEMS_OF_INTEREST:
        item_mask = exploded["item"].eq(item)
        summary[f"item_{item.replace('.', '_')}_all"] = int(item_mask.sum())
        summary[f"item_{item.replace('.', '_')}_2015_2024"] = int((item_mask & mask_2015_2024).sum())
        summary[f"item_{item.replace('.', '_')}_2015_2025"] = int((item_mask & mask_2015_2025).sum())

    examples = []
    for item, meaning in OUTCOME_ITEMS.items():
        rows = exploded[exploded["item"].eq(item)].sort_values("filed_date").head(10)
        for row in rows.itertuples(index=False):
            examples.append(
                {
                    "source_file": path.name,
                    "item": item,
                    "meaning": meaning,
                    "cik": row.cik_norm,
                    "company_name": getattr(row, "company_name", ""),
                    "filed_date": row.filed_date.date().isoformat(),
                    "items_raw": getattr(row, "items", ""),
                }
            )

    keep_cols = ["source_file", "cik_norm", "company_name", "filed_date", "item", "items"]
    exploded["source_file"] = path.name
    return summary, examples, exploded[keep_cols]


def aggregate_inventory(exploded_frames: list[pd.DataFrame]) -> dict:
    if not exploded_frames:
        return {}
    all_items = pd.concat(exploded_frames, ignore_index=True)
    unique_items = all_items.drop_duplicates(
        subset=["cik_norm", "company_name", "filed_date", "item", "items"]
    ).copy()
    mask_2015_2024 = period_mask(unique_items["filed_date"], 2015, 2024)
    mask_2015_2025 = period_mask(unique_items["filed_date"], 2015, 2025)

    aggregate: dict[str, object] = {
        "raw_exploded_rows": int(len(all_items)),
        "unique_filing_item_rows": int(len(unique_items)),
        "firms": int(unique_items["cik_norm"].nunique()),
        "date_min": unique_items["filed_date"].min().date().isoformat(),
        "date_max": unique_items["filed_date"].max().date().isoformat(),
        "firms_2015_2024": int(unique_items.loc[mask_2015_2024, "cik_norm"].nunique()),
        "firms_2015_2025": int(unique_items.loc[mask_2015_2025, "cik_norm"].nunique()),
    }
    for item in OUTCOME_ITEMS:
        item_mask = unique_items["item"].eq(item)
        aggregate[item] = {
            "all_rows": int(item_mask.sum()),
            "rows_2015_2024": int((item_mask & mask_2015_2024).sum()),
            "firms_2015_2024": int(unique_items.loc[item_mask & mask_2015_2024, "cik_norm"].nunique()),
            "rows_2015_2025": int((item_mask & mask_2015_2025).sum()),
            "firms_2015_2025": int(unique_items.loc[item_mask & mask_2015_2025, "cik_norm"].nunique()),
        }
    return aggregate


def write_markdown(summary_df: pd.DataFrame, examples_df: pd.DataFrame, aggregate: dict, path: Path) -> None:
    lines = [
        "# SEC Gap Inventory",
        "",
        "This report scans local EDGAR 8-K item CSV files to document whether Paper 4 can support a standalone U.S. distress-prediction extension from existing files.",
        "",
        "## Bottom Line",
        "",
    ]

    item_103 = aggregate.get("1.03", {})
    rows_103_2015_2024 = safe_int(item_103.get("rows_2015_2024", 0))
    rows_103_2015_2025 = safe_int(item_103.get("rows_2015_2025", 0))
    if rows_103_2015_2024 == 0:
        lines.append("- Local SEC files still have no Item 1.03 bankruptcy/receivership rows inside the current 2015-2024 event window.")
    else:
        lines.append(f"- Local SEC files contain {rows_103_2015_2024} Item 1.03 rows inside the current 2015-2024 event window.")
    lines.append(f"- Local SEC files contain {rows_103_2015_2025} Item 1.03 rows when 2025 is included.")
    lines.append("- Aggregate counts are de-duplicated by CIK, company, filing date, item, and raw item string because the local folder contains overlapping master, expanded, and 2025 split files.")
    lines.append("- Therefore Paper 4 v1 should keep SEC as taxonomy-transfer/alignment evidence, not as the main standalone predictive validation.")
    lines.append("- A v2 SEC expansion should collect a broader 8-K universe centered on Item 1.03, Item 2.04, and Item 3.01 outcomes before claiming U.S. distress early-warning performance.")
    lines.append("")

    lines.extend(
        [
            "## Outcome Item Coverage",
            "",
            "| SEC item | Meaning | Rows 2015-2024 | Firms 2015-2024 | Rows 2015-2025 | Firms 2015-2025 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item, meaning in OUTCOME_ITEMS.items():
        values = aggregate.get(item, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item}`",
                    meaning,
                    str(safe_int(values.get("rows_2015_2024", 0))),
                    str(safe_int(values.get("firms_2015_2024", 0))),
                    str(safe_int(values.get("rows_2015_2025", 0))),
                    str(safe_int(values.get("firms_2015_2025", 0))),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## File-Level Scan",
            "",
            "| File | Rows | Firms | Date range | Item 1.03 2015-2024 | Item 1.03 2015-2025 | Item 2.04 2015-2024 | Item 3.01 2015-2024 |",
            "|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary_df.sort_values("source_file").itertuples(index=False):
        if not getattr(row, "usable"):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{Path(row.source_file).name}`",
                    str(int(row.rows)),
                    str(int(row.firms)),
                    f"{row.date_min} to {row.date_max}",
                    str(int(getattr(row, "item_1_03_2015_2024"))),
                    str(int(getattr(row, "item_1_03_2015_2025"))),
                    str(int(getattr(row, "item_2_04_2015_2024"))),
                    str(int(getattr(row, "item_3_01_2015_2024"))),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Item 1.03 Examples", ""])
    if examples_df.empty or not (examples_df["item"] == "1.03").any():
        lines.append("No local Item 1.03 rows were found.")
    else:
        item_103 = (
            examples_df[examples_df["item"] == "1.03"]
            .groupby(["cik", "company_name", "filed_date", "items_raw"], as_index=False)
            .agg({"source_file": lambda values: ", ".join(sorted(set(values)))})
            .sort_values(["filed_date", "source_file"])
        )
        lines.extend(
            [
                "| Source files | CIK | Company | Filed date | Items |",
                "|---|---|---|---|---|",
            ]
        )
        for row in item_103.itertuples(index=False):
            company = str(row.company_name).replace("|", "\\|")
            items_raw = str(row.items_raw).replace("|", "\\|")
            lines.append(f"| `{row.source_file}` | `{row.cik}` | {company} | {row.filed_date} | `{items_raw}` |")

    lines.extend(
        [
            "",
            "## Manuscript Use",
            "",
            "- Use this as evidence for the limitation paragraph: the local SEC sample has useful event taxonomy coverage but insufficient bankruptcy/receivership outcomes in 2015-2024.",
            "- Do not move SEC into the headline predictive result without a new collection pass.",
            "- If Paper 4 v2 targets a stronger cross-market paper, expand SEC first, then add EDINET second.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    examples: list[dict] = []
    exploded_frames: list[pd.DataFrame] = []

    for path in load_sec_files(RAW_ROOT):
        summary, file_examples, exploded = summarize_file(path)
        summaries.append(summary)
        examples.extend(file_examples)
        if not exploded.empty:
            exploded_frames.append(exploded)

    summary_df = pd.DataFrame(summaries)
    examples_df = pd.DataFrame(examples)
    aggregate = aggregate_inventory(exploded_frames)

    summary_path = RESULTS_DIR / "sec_gap_inventory.csv"
    examples_path = RESULTS_DIR / "sec_gap_item_examples.csv"
    json_path = RESULTS_DIR / "sec_gap_inventory.json"
    md_path = RESULTS_DIR / "sec_gap_inventory.md"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    examples_df.to_csv(examples_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(
            {
                "raw_root": str(RAW_ROOT),
                "files_scanned": len(summaries),
                "usable_files": int(summary_df.get("usable", pd.Series(dtype=bool)).fillna(False).sum()) if not summary_df.empty else 0,
                "aggregate": aggregate,
                "summary_file": str(summary_path),
                "examples_file": str(examples_path),
                "markdown_file": str(md_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_markdown(summary_df, examples_df, aggregate, md_path)

    print(
        json.dumps(
            {
                "files_scanned": len(summaries),
                "usable_files": int(summary_df.get("usable", pd.Series(dtype=bool)).fillna(False).sum()) if not summary_df.empty else 0,
                "item_1_03_2015_2024": aggregate.get("1.03", {}).get("rows_2015_2024", 0),
                "item_1_03_2015_2025": aggregate.get("1.03", {}).get("rows_2015_2025", 0),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
