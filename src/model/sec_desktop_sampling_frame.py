from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path("Paper 4")
TARGET_ITEMS = ("1.03", "2.04", "3.01", "3.03", "5.02")


def normalize_cik(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else ""


def stable_bucket(value: object, folds: int = 5) -> int:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def first_nonempty(values: pd.Series) -> str:
    clean = values.dropna().astype(str).str.strip()
    clean = clean[clean.ne("")]
    return clean.iloc[0] if not clean.empty else ""


def build_firm_frame(filings: pd.DataFrame, file_summary: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if filings.empty:
        return pd.DataFrame()
    filings = filings.copy()
    filings["cik"] = filings["cik"].map(normalize_cik)
    filings["filing_date"] = pd.to_datetime(filings["filing_date"], errors="coerce")
    for item in TARGET_ITEMS:
        col = f"has_item_{item.replace('.', '_')}"
        if col not in filings:
            filings[col] = False
        filings[col] = filings[col].fillna(False).astype(bool)

    agg_spec = {
        "entity_name": first_nonempty,
        "tickers": first_nonempty,
        "exchanges": first_nonempty,
        "sic": first_nonempty,
        "filing_date": ["min", "max"],
        "accession_number": "nunique",
        "form": "count",
        "source_path": "nunique",
    }
    grouped = filings.groupby("cik").agg(agg_spec)
    grouped.columns = [
        "entity_name",
        "tickers",
        "exchanges",
        "sic",
        "first_filing_date",
        "last_filing_date",
        "unique_accessions",
        "filing_rows",
        "source_files",
    ]
    firm_frame = grouped.reset_index()
    firm_frame["first_filing_date"] = firm_frame["first_filing_date"].dt.date.astype(str)
    firm_frame["last_filing_date"] = firm_frame["last_filing_date"].dt.date.astype(str)

    form_counts = filings.pivot_table(index="cik", columns="form", values="accession_number", aggfunc="count", fill_value=0)
    for form in ("8-K", "8-K/A"):
        firm_frame[f"form_{form.replace('-', '').replace('/', 'a').lower()}_rows"] = firm_frame["cik"].map(form_counts.get(form, pd.Series(dtype=int))).fillna(0).astype(int)

    for item in TARGET_ITEMS:
        col = f"has_item_{item.replace('.', '_')}"
        out_col = f"item_{item.replace('.', '_')}_rows"
        firm_frame[out_col] = firm_frame["cik"].map(filings.groupby("cik")[col].sum()).fillna(0).astype(int)
    firm_frame["target_item_rows"] = firm_frame[[f"item_{item.replace('.', '_')}_rows" for item in TARGET_ITEMS]].sum(axis=1)
    firm_frame["has_item_103_bankruptcy"] = firm_frame["item_1_03_rows"].gt(0)
    firm_frame["has_any_distress_item_103_204_301"] = (
        firm_frame["item_1_03_rows"].gt(0) | firm_frame["item_2_04_rows"].gt(0) | firm_frame["item_3_01_rows"].gt(0)
    )

    if not file_summary.empty:
        file_summary = file_summary.copy()
        file_summary["cik"] = file_summary["cik"].map(normalize_cik)
        fs = file_summary.groupby("cik").agg(
            recent_rows=("recent_rows", "max"),
            historical_file_references=("historical_file_references", "max"),
            filing_rows_kept=("filing_rows_kept", "sum"),
        )
        firm_frame = firm_frame.merge(fs.reset_index(), on="cik", how="left")
    else:
        firm_frame["recent_rows"] = pd.NA
        firm_frame["historical_file_references"] = pd.NA
        firm_frame["filing_rows_kept"] = pd.NA

    if not panel.empty:
        panel = panel.copy()
        panel["firm_id"] = panel["firm_id"].map(normalize_cik)
        panel_counts = panel.groupby("firm_id").agg(
            validation_anchor_rows=("anchor_year", "count"),
            validation_tasks=("task", lambda values: ",".join(sorted(set(map(str, values))))),
            validation_anchor_year_min=("anchor_year", "min"),
            validation_anchor_year_max=("anchor_year", "max"),
            validation_test_anchor_rows=("anchor_year", lambda values: int((values >= 2022).sum())),
            validation_positive_365_rows=("label_365d", "sum"),
            validation_positive_730_rows=("label_730d", "sum"),
        )
        firm_frame = firm_frame.merge(panel_counts.reset_index().rename(columns={"firm_id": "cik"}), on="cik", how="left")
    for col in (
        "validation_anchor_rows",
        "validation_test_anchor_rows",
        "validation_positive_365_rows",
        "validation_positive_730_rows",
    ):
        if col not in firm_frame:
            firm_frame[col] = 0
        firm_frame[col] = firm_frame[col].fillna(0).astype(int)
    if "validation_tasks" not in firm_frame:
        firm_frame["validation_tasks"] = ""
    firm_frame["validation_tasks"] = firm_frame["validation_tasks"].fillna("")
    for col in ("validation_anchor_year_min", "validation_anchor_year_max"):
        if col not in firm_frame:
            firm_frame[col] = pd.NA
        firm_frame[col] = pd.to_numeric(firm_frame[col], errors="coerce").astype("Int64")

    firm_frame["included_in_sequence_validation"] = firm_frame["validation_anchor_rows"].gt(0)
    firm_frame["sampling_frame_status"] = firm_frame["included_in_sequence_validation"].map(
        {True: "included_in_sequence_validation", False: "retained_filings_no_eligible_validation_anchor"}
    )
    firm_frame["firm_disjoint_bucket_5"] = firm_frame["cik"].map(lambda value: stable_bucket(value, 5))
    firm_frame["source_limitation"] = "cached SEC submissions metadata plus available archive JSON; final U.S. validation scope remains pending"
    return firm_frame.sort_values(["included_in_sequence_validation", "cik"], ascending=[False, True])


def build_anchor_frame(panel: pd.DataFrame, firm_frame: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    anchor = panel[
        [
            "task",
            "firm_id",
            "firm_name",
            "anchor_year",
            "anchor_date",
            "outcome_date",
            "outcome_label",
            "label_365d",
            "label_730d",
        ]
    ].copy()
    anchor["firm_id"] = anchor["firm_id"].map(normalize_cik)
    anchor["temporal_split"] = anchor["anchor_year"].map(lambda year: "train_pre_2022" if int(year) <= 2021 else "test_2022_2023")
    anchor["firm_disjoint_bucket_5"] = anchor["firm_id"].map(lambda value: stable_bucket(value, 5))
    if not firm_frame.empty:
        meta = firm_frame[["cik", "tickers", "exchanges", "sic", "filing_rows", "target_item_rows"]].copy()
        anchor = anchor.merge(meta.rename(columns={"cik": "firm_id"}), on="firm_id", how="left")
    return anchor.sort_values(["task", "firm_id", "anchor_year"])


def label_summary(anchor: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if anchor.empty:
        return rows
    for task, task_frame in anchor.groupby("task"):
        for split, split_frame in task_frame.groupby("temporal_split"):
            for horizon in (365, 730):
                label = f"label_{horizon}d"
                rows.append(
                    {
                        "task": task,
                        "temporal_split": split,
                        "horizon_days": horizon,
                        "anchor_rows": int(len(split_frame)),
                        "firms": int(split_frame["firm_id"].nunique()),
                        "positive_rows": int(split_frame[label].sum()),
                        "positive_rate": float(split_frame[label].mean()) if len(split_frame) else None,
                    }
                )
    return rows


def write_markdown(summary: dict, label_rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Local SEC Sampling Frame",
        "",
        "This file documents the formal local sampling frame used for the SEC local-cache v2 sequence validation artifacts.",
        "",
        "## Inclusion Criteria",
        "",
        "- Source files are cached SEC submissions JSON files under the local SEC raw-data directory.",
        "- Retained filings are Form 8-K or 8-K/A with filing dates from 2015-01-01 through 2024-12-31.",
        "- Validation anchors are firm-year year-end anchors from 2016 through 2023 generated from firms with prior retained SEC event history.",
        "- Outcome tasks are all-distress Items 1.03/2.04/3.01 and bankruptcy-only Item 1.03.",
        "",
        "## Counts",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Retained SEC firms | {summary['firm_rows']} |",
        f"| Firms included in validation panel | {summary['validation_firms']} |",
        f"| Retained 8-K/8-K/A filings | {summary['filing_rows']} |",
        f"| Task-anchor rows | {summary['anchor_rows']} |",
        f"| Train anchor rows | {summary['train_anchor_rows']} |",
        f"| Test anchor rows | {summary['test_anchor_rows']} |",
        f"| Item 1.03 firms | {summary['item_103_firms']} |",
        f"| Item 2.04 firms | {summary['item_204_firms']} |",
        f"| Item 3.01 firms | {summary['item_301_firms']} |",
        "",
        "## Label Distribution",
        "",
        "| Task | Split | Horizon | Anchor rows | Firms | Positives | Positive rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in label_rows:
        rate = row["positive_rate"]
        rate_text = "" if rate is None else f"{float(rate):.4f}"
        lines.append(
            f"| {row['task']} | {row['temporal_split']} | {row['horizon_days']} | {row['anchor_rows']} | {row['firms']} | {row['positive_rows']} | {rate_text} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This makes the current SEC v2 validation frame explicit and incorporates cached archive submissions metadata where available.",
            "- It does not itself add SEC financial fundamentals or all-8-K filing-body text features; those are handled by separate v2 artifacts.",
            "- Bankruptcy-only positives remain sparse; all-distress results are the more stable SEC feasibility signal.",
            "- Use this as a v2 sampling-frame artifact, not as the DART-first v1 headline.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filings-path", default=str(ROOT / "data" / "raw" / "sec_desktop_population" / "sec_desktop_population_8k_filings.parquet"))
    parser.add_argument("--file-summary-path", default=str(ROOT / "data" / "raw" / "sec_desktop_population" / "sec_desktop_population_file_summary.csv"))
    parser.add_argument("--panel-path", default=str(ROOT / "data" / "model_sec_desktop" / "sec_desktop_validation_panel.parquet"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "model_sec_desktop"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    filings = read_table(Path(args.filings_path))
    file_summary = read_table(Path(args.file_summary_path))
    panel = read_table(Path(args.panel_path))

    firm_frame = build_firm_frame(filings, file_summary, panel)
    anchor_frame = build_anchor_frame(panel, firm_frame)
    label_rows = label_summary(anchor_frame)

    firm_csv = out_dir / "sec_desktop_sampling_frame_firms.csv"
    firm_parquet = out_dir / "sec_desktop_sampling_frame_firms.parquet"
    anchor_csv = out_dir / "sec_desktop_sampling_frame_anchors.csv"
    anchor_parquet = out_dir / "sec_desktop_sampling_frame_anchors.parquet"
    summary_json = results_dir / "sec_desktop_sampling_frame_summary.json"
    summary_md = results_dir / "sec_desktop_sampling_frame_summary.md"

    firm_frame.to_csv(firm_csv, index=False, encoding="utf-8-sig")
    anchor_frame.to_csv(anchor_csv, index=False, encoding="utf-8-sig")
    firm_frame.to_parquet(firm_parquet, index=False)
    anchor_frame.to_parquet(anchor_parquet, index=False)

    summary = {
        "status": "created",
        "filings_path": str(args.filings_path),
        "file_summary_path": str(args.file_summary_path),
        "panel_path": str(args.panel_path),
        "firm_rows": int(len(firm_frame)),
        "validation_firms": int(firm_frame["included_in_sequence_validation"].sum()) if not firm_frame.empty else 0,
        "filing_rows": int(firm_frame["filing_rows"].sum()) if not firm_frame.empty else 0,
        "anchor_rows": int(len(anchor_frame)),
        "train_anchor_rows": int(anchor_frame["temporal_split"].eq("train_pre_2022").sum()) if not anchor_frame.empty else 0,
        "test_anchor_rows": int(anchor_frame["temporal_split"].eq("test_2022_2023").sum()) if not anchor_frame.empty else 0,
        "item_103_firms": int(firm_frame["item_1_03_rows"].gt(0).sum()) if not firm_frame.empty else 0,
        "item_204_firms": int(firm_frame["item_2_04_rows"].gt(0).sum()) if not firm_frame.empty else 0,
        "item_301_firms": int(firm_frame["item_3_01_rows"].gt(0).sum()) if not firm_frame.empty else 0,
        "label_summary": label_rows,
        "firm_frame_csv": str(firm_csv),
        "anchor_frame_csv": str(anchor_csv),
        "guardrail": "Formal local SEC v2 sampling-frame artifact; sequence-only sampling frame based on cached submissions metadata plus available archive JSON.",
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, label_rows, summary_md)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
