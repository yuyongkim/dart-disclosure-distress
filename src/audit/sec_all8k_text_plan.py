from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("Paper 4")
DEFAULT_FILINGS = ROOT / "data" / "raw" / "sec_desktop_population" / "sec_desktop_population_8k_filings.csv"
DEFAULT_TEXT_ROOT = ROOT / "data" / "raw" / "sec_filing_text"
OUT_ROOT = ROOT / "data" / "raw" / "sec_all8k_text"
RESULTS_DIR = ROOT / "results"
TARGET_ITEMS = ("1.03", "2.04", "3.01")


def normalize_cik(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else ""


def accession_nodash(value: object) -> str:
    return str(value or "").replace("-", "").strip()


def split_items(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()}


def merge_items(values: pd.Series) -> str:
    items: set[str] = set()
    for value in values:
        items.update(split_items(value))
    return ",".join(sorted(items, key=lambda x: tuple(int(p) if p.isdigit() else p for p in x.split("."))))


def first_nonempty(values: pd.Series) -> str:
    for value in values:
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    return ""


def archive_txt_url(cik: str, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash(accession)}/{accession}.txt"


def local_text_path(text_root: Path, cik: str, accession: str) -> Path:
    return text_root / cik / f"{accession_nodash(accession)}.txt"


def bytes_mb(value: float | int) -> float:
    return round(float(value) / 1_000_000, 3)


def build_manifest(filings_path: Path, text_root: Path) -> pd.DataFrame:
    cols = [
        "cik",
        "entity_name",
        "accession_number",
        "filing_date",
        "form",
        "items",
        "primary_document",
        "primary_doc_description",
        "source_dataset",
    ]
    filings = pd.read_csv(filings_path, usecols=lambda col: col in cols, low_memory=False)
    filings["cik"] = filings["cik"].map(normalize_cik)
    filings = filings[filings["cik"].ne("") & filings["accession_number"].notna()].copy()
    filings["accession_number"] = filings["accession_number"].astype(str).str.strip()
    filings["filing_date_ts"] = pd.to_datetime(filings["filing_date"], errors="coerce")
    grouped = (
        filings.sort_values(["cik", "accession_number", "filing_date_ts"], kind="mergesort")
        .groupby(["cik", "accession_number"], sort=False)
        .agg(
            entity_name=("entity_name", first_nonempty),
            filing_date=("filing_date", first_nonempty),
            form=("form", first_nonempty),
            items=("items", merge_items),
            primary_document=("primary_document", first_nonempty),
            primary_doc_description=("primary_doc_description", first_nonempty),
            source_dataset=("source_dataset", first_nonempty),
            source_row_count=("accession_number", "size"),
        )
        .reset_index()
    )
    grouped["archive_txt_url"] = grouped.apply(lambda row: archive_txt_url(row["cik"], row["accession_number"]), axis=1)
    grouped["local_text_path"] = grouped.apply(lambda row: str(local_text_path(text_root, row["cik"], row["accession_number"])), axis=1)
    grouped["local_text_exists"] = grouped["local_text_path"].map(lambda value: Path(value).exists())
    grouped["local_text_bytes"] = grouped["local_text_path"].map(lambda value: int(Path(value).stat().st_size) if Path(value).exists() else 0)
    item_sets = grouped["items"].map(split_items)
    grouped["has_target_distress_item"] = item_sets.map(lambda values: bool(values.intersection(TARGET_ITEMS)))
    for item in TARGET_ITEMS:
        grouped[f"has_item_{item.replace('.', '_')}"] = item_sets.map(lambda values, item=item: item in values)
    return grouped


def build_batch_plan(manifest: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    missing = manifest[~manifest["local_text_exists"]].copy()
    missing["download_priority"] = missing["has_target_distress_item"].map(lambda value: 0 if value else 1)
    missing = missing.sort_values(["download_priority", "filing_date", "cik", "accession_number"], kind="mergesort").reset_index(drop=True)
    if missing.empty:
        return pd.DataFrame(columns=["batch_id", "download_priority", "rows", "target_rows", "first_filing_date", "last_filing_date"])
    missing["batch_id"] = (missing.index // batch_size) + 1
    plan = (
        missing.groupby("batch_id", sort=True)
        .agg(
            download_priority=("download_priority", "min"),
            rows=("accession_number", "size"),
            target_rows=("has_target_distress_item", "sum"),
            first_filing_date=("filing_date", "min"),
            last_filing_date=("filing_date", "max"),
        )
        .reset_index()
    )
    plan["target_rows"] = plan["target_rows"].astype(int)
    return plan


def summarize(manifest: pd.DataFrame, filings_path: Path, text_root: Path, batch_size: int, batch_plan: pd.DataFrame) -> dict[str, Any]:
    local_files = int(manifest["local_text_exists"].sum())
    local_bytes = int(manifest["local_text_bytes"].sum())
    missing_rows = int((~manifest["local_text_exists"]).sum())
    target_manifest = manifest[manifest["has_target_distress_item"]]
    target_local_files = int(target_manifest["local_text_exists"].sum())
    observed_avg_bytes = float(local_bytes / local_files) if local_files else 0.0
    estimated_total_bytes = int(observed_avg_bytes * len(manifest)) if observed_avg_bytes else 0
    estimated_missing_bytes = int(observed_avg_bytes * missing_rows) if observed_avg_bytes else 0
    return {
        "status": "plan_created",
        "filings_path": str(filings_path),
        "text_root_reused": str(text_root),
        "manifest_rows": int(len(manifest)),
        "source_duplicate_rows_collapsed": int(manifest["source_row_count"].sum() - len(manifest)),
        "firms": int(manifest["cik"].nunique()),
        "target_distress_item_rows": int(len(target_manifest)),
        "target_distress_item_local_files": target_local_files,
        "target_distress_item_coverage_rate": float(target_local_files / len(target_manifest)) if len(target_manifest) else None,
        "local_text_files": local_files,
        "local_text_bytes": local_bytes,
        "all8k_text_coverage_rate": float(local_files / len(manifest)) if len(manifest) else None,
        "missing_text_files": missing_rows,
        "observed_avg_cached_text_bytes": observed_avg_bytes,
        "estimated_total_text_bytes": estimated_total_bytes,
        "estimated_missing_text_bytes": estimated_missing_bytes,
        "estimated_total_text_gb": round(estimated_total_bytes / 1_000_000_000, 3),
        "estimated_missing_text_gb": round(estimated_missing_bytes / 1_000_000_000, 3),
        "batch_size": int(batch_size),
        "planned_batches": int(len(batch_plan)),
        "guardrail": "Planning artifact only. It builds SEC Archives URLs for all local 8-K/8-K/A accessions but performs no downloads.",
    }


def write_markdown(summary: dict[str, Any], batch_plan: pd.DataFrame, path: Path) -> None:
    lines = [
        "# SEC All-8-K Filing-Body Text Plan",
        "",
        "This file defines the full filing-body text scope for the local SEC local-cache 8-K population. It performs no network calls.",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Manifest rows, unique CIK-accessions | {summary['manifest_rows']} |",
        f"| Source duplicate rows collapsed | {summary['source_duplicate_rows_collapsed']} |",
        f"| Firms | {summary['firms']} |",
        f"| Distress target-item rows | {summary['target_distress_item_rows']} |",
        f"| Distress target-item local files | {summary['target_distress_item_local_files']} |",
        f"| Distress target-item coverage | {pct(summary['target_distress_item_coverage_rate'])} |",
        f"| All-8-K local text files | {summary['local_text_files']} |",
        f"| All-8-K text coverage | {pct(summary['all8k_text_coverage_rate'])} |",
        f"| Missing all-8-K text files | {summary['missing_text_files']} |",
        f"| Existing cached text size | {bytes_mb(summary['local_text_bytes'])} MB |",
        f"| Observed average cached text size | {round(summary['observed_avg_cached_text_bytes'], 1)} bytes |",
        f"| Estimated total all-8-K text size | {summary['estimated_total_text_gb']} GB |",
        f"| Estimated missing text size | {summary['estimated_missing_text_gb']} GB |",
        f"| Batch size | {summary['batch_size']} |",
        f"| Planned batches | {summary['planned_batches']} |",
        "",
        "## First Download Batches",
        "",
        "| Batch | Priority | Rows | Target rows | First filing date | Last filing date |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for row in batch_plan.head(20).itertuples(index=False):
        lines.append(
            f"| {row.batch_id} | {row.download_priority} | {row.rows} | {row.target_rows} | {row.first_filing_date} | {row.last_filing_date} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This is a planning artifact only; it does not download SEC Archives files.",
            "- Use a real identifying `SEC_USER_AGENT` before any future download step.",
            "- The current target-item cache should not be confused with all-8-K text coverage.",
            "- A standalone SEC text validation needs this all-8-K scope completed or an explicitly justified sampling design.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filings-path", default=str(DEFAULT_FILINGS))
    parser.add_argument("--text-root", default=str(DEFAULT_TEXT_ROOT))
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    results_dir = Path(args.results_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(Path(args.filings_path), Path(args.text_root))
    batch_plan = build_batch_plan(manifest, args.batch_size)
    summary = summarize(manifest, Path(args.filings_path), Path(args.text_root), args.batch_size, batch_plan)

    manifest_path = out_root / "sec_all8k_text_manifest.csv"
    batch_path = out_root / "sec_all8k_text_batch_plan.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    batch_plan.to_csv(batch_path, index=False, encoding="utf-8-sig")
    summary["manifest_csv"] = str(manifest_path)
    summary["batch_plan_csv"] = str(batch_path)

    (results_dir / "sec_all8k_text_scope_plan.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(summary, batch_plan, results_dir / "sec_all8k_text_scope_plan.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
