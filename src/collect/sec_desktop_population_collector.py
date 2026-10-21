from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("Paper 4")
DEFAULT_SEC_ROOT = Path.home() / "Desktop" / "M&A" / "data" / "raw" / "sec"
TARGET_ITEMS = ("1.03", "2.04", "3.01", "3.03", "5.02")
FORMS = {"8-K", "8-K/A"}


def normalize_cik(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else ""


def split_items(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()]


def records_from_recent(recent: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(recent.keys())
    if not keys:
        return []
    count = max(len(recent.get(key, [])) for key in keys)
    rows = []
    for idx in range(count):
        row = {}
        for key in keys:
            values = recent.get(key, [])
            row[key] = values[idx] if idx < len(values) else None
        rows.append(row)
    return rows


def filing_rows_from_records(
    records: list[dict[str, Any]],
    *,
    cik: str,
    entity_name: str,
    tickers: list[Any],
    exchanges: list[Any],
    sic: object,
    source_path: Path,
    source_dataset: str,
    start_year: int,
    end_year: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filing in records:
        form = str(filing.get("form") or "")
        if form not in FORMS:
            continue
        filing_date = pd.to_datetime(filing.get("filingDate"), errors="coerce")
        if pd.isna(filing_date) or not (start_year <= filing_date.year <= end_year):
            continue
        items = split_items(filing.get("items"))
        rows.append(
            {
                "cik": cik,
                "entity_name": entity_name,
                "tickers": ",".join(map(str, tickers)),
                "exchanges": ",".join(map(str, exchanges)),
                "sic": sic,
                "accession_number": filing.get("accessionNumber"),
                "filing_date": filing_date.date().isoformat(),
                "report_date": filing.get("reportDate"),
                "acceptance_datetime": filing.get("acceptanceDateTime"),
                "form": form,
                "items": ",".join(items),
                "primary_document": filing.get("primaryDocument"),
                "primary_doc_description": filing.get("primaryDocDescription"),
                "source_path": str(source_path),
                "source_dataset": source_dataset,
                "has_target_item": any(item in TARGET_ITEMS for item in items),
                **{f"has_item_{item.replace('.', '_')}": item in items for item in TARGET_ITEMS},
            }
        )
    return rows


def flatten_submission(path: Path, start_year: int, end_year: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cik = normalize_cik(payload.get("cik"))
    entity_name = str(payload.get("name") or "")
    tickers = payload.get("tickers") or []
    exchanges = payload.get("exchanges") or []
    sic = payload.get("sic") or ""
    recent = payload.get("filings", {}).get("recent", {})
    historical_files = payload.get("filings", {}).get("files", []) or []

    rows = filing_rows_from_records(
        records_from_recent(recent),
        cik=cik,
        entity_name=entity_name,
        tickers=tickers,
        exchanges=exchanges,
        sic=sic,
        source_path=path,
        source_dataset="local SEC submissions current metadata",
        start_year=start_year,
        end_year=end_year,
    )
    archive_files_present = 0
    archive_rows_kept = 0
    for ref in historical_files:
        name = str(ref.get("name") or "").strip()
        if not name:
            continue
        archive_path = path.parent / name
        if not archive_path.exists():
            continue
        archive_files_present += 1
        archive_payload = json.loads(archive_path.read_text(encoding="utf-8"))
        archive_rows = filing_rows_from_records(
            records_from_recent(archive_payload),
            cik=cik,
            entity_name=entity_name,
            tickers=tickers,
            exchanges=exchanges,
            sic=sic,
            source_path=archive_path,
            source_dataset="local SEC submissions archive metadata",
            start_year=start_year,
            end_year=end_year,
        )
        archive_rows_kept += len(archive_rows)
        rows.extend(archive_rows)
    meta = {
        "cik": cik,
        "entity_name": entity_name,
        "source_path": str(path),
        "recent_rows": len(next(iter(recent.values()), [])) if recent else 0,
        "historical_file_references": len(historical_files),
        "historical_files_present": archive_files_present,
        "historical_filing_rows_kept": archive_rows_kept,
        "filing_rows_kept": len(rows),
    }
    return rows, meta


def output_paths(root: Path, prefix: str) -> dict[str, Path]:
    base = root / "data" / "raw" / prefix
    return {
        "base": base,
        "filings_csv": base / f"{prefix}_8k_filings.csv",
        "filings_parquet": base / f"{prefix}_8k_filings.parquet",
        "outcome_csv": base / f"{prefix}_outcome_candidates.csv",
        "file_summary_csv": base / f"{prefix}_file_summary.csv",
        "status_json": root / "results" / f"{prefix}_collection_status.json",
        "status_md": root / "results" / f"{prefix}_collection_status.md",
    }


def item_counts(filings: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if filings.empty or "items" not in filings:
        return counts
    for value in filings["items"]:
        for item in split_items(value):
            counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def write_markdown(status: dict[str, Any], path: Path) -> None:
    lines = [
        "# Local SEC Population Collection Status",
        "",
        "This scan reuses locally cached SEC submissions JSON files. It performs no new SEC API calls.",
        "",
        "## Coverage",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Status | `{status['status']}` |",
        f"| Source scope | `{status.get('source_scope', 'external_local_sec_cache')}` |",
        f"| Submission files found | {status['submission_files_found']} |",
        f"| Submission files parsed | {status['submission_files_parsed']} |",
        f"| Archive file references | {status.get('archive_file_references', 0)} |",
        f"| Archive files present locally | {status.get('archive_files_present', 0)} |",
        f"| 8-K / 8-K/A rows from archive files | {status.get('archive_filing_rows_kept', 0)} |",
        f"| Parse failures | {status['parse_failures']} |",
        f"| 8-K / 8-K/A filing rows, {status['start_year']}-{status['end_year']} | {status['filing_rows']} |",
        f"| SEC firms with retained filings | {status['filing_firms']} |",
        f"| Target-item candidate filings | {status['outcome_candidate_rows']} |",
        f"| Item 1.03 candidate filings | {status['item_counts'].get('1.03', 0)} |",
        f"| Item 2.04 candidate filings | {status['item_counts'].get('2.04', 0)} |",
        f"| Item 3.01 candidate filings | {status['item_counts'].get('3.01', 0)} |",
        f"| Date range | {status['date_min']} to {status['date_max']} |",
        "",
        "## Item Counts",
        "",
        "| 8-K item | Filing rows |",
        "|---|---:|",
    ]
    for item, count in list(status["item_counts"].items())[:40]:
        lines.append(f"| `{item}` | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a broad local SEC metadata scan with cached archive submissions JSON where available, not a freshly designed population sample.",
            "- The archive-reference audit now documents target-window archive metadata coverage; use this as v2 population-style evidence until the final U.S. validation scope is fixed.",
            "- If Item 1.03 rows are present, the data can support real U.S. bankruptcy/receivership sequence analysis beyond the targeted case sample.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scan(sec_root: Path, root: Path, prefix: str, start_year: int, end_year: int, max_files: int | None, refresh: bool) -> dict[str, Any]:
    paths = output_paths(root, prefix)
    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["status_json"].parent.mkdir(parents=True, exist_ok=True)

    if not refresh and max_files is None and paths["status_json"].exists() and paths["filings_csv"].exists():
        status = json.loads(paths["status_json"].read_text(encoding="utf-8-sig"))
        status.pop("source_root", None)
        status["source_scope"] = "external_local_sec_cache"
        status["source_root_redacted"] = True
        status["preserved_existing_collection"] = True
        paths["status_json"].write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        write_markdown(status, paths["status_md"])
        return status

    files = sorted(sec_root.glob("CIK*_submissions.json")) if sec_root.exists() else []
    if max_files is not None:
        files = files[:max_files]

    rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in files:
        try:
            flat, meta = flatten_submission(path, start_year, end_year)
            rows.extend(flat)
            file_rows.append(meta)
        except Exception as exc:
            failures.append({"path": str(path), "error": repr(exc)[:500]})

    filings = pd.DataFrame(rows)
    if filings.empty:
        filings = pd.DataFrame(
            columns=[
                "cik",
                "entity_name",
                "tickers",
                "exchanges",
                "sic",
                "accession_number",
                "filing_date",
                "report_date",
                "acceptance_datetime",
                "form",
                "items",
                "primary_document",
                "primary_doc_description",
                "source_path",
                "source_dataset",
                "has_target_item",
                *[f"has_item_{item.replace('.', '_')}" for item in TARGET_ITEMS],
            ]
        )
    else:
        filings = filings.drop_duplicates(["cik", "accession_number"], keep="first")
        filings = filings.sort_values(["cik", "filing_date", "accession_number"])

    outcome_candidates = filings[filings["has_target_item"]].copy() if not filings.empty else filings.copy()
    file_summary = pd.DataFrame(file_rows)
    if file_summary.empty:
        file_summary = pd.DataFrame(columns=["cik", "entity_name", "source_path", "recent_rows", "historical_file_references", "filing_rows_kept"])

    filings.to_csv(paths["filings_csv"], index=False, encoding="utf-8-sig")
    outcome_candidates.to_csv(paths["outcome_csv"], index=False, encoding="utf-8-sig")
    file_summary.to_csv(paths["file_summary_csv"], index=False, encoding="utf-8-sig")
    try:
        filings.to_parquet(paths["filings_parquet"], index=False)
    except Exception:
        pass

    counts = item_counts(filings)
    dates = pd.to_datetime(filings["filing_date"], errors="coerce") if not filings.empty else pd.Series(dtype="datetime64[ns]")
    status = {
        "status": "collected" if files else "source_missing_or_empty",
        "source_scope": "external_local_sec_cache",
        "source_root_redacted": True,
        "prefix": prefix,
        "start_year": start_year,
        "end_year": end_year,
        "submission_files_found": int(len(files)),
        "submission_files_parsed": int(len(file_rows)),
        "archive_file_references": int(file_summary["historical_file_references"].sum()) if not file_summary.empty else 0,
        "archive_files_present": int(file_summary["historical_files_present"].sum()) if "historical_files_present" in file_summary else 0,
        "archive_filing_rows_kept": int(file_summary["historical_filing_rows_kept"].sum()) if "historical_filing_rows_kept" in file_summary else 0,
        "parse_failures": int(len(failures)),
        "failures_sample": failures[:20],
        "filing_rows": int(len(filings)),
        "filing_firms": int(filings["cik"].nunique()) if not filings.empty else 0,
        "outcome_candidate_rows": int(len(outcome_candidates)),
        "outcome_candidate_firms": int(outcome_candidates["cik"].nunique()) if not outcome_candidates.empty else 0,
        "date_min": dates.min().date().isoformat() if not dates.empty and not pd.isna(dates.min()) else "",
        "date_max": dates.max().date().isoformat() if not dates.empty and not pd.isna(dates.max()) else "",
        "item_counts": {str(k): int(v) for k, v in counts.items()},
        "filings_csv": str(paths["filings_csv"]),
        "filings_parquet": str(paths["filings_parquet"]),
        "outcome_csv": str(paths["outcome_csv"]),
        "file_summary_csv": str(paths["file_summary_csv"]),
        "note": "Local SEC submissions metadata scan with cached archive files where available; no SEC API calls were made.",
    }
    paths["status_json"].write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(status, paths["status_md"])
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sec-root", default=str(DEFAULT_SEC_ROOT))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--prefix", default="sec_desktop_population")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    status = run_scan(Path(args.sec_root), Path(args.root), args.prefix, args.start_year, args.end_year, args.max_files, args.refresh)
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
