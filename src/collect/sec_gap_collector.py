from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("Paper 4")
PAPER3_RAW = Path("Paper3") / "dart_apriori" / "data" / "raw"
DEFAULT_BASE_URL = "https://data.sec.gov"
OFFICIAL_DOC_URL = "https://www.sec.gov/search-filings/edgar-application-programming-interfaces"
TARGET_ITEMS = ("1.03", "2.04", "3.01", "3.03")


def normalize_cik(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else ""


def read_targets(path: Path) -> pd.DataFrame:
    targets = pd.read_csv(path, low_memory=False)
    if "cik" not in targets.columns:
        raise ValueError(f"Target file lacks cik column: {path}")
    targets = targets.copy()
    targets["cik10"] = targets["cik"].map(normalize_cik)
    targets = targets[targets["cik10"].ne("")].drop_duplicates("cik10")
    return targets.sort_values("cik10")


def output_paths(root: Path) -> dict[str, Path]:
    base = root / "data" / "raw" / "sec_expanded"
    return {
        "base": base,
        "submissions_dir": base / "submissions",
        "plan_csv": base / "sec_expanded_collection_plan.csv",
        "filings_csv": base / "sec_expanded_8k_filings.csv",
        "filings_parquet": base / "sec_expanded_8k_filings.parquet",
        "outcome_csv": base / "sec_expanded_outcome_candidates.csv",
        "status_json": root / "results" / "sec_expanded_collection_status.json",
        "status_md": root / "results" / "sec_expanded_collection_status.md",
    }


def build_plan(targets: pd.DataFrame, base_url: str, start_year: int, end_year: int) -> pd.DataFrame:
    rows = []
    for row in targets.itertuples(index=False):
        ticker = getattr(row, "ticker", "")
        title = getattr(row, "title", getattr(row, "company_name", ""))
        rows.append(
            {
                "cik": row.cik10,
                "ticker": ticker,
                "company_name": title,
                "sic": getattr(row, "sic", ""),
                "exchange": getattr(row, "exchange", ""),
                "api_endpoint": f"/submissions/CIK{row.cik10}.json",
                "api_url": f"{base_url.rstrip('/')}/submissions/CIK{row.cik10}.json",
                "start_year": start_year,
                "end_year": end_year,
                "target_items": ",".join(TARGET_ITEMS),
            }
        )
    return pd.DataFrame(rows)


def fetch_json(url: str, user_agent: str, timeout: int) -> dict[str, Any]:
    host = urllib.parse.urlparse(url).netloc or "data.sec.gov"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": host,
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        encoding = response.headers.get("Content-Encoding", "").lower()
        if "gzip" in encoding or raw.startswith(b"\x1f\x8b"):
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))


def records_from_filings_dict(filings: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(filings.keys())
    if not keys:
        return []
    n = max(len(filings.get(key, [])) for key in keys)
    rows = []
    for i in range(n):
        row = {}
        for key in keys:
            values = filings.get(key, [])
            row[key] = values[i] if i < len(values) else None
        rows.append(row)
    return rows


def split_items(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def flatten_submission(
    payload: dict[str, Any],
    source_url: str,
    start_year: int,
    end_year: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    cik = normalize_cik(payload.get("cik"))
    entity_name = str(payload.get("name") or "")
    tickers = payload.get("tickers") or []
    sic = payload.get("sic") or ""
    rows = []
    historical_files = []
    recent = payload.get("filings", {}).get("recent", {})
    for filing in records_from_filings_dict(recent):
        form = str(filing.get("form") or "")
        filing_date = pd.to_datetime(filing.get("filingDate"), errors="coerce")
        if form not in {"8-K", "8-K/A"} or pd.isna(filing_date) or not (start_year <= filing_date.year <= end_year):
            continue
        items = split_items(filing.get("items"))
        rows.append(
            {
                "cik": cik,
                "entity_name": entity_name,
                "tickers": ",".join(map(str, tickers)),
                "sic": sic,
                "accession_number": filing.get("accessionNumber"),
                "filing_date": filing_date.date().isoformat(),
                "report_date": filing.get("reportDate"),
                "form": form,
                "items": ",".join(items),
                "primary_document": filing.get("primaryDocument"),
                "primary_doc_description": filing.get("primaryDocDescription"),
                "source_url": source_url,
                "has_target_item": any(item in TARGET_ITEMS for item in items),
                **{f"has_item_{item.replace('.', '_')}": item in items for item in TARGET_ITEMS},
            }
        )
    for item in payload.get("filings", {}).get("files", []) or []:
        name = item.get("name")
        if name:
            historical_files.append(str(name))
    return rows, historical_files


def run_collection(
    target_path: Path,
    start_year: int,
    end_year: int,
    execute: bool,
    user_agent: str,
    base_url: str,
    root: Path,
    max_ciks: int | None,
    include_historical_files: bool,
    sleep_seconds: float,
    timeout: int,
    force_plan_only: bool,
) -> dict[str, Any]:
    paths = output_paths(root)
    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["submissions_dir"].mkdir(parents=True, exist_ok=True)
    paths["status_json"].parent.mkdir(parents=True, exist_ok=True)

    if not execute and not force_plan_only and paths["status_json"].exists() and paths["filings_csv"].exists():
        try:
            existing_status = json.loads(paths["status_json"].read_text(encoding="utf-8"))
            if str(existing_status.get("status", "")).startswith("collected") and int(existing_status.get("filing_rows", 0)) > 0:
                existing_status["preserved_existing_collection"] = True
                paths["status_json"].write_text(json.dumps(existing_status, indent=2, ensure_ascii=False), encoding="utf-8")
                write_markdown(existing_status, paths["status_md"])
                return existing_status
        except Exception:
            pass

    targets = read_targets(target_path)
    if max_ciks is not None:
        targets = targets.head(max_ciks).copy()
    plan = build_plan(targets, base_url, start_year, end_year)
    plan.to_csv(paths["plan_csv"], index=False, encoding="utf-8-sig")

    status = {
        "status": "dry_run_plan_only",
        "execute": bool(execute),
        "user_agent_present": bool(user_agent),
        "official_doc_url": OFFICIAL_DOC_URL,
        "base_url": base_url,
        "target_path": str(target_path),
        "planned_ciks": int(len(plan)),
        "start_year": start_year,
        "end_year": end_year,
        "target_items": list(TARGET_ITEMS),
        "plan_csv": str(paths["plan_csv"]),
        "filings_csv": str(paths["filings_csv"]),
        "outcome_csv": str(paths["outcome_csv"]),
        "note": "SEC expanded collection remains Phase 2 unless SEC_USER_AGENT is configured and collection is explicitly executed.",
    }

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if execute:
        if not user_agent:
            status["status"] = "blocked_missing_user_agent"
            status["note"] = "Set SEC_USER_AGENT with contact information and rerun with --execute."
        else:
            for plan_row in plan.itertuples(index=False):
                url = str(plan_row.api_url)
                cik = str(plan_row.cik)
                raw_path = paths["submissions_dir"] / f"CIK{cik}.json"
                try:
                    payload = fetch_json(url, user_agent, timeout)
                    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                    flat, historical = flatten_submission(payload, url, start_year, end_year)
                    rows.extend(flat)
                    if include_historical_files:
                        for filename in historical:
                            hist_url = f"{base_url.rstrip('/')}/submissions/{filename}"
                            hist_path = paths["submissions_dir"] / filename
                            hist_payload = fetch_json(hist_url, user_agent, timeout)
                            hist_path.write_text(json.dumps(hist_payload, indent=2, ensure_ascii=False), encoding="utf-8")
                            hist_rows, _ = flatten_submission(hist_payload, hist_url, start_year, end_year)
                            for item in hist_rows:
                                item["cik"] = cik
                            rows.extend(hist_rows)
                            if sleep_seconds:
                                time.sleep(sleep_seconds)
                    if sleep_seconds:
                        time.sleep(sleep_seconds)
                except Exception as exc:
                    failures.append({"cik": cik, "url": url, "error": repr(exc)[:1000]})
            status["status"] = "collected" if not failures else "collected_with_failures"

    filings = pd.DataFrame(rows)
    if filings.empty:
        filings = pd.DataFrame(
            columns=[
                "cik",
                "entity_name",
                "tickers",
                "sic",
                "accession_number",
                "filing_date",
                "form",
                "items",
                "has_target_item",
                *[f"has_item_{item.replace('.', '_')}" for item in TARGET_ITEMS],
            ]
        )
    outcome = filings[filings.get("has_target_item", pd.Series(dtype=bool)).fillna(False)].copy()
    filings.to_csv(paths["filings_csv"], index=False, encoding="utf-8-sig")
    outcome.to_csv(paths["outcome_csv"], index=False, encoding="utf-8-sig")
    try:
        filings.to_parquet(paths["filings_parquet"], index=False)
    except Exception as exc:
        status["parquet_error"] = repr(exc)

    item_counts = {}
    for item in TARGET_ITEMS:
        col = f"has_item_{item.replace('.', '_')}"
        item_counts[item] = int(filings[col].sum()) if col in filings.columns and not filings.empty else 0
    status.update(
        {
            "raw_submission_files": int(len(list(paths["submissions_dir"].glob("*.json")))),
            "filing_rows": int(len(filings)),
            "outcome_candidate_rows": int(len(outcome)),
            "item_counts": item_counts,
            "failure_count": int(len(failures)),
            "failures": failures[:20],
        }
    )
    paths["status_json"].write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(status, paths["status_md"])
    return status


def write_markdown(status: dict[str, Any], path: Path) -> None:
    lines = [
        "# SEC Expanded Collection Status",
        "",
        "This file documents the Phase 2 SEC expansion path for Paper 4. The goal is to collect a broader Form 8-K universe centered on Item 1.03, 2.04, 3.01, and 3.03 outcome candidates.",
        "",
        "## Status",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Status | `{status['status']}` |",
        f"| Execute requested | `{status['execute']}` |",
        f"| SEC user agent present | `{status['user_agent_present']}` |",
        f"| Planned CIKs | {status['planned_ciks']} |",
        f"| Years | {status['start_year']} to {status['end_year']} |",
        f"| Filing rows collected | {status['filing_rows']} |",
        f"| Outcome candidate rows | {status['outcome_candidate_rows']} |",
        f"| Raw submission files | {status['raw_submission_files']} |",
        f"| Failure count | {status['failure_count']} |",
        "",
        "## Target Item Counts",
        "",
        "| Item | Rows |",
        "|---|---:|",
    ]
    for item, count in status["item_counts"].items():
        lines.append(f"| `{item}` | {count} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Collection plan: `{status['plan_csv']}`",
            f"- Filing index: `{status['filings_csv']}`",
            f"- Outcome candidates: `{status['outcome_csv']}`",
            f"- Official SEC APIs page: {status['official_doc_url']}",
            "",
            "## Interpretation",
            "",
        ]
    )
    if status["status"] == "dry_run_plan_only":
        lines.append("- No SEC submissions were collected in this run. The script generated a reproducible CIK-level collection plan only.")
    elif status["status"] == "blocked_missing_user_agent":
        lines.append("- SEC collection was explicitly requested but no `SEC_USER_AGENT` was available.")
    elif status["status"].startswith("collected"):
        lines.append("- SEC submissions metadata were collected. Downstream processing should merge these rows with the existing 8-K item parser and rebuild SEC outcome labels.")
        if status.get("item_counts", {}).get("1.03", 0) == 0:
            lines.append("- Item 1.03 remains absent in the collected 2015-2024 metadata, so this does not yet support a bankruptcy-centered U.S. validation claim.")
    lines.extend(
        [
            "- Keep standalone SEC prediction out of Paper 4 v1 unless the expanded data are normalized, validated, and judged sufficient for the exact U.S. outcome definition.",
            "- SEC fair-access practice requires an identifying user agent with contact information.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-path", default=str(PAPER3_RAW / "edgar_target_ciks_expanded.csv"))
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-agent", default=os.getenv("SEC_USER_AGENT", ""))
    parser.add_argument("--base-url", default=os.getenv("SEC_DATA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--max-ciks", type=int, default=None)
    parser.add_argument("--no-historical-files", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.12)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--force-plan-only", action="store_true")
    args = parser.parse_args()

    status = run_collection(
        target_path=Path(args.target_path),
        start_year=args.start_year,
        end_year=args.end_year,
        execute=args.execute,
        user_agent=args.user_agent,
        base_url=args.base_url,
        root=Path(args.root),
        max_ciks=args.max_ciks,
        include_historical_files=not args.no_historical_files,
        sleep_seconds=args.sleep_seconds,
        timeout=args.timeout,
        force_plan_only=args.force_plan_only,
    )
    print(
        json.dumps(
            {
                "status": status["status"],
                "planned_ciks": status["planned_ciks"],
                "filing_rows": status["filing_rows"],
                "outcome_candidate_rows": status["outcome_candidate_rows"],
                "user_agent_present": status["user_agent_present"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
