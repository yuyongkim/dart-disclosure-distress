from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("Paper 4")
DEFAULT_BASE_URL = "https://api.edinet-fsa.go.jp/api/v2"
OFFICIAL_DOC_URL = "https://disclosure2dl.edinet-fsa.go.jp/guide/static/disclosure/WZEK0110.html"


def date_range(start: str, end: str) -> list[pd.Timestamp]:
    dates = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")
    return [pd.Timestamp(value) for value in dates]


def output_paths(root: Path) -> dict[str, Path]:
    return {
        "raw_dir": root / "data" / "raw" / "edinet",
        "documents_dir": root / "data" / "raw" / "edinet" / "documents",
        "plan_csv": root / "data" / "raw" / "edinet" / "edinet_collection_plan.csv",
        "index_csv": root / "data" / "raw" / "edinet" / "edinet_documents_index.csv",
        "index_parquet": root / "data" / "raw" / "edinet" / "edinet_documents_index.parquet",
        "status_json": root / "results" / "edinet_collection_status.json",
        "status_md": root / "results" / "edinet_collection_status.md",
    }


def build_plan(start_date: str, end_date: str) -> pd.DataFrame:
    rows = []
    for value in date_range(start_date, end_date):
        rows.append(
            {
                "date": value.date().isoformat(),
                "year": int(value.year),
                "month": int(value.month),
                "api_endpoint": "/documents.json",
                "query_type": 2,
                "purpose": "EDINET document-list index for temporary reports and outcome screening",
            }
        )
    return pd.DataFrame(rows)


def fetch_document_list(base_url: str, api_key: str, target_date: str, timeout: int) -> dict[str, Any]:
    query = urllib.parse.urlencode({"date": target_date, "type": 2, "Subscription-Key": api_key})
    url = f"{base_url.rstrip('/')}/documents.json?{query}"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def flatten_document_rows(payload: dict[str, Any], target_date: str, raw_path: Path) -> list[dict[str, Any]]:
    results = payload.get("results") or []
    rows = []
    for item in results:
        description = str(item.get("docDescription") or "")
        submitter = str(item.get("filerName") or "")
        ordinance = str(item.get("ordinanceCode") or "")
        form_code = str(item.get("formCode") or "")
        rows.append(
            {
                "date": target_date,
                "doc_id": item.get("docID"),
                "edinet_code": item.get("edinetCode"),
                "sec_code": item.get("secCode"),
                "jcn": item.get("JCN"),
                "filer_name": submitter,
                "fund_code": item.get("fundCode"),
                "ordinance_code": ordinance,
                "form_code": form_code,
                "doc_type_code": item.get("docTypeCode"),
                "period_start": item.get("periodStart"),
                "period_end": item.get("periodEnd"),
                "submit_date_time": item.get("submitDateTime"),
                "doc_description": description,
                "issuer_edinet_code": item.get("issuerEdinetCode"),
                "subject_edinet_code": item.get("subjectEdinetCode"),
                "subsidiary_edinet_code": item.get("subsidiaryEdinetCode"),
                "current_report_candidate": "臨時報告書" in description or "訂正臨時報告書" in description,
                "raw_json_path": str(raw_path),
            }
        )
    return rows


def run_collection(
    start_date: str,
    end_date: str,
    execute: bool,
    api_key: str,
    base_url: str,
    root: Path,
    max_days: int | None,
    sleep_seconds: float,
    timeout: int,
) -> dict[str, Any]:
    paths = output_paths(root)
    paths["raw_dir"].mkdir(parents=True, exist_ok=True)
    paths["documents_dir"].mkdir(parents=True, exist_ok=True)
    paths["status_json"].parent.mkdir(parents=True, exist_ok=True)

    plan = build_plan(start_date, end_date)
    if max_days is not None:
        plan = plan.head(max_days).copy()
    plan.to_csv(paths["plan_csv"], index=False, encoding="utf-8-sig")

    status = {
        "status": "dry_run_plan_only",
        "execute": bool(execute),
        "api_key_present": bool(api_key),
        "api_base_url": base_url,
        "official_doc_url": OFFICIAL_DOC_URL,
        "start_date": start_date,
        "end_date": end_date,
        "planned_days": int(len(plan)),
        "plan_csv": str(paths["plan_csv"]),
        "index_csv": str(paths["index_csv"]),
        "note": "EDINET remains Phase 2 unless an EDINET_API_KEY is configured and collection is explicitly executed.",
    }

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if execute:
        if not api_key:
            status["status"] = "blocked_missing_api_key"
            status["note"] = "Set EDINET_API_KEY and rerun with --execute to collect EDINET document-list metadata."
        else:
            for row in plan.itertuples(index=False):
                target_date = str(row.date)
                raw_path = paths["documents_dir"] / f"{target_date}.json"
                try:
                    payload = fetch_document_list(base_url, api_key, target_date, timeout)
                    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                    rows.extend(flatten_document_rows(payload, target_date, raw_path))
                    if sleep_seconds:
                        time.sleep(sleep_seconds)
                except Exception as exc:  # network/API failures should be captured, not crash long collection.
                    failures.append({"date": target_date, "error": repr(exc)})
            status["status"] = "collected" if not failures else "collected_with_failures"

    index = pd.DataFrame(rows)
    if index.empty:
        index = pd.DataFrame(
            columns=[
                "date",
                "doc_id",
                "edinet_code",
                "sec_code",
                "jcn",
                "filer_name",
                "form_code",
                "doc_type_code",
                "submit_date_time",
                "doc_description",
                "current_report_candidate",
                "raw_json_path",
            ]
        )
    index.to_csv(paths["index_csv"], index=False, encoding="utf-8-sig")
    try:
        index.to_parquet(paths["index_parquet"], index=False)
    except Exception as exc:
        status["parquet_error"] = repr(exc)

    status.update(
        {
            "raw_json_files": int(len(list(paths["documents_dir"].glob("*.json")))),
            "document_rows": int(len(index)),
            "current_report_candidate_rows": int(index["current_report_candidate"].sum()) if "current_report_candidate" in index.columns and not index.empty else 0,
            "failures": failures[:20],
            "failure_count": int(len(failures)),
        }
    )
    paths["status_json"].write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(status, paths["status_md"])
    return status


def write_markdown(status: dict[str, Any], path: Path) -> None:
    lines = [
        "# EDINET Collection Status",
        "",
        "This file documents the EDINET Phase 2 collection state for Paper 4.",
        "",
        "## Status",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Status | `{status['status']}` |",
        f"| Execute requested | `{status['execute']}` |",
        f"| API key present | `{status['api_key_present']}` |",
        f"| Date range | {status['start_date']} to {status['end_date']} |",
        f"| Planned days | {status['planned_days']} |",
        f"| Document rows collected | {status['document_rows']} |",
        f"| Current-report candidate rows | {status['current_report_candidate_rows']} |",
        f"| Raw JSON files | {status['raw_json_files']} |",
        f"| Failure count | {status['failure_count']} |",
        "",
        "## Files",
        "",
        f"- Collection plan: `{status['plan_csv']}`",
        f"- Document index: `{status['index_csv']}`",
        f"- Official guide page: {status['official_doc_url']}",
        "",
        "## Interpretation",
        "",
    ]
    if status["status"] == "blocked_missing_api_key":
        lines.append("- EDINET collection was explicitly requested but no `EDINET_API_KEY` was available.")
    elif status["status"] == "dry_run_plan_only":
        lines.append("- No EDINET data were collected in this run. The script generated a reproducible date-level collection plan only.")
    elif status["status"].startswith("collected"):
        lines.append("- EDINET document-list metadata were collected. Downstream normalization should filter current-report candidates and map them into the T01-T06 taxonomy.")
    lines.extend(
        [
            "- Keep EDINET out of Paper 4 v1 claims unless actual document metadata and outcome labels are collected and normalized.",
            "- EDINET can become v2 evidence after `edinet_documents_index.csv` has nonzero current-report candidate rows and an event mapper is implemented.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--api-key", default=os.getenv("EDINET_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("EDINET_API_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()

    status = run_collection(
        start_date=args.start_date,
        end_date=args.end_date,
        execute=args.execute,
        api_key=args.api_key,
        base_url=args.base_url,
        root=Path(args.root),
        max_days=args.max_days,
        sleep_seconds=args.sleep_seconds,
        timeout=args.timeout,
    )
    print(json.dumps({k: status[k] for k in ("status", "planned_days", "document_rows", "current_report_candidate_rows", "api_key_present")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
