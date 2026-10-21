from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


SRC = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC / "collect"))

from sec_gap_collector import TARGET_ITEMS, fetch_json, flatten_submission, normalize_cik  # noqa: E402


ROOT = Path("Paper 4")
DEFAULT_BASE_URL = "https://data.sec.gov"
OFFICIAL_DOC_URL = "https://www.sec.gov/search-filings/edgar-application-programming-interfaces"


def output_paths(root: Path) -> dict[str, Path]:
    base = root / "data" / "raw" / "sec_bankruptcy_cases"
    return {
        "base": base,
        "submissions_dir": base / "submissions",
        "seed_csv": base / "sec_bankruptcy_case_seed.csv",
        "filings_csv": base / "sec_bankruptcy_case_filings.csv",
        "filings_parquet": base / "sec_bankruptcy_case_filings.parquet",
        "item103_csv": base / "sec_bankruptcy_case_item103.csv",
        "status_json": root / "results" / "sec_bankruptcy_case_status.json",
        "status_md": root / "results" / "sec_bankruptcy_case_status.md",
    }


def read_seed(path: Path) -> pd.DataFrame:
    seed = pd.read_csv(path, low_memory=False)
    if "cik" not in seed.columns:
        raise ValueError(f"Seed file lacks cik column: {path}")
    seed = seed.copy()
    seed["cik10"] = seed["cik"].map(normalize_cik)
    seed = seed[seed["cik10"].ne("")].drop_duplicates("cik10")
    return seed.sort_values(["expected_event_year", "case_label", "cik10"])


def existing_collected_status(paths: dict[str, Path]) -> dict[str, Any] | None:
    if not paths["status_json"].exists() or not paths["filings_csv"].exists():
        return None
    try:
        status = json.loads(paths["status_json"].read_text(encoding="utf-8"))
    except Exception:
        return None
    if str(status.get("status", "")).startswith("collected") and int(status.get("item103_rows", 0)) > 0:
        status["preserved_existing_collection"] = True
        paths["status_json"].write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        write_markdown(status, paths["status_md"])
        return status
    return None


def collect_cases(
    seed_path: Path,
    root: Path,
    start_year: int,
    end_year: int,
    user_agent: str,
    base_url: str,
    refresh: bool,
    sleep_seconds: float,
    timeout: int,
) -> dict[str, Any]:
    paths = output_paths(root)
    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["submissions_dir"].mkdir(parents=True, exist_ok=True)
    paths["status_json"].parent.mkdir(parents=True, exist_ok=True)

    if not refresh:
        existing = existing_collected_status(paths)
        if existing is not None:
            return existing

    seed = read_seed(seed_path)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    status: dict[str, Any] = {
        "status": "blocked_missing_user_agent" if not user_agent else "collecting",
        "official_doc_url": OFFICIAL_DOC_URL,
        "base_url": base_url,
        "seed_path": str(seed_path),
        "seed_cases": int(len(seed)),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "target_items": list(TARGET_ITEMS),
        "filings_csv": str(paths["filings_csv"]),
        "item103_csv": str(paths["item103_csv"]),
    }

    if user_agent:
        for seed_row in seed.itertuples(index=False):
            cik = str(seed_row.cik10)
            url = f"{base_url.rstrip('/')}/submissions/CIK{cik}.json"
            try:
                payload = fetch_json(url, user_agent, timeout)
                (paths["submissions_dir"] / f"CIK{cik}.json").write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                flat, historical = flatten_submission(payload, url, start_year, end_year)
                for item in flat:
                    item["seed_case_label"] = getattr(seed_row, "case_label", "")
                    item["seed_expected_event_year"] = getattr(seed_row, "expected_event_year", "")
                    item["seed_notes"] = getattr(seed_row, "notes", "")
                rows.extend(flat)
                for filename in historical:
                    hist_url = f"{base_url.rstrip('/')}/submissions/{filename}"
                    hist_payload = fetch_json(hist_url, user_agent, timeout)
                    (paths["submissions_dir"] / filename).write_text(
                        json.dumps(hist_payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    hist_rows, _ = flatten_submission(hist_payload, hist_url, start_year, end_year)
                    for item in hist_rows:
                        item["cik"] = cik
                        item["entity_name"] = payload.get("name") or item.get("entity_name", "")
                        item["tickers"] = ",".join(map(str, payload.get("tickers") or []))
                        item["sic"] = payload.get("sic") or item.get("sic", "")
                        item["seed_case_label"] = getattr(seed_row, "case_label", "")
                        item["seed_expected_event_year"] = getattr(seed_row, "expected_event_year", "")
                        item["seed_notes"] = getattr(seed_row, "notes", "")
                    rows.extend(hist_rows)
                    if sleep_seconds:
                        time.sleep(sleep_seconds)
                if sleep_seconds:
                    time.sleep(sleep_seconds)
            except Exception as exc:
                failures.append({"cik": cik, "case_label": getattr(seed_row, "case_label", ""), "error": repr(exc)[:1000]})
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
                "has_item_1_03",
                "seed_case_label",
            ]
        )
    item103 = filings[filings.get("has_item_1_03", pd.Series(dtype=bool)).fillna(False)].copy()
    filings.to_csv(paths["filings_csv"], index=False, encoding="utf-8-sig")
    item103.to_csv(paths["item103_csv"], index=False, encoding="utf-8-sig")
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
            "item103_rows": int(len(item103)),
            "item103_firms": int(item103["cik"].nunique()) if not item103.empty else 0,
            "target_candidate_rows": int(filings["has_target_item"].sum()) if "has_target_item" in filings.columns else 0,
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
        "# SEC Bankruptcy Case Collection Status",
        "",
        "This file documents a targeted SEC Item 1.03 case sample for Paper 4. It is designed to show real U.S. bankruptcy/receivership disclosure sequences separately from the 132-CIK SEC expansion and the broader local SEC population-style scan.",
        "",
        "## Status",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Status | `{status['status']}` |",
        f"| Seed cases | {status['seed_cases']} |",
        f"| Years | {status['start_year']} to {status['end_year']} |",
        f"| Filing rows collected | {status.get('filing_rows', 0)} |",
        f"| Item 1.03 rows | {status.get('item103_rows', 0)} |",
        f"| Item 1.03 firms | {status.get('item103_firms', 0)} |",
        f"| Target candidate rows | {status.get('target_candidate_rows', 0)} |",
        f"| Raw submission files | {status.get('raw_submission_files', 0)} |",
        f"| Failure count | {status.get('failure_count', 0)} |",
        "",
        "## Target Item Counts",
        "",
        "| Item | Rows |",
        "|---|---:|",
    ]
    for item, count in status.get("item_counts", {}).items():
        lines.append(f"| `{item}` | {count} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Filing index: `{status['filings_csv']}`",
            f"- Item 1.03 rows: `{status['item103_csv']}`",
            f"- Official SEC APIs page: {status['official_doc_url']}",
            "",
            "## Interpretation",
            "",
        ]
    )
    if status["status"] == "blocked_missing_user_agent":
        lines.append("- Collection requires `SEC_USER_AGENT` with contact information.")
    elif status["status"].startswith("collected"):
        lines.append("- The targeted case sample contains actual Item 1.03 rows and can support SEC sequence case examples.")
        lines.append("- This remains a selected case sample, not a population-representative U.S. bankruptcy validation panel.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-path", default=str(ROOT / "data" / "raw" / "sec_bankruptcy_cases" / "sec_bankruptcy_case_seed.csv"))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--user-agent", default=os.getenv("SEC_USER_AGENT", ""))
    parser.add_argument("--base-url", default=os.getenv("SEC_DATA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.12)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    status = collect_cases(
        seed_path=Path(args.seed_path),
        root=Path(args.root),
        start_year=args.start_year,
        end_year=args.end_year,
        user_agent=args.user_agent,
        base_url=args.base_url,
        refresh=args.refresh,
        sleep_seconds=args.sleep_seconds,
        timeout=args.timeout,
    )
    print(
        json.dumps(
            {
                "status": status["status"],
                "seed_cases": status["seed_cases"],
                "filing_rows": status.get("filing_rows", 0),
                "item103_rows": status.get("item103_rows", 0),
                "item103_firms": status.get("item103_firms", 0),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
