from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("Paper 4")
DEFAULT_BASE_URL = "https://api.edinet-fsa.go.jp/api/v2"
OFFICIAL_DOC_URL = "https://disclosure2dl.edinet-fsa.go.jp/guide/static/disclosure/WZEK0110.html"


def output_paths(root: Path) -> dict[str, Path]:
    return {
        "feasibility_csv": root / "data" / "raw" / "edinet" / "edinet_collection_plan_feasibility.csv",
        "status_json": root / "results" / "edinet_api_feasibility.json",
        "status_md": root / "results" / "edinet_api_feasibility.md",
    }


def build_feasibility(plan_path: Path, as_of_date: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    plan = pd.read_csv(plan_path, low_memory=False)
    plan["date"] = pd.to_datetime(plan["date"], errors="coerce")
    as_of = pd.Timestamp(as_of_date)
    approximate_cutoff = as_of - pd.DateOffset(years=10)
    plan["api_10y_window_feasible"] = plan["date"].ge(approximate_cutoff)
    plan["api_window_note"] = plan["api_10y_window_feasible"].map(
        {
            True: "within approximate 10-year API date window",
            False: "outside approximate 10-year API date window; needs archive/prior collection",
        }
    )
    summary = {
        "as_of_date": as_of.date().isoformat(),
        "approximate_10y_cutoff": approximate_cutoff.date().isoformat(),
        "planned_days": int(len(plan)),
        "api_window_feasible_days": int(plan["api_10y_window_feasible"].sum()),
        "api_window_infeasible_days": int((~plan["api_10y_window_feasible"]).sum()),
        "planned_start_date": str(plan["date"].min().date()) if not plan.empty else None,
        "planned_end_date": str(plan["date"].max().date()) if not plan.empty else None,
        "official_doc_url": OFFICIAL_DOC_URL,
    }
    return plan, summary


def no_key_preflight(base_url: str, target_date: str, timeout: int) -> dict[str, Any]:
    query = urllib.parse.urlencode({"date": target_date, "type": 2})
    url = f"{base_url.rstrip('/')}/documents.json?{query}"
    result: dict[str, Any] = {
        "url_without_subscription_key": url,
        "target_date": target_date,
        "status": "not_run",
        "http_status": None,
        "error": None,
        "body_excerpt": "",
    }
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            body_status = None
            try:
                parsed = json.loads(body)
                body_status = parsed.get("StatusCode")
            except Exception:
                parsed = {}
            api_blocked = str(body_status) == "401"
            result.update(
                {
                    "status": "blocked_without_subscription_key" if api_blocked else "unexpected_success_without_key",
                    "http_status": int(response.status),
                    "body_status_code": body_status,
                    "body_message": str(parsed.get("message", ""))[:500] if isinstance(parsed, dict) else "",
                    "body_excerpt": body[:500],
                }
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        result.update(
            {
                "status": "blocked_without_subscription_key",
                "http_status": int(exc.code),
                "error": str(exc),
                "body_excerpt": body[:500],
            }
        )
    except Exception as exc:
        result.update({"status": "request_failed", "error": repr(exc)[:500]})
    return result


def write_markdown(status: dict[str, Any], path: Path) -> None:
    lines = [
        "# EDINET API Feasibility",
        "",
        "This file records EDINET API feasibility constraints for Paper 4. It is separate from actual document-list collection.",
        "",
        "## Date-Window Feasibility",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| As-of date | {status['as_of_date']} |",
        f"| Approximate 10-year cutoff | {status['approximate_10y_cutoff']} |",
        f"| Planned days | {status['planned_days']} |",
        f"| Days within approximate API date window | {status['api_window_feasible_days']} |",
        f"| Days outside approximate API date window | {status['api_window_infeasible_days']} |",
        "",
        "## No-Key Preflight",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Preflight status | `{status['preflight']['status']}` |",
        f"| HTTP status | `{status['preflight']['http_status']}` |",
        f"| Target date | {status['preflight']['target_date']} |",
        "",
        "## Interpretation",
        "",
        "- EDINET v2 document-list collection should be treated as Phase 2 until a valid subscription key is configured.",
        "- The 2015-2024 plan is reproducible, but live API backfill is constrained by the EDINET date-window rule; older dates outside the approximate 10-year window require archive data or prior collection.",
        "- Keep EDINET out of v1 empirical claims unless actual document rows are collected and normalized.",
        f"- Official EDINET guide page: {status['official_doc_url']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-path", default=str(ROOT / "data" / "raw" / "edinet" / "edinet_collection_plan.csv"))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--preflight-date", default="2024-01-04")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    root = Path(args.root)
    paths = output_paths(root)
    paths["feasibility_csv"].parent.mkdir(parents=True, exist_ok=True)
    paths["status_json"].parent.mkdir(parents=True, exist_ok=True)

    feasibility, summary = build_feasibility(Path(args.plan_path), args.as_of_date)
    feasibility.to_csv(paths["feasibility_csv"], index=False, encoding="utf-8-sig")
    summary["feasibility_csv"] = str(paths["feasibility_csv"])
    summary["preflight"] = no_key_preflight(args.base_url, args.preflight_date, args.timeout)

    paths["status_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, paths["status_md"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
