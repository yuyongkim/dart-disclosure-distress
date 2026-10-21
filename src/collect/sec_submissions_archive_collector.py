from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path("Paper 4")
DEFAULT_SEC_ROOT = Path.home() / "Desktop" / "M&A" / "data" / "raw" / "sec"
ARCHIVE_BASE_URL = "https://data.sec.gov/submissions"


def normalize_cik(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else ""


def overlaps_window(filing_from: object, filing_to: object, start_year: int, end_year: int) -> bool:
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-12-31")
    left = pd.to_datetime(filing_from, errors="coerce")
    right = pd.to_datetime(filing_to, errors="coerce")
    if pd.isna(left) or pd.isna(right):
        return False
    return bool(left <= end and right >= start)


def build_manifest(sec_root: Path, start_year: int, end_year: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(sec_root.glob("CIK*_submissions.json")) if sec_root.exists() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append(
                {
                    "cik": "",
                    "entity_name": "",
                    "parent_path": str(path),
                    "archive_name": "",
                    "status": "parent_parse_error",
                    "error": repr(exc)[:500],
                }
            )
            continue
        cik = normalize_cik(payload.get("cik"))
        entity_name = str(payload.get("name") or "")
        for ref in payload.get("filings", {}).get("files", []) or []:
            name = str(ref.get("name") or "").strip()
            if not name:
                continue
            local_path = sec_root / name
            rows.append(
                {
                    "cik": cik,
                    "entity_name": entity_name,
                    "parent_path": str(path),
                    "archive_name": name,
                    "filing_count": int(ref.get("filingCount") or 0),
                    "filing_from": ref.get("filingFrom"),
                    "filing_to": ref.get("filingTo"),
                    "overlaps_target_window": overlaps_window(ref.get("filingFrom"), ref.get("filingTo"), start_year, end_year),
                    "archive_url": f"{ARCHIVE_BASE_URL}/{name}",
                    "local_path": str(local_path),
                    "local_exists": local_path.exists(),
                    "local_bytes": int(local_path.stat().st_size) if local_path.exists() else 0,
                    "status": "manifested",
                    "error": "",
                }
            )
    return pd.DataFrame(rows)


def refresh_local_state(manifest: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.copy()
    if manifest.empty or "local_path" not in manifest:
        return manifest
    exists: list[bool] = []
    sizes: list[int] = []
    for value in manifest["local_path"].fillna("").astype(str):
        path = Path(value)
        ok = path.exists()
        exists.append(ok)
        sizes.append(int(path.stat().st_size) if ok else 0)
    manifest["local_exists"] = exists
    manifest["local_bytes"] = sizes
    return manifest


def download_one(url: str, path: Path, user_agent: str, timeout: int = 30) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
    started = time.time()
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
        path.write_bytes(data)
        return {
            "status": "downloaded",
            "url": url,
            "path": str(path),
            "bytes": len(data),
            "seconds": round(time.time() - started, 3),
        }
    except HTTPError as exc:
        return {"status": "http_error", "url": url, "path": str(path), "code": exc.code, "error": str(exc)[:500]}
    except URLError as exc:
        return {"status": "url_error", "url": url, "path": str(path), "error": repr(exc)[:500]}
    except Exception as exc:
        return {"status": "error", "url": url, "path": str(path), "error": repr(exc)[:500]}


def append_log(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    new_log = pd.DataFrame(rows)
    if path.exists():
        try:
            old_log = pd.read_csv(path, low_memory=False)
            new_log = pd.concat([old_log, new_log], ignore_index=True, sort=False)
        except Exception:
            pass
    new_log.to_csv(path, index=False, encoding="utf-8-sig")


def write_markdown(status: dict[str, Any], path: Path) -> None:
    lines = [
        "# SEC Submissions Archive Coverage",
        "",
        "This file audits and optionally downloads archived SEC submissions JSON files referenced by each local company submissions JSON.",
        "",
        "## Status",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Status | `{status['status']}` |",
        f"| Source scope | `{status.get('source_scope', 'external_local_sec_cache')}` |",
        f"| Archive references | {status['archive_references']} |",
        f"| References overlapping target window | {status['overlap_references']} |",
        f"| Overlap references cached | {status['overlap_references_cached']} |",
        f"| Overlap reference cache rate | {status['overlap_reference_cache_rate']} |",
        f"| Overlap referenced filing count | {status['overlap_referenced_filing_count']} |",
        f"| Execute downloads | {status['execute']} |",
        f"| Download attempts | {status['download_attempts']} |",
        f"| Downloaded files | {status['downloaded_files']} |",
        f"| User agent present | {status['user_agent_present']} |",
        "",
        "## Guardrails",
        "",
        "- The default command builds/refreshes the archive-reference manifest only.",
        "- Use `--execute --max-downloads N` only with a real identifying `SEC_USER_AGENT`.",
        "- These archive JSON files improve submissions metadata coverage; they are not filing-body text files.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sec-root", default=str(DEFAULT_SEC_ROOT))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "raw" / "sec_submissions_archive"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-downloads", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    args = parser.parse_args()

    sec_root = Path(args.sec_root)
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(sec_root, args.start_year, args.end_year)
    manifest_path = out_dir / "sec_submissions_archive_manifest.csv"
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    downloads: list[dict[str, Any]] = []
    if args.execute:
        if not user_agent:
            downloads.append({"status": "blocked_missing_sec_user_agent"})
        else:
            todo = manifest[manifest["overlaps_target_window"] & ~manifest["local_exists"]].copy()
            todo = todo.sort_values(["filing_to", "filing_from", "cik", "archive_name"], kind="mergesort")
            todo = todo.head(args.max_downloads if args.max_downloads > 0 else len(todo))
            for row in todo.to_dict(orient="records"):
                result = download_one(str(row["archive_url"]), Path(str(row["local_path"])), user_agent)
                result["cik"] = row["cik"]
                result["archive_name"] = row["archive_name"]
                result["filing_from"] = row["filing_from"]
                result["filing_to"] = row["filing_to"]
                downloads.append(result)
                time.sleep(args.sleep_seconds)
    append_log(downloads, out_dir / "sec_submissions_archive_download_log.csv")
    manifest = refresh_local_state(manifest)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    overlap = manifest[manifest["overlaps_target_window"]] if not manifest.empty else manifest
    downloaded_files = sum(1 for row in downloads if row.get("status") == "downloaded")
    overlap_count = int(len(overlap))
    overlap_cached = int(overlap["local_exists"].sum()) if not overlap.empty else 0
    status_value = "downloaded" if args.execute and downloaded_files else "download_not_completed" if args.execute else "manifest_created"
    if not args.execute and overlap_cached:
        status_value = "manifest_created_with_local_cache"
    status = {
        "status": status_value,
        "source_scope": "external_local_sec_cache",
        "sec_root_redacted": True,
        "start_year": int(args.start_year),
        "end_year": int(args.end_year),
        "archive_references": int(len(manifest)),
        "overlap_references": overlap_count,
        "overlap_references_cached": overlap_cached,
        "overlap_reference_cache_rate": float(overlap_cached / overlap_count) if overlap_count else None,
        "overlap_referenced_filing_count": int(overlap["filing_count"].sum()) if not overlap.empty else 0,
        "execute": bool(args.execute),
        "max_downloads": int(args.max_downloads),
        "download_attempts": int(len([row for row in downloads if row.get("status") != "blocked_missing_sec_user_agent"])),
        "downloaded_files": int(downloaded_files),
        "user_agent_present": bool(user_agent),
        "manifest_csv": str(manifest_path),
        "download_log_csv": str(out_dir / "sec_submissions_archive_download_log.csv"),
        "download_status_counts": {
            str(key): int(value)
            for key, value in pd.Series([row.get("status") for row in downloads]).value_counts().items()
        } if downloads else {},
        "guardrail": "Archived submissions metadata coverage, not filing-body text coverage.",
    }
    (results_dir / "sec_submissions_archive_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(status, results_dir / "sec_submissions_archive_status.md")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
