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
DEFAULT_FILINGS = ROOT / "data" / "raw" / "sec_desktop_population" / "sec_desktop_population_8k_filings.csv"
TEXT_ROOT = ROOT / "data" / "raw" / "sec_filing_text"
TARGET_ITEMS = ("1.03", "2.04", "3.01")
DOWNLOAD_ITEM_PRIORITY = ("1.03", "2.04", "3.01")


def normalize_cik(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else ""


def split_items(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()}


def accession_nodash(accession: object) -> str:
    return str(accession or "").replace("-", "").strip()


def archive_base_url(cik: str, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash(accession)}"


def local_text_path(text_root: Path, cik: str, accession: str) -> Path:
    return text_root / cik / f"{accession_nodash(accession)}.txt"


def item_download_priority(items: object, priority_items: tuple[str, ...] = DOWNLOAD_ITEM_PRIORITY) -> tuple[int, str]:
    values = split_items(items)
    for idx, item in enumerate(priority_items):
        if item in values:
            return idx, item
    return len(priority_items), ""


def refresh_local_text_state(manifest: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.copy()
    exists: list[bool] = []
    sizes: list[int] = []
    for value in manifest["local_text_path"]:
        path = Path(str(value))
        ok = path.exists()
        exists.append(ok)
        sizes.append(int(path.stat().st_size) if ok else 0)
    manifest["local_text_exists"] = exists
    manifest["local_text_bytes"] = sizes
    return manifest


def build_manifest(filings_path: Path, text_root: Path, target_items: tuple[str, ...], max_rows: int | None = None) -> pd.DataFrame:
    cols = ["cik", "entity_name", "accession_number", "filing_date", "form", "items", "primary_document", "primary_doc_description"]
    filings = pd.read_csv(filings_path, usecols=lambda col: col in cols, low_memory=False)
    filings["cik"] = filings["cik"].map(normalize_cik)
    item_sets = filings["items"].map(split_items)
    mask = item_sets.map(lambda values: bool(values.intersection(target_items)))
    filings = filings[mask & filings["cik"].ne("") & filings["accession_number"].notna()].copy()
    if max_rows is not None:
        filings = filings.head(max_rows).copy()
    rows: list[dict[str, Any]] = []
    for row in filings.itertuples(index=False):
        accession = str(row.accession_number)
        cik = str(row.cik)
        base = archive_base_url(cik, accession)
        local_path = local_text_path(text_root, cik, accession)
        primary_document = str(row.primary_document or "").strip()
        rows.append(
            {
                "cik": cik,
                "entity_name": row.entity_name,
                "accession_number": accession,
                "filing_date": row.filing_date,
                "form": row.form,
                "items": row.items,
                "primary_document": primary_document,
                "primary_doc_description": row.primary_doc_description,
                "archive_txt_url": f"{base}/{accession}.txt",
                "primary_doc_url": f"{base}/{primary_document}" if primary_document else "",
                "local_text_path": str(local_path),
                "local_text_exists": local_path.exists(),
                "local_text_bytes": int(local_path.stat().st_size) if local_path.exists() else 0,
            }
        )
    return pd.DataFrame(rows)


def download_one(url: str, path: Path, user_agent: str, timeout: int = 30) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_markdown(status: dict[str, Any], path: Path) -> None:
    lines = [
        "# SEC Filing Text Manifest",
        "",
        "This file creates SEC Archives URLs for target-item 8-K filings found in the local SEC metadata scan.",
        "",
        "## Status",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Status | `{status['status']}` |",
        f"| Manifest rows | {status['manifest_rows']} |",
        f"| Local text files already present | {status['local_text_files']} |",
        f"| Local text bytes | {status['local_text_bytes']} |",
        f"| Execute downloads | {status['execute']} |",
        f"| Download attempts | {status['download_attempts']} |",
        f"| Downloaded files | {status['downloaded_files']} |",
        f"| User agent present | {status['user_agent_present']} |",
        "",
        "## Guardrails",
        "",
        "- The default command builds a manifest only and does not call SEC Archives.",
        "- Use `--execute --max-downloads N` only with a real identifying `SEC_USER_AGENT`.",
        "- Downloaded files should be cached under `data/raw/sec_filing_text/` and reused rather than repeatedly fetched.",
        "- This manifest/cache supports the target-item filing-body baseline; it is not itself a text model.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filings-path", default=str(DEFAULT_FILINGS))
    parser.add_argument("--text-root", default=str(TEXT_ROOT))
    parser.add_argument("--out-dir", default=str(TEXT_ROOT))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--target-items", default=",".join(TARGET_ITEMS))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-downloads", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    args = parser.parse_args()

    text_root = Path(args.text_root)
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    target_items = tuple(item.strip() for item in args.target_items.split(",") if item.strip())

    manifest = build_manifest(Path(args.filings_path), text_root, target_items, args.max_rows)
    manifest_path = out_dir / "sec_filing_text_manifest.csv"

    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    downloads: list[dict[str, Any]] = []
    if args.execute:
        if not user_agent:
            downloads.append({"status": "blocked_missing_sec_user_agent"})
        else:
            todo = manifest[~manifest["local_text_exists"]].copy()
            priority = todo["items"].map(item_download_priority)
            todo["_download_priority_rank"] = priority.map(lambda value: value[0])
            todo["_download_priority_item"] = priority.map(lambda value: value[1])
            todo = todo.sort_values(
                ["_download_priority_rank", "filing_date", "cik", "accession_number"],
                kind="mergesort",
            )
            todo = todo.head(args.max_downloads if args.max_downloads > 0 else len(todo))
            for row in todo.to_dict(orient="records"):
                result = download_one(str(row["archive_txt_url"]), Path(str(row["local_text_path"])), user_agent)
                result["priority_item"] = str(row["_download_priority_item"])
                result["filing_date"] = str(row["filing_date"])
                result["cik"] = str(row["cik"])
                result["accession_number"] = str(row["accession_number"])
                downloads.append(result)
                time.sleep(args.sleep_seconds)

    if downloads:
        log_path = out_dir / "sec_filing_text_download_log.csv"
        new_log = pd.DataFrame(downloads)
        if log_path.exists():
            try:
                old_log = pd.read_csv(log_path, low_memory=False)
                new_log = pd.concat([old_log, new_log], ignore_index=True, sort=False)
            except Exception:
                pass
        new_log.to_csv(log_path, index=False, encoding="utf-8-sig")

    manifest = refresh_local_text_state(manifest)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    downloaded_files = sum(1 for row in downloads if row.get("status") == "downloaded")
    local_text_files = int(manifest["local_text_exists"].sum()) if not manifest.empty else 0
    status_value = "downloaded" if args.execute and downloaded_files else "download_not_completed" if args.execute else "manifest_created"
    if not args.execute and local_text_files:
        status_value = "manifest_created_with_local_cache"
    status = {
        "status": status_value,
        "filings_path": str(args.filings_path),
        "text_root": str(text_root),
        "target_items": list(target_items),
        "manifest_rows": int(len(manifest)),
        "local_text_files": local_text_files,
        "local_text_bytes": int(manifest["local_text_bytes"].sum()) if not manifest.empty else 0,
        "execute": bool(args.execute),
        "max_downloads": int(args.max_downloads),
        "download_attempts": int(len([row for row in downloads if row.get("status") != "blocked_missing_sec_user_agent"])),
        "downloaded_files": int(downloaded_files),
        "user_agent_present": bool(user_agent),
        "manifest_csv": str(manifest_path),
        "download_log_csv": str(out_dir / "sec_filing_text_download_log.csv"),
        "download_status_counts": {
            str(key): int(value)
            for key, value in pd.Series([row.get("status") for row in downloads]).value_counts().items()
        } if downloads else {},
        "guardrail": "Target-item filing-body cache for Items 1.03/2.04/3.01; metadata-text baseline is separate.",
    }
    (results_dir / "sec_filing_text_collection_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(status, results_dir / "sec_filing_text_collection_status.md")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
