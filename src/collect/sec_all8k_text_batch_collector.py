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
DEFAULT_MANIFEST = ROOT / "data" / "raw" / "sec_all8k_text" / "sec_all8k_text_manifest.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "raw" / "sec_all8k_text"
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_TEXT_ROOT = ROOT / "data" / "raw" / "sec_filing_text"
DEFAULT_USER_AGENT = "Research (Yuyong Kim) ykim288@wisc.edu"


def path_exists(value: object) -> bool:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    return bool(text) and Path(text).exists()


def path_size(value: object) -> int:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if not text:
        return 0
    path = Path(text)
    return int(path.stat().st_size) if path.exists() else 0


def normalize_cik(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else ""


def accession_nodash(value: object) -> str:
    return str(value or "").replace("-", "").strip()


def canonical_text_path(text_root: Path, cik: object, accession: object) -> str:
    cik_norm = normalize_cik(cik)
    accession_key = accession_nodash(accession)
    if not cik_norm or not accession_key:
        return ""
    return str(text_root / cik_norm / f"{accession_key}.txt")


def add_canonical_text_paths(manifest: pd.DataFrame, text_root: Path) -> pd.DataFrame:
    manifest = manifest.copy()
    original_path = (
        manifest["local_text_path"].fillna("").astype(str)
        if "local_text_path" in manifest
        else pd.Series([""] * len(manifest), index=manifest.index)
    )
    manifest["manifest_local_path_redacted"] = original_path.str.startswith("public_")
    manifest["download_text_path"] = manifest.apply(
        lambda row: canonical_text_path(text_root, row.get("cik"), row.get("accession_number")),
        axis=1,
    )
    missing_path = manifest["download_text_path"].eq("")
    if missing_path.any() and "local_text_path" in manifest:
        manifest.loc[missing_path, "download_text_path"] = manifest.loc[missing_path, "local_text_path"].fillna("").astype(str)
    return manifest


def refresh_manifest(manifest: pd.DataFrame, text_root: Path) -> pd.DataFrame:
    manifest = manifest.copy()
    manifest = add_canonical_text_paths(manifest, text_root)
    manifest["local_text_exists_now"] = manifest["download_text_path"].map(path_exists)
    manifest["local_text_bytes_now"] = manifest["download_text_path"].map(path_size)
    manifest["needs_download_now"] = ~manifest["local_text_exists_now"]
    manifest["download_priority_now"] = manifest["has_target_distress_item"].map(lambda value: 0 if bool(value) else 1)
    return manifest


def assign_current_batches(manifest: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    missing = manifest[manifest["needs_download_now"]].copy()
    if missing.empty:
        manifest["current_batch_id"] = pd.NA
        return manifest
    missing = missing.sort_values(
        ["download_priority_now", "filing_date", "cik", "accession_number"],
        kind="mergesort",
    )
    missing["current_batch_id"] = (range(len(missing)))
    missing["current_batch_id"] = (missing["current_batch_id"] // batch_size) + 1
    return manifest.merge(
        missing[["cik", "accession_number", "current_batch_id"]],
        on=["cik", "accession_number"],
        how="left",
    )


def build_batch_status(manifest: pd.DataFrame) -> pd.DataFrame:
    missing = manifest[manifest["needs_download_now"] & manifest["current_batch_id"].notna()].copy()
    if missing.empty:
        return pd.DataFrame(
            columns=[
                "batch_id",
                "download_priority",
                "rows",
                "target_rows",
                "first_filing_date",
                "last_filing_date",
                "cached_rows_in_batch",
                "missing_rows_in_batch",
            ]
        )
    status = (
        missing.groupby("current_batch_id", sort=True)
        .agg(
            download_priority=("download_priority_now", "min"),
            rows=("accession_number", "size"),
            target_rows=("has_target_distress_item", "sum"),
            first_filing_date=("filing_date", "min"),
            last_filing_date=("filing_date", "max"),
        )
        .reset_index()
        .rename(columns={"current_batch_id": "batch_id"})
    )
    status["target_rows"] = status["target_rows"].astype(int)
    status["cached_rows_in_batch"] = 0
    status["missing_rows_in_batch"] = status["rows"].astype(int)
    return status


def select_download_rows(manifest: pd.DataFrame, batch_id: int | None, max_downloads: int) -> pd.DataFrame:
    todo = manifest[manifest["needs_download_now"]].copy()
    if batch_id is not None:
        todo = todo[todo["current_batch_id"].astype("Int64").eq(int(batch_id))]
    todo = todo.sort_values(
        ["download_priority_now", "filing_date", "cik", "accession_number"],
        kind="mergesort",
    )
    if max_downloads > 0:
        todo = todo.head(max_downloads)
    else:
        todo = todo.head(0)
    return todo


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


def append_download_log(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    new_log = pd.DataFrame(rows)
    if path.exists():
        try:
            old_log = pd.read_csv(path, low_memory=False)
            new_log = pd.concat([old_log, new_log], ignore_index=True, sort=False)
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    new_log.to_csv(path, index=False, encoding="utf-8-sig")


def summarize_download_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "download_log_rows": 0,
            "download_log_downloaded_files": 0,
            "download_log_existing_files": 0,
            "download_log_bytes": 0,
            "download_log_status_counts": {},
            "download_log_batch_ids": [],
        }
    try:
        log = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        log = pd.DataFrame()
    if log.empty:
        return {
            "download_log_rows": 0,
            "download_log_downloaded_files": 0,
            "download_log_existing_files": 0,
            "download_log_bytes": 0,
            "download_log_status_counts": {},
            "download_log_batch_ids": [],
        }
    downloaded = log[log.get("status", pd.Series(dtype=str)).astype(str).eq("downloaded")].copy()
    batch_ids: list[int] = []
    if "batch_id" in log:
        batch_ids = [
            int(value)
            for value in sorted(pd.to_numeric(log["batch_id"], errors="coerce").dropna().unique())
        ]
    existing_files = int(downloaded["path"].map(path_exists).sum()) if "path" in downloaded else 0
    byte_count = int(pd.to_numeric(downloaded.get("bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    status_counts = {
        str(key): int(value)
        for key, value in log.get("status", pd.Series(dtype=str)).astype(str).value_counts().items()
    }
    return {
        "download_log_rows": int(len(log)),
        "download_log_downloaded_files": int(len(downloaded)),
        "download_log_existing_files": existing_files,
        "download_log_bytes": byte_count,
        "download_log_status_counts": status_counts,
        "download_log_batch_ids": batch_ids,
    }


def summarize(
    manifest: pd.DataFrame,
    batch_status: pd.DataFrame,
    downloads: list[dict[str, Any]],
    args: argparse.Namespace,
    user_agent: str,
) -> dict[str, Any]:
    cached = int(manifest["local_text_exists_now"].sum())
    missing = int(manifest["needs_download_now"].sum())
    redacted_paths = int(manifest.get("manifest_local_path_redacted", pd.Series(dtype=bool)).sum())
    canonical_paths = int(manifest["download_text_path"].fillna("").astype(str).ne("").sum())
    target = manifest[manifest["has_target_distress_item"]]
    target_cached = int(target["local_text_exists_now"].sum())
    downloaded_files = sum(1 for row in downloads if row.get("status") == "downloaded")
    if not args.execute:
        status = "status_only"
    elif args.max_downloads <= 0:
        status = "download_not_requested"
    elif downloaded_files:
        status = "downloaded_partial" if missing else "downloaded_complete"
    else:
        status = "download_not_completed"
    next_batch_id = None if batch_status.empty else int(batch_status.iloc[0]["batch_id"])
    download_log_path = Path(args.out_dir) / "sec_all8k_text_batch_download_log.csv"
    log_summary = summarize_download_log(download_log_path)
    return {
        "status": status,
        "manifest_path": str(args.manifest_path),
        "manifest_rows": int(len(manifest)),
        "cached_text_files": cached,
        "missing_text_files": missing,
        "cached_text_bytes": int(manifest["local_text_bytes_now"].sum()),
        "coverage_rate": float(cached / len(manifest)) if len(manifest) else None,
        "path_resolution_mode": "canonical_cik_accession_text_root",
        "manifest_local_path_redacted_rows": redacted_paths,
        "canonical_text_path_rows": canonical_paths,
        "target_distress_item_rows": int(len(target)),
        "target_distress_item_cached_files": target_cached,
        "target_distress_item_coverage_rate": float(target_cached / len(target)) if len(target) else None,
        "batch_size": int(args.batch_size),
        "open_batches": int(len(batch_status)),
        "next_batch_id": next_batch_id,
        "execute": bool(args.execute),
        "requested_batch_id": args.batch_id,
        "max_downloads": int(args.max_downloads),
        "download_attempts": int(len(downloads)),
        "downloaded_files": int(downloaded_files),
        "download_status_counts": {
            str(key): int(value)
            for key, value in pd.Series([row.get("status") for row in downloads]).value_counts().items()
        } if downloads else {},
        "user_agent_present": bool(user_agent),
        "batch_status_csv": str(Path(args.out_dir) / "sec_all8k_text_batch_status.csv"),
        "download_log_csv": str(download_log_path),
        **log_summary,
        "text_root": str(args.text_root),
        "guardrail": "Default run is status-only and performs no SEC Archives requests. Use --execute --batch-id N --max-downloads K with a real identifying SEC user agent for controlled full-scope collection. Download paths are reconstructed from CIK and accession so public redaction of manifest path columns cannot redirect the cache.",
    }


def write_markdown(status: dict[str, Any], batch_status: pd.DataFrame, path: Path) -> None:
    lines = [
        "# SEC All-8-K Full Text Batch Collection Status",
        "",
        "This artifact tracks the full all-8-K filing-body text scope. It does not download by default.",
        "",
        "## Status",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Status | `{status['status']}` |",
        f"| Manifest rows | {status['manifest_rows']} |",
        f"| Cached text files | {status['cached_text_files']} |",
        f"| Missing text files | {status['missing_text_files']} |",
        f"| Coverage rate | {pct(status['coverage_rate'])} |",
        f"| Path resolution mode | `{status['path_resolution_mode']}` |",
        f"| Redacted manifest path rows | {status['manifest_local_path_redacted_rows']} |",
        f"| Canonical text path rows | {status['canonical_text_path_rows']} |",
        f"| Target distress rows | {status['target_distress_item_rows']} |",
        f"| Target distress cached files | {status['target_distress_item_cached_files']} |",
        f"| Target distress coverage | {pct(status['target_distress_item_coverage_rate'])} |",
        f"| Batch size | {status['batch_size']} |",
        f"| Open batches | {status['open_batches']} |",
        f"| Next batch id | {status['next_batch_id'] or ''} |",
        f"| Execute downloads | {status['execute']} |",
        f"| Requested batch id | {status['requested_batch_id'] or ''} |",
        f"| Max downloads | {status['max_downloads']} |",
        f"| Download attempts | {status['download_attempts']} |",
        f"| Downloaded files | {status['downloaded_files']} |",
        f"| Historical log rows | {status['download_log_rows']} |",
        f"| Historical log downloaded files | {status['download_log_downloaded_files']} |",
        f"| Historical log existing files | {status['download_log_existing_files']} |",
        f"| Historical log bytes | {bytes_mb(status['download_log_bytes'])} MB |",
        "",
        "## First Open Batches",
        "",
        "| Batch | Priority | Missing rows | Target rows | First filing date | Last filing date |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for row in batch_status.head(20).itertuples(index=False):
        lines.append(
            f"| {int(row.batch_id)} | {int(row.download_priority)} | {int(row.missing_rows_in_batch)} | {int(row.target_rows)} | {row.first_filing_date} | {row.last_filing_date} |"
        )
    lines.extend(
        [
            "",
            "## Historical Download Log",
            "",
            f"- Log path: `{status['download_log_csv']}`",
            f"- Status counts: `{json.dumps(status['download_log_status_counts'], ensure_ascii=False)}`",
            f"- Batch ids represented: `{status['download_log_batch_ids']}`",
            "",
            "## Guardrail",
            "",
            status["guardrail"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}%"


def bytes_mb(value: float | int) -> str:
    return f"{float(value) / 1_000_000:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--text-root", default=str(DEFAULT_TEXT_ROOT))
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--batch-id", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-downloads", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("SEC_USER_AGENT") or os.environ.get("EDGAR_USER_AGENT") or DEFAULT_USER_AGENT,
        help="Identifying SEC user agent used only when --execute is set.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    text_root = Path(args.text_root)
    manifest = pd.read_csv(args.manifest_path, low_memory=False)
    manifest = refresh_manifest(manifest, text_root)
    manifest = assign_current_batches(manifest, args.batch_size)
    batch_status = build_batch_status(manifest)

    user_agent = str(args.user_agent or "").strip()
    downloads: list[dict[str, Any]] = []
    if args.execute and args.max_downloads > 0:
        todo = select_download_rows(manifest, args.batch_id, args.max_downloads)
        for row in todo.to_dict(orient="records"):
            result = download_one(str(row["archive_txt_url"]), Path(str(row["download_text_path"])), user_agent)
            result["batch_id"] = "" if pd.isna(row.get("current_batch_id")) else int(row["current_batch_id"])
            result["filing_date"] = str(row.get("filing_date", ""))
            result["cik"] = str(row.get("cik", ""))
            result["accession_number"] = str(row.get("accession_number", ""))
            result["items"] = str(row.get("items", ""))
            downloads.append(result)
            time.sleep(args.sleep_seconds)
        append_download_log(downloads, out_dir / "sec_all8k_text_batch_download_log.csv")
        manifest = refresh_manifest(pd.read_csv(args.manifest_path, low_memory=False), text_root)
        manifest = assign_current_batches(manifest, args.batch_size)
        batch_status = build_batch_status(manifest)

    status = summarize(manifest, batch_status, downloads, args, user_agent)
    batch_status.to_csv(out_dir / "sec_all8k_text_batch_status.csv", index=False, encoding="utf-8-sig")
    (results_dir / "sec_all8k_text_batch_collection_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(status, batch_status, results_dir / "sec_all8k_text_batch_collection_status.md")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
