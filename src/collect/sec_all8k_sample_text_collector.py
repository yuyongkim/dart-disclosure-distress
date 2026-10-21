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
DEFAULT_SAMPLE_PLAN = ROOT / "data" / "raw" / "sec_all8k_text" / "sec_all8k_text_sample_plan.csv"
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


def add_canonical_text_paths(sample: pd.DataFrame, text_root: Path) -> pd.DataFrame:
    sample = sample.copy()
    original_path = (
        sample["local_text_path"].fillna("").astype(str)
        if "local_text_path" in sample
        else pd.Series([""] * len(sample), index=sample.index)
    )
    sample["sample_local_path_redacted"] = original_path.str.startswith("public_")
    sample["download_text_path"] = sample.apply(
        lambda row: canonical_text_path(text_root, row.get("cik"), row.get("accession_number")),
        axis=1,
    )
    missing_path = sample["download_text_path"].eq("")
    if missing_path.any() and "local_text_path" in sample:
        sample.loc[missing_path, "download_text_path"] = sample.loc[missing_path, "local_text_path"].fillna("").astype(str)
    return sample


def refresh_status(sample: pd.DataFrame, text_root: Path) -> pd.DataFrame:
    sample = sample.copy()
    sample = add_canonical_text_paths(sample, text_root)
    sample["local_text_exists_now"] = sample["download_text_path"].map(path_exists)
    sample["local_text_bytes_now"] = sample["download_text_path"].map(path_size)
    sample["needs_download"] = ~sample["local_text_exists_now"]
    sample["text_status_now"] = sample["local_text_exists_now"].map(lambda ok: "cached" if ok else "missing")
    return sample


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


def select_download_rows(sample: pd.DataFrame, max_downloads: int) -> pd.DataFrame:
    todo = sample[sample["needs_download"]].copy()
    if todo.empty:
        return todo
    role_priority = {
        "non_target_text_scope_sample": 0,
        "target_cached_reference_sample": 1,
    }
    todo["_role_priority"] = todo["sample_role"].map(role_priority).fillna(9).astype(int)
    todo = todo.sort_values(
        ["_role_priority", "filing_year", "item_bucket", "cik", "accession_number"],
        kind="mergesort",
    )
    if max_downloads > 0:
        todo = todo.head(max_downloads)
    return todo


def summarize(sample: pd.DataFrame, downloads: list[dict[str, Any]], args: argparse.Namespace, user_agent: str) -> dict[str, Any]:
    role_counts = sample.groupby("sample_role", dropna=False).size().to_dict()
    cached_by_role = sample[sample["local_text_exists_now"]].groupby("sample_role", dropna=False).size().to_dict()
    missing_by_role = sample[sample["needs_download"]].groupby("sample_role", dropna=False).size().to_dict()
    downloaded_files = sum(1 for row in downloads if row.get("status") == "downloaded")
    blocked = any(row.get("status") == "blocked_missing_sec_user_agent" for row in downloads)
    if args.execute and downloaded_files:
        status = "sample_downloaded_partial" if int(sample["needs_download"].sum()) else "sample_downloaded_complete"
    elif args.execute and blocked:
        status = "blocked_missing_sec_user_agent"
    elif args.execute:
        status = "download_not_completed"
    elif int(sample["needs_download"].sum()) == 0:
        status = "sample_cache_complete"
    else:
        status = "sample_status_created"
    return {
        "status": status,
        "sample_plan_path": str(args.sample_plan_path),
        "sample_rows": int(len(sample)),
        "path_resolution_mode": "canonical_cik_accession_text_root",
        "sample_local_path_redacted_rows": int(sample.get("sample_local_path_redacted", pd.Series(dtype=bool)).sum()),
        "canonical_text_path_rows": int(sample["download_text_path"].fillna("").astype(str).ne("").sum()),
        "cached_text_files": int(sample["local_text_exists_now"].sum()),
        "missing_text_files": int(sample["needs_download"].sum()),
        "cached_text_bytes": int(sample["local_text_bytes_now"].sum()),
        "role_counts": {str(k): int(v) for k, v in role_counts.items()},
        "cached_by_role": {str(k): int(v) for k, v in cached_by_role.items()},
        "missing_by_role": {str(k): int(v) for k, v in missing_by_role.items()},
        "execute": bool(args.execute),
        "max_downloads": int(args.max_downloads),
        "download_attempts": int(len([row for row in downloads if row.get("status") != "blocked_missing_sec_user_agent"])),
        "downloaded_files": int(downloaded_files),
        "download_status_counts": {
            str(key): int(value)
            for key, value in pd.Series([row.get("status") for row in downloads]).value_counts().items()
        } if downloads else {},
        "user_agent_present": bool(user_agent),
        "status_csv": str(Path(args.out_dir) / "sec_all8k_text_sample_status.csv"),
        "download_log_csv": str(Path(args.out_dir) / "sec_all8k_text_sample_download_log.csv"),
        "text_root": str(args.text_root),
        "guardrail": "Default run performs no downloads. Use --execute with a real identifying SEC user agent for bounded sample-text retrieval. Cache paths are reconstructed from CIK and accession so public redaction of sample path columns cannot redirect downloads or cache checks.",
    }


def write_markdown(status: dict[str, Any], path: Path) -> None:
    lines = [
        "# SEC All-8-K Sample Text Collection Status",
        "",
        "This artifact tracks the bounded all-8-K sample text cache. It does not download by default.",
        "",
        "## Status",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Status | `{status['status']}` |",
        f"| Sample rows | {status['sample_rows']} |",
        f"| Path resolution mode | `{status['path_resolution_mode']}` |",
        f"| Redacted sample path rows | {status['sample_local_path_redacted_rows']} |",
        f"| Canonical text path rows | {status['canonical_text_path_rows']} |",
        f"| Cached text files | {status['cached_text_files']} |",
        f"| Missing text files | {status['missing_text_files']} |",
        f"| Cached text bytes | {status['cached_text_bytes']} |",
        f"| Execute downloads | {status['execute']} |",
        f"| User agent present | {status['user_agent_present']} |",
        f"| Download attempts | {status['download_attempts']} |",
        f"| Downloaded files | {status['downloaded_files']} |",
        "",
        "## Role Counts",
        "",
        "| Role | Rows | Cached | Missing |",
        "|---|---:|---:|---:|",
    ]
    roles = sorted(set(status["role_counts"]) | set(status["cached_by_role"]) | set(status["missing_by_role"]))
    for role in roles:
        lines.append(
            f"| {role} | {status['role_counts'].get(role, 0)} | {status['cached_by_role'].get(role, 0)} | {status['missing_by_role'].get(role, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Default execution is status-only and performs no SEC Archives requests.",
            "- Use `--execute --max-downloads N` only with a real identifying SEC user agent.",
            "- Until non-target sample files are downloaded and modeled, this remains a collection-status artifact rather than validation evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-plan-path", default=str(DEFAULT_SAMPLE_PLAN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--text-root", default=str(DEFAULT_TEXT_ROOT))
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
    sample = pd.read_csv(args.sample_plan_path, low_memory=False)
    sample = refresh_status(sample, text_root)

    user_agent = str(args.user_agent or "").strip()
    downloads: list[dict[str, Any]] = []
    if args.execute:
        if not user_agent:
            downloads.append({"status": "blocked_missing_sec_user_agent"})
        else:
            todo = select_download_rows(sample, args.max_downloads)
            for row in todo.to_dict(orient="records"):
                result = download_one(str(row["archive_txt_url"]), Path(str(row["download_text_path"])), user_agent)
                result["sample_id"] = str(row.get("sample_id", ""))
                result["sample_role"] = str(row.get("sample_role", ""))
                result["sample_stratum"] = str(row.get("sample_stratum", ""))
                result["filing_date"] = str(row.get("filing_date", ""))
                result["cik"] = str(row.get("cik", ""))
                result["accession_number"] = str(row.get("accession_number", ""))
                downloads.append(result)
                time.sleep(args.sleep_seconds)
            append_download_log(downloads, out_dir / "sec_all8k_text_sample_download_log.csv")
            sample = refresh_status(sample, text_root)

    status = summarize(sample, downloads, args, user_agent)
    sample.to_csv(out_dir / "sec_all8k_text_sample_status.csv", index=False, encoding="utf-8-sig")
    (results_dir / "sec_all8k_text_sample_collection_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    write_markdown(status, results_dir / "sec_all8k_text_sample_collection_status.md")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
