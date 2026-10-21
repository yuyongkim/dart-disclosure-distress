from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
MANIFEST = ROOT / "data" / "audit" / "public_submission_manifest.csv"
OUT_DIR = ROOT / "submission_package"
LATEST_POINTER = OUT_DIR / "LATEST_PACKAGE.md"
DENIED_PREFIXES = (
    ".env",
    ".omx/",
    ".claude/",
    ".git/",
    "src/",
    "data/raw/",
    "data/processed/",
    "data/interim/",
    "data/model_",
    "docs_cache/",
)
DENIED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
DENIED_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pkl", ".pickle", ".pyc", ".log", ".err"}
FORBIDDEN_BYTES = (
    b"C:" + b"/Users",
    b"C:" + b"\\Users",
    b"C:" + b"\\\\" + b"Users",
    b"Desktop/" + b"M&A",
    b"Desktop\\" + b"M&A",
    b"Desktop\\\\" + b"M&A",
    b"Desktop/" + b"Paper 4",
    b"Desktop\\" + b"Paper 4",
    b"Desktop\\\\" + b"Paper 4",
    b"desktop" + b"_external",
    b"grant" + b"_back",
    b"SEC" + b" Desktop",
)
ACTIVE_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|gsk_[A-Za-z0-9_-]{16,}|"
    r"\b(?:DART_API_KEY|OPENAI_API_KEY|GROQ_API_KEY|DEEPSEEK_API_KEY|ANTHROPIC_API_KEY|"
    r"EDINET_API_KEY|SEC_USER_AGENT|EDGAR_USER_AGENT|NEO4J_PASSWORD)\s*=\s*"
    r"(?![\"']?(?:\.\.\.|<[^>]+>|your[ _-]|YOUR[ _-]|$))[\"']?[^\"'\s<>]{8,})"
)
TEXT_PACKAGE_SUFFIXES = {".bib", ".csv", ".json", ".md", ".txt"}
STALE_STATUS_MARKERS = [
    "pass_with_handoff_refresh_needed",
    "fail_internal_risk",
    "blocked_claims_or_text_integrity_failures",
    "Traceback (most recent call last)",
    '"status": "fail"',
    '"internal_failure_count": 1',
    '"warnings": 1',
    '"failures": 1',
]


def normalized(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def forbidden_token(data: bytes) -> str:
    for token in FORBIDDEN_BYTES:
        if token in data:
            return token.decode("ascii", errors="ignore")
    return ""


def active_secret_hits(package_entry_name: str, data: bytes, part_name: str = "") -> list[dict[str, Any]]:
    text = data.decode("utf-8", errors="ignore")
    hits: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if ACTIVE_SECRET_RE.search(line):
            hit = {"path": package_entry_name, "line_number": line_number}
            if part_name:
                hit["part"] = part_name
            hits.append(hit)
    return hits


def stale_status_marker_hits(package_entry_name: str, data: bytes) -> list[dict[str, Any]]:
    if Path(package_entry_name).suffix.lower() not in TEXT_PACKAGE_SUFFIXES:
        return []
    text = data.decode("utf-8", errors="ignore")
    hits: list[dict[str, Any]] = []
    for marker in STALE_STATUS_MARKERS:
        line_numbers = [
            idx
            for idx, line in enumerate(text.splitlines(), start=1)
            if marker in line
        ]
        if line_numbers:
            hits.append({"path": package_entry_name, "marker": marker, "line_numbers": line_numbers[:10]})
    return hits


def nested_zip_hits(
    package_entry_name: str,
    data: bytes,
    part_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    forbidden_hits: list[dict[str, Any]] = []
    secret_hits: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as nested_zip:
            for inner_name in nested_zip.namelist():
                if inner_name.endswith("/"):
                    continue
                inner_data = nested_zip.read(inner_name)
                token = forbidden_token(inner_data)
                if token:
                    forbidden_hits.append({"path": package_entry_name, part_key: inner_name, "token": token})
                secret_hits.extend(active_secret_hits(package_entry_name, inner_data, part_name=inner_name))
    except zipfile.BadZipFile as exc:
        return forbidden_hits, secret_hits, f"BadZipFile: {exc}"
    return forbidden_hits, secret_hits, ""


def denied_reasons(relative_path: str) -> list[str]:
    rel = normalized(relative_path)
    path = Path(rel)
    reasons: list[str] = []
    for prefix in DENIED_PREFIXES:
        if rel == prefix.rstrip("/") or rel.startswith(prefix):
            reasons.append(f"denied prefix: {prefix}")
    if any(part in DENIED_PARTS for part in path.parts):
        reasons.append("denied cache path")
    if path.suffix.lower() in DENIED_SUFFIXES:
        reasons.append(f"denied suffix: {path.suffix.lower()}")
    return reasons


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def package_name(stamp: str) -> str:
    return f"paper4_core_package_{stamp}.zip"


def superseded_zip_packages(current_zip: Path) -> list[str]:
    if not OUT_DIR.exists():
        return []
    current = current_zip.name
    return sorted(
        path.name
        for path in OUT_DIR.glob("paper4_core_package_*.zip")
        if path.name != current
    )


def write_latest_pointer(audit: dict[str, Any], superseded_zips: list[str]) -> None:
    status = audit.get("status", "unknown")
    zip_path = audit.get("zip_path", "")
    timestamp_zip_note = (
        "Timestamped ZIPs in this folder are superseded build artifacts only; use `latest` for handoff."
        if superseded_zips
        else "No timestamped superseded ZIPs are currently retained in this folder."
    )
    lines = [
        "# Latest Paper 4 Core Package",
        "",
        f"Status: `{status}`",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Latest ZIP | `{zip_path}` |",
        f"| Audit JSON | `{audit.get('audit_path', '')}` |",
        f"| Manifest | `{audit.get('manifest_path', '')}` |",
        f"| ZIP entries | `{audit.get('zip_entries', '')}` |",
        f"| ZIP SHA-256 | `{audit.get('zip_sha256', '')}` |",
        f"| ZIP size bytes | `{audit.get('zip_size_bytes', '')}` |",
        f"| Generated at UTC | `{datetime.now(timezone.utc).isoformat()}` |",
        "",
        f"Use the `latest` ZIP above for email/submission handoff. {timestamp_zip_note}",
        "`Generated at UTC` is the local package-audit/pointer regeneration time; it is not the Gmail send timestamp.",
        "When the package has been emailed, `email_handoff_receipt_audit.md` records whether the last verified Gmail handoff receipt still matches this latest ZIP SHA and size.",
        "",
        "## Superseded Timestamped ZIPs",
        "",
    ]
    if superseded_zips:
        lines.extend(f"- `{name}`" for name in superseded_zips)
    else:
        lines.append("None.")
    lines.append("")
    LATEST_POINTER.write_text("\n".join(lines), encoding="utf-8")


def build_package(stamp: str) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(MANIFEST)
    zip_path = OUT_DIR / package_name(stamp)
    audit_path = OUT_DIR / f"paper4_core_package_{stamp}_audit.json"
    superseded_zips = superseded_zip_packages(zip_path)

    missing: list[str] = []
    denied: list[dict[str, Any]] = []
    manifest_paths: list[str] = []
    entries: list[tuple[str, Path]] = []
    for row in rows:
        rel = normalized(row["path"])
        manifest_paths.append(rel)
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        reasons = denied_reasons(rel)
        if row.get("denied_reasons", "").strip():
            reasons.append(f"manifest denied: {row['denied_reasons']}")
        if reasons:
            denied.append({"path": rel, "denied_reasons": reasons})
            continue
        entries.append((rel, path))
    duplicate_manifest_paths = sorted(
        path for path, count in Counter(manifest_paths).items() if count > 1
    )

    if missing or denied or duplicate_manifest_paths:
        audit = {
            "status": "fail",
            "zip_path": str(zip_path.relative_to(ROOT)).replace("\\", "/"),
            "audit_path": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
            "manifest_path": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "missing": missing,
            "denied": denied,
            "duplicate_manifest_paths": duplicate_manifest_paths,
            "duplicate_manifest_path_count": len(duplicate_manifest_paths),
            "latest_pointer_path": str(LATEST_POINTER.relative_to(ROOT)).replace("\\", "/"),
            "superseded_zip_package_count": len(superseded_zips),
            "superseded_zip_packages": superseded_zips,
        }
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_latest_pointer(audit, superseded_zips)
        return audit

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for rel, path in entries:
            archive.write(path, rel)

    stale_entries: list[str] = []
    forbidden_hits: list[dict[str, Any]] = []
    docx_forbidden_hits_all: list[dict[str, Any]] = []
    xlsx_forbidden_hits_all: list[dict[str, Any]] = []
    active_secret_hits_all: list[dict[str, Any]] = []
    docx_active_secret_hits_all: list[dict[str, Any]] = []
    xlsx_active_secret_hits_all: list[dict[str, Any]] = []
    stale_status_marker_hits_all: list[dict[str, Any]] = []
    docx_scan_errors: list[dict[str, str]] = []
    xlsx_scan_errors: list[dict[str, str]] = []
    bad_names: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        duplicate_zip_entries = sorted(
            name for name, count in Counter(names).items() if count > 1
        )
        expected_names = {rel for rel, _ in entries}
        extra = sorted(set(names) - expected_names)
        missing_zip_entries = sorted(expected_names - set(names))
        for name in names:
            reasons = denied_reasons(name)
            if reasons:
                bad_names.append(name)
            data = archive.read(name)
            current_path = ROOT / name
            if current_path.exists() and sha256_bytes(data) != sha256_file(current_path):
                stale_entries.append(name)
            token = forbidden_token(data)
            if token:
                forbidden_hits.append({"path": name, "token": token})
            active_secret_hits_all.extend(active_secret_hits(name, data))
            stale_status_marker_hits_all.extend(stale_status_marker_hits(name, data))
            if Path(name).suffix.lower() == ".docx":
                docx_hits, docx_secret_hits, docx_error = nested_zip_hits(name, data, "docx_part")
                docx_forbidden_hits_all.extend(docx_hits)
                docx_active_secret_hits_all.extend(docx_secret_hits)
                if docx_error:
                    docx_scan_errors.append({"path": name, "error": docx_error})
            if Path(name).suffix.lower() == ".xlsx":
                xlsx_hits, xlsx_secret_hits, xlsx_error = nested_zip_hits(name, data, "xlsx_part")
                xlsx_forbidden_hits_all.extend(xlsx_hits)
                xlsx_active_secret_hits_all.extend(xlsx_secret_hits)
                if xlsx_error:
                    xlsx_scan_errors.append({"path": name, "error": xlsx_error})

    zip_sha = sha256_file(zip_path)
    status = "pass"
    if (
        extra
        or missing_zip_entries
        or stale_entries
        or duplicate_zip_entries
        or forbidden_hits
        or docx_forbidden_hits_all
        or xlsx_forbidden_hits_all
        or active_secret_hits_all
        or docx_active_secret_hits_all
        or xlsx_active_secret_hits_all
        or stale_status_marker_hits_all
        or docx_scan_errors
        or xlsx_scan_errors
        or bad_names
    ):
        status = "fail"
    audit = {
        "status": status,
        "zip_path": str(zip_path.relative_to(ROOT)).replace("\\", "/"),
        "audit_path": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
        "manifest_path": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "manifest_rows": len(rows),
        "zip_entries": len(entries),
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_sha256": zip_sha,
        "extra_entries": extra,
        "missing_zip_entries": missing_zip_entries,
        "stale_entries": stale_entries,
        "duplicate_manifest_paths": duplicate_manifest_paths,
        "duplicate_manifest_path_count": len(duplicate_manifest_paths),
        "duplicate_zip_entries": duplicate_zip_entries,
        "duplicate_zip_entry_count": len(duplicate_zip_entries),
        "forbidden_hits": forbidden_hits,
        "docx_forbidden_hits": docx_forbidden_hits_all,
        "docx_forbidden_hit_count": len(docx_forbidden_hits_all),
        "xlsx_forbidden_hits": xlsx_forbidden_hits_all,
        "xlsx_forbidden_hit_count": len(xlsx_forbidden_hits_all),
        "active_secret_hits": active_secret_hits_all,
        "active_secret_hit_count": len(active_secret_hits_all),
        "docx_active_secret_hits": docx_active_secret_hits_all,
        "docx_active_secret_hit_count": len(docx_active_secret_hits_all),
        "xlsx_active_secret_hits": xlsx_active_secret_hits_all,
        "xlsx_active_secret_hit_count": len(xlsx_active_secret_hits_all),
        "stale_status_marker_hits": stale_status_marker_hits_all,
        "stale_status_marker_hit_count": len(stale_status_marker_hits_all),
        "docx_scan_errors": docx_scan_errors,
        "xlsx_scan_errors": xlsx_scan_errors,
        "bad_names": bad_names,
        "latest_pointer_path": str(LATEST_POINTER.relative_to(ROOT)).replace("\\", "/"),
        "superseded_zip_package_count": len(superseded_zips),
        "superseded_zip_packages": superseded_zips,
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_latest_pointer(audit, superseded_zips)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    audit = build_package(args.stamp)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
