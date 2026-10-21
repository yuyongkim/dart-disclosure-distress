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
CORE_PACKAGE_AUDIT_JSON = ROOT / "submission_package" / "paper4_core_package_latest_audit.json"
EMAIL_HANDOFF_AUDIT_JSON = ROOT / "submission_package" / "email_handoff_receipt_audit.json"
SUBMISSION_PACKAGE_DIR = ROOT / "submission_package"
OUT_JSON = ROOT / "results" / "post_run_artifact_status.json"
OUT_MD = ROOT / "results" / "post_run_artifact_status.md"
ALLOWED_SUBMISSION_PACKAGE_ARTIFACTS = {
    "email_handoff_receipt.json",
    "email_handoff_receipt_audit.json",
    "email_handoff_receipt_audit.md",
    "LATEST_PACKAGE.md",
    "paper4_core_package_latest.zip",
    "paper4_core_package_latest_audit.json",
}
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


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


def current_superseded_package_artifacts() -> dict[str, Any]:
    allowed = {"paper4_core_package_latest.zip", "paper4_core_package_latest_audit.json"}
    artifacts: list[str] = []
    if SUBMISSION_PACKAGE_DIR.exists():
        for pattern in ("paper4_core_package_*.zip", "paper4_core_package_*_audit.json"):
            for path in SUBMISSION_PACKAGE_DIR.glob(pattern):
                if path.name not in allowed:
                    artifacts.append(path.name)
    return {
        "current_superseded_package_artifact_count": len(sorted(set(artifacts))),
        "current_superseded_package_artifacts": sorted(set(artifacts)),
    }


def current_submission_package_unexpected_artifacts() -> dict[str, Any]:
    artifacts: list[str] = []
    present_allowed: set[str] = set()
    non_file_allowed: list[str] = []
    if SUBMISSION_PACKAGE_DIR.exists():
        for path in SUBMISSION_PACKAGE_DIR.iterdir():
            if path.name not in ALLOWED_SUBMISSION_PACKAGE_ARTIFACTS:
                artifacts.append(path.name)
                continue
            present_allowed.add(path.name)
            if not path.is_file():
                non_file_allowed.append(path.name)
    missing_required = sorted(ALLOWED_SUBMISSION_PACKAGE_ARTIFACTS - present_allowed)
    return {
        "current_submission_package_unexpected_artifact_count": len(sorted(artifacts)),
        "current_submission_package_unexpected_artifacts": sorted(artifacts),
        "current_submission_package_missing_required_artifact_count": len(missing_required),
        "current_submission_package_missing_required_artifacts": missing_required,
        "current_submission_package_allowed_non_file_artifact_count": len(sorted(non_file_allowed)),
        "current_submission_package_allowed_non_file_artifacts": sorted(non_file_allowed),
    }


def current_zip_security_scan(current_zip: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "current_zip_forbidden_hit_count": None,
        "current_zip_active_secret_hit_count": None,
        "current_zip_docx_forbidden_hit_count": None,
        "current_zip_docx_active_secret_hit_count": None,
        "current_zip_xlsx_forbidden_hit_count": None,
        "current_zip_xlsx_active_secret_hit_count": None,
        "current_zip_docx_scan_error_count": None,
        "current_zip_xlsx_scan_error_count": None,
        "current_zip_forbidden_hits": [],
        "current_zip_active_secret_hits": [],
        "current_zip_docx_forbidden_hits": [],
        "current_zip_docx_active_secret_hits": [],
        "current_zip_xlsx_forbidden_hits": [],
        "current_zip_xlsx_active_secret_hits": [],
        "current_zip_docx_scan_errors": [],
        "current_zip_xlsx_scan_errors": [],
        "current_zip_security_scan_error": "",
    }
    if current_zip is None or not current_zip.exists():
        return result

    forbidden_hits: list[dict[str, Any]] = []
    active_secret_hits_all: list[dict[str, Any]] = []
    docx_forbidden_hits_all: list[dict[str, Any]] = []
    docx_active_secret_hits_all: list[dict[str, Any]] = []
    xlsx_forbidden_hits_all: list[dict[str, Any]] = []
    xlsx_active_secret_hits_all: list[dict[str, Any]] = []
    docx_scan_errors: list[dict[str, str]] = []
    xlsx_scan_errors: list[dict[str, str]] = []
    try:
        with zipfile.ZipFile(current_zip, "r") as archive:
            for name in archive.namelist():
                if name.endswith("/"):
                    continue
                data = archive.read(name)
                token = forbidden_token(data)
                if token:
                    forbidden_hits.append({"path": name, "token": token})
                active_secret_hits_all.extend(active_secret_hits(name, data))
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
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        result["current_zip_security_scan_error"] = f"{type(exc).__name__}: {exc}"
        return result

    result.update(
        {
            "current_zip_forbidden_hit_count": len(forbidden_hits),
            "current_zip_active_secret_hit_count": len(active_secret_hits_all),
            "current_zip_docx_forbidden_hit_count": len(docx_forbidden_hits_all),
            "current_zip_docx_active_secret_hit_count": len(docx_active_secret_hits_all),
            "current_zip_xlsx_forbidden_hit_count": len(xlsx_forbidden_hits_all),
            "current_zip_xlsx_active_secret_hit_count": len(xlsx_active_secret_hits_all),
            "current_zip_docx_scan_error_count": len(docx_scan_errors),
            "current_zip_xlsx_scan_error_count": len(xlsx_scan_errors),
            "current_zip_forbidden_hits": forbidden_hits[:20],
            "current_zip_active_secret_hits": active_secret_hits_all[:20],
            "current_zip_docx_forbidden_hits": docx_forbidden_hits_all[:20],
            "current_zip_docx_active_secret_hits": docx_active_secret_hits_all[:20],
            "current_zip_xlsx_forbidden_hits": xlsx_forbidden_hits_all[:20],
            "current_zip_xlsx_active_secret_hits": xlsx_active_secret_hits_all[:20],
            "current_zip_docx_scan_errors": docx_scan_errors[:20],
            "current_zip_xlsx_scan_errors": xlsx_scan_errors[:20],
        }
    )
    return result


def manifest_zip_freshness(package: dict[str, Any], current_zip: Path | None) -> dict[str, Any]:
    manifest_rel = package.get("manifest_path", "")
    manifest_path = ROOT / manifest_rel if manifest_rel else None
    result: dict[str, Any] = {
        "manifest_path": manifest_rel,
        "manifest_exists": bool(manifest_path and manifest_path.exists()),
        "manifest_row_count": None,
        "manifest_expected_entry_count": None,
        "current_zip_entry_count": None,
        "manifest_zip_missing_entry_count": None,
        "manifest_zip_extra_entry_count": None,
        "manifest_zip_stale_entry_count": None,
        "manifest_duplicate_entry_count": None,
        "current_zip_duplicate_entry_count": None,
        "manifest_source_missing_count": None,
        "manifest_zip_missing_entries": [],
        "manifest_zip_extra_entries": [],
        "manifest_zip_stale_entries": [],
        "manifest_duplicate_entries": [],
        "current_zip_duplicate_entries": [],
        "manifest_source_missing_entries": [],
        "manifest_zip_scan_error": "",
    }
    if not result["manifest_exists"] or current_zip is None or not current_zip.exists():
        return result

    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_list = [normalized(row.get("path", "")) for row in rows if row.get("path")]
        expected = set(expected_list)
        duplicate_manifest_entries = sorted(
            name for name, count in Counter(expected_list).items() if count > 1
        )
        missing_sources = sorted(rel for rel in expected if not (ROOT / rel).exists())
        with zipfile.ZipFile(current_zip, "r") as archive:
            name_list = archive.namelist()
            names = set(name_list)
            duplicate_zip_entries = sorted(
                name for name, count in Counter(name_list).items() if count > 1
            )
            stale = []
            for name in sorted(expected & names):
                source_path = ROOT / name
                if source_path.exists() and sha256_bytes(archive.read(name)) != sha256_file(source_path):
                    stale.append(name)
        result.update(
            {
                "manifest_row_count": len(rows),
                "manifest_expected_entry_count": len(expected),
                "current_zip_entry_count": len(names),
                "manifest_zip_missing_entry_count": len(expected - names),
                "manifest_zip_extra_entry_count": len(names - expected),
                "manifest_zip_stale_entry_count": len(stale),
                "manifest_duplicate_entry_count": len(duplicate_manifest_entries),
                "current_zip_duplicate_entry_count": len(duplicate_zip_entries),
                "manifest_source_missing_count": len(missing_sources),
                "manifest_zip_missing_entries": sorted(expected - names)[:20],
                "manifest_zip_extra_entries": sorted(names - expected)[:20],
                "manifest_zip_stale_entries": stale[:20],
                "manifest_duplicate_entries": duplicate_manifest_entries[:20],
                "current_zip_duplicate_entries": duplicate_zip_entries[:20],
                "manifest_source_missing_entries": missing_sources[:20],
            }
        )
    except (OSError, csv.Error, zipfile.BadZipFile, KeyError) as exc:
        result["manifest_zip_scan_error"] = f"{type(exc).__name__}: {exc}"
    return result


def package_summary(package: dict[str, Any]) -> dict[str, Any]:
    zip_path = package.get("zip_path", "")
    current_zip = ROOT / zip_path if zip_path else None
    current_zip_exists = bool(current_zip and current_zip.exists())
    current_zip_size = current_zip.stat().st_size if current_zip_exists and current_zip else None
    current_zip_sha = sha256_file(current_zip) if current_zip_exists and current_zip else ""
    manifest_freshness = manifest_zip_freshness(package, current_zip)
    zip_security = current_zip_security_scan(current_zip)
    current_superseded = current_superseded_package_artifacts()
    current_unexpected = current_submission_package_unexpected_artifacts()
    return {
        "status": package.get("status", "missing"),
        "zip_path": zip_path,
        "audit_path": package.get("audit_path", ""),
        "manifest_path": package.get("manifest_path", ""),
        "zip_entries": package.get("zip_entries"),
        "zip_size_bytes": package.get("zip_size_bytes"),
        "zip_sha256": package.get("zip_sha256", ""),
        "current_zip_exists": current_zip_exists,
        "current_zip_size_bytes": current_zip_size,
        "current_zip_sha256": current_zip_sha,
        "current_zip_size_matches_audit": current_zip_size == package.get("zip_size_bytes"),
        "current_zip_sha_matches_audit": current_zip_sha == package.get("zip_sha256"),
        **manifest_freshness,
        **zip_security,
        "stale_entry_count": len(package.get("stale_entries") or []),
        "forbidden_hit_count": len(package.get("forbidden_hits") or []),
        "active_secret_hit_count": package.get("active_secret_hit_count", 0),
        "docx_forbidden_hit_count": package.get("docx_forbidden_hit_count", 0),
        "docx_active_secret_hit_count": package.get("docx_active_secret_hit_count", 0),
        "xlsx_forbidden_hit_count": package.get("xlsx_forbidden_hit_count", 0),
        "xlsx_active_secret_hit_count": package.get("xlsx_active_secret_hit_count", 0),
        "stale_status_marker_hit_count": package.get("stale_status_marker_hit_count", 0),
        "duplicate_manifest_path_count": package.get("duplicate_manifest_path_count", 0),
        "duplicate_zip_entry_count": package.get("duplicate_zip_entry_count", 0),
        "superseded_zip_package_count": package.get("superseded_zip_package_count", 0),
        "superseded_zip_packages": package.get("superseded_zip_packages") or [],
        **current_superseded,
        **current_unexpected,
        "docx_scan_error_count": len(package.get("docx_scan_errors") or []),
        "xlsx_scan_error_count": len(package.get("xlsx_scan_errors") or []),
        "bad_name_count": len(package.get("bad_names") or []),
    }


def handoff_summary(handoff: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": handoff.get("status", "missing"),
        "recipient": handoff.get("recipient", ""),
        "gmail_message_id": handoff.get("gmail_message_id", ""),
        "gmail_thread_id": handoff.get("gmail_thread_id", ""),
        "gmail_display_url": handoff.get("gmail_display_url", ""),
        "gmail_email_ts": handoff.get("gmail_email_ts", ""),
        "gmail_subject": handoff.get("gmail_subject", ""),
        "attachment_filename": handoff.get("attachment_filename", ""),
        "latest_zip_sha256": handoff.get("latest_zip_sha256", ""),
        "latest_zip_size_bytes": handoff.get("latest_zip_size_bytes"),
        "failure_count": handoff.get("failure_count"),
        "failures": handoff.get("failures") or [],
    }


def build_status(require_email_handoff: bool) -> dict[str, Any]:
    package = package_summary(read_json(CORE_PACKAGE_AUDIT_JSON))
    handoff = handoff_summary(read_json(EMAIL_HANDOFF_AUDIT_JSON))
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if package["status"] != "pass":
        failures.append({"check": "core_package_status_pass", "observed": package["status"]})
    if not package["current_zip_exists"]:
        failures.append({"check": "core_package_current_zip_exists", "observed": package["zip_path"]})
    if package["current_zip_exists"] and not package["current_zip_size_matches_audit"]:
        failures.append(
            {
                "check": "core_package_current_zip_size_matches_audit",
                "observed": f"{package['current_zip_size_bytes']}/{package['zip_size_bytes']}",
            }
        )
    if package["current_zip_exists"] and not package["current_zip_sha_matches_audit"]:
        failures.append(
            {
                "check": "core_package_current_zip_sha_matches_audit",
                "observed": f"{package['current_zip_sha256']}/{package['zip_sha256']}",
            }
        )
    if not package["manifest_exists"]:
        failures.append({"check": "core_package_manifest_exists", "observed": package["manifest_path"]})
    if package["manifest_zip_scan_error"]:
        failures.append({"check": "core_package_manifest_zip_scan_error_absent", "observed": package["manifest_zip_scan_error"]})
    if package["manifest_zip_missing_entry_count"] not in (0, None):
        failures.append({"check": "core_package_manifest_zip_missing_entries_zero", "observed": package["manifest_zip_missing_entries"]})
    if package["manifest_zip_extra_entry_count"] not in (0, None):
        failures.append({"check": "core_package_manifest_zip_extra_entries_zero", "observed": package["manifest_zip_extra_entries"]})
    if package["manifest_zip_stale_entry_count"] not in (0, None):
        failures.append({"check": "core_package_manifest_zip_stale_entries_zero", "observed": package["manifest_zip_stale_entries"]})
    if package["manifest_duplicate_entry_count"] not in (0, None):
        failures.append({"check": "core_package_manifest_duplicate_entries_zero", "observed": package["manifest_duplicate_entries"]})
    if package["current_zip_duplicate_entry_count"] not in (0, None):
        failures.append({"check": "core_package_current_zip_duplicate_entries_zero", "observed": package["current_zip_duplicate_entries"]})
    if package["manifest_source_missing_count"] not in (0, None):
        failures.append({"check": "core_package_manifest_source_files_exist", "observed": package["manifest_source_missing_entries"]})
    if package["current_zip_security_scan_error"]:
        failures.append({"check": "core_package_current_zip_security_scan_error_absent", "observed": package["current_zip_security_scan_error"]})
    if package["current_zip_forbidden_hit_count"] not in (0, None):
        failures.append({"check": "core_package_current_zip_forbidden_hits_zero", "observed": package["current_zip_forbidden_hits"]})
    if package["current_zip_active_secret_hit_count"] not in (0, None):
        failures.append({"check": "core_package_current_zip_active_secret_hits_zero", "observed": package["current_zip_active_secret_hits"]})
    if package["current_zip_docx_forbidden_hit_count"] not in (0, None):
        failures.append({"check": "core_package_current_zip_docx_forbidden_hits_zero", "observed": package["current_zip_docx_forbidden_hits"]})
    if package["current_zip_docx_active_secret_hit_count"] not in (0, None):
        failures.append({"check": "core_package_current_zip_docx_active_secret_hits_zero", "observed": package["current_zip_docx_active_secret_hits"]})
    if package["current_zip_docx_scan_error_count"] not in (0, None):
        failures.append({"check": "core_package_current_zip_docx_scan_errors_zero", "observed": package["current_zip_docx_scan_errors"]})
    if package["current_zip_xlsx_forbidden_hit_count"] not in (0, None):
        failures.append({"check": "core_package_current_zip_xlsx_forbidden_hits_zero", "observed": package["current_zip_xlsx_forbidden_hits"]})
    if package["current_zip_xlsx_active_secret_hit_count"] not in (0, None):
        failures.append({"check": "core_package_current_zip_xlsx_active_secret_hits_zero", "observed": package["current_zip_xlsx_active_secret_hits"]})
    if package["current_zip_xlsx_scan_error_count"] not in (0, None):
        failures.append({"check": "core_package_current_zip_xlsx_scan_errors_zero", "observed": package["current_zip_xlsx_scan_errors"]})
    if package["stale_entry_count"] != 0:
        failures.append({"check": "core_package_stale_entries_zero", "observed": package["stale_entry_count"]})
    if package["forbidden_hit_count"] != 0:
        failures.append({"check": "core_package_forbidden_hits_zero", "observed": package["forbidden_hit_count"]})
    if package["active_secret_hit_count"] != 0:
        failures.append({"check": "core_package_active_secret_hits_zero", "observed": package["active_secret_hit_count"]})
    if package["docx_forbidden_hit_count"] != 0:
        failures.append({"check": "core_package_docx_forbidden_hits_zero", "observed": package["docx_forbidden_hit_count"]})
    if package["docx_active_secret_hit_count"] != 0:
        failures.append({"check": "core_package_docx_active_secret_hits_zero", "observed": package["docx_active_secret_hit_count"]})
    if package["docx_scan_error_count"] != 0:
        failures.append({"check": "core_package_docx_scan_errors_zero", "observed": package["docx_scan_error_count"]})
    if package["xlsx_forbidden_hit_count"] != 0:
        failures.append({"check": "core_package_xlsx_forbidden_hits_zero", "observed": package["xlsx_forbidden_hit_count"]})
    if package["xlsx_active_secret_hit_count"] != 0:
        failures.append({"check": "core_package_xlsx_active_secret_hits_zero", "observed": package["xlsx_active_secret_hit_count"]})
    if package["xlsx_scan_error_count"] != 0:
        failures.append({"check": "core_package_xlsx_scan_errors_zero", "observed": package["xlsx_scan_error_count"]})
    if package["stale_status_marker_hit_count"] != 0:
        failures.append(
            {
                "check": "core_package_stale_status_marker_hits_zero",
                "observed": package["stale_status_marker_hit_count"],
            }
        )
    if package["duplicate_manifest_path_count"] != 0:
        failures.append(
            {
                "check": "core_package_audit_duplicate_manifest_paths_zero",
                "observed": package["duplicate_manifest_path_count"],
            }
        )
    if package["duplicate_zip_entry_count"] != 0:
        failures.append(
            {
                "check": "core_package_audit_duplicate_zip_entries_zero",
                "observed": package["duplicate_zip_entry_count"],
            }
        )
    if package["superseded_zip_package_count"] != 0:
        failures.append(
            {
                "check": "core_package_superseded_zip_packages_zero",
                "observed": package["superseded_zip_packages"],
            }
        )
    if package["current_superseded_package_artifact_count"] != 0:
        failures.append(
            {
                "check": "current_submission_package_superseded_artifacts_zero",
                "observed": package["current_superseded_package_artifacts"],
            }
        )
    if package["current_submission_package_unexpected_artifact_count"] != 0:
        failures.append(
            {
                "check": "current_submission_package_unexpected_artifacts_zero",
                "observed": package["current_submission_package_unexpected_artifacts"],
            }
        )
    if package["current_submission_package_missing_required_artifact_count"] != 0:
        failures.append(
            {
                "check": "current_submission_package_required_artifacts_present",
                "observed": package["current_submission_package_missing_required_artifacts"],
            }
        )
    if package["current_submission_package_allowed_non_file_artifact_count"] != 0:
        failures.append(
            {
                "check": "current_submission_package_allowed_artifacts_are_files",
                "observed": package["current_submission_package_allowed_non_file_artifacts"],
            }
        )
    if package["bad_name_count"] != 0:
        failures.append({"check": "core_package_bad_names_zero", "observed": package["bad_name_count"]})
    if handoff["status"] != "pass":
        row = {"check": "email_handoff_status_pass", "observed": handoff["status"]}
        if require_email_handoff:
            failures.append(row)
        else:
            warnings.append(row)

    if failures:
        status = "fail"
    elif warnings:
        status = "pass_with_handoff_refresh_needed"
    else:
        status = "pass"

    return {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "require_email_handoff": require_email_handoff,
        "core_package": package,
        "email_handoff": handoff,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "guardrail": (
            "This post-run status is written after the latest core package build and local email handoff "
            "receipt audit. It verifies the current ZIP file still matches the package audit SHA/size, "
            "rechecks ZIP entries against the current public submission manifest/source files, independently "
            "rescans the current ZIP and nested DOCX/XLSX parts for local-path and active-secret leaks, and "
            "checks the package audit's stale status-marker scan, both recorded/current superseded package counts, "
            "and a strict submission_package allowlist with required-file and file-type checks, then "
            "preserves package/handoff state without requiring console output."
        ),
    }


def write_markdown(status_doc: dict[str, Any]) -> None:
    package = status_doc["core_package"]
    handoff = status_doc["email_handoff"]
    lines = [
        "# Post-Run Artifact Status",
        "",
        f"Status: `{status_doc['status']}`",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Checked at | `{status_doc['checked_at']}` |",
        f"| Require email handoff | `{status_doc['require_email_handoff']}` |",
        f"| Core package status | `{package['status']}` |",
        f"| Core package ZIP | `{package['zip_path']}` |",
        f"| Core package manifest | `{package['manifest_path']}` |",
        f"| Core package SHA-256 | `{package['zip_sha256']}` |",
        f"| Core package size bytes | `{package['zip_size_bytes']}` |",
        f"| Current ZIP exists | `{package['current_zip_exists']}` |",
        f"| Current ZIP SHA-256 | `{package['current_zip_sha256']}` |",
        f"| Current ZIP size bytes | `{package['current_zip_size_bytes']}` |",
        f"| Current ZIP SHA matches audit | `{package['current_zip_sha_matches_audit']}` |",
        f"| Current ZIP size matches audit | `{package['current_zip_size_matches_audit']}` |",
        f"| Manifest exists | `{package['manifest_exists']}` |",
        f"| Manifest rows | `{package['manifest_row_count']}` |",
        f"| Manifest expected entries | `{package['manifest_expected_entry_count']}` |",
        f"| Current ZIP entries | `{package['current_zip_entry_count']}` |",
        f"| Manifest ZIP missing entries | `{package['manifest_zip_missing_entry_count']}` |",
        f"| Manifest ZIP extra entries | `{package['manifest_zip_extra_entry_count']}` |",
        f"| Manifest ZIP stale entries | `{package['manifest_zip_stale_entry_count']}` |",
        f"| Manifest duplicate entries | `{package['manifest_duplicate_entry_count']}` |",
        f"| Current ZIP duplicate entries | `{package['current_zip_duplicate_entry_count']}` |",
        f"| Manifest source missing files | `{package['manifest_source_missing_count']}` |",
        f"| Manifest ZIP scan error | `{package['manifest_zip_scan_error']}` |",
        f"| Current ZIP forbidden hits | `{package['current_zip_forbidden_hit_count']}` |",
        f"| Current ZIP active secret hits | `{package['current_zip_active_secret_hit_count']}` |",
        f"| Current ZIP DOCX forbidden hits | `{package['current_zip_docx_forbidden_hit_count']}` |",
        f"| Current ZIP DOCX active secret hits | `{package['current_zip_docx_active_secret_hit_count']}` |",
        f"| Current ZIP DOCX scan errors | `{package['current_zip_docx_scan_error_count']}` |",
        f"| Current ZIP XLSX forbidden hits | `{package['current_zip_xlsx_forbidden_hit_count']}` |",
        f"| Current ZIP XLSX active secret hits | `{package['current_zip_xlsx_active_secret_hit_count']}` |",
        f"| Current ZIP XLSX scan errors | `{package['current_zip_xlsx_scan_error_count']}` |",
        f"| Current ZIP security scan error | `{package['current_zip_security_scan_error']}` |",
        f"| Core package stale entries | `{package['stale_entry_count']}` |",
        f"| Core package forbidden hits | `{package['forbidden_hit_count']}` |",
        f"| Core package active secret hits | `{package['active_secret_hit_count']}` |",
        f"| Core package DOCX forbidden hits | `{package['docx_forbidden_hit_count']}` |",
        f"| Core package DOCX active secret hits | `{package['docx_active_secret_hit_count']}` |",
        f"| Core package DOCX scan errors | `{package['docx_scan_error_count']}` |",
        f"| Core package XLSX forbidden hits | `{package['xlsx_forbidden_hit_count']}` |",
        f"| Core package XLSX active secret hits | `{package['xlsx_active_secret_hit_count']}` |",
        f"| Core package XLSX scan errors | `{package['xlsx_scan_error_count']}` |",
        f"| Core package stale status marker hits | `{package['stale_status_marker_hit_count']}` |",
        f"| Core package audit duplicate manifest paths | `{package['duplicate_manifest_path_count']}` |",
        f"| Core package audit duplicate ZIP entries | `{package['duplicate_zip_entry_count']}` |",
        f"| Core package superseded ZIP packages | `{package['superseded_zip_package_count']}` |",
        f"| Current superseded package artifacts | `{package['current_superseded_package_artifact_count']}` |",
        f"| Current unexpected submission package artifacts | `{package['current_submission_package_unexpected_artifact_count']}` |",
        f"| Missing required submission package artifacts | `{package['current_submission_package_missing_required_artifact_count']}` |",
        f"| Allowed submission package non-file artifacts | `{package['current_submission_package_allowed_non_file_artifact_count']}` |",
        f"| Core package bad names | `{package['bad_name_count']}` |",
        f"| Email handoff status | `{handoff['status']}` |",
        f"| Email handoff message ID | `{handoff['gmail_message_id']}` |",
        f"| Email handoff thread ID | `{handoff['gmail_thread_id']}` |",
        f"| Email handoff display URL | `{handoff['gmail_display_url']}` |",
        f"| Email handoff timestamp | `{handoff['gmail_email_ts']}` |",
        f"| Email handoff attachment | `{handoff['attachment_filename']}` |",
        f"| Email handoff SHA-256 | `{handoff['latest_zip_sha256']}` |",
        f"| Email handoff size bytes | `{handoff['latest_zip_size_bytes']}` |",
        f"| Failures | `{status_doc['failure_count']}` |",
        f"| Warnings | `{status_doc['warning_count']}` |",
        "",
        "## Warnings",
        "",
    ]
    if status_doc["warnings"]:
        lines.extend(["| Check | Observed |", "|---|---|"])
        for row in status_doc["warnings"]:
            lines.append(f"| `{row['check']}` | `{row['observed']}` |")
    else:
        lines.append("No post-run warnings.")
    lines.extend(["", "## Failures", ""])
    if status_doc["failures"]:
        lines.extend(["| Check | Observed |", "|---|---|"])
        for row in status_doc["failures"]:
            lines.append(f"| `{row['check']}` | `{row['observed']}` |")
    else:
        lines.append("No post-run failures.")
    lines.extend(["", "## Guardrail", "", status_doc["guardrail"], ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-email-handoff", action="store_true")
    args = parser.parse_args()
    status_doc = build_status(args.require_email_handoff)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(status_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(status_doc)
    print(json.dumps({"status": status_doc["status"], "warnings": status_doc["warning_count"], "failures": status_doc["failure_count"]}, indent=2))
    if status_doc["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
