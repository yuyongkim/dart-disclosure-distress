from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
OUT_JSON = ROOT / "results" / "remaining_risk_audit.json"
OUT_MD = ROOT / "results" / "remaining_risk_audit.md"

EXPECTED_EXTERNAL_PHASES = {
    "RQ1 alignment": "Human labels are not ready until the workbook or single-expert sheet is filled, synced, and evaluated.",
    "EDINET/Japanese disclosure data": "Empirical EDINET use requires an API key, archive or prior-collection coverage, and nonzero document/current-report rows.",
    "SEC all-8-K body-text scope": "The all-8-K body-text scope is not ready until storage headroom and an explicit SEC/EDGAR user agent are available.",
}

EXPECTED_EXTERNAL_BLOCKERS = {
    "RQ1 alignment": {
        "status": "blocked_missing_human_labels",
        "blockers": {
            "main_evaluation_not_ready",
            "no_gold_labels",
            "single_expert_labels_missing",
            "supplemental_sec_labels_missing",
            "two_annotator_labels_missing",
        },
    },
    "EDINET/Japanese disclosure data": {
        "status": "blocked_key_and_archive_gap",
        "blockers": {
            "missing_env_edinet_api_key",
            "archive_or_prior_collection_required",
            "no_edinet_document_rows_collected",
            "no_current_report_candidates_collected",
            "no_external_reusable_edinet_dataset_found",
        },
    },
    "SEC all-8-K body-text scope": {
        "status": "blocked_storage_and_user_agent",
        "blockers": {
            "insufficient_storage_for_full_scope",
            "missing_env_sec_user_agent",
        },
    },
}

TRACEABILITY_EXTERNAL_BLOCKER_ROWS = {
    "RQ1 alignment": {
        "phase": "RQ1 alignment",
        "requirement": "Report Precision@1/MRR against a 100-row human gold standard",
        "status": "deferred_v2_scope",
        "action_markers": ["filling human labels", "evaluate_alignment.py"],
    },
    "EDINET/Japanese disclosure data": {
        "phase": "Data layer",
        "requirement": "Prepare EDINET/Japanese disclosure extension path",
        "status": "deferred_v2_scope",
        "action_markers": ["EDINET_API_KEY", "nonzero current-report rows"],
    },
    "SEC all-8-K body-text scope": {
        "phase": "SEC v2 full-text extension",
        "requirement": "Track full all-8-K filing-body scope before any SEC full-text validation claim",
        "status": "deferred_v2_scope",
        "action_markers": ["explicit SEC user agent", "storage"],
    },
}

JOURNAL_TEXT_RESIDUE_PATTERNS = [
    r"&amp;",
    r"<scp>",
    r"</scp>",
    "\u2010",
]

TEXT_PACKAGE_SUFFIXES = {".bib", ".csv", ".json", ".md", ".txt"}
PUBLIC_PACKAGE_STALE_STATUS_MARKERS = [
    "pass_with_handoff_refresh_needed",
    "fail_internal_risk",
    "blocked_claims_or_text_integrity_failures",
    "Traceback (most recent call last)",
    '"status": "fail"',
    '"internal_failure_count": 1',
    '"warnings": 1',
    '"failures": 1',
]

CORE_POST_RUN_AUDIT_SCRIPTS = [
    "src/audit/alignment_label_preflight.py",
    "src/audit/build_core_submission_package.py",
    "src/audit/build_calculation_sheet.py",
    "src/audit/build_paper_figures.py",
    "src/audit/bilingual_manuscript_audit.py",
    "src/audit/completion_audit.py",
    "src/audit/edinet_execution_preflight.py",
    "src/audit/email_handoff_receipt_audit.py",
    "src/audit/final_claim_boundary_audit.py",
    "src/audit/journal_candidate_source_audit.py",
    "src/audit/manuscript_claim_audit.py",
    "src/audit/manuscript_metric_audit.py",
    "src/audit/market_residual_coverage_audit.py",
    "src/audit/post_run_artifact_status.py",
    "src/audit/public_path_redaction.py",
    "src/audit/public_submission_manifest_audit.py",
    "src/audit/reference_consistency_audit.py",
    "src/audit/remaining_risk_audit.py",
    "src/audit/research_plan_traceability.py",
    "src/audit/sec_all8k_storage_preflight.py",
    "src/audit/sec_validation_scope.py",
    "src/audit/submission_package_audit.py",
    "src/audit/word_export_audit.py",
]

PUBLIC_MANIFEST_FIELDNAMES = [
    "path",
    "category",
    "role",
    "exists",
    "size_bytes",
    "sha256",
    "denied_reasons",
]

SMOKE_ALLOWED_STATUSES = {
    "src/audit/alignment_label_preflight.py": {"blocked_missing_human_labels"},
    "src/audit/bilingual_manuscript_audit.py": {"pass"},
    "src/audit/build_calculation_sheet.py": {"pass"},
    "src/audit/build_paper_figures.py": {"pass"},
    "src/audit/completion_audit.py": {"pass"},
    "src/audit/edinet_execution_preflight.py": {"blocked_key_and_archive_gap"},
    "src/audit/journal_candidate_source_audit.py": {"pass"},
    "src/audit/manuscript_claim_audit.py": {"passed"},
    "src/audit/manuscript_metric_audit.py": {"passed"},
    "src/audit/market_residual_coverage_audit.py": {"pass"},
    "src/audit/post_run_artifact_status.py": {"pass", "pass_with_handoff_refresh_needed"},
    "src/audit/reference_consistency_audit.py": {"pass"},
    "src/audit/research_plan_traceability.py": {"v1_traceable_for_submission_scope"},
    "src/audit/sec_all8k_storage_preflight.py": {"blocked_storage_and_user_agent"},
    "src/audit/sec_validation_scope.py": {"created"},
    "src/audit/submission_package_audit.py": {"pass"},
    "src/audit/word_export_audit.py": {"pass"},
}

EXPECTED_EXTERNAL_CWD_SMOKE_SCRIPTS = [
    "src/audit/post_run_artifact_status.py",
    "src/audit/build_calculation_sheet.py",
    "src/audit/completion_audit.py",
    "src/audit/alignment_label_preflight.py",
    "src/audit/edinet_execution_preflight.py",
    "src/audit/sec_all8k_storage_preflight.py",
    "src/audit/sec_validation_scope.py",
    "src/audit/build_paper_figures.py",
    "src/audit/bilingual_manuscript_audit.py",
    "src/audit/journal_candidate_source_audit.py",
    "src/audit/manuscript_claim_audit.py",
    "src/audit/manuscript_metric_audit.py",
    "src/audit/market_residual_coverage_audit.py",
    "src/audit/reference_consistency_audit.py",
    "src/audit/research_plan_traceability.py",
    "src/audit/word_export_audit.py",
    "src/audit/submission_package_audit.py",
]

EXPECTED_EXTERNAL_CWD_SMOKE_ARGS = {
    "src/audit/post_run_artifact_status.py": [],
    "src/audit/build_calculation_sheet.py": ["--check-only"],
    "src/audit/completion_audit.py": ["--check-only"],
    "src/audit/alignment_label_preflight.py": [],
    "src/audit/edinet_execution_preflight.py": [],
    "src/audit/sec_all8k_storage_preflight.py": [],
    "src/audit/sec_validation_scope.py": [],
    "src/audit/build_paper_figures.py": [],
    "src/audit/bilingual_manuscript_audit.py": [],
    "src/audit/journal_candidate_source_audit.py": ["--check-only"],
    "src/audit/manuscript_claim_audit.py": [],
    "src/audit/manuscript_metric_audit.py": [],
    "src/audit/market_residual_coverage_audit.py": [],
    "src/audit/reference_consistency_audit.py": [],
    "src/audit/research_plan_traceability.py": [],
    "src/audit/word_export_audit.py": [],
    "src/audit/submission_package_audit.py": ["--check-only"],
}

LOCAL_PATH_TOKEN_PARTS = (
    ("win_users_forward", ("C:", "/Users")),
    ("win_users_single_sep", ("C:", "\\", "Users")),
    ("win_users_double_sep", ("C:", "\\\\", "Users")),
    ("ma_forward", ("Desktop", "/", "M&A")),
    ("ma_single_sep", ("Desktop", "\\", "M&A")),
    ("ma_double_sep", ("Desktop", "\\\\", "M&A")),
    ("paper_forward", ("Desktop", "/", "Paper 4")),
    ("paper_single_sep", ("Desktop", "\\", "Paper 4")),
    ("paper_double_sep", ("Desktop", "\\\\", "Paper 4")),
    ("local_external_marker", ("desktop", "_external")),
    ("prior_work_marker", ("grant", "_back")),
    ("sec_local_marker", ("SEC", " Desktop")),
)

TEXT_SURFACE_CONSTANTS = {
    "src/audit/audit_forbidden_tokens.py": "FORBIDDEN_STALE_TOKENS",
    "src/audit/public_path_redaction.py": "FORBIDDEN_PATTERNS",
    "src/audit/public_submission_manifest_audit.py": "FORBIDDEN_TEXT_PATTERNS",
}

BYTE_SURFACE_CONSTANTS = {
    "src/audit/build_core_submission_package.py": "FORBIDDEN_BYTES",
    "src/audit/post_run_artifact_status.py": "FORBIDDEN_BYTES",
}

ACTIVE_SECRET_SOURCES = [
    "src/audit/public_path_redaction.py",
    "src/audit/public_submission_manifest_audit.py",
    "src/audit/build_core_submission_package.py",
    "src/audit/post_run_artifact_status.py",
]

ACTIVE_SECRET_ENV_NAMES = (
    "DART_API_KEY",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_API_KEY",
    "EDINET_API_KEY",
    "SEC_USER_AGENT",
    "EDGAR_USER_AGENT",
    "NEO4J_PASSWORD",
)
GMAIL_ID_RE = re.compile(r"^[0-9a-f]+$")
FULL_ISO_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?$"
)
EXPECTED_GMAIL_SUBJECT = "Re: Paper 4 \ud575\uc2ec \ub17c\ubb38 \ud328\ud0a4\uc9c0 \ucd5c\uc2e0\ubcf8 \uc7ac\uc1a1\ubd80"


def read_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    if not path.exists():
        return {"_missing": True, "_path": relative_path}
    return json.loads(path.read_text(encoding="utf-8"))


def value(doc: dict[str, Any], key: str, default: Any = None) -> Any:
    return doc.get(key, default) if isinstance(doc, dict) else default


def as_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw if raw is not None else default)
    except (TypeError, ValueError):
        return default


def is_iso_like_timestamp(raw: Any) -> bool:
    text = str(raw)
    if not FULL_ISO_TIMESTAMP_RE.fullmatch(text):
        return False
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def gmail_display_url_matches_message_id(display_url: Any, message_id: Any) -> bool:
    message_id_text = str(message_id)
    return bool(message_id_text) and str(display_url) == f"https://mail.google.com/mail/#all/{message_id_text}"


def is_gmail_hex_id(value: Any) -> bool:
    text = str(value)
    return len(text) >= 12 and bool(GMAIL_ID_RE.fullmatch(text))


def check(name: str, ok: bool, observed: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "observed": observed}


def local_path_surface_tokens() -> list[tuple[str, str]]:
    return [(label, "".join(parts)) for label, parts in LOCAL_PATH_TOKEN_PARTS]


def evaluate_static_expr(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [evaluate_static_expr(item) for item in node.elts]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = evaluate_static_expr(node.left)
        right = evaluate_static_expr(node.right)
        if isinstance(left, (str, bytes)) and isinstance(right, type(left)):
            return left + right
    if isinstance(node, ast.Call) and node.args:
        return evaluate_static_expr(node.args[0])
    raise ValueError(f"Unsupported static expression: {ast.dump(node, include_attributes=False)}")


def static_assignment_value(source_path: Path, constant_name: str) -> Any:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=source_path.as_posix())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == constant_name:
                    return evaluate_static_expr(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == constant_name:
            return evaluate_static_expr(node.value)
    raise KeyError(f"{constant_name} not found in {source_path.as_posix()}")


def required_public_package_paths() -> list[str]:
    required_paths = static_assignment_value(
        ROOT / "src" / "audit" / "public_submission_manifest_audit.py",
        "REQUIRED_PACKAGE_FILES",
    )
    if not isinstance(required_paths, list) or not all(isinstance(path, str) for path in required_paths):
        raise ValueError("REQUIRED_PACKAGE_FILES did not resolve to a path string list")
    return required_paths


def expected_public_package_file_count() -> int | None:
    try:
        return len(required_public_package_paths())
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError, KeyError):
        return None


def public_manifest_zip_snapshot() -> dict[str, Any]:
    manifest_path = ROOT / "data" / "audit" / "public_submission_manifest.csv"
    zip_path = ROOT / "submission_package" / "paper4_core_package_latest.zip"
    snapshot: dict[str, Any] = {
        "manifest_path": manifest_path.as_posix(),
        "zip_path": zip_path.as_posix(),
        "manifest_exists": manifest_path.exists(),
        "zip_exists": zip_path.exists(),
        "manifest_row_count": None,
        "manifest_blank_path_count": None,
        "manifest_fieldnames": [],
        "manifest_expected_path_count": None,
        "manifest_path_order_matches_required": None,
        "manifest_missing_expected_paths": [],
        "manifest_unexpected_paths": [],
        "manifest_metadata_mismatch_count": None,
        "manifest_metadata_mismatches": [],
        "manifest_entry_count": None,
        "zip_entry_count": None,
        "manifest_duplicate_entry_count": None,
        "zip_duplicate_entry_count": None,
        "manifest_duplicate_entries": [],
        "zip_duplicate_entries": [],
        "manifest_source_missing_count": None,
        "manifest_source_missing_entries": [],
        "missing": [],
        "extra": [],
        "stale": [],
        "scan_error": "",
    }
    if not manifest_path.exists() or not zip_path.exists():
        return snapshot
    try:
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            manifest_source_rows = list(reader)
            manifest_fieldnames = list(reader.fieldnames or [])
            manifest_rows = [
                str(row.get("path", "")).replace("\\", "/").lstrip("/")
                for row in manifest_source_rows
                if str(row.get("path", "")).strip()
            ]
        manifest_blank_path_count = sum(
            1 for row in manifest_source_rows if not str(row.get("path", "")).strip()
        )
        required_manifest_paths = required_public_package_paths()
        metadata_mismatches: list[dict[str, Any]] = []
        for row in manifest_source_rows:
            rel_path = str(row.get("path", "")).replace("\\", "/").lstrip("/")
            if not rel_path.strip():
                continue
            source_path = ROOT / rel_path
            actual_exists = source_path.exists()
            actual_size = str(source_path.stat().st_size) if actual_exists else "0"
            actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest() if actual_exists else ""
            expected_values = {
                "exists": str(actual_exists),
                "size_bytes": actual_size,
                "sha256": actual_sha,
                "denied_reasons": "",
            }
            mismatched_fields = [
                field
                for field, expected in expected_values.items()
                if str(row.get(field, "")) != expected
            ]
            if mismatched_fields:
                metadata_mismatches.append(
                    {
                        "path": rel_path,
                        "mismatched_fields": mismatched_fields,
                    }
                )
        manifest_duplicate_entries = sorted(
            path for path, count in Counter(manifest_rows).items() if count > 1
        )
        manifest_paths = sorted(set(manifest_rows))
        source_missing = sorted(rel_path for rel_path in manifest_paths if not (ROOT / rel_path).exists())
        with zipfile.ZipFile(zip_path, "r") as archive:
            zip_rows = archive.namelist()
            zip_duplicate_entries = sorted(
                path for path, count in Counter(zip_rows).items() if count > 1
            )
            zip_names = set(zip_rows)
            stale = []
            for rel_path in manifest_paths:
                source_path = ROOT / rel_path
                if rel_path not in zip_names or not source_path.exists():
                    continue
                source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
                zip_hash = hashlib.sha256(archive.read(rel_path)).hexdigest()
                if source_hash != zip_hash:
                    stale.append(rel_path)
        snapshot.update(
            {
                "manifest_row_count": len(manifest_source_rows),
                "manifest_blank_path_count": manifest_blank_path_count,
                "manifest_fieldnames": manifest_fieldnames,
                "manifest_expected_path_count": len(required_manifest_paths),
                "manifest_path_order_matches_required": manifest_rows == required_manifest_paths,
                "manifest_missing_expected_paths": sorted(set(required_manifest_paths) - set(manifest_rows)),
                "manifest_unexpected_paths": sorted(set(manifest_rows) - set(required_manifest_paths)),
                "manifest_metadata_mismatch_count": len(metadata_mismatches),
                "manifest_metadata_mismatches": metadata_mismatches[:20],
                "manifest_entry_count": len(manifest_paths),
                "zip_entry_count": len(zip_names),
                "manifest_duplicate_entry_count": len(manifest_duplicate_entries),
                "zip_duplicate_entry_count": len(zip_duplicate_entries),
                "manifest_duplicate_entries": manifest_duplicate_entries,
                "zip_duplicate_entries": zip_duplicate_entries,
                "manifest_source_missing_count": len(source_missing),
                "manifest_source_missing_entries": source_missing,
                "missing": sorted(set(manifest_paths) - zip_names),
                "extra": sorted(zip_names - set(manifest_paths)),
                "stale": sorted(stale),
            }
        )
    except (OSError, UnicodeError, SyntaxError, ValueError, TypeError, zipfile.BadZipFile, KeyError, csv.Error) as exc:
        snapshot["scan_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def package_snapshot_is_fresh(snapshot: dict[str, Any]) -> bool:
    expected_count = snapshot["manifest_expected_path_count"]
    return (
        snapshot["manifest_exists"] is True
        and snapshot["zip_exists"] is True
        and isinstance(expected_count, int)
        and expected_count > 0
        and snapshot["manifest_row_count"] == expected_count
        and snapshot["manifest_blank_path_count"] == 0
        and snapshot["manifest_fieldnames"] == PUBLIC_MANIFEST_FIELDNAMES
        and snapshot["manifest_expected_path_count"] == expected_count
        and snapshot["manifest_path_order_matches_required"] is True
        and snapshot["manifest_missing_expected_paths"] == []
        and snapshot["manifest_unexpected_paths"] == []
        and snapshot["manifest_metadata_mismatch_count"] == 0
        and snapshot["manifest_metadata_mismatches"] == []
        and snapshot["manifest_entry_count"] == snapshot["zip_entry_count"] == expected_count
        and snapshot["manifest_duplicate_entry_count"] == 0
        and snapshot["zip_duplicate_entry_count"] == 0
        and snapshot["manifest_duplicate_entries"] == []
        and snapshot["zip_duplicate_entries"] == []
        and snapshot["manifest_source_missing_count"] == 0
        and snapshot["manifest_source_missing_entries"] == []
        and snapshot["missing"] == []
        and snapshot["extra"] == []
        and snapshot["stale"] == []
        and snapshot["scan_error"] == ""
    )


def journal_text_residue_hits() -> list[dict[str, Any]]:
    paths = [
        ROOT / "data" / "audit" / "submission_candidate_journal_sources.json",
        ROOT / "results" / "journal_candidate_source_audit.json",
        ROOT / "results" / "journal_candidate_source_audit.md",
    ]
    hits: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            hits.append({"path": str(path), "pattern": "missing", "line_numbers": []})
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in JOURNAL_TEXT_RESIDUE_PATTERNS:
            lines = [
                idx
                for idx, line in enumerate(text.splitlines(), start=1)
                if re.search(pattern, line, flags=re.IGNORECASE)
            ]
            if lines:
                hits.append({"path": str(path), "pattern": pattern, "line_numbers": lines})
    return hits


def pipeline_doc_sync_check() -> dict[str, Any]:
    manifest = read_json("reproducibility_manifest.json")
    run_report = read_json("reproducibility_run_report.json")
    if not isinstance(manifest, list):
        return check("pipeline_status_docs_match_manifest", False, {"reason": "manifest_not_list"})
    manifest_count = len(manifest)
    heavy_count = sum(1 for row in manifest if row.get("heavy") is True)
    expected_ok = manifest_count - heavy_count
    expected_freshness_exempt_outputs = sum(len(row.get("freshness_exempt_outputs", [])) for row in manifest)
    run_counts = {
        "step_count": value(run_report, "step_count"),
        "ok_count": value(run_report, "ok_count"),
        "skipped_heavy_count": value(run_report, "skipped_heavy_count"),
        "failed_step_count": value(run_report, "failed_step_count"),
        "output_failure_count": value(run_report, "output_failure_count"),
        "freshness_exempt_output_count": value(run_report, "freshness_exempt_output_count"),
    }
    run_report_current = (
        run_counts["step_count"] == manifest_count
        and run_counts["ok_count"] == expected_ok
        and run_counts["skipped_heavy_count"] == heavy_count
        and run_counts["failed_step_count"] == 0
        and run_counts["output_failure_count"] == 0
        and run_counts["freshness_exempt_output_count"] == expected_freshness_exempt_outputs
    )
    expected_strings = [
        f"{manifest_count} steps",
        f"{expected_ok} ok",
        f"{heavy_count} skipped_heavy",
        "0 failed",
        "0 output failures",
        f"{expected_freshness_exempt_outputs} freshness-exempt outputs",
    ]
    data_map_text = (ROOT / "data_map.md").read_text(encoding="utf-8")
    status_text = (ROOT / "paper4_execution_status.md").read_text(encoding="utf-8")
    observed = {
        "manifest_count": manifest_count,
        "heavy_count": heavy_count,
        "expected_ok": expected_ok,
        "expected_freshness_exempt_outputs": expected_freshness_exempt_outputs,
        "run_report_current": run_report_current,
        "run_report_check": "current" if run_report_current else "deferred_to_post_run_self_audit",
        "run_counts_if_current": run_counts if run_report_current else {},
        "data_map_has_manifest_count": f"| Reproducibility manifest | {manifest_count} steps |" in data_map_text,
        "data_map_has_report_count": f"| Pipeline execution report | {manifest_count} steps |" in data_map_text,
        "data_map_has_self_audit_count": f"| Pipeline execution report self-audit | {manifest_count} steps |" in data_map_text,
        "status_text_expected_strings": {
            text: text in status_text for text in expected_strings
        },
    }
    ok = (
        observed["data_map_has_manifest_count"]
        and observed["data_map_has_report_count"]
        and observed["data_map_has_self_audit_count"]
        and all(observed["status_text_expected_strings"].values())
    )
    return check("pipeline_status_docs_match_manifest", ok, observed)


def final_claim_boundary_doc_sync_check() -> dict[str, Any]:
    expected_scanned_count = 46
    required_mentions = [
        f"{expected_scanned_count} Markdown files",
        "public-submission-manifest Markdown",
        "paper4_research_plan.md",
        "results/remaining_risk_audit.md",
        "results/public_submission_manifest_audit.md",
        "results/post_run_artifact_status.md",
        "results/market_residual_coverage_audit.md",
        "submission_package/email_handoff_receipt_audit.md",
        "submission_package/LATEST_PACKAGE.md",
    ]
    final_claim_source = (ROOT / "src" / "audit" / "final_claim_boundary_audit.py").read_text(encoding="utf-8")
    docs = {
        "data_map.md": (ROOT / "data_map.md").read_text(encoding="utf-8"),
        "paper4_execution_status.md": (ROOT / "paper4_execution_status.md").read_text(encoding="utf-8"),
        "REPRODUCIBILITY.md": (ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8"),
        "VERSIONS.md": (ROOT / "VERSIONS.md").read_text(encoding="utf-8"),
    }
    source_checks = {
        "has_expected_count_constant": f"EXPECTED_SCANNED_FILE_COUNT = {expected_scanned_count}" in final_claim_source,
        "loads_public_submission_manifest": "PUBLIC_SUBMISSION_MANIFEST" in final_claim_source
        and "manifest_markdown_paths" in final_claim_source,
        "requires_public_submission_manifest_exists": "public_submission_manifest_exists" in final_claim_source
        and "PUBLIC_SUBMISSION_MANIFEST.exists()" in final_claim_source,
        "requires_nonempty_manifest_markdown": "len(manifest_markdown_presence) > 0" in final_claim_source,
        "requires_research_plan": "REQUIRED_SCANNED_FILES" in final_claim_source and "paper4_research_plan.md" in final_claim_source,
        "requires_remaining_risk": "REQUIRED_SCANNED_FILES" in final_claim_source and "remaining_risk_audit.md" in final_claim_source,
        "requires_public_submission_manifest": "REQUIRED_SCANNED_FILES" in final_claim_source
        and "public_submission_manifest_audit.md" in final_claim_source,
        "requires_post_run_status": "REQUIRED_SCANNED_FILES" in final_claim_source
        and "post_run_artifact_status.md" in final_claim_source,
        "requires_email_handoff": "REQUIRED_SCANNED_FILES" in final_claim_source
        and "email_handoff_receipt_audit.md" in final_claim_source,
        "requires_latest_package_pointer": "REQUIRED_SCANNED_FILES" in final_claim_source
        and "LATEST_PACKAGE.md" in final_claim_source,
        "blocks_scope_failure": 'not scope_check["ok"]' in final_claim_source,
        "exits_nonzero_on_non_pass_status": 'audit["status"] != "passed"' in final_claim_source,
    }
    observed = {
        "expected_scanned_count": expected_scanned_count,
        "source_checks": source_checks,
        "doc_mentions": {
            name: {text: text in content for text in required_mentions}
            for name, content in docs.items()
        },
    }
    ok = (
        all(source_checks.values())
        and all(all(row.values()) for row in observed["doc_mentions"].values())
    )
    return check("final_claim_boundary_docs_match_static_scope", ok, observed)


def remaining_risk_data_map_sync_check(expected_internal_check_count: int, external_blocker_count: int) -> dict[str, Any]:
    data_map_text = (ROOT / "data_map.md").read_text(encoding="utf-8")
    expected_fragment = (
        f"| Remaining risk audit | {expected_internal_check_count} internal checks; "
        f"{external_blocker_count} active external blocker phases in current audit JSON |"
    )
    observed = {
        "expected_internal_check_count": expected_internal_check_count,
        "expected_external_blocker_count": external_blocker_count,
        "expected_fragment_present": expected_fragment in data_map_text,
    }
    return check("remaining_risk_data_map_summary_matches_audit", observed["expected_fragment_present"], observed)


def email_handoff_audit_contract_check(email_handoff: dict[str, Any]) -> dict[str, Any]:
    data_map_text = (ROOT / "data_map.md").read_text(encoding="utf-8")
    expected_check_count = value(email_handoff, "check_count")
    expected_fragment = f"| Email handoff receipt audit | {expected_check_count} local receipt checks |"
    observed = {
        "expected_check_count": expected_check_count,
        "email_handoff_status": value(email_handoff, "status"),
        "email_handoff_failure_count": value(email_handoff, "failure_count"),
        "latest_package_sha_recorded": bool(value(email_handoff, "latest_zip_sha256")),
        "latest_package_size_recorded": isinstance(value(email_handoff, "latest_zip_size_bytes"), int),
        "zip_attachment_sha_limit_documented": value(email_handoff, "gmail_attachment_content_sha256_verified") is False
        and "unsupported_attachment_type" in str(value(email_handoff, "gmail_attachment_content_sha256_verification_limit", "")),
        "expected_fragment_present": expected_fragment in data_map_text,
    }
    return check(
        "email_handoff_audit_contract_documented",
        isinstance(expected_check_count, int)
        and expected_check_count >= 18
        and observed["latest_package_sha_recorded"]
        and observed["latest_package_size_recorded"]
        and observed["zip_attachment_sha_limit_documented"]
        and observed["expected_fragment_present"],
        observed,
    )


def forbidden_surface_consistency_check() -> dict[str, Any]:
    expected_text = local_path_surface_tokens()
    expected_bytes = [(label, token.encode("utf-8")) for label, token in expected_text]
    text_rows: list[dict[str, Any]] = []
    byte_rows: list[dict[str, Any]] = []
    secret_rows: list[dict[str, Any]] = []

    for rel_path, constant_name in TEXT_SURFACE_CONSTANTS.items():
        source_path = ROOT / rel_path
        row: dict[str, Any] = {
            "path": rel_path,
            "constant": constant_name,
            "exists": source_path.exists(),
            "declared_count": None,
            "non_text_value_count": None,
            "missing_token_labels": [],
            "error": "",
        }
        try:
            declared = static_assignment_value(source_path, constant_name)
            values = list(declared) if isinstance(declared, list) else []
            row["declared_count"] = len(values)
            row["non_text_value_count"] = sum(1 for item in values if not isinstance(item, str))
            row["missing_token_labels"] = [
                label for label, token in expected_text if token not in values
            ]
        except (OSError, SyntaxError, KeyError, ValueError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        text_rows.append(row)

    for rel_path, constant_name in BYTE_SURFACE_CONSTANTS.items():
        source_path = ROOT / rel_path
        row = {
            "path": rel_path,
            "constant": constant_name,
            "exists": source_path.exists(),
            "declared_count": None,
            "non_byte_value_count": None,
            "missing_token_labels": [],
            "error": "",
        }
        try:
            declared = static_assignment_value(source_path, constant_name)
            values = list(declared) if isinstance(declared, list) else []
            row["declared_count"] = len(values)
            row["non_byte_value_count"] = sum(1 for item in values if not isinstance(item, bytes))
            row["missing_token_labels"] = [
                label for label, token in expected_bytes if token not in values
            ]
        except (OSError, SyntaxError, KeyError, ValueError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        byte_rows.append(row)

    for rel_path in ACTIVE_SECRET_SOURCES:
        source_path = ROOT / rel_path
        row = {
            "path": rel_path,
            "constant": "ACTIVE_SECRET_RE",
            "exists": source_path.exists(),
            "missing_env_names": [],
            "regex_length": None,
            "error": "",
        }
        try:
            regex_text = static_assignment_value(source_path, "ACTIVE_SECRET_RE")
            row["regex_length"] = len(regex_text) if isinstance(regex_text, str) else None
            row["missing_env_names"] = [
                name for name in ACTIVE_SECRET_ENV_NAMES if name not in str(regex_text)
            ]
        except (OSError, SyntaxError, KeyError, ValueError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        secret_rows.append(row)

    ok = (
        all(
            row["exists"]
            and row["error"] == ""
            and row["non_text_value_count"] == 0
            and row["missing_token_labels"] == []
            for row in text_rows
        )
        and all(
            row["exists"]
            and row["error"] == ""
            and row["non_byte_value_count"] == 0
            and row["missing_token_labels"] == []
            for row in byte_rows
        )
        and all(
            row["exists"]
            and row["error"] == ""
            and row["missing_env_names"] == []
            for row in secret_rows
        )
    )
    return check(
        "forbidden_surface_consistency_static",
        ok,
        {
            "expected_local_path_token_labels": [label for label, _ in expected_text],
            "expected_active_secret_env_count": len(ACTIVE_SECRET_ENV_NAMES),
            "text_sources": text_rows,
            "byte_sources": byte_rows,
            "active_secret_sources": secret_rows,
        },
    )


def post_run_artifact_status_check(post_run: dict[str, Any]) -> dict[str, Any]:
    package = value(post_run, "core_package", {})
    handoff = value(post_run, "email_handoff", {})
    expected_package_count = expected_public_package_file_count()
    observed = {
        "status": value(post_run, "status"),
        "failure_count": value(post_run, "failure_count"),
        "warning_count": value(post_run, "warning_count"),
        "require_email_handoff": value(post_run, "require_email_handoff"),
        "expected_package_file_count": expected_package_count,
        "core_package_status": value(package, "status"),
        "current_zip_exists": value(package, "current_zip_exists"),
        "current_zip_sha_matches_audit": value(package, "current_zip_sha_matches_audit"),
        "current_zip_size_matches_audit": value(package, "current_zip_size_matches_audit"),
        "manifest_exists": value(package, "manifest_exists"),
        "manifest_expected_entry_count": value(package, "manifest_expected_entry_count"),
        "current_zip_entry_count": value(package, "current_zip_entry_count"),
        "manifest_zip_missing_entry_count": value(package, "manifest_zip_missing_entry_count"),
        "manifest_zip_extra_entry_count": value(package, "manifest_zip_extra_entry_count"),
        "manifest_zip_stale_entry_count": value(package, "manifest_zip_stale_entry_count"),
        "manifest_duplicate_entry_count": value(package, "manifest_duplicate_entry_count"),
        "current_zip_duplicate_entry_count": value(package, "current_zip_duplicate_entry_count"),
        "manifest_source_missing_count": value(package, "manifest_source_missing_count"),
        "manifest_zip_scan_error": value(package, "manifest_zip_scan_error"),
        "current_zip_forbidden_hit_count": value(package, "current_zip_forbidden_hit_count"),
        "current_zip_active_secret_hit_count": value(package, "current_zip_active_secret_hit_count"),
        "current_zip_docx_forbidden_hit_count": value(package, "current_zip_docx_forbidden_hit_count"),
        "current_zip_docx_active_secret_hit_count": value(package, "current_zip_docx_active_secret_hit_count"),
        "current_zip_docx_scan_error_count": value(package, "current_zip_docx_scan_error_count"),
        "current_zip_xlsx_forbidden_hit_count": value(package, "current_zip_xlsx_forbidden_hit_count"),
        "current_zip_xlsx_active_secret_hit_count": value(package, "current_zip_xlsx_active_secret_hit_count"),
        "current_zip_xlsx_scan_error_count": value(package, "current_zip_xlsx_scan_error_count"),
        "current_zip_security_scan_error": value(package, "current_zip_security_scan_error"),
        "core_package_stale_entries": value(package, "stale_entry_count"),
        "core_package_active_secret_hits": value(package, "active_secret_hit_count"),
        "core_package_docx_forbidden_hits": value(package, "docx_forbidden_hit_count"),
        "core_package_docx_active_secret_hits": value(package, "docx_active_secret_hit_count"),
        "core_package_docx_scan_errors": value(package, "docx_scan_error_count"),
        "core_package_xlsx_forbidden_hits": value(package, "xlsx_forbidden_hit_count"),
        "core_package_xlsx_active_secret_hits": value(package, "xlsx_active_secret_hit_count"),
        "core_package_xlsx_scan_errors": value(package, "xlsx_scan_error_count"),
        "core_package_stale_status_marker_hits": value(package, "stale_status_marker_hit_count"),
        "core_package_duplicate_manifest_paths": value(package, "duplicate_manifest_path_count"),
        "core_package_duplicate_zip_entries": value(package, "duplicate_zip_entry_count"),
        "core_package_superseded_zip_packages": value(package, "superseded_zip_package_count"),
        "core_package_superseded_zip_package_names": value(package, "superseded_zip_packages", []),
        "current_superseded_package_artifacts": value(package, "current_superseded_package_artifact_count"),
        "current_superseded_package_artifact_names": value(package, "current_superseded_package_artifacts", []),
        "current_submission_package_unexpected_artifacts": value(package, "current_submission_package_unexpected_artifact_count"),
        "current_submission_package_unexpected_artifact_names": value(package, "current_submission_package_unexpected_artifacts", []),
        "current_submission_package_missing_required_artifacts": value(package, "current_submission_package_missing_required_artifact_count"),
        "current_submission_package_missing_required_artifact_names": value(package, "current_submission_package_missing_required_artifacts", []),
        "current_submission_package_allowed_non_file_artifacts": value(package, "current_submission_package_allowed_non_file_artifact_count"),
        "current_submission_package_allowed_non_file_artifact_names": value(package, "current_submission_package_allowed_non_file_artifacts", []),
        "email_handoff_status": value(handoff, "status"),
        "email_handoff_failure_count": value(handoff, "failure_count"),
        "email_handoff_message_id": value(handoff, "gmail_message_id"),
        "email_handoff_message_id_hex_like": is_gmail_hex_id(value(handoff, "gmail_message_id", "")),
        "email_handoff_thread_id": value(handoff, "gmail_thread_id"),
        "email_handoff_thread_id_hex_like": is_gmail_hex_id(value(handoff, "gmail_thread_id", "")),
        "email_handoff_display_url": value(handoff, "gmail_display_url"),
        "email_handoff_display_url_matches_message_id": gmail_display_url_matches_message_id(
            value(handoff, "gmail_display_url", ""),
            value(handoff, "gmail_message_id", ""),
        ),
        "email_handoff_email_ts": value(handoff, "gmail_email_ts"),
        "email_handoff_email_ts_iso_like": is_iso_like_timestamp(value(handoff, "gmail_email_ts")),
        "email_handoff_subject_matches_expected": value(handoff, "gmail_subject", "") == EXPECTED_GMAIL_SUBJECT,
        "package_sha_matches_handoff": value(package, "zip_sha256") == value(handoff, "latest_zip_sha256"),
        "package_size_matches_handoff": value(package, "zip_size_bytes") == value(handoff, "latest_zip_size_bytes"),
    }
    handoff_required = observed["require_email_handoff"] is True
    handoff_ok = (
        not handoff_required
        or (
            observed["email_handoff_status"] == "pass"
            and observed["email_handoff_failure_count"] == 0
            and observed["email_handoff_message_id_hex_like"] is True
            and observed["email_handoff_thread_id_hex_like"] is True
            and bool(observed["email_handoff_display_url"])
            and observed["email_handoff_display_url_matches_message_id"] is True
            and observed["email_handoff_email_ts_iso_like"] is True
            and observed["email_handoff_subject_matches_expected"] is True
            and observed["package_sha_matches_handoff"] is True
            and observed["package_size_matches_handoff"] is True
        )
    )
    ok = (
        observed["status"] in {"pass", "pass_with_handoff_refresh_needed"}
        and observed["failure_count"] == 0
        and observed["warning_count"] in {0, 1}
        and observed["require_email_handoff"] is False
        and isinstance(observed["expected_package_file_count"], int)
        and observed["core_package_status"] == "pass"
        and observed["current_zip_exists"] is True
        and observed["current_zip_sha_matches_audit"] is True
        and observed["current_zip_size_matches_audit"] is True
        and observed["manifest_exists"] is True
        and observed["manifest_expected_entry_count"]
        == observed["current_zip_entry_count"]
        == observed["expected_package_file_count"]
        and observed["manifest_zip_missing_entry_count"] == 0
        and observed["manifest_zip_extra_entry_count"] == 0
        and observed["manifest_zip_stale_entry_count"] == 0
        and observed["manifest_duplicate_entry_count"] == 0
        and observed["current_zip_duplicate_entry_count"] == 0
        and observed["manifest_source_missing_count"] == 0
        and observed["manifest_zip_scan_error"] == ""
        and observed["current_zip_forbidden_hit_count"] == 0
        and observed["current_zip_active_secret_hit_count"] == 0
        and observed["current_zip_docx_forbidden_hit_count"] == 0
        and observed["current_zip_docx_active_secret_hit_count"] == 0
        and observed["current_zip_docx_scan_error_count"] == 0
        and observed["current_zip_xlsx_forbidden_hit_count"] == 0
        and observed["current_zip_xlsx_active_secret_hit_count"] == 0
        and observed["current_zip_xlsx_scan_error_count"] == 0
        and observed["current_zip_security_scan_error"] == ""
        and observed["core_package_stale_entries"] == 0
        and observed["core_package_active_secret_hits"] == 0
        and observed["core_package_docx_forbidden_hits"] == 0
        and observed["core_package_docx_active_secret_hits"] == 0
        and observed["core_package_docx_scan_errors"] == 0
        and observed["core_package_xlsx_forbidden_hits"] == 0
        and observed["core_package_xlsx_active_secret_hits"] == 0
        and observed["core_package_xlsx_scan_errors"] == 0
        and observed["core_package_stale_status_marker_hits"] == 0
        and observed["core_package_duplicate_manifest_paths"] == 0
        and observed["core_package_duplicate_zip_entries"] == 0
        and observed["core_package_superseded_zip_packages"] == 0
        and observed["current_superseded_package_artifacts"] == 0
        and observed["current_submission_package_unexpected_artifacts"] == 0
        and observed["current_submission_package_missing_required_artifacts"] == 0
        and observed["current_submission_package_allowed_non_file_artifacts"] == 0
        and handoff_ok
    )
    return check("post_run_artifact_package_controls_passed", ok, observed)


def post_run_handoff_metadata_doc_sync_check(post_run: dict[str, Any]) -> dict[str, Any]:
    handoff = value(post_run, "email_handoff", {})
    required_terms = ["message ID", "thread ID", "display URL", "timestamp", "subject"]
    doc_paths = [
        "data_map.md",
        "paper4_execution_status.md",
        "REPRODUCIBILITY.md",
    ]
    doc_mentions: dict[str, dict[str, bool]] = {}
    for rel_path in doc_paths:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        doc_mentions[rel_path] = {term: term in text for term in required_terms}
    observed = {
        "email_handoff_message_id": value(handoff, "gmail_message_id"),
        "email_handoff_message_id_hex_like": is_gmail_hex_id(value(handoff, "gmail_message_id", "")),
        "email_handoff_thread_id": value(handoff, "gmail_thread_id"),
        "email_handoff_thread_id_hex_like": is_gmail_hex_id(value(handoff, "gmail_thread_id", "")),
        "email_handoff_display_url": value(handoff, "gmail_display_url"),
        "email_handoff_display_url_matches_message_id": gmail_display_url_matches_message_id(
            value(handoff, "gmail_display_url", ""),
            value(handoff, "gmail_message_id", ""),
        ),
        "email_handoff_email_ts": value(handoff, "gmail_email_ts"),
        "email_handoff_email_ts_iso_like": is_iso_like_timestamp(value(handoff, "gmail_email_ts")),
        "email_handoff_subject_matches_expected": value(handoff, "gmail_subject", "") == EXPECTED_GMAIL_SUBJECT,
        "doc_mentions": doc_mentions,
    }
    return check(
        "post_run_handoff_metadata_contract_documented",
        observed["email_handoff_message_id_hex_like"] is True
        and observed["email_handoff_thread_id_hex_like"] is True
        and bool(observed["email_handoff_display_url"])
        and observed["email_handoff_display_url_matches_message_id"] is True
        and observed["email_handoff_email_ts_iso_like"] is True
        and observed["email_handoff_subject_matches_expected"] is True
        and all(all(mentions.values()) for mentions in doc_mentions.values()),
        observed,
    )


def public_package_stale_status_marker_check() -> dict[str, Any]:
    package = read_json("submission_package/paper4_core_package_latest_audit.json")
    zip_rel = value(package, "zip_path", "submission_package/paper4_core_package_latest.zip")
    zip_path = ROOT / str(zip_rel)
    hits: list[dict[str, Any]] = []
    scan_error = ""
    entry_count = None
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            entry_count = len(names)
            for name in names:
                if Path(name).suffix.lower() not in TEXT_PACKAGE_SUFFIXES:
                    continue
                text = archive.read(name).decode("utf-8", errors="ignore")
                for marker in PUBLIC_PACKAGE_STALE_STATUS_MARKERS:
                    lines = [
                        idx
                        for idx, line in enumerate(text.splitlines(), start=1)
                        if marker in line
                    ]
                    if lines:
                        hits.append({"path": name, "marker": marker, "line_numbers": lines[:10]})
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        scan_error = f"{type(exc).__name__}: {exc}"
    observed = {
        "zip_path": zip_rel,
        "zip_exists": zip_path.exists(),
        "entry_count": entry_count,
        "scan_error": scan_error,
        "stale_status_marker_count": len(hits),
        "stale_status_marker_hits": hits[:20],
    }
    return check(
        "public_package_has_no_stale_status_markers",
        observed["zip_exists"] is True and scan_error == "" and not hits,
        observed,
    )


def traceability_external_blocker_sync_check() -> dict[str, Any]:
    traceability = read_json("results/research_plan_traceability.json")
    external_blockers = build_external_blockers()
    external_by_phase = {str(row.get("phase")): row for row in external_blockers}
    trace_items = value(traceability, "items", [])
    trace_blockers = value(traceability, "blockers", [])
    status_counts = value(traceability, "status_counts", {})
    item_by_key = {
        (row.get("phase"), row.get("requirement")): row
        for row in trace_items
        if isinstance(row, dict)
    }
    blocker_by_key = {
        (row.get("phase"), row.get("requirement")): row
        for row in trace_blockers
        if isinstance(row, dict)
    }
    expected_status_counts = Counter()
    expected_trace_keys: list[tuple[str, str]] = []
    failures: list[dict[str, Any]] = []

    for external_phase, spec in TRACEABILITY_EXTERNAL_BLOCKER_ROWS.items():
        key = (spec["phase"], spec["requirement"])
        external = external_by_phase.get(external_phase)
        trace_item = item_by_key.get(key)
        trace_blocker = blocker_by_key.get(key)
        if external is None:
            if trace_blocker is not None:
                failures.append(
                    {
                        "external_phase": external_phase,
                        "reason": "traceability_blocker_present_without_external_blocker",
                        "trace_status": trace_blocker.get("status"),
                    }
                )
            continue

        expected_trace_keys.append(key)
        expected_status_counts[spec["status"]] += 1
        if trace_item is None:
            failures.append({"external_phase": external_phase, "reason": "missing_traceability_item"})
            continue
        if trace_blocker is None:
            failures.append({"external_phase": external_phase, "reason": "missing_traceability_blocker"})

        for surface, row in [("items", trace_item), ("blockers", trace_blocker)]:
            if row is None:
                continue
            notes = str(row.get("notes", ""))
            action_required = str(row.get("action_required", ""))
            missing_note_markers = [
                marker
                for marker in [external.get("status"), *(external.get("blockers") or [])]
                if marker not in notes
            ]
            missing_action_markers = [
                marker for marker in spec["action_markers"] if marker not in action_required
            ]
            if row.get("status") != spec["status"]:
                failures.append(
                    {
                        "external_phase": external_phase,
                        "surface": surface,
                        "reason": "unexpected_traceability_status",
                        "status": row.get("status"),
                        "expected_status": spec["status"],
                    }
                )
            if missing_note_markers:
                failures.append(
                    {
                        "external_phase": external_phase,
                        "surface": surface,
                        "reason": "traceability_notes_missing_external_markers",
                        "missing_markers": missing_note_markers,
                    }
                )
            if missing_action_markers:
                failures.append(
                    {
                        "external_phase": external_phase,
                        "surface": surface,
                        "reason": "traceability_action_missing_required_markers",
                        "missing_markers": missing_action_markers,
                    }
                )

    expected_key_strings = [f"{phase} :: {requirement}" for phase, requirement in expected_trace_keys]
    actual_blocker_key_strings = [
        f"{row.get('phase')} :: {row.get('requirement')}"
        for row in trace_blockers
        if isinstance(row, dict)
    ]
    count_checks = {
        "traceability_blocker_count": value(traceability, "blocker_count"),
        "expected_blocker_count": len(external_blockers),
        "actual_blocker_rows": len(trace_blockers) if isinstance(trace_blockers, list) else None,
        "pending_human_labels": value(status_counts, "pending_human_labels", 0),
        "expected_pending_human_labels": expected_status_counts["pending_human_labels"],
        "phase_2_pending": value(status_counts, "phase_2_pending", 0),
        "expected_phase_2_pending": expected_status_counts["phase_2_pending"],
        "missing_artifacts": value(status_counts, "missing_artifacts", 0),
    }
    if count_checks["traceability_blocker_count"] != count_checks["expected_blocker_count"]:
        failures.append({"reason": "traceability_blocker_count_mismatch", **count_checks})
    if count_checks["actual_blocker_rows"] != count_checks["expected_blocker_count"]:
        failures.append({"reason": "traceability_blocker_rows_mismatch", **count_checks})
    if count_checks["pending_human_labels"] != count_checks["expected_pending_human_labels"]:
        failures.append({"reason": "pending_human_labels_count_mismatch", **count_checks})
    if count_checks["phase_2_pending"] != count_checks["expected_phase_2_pending"]:
        failures.append({"reason": "phase_2_pending_count_mismatch", **count_checks})
    if count_checks["missing_artifacts"] != 0:
        failures.append({"reason": "missing_artifacts_blocker_present", **count_checks})
    if sorted(actual_blocker_key_strings) != sorted(expected_key_strings):
        failures.append(
            {
                "reason": "traceability_blocker_keys_mismatch",
                "actual": sorted(actual_blocker_key_strings),
                "expected": sorted(expected_key_strings),
            }
        )

    return check(
        "research_plan_traceability_external_blockers_match_remaining_risk",
        not failures,
        {
            "failures": failures,
            "count_checks": count_checks,
            "expected_traceability_blockers": expected_key_strings,
            "actual_traceability_blockers": actual_blocker_key_strings,
        },
    )


def final_claim_boundary_output_check(final_claim: dict[str, Any]) -> dict[str, Any]:
    scope = value(final_claim, "scope_requirements", {})
    scanned_files = value(final_claim, "scanned_files", [])
    observed = {
        "status": value(final_claim, "status"),
        "finding_count": value(final_claim, "finding_count"),
        "blocker_count": value(final_claim, "blocker_count"),
        "review_count": value(final_claim, "review_count"),
        "forbidden_hit_count": value(final_claim, "forbidden_hit_count"),
        "scanned_file_count": len(scanned_files) if isinstance(scanned_files, list) else None,
        "scope_ok": value(scope, "ok"),
        "scope_actual_scanned_file_count": value(scope, "actual_scanned_file_count"),
        "scope_expected_scanned_file_count": value(scope, "expected_scanned_file_count"),
        "public_submission_manifest_exists": value(scope, "public_submission_manifest_exists"),
        "manifest_markdown_file_count": value(scope, "manifest_markdown_file_count"),
        "manifest_markdown_missing": value(scope, "manifest_markdown_missing", []),
    }
    ok = (
        observed["status"] == "passed"
        and observed["finding_count"] == 0
        and observed["blocker_count"] == 0
        and observed["review_count"] == 0
        and observed["forbidden_hit_count"] == 0
        and observed["scanned_file_count"] == 46
        and observed["scope_ok"] is True
        and observed["scope_actual_scanned_file_count"] == observed["scope_expected_scanned_file_count"] == 46
        and observed["public_submission_manifest_exists"] is True
        and isinstance(observed["manifest_markdown_file_count"], int)
        and observed["manifest_markdown_file_count"] > 0
        and observed["manifest_markdown_missing"] == []
    )
    return check("final_claim_boundary_output_passed", ok, observed)


def public_path_self_exclusion_check(public_paths: dict[str, Any], post_run: dict[str, Any]) -> dict[str, Any]:
    exclusions = value(public_paths, "self_referential_manifest_exclusions", [])
    package = value(post_run, "core_package", {})
    observed = {
        "public_path_status": value(public_paths, "status"),
        "self_referential_manifest_exclusion_count": value(
            public_paths, "self_referential_manifest_exclusion_count"
        ),
        "self_referential_manifest_exclusions": exclusions,
        "package_current_zip_forbidden_hit_count": value(package, "current_zip_forbidden_hit_count"),
        "package_current_zip_active_secret_hit_count": value(package, "current_zip_active_secret_hit_count"),
        "package_manifest_zip_stale_entry_count": value(package, "manifest_zip_stale_entry_count"),
    }
    ok = (
        observed["public_path_status"] == "pass"
        and observed["self_referential_manifest_exclusion_count"] == 1
        and exclusions == ["results/public_path_redaction_audit.md"]
        and observed["package_current_zip_forbidden_hit_count"] == 0
        and observed["package_current_zip_active_secret_hit_count"] == 0
        and observed["package_manifest_zip_stale_entry_count"] == 0
    )
    return check("public_path_self_exclusions_have_package_scan_backstop", ok, observed)


def core_post_run_audit_scripts_cwd_stable_check() -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for relative_path in CORE_POST_RUN_AUDIT_SCRIPTS:
        path = ROOT / relative_path
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        observed[relative_path] = {
            "exists": path.exists(),
            "has_project_root_bootstrap": "PROJECT_ROOT = Path(__file__).resolve().parents[2]" in text,
            "changes_to_workspace_root": "os.chdir(PROJECT_ROOT.parent)" in text,
            "keeps_relative_project_root": 'ROOT = Path("Paper 4")' in text or "import DEFAULT_MANUSCRIPT_PATHS, ROOT" in text,
        }
    ok = all(
        row["exists"]
        and row["has_project_root_bootstrap"]
        and row["changes_to_workspace_root"]
        and row["keeps_relative_project_root"]
        for row in observed.values()
    )
    return check("core_post_run_audit_scripts_are_cwd_stable", ok, observed)


def core_post_run_audit_scripts_external_execution_check() -> dict[str, Any]:
    external_cwd = PROJECT_ROOT.parent.parent
    local_roots = [str(PROJECT_ROOT.parent), str(external_cwd)]
    pre_smoke_package = public_manifest_zip_snapshot()

    def redacted_excerpt(text: str) -> str:
        redacted = text
        for local_root in local_roots:
            redacted = redacted.replace(local_root, "<local_workspace>")
        return redacted.strip()[:500]

    smoke_commands = [
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "post_run_artifact_status.py"),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "build_calculation_sheet.py"),
            "--check-only",
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "completion_audit.py"),
            "--check-only",
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "alignment_label_preflight.py"),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "edinet_execution_preflight.py"),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "sec_all8k_storage_preflight.py"),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "sec_validation_scope.py"),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "build_paper_figures.py"),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "bilingual_manuscript_audit.py"),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "journal_candidate_source_audit.py"),
            "--check-only",
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "manuscript_claim_audit.py"),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "manuscript_metric_audit.py"),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "market_residual_coverage_audit.py"),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "reference_consistency_audit.py"),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "research_plan_traceability.py"),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "word_export_audit.py"),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "audit" / "submission_package_audit.py"),
            "--check-only",
        ],
    ]
    smoke_script_paths = [
        Path(command[1]).relative_to(PROJECT_ROOT).as_posix()
        for command in smoke_commands
    ]
    smoke_script_order_matches_expected = smoke_script_paths == EXPECTED_EXTERNAL_CWD_SMOKE_SCRIPTS
    smoke_expected_script_missing = sorted(set(EXPECTED_EXTERNAL_CWD_SMOKE_SCRIPTS) - set(smoke_script_paths))
    smoke_unexpected_script_paths = sorted(set(smoke_script_paths) - set(EXPECTED_EXTERNAL_CWD_SMOKE_SCRIPTS))
    smoke_allowed_status_missing = sorted(set(smoke_script_paths) - set(SMOKE_ALLOWED_STATUSES))
    smoke_allowed_status_extra = sorted(set(SMOKE_ALLOWED_STATUSES) - set(smoke_script_paths))
    smoke_args_by_script = {
        Path(command[1]).relative_to(PROJECT_ROOT).as_posix(): command[2:]
        for command in smoke_commands
    }
    expected_args_script_missing = sorted(
        set(EXPECTED_EXTERNAL_CWD_SMOKE_SCRIPTS) - set(EXPECTED_EXTERNAL_CWD_SMOKE_ARGS)
    )
    expected_args_script_extra = sorted(
        set(EXPECTED_EXTERNAL_CWD_SMOKE_ARGS) - set(EXPECTED_EXTERNAL_CWD_SMOKE_SCRIPTS)
    )
    smoke_args_missing_scripts = sorted(set(EXPECTED_EXTERNAL_CWD_SMOKE_ARGS) - set(smoke_args_by_script))
    smoke_args_unexpected_scripts = sorted(set(smoke_args_by_script) - set(EXPECTED_EXTERNAL_CWD_SMOKE_ARGS))
    smoke_expected_args_mismatches = [
        {
            "script": script,
            "expected": expected_args,
            "actual": smoke_args_by_script.get(script),
        }
        for script, expected_args in EXPECTED_EXTERNAL_CWD_SMOKE_ARGS.items()
        if smoke_args_by_script.get(script) != expected_args
    ]
    observed: list[dict[str, Any]] = []
    for command in smoke_commands:
        script_rel = Path(command[1]).relative_to(PROJECT_ROOT).as_posix()
        timed_out = False
        execution_error = ""
        try:
            result = subprocess.run(
                command,
                cwd=external_cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                check=False,
            )
            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            execution_error = f"{type(exc).__name__}: timeout after {exc.timeout} seconds"
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            returncode = None
        except OSError as exc:
            execution_error = f"{type(exc).__name__}: {exc}"
            stdout = ""
            stderr = ""
            returncode = None
        combined_output = f"{stdout}\n{stderr}"
        parsed_output_json = False
        parsed_status = None
        parse_error = ""
        try:
            parsed_output = json.loads(stdout)
            parsed_output_json = isinstance(parsed_output, dict)
            if parsed_output_json:
                parsed_status = parsed_output.get("status") or parsed_output.get("overall_status")
        except json.JSONDecodeError as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
        status_allowed = parsed_status in SMOKE_ALLOWED_STATUSES.get(script_rel, set())
        stderr_empty = stderr.strip() == ""
        traceback_absent = "Traceback (most recent call last)" not in combined_output
        observed.append(
            {
                "script": script_rel,
                "args": command[2:],
                "cwd": "external_workspace_parent",
                "returncode": returncode,
                "timed_out": timed_out,
                "execution_error": execution_error,
                "emitted_status_json": parsed_output_json,
                "parsed_status": parsed_status,
                "status_allowed": status_allowed,
                "parse_error": parse_error,
                "stderr_empty": stderr_empty,
                "traceback_absent": traceback_absent,
                "output_excerpt": redacted_excerpt(combined_output),
            }
        )
    rebuild_command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "audit" / "build_core_submission_package.py"),
        "--stamp",
        "latest",
    ]
    rebuild_timed_out = False
    rebuild_execution_error = ""
    try:
        rebuild_result = subprocess.run(
            rebuild_command,
            cwd=external_cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        rebuild_stdout = rebuild_result.stdout
        rebuild_stderr = rebuild_result.stderr
        rebuild_returncode = rebuild_result.returncode
    except subprocess.TimeoutExpired as exc:
        rebuild_timed_out = True
        rebuild_execution_error = f"{type(exc).__name__}: timeout after {exc.timeout} seconds"
        rebuild_stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        rebuild_stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        rebuild_returncode = None
    except OSError as exc:
        rebuild_execution_error = f"{type(exc).__name__}: {exc}"
        rebuild_stdout = ""
        rebuild_stderr = ""
        rebuild_returncode = None
    rebuild_combined_output = f"{rebuild_stdout}\n{rebuild_stderr}"
    rebuild_parsed_json = False
    rebuild_status = None
    rebuild_parse_error = ""
    try:
        rebuild_output = json.loads(rebuild_stdout)
        rebuild_parsed_json = isinstance(rebuild_output, dict)
        if rebuild_parsed_json:
            rebuild_status = rebuild_output.get("status")
    except json.JSONDecodeError as exc:
        rebuild_parse_error = f"{type(exc).__name__}: {exc}"
    rebuild_observed = {
        "script": "src/audit/build_core_submission_package.py",
        "args": ["--stamp", "latest"],
        "cwd": "external_workspace_parent",
        "returncode": rebuild_returncode,
        "timed_out": rebuild_timed_out,
        "execution_error": rebuild_execution_error,
        "emitted_status_json": rebuild_parsed_json,
        "parsed_status": rebuild_status,
        "status_allowed": rebuild_status == "pass",
        "parse_error": rebuild_parse_error,
        "stderr_empty": rebuild_stderr.strip() == "",
        "traceback_absent": "Traceback (most recent call last)" not in rebuild_combined_output,
        "output_excerpt": redacted_excerpt(rebuild_combined_output),
    }
    rebuild_ok = (
        rebuild_observed["returncode"] == 0
        and rebuild_observed["timed_out"] is False
        and rebuild_observed["execution_error"] == ""
        and rebuild_observed["emitted_status_json"]
        and rebuild_observed["status_allowed"]
        and rebuild_observed["parse_error"] == ""
        and rebuild_observed["stderr_empty"]
        and rebuild_observed["traceback_absent"]
    )
    post_smoke_package = public_manifest_zip_snapshot()
    pre_smoke_package_fresh = package_snapshot_is_fresh(pre_smoke_package)
    post_smoke_package_fresh = package_snapshot_is_fresh(post_smoke_package)
    ok = (
        all(
            row["returncode"] == 0
            and row["timed_out"] is False
            and row["execution_error"] == ""
            and row["emitted_status_json"]
            and row["status_allowed"]
            and row["parse_error"] == ""
            and row["stderr_empty"]
            and row["traceback_absent"]
            for row in observed
        )
        and smoke_script_order_matches_expected is True
        and smoke_expected_script_missing == []
        and smoke_unexpected_script_paths == []
        and smoke_allowed_status_missing == []
        and smoke_allowed_status_extra == []
        and expected_args_script_missing == []
        and expected_args_script_extra == []
        and smoke_args_missing_scripts == []
        and smoke_args_unexpected_scripts == []
        and smoke_expected_args_mismatches == []
        and pre_smoke_package_fresh
        and rebuild_ok
        and post_smoke_package_fresh
    )
    return check(
        "core_post_run_audit_scripts_execute_from_external_cwd",
        ok,
        {
            "smoke_commands": observed,
            "expected_smoke_script_count": len(EXPECTED_EXTERNAL_CWD_SMOKE_SCRIPTS),
            "smoke_script_order_matches_expected": smoke_script_order_matches_expected,
            "smoke_expected_script_missing": smoke_expected_script_missing,
            "smoke_unexpected_script_paths": smoke_unexpected_script_paths,
            "smoke_allowed_status_missing": smoke_allowed_status_missing,
            "smoke_allowed_status_extra": smoke_allowed_status_extra,
            "expected_args_script_missing": expected_args_script_missing,
            "expected_args_script_extra": expected_args_script_extra,
            "smoke_args_missing_scripts": smoke_args_missing_scripts,
            "smoke_args_unexpected_scripts": smoke_args_unexpected_scripts,
            "smoke_expected_args_mismatches": smoke_expected_args_mismatches,
            "pre_smoke_package_fresh": pre_smoke_package_fresh,
            "pre_smoke_package": pre_smoke_package,
            "post_smoke_rebuild": rebuild_observed,
            "post_smoke_package_fresh": post_smoke_package_fresh,
            "post_smoke_package": post_smoke_package,
        },
    )


def build_internal_checks() -> list[dict[str, Any]]:
    metric = read_json("results/manuscript_metric_audit.json")
    claim = read_json("results/manuscript_claim_audit.json")
    bilingual = read_json("results/bilingual_manuscript_audit.json")
    figures = read_json("results/figure_generation_audit.json")
    journal = read_json("results/journal_candidate_source_audit.json")
    submission = read_json("results/submission_package_audit.json")
    public_manifest = read_json("results/public_submission_manifest_audit.json")
    calculation = read_json("results/calculation_sheet_audit.json")
    references = read_json("results/reference_consistency_audit.json")
    word = read_json("results/word_export_audit.json")
    public_paths = read_json("results/public_path_redaction_audit.json")
    email_handoff = read_json("submission_package/email_handoff_receipt_audit.json")
    post_run = read_json("results/post_run_artifact_status.json")
    final_claim = read_json("results/final_claim_boundary_audit.json")
    completion = read_json("paper4_completion_audit.json")
    traceability = read_json("results/research_plan_traceability.json")
    traceability_md = ROOT / "results" / "research_plan_traceability.md"
    traceability_text = traceability_md.read_text(encoding="utf-8") if traceability_md.exists() else ""
    expected_package_count = expected_public_package_file_count()

    word_documents = value(word, "documents", [])
    korean_word = next((row for row in word_documents if row.get("label") == "Korean final Word manuscript"), {})
    candidate_word = next((row for row in word_documents if row.get("label") == "Korean submission candidate journal table"), {})
    completion_counts = value(completion, "status_counts", {})
    residue_hits = journal_text_residue_hits()
    candidate_journals = value(journal, "candidate_journals", [])
    candidate_doi_evidence = value(journal, "candidate_doi_evidence", [])
    journal_policy_urls = value(journal, "policy_urls", [])
    submission_fee_policy_url_present = any(
        isinstance(row, dict)
        and row.get("ok") is True
        and "submission-fees" in str(row.get("url", ""))
        for row in journal_policy_urls
    )
    applied_soft_computing_doi_ok = any(
        isinstance(row, dict)
        and row.get("journal") == "Applied Soft Computing"
        and row.get("ok") is True
        and "10.1016/j.asoc.2026.115085" in (row.get("matched_expected_dois") or [])
        for row in candidate_doi_evidence
    )
    candidate_table_path = ROOT / "manuscript" / "submission_candidate_journals_ko.md"
    candidate_table_text = candidate_table_path.read_text(encoding="utf-8") if candidate_table_path.exists() else ""
    fee_risk_exclusion_documented = (
        "*Finance Research Letters*" in candidate_table_text
        and "제출수수료" in candidate_table_text
        and "후보에서 제외" in candidate_table_text
    )

    return [
        check("manuscript_metric_audit_passed", value(metric, "status") == "passed", value(metric, "status")),
        check(
            "manuscript_claim_audit_clean",
            value(claim, "status") == "passed" and value(claim, "blocker_count") == 0 and value(claim, "review_count") == 0,
            {"status": value(claim, "status"), "blockers": value(claim, "blocker_count"), "review": value(claim, "review_count")},
        ),
        check(
            "bilingual_manuscript_audit_passed",
            value(bilingual, "status") == "pass" and value(bilingual, "failure_count") == 0,
            {"status": value(bilingual, "status"), "failures": value(bilingual, "failure_count")},
        ),
        check(
            "figure_generation_audit_passed",
            value(figures, "status") == "pass" and value(figures, "figure_count", 0) >= 3,
            {"status": value(figures, "status"), "figures": value(figures, "figure_count")},
        ),
        check(
            "journal_candidate_source_audit_passed",
            value(journal, "status") == "pass"
            and value(journal, "candidate_journal_count") == 10
            and value(journal, "expected_candidate_journal_count") == 10
            and value(journal, "doi_count") == 12
            and value(journal, "policy_url_count") == 5
            and value(journal, "candidate_doi_evidence_ok") is True
            and value(journal, "candidate_policy_evidence_count") == 10
            and value(journal, "candidate_policy_evidence_ok") is True
            and value(journal, "failed_doi_count") == 0
            and value(journal, "failed_policy_url_count") == 0
            and "Applied Soft Computing" in candidate_journals
            and "Finance Research Letters" not in candidate_journals
            and applied_soft_computing_doi_ok
            and submission_fee_policy_url_present
            and fee_risk_exclusion_documented,
            {
                "status": value(journal, "status"),
                "candidate_journal_count": value(journal, "candidate_journal_count"),
                "expected_candidate_journal_count": value(journal, "expected_candidate_journal_count"),
                "doi_count": value(journal, "doi_count"),
                "policy_url_count": value(journal, "policy_url_count"),
                "candidate_doi_evidence_ok": value(journal, "candidate_doi_evidence_ok"),
                "candidate_policy_evidence_count": value(journal, "candidate_policy_evidence_count"),
                "candidate_policy_evidence_ok": value(journal, "candidate_policy_evidence_ok"),
                "failed_doi_count": value(journal, "failed_doi_count"),
                "failed_policy_url_count": value(journal, "failed_policy_url_count"),
                "includes_applied_soft_computing": "Applied Soft Computing" in candidate_journals,
                "excludes_finance_research_letters": "Finance Research Letters" not in candidate_journals,
                "applied_soft_computing_doi_ok": applied_soft_computing_doi_ok,
                "submission_fee_policy_url_present": submission_fee_policy_url_present,
                "fee_risk_exclusion_documented": fee_risk_exclusion_documented,
            },
        ),
        check("journal_candidate_text_residue_absent", not residue_hits, residue_hits),
        check(
            "submission_package_audit_passed",
            value(submission, "status") == "pass" and value(submission, "failure_count") == 0,
            {"status": value(submission, "status"), "documents": value(submission, "document_count"), "failures": value(submission, "failure_count")},
        ),
        check(
            "public_submission_manifest_audit_passed",
            value(public_manifest, "status") == "pass"
            and isinstance(expected_package_count, int)
            and value(public_manifest, "package_file_count") == expected_package_count
            and value(public_manifest, "missing_required_count") == 0
            and value(public_manifest, "denied_manifest_entry_count") == 0
            and value(public_manifest, "local_path_hit_count") == 0
            and value(public_manifest, "active_secret_hit_count") == 0
            and value(public_manifest, "public_path_dependency_status") == "pass"
            and value(public_manifest, "public_path_residual_hit_count") == 0
            and value(public_manifest, "public_path_active_secret_value_hit_count") == 0
            and value(public_manifest, "sensitive_workspace_guard_status") == "pass",
            {
                "status": value(public_manifest, "status"),
                "package_file_count": value(public_manifest, "package_file_count"),
                "expected_package_file_count": expected_package_count,
                "missing_required_count": value(public_manifest, "missing_required_count"),
                "denied_manifest_entry_count": value(public_manifest, "denied_manifest_entry_count"),
                "local_path_hit_count": value(public_manifest, "local_path_hit_count"),
                "active_secret_hit_count": value(public_manifest, "active_secret_hit_count"),
                "public_path_dependency_status": value(public_manifest, "public_path_dependency_status"),
                "sensitive_workspace_guard_status": value(public_manifest, "sensitive_workspace_guard_status"),
            },
        ),
        check(
            "calculation_sheet_audit_passed",
            value(calculation, "status") == "pass"
            and len(value(calculation, "failures", [])) == 0
            and len(value(calculation, "output_integrity_failures", [])) == 0,
            {
                "status": value(calculation, "status"),
                "failures": len(value(calculation, "failures", [])),
                "output_integrity_failures": len(value(calculation, "output_integrity_failures", [])),
            },
        ),
        check(
            "calculation_sheet_traceability_complete",
            value(calculation, "source_rows") == 18
            and value(calculation, "source_data_rows") == 10
            and value(calculation, "source_code_rows") == 8
            and value(calculation, "source_existing_rows") == 18
            and value(calculation, "source_hash_rows") == 18
            and value(calculation, "source_missing_rows") == 0
            and value(calculation, "calculation_traceability_ok") is True
            and value(calculation, "calculation_rows_with_source_data") == value(calculation, "calculation_rows") == 47
            and value(calculation, "calculation_rows_with_source_code") == value(calculation, "calculation_rows") == 47
            and value(calculation, "calculation_rows_with_formula") == value(calculation, "calculation_rows") == 47
            and value(calculation, "model_rows_with_source_data") == value(calculation, "model_rows") == 6
            and value(calculation, "model_rows_with_source_code") == value(calculation, "model_rows") == 6
            and value(calculation, "model_rows_with_formula") == value(calculation, "model_rows") == 6
            and value(calculation, "verification_rows_with_source_data") == value(calculation, "verification_checks") == 14
            and value(calculation, "verification_rows_with_source_code") == value(calculation, "verification_checks") == 14
            and value(calculation, "verification_rows_with_formula") == value(calculation, "verification_checks") == 14
            and value(calculation, "output_integrity_check_count") == 12
            and value(calculation, "output_integrity_failure_count") == 0,
            {
                "source_rows": value(calculation, "source_rows"),
                "source_data_rows": value(calculation, "source_data_rows"),
                "source_code_rows": value(calculation, "source_code_rows"),
                "source_existing_rows": value(calculation, "source_existing_rows"),
                "source_hash_rows": value(calculation, "source_hash_rows"),
                "source_missing_rows": value(calculation, "source_missing_rows"),
                "calculation_traceability_ok": value(calculation, "calculation_traceability_ok"),
                "calculation_rows": value(calculation, "calculation_rows"),
                "model_rows": value(calculation, "model_rows"),
                "verification_checks": value(calculation, "verification_checks"),
                "output_integrity_check_count": value(calculation, "output_integrity_check_count"),
                "output_integrity_failure_count": value(calculation, "output_integrity_failure_count"),
            },
        ),
        check(
            "reference_consistency_audit_passed",
            value(references, "status") == "pass" and len(value(references, "failures", [])) == 0,
            {"status": value(references, "status"), "failures": len(value(references, "failures", []))},
        ),
        check(
            "word_export_audit_passed",
            value(word, "status") == "pass"
            and value(word, "failure_count") == 0
            and value(word, "claim_blocker_count") == 0
            and value(word, "claim_review_count") == 0
            and korean_word.get("docx_media_count", 0) >= korean_word.get("min_media_count", 0) >= 3
            and candidate_word.get("ranked_table_matched_rows") == candidate_word.get("ranked_table_expected_rows") == 10,
            {
                "status": value(word, "status"),
                "failures": value(word, "failure_count"),
                "claim_blockers": value(word, "claim_blocker_count"),
                "claim_review": value(word, "claim_review_count"),
                "korean_media": korean_word.get("docx_media_count"),
                "candidate_rows": candidate_word.get("ranked_table_matched_rows"),
            },
        ),
        check(
            "public_path_redaction_audit_passed",
            value(public_paths, "status") == "pass"
            and value(public_paths, "residual_hit_count") == 0
            and value(public_paths, "active_secret_value_hit_count") == 0
            and value(value(public_paths, "sensitive_workspace_guard", {}), "status") == "pass",
            {
                "status": value(public_paths, "status"),
                "residual_hit_count": value(public_paths, "residual_hit_count"),
                "active_secret_value_hit_count": value(public_paths, "active_secret_value_hit_count"),
                "sensitive_workspace_guard": value(value(public_paths, "sensitive_workspace_guard", {}), "status"),
            },
        ),
        check(
            "public_path_active_secret_scans_submission_package",
            "submission_package" in value(public_paths, "active_secret_scan_roots", [])
            and value(public_paths, "active_secret_candidate_file_count", 0) > 0,
            {
                "active_secret_scan_roots": value(public_paths, "active_secret_scan_roots", []),
                "active_secret_candidate_file_count": value(public_paths, "active_secret_candidate_file_count"),
            },
        ),
        check(
            "completion_audit_has_no_internal_incomplete_items",
            value(completion, "overall_status") == "v1_ready_no_open_submission_risks"
            and completion_counts.get("incomplete", 0) == 0,
            {"overall_status": value(completion, "overall_status"), "status_counts": completion_counts},
        ),
        check(
            "research_plan_traceability_is_bounded",
            value(traceability, "status") == "v1_traceable_for_submission_scope"
            and value(traceability, "trace_item_count") == 15
            and value(traceability, "blocker_count") == 0,
            {"status": value(traceability, "status"), "trace_item_count": value(traceability, "trace_item_count"), "blocker_count": value(traceability, "blocker_count")},
        ),
        check(
            "research_plan_traceability_avoids_in_pipeline_run_counts",
            "Latest run:" not in traceability_text and re.search(r"steps=\d+;\s*ok=\d+", traceability_text) is None,
            {"contains_latest_run": "Latest run:" in traceability_text, "contains_steps_ok_counts": bool(re.search(r"steps=\d+;\s*ok=\d+", traceability_text))},
        ),
        pipeline_doc_sync_check(),
        email_handoff_audit_contract_check(email_handoff),
        forbidden_surface_consistency_check(),
        post_run_artifact_status_check(post_run),
        post_run_handoff_metadata_doc_sync_check(post_run),
        public_package_stale_status_marker_check(),
        final_claim_boundary_output_check(final_claim),
        public_path_self_exclusion_check(public_paths, post_run),
        core_post_run_audit_scripts_cwd_stable_check(),
        core_post_run_audit_scripts_external_execution_check(),
        traceability_external_blocker_sync_check(),
        final_claim_boundary_doc_sync_check(),
    ]


def build_external_blockers() -> list[dict[str, Any]]:
    return []


def build_deferred_v2_scope() -> list[dict[str, Any]]:
    alignment = read_json("results/alignment_label_preflight.json")
    edinet = read_json("results/edinet_execution_preflight.json")
    sec = read_json("results/sec_all8k_storage_preflight.json")
    deferred: list[dict[str, Any]] = []

    if value(alignment, "final_metrics_ready") is not True:
        deferred.append(
            {
                "phase": "RQ1 alignment",
                "status": value(alignment, "status"),
                "blockers": value(alignment, "blockers", []),
                "guardrail": EXPECTED_EXTERNAL_PHASES["RQ1 alignment"],
            }
        )
    if value(edinet, "empirical_collection_ready") is not True:
        deferred.append(
            {
                "phase": "EDINET/Japanese disclosure data",
                "status": value(edinet, "status"),
                "blockers": value(edinet, "blockers", []),
                "guardrail": EXPECTED_EXTERNAL_PHASES["EDINET/Japanese disclosure data"],
            }
        )
    if value(sec, "full_scope_download_ready") is not True:
        deferred.append(
            {
                "phase": "SEC all-8-K body-text scope",
                "status": value(sec, "status"),
                "blockers": value(sec, "blockers", []),
                "guardrail": EXPECTED_EXTERNAL_PHASES["SEC all-8-K body-text scope"],
            }
        )
    return deferred


def expected_alignment_blocker_state(alignment: dict[str, Any]) -> dict[str, Any]:
    main_seed = value(alignment, "main_seed", {})
    single = value(alignment, "single_expert_sheet", {})
    workbook = value(alignment, "two_annotator_workbook", {})
    supplemental = value(alignment, "supplemental_sec_seed", {})

    candidate_rows = as_int(value(alignment, "candidate_rows"))
    evaluated_labeled_rows = as_int(value(alignment, "main_evaluation_labeled_rows"))
    evaluated_status = value(alignment, "main_evaluation_status")
    invalid_label_rows = as_int(value(alignment, "invalid_label_row_count"))
    final_metrics_ready = (
        candidate_rows > 0
        and evaluated_labeled_rows == candidate_rows
        and evaluated_status == "evaluated_all_labels"
        and invalid_label_rows == 0
    )

    blockers: list[str] = []
    if as_int(value(main_seed, "rows")) == 0:
        blockers.append("missing_main_alignment_seed")
    if evaluated_status is None:
        blockers.append("missing_main_alignment_evaluation")
    if as_int(value(single, "rows")) == 0:
        blockers.append("missing_single_expert_label_sheet")
    if as_int(value(workbook, "rows")) == 0:
        blockers.append("missing_two_annotator_workbook")
    if as_int(value(supplemental, "rows")) == 0:
        blockers.append("missing_supplemental_sec_alignment_seed")
    if invalid_label_rows:
        blockers.append("invalid_label_values_present")
    if evaluated_labeled_rows == 0:
        blockers.append("no_gold_labels")
    elif evaluated_labeled_rows < candidate_rows:
        blockers.append("incomplete_gold_labels")
    if as_int(value(single, "human_labeled_rows")) == 0:
        blockers.append("single_expert_labels_missing")
    if as_int(value(workbook, "annotator1_labeled_rows")) == 0 or as_int(value(workbook, "annotator2_labeled_rows")) == 0:
        blockers.append("two_annotator_labels_missing")
    if as_int(value(workbook, "needs_adjudication_rows")):
        blockers.append("adjudication_incomplete")
    supplemental_labeled_rows = as_int(
        value(supplemental, "evaluation_labeled_rows", value(supplemental, "gold_labeled_rows"))
    )
    if supplemental_labeled_rows == 0:
        blockers.append("supplemental_sec_labels_missing")
    if evaluated_status == "needs_human_labels":
        blockers.append("main_evaluation_not_ready")

    if final_metrics_ready:
        status = "ready_for_rq1_metrics"
    elif invalid_label_rows:
        status = "blocked_invalid_labels"
    elif evaluated_labeled_rows > 0:
        status = "partially_labeled_not_final"
    else:
        status = "blocked_missing_human_labels"

    return {
        "phase": "RQ1 alignment",
        "ready": final_metrics_ready,
        "expected_status": status,
        "expected_blockers": sorted(set(blockers)),
        "expected_ready_field": final_metrics_ready,
        "actual_ready_field": value(alignment, "final_rq1_metrics_ready"),
        "evidence": {
            "candidate_rows": candidate_rows,
            "main_evaluation_status": evaluated_status,
            "main_evaluation_labeled_rows": evaluated_labeled_rows,
            "invalid_label_row_count": invalid_label_rows,
            "single_expert_human_labeled_rows": as_int(value(single, "human_labeled_rows")),
            "annotator1_labeled_rows": as_int(value(workbook, "annotator1_labeled_rows")),
            "annotator2_labeled_rows": as_int(value(workbook, "annotator2_labeled_rows")),
            "supplemental_labeled_rows": supplemental_labeled_rows,
        },
    }


def expected_edinet_blocker_state(edinet: dict[str, Any]) -> dict[str, Any]:
    env_api_key_present = value(edinet, "env_api_key_present") is True
    document_rows = as_int(value(edinet, "document_rows"))
    current_report_rows = as_int(value(edinet, "current_report_candidate_rows"))
    api_window_infeasible_days = as_int(value(edinet, "api_window_infeasible_days"))
    external_reusable_candidates = as_int(value(edinet, "external_reusable_candidate_rows"))
    empirical_ready = (
        env_api_key_present
        and document_rows > 0
        and current_report_rows > 0
        and api_window_infeasible_days == 0
    )

    blockers: list[str] = []
    if not env_api_key_present:
        blockers.append("missing_env_edinet_api_key")
    if api_window_infeasible_days:
        blockers.append("archive_or_prior_collection_required")
    if document_rows == 0:
        blockers.append("no_edinet_document_rows_collected")
    if current_report_rows == 0:
        blockers.append("no_current_report_candidates_collected")
    if external_reusable_candidates == 0:
        blockers.append("no_external_reusable_edinet_dataset_found")

    if empirical_ready:
        status = "ready_for_edinet_normalization"
    elif not env_api_key_present and api_window_infeasible_days:
        status = "blocked_key_and_archive_gap"
    elif not env_api_key_present:
        status = "blocked_missing_api_key"
    elif api_window_infeasible_days:
        status = "blocked_archive_gap"
    else:
        status = "blocked_no_empirical_rows"

    return {
        "phase": "EDINET/Japanese disclosure data",
        "ready": empirical_ready,
        "expected_status": status,
        "expected_blockers": sorted(set(blockers)),
        "expected_ready_field": empirical_ready,
        "actual_ready_field": value(edinet, "empirical_collection_ready"),
        "evidence": {
            "env_api_key_present": env_api_key_present,
            "document_rows": document_rows,
            "current_report_candidate_rows": current_report_rows,
            "api_window_infeasible_days": api_window_infeasible_days,
            "external_reusable_candidate_rows": external_reusable_candidates,
        },
    }


def expected_sec_blocker_state(sec: dict[str, Any]) -> dict[str, Any]:
    missing_files = as_int(value(sec, "missing_text_files"))
    storage_ready = value(sec, "storage_ready_for_full_scope") is True
    env_user_agent_present = value(sec, "env_user_agent_present") is True
    scope_plan_ready = value(sec, "scope_plan_status") == "plan_created"
    batch_status_ready = value(sec, "batch_status") in {
        "status_only",
        "download_not_requested",
        "downloaded_partial",
        "downloaded_complete",
    }

    blockers: list[str] = []
    if not scope_plan_ready:
        blockers.append("scope_plan_not_ready")
    if not batch_status_ready:
        blockers.append("batch_status_not_ready")
    if missing_files and not storage_ready:
        blockers.append("insufficient_storage_for_full_scope")
    if missing_files and not env_user_agent_present:
        blockers.append("missing_env_sec_user_agent")

    full_scope_ready = not blockers and missing_files > 0
    if missing_files == 0 and scope_plan_ready and batch_status_ready:
        status = "full_scope_cache_complete"
    elif full_scope_ready:
        status = "ready_for_controlled_full_scope_batches"
    elif "insufficient_storage_for_full_scope" in blockers and "missing_env_sec_user_agent" in blockers:
        status = "blocked_storage_and_user_agent"
    elif "insufficient_storage_for_full_scope" in blockers:
        status = "blocked_insufficient_storage"
    elif "missing_env_sec_user_agent" in blockers:
        status = "blocked_missing_env_user_agent"
    else:
        status = "blocked_missing_preflight_inputs"

    return {
        "phase": "SEC all-8-K body-text scope",
        "ready": full_scope_ready or status == "full_scope_cache_complete",
        "expected_status": status,
        "expected_blockers": sorted(set(blockers)),
        "expected_ready_field": full_scope_ready,
        "actual_ready_field": value(sec, "full_scope_download_ready"),
        "evidence": {
            "missing_text_files": missing_files,
            "storage_ready_for_full_scope": storage_ready,
            "env_user_agent_present": env_user_agent_present,
            "scope_plan_status": value(sec, "scope_plan_status"),
            "batch_status": value(sec, "batch_status"),
        },
    }


def external_blocker_shape_check(external_blockers: list[dict[str, Any]]) -> dict[str, Any]:
    alignment = read_json("results/alignment_label_preflight.json")
    edinet = read_json("results/edinet_execution_preflight.json")
    sec = read_json("results/sec_all8k_storage_preflight.json")
    expected_states = [
        expected_alignment_blocker_state(alignment),
        expected_edinet_blocker_state(edinet),
        expected_sec_blocker_state(sec),
    ]
    expected_phase_states = {row["phase"]: row for row in expected_states if row["ready"] is not True}
    actual_phase_rows = {str(row.get("phase")): row for row in external_blockers}
    failures: list[dict[str, Any]] = []
    missing_phases = sorted(set(expected_phase_states) - set(actual_phase_rows))
    unexpected_phases = sorted(set(actual_phase_rows) - set(expected_phase_states))
    if missing_phases:
        failures.append({"reason": "missing_blocked_phases_from_evidence", "phases": missing_phases})
    if unexpected_phases:
        failures.append({"reason": "unexpected_blocked_phases_from_evidence", "phases": unexpected_phases})
    for row in external_blockers:
        phase = row.get("phase")
        expected = EXPECTED_EXTERNAL_BLOCKERS.get(str(phase))
        blockers = set(row.get("blockers") or [])
        evidence_expected = expected_phase_states.get(str(phase))
        if expected is None:
            failures.append({"phase": phase, "reason": "unexpected_phase"})
            continue
        expected_blockers = set(expected["blockers"])
        unexpected_blockers = sorted(blockers - expected_blockers)
        if row.get("status") != expected["status"]:
            failures.append(
                {
                    "phase": phase,
                    "reason": "unexpected_status",
                    "status": row.get("status"),
                    "expected_status": expected["status"],
                }
            )
        if unexpected_blockers:
            failures.append(
                {
                    "phase": phase,
                    "reason": "unexpected_blockers",
                    "unexpected_blockers": unexpected_blockers,
                }
            )
        if evidence_expected is not None:
            evidence_blockers = set(evidence_expected["expected_blockers"])
            missing_evidence_blockers = sorted(evidence_blockers - blockers)
            extra_evidence_blockers = sorted(blockers - evidence_blockers)
            if evidence_expected.get("actual_ready_field") != evidence_expected.get("expected_ready_field"):
                failures.append(
                    {
                        "phase": phase,
                        "reason": "ready_field_does_not_match_evidence",
                        "actual_ready_field": evidence_expected.get("actual_ready_field"),
                        "expected_ready_field": evidence_expected.get("expected_ready_field"),
                        "evidence": evidence_expected["evidence"],
                    }
                )
            if row.get("status") != evidence_expected["expected_status"]:
                failures.append(
                    {
                        "phase": phase,
                        "reason": "status_does_not_match_evidence",
                        "status": row.get("status"),
                        "expected_status": evidence_expected["expected_status"],
                        "evidence": evidence_expected["evidence"],
                    }
                )
            if missing_evidence_blockers or extra_evidence_blockers:
                failures.append(
                    {
                        "phase": phase,
                        "reason": "blockers_do_not_match_evidence",
                        "missing_evidence_blockers": missing_evidence_blockers,
                        "extra_evidence_blockers": extra_evidence_blockers,
                        "evidence": evidence_expected["evidence"],
                    }
                )
    return check(
        "deferred_v2_scope_shape_expected",
        not failures,
        {
            "failures": failures,
            "deferred_v2_phases": [row.get("phase") for row in external_blockers],
            "expected_blocked_phases_from_evidence": list(expected_phase_states),
            "expected_states_from_evidence": expected_states,
        },
    )


def build_audit() -> dict[str, Any]:
    checks = build_internal_checks()
    external_blockers = build_external_blockers()
    deferred_v2_scope = build_deferred_v2_scope()
    checks.append(external_blocker_shape_check(deferred_v2_scope))
    checks.append(remaining_risk_data_map_sync_check(len(checks) + 1, len(external_blockers)))
    failures = [row for row in checks if not row["ok"]]
    status = "pass"
    if failures:
        status = "fail_internal_risk"
    return {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "internal_check_count": len(checks),
        "internal_failure_count": len(failures),
        "known_external_blocker_count": len(external_blockers),
        "known_external_blocker_detail_count": sum(len(row.get("blockers") or []) for row in external_blockers),
        "deferred_v2_scope_count": len(deferred_v2_scope),
        "deferred_v2_scope_detail_count": sum(len(row.get("blockers") or []) for row in deferred_v2_scope),
        "internal_checks": checks,
        "internal_failures": failures,
        "known_external_blockers": external_blockers,
        "deferred_v2_scope": deferred_v2_scope,
        "guardrail": (
            "This audit separates APFM v1 submission risks from deferred v2 research scope. "
            "Human-label metrics, EDINET empirical collection, and full SEC all-8-K filing-body validation are not converted into manuscript claims or active submission blockers unless the required labels, keys, rows, storage, or user-agent evidence exists. "
            "Final email handoff freshness is verified after package generation by post_run_artifact_status and email_handoff_receipt_audit, outside this package-included internal-readiness artifact, to avoid circular ZIP/handoff locks."
        ),
    }


def write_markdown(audit: dict[str, Any]) -> None:
    lines = [
        "# Remaining Risk Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Internal checks | {audit['internal_check_count']} |",
        f"| Internal failures | {audit['internal_failure_count']} |",
        f"| Active external blocker phases | {audit['known_external_blocker_count']} |",
        f"| Active external blocker details | {audit['known_external_blocker_detail_count']} |",
        f"| Deferred v2 scope phases | {audit['deferred_v2_scope_count']} |",
        f"| Deferred v2 scope details | {audit['deferred_v2_scope_detail_count']} |",
        "",
        "## Internal Gate Checks",
        "",
        "| Check | OK | Observed |",
        "|---|---:|---|",
    ]
    for row in audit["internal_checks"]:
        observed = json.dumps(row["observed"], ensure_ascii=False, sort_keys=True)
        lines.append(f"| `{row['name']}` | {row['ok']} | `{observed}` |")
    lines.extend(["", "## Active External Blockers", "", "| Phase | Status | Blockers | Guardrail |", "|---|---|---|---|"])
    if audit["known_external_blockers"]:
        for row in audit["known_external_blockers"]:
            blockers = ", ".join(f"`{item}`" for item in row.get("blockers") or [])
            lines.append(f"| {row['phase']} | `{row['status']}` | {blockers or '-'} | {row['guardrail']} |")
    else:
        lines.append("| None | `ready` | - | No active external blocker phases remain in APFM v1 submission scope. |")
    lines.extend(["", "## Deferred V2 Scope", "", "| Phase | Evidence status | Deferred inputs | Guardrail |", "|---|---|---|---|"])
    if audit["deferred_v2_scope"]:
        for row in audit["deferred_v2_scope"]:
            blockers = ", ".join(f"`{item}`" for item in row.get("blockers") or [])
            lines.append(f"| {row['phase']} | `{row['status']}` | {blockers or '-'} | {row['guardrail']} |")
    else:
        lines.append("| None | `ready` | - | No deferred v2 scope phases remain. |")
    lines.extend(["", "## Guardrail", "", audit["guardrail"]])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "internal_check_count": audit["internal_check_count"],
                "internal_failure_count": audit["internal_failure_count"],
                "known_external_blocker_count": audit["known_external_blocker_count"],
                "known_external_blocker_detail_count": audit["known_external_blocker_detail_count"],
                "deferred_v2_scope_count": audit["deferred_v2_scope_count"],
                "deferred_v2_scope_detail_count": audit["deferred_v2_scope_detail_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if audit["internal_failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
