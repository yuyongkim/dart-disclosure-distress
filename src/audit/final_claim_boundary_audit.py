from __future__ import annotations

import csv
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)

from audit_forbidden_tokens import FORBIDDEN_TOKENS
from manuscript_claim_audit import DEFAULT_MANUSCRIPT_PATHS, ROOT, build_audit, write_markdown


EXPECTED_SCANNED_FILE_COUNT = 46
PUBLIC_SUBMISSION_MANIFEST = ROOT / "data" / "audit" / "public_submission_manifest.csv"
REQUIRED_SCANNED_FILES = [
    ROOT / "paper4_research_plan.md",
    ROOT / "results" / "public_submission_manifest_audit.md",
    ROOT / "results" / "remaining_risk_audit.md",
    ROOT / "results" / "post_run_artifact_status.md",
    ROOT / "submission_package" / "email_handoff_receipt_audit.md",
    ROOT / "submission_package" / "LATEST_PACKAGE.md",
]

POST_CLAIM_GENERATED_PATHS = [
    ROOT / "results" / "manuscript_claim_audit.md",
    ROOT / "results" / "bilingual_manuscript_audit.md",
    ROOT / "results" / "word_export_audit.md",
    ROOT / "results" / "submission_package_audit.md",
    ROOT / "results" / "research_plan_traceability.md",
    ROOT / "paper4_completion_audit.md",
    ROOT / "results" / "remaining_risk_audit.md",
    ROOT / "results" / "calculation_sheet.md",
    ROOT / "results" / "reference_consistency_audit.md",
    ROOT / "results" / "manuscript_metric_audit.md",
    ROOT / "results" / "market_extension_summary.md",
    ROOT / "results" / "figure_generation_audit.md",
    ROOT / "results" / "submission_figure_generation_audit.md",
    ROOT / "results" / "journal_candidate_source_audit.md",
    ROOT / "results" / "public_path_redaction_audit.md",
    ROOT / "results" / "public_submission_manifest_audit.md",
    ROOT / "results" / "reproducibility_run_report_audit.md",
    ROOT / "results" / "post_run_artifact_status.md",
    ROOT / "results" / "alignment_label_preflight.md",
    ROOT / "results" / "edinet_execution_preflight.md",
    ROOT / "results" / "sec_all8k_storage_preflight.md",
    ROOT / "submission_package" / "email_handoff_receipt_audit.md",
    ROOT / "submission_package" / "LATEST_PACKAGE.md",
]

TEXT_INTEGRITY_EXTRA_PATHS = [
    ROOT / "paper4_research_plan.md",
    ROOT / "audit_report.md",
    ROOT / "results" / "alignment_label_discovery.md",
    ROOT / "results" / "alignment_label_preflight.md",
    ROOT / "results" / "edinet_local_discovery.md",
    ROOT / "results" / "jp_report_parser_prototype_summary.md",
    ROOT / "results" / "sec_desktop_population_collection_status.md",
    ROOT / "results" / "sec_desktop_population_smoke_collection_status.md",
    ROOT / "results" / "sec_submissions_archive_status.md",
    ROOT / "results" / "sec_all8k_text_scope_plan.md",
    ROOT / "results" / "sec_all8k_text_sample_design.md",
    ROOT / "results" / "sec_all8k_text_batch_collection_status.md",
    ROOT / "results" / "edinet_execution_preflight.md",
    ROOT / "results" / "sec_all8k_storage_preflight.md",
    ROOT / "manuscript" / "submission_candidate_journals_ko.md",
]


def manifest_markdown_paths() -> list[Path]:
    if not PUBLIC_SUBMISSION_MANIFEST.exists():
        return []
    paths: list[Path] = []
    with PUBLIC_SUBMISSION_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            manifest_path = str(row.get("path", "")).replace("\\", "/").lstrip("/")
            if manifest_path.endswith(".md"):
                paths.append(ROOT / manifest_path)
    return paths


def unique_existing(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def line_numbers(text: str, token: str) -> list[int]:
    return [idx for idx, line in enumerate(text.splitlines(), start=1) if token in line]


def forbidden_hits(paths: list[Path]) -> list[dict]:
    hits: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            lines = line_numbers(text, token)
            if lines:
                hits.append({"file": str(path), "token": token, "line_numbers": lines})
    return hits


def scope_requirements(paths: list[Path]) -> dict:
    scanned_files = {str(path) for path in paths if path.exists()}
    required_presence = {
        str(path): str(path) in scanned_files
        for path in REQUIRED_SCANNED_FILES
    }
    public_submission_manifest_exists = PUBLIC_SUBMISSION_MANIFEST.exists()
    manifest_markdown_presence = {
        str(path): str(path) in scanned_files
        for path in manifest_markdown_paths()
    }
    return {
        "expected_scanned_file_count": EXPECTED_SCANNED_FILE_COUNT,
        "actual_scanned_file_count": len(scanned_files),
        "public_submission_manifest_path": str(PUBLIC_SUBMISSION_MANIFEST),
        "public_submission_manifest_exists": public_submission_manifest_exists,
        "required_scanned_files_present": required_presence,
        "manifest_markdown_files_present": manifest_markdown_presence,
        "manifest_markdown_file_count": len(manifest_markdown_presence),
        "manifest_markdown_missing": [
            path for path, present in manifest_markdown_presence.items() if not present
        ],
        "ok": (
            public_submission_manifest_exists
            and
            len(scanned_files) == EXPECTED_SCANNED_FILE_COUNT
            and all(required_presence.values())
            and len(manifest_markdown_presence) > 0
            and all(manifest_markdown_presence.values())
        ),
    }


def main() -> None:
    paths = unique_existing(
        DEFAULT_MANUSCRIPT_PATHS
        + [ROOT / "paper4_research_plan.md"]
        + POST_CLAIM_GENERATED_PATHS
        + manifest_markdown_paths()
    )
    audit = build_audit(paths)
    scope_check = scope_requirements(paths)
    text_integrity_paths = unique_existing(paths + TEXT_INTEGRITY_EXTRA_PATHS)
    text_hits = forbidden_hits(text_integrity_paths)
    audit["scope_requirements"] = scope_check
    audit["forbidden_hits"] = text_hits
    audit["forbidden_hit_count"] = len(text_hits)
    audit["forbidden_token_count"] = len(FORBIDDEN_TOKENS)
    if text_hits or not scope_check["ok"]:
        audit["status"] = "blocked_claims_or_text_integrity_failures"
    audit["guardrail"] = (
        "Final claim-boundary audit runs after completion_audit so post-claim generated Markdown files "
        "cannot introduce unreviewed alignment, EDINET, SEC, exact-score overclaims, stale values, local absolute paths, or internal workspace labels."
    )
    audit["audit_scope"] = "final_post_completion_claim_boundary"
    audit["post_claim_generated_files"] = [str(path) for path in POST_CLAIM_GENERATED_PATHS if path.exists()]
    audit["text_integrity_files"] = [str(path) for path in text_integrity_paths if path.exists()]

    out_json = ROOT / "results" / "final_claim_boundary_audit.json"
    out_md = ROOT / "results" / "final_claim_boundary_audit.md"
    out_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(
        audit,
        out_md,
        title="Final Claim-Boundary Audit",
        clean_subject="final claim-boundary Markdown files",
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "scanned_files": len(audit["scanned_files"]),
                "finding_count": audit["finding_count"],
                "blocker_count": audit["blocker_count"],
                "review_count": audit["review_count"],
                "forbidden_hit_count": audit["forbidden_hit_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
