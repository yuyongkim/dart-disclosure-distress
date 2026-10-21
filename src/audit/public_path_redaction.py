from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
WORKSPACE_ROOT = ROOT.parent
GITIGNORE = WORKSPACE_ROOT / ".gitignore"
ENV_EXAMPLE = WORKSPACE_ROOT / ".env.example"
PUBLIC_SUBMISSION_MANIFEST = ROOT / "data" / "audit" / "public_submission_manifest.csv"

CSV_PATH_COLUMNS = {
    "path",
    "relpath",
    "source_path",
    "parent_path",
    "local_path",
    "raw_pdf_path",
    "text_path",
}
CSV_SOURCE_COLUMNS = {"source", "source_dataset"}
CSV_FILES = [
    ROOT / "data" / "audit" / "alignment_label_discovery.csv",
    ROOT / "data" / "audit" / "data_inventory.csv",
    ROOT / "data" / "audit" / "desktop_root_data_counts.csv",
    ROOT / "data" / "model_sec_desktop" / "sec_companyfacts_file_inventory.csv",
    ROOT / "data" / "model_sec_desktop" / "sec_companyfacts_validation_coverage.csv",
    ROOT / "data" / "processed" / "jp_report_parser_prototype_events.csv",
    ROOT / "data" / "raw" / "edinet" / "jp_report_parser_prototype_index.csv",
    ROOT / "data" / "audit" / "t04_local_candidate_files.csv",
    ROOT / "data" / "raw" / "sec_all8k_text" / "sec_all8k_text_manifest.csv",
    ROOT / "data" / "raw" / "sec_desktop_population" / "sec_desktop_population_8k_filings.csv",
    ROOT / "data" / "raw" / "sec_desktop_population" / "sec_desktop_population_file_summary.csv",
    ROOT / "data" / "raw" / "sec_desktop_population" / "sec_desktop_population_outcome_candidates.csv",
    ROOT / "data" / "raw" / "sec_desktop_population_smoke" / "sec_desktop_population_smoke_8k_filings.csv",
    ROOT / "data" / "raw" / "sec_desktop_population_smoke" / "sec_desktop_population_smoke_file_summary.csv",
    ROOT / "data" / "raw" / "sec_desktop_population_smoke" / "sec_desktop_population_smoke_outcome_candidates.csv",
    ROOT / "data" / "raw" / "sec_submissions_archive" / "sec_submissions_archive_download_log.csv",
    ROOT / "data" / "raw" / "sec_submissions_archive" / "sec_submissions_archive_manifest.csv",
]
TEXT_INDEX_FILES = [
    ROOT / "data" / "audit" / "desktop_data_files_rg.txt",
]
JSON_FILES = [
    ROOT / "data" / "audit" / "inspected_files.json",
    ROOT / "data" / "audit" / "summary.json",
]
SOURCE_TEXT_ROOTS = [
    ROOT / "data" / "raw" / "sec_filing_text",
]
ACTIVE_SECRET_SCAN_ROOTS = [
    ROOT / "manuscript",
    ROOT / "results",
    ROOT / "src",
    ROOT / "submission_package",
]
FORBIDDEN_PATTERNS = [
    "C:/Users",
    "C:\\Users",
    "C:\\\\Users",
    "Desktop/M&A",
    "Desktop\\M&A",
    "Desktop\\\\M&A",
    "Desktop/Paper 4",
    "Desktop\\Paper 4",
    "Desktop\\\\Paper 4",
    "desktop_external",
    "grant_back",
    "SEC Desktop",
]
SELF_REFERENTIAL_AUDIT_OUTPUTS = {
    "reproducibility_run_report.json",
    "results/public_path_redaction_audit.json",
    "results/public_path_redaction_audit.md",
    "results/public_submission_manifest_audit.json",
    "results/public_submission_manifest_audit.md",
    "results/remaining_risk_audit.json",
    "results/remaining_risk_audit.md",
    "results/final_claim_boundary_audit.json",
    "results/final_claim_boundary_audit.md",
    "results/reproducibility_run_report_audit.json",
    "results/reproducibility_run_report_audit.md",
}
SECRET_LIKE_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|gsk_[A-Za-z0-9_-]{16,}|[A-Fa-f0-9]{32,})"
)
ACTIVE_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|gsk_[A-Za-z0-9_-]{16,}|"
    r"\b(?:DART_API_KEY|OPENAI_API_KEY|GROQ_API_KEY|DEEPSEEK_API_KEY|ANTHROPIC_API_KEY|"
    r"EDINET_API_KEY|SEC_USER_AGENT|EDGAR_USER_AGENT|NEO4J_PASSWORD)\s*=\s*"
    r"(?![\"']?(?:\.\.\.|<[^>]+>|your[ _-]|YOUR[ _-]|$))[\"']?[^\"'\s<>]{8,})"
)
WINDOWS_SOURCE_PATH_RE = re.compile(
    r"(?:file:///)?[A-Za-z]:(?:/|\\)Users(?:/|\\)[^\"'<>\n\r]*",
    flags=re.IGNORECASE,
)


def public_id(value: object, scope: str = "file") -> str:
    text = str(value or "")
    if text.startswith("public_") or text.startswith("local_inventory/") or text.startswith("source_group_"):
        return text
    suffix = Path(text).suffix.lower() or ".file"
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"public_{scope}/source_{digest}{suffix}"


def redact_source_label(value: object) -> str:
    text = str(value or "")
    replacements = {
        "Desktop/M&A SEC submissions": "local SEC submissions",
        "Desktop\\M&A SEC submissions": "local SEC submissions",
        "SEC Desktop": "Local SEC",
        "desktop_external": "external_local",
        "grant_back": "external_prior_work",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def redact_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing", "changed_cells": 0}
    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    changed = 0
    for column in df.columns:
        if column in CSV_PATH_COLUMNS or column.endswith("_path"):
            original = df[column].fillna("").astype(str)
            redacted = original.map(lambda value, column=column: public_id(value, column) if value else "")
            changed += int((original != redacted).sum())
            df[column] = redacted
        elif column in CSV_SOURCE_COLUMNS:
            original = df[column].fillna("").astype(str)
            redacted = original.map(redact_source_label)
            changed += int((original != redacted).sum())
            df[column] = redacted
        elif column == "desktop_root":
            original = df[column].fillna("").astype(str)
            redacted = [f"source_group_{idx:02d}" for idx in range(1, len(df) + 1)]
            changed += int((original != pd.Series(redacted, index=df.index)).sum())
            df[column] = redacted
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return {"path": str(path), "status": "redacted", "changed_cells": changed, "rows": int(len(df))}


def redact_text_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing", "changed_lines": 0}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    redacted = [public_id(line, "indexed_file") if line.strip() else "" for line in lines]
    changed = sum(1 for old, new in zip(lines, redacted) if old != new)
    path.write_text("\n".join(redacted) + ("\n" if redacted else ""), encoding="utf-8")
    return {"path": str(path), "status": "redacted", "changed_lines": changed, "rows": len(lines)}


def redact_json_object(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if child_key == "top_desktop_roots" and isinstance(child_value, dict):
                result["top_source_groups"] = {
                    f"source_group_{idx:02d}": count
                    for idx, (_name, count) in enumerate(child_value.items(), start=1)
                }
            elif child_key in CSV_PATH_COLUMNS or child_key.endswith("_path") or child_key == "file":
                result[child_key] = public_id(child_value, child_key) if child_value else child_value
            elif child_key in CSV_SOURCE_COLUMNS:
                result[child_key] = redact_source_label(child_value)
            else:
                result[child_key] = redact_json_object(child_value, child_key)
        return result
    if isinstance(value, list):
        if key == "date_ranges":
            result = []
            for item in value:
                if isinstance(item, list) and len(item) >= 4:
                    item = list(item)
                    item[3] = public_id(item[3], "date_range_file")
                result.append(redact_json_object(item, key))
            return result
        return [redact_json_object(item, key) for item in value]
    if isinstance(value, str):
        return redact_source_label(value)
    return value


def count_forbidden(text: str) -> int:
    return sum(text.count(pattern) for pattern in FORBIDDEN_PATTERNS)


def is_scan_excluded(rel: str, extra_excluded: set[str] | None = None) -> bool:
    excluded = set(SELF_REFERENTIAL_AUDIT_OUTPUTS)
    if extra_excluded:
        excluded.update(extra_excluded)
    if rel in excluded:
        return True
    return rel.startswith("results/run_v1_pipeline_") and (rel.endswith(".err") or rel.endswith(".log"))


def public_manifest_paths() -> set[str]:
    if not PUBLIC_SUBMISSION_MANIFEST.exists():
        return set()
    with PUBLIC_SUBMISSION_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        return {
            str(row.get("path", "")).replace("\\", "/").lstrip("/")
            for row in csv.DictReader(handle)
            if row.get("path")
        }


def self_referential_manifest_exclusions() -> list[str]:
    return sorted(public_manifest_paths() & SELF_REFERENTIAL_AUDIT_OUTPUTS)


def redact_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing", "forbidden_before": 0, "forbidden_after": 0}
    before = path.read_text(encoding="utf-8", errors="ignore")
    data = json.loads(before)
    redacted = redact_json_object(data)
    path.write_text(json.dumps(redacted, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    after = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "path": str(path),
        "status": "redacted",
        "forbidden_before": count_forbidden(before),
        "forbidden_after": count_forbidden(after),
    }


def redact_source_texts(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"path": str(root), "status": "missing", "changed_files": 0, "replacements": 0}
    changed_files = 0
    replacements = 0
    for path in root.rglob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        new_text, count = WINDOWS_SOURCE_PATH_RE.subn("[redacted_source_local_path]", text)
        if count:
            path.write_text(new_text, encoding="utf-8")
            changed_files += 1
            replacements += count
    return {
        "path": str(root),
        "status": "redacted",
        "changed_files": changed_files,
        "replacements": replacements,
    }


def scan_forbidden() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    extensions = {".csv", ".txt", ".tsv", ".md", ".json", ".py", ".err", ".log"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if is_scan_excluded(rel, extra_excluded={
            "src/audit/audit_forbidden_tokens.py",
            "src/audit/public_path_redaction.py",
        }):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in text:
                hits.append({"path": rel, "pattern": pattern})
                break
    return hits


def active_secret_candidate_paths() -> list[Path]:
    top_level_files = [
        ROOT / "audit_report.md",
        ROOT / "data_map.md",
        ROOT / "paper4_completion_audit.md",
        ROOT / "paper4_completion_audit.json",
        ROOT / "paper4_execution_status.md",
        ROOT / "paper4_research_plan.md",
        ROOT / "REPRODUCIBILITY.md",
        ROOT / "reproducibility_manifest.json",
        ROOT / "reproducibility_run_report.json",
        ROOT / "VERSIONS.md",
    ]
    candidate_paths: list[Path] = []
    for root in ACTIVE_SECRET_SCAN_ROOTS:
        if root.exists():
            candidate_paths.extend(path for path in root.rglob("*") if path.is_file())
    candidate_paths.extend(path for path in top_level_files if path.exists())
    return candidate_paths


def scan_active_secret_values() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    extensions = {".md", ".json", ".py"}
    seen: set[str] = set()
    for path in active_secret_candidate_paths():
        key = str(path)
        if key in seen or path.suffix.lower() not in extensions:
            continue
        seen.add(key)
        rel = path.relative_to(ROOT).as_posix()
        if is_scan_excluded(rel, extra_excluded={
            "src/audit/public_path_redaction.py",
        }):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if ACTIVE_SECRET_RE.search(line):
                hits.append({"path": rel, "line_number": line_number})
                break
    return hits


def sensitive_workspace_guard() -> dict[str, Any]:
    gitignore_text = GITIGNORE.read_text(encoding="utf-8", errors="ignore") if GITIGNORE.exists() else ""
    env_example_text = ENV_EXAMPLE.read_text(encoding="utf-8", errors="ignore") if ENV_EXAMPLE.exists() else ""
    paper4_env_files = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob(".env*")
        if path.is_file()
    ]
    checks = {
            "gitignore_exists": GITIGNORE.exists(),
        "env_ignored": re.search(r"(?m)^\.env$", gitignore_text) is not None,
        "env_wildcard_ignored": re.search(r"(?m)^\.env\.\*$", gitignore_text) is not None,
        "env_example_unignored": re.search(r"(?m)^!\.env\.example$", gitignore_text) is not None,
        "env_example_exists": ENV_EXAMPLE.exists(),
        "env_example_has_no_secret_like_values": SECRET_LIKE_RE.search(env_example_text) is None,
        "paper4_contains_no_env_files": not paper4_env_files,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "paper4_env_files": paper4_env_files,
        "guardrail": (
            "Local credential files may exist outside Paper 4 for execution, but public/submission artifacts "
            "must keep .env ignored, provide only a template-only .env.example, and contain no Paper 4-local .env files."
        ),
    }


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# Public Path Redaction Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| CSV files checked | {len(audit['csv_files'])} |",
        f"| Text index files checked | {len(audit['text_index_files'])} |",
        f"| JSON files checked | {len(audit['json_files'])} |",
        f"| Source text roots checked | {len(audit['source_text_roots'])} |",
        f"| Active secret scan roots | {len(audit['active_secret_scan_roots'])} |",
        f"| Active secret candidate files | {audit['active_secret_candidate_file_count']} |",
        f"| Sensitive workspace guard | {audit['sensitive_workspace_guard']['status']} |",
        f"| Active secret value hits | {audit['active_secret_value_hit_count']} |",
        f"| Residual forbidden hits | {audit['residual_hit_count']} |",
        f"| Self-referential manifest exclusions | {audit['self_referential_manifest_exclusion_count']} |",
        "",
        "## Guardrail",
        "",
        audit["guardrail"],
        "",
    ]
    if audit["residual_hits"]:
        lines.extend(["## Residual Hits", "", "| Path | Pattern |", "|---|---|"])
        for row in audit["residual_hits"][:50]:
            lines.append(f"| `{row['path']}` | `{row['pattern']}` |")
        lines.append("")
    if audit["self_referential_manifest_exclusions"]:
        lines.extend(["## Self-Referential Manifest Exclusions", "", "| Path | Reason |", "|---|---|"])
        for item in audit["self_referential_manifest_exclusions"]:
            lines.append(
                f"| `{item}` | Excluded from this self-scan to avoid circular self-hits; covered by core package ZIP and post-run security scans. |"
            )
        lines.append("")
    if audit["active_secret_value_hits"]:
        lines.extend(["## Active Secret Value Hits", "", "| Path | First line |", "|---|---:|"])
        for row in audit["active_secret_value_hits"][:50]:
            lines.append(f"| `{row['path']}` | {row['line_number']} |")
        lines.append("")
    guard = audit["sensitive_workspace_guard"]
    lines.extend(["## Sensitive Workspace Guard", "", "| Check | OK |", "|---|---:|"])
    for name, ok in guard["checks"].items():
        lines.append(f"| `{name}` | {ok} |")
    if guard["paper4_env_files"]:
        lines.extend(["", "Paper 4-local environment files:", ""])
        for item in guard["paper4_env_files"]:
            lines.append(f"- `{item}`")
    lines.extend(["", guard["guardrail"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    csv_results = [redact_csv(path) for path in CSV_FILES]
    text_results = [redact_text_index(path) for path in TEXT_INDEX_FILES]
    json_results = [redact_json(path) for path in JSON_FILES]
    source_text_results = [redact_source_texts(path) for path in SOURCE_TEXT_ROOTS]
    residual_hits = scan_forbidden()
    secret_hits = scan_active_secret_values()
    sensitive_guard = sensitive_workspace_guard()
    manifest_self_exclusions = self_referential_manifest_exclusions()
    audit = {
        "status": "pass" if not residual_hits and not secret_hits and sensitive_guard["status"] == "pass" else "fail",
        "csv_files": csv_results,
        "text_index_files": text_results,
        "json_files": json_results,
        "source_text_roots": source_text_results,
        "active_secret_scan_roots": [
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in ACTIVE_SECRET_SCAN_ROOTS
            if path.exists()
        ],
        "active_secret_candidate_file_count": len(
            {
                str(path)
                for path in active_secret_candidate_paths()
                if path.suffix.lower() in {".md", ".json", ".py"}
                and not is_scan_excluded(
                    path.relative_to(ROOT).as_posix(),
                    extra_excluded={"src/audit/public_path_redaction.py"},
                )
            }
        ),
        "sensitive_workspace_guard": sensitive_guard,
        "active_secret_value_hits": secret_hits,
        "active_secret_value_hit_count": len(secret_hits),
        "self_referential_manifest_exclusions": manifest_self_exclusions,
        "self_referential_manifest_exclusion_count": len(manifest_self_exclusions),
        "self_referential_manifest_exclusion_guardrail": (
            "Self-referential audit outputs listed here are intentionally excluded from public-path self-scans "
            "to avoid circular self-hits, but are included in the public submission package and must be covered "
            "by core package ZIP and post-run security scans."
        ),
        "residual_hits": residual_hits,
        "residual_hit_count": len(residual_hits),
        "guardrail": (
            "Submission-facing public artifacts must not expose local absolute paths, internal workspace labels, "
            "or prior-project root names. Raw source text path fragments are masked without changing analytical tables."
        ),
    }
    out_json = ROOT / "results" / "public_path_redaction_audit.json"
    out_md = ROOT / "results" / "public_path_redaction_audit.md"
    out_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(audit, out_md)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "residual_hit_count": audit["residual_hit_count"],
                "active_secret_value_hit_count": audit["active_secret_value_hit_count"],
                "sensitive_workspace_guard": sensitive_guard["status"],
            },
            indent=2,
        )
    )
    if residual_hits or secret_hits or sensitive_guard["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
