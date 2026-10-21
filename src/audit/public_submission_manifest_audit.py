from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
RESULTS = ROOT / "results"
AUDIT_DIR = ROOT / "data" / "audit"
OUT_JSON = RESULTS / "public_submission_manifest_audit.json"
OUT_MD = RESULTS / "public_submission_manifest_audit.md"
OUT_CSV = AUDIT_DIR / "public_submission_manifest.csv"


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    category: str
    role: str


REQUIRED_PACKAGE_FILES = [
    ManifestEntry("manuscript/paper4_v4_manuscript_en.docx", "manuscript", "English Word manuscript"),
    ManifestEntry("manuscript/paper4_v4_manuscript_ko.docx", "manuscript", "Korean Word manuscript"),
    ManifestEntry("manuscript/paper4_v4_title_page.docx", "manuscript", "title-page Word document"),
    ManifestEntry("manuscript/submission_candidate_journals_ko.docx", "manuscript", "candidate-journal Word table"),
    ManifestEntry("manuscript/paper4_v4_manuscript_en.md", "manuscript_source", "English manuscript source"),
    ManifestEntry("manuscript/paper4_v4_manuscript_ko.md", "manuscript_source", "Korean manuscript source"),
    ManifestEntry("manuscript/paper4_v4_title_page.md", "manuscript_source", "title-page source"),
    ManifestEntry("manuscript/submission_candidate_journals_ko.md", "manuscript_source", "candidate-journal table source"),
    ManifestEntry("manuscript/paper4_target_journal_expansion_ko.md", "submission_support", "target-journal expansion"),
    ManifestEntry("manuscript/submission_strategy.md", "submission_support", "submission strategy"),
    ManifestEntry("manuscript/submission_checklist_v1.md", "submission_support", "submission checklist"),
    ManifestEntry("manuscript/cover_letter_v1_generic.md", "submission_support", "generic cover letter"),
    ManifestEntry("manuscript/claim_readiness_matrix.md", "submission_support", "claim readiness matrix"),
    ManifestEntry("manuscript/references.bib", "submission_support", "reference bibliography"),
    ManifestEntry("paper4_research_plan.md", "readiness", "research plan"),
    ManifestEntry("REPRODUCIBILITY.md", "readiness", "reproducibility instructions"),
    ManifestEntry("VERSIONS.md", "readiness", "version boundary"),
    ManifestEntry("data_map.md", "readiness", "data map"),
    ManifestEntry("paper4_execution_status.md", "readiness", "execution status"),
    ManifestEntry("paper4_completion_audit.md", "readiness", "completion audit"),
    ManifestEntry("results/calculation_sheet.md", "calculation", "calculation sheet"),
    ManifestEntry("results/calculation_sheet.xlsx", "calculation", "calculation workbook"),
    ManifestEntry("results/calculation_sheet.csv", "calculation", "calculation table"),
    ManifestEntry("figures/paper4_framework.png", "figure", "framework figure"),
    ManifestEntry("figures/paper4_model_performance.png", "figure", "model-performance figure"),
    ManifestEntry("figures/paper4_market_coverage.png", "figure", "market-coverage figure"),
    ManifestEntry("figures/en/paper4_framework_en.png", "figure", "English framework figure"),
    ManifestEntry("figures/en/paper4_data_scope_en.png", "figure", "English data-scope figure"),
    ManifestEntry("figures/en/paper4_market_coverage_en.png", "figure", "English market-coverage figure"),
    ManifestEntry("figures/en/paper4_model_performance_en.png", "figure", "English model-performance figure"),
    ManifestEntry("figures/en/paper4_permutation_importance_en.png", "figure", "English permutation-importance figure"),
    ManifestEntry("figures/en/paper4_sequence_market_reaction_en.png", "figure", "English sequence-market reaction figure"),
    ManifestEntry("figures/en/paper4_disclosure_price_reaction_en.png", "figure", "English disclosure-window price-reaction figure"),
    ManifestEntry("figures/en/paper4_llm_context_ablation_en.png", "figure", "English LLM context-ablation figure"),
    ManifestEntry("results/figure_generation_audit.md", "audit", "figure generation audit"),
    ManifestEntry("results/submission_figure_generation_audit.md", "audit", "English submission figure generation audit"),
    ManifestEntry("results/submission_figure_generation_audit.json", "audit", "English submission figure generation audit data"),
    ManifestEntry("results/word_export_audit.md", "audit", "Word export audit"),
    ManifestEntry("results/journal_candidate_source_audit.md", "audit", "journal candidate source audit"),
    ManifestEntry("results/public_path_redaction_audit.md", "audit", "public path and sensitive workspace audit"),
    ManifestEntry("results/submission_package_audit.md", "audit", "submission package text audit"),
    ManifestEntry("results/research_plan_traceability.md", "audit", "research plan traceability"),
    ManifestEntry("results/bilingual_manuscript_audit.md", "audit", "bilingual manuscript audit"),
    ManifestEntry("results/manuscript_claim_audit.md", "audit", "manuscript claim audit"),
    ManifestEntry("results/manuscript_metric_audit.md", "audit", "manuscript metric audit"),
    ManifestEntry("results/reference_consistency_audit.md", "audit", "reference consistency audit"),
    ManifestEntry("results/market_extension_summary.md", "evidence", "market extension summary"),
    ManifestEntry("results/market_residual_coverage_audit.md", "evidence", "market residual coverage audit"),
    ManifestEntry("results/graphrag/disclosure_price_reaction_summary.csv", "evidence", "disclosure-window price-reaction summary"),
    ManifestEntry("results/graphrag/disclosure_price_reaction_case_table.csv", "evidence", "benchmark case price-reaction table"),
    ManifestEntry("results/graphrag/disclosure_price_reaction_by_llm_prediction.csv", "evidence", "LLM prediction price-reaction grouping"),
    ManifestEntry("results/graphrag/llm_market_context_summary.md", "evidence", "LLM context ablation summary"),
    ManifestEntry("results/graphrag/llm_market_context_metrics.csv", "evidence", "LLM context ablation metrics"),
]

DENIED_PREFIXES = (
    ".env",
    ".omx/",
    ".claude/",
    ".git/",
    "src/",
    "data/raw/",
    "data/interim/",
    "data/processed/",
    "data/model_",
    "docs_cache/",
)
DENIED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
DENIED_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pkl", ".pickle", ".pyc", ".log", ".zip", ".tar", ".7z"}
TEXT_SUFFIXES = {".md", ".bib", ".csv", ".json", ".txt"}
FORBIDDEN_TEXT_PATTERNS = (
    "C:" + "/Users",
    "C:" + "\\Users",
    "C:" + "\\\\" + "Users",
    "Desktop/" + "M&A",
    "Desktop\\" + "M&A",
    "Desktop\\\\" + "M&A",
    "Desktop/" + "Paper 4",
    "Desktop\\" + "Paper 4",
    "Desktop\\\\" + "Paper 4",
    "desktop" + "_external",
    "grant" + "_back",
    "SEC" + " Desktop",
)
ACTIVE_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|gsk_[A-Za-z0-9_-]{16,}|"
    r"\b(?:DART_API_KEY|OPENAI_API_KEY|GROQ_API_KEY|DEEPSEEK_API_KEY|ANTHROPIC_API_KEY|"
    r"EDINET_API_KEY|SEC_USER_AGENT|EDGAR_USER_AGENT|NEO4J_PASSWORD)\s*=\s*"
    r"(?![\"']?(?:\.\.\.|<[^>]+>|your[ _-]|YOUR[ _-]|$))[\"']?[^\"'\s<>]{8,})"
)


def rel(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing"}
    return json.loads(path.read_text(encoding="utf-8"))


def denied_reasons(relative_path: str) -> list[str]:
    normalized = rel(relative_path)
    path = Path(normalized)
    reasons: list[str] = []
    for prefix in DENIED_PREFIXES:
        if normalized == prefix.rstrip("/") or normalized.startswith(prefix):
            reasons.append(f"denied prefix: {prefix}")
    if any(part in DENIED_PARTS for part in path.parts):
        reasons.append("denied cache path")
    if path.suffix.lower() in DENIED_SUFFIXES:
        reasons.append(f"denied suffix: {path.suffix.lower()}")
    return reasons


def scan_text_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in TEXT_SUFFIXES or not path.exists():
        return {"local_path_hits": [], "active_secret_hits": 0}
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"local_path_hits": [], "active_secret_hits": 0}
    local_hits = [pattern for pattern in FORBIDDEN_TEXT_PATTERNS if pattern in text]
    return {
        "local_path_hits": local_hits,
        "active_secret_hits": len(ACTIVE_SECRET_RE.findall(text)),
    }


def build_manifest_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in REQUIRED_PACKAGE_FILES:
        path = ROOT / entry.path
        exists = path.exists()
        text_scan = scan_text_file(path)
        rows.append(
            {
                "path": entry.path,
                "category": entry.category,
                "role": entry.role,
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
                "sha256": sha256_file(path) if exists else "",
                "denied_reasons": denied_reasons(entry.path),
                "local_path_hits": text_scan["local_path_hits"],
                "active_secret_hits": text_scan["active_secret_hits"],
            }
        )
    return rows


def build_audit() -> dict[str, Any]:
    rows = build_manifest_rows()
    public_path_audit = read_json(RESULTS / "public_path_redaction_audit.json")
    sensitive_guard = public_path_audit.get("sensitive_workspace_guard", {})
    missing = [row for row in rows if not row["exists"]]
    denied = [row for row in rows if row["denied_reasons"]]
    local_path_hits = [
        {"path": row["path"], "hits": row["local_path_hits"]}
        for row in rows
        if row["local_path_hits"]
    ]
    active_secret_hits = sum(int(row["active_secret_hits"]) for row in rows)
    public_dependency_ok = (
        public_path_audit.get("status") == "pass"
        and public_path_audit.get("residual_hit_count") == 0
        and public_path_audit.get("active_secret_value_hit_count") == 0
        and sensitive_guard.get("status") == "pass"
    )
    status = "pass"
    if missing or denied or local_path_hits or active_secret_hits or not public_dependency_ok:
        status = "fail"
    return {
        "status": status,
        "package_file_count": len(rows),
        "package_total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "missing_required_count": len(missing),
        "denied_manifest_entry_count": len(denied),
        "local_path_hit_count": len(local_path_hits),
        "active_secret_hit_count": active_secret_hits,
        "public_path_dependency_status": public_path_audit.get("status"),
        "public_path_residual_hit_count": public_path_audit.get("residual_hit_count"),
        "public_path_active_secret_value_hit_count": public_path_audit.get("active_secret_value_hit_count"),
        "sensitive_workspace_guard_status": sensitive_guard.get("status"),
        "manifest_rows": rows,
        "missing_required_paths": [row["path"] for row in missing],
        "denied_manifest_entries": [
            {"path": row["path"], "denied_reasons": row["denied_reasons"]}
            for row in denied
        ],
        "local_path_hits": local_path_hits,
        "guardrail": (
            "This audit defines a core public/submission bundle allowlist and verifies that raw/cache, "
            "environment, database, archive, and local-path material is not selected for that bundle."
        ),
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "category",
                "role",
                "exists",
                "size_bytes",
                "sha256",
                "denied_reasons",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row["path"],
                    "category": row["category"],
                    "role": row["role"],
                    "exists": row["exists"],
                    "size_bytes": row["size_bytes"],
                    "sha256": row["sha256"],
                    "denied_reasons": "; ".join(row["denied_reasons"]),
                }
            )


def write_markdown(audit: dict[str, Any]) -> None:
    lines = [
        "# Public Submission Manifest Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Package files | {audit['package_file_count']} |",
        f"| Total bytes | {audit['package_total_bytes']} |",
        f"| Missing required files | {audit['missing_required_count']} |",
        f"| Denied manifest entries | {audit['denied_manifest_entry_count']} |",
        f"| Local path text hits | {audit['local_path_hit_count']} |",
        f"| Active secret text hits | {audit['active_secret_hit_count']} |",
        f"| Public path dependency | `{audit['public_path_dependency_status']}` |",
        f"| Sensitive workspace guard | `{audit['sensitive_workspace_guard_status']}` |",
        "",
        "## Included Files",
        "",
        "| Category | Path | Size bytes | SHA-256 |",
        "|---|---|---:|---|",
    ]
    for row in audit["manifest_rows"]:
        sha = row["sha256"][:16] if row["sha256"] else ""
        lines.append(f"| {row['category']} | `{row['path']}` | {row['size_bytes']} | `{sha}` |")
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            audit["guardrail"],
        ]
    )
    if audit["denied_manifest_entries"] or audit["missing_required_paths"] or audit["local_path_hits"]:
        lines.extend(["", "## Findings", ""])
        for path in audit["missing_required_paths"]:
            lines.append(f"- Missing required file: `{path}`")
        for row in audit["denied_manifest_entries"]:
            reasons = "; ".join(row["denied_reasons"])
            lines.append(f"- Denied manifest entry: `{row['path']}` ({reasons})")
        for row in audit["local_path_hits"]:
            lines.append(f"- Local path text hit: `{row['path']}`")
    else:
        lines.extend(["", "No package-boundary findings."])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    write_csv(audit["manifest_rows"])
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "package_file_count": audit["package_file_count"],
                "denied_manifest_entry_count": audit["denied_manifest_entry_count"],
                "missing_required_count": audit["missing_required_count"],
            },
            indent=2,
        )
    )
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
