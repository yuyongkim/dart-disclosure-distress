from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audit_forbidden_tokens import FORBIDDEN_TOKENS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
RESULTS = ROOT / "results"
MIN_DOCX_TEXT_CHARS = 10000
CLAIM_AUDIT_PATH = ROOT / "src" / "audit" / "manuscript_claim_audit.py"
EN_SUBMISSION_FIGURES = (
    "figures/en/paper4_framework_en.png",
    "figures/en/paper4_data_scope_en.png",
    "figures/en/paper4_market_coverage_en.png",
    "figures/en/paper4_model_performance_en.png",
    "figures/en/paper4_permutation_importance_en.png",
    "figures/en/paper4_sequence_market_reaction_en.png",
    "figures/en/paper4_disclosure_price_reaction_en.png",
    "figures/en/paper4_llm_context_ablation_en.png",
)


@dataclass(frozen=True)
class WordDoc:
    label: str
    md_path: str
    docx_path: str
    required_groups: list[list[str]]
    required_contexts: list[list[str]]
    min_media_count: int = 0
    min_table_count: int = 0
    required_media_sources: tuple[str, ...] = ()
    min_hangul_count: int = 0
    max_replacement_chars: int = 0
    min_text_chars: int = MIN_DOCX_TEXT_CHARS
    ranked_table_expected_journals: tuple[str, ...] = ()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_docx_text(path: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["pandoc", str(path), "-t", "plain"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def extract_docx_structure(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as docx:
        names = docx.namelist()
        media_files = [name for name in names if name.startswith("word/media/")]
        media_hashes = {name: hashlib.sha256(docx.read(name)).hexdigest() for name in media_files}
        document_xml = docx.read("word/document.xml").decode("utf-8", errors="ignore")
    return {
        "media_count": len(media_files),
        "media_files": media_files,
        "media_hashes": media_hashes,
        "table_count": document_xml.count("<w:tbl>"),
    }


def extract_docx_xml_text(path: Path) -> str:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as docx:
        root = ET.fromstring(docx.read("word/document.xml"))
    return " ".join(node.text for node in root.findall(".//w:t", namespace) if node.text)


def extract_docx_tables(path: Path) -> list[list[list[str]]]:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as docx:
        root = ET.fromstring(docx.read("word/document.xml"))

    tables = []
    for table in root.findall(".//w:tbl", namespace):
        rows = []
        for row in table.findall("./w:tr", namespace):
            cells = []
            for cell in row.findall("./w:tc", namespace):
                text = " ".join(node.text for node in cell.findall(".//w:t", namespace) if node.text)
                cells.append(re.sub(r"\s+", " ", text).strip())
            rows.append(cells)
        tables.append(rows)
    return tables


def ranked_journal_table_check(
    tables: list[list[list[str]]],
    expected_journals: tuple[str, ...],
) -> dict[str, Any]:
    expected = list(expected_journals)
    expected_ranks = list(range(1, len(expected) + 1))
    best: dict[str, Any] = {
        "ok": False,
        "expected_rows": len(expected),
        "matched_rows": 0,
        "table_index": None,
        "observed_ranks": [],
        "expected_ranks": expected_ranks,
        "observed_journals": [],
        "missing_journals": expected,
        "unexpected_journals": [],
        "ranks_ok": False,
    }
    for table_index, table in enumerate(tables, start=1):
        ranked_rows = []
        for cells in table:
            if len(cells) >= 2 and re.fullmatch(r"\d+", cells[0].strip()):
                ranked_rows.append({"rank": int(cells[0]), "journal": cells[1].strip()})
        if not ranked_rows:
            continue
        ranks = [row["rank"] for row in ranked_rows]
        journals = [row["journal"] for row in ranked_rows]
        missing = [journal for journal in expected if journal not in journals]
        unexpected = [journal for journal in journals if journal not in expected]
        matched_rows = len(expected) - len(missing)
        table_check = {
            "ok": ranks == expected_ranks and matched_rows == len(expected) and not unexpected,
            "expected_rows": len(expected),
            "matched_rows": matched_rows,
            "table_index": table_index,
            "observed_ranks": ranks,
            "expected_ranks": expected_ranks,
            "observed_journals": journals,
            "missing_journals": missing,
            "unexpected_journals": unexpected,
            "ranks_ok": ranks == expected_ranks,
        }
        if table_check["ok"] or matched_rows > best["matched_rows"]:
            best = table_check
        if table_check["ok"]:
            break
    return best


def claim_findings_for_text(label: str, text: str) -> tuple[list[dict[str, Any]], str]:
    if not CLAIM_AUDIT_PATH.exists():
        return [], f"missing claim audit module: {CLAIM_AUDIT_PATH}"
    spec = importlib.util.spec_from_file_location("manuscript_claim_audit", CLAIM_AUDIT_PATH)
    if spec is None or spec.loader is None:
        return [], f"cannot load claim audit module: {CLAIM_AUDIT_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    audit_text = getattr(module, "audit_text", None)
    if audit_text is None:
        return [], "claim audit module does not expose audit_text"
    return audit_text(label, text), ""


def line_numbers(text: str, token: str) -> list[int]:
    return [idx for idx, line in enumerate(text.splitlines(), start=1) if token in line]


def context_lines(text: str, context: list[str], max_span: int = 500) -> list[int]:
    split_lines = text.splitlines()
    exact = [idx for idx, line in enumerate(split_lines, start=1) if all(token in line for token in context)]
    if exact:
        return exact
    for start in range(len(split_lines)):
        window_lines = split_lines[start : start + 3]
        window = " ".join(window_lines)
        if all(token in window for token in context):
            return list(range(start + 1, start + 1 + len(window_lines)))
    return []


def evaluate_doc(doc: WordDoc) -> dict[str, Any]:
    md_path = ROOT / doc.md_path
    docx_path = ROOT / doc.docx_path
    base = {
        "label": doc.label,
        "markdown": doc.md_path,
        "docx": doc.docx_path,
        "markdown_exists": md_path.exists(),
        "docx_exists": docx_path.exists(),
        "status": "passed",
    }
    if not md_path.exists() or not docx_path.exists():
        base["status"] = "missing_file"
        return base

    returncode, text, stderr = extract_docx_text(docx_path)
    xml_text = ""
    xml_text_error = ""
    try:
        xml_text = extract_docx_xml_text(docx_path)
    except Exception as exc:  # pragma: no cover - defensive audit detail
        xml_text_error = f"{type(exc).__name__}: {exc}"
    matching_text = text + "\n" + xml_text
    structure_error = ""
    structure: dict[str, Any] = {"media_count": 0, "media_files": [], "table_count": 0}
    try:
        structure = extract_docx_structure(docx_path)
    except Exception as exc:  # pragma: no cover - defensive audit detail
        structure_error = f"{type(exc).__name__}: {exc}"
    ranked_table_error = ""
    ranked_table_check = {
        "ok": not doc.ranked_table_expected_journals,
        "expected_rows": len(doc.ranked_table_expected_journals),
        "matched_rows": 0,
    }
    if doc.ranked_table_expected_journals:
        try:
            ranked_table_check = ranked_journal_table_check(
                extract_docx_tables(docx_path),
                doc.ranked_table_expected_journals,
            )
        except Exception as exc:  # pragma: no cover - defensive audit detail
            ranked_table_error = f"{type(exc).__name__}: {exc}"
    docx_newer_or_equal_markdown = docx_path.stat().st_mtime >= md_path.stat().st_mtime
    base.update(
        {
            "pandoc_returncode": returncode,
            "pandoc_stderr_tail": stderr[-1000:],
            "docx_text_chars": len(text),
            "docx_xml_text_chars": len(xml_text),
            "docx_xml_text_error": xml_text_error,
            "docx_hangul_count": sum(0xAC00 <= ord(char) <= 0xD7A3 for char in text),
            "min_hangul_count": doc.min_hangul_count,
            "docx_replacement_char_count": text.count("\ufffd"),
            "max_replacement_chars": doc.max_replacement_chars,
            "min_docx_text_chars": doc.min_text_chars,
            "docx_media_count": structure["media_count"],
            "min_media_count": doc.min_media_count,
            "docx_table_count": structure["table_count"],
            "min_table_count": doc.min_table_count,
            "docx_media_files": structure["media_files"],
            "required_media_sources": list(doc.required_media_sources),
            "docx_structure_error": structure_error,
            "ranked_table_check": ranked_table_check,
            "ranked_table_expected_rows": ranked_table_check.get("expected_rows", 0),
            "ranked_table_matched_rows": ranked_table_check.get("matched_rows", 0),
            "ranked_table_ok": ranked_table_check.get("ok", False),
            "ranked_table_error": ranked_table_error,
            "markdown_sha256": sha256(md_path),
            "docx_sha256": sha256(docx_path),
            "docx_newer_or_equal_markdown": docx_newer_or_equal_markdown,
        }
    )
    if returncode != 0:
        base["status"] = "pandoc_extract_failed"
        return base

    claim_findings, claim_error = claim_findings_for_text(doc.docx_path, text)
    claim_blockers = [row for row in claim_findings if row.get("severity") == "blocker"]
    claim_review = [row for row in claim_findings if row.get("severity") != "blocker"]

    groups = []
    for group in doc.required_groups:
        matched = [token for token in group if token in matching_text]
        groups.append(
            {
                "accepted_values": group,
                "matched": bool(matched),
                "matched_values": matched,
                "line_numbers": sorted({line for token in matched for line in line_numbers(matching_text, token)}),
            }
        )
    contexts = []
    for context in doc.required_contexts:
        lines = context_lines(matching_text, context)
        contexts.append(
            {
                "required_nearby": context,
                "matched": bool(lines),
                "line_numbers": lines,
            }
        )
    forbidden = [
        {"token": token, "line_numbers": line_numbers(text, token)}
        for token in FORBIDDEN_TOKENS
        if token in text
    ]
    missing_groups = [group["accepted_values"] for group in groups if not group["matched"]]
    missing_contexts = [context["required_nearby"] for context in contexts if not context["matched"]]
    docx_media_hashes = set(structure.get("media_hashes", {}).values())
    media_source_rows = []
    for source in doc.required_media_sources:
        source_path = ROOT / source
        exists = source_path.exists()
        source_hash = sha256(source_path) if exists else None
        matched = bool(source_hash and source_hash in docx_media_hashes)
        media_source_rows.append(
            {
                "source": source,
                "exists": exists,
                "sha256": source_hash,
                "matched_docx_media": matched,
            }
        )
    missing_media_sources = [row["source"] for row in media_source_rows if not row["matched_docx_media"]]
    failure_reasons = []
    if missing_groups:
        failure_reasons.append("missing_required_value")
    if missing_contexts:
        failure_reasons.append("missing_required_context")
    if forbidden:
        failure_reasons.append("forbidden_stale_value")
    if claim_error:
        failure_reasons.append("docx_claim_audit_error")
    if xml_text_error:
        failure_reasons.append("docx_xml_text_error")
    if claim_blockers:
        failure_reasons.append("docx_claim_blockers")
    if structure_error:
        failure_reasons.append("docx_structure_error")
    if structure["media_count"] < doc.min_media_count:
        failure_reasons.append("missing_required_docx_media")
    if missing_media_sources:
        failure_reasons.append("missing_required_media_source")
    if structure["table_count"] < doc.min_table_count:
        failure_reasons.append("missing_required_docx_tables")
    if ranked_table_error:
        failure_reasons.append("docx_ranked_table_error")
    if doc.ranked_table_expected_journals and not ranked_table_check.get("ok"):
        failure_reasons.append("missing_required_ranked_table")
    if not docx_newer_or_equal_markdown:
        failure_reasons.append("stale_docx_export")
    if len(text) < doc.min_text_chars:
        failure_reasons.append("short_docx_text")
    if base["docx_hangul_count"] < doc.min_hangul_count:
        failure_reasons.append("insufficient_hangul_text")
    if base["docx_replacement_char_count"] > doc.max_replacement_chars:
        failure_reasons.append("replacement_chars_in_docx_text")
    base.update(
        {
            "required": groups,
            "required_contexts": contexts,
            "missing_groups": missing_groups,
            "missing_contexts": missing_contexts,
            "media_source_matches": media_source_rows,
            "missing_media_sources": missing_media_sources,
            "forbidden_hits": forbidden,
            "claim_audit": {
                "status": "claim_audit_error" if claim_error else ("blocked_claims_found" if claim_blockers else "passed"),
                "error": claim_error,
                "finding_count": len(claim_findings),
                "blocker_count": len(claim_blockers),
                "review_count": len(claim_review),
                "findings": claim_findings,
            },
            "failure_reasons": failure_reasons,
        }
    )
    if failure_reasons:
        base["status"] = "failed_checks"
    return base


def build_docs() -> list[WordDoc]:
    market_groups = [
        ["2,067,411"],
        ["411,423"],
        ["69"],
        ["473"],
        ["2.37%"],
        ["3/5"],
        ["96.9%"],
        ["0.9049"],
        ["0.1482"],
        ["7.45"],
        ["0.8956"],
        ["0.1531"],
        ["6.81"],
        ["7.38"],
        ["Merton"],
        ["Munoz-Izquierdo"],
        ["MacKinlay"],
        ["Kothari"],
        ["Chava and Jarrow"],
        ["10,575"],
        ["0.867"],
        ["0.200"],
        ["12.0%"],
        ["5.5%"],
    ]
    return [
        WordDoc(
            "English final Word manuscript",
            "manuscript/paper4_v4_manuscript_en.md",
            "manuscript/paper4_v4_manuscript_en.docx",
            market_groups,
            [["473", "2.37%", "3/5"]],
            min_media_count=8,
            min_table_count=7,
            required_media_sources=EN_SUBMISSION_FIGURES,
        ),
        WordDoc(
            "Korean final Word manuscript",
            "manuscript/paper4_v4_manuscript_ko.md",
            "manuscript/paper4_v4_manuscript_ko.docx",
            market_groups,
            [["473", "2.37%", "3/5"]],
            min_media_count=5,
            min_table_count=5,
            required_media_sources=(
                "figures/paper4_framework.png",
                "figures/paper4_model_performance.png",
                "figures/paper4_market_coverage.png",
                "figures/en/paper4_disclosure_price_reaction_en.png",
                "figures/en/paper4_llm_context_ablation_en.png",
            ),
            min_hangul_count=5000,
        ),
        WordDoc(
            "Title-page Word draft",
            "manuscript/paper4_v4_title_page.md",
            "manuscript/paper4_v4_title_page.docx",
            [
                ["Title Page"],
                ["Regulatory Disclosure Sequences and Corporate Distress Risk"],
                ["title page only"],
                ["paper4_v4_manuscript_en.docx"],
                ["Author information"],
            ],
            [["full English manuscript", "paper4_v4_manuscript_en.md"]],
            min_text_chars=250,
        ),
        WordDoc(
            "Korean submission candidate journal table",
            "manuscript/submission_candidate_journals_ko.md",
            "manuscript/submission_candidate_journals_ko.docx",
            [
                ["Expert Systems with Applications"],
                ["Knowledge-Based Systems"],
                ["Decision Support Systems"],
                ["Information Sciences"],
                ["Annals of Operations Research"],
                ["Computational Economics"],
                ["Asia-Pacific Financial Markets"],
                ["Review of Quantitative Finance and Accounting"],
                ["International Journal of Finance & Economics"],
                ["Applied Soft Computing"],
                ["10.1016/j.eswa.2025.127832"],
                ["10.1016/j.knosys.2026.115790"],
                ["10.1016/j.dss.2026.114615"],
                ["10.1016/j.asoc.2026.115085"],
                ["APC"],
                ["subscription"],
                ["2026-07-01"],
                ["2025-07-01"],
            ],
            [
                ["2025-07-01", "DOI"],
                ["Expert Systems with Applications", "10.1016/j.eswa.2025.127832"],
                ["Applied Soft Computing", "10.1016/j.asoc.2026.115085"],
                ["APC", "subscription"],
            ],
            min_table_count=2,
            min_hangul_count=500,
            ranked_table_expected_journals=(
                "Expert Systems with Applications",
                "Knowledge-Based Systems",
                "Decision Support Systems",
                "Information Sciences",
                "Annals of Operations Research",
                "Computational Economics",
                "Asia-Pacific Financial Markets",
                "Review of Quantitative Finance and Accounting",
                "International Journal of Finance & Economics",
                "Applied Soft Computing",
            ),
        ),
    ]


def build_audit() -> dict[str, Any]:
    docs = [evaluate_doc(doc) for doc in build_docs()]
    failures = [doc for doc in docs if doc["status"] != "passed"]
    claim_finding_count = sum(doc.get("claim_audit", {}).get("finding_count", 0) for doc in docs)
    claim_blocker_count = sum(doc.get("claim_audit", {}).get("blocker_count", 0) for doc in docs)
    claim_review_count = sum(doc.get("claim_audit", {}).get("review_count", 0) for doc in docs)
    en_md = ROOT / "manuscript/paper4_v4_manuscript_en.md"
    draft_md = ROOT / "manuscript/paper4_v4_title_page.md"
    draft_sync = {
        "status": "passed",
        "mode": "title_page_only",
        "english_markdown": str(en_md),
        "draft_markdown": str(draft_md),
        "english_markdown_exists": en_md.exists(),
        "draft_markdown_exists": draft_md.exists(),
    }
    if en_md.exists() and draft_md.exists():
        draft_text = draft_md.read_text(encoding="utf-8")
        draft_sync.update(
            {
                "english_sha256": sha256(en_md),
                "draft_sha256": sha256(draft_md),
                "same_content": sha256(en_md) == sha256(draft_md),
                "title_page_only": (
                    "Title Page" in draft_text
                    and "title page only" in draft_text
                    and "paper4_v4_manuscript_en.md" in draft_text
                    and "paper4_v4_manuscript_en.docx" in draft_text
                ),
            }
        )
        if not draft_sync["title_page_only"]:
            draft_sync["status"] = "title_page_missing"
    else:
        draft_sync["status"] = "missing_file"

    status = "pass" if not failures and draft_sync["status"] == "passed" else "fail"
    return {
        "status": status,
        "document_count": len(docs),
        "failure_count": len(failures) + (0 if draft_sync["status"] == "passed" else 1),
        "documents": docs,
        "draft_sync": draft_sync,
        "claim_finding_count": claim_finding_count,
        "claim_blocker_count": claim_blocker_count,
        "claim_review_count": claim_review_count,
        "forbidden_token_count": len(FORBIDDEN_TOKENS),
        "forbidden_token_policy": "src/audit/audit_forbidden_tokens.py",
        "guardrail": "Word exports must contain the current final manuscript metrics and reference tokens, preserve manuscript claim boundaries, keep the draft as a title-page-only artifact that points to the full English manuscript, verify Korean DOCX text is not mojibake/replacement-char output, and preserve the ranked 10-row Korean submission-candidate journal table as a reproducible DOCX artifact.",
    }


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# Word Export Consistency Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Word documents checked | {audit['document_count']} |",
        f"| Failures | {audit['failure_count']} |",
        f"| DOCX claim findings | {audit['claim_finding_count']} |",
        f"| DOCX claim blockers | {audit['claim_blocker_count']} |",
        f"| DOCX claim review items | {audit['claim_review_count']} |",
        f"| Forbidden stale/marker patterns | {audit['forbidden_token_count']} |",
        f"| Forbidden-token policy | `{audit['forbidden_token_policy']}` |",
        f"| Draft title page only | {audit['draft_sync'].get('title_page_only', False)} |",
        "",
        "## Document Checks",
        "",
        "| Document | Status | Failure reasons | Text chars | Hangul | Replacement chars | Media | Source media | Tables | Ranked rows | Fresh | Missing required values | Missing required context | Forbidden stale values | Claim blockers | Claim review | Evidence lines |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|---|",
    ]
    for doc in audit["documents"]:
        missing = "; ".join(", ".join(group) for group in doc.get("missing_groups", [])) or "-"
        missing_context = "; ".join(", ".join(group) for group in doc.get("missing_contexts", [])) or "-"
        forbidden = "; ".join(hit["token"] for hit in doc.get("forbidden_hits", [])) or "-"
        source_media_total = len(doc.get("required_media_sources", []))
        source_media_matched = source_media_total - len(doc.get("missing_media_sources", []))
        claim_audit = doc.get("claim_audit", {})
        claim_blockers = claim_audit.get("blocker_count", 0)
        claim_review = claim_audit.get("review_count", 0)
        evidence_lines = sorted(
            {
                line
                for group in doc.get("required", [])
                for line in group.get("line_numbers", [])
            }
            | {
                line
                for context in doc.get("required_contexts", [])
                for line in context.get("line_numbers", [])
            }
        )
        evidence = ", ".join(str(line) for line in evidence_lines[:16]) if evidence_lines else "-"
        if len(evidence_lines) > 16:
            evidence += ", ..."
        ranked_expected = doc.get("ranked_table_expected_rows", 0)
        ranked_rows = (
            f"{doc.get('ranked_table_matched_rows', 0)}/{ranked_expected}"
            if ranked_expected
            else "-"
        )
        lines.append(
            "| "
            f"`{escape_md(doc['docx'])}` | "
            f"`{doc['status']}` | "
            f"{escape_md(', '.join(doc.get('failure_reasons', [])) or '-')} | "
            f"{doc.get('docx_text_chars', 0)} | "
            f"{doc.get('docx_hangul_count', 0)}/{doc.get('min_hangul_count', 0)} | "
            f"{doc.get('docx_replacement_char_count', 0)}/{doc.get('max_replacement_chars', 0)} | "
            f"{doc.get('docx_media_count', 0)}/{doc.get('min_media_count', 0)} | "
            f"{source_media_matched}/{source_media_total} | "
            f"{doc.get('docx_table_count', 0)}/{doc.get('min_table_count', 0)} | "
            f"{ranked_rows} | "
            f"{doc.get('docx_newer_or_equal_markdown', False)} | "
            f"{escape_md(missing)} | "
            f"{escape_md(missing_context)} | "
            f"{escape_md(forbidden)} | "
            f"{claim_blockers} | "
            f"{claim_review} | "
            f"{escape_md(evidence)} |"
        )
    lines.extend(
        [
            "",
            "## Draft Title Page",
            "",
            f"Status: `{audit['draft_sync']['status']}`",
            "",
            "## Guardrail",
            "",
            audit["guardrail"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    (RESULTS / "word_export_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(audit, RESULTS / "word_export_audit.md")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "document_count": audit["document_count"],
                "failure_count": audit["failure_count"],
            },
            indent=2,
        )
    )
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
