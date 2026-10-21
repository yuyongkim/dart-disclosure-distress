from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")

DEFAULT_MANUSCRIPT_PATHS = [
    ROOT / "manuscript" / "paper4_v4_manuscript_en.md",
    ROOT / "manuscript" / "paper4_v4_manuscript_ko.md",
    ROOT / "manuscript" / "paper4_v4_title_page.md",
    ROOT / "manuscript" / "submission_strategy.md",
    ROOT / "manuscript" / "submission_checklist_v1.md",
    ROOT / "manuscript" / "claim_readiness_matrix.md",
    ROOT / "manuscript" / "paper4_target_journal_expansion_ko.md",
    ROOT / "manuscript" / "submission_candidate_journals_ko.md",
    ROOT / "manuscript" / "cover_letter_v1_generic.md",
    ROOT / "manuscript" / "alignment_annotation_guide.md",
    ROOT / "manuscript" / "alignment_single_expert_labeling_guide.md",
    ROOT / "manuscript" / "alignment_sec_desktop_labeling_guide.md",
    ROOT / "results" / "sec_companyfacts_availability.md",
    ROOT / "results" / "sec_filing_body_text_baseline_summary.md",
    ROOT / "results" / "sec_all8k_sample_text_baseline_summary.md",
    ROOT / "results" / "sec_validation_scope.md",
    ROOT / "VERSIONS.md",
    ROOT / "REPRODUCIBILITY.md",
    ROOT / "data_map.md",
    ROOT / "paper4_execution_status.md",
]


@dataclass(frozen=True)
class ClaimRule:
    name: str
    pattern: str
    allowed_context: tuple[str, ...]
    severity: str
    guidance: str


RULES = [
    ClaimRule(
        name="alignment_accuracy_claim",
        pattern=(
            r"\b(Precision@1|MRR|alignment accuracy|validated cross-lingual benchmark)\b"
            r"|(\uc815\ub82c\s*\uc815\ud655\ub3c4|\uad50\ucc28\uc5b8\uc5b4\s*\ubca4\uce58\ub9c8\ud06c|\uace8\ub4dc\s*\ud45c\uc900)"
        ),
        allowed_context=(
            "not yet",
            "not reported",
            "not used",
            "should not claim",
            "should only be reported after",
            "require human labels",
            "requires human labels",
            "only after labels are filled",
            "after labels are filled",
            "until human labels are filled",
            "until those human labels",
            "after human_label",
            "after `human_label`",
            "`human_label` is filled",
            "filled and synced",
            "single-expert-gold caveat",
            "blocks",
            "before use",
            "before reporting",
            "triage evidence only",
            "annotation readiness only",
            "not precision@1/mrr evidence",
            "excluded from final",
            "after gold labels are available",
            "unavailable until",
            "all remain unavailable",
            "does not",
            "once",
            "after human labels",
            "after those labels",
            "human labels 입력 후",
            "채워진 뒤",
            "뒤에만",
            "입력 후",
            "보고하지 않음",
            "coverage and annotation readiness",
            "annotation-ready",
            "blank labels",
            "requires",
            "do not",
            "not ready",
            "\uc778\uac04 \ub77c\ubca8",
            "\uc778\uac04 \uc8fc\uc11d",
            "\ub77c\ubca8\uc774 \uc785\ub825",
            "\ub3d9\uae30\ud654",
            "\uc774\ud6c4\uc5d0\ub9cc",
            "\uc544\uc9c1",
            "\uc544\ub2c8",
            "\uc9c0\uc6d0\ud558\uace0",
            "\ud544\uc694",
            "\uacfc\uc7a5\ud558\uc9c0",
            "\uc900\ube44\uc0c1\ud0dc",
            "\ubcf4\uc870 \uc0b0\ucd9c\ubb3c",
        ),
        severity="blocker",
        guidance="Do not claim DART-SEC Precision@1/MRR or validated benchmark status until human labels are filled and synced.",
    ),
    ClaimRule(
        name="inter_annotator_claim",
        pattern=r"\b(Cohen'?s kappa|inter-annotator agreement|inter-annotator reliability)\b",
        allowed_context=(
            "requires",
            "additionally requires",
            "do not report",
            "until those human labels",
            "until human labels",
            "filled, synced, and evaluated",
            "only after both annotator columns",
            "after both annotator columns",
            "cannot support inter-annotator agreement",
            "but not",
            "not cohen",
            "채워진 뒤",
            "뒤에만",
            "보고하지 않음",
            "when two annotators are available",
            "until the second label",
            "before reporting",
            "after gold labels are available",
            "still needs human labels",
            "unavailable until",
            "all remain unavailable",
            "does not",
            "cannot support",
            "do not claim",
            "not ready",
            "two-annotator",
            "if independent labels are later collected",
            "\ud544\uc694",
            "\ub3c5\ub9bd \uc8fc\uc11d\uc790",
            "\uc870\uc815 \ub77c\ubca8",
            "\uc774\ud6c4",
            "\uacc4\uc0b0",
        ),
        severity="blocker",
        guidance="Do not claim Cohen's kappa or inter-annotator agreement until both annotator columns are filled.",
    ),
    ClaimRule(
        name="edinet_results_claim",
        pattern=r"\b(EDINET|Japanese disclosure|Japan(?:ese)? disclosure)\b|(\uc77c\ubcf8\s*\uacf5\uc2dc)",
        allowed_context=(
            "future work",
            "v2 extension",
            "should not claim",
            "does not redefine",
            "does not mark",
            "not full completion",
            "not complete",
            "remain outside",
            "no collected",
            "no actual",
            "no reusable",
            "no-key",
            "collection plan",
            "collection path",
            "collector",
            "requires",
            "retained as",
            "not ready",
            "api",
            "v2",
            "phase 2",
            "planned",
            "planned registry",
            "registry entry",
            "parser-prototype",
            "preflight",
            "blocker",
            "blockers",
            "prevents",
            "no edinet requests",
            "empirical collection ready",
            "do not provide edinet",
            "current-report empirical evidence",
            "reproducible",
            "\ud5a5\ud6c4",
            "\uc218\uc9d1 \uacc4\ud68d",
            "\uc218\uc9d1\ub41c",
            "\uc218\uc9d1\ub41c edinet",
            "\uc5c6",
            "\uc544\uc9c1",
            "\ud544\uc694",
            "\ubcf4\uc874",
            "\ud655\uc7a5",
            "\uc544\uce74\uc774\ube0c",
        ),
        severity="blocker",
        guidance="Do not claim EDINET/Japanese empirical results until actual EDINET rows are collected and normalized.",
    ),
    ClaimRule(
        name="sec_full_validation_claim",
        pattern=r"\b(standalone U\.?S\.? validation|finalized U\.?S\.? predictive validation|full all-8-K|all-8-K text validation|full all-8-K text validation)\b",
        allowed_context=(
            "not",
            "not yet",
            "do not",
            "outside",
            "v2",
            "future",
            "scope is not yet complete",
            "not full",
            "not standalone",
            "missing",
            "remain",
            "remains",
            "separate",
            "not ready",
            "status-only",
            "makes no sec archives requests",
            "inspect open batches",
            "\uc544\uc9c1",
            "\uc644\ub8cc\ub418\uc9c0",
            "\ud575\uc2ec \uc8fc\uc7a5\uc774 \uc544\ub2c8",
            "\ub0a8\uc544",
            "\ubc16\uc5d0",
            "\ud5a5\ud6c4",
            "v2",
            "전에는",
            "금지",
        ),
        severity="blocker",
        guidance="Do not claim finalized SEC standalone validation or full all-8-K text validation before the full filing-body scope is complete.",
    ),
    ClaimRule(
        name="sec_sample_text_overclaim",
        pattern=r"\b(target-item-vs-non-target|bounded all-8-K text sample|sample-level sensitivity|ROC AUC 0\.8875|top-decile lift 6\.27)\b",
        allowed_context=(
            "sample-level",
            "sensitivity",
            "not standalone",
            "not future-distress",
            "not full",
            "bounded",
            "sample",
            "worthwhile",
            "\ud45c\ubcf8",
            "\uc0d8\ud50c",
            "\ubbfc\uac10\ub3c4",
            "\ub3c5\ub9bd",
            "\uc544\ub2c8",
            "\uc804\uccb4",
            "\uc81c\ud55c",
        ),
        severity="review",
        guidance="Bounded all-8-K sample text results must be described as sample-level sensitivity, not future-distress validation.",
    ),
    ClaimRule(
        name="exact_classic_score_claim",
        pattern=r"\b(exact Altman|exact Ohlson|Altman Z-score|Ohlson O-score)\b",
        allowed_context=(
            "does not claim",
            "do not claim",
            "not claim",
            "not computable",
            "available-component",
            "proxy",
            "proxies",
            "missing components",
            "calls for",
            "\uc8fc\uc7a5\ud558\uc9c0",
            "\uc815\ud655\ud55c",
            "\uc6d0 \uacf5\uc2dd",
            "\ud3ec\ud568\ub418\uc5b4 \uc788\uc9c0",
            "\ub300\uc6a9",
            "\ud544\uc694",
            "\uc694\uad6c",
        ),
        severity="blocker",
        guidance="Do not claim exact Altman/Ohlson scores; current evidence supports available-component proxies only.",
    ),
    ClaimRule(
        name="trilingual_claim",
        pattern=r"\b(trilingual|DART\+SEC\+EDINET|DART-SEC-EDINET)\b",
        allowed_context=(
            "research plan",
            "future",
            "v2",
            "not",
            "no collected",
            "edinet",
            "\uc5f0\uad6c\uacc4\ud68d",
            "\ud5a5\ud6c4",
            "\uc544\ub2c8",
            "\uc218\uc9d1\ub41c",
        ),
        severity="review",
        guidance="Do not frame v1 as a completed trilingual benchmark while EDINET rows are absent.",
    ),
]


def sentence_windows(text: str) -> list[tuple[int, str]]:
    lines = text.splitlines()
    windows: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        windows.append((idx, stripped))
    return windows


def contextual_windows(text: str) -> list[tuple[int, str, str]]:
    lines = text.splitlines()
    windows: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        start = idx
        while start > 0 and lines[start - 1].strip():
            start -= 1
        end = idx + 1
        while end < len(lines) and lines[end].strip():
            end += 1
        context = " ".join(item.strip() for item in lines[start:end] if item.strip())
        windows.append((idx + 1, stripped, context))
    return windows


def allowed(line: str, rule: ClaimRule) -> bool:
    lowered = line.lower()
    return any(term.lower() in lowered for term in rule.allowed_context)


def audit_text(label: str, text: str) -> list[dict]:
    findings: list[dict] = []
    for line_no, line, context in contextual_windows(text):
        for rule in RULES:
            if re.search(rule.pattern, line, flags=re.IGNORECASE) and not allowed(context, rule):
                findings.append(
                    {
                        "file": label,
                        "line": line_no,
                        "rule": rule.name,
                        "severity": rule.severity,
                        "text": line,
                        "guidance": rule.guidance,
                    }
                )
    return findings


def audit_file(path: Path) -> list[dict]:
    return audit_text(str(path), path.read_text(encoding="utf-8"))


def build_audit(paths: list[Path]) -> dict:
    findings: list[dict] = []
    for path in paths:
        if path.exists():
            findings.extend(audit_file(path))
    blockers = [row for row in findings if row["severity"] == "blocker"]
    review = [row for row in findings if row["severity"] != "blocker"]
    return {
        "status": "passed" if not blockers else "blocked_claims_found",
        "scanned_files": [str(path) for path in paths if path.exists()],
        "rules": [rule.name for rule in RULES],
        "finding_count": len(findings),
        "blocker_count": len(blockers),
        "review_count": len(review),
        "findings": findings,
        "guardrail": "The v1 manuscript, submission-support files, and operational readiness docs should not claim alignment accuracy, EDINET results, or full SEC all-8-K validation before the required evidence exists.",
    }


def write_markdown(
    audit: dict,
    path: Path,
    title: str = "Manuscript Claim Audit",
    clean_subject: str = "scanned claim-audited Markdown files",
) -> None:
    lines = [
        f"# {title}",
        "",
        f"Status: `{audit['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Scanned files | {len(audit['scanned_files'])} |",
        f"| Rules | {len(audit['rules'])} |",
        f"| Findings | {audit['finding_count']} |",
        f"| Blockers | {audit['blocker_count']} |",
        f"| Review items | {audit['review_count']} |",
    ]
    if "forbidden_hit_count" in audit:
        lines.append(f"| Forbidden text hits | {audit['forbidden_hit_count']} |")
    if "audit_scope" in audit:
        lines.append(f"| Audit scope | `{escape(audit['audit_scope'])}` |")
    if "post_claim_generated_files" in audit:
        lines.append(f"| Post-claim generated files | {len(audit['post_claim_generated_files'])} |")
    if "text_integrity_files" in audit:
        lines.append(f"| Text-integrity files | {len(audit['text_integrity_files'])} |")
    lines.append("")
    if audit["findings"]:
        lines.extend(
            [
                "## Findings",
                "",
                "| Severity | Rule | File | Line | Text | Guidance |",
                "|---|---|---|---:|---|---|",
            ]
        )
        for row in audit["findings"]:
            lines.append(
                "| "
                f"`{row['severity']}` | "
                f"`{row['rule']}` | "
                f"`{escape(Path(row['file']).name)}` | "
                f"{row['line']} | "
                f"{escape(row['text'])} | "
                f"{escape(row['guidance'])} |"
            )
    else:
        lines.append(f"No blocker claims found in the {len(audit['scanned_files'])} {clean_subject}.")
    if audit.get("forbidden_hits"):
        lines.extend(
            [
                "",
                "## Forbidden Text Hits",
                "",
                "| File | Token | Lines |",
                "|---|---|---|",
            ]
        )
        for row in audit["forbidden_hits"]:
            lines.append(
                "| "
                f"`{escape(Path(row['file']).name)}` | "
                f"`{escape(row['token'])}` | "
                f"{escape(', '.join(str(line) for line in row['line_numbers']))} |"
            )
    lines.extend(["", "## Guardrail", "", audit["guardrail"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths",
        nargs="*",
        default=[str(path) for path in DEFAULT_MANUSCRIPT_PATHS],
        help="Markdown manuscript/support files to audit. Defaults to v1 manuscripts plus submission, alignment, SEC claim-boundary, and operational readiness support files.",
    )
    parser.add_argument("--out-json", default=str(ROOT / "results" / "manuscript_claim_audit.json"))
    parser.add_argument("--out-md", default=str(ROOT / "results" / "manuscript_claim_audit.md"))
    args = parser.parse_args()
    audit = build_audit([Path(item) for item in args.paths])
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(audit, Path(args.out_md))
    print(json.dumps({k: audit[k] for k in ["status", "finding_count", "blocker_count", "review_count"]}, indent=2))
    if audit["blocker_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
