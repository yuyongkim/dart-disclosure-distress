from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class ReferenceCheck:
    group: str
    token: str
    manuscript_en: bool
    manuscript_ko: bool
    bibtex: bool

    @property
    def status(self) -> str:
        return "pass" if self.manuscript_en and self.manuscript_ko and self.bibtex else "fail"


KEY_REFERENCES = [
    ("hazard_market_distress", "Chava"),
    ("hazard_market_distress", "Hillegeist"),
    ("hazard_market_distress", "Merton"),
    ("korean_distress", "Eom"),
    ("korean_distress", "Nam, C"),
    ("ml_bankruptcy", "Barboza"),
    ("ml_bankruptcy", "Lessmann"),
    ("ml_bankruptcy", "Jones"),
    ("event_disclosure", "Carter"),
    ("event_disclosure", "Form 8-K"),
    ("event_study_boundary", "MacKinlay"),
    ("event_study_boundary", "Kothari"),
    ("audit_report_text", "Munoz"),
]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def contains(text: str, token: str) -> bool:
    return token.lower() in text.lower()


def write_markdown(checks: list[ReferenceCheck], path: Path) -> None:
    status = "pass" if all(check.status == "pass" for check in checks) else "fail"
    lines = [
        "# Reference Consistency Audit",
        "",
        f"Status: `{status}`",
        "",
        "| Group | Token | EN manuscript | KO manuscript | BibTeX | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check.group} | {check.token} | {check.manuscript_en} | {check.manuscript_ko} | {check.bibtex} | {check.status} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    en = read("manuscript/paper4_v4_manuscript_en.md")
    ko = read("manuscript/paper4_v4_manuscript_ko.md")
    bib = read("manuscript/references.bib")
    checks = [
        ReferenceCheck(
            group=group,
            token=token,
            manuscript_en=contains(en, token),
            manuscript_ko=contains(ko, token),
            bibtex=contains(bib, token),
        )
        for group, token in KEY_REFERENCES
    ]
    failures = [check.__dict__ for check in checks if check.status != "pass"]
    summary = {
        "status": "pass" if not failures else "fail",
        "checked_references": len(checks),
        "failures": failures,
        "outputs": {
            "markdown": str(RESULTS / "reference_consistency_audit.md"),
            "json": str(RESULTS / "reference_consistency_audit.json"),
        },
    }
    (RESULTS / "reference_consistency_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(checks, RESULTS / "reference_consistency_audit.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
