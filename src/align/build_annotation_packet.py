from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alignment_annotation_qc import build_priority


LABEL_DEFINITIONS = [
    ("MATCH", "The DART event and SEC event describe substantially the same legal/economic disclosure event."),
    ("PARTIAL", "The events are related or share a risk theme, but they are not strict equivalents."),
    ("NO_MATCH", "The events do not describe the same legal/economic disclosure event."),
]


def write_guide(seed: pd.DataFrame, guide_path: Path, packet_path: Path) -> None:
    lines = ["# DART-SEC Alignment Annotation Guide", ""]
    lines.append("Purpose: prepare independent annotator labels and adjudicated labels. Do not report RQ1 Precision@1, MRR, or inter-annotator agreement until those human labels are filled, synced, and evaluated.")
    lines.append("")
    lines.append("## Labels")
    lines.append("")
    lines.append("| Label | Use when |")
    lines.append("|---|---|")
    for label, description in LABEL_DEFINITIONS:
        lines.append(f"| `{label}` | {description} |")
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    lines.append("- Judge the legal/economic event, not the exact wording.")
    lines.append("- Prefer `MATCH` only when the events would plausibly belong to the same supervisory taxonomy cell and the same disclosure trigger.")
    lines.append("- Use `PARTIAL` when the events share a risk domain but differ in legal trigger, timing, or specificity.")
    lines.append("- Use `NO_MATCH` when the link exists only because both are broadly adverse or both are corporate events.")
    lines.append("- Do not mark deterministic taxonomy agreement as automatically correct; the point is to validate or reject that mapping.")
    lines.append("")
    lines.append("## Recommended Workflow")
    lines.append("")
    lines.append("1. Run `python \"Paper 4\\src\\align\\alignment_annotation_qc.py\"` to create priority files.")
    lines.append("2. Run `python \"Paper 4\\src\\align\\alignment_candidate_diagnostics.py\"` to create pre-label ambiguity diagnostics.")
    lines.append("3. For the fast single-expert route, run `python \"Paper 4\\src\\align\\build_single_expert_label_sheet.py\"`, fill `human_label`, then run `python \"Paper 4\\src\\align\\sync_single_expert_label_sheet.py\"`.")
    lines.append("4. For the two-annotator route, run `python \"Paper 4\\src\\align\\build_annotation_workbook.py\"`.")
    lines.append("5. Open `data/processed/manual_alignment_seed_100_annotation_workbook.xlsx`.")
    lines.append("6. Sort by `review_priority_score` or use `manual_alignment_seed_100_first_pass_30.csv` for the first pass.")
    lines.append("7. Fill `annotator1_label` and `annotator2_label` independently with `MATCH`, `PARTIAL`, or `NO_MATCH` in the `annotation` sheet.")
    lines.append("8. If the two annotators disagree, resolve the row in `adjudicated_label` and add a short note in `adjudication_notes`.")
    lines.append("9. Run `python \"Paper 4\\src\\align\\sync_annotation_workbook.py\"` to copy two-annotator labels back to `manual_alignment_seed_100.csv` and rerun evaluation.")
    lines.append("10. Use `results/alignment_evaluation.md` only after labeled rows are nonzero.")
    lines.append("")
    lines.append("AI suggested labels in the workbook are review aids only and must not be reported as human labels. After `human_label` is filled and synced, single-expert labels can support Precision@1/MRR with a single-expert-gold caveat, but not Cohen's kappa. Rows where only one annotator has labeled in the two-annotator workbook, or where annotators disagree, are excluded from final two-annotator RQ1 metrics until the second label or `adjudicated_label` is filled.")
    lines.append("")
    lines.append("## First 30 Candidate Pairs")
    lines.append("")
    lines.append("| Rank | Priority | Proposed | Taxonomy | DART event | SEC event | Review reason |")
    lines.append("|---:|---:|---|---|---|---|---|")
    for row in seed.head(30).itertuples(index=False):
        priority = getattr(row, "review_priority_score", "")
        reason = getattr(row, "review_reason", "")
        lines.append(
            f"| {row.candidate_rank} | {priority} | `{row.proposed_label}` | `{row.taxonomy_code}` | {escape(row.dart_event)} | {escape(row.sec_event)} | {escape(reason)} |"
        )
    guide_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    packet = seed.copy()
    packet["annotation_instruction"] = "Fill annotator1_label and annotator2_label independently; use adjudicated_label for disagreements."
    packet.to_csv(packet_path, index=False, encoding="utf-8-sig")


def escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-path", default=str(Path("Paper 4") / "data" / "processed" / "manual_alignment_seed_100.csv"))
    parser.add_argument("--guide-path", default=str(Path("Paper 4") / "manuscript" / "alignment_annotation_guide.md"))
    parser.add_argument("--packet-path", default=str(Path("Paper 4") / "data" / "processed" / "manual_alignment_seed_100_annotation_packet.csv"))
    args = parser.parse_args()

    seed_path = Path(args.seed_path)
    out_dir = seed_path.parent
    llm_path = out_dir / "llm_alignment_results.csv"
    priority_path = out_dir / "manual_alignment_seed_100_annotation_priority.csv"
    build_priority(seed_path, llm_path, out_dir)
    source_path = priority_path if priority_path.exists() else seed_path
    seed = pd.read_csv(source_path, low_memory=False)
    write_guide(seed, Path(args.guide_path), Path(args.packet_path))
    print(f"wrote {args.guide_path}")
    print(f"wrote {args.packet_path}")


if __name__ == "__main__":
    main()
