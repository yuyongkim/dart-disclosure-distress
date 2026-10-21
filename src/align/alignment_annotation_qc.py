from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path("Paper 4")
VALID_LABELS = {"MATCH", "PARTIAL", "NO_MATCH"}


def norm_label(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    return text if text in VALID_LABELS else ""


def confidence_bucket(value: object) -> str:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return "missing"
    if conf >= 0.75:
        return "high"
    if conf >= 0.65:
        return "medium"
    return "low"


def gold_label_source(row: pd.Series) -> str:
    adjudicated = row.get("adjudicated_label_norm", "")
    legacy = row.get("human_label_norm", "")
    ann1 = row.get("annotator1_label_norm", "")
    ann2 = row.get("annotator2_label_norm", "")
    if adjudicated:
        return "adjudicated_label"
    if legacy:
        return "human_label"
    if ann1 and ann2 and ann1 == ann2:
        return "annotator_consensus"
    if ann1 and ann2 and ann1 != ann2:
        return "needs_adjudication"
    if ann1:
        return "annotator1_only"
    if ann2:
        return "annotator2_only"
    return "none"


def build_priority(seed_path: Path, llm_path: Path, out_dir: Path) -> dict:
    seed = pd.read_csv(seed_path, low_memory=False)
    if llm_path.exists():
        llm = pd.read_csv(llm_path, low_memory=False)
        llm_cols = [
            "candidate_rank",
            "llm_decision",
            "llm_confidence",
            "llm_rationale",
            "llm_mode",
            "lexical_overlap",
            "api_status",
        ]
        llm = llm[[c for c in llm_cols if c in llm.columns]].drop_duplicates("candidate_rank")
        df = seed.merge(llm, on="candidate_rank", how="left")
    else:
        df = seed.copy()
        df["llm_decision"] = ""
        df["llm_confidence"] = ""
        df["llm_rationale"] = ""
        df["llm_mode"] = ""
        df["lexical_overlap"] = ""
        df["api_status"] = ""

    for optional_col in ["annotator1_label", "annotator2_label", "adjudicated_label", "human_label", "human_notes", "adjudication_notes"]:
        if optional_col not in df.columns:
            df[optional_col] = ""

    df["proposed_label_norm"] = df["proposed_label"].map(norm_label)
    df["ai_suggested_label"] = df["llm_decision"].map(norm_label)
    df["annotator1_label_norm"] = df["annotator1_label"].map(norm_label)
    df["annotator2_label_norm"] = df["annotator2_label"].map(norm_label)
    df["adjudicated_label_norm"] = df["adjudicated_label"].map(norm_label)
    df["human_label_norm"] = df["human_label"].map(norm_label)
    df["gold_label_source"] = df.apply(gold_label_source, axis=1)
    df["ai_confidence"] = pd.to_numeric(df.get("llm_confidence", ""), errors="coerce")
    df["confidence_bucket"] = df["ai_confidence"].map(confidence_bucket)
    df["proposed_ai_agree"] = df["proposed_label_norm"].eq(df["ai_suggested_label"])
    df["is_human_labeled"] = ~df["gold_label_source"].isin(["none", "needs_adjudication", "annotator1_only", "annotator2_only"])

    priority_rows = []
    for row in df.itertuples(index=False):
        reasons: list[str] = []
        score = 0
        if not row.is_human_labeled:
            score += 50
        if getattr(row, "gold_label_source") == "needs_adjudication":
            score += 40
            reasons.append("annotator_disagreement_needs_adjudication")
        if not row.proposed_ai_agree:
            score += 30
            reasons.append("proposed_vs_ai_disagreement")
        if row.confidence_bucket in {"low", "missing"}:
            score += 20
            reasons.append(f"ai_confidence_{row.confidence_bucket}")
        if getattr(row, "proposed_label_norm") == "MATCH" and getattr(row, "ai_suggested_label") != "MATCH":
            score += 10
            reasons.append("strict_match_requires_review")
        if getattr(row, "taxonomy_code", "") in {"T05", "T06"}:
            score += 5
            reasons.append("broad_taxonomy_cell")
        if not reasons:
            reasons.append("routine_confirmation")
        priority_rows.append((score, "; ".join(reasons)))

    df["review_priority_score"] = [row[0] for row in priority_rows]
    df["review_reason"] = [row[1] for row in priority_rows]
    df = df.sort_values(["review_priority_score", "candidate_rank"], ascending=[False, True])

    keep = [
        "candidate_rank",
        "review_priority_score",
        "review_reason",
        "taxonomy_code",
        "dart_event",
        "dart_subtype",
        "sec_event",
        "sec_subtype",
        "proposed_label",
        "ai_suggested_label",
        "ai_confidence",
        "confidence_bucket",
        "proposed_ai_agree",
        "gold_label_source",
        "annotator1_label",
        "annotator2_label",
        "adjudicated_label",
        "human_label",
        "human_notes",
        "adjudication_notes",
        "llm_rationale",
        "rationale",
        "dart_event_count",
        "sec_event_count",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = ""
    out_dir.mkdir(parents=True, exist_ok=True)
    priority_csv = out_dir / "manual_alignment_seed_100_annotation_priority.csv"
    first_pass_csv = out_dir / "manual_alignment_seed_100_first_pass_30.csv"
    df[keep].to_csv(priority_csv, index=False, encoding="utf-8-sig")
    df[keep].head(30).to_csv(first_pass_csv, index=False, encoding="utf-8-sig")

    summary = {
        "status": "created",
        "candidate_rows": int(len(df)),
        "human_labeled_rows": int(df["is_human_labeled"].sum()),
        "unlabeled_rows": int((~df["is_human_labeled"]).sum()),
        "annotator1_labeled_rows": int(df["annotator1_label_norm"].ne("").sum()),
        "annotator2_labeled_rows": int(df["annotator2_label_norm"].ne("").sum()),
        "adjudicated_labeled_rows": int(df["adjudicated_label_norm"].ne("").sum()),
        "needs_adjudication_rows": int(df["gold_label_source"].eq("needs_adjudication").sum()),
        "proposed_ai_disagreements": int((~df["proposed_ai_agree"]).sum()),
        "low_or_missing_confidence_rows": int(df["confidence_bucket"].isin(["low", "missing"]).sum()),
        "first_pass_rows": int(min(30, len(df))),
        "priority_csv": str(priority_csv),
        "first_pass_csv": str(first_pass_csv),
        "guardrail": "QC prioritizes human review only; it is not a substitute for human labels.",
    }
    write_markdown(summary, df, ROOT / "results" / "alignment_annotation_qc.md")
    (ROOT / "results" / "alignment_annotation_qc.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def write_markdown(summary: dict, df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Alignment Annotation QC",
        "",
        "This file prioritizes the 100 DART-SEC candidate pairs for human review. It does not create gold labels.",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Candidate rows | {summary['candidate_rows']} |",
        f"| Gold-labeled rows | {summary['human_labeled_rows']} |",
        f"| Unlabeled rows | {summary['unlabeled_rows']} |",
        f"| Annotator 1 labeled rows | {summary['annotator1_labeled_rows']} |",
        f"| Annotator 2 labeled rows | {summary['annotator2_labeled_rows']} |",
        f"| Adjudicated labeled rows | {summary['adjudicated_labeled_rows']} |",
        f"| Rows needing adjudication | {summary['needs_adjudication_rows']} |",
        f"| Proposed-vs-AI disagreements | {summary['proposed_ai_disagreements']} |",
        f"| Low/missing confidence rows | {summary['low_or_missing_confidence_rows']} |",
        f"| First-pass priority rows | {summary['first_pass_rows']} |",
        "",
        "## First-Pass Review Rows",
        "",
        "| Rank | Priority | Reason | Proposed | AI | Confidence | DART event | SEC event |",
        "|---:|---:|---|---|---|---:|---|---|",
    ]
    for row in df.head(30).itertuples(index=False):
        lines.append(
            "| "
            f"{row.candidate_rank} | "
            f"{row.review_priority_score} | "
            f"{escape(row.review_reason)} | "
            f"`{escape(row.proposed_label)}` | "
            f"`{escape(row.ai_suggested_label)}` | "
            f"{format_conf(row.ai_confidence)} | "
            f"{escape(row.dart_event)} | "
            f"{escape(row.sec_event)} |"
        )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "Use this file to decide which rows to label first. Do not report Precision@1 or MRR until `human_label` is filled and `evaluate_alignment.py` is rerun.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def format_conf(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-path", default=str(ROOT / "data" / "processed" / "manual_alignment_seed_100.csv"))
    parser.add_argument("--llm-path", default=str(ROOT / "data" / "processed" / "llm_alignment_results.csv"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed"))
    args = parser.parse_args()
    result = build_priority(Path(args.seed_path), Path(args.llm_path), Path(args.out_dir))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
