from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_alignment import VALID_LABELS, evaluate, normalize_label


ROOT = Path("Paper 4")
SHEET_NAME = "single_expert_labels"


def read_label_sheet(label_path: Path, sheet_name: str) -> pd.DataFrame:
    if label_path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        labels = pd.read_excel(label_path, sheet_name=sheet_name)
    else:
        labels = pd.read_csv(label_path, low_memory=False)
    required = {"candidate_rank", "human_label"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Single-expert label sheet is missing columns: {sorted(missing)}")
    if "human_notes" not in labels.columns:
        labels["human_notes"] = ""
    if "expert_confidence" not in labels.columns:
        labels["expert_confidence"] = ""
    labels = labels[["candidate_rank", "human_label", "human_notes", "expert_confidence"]].copy()
    labels["candidate_rank"] = pd.to_numeric(labels["candidate_rank"], errors="coerce").astype("Int64")
    labels = labels.dropna(subset=["candidate_rank"]).drop_duplicates("candidate_rank", keep="last")
    labels["human_label_norm"] = labels["human_label"].map(normalize_label)
    invalid = labels[
        labels["human_label"].notna()
        & labels["human_label"].astype(str).str.strip().ne("")
        & labels["human_label_norm"].eq("")
    ]
    if not invalid.empty:
        ranks = invalid["candidate_rank"].dropna().astype(int).tolist()
        raise ValueError(f"Invalid single-expert labels found at candidate_rank values: {ranks}")
    labels["human_label"] = labels["human_label_norm"].fillna("")
    labels["human_notes"] = labels["human_notes"].fillna("").astype(str)
    labels["expert_confidence"] = labels["expert_confidence"].fillna("").astype(str)
    return labels[["candidate_rank", "human_label", "human_notes", "expert_confidence"]]


def sync_labels(label_path: Path, seed_path: Path, output_path: Path, sheet_name: str, evaluate_after: bool) -> dict:
    labels = read_label_sheet(label_path, sheet_name)
    seed = pd.read_csv(seed_path, low_memory=False).copy()
    for col in ["human_label", "human_notes", "expert_confidence"]:
        if col not in seed.columns:
            seed[col] = ""
        seed[col] = seed[col].fillna("").astype(str)

    merged = seed.merge(labels, on="candidate_rank", how="left", suffixes=("", "_sheet"))
    nonblank_label = merged["human_label_sheet"].fillna("").astype(str).str.strip().ne("")
    nonblank_notes = merged["human_notes_sheet"].fillna("").astype(str).str.strip().ne("")
    nonblank_confidence = merged["expert_confidence_sheet"].fillna("").astype(str).str.strip().ne("")
    merged.loc[nonblank_label, "human_label"] = merged.loc[nonblank_label, "human_label_sheet"]
    merged.loc[nonblank_notes, "human_notes"] = merged.loc[nonblank_notes, "human_notes_sheet"]
    merged.loc[nonblank_confidence, "expert_confidence"] = merged.loc[nonblank_confidence, "expert_confidence_sheet"]
    merged = merged.drop(columns=[col for col in merged.columns if col.endswith("_sheet")])

    for col in ["annotator1_label", "annotator2_label", "adjudicated_label", "adjudication_notes"]:
        if col not in merged.columns:
            merged[col] = ""
        merged[col] = merged[col].fillna("").astype(str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    labeled_rows = int(merged["human_label"].map(normalize_label).ne("").sum())
    result = {
        "label_path": str(label_path),
        "seed_path": str(seed_path),
        "output_path": str(output_path),
        "candidate_rows": int(len(merged)),
        "sheet_nonblank_labels": int(nonblank_label.sum()),
        "sheet_nonblank_notes": int(nonblank_notes.sum()),
        "human_labeled_rows_after_sync": labeled_rows,
        "human_unlabeled_rows_after_sync": int(len(merged) - labeled_rows),
        "evaluated": False,
        "guardrail": "Single-expert labels support Precision@1/MRR only as single-expert gold labels; they do not support inter-annotator agreement.",
    }
    if evaluate_after:
        eval_result = evaluate(
            output_path,
            ROOT / "results" / "alignment_evaluation.json",
            ROOT / "results" / "alignment_evaluation.md",
        )
        result["evaluated"] = True
        result["evaluation_status"] = eval_result["status"]
        result["precision_at_1_match_only"] = eval_result["precision_at_1_match_only"]
        result["precision_at_1_match_or_partial"] = eval_result["precision_at_1_match_or_partial"]
        result["mrr_binary_match"] = eval_result["mrr_binary_match"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-path", default=str(ROOT / "data" / "processed" / "manual_alignment_seed_100_single_expert_label_sheet.xlsx"))
    parser.add_argument("--sheet-name", default=SHEET_NAME)
    parser.add_argument("--seed-path", default=str(ROOT / "data" / "processed" / "manual_alignment_seed_100.csv"))
    parser.add_argument("--output-path", default=str(ROOT / "data" / "processed" / "manual_alignment_seed_100.csv"))
    parser.add_argument("--no-evaluate", action="store_true")
    args = parser.parse_args()
    result = sync_labels(
        Path(args.label_path),
        Path(args.seed_path),
        Path(args.output_path),
        args.sheet_name,
        evaluate_after=not args.no_evaluate,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
