from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_alignment import VALID_LABELS, evaluate, normalize_label


def sync_labels(workbook_path: Path, seed_path: Path, output_path: Path, evaluate_after: bool) -> dict:
    annotation = pd.read_excel(workbook_path, sheet_name="annotation")
    required = {"candidate_rank"}
    missing = required - set(annotation.columns)
    if missing:
        raise ValueError(f"Workbook annotation sheet is missing columns: {sorted(missing)}")

    seed = pd.read_csv(seed_path, low_memory=False).copy()
    label_cols = [
        "annotator1_label",
        "annotator2_label",
        "adjudicated_label",
        "human_label",
        "human_notes",
        "adjudication_notes",
    ]
    for col in label_cols:
        if col not in annotation.columns:
            annotation[col] = ""
    labels = annotation[["candidate_rank", *label_cols]].copy()
    labels["candidate_rank"] = pd.to_numeric(labels["candidate_rank"], errors="coerce").astype("Int64")
    label_only_cols = ["annotator1_label", "annotator2_label", "adjudicated_label", "human_label"]
    invalid_ranks: set[int] = set()
    for col in label_only_cols:
        norm_col = f"{col}_norm"
        labels[norm_col] = labels[col].map(normalize_label)
        invalid = labels[labels[col].notna() & labels[col].astype(str).str.strip().ne("") & labels[norm_col].eq("")]
        invalid_ranks.update(invalid["candidate_rank"].dropna().astype(int).tolist())
        labels[col] = labels[norm_col].fillna("")
    if invalid_ranks:
        raise ValueError(f"Invalid labels found at candidate_rank values: {sorted(invalid_ranks)}")

    merged = seed.drop(columns=[c for c in label_cols if c in seed.columns]).merge(
        labels[["candidate_rank", *label_cols]],
        on="candidate_rank",
        how="left",
    )
    for col in label_cols:
        if col not in merged.columns:
            merged[col] = ""
        merged[col] = merged[col].fillna("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    eval_frame = merged.copy()
    eval_frame["adjudicated_label_norm"] = eval_frame["adjudicated_label"].map(normalize_label)
    eval_frame["human_label_norm"] = eval_frame["human_label"].map(normalize_label)
    eval_frame["annotator1_label_norm"] = eval_frame["annotator1_label"].map(normalize_label)
    eval_frame["annotator2_label_norm"] = eval_frame["annotator2_label"].map(normalize_label)
    labeled_rows = int(
        (
            eval_frame["adjudicated_label_norm"].ne("")
            | eval_frame["human_label_norm"].ne("")
            | (eval_frame["annotator1_label_norm"].ne("") & eval_frame["annotator2_label_norm"].ne("") & eval_frame["annotator1_label_norm"].eq(eval_frame["annotator2_label_norm"]))
        ).sum()
    )
    result = {
        "workbook": str(workbook_path),
        "seed_path": str(seed_path),
        "output_path": str(output_path),
        "candidate_rows": int(len(merged)),
        "labeled_rows": labeled_rows,
        "unlabeled_rows": int(len(merged) - labeled_rows),
        "annotator1_labeled_rows": int(merged["annotator1_label"].isin(VALID_LABELS).sum()),
        "annotator2_labeled_rows": int(merged["annotator2_label"].isin(VALID_LABELS).sum()),
        "adjudicated_labeled_rows": int(merged["adjudicated_label"].isin(VALID_LABELS).sum()),
        "legacy_human_labeled_rows": int(merged["human_label"].isin(VALID_LABELS).sum()),
        "evaluated": False,
    }
    if evaluate_after:
        eval_result = evaluate(
            output_path,
            Path("Paper 4") / "results" / "alignment_evaluation.json",
            Path("Paper 4") / "results" / "alignment_evaluation.md",
        )
        result["evaluated"] = True
        result["evaluation_status"] = eval_result["status"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook-path", default=str(Path("Paper 4") / "data" / "processed" / "manual_alignment_seed_100_annotation_workbook.xlsx"))
    parser.add_argument("--seed-path", default=str(Path("Paper 4") / "data" / "processed" / "manual_alignment_seed_100.csv"))
    parser.add_argument("--output-path", default=str(Path("Paper 4") / "data" / "processed" / "manual_alignment_seed_100.csv"))
    parser.add_argument("--no-evaluate", action="store_true")
    args = parser.parse_args()

    result = sync_labels(
        Path(args.workbook_path),
        Path(args.seed_path),
        Path(args.output_path),
        evaluate_after=not args.no_evaluate,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
