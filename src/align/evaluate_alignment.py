from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VALID_LABELS = {"MATCH", "PARTIAL", "NO_MATCH"}
LABEL_ORDER = ["MATCH", "PARTIAL", "NO_MATCH"]


def normalize_label(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper()
    return text if text in VALID_LABELS else ""


def ensure_label_columns(seed: pd.DataFrame) -> pd.DataFrame:
    seed = seed.copy()
    for col in ["annotator1_label", "annotator2_label", "adjudicated_label", "human_label"]:
        if col not in seed.columns:
            seed[col] = ""
        seed[f"{col}_norm"] = seed[col].map(normalize_label)
    return seed


def choose_gold_label(row: pd.Series) -> tuple[str, str]:
    adjudicated = row.get("adjudicated_label_norm", "")
    legacy = row.get("human_label_norm", "")
    ann1 = row.get("annotator1_label_norm", "")
    ann2 = row.get("annotator2_label_norm", "")
    if adjudicated:
        return adjudicated, "adjudicated_label"
    if legacy:
        return legacy, "human_label"
    if ann1 and ann2 and ann1 == ann2:
        return ann1, "annotator_consensus"
    if ann1 and ann2 and ann1 != ann2:
        return "", "needs_adjudication"
    if ann1:
        return "", "annotator1_only"
    if ann2:
        return "", "annotator2_only"
    return "", "none"


def cohen_kappa(labels_a: pd.Series, labels_b: pd.Series) -> float | None:
    n = int(len(labels_a))
    if n == 0:
        return None
    observed = float((labels_a == labels_b).mean())
    expected = 0.0
    for label in LABEL_ORDER:
        expected += float((labels_a == label).mean()) * float((labels_b == label).mean())
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else None
    return float((observed - expected) / (1.0 - expected))


def evaluate(seed_path: Path, out_json: Path, out_md: Path) -> dict:
    seed = pd.read_csv(seed_path, low_memory=False)
    seed = ensure_label_columns(seed)
    gold = seed.apply(choose_gold_label, axis=1, result_type="expand")
    seed["gold_label_norm"] = gold[0]
    seed["gold_label_source"] = gold[1]
    labeled = seed[seed["gold_label_norm"].ne("")].copy()
    overlap = seed[seed["annotator1_label_norm"].ne("") & seed["annotator2_label_norm"].ne("")].copy()
    agreement = float((overlap["annotator1_label_norm"] == overlap["annotator2_label_norm"]).mean()) if not overlap.empty else None
    kappa = cohen_kappa(overlap["annotator1_label_norm"], overlap["annotator2_label_norm"]) if not overlap.empty else None
    needs_adjudication = seed[seed["gold_label_source"].eq("needs_adjudication")]

    result = {
        "candidate_rows": int(len(seed)),
        "labeled_rows": int(len(labeled)),
        "unlabeled_rows": int(len(seed) - len(labeled)),
        "label_counts": {str(k): int(v) for k, v in labeled["gold_label_norm"].value_counts().items()},
        "gold_label_source_counts": {str(k): int(v) for k, v in seed["gold_label_source"].value_counts().items()},
        "annotator1_labeled_rows": int(seed["annotator1_label_norm"].ne("").sum()),
        "annotator2_labeled_rows": int(seed["annotator2_label_norm"].ne("").sum()),
        "adjudicated_labeled_rows": int(seed["adjudicated_label_norm"].ne("").sum()),
        "legacy_human_labeled_rows": int(seed["human_label_norm"].ne("").sum()),
        "annotator_overlap_rows": int(len(overlap)),
        "annotator_exact_agreement": agreement,
        "annotator_cohen_kappa": kappa,
        "needs_adjudication_rows": int(len(needs_adjudication)),
        "status": "needs_human_labels",
        "precision_at_1_match_only": None,
        "precision_at_1_match_or_partial": None,
        "mrr_binary_match": None,
    }

    if not labeled.empty:
        proposed_positive = labeled["proposed_label"].astype(str).str.upper().isin(["MATCH", "PARTIAL"])
        positive_rows = labeled[proposed_positive]
        if not positive_rows.empty:
            result["precision_at_1_match_only"] = float((positive_rows["gold_label_norm"].eq("MATCH")).mean())
            result["precision_at_1_match_or_partial"] = float((positive_rows["gold_label_norm"].isin(["MATCH", "PARTIAL"])).mean())

        reciprocal_ranks: list[float] = []
        for _, group in labeled.sort_values("candidate_rank").groupby("dart_event", dropna=False):
            ranks = group.reset_index(drop=True)
            match_positions = ranks.index[ranks["gold_label_norm"].eq("MATCH")].tolist()
            reciprocal_ranks.append(1.0 / (match_positions[0] + 1) if match_positions else 0.0)
        if reciprocal_ranks:
            result["mrr_binary_match"] = float(sum(reciprocal_ranks) / len(reciprocal_ranks))
        result["status"] = "evaluated_partial_labels" if len(labeled) < len(seed) else "evaluated_all_labels"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, out_md)
    return result


def write_markdown(result: dict, path: Path) -> None:
    lines = ["# Alignment Evaluation", ""]
    lines.append(f"Status: `{result['status']}`")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Candidate rows | {result['candidate_rows']} |")
    lines.append(f"| Labeled rows | {result['labeled_rows']} |")
    lines.append(f"| Unlabeled rows | {result['unlabeled_rows']} |")
    lines.append(f"| Annotator 1 labeled rows | {result['annotator1_labeled_rows']} |")
    lines.append(f"| Annotator 2 labeled rows | {result['annotator2_labeled_rows']} |")
    lines.append(f"| Adjudicated labeled rows | {result['adjudicated_labeled_rows']} |")
    lines.append(f"| Legacy human_label rows | {result['legacy_human_labeled_rows']} |")
    lines.append(f"| Annotator overlap rows | {result['annotator_overlap_rows']} |")
    lines.append(f"| Annotator exact agreement | {format_metric(result['annotator_exact_agreement'])} |")
    lines.append(f"| Cohen's kappa | {format_metric(result['annotator_cohen_kappa'])} |")
    lines.append(f"| Rows needing adjudication | {result['needs_adjudication_rows']} |")
    lines.append(f"| Precision@1, MATCH only | {format_metric(result['precision_at_1_match_only'])} |")
    lines.append(f"| Precision@1, MATCH or PARTIAL | {format_metric(result['precision_at_1_match_or_partial'])} |")
    lines.append(f"| MRR, binary MATCH | {format_metric(result['mrr_binary_match'])} |")
    lines.append("")
    if result["unlabeled_rows"]:
        lines.append("Human labels are still required in `annotator1_label`/`annotator2_label` plus `adjudicated_label` for disagreements before this can be reported as an RQ1 result.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def format_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-path", default=str(Path("Paper 4") / "data" / "processed" / "manual_alignment_seed_100.csv"))
    parser.add_argument("--out-json", default=str(Path("Paper 4") / "results" / "alignment_evaluation.json"))
    parser.add_argument("--out-md", default=str(Path("Paper 4") / "results" / "alignment_evaluation.md"))
    args = parser.parse_args()
    result = evaluate(Path(args.seed_path), Path(args.out_json), Path(args.out_md))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
