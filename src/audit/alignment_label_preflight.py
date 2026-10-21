from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
VALID_LABELS = {"MATCH", "PARTIAL", "NO_MATCH"}
LABEL_COLUMNS = ["annotator1_label", "annotator2_label", "adjudicated_label", "human_label"]

DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_MAIN_SEED = ROOT / "data" / "processed" / "manual_alignment_seed_100.csv"
DEFAULT_SINGLE_EXPERT_XLSX = ROOT / "data" / "processed" / "manual_alignment_seed_100_single_expert_label_sheet.xlsx"
DEFAULT_ANNOTATION_XLSX = ROOT / "data" / "processed" / "manual_alignment_seed_100_annotation_workbook.xlsx"
DEFAULT_SEC_SEED = ROOT / "data" / "processed" / "manual_alignment_seed_sec_desktop_100.csv"
DEFAULT_MAIN_EVALUATION = DEFAULT_RESULTS_DIR / "alignment_evaluation.json"
DEFAULT_SINGLE_READINESS = DEFAULT_RESULTS_DIR / "alignment_single_expert_readiness.json"
DEFAULT_ANNOTATION_READINESS = DEFAULT_RESULTS_DIR / "alignment_annotation_readiness.json"
DEFAULT_SEC_EVALUATION = DEFAULT_RESULTS_DIR / "alignment_sec_desktop_evaluation.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def read_excel(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except ValueError:
        return pd.DataFrame()


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    return text if text in VALID_LABELS else ""


def invalid_label_rows(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    if df.empty:
        return invalid
    rank = df["candidate_rank"] if "candidate_rank" in df.columns else pd.Series(range(1, len(df) + 1))
    for col in columns:
        if col not in df.columns:
            continue
        raw = df[col]
        nonblank = raw.notna() & raw.astype(str).str.strip().ne("")
        normalized = raw.map(normalize_label)
        bad = df[nonblank & normalized.eq("")]
        for idx, row in bad.iterrows():
            invalid.append(
                {
                    "candidate_rank": int(rank.loc[idx]) if not pd.isna(rank.loc[idx]) else None,
                    "column": col,
                    "value": str(row.get(col, "")),
                }
            )
    return invalid


def with_label_norms(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in LABEL_COLUMNS:
        if col not in result.columns:
            result[col] = ""
        result[f"{col}_norm"] = result[col].map(normalize_label)
    return result


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


def label_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "gold_labeled_rows": 0,
            "unlabeled_rows": 0,
            "annotator1_labeled_rows": 0,
            "annotator2_labeled_rows": 0,
            "adjudicated_labeled_rows": 0,
            "human_label_rows": 0,
            "annotator_overlap_rows": 0,
            "needs_adjudication_rows": 0,
            "gold_label_source_counts": {},
            "invalid_label_rows": [],
        }

    labeled = with_label_norms(df)
    gold = labeled.apply(choose_gold_label, axis=1, result_type="expand")
    labeled["gold_label_norm"] = gold[0]
    labeled["gold_label_source"] = gold[1]
    invalid = invalid_label_rows(labeled, LABEL_COLUMNS)
    return {
        "rows": int(len(labeled)),
        "gold_labeled_rows": int(labeled["gold_label_norm"].ne("").sum()),
        "unlabeled_rows": int(labeled["gold_label_norm"].eq("").sum()),
        "annotator1_labeled_rows": int(labeled["annotator1_label_norm"].ne("").sum()),
        "annotator2_labeled_rows": int(labeled["annotator2_label_norm"].ne("").sum()),
        "adjudicated_labeled_rows": int(labeled["adjudicated_label_norm"].ne("").sum()),
        "human_label_rows": int(labeled["human_label_norm"].ne("").sum()),
        "annotator_overlap_rows": int(
            (labeled["annotator1_label_norm"].ne("") & labeled["annotator2_label_norm"].ne("")).sum()
        ),
        "needs_adjudication_rows": int(labeled["gold_label_source"].eq("needs_adjudication").sum()),
        "gold_label_source_counts": {
            str(k): int(v) for k, v in labeled["gold_label_source"].value_counts().items()
        },
        "invalid_label_rows": invalid,
    }


def single_expert_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "human_labeled_rows": 0, "unlabeled_rows": 0, "invalid_label_rows": []}
    result = df.copy()
    if "human_label" not in result.columns:
        result["human_label"] = ""
    result["human_label_norm"] = result["human_label"].map(normalize_label)
    invalid = invalid_label_rows(result, ["human_label"])
    labeled_rows = int(result["human_label_norm"].ne("").sum())
    return {
        "rows": int(len(result)),
        "human_labeled_rows": labeled_rows,
        "unlabeled_rows": int(len(result) - labeled_rows),
        "invalid_label_rows": invalid,
    }


def build_preflight(
    main_seed: Path,
    single_expert_xlsx: Path,
    annotation_xlsx: Path,
    sec_seed: Path,
    main_evaluation_path: Path,
    single_readiness_path: Path,
    annotation_readiness_path: Path,
    sec_evaluation_path: Path,
) -> dict[str, Any]:
    main_seed_df = read_csv(main_seed)
    single_xlsx_df = read_excel(single_expert_xlsx, "single_expert_labels")
    annotation_xlsx_df = read_excel(annotation_xlsx, "annotation")
    sec_seed_df = read_csv(sec_seed)

    main_seed_labels = label_summary(main_seed_df)
    single_labels = single_expert_summary(single_xlsx_df)
    annotation_labels = label_summary(annotation_xlsx_df)
    sec_seed_labels = label_summary(sec_seed_df)
    main_evaluation = read_json(main_evaluation_path)
    single_readiness = read_json(single_readiness_path)
    annotation_readiness = read_json(annotation_readiness_path)
    sec_evaluation = read_json(sec_evaluation_path)

    invalid_rows = (
        main_seed_labels["invalid_label_rows"]
        + single_labels["invalid_label_rows"]
        + annotation_labels["invalid_label_rows"]
        + sec_seed_labels["invalid_label_rows"]
    )

    candidate_rows = int(main_evaluation.get("candidate_rows") or main_seed_labels["rows"] or 0)
    evaluated_labeled_rows = int(main_evaluation.get("labeled_rows") or 0)
    evaluated_status = main_evaluation.get("status")
    final_metrics_ready = (
        candidate_rows > 0
        and evaluated_labeled_rows == candidate_rows
        and evaluated_status == "evaluated_all_labels"
        and not invalid_rows
    )

    blockers: list[str] = []
    if main_seed_df.empty:
        blockers.append("missing_main_alignment_seed")
    if not main_evaluation:
        blockers.append("missing_main_alignment_evaluation")
    if single_xlsx_df.empty:
        blockers.append("missing_single_expert_label_sheet")
    if annotation_xlsx_df.empty:
        blockers.append("missing_two_annotator_workbook")
    if sec_seed_df.empty:
        blockers.append("missing_supplemental_sec_alignment_seed")
    if invalid_rows:
        blockers.append("invalid_label_values_present")
    if evaluated_labeled_rows == 0:
        blockers.append("no_gold_labels")
    elif evaluated_labeled_rows < candidate_rows:
        blockers.append("incomplete_gold_labels")
    if single_labels["human_labeled_rows"] == 0:
        blockers.append("single_expert_labels_missing")
    if annotation_labels["annotator1_labeled_rows"] == 0 or annotation_labels["annotator2_labeled_rows"] == 0:
        blockers.append("two_annotator_labels_missing")
    if annotation_labels["needs_adjudication_rows"]:
        blockers.append("adjudication_incomplete")
    if int(sec_evaluation.get("labeled_rows") or sec_seed_labels["gold_labeled_rows"]) == 0:
        blockers.append("supplemental_sec_labels_missing")
    if evaluated_status == "needs_human_labels":
        blockers.append("main_evaluation_not_ready")

    if final_metrics_ready:
        status = "ready_for_rq1_metrics"
    elif invalid_rows:
        status = "blocked_invalid_labels"
    elif evaluated_labeled_rows > 0:
        status = "partially_labeled_not_final"
    else:
        status = "blocked_missing_human_labels"

    return {
        "status": status,
        "final_rq1_metrics_ready": final_metrics_ready,
        "candidate_rows": candidate_rows,
        "main_evaluation_status": evaluated_status,
        "main_evaluation_labeled_rows": evaluated_labeled_rows,
        "main_evaluation_unlabeled_rows": int(main_evaluation.get("unlabeled_rows") or 0),
        "main_seed": main_seed_labels,
        "single_expert_sheet": {
            **single_labels,
            "readiness_status": single_readiness.get("status"),
            "readiness_workbook_structure_status": (single_readiness.get("workbook_structure") or {}).get("status"),
        },
        "two_annotator_workbook": {
            **annotation_labels,
            "readiness_status": annotation_readiness.get("status"),
            "readiness_workbook_structure_status": (annotation_readiness.get("workbook_structure") or {}).get("status"),
        },
        "supplemental_sec_seed": {
            **sec_seed_labels,
            "evaluation_status": sec_evaluation.get("status"),
            "evaluation_labeled_rows": int(sec_evaluation.get("labeled_rows") or 0),
        },
        "invalid_label_row_count": len(invalid_rows),
        "invalid_label_rows": invalid_rows[:50],
        "blockers": sorted(set(blockers)),
        "guardrail": (
            "This preflight creates no labels and does not estimate alignment accuracy. RQ1 Precision@1/MRR "
            "and inter-annotator reliability are not ready until human labels are filled, synced, and evaluated."
        ),
    }


def write_markdown(preflight: dict[str, Any], path: Path) -> None:
    single = preflight["single_expert_sheet"]
    workbook = preflight["two_annotator_workbook"]
    supplemental = preflight["supplemental_sec_seed"]
    lines = [
        "# Alignment Label Preflight",
        "",
        "This audit checks whether RQ1 human-label evidence is ready for reporting. It creates no labels and does not estimate alignment accuracy.",
        "",
        "## Status",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Status | `{preflight['status']}` |",
        f"| Final RQ1 metrics ready | {preflight['final_rq1_metrics_ready']} |",
        f"| Candidate rows | {preflight['candidate_rows']} |",
        f"| Main evaluator status | `{preflight['main_evaluation_status']}` |",
        f"| Main evaluator labeled rows | {preflight['main_evaluation_labeled_rows']} |",
        f"| Main evaluator unlabeled rows | {preflight['main_evaluation_unlabeled_rows']} |",
        f"| Single-expert human_label rows | {single['human_labeled_rows']} |",
        f"| Single-expert workbook rows | {single['rows']} |",
        f"| Single-expert workbook structure | `{single['readiness_workbook_structure_status']}` |",
        f"| Annotator 1 labeled rows | {workbook['annotator1_labeled_rows']} |",
        f"| Annotator 2 labeled rows | {workbook['annotator2_labeled_rows']} |",
        f"| Adjudicated labeled rows | {workbook['adjudicated_labeled_rows']} |",
        f"| Annotator overlap rows | {workbook['annotator_overlap_rows']} |",
        f"| Rows needing adjudication | {workbook['needs_adjudication_rows']} |",
        f"| Two-annotator workbook structure | `{workbook['readiness_workbook_structure_status']}` |",
        f"| Supplemental SEC evaluator status | `{supplemental['evaluation_status']}` |",
        f"| Supplemental SEC labeled rows | {supplemental['evaluation_labeled_rows']} |",
        f"| Invalid label rows | {preflight['invalid_label_row_count']} |",
        "",
        "## Blockers",
        "",
    ]
    if preflight["blockers"]:
        for blocker in preflight["blockers"]:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("No RQ1 label preflight blockers.")
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            preflight["guardrail"],
            "",
            "Precision@1, MRR, Cohen's kappa, and inter-annotator agreement are not ready for reporting until the listed blockers are cleared.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-seed", default=str(DEFAULT_MAIN_SEED))
    parser.add_argument("--single-expert-xlsx", default=str(DEFAULT_SINGLE_EXPERT_XLSX))
    parser.add_argument("--annotation-xlsx", default=str(DEFAULT_ANNOTATION_XLSX))
    parser.add_argument("--sec-seed", default=str(DEFAULT_SEC_SEED))
    parser.add_argument("--main-evaluation", default=str(DEFAULT_MAIN_EVALUATION))
    parser.add_argument("--single-readiness", default=str(DEFAULT_SINGLE_READINESS))
    parser.add_argument("--annotation-readiness", default=str(DEFAULT_ANNOTATION_READINESS))
    parser.add_argument("--sec-evaluation", default=str(DEFAULT_SEC_EVALUATION))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    preflight = build_preflight(
        main_seed=Path(args.main_seed),
        single_expert_xlsx=Path(args.single_expert_xlsx),
        annotation_xlsx=Path(args.annotation_xlsx),
        sec_seed=Path(args.sec_seed),
        main_evaluation_path=Path(args.main_evaluation),
        single_readiness_path=Path(args.single_readiness),
        annotation_readiness_path=Path(args.annotation_readiness),
        sec_evaluation_path=Path(args.sec_evaluation),
    )
    (results_dir / "alignment_label_preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(preflight, results_dir / "alignment_label_preflight.md")
    print(json.dumps(preflight, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
