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


def lenient_agreement(proposed: str, llm: str) -> bool:
    if not proposed or not llm:
        return False
    if proposed == llm:
        return True
    return proposed in {"MATCH", "PARTIAL"} and llm in {"MATCH", "PARTIAL"}


def safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def pct(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def load_inputs(seed_path: Path, llm_path: Path) -> pd.DataFrame:
    seed = pd.read_csv(seed_path, low_memory=False)
    if not llm_path.exists():
        raise FileNotFoundError(f"Missing LLM alignment result file: {llm_path}")
    llm = pd.read_csv(llm_path, low_memory=False)
    keep = [
        "candidate_rank",
        "llm_decision",
        "llm_confidence",
        "llm_rationale",
        "llm_mode",
        "lexical_overlap",
        "api_status",
    ]
    llm = llm[[col for col in keep if col in llm.columns]].drop_duplicates("candidate_rank")
    df = seed.merge(llm, on="candidate_rank", how="left")
    for col in [
        "candidate_rank",
        "taxonomy_code",
        "dart_event",
        "dart_subtype",
        "dart_event_count",
        "sec_event",
        "sec_subtype",
        "sec_event_count",
        "proposed_label",
        "rationale",
        "llm_decision",
        "llm_confidence",
        "llm_rationale",
        "llm_mode",
        "lexical_overlap",
    ]:
        if col not in df.columns:
            df[col] = ""
    return df


def add_silver_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["proposed_label_norm"] = df["proposed_label"].map(norm_label)
    df["llm_label_norm"] = df["llm_decision"].map(norm_label)
    df["llm_confidence_num"] = pd.to_numeric(df["llm_confidence"], errors="coerce")
    df["lexical_overlap_num"] = pd.to_numeric(df["lexical_overlap"], errors="coerce")
    df["strict_agreement"] = df["proposed_label_norm"].eq(df["llm_label_norm"])
    df["lenient_agreement"] = [
        lenient_agreement(proposed, llm)
        for proposed, llm in zip(df["proposed_label_norm"], df["llm_label_norm"], strict=False)
    ]
    df["high_confidence"] = df["llm_confidence_num"].ge(0.75)
    df["medium_or_high_confidence"] = df["llm_confidence_num"].ge(0.65)
    df["silver_consensus"] = df["strict_agreement"] & df["medium_or_high_confidence"]
    df["silver_review_needed"] = ~df["silver_consensus"]
    df["broad_taxonomy_cell"] = df["taxonomy_code"].astype(str).isin({"T05", "T06", "T99"})
    df["review_priority_score"] = 0
    df.loc[~df["strict_agreement"], "review_priority_score"] += 40
    df.loc[~df["lenient_agreement"], "review_priority_score"] += 20
    df.loc[~df["medium_or_high_confidence"], "review_priority_score"] += 20
    df.loc[df["broad_taxonomy_cell"], "review_priority_score"] += 10
    df.loc[df["lexical_overlap_num"].fillna(0).lt(0.05), "review_priority_score"] += 5
    df["review_reason"] = df.apply(review_reason, axis=1)
    return df.sort_values(["review_priority_score", "candidate_rank"], ascending=[False, True])


def review_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if not bool(row["strict_agreement"]):
        reasons.append("proposed_llm_disagreement")
    if not bool(row["lenient_agreement"]):
        reasons.append("strict_label_family_disagreement")
    if not bool(row["medium_or_high_confidence"]):
        reasons.append("low_or_missing_llm_confidence")
    if bool(row["broad_taxonomy_cell"]):
        reasons.append("broad_taxonomy_cell")
    if safe_float(row["lexical_overlap_num"]) is None or float(row["lexical_overlap_num"]) < 0.05:
        reasons.append("low_lexical_overlap")
    return "; ".join(reasons) if reasons else "silver_consensus"


def by_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for code, group in df.groupby("taxonomy_code", dropna=False):
        total = len(group)
        strict = int(group["strict_agreement"].sum())
        lenient = int(group["lenient_agreement"].sum())
        consensus = int(group["silver_consensus"].sum())
        rows.append(
            {
                "taxonomy_code": str(code),
                "candidate_rows": int(total),
                "strict_agreement_rows": strict,
                "strict_agreement_rate": pct(strict, total),
                "lenient_agreement_rows": lenient,
                "lenient_agreement_rate": pct(lenient, total),
                "silver_consensus_rows": consensus,
                "review_needed_rows": int(total - consensus),
                "mean_llm_confidence": mean_or_none(group["llm_confidence_num"]),
            }
        )
    return pd.DataFrame(rows).sort_values("taxonomy_code")


def mean_or_none(values: pd.Series) -> float | None:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_outputs(df: pd.DataFrame, out_queue: Path, out_consensus: Path, out_json: Path, out_md: Path) -> dict:
    out_queue.parent.mkdir(parents=True, exist_ok=True)
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
        "llm_decision",
        "llm_confidence_num",
        "strict_agreement",
        "lenient_agreement",
        "silver_consensus",
        "lexical_overlap_num",
        "llm_rationale",
        "rationale",
    ]
    df[keep].to_csv(out_queue, index=False, encoding="utf-8-sig")
    df[df["silver_consensus"]][keep].to_csv(out_consensus, index=False, encoding="utf-8-sig")

    total = int(len(df))
    strict = int(df["strict_agreement"].sum())
    lenient = int(df["lenient_agreement"].sum())
    consensus = int(df["silver_consensus"].sum())
    review_needed = int(df["silver_review_needed"].sum())
    code_summary = by_taxonomy(df)
    result = {
        "status": "silver_evaluation_created",
        "candidate_rows": total,
        "strict_agreement_rows": strict,
        "strict_agreement_rate": pct(strict, total),
        "lenient_agreement_rows": lenient,
        "lenient_agreement_rate": pct(lenient, total),
        "silver_consensus_rows": consensus,
        "silver_consensus_rate": pct(consensus, total),
        "review_needed_rows": review_needed,
        "mean_llm_confidence": mean_or_none(df["llm_confidence_num"]),
        "queue_csv": str(out_queue),
        "consensus_csv": str(out_consensus),
        "guardrail": "Silver evaluation compares deterministic seed labels with offline LLM judgments for annotation triage only. It is not a human-gold Precision@1/MRR result.",
        "taxonomy_summary": code_summary.to_dict(orient="records"),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(result, code_summary, df, out_md)
    return result


def write_markdown(result: dict, code_summary: pd.DataFrame, df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Alignment Silver Evaluation",
        "",
        "This artifact compares deterministic seed labels with offline LLM alignment judgments. It is useful for annotation triage and reviewer-facing transparency, but it is not a human-gold alignment score.",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Candidate rows | {result['candidate_rows']} |",
        f"| Strict proposed-vs-LLM agreement rows | {result['strict_agreement_rows']} |",
        f"| Strict proposed-vs-LLM agreement rate | {fmt(result['strict_agreement_rate'])} |",
        f"| Lenient MATCH/PARTIAL-family agreement rows | {result['lenient_agreement_rows']} |",
        f"| Lenient MATCH/PARTIAL-family agreement rate | {fmt(result['lenient_agreement_rate'])} |",
        f"| Silver consensus rows | {result['silver_consensus_rows']} |",
        f"| Silver consensus rate | {fmt(result['silver_consensus_rate'])} |",
        f"| Rows still requiring human review | {result['review_needed_rows']} |",
        f"| Mean LLM confidence | {fmt(result['mean_llm_confidence'])} |",
        "",
        "## By Taxonomy Cell",
        "",
        "| Code | Rows | Strict agree | Strict rate | Lenient agree | Lenient rate | Silver consensus | Review needed | Mean confidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in code_summary.itertuples(index=False):
        lines.append(
            f"| {row.taxonomy_code} | {row.candidate_rows} | {row.strict_agreement_rows} | "
            f"{fmt(row.strict_agreement_rate)} | {row.lenient_agreement_rows} | {fmt(row.lenient_agreement_rate)} | "
            f"{row.silver_consensus_rows} | {row.review_needed_rows} | {fmt(row.mean_llm_confidence)} |"
        )

    lines.extend(
        [
            "",
            "## Top Review Queue",
            "",
            "| Rank | Score | Taxonomy | DART event | SEC event | Proposed | LLM | Confidence | Reason |",
            "|---:|---:|---|---|---|---|---|---:|---|",
        ]
    )
    for row in df.head(15).itertuples(index=False):
        lines.append(
            f"| {row.candidate_rank} | {row.review_priority_score} | {row.taxonomy_code} | "
            f"{escape(row.dart_event)} | {escape(row.sec_event)} | {escape(row.proposed_label)} | "
            f"{escape(row.llm_decision)} | {fmt(row.llm_confidence_num)} | {escape(row.review_reason)} |"
        )

    lines.extend(["", f"Guardrail: {result['guardrail']}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def escape(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-path", default=str(ROOT / "data" / "processed" / "manual_alignment_seed_100.csv"))
    parser.add_argument("--llm-path", default=str(ROOT / "data" / "processed" / "llm_alignment_results.csv"))
    parser.add_argument(
        "--queue-csv",
        default=str(ROOT / "data" / "processed" / "manual_alignment_seed_100_silver_review_queue.csv"),
    )
    parser.add_argument(
        "--consensus-csv",
        default=str(ROOT / "data" / "processed" / "manual_alignment_seed_100_silver_consensus.csv"),
    )
    parser.add_argument("--out-json", default=str(ROOT / "results" / "alignment_silver_evaluation.json"))
    parser.add_argument("--out-md", default=str(ROOT / "results" / "alignment_silver_evaluation.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = add_silver_fields(load_inputs(Path(args.seed_path), Path(args.llm_path)))
    result = write_outputs(
        df,
        Path(args.queue_csv),
        Path(args.consensus_csv),
        Path(args.out_json),
        Path(args.out_md),
    )
    print(json.dumps({k: result[k] for k in ["status", "candidate_rows", "silver_consensus_rows", "review_needed_rows"]}, indent=2))


if __name__ == "__main__":
    main()
