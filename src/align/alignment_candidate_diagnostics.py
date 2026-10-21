from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path("Paper 4")
VALID_LABELS = {"MATCH", "PARTIAL", "NO_MATCH"}
BROAD_TAXONOMY_CODES = {"T05", "T06", "T99"}


def norm_label(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    return text if text in VALID_LABELS else ""


def load_candidates(seed_path: Path, llm_path: Path) -> pd.DataFrame:
    seed = pd.read_csv(seed_path, low_memory=False)
    if llm_path.exists():
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
    else:
        df = seed.copy()
        for col in ["llm_decision", "llm_confidence", "llm_rationale", "llm_mode", "lexical_overlap", "api_status"]:
            df[col] = ""

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
    ]:
        if col not in df.columns:
            df[col] = ""
    return df


def add_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["proposed_label_norm"] = df["proposed_label"].map(norm_label)
    df["ai_suggested_label"] = df["llm_decision"].map(norm_label)
    df["ai_confidence"] = pd.to_numeric(df["llm_confidence"], errors="coerce")
    df["lexical_overlap_value"] = pd.to_numeric(df["lexical_overlap"], errors="coerce")
    df["dart_event_count_num"] = pd.to_numeric(df["dart_event_count"], errors="coerce").fillna(0)
    df["sec_event_count_num"] = pd.to_numeric(df["sec_event_count"], errors="coerce").fillna(0)
    smaller = df[["dart_event_count_num", "sec_event_count_num"]].min(axis=1).clip(lower=1)
    larger = df[["dart_event_count_num", "sec_event_count_num"]].max(axis=1)
    df["event_count_ratio"] = larger / smaller

    df["flag_proposed_ai_disagreement"] = ~df["proposed_label_norm"].eq(df["ai_suggested_label"])
    df["flag_low_or_missing_confidence"] = df["ai_confidence"].isna() | df["ai_confidence"].lt(0.65)
    df["flag_broad_taxonomy_cell"] = df["taxonomy_code"].astype(str).isin(BROAD_TAXONOMY_CODES)
    df["flag_other_event_cell"] = df["taxonomy_code"].astype(str).eq("T99")
    df["flag_low_lexical_overlap"] = df["lexical_overlap_value"].isna() | df["lexical_overlap_value"].lt(0.05)
    df["flag_event_count_imbalance"] = df["event_count_ratio"].ge(10.0)

    flag_cols = [col for col in df.columns if col.startswith("flag_")]
    df["ambiguity_score"] = df[flag_cols].sum(axis=1).astype(int)
    df["diagnostic_reason"] = df[flag_cols].apply(
        lambda row: "; ".join(col.removeprefix("flag_") for col, flagged in row.items() if bool(flagged))
        or "routine_confirmation",
        axis=1,
    )
    return df.sort_values(["ambiguity_score", "candidate_rank"], ascending=[False, True])


def taxonomy_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for code, group in df.groupby("taxonomy_code", dropna=False):
        rows.append(
            {
                "taxonomy_code": code,
                "candidate_rows": int(len(group)),
                "unique_dart_events": int(group["dart_event"].nunique(dropna=True)),
                "unique_sec_events": int(group["sec_event"].nunique(dropna=True)),
                "proposed_match_rows": int(group["proposed_label_norm"].eq("MATCH").sum()),
                "proposed_partial_rows": int(group["proposed_label_norm"].eq("PARTIAL").sum()),
                "ai_match_rows": int(group["ai_suggested_label"].eq("MATCH").sum()),
                "ai_partial_rows": int(group["ai_suggested_label"].eq("PARTIAL").sum()),
                "mean_ai_confidence": safe_mean(group["ai_confidence"]),
                "proposed_ai_disagreements": int(group["flag_proposed_ai_disagreement"].sum()),
                "high_ambiguity_rows": int(group["ambiguity_score"].ge(3).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("taxonomy_code")


def safe_mean(values: pd.Series) -> float | None:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def build_diagnostics(seed_path: Path, llm_path: Path, taxonomy_path: Path, out_csv: Path, out_json: Path, out_md: Path) -> dict:
    df = add_diagnostics(load_candidates(seed_path, llm_path))
    taxonomy = pd.read_csv(taxonomy_path, low_memory=False) if taxonomy_path.exists() else pd.DataFrame()
    taxonomy_codes = sorted(str(code) for code in taxonomy.get("taxonomy_code", pd.Series(dtype=str)).dropna().unique())
    seed_codes = sorted(str(code) for code in df["taxonomy_code"].dropna().unique())
    missing_codes = sorted(set(taxonomy_codes) - set(seed_codes))
    summary_by_code = taxonomy_summary(df)

    keep = [
        "candidate_rank",
        "ambiguity_score",
        "diagnostic_reason",
        "taxonomy_code",
        "dart_event",
        "dart_subtype",
        "sec_event",
        "sec_subtype",
        "proposed_label",
        "ai_suggested_label",
        "ai_confidence",
        "lexical_overlap_value",
        "event_count_ratio",
        "dart_event_count",
        "sec_event_count",
        "llm_rationale",
        "rationale",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(out_csv, index=False, encoding="utf-8-sig")

    result = {
        "status": "created_prelabel_diagnostics",
        "candidate_rows": int(len(df)),
        "taxonomy_reference_rows": int(len(taxonomy)) if not taxonomy.empty else 0,
        "taxonomy_codes_in_seed": seed_codes,
        "taxonomy_codes_missing_from_seed": missing_codes,
        "proposed_label_counts": {str(k): int(v) for k, v in df["proposed_label_norm"].value_counts(dropna=False).items()},
        "ai_suggested_label_counts": {str(k): int(v) for k, v in df["ai_suggested_label"].value_counts(dropna=False).items()},
        "proposed_ai_disagreements": int(df["flag_proposed_ai_disagreement"].sum()),
        "low_or_missing_confidence_rows": int(df["flag_low_or_missing_confidence"].sum()),
        "broad_taxonomy_rows": int(df["flag_broad_taxonomy_cell"].sum()),
        "other_event_rows": int(df["flag_other_event_cell"].sum()),
        "low_lexical_overlap_rows": int(df["flag_low_lexical_overlap"].sum()),
        "event_count_imbalance_rows": int(df["flag_event_count_imbalance"].sum()),
        "high_ambiguity_rows": int(df["ambiguity_score"].ge(3).sum()),
        "max_ambiguity_score": int(df["ambiguity_score"].max()) if not df.empty else 0,
        "mean_ai_confidence": safe_mean(df["ai_confidence"]),
        "diagnostics_csv": str(out_csv),
        "guardrail": "Pre-label diagnostics support annotation triage only; they are not human labels and must not be reported as Precision@1 or MRR.",
    }
    result["taxonomy_summary"] = summary_by_code.to_dict(orient="records")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, summary_by_code, df, out_md)
    return result


def write_markdown(result: dict, summary_by_code: pd.DataFrame, df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Alignment Candidate Diagnostics",
        "",
        "This is a pre-label diagnostic artifact for the DART-SEC alignment seed. It describes coverage and ambiguity before human annotation; it does not create gold labels.",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Candidate rows | {result['candidate_rows']} |",
        f"| Taxonomy codes in seed | {', '.join(result['taxonomy_codes_in_seed'])} |",
        f"| Taxonomy codes not represented in seed | {', '.join(result['taxonomy_codes_missing_from_seed']) or 'none'} |",
        f"| Proposed-vs-AI disagreements | {result['proposed_ai_disagreements']} |",
        f"| Low/missing confidence rows | {result['low_or_missing_confidence_rows']} |",
        f"| Broad taxonomy rows | {result['broad_taxonomy_rows']} |",
        f"| Other-event rows | {result['other_event_rows']} |",
        f"| Low lexical-overlap rows | {result['low_lexical_overlap_rows']} |",
        f"| Event-count imbalance rows | {result['event_count_imbalance_rows']} |",
        f"| High-ambiguity rows, score >= 3 | {result['high_ambiguity_rows']} |",
        f"| Mean AI confidence | {format_float(result['mean_ai_confidence'])} |",
        "",
        "## By Taxonomy Cell",
        "",
        "| Code | Rows | DART events | SEC events | Proposed MATCH | Proposed PARTIAL | AI MATCH | AI PARTIAL | Mean confidence | Disagreements | High ambiguity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_by_code.itertuples(index=False):
        lines.append(
            "| "
            f"{escape(row.taxonomy_code)} | "
            f"{row.candidate_rows} | "
            f"{row.unique_dart_events} | "
            f"{row.unique_sec_events} | "
            f"{row.proposed_match_rows} | "
            f"{row.proposed_partial_rows} | "
            f"{row.ai_match_rows} | "
            f"{row.ai_partial_rows} | "
            f"{format_float(row.mean_ai_confidence)} | "
            f"{row.proposed_ai_disagreements} | "
            f"{row.high_ambiguity_rows} |"
        )

    lines.extend(
        [
            "",
            "## Highest-Ambiguity Rows",
            "",
            "| Rank | Score | Reasons | Proposed | AI | Confidence | DART event | SEC event |",
            "|---:|---:|---|---|---|---:|---|---|",
        ]
    )
    for row in df.head(20).itertuples(index=False):
        lines.append(
            "| "
            f"{row.candidate_rank} | "
            f"{row.ambiguity_score} | "
            f"{escape(row.diagnostic_reason)} | "
            f"`{escape(row.proposed_label)}` | "
            f"`{escape(row.ai_suggested_label)}` | "
            f"{format_float(row.ai_confidence)} | "
            f"{escape(row.dart_event)} | "
            f"{escape(row.sec_event)} |"
        )

    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            result["guardrail"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def format_float(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-path", default=str(ROOT / "data" / "processed" / "manual_alignment_seed_100.csv"))
    parser.add_argument("--llm-path", default=str(ROOT / "data" / "processed" / "llm_alignment_results.csv"))
    parser.add_argument("--taxonomy-path", default=str(ROOT / "data" / "processed" / "regulatory_event_taxonomy.csv"))
    parser.add_argument("--out-csv", default=str(ROOT / "data" / "processed" / "manual_alignment_seed_100_candidate_diagnostics.csv"))
    parser.add_argument("--out-json", default=str(ROOT / "results" / "alignment_candidate_diagnostics.json"))
    parser.add_argument("--out-md", default=str(ROOT / "results" / "alignment_candidate_diagnostics.md"))
    args = parser.parse_args()
    result = build_diagnostics(
        Path(args.seed_path),
        Path(args.llm_path),
        Path(args.taxonomy_path),
        Path(args.out_csv),
        Path(args.out_json),
        Path(args.out_md),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
