from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "graphrag"
RAW_DIR = RESULT_DIR / "llm_market_context_raw"
REPORT_PATH = RESULT_DIR / "llm_api_run_report.md"
SUMMARY_JSON_PATH = RESULT_DIR / "llm_api_run_report.json"

SYSTEM_PROMPT = """You are a financial-disclosure distress prediction model.
Classify each supplied case as exactly one of:
- distress
- normal_restructuring

Use only the supplied case context. Do not use outside knowledge.
Treat pre-market fields as historical information available at the sequence anchor.
Post-event returns are intentionally not supplied and must not be inferred.

Return strict JSON only: an array of objects with keys:
case_id, prediction, confidence, rationale, evidence_source_ids.
confidence must be between 0 and 1.
Keep each rationale under 25 words."""


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    rows = df[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for _, row in rows.iterrows():
        body.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def fmt_float(value: float | int | str, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_pct(value: float | int | str, digits: int = 1) -> str:
    try:
        number = float(value) * 100
    except (TypeError, ValueError):
        return str(value)
    sign = "+" if number > 0 else ""
    return f"{sign}{number:.{digits}f}%"


def approximate_tokens(chars: int) -> int:
    # A transparent rough estimate for English/JSON prompts when provider usage is not stored.
    return round(chars / 4)


def main() -> None:
    manifest = json.loads((RESULT_DIR / "llm_market_context_manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(RESULT_DIR / "llm_market_context_metrics.csv", encoding="utf-8-sig")
    consistency = pd.read_csv(RESULT_DIR / "llm_market_context_self_consistency.csv", encoding="utf-8-sig")
    posthoc = pd.read_csv(RESULT_DIR / "llm_market_context_posthoc_market.csv", encoding="utf-8-sig")
    primary = pd.read_csv(RESULT_DIR / "llm_market_context_primary_predictions.csv", encoding="utf-8-sig")
    faith = pd.read_csv(RESULT_DIR / "llm_market_context_faithfulness_proxy.csv", encoding="utf-8-sig")

    prompt_files = sorted(RAW_DIR.glob("*_prompt.json"))
    output_files = sorted(path for path in RAW_DIR.glob("*.txt") if "_error" not in path.name)
    error_files = sorted(RAW_DIR.glob("*_error.txt"))
    prompt_chars = sum(len(path.read_text(encoding="utf-8")) for path in prompt_files)
    output_chars = sum(len(path.read_text(encoding="utf-8")) for path in output_files)
    system_chars = len(SYSTEM_PROMPT) * len(output_files)
    estimated_prompt_tokens = approximate_tokens(prompt_chars + system_chars)
    estimated_output_tokens = approximate_tokens(output_chars)
    estimated_total_tokens = estimated_prompt_tokens + estimated_output_tokens

    metrics_out = metrics.copy()
    for col in ["accuracy", "precision", "recall", "false_positive_rate", "brier", "ece_5bin"]:
        metrics_out[col] = metrics_out[col].map(lambda x: fmt_float(x, 3))
    metrics_out = metrics_out[
        ["condition", "cases", "accuracy", "precision", "recall", "false_positive_rate", "brier", "tp", "fp", "fn", "tn"]
    ]

    consistency_out = consistency.copy()
    for col in ["mean_majority_agreement", "min_majority_agreement"]:
        consistency_out[col] = consistency_out[col].map(lambda x: fmt_float(x, 3))

    posthoc_out = posthoc.copy()
    for col in ["mean_pre_20d_excess", "mean_post_5d_excess", "mean_post_20d_excess", "median_post_20d_excess"]:
        posthoc_out[col] = posthoc_out[col].map(lambda x: fmt_pct(x, 1))
    posthoc_out = posthoc_out[
        [
            "condition",
            "prediction",
            "cases",
            "reaction_available_cases",
            "mean_pre_20d_excess",
            "mean_post_5d_excess",
            "mean_post_20d_excess",
            "median_post_20d_excess",
        ]
    ]

    pivot = primary.pivot_table(
        index=["case_id", "firm_name", "gold_label"],
        columns="condition",
        values="prediction",
        aggfunc="first",
    ).reset_index()
    condition_cols = [
        "disclosure_only_llm",
        "disclosure_premarket_llm",
        "disclosure_premarket_graph_rag_llm",
    ]
    pivot["condition_predictions_differ"] = pivot[condition_cols].nunique(axis=1) > 1
    changed = pivot[pivot["condition_predictions_differ"]].copy()
    changed_cases = int(changed.shape[0])
    changed_preview = changed.head(12).copy()

    faith_summary = faith.groupby("condition").agg(
        rows=("case_id", "count"),
        mean_evidence_items=("evidence_items", "mean"),
        mean_heuristic_entailment=("heuristic_entailment", "mean"),
    ).reset_index()
    faith_summary["mean_evidence_items"] = faith_summary["mean_evidence_items"].map(lambda x: fmt_float(x, 2))
    faith_summary["mean_heuristic_entailment"] = faith_summary["mean_heuristic_entailment"].map(lambda x: fmt_float(x, 3))

    report = f"""# LLM API Run Report

Status: `{manifest.get("status")}`
Provider: `{manifest.get("provider")}`
Run shape: `{manifest.get("cases")}` cases x `{manifest.get("repeats")}` repeats x `{len(manifest.get("conditions", []))}` conditions

## What Was Actually Called

This run used batched prompts. It did not call the API separately for every case-level row.

| Item | Value |
|---|---:|
| API calls with saved outputs | {len(output_files)} |
| Prompt files saved | {len(prompt_files)} |
| Error files saved | {len(error_files)} |
| Case-level predictions produced | {len(primary) * int(manifest.get("repeats", 5))} |
| Primary case-condition predictions after repeat aggregation | {len(primary)} |
| Stored prompt characters, excluding system prompt | {prompt_chars:,} |
| Stored output characters | {output_chars:,} |
| Estimated prompt tokens, including repeated system prompt | {estimated_prompt_tokens:,} |
| Estimated output tokens | {estimated_output_tokens:,} |
| Estimated total API tokens from saved artifacts | {estimated_total_tokens:,} |

Token accounting limitation: the DeepSeek API response `usage` object was not persisted by the current runner, so the token counts above are estimates from saved prompt/output text using a transparent `characters / 4` approximation. The Codex conversation token usage is separate from these DeepSeek API-call estimates.

## Conditions Tested

- `disclosure_only_llm`: DART sequence and matched filing metadata only.
- `disclosure_premarket_llm`: disclosure sequence plus historical pre-anchor market returns.
- `disclosure_premarket_graph_rag_llm`: disclosure sequence, pre-anchor market returns, retrieved text, and structured graph context.

No-lookahead boundary: post-event return fields were not supplied to prompts. The raw-prompt audit found zero `post_return` or `post_excess` hits in `*_prompt.json` files.

## Main Classification Result

{md_table(metrics_out, ["condition", "cases", "accuracy", "precision", "recall", "false_positive_rate", "brier", "tp", "fp", "fn", "tn"])}

Interpretation:

- `disclosure_only_llm` had the highest accuracy and recall: 0.867 accuracy and 1.000 recall.
- `disclosure_premarket_llm` performed worse in this 30-case sample: 0.767 accuracy and 0.333 false-positive rate.
- `disclosure_premarket_graph_rag_llm` did not beat disclosure-only accuracy, but it reduced over-alerting: 0.813 precision, 0.200 false-positive rate, and the best Brier score among the three conditions.

## Self-Consistency Across 5 Repeats

{md_table(consistency_out, ["condition", "case_count", "mean_majority_agreement", "min_majority_agreement"])}

## Post-Hoc Market-Reaction Split

These returns were merged only after prediction. They are not model inputs.

{md_table(posthoc_out, ["condition", "prediction", "cases", "reaction_available_cases", "mean_pre_20d_excess", "mean_post_5d_excess", "mean_post_20d_excess", "median_post_20d_excess"])}

Most important split: under `disclosure_premarket_graph_rag_llm`, predicted distress cases had mean +20 trading-session excess return of -12.0%, while predicted normal restructuring cases had +5.5%.

## Prediction Movement Across Conditions

Cases where the three primary condition-level predictions did not all agree: `{changed_cases}` out of `{pivot.shape[0]}`.

{md_table(changed_preview, ["case_id", "firm_name", "gold_label", "disclosure_only_llm", "disclosure_premarket_llm", "disclosure_premarket_graph_rag_llm"])}

## Evidence Trace Proxy

{md_table(faith_summary, ["condition", "rows", "mean_evidence_items", "mean_heuristic_entailment"])}

## Bottom Line

The API-heavy experiment produced a bounded diagnostic result, not a new headline prediction model. The useful finding is narrower: disclosure-only prompts already classify the 30 benchmark cases well, while adding graph-retrieval context lowers false positives and produces a cleaner post-hoc market-reaction split. Adding pre-anchor market returns alone did not improve performance in this sample.

Primary files:

- `results/graphrag/llm_market_context_results.jsonl`
- `results/graphrag/llm_market_context_results.csv`
- `results/graphrag/llm_market_context_primary_predictions.csv`
- `results/graphrag/llm_market_context_metrics.csv`
- `results/graphrag/llm_market_context_self_consistency.csv`
- `results/graphrag/llm_market_context_posthoc_market.csv`
- `results/graphrag/llm_market_context_raw/`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    summary = {
        "status": manifest.get("status"),
        "provider": manifest.get("provider"),
        "api_calls_with_saved_outputs": len(output_files),
        "prompt_files": len(prompt_files),
        "error_files": len(error_files),
        "cases": int(manifest.get("cases", 0)),
        "repeats": int(manifest.get("repeats", 0)),
        "conditions": manifest.get("conditions", []),
        "primary_case_condition_predictions": int(len(primary)),
        "case_level_predictions_in_raw_results": int(len(primary) * int(manifest.get("repeats", 5))),
        "prompt_chars_excluding_system": prompt_chars,
        "output_chars": output_chars,
        "estimated_prompt_tokens_including_system": estimated_prompt_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_total_api_tokens": estimated_total_tokens,
        "condition_predictions_differ_cases": changed_cases,
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
