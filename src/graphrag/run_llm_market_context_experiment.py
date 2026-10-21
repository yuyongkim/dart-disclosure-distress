from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "llm"))
sys.path.insert(0, str(ROOT / "src"))

from client import LLM  # noqa: E402
from graphrag.run_upgrade_experiment import (  # noqa: E402
    DISTRESS,
    NORMAL,
    format_pct,
    safe_float,
    summarize_metrics,
    write_jsonl,
)


CONDITION_DISCLOSURE_ONLY = "disclosure_only_llm"
CONDITION_DISCLOSURE_PREMARKET = "disclosure_premarket_llm"
CONDITION_DISCLOSURE_PREMARKET_GRAPH_RAG = "disclosure_premarket_graph_rag_llm"
CONDITIONS = [
    CONDITION_DISCLOSURE_ONLY,
    CONDITION_DISCLOSURE_PREMARKET,
    CONDITION_DISCLOSURE_PREMARKET_GRAPH_RAG,
]
PROMPT_STRATEGY = "predictive_no_lookahead"

RESULT_DIR = ROOT / "results" / "graphrag"
RAW_DIR = RESULT_DIR / "llm_market_context_raw"
CASE_PATH = ROOT / "data" / "labels" / "disclosure_sequence_distress_reasoning_benchmark.csv"
CONTEXT_PATH = RESULT_DIR / "case_contexts_graph.json"
CASE_MARKET_REACTIONS = RESULT_DIR / "case_market_reactions.csv"


SYSTEM = """You are a financial-disclosure distress prediction model.
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


def extract_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("LLM response was not a JSON array.")
    return data


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, float):
        return round(value, 6)
    text = str(value)
    if text.lower() == "nan":
        return None
    return value


def compact_filings(case: dict[str, Any]) -> list[dict[str, Any]]:
    filings = json.loads(case.get("matched_filings_json") or "[]")
    return [
        {
            "date": item.get("date", ""),
            "event_type": item.get("event_type", ""),
            "event_label": item.get("event_label", ""),
            "event_subtype": item.get("event_subtype", ""),
            "title": item.get("title", ""),
        }
        for item in filings
    ]


def premarket_context(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "reaction_available": boolish(case.get("reaction_available", False)),
        "missing_reason": clean_value(case.get("missing_reason")),
        "event_trade_date": clean_value(case.get("event_trade_date")),
        "pre_close_date": clean_value(case.get("pre_close_date")),
        "days_pattern_to_event_trade": clean_value(case.get("days_pattern_to_event_trade")),
        "pre_return_20d": clean_value(case.get("pre_return_20d")),
        "pre_excess_return_20d": clean_value(case.get("pre_excess_return_20d")),
        "pre_return_60d": clean_value(case.get("pre_return_60d")),
        "pre_excess_return_60d": clean_value(case.get("pre_excess_return_60d")),
    }


def strip_post_event_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_post_event_fields(item)
            for key, item in value.items()
            if not str(key).startswith("post_")
        }
    if isinstance(value, list):
        return [strip_post_event_fields(item) for item in value]
    return clean_value(value)


def safe_graph_context(case: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    graph_summary = context.get("graph_summary", {})
    graph = context.get("graph", {})
    vector_documents = [
        doc
        for doc in context.get("vector_documents", [])
        if not str(doc).startswith("Market reaction around sequence match")
    ]
    vector_documents.append(
        "Historical market context before sequence anchor: "
        f"20d excess={format_pct(case.get('pre_excess_return_20d'))}, "
        f"60d excess={format_pct(case.get('pre_excess_return_60d'))}."
    )
    nodes = strip_post_event_fields(graph.get("nodes", []))[:8]
    edges = strip_post_event_fields(graph.get("edges", []))[:12]
    return {
        "retrieved_text": vector_documents[:5],
        "graph_context": {
            "nodes": nodes,
            "edges": edges,
            "event_density": {
                "n_events": graph_summary.get("n_events", 0),
                "events_pre_365d": graph_summary.get("events_pre_365d", 0),
                "T05_pre_365d": graph_summary.get("T05_pre_365d", 0),
                "T06_pre_365d": graph_summary.get("T06_pre_365d", 0),
                "flagged": graph_summary.get("flagged", False),
                "flag_rule": graph_summary.get("flag_rule", ""),
            },
            "relation_counterparties": graph_summary.get("relation_counterparties", [])[:8],
        },
    }


def compact_case(case: dict[str, Any], context: dict[str, Any], condition: str) -> dict[str, Any]:
    payload = {
        "case_id": case["case_id"],
        "firm_name": case["firm_name"],
        "rule_pattern": case["rule_pattern"],
        "first_pattern_date": case["first_pattern_date"],
        "rule_statistics": {
            "confidence": round(safe_float(case.get("rule_confidence")), 5),
            "lift": round(safe_float(case.get("rule_lift")), 5),
            "support_firms": int(safe_float(case.get("rule_support_firms"))),
        },
        "matched_filings": compact_filings(case),
        "sequence_history": {
            "n_events": int(safe_float(case.get("n_events"))),
            "events_pre_365d": int(safe_float(case.get("events_pre_365d"))),
            "T05_pre_365d": int(safe_float(case.get("T05_pre_365d"))),
            "T06_pre_365d": int(safe_float(case.get("T06_pre_365d"))),
            "T99_pre_365d": int(safe_float(case.get("T99_pre_365d"))),
            "flagged": boolish(case.get("flagged", False)),
            "flag_rule": clean_value(case.get("flag_rule")),
        },
    }
    if condition in {CONDITION_DISCLOSURE_PREMARKET, CONDITION_DISCLOSURE_PREMARKET_GRAPH_RAG}:
        payload["historical_market_context"] = premarket_context(case)
    if condition == CONDITION_DISCLOSURE_PREMARKET_GRAPH_RAG:
        payload.update(safe_graph_context(case, context))
    return payload


def make_prompt(cases: list[dict[str, Any]], contexts: dict[str, Any], condition: str) -> str:
    condition_notes = {
        CONDITION_DISCLOSURE_ONLY: "Use disclosure sequence and matched filing metadata only.",
        CONDITION_DISCLOSURE_PREMARKET: "Use disclosure sequence plus historical pre-anchor market returns.",
        CONDITION_DISCLOSURE_PREMARKET_GRAPH_RAG: (
            "Use disclosure sequence, historical pre-anchor market returns, retrieved text, and structured graph context."
        ),
    }
    payload = [compact_case(case, contexts[case["case_id"]], condition) for case in cases]
    return (
        f"Condition: {condition}\n"
        f"Instruction: {condition_notes[condition]}\n"
        "Prediction target: whether the case is a distress-risk signal or normal restructuring.\n"
        "Return only the JSON array.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def ask_with_retries(llm: LLM, prompt: str, provider: str, condition: str, repeat: int, args: argparse.Namespace) -> str:
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 2):
        try:
            return llm.ask(prompt, system=SYSTEM, temperature=args.temperature, max_tokens=args.max_tokens)
        except Exception as exc:  # noqa: BLE001 - write retry context to raw artifacts.
            last_error = exc
            error_path = RAW_DIR / f"{provider}_{condition}_r{repeat}_attempt{attempt}_error.txt"
            error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            time.sleep(min(10, 2 * attempt))
    assert last_error is not None
    raise last_error


def run_llm_experiment(args: argparse.Namespace) -> pd.DataFrame:
    cases = pd.read_csv(CASE_PATH, encoding="utf-8-sig")
    contexts = json.loads(CONTEXT_PATH.read_text(encoding="utf-8"))
    case_records = cases.to_dict("records")
    gold = {str(row["case_id"]): str(row["human_label"]) for row in case_records}
    firm = {str(row["case_id"]): (row.get("firm_id"), row.get("firm_name")) for row in case_records}
    llm = LLM(args.provider)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for repeat in range(1, args.repeats + 1):
            prompt = make_prompt(case_records, contexts, condition)
            prompt_path = RAW_DIR / f"{args.provider}_{condition}_r{repeat}_prompt.json"
            prompt_path.write_text(prompt, encoding="utf-8")
            raw = ask_with_retries(llm, prompt, args.provider, condition, repeat, args)
            raw_path = RAW_DIR / f"{args.provider}_{condition}_r{repeat}.txt"
            raw_path.write_text(raw, encoding="utf-8")
            parsed = extract_json_array(raw)
            by_id = {str(item.get("case_id")): item for item in parsed}
            missing = sorted(set(gold) - set(by_id))
            if missing:
                raise RuntimeError(f"LLM response missing cases for {condition}/r{repeat}: {missing[:8]}")
            for cid in sorted(gold):
                item = by_id[cid]
                prediction = str(item.get("prediction", "")).strip()
                if prediction not in {DISTRESS, NORMAL}:
                    prediction = DISTRESS if "distress" in prediction.lower() else NORMAL
                confidence = max(0.01, min(0.99, safe_float(item.get("confidence"), 0.5)))
                distress_score = confidence if prediction == DISTRESS else 1 - confidence
                firm_id, firm_name = firm[cid]
                rows.append(
                    {
                        "case_id": cid,
                        "firm_id": firm_id,
                        "firm_name": firm_name,
                        "condition": condition,
                        "prompt_strategy": PROMPT_STRATEGY,
                        "repeat": repeat,
                        "prediction": prediction,
                        "confidence": confidence,
                        "distress_score": distress_score,
                        "gold_label": gold[cid],
                        "is_correct": prediction == gold[cid],
                        "rationale": str(item.get("rationale", "")),
                        "evidence": [
                            {"source_id": x, "source_type": "llm_selected_context"}
                            for x in item.get("evidence_source_ids", [])
                        ],
                        "retrieval_backend": {
                            "llm_backend": args.provider,
                            "lookahead_policy": "premarket_only_inputs_no_post_event_returns",
                        },
                    }
                )
    return pd.DataFrame(rows)


def posthoc_market_summary(primary: pd.DataFrame) -> pd.DataFrame:
    if not CASE_MARKET_REACTIONS.exists():
        return pd.DataFrame()
    reactions = pd.read_csv(CASE_MARKET_REACTIONS, encoding="utf-8-sig")
    keep = [
        "case_id",
        "reaction_available",
        "pre_excess_return_20d",
        "pre_excess_return_60d",
        "post_excess_return_5d",
        "post_excess_return_20d",
        "post_excess_return_60d",
    ]
    keep = [col for col in keep if col in reactions.columns]
    reactions = reactions[keep].drop_duplicates("case_id")
    merged = primary.merge(reactions, on="case_id", how="left")
    rows: list[dict[str, Any]] = []
    for (condition, prediction), group in merged.groupby(["condition", "prediction"]):
        rows.append(
            {
                "condition": condition,
                "prediction": prediction,
                "cases": len(group),
                "reaction_available_cases": int(group.get("reaction_available", pd.Series(dtype=object)).astype(str).str.lower().eq("true").sum()),
                "mean_pre_20d_excess": group.get("pre_excess_return_20d", pd.Series(dtype=float)).astype(float).mean(),
                "mean_post_5d_excess": group.get("post_excess_return_5d", pd.Series(dtype=float)).astype(float).mean(),
                "mean_post_20d_excess": group.get("post_excess_return_20d", pd.Series(dtype=float)).astype(float).mean(),
                "median_post_20d_excess": group.get("post_excess_return_20d", pd.Series(dtype=float)).astype(float).median(),
            }
        )
    return pd.DataFrame(rows).sort_values(["condition", "prediction"])


def majority_primary(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["case_id", "condition", "prompt_strategy"], as_index=False)
        .agg(
            distress_score=("distress_score", "mean"),
            confidence=("confidence", "mean"),
            gold_label=("gold_label", "first"),
            firm_id=("firm_id", "first"),
            firm_name=("firm_name", "first"),
            prediction=("prediction", lambda x: Counter(x).most_common(1)[0][0]),
            is_correct=("is_correct", "mean"),
            rationale=("rationale", lambda x: " | ".join(list(dict.fromkeys(str(v) for v in x))[:2])),
        )
    )


def write_summary(
    metrics: pd.DataFrame,
    self_consistency: pd.DataFrame,
    primary: pd.DataFrame,
    market_summary: pd.DataFrame,
    path: Path,
    provider: str,
    repeats: int,
) -> None:
    lines = [
        "# LLM Premarket Context Experiment",
        "",
        f"Provider: `{provider}`",
        f"Cases: `{primary['case_id'].nunique()}`",
        f"Repeats: `{repeats}`",
        "",
        "Boundary: LLM inputs use disclosure information and, where applicable, historical pre-anchor market returns only. Post-event returns are not supplied to the LLM and are used only for post-hoc market-reaction description.",
        "",
        "## Conditions",
        "",
        "- `disclosure_only_llm`: matched DART disclosure sequence and filing metadata only.",
        "- `disclosure_premarket_llm`: disclosure sequence plus historical pre-anchor market returns.",
        "- `disclosure_premarket_graph_rag_llm`: disclosure sequence, historical pre-anchor market returns, retrieved text, and structured graph context.",
        "",
        "## Predictive Classification Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Self-Consistency",
        "",
        self_consistency.to_markdown(index=False),
        "",
        "## Post-Hoc Market-Reaction Description",
        "",
    ]
    if market_summary.empty:
        lines.append("No case-market reaction file was available.")
    else:
        lines.append(market_summary.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `results/graphrag/llm_market_context_results.jsonl`",
            "- `results/graphrag/llm_market_context_results.csv`",
            "- `results/graphrag/llm_market_context_metrics.csv`",
            "- `results/graphrag/llm_market_context_posthoc_market.csv`",
            "- `results/graphrag/llm_market_context_summary.md`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_llm_experiment(args)
    write_jsonl(results, RESULT_DIR / "llm_market_context_results.jsonl")
    results.drop(columns=["evidence", "retrieval_backend"]).to_csv(
        RESULT_DIR / "llm_market_context_results.csv", index=False, encoding="utf-8-sig"
    )
    metrics, self_consistency, faithfulness = summarize_metrics(results)
    metrics.to_csv(RESULT_DIR / "llm_market_context_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_json(RESULT_DIR / "llm_market_context_metrics.json", orient="records", force_ascii=False, indent=2)
    self_consistency.to_csv(RESULT_DIR / "llm_market_context_self_consistency.csv", index=False, encoding="utf-8-sig")
    faithfulness.to_csv(RESULT_DIR / "llm_market_context_faithfulness_proxy.csv", index=False, encoding="utf-8-sig")
    primary = majority_primary(results)
    primary.to_csv(RESULT_DIR / "llm_market_context_primary_predictions.csv", index=False, encoding="utf-8-sig")
    market_summary = posthoc_market_summary(primary)
    if not market_summary.empty:
        market_summary.to_csv(RESULT_DIR / "llm_market_context_posthoc_market.csv", index=False, encoding="utf-8-sig")
    write_summary(
        metrics,
        self_consistency,
        primary,
        market_summary,
        RESULT_DIR / "llm_market_context_summary.md",
        args.provider,
        args.repeats,
    )
    manifest = {
        "status": "completed",
        "provider": args.provider,
        "cases": int(primary["case_id"].nunique()),
        "repeats": args.repeats,
        "conditions": CONDITIONS,
        "lookahead_policy": "LLM inputs exclude post-event returns; post-event returns are post-hoc only.",
        "outputs": [
            str((RESULT_DIR / "llm_market_context_results.jsonl").relative_to(ROOT)),
            str((RESULT_DIR / "llm_market_context_metrics.csv").relative_to(ROOT)),
            str((RESULT_DIR / "llm_market_context_posthoc_market.csv").relative_to(ROOT)),
            str((RESULT_DIR / "llm_market_context_summary.md").relative_to(ROOT)),
        ],
    }
    (RESULT_DIR / "llm_market_context_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Paper 4 LLM context ablation with no-lookahead market inputs.")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
