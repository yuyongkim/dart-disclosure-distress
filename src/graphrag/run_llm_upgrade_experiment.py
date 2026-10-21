from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "llm"))
sys.path.insert(0, str(ROOT / "src"))

from client import LLM  # noqa: E402
from graphrag.run_upgrade_experiment import (  # noqa: E402
    CONDITION_GRAPH_RAG_LLM,
    CONDITION_RULE_LLM,
    CONDITION_RULE_ONLY,
    CONDITION_VECTOR_RAG_LLM,
    DISTRESS,
    GRAPH_PROMPTS,
    NORMAL,
    summarize_metrics,
    write_jsonl,
)


SYSTEM = """You are a financial-disclosure distress reasoning model.
Classify each disclosure-sequence match as either:
- distress
- normal_restructuring

Use only the supplied case context. Do not use outside knowledge.
Return strict JSON only: an array of objects with keys:
case_id, prediction, confidence, rationale, evidence_source_ids.
confidence must be between 0 and 1.
Keep each rationale under 25 words."""


def extract_json_array(text: str) -> list[dict]:
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


def compact_case(case: dict, context: dict, condition: str, prompt_strategy: str) -> dict:
    matched = json.loads(case.get("matched_filings_json") or "[]")
    base = {
        "case_id": case["case_id"],
        "firm_name": case["firm_name"],
        "rule_pattern": case["rule_pattern"],
        "rule_confidence": round(float(case["rule_confidence"]), 5),
        "rule_lift": round(float(case["rule_lift"]), 5),
        "first_pattern_date": case["first_pattern_date"],
        "matched_filings": [
            {
                "date": m.get("date", ""),
                "event_type": m.get("event_type", ""),
                "event_subtype": m.get("event_subtype", ""),
                "title": m.get("title", ""),
            }
            for m in matched
        ],
    }
    if condition == CONDITION_RULE_LLM:
        return base
    if condition == CONDITION_VECTOR_RAG_LLM:
        base["retrieved_text"] = context.get("vector_documents", [])[:4]
        return base
    graph_summary = context.get("graph_summary", {})
    graph = context.get("graph", {})
    base["prompt_strategy"] = prompt_strategy
    base["graph_context"] = {
        "nodes": graph.get("nodes", [])[:8],
        "edges": graph.get("edges", [])[:12],
        "event_density": {
            "n_events": graph_summary.get("n_events", 0),
            "events_pre_365d": graph_summary.get("events_pre_365d", 0),
            "T05_pre_365d": graph_summary.get("T05_pre_365d", 0),
            "T06_pre_365d": graph_summary.get("T06_pre_365d", 0),
            "flagged": graph_summary.get("flagged", False),
            "flag_rule": graph_summary.get("flag_rule", ""),
        },
        "relation_counterparties": graph_summary.get("relation_counterparties", [])[:8],
    }
    if graph_summary.get("market_reaction"):
        base["graph_context"]["market_reaction"] = graph_summary["market_reaction"]
    return base


def make_prompt(cases: list[dict], contexts: dict, condition: str, prompt_strategy: str) -> str:
    payload = [compact_case(c, contexts[c["case_id"]], condition, prompt_strategy) for c in cases]
    return (
        f"Condition: {condition}\n"
        f"Prompt strategy: {prompt_strategy}\n"
        "Classify every case. Return only the JSON array.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def run(args: argparse.Namespace) -> None:
    result_dir = ROOT / "results" / "graphrag"
    cases = pd.read_csv(ROOT / "data" / "labels" / "disclosure_sequence_distress_reasoning_benchmark.csv", encoding="utf-8-sig")
    contexts = json.loads((result_dir / "case_contexts_graph.json").read_text(encoding="utf-8"))
    llm = LLM(args.provider)

    combos = [
        (CONDITION_RULE_LLM, "none"),
        (CONDITION_VECTOR_RAG_LLM, "none"),
        *[(CONDITION_GRAPH_RAG_LLM, p) for p in GRAPH_PROMPTS],
    ]
    rows: list[dict] = []
    raw_dir = result_dir / "llm_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    case_records = cases.to_dict("records")
    gold = {r["case_id"]: r["human_label"] for r in case_records}
    firm = {r["case_id"]: (r["firm_id"], r["firm_name"]) for r in case_records}

    # Include deterministic rule-only for comparable ablation table.
    for repeat in range(1, args.repeats + 1):
        for c in case_records:
            rows.append(
                {
                    "case_id": c["case_id"],
                    "firm_id": c["firm_id"],
                    "firm_name": c["firm_name"],
                    "condition": CONDITION_RULE_ONLY,
                    "prompt_strategy": "none",
                    "repeat": repeat,
                    "prediction": DISTRESS,
                    "confidence": float(c["rule_confidence"]),
                    "distress_score": 0.75,
                    "gold_label": c["human_label"],
                    "is_correct": c["human_label"] == DISTRESS,
                    "rationale": "Rule-only baseline treats every sequence match as distress.",
                    "evidence": [{"source_id": c["rule_id"], "source_type": "sequence_rule"}],
                    "retrieval_backend": {"llm_backend": "none_rule_only"},
                }
            )

    for condition, prompt_strategy in combos:
        for repeat in range(1, args.repeats + 1):
            prompt = make_prompt(case_records, contexts, condition, prompt_strategy)
            raw = llm.ask(prompt, system=SYSTEM, temperature=args.temperature, max_tokens=args.max_tokens)
            raw_path = raw_dir / f"{args.provider}_{condition}_{prompt_strategy}_r{repeat}.txt"
            raw_path.write_text(raw, encoding="utf-8")
            parsed = extract_json_array(raw)
            by_id = {str(item.get("case_id")): item for item in parsed}
            missing = sorted(set(gold) - set(by_id))
            if missing:
                raise RuntimeError(f"LLM response missing cases for {condition}/{prompt_strategy}/r{repeat}: {missing[:5]}")
            for cid, item in by_id.items():
                pred = str(item.get("prediction", "")).strip()
                if pred not in {DISTRESS, NORMAL}:
                    pred = DISTRESS if "distress" in pred.lower() else NORMAL
                conf = float(item.get("confidence", 0.5))
                conf = max(0.01, min(0.99, conf))
                distress_score = conf if pred == DISTRESS else 1 - conf
                firm_id, firm_name = firm[cid]
                rows.append(
                    {
                        "case_id": cid,
                        "firm_id": firm_id,
                        "firm_name": firm_name,
                        "condition": condition,
                        "prompt_strategy": prompt_strategy,
                        "repeat": repeat,
                        "prediction": pred,
                        "confidence": conf,
                        "distress_score": distress_score,
                        "gold_label": gold[cid],
                        "is_correct": pred == gold[cid],
                        "rationale": str(item.get("rationale", "")),
                        "evidence": [
                            {"source_id": x, "source_type": "llm_selected_context"}
                            for x in item.get("evidence_source_ids", [])
                        ],
                        "retrieval_backend": {"llm_backend": args.provider},
                    }
                )

    results = pd.DataFrame(rows)
    prefix = f"llm_{args.provider}"
    write_jsonl(results, result_dir / f"{prefix}_experiment_results.jsonl")
    results.drop(columns=["evidence", "retrieval_backend"]).to_csv(
        result_dir / f"{prefix}_experiment_results.csv", index=False, encoding="utf-8-sig"
    )
    metrics, self_consistency, faithfulness = summarize_metrics(results)
    metrics.to_csv(result_dir / f"{prefix}_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    self_consistency.to_csv(result_dir / f"{prefix}_self_consistency.csv", index=False, encoding="utf-8-sig")
    faithfulness.to_csv(result_dir / f"{prefix}_faithfulness_proxy.csv", index=False, encoding="utf-8-sig")

    summary = [
        f"# {args.provider} LLM Upgrade Experiment",
        "",
        f"Cases: {len(cases)}",
        f"Repeats: {args.repeats}",
        "",
        metrics.to_markdown(index=False),
        "",
        "Boundary: benchmark labels remain silver observed-outcome proxies, not human-adjudicated gold labels.",
    ]
    (result_dir / f"{prefix}_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    manifest = {
        "status": "completed",
        "provider": args.provider,
        "cases": len(cases),
        "repeats": args.repeats,
        "outputs": [
            str((result_dir / f"{prefix}_experiment_results.jsonl").relative_to(ROOT)),
            str((result_dir / f"{prefix}_ablation_metrics.csv").relative_to(ROOT)),
            str((result_dir / f"{prefix}_summary.md").relative_to(ROOT)),
        ],
    }
    (result_dir / f"{prefix}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external LLM ablation on Paper 4 Graph RAG upgrade cases.")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
