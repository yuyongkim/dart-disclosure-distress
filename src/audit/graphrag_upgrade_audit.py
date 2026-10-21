from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "graphrag"
LABELS = ROOT / "data" / "labels"
FIGURES = ROOT / "figures" / "graphrag"

REQUIRED_CONDITIONS = {"rule_only", "rule_llm", "vector_rag_llm", "graph_rag_llm"}
REQUIRED_GRAPH_PROMPTS = {"zero_shot", "few_shot", "cot"}


def exists_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def main() -> None:
    checks: list[dict] = []

    files = {
        "cases_csv": LABELS / "disclosure_sequence_distress_reasoning_benchmark.csv",
        "cases_json": LABELS / "disclosure_sequence_distress_reasoning_benchmark.json",
        "contexts": RESULTS / "case_contexts_graph.json",
        "results_jsonl": RESULTS / "upgrade_experiment_results.jsonl",
        "results_csv": RESULTS / "upgrade_experiment_results.csv",
        "metrics_csv": RESULTS / "ablation_metrics.csv",
        "self_consistency": RESULTS / "self_consistency.csv",
        "faithfulness": RESULTS / "faithfulness_nli.csv",
        "prompt_metrics": RESULTS / "prompt_strategy_metrics.csv",
        "summary": RESULTS / "upgrade_experiment_summary.md",
        "manifest": RESULTS / "upgrade_run_manifest.json",
        "llm_deepseek_results": RESULTS / "llm_deepseek_experiment_results.jsonl",
        "llm_deepseek_metrics": RESULTS / "llm_deepseek_ablation_metrics.csv",
        "llm_deepseek_summary": RESULTS / "llm_deepseek_summary.md",
        "llm_deepseek_manifest": RESULTS / "llm_deepseek_manifest.json",
        "sequence_market_reactions": RESULTS / "sequence_market_reactions.csv",
        "case_market_reactions": RESULTS / "case_market_reactions.csv",
        "sequence_market_reaction_summary": RESULTS / "sequence_market_reaction_summary.md",
        "sequence_market_reaction_json": RESULTS / "sequence_market_reaction_summary.json",
        "outline": ROOT / "manuscript" / "archive" / "paper4_upgrade_outline.md",
        "figure_ablation": FIGURES / "ablation_accuracy.png",
        "figure_calibration": FIGURES / "calibration_curve.png",
        "figure_prompt": FIGURES / "prompt_strategy_accuracy.png",
    }
    for name, path in files.items():
        checks.append({"check": f"file:{name}", "passed": exists_nonempty(path), "detail": str(path.relative_to(ROOT))})

    cases = pd.read_csv(files["cases_csv"], encoding="utf-8-sig")
    results = pd.read_csv(files["results_csv"], encoding="utf-8-sig")
    result_records = [
        json.loads(line)
        for line in files["results_jsonl"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result_details = pd.DataFrame(result_records)
    metrics = pd.read_csv(files["metrics_csv"], encoding="utf-8-sig")
    llm_metrics = pd.read_csv(files["llm_deepseek_metrics"], encoding="utf-8-sig")
    market_reactions = pd.read_csv(files["sequence_market_reactions"], encoding="utf-8-sig")
    case_market_reactions = pd.read_csv(files["case_market_reactions"], encoding="utf-8-sig")
    contexts = json.loads(files["contexts"].read_text(encoding="utf-8"))
    market_context_count = sum(1 for cid in cases["case_id"] if "market_reaction" in contexts[cid])
    market_node_count = sum(
        1
        for cid in cases["case_id"]
        if any(n.get("type") == "MarketReaction" for n in contexts[cid]["graph"]["nodes"])
    )
    market_vector_first4_count = sum(
        1
        for cid in cases["case_id"]
        if any(
            str(doc).startswith("Market reaction around sequence match")
            for doc in contexts[cid].get("vector_documents", [])[:4]
        )
    )
    market_result_rows = result_details["condition"].isin(["vector_rag_llm", "graph_rag_llm"])
    market_evidence_count = int(
        result_details.loc[market_result_rows, "evidence"]
        .astype(str)
        .str.contains("MarketReaction around sequence match|Market reaction around sequence match", regex=True)
        .sum()
    )

    checks.extend(
        [
            {"check": "case_count_20_to_30", "passed": 20 <= len(cases) <= 30, "detail": len(cases)},
            {
                "check": "both_label_classes_present",
                "passed": {"distress", "normal_restructuring"}.issubset(set(cases["human_label"])),
                "detail": cases["human_label"].value_counts().to_dict(),
            },
            {"check": "five_rules_covered", "passed": cases["rule_id"].nunique() >= 5, "detail": int(cases["rule_id"].nunique())},
            {
                "check": "all_ablation_conditions_present",
                "passed": REQUIRED_CONDITIONS.issubset(set(results["condition"])),
                "detail": sorted(set(results["condition"])),
            },
            {
                "check": "graph_prompt_strategies_present",
                "passed": REQUIRED_GRAPH_PROMPTS.issubset(
                    set(results.loc[results["condition"] == "graph_rag_llm", "prompt_strategy"])
                ),
                "detail": sorted(set(results.loc[results["condition"] == "graph_rag_llm", "prompt_strategy"])),
            },
            {
                "check": "five_repeats_per_case_condition_prompt",
                "passed": results.groupby(["case_id", "condition", "prompt_strategy"]).size().min() >= 5,
                "detail": int(results.groupby(["case_id", "condition", "prompt_strategy"]).size().min()),
            },
            {
                "check": "graph_contexts_for_all_cases",
                "passed": set(cases["case_id"]).issubset(set(contexts)),
                "detail": len(contexts),
            },
            {
                "check": "graph_nodes_edges_nonempty",
                "passed": all(
                    contexts[cid]["graph"]["nodes"] and contexts[cid]["graph"]["edges"] for cid in cases["case_id"]
                ),
                "detail": "all case contexts have nodes and edges",
            },
            {
                "check": "sequence_market_reaction_rows_present",
                "passed": len(market_reactions) >= len(cases),
                "detail": len(market_reactions),
            },
            {
                "check": "sequence_market_reaction_case_rows",
                "passed": set(cases["case_id"]).issubset(set(case_market_reactions["case_id"])),
                "detail": int(len(case_market_reactions)),
            },
            {
                "check": "market_reaction_contexts_for_all_cases",
                "passed": market_context_count == len(cases),
                "detail": market_context_count,
            },
            {
                "check": "market_reaction_graph_nodes_for_all_cases",
                "passed": market_node_count == len(cases),
                "detail": market_node_count,
            },
            {
                "check": "market_reaction_vector_docs_first4_for_all_cases",
                "passed": market_vector_first4_count == len(cases),
                "detail": market_vector_first4_count,
            },
            {
                "check": "market_reaction_evidence_in_vector_and_graph_results",
                "passed": market_evidence_count == int(market_result_rows.sum()),
                "detail": {"rows_with_market_evidence": market_evidence_count, "expected_rows": int(market_result_rows.sum())},
            },
            {
                "check": "market_reaction_available_for_most_cases",
                "passed": int(case_market_reactions["reaction_available"].sum()) >= 25,
                "detail": int(case_market_reactions["reaction_available"].sum()),
            },
        ]
    )

    best_graph = metrics[metrics["condition"] == "graph_rag_llm"].sort_values("accuracy", ascending=False).head(1)
    rule_only = metrics[metrics["condition"] == "rule_only"].head(1)
    if best_graph.empty or rule_only.empty:
        checks.append({"check": "claim_metrics_available", "passed": False, "detail": "missing graph or rule-only metrics"})
    else:
        bg = best_graph.iloc[0]
        ro = rule_only.iloc[0]
        checks.extend(
            [
                {
                    "check": "graph_accuracy_at_least_rule_only",
                    "passed": float(bg["accuracy"]) >= float(ro["accuracy"]),
                    "detail": {"graph": float(bg["accuracy"]), "rule_only": float(ro["accuracy"])},
                },
                {
                    "check": "graph_false_positive_rate_below_rule_only",
                    "passed": float(bg["false_positive_rate"]) < float(ro["false_positive_rate"]),
                    "detail": {
                        "graph": float(bg["false_positive_rate"]),
                        "rule_only": float(ro["false_positive_rate"]),
                    },
                },
            ]
        )

    raw_dir = RESULTS / "llm_raw"
    raw_count = len(list(raw_dir.glob("deepseek_*.txt"))) if raw_dir.exists() else 0
    llm_best_graph = llm_metrics[llm_metrics["condition"] == "graph_rag_llm"].sort_values("accuracy", ascending=False).head(1)
    llm_rule_only = llm_metrics[llm_metrics["condition"] == "rule_only"].head(1)
    checks.extend(
        [
            {"check": "deepseek_raw_batches_25", "passed": raw_count >= 25, "detail": raw_count},
            {
                "check": "deepseek_all_conditions_present",
                "passed": REQUIRED_CONDITIONS.issubset(set(llm_metrics["condition"])),
                "detail": sorted(set(llm_metrics["condition"])),
            },
        ]
    )
    if llm_best_graph.empty or llm_rule_only.empty:
        checks.append({"check": "deepseek_graph_metrics_available", "passed": False, "detail": "missing graph or rule-only"})
    else:
        lbg = llm_best_graph.iloc[0]
        lro = llm_rule_only.iloc[0]
        checks.extend(
            [
                {
                    "check": "deepseek_graph_accuracy_at_least_rule_only",
                    "passed": float(lbg["accuracy"]) >= float(lro["accuracy"]),
                    "detail": {"graph": float(lbg["accuracy"]), "rule_only": float(lro["accuracy"])},
                },
                {
                    "check": "deepseek_graph_false_positive_rate_below_rule_only",
                    "passed": float(lbg["false_positive_rate"]) < float(lro["false_positive_rate"]),
                    "detail": {
                        "graph": float(lbg["false_positive_rate"]),
                        "rule_only": float(lro["false_positive_rate"]),
                    },
                },
            ]
        )

    for c in checks:
        c["passed"] = bool(c["passed"])
    failed = [c for c in checks if not c["passed"]]
    status = "pass" if not failed else "fail"
    report = {"status": status, "checks": checks, "failed": failed}
    (RESULTS / "upgrade_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Graph RAG Upgrade Audit", "", f"Status: `{status}`", ""]
    lines.append("| Check | Passed | Detail |")
    lines.append("|---|---:|---|")
    for c in checks:
        detail = c["detail"]
        if isinstance(detail, (dict, list)):
            detail = json.dumps(detail, ensure_ascii=False)
        lines.append(f"| {c['check']} | {c['passed']} | {str(detail).replace('|', '/')} |")
    (RESULTS / "upgrade_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "failed": len(failed), "checks": len(checks)}, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
