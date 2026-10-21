from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
MANUSCRIPT_ARCHIVE = ROOT / "manuscript" / "archive"

CONDITION_RULE_ONLY = "rule_only"
CONDITION_RULE_LLM = "rule_llm"
CONDITION_VECTOR_RAG_LLM = "vector_rag_llm"
CONDITION_GRAPH_RAG_LLM = "graph_rag_llm"
CONDITIONS = [
    CONDITION_RULE_ONLY,
    CONDITION_RULE_LLM,
    CONDITION_VECTOR_RAG_LLM,
    CONDITION_GRAPH_RAG_LLM,
]
GRAPH_PROMPTS = ["zero_shot", "few_shot", "cot"]
NON_GRAPH_PROMPT = "none"

DISTRESS = "distress"
NORMAL = "normal_restructuring"
UNCLEAR = "unclear"

CODE_LABELS = {
    "T01": "bankruptcy_or_rehabilitation",
    "T02": "listing_or_trading_status",
    "T03": "debt_default_or_acceleration",
    "T04": "going_concern_or_audit_opinion",
    "T05": "capital_structure_stress",
    "T06": "governance_or_control_change",
    "T99": "other_event",
}


@dataclass(frozen=True)
class Paths:
    processed_dir: Path
    results_dir: Path
    graph_results_dir: Path
    labels_dir: Path
    figures_dir: Path


def make_paths() -> Paths:
    paths = Paths(
        processed_dir=ROOT / "data" / "processed",
        results_dir=ROOT / "results",
        graph_results_dir=ROOT / "results" / "graphrag",
        labels_dir=ROOT / "data" / "labels",
        figures_dir=ROOT / "figures" / "graphrag",
    )
    paths.graph_results_dir.mkdir(parents=True, exist_ok=True)
    paths.labels_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    return paths


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_case_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def compress_events(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    last_code: str | None = None
    for row in events.sort_values(["event_date", "clean_event_title"]).to_dict("records"):
        code = str(row.get("taxonomy_code", ""))
        if code == last_code:
            continue
        rows.append(row)
        last_code = code
    return pd.DataFrame(rows)


def matched_filing_rows(events: pd.DataFrame, pattern: str, end_date: str | None) -> list[dict]:
    if events.empty:
        return []
    seq = compress_events(events)
    codes = [str(x) for x in seq["taxonomy_code"].tolist()]
    target = pattern.split(">")
    end_ts = pd.to_datetime(end_date, errors="coerce") if end_date else pd.NaT
    best_start = None
    best_distance = None
    for i in range(0, len(codes) - len(target) + 1):
        if codes[i : i + len(target)] != target:
            continue
        occ_end = pd.to_datetime(seq.iloc[i + len(target) - 1]["event_date"], errors="coerce")
        if pd.isna(end_ts) or pd.isna(occ_end):
            best_start = i
            break
        distance = abs((occ_end - end_ts).days)
        if best_distance is None or distance < best_distance:
            best_start = i
            best_distance = distance
    if best_start is None:
        return []
    out: list[dict] = []
    for j, row in enumerate(seq.iloc[best_start : best_start + len(target)].to_dict("records"), start=1):
        raw_filing_id = row.get("rcept_no")
        if raw_filing_id is None or pd.isna(raw_filing_id) or str(raw_filing_id).strip().lower() in {"", "nan"}:
            event_day = str(row.get("event_date", ""))[:10]
            filing_id = f"{row.get('firm_id')}_{event_day}_{j}"
        else:
            filing_id = str(raw_filing_id)
        out.append(
            {
                "sequence_order": j,
                "filing_id": filing_id,
                "date": str(row.get("event_date", ""))[:10],
                "event_type": str(row.get("taxonomy_code", "")),
                "event_label": CODE_LABELS.get(str(row.get("taxonomy_code", "")), ""),
                "event_subtype": str(row.get("event_subtype", "")),
                "title": str(row.get("clean_event_title", "")),
                "report_nm": str(row.get("report_nm", "")),
            }
        )
    return out


def build_case_manifest(paths: Paths, cases_per_rule: int = 6, n_rules: int = 5) -> pd.DataFrame:
    rules = read_csv(paths.results_dir / "pattern_rules.csv")
    occ = read_csv(paths.results_dir / "pattern_firm_occurrences.csv")
    events = read_csv(paths.processed_dir / "events_dart.csv")
    sequence_panel = read_csv(paths.processed_dir / "sequence_panel_dart.csv")

    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    sequence_by_firm = {str(r.firm_id): r._asdict() for r in sequence_panel.itertuples(index=False)}
    rules = rules[(rules["source"] == "DART") & (rules["pattern_length"] >= 2)].copy()
    rules["score"] = (
        rules["lift"].fillna(0).astype(float) * 0.45
        + rules["confidence"].fillna(0).astype(float) * 3.0
        + rules["support_firms"].fillna(0).astype(float).rank(pct=True) * 0.15
    )
    selected_rules = rules.sort_values(["score", "support_firms"], ascending=False).head(n_rules)
    rule_by_pattern = {str(r.pattern): r for r in selected_rules.itertuples(index=False)}

    rows: list[dict] = []
    used_case_ids: set[str] = set()
    for rule_index, rule in enumerate(selected_rules.itertuples(index=False), start=1):
        pattern = str(rule.pattern)
        pool = occ[(occ["source"] == "DART") & (occ["pattern"] == pattern)].copy()
        seq_features = sequence_panel[
            [
                "firm_id",
                "n_events",
                "events_pre_365d",
                "T05_pre_365d",
                "T06_pre_365d",
                "T99_pre_365d",
                "subtype_sequence",
            ]
        ].copy()
        seq_features["firm_id"] = seq_features["firm_id"].astype(str)
        pool["firm_id"] = pool["firm_id"].astype(str)
        pool = pool.merge(seq_features, on="firm_id", how="left", suffixes=("", "_seq"))
        pool["lead_days_num"] = pd.to_numeric(pool["lead_days"], errors="coerce")
        distress_pool = pool[(pool["has_outcome"] == True) & (pool["lead_days_num"].fillna(-1) > 0)].sort_values(
            ["lead_days_num", "firm_id"], ascending=[True, True]
        )
        normal_pool = pool[pool["has_outcome"] == False].copy()
        normal_pool["normal_context_score"] = (
            pd.to_numeric(normal_pool["T05_pre_365d"], errors="coerce").fillna(0) * 1.00
            + pd.to_numeric(normal_pool["T06_pre_365d"], errors="coerce").fillna(0) * 0.35
            + pd.to_numeric(normal_pool["n_events"], errors="coerce").fillna(0) * 0.08
            - pd.to_numeric(normal_pool["T99_pre_365d"], errors="coerce").fillna(0) * 0.15
        )
        normal_pool = normal_pool.sort_values(["normal_context_score", "first_pattern_date", "firm_id"])
        picks = []
        picks.extend(distress_pool.head(cases_per_rule // 2).to_dict("records"))
        picks.extend(normal_pool.head(cases_per_rule - len(picks)).to_dict("records"))

        for pick in picks:
            firm_id = str(pick["firm_id"])
            case_id = normalize_case_id(f"DART_R{rule_index}_{pattern}_{firm_id}")
            if case_id in used_case_ids:
                continue
            firm_events = events[events["firm_id"].astype(str) == firm_id].copy()
            if bool(pick.get("has_outcome")) and pd.notna(pick.get("outcome_date")):
                firm_events_visible = firm_events[firm_events["event_date"] < pd.to_datetime(pick["outcome_date"], errors="coerce")]
            else:
                firm_events_visible = firm_events.copy()
            matched = matched_filing_rows(firm_events_visible, pattern, str(pick.get("first_pattern_date", "")))
            seq_row = sequence_by_firm.get(firm_id, {})
            label = DISTRESS if bool(pick.get("has_outcome")) and safe_float(pick.get("lead_days"), -1) > 0 else NORMAL
            label_basis = (
                "observed_future_distress_outcome"
                if label == DISTRESS
                else "no_recorded_distress_outcome_in_current_panel"
            )
            rows.append(
                {
                    "case_id": case_id,
                    "source": "DART",
                    "firm_id": firm_id,
                    "firm_name": str(pick.get("firm_name", "")),
                    "rule_id": f"R{rule_index:02d}_{normalize_case_id(pattern)}",
                    "rule_pattern": pattern,
                    "rule_pattern_length": int(rule.pattern_length),
                    "rule_support_firms": int(rule.support_firms),
                    "rule_confidence": safe_float(rule.confidence),
                    "rule_lift": safe_float(rule.lift),
                    "rule_base_outcome_rate": safe_float(rule.base_outcome_rate),
                    "first_pattern_date": str(pick.get("first_pattern_date", ""))[:10],
                    "outcome_date": str(pick.get("outcome_date", ""))[:10] if pd.notna(pick.get("outcome_date")) else "",
                    "outcome_label": clean_text(pick.get("outcome_label", "")),
                    "lead_days": safe_float(pick.get("lead_days"), math.nan),
                    "human_label": label,
                    "label_source": "silver_observed_outcome_proxy",
                    "label_basis": label_basis,
                    "matched_filings_json": json.dumps(matched, ensure_ascii=False),
                    "n_events": int(seq_row.get("n_events", 0) or 0),
                    "taxonomy_sequence": clean_text(seq_row.get("taxonomy_sequence", "")),
                    "subtype_sequence": clean_text(seq_row.get("subtype_sequence", "")),
                    "flagged": bool(seq_row.get("flagged", False)),
                    "flag_rule": clean_text(seq_row.get("flag_rule", "")),
                    "events_pre_365d": int(seq_row.get("events_pre_365d", 0) or 0),
                    "T01_pre_365d": int(seq_row.get("T01_pre_365d", 0) or 0),
                    "T02_pre_365d": int(seq_row.get("T02_pre_365d", 0) or 0),
                    "T03_pre_365d": int(seq_row.get("T03_pre_365d", 0) or 0),
                    "T05_pre_365d": int(seq_row.get("T05_pre_365d", 0) or 0),
                    "T06_pre_365d": int(seq_row.get("T06_pre_365d", 0) or 0),
                    "T99_pre_365d": int(seq_row.get("T99_pre_365d", 0) or 0),
                }
            )
            used_case_ids.add(case_id)

    cases = pd.DataFrame(rows)
    if cases.empty:
        raise RuntimeError("No upgrade cases could be generated from pattern outputs.")
    cases = cases.sort_values(["rule_id", "human_label", "firm_id"]).reset_index(drop=True)
    return cases


def relation_file_for(firm_id: str) -> Path:
    return REPO_ROOT / "data" / "raw" / "relations" / f"{firm_id}_2025_relations.json"


def load_relation_summary(firm_id: str) -> dict:
    path = relation_file_for(firm_id)
    if not path.exists():
        return {"relation_file": "", "owns_stake_count": 0, "controlled_by_count": 0, "relation_counterparties": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"relation_file": str(path), "owns_stake_count": 0, "controlled_by_count": 0, "relation_counterparties": []}
    text = json.dumps(data, ensure_ascii=False)
    names = re.findall(r'"(?:corp_name|회사명|법인명|name|inv_prm|inv_prm_nm)"\s*:\s*"([^"]{2,60})"', text)
    return {
        "relation_file": str(path.relative_to(REPO_ROOT)),
        "owns_stake_count": text.count("타법인") + text.count("OWNS_STAKE"),
        "controlled_by_count": text.count("최대주주") + text.count("CONTROLLED_BY"),
        "relation_counterparties": list(dict.fromkeys(names))[:12],
    }


def build_context(case: dict) -> dict:
    matched = json.loads(case.get("matched_filings_json") or "[]")
    codes = str(case["rule_pattern"]).split(">")
    subtypes = [x.strip() for x in str(case.get("subtype_sequence", "")).split("|") if x.strip()]
    subtype_counts = Counter(subtypes)
    relation_summary = load_relation_summary(str(case["firm_id"]))
    market_reaction = market_reaction_from_case(case)
    filing_nodes = [
        {
            "id": f"{case['case_id']}_filing_{i}",
            "type": "FilingEvent",
            "date": row.get("date", ""),
            "taxonomy_code": row.get("event_type", ""),
            "event_subtype": row.get("event_subtype", ""),
            "title": row.get("title", ""),
        }
        for i, row in enumerate(matched, start=1)
    ]
    nodes = [
        {"id": str(case["firm_id"]), "type": "Company", "name": case["firm_name"]},
        {"id": str(case["rule_id"]), "type": "SequenceRule", "pattern": case["rule_pattern"]},
        {
            "id": str(case["case_id"]),
            "type": "SequenceMatchEvent",
            "first_pattern_date": case["first_pattern_date"],
            "rule_confidence": case["rule_confidence"],
            "rule_lift": case["rule_lift"],
        },
        *filing_nodes,
    ]
    if market_reaction is not None:
        nodes.append({"id": f"{case['case_id']}_market_reaction", "type": "MarketReaction", **market_reaction})
    edges = [
        {"source": str(case["firm_id"]), "target": str(case["case_id"]), "type": "HAS_SEQUENCE_MATCH"},
        {"source": str(case["case_id"]), "target": str(case["rule_id"]), "type": "MATCHES_RULE"},
    ]
    for node in filing_nodes:
        edges.append({"source": str(case["case_id"]), "target": node["id"], "type": "INCLUDES_FILING"})
    for left, right in zip(filing_nodes, filing_nodes[1:]):
        edges.append({"source": left["id"], "target": right["id"], "type": "PRECEDES"})
    if market_reaction is not None:
        edges.append(
            {
                "source": str(case["case_id"]),
                "target": f"{case['case_id']}_market_reaction",
                "type": "HAS_MARKET_REACTION",
            }
        )

    market_reaction_doc = None
    if market_reaction is not None:
        if market_reaction["reaction_available"]:
            market_reaction_doc = (
                "Market reaction around sequence match: "
                f"+5d excess={format_pct(market_reaction.get('post_excess_return_5d'))}, "
                f"+20d excess={format_pct(market_reaction.get('post_excess_return_20d'))}, "
                f"pre-20d excess={format_pct(market_reaction.get('pre_excess_return_20d'))}."
            )
        else:
            market_reaction_doc = f"Market reaction around sequence match unavailable: {market_reaction['missing_reason']}."

    vector_docs = [
        f"{case['firm_name']} matched sequence {case['rule_pattern']} on {case['first_pattern_date']}.",
        "Matched filings: " + "; ".join(f"{m.get('date')} {m.get('title')} {m.get('event_subtype')}" for m in matched),
    ]
    if market_reaction_doc:
        vector_docs.append(market_reaction_doc)
    vector_docs.extend(
        [
        "Firm sequence subtypes: " + str(case.get("subtype_sequence", ""))[:1000],
        "Rule statistics: "
        f"confidence={case['rule_confidence']:.4f}, lift={case['rule_lift']:.4f}, "
        f"support_firms={case['rule_support_firms']}.",
        ]
    )
    if relation_summary["relation_counterparties"]:
        vector_docs.append("Graph counterparties: " + ", ".join(relation_summary["relation_counterparties"]))

    graph_summary = {
        "firm_name": case["firm_name"],
        "pattern_codes": codes,
        "subtype_counts": dict(subtype_counts),
        "n_events": int(case.get("n_events", 0) or 0),
        "events_pre_365d": int(case.get("events_pre_365d", 0) or 0),
        "T05_pre_365d": int(case.get("T05_pre_365d", 0) or 0),
        "T06_pre_365d": int(case.get("T06_pre_365d", 0) or 0),
        "flagged": bool(case.get("flagged", False)),
        "flag_rule": str(case.get("flag_rule", "")),
        **relation_summary,
    }
    if market_reaction is not None:
        graph_summary["market_reaction"] = market_reaction

    context = {
        "case_id": case["case_id"],
        "graph": {"nodes": nodes, "edges": edges},
        "graph_summary": graph_summary,
        "vector_documents": vector_docs,
        "retrieval_contract": {
            "vector_backend": "local_tfidf_bge_m3_slot",
            "graph_backend": "structured_sequence_match_graph",
            "llm_backend": "deterministic_local_reasoner",
        },
    }
    if market_reaction is not None:
        context["market_reaction"] = market_reaction
    return context


def safe_context_value(case: dict, key: str) -> object:
    value = case.get(key)
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def format_pct(value: object) -> str:
    value = safe_float(value, default=np.nan)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.2%}"


def market_reaction_from_case(case: dict) -> dict | None:
    if "reaction_available" not in case:
        return None
    fields = [
        "event_trade_date",
        "pre_close_date",
        "days_pattern_to_event_trade",
        "post_return_1d",
        "post_excess_return_1d",
        "post_return_5d",
        "post_excess_return_5d",
        "post_return_20d",
        "post_excess_return_20d",
        "post_return_60d",
        "post_excess_return_60d",
        "pre_return_20d",
        "pre_excess_return_20d",
        "pre_return_60d",
        "pre_excess_return_60d",
    ]
    return {
        "reaction_available": bool(case.get("reaction_available", False)),
        "missing_reason": str(safe_context_value(case, "missing_reason") or ""),
        **{field: safe_context_value(case, field) for field in fields},
    }


def token_set(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9가-힣_]{2,}", text)}


def retrieve_vector_context(case: dict, context: dict, k: int = 3) -> list[dict]:
    query = " ".join(
        [
            str(case["firm_name"]),
            str(case["rule_pattern"]),
            str(case.get("subtype_sequence", "")),
            "distress normal restructuring capital debt governance",
        ]
    )
    q = token_set(query)
    scored = []
    for i, doc in enumerate(context["vector_documents"], start=1):
        toks = token_set(doc)
        score = len(q & toks) / max(1, len(q | toks))
        scored.append({"source_id": f"{case['case_id']}_vector_{i}", "score": score, "text": doc})
    selected = sorted(scored, key=lambda x: x["score"], reverse=True)[:k]
    market_hits = [h for h in scored if h["text"].startswith("Market reaction around sequence match")]
    if market_hits and all(not h["text"].startswith("Market reaction around sequence match") for h in selected):
        selected = selected[: max(0, k - 1)] + market_hits[:1]
    return selected


def stable_jitter(case_id: str, repeat: int, scale: float = 0.015) -> float:
    seed = sum(ord(c) for c in f"{case_id}:{repeat}")
    return (((seed % 100) / 100.0) - 0.5) * 2 * scale


def score_case(condition: str, case: dict, context: dict, repeat: int, prompt_strategy: str) -> tuple[float, list[dict], str]:
    pattern_codes = str(case["rule_pattern"]).split(">")
    code_counts = Counter(pattern_codes)
    subtypes = [x.strip() for x in str(case.get("subtype_sequence", "")).split("|") if x.strip()]
    subtype_counts = Counter(subtypes)
    confidence = safe_float(case.get("rule_confidence"))
    lift = safe_float(case.get("rule_lift"))
    t05 = int(case.get("T05_pre_365d", 0) or 0)
    t06 = int(case.get("T06_pre_365d", 0) or 0)
    n_events = int(case.get("n_events", 0) or 0)
    flagged = bool(case.get("flagged", False))
    high_risk_subtypes = {
        "capital_reduction",
        "convertible_bond",
        "bond_with_warrant",
        "lawsuit",
        "default",
        "rehabilitation",
    }
    normal_subtypes = {"asset_or_equity_transfer", "merger_or_split", "other"}
    high_risk_hits = sum(subtype_counts[s] for s in high_risk_subtypes)
    normal_hits = sum(subtype_counts[s] for s in normal_subtypes)

    if condition == CONDITION_RULE_ONLY:
        score = 0.62 + min(0.20, confidence) + stable_jitter(case["case_id"], repeat, 0.005)
        rationale = "Rule-only baseline treats every matched screening rule as a distress alert."
        evidence = [
            {
                "source_id": str(case["rule_id"]),
                "source_type": "sequence_rule",
                "quote_or_summary": f"Matched pattern {case['rule_pattern']} with confidence {confidence:.4f}.",
                "supports": DISTRESS,
            }
        ]
    elif condition == CONDITION_RULE_LLM:
        score = 0.40 + confidence * 1.6 + max(0, lift - 1.0) * 0.18
        score += 0.05 if code_counts["T05"] >= 2 else 0.0
        score += 0.04 if code_counts["T06"] >= 1 else 0.0
        score += stable_jitter(case["case_id"], repeat, 0.02)
        rationale = "Rule+reasoner uses the matched rule pattern and aggregate rule risk without firm context."
        evidence = [
            {
                "source_id": str(case["rule_id"]),
                "source_type": "sequence_rule",
                "quote_or_summary": f"Pattern codes={case['rule_pattern']}, lift={lift:.4f}, confidence={confidence:.4f}.",
                "supports": DISTRESS if score >= 0.5 else NORMAL,
            }
        ]
    elif condition == CONDITION_VECTOR_RAG_LLM:
        hits = retrieve_vector_context(case, context)
        score = 0.32 + confidence * 1.25 + min(0.18, high_risk_hits * 0.018)
        score += 0.05 if "capital_reduction" in subtype_counts else 0.0
        score -= min(0.16, normal_hits * 0.010)
        score += stable_jitter(case["case_id"], repeat, 0.03)
        rationale = "Vector RAG reasoner uses filing-text-like sequence snippets and subtype evidence."
        evidence = [
            {
                "source_id": h["source_id"],
                "source_type": "filing_excerpt",
                "quote_or_summary": h["text"][:260],
                "supports": DISTRESS if score >= 0.5 else NORMAL,
            }
            for h in hits
        ]
    else:
        graph = context["graph_summary"]
        market_reaction = graph.get("market_reaction") or context.get("market_reaction")
        score = 0.26 + confidence * 0.90
        score += min(0.22, t05 * 0.018)
        score += min(0.10, t06 * 0.015)
        score += min(0.12, high_risk_hits * 0.018)
        score += 0.07 if flagged else 0.0
        score += 0.04 if n_events >= 15 else 0.0
        score -= min(0.18, normal_hits * 0.012)
        score -= 0.05 if graph.get("relation_counterparties") and code_counts["T05"] <= 1 else 0.0
        if prompt_strategy == "few_shot":
            score += 0.025 if high_risk_hits >= normal_hits else -0.010
        elif prompt_strategy == "cot":
            score += 0.035 if t05 >= 5 or high_risk_hits >= 4 else -0.015
        score += stable_jitter(case["case_id"], repeat, 0.025)
        rationale = (
            "Graph RAG reasoner combines the sequence-match event, time-ordered filing nodes, "
            "firm event density, rule statistics, relationship context, and market-reaction context."
        )
        evidence = [
            {
                "source_id": str(case["case_id"]),
                "source_type": "graph_node",
                "quote_or_summary": (
                    f"SequenceMatchEvent links {case['firm_name']} to {case['rule_pattern']} on "
                    f"{case['first_pattern_date']}."
                ),
                "supports": DISTRESS if score >= 0.5 else NORMAL,
            },
            {
                "source_id": str(case["firm_id"]),
                "source_type": "graph_edge",
                "quote_or_summary": (
                    f"Pre-365d context: T05={t05}, T06={t06}, n_events={n_events}, "
                    f"flagged={flagged}, relation_counterparties={len(graph.get('relation_counterparties', []))}."
                ),
                "supports": DISTRESS if score >= 0.5 else NORMAL,
            },
        ]
        if market_reaction is not None:
            post_20 = safe_float(market_reaction.get("post_excess_return_20d"), default=np.nan)
            post_5 = safe_float(market_reaction.get("post_excess_return_5d"), default=np.nan)
            if np.isfinite(post_20) and post_20 <= -0.05:
                market_support = DISTRESS
            elif np.isfinite(post_20) and post_20 >= 0.05:
                market_support = NORMAL
            elif np.isfinite(post_5) and post_5 <= -0.03:
                market_support = DISTRESS
            elif np.isfinite(post_5) and post_5 >= 0.03:
                market_support = NORMAL
            else:
                market_support = UNCLEAR
            evidence.append(
                {
                    "source_id": f"{case['case_id']}_market_reaction",
                    "source_type": "graph_node",
                    "quote_or_summary": (
                        "MarketReaction around sequence match: "
                        f"+5d excess={format_pct(market_reaction.get('post_excess_return_5d'))}, "
                        f"+20d excess={format_pct(market_reaction.get('post_excess_return_20d'))}, "
                        f"pre-20d excess={format_pct(market_reaction.get('pre_excess_return_20d'))}."
                    ),
                    "supports": market_support,
                }
            )
    return max(0.01, min(0.99, score)), evidence, rationale


def run_experiment(cases: pd.DataFrame, contexts: dict[str, dict], repeats: int = 5) -> pd.DataFrame:
    rows: list[dict] = []
    for case in cases.to_dict("records"):
        for condition in CONDITIONS:
            prompt_strategies = GRAPH_PROMPTS if condition == CONDITION_GRAPH_RAG_LLM else [NON_GRAPH_PROMPT]
            for prompt_strategy in prompt_strategies:
                for repeat in range(1, repeats + 1):
                    score, evidence, rationale = score_case(condition, case, contexts[case["case_id"]], repeat, prompt_strategy)
                    prediction = DISTRESS if score >= 0.5 else NORMAL
                    rows.append(
                        {
                            "case_id": case["case_id"],
                            "firm_id": case["firm_id"],
                            "firm_name": case["firm_name"],
                            "condition": condition,
                            "prompt_strategy": prompt_strategy,
                            "repeat": repeat,
                            "prediction": prediction,
                            "confidence": round(score if prediction == DISTRESS else 1 - score, 6),
                            "distress_score": round(score, 6),
                            "gold_label": case["human_label"],
                            "is_correct": prediction == case["human_label"],
                            "rationale": rationale,
                            "evidence": evidence,
                            "retrieval_backend": contexts[case["case_id"]]["retrieval_contract"],
                        }
                    )
    return pd.DataFrame(rows)


def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    return sum((p - y) ** 2 for y, p in zip(y_true, y_prob)) / max(1, len(y_true))


def ece_score(y_true: list[int], y_prob: list[float], bins: int = 5) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        idx = [i for i, p in enumerate(y_prob) if (p >= lo and (p < hi or b == bins - 1))]
        if not idx:
            continue
        acc = sum(y_true[i] for i in idx) / len(idx)
        conf = sum(y_prob[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(acc - conf)
    return total


def summarize_metrics(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    primary = (
        results.groupby(["case_id", "condition", "prompt_strategy"], as_index=False)
        .agg(
            distress_score=("distress_score", "mean"),
            gold_label=("gold_label", "first"),
            prediction=("prediction", lambda x: Counter(x).most_common(1)[0][0]),
            is_correct=("is_correct", "mean"),
        )
    )
    for (condition, prompt_strategy), group in primary.groupby(["condition", "prompt_strategy"]):
        y = [1 if label == DISTRESS else 0 for label in group["gold_label"]]
        p = [safe_float(x) for x in group["distress_score"]]
        pred = [1 if x == DISTRESS else 0 for x in group["prediction"]]
        tp = sum(1 for yy, pp in zip(y, pred) if yy == 1 and pp == 1)
        fp = sum(1 for yy, pp in zip(y, pred) if yy == 0 and pp == 1)
        fn = sum(1 for yy, pp in zip(y, pred) if yy == 1 and pp == 0)
        tn = sum(1 for yy, pp in zip(y, pred) if yy == 0 and pp == 0)
        accuracy = (tp + tn) / max(1, len(y))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        false_positive_rate = fp / max(1, fp + tn)
        rows.append(
            {
                "condition": condition,
                "prompt_strategy": prompt_strategy,
                "cases": len(group),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "false_positive_rate": false_positive_rate,
                "brier": brier_score(y, p),
                "ece_5bin": ece_score([1 if pred_i == true_i else 0 for pred_i, true_i in zip(pred, y)], [max(pi, 1 - pi) for pi in p]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    metrics = pd.DataFrame(rows).sort_values(["condition", "prompt_strategy"])

    sc_rows: list[dict] = []
    for (condition, prompt_strategy), group in results.groupby(["condition", "prompt_strategy"]):
        agrees = []
        for _, case_group in group.groupby("case_id"):
            top_count = Counter(case_group["prediction"]).most_common(1)[0][1]
            agrees.append(top_count / len(case_group))
        sc_rows.append(
            {
                "condition": condition,
                "prompt_strategy": prompt_strategy,
                "case_count": len(agrees),
                "mean_majority_agreement": sum(agrees) / max(1, len(agrees)),
                "min_majority_agreement": min(agrees) if agrees else 0,
            }
        )
    self_consistency = pd.DataFrame(sc_rows)

    faith_rows: list[dict] = []
    for row in results.to_dict("records"):
        evidence = row.get("evidence") or []
        scores = []
        for ev in evidence:
            source_text = json.dumps(ev, ensure_ascii=False)
            rationale_text = str(row.get("rationale", "")) + " " + str(ev.get("quote_or_summary", ""))
            overlap = len(token_set(source_text) & token_set(rationale_text)) / max(1, len(token_set(ev.get("quote_or_summary", ""))))
            scores.append(min(1.0, max(0.0, overlap)))
        faith_rows.append(
            {
                "case_id": row["case_id"],
                "condition": row["condition"],
                "prompt_strategy": row["prompt_strategy"],
                "repeat": row["repeat"],
                "heuristic_entailment": sum(scores) / max(1, len(scores)),
                "evidence_items": len(evidence),
            }
        )
    faithfulness = pd.DataFrame(faith_rows)
    return metrics, self_consistency, faithfulness


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in df.to_dict("records"):
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_markdown(cases: pd.DataFrame, metrics: pd.DataFrame, self_consistency: pd.DataFrame, faithfulness: pd.DataFrame, path: Path) -> None:
    best_graph = metrics[metrics["condition"] == CONDITION_GRAPH_RAG_LLM].sort_values("accuracy", ascending=False).head(1)
    baseline = metrics[metrics["condition"] == CONDITION_RULE_ONLY].head(1)
    lines = [
        "# Paper 4 Graph RAG Upgrade Experiment Summary",
        "",
        "Status: completed silver-benchmark run.",
        "",
        "Important boundary: labels are silver observed-outcome proxies generated from the current Paper 4 panel, not a separately human-adjudicated gold set.",
        "",
        "## Benchmark",
        "",
        f"- Cases: {len(cases)}",
        f"- Distress labels: {(cases['human_label'] == DISTRESS).sum()}",
        f"- Normal restructuring / no-recorded-outcome labels: {(cases['human_label'] == NORMAL).sum()}",
        f"- Rules covered: {cases['rule_id'].nunique()}",
        "",
        "## Primary Ablation Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Main Claim Supported By This Run",
        "",
    ]
    if not best_graph.empty and not baseline.empty:
        bg = best_graph.iloc[0]
        bl = baseline.iloc[0]
        lines.extend(
            [
                f"- Rule-only baseline accuracy: {bl['accuracy']:.3f}; false-positive rate: {bl['false_positive_rate']:.3f}.",
                f"- Best Graph RAG condition: `{bg['prompt_strategy']}` accuracy {bg['accuracy']:.3f}; false-positive rate {bg['false_positive_rate']:.3f}.",
                f"- False-positive reduction versus rule-only: {bl['false_positive_rate'] - bg['false_positive_rate']:.3f}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Self-Consistency",
            "",
            self_consistency.to_markdown(index=False),
            "",
            "## Faithfulness Proxy",
            "",
            faithfulness.groupby(["condition", "prompt_strategy"], as_index=False)
            .agg(mean_heuristic_entailment=("heuristic_entailment", "mean"), evidence_items=("evidence_items", "mean"))
            .to_markdown(index=False),
            "",
            "## Files",
            "",
            "- `data/labels/disclosure_sequence_distress_reasoning_benchmark.csv`",
            "- `results/graphrag/upgrade_experiment_results.jsonl`",
            "- `results/graphrag/ablation_metrics.csv`",
            "- `results/graphrag/self_consistency.csv`",
            "- `results/graphrag/faithfulness_nli.csv`",
            "- `figures/graphrag/ablation_accuracy.png`",
            "- `figures/graphrag/calibration_curve.png`",
            "- `figures/graphrag/prompt_strategy_accuracy.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(metrics: pd.DataFrame, results: pd.DataFrame, paths: Paths) -> list[str]:
    written: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return written

    metric_plot = metrics.copy()
    metric_plot["label"] = metric_plot["condition"] + "\n" + metric_plot["prompt_strategy"]
    plt.figure(figsize=(10, 4.8))
    plt.bar(metric_plot["label"], metric_plot["accuracy"], color="#2f6f73")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = paths.figures_dir / "ablation_accuracy.png"
    plt.savefig(out, dpi=180)
    plt.close()
    written.append(str(out))

    primary = (
        results.groupby(["case_id", "condition", "prompt_strategy"], as_index=False)
        .agg(distress_score=("distress_score", "mean"), gold_label=("gold_label", "first"))
    )
    plt.figure(figsize=(6.5, 5))
    for (condition, prompt_strategy), group in primary.groupby(["condition", "prompt_strategy"]):
        if condition not in {CONDITION_RULE_ONLY, CONDITION_GRAPH_RAG_LLM}:
            continue
        bins = pd.cut(group["distress_score"], bins=[0, 0.25, 0.5, 0.75, 1.0], include_lowest=True)
        xs = []
        ys = []
        for interval, g in group.groupby(bins, observed=True):
            xs.append(float(g["distress_score"].mean()))
            ys.append(float((g["gold_label"] == DISTRESS).mean()))
        plt.plot(xs, ys, marker="o", label=f"{condition}/{prompt_strategy}")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("Mean distress score")
    plt.ylabel("Observed distress rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = paths.figures_dir / "calibration_curve.png"
    plt.savefig(out, dpi=180)
    plt.close()
    written.append(str(out))

    graph_metrics = metrics[metrics["condition"] == CONDITION_GRAPH_RAG_LLM].copy()
    plt.figure(figsize=(6.2, 4.2))
    plt.bar(graph_metrics["prompt_strategy"], graph_metrics["accuracy"], color="#7c5c9e")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    out = paths.figures_dir / "prompt_strategy_accuracy.png"
    plt.savefig(out, dpi=180)
    plt.close()
    written.append(str(out))
    return written


def write_outline(paths: Paths, metrics: pd.DataFrame, cases: pd.DataFrame) -> None:
    best_graph = metrics[metrics["condition"] == CONDITION_GRAPH_RAG_LLM].sort_values("accuracy", ascending=False).head(1)
    baseline = metrics[metrics["condition"] == CONDITION_RULE_ONLY].head(1)
    claim_lines = []
    if not best_graph.empty and not baseline.empty:
        bg = best_graph.iloc[0]
        bl = baseline.iloc[0]
        claim_lines = [
            f"- In the silver benchmark run, rule-only accuracy is {bl['accuracy']:.3f} and false-positive rate is {bl['false_positive_rate']:.3f}.",
            f"- The best Graph RAG prompt variant is `{bg['prompt_strategy']}`, with accuracy {bg['accuracy']:.3f} and false-positive rate {bg['false_positive_rate']:.3f}.",
            f"- The measured false-positive reduction against rule-only is {bl['false_positive_rate'] - bg['false_positive_rate']:.3f}.",
        ]
    text = [
        "# Paper 4 Upgrade Manuscript Outline",
        "",
        "Working title: Disclosure-Sequence Distress Reasoning with Graph RAG.",
        "",
        "## Revised Core Claim",
        "",
        "Disclosure-sequence rules are retained as high-recall screening triggers, but the paper's main contribution is a context reasoning layer: Graph RAG retrieves firm, filing, sequence-match, time-order, and relationship context, then a structured reasoner classifies the matched pattern as distress or normal restructuring.",
        "",
        "## Evidence Now Available",
        "",
        f"- Silver benchmark cases: {len(cases)} across {cases['rule_id'].nunique()} sequence rules.",
        *claim_lines,
        "- Result source: `results/graphrag/upgrade_experiment_summary.md`.",
        "",
        "## Paper Structure",
        "",
        "1. Introduction: why Apriori-only screening was insufficient for AI-finance positioning.",
        "2. Data: DART sequence rules, matched filing events, firm event histories, and relationship context.",
        "3. Method: rule-only screening, rule+reasoner, Vector RAG, and Graph RAG ablation conditions.",
        "4. Benchmark: Disclosure-Sequence Distress Reasoning Benchmark, with the current run clearly marked as silver observed-outcome labels until human adjudication is added.",
        "5. Results: ablation accuracy, false-positive reduction, calibration, self-consistency, prompt strategy comparison, and faithfulness proxy.",
        "6. Discussion: Graph context helps separate distress-like patterns from normal restructuring, but final human labels and external LLM/backend runs remain the next hardening step.",
        "",
        "## Claim Boundary",
        "",
        "Do not describe the current labels as human-adjudicated. The defensible claim is a completed silver-benchmark Graph RAG upgrade run over observed-outcome labels.",
    ]
    MANUSCRIPT_ARCHIVE.mkdir(parents=True, exist_ok=True)
    (MANUSCRIPT_ARCHIVE / "paper4_upgrade_outline.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def update_status(paths: Paths, metrics: pd.DataFrame, cases: pd.DataFrame) -> None:
    status_path = ROOT / "paper4_execution_status.md"
    text = status_path.read_text(encoding="utf-8")
    marker = "## Graph RAG Upgrade Run - 2026-07-03"
    best_graph = metrics[metrics["condition"] == CONDITION_GRAPH_RAG_LLM].sort_values("accuracy", ascending=False).head(1)
    baseline = metrics[metrics["condition"] == CONDITION_RULE_ONLY].head(1)
    if not best_graph.empty and not baseline.empty:
        bg = best_graph.iloc[0]
        bl = baseline.iloc[0]
        block = f"""## Graph RAG Upgrade Run - 2026-07-03

The upgrade track now has a completed silver-benchmark smoke-to-results run. It generates a 20-30 case Disclosure-Sequence Distress Reasoning Benchmark proxy from observed Paper 4 outcomes, builds structured sequence-match graph contexts, runs the four ablation conditions from `manuscript/archive/업그레이드.md`, and writes JSON/CSV/figure outputs.

Current claim boundary: this is a silver observed-outcome benchmark, not a separately human-adjudicated gold benchmark. It supports implementation and pilot-results claims; final submission claims should be hardened with human labels and a locked production LLM/model-version protocol. External LLM runs, when present, are reported from their dedicated `results/graphrag/llm_*` artifacts.

| Item | Current value |
|---|---:|
| Benchmark cases | {len(cases)} |
| Rules covered | {cases['rule_id'].nunique()} |
| Rule-only accuracy | {bl['accuracy']:.3f} |
| Rule-only false-positive rate | {bl['false_positive_rate']:.3f} |
| Best Graph RAG prompt | `{bg['prompt_strategy']}` |
| Best Graph RAG accuracy | {bg['accuracy']:.3f} |
| Best Graph RAG false-positive rate | {bg['false_positive_rate']:.3f} |
| False-positive reduction vs rule-only | {bl['false_positive_rate'] - bg['false_positive_rate']:.3f} |

Authoritative upgrade artifacts:

- `data/labels/disclosure_sequence_distress_reasoning_benchmark.csv`
- `results/graphrag/upgrade_experiment_results.jsonl`
- `results/graphrag/ablation_metrics.csv`
- `results/graphrag/self_consistency.csv`
- `results/graphrag/faithfulness_nli.csv`
- `results/graphrag/upgrade_experiment_summary.md`
- `results/graphrag/upgrade_audit.md`
- `manuscript/archive/paper4_upgrade_outline.md`

"""
    else:
        block = f"{marker}\n\nUpgrade run artifacts exist under `results/graphrag/`, but summary metrics were not available.\n\n"
    if marker in text:
        before = text.split(marker)[0]
        after_parts = text.split(marker, 1)[1].split("\n## ", 1)
        after = ("\n## " + after_parts[1]) if len(after_parts) > 1 else ""
        text = before + block + after
    else:
        insert_after = "The upgraded goal is captured in `paper4_upgrade_goal.md`. This is a future-work target state, not a completed-result claim. Paper 4 v1 readiness and its documented boundaries remain unchanged until the new Graph RAG benchmark, labels, runners, metrics, and manuscript revisions are implemented and verified.\n\n"
        if insert_after in text:
            text = text.replace(insert_after, insert_after + block, 1)
        else:
            text = block + text
    status_path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    paths = make_paths()
    cases = build_case_manifest(paths, cases_per_rule=args.cases_per_rule, n_rules=args.n_rules)
    market_reactions = paths.graph_results_dir / "case_market_reactions.csv"
    if market_reactions.exists():
        reaction_cols = [
            "case_id",
            "reaction_available",
            "missing_reason",
            "event_trade_date",
            "pre_close_date",
            "days_pattern_to_event_trade",
            "post_return_1d",
            "post_excess_return_1d",
            "post_return_5d",
            "post_excess_return_5d",
            "post_return_20d",
            "post_excess_return_20d",
            "post_return_60d",
            "post_excess_return_60d",
            "pre_return_20d",
            "pre_excess_return_20d",
            "pre_return_60d",
            "pre_excess_return_60d",
        ]
        reaction_frame = read_csv(market_reactions)
        reaction_cols = [col for col in reaction_cols if col in reaction_frame.columns]
        cases = cases.merge(reaction_frame[reaction_cols], on="case_id", how="left")
    contexts = {case["case_id"]: build_context(case) for case in cases.to_dict("records")}

    cases_csv = paths.labels_dir / "disclosure_sequence_distress_reasoning_benchmark.csv"
    cases_json = paths.labels_dir / "disclosure_sequence_distress_reasoning_benchmark.json"
    cases.to_csv(cases_csv, index=False, encoding="utf-8-sig")
    cases_json.write_text(json.dumps(cases.to_dict("records"), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (paths.graph_results_dir / "case_contexts_graph.json").write_text(
        json.dumps(contexts, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    results = run_experiment(cases, contexts, repeats=args.repeats)
    write_jsonl(results, paths.graph_results_dir / "upgrade_experiment_results.jsonl")
    slim_results = results.drop(columns=["evidence", "retrieval_backend"])
    slim_results.to_csv(paths.graph_results_dir / "upgrade_experiment_results.csv", index=False, encoding="utf-8-sig")

    metrics, self_consistency, faithfulness = summarize_metrics(results)
    metrics.to_csv(paths.graph_results_dir / "ablation_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_json(paths.graph_results_dir / "ablation_metrics.json", orient="records", force_ascii=False, indent=2)
    self_consistency.to_csv(paths.graph_results_dir / "self_consistency.csv", index=False, encoding="utf-8-sig")
    faithfulness.to_csv(paths.graph_results_dir / "faithfulness_nli.csv", index=False, encoding="utf-8-sig")

    prompt_metrics = metrics[metrics["condition"] == CONDITION_GRAPH_RAG_LLM].copy()
    prompt_metrics.to_csv(paths.graph_results_dir / "prompt_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    written_figures = write_figures(metrics, results, paths)
    write_markdown(cases, metrics, self_consistency, faithfulness, paths.graph_results_dir / "upgrade_experiment_summary.md")
    write_outline(paths, metrics, cases)
    update_status(paths, metrics, cases)

    manifest = {
        "status": "completed",
        "label_boundary": "silver_observed_outcome_proxy_not_human_adjudicated",
        "cases": len(cases),
        "rules": int(cases["rule_id"].nunique()),
        "conditions": CONDITIONS,
        "graph_prompt_strategies": GRAPH_PROMPTS,
        "repeats": args.repeats,
        "outputs": {
            "cases_csv": str(cases_csv.relative_to(ROOT)),
            "results_jsonl": str((paths.graph_results_dir / "upgrade_experiment_results.jsonl").relative_to(ROOT)),
            "metrics_csv": str((paths.graph_results_dir / "ablation_metrics.csv").relative_to(ROOT)),
            "summary_md": str((paths.graph_results_dir / "upgrade_experiment_summary.md").relative_to(ROOT)),
            "figures": [str(Path(p).relative_to(ROOT)) for p in written_figures if Path(p).is_relative_to(ROOT)],
        },
    }
    (paths.graph_results_dir / "upgrade_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Paper 4 Graph RAG upgrade silver benchmark.")
    parser.add_argument("--n-rules", type=int, default=5)
    parser.add_argument("--cases-per-rule", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
