from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("Paper 4")
VALID_DECISIONS = {"MATCH", "PARTIAL", "NO_MATCH"}


SYSTEM_PROMPT = (
    "You are a regulatory disclosure expert comparing Korean DART major-event "
    "disclosures with U.S. SEC Form 8-K items. Decide whether the two events "
    "describe substantially the same legal or economic disclosure trigger. "
    "Return concise JSON only."
)


def load_taxonomy(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, low_memory=False)
    return {
        str(row.taxonomy_code): {
            "label": row.taxonomy_label,
            "severity": int(row.severity),
            "dart_examples": row.dart_examples,
            "sec_examples": row.sec_examples,
            "edinet_examples": row.edinet_examples,
        }
        for row in frame.itertuples(index=False)
    }


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").strip()


def prompt_for_row(row: pd.Series, taxonomy: dict[str, dict[str, Any]]) -> dict[str, str]:
    code = clean_text(row.get("taxonomy_code", ""))
    tax = taxonomy.get(code, {})
    user = f"""
Compare the DART and SEC disclosure events below.

Common taxonomy candidate:
- code: {code}
- label: {tax.get("label", "")}
- DART examples: {tax.get("dart_examples", "")}
- SEC examples: {tax.get("sec_examples", "")}

DART event:
- event: {clean_text(row.get("dart_event"))}
- subtype: {clean_text(row.get("dart_subtype"))}
- observed count: {clean_text(row.get("dart_event_count"))}

SEC event:
- event: {clean_text(row.get("sec_event"))}
- subtype: {clean_text(row.get("sec_subtype"))}
- observed count: {clean_text(row.get("sec_event_count"))}

Decision labels:
- MATCH: substantially the same legal/economic disclosure trigger.
- PARTIAL: related risk domain, but different trigger, specificity, or timing.
- NO_MATCH: not the same event except for broad adverse/corporate-event similarity.

Return JSON with exactly these keys:
{{"decision":"MATCH|PARTIAL|NO_MATCH","confidence":0.0-1.0,"rationale":"one concise sentence"}}
""".strip()
    return {"system": SYSTEM_PROMPT, "user": user}


def same_taxonomy_code(code: str) -> bool:
    return bool(re.fullmatch(r"T\d{2}|T99", code.strip()))


def lexical_overlap(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"[A-Za-z0-9가-힣]+", a.lower()))
    tokens_b = set(re.findall(r"[A-Za-z0-9가-힣]+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def offline_decision(row: pd.Series) -> dict[str, Any]:
    code = clean_text(row.get("taxonomy_code", ""))
    dart_event = clean_text(row.get("dart_event", ""))
    sec_event = clean_text(row.get("sec_event", ""))
    dart_subtype = clean_text(row.get("dart_subtype", ""))
    sec_subtype = clean_text(row.get("sec_subtype", ""))

    overlap = lexical_overlap(f"{dart_event} {dart_subtype}", f"{sec_event} {sec_subtype}")
    if not same_taxonomy_code(code):
        return {
            "decision": "NO_MATCH",
            "confidence": 0.82,
            "rationale": "The candidate links different supervisory taxonomy codes and is retained as a negative seed.",
            "mode": "offline_taxonomy_fallback",
            "lexical_overlap": overlap,
        }

    exact_pairs = {
        ("listing_or_trading_status", "delisting_notice"),
        ("default", "debt_acceleration_trigger"),
        ("paid_in_capital_increase", "unregistered_securities_sale"),
        ("asset_or_equity_transfer", "asset_acquisition_or_disposition"),
        ("director_or_officer_change", "director_or_officer_change"),
    }
    if (dart_subtype, sec_subtype) in exact_pairs:
        return {
            "decision": "MATCH",
            "confidence": 0.76,
            "rationale": "The source-specific subtypes describe closely corresponding disclosure triggers within the same taxonomy cell.",
            "mode": "offline_taxonomy_fallback",
            "lexical_overlap": overlap,
        }

    if code in {"T02", "T03"}:
        decision, confidence = "MATCH", 0.68
        rationale = "The event pair maps to the same high-severity distress taxonomy cell, but should still be human-checked."
    elif code in {"T05", "T06"}:
        decision, confidence = "PARTIAL", 0.61
        rationale = "The events share a broad supervisory risk domain but may differ in the precise legal trigger."
    else:
        decision, confidence = "PARTIAL", 0.55
        rationale = "The deterministic taxonomy agrees, but local data are insufficient for a stronger automated claim."
    return {
        "decision": decision,
        "confidence": confidence,
        "rationale": rationale,
        "mode": "offline_taxonomy_fallback",
        "lexical_overlap": overlap,
    }


def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    data = json.loads(match.group(0))
    decision = str(data.get("decision", "")).strip().upper()
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid decision: {decision}")
    confidence = float(data.get("confidence", 0.0))
    return {
        "decision": decision,
        "confidence": max(0.0, min(1.0, confidence)),
        "rationale": clean_text(data.get("rationale", "")),
        "mode": "anthropic_messages_api",
    }


def anthropic_decision(
    prompt: dict[str, str],
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": prompt["system"],
        "messages": [{"role": "user", "content": prompt["user"]}],
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    text_blocks = body.get("content", [])
    text = "\n".join(block.get("text", "") for block in text_blocks if block.get("type") == "text")
    parsed = extract_json(text)
    parsed["raw_response"] = text
    return parsed


def run_alignment(
    seed_path: Path,
    taxonomy_path: Path,
    out_csv: Path,
    out_json: Path,
    prompt_jsonl: Path,
    summary_md: Path,
    mode: str,
    model: str,
    limit: int | None,
    sleep_seconds: float,
) -> dict[str, Any]:
    seed = pd.read_csv(seed_path, low_memory=False)
    taxonomy = load_taxonomy(taxonomy_path)
    rows = seed.head(limit).copy() if limit else seed.copy()
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    effective_mode = mode
    if mode == "anthropic" and not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for --mode anthropic")

    prompt_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with prompt_jsonl.open("w", encoding="utf-8") as prompt_file:
        for row in rows.to_dict(orient="records"):
            series = pd.Series(row)
            prompt = prompt_for_row(series, taxonomy)
            prompt_file.write(
                json.dumps(
                    {
                        "candidate_rank": int(row["candidate_rank"]),
                        "system": prompt["system"],
                        "user": prompt["user"],
                        "expected_schema": {
                            "decision": "MATCH|PARTIAL|NO_MATCH",
                            "confidence": "float between 0 and 1",
                            "rationale": "one concise sentence",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            try:
                if effective_mode == "anthropic":
                    decision = anthropic_decision(
                        prompt=prompt,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                        max_tokens=240,
                        timeout=60,
                    )
                    if sleep_seconds:
                        time.sleep(sleep_seconds)
                else:
                    decision = offline_decision(series)
                status = "ok"
                error = ""
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError) as exc:
                decision = offline_decision(series)
                status = "fallback_after_error"
                error = repr(exc)

            results.append(
                {
                    **row,
                    "llm_decision": decision["decision"],
                    "llm_confidence": decision["confidence"],
                    "llm_rationale": decision["rationale"],
                    "llm_mode": decision.get("mode", effective_mode),
                    "lexical_overlap": decision.get("lexical_overlap"),
                    "api_status": status,
                    "api_error": error,
                }
            )

    result_df = pd.DataFrame(results)
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    summary = build_summary(result_df, seed_path, taxonomy_path, prompt_jsonl, out_csv, effective_mode, model)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_md(summary, summary_md)
    return summary


def build_summary(
    result_df: pd.DataFrame,
    seed_path: Path,
    taxonomy_path: Path,
    prompt_jsonl: Path,
    out_csv: Path,
    mode: str,
    model: str,
) -> dict[str, Any]:
    decision_counts = {str(k): int(v) for k, v in result_df["llm_decision"].value_counts().items()}
    by_code = (
        result_df.groupby(["taxonomy_code", "llm_decision"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["taxonomy_code", "llm_decision"])
    )
    return {
        "status": "completed_prompt_export_and_alignment_candidates",
        "mode": mode,
        "model": model if mode == "anthropic" else "offline_taxonomy_fallback",
        "rows": int(len(result_df)),
        "decision_counts": decision_counts,
        "mean_confidence": float(result_df["llm_confidence"].mean()) if not result_df.empty else None,
        "seed_path": str(seed_path),
        "taxonomy_path": str(taxonomy_path),
        "prompt_jsonl": str(prompt_jsonl),
        "results_csv": str(out_csv),
        "by_taxonomy_decision": by_code.to_dict(orient="records"),
        "claim_guardrail": "These are model/offline candidate judgments only. RQ1 Precision@1 and MRR still require human labels in manual_alignment_seed_100.csv.",
    }


def write_summary_md(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# LLM-Ready Alignment Run",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This run creates prompts and candidate alignment judgments for DART-SEC event pairs. It does not replace human labels and must not be reported as RQ1 Precision@1/MRR.",
        "",
        "## Run Metadata",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Mode | `{summary['mode']}` |",
        f"| Model | `{summary['model']}` |",
        f"| Rows | {summary['rows']} |",
        f"| Mean confidence | {summary['mean_confidence']:.4f} |",
        f"| Prompt JSONL | `{summary['prompt_jsonl']}` |",
        f"| Results CSV | `{summary['results_csv']}` |",
        "",
        "## Candidate Decision Counts",
        "",
        "| Decision | Count |",
        "|---|---:|",
    ]
    for decision, count in sorted(summary["decision_counts"].items()):
        lines.append(f"| `{decision}` | {count} |")
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            summary["claim_guardrail"],
            "",
            "To report RQ1, fill `human_label` in `data/processed/manual_alignment_seed_100.csv`, then run `src/align/evaluate_alignment.py`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-path", default=str(ROOT / "data" / "processed" / "manual_alignment_seed_100.csv"))
    parser.add_argument("--taxonomy-path", default=str(ROOT / "data" / "processed" / "regulatory_event_taxonomy.csv"))
    parser.add_argument("--out-csv", default=str(ROOT / "data" / "processed" / "llm_alignment_results.csv"))
    parser.add_argument("--out-json", default=str(ROOT / "results" / "llm_alignment_run.json"))
    parser.add_argument("--prompt-jsonl", default=str(ROOT / "data" / "processed" / "llm_alignment_prompts.jsonl"))
    parser.add_argument("--summary-md", default=str(ROOT / "results" / "llm_alignment_summary.md"))
    parser.add_argument("--mode", choices=["offline", "anthropic"], default="offline")
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    args = parser.parse_args()

    summary = run_alignment(
        seed_path=Path(args.seed_path),
        taxonomy_path=Path(args.taxonomy_path),
        out_csv=Path(args.out_csv),
        out_json=Path(args.out_json),
        prompt_jsonl=Path(args.prompt_jsonl),
        summary_md=Path(args.summary_md),
        mode=args.mode,
        model=args.model,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
