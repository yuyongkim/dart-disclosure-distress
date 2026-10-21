from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audit_forbidden_tokens import (
    BILINGUAL_EXTRA_FORBIDDEN_TOKENS,
    BILINGUAL_FORBIDDEN_TOKENS as FORBIDDEN_TOKENS,
    FORBIDDEN_TOKENS as SHARED_FORBIDDEN_TOKENS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Manuscript:
    label: str
    path: str


@dataclass(frozen=True)
class BilingualCheck:
    name: str
    source: str
    required_groups: dict[str, list[list[str]]]
    required_contexts: dict[str, list[list[str]]] = field(default_factory=dict)
    note: str = ""


MANUSCRIPTS = [
    Manuscript("en", "manuscript/paper4_v4_manuscript_en.md"),
    Manuscript("ko", "manuscript/paper4_v4_manuscript_ko.md"),
]


def read_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def metric_row(path: str, horizon: int, model: str) -> dict[str, Any]:
    rows = read_json(path)
    for row in rows:
        if int(row.get("horizon_days", -1)) == horizon and row.get("model") == model:
            return row
    raise KeyError(f"Missing metric row: {path} horizon={horizon} model={model}")


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def fmt_dec(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_numbers(text: str, token: str) -> list[int]:
    return [idx for idx, line in enumerate(text.splitlines(), start=1) if token in line]


def context_lines(text: str, context: list[str]) -> list[int]:
    return [
        idx
        for idx, line in enumerate(text.splitlines(), start=1)
        if all(token in line for token in context)
    ]


def build_checks() -> list[BilingualCheck]:
    phase = read_json("data/processed/phase1_event_summary.json")
    main_panel = read_json("data/model/dart_model_panel_summary.json")
    naver = read_json("data/raw/market_coverage/naver_ohlcv_supplement_summary.json")
    fdr_gap = read_json("data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json")
    market_index = read_json("data/raw/market_index/korea_market_index_summary.json")
    market_panel = read_json("data/model_market/dart_model_panel_market_summary.json")
    residual = read_json("results/market_residual_coverage_audit.json")
    alignment_single = read_json("results/alignment_single_expert_readiness.json")
    market_365 = metric_row(
        "results/market/dart_model_metrics.json",
        365,
        "combined_market_calibrated_random_forest",
    )
    market_730 = metric_row(
        "results/market/dart_model_metrics.json",
        730,
        "combined_market_calibrated_random_forest",
    )
    robust_market_730 = metric_row(
        "results/market_robust/dart_model_metrics.json",
        730,
        "combined_market_calibrated_random_forest",
    )

    residual_pair = (
        f"{int(residual['main']['residual_target_365d_rows'])}/"
        f"{int(residual['main']['residual_target_730d_rows'])}"
    )

    common_market_groups = [
        [fmt_int(naver["output_rows_total"])],
        [fmt_int(fdr_gap["output_rows_total"])],
        [fmt_int(fdr_gap["no_row_symbols"])],
        [fmt_int(market_index["rows"])],
        [fmt_int(market_panel["ohlcv_rows_loaded"])],
        [fmt_pct(market_panel["price_record_rate"])],
        [fmt_dec(market_365["roc_auc"])],
        [fmt_dec(market_365["average_precision"])],
        [fmt_dec(market_365["top_10pct"]["lift"], 2)],
        [fmt_dec(market_730["roc_auc"])],
        [fmt_dec(market_730["average_precision"])],
        [fmt_dec(market_730["top_10pct"]["lift"], 2)],
        [fmt_dec(robust_market_730["top_10pct"]["lift"], 2)],
    ]
    residual_groups = [
        [fmt_int(residual["fdr_residual_symbols"])],
        [fmt_int(residual["main"]["residual_rows"])],
        [fmt_pct(residual["main"]["residual_row_share"], 2)],
        [residual_pair],
    ]
    return [
        BilingualCheck(
            "DART panel and event scope",
            "data/processed/phase1_event_summary.json; data/model/dart_model_panel_summary.json",
            {
                "en": [
                    [fmt_int(phase["dart_events"]["rows"])],
                    [fmt_int(main_panel["rows"])],
                    [fmt_pct(main_panel["financial_match_rate"])],
                ],
                "ko": [
                    [fmt_int(phase["dart_events"]["rows"])],
                    [fmt_int(main_panel["rows"])],
                    [fmt_pct(main_panel["financial_match_rate"])],
                ],
            },
        ),
        BilingualCheck(
            "Market extension headline values",
            "market coverage, index, and model artifacts",
            {"en": common_market_groups, "ko": common_market_groups},
        ),
        BilingualCheck(
            "Residual OHLCV limitation",
            "results/market_residual_coverage_audit.json",
            {"en": residual_groups, "ko": residual_groups},
            required_contexts={
                "en": [["residual", "473", "2.37%", "3/5"]],
                "ko": [["\uc794\uc5ec", "473", "2.37%", "3/5"]],
            },
            note="The residual no-row OHLCV limitation must be carried in both final manuscripts.",
        ),
        BilingualCheck(
            "Event-study and causal-price guardrail",
            "manuscript guardrail",
            {
                "en": [["not a security-level exchange-specific event-study abnormal-return design"]],
                "ko": [["\uc0ac\uac74\uc5f0\uad6c \ube44\uc815\uc0c1\uc218\uc775\ub960 \uc124\uacc4\ub294 \uc544\ub2c8\ub2e4"]],
            },
        ),
        BilingualCheck(
            "RQ1 human-label boundary",
            "alignment readiness artifacts",
            {
                "en": [["human-labeled alignment metrics"], ["outside the empirical claims"]],
                "ko": [["human_label"], ["Cohen"], [str(alignment_single["human_labeled_rows"])]],
            },
        ),
        BilingualCheck(
            "EDINET v2 boundary",
            "results/edinet_collection_status.json",
            {
                "en": [["Additional cross-market taxonomy transfer"], ["outside the empirical claims"]],
                "ko": [["EDINET"], ["v2"], ["\uc218\uc9d1\ub41c EDINET \ubb38\uc11c \ud589\uc774 \uc5c6\ub2e4"]],
            },
        ),
    ]


def evaluate_text(label: str, path: Path, text: str, checks: list[BilingualCheck]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for check in checks:
        group_results = []
        for group in check.required_groups.get(label, []):
            matched = [token for token in group if token in text]
            group_results.append(
                {
                    "accepted_values": group,
                    "matched": bool(matched),
                    "matched_values": matched,
                    "line_numbers": sorted({line for token in matched for line in line_numbers(text, token)}),
                }
            )
        context_results = []
        for context in check.required_contexts.get(label, []):
            lines = context_lines(text, context)
            context_results.append(
                {
                    "required_same_line": context,
                    "matched": bool(lines),
                    "line_numbers": lines,
                }
            )
        missing_groups = [group["accepted_values"] for group in group_results if not group["matched"]]
        missing_contexts = [context["required_same_line"] for context in context_results if not context["matched"]]
        rows.append(
            {
                "manuscript": label,
                "path": str(path.relative_to(ROOT)),
                "check": check.name,
                "source": check.source,
                "status": "passed" if not missing_groups and not missing_contexts else "failed",
                "missing_groups": missing_groups,
                "missing_contexts": missing_contexts,
                "groups": group_results,
                "contexts": context_results,
                "note": check.note,
            }
        )
    return rows


def build_audit() -> dict[str, Any]:
    checks = build_checks()
    manuscript_rows = []
    file_rows = []
    for manuscript in MANUSCRIPTS:
        path = ROOT / manuscript.path
        exists = path.exists()
        text = path.read_text(encoding="utf-8") if exists else ""
        forbidden_hits = [
            {"token": token, "line_numbers": line_numbers(text, token)}
            for token in FORBIDDEN_TOKENS
            if token in text
        ]
        file_rows.append(
            {
                "label": manuscript.label,
                "path": manuscript.path,
                "exists": exists,
                "bytes": int(path.stat().st_size) if exists else 0,
                "sha256": sha256(path) if exists else None,
                "forbidden_hits": forbidden_hits,
                "forbidden_status": "passed" if not forbidden_hits else "failed",
            }
        )
        if exists:
            manuscript_rows.extend(evaluate_text(manuscript.label, path, text, checks))
        else:
            manuscript_rows.append(
                {
                    "manuscript": manuscript.label,
                    "path": manuscript.path,
                    "check": "file_exists",
                    "source": manuscript.path,
                    "status": "failed",
                    "missing_groups": [[manuscript.path]],
                    "missing_contexts": [],
                    "groups": [],
                    "contexts": [],
                    "note": "Missing manuscript file.",
                }
            )

    failures = [row for row in manuscript_rows if row["status"] != "passed"]
    forbidden_failures = [row for row in file_rows if row["forbidden_status"] != "passed"]
    return {
        "status": "pass" if not failures and not forbidden_failures else "fail",
        "manuscript_count": len(MANUSCRIPTS),
        "check_count": len(checks),
        "failure_count": len(failures) + len(forbidden_failures),
        "forbidden_token_count": len(FORBIDDEN_TOKENS),
        "shared_forbidden_token_count": len(SHARED_FORBIDDEN_TOKENS),
        "bilingual_extra_forbidden_token_count": len(BILINGUAL_EXTRA_FORBIDDEN_TOKENS),
        "forbidden_token_policy": "src/audit/audit_forbidden_tokens.py",
        "files": file_rows,
        "checks": manuscript_rows,
        "guardrail": "English and Korean final manuscripts must carry the same DART and market-price headline values and documented limitation boundaries.",
    }


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# Bilingual Manuscript Consistency Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Manuscripts checked | {audit['manuscript_count']} |",
        f"| Check families | {audit['check_count']} |",
        f"| Failures | {audit['failure_count']} |",
        f"| Forbidden stale/overclaim patterns | {audit['forbidden_token_count']} |",
        f"| Shared forbidden patterns | {audit['shared_forbidden_token_count']} |",
        f"| Bilingual-only forbidden patterns | {audit['bilingual_extra_forbidden_token_count']} |",
        f"| Forbidden-token policy | `{audit['forbidden_token_policy']}` |",
        "",
        "## File Checks",
        "",
        "| Manuscript | Exists | Bytes | Forbidden hits |",
        "|---|---:|---:|---|",
    ]
    for row in audit["files"]:
        forbidden = "; ".join(hit["token"] for hit in row["forbidden_hits"]) or "-"
        lines.append(
            "| "
            f"`{row['path']}` | "
            f"{row['exists']} | "
            f"{row['bytes']} | "
            f"{escape_md(forbidden)} |"
        )

    lines.extend(
        [
            "",
            "## Bilingual Checks",
            "",
            "| Manuscript | Check | Status | Missing values | Missing context | Evidence lines |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in audit["checks"]:
        missing = "; ".join(", ".join(group) for group in row["missing_groups"]) or "-"
        missing_context = "; ".join(", ".join(group) for group in row["missing_contexts"]) or "-"
        evidence_lines = sorted(
            {
                line
                for group in row.get("groups", [])
                for line in group.get("line_numbers", [])
            }
            | {
                line
                for context in row.get("contexts", [])
                for line in context.get("line_numbers", [])
            }
        )
        evidence = ", ".join(str(line) for line in evidence_lines[:12]) if evidence_lines else "-"
        if len(evidence_lines) > 12:
            evidence += ", ..."
        lines.append(
            "| "
            f"`{row['manuscript']}` | "
            f"{escape_md(row['check'])} | "
            f"`{row['status']}` | "
            f"{escape_md(missing)} | "
            f"{escape_md(missing_context)} | "
            f"{escape_md(evidence)} |"
        )

    lines.extend(["", "## Guardrail", "", audit["guardrail"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    (RESULTS / "bilingual_manuscript_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(audit, RESULTS / "bilingual_manuscript_audit.md")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "manuscript_count": audit["manuscript_count"],
                "check_count": audit["check_count"],
                "failure_count": audit["failure_count"],
            },
            indent=2,
        )
    )
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
