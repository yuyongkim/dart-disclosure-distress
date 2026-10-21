from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
MANUSCRIPT = ROOT / "manuscript" / "paper4_v4_manuscript_en.md"


@dataclass(frozen=True)
class MetricCheck:
    name: str
    source: str
    required_groups: list[list[str]]
    note: str = ""
    required_contexts: list[list[str]] = field(default_factory=list)


def read_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def thousands(value: Any) -> str:
    return f"{int(value):,}"


def pct(value: Any, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def dec(value: Any, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def metric_json(path: str, horizon: int, model: str) -> dict[str, Any]:
    rows = read_json(path)
    for row in rows:
        if int(row.get("horizon_days", -1)) == horizon and row.get("model") == model:
            return row
    raise KeyError(f"Missing metric row: {path}, horizon={horizon}, model={model}")


def metric_csv(
    path: str,
    task: str,
    horizon: int,
    model: str,
    column: str,
    feature_set: str | None = None,
) -> float:
    rows = read_csv(path)
    for row in rows:
        if row.get("task") != task:
            continue
        if int(row.get("horizon_days", "-1")) != horizon:
            continue
        if row.get("model") != model:
            continue
        if feature_set is not None and row.get("feature_set") != feature_set:
            continue
        return float(row[column])
    raise KeyError(
        f"Missing metric row: {path}, task={task}, horizon={horizon}, "
        f"model={model}, feature_set={feature_set}, column={column}"
    )


def metric_csv_any(path: str, task: str, model: str, column: str, feature_set: str | None = None) -> float:
    rows = read_csv(path)
    for row in rows:
        if row.get("task") != task or row.get("model") != model:
            continue
        if feature_set is not None and row.get("feature_set") != feature_set:
            continue
        return float(row[column])
    raise KeyError(f"Missing metric row: {path}, task={task}, model={model}, column={column}")


def bootstrap_value(horizon: int, model: str, metric: str, column: str) -> float:
    rows = read_csv("results/dart_model_bootstrap_ci.csv")
    for row in rows:
        if int(row.get("horizon_days", "-1")) != horizon:
            continue
        if row.get("model") == model and row.get("metric") == metric:
            return float(row[column])
    raise KeyError(f"Missing bootstrap row: horizon={horizon}, model={model}, metric={metric}, column={column}")


def build_checks() -> list[MetricCheck]:
    phase = read_json("data/processed/phase1_event_summary.json")
    main_panel = read_json("data/model/dart_model_panel_summary.json")
    robust_panel = read_json("data/model_robust/dart_model_panel_summary.json")
    market_collection = read_json("data/raw/market_coverage/naver_ohlcv_supplement_summary.json")
    market_fdr_gap = read_json("data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json")
    market_index = read_json("data/raw/market_index/korea_market_index_summary.json")
    market_panel = read_json("data/model_market/dart_model_panel_market_summary.json")
    robust_market_panel = read_json("data/model_market_robust/dart_model_panel_market_summary.json")
    residual_audit = read_json("results/market_residual_coverage_audit.json")
    dart_365_rf = metric_json("results/dart_model_metrics.json", 365, "combined_random_forest")
    dart_730_cal = metric_json("results/dart_model_metrics.json", 730, "combined_calibrated_random_forest")
    robust_730_rf = metric_json("results/robust/dart_model_metrics.json", 730, "combined_random_forest")
    market_365_cal = metric_json("results/market/dart_model_metrics.json", 365, "combined_market_calibrated_random_forest")
    market_730_cal = metric_json("results/market/dart_model_metrics.json", 730, "combined_market_calibrated_random_forest")
    robust_market_365_cal = metric_json(
        "results/market_robust/dart_model_metrics.json",
        365,
        "combined_market_calibrated_random_forest",
    )
    robust_market_730_cal = metric_json(
        "results/market_robust/dart_model_metrics.json",
        730,
        "combined_market_calibrated_random_forest",
    )

    dart_rows = phase["dart_events"]["rows"]
    dart_firms = phase["dart_events"]["firms"]

    return [
        MetricCheck(
            "DART event counts",
            "data/processed/phase1_event_summary.json",
            [
                [thousands(dart_rows)],
                [thousands(dart_firms)],
            ],
        ),
        MetricCheck(
            "Main DART firm-year panel",
            "data/model/dart_model_panel_summary.json",
            [
                [thousands(main_panel["rows"])],
                [thousands(main_panel["firms"])],
                [pct(main_panel["financial_match_rate"])],
            ],
        ),
        MetricCheck(
            "Robust DART firm-year panel",
            "data/model_robust/dart_model_panel_summary.json",
            [
                [thousands(robust_panel["rows"])],
                [thousands(robust_panel["firms"])],
            ],
        ),
        MetricCheck(
            "DART market-price extension coverage",
            "data/raw/market_coverage/naver_ohlcv_supplement_summary.json; data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json; data/raw/market_index/korea_market_index_summary.json; data/model_market/dart_model_panel_market_summary.json",
            [
                [thousands(market_collection["output_rows_total"])],
                [thousands(market_collection["fetched_symbols"])],
                [thousands(market_collection["target_symbols"])],
                [thousands(market_collection["no_row_symbols"])],
                [thousands(market_fdr_gap["output_rows_total"])],
                [thousands(market_fdr_gap["fetched_symbols"])],
                [thousands(market_fdr_gap["no_row_symbols"])],
                [thousands(market_index["rows"])],
                [thousands(market_panel["ohlcv_rows_loaded"])],
                [thousands(market_panel["ohlcv_symbols_loaded"])],
                [thousands(market_panel["market_feature_count"])],
                [pct(market_panel["price_record_rate"])],
                [pct(market_panel["recent_price_record_rate"])],
                [pct(market_panel["market_index_coverage_rate"])],
                [pct(robust_market_panel["price_record_rate"])],
            ],
        ),
        MetricCheck(
            "DART residual OHLCV coverage impact",
            "results/market_residual_coverage_audit.json",
            [
                [thousands(residual_audit["fdr_residual_symbols"])],
                [thousands(residual_audit["main"]["residual_rows"])],
                [pct(residual_audit["main"]["residual_row_share"], 2)],
                [
                    f"{int(residual_audit['main']['residual_target_365d_rows'])}/"
                    f"{int(residual_audit['main']['residual_target_730d_rows'])}"
                ],
            ],
            required_contexts=[["residual", "473", "2.37%", "3/5"]],
            note="Residual no-row OHLCV symbols should be disclosed as a quantified coverage limitation.",
        ),
        MetricCheck(
            "DART 365-day combined random forest headline",
            "results/dart_model_metrics.json",
            [
                [dec(dart_365_rf["roc_auc"], 3), dec(dart_365_rf["roc_auc"], 4)],
                [dec(dart_365_rf["top_10pct"]["lift"], 2)],
            ],
        ),
        MetricCheck(
            "DART 730-day calibrated random forest headline",
            "results/dart_model_metrics.json",
            [
                [dec(dart_730_cal["roc_auc"], 3), dec(dart_730_cal["roc_auc"], 4)],
                [dec(dart_730_cal["top_10pct"]["lift"], 2)],
            ],
        ),
        MetricCheck(
            "DART bootstrap intervals",
            "results/dart_model_bootstrap_ci.csv",
            [
                [dec(bootstrap_value(365, "combined_random_forest", "roc_auc", "mean"), 4)],
                [dec(bootstrap_value(365, "combined_random_forest", "roc_auc", "ci_low"), 4)],
                [dec(bootstrap_value(365, "combined_random_forest", "roc_auc", "ci_high"), 4)],
                [dec(bootstrap_value(730, "combined_calibrated_random_forest", "roc_auc", "mean"), 4)],
                [dec(bootstrap_value(730, "combined_calibrated_random_forest", "roc_auc", "ci_low"), 4)],
                [dec(bootstrap_value(730, "combined_calibrated_random_forest", "roc_auc", "ci_high"), 4)],
            ],
        ),
        MetricCheck(
            "Robust 730-day DART random forest headline",
            "results/robust/dart_model_metrics.json",
            [
                [dec(robust_730_rf["roc_auc"], 3), dec(robust_730_rf["roc_auc"], 4)],
                [dec(robust_730_rf["top_10pct"]["lift"], 2)],
            ],
        ),
        MetricCheck(
            "DART market extension headline metrics",
            "results/market/dart_model_metrics.json; results/market_robust/dart_model_metrics.json",
            [
                [dec(market_365_cal["roc_auc"], 3), dec(market_365_cal["roc_auc"], 4)],
                [dec(market_365_cal["average_precision"], 4)],
                [dec(market_365_cal["top_10pct"]["lift"], 2)],
                [dec(market_730_cal["roc_auc"], 4)],
                [dec(market_730_cal["average_precision"], 4)],
                [dec(market_730_cal["top_10pct"]["lift"], 2)],
                [dec(robust_market_365_cal["roc_auc"], 4)],
                [dec(robust_market_730_cal["top_10pct"]["lift"], 2)],
            ],
            "Market extension metrics should be described as a coverage-expanded extension/control layer, not as causal price-reaction evidence.",
        ),
    ]


def find_lines(text: str, token: str) -> list[int]:
    return [idx for idx, line in enumerate(text.splitlines(), start=1) if token in line]


def context_lines(text: str, context: list[str]) -> list[int]:
    return [
        idx
        for idx, line in enumerate(text.splitlines(), start=1)
        if all(token in line for token in context)
    ]


def evaluate_check(text: str, check: MetricCheck) -> dict[str, Any]:
    group_results: list[dict[str, Any]] = []
    for group in check.required_groups:
        matched_tokens = [token for token in group if token in text]
        lines = sorted({line for token in matched_tokens for line in find_lines(text, token)})
        group_results.append(
            {
                "accepted_values": group,
                "matched": bool(matched_tokens),
                "matched_values": matched_tokens,
                "line_numbers": lines,
            }
        )
    context_results: list[dict[str, Any]] = []
    for context in check.required_contexts:
        lines = context_lines(text, context)
        context_results.append(
            {
                "required_same_line": context,
                "matched": bool(lines),
                "line_numbers": lines,
            }
        )
    missing = [group for group in group_results if not group["matched"]]
    missing_contexts = [context for context in context_results if not context["matched"]]
    return {
        "name": check.name,
        "source": check.source,
        "status": "passed" if not missing and not missing_contexts else "missing_or_stale_metric",
        "groups": group_results,
        "contexts": context_results,
        "missing_groups": [group["accepted_values"] for group in missing],
        "missing_contexts": [context["required_same_line"] for context in missing_contexts],
        "note": check.note,
    }


def build_audit() -> dict[str, Any]:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    checks = [evaluate_check(text, check) for check in build_checks()]
    missing = [check for check in checks if check["status"] != "passed"]
    return {
        "status": "passed" if not missing else "metric_mismatches_found",
        "manuscript": str(MANUSCRIPT),
        "check_count": len(checks),
        "missing_count": len(missing),
        "checks": checks,
        "guardrail": "This audit checks selected manuscript headline numbers against current local result artifacts; it does not replace substantive methods review.",
    }


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# Manuscript Metric Consistency Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Checks | {audit['check_count']} |",
        f"| Missing/stale checks | {audit['missing_count']} |",
        "",
        "## Checks",
        "",
        "| Check | Status | Source | Missing accepted values | Missing required context | Evidence lines |",
        "|---|---|---|---|---|---|",
    ]
    for check in audit["checks"]:
        missing = "; ".join(", ".join(group) for group in check["missing_groups"]) or "-"
        missing_context = "; ".join(", ".join(group) for group in check.get("missing_contexts", [])) or "-"
        evidence_lines = sorted(
            {
                line
                for group in check["groups"]
                for line in group["line_numbers"]
            }
            | {
                line
                for context in check.get("contexts", [])
                for line in context["line_numbers"]
            }
        )
        evidence = ", ".join(str(line) for line in evidence_lines[:12]) if evidence_lines else "-"
        if len(evidence_lines) > 12:
            evidence += ", ..."
        lines.append(
            "| "
            f"{escape_md(check['name'])} | "
            f"`{check['status']}` | "
            f"{escape_md(check['source'])} | "
            f"{escape_md(missing)} | "
            f"{escape_md(missing_context)} | "
            f"{escape_md(evidence)} |"
        )
    lines.extend(["", "## Guardrail", "", audit["guardrail"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = build_audit()
    write_markdown(audit, ROOT / "results" / "manuscript_metric_audit.md")
    (ROOT / "results" / "manuscript_metric_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": audit["status"], "missing_count": audit["missing_count"]}, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
