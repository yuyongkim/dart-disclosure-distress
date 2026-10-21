from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
OUT_DIR = ROOT / "results"

CALCULATION_COLUMNS = [
    "section",
    "claim_id",
    "item",
    "metric",
    "value",
    "display",
    "unit",
    "source_data",
    "source_code",
    "calculation",
    "verification",
]
MODEL_COLUMNS = [
    "outcome",
    "horizon_days",
    "model",
    "test_positives",
    "roc_auc",
    "average_precision",
    "brier_score",
    "top_10pct_recall",
    "top_10pct_lift",
    "lift_formula",
    "lift_verification",
    "source_data",
    "source_code",
]
MANUSCRIPT_CHECK_COLUMNS = ["manuscript", "token", "verification"]
VERIFICATION_COLUMNS = [
    "check_name",
    "formula",
    "source_data",
    "source_code",
    "source_value",
    "recomputed_value",
    "delta",
    "tolerance",
    "status",
]
SOURCE_COLUMNS = ["path", "type", "exists", "size_bytes", "sha256"]


@dataclass(frozen=True)
class CalcRow:
    section: str
    claim_id: str
    item: str
    metric: str
    value: float | int | str
    display: str
    unit: str
    source_data: str
    source_code: str
    calculation: str
    verification: str


def read_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def fmt_dec(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def metric(metrics: list[dict[str, Any]], horizon: int, model: str) -> dict[str, Any]:
    for row in metrics:
        if int(row.get("horizon_days", 0)) == horizon and row.get("model") == model:
            return row
    raise KeyError(f"Missing metric row: horizon={horizon}, model={model}")


def check_close(name: str, actual: float, expected: float, tolerance: float = 1e-9) -> str:
    diff = abs(float(actual) - float(expected))
    return "PASS" if diff <= tolerance else f"FAIL {name}: actual={actual} expected={expected} diff={diff}"


def source_exists(path: str) -> bool:
    return (ROOT / path).exists()


def source_size(path: str) -> int:
    source_path = ROOT / path
    return int(source_path.stat().st_size) if source_path.exists() else 0


def source_sha256(path: str) -> str:
    source_path = ROOT / path
    return hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.exists() else ""


def build_rows() -> tuple[list[CalcRow], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    phase = read_json("data/processed/phase1_event_summary.json")
    main_panel = read_json("data/model/dart_model_panel_summary.json")
    robust_panel = read_json("data/model_robust/dart_model_panel_summary.json")
    naver = read_json("data/raw/market_coverage/naver_ohlcv_supplement_summary.json")
    fdr_gap = read_json("data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json")
    market_index = read_json("data/raw/market_index/korea_market_index_summary.json")
    market_panel = read_json("data/model_market/dart_model_panel_market_summary.json")
    robust_market_panel = read_json("data/model_market_robust/dart_model_panel_market_summary.json")
    residual_audit = read_json("results/market_residual_coverage_audit.json")
    main_market_metrics = read_json("results/market/dart_model_metrics.json")
    robust_market_metrics = read_json("results/market_robust/dart_model_metrics.json")

    checks: list[str] = []
    verification_checks: list[dict[str, Any]] = []
    rows: list[CalcRow] = []

    def verify(
        name: str,
        formula: str,
        source_value: float | int | str,
        recomputed_value: float | int | str,
        source_data: str,
        source_code: str,
        tolerance: float = 0.0,
    ) -> str:
        if isinstance(source_value, (int, float)) and isinstance(recomputed_value, (int, float)):
            delta: float | str = abs(float(source_value) - float(recomputed_value))
            status = "PASS" if float(delta) <= tolerance else "FAIL"
        else:
            delta = "" if str(source_value) == str(recomputed_value) else "mismatch"
            status = "PASS" if str(source_value) == str(recomputed_value) else "FAIL"
        verification_checks.append(
            {
                "check_name": name,
                "formula": formula,
                "source_value": source_value,
                "recomputed_value": recomputed_value,
                "delta": delta,
                "tolerance": tolerance,
                "status": status,
                "source_data": source_data,
                "source_code": source_code,
            }
        )
        if status != "PASS":
            checks.append(f"FAIL {name}: source={source_value} recomputed={recomputed_value} delta={delta}")
        return status

    def add(
        section: str,
        claim_id: str,
        item: str,
        metric_name: str,
        value: float | int | str,
        display: str,
        unit: str,
        source_data: str,
        source_code: str,
        calculation: str,
        verification: str = "PASS",
    ) -> None:
        rows.append(
            CalcRow(
                section=section,
                claim_id=claim_id,
                item=item,
                metric=metric_name,
                value=value,
                display=display,
                unit=unit,
                source_data=source_data,
                source_code=source_code,
                calculation=calculation,
                verification=verification,
            )
        )
        if not verification.startswith("PASS"):
            checks.append(verification)

    # Coverage and data construction claims.
    fdr_residual_expected = int(fdr_gap["target_symbols"]) - int(fdr_gap["fetched_symbols"]) - int(fdr_gap["error_symbols"])
    fdr_residual_check = verify(
        "fdr_gap_residual_symbols",
        "target_symbols - fetched_symbols - error_symbols",
        int(fdr_gap["no_row_symbols"]),
        fdr_residual_expected,
        "data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json",
        "src/collect/market_fdr_gap_collector.py",
    )
    residual_status_check = verify(
        "market_residual_audit_status",
        "status == pass",
        "pass",
        str(residual_audit["status"]),
        "results/market_residual_coverage_audit.json",
        "src/audit/market_residual_coverage_audit.py",
    )
    residual_symbol_check = verify(
        "residual_audit_symbol_count_matches_fdr",
        "residual_audit.fdr_residual_symbols == fdr_gap.no_row_symbols",
        int(residual_audit["fdr_residual_symbols"]),
        int(fdr_gap["no_row_symbols"]),
        "results/market_residual_coverage_audit.json; data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json",
        "src/audit/market_residual_coverage_audit.py",
    )
    residual_main_missing_check = verify(
        "main_residual_missing_rows_bounded",
        "residual_missing_price_rows <= missing_price_rows_total",
        "PASS",
        "PASS"
        if int(residual_audit["main"]["residual_missing_price_rows"]) <= int(residual_audit["main"]["missing_price_rows_total"])
        else "FAIL",
        "results/market_residual_coverage_audit.json",
        "src/audit/market_residual_coverage_audit.py",
    )
    residual_robust_missing_check = verify(
        "robust_residual_missing_rows_bounded",
        "residual_missing_price_rows <= missing_price_rows_total",
        "PASS",
        "PASS"
        if int(residual_audit["robust"]["residual_missing_price_rows"]) <= int(residual_audit["robust"]["missing_price_rows_total"])
        else "FAIL",
        "results/market_residual_coverage_audit.json",
        "src/audit/market_residual_coverage_audit.py",
    )
    naver_plus_fdr_rows = int(naver["output_rows_total"]) + int(fdr_gap["output_rows_total"])
    robust_loaded_supp_check = verify(
        "robust_supplemental_loaded_rows",
        "Naver output_rows_total + FDR output_rows_total",
        int(robust_market_panel["supplemental_ohlcv_rows_loaded"]),
        naver_plus_fdr_rows,
        "data/model_market_robust/dart_model_panel_market_summary.json; data/raw/market_coverage/naver_ohlcv_supplement_summary.json; data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json",
        "src/model/build_market_features.py",
    )
    verify(
        "main_residual_row_share",
        "main.residual_rows / main.panel_rows",
        float(residual_audit["main"]["residual_row_share"]),
        int(residual_audit["main"]["residual_rows"]) / int(residual_audit["main"]["panel_rows"]),
        "results/market_residual_coverage_audit.json",
        "src/audit/market_residual_coverage_audit.py",
        tolerance=1e-12,
    )
    verify(
        "robust_residual_row_share",
        "robust.residual_rows / robust.panel_rows",
        float(residual_audit["robust"]["residual_row_share"]),
        int(residual_audit["robust"]["residual_rows"]) / int(residual_audit["robust"]["panel_rows"]),
        "results/market_residual_coverage_audit.json",
        "src/audit/market_residual_coverage_audit.py",
        tolerance=1e-12,
    )

    add(
        "coverage",
        "dart_events",
        "DART event layer",
        "event_rows",
        int(phase["dart_events"]["rows"]),
        fmt_int(phase["dart_events"]["rows"]),
        "rows",
        "data/processed/phase1_event_summary.json",
        "src/prepare/build_phase1_events.py",
        "count(DART normalized event rows)",
    )
    add(
        "coverage",
        "main_panel",
        "Main DART firm-year panel",
        "panel_rows",
        int(main_panel["rows"]),
        fmt_int(main_panel["rows"]),
        "firm-years",
        "data/model/dart_model_panel_summary.json",
        "src/model/build_dart_model_dataset.py",
        "count(model panel rows after pre-outcome anchor filtering)",
    )
    add(
        "coverage",
        "naver_supplement",
        "Naver fchart supplemental OHLCV",
        "output_rows_total",
        int(naver["output_rows_total"]),
        fmt_int(naver["output_rows_total"]),
        "OHLCV rows",
        "data/raw/market_coverage/naver_ohlcv_supplement_summary.json",
        "src/collect/market_coverage_collector.py",
        "count(saved supplemental rows for deficient symbols)",
    )
    add(
        "coverage",
        "fdr_gap_supplement",
        "FinanceDataReader gap OHLCV",
        "output_rows_total",
        int(fdr_gap["output_rows_total"]),
        fmt_int(fdr_gap["output_rows_total"]),
        "OHLCV rows",
        "data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json",
        "src/collect/market_fdr_gap_collector.py",
        "count(saved rows for Naver no-row symbols)",
    )
    add(
        "coverage",
        "fdr_gap_residual",
        "FinanceDataReader gap OHLCV",
        "residual_no_row_symbols",
        int(fdr_gap["no_row_symbols"]),
        fmt_int(fdr_gap["no_row_symbols"]),
        "symbols",
        "data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json",
        "src/collect/market_fdr_gap_collector.py",
        "target_symbols - fetched_symbols - error_symbols",
        fdr_residual_check,
    )
    add(
        "coverage",
        "market_residual_audit",
        "FDR residual OHLCV symbols",
        "residual_symbols",
        int(residual_audit["fdr_residual_symbols"]),
        fmt_int(residual_audit["fdr_residual_symbols"]),
        "symbols",
        "results/market_residual_coverage_audit.json",
        "src/audit/market_residual_coverage_audit.py",
        "count(FDR gap no_rows symbols after Naver and FDR passes)",
        residual_symbol_check,
    )
    add(
        "coverage",
        "market_index",
        "KOSPI/KOSDAQ benchmark",
        "index_rows",
        int(market_index["rows"]),
        fmt_int(market_index["rows"]),
        "index rows",
        "data/raw/market_index/korea_market_index_summary.json",
        "src/collect/market_index_collector.py",
        "count(KS11 and KQ11 daily index rows)",
    )
    add(
        "coverage",
        "main_market_panel",
        "Main market panel",
        "ohlcv_rows_loaded",
        int(market_panel["ohlcv_rows_loaded"]),
        fmt_int(market_panel["ohlcv_rows_loaded"]),
        "deduplicated OHLCV rows",
        "data/model_market/dart_model_panel_market_summary.json",
        "src/model/build_market_features.py",
        "base SQLite rows + supplemental CSV rows, normalized and de-duplicated by symbol-date",
    )
    add(
        "coverage",
        "market_residual_audit",
        "Main market residual coverage impact",
        "residual_rows",
        int(residual_audit["main"]["residual_rows"]),
        fmt_int(residual_audit["main"]["residual_rows"]),
        "firm-year anchors",
        "results/market_residual_coverage_audit.json",
        "src/audit/market_residual_coverage_audit.py",
        "count(main market-panel anchors whose security_id is in FDR gap no_rows symbols)",
        residual_status_check,
    )
    add(
        "coverage",
        "market_residual_audit",
        "Main market residual coverage impact",
        "residual_row_share",
        float(residual_audit["main"]["residual_row_share"]),
        fmt_pct(residual_audit["main"]["residual_row_share"], digits=2),
        "share",
        "results/market_residual_coverage_audit.json",
        "src/audit/market_residual_coverage_audit.py",
        "residual_rows / panel_rows",
        residual_main_missing_check,
    )
    add(
        "coverage",
        "market_residual_audit",
        "Main market residual coverage impact",
        "residual_target_365d_rows",
        int(residual_audit["main"]["residual_target_365d_rows"]),
        fmt_int(residual_audit["main"]["residual_target_365d_rows"]),
        "positive anchors",
        "results/market_residual_coverage_audit.json",
        "src/audit/market_residual_coverage_audit.py",
        "count(target_365d == true among residual-symbol main-panel anchors)",
    )
    add(
        "coverage",
        "market_residual_audit",
        "Main market residual coverage impact",
        "residual_target_730d_rows",
        int(residual_audit["main"]["residual_target_730d_rows"]),
        fmt_int(residual_audit["main"]["residual_target_730d_rows"]),
        "positive anchors",
        "results/market_residual_coverage_audit.json",
        "src/audit/market_residual_coverage_audit.py",
        "count(target_730d == true among residual-symbol main-panel anchors)",
    )
    add(
        "coverage",
        "main_market_panel",
        "Main market panel",
        "price_record_rate",
        float(market_panel["price_record_rate"]),
        fmt_pct(market_panel["price_record_rate"]),
        "share",
        "data/model_market/dart_model_panel_market_summary.json",
        "src/model/build_market_features.py",
        "mean(mkt_has_price_record)",
    )
    add(
        "coverage",
        "main_market_panel",
        "Main market panel",
        "recent_price_record_rate",
        float(market_panel["recent_price_record_rate"]),
        fmt_pct(market_panel["recent_price_record_rate"]),
        "share",
        "data/model_market/dart_model_panel_market_summary.json",
        "src/model/build_market_features.py",
        "mean(mkt_has_price_record == 1 and mkt_stale_30d_flag == 0)",
    )
    add(
        "coverage",
        "robust_market_panel",
        "Robust market panel",
        "supplemental_ohlcv_rows_loaded",
        int(robust_market_panel["supplemental_ohlcv_rows_loaded"]),
        fmt_int(robust_market_panel["supplemental_ohlcv_rows_loaded"]),
        "supplemental OHLCV rows",
        "data/model_market_robust/dart_model_panel_market_summary.json",
        "src/model/build_market_features.py",
        "Naver output_rows_total + FDR output_rows_total for the robust symbol universe",
        robust_loaded_supp_check,
    )
    add(
        "coverage",
        "market_residual_audit",
        "Robust market residual coverage impact",
        "residual_rows",
        int(residual_audit["robust"]["residual_rows"]),
        fmt_int(residual_audit["robust"]["residual_rows"]),
        "firm-year anchors",
        "results/market_residual_coverage_audit.json",
        "src/audit/market_residual_coverage_audit.py",
        "count(robust market-panel anchors whose security_id is in FDR gap no_rows symbols)",
    )
    add(
        "coverage",
        "market_residual_audit",
        "Robust market residual coverage impact",
        "residual_row_share",
        float(residual_audit["robust"]["residual_row_share"]),
        fmt_pct(residual_audit["robust"]["residual_row_share"], digits=2),
        "share",
        "results/market_residual_coverage_audit.json",
        "src/audit/market_residual_coverage_audit.py",
        "residual_rows / panel_rows",
        residual_robust_missing_check,
    )

    # Model headline rows. Lift is recalculated from precision / test positive rate.
    model_rows: list[dict[str, Any]] = []
    headline_specs = [
        ("main", 365, "combined_market_calibrated_random_forest", main_market_metrics),
        ("main", 730, "combined_market_calibrated_random_forest", main_market_metrics),
        ("robust", 365, "combined_market_calibrated_random_forest", robust_market_metrics),
        ("robust", 730, "combined_market_calibrated_random_forest", robust_market_metrics),
        ("main", 365, "market_logit", main_market_metrics),
        ("main", 365, "sequence_market_logit", main_market_metrics),
    ]
    for outcome, horizon, model_name, metrics in headline_specs:
        row = metric(metrics, horizon, model_name)
        lift_expected = float(row["top_10pct"]["precision"]) / float(row["test_positive_rate"])
        lift_check = verify(
            f"{outcome}_{horizon}_{model_name}_lift",
            "top_10pct.precision / test_positive_rate",
            float(row["top_10pct"]["lift"]),
            lift_expected,
            "results/market/dart_model_metrics.json" if outcome == "main" else "results/market_robust/dart_model_metrics.json",
            "src/model/train_dart_baselines.py",
            tolerance=1e-9,
        )
        source_dir = "results/market" if outcome == "main" else "results/market_robust"
        model_rows.append(
            {
                "outcome": outcome,
                "horizon_days": horizon,
                "model": model_name,
                "test_positives": int(row["test_positives"]),
                "roc_auc": float(row["roc_auc"]),
                "average_precision": float(row["average_precision"]),
                "brier_score": float(row["brier_score"]),
                "top_10pct_recall": float(row["top_10pct"]["recall"]),
                "top_10pct_lift": float(row["top_10pct"]["lift"]),
                "lift_formula": "top_10pct.precision / test_positive_rate",
                "lift_verification": lift_check,
                "source_data": f"{source_dir}/dart_model_metrics.json",
                "source_code": "src/model/train_dart_baselines.py",
            }
        )
        for metric_name, value, display in [
            ("roc_auc", row["roc_auc"], fmt_dec(row["roc_auc"])),
            ("average_precision", row["average_precision"], fmt_dec(row["average_precision"])),
            ("brier_score", row["brier_score"], fmt_dec(row["brier_score"])),
            ("top_10pct_recall", row["top_10pct"]["recall"], fmt_dec(row["top_10pct"]["recall"])),
            ("top_10pct_lift", row["top_10pct"]["lift"], f"{float(row['top_10pct']['lift']):.2f}"),
        ]:
            add(
                "model",
                f"{outcome}_{horizon}_{model_name}",
                f"{outcome} {horizon} {model_name}",
                metric_name,
                float(value),
                display,
                "model metric",
                f"{source_dir}/dart_model_metrics.json",
                "src/model/train_dart_baselines.py",
                "computed in train_dart_baselines.py; lift is independently rechecked as precision / test_positive_rate",
                lift_check if metric_name == "top_10pct_lift" else "PASS",
            )

    manuscript_tokens = [
        "411,423",
        "69",
        "5,443,762",
        "0.9049",
        "0.1482",
        "7.45",
        "0.8956",
        "0.1531",
        "7.38",
        "473",
        "2.37%",
        "3/5",
    ]
    manuscript_checks: list[dict[str, Any]] = []
    for manuscript_path in ["manuscript/paper4_v4_manuscript_en.md", "manuscript/paper4_v4_manuscript_ko.md"]:
        text = (ROOT / manuscript_path).read_text(encoding="utf-8")
        for token in manuscript_tokens:
            ok = token in text
            check = "PASS" if ok else f"FAIL manuscript missing token {token}: {manuscript_path}"
            manuscript_checks.append({"manuscript": manuscript_path, "token": token, "verification": check})
            if not ok:
                checks.append(check)

    source_files = sorted(
        {
            path
            for row in rows
            for value in [row.source_data, row.source_code]
            for path in split_source_paths(value)
        }
        | {
            path
            for row in model_rows + verification_checks
            for value in [row.get("source_data"), row.get("source_code")]
            for path in split_source_paths(value)
        }
    )
    source_rows = [{"path": path, "exists": source_exists(path)} for path in source_files]
    for row in source_rows:
        if not row["exists"]:
            checks.append(f"FAIL missing source file: {row['path']}")

    return rows, model_rows, manuscript_checks, verification_checks, checks


def write_markdown(
    calc_df: pd.DataFrame,
    model_df: pd.DataFrame,
    manuscript_df: pd.DataFrame,
    verification_df: pd.DataFrame,
    source_df: pd.DataFrame,
    checks: list[str],
    path: Path,
) -> None:
    status = "pass" if not checks else "fail"
    lines = [
        "# Paper 4 Calculation Sheet",
        "",
        f"Status: `{status}`",
        "",
        "This sheet records the source data, source code, formulas, manuscript checks, source-file existence, and recomputation checks used for headline Paper 4 values. The XLSX version carries the same core evidence on separate tabs plus an audit tab.",
        "",
        "## Headline Model Metrics",
        "",
        model_df[
            [
                "outcome",
                "horizon_days",
                "model",
                "test_positives",
                "roc_auc",
                "average_precision",
                "brier_score",
                "top_10pct_recall",
                "top_10pct_lift",
                "lift_formula",
                "source_data",
                "source_code",
                "lift_verification",
            ]
        ].to_markdown(index=False),
        "",
        "## Coverage Calculations",
        "",
        calc_df.loc[
            calc_df["section"].eq("coverage"),
            ["claim_id", "item", "metric", "display", "source_data", "source_code", "calculation", "verification"],
        ].to_markdown(index=False),
        "",
        "## Model Metric Calculations",
        "",
        calc_df.loc[
            calc_df["section"].eq("model"),
            ["claim_id", "item", "metric", "display", "source_data", "source_code", "calculation", "verification"],
        ].to_markdown(index=False),
        "",
        "## Manuscript Token Checks",
        "",
        manuscript_df[["manuscript", "token", "verification"]].to_markdown(index=False),
        "",
        "## Verification Checks",
        "",
        verification_df[
            [
                "check_name",
                "formula",
                "source_data",
                "source_code",
                "source_value",
                "recomputed_value",
                "delta",
                "tolerance",
                "status",
            ]
        ].to_markdown(index=False),
        "",
        "## Source Inventory",
        "",
        source_df[["path", "type", "exists", "size_bytes", "sha256"]].to_markdown(index=False),
        "",
        "## Verification",
        "",
    ]
    if checks:
        lines.extend(f"- {check}" for check in checks)
    else:
        lines.append("- All calculation-sheet consistency checks passed.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_source_paths(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def collect_source_references(*dataframes: pd.DataFrame) -> set[tuple[str, str]]:
    references: set[tuple[str, str]] = set()
    for df in dataframes:
        for column, source_type in [("source_data", "data"), ("source_code", "code")]:
            if column not in df.columns:
                continue
            for value in df[column].dropna():
                for path in split_source_paths(value):
                    references.add((path, source_type))
    return references


def build_source_df(calc_df: pd.DataFrame, model_df: pd.DataFrame, verification_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "path": path,
                "type": source_type,
                "exists": source_exists(path),
                "size_bytes": source_size(path),
                "sha256": source_sha256(path),
            }
            for path, source_type in sorted(collect_source_references(calc_df, model_df, verification_df))
        ]
    )


def nonblank_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].fillna("").astype(str).str.strip().ne("").sum())


def build_traceability_summary(
    calc_df: pd.DataFrame,
    model_df: pd.DataFrame,
    verification_df: pd.DataFrame,
    source_df: pd.DataFrame,
) -> dict[str, Any]:
    source_type_counts = source_df["type"].value_counts().to_dict() if "type" in source_df.columns else {}
    hash_ok = source_df.apply(
        lambda row: bool(row.get("exists")) and int(row.get("size_bytes", 0)) > 0 and len(str(row.get("sha256", ""))) == 64,
        axis=1,
    )
    summary = {
        "source_rows": int(len(source_df)),
        "source_data_rows": int(source_type_counts.get("data", 0)),
        "source_code_rows": int(source_type_counts.get("code", 0)),
        "source_existing_rows": int(source_df["exists"].astype(bool).sum()) if "exists" in source_df.columns else 0,
        "source_hash_rows": int(hash_ok.sum()),
        "source_missing_rows": int((~source_df["exists"].astype(bool)).sum()) if "exists" in source_df.columns else 0,
        "calculation_rows_with_source_data": nonblank_count(calc_df, "source_data"),
        "calculation_rows_with_source_code": nonblank_count(calc_df, "source_code"),
        "calculation_rows_with_formula": nonblank_count(calc_df, "calculation"),
        "model_rows_with_source_data": nonblank_count(model_df, "source_data"),
        "model_rows_with_source_code": nonblank_count(model_df, "source_code"),
        "model_rows_with_formula": nonblank_count(model_df, "lift_formula"),
        "verification_rows_with_source_data": nonblank_count(verification_df, "source_data"),
        "verification_rows_with_source_code": nonblank_count(verification_df, "source_code"),
        "verification_rows_with_formula": nonblank_count(verification_df, "formula"),
    }
    summary["calculation_traceability_ok"] = (
        summary["source_rows"] > 0
        and summary["source_rows"] == summary["source_existing_rows"] == summary["source_hash_rows"]
        and summary["source_data_rows"] > 0
        and summary["source_code_rows"] > 0
        and summary["calculation_rows_with_source_data"] == len(calc_df)
        and summary["calculation_rows_with_source_code"] == len(calc_df)
        and summary["calculation_rows_with_formula"] == len(calc_df)
        and summary["model_rows_with_source_data"] == len(model_df)
        and summary["model_rows_with_source_code"] == len(model_df)
        and summary["model_rows_with_formula"] == len(model_df)
        and summary["verification_rows_with_source_data"] == len(verification_df)
        and summary["verification_rows_with_source_code"] == len(verification_df)
        and summary["verification_rows_with_formula"] == len(verification_df)
    )
    return summary


def missing_columns(df: pd.DataFrame, expected: list[str]) -> list[str]:
    return [column for column in expected if column not in df.columns]


def sheet_check(sheet: str, df: pd.DataFrame, expected_rows: int, expected_columns: list[str]) -> dict[str, Any]:
    missing = missing_columns(df, expected_columns)
    return {
        "sheet": sheet,
        "rows": int(len(df)),
        "expected_rows": int(expected_rows),
        "missing_columns": missing,
        "status": "pass" if len(df) == expected_rows and not missing else "fail",
    }


def verify_output_integrity(
    calc_df: pd.DataFrame,
    model_df: pd.DataFrame,
    manuscript_df: pd.DataFrame,
    verification_df: pd.DataFrame,
    source_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    csv_path = OUT_DIR / "calculation_sheet.csv"
    xlsx_path = OUT_DIR / "calculation_sheet.xlsx"
    md_path = OUT_DIR / "calculation_sheet.md"
    expected_sources = collect_source_references(calc_df, model_df, verification_df)
    actual_sources = {
        (str(row["path"]), str(row["type"]))
        for row in source_df[["path", "type"]].dropna().to_dict(orient="records")
    }

    if not csv_path.exists():
        failures.append("missing calculation_sheet.csv")
    else:
        csv_df = pd.read_csv(csv_path)
        checks.append(sheet_check("csv:calculation_sheet", csv_df, len(calc_df), CALCULATION_COLUMNS))

    missing_sources = sorted(expected_sources - actual_sources)
    extra_sources = sorted(actual_sources - expected_sources)
    missing_source_files = sorted(
        (path, source_type)
        for path, source_type, exists in source_df[["path", "type", "exists"]].dropna().itertuples(index=False, name=None)
        if not bool(exists)
    )
    checks.append(
        {
            "sheet": "sources:coverage",
            "rows": int(len(actual_sources)),
            "expected_rows": int(len(expected_sources)),
            "missing_columns": [f"{source_type}:{path}" for path, source_type in missing_sources]
            + [f"extra:{source_type}:{path}" for path, source_type in extra_sources],
            "status": "pass" if not missing_sources and not extra_sources else "fail",
        }
    )
    checks.append(
        {
            "sheet": "sources:existence",
            "rows": int(len(actual_sources) - len(missing_source_files)),
            "expected_rows": int(len(actual_sources)),
            "missing_columns": [f"{source_type}:{path}" for path, source_type in missing_source_files],
            "status": "pass" if not missing_source_files else "fail",
        }
    )
    bad_source_hashes = sorted(
        (str(row["path"]), str(row["type"]))
        for row in source_df[["path", "type", "exists", "size_bytes", "sha256"]].dropna().to_dict(orient="records")
        if bool(row["exists"]) and (int(row["size_bytes"]) <= 0 or len(str(row["sha256"])) != 64)
    )
    checks.append(
        {
            "sheet": "sources:hashes",
            "rows": int(len(actual_sources) - len(bad_source_hashes)),
            "expected_rows": int(len(actual_sources)),
            "missing_columns": [f"{source_type}:{path}" for path, source_type in bad_source_hashes],
            "status": "pass" if not bad_source_hashes else "fail",
        }
    )

    if not xlsx_path.exists():
        failures.append("missing calculation_sheet.xlsx")
    else:
        xlsx = pd.ExcelFile(xlsx_path)
        xlsx_frames: dict[str, pd.DataFrame] = {}
        required_sheets = {
            "calculations": (len(calc_df), CALCULATION_COLUMNS),
            "model_metrics": (len(model_df), MODEL_COLUMNS),
            "manuscript_checks": (len(manuscript_df), MANUSCRIPT_CHECK_COLUMNS),
            "verification_checks": (len(verification_df), VERIFICATION_COLUMNS),
            "sources": (len(source_df), SOURCE_COLUMNS),
            "audit": (
                1,
                [
                    "status",
                    "calculation_rows",
                    "model_rows",
                    "manuscript_checks",
                    "verification_checks",
                    "source_rows",
                    "source_data_rows",
                    "source_code_rows",
                    "source_hash_rows",
                    "source_missing_rows",
                    "calculation_traceability_ok",
                    "output_integrity",
                    "output_integrity_failures",
                    "output_integrity_check_count",
                    "output_integrity_failure_count",
                ],
            ),
        }
        for sheet, (expected_rows, expected_columns) in required_sheets.items():
            if sheet not in xlsx.sheet_names:
                failures.append(f"missing XLSX sheet: {sheet}")
                checks.append(
                    {
                        "sheet": f"xlsx:{sheet}",
                        "rows": 0,
                        "expected_rows": int(expected_rows),
                        "missing_columns": expected_columns,
                        "status": "fail",
                    }
                )
                continue
            sheet_df = pd.read_excel(xlsx, sheet_name=sheet)
            xlsx_frames[sheet] = sheet_df
            checks.append(sheet_check(f"xlsx:{sheet}", sheet_df, expected_rows, expected_columns))

        pass_status_failures: list[str] = []
        pass_status_total = 0
        for sheet, column in [
            ("calculations", "verification"),
            ("model_metrics", "lift_verification"),
            ("manuscript_checks", "verification"),
            ("verification_checks", "status"),
        ]:
            sheet_df = xlsx_frames.get(sheet)
            if sheet_df is None:
                pass_status_failures.append(f"missing sheet for PASS status check: {sheet}")
                continue
            if column not in sheet_df.columns:
                pass_status_failures.append(f"missing PASS status column: {sheet}.{column}")
                continue
            pass_status_total += int(len(sheet_df))
            bad_values = sheet_df.loc[~sheet_df[column].astype(str).str.startswith("PASS"), column].astype(str).tolist()
            pass_status_failures.extend(f"{sheet}.{column}={value}" for value in bad_values)
        checks.append(
            {
                "sheet": "xlsx:pass_statuses",
                "rows": int(pass_status_total - len(pass_status_failures)),
                "expected_rows": int(pass_status_total),
                "missing_columns": pass_status_failures,
                "status": "pass" if not pass_status_failures else "fail",
            }
        )

    if not md_path.exists():
        failures.append("missing calculation_sheet.md")
    else:
        markdown = md_path.read_text(encoding="utf-8")
        markdown_tokens = [
            "## Headline Model Metrics",
            "## Coverage Calculations",
            "## Verification Checks",
            "source_data",
            "source_code",
            "calculation",
            "lift_formula",
            "## Model Metric Calculations",
            "## Manuscript Token Checks",
            "## Source Inventory",
            "exists",
            "size_bytes",
            "sha256",
            "All calculation-sheet consistency checks passed.",
        ]
        missing_tokens = [token for token in markdown_tokens if token not in markdown]
        checks.append(
            {
                "sheet": "markdown:calculation_sheet",
                "rows": int(markdown.count("\n")),
                "expected_rows": 1,
                "missing_columns": missing_tokens,
                "status": "pass" if not missing_tokens else "fail",
            }
        )

    for check in checks:
        if check["status"] != "pass":
            failures.append(
                f"{check['sheet']} integrity failed: rows={check['rows']}/{check['expected_rows']} missing={check['missing_columns']}"
            )
    return checks, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate calculated rows against existing outputs without rewriting package-included files.",
    )
    args = parser.parse_args()
    if not args.check_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, model_rows, manuscript_checks, verification_checks, checks = build_rows()
    calc_df = pd.DataFrame([row.__dict__ for row in rows])
    model_df = pd.DataFrame(model_rows)
    manuscript_df = pd.DataFrame(manuscript_checks)
    verification_df = pd.DataFrame(verification_checks)
    source_df = build_source_df(calc_df, model_df, verification_df)
    traceability_summary = build_traceability_summary(calc_df, model_df, verification_df, source_df)
    audit = {
        "status": "pass" if not checks else "fail",
        "calculation_rows": int(len(calc_df)),
        "model_rows": int(len(model_df)),
        "manuscript_checks": int(len(manuscript_df)),
        "verification_checks": int(len(verification_df)),
        **traceability_summary,
        "failures": checks,
        "output_integrity": [],
        "output_integrity_failures": [],
        "output_integrity_check_count": 0,
        "output_integrity_failure_count": 0,
        "outputs": {
            "xlsx": str(OUT_DIR / "calculation_sheet.xlsx"),
            "csv": str(OUT_DIR / "calculation_sheet.csv"),
            "markdown": str(OUT_DIR / "calculation_sheet.md"),
            "audit_json": str(OUT_DIR / "calculation_sheet_audit.json"),
        },
    }

    if args.check_only:
        output_integrity, output_failures = verify_output_integrity(
            calc_df,
            model_df,
            manuscript_df,
            verification_df,
            source_df,
        )
        audit["mode"] = "check_only"
        audit["output_integrity"] = output_integrity
        audit["output_integrity_failures"] = output_failures
        audit["output_integrity_check_count"] = len(output_integrity)
        audit["output_integrity_failure_count"] = len(output_failures)
        audit["status"] = "pass" if not checks and not output_failures else "fail"
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        if audit["status"] != "pass":
            raise SystemExit(1)
        return

    calc_df.to_csv(OUT_DIR / "calculation_sheet.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT_DIR / "calculation_sheet.xlsx", engine="openpyxl") as writer:
        calc_df.to_excel(writer, sheet_name="calculations", index=False)
        model_df.to_excel(writer, sheet_name="model_metrics", index=False)
        manuscript_df.to_excel(writer, sheet_name="manuscript_checks", index=False)
        verification_df.to_excel(writer, sheet_name="verification_checks", index=False)
        source_df.to_excel(writer, sheet_name="sources", index=False)
        pd.DataFrame([audit]).to_excel(writer, sheet_name="audit", index=False)
    write_markdown(calc_df, model_df, manuscript_df, verification_df, source_df, checks, OUT_DIR / "calculation_sheet.md")
    output_integrity, output_failures = verify_output_integrity(calc_df, model_df, manuscript_df, verification_df, source_df)
    audit["output_integrity"] = output_integrity
    audit["output_integrity_failures"] = output_failures
    audit["output_integrity_check_count"] = len(output_integrity)
    audit["output_integrity_failure_count"] = len(output_failures)
    audit["status"] = "pass" if not checks and not output_failures else "fail"
    with pd.ExcelWriter(OUT_DIR / "calculation_sheet.xlsx", engine="openpyxl") as writer:
        calc_df.to_excel(writer, sheet_name="calculations", index=False)
        model_df.to_excel(writer, sheet_name="model_metrics", index=False)
        manuscript_df.to_excel(writer, sheet_name="manuscript_checks", index=False)
        verification_df.to_excel(writer, sheet_name="verification_checks", index=False)
        source_df.to_excel(writer, sheet_name="sources", index=False)
        pd.DataFrame([audit]).to_excel(writer, sheet_name="audit", index=False)
    final_output_integrity, final_output_failures = verify_output_integrity(
        calc_df,
        model_df,
        manuscript_df,
        verification_df,
        source_df,
    )
    audit["output_integrity"] = final_output_integrity
    audit["output_integrity_failures"] = final_output_failures
    audit["output_integrity_check_count"] = len(final_output_integrity)
    audit["output_integrity_failure_count"] = len(final_output_failures)
    audit["status"] = "pass" if not checks and not final_output_failures else "fail"
    (OUT_DIR / "calculation_sheet_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))

    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
