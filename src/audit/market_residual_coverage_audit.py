from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
RESULTS = ROOT / "results"
AUDIT_DIR = ROOT / "data" / "audit"


def read_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def norm_symbol(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)")[0].str.zfill(6)


def pct(part: int | float, whole: int | float) -> float | None:
    return float(part) / float(whole) if whole else None


def summarize_panel(panel_path: str, residual_symbols: set[str], label: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(ROOT / panel_path)
    df = df.copy()
    df["security_id_norm"] = norm_symbol(df["security_id"])
    df["is_residual_symbol"] = df["security_id_norm"].isin(residual_symbols)
    df["has_price_record_bool"] = df["mkt_has_price_record"].astype(float).eq(1.0)
    df["recent_price_record_bool"] = df["has_price_record_bool"] & df["mkt_stale_30d_flag"].astype(float).eq(0.0)
    df["missing_price_bool"] = ~df["has_price_record_bool"]
    df["target_365d_bool"] = df["target_365d"].astype(bool)
    df["target_730d_bool"] = df["target_730d"].astype(bool)

    residual = df[df["is_residual_symbol"]].copy()
    missing_all = df[df["missing_price_bool"]].copy()
    residual_missing = residual[residual["missing_price_bool"]].copy()

    residual_symbol_groups = residual.groupby("security_id_norm", dropna=False)
    residual_symbols_with_any_price = int(
        residual_symbol_groups["has_price_record_bool"].any().sum()
    ) if not residual.empty else 0
    residual_symbols_all_missing = int(
        (~residual_symbol_groups["has_price_record_bool"].any()).sum()
    ) if not residual.empty else 0

    summary = {
        "label": label,
        "panel_path": panel_path,
        "panel_rows": int(len(df)),
        "panel_firms": int(df["firm_id"].nunique()),
        "panel_symbols": int(df["security_id_norm"].nunique()),
        "missing_price_rows_total": int(missing_all.shape[0]),
        "missing_price_rate_total": pct(int(missing_all.shape[0]), int(len(df))),
        "target_365d_total": int(df["target_365d_bool"].sum()),
        "target_730d_total": int(df["target_730d_bool"].sum()),
        "residual_symbols_in_panel": int(residual["security_id_norm"].nunique()),
        "residual_rows": int(residual.shape[0]),
        "residual_row_share": pct(int(residual.shape[0]), int(len(df))),
        "residual_missing_price_rows": int(residual_missing.shape[0]),
        "residual_missing_price_share_of_panel": pct(int(residual_missing.shape[0]), int(len(df))),
        "residual_share_of_all_missing_price_rows": pct(int(residual_missing.shape[0]), int(missing_all.shape[0])),
        "residual_rows_with_any_price": int(residual["has_price_record_bool"].sum()),
        "residual_rows_with_recent_price": int(residual["recent_price_record_bool"].sum()),
        "residual_symbols_with_any_price": residual_symbols_with_any_price,
        "residual_symbols_all_missing": residual_symbols_all_missing,
        "residual_target_365d_rows": int(residual["target_365d_bool"].sum()),
        "residual_target_730d_rows": int(residual["target_730d_bool"].sum()),
        "residual_target_365d_share_of_panel_positives": pct(
            int(residual["target_365d_bool"].sum()),
            int(df["target_365d_bool"].sum()),
        ),
        "residual_target_730d_share_of_panel_positives": pct(
            int(residual["target_730d_bool"].sum()),
            int(df["target_730d_bool"].sum()),
        ),
    }

    by_year = (
        df.groupby("anchor_year", dropna=False)
        .agg(
            panel_rows=("firm_id", "size"),
            panel_missing_price_rows=("missing_price_bool", "sum"),
            residual_rows=("is_residual_symbol", "sum"),
            residual_missing_price_rows=("missing_price_bool", lambda s: int(s[df.loc[s.index, "is_residual_symbol"]].sum())),
            residual_target_365d=("target_365d_bool", lambda s: int(s[df.loc[s.index, "is_residual_symbol"]].sum())),
            residual_target_730d=("target_730d_bool", lambda s: int(s[df.loc[s.index, "is_residual_symbol"]].sum())),
        )
        .reset_index()
    )
    by_year["label"] = label
    by_year["residual_row_share"] = by_year.apply(lambda r: pct(r["residual_rows"], r["panel_rows"]), axis=1)

    if residual.empty:
        by_symbol = pd.DataFrame()
    else:
        by_symbol = (
            residual.groupby("security_id_norm", dropna=False)
            .agg(
                firm_ids=("firm_id", lambda s: ";".join(sorted(set(s.astype(str))))),
                firm_names=("firm_name", lambda s: ";".join(sorted(set(s.astype(str))))),
                anchor_rows=("firm_id", "size"),
                anchor_year_min=("anchor_year", "min"),
                anchor_year_max=("anchor_year", "max"),
                price_record_rows=("has_price_record_bool", "sum"),
                recent_price_rows=("recent_price_record_bool", "sum"),
                missing_price_rows=("missing_price_bool", "sum"),
                target_365d_rows=("target_365d_bool", "sum"),
                target_730d_rows=("target_730d_bool", "sum"),
                first_anchor_date=("anchor_date", "min"),
                last_anchor_date=("anchor_date", "max"),
            )
            .reset_index()
        )
        by_symbol["label"] = label

    return summary, by_year, by_symbol


def build_audit() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    fdr_status = pd.read_csv(ROOT / "data/raw/market_coverage/fdr_ohlcv_gap_supplement_status.csv")
    fdr_summary = read_json("data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json")
    fdr_status["symbol_norm"] = norm_symbol(fdr_status["symbol"])
    residual_symbols = set(fdr_status.loc[fdr_status["status"].eq("no_rows"), "symbol_norm"])

    main_summary, main_year, main_symbol = summarize_panel(
        "data/model_market/dart_model_panel_market.parquet",
        residual_symbols,
        "main",
    )
    robust_summary, robust_year, robust_symbol = summarize_panel(
        "data/model_market_robust/dart_model_panel_market.parquet",
        residual_symbols,
        "robust",
    )

    expected_residual = int(fdr_summary["no_row_symbols"])
    checks = []
    checks.append(
        {
            "name": "fdr_residual_symbol_count",
            "status": "passed" if len(residual_symbols) == expected_residual else "failed",
            "actual": len(residual_symbols),
            "expected": expected_residual,
        }
    )
    checks.append(
        {
            "name": "fdr_no_error_symbols",
            "status": "passed" if int(fdr_summary["error_symbols"]) == 0 else "failed",
            "actual": int(fdr_summary["error_symbols"]),
            "expected": 0,
        }
    )

    symbol_impact = pd.concat([main_symbol, robust_symbol], ignore_index=True)
    year_impact = pd.concat([main_year, robust_year], ignore_index=True)
    status = "pass" if all(row["status"] == "passed" for row in checks) else "fail"
    audit = {
        "status": status,
        "fdr_status_path": "data/raw/market_coverage/fdr_ohlcv_gap_supplement_status.csv",
        "fdr_residual_symbols": len(residual_symbols),
        "fdr_expected_residual_symbols": expected_residual,
        "residual_symbol_list": sorted(residual_symbols),
        "main": main_summary,
        "robust": robust_summary,
        "checks": checks,
        "outputs": {
            "markdown": "Paper 4/results/market_residual_coverage_audit.md",
            "json": "Paper 4/results/market_residual_coverage_audit.json",
            "symbol_impact_csv": "Paper 4/data/audit/market_residual_symbol_impact.csv",
            "year_impact_csv": "Paper 4/data/audit/market_residual_year_impact.csv",
        },
        "guardrail": "The FDR residual no-row symbols are quantified as a bounded market-feature coverage limitation; this audit does not impute missing prices or claim complete security-level price coverage.",
    }
    return audit, symbol_impact, year_impact


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# Market Residual Coverage Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| FDR residual no-row symbols | {audit['fdr_residual_symbols']} |",
        f"| Expected residual symbols from FDR summary | {audit['fdr_expected_residual_symbols']} |",
        "",
        "## Panel Impact",
        "",
        "| Panel | Residual symbols in panel | Residual rows | Residual row share | Residual missing-price rows | Share of all missing-price rows | Residual 365d positives | Residual 730d positives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ["main", "robust"]:
        row = audit[label]
        lines.append(
            "| "
            f"{label} | "
            f"{row['residual_symbols_in_panel']} | "
            f"{row['residual_rows']} | "
            f"{fmt_pct(row['residual_row_share'])} | "
            f"{row['residual_missing_price_rows']} | "
            f"{fmt_pct(row['residual_share_of_all_missing_price_rows'])} | "
            f"{row['residual_target_365d_rows']} | "
            f"{row['residual_target_730d_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The residual symbols are the FDR gap-pass `no_rows` symbols after the Naver fchart supplemental collection also returned no rows.",
            "- The audit quantifies how those symbols map into the model panels and how many target-positive anchors are affected.",
            "- Missing/stale market information remains represented through explicit `mkt_missing_price_flag`, `mkt_has_price_record`, and `mkt_stale_30d_flag` features rather than silently dropping anchors.",
            "- This does not turn the OHLCV extension into complete security-level price coverage.",
            "",
            "## Outputs",
            "",
            f"- `{audit['outputs']['symbol_impact_csv']}`",
            f"- `{audit['outputs']['year_impact_csv']}`",
            "",
            "## Checks",
            "",
            "| Check | Status | Actual | Expected |",
            "|---|---|---:|---:|",
        ]
    )
    for check in audit["checks"]:
        lines.append(f"| {check['name']} | `{check['status']}` | {check['actual']} | {check['expected']} |")
    lines.extend(["", "## Guardrail", "", audit["guardrail"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit, symbol_impact, year_impact = build_audit()
    symbol_impact.to_csv(AUDIT_DIR / "market_residual_symbol_impact.csv", index=False, encoding="utf-8-sig")
    year_impact.to_csv(AUDIT_DIR / "market_residual_year_impact.csv", index=False, encoding="utf-8-sig")
    (RESULTS / "market_residual_coverage_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(audit, RESULTS / "market_residual_coverage_audit.md")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "fdr_residual_symbols": audit["fdr_residual_symbols"],
                "main_residual_rows": audit["main"]["residual_rows"],
                "robust_residual_rows": audit["robust"]["residual_rows"],
            },
            indent=2,
        )
    )
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
