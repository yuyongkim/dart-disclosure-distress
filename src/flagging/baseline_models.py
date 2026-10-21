from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


EXACT_COMPONENT_AUDIT = {
    "altman_z_score_original": {
        "status": "not_computable_exactly",
        "available_components": [
            "EBIT-like operating income / total assets",
            "book equity / total liabilities",
            "sales-like revenue / total assets",
        ],
        "missing_components": [
            "working capital / total assets",
            "retained earnings / total assets",
            "market value of equity / total liabilities",
        ],
        "proxy_used": "altman_partial_available_components",
    },
    "ohlson_o_score_original": {
        "status": "not_computable_exactly",
        "available_components": [
            "log total assets",
            "total liabilities / total assets",
            "net income / total assets",
            "negative equity indicator",
            "two-year loss indicator",
            "net-income change ratio",
        ],
        "missing_components": [
            "working capital / total assets",
            "current liabilities / current assets",
            "funds from operations / total liabilities",
            "GNP price-level adjustment",
        ],
        "proxy_used": "ohlson_partial_available_components",
    },
}


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def add_proxy_scores(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["book_equity_to_liabilities"] = safe_divide(df["fin_total_equity"], df["fin_total_liabilities"])
    df["sales_to_assets"] = safe_divide(df["fin_revenue"], df["fin_total_assets"])
    df["operating_income_to_assets"] = safe_divide(df["fin_operating_income"], df["fin_total_assets"])
    df["net_income_to_assets"] = safe_divide(df["fin_net_income"], df["fin_total_assets"])
    df["liabilities_to_assets"] = safe_divide(df["fin_total_liabilities"], df["fin_total_assets"])
    df["log_assets"] = np.log(df["fin_total_assets"].where(df["fin_total_assets"] > 0))

    df = df.sort_values(["firm_id", "anchor_year"]).copy()
    prev_ni = df.groupby("firm_id")["fin_net_income"].shift(1)
    prev_loss = df.groupby("firm_id")["fin_loss_flag"].shift(1)
    df["two_year_loss_flag"] = ((df["fin_loss_flag"].eq(1.0)) & (prev_loss.eq(1.0))).astype(float)
    denom = df["fin_net_income"].abs() + prev_ni.abs()
    df["net_income_change_ratio"] = (df["fin_net_income"] - prev_ni) / denom.replace(0, np.nan)

    # This is an available-component proxy, not the original Altman Z-score.
    # Higher values should indicate healthier firms, so the risk score is negated.
    df["altman_partial_health_score"] = (
        6.72 * df["operating_income_to_assets"]
        + 1.05 * df["book_equity_to_liabilities"]
        + 1.00 * df["sales_to_assets"]
    )
    df["altman_partial_risk_score"] = -df["altman_partial_health_score"]

    # Available-component proxy based on the signs and coefficients of the Ohlson model.
    # Missing original terms are intentionally not imputed.
    df["ohlson_partial_risk_score"] = (
        -0.407 * df["log_assets"]
        + 6.03 * df["liabilities_to_assets"]
        - 2.37 * df["net_income_to_assets"]
        + 0.285 * df["two_year_loss_flag"]
        - 1.72 * df["fin_negative_equity_flag"]
        - 0.521 * df["net_income_change_ratio"]
    )

    df["leverage_risk_score"] = df["liabilities_to_assets"]
    df["loss_leverage_risk_score"] = (
        df["liabilities_to_assets"]
        + 0.5 * df["fin_loss_flag"]
        + 1.0 * df["fin_negative_equity_flag"]
    )
    return df


def top_fraction_metrics(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.10) -> dict:
    n = len(y_true)
    if n == 0:
        return {"top_n": 0, "precision": None, "recall": None, "lift": None}
    top_n = max(1, int(np.ceil(n * fraction)))
    order = np.argsort(-y_score)
    top = order[:top_n]
    base_rate = float(np.mean(y_true)) if n else 0.0
    precision = float(np.mean(y_true[top])) if top_n else None
    recall = float(np.sum(y_true[top]) / np.sum(y_true)) if np.sum(y_true) else None
    lift = float(precision / base_rate) if base_rate and precision is not None else None
    return {
        "top_n": int(top_n),
        "precision": precision,
        "recall": recall,
        "lift": lift,
    }


def evaluate_score(data: pd.DataFrame, target: str, score_col: str, horizon: int) -> dict:
    usable = data[[target, score_col, "days_to_outcome"]].dropna(subset=[score_col]).copy()
    y_true = usable[target].astype(int).to_numpy()
    y_score = usable[score_col].astype(float).to_numpy()
    top = top_fraction_metrics(y_true, y_score, 0.10)
    top_idx = np.argsort(-y_score)[: top["top_n"]]
    top_rows = usable.iloc[top_idx]
    leads = pd.to_numeric(top_rows.loc[top_rows[target].astype(bool), "days_to_outcome"], errors="coerce").dropna()
    result = {
        "horizon_days": horizon,
        "method": score_col,
        "n_test": int(len(usable)),
        "test_positives": int(y_true.sum()),
        "coverage_rate": float(len(usable) / len(data)) if len(data) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "average_precision": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "top_10pct_precision": top["precision"],
        "top_10pct_recall": top["recall"],
        "top_10pct_lift": top["lift"],
        "top_10pct_median_lead_days": float(leads.median()) if not leads.empty else None,
        "top_10pct_mean_lead_days": float(leads.mean()) if not leads.empty else None,
    }
    return result


def evaluate(panel: pd.DataFrame, observation_end: pd.Timestamp) -> pd.DataFrame:
    scored = add_proxy_scores(panel)
    methods = [
        "altman_partial_risk_score",
        "ohlson_partial_risk_score",
        "leverage_risk_score",
        "loss_leverage_risk_score",
    ]
    rows: list[dict] = []
    for horizon in (365, 730):
        target = f"target_{horizon}d"
        horizon_data = scored[
            (scored["anchor_date"] + pd.to_timedelta(horizon, unit="D") <= observation_end)
            & scored["anchor_year"].ge(2022)
        ].copy()
        for method in methods:
            rows.append(evaluate_score(horizon_data, target, method, horizon))
    return pd.DataFrame(rows)


def write_markdown(results: pd.DataFrame, audit: dict, path: Path) -> None:
    lines = ["# Classic Financial Baseline Audit and Proxy Results", ""]
    lines.append("Exact Altman Z-score and Ohlson O-score cannot be computed from the currently available point-in-time financial panel because several required components are missing. This file reports that component audit and evaluates transparent available-component proxy scores as ranking baselines.")
    lines.append("")
    lines.append("## Component Audit")
    lines.append("")
    for name, item in audit.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Status: `{item['status']}`")
        lines.append(f"- Proxy used: `{item['proxy_used']}`")
        lines.append(f"- Available: {', '.join(item['available_components'])}.")
        lines.append(f"- Missing: {', '.join(item['missing_components'])}.")
        lines.append("")

    lines.append("## Out-of-Time Proxy Baseline Results")
    lines.append("")
    lines.append("Scores are evaluated on anchor years >= 2022. Higher score means higher predicted distress risk.")
    lines.append("")
    lines.append("| Horizon | Method | Test rows | Positives | Coverage | ROC AUC | Avg precision | Top 10% precision | Top 10% recall | Top 10% lift | Median lead days |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in results.sort_values(["horizon_days", "average_precision"], ascending=[True, False]).itertuples(index=False):
        median = "" if pd.isna(row.top_10pct_median_lead_days) else f"{row.top_10pct_median_lead_days:.1f}"
        lines.append(
            "| {h} | `{m}` | {n} | {pos} | {cov:.3f} | {auc:.4f} | {ap:.4f} | {p:.4f} | {r:.4f} | {lift:.2f} | {median} |".format(
                h=row.horizon_days,
                m=row.method,
                n=row.n_test,
                pos=row.test_positives,
                cov=row.coverage_rate,
                auc=row.roc_auc if row.roc_auc is not None else float("nan"),
                ap=row.average_precision if row.average_precision is not None else float("nan"),
                p=row.top_10pct_precision if row.top_10pct_precision is not None else float("nan"),
                r=row.top_10pct_recall if row.top_10pct_recall is not None else float("nan"),
                lift=row.top_10pct_lift if row.top_10pct_lift is not None else float("nan"),
                median=median,
            )
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("These proxy scores are useful as transparent financial-only comparators, but they should not be described as exact Altman or exact Ohlson scores. The main manuscript should continue to compare against the financial-only logistic model as the stronger financial baseline, while citing these proxy results as an additional robustness check.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-path", default=str(Path("Paper 4") / "data" / "model" / "dart_model_panel.parquet"))
    parser.add_argument("--summary-path", default=str(Path("Paper 4") / "data" / "model" / "dart_model_panel_summary.json"))
    parser.add_argument("--results-dir", default=str(Path("Paper 4") / "results"))
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel_path)
    panel["anchor_date"] = pd.to_datetime(panel["anchor_date"], errors="coerce")
    summary = json.loads(Path(args.summary_path).read_text(encoding="utf-8"))
    observation_end = pd.to_datetime(summary["outcome_observation_end"])

    results = evaluate(panel, observation_end)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "classic_financial_baselines.csv", index=False, encoding="utf-8-sig")
    (results_dir / "classic_financial_baseline_audit.json").write_text(
        json.dumps(EXACT_COMPONENT_AUDIT, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(results, EXACT_COMPONENT_AUDIT, results_dir / "classic_financial_baselines.md")
    print(json.dumps(results.to_dict(orient="records"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
