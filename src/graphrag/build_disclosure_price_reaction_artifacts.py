from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "graphrag"
FIGURE_DIR = ROOT / "figures" / "en"

SEQUENCE_REACTIONS = RESULT_DIR / "sequence_market_reactions.csv"
CASE_REACTIONS = RESULT_DIR / "case_market_reactions.csv"
LLM_PRIMARY = RESULT_DIR / "llm_market_context_primary_predictions.csv"
LLM_METRICS = RESULT_DIR / "llm_market_context_metrics.csv"

OUT_SUMMARY = RESULT_DIR / "disclosure_price_reaction_summary.csv"
OUT_CASE_TABLE = RESULT_DIR / "disclosure_price_reaction_case_table.csv"
OUT_LLM_GROUPS = RESULT_DIR / "disclosure_price_reaction_by_llm_prediction.csv"
OUT_MANIFEST = RESULT_DIR / "disclosure_price_reaction_artifacts_manifest.json"

FIG_PRICE = FIGURE_DIR / "paper4_disclosure_price_reaction_en.png"
FIG_LLM = FIGURE_DIR / "paper4_llm_context_ablation_en.png"

GRAPH_RAG_CONDITION = "disclosure_premarket_graph_rag_llm"

PALETTE = {
    "ink": "#1F2933",
    "muted": "#5E6B73",
    "blue": "#2F6F9F",
    "blue_light": "#8BB7D7",
    "green": "#5F8E4E",
    "orange": "#C77D3A",
    "red": "#B65C4A",
    "gray": "#DDE4EA",
}


WINDOWS = [
    ("-20d pre", "pre_excess_return_20d", "pre_return_20d", "pre_complete_20d"),
    ("+1d post", "post_excess_return_1d", "post_return_1d", "post_complete_1d"),
    ("+5d post", "post_excess_return_5d", "post_return_5d", "post_complete_5d"),
    ("+20d post", "post_excess_return_20d", "post_return_20d", "post_complete_20d"),
    ("+60d post", "post_excess_return_60d", "post_return_60d", "post_complete_60d"),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.1f}%"


def pct_label(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.1f}%"


def summarize_sequence_reactions(reactions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, excess_col, raw_col, complete_col in WINDOWS:
        available = reactions[excess_col].notna()
        complete = to_bool(reactions[complete_col]) if complete_col in reactions else available
        rows.append(
            {
                "window": label,
                "rows": int(len(reactions)),
                "available_rows": int(available.sum()),
                "complete_rows": int(complete.sum()),
                "mean_raw_return": float(pd.to_numeric(reactions[raw_col], errors="coerce").mean()),
                "median_raw_return": float(pd.to_numeric(reactions[raw_col], errors="coerce").median()),
                "mean_excess_return": float(pd.to_numeric(reactions[excess_col], errors="coerce").mean()),
                "median_excess_return": float(pd.to_numeric(reactions[excess_col], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def build_case_table(case_reactions: pd.DataFrame) -> pd.DataFrame:
    df = case_reactions.copy()
    for col in [
        "pre_close",
        "post_return_1d",
        "post_return_5d",
        "post_return_20d",
        "post_return_60d",
        "pre_excess_return_20d",
        "post_excess_return_1d",
        "post_excess_return_5d",
        "post_excess_return_20d",
        "post_excess_return_60d",
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for window in (1, 5, 20, 60):
        df[f"implied_post_close_{window}d"] = df["pre_close"] * (1.0 + df[f"post_return_{window}d"])
    keep = [
        "case_id",
        "firm_name",
        "stock_code",
        "pattern",
        "first_pattern_date",
        "event_trade_date",
        "pre_close_date",
        "pre_close",
        "implied_post_close_1d",
        "implied_post_close_5d",
        "implied_post_close_20d",
        "implied_post_close_60d",
        "pre_excess_return_20d",
        "post_excess_return_1d",
        "post_excess_return_5d",
        "post_excess_return_20d",
        "post_excess_return_60d",
        "has_outcome",
        "outcome_label",
        "lead_days",
        "reaction_available",
        "missing_reason",
    ]
    keep = [col for col in keep if col in df.columns]
    return df[keep].sort_values(["first_pattern_date", "firm_name"], na_position="last")


def summarize_llm_groups(primary: pd.DataFrame, case_reactions: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "case_id",
        "reaction_available",
        "pre_excess_return_20d",
        "post_excess_return_1d",
        "post_excess_return_5d",
        "post_excess_return_20d",
        "post_excess_return_60d",
    ]
    market = case_reactions[keep_cols].drop_duplicates("case_id")
    merged = primary.merge(market, on="case_id", how="left")
    rows: list[dict[str, object]] = []
    for (condition, prediction), group in merged.groupby(["condition", "prediction"], dropna=False):
        rows.append(
            {
                "condition": condition,
                "prediction": prediction,
                "cases": int(len(group)),
                "reaction_available_cases": int(to_bool(group["reaction_available"]).sum()),
                "mean_pre_20d_excess": float(pd.to_numeric(group["pre_excess_return_20d"], errors="coerce").mean()),
                "mean_post_1d_excess": float(pd.to_numeric(group["post_excess_return_1d"], errors="coerce").mean()),
                "mean_post_5d_excess": float(pd.to_numeric(group["post_excess_return_5d"], errors="coerce").mean()),
                "mean_post_20d_excess": float(pd.to_numeric(group["post_excess_return_20d"], errors="coerce").mean()),
                "mean_post_60d_excess": float(pd.to_numeric(group["post_excess_return_60d"], errors="coerce").mean()),
                "median_post_20d_excess": float(pd.to_numeric(group["post_excess_return_20d"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["condition", "prediction"])


def write_price_figure(summary: pd.DataFrame, llm_groups: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), gridspec_kw={"width_ratios": [1.05, 0.95]})

    x = list(range(len(summary)))
    mean_vals = summary["mean_excess_return"].mul(100).tolist()
    median_vals = summary["median_excess_return"].mul(100).tolist()
    axes[0].bar(x, mean_vals, color=PALETTE["blue_light"], label="Mean excess return")
    axes[0].plot(x, median_vals, marker="o", color=PALETTE["red"], linewidth=2.0, label="Median excess return")
    axes[0].axhline(0, color=PALETTE["ink"], linewidth=1.0)
    axes[0].set_xticks(x, summary["window"].tolist())
    axes[0].set_ylabel("Market-adjusted return (%)")
    axes[0].set_title("All DART Sequence Matches")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    for xpos, value in zip(x, mean_vals):
        axes[0].text(xpos, value + (0.7 if value >= 0 else -1.2), pct_label(value), ha="center", va="bottom" if value >= 0 else "top", fontsize=8.5)

    graph = llm_groups[llm_groups["condition"].eq(GRAPH_RAG_CONDITION)].copy()
    labels = ["-20d pre", "+1d", "+5d", "+20d", "+60d"]
    cols = [
        "mean_pre_20d_excess",
        "mean_post_1d_excess",
        "mean_post_5d_excess",
        "mean_post_20d_excess",
        "mean_post_60d_excess",
    ]
    colors = {"distress": PALETTE["red"], "normal_restructuring": PALETTE["green"]}
    names = {"distress": "LLM distress", "normal_restructuring": "LLM normal"}
    for prediction in ["distress", "normal_restructuring"]:
        row = graph[graph["prediction"].eq(prediction)]
        if row.empty:
            continue
        vals = [float(row.iloc[0][col]) * 100 for col in cols]
        axes[1].plot(labels, vals, marker="o", linewidth=2.2, color=colors[prediction], label=names[prediction])
        for xpos, value in enumerate(vals):
            offset = -1.0 if prediction == "distress" else 0.8
            axes[1].text(
                xpos,
                value + offset,
                pct_label(value),
                ha="center",
                va="top" if prediction == "distress" else "bottom",
                fontsize=8,
            )
    axes[1].axhline(0, color=PALETTE["ink"], linewidth=1.0)
    axes[1].set_title("Benchmark Cases by LLM Prediction")
    axes[1].set_ylabel("Mean market-adjusted return (%)")
    all_vals = [
        float(row.iloc[0][col]) * 100
        for _, row in graph.groupby("prediction")
        for col in cols
    ]
    if all_vals:
        axes[1].set_ylim(min(all_vals) - 3.0, max(all_vals) + 2.5)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, loc="lower left")

    fig.suptitle("Disclosure-Window Stock-Return Reaction", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def condition_short_name(condition: str) -> str:
    return {
        "disclosure_only_llm": "Disclosure\nonly",
        "disclosure_premarket_llm": "Disclosure +\npre-market",
        "disclosure_premarket_graph_rag_llm": "Disclosure +\nmarket + graph/RAG",
    }.get(condition, condition)


def write_llm_figure(metrics: pd.DataFrame, llm_groups: pd.DataFrame, path: Path) -> None:
    order = [
        "disclosure_only_llm",
        "disclosure_premarket_llm",
        "disclosure_premarket_graph_rag_llm",
    ]
    metrics = metrics.set_index("condition").loc[order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), gridspec_kw={"width_ratios": [1.1, 0.9]})
    x = list(range(len(metrics)))
    width = 0.26
    series = [
        ("accuracy", "Accuracy", PALETTE["blue"]),
        ("precision", "Precision", PALETTE["green"]),
        ("false_positive_rate", "False-positive rate", PALETTE["orange"]),
    ]
    for offset, (col, label, color) in zip([-width, 0, width], series):
        values = metrics[col].astype(float).mul(100).tolist()
        axes[0].bar([i + offset for i in x], values, width=width, color=color, label=label)
        for xpos, value in zip([i + offset for i in x], values):
            axes[0].text(xpos, value + 1.0, pct_label(value), ha="center", fontsize=8)
    axes[0].set_xticks(x, [condition_short_name(c) for c in metrics["condition"]])
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("Percent")
    axes[0].set_title("Predictive Classification")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    graph = llm_groups[llm_groups["condition"].eq(GRAPH_RAG_CONDITION)].copy()
    graph["label"] = graph["prediction"].replace({"distress": "LLM distress", "normal_restructuring": "LLM normal"})
    graph = graph.set_index("prediction").loc[["distress", "normal_restructuring"]].reset_index()
    vals = graph["mean_post_20d_excess"].astype(float).mul(100).tolist()
    bars = axes[1].bar(graph["label"], vals, color=[PALETTE["red"], PALETTE["green"]])
    axes[1].axhline(0, color=PALETTE["ink"], linewidth=1.0)
    axes[1].set_title("Post-Hoc +20d Market Separation")
    axes[1].set_ylabel("Mean excess return (%)")
    axes[1].set_ylim(min(vals) - 3.0, max(vals) + 2.5)
    axes[1].spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, vals):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.7 if value >= 0 else -0.8),
            pct_label(value),
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    fig.suptitle("LLM Context Ablation and Market-Reaction Check", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    configure_style()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    reactions = pd.read_csv(SEQUENCE_REACTIONS, encoding="utf-8-sig")
    case_reactions = pd.read_csv(CASE_REACTIONS, encoding="utf-8-sig")
    primary = pd.read_csv(LLM_PRIMARY, encoding="utf-8-sig")
    metrics = pd.read_csv(LLM_METRICS, encoding="utf-8-sig")

    summary = summarize_sequence_reactions(reactions)
    case_table = build_case_table(case_reactions)
    llm_groups = summarize_llm_groups(primary, case_reactions)

    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    case_table.to_csv(OUT_CASE_TABLE, index=False, encoding="utf-8-sig")
    llm_groups.to_csv(OUT_LLM_GROUPS, index=False, encoding="utf-8-sig")

    write_price_figure(summary, llm_groups, FIG_PRICE)
    write_llm_figure(metrics, llm_groups, FIG_LLM)

    manifest = {
        "status": "completed",
        "inputs": [
            str(SEQUENCE_REACTIONS.relative_to(ROOT)),
            str(CASE_REACTIONS.relative_to(ROOT)),
            str(LLM_PRIMARY.relative_to(ROOT)),
            str(LLM_METRICS.relative_to(ROOT)),
        ],
        "outputs": [
            str(OUT_SUMMARY.relative_to(ROOT)),
            str(OUT_CASE_TABLE.relative_to(ROOT)),
            str(OUT_LLM_GROUPS.relative_to(ROOT)),
            str(FIG_PRICE.relative_to(ROOT)),
            str(FIG_LLM.relative_to(ROOT)),
        ],
        "claim_boundary": (
            "Close-to-close returns are descriptive sequence-window reactions anchored on first_pattern_date; "
            "they are not a causal intraday disclosure event study."
        ),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
