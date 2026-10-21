from __future__ import annotations

import csv
import json
import os
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
FIGURES = ROOT / "figures" / "en"
RESULTS = ROOT / "results"
OUT_JSON = RESULTS / "submission_figure_generation_audit.json"
OUT_MD = RESULTS / "submission_figure_generation_audit.md"

SOURCE_FILES = [
    "data/processed/phase1_event_summary.json",
    "data/model/dart_model_panel_summary.json",
    "data/model_market/dart_model_panel_market_summary.json",
    "data/raw/market_coverage/naver_ohlcv_supplement_summary.json",
    "data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json",
    "results/market_residual_coverage_audit.json",
    "results/dart_model_metrics.json",
    "results/market/dart_model_metrics.json",
    "results/dart_model_permutation_importance.csv",
    "results/graphrag/sequence_market_reaction_summary.json",
    "results/graphrag/ablation_metrics.csv",
]

PALETTE = {
    "ink": "#1F2933",
    "muted": "#5E6B73",
    "grid": "#D6DEE5",
    "blue": "#2F6F9F",
    "blue_light": "#8BB7D7",
    "green": "#5F8E4E",
    "green_light": "#A4BE70",
    "orange": "#C77D3A",
    "red": "#B65C4A",
    "yellow": "#D7AD47",
    "gray": "#E8EEF3",
}


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


def read_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def metric_row(path: str, horizon: int, model: str) -> dict[str, Any]:
    rows = read_json(path)
    for row in rows:
        if int(row.get("horizon_days", -1)) == horizon and row.get("model") == model:
            return row
    raise KeyError(f"Missing metric row: {path} horizon={horizon} model={model}")


def wrap_label(label: str, width: int = 20) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def fmt_int(value: int | float) -> str:
    return f"{int(round(float(value))):,}"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def pct_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0, 105)
    ax.set_ylabel("Percent")
    ax.grid(axis="y", color=PALETTE["grid"], alpha=0.7, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_record(path: Path, kind: str, description: str, values: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "kind": kind,
        "description": description,
        "size_bytes": path.stat().st_size,
        "values": values,
    }


def draw_box(ax: plt.Axes, x: float, y: float, w: float, h: float, label: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=PALETTE["ink"],
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10.5)


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.3,
            color=PALETTE["ink"],
        )
    )


def write_framework(path: Path) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(12, 5.1))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)

    boxes = [
        (0.35, 3.25, 2.15, 0.95, "DART\nmajor-event disclosures", PALETTE["gray"]),
        (0.35, 1.85, 2.15, 0.95, "Financial\nfundamentals", "#EDF5EA"),
        (0.35, 0.45, 2.15, 0.95, "OHLCV and\nKOSPI/KOSDAQ indices", "#F6EFE6"),
        (3.15, 3.05, 2.0, 1.15, "T-code sequence\nfeatures", "#DBEAF4"),
        (3.15, 1.45, 2.0, 1.15, "Point-in-time\ncontrols", "#E8F1DD"),
        (6.0, 2.2, 2.05, 1.15, "Calibrated\nranking model", "#F7E6D2"),
        (8.95, 2.2, 2.1, 1.15, "Top-decile\nwatchlist", "#DDECF2"),
        (8.95, 0.55, 2.1, 1.05, "Optional\ncase review", "#F4E7EB"),
    ]
    for box in boxes:
        draw_box(ax, *box)
    arrows = [
        ((2.5, 3.72), (3.15, 3.62)),
        ((2.5, 2.32), (3.15, 2.03)),
        ((2.5, 0.92), (3.15, 1.88)),
        ((5.15, 3.58), (6.0, 2.92)),
        ((5.15, 2.03), (6.0, 2.62)),
        ((8.05, 2.77), (8.95, 2.77)),
        ((10.0, 2.2), (10.0, 1.6)),
    ]
    for start, end in arrows:
        draw_arrow(ax, start, end)
    ax.text(6.0, 4.85, "Finance-First Disclosure-Sequence Early-Warning Design", ha="center", fontsize=15, weight="bold")
    ax.text(
        6.0,
        0.12,
        "The prediction model ranks firms; the case-review layer attaches sequence, market-reaction, and rule evidence.",
        ha="center",
        fontsize=10,
        color=PALETTE["muted"],
    )
    save(fig, path)
    return figure_record(
        path,
        "framework",
        "Workflow diagram separating source data, feature layers, ranking model, watchlist output, and case review.",
        {"nodes": len(boxes)},
    )


def write_data_scope(path: Path) -> dict[str, Any]:
    event_summary = read_json("data/processed/phase1_event_summary.json")
    panel = read_json("data/model/dart_model_panel_summary.json")
    dart_counts = event_summary["dart_events"]["taxonomy_counts"]
    scope_labels = [
        "DART events",
        "SEC 8-K items",
        "Firm-year panel",
        "DART outcome rows",
    ]
    scope_values = [
        event_summary["dart_events"]["rows"],
        event_summary["sec_events"]["rows"],
        panel["rows"],
        event_summary["dart_outcomes"]["rows"],
    ]
    taxonomy_items = sorted(dart_counts.items(), key=lambda item: item[1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [0.9, 1.1]})
    axes[0].bar(scope_labels, scope_values, color=[PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["red"]])
    axes[0].set_title("Executable Data Scope")
    axes[0].set_ylabel("Rows")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", color=PALETTE["grid"], alpha=0.7)
    axes[0].spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(scope_values):
        axes[0].text(index, value * 1.02, fmt_int(value), ha="center", fontsize=9)

    labels = [item[0] for item in taxonomy_items]
    values = [item[1] for item in taxonomy_items]
    axes[1].barh(labels, values, color=PALETTE["blue_light"])
    axes[1].set_title("DART Taxonomy Distribution")
    axes[1].set_xlabel("Event rows")
    axes[1].spines[["top", "right"]].set_visible(False)
    for y_pos, value in enumerate(values):
        axes[1].text(value * 1.01, y_pos, fmt_int(value), va="center", fontsize=9)

    fig.suptitle("Data Scope and Disclosure Taxonomy Coverage", fontsize=15, weight="bold")
    save(fig, path)
    return figure_record(
        path,
        "data_scope",
        "Counts the executable DART, SEC, firm-year panel, outcome, and DART taxonomy coverage.",
        {
            "dart_event_rows": event_summary["dart_events"]["rows"],
            "sec_event_rows": event_summary["sec_events"]["rows"],
            "panel_rows": panel["rows"],
            "panel_firms": panel["firms"],
            "dart_outcome_rows": event_summary["dart_outcomes"]["rows"],
        },
    )


def write_model_performance(path: Path) -> dict[str, Any]:
    rows = {
        "365 non-market RF": metric_row("results/dart_model_metrics.json", 365, "combined_random_forest"),
        "365 market calibrated RF": metric_row(
            "results/market/dart_model_metrics.json",
            365,
            "combined_market_calibrated_random_forest",
        ),
        "730 non-market calibrated RF": metric_row(
            "results/dart_model_metrics.json",
            730,
            "combined_calibrated_random_forest",
        ),
        "730 market calibrated RF": metric_row(
            "results/market/dart_model_metrics.json",
            730,
            "combined_market_calibrated_random_forest",
        ),
    }
    labels = list(rows)
    x = list(range(len(labels)))
    auc = [rows[label]["roc_auc"] for label in labels]
    ap = [rows[label]["average_precision"] for label in labels]
    lift = [rows[label]["top_10pct"]["lift"] for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), gridspec_kw={"width_ratios": [1.12, 0.88]})
    width = 0.34
    axes[0].bar([i - width / 2 for i in x], auc, width, label="ROC AUC", color=PALETTE["blue"])
    axes[0].bar([i + width / 2 for i in x], ap, width, label="Average precision", color=PALETTE["green"])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Discrimination and Precision")
    axes[0].set_xticks(x, [wrap_label(label, 16) for label in labels])
    axes[0].grid(axis="y", color=PALETTE["grid"], alpha=0.7)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].bar(x, lift, color=[PALETTE["blue_light"], PALETTE["blue"], PALETTE["green_light"], PALETTE["green"]])
    axes[1].set_title("Top-Decile Watchlist Lift")
    axes[1].set_ylabel("Lift")
    axes[1].set_xticks(x, [wrap_label(label, 16) for label in labels])
    axes[1].grid(axis="y", color=PALETTE["grid"], alpha=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(lift):
        axes[1].text(index, value + 0.12, f"{value:.2f}", ha="center", fontsize=9)

    fig.suptitle("Out-of-Time Model Performance", fontsize=15, weight="bold")
    save(fig, path)
    return figure_record(
        path,
        "model_performance",
        "Compares non-market and market-extended combined random-forest specifications.",
        {
            label: {
                "roc_auc": rows[label]["roc_auc"],
                "average_precision": rows[label]["average_precision"],
                "top_10pct_lift": rows[label]["top_10pct"]["lift"],
            }
            for label in labels
        },
    )


def feature_label(feature: str) -> str:
    replacements = {
        "seq_days_since_last_T06": "Days since last T06",
        "fin_log_total_assets": "Log total assets",
        "fin_log_revenue": "Log revenue",
        "seq_events_to_date": "Cumulative events",
        "seq_days_since_last_event": "Days since last event",
        "seq_days_since_last_T99": "Days since last T99",
        "fin_operating_margin": "Operating margin",
        "seq_T06_365d": "T06 count, 365 days",
        "seq_T99_to_date": "Cumulative T99",
        "fin_net_margin": "Net margin",
    }
    if feature in replacements:
        return replacements[feature]
    cleaned = feature
    for prefix in ("seq_", "fin_", "mkt_"):
        cleaned = cleaned.removeprefix(prefix)
    return cleaned.replace("_", " ").title()


def write_permutation_importance(path: Path) -> dict[str, Any]:
    rows = [
        row
        for row in read_csv("results/dart_model_permutation_importance.csv")
        if row["horizon_days"] == "365"
        and row["model"] == "combined_random_forest"
        and row["scoring"] == "average_precision"
        and int(row["rank"]) <= 10
    ]
    rows.sort(key=lambda row: int(row["rank"]), reverse=True)
    labels = [feature_label(row["feature"]) for row in rows]
    values = [float(row["importance_mean"]) for row in rows]
    colors = [PALETTE["blue"] if row["feature_group"] == "sequence" else PALETTE["green"] for row in rows]

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    ax.barh(labels, values, color=colors)
    ax.set_title("Top Permutation Importance for 365-Day Ranking")
    ax.set_xlabel("Average-precision loss when permuted")
    ax.spines[["top", "right"]].set_visible(False)
    for y_pos, value in enumerate(values):
        ax.text(value + max(values) * 0.015, y_pos, f"{value:.4f}", va="center", fontsize=9)
    seq_proxy = plt.Rectangle((0, 0), 1, 1, color=PALETTE["blue"])
    fin_proxy = plt.Rectangle((0, 0), 1, 1, color=PALETTE["green"])
    ax.legend([seq_proxy, fin_proxy], ["Sequence", "Financial"], frameon=False, loc="lower right")
    save(fig, path)
    return figure_record(
        path,
        "permutation_importance",
        "Shows the ten strongest 365-day combined-model features under average-precision permutation importance.",
        {
            row["feature"]: {
                "rank": int(row["rank"]),
                "feature_group": row["feature_group"],
                "importance_mean": float(row["importance_mean"]),
            }
            for row in sorted(rows, key=lambda row: int(row["rank"]))
        },
    )


def write_market_coverage(path: Path) -> dict[str, Any]:
    naver = read_json("data/raw/market_coverage/naver_ohlcv_supplement_summary.json")
    fdr = read_json("data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json")
    panel = read_json("data/model_market/dart_model_panel_market_summary.json")
    residual = read_json("results/market_residual_coverage_audit.json")

    base_rows = panel["base_ohlcv_rows_loaded_before_supplement"]
    naver_rows = naver["output_rows_total"]
    fdr_rows = fdr["output_rows_total"]
    coverage_rates = [
        panel["price_record_rate"] * 100,
        panel["recent_price_record_rate"] * 100,
        residual["main"]["residual_row_share"] * 100,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), gridspec_kw={"width_ratios": [1.1, 0.9]})
    left = 0
    segments = [
        ("Base SQLite rows", base_rows, PALETTE["blue_light"]),
        ("Naver supplement", naver_rows, PALETTE["green_light"]),
        ("FDR gap supplement", fdr_rows, PALETTE["orange"]),
    ]
    for label, value, color in segments:
        axes[0].barh(["Source rows"], [value], left=[left], color=color, label=label)
        axes[0].text(left + value / 2, 0, fmt_int(value), ha="center", va="center", fontsize=9)
        left += value
    axes[0].set_title("OHLCV Source Composition")
    axes[0].set_xlabel("Rows before final de-duplication", labelpad=10)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    axes[0].spines[["top", "right"]].set_visible(False)

    bars = axes[1].bar(
        [0, 1, 2],
        coverage_rates,
        color=[PALETTE["blue"], PALETTE["green"], PALETTE["red"]],
    )
    axes[1].set_xticks(
        [0, 1, 2],
        ["Any price\nrecord", "Recent price\nrecord", "Residual\nno-row anchors"],
    )
    axes[1].set_title("Firm-Year Market Coverage")
    pct_axis(axes[1])
    for bar, value in zip(bars, coverage_rates):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.4, f"{value:.1f}%", ha="center", fontsize=9)

    fig.suptitle("Market-Price Extension Coverage", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_record(
        path,
        "market_coverage",
        "Documents supplemental OHLCV source rows and remaining market-feature coverage limits.",
        {
            "base_rows_loaded_before_supplement": base_rows,
            "naver_rows": naver_rows,
            "fdr_rows": fdr_rows,
            "price_record_rate": panel["price_record_rate"],
            "recent_price_record_rate": panel["recent_price_record_rate"],
            "residual_row_share": residual["main"]["residual_row_share"],
        },
    )


def write_sequence_market_reaction(path: Path) -> dict[str, Any]:
    summary = read_json("results/graphrag/sequence_market_reaction_summary.json")
    top_patterns = summary["by_pattern_top20"][:10]
    top_patterns = list(reversed(top_patterns))
    pattern_labels = [item["pattern"] for item in top_patterns]
    mean_20 = [item["mean_post_excess_return_20d"] * 100 for item in top_patterns]
    median_20 = [item["median_post_excess_return_20d"] * 100 for item in top_patterns]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [0.82, 1.18]})
    rates = [
        summary["reaction_available_rate"] * 100,
        summary["post_20d_complete_rate"] * 100,
        summary["post_60d_complete_rate"] * 100,
    ]
    rate_labels = ["Reaction\navailable", "+20d window\ncomplete", "+60d window\ncomplete"]
    bars = axes[0].bar(range(3), rates, color=[PALETTE["blue"], PALETTE["green"], PALETTE["orange"]])
    axes[0].set_title("Reaction-Evidence Coverage")
    axes[0].set_xticks(range(3), rate_labels)
    pct_axis(axes[0])
    for bar, value in zip(bars, rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}%", ha="center", fontsize=9)
    axes[0].text(
        1.0,
        8,
        f"{fmt_int(summary['dart_pattern_occurrences'])} DART matches\n{fmt_int(summary['dart_unique_firms'])} firms",
        ha="center",
        fontsize=10,
        color=PALETTE["muted"],
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )

    axes[1].barh(pattern_labels, mean_20, color=PALETTE["blue_light"], label="Mean +20d excess return")
    axes[1].scatter(median_20, pattern_labels, color=PALETTE["red"], zorder=3, label="Median +20d excess return")
    axes[1].axvline(0, color=PALETTE["ink"], linewidth=1)
    axes[1].set_title("Common Sequence Patterns")
    axes[1].set_xlabel("Post-event excess return (%)")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, loc="lower right")

    fig.suptitle("Sequence-Market Reaction Screen", fontsize=15, weight="bold")
    save(fig, path)
    return figure_record(
        path,
        "sequence_market_reaction",
        "Summarizes reaction-evidence coverage and +20 trading-session excess returns for common DART sequence patterns.",
        {
            "dart_pattern_occurrences": summary["dart_pattern_occurrences"],
            "dart_unique_firms": summary["dart_unique_firms"],
            "reaction_available_rate": summary["reaction_available_rate"],
            "post_20d_complete_rate": summary["post_20d_complete_rate"],
            "mean_post_excess_return_20d": summary["mean_post_excess_return_20d"],
            "median_post_excess_return_20d": summary["median_post_excess_return_20d"],
        },
    )


def condition_label(row: dict[str, str]) -> str:
    condition = row["condition"]
    prompt = row["prompt_strategy"]
    labels = {
        ("graph_rag_llm", "cot"): "Graph RAG\nCoT",
        ("graph_rag_llm", "few_shot"): "Graph RAG\nfew-shot",
        ("graph_rag_llm", "zero_shot"): "Graph RAG\nzero-shot",
        ("vector_rag_llm", "none"): "Vector RAG\nLLM",
        ("rule_llm", "none"): "Rule + LLM",
        ("rule_only", "none"): "Rule only",
    }
    return labels.get((condition, prompt), f"{condition}\n{prompt}")


def ordered_ablation_rows() -> list[dict[str, str]]:
    order = [
        ("graph_rag_llm", "cot"),
        ("graph_rag_llm", "few_shot"),
        ("graph_rag_llm", "zero_shot"),
        ("vector_rag_llm", "none"),
        ("rule_llm", "none"),
        ("rule_only", "none"),
    ]
    rows = read_csv("results/graphrag/ablation_metrics.csv")
    lookup = {(row["condition"], row["prompt_strategy"]): row for row in rows}
    return [lookup[key] for key in order if key in lookup]


def write_graphrag_ablation(path: Path) -> dict[str, Any]:
    rows = ordered_ablation_rows()
    labels = [condition_label(row) for row in rows]
    x = list(range(len(labels)))
    width = 0.36
    accuracy = [float(row["accuracy"]) for row in rows]
    fpr = [float(row["false_positive_rate"]) for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5.1))
    ax.bar([i - width / 2 for i in x], accuracy, width, color=PALETTE["green"], label="Accuracy")
    ax.bar([i + width / 2 for i in x], fpr, width, color=PALETTE["red"], label="False-positive rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Silver-Benchmark Case-Review Ablation")
    ax.set_ylabel("Rate")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color=PALETTE["grid"], alpha=0.7)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(accuracy):
        ax.text(index - width / 2, value + 0.025, f"{value:.2f}", ha="center", fontsize=8.5)
    for index, value in enumerate(fpr):
        ax.text(index + width / 2, value + 0.025, f"{value:.2f}", ha="center", fontsize=8.5)
    save(fig, path)
    return figure_record(
        path,
        "graphrag_ablation",
        "Compares Graph RAG, vector retrieval, rule-plus-LLM, and rule-only case-review conditions.",
        {
            condition_label(row).replace("\n", " "): {
                "accuracy": float(row["accuracy"]),
                "false_positive_rate": float(row["false_positive_rate"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
            }
            for row in rows
        },
    )


def write_graphrag_calibration(path: Path) -> dict[str, Any]:
    rows = ordered_ablation_rows()
    labels = [condition_label(row) for row in rows]
    x = list(range(len(labels)))
    width = 0.36
    brier = [float(row["brier"]) for row in rows]
    ece = [float(row["ece_5bin"]) for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5.1))
    ax.bar([i - width / 2 for i in x], brier, width, color=PALETTE["blue"], label="Brier score")
    ax.bar([i + width / 2 for i in x], ece, width, color=PALETTE["yellow"], label="ECE, 5 bins")
    ax.set_title("Case-Review Probability Diagnostics")
    ax.set_ylabel("Lower is better")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color=PALETTE["grid"], alpha=0.7)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(brier):
        ax.text(index - width / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=8.5)
    for index, value in enumerate(ece):
        ax.text(index + width / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=8.5)
    save(fig, path)
    return figure_record(
        path,
        "graphrag_calibration",
        "Reports Brier and five-bin expected-calibration-error diagnostics for the silver case-review benchmark.",
        {
            condition_label(row).replace("\n", " "): {
                "brier": float(row["brier"]),
                "ece_5bin": float(row["ece_5bin"]),
            }
            for row in rows
        },
    )


def build_figures() -> list[dict[str, Any]]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    return [
        write_framework(FIGURES / "paper4_framework_en.png"),
        write_data_scope(FIGURES / "paper4_data_scope_en.png"),
        write_model_performance(FIGURES / "paper4_model_performance_en.png"),
        write_permutation_importance(FIGURES / "paper4_permutation_importance_en.png"),
        write_market_coverage(FIGURES / "paper4_market_coverage_en.png"),
        write_sequence_market_reaction(FIGURES / "paper4_sequence_market_reaction_en.png"),
        write_graphrag_ablation(FIGURES / "paper4_graphrag_ablation_en.png"),
        write_graphrag_calibration(FIGURES / "paper4_graphrag_calibration_en.png"),
    ]


def build_audit(figures: list[dict[str, Any]]) -> dict[str, Any]:
    source_rows = []
    missing_sources = []
    for source in SOURCE_FILES:
        path = ROOT / source
        exists = path.exists()
        if not exists:
            missing_sources.append(source)
        source_rows.append({"path": source, "exists": exists, "size_bytes": path.stat().st_size if exists else 0})
    missing_figures = [row["path"] for row in figures if not Path(row["path"]).exists()]
    status = "pass" if not missing_sources and not missing_figures and len(figures) == 8 else "fail"
    return {
        "status": status,
        "figure_count": len(figures),
        "figures": figures,
        "source_files": source_rows,
        "missing_source_files": missing_sources,
        "missing_figures": missing_figures,
        "guardrail": (
            "Figures are generated from existing Paper 4 result artifacts and are descriptive submission exhibits, "
            "not new empirical claims beyond the audited manuscript tables."
        ),
    }


def write_markdown(audit: dict[str, Any]) -> None:
    lines = [
        "# Submission Figure Generation Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        f"Figure count: {audit['figure_count']}",
        "",
        "## Figures",
        "",
    ]
    for row in audit["figures"]:
        lines.append(f"- `{row['path']}` ({row['kind']}, {fmt_int(row['size_bytes'])} bytes)")
    lines.extend(
        [
            "",
            "## Source Files",
            "",
        ]
    )
    for row in audit["source_files"]:
        marker = "present" if row["exists"] else "missing"
        lines.append(f"- `{row['path']}`: {marker}")
    if audit["missing_source_files"]:
        lines.extend(["", "Missing source files:", ""])
        lines.extend(f"- `{path}`" for path in audit["missing_source_files"])
    lines.extend(["", audit["guardrail"], ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_style()
    RESULTS.mkdir(parents=True, exist_ok=True)
    figures = build_figures()
    audit = build_audit(figures)
    OUT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(audit)
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
