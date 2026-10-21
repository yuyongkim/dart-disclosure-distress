from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")


def configure_fonts() -> str:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
        plt.rcParams["font.family"] = family
    else:
        family = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return family


def read_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def metric_row(path: str, horizon: int, model: str) -> dict[str, Any]:
    rows = read_json(path)
    for row in rows:
        if int(row.get("horizon_days", -1)) == horizon and row.get("model") == model:
            return row
    raise KeyError(f"Missing metric row: {path} horizon={horizon} model={model}")


def write_framework(path: Path) -> dict[str, Any]:
    labels = [
        "DART 사건\n공시",
        "재무제표",
        "OHLCV와\n시장지수",
        "규제 T-code\n시퀀스",
        "시점 보존\n통제변수",
        "보정된\n순위화 모형",
        "상위 10%\n감시목록",
        "추적 가능한\n사례 검토",
    ]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)

    colors = {
        "source": "#E8EEF3",
        "feature": "#EEF5E8",
        "model": "#F6EDE1",
        "output": "#E8F1F5",
        "edge": "#27313A",
    }
    boxes = [
        (0.35, 3.0, 2.0, 1.0, labels[0], colors["source"]),
        (0.35, 1.55, 2.0, 1.0, labels[1], colors["source"]),
        (0.35, 0.25, 2.0, 1.0, labels[2], colors["source"]),
        (3.05, 2.85, 2.0, 1.25, labels[3], colors["feature"]),
        (3.05, 1.25, 2.0, 1.25, labels[4], colors["feature"]),
        (5.85, 2.05, 2.05, 1.25, labels[5], colors["model"]),
        (8.7, 2.05, 2.0, 1.25, labels[6], colors["output"]),
        (8.7, 0.55, 2.0, 1.0, labels[7], colors["output"]),
    ]
    for x, y, w, h, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.1,
            edgecolor=colors["edge"],
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

    arrows = [
        ((2.35, 3.5), (3.05, 3.5)),
        ((2.35, 2.05), (3.05, 1.9)),
        ((2.35, 0.75), (3.05, 1.6)),
        ((5.05, 3.45), (5.85, 2.75)),
        ((5.05, 1.85), (5.85, 2.45)),
        ((7.9, 2.65), (8.7, 2.65)),
        ((9.7, 2.05), (9.7, 1.55)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.2,
                color=colors["edge"],
            )
        )

    ax.text(
        5.5,
        4.55,
        "감사 가능한 SupTech 조기경보 워크플로",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
    )
    ax.text(
        5.5,
        0.12,
        "공시 사건의 추적가능성을 유지하면서 재무 및 시장가격 통제를 별도 계층으로 결합한다.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#46515A",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(path), "type": "framework", "language": "ko", "required_labels": labels}


def write_performance(path: Path) -> dict[str, Any]:
    rows = {
        "365 비시장 RF": metric_row("results/dart_model_metrics.json", 365, "combined_random_forest"),
        "365 시장 RF": metric_row(
            "results/market/dart_model_metrics.json",
            365,
            "combined_market_calibrated_random_forest",
        ),
        "730 비시장 RF": metric_row(
            "results/dart_model_metrics.json",
            730,
            "combined_calibrated_random_forest",
        ),
        "730 시장 RF": metric_row(
            "results/market/dart_model_metrics.json",
            730,
            "combined_market_calibrated_random_forest",
        ),
    }
    labels = list(rows)
    auc = [rows[label]["roc_auc"] for label in labels]
    ap = [rows[label]["average_precision"] for label in labels]
    lift = [rows[label]["top_10pct"]["lift"] for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 0.9]})
    x = range(len(labels))
    width = 0.36
    axes[0].bar([i - width / 2 for i in x], auc, width, label="ROC AUC", color="#3B6EA8")
    axes[0].bar([i + width / 2 for i in x], ap, width, label="평균정밀도", color="#6EA85B")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("판별력과 정밀도")
    axes[0].set_xticks(list(x), labels, rotation=18, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].bar(labels, lift, color=["#91A8C2", "#2F6F9F", "#91A8C2", "#2F6F9F"])
    axes[1].set_title("상위 10% 감시목록 리프트")
    axes[1].set_ylabel("리프트")
    axes[1].set_xticks(list(x), labels, rotation=18, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    for idx, value in enumerate(lift):
        axes[1].text(idx, value + 0.12, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("주요 시계열 외 표본 성능 비교", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(path),
        "type": "performance",
        "language": "ko",
        "required_labels": ["평균정밀도", "판별력과 정밀도", "상위 10% 감시목록 리프트", "주요 시계열 외 표본 성능 비교"],
        "values": {
            label: {
                "roc_auc": rows[label]["roc_auc"],
                "average_precision": rows[label]["average_precision"],
                "top_10pct_lift": rows[label]["top_10pct"]["lift"],
            }
            for label in labels
        },
    }


def write_market_coverage(path: Path) -> dict[str, Any]:
    naver = read_json("data/raw/market_coverage/naver_ohlcv_supplement_summary.json")
    fdr = read_json("data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json")
    panel = read_json("data/model_market/dart_model_panel_market_summary.json")
    residual = read_json("results/market_residual_coverage_audit.json")

    base_rows = panel["base_ohlcv_rows_loaded_before_supplement"]
    naver_rows = naver["output_rows_total"]
    fdr_rows = fdr["output_rows_total"]
    coverage = panel["price_record_rate"] * 100
    recent = panel["recent_price_record_rate"] * 100
    residual_share = residual["main"]["residual_row_share"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 0.85]})
    axes[0].barh(["로드된 OHLCV 행"], [base_rows], color="#6B8E9E", label="기본 SQLite")
    axes[0].barh(["로드된 OHLCV 행"], [naver_rows], left=[base_rows], color="#A2B86C", label="Naver 보조")
    axes[0].barh(["로드된 OHLCV 행"], [fdr_rows], left=[base_rows + naver_rows], color="#D39B55", label="FDR gap 보강")
    axes[0].set_xlabel("행 수")
    axes[0].set_title("OHLCV 원천 구성")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].ticklabel_format(axis="x", style="plain")

    bars = axes[1].bar(
        ["Any price\nrecord", "Recent price\nrecord", "Residual\nno-row anchors"],
        [coverage, recent, residual_share],
        color=["#2F6F9F", "#6EA85B", "#C86F5A"],
    )
    axes[1].set_ylim(0, 105)
    axes[1].set_xticks(range(3), ["가격기록\n존재", "최근 가격\n기록", "잔여 무자료\n앵커"])
    axes[1].set_ylabel("기업-연도 앵커 비율(%)")
    axes[1].set_title("시장패널 커버리지")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, [coverage, recent, residual_share]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}%", ha="center", fontsize=9)

    fig.suptitle("시장가격 확장 커버리지", fontsize=14, weight="bold")
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(path),
        "type": "market_coverage",
        "language": "ko",
        "required_labels": ["로드된 OHLCV 행", "기본 SQLite", "Naver 보조", "FDR gap 보강", "시장패널 커버리지", "시장가격 확장 커버리지"],
        "values": {
            "base_ohlcv_rows": base_rows,
            "naver_rows": naver_rows,
            "fdr_rows": fdr_rows,
            "price_record_rate": panel["price_record_rate"],
            "recent_price_record_rate": panel["recent_price_record_rate"],
            "residual_row_share": residual["main"]["residual_row_share"],
        },
    }


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# Paper Figure Generation Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "| Figure | Language | Source | Output | Required labels |",
        "|---|---|---|---|---|",
    ]
    for row in audit["figures"]:
        labels = ", ".join(str(label).replace("\n", " ") for label in row.get("required_labels", []))
        lines.append(f"| {row['type']} | `{row.get('language', '')}` | calculation/model artifacts | `{row['path']}` | {labels} |")
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "Figures are Korean-labeled derivatives of audited Paper 4 JSON artifacts and do not introduce new empirical values beyond the calculation sheet and model summaries.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    font_family = configure_fonts()
    figures = [
        write_framework(FIGURES / "paper4_framework.png"),
        write_performance(FIGURES / "paper4_model_performance.png"),
        write_market_coverage(FIGURES / "paper4_market_coverage.png"),
    ]
    missing = [row["path"] for row in figures if not Path(row["path"]).exists()]
    non_korean = [row["path"] for row in figures if row.get("language") != "ko"]
    audit = {
        "status": "pass" if not missing and not non_korean else "fail",
        "figure_count": len(figures),
        "figure_language": "ko",
        "font_family": font_family,
        "missing": missing,
        "non_korean": non_korean,
        "figures": figures,
    }
    (RESULTS / "figure_generation_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(audit, RESULTS / "figure_generation_audit.md")
    print(json.dumps({"status": audit["status"], "figure_count": audit["figure_count"]}, indent=2))
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
