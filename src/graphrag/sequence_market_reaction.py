from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src" / "model"))

from build_market_features import read_ohlcv  # noqa: E402


PRE_WINDOWS = (20, 60)
POST_WINDOWS = (1, 5, 20, 60)


def normalize_id(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"[^A-Za-z0-9가-힣_]+", "_", text).strip("_")


def normalize_symbol(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    match = re.search(r"(\d+)", text)
    return match.group(1).zfill(6) if match else None


def finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def read_patterns() -> pd.DataFrame:
    occ = pd.read_csv(ROOT / "results" / "pattern_firm_occurrences.csv", encoding="utf-8-sig")
    dart = occ[occ["source"].eq("DART")].copy()
    dart["firm_id"] = dart["firm_id"].astype(str).str.extract(r"(\d+)")[0].astype(str)
    dart["first_pattern_date"] = pd.to_datetime(dart["first_pattern_date"], errors="coerce")
    dart = dart.dropna(subset=["firm_id", "first_pattern_date", "pattern"]).reset_index(drop=True)

    events = pd.read_csv(
        ROOT / "data" / "processed" / "events_dart.csv",
        usecols=["firm_id", "corp_name", "security_id", "stock_code"],
        encoding="utf-8-sig",
    )
    events["firm_id"] = events["firm_id"].astype(str).str.extract(r"(\d+)")[0].astype(str)
    events["symbol"] = events["security_id"].combine_first(events["stock_code"]).map(normalize_symbol)
    symbol_map = (
        events.dropna(subset=["firm_id", "symbol"])
        .sort_values(["firm_id", "symbol"])
        .groupby("firm_id", as_index=False)
        .agg(
            security_id=("symbol", "first"),
            stock_code=("symbol", "first"),
            event_corp_name=("corp_name", "first"),
        )
    )
    dart = dart.merge(symbol_map, on="firm_id", how="left")
    dart["firm_name"] = dart["firm_name"].fillna(dart["event_corp_name"])
    dart = dart.drop(columns=["event_corp_name"])
    dart["occurrence_id"] = [
        f"DART_REACTION_{i:05d}_{normalize_id(row.firm_id)}_{row.first_pattern_date:%Y%m%d}_{normalize_id(row.pattern)}"
        for i, row in enumerate(dart.itertuples(index=False), start=1)
    ]

    cases_path = ROOT / "data" / "labels" / "disclosure_sequence_distress_reasoning_benchmark.csv"
    if cases_path.exists():
        cases = pd.read_csv(cases_path, encoding="utf-8-sig")
        cases["firm_id"] = cases["firm_id"].astype(str).str.extract(r"(\d+)")[0].astype(str)
        cases["first_pattern_date"] = pd.to_datetime(cases["first_pattern_date"], errors="coerce")
        case_key = cases[["case_id", "firm_id", "rule_pattern", "first_pattern_date"]].rename(
            columns={"rule_pattern": "pattern"}
        )
        dart = dart.merge(case_key, on=["firm_id", "pattern", "first_pattern_date"], how="left")
    else:
        dart["case_id"] = np.nan
    return dart


def compounded(series: pd.Series | np.ndarray) -> float:
    values = pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return np.nan
    return float(np.expm1(values.sum()))


def return_from_close(close: np.ndarray, start_idx: int, end_idx: int) -> float:
    if start_idx < 0 or end_idx < 0 or start_idx >= len(close) or end_idx >= len(close):
        return np.nan
    start = close[start_idx]
    end = close[end_idx]
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0:
        return np.nan
    return float(end / start - 1.0)


def empty_reaction(row: pd.Series, reason: str) -> dict[str, object]:
    rec: dict[str, object] = {
        "occurrence_id": row["occurrence_id"],
        "case_id": row.get("case_id", np.nan),
        "source": row["source"],
        "firm_id": row["firm_id"],
        "firm_name": row["firm_name"],
        "security_id": row.get("security_id"),
        "stock_code": row.get("stock_code"),
        "pattern": row["pattern"],
        "pattern_length": row["pattern_length"],
        "first_pattern_date": row["first_pattern_date"].date().isoformat(),
        "has_outcome": row.get("has_outcome"),
        "outcome_date": row.get("outcome_date"),
        "outcome_label": row.get("outcome_label"),
        "lead_days": row.get("lead_days"),
        "reaction_available": False,
        "missing_reason": reason,
        "event_trade_date": None,
        "pre_close_date": None,
        "days_pattern_to_event_trade": np.nan,
        "pre_close": np.nan,
    }
    for window in PRE_WINDOWS:
        rec[f"pre_return_{window}d"] = np.nan
        rec[f"pre_market_return_{window}d"] = np.nan
        rec[f"pre_excess_return_{window}d"] = np.nan
        rec[f"pre_complete_{window}d"] = False
    for window in POST_WINDOWS:
        rec[f"post_return_{window}d"] = np.nan
        rec[f"post_market_return_{window}d"] = np.nan
        rec[f"post_excess_return_{window}d"] = np.nan
        rec[f"post_obs_{window}d"] = 0
        rec[f"post_complete_{window}d"] = False
    return rec


def reaction_for_row(row: pd.Series, group: pd.DataFrame) -> dict[str, object]:
    if group.empty:
        return empty_reaction(row, "no_ohlcv_rows_for_symbol")
    dates = pd.to_datetime(group["trade_date"]).to_numpy(dtype="datetime64[ns]")
    anchor = np.datetime64(pd.Timestamp(row["first_pattern_date"]))
    event_idx = int(np.searchsorted(dates, anchor, side="left"))
    if event_idx >= len(group):
        return empty_reaction(row, "no_trade_on_or_after_pattern_date")
    pre_idx = event_idx - 1
    if pre_idx < 0:
        return empty_reaction(row, "no_prior_trade_before_pattern_date")

    close = group["close"].to_numpy(dtype=float)
    log_ret = group["log_ret"].to_numpy(dtype=float)
    market_log_ret = group["market_log_ret"].to_numpy(dtype=float)
    excess_log_ret = group["excess_log_ret"].to_numpy(dtype=float)
    event_trade_date = pd.Timestamp(group.iloc[event_idx]["trade_date"])
    pre_close_date = pd.Timestamp(group.iloc[pre_idx]["trade_date"])
    days_to_trade = int((event_trade_date - pd.Timestamp(row["first_pattern_date"])).days)

    rec = empty_reaction(row, "")
    rec.update(
        {
            "reaction_available": True,
            "missing_reason": "",
            "event_trade_date": event_trade_date.date().isoformat(),
            "pre_close_date": pre_close_date.date().isoformat(),
            "days_pattern_to_event_trade": days_to_trade,
            "pre_close": finite_float(close[pre_idx]),
        }
    )

    for window in PRE_WINDOWS:
        start_idx = pre_idx - window
        complete = start_idx >= 0
        rec[f"pre_complete_{window}d"] = bool(complete)
        if complete:
            rec[f"pre_return_{window}d"] = return_from_close(close, start_idx, pre_idx)
            sl = slice(start_idx + 1, pre_idx + 1)
            rec[f"pre_market_return_{window}d"] = compounded(market_log_ret[sl])
            rec[f"pre_excess_return_{window}d"] = compounded(excess_log_ret[sl])

    for window in POST_WINDOWS:
        end_idx = min(event_idx + window - 1, len(group) - 1)
        obs = end_idx - event_idx + 1
        complete = obs >= window
        rec[f"post_obs_{window}d"] = int(max(0, obs))
        rec[f"post_complete_{window}d"] = bool(complete)
        if complete:
            rec[f"post_return_{window}d"] = return_from_close(close, pre_idx, end_idx)
            sl = slice(event_idx, end_idx + 1)
            rec[f"post_market_return_{window}d"] = compounded(market_log_ret[sl])
            rec[f"post_excess_return_{window}d"] = compounded(excess_log_ret[sl])
    return rec


def summarize(reactions: pd.DataFrame, all_source_rows: int, ohlcv: pd.DataFrame) -> dict[str, object]:
    def rate(col: str) -> float:
        return float(reactions[col].mean()) if col in reactions and len(reactions) else 0.0

    by_pattern = []
    for pattern, group in reactions.groupby("pattern", sort=False):
        by_pattern.append(
            {
                "pattern": pattern,
                "rows": int(len(group)),
                "firms": int(group["firm_id"].nunique()),
                "reaction_available_rate": float(group["reaction_available"].mean()),
                "post_5d_complete_rate": float(group["post_complete_5d"].mean()),
                "post_20d_complete_rate": float(group["post_complete_20d"].mean()),
                "mean_post_excess_return_5d": finite_float(group["post_excess_return_5d"].mean()),
                "median_post_excess_return_5d": finite_float(group["post_excess_return_5d"].median()),
                "mean_post_excess_return_20d": finite_float(group["post_excess_return_20d"].mean()),
                "median_post_excess_return_20d": finite_float(group["post_excess_return_20d"].median()),
            }
        )
    by_pattern = sorted(by_pattern, key=lambda x: (-x["rows"], str(x["pattern"])))

    case_rows = reactions[reactions["case_id"].notna()].copy()
    return {
        "all_source_pattern_occurrences": int(all_source_rows),
        "dart_pattern_occurrences": int(len(reactions)),
        "dart_unique_firms": int(reactions["firm_id"].nunique()),
        "dart_unique_patterns": int(reactions["pattern"].nunique()),
        "patterns_by_length": {str(k): int(v) for k, v in reactions["pattern_length"].value_counts().sort_index().items()},
        "security_mapped_rows": int(reactions["security_id"].notna().sum()),
        "ohlcv_rows_loaded": int(len(ohlcv)),
        "ohlcv_symbols_loaded": int(ohlcv["symbol"].nunique()) if not ohlcv.empty else 0,
        "base_ohlcv_rows_loaded_before_supplement": ohlcv.attrs.get("base_ohlcv_rows_loaded_before_supplement"),
        "supplemental_ohlcv_rows_loaded": ohlcv.attrs.get("supplemental_ohlcv_rows_loaded"),
        "market_benchmark_source": ohlcv.attrs.get("market_benchmark_source"),
        "reaction_available_rows": int(reactions["reaction_available"].sum()),
        "reaction_available_rate": rate("reaction_available"),
        "post_1d_complete_rows": int(reactions["post_complete_1d"].sum()),
        "post_5d_complete_rows": int(reactions["post_complete_5d"].sum()),
        "post_20d_complete_rows": int(reactions["post_complete_20d"].sum()),
        "post_60d_complete_rows": int(reactions["post_complete_60d"].sum()),
        "post_1d_complete_rate": rate("post_complete_1d"),
        "post_5d_complete_rate": rate("post_complete_5d"),
        "post_20d_complete_rate": rate("post_complete_20d"),
        "post_60d_complete_rate": rate("post_complete_60d"),
        "mean_post_excess_return_5d": finite_float(reactions["post_excess_return_5d"].mean()),
        "median_post_excess_return_5d": finite_float(reactions["post_excess_return_5d"].median()),
        "mean_post_excess_return_20d": finite_float(reactions["post_excess_return_20d"].mean()),
        "median_post_excess_return_20d": finite_float(reactions["post_excess_return_20d"].median()),
        "benchmark_case_rows": int(len(case_rows)),
        "benchmark_case_reaction_available_rows": int(case_rows["reaction_available"].sum()) if len(case_rows) else 0,
        "benchmark_case_post_20d_complete_rows": int(case_rows["post_complete_20d"].sum()) if len(case_rows) else 0,
        "missing_reasons": {
            str(k): int(v)
            for k, v in reactions.loc[~reactions["reaction_available"], "missing_reason"].value_counts().items()
        },
        "by_pattern_top20": by_pattern[:20],
        "method_notes": [
            "Anchor is the first_pattern_date from pattern_firm_occurrences.csv.",
            "Event trade date is the first trading day on or after the sequence-match date.",
            "Post-event returns are close-to-close returns from the prior trading close to +1/+5/+20/+60 trading sessions.",
            "Market-adjusted returns use the same KOSPI/KOSDAQ benchmark already used by the Paper 4 market extension.",
            "This is a sequence-match reaction screen, not a full disclosure-level event-study with intraday release timing or firm-specific expected-return models.",
        ],
    }


def pct(value: object) -> str:
    val = finite_float(value)
    return "" if val is None else f"{val:.2%}"


def write_summary_md(summary: dict[str, object], out_path: Path) -> None:
    lines = [
        "# Disclosure-Sequence Market Reaction Summary",
        "",
        "Status: generated",
        "",
        "## Scope",
        "",
        f"- All source pattern occurrences: {summary['all_source_pattern_occurrences']:,}",
        f"- DART pattern occurrences analyzed: {summary['dart_pattern_occurrences']:,}",
        f"- DART firms: {summary['dart_unique_firms']:,}",
        f"- DART patterns: {summary['dart_unique_patterns']:,}",
        f"- Security-mapped rows: {summary['security_mapped_rows']:,}",
        "",
        "## Price-Reaction Coverage",
        "",
        f"- OHLCV rows loaded: {summary['ohlcv_rows_loaded']:,}",
        f"- OHLCV symbols loaded: {summary['ohlcv_symbols_loaded']:,}",
        f"- Reaction available rows: {summary['reaction_available_rows']:,} ({pct(summary['reaction_available_rate'])})",
        f"- Complete +1d rows: {summary['post_1d_complete_rows']:,} ({pct(summary['post_1d_complete_rate'])})",
        f"- Complete +5d rows: {summary['post_5d_complete_rows']:,} ({pct(summary['post_5d_complete_rate'])})",
        f"- Complete +20d rows: {summary['post_20d_complete_rows']:,} ({pct(summary['post_20d_complete_rate'])})",
        f"- Complete +60d rows: {summary['post_60d_complete_rows']:,} ({pct(summary['post_60d_complete_rate'])})",
        "",
        "## Aggregate Reaction",
        "",
        f"- Mean +5d market-adjusted return: {pct(summary['mean_post_excess_return_5d'])}",
        f"- Median +5d market-adjusted return: {pct(summary['median_post_excess_return_5d'])}",
        f"- Mean +20d market-adjusted return: {pct(summary['mean_post_excess_return_20d'])}",
        f"- Median +20d market-adjusted return: {pct(summary['median_post_excess_return_20d'])}",
        "",
        "## Graph RAG Benchmark Cases",
        "",
        f"- Benchmark case rows matched: {summary['benchmark_case_rows']:,}",
        f"- Benchmark case reaction rows: {summary['benchmark_case_reaction_available_rows']:,}",
        f"- Benchmark case complete +20d rows: {summary['benchmark_case_post_20d_complete_rows']:,}",
        "",
        "## Top Pattern Coverage",
        "",
        "| Pattern | Rows | Firms | Reaction coverage | +5d median excess | +20d median excess |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["by_pattern_top20"]:
        lines.append(
            f"| {item['pattern']} | {item['rows']:,} | {item['firms']:,} | "
            f"{pct(item['reaction_available_rate'])} | {pct(item['median_post_excess_return_5d'])} | "
            f"{pct(item['median_post_excess_return_20d'])} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "These returns are sequence-match reaction screens anchored on `first_pattern_date`. They are useful as a market-reaction context layer for Graph RAG and as a descriptive robustness extension. They should not be described as a full security-level event study or causal abnormal-return design.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_reaction(row: pd.Series) -> dict[str, object]:
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
    out = {"reaction_available": bool(row.get("reaction_available", False)), "missing_reason": row.get("missing_reason", "")}
    for field in fields:
        value = row.get(field)
        out[field] = None if pd.isna(value) else value
    return out


def update_graph_contexts(case_reactions: pd.DataFrame, context_path: Path) -> None:
    if case_reactions.empty or not context_path.exists():
        return
    contexts = json.loads(context_path.read_text(encoding="utf-8"))
    for _, row in case_reactions.iterrows():
        case_id = str(row["case_id"])
        if case_id not in contexts:
            continue
        reaction = compact_reaction(row)
        ctx = contexts[case_id]
        ctx["market_reaction"] = reaction
        graph = ctx.setdefault("graph", {"nodes": [], "edges": []})
        graph["nodes"] = [n for n in graph.get("nodes", []) if n.get("type") != "MarketReaction"]
        graph["edges"] = [e for e in graph.get("edges", []) if e.get("type") != "HAS_MARKET_REACTION"]
        reaction_node_id = f"{case_id}_market_reaction"
        graph["nodes"].append({"id": reaction_node_id, "type": "MarketReaction", **reaction})
        graph["edges"].append({"source": case_id, "target": reaction_node_id, "type": "HAS_MARKET_REACTION"})
        graph_summary = ctx.setdefault("graph_summary", {})
        graph_summary["market_reaction"] = reaction
        vector_docs = [
            doc
            for doc in ctx.get("vector_documents", [])
            if not str(doc).startswith("Market reaction around sequence match:")
        ]
        if reaction["reaction_available"]:
            vector_docs.append(
                "Market reaction around sequence match: "
                f"+5d excess={pct(reaction.get('post_excess_return_5d'))}, "
                f"+20d excess={pct(reaction.get('post_excess_return_20d'))}, "
                f"pre-20d excess={pct(reaction.get('pre_excess_return_20d'))}."
            )
        else:
            vector_docs.append(f"Market reaction around sequence match unavailable: {reaction['missing_reason']}.")
        ctx["vector_documents"] = vector_docs
    context_path.write_text(json.dumps(contexts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = ROOT / "results" / "graphrag"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_occ = pd.read_csv(ROOT / "results" / "pattern_firm_occurrences.csv", usecols=["source"], encoding="utf-8-sig")
    patterns = read_patterns()
    symbols = sorted(s for s in patterns["security_id"].dropna().astype(str).unique() if s and s != "nan")
    ohlcv = read_ohlcv(
        Path(args.ohlcv_db),
        symbols,
        args.max_trade_date,
        tuple(Path(path) for path in args.supplemental_ohlcv if path),
        None if args.no_market_index else Path(args.market_index_path),
    )
    groups = {symbol: group.reset_index(drop=True) for symbol, group in ohlcv.groupby("symbol", sort=False)}

    rows = []
    for _, row in patterns.iterrows():
        symbol = row.get("security_id")
        if pd.isna(symbol):
            rows.append(empty_reaction(row, "missing_security_id"))
            continue
        group = groups.get(str(symbol).zfill(6))
        if group is None:
            rows.append(empty_reaction(row, "no_ohlcv_rows_for_symbol"))
        else:
            rows.append(reaction_for_row(row, group))
    reactions = pd.DataFrame(rows)

    reactions_path = out_dir / "sequence_market_reactions.csv"
    reactions.to_csv(reactions_path, index=False, encoding="utf-8-sig")
    case_reactions = reactions[reactions["case_id"].notna()].copy()
    case_reactions_path = out_dir / "case_market_reactions.csv"
    case_reactions.to_csv(case_reactions_path, index=False, encoding="utf-8-sig")

    summary = summarize(reactions, all_source_rows=len(all_occ), ohlcv=ohlcv)
    (out_dir / "sequence_market_reaction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_md(summary, out_dir / "sequence_market_reaction_summary.md")
    update_graph_contexts(case_reactions, out_dir / "case_contexts_graph.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute stock-price reactions around Paper 4 sequence matches.")
    parser.add_argument("--ohlcv-db", default=str(Path.home() / "Desktop" / "Company_Credit" / "data" / "ohlcv.db"))
    parser.add_argument(
        "--supplemental-ohlcv",
        action="append",
        default=[
            str(ROOT / "data" / "raw" / "market_coverage" / "naver_ohlcv_supplement.csv"),
            str(ROOT / "data" / "raw" / "market_coverage" / "fdr_ohlcv_gap_supplement.csv"),
        ],
    )
    parser.add_argument("--market-index-path", default=str(ROOT / "data" / "raw" / "market_index" / "korea_market_indices.csv"))
    parser.add_argument("--no-market-index", action="store_true")
    parser.add_argument("--max-trade-date", default="20241231")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
