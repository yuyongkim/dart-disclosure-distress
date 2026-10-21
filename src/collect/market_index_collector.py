from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("Paper 4")
INDEXES = {
    "KS11": "KOSPI Composite",
    "KQ11": "KOSDAQ Composite",
}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def normalize_date(value: object) -> str:
    return str(value).replace("-", "").replace("/", "").strip()[:8]


def fetch_index(symbol: str, name: str, start: str, end: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    df = fdr.DataReader(symbol, start, end)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    rename = {
        "Date": "trade_date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Amount": "amount",
        "MarCap": "marcap",
        "Change": "source_change",
        "UpDown": "source_up_down",
        "Comp": "source_comp",
    }
    df = df.rename(columns={col: rename.get(col, col) for col in df.columns})
    if "trade_date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "trade_date"})
    df["trade_date"] = df["trade_date"].map(normalize_date)
    df["index_symbol"] = symbol
    df["index_name"] = name
    for col in ("open", "high", "low", "close", "volume", "amount", "marcap", "source_change"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = [
        "trade_date",
        "index_symbol",
        "index_name",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "marcap",
        "source_change",
    ]
    out = df[keep].copy()
    out = out.dropna(subset=["trade_date", "close"])
    out = out[out["close"].gt(0)].sort_values(["index_symbol", "trade_date"])
    out["daily_ret"] = out.groupby("index_symbol")["close"].pct_change()
    return out


def collect_indices(start: str, end: str) -> pd.DataFrame:
    frames = [fetch_index(symbol, name, start, end) for symbol, name in INDEXES.items()]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["index_symbol", "trade_date"])


def build_summary(df: pd.DataFrame, args: argparse.Namespace, output_path: Path, summary_json_path: Path) -> dict[str, Any]:
    rows_by_index = {
        symbol: int(count)
        for symbol, count in df.groupby("index_symbol")["trade_date"].size().sort_index().items()
    } if not df.empty else {}
    return {
        "status": "collected" if not df.empty else "empty",
        "source": "FinanceDataReader",
        "finance_datareader_version": package_version("finance-datareader") or package_version("FinanceDataReader"),
        "requested_window": [args.start, args.end],
        "index_symbols": INDEXES,
        "rows": int(len(df)),
        "rows_by_index": rows_by_index,
        "min_trade_date": str(df["trade_date"].min()) if not df.empty else None,
        "max_trade_date": str(df["trade_date"].max()) if not df.empty else None,
        "output_path": str(output_path),
        "summary_json_path": str(summary_json_path),
        "benchmark_method": (
            "The market feature builder uses KOSPI and KOSDAQ composite index daily returns "
            "as the Korean market benchmark, weighted by prior-day index market capitalization "
            "when available and equal-weighted when market-cap weights are unavailable."
        ),
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    rows_by_index = summary.get("rows_by_index") or {}
    lines = [
        "# Korean Market Index Collection",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Source | {summary['source']} |",
        f"| FinanceDataReader version | {summary.get('finance_datareader_version') or 'unknown'} |",
        f"| Requested start | {summary['requested_window'][0]} |",
        f"| Requested end | {summary['requested_window'][1]} |",
        f"| Rows | {summary['rows']} |",
        f"| Min trade date | {summary.get('min_trade_date') or ''} |",
        f"| Max trade date | {summary.get('max_trade_date') or ''} |",
        "",
        "## Rows by Index",
        "",
        "| Index | Name | Rows |",
        "|---|---|---:|",
    ]
    for symbol, name in INDEXES.items():
        lines.append(f"| `{symbol}` | {name} | {rows_by_index.get(symbol, 0)} |")
    lines.extend(
        [
            "",
            "## Method",
            "",
            str(summary["benchmark_method"]),
            "",
            "The collector stores this benchmark as a Paper 4 artifact under `data/raw/market_index/`; it does not mutate the KY-PLATFORM/Company_Credit OHLCV database.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "raw" / "market_index"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "korea_market_indices.csv"
    summary_json_path = out_dir / "korea_market_index_summary.json"
    summary_md_path = out_dir / "korea_market_index_summary.md"

    df = collect_indices(args.start, args.end)
    if df.empty:
        raise SystemExit("No KOSPI/KOSDAQ index rows were fetched.")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = build_summary(df, args, output_path, summary_json_path)
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, summary_md_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
