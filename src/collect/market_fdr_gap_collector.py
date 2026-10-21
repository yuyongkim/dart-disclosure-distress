from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("Paper 4")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def normalize_date(value: object) -> str:
    return str(value).replace("-", "").replace("/", "").strip()[:8]


def read_gap_symbols(status_path: Path) -> list[str]:
    if not status_path.exists():
        raise FileNotFoundError(f"Missing Naver coverage status file: {status_path}")
    status = pd.read_csv(status_path, dtype={"symbol": str})
    if "symbol" not in status.columns or "status" not in status.columns:
        raise ValueError(f"Expected symbol/status columns in {status_path}")
    symbols = (
        status.loc[status["status"].eq("no_rows"), "symbol"]
        .dropna()
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .dropna()
        .str.zfill(6)
        .unique()
    )
    return sorted(symbols)


def existing_done_symbols(status_path: Path, output_path: Path) -> set[str]:
    done: set[str] = set()
    if status_path.exists():
        status = pd.read_csv(status_path, dtype={"symbol": str})
        done.update(status.loc[status["status"].isin(["fetched", "no_rows"]), "symbol"].astype(str).str.zfill(6))
    elif output_path.exists():
        rows = pd.read_csv(output_path, usecols=["symbol"], dtype={"symbol": str})
        done.update(rows["symbol"].astype(str).str.zfill(6).unique())
    return done


def fetch_symbol(symbol: str, start: str, end: str, retries: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    import FinanceDataReader as fdr

    last_error = ""
    raw_rows = 0
    for attempt in range(retries + 1):
        try:
            df = fdr.DataReader(symbol, start, end)
            raw_rows = int(len(df))
            if df.empty:
                return [], {
                    "symbol": symbol,
                    "status": "no_rows",
                    "rows_fetched": 0,
                    "rows_saved": 0,
                    "min_trade_date": "",
                    "max_trade_date": "",
                    "error": "",
                }
            rows = normalize_frame(symbol, df)
            return rows, {
                "symbol": symbol,
                "status": "fetched" if rows else "no_rows",
                "rows_fetched": raw_rows,
                "rows_saved": len(rows),
                "min_trade_date": min((str(row["trade_date"]) for row in rows), default=""),
                "max_trade_date": max((str(row["trade_date"]) for row in rows), default=""),
                "error": "",
            }
        except Exception as exc:  # Network/API variance is recorded for reproducibility.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    return [], {
        "symbol": symbol,
        "status": "error",
        "rows_fetched": raw_rows,
        "rows_saved": 0,
        "min_trade_date": "",
        "max_trade_date": "",
        "error": last_error,
    }


def normalize_frame(symbol: str, df: pd.DataFrame) -> list[dict[str, object]]:
    df = df.reset_index()
    rename = {
        "Date": "trade_date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Change": "source_change",
    }
    df = df.rename(columns={col: rename.get(col, col) for col in df.columns})
    if "trade_date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "trade_date"})
    df["trade_date"] = df["trade_date"].map(normalize_date)
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = df.dropna(subset=["trade_date", "close"]).copy()
    keep = keep[keep["close"].gt(0)]
    keep = keep.sort_values("trade_date")
    rows: list[dict[str, object]] = []
    for row in keep.itertuples(index=False):
        rows.append(
            {
                "symbol": symbol,
                "trade_date": str(getattr(row, "trade_date")),
                "open": none_if_na(getattr(row, "open")),
                "high": none_if_na(getattr(row, "high")),
                "low": none_if_na(getattr(row, "low")),
                "close": none_if_na(getattr(row, "close")),
                "volume": none_if_na(getattr(row, "volume")),
                "source": "finance_datareader",
            }
        )
    return rows


def none_if_na(value: object) -> object:
    return None if pd.isna(value) else value


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# FDR OHLCV Gap Collection",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Source | {summary['source']} |",
        f"| FinanceDataReader version | {summary.get('finance_datareader_version') or 'unknown'} |",
        f"| Naver no-row target symbols | {summary['target_symbols']} |",
        f"| Attempted symbols | {summary['attempted_symbols']} |",
        f"| Fetched symbols | {summary['fetched_symbols']} |",
        f"| Residual no-row symbols | {summary['no_row_symbols']} |",
        f"| Error symbols | {summary['error_symbols']} |",
        f"| Rows saved this run | {summary['rows_saved_this_run']} |",
        f"| Output rows total | {summary['output_rows_total']} |",
        "",
        "## Outputs",
        "",
        f"- Data: `{summary['output_path']}`",
        f"- Status: `{summary['status_path']}`",
        f"- Summary JSON: `{summary['summary_json_path']}`",
        "",
        "## Method",
        "",
        "FinanceDataReader is used only for symbols that the Naver fchart supplemental collector marked as `no_rows`. The rows are stored as a separate Paper 4 artifact and are merged later by the market feature builder; the KY-PLATFORM/Company_Credit source SQLite database is not modified.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--naver-status-path",
        default=str(ROOT / "data" / "raw" / "market_coverage" / "naver_ohlcv_supplement_status.csv"),
    )
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "raw" / "market_coverage"))
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "fdr_ohlcv_gap_supplement.csv"
    status_path = out_dir / "fdr_ohlcv_gap_supplement_status.csv"
    summary_json_path = out_dir / "fdr_ohlcv_gap_supplement_summary.json"
    summary_md_path = out_dir / "fdr_ohlcv_gap_supplement_summary.md"

    all_symbols = read_gap_symbols(Path(args.naver_status_path))
    symbols = list(all_symbols)
    if not args.refresh:
        done = existing_done_symbols(status_path, output_path)
        symbols = [symbol for symbol in symbols if symbol not in done]
    if args.limit > 0:
        symbols = symbols[: args.limit]

    data_mode = "w" if args.refresh or not output_path.exists() else "a"
    status_mode = "w" if args.refresh or not status_path.exists() else "a"
    data_fields = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "source"]
    status_fields = ["symbol", "status", "rows_fetched", "rows_saved", "min_trade_date", "max_trade_date", "error"]

    rows_saved = 0
    t0 = time.time()
    with output_path.open(data_mode, encoding="utf-8-sig", newline="") as data_handle, status_path.open(
        status_mode, encoding="utf-8-sig", newline=""
    ) as status_handle:
        data_writer = csv.DictWriter(data_handle, fieldnames=data_fields)
        status_writer = csv.DictWriter(status_handle, fieldnames=status_fields)
        if data_mode == "w":
            data_writer.writeheader()
        if status_mode == "w":
            status_writer.writeheader()

        print(f"Target symbols: {len(symbols)}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(fetch_symbol, symbol, args.start, args.end, args.retries): symbol for symbol in symbols}
            for idx, future in enumerate(as_completed(futures), start=1):
                rows, status = future.result()
                if rows:
                    data_writer.writerows(rows)
                    rows_saved += len(rows)
                status_writer.writerow(status)
                if idx % 25 == 0 or idx == len(symbols):
                    elapsed = time.time() - t0
                    rate = idx / elapsed if elapsed else 0.0
                    remain = (len(symbols) - idx) / rate if rate else 0.0
                    print(
                        f"[{idx}/{len(symbols)}] rows_saved={rows_saved:,} elapsed={elapsed:.0f}s eta={remain:.0f}s",
                        flush=True,
                    )

    status_df = pd.read_csv(status_path, dtype={"symbol": str}) if status_path.exists() else pd.DataFrame()
    output_rows_total = int(sum(1 for _ in output_path.open("r", encoding="utf-8-sig")) - 1) if output_path.exists() else 0
    summary = {
        "status": "collected",
        "source": "FinanceDataReader",
        "finance_datareader_version": package_version("finance-datareader") or package_version("FinanceDataReader"),
        "naver_status_path": str(Path(args.naver_status_path)),
        "requested_window": [args.start, args.end],
        "target_symbols": int(len(all_symbols)),
        "attempted_symbols": int(len(symbols)),
        "fetched_symbols": int(status_df["status"].eq("fetched").sum()) if not status_df.empty else 0,
        "no_row_symbols": int(status_df["status"].eq("no_rows").sum()) if not status_df.empty else 0,
        "error_symbols": int(status_df["status"].eq("error").sum()) if not status_df.empty else 0,
        "rows_saved_this_run": int(rows_saved),
        "output_rows_total": int(output_rows_total),
        "output_path": str(output_path),
        "status_path": str(status_path),
        "summary_json_path": str(summary_json_path),
    }
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, summary_md_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
