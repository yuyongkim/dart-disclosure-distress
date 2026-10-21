from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path("Paper 4")
FCHART_URL = "https://fchart.stock.naver.com/sise.nhn"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Paper4 market coverage collector"
}


@dataclass(frozen=True)
class SymbolPlan:
    symbol: str
    keep_start: str
    keep_end: str
    base_rows_coverage: int
    base_rows_total: int
    base_min_date: str | None
    base_max_date: str | None


def normalize_date(value: object) -> str:
    return str(value).replace("-", "").replace("/", "").strip()[:8]


def read_panel_symbols(paths: list[Path]) -> list[str]:
    symbols: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        panel = pd.read_parquet(path, columns=["security_id"])
        extracted = panel["security_id"].dropna().astype(str).str.extract(r"(\d+)")[0].dropna()
        symbols.update(extracted.astype(str).str.zfill(6))
    return sorted(symbol for symbol in symbols if symbol and symbol != "nan")


def base_coverage(
    db_path: Path,
    symbols: list[str],
    coverage_start: str,
    coverage_end: str,
) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("create temp table wanted(symbol text primary key)")
        cur.executemany("insert into wanted(symbol) values (?)", [(symbol,) for symbol in symbols])
        query = """
            select
                w.symbol,
                coalesce(sum(case
                    when substr(o.trade_date, 1, 8) between ? and ? then 1
                    else 0
                end), 0) as rows_coverage,
                count(o.trade_date) as rows_total,
                min(substr(o.trade_date, 1, 8)) as min_date,
                max(substr(o.trade_date, 1, 8)) as max_date
            from wanted w
            left join ohlcv o on o.symbol = w.symbol
            group by w.symbol
            order by w.symbol
        """
        return pd.read_sql_query(query, con, params=(coverage_start, coverage_end))
    finally:
        con.close()


def build_symbol_plan(
    coverage: pd.DataFrame,
    full_start: str,
    full_end: str,
    coverage_start: str,
    coverage_end: str,
    min_coverage_rows: int,
) -> list[SymbolPlan]:
    plans: list[SymbolPlan] = []
    for row in coverage.itertuples(index=False):
        rows_coverage = int(row.rows_coverage or 0)
        rows_total = int(row.rows_total or 0)
        if rows_coverage >= min_coverage_rows:
            continue
        keep_end = full_end if rows_total == 0 else coverage_end
        plans.append(
            SymbolPlan(
                symbol=str(row.symbol).zfill(6),
                keep_start=full_start,
                keep_end=keep_end,
                base_rows_coverage=rows_coverage,
                base_rows_total=rows_total,
                base_min_date=row.min_date,
                base_max_date=row.max_date,
            )
        )
    return plans


def parse_fchart(raw: bytes) -> list[dict[str, object]]:
    text = raw.decode("euc-kr", errors="replace")
    text = text.replace('encoding="EUC-KR"', 'encoding="UTF-8"')
    text = text.replace('encoding="euc-kr"', 'encoding="UTF-8"')
    text = text.lstrip()
    root = ET.fromstring(text.encode("utf-8"))
    rows: list[dict[str, object]] = []
    for item in root.findall(".//item"):
        parts = item.attrib.get("data", "").split("|")
        if len(parts) < 6:
            continue
        try:
            rows.append(
                {
                    "trade_date": normalize_date(parts[0]),
                    "open": float(parts[1]),
                    "high": float(parts[2]),
                    "low": float(parts[3]),
                    "close": float(parts[4]),
                    "volume": int(float(parts[5])),
                }
            )
        except (ValueError, TypeError):
            continue
    return rows


def fetch_symbol(symbol: str, count: int, retries: int) -> tuple[list[dict[str, object]], str]:
    url = f"{FCHART_URL}?symbol={symbol}&timeframe=day&count={count}&requestType=0"
    last_error = ""
    for attempt in range(retries + 1):
        try:
            with urlopen(Request(url, headers=HEADERS), timeout=20) as response:
                return parse_fchart(response.read()), ""
        except Exception as exc:  # Network/API variance is recorded in the status file.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    return [], last_error


def process_plan(plan: SymbolPlan, count: int, retries: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    fetched, error = fetch_symbol(plan.symbol, count=count, retries=retries)
    kept = [
        {
            "symbol": plan.symbol,
            **row,
            "source": "naver_fchart",
        }
        for row in fetched
        if plan.keep_start <= str(row["trade_date"]) <= plan.keep_end
    ]
    status = "fetched" if kept else "no_rows"
    if error and not kept:
        status = "error"
    return kept, {
        "symbol": plan.symbol,
        "status": status,
        "rows_fetched": len(fetched),
        "rows_saved": len(kept),
        "keep_start": plan.keep_start,
        "keep_end": plan.keep_end,
        "base_rows_coverage": plan.base_rows_coverage,
        "base_rows_total": plan.base_rows_total,
        "base_min_date": plan.base_min_date or "",
        "base_max_date": plan.base_max_date or "",
        "error": error,
    }


def existing_done_symbols(status_path: Path, output_path: Path) -> set[str]:
    done: set[str] = set()
    if status_path.exists():
        status = pd.read_csv(status_path, dtype={"symbol": str})
        done.update(status.loc[status["status"].isin(["fetched", "no_rows"]), "symbol"].astype(str).str.zfill(6))
    elif output_path.exists():
        rows = pd.read_csv(output_path, usecols=["symbol"], dtype={"symbol": str})
        done.update(rows["symbol"].astype(str).str.zfill(6).unique())
    return done


def write_markdown(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# Market OHLCV Coverage Collection",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Panel symbols | {summary['panel_symbols']} |",
        f"| Target symbols | {summary['target_symbols']} |",
        f"| Attempted symbols | {summary['attempted_symbols']} |",
        f"| Fetched symbols | {summary['fetched_symbols']} |",
        f"| No-row symbols | {summary['no_row_symbols']} |",
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
        "Naver fchart OHLCV is used as a supplemental source. Symbols with enough 2015-2017 rows in the base OHLCV DB are skipped. Symbols already present in the base DB are saved for the historical coverage window only; symbols absent from the base DB are saved through the full requested end date.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel-path",
        action="append",
        default=[
            str(ROOT / "data" / "model" / "dart_model_panel.parquet"),
            str(ROOT / "data" / "model_robust" / "dart_model_panel.parquet"),
        ],
    )
    parser.add_argument(
        "--base-db",
        default=str(Path.home() / "Desktop" / "Company_Credit" / "data" / "ohlcv.db"),
    )
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "raw" / "market_coverage"))
    parser.add_argument("--full-start", default="20140101")
    parser.add_argument("--full-end", default="20241231")
    parser.add_argument("--coverage-start", default="20150101")
    parser.add_argument("--coverage-end", default="20171231")
    parser.add_argument("--min-coverage-rows", type=int, default=200)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "naver_ohlcv_supplement.csv"
    status_path = out_dir / "naver_ohlcv_supplement_status.csv"
    summary_json_path = out_dir / "naver_ohlcv_supplement_summary.json"
    summary_md_path = out_dir / "naver_ohlcv_supplement_summary.md"

    panel_paths = [Path(path) for path in args.panel_path]
    symbols = read_panel_symbols(panel_paths)
    coverage = base_coverage(Path(args.base_db), symbols, args.coverage_start, args.coverage_end)
    plans = build_symbol_plan(
        coverage,
        full_start=args.full_start,
        full_end=args.full_end,
        coverage_start=args.coverage_start,
        coverage_end=args.coverage_end,
        min_coverage_rows=args.min_coverage_rows,
    )
    if not args.refresh:
        done = existing_done_symbols(status_path, output_path)
        plans = [plan for plan in plans if plan.symbol not in done]
    if args.limit > 0:
        plans = plans[: args.limit]

    data_mode = "w" if args.refresh or not output_path.exists() else "a"
    status_mode = "w" if args.refresh or not status_path.exists() else "a"
    data_fields = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "source"]
    status_fields = [
        "symbol",
        "status",
        "rows_fetched",
        "rows_saved",
        "keep_start",
        "keep_end",
        "base_rows_coverage",
        "base_rows_total",
        "base_min_date",
        "base_max_date",
        "error",
    ]

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

        print(f"Target symbols: {len(plans)}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_plan, plan, args.count, args.retries): plan.symbol for plan in plans}
            for idx, future in enumerate(as_completed(futures), start=1):
                rows, status = future.result()
                if rows:
                    data_writer.writerows(rows)
                    rows_saved += len(rows)
                status_writer.writerow(status)
                if idx % 100 == 0 or idx == len(plans):
                    elapsed = time.time() - t0
                    rate = idx / elapsed if elapsed else 0.0
                    remain = (len(plans) - idx) / rate if rate else 0.0
                    print(
                        f"[{idx}/{len(plans)}] rows_saved={rows_saved:,} elapsed={elapsed:.0f}s eta={remain:.0f}s",
                        flush=True,
                    )

    status_df = pd.read_csv(status_path, dtype={"symbol": str}) if status_path.exists() else pd.DataFrame()
    output_rows_total = int(sum(1 for _ in output_path.open("r", encoding="utf-8-sig")) - 1) if output_path.exists() else 0
    summary = {
        "status": "collected",
        "base_db": "external_ky_platform_ohlcv_db",
        "panel_paths": [str(path) for path in panel_paths],
        "panel_symbols": len(symbols),
        "target_symbols": int(len(build_symbol_plan(
            coverage,
            full_start=args.full_start,
            full_end=args.full_end,
            coverage_start=args.coverage_start,
            coverage_end=args.coverage_end,
            min_coverage_rows=args.min_coverage_rows,
        ))),
        "attempted_symbols": int(len(plans)),
        "fetched_symbols": int(status_df["status"].eq("fetched").sum()) if not status_df.empty else 0,
        "no_row_symbols": int(status_df["status"].eq("no_rows").sum()) if not status_df.empty else 0,
        "error_symbols": int(status_df["status"].eq("error").sum()) if not status_df.empty else 0,
        "rows_saved_this_run": rows_saved,
        "output_rows_total": output_rows_total,
        "output_path": str(output_path),
        "status_path": str(status_path),
        "summary_json_path": str(summary_json_path),
        "coverage_window": [args.coverage_start, args.coverage_end],
        "full_window": [args.full_start, args.full_end],
        "min_coverage_rows": args.min_coverage_rows,
    }
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, summary_md_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
