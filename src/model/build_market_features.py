from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS = (20, 60, 120, 252)


@dataclass(frozen=True)
class Config:
    panel_path: Path
    ohlcv_db: Path
    supplemental_ohlcv: tuple[Path, ...]
    market_index_path: Path | None
    out_dir: Path
    max_trade_date: str


def safe_log1p(value: pd.Series) -> pd.Series:
    return np.log1p(value.clip(lower=0))


def normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str).str.slice(0, 8), format="%Y%m%d", errors="coerce")
    for col in ("close", "volume", "open", "high", "low"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["symbol", "trade_date", "close"])
    df = df[df["close"].gt(0)].copy()
    return df


def read_supplemental_ohlcv(path: Path | None, symbols: list[str], max_trade_date: str) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    wanted = set(symbols)
    usecols = ["symbol", "trade_date", "open", "high", "low", "close", "volume"]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, dtype={"symbol": str, "trade_date": str}, usecols=usecols, chunksize=500_000):
        chunk["symbol"] = chunk["symbol"].astype(str).str.zfill(6)
        chunk["trade_date_norm"] = chunk["trade_date"].astype(str).str.slice(0, 8)
        chunk = chunk[chunk["symbol"].isin(wanted) & chunk["trade_date_norm"].le(max_trade_date)].copy()
        chunk = chunk.drop(columns=["trade_date_norm"])
        if not chunk.empty:
            chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def read_supplemental_ohlcvs(paths: tuple[Path, ...], symbols: list[str], max_trade_date: str) -> pd.DataFrame:
    frames = [read_supplemental_ohlcv(path, symbols, max_trade_date) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def read_market_index(path: Path | None, max_trade_date: str) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"trade_date": str, "index_symbol": str})
    if df.empty or "trade_date" not in df.columns or "index_symbol" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str).str.slice(0, 8), format="%Y%m%d", errors="coerce")
    max_date = pd.to_datetime(max_trade_date, format="%Y%m%d", errors="coerce")
    if pd.notna(max_date):
        df = df[df["trade_date"].le(max_date)].copy()
    for col in ("close", "marcap"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["trade_date", "index_symbol", "close"])
    df = df[df["close"].gt(0)].sort_values(["index_symbol", "trade_date"])
    if df.empty:
        return pd.DataFrame()

    df["daily_ret"] = df.groupby("index_symbol")["close"].pct_change()
    df["daily_ret"] = df["daily_ret"].replace([np.inf, -np.inf], np.nan)
    df["weight"] = df.groupby("index_symbol")["marcap"].shift(1)
    rows: list[dict[str, object]] = []
    valid = df.dropna(subset=["daily_ret"]).copy()
    for trade_date, group in valid.groupby("trade_date", sort=True):
        returns = group["daily_ret"].astype(float)
        weights = group["weight"].astype(float)
        weighted = weights.notna() & weights.gt(0) & returns.notna()
        if weighted.any() and float(weights.loc[weighted].sum()) > 0:
            market_ret = float(np.average(returns.loc[weighted], weights=weights.loc[weighted]))
            weight_method = "prior_market_cap_weighted"
        else:
            market_ret = float(returns.mean())
            weight_method = "equal_weighted_fallback"
        rows.append(
            {
                "trade_date": trade_date,
                "market_daily_ret": market_ret,
                "market_index_symbols": ",".join(sorted(group["index_symbol"].astype(str).unique())),
                "market_index_weight_method": weight_method,
            }
        )

    benchmark = pd.DataFrame(rows)
    benchmark.attrs.update(
        {
            "market_benchmark_source": "kospi_kosdaq_index",
            "market_benchmark_method": "prior_market_cap_weighted",
            "market_index_rows_loaded": int(len(df)),
            "market_index_symbols_loaded": int(df["index_symbol"].nunique()),
            "market_index_min_date": str(df["trade_date"].min().date()),
            "market_index_max_date": str(df["trade_date"].max().date()),
        }
    )
    return benchmark


def panel_median_market_proxy(df: pd.DataFrame) -> pd.DataFrame:
    market = (
        df.dropna(subset=["daily_ret"])
        .groupby("trade_date", as_index=False)["daily_ret"]
        .median()
        .rename(columns={"daily_ret": "market_daily_ret"})
    )
    market.attrs.update(
        {
            "market_benchmark_source": "covered_panel_median_fallback",
            "market_benchmark_method": "covered_panel_median",
            "market_index_rows_loaded": 0,
            "market_index_symbols_loaded": 0,
            "market_index_min_date": None,
            "market_index_max_date": None,
        }
    )
    return market


def read_ohlcv(
    db_path: Path,
    symbols: list[str],
    max_trade_date: str,
    supplemental_paths: tuple[Path, ...] = (),
    market_index_path: Path | None = None,
) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("create temp table wanted(symbol text primary key)")
        cur.executemany("insert into wanted(symbol) values (?)", [(symbol,) for symbol in symbols])
        query = """
            select o.symbol, o.trade_date, o.close, o.volume, o.open, o.high, o.low
            from ohlcv o
            join wanted w on o.symbol = w.symbol
            where substr(o.trade_date, 1, 8) <= ?
        """
        df = pd.read_sql_query(query, con, params=(max_trade_date,))
    finally:
        con.close()
    base_rows = len(df)
    supplemental = read_supplemental_ohlcvs(supplemental_paths, symbols, max_trade_date)
    supplemental_rows = len(supplemental)
    if not supplemental.empty:
        df = pd.concat([df, supplemental], ignore_index=True, sort=False)
    if df.empty:
        return df
    df = normalize_ohlcv_frame(df)
    if df.empty:
        return df
    df["_source_priority"] = 0
    if not supplemental.empty and base_rows < len(df):
        df.loc[df.index >= base_rows, "_source_priority"] = 1
    df = df.sort_values(["symbol", "trade_date", "_source_priority"])
    df = df.drop_duplicates(["symbol", "trade_date"], keep="last").drop(columns=["_source_priority"])
    df["daily_ret"] = df.groupby("symbol")["close"].pct_change()
    df["daily_ret"] = df["daily_ret"].replace([np.inf, -np.inf], np.nan)
    market = read_market_index(market_index_path, max_trade_date)
    if market.empty:
        market = panel_median_market_proxy(df)
    df = df.merge(market[["trade_date", "market_daily_ret"]], on="trade_date", how="left")
    df["log_ret"] = np.log1p(df["daily_ret"].clip(lower=-0.999999))
    df["market_log_ret"] = np.log1p(df["market_daily_ret"].clip(lower=-0.999999))
    df["excess_log_ret"] = df["log_ret"] - df["market_log_ret"]
    df.attrs.update(market.attrs)
    df.attrs["market_index_coverage_rate"] = float(df["market_daily_ret"].notna().mean()) if not df.empty else None
    df.attrs["base_ohlcv_rows_loaded_before_supplement"] = int(base_rows)
    df.attrs["supplemental_ohlcv_rows_loaded"] = int(supplemental_rows)
    df.attrs["supplemental_ohlcv_paths"] = [str(path) for path in supplemental_paths]
    return df


def max_drawdown(close: np.ndarray) -> float:
    close = close[np.isfinite(close) & (close > 0)]
    if close.size < 2:
        return np.nan
    running_max = np.maximum.accumulate(close)
    drawdown = close / running_max - 1.0
    return float(np.nanmin(drawdown))


def window_slice(idx: int, window: int) -> slice:
    start = max(0, idx - window + 1)
    return slice(start, idx + 1)


def feature_row(group: pd.DataFrame, anchor_date: pd.Timestamp) -> dict[str, float | str | None]:
    dates = group["trade_date"].to_numpy(dtype="datetime64[ns]")
    idx = int(np.searchsorted(dates, np.datetime64(anchor_date), side="right") - 1)
    if idx < 0:
        rec: dict[str, float | str | None] = {
            "mkt_has_price_record": 0.0,
            "mkt_missing_price_flag": 1.0,
            "mkt_stale_30d_flag": 1.0,
            "mkt_days_since_last_trade": 9999.0,
            "mkt_last_close_log": np.nan,
            "mkt_last_trade_date": None,
        }
        for window in WINDOWS:
            rec.update(empty_window_features(window))
        return rec

    last_date = pd.Timestamp(group.iloc[idx]["trade_date"])
    days_since = float((anchor_date - last_date).days)
    rec = {
        "mkt_has_price_record": 1.0,
        "mkt_missing_price_flag": 0.0,
        "mkt_stale_30d_flag": float(days_since > 30),
        "mkt_days_since_last_trade": days_since,
        "mkt_last_close_log": float(np.log(group.iloc[idx]["close"])),
        "mkt_last_trade_date": str(last_date.date()),
    }
    close = group["close"].to_numpy(dtype=float)
    volume = group["volume"].fillna(0).to_numpy(dtype=float)
    log_ret = group["log_ret"].to_numpy(dtype=float)
    excess_log_ret = group["excess_log_ret"].to_numpy(dtype=float)
    market_log_ret = group["market_log_ret"].to_numpy(dtype=float)
    daily_ret = group["daily_ret"].to_numpy(dtype=float)

    for window in WINDOWS:
        sl = window_slice(idx, window)
        n_obs = idx - sl.start + 1
        close_window = close[sl]
        volume_window = volume[sl]
        ret_window = daily_ret[sl]
        log_window = log_ret[sl]
        excess_window = excess_log_ret[sl]
        market_window = market_log_ret[sl]
        enough_for_return = idx - window >= 0 and np.isfinite(close[idx]) and np.isfinite(close[idx - window])
        stock_return = float(close[idx] / close[idx - window] - 1.0) if enough_for_return and close[idx - window] > 0 else np.nan
        market_return = float(np.expm1(np.nansum(market_window))) if np.isfinite(market_window).any() else np.nan
        excess_return = float(np.expm1(np.nansum(excess_window))) if np.isfinite(excess_window).any() else np.nan
        rec.update(
            {
                f"mkt_obs_{window}d": float(n_obs),
                f"mkt_return_{window}d": stock_return,
                f"mkt_market_return_{window}d": market_return,
                f"mkt_excess_return_{window}d": excess_return,
                f"mkt_volatility_{window}d": float(np.nanstd(ret_window, ddof=1)) if np.isfinite(ret_window).sum() > 1 else np.nan,
                f"mkt_excess_volatility_{window}d": float(np.nanstd(excess_window, ddof=1)) if np.isfinite(excess_window).sum() > 1 else np.nan,
                f"mkt_max_drawdown_{window}d": max_drawdown(close_window),
                f"mkt_log_volume_mean_{window}d": float(np.nanmean(np.log1p(np.clip(volume_window, 0, None)))) if volume_window.size else np.nan,
                f"mkt_zero_volume_share_{window}d": float(np.mean(volume_window <= 0)) if volume_window.size else np.nan,
                f"mkt_abs_return_mean_{window}d": float(np.nanmean(np.abs(ret_window))) if np.isfinite(ret_window).any() else np.nan,
                f"mkt_log_return_sum_{window}d": float(np.nansum(log_window)) if np.isfinite(log_window).any() else np.nan,
            }
        )
    if all(k in rec for k in ("mkt_log_volume_mean_20d", "mkt_log_volume_mean_120d")):
        rec["mkt_log_volume_ratio_20_120"] = float(rec["mkt_log_volume_mean_20d"] - rec["mkt_log_volume_mean_120d"])
    else:
        rec["mkt_log_volume_ratio_20_120"] = np.nan
    return rec


def empty_window_features(window: int) -> dict[str, float]:
    return {
        f"mkt_obs_{window}d": 0.0,
        f"mkt_return_{window}d": np.nan,
        f"mkt_market_return_{window}d": np.nan,
        f"mkt_excess_return_{window}d": np.nan,
        f"mkt_volatility_{window}d": np.nan,
        f"mkt_excess_volatility_{window}d": np.nan,
        f"mkt_max_drawdown_{window}d": np.nan,
        f"mkt_log_volume_mean_{window}d": np.nan,
        f"mkt_zero_volume_share_{window}d": np.nan,
        f"mkt_abs_return_mean_{window}d": np.nan,
        f"mkt_log_return_sum_{window}d": np.nan,
    }


def attach_market_features(panel: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    groups = {symbol: g.reset_index(drop=True) for symbol, g in ohlcv.groupby("symbol", sort=False)}
    rows: list[dict] = []
    for row in panel[["security_id", "anchor_date"]].itertuples(index=False):
        symbol = str(row.security_id).zfill(6)
        anchor_date = pd.Timestamp(row.anchor_date)
        group = groups.get(symbol)
        if group is None or group.empty:
            rec = {
                "mkt_has_price_record": 0.0,
                "mkt_missing_price_flag": 1.0,
                "mkt_stale_30d_flag": 1.0,
                "mkt_days_since_last_trade": 9999.0,
                "mkt_last_close_log": np.nan,
                "mkt_last_trade_date": None,
                "mkt_log_volume_ratio_20_120": np.nan,
            }
            for window in WINDOWS:
                rec.update(empty_window_features(window))
        else:
            rec = feature_row(group, anchor_date)
        rows.append(rec)
    market = pd.DataFrame(rows)
    return pd.concat([panel.reset_index(drop=True), market.reset_index(drop=True)], axis=1)


def build_summary(panel: pd.DataFrame, ohlcv: pd.DataFrame, config: Config) -> dict:
    market_cols = [
        col
        for col in panel.columns
        if col.startswith("mkt_") and col != "mkt_last_trade_date"
    ]
    if ohlcv.attrs.get("market_benchmark_source") == "kospi_kosdaq_index":
        benchmark_note = (
            "Market-adjusted returns use KOSPI/KOSDAQ composite index returns from the collected "
            "market-index artifact, weighted by prior-day index market capitalization where available."
        )
    else:
        benchmark_note = (
            "Market-adjusted returns fell back to the median daily return across covered Paper 4 panel "
            "securities because no market-index artifact was available."
        )
    return {
        "created_from": {
            "panel": str(config.panel_path),
            "ohlcv_db": "external_ky_platform_ohlcv_db",
            "supplemental_ohlcv": [str(path) for path in config.supplemental_ohlcv],
            "market_index": str(config.market_index_path) if config.market_index_path else None,
        },
        "max_trade_date": config.max_trade_date,
        "ohlcv_rows_loaded": int(len(ohlcv)),
        "ohlcv_symbols_loaded": int(ohlcv["symbol"].nunique()) if not ohlcv.empty else 0,
        "base_ohlcv_rows_loaded_before_supplement": ohlcv.attrs.get("base_ohlcv_rows_loaded_before_supplement"),
        "supplemental_ohlcv_rows_loaded": ohlcv.attrs.get("supplemental_ohlcv_rows_loaded"),
        "supplemental_ohlcv_paths": ohlcv.attrs.get("supplemental_ohlcv_paths"),
        "market_benchmark_source": ohlcv.attrs.get("market_benchmark_source"),
        "market_benchmark_method": ohlcv.attrs.get("market_benchmark_method"),
        "market_index_rows_loaded": ohlcv.attrs.get("market_index_rows_loaded"),
        "market_index_symbols_loaded": ohlcv.attrs.get("market_index_symbols_loaded"),
        "market_index_min_date": ohlcv.attrs.get("market_index_min_date"),
        "market_index_max_date": ohlcv.attrs.get("market_index_max_date"),
        "market_index_coverage_rate": ohlcv.attrs.get("market_index_coverage_rate"),
        "panel_rows": int(len(panel)),
        "panel_firms": int(panel["firm_id"].nunique()),
        "panel_securities": int(panel["security_id"].nunique()),
        "market_feature_columns": market_cols,
        "market_feature_count": int(len(market_cols)),
        "price_record_rate": float(panel["mkt_has_price_record"].mean()),
        "recent_price_record_rate": float((panel["mkt_has_price_record"].eq(1) & panel["mkt_stale_30d_flag"].eq(0)).mean()),
        "coverage_by_anchor_year": {
            str(year): {
                "rows": int(len(g)),
                "price_record_rate": float(g["mkt_has_price_record"].mean()),
                "recent_price_record_rate": float((g["mkt_has_price_record"].eq(1) & g["mkt_stale_30d_flag"].eq(0)).mean()),
            }
            for year, g in panel.groupby("anchor_year")
        },
        "method_notes": [
            "Market features use only the last available OHLCV records on or before each firm-year anchor date.",
            "Optional supplemental OHLCV CSV rows are concatenated with the base SQLite rows after symbol/date normalization.",
            "Duplicate symbol-date rows are de-duplicated by keeping supplemental rows after the base DB rows.",
            "Returns are trailing trading-day returns ending at the last available trade before the anchor.",
            benchmark_note,
            "Stale or missing price coverage is retained through explicit missing and stale flags to avoid dropping distressed or suspended firms mechanically.",
        ],
    }


def write_outputs(panel: pd.DataFrame, summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_dir / "dart_model_panel_market.parquet", index=False)
    panel.to_csv(out_dir / "dart_model_panel_market.csv", index=False, encoding="utf-8-sig")
    (out_dir / "dart_model_panel_market_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-path", default=str(Path("Paper 4") / "data" / "model" / "dart_model_panel.parquet"))
    parser.add_argument(
        "--ohlcv-db",
        default=str(Path.home() / "Desktop" / "Company_Credit" / "data" / "ohlcv.db"),
    )
    parser.add_argument(
        "--supplemental-ohlcv",
        default=str(Path("Paper 4") / "data" / "raw" / "market_coverage" / "naver_ohlcv_supplement.csv"),
    )
    parser.add_argument(
        "--extra-supplemental-ohlcv",
        action="append",
        default=[str(Path("Paper 4") / "data" / "raw" / "market_coverage" / "fdr_ohlcv_gap_supplement.csv")],
    )
    parser.add_argument("--no-supplemental-ohlcv", action="store_true")
    parser.add_argument(
        "--market-index-path",
        default=str(Path("Paper 4") / "data" / "raw" / "market_index" / "korea_market_indices.csv"),
    )
    parser.add_argument("--no-market-index", action="store_true")
    parser.add_argument("--out-dir", default=str(Path("Paper 4") / "data" / "model_market"))
    parser.add_argument("--max-trade-date", default="20241231")
    args = parser.parse_args()

    supplemental_paths = ()
    if not args.no_supplemental_ohlcv:
        supplemental_paths = tuple(
            Path(path)
            for path in [args.supplemental_ohlcv, *(args.extra_supplemental_ohlcv or [])]
            if path
        )

    config = Config(
        panel_path=Path(args.panel_path),
        ohlcv_db=Path(args.ohlcv_db),
        supplemental_ohlcv=supplemental_paths,
        market_index_path=None if args.no_market_index else Path(args.market_index_path),
        out_dir=Path(args.out_dir),
        max_trade_date=args.max_trade_date,
    )
    panel = pd.read_parquet(config.panel_path)
    panel["anchor_date"] = pd.to_datetime(panel["anchor_date"], errors="coerce")
    symbols = sorted(panel["security_id"].dropna().astype(str).str.extract(r"(\d+)")[0].str.zfill(6).unique())
    ohlcv = read_ohlcv(
        config.ohlcv_db,
        symbols,
        config.max_trade_date,
        config.supplemental_ohlcv,
        config.market_index_path,
    )
    out = attach_market_features(panel, ohlcv)
    summary = build_summary(out, ohlcv, config)
    write_outputs(out, summary, config.out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
