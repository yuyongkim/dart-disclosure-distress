from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


K = {
    "correction": "\uc815\uc815",
    "attachment_added": "\ucca8\ubd80\ucd94\uac00",
    "attachment_correction": "\ucca8\ubd80\uc815\uc815",
    "extension": "\uc5f0\uc7a5",
    "withdrawal": "\ucca0\ud68c",
    "rehab": "\ud68c\uc0dd\uc808\ucc28",
    "default": "\ubd80\ub3c4\ubc1c\uc0dd",
    "dissolution": "\ud574\uc0b0\uc0ac\uc720",
    "delisting": "\uc0c1\uc7a5\ud3d0\uc9c0",
    "trading_suspension": "\uac70\ub798\uc815\uc9c0",
}

OUTCOME_SUBTYPES = {
    "rehabilitation",
    "dissolution",
    "default",
    "listing_or_trading_status",
}

TAXONOMY_CODES = ("T01", "T02", "T03", "T05", "T06", "T99")

FINANCIAL_RAW_COLUMNS = [
    "revenue",
    "operating_income",
    "net_income",
    "total_assets",
    "total_liabilities",
    "total_equity",
]

FINANCIAL_FEATURE_COLUMNS = [
    "fin_log_revenue",
    "fin_log_operating_income_signed",
    "fin_log_net_income_signed",
    "fin_log_total_assets",
    "fin_leverage",
    "fin_equity_ratio",
    "fin_roa",
    "fin_operating_roa",
    "fin_net_margin",
    "fin_operating_margin",
    "fin_loss_flag",
    "fin_negative_equity_flag",
    "fin_missing_record_flag",
]


@dataclass(frozen=True)
class Config:
    events_path: Path
    dart_raw_path: Path
    financial_path: Path
    out_dir: Path
    start_year: int
    end_year: int
    outcome_end_year: int
    outcome_mode: str


def contains(text: pd.Series, needle: str) -> pd.Series:
    return text.astype(str).str.contains(needle, regex=False, na=False)


def noncorrection_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for token in (
        K["correction"],
        K["attachment_added"],
        K["attachment_correction"],
        K["extension"],
        K["withdrawal"],
    ):
        mask &= ~contains(df["report_nm"], token)
    return mask


def outcome_subtype(report_name: str) -> str | None:
    title = str(report_name)
    if K["rehab"] in title:
        return "rehabilitation"
    if K["dissolution"] in title:
        return "dissolution"
    if K["default"] in title:
        return "default"
    if K["delisting"] in title or K["trading_suspension"] in title:
        return "listing_or_trading_status"
    return None


def outcome_label(subtype: str) -> str:
    return {
        "rehabilitation": "BANKRUPTCY_OR_REHABILITATION",
        "dissolution": "DISSOLUTION",
        "default": "DEFAULT",
        "listing_or_trading_status": "DELISTING_OR_TRADING_SUSPENSION",
    }[subtype]


def read_events(path: Path) -> pd.DataFrame:
    events = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    events = events.copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events["firm_id"] = events["firm_id"].astype(str).str.zfill(8)
    events["security_id"] = events["security_id"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    return events


def read_outcomes_from_raw(path: Path, start_year: int, outcome_end_year: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["outcome_date"] = pd.to_datetime(df["rcept_dt"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.loc[
        noncorrection_mask(df)
        & df["outcome_date"].dt.year.between(start_year, outcome_end_year, inclusive="both")
    ].copy()
    df["event_subtype"] = df["report_nm"].map(outcome_subtype)
    df = df[df["event_subtype"].notna()].copy()
    df["outcome_label"] = df["event_subtype"].map(outcome_label)
    df["firm_id"] = df["corp_code"].astype(str).str.zfill(8)
    df["security_id"] = df["stock_code"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    cols = [
        "firm_id",
        "corp_code",
        "corp_name",
        "security_id",
        "stock_code",
        "outcome_date",
        "outcome_label",
        "event_subtype",
        "report_nm",
        "rcept_no",
    ]
    return df[cols].sort_values(["firm_id", "outcome_date"])


def is_spac_like_name(name: pd.Series) -> pd.Series:
    text = name.fillna("").astype(str)
    return (
        text.str.contains("\uae30\uc5c5\uc778\uc218\ubaa9\uc801", regex=False)
        | text.str.contains("\uc2a4\ud329", regex=False)
        | text.str.upper().str.contains("SPAC", regex=False)
    )


def filter_outcomes(outcomes: pd.DataFrame, outcome_mode: str) -> tuple[pd.DataFrame, dict]:
    outcomes = outcomes.copy()
    if outcome_mode == "all":
        return outcomes, {
            "outcome_mode": outcome_mode,
            "excluded_rows": 0,
            "excluded_firms": 0,
            "exclusion_rules": [],
        }
    if outcome_mode == "robust":
        weak_t02 = outcomes["outcome_label"].eq("DELISTING_OR_TRADING_SUSPENSION")
        spac_dissolution = outcomes["outcome_label"].eq("DISSOLUTION") & is_spac_like_name(outcomes["corp_name"])
        excluded = weak_t02 | spac_dissolution
        filtered = outcomes.loc[~excluded].copy()
        return filtered, {
            "outcome_mode": outcome_mode,
            "excluded_rows": int(excluded.sum()),
            "excluded_firms": int(outcomes.loc[excluded, "firm_id"].nunique()),
            "exclusion_rules": [
                "Exclude all DELISTING_OR_TRADING_SUSPENSION outcomes because T02 includes weak listing-status proxies such as overseas listing termination notices.",
                "Exclude DISSOLUTION outcomes for SPAC-like firms whose names contain Korean enterprise-acquisition-purpose terms, Korean SPAC spelling, or SPAC.",
            ],
            "excluded_by_rule": {
                "weak_t02_rows": int(weak_t02.sum()),
                "weak_t02_firms": int(outcomes.loc[weak_t02, "firm_id"].nunique()),
                "spac_dissolution_rows": int(spac_dissolution.sum()),
                "spac_dissolution_firms": int(outcomes.loc[spac_dissolution, "firm_id"].nunique()),
            },
        }
    raise ValueError(f"Unsupported outcome_mode: {outcome_mode}")


def financial_available_date(df: pd.DataFrame) -> pd.Series:
    report_date = pd.to_datetime(df["report_date"], errors="coerce")
    report_type = df["report_type"].astype(str)
    lag_days = np.where(report_type.eq("\uc0ac\uc5c5\ubcf4\uace0\uc11c"), 120, 60)
    return report_date + pd.to_timedelta(lag_days, unit="D")


def signed_log1p(value: pd.Series) -> pd.Series:
    return np.sign(value) * np.log1p(np.abs(value))


def read_financials(path: Path) -> pd.DataFrame:
    cols = [
        "stock_code",
        "corp_name",
        "report_year",
        "report_quarter",
        "report_type",
        "report_date",
        "publish_date",
        *FINANCIAL_RAW_COLUMNS,
    ]
    fin = pd.read_csv(path, usecols=cols, low_memory=False)
    fin["security_id"] = fin["stock_code"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    fin["report_date"] = pd.to_datetime(fin["report_date"], errors="coerce")
    fin["source_publish_date"] = pd.to_datetime(fin["publish_date"], errors="coerce")
    fin["available_date"] = financial_available_date(fin)
    for col in FINANCIAL_RAW_COLUMNS:
        fin[col] = pd.to_numeric(fin[col], errors="coerce")

    fin["fin_log_revenue"] = np.log1p(fin["revenue"].clip(lower=0))
    fin["fin_log_operating_income_signed"] = signed_log1p(fin["operating_income"])
    fin["fin_log_net_income_signed"] = signed_log1p(fin["net_income"])
    fin["fin_log_total_assets"] = np.log1p(fin["total_assets"].clip(lower=0))
    fin["fin_leverage"] = fin["total_liabilities"] / fin["total_assets"].replace(0, np.nan)
    fin["fin_equity_ratio"] = fin["total_equity"] / fin["total_assets"].replace(0, np.nan)
    fin["fin_roa"] = fin["net_income"] / fin["total_assets"].replace(0, np.nan)
    fin["fin_operating_roa"] = fin["operating_income"] / fin["total_assets"].replace(0, np.nan)
    fin["fin_net_margin"] = fin["net_income"] / fin["revenue"].replace(0, np.nan)
    fin["fin_operating_margin"] = fin["operating_income"] / fin["revenue"].replace(0, np.nan)
    fin["fin_loss_flag"] = (fin["net_income"] < 0).astype(float)
    fin["fin_negative_equity_flag"] = (fin["total_equity"] < 0).astype(float)
    fin["fin_missing_record_flag"] = 0.0
    fin = fin.sort_values(["security_id", "available_date", "report_date"])
    return fin


def safe_days_since(anchor_date: pd.Timestamp, events: pd.DataFrame, code: str | None = None) -> float:
    subset = events if code is None else events[events["taxonomy_code"].eq(code)]
    dates = pd.to_datetime(subset["event_date"], errors="coerce").dropna()
    if dates.empty:
        return 9999.0
    return float((anchor_date - dates.max()).days)


def sequence_features(events: pd.DataFrame, anchor_date: pd.Timestamp) -> dict[str, float]:
    hist = events[events["event_date"].le(anchor_date)].copy()
    feats: dict[str, float] = {
        "seq_events_to_date": float(len(hist)),
        "seq_distinct_taxonomies_to_date": float(hist["taxonomy_code"].nunique()) if not hist.empty else 0.0,
        "seq_days_since_last_event": safe_days_since(anchor_date, hist),
    }
    for code in TAXONOMY_CODES:
        code_hist = hist[hist["taxonomy_code"].eq(code)]
        feats[f"seq_{code}_to_date"] = float(len(code_hist))
        feats[f"seq_days_since_last_{code}"] = safe_days_since(anchor_date, hist, code)

    for window in (90, 180, 365, 730):
        start = anchor_date - pd.Timedelta(days=window)
        w = hist[hist["event_date"].ge(start)]
        feats[f"seq_events_{window}d"] = float(len(w))
        feats[f"seq_distinct_taxonomies_{window}d"] = float(w["taxonomy_code"].nunique()) if not w.empty else 0.0
        for code in TAXONOMY_CODES:
            feats[f"seq_{code}_{window}d"] = float((w["taxonomy_code"] == code).sum())

    w180 = hist[hist["event_date"].ge(anchor_date - pd.Timedelta(days=180))]
    w365 = hist[hist["event_date"].ge(anchor_date - pd.Timedelta(days=365))]
    feats["seq_repeated_T05_365d_flag"] = float((w365["taxonomy_code"] == "T05").sum() >= 2)
    feats["seq_T05_T06_180d_flag"] = float({"T05", "T06"}.issubset(set(w180["taxonomy_code"])))
    feats["seq_prior_T03_flag"] = float((hist["taxonomy_code"] == "T03").any())
    return feats


def build_anchor_rows(events: pd.DataFrame, outcomes: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    non_outcome_events = events[~events["event_subtype"].isin(OUTCOME_SUBTYPES)].copy()
    first_outcomes = outcomes.sort_values("outcome_date").groupby("firm_id").first()
    rows: list[dict] = []
    firm_meta = (
        events.sort_values("event_date")
        .groupby("firm_id")
        .agg(
            firm_name=("corp_name", "last"),
            security_id=("security_id", "last"),
            first_event_date=("event_date", "min"),
            last_event_date=("event_date", "max"),
        )
        .reset_index()
    )

    events_by_firm = {firm_id: g.sort_values("event_date") for firm_id, g in non_outcome_events.groupby("firm_id")}
    for row in firm_meta.itertuples(index=False):
        firm_id = row.firm_id
        first_outcome = first_outcomes.loc[firm_id] if firm_id in first_outcomes.index else None
        firm_events = events_by_firm.get(firm_id, non_outcome_events.iloc[0:0])
        for year in range(start_year, end_year + 1):
            anchor_date = pd.Timestamp(year=year, month=12, day=31)
            if pd.notna(row.first_event_date) and anchor_date < row.first_event_date:
                continue
            if first_outcome is not None and pd.notna(first_outcome["outcome_date"]) and first_outcome["outcome_date"] <= anchor_date:
                continue

            future_outcome_date = pd.NaT
            future_outcome_label = None
            days_to_outcome = np.nan
            if first_outcome is not None and pd.notna(first_outcome["outcome_date"]):
                future_outcome_date = first_outcome["outcome_date"]
                future_outcome_label = first_outcome["outcome_label"]
                days_to_outcome = float((future_outcome_date - anchor_date).days)

            feats = sequence_features(firm_events, anchor_date)
            rows.append(
                {
                    "firm_id": firm_id,
                    "firm_name": row.firm_name,
                    "security_id": row.security_id,
                    "anchor_year": year,
                    "anchor_date": anchor_date,
                    "first_event_date": row.first_event_date,
                    "last_event_date": row.last_event_date,
                    "future_outcome_date": future_outcome_date,
                    "future_outcome_label": future_outcome_label,
                    "days_to_outcome": days_to_outcome,
                    "target_365d": bool(pd.notna(days_to_outcome) and 0 < days_to_outcome <= 365),
                    "target_730d": bool(pd.notna(days_to_outcome) and 0 < days_to_outcome <= 730),
                    **feats,
                }
            )
    return pd.DataFrame(rows)


def attach_financials(panel: pd.DataFrame, financials: pd.DataFrame) -> pd.DataFrame:
    selected_rows: list[dict] = []
    fin_by_security = {sec: g.sort_values("available_date").reset_index(drop=True) for sec, g in financials.groupby("security_id")}
    metadata_cols = [
        "report_year",
        "report_quarter",
        "report_type",
        "report_date",
        "source_publish_date",
        "available_date",
        *FINANCIAL_RAW_COLUMNS,
    ]
    empty = {f"fin_{col}": np.nan for col in metadata_cols}
    empty.update({col: np.nan for col in FINANCIAL_FEATURE_COLUMNS})
    for row in panel.itertuples(index=False):
        security_id = str(row.security_id).zfill(6)
        fin = fin_by_security.get(security_id)
        if fin is None or fin.empty:
            rec = empty.copy()
            rec["fin_missing_record_flag"] = 1.0
            selected_rows.append(rec)
            continue
        dates = fin["available_date"].to_numpy(dtype="datetime64[ns]")
        idx = np.searchsorted(dates, np.datetime64(row.anchor_date), side="right") - 1
        if idx < 0:
            rec = empty.copy()
            rec["fin_missing_record_flag"] = 1.0
            selected_rows.append(rec)
            continue
        selected = fin.iloc[int(idx)]
        rec = {f"fin_{col}": selected[col] for col in metadata_cols}
        rec.update({col: selected[col] for col in FINANCIAL_FEATURE_COLUMNS})
        selected_rows.append(rec)

    fin_panel = pd.DataFrame(selected_rows)
    out = pd.concat([panel.reset_index(drop=True), fin_panel.reset_index(drop=True)], axis=1)
    out["has_financial_record"] = ~out["fin_available_date"].isna()
    return out


def build_summary(panel: pd.DataFrame, outcomes: pd.DataFrame, config: Config) -> dict:
    max_outcome_date = pd.Timestamp(year=config.outcome_end_year, month=12, day=31)
    summary = {
        "created_from": {
            "events": str(config.events_path),
            "outcomes_raw": str(config.dart_raw_path),
            "financials": "external_phase2_pit_financials",
        },
        "method_notes": [
            "Financial available_date is approximated as report_date + 120 days for annual reports and + 60 days for interim reports.",
            "Firm-year anchors after a firm's first distress outcome are excluded.",
            "Outcome labels are observed through outcome_end_year so 2024 anchors can support 365-day targets.",
        ],
        "anchor_years": [config.start_year, config.end_year],
        "outcome_mode": config.outcome_mode,
        "outcome_observation_end": str(max_outcome_date.date()),
        "rows": int(len(panel)),
        "firms": int(panel["firm_id"].nunique()),
        "securities": int(panel["security_id"].nunique()),
        "outcome_rows": int(len(outcomes)),
        "outcome_firms": int(outcomes["firm_id"].nunique()),
        "financial_match_rate": float(panel["has_financial_record"].mean()) if len(panel) else None,
        "targets": {},
    }
    for horizon in (365, 730):
        usable = panel["anchor_date"] + pd.to_timedelta(horizon, unit="D") <= max_outcome_date
        target = panel[f"target_{horizon}d"]
        summary["targets"][f"{horizon}d"] = {
            "usable_rows": int(usable.sum()),
            "positive_rows": int((usable & target).sum()),
            "positive_rate": float((usable & target).sum() / usable.sum()) if usable.sum() else None,
        }
    return summary


def write_outputs(panel: pd.DataFrame, outcomes: pd.DataFrame, summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_dir / "dart_model_panel.parquet", index=False)
    panel.to_csv(out_dir / "dart_model_panel.csv", index=False, encoding="utf-8-sig")
    outcomes.to_parquet(out_dir / "dart_outcomes_for_model.parquet", index=False)
    outcomes.to_csv(out_dir / "dart_outcomes_for_model.csv", index=False, encoding="utf-8-sig")
    (out_dir / "dart_model_panel_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-path", default=str(Path("Paper 4") / "data" / "processed" / "events_dart.parquet"))
    parser.add_argument("--dart-raw-path", default=str(Path("Paper3") / "dart_apriori" / "data" / "raw" / "dart_major_reports_full.csv"))
    parser.add_argument(
        "--financial-path",
        default=str(
            Path.home()
            / "Desktop"
            / "Phase 2"
            / "dart_only_mvp"
            / "data"
            / "raw"
            / "data_v3"
            / "quarterly_financials_pit.csv"
        ),
    )
    parser.add_argument("--out-dir", default=str(Path("Paper 4") / "data" / "model"))
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--outcome-end-year", type=int, default=2025)
    parser.add_argument("--outcome-mode", choices=["all", "robust"], default="all")
    args = parser.parse_args()

    config = Config(
        events_path=Path(args.events_path),
        dart_raw_path=Path(args.dart_raw_path),
        financial_path=Path(args.financial_path),
        out_dir=Path(args.out_dir),
        start_year=args.start_year,
        end_year=args.end_year,
        outcome_end_year=args.outcome_end_year,
        outcome_mode=args.outcome_mode,
    )
    events = read_events(config.events_path)
    outcomes_all = read_outcomes_from_raw(config.dart_raw_path, config.start_year, config.outcome_end_year)
    outcomes, filter_summary = filter_outcomes(outcomes_all, config.outcome_mode)
    financials = read_financials(config.financial_path)
    panel = build_anchor_rows(events, outcomes, config.start_year, config.end_year)
    panel = attach_financials(panel, financials)
    summary = build_summary(panel, outcomes, config)
    summary["outcome_filter"] = filter_summary
    summary["unfiltered_outcome_rows"] = int(len(outcomes_all))
    summary["unfiltered_outcome_firms"] = int(outcomes_all["firm_id"].nunique())
    write_outputs(panel, outcomes, summary, config.out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
