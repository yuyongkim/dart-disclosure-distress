from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

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
    "capital_reduction": "\uac10\uc790\uacb0\uc815",
    "paid_in_capital_increase": "\uc720\uc0c1\uc99d\uc790\uacb0\uc815",
    "convertible_bond": "\uc804\ud658\uc0ac\ucc44\uad8c\ubc1c\ud589\uacb0\uc815",
    "bw_bond": "\uc2e0\uc8fc\uc778\uc218\uad8c\ubd80\uc0ac\ucc44\uad8c\ubc1c\ud589\uacb0\uc815",
    "asset_transfer": "\uc790\uc0b0\uc591\uc218\ub3c4",
    "other_equity_acquire": "\ud0c0\ubc95\uc778\uc8fc\uc2dd\ubc0f\ucd9c\uc790\uc99d\uad8c\uc591\uc218\uacb0\uc815",
    "other_equity_dispose": "\ud0c0\ubc95\uc778\uc8fc\uc2dd\ubc0f\ucd9c\uc790\uc99d\uad8c\uc591\ub3c4\uacb0\uc815",
    "merger": "\ud569\ubcd1\uacb0\uc815",
    "split": "\ubd84\ud560",
    "lawsuit": "\uc18c\uc1a1",
}


@dataclass(frozen=True)
class Config:
    paper3_root: Path
    out_dir: Path
    start_year: int
    end_year: int


def contains(text: pd.Series, needle: str) -> pd.Series:
    return text.astype(str).str.contains(needle, regex=False, na=False)


def clean_dart_title(value: str) -> str:
    text = str(value)
    for token in (
        "[\uae30\uc7ac\uc815\uc815]",
        "[\ucca8\ubd80\uc815\uc815]",
        "[\ucca8\ubd80\ucd94\uac00]",
        "[\uc5f0\uc7a5\uacb0\uc815]",
    ):
        text = text.replace(token, "")
    text = text.replace("\uc8fc\uc694\uc0ac\ud56d\ubcf4\uace0\uc11c(", "")
    text = text.replace("\uc8fc\uc694\uc0ac\ud56d\ubcf4\uace0\uc11c", "")
    text = text.rstrip(")")
    return text.strip()


def dart_taxonomy(report_nm: str) -> tuple[str, str, str]:
    title = str(report_nm)
    if K["rehab"] in title:
        return "T01", "insolvency_or_rehabilitation", "rehabilitation"
    if K["dissolution"] in title:
        return "T01", "insolvency_or_rehabilitation", "dissolution"
    if K["delisting"] in title or K["trading_suspension"] in title:
        return "T02", "delisting_or_trading_suspension", "listing_or_trading_status"
    if K["default"] in title:
        return "T03", "debt_default_or_acceleration", "default"
    if K["capital_reduction"] in title:
        return "T05", "capital_structure_stress", "capital_reduction"
    if K["paid_in_capital_increase"] in title:
        return "T05", "capital_structure_stress", "paid_in_capital_increase"
    if K["convertible_bond"] in title:
        return "T05", "capital_structure_stress", "convertible_bond"
    if K["bw_bond"] in title:
        return "T05", "capital_structure_stress", "bond_with_warrant"
    if K["asset_transfer"] in title or K["other_equity_acquire"] in title or K["other_equity_dispose"] in title:
        return "T05", "capital_structure_stress", "asset_or_equity_transfer"
    if K["merger"] in title or K["split"] in title:
        return "T06", "governance_or_control_change", "merger_or_split"
    if K["lawsuit"] in title:
        return "T06", "governance_or_control_change", "lawsuit"
    return "T99", "other_event", "other"


def sec_taxonomy(item: str) -> tuple[str, str, str, bool]:
    item = str(item).strip()
    mapping = {
        "1.03": ("T01", "insolvency_or_rehabilitation", "bankruptcy_or_receivership", True),
        "3.01": ("T02", "delisting_or_trading_suspension", "delisting_notice", True),
        "2.04": ("T03", "debt_default_or_acceleration", "debt_acceleration_trigger", True),
        "2.03": ("T03", "debt_default_or_acceleration", "financial_obligation", False),
        "3.03": ("T05", "capital_structure_stress", "security_holder_rights_modification", False),
        "3.02": ("T05", "capital_structure_stress", "unregistered_securities_sale", False),
        "5.03": ("T06", "governance_or_control_change", "charter_or_bylaw_change", False),
        "5.02": ("T06", "governance_or_control_change", "director_or_officer_change", False),
        "2.01": ("T05", "capital_structure_stress", "asset_acquisition_or_disposition", False),
        "1.01": ("T05", "capital_structure_stress", "material_definitive_agreement", False),
    }
    return mapping.get(item, ("T99", "other_event", "other", False))


def target_period_mask(dates: pd.Series, start_year: int, end_year: int) -> pd.Series:
    return dates.dt.year.between(start_year, end_year, inclusive="both")


def build_dart(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = config.paper3_root / "data" / "raw" / "dart_major_reports_full.csv"
    df = pd.read_csv(src, low_memory=False)
    df["event_date"] = pd.to_datetime(df["rcept_dt"].astype(str), format="%Y%m%d", errors="coerce")

    noncorr = pd.Series(True, index=df.index)
    for token in (K["correction"], K["attachment_added"], K["attachment_correction"], K["extension"], K["withdrawal"]):
        noncorr &= ~contains(df["report_nm"], token)

    df = df.loc[noncorr & target_period_mask(df["event_date"], config.start_year, config.end_year)].copy()
    tax = df["report_nm"].map(dart_taxonomy)
    df["taxonomy_code"] = [x[0] for x in tax]
    df["taxonomy_label"] = [x[1] for x in tax]
    df["event_subtype"] = [x[2] for x in tax]
    df["clean_event_title"] = df["report_nm"].map(clean_dart_title)
    df["source"] = "DART"
    df["firm_id"] = df["corp_code"].astype(str).str.zfill(8)
    df["security_id"] = df["stock_code"].astype(str).str.zfill(6)

    events = df[
        [
            "source",
            "firm_id",
            "corp_code",
            "corp_name",
            "security_id",
            "stock_code",
            "event_date",
            "taxonomy_code",
            "taxonomy_label",
            "event_subtype",
            "clean_event_title",
            "report_nm",
            "rcept_no",
        ]
    ].sort_values(["firm_id", "event_date", "taxonomy_code"])

    outcome_subtypes = {"rehabilitation", "dissolution", "default", "listing_or_trading_status"}
    outcomes = events[events["event_subtype"].isin(outcome_subtypes)].copy()
    outcomes["outcome_label"] = outcomes["event_subtype"].map(
        {
            "rehabilitation": "BANKRUPTCY_OR_REHABILITATION",
            "dissolution": "DISSOLUTION",
            "default": "DEFAULT",
            "listing_or_trading_status": "DELISTING_OR_TRADING_SUSPENSION",
        }
    )
    outcomes = outcomes.rename(columns={"event_date": "outcome_date"})
    outcomes = outcomes[
        [
            "source",
            "firm_id",
            "corp_code",
            "corp_name",
            "security_id",
            "stock_code",
            "outcome_date",
            "outcome_label",
            "taxonomy_code",
            "event_subtype",
            "clean_event_title",
            "report_nm",
            "rcept_no",
        ]
    ].sort_values(["firm_id", "outcome_date"])
    return events, outcomes


def build_sec(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = config.paper3_root / "data" / "raw" / "edgar_8k_items.csv"
    df = pd.read_csv(src, low_memory=False)
    df["event_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    df = df.loc[target_period_mask(df["event_date"], config.start_year, config.end_year)].copy()
    df["items"] = df["items"].fillna("").astype(str)

    rows: list[dict] = []
    for row in df.itertuples(index=False):
        items = [x.strip() for x in str(row.items).split(",") if x.strip()]
        if not items:
            items = ["NO_ITEM"]
        for item in items:
            code, label, subtype, is_outcome = sec_taxonomy(item)
            rows.append(
                {
                    "source": "SEC",
                    "firm_id": str(row.cik).zfill(10),
                    "cik": row.cik,
                    "company_name": row.company_name,
                    "sic": row.sic,
                    "event_date": row.event_date,
                    "taxonomy_code": code,
                    "taxonomy_label": label,
                    "event_subtype": subtype,
                    "item": item,
                    "is_outcome_item": is_outcome,
                }
            )

    events = pd.DataFrame(rows).sort_values(["firm_id", "event_date", "item"])
    outcomes = events[events["is_outcome_item"]].copy()
    outcomes["outcome_label"] = outcomes["event_subtype"].map(
        {
            "bankruptcy_or_receivership": "BANKRUPTCY_OR_RECEIVERSHIP",
            "delisting_notice": "DELISTING_NOTICE",
            "debt_acceleration_trigger": "DEFAULT_OR_ACCELERATION",
        }
    )
    outcomes = outcomes.rename(columns={"event_date": "outcome_date"})
    outcomes = outcomes[
        [
            "source",
            "firm_id",
            "cik",
            "company_name",
            "sic",
            "outcome_date",
            "outcome_label",
            "taxonomy_code",
            "event_subtype",
            "item",
        ]
    ].sort_values(["firm_id", "outcome_date"])
    return events, outcomes


def write_frame(df: pd.DataFrame, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path_base.with_suffix(".parquet"), index=False)
    df.to_csv(path_base.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def summarize(events_dart: pd.DataFrame, outcomes_dart: pd.DataFrame, events_sec: pd.DataFrame, outcomes_sec: pd.DataFrame) -> dict:
    def event_summary(df: pd.DataFrame, firm_col: str = "firm_id") -> dict:
        return {
            "rows": int(len(df)),
            "firms": int(df[firm_col].nunique()) if not df.empty else 0,
            "date_min": str(df["event_date"].min().date()) if not df.empty else None,
            "date_max": str(df["event_date"].max().date()) if not df.empty else None,
            "taxonomy_counts": {str(k): int(v) for k, v in df["taxonomy_code"].value_counts().items()},
            "subtype_counts": {str(k): int(v) for k, v in df["event_subtype"].value_counts().head(25).items()},
        }

    def outcome_summary(df: pd.DataFrame) -> dict:
        return {
            "rows": int(len(df)),
            "firms": int(df["firm_id"].nunique()) if not df.empty else 0,
            "date_min": str(df["outcome_date"].min().date()) if not df.empty else None,
            "date_max": str(df["outcome_date"].max().date()) if not df.empty else None,
            "outcome_counts": {str(k): int(v) for k, v in df["outcome_label"].value_counts().items()},
        }

    return {
        "dart_events": event_summary(events_dart),
        "dart_outcomes": outcome_summary(outcomes_dart),
        "sec_events": event_summary(events_sec),
        "sec_outcomes": outcome_summary(outcomes_sec),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper3-root", default=str(Path("Paper3") / "dart_apriori"))
    parser.add_argument("--out-dir", default=str(Path("Paper 4") / "data" / "processed"))
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()

    config = Config(
        paper3_root=Path(args.paper3_root),
        out_dir=Path(args.out_dir),
        start_year=args.start_year,
        end_year=args.end_year,
    )
    events_dart, outcomes_dart = build_dart(config)
    events_sec, outcomes_sec = build_sec(config)

    write_frame(events_dart, config.out_dir / "events_dart")
    write_frame(outcomes_dart, config.out_dir / "outcomes_dart")
    write_frame(events_sec, config.out_dir / "events_sec")
    write_frame(outcomes_sec, config.out_dir / "outcomes_sec")

    summary = summarize(events_dart, outcomes_dart, events_sec, outcomes_sec)
    (config.out_dir / "phase1_event_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
