from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path("Paper 4")
DEFAULT_RAW = Path("Paper3") / "dart_apriori" / "data" / "raw" / "dart_major_reports_full.csv"

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
    "rehabilitation": "BANKRUPTCY_OR_REHABILITATION",
    "dissolution": "DISSOLUTION",
    "default": "DEFAULT",
    "listing_or_trading_status": "DELISTING_OR_TRADING_SUSPENSION",
}


def output_paths(root: Path) -> dict[str, Path]:
    base = root / "data" / "raw" / "dart_gap"
    return {
        "base": base,
        "outcome_csv": base / "dart_gap_outcome_candidates.csv",
        "outcome_parquet": base / "dart_gap_outcome_candidates.parquet",
        "status_json": root / "results" / "dart_gap_collection_status.json",
        "status_md": root / "results" / "dart_gap_collection_status.md",
    }


def contains(text: pd.Series, needle: str) -> pd.Series:
    return text.astype(str).str.contains(needle, regex=False, na=False)


def clean_dart_title(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
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


def classify_outcome(report_nm: object) -> tuple[str, str, str] | None:
    title = "" if pd.isna(report_nm) else str(report_nm)
    if K["rehab"] in title:
        return "T01", "rehabilitation", OUTCOME_SUBTYPES["rehabilitation"]
    if K["dissolution"] in title:
        return "T01", "dissolution", OUTCOME_SUBTYPES["dissolution"]
    if K["default"] in title:
        return "T03", "default", OUTCOME_SUBTYPES["default"]
    if K["delisting"] in title or K["trading_suspension"] in title:
        return "T02", "listing_or_trading_status", OUTCOME_SUBTYPES["listing_or_trading_status"]
    return None


def load_candidates(raw_path: Path, start_year: int, end_year: int) -> tuple[pd.DataFrame, dict]:
    if not raw_path.exists():
        raise FileNotFoundError(f"DART raw major-report file not found: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    required = {"corp_code", "corp_name", "stock_code", "rcept_dt", "report_nm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DART raw file missing required columns: {sorted(missing)}")

    df = df.copy()
    df["source_row_index"] = df.index
    df["event_date"] = pd.to_datetime(df["rcept_dt"].astype(str), format="%Y%m%d", errors="coerce")
    in_window = df["event_date"].dt.year.between(start_year, end_year, inclusive="both")

    noncorr = pd.Series(True, index=df.index)
    for token in (K["correction"], K["attachment_added"], K["attachment_correction"], K["extension"], K["withdrawal"]):
        noncorr &= ~contains(df["report_nm"], token)

    window = df.loc[in_window & noncorr].copy()
    classified = window["report_nm"].map(classify_outcome)
    outcomes = window.loc[classified.notna()].copy()
    if not outcomes.empty:
        outcome_values = classified.loc[classified.notna()]
        outcomes["taxonomy_code"] = [value[0] for value in outcome_values]
        outcomes["event_subtype"] = [value[1] for value in outcome_values]
        outcomes["outcome_label"] = [value[2] for value in outcome_values]
        outcomes["clean_event_title"] = outcomes["report_nm"].map(clean_dart_title)
        outcomes["source"] = "DART"
        outcomes["firm_id"] = outcomes["corp_code"].astype(str).str.zfill(8)
        outcomes["security_id"] = outcomes["stock_code"].astype(str).str.zfill(6)
        keep = [
            "source",
            "firm_id",
            "corp_code",
            "corp_name",
            "security_id",
            "stock_code",
            "event_date",
            "outcome_label",
            "taxonomy_code",
            "event_subtype",
            "clean_event_title",
            "report_nm",
            "rcept_no",
            "source_row_index",
        ]
        outcomes = outcomes[keep].sort_values(["firm_id", "event_date", "taxonomy_code", "report_nm"])

    status = {
        "status": "resolved_from_existing_dart_major_reports",
        "raw_path": str(raw_path),
        "raw_rows": int(len(df)),
        "target_start_year": int(start_year),
        "target_end_year": int(end_year),
        "filtered_event_rows": int(len(window)),
        "outcome_candidate_rows": int(len(outcomes)),
        "outcome_candidate_firms": int(outcomes["firm_id"].nunique()) if not outcomes.empty else 0,
        "date_min": str(outcomes["event_date"].min().date()) if not outcomes.empty else None,
        "date_max": str(outcomes["event_date"].max().date()) if not outcomes.empty else None,
        "outcome_counts": {str(k): int(v) for k, v in outcomes["outcome_label"].value_counts().items()} if not outcomes.empty else {},
        "note": "The Sprint 0 audit found no explicit DART outcome column in the inspected files. This step resolves the gap by deriving distress outcomes from DART major-event report titles already collected for Paper 3; no new DART API call is made.",
    }
    return outcomes, status


def write_markdown(status: dict, path: Path) -> None:
    lines = ["# DART Gap Collection Status", ""]
    lines.append(f"Status: `{status['status']}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(status["note"])
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Raw rows: {status['raw_rows']:,}")
    lines.append(f"- Target window: {status['target_start_year']}-{status['target_end_year']}")
    lines.append(f"- Filtered non-correction event rows: {status['filtered_event_rows']:,}")
    lines.append(f"- Outcome candidate rows: {status['outcome_candidate_rows']:,}")
    lines.append(f"- Outcome candidate firms: {status['outcome_candidate_firms']:,}")
    lines.append(f"- Outcome date range: {status['date_min']} to {status['date_max']}")
    lines.append("")
    lines.append("## Outcome Counts")
    lines.append("")
    lines.append("| Outcome | Rows |")
    lines.append("|---|---:|")
    for label, count in status["outcome_counts"].items():
        lines.append(f"| {label} | {count:,} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()

    root = Path(args.root)
    paths = output_paths(root)
    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["status_json"].parent.mkdir(parents=True, exist_ok=True)

    outcomes, status = load_candidates(Path(args.raw_path), args.start_year, args.end_year)
    outcomes.to_csv(paths["outcome_csv"], index=False, encoding="utf-8-sig")
    outcomes.to_parquet(paths["outcome_parquet"], index=False)
    paths["status_json"].write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(status, paths["status_md"])
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
