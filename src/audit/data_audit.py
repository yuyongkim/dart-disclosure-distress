from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DATA_EXTENSIONS = {".csv", ".parquet", ".json", ".pkl"}

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".next",
    "playwright-report",
    "test-results",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
}

DATE_HINTS = (
    "date",
    "dt",
    "rcept",
    "filed",
    "accepted",
    "report",
    "trd",
    "time",
    "year",
)


@dataclass
class DataFile:
    path: str
    relpath: str
    extension: str
    size_mb: float
    mtime: str
    category: str
    priority: int


def classify(path: Path) -> tuple[str, int]:
    s = str(path).lower()
    name = path.name.lower()
    priority = 0
    category = "other"

    if any(k in s for k in ("dart", "opendart", "전자공시", "major_report", "major_reports")):
        category = "dart"
        priority += 80
    if any(k in s for k in ("sec", "edgar", "8k", "8-k", "form_8", "cik")):
        category = "sec"
        priority += 80
    if any(k in s for k in ("stock", "price", "ohlc", "ohlcv", "kospi", "kosdaq", "krx", "quote", "daily")):
        if category == "other":
            category = "stock"
        priority += 60
    if any(k in s for k in ("outcome", "distress", "default", "bankrupt", "delist", "suspension", "rehabilitation", "going_concern")):
        if category == "other":
            category = "outcome"
        priority += 70
    if any(k in s for k in ("financial", "companyfacts", "altman", "ohlson", "credit", "rating")):
        if category == "other":
            category = "financial"
        priority += 45

    if "paper3" in s or "paper 4" in s or "paper4" in s:
        priority += 30
    if any(k in name for k in ("raw", "processed", "rule", "transaction", "event", "item")):
        priority += 15
    if path.suffix.lower() in {".parquet", ".csv"}:
        priority += 10
    return category, priority


def iter_data_files(root: Path) -> list[DataFile]:
    files: list[DataFile] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIR_NAMES and not d.endswith(".egg-info")
        ]
        current = Path(dirpath)
        for filename in filenames:
            p = current / filename
            ext = p.suffix.lower()
            if ext not in DATA_EXTENSIONS:
                continue
            try:
                stat = p.stat()
            except OSError:
                continue
            category, priority = classify(p)
            try:
                relpath = str(p.relative_to(root))
            except ValueError:
                relpath = str(p)
            files.append(
                DataFile(
                    path=str(p),
                    relpath=relpath,
                    extension=ext,
                    size_mb=round(stat.st_size / 1_000_000, 4),
                    mtime=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    category=category,
                    priority=priority,
                )
            )
    files.sort(key=lambda f: (f.priority, f.size_mb), reverse=True)
    return files


def load_data_files_from_list(root: Path, file_list: Path) -> list[DataFile]:
    files: list[DataFile] = []
    for line in file_list.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        p = Path(line.strip())
        if p.suffix.lower() not in DATA_EXTENSIONS:
            continue
        try:
            stat = p.stat()
        except OSError:
            continue
        category, priority = classify(p)
        try:
            relpath = str(p.relative_to(root))
        except ValueError:
            relpath = str(p)
        files.append(
            DataFile(
                path=str(p),
                relpath=relpath,
                extension=p.suffix.lower(),
                size_mb=round(stat.st_size / 1_000_000, 4),
                mtime=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                category=category,
                priority=priority,
            )
        )
    files.sort(key=lambda f: (f.priority, f.size_mb), reverse=True)
    return files


def date_like_columns(columns: list[str]) -> list[str]:
    result = []
    for c in columns:
        cl = c.lower()
        if any(h in cl for h in DATE_HINTS):
            result.append(c)
    return result


def as_date_range(series: pd.Series) -> tuple[str | None, str | None]:
    values = series.dropna()
    if values.empty:
        return None, None

    if pd.api.types.is_numeric_dtype(values):
        raw = values.astype(str)
        if raw.str.fullmatch(r"\d{8}").mean() > 0.5:
            parsed = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
        elif raw.str.fullmatch(r"\d{4}").mean() > 0.5:
            parsed = pd.to_datetime(raw, format="%Y", errors="coerce")
        else:
            parsed = pd.to_datetime(values, errors="coerce")
    else:
        raw = values.astype(str).str.strip()
        parsed = pd.to_datetime(raw, errors="coerce")
        if parsed.notna().mean() < 0.2 and raw.str.fullmatch(r"\d{8}").mean() > 0.5:
            parsed = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")

    parsed = parsed.dropna()
    if parsed.empty:
        return str(values.min()), str(values.max())
    return str(parsed.min().date()), str(parsed.max().date())


def inspect_dataframe(df: pd.DataFrame, sampled: bool) -> dict[str, Any]:
    columns = [str(c) for c in df.columns]
    date_ranges: dict[str, tuple[str | None, str | None]] = {}
    for c in date_like_columns(columns)[:8]:
        try:
            date_ranges[c] = as_date_range(df[c])
        except Exception as exc:  # noqa: BLE001 - audit should not fail on one column.
            date_ranges[c] = (f"ERROR: {exc}", None)

    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .round(4)
        .to_dict()
        if not df.empty
        else {}
    )
    return {
        "rows": int(len(df)),
        "sampled": sampled,
        "columns": columns,
        "date_ranges": date_ranges,
        "top_missing_ratio": {str(k): float(v) for k, v in missing.items()},
    }


def inspect_csv(path: Path, max_rows: int) -> dict[str, Any]:
    df = pd.read_csv(path, nrows=max_rows, low_memory=False)
    info = inspect_dataframe(df, sampled=True)
    try:
        with path.open("rb") as handle:
            line_count = sum(1 for _ in handle)
        info["estimated_total_rows"] = max(line_count - 1, 0)
    except Exception as exc:  # noqa: BLE001
        info["estimated_total_rows_error"] = str(exc)
    return info


def inspect_parquet(path: Path, max_rows: int) -> dict[str, Any]:
    df = pd.read_parquet(path)
    sampled = len(df) > max_rows
    if sampled:
        df = df.head(max_rows)
    info = inspect_dataframe(df, sampled=sampled)
    if sampled:
        info["estimated_total_rows"] = "read_head_only"
    return info


def inspect_json(path: Path, max_json_bytes: int) -> dict[str, Any]:
    if path.stat().st_size > max_json_bytes:
        return {"sampled": True, "json_skipped": f"larger than {max_json_bytes} bytes"}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        obj = json.load(handle)
    info: dict[str, Any] = {"sampled": False, "json_type": type(obj).__name__}
    if isinstance(obj, dict):
        info["top_level_keys"] = list(obj.keys())[:30]
        candidates = [
            v
            for v in obj.values()
            if isinstance(v, list) and v and isinstance(v[0], dict)
        ]
        if candidates:
            df = pd.DataFrame(candidates[0][:5000])
            info.update(inspect_dataframe(df, sampled=len(candidates[0]) > 5000))
            info["estimated_total_rows"] = len(candidates[0])
    elif isinstance(obj, list):
        info["estimated_total_rows"] = len(obj)
        if obj and isinstance(obj[0], dict):
            df = pd.DataFrame(obj[:5000])
            info.update(inspect_dataframe(df, sampled=len(obj) > 5000))
    return info


def inspect_pickle(path: Path, max_pickle_mb: float) -> dict[str, Any]:
    if path.stat().st_size / 1_000_000 > max_pickle_mb:
        return {"sampled": True, "pickle_skipped": f"larger than {max_pickle_mb} MB"}
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    info: dict[str, Any] = {"sampled": False, "pickle_type": type(obj).__name__}
    if isinstance(obj, pd.DataFrame):
        info.update(inspect_dataframe(obj.head(5000), sampled=len(obj) > 5000))
        info["estimated_total_rows"] = len(obj)
    elif isinstance(obj, list):
        info["estimated_total_rows"] = len(obj)
        if obj and isinstance(obj[0], dict):
            info.update(inspect_dataframe(pd.DataFrame(obj[:5000]), sampled=len(obj) > 5000))
        elif obj and isinstance(obj[0], (list, tuple, set)):
            info["example_item_length"] = len(obj[0])
            info["example_item"] = list(obj[0])[:20]
    elif isinstance(obj, dict):
        info["top_level_keys"] = list(obj.keys())[:30]
    return info


def inspect_file(data_file: DataFile, max_rows: int, max_json_bytes: int, max_pickle_mb: float) -> dict[str, Any]:
    path = Path(data_file.path)
    result = asdict(data_file)
    try:
        if path.suffix.lower() == ".csv":
            result.update(inspect_csv(path, max_rows))
        elif path.suffix.lower() == ".parquet":
            result.update(inspect_parquet(path, max_rows))
        elif path.suffix.lower() == ".json":
            result.update(inspect_json(path, max_json_bytes))
        elif path.suffix.lower() == ".pkl":
            result.update(inspect_pickle(path, max_pickle_mb))
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    return result


def choose_files_to_inspect(files: list[DataFile], max_inspect: int) -> list[DataFile]:
    selected: list[DataFile] = []
    seen: set[str] = set()

    for category in ("dart", "sec", "stock", "outcome", "financial"):
        category_files = [f for f in files if f.category == category and f.priority > 0]
        for f in category_files[: max(25, max_inspect // 8)]:
            if f.path not in seen:
                selected.append(f)
                seen.add(f.path)

    for f in files:
        if len(selected) >= max_inspect:
            break
        if f.path not in seen and f.priority >= 45:
            selected.append(f)
            seen.add(f.path)
    return selected


def desktop_root_name(path: str) -> str:
    parts = Path(path).parts
    lowered = [p.lower() for p in parts]
    if "desktop" in lowered:
        idx = lowered.index("desktop")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return parts[-2] if len(parts) >= 2 else path


def public_file_id(path: str, category: str = "file") -> str:
    suffix = Path(path).suffix.lower() or ".file"
    digest = hashlib.sha1(path.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"local_inventory/{category}/source_file_{digest}{suffix}"


def sanitize_inspection(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    public_id = public_file_id(str(item.get("path", item.get("relpath", ""))), str(item.get("category", "file")))
    out["path"] = public_id
    out["relpath"] = public_id
    return out


def summarize_inventory(files: list[DataFile]) -> dict[str, Any]:
    by_category = Counter(f.category for f in files)
    by_extension = Counter(f.extension for f in files)
    by_root = Counter(desktop_root_name(f.path) for f in files)
    total_mb = sum(f.size_mb for f in files)
    top_source_groups = {
        f"source_group_{idx:02d}": count
        for idx, (_root, count) in enumerate(by_root.most_common(25), start=1)
    }
    return {
        "total_files": len(files),
        "total_size_mb": round(total_mb, 2),
        "by_category": dict(by_category),
        "by_extension": dict(by_extension),
        "top_source_groups": top_source_groups,
    }


def coverage_from_inspections(inspections: list[dict[str, Any]]) -> dict[str, Any]:
    coverage: dict[str, Any] = {}

    def items(category: str) -> list[dict[str, Any]]:
        return [x for x in inspections if x.get("category") == category and not x.get("error")]

    for category in ("dart", "sec", "stock", "outcome", "financial"):
        category_items = items(category)
        date_ranges = []
        columns = Counter()
        for item in category_items:
            columns.update(item.get("columns", []))
            for col, rng in item.get("date_ranges", {}).items():
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    date_ranges.append((col, rng[0], rng[1], item["relpath"]))
        coverage[category] = {
            "inspected_files": len(category_items),
            "top_columns": columns.most_common(25),
            "date_ranges": date_ranges[:25],
        }
    return coverage


def has_column_like(inspections: list[dict[str, Any]], category: str, patterns: list[str]) -> bool:
    regexes = [re.compile(p, re.I) for p in patterns]
    for item in inspections:
        if item.get("category") != category:
            continue
        for col in item.get("columns", []):
            if any(r.search(col) for r in regexes):
                return True
    return False


def date_min_max(inspections: list[dict[str, Any]], category: str) -> tuple[str | None, str | None]:
    mins: list[str] = []
    maxs: list[str] = []
    for item in inspections:
        if item.get("category") != category:
            continue
        for rng in item.get("date_ranges", {}).values():
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                lo, hi = rng
                if lo and re.match(r"\d{4}-\d{2}-\d{2}", str(lo)):
                    mins.append(str(lo))
                if hi and re.match(r"\d{4}-\d{2}-\d{2}", str(hi)):
                    maxs.append(str(hi))
    return (min(mins) if mins else None, max(maxs) if maxs else None)


def checklist(inspections: list[dict[str, Any]]) -> dict[str, list[tuple[str, str]]]:
    dart_min, dart_max = date_min_max(inspections, "dart")
    sec_min, sec_max = date_min_max(inspections, "sec")
    stock_min, stock_max = date_min_max(inspections, "stock")
    return {
        "DART": [
            ("Period coverage", f"{dart_min or 'UNKNOWN'} to {dart_max or 'UNKNOWN'}; target 2015-2024"),
            ("Report type columns", "OK" if has_column_like(inspections, "dart", [r"report", r"pblntf", r"rcept", r"rm"]) else "MISSING"),
            ("Outcome-like labels", "OK" if has_column_like(inspections, "dart", [r"report_nm", r"category", r"outcome", r"rehabilitation", r"default"]) else "MISSING"),
            ("Stock code column", "OK" if has_column_like(inspections, "dart", [r"stock_code", r"ticker"]) else "MISSING"),
        ],
        "SEC": [
            ("Period coverage", f"{sec_min or 'UNKNOWN'} to {sec_max or 'UNKNOWN'}; target 2015-2024"),
            ("8-K item parsing", "OK" if has_column_like(inspections, "sec", [r"item", r"items"]) else "MISSING"),
            ("Outcome item columns", "OK" if has_column_like(inspections, "sec", [r"item", r"outcome", r"bankrupt", r"delist"]) else "MISSING"),
        ],
        "Stock/price": [
            ("Period coverage", f"{stock_min or 'UNKNOWN'} to {stock_max or 'UNKNOWN'}"),
            ("Korean price files", "OK" if any("kos" in x.get("path", "").lower() or "krx" in x.get("path", "").lower() for x in inspections if x.get("category") == "stock") else "UNKNOWN"),
            ("U.S. price files", "OK" if any("us" in x.get("path", "").lower() for x in inspections if x.get("category") == "stock") else "UNKNOWN"),
        ],
    }


def write_inventory_csv(files: list[DataFile], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(files[0]).keys()) if files else [])
        if files:
            writer.writeheader()
            for f in files:
                writer.writerow(asdict(f))


def write_desktop_root_counts_csv(files: list[DataFile], path: Path) -> None:
    categories = ("dart", "sec", "stock", "outcome", "financial", "other")
    rows: dict[str, dict[str, Any]] = {}
    root_counts = Counter(desktop_root_name(data_file.path) for data_file in files)
    root_aliases = {
        root: f"source_group_{idx:02d}"
        for idx, (root, _count) in enumerate(root_counts.most_common(), start=1)
    }
    for data_file in files:
        root = root_aliases[desktop_root_name(data_file.path)]
        row = rows.setdefault(
            root,
            {
                "desktop_root": root,
                "total_files": 0,
                "total_size_mb": 0.0,
                **{category: 0 for category in categories},
            },
        )
        row["total_files"] += 1
        row["total_size_mb"] += data_file.size_mb
        row[data_file.category] = row.get(data_file.category, 0) + 1

    ordered = sorted(rows.values(), key=lambda r: (r["total_files"], r["total_size_mb"]), reverse=True)
    fieldnames = ["desktop_root", "total_files", "total_size_mb", *categories]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered:
            row = dict(row)
            row["total_size_mb"] = round(float(row["total_size_mb"]), 4)
            writer.writerow(row)


def write_report(
    report_path: Path,
    root: Path,
    summary: dict[str, Any],
    inspections: list[dict[str, Any]],
    coverage: dict[str, Any],
    checks: dict[str, list[tuple[str, str]]],
) -> None:
    lines: list[str] = []
    lines.append("# Paper 4 Data Audit Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("Scan scope: `local source inventory`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Data files found: {summary['total_files']:,}")
    lines.append(f"- Total scanned size: {summary['total_size_mb']:,} MB")
    lines.append(f"- Files inspected in detail: {len(inspections):,}")
    lines.append("")
    lines.append("### By Category")
    lines.append("")
    lines.append("| Category | Files |")
    lines.append("|---|---:|")
    for k, v in sorted(summary["by_category"].items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"| {k} | {v:,} |")
    lines.append("")
    lines.append("### Top Source Roots")
    lines.append("")
    lines.append("| Root | Data files |")
    lines.append("|---|---:|")
    for k, v in summary["top_source_groups"].items():
        lines.append(f"| {k} | {v:,} |")

    lines.append("")
    lines.append("## Paper 4 Data Readiness Checklist")
    for section, rows in checks.items():
        lines.append("")
        lines.append(f"### {section}")
        lines.append("")
        lines.append("| Requirement | Audit result |")
        lines.append("|---|---|")
        for name, result in rows:
            tag = "MISSING" if "MISSING" in result else result
            lines.append(f"| {name} | {tag} |")

    lines.append("")
    lines.append("## High-Value Reusable Files")
    lines.append("")
    lines.append("| Category | File | Size MB | Rows/Sample | Date ranges | Notes |")
    lines.append("|---|---|---:|---:|---|---|")
    for item in inspections[:80]:
        rows = item.get("estimated_total_rows", item.get("rows", ""))
        dr = item.get("date_ranges", {})
        dr_text = "; ".join(f"{k}: {v[0]} to {v[1]}" for k, v in list(dr.items())[:3]) if isinstance(dr, dict) else ""
        notes = item.get("error") or item.get("json_skipped") or item.get("pickle_skipped") or ""
        lines.append(
            f"| {item.get('category')} | `{item.get('relpath')}` | {item.get('size_mb')} | {rows} | {dr_text} | {notes} |"
        )

    lines.append("")
    lines.append("## Coverage Detail")
    for category, info in coverage.items():
        lines.append("")
        lines.append(f"### {category}")
        lines.append("")
        lines.append(f"- Inspected files: {info['inspected_files']}")
        if info["top_columns"]:
            lines.append("- Common columns: " + ", ".join(f"`{c}` ({n})" for c, n in info["top_columns"][:15]))
        if info["date_ranges"]:
            lines.append("")
            lines.append("| Date column | Min | Max | File |")
            lines.append("|---|---|---|---|")
            for col, lo, hi, relpath in info["date_ranges"][:10]:
                lines.append(f"| `{col}` | {lo} | {hi} | `{relpath}` |")

    lines.append("")
    lines.append("## Immediate Interpretation")
    lines.append("")
    lines.append("- Paper 3 DART and SEC files are the first reuse target for Paper 4 Phase 1.")
    lines.append("- If DART/SEC outcome labels are present only as event rows, Paper 4 should build firm-level outcome labels from those rows rather than recollecting full data.")
    lines.append("- EDINET is not required for the first executable version unless local EDINET files appear in the inventory.")
    lines.append("- Price data should be used as validation/supporting evidence, not as the dependent variable.")
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path.home() / "Desktop"))
    parser.add_argument("--out-dir", default=str(Path("Paper 4") / "data" / "audit"))
    parser.add_argument("--report", default=str(Path("Paper 4") / "audit_report.md"))
    parser.add_argument("--file-list", default=None, help="Optional newline-delimited file list from rg --files.")
    parser.add_argument("--max-inspect", type=int, default=300)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--max-json-bytes", type=int, default=15_000_000)
    parser.add_argument("--max-pickle-mb", type=float, default=50.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = load_data_files_from_list(root, Path(args.file_list)) if args.file_list else iter_data_files(root)
    write_inventory_csv(files, out_dir / "data_inventory.csv")
    write_desktop_root_counts_csv(files, out_dir / "desktop_root_data_counts.csv")

    selected = choose_files_to_inspect(files, args.max_inspect)
    raw_inspections = [
        inspect_file(f, args.max_rows, args.max_json_bytes, args.max_pickle_mb)
        for f in selected
    ]

    summary = summarize_inventory(files)
    inspections = [sanitize_inspection(item) for item in raw_inspections]
    coverage = coverage_from_inspections(inspections)
    checks = checklist(raw_inspections)

    (out_dir / "inspected_files.json").write_text(
        json.dumps(inspections, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "coverage": coverage, "checklist": checks}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    write_report(Path(args.report), root, summary, inspections, coverage, checks)

    print(json.dumps({"files": len(files), "inspected": len(inspections), "report": args.report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
