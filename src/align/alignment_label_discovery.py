from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path("Paper 4")
DESKTOP = Path.home() / "Desktop"
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".next",
    "dist",
    "build",
    "site-packages",
}
NAME_HINTS = (
    "manual_alignment",
    "alignment",
    "annotation",
    "annotator",
    "label_sheet",
    "adjudication",
    "human_label",
)
EXTENSIONS = {".csv", ".xlsx", ".xlsm"}
PAPER4_REQUIRED_COLUMNS = {"candidate_rank", "dart_event", "sec_event"}
PAPER4_LABEL_COLUMNS = ["human_label", "annotator1_label", "annotator2_label", "adjudicated_label"]
OTHER_LABEL_COLUMNS = ["reviewer_1_label", "reviewer_2_label", "suggested_label", "adjudicated_label"]


def normalize_col(value: object) -> str:
    return str(value).strip().lower()


def nonblank_count(series: pd.Series) -> int:
    if series.empty:
        return 0
    return int(series.fillna("").astype(str).str.strip().ne("").sum())


def candidate_paths(index_path: Path, include_root: Path) -> list[Path]:
    paths: dict[str, Path] = {}
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            p = Path(line.strip())
            if is_candidate_name(p):
                paths[str(p).lower()] = p
    if include_root.exists():
        for p in include_root.rglob("*"):
            if p.is_file() and is_candidate_name(p):
                paths[str(p).lower()] = p
    for p in walk_desktop_candidates(DESKTOP):
        paths[str(p).lower()] = p
    return sorted(paths.values(), key=lambda p: str(p).lower())


def is_candidate_name(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix not in EXTENSIONS:
        return False
    s = str(path).lower()
    if "alignment_label_discovery" in s:
        return False
    return any(hint in s for hint in NAME_HINTS)


def walk_desktop_candidates(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            p = Path(dirpath) / filename
            if is_candidate_name(p):
                yield p


def read_table(path: Path, max_mb: float) -> tuple[pd.DataFrame | None, str, str]:
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
    except OSError as exc:
        return None, "unreadable", repr(exc)
    if size_mb > max_mb:
        return None, "skipped_large_file", f"size_mb={size_mb:.2f}"
    try:
        if path.suffix.lower() == ".csv":
            for encoding in ("utf-8-sig", "utf-8", "cp949"):
                try:
                    return pd.read_csv(path, low_memory=False, encoding=encoding), "read", ""
                except UnicodeDecodeError:
                    continue
            return pd.read_csv(path, low_memory=False, encoding_errors="ignore"), "read", ""
        return pd.read_excel(path, sheet_name=0), "read", ""
    except Exception as exc:
        return None, "unreadable", repr(exc)


def inspect_file(path: Path, max_mb: float) -> dict:
    size_mb = None
    try:
        size_mb = round(path.stat().st_size / (1024 * 1024), 4)
    except OSError:
        pass
    df, read_status, read_error = read_table(path, max_mb)
    base = {
        "path": str(path),
        "extension": path.suffix.lower(),
        "size_mb": size_mb,
        "read_status": read_status,
        "read_error": read_error,
        "rows": 0,
        "columns": 0,
        "paper4_compatible": False,
        "paper4_label_columns_present": "",
        "paper4_filled_label_cells": 0,
        "paper4_rows_with_any_label": 0,
        "other_label_columns_present": "",
        "other_filled_label_cells": 0,
        "scope": classify_scope(path),
        "action": "unreadable_or_skipped",
    }
    if df is None:
        return base

    columns = {normalize_col(col): col for col in df.columns}
    paper4_present = [col for col in PAPER4_LABEL_COLUMNS if col in columns]
    other_present = [col for col in OTHER_LABEL_COLUMNS if col in columns]
    paper4_compatible = PAPER4_REQUIRED_COLUMNS.issubset(set(columns)) and bool(paper4_present)
    paper4_filled = 0
    paper4_rows_any = 0
    if paper4_present:
        paper4_label_df = pd.DataFrame({col: df[columns[col]] for col in paper4_present})
        paper4_filled = sum(nonblank_count(paper4_label_df[col]) for col in paper4_label_df.columns)
        paper4_rows_any = int(paper4_label_df.fillna("").astype(str).apply(lambda row: any(v.strip() for v in row), axis=1).sum())
    other_filled = 0
    if other_present:
        other_label_df = pd.DataFrame({col: df[columns[col]] for col in other_present})
        other_filled = sum(nonblank_count(other_label_df[col]) for col in other_label_df.columns)

    action = "not_paper4_alignment"
    if paper4_compatible and paper4_filled > 0:
        action = "paper4_labels_available_for_sync_review"
    elif paper4_compatible:
        action = "paper4_template_no_labels"
    elif other_present and other_filled > 0:
        action = "unrelated_label_data_not_importable"

    return {
        **base,
        "read_status": read_status,
        "read_error": read_error,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "paper4_compatible": bool(paper4_compatible),
        "paper4_label_columns_present": ";".join(paper4_present),
        "paper4_filled_label_cells": int(paper4_filled),
        "paper4_rows_with_any_label": int(paper4_rows_any),
        "other_label_columns_present": ";".join(other_present),
        "other_filled_label_cells": int(other_filled),
        "action": action,
    }


def classify_scope(path: Path) -> str:
    s = str(path).lower().replace("/", "\\")
    if not path.is_absolute() and s.startswith("paper 4\\"):
        return "paper4_current"
    if "\\사업보고서_논문\\paper 4\\" in s or "\\paper 4\\" in s:
        return "paper4_current"
    if "\\사업보고서_논문\\" in s:
        return "workspace_other"
    return "external_local"


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "extension",
        "size_mb",
        "read_status",
        "read_error",
        "rows",
        "columns",
        "scope",
        "paper4_compatible",
        "paper4_label_columns_present",
        "paper4_filled_label_cells",
        "paper4_rows_with_any_label",
        "other_label_columns_present",
        "other_filled_label_cells",
        "action",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["path"] = display_path(row)
            writer.writerow(out)


def summarize(rows: list[dict], out_csv: Path) -> dict:
    compatible = [row for row in rows if row["paper4_compatible"]]
    available = [row for row in compatible if row["paper4_filled_label_cells"] > 0]
    unrelated = [row for row in rows if row["action"] == "unrelated_label_data_not_importable"]
    return {
        "status": "paper4_labels_found" if available else "no_filled_paper4_labels_found",
        "candidate_files_scanned": len(rows),
        "readable_files": sum(1 for row in rows if row["read_status"] == "read"),
        "paper4_compatible_files": len(compatible),
        "paper4_compatible_files_with_labels": len(available),
        "paper4_filled_label_cells": sum(int(row["paper4_filled_label_cells"]) for row in compatible),
        "paper4_rows_with_any_label": sum(int(row["paper4_rows_with_any_label"]) for row in compatible),
        "unrelated_label_files_with_filled_labels": len(unrelated),
        "discovery_csv": str(out_csv),
        "guardrail": "Discovery does not create labels. Only Paper 4-compatible files with filled human/adjudication labels can be reviewed for sync.",
    }


def write_markdown(summary: dict, rows: list[dict], path: Path) -> None:
    compatible = [row for row in rows if row["paper4_compatible"]]
    available = [row for row in compatible if row["paper4_filled_label_cells"] > 0]
    top_rows = sorted(rows, key=lambda row: (row["action"], -int(row["paper4_filled_label_cells"]), row["path"]))[:20]
    lines = [
        "# Alignment Label Discovery",
        "",
        "This artifact scans local Paper 4 and external candidate label files for already-filled RQ1 human labels. It does not create labels or import them automatically.",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Status | `{summary['status']}` |",
        f"| Candidate files scanned | {summary['candidate_files_scanned']} |",
        f"| Readable files | {summary['readable_files']} |",
        f"| Paper 4-compatible files | {summary['paper4_compatible_files']} |",
        f"| Paper 4-compatible files with filled labels | {summary['paper4_compatible_files_with_labels']} |",
        f"| Paper 4 filled label cells | {summary['paper4_filled_label_cells']} |",
        f"| Paper 4 rows with any label | {summary['paper4_rows_with_any_label']} |",
        f"| Unrelated filled label files | {summary['unrelated_label_files_with_filled_labels']} |",
        "",
    ]
    if available:
        lines.extend(["## Paper 4-Compatible Filled Label Files", "", "| Filled cells | Rows with labels | Path |", "|---:|---:|---|"])
        for row in available:
            lines.append(
                f"| {row['paper4_filled_label_cells']} | "
                f"{row['paper4_rows_with_any_label']} | `{display_path(row)}` |"
            )
    else:
        lines.extend(
            [
                "## Paper 4-Compatible Filled Label Files",
                "",
                "No Paper 4-compatible file with filled `human_label`, `annotator1_label`, `annotator2_label`, or `adjudicated_label` cells was found.",
            ]
        )

    lines.extend(["", "## Candidate File Sample", "", "| Action | Scope | Rows | Paper 4 labels | Path |", "|---|---|---:|---:|---|"])
    for row in top_rows:
        lines.append(
            f"| `{row['action']}` | `{row['scope']}` | {row['rows']} | "
            f"{row['paper4_filled_label_cells']} | `{display_path(row)}` |"
        )
    lines.extend(["", f"Guardrail: {summary['guardrail']}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def display_path(row: dict) -> str:
    path = Path(str(row.get("path", "")))
    scope = str(row.get("scope", "local"))
    action = str(row.get("action", "candidate"))
    if scope == "paper4_current":
        return str(path)
    digest = hashlib.sha1(str(path).encode("utf-8", errors="ignore")).hexdigest()[:10]
    suffix = path.suffix.lower() or ".file"
    return f"{scope}/{action}/candidate_{digest}{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", default=str(ROOT / "data" / "audit" / "desktop_data_files_rg.txt"))
    parser.add_argument("--include-root", default=str(ROOT))
    parser.add_argument("--max-mb", type=float, default=25.0)
    parser.add_argument("--out-csv", default=str(ROOT / "data" / "audit" / "alignment_label_discovery.csv"))
    parser.add_argument("--out-json", default=str(ROOT / "results" / "alignment_label_discovery.json"))
    parser.add_argument("--out-md", default=str(ROOT / "results" / "alignment_label_discovery.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = candidate_paths(Path(args.index_path), Path(args.include_root))
    rows = [inspect_file(path, args.max_mb) for path in paths]
    out_csv = Path(args.out_csv)
    write_csv(rows, out_csv)
    summary = summarize(rows, out_csv)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, rows, Path(args.out_md))
    print(json.dumps({k: summary[k] for k in ["status", "candidate_files_scanned", "paper4_compatible_files_with_labels"]}, indent=2))


if __name__ == "__main__":
    main()
