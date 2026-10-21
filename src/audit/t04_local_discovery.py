from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("Paper 4")
WORKSPACE = Path(".")
DESKTOP_ROOT = Path.home() / "Desktop"
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

KEYWORDS = [
    "going concern",
    "audit opinion",
    "auditor opinion",
    "modified opinion",
    "qualified opinion",
    "adverse opinion",
    "disclaimer of opinion",
    "emphasis of matter",
    "material uncertainty",
    "gc opinion",
    "\uac10\uc0ac\uc758\uacac",  # audit opinion
    "\uacc4\uc18d\uae30\uc5c5",  # going concern
    "\ube44\uc801\uc815",  # non-standard/inappropriate
    "\uc758\uacac\uac70\uc808",  # disclaimer of opinion
    "\ud55c\uc815\uc758\uacac",  # qualified opinion
    "\ubd80\uc801\uc815",  # adverse opinion
    "\uac10\uc0ac\ubcf4\uace0\uc11c",  # audit report
    "\uac80\ud1a0\uc758\uacac",  # review opinion
]

GENERATED_MARKERS = [
    "Paper 4\\data\\audit",
    "Paper 4/results",
    "Paper 4\\results",
    "Paper 4\\src",
    "Paper 4\\manuscript",
    "Paper 4\\paper4_completion_audit",
    "Paper 4\\data_map",
    "Paper 4\\REPRODUCIBILITY",
    "Paper 4\\VERSIONS",
]

DATA_EXTENSIONS = {".csv", ".json", ".parquet", ".pkl", ".xlsx", ".xls", ".zip", ".xml", ".xbrl", ".txt"}

DART_TITLE_SOURCES = [
    {
        "path": Path("Paper3") / "dart_apriori" / "data" / "raw" / "dart_major_reports_full.csv",
        "label": "Paper3 DART major-event reports",
    },
    {
        "path": ROOT / "data" / "processed" / "events_dart.csv",
        "label": "Paper 4 normalized DART events",
    },
    {
        "path": ROOT / "data" / "raw" / "dart_gap" / "dart_gap_outcome_candidates.csv",
        "label": "Paper 4 DART outcome candidates",
    },
]


def normalize_path(value: object) -> str:
    return str(value).replace("/", "\\")


def match_keywords(text: object) -> list[str]:
    lowered = str(text).lower()
    return [keyword for keyword in KEYWORDS if keyword.lower() in lowered]


def is_generated(path: str) -> bool:
    normalized = normalize_path(path).lower()
    return any(marker.lower() in normalized for marker in GENERATED_MARKERS)


def classify(row: dict[str, Any]) -> str:
    if row["source"] == "dart_title_content":
        return "dart_title_candidate"
    path = normalize_path(row.get("path", ""))
    ext = str(row.get("extension", "")).lower()
    if is_generated(path):
        return "generated_paper4_reference"
    if ext and ext not in DATA_EXTENSIONS:
        return "non_data_reference"
    if row["source"] == "inspected_file_metadata" and row.get("matched_field") == "columns":
        return "external_column_candidate"
    return "external_path_candidate"


def rows_from_inventory(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path, low_memory=False)
    rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        full_path = str(getattr(row, "path", ""))
        relpath = str(getattr(row, "relpath", ""))
        text = f"{full_path} {relpath}"
        keywords = match_keywords(text)
        if not keywords:
            continue
        rows.append(
            {
                "source": "data_inventory_path",
                "path": full_path,
                "relpath": relpath,
                "extension": str(getattr(row, "extension", Path(full_path).suffix)).lower(),
                "size_mb": float(getattr(row, "size_mb", 0) or 0),
                "matched_field": "path",
                "matched_keywords": ";".join(keywords),
                "sample_text": "",
            }
        )
    return rows


def rows_from_inspected(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in records:
        full_path = str(item.get("path", ""))
        relpath = str(item.get("relpath", ""))
        columns = item.get("columns") or []
        text = " ".join([full_path, relpath, " ".join(map(str, columns))])
        keywords = match_keywords(text)
        if not keywords:
            continue
        matched_columns = [column for column in columns if match_keywords(column)]
        rows.append(
            {
                "source": "inspected_file_metadata",
                "path": full_path,
                "relpath": relpath,
                "extension": str(item.get("extension", Path(full_path).suffix)).lower(),
                "size_mb": item.get("size_mb", ""),
                "matched_field": "columns" if matched_columns else "path",
                "matched_keywords": ";".join(keywords),
                "sample_text": "; ".join(map(str, matched_columns[:12])),
            }
        )
    return rows


def rows_from_file_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        file_path = line.strip()
        if not file_path:
            continue
        keywords = match_keywords(file_path)
        if not keywords:
            continue
        rows.append(
            {
                "source": "desktop_file_index_path",
                "path": file_path,
                "relpath": file_path,
                "extension": Path(file_path).suffix.lower(),
                "size_mb": "",
                "matched_field": "path",
                "matched_keywords": ";".join(keywords),
                "sample_text": "",
            }
        )
    return rows


def rows_from_desktop_keyword_scan(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            item = Path(dirpath) / filename
            file_path = str(item)
            keywords = match_keywords(file_path)
            if not keywords:
                continue
            rows.append(
                {
                    "source": "live_desktop_keyword_scan",
                    "path": file_path,
                    "relpath": str(item.relative_to(root)) if item.is_relative_to(root) else item.name,
                    "extension": item.suffix.lower(),
                    "size_mb": round(item.stat().st_size / (1024 * 1024), 4) if item.exists() else "",
                    "matched_field": "path",
                    "matched_keywords": ";".join(keywords),
                    "sample_text": "",
                }
            )
    return rows


def rows_from_live_project(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        file_path = str(item)
        keywords = match_keywords(file_path)
        if not keywords:
            continue
        rows.append(
            {
                "source": "live_project_tree_path",
                "path": file_path,
                "relpath": str(item.relative_to(path)) if item.is_relative_to(path) else file_path,
                "extension": item.suffix.lower(),
                "size_mb": round(item.stat().st_size / (1024 * 1024), 4),
                "matched_field": "path",
                "matched_keywords": ";".join(keywords),
                "sample_text": "",
            }
        )
    return rows


def rows_from_dart_titles(sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source in sources:
        path = Path(source["path"])
        label = str(source["label"])
        if not path.exists():
            source_summaries.append(
                {"label": label, "path": public_path(str(path), "dart_source"), "exists": False, "rows": 0, "matches": 0}
            )
            continue
        df = pd.read_csv(path, low_memory=False)
        match_mask = pd.Series(False, index=df.index)
        text = df.astype(str).agg(" ".join, axis=1)
        lowered = text.str.lower()
        for keyword in KEYWORDS:
            match_mask = match_mask | lowered.str.contains(keyword.lower(), regex=False, na=False)
        matches = df.loc[match_mask]
        source_summaries.append(
            {
                "label": label,
                "path": public_path(str(path), "dart_source"),
                "exists": True,
                "rows": int(len(df)),
                "matches": int(len(matches)),
            }
        )
        for idx, row in matches.head(200).iterrows():
            row_text = " ".join(str(value) for value in row.to_dict().values())
            keywords = match_keywords(row_text)
            rows.append(
                {
                    "source": "dart_title_content",
                    "path": str(path),
                    "relpath": label,
                    "extension": path.suffix.lower(),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 4),
                    "matched_field": "row_content",
                    "matched_keywords": ";".join(keywords),
                    "sample_text": truncate(row_text),
                    "row_index": int(idx),
                }
            )
    return rows, source_summaries


def truncate(value: object, limit: int = 500) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("source", "")),
            normalize_path(row.get("path", "")).lower(),
            str(row.get("row_index", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        item = dict(row)
        item["classification"] = classify(item)
        out.append(item)
    return out


def summarize(rows: list[dict[str, Any]], dart_sources: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    keyword_counts: dict[str, int] = {}
    for row in rows:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
        for keyword in str(row.get("matched_keywords", "")).split(";"):
            if keyword:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    empirical_rows = [
        row
        for row in rows
        if row["classification"] in {"dart_title_candidate", "external_column_candidate", "external_path_candidate"}
    ]
    dart_title_matches = sum(int(item.get("matches", 0)) for item in dart_sources)
    return {
        "status": "t04_candidates_found" if empirical_rows else "no_t04_empirical_source_found",
        "matched_rows": len(rows),
        "empirical_candidate_rows": len(empirical_rows),
        "dart_title_match_rows": dart_title_matches,
        "classification_counts": counts,
        "keyword_counts": dict(sorted(keyword_counts.items())),
        "dart_title_sources": dart_sources,
        "empirical_candidate_examples": [public_row(row) for row in empirical_rows[:20]],
        "guardrail": "Current evidence does not support T04 going-concern/audit-opinion coverage unless empirical candidates are inspected and normalized. T04 rules should remain registry-only in v1 when this audit reports no empirical source.",
    }


def public_path(value: object, scope: str = "candidate") -> str:
    text = str(value or "")
    if text.startswith("public_") or text.startswith("local_inventory/"):
        return text
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]
    suffix = Path(text).suffix.lower() or ".file"
    return f"public_{scope}/source_{digest}{suffix}"


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["path"] = public_path(row.get("path"), str(row.get("classification", "candidate")))
    out["relpath"] = out["path"]
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "classification",
        "source",
        "path",
        "relpath",
        "extension",
        "size_mb",
        "matched_field",
        "matched_keywords",
        "sample_text",
        "row_index",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = public_row(row)
            writer.writerow({field: out.get(field, "") for field in fields})


def escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(summary: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# T04 Local Data Discovery",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Matched rows | {summary['matched_rows']} |",
        f"| Empirical candidate rows | {summary['empirical_candidate_rows']} |",
        f"| DART title/content matches | {summary['dart_title_match_rows']} |",
        "",
        "## Classification Counts",
        "",
        "| Classification | Rows |",
        "|---|---:|",
    ]
    if summary["classification_counts"]:
        for key, value in sorted(summary["classification_counts"].items()):
            lines.append(f"| `{key}` | {value} |")
    else:
        lines.append("| `none` | 0 |")

    lines.extend(
        [
            "",
            "## DART Title Sources",
            "",
            "| Source | Rows | Matches | Path |",
            "|---|---:|---:|---|",
        ]
    )
    for source in summary["dart_title_sources"]:
        lines.append(
            "| "
            f"{escape(source['label'])} | "
            f"{source['rows']} | "
            f"{source['matches']} | "
            f"`{escape(source['path'])}` |"
        )

    empirical = [
        row
        for row in rows
        if row["classification"] in {"dart_title_candidate", "external_column_candidate", "external_path_candidate"}
    ]
    lines.extend(["", "## Empirical Candidate Examples", ""])
    if empirical:
        lines.extend(["| Classification | Path | Keywords | Sample |", "|---|---|---|---|"])
        for row in empirical[:20]:
            lines.append(
                "| "
                f"`{row['classification']}` | "
                f"`{escape(public_path(row.get('path'), str(row.get('classification', 'candidate'))))}` | "
                f"`{escape(row['matched_keywords'])}` | "
                f"{escape(truncate(row.get('sample_text', ''), 160))} |"
            )
    else:
        lines.append("No empirical local T04 going-concern/audit-opinion candidate source was found.")

    lines.extend(["", "## Guardrail", "", summary["guardrail"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default=str(ROOT / "data" / "audit" / "data_inventory.csv"))
    parser.add_argument("--inspected", default=str(ROOT / "data" / "audit" / "inspected_files.json"))
    parser.add_argument("--file-index", default=str(ROOT / "data" / "audit" / "desktop_data_files_rg.txt"))
    parser.add_argument(
        "--desktop-root",
        default="",
        help="Optional broad local root scan. Disabled by default because full Desktop traversal is a heavy audit path.",
    )
    parser.add_argument("--live-root", default=str(ROOT))
    parser.add_argument("--out-csv", default=str(ROOT / "data" / "audit" / "t04_local_candidate_files.csv"))
    parser.add_argument("--out-json", default=str(ROOT / "results" / "t04_local_discovery.json"))
    parser.add_argument("--out-md", default=str(ROOT / "results" / "t04_local_discovery.md"))
    args = parser.parse_args()

    dart_rows, dart_source_summaries = rows_from_dart_titles(DART_TITLE_SOURCES)
    rows = deduplicate(
        rows_from_inventory(Path(args.inventory))
        + rows_from_inspected(Path(args.inspected))
        + rows_from_file_index(Path(args.file_index))
        + (rows_from_desktop_keyword_scan(Path(args.desktop_root)) if args.desktop_root else [])
        + rows_from_live_project(Path(args.live_root))
        + dart_rows
    )
    summary = summarize(rows, dart_source_summaries)

    write_csv(rows, Path(args.out_csv))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, rows, Path(args.out_md))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "empirical_candidate_rows": summary["empirical_candidate_rows"],
                "dart_title_match_rows": summary["dart_title_match_rows"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
