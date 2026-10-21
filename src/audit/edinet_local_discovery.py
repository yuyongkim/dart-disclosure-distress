from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path("Paper 4")
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
    "edinet",
    "japan",
    "japanese",
    "fsa",
    "金融庁",
    "臨時報告",
    "有価証券",
    "xbrl",
    "jpfr",
    "edinetcode",
]
GENERATED_MARKERS = [
    "Paper 4\\data\\raw\\edinet",
    "Paper 4/results/edinet",
    "Paper 4\\results\\edinet",
    "Paper 4\\src\\collect\\edinet",
    "Paper 4\\data\\audit\\edinet_local",
    "Paper 4\\results\\edinet_local",
    "Paper 4\\src\\audit\\edinet_local",
]
DATA_EXTENSIONS = {".csv", ".json", ".parquet", ".pkl", ".xlsx", ".xls", ".zip", ".xml", ".xbrl"}
JP_REPORT_ROOTS = [
    Path.home() / "Desktop" / "M&A" / "data" / "raw" / "jp_reports",
    Path.home() / "Desktop" / "M&A" / "data" / "processed" / "jp_reports_text",
]


def normalize_path(value: object) -> str:
    return str(value).replace("/", "\\")


def is_generated(path: str) -> bool:
    normalized = normalize_path(path).lower()
    return any(marker.lower() in normalized for marker in GENERATED_MARKERS)


def match_keywords(path: str) -> list[str]:
    lowered = path.lower()
    return [kw for kw in KEYWORDS if kw.lower() in lowered]


def rows_from_inventory(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path, low_memory=False)
    rows = []
    for row in df.itertuples(index=False):
        full_path = str(getattr(row, "path", ""))
        relpath = str(getattr(row, "relpath", ""))
        text = f"{full_path} {relpath}"
        keywords = match_keywords(text)
        if not keywords:
            continue
        rows.append(
            {
                "source": "data_inventory",
                "path": full_path,
                "relpath": relpath,
                "extension": str(getattr(row, "extension", Path(full_path).suffix)).lower(),
                "size_mb": float(getattr(row, "size_mb", 0) or 0),
                "mtime": str(getattr(row, "mtime", "")),
                "matched_keywords": ";".join(keywords),
            }
        )
    return rows


def rows_from_file_index(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        file_path = line.strip()
        if not file_path:
            continue
        keywords = match_keywords(file_path)
        if not keywords:
            continue
        p = Path(file_path)
        rows.append(
            {
                "source": "local_file_index",
                "path": file_path,
                "relpath": file_path,
                "extension": p.suffix.lower(),
                "size_mb": "",
                "mtime": "",
                "matched_keywords": ";".join(keywords),
            }
        )
    return rows


def rows_from_desktop_keyword_scan(root: Path) -> list[dict]:
    if not root.exists():
        return []
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            item = Path(dirpath) / filename
            file_path = str(item)
            keywords = match_keywords(file_path)
            if not keywords:
                continue
            try:
                stat = item.stat()
                size_mb = round(stat.st_size / (1024 * 1024), 4)
            except OSError:
                size_mb = ""
            rows.append(
                {
                    "source": "live_desktop_keyword_scan",
                    "path": file_path,
                    "relpath": str(item.relative_to(root)) if item.is_relative_to(root) else item.name,
                    "extension": item.suffix.lower(),
                    "size_mb": size_mb,
                    "mtime": "",
                    "matched_keywords": ";".join(keywords),
                }
            )
    return rows


def rows_from_live_root(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        file_path = str(item)
        keywords = match_keywords(file_path)
        if not keywords:
            continue
        stat = item.stat()
        rows.append(
            {
                "source": "live_project_tree",
                "path": file_path,
                "relpath": str(item.relative_to(path)) if item.is_relative_to(path) else file_path,
                "extension": item.suffix.lower(),
                "size_mb": round(stat.st_size / (1024 * 1024), 4),
                "mtime": "",
                "matched_keywords": ";".join(keywords),
            }
        )
    return rows


def rows_from_jp_report_roots(paths: list[Path]) -> list[dict]:
    rows = []
    for root in paths:
        if not root.exists():
            continue
        for item in root.rglob("*"):
            if not item.is_file() or item.suffix.lower() not in {".pdf", ".txt"}:
                continue
            stat = item.stat()
            filename = item.name.lower()
            keywords = ["jp_reports"]
            if "annual_securities_report" in filename:
                keywords.append("annual_securities_report")
            if "annual_report" in filename:
                keywords.append("annual_report")
            if "integrated_report" in filename:
                keywords.append("integrated_report")
            if "financial_report" in filename or "financials" in filename:
                keywords.append("financial_report")
            row = {
                "source": "external_local_japanese_report_cache",
                "path": str(item),
                "relpath": str(item.relative_to(root)),
                "extension": item.suffix.lower(),
                "size_mb": round(stat.st_size / (1024 * 1024), 4),
                "mtime": "",
                "matched_keywords": ";".join(keywords),
                "classification_hint": "external_japanese_report_candidate",
            }
            if item.suffix.lower() == ".txt":
                try:
                    head = item.read_text(encoding="utf-8", errors="ignore")[:5000].lower()
                except OSError:
                    head = ""
                row["mentions_edinet"] = "yes" if "edinet" in head else "no"
                row["mentions_annual_securities_report"] = "yes" if "annual securities report" in head else "no"
            rows.append(row)
    return rows


def classify(row: dict) -> str:
    path = normalize_path(row["path"])
    ext = str(row.get("extension", "")).lower()
    if row.get("classification_hint") == "external_japanese_report_candidate":
        return "external_japanese_report_candidate"
    if is_generated(path):
        return "generated_paper4_edinet_artifact"
    if ext and ext not in DATA_EXTENSIONS:
        return "non_data_or_code_reference"
    return "external_reusable_candidate"


def deduplicate(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for row in rows:
        key = normalize_path(row["path"]).lower()
        if key in seen:
            continue
        seen.add(key)
        row = dict(row)
        row["classification"] = classify(row)
        out.append(row)
    return out


def summarize(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    keyword_counts: dict[str, int] = {}
    for row in rows:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
        for keyword in str(row.get("matched_keywords", "")).split(";"):
            if keyword:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    external = [row for row in rows if row["classification"] == "external_reusable_candidate"]
    jp_reports = [row for row in rows if row["classification"] == "external_japanese_report_candidate"]
    generated = [row for row in rows if row["classification"] == "generated_paper4_edinet_artifact"]
    edinet_mentions = [row for row in jp_reports if row.get("mentions_edinet") == "yes"]
    return {
        "status": (
            "external_edinet_candidates_found"
            if external
            else "japanese_report_support_found_no_edinet_collection"
            if jp_reports
            else "no_external_edinet_dataset_found"
        ),
        "matched_rows": len(rows),
        "external_reusable_candidate_rows": len(external),
        "external_japanese_report_candidate_rows": len(jp_reports),
        "external_japanese_report_edinet_mentions": len(edinet_mentions),
        "generated_paper4_edinet_artifact_rows": len(generated),
        "classification_counts": counts,
        "keyword_counts": dict(sorted(keyword_counts.items())),
        "external_candidate_examples": [public_row(row) for row in external[:20]],
        "japanese_report_candidate_examples": [public_row(row) for row in jp_reports[:20]],
        "guardrail": "Generated Paper 4 EDINET plans/status files are not reusable EDINET empirical data. local Japanese annual-report texts can support parser prototyping, but they are not a 2015-2024 EDINET document-list collection and do not move EDINET out of Phase 2.",
    }


def public_row(row: dict) -> dict:
    out = dict(row)
    out["path"] = display_path(row)
    out["relpath"] = display_path(row)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "classification",
        "source",
        "path",
        "relpath",
        "extension",
        "size_mb",
        "mtime",
        "matched_keywords",
        "mentions_edinet",
        "mentions_annual_securities_report",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["path"] = display_path(row)
            writer.writerow({field: out.get(field, "") for field in fields})


def write_markdown(summary: dict, rows: list[dict], path: Path) -> None:
    lines = [
        "# EDINET Local Data Discovery",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Matched rows | {summary['matched_rows']} |",
        f"| External reusable candidates | {summary['external_reusable_candidate_rows']} |",
        f"| Japanese report candidates | {summary['external_japanese_report_candidate_rows']} |",
        f"| Japanese report candidates mentioning EDINET | {summary['external_japanese_report_edinet_mentions']} |",
        f"| Generated Paper 4 EDINET artifacts | {summary['generated_paper4_edinet_artifact_rows']} |",
        "",
        "## Classification Counts",
        "",
        "| Classification | Rows |",
        "|---|---:|",
    ]
    for key, value in sorted(summary["classification_counts"].items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## External Candidate Examples", ""])
    external = [row for row in rows if row["classification"] == "external_reusable_candidate"]
    if external:
        lines.extend(["| Path | Keywords |", "|---|---|"])
        for row in external[:20]:
            lines.append(f"| `{escape(display_path(row))}` | `{escape(row['matched_keywords'])}` |")
    else:
        lines.append("No external reusable EDINET document-list/current-report dataset was found in the current local inventory/index.")
    lines.extend(["", "## Japanese Report Candidate Examples", ""])
    jp_reports = [row for row in rows if row["classification"] == "external_japanese_report_candidate"]
    if jp_reports:
        lines.extend(["| Path | Keywords | Mentions EDINET |", "|---|---|---:|"])
        for row in jp_reports[:20]:
            lines.append(
                f"| `{escape(display_path(row))}` | `{escape(row['matched_keywords'])}` | {escape(row.get('mentions_edinet', ''))} |"
            )
    else:
        lines.append("No local Japanese report PDF/text candidates were found.")
    lines.extend(["", "## Guardrail", "", summary["guardrail"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def display_path(row: dict) -> str:
    relpath = str(row.get("relpath") or Path(str(row.get("path", ""))).name)
    if row.get("classification") == "external_japanese_report_candidate":
        return f"external_local_japanese_report_cache/{relpath}"
    if row.get("classification") == "external_reusable_candidate":
        source = str(row.get("path") or relpath)
        digest = hashlib.sha1(source.encode("utf-8", errors="ignore")).hexdigest()[:10]
        suffix = Path(source).suffix.lower() or ".file"
        return f"external_local_candidate/candidate_{digest}{suffix}"
    return relpath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default=str(ROOT / "data" / "audit" / "data_inventory.csv"))
    parser.add_argument("--file-index", default=str(ROOT / "data" / "audit" / "desktop_data_files_rg.txt"))
    parser.add_argument(
        "--desktop-root",
        default="",
        help="Optional broad local root scan. Disabled by default because full Desktop traversal is a heavy audit path.",
    )
    parser.add_argument("--live-root", default=str(ROOT))
    parser.add_argument("--out-csv", default=str(ROOT / "data" / "audit" / "edinet_local_candidate_files.csv"))
    parser.add_argument("--out-json", default=str(ROOT / "results" / "edinet_local_discovery.json"))
    parser.add_argument("--out-md", default=str(ROOT / "results" / "edinet_local_discovery.md"))
    args = parser.parse_args()

    rows = deduplicate(
        rows_from_inventory(Path(args.inventory))
        + rows_from_file_index(Path(args.file_index))
        + (rows_from_desktop_keyword_scan(Path(args.desktop_root)) if args.desktop_root else [])
        + rows_from_live_root(Path(args.live_root))
        + rows_from_jp_report_roots(JP_REPORT_ROOTS)
    )
    summary = summarize(rows)
    write_csv(rows, Path(args.out_csv))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, rows, Path(args.out_md))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
