from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("Paper 4")
DEFAULT_RAW_ROOT = Path.home() / "Desktop" / "M&A" / "data" / "raw" / "jp_reports"
DEFAULT_TEXT_ROOT = Path.home() / "Desktop" / "M&A" / "data" / "processed" / "jp_reports_text"

TAXONOMY_KEYWORDS = {
    "T01": [
        "bankruptcy",
        "insolvency",
        "civil rehabilitation",
        "corporate reorganization",
        "liquidation",
        "dissolution",
        "receivership",
    ],
    "T02": [
        "delisting",
        "trading suspension",
        "securities listing regulations",
        "stock exchange delisting",
    ],
    "T03": [
        "default",
        "acceleration",
        "accelerate",
        "covenant",
        "breach of covenant",
        "failure to pay",
    ],
    "T04": [
        "going concern",
        "material uncertainty",
        "audit opinion",
        "qualified opinion",
        "emphasis of matter",
    ],
    "T05": [
        "capital reduction",
        "capital increase",
        "impairment",
        "liquidity",
        "net assets",
        "convertible bonds",
        "financing",
        "restructuring",
    ],
    "T06": [
        "director",
        "officer",
        "representative executive officer",
        "management change",
        "governance",
        "board of directors",
    ],
}


def report_type_from_name(name: str) -> str:
    lowered = name.lower()
    if "annual_securities_report" in lowered:
        return "annual_securities_report"
    if "integrated_report" in lowered:
        return "integrated_report"
    if "annual_report" in lowered:
        return "annual_report"
    if "financial_report" in lowered or "financials" in lowered:
        return "financial_report"
    return "other_report"


def company_slug_from_name(name: str) -> str:
    stem = Path(name).stem
    match = re.match(r"(.+?)_20\d{2}_", stem)
    return match.group(1) if match else stem


def report_year_from_name(name: str) -> int | None:
    match = re.search(r"(20\d{2})", name)
    return int(match.group(1)) if match else None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_cover_fields(text: str) -> dict[str, str]:
    fields = {
        "document_filed": "",
        "applicable_law": "",
        "filed_with": "",
        "filing_date": "",
        "fiscal_year": "",
        "company_name": "",
        "company_name_english": "",
    }
    patterns = {
        "document_filed": r"\[Document Filed\]\s+(.+)",
        "applicable_law": r"\[Applicable Law\]\s+(.+)",
        "filed_with": r"\[Filed with\]\s+(.+)",
        "filing_date": r"\[Filing Date\]\s+(.+)",
        "fiscal_year": r"\[Fiscal Year\]\s+(.+)",
        "company_name": r"\[Company Name\]\s+(.+)",
        "company_name_english": r"\[Company Name in English\]\s+(.+)",
    }
    head = "\n".join(text.splitlines()[:200])
    for key, pattern in patterns.items():
        match = re.search(pattern, head, flags=re.IGNORECASE)
        if match:
            fields[key] = " ".join(match.group(1).split())
    return fields


def keyword_count(text_lower: str, keyword: str) -> int:
    return len(re.findall(re.escape(keyword.lower()), text_lower))


def first_excerpt(text: str, keyword: str, window: int = 180) -> str:
    match = re.search(re.escape(keyword), text, flags=re.IGNORECASE)
    if not match:
        return ""
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    return " ".join(text[start:end].split())


def build_file_index(raw_root: Path, text_root: Path) -> pd.DataFrame:
    records: dict[str, dict[str, Any]] = {}
    for root, kind in [(raw_root, "raw_pdf"), (text_root, "extracted_text")]:
        if not root.exists():
            continue
        for path in root.glob("*"):
            if not path.is_file() or path.suffix.lower() not in {".pdf", ".txt"}:
                continue
            report_id = path.stem
            slug = company_slug_from_name(path.name)
            record = records.setdefault(
                report_id,
                {
                    "report_id": report_id,
                    "company_slug": slug,
                    "report_year": report_year_from_name(path.name),
                    "report_type": report_type_from_name(path.name),
                    "raw_pdf_path": "",
                    "text_path": "",
                    "raw_pdf_size_bytes": 0,
                    "text_size_bytes": 0,
                },
            )
            if kind == "raw_pdf":
                record["raw_pdf_path"] = str(path)
                record["raw_pdf_size_bytes"] = int(path.stat().st_size)
            else:
                record["text_path"] = str(path)
                record["text_size_bytes"] = int(path.stat().st_size)
                record["report_type"] = report_type_from_name(path.name)
                record["report_year"] = report_year_from_name(path.name)
    return pd.DataFrame(records.values()).sort_values(["company_slug", "report_type"]).reset_index(drop=True)


def enrich_with_text(index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for row in index.to_dict(orient="records"):
        text_path = str(row.get("text_path") or "")
        text = read_text(Path(text_path)) if text_path else ""
        text_lower = text.lower()
        cover = parse_cover_fields(text) if text else {}
        edinet_mention = "edinet" in text_lower
        annual_securities_report_mention = "annual securities report" in text_lower
        taxonomy_hit_counts: dict[str, int] = {}
        for code, keywords in TAXONOMY_KEYWORDS.items():
            hits = {kw: keyword_count(text_lower, kw) for kw in keywords}
            hit_total = sum(hits.values())
            taxonomy_hit_counts[code] = hit_total
            if hit_total:
                first_kw = next((kw for kw, count in hits.items() if count), "")
                event_rows.append(
                    {
                        "source": "JP_REPORT_PROTOTYPE",
                        "company_slug": row["company_slug"],
                        "report_year": row.get("report_year"),
                        "report_type": row.get("report_type"),
                        "taxonomy_code": code,
                        "taxonomy_label": taxonomy_label(code),
                        "keyword_count": int(hit_total),
                        "matched_keywords": ";".join(kw for kw, count in hits.items() if count),
                        "first_match_keyword": first_kw,
                        "first_match_excerpt": first_excerpt(text, first_kw) if first_kw else "",
                        "text_path": text_path,
                        "prototype_only": True,
                    }
                )
        file_row = dict(row)
        file_row.update(
            {
                "has_text": bool(text_path),
                "text_chars": int(len(text)),
                "mentions_edinet": bool(edinet_mention),
                "mentions_annual_securities_report": bool(annual_securities_report_mention),
                "document_filed": cover.get("document_filed", ""),
                "applicable_law": cover.get("applicable_law", ""),
                "filed_with": cover.get("filed_with", ""),
                "filing_date": cover.get("filing_date", ""),
                "fiscal_year": cover.get("fiscal_year", ""),
                "company_name": cover.get("company_name", ""),
                "company_name_english": cover.get("company_name_english", ""),
                "taxonomy_hit_counts_json": json.dumps(taxonomy_hit_counts, ensure_ascii=False),
            }
        )
        file_rows.append(file_row)
    return pd.DataFrame(file_rows), pd.DataFrame(event_rows)


def taxonomy_label(code: str) -> str:
    return {
        "T01": "insolvency_or_rehabilitation",
        "T02": "delisting_or_trading_suspension",
        "T03": "debt_default_or_acceleration",
        "T04": "audit_opinion_or_going_concern",
        "T05": "capital_structure_stress",
        "T06": "governance_or_control_change",
    }.get(code, "other_event")


def summarize(files: pd.DataFrame, events: pd.DataFrame, raw_root: Path, text_root: Path) -> dict[str, Any]:
    by_taxonomy = {}
    if not events.empty:
        by_taxonomy = {
            str(row.taxonomy_code): {
                "candidate_rows": int(row.candidate_rows),
                "companies": int(row.companies),
                "keyword_hits": int(row.keyword_hits),
            }
            for row in events.groupby("taxonomy_code").agg(
                candidate_rows=("taxonomy_code", "size"),
                companies=("company_slug", "nunique"),
                keyword_hits=("keyword_count", "sum"),
            ).reset_index().itertuples(index=False)
        }
    return {
        "status": "prototype_created",
        "source": "local Japanese reports",
        "raw_root": "external_japanese_report_cache/raw_pdf",
        "text_root": "external_japanese_report_cache/extracted_text",
        "report_files": int(len(files)),
        "text_files": int(files["has_text"].sum()) if "has_text" in files else 0,
        "reports_mentioning_edinet": int(files["mentions_edinet"].sum()) if "mentions_edinet" in files else 0,
        "annual_securities_report_texts": int(files["mentions_annual_securities_report"].sum())
        if "mentions_annual_securities_report" in files
        else 0,
        "taxonomy_candidate_rows": int(len(events)),
        "taxonomy_candidate_counts": by_taxonomy,
        "guardrail": "This is parser-prototype support from 2025 local Japanese report files. It is not EDINET API document-list collection, not a 2015-2024 EDINET empirical panel, and not reportable as EDINET validation evidence.",
    }


def write_outputs(files: pd.DataFrame, events: pd.DataFrame, summary: dict[str, Any], root: Path) -> None:
    raw_dir = root / "data" / "raw" / "edinet"
    processed_dir = root / "data" / "processed"
    results_dir = root / "results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    files_csv = raw_dir / "jp_report_parser_prototype_index.csv"
    events_csv = processed_dir / "jp_report_parser_prototype_events.csv"
    events_parquet = processed_dir / "jp_report_parser_prototype_events.parquet"
    summary_json = results_dir / "jp_report_parser_prototype_summary.json"
    summary_md = results_dir / "jp_report_parser_prototype_summary.md"

    files.to_csv(files_csv, index=False, encoding="utf-8-sig")
    events.to_csv(events_csv, index=False, encoding="utf-8-sig")
    try:
        events.to_parquet(events_parquet, index=False)
    except Exception as exc:
        summary["parquet_error"] = repr(exc)
    summary.update(
        {
            "index_csv": str(files_csv),
            "events_csv": str(events_csv),
            "events_parquet": str(events_parquet),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        }
    )
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, summary_md)


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Japanese Report Parser Prototype",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This artifact indexes locally available local Japanese report PDF/text files and extracts prototype taxonomy keyword candidates.",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Report files | {summary['report_files']} |",
        f"| Text files | {summary['text_files']} |",
        f"| Reports mentioning EDINET | {summary['reports_mentioning_edinet']} |",
        f"| Annual securities report texts | {summary['annual_securities_report_texts']} |",
        f"| Taxonomy candidate rows | {summary['taxonomy_candidate_rows']} |",
        "",
        "## Taxonomy Candidate Counts",
        "",
        "| Taxonomy | Candidate rows | Companies | Keyword hits |",
        "|---|---:|---:|---:|",
    ]
    for code, item in sorted(summary["taxonomy_candidate_counts"].items()):
        lines.append(
            f"| `{code}` | {item['candidate_rows']} | {item['companies']} | {item['keyword_hits']} |"
        )
    if not summary["taxonomy_candidate_counts"]:
        lines.append("| n/a | 0 | 0 | 0 |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Index CSV: `{summary['index_csv']}`",
            f"- Event candidates CSV: `{summary['events_csv']}`",
            f"- Event candidates parquet: `{summary['events_parquet']}`",
            "",
            "## Guardrail",
            "",
            summary["guardrail"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(raw_root: Path, text_root: Path, root: Path) -> dict[str, Any]:
    index = build_file_index(raw_root, text_root)
    files, events = enrich_with_text(index)
    summary = summarize(files, events, raw_root, text_root)
    write_outputs(files, events, summary, root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--text-root", default=str(DEFAULT_TEXT_ROOT))
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    summary = run(Path(args.raw_root), Path(args.text_root), Path(args.root))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "report_files": summary["report_files"],
                "text_files": summary["text_files"],
                "taxonomy_candidate_rows": summary["taxonomy_candidate_rows"],
                "reports_mentioning_edinet": summary["reports_mentioning_edinet"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
