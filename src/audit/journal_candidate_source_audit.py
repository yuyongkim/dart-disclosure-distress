from __future__ import annotations

import argparse
import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
MANUSCRIPT = ROOT / "manuscript" / "submission_candidate_journals_ko.md"
CACHE = ROOT / "data" / "audit" / "submission_candidate_journal_sources.json"
RESULTS = ROOT / "results"
OUT_JSON = RESULTS / "journal_candidate_source_audit.json"
OUT_MD = RESULTS / "journal_candidate_source_audit.md"
MAX_CACHE_AGE_DAYS = 90
EXPECTED_CANDIDATE_JOURNAL_COUNT = 10
EXPECTED_CANDIDATE_JOURNALS = [
    "Expert Systems with Applications",
    "Knowledge-Based Systems",
    "Decision Support Systems",
    "Information Sciences",
    "Annals of Operations Research",
    "Computational Economics",
    "Asia-Pacific Financial Markets",
    "Review of Quantitative Finance and Accounting",
    "International Journal of Finance & Economics",
    "Applied Soft Computing",
]
MIN_RELATED_WORK_DATE = date(2025, 7, 1)
HTML_TAG_RE = re.compile(r"<[^>]+>")

POLICY_URLS = {
    "elsevier_pricing": "https://www.elsevier.com/about/policies-and-standards/pricing",
    "elsevier_choice": "https://www.elsevier.com/researcher/author/open-access/choice",
    "elsevier_submission_fees": "https://www.elsevier.com/researcher/author/policies-and-guidelines/submission-fees",
    "springer_cost": "https://support.nature.com/en/support/solutions/articles/6000084580-costs-of-publishing-in-springer-nature-journal",
    "wiley_hybrid": "https://authors.wiley.com/author-resources/Journal-Authors/open-access/hybrid-open-access/index.html",
}

POLICY_FOOTNOTE_GROUPS = {
    "elsevier_oa": {
        "footnote": "[^elsevier-oa]",
        "policy_labels": ["elsevier_pricing", "elsevier_choice"],
    },
    "springer_cost": {
        "footnote": "[^springer-cost]",
        "policy_labels": ["springer_cost"],
    },
    "wiley_hybrid": {
        "footnote": "[^wiley-hybrid]",
        "policy_labels": ["wiley_hybrid"],
    },
}

EXPECTED_CANDIDATE_POLICY_GROUP = {
    "Expert Systems with Applications": "elsevier_oa",
    "Knowledge-Based Systems": "elsevier_oa",
    "Decision Support Systems": "elsevier_oa",
    "Information Sciences": "elsevier_oa",
    "Annals of Operations Research": "springer_cost",
    "Computational Economics": "springer_cost",
    "Asia-Pacific Financial Markets": "springer_cost",
    "Review of Quantitative Finance and Accounting": "springer_cost",
    "International Journal of Finance & Economics": "wiley_hybrid",
    "Applied Soft Computing": "elsevier_oa",
}

DOIS = {
    "eswa_2025": {
        "doi": "10.1016/j.eswa.2025.127832",
        "container": "Expert Systems with Applications",
        "title_contains": "financial distress prediction",
    },
    "kbs_2026": {
        "doi": "10.1016/j.knosys.2026.115790",
        "container": "Knowledge-Based Systems",
        "title_contains": "early warning of default risk",
    },
    "dss_2026": {
        "doi": "10.1016/j.dss.2026.114615",
        "container": "Decision Support Systems",
        "title_contains": "Financial statement fraud detection",
    },
    "ins_2025": {
        "doi": "10.1016/j.ins.2025.122870",
        "container": "Information Sciences",
        "title_contains": "financial distress prediction",
    },
    "aor_2025": {
        "doi": "10.1007/s10479-025-06706-5",
        "container": "Annals of Operations Research",
        "title_contains": "financial distress prediction",
    },
    "comecon_2026": {
        "doi": "10.1007/s10614-026-11387-8",
        "container": "Computational Economics",
        "title_contains": "Financial Distress Prediction",
    },
    "comecon_2025": {
        "doi": "10.1007/s10614-024-10761-8",
        "container": "Computational Economics",
        "title_contains": "Financial Distress Prediction",
    },
    "apfm_2026": {
        "doi": "10.1007/s10690-025-09525-7",
        "container": "Asia-Pacific Financial Markets",
        "title_contains": "Predicting Financial Distress",
    },
    "rqfa_2025": {
        "doi": "10.1007/s11156-023-01192-x",
        "container": "Review of Quantitative Finance and Accounting",
        "title_contains": "Bankruptcy prediction",
    },
    "ijfe_2025_esg": {
        "doi": "10.1002/ijfe.70077",
        "container": "International Journal of Finance & Economics",
        "title_contains": "Predicting Financial Distress",
    },
    "ijfe_2025_stacking": {
        "doi": "10.1002/ijfe.3020",
        "container": "International Journal of Finance & Economics",
        "title_contains": "Multi-class financial distress prediction",
    },
    "asoc_2026": {
        "doi": "10.1016/j.asoc.2026.115085",
        "container": "Applied Soft Computing",
        "title_contains": "financial distress prediction",
    },
}

EXPECTED_CANDIDATE_DOIS = {
    "Expert Systems with Applications": ["10.1016/j.eswa.2025.127832"],
    "Knowledge-Based Systems": ["10.1016/j.knosys.2026.115790"],
    "Decision Support Systems": ["10.1016/j.dss.2026.114615"],
    "Information Sciences": ["10.1016/j.ins.2025.122870"],
    "Annals of Operations Research": ["10.1007/s10479-025-06706-5"],
    "Computational Economics": ["10.1007/s10614-026-11387-8", "10.1007/s10614-024-10761-8"],
    "Asia-Pacific Financial Markets": ["10.1007/s10690-025-09525-7"],
    "Review of Quantitative Finance and Accounting": ["10.1007/s11156-023-01192-x"],
    "International Journal of Finance & Economics": ["10.1002/ijfe.70077", "10.1002/ijfe.3020"],
    "Applied Soft Computing": ["10.1016/j.asoc.2026.115085"],
}


def http_json(url: str, timeout: int = 20) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "paper4-journal-source-audit/1.0 (local reproducibility audit)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_status(url: str, timeout: int = 20) -> dict[str, Any]:
    headers = {"User-Agent": "paper4-journal-source-audit/1.0 (local reproducibility audit)"}
    for method in ["HEAD", "GET"]:
        try:
            req = urllib.request.Request(url, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return {
                    "url": url,
                    "method": method,
                    "status_code": int(response.status),
                    "final_url": response.geturl(),
                    "ok": 200 <= int(response.status) < 400,
                    "error": "",
                }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return {"url": url, "method": "HEAD/GET", "status_code": None, "final_url": "", "ok": False, "error": last_error}


def date_parts_to_text(item: dict[str, Any]) -> str:
    for key in ["published-print", "published-online", "published", "issued"]:
        parts = item.get(key, {}).get("date-parts", [[]])[0]
        if parts:
            return "-".join(str(part) for part in parts)
    return ""


def normalize(text: str) -> str:
    text = clean_crossref_text(text)
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def clean_crossref_text(text: Any) -> str:
    text = html.unescape(str(text))
    text = HTML_TAG_RE.sub("", text)
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_doi_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(row)
    for field in ["title", "container"]:
        cleaned[field] = clean_crossref_text(cleaned.get(field, ""))
    return cleaned


def crossref_entry(label: str, spec: dict[str, str]) -> dict[str, Any]:
    doi = spec["doi"]
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    try:
        item = http_json(url)["message"]
        title = clean_crossref_text(" ".join(item.get("title") or []))
        containers = item.get("container-title") or []
        container = clean_crossref_text(" / ".join(containers))
        expected_container = spec["container"]
        title_ok = normalize(spec["title_contains"]) in normalize(title)
        container_ok = normalize(expected_container) in normalize(container)
        return {
            "label": label,
            "doi": doi,
            "url": f"https://doi.org/{doi}",
            "ok": title_ok and container_ok,
            "title": title,
            "container": container,
            "published": date_parts_to_text(item),
            "title_ok": title_ok,
            "container_ok": container_ok,
            "error": "",
        }
    except Exception as exc:
        return {
            "label": label,
            "doi": doi,
            "url": f"https://doi.org/{doi}",
            "ok": False,
            "title": "",
            "container": "",
            "published": "",
            "title_ok": False,
            "container_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def refresh_cache() -> dict[str, Any]:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    checked_at = datetime.now(timezone.utc).isoformat()
    data = {
        "checked_at": checked_at,
        "policy_urls": {label: http_status(url) for label, url in POLICY_URLS.items()},
        "doi_entries": {label: crossref_entry(label, spec) for label, spec in DOIS.items()},
    }
    CACHE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def load_cache() -> dict[str, Any]:
    if not CACHE.exists():
        return {"checked_at": "", "policy_urls": {}, "doi_entries": {}}
    return json.loads(CACHE.read_text(encoding="utf-8"))


def cache_age_days(checked_at: str) -> float | None:
    if not checked_at:
        return None
    try:
        stamp = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc).date() - stamp.date()).days
    except ValueError:
        return None


def manuscript_presence(text: str) -> dict[str, Any]:
    policy = [
        {"label": label, "url": url, "present": url in text}
        for label, url in POLICY_URLS.items()
    ]
    dois = [
        {"label": label, "doi": spec["doi"], "present": spec["doi"] in text and f"https://doi.org/{spec['doi']}" in text}
        for label, spec in DOIS.items()
    ]
    return {"policy_urls": policy, "doi_entries": dois}


def candidate_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        cells = [cell.strip().strip("*") for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        try:
            rank = int(cells[0])
        except ValueError:
            continue
        dois = re.findall(r"10\.\d{4,9}/[^\s\]\[;,.]+(?:\.[^\s\]\[;,.]+)*", line)
        rows.append({"rank": rank, "journal": cells[1], "dois": dois, "line": line})
    return rows


def published_date(row: dict[str, Any]) -> date | None:
    parts = [int(part) for part in re.findall(r"\d+", str(row.get("published", "")))]
    if not parts:
        return None
    year = parts[0]
    month = parts[1] if len(parts) > 1 else 1
    day = parts[2] if len(parts) > 2 else 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


def candidate_doi_evidence(candidates: list[dict[str, Any]], doi_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    doi_lookup = {row.get("doi"): row for row in doi_rows}
    evidence = []
    for row in candidates:
        journal = row["journal"]
        expected = EXPECTED_CANDIDATE_DOIS.get(journal, [])
        row_dois = row.get("dois", [])
        matched = [doi for doi in row_dois if doi in expected]
        matched_cache = [doi_lookup.get(doi, {}) for doi in matched]
        bad_matched = [
            doi
            for doi, cache_row in zip(matched, matched_cache)
            if not cache_row.get("ok") or (published_date(cache_row) or date.min) < MIN_RELATED_WORK_DATE
        ]
        evidence.append(
            {
                "rank": row["rank"],
                "journal": journal,
                "row_dois": row_dois,
                "expected_dois": expected,
                "matched_expected_dois": matched,
                "missing_expected_dois": [doi for doi in expected if doi not in row_dois],
                "bad_matched_dois": bad_matched,
                "ok": bool(matched) and not bad_matched,
            }
        )
    return evidence


def candidate_policy_evidence(
    candidates: list[dict[str, Any]],
    manuscript_text: str,
    policy_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    policy_ok = {row.get("label"): bool(row.get("ok")) for row in policy_rows}
    evidence = []
    for row in candidates:
        journal = row["journal"]
        expected_group = EXPECTED_CANDIDATE_POLICY_GROUP.get(journal, "")
        group = POLICY_FOOTNOTE_GROUPS.get(expected_group, {})
        policy_labels = group.get("policy_labels", [])
        footnote = group.get("footnote", "")
        missing_policy_labels = [
            label
            for label in policy_labels
            if POLICY_URLS.get(label, "") not in manuscript_text
        ]
        failed_policy_labels = [
            label
            for label in policy_labels
            if not policy_ok.get(label, False)
        ]
        evidence.append(
            {
                "rank": row["rank"],
                "journal": journal,
                "expected_policy_group": expected_group,
                "expected_footnote": footnote,
                "matched_footnote": bool(footnote and footnote in row.get("line", "")),
                "policy_labels": policy_labels,
                "missing_policy_labels": missing_policy_labels,
                "failed_policy_labels": failed_policy_labels,
                "ok": bool(expected_group)
                and bool(policy_labels)
                and bool(footnote)
                and footnote in row.get("line", "")
                and not missing_policy_labels
                and not failed_policy_labels,
            }
        )
    return evidence


def build_audit(cache: dict[str, Any]) -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manuscript_text = MANUSCRIPT.read_text(encoding="utf-8") if MANUSCRIPT.exists() else ""
    presence = manuscript_presence(manuscript_text)
    age = cache_age_days(cache.get("checked_at", ""))
    cache_fresh = age is not None and age <= MAX_CACHE_AGE_DAYS
    policy_rows = [
        {"label": label, **row}
        for label, row in cache.get("policy_urls", {}).items()
    ]
    doi_rows = [clean_doi_row(row) for row in cache.get("doi_entries", {}).values()]
    missing_policy = [row for row in presence["policy_urls"] if not row["present"]]
    missing_doi = [row for row in presence["doi_entries"] if not row["present"]]
    failed_policy = [row for row in policy_rows if not row.get("ok")]
    failed_doi = [row for row in doi_rows if not row.get("ok")]
    candidates = candidate_rows(manuscript_text)
    candidate_journals = [row["journal"] for row in candidates]
    missing_candidate_journals = [
        journal for journal in EXPECTED_CANDIDATE_JOURNALS if journal not in candidate_journals
    ]
    unexpected_candidate_journals = [
        journal for journal in candidate_journals if journal not in EXPECTED_CANDIDATE_JOURNALS
    ]
    candidate_ranks = [row["rank"] for row in candidates]
    expected_ranks = list(range(1, EXPECTED_CANDIDATE_JOURNAL_COUNT + 1))
    candidate_count_ok = len(candidates) == EXPECTED_CANDIDATE_JOURNAL_COUNT
    candidate_ranks_ok = candidate_ranks == expected_ranks
    candidate_journals_ok = not missing_candidate_journals and not unexpected_candidate_journals
    candidate_evidence = candidate_doi_evidence(candidates, doi_rows)
    candidate_doi_evidence_ok = (
        len(candidate_evidence) == EXPECTED_CANDIDATE_JOURNAL_COUNT
        and all(row["ok"] for row in candidate_evidence)
    )
    candidate_policy = candidate_policy_evidence(candidates, manuscript_text, policy_rows)
    candidate_policy_evidence_ok = (
        len(candidate_policy) == EXPECTED_CANDIDATE_JOURNAL_COUNT
        and all(row["ok"] for row in candidate_policy)
    )
    status = "pass"
    if (
        not MANUSCRIPT.exists()
        or not cache_fresh
        or missing_policy
        or missing_doi
        or failed_policy
        or failed_doi
        or not candidate_count_ok
        or not candidate_ranks_ok
        or not candidate_journals_ok
        or not candidate_doi_evidence_ok
        or not candidate_policy_evidence_ok
    ):
        status = "fail"
    return {
        "status": status,
        "checked_at": cache.get("checked_at", ""),
        "cache_path": str(CACHE),
        "manuscript": str(MANUSCRIPT),
        "cache_age_days": age,
        "max_cache_age_days": MAX_CACHE_AGE_DAYS,
        "cache_fresh": cache_fresh,
        "policy_url_count": len(POLICY_URLS),
        "doi_count": len(DOIS),
        "missing_policy_url_count": len(missing_policy),
        "missing_doi_count": len(missing_doi),
        "failed_policy_url_count": len(failed_policy),
        "failed_doi_count": len(failed_doi),
        "expected_candidate_journal_count": EXPECTED_CANDIDATE_JOURNAL_COUNT,
        "candidate_journal_count": len(candidates),
        "candidate_count_ok": candidate_count_ok,
        "candidate_ranks": candidate_ranks,
        "candidate_ranks_ok": candidate_ranks_ok,
        "candidate_journals": candidate_journals,
        "missing_candidate_journals": missing_candidate_journals,
        "unexpected_candidate_journals": unexpected_candidate_journals,
        "candidate_journals_ok": candidate_journals_ok,
        "min_related_work_date": MIN_RELATED_WORK_DATE.isoformat(),
        "candidate_doi_evidence_ok": candidate_doi_evidence_ok,
        "candidate_doi_evidence": candidate_evidence,
        "candidate_policy_evidence_count": len(candidate_policy),
        "candidate_policy_evidence_ok": candidate_policy_evidence_ok,
        "candidate_policy_evidence": candidate_policy,
        "manuscript_presence": presence,
        "policy_urls": policy_rows,
        "doi_entries": doi_rows,
        "guardrail": "The Korean submission-candidate journal table must carry exactly 10 ranked candidate journals, at least one expected 2025-07-01-or-later related-work DOI per candidate row, plus candidate-row policy-footnote evidence mapped to currently verified official publisher policy URLs; default pipeline validation is offline against this refreshed cache.",
    }


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Journal Candidate Source Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Checked at | `{audit['checked_at']}` |",
        f"| Cache age days | {audit['cache_age_days']:.2f} |" if audit["cache_age_days"] is not None else "| Cache age days | missing |",
        f"| Max cache age days | {audit['max_cache_age_days']} |",
        f"| Policy URLs | {audit['policy_url_count']} |",
        f"| DOI entries | {audit['doi_count']} |",
        f"| Candidate journal rows | {audit['candidate_journal_count']} / {audit['expected_candidate_journal_count']} |",
        f"| Candidate ranks ok | {audit['candidate_ranks_ok']} |",
        f"| Candidate journals ok | {audit['candidate_journals_ok']} |",
        f"| Candidate DOI evidence ok | {audit['candidate_doi_evidence_ok']} |",
        f"| Candidate policy evidence rows | {audit['candidate_policy_evidence_count']} / {audit['expected_candidate_journal_count']} |",
        f"| Candidate policy evidence ok | {audit['candidate_policy_evidence_ok']} |",
        f"| Minimum related-work date | {audit['min_related_work_date']} |",
        f"| Missing policy URLs in manuscript | {audit['missing_policy_url_count']} |",
        f"| Missing DOI entries in manuscript | {audit['missing_doi_count']} |",
        f"| Failed policy URL checks | {audit['failed_policy_url_count']} |",
        f"| Failed DOI checks | {audit['failed_doi_count']} |",
        "",
        "## Policy URLs",
        "",
        "| Label | Status | URL | Final URL | Error |",
        "|---|---:|---|---|---|",
    ]
    for row in audit["policy_urls"]:
        lines.append(
            f"| `{escape_md(row.get('label', ''))}` | {row.get('status_code', '')} | "
            f"{escape_md(row.get('url', ''))} | {escape_md(row.get('final_url', ''))} | {escape_md(row.get('error', '')) or '-'} |"
        )
    lines.extend(["", "## Candidate Journals", "", "| Rank | Journal |", "|---:|---|"])
    for rank, journal in zip(audit["candidate_ranks"], audit["candidate_journals"]):
        lines.append(f"| {rank} | {escape_md(journal)} |")
    lines.extend(
        [
            "",
            "## Candidate DOI Evidence",
            "",
            "| Rank | Journal | Matched expected DOI evidence | Status |",
            "|---:|---|---|---:|",
        ]
    )
    for row in audit["candidate_doi_evidence"]:
        lines.append(
            f"| {row['rank']} | {escape_md(row['journal'])} | "
            f"{escape_md('; '.join(row['matched_expected_dois']))} | {row['ok']} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Policy Evidence",
            "",
            "| Rank | Journal | Policy group | Policy labels | Footnote matched | Status |",
            "|---:|---|---|---|---:|---:|",
        ]
    )
    for row in audit["candidate_policy_evidence"]:
        lines.append(
            f"| {row['rank']} | {escape_md(row['journal'])} | {escape_md(row['expected_policy_group'])} | "
            f"{escape_md('; '.join(row['policy_labels']))} | {row['matched_footnote']} | {row['ok']} |"
        )
    lines.extend(["", "## DOI Entries", "", "| DOI | Status | Container | Published | Title | Error |", "|---|---:|---|---|---|---|"])
    for row in audit["doi_entries"]:
        lines.append(
            f"| `{escape_md(row.get('doi', ''))}` | {row.get('ok', False)} | "
            f"{escape_md(row.get('container', ''))} | {escape_md(row.get('published', ''))} | "
            f"{escape_md(row.get('title', ''))} | {escape_md(row.get('error', '')) or '-'} |"
        )
    lines.extend(["", "## Guardrail", "", audit["guardrail"]])
    return "\n".join(lines) + "\n"


def write_markdown(audit: dict[str, Any]) -> None:
    OUT_MD.write_text(render_markdown(audit), encoding="utf-8")


def check_existing_outputs(audit: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {
        "json": (OUT_JSON, json.dumps(audit, indent=2, ensure_ascii=False) + "\n"),
        "markdown": (OUT_MD, render_markdown(audit)),
    }
    checks = []
    for label, (path, expected_text) in expected.items():
        actual = path.read_text(encoding="utf-8") if path.exists() else None
        checks.append(
            {
                "output": label,
                "path": path.as_posix(),
                "exists": path.exists(),
                "matches_recomputed": actual == expected_text,
            }
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Refresh policy URL and DOI evidence over the network.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Recompute from the cached source evidence and compare existing outputs without rewriting them.",
    )
    args = parser.parse_args()
    cache = refresh_cache() if args.refresh else load_cache()
    audit = build_audit(cache)
    if args.check_only:
        checks = check_existing_outputs(audit)
        ok = all(row["exists"] and row["matches_recomputed"] for row in checks)
        print(
            json.dumps(
                {
                    "status": "pass" if ok else "fail",
                    "mode": "check_only",
                    "candidate_journal_count": audit["candidate_journal_count"],
                    "candidate_doi_evidence_ok": audit["candidate_doi_evidence_ok"],
                    "candidate_policy_evidence_ok": audit["candidate_policy_evidence_ok"],
                    "output_integrity": checks,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(0 if ok else 1)
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "candidate_journal_count": audit["candidate_journal_count"],
                "candidate_doi_evidence_ok": audit["candidate_doi_evidence_ok"],
                "candidate_policy_evidence_ok": audit["candidate_policy_evidence_ok"],
                "doi_count": audit["doi_count"],
                "policy_url_count": audit["policy_url_count"],
            },
            indent=2,
        )
    )
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
