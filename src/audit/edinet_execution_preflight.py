from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_COLLECTION_STATUS = DEFAULT_RESULTS_DIR / "edinet_collection_status.json"
DEFAULT_FEASIBILITY = DEFAULT_RESULTS_DIR / "edinet_api_feasibility.json"
DEFAULT_LOCAL_DISCOVERY = DEFAULT_RESULTS_DIR / "edinet_local_discovery.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_preflight(
    collection_status_path: Path,
    feasibility_path: Path,
    local_discovery_path: Path,
) -> dict[str, Any]:
    collection = read_json(collection_status_path)
    feasibility = read_json(feasibility_path)
    local_discovery = read_json(local_discovery_path)

    env_api_key_present = bool(os.environ.get("EDINET_API_KEY", "").strip())
    document_rows = int(collection.get("document_rows") or 0)
    current_report_rows = int(collection.get("current_report_candidate_rows") or 0)
    api_window_infeasible_days = int(feasibility.get("api_window_infeasible_days") or 0)
    external_reusable_candidates = int(local_discovery.get("external_reusable_candidate_rows") or 0)
    generated_artifacts = int(local_discovery.get("generated_paper4_edinet_artifact_rows") or 0)

    blockers: list[str] = []
    if not env_api_key_present:
        blockers.append("missing_env_edinet_api_key")
    if api_window_infeasible_days:
        blockers.append("archive_or_prior_collection_required")
    if document_rows == 0:
        blockers.append("no_edinet_document_rows_collected")
    if current_report_rows == 0:
        blockers.append("no_current_report_candidates_collected")
    if external_reusable_candidates == 0:
        blockers.append("no_external_reusable_edinet_dataset_found")

    empirical_ready = (
        env_api_key_present
        and document_rows > 0
        and current_report_rows > 0
        and api_window_infeasible_days == 0
    )
    if empirical_ready:
        status = "ready_for_edinet_normalization"
    elif not env_api_key_present and api_window_infeasible_days:
        status = "blocked_key_and_archive_gap"
    elif not env_api_key_present:
        status = "blocked_missing_api_key"
    elif api_window_infeasible_days:
        status = "blocked_archive_gap"
    else:
        status = "blocked_no_empirical_rows"

    return {
        "status": status,
        "env_api_key_present": env_api_key_present,
        "collection_status": collection.get("status"),
        "collection_execute": collection.get("execute"),
        "collection_api_key_present": collection.get("api_key_present"),
        "document_rows": document_rows,
        "current_report_candidate_rows": current_report_rows,
        "raw_json_files": collection.get("raw_json_files"),
        "planned_days": feasibility.get("planned_days"),
        "api_window_feasible_days": feasibility.get("api_window_feasible_days"),
        "api_window_infeasible_days": api_window_infeasible_days,
        "as_of_date": feasibility.get("as_of_date"),
        "no_key_preflight_status": (feasibility.get("preflight") or {}).get("status"),
        "no_key_body_status_code": (feasibility.get("preflight") or {}).get("body_status_code"),
        "local_discovery_status": local_discovery.get("status"),
        "external_reusable_candidate_rows": external_reusable_candidates,
        "external_japanese_report_candidate_rows": local_discovery.get("external_japanese_report_candidate_rows"),
        "external_japanese_report_edinet_mentions": local_discovery.get("external_japanese_report_edinet_mentions"),
        "generated_paper4_edinet_artifact_rows": generated_artifacts,
        "empirical_collection_ready": empirical_ready,
        "blockers": blockers,
        "guardrail": (
            "This preflight performs no EDINET requests. It prevents treating EDINET as empirical evidence unless "
            "document-list metadata, current-report candidates, API-key execution, and archive coverage are all present."
        ),
    }


def write_markdown(preflight: dict[str, Any], path: Path) -> None:
    lines = [
        "# EDINET Execution Preflight",
        "",
        "This audit checks whether EDINET can be used as empirical Paper 4 evidence. It performs no EDINET requests.",
        "",
        "## Status",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Status | `{preflight['status']}` |",
        f"| Empirical collection ready | {preflight['empirical_collection_ready']} |",
        f"| Environment EDINET API key present | {preflight['env_api_key_present']} |",
        f"| Collection status | `{preflight['collection_status']}` |",
        f"| Collection execute | {preflight['collection_execute']} |",
        f"| Collection API key present | {preflight['collection_api_key_present']} |",
        f"| Document rows | {preflight['document_rows']} |",
        f"| Current-report candidate rows | {preflight['current_report_candidate_rows']} |",
        f"| Raw JSON files | {preflight['raw_json_files']} |",
        f"| Planned days | {preflight['planned_days']} |",
        f"| API-window feasible days | {preflight['api_window_feasible_days']} |",
        f"| API-window infeasible days | {preflight['api_window_infeasible_days']} |",
        f"| As-of date | {preflight['as_of_date']} |",
        f"| No-key preflight status | `{preflight['no_key_preflight_status']}` |",
        f"| No-key body status code | `{preflight['no_key_body_status_code']}` |",
        f"| External reusable EDINET dataset candidates | {preflight['external_reusable_candidate_rows']} |",
        f"| Local Japanese report candidates | {preflight['external_japanese_report_candidate_rows']} |",
        f"| Local Japanese report EDINET mentions | {preflight['external_japanese_report_edinet_mentions']} |",
        f"| Generated Paper 4 EDINET artifacts | {preflight['generated_paper4_edinet_artifact_rows']} |",
        "",
        "## Blockers",
        "",
    ]
    if preflight["blockers"]:
        for blocker in preflight["blockers"]:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("No EDINET execution preflight blockers.")
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            preflight["guardrail"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-status", default=str(DEFAULT_COLLECTION_STATUS))
    parser.add_argument("--feasibility", default=str(DEFAULT_FEASIBILITY))
    parser.add_argument("--local-discovery", default=str(DEFAULT_LOCAL_DISCOVERY))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    preflight = build_preflight(
        collection_status_path=Path(args.collection_status),
        feasibility_path=Path(args.feasibility),
        local_discovery_path=Path(args.local_discovery),
    )
    (results_dir / "edinet_execution_preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(preflight, results_dir / "edinet_execution_preflight.md")
    print(json.dumps(preflight, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
