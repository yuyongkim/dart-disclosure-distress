from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
DISTRESS_TEXT_ITEMS = ("1.03", "2.04", "3.01")


def json_read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_state(path: Path, payload: dict[str, Any]) -> str:
    if payload.get("status"):
        return str(payload["status"])
    if path.exists():
        return "available_no_status_field"
    return "missing"


def split_items(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()}


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}%"


def build_scope(
    filings_path: Path,
    manifest_path: Path,
    all8k_plan_path: Path,
    all8k_sample_path: Path,
    all8k_sample_collection_path: Path,
    all8k_sample_baseline_path: Path,
    population_status_path: Path,
    sampling_status_path: Path,
    sequence_validation_path: Path,
    firm_disjoint_path: Path,
    financial_baseline_path: Path,
    text_metadata_path: Path,
    filing_body_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    filings = pd.read_csv(filings_path, low_memory=False)
    manifest = pd.read_csv(manifest_path, low_memory=False)

    item_sets = filings["items"].map(split_items)
    target_mask = item_sets.map(lambda values: bool(values.intersection(DISTRESS_TEXT_ITEMS)))
    target_filings = filings[target_mask].copy()

    manifest_unique = int(manifest["accession_number"].nunique()) if "accession_number" in manifest else 0
    target_unique = int(target_filings["accession_number"].nunique()) if "accession_number" in target_filings else 0
    manifest_local = int(manifest["local_text_exists"].sum()) if "local_text_exists" in manifest else 0
    population_status = json_read(population_status_path)
    all8k_plan = json_read(all8k_plan_path)
    all8k_sample = json_read(all8k_sample_path)
    all8k_sample_collection = json_read(all8k_sample_collection_path)
    all8k_sample_baseline = json_read(all8k_sample_baseline_path)
    sample_rf_metric = next(
        (
            row
            for row in all8k_sample_baseline.get("metrics", [])
            if str(row.get("model")) == "random_forest"
        ),
        {},
    )
    sampling_status = json_read(sampling_status_path)
    sequence_validation = json_read(sequence_validation_path)
    firm_disjoint = json_read(firm_disjoint_path)
    financial_baseline = json_read(financial_baseline_path)
    text_metadata = json_read(text_metadata_path)
    filing_body = json_read(filing_body_path)

    all_8k_rows = int(len(filings))
    target_rows = int(len(target_filings))
    manifest_rows = int(len(manifest))
    planned_missing = all8k_plan.get("missing_text_files")
    all_8k_text_missing = int(planned_missing) if planned_missing is not None else max(0, all_8k_rows - manifest_rows)
    all_8k_text_cached = max(0, all_8k_rows - all_8k_text_missing)
    non_target_text_missing = max(0, all_8k_text_missing)

    scope_summary = {
        "status": "created",
        "filings_path": str(filings_path),
        "manifest_path": str(manifest_path),
        "all_8k_filing_rows": all_8k_rows,
        "firms": int(filings["cik"].nunique()) if "cik" in filings else population_status.get("filing_firms"),
        "target_text_items": list(DISTRESS_TEXT_ITEMS),
        "target_item_filing_rows": target_rows,
        "target_item_unique_accessions": target_unique,
        "manifest_rows": manifest_rows,
        "manifest_unique_accessions": manifest_unique,
        "local_text_files": manifest_local,
        "target_text_accession_coverage_rate": ratio(manifest_unique, target_unique),
        "target_text_file_coverage_rate": ratio(manifest_local, manifest_rows),
        "all_8k_text_cached_rows": all_8k_text_cached,
        "all_8k_text_scope_coverage_rate": ratio(all_8k_text_cached, all_8k_rows),
        "all_8k_text_missing_rows": all_8k_text_missing,
        "non_target_text_missing_rows": non_target_text_missing,
        "all8k_text_plan": {
            "status": evidence_state(all8k_plan_path, all8k_plan),
            "manifest_rows": all8k_plan.get("manifest_rows"),
            "missing_text_files": all8k_plan.get("missing_text_files"),
            "planned_batches": all8k_plan.get("planned_batches"),
            "estimated_missing_text_gb": all8k_plan.get("estimated_missing_text_gb"),
        },
        "all8k_text_sample_design": {
            "status": evidence_state(all8k_sample_path, all8k_sample),
            "sample_rows": all8k_sample.get("sample_rows"),
            "non_target_sample_rows": all8k_sample.get("non_target_sample_rows"),
            "target_cached_reference_rows": all8k_sample.get("target_cached_reference_rows"),
            "sample_strata_with_rows": all8k_sample.get("sample_strata_with_rows"),
            "population_strata": all8k_sample.get("population_strata"),
            "estimated_planned_missing_sample_mb": all8k_sample.get("estimated_planned_missing_sample_mb"),
            "path_resolution_mode": all8k_sample.get("path_resolution_mode"),
            "manifest_local_path_redacted_rows": all8k_sample.get("manifest_local_path_redacted_rows"),
            "canonical_text_path_rows": all8k_sample.get("canonical_text_path_rows"),
        },
        "all8k_text_sample_collection": {
            "status": evidence_state(all8k_sample_collection_path, all8k_sample_collection),
            "sample_rows": all8k_sample_collection.get("sample_rows"),
            "cached_text_files": all8k_sample_collection.get("cached_text_files"),
            "missing_text_files": all8k_sample_collection.get("missing_text_files"),
            "download_attempts": all8k_sample_collection.get("download_attempts"),
            "downloaded_files": all8k_sample_collection.get("downloaded_files"),
            "user_agent_present": all8k_sample_collection.get("user_agent_present"),
            "path_resolution_mode": all8k_sample_collection.get("path_resolution_mode"),
            "sample_local_path_redacted_rows": all8k_sample_collection.get("sample_local_path_redacted_rows"),
            "canonical_text_path_rows": all8k_sample_collection.get("canonical_text_path_rows"),
        },
        "all8k_text_sample_baseline": {
            "status": evidence_state(all8k_sample_baseline_path, all8k_sample_baseline),
            "cached_text_rows_used": all8k_sample_baseline.get("cached_text_rows_used"),
            "missing_sample_text_rows": all8k_sample_baseline.get("missing_sample_text_rows"),
            "non_target_rows": all8k_sample_baseline.get("non_target_rows"),
            "target_reference_rows": all8k_sample_baseline.get("target_reference_rows"),
            "feature_count": all8k_sample_baseline.get("feature_count"),
            "path_resolution_mode": all8k_sample_baseline.get("path_resolution_mode"),
            "sample_local_path_redacted_rows": all8k_sample_baseline.get("sample_local_path_redacted_rows"),
            "canonical_text_path_rows": all8k_sample_baseline.get("canonical_text_path_rows"),
            "random_forest_roc_auc": sample_rf_metric.get("roc_auc"),
            "random_forest_top_10pct_lift": sample_rf_metric.get("top_10pct_lift"),
        },
        "sampling_frame": {
            "validation_firms": sampling_status.get("validation_firms"),
            "anchor_rows": sampling_status.get("anchor_rows"),
            "test_anchor_rows": sampling_status.get("test_anchor_rows"),
            "item_103_firms": sampling_status.get("item_103_firms"),
        },
        "evidence_status": {
            "population_scan": evidence_state(population_status_path, population_status),
            "sampling_frame": evidence_state(sampling_status_path, sampling_status),
            "sequence_validation": evidence_state(sequence_validation_path, sequence_validation),
            "firm_disjoint_robustness": evidence_state(firm_disjoint_path, firm_disjoint),
            "financial_baseline": evidence_state(financial_baseline_path, financial_baseline),
            "metadata_text_baseline": evidence_state(text_metadata_path, text_metadata),
            "filing_body_baseline": evidence_state(filing_body_path, filing_body),
            "all8k_sample_text_baseline": evidence_state(all8k_sample_baseline_path, all8k_sample_baseline),
        },
        "allowed_claim": "SEC v2 supports population-style feasibility evidence from cached submissions metadata, formal local sampling, sequence validation, financial baselines, metadata-text baselines, and complete target-item filing-body cache for Items 1.03/2.04/3.01.",
        "disallowed_claim": "Do not claim finalized standalone U.S. validation or all-8-K filing-body text validation from the current evidence.",
        "blocking_reason": "The filing-body text cache is complete for target distress items 1.03/2.04/3.01, and the bounded non-target all-8-K sample text has now been downloaded and modeled as a sample-level target-item-vs-non-target sensitivity check. This still does not cover the full all-8-K filing-body population, and the sample task is not standalone future-distress prediction.",
    }

    decision_rows = [
        {
            "claim_level": "v1 manuscript",
            "claim": "DART-first SupTech early-warning model with SEC taxonomy-transfer support",
            "status": "allowed",
            "evidence": "DART model results plus SEC taxonomy/alignment infrastructure",
            "guardrail": "Do not make SEC the empirical headline in v1.",
        },
        {
            "claim_level": "SEC v2 feasibility",
            "claim": "SEC cached-metadata sequence validation with financial and text comparators",
            "status": "allowed_with_limitations",
            "evidence": f"{sampling_status.get('validation_firms')} validation firms; {sampling_status.get('anchor_rows')} task-anchor rows; sequence, firm-disjoint, financial, metadata-text, and target-item filing-body baselines exist",
            "guardrail": "Report as v2 feasibility evidence, not final U.S. validation.",
        },
        {
            "claim_level": "SEC target-item filing-body text",
            "claim": "Target-item filing-body cache for Items 1.03/2.04/3.01 is complete",
            "status": "allowed",
            "evidence": f"{manifest_rows} manifest rows; {manifest_local} local text files; accession coverage {pct(scope_summary['target_text_accession_coverage_rate'])}",
            "guardrail": "Limit wording to target-item filing bodies.",
        },
        {
            "claim_level": "SEC non-target all-8-K filing-body sample",
            "claim": "A bounded non-target filing-body text sample has been downloaded and modeled",
            "status": "allowed_sample_sensitivity_only",
            "evidence": f"{all8k_sample.get('non_target_sample_rows')} non-target sample rows plus {all8k_sample.get('target_cached_reference_rows')} target-reference rows across {all8k_sample.get('sample_strata_with_rows')}/{all8k_sample.get('population_strata')} strata; collection status has {all8k_sample_collection.get('cached_text_files')} cached and {all8k_sample_collection.get('missing_text_files')} missing sample files; sample baseline status {all8k_sample_baseline.get('status')} with RF ROC AUC {sample_rf_metric.get('roc_auc')} and top-decile lift {sample_rf_metric.get('top_10pct_lift')}",
            "guardrail": "This supports only bounded sample-level text sensitivity, not full all-8-K text validation or standalone U.S. distress prediction.",
        },
        {
            "claim_level": "SEC all-8-K filing-body text",
            "claim": "All 8-K/8-K/A filing bodies are available for text modeling",
            "status": "not_allowed",
            "evidence": f"{all_8k_text_cached}/{all_8k_rows} filing-body rows currently cached across target-item and bounded-sample files ({pct(scope_summary['all_8k_text_scope_coverage_rate'])}); all-8-K plan has {all8k_plan.get('missing_text_files')} missing files and about {all8k_plan.get('estimated_missing_text_gb')} GB estimated missing text; sample design is not a full cache",
            "guardrail": "Requires a separate full all-8-K text cache before full-scope wording; the current sample model is explicitly sample-limited.",
        },
        {
            "claim_level": "SEC standalone U.S. validation",
            "claim": "Population-representative U.S. distress prediction validation",
            "status": "not_allowed_yet",
            "evidence": f"Current frame is local SEC cached submissions metadata with {sampling_status.get('item_103_firms')} Item 1.03 firms and sparse bankruptcy test positives",
            "guardrail": "Finalize sampling universe, archive coverage, non-target text scope, and bankruptcy-positive adequacy before making this claim.",
        },
    ]
    decision_table = pd.DataFrame(decision_rows)
    decision_csv = out_dir / "sec_validation_scope_decision_table.csv"
    decision_table.to_csv(decision_csv, index=False, encoding="utf-8-sig")
    scope_summary["decision_table_csv"] = str(decision_csv)

    json_path = ROOT / "results" / "sec_validation_scope.json"
    md_path = ROOT / "results" / "sec_validation_scope.md"
    json_path.write_text(json.dumps(scope_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(scope_summary, decision_table, md_path)
    return scope_summary


def write_markdown(summary: dict[str, Any], decisions: pd.DataFrame, path: Path) -> None:
    lines = [
        "# SEC Validation Scope",
        "",
        "This audit defines what the current SEC v2 evidence can and cannot support. It is a claim-control file for manuscript drafting.",
        "",
        "## Coverage",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| All SEC 8-K/8-K/A filing rows | {summary['all_8k_filing_rows']} |",
        f"| SEC firms in filing table | {summary['firms']} |",
        f"| Target filing-body items | {', '.join(summary['target_text_items'])} |",
        f"| Target-item filing rows | {summary['target_item_filing_rows']} |",
        f"| Target-item unique accessions | {summary['target_item_unique_accessions']} |",
        f"| Filing-body manifest rows | {summary['manifest_rows']} |",
        f"| Local filing-body text files | {summary['local_text_files']} |",
        f"| Target-item accession coverage | {pct(summary['target_text_accession_coverage_rate'])} |",
        f"| Target-item file coverage | {pct(summary['target_text_file_coverage_rate'])} |",
        f"| All-8-K filing-body rows currently cached | {summary['all_8k_text_cached_rows']} |",
        f"| All-8-K text-scope coverage | {pct(summary['all_8k_text_scope_coverage_rate'])} |",
        f"| All-8-K filing rows outside current text cache | {summary['all_8k_text_missing_rows']} |",
        f"| Non-target filing rows outside current target-item text cache | {summary['non_target_text_missing_rows']} |",
        f"| All-8-K text plan status | `{summary['all8k_text_plan']['status']}` |",
        f"| All-8-K text plan missing files | {summary['all8k_text_plan']['missing_text_files']} |",
        f"| All-8-K text plan estimated missing GB | {summary['all8k_text_plan']['estimated_missing_text_gb']} |",
        f"| All-8-K text plan batches | {summary['all8k_text_plan']['planned_batches']} |",
        f"| All-8-K sample design status | `{summary['all8k_text_sample_design']['status']}` |",
        f"| All-8-K sample design path mode | `{summary['all8k_text_sample_design']['path_resolution_mode']}` |",
        f"| All-8-K sample design redacted/canonical path rows | {summary['all8k_text_sample_design']['manifest_local_path_redacted_rows']} / {summary['all8k_text_sample_design']['canonical_text_path_rows']} |",
        f"| All-8-K non-target sample rows | {summary['all8k_text_sample_design']['non_target_sample_rows']} |",
        f"| All-8-K target reference rows | {summary['all8k_text_sample_design']['target_cached_reference_rows']} |",
        f"| All-8-K sampled strata | {summary['all8k_text_sample_design']['sample_strata_with_rows']} / {summary['all8k_text_sample_design']['population_strata']} |",
        f"| All-8-K sample estimated missing text MB | {summary['all8k_text_sample_design']['estimated_planned_missing_sample_mb']} |",
        f"| All-8-K sample collection status | `{summary['all8k_text_sample_collection']['status']}` |",
        f"| All-8-K sample collection path mode | `{summary['all8k_text_sample_collection']['path_resolution_mode']}` |",
        f"| All-8-K sample collection redacted/canonical path rows | {summary['all8k_text_sample_collection']['sample_local_path_redacted_rows']} / {summary['all8k_text_sample_collection']['canonical_text_path_rows']} |",
        f"| All-8-K sample cached text files | {summary['all8k_text_sample_collection']['cached_text_files']} |",
        f"| All-8-K sample missing text files | {summary['all8k_text_sample_collection']['missing_text_files']} |",
        f"| All-8-K sample download attempts | {summary['all8k_text_sample_collection']['download_attempts']} |",
        f"| All-8-K sample text baseline status | `{summary['all8k_text_sample_baseline']['status']}` |",
        f"| All-8-K sample baseline path mode | `{summary['all8k_text_sample_baseline']['path_resolution_mode']}` |",
        f"| All-8-K sample baseline redacted/canonical path rows | {summary['all8k_text_sample_baseline']['sample_local_path_redacted_rows']} / {summary['all8k_text_sample_baseline']['canonical_text_path_rows']} |",
        f"| All-8-K sample text rows modeled | {summary['all8k_text_sample_baseline']['cached_text_rows_used']} |",
        f"| All-8-K sample text feature count | {summary['all8k_text_sample_baseline']['feature_count']} |",
        f"| All-8-K sample RF ROC AUC | {summary['all8k_text_sample_baseline']['random_forest_roc_auc']} |",
        f"| All-8-K sample RF top-decile lift | {summary['all8k_text_sample_baseline']['random_forest_top_10pct_lift']} |",
        "",
        "## Decision Table",
        "",
        "| Claim level | Claim | Status | Evidence | Guardrail |",
        "|---|---|---|---|---|",
    ]
    for row in decisions.itertuples(index=False):
        lines.append(
            f"| {escape(row.claim_level)} | {escape(row.claim)} | `{escape(row.status)}` | {escape(row.evidence)} | {escape(row.guardrail)} |"
        )
    lines.extend(
        [
            "",
            "## Manuscript Rule",
            "",
            f"Allowed: {summary['allowed_claim']}",
            "",
            f"Not allowed: {summary['disallowed_claim']}",
            "",
            f"Reason: {summary['blocking_reason']}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filings-path", default=str(ROOT / "data" / "raw" / "sec_desktop_population" / "sec_desktop_population_8k_filings.csv"))
    parser.add_argument("--manifest-path", default=str(ROOT / "data" / "raw" / "sec_filing_text" / "sec_filing_text_manifest.csv"))
    parser.add_argument("--all8k-plan-path", default=str(ROOT / "results" / "sec_all8k_text_scope_plan.json"))
    parser.add_argument("--all8k-sample-path", default=str(ROOT / "results" / "sec_all8k_text_sample_design.json"))
    parser.add_argument("--all8k-sample-collection-path", default=str(ROOT / "results" / "sec_all8k_text_sample_collection_status.json"))
    parser.add_argument("--all8k-sample-baseline-path", default=str(ROOT / "results" / "sec_all8k_sample_text_baseline_summary.json"))
    parser.add_argument("--population-status-path", default=str(ROOT / "results" / "sec_desktop_population_collection_status.json"))
    parser.add_argument("--sampling-status-path", default=str(ROOT / "results" / "sec_desktop_sampling_frame_summary.json"))
    parser.add_argument("--sequence-validation-path", default=str(ROOT / "results" / "sec_desktop_validation_summary.json"))
    parser.add_argument("--firm-disjoint-path", default=str(ROOT / "results" / "sec_desktop_validation_firm_disjoint_summary.json"))
    parser.add_argument("--financial-baseline-path", default=str(ROOT / "results" / "sec_desktop_financial_baseline_summary.json"))
    parser.add_argument("--text-metadata-path", default=str(ROOT / "results" / "sec_desktop_text_metadata_baseline_summary.json"))
    parser.add_argument("--filing-body-path", default=str(ROOT / "results" / "sec_filing_body_text_baseline_summary.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "model_sec_desktop"))
    args = parser.parse_args()
    summary = build_scope(
        filings_path=Path(args.filings_path),
        manifest_path=Path(args.manifest_path),
        all8k_plan_path=Path(args.all8k_plan_path),
        all8k_sample_path=Path(args.all8k_sample_path),
        all8k_sample_collection_path=Path(args.all8k_sample_collection_path),
        all8k_sample_baseline_path=Path(args.all8k_sample_baseline_path),
        population_status_path=Path(args.population_status_path),
        sampling_status_path=Path(args.sampling_status_path),
        sequence_validation_path=Path(args.sequence_validation_path),
        firm_disjoint_path=Path(args.firm_disjoint_path),
        financial_baseline_path=Path(args.financial_baseline_path),
        text_metadata_path=Path(args.text_metadata_path),
        filing_body_path=Path(args.filing_body_path),
        out_dir=Path(args.out_dir),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
