from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
EXPECTED_CALCULATION_OUTPUT_INTEGRITY_CHECKS = 12
EXPECTED_SUBMISSION_DOCUMENTS = 25
EXPECTED_WORD_EXPORT_DOCUMENTS = 4
EXPECTED_CANDIDATE_JOURNALS = 10
EXPECTED_CANDIDATE_DOIS = 12
EXPECTED_CANDIDATE_POLICY_URLS = 5
EXPECTED_CANDIDATE_MIN_RELATED_WORK_DATE = "2025-07-01"


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def size(path: str) -> int:
    p = ROOT / path
    return int(p.stat().st_size) if p.exists() else 0


def csv_rows(path: str) -> int | None:
    p = ROOT / path
    if not p.exists():
        return None
    return int(len(pd.read_csv(p, low_memory=False)))


def json_value(path: str, *keys: str):
    p = ROOT / path
    if not p.exists():
        return None
    value = json.loads(p.read_text(encoding="utf-8"))
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def json_doc(path: str):
    p = ROOT / path
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def metric_value(path: str, task: str, horizon: int, model: str, column: str, feature_set: str | None = None):
    p = ROOT / path
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return None
    required = {"task", "horizon_days", "model"}
    if df.empty or not required.issubset(df.columns):
        return None
    match = df[
        df["task"].astype(str).eq(task)
        & df["horizon_days"].astype(int).eq(int(horizon))
        & df["model"].astype(str).eq(model)
    ]
    if feature_set is not None and "feature_set" in df.columns:
        match = match[match["feature_set"].astype(str).eq(feature_set)]
    if match.empty or column not in match:
        return None
    value = match.iloc[0][column]
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return value


def metric_value_any(path: str, task: str, model: str, column: str, feature_set: str | None = None):
    p = ROOT / path
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return None
    required = {"task", "model"}
    if df.empty or not required.issubset(df.columns):
        return None
    match = df[
        df["task"].astype(str).eq(task)
        & df["model"].astype(str).eq(model)
    ]
    if feature_set is not None and "feature_set" in df.columns:
        match = match[match["feature_set"].astype(str).eq(feature_set)]
    if match.empty or column not in match:
        return None
    value = match.iloc[0][column]
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return value


def item(requirement: str, status: str, evidence: list[str], notes: str = "") -> dict:
    return {
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "notes": notes,
    }


def build_audit() -> dict:
    alignment_status = json_value("results/alignment_evaluation.json", "status")
    alignment_labeled = json_value("results/alignment_evaluation.json", "labeled_rows")
    alignment_annotator1_labeled = json_value("results/alignment_evaluation.json", "annotator1_labeled_rows")
    alignment_annotator2_labeled = json_value("results/alignment_evaluation.json", "annotator2_labeled_rows")
    alignment_adjudicated_labeled = json_value("results/alignment_evaluation.json", "adjudicated_labeled_rows")
    alignment_overlap_rows = json_value("results/alignment_evaluation.json", "annotator_overlap_rows")
    alignment_kappa = json_value("results/alignment_evaluation.json", "annotator_cohen_kappa")
    alignment_needs_adjudication = json_value("results/alignment_evaluation.json", "needs_adjudication_rows")
    manual_seed_rows = csv_rows("data/processed/manual_alignment_seed_100.csv")
    alignment_qc_status = json_value("results/alignment_annotation_qc.json", "status")
    alignment_qc_disagreements = json_value("results/alignment_annotation_qc.json", "proposed_ai_disagreements")
    alignment_qc_low_confidence = json_value("results/alignment_annotation_qc.json", "low_or_missing_confidence_rows")
    alignment_qc_first_pass = json_value("results/alignment_annotation_qc.json", "first_pass_rows")
    alignment_diag_status = json_value("results/alignment_candidate_diagnostics.json", "status")
    alignment_diag_codes = json_value("results/alignment_candidate_diagnostics.json", "taxonomy_codes_in_seed") or []
    alignment_diag_high_ambiguity = json_value("results/alignment_candidate_diagnostics.json", "high_ambiguity_rows")
    alignment_diag_other_rows = json_value("results/alignment_candidate_diagnostics.json", "other_event_rows")
    alignment_diag_imbalance_rows = json_value("results/alignment_candidate_diagnostics.json", "event_count_imbalance_rows")
    alignment_diag_mean_confidence = json_value("results/alignment_candidate_diagnostics.json", "mean_ai_confidence")
    alignment_silver_status = json_value("results/alignment_silver_evaluation.json", "status")
    alignment_silver_consensus = json_value("results/alignment_silver_evaluation.json", "silver_consensus_rows")
    alignment_silver_review_needed = json_value("results/alignment_silver_evaluation.json", "review_needed_rows")
    alignment_silver_strict_rate = json_value("results/alignment_silver_evaluation.json", "strict_agreement_rate")
    alignment_silver_lenient_rate = json_value("results/alignment_silver_evaluation.json", "lenient_agreement_rate")
    sec_desktop_seed_status = json_value("results/alignment_sec_desktop_seed_summary.json", "status")
    sec_desktop_seed_rows = json_value("results/alignment_sec_desktop_seed_summary.json", "candidate_rows")
    sec_desktop_seed_t01_rows = json_value("results/alignment_sec_desktop_seed_summary.json", "t01_candidate_rows")
    sec_desktop_seed_match_rows = json_value("results/alignment_sec_desktop_seed_summary.json", "match_rows")
    sec_desktop_seed_negative_rows = json_value("results/alignment_sec_desktop_seed_summary.json", "negative_rows")
    sec_desktop_seed_codes = json_value("results/alignment_sec_desktop_seed_summary.json", "taxonomy_codes_in_match_seed") or []
    sec_desktop_seed_codes_text = ", ".join(sec_desktop_seed_codes) if isinstance(sec_desktop_seed_codes, list) else str(sec_desktop_seed_codes)
    sec_desktop_seed_preserved_rows = json_value("results/alignment_sec_desktop_seed_summary.json", "preservation", "preserved_rows")
    sec_desktop_seed_preserved_cells = json_value("results/alignment_sec_desktop_seed_summary.json", "preservation", "preserved_cells")
    sec_desktop_eval_status = json_value("results/alignment_sec_desktop_evaluation.json", "status")
    sec_desktop_eval_labeled = json_value("results/alignment_sec_desktop_evaluation.json", "labeled_rows")
    sec_desktop_eval_kappa = json_value("results/alignment_sec_desktop_evaluation.json", "annotator_cohen_kappa")
    alignment_label_preflight_status = json_value("results/alignment_label_preflight.json", "status")
    alignment_label_preflight_ready = json_value("results/alignment_label_preflight.json", "final_rq1_metrics_ready")
    alignment_label_preflight_blockers = json_value("results/alignment_label_preflight.json", "blockers") or []
    alignment_single_status = json_value("results/alignment_single_expert_readiness.json", "status")
    alignment_single_labeled = json_value("results/alignment_single_expert_readiness.json", "human_labeled_rows")
    alignment_single_first_pass = json_value("results/alignment_single_expert_readiness.json", "first_pass_rows")
    alignment_single_high_ambiguity = json_value("results/alignment_single_expert_readiness.json", "high_ambiguity_rows")
    alignment_single_workbook_structure_status = json_value(
        "results/alignment_single_expert_readiness.json",
        "workbook_structure",
        "status",
    )
    alignment_single_workbook_structure_checks = json_value(
        "results/alignment_single_expert_readiness.json",
        "workbook_structure",
        "checks",
    ) or []
    alignment_preserved_rows = json_value("results/alignment_annotation_readiness.json", "preservation", "preserved_rows")
    alignment_preserved_cells = json_value("results/alignment_annotation_readiness.json", "preservation", "preserved_cells")
    alignment_workbook_structure_status = json_value(
        "results/alignment_annotation_readiness.json",
        "workbook_structure",
        "status",
    )
    alignment_workbook_structure_checks = json_value(
        "results/alignment_annotation_readiness.json",
        "workbook_structure",
        "checks",
    ) or []
    alignment_diag_codes_text = ", ".join(alignment_diag_codes) if isinstance(alignment_diag_codes, list) else str(alignment_diag_codes)
    metric_audit_status = json_value("results/manuscript_metric_audit.json", "status")
    metric_audit_missing = json_value("results/manuscript_metric_audit.json", "missing_count")
    metric_audit_checks = json_value("results/manuscript_metric_audit.json", "check_count")
    claim_audit_status = json_value("results/manuscript_claim_audit.json", "status")
    claim_audit_blockers = json_value("results/manuscript_claim_audit.json", "blocker_count")
    claim_audit_review = json_value("results/manuscript_claim_audit.json", "review_count")
    claim_audit_scanned = json_value("results/manuscript_claim_audit.json", "scanned_files") or []
    claim_audit_scanned_count = len(claim_audit_scanned) if isinstance(claim_audit_scanned, list) else 0
    bilingual_audit_status = json_value("results/bilingual_manuscript_audit.json", "status")
    bilingual_audit_checks = json_value("results/bilingual_manuscript_audit.json", "check_count")
    bilingual_audit_failures = json_value("results/bilingual_manuscript_audit.json", "failure_count")
    figure_generation_status = json_value("results/figure_generation_audit.json", "status")
    figure_generation_count = json_value("results/figure_generation_audit.json", "figure_count")
    submission_figure_generation_status = json_value("results/submission_figure_generation_audit.json", "status")
    submission_figure_generation_count = json_value("results/submission_figure_generation_audit.json", "figure_count")
    submission_package_status = json_value("results/submission_package_audit.json", "status")
    submission_package_documents = json_value("results/submission_package_audit.json", "document_count")
    submission_package_failures = json_value("results/submission_package_audit.json", "failure_count")
    traceability_status = json_value("results/research_plan_traceability.json", "status")
    traceability_items = json_value("results/research_plan_traceability.json", "trace_item_count")
    traceability_blockers = json_value("results/research_plan_traceability.json", "blocker_count")
    calculation_audit_status = json_value("results/calculation_sheet_audit.json", "status")
    calculation_rows = json_value("results/calculation_sheet_audit.json", "calculation_rows")
    calculation_model_rows = json_value("results/calculation_sheet_audit.json", "model_rows")
    calculation_manuscript_checks = json_value("results/calculation_sheet_audit.json", "manuscript_checks")
    calculation_verification_checks = json_value("results/calculation_sheet_audit.json", "verification_checks")
    calculation_failures = json_value("results/calculation_sheet_audit.json", "failures") or []
    calculation_failure_count = len(calculation_failures) if isinstance(calculation_failures, list) else None
    calculation_output_integrity = json_value("results/calculation_sheet_audit.json", "output_integrity") or []
    calculation_output_integrity_count = (
        len(calculation_output_integrity) if isinstance(calculation_output_integrity, list) else 0
    )
    calculation_output_failures = json_value("results/calculation_sheet_audit.json", "output_integrity_failures") or []
    calculation_output_failure_count = (
        len(calculation_output_failures) if isinstance(calculation_output_failures, list) else None
    )
    reference_audit_status = json_value("results/reference_consistency_audit.json", "status")
    reference_checked = json_value("results/reference_consistency_audit.json", "checked_references")
    reference_failures = json_value("results/reference_consistency_audit.json", "failures") or []
    reference_failure_count = len(reference_failures) if isinstance(reference_failures, list) else None
    word_export_status = json_value("results/word_export_audit.json", "status")
    word_export_documents = json_value("results/word_export_audit.json", "document_count")
    word_export_failures = json_value("results/word_export_audit.json", "failure_count")
    word_export_draft_sync = json_value("results/word_export_audit.json", "draft_sync", "status")
    word_export_claim_blockers = json_value("results/word_export_audit.json", "claim_blocker_count")
    word_export_claim_review = json_value("results/word_export_audit.json", "claim_review_count")
    journal_source_status = json_value("results/journal_candidate_source_audit.json", "status")
    journal_source_doi_count = json_value("results/journal_candidate_source_audit.json", "doi_count")
    journal_source_policy_count = json_value("results/journal_candidate_source_audit.json", "policy_url_count")
    journal_source_failed_doi_count = json_value("results/journal_candidate_source_audit.json", "failed_doi_count")
    journal_source_failed_policy_count = json_value("results/journal_candidate_source_audit.json", "failed_policy_url_count")
    journal_source_candidate_count = json_value("results/journal_candidate_source_audit.json", "candidate_journal_count")
    journal_source_expected_candidate_count = json_value(
        "results/journal_candidate_source_audit.json",
        "expected_candidate_journal_count",
    )
    journal_source_candidate_ranks_ok = json_value("results/journal_candidate_source_audit.json", "candidate_ranks_ok")
    journal_source_candidate_journals_ok = json_value("results/journal_candidate_source_audit.json", "candidate_journals_ok")
    journal_source_candidate_doi_evidence_ok = json_value(
        "results/journal_candidate_source_audit.json",
        "candidate_doi_evidence_ok",
    )
    journal_source_min_related_work_date = json_value("results/journal_candidate_source_audit.json", "min_related_work_date")
    run_report = json_doc("reproducibility_run_report.json")
    run_report_steps = run_report.get("steps", []) if isinstance(run_report, dict) else []
    run_status_counts = run_report.get("status_counts", {}) if isinstance(run_report, dict) else {}
    if not run_status_counts and isinstance(run_report_steps, list):
        run_status_counts = {}
        for row in run_report_steps:
            status = str(row.get("status", "unknown"))
            run_status_counts[status] = run_status_counts.get(status, 0) + 1
    run_execute = run_report.get("execute") if isinstance(run_report, dict) else None
    run_step_count = run_report.get("step_count") or len(run_report_steps)
    run_ok_count = run_report.get("ok_count") or run_status_counts.get("ok", 0)
    run_skipped_heavy_count = run_report.get("skipped_heavy_count") or run_status_counts.get("skipped_heavy", 0)
    run_failed_step_count = run_report.get("failed_step_count")
    if run_failed_step_count is None:
        run_failed_step_count = sum(
            count for status, count in run_status_counts.items() if status not in {"ok", "skipped_heavy"}
        )
    run_output_failure_count = run_report.get("output_failure_count")
    if run_output_failure_count is None:
        run_output_failure_count = sum(len(row.get("output_failures") or []) for row in run_report_steps)
    run_freshness_exempt_output_count = run_report.get("freshness_exempt_output_count")
    if run_freshness_exempt_output_count is None:
        run_freshness_exempt_output_count = sum(
            1
            for step in run_report_steps
            for output in (step.get("outputs") or [])
            if output.get("freshness_exempt")
        )
    t04_discovery_status = json_value("results/t04_local_discovery.json", "status")
    t04_empirical_candidates = json_value("results/t04_local_discovery.json", "empirical_candidate_rows")
    t04_dart_title_matches = json_value("results/t04_local_discovery.json", "dart_title_match_rows")
    dart_gap_status = json_value("results/dart_gap_collection_status.json", "status")
    dart_gap_candidates = json_value("results/dart_gap_collection_status.json", "outcome_candidate_rows")
    dart_gap_firms = json_value("results/dart_gap_collection_status.json", "outcome_candidate_firms")
    sec_expanded_status = json_value("results/sec_expanded_collection_status.json", "status")
    sec_expanded_filings = json_value("results/sec_expanded_collection_status.json", "filing_rows")
    sec_expanded_candidates = json_value("results/sec_expanded_collection_status.json", "outcome_candidate_rows")
    sec_expanded_item_103 = json_value("results/sec_expanded_collection_status.json", "item_counts", "1.03")
    sec_expanded_sequence_events = json_value("results/sec_expanded_sequence_summary.json", "event_rows")
    sec_expanded_sequence_outcomes = json_value("results/sec_expanded_sequence_summary.json", "outcome_rows")
    sec_expanded_sequence_outcome_firms = json_value("results/sec_expanded_sequence_summary.json", "outcome_firms")
    sec_case_item103 = json_value("results/sec_bankruptcy_case_status.json", "item103_rows")
    sec_case_item103_firms = json_value("results/sec_bankruptcy_case_status.json", "item103_firms")
    sec_case_sequence_events = json_value("results/sec_bankruptcy_case_sequence_summary.json", "event_rows")
    sec_case_sequence_outcomes = json_value("results/sec_bankruptcy_case_sequence_summary.json", "outcome_rows")
    sec_desktop_status = json_value("results/sec_desktop_population_collection_status.json", "status")
    sec_desktop_filings = json_value("results/sec_desktop_population_collection_status.json", "filing_rows")
    sec_desktop_firms = json_value("results/sec_desktop_population_collection_status.json", "filing_firms")
    sec_desktop_candidates = json_value("results/sec_desktop_population_collection_status.json", "outcome_candidate_rows")
    sec_desktop_item103 = json_value("results/sec_desktop_population_collection_status.json", "item_counts", "1.03")
    sec_desktop_item204 = json_value("results/sec_desktop_population_collection_status.json", "item_counts", "2.04")
    sec_desktop_item301 = json_value("results/sec_desktop_population_collection_status.json", "item_counts", "3.01")
    sec_archive_overlap_cached = json_value("results/sec_submissions_archive_status.json", "overlap_references_cached")
    sec_archive_overlap_rate = json_value("results/sec_submissions_archive_status.json", "overlap_reference_cache_rate")
    sec_archive_referenced_filing_count = json_value("results/sec_submissions_archive_status.json", "overlap_referenced_filing_count")
    sec_desktop_sequence_events = json_value("results/sec_desktop_population_sequence_summary.json", "event_rows")
    sec_desktop_sequence_outcomes = json_value("results/sec_desktop_population_sequence_summary.json", "outcome_rows")
    sec_desktop_sequence_outcome_firms = json_value("results/sec_desktop_population_sequence_summary.json", "outcome_firms")
    sec_desktop_bankruptcy_firms = csv_rows("results/sec_desktop_population_bankruptcy_sequences.csv")
    sec_validation_panel_rows = json_value("results/sec_desktop_validation_summary.json", "panel_rows")
    sec_validation_firms = json_value("results/sec_desktop_validation_summary.json", "firms")
    sec_all_730_auc = metric_value("results/sec_desktop_validation_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "roc_auc")
    sec_all_730_lift = metric_value("results/sec_desktop_validation_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "top_10pct_lift")
    sec_bank_730_auc = metric_value("results/sec_desktop_validation_metrics.csv", "bankruptcy_item_103", 730, "logistic", "roc_auc")
    sec_bank_730_test_pos = metric_value("results/sec_desktop_validation_metrics.csv", "bankruptcy_item_103", 730, "logistic", "test_positives")
    sec_bank_730_lift = metric_value("results/sec_desktop_validation_metrics.csv", "bankruptcy_item_103", 730, "logistic", "top_10pct_lift")
    sec_fd_all_730_auc = metric_value("results/sec_desktop_validation_firm_disjoint_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "roc_auc")
    sec_fd_all_730_lift = metric_value("results/sec_desktop_validation_firm_disjoint_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "top_10pct_lift")
    sec_fd_bank_730_auc = metric_value("results/sec_desktop_validation_firm_disjoint_metrics.csv", "bankruptcy_item_103", 730, "logistic", "roc_auc")
    sec_fd_bank_730_lift = metric_value("results/sec_desktop_validation_firm_disjoint_metrics.csv", "bankruptcy_item_103", 730, "logistic", "top_10pct_lift")
    sec_frame_firms = json_value("results/sec_desktop_sampling_frame_summary.json", "firm_rows")
    sec_frame_validation_firms = json_value("results/sec_desktop_sampling_frame_summary.json", "validation_firms")
    sec_frame_anchor_rows = json_value("results/sec_desktop_sampling_frame_summary.json", "anchor_rows")
    sec_frame_test_rows = json_value("results/sec_desktop_sampling_frame_summary.json", "test_anchor_rows")
    sec_companyfacts_files = json_value("results/sec_companyfacts_availability.json", "inventory_files")
    sec_companyfacts_gb = json_value("results/sec_companyfacts_availability.json", "inventory_total_gb")
    sec_companyfacts_gb_fmt = round(sec_companyfacts_gb, 2) if isinstance(sec_companyfacts_gb, (int, float)) else sec_companyfacts_gb
    sec_companyfacts_validation_ciks = json_value("results/sec_companyfacts_availability.json", "validation_ciks")
    sec_companyfacts_validation_coverage = json_value("results/sec_companyfacts_availability.json", "validation_file_coverage_rate")
    sec_fin_panel_rows = json_value("results/sec_desktop_financial_baseline_summary.json", "panel_rows")
    sec_fin_facts = json_value("results/sec_desktop_financial_baseline_summary.json", "facts_extraction", "selected_fact_records")
    sec_fin_files = json_value("results/sec_desktop_financial_baseline_summary.json", "facts_extraction", "files_with_selected_facts")
    sec_fin_all_730_auc = metric_value("results/sec_desktop_financial_baseline_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "roc_auc", "sequence_plus_financial")
    sec_fin_all_730_lift = metric_value("results/sec_desktop_financial_baseline_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "top_10pct_lift", "sequence_plus_financial")
    sec_fin_all_730_coverage = metric_value("results/sec_desktop_financial_baseline_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "financial_coverage_rate", "sequence_plus_financial")
    sec_fin_only_all_730_auc = metric_value("results/sec_desktop_financial_baseline_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "roc_auc", "financial_only")
    sec_fin_only_all_730_lift = metric_value("results/sec_desktop_financial_baseline_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "top_10pct_lift", "financial_only")
    sec_fin_bank_730_auc = metric_value("results/sec_desktop_financial_baseline_metrics.csv", "bankruptcy_item_103", 730, "logistic", "roc_auc", "sequence_plus_financial")
    sec_fin_bank_730_lift = metric_value("results/sec_desktop_financial_baseline_metrics.csv", "bankruptcy_item_103", 730, "logistic", "top_10pct_lift", "sequence_plus_financial")
    sec_text_manifest_rows = json_value("results/sec_filing_text_collection_status.json", "manifest_rows")
    sec_text_manifest_local = json_value("results/sec_filing_text_collection_status.json", "local_text_files")
    sec_textmeta_features = json_value("results/sec_desktop_text_metadata_baseline_summary.json", "textmeta_feature_count")
    sec_textmeta_all_730_auc = metric_value("results/sec_desktop_text_metadata_baseline_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "roc_auc", "metadata_text_only")
    sec_textmeta_all_730_lift = metric_value("results/sec_desktop_text_metadata_baseline_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "top_10pct_lift", "metadata_text_only")
    sec_all_730_full_auc = metric_value("results/sec_desktop_text_metadata_baseline_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "roc_auc", "sequence_financial_metadata_text")
    sec_all_730_full_lift = metric_value("results/sec_desktop_text_metadata_baseline_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "top_10pct_lift", "sequence_financial_metadata_text")
    sec_body_text_status = json_value("results/sec_filing_body_text_baseline_summary.json", "status")
    sec_body_text_cached = json_value("results/sec_filing_body_text_baseline_summary.json", "cached_text_files")
    sec_body_text_rows = json_value("results/sec_filing_body_text_baseline_summary.json", "cached_filing_rows_used")
    sec_body_text_features = json_value("results/sec_filing_body_text_baseline_summary.json", "textbody_feature_count")
    sec_body_all_730_auc = metric_value("results/sec_filing_body_text_baseline_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "roc_auc", "filing_body_text_only")
    sec_body_all_730_lift = metric_value("results/sec_filing_body_text_baseline_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "top_10pct_lift", "filing_body_text_only")
    sec_body_combo_730_auc = metric_value("results/sec_filing_body_text_baseline_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "roc_auc", "sequence_financial_filing_body_text")
    sec_body_combo_730_lift = metric_value("results/sec_filing_body_text_baseline_metrics.csv", "all_distress_items_103_204_301", 730, "random_forest", "top_10pct_lift", "sequence_financial_filing_body_text")
    sec_scope_status = json_value("results/sec_validation_scope.json", "status")
    sec_scope_all_8k_rows = json_value("results/sec_validation_scope.json", "all_8k_filing_rows")
    sec_scope_target_rows = json_value("results/sec_validation_scope.json", "target_item_filing_rows")
    sec_scope_target_coverage = json_value("results/sec_validation_scope.json", "target_text_accession_coverage_rate")
    sec_scope_all_text_coverage = json_value("results/sec_validation_scope.json", "all_8k_text_scope_coverage_rate")
    sec_scope_missing_all_8k_text = json_value("results/sec_validation_scope.json", "all_8k_text_missing_rows")
    sec_all8k_plan_status = json_value("results/sec_all8k_text_scope_plan.json", "status")
    sec_all8k_plan_rows = json_value("results/sec_all8k_text_scope_plan.json", "manifest_rows")
    sec_all8k_missing_files = json_value("results/sec_all8k_text_scope_plan.json", "missing_text_files")
    sec_all8k_missing_gb = json_value("results/sec_all8k_text_scope_plan.json", "estimated_missing_text_gb")
    sec_all8k_batches = json_value("results/sec_all8k_text_scope_plan.json", "planned_batches")
    sec_all8k_batch_status = json_value("results/sec_all8k_text_batch_collection_status.json", "status")
    sec_all8k_batch_open = json_value("results/sec_all8k_text_batch_collection_status.json", "open_batches")
    sec_all8k_batch_next = json_value("results/sec_all8k_text_batch_collection_status.json", "next_batch_id")
    sec_all8k_batch_missing = json_value("results/sec_all8k_text_batch_collection_status.json", "missing_text_files")
    sec_all8k_batch_downloads = json_value("results/sec_all8k_text_batch_collection_status.json", "downloaded_files")
    sec_all8k_batch_path_mode = json_value("results/sec_all8k_text_batch_collection_status.json", "path_resolution_mode")
    sec_all8k_batch_redacted_paths = json_value("results/sec_all8k_text_batch_collection_status.json", "manifest_local_path_redacted_rows")
    sec_all8k_batch_canonical_paths = json_value("results/sec_all8k_text_batch_collection_status.json", "canonical_text_path_rows")
    sec_all8k_batch_log_rows = json_value("results/sec_all8k_text_batch_collection_status.json", "download_log_rows")
    sec_all8k_batch_log_downloaded = json_value("results/sec_all8k_text_batch_collection_status.json", "download_log_downloaded_files")
    sec_all8k_batch_log_existing = json_value("results/sec_all8k_text_batch_collection_status.json", "download_log_existing_files")
    sec_all8k_preflight_status = json_value("results/sec_all8k_storage_preflight.json", "status")
    sec_all8k_preflight_required_gb = json_value("results/sec_all8k_storage_preflight.json", "required_free_gb")
    sec_all8k_preflight_storage_ready = json_value(
        "results/sec_all8k_storage_preflight.json",
        "storage_ready_for_full_scope",
    )
    sec_all8k_preflight_env_user_agent = json_value("results/sec_all8k_storage_preflight.json", "env_user_agent_present")
    sec_all8k_preflight_ready = json_value("results/sec_all8k_storage_preflight.json", "full_scope_download_ready")
    sec_all8k_preflight_blockers = json_value("results/sec_all8k_storage_preflight.json", "blockers") or []
    sec_all8k_sample_status = json_value("results/sec_all8k_text_sample_design.json", "status")
    sec_all8k_sample_rows = json_value("results/sec_all8k_text_sample_design.json", "sample_rows")
    sec_all8k_sample_non_target = json_value("results/sec_all8k_text_sample_design.json", "non_target_sample_rows")
    sec_all8k_sample_target_reference = json_value("results/sec_all8k_text_sample_design.json", "target_cached_reference_rows")
    sec_all8k_sample_strata = json_value("results/sec_all8k_text_sample_design.json", "sample_strata_with_rows")
    sec_all8k_sample_population_strata = json_value("results/sec_all8k_text_sample_design.json", "population_strata")
    sec_all8k_sample_mb = json_value("results/sec_all8k_text_sample_design.json", "estimated_planned_missing_sample_mb")
    sec_sample_collection_status = json_value("results/sec_all8k_text_sample_collection_status.json", "status")
    sec_sample_collection_cached = json_value("results/sec_all8k_text_sample_collection_status.json", "cached_text_files")
    sec_sample_collection_missing = json_value("results/sec_all8k_text_sample_collection_status.json", "missing_text_files")
    sec_sample_collection_attempts = json_value("results/sec_all8k_text_sample_collection_status.json", "download_attempts")
    sec_sample_text_status = json_value("results/sec_all8k_sample_text_baseline_summary.json", "status")
    sec_sample_text_rows = json_value("results/sec_all8k_sample_text_baseline_summary.json", "cached_text_rows_used")
    sec_sample_text_features = json_value("results/sec_all8k_sample_text_baseline_summary.json", "feature_count")
    sec_sample_text_rf_auc = metric_value_any(
        "results/sec_all8k_sample_text_baseline_metrics.csv",
        "target_item_vs_non_target_sample",
        "random_forest",
        "roc_auc",
        "sample_filing_body_keyword_counts",
    )
    sec_sample_text_rf_lift = metric_value_any(
        "results/sec_all8k_sample_text_baseline_metrics.csv",
        "target_item_vs_non_target_sample",
        "random_forest",
        "top_10pct_lift",
        "sample_filing_body_keyword_counts",
    )
    market_collect_target = json_value("data/raw/market_coverage/naver_ohlcv_supplement_summary.json", "target_symbols")
    market_collect_fetched = json_value("data/raw/market_coverage/naver_ohlcv_supplement_summary.json", "fetched_symbols")
    market_collect_no_rows = json_value("data/raw/market_coverage/naver_ohlcv_supplement_summary.json", "no_row_symbols")
    market_collect_rows = json_value("data/raw/market_coverage/naver_ohlcv_supplement_summary.json", "output_rows_total")
    market_fdr_gap_target = json_value("data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json", "target_symbols")
    market_fdr_gap_fetched = json_value("data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json", "fetched_symbols")
    market_fdr_gap_no_rows = json_value("data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json", "no_row_symbols")
    market_fdr_gap_rows = json_value("data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json", "output_rows_total")
    market_index_rows = json_value("data/raw/market_index/korea_market_index_summary.json", "rows")
    market_index_min_date = json_value("data/raw/market_index/korea_market_index_summary.json", "min_trade_date")
    market_index_max_date = json_value("data/raw/market_index/korea_market_index_summary.json", "max_trade_date")
    market_main_rows_loaded = json_value("data/model_market/dart_model_panel_market_summary.json", "ohlcv_rows_loaded")
    market_main_symbols = json_value("data/model_market/dart_model_panel_market_summary.json", "ohlcv_symbols_loaded")
    market_main_price_rate = json_value("data/model_market/dart_model_panel_market_summary.json", "price_record_rate")
    market_main_recent_rate = json_value("data/model_market/dart_model_panel_market_summary.json", "recent_price_record_rate")
    market_main_index_coverage = json_value("data/model_market/dart_model_panel_market_summary.json", "market_index_coverage_rate")
    market_robust_price_rate = json_value("data/model_market_robust/dart_model_panel_market_summary.json", "price_record_rate")
    market_robust_recent_rate = json_value("data/model_market_robust/dart_model_panel_market_summary.json", "recent_price_record_rate")
    market_robust_index_coverage = json_value("data/model_market_robust/dart_model_panel_market_summary.json", "market_index_coverage_rate")
    market_main_price_rate_text = f"{float(market_main_price_rate) * 100:.1f}%" if isinstance(market_main_price_rate, (int, float)) else market_main_price_rate
    market_main_recent_rate_text = f"{float(market_main_recent_rate) * 100:.1f}%" if isinstance(market_main_recent_rate, (int, float)) else market_main_recent_rate
    market_robust_price_rate_text = f"{float(market_robust_price_rate) * 100:.1f}%" if isinstance(market_robust_price_rate, (int, float)) else market_robust_price_rate
    market_robust_recent_rate_text = f"{float(market_robust_recent_rate) * 100:.1f}%" if isinstance(market_robust_recent_rate, (int, float)) else market_robust_recent_rate
    market_main_index_coverage_text = f"{float(market_main_index_coverage) * 100:.1f}%" if isinstance(market_main_index_coverage, (int, float)) else market_main_index_coverage
    market_robust_index_coverage_text = f"{float(market_robust_index_coverage) * 100:.1f}%" if isinstance(market_robust_index_coverage, (int, float)) else market_robust_index_coverage
    market_residual_status = json_value("results/market_residual_coverage_audit.json", "status")
    market_residual_symbols = json_value("results/market_residual_coverage_audit.json", "fdr_residual_symbols")
    market_residual_main_rows = json_value("results/market_residual_coverage_audit.json", "main", "residual_rows")
    market_residual_main_share = json_value("results/market_residual_coverage_audit.json", "main", "residual_row_share")
    market_residual_main_365 = json_value("results/market_residual_coverage_audit.json", "main", "residual_target_365d_rows")
    market_residual_main_730 = json_value("results/market_residual_coverage_audit.json", "main", "residual_target_730d_rows")
    market_residual_robust_rows = json_value("results/market_residual_coverage_audit.json", "robust", "residual_rows")
    market_residual_robust_share = json_value("results/market_residual_coverage_audit.json", "robust", "residual_row_share")
    market_residual_robust_365 = json_value("results/market_residual_coverage_audit.json", "robust", "residual_target_365d_rows")
    market_residual_robust_730 = json_value("results/market_residual_coverage_audit.json", "robust", "residual_target_730d_rows")
    market_residual_main_share_text = f"{float(market_residual_main_share) * 100:.2f}%" if isinstance(market_residual_main_share, (int, float)) else market_residual_main_share
    market_residual_robust_share_text = f"{float(market_residual_robust_share) * 100:.2f}%" if isinstance(market_residual_robust_share, (int, float)) else market_residual_robust_share
    edinet_feasible_days = json_value("results/edinet_api_feasibility.json", "api_window_feasible_days")
    edinet_infeasible_days = json_value("results/edinet_api_feasibility.json", "api_window_infeasible_days")
    edinet_as_of_date = json_value("results/edinet_api_feasibility.json", "as_of_date")
    edinet_preflight_status = json_value("results/edinet_api_feasibility.json", "preflight", "status")
    edinet_execution_preflight_status = json_value("results/edinet_execution_preflight.json", "status")
    edinet_execution_ready = json_value("results/edinet_execution_preflight.json", "empirical_collection_ready")
    edinet_execution_blockers = json_value("results/edinet_execution_preflight.json", "blockers") or []
    edinet_execution_env_key = json_value("results/edinet_execution_preflight.json", "env_api_key_present")
    edinet_execution_document_rows = json_value("results/edinet_execution_preflight.json", "document_rows")
    edinet_execution_current_rows = json_value("results/edinet_execution_preflight.json", "current_report_candidate_rows")
    edinet_local_status = json_value("results/edinet_local_discovery.json", "status")
    edinet_local_external = json_value("results/edinet_local_discovery.json", "external_reusable_candidate_rows")
    edinet_jp_report_candidates = json_value(
        "results/edinet_local_discovery.json", "external_japanese_report_candidate_rows"
    )
    edinet_jp_report_edinet_mentions = json_value(
        "results/edinet_local_discovery.json", "external_japanese_report_edinet_mentions"
    )
    edinet_local_generated = json_value("results/edinet_local_discovery.json", "generated_paper4_edinet_artifact_rows")
    jp_report_proto_status = json_value("results/jp_report_parser_prototype_summary.json", "status")
    jp_report_proto_files = json_value("results/jp_report_parser_prototype_summary.json", "report_files")
    jp_report_proto_texts = json_value("results/jp_report_parser_prototype_summary.json", "text_files")
    jp_report_proto_events = json_value("results/jp_report_parser_prototype_summary.json", "taxonomy_candidate_rows")
    desktop_total_files = json_value("data/audit/summary.json", "summary", "total_files")
    desktop_total_size_mb = json_value("data/audit/summary.json", "summary", "total_size_mb")
    desktop_root_count_rows = csv_rows("data/audit/desktop_root_data_counts.csv")

    items = [
        item(
            "Sprint 0 local data audit",
            "complete" if all(exists(p) for p in [
                "src/audit/data_audit.py",
                "audit_report.md",
                "data/audit/desktop_data_files_rg.txt",
                "data/audit/data_inventory.csv",
                "data/audit/desktop_root_data_counts.csv",
                "data/audit/summary.json",
                "data_map.md",
            ]) else "incomplete",
            [
                "src/audit/data_audit.py",
                "audit_report.md",
                "data/audit/desktop_data_files_rg.txt",
                "data/audit/data_inventory.csv",
                "data/audit/desktop_root_data_counts.csv",
                "data/audit/summary.json",
                "data_map.md",
            ],
            f"Local inventory found reusable DART, SEC, Korean price, and financial data; no in-scope EDINET document-list/current-report dataset was found. "
            f"A focused local Japanese report probe found {edinet_jp_report_candidates} PDF/text annual-report candidates, "
            f"with {edinet_jp_report_edinet_mentions} text file(s) mentioning EDINET; these are indexed as parser-prototype support only. "
            f"Current scan evidence: {desktop_total_files} data files, {desktop_total_size_mb} MB, "
            f"and {desktop_root_count_rows} rows in the source-count table; detailed local source names are retained only in the audit CSV.",
        ),
        item(
            "DART and SEC event/outcome normalization for 2015-2024",
            "complete" if all(exists(p) for p in [
                "src/prepare/build_phase1_events.py",
                "src/collect/dart_gap_collector.py",
                "data/raw/dart_gap/dart_gap_outcome_candidates.csv",
                "results/dart_gap_collection_status.md",
                "data/processed/events_dart.parquet",
                "data/processed/outcomes_dart.parquet",
                "data/processed/events_sec.parquet",
                "data/processed/outcomes_sec.parquet",
                "data/processed/phase1_event_summary.json",
                "results/sec_gap_inventory.md",
            ]) else "incomplete",
            [
                "src/prepare/build_phase1_events.py",
                "src/collect/dart_gap_collector.py",
                "data/raw/dart_gap/dart_gap_outcome_candidates.csv",
                "results/dart_gap_collection_status.md",
                "data/processed/events_dart.parquet",
                "data/processed/outcomes_dart.parquet",
                "data/processed/events_sec.parquet",
                "data/processed/outcomes_sec.parquet",
                "data/processed/phase1_event_summary.json",
                "results/sec_gap_inventory.md",
            ],
            f"DART and SEC are executable for v1. DART Sprint 1 gap status: {dart_gap_status}; outcome candidates derived from existing Paper 3 major-event rows: {dart_gap_candidates}; outcome firms: {dart_gap_firms}. The SEC gap inventory verifies that the local SEC sample lacks Item 1.03 in 2015-2024.",
        ),
        item(
            "Common regulatory taxonomy and DART-SEC alignment layer",
            "complete" if all(exists(p) for p in [
                "src/align/taxonomy_builder.py",
                "data/processed/regulatory_event_taxonomy.csv",
                "data/processed/events_aligned.parquet",
                "data/processed/event_dictionary.csv",
                "results/alignment_summary.md",
                "src/audit/t04_local_discovery.py",
                "data/audit/t04_local_candidate_files.csv",
                "results/t04_local_discovery.md",
                "results/t04_local_discovery.json",
            ]) else "incomplete",
            [
                "src/align/taxonomy_builder.py",
                "data/processed/regulatory_event_taxonomy.csv",
                "data/processed/events_aligned.parquet",
                "data/processed/event_dictionary.csv",
                "results/alignment_summary.md",
                "src/audit/t04_local_discovery.py",
                "data/audit/t04_local_candidate_files.csv",
                "results/t04_local_discovery.md",
                "results/t04_local_discovery.json",
            ],
            f"This is deterministic taxonomy alignment and coverage, not validated LLM/human alignment accuracy. Focused T04 local discovery status: {t04_discovery_status}; empirical candidate rows: {t04_empirical_candidates}; DART title/content matches: {t04_dart_title_matches}. T04 remains registry-only unless a going-concern/audit-opinion source is collected and normalized.",
        ),
        item(
            "RQ1 Precision@1/MRR alignment evaluation",
            "incomplete"
            if alignment_workbook_structure_status != "pass" or alignment_single_workbook_structure_status != "pass"
            else ("deferred_v2_scope" if alignment_labeled == 0 else "partial"),
            [
                "src/align/llm_aligner.py",
                "data/processed/manual_alignment_seed_100.csv",
                "data/processed/llm_alignment_prompts.jsonl",
                "data/processed/llm_alignment_results.csv",
                "results/llm_alignment_summary.md",
                "src/align/alignment_annotation_qc.py",
                "data/processed/manual_alignment_seed_100_annotation_priority.csv",
                "data/processed/manual_alignment_seed_100_first_pass_30.csv",
                "results/alignment_annotation_qc.md",
                "results/alignment_annotation_qc.json",
                "src/align/alignment_candidate_diagnostics.py",
                "data/processed/manual_alignment_seed_100_candidate_diagnostics.csv",
                "results/alignment_candidate_diagnostics.md",
                "results/alignment_candidate_diagnostics.json",
                "src/align/alignment_silver_evaluation.py",
                "data/processed/manual_alignment_seed_100_silver_review_queue.csv",
                "data/processed/manual_alignment_seed_100_silver_consensus.csv",
                "results/alignment_silver_evaluation.md",
                "results/alignment_silver_evaluation.json",
                "src/align/sec_desktop_alignment_seed.py",
                "data/processed/manual_alignment_seed_sec_desktop_100.csv",
                "data/processed/manual_alignment_seed_sec_desktop_100.xlsx",
                "results/alignment_sec_desktop_seed_summary.md",
                "results/alignment_sec_desktop_seed_summary.json",
                "manuscript/alignment_sec_desktop_labeling_guide.md",
                "results/alignment_sec_desktop_evaluation.md",
                "results/alignment_sec_desktop_evaluation.json",
                "src/align/build_single_expert_label_sheet.py",
                "src/align/sync_single_expert_label_sheet.py",
                "data/processed/manual_alignment_seed_100_single_expert_label_sheet.csv",
                "data/processed/manual_alignment_seed_100_single_expert_label_sheet.xlsx",
                "manuscript/alignment_single_expert_labeling_guide.md",
                "results/alignment_single_expert_readiness.md",
                "results/alignment_single_expert_readiness.json",
                "src/align/build_annotation_workbook.py",
                "src/align/build_annotation_packet.py",
                "src/align/sync_annotation_workbook.py",
                "data/processed/manual_alignment_seed_100_annotation_packet.csv",
                "data/processed/manual_alignment_seed_100_annotation_workbook.xlsx",
                "manuscript/alignment_annotation_guide.md",
                "results/alignment_annotation_readiness.md",
                "src/align/evaluate_alignment.py",
                "results/alignment_evaluation.md",
                "results/alignment_evaluation.json",
                "src/audit/alignment_label_preflight.py",
                "results/alignment_label_preflight.md",
                "results/alignment_label_preflight.json",
            ],
            f"Candidate rows: {manual_seed_rows}; gold-labeled rows: {alignment_labeled}; annotator1/annotator2/adjudicated rows: {alignment_annotator1_labeled}/{alignment_annotator2_labeled}/{alignment_adjudicated_labeled}; annotator-overlap rows: {alignment_overlap_rows}; Cohen kappa: {alignment_kappa}; rows needing adjudication: {alignment_needs_adjudication}; evaluator status: {alignment_status}. LLM-ready prompts, offline candidate judgments, QC priority artifacts, pre-label candidate diagnostics, silver pre-label evaluation, a local SEC supplemental alignment seed, a single-expert labeling fast-lane, a two-annotator/adjudication Excel workbook, annotation guides, and sync/evaluation scripts are generated. Two-annotator workbook structure status: {alignment_workbook_structure_status} ({len(alignment_workbook_structure_checks)} checks); single-expert workbook structure status: {alignment_single_workbook_structure_status} ({len(alignment_single_workbook_structure_checks)} checks). Workbook regeneration preserves existing annotation entries when present; latest preservation count: {alignment_preserved_rows} rows / {alignment_preserved_cells} cells. Single-annotator-only rows are tracked but are not scored as final gold labels unless entered through the explicit single-expert `human_label` route. QC status: {alignment_qc_status}; proposed-vs-AI disagreements: {alignment_qc_disagreements}; low/missing-confidence rows: {alignment_qc_low_confidence}; first-pass priority rows: {alignment_qc_first_pass}. Diagnostic status: {alignment_diag_status}; taxonomy codes covered by original seed: {alignment_diag_codes_text}; high-ambiguity rows: {alignment_diag_high_ambiguity}; other-event rows: {alignment_diag_other_rows}; event-count imbalance rows: {alignment_diag_imbalance_rows}; mean AI confidence: {alignment_diag_mean_confidence}. Silver status: {alignment_silver_status}; strict agreement rate: {alignment_silver_strict_rate}; lenient MATCH/PARTIAL-family agreement rate: {alignment_silver_lenient_rate}; silver consensus/review-needed rows: {alignment_silver_consensus}/{alignment_silver_review_needed}. Local SEC supplemental seed status: {sec_desktop_seed_status}; rows: {sec_desktop_seed_rows}; match/negative rows: {sec_desktop_seed_match_rows}/{sec_desktop_seed_negative_rows}; T01 candidate rows: {sec_desktop_seed_t01_rows}; match-seed taxonomy codes: {sec_desktop_seed_codes_text}; preserved rows/cells: {sec_desktop_seed_preserved_rows}/{sec_desktop_seed_preserved_cells}; supplemental evaluator status: {sec_desktop_eval_status}; supplemental labeled rows: {sec_desktop_eval_labeled}; supplemental Cohen kappa: {sec_desktop_eval_kappa}. Single-expert sheet status: {alignment_single_status}; labeled rows in that sheet: {alignment_single_labeled}; first-pass rows: {alignment_single_first_pass}; high-ambiguity rows: {alignment_single_high_ambiguity}. Label preflight status: {alignment_label_preflight_status}; final metrics ready: {alignment_label_preflight_ready}; blockers: {alignment_label_preflight_blockers}. Human labels are still required before claiming Precision@1 or MRR; silver labels are annotation triage only, and inter-annotator agreement additionally requires both annotator columns.",
        ),
        item(
            "EDINET/Japanese disclosure data",
            "deferred_v2_scope",
            [
                "src/collect/edinet_collector.py",
                "src/collect/edinet_feasibility.py",
                "src/audit/edinet_local_discovery.py",
                "src/audit/edinet_execution_preflight.py",
                "src/prepare/jp_report_parser_prototype.py",
                "data/audit/summary.json",
                "data/audit/edinet_local_candidate_files.csv",
                "data/raw/edinet/edinet_collection_plan.csv",
                "data/raw/edinet/edinet_collection_plan_feasibility.csv",
                "data/raw/edinet/edinet_documents_index.csv",
                "data/raw/edinet/jp_report_parser_prototype_index.csv",
                "data/processed/jp_report_parser_prototype_events.csv",
                "data/processed/jp_report_parser_prototype_events.parquet",
                "results/edinet_collection_status.md",
                "results/edinet_api_feasibility.md",
                "results/edinet_local_discovery.md",
                "results/edinet_execution_preflight.md",
                "results/edinet_execution_preflight.json",
                "results/jp_report_parser_prototype_summary.md",
                "data_map.md",
                "manuscript/submission_strategy.md",
            ],
            f"No local reusable 2015-2024 EDINET document-list/current-report dataset was found. Focused local discovery status: {edinet_local_status}; external reusable EDINET candidates: {edinet_local_external}; local Japanese report PDF/text candidates: {edinet_jp_report_candidates}; Japanese report EDINET mentions: {edinet_jp_report_edinet_mentions}; generated Paper 4 EDINET planning/status artifacts: {edinet_local_generated}. The Japanese report parser prototype status is {jp_report_proto_status}: report/text files {jp_report_proto_files}/{jp_report_proto_texts}, taxonomy candidate rows {jp_report_proto_events}. The Japanese report files are parser-design support only; actual EDINET document metadata require EDINET_API_KEY and explicit execution. As of {edinet_as_of_date}, approximate API date-window feasibility is {edinet_feasible_days} planned days feasible and {edinet_infeasible_days} planned days outside the 10-year window; no-key preflight status is {edinet_preflight_status}. EDINET execution preflight status is {edinet_execution_preflight_status}: empirical ready {edinet_execution_ready}, env key present {edinet_execution_env_key}, document/current-report rows {edinet_execution_document_rows}/{edinet_execution_current_rows}, blockers {edinet_execution_blockers}. v1 remains DART-first with SEC taxonomy-transfer support.",
        ),
        item(
            "Expanded SEC standalone distress validation",
            "deferred_v2_scope",
            [
                "src/collect/sec_gap_collector.py",
                "data/raw/sec_expanded/sec_expanded_collection_plan.csv",
                "data/raw/sec_expanded/sec_expanded_8k_filings.csv",
                "data/raw/sec_expanded/sec_expanded_outcome_candidates.csv",
                "results/sec_expanded_collection_status.md",
                "src/sequence/sec_expanded_sequences.py",
                "data/processed/sec_expanded_events.parquet",
                "data/processed/sec_expanded_outcomes.parquet",
                "data/processed/sec_expanded_sequence_panel.parquet",
                "results/sec_expanded_sequence_summary.md",
                "data/raw/sec_bankruptcy_cases/sec_bankruptcy_case_seed.csv",
                "src/collect/sec_bankruptcy_case_collector.py",
                "data/raw/sec_bankruptcy_cases/sec_bankruptcy_case_filings.csv",
                "data/raw/sec_bankruptcy_cases/sec_bankruptcy_case_item103.csv",
                "results/sec_bankruptcy_case_status.md",
                "data/processed/sec_bankruptcy_case_events.parquet",
                "data/processed/sec_bankruptcy_case_outcomes.parquet",
                "data/processed/sec_bankruptcy_case_sequence_panel.parquet",
                "results/sec_bankruptcy_case_sequence_summary.md",
                "results/sec_bankruptcy_case_sequences.md",
                "src/collect/sec_submissions_archive_collector.py",
                "data/raw/sec_submissions_archive/sec_submissions_archive_manifest.csv",
                "results/sec_submissions_archive_status.md",
                "results/sec_submissions_archive_status.json",
                "src/collect/sec_desktop_population_collector.py",
                "data/raw/sec_desktop_population/sec_desktop_population_8k_filings.csv",
                "data/raw/sec_desktop_population/sec_desktop_population_outcome_candidates.csv",
                "results/sec_desktop_population_collection_status.md",
                "data/processed/sec_desktop_population_events.parquet",
                "data/processed/sec_desktop_population_outcomes.parquet",
                "data/processed/sec_desktop_population_sequence_panel.parquet",
                "results/sec_desktop_population_sequence_summary.md",
                "results/sec_desktop_population_bankruptcy_sequences.md",
                "src/model/sec_desktop_sampling_frame.py",
                "data/model_sec_desktop/sec_desktop_sampling_frame_firms.csv",
                "data/model_sec_desktop/sec_desktop_sampling_frame_anchors.csv",
                "results/sec_desktop_sampling_frame_summary.md",
                "src/model/sec_desktop_sequence_validation.py",
                "src/model/sec_desktop_validation_robustness.py",
                "data/model_sec_desktop/sec_desktop_validation_panel.parquet",
                "results/sec_desktop_validation_summary.md",
                "results/sec_desktop_validation_metrics.csv",
                "results/sec_desktop_validation_firm_disjoint_summary.md",
                "results/sec_desktop_validation_firm_disjoint_metrics.csv",
                "src/audit/sec_companyfacts_availability.py",
                "data/model_sec_desktop/sec_companyfacts_file_inventory.csv",
                "data/model_sec_desktop/sec_companyfacts_validation_coverage.csv",
                "results/sec_companyfacts_availability.md",
                "results/sec_companyfacts_availability.json",
                "src/model/sec_desktop_financial_baseline.py",
                "data/model_sec_desktop/sec_desktop_companyfacts_selected.parquet",
                "data/model_sec_desktop/sec_desktop_financial_panel.parquet",
                "results/sec_desktop_financial_baseline_summary.md",
                "results/sec_desktop_financial_baseline_metrics.csv",
                "results/sec_desktop_financial_baseline_predictions.csv",
                "src/collect/sec_filing_text_manifest.py",
                "data/raw/sec_filing_text/sec_filing_text_manifest.csv",
                "results/sec_filing_text_collection_status.md",
                "results/sec_filing_text_collection_status.json",
                "src/audit/sec_all8k_text_plan.py",
                "data/raw/sec_all8k_text/sec_all8k_text_manifest.csv",
                "data/raw/sec_all8k_text/sec_all8k_text_batch_plan.csv",
                "results/sec_all8k_text_scope_plan.md",
                "results/sec_all8k_text_scope_plan.json",
                "src/collect/sec_all8k_text_batch_collector.py",
                "data/raw/sec_all8k_text/sec_all8k_text_batch_status.csv",
                "results/sec_all8k_text_batch_collection_status.md",
                "results/sec_all8k_text_batch_collection_status.json",
                "src/audit/sec_all8k_storage_preflight.py",
                "results/sec_all8k_storage_preflight.md",
                "results/sec_all8k_storage_preflight.json",
                "src/audit/sec_all8k_text_sample_design.py",
                "data/raw/sec_all8k_text/sec_all8k_text_sample_plan.csv",
                "data/raw/sec_all8k_text/sec_all8k_text_sample_strata.csv",
                "results/sec_all8k_text_sample_design.md",
                "results/sec_all8k_text_sample_design.json",
                "src/collect/sec_all8k_sample_text_collector.py",
                "data/raw/sec_all8k_text/sec_all8k_text_sample_status.csv",
                "results/sec_all8k_text_sample_collection_status.md",
                "results/sec_all8k_text_sample_collection_status.json",
                "src/model/sec_all8k_sample_text_baseline.py",
                "data/model_sec_desktop/sec_all8k_sample_text_features.csv",
                "data/model_sec_desktop/sec_all8k_sample_text_features.parquet",
                "results/sec_all8k_sample_text_baseline_summary.md",
                "results/sec_all8k_sample_text_baseline_metrics.csv",
                "results/sec_all8k_sample_text_baseline_predictions.csv",
                "src/model/sec_desktop_text_metadata_baseline.py",
                "data/model_sec_desktop/sec_desktop_text_metadata_panel.parquet",
                "results/sec_desktop_text_metadata_baseline_summary.md",
                "results/sec_desktop_text_metadata_baseline_metrics.csv",
                "results/sec_desktop_text_metadata_baseline_predictions.csv",
                "src/model/sec_filing_body_text_baseline.py",
                "data/model_sec_desktop/sec_filing_body_text_panel.parquet",
                "results/sec_filing_body_text_baseline_summary.md",
                "results/sec_filing_body_text_baseline_metrics.csv",
                "results/sec_filing_body_text_baseline_predictions.csv",
                "src/audit/sec_validation_scope.py",
                "data/model_sec_desktop/sec_validation_scope_decision_table.csv",
                "results/sec_validation_scope.md",
                "results/sec_validation_scope.json",
                "src/audit/sec_gap_inventory.py",
                "results/sec_gap_inventory.md",
                "results/sec_gap_inventory.csv",
                "results/sec_gap_inventory.json",
            ],
            f"Local EDGAR files contain zero Item 1.03 bankruptcy/receivership rows in 2015-2024 and only two de-duplicated Item 1.03 rows when 2025 is included. Expanded SEC status: {sec_expanded_status}; filings: {sec_expanded_filings}; target-item candidates: {sec_expanded_candidates}; Item 1.03 rows: {sec_expanded_item_103}; item-level sequence rows: {sec_expanded_sequence_events}; outcome rows from Items 2.04/3.01: {sec_expanded_sequence_outcomes}; outcome firms: {sec_expanded_sequence_outcome_firms}. A targeted SEC bankruptcy case sample adds actual Item 1.03 sequence evidence: Item 1.03 rows: {sec_case_item103}; Item 1.03 firms: {sec_case_item103_firms}; sequence rows: {sec_case_sequence_events}; outcome rows: {sec_case_sequence_outcomes}. A broader local SEC submissions scan is now available as population-style v2 evidence: archive overlap references cached: {sec_archive_overlap_cached}; archive overlap cache rate: {sec_archive_overlap_rate}; referenced archive filing count: {sec_archive_referenced_filing_count}; status: {sec_desktop_status}; 8-K rows: {sec_desktop_filings}; firms: {sec_desktop_firms}; target-item candidates: {sec_desktop_candidates}; Item 1.03/2.04/3.01 rows: {sec_desktop_item103}/{sec_desktop_item204}/{sec_desktop_item301}; item-level sequence rows: {sec_desktop_sequence_events}; outcome rows: {sec_desktop_sequence_outcomes}; outcome firms: {sec_desktop_sequence_outcome_firms}; Item 1.03 case firms in sequence table: {sec_desktop_bankruptcy_firms}. The formal local SEC sampling frame is now documented: retained firms: {sec_frame_firms}; validation firms: {sec_frame_validation_firms}; task-anchor rows: {sec_frame_anchor_rows}; test anchor rows: {sec_frame_test_rows}. SEC sequence-only out-of-time validation is generated: panel rows: {sec_validation_panel_rows}; firms: {sec_validation_firms}; all-distress 730-day random-forest ROC AUC/top-decile lift: {sec_all_730_auc}/{sec_all_730_lift}; bankruptcy-only 730-day logistic ROC AUC/top-decile lift/test positives: {sec_bank_730_auc}/{sec_bank_730_lift}/{sec_bank_730_test_pos}. Firm-disjoint temporal robustness gives all-distress 730-day random-forest ROC AUC/top-decile lift: {sec_fd_all_730_auc}/{sec_fd_all_730_lift}; bankruptcy-only 730-day logistic ROC AUC/top-decile lift: {sec_fd_bank_730_auc}/{sec_fd_bank_730_lift}. SEC companyfacts availability is verified locally: {sec_companyfacts_files} files, approximately {sec_companyfacts_gb_fmt} GB, with coverage for {sec_companyfacts_validation_ciks} validation CIKs at rate {sec_companyfacts_validation_coverage}. The full SEC companyfacts financial-ratio baseline is generated: selected fact records: {sec_fin_facts}; files with selected facts: {sec_fin_files}; financial panel rows: {sec_fin_panel_rows}; all-distress 730-day financial-only random-forest ROC AUC/top-decile lift: {sec_fin_only_all_730_auc}/{sec_fin_only_all_730_lift}; sequence-plus-financial ROC AUC/top-decile lift/coverage: {sec_fin_all_730_auc}/{sec_fin_all_730_lift}/{sec_fin_all_730_coverage}; bankruptcy-only 730-day sequence-plus-financial logistic ROC AUC/top-decile lift: {sec_fin_bank_730_auc}/{sec_fin_bank_730_lift}. A SEC filing-body text manifest is also created for {sec_text_manifest_rows} target-item filings, with {sec_text_manifest_local} local filing-text files currently present in the manifest status. A full all-8-K text-scope plan is generated with status {sec_all8k_plan_status}: manifest rows {sec_all8k_plan_rows}, missing files {sec_all8k_missing_files}, planned batches {sec_all8k_batches}, and estimated missing text size {sec_all8k_missing_gb} GB. The all-8-K batch collector is status-only by default and currently reports status {sec_all8k_batch_status}, open batches {sec_all8k_batch_open}, next batch {sec_all8k_batch_next}, missing files {sec_all8k_batch_missing}, and downloaded files in this run {sec_all8k_batch_downloads}; path resolution mode is {sec_all8k_batch_path_mode}, with {sec_all8k_batch_redacted_paths} redacted manifest path rows and {sec_all8k_batch_canonical_paths} canonical CIK/accession text paths resolved; its historical download log records {sec_all8k_batch_log_downloaded}/{sec_all8k_batch_log_rows} downloaded rows and {sec_all8k_batch_log_existing} existing local files. Full-scope download preflight status is {sec_all8k_preflight_status}: required free disk space {sec_all8k_preflight_required_gb} GB, storage ready {sec_all8k_preflight_storage_ready}, explicit SEC/EDGAR user agent present {sec_all8k_preflight_env_user_agent}, full-scope download ready {sec_all8k_preflight_ready}, blockers {sec_all8k_preflight_blockers}. A bounded all-8-K text sampling design is generated with status {sec_all8k_sample_status}: {sec_all8k_sample_non_target} non-target sample rows, {sec_all8k_sample_target_reference} target-reference rows, {sec_all8k_sample_strata}/{sec_all8k_sample_population_strata} sampled strata, and estimated missing sample text size {sec_all8k_sample_mb} MB; current sample collection status is {sec_sample_collection_status} with {sec_sample_collection_cached} cached sample files, {sec_sample_collection_missing} missing sample files, and {sec_sample_collection_attempts} download attempts. The bounded sample text baseline is now generated with status {sec_sample_text_status}: {sec_sample_text_rows} sample filing bodies modeled, {sec_sample_text_features} transparent body-text features, and target-item-vs-non-target random-forest ROC AUC/top-decile lift {sec_sample_text_rf_auc}/{sec_sample_text_rf_lift}. This is sample-level sensitivity evidence, not standalone future-distress prediction. A local metadata-text baseline is generated with {sec_textmeta_features} text-like metadata features: all-distress 730-day metadata-text-only random-forest ROC AUC/top-decile lift: {sec_textmeta_all_730_auc}/{sec_textmeta_all_730_lift}; sequence-plus-financial-plus-metadata-text ROC AUC/top-decile lift: {sec_all_730_full_auc}/{sec_all_730_full_lift}. A separate cached filing-body baseline now records status {sec_body_text_status}, cached files {sec_body_text_cached}, cached rows used {sec_body_text_rows}, and body-text feature count {sec_body_text_features}; all-distress 730-day filing-body-only random-forest ROC AUC/top-decile lift: {sec_body_all_730_auc}/{sec_body_all_730_lift}; sequence-plus-financial-plus-filing-body ROC AUC/top-decile lift: {sec_body_combo_730_auc}/{sec_body_combo_730_lift}. The SEC validation-scope audit is now generated with status {sec_scope_status}: target-item body-text accession coverage is {sec_scope_target_coverage} for {sec_scope_target_rows} target-item filings, but all-8-K body-text scope coverage is only {sec_scope_all_text_coverage} across {sec_scope_all_8k_rows} 8-K rows, leaving {sec_scope_missing_all_8k_text} rows outside the current full filing-body cache. Treat this as v2 evidence until full all-8-K filing-body scope and final standalone U.S. validation scope are added.",
        ),
        item(
            "Firm-level sequence panels and pattern mining",
            "complete" if all(exists(p) for p in [
                "src/sequence/build_sequence_panel.py",
                "data/processed/sequence_panel_dart.parquet",
                "data/processed/sequence_panel_sec.parquet",
                "src/sequence/pattern_miner.py",
                "results/pattern_rules.md",
                "results/pattern_rules.csv",
            ]) else "incomplete",
            [
                "src/sequence/build_sequence_panel.py",
                "data/processed/sequence_panel_dart.parquet",
                "data/processed/sequence_panel_sec.parquet",
                "src/sequence/pattern_miner.py",
                "results/pattern_rules.md",
                "results/pattern_rules.csv",
            ],
            "Pattern mining is implemented as contiguous taxonomy-code n-grams; no external PrefixSpan dependency was added.",
        ),
        item(
            "Survival/time-to-event hazard analysis",
            "complete" if all(exists(p) for p in [
                "src/sequence/survival_model.py",
                "results/survival_hazard_summary.md",
                "results/survival_hazard_metrics.csv",
                "results/survival_hazard_coefficients.csv",
            ]) else "incomplete",
            [
                "src/sequence/survival_model.py",
                "results/survival_hazard_summary.md",
                "results/survival_hazard_metrics.csv",
                "results/survival_hazard_coefficients.csv",
            ],
            "Discrete-time complementary log-log hazard models provide the current time-to-event layer; full multivariable Cox PH is documented as numerically unstable on the sparse landmark panel.",
        ),
        item(
            "SupTech flagging evaluation",
            "complete" if all(exists(p) for p in [
                "src/flagging/rule_engine.py",
                "results/flagging_rule_registry.csv",
                "results/flagging_rule_engine_summary.md",
                "src/flagging/evaluate_flagging.py",
                "results/flagging_performance.md",
                "results/flagging_performance.csv",
                "results/flagging_rule_firm_flags.csv",
            ]) else "incomplete",
            [
                "src/flagging/rule_engine.py",
                "results/flagging_rule_registry.csv",
                "results/flagging_rule_engine_summary.md",
                "src/flagging/evaluate_flagging.py",
                "results/flagging_performance.md",
                "results/flagging_performance.csv",
                "results/flagging_rule_firm_flags.csv",
            ],
            f"The explicit SupTech rule registry is generated separately, and rule-based triggers are evaluated against model-based top-decile watchlists. T04-based rules are retained for research-plan fidelity, but the focused local discovery audit currently reports {t04_empirical_candidates} empirical T04 source candidates.",
        ),
        item(
            "Financial fundamentals and classic baseline comparison",
            "complete" if all(exists(p) for p in [
                "src/model/build_dart_model_dataset.py",
                "data/model/dart_model_panel.parquet",
                "src/flagging/baseline_models.py",
                "results/classic_financial_baselines.md",
                "results/classic_financial_baseline_audit.json",
            ]) else "incomplete",
            [
                "src/model/build_dart_model_dataset.py",
                "data/model/dart_model_panel.parquet",
                "src/flagging/baseline_models.py",
                "results/classic_financial_baselines.md",
                "results/classic_financial_baseline_audit.json",
            ],
            "Exact Altman/Ohlson formulas are not computable from current fields; available-component proxies are documented and evaluated.",
        ),
        item(
            "Out-of-time sequence/financial model training and robustness",
            "complete" if all(exists(p) for p in [
                "src/model/train_dart_baselines.py",
                "results/dart_model_metrics.md",
                "results/dart_model_bootstrap_ci.csv",
                "results/dart_model_calibration.csv",
                "results/dart_robustness_summary.md",
            ]) else "incomplete",
            [
                "src/model/train_dart_baselines.py",
                "results/dart_model_metrics.md",
                "results/dart_model_bootstrap_ci.csv",
                "results/dart_model_calibration.csv",
                "results/dart_robustness_summary.md",
            ],
            "Main and robust outcome definitions are both evaluated.",
        ),
        item(
            "Market-price OHLCV extension",
            "complete" if all(exists(p) for p in [
                "src/collect/market_coverage_collector.py",
                "data/raw/market_coverage/naver_ohlcv_supplement.csv",
                "data/raw/market_coverage/naver_ohlcv_supplement_status.csv",
                "data/raw/market_coverage/naver_ohlcv_supplement_summary.md",
                "data/raw/market_coverage/naver_ohlcv_supplement_summary.json",
                "src/collect/market_fdr_gap_collector.py",
                "data/raw/market_coverage/fdr_ohlcv_gap_supplement.csv",
                "data/raw/market_coverage/fdr_ohlcv_gap_supplement_status.csv",
                "data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.md",
                "data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json",
                "src/collect/market_index_collector.py",
                "data/raw/market_index/korea_market_indices.csv",
                "data/raw/market_index/korea_market_index_summary.md",
                "data/raw/market_index/korea_market_index_summary.json",
                "src/model/build_market_features.py",
                "data/model_market/dart_model_panel_market.parquet",
                "data/model_market/dart_model_panel_market_summary.json",
                "results/market/dart_model_metrics.md",
                "data/model_market_robust/dart_model_panel_market.parquet",
                "data/model_market_robust/dart_model_panel_market_summary.json",
                "results/market_robust/dart_model_metrics.md",
                "results/market_extension_summary.md",
                "src/audit/market_residual_coverage_audit.py",
                "results/market_residual_coverage_audit.md",
                "results/market_residual_coverage_audit.json",
                "data/audit/market_residual_symbol_impact.csv",
                "data/audit/market_residual_year_impact.csv",
            ]) and market_residual_status == "pass" else "incomplete",
            [
                "src/collect/market_coverage_collector.py",
                "data/raw/market_coverage/naver_ohlcv_supplement.csv",
                "data/raw/market_coverage/naver_ohlcv_supplement_status.csv",
                "data/raw/market_coverage/naver_ohlcv_supplement_summary.md",
                "data/raw/market_coverage/naver_ohlcv_supplement_summary.json",
                "src/collect/market_fdr_gap_collector.py",
                "data/raw/market_coverage/fdr_ohlcv_gap_supplement.csv",
                "data/raw/market_coverage/fdr_ohlcv_gap_supplement_status.csv",
                "data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.md",
                "data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json",
                "src/collect/market_index_collector.py",
                "data/raw/market_index/korea_market_indices.csv",
                "data/raw/market_index/korea_market_index_summary.md",
                "data/raw/market_index/korea_market_index_summary.json",
                "src/model/build_market_features.py",
                "data/model_market/dart_model_panel_market.parquet",
                "data/model_market/dart_model_panel_market_summary.json",
                "results/market/dart_model_metrics.md",
                "data/model_market_robust/dart_model_panel_market.parquet",
                "data/model_market_robust/dart_model_panel_market_summary.json",
                "results/market_robust/dart_model_metrics.md",
                "results/market_extension_summary.md",
                "src/audit/market_residual_coverage_audit.py",
                "results/market_residual_coverage_audit.md",
                "results/market_residual_coverage_audit.json",
                "data/audit/market_residual_symbol_impact.csv",
                "data/audit/market_residual_year_impact.csv",
            ],
            f"OHLCV features are added as a market-price control and incremental-signal layer. Supplemental Naver fchart collection targets {market_collect_target} deficient symbols, fetches {market_collect_fetched}, records {market_collect_no_rows} no-row symbols, and saves {market_collect_rows} rows without mutating the KY-PLATFORM source DB. The FDR gap collector then targets {market_fdr_gap_target} Naver no-row symbols, fetches {market_fdr_gap_fetched}, leaves {market_fdr_gap_no_rows} residual no-row symbols, and saves {market_fdr_gap_rows} rows. The KOSPI/KOSDAQ benchmark artifact contains {market_index_rows} index rows from {market_index_min_date} to {market_index_max_date}. Main market panel loads {market_main_rows_loaded} OHLCV rows for {market_main_symbols} symbols with any/recent/index coverage {market_main_price_rate_text}/{market_main_recent_rate_text}/{market_main_index_coverage_text}; robust any/recent/index coverage is {market_robust_price_rate_text}/{market_robust_recent_rate_text}/{market_robust_index_coverage_text}. Residual coverage audit status is {market_residual_status}: {market_residual_symbols} residual symbols map to {market_residual_main_rows} main-panel rows ({market_residual_main_share_text}) and {market_residual_main_365}/{market_residual_main_730} residual 365/730-day positive rows; robust panel impact is {market_residual_robust_rows} rows ({market_residual_robust_share_text}) and {market_residual_robust_365}/{market_residual_robust_730} positives. Treat this as a coverage-expanded control/signal extension, not causal price-reaction or security-level event-study evidence.",
        ),
        item(
            "Interpretability and real-case evidence",
            "complete" if all(exists(p) for p in [
                "results/dart_model_feature_importance.csv",
                "results/dart_model_permutation_importance.md",
                "results/dart_distress_case_sequences.md",
            ]) else "incomplete",
            [
                "results/dart_model_feature_importance.csv",
                "results/dart_model_permutation_importance.md",
                "results/dart_distress_case_sequences.md",
            ],
            "Feature importance, permutation importance, and real DART distress cases are available for manuscript support.",
        ),
        item(
            "Bilingual manuscript, references, Word drafts, submission strategy, target-journal routing, and version control",
            "complete" if all(exists(p) for p in [
                "manuscript/paper4_v4_manuscript_en.md",
                "manuscript/paper4_v4_manuscript_en.docx",
                "manuscript/paper4_v4_manuscript_ko.md",
                "manuscript/paper4_v4_manuscript_ko.docx",
                "manuscript/paper4_v4_title_page.md",
                "manuscript/paper4_v4_title_page.docx",
                "manuscript/references.bib",
                "manuscript/submission_strategy.md",
                "manuscript/paper4_target_journal_expansion_ko.md",
                "manuscript/submission_candidate_journals_ko.md",
                "manuscript/submission_candidate_journals_ko.docx",
                "data/audit/submission_candidate_journal_sources.json",
                "src/audit/journal_candidate_source_audit.py",
                "results/journal_candidate_source_audit.md",
                "results/journal_candidate_source_audit.json",
                "manuscript/claim_readiness_matrix.md",
                "src/audit/audit_forbidden_tokens.py",
                "results/manuscript_metric_audit.md",
                "results/manuscript_metric_audit.json",
                "results/manuscript_claim_audit.md",
                "results/manuscript_claim_audit.json",
                "src/audit/bilingual_manuscript_audit.py",
                "results/bilingual_manuscript_audit.md",
                "results/bilingual_manuscript_audit.json",
                "src/audit/build_paper_figures.py",
                "figures/paper4_framework.png",
                "figures/paper4_model_performance.png",
                "figures/paper4_market_coverage.png",
                "results/figure_generation_audit.md",
                "results/figure_generation_audit.json",
                "src/audit/build_submission_figures_en.py",
                "figures/en/paper4_framework_en.png",
                "figures/en/paper4_data_scope_en.png",
                "figures/en/paper4_market_coverage_en.png",
                "figures/en/paper4_model_performance_en.png",
                "figures/en/paper4_permutation_importance_en.png",
                "figures/en/paper4_sequence_market_reaction_en.png",
                "figures/en/paper4_graphrag_ablation_en.png",
                "figures/en/paper4_graphrag_calibration_en.png",
                "results/submission_figure_generation_audit.md",
                "results/submission_figure_generation_audit.json",
                "manuscript/cover_letter_v1_generic.md",
                "manuscript/submission_checklist_v1.md",
                "paper4_execution_status.md",
                "src/audit/submission_package_audit.py",
                "results/submission_package_audit.md",
                "results/submission_package_audit.json",
                "results/calculation_sheet.xlsx",
                "results/calculation_sheet.csv",
                "results/calculation_sheet.md",
                "results/calculation_sheet_audit.json",
                "results/reference_consistency_audit.md",
                "results/reference_consistency_audit.json",
                "src/audit/word_export_audit.py",
                "results/word_export_audit.md",
                "results/word_export_audit.json",
                "VERSIONS.md",
            ])
            and metric_audit_status == "passed"
            and claim_audit_status == "passed"
            and bilingual_audit_status == "pass"
            and figure_generation_status == "pass"
            and submission_figure_generation_status == "pass"
            and submission_package_status == "pass"
            and submission_package_documents == EXPECTED_SUBMISSION_DOCUMENTS
            and submission_package_failures == 0
            and calculation_audit_status == "pass"
            and calculation_output_integrity_count >= EXPECTED_CALCULATION_OUTPUT_INTEGRITY_CHECKS
            and calculation_output_failure_count == 0
            and reference_audit_status == "pass"
            and word_export_status == "pass"
            and word_export_documents == EXPECTED_WORD_EXPORT_DOCUMENTS
            and word_export_failures == 0
            and word_export_claim_blockers == 0
            and word_export_claim_review == 0
            and journal_source_status == "pass"
            and journal_source_candidate_count == EXPECTED_CANDIDATE_JOURNALS
            and journal_source_expected_candidate_count == EXPECTED_CANDIDATE_JOURNALS
            and journal_source_doi_count == EXPECTED_CANDIDATE_DOIS
            and journal_source_policy_count == EXPECTED_CANDIDATE_POLICY_URLS
            and journal_source_failed_doi_count == 0
            and journal_source_failed_policy_count == 0
            and journal_source_candidate_ranks_ok is True
            and journal_source_candidate_journals_ok is True
            and journal_source_candidate_doi_evidence_ok is True
            and journal_source_min_related_work_date == EXPECTED_CANDIDATE_MIN_RELATED_WORK_DATE
            else "incomplete",
            [
                "manuscript/paper4_v4_manuscript_en.md",
                "manuscript/paper4_v4_manuscript_en.docx",
                "manuscript/paper4_v4_manuscript_ko.md",
                "manuscript/paper4_v4_manuscript_ko.docx",
                "manuscript/paper4_v4_title_page.md",
                "manuscript/paper4_v4_title_page.docx",
                "manuscript/references.bib",
                "manuscript/submission_strategy.md",
                "manuscript/paper4_target_journal_expansion_ko.md",
                "manuscript/submission_candidate_journals_ko.md",
                "manuscript/submission_candidate_journals_ko.docx",
                "data/audit/submission_candidate_journal_sources.json",
                "src/audit/journal_candidate_source_audit.py",
                "results/journal_candidate_source_audit.md",
                "results/journal_candidate_source_audit.json",
                "manuscript/claim_readiness_matrix.md",
                "src/audit/audit_forbidden_tokens.py",
                "src/audit/manuscript_metric_audit.py",
                "results/manuscript_metric_audit.md",
                "results/manuscript_metric_audit.json",
                "src/audit/manuscript_claim_audit.py",
                "results/manuscript_claim_audit.md",
                "results/manuscript_claim_audit.json",
                "src/audit/bilingual_manuscript_audit.py",
                "results/bilingual_manuscript_audit.md",
                "results/bilingual_manuscript_audit.json",
                "src/audit/build_paper_figures.py",
                "figures/paper4_framework.png",
                "figures/paper4_model_performance.png",
                "figures/paper4_market_coverage.png",
                "results/figure_generation_audit.md",
                "results/figure_generation_audit.json",
                "src/audit/build_submission_figures_en.py",
                "figures/en/paper4_framework_en.png",
                "figures/en/paper4_data_scope_en.png",
                "figures/en/paper4_market_coverage_en.png",
                "figures/en/paper4_model_performance_en.png",
                "figures/en/paper4_permutation_importance_en.png",
                "figures/en/paper4_sequence_market_reaction_en.png",
                "figures/en/paper4_graphrag_ablation_en.png",
                "figures/en/paper4_graphrag_calibration_en.png",
                "results/submission_figure_generation_audit.md",
                "results/submission_figure_generation_audit.json",
                "manuscript/cover_letter_v1_generic.md",
                "manuscript/submission_checklist_v1.md",
                "paper4_execution_status.md",
                "src/audit/submission_package_audit.py",
                "results/submission_package_audit.md",
                "results/submission_package_audit.json",
                "src/audit/build_calculation_sheet.py",
                "results/calculation_sheet.xlsx",
                "results/calculation_sheet.csv",
                "results/calculation_sheet.md",
                "results/calculation_sheet_audit.json",
                "src/audit/reference_consistency_audit.py",
                "results/reference_consistency_audit.md",
                "results/reference_consistency_audit.json",
                "src/audit/word_export_audit.py",
                "results/word_export_audit.md",
                "results/word_export_audit.json",
                "src/audit/research_plan_traceability.py",
                "results/research_plan_traceability.md",
                "results/research_plan_traceability.json",
                "VERSIONS.md",
            ],
            f"Current v1 is DART-first. ESWA/KBS/DSS are the leading target routes, and the 10-journal Korean Word candidate table is source-audited; paid OA routes remain avoided. Manuscript metric audit status: {metric_audit_status}; checks: {metric_audit_checks}; missing/stale checks: {metric_audit_missing}. Manuscript claim audit status: {claim_audit_status}; claim-audited Markdown files: {claim_audit_scanned_count}; blockers: {claim_audit_blockers}; review items: {claim_audit_review}. Bilingual manuscript audit status: {bilingual_audit_status}; check families: {bilingual_audit_checks}; failures: {bilingual_audit_failures}. Korean figure generation audit status: {figure_generation_status}; figures: {figure_generation_count}. English submission figure audit status: {submission_figure_generation_status}; figures: {submission_figure_generation_count}. Journal candidate source audit status: {journal_source_status}; candidate rows: {journal_source_candidate_count}/{journal_source_expected_candidate_count}; minimum related-work date: {journal_source_min_related_work_date}; candidate ranks/journals/DOI evidence ok: {journal_source_candidate_ranks_ok}/{journal_source_candidate_journals_ok}/{journal_source_candidate_doi_evidence_ok}; DOI/policy checks: {journal_source_doi_count}/{journal_source_policy_count}; failed DOI/policy checks: {journal_source_failed_doi_count}/{journal_source_failed_policy_count}. Submission package audit status: {submission_package_status}; documents: {submission_package_documents}; failures: {submission_package_failures}. Calculation sheet audit status: {calculation_audit_status}; calculation/model/manuscript/verification checks: {calculation_rows}/{calculation_model_rows}/{calculation_manuscript_checks}/{calculation_verification_checks}; output integrity checks/failures: {calculation_output_integrity_count}/{calculation_output_failure_count}; failures: {calculation_failure_count}. Reference consistency audit status: {reference_audit_status}; checked references: {reference_checked}; failures: {reference_failure_count}. Word export audit status: {word_export_status}; documents: {word_export_documents}; failures: {word_export_failures}; docx claim blockers/review: {word_export_claim_blockers}/{word_export_claim_review}; draft title-page status: {word_export_draft_sync}. Reproducibility run-report self-audit is emitted after the current run report is written; see results/reproducibility_run_report_audit.md and .json for current-run step/output/freshness verification. Research-plan traceability status: {traceability_status}; trace items: {traceability_items}; blockers: {traceability_blockers}.",
        ),
    ]

    counts = {}
    for row in items:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    overall = "v1_ready_no_open_submission_risks"
    if any(row["status"] == "incomplete" for row in items):
        overall = "incomplete"
    return {
        "overall_status": overall,
        "status_counts": counts,
        "items": items,
    }


def render_markdown(audit: dict) -> str:
    lines = ["# Paper 4 Completion Audit", ""]
    lines.append(f"Overall status: `{audit['overall_status']}`")
    lines.append("")
    lines.append("This audit checks the current workspace against `paper4_research_plan.md`. APFM v1 is a DART-first finance manuscript; RQ1 human-label metrics, EDINET actual collection, and population-representative SEC validation are deferred v2 scope and are not active v1 submission claims.")
    lines.append("")
    lines.append("## Status Counts")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|---|---:|")
    for status, count in sorted(audit["status_counts"].items()):
        lines.append(f"| `{status}` | {count} |")
    lines.append("")
    lines.append("## Requirement Evidence")
    lines.append("")
    lines.append("| Requirement | Status | Evidence | Notes |")
    lines.append("|---|---|---|---|")
    for row in audit["items"]:
        evidence = "<br>".join(f"`{e}` (present)" if exists(e) else f"`{e}` (missing)" for e in row["evidence"])
        notes = row["notes"].replace("|", "\\|")
        lines.append(f"| {row['requirement']} | `{row['status']}` | {evidence} | {notes} |")
    lines.append("")
    lines.append("## Deferred V2 Items")
    lines.append("")
    lines.append("- Human-label alignment metrics: fill `human_label` in the single-expert sheet or `annotator1_label`/`annotator2_label` plus adjudication fields in the two-annotator workbook, then rerun the sync and evaluation scripts before reporting Precision@1, MRR, or Cohen's kappa.")
    lines.append("- EDINET: keep Japanese disclosure collection out of v1 unless an API key, archive coverage for older planned dates, and nonzero document/current-report rows are available.")
    lines.append("- SEC full-text validation: keep full all-8-K filing-body validation out of v1 until full body-text scope, storage, SEC/EDGAR user-agent configuration, and final standalone U.S. validation are added.")
    return "\n".join(lines) + "\n"


def write_markdown(audit: dict, path: Path) -> None:
    path.write_text(render_markdown(audit), encoding="utf-8")


def check_existing_outputs(audit: dict) -> list[dict]:
    expected_json = json.dumps(audit, indent=2, ensure_ascii=True)
    expected_markdown = render_markdown(audit)
    checks = []
    for label, path, expected in [
        ("json", ROOT / "paper4_completion_audit.json", expected_json),
        ("markdown", ROOT / "paper4_completion_audit.md", expected_markdown),
    ]:
        actual = path.read_text(encoding="utf-8") if path.exists() else None
        checks.append(
            {
                "output": label,
                "path": path.as_posix(),
                "exists": path.exists(),
                "matches_recomputed": actual == expected,
            }
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify the Paper 4 completion audit.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Recompute the audit and compare existing JSON/Markdown outputs without rewriting them.",
    )
    args = parser.parse_args()

    audit = build_audit()
    if args.check_only:
        checks = check_existing_outputs(audit)
        ok = all(row["exists"] and row["matches_recomputed"] for row in checks)
        print(
            json.dumps(
                {
                    "status": "pass" if ok else "fail",
                    "mode": "check_only",
                    "overall_status": audit["overall_status"],
                    "status_counts": audit["status_counts"],
                    "output_integrity": checks,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(0 if ok else 1)

    (ROOT / "paper4_completion_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    write_markdown(audit, ROOT / "paper4_completion_audit.md")
    print(json.dumps({"overall_status": audit["overall_status"], "status_counts": audit["status_counts"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
