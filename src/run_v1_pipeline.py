from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("Paper 4")
RUN_REPORT_AUDIT_JSON = ROOT / "results" / "reproducibility_run_report_audit.json"
RUN_REPORT_AUDIT_MD = ROOT / "results" / "reproducibility_run_report_audit.md"
CORE_PACKAGE_AUDIT_JSON = ROOT / "submission_package" / "paper4_core_package_latest_audit.json"
EMAIL_HANDOFF_AUDIT_JSON = ROOT / "submission_package" / "email_handoff_receipt_audit.json"
POST_RUN_STATUS_JSON = ROOT / "results" / "post_run_artifact_status.json"


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    outputs: list[str]
    heavy: bool = False
    freshness_exempt_outputs: tuple[str, ...] = ()


STEPS = [
    Step(
        "desktop_data_audit",
        [sys.executable, str(ROOT / "src" / "audit" / "data_audit.py")],
        [
            "audit_report.md",
            "data/audit/data_inventory.csv",
            "data/audit/summary.json",
        ],
        heavy=True,
    ),
    Step(
        "edinet_collection_plan",
        [sys.executable, str(ROOT / "src" / "collect" / "edinet_collector.py")],
        [
            "data/raw/edinet/edinet_collection_plan.csv",
            "data/raw/edinet/edinet_documents_index.csv",
            "results/edinet_collection_status.md",
            "results/edinet_collection_status.json",
        ],
    ),
    Step(
        "edinet_api_feasibility",
        [sys.executable, str(ROOT / "src" / "collect" / "edinet_feasibility.py")],
        [
            "data/raw/edinet/edinet_collection_plan_feasibility.csv",
            "results/edinet_api_feasibility.md",
            "results/edinet_api_feasibility.json",
        ],
    ),
    Step(
        "edinet_local_discovery",
        [sys.executable, str(ROOT / "src" / "audit" / "edinet_local_discovery.py")],
        [
            "data/audit/edinet_local_candidate_files.csv",
            "results/edinet_local_discovery.md",
            "results/edinet_local_discovery.json",
        ],
    ),
    Step(
        "edinet_execution_preflight",
        [sys.executable, str(ROOT / "src" / "audit" / "edinet_execution_preflight.py")],
        [
            "results/edinet_execution_preflight.md",
            "results/edinet_execution_preflight.json",
        ],
    ),
    Step(
        "jp_report_parser_prototype",
        [sys.executable, str(ROOT / "src" / "prepare" / "jp_report_parser_prototype.py")],
        [
            "data/raw/edinet/jp_report_parser_prototype_index.csv",
            "data/processed/jp_report_parser_prototype_events.csv",
            "data/processed/jp_report_parser_prototype_events.parquet",
            "results/jp_report_parser_prototype_summary.md",
            "results/jp_report_parser_prototype_summary.json",
        ],
    ),
    Step(
        "t04_local_discovery",
        [sys.executable, str(ROOT / "src" / "audit" / "t04_local_discovery.py")],
        [
            "data/audit/t04_local_candidate_files.csv",
            "results/t04_local_discovery.md",
            "results/t04_local_discovery.json",
        ],
    ),
    Step(
        "dart_gap_outcome_derivation",
        [sys.executable, str(ROOT / "src" / "collect" / "dart_gap_collector.py")],
        [
            "data/raw/dart_gap/dart_gap_outcome_candidates.csv",
            "results/dart_gap_collection_status.md",
            "results/dart_gap_collection_status.json",
        ],
    ),
    Step(
        "phase1_events",
        [sys.executable, str(ROOT / "src" / "prepare" / "build_phase1_events.py")],
        [
            "data/processed/events_dart.parquet",
            "data/processed/outcomes_dart.parquet",
            "data/processed/events_sec.parquet",
            "data/processed/outcomes_sec.parquet",
        ],
    ),
    Step(
        "sec_gap_inventory",
        [sys.executable, str(ROOT / "src" / "audit" / "sec_gap_inventory.py")],
        [
            "results/sec_gap_inventory.md",
            "results/sec_gap_inventory.csv",
            "results/sec_gap_inventory.json",
        ],
    ),
    Step(
        "sec_expanded_collection_plan",
        [sys.executable, str(ROOT / "src" / "collect" / "sec_gap_collector.py")],
        [
            "data/raw/sec_expanded/sec_expanded_collection_plan.csv",
            "data/raw/sec_expanded/sec_expanded_8k_filings.csv",
            "results/sec_expanded_collection_status.md",
            "results/sec_expanded_collection_status.json",
        ],
        freshness_exempt_outputs=(
            "data/raw/sec_expanded/sec_expanded_collection_plan.csv",
            "data/raw/sec_expanded/sec_expanded_8k_filings.csv",
        ),
    ),
    Step(
        "sec_expanded_sequences",
        [sys.executable, str(ROOT / "src" / "sequence" / "sec_expanded_sequences.py")],
        [
            "data/processed/sec_expanded_events.parquet",
            "data/processed/sec_expanded_outcomes.parquet",
            "data/processed/sec_expanded_sequence_panel.parquet",
            "results/sec_expanded_sequence_summary.md",
            "results/sec_expanded_sequence_summary.json",
        ],
    ),
    Step(
        "sec_bankruptcy_case_collection",
        [sys.executable, str(ROOT / "src" / "collect" / "sec_bankruptcy_case_collector.py")],
        [
            "data/raw/sec_bankruptcy_cases/sec_bankruptcy_case_filings.csv",
            "data/raw/sec_bankruptcy_cases/sec_bankruptcy_case_item103.csv",
            "results/sec_bankruptcy_case_status.md",
            "results/sec_bankruptcy_case_status.json",
        ],
        freshness_exempt_outputs=(
            "data/raw/sec_bankruptcy_cases/sec_bankruptcy_case_filings.csv",
            "data/raw/sec_bankruptcy_cases/sec_bankruptcy_case_item103.csv",
        ),
    ),
    Step(
        "sec_bankruptcy_case_sequences",
        [
            sys.executable,
            str(ROOT / "src" / "sequence" / "sec_expanded_sequences.py"),
            "--raw-path",
            str(ROOT / "data" / "raw" / "sec_bankruptcy_cases" / "sec_bankruptcy_case_filings.csv"),
            "--prefix",
            "sec_bankruptcy_case",
            "--source-name",
            "SEC_BANKRUPTCY_CASE",
            "--title",
            "SEC Bankruptcy Case Sequence Summary",
            "--description",
            "This file converts a targeted SEC Item 1.03 case sample into the same taxonomy-sequence structure used by Paper 4. It is designed for U.S. bankruptcy sequence case evidence, not population-representative validation.",
        ],
        [
            "data/processed/sec_bankruptcy_case_events.parquet",
            "data/processed/sec_bankruptcy_case_outcomes.parquet",
            "data/processed/sec_bankruptcy_case_sequence_panel.parquet",
            "results/sec_bankruptcy_case_sequence_summary.md",
            "results/sec_bankruptcy_case_sequence_summary.json",
        ],
    ),
    Step(
        "sec_bankruptcy_case_examples",
        [sys.executable, str(ROOT / "src" / "sequence" / "sec_bankruptcy_case_examples.py")],
        [
            "results/sec_bankruptcy_case_sequences.md",
            "results/sec_bankruptcy_case_sequences.csv",
        ],
    ),
    Step(
        "sec_submissions_archive_coverage",
        [sys.executable, str(ROOT / "src" / "collect" / "sec_submissions_archive_collector.py")],
        [
            "data/raw/sec_submissions_archive/sec_submissions_archive_manifest.csv",
            "results/sec_submissions_archive_status.md",
            "results/sec_submissions_archive_status.json",
        ],
    ),
    Step(
        "sec_desktop_population_collection",
        [sys.executable, str(ROOT / "src" / "collect" / "sec_desktop_population_collector.py")],
        [
            "data/raw/sec_desktop_population/sec_desktop_population_8k_filings.csv",
            "data/raw/sec_desktop_population/sec_desktop_population_outcome_candidates.csv",
            "results/sec_desktop_population_collection_status.md",
            "results/sec_desktop_population_collection_status.json",
        ],
        freshness_exempt_outputs=(
            "data/raw/sec_desktop_population/sec_desktop_population_8k_filings.csv",
            "data/raw/sec_desktop_population/sec_desktop_population_outcome_candidates.csv",
        ),
    ),
    Step(
        "sec_desktop_population_sequences",
        [
            sys.executable,
            str(ROOT / "src" / "sequence" / "sec_expanded_sequences.py"),
            "--raw-path",
            str(ROOT / "data" / "raw" / "sec_desktop_population" / "sec_desktop_population_8k_filings.csv"),
            "--prefix",
            "sec_desktop_population",
            "--source-name",
            "SEC_DESKTOP_POPULATION",
            "--title",
            "Local SEC Population Sequence Summary",
            "--description",
            "This file converts locally cached SEC submissions metadata, including cached archive submissions JSON where available, into the same taxonomy-sequence structure used by Paper 4. It is a broad local population-style v2 support artifact with no new SEC API calls; use it separately from the DART-first v1 claim.",
        ],
        [
            "data/processed/sec_desktop_population_events.parquet",
            "data/processed/sec_desktop_population_outcomes.parquet",
            "data/processed/sec_desktop_population_sequence_panel.parquet",
            "results/sec_desktop_population_sequence_summary.md",
            "results/sec_desktop_population_sequence_summary.json",
        ],
    ),
    Step(
        "sec_desktop_population_bankruptcy_examples",
        [
            sys.executable,
            str(ROOT / "src" / "sequence" / "sec_bankruptcy_case_examples.py"),
            "--events-path",
            str(ROOT / "data" / "processed" / "sec_desktop_population_events.parquet"),
            "--output-prefix",
            "sec_desktop_population_bankruptcy_sequences",
            "--title",
            "Local SEC Population Bankruptcy Sequences",
            "--description",
            "This file lists SEC Item 1.03 bankruptcy/receivership firms found in the local SEC metadata scan and the trailing regulatory-event sequence within {window_days} days before the first Item 1.03 filing.",
            "--interpretation",
            "These cases come from a broad local SEC metadata scan rather than the targeted bankruptcy seed list, with cached archive submissions metadata included where available. Describe them as population-style v2 evidence, not a finalized U.S. predictive validation design.",
        ],
        [
            "results/sec_desktop_population_bankruptcy_sequences.md",
            "results/sec_desktop_population_bankruptcy_sequences.csv",
        ],
    ),
    Step(
        "sec_desktop_sequence_validation",
        [sys.executable, str(ROOT / "src" / "model" / "sec_desktop_sequence_validation.py")],
        [
            "data/model_sec_desktop/sec_desktop_validation_panel.parquet",
            "results/sec_desktop_validation_summary.md",
            "results/sec_desktop_validation_metrics.csv",
            "results/sec_desktop_validation_predictions.csv",
        ],
        heavy=True,
    ),
    Step(
        "sec_desktop_validation_robustness",
        [sys.executable, str(ROOT / "src" / "model" / "sec_desktop_validation_robustness.py")],
        [
            "results/sec_desktop_validation_firm_disjoint_summary.md",
            "results/sec_desktop_validation_firm_disjoint_metrics.csv",
            "results/sec_desktop_validation_firm_disjoint_predictions.csv",
        ],
        heavy=True,
    ),
    Step(
        "sec_desktop_financial_baseline",
        [sys.executable, str(ROOT / "src" / "model" / "sec_desktop_financial_baseline.py")],
        [
            "data/model_sec_desktop/sec_desktop_financial_panel.parquet",
            "results/sec_desktop_financial_baseline_summary.md",
            "results/sec_desktop_financial_baseline_metrics.csv",
            "results/sec_desktop_financial_baseline_predictions.csv",
        ],
        heavy=True,
    ),
    Step(
        "sec_filing_text_manifest",
        [sys.executable, str(ROOT / "src" / "collect" / "sec_filing_text_manifest.py")],
        [
            "data/raw/sec_filing_text/sec_filing_text_manifest.csv",
            "results/sec_filing_text_collection_status.md",
            "results/sec_filing_text_collection_status.json",
        ],
    ),
    Step(
        "sec_all8k_text_scope_plan",
        [sys.executable, str(ROOT / "src" / "audit" / "sec_all8k_text_plan.py")],
        [
            "data/raw/sec_all8k_text/sec_all8k_text_manifest.csv",
            "data/raw/sec_all8k_text/sec_all8k_text_batch_plan.csv",
            "results/sec_all8k_text_scope_plan.md",
            "results/sec_all8k_text_scope_plan.json",
        ],
    ),
    Step(
        "sec_all8k_text_batch_status",
        [sys.executable, str(ROOT / "src" / "collect" / "sec_all8k_text_batch_collector.py")],
        [
            "data/raw/sec_all8k_text/sec_all8k_text_batch_status.csv",
            "results/sec_all8k_text_batch_collection_status.md",
            "results/sec_all8k_text_batch_collection_status.json",
        ],
    ),
    Step(
        "sec_all8k_text_sample_design",
        [sys.executable, str(ROOT / "src" / "audit" / "sec_all8k_text_sample_design.py")],
        [
            "data/raw/sec_all8k_text/sec_all8k_text_sample_plan.csv",
            "data/raw/sec_all8k_text/sec_all8k_text_sample_strata.csv",
            "results/sec_all8k_text_sample_design.md",
            "results/sec_all8k_text_sample_design.json",
        ],
    ),
    Step(
        "sec_all8k_sample_text_status",
        [sys.executable, str(ROOT / "src" / "collect" / "sec_all8k_sample_text_collector.py")],
        [
            "data/raw/sec_all8k_text/sec_all8k_text_sample_status.csv",
            "results/sec_all8k_text_sample_collection_status.md",
            "results/sec_all8k_text_sample_collection_status.json",
        ],
    ),
    Step(
        "sec_all8k_sample_text_baseline",
        [sys.executable, str(ROOT / "src" / "model" / "sec_all8k_sample_text_baseline.py")],
        [
            "data/model_sec_desktop/sec_all8k_sample_text_features.csv",
            "data/model_sec_desktop/sec_all8k_sample_text_features.parquet",
            "results/sec_all8k_sample_text_baseline_summary.md",
            "results/sec_all8k_sample_text_baseline_metrics.csv",
            "results/sec_all8k_sample_text_baseline_predictions.csv",
        ],
    ),
    Step(
        "sec_desktop_text_metadata_baseline",
        [sys.executable, str(ROOT / "src" / "model" / "sec_desktop_text_metadata_baseline.py")],
        [
            "data/model_sec_desktop/sec_desktop_text_metadata_panel.parquet",
            "results/sec_desktop_text_metadata_baseline_summary.md",
            "results/sec_desktop_text_metadata_baseline_metrics.csv",
            "results/sec_desktop_text_metadata_baseline_predictions.csv",
        ],
        heavy=True,
    ),
    Step(
        "sec_filing_body_text_baseline",
        [sys.executable, str(ROOT / "src" / "model" / "sec_filing_body_text_baseline.py")],
        [
            "data/model_sec_desktop/sec_filing_body_text_panel.parquet",
            "results/sec_filing_body_text_baseline_summary.md",
            "results/sec_filing_body_text_baseline_metrics.csv",
            "results/sec_filing_body_text_baseline_predictions.csv",
        ],
        heavy=True,
    ),
    Step(
        "sec_companyfacts_availability",
        [sys.executable, str(ROOT / "src" / "audit" / "sec_companyfacts_availability.py")],
        [
            "data/model_sec_desktop/sec_companyfacts_file_inventory.csv",
            "data/model_sec_desktop/sec_companyfacts_validation_coverage.csv",
            "results/sec_companyfacts_availability.md",
            "results/sec_companyfacts_availability.json",
        ],
    ),
    Step(
        "sec_desktop_sampling_frame",
        [sys.executable, str(ROOT / "src" / "model" / "sec_desktop_sampling_frame.py")],
        [
            "data/model_sec_desktop/sec_desktop_sampling_frame_firms.csv",
            "data/model_sec_desktop/sec_desktop_sampling_frame_anchors.csv",
            "results/sec_desktop_sampling_frame_summary.md",
            "results/sec_desktop_sampling_frame_summary.json",
        ],
    ),
    Step(
        "sec_validation_scope",
        [sys.executable, str(ROOT / "src" / "audit" / "sec_validation_scope.py")],
        [
            "data/model_sec_desktop/sec_validation_scope_decision_table.csv",
            "results/sec_validation_scope.md",
            "results/sec_validation_scope.json",
        ],
    ),
    Step(
        "sequence_panels",
        [sys.executable, str(ROOT / "src" / "sequence" / "build_sequence_panel.py")],
        [
            "data/processed/sequence_panel_dart.parquet",
            "data/processed/sequence_panel_sec.parquet",
            "results/preliminary_flagging_metrics.md",
        ],
    ),
    Step(
        "alignment_layer",
        [sys.executable, str(ROOT / "src" / "align" / "taxonomy_builder.py")],
        [
            "data/processed/regulatory_event_taxonomy.csv",
            "data/processed/events_aligned.parquet",
            "data/processed/manual_alignment_seed_100.csv",
            "results/alignment_summary.md",
        ],
    ),
    Step(
        "llm_alignment_prompt_export",
        [sys.executable, str(ROOT / "src" / "align" / "llm_aligner.py")],
        [
            "data/processed/llm_alignment_prompts.jsonl",
            "data/processed/llm_alignment_results.csv",
            "results/llm_alignment_summary.md",
            "results/llm_alignment_run.json",
        ],
    ),
    Step(
        "alignment_annotation_qc",
        [sys.executable, str(ROOT / "src" / "align" / "alignment_annotation_qc.py")],
        [
            "data/processed/manual_alignment_seed_100_annotation_priority.csv",
            "data/processed/manual_alignment_seed_100_first_pass_30.csv",
            "results/alignment_annotation_qc.md",
            "results/alignment_annotation_qc.json",
        ],
    ),
    Step(
        "alignment_candidate_diagnostics",
        [sys.executable, str(ROOT / "src" / "align" / "alignment_candidate_diagnostics.py")],
        [
            "data/processed/manual_alignment_seed_100_candidate_diagnostics.csv",
            "results/alignment_candidate_diagnostics.md",
            "results/alignment_candidate_diagnostics.json",
        ],
    ),
    Step(
        "alignment_silver_evaluation",
        [sys.executable, str(ROOT / "src" / "align" / "alignment_silver_evaluation.py")],
        [
            "data/processed/manual_alignment_seed_100_silver_review_queue.csv",
            "data/processed/manual_alignment_seed_100_silver_consensus.csv",
            "results/alignment_silver_evaluation.md",
            "results/alignment_silver_evaluation.json",
        ],
    ),
    Step(
        "sec_desktop_alignment_seed",
        [sys.executable, str(ROOT / "src" / "align" / "sec_desktop_alignment_seed.py")],
        [
            "data/processed/manual_alignment_seed_sec_desktop_100.csv",
            "data/processed/manual_alignment_seed_sec_desktop_100.xlsx",
            "results/alignment_sec_desktop_seed_summary.md",
            "results/alignment_sec_desktop_seed_summary.json",
            "manuscript/alignment_sec_desktop_labeling_guide.md",
        ],
    ),
    Step(
        "sec_desktop_alignment_evaluation",
        [
            sys.executable,
            str(ROOT / "src" / "align" / "evaluate_alignment.py"),
            "--seed-path",
            str(ROOT / "data" / "processed" / "manual_alignment_seed_sec_desktop_100.csv"),
            "--out-json",
            str(ROOT / "results" / "alignment_sec_desktop_evaluation.json"),
            "--out-md",
            str(ROOT / "results" / "alignment_sec_desktop_evaluation.md"),
        ],
        [
            "results/alignment_sec_desktop_evaluation.md",
            "results/alignment_sec_desktop_evaluation.json",
        ],
    ),
    Step(
        "alignment_single_expert_label_sheet",
        [sys.executable, str(ROOT / "src" / "align" / "build_single_expert_label_sheet.py")],
        [
            "data/processed/manual_alignment_seed_100_single_expert_label_sheet.csv",
            "data/processed/manual_alignment_seed_100_single_expert_label_sheet.xlsx",
            "results/alignment_single_expert_readiness.md",
            "results/alignment_single_expert_readiness.json",
            "manuscript/alignment_single_expert_labeling_guide.md",
        ],
    ),
    Step(
        "alignment_annotation_packet",
        [sys.executable, str(ROOT / "src" / "align" / "build_annotation_packet.py")],
        [
            "manuscript/alignment_annotation_guide.md",
            "data/processed/manual_alignment_seed_100_annotation_packet.csv",
        ],
    ),
    Step(
        "alignment_annotation_workbook",
        [sys.executable, str(ROOT / "src" / "align" / "build_annotation_workbook.py")],
        [
            "data/processed/manual_alignment_seed_100_annotation_workbook.xlsx",
            "results/alignment_annotation_readiness.md",
            "results/alignment_annotation_readiness.json",
        ],
    ),
    Step(
        "alignment_evaluation",
        [sys.executable, str(ROOT / "src" / "align" / "evaluate_alignment.py")],
        [
            "results/alignment_evaluation.md",
            "results/alignment_evaluation.json",
        ],
    ),
    Step(
        "alignment_label_preflight",
        [sys.executable, str(ROOT / "src" / "audit" / "alignment_label_preflight.py")],
        [
            "results/alignment_label_preflight.md",
            "results/alignment_label_preflight.json",
        ],
    ),
    Step(
        "dart_model_panel",
        [sys.executable, str(ROOT / "src" / "model" / "build_dart_model_dataset.py")],
        [
            "data/model/dart_model_panel.parquet",
            "data/model/dart_model_panel_summary.json",
        ],
    ),
    Step(
        "survival_hazard",
        [sys.executable, str(ROOT / "src" / "sequence" / "survival_model.py")],
        [
            "results/survival_hazard_summary.md",
            "results/survival_hazard_metrics.csv",
            "results/survival_hazard_coefficients.csv",
        ],
    ),
    Step(
        "dart_model_training",
        [sys.executable, str(ROOT / "src" / "model" / "train_dart_baselines.py")],
        [
            "results/dart_model_metrics.md",
            "results/dart_model_predictions.csv",
            "results/dart_model_bootstrap_ci.csv",
        ],
        heavy=True,
    ),
    Step(
        "market_coverage_collection",
        [sys.executable, str(ROOT / "src" / "collect" / "market_coverage_collector.py")],
        [
            "data/raw/market_coverage/naver_ohlcv_supplement.csv",
            "data/raw/market_coverage/naver_ohlcv_supplement_status.csv",
            "data/raw/market_coverage/naver_ohlcv_supplement_summary.md",
            "data/raw/market_coverage/naver_ohlcv_supplement_summary.json",
        ],
        heavy=True,
    ),
    Step(
        "market_fdr_gap_collection",
        [sys.executable, str(ROOT / "src" / "collect" / "market_fdr_gap_collector.py")],
        [
            "data/raw/market_coverage/fdr_ohlcv_gap_supplement.csv",
            "data/raw/market_coverage/fdr_ohlcv_gap_supplement_status.csv",
            "data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.md",
            "data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json",
        ],
        heavy=True,
    ),
    Step(
        "market_index_collection",
        [sys.executable, str(ROOT / "src" / "collect" / "market_index_collector.py")],
        [
            "data/raw/market_index/korea_market_indices.csv",
            "data/raw/market_index/korea_market_index_summary.md",
            "data/raw/market_index/korea_market_index_summary.json",
        ],
        heavy=True,
    ),
    Step(
        "market_feature_panel",
        [sys.executable, str(ROOT / "src" / "model" / "build_market_features.py")],
        [
            "data/model_market/dart_model_panel_market.parquet",
            "data/model_market/dart_model_panel_market.csv",
            "data/model_market/dart_model_panel_market_summary.json",
        ],
        heavy=True,
    ),
    Step(
        "market_model_training",
        [
            sys.executable,
            str(ROOT / "src" / "model" / "train_dart_baselines.py"),
            "--panel-path",
            str(ROOT / "data" / "model_market" / "dart_model_panel_market.parquet"),
            "--summary-path",
            str(ROOT / "data" / "model" / "dart_model_panel_summary.json"),
            "--results-dir",
            str(ROOT / "results" / "market"),
            "--n-bootstrap",
            "300",
        ],
        [
            "results/market/dart_model_metrics.md",
            "results/market/dart_model_predictions.csv",
            "results/market/dart_model_bootstrap_ci.csv",
        ],
        heavy=True,
    ),
    Step(
        "case_sequences",
        [sys.executable, str(ROOT / "src" / "model" / "extract_dart_case_sequences.py")],
        [
            "results/dart_distress_case_sequences.md",
            "results/dart_distress_cases.csv",
        ],
    ),
    Step(
        "robust_panel",
        [
            sys.executable,
            str(ROOT / "src" / "model" / "build_dart_model_dataset.py"),
            "--outcome-mode",
            "robust",
            "--out-dir",
            str(ROOT / "data" / "model_robust"),
        ],
        [
            "data/model_robust/dart_model_panel.parquet",
            "data/model_robust/dart_model_panel_summary.json",
        ],
    ),
    Step(
        "robust_training",
        [
            sys.executable,
            str(ROOT / "src" / "model" / "train_dart_baselines.py"),
            "--panel-path",
            str(ROOT / "data" / "model_robust" / "dart_model_panel.parquet"),
            "--summary-path",
            str(ROOT / "data" / "model_robust" / "dart_model_panel_summary.json"),
            "--results-dir",
            str(ROOT / "results" / "robust"),
        ],
        [
            "results/robust/dart_model_metrics.md",
            "results/robust/dart_model_predictions.csv",
        ],
        heavy=True,
    ),
    Step(
        "robust_market_feature_panel",
        [
            sys.executable,
            str(ROOT / "src" / "model" / "build_market_features.py"),
            "--panel-path",
            str(ROOT / "data" / "model_robust" / "dart_model_panel.parquet"),
            "--out-dir",
            str(ROOT / "data" / "model_market_robust"),
        ],
        [
            "data/model_market_robust/dart_model_panel_market.parquet",
            "data/model_market_robust/dart_model_panel_market.csv",
            "data/model_market_robust/dart_model_panel_market_summary.json",
        ],
        heavy=True,
    ),
    Step(
        "robust_market_training",
        [
            sys.executable,
            str(ROOT / "src" / "model" / "train_dart_baselines.py"),
            "--panel-path",
            str(ROOT / "data" / "model_market_robust" / "dart_model_panel_market.parquet"),
            "--summary-path",
            str(ROOT / "data" / "model_robust" / "dart_model_panel_summary.json"),
            "--results-dir",
            str(ROOT / "results" / "market_robust"),
            "--n-bootstrap",
            "300",
        ],
        [
            "results/market_robust/dart_model_metrics.md",
            "results/market_robust/dart_model_predictions.csv",
            "results/market_robust/dart_model_bootstrap_ci.csv",
        ],
        heavy=True,
    ),
    Step(
        "market_residual_coverage_audit",
        [sys.executable, str(ROOT / "src" / "audit" / "market_residual_coverage_audit.py")],
        [
            "results/market_residual_coverage_audit.md",
            "results/market_residual_coverage_audit.json",
            "data/audit/market_residual_symbol_impact.csv",
            "data/audit/market_residual_year_impact.csv",
        ],
    ),
    Step(
        "robust_cases",
        [
            sys.executable,
            str(ROOT / "src" / "model" / "extract_dart_case_sequences.py"),
            "--predictions-path",
            str(ROOT / "results" / "robust" / "dart_model_predictions.csv"),
            "--results-dir",
            str(ROOT / "results" / "robust"),
        ],
        [
            "results/robust/dart_distress_case_sequences.md",
        ],
    ),
    Step(
        "robust_summary",
        [sys.executable, str(ROOT / "src" / "model" / "summarize_robustness.py")],
        [
            "results/dart_robustness_summary.md",
            "results/dart_robustness_metrics.csv",
        ],
    ),
    Step(
        "permutation_importance",
        [sys.executable, str(ROOT / "src" / "model" / "permutation_importance.py")],
        [
            "results/dart_model_permutation_importance.md",
            "results/dart_model_permutation_importance.csv",
        ],
        heavy=True,
    ),
    Step(
        "pattern_mining",
        [sys.executable, str(ROOT / "src" / "sequence" / "pattern_miner.py")],
        [
            "results/pattern_rules.md",
            "results/pattern_rules.csv",
        ],
    ),
    Step(
        "flagging_rule_engine",
        [sys.executable, str(ROOT / "src" / "flagging" / "rule_engine.py")],
        [
            "results/flagging_rule_registry.csv",
            "results/flagging_rule_engine_summary.md",
            "results/flagging_rule_engine_flags.csv",
        ],
    ),
    Step(
        "flagging_evaluation",
        [sys.executable, str(ROOT / "src" / "flagging" / "evaluate_flagging.py")],
        [
            "results/flagging_performance.md",
            "results/flagging_performance.csv",
        ],
    ),
    Step(
        "classic_financial_baselines",
        [sys.executable, str(ROOT / "src" / "flagging" / "baseline_models.py")],
        [
            "results/classic_financial_baselines.md",
            "results/classic_financial_baselines.csv",
        ],
    ),
    Step(
        "calculation_sheet",
        [sys.executable, str(ROOT / "src" / "audit" / "build_calculation_sheet.py")],
        [
            "results/calculation_sheet.xlsx",
            "results/calculation_sheet.csv",
            "results/calculation_sheet.md",
            "results/calculation_sheet_audit.json",
        ],
    ),
    Step(
        "reference_consistency_audit",
        [sys.executable, str(ROOT / "src" / "audit" / "reference_consistency_audit.py")],
        [
            "results/reference_consistency_audit.md",
            "results/reference_consistency_audit.json",
        ],
    ),
    Step(
        "manuscript_metric_audit",
        [sys.executable, str(ROOT / "src" / "audit" / "manuscript_metric_audit.py")],
        [
            "results/manuscript_metric_audit.md",
            "results/manuscript_metric_audit.json",
        ],
    ),
    Step(
        "manuscript_claim_audit",
        [sys.executable, str(ROOT / "src" / "audit" / "manuscript_claim_audit.py")],
        [
            "results/manuscript_claim_audit.md",
            "results/manuscript_claim_audit.json",
        ],
    ),
    Step(
        "bilingual_manuscript_audit",
        [sys.executable, str(ROOT / "src" / "audit" / "bilingual_manuscript_audit.py")],
        [
            "results/bilingual_manuscript_audit.md",
            "results/bilingual_manuscript_audit.json",
        ],
    ),
    Step(
        "paper_figures",
        [sys.executable, str(ROOT / "src" / "audit" / "build_paper_figures.py")],
        [
            "figures/paper4_framework.png",
            "figures/paper4_model_performance.png",
            "figures/paper4_market_coverage.png",
            "results/figure_generation_audit.md",
            "results/figure_generation_audit.json",
        ],
    ),
    Step(
        "word_draft",
        [
            "pandoc",
            str(ROOT / "manuscript" / "paper4_v1_draft.md"),
            "--resource-path",
            str(ROOT),
            "-o",
            str(ROOT / "manuscript" / "paper4_v1_draft.docx"),
        ],
        [
            "manuscript/paper4_v1_draft.docx",
        ],
    ),
    Step(
        "word_manuscript_en",
        [
            "pandoc",
            str(ROOT / "manuscript" / "paper4_v1_manuscript_en.md"),
            "--resource-path",
            str(ROOT),
            "-o",
            str(ROOT / "manuscript" / "paper4_v1_manuscript_en.docx"),
        ],
        [
            "manuscript/paper4_v1_manuscript_en.docx",
        ],
    ),
    Step(
        "word_manuscript_ko",
        [
            "pandoc",
            str(ROOT / "manuscript" / "paper4_v1_manuscript_ko.md"),
            "--resource-path",
            str(ROOT),
            "-o",
            str(ROOT / "manuscript" / "paper4_v1_manuscript_ko.docx"),
        ],
        [
            "manuscript/paper4_v1_manuscript_ko.docx",
        ],
    ),
    Step(
        "word_submission_candidate_journals_ko",
        [
            "pandoc",
            str(ROOT / "manuscript" / "submission_candidate_journals_ko.md"),
            "-o",
            str(ROOT / "manuscript" / "submission_candidate_journals_ko.docx"),
        ],
        [
            "manuscript/submission_candidate_journals_ko.docx",
        ],
    ),
    Step(
        "word_export_audit",
        [sys.executable, str(ROOT / "src" / "audit" / "word_export_audit.py")],
        [
            "results/word_export_audit.md",
            "results/word_export_audit.json",
        ],
    ),
    Step(
        "journal_candidate_source_audit",
        [sys.executable, str(ROOT / "src" / "audit" / "journal_candidate_source_audit.py")],
        [
            "results/journal_candidate_source_audit.md",
            "results/journal_candidate_source_audit.json",
            "data/audit/submission_candidate_journal_sources.json",
        ],
        freshness_exempt_outputs=("data/audit/submission_candidate_journal_sources.json",),
    ),
    Step(
        "public_path_redaction",
        [sys.executable, str(ROOT / "src" / "audit" / "public_path_redaction.py")],
        [
            "results/public_path_redaction_audit.md",
            "results/public_path_redaction_audit.json",
        ],
    ),
    Step(
        "sec_all8k_text_batch_status_after_redaction",
        [sys.executable, str(ROOT / "src" / "collect" / "sec_all8k_text_batch_collector.py")],
        [
            "data/raw/sec_all8k_text/sec_all8k_text_batch_status.csv",
            "results/sec_all8k_text_batch_collection_status.md",
            "results/sec_all8k_text_batch_collection_status.json",
        ],
    ),
    Step(
        "sec_all8k_storage_preflight",
        [sys.executable, str(ROOT / "src" / "audit" / "sec_all8k_storage_preflight.py")],
        [
            "results/sec_all8k_storage_preflight.md",
            "results/sec_all8k_storage_preflight.json",
        ],
    ),
    Step(
        "sec_all8k_text_sample_design_after_redaction",
        [sys.executable, str(ROOT / "src" / "audit" / "sec_all8k_text_sample_design.py")],
        [
            "data/raw/sec_all8k_text/sec_all8k_text_sample_plan.csv",
            "data/raw/sec_all8k_text/sec_all8k_text_sample_strata.csv",
            "results/sec_all8k_text_sample_design.md",
            "results/sec_all8k_text_sample_design.json",
        ],
    ),
    Step(
        "sec_all8k_text_sample_status_after_redaction",
        [sys.executable, str(ROOT / "src" / "collect" / "sec_all8k_sample_text_collector.py")],
        [
            "data/raw/sec_all8k_text/sec_all8k_text_sample_status.csv",
            "results/sec_all8k_text_sample_collection_status.md",
            "results/sec_all8k_text_sample_collection_status.json",
        ],
    ),
    Step(
        "sec_all8k_sample_text_baseline_after_redaction",
        [sys.executable, str(ROOT / "src" / "model" / "sec_all8k_sample_text_baseline.py")],
        [
            "data/model_sec_desktop/sec_all8k_sample_text_features.csv",
            "data/model_sec_desktop/sec_all8k_sample_text_features.parquet",
            "results/sec_all8k_sample_text_baseline_summary.md",
            "results/sec_all8k_sample_text_baseline_metrics.csv",
            "results/sec_all8k_sample_text_baseline_predictions.csv",
        ],
    ),
    Step(
        "sec_validation_scope_after_redaction",
        [sys.executable, str(ROOT / "src" / "audit" / "sec_validation_scope.py")],
        [
            "data/model_sec_desktop/sec_validation_scope_decision_table.csv",
            "results/sec_validation_scope.md",
            "results/sec_validation_scope.json",
        ],
    ),
    Step(
        "submission_package_audit",
        [sys.executable, str(ROOT / "src" / "audit" / "submission_package_audit.py")],
        [
            "results/submission_package_audit.md",
            "results/submission_package_audit.json",
        ],
    ),
    Step(
        "research_plan_traceability",
        [sys.executable, str(ROOT / "src" / "audit" / "research_plan_traceability.py")],
        [
            "results/research_plan_traceability.md",
            "results/research_plan_traceability.json",
        ],
    ),
    Step(
        "completion_audit",
        [sys.executable, str(ROOT / "src" / "audit" / "completion_audit.py")],
        [
            "paper4_completion_audit.md",
            "paper4_completion_audit.json",
        ],
    ),
    Step(
        "public_submission_manifest_audit",
        [sys.executable, str(ROOT / "src" / "audit" / "public_submission_manifest_audit.py")],
        [
            "results/public_submission_manifest_audit.md",
            "results/public_submission_manifest_audit.json",
            "data/audit/public_submission_manifest.csv",
        ],
    ),
    Step(
        "remaining_risk_audit",
        [sys.executable, str(ROOT / "src" / "audit" / "remaining_risk_audit.py")],
        [
            "results/remaining_risk_audit.md",
            "results/remaining_risk_audit.json",
        ],
    ),
    Step(
        "final_claim_boundary_audit",
        [sys.executable, str(ROOT / "src" / "audit" / "final_claim_boundary_audit.py")],
        [
            "results/final_claim_boundary_audit.md",
            "results/final_claim_boundary_audit.json",
        ],
    ),
]


def output_state(
    outputs: list[str],
    started_at: float | None = None,
    freshness_exempt_outputs: tuple[str, ...] = (),
) -> list[dict]:
    rows = []
    freshness_exempt = set(freshness_exempt_outputs)
    for item in outputs:
        path = ROOT / item
        exists = path.exists()
        size = int(path.stat().st_size) if exists else 0
        mtime = float(path.stat().st_mtime) if exists else None
        freshness_required = started_at is not None and item not in freshness_exempt
        rows.append(
            {
                "path": item,
                "exists": exists,
                "size": size,
                "mtime": mtime,
                "fresh": None if not freshness_required or not exists else mtime >= started_at,
                "freshness_exempt": item in freshness_exempt,
            }
        )
    return rows


def output_failures(outputs: list[dict]) -> list[dict]:
    failures = []
    for row in outputs:
        reasons = []
        if not row["exists"]:
            reasons.append("missing")
        if row["exists"] and row["size"] <= 0:
            reasons.append("empty")
        if row.get("fresh") is False:
            reasons.append("stale")
        if reasons:
            failures.append({"path": row["path"], "reasons": reasons})
    return failures


def run_step(step: Step, execute: bool, include_heavy: bool) -> dict:
    if step.heavy and not include_heavy:
        return {
            "name": step.name,
            "status": "skipped_heavy",
            "command": step.command,
            "outputs": output_state(step.outputs),
        }
    if not execute:
        return {
            "name": step.name,
            "status": "dry_run",
            "command": step.command,
            "outputs": output_state(step.outputs),
        }
    started_at = time.time() - 1.0
    proc = subprocess.run(step.command, text=True, capture_output=True)
    outputs = output_state(
        step.outputs,
        started_at=started_at,
        freshness_exempt_outputs=step.freshness_exempt_outputs,
    )
    failures = output_failures(outputs)
    status = "ok"
    if proc.returncode != 0:
        status = "failed"
    elif failures:
        status = "failed_outputs"
    return {
        "name": step.name,
        "status": status,
        "returncode": proc.returncode,
        "command": step.command,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "outputs": outputs,
        "output_failures": failures,
    }


def write_manifest(path: Path) -> None:
    rows = [
        {
            "name": step.name,
            "heavy": step.heavy,
            "command": step.command,
            "outputs": step.outputs,
            "freshness_exempt_outputs": list(step.freshness_exempt_outputs),
        }
        for step in STEPS
    ]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize_results(results: list[dict]) -> dict:
    status_counts: dict[str, int] = {}
    output_failure_count = 0
    freshness_exempt_output_count = 0
    for row in results:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        output_failure_count += len(row.get("output_failures") or [])
        for output in row.get("outputs") or []:
            if output.get("freshness_exempt"):
                freshness_exempt_output_count += 1
    failed_steps = [
        row.get("name", "")
        for row in results
        if row.get("status") not in {"ok", "skipped_heavy"}
    ]
    return {
        "step_count": len(results),
        "status_counts": status_counts,
        "ok_count": status_counts.get("ok", 0),
        "skipped_heavy_count": status_counts.get("skipped_heavy", 0),
        "failed_step_count": len(failed_steps),
        "failed_steps": failed_steps,
        "output_failure_count": output_failure_count,
        "freshness_exempt_output_count": freshness_exempt_output_count,
    }


def build_run_report_audit(report: dict, report_path: Path) -> dict:
    steps = report.get("steps", [])
    manifest_path = Path(report.get("manifest_path", ROOT / "reproducibility_manifest.json"))
    manifest = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_step_count = len(manifest) if isinstance(manifest, list) else 0
    heavy_count = sum(1 for step in manifest if step.get("heavy")) if isinstance(manifest, list) else 0
    expected_freshness_exempt_outputs = 0
    if isinstance(manifest, list):
        expected_freshness_exempt_outputs = sum(
            len(step.get("freshness_exempt_outputs") or [])
            for step in manifest
            if report.get("include_heavy") or not step.get("heavy")
        )
    expected_ok = manifest_step_count - heavy_count if not report.get("include_heavy") else manifest_step_count
    expected_skipped_heavy = heavy_count if not report.get("include_heavy") else 0
    summary = summarize_results(steps)
    failures: list[dict] = []
    checks = [
        ("report_exists", report_path.exists(), str(report_path)),
        ("execute_true", report.get("execute") is True, report.get("execute")),
        ("step_count_matches_manifest", summary["step_count"] == manifest_step_count, f"{summary['step_count']}/{manifest_step_count}"),
        ("ok_count_expected", summary["ok_count"] == expected_ok, f"{summary['ok_count']}/{expected_ok}"),
        (
            "skipped_heavy_count_expected",
            summary["skipped_heavy_count"] == expected_skipped_heavy,
            f"{summary['skipped_heavy_count']}/{expected_skipped_heavy}",
        ),
        ("failed_step_count_zero", summary["failed_step_count"] == 0, summary["failed_step_count"]),
        ("output_failure_count_zero", summary["output_failure_count"] == 0, summary["output_failure_count"]),
        (
            "freshness_exempt_outputs_match_manifest",
            summary["freshness_exempt_output_count"] == expected_freshness_exempt_outputs,
            f"{summary['freshness_exempt_output_count']}/{expected_freshness_exempt_outputs}",
        ),
    ]
    for name, ok, observed in checks:
        if not ok:
            failures.append({"check": name, "observed": observed})
    return {
        "status": "pass" if not failures else "fail",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "report_path": str(report_path),
        "manifest_path": str(manifest_path),
        "manifest_step_count": manifest_step_count,
        "manifest_heavy_count": heavy_count,
        "expected_ok": expected_ok,
        "expected_skipped_heavy": expected_skipped_heavy,
        "expected_freshness_exempt_outputs": expected_freshness_exempt_outputs,
        **summary,
        "failures": failures,
        "guardrail": "Post-run audit generated after reproducibility_run_report.json is written; verifies the current run report rather than the previous run report read by in-pipeline audits.",
    }


def write_run_report_audit(audit: dict) -> None:
    RUN_REPORT_AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RUN_REPORT_AUDIT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Reproducibility Run Report Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Checked at | `{audit['checked_at']}` |",
        f"| Manifest steps | {audit['manifest_step_count']} |",
        f"| Manifest heavy steps | {audit['manifest_heavy_count']} |",
        f"| Run steps | {audit['step_count']} |",
        f"| OK steps | {audit['ok_count']} / {audit['expected_ok']} |",
        f"| Skipped heavy steps | {audit['skipped_heavy_count']} / {audit['expected_skipped_heavy']} |",
        f"| Failed steps | {audit['failed_step_count']} |",
        f"| Output failures | {audit['output_failure_count']} |",
        f"| Freshness-exempt outputs | {audit['freshness_exempt_output_count']} / {audit['expected_freshness_exempt_outputs']} |",
        "",
        "## Failures",
        "",
    ]
    if audit["failures"]:
        lines.extend(["| Check | Observed |", "|---|---|"])
        for row in audit["failures"]:
            lines.append(f"| `{row['check']}` | `{row['observed']}` |")
    else:
        lines.append("No run-report audit failures.")
    lines.extend(["", "## Guardrail", "", audit["guardrail"]])
    RUN_REPORT_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_latest_core_package() -> dict:
    command = [
        sys.executable,
        str(ROOT / "src" / "audit" / "build_core_submission_package.py"),
        "--stamp",
        "latest",
    ]
    started_at = time.time()
    proc = subprocess.run(command, text=True, capture_output=True)
    duration_seconds = round(time.time() - started_at, 3)

    audit: dict = {}
    audit_load_error = ""
    if CORE_PACKAGE_AUDIT_JSON.exists():
        try:
            audit = json.loads(CORE_PACKAGE_AUDIT_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            audit_load_error = str(exc)

    failures: list[dict] = []
    if proc.returncode != 0:
        failures.append({"check": "builder_returncode_zero", "observed": proc.returncode})
    if not CORE_PACKAGE_AUDIT_JSON.exists():
        failures.append({"check": "audit_json_exists", "observed": str(CORE_PACKAGE_AUDIT_JSON)})
    if audit_load_error:
        failures.append({"check": "audit_json_loadable", "observed": audit_load_error})
    if audit and audit.get("status") != "pass":
        failures.append({"check": "package_audit_status_pass", "observed": audit.get("status")})

    return {
        "name": "core_submission_package_latest",
        "status": "pass" if not failures else "fail",
        "returncode": proc.returncode,
        "duration_seconds": duration_seconds,
        "command": command,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "audit_path": str(CORE_PACKAGE_AUDIT_JSON),
        "zip_path": audit.get("zip_path", ""),
        "zip_entries": audit.get("zip_entries"),
        "zip_sha256": audit.get("zip_sha256", ""),
        "failures": failures,
    }


def run_email_handoff_receipt_audit() -> dict:
    command = [
        sys.executable,
        str(ROOT / "src" / "audit" / "email_handoff_receipt_audit.py"),
    ]
    started_at = time.time()
    proc = subprocess.run(command, text=True, capture_output=True)
    duration_seconds = round(time.time() - started_at, 3)

    audit: dict = {}
    audit_load_error = ""
    if EMAIL_HANDOFF_AUDIT_JSON.exists():
        try:
            audit = json.loads(EMAIL_HANDOFF_AUDIT_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            audit_load_error = str(exc)

    failures: list[dict] = []
    if proc.returncode != 0:
        failures.append({"check": "handoff_audit_returncode_zero", "observed": proc.returncode})
    if not EMAIL_HANDOFF_AUDIT_JSON.exists():
        failures.append({"check": "handoff_audit_json_exists", "observed": str(EMAIL_HANDOFF_AUDIT_JSON)})
    if audit_load_error:
        failures.append({"check": "handoff_audit_json_loadable", "observed": audit_load_error})
    if audit and audit.get("status") != "pass":
        failures.append({"check": "handoff_audit_status_pass", "observed": audit.get("status")})

    return {
        "name": "email_handoff_receipt",
        "status": "pass" if not failures else "fail",
        "returncode": proc.returncode,
        "duration_seconds": duration_seconds,
        "command": command,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "audit_path": str(EMAIL_HANDOFF_AUDIT_JSON),
        "gmail_message_id": audit.get("gmail_message_id", ""),
        "attachment_filename": audit.get("attachment_filename", ""),
        "latest_zip_sha256": audit.get("latest_zip_sha256", ""),
        "failure_count": audit.get("failure_count"),
        "failures": failures,
    }


def run_post_run_artifact_status(require_email_handoff: bool) -> dict:
    command = [
        sys.executable,
        str(ROOT / "src" / "audit" / "post_run_artifact_status.py"),
    ]
    if require_email_handoff:
        command.append("--require-email-handoff")
    started_at = time.time()
    proc = subprocess.run(command, text=True, capture_output=True)
    duration_seconds = round(time.time() - started_at, 3)

    audit: dict = {}
    audit_load_error = ""
    if POST_RUN_STATUS_JSON.exists():
        try:
            audit = json.loads(POST_RUN_STATUS_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            audit_load_error = str(exc)

    failures: list[dict] = []
    if proc.returncode != 0:
        failures.append({"check": "post_run_status_returncode_zero", "observed": proc.returncode})
    if not POST_RUN_STATUS_JSON.exists():
        failures.append({"check": "post_run_status_json_exists", "observed": str(POST_RUN_STATUS_JSON)})
    if audit_load_error:
        failures.append({"check": "post_run_status_json_loadable", "observed": audit_load_error})
    if audit and audit.get("status") == "fail":
        failures.append({"check": "post_run_status_not_fail", "observed": audit.get("status")})

    return {
        "name": "post_run_artifact_status",
        "status": "pass" if not failures else "fail",
        "returncode": proc.returncode,
        "duration_seconds": duration_seconds,
        "command": command,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "audit_path": str(POST_RUN_STATUS_JSON),
        "artifact_status": audit.get("status", ""),
        "warning_count": audit.get("warning_count"),
        "failure_count": audit.get("failure_count"),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run commands. Without this flag, only report planned steps and output states.")
    parser.add_argument("--include-heavy", action="store_true", help="Include heavy audit/training/permutation steps.")
    parser.add_argument("--manifest-path", default=str(ROOT / "reproducibility_manifest.json"))
    parser.add_argument("--report-path", default=str(ROOT / "reproducibility_run_report.json"))
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    report_path = Path(args.report_path)
    write_manifest(manifest_path)

    results = [run_step(step, execute=args.execute, include_heavy=args.include_heavy) for step in STEPS]
    summary = summarize_results(results)
    report = {
        "execute": args.execute,
        "include_heavy": args.include_heavy,
        "manifest_path": str(manifest_path),
        **summary,
        "steps": results,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    run_report_audit = build_run_report_audit(report, report_path) if args.execute else None
    if run_report_audit is not None:
        write_run_report_audit(run_report_audit)

    if args.execute and any(row["status"] not in {"ok", "skipped_heavy"} for row in results):
        raise SystemExit(1)
    if run_report_audit is not None and run_report_audit["status"] != "pass":
        raise SystemExit(1)

    core_package = build_latest_core_package() if args.execute else None
    email_handoff = run_email_handoff_receipt_audit() if args.execute else None
    post_run_status = run_post_run_artifact_status(False) if args.execute else None
    print(
        json.dumps(
            {
                "steps": len(results),
                "execute": args.execute,
                "include_heavy": args.include_heavy,
                "post_run_email_handoff_mode": "report_only",
                "post_run_core_package": core_package,
                "post_run_email_handoff": email_handoff,
                "post_run_artifact_status": post_run_status,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if core_package is not None and core_package["status"] != "pass":
        raise SystemExit(1)
    if post_run_status is not None and post_run_status["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
