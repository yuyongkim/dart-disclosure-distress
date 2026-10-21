from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")


@dataclass(frozen=True)
class TraceItem:
    phase: str
    requirement: str
    status: str
    evidence: list[str]
    notes: str
    action_required: str


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def size(path: str) -> int:
    p = ROOT / path
    return int(p.stat().st_size) if p.exists() else 0


def csv_rows(path: str) -> int | None:
    p = ROOT / path
    if not p.exists():
        return None
    try:
        return int(len(pd.read_csv(p, low_memory=False)))
    except pd.errors.EmptyDataError:
        return 0


def json_value(path: str, *keys: str) -> Any:
    p = ROOT / path
    if not p.exists():
        return None
    value: Any = json.loads(p.read_text(encoding="utf-8"))
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def pct(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def status_from_evidence(evidence: list[str], ready_status: str = "complete") -> str:
    return ready_status if all(exists(path) for path in evidence) else "missing_artifacts"


def build_traceability() -> dict[str, Any]:
    alignment_labeled = json_value("results/alignment_evaluation.json", "labeled_rows") or 0
    alignment_status = json_value("results/alignment_evaluation.json", "status")
    alignment_label_preflight_status = json_value("results/alignment_label_preflight.json", "status")
    alignment_label_preflight_ready = json_value("results/alignment_label_preflight.json", "final_rq1_metrics_ready")
    alignment_label_preflight_blockers = json_value("results/alignment_label_preflight.json", "blockers") or []
    sec_desktop_eval_labeled = json_value("results/alignment_sec_desktop_evaluation.json", "labeled_rows") or 0
    edinet_document_rows = json_value("results/edinet_collection_status.json", "document_rows") or 0
    edinet_status = json_value("results/edinet_collection_status.json", "status")
    edinet_local_status = json_value("results/edinet_local_discovery.json", "status")
    edinet_local_external = json_value("results/edinet_local_discovery.json", "external_reusable_candidate_rows")
    edinet_jp_report_candidates = json_value(
        "results/edinet_local_discovery.json", "external_japanese_report_candidate_rows"
    )
    edinet_jp_report_edinet_mentions = json_value(
        "results/edinet_local_discovery.json", "external_japanese_report_edinet_mentions"
    )
    edinet_execution_preflight_status = json_value("results/edinet_execution_preflight.json", "status")
    edinet_execution_ready = json_value("results/edinet_execution_preflight.json", "empirical_collection_ready")
    edinet_execution_env_key = json_value("results/edinet_execution_preflight.json", "env_api_key_present")
    edinet_execution_current_rows = json_value("results/edinet_execution_preflight.json", "current_report_candidate_rows")
    edinet_execution_blockers = json_value("results/edinet_execution_preflight.json", "blockers") or []
    jp_report_proto_status = json_value("results/jp_report_parser_prototype_summary.json", "status")
    jp_report_proto_files = json_value("results/jp_report_parser_prototype_summary.json", "report_files")
    jp_report_proto_texts = json_value("results/jp_report_parser_prototype_summary.json", "text_files")
    jp_report_proto_events = json_value("results/jp_report_parser_prototype_summary.json", "taxonomy_candidate_rows")
    sec_all8k_missing = json_value("results/sec_all8k_text_batch_collection_status.json", "missing_text_files")
    sec_all8k_open_batches = json_value("results/sec_all8k_text_batch_collection_status.json", "open_batches")
    sec_all8k_downloaded = json_value("results/sec_all8k_text_batch_collection_status.json", "downloaded_files")
    sec_all8k_path_mode = json_value("results/sec_all8k_text_batch_collection_status.json", "path_resolution_mode")
    sec_all8k_redacted_paths = json_value("results/sec_all8k_text_batch_collection_status.json", "manifest_local_path_redacted_rows")
    sec_all8k_canonical_paths = json_value("results/sec_all8k_text_batch_collection_status.json", "canonical_text_path_rows")
    sec_all8k_log_downloaded = json_value("results/sec_all8k_text_batch_collection_status.json", "download_log_downloaded_files")
    sec_all8k_log_existing = json_value("results/sec_all8k_text_batch_collection_status.json", "download_log_existing_files")
    sec_all8k_preflight_status = json_value("results/sec_all8k_storage_preflight.json", "status")
    sec_all8k_preflight_required_gb = json_value("results/sec_all8k_storage_preflight.json", "required_free_gb")
    sec_all8k_preflight_storage_ready = json_value(
        "results/sec_all8k_storage_preflight.json",
        "storage_ready_for_full_scope",
    )
    sec_all8k_preflight_env_user_agent = json_value("results/sec_all8k_storage_preflight.json", "env_user_agent_present")
    sec_all8k_preflight_ready = json_value("results/sec_all8k_storage_preflight.json", "full_scope_download_ready")
    sec_all8k_preflight_blockers = json_value("results/sec_all8k_storage_preflight.json", "blockers") or []
    sec_scope_coverage = json_value("results/sec_validation_scope.json", "all_8k_text_scope_coverage_rate")
    metric_status = json_value("results/manuscript_metric_audit.json", "status")
    claim_status = json_value("results/manuscript_claim_audit.json", "status")
    claim_scanned_files = json_value("results/manuscript_claim_audit.json", "scanned_files") or []
    bilingual_status = json_value("results/bilingual_manuscript_audit.json", "status")
    word_export_status = json_value("results/word_export_audit.json", "status")
    word_export_documents = json_value("results/word_export_audit.json", "document_count")
    word_export_claim_blockers = json_value("results/word_export_audit.json", "claim_blocker_count")
    alignment_silver_status = json_value("results/alignment_silver_evaluation.json", "status")
    alignment_silver_consensus = json_value("results/alignment_silver_evaluation.json", "silver_consensus_rows")
    alignment_silver_review_needed = json_value("results/alignment_silver_evaluation.json", "review_needed_rows")
    alignment_silver_strict_rate = json_value("results/alignment_silver_evaluation.json", "strict_agreement_rate")
    dart_event_rows = json_value("data/processed/phase1_event_summary.json", "dart_events", "rows")
    dart_event_firms = json_value("data/processed/phase1_event_summary.json", "dart_events", "firms")
    dart_outcome_rows = json_value("data/processed/phase1_event_summary.json", "dart_outcomes", "rows")
    dart_outcome_firms = json_value("data/processed/phase1_event_summary.json", "dart_outcomes", "firms")

    trace_items = [
        TraceItem(
            phase="Data discovery",
            requirement="Scan current folder and external local source folders for reusable data",
            status=status_from_evidence(
                [
                    "src/audit/data_audit.py",
                    "data/audit/desktop_data_files_rg.txt",
                    "data/audit/data_inventory.csv",
                    "data/audit/summary.json",
                    "audit_report.md",
                ]
            ),
            evidence=[
                "src/audit/data_audit.py",
                "data/audit/desktop_data_files_rg.txt",
                "data/audit/data_inventory.csv",
                "data/audit/summary.json",
                "audit_report.md",
            ],
            notes=(
                f"Local inventory rows: {json_value('data/audit/summary.json', 'summary', 'total_files')}; "
                f"size MB: {json_value('data/audit/summary.json', 'summary', 'total_size_mb')}."
            ),
            action_required="None for v1; rerun only if new local data folders are added.",
        ),
        TraceItem(
            phase="Data layer",
            requirement="Build DART event and outcome layer for 2015-2024",
            status=status_from_evidence(
                [
                    "data/processed/events_dart.parquet",
                    "data/processed/outcomes_dart.parquet",
                    "data/raw/dart_gap/dart_gap_outcome_candidates.csv",
                    "results/dart_gap_collection_status.md",
                ]
            ),
            evidence=[
                "data/processed/events_dart.parquet",
                "data/processed/outcomes_dart.parquet",
                "data/raw/dart_gap/dart_gap_outcome_candidates.csv",
                "results/dart_gap_collection_status.md",
            ],
            notes=(
                f"DART event rows/firms: {dart_event_rows}/{dart_event_firms}; "
                f"DART outcome rows/firms: {dart_outcome_rows}/{dart_outcome_firms}."
            ),
            action_required="None for v1.",
        ),
        TraceItem(
            phase="Data layer",
            requirement="Build SEC 8-K event/outcome layer and v2 expansion evidence",
            status="phase_2_support_ready" if exists("results/sec_desktop_population_collection_status.md") else "missing_artifacts",
            evidence=[
                "data/processed/events_sec.parquet",
                "data/processed/outcomes_sec.parquet",
                "results/sec_gap_inventory.md",
                "results/sec_desktop_population_collection_status.md",
                "results/sec_desktop_population_sequence_summary.md",
                "results/sec_validation_scope.md",
            ],
            notes=(
                f"Local SEC 8-K rows: {json_value('results/sec_desktop_population_collection_status.json', 'filing_rows')}; "
                f"Item 1.03 rows: {json_value('results/sec_desktop_population_collection_status.json', 'item_counts', '1.03')}; "
                f"all-8-K body-text coverage: {pct(sec_scope_coverage)}."
            ),
            action_required="Keep as v2 support unless full all-8-K filing-body coverage and final U.S. validation scope are completed.",
        ),
        TraceItem(
            phase="Data layer",
            requirement="Prepare EDINET/Japanese disclosure extension path",
            status="deferred_v2_scope" if edinet_document_rows == 0 else "collected",
            evidence=[
                "src/collect/edinet_collector.py",
                "src/collect/edinet_feasibility.py",
                "src/audit/edinet_local_discovery.py",
                "src/audit/edinet_execution_preflight.py",
                "src/prepare/jp_report_parser_prototype.py",
                "results/edinet_collection_status.md",
                "results/edinet_api_feasibility.md",
                "results/edinet_local_discovery.md",
                "results/edinet_execution_preflight.md",
                "results/jp_report_parser_prototype_summary.md",
            ],
            notes=(
                f"EDINET collection status: {edinet_status}; document rows: {edinet_document_rows}; "
                f"local discovery status: {edinet_local_status}; external reusable EDINET candidates: {edinet_local_external}; "
                f"local Japanese report PDF/text candidates: {edinet_jp_report_candidates}; "
                f"Japanese report text files mentioning EDINET: {edinet_jp_report_edinet_mentions}. "
                f"Execution preflight: {edinet_execution_preflight_status}; empirical ready {edinet_execution_ready}; "
                f"env API key present {edinet_execution_env_key}; current-report rows {edinet_execution_current_rows}; "
                f"blockers {edinet_execution_blockers}. "
                f"Parser prototype status: {jp_report_proto_status}; report/text files: {jp_report_proto_files}/{jp_report_proto_texts}; "
                f"taxonomy candidate rows: {jp_report_proto_events}."
            ),
            action_required="Deferred to v2; requires EDINET_API_KEY, archive/prior-collection coverage for older planned dates, and nonzero current-report rows before any Japanese empirical claim.",
        ),
        TraceItem(
            phase="RQ1 alignment",
            requirement="Normalize event titles and build common DART-SEC-EDINET taxonomy",
            status=status_from_evidence(
                [
                    "src/align/taxonomy_builder.py",
                    "data/processed/regulatory_event_taxonomy.csv",
                    "data/processed/events_aligned.parquet",
                    "data/processed/event_dictionary.csv",
                    "results/alignment_summary.md",
                ]
            ),
            evidence=[
                "src/align/taxonomy_builder.py",
                "data/processed/regulatory_event_taxonomy.csv",
                "data/processed/events_aligned.parquet",
                "data/processed/event_dictionary.csv",
                "results/alignment_summary.md",
            ],
            notes="Deterministic taxonomy layer is available; EDINET remains registry/planning-only.",
            action_required="None for taxonomy coverage; EDINET rows are required for completed trilingual empirical coverage.",
        ),
        TraceItem(
            phase="RQ1 alignment",
            requirement="Prepare LLM/candidate alignment workflow",
            status=status_from_evidence(
                [
                    "src/align/llm_aligner.py",
                    "data/processed/llm_alignment_prompts.jsonl",
                    "data/processed/llm_alignment_results.csv",
                    "results/llm_alignment_summary.md",
                    "results/alignment_candidate_diagnostics.md",
                    "results/alignment_silver_evaluation.md",
                ]
            ),
            evidence=[
                "src/align/llm_aligner.py",
                "src/align/alignment_silver_evaluation.py",
                "data/processed/llm_alignment_prompts.jsonl",
                "data/processed/llm_alignment_results.csv",
                "results/llm_alignment_summary.md",
                "results/alignment_candidate_diagnostics.md",
                "results/alignment_silver_evaluation.md",
            ],
            notes=(
                "Prompt export and offline candidate judgments exist, but they are not human labels. "
                f"Silver pre-label evaluation status: {alignment_silver_status}; consensus rows: {alignment_silver_consensus}; "
                f"review-needed rows: {alignment_silver_review_needed}; strict proposed-vs-LLM agreement rate: {alignment_silver_strict_rate}."
            ),
            action_required="Use only as annotation preparation until human labels are filled.",
        ),
        TraceItem(
            phase="RQ1 alignment",
            requirement="Report Precision@1/MRR against a 100-row human gold standard",
            status="deferred_v2_scope" if int(alignment_labeled) == 0 else "partially_labeled",
            evidence=[
                "data/processed/manual_alignment_seed_100.csv",
                "data/processed/manual_alignment_seed_100_single_expert_label_sheet.xlsx",
                "data/processed/manual_alignment_seed_100_annotation_workbook.xlsx",
                "data/processed/manual_alignment_seed_100_silver_review_queue.csv",
                "results/alignment_evaluation.md",
                "results/alignment_evaluation.json",
                "results/alignment_silver_evaluation.md",
                "results/alignment_label_preflight.md",
                "results/alignment_label_preflight.json",
            ],
            notes=(
                f"Main alignment evaluator status: {alignment_status}; labeled rows: {alignment_labeled}. "
                f"Local SEC supplemental labeled rows: {sec_desktop_eval_labeled}. "
                f"Label preflight status: {alignment_label_preflight_status}; final metrics ready: "
                f"{alignment_label_preflight_ready}; blockers: {alignment_label_preflight_blockers}."
            ),
            action_required="Deferred to v2; fill human labels, sync the workbook/sheet, and rerun evaluate_alignment.py before reporting Precision@1 or MRR.",
        ),
        TraceItem(
            phase="RQ2 sequence patterns",
            requirement="Construct firm-level event sequences and mine common sequence patterns",
            status=status_from_evidence(
                [
                    "src/sequence/build_sequence_panel.py",
                    "data/processed/sequence_panel_dart.parquet",
                    "data/processed/sequence_panel_sec.parquet",
                    "src/sequence/pattern_miner.py",
                    "results/pattern_rules.md",
                    "results/pattern_rules.csv",
                ]
            ),
            evidence=[
                "src/sequence/build_sequence_panel.py",
                "data/processed/sequence_panel_dart.parquet",
                "data/processed/sequence_panel_sec.parquet",
                "src/sequence/pattern_miner.py",
                "results/pattern_rules.md",
                "results/pattern_rules.csv",
            ],
            notes="Contiguous taxonomy-code n-gram mining is implemented without external dependencies.",
            action_required="None for v1; PrefixSpan can be a later method extension if needed.",
        ),
        TraceItem(
            phase="RQ2 sequence patterns",
            requirement="Evaluate timing with a survival/time-to-event design",
            status=status_from_evidence(
                [
                    "src/sequence/survival_model.py",
                    "results/survival_hazard_summary.md",
                    "results/survival_hazard_metrics.csv",
                    "results/survival_hazard_coefficients.csv",
                ]
            ),
            evidence=[
                "src/sequence/survival_model.py",
                "results/survival_hazard_summary.md",
                "results/survival_hazard_metrics.csv",
                "results/survival_hazard_coefficients.csv",
            ],
            notes="Discrete-time complementary log-log hazard check is available; full Cox PH is documented as unstable.",
            action_required="None for v1.",
        ),
        TraceItem(
            phase="RQ3 SupTech flagging",
            requirement="Convert sequence patterns into supervisory flagging rules",
            status=status_from_evidence(
                [
                    "src/flagging/rule_engine.py",
                    "results/flagging_rule_registry.csv",
                    "results/flagging_rule_engine_summary.md",
                    "results/flagging_rule_engine_flags.csv",
                ]
            ),
            evidence=[
                "src/flagging/rule_engine.py",
                "results/flagging_rule_registry.csv",
                "results/flagging_rule_engine_summary.md",
                "results/flagging_rule_engine_flags.csv",
            ],
            notes="CRITICAL/HIGH/MEDIUM rule registry is generated and applied to firms.",
            action_required="None for v1.",
        ),
        TraceItem(
            phase="RQ3 SupTech flagging",
            requirement="Compare flags against financial baselines and model watchlists",
            status=status_from_evidence(
                [
                    "src/flagging/evaluate_flagging.py",
                    "results/flagging_performance.md",
                    "results/flagging_performance.csv",
                    "results/classic_financial_baselines.md",
                ]
            ),
            evidence=[
                "src/flagging/evaluate_flagging.py",
                "results/flagging_performance.md",
                "results/flagging_performance.csv",
                "results/classic_financial_baselines.md",
            ],
            notes="Rule-based triggers are weaker than the model watchlist but useful for auditability and case discovery.",
            action_required="None for v1.",
        ),
        TraceItem(
            phase="Prediction model",
            requirement="Train out-of-time sequence-only, financial-only, and combined models",
            status=status_from_evidence(
                [
                    "src/model/build_dart_model_dataset.py",
                    "src/model/train_dart_baselines.py",
                    "data/model/dart_model_panel.parquet",
                    "results/dart_model_metrics.md",
                    "results/dart_model_bootstrap_ci.csv",
                    "results/dart_robustness_summary.md",
                ]
            ),
            evidence=[
                "src/model/build_dart_model_dataset.py",
                "src/model/train_dart_baselines.py",
                "data/model/dart_model_panel.parquet",
                "results/dart_model_metrics.md",
                "results/dart_model_bootstrap_ci.csv",
                "results/dart_robustness_summary.md",
            ],
            notes="DART-first early-warning model is the verified v1 empirical backbone.",
            action_required="None for v1.",
        ),
        TraceItem(
            phase="SEC v2 full-text extension",
            requirement="Track full all-8-K filing-body scope before any SEC full-text validation claim",
            status="deferred_v2_scope",
            evidence=[
                "src/audit/sec_all8k_text_plan.py",
                "src/collect/sec_all8k_text_batch_collector.py",
                "data/raw/sec_all8k_text/sec_all8k_text_manifest.csv",
                "data/raw/sec_all8k_text/sec_all8k_text_batch_status.csv",
                "results/sec_all8k_text_scope_plan.md",
                "results/sec_all8k_text_batch_collection_status.md",
                "results/sec_all8k_storage_preflight.md",
            ],
            notes=(
                f"Missing all-8-K filing bodies: {sec_all8k_missing}; open batches: {sec_all8k_open_batches}; "
                f"downloads in latest default/status run: {sec_all8k_downloaded}; "
                f"historical smoke downloads/existing files: {sec_all8k_log_downloaded}/{sec_all8k_log_existing}; "
                f"path mode: {sec_all8k_path_mode}; redacted/canonical path rows: {sec_all8k_redacted_paths}/{sec_all8k_canonical_paths}; "
                f"full-scope preflight: {sec_all8k_preflight_status}, required free disk GB "
                f"{sec_all8k_preflight_required_gb}, storage ready "
                f"{sec_all8k_preflight_storage_ready}, explicit user agent present {sec_all8k_preflight_env_user_agent}, "
                f"download ready {sec_all8k_preflight_ready}, blockers {sec_all8k_preflight_blockers}."
            ),
            action_required="Requires controlled batch downloads with an explicit SEC user agent and enough storage for the full-scope preflight.",
        ),
        TraceItem(
            phase="Manuscript readiness",
            requirement="Draft manuscript, references, submission strategy, target-journal routing, and claim controls",
            status=status_from_evidence(
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
                    "manuscript/claim_readiness_matrix.md",
                    "src/audit/audit_forbidden_tokens.py",
                    "results/manuscript_metric_audit.md",
                    "results/manuscript_claim_audit.md",
                    "results/bilingual_manuscript_audit.md",
                    "results/word_export_audit.md",
                    "results/submission_package_audit.md",
                ]
            ),
            evidence=[
                "manuscript/paper4_v4_manuscript_en.md",
                "manuscript/paper4_v4_manuscript_en.docx",
                "manuscript/paper4_v4_manuscript_ko.md",
                "manuscript/paper4_v4_manuscript_ko.docx",
                "manuscript/paper4_v4_title_page.md",
                "manuscript/paper4_v4_title_page.docx",
                "manuscript/references.bib",
                "manuscript/submission_strategy.md",
                "manuscript/paper4_target_journal_expansion_ko.md",
                "manuscript/claim_readiness_matrix.md",
                "src/audit/audit_forbidden_tokens.py",
                "results/manuscript_metric_audit.md",
                "results/manuscript_claim_audit.md",
                "results/bilingual_manuscript_audit.md",
                "results/word_export_audit.md",
                "results/submission_package_audit.md",
            ],
            notes=(
                f"Metric audit: {metric_status}; claim audit: {claim_status}; "
                f"claim-audited Markdown files: {len(claim_scanned_files)}; bilingual audit: {bilingual_status}; "
                f"Word export audit: {word_export_status}; Word documents: {word_export_documents}; "
                f"DOCX claim blockers: {word_export_claim_blockers}."
            ),
            action_required="Choose target journal and format to that outlet before submission.",
        ),
        TraceItem(
            phase="Reproducibility",
            requirement="Provide one-command local reproducibility for lightweight v1 artifacts",
            status=status_from_evidence(
                [
                    "src/run_v1_pipeline.py",
                    "reproducibility_manifest.json",
                    "reproducibility_run_report.json",
                    "REPRODUCIBILITY.md",
                ]
            ),
            evidence=[
                "src/run_v1_pipeline.py",
                "reproducibility_manifest.json",
                "reproducibility_run_report.json",
                "REPRODUCIBILITY.md",
            ],
            notes=reproducibility_note(),
            action_required="Rerun after any new artifact is added.",
        ),
    ]

    rows = [item.__dict__ for item in trace_items]
    status_counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    blockers = [row for row in rows if row["status"] == "missing_artifacts"]
    deferred = [row for row in rows if row["status"] == "deferred_v2_scope"]
    return {
        "status": "v1_traceable_for_submission_scope" if not blockers else "missing_submission_artifacts",
        "source_plan": "paper4_research_plan.md",
        "trace_item_count": len(rows),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "blocker_count": len(blockers),
        "blockers": blockers,
        "deferred_v2_scope_count": len(deferred),
        "deferred_v2_scope": deferred,
        "items": rows,
        "guardrail": "This traceability audit proves APFM v1 submission-scope artifact coverage. Human-label alignment metrics, EDINET empirical collection, and full SEC all-8-K filing-body validation are deferred v2 scope and are not active v1 submission claims.",
    }


def reproducibility_note() -> str:
    manifest = ROOT / "reproducibility_manifest.json"
    report = ROOT / "reproducibility_run_report.json"
    if not manifest.exists() and not report.exists():
        return "No reproducibility manifest or run report found yet."
    try:
        if manifest.exists():
            manifest_steps = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_count = len(manifest_steps) if isinstance(manifest_steps, list) else "unknown"
        else:
            manifest_count = "unknown"
        report_exists = report.exists()
        report_audit = ROOT / "results" / "reproducibility_run_report_audit.json"
        report_audit_exists = report_audit.exists()
        return (
            f"The lightweight pipeline manifest defines {manifest_count} steps. "
            f"The run report exists: {report_exists}. "
            f"The post-run run-report self-audit exists: {report_audit_exists}. "
            "Because this traceability audit can run inside the pipeline before the final report is written, "
            "it does not embed run status counts; use reproducibility_run_report.json and results/reproducibility_run_report_audit.json as the authoritative source for latest run status counts."
        )
    except Exception as exc:
        return f"Could not parse reproducibility manifest/report: {exc!r}."


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# Paper 4 Research Plan Traceability",
        "",
        f"Status: `{audit['status']}`",
        "",
        "This file maps `paper4_research_plan.md` requirements to current local evidence. APFM v1 is a DART-first finance manuscript; human-label alignment metrics, EDINET empirical collection, and full SEC all-8-K filing-body validation are deferred v2 scope rather than active submission blockers.",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in audit["status_counts"].items():
        lines.append(f"| `{status}` | {count} |")
    lines.extend(
        [
            "",
            "## Traceability Matrix",
            "",
            "| Phase | Requirement | Status | Evidence | Notes | Action required |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in audit["items"]:
        evidence = "<br>".join(
            f"`{item}` (present)" if exists(item) else f"`{item}` (missing)"
            for item in row["evidence"]
        )
        notes = str(row["notes"]).replace("|", "\\|")
        action = str(row["action_required"]).replace("|", "\\|")
        lines.append(
            f"| {row['phase']} | {row['requirement']} | `{row['status']}` | {evidence} | {notes} | {action} |"
        )
    lines.extend(
        [
            "",
            "## V1 Submission Blockers",
            "",
        ]
    )
    if audit["blockers"]:
        for row in audit["blockers"]:
            lines.append(f"- `{row['status']}`: {row['phase']} - {row['requirement']}. {row['action_required']}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Deferred V2 Scope",
            "",
        ]
    )
    if audit["deferred_v2_scope"]:
        for row in audit["deferred_v2_scope"]:
            lines.append(f"- `{row['status']}`: {row['phase']} - {row['requirement']}. {row['action_required']}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Guardrail", "", audit["guardrail"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = build_traceability()
    out_json = ROOT / "results" / "research_plan_traceability.json"
    out_md = ROOT / "results" / "research_plan_traceability.md"
    out_json.write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_markdown(audit, out_md)
    print(json.dumps({"status": audit["status"], "trace_item_count": audit["trace_item_count"], "blocker_count": audit["blocker_count"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
