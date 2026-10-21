from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audit_forbidden_tokens import FORBIDDEN_TOKENS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class DocumentRule:
    path: str
    required_groups: list[list[str]]
    required_contexts: list[list[str]] = field(default_factory=list)
    note: str = ""


def read_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def metric_row(path: str, horizon: int, model: str) -> dict[str, Any]:
    rows = read_json(path)
    for row in rows:
        if int(row.get("horizon_days", -1)) == int(horizon) and row.get("model") == model:
            return row
    raise KeyError(f"Missing metric row: {path} horizon={horizon} model={model}")


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def fmt_dec(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def build_rules() -> list[DocumentRule]:
    naver = read_json("data/raw/market_coverage/naver_ohlcv_supplement_summary.json")
    fdr_gap = read_json("data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.json")
    market_index = read_json("data/raw/market_index/korea_market_index_summary.json")
    main_market_panel = read_json("data/model_market/dart_model_panel_market_summary.json")
    robust_market_panel = read_json("data/model_market_robust/dart_model_panel_market_summary.json")
    residual_audit = read_json("results/market_residual_coverage_audit.json")
    completion = read_json("paper4_completion_audit.json")
    calculation = read_json("results/calculation_sheet_audit.json")
    reference = read_json("results/reference_consistency_audit.json")
    metric = read_json("results/manuscript_metric_audit.json")
    claim = read_json("results/manuscript_claim_audit.json")
    word_export = read_json("results/word_export_audit.json")
    journal_source = read_json("results/journal_candidate_source_audit.json")
    edinet_collection = read_json("results/edinet_collection_status.json")
    edinet_feasibility = read_json("results/edinet_api_feasibility.json")
    edinet_execution_preflight = read_json("results/edinet_execution_preflight.json")
    manifest = read_json("reproducibility_manifest.json")
    run_report = read_json("reproducibility_run_report.json")
    sec_all8k_preflight = read_json("results/sec_all8k_storage_preflight.json")
    alignment_label_preflight = read_json("results/alignment_label_preflight.json")

    main_365 = metric_row("results/market/dart_model_metrics.json", 365, "combined_market_calibrated_random_forest")
    main_730 = metric_row("results/market/dart_model_metrics.json", 730, "combined_market_calibrated_random_forest")
    robust_730 = metric_row(
        "results/market_robust/dart_model_metrics.json",
        730,
        "combined_market_calibrated_random_forest",
    )

    naver_rows = fmt_int(naver["output_rows_total"])
    fdr_rows = fmt_int(fdr_gap["output_rows_total"])
    fdr_no_rows = fmt_int(fdr_gap["no_row_symbols"])
    main_supp_rows = fmt_int(main_market_panel["supplemental_ohlcv_rows_loaded"])
    robust_supp_rows = fmt_int(robust_market_panel["supplemental_ohlcv_rows_loaded"])
    market_index_rows = fmt_int(market_index["rows"])
    main_coverage = fmt_pct(main_market_panel["price_record_rate"])
    main_ohlcv_rows = fmt_int(main_market_panel["ohlcv_rows_loaded"])
    robust_ohlcv_rows = fmt_int(robust_market_panel["ohlcv_rows_loaded"])
    residual_symbols = fmt_int(residual_audit["fdr_residual_symbols"])
    residual_main_rows = fmt_int(residual_audit["main"]["residual_rows"])
    residual_main_share = fmt_pct(residual_audit["main"]["residual_row_share"], 2)
    residual_positive_pair = (
        f"{int(residual_audit['main']['residual_target_365d_rows'])}/"
        f"{int(residual_audit['main']['residual_target_730d_rows'])}"
    )
    auc_365 = fmt_dec(main_365["roc_auc"])
    ap_365 = fmt_dec(main_365["average_precision"])
    lift_365 = fmt_dec(main_365["top_10pct"]["lift"], 2)
    lift_365_exact = fmt_dec(main_365["top_10pct"]["lift"])
    auc_730 = fmt_dec(main_730["roc_auc"])
    ap_730 = fmt_dec(main_730["average_precision"])
    lift_730 = fmt_dec(main_730["top_10pct"]["lift"], 2)
    lift_730_exact = fmt_dec(main_730["top_10pct"]["lift"])
    robust_lift_730 = fmt_dec(robust_730["top_10pct"]["lift"], 2)
    robust_lift_730_exact = fmt_dec(robust_730["top_10pct"]["lift"])
    completion_status = str(completion["overall_status"])
    calculation_verification_checks = str(calculation["verification_checks"])
    calculation_output_integrity_checks = str(len(calculation.get("output_integrity", [])))
    calculation_output_integrity_failures = str(len(calculation.get("output_integrity_failures", [])))
    claim_scanned_count = str(len(claim.get("scanned_files", [])))
    claim_blockers = str(claim["blocker_count"])
    claim_review = str(claim["review_count"])
    word_export_documents = str(word_export["document_count"])
    word_export_failures = str(word_export["failure_count"])
    word_export_claim_blockers = str(word_export["claim_blocker_count"])
    expected_submission_documents = "25"
    manifest_steps = str(len(manifest))
    manifest_heavy_steps = sum(1 for step in manifest if step.get("heavy"))
    run_step_rows = run_report.get("steps", []) if isinstance(run_report, dict) else []
    run_status_counts = run_report.get("status_counts", {}) if isinstance(run_report, dict) else {}
    if not run_status_counts and isinstance(run_step_rows, list):
        run_status_counts = {}
        for row in run_step_rows:
            status = str(row.get("status", "unknown"))
            run_status_counts[status] = run_status_counts.get(status, 0) + 1
    run_steps = str(run_report.get("step_count") or len(run_step_rows) or len(manifest))
    run_ok_steps = str(run_report.get("ok_count") or run_status_counts.get("ok", 0))
    run_skipped_heavy_steps = str(run_report.get("skipped_heavy_count") or run_status_counts.get("skipped_heavy", 0))
    run_failed_steps = str(run_report.get("failed_step_count") or sum(
        count for status, count in run_status_counts.items() if status not in {"ok", "skipped_heavy"}
    ))
    run_output_failures = str(run_report.get("output_failure_count") or 0)
    run_freshness_exempt_output_count = run_report.get("freshness_exempt_output_count")
    if run_freshness_exempt_output_count is None and isinstance(run_step_rows, list):
        run_freshness_exempt_output_count = sum(
            1
            for step in run_step_rows
            for output in (step.get("outputs") or [])
            if output.get("freshness_exempt")
        )
    run_freshness_exempt_outputs = str(run_freshness_exempt_output_count or 0)
    run_execute = str(run_report.get("execute"))
    journal_source_status = str(journal_source["status"])
    journal_source_doi_count = str(journal_source["doi_count"])
    journal_source_policy_count = str(journal_source["policy_url_count"])
    journal_source_doi_count_ko = f"{journal_source_doi_count}개"
    journal_source_policy_count_ko = f"{journal_source_policy_count}개"
    journal_source_failed_doi_count = str(journal_source["failed_doi_count"])
    journal_source_failed_policy_count = str(journal_source["failed_policy_url_count"])
    journal_source_candidate_count = str(journal_source["candidate_journal_count"])
    journal_source_expected_candidate_count = str(journal_source["expected_candidate_journal_count"])
    journal_source_candidate_doi_evidence_ok = str(journal_source["candidate_doi_evidence_ok"])
    journal_source_candidate_policy_evidence_count = str(journal_source["candidate_policy_evidence_count"])
    journal_source_candidate_policy_evidence_ok = str(journal_source["candidate_policy_evidence_ok"])
    journal_source_min_related_work_date = str(journal_source["min_related_work_date"])
    edinet_collection_status = str(edinet_collection["status"])
    edinet_planned_days = fmt_int(edinet_feasibility["planned_days"])
    edinet_feasible_days = fmt_int(edinet_feasibility["api_window_feasible_days"])
    edinet_infeasible_days = fmt_int(edinet_feasibility["api_window_infeasible_days"])
    edinet_as_of_date = str(edinet_feasibility["as_of_date"])
    edinet_preflight_status = str(edinet_feasibility["preflight"]["status"])
    edinet_execution_preflight_status = str(edinet_execution_preflight["status"])
    sec_all8k_preflight_status = str(sec_all8k_preflight["status"])
    alignment_label_preflight_status = str(alignment_label_preflight["status"])
    alignment_label_preflight_ready = str(alignment_label_preflight["final_rq1_metrics_ready"])

    manuscript_market_groups = [
        [naver_rows],
        [fdr_rows],
        [residual_symbols],
        [residual_main_rows],
        [residual_main_share],
        [residual_positive_pair],
        [main_coverage],
        [auc_365],
        [ap_365],
        [lift_365, lift_365_exact],
        [auc_730],
        [ap_730],
        [lift_730, lift_730_exact],
        [robust_lift_730, robust_lift_730_exact],
        [calculation_verification_checks],
    ]
    residual_groups = [
        [residual_symbols],
        [residual_main_rows],
        [residual_main_share],
        [residual_positive_pair],
    ]
    support_market_groups = [
        [auc_365],
        [ap_365],
        [lift_365, lift_365_exact],
        [auc_730],
        [ap_730],
        [lift_730, lift_730_exact],
    ]
    status_groups = [
        [naver_rows],
        [fdr_rows],
        [fdr_no_rows],
        [market_index_rows],
        [main_ohlcv_rows],
        [robust_ohlcv_rows],
        [residual_main_rows],
        [residual_main_share],
        [residual_positive_pair],
        [auc_365],
        [ap_365],
        [lift_365, lift_365_exact],
        [auc_730],
        [ap_730],
        [lift_730, lift_730_exact],
        [robust_lift_730, robust_lift_730_exact],
    ]
    residual_context_en = [["residual", residual_main_rows, residual_main_share, residual_positive_pair]]
    residual_context_ko = [["\uc794\uc5ec", residual_main_rows, residual_main_share, residual_positive_pair]]
    residual_context_audit = [["residual audit", residual_main_rows, residual_main_share, residual_positive_pair]]
    residual_context_completion = [["Residual coverage audit", residual_main_rows, residual_main_share, residual_positive_pair]]
    calculation_context_status = [
        [
            "Calculation sheet audit",
            calculation_verification_checks,
            "verification checks",
            calculation_output_integrity_checks,
            "output integrity checks",
            calculation_output_integrity_failures,
            "output integrity failures",
        ]
    ]
    calculation_context_completion = [
        ["calculation/model/manuscript/verification checks", calculation_verification_checks],
        ["output integrity checks/failures", calculation_output_integrity_checks, calculation_output_integrity_failures],
    ]
    claim_context_status = [["claim audit", claim_scanned_count, "Markdown files checked", claim_blockers, "blockers", claim_review, "review items"]]
    claim_context_completion = [["Manuscript claim audit status", "claim-audited Markdown files", claim_scanned_count, "blockers", claim_blockers, "review items", claim_review]]
    word_context_status = [["Word export audit", word_export_documents, "Word files", "DOCX claim boundaries", word_export_failures, "failures", word_export_claim_blockers, "DOCX claim blockers"]]
    journal_source_context_status = [["Journal candidate source audit", journal_source_candidate_count, "candidate journals", journal_source_doi_count, "DOI", journal_source_policy_count, "policy URLs", journal_source_candidate_policy_evidence_count, "candidate policy evidence rows", journal_source_failed_doi_count, "failed DOI", journal_source_failed_policy_count, "failed policy URL"]]
    submission_context_status = [["Submission package audit", expected_submission_documents, "documents"]]
    pipeline_context_status = [[
        "Reproducibility pipeline manifest/report",
        "steps",
    ]]
    run_report_self_audit_context_status = [["Reproducibility run-report self-audit", "pass", "current run report"]]
    edinet_execution_preflight_context = [["EDINET execution preflight", "edinet_execution_preflight.md"]]
    sec_all8k_preflight_context = [["SEC all-8-K full-scope download preflight", "sec_all8k_storage_preflight.md"]]
    alignment_label_preflight_context = [["Alignment label preflight", "alignment_label_preflight.md"]]
    target_journal_groups = [
        ["2026-07-01"],
        ["Expert Systems with Applications"],
        ["Knowledge-Based Systems"],
        ["Decision Support Systems"],
        ["Information Sciences"],
        ["Annals of Operations Research"],
        ["Computational Economics"],
        ["Asia-Pacific Financial Markets"],
        ["Review of Quantitative Finance and Accounting"],
        ["International Journal of Finance"],
        ["Applied Soft Computing"],
        ["DART"],
        ["SEC"],
        ["all-8-K"],
        ["v1"],
        ["v2"],
        ["v3"],
        ["subscription"],
        ["OA", "APC"],
        ["저자 부담 없음"],
        ["10-journal"],
        ["mandatory paid open-access"],
        [journal_source_doi_count_ko],
        [journal_source_policy_count_ko],
    ]
    target_journal_contexts = [
        ["Expert Systems with Applications", "v1"],
        ["Knowledge-Based Systems", "v1"],
        ["Decision Support Systems", "v1"],
        ["Information Sciences", "Annals of Operations Research"],
        ["Asia-Pacific Financial Markets", "Applied Soft Computing"],
        ["SEC", "all-8-K"],
        ["OA", "APC"],
        ["subscription/non-OA", "경로"],
        ["Open Access", "paid OA", "강제"],
        ["10-journal", "submission_candidate_journals_ko.docx"],
        ["mandatory paid open-access", "no-payment instruction"],
        ["정책 URL", journal_source_policy_count_ko, "DOI", journal_source_doi_count_ko],
    ]
    strategy_candidate_groups = [
        ["Expert Systems with Applications"],
        ["Knowledge-Based Systems"],
        ["Decision Support Systems"],
        ["Information Sciences"],
        ["Annals of Operations Research"],
        ["Computational Economics"],
        ["Asia-Pacific Financial Markets"],
        ["Review of Quantitative Finance and Accounting"],
        ["International Journal of Finance"],
        ["Applied Soft Computing"],
        ["10.1016/j.eswa.2025.127832"],
        ["10.1016/j.knosys.2026.115790"],
        ["10.1016/j.dss.2026.114615"],
        ["10.1016/j.ins.2025.122870"],
        ["10.1007/s10479-025-06706-5"],
        ["10.1007/s10614-026-11387-8"],
        ["10.1007/s10690-025-09525-7"],
        ["10.1007/s11156-023-01192-x"],
        ["10.1002/ijfe.70077"],
        ["10.1016/j.asoc.2026.115085"],
        ["10-journal"],
        ["subscription/non-OA"],
        ["mandatory paid open-access"],
    ]
    strategy_candidate_contexts = [
        ["Expert Systems with Applications", "10.1016/j.eswa.2025.127832"],
        ["Knowledge-Based Systems", "10.1016/j.knosys.2026.115790"],
        ["Decision Support Systems", "10.1016/j.dss.2026.114615"],
        ["Information Sciences", "10.1016/j.ins.2025.122870"],
        ["Applied Soft Computing", "10.1016/j.asoc.2026.115085"],
        ["10-journal", "submission_candidate_journals_ko.md"],
        ["mandatory paid open-access", "no-payment instruction"],
    ]
    candidate_journal_groups = [
        ["2026-07-01"],
        ["Expert Systems with Applications"],
        ["Knowledge-Based Systems"],
        ["Decision Support Systems"],
        ["Information Sciences"],
        ["Annals of Operations Research"],
        ["Computational Economics"],
        ["Asia-Pacific Financial Markets"],
        ["Review of Quantitative Finance and Accounting"],
        ["International Journal of Finance"],
        ["Applied Soft Computing"],
        ["10.1016/j.eswa.2025.127832"],
        ["10.1016/j.knosys.2026.115790"],
        ["10.1016/j.dss.2026.114615"],
        ["10.1016/j.ins.2025.122870"],
        ["10.1007/s10479-025-06706-5"],
        ["10.1007/s10614-026-11387-8"],
        ["10.1007/s10690-025-09525-7"],
        ["10.1007/s11156-023-01192-x"],
        ["10.1002/ijfe.70077"],
        ["10.1016/j.asoc.2026.115085"],
        ["APC"],
        ["subscription"],
        ["Open Access"],
        ["2025-07-01"],
        ["https://www.elsevier.com/researcher/author/open-access/choice"],
    ]
    candidate_journal_contexts = [
        ["선정 기준", "2025-07-01", "최소 1개"],
        ["Expert Systems with Applications", "10.1016/j.eswa.2025.127832"],
        ["Knowledge-Based Systems", "10.1016/j.knosys.2026.115790"],
        ["Decision Support Systems", "10.1016/j.dss.2026.114615"],
        ["Applied Soft Computing", "10.1016/j.asoc.2026.115085"],
        ["APC", "subscription"],
    ]
    journal_source_groups = [
        ["Status: `pass`"],
        [journal_source_doi_count],
        [journal_source_policy_count],
        [journal_source_candidate_count],
        [journal_source_expected_candidate_count],
        [journal_source_failed_doi_count],
        [journal_source_failed_policy_count],
        [journal_source_candidate_policy_evidence_count],
        ["Candidate policy evidence"],
        ["10.1016/j.eswa.2025.127832"],
        ["10.1016/j.knosys.2026.115790"],
        ["10.1016/j.dss.2026.114615"],
        ["10.1016/j.ins.2025.122870"],
        ["10.1007/s10479-025-06706-5"],
        ["10.1007/s10614-026-11387-8"],
        ["10.1007/s10690-025-09525-7"],
        ["10.1007/s11156-023-01192-x"],
        ["10.1002/ijfe.70077"],
        ["10.1016/j.asoc.2026.115085"],
        ["https://www.elsevier.com/about/policies-and-standards/pricing"],
        ["https://www.elsevier.com/researcher/author/open-access/choice"],
        ["https://support.nature.com/en/support/solutions/articles/6000084580-costs-of-publishing-in-springer-nature-journal"],
        ["https://authors.wiley.com/author-resources/Journal-Authors/open-access/hybrid-open-access/index.html"],
    ]
    journal_source_contexts = [
        ["Candidate journal rows", journal_source_candidate_count, journal_source_expected_candidate_count],
        ["Candidate ranks ok", "True"],
        ["Candidate journals ok", "True"],
        ["Candidate DOI evidence ok", journal_source_candidate_doi_evidence_ok],
        ["Candidate policy evidence ok", journal_source_candidate_policy_evidence_ok],
        ["Candidate policy evidence rows", journal_source_candidate_policy_evidence_count],
        ["Minimum related-work date", journal_source_min_related_work_date],
        ["Failed DOI checks", journal_source_failed_doi_count],
        ["Failed policy URL checks", journal_source_failed_policy_count],
        ["DOI entries", journal_source_doi_count],
        ["Policy URLs", journal_source_policy_count],
    ]
    alignment_guide_groups = [
        ["MATCH"],
        ["PARTIAL"],
        ["NO_MATCH"],
        ["Precision@1"],
        ["MRR"],
        ["human_label"],
    ]
    alignment_guide_contexts = [
        ["Precision@1", "MRR"],
        ["human_label"],
    ]
    two_annotator_contexts = [
        ["Cohen's kappa", "annotator"],
    ]
    sec_support_groups = [
        ["v2"],
        ["support evidence", "support"],
        ["full all-8-K"],
        ["U.S. validation"],
    ]

    return [
        DocumentRule(
            "manuscript/paper4_v4_manuscript_en.md",
            manuscript_market_groups,
            required_contexts=residual_context_en,
        ),
        DocumentRule(
            "manuscript/paper4_v4_manuscript_ko.md",
            manuscript_market_groups,
            required_contexts=residual_context_ko,
        ),
        DocumentRule(
            "manuscript/paper4_v4_title_page.md",
            [
                ["Title Page"],
                ["Regulatory Disclosure Sequences and Corporate Distress Risk"],
                ["title page only"],
                ["paper4_v4_manuscript_en.md"],
                ["paper4_v4_manuscript_en.docx"],
            ],
            required_contexts=[["full English manuscript", "paper4_v4_manuscript_en.md"]],
        ),
        DocumentRule("manuscript/cover_letter_v1_generic.md", support_market_groups + [[robust_lift_730]]),
        DocumentRule("manuscript/submission_checklist_v1.md", support_market_groups),
        DocumentRule(
            "manuscript/submission_strategy.md",
            support_market_groups
            + strategy_candidate_groups
            + [
                [robust_supp_rows],
                [main_supp_rows],
                [fdr_no_rows],
                [residual_main_rows],
                [residual_main_share],
                [residual_positive_pair],
                [market_index_rows],
                [main_coverage],
                [edinet_collection_status],
                [edinet_planned_days],
                [edinet_infeasible_days],
                [edinet_as_of_date],
                [edinet_preflight_status],
                ["edinet_execution_preflight.md"],
                ["alignment_label_preflight.md"],
            ],
            required_contexts=residual_context_audit
            + strategy_candidate_contexts
            + [["EDINET extension evidence", edinet_collection_status, edinet_planned_days, edinet_infeasible_days, edinet_as_of_date, edinet_preflight_status]]
            + edinet_execution_preflight_context
            + alignment_label_preflight_context,
        ),
        DocumentRule(
            "manuscript/paper4_target_journal_expansion_ko.md",
            target_journal_groups,
            required_contexts=target_journal_contexts,
        ),
        DocumentRule(
            "manuscript/submission_candidate_journals_ko.md",
            candidate_journal_groups,
            required_contexts=candidate_journal_contexts,
        ),
        DocumentRule(
            "results/journal_candidate_source_audit.md",
            journal_source_groups,
            required_contexts=journal_source_contexts,
        ),
        DocumentRule(
            "manuscript/alignment_annotation_guide.md",
            alignment_guide_groups + [["annotator1_label"], ["annotator2_label"], ["adjudicated_label"]],
            required_contexts=alignment_guide_contexts + two_annotator_contexts,
        ),
        DocumentRule(
            "manuscript/alignment_single_expert_labeling_guide.md",
            alignment_guide_groups,
            required_contexts=alignment_guide_contexts + [["Cohen's kappa"]],
        ),
        DocumentRule(
            "manuscript/alignment_sec_desktop_labeling_guide.md",
            alignment_guide_groups + [["manual_alignment_seed_sec_desktop_100"]],
            required_contexts=alignment_guide_contexts + two_annotator_contexts,
        ),
        DocumentRule(
            "paper4_execution_status.md",
            status_groups
            + [
                [expected_submission_documents],
                ["alignment_label_preflight.md"],
                ["edinet_execution_preflight.md"],
                ["sec_all8k_storage_preflight.md"],
            ],
            required_contexts=(
                residual_context_audit
                + calculation_context_status
                + claim_context_status
                + word_context_status
                + journal_source_context_status
                + submission_context_status
                + pipeline_context_status
                + run_report_self_audit_context_status
                + alignment_label_preflight_context
                + edinet_execution_preflight_context
                + sec_all8k_preflight_context
            ),
        ),
        DocumentRule(
            "results/market_extension_summary.md",
            [
                [naver_rows],
                [fdr_rows],
                [fdr_no_rows],
                [residual_main_rows],
                [residual_main_share],
                [residual_positive_pair],
                [market_index_rows],
                [main_coverage],
                [auc_365],
                [ap_365],
                [lift_365],
                [auc_730],
                [ap_730],
                [lift_730],
                [robust_lift_730],
            ],
            required_contexts=residual_context_en,
        ),
        DocumentRule(
            "results/sec_companyfacts_availability.md",
            sec_support_groups,
            required_contexts=[["full all-8-K filing-body coverage", "final U.S. validation scope"]],
        ),
        DocumentRule(
            "results/sec_filing_body_text_baseline_summary.md",
            [["target-item"], ["not every SEC 8-K"], ["full all-8-K"], ["U.S. validation"]],
            required_contexts=[["full all-8-K filing-body coverage", "final U.S. validation scope"]],
        ),
        DocumentRule(
            "results/sec_all8k_sample_text_baseline_summary.md",
            [["sample-level"], ["full all-8-K text validation"], ["standalone U.S. distress validation"]],
            required_contexts=[["full all-8-K text validation", "standalone U.S. distress validation"]],
        ),
        DocumentRule(
            "results/alignment_label_preflight.md",
            [
                ["Alignment Label Preflight"],
                [alignment_label_preflight_status],
                [alignment_label_preflight_ready],
                ["Final RQ1 metrics ready"],
                ["Single-expert human_label rows"],
                ["Annotator 1 labeled rows"],
                ["Annotator 2 labeled rows"],
                ["Supplemental SEC labeled rows"],
                ["Blockers"],
            ],
            required_contexts=[
                ["Status", alignment_label_preflight_status],
                ["Precision@1", "MRR", "not ready"],
            ],
        ),
        DocumentRule(
            "results/edinet_execution_preflight.md",
            [
                ["EDINET Execution Preflight"],
                [edinet_execution_preflight_status],
                ["Environment EDINET API key present"],
                ["API-window infeasible days"],
                ["Document rows"],
                ["Current-report candidate rows"],
                ["Blockers"],
            ],
            required_contexts=[["Status", edinet_execution_preflight_status], ["Empirical collection ready"]],
        ),
        DocumentRule(
            "results/sec_all8k_storage_preflight.md",
            [
                ["SEC All-8-K Full-Scope Download Preflight"],
                [sec_all8k_preflight_status],
                ["Required free space"],
                ["Storage ready"],
                ["Environment SEC user agent present"],
                ["Blockers"],
            ],
            required_contexts=[["Status", sec_all8k_preflight_status], ["Full-scope download ready"]],
        ),
        DocumentRule(
            "results/sec_validation_scope.md",
            [["not_allowed"], ["not_allowed_yet"], ["1.62%"], ["sample-limited"]],
            required_contexts=[["not_allowed", "full all-8-K"], ["not_allowed_yet", "standalone U.S. validation"]],
        ),
        DocumentRule(
            "manuscript/claim_readiness_matrix.md",
            [[fdr_no_rows], [residual_main_rows], [residual_main_share], ["Coverage-expanded"]],
            required_contexts=residual_context_en,
        ),
        DocumentRule(
            "VERSIONS.md",
            [
                [naver_rows],
                [fdr_rows],
                [fdr_no_rows],
                [main_coverage],
                ["submission_candidate_journals_ko.md"],
                ["submission_candidate_journals_ko.docx"],
                ["journal_candidate_source_audit.md"],
                ["reproducibility_run_report_audit.md"],
                ["edinet_execution_preflight.md"],
                ["sec_all8k_storage_preflight.md"],
                ["alignment_label_preflight.md"],
            ]
            + residual_groups,
            required_contexts=residual_context_en
            + [["Korean submission candidate table", "submission_candidate_journals_ko.md", "submission_candidate_journals_ko.docx"]]
            + [["Journal candidate source audit", "journal_candidate_source_audit.md", "journal_candidate_source_audit.json"]]
            + [["Reproducibility run-report self-audit", "reproducibility_run_report_audit.md", "reproducibility_run_report_audit.json"]]
            + edinet_execution_preflight_context
            + alignment_label_preflight_context
            + sec_all8k_preflight_context,
        ),
        DocumentRule(
            "REPRODUCIBILITY.md",
            [
                [naver_rows],
                [fdr_rows],
                [fdr_no_rows],
                [main_coverage],
                [edinet_as_of_date],
                [edinet_infeasible_days],
                [edinet_preflight_status],
                ["edinet_execution_preflight.md"],
                ["submission_candidate_journals_ko.md"],
                ["submission_candidate_journals_ko.docx"],
                ["journal_candidate_source_audit.md"],
                ["reproducibility_run_report_audit.md"],
                ["sec_all8k_storage_preflight.md"],
                ["alignment_label_preflight.md"],
            ]
            + residual_groups,
            required_contexts=residual_context_en
            + [["As of", edinet_as_of_date, edinet_infeasible_days, edinet_preflight_status]]
            + edinet_execution_preflight_context
            + [["Korean submission candidate table", "submission_candidate_journals_ko.md", "submission_candidate_journals_ko.docx"]]
            + [["Journal candidate source audit", "journal_candidate_source_audit.md", "journal_candidate_source_audit.json"]]
            + [["Reproducibility run-report self-audit", "reproducibility_run_report_audit.md", "reproducibility_run_report_audit.json"]]
            + alignment_label_preflight_context
            + sec_all8k_preflight_context,
        ),
        DocumentRule(
            "data_map.md",
            [
                [fdr_rows],
                [fdr_no_rows],
                [main_coverage],
                [auc_365],
                [auc_730],
                [expected_submission_documents],
                [word_export_documents],
                [calculation_verification_checks],
                [calculation_output_integrity_checks],
                ["source-file hashes"],
                [edinet_as_of_date],
                [edinet_feasible_days],
                [edinet_infeasible_days],
                [edinet_preflight_status],
                ["edinet_execution_preflight.md"],
                ["submission_candidate_journals_ko.md"],
                ["submission_candidate_journals_ko.docx"],
                ["journal_candidate_source_audit.md"],
                ["reproducibility_run_report_audit.md"],
                ["sec_all8k_storage_preflight.md"],
                ["alignment_label_preflight.md"],
                [journal_source_candidate_count],
                [journal_source_doi_count],
                [journal_source_policy_count],
            ]
            + residual_groups,
            required_contexts=residual_context_en
            + [
                ["Calculation sheet", calculation_verification_checks, "verification checks", calculation_output_integrity_checks, "output integrity checks", "source-file hashes"],
                ["EDINET API feasibility", edinet_as_of_date, edinet_feasible_days, edinet_infeasible_days, edinet_preflight_status],
                ["EDINET execution preflight", "edinet_execution_preflight.md"],
                ["Korean submission candidate table", journal_source_candidate_count, "candidate journals", "submission_candidate_journals_ko.md", "submission_candidate_journals_ko.docx"],
                ["Journal candidate source audit", journal_source_candidate_count, "candidate journals", journal_source_doi_count, "DOI entries", journal_source_policy_count, "policy URLs", journal_source_candidate_policy_evidence_count, "candidate policy evidence rows"],
                ["Pipeline execution report self-audit", "reproducibility_run_report_audit.md", "reproducibility_run_report_audit.json"],
                ["SEC all-8-K full-scope download preflight", "sec_all8k_storage_preflight.md"],
                ["Alignment label preflight", "alignment_label_preflight.md"],
            ],
        ),
    ]


def line_numbers(text: str, token: str) -> list[int]:
    return [idx for idx, line in enumerate(text.splitlines(), start=1) if token in line]


def context_matches(text: str, context: list[str]) -> list[int]:
    return [
        idx
        for idx, line in enumerate(text.splitlines(), start=1)
        if all(token in line for token in context)
    ]


def evaluate_rule(rule: DocumentRule) -> dict[str, Any]:
    path = ROOT / rule.path
    if not path.exists():
        return {
            "path": rule.path,
            "status": "missing_file",
            "required": rule.required_groups,
            "required_contexts": rule.required_contexts,
            "missing_groups": rule.required_groups,
            "missing_contexts": rule.required_contexts,
            "forbidden_hits": [],
            "note": rule.note,
        }
    text = path.read_text(encoding="utf-8")
    groups = []
    for group in rule.required_groups:
        matched = [token for token in group if token in text]
        groups.append(
            {
                "accepted_values": group,
                "matched": bool(matched),
                "matched_values": matched,
                "line_numbers": sorted({line for token in matched for line in line_numbers(text, token)}),
            }
        )
    contexts = []
    for context in rule.required_contexts:
        lines = context_matches(text, context)
        contexts.append(
            {
                "required_same_line": context,
                "matched": bool(lines),
                "line_numbers": lines,
            }
        )
    forbidden_hits = [
        {"token": token, "line_numbers": line_numbers(text, token)}
        for token in FORBIDDEN_TOKENS
        if token in text
    ]
    missing_groups = [group["accepted_values"] for group in groups if not group["matched"]]
    missing_contexts = [context["required_same_line"] for context in contexts if not context["matched"]]
    status = "passed"
    if missing_groups or missing_contexts:
        status = "missing_required_value"
    if forbidden_hits:
        status = "forbidden_stale_value"
    return {
        "path": rule.path,
        "status": status,
        "required": groups,
        "required_contexts": contexts,
        "missing_groups": missing_groups,
        "missing_contexts": missing_contexts,
        "forbidden_hits": forbidden_hits,
        "note": rule.note,
    }


def build_audit() -> dict[str, Any]:
    checks = [evaluate_rule(rule) for rule in build_rules()]
    failures = [check for check in checks if check["status"] != "passed"]
    return {
        "status": "pass" if not failures else "fail",
        "document_count": len(checks),
        "failure_count": len(failures),
        "forbidden_token_count": len(FORBIDDEN_TOKENS),
        "forbidden_token_policy": "src/audit/audit_forbidden_tokens.py",
        "checks": checks,
        "guardrail": "Submission-facing, manuscript-readiness, and SEC claim-boundary support Markdown files generated before the completion audit must carry required current values where applicable and must not contain known stale headline values or unfinished markers. Post-completion generated audit files are checked by the final claim-boundary audit.",
    }


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Submission Package Consistency Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Documents checked | {audit['document_count']} |",
        f"| Failures | {audit['failure_count']} |",
        "",
        "## Document Checks",
        "",
        "| Document | Status | Missing required values | Missing required context | Forbidden stale values | Evidence lines |",
        "|---|---|---|---|---|---|",
    ]
    for check in audit["checks"]:
        missing = "; ".join(", ".join(group) for group in check["missing_groups"]) or "-"
        missing_context = "; ".join(", ".join(group) for group in check.get("missing_contexts", [])) or "-"
        forbidden = "; ".join(hit["token"] for hit in check["forbidden_hits"]) or "-"
        evidence_lines = sorted(
            {
                line
                for group in check.get("required", [])
                for line in group.get("line_numbers", [])
            }
            | {
                line
                for context in check.get("required_contexts", [])
                for line in context.get("line_numbers", [])
            }
        )
        evidence = ", ".join(str(line) for line in evidence_lines[:16]) if evidence_lines else "-"
        if len(evidence_lines) > 16:
            evidence += ", ..."
        lines.append(
            "| "
            f"`{escape_md(check['path'])}` | "
            f"`{check['status']}` | "
            f"{escape_md(missing)} | "
            f"{escape_md(missing_context)} | "
            f"{escape_md(forbidden)} | "
            f"{escape_md(evidence)} |"
        )
    lines.extend(
        [
            "",
            "## Forbidden Stale Values",
            "",
            f"{audit['forbidden_token_count']} configured stale, mojibake, and unfinished-marker patterns from `{audit['forbidden_token_policy']}` are checked across submission-facing, manuscript-readiness, and SEC claim-boundary support Markdown files by `src/audit/submission_package_audit.py`. Passing reports omit the literal pattern list to keep repository-wide stale-value searches clean.",
            "",
            "## Guardrail",
            "",
            audit["guardrail"],
        ]
    )
    return "\n".join(lines) + "\n"


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    path.write_text(render_markdown(audit), encoding="utf-8")


def check_existing_outputs(audit: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {
        "json": (
            RESULTS / "submission_package_audit.json",
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        ),
        "markdown": (RESULTS / "submission_package_audit.md", render_markdown(audit)),
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
    parser = argparse.ArgumentParser(description="Build or verify the Paper 4 submission package audit.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Recompute and compare existing JSON/Markdown outputs without rewriting them.",
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    if args.check_only:
        checks = check_existing_outputs(audit)
        ok = all(row["exists"] and row["matches_recomputed"] for row in checks)
        print(
            json.dumps(
                {
                    "status": "pass" if ok else "fail",
                    "mode": "check_only",
                    "document_count": audit["document_count"],
                    "failure_count": audit["failure_count"],
                    "output_integrity": checks,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(0 if ok else 1)
    (RESULTS / "submission_package_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(audit, RESULTS / "submission_package_audit.md")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "document_count": audit["document_count"],
                "failure_count": audit["failure_count"],
            },
            indent=2,
        )
    )
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
