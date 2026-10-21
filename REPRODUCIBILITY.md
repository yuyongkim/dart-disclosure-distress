# Paper 4 Reproducibility Guide

Last updated: 2026-07-02

## Current Status

The current v1 artifact is reproducible from local data already found on the Desktop audit. The v1 paper is DART-first, with SEC used for taxonomy-transfer and alignment infrastructure. SEC expansion has been collected as a v2 support artifact, while EDINET remains Phase 2 because only a collection plan exists.

## One-Command Pipeline

Dry run:

```powershell
python "Paper 4\src\run_v1_pipeline.py"
```

Execute lightweight steps only:

```powershell
python "Paper 4\src\run_v1_pipeline.py" --execute
```

Execute every step, including data audit, model training, robust training, market-price extension training, and permutation importance:

```powershell
python "Paper 4\src\run_v1_pipeline.py" --execute --include-heavy
```

The runner writes:

- `reproducibility_manifest.json`
- `reproducibility_run_report.json`
- `reproducibility_run_report.json` records command status counts plus declared-output existence, size, freshness, output-failure, and freshness-exemption checks for each executed non-heavy step
- `results/reproducibility_run_report_audit.md` and `results/reproducibility_run_report_audit.json` are written after the run report itself and verify the current run-report summary
- `submission_package/paper4_core_package_latest.zip`, `submission_package/paper4_core_package_latest_audit.json`, and `submission_package/LATEST_PACKAGE.md` are built after the run-report self-audit passes, so the package can include the current generated audit files without becoming a manifest-step dependency cycle while the folder-level pointer records the latest ZIP path, SHA, size, local UTC regeneration timestamp, and superseded timestamped ZIP status; the pointer timestamp is not the Gmail send timestamp
- `submission_package/email_handoff_receipt_audit.md` and `submission_package/email_handoff_receipt_audit.json` are run after the latest package build; pipeline runs are report-only because a normal run may legitimately change the ZIP SHA. After refreshing the receipt, use `python "Paper 4\src\audit\post_run_artifact_status.py" --require-email-handoff` for a final handoff lock without rebuilding the ZIP.
- `results/post_run_artifact_status.md` and `results/post_run_artifact_status.json` preserve the post-run package and email-handoff status that is also printed to the console, including message ID, thread ID, display URL, timestamp, subject, attachment filename, SHA, and size
- `results/dart_gap_collection_status.md` and `data/raw/dart_gap/dart_gap_outcome_candidates.csv` through the non-heavy DART gap/outcome derivation step
- `results/sec_gap_inventory.md` through the non-heavy SEC gap inventory step
- `results/sec_expanded_collection_status.md`, `results/sec_expanded_sequence_summary.md`, `results/sec_bankruptcy_case_sequences.md`, `results/sec_desktop_population_collection_status.md`, `results/sec_desktop_population_sequence_summary.md`, and `results/sec_desktop_population_bankruptcy_sequences.md` through the non-heavy SEC expanded/case/population-style steps
- `results/sec_companyfacts_availability.md` and `data/model_sec_desktop/sec_companyfacts_validation_coverage.csv` through the non-heavy SEC companyfacts availability audit
- `results/sec_filing_text_collection_status.md` and `data/raw/sec_filing_text/sec_filing_text_manifest.csv` through the non-heavy SEC filing-body text manifest step
- `results/sec_all8k_text_scope_plan.md`, `data/raw/sec_all8k_text/sec_all8k_text_manifest.csv`, and `data/raw/sec_all8k_text/sec_all8k_text_batch_plan.csv` through the non-heavy all-8-K text planning step
- `results/sec_all8k_text_batch_collection_status.md`, `results/sec_all8k_text_batch_collection_status.json`, `data/raw/sec_all8k_text/sec_all8k_text_batch_status.csv`, and, when explicit bounded execution has been used, `data/raw/sec_all8k_text/sec_all8k_text_batch_download_log.csv` through the non-heavy all-8-K text batch-status step
- `results/sec_all8k_storage_preflight.md` and `results/sec_all8k_storage_preflight.json` through the non-heavy SEC all-8-K full-scope download preflight step
- `results/sec_all8k_text_sample_design.md`, `data/raw/sec_all8k_text/sec_all8k_text_sample_plan.csv`, and `data/raw/sec_all8k_text/sec_all8k_text_sample_strata.csv` through the non-heavy all-8-K text sample-design step
- `results/sec_all8k_text_sample_collection_status.md` and `data/raw/sec_all8k_text/sec_all8k_text_sample_status.csv` through the non-heavy all-8-K sample text status step
- `results/sec_all8k_sample_text_baseline_summary.md`, `results/sec_all8k_sample_text_baseline_metrics.csv`, and `data/model_sec_desktop/sec_all8k_sample_text_features.csv` through the non-heavy all-8-K sample text baseline step
- `results/sec_validation_scope.md` and `data/model_sec_desktop/sec_validation_scope_decision_table.csv` through the non-heavy SEC claim-scope audit
- `results/edinet_collection_status.md`, `results/edinet_api_feasibility.md`, `results/edinet_execution_preflight.md`, `results/edinet_local_discovery.md`, and `results/jp_report_parser_prototype_summary.md` through the non-heavy EDINET planning/feasibility/execution-preflight/local-discovery/parser-prototype steps
- `results/t04_local_discovery.md` and `data/audit/t04_local_candidate_files.csv` through the non-heavy T04 going-concern/audit-opinion local-discovery step
- `results/alignment_annotation_qc.md`, `data/processed/manual_alignment_seed_100_annotation_priority.csv`, and `data/processed/manual_alignment_seed_100_first_pass_30.csv` through the non-heavy RQ1 annotation-QC step
- `results/alignment_candidate_diagnostics.md` and `data/processed/manual_alignment_seed_100_candidate_diagnostics.csv` through the non-heavy RQ1 pre-label diagnostics step
- `results/alignment_silver_evaluation.md`, `data/processed/manual_alignment_seed_100_silver_review_queue.csv`, and `data/processed/manual_alignment_seed_100_silver_consensus.csv` through the non-heavy RQ1 silver pre-label evaluation step
- `results/alignment_sec_desktop_seed_summary.md`, `results/alignment_sec_desktop_evaluation.md`, `data/processed/manual_alignment_seed_sec_desktop_100.csv`, `data/processed/manual_alignment_seed_sec_desktop_100.xlsx`, and `manuscript/alignment_sec_desktop_labeling_guide.md` through the non-heavy local SEC supplemental alignment-seed/evaluation steps
- `results/alignment_single_expert_readiness.md`, `data/processed/manual_alignment_seed_100_single_expert_label_sheet.xlsx`, and `manuscript/alignment_single_expert_labeling_guide.md` through the non-heavy RQ1 single-expert label-sheet step
- `results/alignment_label_preflight.md` and `results/alignment_label_preflight.json` through the non-heavy RQ1 label-readiness preflight step
- `results/manuscript_metric_audit.md` and `results/manuscript_metric_audit.json` through the non-heavy manuscript metric-consistency audit step
- `results/calculation_sheet.xlsx`, `results/calculation_sheet.md`, and `results/calculation_sheet_audit.json` through the non-heavy calculation-sheet consistency step
- `results/market_residual_coverage_audit.md`, `data/audit/market_residual_symbol_impact.csv`, and `data/audit/market_residual_year_impact.csv` through the non-heavy residual OHLCV impact audit step
- `results/reference_consistency_audit.md` and `results/reference_consistency_audit.json` through the non-heavy reference-consistency audit step
- `manuscript/submission_candidate_journals_ko.md`, `manuscript/submission_candidate_journals_ko.docx`, `results/journal_candidate_source_audit.md`, and `results/journal_candidate_source_audit.json` through the non-heavy 10-journal no-payment/recent-related-work candidate audit and Word-export checks
- `results/manuscript_claim_audit.md` and `results/manuscript_claim_audit.json` through the non-heavy English/Korean/draft manuscript claim-audit step
- `results/bilingual_manuscript_audit.md` and `results/bilingual_manuscript_audit.json` through the non-heavy English/Korean final-manuscript consistency audit step
- `results/word_export_audit.md` and `results/word_export_audit.json` through the non-heavy Word-export consistency, DOCX claim-boundary, and text-integrity audit step after DOCX conversion
- `results/submission_package_audit.md` and `results/submission_package_audit.json` through the non-heavy submission-package consistency and text-integrity audit step after DOCX conversion and Word-export auditing
- `results/research_plan_traceability.md` and `results/research_plan_traceability.json` through the non-heavy research-plan traceability audit step
- `results/public_submission_manifest_audit.md`, `results/public_submission_manifest_audit.json`, and `data/audit/public_submission_manifest.csv` through the non-heavy public/submission package allowlist audit step
- `results/remaining_risk_audit.md` and `results/remaining_risk_audit.json` through the non-heavy remaining-risk audit step
- `results/final_claim_boundary_audit.md` and `results/final_claim_boundary_audit.json` through the final post-completion claim-boundary audit step, currently covering 46 Markdown files including every public-submission-manifest Markdown file such as `results/market_residual_coverage_audit.md`, plus `paper4_research_plan.md`, `results/public_submission_manifest_audit.md`, `results/remaining_risk_audit.md`, `results/post_run_artifact_status.md`, `submission_package/email_handoff_receipt_audit.md`, and `submission_package/LATEST_PACKAGE.md`
- `data/raw/market_coverage/naver_ohlcv_supplement_summary.md`, `data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.md`, `data/raw/market_index/korea_market_index_summary.md`, `data/model_market/dart_model_panel_market.parquet`, `results/market/dart_model_metrics.md`, `data/model_market_robust/dart_model_panel_market.parquet`, and `results/market_robust/dart_model_metrics.md` through heavy OHLCV market-price extension steps when `--include-heavy` is passed

## Manual Steps Not Automated

The DART gap step does not call the DART API. It resolves the Sprint 0 "outcome-like labels" gap by deriving distress outcomes from already collected Paper 3 DART major-event titles:

```powershell
python "Paper 4\src\collect\dart_gap_collector.py"
```

SEC expanded collection is a v2 support path. The current workspace already contains a collected 132-CIK SEC submissions pull. The default command preserves that collected state instead of overwriting it with an empty dry-run:

```powershell
python "Paper 4\src\collect\sec_gap_collector.py"
```

To refresh SEC submissions metadata, configure an identifying user agent with contact information and explicitly execute:

```powershell
$env:SEC_USER_AGENT="Your Name your.email@example.com"
python "Paper 4\src\collect\sec_gap_collector.py" --execute
```

After collection, rebuild the expanded SEC taxonomy sequence artifacts:

```powershell
python "Paper 4\src\sequence\sec_expanded_sequences.py"
```

The targeted SEC bankruptcy case sample is also reproducible. It preserves an existing collection unless `--refresh` is passed:

```powershell
python "Paper 4\src\collect\sec_bankruptcy_case_collector.py"
python "Paper 4\src\sequence\sec_expanded_sequences.py" --raw-path "Paper 4\data\raw\sec_bankruptcy_cases\sec_bankruptcy_case_filings.csv" --prefix sec_bankruptcy_case --source-name SEC_BANKRUPTCY_CASE --title "SEC Bankruptcy Case Sequence Summary"
python "Paper 4\src\sequence\sec_bankruptcy_case_examples.py"
```

The local SEC population-style scan is reproducible without new SEC API calls. The first full scan reads 8,021 local submissions files and may take several minutes; subsequent default runs preserve the existing output unless `--refresh` is passed:

```powershell
python "Paper 4\src\collect\sec_desktop_population_collector.py"
python "Paper 4\src\sequence\sec_expanded_sequences.py" --raw-path "Paper 4\data\raw\sec_desktop_population\sec_desktop_population_8k_filings.csv" --prefix sec_desktop_population --source-name SEC_DESKTOP_POPULATION --title "Local SEC Population Sequence Summary"
python "Paper 4\src\sequence\sec_bankruptcy_case_examples.py" --events-path "Paper 4\data\processed\sec_desktop_population_events.parquet" --output-prefix sec_desktop_population_bankruptcy_sequences --title "Local SEC Population Bankruptcy Sequences"
python "Paper 4\src\model\sec_desktop_sequence_validation.py"
python "Paper 4\src\model\sec_desktop_validation_robustness.py"
python "Paper 4\src\audit\sec_companyfacts_availability.py"
python "Paper 4\src\collect\sec_filing_text_manifest.py"
python "Paper 4\src\audit\sec_all8k_text_plan.py"
python "Paper 4\src\collect\sec_all8k_text_batch_collector.py"
python "Paper 4\src\audit\sec_all8k_storage_preflight.py"
python "Paper 4\src\audit\sec_all8k_text_sample_design.py"
python "Paper 4\src\collect\sec_all8k_sample_text_collector.py"
python "Paper 4\src\model\sec_all8k_sample_text_baseline.py"
python "Paper 4\src\audit\sec_validation_scope.py"
```

The companyfacts availability audit is fast because it checks file inventory and validation-CIK coverage only. The full SEC financial-ratio baseline parses local companyfacts payloads and is intentionally treated as heavy. The current workspace already has this heavy output generated:

```powershell
python "Paper 4\src\model\sec_desktop_financial_baseline.py"
python "Paper 4\src\model\sec_desktop_text_metadata_baseline.py"
python "Paper 4\src\model\sec_filing_body_text_baseline.py"
```

The DART market-price extension is also heavy because it can fetch supplemental Naver fchart coverage, recover Naver no-row gaps with FinanceDataReader, collect a KOSPI/KOSDAQ market-index benchmark, read the external KY-PLATFORM/Company_Credit OHLCV SQLite database, and retrain the DART baselines with market features. The supplemental collectors save rows inside `Paper 4\data\raw\market_coverage\`, the index collector saves rows inside `Paper 4\data\raw\market_index\`, and neither step mutates the external OHLCV source database:

```powershell
python "Paper 4\src\collect\market_coverage_collector.py"
python "Paper 4\src\collect\market_fdr_gap_collector.py"
python "Paper 4\src\collect\market_index_collector.py"
python "Paper 4\src\model\build_market_features.py"
python "Paper 4\src\model\train_dart_baselines.py" --panel-path "Paper 4\data\model_market\dart_model_panel_market.parquet" --summary-path "Paper 4\data\model\dart_model_panel_summary.json" --results-dir "Paper 4\results\market" --n-bootstrap 300
python "Paper 4\src\model\build_market_features.py" --panel-path "Paper 4\data\model_robust\dart_model_panel.parquet" --out-dir "Paper 4\data\model_market_robust"
python "Paper 4\src\model\train_dart_baselines.py" --panel-path "Paper 4\data\model_market_robust\dart_model_panel_market.parquet" --summary-path "Paper 4\data\model_robust\dart_model_panel_summary.json" --results-dir "Paper 4\results\market_robust" --n-bootstrap 300
```

The SEC filing-body text manifest is prepared without network calls by default. To download the target-item filing-body cache from SEC Archives, set a real identifying user agent and pass `--execute`:

```powershell
$env:SEC_USER_AGENT="Your Name your.email@example.com"
python "Paper 4\src\collect\sec_filing_text_manifest.py" --execute --max-downloads 5859
python "Paper 4\src\model\sec_filing_body_text_baseline.py"
```

The full all-8-K filing-body collector is status-only by default and makes no SEC Archives requests. It also summarizes any historical bounded download log, so a default run can verify prior smoke/batch execution evidence without making new requests. Use it first to inspect open batches:

```powershell
python "Paper 4\src\collect\sec_all8k_text_batch_collector.py"
python "Paper 4\src\audit\sec_all8k_storage_preflight.py"
```

To execute a bounded future batch, first confirm `results/sec_all8k_storage_preflight.md` reports storage and user-agent readiness, then configure a real identifying SEC user agent and explicitly cap the download count:

```powershell
$env:SEC_USER_AGENT="Your Name your.email@example.com"
python "Paper 4\src\collect\sec_all8k_text_batch_collector.py" --execute --batch-id 1 --max-downloads 100
```

The collector reconstructs cache paths from CIK and accession rather than trusting public-facing `local_text_path` values, so path redaction in `sec_all8k_text_manifest.csv` cannot redirect future downloads or cache checks.

EDINET remains Phase 2. The default command only creates a plan and does not call the API:

```powershell
python "Paper 4\src\collect\edinet_collector.py"
python "Paper 4\src\collect\edinet_feasibility.py"
python "Paper 4\src\audit\edinet_execution_preflight.py"
python "Paper 4\src\audit\edinet_local_discovery.py"
python "Paper 4\src\prepare\jp_report_parser_prototype.py"
```

To actually collect EDINET document-list metadata, configure `EDINET_API_KEY` and explicitly execute:

```powershell
$env:EDINET_API_KEY="..."
python "Paper 4\src\collect\edinet_collector.py" --execute
```

As of 2026-07-02, the EDINET feasibility report flags 548 days in the 2015-2024 plan as outside the approximate live API 10-year date window; the no-key preflight status is `blocked_without_subscription_key`. `results/edinet_execution_preflight.md` keeps empirical EDINET use in v2 scope until an API key, archive/prior-collection coverage for older planned dates, and nonzero document/current-report rows are present.

The focused local discovery audit reports 0 external reusable 2015-2024 EDINET document-list/current-report files. It also records 16 local 2025 Japanese report PDF/text candidates, including one text file mentioning EDINET. The parser prototype indexes 8 local Japanese report PDFs and 8 text files and extracts 30 taxonomy keyword candidate rows. Generated Paper 4 EDINET collection plans/status files and parser-prototype outputs are not empirical EDINET data.

RQ1 alignment validation is not complete until `data/processed/manual_alignment_seed_100.csv` has human labels.

Workflow:

```powershell
python "Paper 4\src\align\llm_aligner.py"
python "Paper 4\src\align\alignment_annotation_qc.py"
python "Paper 4\src\align\alignment_candidate_diagnostics.py"
python "Paper 4\src\align\alignment_silver_evaluation.py"
python "Paper 4\src\align\sec_desktop_alignment_seed.py"
python "Paper 4\src\align\build_single_expert_label_sheet.py"
python "Paper 4\src\align\build_annotation_packet.py"
python "Paper 4\src\align\build_annotation_workbook.py"
```

Fast single-expert route:

```powershell
# Fill human_label in manual_alignment_seed_100_single_expert_label_sheet.xlsx first.
python "Paper 4\src\align\sync_single_expert_label_sheet.py"
python "Paper 4\src\align\evaluate_alignment.py"
```

Two-annotator route:

```powershell
# Fill annotator1_label and annotator2_label independently, then adjudicated_label for disagreements.
python "Paper 4\src\align\sync_annotation_workbook.py"
python "Paper 4\src\align\evaluate_alignment.py"
```

`llm_aligner.py` defaults to offline prompt/export mode and writes `data/processed/llm_alignment_prompts.jsonl`, `data/processed/llm_alignment_results.csv`, and `results/llm_alignment_summary.md`. Use `--mode anthropic` only when an `ANTHROPIC_API_KEY` is intentionally configured.

`alignment_annotation_qc.py` writes a priority-sorted full queue and a 30-row first-pass queue. These files tell annotators what to label first; they do not create gold labels.

`alignment_candidate_diagnostics.py` writes pre-label coverage and ambiguity diagnostics. The current seed covers T02/T03/T05/T06/T99, excludes T01/T04, and flags 91 high-ambiguity rows; these diagnostics are triage evidence only.

`alignment_silver_evaluation.py` compares deterministic seed labels with offline LLM judgments. The current run has 11 strict silver-consensus rows and 89 rows still requiring human review. This is triage evidence only, not Precision@1/MRR.

`sec_desktop_alignment_seed.py` writes a supplemental T01-inclusive annotation seed using the broad local SEC population scan. The current output has 100 rows, 54 same-code candidates, 46 negative candidates, and 5 T01 candidates. It is annotation readiness only, not Precision@1/MRR evidence.

The same generic evaluator is run on the supplemental seed:

```powershell
python "Paper 4\src\align\evaluate_alignment.py" --seed-path "Paper 4\data\processed\manual_alignment_seed_sec_desktop_100.csv" --out-json "Paper 4\results\alignment_sec_desktop_evaluation.json" --out-md "Paper 4\results\alignment_sec_desktop_evaluation.md"
```

`alignment_sec_desktop_evaluation.md` should remain `needs_human_labels` until supplemental labels are filled.

`build_single_expert_label_sheet.py` writes a fast single-expert labeling sheet and guide. Filling `human_label` there and running `sync_single_expert_label_sheet.py` supports Precision@1/MRR with a single-expert-gold caveat, but it still does not support Cohen's kappa.

`build_annotation_workbook.py` writes an Excel workbook with dropdown labels, AI-suggested review aids, annotator1/annotator2 columns, an adjudicated-label column, and QA formulas. AI suggestions are not human labels.

`evaluate_alignment.py` reports Precision@1/MRR from `adjudicated_label`, legacy `human_label`, or exact annotator consensus. It also reports annotator overlap, exact agreement, and Cohen's kappa when two annotators are available. Rows where only one annotator has labeled, or where annotators disagree, are excluded from final RQ1 metrics until the second label or `adjudicated_label` is filled.

`alignment_label_preflight.py` consolidates the main seed, single-expert sheet, two-annotator workbook, and supplemental SEC seed into one readiness gate. Do not report Precision@1 or MRR while `results/alignment_evaluation.md` says labels are still required or while `results/alignment_label_preflight.md` is not `ready_for_rq1_metrics`.

## Heavy Steps

The following steps are intentionally skipped unless `--include-heavy` is passed:

- Desktop-wide data audit.
- Main DART model training.
- DART market-price supplemental OHLCV collection.
- DART market-price FDR gap OHLCV collection.
- DART KOSPI/KOSDAQ market-index collection.
- DART market-price feature generation.
- DART market-price model training.
- Robust DART model training.
- Robust DART market-price feature generation.
- Robust DART market-price model training.
- Permutation importance.
- SEC desktop sequence-only validation.
- SEC desktop firm-disjoint temporal robustness.
- SEC desktop companyfacts financial-ratio baseline.
- SEC desktop metadata-text baseline.
- SEC desktop target-item filing-body text baseline.

This prevents accidental long reruns when only checking the manuscript or final outputs.

## Core Evidence Files

| Purpose | File |
|---|---|
| Completion audit | `paper4_completion_audit.md` |
| Data map | `data_map.md` |
| DART gap/outcome derivation | `results/dart_gap_collection_status.md`; `data/raw/dart_gap/dart_gap_outcome_candidates.csv` |
| EDINET collection status | `results/edinet_collection_status.md` |
| EDINET API feasibility | `results/edinet_api_feasibility.md` |
| EDINET execution preflight | `results/edinet_execution_preflight.md`; `results/edinet_execution_preflight.json` |
| EDINET local discovery | `results/edinet_local_discovery.md`; `data/audit/edinet_local_candidate_files.csv` |
| Japanese report parser prototype | `results/jp_report_parser_prototype_summary.md`; `data/raw/edinet/jp_report_parser_prototype_index.csv`; `data/processed/jp_report_parser_prototype_events.csv` |
| T04 local discovery | `results/t04_local_discovery.md`; `data/audit/t04_local_candidate_files.csv` |
| Main model results | `results/dart_model_metrics.md` |
| Market-price extension | `data/raw/market_coverage/naver_ohlcv_supplement_summary.md`; `data/raw/market_coverage/fdr_ohlcv_gap_supplement_summary.md`; `data/raw/market_index/korea_market_index_summary.md`; `results/market_extension_summary.md`; `results/market_residual_coverage_audit.md`; `results/market/dart_model_metrics.md`; `results/market_robust/dart_model_metrics.md`; `data/model_market/dart_model_panel_market_summary.json` |
| Survival/time-to-event check | `results/survival_hazard_summary.md` |
| Robustness | `results/dart_robustness_summary.md` |
| SupTech rule engine | `results/flagging_rule_engine_summary.md`; `results/flagging_rule_registry.csv` |
| Flagging comparison | `results/flagging_performance.md` |
| Classic baseline audit | `results/classic_financial_baselines.md` |
| Alignment status | `results/alignment_summary.md`; `results/alignment_evaluation.md` |
| LLM alignment prompt/export | `results/llm_alignment_summary.md`; `data/processed/llm_alignment_prompts.jsonl` |
| Alignment annotation QC | `results/alignment_annotation_qc.md`; `data/processed/manual_alignment_seed_100_annotation_priority.csv`; `data/processed/manual_alignment_seed_100_first_pass_30.csv` |
| Alignment candidate diagnostics | `results/alignment_candidate_diagnostics.md`; `data/processed/manual_alignment_seed_100_candidate_diagnostics.csv` |
| Alignment silver evaluation | `results/alignment_silver_evaluation.md`; `data/processed/manual_alignment_seed_100_silver_review_queue.csv`; `data/processed/manual_alignment_seed_100_silver_consensus.csv` |
| Local SEC supplemental alignment seed/evaluation | `results/alignment_sec_desktop_seed_summary.md`; `results/alignment_sec_desktop_evaluation.md`; `data/processed/manual_alignment_seed_sec_desktop_100.csv`; `data/processed/manual_alignment_seed_sec_desktop_100.xlsx`; `manuscript/alignment_sec_desktop_labeling_guide.md` |
| Single-expert alignment sheet | `results/alignment_single_expert_readiness.md`; `data/processed/manual_alignment_seed_100_single_expert_label_sheet.xlsx`; `manuscript/alignment_single_expert_labeling_guide.md` |
| Alignment annotation workbook | `data/processed/manual_alignment_seed_100_annotation_workbook.xlsx`; `results/alignment_annotation_readiness.md` |
| Alignment label preflight | `results/alignment_label_preflight.md`; `results/alignment_label_preflight.json` |
| SEC standalone-validation gap | `results/sec_gap_inventory.md` |
| SEC expanded collection status | `results/sec_expanded_collection_status.md` |
| SEC expanded sequence summary | `results/sec_expanded_sequence_summary.md` |
| SEC bankruptcy case sequences | `results/sec_bankruptcy_case_sequence_summary.md`; `results/sec_bankruptcy_case_sequences.md` |
| SEC desktop population scan | `results/sec_desktop_population_collection_status.md`; `results/sec_desktop_population_sequence_summary.md` |
| SEC desktop sampling frame | `results/sec_desktop_sampling_frame_summary.md`; `data/model_sec_desktop/sec_desktop_sampling_frame_firms.csv`; `data/model_sec_desktop/sec_desktop_sampling_frame_anchors.csv` |
| SEC desktop validation | `results/sec_desktop_validation_summary.md`; `results/sec_desktop_validation_firm_disjoint_summary.md` |
| SEC companyfacts availability | `results/sec_companyfacts_availability.md`; `data/model_sec_desktop/sec_companyfacts_validation_coverage.csv` |
| SEC financial baseline | `results/sec_desktop_financial_baseline_summary.md`; `results/sec_desktop_financial_baseline_metrics.csv`; `data/model_sec_desktop/sec_desktop_financial_panel.parquet` |
| SEC filing text manifest | `results/sec_filing_text_collection_status.md`; `data/raw/sec_filing_text/sec_filing_text_manifest.csv` |
| SEC all-8-K text plan | `results/sec_all8k_text_scope_plan.md`; `data/raw/sec_all8k_text/sec_all8k_text_manifest.csv`; `data/raw/sec_all8k_text/sec_all8k_text_batch_plan.csv` |
| SEC all-8-K text batch status | `results/sec_all8k_text_batch_collection_status.md`; `data/raw/sec_all8k_text/sec_all8k_text_batch_status.csv` |
| SEC all-8-K full-scope download preflight | `results/sec_all8k_storage_preflight.md`; `results/sec_all8k_storage_preflight.json` |
| SEC all-8-K text sample design | `results/sec_all8k_text_sample_design.md`; `data/raw/sec_all8k_text/sec_all8k_text_sample_plan.csv`; `data/raw/sec_all8k_text/sec_all8k_text_sample_strata.csv` |
| SEC all-8-K sample text status | `results/sec_all8k_text_sample_collection_status.md`; `data/raw/sec_all8k_text/sec_all8k_text_sample_status.csv` |
| SEC all-8-K sample text baseline | `results/sec_all8k_sample_text_baseline_summary.md`; `results/sec_all8k_sample_text_baseline_metrics.csv`; `data/model_sec_desktop/sec_all8k_sample_text_features.csv` |
| SEC metadata-text baseline | `results/sec_desktop_text_metadata_baseline_summary.md`; `results/sec_desktop_text_metadata_baseline_metrics.csv`; `data/model_sec_desktop/sec_desktop_text_metadata_panel.parquet` |
| SEC filing-body text baseline | `results/sec_filing_body_text_baseline_summary.md`; `results/sec_filing_body_text_baseline_metrics.csv`; `data/model_sec_desktop/sec_filing_body_text_panel.parquet` |
| SEC validation scope | `results/sec_validation_scope.md`; `data/model_sec_desktop/sec_validation_scope_decision_table.csv` |
| Manuscript | `manuscript/paper4_v1_draft.md`; `manuscript/paper4_v1_draft.docx` |
| Korean submission candidate table | `manuscript/submission_candidate_journals_ko.md`; `manuscript/submission_candidate_journals_ko.docx` |
| Journal candidate source audit | `results/journal_candidate_source_audit.md`; `results/journal_candidate_source_audit.json` |
| Manuscript metric audit | `results/manuscript_metric_audit.md`; `results/manuscript_metric_audit.json` |
| Manuscript claim audit | `results/manuscript_claim_audit.md`; `manuscript/claim_readiness_matrix.md` |
| Bilingual manuscript audit | `results/bilingual_manuscript_audit.md`; `results/bilingual_manuscript_audit.json` |
| Submission package audit | `results/submission_package_audit.md`; `results/submission_package_audit.json` |
| Word export audit | `results/word_export_audit.md`; `results/word_export_audit.json` |
| Research-plan traceability audit | `results/research_plan_traceability.md`; `results/research_plan_traceability.json` |
| Public submission manifest audit | `results/public_submission_manifest_audit.md`; `results/public_submission_manifest_audit.json`; `data/audit/public_submission_manifest.csv` |
| Remaining risk audit | `results/remaining_risk_audit.md`; `results/remaining_risk_audit.json` |
| Final claim-boundary audit | `results/final_claim_boundary_audit.md`; `results/final_claim_boundary_audit.json`; 46 Markdown files including every public-submission-manifest Markdown file such as `results/market_residual_coverage_audit.md`, plus `paper4_research_plan.md`, `results/public_submission_manifest_audit.md`, `results/remaining_risk_audit.md`, `results/post_run_artifact_status.md`, `submission_package/email_handoff_receipt_audit.md`, and `submission_package/LATEST_PACKAGE.md` |
| Reproducibility run-report self-audit | `results/reproducibility_run_report_audit.md`; `results/reproducibility_run_report_audit.json` |
| Post-run artifact status | `results/post_run_artifact_status.md`; `results/post_run_artifact_status.json`; preserves handoff message ID, thread ID, display URL, timestamp, subject, attachment filename, SHA, and size |
| Latest core submission package | `submission_package/paper4_core_package_latest.zip`; `submission_package/paper4_core_package_latest_audit.json`; `submission_package/LATEST_PACKAGE.md`; post-run package generated after the run-report self-audit; pointer records latest ZIP path, SHA, size, local UTC regeneration timestamp, and superseded ZIP status |
| Email handoff receipt audit | `submission_package/email_handoff_receipt.json`; `submission_package/email_handoff_receipt_audit.md`; `submission_package/email_handoff_receipt_audit.json`; post-run local receipt check generated after the latest package build; verifies latest pointer SHA/size/timestamp plus handoff metadata; pipeline report-only, strict via `src/audit/post_run_artifact_status.py --require-email-handoff` after receipt refresh |

## Interpretation Guardrails

- Describe Altman/Ohlson outputs as available-component proxies, not exact original scores.
- Describe DART-SEC alignment as deterministic taxonomy coverage until human labels are completed.
- Treat `llm_alignment_results.csv` as candidate judgments only; it is not a substitute for human labels.
- Treat `alignment_candidate_diagnostics.md` as annotation triage evidence only; it is not an accuracy result.
- Treat `alignment_silver_evaluation.md` as deterministic-vs-offline-LLM pre-label triage only; it is not a human-gold accuracy result.
- Treat `alignment_label_preflight.md` as the reporting gate for RQ1 metrics; until it is ready, Precision@1, MRR, Cohen's kappa, and inter-annotator agreement remain deferred v2 metrics.
- Treat T04 going-concern/audit-opinion rules as registry-only in v1; `results/t04_local_discovery.md` currently reports 0 empirical local T04 source candidates and 0 DART title/content matches.
- Treat the single-expert sheet as a fast gold-label route only after `human_label` is actually filled and synced; do not use it for Cohen's kappa.
- Treat Cohen's kappa and inter-annotator agreement as unavailable until both annotator columns have labels.
- Describe the survival result as a discrete-time landmark hazard check; do not claim a full multivariable Cox PH result.
- Describe the OHLCV extension as a coverage-expanded market-control and incremental-signal layer. The supplemental Naver fchart collection adds 2,067,411 rows for 2,154 of 2,648 targeted symbols and the FDR gap collector adds 411,423 rows for 425 of the 494 Naver no-row symbols, leaving 69 residual no-row symbols. The residual audit maps those symbols to 473 main-panel anchors (2.37%) and 3/5 residual 365/730-day positive anchors. The extension raises main any-price coverage to 96.9% and robust any-price coverage to 97.0%. Market-adjusted returns now use a collected 5,406-row KOSPI/KOSDAQ benchmark with 99.9% benchmark coverage in the loaded OHLCV rows, but the extension is still not a causal price-reaction or security-level event-study abnormal-return design.
- Describe rule-engine outputs as auditable triggers; use `flagging_performance.md` for the comparison against model-based watchlists.
- Keep SEC as taxonomy-transfer/v2 evidence in the v1 manuscript. The 132-CIK expanded SEC pull has 8,169 filing rows and a usable sequence panel but no Item 1.03 rows. The local SEC scan adds 429,100 8-K/8-K/A rows, 5,085 firms, 97 Item 1.03 rows, and cached target-window archive metadata coverage, plus a formal local sampling frame, sequence-only validation, firm-disjoint robustness, a companyfacts financial baseline, a metadata-text baseline, a filing-body text download manifest, a fully cached 5,859-file target-item filing-body baseline, a bounded non-target all-8-K filing-body sample design, and a status-only full all-8-K batch tracker. The full companyfacts baseline extracts 5,812,678 selected fact records and reaches all-distress 730-day sequence-plus-financial random-forest ROC AUC 0.8560/top-decile lift 5.19. The metadata-text-only 730-day all-distress random forest reaches ROC AUC 0.7828/top-decile lift 3.88. The filing-body-only target-item baseline is weak at ROC AUC 0.5000/top-decile lift 0.46 with transparent keyword/count features. SEC should not be described as finalized U.S. predictive validation until full all-8-K filing-body scope and final standalone U.S. validation scope are completed.
- `sec_validation_scope.py` documents the current SEC text boundary: target-item filing-body accession coverage is 100% for Items 1.03/2.04/3.01, and the current target-plus-sample/bounded-batch cache covers 6,944 of 429,100 all-8-K rows. Do not call it full all-8-K text validation.
- `sec_all8k_text_plan.py` documents the full all-8-K text work remaining: 422,156 filing bodies are still missing after the bounded sample cache and capped batch-download smoke runs, estimated at about 208.757 GB from the currently cached filing bodies. This is a planning artifact only and performs no downloads.
- `sec_all8k_text_batch_collector.py` is status-only by default and reports 85 open 5,000-row batches, next batch 1, and 0 downloads in the default run; the historical log currently records 225 successful bounded download rows resolving to existing local files. It performs SEC Archives downloads only with `--execute --batch-id N --max-downloads K` and a real identifying SEC user agent, and reconstructs cache paths from CIK/accession when public manifest paths are redacted.
- `sec_all8k_storage_preflight.py` performs no downloads and keeps full-scope readiness in v2 unless the current disk has enough storage headroom and an explicit `SEC_USER_AGENT` or `EDGAR_USER_AGENT` is configured.
- `sec_all8k_text_sample_design.py` defines the bounded sample option: 960 non-target rows plus 119 target-reference rows across all 110 year-item strata.
- `sec_all8k_sample_text_collector.py` is status-only by default. The current local cache has all 1,079 sample filing bodies; use `--execute --max-downloads N` only when new missing files appear and a real identifying SEC user agent is available.
- `sec_all8k_sample_text_baseline.py` models the bounded sample as target-item-vs-non-target text sensitivity. Do not treat this as full all-8-K text validation or standalone U.S. distress prediction.
- Use the targeted SEC bankruptcy case sample only as case evidence: it has 84 Item 1.03 rows across 35 firms, but it is selected by known bankruptcy status and is not a population validation sample.
- Keep EDINET in v2 unless Japanese disclosure data are collected and normalized; current EDINET status is dry-run planning only, no-key preflight status is `blocked_without_subscription_key`, older 2015-early 2016 dates require an archive/prior-collection strategy, and the Japanese report parser prototype is parser-design support only.
- Keep the headline v1 contribution on DART supervisory watchlist ranking.
