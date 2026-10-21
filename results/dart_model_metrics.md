# DART Early-Warning Baseline Results

Split: train anchor years <= 2021; test anchor years >= 2022. Rows whose full outcome horizon is not observable are excluded per horizon.

| Horizon | Model | Train rows | Test rows | Test positives | ROC AUC | Avg precision | Brier | Top 10% precision | Top 10% recall | Top 10% lift |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 365 | sequence_logit | 12312 | 7612 | 55 | 0.7761 | 0.0267 | 0.1285 | 0.0315 | 0.4364 | 4.36 |
| 365 | financial_logit | 12312 | 7612 | 55 | 0.7285 | 0.0185 | 0.1807 | 0.0249 | 0.3455 | 3.45 |
| 365 | combined_logit | 12312 | 7612 | 55 | 0.776 | 0.0313 | 0.1278 | 0.0302 | 0.4182 | 4.18 |
| 365 | combined_random_forest | 12312 | 7612 | 55 | 0.8922 | 0.071 | 0.0105 | 0.0486 | 0.6727 | 6.72 |
| 365 | combined_calibrated_random_forest | 12312 | 7612 | 55 | 0.8882 | 0.0696 | 0.0071 | 0.0486 | 0.6727 | 6.72 |
| 730 | sequence_logit | 12312 | 4961 | 63 | 0.754 | 0.0497 | 0.1355 | 0.0584 | 0.4603 | 4.59 |
| 730 | financial_logit | 12312 | 4961 | 63 | 0.7303 | 0.0403 | 0.1903 | 0.0423 | 0.3333 | 3.33 |
| 730 | combined_logit | 12312 | 4961 | 63 | 0.7996 | 0.0595 | 0.1293 | 0.0543 | 0.4286 | 4.28 |
| 730 | combined_random_forest | 12312 | 4961 | 63 | 0.8941 | 0.1022 | 0.0176 | 0.0765 | 0.6032 | 6.02 |
| 730 | combined_calibrated_random_forest | 12312 | 4961 | 63 | 0.8997 | 0.0993 | 0.0122 | 0.0845 | 0.6667 | 6.65 |

## Best Current Signal

- 365 days: `combined_random_forest` has average precision 0.0710; top-decile precision 0.0486, recall 0.6727, lift 6.72.
- 730 days: `combined_random_forest` has average precision 0.1022; top-decile precision 0.0765, recall 0.6032, lift 6.02.

## Bootstrap Confidence Intervals

Bootstrap intervals use row-level resampling of the out-of-time test set.

| Horizon | Model | Metric | Mean | 95% CI low | 95% CI high | Valid resamples |
|---:|---|---|---:|---:|---:|---:|
| 365 | sequence_logit | `roc_auc` | 0.7775 | 0.7037 | 0.8390 | 500 |
| 365 | sequence_logit | `average_precision` | 0.0303 | 0.0177 | 0.0549 | 500 |
| 365 | sequence_logit | `brier_score` | 0.1283 | 0.1244 | 0.1321 | 500 |
| 365 | sequence_logit | `top_10pct_recall` | 0.4274 | 0.2910 | 0.5569 | 500 |
| 365 | sequence_logit | `top_10pct_lift` | 4.2699 | 2.9066 | 5.5633 | 500 |
| 365 | financial_logit | `roc_auc` | 0.7293 | 0.6692 | 0.7839 | 500 |
| 365 | financial_logit | `average_precision` | 0.0201 | 0.0122 | 0.0298 | 500 |
| 365 | financial_logit | `brier_score` | 0.1807 | 0.1762 | 0.1858 | 500 |
| 365 | financial_logit | `top_10pct_recall` | 0.3458 | 0.2348 | 0.4643 | 500 |
| 365 | financial_logit | `top_10pct_lift` | 3.4542 | 2.3457 | 4.6380 | 500 |
| 365 | combined_logit | `roc_auc` | 0.7772 | 0.7228 | 0.8320 | 500 |
| 365 | combined_logit | `average_precision` | 0.0357 | 0.0179 | 0.0652 | 500 |
| 365 | combined_logit | `brier_score` | 0.1276 | 0.1228 | 0.1323 | 500 |
| 365 | combined_logit | `top_10pct_recall` | 0.4180 | 0.2766 | 0.5544 | 500 |
| 365 | combined_logit | `top_10pct_lift` | 4.1755 | 2.7631 | 5.5385 | 500 |
| 365 | combined_random_forest | `roc_auc` | 0.8931 | 0.8498 | 0.9328 | 500 |
| 365 | combined_random_forest | `average_precision` | 0.0768 | 0.0489 | 0.1140 | 500 |
| 365 | combined_random_forest | `brier_score` | 0.0105 | 0.0090 | 0.0120 | 500 |
| 365 | combined_random_forest | `top_10pct_recall` | 0.6742 | 0.5452 | 0.7985 | 500 |
| 365 | combined_random_forest | `top_10pct_lift` | 6.7348 | 5.4466 | 7.9768 | 500 |
| 365 | combined_calibrated_random_forest | `roc_auc` | 0.8886 | 0.8479 | 0.9294 | 500 |
| 365 | combined_calibrated_random_forest | `average_precision` | 0.0740 | 0.0465 | 0.1060 | 500 |
| 365 | combined_calibrated_random_forest | `brier_score` | 0.0071 | 0.0053 | 0.0089 | 500 |
| 365 | combined_calibrated_random_forest | `top_10pct_recall` | 0.6789 | 0.5477 | 0.8000 | 500 |
| 365 | combined_calibrated_random_forest | `top_10pct_lift` | 6.7816 | 5.4717 | 7.9916 | 500 |
| 730 | sequence_logit | `roc_auc` | 0.7551 | 0.6846 | 0.8190 | 500 |
| 730 | sequence_logit | `average_precision` | 0.0538 | 0.0331 | 0.0806 | 500 |
| 730 | sequence_logit | `brier_score` | 0.1356 | 0.1309 | 0.1404 | 500 |
| 730 | sequence_logit | `top_10pct_recall` | 0.4661 | 0.3594 | 0.5856 | 500 |
| 730 | sequence_logit | `top_10pct_lift` | 4.6521 | 3.5872 | 5.8453 | 500 |
| 730 | financial_logit | `roc_auc` | 0.7289 | 0.6717 | 0.7842 | 500 |
| 730 | financial_logit | `average_precision` | 0.0443 | 0.0268 | 0.0687 | 500 |
| 730 | financial_logit | `brier_score` | 0.1903 | 0.1848 | 0.1959 | 500 |
| 730 | financial_logit | `top_10pct_recall` | 0.3381 | 0.2296 | 0.4581 | 500 |
| 730 | financial_logit | `top_10pct_lift` | 3.3746 | 2.2920 | 4.5730 | 500 |
| 730 | combined_logit | `roc_auc` | 0.7981 | 0.7420 | 0.8494 | 500 |
| 730 | combined_logit | `average_precision` | 0.0638 | 0.0367 | 0.0973 | 500 |
| 730 | combined_logit | `brier_score` | 0.1293 | 0.1243 | 0.1347 | 500 |
| 730 | combined_logit | `top_10pct_recall` | 0.4286 | 0.2991 | 0.5447 | 500 |
| 730 | combined_logit | `top_10pct_lift` | 4.2779 | 2.9857 | 5.4371 | 500 |
| 730 | combined_random_forest | `roc_auc` | 0.8938 | 0.8565 | 0.9236 | 500 |
| 730 | combined_random_forest | `average_precision` | 0.1098 | 0.0697 | 0.1652 | 500 |
| 730 | combined_random_forest | `brier_score` | 0.0177 | 0.0155 | 0.0201 | 500 |
| 730 | combined_random_forest | `top_10pct_recall` | 0.6067 | 0.4857 | 0.7248 | 500 |
| 730 | combined_random_forest | `top_10pct_lift` | 6.0565 | 4.8480 | 7.2352 | 500 |
| 730 | combined_calibrated_random_forest | `roc_auc` | 0.9000 | 0.8677 | 0.9286 | 500 |
| 730 | combined_calibrated_random_forest | `average_precision` | 0.1080 | 0.0679 | 0.1536 | 500 |
| 730 | combined_calibrated_random_forest | `brier_score` | 0.0123 | 0.0095 | 0.0153 | 500 |
| 730 | combined_calibrated_random_forest | `top_10pct_recall` | 0.6691 | 0.5535 | 0.7822 | 500 |
| 730 | combined_calibrated_random_forest | `top_10pct_lift` | 6.6787 | 5.5250 | 7.8082 | 500 |

## Calibration Snapshot

The table below shows the highest-risk decile only; full decile tables are saved in `dart_model_calibration.csv`.

| Horizon | Model | Top bin count | Mean predicted risk | Observed rate | Positives |
|---:|---|---:|---:|---:|---:|
| 365 | combined_calibrated_random_forest | 761 | 0.0193 | 0.0486 | 37 |
| 365 | combined_logit | 761 | 0.8061 | 0.0302 | 23 |
| 365 | combined_random_forest | 761 | 0.1963 | 0.0486 | 37 |
| 365 | financial_logit | 761 | 0.8003 | 0.0250 | 19 |
| 365 | sequence_logit | 761 | 0.7461 | 0.0315 | 24 |
| 730 | combined_calibrated_random_forest | 496 | 0.0407 | 0.0847 | 42 |
| 730 | combined_logit | 496 | 0.7790 | 0.0544 | 27 |
| 730 | combined_random_forest | 496 | 0.2706 | 0.0766 | 38 |
| 730 | financial_logit | 496 | 0.7915 | 0.0423 | 21 |
| 730 | sequence_logit | 496 | 0.7508 | 0.0585 | 29 |

## Strongest Combined Logistic Coefficients

### 365 days

| Feature | Coefficient |
|---|---:|
| `fin_operating_margin` | 3.4141 |
| `fin_loss_flag` | 2.3381 |
| `fin_log_net_income_signed` | 2.2852 |
| `seq_days_since_last_T05` | 0.9191 |
| `fin_missing_record_flag` | 0.9096 |
| `seq_distinct_taxonomies_to_date` | 0.8208 |
| `seq_days_since_last_T99` | 0.7915 |
| `seq_T99_180d` | 0.6833 |
| `fin_log_operating_income_signed` | -0.6717 |
| `seq_days_since_last_event` | -0.6648 |
| `seq_T99_365d` | -0.6189 |
| `seq_days_since_last_T06` | 0.4772 |

### 730 days

| Feature | Coefficient |
|---|---:|
| `fin_operating_margin` | 3.5292 |
| `seq_days_since_last_event` | -0.5331 |
| `fin_missing_record_flag` | 0.4870 |
| `seq_days_since_last_T05` | 0.4841 |
| `seq_distinct_taxonomies_to_date` | 0.4833 |
| `fin_loss_flag` | 0.4467 |
| `seq_days_since_last_T06` | 0.4289 |
| `fin_equity_ratio` | -0.4167 |
| `seq_days_since_last_T99` | 0.4164 |
| `fin_log_operating_income_signed` | -0.3846 |
| `fin_log_net_income_signed` | 0.2958 |
| `fin_operating_roa` | -0.2760 |

## Strongest Random Forest Feature Importances

### 365 days

| Feature | Importance |
|---|---:|
| `seq_days_since_last_event` | 0.0631 |
| `seq_days_since_last_T05` | 0.0592 |
| `fin_operating_roa` | 0.0571 |
| `fin_log_operating_income_signed` | 0.0539 |
| `seq_T05_730d` | 0.0535 |
| `fin_net_margin` | 0.0530 |
| `fin_log_net_income_signed` | 0.0506 |
| `fin_operating_margin` | 0.0493 |
| `seq_T05_to_date` | 0.0414 |
| `fin_log_total_assets` | 0.0399 |
| `fin_roa` | 0.0384 |
| `seq_days_since_last_T99` | 0.0383 |

### 730 days

| Feature | Importance |
|---|---:|
| `seq_days_since_last_T05` | 0.0606 |
| `seq_days_since_last_event` | 0.0584 |
| `seq_T05_730d` | 0.0571 |
| `fin_log_operating_income_signed` | 0.0546 |
| `fin_operating_roa` | 0.0519 |
| `fin_log_net_income_signed` | 0.0512 |
| `fin_operating_margin` | 0.0437 |
| `fin_log_total_assets` | 0.0426 |
| `fin_roa` | 0.0423 |
| `fin_net_margin` | 0.0420 |
| `seq_T05_to_date` | 0.0418 |
| `fin_equity_ratio` | 0.0385 |

