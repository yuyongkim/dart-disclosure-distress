# Permutation Importance

Permutation importance is computed on the out-of-time test split for the combined random forest. It complements the built-in random-forest feature importance table because it measures score degradation after shuffling each feature.

## Baseline Scores

| Horizon | Test rows | Test positives | ROC AUC | Average precision | Top 10% lift |
|---:|---:|---:|---:|---:|---:|
| 365 | 7,612 | 55 | 0.8922 | 0.0710 | 6.72 |
| 730 | 4,961 | 63 | 0.8941 | 0.1022 | 6.02 |

## Top Features by average_precision

| Horizon | Rank | Feature | Group | Importance mean | Std |
|---:|---:|---|---|---:|---:|
| 365 | 1 | `seq_days_since_last_T06` | sequence | 0.00523 | 0.00296 |
| 365 | 2 | `fin_log_total_assets` | financial | 0.00513 | 0.00131 |
| 365 | 3 | `fin_log_revenue` | financial | 0.00413 | 0.00150 |
| 365 | 4 | `seq_events_to_date` | sequence | 0.00301 | 0.00197 |
| 365 | 5 | `seq_days_since_last_event` | sequence | 0.00267 | 0.00132 |
| 365 | 6 | `seq_days_since_last_T99` | sequence | 0.00235 | 0.00175 |
| 365 | 7 | `fin_operating_margin` | financial | 0.00213 | 0.00172 |
| 365 | 8 | `seq_T06_365d` | sequence | 0.00165 | 0.00015 |
| 365 | 9 | `seq_T99_to_date` | sequence | 0.00141 | 0.00127 |
| 365 | 10 | `fin_net_margin` | financial | 0.00136 | 0.00344 |
| 365 | 11 | `seq_T06_730d` | sequence | 0.00110 | 0.00090 |
| 365 | 12 | `seq_T05_to_date` | sequence | 0.00102 | 0.00267 |
| 365 | 13 | `seq_days_since_last_T05` | sequence | 0.00098 | 0.00184 |
| 365 | 14 | `seq_distinct_taxonomies_730d` | sequence | 0.00041 | 0.00040 |
| 365 | 15 | `seq_T99_365d` | sequence | 0.00041 | 0.00040 |
| 730 | 1 | `fin_log_total_assets` | financial | 0.01258 | 0.00362 |
| 730 | 2 | `seq_days_since_last_T06` | sequence | 0.00867 | 0.00112 |
| 730 | 3 | `seq_events_to_date` | sequence | 0.00783 | 0.00353 |
| 730 | 4 | `fin_operating_roa` | financial | 0.00684 | 0.00618 |
| 730 | 5 | `fin_log_operating_income_signed` | financial | 0.00523 | 0.00453 |
| 730 | 6 | `fin_log_revenue` | financial | 0.00508 | 0.00320 |
| 730 | 7 | `fin_operating_margin` | financial | 0.00411 | 0.00410 |
| 730 | 8 | `seq_days_since_last_T99` | sequence | 0.00409 | 0.00610 |
| 730 | 9 | `fin_equity_ratio` | financial | 0.00321 | 0.00648 |
| 730 | 10 | `seq_T06_730d` | sequence | 0.00282 | 0.00066 |
| 730 | 11 | `seq_T06_to_date` | sequence | 0.00225 | 0.00095 |
| 730 | 12 | `fin_net_margin` | financial | 0.00208 | 0.00415 |
| 730 | 13 | `seq_T99_365d` | sequence | 0.00195 | 0.00134 |
| 730 | 14 | `seq_distinct_taxonomies_365d` | sequence | 0.00175 | 0.00113 |
| 730 | 15 | `seq_T99_to_date` | sequence | 0.00160 | 0.00556 |

## Top Features by roc_auc

| Horizon | Rank | Feature | Group | Importance mean | Std |
|---:|---:|---|---|---:|---:|
| 365 | 1 | `fin_operating_roa` | financial | 0.01818 | 0.00534 |
| 365 | 2 | `fin_leverage` | financial | 0.01348 | 0.00426 |
| 365 | 3 | `fin_equity_ratio` | financial | 0.01199 | 0.00630 |
| 365 | 4 | `fin_log_operating_income_signed` | financial | 0.01196 | 0.00650 |
| 365 | 5 | `seq_days_since_last_event` | sequence | 0.01003 | 0.00301 |
| 365 | 6 | `seq_days_since_last_T99` | sequence | 0.00927 | 0.00294 |
| 365 | 7 | `fin_log_total_assets` | financial | 0.00891 | 0.00232 |
| 365 | 8 | `fin_operating_margin` | financial | 0.00852 | 0.00399 |
| 365 | 9 | `fin_log_revenue` | financial | 0.00823 | 0.00277 |
| 365 | 10 | `seq_distinct_taxonomies_730d` | sequence | 0.00740 | 0.00095 |
| 365 | 11 | `fin_net_margin` | financial | 0.00711 | 0.00431 |
| 365 | 12 | `fin_roa` | financial | 0.00578 | 0.00390 |
| 365 | 13 | `fin_loss_flag` | financial | 0.00578 | 0.00218 |
| 365 | 14 | `seq_T99_730d` | sequence | 0.00568 | 0.00177 |
| 365 | 15 | `seq_days_since_last_T06` | sequence | 0.00542 | 0.00214 |
| 730 | 1 | `fin_equity_ratio` | financial | 0.01097 | 0.00121 |
| 730 | 2 | `seq_days_since_last_T99` | sequence | 0.00944 | 0.00133 |
| 730 | 3 | `seq_days_since_last_event` | sequence | 0.00720 | 0.00359 |
| 730 | 4 | `fin_log_total_assets` | financial | 0.00612 | 0.00453 |
| 730 | 5 | `seq_events_to_date` | sequence | 0.00574 | 0.00223 |
| 730 | 6 | `fin_roa` | financial | 0.00558 | 0.00067 |
| 730 | 7 | `fin_log_revenue` | financial | 0.00512 | 0.00174 |
| 730 | 8 | `seq_days_since_last_T06` | sequence | 0.00399 | 0.00199 |
| 730 | 9 | `seq_events_730d` | sequence | 0.00376 | 0.00165 |
| 730 | 10 | `seq_T99_to_date` | sequence | 0.00369 | 0.00164 |
| 730 | 11 | `fin_net_margin` | financial | 0.00353 | 0.00183 |
| 730 | 12 | `seq_distinct_taxonomies_730d` | sequence | 0.00321 | 0.00090 |
| 730 | 13 | `fin_leverage` | financial | 0.00317 | 0.00113 |
| 730 | 14 | `seq_T99_365d` | sequence | 0.00269 | 0.00055 |
| 730 | 15 | `fin_operating_margin` | financial | 0.00259 | 0.00264 |
