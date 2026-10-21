# DART Robustness Summary

Robust outcomes exclude weak `T02` listing-status outcomes and SPAC-like dissolution cases. The main outcomes keep all DART-derived distress proxies.

## Dataset Comparison

| Scenario | Rows | Firms | Outcome rows | Outcome firms | 365d positives | 730d positives | Financial match |
|---|---:|---:|---:|---:|---:|---:|---:|
| main | 19924 | 2798 | 386 | 274 | 170 | 283 | 0.822 |
| robust | 20373 | 2853 | 254 | 161 | 126 | 227 | 0.812 |

Robust filter exclusions:

- Excluded rows: 132 across 113 firms.
- Weak T02 listing-status rows: 40 rows, 21 firms.
- SPAC-like dissolution rows: 92 rows, 92 firms.

## Model Comparison

| Scenario | Horizon | Model | Test positives | ROC AUC | Avg precision | Brier | Top 10% recall | Top 10% lift |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| main | 365 | `combined_calibrated_random_forest` | 55 | 0.8882 | 0.0696 | 0.0071 | 0.6727 | 6.72 |
| main | 365 | `combined_random_forest` | 55 | 0.8922 | 0.0710 | 0.0105 | 0.6727 | 6.72 |
| main | 365 | `financial_logit` | 55 | 0.7285 | 0.0185 | 0.1807 | 0.3455 | 3.45 |
| main | 365 | `sequence_logit` | 55 | 0.7761 | 0.0267 | 0.1285 | 0.4364 | 4.36 |
| main | 730 | `combined_calibrated_random_forest` | 63 | 0.8997 | 0.0993 | 0.0122 | 0.6667 | 6.65 |
| main | 730 | `combined_random_forest` | 63 | 0.8941 | 0.1022 | 0.0176 | 0.6032 | 6.02 |
| main | 730 | `financial_logit` | 63 | 0.7303 | 0.0403 | 0.1903 | 0.3333 | 3.33 |
| main | 730 | `sequence_logit` | 63 | 0.7540 | 0.0497 | 0.1355 | 0.4603 | 4.59 |
| robust | 365 | `combined_calibrated_random_forest` | 34 | 0.8682 | 0.0463 | 0.0043 | 0.6176 | 6.17 |
| robust | 365 | `combined_random_forest` | 34 | 0.8645 | 0.0445 | 0.0060 | 0.7059 | 7.06 |
| robust | 365 | `financial_logit` | 34 | 0.8154 | 0.0216 | 0.1608 | 0.5588 | 5.59 |
| robust | 365 | `sequence_logit` | 34 | 0.7863 | 0.0180 | 0.1200 | 0.4706 | 4.70 |
| robust | 730 | `combined_calibrated_random_forest` | 42 | 0.8946 | 0.0771 | 0.0080 | 0.6667 | 6.66 |
| robust | 730 | `combined_random_forest` | 42 | 0.8895 | 0.0802 | 0.0117 | 0.6905 | 6.90 |
| robust | 730 | `financial_logit` | 42 | 0.8088 | 0.0387 | 0.1670 | 0.4286 | 4.28 |
| robust | 730 | `sequence_logit` | 42 | 0.7871 | 0.0475 | 0.1244 | 0.5238 | 5.24 |

## Interpretation

- After excluding weak outcomes, the 730-day combined random forest still reaches ROC AUC 0.8895 and top-decile lift 6.90.
- The robust calibrated random forest has lower Brier score (0.0080) and ROC AUC 0.8946, but the raw random forest has slightly higher average precision.
- This supports reporting the main model as a watchlist-ranking model, with calibrated RF as the probability-oriented sensitivity check.
