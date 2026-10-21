# SupTech Flagging Performance

Rule-based flags use only pre-outcome non-outcome event sequences. Model watchlists are top-decile out-of-time firm-year predictions from the DART model dataset.

## Best DART Watchlist Rows

| Type | Horizon | Method | Flagged | TP | FP | Precision | Recall | F1 | Median lead days |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| rule | 180 | `R2_T05_T06_180D` | 391 | 4 | 387 | 0.0102 | 0.0170 | 0.0128 | 26.0 |
| rule | 180 | `R1_REPEATED_T05_365D` | 1105 | 3 | 1102 | 0.0027 | 0.0128 | 0.0045 | 65.0 |
| rule | 180 | `R3_PRIOR_T03` | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |  |
| rule | 180 | `R4_T06_THEN_T05_365D` | 382 | 0 | 382 | 0.0000 | 0.0000 | 0.0000 |  |
| model_top10pct | 365 | `combined_calibrated_random_forest` | 761 | 37 | 724 | 0.0486 | 0.6727 | 0.0907 | 135.0 |
| model_top10pct | 365 | `combined_random_forest` | 761 | 37 | 724 | 0.0486 | 0.6727 | 0.0907 | 135.0 |
| model_top10pct | 365 | `sequence_logit` | 761 | 24 | 737 | 0.0315 | 0.4364 | 0.0588 | 128.0 |
| model_top10pct | 365 | `combined_logit` | 761 | 23 | 738 | 0.0302 | 0.4182 | 0.0564 | 114.0 |
| model_top10pct | 365 | `financial_logit` | 761 | 19 | 742 | 0.0250 | 0.3455 | 0.0466 | 114.0 |
| rule | 365 | `R2_T05_T06_180D` | 391 | 9 | 382 | 0.0230 | 0.0383 | 0.0288 | 248.0 |
| rule | 365 | `R1_REPEATED_T05_365D` | 1105 | 7 | 1098 | 0.0063 | 0.0298 | 0.0104 | 245.0 |
| rule | 365 | `R4_T06_THEN_T05_365D` | 382 | 4 | 378 | 0.0105 | 0.0170 | 0.0130 | 305.5 |
| rule | 365 | `R3_PRIOR_T03` | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |  |
| model_top10pct | 730 | `combined_calibrated_random_forest` | 496 | 42 | 454 | 0.0847 | 0.6667 | 0.1503 | 415.0 |
| model_top10pct | 730 | `combined_random_forest` | 496 | 38 | 458 | 0.0766 | 0.6032 | 0.1360 | 332.0 |
| model_top10pct | 730 | `sequence_logit` | 496 | 29 | 467 | 0.0585 | 0.4603 | 0.1038 | 431.0 |
| model_top10pct | 730 | `combined_logit` | 496 | 27 | 469 | 0.0544 | 0.4286 | 0.0966 | 318.0 |
| model_top10pct | 730 | `financial_logit` | 496 | 21 | 475 | 0.0423 | 0.3333 | 0.0751 | 445.0 |
| rule | 730 | `R1_REPEATED_T05_365D` | 1105 | 27 | 1078 | 0.0244 | 0.1149 | 0.0403 | 500.0 |
| rule | 730 | `R2_T05_T06_180D` | 391 | 20 | 371 | 0.0512 | 0.0851 | 0.0639 | 423.0 |
| rule | 730 | `R4_T06_THEN_T05_365D` | 382 | 13 | 369 | 0.0340 | 0.0553 | 0.0421 | 412.0 |
| rule | 730 | `R3_PRIOR_T03` | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |  |
| rule | any_positive_lead | `R1_REPEATED_T05_365D` | 1105 | 98 | 1007 | 0.0887 | 0.4170 | 0.1463 | 1229.5 |
| rule | any_positive_lead | `R2_T05_T06_180D` | 391 | 37 | 354 | 0.0946 | 0.1574 | 0.1182 | 717.0 |
| rule | any_positive_lead | `R4_T06_THEN_T05_365D` | 382 | 30 | 352 | 0.0785 | 0.1277 | 0.0972 | 1032.0 |
| rule | any_positive_lead | `R3_PRIOR_T03` | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |  |

## Full Rule and Model Table

| Source | Type | Horizon | Method | Severity | Flagged | TP | FP | FN | Precision | Recall | F1 | Median lead days |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DART | model_top10pct | 365 | `combined_calibrated_random_forest` | WATCHLIST | 761 | 37 | 724 | 18 | 0.0486 | 0.6727 | 0.0907 | 135.0 |
| DART | model_top10pct | 365 | `combined_logit` | WATCHLIST | 761 | 23 | 738 | 32 | 0.0302 | 0.4182 | 0.0564 | 114.0 |
| DART | model_top10pct | 365 | `combined_random_forest` | WATCHLIST | 761 | 37 | 724 | 18 | 0.0486 | 0.6727 | 0.0907 | 135.0 |
| DART | model_top10pct | 365 | `financial_logit` | WATCHLIST | 761 | 19 | 742 | 36 | 0.0250 | 0.3455 | 0.0466 | 114.0 |
| DART | model_top10pct | 365 | `sequence_logit` | WATCHLIST | 761 | 24 | 737 | 31 | 0.0315 | 0.4364 | 0.0588 | 128.0 |
| DART | model_top10pct | 730 | `combined_calibrated_random_forest` | WATCHLIST | 496 | 42 | 454 | 21 | 0.0847 | 0.6667 | 0.1503 | 415.0 |
| DART | model_top10pct | 730 | `combined_logit` | WATCHLIST | 496 | 27 | 469 | 36 | 0.0544 | 0.4286 | 0.0966 | 318.0 |
| DART | model_top10pct | 730 | `combined_random_forest` | WATCHLIST | 496 | 38 | 458 | 25 | 0.0766 | 0.6032 | 0.1360 | 332.0 |
| DART | model_top10pct | 730 | `financial_logit` | WATCHLIST | 496 | 21 | 475 | 42 | 0.0423 | 0.3333 | 0.0751 | 445.0 |
| DART | model_top10pct | 730 | `sequence_logit` | WATCHLIST | 496 | 29 | 467 | 34 | 0.0585 | 0.4603 | 0.1038 | 431.0 |
| DART | rule | 180 | `R1_REPEATED_T05_365D` | MEDIUM | 1105 | 3 | 1102 | 232 | 0.0027 | 0.0128 | 0.0045 | 65.0 |
| DART | rule | 180 | `R2_T05_T06_180D` | HIGH | 391 | 4 | 387 | 231 | 0.0102 | 0.0170 | 0.0128 | 26.0 |
| DART | rule | 180 | `R3_PRIOR_T03` | HIGH | 0 | 0 | 0 | 235 | 0.0000 | 0.0000 | 0.0000 |  |
| DART | rule | 180 | `R4_T06_THEN_T05_365D` | MEDIUM | 382 | 0 | 382 | 235 | 0.0000 | 0.0000 | 0.0000 |  |
| DART | rule | 365 | `R1_REPEATED_T05_365D` | MEDIUM | 1105 | 7 | 1098 | 228 | 0.0063 | 0.0298 | 0.0104 | 245.0 |
| DART | rule | 365 | `R2_T05_T06_180D` | HIGH | 391 | 9 | 382 | 226 | 0.0230 | 0.0383 | 0.0288 | 248.0 |
| DART | rule | 365 | `R3_PRIOR_T03` | HIGH | 0 | 0 | 0 | 235 | 0.0000 | 0.0000 | 0.0000 |  |
| DART | rule | 365 | `R4_T06_THEN_T05_365D` | MEDIUM | 382 | 4 | 378 | 231 | 0.0105 | 0.0170 | 0.0130 | 305.5 |
| DART | rule | 730 | `R1_REPEATED_T05_365D` | MEDIUM | 1105 | 27 | 1078 | 208 | 0.0244 | 0.1149 | 0.0403 | 500.0 |
| DART | rule | 730 | `R2_T05_T06_180D` | HIGH | 391 | 20 | 371 | 215 | 0.0512 | 0.0851 | 0.0639 | 423.0 |
| DART | rule | 730 | `R3_PRIOR_T03` | HIGH | 0 | 0 | 0 | 235 | 0.0000 | 0.0000 | 0.0000 |  |
| DART | rule | 730 | `R4_T06_THEN_T05_365D` | MEDIUM | 382 | 13 | 369 | 222 | 0.0340 | 0.0553 | 0.0421 | 412.0 |
| DART | rule | any_positive_lead | `R1_REPEATED_T05_365D` | MEDIUM | 1105 | 98 | 1007 | 137 | 0.0887 | 0.4170 | 0.1463 | 1229.5 |
| DART | rule | any_positive_lead | `R2_T05_T06_180D` | HIGH | 391 | 37 | 354 | 198 | 0.0946 | 0.1574 | 0.1182 | 717.0 |
| DART | rule | any_positive_lead | `R3_PRIOR_T03` | HIGH | 0 | 0 | 0 | 235 | 0.0000 | 0.0000 | 0.0000 |  |
| DART | rule | any_positive_lead | `R4_T06_THEN_T05_365D` | MEDIUM | 382 | 30 | 352 | 205 | 0.0785 | 0.1277 | 0.0972 | 1032.0 |
| SEC | rule | 180 | `R1_REPEATED_T05_365D` | MEDIUM | 74 | 2 | 72 | 21 | 0.0270 | 0.0870 | 0.0412 | 144.0 |
| SEC | rule | 180 | `R2_T05_T06_180D` | HIGH | 81 | 3 | 78 | 20 | 0.0370 | 0.1304 | 0.0577 | 169.0 |
| SEC | rule | 180 | `R3_PRIOR_T03` | HIGH | 63 | 1 | 62 | 22 | 0.0159 | 0.0435 | 0.0233 | 115.0 |
| SEC | rule | 180 | `R4_T06_THEN_T05_365D` | MEDIUM | 78 | 2 | 76 | 21 | 0.0256 | 0.0870 | 0.0396 | 159.5 |
| SEC | rule | 365 | `R1_REPEATED_T05_365D` | MEDIUM | 74 | 3 | 71 | 20 | 0.0405 | 0.1304 | 0.0619 | 173.0 |
| SEC | rule | 365 | `R2_T05_T06_180D` | HIGH | 81 | 3 | 78 | 20 | 0.0370 | 0.1304 | 0.0577 | 169.0 |
| SEC | rule | 365 | `R3_PRIOR_T03` | HIGH | 63 | 5 | 58 | 18 | 0.0794 | 0.2174 | 0.1163 | 280.0 |
| SEC | rule | 365 | `R4_T06_THEN_T05_365D` | MEDIUM | 78 | 2 | 76 | 21 | 0.0256 | 0.0870 | 0.0396 | 159.5 |
| SEC | rule | 730 | `R1_REPEATED_T05_365D` | MEDIUM | 74 | 8 | 66 | 15 | 0.1081 | 0.3478 | 0.1649 | 488.5 |
| SEC | rule | 730 | `R2_T05_T06_180D` | HIGH | 81 | 6 | 75 | 17 | 0.0741 | 0.2609 | 0.1154 | 331.0 |
| SEC | rule | 730 | `R3_PRIOR_T03` | HIGH | 63 | 5 | 58 | 18 | 0.0794 | 0.2174 | 0.1163 | 280.0 |
| SEC | rule | 730 | `R4_T06_THEN_T05_365D` | MEDIUM | 78 | 5 | 73 | 18 | 0.0641 | 0.2174 | 0.0990 | 491.0 |
| SEC | rule | any_positive_lead | `R1_REPEATED_T05_365D` | MEDIUM | 74 | 17 | 57 | 6 | 0.2297 | 0.7391 | 0.3505 | 807.0 |
| SEC | rule | any_positive_lead | `R2_T05_T06_180D` | HIGH | 81 | 18 | 63 | 5 | 0.2222 | 0.7826 | 0.3462 | 922.5 |
| SEC | rule | any_positive_lead | `R3_PRIOR_T03` | HIGH | 63 | 14 | 49 | 9 | 0.2222 | 0.6087 | 0.3256 | 870.0 |
| SEC | rule | any_positive_lead | `R4_T06_THEN_T05_365D` | MEDIUM | 78 | 18 | 60 | 5 | 0.2308 | 0.7826 | 0.3564 | 1022.0 |

## Interpretation

Broad hand-written sequence rules provide long lead times but low precision in DART. The combined model watchlist is the stronger v1 SupTech artifact because it converts the same sequence layer plus financial fundamentals into a ranked review queue.
