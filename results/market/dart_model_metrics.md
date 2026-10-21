# DART Early-Warning Baseline Results

Split: train anchor years <= 2021; test anchor years >= 2022. Rows whose full outcome horizon is not observable are excluded per horizon.

| Horizon | Model | Train rows | Test rows | Test positives | ROC AUC | Avg precision | Brier | Top 10% precision | Top 10% recall | Top 10% lift |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 365 | sequence_logit | 12312 | 7612 | 55 | 0.7761 | 0.0267 | 0.1285 | 0.0315 | 0.4364 | 4.36 |
| 365 | financial_logit | 12312 | 7612 | 55 | 0.7285 | 0.0185 | 0.1807 | 0.0249 | 0.3455 | 3.45 |
| 365 | combined_logit | 12312 | 7612 | 55 | 0.776 | 0.0313 | 0.1278 | 0.0302 | 0.4182 | 4.18 |
| 365 | combined_random_forest | 12312 | 7612 | 55 | 0.8922 | 0.071 | 0.0105 | 0.0486 | 0.6727 | 6.72 |
| 365 | combined_calibrated_random_forest | 12312 | 7612 | 55 | 0.8882 | 0.0696 | 0.0071 | 0.0486 | 0.6727 | 6.72 |
| 365 | market_logit | 12312 | 7612 | 55 | 0.7116 | 0.0446 | 0.2843 | 0.021 | 0.2909 | 2.91 |
| 365 | sequence_market_logit | 12312 | 7612 | 55 | 0.7775 | 0.0545 | 0.1868 | 0.0328 | 0.4545 | 4.54 |
| 365 | financial_market_logit | 12312 | 7612 | 55 | 0.7368 | 0.039 | 0.1736 | 0.0289 | 0.4 | 4.0 |
| 365 | combined_market_logit | 12312 | 7612 | 55 | 0.8041 | 0.0455 | 0.128 | 0.0341 | 0.4727 | 4.72 |
| 365 | market_random_forest | 12312 | 7612 | 55 | 0.8549 | 0.0646 | 0.015 | 0.0499 | 0.6909 | 6.9 |
| 365 | financial_market_random_forest | 12312 | 7612 | 55 | 0.8908 | 0.0811 | 0.0131 | 0.0446 | 0.6182 | 6.18 |
| 365 | combined_market_random_forest | 12312 | 7612 | 55 | 0.903 | 0.1387 | 0.0087 | 0.0525 | 0.7273 | 7.27 |
| 365 | combined_market_calibrated_random_forest | 12312 | 7612 | 55 | 0.9049 | 0.1482 | 0.0069 | 0.0538 | 0.7455 | 7.45 |
| 730 | sequence_logit | 12312 | 4961 | 63 | 0.754 | 0.0497 | 0.1355 | 0.0584 | 0.4603 | 4.59 |
| 730 | financial_logit | 12312 | 4961 | 63 | 0.7303 | 0.0403 | 0.1903 | 0.0423 | 0.3333 | 3.33 |
| 730 | combined_logit | 12312 | 4961 | 63 | 0.7996 | 0.0595 | 0.1293 | 0.0543 | 0.4286 | 4.28 |
| 730 | combined_random_forest | 12312 | 4961 | 63 | 0.8941 | 0.1022 | 0.0176 | 0.0765 | 0.6032 | 6.02 |
| 730 | combined_calibrated_random_forest | 12312 | 4961 | 63 | 0.8997 | 0.0993 | 0.0122 | 0.0845 | 0.6667 | 6.65 |
| 730 | market_logit | 12312 | 4961 | 63 | 0.718 | 0.0498 | 0.2913 | 0.0362 | 0.2857 | 2.85 |
| 730 | sequence_market_logit | 12312 | 4961 | 63 | 0.7671 | 0.0706 | 0.1956 | 0.0684 | 0.5397 | 5.39 |
| 730 | financial_market_logit | 12312 | 4961 | 63 | 0.7892 | 0.0544 | 0.2181 | 0.0423 | 0.3333 | 3.33 |
| 730 | combined_market_logit | 12312 | 4961 | 63 | 0.8159 | 0.0653 | 0.169 | 0.0584 | 0.4603 | 4.59 |
| 730 | market_random_forest | 12312 | 4961 | 63 | 0.8156 | 0.0811 | 0.0216 | 0.0684 | 0.5397 | 5.39 |
| 730 | financial_market_random_forest | 12312 | 4961 | 63 | 0.8699 | 0.1211 | 0.0199 | 0.0724 | 0.5714 | 5.7 |
| 730 | combined_market_random_forest | 12312 | 4961 | 63 | 0.8948 | 0.1607 | 0.0145 | 0.0845 | 0.6667 | 6.65 |
| 730 | combined_market_calibrated_random_forest | 12312 | 4961 | 63 | 0.8956 | 0.1531 | 0.0116 | 0.0865 | 0.6825 | 6.81 |

## Best Current Signal

- 365 days: `combined_market_calibrated_random_forest` has average precision 0.1482; top-decile precision 0.0538, recall 0.7455, lift 7.45.
- 730 days: `combined_market_random_forest` has average precision 0.1607; top-decile precision 0.0845, recall 0.6667, lift 6.65.

## Bootstrap Confidence Intervals

Bootstrap intervals use row-level resampling of the out-of-time test set.

| Horizon | Model | Metric | Mean | 95% CI low | 95% CI high | Valid resamples |
|---:|---|---|---:|---:|---:|---:|
| 365 | sequence_logit | `roc_auc` | 0.7765 | 0.7046 | 0.8375 | 300 |
| 365 | sequence_logit | `average_precision` | 0.0303 | 0.0173 | 0.0575 | 300 |
| 365 | sequence_logit | `brier_score` | 0.1284 | 0.1245 | 0.1323 | 300 |
| 365 | sequence_logit | `top_10pct_recall` | 0.4279 | 0.2830 | 0.5525 | 300 |
| 365 | sequence_logit | `top_10pct_lift` | 4.2741 | 2.8272 | 5.5191 | 300 |
| 365 | financial_logit | `roc_auc` | 0.7287 | 0.6692 | 0.7820 | 300 |
| 365 | financial_logit | `average_precision` | 0.0199 | 0.0119 | 0.0286 | 300 |
| 365 | financial_logit | `brier_score` | 0.1807 | 0.1761 | 0.1853 | 300 |
| 365 | financial_logit | `top_10pct_recall` | 0.3443 | 0.2338 | 0.4556 | 300 |
| 365 | financial_logit | `top_10pct_lift` | 3.4392 | 2.3358 | 4.5510 | 300 |
| 365 | combined_logit | `roc_auc` | 0.7780 | 0.7223 | 0.8345 | 300 |
| 365 | combined_logit | `average_precision` | 0.0357 | 0.0177 | 0.0642 | 300 |
| 365 | combined_logit | `brier_score` | 0.1278 | 0.1233 | 0.1324 | 300 |
| 365 | combined_logit | `top_10pct_recall` | 0.4170 | 0.2856 | 0.5503 | 300 |
| 365 | combined_logit | `top_10pct_lift` | 4.1657 | 2.8529 | 5.4975 | 300 |
| 365 | combined_random_forest | `roc_auc` | 0.8922 | 0.8489 | 0.9317 | 300 |
| 365 | combined_random_forest | `average_precision` | 0.0763 | 0.0490 | 0.1128 | 300 |
| 365 | combined_random_forest | `brier_score` | 0.0106 | 0.0090 | 0.0122 | 300 |
| 365 | combined_random_forest | `top_10pct_recall` | 0.6738 | 0.5472 | 0.8000 | 300 |
| 365 | combined_random_forest | `top_10pct_lift` | 6.7306 | 5.4660 | 7.9916 | 300 |
| 365 | combined_calibrated_random_forest | `roc_auc` | 0.8892 | 0.8473 | 0.9304 | 300 |
| 365 | combined_calibrated_random_forest | `average_precision` | 0.0737 | 0.0459 | 0.1044 | 300 |
| 365 | combined_calibrated_random_forest | `brier_score` | 0.0071 | 0.0053 | 0.0088 | 300 |
| 365 | combined_calibrated_random_forest | `top_10pct_recall` | 0.6802 | 0.5446 | 0.8021 | 300 |
| 365 | combined_calibrated_random_forest | `top_10pct_lift` | 6.7953 | 5.4405 | 8.0122 | 300 |
| 365 | market_logit | `roc_auc` | 0.7091 | 0.6388 | 0.7750 | 300 |
| 365 | market_logit | `average_precision` | 0.0461 | 0.0140 | 0.0930 | 300 |
| 365 | market_logit | `brier_score` | 0.2843 | 0.2772 | 0.2921 | 300 |
| 365 | market_logit | `top_10pct_recall` | 0.2857 | 0.1739 | 0.4239 | 300 |
| 365 | market_logit | `top_10pct_lift` | 2.8536 | 1.7373 | 4.2342 | 300 |
| 365 | sequence_market_logit | `roc_auc` | 0.7778 | 0.7142 | 0.8439 | 300 |
| 365 | sequence_market_logit | `average_precision` | 0.0589 | 0.0228 | 0.1133 | 300 |
| 365 | sequence_market_logit | `brier_score` | 0.1867 | 0.1811 | 0.1926 | 300 |
| 365 | sequence_market_logit | `top_10pct_recall` | 0.4569 | 0.3274 | 0.5854 | 300 |
| 365 | sequence_market_logit | `top_10pct_lift` | 4.5639 | 3.2703 | 5.8480 | 300 |
| 365 | financial_market_logit | `roc_auc` | 0.7364 | 0.6631 | 0.8040 | 300 |
| 365 | financial_market_logit | `average_precision` | 0.0474 | 0.0205 | 0.0906 | 300 |
| 365 | financial_market_logit | `brier_score` | 0.1733 | 0.1683 | 0.1797 | 300 |
| 365 | financial_market_logit | `top_10pct_recall` | 0.4148 | 0.2857 | 0.5302 | 300 |
| 365 | financial_market_logit | `top_10pct_lift` | 4.1439 | 2.8541 | 5.2968 | 300 |
| 365 | combined_market_logit | `roc_auc` | 0.8050 | 0.7561 | 0.8564 | 300 |
| 365 | combined_market_logit | `average_precision` | 0.0534 | 0.0267 | 0.1002 | 300 |
| 365 | combined_market_logit | `brier_score` | 0.1280 | 0.1231 | 0.1330 | 300 |
| 365 | combined_market_logit | `top_10pct_recall` | 0.4838 | 0.3706 | 0.6087 | 300 |
| 365 | combined_market_logit | `top_10pct_lift` | 4.8331 | 3.7020 | 6.0806 | 300 |
| 365 | market_random_forest | `roc_auc` | 0.8546 | 0.8001 | 0.9008 | 300 |
| 365 | market_random_forest | `average_precision` | 0.0730 | 0.0376 | 0.1347 | 300 |
| 365 | market_random_forest | `brier_score` | 0.0151 | 0.0136 | 0.0167 | 300 |
| 365 | market_random_forest | `top_10pct_recall` | 0.6860 | 0.5556 | 0.7967 | 300 |
| 365 | market_random_forest | `top_10pct_lift` | 6.8525 | 5.5497 | 7.9589 | 300 |
| 365 | financial_market_random_forest | `roc_auc` | 0.8898 | 0.8417 | 0.9251 | 300 |
| 365 | financial_market_random_forest | `average_precision` | 0.0908 | 0.0419 | 0.1606 | 300 |
| 365 | financial_market_random_forest | `brier_score` | 0.0131 | 0.0118 | 0.0147 | 300 |
| 365 | financial_market_random_forest | `top_10pct_recall` | 0.6158 | 0.4891 | 0.7635 | 300 |
| 365 | financial_market_random_forest | `top_10pct_lift` | 6.1511 | 4.8860 | 7.6274 | 300 |
| 365 | combined_market_random_forest | `roc_auc` | 0.9039 | 0.8445 | 0.9460 | 300 |
| 365 | combined_market_random_forest | `average_precision` | 0.1542 | 0.0806 | 0.2517 | 300 |
| 365 | combined_market_random_forest | `brier_score` | 0.0087 | 0.0073 | 0.0104 | 300 |
| 365 | combined_market_random_forest | `top_10pct_recall` | 0.7297 | 0.6092 | 0.8560 | 300 |
| 365 | combined_market_random_forest | `top_10pct_lift` | 7.2898 | 6.0856 | 8.5510 | 300 |
| 365 | combined_market_calibrated_random_forest | `roc_auc` | 0.9033 | 0.8580 | 0.9389 | 300 |
| 365 | combined_market_calibrated_random_forest | `average_precision` | 0.1607 | 0.0887 | 0.2408 | 300 |
| 365 | combined_market_calibrated_random_forest | `brier_score` | 0.0069 | 0.0054 | 0.0084 | 300 |
| 365 | combined_market_calibrated_random_forest | `top_10pct_recall` | 0.7353 | 0.6195 | 0.8351 | 300 |
| 365 | combined_market_calibrated_random_forest | `top_10pct_lift` | 7.3450 | 6.1885 | 8.3424 | 300 |
| 730 | sequence_logit | `roc_auc` | 0.7560 | 0.6853 | 0.8183 | 300 |
| 730 | sequence_logit | `average_precision` | 0.0535 | 0.0337 | 0.0804 | 300 |
| 730 | sequence_logit | `brier_score` | 0.1356 | 0.1311 | 0.1403 | 300 |
| 730 | sequence_logit | `top_10pct_recall` | 0.4704 | 0.3623 | 0.5933 | 300 |
| 730 | sequence_logit | `top_10pct_lift` | 4.6953 | 3.6162 | 5.9223 | 300 |
| 730 | financial_logit | `roc_auc` | 0.7285 | 0.6665 | 0.7823 | 300 |
| 730 | financial_logit | `average_precision` | 0.0439 | 0.0253 | 0.0709 | 300 |
| 730 | financial_logit | `brier_score` | 0.1904 | 0.1842 | 0.1959 | 300 |
| 730 | financial_logit | `top_10pct_recall` | 0.3384 | 0.2317 | 0.4691 | 300 |
| 730 | financial_logit | `top_10pct_lift` | 3.3776 | 2.3124 | 4.6829 | 300 |
| 730 | combined_logit | `roc_auc` | 0.7966 | 0.7382 | 0.8439 | 300 |
| 730 | combined_logit | `average_precision` | 0.0638 | 0.0357 | 0.0973 | 300 |
| 730 | combined_logit | `brier_score` | 0.1296 | 0.1249 | 0.1356 | 300 |
| 730 | combined_logit | `top_10pct_recall` | 0.4279 | 0.2961 | 0.5428 | 300 |
| 730 | combined_logit | `top_10pct_lift` | 4.2712 | 2.9559 | 5.4184 | 300 |
| 730 | combined_random_forest | `roc_auc` | 0.8932 | 0.8575 | 0.9241 | 300 |
| 730 | combined_random_forest | `average_precision` | 0.1091 | 0.0698 | 0.1554 | 300 |
| 730 | combined_random_forest | `brier_score` | 0.0177 | 0.0155 | 0.0203 | 300 |
| 730 | combined_random_forest | `top_10pct_recall` | 0.6058 | 0.4957 | 0.7112 | 300 |
| 730 | combined_random_forest | `top_10pct_lift` | 6.0475 | 4.9480 | 7.0989 | 300 |
| 730 | combined_calibrated_random_forest | `roc_auc` | 0.8996 | 0.8677 | 0.9286 | 300 |
| 730 | combined_calibrated_random_forest | `average_precision` | 0.1075 | 0.0696 | 0.1533 | 300 |
| 730 | combined_calibrated_random_forest | `brier_score` | 0.0123 | 0.0095 | 0.0153 | 300 |
| 730 | combined_calibrated_random_forest | `top_10pct_recall` | 0.6671 | 0.5571 | 0.7786 | 300 |
| 730 | combined_calibrated_random_forest | `top_10pct_lift` | 6.6592 | 5.5605 | 7.7720 | 300 |
| 730 | market_logit | `roc_auc` | 0.7185 | 0.6591 | 0.7712 | 300 |
| 730 | market_logit | `average_precision` | 0.0523 | 0.0238 | 0.0955 | 300 |
| 730 | market_logit | `brier_score` | 0.2914 | 0.2847 | 0.2976 | 300 |
| 730 | market_logit | `top_10pct_recall` | 0.2845 | 0.1603 | 0.4017 | 300 |
| 730 | market_logit | `top_10pct_lift` | 2.8398 | 1.6005 | 4.0097 | 300 |
| 730 | sequence_market_logit | `roc_auc` | 0.7678 | 0.6955 | 0.8392 | 300 |
| 730 | sequence_market_logit | `average_precision` | 0.0740 | 0.0376 | 0.1278 | 300 |
| 730 | sequence_market_logit | `brier_score` | 0.1959 | 0.1899 | 0.2019 | 300 |
| 730 | sequence_market_logit | `top_10pct_recall` | 0.5302 | 0.4014 | 0.6575 | 300 |
| 730 | sequence_market_logit | `top_10pct_lift` | 5.2920 | 4.0069 | 6.5626 | 300 |
| 730 | financial_market_logit | `roc_auc` | 0.7861 | 0.7430 | 0.8242 | 300 |
| 730 | financial_market_logit | `average_precision` | 0.0593 | 0.0335 | 0.0916 | 300 |
| 730 | financial_market_logit | `brier_score` | 0.2182 | 0.2117 | 0.2241 | 300 |
| 730 | financial_market_logit | `top_10pct_recall` | 0.3297 | 0.2188 | 0.4331 | 300 |
| 730 | financial_market_logit | `top_10pct_lift` | 3.2910 | 2.1845 | 4.3231 | 300 |
| 730 | combined_market_logit | `roc_auc` | 0.8150 | 0.7676 | 0.8599 | 300 |
| 730 | combined_market_logit | `average_precision` | 0.0727 | 0.0422 | 0.1160 | 300 |
| 730 | combined_market_logit | `brier_score` | 0.1690 | 0.1624 | 0.1748 | 300 |
| 730 | combined_market_logit | `top_10pct_recall` | 0.4584 | 0.3412 | 0.5751 | 300 |
| 730 | combined_market_logit | `top_10pct_lift` | 4.5762 | 3.4063 | 5.7404 | 300 |
| 730 | market_random_forest | `roc_auc` | 0.8182 | 0.7658 | 0.8712 | 300 |
| 730 | market_random_forest | `average_precision` | 0.0896 | 0.0547 | 0.1377 | 300 |
| 730 | market_random_forest | `brier_score` | 0.0216 | 0.0191 | 0.0240 | 300 |
| 730 | market_random_forest | `top_10pct_recall` | 0.5526 | 0.4230 | 0.6858 | 300 |
| 730 | market_random_forest | `top_10pct_lift` | 5.5159 | 4.2223 | 6.8452 | 300 |
| 730 | financial_market_random_forest | `roc_auc` | 0.8685 | 0.8304 | 0.9023 | 300 |
| 730 | financial_market_random_forest | `average_precision` | 0.1300 | 0.0715 | 0.2197 | 300 |
| 730 | financial_market_random_forest | `brier_score` | 0.0199 | 0.0176 | 0.0224 | 300 |
| 730 | financial_market_random_forest | `top_10pct_recall` | 0.5628 | 0.4408 | 0.6790 | 300 |
| 730 | financial_market_random_forest | `top_10pct_lift` | 5.6177 | 4.4000 | 6.7780 | 300 |
| 730 | combined_market_random_forest | `roc_auc` | 0.8951 | 0.8479 | 0.9333 | 300 |
| 730 | combined_market_random_forest | `average_precision` | 0.1747 | 0.1097 | 0.2533 | 300 |
| 730 | combined_market_random_forest | `brier_score` | 0.0145 | 0.0123 | 0.0166 | 300 |
| 730 | combined_market_random_forest | `top_10pct_recall` | 0.6679 | 0.5461 | 0.7759 | 300 |
| 730 | combined_market_random_forest | `top_10pct_lift` | 6.6668 | 5.4514 | 7.7445 | 300 |
| 730 | combined_market_calibrated_random_forest | `roc_auc` | 0.8983 | 0.8566 | 0.9304 | 300 |
| 730 | combined_market_calibrated_random_forest | `average_precision` | 0.1651 | 0.1010 | 0.2447 | 300 |
| 730 | combined_market_calibrated_random_forest | `brier_score` | 0.0117 | 0.0093 | 0.0145 | 300 |
| 730 | combined_market_calibrated_random_forest | `top_10pct_recall` | 0.6788 | 0.5660 | 0.7836 | 300 |
| 730 | combined_market_calibrated_random_forest | `top_10pct_lift` | 6.7756 | 5.6501 | 7.8217 | 300 |

## Calibration Snapshot

The table below shows the highest-risk decile only; full decile tables are saved in `dart_model_calibration.csv`.

| Horizon | Model | Top bin count | Mean predicted risk | Observed rate | Positives |
|---:|---|---:|---:|---:|---:|
| 365 | combined_calibrated_random_forest | 761 | 0.0193 | 0.0486 | 37 |
| 365 | combined_logit | 761 | 0.8061 | 0.0302 | 23 |
| 365 | combined_market_calibrated_random_forest | 761 | 0.0697 | 0.0539 | 41 |
| 365 | combined_market_logit | 761 | 0.8642 | 0.0342 | 26 |
| 365 | combined_market_random_forest | 761 | 0.1715 | 0.0526 | 40 |
| 365 | combined_random_forest | 761 | 0.1963 | 0.0486 | 37 |
| 365 | financial_logit | 761 | 0.8003 | 0.0250 | 19 |
| 365 | financial_market_logit | 761 | 0.8890 | 0.0289 | 22 |
| 365 | financial_market_random_forest | 761 | 0.2563 | 0.0447 | 34 |
| 365 | market_logit | 761 | 0.9479 | 0.0210 | 16 |
| 365 | market_random_forest | 761 | 0.2857 | 0.0499 | 38 |
| 365 | sequence_logit | 761 | 0.7461 | 0.0315 | 24 |
| 365 | sequence_market_logit | 761 | 0.9152 | 0.0329 | 25 |
| 730 | combined_calibrated_random_forest | 496 | 0.0407 | 0.0847 | 42 |
| 730 | combined_logit | 496 | 0.7790 | 0.0544 | 27 |
| 730 | combined_market_calibrated_random_forest | 496 | 0.0830 | 0.0867 | 43 |
| 730 | combined_market_logit | 496 | 0.8716 | 0.0585 | 29 |
| 730 | combined_market_random_forest | 496 | 0.2297 | 0.0847 | 42 |
| 730 | combined_random_forest | 496 | 0.2706 | 0.0766 | 38 |
| 730 | financial_logit | 496 | 0.7915 | 0.0423 | 21 |
| 730 | financial_market_logit | 496 | 0.8875 | 0.0423 | 21 |
| 730 | financial_market_random_forest | 496 | 0.3115 | 0.0726 | 36 |
| 730 | market_logit | 496 | 0.9005 | 0.0363 | 18 |
| 730 | market_random_forest | 496 | 0.3256 | 0.0685 | 34 |
| 730 | sequence_logit | 496 | 0.7508 | 0.0585 | 29 |
| 730 | sequence_market_logit | 496 | 0.8742 | 0.0685 | 34 |

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

