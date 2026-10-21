# Classic Financial Baseline Audit and Proxy Results

Exact Altman Z-score and Ohlson O-score cannot be computed from the currently available point-in-time financial panel because several required components are missing. This file reports that component audit and evaluates transparent available-component proxy scores as ranking baselines.

## Component Audit

### altman_z_score_original

- Status: `not_computable_exactly`
- Proxy used: `altman_partial_available_components`
- Available: EBIT-like operating income / total assets, book equity / total liabilities, sales-like revenue / total assets.
- Missing: working capital / total assets, retained earnings / total assets, market value of equity / total liabilities.

### ohlson_o_score_original

- Status: `not_computable_exactly`
- Proxy used: `ohlson_partial_available_components`
- Available: log total assets, total liabilities / total assets, net income / total assets, negative equity indicator, two-year loss indicator, net-income change ratio.
- Missing: working capital / total assets, current liabilities / current assets, funds from operations / total liabilities, GNP price-level adjustment.

## Out-of-Time Proxy Baseline Results

Scores are evaluated on anchor years >= 2022. Higher score means higher predicted distress risk.

| Horizon | Method | Test rows | Positives | Coverage | ROC AUC | Avg precision | Top 10% precision | Top 10% recall | Top 10% lift | Median lead days |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 365 | `altman_partial_risk_score` | 6688 | 39 | 0.879 | 0.7282 | 0.0259 | 0.0224 | 0.3846 | 3.85 | 120.0 |
| 365 | `ohlson_partial_risk_score` | 6282 | 39 | 0.825 | 0.6796 | 0.0170 | 0.0175 | 0.2821 | 2.82 | 127.0 |
| 365 | `loss_leverage_risk_score` | 6751 | 48 | 0.887 | 0.5984 | 0.0169 | 0.0207 | 0.2917 | 2.91 | 123.5 |
| 365 | `leverage_risk_score` | 6751 | 48 | 0.887 | 0.5653 | 0.0121 | 0.0178 | 0.2500 | 2.50 | 123.5 |
| 730 | `altman_partial_risk_score` | 4324 | 49 | 0.872 | 0.7302 | 0.0379 | 0.0439 | 0.3878 | 3.87 | 285.0 |
| 730 | `ohlson_partial_risk_score` | 4036 | 44 | 0.814 | 0.7148 | 0.0330 | 0.0347 | 0.3182 | 3.18 | 470.0 |
| 730 | `loss_leverage_risk_score` | 4348 | 56 | 0.876 | 0.6343 | 0.0276 | 0.0345 | 0.2679 | 2.68 | 285.0 |
| 730 | `leverage_risk_score` | 4348 | 56 | 0.876 | 0.5843 | 0.0202 | 0.0276 | 0.2143 | 2.14 | 281.5 |

## Interpretation

These proxy scores are useful as transparent financial-only comparators, but they should not be described as exact Altman or exact Ohlson scores. The main manuscript should continue to compare against the financial-only logistic model as the stronger financial baseline, while citing these proxy results as an additional robustness check.
