# Regulatory Disclosure Sequences and Corporate Distress Risk (Korean DART Filings)

Open replication code, derived data and the public web page for the article accepted at *Asia-Pacific Financial Markets* (DOI [10.1007/s10690-026-09651-w](https://doi.org/10.1007/s10690-026-09651-w), manuscript FEJM-D-26-00495R1).

Live page: https://dart.yule.pics/

This repository is copyleft. Anyone may use, study, change and share it, and anything built from it must stay open under the same terms.

Companion materials for the article accepted at *Asia-Pacific Financial Markets* (manuscript FEJM-D-26-00495R1).

## What is included

| Folder | Content |
|---|---|
| `src/` | Code for data preparation, feature construction, model estimation, evaluation, figure generation and audit checks. `src/run_v1_pipeline.py` is the pipeline runner. |
| `data/model/` | Firm-year panel (19,924 observations, 2,798 firms) and outcome table, non-market specification. |
| `data/model_robust/` | Panel for the robustness specification. |
| `data/model_market/`, `data/model_market_robust/` | Panels with trailing price, volume, volatility, drawdown and market-adjusted return features. |
| `results/` | Metrics, bootstrap intervals, calibration, coefficients, feature and permutation importance, predictions, robustness tables, case sequences. |
| `figures/` | Figure files used in the study. |
| `web/` | Source of the front page at dart.yule.pics: HTML, CSS and ES modules that read the result files, plus `paper4_cases.py`, which exports the case sequences and their score ranks from `results/`. |
| `REPRODUCIBILITY.md` | Pipeline description from the development workspace. Paths in it refer to the development layout. |

## What is not included

- Raw DART filings and the processed event table. They are public and can be re-collected from the DART open API with the collectors in `src/collect/` and a personal API key.
- Raw market-price series. Only derived features are included in the market panels.
- SEC EDGAR and EDINET collections, which are outside the paper's DART panel.

## How the numbers map to the paper

The panel summaries (`*_summary.json`) record row counts, firm counts, positive rates and construction notes. Model metrics are in `results/dart_model_metrics.*` and `results/market/dart_model_metrics.*`.

## Notes

- The financial-statement fields in the panels are point-in-time values with an assumed availability lag (annual report date + 120 days, interim + 60 days).
- Some collectors in `src/collect/` send an HTTP User-Agent containing the author's contact address, as the SEC requires.
- Licence: code (including `web/`) under the GNU Affero General Public License v3.0 (`LICENSE`); derived data, tables and figures under Creative Commons Attribution-ShareAlike 4.0 (`LICENSE-DATA`). Both are copyleft: modified versions, including ones served over a network, must be released under the same licence.
- Financial fields are derived from DART filings collected by the author (point-in-time quarterly fundamentals); market features are derived from daily price series and the KOSPI/KOSDAQ composite indices. Raw price series are not redistributed.

## Citation

Kim, Y. (forthcoming). Regulatory Disclosure Sequences and Corporate Distress Risk: Evidence from Korean DART Filings. *Asia-Pacific Financial Markets*.
