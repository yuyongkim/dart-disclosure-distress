# SupTech Rule Engine

This file documents the explicit supervisory flagging rules requested in `paper4_research_plan.md`. These rules are separate from the model-based watchlist and are best interpreted as auditable trigger definitions.

## Rule Registry

| Rule | Severity | Action | Pattern | Window days | Includes outcome events | Description |
|---|---|---|---|---:|---|---|
| `CRITICAL_T01_IMMEDIATE` | CRITICAL | IMMEDIATE_REVIEW | `T01` | 1 | True | Any insolvency, bankruptcy, rehabilitation, or receivership event. |
| `CRITICAL_T02_IMMEDIATE` | CRITICAL | IMMEDIATE_REVIEW | `T02` | 1 | True | Any delisting or trading-status event. |
| `HIGH_T04_THEN_T03_90D` | HIGH | PRIORITY_REVIEW | `T04>T03` | 90 | False | Going-concern/audit concern followed by debt default or acceleration within 90 days. |
| `HIGH_T05_T06_30D` | HIGH | PRIORITY_REVIEW | `T05>T06` | 30 | False | Capital-structure stress and governance/control change occurring within 30 days. |
| `MEDIUM_T05_THEN_T04_365D` | MEDIUM | WATCHLIST | `T05>T04` | 365 | False | Capital-structure stress followed by going-concern/audit concern within 365 days. |
| `MEDIUM_REPEATED_T05_365D` | MEDIUM | WATCHLIST | `T05>T05` | 365 | False | At least two capital-structure-stress events within 365 days. |

## Rule Output Summary

| Source | Rule | Flagged firms | Early TP | Same-day/reactive | Early precision | Early recall | Median early lead days |
|---|---|---:|---:|---:|---:|---:|---:|
| DART | `CRITICAL_T01_IMMEDIATE` | 207 | 0 | 207 | 0.0000 | 0.0000 |  |
| DART | `CRITICAL_T02_IMMEDIATE` | 21 | 0 | 21 | 0.0000 | 0.0000 |  |
| DART | `HIGH_T04_THEN_T03_90D` | 0 | 0 | 0 | 0.0000 | 0.0000 |  |
| DART | `HIGH_T05_T06_30D` | 198 | 21 | 0 | 0.1061 | 0.0894 | 726.0 |
| DART | `MEDIUM_REPEATED_T05_365D` | 1105 | 98 | 0 | 0.0887 | 0.4170 | 1229.5 |
| DART | `MEDIUM_T05_THEN_T04_365D` | 0 | 0 | 0 | 0.0000 | 0.0000 |  |
| SEC | `CRITICAL_T01_IMMEDIATE` | 0 | 0 | 0 | 0.0000 | 0.0000 |  |
| SEC | `CRITICAL_T02_IMMEDIATE` | 20 | 0 | 20 | 0.0000 | 0.0000 |  |
| SEC | `HIGH_T04_THEN_T03_90D` | 0 | 0 | 0 | 0.0000 | 0.0000 |  |
| SEC | `HIGH_T05_T06_30D` | 65 | 15 | 0 | 0.2308 | 0.6522 | 970.0 |
| SEC | `MEDIUM_REPEATED_T05_365D` | 74 | 17 | 0 | 0.2297 | 0.7391 | 807.0 |
| SEC | `MEDIUM_T05_THEN_T04_365D` | 0 | 0 | 0 | 0.0000 | 0.0000 |  |

## Interpretation

- `CRITICAL` rules are immediate-review triggers and may fire on the same date as an observed formal distress event.
- `HIGH` and `MEDIUM` rules exclude outcome-like event subtypes when measuring early-warning behavior.
- These rules remain interpretable but noisy. The main v1 paper should keep the ranked model watchlist as the headline SupTech artifact.
