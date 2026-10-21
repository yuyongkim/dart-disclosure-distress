# DART Distress Case Sequences

These examples use the `combined_random_forest` test-set predictions. `model_visible` events occurred on or before the prediction anchor date; `post_anchor_to_outcome` events are shown only to narrate the subsequent path to the outcome.

| Firm | Security | Horizon | Anchor | Outcome | Days to outcome | Predicted risk |
|---|---|---:|---|---|---:|---:|
| 디에이테크놀로지 | 196490 | 730 | 2023-12-31 | BANKRUPTCY_OR_REHABILITATION on 2024-04-23 | 114 | 0.5253 |
| 세븐브로이맥주 | 267080 | 730 | 2023-12-31 | BANKRUPTCY_OR_REHABILITATION on 2025-05-28 | 514 | 0.4713 |
| 코스나인 | 082660 | 730 | 2022-12-31 | BANKRUPTCY_OR_REHABILITATION on 2024-10-28 | 667 | 0.4322 |
| 엔케이젠바이오텍코리아 | 182400 | 730 | 2023-12-31 | BANKRUPTCY_OR_REHABILITATION on 2024-04-18 | 109 | 0.4220 |
| 비유테크놀러지 | 230980 | 365 | 2024-12-31 | BANKRUPTCY_OR_REHABILITATION on 2025-04-10 | 100 | 0.4198 |
| 한국비티비 | 219750 | 730 | 2022-12-31 | BANKRUPTCY_OR_REHABILITATION on 2023-03-29 | 88 | 0.3776 |
| 무송지오씨 | 135160 | 365 | 2022-12-31 | DEFAULT on 2023-04-18 | 108 | 0.0752 |
| 태영건설 | 009410 | 730 | 2023-12-31 | DEFAULT on 2024-02-27 | 58 | 0.0423 |

## 디에이테크놀로지 (196490)

- Anchor: 2023-12-31, horizon 730 days, predicted risk 0.5253.
- Outcome: BANKRUPTCY_OR_REHABILITATION, 2024-04-23, 114 days after anchor.

Model-visible sequence:

- 2021-11-29: `T05` / `convertible_bond` / 전환사채권발행결정 (876 days before outcome).
- 2022-01-10: `T05` / `paid_in_capital_increase` / 유상증자결정 (834 days before outcome).
- 2022-05-06: `T05` / `convertible_bond` / 전환사채권발행결정 (718 days before outcome).
- 2022-09-22: `T05` / `paid_in_capital_increase` / 유상증자결정 (579 days before outcome).
- 2023-03-10: `T05` / `convertible_bond` / 전환사채권발행결정 (410 days before outcome).
- 2023-09-18: `T05` / `paid_in_capital_increase` / 유상증자결정 (218 days before outcome).
- 2023-11-22: `T05` / `convertible_bond` / 전환사채권발행결정 (153 days before outcome).

Post-anchor path to outcome:

- 2024-02-22: `T99` / `other` / 자기전환사채매도결정 (61 days before outcome).
- 2024-03-27: `T99` / `other` / 자기전환사채매도결정 (27 days before outcome).
- 2024-04-02: `T99` / `other` / 자기전환사채매도결정 (21 days before outcome).
- 2024-04-05: `T06` / `lawsuit` / 소송등의제기 (18 days before outcome).
- 2024-04-23: `T01` / `rehabilitation` / 회생절차개시신청 (0 days before outcome).

## 세븐브로이맥주 (267080)

- Anchor: 2023-12-31, horizon 730 days, predicted risk 0.4713.
- Outcome: BANKRUPTCY_OR_REHABILITATION, 2025-05-28, 514 days after anchor.

Model-visible sequence:

- 2022-08-11: `T99` / `other` / 무상증자결정 (1021 days before outcome).

Post-anchor path to outcome:

- 2024-10-24: `T06` / `merger_or_split` / 회사합병결정 (216 days before outcome).

## 코스나인 (082660)

- Anchor: 2022-12-31, horizon 730 days, predicted risk 0.4322.
- Outcome: BANKRUPTCY_OR_REHABILITATION, 2024-10-28, 667 days after anchor.

Model-visible sequence:

- 2022-10-14: `T05` / `asset_or_equity_transfer` / 타법인주식및출자증권양수결정 (745 days before outcome).

Post-anchor path to outcome:

- 2023-04-07: `T05` / `paid_in_capital_increase` / 유상증자결정 (570 days before outcome).
- 2023-04-14: `T05` / `convertible_bond` / 전환사채권발행결정 (563 days before outcome).
- 2023-07-19: `T05` / `convertible_bond` / 전환사채권발행결정 (467 days before outcome).
- 2024-05-02: `T05` / `paid_in_capital_increase` / 유상증자결정 (179 days before outcome).
- 2024-06-14: `T99` / `other` / 자기전환사채매도결정 (136 days before outcome).
- 2024-06-18: `T99` / `other` / 자기전환사채매도결정 (132 days before outcome).
- 2024-06-21: `T99` / `other` / 자기전환사채매도결정 (129 days before outcome).
- 2024-07-08: `T99` / `other` / 자기전환사채매도결정 (112 days before outcome).
- 2024-10-28: `T01` / `rehabilitation` / 회생절차개시신청 (0 days before outcome).

## 엔케이젠바이오텍코리아 (182400)

- Anchor: 2023-12-31, horizon 730 days, predicted risk 0.4220.
- Outcome: BANKRUPTCY_OR_REHABILITATION, 2024-04-18, 109 days after anchor.

Model-visible sequence:

- 2021-06-08: `T05` / `convertible_bond` / 전환사채권발행결정 (1045 days before outcome).
- 2021-06-08: `T05` / `paid_in_capital_increase` / 유상증자결정 (1045 days before outcome).
- 2022-04-07: `T05` / `convertible_bond` / 전환사채권발행결정 (742 days before outcome).
- 2023-09-08: `T05` / `convertible_bond` / 전환사채권발행결정 (223 days before outcome).
- 2023-09-11: `T05` / `bond_with_warrant` / 신주인수권부사채권발행결정 (220 days before outcome).
- 2023-10-20: `T05` / `paid_in_capital_increase` / 유상증자결정 (181 days before outcome).
- 2023-12-06: `T99` / `other` / 무상증자결정 (134 days before outcome).

Post-anchor path to outcome:

- 2024-04-18: `T01` / `rehabilitation` / 회생절차개시신청 (0 days before outcome).

## 비유테크놀러지 (230980)

- Anchor: 2024-12-31, horizon 365 days, predicted risk 0.4198.
- Outcome: BANKRUPTCY_OR_REHABILITATION, 2025-04-10, 100 days after anchor.

Model-visible sequence:

- 2022-08-25: `T05` / `paid_in_capital_increase` / 유상증자결정 (959 days before outcome).
- 2022-08-26: `T05` / `paid_in_capital_increase` / 유상증자결정 (958 days before outcome).
- 2022-10-12: `T06` / `merger_or_split` / 회사합병결정 (911 days before outcome).
- 2023-01-18: `T05` / `convertible_bond` / 전환사채권발행결정 (813 days before outcome).
- 2023-05-17: `T05` / `convertible_bond` / 전환사채권발행결정 (694 days before outcome).
- 2023-06-27: `T05` / `convertible_bond` / 전환사채권발행결정 (653 days before outcome).
- 2023-10-19: `T05` / `paid_in_capital_increase` / 유상증자결정 (539 days before outcome).
- 2023-12-08: `T05` / `asset_or_equity_transfer` / 타법인주식및출자증권양수결정 (489 days before outcome).
- 2024-03-29: `T05` / `paid_in_capital_increase` / 유상증자결정 (377 days before outcome).
- 2024-04-11: `T06` / `lawsuit` / 소송등의제기 (364 days before outcome).

## 한국비티비 (219750)

- Anchor: 2022-12-31, horizon 730 days, predicted risk 0.3776.
- Outcome: BANKRUPTCY_OR_REHABILITATION, 2023-03-29, 88 days after anchor.

Model-visible sequence:

- 2020-04-20: `T05` / `convertible_bond` / 전환사채권발행결정 (1073 days before outcome).
- 2020-10-29: `T05` / `paid_in_capital_increase` / 유상증자결정 (881 days before outcome).
- 2021-12-24: `T05` / `paid_in_capital_increase` / 유상증자결정 (460 days before outcome).
- 2022-03-16: `T05` / `convertible_bond` / 전환사채권발행결정 (378 days before outcome).

Post-anchor path to outcome:

- 2023-03-09: `T05` / `capital_reduction` / 감자결정 (20 days before outcome).
- 2023-03-29: `T01` / `rehabilitation` / 회생절차개시신청 (0 days before outcome).

## 무송지오씨 (135160)

- Anchor: 2022-12-31, horizon 365 days, predicted risk 0.0752.
- Outcome: DEFAULT, 2023-04-18, 108 days after anchor.

Model-visible sequence:

- 2020-06-16: `T05` / `paid_in_capital_increase` / 유상증자결정 (1036 days before outcome).
- 2020-09-28: `T99` / `other` / 유형자산양수결정 (932 days before outcome).
- 2022-07-26: `T05` / `convertible_bond` / 전환사채권발행결정 (266 days before outcome).

Post-anchor path to outcome:

- 2023-04-18: `T03` / `default` / 부도발생 (0 days before outcome).

## 태영건설 (009410)

- Anchor: 2023-12-31, horizon 730 days, predicted risk 0.0423.
- Outcome: DEFAULT, 2024-02-27, 58 days after anchor.

Model-visible sequence:

- No DART major-event rows in the case window before the anchor.

Post-anchor path to outcome:

- 2024-01-12: `T99` / `other` / 채권은행등의관리절차개시 (46 days before outcome).
- 2024-02-27: `T03` / `default` / 부도발생 (0 days before outcome).

Method note: T02 listing-status events include overseas listing termination notices for large firms and should be treated as a listing-status proxy, not automatically as financial distress. The representative cases above therefore prioritize rehabilitation and default outcomes.
