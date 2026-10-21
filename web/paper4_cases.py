"""Exports the Paper 4 distress case sequences for the front page.

Reads the accepted-paper result files (read only) and writes one JSON with, per case,
the model-visible disclosure sequence, the post-anchor path, the predicted risk and the
share of the out-of-time test set that scored higher (rank among the same model and horizon).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PAPER4_RESULTS = Path(__file__).resolve().parents[1] / "results"
CASES = PAPER4_RESULTS / "dart_distress_cases.csv"
SEQUENCES = PAPER4_RESULTS / "dart_distress_case_sequences.csv"
PREDICTIONS = PAPER4_RESULTS / "dart_model_predictions.csv"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _scores_by_model(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[float]]:
    scores: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        scores.setdefault((row["horizon_days"], row["model"]), []).append(float(row["predicted_risk"]))
    return scores


def _event(row: dict[str, str]) -> dict[str, object]:
    return {
        "date": row["event_date"],
        "code": row["taxonomy_code"],
        "title": row["clean_event_title"],
        "days_before_outcome": int(float(row["days_before_outcome"])),
    }


def build_cases() -> list[dict[str, object]]:
    scores = _scores_by_model(_read(PREDICTIONS))
    sequences = _read(SEQUENCES)
    cases = []
    for case in _read(CASES):
        risk = float(case["predicted_risk"])
        pool = scores[(case["horizon_days"], case["model"])]
        steps = [row for row in sequences if row["case_firm_id"] == case["firm_id"]]
        steps.sort(key=lambda row: int(row["sequence_order"]))
        cases.append({
            "firm": case["firm_name"],
            "security_id": case["security_id"],
            "model": case["model"],
            "horizon_days": int(case["horizon_days"]),
            "anchor_date": case["anchor_date"],
            "predicted_risk": risk,
            "share_scored_higher": sum(score > risk for score in pool) / len(pool),
            "test_rows": len(pool),
            "outcome_label": case["future_outcome_label"],
            "outcome_date": case["future_outcome_date"],
            "days_to_outcome": int(float(case["days_to_outcome"])),
            "visible": [_event(row) for row in steps if row["is_model_visible"] == "True"],
            "after": [_event(row) for row in steps if row["is_model_visible"] != "True"],
        })
    return cases


def write_cases(path: Path) -> int:
    cases = build_cases()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(cases)
