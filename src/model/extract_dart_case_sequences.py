from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STRESS_LABELS = ["BANKRUPTCY_OR_REHABILITATION", "DEFAULT"]
MODEL_NAME = "combined_random_forest"


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def normalize_predictions(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path)
    pred["target_bool"] = pred["target"].astype(str).str.lower().eq("true")
    pred["predicted_risk"] = pd.to_numeric(pred["predicted_risk"], errors="coerce")
    pred["anchor_date"] = pd.to_datetime(pred["anchor_date"], errors="coerce")
    pred["future_outcome_date"] = pd.to_datetime(pred["future_outcome_date"], errors="coerce")
    pred["security_id"] = pred["security_id"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    pred["firm_id"] = pred["firm_id"].astype(str).str.zfill(8)
    return pred


def select_cases(pred: pd.DataFrame, max_bankruptcy: int, max_default: int) -> pd.DataFrame:
    stress = pred[
        pred["model"].eq(MODEL_NAME)
        & pred["target_bool"]
        & pred["future_outcome_label"].isin(STRESS_LABELS)
    ].copy()
    stress = stress.sort_values("predicted_risk", ascending=False)
    stress = stress.drop_duplicates(["firm_id", "future_outcome_label"], keep="first")

    selected = []
    limits = {
        "BANKRUPTCY_OR_REHABILITATION": max_bankruptcy,
        "DEFAULT": max_default,
    }
    for label, limit in limits.items():
        selected.append(stress[stress["future_outcome_label"].eq(label)].head(limit))
    out = pd.concat(selected, ignore_index=True)
    return out.sort_values(["future_outcome_label", "predicted_risk"], ascending=[True, False])


def event_rows_for_case(case: pd.Series, events: pd.DataFrame, window_days: int) -> list[dict]:
    firm_events = events[events["firm_id"].eq(case["firm_id"])].copy()
    firm_events["event_date"] = pd.to_datetime(firm_events["event_date"], errors="coerce")
    start = case["future_outcome_date"] - pd.Timedelta(days=window_days)

    visible = firm_events[
        firm_events["event_date"].between(start, case["anchor_date"], inclusive="both")
    ].copy()
    bridge = firm_events[
        (firm_events["event_date"] > case["anchor_date"])
        & (firm_events["event_date"] <= case["future_outcome_date"])
    ].copy()

    rows = []
    for segment, frame in (("model_visible", visible), ("post_anchor_to_outcome", bridge)):
        frame = frame.sort_values("event_date")
        for order, event in enumerate(frame.itertuples(index=False), start=1):
            rows.append(
                {
                    "case_firm_id": case["firm_id"],
                    "case_firm_name": case["firm_name"],
                    "security_id": case["security_id"],
                    "horizon_days": int(case["horizon_days"]),
                    "anchor_year": int(case["anchor_year"]),
                    "anchor_date": case["anchor_date"],
                    "predicted_risk": float(case["predicted_risk"]),
                    "future_outcome_date": case["future_outcome_date"],
                    "future_outcome_label": case["future_outcome_label"],
                    "days_to_outcome": case["days_to_outcome"],
                    "segment": segment,
                    "sequence_order": order,
                    "event_date": event.event_date,
                    "days_before_outcome": int((case["future_outcome_date"] - event.event_date).days),
                    "taxonomy_code": event.taxonomy_code,
                    "event_subtype": event.event_subtype,
                    "clean_event_title": event.clean_event_title,
                    "report_nm": event.report_nm,
                    "is_model_visible": segment == "model_visible",
                }
            )
    return rows


def write_cases_md(cases: pd.DataFrame, event_rows: pd.DataFrame, path: Path) -> None:
    lines = ["# DART Distress Case Sequences", ""]
    lines.append("These examples use the `combined_random_forest` test-set predictions. `model_visible` events occurred on or before the prediction anchor date; `post_anchor_to_outcome` events are shown only to narrate the subsequent path to the outcome.")
    lines.append("")
    lines.append("| Firm | Security | Horizon | Anchor | Outcome | Days to outcome | Predicted risk |")
    lines.append("|---|---|---:|---|---|---:|---:|")
    for case in cases.itertuples(index=False):
        lines.append(
            f"| {case.firm_name} | {case.security_id} | {case.horizon_days} | {str(case.anchor_date)[:10]} | {case.future_outcome_label} on {str(case.future_outcome_date)[:10]} | {case.days_to_outcome:.0f} | {case.predicted_risk:.4f} |"
        )
    lines.append("")

    for case in cases.itertuples(index=False):
        case_events = event_rows[event_rows["case_firm_id"].eq(case.firm_id)].copy()
        visible = case_events[case_events["segment"].eq("model_visible")]
        bridge = case_events[case_events["segment"].eq("post_anchor_to_outcome")]
        lines.append(f"## {case.firm_name} ({case.security_id})")
        lines.append("")
        lines.append(f"- Anchor: {str(case.anchor_date)[:10]}, horizon {case.horizon_days} days, predicted risk {case.predicted_risk:.4f}.")
        lines.append(f"- Outcome: {case.future_outcome_label}, {str(case.future_outcome_date)[:10]}, {case.days_to_outcome:.0f} days after anchor.")
        lines.append("")
        lines.append("Model-visible sequence:")
        lines.append("")
        if visible.empty:
            lines.append("- No DART major-event rows in the case window before the anchor.")
        else:
            for row in visible.sort_values("event_date").itertuples(index=False):
                lines.append(
                    f"- {str(row.event_date)[:10]}: `{row.taxonomy_code}` / `{row.event_subtype}` / {row.clean_event_title} ({row.days_before_outcome} days before outcome)."
                )
        lines.append("")
        if not bridge.empty:
            lines.append("Post-anchor path to outcome:")
            lines.append("")
            for row in bridge.sort_values("event_date").itertuples(index=False):
                lines.append(
                    f"- {str(row.event_date)[:10]}: `{row.taxonomy_code}` / `{row.event_subtype}` / {row.clean_event_title} ({row.days_before_outcome} days before outcome)."
                )
            lines.append("")
    lines.append("Method note: T02 listing-status events include overseas listing termination notices for large firms and should be treated as a listing-status proxy, not automatically as financial distress. The representative cases above therefore prioritize rehabilitation and default outcomes.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-path", default=str(Path("Paper 4") / "data" / "processed" / "events_dart.parquet"))
    parser.add_argument("--predictions-path", default=str(Path("Paper 4") / "results" / "dart_model_predictions.csv"))
    parser.add_argument("--results-dir", default=str(Path("Paper 4") / "results"))
    parser.add_argument("--window-days", type=int, default=1095)
    parser.add_argument("--max-bankruptcy", type=int, default=6)
    parser.add_argument("--max-default", type=int, default=2)
    args = parser.parse_args()

    events = read_frame(Path(args.events_path))
    events["firm_id"] = events["firm_id"].astype(str).str.zfill(8)
    pred = normalize_predictions(Path(args.predictions_path))
    cases = select_cases(pred, args.max_bankruptcy, args.max_default)
    rows: list[dict] = []
    for case in cases.to_dict(orient="records"):
        rows.extend(event_rows_for_case(pd.Series(case), events, args.window_days))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    event_rows = pd.DataFrame(rows)
    cases.to_csv(results_dir / "dart_distress_cases.csv", index=False, encoding="utf-8-sig")
    event_rows.to_csv(results_dir / "dart_distress_case_sequences.csv", index=False, encoding="utf-8-sig")
    write_cases_md(cases, event_rows, results_dir / "dart_distress_case_sequences.md")
    print(f"selected_cases={len(cases)} event_rows={len(event_rows)}")


if __name__ == "__main__":
    main()
