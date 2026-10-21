from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTCOME_SUBTYPES = {"bankruptcy_or_receivership", "delisting_notice", "debt_acceleration_trigger"}
BANKRUPTCY_SUBTYPES = {"bankruptcy_or_receivership"}
TAXONOMY_CODES = ("T01", "T02", "T03", "T05", "T06", "T99")
WINDOWS = (90, 180, 365, 730)


def finite_float(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    return float(value) if isinstance(value, (np.floating, float)) else int(value)


def safe_metric(fn, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return finite_float(fn(y_true, y_score))
    except Exception:
        return None


def top_fraction_metrics(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.1) -> dict:
    n = len(y_true)
    if n == 0:
        return {"top_fraction": fraction, "top_n": 0, "precision": None, "recall": None, "lift": None}
    top_n = max(1, int(np.ceil(n * fraction)))
    order = np.argsort(-y_score)
    top = order[:top_n]
    base_rate = float(np.mean(y_true)) if n else 0.0
    precision = float(np.mean(y_true[top])) if top_n else None
    positives = float(np.sum(y_true))
    recall = float(np.sum(y_true[top]) / positives) if positives else None
    lift = float(precision / base_rate) if base_rate and precision is not None else None
    return {
        "top_fraction": fraction,
        "top_n": int(top_n),
        "precision": finite_float(precision),
        "recall": finite_float(recall),
        "lift": finite_float(lift),
    }


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "roc_auc": safe_metric(roc_auc_score, y_true, y_score),
        "average_precision": safe_metric(average_precision_score, y_true, y_score),
        "brier_score": finite_float(brier_score_loss(y_true, y_score)),
        "top_10pct": top_fraction_metrics(y_true, y_score, 0.10),
        "top_5pct": top_fraction_metrics(y_true, y_score, 0.05),
    }


def first_outcome_dates(events: pd.DataFrame, subtype_set: set[str]) -> pd.DataFrame:
    out = events[events["event_subtype"].isin(subtype_set)].copy()
    if out.empty:
        return pd.DataFrame(columns=["firm_id", "outcome_date", "outcome_label"])
    out = out.sort_values(["firm_id", "event_date"])
    first = out.groupby("firm_id", as_index=False).first()
    first = first.rename(columns={"event_date": "outcome_date", "event_subtype": "outcome_label"})
    return first[["firm_id", "outcome_date", "outcome_label"]]


def days_since(anchor: pd.Timestamp, dates: Iterable[pd.Timestamp]) -> int:
    clean = [pd.Timestamp(d) for d in dates if pd.notna(d) and pd.Timestamp(d) <= anchor]
    if not clean:
        return 9999
    return int((anchor - max(clean)).days)


def add_history_features(history: pd.DataFrame, anchor: pd.Timestamp, prefix: str = "seq") -> dict[str, float | int]:
    features: dict[str, float | int] = {
        f"{prefix}_history_events": int(len(history)),
        f"{prefix}_history_unique_items": int(history["item"].nunique()) if "item" in history else 0,
        f"{prefix}_days_since_last_event": days_since(anchor, history["event_date"]) if not history.empty else 9999,
    }
    for code in TAXONOMY_CODES:
        code_events = history[history["taxonomy_code"].eq(code)]
        features[f"{prefix}_count_{code}_all"] = int(len(code_events))
        features[f"{prefix}_days_since_{code}"] = days_since(anchor, code_events["event_date"]) if not code_events.empty else 9999
    for window in WINDOWS:
        start = anchor - pd.Timedelta(days=window)
        w = history[(history["event_date"] > start) & (history["event_date"] <= anchor)]
        features[f"{prefix}_events_{window}d"] = int(len(w))
        features[f"{prefix}_unique_items_{window}d"] = int(w["item"].nunique()) if "item" in w else 0
        for code in TAXONOMY_CODES:
            features[f"{prefix}_{code}_{window}d"] = int(w["taxonomy_code"].eq(code).sum())
    events_365 = features[f"{prefix}_events_365d"]
    features[f"{prefix}_events_90_to_365_ratio"] = float(features[f"{prefix}_events_90d"] / events_365) if events_365 else 0.0
    return features


def build_panel(
    events: pd.DataFrame,
    task: str,
    subtype_set: set[str],
    predictor_excluded_subtypes: set[str],
    start_anchor_year: int,
    end_anchor_year: int,
    horizons: Iterable[int],
) -> pd.DataFrame:
    events = events.copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events[events["event_date"].notna()].copy()
    outcomes = first_outcome_dates(events, subtype_set)
    outcome_by_firm = {
        row.firm_id: (pd.Timestamp(row.outcome_date), row.outcome_label)
        for row in outcomes.itertuples(index=False)
    }
    predictors = events[~events["event_subtype"].isin(predictor_excluded_subtypes)].copy()
    predictors_by_firm = {
        firm_id: group.sort_values("event_date")
        for firm_id, group in predictors.groupby("firm_id")
    }

    rows: list[dict[str, object]] = []
    for firm_id, firm_events in events.groupby("firm_id"):
        firm_predictors = predictors_by_firm.get(firm_id, predictors.iloc[0:0])
        first_event = pd.Timestamp(firm_events["event_date"].min())
        outcome_date, outcome_label = outcome_by_firm.get(firm_id, (pd.NaT, None))
        for anchor_year in range(start_anchor_year, end_anchor_year + 1):
            anchor = pd.Timestamp(f"{anchor_year}-12-31")
            if first_event > anchor:
                continue
            if pd.notna(outcome_date) and outcome_date <= anchor:
                continue
            history = firm_predictors[firm_predictors["event_date"] <= anchor]
            base = {
                "task": task,
                "firm_id": firm_id,
                "firm_name": str(firm_events["company_name"].dropna().iloc[0])
                if "company_name" in firm_events and firm_events["company_name"].notna().any()
                else "",
                "anchor_year": anchor_year,
                "anchor_date": anchor.date().isoformat(),
                "outcome_date": outcome_date.date().isoformat() if pd.notna(outcome_date) else "",
                "outcome_label": outcome_label or "",
            }
            base.update(add_history_features(history, anchor))
            for horizon in horizons:
                label = int(pd.notna(outcome_date) and anchor < outcome_date <= anchor + pd.Timedelta(days=horizon))
                base[f"label_{horizon}d"] = label
            rows.append(base)
    return pd.DataFrame(rows).sort_values(["task", "firm_id", "anchor_year"])


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [col for col in panel.columns if col.startswith("seq_")]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], label: str, model_name: str) -> tuple[np.ndarray, object]:
    if model_name == "logistic":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
            ]
        )
    elif model_name == "random_forest":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(model_name)

    y_train = train[label].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        return np.repeat(float(y_train.mean()) if len(y_train) else 0.0, len(test)), None
    model.fit(train[features], y_train)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(test[features])[:, 1]
    else:
        score = model.decision_function(test[features])
    return np.asarray(score, dtype=float), model


def evaluate_panel(panel: pd.DataFrame, horizons: Iterable[int], train_end_year: int, test_start_year: int) -> tuple[list[dict], pd.DataFrame]:
    features = feature_columns(panel)
    metrics: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    for task, task_panel in panel.groupby("task"):
        train = task_panel[task_panel["anchor_year"] <= train_end_year].copy()
        test = task_panel[task_panel["anchor_year"] >= test_start_year].copy()
        for horizon in horizons:
            label = f"label_{horizon}d"
            for model_name in ("logistic", "random_forest"):
                score, _ = fit_predict(train, test, features, label, model_name)
                y_test = test[label].astype(int).to_numpy()
                m = evaluate_predictions(y_test, score)
                top10 = m["top_10pct"]
                metrics.append(
                    {
                        "task": task,
                        "horizon_days": horizon,
                        "model": model_name,
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "train_positives": int(train[label].sum()),
                        "test_positives": int(test[label].sum()),
                        "roc_auc": m["roc_auc"],
                        "average_precision": m["average_precision"],
                        "brier_score": m["brier_score"],
                        "top_10pct_precision": top10["precision"],
                        "top_10pct_recall": top10["recall"],
                        "top_10pct_lift": top10["lift"],
                    }
                )
                pred = test[["task", "firm_id", "firm_name", "anchor_year", "anchor_date", "outcome_date", "outcome_label", label]].copy()
                pred = pred.rename(columns={label: "label"})
                pred["horizon_days"] = horizon
                pred["model"] = model_name
                pred["score"] = score
                prediction_frames.append(pred)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return metrics, predictions


def write_markdown(summary: dict, metrics: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Local SEC Sequence Validation",
        "",
        "This is a preliminary v2 validation using locally cached local SEC submissions metadata. It performs out-of-time tests on SEC sequence-only features and does not change the DART-first v1 headline claim.",
        "",
        "## Panel",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Panel rows | {summary['panel_rows']} |",
        f"| Firms | {summary['firms']} |",
        f"| Anchor years | {summary['start_anchor_year']}-{summary['end_anchor_year']} |",
        f"| Train anchors | <= {summary['train_end_year']} |",
        f"| Test anchors | >= {summary['test_start_year']} |",
        "",
        "## Metrics",
        "",
        "| Task | Horizon | Model | Train rows | Test rows | Test positives | ROC AUC | Avg precision | Brier | Top 10% precision | Top 10% recall | Top 10% lift |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            "| {task} | {horizon} | {model} | {train_rows} | {test_rows} | {test_positives} | {roc} | {ap} | {brier} | {p10} | {r10} | {l10} |".format(
                task=row.task,
                horizon=row.horizon_days,
                model=row.model,
                train_rows=row.train_rows,
                test_rows=row.test_rows,
                test_positives=row.test_positives,
                roc=round(row.roc_auc, 4) if pd.notna(row.roc_auc) else "",
                ap=round(row.average_precision, 4) if pd.notna(row.average_precision) else "",
                brier=round(row.brier_score, 4) if pd.notna(row.brier_score) else "",
                p10=round(row.top_10pct_precision, 4) if pd.notna(row.top_10pct_precision) else "",
                r10=round(row.top_10pct_recall, 4) if pd.notna(row.top_10pct_recall) else "",
                l10=round(row.top_10pct_lift, 2) if pd.notna(row.top_10pct_lift) else "",
            )
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This is sequence-only SEC validation; no SEC financial fundamentals are included.",
            "- The split is out-of-time by anchor year, not firm-disjoint.",
            "- The source is cached SEC submissions metadata with available archive JSON included; final U.S. claim scope is still pending.",
            "- Use these results as v2 feasibility evidence unless the sampling frame and validation design are expanded in the manuscript.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-path", default=str(Path("Paper 4") / "data" / "processed" / "sec_desktop_population_events.parquet"))
    parser.add_argument("--out-dir", default=str(Path("Paper 4") / "data" / "model_sec_desktop"))
    parser.add_argument("--results-dir", default=str(Path("Paper 4") / "results"))
    parser.add_argument("--start-anchor-year", type=int, default=2016)
    parser.add_argument("--end-anchor-year", type=int, default=2023)
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(args.events_path)
    all_distress = build_panel(
        events,
        "all_distress_items_103_204_301",
        OUTCOME_SUBTYPES,
        OUTCOME_SUBTYPES,
        args.start_anchor_year,
        args.end_anchor_year,
        (365, 730),
    )
    bankruptcy = build_panel(
        events,
        "bankruptcy_item_103",
        BANKRUPTCY_SUBTYPES,
        BANKRUPTCY_SUBTYPES,
        args.start_anchor_year,
        args.end_anchor_year,
        (365, 730),
    )
    panel = pd.concat([all_distress, bankruptcy], ignore_index=True)
    panel.to_parquet(out_dir / "sec_desktop_validation_panel.parquet", index=False)
    panel.to_csv(out_dir / "sec_desktop_validation_panel.csv", index=False, encoding="utf-8-sig")

    metric_rows, predictions = evaluate_panel(panel, (365, 730), args.train_end_year, args.test_start_year)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(results_dir / "sec_desktop_validation_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(results_dir / "sec_desktop_validation_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "panel_rows": int(len(panel)),
        "firms": int(panel["firm_id"].nunique()) if not panel.empty else 0,
        "start_anchor_year": args.start_anchor_year,
        "end_anchor_year": args.end_anchor_year,
        "train_end_year": args.train_end_year,
        "test_start_year": args.test_start_year,
        "feature_count": len(feature_columns(panel)),
        "metrics": metric_rows,
        "guardrail": "Preliminary v2 SEC sequence-only validation; not a DART v1 headline claim.",
    }
    (results_dir / "sec_desktop_validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, metrics, results_dir / "sec_desktop_validation_summary.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
