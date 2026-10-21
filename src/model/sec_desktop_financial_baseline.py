from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import orjson
except ImportError:  # pragma: no cover - optional speed path
    orjson = None


ROOT = Path("Paper 4")
DEFAULT_FACTS_ROOT = Path.home() / "Desktop" / "M&A" / "data" / "raw" / "sec"
HORIZONS = (365, 730)
MODELS = ("logistic", "random_forest")

CONCEPTS: dict[str, dict[str, Any]] = {
    "assets": {"tags": ["Assets"], "flow": False},
    "liabilities": {"tags": ["Liabilities"], "flow": False},
    "equity": {
        "tags": [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "PartnersCapital",
        ],
        "flow": False,
    },
    "current_assets": {"tags": ["AssetsCurrent"], "flow": False},
    "current_liabilities": {"tags": ["LiabilitiesCurrent"], "flow": False},
    "cash": {
        "tags": [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ],
        "flow": False,
    },
    "revenue": {
        "tags": [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
        ],
        "flow": True,
    },
    "net_income": {"tags": ["NetIncomeLoss", "ProfitLoss"], "flow": True},
    "operating_income": {"tags": ["OperatingIncomeLoss"], "flow": True},
}


def normalize_cik(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else ""


def read_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw.decode("utf-8"))


def safe_log(value: float | int | None) -> float:
    if value is None or pd.isna(value) or float(value) <= 0:
        return np.nan
    return float(np.log1p(float(value)))


def signed_log(value: float | int | None) -> float:
    if value is None or pd.isna(value):
        return np.nan
    value = float(value)
    return float(np.sign(value) * np.log1p(abs(value)))


def safe_div(num: float | int | None, den: float | int | None) -> float:
    if num is None or den is None or pd.isna(num) or pd.isna(den) or float(den) == 0:
        return np.nan
    return float(num) / float(den)


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


def top_fraction_metrics(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.1) -> dict[str, Any]:
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


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    return {
        "roc_auc": safe_metric(roc_auc_score, y_true, y_score),
        "average_precision": safe_metric(average_precision_score, y_true, y_score),
        "brier_score": finite_float(brier_score_loss(y_true, y_score)),
        "top_10pct": top_fraction_metrics(y_true, y_score, 0.10),
        "top_5pct": top_fraction_metrics(y_true, y_score, 0.05),
    }


def concept_records(payload: dict[str, Any], concept: str, meta: dict[str, Any]) -> list[dict[str, Any]]:
    usgaap = payload.get("facts", {}).get("us-gaap", {})
    spec = CONCEPTS[concept]
    rows: list[dict[str, Any]] = []
    for tag in spec["tags"]:
        obj = usgaap.get(tag)
        if not obj:
            continue
        units = obj.get("units") or {}
        for unit_name, records in units.items():
            if unit_name != "USD":
                continue
            for rec in records:
                value = rec.get("val")
                filed = rec.get("filed")
                end = rec.get("end")
                start = rec.get("start")
                if value is None or not filed or not end:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "cik": meta["cik"],
                        "entity_name": meta["entity_name"],
                        "concept": concept,
                        "tag": tag,
                        "value": value,
                        "filed": filed,
                        "end": end,
                        "start": start,
                        "duration_days": None,
                        "form": str(rec.get("form") or ""),
                        "fp": str(rec.get("fp") or ""),
                        "fy": rec.get("fy"),
                    }
                )
    return rows


def finalize_facts_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    facts = pd.DataFrame(rows)
    if facts.empty:
        return facts
    facts["filed"] = pd.to_datetime(facts["filed"], errors="coerce")
    facts["end"] = pd.to_datetime(facts["end"], errors="coerce")
    facts["start"] = pd.to_datetime(facts["start"], errors="coerce")
    facts = facts.dropna(subset=["filed", "end"]).copy()
    facts["duration_days"] = (facts["end"] - facts["start"]).dt.days
    facts = facts.sort_values(["cik", "concept", "filed", "end", "tag"])
    return facts


def extract_fact_records(
    facts_root: Path,
    ciks: set[str],
    max_ciks: int | None = None,
    progress_every: int = 250,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cik_list = sorted(ciks)
    if max_ciks and max_ciks > 0:
        cik_list = cik_list[:max_ciks]
    files_seen = 0
    files_with_any_facts = 0
    files_with_selected_facts = 0
    missing_files = 0
    parse_failures: list[dict[str, str]] = []
    for idx, cik in enumerate(cik_list, start=1):
        path = facts_root / f"CIK{cik}_companyfacts.json"
        if not path.exists():
            missing_files += 1
            if progress_every and idx % progress_every == 0:
                print(f"companyfacts progress: {idx}/{len(cik_list)} ciks, {len(rows)} selected rows", file=sys.stderr)
            continue
        files_seen += 1
        try:
            payload = read_json(path)
        except Exception as exc:
            parse_failures.append({"cik": cik, "source": "local_sec_companyfacts_cache", "error": repr(exc)[:500]})
            continue
        if payload.get("facts"):
            files_with_any_facts += 1
        meta = {"cik": cik, "entity_name": str(payload.get("entityName") or "")}
        before = len(rows)
        for concept in CONCEPTS:
            rows.extend(concept_records(payload, concept, meta))
        if len(rows) > before:
            files_with_selected_facts += 1
        if progress_every and idx % progress_every == 0:
            print(f"companyfacts progress: {idx}/{len(cik_list)} ciks, {len(rows)} selected rows", file=sys.stderr)
    facts = finalize_facts_frame(rows)
    summary = {
        "facts_root": "local_sec_companyfacts_cache",
        "requested_ciks": int(len(ciks)),
        "processed_ciks": int(len(cik_list)),
        "max_ciks": int(max_ciks) if max_ciks and max_ciks > 0 else None,
        "files_seen": int(files_seen),
        "missing_files": int(missing_files),
        "files_with_any_facts": int(files_with_any_facts),
        "files_with_selected_facts": int(files_with_selected_facts),
        "selected_fact_records": int(len(facts)),
        "parse_failures": int(len(parse_failures)),
        "parse_failures_sample": parse_failures[:20],
    }
    return facts, summary


def prepare_fact_lookup(facts: pd.DataFrame) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    if facts.empty:
        return {}
    work = facts.copy()
    work["form_rank"] = np.where(work["form"].isin(["10-K", "10-K/A", "20-F", "40-F"]), 1, 0)
    work["duration_rank"] = work["duration_days"].fillna(0)
    work["is_annual"] = (
        work["fp"].eq("FY")
        | work["form"].isin(["10-K", "10-K/A", "20-F", "40-F"])
        | work["duration_days"].fillna(0).ge(250)
    )
    work = work[work["filed"].notna()].sort_values(
        ["cik", "concept", "filed", "end", "form_rank", "duration_rank"],
        ascending=[True, True, True, True, True, True],
    )
    lookup: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for (cik, concept), group in work.groupby(["cik", "concept"], sort=False):
        concept_lookup: dict[str, np.ndarray] = {
            "filed": group["filed"].to_numpy(dtype="datetime64[ns]"),
            "value": group["value"].to_numpy(dtype=float),
            "tag": group["tag"].astype(str).to_numpy(),
        }
        if bool(CONCEPTS[concept]["flow"]):
            annual = group[group["is_annual"]]
            if not annual.empty:
                concept_lookup["annual_filed"] = annual["filed"].to_numpy(dtype="datetime64[ns]")
                concept_lookup["annual_value"] = annual["value"].to_numpy(dtype=float)
                concept_lookup["annual_tag"] = annual["tag"].astype(str).to_numpy()
        lookup.setdefault(str(cik), {})[str(concept)] = concept_lookup
    return lookup


def choose_fact_from_lookup(
    concept_lookup: dict[str, np.ndarray] | None,
    concept: str,
    anchor: pd.Timestamp,
) -> tuple[float | None, int | None, str | None]:
    if not concept_lookup:
        return None, None, None
    anchor64 = np.datetime64(anchor.to_datetime64())
    prefix = "annual_" if bool(CONCEPTS[concept]["flow"]) and "annual_filed" in concept_lookup else ""
    filed = concept_lookup[f"{prefix}filed"]
    if len(filed) == 0:
        return None, None, None
    idx = int(np.searchsorted(filed, anchor64, side="right") - 1)
    if idx < 0:
        return None, None, None
    filed_ts = pd.Timestamp(filed[idx])
    age = int((anchor - filed_ts).days)
    return float(concept_lookup[f"{prefix}value"][idx]), age, str(concept_lookup[f"{prefix}tag"][idx])


def build_financial_features(panel: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["firm_id"] = panel["firm_id"].map(normalize_cik)
    panel["anchor_date_ts"] = pd.to_datetime(panel["anchor_date"], errors="coerce")
    fact_lookup = prepare_fact_lookup(facts)
    rows: list[dict[str, Any]] = []
    anchors = panel[["task", "firm_id", "anchor_year", "anchor_date", "anchor_date_ts"]].drop_duplicates()
    for row in anchors.itertuples(index=False):
        firm_lookup = fact_lookup.get(row.firm_id, {})
        rec: dict[str, Any] = {
            "task": row.task,
            "firm_id": row.firm_id,
            "anchor_year": row.anchor_year,
            "anchor_date": row.anchor_date,
        }
        fact_count = 0
        ages: list[int] = []
        for concept in CONCEPTS:
            value, age, tag = choose_fact_from_lookup(firm_lookup.get(concept), concept, pd.Timestamp(row.anchor_date_ts))
            rec[f"raw_{concept}"] = value
            rec[f"{concept}_fact_age_days"] = age
            rec[f"{concept}_source_tag"] = tag or ""
            if value is not None:
                fact_count += 1
            if age is not None:
                ages.append(age)
        rec["fin_available_fact_count"] = fact_count
        rec["fin_min_fact_age_days"] = min(ages) if ages else np.nan
        rec["fin_max_fact_age_days"] = max(ages) if ages else np.nan
        rows.append(rec)
    features = pd.DataFrame(rows)

    features["fin_log_assets"] = features["raw_assets"].map(safe_log)
    features["fin_log_revenue"] = features["raw_revenue"].map(safe_log)
    features["fin_log_net_income_signed"] = features["raw_net_income"].map(signed_log)
    features["fin_log_operating_income_signed"] = features["raw_operating_income"].map(signed_log)
    features["fin_leverage"] = [safe_div(n, d) for n, d in zip(features["raw_liabilities"], features["raw_assets"])]
    features["fin_equity_ratio"] = [safe_div(n, d) for n, d in zip(features["raw_equity"], features["raw_assets"])]
    features["fin_current_ratio"] = [safe_div(n, d) for n, d in zip(features["raw_current_assets"], features["raw_current_liabilities"])]
    features["fin_cash_assets"] = [safe_div(n, d) for n, d in zip(features["raw_cash"], features["raw_assets"])]
    features["fin_roa"] = [safe_div(n, d) for n, d in zip(features["raw_net_income"], features["raw_assets"])]
    features["fin_operating_roa"] = [safe_div(n, d) for n, d in zip(features["raw_operating_income"], features["raw_assets"])]
    features["fin_net_margin"] = [safe_div(n, d) for n, d in zip(features["raw_net_income"], features["raw_revenue"])]
    features["fin_operating_margin"] = [safe_div(n, d) for n, d in zip(features["raw_operating_income"], features["raw_revenue"])]
    features["fin_loss_flag"] = features["raw_net_income"].lt(0).fillna(False).astype(int)
    features["fin_negative_equity_flag"] = features["raw_equity"].lt(0).fillna(False).astype(int)
    features["fin_missing_record_flag"] = features["fin_available_fact_count"].eq(0).astype(int)
    features = features.drop(columns=["anchor_date_ts"], errors="ignore")
    return panel.drop(columns=["anchor_date_ts"]).merge(features, on=["task", "firm_id", "anchor_year", "anchor_date"], how="left")


def financial_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("fin_")]


def sequence_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("seq_")]


def make_model(model_name: str) -> Pipeline:
    if model_name == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=150,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise ValueError(model_name)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], label: str, model_name: str) -> np.ndarray:
    y_train = train[label].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        return np.repeat(float(y_train.mean()) if len(y_train) else 0.0, len(test))
    model = make_model(model_name)
    model.fit(train[features], y_train)
    return model.predict_proba(test[features])[:, 1]


def evaluate_panel(panel: pd.DataFrame, train_end_year: int, test_start_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fin_features = financial_columns(panel)
    seq_features = sequence_columns(panel)
    feature_sets = {
        "financial_only": fin_features,
        "sequence_plus_financial": seq_features + fin_features,
    }
    metrics: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for task, task_panel in panel.groupby("task"):
        train = task_panel[task_panel["anchor_year"] <= train_end_year].copy()
        test = task_panel[task_panel["anchor_year"] >= test_start_year].copy()
        for horizon in HORIZONS:
            label = f"label_{horizon}d"
            y_test = test[label].astype(int).to_numpy()
            for feature_set, features in feature_sets.items():
                for model_name in MODELS:
                    score = fit_predict(train, test, features, label, model_name)
                    m = evaluate_predictions(y_test, score)
                    top10 = m["top_10pct"]
                    metrics.append(
                        {
                            "task": task,
                            "horizon_days": horizon,
                            "feature_set": feature_set,
                            "model": model_name,
                            "feature_count": int(len(features)),
                            "train_rows": int(len(train)),
                            "test_rows": int(len(test)),
                            "train_positives": int(train[label].sum()),
                            "test_positives": int(test[label].sum()),
                            "financial_coverage_rate": float(test["fin_missing_record_flag"].eq(0).mean())
                            if "fin_missing_record_flag" in test
                            else None,
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
                    pred["feature_set"] = feature_set
                    pred["model"] = model_name
                    pred["score"] = score
                    prediction_frames.append(pred)
    return pd.DataFrame(metrics), pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()


def write_markdown(summary: dict[str, Any], metrics: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Local SEC Financial Baseline",
        "",
        "This file evaluates SEC companyfacts-derived financial features on the same local SEC validation panel used for the sequence-only v2 check.",
        "",
        "## Coverage",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Requested validation CIKs | {summary['facts_extraction']['requested_ciks']} |",
        f"| Companyfacts files seen | {summary['facts_extraction']['files_seen']} |",
        f"| Files with selected facts | {summary['facts_extraction']['files_with_selected_facts']} |",
        f"| Selected fact records | {summary['facts_extraction']['selected_fact_records']} |",
        f"| Financial panel rows | {summary['panel_rows']} |",
        f"| Financial feature count | {summary['financial_feature_count']} |",
        f"| Sequence feature count | {summary['sequence_feature_count']} |",
        "",
        "## Metrics",
        "",
        "| Task | Horizon | Feature set | Model | Test rows | Test positives | Coverage | ROC AUC | Avg precision | Brier | Top 10% precision | Top 10% recall | Top 10% lift |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            "| {task} | {horizon} | {feature_set} | {model} | {test_rows} | {test_positives} | {cov} | {roc} | {ap} | {brier} | {p10} | {r10} | {l10} |".format(
                task=row.task,
                horizon=row.horizon_days,
                feature_set=row.feature_set,
                model=row.model,
                test_rows=row.test_rows,
                test_positives=row.test_positives,
                cov=round(row.financial_coverage_rate, 4) if pd.notna(row.financial_coverage_rate) else "",
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
            "- Companyfacts are local SEC API artifacts already present in the local SEC cache; no new SEC API call is made.",
            "- Features are point-in-time approximations using facts filed on or before each anchor date.",
            "- This adds a financial comparator for SEC v2; filing-body text is handled by separate target-item text artifacts and all-8-K text scope remains pending.",
            "- Bankruptcy-only positives remain sparse; all-distress metrics are the more stable SEC comparator.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts-root", default=str(DEFAULT_FACTS_ROOT))
    parser.add_argument("--panel-path", default=str(ROOT / "data" / "model_sec_desktop" / "sec_desktop_validation_panel.parquet"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "model_sec_desktop"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--max-ciks", type=int, default=0, help="Optional smoke-test cap. The default 0 processes all validation CIKs.")
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(args.panel_path)
    panel["firm_id"] = panel["firm_id"].map(normalize_cik)
    ciks = set(panel["firm_id"].dropna().astype(str))
    facts, facts_summary = extract_fact_records(
        Path(args.facts_root),
        ciks,
        max_ciks=args.max_ciks or None,
        progress_every=args.progress_every,
    )
    facts.to_parquet(out_dir / "sec_desktop_companyfacts_selected.parquet", index=False)

    financial_panel = build_financial_features(panel, facts)
    financial_panel.to_parquet(out_dir / "sec_desktop_financial_panel.parquet", index=False)
    financial_panel.to_csv(out_dir / "sec_desktop_financial_panel.csv", index=False, encoding="utf-8-sig")

    metrics, predictions = evaluate_panel(financial_panel, args.train_end_year, args.test_start_year)
    metrics.to_csv(results_dir / "sec_desktop_financial_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(results_dir / "sec_desktop_financial_baseline_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "created",
        "facts_extraction": facts_summary,
        "panel_rows": int(len(financial_panel)),
        "firms": int(financial_panel["firm_id"].nunique()) if not financial_panel.empty else 0,
        "financial_feature_count": int(len(financial_columns(financial_panel))),
        "sequence_feature_count": int(len(sequence_columns(financial_panel))),
        "train_end_year": args.train_end_year,
        "test_start_year": args.test_start_year,
        "metrics": metrics.to_dict(orient="records"),
        "guardrail": "SEC v2 financial baseline from local companyfacts; separate target-item filing-body baseline exists, while all-8-K text scope remains pending.",
    }
    (results_dir / "sec_desktop_financial_baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(summary, metrics, results_dir / "sec_desktop_financial_baseline_summary.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
