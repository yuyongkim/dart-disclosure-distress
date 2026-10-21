from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("Paper 4")
DEFAULT_MANIFEST = ROOT / "data" / "raw" / "sec_all8k_text" / "sec_all8k_text_manifest.csv"
OUT_ROOT = ROOT / "data" / "raw" / "sec_all8k_text"
RESULTS_DIR = ROOT / "results"
DEFAULT_TEXT_ROOT = ROOT / "data" / "raw" / "sec_filing_text"


BUCKET_ORDER = [
    "earnings_item_2_02",
    "reg_fd_item_7_01",
    "other_events_item_8_01",
    "management_item_5_02",
    "agreement_or_financing_items_1_01_2_03",
    "auditor_item_4_01",
    "securities_items_3_x",
    "other_non_target_items",
]


def split_items(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()}


def normalize_cik(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else ""


def accession_nodash(value: object) -> str:
    return str(value or "").replace("-", "").strip()


def canonical_text_path(text_root: Path, cik: object, accession: object) -> str:
    cik_norm = normalize_cik(cik)
    accession_key = accession_nodash(accession)
    if not cik_norm or not accession_key:
        return ""
    return str(text_root / cik_norm / f"{accession_key}.txt")


def refresh_manifest_text_paths(manifest: pd.DataFrame, text_root: Path) -> pd.DataFrame:
    manifest = manifest.copy()
    original_path = (
        manifest["local_text_path"].fillna("").astype(str)
        if "local_text_path" in manifest
        else pd.Series([""] * len(manifest), index=manifest.index)
    )
    manifest["manifest_local_path_redacted"] = original_path.str.startswith("public_")
    manifest["local_text_path"] = manifest.apply(
        lambda row: canonical_text_path(text_root, row.get("cik"), row.get("accession_number")),
        axis=1,
    )
    manifest["local_text_exists"] = manifest["local_text_path"].map(lambda value: bool(value) and Path(value).exists())
    manifest["local_text_bytes"] = manifest["local_text_path"].map(
        lambda value: int(Path(value).stat().st_size) if value and Path(value).exists() else 0
    )
    return manifest


def item_bucket(items: object, is_target: bool) -> str:
    values = split_items(items)
    if is_target:
        if "1.03" in values:
            return "target_item_1_03"
        if "2.04" in values:
            return "target_item_2_04"
        if "3.01" in values:
            return "target_item_3_01"
        return "target_distress_item"
    if "2.02" in values:
        return "earnings_item_2_02"
    if "7.01" in values:
        return "reg_fd_item_7_01"
    if "8.01" in values:
        return "other_events_item_8_01"
    if "5.02" in values:
        return "management_item_5_02"
    if {"1.01", "2.03"}.intersection(values):
        return "agreement_or_financing_items_1_01_2_03"
    if "4.01" in values:
        return "auditor_item_4_01"
    if any(value.startswith("3.") for value in values):
        return "securities_items_3_x"
    return "other_non_target_items"


def stable_sample(group: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(group) <= n:
        return group.copy()
    return group.sample(n=n, random_state=seed)


def add_sampling_columns(manifest: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.copy()
    manifest["filing_date_ts"] = pd.to_datetime(manifest["filing_date"], errors="coerce")
    manifest["filing_year"] = manifest["filing_date_ts"].dt.year
    manifest["has_target_distress_item"] = manifest["has_target_distress_item"].astype(bool)
    manifest["item_bucket"] = manifest.apply(
        lambda row: item_bucket(row.get("items"), bool(row.get("has_target_distress_item"))),
        axis=1,
    )
    return manifest


def build_non_target_sample(manifest: pd.DataFrame, per_year_bucket: int, max_rows: int, seed: int) -> pd.DataFrame:
    non_target = manifest[
        ~manifest["has_target_distress_item"]
        & manifest["filing_year"].notna()
        & manifest["accession_number"].notna()
    ].copy()
    sampled = []
    for _, group in non_target.groupby(["filing_year", "item_bucket"], sort=True):
        sampled.append(stable_sample(group, per_year_bucket, seed))
    if not sampled:
        return pd.DataFrame(columns=list(manifest.columns))
    sample = pd.concat(sampled, ignore_index=True)
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=seed).copy()
    sample = sample.sort_values(["filing_year", "item_bucket", "cik", "accession_number"], kind="mergesort")
    sample["sample_role"] = "non_target_text_scope_sample"
    return sample


def build_target_reference_sample(manifest: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0:
        return pd.DataFrame(columns=list(manifest.columns))
    target = manifest[
        manifest["has_target_distress_item"]
        & manifest["filing_year"].notna()
        & manifest["accession_number"].notna()
    ].copy()
    sampled = []
    per_group = max(1, max_rows // max(1, target.groupby(["filing_year", "item_bucket"]).ngroups))
    for _, group in target.groupby(["filing_year", "item_bucket"], sort=True):
        sampled.append(stable_sample(group, per_group, seed))
    if not sampled:
        return pd.DataFrame(columns=list(manifest.columns))
    sample = pd.concat(sampled, ignore_index=True)
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=seed).copy()
    sample = sample.sort_values(["filing_year", "item_bucket", "cik", "accession_number"], kind="mergesort")
    sample["sample_role"] = "target_cached_reference_sample"
    return sample


def build_strata(manifest: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    strata = (
        manifest.groupby(["filing_year", "item_bucket"], dropna=False)
        .agg(
            population_rows=("accession_number", "size"),
            local_text_files=("local_text_exists", "sum"),
            target_rows=("has_target_distress_item", "sum"),
        )
        .reset_index()
    )
    if sample.empty:
        strata["sample_rows"] = 0
    else:
        sample_counts = (
            sample.groupby(["filing_year", "item_bucket"], dropna=False)
            .size()
            .rename("sample_rows")
            .reset_index()
        )
        strata = strata.merge(sample_counts, on=["filing_year", "item_bucket"], how="left")
        strata["sample_rows"] = strata["sample_rows"].fillna(0).astype(int)
    strata["target_rows"] = strata["target_rows"].astype(int)
    strata["local_text_files"] = strata["local_text_files"].astype(int)
    strata["sample_rate"] = strata["sample_rows"] / strata["population_rows"]
    return strata.sort_values(["filing_year", "item_bucket"], kind="mergesort")


def observed_average_cached_bytes(manifest: pd.DataFrame) -> float:
    cached = manifest[manifest["local_text_bytes"].fillna(0).astype(float).gt(0)]
    return float(cached["local_text_bytes"].mean()) if not cached.empty else 0.0


def prepare_sample_output(sample: pd.DataFrame, avg_bytes: float) -> pd.DataFrame:
    if sample.empty:
        return sample
    sample = sample.copy()
    sample = sample.reset_index(drop=True)
    sample.insert(0, "sample_id", [f"SEC8K-TEXT-{idx + 1:04d}" for idx in range(len(sample))])
    sample["sample_stratum"] = sample["filing_year"].astype("Int64").astype(str) + "::" + sample["item_bucket"].astype(str)
    sample["text_status_for_sampling"] = sample["local_text_exists"].map(lambda ok: "already_cached" if ok else "planned_missing_text")
    sample["estimated_text_bytes"] = sample["local_text_bytes"].where(sample["local_text_exists"], avg_bytes).round(0).astype(int)
    keep = [
        "sample_id",
        "sample_role",
        "sample_stratum",
        "filing_year",
        "item_bucket",
        "text_status_for_sampling",
        "cik",
        "entity_name",
        "accession_number",
        "filing_date",
        "form",
        "items",
        "primary_document",
        "archive_txt_url",
        "local_text_path",
        "local_text_exists",
        "local_text_bytes",
        "estimated_text_bytes",
        "has_target_distress_item",
        "has_item_1_03",
        "has_item_2_04",
        "has_item_3_01",
    ]
    return sample[[col for col in keep if col in sample.columns]]


def summarize(manifest: pd.DataFrame, sample: pd.DataFrame, strata: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    avg_bytes = observed_average_cached_bytes(manifest)
    non_target_rows = int((~manifest["has_target_distress_item"]).sum())
    non_target_missing = int(((~manifest["has_target_distress_item"]) & (~manifest["local_text_exists"])).sum())
    non_target_sample = sample[sample["sample_role"].eq("non_target_text_scope_sample")] if not sample.empty else sample
    target_reference = sample[sample["sample_role"].eq("target_cached_reference_sample")] if not sample.empty else sample
    planned_missing = sample[sample["text_status_for_sampling"].eq("planned_missing_text")] if not sample.empty else sample
    return {
        "status": "sample_design_created",
        "manifest_path": str(args.manifest_path),
        "manifest_rows": int(len(manifest)),
        "path_resolution_mode": "canonical_cik_accession_text_root",
        "manifest_local_path_redacted_rows": int(manifest.get("manifest_local_path_redacted", pd.Series(dtype=bool)).sum()),
        "canonical_text_path_rows": int(manifest["local_text_path"].fillna("").astype(str).ne("").sum()),
        "firms": int(manifest["cik"].nunique()) if "cik" in manifest else None,
        "years": [int(manifest["filing_year"].min()), int(manifest["filing_year"].max())],
        "non_target_population_rows": non_target_rows,
        "non_target_missing_text_rows": non_target_missing,
        "target_cached_reference_population_rows": int(manifest["has_target_distress_item"].sum()),
        "sample_rows": int(len(sample)),
        "non_target_sample_rows": int(len(non_target_sample)),
        "target_cached_reference_rows": int(len(target_reference)),
        "sample_strata_with_rows": int((strata["sample_rows"] > 0).sum()),
        "population_strata": int(len(strata)),
        "per_year_bucket": int(args.per_year_bucket),
        "max_non_target_rows": int(args.max_non_target_rows),
        "max_target_reference_rows": int(args.max_target_reference_rows),
        "random_seed": int(args.random_seed),
        "observed_avg_cached_text_bytes": avg_bytes,
        "estimated_planned_missing_sample_bytes": int(planned_missing["estimated_text_bytes"].sum()) if not planned_missing.empty else 0,
        "estimated_planned_missing_sample_mb": round(float(planned_missing["estimated_text_bytes"].sum()) / 1_000_000, 3) if not planned_missing.empty else 0.0,
        "guardrail": "Sampling design only. It creates a bounded non-target all-8-K filing-body scope plan and performs no SEC downloads.",
    }


def write_markdown(summary: dict[str, Any], strata: pd.DataFrame, sample: pd.DataFrame, path: Path) -> None:
    lines = [
        "# SEC All-8-K Filing-Body Text Sample Design",
        "",
        "This artifact defines a bounded stratified sample for non-target SEC 8-K/8-K/A filing bodies. It is a scope-control and future-download plan, not a completed text cache.",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Status | `{summary['status']}` |",
        f"| Manifest rows | {summary['manifest_rows']} |",
        f"| Path resolution mode | `{summary['path_resolution_mode']}` |",
        f"| Redacted manifest path rows | {summary['manifest_local_path_redacted_rows']} |",
        f"| Canonical text path rows | {summary['canonical_text_path_rows']} |",
        f"| Firms | {summary['firms']} |",
        f"| Years | {summary['years'][0]}-{summary['years'][1]} |",
        f"| Non-target population rows | {summary['non_target_population_rows']} |",
        f"| Non-target missing text rows | {summary['non_target_missing_text_rows']} |",
        f"| Target cached reference population rows | {summary['target_cached_reference_population_rows']} |",
        f"| Non-target sample rows | {summary['non_target_sample_rows']} |",
        f"| Target cached reference rows | {summary['target_cached_reference_rows']} |",
        f"| Population strata | {summary['population_strata']} |",
        f"| Sampled strata | {summary['sample_strata_with_rows']} |",
        f"| Per year-bucket cap | {summary['per_year_bucket']} |",
        f"| Estimated missing sample text size | {summary['estimated_planned_missing_sample_mb']} MB |",
        "",
        "## Sampling Design",
        "",
        "- Universe: local SEC 8-K/8-K/A manifest rows for 2015-2024.",
        "- Primary sample: non-target filings outside Items 1.03/2.04/3.01, stratified by filing year and item bucket.",
        "- Reference sample: already cached target-item filings, stratified by filing year and target distress item.",
        "- Item buckets prioritize common 8-K disclosure groups: earnings 2.02, Reg FD 7.01, other events 8.01, management 5.02, agreements/financing 1.01/2.03, auditor 4.01, securities 3.x, and other non-target items.",
        "- Random seed is fixed for reproducibility.",
        "",
        "## Strata Preview",
        "",
        "| Year | Bucket | Population rows | Local text files | Target rows | Sample rows | Sample rate |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    preview = strata[strata["sample_rows"].gt(0)].head(30)
    for row in preview.itertuples(index=False):
        lines.append(
            f"| {int(row.filing_year) if pd.notna(row.filing_year) else ''} | {row.item_bucket} | {row.population_rows} | {row.local_text_files} | {row.target_rows} | {row.sample_rows} | {row.sample_rate:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Claim Controls",
            "",
            "- This sample design does not allow an all-8-K filing-body text validation claim by itself.",
            "- It does allow a bounded future non-target filing-body parsing/feature-sensitivity check if the planned sample is downloaded with an identifying SEC user agent.",
            "- Until that download and model rerun exist, SEC text claims remain limited to target-item filing bodies plus submissions metadata text.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--text-root", default=str(DEFAULT_TEXT_ROOT))
    parser.add_argument("--per-year-bucket", type=int, default=12)
    parser.add_argument("--max-non-target-rows", type=int, default=1000)
    parser.add_argument("--max-target-reference-rows", type=int, default=120)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    results_dir = Path(args.results_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest_path, low_memory=False)
    manifest = refresh_manifest_text_paths(manifest, Path(args.text_root))
    manifest = add_sampling_columns(manifest)
    avg_bytes = observed_average_cached_bytes(manifest)
    non_target_sample = build_non_target_sample(manifest, args.per_year_bucket, args.max_non_target_rows, args.random_seed)
    target_reference = build_target_reference_sample(manifest, args.max_target_reference_rows, args.random_seed)
    sample = pd.concat([non_target_sample, target_reference], ignore_index=True, sort=False)
    sample = prepare_sample_output(sample, avg_bytes)
    strata = build_strata(manifest, sample)
    summary = summarize(manifest, sample, strata, args)

    sample_csv = out_root / "sec_all8k_text_sample_plan.csv"
    strata_csv = out_root / "sec_all8k_text_sample_strata.csv"
    sample.to_csv(sample_csv, index=False, encoding="utf-8-sig")
    strata.to_csv(strata_csv, index=False, encoding="utf-8-sig")
    summary["sample_plan_csv"] = str(sample_csv)
    summary["strata_csv"] = str(strata_csv)

    (results_dir / "sec_all8k_text_sample_design.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    write_markdown(summary, strata, sample, results_dir / "sec_all8k_text_sample_design.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
