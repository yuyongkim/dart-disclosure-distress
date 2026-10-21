from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("Paper 4")
DEFAULT_FACTS_ROOT = Path.home() / "Desktop" / "M&A" / "data" / "raw" / "sec"


def normalize_cik(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else ""


def file_cik(path: Path) -> str:
    stem = path.stem
    if stem.startswith("CIK") and stem.endswith("_companyfacts"):
        return normalize_cik(stem.removeprefix("CIK").removesuffix("_companyfacts"))
    return normalize_cik(stem)


def inventory_companyfacts(facts_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if facts_root.exists():
        for path in sorted(facts_root.glob("CIK*_companyfacts.json")):
            stat = path.stat()
            rows.append(
                {
                    "cik": file_cik(path),
                    "path": str(path),
                    "bytes": int(stat.st_size),
                    "modified_utc": pd.to_datetime(stat.st_mtime, unit="s", utc=True).isoformat(),
                }
            )
    return pd.DataFrame(rows)


def validation_ciks(panel_path: Path) -> pd.DataFrame:
    if not panel_path.exists():
        return pd.DataFrame(columns=["cik"])
    panel = pd.read_parquet(panel_path, columns=["firm_id", "firm_name"])
    panel["cik"] = panel["firm_id"].map(normalize_cik)
    panel = panel[panel["cik"].ne("")]
    return (
        panel[["cik", "firm_name"]]
        .drop_duplicates()
        .sort_values(["cik", "firm_name"])
        .reset_index(drop=True)
    )


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# SEC Companyfacts Availability",
        "",
        "This file documents whether local SEC companyfacts files are available for the local SEC validation frame without parsing the full companyfacts payloads.",
        "",
        "## Coverage",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Companyfacts files in local SEC cache | {summary['inventory_files']} |",
        f"| Total companyfacts cache size, GB | {summary['inventory_total_gb']:.2f} |",
        f"| Validation CIKs requested | {summary['validation_ciks']} |",
        f"| Validation CIKs with companyfacts file | {summary['validation_ciks_with_file']} |",
        f"| Validation file coverage rate | {summary['validation_file_coverage_rate']:.4f} |",
        f"| Missing validation CIKs | {summary['validation_missing_files']} |",
        f"| Largest companyfacts file, MB | {summary['inventory_max_mb']:.2f} |",
        "",
        "## Interpretation",
        "",
        "- The local SEC cache has local companyfacts files that support the separate SEC financial baseline step.",
        "- Full companyfacts parsing is intentionally separated into the heavy `sec_desktop_financial_baseline.py` step because the cache is large.",
        "- This availability audit itself does not create financial ratios; use `results/sec_desktop_financial_baseline_summary.md` for the completed companyfacts financial baseline.",
        "- The current v1 manuscript should continue to describe SEC as v2/support evidence until full all-8-K filing-body coverage and final U.S. validation scope are completed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts-root", default=str(DEFAULT_FACTS_ROOT))
    parser.add_argument("--panel-path", default=str(ROOT / "data" / "model_sec_desktop" / "sec_desktop_validation_panel.parquet"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "model_sec_desktop"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    facts_root = Path(args.facts_root)
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    inventory = inventory_companyfacts(facts_root)
    validation = validation_ciks(Path(args.panel_path))
    if inventory.empty:
        inventory = pd.DataFrame(columns=["cik", "path", "bytes", "modified_utc"])
    if validation.empty:
        coverage = pd.DataFrame(columns=["cik", "firm_name", "path", "bytes", "has_companyfacts_file"])
    else:
        coverage = validation.merge(inventory[["cik", "path", "bytes"]], on="cik", how="left")
        coverage["has_companyfacts_file"] = coverage["path"].notna()

    inventory.to_csv(out_dir / "sec_companyfacts_file_inventory.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(out_dir / "sec_companyfacts_validation_coverage.csv", index=False, encoding="utf-8-sig")

    validation_count = int(len(coverage))
    with_file = int(coverage["has_companyfacts_file"].sum()) if "has_companyfacts_file" in coverage else 0
    total_bytes = int(inventory["bytes"].sum()) if "bytes" in inventory and not inventory.empty else 0
    max_bytes = int(inventory["bytes"].max()) if "bytes" in inventory and not inventory.empty else 0
    summary = {
        "status": "created",
        "facts_root": "local_sec_companyfacts_cache",
        "inventory_files": int(len(inventory)),
        "inventory_total_bytes": total_bytes,
        "inventory_total_gb": total_bytes / (1024**3),
        "inventory_max_bytes": max_bytes,
        "inventory_max_mb": max_bytes / (1024**2),
        "validation_ciks": validation_count,
        "validation_ciks_with_file": with_file,
        "validation_missing_files": validation_count - with_file,
        "validation_file_coverage_rate": float(with_file / validation_count) if validation_count else 0.0,
        "outputs": {
            "inventory_csv": str(out_dir / "sec_companyfacts_file_inventory.csv"),
            "coverage_csv": str(out_dir / "sec_companyfacts_validation_coverage.csv"),
            "summary_md": str(results_dir / "sec_companyfacts_availability.md"),
            "summary_json": str(results_dir / "sec_companyfacts_availability.json"),
        },
        "guardrail": "Availability audit only; full SEC financial baseline remains a heavy step.",
    }
    (results_dir / "sec_companyfacts_availability.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(summary, results_dir / "sec_companyfacts_availability.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
