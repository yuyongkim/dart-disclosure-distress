from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
DEFAULT_TEXT_ROOT = ROOT / "data" / "raw" / "sec_filing_text"
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_SCOPE_PLAN = DEFAULT_RESULTS_DIR / "sec_all8k_text_scope_plan.json"
DEFAULT_BATCH_STATUS = DEFAULT_RESULTS_DIR / "sec_all8k_text_batch_collection_status.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current if current.exists() else Path(".")


def gb(value: float | int) -> float:
    return round(float(value) / 1_000_000_000, 3)


def pct(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.2f}%"


def build_preflight(
    text_root: Path,
    scope_plan_path: Path,
    batch_status_path: Path,
    reserve_ratio: float,
    min_free_after_gb: float,
) -> dict[str, Any]:
    scope_plan = read_json(scope_plan_path)
    batch_status = read_json(batch_status_path)
    disk_path = existing_parent(text_root)
    disk = shutil.disk_usage(disk_path)

    estimated_missing_gb = float(scope_plan.get("estimated_missing_text_gb") or 0.0)
    required_free_gb = round((estimated_missing_gb * reserve_ratio) + min_free_after_gb, 3)
    free_gb = gb(disk.free)
    storage_ready = free_gb >= required_free_gb if estimated_missing_gb else True

    env_user_agent = os.environ.get("SEC_USER_AGENT") or os.environ.get("EDGAR_USER_AGENT") or ""
    env_user_agent_present = bool(env_user_agent.strip())
    missing_files = int(scope_plan.get("missing_text_files") or batch_status.get("missing_text_files") or 0)
    open_batches = int(batch_status.get("open_batches") or 0)
    scope_plan_ready = scope_plan.get("status") == "plan_created"
    batch_status_ready = batch_status.get("status") in {
        "status_only",
        "download_not_requested",
        "downloaded_partial",
        "downloaded_complete",
    }

    blockers: list[str] = []
    if not scope_plan_ready:
        blockers.append("scope_plan_not_ready")
    if not batch_status_ready:
        blockers.append("batch_status_not_ready")
    if missing_files and not storage_ready:
        blockers.append("insufficient_storage_for_full_scope")
    if missing_files and not env_user_agent_present:
        blockers.append("missing_env_sec_user_agent")

    full_scope_ready = not blockers and missing_files > 0
    if missing_files == 0 and scope_plan_ready and batch_status_ready:
        status = "full_scope_cache_complete"
    elif full_scope_ready:
        status = "ready_for_controlled_full_scope_batches"
    elif "insufficient_storage_for_full_scope" in blockers and "missing_env_sec_user_agent" in blockers:
        status = "blocked_storage_and_user_agent"
    elif "insufficient_storage_for_full_scope" in blockers:
        status = "blocked_insufficient_storage"
    elif "missing_env_sec_user_agent" in blockers:
        status = "blocked_missing_env_user_agent"
    else:
        status = "blocked_missing_preflight_inputs"

    return {
        "status": status,
        "text_root": str(text_root),
        "disk_path_checked": str(disk_path),
        "estimated_missing_text_gb": estimated_missing_gb,
        "reserve_ratio": float(reserve_ratio),
        "min_free_after_gb": float(min_free_after_gb),
        "required_free_gb": required_free_gb,
        "storage_ready_for_full_scope": storage_ready,
        "env_user_agent_present": env_user_agent_present,
        "scope_plan_status": scope_plan.get("status"),
        "batch_status": batch_status.get("status"),
        "manifest_rows": scope_plan.get("manifest_rows") or batch_status.get("manifest_rows"),
        "missing_text_files": missing_files,
        "open_batches": open_batches,
        "current_coverage_rate": batch_status.get("coverage_rate") or scope_plan.get("all8k_text_coverage_rate"),
        "target_distress_item_coverage_rate": batch_status.get("target_distress_item_coverage_rate")
        or scope_plan.get("target_distress_item_coverage_rate"),
        "download_log_downloaded_files": batch_status.get("download_log_downloaded_files"),
        "download_log_existing_files": batch_status.get("download_log_existing_files"),
        "full_scope_download_ready": full_scope_ready,
        "blockers": blockers,
        "guardrail": (
            "This preflight performs no downloads. It prevents treating the all-8-K full-text scope as executable "
            "unless the current disk has enough free space with reserve headroom and an explicit SEC_USER_AGENT or "
            "EDGAR_USER_AGENT is configured."
        ),
    }


def write_markdown(preflight: dict[str, Any], path: Path) -> None:
    lines = [
        "# SEC All-8-K Full-Scope Download Preflight",
        "",
        "This audit checks whether the current machine is ready to execute the remaining SEC text-cache batch plan. It performs no SEC requests.",
        "",
        "## Status",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Status | `{preflight['status']}` |",
        f"| Full-scope download ready | {preflight['full_scope_download_ready']} |",
        f"| Disk path checked | `{preflight['disk_path_checked']}` |",
        f"| Estimated missing text | {preflight['estimated_missing_text_gb']} GB |",
        f"| Required free space | {preflight['required_free_gb']} GB |",
        f"| Storage ready | {preflight['storage_ready_for_full_scope']} |",
        f"| Environment SEC user agent present | {preflight['env_user_agent_present']} |",
        f"| Manifest rows | {preflight['manifest_rows']} |",
        f"| Missing text files | {preflight['missing_text_files']} |",
        f"| Open batches | {preflight['open_batches']} |",
        f"| Current all-8-K coverage | {pct(preflight['current_coverage_rate'])} |",
        f"| Target distress-item coverage | {pct(preflight['target_distress_item_coverage_rate'])} |",
        f"| Historical downloaded/existing files | {preflight['download_log_downloaded_files']} / {preflight['download_log_existing_files']} |",
        "",
        "## Blockers",
        "",
    ]
    if preflight["blockers"]:
        for blocker in preflight["blockers"]:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("No full-scope download preflight blockers.")
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            preflight["guardrail"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-root", default=str(DEFAULT_TEXT_ROOT))
    parser.add_argument("--scope-plan", default=str(DEFAULT_SCOPE_PLAN))
    parser.add_argument("--batch-status", default=str(DEFAULT_BATCH_STATUS))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--reserve-ratio", type=float, default=1.15)
    parser.add_argument("--min-free-after-gb", type=float, default=10.0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    preflight = build_preflight(
        text_root=Path(args.text_root),
        scope_plan_path=Path(args.scope_plan),
        batch_status_path=Path(args.batch_status),
        reserve_ratio=args.reserve_ratio,
        min_free_after_gb=args.min_free_after_gb,
    )
    (results_dir / "sec_all8k_storage_preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(preflight, results_dir / "sec_all8k_storage_preflight.md")
    print(json.dumps(preflight, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
