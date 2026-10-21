from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT.parent)
ROOT = Path("Paper 4")
PACKAGE_DIR = ROOT / "submission_package"
RECEIPT_PATH = PACKAGE_DIR / "email_handoff_receipt.json"
LATEST_AUDIT_PATH = PACKAGE_DIR / "paper4_core_package_latest_audit.json"
LATEST_POINTER_PATH = PACKAGE_DIR / "LATEST_PACKAGE.md"
OUT_JSON = PACKAGE_DIR / "email_handoff_receipt_audit.json"
OUT_MD = PACKAGE_DIR / "email_handoff_receipt_audit.md"
RECEIPT_TEXT_FIELDS = ("gmail_subject", "evidence_source")
MOJIBAKE_MARKERS = ("\ufffd", "\u5360")
QUESTION_MARK_MOJIBAKE_RE = re.compile(r"\?{2,}")
GMAIL_ID_RE = re.compile(r"^[0-9a-f]+$")
FULL_ISO_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?$"
)
EXPECTED_GMAIL_SUBJECT = "Paper 4 전체 제출 패키지 및 남은 작업 정리"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def receipt_text_integrity_failures(receipt: dict[str, Any]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for field in RECEIPT_TEXT_FIELDS:
        value = str(receipt.get(field, ""))
        markers = [marker for marker in MOJIBAKE_MARKERS if marker in value]
        if QUESTION_MARK_MOJIBAKE_RE.search(value):
            markers.append("repeated_question_marks")
        if markers:
            failures.append({"field": field, "markers": ", ".join(markers)})
    return failures


def is_iso_like_timestamp(value: object) -> bool:
    text = str(value)
    if not FULL_ISO_TIMESTAMP_RE.fullmatch(text):
        return False
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def latest_pointer_table_value(pointer_text: str, label: str) -> str:
    for line in pointer_text.splitlines():
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] == label:
            return cells[1]
    return ""


def latest_pointer_has_table_value(pointer_text: str, label: str, expected: object) -> bool:
    expected_text = str(expected)
    if not expected_text:
        return False
    return latest_pointer_table_value(pointer_text, label) == expected_text


def gmail_display_url_matches_message_id(display_url: str, message_id: str) -> bool:
    return bool(message_id) and display_url == f"https://mail.google.com/mail/#all/{message_id}"


def is_gmail_hex_id(value: object) -> bool:
    text = str(value)
    return len(text) >= 12 and bool(GMAIL_ID_RE.fullmatch(text))


def evidence_source_matches_expected_context(evidence_source: str, message_id: str) -> bool:
    required_fragments = (
        "Gmail read_email_thread result",
        "local latest package audit",
        "latest pointer SHA/size validation",
    )
    return bool(message_id) and message_id in evidence_source and all(
        fragment in evidence_source for fragment in required_fragments
    )


def build_audit() -> dict[str, Any]:
    receipt = read_json(RECEIPT_PATH)
    package = read_json(LATEST_AUDIT_PATH)
    pointer_text = LATEST_POINTER_PATH.read_text(encoding="utf-8", errors="ignore") if LATEST_POINTER_PATH.exists() else ""
    text_integrity_failures = receipt_text_integrity_failures(receipt)

    latest_zip_path = package.get("zip_path", "submission_package/paper4_core_package_latest.zip")
    latest_zip_name = Path(latest_zip_path).name
    message_id = str(receipt.get("gmail_message_id", ""))
    display_url = str(receipt.get("gmail_display_url", ""))
    sha_limit = str(receipt.get("gmail_attachment_content_sha256_verification_limit", ""))
    evidence_source = str(receipt.get("evidence_source", ""))
    latest_pointer_generated_at = latest_pointer_table_value(pointer_text, "Generated at UTC")
    checks = [
        ("receipt_exists", bool(receipt), rel(RECEIPT_PATH)),
        ("receipt_text_integrity_clean", not text_integrity_failures, text_integrity_failures),
        ("receipt_status_sent_verified", receipt.get("status") == "sent_verified", receipt.get("status")),
        ("recipient_expected", receipt.get("recipient") == "recipient@example.com", receipt.get("recipient")),
        ("gmail_message_id_hex_like", is_gmail_hex_id(message_id), message_id),
        ("gmail_thread_id_hex_like", is_gmail_hex_id(receipt.get("gmail_thread_id", "")), receipt.get("gmail_thread_id", "")),
        (
            "gmail_display_url_matches_message_id",
            gmail_display_url_matches_message_id(display_url, message_id),
            display_url,
        ),
        (
            "gmail_subject_matches_expected",
            receipt.get("gmail_subject", "") == EXPECTED_GMAIL_SUBJECT,
            receipt.get("gmail_subject", ""),
        ),
        (
            "gmail_email_ts_iso_like",
            is_iso_like_timestamp(receipt.get("gmail_email_ts")),
            receipt.get("gmail_email_ts", ""),
        ),
        ("attachment_filename_latest", receipt.get("attachment_filename") == latest_zip_name, receipt.get("attachment_filename")),
        ("package_audit_exists", bool(package), rel(LATEST_AUDIT_PATH)),
        ("package_audit_pass", package.get("status") == "pass", package.get("status")),
        ("receipt_local_zip_path_matches", receipt.get("local_zip_path") == latest_zip_path, receipt.get("local_zip_path")),
        (
            "receipt_local_audit_path_matches",
            receipt.get("local_audit_path") == package.get("audit_path"),
            receipt.get("local_audit_path"),
        ),
        (
            "gmail_attachment_size_matches_latest",
            receipt.get("gmail_attachment_size_bytes") == package.get("zip_size_bytes"),
            f"{receipt.get('gmail_attachment_size_bytes')}/{package.get('zip_size_bytes')}",
        ),
        (
            "gmail_attachment_content_sha_limit_documented",
            receipt.get("gmail_attachment_content_sha256_verified") is False
            and "unsupported_attachment_type" in sha_limit,
            sha_limit,
        ),
        (
            "gmail_attachment_content_sha_limit_mentions_message_id",
            bool(message_id) and message_id in sha_limit,
            sha_limit,
        ),
        (
            "local_package_size_at_send_matches_latest",
            receipt.get("local_package_size_bytes_at_send") == package.get("zip_size_bytes"),
            f"{receipt.get('local_package_size_bytes_at_send')}/{package.get('zip_size_bytes')}",
        ),
        (
            "gmail_body_sha_matches_latest",
            receipt.get("gmail_body_reported_sha256") == package.get("zip_sha256"),
            f"{receipt.get('gmail_body_reported_sha256')}/{package.get('zip_sha256')}",
        ),
        (
            "local_package_sha_at_send_matches_latest",
            receipt.get("local_package_sha256_at_send") == package.get("zip_sha256"),
            f"{receipt.get('local_package_sha256_at_send')}/{package.get('zip_sha256')}",
        ),
        (
            "evidence_source_documents_gmail_and_local_audit",
            evidence_source_matches_expected_context(evidence_source, message_id),
            evidence_source,
        ),
        (
            "latest_pointer_mentions_zip",
            latest_pointer_has_table_value(pointer_text, "Latest ZIP", latest_zip_path),
            latest_zip_path,
        ),
        (
            "latest_pointer_mentions_sha",
            latest_pointer_has_table_value(pointer_text, "ZIP SHA-256", package.get("zip_sha256", "")),
            package.get("zip_sha256", ""),
        ),
        (
            "latest_pointer_mentions_size",
            latest_pointer_has_table_value(pointer_text, "ZIP size bytes", package.get("zip_size_bytes", "")),
            package.get("zip_size_bytes", ""),
        ),
        (
            "latest_pointer_generated_at_utc_iso_like",
            is_iso_like_timestamp(latest_pointer_generated_at) and latest_pointer_generated_at.endswith("+00:00"),
            latest_pointer_generated_at,
        ),
    ]
    failures = [
        {"check": name, "observed": observed}
        for name, ok, observed in checks
        if not ok
    ]
    return {
        "status": "pass" if not failures else "fail",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "receipt_path": rel(RECEIPT_PATH),
        "latest_package_audit_path": rel(LATEST_AUDIT_PATH),
        "latest_pointer_path": rel(LATEST_POINTER_PATH),
        "latest_pointer_generated_at_utc": latest_pointer_generated_at,
        "recipient": receipt.get("recipient", ""),
        "gmail_message_id": receipt.get("gmail_message_id", ""),
        "gmail_thread_id": receipt.get("gmail_thread_id", ""),
        "gmail_display_url": receipt.get("gmail_display_url", ""),
        "gmail_email_ts": receipt.get("gmail_email_ts", ""),
        "gmail_subject": receipt.get("gmail_subject", ""),
        "attachment_filename": receipt.get("attachment_filename", ""),
        "gmail_attachment_content_sha256_verified": receipt.get("gmail_attachment_content_sha256_verified"),
        "gmail_attachment_content_sha256_verification_limit": receipt.get(
            "gmail_attachment_content_sha256_verification_limit", ""
        ),
        "latest_zip_sha256": package.get("zip_sha256", ""),
        "latest_zip_size_bytes": package.get("zip_size_bytes"),
        "receipt_text_integrity_failures": text_integrity_failures,
        "check_count": len(checks),
        "failure_count": len(failures),
        "checks": [
            {"check": name, "ok": ok, "observed": observed}
            for name, ok, observed in checks
        ],
        "failures": failures,
        "evidence_limit": (
            "This local audit verifies the recorded Gmail handoff receipt against the current "
            "latest package audit and pointer. It does not hash Gmail's stored ZIP attachment content "
            "because the current Gmail connector cannot ingest application/zip attachments; it verifies "
            "recipient/thread/message metadata, attachment filename and size, body-reported SHA, and the "
            "local latest package audit SHA, while the latest pointer records the same ZIP SHA, size, "
            "and an ISO-like UTC regeneration timestamp. The pointer timestamp is not treated as the "
            "Gmail send time; handoff freshness is locked by matching SHA/size evidence. "
            "Refresh the receipt after any resend or package SHA change."
        ),
    }


def write_markdown(audit: dict[str, Any]) -> None:
    lines = [
        "# Email Handoff Receipt Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Checked at | `{audit['checked_at']}` |",
        f"| Recipient | `{audit['recipient']}` |",
        f"| Gmail message ID | `{audit['gmail_message_id']}` |",
        f"| Gmail thread ID | `{audit['gmail_thread_id']}` |",
        f"| Gmail display URL | `{audit['gmail_display_url']}` |",
        f"| Gmail timestamp | `{audit['gmail_email_ts']}` |",
        f"| Attachment | `{audit['attachment_filename']}` |",
        f"| Gmail attachment content SHA verified | `{audit['gmail_attachment_content_sha256_verified']}` |",
        f"| Latest ZIP SHA-256 | `{audit['latest_zip_sha256']}` |",
        f"| Latest ZIP size bytes | `{audit['latest_zip_size_bytes']}` |",
        f"| Latest pointer generated at UTC | `{audit['latest_pointer_generated_at_utc']}` |",
        f"| Checks | {audit['check_count']} |",
        f"| Failures | {audit['failure_count']} |",
        "",
        "## Failures",
        "",
    ]
    if audit["failures"]:
        lines.extend(["| Check | Observed |", "|---|---|"])
        for row in audit["failures"]:
            lines.append(f"| `{row['check']}` | `{row['observed']}` |")
    else:
        lines.append("No email handoff receipt audit failures.")
    lines.extend(["", "## Evidence Limit", "", audit["evidence_limit"], ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    audit = build_audit()
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(audit)
    print(json.dumps({"status": audit["status"], "failure_count": audit["failure_count"]}, indent=2))
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
