import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.common import REPORT_DIR, save_json


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def validate_records(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    accepted: list[dict[str, str]] = []
    seen: set[str] = set()
    rejected_empty = 0
    rejected_malformed = 0
    duplicates = 0
    input_count = 0

    for row in rows:
        input_count += 1
        if not isinstance(row, dict):
            rejected_malformed += 1
            continue
        instruction = _clean_text(row.get("instruction"))
        user_input = _clean_text(row.get("input", ""))
        output = _clean_text(row.get("output"))
        if not instruction or not output:
            rejected_empty += 1
            continue
        record = {"instruction": instruction, "input": user_input, "output": output}
        digest = hashlib.sha256(
            json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        accepted.append(record)

    return accepted, {
        "input_records": input_count,
        "valid_unique_records": len(accepted),
        "rejected_empty": rejected_empty,
        "rejected_malformed": rejected_malformed,
        "duplicates_removed": duplicates,
    }


def save_validation_report(summary: dict[str, Any], path: Path | None = None) -> Path:
    destination = path or REPORT_DIR / "data_validation.json"
    save_json(destination, summary)
    return destination
