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


def _read_jsonl(path: Path) -> list[Any]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
    return rows


def validate_existing_splits(
    train_path: Path, test_path: Path
) -> dict[str, Any]:
    train_rows = _read_jsonl(train_path)
    test_rows = _read_jsonl(test_path)
    train, train_summary = validate_records(train_rows)
    test, test_summary = validate_records(test_rows)
    combined, summary = validate_records(train_rows + test_rows)
    summary.update(
        {
            "status": (
                "PASS"
                if len(combined) >= 1000
                and len(train) > 0
                and len(test) > 0
                and summary["rejected_empty"] == 0
                and summary["rejected_malformed"] == 0
                and summary["duplicates_removed"] == 0
                else "FAIL"
            ),
            "validation_scope": "existing train/test JSONL splits; source dataset not re-read",
            "train_records": len(train),
            "test_records": len(test),
            "train_rejected_empty": train_summary["rejected_empty"],
            "train_rejected_malformed": train_summary["rejected_malformed"],
            "train_duplicates_removed": train_summary["duplicates_removed"],
            "test_rejected_empty": test_summary["rejected_empty"],
            "test_rejected_malformed": test_summary["rejected_malformed"],
            "test_duplicates_removed": test_summary["duplicates_removed"],
        }
    )
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = validate_existing_splits(
        root / "data" / "train.jsonl",
        root / "data" / "test.jsonl",
    )
    destination = save_validation_report(summary)
    if summary["status"] != "PASS":
        raise RuntimeError(f"Existing data splits failed validation; see {destination}.")
    print(f"Validated existing data splits; report: {destination}")


if __name__ == "__main__":
    main()
