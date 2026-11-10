import json
import random

from src.common import DATASET_ID, DATA_DIR, SEED, ensure_directories, get_logger
from src.data_validation import (
    find_near_duplicate_pairs,
    save_validation_report,
    validate_records,
)


def shuffled_subset(dataset, limit: int, seed: int):
    if limit < 1:
        raise ValueError("limit must be greater than zero.")
    shuffled = dataset.shuffle(seed=seed)
    return shuffled.select(range(min(len(shuffled), limit)))


def main() -> None:
    from datasets import load_dataset

    logger = get_logger("data_preparation", "data_preparation.log")
    ensure_directories()
    max_examples = 2000
    dataset = load_dataset(DATASET_ID, split="train")
    source_limit = min(len(dataset), max_examples * 2)
    rows = iter(shuffled_subset(dataset, source_limit, SEED))
    records, summary = validate_records(rows)
    records = records[:max_examples]
    summary["unique_records_found"] = summary["valid_unique_records"]
    summary["valid_unique_records"] = len(records)
    summary.update(
        {
            "dataset_id": DATASET_ID,
            "seed": SEED,
            "source_scan_limit": source_limit,
            "status": "PASS" if len(records) >= 1000 else "FAIL",
        }
    )
    path = save_validation_report(summary)
    if len(records) < 1000:
        raise RuntimeError(
            f"Only {len(records)} valid unique examples; expected at least 1000. "
            f"Inspection details saved to {path}."
        )

    random.Random(SEED).shuffle(records)
    test_count = max(1, round(len(records) * 0.1))
    test_records, train_records = records[:test_count], records[test_count:]
    for name, split in (("train", train_records), ("test", test_records)):
        with (DATA_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as output:
            for record in split:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

    near_duplicates = find_near_duplicate_pairs(train_records, test_records)
    summary.update(
        {
            "train_records": len(train_records),
            "test_records": len(test_records),
            "near_duplicate_cross_split_count": len(near_duplicates),
            "near_duplicate_cross_split_examples": near_duplicates[:50],
            "status": "FAIL" if near_duplicates else "PASS",
        }
    )
    save_validation_report(summary)
    if near_duplicates:
        raise RuntimeError(
            f"Found {len(near_duplicates)} near-duplicate train/test pairs; "
            f"inspect {path} before training."
        )
    logger.info("Prepared %d train and %d test rows; report: %s", len(train_records), len(test_records), path)


if __name__ == "__main__":
    main()
