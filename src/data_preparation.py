import json
import os
import random

from datasets import load_dataset

from src.common import DATASET_ID, DATA_DIR, SEED, ensure_directories, get_logger
from src.data_validation import save_validation_report, validate_records


def main() -> None:
    logger = get_logger("data_preparation", "data_preparation.log")
    ensure_directories()
    max_examples = 2000
    dataset = load_dataset(DATASET_ID, split="train", token=os.getenv("HF_TOKEN") or None)
    source_limit = min(len(dataset), max_examples * 2)
    rows = (dataset[index] for index in range(source_limit))
    records, summary = validate_records(rows)
    records = records[:max_examples]
    summary["unique_records_found"] = summary["valid_unique_records"]
    summary["valid_unique_records"] = len(records)
    summary.update(
        {
            "dataset_id": DATASET_ID,
            "seed": SEED,
            "source_scan_limit": source_limit,
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

    summary.update({"train_records": len(train_records), "test_records": len(test_records)})
    save_validation_report(summary)
    logger.info("Prepared %d train and %d test rows; report: %s", len(train_records), len(test_records), path)


if __name__ == "__main__":
    main()
