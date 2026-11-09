import json
import tempfile
import unittest
from pathlib import Path

from src.data_validation import validate_existing_splits, validate_records


class ValidateRecordsTests(unittest.TestCase):
    def test_removes_empty_and_normalized_duplicate_records(self):
        rows = [
            {"instruction": "  Say   hello ", "input": "", "output": "Hi"},
            {"instruction": "Say hello", "input": "", "output": "Hi"},
            {"instruction": "", "output": "missing instruction"},
            {"instruction": "broken", "output": None},
            None,
        ]

        valid, summary = validate_records(rows)

        self.assertEqual(valid, [{"instruction": "Say hello", "input": "", "output": "Hi"}])
        self.assertEqual(summary["input_records"], 5)
        self.assertEqual(summary["valid_unique_records"], 1)
        self.assertEqual(summary["duplicates_removed"], 1)
        self.assertEqual(summary["rejected_empty"], 2)
        self.assertEqual(summary["rejected_malformed"], 1)

    def test_preserves_distinct_input_output_pairs(self):
        rows = [
            {"instruction": "Translate", "input": "hello", "output": "hola"},
            {"instruction": "Translate", "input": "goodbye", "output": "adios"},
        ]

        valid, _ = validate_records(rows)

        self.assertEqual(len(valid), 2)

    def test_existing_split_validation_is_read_only_and_detects_overlap(self):
        record = {"instruction": "say hello", "input": "", "output": "hello"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "train.jsonl"
            test_path = root / "test.jsonl"
            train_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            test_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            original_train = train_path.read_bytes()
            original_test = test_path.read_bytes()

            summary = validate_existing_splits(train_path, test_path)

            self.assertEqual(summary["status"], "FAIL")
            self.assertEqual(summary["duplicates_removed"], 1)
            self.assertEqual(summary["train_records"], 1)
            self.assertEqual(summary["test_records"], 1)
            self.assertEqual(train_path.read_bytes(), original_train)
            self.assertEqual(test_path.read_bytes(), original_test)

    def test_existing_split_validation_rejects_malformed_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            train_path = Path(temporary) / "train.jsonl"
            test_path = Path(temporary) / "test.jsonl"
            train_path.write_text("{invalid}\n", encoding="utf-8")
            test_path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1"):
                validate_existing_splits(train_path, test_path)


if __name__ == "__main__":
    unittest.main()
