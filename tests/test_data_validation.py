import json
import tempfile
import unittest
from pathlib import Path

from src.data_validation import (
    find_near_duplicate_pairs,
    split_file_status,
    validate_existing_splits,
    validate_records,
)
from src.training_data import truncate_training_sequence


class ValidateRecordsTests(unittest.TestCase):
    def test_split_file_status_distinguishes_missing_from_present(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"

            self.assertEqual(split_file_status(path), "MISSING")
            path.write_text("", encoding="utf-8")
            self.assertEqual(split_file_status(path), "PRESENT")

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

    def test_preserves_newlines_while_normalizing_horizontal_whitespace(self):
        valid, _ = validate_records(
            [{"instruction": "Line  one\n  Line   two", "output": "a\r\n\tb"}]
        )

        self.assertEqual(valid[0]["instruction"], "Line one\nLine two")
        self.assertEqual(valid[0]["output"], "a\nb")

    def test_detects_near_duplicate_across_train_and_test(self):
        train = [
            {
                "instruction": "Summarize this report",
                "input": "",
                "output": "The project delivered a reliable and carefully tested release for users.",
            }
        ]
        test = [
            {
                "instruction": "Summarize this report",
                "input": "",
                "output": "The project delivered a reliable and carefully tested release for users today.",
            }
        ]

        matches = find_near_duplicate_pairs(train, test)

        self.assertEqual(len(matches), 1)
        self.assertGreaterEqual(matches[0]["jaccard_similarity"], 0.9)

    def test_truncation_preserves_assistant_eos_as_a_supervised_token(self):
        input_ids, labels = truncate_training_sequence(
            full_ids=[1, 2, 3, 4, 5, 99],
            prompt_ids=[1, 2],
            max_length=4,
            eos_token_id=99,
        )

        self.assertEqual(input_ids, [1, 2, 3, 99])
        self.assertEqual(labels, [-100, -100, 3, 99])

    def test_truncation_rejects_examples_without_supervised_assistant_tokens(self):
        with self.assertRaisesRegex(ValueError, "no supervised assistant tokens"):
            truncate_training_sequence(
                full_ids=[1, 2, 3, 99],
                prompt_ids=[1, 2, 3],
                max_length=3,
                eos_token_id=99,
            )

    def test_truncation_rejects_misaligned_prompt_tokens(self):
        with self.assertRaisesRegex(ValueError, "not a prefix"):
            truncate_training_sequence(
                full_ids=[1, 2, 3, 99],
                prompt_ids=[1, 3],
                max_length=4,
                eos_token_id=99,
            )

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
