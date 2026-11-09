import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.common import load_environment
from src.data_validation import validate_records


class EnvironmentTests(unittest.TestCase):
    def test_loads_values_from_env_file(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("SOUP_TASK_TEST_ENV=loaded\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SOUP_TASK_TEST_ENV", None)
                load_environment(env_file)
                self.assertEqual(os.environ["SOUP_TASK_TEST_ENV"], "loaded")


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


if __name__ == "__main__":
    unittest.main()
