import random
import unittest

from src.data_preparation import shuffled_subset


class ShuffledSubsetTests(unittest.TestCase):
    def test_selects_deterministic_rows_across_the_source_dataset(self):
        class Dataset:
            def __init__(self, rows):
                self.rows = rows

            def __len__(self):
                return len(self.rows)

            def shuffle(self, seed):
                rows = list(self.rows)
                random.Random(seed).shuffle(rows)
                return Dataset(rows)

            def select(self, indices):
                return Dataset([self.rows[index] for index in indices])

            def __iter__(self):
                return iter(self.rows)

        rows = list(range(20))

        first = list(shuffled_subset(Dataset(rows), 8, 42))
        repeat = list(shuffled_subset(Dataset(rows), 8, 42))

        self.assertEqual(first, repeat)
        self.assertEqual(len(first), 8)
        self.assertNotEqual(first, rows[:8])

    def test_rejects_nonpositive_sample_limit(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            shuffled_subset([], 0, 42)


if __name__ == "__main__":
    unittest.main()
