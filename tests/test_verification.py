import math
import unittest

try:
    import torch
except ImportError:
    torch = None

from src.verification import (
    nll_metrics,
    token_logit_drift,
    token_nll_sums,
    validate_nf4_evidence,
)


@unittest.skipIf(torch is None, "PyTorch is not installed.")
class VerificationMetricTests(unittest.TestCase):
    def test_token_nll_uses_only_assistant_targets_and_weights_tokens(self):
        logits = torch.tensor(
            [[[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 0.0]]]
        )
        labels = torch.tensor([[-100, -100, 1, 0]])

        total, count = token_nll_sums(logits, labels, token_chunk_size=1)

        expected = 2 * math.log1p(math.exp(-2.0))
        self.assertEqual(count, 2)
        self.assertAlmostEqual(total, expected, places=6)

    def test_token_nll_rejects_unscorable_sequences(self):
        logits = torch.zeros((1, 3, 5))
        labels = torch.full((1, 3), -100)

        with self.assertRaisesRegex(ValueError, "no supervised assistant tokens"):
            token_nll_sums(logits, labels)

    def test_logit_drift_compares_all_supervised_positions(self):
        reference = torch.tensor(
            [[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]]
        )
        labels = torch.tensor([[-100, -100, 1, 0]])

        same = token_logit_drift(reference, reference, labels, token_chunk_size=1)
        candidate = reference.clone()
        candidate[0, 1, 1] = -1.0
        changed = token_logit_drift(reference, candidate, labels, token_chunk_size=1)

        self.assertEqual(same["assistant_token_count"], 2)
        self.assertEqual(same["top1_match_count"], 2)
        self.assertEqual(same["top1_agreement"], 1.0)
        self.assertAlmostEqual(same["mean_kl_reference_to_candidate"], 0.0, places=7)
        self.assertEqual(changed["top1_match_count"], 1)
        self.assertEqual(changed["top1_agreement"], 0.5)
        self.assertGreater(changed["mean_kl_reference_to_candidate"], 0.0)

    def test_logit_drift_rejects_unscorable_sequences(self):
        logits = torch.zeros((1, 3, 5))
        labels = torch.full((1, 3), -100)

        with self.assertRaisesRegex(ValueError, "no supervised assistant tokens"):
            token_logit_drift(logits, logits, labels)

    def test_perplexity_is_exponential_of_token_weighted_mean_nll(self):
        metrics = nll_metrics(total_nll=6.0, token_count=3)

        self.assertEqual(metrics["assistant_token_count"], 3)
        self.assertEqual(metrics["mean_nll"], 2.0)
        self.assertAlmostEqual(metrics["perplexity"], math.exp(2.0))

    def test_perplexity_rejects_invalid_or_unrepresentable_nll(self):
        with self.assertRaisesRegex(ValueError, "token_count"):
            nll_metrics(total_nll=1.0, token_count=0)
        with self.assertRaisesRegex(ValueError, "mean_nll"):
            nll_metrics(total_nll=float("inf"), token_count=1)

    def test_nf4_evidence_requires_initialized_nf4_linear_layers(self):
        evidence = validate_nf4_evidence(True, "nf4", 3, 3, ["nf4", "nf4", "nf4"])

        self.assertEqual(evidence["status"], "PASS")
        self.assertEqual(evidence["linear4bit_module_count"], 3)

    def test_nf4_evidence_rejects_wrong_format_or_missing_quantization(self):
        cases = [
            (False, "nf4", 2, 2, ["nf4", "nf4"]),
            (True, "fp4", 2, 2, ["fp4", "fp4"]),
            (True, None, 2, 2, ["nf4", "nf4"]),
            (True, "nf4", 0, 0, []),
            (True, "nf4", 2, 1, ["nf4"]),
            (True, "nf4", 2, 2, ["fp4", "nf4"]),
        ]

        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(validate_nf4_evidence(*case)["status"], "FAIL")

    def test_nf4_evidence_rejects_impossible_module_counts(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            validate_nf4_evidence(True, "nf4", 1, 2, ["nf4", "nf4"])


if __name__ == "__main__":
    unittest.main()
