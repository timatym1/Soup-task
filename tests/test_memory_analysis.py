import unittest
from types import SimpleNamespace

from src.memory_analysis import build_estimate


class MemoryAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=256,
            tie_word_embeddings=True,
        )

    def test_estimates_fp32_loss_logits_and_adapter_gradients(self):
        estimate = build_estimate(self.config, batch_size=2, sequence_length=16)

        self.assertEqual(
            estimate["training_estimate_bytes"]["logits"],
            2 * 16 * self.config.vocab_size * 4,
        )
        self.assertEqual(
            estimate["training_estimate_bytes"]["loss_workspace"],
            estimate["training_estimate_bytes"]["logits"],
        )
        self.assertEqual(
            estimate["training_estimate_bytes"]["adapter_and_optimizer_states"],
            estimate["lora_parameter_count_estimate"] * 10,
        )
        self.assertIn("FP32 logits", estimate["assumptions"]["logits"])

    def test_rejects_invalid_memory_dimensions(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            build_estimate(self.config, batch_size=0, sequence_length=16)


if __name__ == "__main__":
    unittest.main()
