import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from src.model_evaluation import MODEL_PATHS
from src.silent_failures import (
    evaluation_status,
    path_status,
    sanitize_cli_evidence,
    soup_cli_status,
    verification_complete,
    verify_provenance,
)


class ReleaseEvidenceTests(unittest.TestCase):
    def test_path_status_distinguishes_present_and_missing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            present = root / "present.json"
            present.write_text("{}", encoding="utf-8")

            self.assertEqual(path_status(present), "PRESENT")
            self.assertEqual(path_status(root / "missing.json"), "MISSING")

    def test_historical_metrics_reference_the_original_tracked_report(self):
        evidence = json.loads(
            (Path(__file__).parents[1] / "reports" / "historical_run_evidence.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(evidence["source_commit"], "66c591c4aed2485a41f038525915f88e572e6c94")
        self.assertEqual(evidence["configuration"]["train_examples"], 1800)
        self.assertEqual(evidence["verification"]["changed_base_to_merged_responses"], 17)
        self.assertTrue(any("historical" in item.lower() for item in evidence["limitations"]))

    def test_verification_requires_matched_outputs_and_finite_nll_for_every_model(self):
        names = ("base", "merged", "quantized_4bit")
        report = {
            "status": "COMPLETED",
            "prompt_count": 1,
            "model_response_counts": {name: 1 for name in names},
            "models": {
                name: {
                    "responses": ["answer"],
                    "heldout_nll": {
                        "total_nll": 2.0,
                        "assistant_token_count": 2,
                        "mean_nll": 1.0,
                        "perplexity": 2.718281828459045,
                    },
                    **(
                        {
                            "quantization": {
                                "status": "PASS",
                                "configured_load_in_4bit": True,
                                "configured_quant_type": "nf4",
                                "linear4bit_module_count": 1,
                                "initialized_quant_state_count": 1,
                                "runtime_quant_types": ["nf4"],
                            }
                        }
                        if name == "quantized_4bit"
                        else {}
                    ),
                }
                for name in names
            },
            "merged_vs_quantized_full_answer_logit_drift": {
                "assistant_token_count": 2,
                "top1_match_count": 2,
                "mean_kl_reference_to_candidate": 0.0,
                "top1_agreement": 1.0,
                "max_absolute_logit_delta": 0.0,
            },
        }

        self.assertTrue(verification_complete(report))
        del report["models"]["quantized_4bit"]["quantization"]
        self.assertFalse(verification_complete(report))
        report["models"]["quantized_4bit"]["quantization"] = {
            "status": "PASS",
            "configured_load_in_4bit": True,
            "configured_quant_type": "nf4",
            "linear4bit_module_count": 1,
            "initialized_quant_state_count": 1,
            "runtime_quant_types": ["nf4"],
        }
        del report["models"]["quantized_4bit"]["heldout_nll"]
        self.assertFalse(verification_complete(report))
        report["models"]["quantized_4bit"]["heldout_nll"] = {
            "total_nll": 1.0,
            "assistant_token_count": 1,
            "mean_nll": float("nan"),
            "perplexity": 1.0,
        }
        self.assertFalse(verification_complete(report))
        report["models"]["quantized_4bit"]["heldout_nll"] = {
            "total_nll": 3.0,
            "assistant_token_count": 3,
            "mean_nll": 1.0,
            "perplexity": math.exp(1.0),
        }
        self.assertFalse(verification_complete(report))

    def test_provenance_hashes_are_recomputed_from_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "soup.yaml"
            config.write_bytes(b"task: sft\n")
            expected = hashlib.sha256(config.read_bytes()).hexdigest()
            provenance = {
                "config": {"path": "soup.yaml", "sha256": expected},
                "dataset": {"train": {"path": "missing.jsonl", "sha256": "0" * 64}},
            }

            results = verify_provenance(provenance, root)

            self.assertEqual(results["config"]["status"], "PASS")
            self.assertEqual(results["config"]["actual_sha256"], expected)
            self.assertEqual(results["train"]["status"], "BLOCKED")

            config.write_bytes(b"task: sft\r\n")
            normalized = verify_provenance(provenance, root)["config"]
            self.assertEqual(normalized["status"], "PASS")
            self.assertEqual(normalized["comparison"], "LF-normalized")

            config.write_text("task: changed\n", encoding="utf-8")
            self.assertEqual(verify_provenance(provenance, root)["config"]["status"], "FAIL")

    def test_explicit_evaluation_failure_is_not_downgraded_to_blocked(self):
        self.assertEqual(evaluation_status({"status": "FAIL"}, "quality"), "BLOCKED")
        self.assertEqual(
            evaluation_status({"status": "BLOCKED", "project_reported_status": "FAIL"}, "quality"),
            "BLOCKED",
        )
        self.assertEqual(evaluation_status({"status": "BLOCKED"}, "safety"), "BLOCKED")
        self.assertEqual(
            evaluation_status(
                {"status": "BLOCKED", "cases": [{"human_review": {"safe": False}}]},
                "safety",
            ),
            "BLOCKED",
        )
        self.assertEqual(
            evaluation_status(
                {
                    "status": "NEEDS_HUMAN_REVIEW",
                    "cases": [
                        {
                            "human_review": {"status": "REVIEWED", "assessment": "FAIL"},
                            "responses": {name: "response" for name in MODEL_PATHS},
                        }
                    ],
                    "case_count": 1,
                },
                "quality",
            ),
            "FAIL",
        )
        self.assertEqual(
            evaluation_status(
                {
                    "status": "NEEDS_HUMAN_REVIEW",
                    "cases": [
                        {
                            "human_review": {"status": "REVIEWED", "assessment": "PASS"},
                            "responses": {name: "response" for name in MODEL_PATHS},
                        }
                    ],
                    "case_count": 1,
                },
                "quality",
            ),
            "PASS",
        )

    def test_cli_report_redacts_local_user_and_temporary_paths(self):
        workspace = {
            "commands": [
                {
                    "output": r"Config at C:\Users\alice\repo\.soup-plan-check-abc123.tfstate",
                    "arguments": ["--state", ".soup-plan-check-abc123.tfstate"],
                }
            ]
        }

        sanitize_cli_evidence(workspace)

        self.assertNotIn(r"C:\Users\alice", workspace["commands"][0]["output"])
        self.assertIn("<temporary-state>", workspace["commands"][0]["output"])
        self.assertEqual(workspace["commands"][0]["arguments"][1], "<temporary-state>")

    def test_soup_cli_gate_passes_only_when_all_release_commands_pass(self):
        results = {
            name: {"exit_code": 0}
            for name in ("system_doctor", "data_doctor", "preflight_plan", "ship_offline")
        }

        self.assertEqual(soup_cli_status(results, False, False), "PASS")
        self.assertEqual(soup_cli_status(results, False, True), "BLOCKED")
        self.assertEqual(soup_cli_status(results, True, False), "FAIL")

        del results["ship_offline"]
        self.assertEqual(soup_cli_status(results, False, False), "BLOCKED")

        results["ship_offline"] = {"exit_code": 3}
        self.assertEqual(soup_cli_status(results, False, False), "BLOCKED")


if __name__ == "__main__":
    unittest.main()
