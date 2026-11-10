import tempfile
import unittest
from pathlib import Path

from src.model_evaluation import (
    MODEL_PATHS,
    build_report,
    load_cases,
    model_artifact_errors,
)
from src.silent_failures import review_complete
from src.verification import build_blocked_report, load_records


class ModelEvaluationTests(unittest.TestCase):
    def test_quality_and_safety_suites_are_separate_and_include_edge_cases(self):
        cases = load_cases()

        self.assertEqual(len(cases["quality_cases"]), 9)
        self.assertEqual(len(cases["safety_cases"]), 6)
        self.assertIn(
            "", [case["prompt"] for case in cases["safety_cases"]]
        )
        self.assertTrue(
            all(case.get("criteria") for case in cases["quality_cases"] + cases["safety_cases"])
        )

    def test_report_blocks_without_artifact_evidence_and_does_not_invent_ratings(self):
        case = {"id": "sample", "prompt": "Question", "criteria": "Check the answer."}

        report = build_report(
            "fixture-suite",
            "quality",
            [case],
            {},
            ["models are missing"],
        )

        self.assertEqual(report["status"], "BLOCKED")
        self.assertNotIn("reported_findings", report)
        self.assertEqual(report["cases"][0]["responses"]["base"], None)
        self.assertIsNone(report["cases"][0]["human_review"]["assessment"])

    def test_generated_comparison_requires_every_model_response(self):
        case = {"id": "sample", "prompt": "Question", "criteria": "Check the answer."}
        outputs = {name: {"sample": "answer"} for name in MODEL_PATHS}

        report = build_report("fixture-suite", "quality", [case], outputs, [])

        self.assertEqual(report["status"], "NEEDS_HUMAN_REVIEW")
        self.assertEqual(set(report["cases"][0]["responses"]), set(MODEL_PATHS))
        with self.assertRaisesRegex(ValueError, "missing responses"):
            build_report("fixture-suite", "quality", [case], {}, [])

    def test_artifact_check_requires_weights_as_well_as_model_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "config.json").write_text("{}", encoding="utf-8")

            self.assertTrue(model_artifact_errors({"model": path}))

            (path / "model.safetensors").write_bytes(b"fixture")
            self.assertEqual(model_artifact_errors({"model": path}), [])

    def test_release_review_gate_requires_completed_human_assessments(self):
        report = {
            "status": "NEEDS_HUMAN_REVIEW",
            "case_count": 1,
            "cases": [
                {
                    "responses": {name: "response" for name in MODEL_PATHS},
                    "human_review": {"status": "REVIEWED", "assessment": "PASS"},
                }
            ],
        }

        self.assertTrue(review_complete(report, "quality"))
        self.assertFalse(review_complete(report, "safety"))
        report["cases"][0]["human_review"]["safe"] = True
        self.assertTrue(review_complete(report, "safety"))
        report["cases"][0]["responses"]["base"] = None
        self.assertFalse(review_complete(report, "safety"))

    def test_verification_blocker_report_does_not_claim_success(self):
        report = build_blocked_report(["model artifacts are missing"])

        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["prompt_count"], 0)
        self.assertEqual(report["blockers"], ["model artifacts are missing"])
        self.assertEqual(report["models"], {})
        self.assertNotIn("heldout_nll", report)

    def test_load_records_validates_limit_and_supervised_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.jsonl"
            path.write_text(
                '{"instruction":"Question","input":"","output":"Answer"}\n',
                encoding="utf-8",
            )

            self.assertEqual(load_records(path, 1)[0]["output"], "Answer")
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                load_records(path, 0)

            path.write_text('{"instruction":"Question","output":""}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing assistant output"):
                load_records(path, 1)

    def test_completed_generation_review_is_not_mislabeled_as_quality_pass(self):
        case = {"id": "sample", "prompt": "Question", "criteria": "Check the answer."}
        responses = {name: {"sample": "answer"} for name in MODEL_PATHS}
        report = build_report(
            "fixture-suite",
            "quality",
            [case],
            responses,
            [],
        )

        self.assertEqual(report["status"], "NEEDS_HUMAN_REVIEW")
        self.assertNotIn("reported_findings", report)


if __name__ == "__main__":
    unittest.main()
