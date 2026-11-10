import hashlib
import tempfile
import unittest
from pathlib import Path

from src.silent_failures import (
    evaluation_status,
    path_status,
    sanitize_cli_evidence,
    soup_cli_status,
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
        self.assertEqual(evaluation_status({"status": "FAIL"}, "quality"), "FAIL")
        self.assertEqual(
            evaluation_status({"status": "BLOCKED", "project_reported_status": "FAIL"}, "quality"),
            "FAIL",
        )
        self.assertEqual(evaluation_status({"status": "BLOCKED"}, "safety"), "BLOCKED")

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
