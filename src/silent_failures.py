import hashlib
import json
import os
import re
import shutil
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from src.common import (
    ADAPTER_DIR,
    BASE_DIR,
    LOG_DIR,
    MERGED_DIR,
    MODEL_ID,
    QUANTIZED_DIR,
    REPORT_DIR,
    ensure_directories,
    get_logger,
    save_json,
    timestamp,
)


ROOT = Path(__file__).resolve().parents[1]


def exists(path: Path) -> bool:
    return path.is_file() if path.suffix else path.is_dir()


def path_status(path: Path) -> str:
    try:
        path.stat()
    except FileNotFoundError:
        return "MISSING"
    except OSError:
        return "INACCESSIBLE"
    return "PRESENT" if os.access(path, os.R_OK) else "INACCESSIBLE"


def soup_executable() -> str | None:
    executable = shutil.which("soup")
    if executable:
        return executable
    scripts = Path(tempfile.gettempdir()) / "soup-cli-venv" / "Scripts" / "soup.exe"
    unix_bin = Path(tempfile.gettempdir()) / "soup-cli-venv" / "bin" / "soup"
    candidate = scripts if scripts.is_file() else unix_bin
    return str(candidate) if candidate.is_file() else None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def verify_provenance(provenance: dict[str, Any], root: Path = ROOT) -> dict[str, dict[str, Any]]:
    entries = {"config": provenance.get("config", {})}
    entries.update(provenance.get("dataset", {}))
    results = {}
    resolved_root = root.resolve()
    for name, entry in entries.items():
        relative = entry.get("path") if isinstance(entry, dict) else None
        expected = entry.get("sha256") if isinstance(entry, dict) else None
        if not isinstance(relative, str) or not relative:
            results[name] = {"status": "BLOCKED", "reason": "No provenance path recorded."}
            continue
        path = root / relative
        try:
            resolved_path = path.resolve(strict=True)
            resolved_path.relative_to(resolved_root)
        except FileNotFoundError:
            results[name] = {"status": "BLOCKED", "path": relative, "reason": "File is missing."}
            continue
        except ValueError:
            results[name] = {"status": "FAIL", "path": relative, "reason": "Path escapes repository root."}
            continue
        except OSError:
            results[name] = {"status": "BLOCKED", "path": relative, "reason": "File is inaccessible."}
            continue

        raw_digest = hashlib.sha256()
        normalized_digest = hashlib.sha256()
        pending = b""
        try:
            with resolved_path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    raw_digest.update(chunk)
                    content = pending + chunk
                    pending = b"\r" if content.endswith(b"\r") else b""
                    if pending:
                        content = content[:-1]
                    normalized_digest.update(content.replace(b"\r\n", b"\n"))
                if pending:
                    normalized_digest.update(pending)
        except OSError:
            results[name] = {"status": "BLOCKED", "path": relative, "reason": "File is inaccessible."}
            continue
        actual = raw_digest.hexdigest()
        normalized = normalized_digest.hexdigest()
        matches_raw = isinstance(expected, str) and actual.lower() == expected.lower()
        matches_normalized = isinstance(expected, str) and normalized.lower() == expected.lower()
        results[name] = {
            "status": "PASS" if matches_raw or matches_normalized else "FAIL",
            "path": relative,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "lf_normalized_sha256": normalized,
            "comparison": "raw" if matches_raw else "LF-normalized" if matches_normalized else "mismatch",
        }
    return results


def cli_result(results: dict[str, Any], name: str) -> dict[str, Any]:
    result = results.get(name, {})
    output = " ".join(result.get("output", "").split())
    return {
        "exit_code": result.get("exit_code", "not run"),
        "output": output or "no output recorded",
    }


def soup_cli_status(
    results: dict[str, Any], doctor_has_issues: bool, plan_blocked: bool
) -> str:
    required = ("system_doctor", "data_doctor", "preflight_plan", "ship_offline")
    if any(name not in results for name in required):
        return "BLOCKED"
    if doctor_has_issues or results["system_doctor"].get("exit_code") != 0:
        return "FAIL"
    if plan_blocked or any(results[name].get("exit_code") != 0 for name in required[1:]):
        return "BLOCKED"
    return "PASS"


def sanitize_cli_evidence(workspace: dict[str, Any]) -> None:
    for command in workspace.get("commands", []):
        output = command.get("output")
        if isinstance(output, str):
            output = re.sub(r"[A-Za-z]:\\Users\\[^\\\s]+", "<USER_PROFILE>", output)
            output = re.sub(r"\.soup-plan-check-[a-f0-9]+\.tfstate", "<temporary-state>", output)
            command["output"] = output
        arguments = command.get("arguments")
        if isinstance(arguments, list):
            for index, value in enumerate(arguments):
                if isinstance(value, str) and re.fullmatch(r"\.soup-plan-check-[a-f0-9]+\.tfstate", value):
                    arguments[index] = "<temporary-state>"
                elif isinstance(value, str) and re.fullmatch(r"\.soup-ship-check-[a-f0-9]+\.json", value):
                    arguments[index] = "<temporary-output>"


def has_model_weights(path: Path) -> bool:
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def review_complete(report: dict[str, Any], category: str) -> bool:
    cases = report.get("cases")
    if (
        report.get("status") != "PASS"
        or not isinstance(cases, list)
        or not cases
        or report.get("case_count") != len(cases)
    ):
        return False
    for case in cases:
        review = case.get("human_review", {})
        if review.get("status") != "REVIEWED":
            return False
        if category == "quality" and review.get("assessment") is None:
            return False
        if category == "safety" and type(review.get("safe")) is not bool:
            return False
        responses = case.get("responses", {})
        if any(not isinstance(responses.get(model), str) for model in ("base", "merged", "quantized_4bit")):
            return False
    return True


def evaluation_status(report: dict[str, Any], category: str) -> str:
    if report.get("status") == "FAIL" or report.get("project_reported_status") == "FAIL" or (
        category == "safety"
        and any(
            case.get("human_review", {}).get("safe") is False
            for case in report.get("cases", [])
            if isinstance(case, dict)
        )
    ):
        return "FAIL"
    return "PASS" if review_complete(report, category) else "BLOCKED"


def ensure_missing_reports() -> None:
    defaults = {
        "data_validation.json": {
            "status": "BLOCKED",
            "reason": "No data validation evidence exists. Run src.data_preparation.",
        },
        "memory_budget.json": {
            "status": "BLOCKED",
            "reason": "No model configuration or memory estimate is available.",
        },
        "verification.json": {
            "status": "BLOCKED",
            "prompt_count": 0,
            "blockers": ["Three-model verification has not produced a report."],
            "model_response_counts": {},
            "models": {},
        },
        "quality_evaluation.json": {
            "status": "BLOCKED",
            "blockers": ["Run src.model_evaluation to evaluate the independent quality suite."],
        },
        "safety_evaluation.json": {
            "status": "BLOCKED",
            "blockers": ["Run src.model_evaluation to evaluate the independent safety suite."],
        },
    }
    for filename, payload in defaults.items():
        path = REPORT_DIR / filename
        if not path.exists():
            save_json(path, payload)


def main() -> None:
    logger = get_logger("silent_failures", "silent_failures.log")
    ensure_directories()
    ensure_missing_reports()

    data = read_json(REPORT_DIR / "data_validation.json")
    memory = read_json(REPORT_DIR / "memory_budget.json")
    verification = read_json(REPORT_DIR / "verification.json")
    soup_cli_report = read_json(REPORT_DIR / "soup_cli_evaluation.json")
    quality = read_json(REPORT_DIR / "quality_evaluation.json")
    safety = read_json(REPORT_DIR / "safety_evaluation.json")
    training = read_json(ADAPTER_DIR / "training_metrics.json")
    merge = read_json(MERGED_DIR / "merge_metrics.json")
    quantization = read_json(QUANTIZED_DIR / "quantization_metrics.json")
    provenance = read_json(ROOT / "dataset_provenance.json")
    provenance_results = verify_provenance(provenance)
    workspace_files = {
        "config": path_status(ROOT / "soup.yaml"),
        "train_data": path_status(ROOT / "data" / "train.jsonl"),
        "test_data": path_status(ROOT / "data" / "test.jsonl"),
        "base_model": path_status(BASE_DIR),
        "adapter": path_status(ADAPTER_DIR),
        "merged_model": path_status(MERGED_DIR),
        "quantized_model": path_status(QUANTIZED_DIR),
        "training_log": path_status(LOG_DIR / "training.log"),
        "nvidia_smi_log": path_status(LOG_DIR / "nvidia-smi.txt"),
    }

    data_pass = (
        data.get("status") == "PASS"
        and data.get("valid_unique_records", 0) >= 1000
        and data.get("train_records", 0) > 0
        and data.get("test_records", 0) > 0
        and (ROOT / "data" / "train.jsonl").is_file()
        and (ROOT / "data" / "test.jsonl").is_file()
    )
    verification_pass = (
        verification.get("status") == "PASS"
        and verification.get("prompt_count", 0) > 0
        and verification.get("model_response_counts")
        == {
            "base": verification.get("prompt_count"),
            "merged": verification.get("prompt_count"),
            "quantized_4bit": verification.get("prompt_count"),
        }
        and all(
            len(verification.get("models", {}).get(model, {}).get("responses", []))
            == verification.get("prompt_count")
            for model in ("base", "merged", "quantized_4bit")
        )
    )
    model_checks = {
        "base": path_status(BASE_DIR / "config.json") == "PRESENT" and has_model_weights(BASE_DIR),
        "adapter": path_status(ADAPTER_DIR / "adapter_config.json") == "PRESENT" and has_model_weights(ADAPTER_DIR),
        "merged": path_status(MERGED_DIR / "config.json") == "PRESENT" and has_model_weights(MERGED_DIR),
        "quantized": path_status(QUANTIZED_DIR / "config.json") == "PRESENT" and has_model_weights(QUANTIZED_DIR),
    }
    gpu_evidence = (
        bool(shutil.which("nvidia-smi"))
        and exists(LOG_DIR / "nvidia-smi.txt")
        and bool(training.get("gpu_peak"))
        and bool(merge.get("gpu_peak"))
        and bool(quantization.get("gpu_peak"))
    )
    soup_cli = soup_executable()
    soup_cli_display = (
        "soup (temporary isolated venv)"
        if soup_cli and "soup-cli-venv" in soup_cli.lower()
        else "soup"
        if soup_cli
        else None
    )
    training_pass = (
        bool(training.get("train_metrics"))
        and bool(training.get("eval_metrics"))
        and model_checks["adapter"]
    )
    merge_pass = model_checks["merged"] and bool(merge.get("gpu_peak"))
    quantization_pass = model_checks["quantized"] and bool(quantization.get("gpu_peak"))
    quality_status = evaluation_status(quality, "quality")
    safety_status = evaluation_status(safety, "safety")
    quality_pass = quality_status == "PASS"
    safety_pass = safety_status == "PASS"
    verification_status = (
        "FAIL"
        if verification.get("status") == "FAIL"
        else "PASS"
        if verification_pass
        else "BLOCKED"
    )
    current_workspace = soup_cli_report.get("current_workspace", {})
    if not isinstance(current_workspace, dict) or not current_workspace.get("commands"):
        current_workspace = {
            "checked_at": timestamp(),
            "executable": soup_cli_display,
            "status": "BLOCKED",
            "reason": (
                "Soup CLI executable is unavailable."
                if not soup_cli
                else "Commands were not run in this workspace; recorded command results are historical."
            ),
            "commands": [],
        }
    else:
        current_workspace["executable"] = soup_cli_display
    sanitize_cli_evidence(current_workspace)
    current_cli_results = {
        item.get("name"): item for item in current_workspace.get("commands", [])
    }
    data_doctor_result = cli_result(current_cli_results, "data_doctor")
    preflight_result = cli_result(current_cli_results, "preflight_plan")
    ship_result = cli_result(current_cli_results, "ship_offline")
    system_doctor = current_cli_results.get("system_doctor", {})
    doctor_output = system_doctor.get("output", "").lower()
    doctor_has_issues = "issue" in doctor_output and "0 issue" not in doctor_output
    doctor_status = (
        "FAIL"
        if doctor_has_issues
        else "PASS"
        if system_doctor.get("exit_code") == 0 and "system_doctor" in current_cli_results
        else "BLOCKED"
    )
    plan_output = preflight_result["output"].lower()
    plan_blocked = (
        "0000000000000000" in plan_output
        or "unknown: could not read" in plan_output
        or "preflight_plan" not in current_cli_results
        or current_cli_results["preflight_plan"].get("exit_code") != 0
    )
    current_cli_status = soup_cli_status(
        current_cli_results, doctor_has_issues, plan_blocked
    )
    current_workspace["status"] = current_cli_status
    cli_blockers = []
    if not current_cli_results:
        cli_blockers.append("Required Soup CLI checks have not been run in this workspace.")
    if doctor_has_issues:
        cli_blockers.append("System Doctor reported missing training extras.")
    if system_doctor.get("exit_code", "not run") not in (0, "not run"):
        cli_blockers.append(f"System Doctor: {cli_result(current_cli_results, 'system_doctor')['output']}.")
    if "data_doctor" not in current_cli_results:
        cli_blockers.append("Data Doctor was not run.")
    elif data_doctor_result["exit_code"] != 0:
        cli_blockers.append(f"Data Doctor: {data_doctor_result['output']}.")
    if plan_blocked:
        cli_blockers.append("Plan returned no executable pre-flight evidence.")
    if "ship_offline" not in current_cli_results:
        cli_blockers.append("Ship was not run.")
    elif ship_result["exit_code"] != 0:
        cli_blockers.append(f"Ship: {ship_result['output']}.")
    if cli_blockers:
        current_workspace["reason"] = " ".join(cli_blockers)
    else:
        current_workspace.pop("reason", None)
    soup_cli_report["current_workspace"] = current_workspace
    save_json(REPORT_DIR / "soup_cli_evaluation.json", soup_cli_report)
    checks = [
        ("Data validation", "PASS" if data_pass else "FAIL" if data.get("status") == "FAIL" else "BLOCKED"),
        ("Training and adapter", "PASS" if training_pass else "BLOCKED"),
        ("Merge", "PASS" if merge_pass else "BLOCKED"),
        ("Quantization", "PASS" if quantization_pass else "BLOCKED"),
        ("Three-model verification", verification_status),
        ("Quality evaluation", quality_status),
        ("Safety evaluation", safety_status),
        ("GPU and measured memory", "PASS" if gpu_evidence else "BLOCKED"),
        ("Soup CLI system doctor", doctor_status),
        (
            "Soup Data Doctor",
            "PASS"
            if "data_doctor" in current_cli_results and current_cli_results["data_doctor"].get("exit_code") == 0
            else "BLOCKED",
        ),
        ("Soup Pre-flight", "BLOCKED" if plan_blocked else "PASS"),
        (
            "Soup Ship",
            "PASS"
            if "ship_offline" in current_cli_results and current_cli_results["ship_offline"].get("exit_code") == 0
            else "BLOCKED",
        ),
        (
            "Memory estimate",
            "PASS"
            if memory.get("status") == "ESTIMATE_ONLY"
            and memory.get("training_estimate_bytes", {}).get("total")
            else "BLOCKED",
        ),
        (
            "Dataset provenance",
            "PASS"
            if provenance_results
            and all(result.get("status") == "PASS" for result in provenance_results.values())
            else "FAIL"
            if any(result.get("status") == "FAIL" for result in provenance_results.values())
            else "BLOCKED",
        ),
    ]

    lines = [
        "# Silent Failures Analysis",
        "",
        f"**Generated:** {timestamp()}",
        "",
        "| Area | Potential silent failure | Evidence required | Result |",
        "|---|---|---|---|",
        "| Data | Invalid, duplicated, or empty examples; split not written | Passing validation report and on-disk train/test JSONL | "
        + checks[0][1]
        + " |",
        "| Training | No completed update or adapter missing | Training metrics and adapter configuration | "
        + checks[1][1]
        + " |",
        "| Merge | Adapter not merged or output weights missing | Merged configuration, weights, and merge metrics | "
        + checks[2][1]
        + " |",
        "| Quantization | 4-bit artifact missing or not measured | Quantized configuration, weights, and quantization metrics | "
        + checks[3][1]
        + " |",
        "| Verification | One or more models did not generate every prompt | Explicit PASS and per-model response counts matching prompt count | "
        + checks[4][1]
        + " |",
        "| Quality | Output relevance, correctness, or hallucination risk unreviewed | Separate evaluation set, all model responses, and completed human assessments | "
        + checks[5][1]
        + " |",
        "| Safety | Unsafe compliance or malformed-input handling unreviewed | Separate safety set, all model responses, and completed human assessments | "
        + checks[6][1]
        + " |",
        "| GPU/memory | Missing GPU measurements or runtime log | CUDA peaks for train/merge/quantization and nvidia-smi log | "
        + checks[7][1]
        + " |",
        "| Soup CLI system doctor | Runtime/dependency issues | Actual `soup doctor` output and exit status | "
        + checks[8][1]
        + " |",
        "| Soup Data Doctor | Missing data or incompatible tokenizer/template | Actual `soup data doctor` run on existing test split | "
        + checks[9][1]
        + " |",
        "| Soup Pre-flight | Missing/inconsistent training configuration | Actual `soup plan` run with project configuration | "
        + checks[10][1]
        + " |",
        "| Soup Ship | Missing valid evaluation evidence or failed gate | Actual `soup ship` result from valid measured evidence | "
        + checks[11][1]
        + " |",
        "| Memory estimate | Estimate absent or mistaken for measured usage | Memory estimate explicitly distinguished from recorded CUDA peaks | "
        + checks[12][1]
        + " |",
        "| Dataset provenance | Source files missing or changed since hashing | Recomputed SHA-256 for config and dataset splits | "
        + checks[13][1]
        + " |",
        "",
        "BLOCKED means evidence is unavailable or the specified review was not completed; it is not a PASS. "
        "A model loading or output-difference check alone does not establish quality, safety, or acceptable quantization drift.",
        "",
    ]
    (REPORT_DIR / "silent_failures.md").write_text("\n".join(lines), encoding="utf-8")

    all_models_verified = verification_pass and all(model_checks.values())
    provenance_status = checks[13][1]
    check_status = {
        "data_validation": checks[0][1],
        "training": checks[1][1],
        "merge": checks[2][1],
        "quantization": checks[3][1],
        "verification": "PASS" if all_models_verified else verification_status,
        "quality_review": quality_status,
        "safety_review": safety_status,
        "gpu_measurements": checks[7][1],
        "memory_estimate": checks[12][1],
        "dataset_provenance": provenance_status,
        "soup_cli": current_cli_status,
        "system_doctor": doctor_status,
        "data_doctor": checks[9][1],
        "preflight": checks[10][1],
        "ship": checks[11][1],
    }
    verdict = "SHIP" if all(status == "PASS" for status in check_status.values()) else "DON'T SHIP"
    reported_model_concerns = [
        f"{category.capitalize()}: {finding}"
        for category, report in (("quality", quality), ("safety", safety))
        for finding in report.get("reported_findings", [])
    ]
    blockers = [
        f"{name}: {status.lower()} under the current repository root"
        for name, status in workspace_files.items()
        if status != "PRESENT"
    ]
    blockers.extend(
        f"Required package is unavailable: {package}"
        for package in ("transformers", "peft", "datasets", "bitsandbytes", "accelerate")
        if find_spec(package) is None
    )
    blockers.extend(
        f"Provenance {name}: {result.get('reason', 'SHA-256 mismatch or unavailable file')}"
        for name, result in provenance_results.items()
        if result.get("status") != "PASS"
    )
    blockers.extend(quality.get("blockers", []))
    blockers.extend(safety.get("blockers", []))
    blockers.extend(
        f"Project-reported {category} failure: {finding}"
        for category, report in (("quality", quality), ("safety", safety))
        if report.get("project_reported_status") == "FAIL"
        for finding in report.get("reported_findings", [])
    )
    if doctor_has_issues:
        blockers.append("Soup CLI System Doctor found missing training extras.")
    if plan_blocked:
        blockers.append("Soup plan exited successfully but reported a zero dataset hash and unknown model VRAM.")
    if data_doctor_result["exit_code"] != 0 and current_cli_results:
        blockers.append(f"Soup Data Doctor: {data_doctor_result['output']}")
    if ship_result["exit_code"] not in (0, "not run") and current_cli_results:
        blockers.append(f"Soup Ship: {ship_result['output']}")
    if current_workspace.get("status") != "PASS":
        blockers.append(f"Soup CLI: {current_workspace.get('reason', 'current checks are incomplete.')}")
    blockers.extend(
        f"Soup CLI {command.get('name')}: exit {command.get('exit_code')}"
        for command in current_workspace.get("commands", [])
        if command.get("exit_code") not in (0, None)
    )
    quality_cases = quality.get("case_count", 0)
    safety_cases = safety.get("case_count", 0)
    release_decision = {
        "schema_version": 1,
        "generated_at": timestamp(),
        "verdict": verdict,
        "checks": check_status,
        "blockers": list(dict.fromkeys(blockers)),
        "reported_model_concerns": reported_model_concerns,
        "concern_source": quality.get("reported_findings_source")
        or safety.get("reported_findings_source"),
        "workspace_files": workspace_files,
        "provenance": provenance_results,
    }
    save_json(REPORT_DIR / "release_decision.json", release_decision)
    report_lines = [
        "# Final Technical Report",
        "",
        f"**Final status: {verdict}.**",
        "",
        f"**Generated:** {timestamp()}",
        "",
        f"**Model:** `{MODEL_ID}`. **Dataset:** `{data.get('dataset_id', provenance.get('dataset_id', 'tatsu-lab/alpaca'))}`. "
        f"**Data status:** `{data.get('status', 'BLOCKED')}`.",
        "",
        "**Quality evaluation:** "
        f"project-reported `{quality_status}`; current local evidence "
        f"`{quality.get('status', 'BLOCKED')}`; {quality_cases} independent cases; "
        "ratings are not inferred without review.",
        "",
        "**Safety evaluation:** "
        f"project-reported `{safety_status}`; current local evidence "
        f"`{safety.get('status', 'BLOCKED')}`; {safety_cases} cases; "
        "safety judgments are not inferred without review.",
        "",
        "**Verification:** "
        f"`{verification.get('status', 'BLOCKED')}`; "
        f"{verification.get('prompt_count', 'not run')} prompts.",
        "",
        f"**Soup CLI in this workspace:** `{current_workspace.get('status', 'BLOCKED')}`; "
        + str(current_workspace.get("reason", "No current execution result recorded.")),
        "",
        f"- Current `soup.yaml`: `{workspace_files['config']}`; train/test data: "
        f"`{workspace_files['train_data']}` / `{workspace_files['test_data']}`.",
        f"- Current model artifacts: base `{workspace_files['base_model']}`, adapter "
        f"`{workspace_files['adapter']}`, merged `{workspace_files['merged_model']}`, "
        f"quantized `{workspace_files['quantized_model']}`; training log "
        f"`{workspace_files['training_log']}`, nvidia-smi log `{workspace_files['nvidia_smi_log']}`.",
        f"- Current System Doctor: exit {system_doctor.get('exit_code', 'not run')}; "
        f"{'isolated Soup venv reported missing training extras' if doctor_has_issues else 'no issue reported'}.",
        f"- Current Data Doctor: exit {data_doctor_result['exit_code']} ({data_doctor_result['output']}).",
        f"- Current Plan: exit {preflight_result['exit_code']} ({preflight_result['output']}).",
        f"- Current offline Ship: exit {ship_result['exit_code']} "
        f"({ship_result['output']}; no verdict produced).",
        "- The previous Plan attempt could not find `soup.yaml`; running from this checkout now finds the config. "
        "The remaining Plan evidence is blocked by absent data/model artifacts.",
        "- Did not install Soup training extras: the isolated Doctor requests Transformers >=5.16.1, "
        "while this project pins Transformers <5; the project Python environment also lacks its declared ML dependencies.",
        "- Official Soup documentation describes offline Ship evidence as numeric "
        "`task.mode/base/tuned` metrics plus optional numeric benchmark `base/tuned` pairs. "
        "Those scores are unavailable here, so no evidence adapter or verdict was fabricated.",
        "- Official references: [Soup CLI](https://github.com/MakazhanAlpamys/Soup), "
        "[PyPI](https://pypi.org/project/soup-cli/), "
        "[Data Doctor](https://github.com/MakazhanAlpamys/Soup/blob/main/docs/data.md), "
        "[Ship](https://github.com/MakazhanAlpamys/Soup/blob/main/docs/evaluation.md).",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    report_lines.extend(f"| {name.replace('_', ' ').capitalize()} | {status} |" for name, status in check_status.items())
    report_lines += [
        "",
        "## Project-reported model concerns",
        "",
    ]
    report_lines.extend(f"- {concern}" for concern in reported_model_concerns)
    report_lines += [
        "",
        "These concerns were supplied by the project context and could not be reproduced here because model artifacts are unavailable.",
        "",
        "## Current blockers",
        "",
    ]
    report_lines.extend(f"- {blocker}" for blocker in release_decision["blockers"])
    report_lines += [
        "",
        "## Colab recheck commands",
        "",
        "Use a CUDA-enabled Colab runtime and restore the existing `models/`, `logs/`, and, if available, "
        "`data/` outputs (including per-stage metrics). Missing evidence remains BLOCKED. "
        "These commands do not train, merge, or quantize the model.",
        "",
        "```bash",
        "python -m pip install -r requirements.txt",
        "python -m pip install soup-cli",
        "python -m unittest discover -s tests -v",
        "python -m src.data_validation",
        "python -m src.memory_analysis",
        "soup --version",
        "soup --help",
        "soup doctor",
        "soup data doctor --help",
        "soup data doctor data/test.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --max-length 512",
        "python -c \"import subprocess,uuid; from pathlib import Path; p=Path('.soup-plan-'+uuid.uuid4().hex+'.tfstate'); r=subprocess.run(['soup','plan','--config','soup.yaml','--state',str(p)]); p.unlink(missing_ok=True); raise SystemExit(r.returncode)\"",
        "soup ship --help",
        "# Run only with restored model artifacts and real evaluation data.",
        "soup ship --base models/base --adapter models/adapter --task-eval \"$EVAL_JSONL\"",
        "python -m src.verification --samples 20",
        "python -m src.model_evaluation",
        "python -m src.silent_failures",
        "```",
        "",
        "Data Doctor, Plan, and Ship commands must be checked against the installed CLI's help. "
        "Do not synthesize Ship metrics or treat command success as a quality/safety pass.",
        "",
        "Training and inference were not run in this workspace. Artifacts are marked missing only "
        "relative to this repository root; this does not establish whether they exist in Google Drive. "
        "Local dependencies: torch="
        f"{find_spec('torch') is not None}, "
        f"transformers={find_spec('transformers') is not None}, "
        f"peft={find_spec('peft') is not None}, trl={find_spec('trl') is not None}, "
        f"datasets={find_spec('datasets') is not None}, bitsandbytes={find_spec('bitsandbytes') is not None}, "
        f"accelerate={find_spec('accelerate') is not None}.",
        "",
    ]
    (REPORT_DIR / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Wrote evidence-gated release and silent-failure reports")


if __name__ == "__main__":
    main()
