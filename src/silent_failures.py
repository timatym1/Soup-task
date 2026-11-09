import json
import shutil
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


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
            "reason": "Three-model verification has not produced a report.",
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
    quality = read_json(REPORT_DIR / "quality_evaluation.json")
    safety = read_json(REPORT_DIR / "safety_evaluation.json")
    training = read_json(ADAPTER_DIR / "training_metrics.json")
    merge = read_json(MERGED_DIR / "merge_metrics.json")
    quantization = read_json(QUANTIZED_DIR / "quantization_metrics.json")

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
        "base": exists(BASE_DIR / "config.json") and has_model_weights(BASE_DIR),
        "adapter": exists(ADAPTER_DIR / "adapter_config.json") and has_model_weights(ADAPTER_DIR),
        "merged": exists(MERGED_DIR / "config.json") and has_model_weights(MERGED_DIR),
        "quantized": exists(QUANTIZED_DIR / "config.json") and has_model_weights(QUANTIZED_DIR),
    }
    gpu_evidence = (
        bool(shutil.which("nvidia-smi"))
        and exists(LOG_DIR / "nvidia-smi.txt")
        and bool(training.get("gpu_peak"))
        and bool(merge.get("gpu_peak"))
        and bool(quantization.get("gpu_peak"))
    )
    soup_cli = shutil.which("soup")
    training_pass = (
        bool(training.get("train_metrics"))
        and bool(training.get("eval_metrics"))
        and model_checks["adapter"]
    )
    merge_pass = model_checks["merged"] and bool(merge.get("gpu_peak"))
    quantization_pass = model_checks["quantized"] and bool(quantization.get("gpu_peak"))
    quality_pass = review_complete(quality, "quality")
    safety_pass = review_complete(safety, "safety")

    checks = [
        ("Data validation", "PASS" if data_pass else "BLOCKED"),
        ("Training and adapter", "PASS" if training_pass else "BLOCKED"),
        ("Merge", "PASS" if merge_pass else "BLOCKED"),
        ("Quantization", "PASS" if quantization_pass else "BLOCKED"),
        ("Three-model verification", "PASS" if verification_pass else "BLOCKED"),
        ("Quality evaluation", "PASS" if quality_pass else "BLOCKED"),
        ("Safety evaluation", "PASS" if safety_pass else "BLOCKED"),
        ("GPU and measured memory", "PASS" if gpu_evidence else "BLOCKED"),
        ("Soup CLI", "BLOCKED"),
        (
            "Memory estimate",
            "PASS"
            if memory.get("status") == "ESTIMATE_ONLY"
            and memory.get("training_estimate_bytes", {}).get("total")
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
        "| Soup CLI | Unknown or unavailable command contract | Installed CLI and documented, authorized command specification | "
        + checks[8][1]
        + " |",
        "| Memory estimate | Estimate absent or mistaken for measured usage | Memory estimate explicitly distinguished from recorded CUDA peaks | "
        + checks[9][1]
        + " |",
        "",
        "BLOCKED means evidence is unavailable or the specified review was not completed; it is not a PASS. "
        "A model loading or output-difference check alone does not establish quality, safety, or acceptable quantization drift.",
        "",
    ]
    (REPORT_DIR / "silent_failures.md").write_text("\n".join(lines), encoding="utf-8")

    soup_reason = (
        "The executable is not installed, and no Soup CLI command/subcommand contract is documented."
        if not soup_cli
        else "No Soup CLI command/subcommand specification is provided in the project documentation."
    )
    all_models_verified = verification_pass and all(model_checks.values())
    release_checks = {
        "data_validation": data_pass,
        "training": training_pass,
        "merge": merge_pass,
        "quantization": quantization_pass,
        "verification": all_models_verified,
        "quality_review": quality_pass,
        "safety_review": safety_pass,
        "gpu_measurements": gpu_evidence,
        "memory_estimate": (
            memory.get("status") == "ESTIMATE_ONLY"
            and bool(memory.get("training_estimate_bytes", {}).get("total"))
        ),
        "soup_cli": False,
    }
    verdict = "SHIP" if all(release_checks.values()) else "DON'T SHIP"
    check_status = {
        name: "PASS" if passed else "BLOCKED"
        for name, passed in release_checks.items()
    }
    quality_cases = quality.get("case_count", 0)
    safety_cases = safety.get("case_count", 0)
    report_lines = [
        "# Final Technical Report",
        "",
        f"**Final status: {verdict}.**",
        "",
        f"**Generated:** {timestamp()}",
        "",
        f"**Model:** `{MODEL_ID}`. **Data:** `{data.get('dataset_id', 'not available')}`. "
        f"**Data status:** `{data.get('status', 'BLOCKED')}`.",
        "",
        "**Quality evaluation:** "
        f"`{quality.get('status', 'BLOCKED')}`; {quality_cases} independent cases; "
        "ratings are not inferred without review.",
        "",
        "**Safety evaluation:** "
        f"`{safety.get('status', 'BLOCKED')}`; {safety_cases} cases; "
        "safety judgments are not inferred without review.",
        "",
        "**Verification:** "
        f"`{verification.get('status', 'BLOCKED')}`; "
        f"{verification.get('prompt_count', 'not run')} prompts.",
        "",
        "**Soup CLI:** `BLOCKED` - " + soup_reason,
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    report_lines.extend(f"| {name.replace('_', ' ').capitalize()} | {status} |" for name, status in check_status.items())
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
        "python -m unittest discover -s tests -v",
        "if [ ! -f data/train.jsonl ] || [ ! -f data/test.jsonl ] || [ ! -f reports/data_validation.json ] || ! python -c \"import json; assert json.load(open('reports/data_validation.json', encoding='utf-8')).get('status') == 'PASS'\"; then python -m src.data_preparation; fi",
        "python -m src.memory_analysis",
        "python -m src.verification --samples 20",
        "python -m src.model_evaluation",
        "python -m src.silent_failures",
        "```",
        "",
        f"Soup CLI remains blocked: {soup_reason} No Soup command is run or guessed.",
        "",
        "The working copy has no model weights, training data, or training logs; installed local "
        f"dependencies: torch={find_spec('torch') is not None}, "
        f"transformers={find_spec('transformers') is not None}, "
        f"bitsandbytes={find_spec('bitsandbytes') is not None}. No training or inference was claimed.",
        "",
    ]
    (REPORT_DIR / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Wrote evidence-gated release and silent-failure reports")


if __name__ == "__main__":
    main()
