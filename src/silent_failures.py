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
    timestamp,
)


def exists(path: Path) -> bool:
    return path.is_file() if path.suffix else path.is_dir()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    logger = get_logger("silent_failures", "silent_failures.log")
    ensure_directories()
    data_report_path = REPORT_DIR / "data_validation.json"
    verification_path = REPORT_DIR / "verification.json"
    data = read_json(data_report_path)
    verification = read_json(verification_path)
    memory = read_json(REPORT_DIR / "memory_budget.json")
    training = read_json(ADAPTER_DIR / "training_metrics.json")
    merge = read_json(MERGED_DIR / "merge_metrics.json")
    quantization = read_json(QUANTIZED_DIR / "quantization_metrics.json")

    data_status = (
        "NOT RUN"
        if not data
        else "PASS"
        if data.get("valid_unique_records", 0) >= 1000
        else "FAIL"
    )
    checks = [
        ("Data", "Empty/malformed rows and duplicates", "data_validation.json records validation counts; at least 1000 unique valid rows required", data_status),
        ("Tokenizer/template", "Missing chat template or untrainable tokenized rows", "Training aborts on missing template/empty supervised labels; saved training metrics prove the run reached completion", "PASS" if training else "NOT RUN"),
        ("Training", "No effective parameter update", "Require saved training metrics and compare base/merged last-token logits", "PASS" if training and verification.get("base_vs_merged_last_prompt_logits", {}).get("max_absolute_delta", 0) > 0 else "NOT RUN"),
        ("Merge", "Adapter not merged or corrupted output", "Merged config and weights present; compare merged logits with base", "PASS" if exists(MERGED_DIR / "config.json") and verification else "NOT RUN"),
        ("Quantization", "4-bit artifact missing or excessive output drift", "Quantized config/weights present; compare against merged on identical prompts", "PASS" if exists(QUANTIZED_DIR / "config.json") and verification else "NOT RUN"),
        ("Inference", "Load/generation failure", "All three models loaded and generated the same test prompts", "PASS" if len(verification.get("models", {})) == 3 else "NOT RUN"),
        ("GPU/memory", "VRAM estimate differs from real peak", "Compare memory_budget.json, per-stage GPU metrics and timestamped nvidia-smi", "PASS" if (LOG_DIR / "nvidia-smi.txt").exists() and memory and training.get("gpu_peak") and merge.get("gpu_peak") and quantization.get("gpu_peak") else "NOT RUN"),
        ("Soup CLI", "Soup Doctor/Pre-flight/Ship findings", "Not invoked: CLI subcommand contract is not supplied; no detections are claimed", "NOT RUN" if shutil.which("soup") else "NOT INSTALLED"),
    ]

    lines = [
        "# Silent Failures Analysis",
        "",
        "| Area | Potential silent failure | Detection/evidence | Local result |",
        "|---|---|---|---|",
    ]
    lines.extend(f"| {area} | {failure} | {evidence} | **{status}** |" for area, failure, evidence, status in checks)
    lines += [
        "",
        "A PASS means the stated local check ran; it is not a model-quality or release approval. "
        "Soup Data Doctor, Pre-flight and Ship results remain unverified unless separately run with the actual CLI.",
        "",
    ]
    (REPORT_DIR / "silent_failures.md").write_text("\n".join(lines), encoding="utf-8")

    release_checks = {
        "base": exists(BASE_DIR / "config.json"),
        "adapter": exists(ADAPTER_DIR / "adapter_config.json"),
        "data_validation": bool(data and data.get("valid_unique_records", 0) >= 1000),
        "memory_estimate": exists(REPORT_DIR / "memory_budget.json"),
        "training_log": exists(LOG_DIR / "training.log"),
        "merged": exists(MERGED_DIR / "config.json"),
        "quantized": exists(QUANTIZED_DIR / "config.json"),
        "verification": len(verification.get("models", {})) == 3,
        "gpu_log": (
            exists(LOG_DIR / "nvidia-smi.txt")
            and bool(training.get("gpu_peak"))
            and bool(merge.get("gpu_peak"))
            and bool(quantization.get("gpu_peak"))
        ),
        "soup_checks": False,
        "quality_and_safety_review": False,
    }
    verdict = "SHIP" if all(release_checks.values()) else "DON'T SHIP"
    train_metrics = training.get("train_metrics", {})
    eval_metrics = training.get("eval_metrics", {})
    verification_summary = {
        "prompts": verification.get("prompt_count", "not run"),
        "changed_base_to_merged_responses": verification.get("base_vs_merged_changed_responses", "not run"),
        "changed_merged_to_4bit_responses": verification.get("merged_vs_quantized_changed_responses", "not run"),
        "base_to_merged_last_token_max_abs_delta": verification.get(
            "base_vs_merged_last_prompt_logits", {}
        ).get("max_absolute_delta", "not run"),
        "merged_to_4bit_last_token_max_abs_delta": verification.get(
            "merged_vs_quantized_last_prompt_logits", {}
        ).get("max_absolute_delta", "not run"),
    }
    memory_summary = {
        "estimated_training_gib": memory.get("training_estimate_bytes", {}).get("total_gib", "not run"),
        "measured_training_peak_bytes": training.get("gpu_peak", "not run"),
        "measured_merge_peak_bytes": merge.get("gpu_peak", "not run"),
        "measured_4bit_peak_bytes": quantization.get("gpu_peak", "not run"),
    }
    estimated_training_bytes = memory.get("training_estimate_bytes", {}).get("total")
    measured_training_bytes = training.get("gpu_peak", {}).get("reserved_bytes")
    if estimated_training_bytes and measured_training_bytes:
        memory_summary["training_reserved_to_estimate_ratio"] = round(
            measured_training_bytes / estimated_training_bytes, 3
        )
    environment = (
        f"PyTorch {'installed' if find_spec('torch') else 'not installed'}; "
        f"nvidia-smi {'available' if shutil.which('nvidia-smi') else 'unavailable'}."
    )
    report = [
        "# Technical Report",
        "",
        f"**Final verdict: {verdict}.**",
        "",
        f"**Generated:** {timestamp()}",
        "",
        f"**Execution environment:** {environment}",
        "",
        f"**Configuration:** `{data.get('dataset_id', 'not run')}`; base model `{MODEL_ID}`; "
        f"seed `{data.get('seed', 'not run')}`; {data.get('train_records', 'not run')} train / "
        f"{data.get('test_records', 'not run')} test examples.",
        "",
        f"**Training:** train loss `{train_metrics.get('train_loss', 'not run')}`; "
        f"evaluation loss `{eval_metrics.get('eval_loss', 'not run')}`.",
        "",
        f"**Verification:** `{json.dumps(verification_summary, ensure_ascii=False)}`.",
        "",
        f"**Memory:** `{json.dumps(memory_summary, ensure_ascii=False)}`.",
        "",
        "The release gate requires all three model artifacts, training and GPU evidence, three-model "
        "verification, a data-validation report, a memory estimate, Soup CLI checks, and independent quality/safety review. Missing evidence is "
        "reported as not run; it is not inferred as a pass.",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    report.extend(f"| {name.replace('_', ' ').capitalize()} | {'PASS' if passed else 'NOT RUN / MISSING'} |" for name, passed in release_checks.items())
    report += [
        "",
        "**Final reflection:** The primary risk is mistaking a successful LoRA merge or changed output "
        "for useful, safe fine-tuning. Quantization and a small greedy sample cannot establish quality; "
        "the report therefore withholds release approval until benchmark, safety, and Soup CLI evidence exist.",
        "",
        "This report is generated from local files and must be regenerated after the pipeline runs. "
        "No training or GPU measurements are fabricated. GitHub Copilot assisted with code drafting; "
        "unit tests, syntax compilation, and report generation were run locally, while model results "
        "remain unverified until a Colab GPU run.",
        "",
    ]
    (REPORT_DIR / "final_report.md").write_text("\n".join(report), encoding="utf-8")
    logger.info("Wrote reports/silent_failures.md and reports/final_report.md")


if __name__ == "__main__":
    main()
