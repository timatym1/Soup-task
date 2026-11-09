import json
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from src.common import (
    BASE_DIR,
    MERGED_DIR,
    QUANTIZED_DIR,
    REPORT_DIR,
    ensure_directories,
    get_logger,
    save_json,
)


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "tests" / "fixtures" / "model_evaluation_cases.json"
MODEL_PATHS = {
    "base": BASE_DIR,
    "merged": MERGED_DIR,
    "quantized_4bit": QUANTIZED_DIR,
}
REQUIRED_PACKAGES = ("torch", "transformers", "bitsandbytes")


def load_cases(path: Path = CASE_PATH) -> dict[str, Any]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, dict) or not cases.get("suite"):
        raise ValueError(f"Invalid evaluation suite: {path}")
    for category in ("quality_cases", "safety_cases"):
        rows = cases.get(category)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Evaluation suite must contain {category}.")
        identifiers = [row.get("id") for row in rows if isinstance(row, dict)]
        if len(identifiers) != len(rows) or any(not item for item in identifiers):
            raise ValueError(f"Every {category} case must be an object with an id.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Duplicate case id in {category}.")
        for row in rows:
            if not isinstance(row.get("prompt"), str) or not row.get("criteria"):
                raise ValueError(f"Case {row['id']} needs a string prompt and review criteria.")
    return cases


def model_artifact_errors(model_paths: dict[str, Path]) -> list[str]:
    errors = []
    for name, path in model_paths.items():
        if not (path / "config.json").is_file():
            errors.append(f"{name}: missing {path / 'config.json'}")
            continue
        weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
        if not weights:
            errors.append(f"{name}: no model weight files found in {path}")
    return errors


def build_report(
    suite: str,
    category: str,
    cases: list[dict[str, Any]],
    responses: dict[str, dict[str, str]],
    blockers: list[str],
) -> dict[str, Any]:
    model_names = tuple(MODEL_PATHS)
    if not blockers:
        for model in model_names:
            missing = {case["id"] for case in cases} - responses.get(model, {}).keys()
            if missing:
                raise ValueError(f"{model} is missing responses for: {sorted(missing)}")

    entries = []
    for case in cases:
        entry = dict(case)
        entry["responses"] = {
            model: responses.get(model, {}).get(case["id"]) for model in model_names
        }
        entry["human_review"] = (
            {"status": "PENDING", "assessment": None, "notes": None}
            if category == "quality"
            else {"status": "PENDING", "safe": None, "notes": None}
        )
        entries.append(entry)

    return {
        "schema_version": 1,
        "suite": suite,
        "category": category,
        "status": "BLOCKED" if blockers else "NEEDS_HUMAN_REVIEW",
        "blockers": blockers,
        "case_count": len(entries),
        "model_names": list(model_names),
        "cases": entries,
        "limitations": [
            "Model responses are evidence, not an automatic quality or safety judgment.",
            "Human review fields are deliberately left pending; no ratings are inferred.",
        ],
    }


def generate_responses(
    model_paths: dict[str, Path], prompts: list[str]
) -> dict[str, dict[str, str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    responses: dict[str, dict[str, str]] = {}
    for name, path in model_paths.items():
        tokenizer = AutoTokenizer.from_pretrained(path)
        if not tokenizer.chat_template:
            raise RuntimeError(f"No chat template in {path}")
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model_responses = {}
        for index, prompt in enumerate(prompts):
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            device = model.get_input_embeddings().weight.device
            encoded = encoded.to(device)
            with torch.inference_mode():
                generated = model.generate(
                    encoded,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=(
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                )
            model_responses[str(index)] = tokenizer.decode(
                generated[0, encoded.shape[1] :], skip_special_tokens=True
            )
            del encoded, generated
        responses[name] = model_responses
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return responses


def main() -> None:
    logger = get_logger("model_evaluation", "model_evaluation.log")
    ensure_directories()
    cases = load_cases()
    blockers = model_artifact_errors(MODEL_PATHS)
    blockers.extend(
        f"Required Python package is not installed: {package}"
        for package in REQUIRED_PACKAGES
        if find_spec(package) is None
    )
    if not any(reason.endswith("torch") for reason in blockers):
        import torch

        if not torch.cuda.is_available():
            blockers.append("CUDA GPU is unavailable; this evaluation requires the project GPU runtime.")

    responses: dict[str, dict[str, str]] = {}
    if not blockers:
        for category in ("quality", "safety"):
            save_json(
                REPORT_DIR / f"{category}_evaluation.json",
                build_report(
                    cases["suite"],
                    category,
                    cases[f"{category}_cases"],
                    {},
                    ["Model generation has not completed."],
                ),
            )
        all_cases = cases["quality_cases"] + cases["safety_cases"]
        prompts = [case["prompt"] for case in all_cases]
        indexed = generate_responses(MODEL_PATHS, prompts)
        for model, values in indexed.items():
            if len(values) != len(prompts):
                raise RuntimeError(
                    f"{model} generated {len(values)} responses for {len(prompts)} prompts."
                )
            responses[model] = {
                case["id"]: values[str(index)] for index, case in enumerate(all_cases)
            }

    reports = (
        (
            "quality_evaluation.json",
            build_report(cases["suite"], "quality", cases["quality_cases"], responses, blockers),
        ),
        (
            "safety_evaluation.json",
            build_report(cases["suite"], "safety", cases["safety_cases"], responses, blockers),
        ),
    )
    for filename, report in reports:
        destination = REPORT_DIR / filename
        save_json(destination, report)
        logger.info("Wrote %s with status %s", destination, report["status"])


if __name__ == "__main__":
    main()
