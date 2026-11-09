import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import (
    BASE_DIR,
    MERGED_DIR,
    QUANTIZED_DIR,
    REPORT_DIR,
    cuda_memory,
    ensure_directories,
    get_logger,
    save_json,
)


def load_prompts(path: Path, limit: int) -> list[str]:
    prompts = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            content = row["instruction"]
            if row.get("input"):
                content += f"\n\nInput:\n{row['input']}"
            prompts.append(content)
            if len(prompts) == limit:
                break
    if not prompts:
        raise RuntimeError(f"No prompts found in {path}")
    return prompts


def run_model(model_path: Path, prompts: list[str]) -> tuple[list[str], torch.Tensor, dict[str, int] | None]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.chat_template:
        raise RuntimeError(f"No chat template in {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    model.eval()
    outputs = []
    last_logits = None
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_device = model.get_input_embeddings().weight.device
        encoded = encoded.to(input_device)
        with torch.inference_mode():
            logits = model(input_ids=encoded).logits[0, -1].float().cpu()
            generated = model.generate(
                encoded,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        last_logits = logits
        outputs.append(tokenizer.decode(generated[0, encoded.shape[1]:], skip_special_tokens=True))
    memory = cuda_memory()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs, last_logits, memory


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    reference = reference.float()
    candidate = candidate.float()
    delta = candidate - reference
    cosine = torch.nn.functional.cosine_similarity(reference, candidate, dim=0).item()
    return {
        "max_absolute_delta": float(delta.abs().max().item()),
        "relative_l2_delta": float(
            delta.norm().item() / max(reference.norm().item(), 1e-12)
        ),
        "cosine_similarity": float(cosine),
        "argmax_unchanged": bool(reference.argmax().item() == candidate.argmax().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    logger = get_logger("verification", "verification.log")
    ensure_directories()
    prompts = load_prompts(Path(__file__).resolve().parents[1] / "data" / "test.jsonl", args.samples)
    model_specs = [("base", BASE_DIR), ("merged", MERGED_DIR), ("quantized_4bit", QUANTIZED_DIR)]
    results = {}
    logits_by_model = {}
    for name, path in model_specs:
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Missing model config: {path / 'config.json'}")
        texts, logits, memory = run_model(path, prompts)
        results[name] = {"responses": texts, "gpu_peak": memory}
        logits_by_model[name] = logits
        logger.info("Generated %d deterministic responses for %s", len(texts), name)

    base_text = results["base"]["responses"]
    merged_text = results["merged"]["responses"]
    quant_text = results["quantized_4bit"]["responses"]
    report = {
        "prompt_count": len(prompts),
        "base_vs_merged_changed_responses": sum(a != b for a, b in zip(base_text, merged_text)),
        "merged_vs_quantized_changed_responses": sum(a != b for a, b in zip(merged_text, quant_text)),
        "base_vs_merged_last_prompt_logits": compare_logits(logits_by_model["base"], logits_by_model["merged"]),
        "merged_vs_quantized_last_prompt_logits": compare_logits(logits_by_model["merged"], logits_by_model["quantized_4bit"]),
        "models": results,
        "limitations": [
            "A small greedy-generation sample is not a quality or safety benchmark.",
            "Logit comparison uses only the last prompt and last-token logits.",
            "Changed outputs/logits confirm a difference, not that the change is beneficial.",
        ],
    }
    destination = REPORT_DIR / "verification.json"
    save_json(destination, report)
    logger.info("Verification report saved to %s", destination)


if __name__ == "__main__":
    main()
