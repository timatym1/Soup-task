from __future__ import annotations

import argparse
from importlib.util import find_spec
import json
from pathlib import Path
from typing import Any

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
from src.model_evaluation import MODEL_PATHS, REQUIRED_PACKAGES, model_artifact_errors
from src.training_data import truncate_training_sequence


MAX_LENGTH = 512
LOSS_CHUNK_TOKENS = 16


def build_blocked_report(blockers: list[str]) -> dict[str, object]:
    return {
        "status": "BLOCKED",
        "prompt_count": 0,
        "blockers": blockers,
        "model_response_counts": {},
        "models": {},
    }


def load_prompts(path: Path, limit: int) -> list[str]:
    return [build_prompt(row) for row in load_records(path, limit)]


def build_prompt(row: dict[str, str]) -> str:
    content = row["instruction"]
    if row.get("input"):
        content += f"\n\nInput:\n{row['input']}"
    return content


def load_records(path: Path, limit: int) -> list[dict[str, str]]:
    if limit < 1:
        raise ValueError("limit must be greater than zero.")
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("instruction"), str):
                raise ValueError(f"Invalid test record at {path}, line {line_number}.")
            if not isinstance(row.get("output"), str) or not row["output"]:
                raise ValueError(f"Missing assistant output at {path}, line {line_number}.")
            records.append(row)
            if len(records) == limit:
                break
    if not records:
        raise RuntimeError(f"No prompts found in {path}")
    return records


def token_nll_sums(
    logits,
    labels,
    token_chunk_size: int = LOSS_CHUNK_TOKENS,
) -> tuple[float, int]:
    import torch
    import torch.nn.functional as functional

    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be greater than zero.")
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected logits [batch, sequence, vocab] and matching labels [batch, sequence].")

    targets = labels[:, 1:].reshape(-1)
    positions = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    supervised = torch.nonzero(targets != -100, as_tuple=False).flatten()
    if supervised.numel() == 0:
        raise ValueError("Evaluation sequence contains no supervised assistant tokens.")

    total_nll = 0.0
    for indices in supervised.split(token_chunk_size):
        selected_logits = positions.index_select(0, indices).float()
        selected_targets = targets.index_select(0, indices)
        total_nll += functional.cross_entropy(
            selected_logits, selected_targets, reduction="sum"
        ).item()
    return total_nll, int(supervised.numel())


def token_logit_drift(
    reference_logits,
    candidate_logits,
    labels,
    token_chunk_size: int = LOSS_CHUNK_TOKENS,
) -> dict[str, float | int]:
    import torch
    import torch.nn.functional as functional

    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be greater than zero.")
    if (
        reference_logits.ndim != 3
        or reference_logits.shape != candidate_logits.shape
        or labels.ndim != 2
        or reference_logits.shape[:2] != labels.shape
    ):
        raise ValueError("Expected matching logits [batch, sequence, vocab] and labels [batch, sequence].")

    targets = labels[:, 1:].reshape(-1)
    reference = reference_logits[:, :-1, :].reshape(-1, reference_logits.shape[-1])
    candidate = candidate_logits[:, :-1, :].reshape(-1, candidate_logits.shape[-1])
    supervised = torch.nonzero(targets != -100, as_tuple=False).flatten()
    if supervised.numel() == 0:
        raise ValueError("Evaluation sequence contains no supervised assistant tokens.")

    kl_sum = 0.0
    top1_matches = 0
    max_absolute_delta = 0.0
    for indices in supervised.split(token_chunk_size):
        reference_chunk = reference.index_select(0, indices.to(reference.device)).float()
        candidate_chunk = candidate.index_select(0, indices.to(candidate.device)).float()
        reference_log_probs = functional.log_softmax(reference_chunk, dim=-1)
        candidate_log_probs = functional.log_softmax(candidate_chunk, dim=-1)
        kl_sum += functional.kl_div(
            candidate_log_probs.to(reference_log_probs.device),
            reference_log_probs,
            reduction="sum",
            log_target=True,
        ).item()
        top1_matches += int(
            (
                reference_chunk.argmax(dim=-1)
                == candidate_chunk.argmax(dim=-1).to(reference_chunk.device)
            ).sum().item()
        )
        max_absolute_delta = max(
            max_absolute_delta,
            (reference_chunk - candidate_chunk.to(reference_chunk.device)).abs().max().item(),
        )

    token_count = int(supervised.numel())
    return {
        "assistant_token_count": token_count,
        "top1_match_count": top1_matches,
        "mean_kl_reference_to_candidate": kl_sum / token_count,
        "top1_agreement": top1_matches / token_count,
        "max_absolute_logit_delta": max_absolute_delta,
    }


def encode_evaluation_record(tokenizer, record: dict[str, str]) -> tuple[list[int], list[int]]:
    prompt_messages = [{"role": "user", "content": build_prompt(record)}]
    full_messages = prompt_messages + [{"role": "assistant", "content": record["output"]}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False, return_dict=False
    )
    return truncate_training_sequence(
        full_ids, prompt_ids, MAX_LENGTH, tokenizer.eos_token_id
    )


def nll_metrics(total_nll: float, token_count: int) -> dict[str, float | int]:
    import math

    if token_count < 1:
        raise ValueError("token_count must be greater than zero.")
    mean_nll = total_nll / token_count
    if mean_nll < 0 or not math.isfinite(mean_nll) or mean_nll > 709:
        raise ValueError("mean_nll must be finite, nonnegative, and representable as perplexity.")
    return {
        "assistant_token_count": token_count,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
    }


def validate_nf4_evidence(
    configured_load_in_4bit: bool,
    configured_quant_type: str | None,
    linear4bit_module_count: int,
    initialized_quant_state_count: int,
    runtime_quant_types: list[str],
) -> dict[str, Any]:
    if linear4bit_module_count < 0 or initialized_quant_state_count < 0:
        raise ValueError("Quantized module counts cannot be negative.")
    if initialized_quant_state_count > linear4bit_module_count:
        raise ValueError("Initialized quant-state count cannot exceed 4-bit module count.")
    quant_types = sorted(set(runtime_quant_types))
    valid = (
        configured_load_in_4bit
        and configured_quant_type == "nf4"
        and linear4bit_module_count > 0
        and initialized_quant_state_count == linear4bit_module_count
        and quant_types == ["nf4"]
    )
    return {
        "status": "PASS" if valid else "FAIL",
        "configured_load_in_4bit": configured_load_in_4bit,
        "configured_quant_type": configured_quant_type or "UNKNOWN",
        "linear4bit_module_count": linear4bit_module_count,
        "initialized_quant_state_count": initialized_quant_state_count,
        "runtime_quant_types": quant_types,
    }


def quantization_evidence(model) -> dict[str, Any]:
    from bitsandbytes.nn import Linear4bit

    configuration = getattr(model.config, "quantization_config", None)
    if hasattr(configuration, "to_dict"):
        configuration = configuration.to_dict()
    if not isinstance(configuration, dict):
        configuration = {}
    modules = [
        module
        for module in model.modules()
        if isinstance(module, Linear4bit)
    ]
    quant_types = [
        module.weight.quant_state.quant_type
        for module in modules
        if module.weight.quant_state is not None
    ]
    return validate_nf4_evidence(
        configured_load_in_4bit=configuration.get("load_in_4bit") is True,
        configured_quant_type=configuration.get("bnb_4bit_quant_type"),
        linear4bit_module_count=len(modules),
        initialized_quant_state_count=len(quant_types),
        runtime_quant_types=quant_types,
    )


def run_model(
    model_path: Path, records: list[dict[str, str]], tokenizer_path: Path = BASE_DIR
) -> tuple[
    list[str],
    object,
    dict[str, int] | None,
    dict[str, float | int],
    dict[str, Any] | None,
]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if not tokenizer.chat_template:
        raise RuntimeError(f"No chat template in {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        raise RuntimeError(
            f"Tokenizer vocabulary does not match model embeddings for {model_path}."
        )
    model.eval()
    outputs = []
    last_logits = None
    total_nll = 0.0
    total_tokens = 0
    for record in records:
        prompt = build_prompt(record)
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_device = model.get_input_embeddings().weight.device
        input_ids = encoded["input_ids"].to(input_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_device)

        with torch.inference_mode():
            model_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = model_output.logits
            last_logits = logits[0, -1].float().cpu()
            del model_output, logits
            eval_ids, eval_labels = encode_evaluation_record(tokenizer, record)
            eval_input_ids = torch.tensor([eval_ids], device=input_device)
            eval_label_ids = torch.tensor([eval_labels], device=input_device)
            eval_attention_mask = torch.ones_like(eval_input_ids)
            eval_logits = model(
                input_ids=eval_input_ids,
                attention_mask=eval_attention_mask,
            ).logits
            nll, token_count = token_nll_sums(eval_logits, eval_label_ids)
            total_nll += nll
            total_tokens += token_count
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )

        outputs.append(
            tokenizer.decode(
                generated[0, input_ids.shape[1]:],
                skip_special_tokens=True,
            )
        )
        del (
            encoded,
            input_ids,
            attention_mask,
            eval_input_ids,
            eval_label_ids,
            eval_attention_mask,
            eval_logits,
            generated,
        )
    quantization = (
        quantization_evidence(model) if model_path == QUANTIZED_DIR else None
    )
    memory = cuda_memory()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if last_logits is None:
        raise RuntimeError(f"No prompts were evaluated for {model_path}.")
    return (
        outputs,
        last_logits,
        memory,
        nll_metrics(total_nll, total_tokens),
        quantization,
    )


def compare_merged_quantized(
    records: list[dict[str, str]],
) -> tuple[dict[str, float | int], dict[str, int] | None]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)
    if not tokenizer.chat_template:
        raise RuntimeError(f"No chat template in {BASE_DIR}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    merged = AutoModelForCausalLM.from_pretrained(
        MERGED_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    quantized = AutoModelForCausalLM.from_pretrained(
        QUANTIZED_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if (
        merged.get_input_embeddings().num_embeddings != len(tokenizer)
        or quantized.get_input_embeddings().num_embeddings != len(tokenizer)
    ):
        raise RuntimeError("Base tokenizer vocabulary does not match model embeddings.")
    merged.eval()
    quantized.eval()
    total_kl = 0.0
    total_tokens = 0
    total_top1_matches = 0
    maximum_logit_delta = 0.0
    try:
        with torch.inference_mode():
            for record in records:
                input_ids, label_ids = encode_evaluation_record(tokenizer, record)
                merged_device = merged.get_input_embeddings().weight.device
                quantized_device = quantized.get_input_embeddings().weight.device
                merged_ids = torch.tensor([input_ids], device=merged_device)
                quantized_ids = torch.tensor([input_ids], device=quantized_device)
                labels = torch.tensor([label_ids], device=merged_device)
                merged_mask = torch.ones_like(merged_ids)
                quantized_mask = torch.ones_like(quantized_ids)
                merged_logits = merged(
                    input_ids=merged_ids, attention_mask=merged_mask
                ).logits
                quantized_logits = quantized(
                    input_ids=quantized_ids, attention_mask=quantized_mask
                ).logits
                metrics = token_logit_drift(merged_logits, quantized_logits, labels)
                count = metrics["assistant_token_count"]
                total_kl += metrics["mean_kl_reference_to_candidate"] * count
                total_top1_matches += metrics["top1_match_count"]
                total_tokens += count
                maximum_logit_delta = max(
                    maximum_logit_delta, metrics["max_absolute_logit_delta"]
                )
                del (
                    merged_ids,
                    quantized_ids,
                    labels,
                    merged_mask,
                    quantized_mask,
                    merged_logits,
                    quantized_logits,
                )
    finally:
        del merged, quantized
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not total_tokens:
        raise RuntimeError("No supervised assistant tokens were evaluated for logit drift.")
    memory = cuda_memory()
    return (
        {
            "assistant_token_count": total_tokens,
            "top1_match_count": total_top1_matches,
            "mean_kl_reference_to_candidate": total_kl / total_tokens,
            "top1_agreement": total_top1_matches / total_tokens,
            "max_absolute_logit_delta": maximum_logit_delta,
        },
        memory,
    )


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    import torch

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
    if args.samples < 1:
        parser.error("--samples must be greater than zero.")

    data_path = Path(__file__).resolve().parents[1] / "data" / "test.jsonl"
    blockers = model_artifact_errors(MODEL_PATHS)
    if not data_path.is_file():
        blockers.append("Test data is missing: data/test.jsonl")
    blockers.extend(
        f"Required Python package is not installed: {package}"
        for package in REQUIRED_PACKAGES
        if find_spec(package) is None
    )
    if find_spec("torch") is not None:
        import torch

        if not torch.cuda.is_available():
            blockers.append("CUDA GPU is unavailable; verification requires the project GPU runtime.")
    if blockers:
        report = build_blocked_report(blockers)
        destination = REPORT_DIR / "verification.json"
        save_json(destination, report)
        logger.info("Verification blocked; report saved to %s", destination)
        return

    records = load_records(data_path, args.samples)
    prompts = [build_prompt(record) for record in records]
    model_specs = [("base", BASE_DIR), ("merged", MERGED_DIR), ("quantized_4bit", QUANTIZED_DIR)]
    results = {}
    logits_by_model = {}
    for name, path in model_specs:
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Missing model config: {path / 'config.json'}")
        texts, logits, memory, heldout_nll, quantization = run_model(
            path, records, tokenizer_path=BASE_DIR
        )
        results[name] = {
            "responses": texts,
            "gpu_peak": memory,
            "heldout_nll": heldout_nll,
        }
        if quantization is not None:
            results[name]["quantization"] = quantization
        logits_by_model[name] = logits
        logger.info("Generated %d deterministic responses for %s", len(texts), name)
    quantization_drift, drift_memory = compare_merged_quantized(records)

    base_text = results["base"]["responses"]
    merged_text = results["merged"]["responses"]
    quant_text = results["quantized_4bit"]["responses"]
    report = {
        "status": "COMPLETED",
        "verification_scope": "generation and logit comparison only; not a quality or safety assessment",
        "quality_assessment": "NOT_EVALUATED",
        "prompt_count": len(prompts),
        "tokenizer": str(BASE_DIR.relative_to(Path(__file__).resolve().parents[1])),
        "model_response_counts": {
            name: len(result["responses"]) for name, result in results.items()
        },
        "base_vs_merged_changed_responses": sum(a != b for a, b in zip(base_text, merged_text)),
        "merged_vs_quantized_changed_responses": sum(a != b for a, b in zip(merged_text, quant_text)),
        "base_vs_merged_last_prompt_logits": compare_logits(logits_by_model["base"], logits_by_model["merged"]),
        "merged_vs_quantized_last_prompt_logits": compare_logits(logits_by_model["merged"], logits_by_model["quantized_4bit"]),
        "merged_vs_quantized_full_answer_logit_drift": quantization_drift,
        "quantization_drift_gpu_peak": drift_memory,
        "models": results,
        "limitations": [
            "A small greedy-generation sample is not a quality or safety benchmark.",
            "Held-out NLL is token-weighted over the evaluated test records; perplexity is exp(mean NLL).",
            "The sample count follows --samples; use --samples 200 to evaluate the full held-out split.",
            "The legacy single-position logit comparison uses only the last prompt and last-token logits.",
            "Full-answer KL and top-1 agreement compare merged against quantized logits at supervised answer-token positions only.",
            "NF4 evidence is collected after forward passes initialize the runtime quantization state.",
            "Changed outputs/logits confirm a difference, not that the change is beneficial.",
            "COMPLETED means all requested model comparisons ran; it does not mean model quality or safety passed.",
        ],
    }
    destination = REPORT_DIR / "verification.json"
    save_json(destination, report)
    logger.info("Verification report saved to %s", destination)


if __name__ == "__main__":
    main()
