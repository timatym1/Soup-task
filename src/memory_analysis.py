import argparse

from src.common import BASE_DIR, MODEL_ID, REPORT_DIR, ensure_directories, get_logger, save_json


def estimate_parameters(config: object) -> tuple[int, int]:
    hidden = config.hidden_size
    layers = config.num_hidden_layers
    heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", heads)
    intermediate = config.intermediate_size
    vocab = config.vocab_size
    head_dim = hidden // heads

    attention = layers * (
        hidden * hidden + 2 * hidden * kv_heads * head_dim + hidden * hidden
    )
    gated_mlp = layers * 3 * hidden * intermediate
    embeddings = vocab * hidden * (1 if getattr(config, "tie_word_embeddings", False) else 2)
    total = attention + gated_mlp + embeddings
    lora_trainable = layers * 8 * (3 * hidden + kv_heads * head_dim)
    return total, lora_trainable


def build_estimate(config: object, batch_size: int, sequence_length: int) -> dict[str, object]:
    if batch_size < 1 or sequence_length < 1:
        raise ValueError("batch_size and sequence_length must be greater than zero.")
    params, lora_params = estimate_parameters(config)
    bytes_per_weight = 2
    weight_bytes = params * bytes_per_weight
    activation_bytes = (
        config.num_hidden_layers * batch_size * sequence_length * config.hidden_size * 2 * 2
    )
    logits_bytes = batch_size * sequence_length * config.vocab_size * 4
    loss_workspace_bytes = logits_bytes
    adapter_and_optimizer = lora_params * (4 + 4 + 2)
    train_estimate = (
        weight_bytes
        + activation_bytes
        + logits_bytes
        + loss_workspace_bytes
        + adapter_and_optimizer
    )
    return {
        "parameter_count_estimate": params,
        "lora_parameter_count_estimate": lora_params,
        "assumptions": {
            "base_weights": "FP16",
            "lora_rank": 8,
            "lora_targets": ["q_proj", "v_proj"],
            "optimizer": (
                "FP32 adapter weights and gradients + two 8-bit Adam moments; "
                "excludes blockwise scale overhead"
            ),
            "activations": "rough checkpointed-layer-input estimate; excludes CUDA/workspace overhead",
            "logits": "FP32 logits for causal-language-model loss",
            "loss_workspace": (
                "one additional FP32-sized allowance for loss/backward workspace; "
                "kernel-dependent softmax allocation may differ"
            ),
            "sequence_length": sequence_length,
            "micro_batch_size": batch_size,
        },
        "training_estimate_bytes": {
            "weights": weight_bytes,
            "activations": activation_bytes,
            "logits": logits_bytes,
            "loss_workspace": loss_workspace_bytes,
            "adapter_and_optimizer_states": adapter_and_optimizer,
            "total": train_estimate,
            "total_gib": round(train_estimate / (1024**3), 3),
        },
        "merge_peak_estimate_gib": round(2 * weight_bytes / (1024**3), 3),
        "quantization_peak_estimate_gib": round(
            (weight_bytes + params // 2) / (1024**3), 3
        ),
        "estimate_warning": (
            "Approximate parameter and memory model, not a capacity guarantee. "
            "Compare with timestamped CUDA and nvidia-smi measurements."
        ),
    }


def main() -> None:
    from transformers import AutoConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=512)
    args = parser.parse_args()
    logger = get_logger("memory_analysis", "memory_analysis.log")
    ensure_directories()
    config = AutoConfig.from_pretrained(str(BASE_DIR) if BASE_DIR.exists() else MODEL_ID)
    result = build_estimate(config, args.batch_size, args.sequence_length)
    if result["parameter_count_estimate"] > 3_000_000_000:
        raise RuntimeError("Configured model exceeds the 3B-parameter project limit.")
    result["status"] = "ESTIMATE_ONLY"
    destination = REPORT_DIR / "memory_budget.json"
    save_json(destination, result)
    logger.info("Memory estimate saved to %s: %s GiB training estimate", destination, result["training_estimate_bytes"]["total_gib"])


if __name__ == "__main__":
    main()
