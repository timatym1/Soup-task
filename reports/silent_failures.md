/content/Soup-task
# Silent Failures Analysis

| Area | Potential silent failure | Detection/evidence | Local result |
|---|---|---|---|
| Data | Empty/malformed rows and duplicates | data_validation.json records validation counts; at least 1000 unique valid rows required | **PASS** |
| Tokenizer/template | Missing chat template or untrainable tokenized rows | Training aborts on missing template/empty supervised labels; saved training metrics prove the run reached completion | **PASS** |
| Training | No effective parameter update | Require saved training metrics and compare base/merged last-token logits | **PASS** |
| Merge | Adapter not merged or corrupted output | Merged config and weights present; compare merged logits with base | **PASS** |
| Quantization | 4-bit artifact missing or excessive output drift | Quantized config/weights present; compare against merged on identical prompts | **PASS** |
| Inference | Load/generation failure | All three models loaded and generated the same test prompts | **PASS** |
| GPU/memory | VRAM estimate differs from real peak | Compare memory_budget.json, per-stage GPU metrics and timestamped nvidia-smi | **PASS** |
| Soup CLI | Soup Doctor/Pre-flight/Ship findings | Not invoked: CLI subcommand contract is not supplied; no detections are claimed | **NOT INSTALLED** |

A PASS means the stated local check ran; it is not a model-quality or release approval. Soup Data Doctor, Pre-flight and Ship results remain unverified unless separately run with the actual CLI.
