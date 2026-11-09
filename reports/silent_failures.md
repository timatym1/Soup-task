# Silent Failures Analysis

**Generated:** 2026-11-09T20:39:34+00:00

| Area | Potential silent failure | Evidence required | Result |
|---|---|---|---|
| Data | Invalid, duplicated, or empty examples; split not written | Passing validation report and on-disk train/test JSONL | BLOCKED |
| Training | No completed update or adapter missing | Training metrics and adapter configuration | BLOCKED |
| Merge | Adapter not merged or output weights missing | Merged configuration, weights, and merge metrics | BLOCKED |
| Quantization | 4-bit artifact missing or not measured | Quantized configuration, weights, and quantization metrics | BLOCKED |
| Verification | One or more models did not generate every prompt | Explicit PASS and per-model response counts matching prompt count | BLOCKED |
| Quality | Output relevance, correctness, or hallucination risk unreviewed | Separate evaluation set, all model responses, and completed human assessments | BLOCKED |
| Safety | Unsafe compliance or malformed-input handling unreviewed | Separate safety set, all model responses, and completed human assessments | BLOCKED |
| GPU/memory | Missing GPU measurements or runtime log | CUDA peaks for train/merge/quantization and nvidia-smi log | BLOCKED |
| Soup CLI | Unknown or unavailable command contract | Installed CLI and documented, authorized command specification | BLOCKED |
| Memory estimate | Estimate absent or mistaken for measured usage | Memory estimate explicitly distinguished from recorded CUDA peaks | BLOCKED |

BLOCKED means evidence is unavailable or the specified review was not completed; it is not a PASS. A model loading or output-difference check alone does not establish quality, safety, or acceptable quantization drift.
