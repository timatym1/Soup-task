/content/Soup-task
# Technical Report

**Final verdict: DON'T SHIP.**

**Generated:** 2026-11-09T20:01:33+00:00

**Execution environment:** PyTorch installed; nvidia-smi available.

**Configuration:** `tatsu-lab/alpaca`; base model `Qwen/Qwen2.5-0.5B-Instruct`; seed `42`; 1800 train / 200 test examples.

**Training:** train loss `1.4922131479313943`; evaluation loss `1.474936842918396`.

**Verification:** `{"prompts": 20, "changed_base_to_merged_responses": 17, "changed_merged_to_4bit_responses": 16, "base_to_merged_last_token_max_abs_delta": 8.1328125, "merged_to_4bit_last_token_max_abs_delta": 4.1796875}`.

**Memory:** `{"estimated_training_gib": 1.295, "measured_training_peak_bytes": {"allocated_bytes": 2704173568, "reserved_bytes": 12262047744}, "measured_merge_peak_bytes": {"allocated_bytes": 1015474688, "reserved_bytes": 1025507328}, "measured_4bit_peak_bytes": {"allocated_bytes": 492984320, "reserved_bytes": 524288000}, "training_reserved_to_estimate_ratio": 8.819}`.

The release gate requires all three model artifacts, training and GPU evidence, three-model verification, a data-validation report, a memory estimate, Soup CLI checks, and independent quality/safety review. Missing evidence is reported as not run; it is not inferred as a pass.

| Gate | Result |
|---|---|
| Base | PASS |
| Adapter | PASS |
| Data validation | PASS |
| Memory estimate | PASS |
| Training log | PASS |
| Merged | PASS |
| Quantized | PASS |
| Verification | PASS |
| Gpu log | PASS |
| Soup checks | NOT RUN / MISSING |
| Quality and safety review | NOT RUN / MISSING |

**Final reflection:** The primary risk is mistaking a successful LoRA merge or changed output for useful, safe fine-tuning. Quantization and a small greedy sample cannot establish quality; the report therefore withholds release approval until benchmark, safety, and Soup CLI evidence exist.

This report is generated from local files and must be regenerated after the pipeline runs. No training or GPU measurements are fabricated. GitHub Copilot assisted with code drafting; unit tests, syntax compilation, and report generation were run locally, while model results remain unverified until a Colab GPU run.
