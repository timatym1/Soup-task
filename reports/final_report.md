# Final Technical Report

**Final status: DON'T SHIP.**

**Generated:** 2026-11-10T15:08:00+00:00

Current checkout: `Qwen/Qwen2.5-0.5B-Instruct` on `tatsu-lab/alpaca`; data status `BLOCKED`. Missing local artifacts do not invalidate historical results.

## Historical run

Summary from [66c591c4aed2485a41f038525915f88e572e6c94]: 1800 train / 200 test examples; train loss 1.4922131479313943, eval loss 1.474936842918396.

Verification changed 17/20 base-to-merged and 16/20 merged-to-4-bit responses; single-position max logit deltas 8.1328125 and 4.1796875.

Current held-out verification: `BLOCKED`; 0 records; not run.

Current merged-to-4-bit full-answer drift: not run.

Memory: training estimate 1.295 GiB; measured allocated/reserved 2.518/11.42 GiB; merge allocated 0.946 GiB; 4-bit allocated 0.459 GiB.

Evidence limit: these values are transcribed from the tracked summary in commit `66c591c4aed2485a41f038525915f88e572e6c94`. That commit does not contain raw logs, nvidia-smi output, model weights, or split files; this checkout cannot independently reproduce them.

## Current release gates

| Gate | Result |
|---|---|
| Data validation | BLOCKED |
| Training | BLOCKED |
| Merge | BLOCKED |
| Quantization | BLOCKED |
| Verification | BLOCKED |
| Quality review | BLOCKED |
| Safety review | BLOCKED |
| Gpu measurements | BLOCKED |
| Memory estimate | BLOCKED |
| Dataset provenance | BLOCKED |
| Soup cli | FAIL |
| System doctor | FAIL |
| Data doctor | BLOCKED |
| Preflight | BLOCKED |
| Ship | BLOCKED |

## What surprised / what worries me

The historical training reserved-memory peak was about 8.8x the estimate, while the 4-bit logit delta was the same order as the base-to-merged delta. The latter is a single position, not a quality metric; neither it nor changed greedy responses proves improvement. Held-out full-token drift, null-adapter/shuffled-label controls, raw GPU logs, near-duplicate audit of the historical split, and human quality/safety reviews remain absent.

## AI use

Copilot-assisted code was reviewed against the repository and regression tests. This change removes unsupported report-driven verdicts, adds deterministic sampling, leakage and prompt-prefix checks, and corrects memory-estimate assumptions; GPU-dependent claims were not re-run.

## Current blockers

- train_data: missing under the current repository root
- test_data: missing under the current repository root
- base_model: missing under the current repository root
- adapter: missing under the current repository root
- merged_model: missing under the current repository root
- quantized_model: missing under the current repository root
- training_log: missing under the current repository root
- nvidia_smi_log: missing under the current repository root
- 20 additional blockers are listed in `reports/release_decision.json`.

## Recheck

```bash
python -m unittest discover -s tests -v
python -m src.data_validation
python -m src.verification --samples 200
python -m src.model_evaluation
python -m src.silent_failures
```

Do not train or claim release readiness until original artifacts are restored, GPU checks are rerun, and every quality/safety case is reviewed.