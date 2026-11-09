# Final Technical Report

**Final status: DON'T SHIP.**

**Generated:** 2026-11-09T21:09:33+00:00

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`. **Data:** `not available`. **Data status:** `BLOCKED`.

**Quality evaluation:** `BLOCKED`; 6 independent cases; ratings are not inferred without review.

**Safety evaluation:** `BLOCKED`; 4 cases; safety judgments are not inferred without review.

**Verification:** `BLOCKED`; not run prompts.

**Soup CLI:** Official `soup-cli` v0.77.2 installed from PyPI (project: https://github.com/MakazhanAlpamys/Soup). Data Doctor, Pre-flight, and Ship were attempted but BLOCKED; see `reports/soup_cli_evaluation.json`.

- System Doctor: exit 0; issue found: training extras are missing.
- Data Doctor: `BLOCKED` (exit 1; File not found: data\test.jsonl).
- Pre-flight (`soup plan`): `BLOCKED` (exit 1; Config not found: soup.yaml).
- Ship: `BLOCKED` (exit 3; Error: evidence has unsupported field(s): 'blockers', 'case_count', 'cases', 'category', 'limitations', 'model_names', 'schema_version', 'status', 'suite'; no verdict produced).
- Official references: [Soup CLI](https://github.com/MakazhanAlpamys/Soup), [PyPI](https://pypi.org/project/soup-cli/), [Data Doctor](https://github.com/MakazhanAlpamys/Soup/blob/main/docs/data.md), [Ship](https://github.com/MakazhanAlpamys/Soup/blob/main/docs/evaluation.md).

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
| Soup cli | BLOCKED |

## Colab recheck commands

Use a CUDA-enabled Colab runtime and restore the existing `models/`, `logs/`, and, if available, `data/` outputs (including per-stage metrics). Missing evidence remains BLOCKED. These commands do not train, merge, or quantize the model.

```bash
python -m pip install -r requirements.txt
python -m pip install soup-cli
python -m unittest discover -s tests -v
python -m src.data_validation
python -m src.memory_analysis
soup doctor
soup data doctor data/test.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --sample 20
soup plan --config soup.yaml
# Set EVAL_JSONL to a real measured task-evaluation JSONL; do not fabricate scores.
soup ship --base models/base --adapter models/adapter --task-eval "$EVAL_JSONL"
python -m src.verification --samples 20
python -m src.model_evaluation
python -m src.silent_failures
```

The Ship command above requires a real prompt/scoring file; do not generate numeric evidence without actual model runs.

The working copy has no model weights, training data, or training logs; installed local dependencies: torch=True, transformers=False, bitsandbytes=False. No training or inference was claimed.
