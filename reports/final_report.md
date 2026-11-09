# Final Technical Report

**Final status: DON'T SHIP.**

**Generated:** 2026-11-09T20:39:34+00:00

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`. **Data:** `not available`. **Data status:** `BLOCKED`.

**Quality evaluation:** `BLOCKED`; 6 independent cases; ratings are not inferred without review.

**Safety evaluation:** `BLOCKED`; 4 cases; safety judgments are not inferred without review.

**Verification:** `BLOCKED`; not run prompts.

**Soup CLI:** `BLOCKED` - The executable is not installed, and no Soup CLI command/subcommand contract is documented.

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
python -m unittest discover -s tests -v
if [ ! -f data/train.jsonl ] || [ ! -f data/test.jsonl ] || [ ! -f reports/data_validation.json ] || ! python -c "import json; assert json.load(open('reports/data_validation.json', encoding='utf-8')).get('status') == 'PASS'"; then python -m src.data_preparation; fi
python -m src.memory_analysis
python -m src.verification --samples 20
python -m src.model_evaluation
python -m src.silent_failures
```

Soup CLI remains blocked: The executable is not installed, and no Soup CLI command/subcommand contract is documented. No Soup command is run or guessed.

The working copy has no model weights, training data, or training logs; installed local dependencies: torch=True, transformers=False, bitsandbytes=False. No training or inference was claimed.
