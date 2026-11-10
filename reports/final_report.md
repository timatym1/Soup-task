# Final Technical Report

**Final status: DON'T SHIP.**

**Generated:** 2026-11-10T04:39:49+00:00

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`. **Dataset:** `tatsu-lab/alpaca`. **Data status:** `BLOCKED`.

**Quality evaluation:** project-reported `FAIL`; current local evidence `BLOCKED`; 9 independent cases; ratings are not inferred without review.

**Safety evaluation:** project-reported `FAIL`; current local evidence `BLOCKED`; 6 cases; safety judgments are not inferred without review.

**Verification:** `BLOCKED`; 0 prompts.

**Soup CLI in this workspace:** `BLOCKED`; System Doctor reported missing training extras. Data Doctor: File not found: data\test.jsonl. Plan returned no executable preflight evidence: dataset hash is zero or model config is unavailable. Ship: Error: evidence has unsupported field(s): 'blockers', 'case_count', 'cases', 'category', 'limitations', 'model_names', 'schema_version', 'status', 'suite'.

- Current `soup.yaml`: `PRESENT`; train/test data: `MISSING` / `MISSING`.
- Current model artifacts: base `MISSING`, adapter `MISSING`, merged `MISSING`, quantized `MISSING`; training log `MISSING`, nvidia-smi log `MISSING`.
- Current System Doctor: exit 0; isolated Soup venv reported missing training extras.
- Current Data Doctor: exit 1 (File not found: data\test.jsonl).
- Current Plan: exit 0; command completed, but its output reports `dataset_sha` as all zeroes and model VRAM as unknown because required data/model files are missing.
- Current offline Ship: exit 3 (Error: evidence has unsupported field(s): 'blockers', 'case_count', 'cases', 'category', 'limitations', 'model_names', 'schema_version', 'status', 'suite'; no verdict produced).
- The previous Plan attempt could not find `soup.yaml`; running from this checkout now finds the config. The remaining Plan evidence is blocked by absent data/model artifacts.
- Did not install Soup training extras: the isolated Doctor requests Transformers >=5.16.1, while this project pins Transformers <5; the project Python environment also lacks its declared ML dependencies.
- Official Soup documentation describes offline Ship evidence as numeric `task.mode/base/tuned` metrics plus optional numeric benchmark `base/tuned` pairs. Those scores are unavailable here, so no evidence adapter or verdict was fabricated.
- Official references: [Soup CLI](https://github.com/MakazhanAlpamys/Soup), [PyPI](https://pypi.org/project/soup-cli/), [Data Doctor](https://github.com/MakazhanAlpamys/Soup/blob/main/docs/data.md), [Ship](https://github.com/MakazhanAlpamys/Soup/blob/main/docs/evaluation.md).

| Gate | Result |
|---|---|
| Data validation | BLOCKED |
| Training | BLOCKED |
| Merge | BLOCKED |
| Quantization | BLOCKED |
| Verification | BLOCKED |
| Quality review | FAIL |
| Safety review | FAIL |
| Gpu measurements | BLOCKED |
| Memory estimate | BLOCKED |
| Dataset provenance | BLOCKED |
| Soup cli | BLOCKED |
| System doctor | FAIL |
| Data doctor | BLOCKED |
| Preflight | BLOCKED |
| Ship | BLOCKED |

## Project-reported model concerns

- Quality: False-premise handling reportedly fails.
- Quality: Unknown personal facts reportedly trigger fabricated answers.
- Safety: Weapon-construction requests reportedly received unsafe answers.

These concerns were supplied by the project context and could not be reproduced here because model artifacts are unavailable.

## Current blockers

- train_data: missing under the current repository root
- test_data: missing under the current repository root
- base_model: missing under the current repository root
- adapter: missing under the current repository root
- merged_model: missing under the current repository root
- quantized_model: missing under the current repository root
- training_log: missing under the current repository root
- nvidia_smi_log: missing under the current repository root
- Required package is unavailable: transformers
- Required package is unavailable: peft
- Required package is unavailable: datasets
- Required package is unavailable: bitsandbytes
- Required package is unavailable: accelerate
- Provenance train: File is missing.
- Provenance test: File is missing.
- base: missing models/base/config.json
- merged: missing models/merged/config.json
- quantized_4bit: missing models/quantized/config.json
- Required Python package is not installed: transformers
- Required Python package is not installed: bitsandbytes
- CUDA GPU is unavailable; this evaluation requires the project GPU runtime.
- Project-reported quality failure: False-premise handling reportedly fails.
- Project-reported quality failure: Unknown personal facts reportedly trigger fabricated answers.
- Project-reported safety failure: Weapon-construction requests reportedly received unsafe answers.
- Soup CLI System Doctor found missing training extras.
- Soup plan exited successfully but reported a zero dataset hash and unknown model VRAM.
- Soup Data Doctor: File not found: data\test.jsonl
- Soup Ship: Error: evidence has unsupported field(s): 'blockers', 'case_count', 'cases', 'category', 'limitations', 'model_names', 'schema_version', 'status', 'suite'
- Soup CLI: System Doctor reported missing training extras. Data Doctor: File not found: data\test.jsonl. Plan returned no executable preflight evidence: dataset hash is zero or model config is unavailable. Ship: Error: evidence has unsupported field(s): 'blockers', 'case_count', 'cases', 'category', 'limitations', 'model_names', 'schema_version', 'status', 'suite'.
- Soup CLI data_doctor: exit 1
- Soup CLI ship_offline: exit 3

## Colab recheck commands

Use a CUDA-enabled Colab runtime and restore the existing `models/`, `logs/`, and, if available, `data/` outputs (including per-stage metrics). Missing evidence remains BLOCKED. These commands do not train, merge, or quantize the model.

```bash
python -m pip install -r requirements.txt
python -m pip install soup-cli
python -m unittest discover -s tests -v
python -m src.data_validation
python -m src.memory_analysis
soup --version
soup --help
soup doctor
soup data doctor --help
soup data doctor data/test.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --max-length 512
python -c "import subprocess,uuid; from pathlib import Path; p=Path('.soup-plan-'+uuid.uuid4().hex+'.tfstate'); r=subprocess.run(['soup','plan','--config','soup.yaml','--state',str(p)]); p.unlink(missing_ok=True); raise SystemExit(r.returncode)"
soup ship --help
# Run only with restored model artifacts and real evaluation data.
soup ship --base models/base --adapter models/adapter --task-eval "$EVAL_JSONL"
python -m src.verification --samples 20
python -m src.model_evaluation
python -m src.silent_failures
```

Data Doctor, Plan, and Ship commands must be checked against the installed CLI's help. Do not synthesize Ship metrics or treat command success as a quality/safety pass.

Training and inference were not run in this workspace. Artifacts are marked missing only relative to this repository root; this does not establish whether they exist in Google Drive. Local dependencies: torch=True, transformers=False, peft=False, trl=False, datasets=False, bitsandbytes=False, accelerate=False.
