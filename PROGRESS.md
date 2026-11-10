# Project progress

**As of:** 2026-11-10
**Release readiness:** **0 of 8 release gates currently confirmed passing (0%).**

This percentage measures only current release gates: data validation, training, merge, quantization, verification, quality, safety, and Soup CLI. It is not a percentage of code written or historical project work. All eight are currently blocked or failed by missing local evidence and previously reported concerns.

## Completed

- Reviewed pipeline modules, configuration, reports, tests, Git history, and tracked-file inventory.
- Fixed the Soup CLI release aggregate gate, which previously remained hard-coded as `BLOCKED` even after all required CLI checks passed.
- Added a regression test covering passing, missing, failing, and blocked Soup CLI checks.
- Added GitHub Actions checks for unit tests and source compilation. The workflow is not yet confirmed on GitHub because no push was requested or performed.
- Changed unverified project-reported concerns from `FAIL` to `REPORTED`; only reproduced/system-recorded or human-reviewed failures can set an evaluation to `FAIL`.
- Preserved newlines in normalized dataset fields.
- Preserved supervised assistant EOS on overlength training sequences; tested the truncation boundary without requiring model downloads.
- Made training fail clearly if tokenization/truncation leaves an example with no supervised assistant tokens.
- Marked successful model generation comparisons `COMPLETED`, explicitly not a quality or safety pass.
- Added `TODO.md`, `HUMAN_REQUIRED.md`, `PROGRESS.md`, and `EXECUTION_RULES.md`.

## Test results

- `python -m unittest discover -s tests -v`: **20/20 PASS**.
- `python -m compileall -q src tests`: **PASS**.
- `git diff --check`: **PASS**.
- `python -m src.verification --help`: **PASS**.
- Soup CLI v0.77.2: version and `ship --help` returned 0; Doctor returned 0 but reported missing training extras; Data Doctor returned 1 because `data/test.jsonl` is absent; Plan returned 0 but reported a zero dataset hash and unknown model VRAM; Ship returned 3 because the evaluation report does not match the CLI evidence schema.
- GPU inference and training: **not run**.
- Current report-generation code will retain quality/safety as `BLOCKED` when only project-reported concerns exist and model responses are unavailable; release remains `DON'T SHIP`. The concerns remain recorded as reports, not locally reproduced findings.

## Incomplete or blocked

- Dataset splits, model weights, training logs, and GPU measurements are not present in this checkout. This does not establish whether they exist in Google Drive.
- Current environment has no CUDA GPU and lacks Transformers, PEFT, Datasets, bitsandbytes, and Accelerate.
- Quality and safety responses have not been independently reviewed here.
- Held-out NLL/perplexity, quantization drift, peak allocated/reserved memory, host RAM, and throughput measurements require the original artifacts and CUDA runtime.
- Soup CLI Data Doctor, Pre-flight evidence, and Ship remain blocked by missing data/artifacts and metrics.
- The Soup CLI Doctor requires Transformers `>=5.16.1`, while `requirements.txt` pins Transformers `<5`; do not install CLI training extras into the project environment without compatibility validation.
- The new CI workflow has passed its commands locally but has not yet run on GitHub.

## Next stage

Restore the existing run in Google Colab, run only the non-training checks in `HUMAN_REQUIRED.md`, and return the generated response reports for independent review. Do not train or claim a release pass without the required evidence.
