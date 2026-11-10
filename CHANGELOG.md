# Changelog

## v0.3.0 — 2026-11-10
- Preserve the historical 2026-11-09 run metrics from commit `66c591c4` as explicitly labeled summary evidence; distinguish them from unavailable raw logs and current workspace artifacts.
- Derive quality and safety release status from complete human-reviewed cases, not project-reported concerns or a manually asserted report status.
- Sample the source dataset deterministically across its full range and fail validation on near-duplicate train/test leakage.
- Verify prompt-token prefix alignment before masking labels; estimate FP32 loss logits, FP32 adapter gradients, and 8-bit optimizer states.
- Keep estimates labeled `ESTIMATE_ONLY`; do not count an estimate as measured-memory release evidence.
- Add token-weighted held-out assistant NLL/perplexity for all three model variants and require those metrics before model verification can pass; report the 200-record Colab command and coverage.
- Measure merged-to-4-bit KL divergence and top-1 agreement across supervised answer tokens, and require consistent full-answer drift evidence before verification can pass.
- Verify the quantized checkpoint's runtime configuration, initialized 4-bit modules, and NF4 quantization states before accepting the verification report.

## [Unreleased] — 2026-11-10
- Add `soup.yaml` for Qwen2.5-0.5B-Instruct and `dataset_provenance.json` with dataset/configuration SHA-256 hashes.
- Add LoRA checkpoint resume, periodic checkpoint saving, and non-reentrant gradient checkpointing.
- Add separate quality and safety suites with response comparison across base, merged, and quantized models.
- Fix tensor handling, generation decoding offsets, and recovery from training interruptions.
- Historical run summaries recorded data validation, training, merge, NF4 quantization, and model verification; current checkout does not contain their raw logs or model artifacts.
- Earlier reports marked quality and safety as FAIL from project-supplied concerns; this was superseded by evidence-based review gates and those unverified claims were removed.
- Expand evaluation cases for multi-step reasoning, hallucination resistance, instruction following, privacy, and harmful assistance.
- Make verification write explicit BLOCKED evidence when prerequisites are missing; report current artifact availability and recomputed provenance hashes.
- Generate a machine-readable release decision that distinguishes missing local artifacts from historical run summaries.
- Add unit coverage for release statuses, blocked verification, and SHA-256 provenance checks.
- Extend the unit suite to 15 passing tests.
- Fix the Soup CLI aggregate gate so it passes only when Doctor, Data Doctor, Pre-flight, and Ship evidence all pass.
- Add dependency-free GitHub Actions checks for tests and compilation; the regression suite now passes 16 tests.
- Add project execution, human-required, and progress tracking documents with evidence-based completion criteria.
- Superseded by v0.3.0: remove unsupported project-reported findings from model evaluation evidence.
- Preserve line boundaries during dataset whitespace normalization.
- Preserve the assistant EOS token when a training example is truncated at the sequence limit.
- Label successful generation/logit comparison as `COMPLETED`, not a quality `PASS`; record its scope explicitly.
- Require explicit human `PASS`/`FAIL` quality assessments and fail the release gate for a reviewed failing case.
- Add regression coverage for human-reviewed outcomes, multiline text normalization, supervised EOS truncation, NLL/logit-drift metrics, and NF4 verification; the suite now passes 38 tests.

## v0.2.0
- Add independent, human-reviewable quality and safety evaluation suites for all three model variants.
- Gate release reports on actual artifacts and evidence; report unavailable checks as BLOCKED.
- Document Colab verification commands that do not retrain an existing model.

## v0.1.0
- Add reproducible Alpaca data validation, LoRA fine-tuning, merge, and NF4 quantization pipeline.
- Add memory estimates, three-model verification, timestamped GPU logging, and evidence-gated release reports.
