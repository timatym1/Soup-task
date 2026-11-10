# Changelog

## [Unreleased] — 2026-11-10
- Add `soup.yaml` for Qwen2.5-0.5B-Instruct and `dataset_provenance.json` with dataset/configuration SHA-256 hashes.
- Add LoRA checkpoint resume, periodic checkpoint saving, and non-reentrant gradient checkpointing.
- Add separate quality and safety suites with response comparison across base, merged, and quantized models.
- Fix tensor handling, generation decoding offsets, and recovery from training interruptions.
- Verify dataset validation, completed training, merge, NF4 quantization, model verification, and 9/9 unit tests.
- Record known issues: quality and safety evaluations FAIL; Soup CLI integration is BLOCKED; release decision is DON'T SHIP.

## v0.2.0
- Add independent, human-reviewable quality and safety evaluation suites for all three model variants.
- Gate release reports on actual artifacts and evidence; report unavailable checks as BLOCKED.
- Document Colab verification commands that do not retrain an existing model.

## v0.1.0
- Add reproducible Alpaca data validation, LoRA fine-tuning, merge, and NF4 quantization pipeline.
- Add memory estimates, three-model verification, timestamped GPU logging, and evidence-gated release reports.
