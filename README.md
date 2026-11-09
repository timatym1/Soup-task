# Soup Internship — LoRA fine-tuning and release verification

Reproducible Google Colab/T4 pipeline for fine-tuning `Qwen/Qwen2.5-0.5B-Instruct` on a deterministic, cleaned subset of `tatsu-lab/alpaca` (up to 2,000 source rows), merging the adapter, saving an NF4 4-bit model, and producing evidence-based reports. The base model is under 3B parameters. No model weights, training logs, or benchmark outcomes are committed; run the pipeline to generate actual artifacts.

The 0.5B instruct model is small enough to make FP16 LoRA training and merging practical on a T4, and its chat template supports consistent training/inference formatting. Alpaca's instruction/input/output fields directly support the cleaning, deduplication, and held-out verification steps; the 2,000-row cap keeps this a take-home experiment rather than a full benchmark. Review the current model and dataset licenses before redistribution.

## Run in Google Colab

Select **Runtime → Change runtime type → T4 GPU**, then clone/open this repository and run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.run_pipeline
```

The pipeline intentionally stops on stage errors. It requires a working CUDA runtime and enough free disk for three copies of the model. Colab supplies CUDA-enabled PyTorch; `requirements.txt` intentionally does not replace that runtime package. Verify `torch.cuda.is_available()` before running. The Hugging Face model and dataset are public; set `MODEL_ID`, `DATASET_ID`, or `SEED` in the environment to override defaults. Do not put access tokens in source files; use Colab Secrets/environment variables if a gated model is selected.

## Pipeline and evidence

1. `src.data_preparation` reads Alpaca instruction/input/output rows, normalizes whitespace, rejects empty/malformed records, removes exact normalized duplicates, shuffles with a fixed seed, and writes train/test JSONL plus `reports/data_validation.json`.
2. `src.memory_analysis` estimates FP16 weights, checkpointed activations, logits, and LoRA/8-bit-optimizer states from model configuration. Estimates are intentionally labeled approximate; real peaks are recorded by training, merge, quantization, and verification.
3. `src.train` checks CUDA and the tokenizer chat template, masks prompt tokens from the loss, and performs one epoch with rank-8 LoRA on `q_proj`/`v_proj`, FP16, gradient checkpointing, and paged 8-bit AdamW. Timestamped logs and training metrics are saved.
4. `src.merge_model` safely merges the adapter into FP16 weights. `src.quantization` saves a bitsandbytes NF4 4-bit checkpoint.
5. `src.verification` uses identical held-out prompts and deterministic greedy generation for base, merged, and quantized models. It stores all responses and last-prompt/last-token logit comparisons. This detects a change, not whether that change is good.
6. `src.silent_failures` creates a local evidence matrix and a final verdict report. A `PASS` is limited to the specific local check described in the report.

`MODEL_ID` defaults to `Qwen/Qwen2.5-0.5B-Instruct`; `DATASET_ID` defaults to `tatsu-lab/alpaca`; `SEED` defaults to `42`. Current training defaults target a T4, but actual runtime can exceed 10 minutes due to downloads and evaluation.

## Outputs

- `models/base/`, `models/adapter/`, `models/merged/`, `models/quantized/`
- `data/train.jsonl`, `data/test.jsonl`
- `logs/*.log`, timestamped `logs/nvidia-smi.txt`
- `reports/data_validation.json`, `reports/memory_budget.json`, `reports/verification.json`
- `reports/silent_failures.md`, `reports/final_report.md`

These output directories are ignored by Git to prevent accidental commits of model weights or experiment evidence. Preserve the generated logs and reports with the delivered model artifacts.

## Release status and limitations

The generated final report says **DON'T SHIP** until all artifacts/evidence exist, Soup CLI checks have actually been run, and independent quality/safety review has passed. This project does not invent Soup Data Doctor, Pre-flight, or Ship results: their CLI contract is not provided, so the silent-failure report marks those checks unverified. A small greedy sample and logit delta are not a benchmark, human evaluation, or safety assessment.

The memory estimate is an order-of-magnitude planning aid; attention kernels, allocator fragmentation, CUDA context, tokenizer padding, and temporary merge/quantization buffers affect actual VRAM. Inspect both recorded CUDA peaks and timestamped `nvidia-smi` output. Hugging Face dataset/model terms and licenses must be reviewed before redistribution.

AI assistance: GitHub Copilot was used to draft and review the pipeline. Unit tests, Python syntax compilation, and report generation were executed locally; model training, GPU measurements, and inference must still be independently run in Colab.
