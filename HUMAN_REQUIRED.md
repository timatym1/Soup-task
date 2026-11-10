# Human-required actions

Only actions unavailable to the repository agent are listed here. Do not retrain, merge, or quantize solely to recreate previously reported work.

## 1. Restore the existing run in Google Colab

**Why:** The agent cannot access the user's mounted Google Drive. The current checkout contains neither the dataset splits nor model artifacts; their absence here does not establish that the artifacts are absent from Drive.

**Action:** In the Colab runtime that has access to the existing run, mount Drive and set `SOUP_ARTIFACT_DIR` to the actual run directory containing `data/`, `models/`, and `logs/`. Do not guess the Drive path.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
cd /content/Soup-task
export SOUP_ARTIFACT_DIR="/content/drive/MyDrive/<existing-soup-run>"
test -f "$SOUP_ARTIFACT_DIR/data/train.jsonl"
test -f "$SOUP_ARTIFACT_DIR/data/test.jsonl"
cp -a "$SOUP_ARTIFACT_DIR/data" ./data
cp -a "$SOUP_ARTIFACT_DIR/models" ./models
cp -a "$SOUP_ARTIFACT_DIR/logs" ./logs
```

**Expected result:** Both split files and the existing model/log artifacts are present under the repository root. If any `test -f` fails, stop and locate the original run; do not generate replacement data or claim provenance is verified.

## 2. Run non-training GPU verification and evaluation

**Why:** This environment has no CUDA GPU and lacks Transformers, PEFT, Datasets, bitsandbytes, and Accelerate. Model responses and GPU checks cannot be produced here.

**Action:** In the restored Colab environment, run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.data_validation
python -m src.verification --samples 200
python -m src.model_evaluation
python -m src.silent_failures
```

Do not run `python -m src.run_pipeline`; it starts training. If prerequisites are absent or a command fails, preserve its real error and leave the affected evidence `BLOCKED` or `FAIL`.

**Expected result:** Tests pass, data validation reports `PASS`, the verification report records the common base tokenizer, matched response counts, token-weighted held-out assistant NLL/perplexity, full-answer merged-to-4-bit KL/top-1 drift, and initialized NF4 layers, and the evaluation reports contain real responses. These results do not by themselves establish quality or safety.

## 3. Independently review model responses

**Why:** The quality and safety gates require an independent human judgment; the agent must not invent or self-approve that evidence.

**Action:** Review every case and each base, merged, and quantized response in `reports/quality_evaluation.json` and `reports/safety_evaluation.json`. Complete each `human_review` field with the required status, assessment/safety decision, and concise notes. Mark unsafe or incorrect behavior honestly; do not mark a suite passed if any case fails or remains unreviewed.

**Expected result:** Every case has a completed review tied to the actual generated responses. Any discovered failure remains a release blocker until addressed and re-evaluated.

## 4. Capture actual model quality and resource measurements

**Why:** Perplexity/NLL, quantization drift, GPU allocation/reservation, host RAM peaks, and runtime throughput require the actual model artifacts and CUDA runtime. The local memory estimator is only an approximation.

**Action:** `src.verification` calculates per-model held-out assistant-token NLL/perplexity and merged-to-4-bit KL/top-1 drift across supervised answer tokens, using the training chat template and 512-token truncation boundary. Run it on all 200 held-out examples. Also run negative controls. Preserve the exact prompt set and tokenization settings. Retrieve the original training/merge/quantization metrics and timestamped `nvidia-smi` log; if a measurement is absent, do not claim it was captured. Record peak allocated and reserved CUDA memory separately and host RAM during merge.

**Expected result:** Reports contain measured values, sample counts, model identities, and evaluation settings; estimated values remain explicitly labeled as estimates. These metrics do not replace independent quality or safety review.
