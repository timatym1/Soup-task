import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


def load_environment(path: Path = ROOT / ".env") -> None:
    load_dotenv(path)


load_environment()
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
REPORT_DIR = ROOT / "reports"
BASE_DIR = MODEL_DIR / "base"
ADAPTER_DIR = MODEL_DIR / "adapter"
MERGED_DIR = MODEL_DIR / "merged"
QUANTIZED_DIR = MODEL_DIR / "quantized"
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DATASET_ID = os.getenv("DATASET_ID", "tatsu-lab/alpaca")
SEED = int(os.getenv("SEED", "42"))


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_directories() -> None:
    for path in (DATA_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, filename: str) -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
        formatter.converter = __import__("time").gmtime
        handler = logging.FileHandler(LOG_DIR / filename, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def snapshot_gpu(logger: logging.Logger, label: str) -> None:
    logger.info("GPU snapshot (%s) at %s", label, timestamp())
    try:
        result = subprocess.run(
            ["nvidia-smi"], check=True, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("nvidia-smi unavailable: %s", exc)
        return
    with (LOG_DIR / "nvidia-smi.txt").open("a", encoding="utf-8") as output:
        output.write(f"\n[{timestamp()}] {label}\n{result.stdout}")
    logger.info("nvidia-smi output appended to logs/nvidia-smi.txt")


def cuda_memory() -> dict[str, int] | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return {
        "allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def download_base_model() -> Path:
    from huggingface_hub import snapshot_download

    ensure_directories()
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(BASE_DIR),
        token=os.getenv("HF_TOKEN") or None,
    )
    return BASE_DIR
