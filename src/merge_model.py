from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import (
    ADAPTER_DIR,
    BASE_DIR,
    MERGED_DIR,
    cuda_memory,
    ensure_directories,
    get_logger,
    save_json,
    snapshot_gpu,
)


def main() -> None:
    logger = get_logger("merge", "merge.log")
    ensure_directories()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to measure merge VRAM.")
    snapshot_gpu(logger, "before_merge")
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto"
    )
    adapted = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    merged = adapted.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(BASE_DIR)).save_pretrained(MERGED_DIR)
    save_json(MERGED_DIR / "merge_metrics.json", {"gpu_peak": cuda_memory()})
    snapshot_gpu(logger, "after_merge")
    logger.info("Merged model saved under %s", MERGED_DIR)


if __name__ == "__main__":
    main()
