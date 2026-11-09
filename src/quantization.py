import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.common import (
    MERGED_DIR,
    QUANTIZED_DIR,
    cuda_memory,
    ensure_directories,
    get_logger,
    save_json,
    snapshot_gpu,
)


def main() -> None:
    logger = get_logger("quantization", "quantization.log")
    ensure_directories()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for bitsandbytes 4-bit quantization.")
    snapshot_gpu(logger, "before_quantization")
    torch.cuda.reset_peak_memory_stats()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(MERGED_DIR),
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    QUANTIZED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(QUANTIZED_DIR, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(MERGED_DIR)).save_pretrained(QUANTIZED_DIR)
    save_json(QUANTIZED_DIR / "quantization_metrics.json", {"gpu_peak": cuda_memory()})
    snapshot_gpu(logger, "after_quantization")
    logger.info("4-bit NF4 model saved under %s", QUANTIZED_DIR)


if __name__ == "__main__":
    main()
