import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.common import (
    ADAPTER_DIR,
    BASE_DIR,
    DATA_DIR,
    cuda_memory,
    download_base_model,
    ensure_directories,
    get_logger,
    snapshot_gpu,
)

MAX_LENGTH = 512
IGNORE_INDEX = -100


class TrainingLogCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.info("trainer_step=%d metrics=%s", state.global_step, json.dumps(logs))


def build_prompt(record: dict[str, str]) -> str:
    content = record["instruction"]
    if record.get("input"):
        content += f"\n\nInput:\n{record['input']}"
    return content


def encode_record(tokenizer: AutoTokenizer, record: dict[str, str]) -> dict[str, list[int]]:
    prompt_messages = [{"role": "user", "content": build_prompt(record)}]
    full_messages = prompt_messages + [{"role": "assistant", "content": record["output"]}]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False, return_dict=False
    )

    input_ids = full_ids[:MAX_LENGTH]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [IGNORE_INDEX] * prompt_length + input_ids[prompt_length:]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))
    if not tokenizer.chat_template:
        raise RuntimeError("The selected tokenizer has no chat template.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_datasets(tokenizer: AutoTokenizer) -> tuple[list[dict], list[dict]]:
    train_data = [encode_record(tokenizer, r) for r in read_jsonl(DATA_DIR / "train.jsonl")]
    eval_data = [encode_record(tokenizer, r) for r in read_jsonl(DATA_DIR / "test.jsonl")]

    has_supervised_tokens = all(
        any(label != IGNORE_INDEX for label in row["labels"]) for row in train_data + eval_data
    )
    if not train_data or not eval_data or not has_supervised_tokens:
        raise RuntimeError("Tokenization yielded empty data or examples without supervised tokens.")
    return train_data, eval_data


def build_model():
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto"
    )
    model.config.use_cache = False
    model.enable_input_require_grads()  # needed for gradient checkpointing with LoRA

    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    # fp16 AMP cannot unscale fp16 trainable params -> keep LoRA weights in fp32
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.print_trainable_parameters()
    return model


def build_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(ADAPTER_DIR / "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        fp16=True,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=42,
        data_seed=42,
        dataloader_pin_memory=True,
    )


def main() -> None:
    logger = get_logger("training", "training.log")
    ensure_directories()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; run this pipeline in a Colab T4 runtime.")

    download_base_model()
    snapshot_gpu(logger, "before_training")
    torch.cuda.reset_peak_memory_stats()

    tokenizer = load_tokenizer()
    train_data, eval_data = load_datasets(tokenizer)
    model = build_model()

    trainer = Trainer(
        model=model,
        args=build_training_args(),
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        processing_class=tokenizer,
        callbacks=[TrainingLogCallback(logger)],
    )
    result = trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    metrics = {
        "train_metrics": result.metrics,
        "eval_metrics": trainer.evaluate(),
        "gpu_peak": cuda_memory(),
        "train_examples": len(train_data),
        "eval_examples": len(eval_data),
    }
    (ADAPTER_DIR / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    snapshot_gpu(logger, "after_training")
    logger.info("Adapter and training metrics saved under %s", ADAPTER_DIR)


if __name__ == "__main__":
    main()