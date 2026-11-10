from __future__ import annotations

from collections.abc import Sequence


def truncate_training_sequence(
    full_ids: Sequence[int],
    prompt_ids: Sequence[int],
    max_length: int,
    eos_token_id: int | None,
    ignore_index: int = -100,
) -> tuple[list[int], list[int]]:
    if max_length < 1:
        raise ValueError("max_length must be greater than zero.")
    if not full_ids:
        return [], []

    input_ids = list(full_ids[:max_length])
    if len(full_ids) > max_length and eos_token_id is not None and full_ids[-1] == eos_token_id:
        input_ids[-1] = eos_token_id
    prompt_length = min(len(prompt_ids), len(input_ids))
    if prompt_length == len(input_ids):
        raise ValueError("Sequence truncation left no supervised assistant tokens.")
    labels = [ignore_index] * prompt_length + input_ids[prompt_length:]
    return input_ids, labels
