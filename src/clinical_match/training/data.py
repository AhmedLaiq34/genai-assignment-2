"""Leakage-safe supervised encoding for TrialGPT criterion examples."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch

from ..prompts import build_prompt, target_json


@dataclass(frozen=True)
class EncodedExample:
    annotation_id: int
    patient_id: str
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    prompt_length: int
    answer_text: str
    target_token_count: int
    prompt_truncated: bool
    removed_sentence_ids: list[int]
    kept_sentence_ids: list[int]


def encode_example(
    tokenizer: Any,
    example: dict[str, Any],
    *,
    max_total_tokens: int,
    max_new_tokens: int,
    max_seq_len: int,
) -> EncodedExample:
    """Render the frozen inference prompt and supervise only the assistant answer.

    The chat template used at inference is rendered exactly once for the prompt
    (``add_generation_prompt=True``). The assistant answer is appended with the
    template's end-of-turn token. Any template suffix after that token is masked.
    """
    prompt = build_prompt(
        tokenizer,
        example["input"],
        max_total_tokens=max_total_tokens,
        max_new_tokens=max_new_tokens,
    )
    answer_text = target_json(example["target"])
    full_ids = list(
        tokenizer.apply_chat_template(
            prompt["messages"] + [{"role": "assistant", "content": answer_text}],
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    prompt_ids = list(prompt["input_ids"])
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Assistant rendering does not preserve the inference prompt prefix")

    answer_ids = list(tokenizer.encode(answer_text, add_special_tokens=False))
    start = len(prompt_ids)
    if full_ids[start : start + len(answer_ids)] != answer_ids:
        raise ValueError("Assistant answer tokens do not match the compact JSON target")

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define an end-of-sequence token")

    labels = [-100] * len(full_ids)
    supervised = 0
    found_eos = False
    for index in range(start, len(full_ids)):
        labels[index] = full_ids[index]
        supervised += 1
        if full_ids[index] == eos_token_id:
            found_eos = True
            break
    if not found_eos:
        raise ValueError("Assistant answer is missing the template end-of-turn token")
    if supervised == 0:
        raise ValueError("No assistant target tokens are supervised")
    if len(full_ids) > max_seq_len:
        raise ValueError(
            f"Encoded example has {len(full_ids)} tokens, exceeding max_seq_len={max_seq_len}"
        )

    return EncodedExample(
        annotation_id=int(example["provenance"]["annotation_id"]),
        patient_id=str(example["provenance"]["patient_id"]),
        input_ids=full_ids,
        labels=labels,
        attention_mask=[1] * len(full_ids),
        prompt_length=len(prompt_ids),
        answer_text=answer_text,
        target_token_count=supervised,
        prompt_truncated=bool(prompt["truncated"]),
        removed_sentence_ids=list(prompt["removed_sentence_ids"]),
        kept_sentence_ids=list(prompt["kept_sentence_ids"]),
    )


def build_training_set(
    tokenizer: Any,
    examples: list[dict[str, Any]],
    *,
    max_total_tokens: int,
    max_new_tokens: int,
    max_seq_len: int,
) -> list[EncodedExample]:
    return [
        encode_example(
            tokenizer,
            example,
            max_total_tokens=max_total_tokens,
            max_new_tokens=max_new_tokens,
            max_seq_len=max_seq_len,
        )
        for example in examples
    ]


def order_for_epoch(size: int, seed: int, epoch: int) -> list[int]:
    """Return a deterministic permutation independent of global RNG state."""
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order


def collate_encoded(examples: list[EncodedExample], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a microbatch; padding and prompt tokens remain unsupervised."""
    if not examples:
        raise ValueError("Cannot collate an empty microbatch")
    max_length = max(len(example.input_ids) for example in examples)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for example in examples:
        padding = max_length - len(example.input_ids)
        if padding:
            input_rows.append(example.input_ids + [pad_token_id] * padding)
            label_rows.append(example.labels + [-100] * padding)
            mask_rows.append(example.attention_mask + [0] * padding)
        else:
            input_rows.append(list(example.input_ids))
            label_rows.append(list(example.labels))
            mask_rows.append(list(example.attention_mask))
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
    }


def split_counts(examples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        "inclusion": {},
        "exclusion": {},
    }
    for example in examples:
        kind = example["input"]["criterion_type"]
        label = example["target"]["label"]
        counts[kind][label] = counts[kind].get(label, 0) + 1
    return counts
