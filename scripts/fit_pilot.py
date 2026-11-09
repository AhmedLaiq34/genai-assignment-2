"""Run a short LoRA fit test using synthetic text, never assignment records."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def _encoded_synthetic_example(
    tokenizer: Any,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    messages = [
        {
            "role": "system",
            "content": (
                "Classify the synthetic criterion. Return a JSON object with "
                "label and evidence_sentence_ids."
            ),
        },
        {
            "role": "user",
            "content": (
                "Criterion type: inclusion\n"
                "Criterion: The fictional subject is at least 18 years old.\n"
                "Synthetic record:\n0. The fictional subject is 24 years old."
            ),
        },
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    answer_ids = tokenizer.encode(
        '{"label":"included","evidence_sentence_ids":[0]}',
        add_special_tokens=False,
    )
    if tokenizer.eos_token_id is not None:
        answer_ids.append(tokenizer.eos_token_id)
    if len(prompt_ids) + len(answer_ids) > max_seq_len:
        raise ValueError(
            "Synthetic example exceeds max_seq_len; increase the pilot length."
        )
    token_ids = prompt_ids + answer_ids
    padding_length = max_seq_len - len(token_ids)
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define a pad token for the fit pilot")
    input_ids = torch.tensor(
        [[tokenizer.pad_token_id] * padding_length + token_ids],
        device=device,
    )
    labels = torch.tensor(
        [[-100] * (padding_length + len(prompt_ids)) + answer_ids],
        device=device,
    )
    attention_mask = torch.tensor(
        [[0] * padding_length + [1] * len(token_ids)],
        device=device,
    )
    return input_ids, attention_mask, labels


def run_fit_pilot(
    steps: int = 10,
    gradient_accumulation: int = 8,
    max_seq_len: int = 1024,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; LoRA fit pilot requires a GPU")
    if steps < 1 or gradient_accumulation < 1:
        raise ValueError("steps and gradient accumulation must be positive")

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=True,
    )
    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    model.train()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("LoRA produced no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    input_ids, attention_mask, labels = _encoded_synthetic_example(
        tokenizer, max_seq_len, device
    )

    losses: list[float] = []
    step_times: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        for _ in range(gradient_accumulation):
            with torch.autocast(device_type="cuda", dtype=dtype):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                loss = output.loss / gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite synthetic loss at step {step}")
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - start)
        losses.append(float(loss.detach().item() * gradient_accumulation))

    all_losses_finite = all(
        torch.isfinite(torch.tensor(loss)).item() for loss in losses
    )
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "synthetic_only": True,
        "steps": steps,
        "gradient_accumulation": gradient_accumulation,
        "max_sequence_tokens": int(input_ids.shape[-1]),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "losses": losses,
        "seconds_per_optimizer_step": step_times,
        "mean_seconds_per_step": sum(step_times) / len(step_times),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "all_losses_finite": all_losses_finite,
        "passed": len(losses) == steps and all_losses_finite,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fit_pilot.json"),
    )
    args = parser.parse_args()
    try:
        report = run_fit_pilot(
            args.steps,
            args.gradient_accumulation,
            args.max_seq_len,
        )
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        report = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "synthetic_only": True,
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"GPU memory pilot failed; details saved to {args.output.resolve()}")
        return 1
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote fit pilot results to {args.output.resolve()}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
