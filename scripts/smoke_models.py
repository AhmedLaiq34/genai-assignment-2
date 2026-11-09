"""Measure local inference fit for pinned candidate models on synthetic prompts."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DRAFT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def _load_model(model_id: str, revision: str) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    model.eval()
    return model


def _measure_forward(model: Any, prompt_ids: torch.Tensor) -> dict[str, Any]:
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(
            input_ids=prompt_ids.to(device),
            attention_mask=torch.ones_like(prompt_ids, device=device),
            use_cache=True,
        )
    torch.cuda.synchronize(device)
    prefill_seconds = time.perf_counter() - start
    logits = output.logits[:, -1, :]
    next_token = logits.argmax(dim=-1, keepdim=True)
    cache = output.past_key_values
    del output, logits
    decode_start = time.perf_counter()
    with torch.inference_mode():
        decoded = model(
            input_ids=next_token,
            attention_mask=torch.ones(
                (1, prompt_ids.shape[-1] + 1),
                device=device,
                dtype=torch.long,
            ),
            past_key_values=cache,
            use_cache=True,
        )
    torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - decode_start
    decoded_logits = decoded.logits[:, -1, :]
    return {
        "passed": bool(
            torch.isfinite(decoded_logits).all().item()
            and cache.get_seq_length() == prompt_ids.shape[-1] + 1
        ),
        "prompt_tokens": int(prompt_ids.shape[-1]),
        "prefill_seconds": prefill_seconds,
        "one_token_decode_seconds": decode_seconds,
        "greedy_next_token_id": int(next_token.item()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "cache_length_after_decode": int(cache.get_seq_length()),
    }


def run_local_fit_test() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; local GPU fit cannot be measured")
    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_ID,
        revision=TARGET_REVISION,
        use_fast=True,
    )
    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "precision": "float16",
        "synthetic_only": True,
        "target": {},
        "target_draft_co_residency": {},
    }
    seed_tokens = tokenizer(
        "Synthetic feasibility prompt.",
        add_special_tokens=False,
    )["input_ids"]
    repeated_ids = (seed_tokens * ((1024 // len(seed_tokens)) + 1))[:1024]
    input_ids = torch.tensor([repeated_ids], dtype=torch.long)

    target = None
    try:
        target = _load_model(TARGET_ID, TARGET_REVISION)
        report["target"]["256"] = _measure_forward(target, input_ids[:, :256])
        report["target"]["1024"] = _measure_forward(target, input_ids)
    except torch.cuda.OutOfMemoryError as error:
        report["target"]["load_or_forward_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        torch.cuda.empty_cache()

    draft = None
    if target is not None:
        try:
            draft = _load_model(DRAFT_ID, DRAFT_REVISION)
            co_residency = report["target_draft_co_residency"]
            co_residency["loaded"] = True
            co_residency["by_prompt_tokens"] = {}
            for prompt_length in (256, 1024):
                target_result = _measure_forward(
                    target, input_ids[:, :prompt_length]
                )
                draft_result = _measure_forward(
                    draft, input_ids[:, :prompt_length]
                )
                co_residency["by_prompt_tokens"][str(prompt_length)] = {
                    "target": target_result,
                    "draft": draft_result,
                    "pair_peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated()
                    ),
                    "pair_peak_reserved_bytes": int(
                        torch.cuda.max_memory_reserved()
                    ),
                }
            co_residency["target"] = co_residency["by_prompt_tokens"]["256"][
                "target"
            ]
            co_residency["draft"] = co_residency["by_prompt_tokens"]["256"][
                "draft"
            ]
            co_residency["pair_peak_allocated_bytes"] = co_residency[
                "by_prompt_tokens"
            ]["256"]["pair_peak_allocated_bytes"]
            co_residency["pair_peak_reserved_bytes"] = co_residency[
                "by_prompt_tokens"
            ]["256"]["pair_peak_reserved_bytes"]
        except torch.cuda.OutOfMemoryError as error:
            report["target_draft_co_residency"] = {
                "loaded": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            torch.cuda.empty_cache()
    else:
        report["target_draft_co_residency"] = {
            "loaded": False,
            "not_attempted": "Target model did not fit for the local forward pilot",
        }

    report["target_fit_passed"] = all(
        report["target"].get(length, {}).get("passed", False)
        for length in ("256", "1024")
    )
    report["pair_fit_passed"] = (
        report["target_draft_co_residency"].get("loaded", False)
        and all(
            report["target_draft_co_residency"]
            .get("by_prompt_tokens", {})
            .get(length, {})
            .get(model, {})
            .get("passed", False)
            for length in ("256", "1024")
            for model in ("target", "draft")
        )
    )
    if target is not None:
        del target
    if draft is not None:
        del draft
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/feasibility_local.json"),
    )
    args = parser.parse_args()
    report = run_local_fit_test()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote local model fit results to {args.output.resolve()}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["target_fit_passed"] and report["pair_fit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
