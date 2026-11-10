"""Explicit batch-one greedy decoding with visible prefill and KV-cache updates."""
from __future__ import annotations

import time
from typing import Any

import torch

from .cache import ExplicitKVCache
from .policy import GreedyLogitsPolicy


@torch.inference_mode()
def cached_explicit_greedy(
    model: Any,
    input_ids: list[int],
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    pad_token_id: int | None = None,
    device: torch.device,
    return_cache: bool = False,
    measure_timing: bool = False,
) -> dict[str, Any]:
    """Prefill once, then feed one committed token per step through a KV cache.

    The first token selected from the prefill logits is not in the cache until the
    next iteration. Consequently, the terminal cache length is prompt length plus
    ``max(0, generated - 1)``. EOS is never fed back.
    """
    if not input_ids:
        raise ValueError("A nonempty prompt is required")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    del pad_token_id  # batch size is one and the prompt is unpadded

    prompt = torch.tensor([input_ids], dtype=torch.long, device=device)
    prompt_length = prompt.shape[1]
    attention_mask = torch.empty((1, 0), dtype=torch.long, device=device)
    cache = ExplicitKVCache()
    policy = GreedyLogitsPolicy(model)
    generated: list[int] = []
    cache_lengths: list[int] = []
    cache_positions: list[list[int]] = []
    step_seconds: list[float] = []

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def forward_chunk(chunk: torch.Tensor) -> torch.Tensor:
        nonlocal attention_mask
        start = cache.length
        if start != attention_mask.shape[1]:
            raise RuntimeError("Attention mask and cache length diverged")
        positions = torch.arange(
            start, start + chunk.shape[1], dtype=torch.long, device=device
        )
        mask = torch.cat(
            [attention_mask, torch.ones_like(chunk, dtype=torch.long)], dim=-1
        )
        if measure_timing:
            sync()
            started = time.perf_counter()
        output = model(
            input_ids=chunk,
            attention_mask=mask,
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            past_key_values=cache.cache,
            use_cache=True,
            return_dict=True,
        )
        if measure_timing:
            sync()
            step_seconds.append(time.perf_counter() - started)
        cache.cache = output.past_key_values
        cache.assert_length(start + chunk.shape[1])
        attention_mask = mask
        cache_lengths.append(cache.length)
        cache_positions.append(positions.tolist())
        return output.logits[:, -1, :]

    logits = forward_chunk(prompt)
    while len(generated) < max_new_tokens:
        token = policy.select(logits, input_ids + generated)
        generated.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
        if len(generated) >= max_new_tokens:
            break
        next_token = torch.tensor([[token]], dtype=torch.long, device=device)
        logits = forward_chunk(next_token)

    hit_cap = (
        len(generated) == max_new_tokens
        and (not generated or generated[-1] != eos_token_id)
    )
    result: dict[str, Any] = {
        "token_ids": generated,
        "hit_token_cap": hit_cap,
        "termination": "cap" if hit_cap else "eos",
        "prompt_length": prompt_length,
        "prefill_cache_length": cache_lengths[0] if cache_lengths else 0,
        "final_cache_length": cache.length,
        "cache_lengths": cache_lengths,
        "cache_positions": cache_positions,
        "generated_cache_resident": max(0, len(generated) - 1),
        "cache_state": cache.state_dict(),
        "timings": (
            {
                "prefill_seconds": step_seconds[0] if step_seconds else 0.0,
                "decode_seconds": sum(step_seconds[1:]),
                "total_forward_seconds": sum(step_seconds),
                "forward_steps": len(step_seconds),
            }
            if measure_timing
            else None
        ),
    }
    if return_cache:
        result["cache"] = cache
    return result
