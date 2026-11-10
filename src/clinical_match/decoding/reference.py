"""Ordinary uncached greedy reference and deterministic cached HF greedy path."""
from __future__ import annotations

from typing import Any

import torch


@torch.inference_mode()
def uncached_greedy(model: Any, input_ids: list[int], *, max_new_tokens: int,
                    eos_token_id: int | None, device: torch.device,
                    pad_token_id: int | None = None) -> dict[str, Any]:
    """Recompute every prefix while preserving Transformers' greedy processors.

    Calling ``argmax(model(...))`` directly omits generation-config processors
    such as Qwen's ``repetition_penalty``.  ``generate(use_cache=False)`` is
    the equivalent uncached reference: it recomputes the prefix but applies
    exactly the same logits processors and stopping rules as the cached path.
    """
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    output = model.generate(
        input_ids=tensor,
        do_sample=False,
        num_beams=1,
        use_cache=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id if pad_token_id is not None else eos_token_id,
        return_dict_in_generate=True,
    )
    generated = output.sequences[0, tensor.shape[1]:].tolist()
    return {"token_ids": generated, "hit_token_cap": len(generated) == max_new_tokens and
            (not generated or generated[-1] != eos_token_id)}


@torch.inference_mode()
def cached_library_greedy(model: Any, input_ids: list[int], *, max_new_tokens: int,
                          eos_token_id: int | None, pad_token_id: int | None,
                          device: torch.device) -> dict[str, Any]:
    """Use Transformers' deterministic cache path for the full baseline run."""
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    output = model.generate(
        input_ids=tensor,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id if pad_token_id is not None else eos_token_id,
        return_dict_in_generate=True,
    )
    generated = output.sequences[0, tensor.shape[1]:].tolist()
    hit_cap = len(generated) == max_new_tokens and (not generated or generated[-1] != eos_token_id)
    return {"token_ids": generated, "hit_token_cap": hit_cap}
