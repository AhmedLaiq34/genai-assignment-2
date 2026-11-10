"""Deterministic greedy speculative decoding with explicit cache reconciliation."""
from __future__ import annotations

from typing import Any

import torch

from .cache import ExplicitKVCache
from .policy import GreedyLogitsPolicy


class _ModelState:
    def __init__(self, model: Any, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.cache = ExplicitKVCache()
        self.attention_mask = torch.empty((1, 0), dtype=torch.long, device=device)
        self.logits: torch.Tensor | None = None

    def forward_chunk(self, token_ids: list[int]) -> torch.Tensor:
        if not token_ids:
            raise ValueError("A nonempty token chunk is required")
        start = self.cache.length
        if start != self.attention_mask.shape[1]:
            raise RuntimeError("Attention mask and cache length diverged")
        chunk = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        positions = torch.arange(
            start, start + chunk.shape[1], dtype=torch.long, device=self.device
        )
        mask = torch.cat(
            [self.attention_mask, torch.ones_like(chunk, dtype=torch.long)], dim=-1
        )
        output = self.model(
            input_ids=chunk,
            attention_mask=mask,
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            past_key_values=self.cache.cache,
            use_cache=True,
            return_dict=True,
        )
        self.cache.cache = output.past_key_values
        self.cache.assert_length(start + chunk.shape[1])
        self.attention_mask = mask
        self.logits = output.logits[:, -1, :]
        return output.logits

    def crop(self, length: int) -> None:
        self.cache.crop(length)
        self.attention_mask = self.attention_mask[:, :length]


@torch.inference_mode()
def speculative_greedy(
    target_model: Any,
    draft_model: Any,
    input_ids: list[int],
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    draft_k: int = 4,
    device: torch.device,
    return_cache: bool = False,
) -> dict[str, Any]:
    """Reproduce the target's ordinary greedy token sequence with a draft block.

    Each iteration drafts up to ``draft_k`` tokens, verifies them in one target
    forward pass, commits the matching prefix plus one correction token, and
    reconciles both caches. The target's saved logits after the committed prefix
    predict the first draft token; verification logits after token i predict
    draft token i+1.
    """
    if not input_ids:
        raise ValueError("A nonempty prompt is required")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if draft_k <= 0:
        raise ValueError("draft_k must be positive")

    target = _ModelState(target_model, device)
    draft = _ModelState(draft_model, device)
    target_policy = GreedyLogitsPolicy(target_model)
    draft_policy = GreedyLogitsPolicy(draft_model)
    prompt_length = len(input_ids)
    target.forward_chunk(input_ids)
    draft.forward_chunk(input_ids)
    committed: list[int] = []
    blocks: list[dict[str, Any]] = []
    proposed_total = 0
    accepted_total = 0
    corrections = 0
    bonuses = 0
    target_forward_tokens = prompt_length
    draft_forward_tokens = prompt_length
    termination = "cap"

    while len(committed) < max_new_tokens:
        remaining = max_new_tokens - len(committed)
        block_size = min(draft_k, remaining)
        draft_tokens: list[int] = []
        for index in range(block_size):
            token = draft_policy.select(
                draft.logits, input_ids + committed + draft_tokens
            )
            draft_tokens.append(token)
            if eos_token_id is not None and token == eos_token_id:
                break
            draft.forward_chunk([token])
            draft_forward_tokens += 1
            if index + 1 >= block_size:
                break
        proposed_total += len(draft_tokens)
        target_prefix_logits = target.logits
        if target_prefix_logits is None:
            raise RuntimeError("Target prefix logits are missing")
        verify_logits = target.forward_chunk(draft_tokens)
        target_forward_tokens += len(draft_tokens)
        predictions = torch.cat(
            [target_prefix_logits.unsqueeze(1), verify_logits[:, :-1]], dim=1
        )
        target_predictions = [
            target_policy.select(
                predictions[:, index, :], input_ids + committed + draft_tokens[:index]
            )
            for index in range(len(draft_tokens))
        ]

        accepted: list[int] = []
        correction: int | None = None
        mismatch_index: int | None = None
        for index, draft_token in enumerate(draft_tokens):
            predicted = target_predictions[index]
            if predicted != draft_token:
                correction = predicted
                mismatch_index = index
                break
            accepted.append(draft_token)
            if eos_token_id is not None and draft_token == eos_token_id:
                break
        accepted_total += len(accepted)

        bonus: int | None = None
        stop = False
        if accepted and eos_token_id is not None and accepted[-1] == eos_token_id:
            committed.extend(accepted)
            reconciled_length = prompt_length + len(committed) - 1
            target.crop(reconciled_length)
            draft.crop(reconciled_length)
            termination = "eos"
            stop = True
        elif correction is not None:
            accepted_length = len(accepted)
            prefix_length = prompt_length + len(committed)
            target.crop(prefix_length + accepted_length)
            draft.crop(prefix_length + accepted_length)
            committed.extend(accepted)
            committed.append(correction)
            corrections += 1
            if eos_token_id is not None and correction == eos_token_id:
                termination = "eos"
                stop = True
            elif len(committed) >= max_new_tokens:
                termination = "cap"
                stop = True
            else:
                target.forward_chunk([correction])
                draft.forward_chunk([correction])
                target_forward_tokens += 1
                draft_forward_tokens += 1
        else:
            committed.extend(accepted)
            if len(committed) >= max_new_tokens:
                termination = "cap"
                stop = True
            else:
                bonus = target_policy.select(target.logits, input_ids + committed)
                committed.append(bonus)
                bonuses += 1
                if eos_token_id is not None and bonus == eos_token_id:
                    termination = "eos"
                    stop = True
                elif len(committed) >= max_new_tokens:
                    termination = "cap"
                    stop = True
                else:
                    target.forward_chunk([bonus])
                    draft.forward_chunk([bonus])
                    target_forward_tokens += 1
                    draft_forward_tokens += 1

        blocks.append(
            {
                "draft_tokens": draft_tokens,
                "target_predictions": target_predictions,
                "accepted": accepted,
                "correction": correction,
                "mismatch_index": mismatch_index,
                "bonus": bonus,
                "target_cache_length": target.cache.length,
                "draft_cache_length": draft.cache.length,
                "committed_length": len(committed),
            }
        )
        if stop:
            break

    hit_cap = (
        len(committed) == max_new_tokens
        and (not committed or committed[-1] != eos_token_id)
    )
    if hit_cap:
        termination = "cap"
    result: dict[str, Any] = {
        "token_ids": committed,
        "termination": termination,
        "hit_token_cap": hit_cap,
        "prompt_length": prompt_length,
        "target_cache_length": target.cache.length,
        "draft_cache_length": draft.cache.length,
        "blocks": blocks,
        "stats": {
            "draft_k": draft_k,
            "block_count": len(blocks),
            "proposed_tokens": proposed_total,
            "accepted_tokens": accepted_total,
            "acceptance_rate": (
                accepted_total / proposed_total if proposed_total else 0.0
            ),
            "correction_tokens": corrections,
            "bonus_tokens": bonuses,
            "target_prefill_calls": 1,
            "draft_prefill_calls": 1,
            "target_verification_calls": len(blocks),
            "target_forward_tokens": target_forward_tokens,
            "draft_forward_tokens": draft_forward_tokens,
        },
        "target_cache_state": target.cache.state_dict(),
        "draft_cache_state": draft.cache.state_dict(),
    }
    if return_cache:
        result["target_cache"] = target.cache
        result["draft_cache"] = draft.cache
    return result
