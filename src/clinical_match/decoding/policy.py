"""Greedy logits processors shared by manual cache decoders.

The pinned Qwen checkpoints ship with sampling defaults in their generation
config. Sampling is disabled by the project contract, but Qwen's repetition
penalty remains active in Transformers' greedy ``generate`` path. Manual loops
must apply the same history-dependent processor before argmax.
"""
from __future__ import annotations

from typing import Any

import torch


class GreedyLogitsPolicy:
    """Apply the pinned model's supported greedy processors to next-token logits."""

    def __init__(self, model: Any) -> None:
        generation_config = getattr(model, "generation_config", None)
        self.repetition_penalty = float(
            getattr(generation_config, "repetition_penalty", 1.0) or 1.0
        )
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")

        # Fail loudly if a future pinned generation config adds a processor this
        # manual path has not implemented. Sampling-only fields are ignored by
        # Transformers when do_sample=False and do not affect greedy argmax.
        unsupported = {
            "no_repeat_ngram_size": lambda value: bool(value),
            "encoder_repetition_penalty": lambda value: float(value) != 1.0,
            "bad_words_ids": lambda value: bool(value),
            "force_words_ids": lambda value: bool(value),
            "min_length": lambda value: bool(value),
            "min_new_tokens": lambda value: bool(value),
            "suppress_tokens": lambda value: bool(value),
            "begin_suppress_tokens": lambda value: bool(value),
            "sequence_bias": lambda value: bool(value),
            "forced_bos_token_id": lambda value: value is not None,
            "forced_eos_token_id": lambda value: value is not None,
            "constraints": lambda value: bool(value),
            "watermarking_config": lambda value: value is not None,
        }
        active = [
            name
            for name, is_active in unsupported.items()
            if generation_config is not None
            and is_active(getattr(generation_config, name, None))
        ]
        if active:
            raise ValueError(
                "Unsupported active generation processors for manual greedy "
                f"decoding: {', '.join(active)}"
            )

    def process(self, logits: torch.Tensor, history: list[int]) -> torch.Tensor:
        """Return scores after applying repetition penalty over prompt+output."""
        if self.repetition_penalty == 1.0 or not history:
            return logits
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("Manual greedy decoding supports batch size one")
        token_ids = torch.tensor([history], dtype=torch.long, device=logits.device)
        scores = logits.clone()
        previous_scores = torch.gather(scores, 1, token_ids)
        adjusted = torch.where(
            previous_scores < 0,
            previous_scores * self.repetition_penalty,
            previous_scores / self.repetition_penalty,
        )
        scores.scatter_(1, token_ids, adjusted)
        return scores

    def select(self, logits: torch.Tensor, history: list[int]) -> int:
        return int(torch.argmax(self.process(logits, history), dim=-1).item())
