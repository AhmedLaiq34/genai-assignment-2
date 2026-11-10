"""Explicit boolean attention masks using absolute token positions."""
from __future__ import annotations

import torch


def causal_mask(
    query_length: int,
    key_length: int,
    *,
    query_offset: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return [query, key] visibility for prefill or cached rectangular attention.

    A key at absolute position j is visible to a query at absolute position i when
    j <= i. With a rectangular cached shape, the default query offset is
    ``max(key_length - query_length, 0)`` so the newest queries see the full
    prefix; callers with a different alignment can pass ``query_offset``.
    """
    if query_length <= 0 or key_length <= 0:
        raise ValueError("Mask lengths must be positive")
    if query_offset is None:
        query_offset = max(key_length - query_length, 0)
    if query_offset < 0:
        raise ValueError("query_offset must be nonnegative")
    query_positions = torch.arange(
        query_offset, query_offset + query_length, device=device
    )
    key_positions = torch.arange(key_length, device=device)
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
