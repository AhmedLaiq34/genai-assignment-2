"""Tiled attention with online maximum, normalizer and value accumulation."""
from __future__ import annotations

import math
from typing import Any

import torch

from .dense import _expand_kv


def _normalize_mask(mask: Any, batch: int, heads: int, q_len: int, k_len: int, device) -> torch.Tensor | None:
    if mask is None:
        return None
    mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError("Attention mask must have 2, 3 or 4 dimensions")
    try:
        return mask.expand(batch, heads, q_len, k_len)
    except RuntimeError as exc:
        raise ValueError("Attention mask is not broadcastable to [batch, heads, q, k]") from exc


def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Any | None = None,
    *,
    query_tile: int = 32,
    key_tile: int = 32,
    scale: float | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Online-softmax attention without materializing the full score matrix.

    Shapes match ``dense_attention``. Running maximum, normalizer and weighted
    value accumulator are FP32. Each score tile is ``[batch, heads, query_tile,
    key_tile]``; a fully masked row retains zero state and returns zero.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K and V must be 4-dimensional")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("Head dimensions must match")
    if k.shape[:2] != v.shape[:2] or k.shape[-2] != v.shape[-2]:
        raise ValueError("K and V batch/head/key shapes must match")
    if k.shape[-2] <= 0:
        raise ValueError("Key/value length must be positive")
    if query_tile <= 0 or key_tile <= 0:
        raise ValueError("Tile sizes must be positive")
    k, v = _expand_kv(k, v, q.shape[1])
    if k.shape[0] != q.shape[0]:
        raise ValueError("Batch sizes must match")
    batch, heads, q_len, head_dim = q.shape
    k_len = k.shape[-2]
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    visible = _normalize_mask(mask, batch, heads, q_len, k_len, q.device)
    q32 = q.float()
    k32 = k.float()
    v32 = v.float()
    output = torch.zeros(
        (batch, heads, q_len, head_dim), dtype=torch.float32, device=q.device
    )

    for q_start in range(0, q_len, query_tile):
        q_end = min(q_start + query_tile, q_len)
        q_block = q32[:, :, q_start:q_end, :]
        block_len = q_end - q_start
        running_max = torch.full(
            (batch, heads, block_len), float("-inf"), dtype=torch.float32, device=q.device
        )
        normalizer = torch.zeros_like(running_max)
        accumulator = torch.zeros(
            (batch, heads, block_len, head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        for k_start in range(0, k_len, key_tile):
            k_end = min(k_start + key_tile, k_len)
            k_block = k32[:, :, k_start:k_end, :]
            v_block = v32[:, :, k_start:k_end, :]
            scores = (q_block @ k_block.transpose(-2, -1)) * scale
            tile_mask = (
                visible[:, :, q_start:q_end, k_start:k_end]
                if visible is not None
                else torch.ones_like(scores, dtype=torch.bool)
            )
            scores = scores.masked_fill(~tile_mask, float("-inf"))
            row_has = tile_mask.any(dim=-1)
            tile_max = scores.max(dim=-1).values
            tile_max = torch.where(row_has, tile_max, float("-inf"))
            new_max = torch.maximum(running_max, tile_max)
            finite_rows = torch.isfinite(new_max)
            alpha = torch.where(
                finite_rows,
                torch.exp(running_max - new_max),
                torch.zeros_like(new_max),
            )
            weights = torch.exp(scores - new_max.unsqueeze(-1))
            weights = torch.where(
                tile_mask & finite_rows.unsqueeze(-1),
                weights,
                torch.zeros_like(weights),
            )
            normalizer = alpha * normalizer + weights.sum(dim=-1)
            accumulator = (
                alpha.unsqueeze(-1) * accumulator
                + weights @ v_block
            )
            running_max = new_max
        result = torch.where(
            normalizer.unsqueeze(-1) > 0,
            accumulator / normalizer.unsqueeze(-1),
            torch.zeros_like(accumulator),
        )
        output[:, :, q_start:q_end, :] = result
    return output.to(output_dtype or q.dtype)
