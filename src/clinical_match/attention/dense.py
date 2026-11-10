"""Dense scaled dot-product attention with explicit scaling and masking."""
from __future__ import annotations

import math
from typing import Any

import torch


def _expand_kv(k: torch.Tensor, v: torch.Tensor, query_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    key_heads = k.shape[1]
    if key_heads <= 0:
        raise ValueError("Key/value head count must be positive")
    if query_heads % key_heads != 0:
        raise ValueError("Query heads must be a multiple of key/value heads")
    groups = query_heads // key_heads
    if groups == 1:
        return k, v
    return (
        k.repeat_interleave(groups, dim=1),
        v.repeat_interleave(groups, dim=1),
    )


def _prepare_mask(mask: Any, scores: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    mask = torch.as_tensor(mask, device=scores.device, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError("Attention mask must have 2, 3 or 4 dimensions")
    try:
        return mask.expand_as(scores)
    except RuntimeError as exc:
        raise ValueError("Attention mask is not broadcastable to scores") from exc


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Any | None = None,
    *,
    scale: float | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute ``softmax(QK^T / sqrt(d) + mask) V`` in FP32.

    Shapes are ``[batch, heads, length, head_dim]`` for Q and V, and
    ``[batch, key_heads, key_length, head_dim]`` for K. GQA/MQA key/value heads
    are repeated to match query heads. Fully masked rows return zero.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K and V must be 4-dimensional")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("Head dimensions must match")
    if k.shape[:2] != v.shape[:2] or k.shape[-2] != v.shape[-2]:
        raise ValueError("K and V batch/head/key shapes must match")
    if k.shape[-2] <= 0:
        raise ValueError("Key/value length must be positive")
    k, v = _expand_kv(k, v, q.shape[1])
    if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0]:
        raise ValueError("Batch sizes must match")
    q32 = q.float()
    k32 = k.float()
    v32 = v.float()
    scores = (q32 @ k32.transpose(-2, -1)) * (
        scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    )
    visible = _prepare_mask(mask, scores)
    if visible is not None:
        scores = scores.masked_fill(~visible, float("-inf"))
    row_max = scores.max(dim=-1).values
    finite_rows = torch.isfinite(row_max)
    safe_max = torch.where(finite_rows, row_max, torch.zeros_like(row_max))
    weights = torch.exp(scores - safe_max.unsqueeze(-1))
    if visible is not None:
        weights = weights.masked_fill(~visible, 0.0)
    else:
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.where(
        finite_rows.unsqueeze(-1), weights, torch.zeros_like(weights)
    )
    denominator = weights.sum(dim=-1, keepdim=True)
    output = weights @ v32
    output = torch.where(
        denominator > 0, output / denominator, torch.zeros_like(output)
    )
    return output.to(output_dtype or q.dtype)
