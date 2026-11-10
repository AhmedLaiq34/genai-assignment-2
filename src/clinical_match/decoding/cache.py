"""Explicit, inspectable key/value cache state for batch-one decoding."""
from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache


class ExplicitKVCache:
    """Small wrapper around Transformers' DynamicCache with checked invariants."""

    def __init__(self, cache: Any | None = None) -> None:
        self.cache = cache if cache is not None else DynamicCache()

    @property
    def length(self) -> int:
        return int(self.cache.get_seq_length())

    def assert_length(self, expected: int) -> None:
        observed = self.length
        if observed != expected:
            raise RuntimeError(
                f"KV cache length mismatch: expected {expected}, observed {observed}"
            )

    def layer_shapes(self) -> list[dict[str, tuple[int, ...]]]:
        shapes: list[dict[str, tuple[int, ...]]] = []
        for index, layer in enumerate(self.cache.layers):
            if not getattr(layer, "is_initialized", False):
                continue
            shapes.append(
                {
                    "layer": index,
                    "keys": tuple(layer.keys.shape),
                    "values": tuple(layer.values.shape),
                }
            )
        return shapes

    def assert_layer_count(self, expected: int) -> None:
        observed = len(self.layer_shapes())
        if observed != expected:
            raise RuntimeError(
                f"KV cache layer count mismatch: expected {expected}, observed {observed}"
            )

    def crop(self, length: int) -> None:
        if length < 0 or length > self.length:
            raise ValueError("Invalid cache crop length")
        self.cache.crop(length)
        self.assert_length(length)

    def state_dict(self) -> dict[str, Any]:
        initialized = [
            layer
            for layer in self.cache.layers
            if getattr(layer, "is_initialized", False)
        ]
        return {
            "length": self.length,
            "layers": self.layer_shapes(),
            "dtype": str(initialized[0].keys.dtype) if initialized else None,
        }

    def tensors(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (layer.keys, layer.values)
            for layer in self.cache.layers
            if getattr(layer, "is_initialized", False)
        ]
