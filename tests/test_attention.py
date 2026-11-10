import math
import unittest

import torch

from src.clinical_match.attention import causal_mask, dense_attention, tiled_attention


def make_qkv(batch, heads, length, head_dim, kv_heads, seed=0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, length, head_dim, generator=generator, dtype=dtype)
    k = torch.randn(batch, kv_heads, length, head_dim, generator=generator, dtype=dtype)
    v = torch.randn(batch, kv_heads, length, head_dim, generator=generator, dtype=dtype)
    return q, k, v


def oracle_attention(q, k, v, mask):
    groups = q.shape[1] // k.shape[1]
    k = torch.stack([k[:, head // groups] for head in range(q.shape[1])], dim=1).double()
    v = torch.stack([v[:, head // groups] for head in range(q.shape[1])], dim=1).double()
    scores = q.double() @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    row_max = scores.max(dim=-1).values
    safe = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    weights = torch.exp(scores - safe.unsqueeze(-1))
    weights = torch.where(
        mask & torch.isfinite(row_max).unsqueeze(-1),
        weights,
        torch.zeros_like(weights),
    )
    denominator = weights.sum(dim=-1, keepdim=True)
    output = weights @ v
    return torch.where(
        denominator > 0, output / denominator, torch.zeros_like(output)
    ).to(q.dtype)


class AttentionTests(unittest.TestCase):
    def assert_attention_close(self, actual, expected, atol=1e-5, rtol=1e-4):
        self.assertFalse(torch.isnan(actual).any(), "Attention produced NaN")
        self.assertFalse(torch.isinf(actual).any(), "Attention produced infinity")
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_dense_matches_float64_oracle(self):
        q, k, v = make_qkv(2, 4, 33, 32, 2, seed=1)
        mask = causal_mask(33, 33).unsqueeze(0).unsqueeze(0)
        actual = dense_attention(q, k, v, mask)
        self.assert_attention_close(actual, oracle_attention(q, k, v, mask))

    def test_causal_mask_direct_geometry_and_default_offsets(self):
        mask = causal_mask(4, 4)
        expected = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))
        rectangular = causal_mask(2, 10)
        self.assertTrue(torch.equal(rectangular[0], torch.arange(10) <= 8))
        self.assertTrue(torch.equal(rectangular[1], torch.arange(10) <= 9))
        wider = causal_mask(10, 3)
        self.assertTrue(
            torch.equal(wider, torch.arange(3).unsqueeze(0) <= torch.arange(10).unsqueeze(1))
        )

    def test_2d_and_3d_mask_broadcast(self):
        q, k, v = make_qkv(2, 4, 8, 32, 2, seed=10)
        mask2d = causal_mask(8, 8)
        mask3d = torch.stack([mask2d, torch.ones_like(mask2d)])
        masks4d = {
            "2d": mask2d,
            "3d": mask3d,
            "4d_per_head": torch.stack([mask2d, mask2d.flip(0)]).unsqueeze(1).expand(2, 4, 8, 8),
        }
        for name, mask in masks4d.items():
            expanded = mask if mask.ndim == 4 else (
                mask.unsqueeze(0).unsqueeze(0) if mask.ndim == 2 else mask.unsqueeze(1)
            )
            expanded = expanded.expand(2, 4, 8, 8)
            with self.subTest(mask=name):
                expected = oracle_attention(q, k, v, expanded)
                self.assert_attention_close(dense_attention(q, k, v, mask), expected)
                self.assert_attention_close(
                    tiled_attention(q, k, v, mask, query_tile=3, key_tile=3),
                    expected,
                )

    def test_lengths_tiles_and_causal_rectangular_shapes(self):
        for length in (1, 17, 64, 129, 256):
            q, k, v = make_qkv(2, 4, length, 32, 2, seed=length)
            masks = {
                "causal": causal_mask(length, length).unsqueeze(0).unsqueeze(0),
                "full": None,
            }
            for mask_name, mask in masks.items():
                expected = dense_attention(q, k, v, mask)
                for query_tile in (16, 32, 64):
                    for key_tile in (16, 32, 64):
                        with self.subTest(
                            length=length,
                            mask=mask_name,
                            query_tile=query_tile,
                            key_tile=key_tile,
                        ):
                            actual = tiled_attention(
                                q,
                                k,
                                v,
                                mask,
                                query_tile=query_tile,
                                key_tile=key_tile,
                            )
                            self.assert_attention_close(actual, expected)

    def test_head_dimensions_and_gqa_mapping(self):
        for head_dim in (32, 64, 128):
            for kv_heads in (1, 2, 4):
                with self.subTest(head_dim=head_dim, kv_heads=kv_heads):
                    q, k, v = make_qkv(2, 4, 65, head_dim, kv_heads, seed=head_dim)
                    mask = causal_mask(65, 65).unsqueeze(0).unsqueeze(0)
                    expected = oracle_attention(q, k, v, mask)
                    dense = dense_attention(q, k, v, mask)
                    actual = tiled_attention(
                        q, k, v, mask, query_tile=16, key_tile=32
                    )
                    self.assert_attention_close(dense, expected)
                    self.assert_attention_close(actual, expected)

    def test_large_logits_do_not_overflow(self):
        q, k, v = make_qkv(1, 4, 65, 64, 2, seed=4)
        q = q * 50.0
        for mask in (None, causal_mask(65, 65).unsqueeze(0).unsqueeze(0)):
            expected = dense_attention(q, k, v, mask)
            actual = tiled_attention(q, k, v, mask, query_tile=17, key_tile=17)
            self.assert_attention_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_fully_masked_rows_return_zero(self):
        q, k, v = make_qkv(1, 4, 8, 32, 2, seed=5)
        mask = torch.ones(8, 8, dtype=torch.bool)
        mask[2, :] = False
        mask[5, 3:] = False
        mask = mask.unsqueeze(0).unsqueeze(0)
        expected = dense_attention(q, k, v, mask)
        actual = tiled_attention(q, k, v, mask, query_tile=3, key_tile=3)
        self.assertTrue(torch.equal(expected[0, :, 2], torch.zeros_like(expected[0, :, 2])))
        self.assert_attention_close(actual, expected)

    def test_rectangular_cached_offset_has_no_future_leakage(self):
        full_q, k, v = make_qkv(1, 4, 10, 32, 2, seed=7)
        for query_start in (7, 8):
            q = full_q[:, :, query_start : query_start + 1]
            mask = causal_mask(1, 10, query_offset=query_start).unsqueeze(0).unsqueeze(0)
            expected = dense_attention(q, k, v, mask)
            actual = tiled_attention(q, k, v, mask, query_tile=2, key_tile=4)
            self.assert_attention_close(actual, expected)
            for future_key in range(query_start + 1, 10):
                changed_k, changed_v = k.clone(), v.clone()
                changed_k[:, :, future_key] += 1000.0
                changed_v[:, :, future_key] += 1000.0
                changed = tiled_attention(
                    q, changed_k, changed_v, mask, query_tile=2, key_tile=4
                )
                with self.subTest(query_start=query_start, future_key=future_key):
                    self.assert_attention_close(changed, actual)
        default_mask = causal_mask(2, 10)
        explicit_mask = causal_mask(2, 10, query_offset=8)
        self.assertTrue(torch.equal(default_mask, explicit_mask))

    def test_fp16_inputs_accumulate_at_least_near_fp32(self):
        q, k, v = make_qkv(1, 4, 65, 64, 2, seed=8, dtype=torch.float16)
        mask = causal_mask(65, 65).unsqueeze(0).unsqueeze(0)
        expected = dense_attention(q.float(), k.float(), v.float(), mask)
        actual = tiled_attention(
            q, k, v, mask, query_tile=16, key_tile=16
        )
        self.assertEqual(actual.dtype, torch.float16)
        self.assert_attention_close(actual.float(), expected, atol=2e-3, rtol=2e-3)

    def test_invalid_gqa_zero_length_and_mask_shape_raise(self):
        q, _, _ = make_qkv(1, 4, 8, 32, 2, seed=9)
        k = torch.randn(1, 3, 8, 32)
        v = torch.randn(1, 3, 8, 32)
        with self.assertRaises(ValueError):
            dense_attention(q, k, v)
        with self.assertRaises(ValueError):
            tiled_attention(q, k, v)
        _, k, v = make_qkv(1, 4, 8, 32, 2, seed=9)
        with self.assertRaises(ValueError):
            tiled_attention(q, k, v, torch.ones(3, 3, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            dense_attention(q, k, v, torch.ones(3, 3, 3, dtype=torch.bool))
        empty_k = torch.empty(1, 2, 0, 32)
        empty_v = torch.empty(1, 2, 0, 32)
        with self.assertRaises(ValueError):
            dense_attention(q, empty_k, empty_v)
        with self.assertRaises(ValueError):
            tiled_attention(q, empty_k, empty_v)


if __name__ == "__main__":
    unittest.main()
