import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.clinical_match.decoding import (
    ExplicitKVCache,
    cached_explicit_greedy,
    uncached_greedy,
)


def tiny_model():
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return Qwen2ForCausalLM(config).eval()


class CachedDecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.device = torch.device("cpu")

    def test_token_identity_across_three_lengths(self):
        cases = [([1, 2, 3], 1), ([4, 5, 6, 7, 8, 9, 10], 4), (list(range(11, 24)), 8)]
        for prompt, cap in cases:
            with self.subTest(prompt=len(prompt), cap=cap):
                reference = uncached_greedy(
                    self.model,
                    prompt,
                    max_new_tokens=cap,
                    eos_token_id=None,
                    device=self.device,
                )
                cached = cached_explicit_greedy(
                    self.model,
                    prompt,
                    max_new_tokens=cap,
                    eos_token_id=None,
                    device=self.device,
                )
                self.assertEqual(reference["token_ids"], cached["token_ids"])
                self.assertEqual(cached["prefill_cache_length"], len(prompt))
                self.assertEqual(
                    cached["final_cache_length"], len(prompt) + max(0, cap - 1)
                )
                self.assertEqual(
                    cached["cache_lengths"],
                    list(range(len(prompt), len(prompt) + cap)),
                )
                self.assertTrue(cached["hit_token_cap"])

    def test_eos_stops_without_feeding_eos_to_cache(self):
        first = uncached_greedy(
            self.model, [1, 2, 3], max_new_tokens=1, eos_token_id=None, device=self.device
        )["token_ids"][0]
        cached = cached_explicit_greedy(
            self.model,
            [1, 2, 3],
            max_new_tokens=8,
            eos_token_id=first,
            device=self.device,
        )
        self.assertEqual(cached["token_ids"], [first])
        self.assertFalse(cached["hit_token_cap"])
        self.assertEqual(cached["termination"], "eos")
        self.assertEqual(cached["final_cache_length"], 3)
        self.assertEqual(cached["generated_cache_resident"], 0)

    def test_cache_layers_positions_and_crop(self):
        result = cached_explicit_greedy(
            self.model,
            [1, 2, 3, 4],
            max_new_tokens=3,
            eos_token_id=None,
            device=self.device,
            return_cache=True,
        )
        cache = result["cache"]
        self.assertIsInstance(cache, ExplicitKVCache)
        self.assertEqual(cache.length, 6)
        self.assertEqual(cache.layer_shapes()[0]["keys"][0], 1)
        self.assertEqual(cache.layer_shapes()[0]["keys"][1], 2)
        cache.assert_layer_count(2)
        self.assertEqual(result["cache_positions"][0], [0, 1, 2, 3])
        self.assertEqual(result["cache_positions"][1], [4])
        cache.crop(4)
        self.assertEqual(cache.length, 4)
        with self.assertRaises(RuntimeError):
            cache.assert_length(5)
        with self.assertRaises(ValueError):
            cache.crop(99)

    def test_rejects_empty_prompt_and_nonpositive_cap(self):
        with self.assertRaises(ValueError):
            cached_explicit_greedy(
                self.model,
                [],
                max_new_tokens=2,
                eos_token_id=None,
                device=self.device,
            )
        with self.assertRaises(ValueError):
            cached_explicit_greedy(
                self.model,
                [1],
                max_new_tokens=0,
                eos_token_id=None,
                device=self.device,
            )


if __name__ == "__main__":
    unittest.main()
