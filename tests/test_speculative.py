import unittest
from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.clinical_match.decoding import cached_explicit_greedy, speculative_greedy


class MockCache:
    def __init__(self, length=0):
        self.length = length
        self.layers = []

    def get_seq_length(self):
        return self.length

    def crop(self, length):
        if length < 0 or length > self.length:
            raise ValueError("invalid crop")
        self.length = length


class MockLM:
    def __init__(self, next_by_length, vocab_size=128):
        self.next_by_length = next_by_length
        self.vocab_size = vocab_size

    def __call__(self, input_ids, past_key_values=None, **kwargs):
        start = past_key_values.get_seq_length() if past_key_values is not None else 0
        length = input_ids.shape[1]
        logits = torch.zeros(1, length, self.vocab_size)
        for index in range(length):
            token = self.next_by_length.get(start + index + 1, 0)
            logits[0, index, token] = 10.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=MockCache(start + length),
        )


def tiny_model(seed):
    torch.manual_seed(seed)
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


class SpeculativeStateTests(unittest.TestCase):
    def test_full_acceptance_bonus_and_cap(self):
        target_map = {2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}
        result = speculative_greedy(
            MockLM(target_map),
            MockLM(target_map),
            [1, 2],
            max_new_tokens=6,
            eos_token_id=None,
            draft_k=4,
            device=torch.device("cpu"),
        )
        self.assertEqual(result["token_ids"], [10, 11, 12, 13, 14, 15])
        self.assertEqual(result["termination"], "cap")
        self.assertEqual(result["stats"]["accepted_tokens"], 5)
        self.assertEqual(result["stats"]["proposed_tokens"], 5)
        self.assertEqual(result["stats"]["bonus_tokens"], 1)
        self.assertEqual(result["stats"]["correction_tokens"], 0)
        self.assertEqual(result["target_cache_length"], 8)
        self.assertEqual(result["draft_cache_length"], 8)

    def test_partial_rejection_and_repeated_correction(self):
        target_map = {2: 10, 3: 99, 4: 20, 5: 21, 6: 22}
        draft_map = {2: 10, 3: 11, 4: 12, 5: 13, 6: 14}
        result = speculative_greedy(
            MockLM(target_map),
            MockLM(draft_map),
            [1, 2],
            max_new_tokens=3,
            eos_token_id=None,
            draft_k=3,
            device=torch.device("cpu"),
        )
        self.assertEqual(result["token_ids"], [10, 99, 20])
        self.assertEqual(result["blocks"][0]["accepted"], [10])
        self.assertEqual(result["blocks"][0]["correction"], 99)
        self.assertEqual(result["stats"]["accepted_tokens"], 1)
        self.assertEqual(result["stats"]["correction_tokens"], 2)
        self.assertEqual(result["target_cache_length"], 4)

    def test_rejection_then_full_acceptance_keeps_committed_prefix(self):
        target_map = {2: 10, 3: 99, 4: 20, 5: 21, 6: 22}
        draft_map = {2: 10, 3: 11, 4: 20, 5: 21, 6: 22}
        result = speculative_greedy(
            MockLM(target_map),
            MockLM(draft_map),
            [1, 2],
            max_new_tokens=5,
            eos_token_id=None,
            draft_k=3,
            device=torch.device("cpu"),
        )
        self.assertEqual(result["token_ids"], [10, 99, 20, 21, 22])
        self.assertEqual(result["blocks"][0]["accepted"], [10])
        self.assertEqual(result["blocks"][0]["correction"], 99)
        self.assertEqual(result["blocks"][1]["accepted"], [20, 21, 22])
        self.assertEqual(result["target_cache_length"], 7)
        self.assertEqual(result["draft_cache_length"], 7)

    def test_zero_acceptance_and_one_token_budget(self):
        result = speculative_greedy(
            MockLM({2: 99}),
            MockLM({2: 10}),
            [1, 2],
            max_new_tokens=1,
            eos_token_id=None,
            draft_k=4,
            device=torch.device("cpu"),
        )
        self.assertEqual(result["token_ids"], [99])
        self.assertEqual(result["blocks"][0]["accepted"], [])
        self.assertEqual(result["stats"]["accepted_tokens"], 0)
        self.assertEqual(result["stats"]["proposed_tokens"], 1)
        self.assertEqual(result["target_cache_length"], 2)

    def test_correction_eos_stops_without_committing_later_tokens(self):
        result = speculative_greedy(
            MockLM({2: 5}),
            MockLM({2: 10}),
            [1, 2],
            max_new_tokens=5,
            eos_token_id=5,
            draft_k=4,
            device=torch.device("cpu"),
        )
        self.assertEqual(result["token_ids"], [5])
        self.assertEqual(result["termination"], "eos")
        self.assertFalse(result["hit_token_cap"])

    def test_draft_eos_stops_and_caches_are_reconciled(self):
        result = speculative_greedy(
            MockLM({2: 10, 3: 5}),
            MockLM({2: 10, 3: 5}),
            [1, 2],
            max_new_tokens=5,
            eos_token_id=5,
            draft_k=4,
            device=torch.device("cpu"),
        )
        self.assertEqual(result["token_ids"], [10, 5])
        self.assertEqual(result["termination"], "eos")
        self.assertEqual(result["target_cache_length"], 3)
        self.assertEqual(result["draft_cache_length"], 3)
        self.assertEqual(result["blocks"][0]["target_cache_length"], 3)
        self.assertEqual(result["blocks"][0]["draft_cache_length"], 3)

    def test_bonus_eos_stops(self):
        target_map = {2: 10, 3: 11, 4: 12, 5: 13, 6: 5}
        result = speculative_greedy(
            MockLM(target_map),
            MockLM(target_map),
            [1, 2],
            max_new_tokens=10,
            eos_token_id=5,
            draft_k=4,
            device=torch.device("cpu"),
        )
        self.assertEqual(result["token_ids"], [10, 11, 12, 13, 5])
        self.assertEqual(result["termination"], "eos")
        self.assertEqual(result["stats"]["bonus_tokens"], 1)

    def test_one_token_remaining_budget_omits_bonus(self):
        result = speculative_greedy(
            MockLM({2: 10, 3: 11}),
            MockLM({2: 10, 3: 11}),
            [1, 2],
            max_new_tokens=1,
            eos_token_id=None,
            draft_k=4,
            device=torch.device("cpu"),
        )
        self.assertEqual(result["token_ids"], [10])
        self.assertEqual(result["stats"]["bonus_tokens"], 0)
        self.assertEqual(result["target_cache_length"], 3)


class SpeculativeRealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = tiny_model(11)
        cls.draft = tiny_model(12)
        cls.device = torch.device("cpu")

    def test_target_greedy_token_equivalence_for_k_values(self):
        prompt = [1, 2, 3, 4, 5, 6]
        reference = cached_explicit_greedy(
            self.target,
            prompt,
            max_new_tokens=10,
            eos_token_id=None,
            device=self.device,
        )
        for draft_k in (1, 2, 4):
            with self.subTest(draft_k=draft_k):
                speculative = speculative_greedy(
                    self.target,
                    self.draft,
                    prompt,
                    max_new_tokens=10,
                    eos_token_id=None,
                    draft_k=draft_k,
                    device=self.device,
                )
                self.assertEqual(
                    speculative["token_ids"], reference["token_ids"]
                )
                self.assertGreaterEqual(
                    speculative["target_cache_length"], len(prompt)
                )

    def test_eos_equivalence(self):
        prompt = [7, 8, 9]
        reference = cached_explicit_greedy(
            self.target,
            prompt,
            max_new_tokens=12,
            eos_token_id=2,
            device=self.device,
        )
        speculative = speculative_greedy(
            self.target,
            self.draft,
            prompt,
            max_new_tokens=12,
            eos_token_id=2,
            draft_k=4,
            device=self.device,
        )
        self.assertEqual(speculative["token_ids"], reference["token_ids"])


if __name__ == "__main__":
    unittest.main()

