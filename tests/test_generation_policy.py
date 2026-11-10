import unittest
from types import SimpleNamespace

import torch

from src.clinical_match.decoding.policy import GreedyLogitsPolicy


class GreedyLogitsPolicyTests(unittest.TestCase):
    def test_repetition_penalty_adjusts_seen_positive_and_negative_scores(self):
        model = SimpleNamespace(
            generation_config=SimpleNamespace(
                repetition_penalty=2.0,
                no_repeat_ngram_size=0,
                encoder_repetition_penalty=1.0,
                bad_words_ids=None,
                force_words_ids=None,
                min_length=0,
                min_new_tokens=None,
                suppress_tokens=None,
                begin_suppress_tokens=None,
                sequence_bias=None,
                forced_bos_token_id=None,
                forced_eos_token_id=None,
                constraints=None,
                watermarking_config=None,
            )
        )
        policy = GreedyLogitsPolicy(model)
        scores = torch.tensor([[4.0, -4.0, 3.0, -3.0]])
        adjusted = policy.process(scores, [0, 1, 1])
        self.assertEqual(adjusted.tolist(), [[2.0, -8.0, 3.0, -3.0]])

    def test_rejects_unimplemented_active_processor(self):
        config = SimpleNamespace(
            repetition_penalty=1.0,
            no_repeat_ngram_size=2,
            encoder_repetition_penalty=1.0,
            bad_words_ids=None,
            force_words_ids=None,
            min_length=0,
            min_new_tokens=None,
            suppress_tokens=None,
            begin_suppress_tokens=None,
            sequence_bias=None,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            constraints=None,
            watermarking_config=None,
        )
        with self.assertRaisesRegex(ValueError, "no_repeat_ngram_size"):
            GreedyLogitsPolicy(SimpleNamespace(generation_config=config))


if __name__ == "__main__":
    unittest.main()
