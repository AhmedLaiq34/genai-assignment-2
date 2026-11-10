import unittest

from src.clinical_match.training.data import (
    build_training_set,
    collate_encoded,
    encode_example,
    order_for_epoch,
)


class FakeTokenizer:
    eos_token_id = 999
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [(ord(character) % 200) + 1 for character in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = []
        for message in messages:
            if message["role"] == "assistant":
                ids.append(777)
            ids.extend(self.encode(message["content"]))
            ids.append(len(message["role"]))
        if add_generation_prompt:
            ids.append(777)
        else:
            ids.extend([self.eos_token_id, 998])
        return ids


def example(label="included", evidence=(2, 0)):
    return {
        "provenance": {"annotation_id": 7, "patient_id": "p1", "trial_id": "NCT1"},
        "input": {
            "criterion_type": "inclusion",
            "criterion_text": "Age at least 18",
            "sentences": [
                {"id": 0, "text": "The patient is 24 years old."},
                {"id": 2, "text": "No exclusion applies."},
            ],
        },
        "target": {"label": label, "evidence_sentence_ids": list(evidence)},
    }


class TrainingDataTests(unittest.TestCase):
    def test_only_assistant_answer_is_supervised(self):
        tokenizer = FakeTokenizer()
        encoded = encode_example(
            tokenizer,
            example(),
            max_total_tokens=900,
            max_new_tokens=50,
            max_seq_len=900,
        )
        self.assertEqual(encoded.labels[: encoded.prompt_length], [-100] * encoded.prompt_length)
        self.assertEqual(encoded.answer_text, '{"label":"included","evidence_sentence_ids":[0,2]}')
        tail_labels = encoded.labels[encoded.prompt_length :]
        tail_ids = encoded.input_ids[encoded.prompt_length :]
        eos_index = tail_ids.index(tokenizer.eos_token_id)
        self.assertEqual(tail_labels[: eos_index + 1], tail_ids[: eos_index + 1])
        self.assertTrue(all(value == -100 for value in tail_labels[eos_index + 1 :]))
        self.assertEqual(tail_labels[-1], -100)
        self.assertGreater(encoded.target_token_count, 0)

    def test_prompt_prefix_is_preserved(self):
        tokenizer = FakeTokenizer()
        encoded = encode_example(
            tokenizer,
            example(),
            max_total_tokens=900,
            max_new_tokens=50,
            max_seq_len=900,
        )
        self.assertEqual(encoded.input_ids[: encoded.prompt_length][-1], 777)
        self.assertEqual(
            encoded.input_ids[encoded.prompt_length : encoded.prompt_length + len(encoded.answer_text)],
            tokenizer.encode(encoded.answer_text),
        )

    def test_collation_masks_padding(self):
        tokenizer = FakeTokenizer()
        short = encode_example(
            tokenizer, example(), max_total_tokens=900, max_new_tokens=50, max_seq_len=900
        )
        long_example = example()
        long_example["input"]["sentences"].append({"id": 3, "text": "A much longer sentence " * 5})
        longer = encode_example(
            tokenizer, long_example, max_total_tokens=900, max_new_tokens=50, max_seq_len=900
        )
        batch = collate_encoded([short, longer], tokenizer.pad_token_id)
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertTrue((batch["labels"][0, -1] == -100).item())
        self.assertTrue((batch["attention_mask"][1, -1] == 1).item())
        self.assertEqual(batch["attention_mask"].shape, batch["input_ids"].shape)

    def test_order_is_deterministic_and_epoch_dependent(self):
        first = order_for_epoch(50, 42, 0)
        self.assertEqual(first, order_for_epoch(50, 42, 0))
        self.assertNotEqual(first, order_for_epoch(50, 42, 1))
        self.assertEqual(sorted(first), list(range(50)))

    def test_build_training_set_preserves_source_ids(self):
        tokenizer = FakeTokenizer()
        encoded = build_training_set(
            tokenizer,
            [example(), example(label="not included", evidence=())],
            max_total_tokens=900,
            max_new_tokens=50,
            max_seq_len=900,
        )
        self.assertEqual(len(encoded), 2)
        self.assertEqual(encoded[0].annotation_id, 7)
        self.assertIn("[0,2]", encoded[0].answer_text)
        self.assertIn("[]", encoded[1].answer_text)


if __name__ == "__main__":
    unittest.main()
