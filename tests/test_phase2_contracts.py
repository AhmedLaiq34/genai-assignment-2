import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.clinical_match.data_access import load_examples
from src.clinical_match.prompts import build_prompt, target_json


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(message["content"]) for message in messages)))


class ContractsTest(unittest.TestCase):
    def test_unapproved_quarantine_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/splits').mkdir(parents=True)
            manifest = {'row_counts': {'train': {'quarantined': 1}},
                        'excluded_training_annotation_ids': [884]}
            (root / 'data/splits/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Unapproved'):
                load_examples(root, 'train', allow_quarantined_training=True)

    def test_prompt_allowlist_and_sentence_preserving_truncation(self):
        tokenizer = CharacterTokenizer()
        base = {"criterion_type": "inclusion", "criterion_text": "Age >= 18",
                "sentences": [{"id": 3, "text": "Age 18.", "gpt4_explanation": "NESTED_LEAK"},
                              {"id": 8, "text": "x" * 500}],
                "expert_eligibility": "SECRET_GOLD", "gpt4_explanation": "LEAK"}
        output = build_prompt(tokenizer, base, max_total_tokens=900, max_new_tokens=96)
        self.assertEqual(output["kept_sentence_ids"], [3])
        self.assertEqual(output["removed_sentence_ids"], [8])
        self.assertNotIn("SECRET_GOLD", str(output["messages"]))
        self.assertNotIn("LEAK", str(output["messages"]))
        self.assertIn('"id":3', output["messages"][1]["content"])
        with self.assertRaises(ValueError):
            build_prompt(tokenizer, base, max_total_tokens=10, max_new_tokens=9)

    def test_target_is_canonical(self):
        self.assertEqual(target_json({"label": "included", "evidence_sentence_ids": [2, 0, 2]}),
                         '{"label":"included","evidence_sentence_ids":[0,2]}')
        with self.assertRaises(ValueError):
            target_json({"label": "included", "evidence_sentence_ids": [True]})

    def test_test_access_requires_matching_complete_frozen_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/splits').mkdir(parents=True)
            (root / 'data/processed/protected_test').mkdir(parents=True)
            manifest = root / 'data/splits/manifest.json'
            manifest.write_text(json.dumps({'dataset_revision': 'abc'}))
            (root / 'data/processed/protected_test/test.jsonl').write_text('{"synthetic":true}\n')
            with self.assertRaises(PermissionError):
                load_examples(root, 'test')
            frozen = root / 'frozen.json'
            data = {'state': 'frozen', 'dataset_revision': 'abc',
                    'split_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    'model_revision': 'model', 'adapter_sha256': 'adapter', 'prompt_sha256': 'prompt',
                    'generation_config_sha256': 'generation', 'metric_config_sha256': 'metrics'}
            frozen.write_text(json.dumps(data))
            sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                load_examples(root, 'test', allow_final_test=True, frozen_manifest=frozen, frozen_sha256='bad')
            self.assertEqual(load_examples(root, 'test', allow_final_test=True, frozen_manifest=frozen,
                                           frozen_sha256=sha), [{'synthetic': True}])
            manifest.write_text('{"dataset_revision":"changed"}')
            with self.assertRaises(ValueError):
                load_examples(root, 'test', allow_final_test=True, frozen_manifest=frozen, frozen_sha256=sha)


if __name__ == '__main__':
    unittest.main()
