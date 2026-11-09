import unittest
import tempfile
from pathlib import Path

from scripts.prepare_data import (
    build_example,
    parse_evidence,
    parse_note,
    split_patients,
    _write_json,
)


class PrepareDataTests(unittest.TestCase):
    def test_manifest_json_serialization_has_portable_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            _write_json(path, {"x": 1})
            self.assertEqual(path.read_bytes(), b'{\n  "x": 1\n}\n')

    def test_split_is_deterministic_and_patient_grouped(self):
        patient_ids = [f"patient-{index:02d}" for index in range(53)]
        first = split_patients(patient_ids)
        second = split_patients(list(reversed(patient_ids)))
        self.assertEqual(first, second)
        self.assertEqual(
            [len(first[name]) for name in ("train", "validation", "test")],
            [37, 8, 8],
        )
        all_ids = [patient_id for group in first.values() for patient_id in group]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), set(patient_ids))

    def test_note_parser_preserves_original_ids_and_decimal_periods(self):
        sentences = parse_note("0. EF was 3.5%.\n1. No prior history.")
        self.assertEqual(
            sentences,
            [
                {"id": 0, "text": "EF was 3.5%."},
                {"id": 1, "text": "No prior history."},
            ],
        )

    def test_evidence_parser_sorts_deduplicates_and_checks_ids(self):
        self.assertEqual(parse_evidence("[2, 0, 2]", {0, 1, 2}), [0, 2])
        self.assertEqual(parse_evidence("[]", {0, 1, 2}), [])
        with self.assertRaisesRegex(ValueError, "do not exist"):
            parse_evidence("[3]", {0, 1, 2})
        with self.assertRaises(ValueError):
            parse_evidence("[True]", {0, 1, 2})

    def test_wrapped_sentence_keeps_id_and_continuation(self):
        self.assertEqual(parse_note("0. First line\n3.5 mg continued.\n2. Last."),
                         [{"id": 0, "text": "First line\n3.5 mg continued."},
                          {"id": 2, "text": "Last."}])
        with self.assertRaises(ValueError):
            parse_note("unheaded text\n0. Sentence")
        with self.assertRaises(ValueError):
            parse_note("0. First\n0. Duplicate")

    def test_prepared_example_omits_prior_ai_fields(self):
        row = {
            "annotation_id": 7,
            "patient_id": "patient-a",
            "trial_id": "trial-a",
            "note": "0. Patient is 58.",
            "criterion_type": "inclusion",
            "criterion_text": "Age is at least 18.",
            "expert_sentences": "[0]",
            "expert_eligibility": "included",
            "gpt4_explanation": "must not appear",
            "gpt4_sentences": "[0]",
            "gpt4_eligibility": "included",
        }
        example = build_example(row)
        self.assertEqual(
            set(example),
            {"provenance", "input", "target"},
        )
        self.assertEqual(
            set(example["input"]),
            {"criterion_type", "criterion_text", "sentences"},
        )
        self.assertEqual(
            example["target"],
            {"label": "included", "evidence_sentence_ids": [0]},
        )
        self.assertNotIn("gpt4_explanation", str(example))
        self.assertEqual(example["provenance"]["source_expert_sentences"], "[0]")


if __name__ == "__main__":
    unittest.main()
