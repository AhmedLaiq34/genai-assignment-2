import json
import unittest
from pathlib import Path

import jsonschema
import torch

from src.clinical_match.screening import (
    ScreeningService,
    aggregate_results,
    load_aggregation_policy,
    validate_screening_input,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    eos_token_id = 999
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = "\n".join(message["content"] for message in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return [ord(character) + 1 for character in text]

    def encode(self, text, add_special_tokens=False):
        return [ord(character) + 1 for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id - 1) for token_id in token_ids)


def raw_tokens(text):
    return [ord(character) + 1 for character in text]


SENTENCES = [
    {"id": 0, "text": "The patient is 58 years old."},
    {"id": 2, "text": "No prior chemotherapy is recorded."},
]


def base_cfg():
    return {
        "model_id": "fake/model",
        "model_revision": "revision",
        "prompt_revision": "v1-provisional",
        "max_total_tokens": 1024,
        "max_new_tokens": 96,
        "generation_backend": "fake",
    }


def service_with(raw, cfg_overrides=None):
    tokenizer = FakeTokenizer()
    model = object()
    cfg = base_cfg()
    cfg.update(cfg_overrides or {})

    def generate(*args, **kwargs):
        if isinstance(raw, Exception):
            raise raw
        return {"token_ids": raw_tokens(raw)}

    return ScreeningService(
        tokenizer, model, cfg, torch.device("cpu"), generate_fn=generate
    )


def capped_service(raw):
    tokenizer = FakeTokenizer()

    def generate(*args, **kwargs):
        return {"token_ids": raw_tokens(raw), "hit_token_cap": True}

    return ScreeningService(
        tokenizer, object(), base_cfg(), torch.device("cpu"), generate_fn=generate
    )


class ScreeningServiceTests(unittest.TestCase):
    def test_valid_result_has_exact_source_text_and_no_gold(self):
        raw = '{"label":"included","evidence_sentence_ids":[0]}'
        result = service_with(raw).screen(SENTENCES, "inclusion", "Age 18 or older")
        self.assertEqual(result["label"], "included")
        self.assertEqual(result["evidence_sentence_ids"], [0])
        self.assertEqual(result["evidence_excerpts"], [{"id": 0, "text": SENTENCES[0]["text"]}])
        self.assertFalse(any(result["flags"].values()))
        self.assertNotIn("gold_label", result)
        self.assertNotIn("gold_evidence_ids", result)
        self.assertIn("not a clinical enrollment decision", result["educational_notice"])
        schema = json.loads((ROOT / "schemas/screening_result.schema.json").read_text())
        jsonschema.validate(result, schema)

    def test_invalid_evidence_id_is_visible_and_not_quoted(self):
        raw = '{"label":"included","evidence_sentence_ids":[0,9]}'
        result = service_with(raw).screen(SENTENCES, "inclusion", "Age 18 or older")
        self.assertEqual(result["evidence_sentence_ids"], [0])
        self.assertEqual(result["invalid_evidence_values"], [9])
        self.assertEqual(len(result["evidence_excerpts"]), 1)
        self.assertTrue(result["flags"]["invalid_evidence_ids"])
        self.assertIn("does not exist", result["explanation"])

    def test_duplicate_ids_are_flagged(self):
        raw = '{"label":"included","evidence_sentence_ids":[0,0]}'
        result = service_with(raw).screen(SENTENCES, "inclusion", "Age 18 or older")
        self.assertEqual(result["evidence_sentence_ids"], [0])
        self.assertEqual(result["raw_evidence_values"], [0, 0])
        self.assertTrue(result["flags"]["duplicate_evidence_ids"])
        self.assertEqual(result["evidence_status"], "duplicate_ids")

    def test_output_cap_is_truncation(self):
        raw = '{"label":"included","evidence_sentence_ids":[0]}'
        result = capped_service(raw).screen(SENTENCES, "inclusion", "Age 18 or older")
        self.assertFalse(result["input_truncated"])
        self.assertTrue(result["output_truncated"])
        self.assertTrue(result["truncated"])
        self.assertTrue(result["flags"]["truncation"])

    def test_prompt_truncation_is_explicit_and_removed_ids_are_unseen(self):
        sentences = [
            {"id": 0, "text": "The patient is 58 years old."},
            {"id": 1, "text": "x" * 200},
        ]
        raw = '{"label":"included","evidence_sentence_ids":[1]}'
        result = service_with(
            raw, {"max_total_tokens": 700, "max_new_tokens": 10}
        ).screen(sentences, "inclusion", "Age 18 or older")
        self.assertTrue(result["input_truncated"])
        self.assertFalse(result["output_truncated"])
        self.assertEqual(result["removed_sentence_ids"], [1])
        self.assertEqual(result["omitted_sentence_count"], 1)
        self.assertEqual(result["invalid_evidence_values"], [1])
        self.assertTrue(result["flags"]["invalid_evidence_ids"])

    def test_prompt_build_failure_is_structured_not_raised(self):
        result = service_with(
            '{"label":"included","evidence_sentence_ids":[0]}',
            {"max_total_tokens": 200, "max_new_tokens": 10},
        ).screen(
            [{"id": 0, "text": "x" * 5000}],
            "inclusion",
            "Age 18 or older",
        )
        self.assertTrue(result["flags"]["runtime_failure"])
        self.assertTrue(result["flags"]["truncation"])
        self.assertEqual(result["parse_status"], "runtime_error")
        self.assertEqual(result["kept_sentence_ids"], [])
        self.assertEqual(result["removed_sentence_ids"], [0])

    def test_contradictory_sentences_force_review(self):
        sentences = [
            {"id": 0, "text": "The patient has no prior chemotherapy."},
            {"id": 1, "text": "Prior chemotherapy was completed."},
        ]
        raw = '{"label":"included","evidence_sentence_ids":[0]}'
        result = service_with(raw).screen(sentences, "inclusion", "Chemotherapy naive")
        self.assertTrue(result["flags"]["contradiction"])

    def test_malformed_json_preserves_raw_and_does_not_become_abstention(self):
        raw = "not json"
        result = service_with(raw).screen(SENTENCES, "exclusion", "No prior chemotherapy")
        self.assertEqual(result["label"], None)
        self.assertEqual(result["parse_status"], "json_error")
        self.assertTrue(result["flags"]["parse_failure"])
        self.assertEqual(result["raw_output"], raw)
        self.assertIn("requires review", result["explanation"])

    def test_runtime_failure_is_operational_not_a_label(self):
        result = service_with(RuntimeError("boom")).screen(
            SENTENCES, "inclusion", "Age 18 or older"
        )
        self.assertEqual(result["label"], None)
        self.assertTrue(result["flags"]["runtime_failure"])
        self.assertFalse(result["flags"]["parse_failure"])
        self.assertEqual(result["parse_status"], "runtime_error")
        self.assertEqual(result["evidence_status"], "runtime_error")
        self.assertEqual(result["runtime_error"], {"type": "RuntimeError", "message": "boom"})
        self.assertEqual(result["evidence_excerpts"], [])

    def test_not_enough_information_is_flagged(self):
        raw = '{"label":"not enough information","evidence_sentence_ids":[]}'
        result = service_with(raw).screen(SENTENCES, "exclusion", "No prior chemotherapy")
        self.assertEqual(result["label"], "not enough information")
        self.assertTrue(result["flags"]["missing_information"])
        self.assertEqual(result["evidence_status"], "no_ids")

    def test_shape_and_category_errors_are_parse_failures(self):
        shape = service_with("[1,2]").screen(SENTENCES, "inclusion", "Age 18 or older")
        self.assertEqual(shape["parse_status"], "shape_error")
        self.assertTrue(shape["flags"]["parse_failure"])
        illegal = service_with(
            '{"label":"excluded","evidence_sentence_ids":[0]}'
        ).screen(SENTENCES, "inclusion", "Age 18 or older")
        self.assertEqual(illegal["parse_status"], "illegal_category")
        self.assertEqual(illegal["label"], None)

    def test_non_integer_evidence_is_invalid(self):
        raw = '{"label":"included","evidence_sentence_ids":[0,"1",null]}'
        result = service_with(raw).screen(SENTENCES, "inclusion", "Age 18 or older")
        self.assertEqual(result["evidence_sentence_ids"], [0])
        self.assertEqual(result["invalid_evidence_values"], ["1", None])
        self.assertTrue(result["flags"]["invalid_evidence_ids"])

    def test_edge_case_fixtures_are_executable(self):
        fixtures = json.loads(
            (ROOT / "examples/screening_edge_cases.json").read_text(encoding="utf-8")
        )
        for case in fixtures["cases"]:
            validate_screening_input(
                case["patient_sentences"],
                case["criterion_type"],
                case["criterion_text"],
            )
        contradiction = next(
            case for case in fixtures["cases"] if case["name"] == "lexical_contradiction"
        )
        result = service_with(
            '{"label":"included","evidence_sentence_ids":[0]}'
        ).screen(
            contradiction["patient_sentences"],
            contradiction["criterion_type"],
            contradiction["criterion_text"],
        )
        self.assertTrue(result["flags"]["contradiction"])

    def test_input_with_gold_fields_is_rejected(self):
        poisoned = [{"id": 0, "text": "The patient is 58.", "gold_label": "included"}]
        with self.assertRaises(ValueError):
            validate_screening_input(poisoned, "inclusion", "Age 18 or older")

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_screening_input([], "inclusion", "Age 18 or older")
        with self.assertRaises(ValueError):
            validate_screening_input(SENTENCES, "bad", "Age 18 or older")
        with self.assertRaises(ValueError):
            validate_screening_input(
                [{"id": 0, "text": "a"}, {"id": 0, "text": "b"}],
                "inclusion",
                "Age 18 or older",
            )


class AggregationTests(unittest.TestCase):
    def result(self, kind, label, **flags):
        base = {
            "request_id": f"{kind}-{label}",
            "criterion_type": kind,
            "criterion_text": "text",
            "label": label,
            "evidence_status": "valid",
            "flags": {
                "parse_failure": False,
                "invalid_evidence_ids": False,
                "duplicate_evidence_ids": False,
                "truncation": False,
                "runtime_failure": False,
                "missing_information": False,
                "contradiction": False,
            },
        }
        base["flags"].update(flags)
        return base

    def test_no_criteria(self):
        report = aggregate_results([])
        self.assertEqual(report["outcome"], "no_assessment")
        self.assertFalse(report["is_ground_truth_trial_outcome"])

    def test_failure_precedes_unknown_and_flags(self):
        report = aggregate_results(
            [
                self.result("inclusion", "not included"),
                self.result("exclusion", "not enough information"),
                self.result("inclusion", "included", truncation=True),
            ]
        )
        self.assertEqual(report["outcome"], "preliminary_criterion_failure")
        self.assertTrue(any("criterion failure" in reason for reason in report["reasons"]))

    def test_unknown_or_invalid_requires_review(self):
        self.assertEqual(
            aggregate_results([self.result("inclusion", "not enough information")])["outcome"],
            "review_needed",
        )
        self.assertEqual(
            aggregate_results([self.result("exclusion", "not enough information")])[
                "outcome"
            ],
            "review_needed",
        )
        self.assertEqual(
            aggregate_results([self.result("inclusion", "included", parse_failure=True)])[
                "outcome"
            ],
            "review_needed",
        )

    def test_failure_keeps_per_criterion_flags_visible(self):
        report = aggregate_results(
            [
                self.result("inclusion", "not included"),
                self.result("inclusion", "included", truncation=True),
            ]
        )
        self.assertEqual(report["outcome"], "preliminary_criterion_failure")
        self.assertTrue(report["criteria"][1]["flags"]["truncation"])

    def test_valid_labels_can_match(self):
        report = aggregate_results(
            [
                self.result("inclusion", "included"),
                self.result("exclusion", "not excluded"),
            ]
        )
        self.assertEqual(report["outcome"], "preliminary_match")

    def test_success_label_without_evidence_requires_review(self):
        result = self.result("inclusion", "included")
        result["evidence_status"] = "no_ids"
        self.assertEqual(aggregate_results([result])["outcome"], "review_needed")

    def test_not_applicable_forces_review_not_a_match(self):
        report = aggregate_results(
            [
                self.result("inclusion", "included"),
                self.result("inclusion", "not applicable"),
                self.result("exclusion", "not excluded"),
                self.result("exclusion", "not applicable"),
            ]
        )
        self.assertEqual(report["outcome"], "review_needed")

    def test_aggregation_policy_loads_from_disk(self):
        policy = load_aggregation_policy(ROOT / "configs/aggregation.yaml")
        self.assertEqual(policy["version"], "v1")
        report = aggregate_results(
            [self.result("exclusion", "excluded")], policy=policy
        )
        self.assertEqual(report["outcome"], "preliminary_criterion_failure")


if __name__ == "__main__":
    unittest.main()
