"""Integration checks on development examples and the public split manifest."""
import ast
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from src.clinical_match.data_access import load_examples

ROOT = Path(__file__).resolve().parents[1]


def test_development_examples_match_schema_and_evidence_roundtrips():
    schema = json.loads((ROOT / "schemas/prepared_example.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    manifest = json.loads((ROOT / "data/splits/manifest.json").read_text(encoding="utf-8"))
    for split, expected in (("train", 602), ("validation", 206)):
        rows = load_examples(ROOT, split, allow_quarantined_training=(split == "train"))
        assert len(rows) == expected
        assert {r["provenance"]["annotation_id"] for r in rows} <= set(manifest["annotation_ids"][split])
        for row in rows:
            validator.validate(row)
            ids = [s["id"] for s in row["input"]["sentences"]]
            assert len(ids) == len(set(ids))
            assert set(row["target"]["evidence_sentence_ids"]) <= set(ids)
            assert row["target"]["evidence_sentence_ids"] == sorted(set(ast.literal_eval(
                row["provenance"]["source_expert_sentences"])))
            assert not ({"gpt4_explanation", "gpt4_sentences", "gpt4_eligibility",
                         "explanation_correctness", "training"} & set(row["input"]))
            assert set(row["input"]) == {"criterion_type", "criterion_text", "sentences"}
        first = rows[0]
        wrong_label = "excluded" if first["input"]["criterion_type"] == "inclusion" else "included"
        invalid = {**first, "target": {**first["target"], "label": wrong_label}}
        assert not validator.is_valid(invalid)


def test_manifest_hash_partition_and_source_accounting():
    path = ROOT / "data/splits/manifest.json"
    expected_hash = (ROOT / "data/splits/manifest.sha256").read_text().split()[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    manifest = json.loads(path.read_text(encoding="utf-8"))
    groups = [set(manifest["patient_ids"][s]) for s in ("train", "validation", "test")]
    assert [len(g) for g in groups] == [37, 8, 8]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert sum(manifest["row_counts"][s]["source"] for s in manifest["row_counts"]) == 1015
    assert manifest["row_counts"]["train"]["quarantined"] == 1
    assert manifest["row_counts"]["test"]["prepared"] == 206
    with pytest.raises(PermissionError):
        load_examples(ROOT, "test")
    assert manifest["excluded_training_annotation_ids"] == [883]
    assert len(load_examples(ROOT, "train")) == 602


def test_inference_schema_rejects_gold_target():
    schema = json.loads((ROOT / "schemas/inference_result.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    result = {"criterion_type": "inclusion", "criterion_text": "Age", "raw_output": "{}",
              "label": None, "evidence_sentence_ids": [], "parse_status": "invalid",
              "evidence_status": "unavailable", "evidence_excerpts": [], "explanation": "",
              "truncated": False, "model_revision": "rev", "prompt_revision": "v1", "timing_ms": 1}
    validator.validate(result)
    assert not validator.is_valid({**result, "target": {"label": "included"}})
