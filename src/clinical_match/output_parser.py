"""Strictly parse raw model JSON while preserving every failure for scoring."""
from __future__ import annotations

import json
from typing import Any

from .prompts import LABELS


def parse_output(raw_output: str, criterion_type: str, valid_sentence_ids: set[int]) -> dict[str, Any]:
    """Return raw and scoreable fields; never turn an invalid answer into abstention."""
    result: dict[str, Any] = {
        "raw_output": raw_output,
        "parsed_json": None,
        "label_raw": None,
        "label_for_scoring": "__invalid__",
        "evidence_values_raw": [],
        "evidence_ids_for_scoring": [],
        "evidence_ids_validated": [],
        "invalid_evidence_values": [],
        "parse_status": "json_error",
        "evidence_parse_failed": True,
    }
    try:
        value = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return result
    result["parsed_json"] = value
    if not isinstance(value, dict):
        result["parse_status"] = "shape_error"
        return result
    if set(value) != {"label", "evidence_sentence_ids"}:
        result["parse_status"] = "shape_error"
        return result
    label = value.get("label")
    result["label_raw"] = label
    allowed = LABELS.get(criterion_type, [])
    if isinstance(label, str) and label in allowed:
        result["label_for_scoring"] = label
        result["parse_status"] = "valid"
    else:
        result["parse_status"] = "illegal_category"
    evidence = value.get("evidence_sentence_ids")
    if not isinstance(evidence, list):
        if result["parse_status"] == "valid":
            result["parse_status"] = "invalid_evidence_list"
        return result
    result["evidence_parse_failed"] = False
    result["evidence_values_raw"] = evidence
    score_ids: set[Any] = set()
    valid_ids: set[int] = set()
    invalid: list[Any] = []
    for index, item in enumerate(evidence):
        if type(item) is int and item >= 0:
            score_ids.add(item)
            if item in valid_sentence_ids:
                valid_ids.add(item)
            else:
                invalid.append(item)
        else:
            # A stable, annotation-scoped sentinel scores each malformed value as FP.
            sentinel = f"__invalid_evidence_{index}:{type(item).__name__}:{repr(item)}"
            score_ids.add(sentinel)
            invalid.append(item)
    result["evidence_ids_for_scoring"] = sorted(score_ids, key=lambda x: (type(x).__name__, repr(x)))
    result["evidence_ids_validated"] = sorted(valid_ids)
    result["invalid_evidence_values"] = invalid
    return result
