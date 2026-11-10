"""UI-independent criterion screening with exact evidence traceability.

The model emits sentence IDs. This module resolves those IDs against the original
patient sentence map, keeping prediction, parse, evidence and operational states
separate. It never includes gold labels or gold evidence in a live result.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import yaml

from .decoding import cached_library_greedy
from .output_parser import parse_output
from .prompts import LABELS, build_prompt

EDUCATIONAL_NOTICE = "Educational screening prototype; not a clinical enrollment decision."
AGGREGATION_POLICY_PATH = Path("configs/aggregation.yaml")
REVIEW_FLAGS = (
    "parse_failure",
    "invalid_evidence_ids",
    "duplicate_evidence_ids",
    "truncation",
    "runtime_failure",
    "missing_information",
    "contradiction",
)
DEFAULT_LIMITS = {
    "max_sentences": 500,
    "max_sentence_characters": 10000,
    "max_criterion_characters": 4000,
    "max_total_characters": 200000,
}

GenerationFn = Callable[..., dict[str, Any]]

_NEGATION_CUES = {
    "no",
    "not",
    "never",
    "without",
    "denies",
    "denied",
    "none",
    "neither",
    "nor",
}
_CONTRADICTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "patient",
    "she",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
} | _NEGATION_CUES


def _content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if token not in _CONTRADICTION_STOPWORDS}


def _detect_possible_contradiction(sentences: list[dict[str, Any]]) -> bool:
    """Flag only clear lexical negation/affirmation pairs, never a clinical inference."""
    positive: list[set[str]] = []
    negative: list[set[str]] = []
    for sentence in sentences:
        text = sentence["text"]
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        content = _content_tokens(text)
        if tokens & _NEGATION_CUES:
            negative.append(content)
        else:
            positive.append(content)
    return any(len(left & right) >= 2 for left in negative for right in positive)


def _empty_flags() -> dict[str, bool]:
    return {name: False for name in REVIEW_FLAGS}


def validate_screening_input(
    sentences: list[dict[str, Any]],
    criterion_type: str,
    criterion_text: str,
    *,
    limits: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Validate and copy a patient note and one criterion without changing IDs."""
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    if criterion_type not in LABELS:
        raise ValueError("criterion_type must be 'inclusion' or 'exclusion'")
    if not isinstance(criterion_text, str) or not criterion_text.strip():
        raise ValueError("criterion_text is required")
    if len(criterion_text) > limits["max_criterion_characters"]:
        raise ValueError("criterion_text exceeds the configured character limit")
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("At least one numbered patient sentence is required")
    if len(sentences) > limits["max_sentences"]:
        raise ValueError("Patient note exceeds the configured sentence limit")
    total_characters = len(criterion_text)
    seen: set[int] = set()
    clean: list[dict[str, Any]] = []
    for sentence in sentences:
        if not isinstance(sentence, dict) or set(sentence) != {"id", "text"}:
            raise ValueError("Each sentence must contain exactly id and text")
        identifier = sentence["id"]
        text = sentence["text"]
        if type(identifier) is not int or identifier < 0:
            raise ValueError("Sentence IDs must be nonnegative integers")
        if identifier in seen:
            raise ValueError("Sentence IDs must be unique")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Sentence text must be nonempty")
        if len(text) > limits["max_sentence_characters"]:
            raise ValueError("A patient sentence exceeds the configured character limit")
        total_characters += len(text)
        seen.add(identifier)
        clean.append({"id": identifier, "text": text})
    if total_characters > limits["max_total_characters"]:
        raise ValueError("Patient note and criterion exceed the configured character limit")
    return clean


def _has_duplicate_ids(values: list[Any]) -> bool:
    counts = Counter(value for value in values if type(value) is int)
    return any(count > 1 for count in counts.values())


def _resolve_excerpts(
    sentence_map: dict[int, str], evidence_ids: list[int]
) -> list[dict[str, Any]]:
    return [{"id": identifier, "text": sentence_map[identifier]} for identifier in evidence_ids]


def _runtime_failure_parse_result() -> dict[str, Any]:
    return {
        "label_for_scoring": "__invalid__",
        "evidence_ids_validated": [],
        "invalid_evidence_values": [],
        "evidence_values_raw": [],
        "parse_status": "runtime_error",
        "evidence_parse_failed": True,
    }


def _explanation(
    *,
    label: str | None,
    excerpts: list[dict[str, Any]],
    parse_status: str,
    runtime_failure: bool,
    missing_information: bool,
    evidence_status: str,
    duplicate_ids: bool,
    contradiction: bool,
) -> str:
    if runtime_failure:
        return (
            "The model did not return a usable answer because of a runtime failure. "
            "No criterion decision or source trace is available."
        )
    if parse_status in {
        "json_error",
        "shape_error",
        "illegal_category",
        "invalid_evidence_list",
    }:
        return (
            "The model output could not be interpreted as an allowed structured answer. "
            "No verified evidence can be shown, and this result requires review."
        )
    if label is None:
        return "The model did not return an allowed label, so no verified decision can be shown."
    parts = [f"The model predicted '{label}'."]
    if excerpts:
        quoted = " ".join(f"[{item['id']}] {item['text']}" for item in excerpts)
        parts.append(f"It cites these exact source sentences: {quoted}")
    else:
        parts.append(
            "No source sentence was cited, so the rationale cannot be verified from citations."
        )
    if missing_information:
        parts.append("The provided record was treated as insufficient to establish this criterion.")
    if label == "not applicable":
        parts.append(
            "The model chose not applicable; this label alone does not establish a clinical reason."
        )
    if evidence_status == "contains_invalid":
        parts.append(
            "At least one cited ID does not exist in the sentences supplied to the model."
        )
    if duplicate_ids:
        parts.append("At least one cited sentence ID was repeated.")
    if contradiction:
        parts.append(
            "A possible lexical contradiction was detected between source sentences; "
            "review is required."
        )
    parts.append("A cited sentence is traceability, not proof of clinical entailment.")
    return " ".join(parts)


class ScreeningService:
    """Screen one patient criterion at a time with an already-loaded model."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        cfg: dict[str, Any],
        device: Any,
        *,
        generate_fn: GenerationFn | None = None,
        limits: dict[str, int] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.cfg = dict(cfg)
        self.device = device
        self.generate_fn = generate_fn or cached_library_greedy
        self.limits = {**DEFAULT_LIMITS, **(limits or {})}

    def screen(
        self,
        sentences: list[dict[str, Any]],
        criterion_type: str,
        criterion_text: str,
        *,
        request_id: str | None = None,
        generation_mode: str | None = None,
    ) -> dict[str, Any]:
        clean_sentences = validate_screening_input(
            sentences, criterion_type, criterion_text, limits=self.limits
        )
        sentence_map = {sentence["id"]: sentence["text"] for sentence in clean_sentences}
        model_input = {
            "criterion_type": criterion_type,
            "criterion_text": criterion_text.strip(),
            "sentences": clean_sentences,
        }
        started = time.perf_counter()
        raw_output = ""
        runtime_error: dict[str, str] | None = None
        generated: dict[str, Any] = {"token_ids": []}
        prompt: dict[str, Any] | None = None
        try:
            prompt = build_prompt(
                self.tokenizer,
                model_input,
                max_total_tokens=int(self.cfg["max_total_tokens"]),
                max_new_tokens=int(self.cfg["max_new_tokens"]),
            )
            generated = self.generate_fn(
                self.model,
                prompt["input_ids"],
                max_new_tokens=int(self.cfg["max_new_tokens"]),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                device=self.device,
            )
            raw_output = self.tokenizer.decode(
                generated.get("token_ids", []), skip_special_tokens=True
            )
            parsed = parse_output(
                raw_output, criterion_type, set(prompt["kept_sentence_ids"])
            )
        except Exception as exc:  # noqa: BLE001 - operational failures must stay visible
            runtime_error = {
                "type": type(exc).__name__,
                "message": str(exc)[:1000],
            }
            parsed = _runtime_failure_parse_result()
        label = parsed["label_for_scoring"]
        if label == "__invalid__":
            label = None
        evidence_ids = list(parsed["evidence_ids_validated"])
        invalid_values = list(parsed["invalid_evidence_values"])
        raw_values = list(parsed["evidence_values_raw"])
        excerpts = _resolve_excerpts(sentence_map, evidence_ids)
        duplicate_ids = _has_duplicate_ids(raw_values)
        evidence_status = _evidence_status(
            parsed, evidence_ids, invalid_values, duplicate_ids
        )
        kept_sentence_ids = (
            list(prompt["kept_sentence_ids"]) if prompt is not None else []
        )
        removed_sentence_ids = (
            list(prompt["removed_sentence_ids"])
            if prompt is not None
            else [sentence["id"] for sentence in clean_sentences]
        )
        input_truncated = True if prompt is None else bool(prompt["truncated"])
        output_truncated = bool(generated.get("hit_token_cap", False))
        flags = _empty_flags()
        flags["parse_failure"] = (
            runtime_error is None and parsed["parse_status"] != "valid"
        )
        flags["invalid_evidence_ids"] = bool(invalid_values)
        flags["duplicate_evidence_ids"] = duplicate_ids
        flags["truncation"] = input_truncated or output_truncated
        flags["runtime_failure"] = runtime_error is not None
        flags["missing_information"] = label == "not enough information"
        flags["contradiction"] = _detect_possible_contradiction(clean_sentences)
        explanation = _explanation(
            label=label,
            excerpts=excerpts,
            parse_status=parsed["parse_status"],
            runtime_failure=flags["runtime_failure"],
            missing_information=flags["missing_information"],
            evidence_status=evidence_status,
            duplicate_ids=duplicate_ids,
            contradiction=flags["contradiction"],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        prompt_ids = prompt["input_ids"] if prompt is not None else []
        prompt_hash = hashlib.sha256(
            json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        adapter_path = self.cfg.get("adapter_path")
        adapter_sha256 = self.cfg.get("adapter_sha256")
        return {
            "request_id": request_id or str(uuid.uuid4()),
            "criterion_type": criterion_type,
            "criterion_text": criterion_text.strip(),
            "raw_output": raw_output,
            "label": label,
            "evidence_sentence_ids": evidence_ids,
            "raw_evidence_values": raw_values,
            "invalid_evidence_values": invalid_values,
            "parse_status": parsed["parse_status"],
            "evidence_status": evidence_status,
            "evidence_excerpts": excerpts,
            "explanation": explanation,
            "flags": flags,
            "truncated": input_truncated or output_truncated,
            "input_truncated": input_truncated,
            "output_truncated": output_truncated,
            "removed_sentence_ids": removed_sentence_ids,
            "kept_sentence_ids": kept_sentence_ids,
            "omitted_sentence_count": len(removed_sentence_ids),
            "model_id": str(self.cfg.get("model_id", "")),
            "model_revision": str(self.cfg.get("model_revision", "")),
            "adapter_path": str(adapter_path) if adapter_path is not None else None,
            "adapter_sha256": (
                str(adapter_sha256) if adapter_sha256 is not None else None
            ),
            "prompt_revision": str(self.cfg.get("prompt_revision", "")),
            "prompt_sha256": prompt_hash,
            "generation_mode": generation_mode
            or str(self.cfg.get("generation_backend", "cached_library_greedy")),
            "runtime_error": runtime_error,
            "timing_ms": elapsed_ms,
            "educational_notice": EDUCATIONAL_NOTICE,
        }


def _evidence_status(
    parsed: dict[str, Any],
    evidence_ids: list[int],
    invalid_values: list[Any],
    duplicate_ids: bool,
) -> str:
    if parsed["parse_status"] == "runtime_error":
        return "runtime_error"
    if parsed["evidence_parse_failed"]:
        return "parse_failed"
    if invalid_values:
        return "contains_invalid"
    if duplicate_ids:
        return "duplicate_ids"
    if not evidence_ids:
        return "no_ids"
    return "valid"


def load_aggregation_policy(path: Path | str = AGGREGATION_POLICY_PATH) -> dict[str, Any]:
    policy_path = Path(path)
    if not policy_path.is_absolute():
        policy_path = Path(__file__).resolve().parents[2] / policy_path
    return yaml.safe_load(policy_path.read_text(encoding="utf-8"))


def _criterion_effect(result: dict[str, Any], policy: dict[str, Any]) -> str:
    label = result.get("label")
    criterion_type = result.get("criterion_type")
    effects = policy.get("criterion_effects", {})
    if (
        criterion_type == "inclusion"
        and label == effects.get("inclusion_failure")
    ):
        return "criterion_failure"
    if (
        criterion_type == "exclusion"
        and label == effects.get("exclusion_failure")
    ):
        return "criterion_failure"
    flags = result.get("flags") or {}
    review_flags = policy.get("review_flags", REVIEW_FLAGS)
    if any(bool(flags.get(name)) for name in review_flags):
        return "review"
    evidence_status = result.get("evidence_status", "valid")
    if evidence_status in set(policy.get("evidence_statuses_that_force_review", [])):
        return "review"
    if label is None or label in set(effects.get("review_labels", [])):
        return "review"
    if criterion_type == "inclusion" and label == effects.get("inclusion_success"):
        return "satisfied"
    if criterion_type == "exclusion" and label == effects.get("exclusion_success"):
        return "satisfied"
    return "review"


def aggregate_results(
    results: list[dict[str, Any]],
    *,
    policy: dict[str, Any] | None = None,
    policy_path: Path | str = AGGREGATION_POLICY_PATH,
) -> dict[str, Any]:
    """Apply the transparent preliminary aggregation policy to criterion results."""
    if policy is None:
        policy = load_aggregation_policy(policy_path)
    criteria = []
    reasons: list[str] = []
    for result in results:
        effect = _criterion_effect(result, policy)
        criteria.append(
            {
                "request_id": result.get("request_id"),
                "criterion_type": result.get("criterion_type"),
                "criterion_text": result.get("criterion_text"),
                "label": result.get("label"),
                "flags": result.get("flags") or {},
                "effect": effect,
            }
        )
        if effect == "criterion_failure":
            reasons.append("A definite criterion failure was predicted.")
        elif effect == "review":
            reasons.append("At least one criterion requires review or has no verified decision.")
    effects = {item["effect"] for item in criteria}
    if not results:
        outcome = "no_assessment"
        reasons = ["No criteria were supplied."]
    elif "criterion_failure" in effects:
        outcome = "preliminary_criterion_failure"
    elif "review" in effects:
        outcome = "review_needed"
    elif effects == {"satisfied"}:
        outcome = "preliminary_match"
    else:
        outcome = "review_needed"
        reasons.append("An unrecognized or inconsistent state was encountered.")
    return {
        "outcome": outcome,
        "outcome_description": policy.get("outcomes", {}).get(outcome, ""),
        "policy_version": policy.get("version"),
        "criterion_count": len(criteria),
        "criteria": criteria,
        "reasons": sorted(set(reasons)),
        "not_applicable_policy": policy.get("not_applicable_policy"),
        "educational_notice": EDUCATIONAL_NOTICE,
        "is_ground_truth_trial_outcome": False,
    }
