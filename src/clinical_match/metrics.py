"""Fixed-class criterion and annotation-scoped evidence metrics."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .prompts import LABELS

ALL_CLASSES = ["included", "not included", "excluded", "not excluded", "not enough information", "not applicable"]
INVALID = "__invalid__"


def _classification(rows: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": None, "macro_f1": None, "per_class": {}, "confusion_matrix": []}
    truth = [row["gold_label"] for row in rows]
    pred = [row.get("label_for_scoring", INVALID) for row in rows]
    p, r, f, support = precision_recall_fscore_support(
        truth, pred, labels=classes, zero_division=0
    )
    cm = confusion_matrix(truth, pred, labels=classes + [INVALID])
    per_class = {
        name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
        for i, name in enumerate(classes)
    }
    return {
        "n": len(rows),
        "accuracy": sum(a == b for a, b in zip(truth, pred)) / len(rows),
        "macro_f1": float(sum(f) / len(classes)),
        "classes": classes,
        "per_class": per_class,
        "confusion_matrix_labels": classes + [INVALID],
        "confusion_matrix": cm.tolist(),
        "zero_support_classes": [name for name in classes if per_class[name]["support"] == 0],
        "invalid_prediction_count": sum(value == INVALID for value in pred),
    }


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    if tp + fp + fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if tp + fp == 0 or tp + fn == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = Counter(tp=0, fp=0, fn=0)
    records: list[dict[str, Any]] = []
    nonempty = Counter(tp=0, fp=0, fn=0)
    for row in rows:
        gold = set(row["gold_evidence_ids"])
        pred = set(row.get("evidence_ids_for_scoring", []))
        tp, fp, fn = len(gold & pred), len(pred - gold), len(gold - pred)
        total.update(tp=tp, fp=fp, fn=fn)
        if gold:
            nonempty.update(tp=tp, fp=fp, fn=fn)
        records.append(_prf(tp, fp, fn))
    macro = {key: (sum(item[key] for item in records) / len(records) if records else None)
             for key in ("precision", "recall", "f1")}
    return {
        "n": len(rows),
        "micro_counts": dict(total),
        "micro": _prf(total["tp"], total["fp"], total["fn"]),
        "macro_per_annotation": macro,
        "both_empty_annotations": sum(not row["gold_evidence_ids"] and not row.get("evidence_ids_for_scoring", []) for row in rows),
        "nonempty_gold_n": sum(bool(row["gold_evidence_ids"]) for row in rows),
        "nonempty_gold_micro_counts": dict(nonempty),
        "nonempty_gold_micro": _prf(nonempty["tp"], nonempty["fp"], nonempty["fn"]),
    }


def score_predictions(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    result: dict[str, Any] = {
        "n": len(materialized),
        "classification": _classification(materialized, ALL_CLASSES),
        "by_criterion_type": {},
        "evidence": _evidence(materialized),
        "parse_status_counts": dict(Counter(row.get("parse_status", "runtime_error") for row in materialized)),
        "runtime_failure_count": sum(row.get("runtime_status") == "error" for row in materialized),
        "truncated_count": sum(row.get("truncated", False) for row in materialized),
        "evidence_parse_failure_count": sum(row.get("evidence_parse_failed", True) for row in materialized),
        "invalid_evidence_value_count": sum(len(row.get("invalid_evidence_values", [])) for row in materialized),
        "raw_evidence_validity_rate": (
            sum(not row.get("invalid_evidence_values") and not row.get("evidence_parse_failed", True) for row in materialized)
            / len(materialized) if materialized else None
        ),
        "validated_evidence_validity_rate": (
            sum(not row.get("invalid_evidence_values") and not row.get("evidence_parse_failed", True)
                and set(row.get("evidence_ids_validated", [])) == set(row.get("evidence_ids_for_scoring", []))
                for row in materialized) / len(materialized) if materialized else None
        ),
    }
    for kind, classes in LABELS.items():
        subset = [row for row in materialized if row.get("criterion_type") == kind]
        result["by_criterion_type"][kind] = {
            "classification": _classification(subset, classes),
            "evidence": _evidence(subset),
        }
    return result
