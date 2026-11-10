"""Hand-check the Phase 3 metric conventions without opening assignment test data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.clinical_match.metrics import score_predictions
from src.clinical_match.output_parser import parse_output


def prediction(annotation_id: int, kind: str, gold: str, evidence: list[int], raw: str) -> dict:
    parsed = parse_output(raw, kind, {0, 1, 3})
    return {"annotation_id": annotation_id, "criterion_type": kind, "gold_label": gold,
            "gold_evidence_ids": evidence, **parsed, "runtime_status": "ok", "truncated": False}


def main() -> None:
    rows = [
        prediction(1, "inclusion", "included", [0, 1], '{"label":"included","evidence_sentence_ids":[0,9]}'),
        prediction(2, "inclusion", "not included", [], '{"label":"impossible","evidence_sentence_ids":[]}'),
        prediction(3, "exclusion", "excluded", [], '{"label":"excluded","evidence_sentence_ids":[]}'),
        prediction(4, "exclusion", "not excluded", [3], '{"label":"not excluded","evidence_sentence_ids":[]}'),
        prediction(5, "inclusion", "not enough information", [], '{"label":"not enough information","evidence_sentence_ids":["bad",8]}'),
        prediction(6, "exclusion", "not applicable", [], '{"label":"not applicable","evidence_sentence_ids":"bad"}'),
    ]
    scored = score_predictions(rows)
    evidence = scored["evidence"]
    assert evidence["micro_counts"] == {"tp": 1, "fp": 3, "fn": 2}, evidence
    assert abs(evidence["micro"]["precision"] - 1 / 4) < 1e-12
    assert abs(evidence["micro"]["recall"] - 1 / 3) < 1e-12
    assert abs(evidence["micro"]["f1"] - 2 / 7) < 1e-12
    assert evidence["both_empty_annotations"] == 3
    assert scored["classification"]["invalid_prediction_count"] == 1
    assert scored["classification"]["zero_support_classes"] == []
    assert scored["evidence_parse_failure_count"] == 1
    assert scored["invalid_evidence_value_count"] == 3
    out = ROOT / "results/phase03_metric_fixture.json"
    out.write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Hand-checked metric fixture passed: {out}")


if __name__ == "__main__":
    main()
