from scripts.verify_metrics_fixture import prediction
from src.clinical_match.metrics import score_predictions


def test_hand_calculated_classification_and_evidence_fixture():
    rows = [
        prediction(1, "inclusion", "included", [0, 1],
                   '{"label":"included","evidence_sentence_ids":[0,9]}'),
        prediction(2, "inclusion", "not included", [],
                   '{"label":"impossible","evidence_sentence_ids":[]}'),
        prediction(3, "exclusion", "excluded", [],
                   '{"label":"excluded","evidence_sentence_ids":[]}'),
        prediction(4, "exclusion", "not excluded", [3],
                   '{"label":"not excluded","evidence_sentence_ids":[]}'),
        prediction(5, "inclusion", "not enough information", [],
                   '{"label":"not enough information","evidence_sentence_ids":["bad",8]}'),
        prediction(6, "exclusion", "not applicable", [],
                   '{"label":"not applicable","evidence_sentence_ids":"bad"}'),
    ]

    scored = score_predictions(rows)

    assert scored["evidence"]["micro_counts"] == {"tp": 1, "fp": 3, "fn": 2}
    assert scored["evidence"]["micro"] == pytest.approx({
        "precision": 0.25, "recall": 1 / 3, "f1": 2 / 7,
    })
    assert scored["evidence"]["both_empty_annotations"] == 3
    assert scored["classification"]["invalid_prediction_count"] == 1
    assert scored["classification"]["zero_support_classes"] == []
    assert scored["evidence_parse_failure_count"] == 1
    assert scored["invalid_evidence_value_count"] == 3


def test_malformed_json_is_invalid_label_and_empty_retrieval():
    row = prediction(1, "inclusion", "included", [4], "not json")

    scored = score_predictions([row])

    assert scored["classification"]["invalid_prediction_count"] == 1
    assert scored["classification"]["accuracy"] == 0
    assert scored["evidence"]["micro_counts"] == {"tp": 0, "fp": 0, "fn": 1}
    assert scored["evidence_parse_failure_count"] == 1
import pytest

