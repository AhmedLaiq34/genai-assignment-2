"""Recompute the saved validation baseline metrics and verify annotation coverage."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.clinical_match.metrics import score_predictions


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    manifest = json.loads((ROOT / "data/splits/manifest.json").read_text(encoding="utf-8"))
    expected_ids = manifest["annotation_ids"]["validation"]
    predictions_path = ROOT / "results/baseline/base_greedy_v1_validation/validation_predictions.jsonl"
    metrics_path = predictions_path.with_name("metrics.json")
    predictions = read_jsonl(predictions_path)
    observed_ids = [row["annotation_id"] for row in predictions]
    assert len(observed_ids) == len(set(observed_ids)), "duplicate validation annotation IDs"
    assert set(observed_ids) == set(expected_ids), "validation output coverage differs from manifest"
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    recomputed = score_predictions(predictions)
    metric_keys = set(recomputed)
    assert {key: saved_metrics[key] for key in metric_keys} == recomputed, "saved metrics do not reproduce"
    assert saved_metrics["scope"]["test_data_opened"] is False
    print(f"Verified {len(predictions)} validation outputs and reproduced saved metrics.")


if __name__ == "__main__":
    main()
