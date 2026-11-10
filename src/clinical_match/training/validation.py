"""Validation-only generation and scoring for checkpoint selection."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..data_access import load_examples
from ..evaluation import generate_predictions, hash_directory, prepare_prompts, sha256_file
from ..metrics import score_predictions


def predict_validation(
    model: Any,
    tokenizer: Any,
    cfg: dict[str, Any],
    root: Path,
    output_dir: Path,
    device: torch.device,
    *,
    adapter_path: Path | None = None,
) -> dict[str, Any]:
    """Generate and score all validation rows with the frozen prompt and greedy path."""
    rows = load_examples(root, "validation")
    manifest = json.loads((root / cfg["split_manifest"]).read_text(encoding="utf-8"))
    expected = manifest["annotation_ids"]["validation"]
    observed = [row["provenance"]["annotation_id"] for row in rows]
    if set(observed) != set(expected) or len(observed) != len(set(observed)):
        raise ValueError("Validation records do not match the canonical split manifest")

    prompts = prepare_prompts(tokenizer, rows, cfg)
    predictions, total_seconds = generate_predictions(
        model, tokenizer, rows, cfg, device, prompts=prompts
    )
    metrics = score_predictions(predictions)
    metrics["scope"] = {
        "split": "validation",
        "n": len(predictions),
        "requested_n": len(rows),
        "full_split": True,
        "test_data_opened": False,
    }
    metrics["generation"] = {
        "backend": cfg["generation_backend"],
        "total_seconds": total_seconds,
        "seconds_per_annotation": total_seconds / len(predictions) if predictions else None,
        "tokens_per_second": (
            sum(len(row["generated_token_ids"]) for row in predictions) / total_seconds
            if total_seconds
            else None
        ),
    }
    metrics["artifacts"] = {
        "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "model_config_sha256": sha256_file(root / cfg["model_config"]),
        "prompt_config_sha256": sha256_file(root / cfg["prompt_config"]),
        "split_manifest_sha256": sha256_file(root / cfg["split_manifest"]),
        "validation_jsonl_sha256": sha256_file(
            root / "data/processed/development/validation.jsonl"
        ),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_sha256": hash_directory(adapter_path) if adapter_path else None,
    }
    metrics["environment"] = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "validation_predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics
