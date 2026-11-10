"""Evaluate a frozen base model or adapter on train/validation only; test is never accepted."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.clinical_match.decoding import cached_library_greedy, uncached_greedy
from src.clinical_match.evaluation import (
    generate_predictions,
    hash_directory,
    load_eval_model,
    prepare_prompts,
    sha256_file,
)
from src.clinical_match.metrics import score_predictions
from src.clinical_match.output_parser import parse_output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def evaluate(
    config_path: Path,
    *,
    split_override: str | None = None,
    limit: int | None = None,
    output_dir: Path | None = None,
    adapter_override: Path | None = None,
) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if split_override is not None:
        cfg["split"] = split_override
    split = cfg["split"]
    if split not in {"train", "validation"}:
        raise ValueError("Development evaluator refuses protected test access")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    examples_path = ROOT / "data/processed/development" / f"{split}.jsonl"
    manifest_path = ROOT / "data/splits/manifest.json"
    examples = read_jsonl(examples_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = manifest["annotation_ids"][split]
    observed_ids = [row["provenance"]["annotation_id"] for row in examples]
    if set(observed_ids) != set(expected_ids) or len(observed_ids) != len(set(observed_ids)):
        raise ValueError("Development records do not match the canonical split manifest")
    selected = examples[:limit] if limit is not None else examples

    adapter_path = adapter_override or cfg.get("adapter_path")
    if adapter_path:
        adapter_path = Path(adapter_path)
        if not adapter_path.is_absolute():
            adapter_path = ROOT / adapter_path
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")
    else:
        adapter_path = None

    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    tokenizer, model = load_eval_model(cfg, device, adapter_path=adapter_path)

    prepared = prepare_prompts(tokenizer, selected, cfg)

    # Validate the accelerated library path against the explicit uncached reference.
    n_equiv = min(int(cfg["reference_equivalence_examples"]), len(selected))
    equivalence: list[dict[str, Any]] = []
    for row, prompt in list(zip(selected, prepared))[:n_equiv]:
        ref = uncached_greedy(model, prompt["input_ids"], max_new_tokens=cfg["max_new_tokens"],
                              eos_token_id=tokenizer.eos_token_id, device=device,
                              pad_token_id=tokenizer.pad_token_id)
        fast = cached_library_greedy(model, prompt["input_ids"], max_new_tokens=cfg["max_new_tokens"],
                                     eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.pad_token_id, device=device)
        exact_match = ref["token_ids"] == fast["token_ids"]
        first_difference = next((i for i, (a, b) in enumerate(zip(ref["token_ids"], fast["token_ids"])) if a != b), None)
        ref_text = tokenizer.decode(ref["token_ids"], skip_special_tokens=True)
        fast_text = tokenizer.decode(fast["token_ids"], skip_special_tokens=True)
        sentence_ids = {sentence["id"] for sentence in row["input"]["sentences"]}
        ref_parsed = parse_output(ref_text, row["input"]["criterion_type"], sentence_ids)
        fast_parsed = parse_output(fast_text, row["input"]["criterion_type"], sentence_ids)
        parsed_match = all(ref_parsed[key] == fast_parsed[key] for key in
                           ("label_for_scoring", "evidence_ids_for_scoring", "parse_status", "evidence_parse_failed"))
        equivalence.append({"annotation_id": row["provenance"]["annotation_id"],
                            "exact_token_match": exact_match,
                            "parsed_output_match": parsed_match,
                            "reference_token_count": len(ref["token_ids"]),
                            "cached_token_count": len(fast["token_ids"]),
                            "first_difference": first_difference,
                            "reference_tokens": ref["token_ids"], "cached_tokens": fast["token_ids"]})
        destination = output_dir or ROOT / "results/baseline" / f"{cfg['baseline_name']}_{split}"
        if not exact_match:
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "equivalence_diagnostic.json").write_text(
                json.dumps(equivalence, indent=2) + "\n", encoding="utf-8")
        if not parsed_match:
            raise RuntimeError(f"Cached library greedy changes the parsed prediction; diagnostic saved to {destination / 'equivalence_diagnostic.json'}")

    predictions, total_generation_seconds = generate_predictions(
        model, tokenizer, selected, cfg, device, prompts=prepared
    )

    metrics = score_predictions(predictions)
    metrics["scope"] = {"split": split, "n": len(predictions), "requested_n": len(selected),
                        "full_split": limit is None, "test_data_opened": False}
    metrics["generation"] = {"backend": cfg["generation_backend"],
                             "total_seconds": total_generation_seconds,
                             "seconds_per_annotation": total_generation_seconds / len(predictions) if predictions else None,
                             "tokens_per_second": sum(len(row["generated_token_ids"]) for row in predictions) / total_generation_seconds if total_generation_seconds else None,
                             "equivalence_examples": equivalence}
    metrics["artifacts"] = {
        "model_id": cfg["model_id"], "model_revision": cfg["model_revision"],
        "model_config_sha256": sha256_file(ROOT / cfg["model_config"]),
        "prompt_config_sha256": sha256_file(ROOT / cfg["prompt_config"]),
        "eval_config_sha256": sha256_file(config_path),
        "split_manifest_sha256": sha256_file(manifest_path),
        "input_jsonl_sha256": sha256_file(examples_path),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_sha256": hash_directory(adapter_path) if adapter_path else None,
    }
    metrics["environment"] = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                               "platform": platform.platform(), "python": sys.version,
                               "torch": torch.__version__, "cuda": torch.version.cuda,
                               "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    destination = output_dir or ROOT / "results/baseline" / f"{cfg['baseline_name']}_{split}"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / f"{split}_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
    (destination / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    resolved = dict(cfg)
    resolved["adapter_path"] = str(adapter_path) if adapter_path else None
    (destination / "run_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(destination), "scope": metrics["scope"], "macro_f1": metrics["classification"]["macro_f1"], "evidence_micro_f1": metrics["evidence"]["micro"]["f1"]}, indent=2))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval.yaml")
    parser.add_argument("--split", choices=["train", "validation"], default=None)
    parser.add_argument("--limit", type=int, default=None, help="Development-only pilot size; omitting evaluates the full split")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--adapter", type=Path, default=None, help="Local PEFT adapter directory or null for the base model")
    args = parser.parse_args()
    if args.split:
        evaluate(args.config, split_override=args.split, limit=args.limit,
                 output_dir=args.output_dir, adapter_override=args.adapter)
    else:
        evaluate(args.config, limit=args.limit, output_dir=args.output_dir,
                 adapter_override=args.adapter)


if __name__ == "__main__":
    main()
