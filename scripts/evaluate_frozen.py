"""One-time paired evaluation against the protected test split after freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.clinical_match.data_access import load_examples
from src.clinical_match.evaluation import (
    generate_predictions, hash_directory, load_eval_model, prepare_prompts,
)
from src.clinical_match.metrics import score_predictions

FROZEN_FILES = [
    "configs/eval.yaml", "configs/model.yaml", "configs/prompt.yaml",
    "configs/aggregation.yaml", "src/clinical_match/data_access.py",
    "src/clinical_match/evaluation.py", "src/clinical_match/metrics.py",
    "src/clinical_match/output_parser.py", "src/clinical_match/prompts.py",
    "src/clinical_match/screening.py", "src/clinical_match/decoding/policy.py",
    "src/clinical_match/decoding/reference.py", "src/clinical_match/decoding/cached.py",
    "src/clinical_match/decoding/speculative.py", "scripts/evaluate.py",
    "app/main.py", "scripts/evaluate_frozen.py", "scripts/freeze_experiment.py",
    "scripts/smoke_local_app.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-final-test", action="store_true", required=True,
                        help="Explicitly authorize the final held-out test evaluation")
    parser.add_argument("--variant", choices=["base", "adapted"], required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/frozen_manifest.json")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    contents = manifest_path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if digest != args.manifest_sha256:
        raise ValueError("Frozen manifest checksum does not match --manifest-sha256")
    frozen = json.loads(contents)
    if frozen.get("state") != "frozen" or frozen.get("schema_version") != "1.0":
        raise ValueError("A supported frozen manifest is required")

    for relative in FROZEN_FILES:
        if sha256(ROOT / relative) != frozen.get("files", {}).get(relative):
            raise ValueError(f"Frozen source/configuration changed: {relative}")
    eval_path = ROOT / "configs/eval.yaml"
    model_path = ROOT / "configs/model.yaml"
    prompt_path = ROOT / "configs/prompt.yaml"
    aggregation_path = ROOT / "configs/aggregation.yaml"
    cfg = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
    if cfg != frozen["resolved"]["eval"]:
        raise ValueError("Resolved evaluation fields differ from the frozen manifest")
    for path, key in ((model_path, "model"), (prompt_path, "prompt"), (aggregation_path, "aggregation")):
        current = yaml.safe_load(path.read_text(encoding="utf-8"))
        if current != frozen["resolved"][key]:
            raise ValueError(f"Resolved {key} fields differ from the frozen manifest")
    if frozen["resolved"]["backend"] != "transformers_generate_greedy_cache":
        raise ValueError("Unsupported frozen generation backend")
    variant = frozen["variants"][args.variant]
    adapter_path = None
    if args.variant == "adapted":
        adapter_path = ROOT / variant["adapter_path"]
        if hash_directory(adapter_path) != variant["adapter_sha256"]:
            raise ValueError("Selected adapter does not match the frozen checksum")
    elif variant.get("adapter_sha256") is not None or variant.get("adapter_path") is not None:
        raise ValueError("Base evaluation must not load an adapter")

    split_path = ROOT / "data/splits/manifest.json"
    test_path = ROOT / "data/processed/protected_test/test.jsonl"
    if sha256(split_path) != frozen["split_manifest_sha256"]:
        raise ValueError("Split manifest changed after freeze")
    rows = load_examples(ROOT, "test", allow_final_test=True,
                         frozen_manifest=manifest_path, frozen_sha256=digest,
                         variant=args.variant)
    expected = set(frozen["expected_test_ids"])
    observed = [row["provenance"]["annotation_id"] for row in rows]
    if len(rows) != frozen["expected_test_count"] or len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("Loaded test records do not exactly cover the frozen IDs")

    out_dir = args.output_dir or ROOT / "results/final_test" / args.variant
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing final evaluation artifacts: {out_dir}")
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Frozen runtime requires CUDA, which is unavailable")
    tokenizer, model = load_eval_model(cfg, device, adapter_path=adapter_path)
    prompts = prepare_prompts(tokenizer, rows, cfg)
    predictions, elapsed = generate_predictions(model, tokenizer, rows, cfg, device, prompts=prompts)
    pred_ids = [prediction["annotation_id"] for prediction in predictions]
    if len(predictions) != len(rows) or len(pred_ids) != len(set(pred_ids)) or set(pred_ids) != expected:
        raise RuntimeError("Final prediction coverage is partial, duplicated or has unexpected IDs")
    metrics = score_predictions(predictions)
    metrics["scope"] = {"split": "test", "n": len(predictions), "expected_n": len(expected),
                        "full_split": True, "test_data_opened": True}
    metrics["final_evaluation"] = {"variant": args.variant, "manifest_sha256": digest,
                                   "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                                   "generation_seconds": elapsed,
                                   "test_jsonl_sha256": sha256(test_path),
                                   "adapter_sha256": variant.get("adapter_sha256")}
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "test_predictions.jsonl"
    pred_path.write_text("".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in predictions), encoding="utf-8")
    metrics["final_evaluation"]["predictions_sha256"] = sha256(pred_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps({"variant": args.variant, "resolved_eval": cfg,
        "model": variant, "manifest_sha256": digest}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"variant": args.variant, "n": len(predictions), "macro_f1": metrics["classification"]["macro_f1"],
                      "evidence_micro_f1": metrics["evidence"]["micro"]["f1"], "output_dir": str(out_dir)}, indent=2))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
