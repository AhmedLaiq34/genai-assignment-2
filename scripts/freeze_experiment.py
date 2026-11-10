"""Freeze the validated experiment identity before protected test evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.clinical_match.evaluation import hash_directory

ADAPTER = Path("artifacts/phase04/phase04_lora_colab_cli_20261110T1650Z_bundle/selected_adapter")
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


def git_state() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"commit": commit, "working_tree_dirty_at_freeze": dirty}
    except Exception:
        return {"commit": None, "working_tree_dirty_at_freeze": None}


def verify_bundle(bundle: Path) -> dict:
    """Verify all exported bundle files before recording selected-run provenance."""
    manifest_path = bundle / "bundle_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    bundle_digest = hashlib.sha256(manifest_bytes).hexdigest()
    complete = json.loads((bundle / "COMPLETE.json").read_text(encoding="utf-8"))
    if complete.get("manifest_sha256") != bundle_digest:
        raise ValueError("Selected training bundle manifest checksum is invalid")
    bundle_manifest = json.loads(manifest_bytes)
    for relative, expected in bundle_manifest["files"].items():
        path = bundle / relative
        if not path.is_file() or path.stat().st_size != expected["size_bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Selected training bundle file failed integrity verification: {relative}")
    selected = json.loads((bundle / "selected_manifest.json").read_text(encoding="utf-8"))
    selected_dir = bundle / "selected_adapter"
    local_dir = ROOT / ADAPTER
    selected_files = {p.relative_to(selected_dir).as_posix(): p for p in selected_dir.rglob("*") if p.is_file()}
    local_files = {p.relative_to(local_dir).as_posix(): p for p in local_dir.rglob("*") if p.is_file()}
    if selected_files.keys() != local_files.keys() or any(sha256(selected_files[name]) != sha256(local_files[name]) for name in selected_files):
        raise ValueError("Local selected adapter directory differs from the integrity-verified bundle")
    if selected.get("run_name") != bundle_manifest.get("metadata", {}).get("run_name") or selected.get("selected", {}).get("epoch") != 3:
        raise ValueError("Selected training run or validation-selected epoch does not match the verified bundle")
    return {"bundle_manifest_sha256": bundle_digest,
            "selected_manifest_sha256": sha256(bundle / "selected_manifest.json"),
            "selected_adapter_model_sha256": sha256(bundle / "selected_adapter/adapter_model.safetensors"),
            "selected_run_name": selected.get("run_name"),
            "selected_epoch": selected.get("selected", {}).get("epoch"),
            "run_name": bundle_manifest.get("metadata", {}).get("run_name"),
            "global_step": bundle_manifest.get("metadata", {}).get("global_step"),
            "selection_source": bundle_manifest.get("metadata", {}).get("selection_source")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/frozen_manifest.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing freeze: {args.output}")
    adapter = ROOT / ADAPTER
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)

    # Require the full validation-selected adapter result, adapter parity and
    # a real local application action before opening the protected split.
    validation_dir = ROOT / "results/phase04/local_eval_fixed_full"
    validation_metrics = json.loads((validation_dir / "metrics.json").read_text(encoding="utf-8"))
    adapter_hash = hash_directory(adapter)
    bundle_identity = verify_bundle(ROOT / "artifacts/phase04/phase04_lora_colab_cli_20261110T1650Z_bundle")
    if bundle_identity["selected_adapter_model_sha256"] != sha256(adapter / "adapter_model.safetensors"):
        raise ValueError("Local adapter weights do not match the integrity-verified selected bundle")
    if validation_metrics["artifacts"].get("adapter_sha256") != adapter_hash or validation_metrics["scope"].get("n") != 206:
        raise ValueError("Full adapter validation evidence is missing or does not match the selected adapter")
    decode_summary = json.loads((ROOT / "results/cache/correctness_adapter_summary.json").read_text(encoding="utf-8"))
    if not decode_summary.get("all_completed_match") or decode_summary.get("exact_matches") != 6 \
            or decode_summary.get("provenance", {}).get("adapter_sha256") != adapter_hash:
        raise ValueError("All six adapter explicit-cache cases must match before freezing")
    spec = [json.loads(line) for line in (ROOT / "results/speculative/correctness_adapter.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_spec_cases = 3 * 3 * 4
    spec_summary = json.loads((ROOT / "results/speculative/correctness_adapter_summary.json").read_text(encoding="utf-8"))
    if (len(spec) != expected_spec_cases or not all(row.get("match") for row in spec)
            or spec_summary.get("cases") != expected_spec_cases
            or spec_summary.get("matches") != expected_spec_cases
            or not spec_summary.get("all_match")
            or spec_summary.get("adapter_sha256") != adapter_hash):
        raise ValueError("All 36 adapter speculative cases must match before freezing")
    app_smoke = ROOT / "results/dev_checks/app_smoke_summary.json"
    if not app_smoke.is_file():
        raise ValueError("A successful adapter-backed app smoke record is required before freezing")
    app_result = json.loads(app_smoke.read_text(encoding="utf-8"))
    if app_result.get("adapter_sha256") != adapter_hash or not app_result.get("passed"):
        raise ValueError("Latest app smoke artifact did not use the selected adapter")

    split_manifest_path = ROOT / "data/splits/manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    expected_ids = sorted(split_manifest["annotation_ids"]["test"])
    if len(expected_ids) != 206 or len(set(expected_ids)) != 206:
        raise ValueError("Canonical manifest must declare 206 unique protected test IDs")
    eval_cfg = yaml.safe_load((ROOT / "configs/eval.yaml").read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load((ROOT / "configs/model.yaml").read_text(encoding="utf-8"))
    prompt_cfg = yaml.safe_load((ROOT / "configs/prompt.yaml").read_text(encoding="utf-8"))
    metric_classes = ["included", "not included", "excluded", "not excluded", "not enough information", "not applicable"]
    files = {name: sha256(ROOT / name) for name in FROZEN_FILES}
    manifest = {
        "schema_version": "1.0", "state": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": {**git_state(), "frozen_file_sha256": files},
        "dataset_id": split_manifest["dataset_id"],
        "dataset_revision": split_manifest["dataset_revision"],
        "split_manifest_sha256": sha256(split_manifest_path),
        "split_ids": {key: {"count": len(split_manifest["annotation_ids"][key]),
                            "sha256": hashlib.sha256(json.dumps(sorted(split_manifest["annotation_ids"][key]), separators=(",", ":")).encode()).hexdigest()}
                      for key in ("train", "validation", "test")},
        "expected_test_count": len(expected_ids), "expected_test_ids": expected_ids,
        "files": files,
        "resolved": {
            "eval": eval_cfg, "model": model_cfg, "prompt": prompt_cfg,
            "aggregation": yaml.safe_load((ROOT / "configs/aggregation.yaml").read_text(encoding="utf-8")),
            "metric_class_order": metric_classes,
            "generation_policy": "transformers-greedy-v1+repetition-policy-v1",
            "backend": "transformers_generate_greedy_cache", "dtype": eval_cfg["dtype"],
        },
        "shared_hashes": {"eval_config": files["configs/eval.yaml"],
                          "model_config": files["configs/model.yaml"],
                          "prompt_config": files["configs/prompt.yaml"],
                          "aggregation_config": files["configs/aggregation.yaml"],
                          "metric_implementation": files["src/clinical_match/metrics.py"]},
        "variants": {
            "base": {"model_id": eval_cfg["model_id"], "model_revision": eval_cfg["model_revision"],
                     "adapter_path": None, "adapter_sha256": None},
            "adapted": {"model_id": eval_cfg["model_id"], "model_revision": eval_cfg["model_revision"],
                        "adapter_path": ADAPTER.as_posix(), "adapter_sha256": adapter_hash},
        },
        "selected_training_bundle": bundle_identity,
        "runtime": {"platform": platform.platform(), "python": sys.version,
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "development_evidence": {
            "validation_metrics_sha256": sha256(validation_dir / "metrics.json"),
            "adapter_decode_correctness_sha256": sha256(ROOT / "results/cache/correctness_adapter.jsonl"),
            "adapter_speculative_correctness_sha256": sha256(ROOT / "results/speculative/correctness_adapter.jsonl"),
            "adapter_validation_ledger_sha256": sha256(ROOT / "results/errors/development_adapter.md"),
            "app_smoke_sha256": sha256(app_smoke),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(contents, encoding="utf-8", newline="\n")
    digest = sha256(args.output)
    args.output.with_suffix(".sha256").write_text(f"{digest}  {args.output.name}\n", encoding="ascii", newline="\n")
    print(json.dumps({"manifest": str(args.output), "sha256": digest,
                      "test_count": len(expected_ids), "variants": list(manifest["variants"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
