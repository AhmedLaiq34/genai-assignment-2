"""Compare speculative target tokens with cached target-greedy tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.clinical_match.decoding import cached_explicit_greedy, speculative_greedy
from src.clinical_match.evaluation import (
    hash_directory,
    load_eval_model,
    prepare_prompts,
    resolve_dtype,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, cwd=ROOT
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL, cwd=ROOT
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def first_difference(reference: list[int], candidate: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return index
    return None if len(reference) == len(candidate) else min(len(reference), len(candidate))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--caps", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--draft-k", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/speculative/correctness.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/speculative/correctness_summary.json"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    if args.rows <= 0:
        raise ValueError("rows must be positive")
    if any(cap <= 0 or cap > cfg["max_new_tokens"] for cap in args.caps):
        raise ValueError("caps must be positive and within the configured max_new_tokens")
    if any(k <= 0 for k in args.draft_k):
        raise ValueError("draft-k values must be positive")

    split = "validation"
    examples_path = ROOT / f"data/processed/development/{split}.jsonl"
    manifest_path = ROOT / "data/splits/manifest.json"
    examples = read_jsonl(examples_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = manifest["annotation_ids"][split]
    observed_ids = [row["provenance"]["annotation_id"] for row in examples]
    if set(observed_ids) != set(expected_ids) or len(observed_ids) != len(set(observed_ids)):
        raise ValueError("Development records do not match the canonical split manifest")
    examples = examples[: args.rows]

    device = torch.device(cfg.get("device", "cuda"))
    adapter_path = args.adapter
    if adapter_path is not None and not adapter_path.is_absolute():
        adapter_path = ROOT / adapter_path
    tokenizer, target = load_eval_model(cfg, device, adapter_path=adapter_path)
    draft_cfg = model_cfg["draft"]
    draft_tokenizer = AutoTokenizer.from_pretrained(
        draft_cfg["id"], revision=draft_cfg["revision"], use_fast=True
    )
    if draft_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError("Target and draft tokenizers do not have identical vocabularies")
    draft = AutoModelForCausalLM.from_pretrained(
        draft_cfg["id"],
        revision=draft_cfg["revision"],
        dtype=resolve_dtype(cfg.get("dtype", "float16"), device),
        low_cpu_mem_usage=True,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    ).to(device)
    draft.eval()

    prepared = prepare_prompts(tokenizer, examples, cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, prompt_row in enumerate(prepared):
        prompt = prompt_row["input_ids"]
        for cap in args.caps:
            reference = cached_explicit_greedy(
                target,
                prompt,
                max_new_tokens=cap,
                eos_token_id=tokenizer.eos_token_id,
                device=device,
            )
            for draft_k in args.draft_k:
                speculative = speculative_greedy(
                    target,
                    draft,
                    prompt,
                    max_new_tokens=cap,
                    eos_token_id=tokenizer.eos_token_id,
                    draft_k=draft_k,
                    device=device,
                )
                difference = first_difference(
                    reference["token_ids"], speculative["token_ids"]
                )
                record = {
                    "annotation_id": examples[index]["provenance"]["annotation_id"],
                    "prompt_tokens": len(prompt),
                    "cap": cap,
                    "draft_k": draft_k,
                    "match": difference is None,
                    "first_difference": difference,
                    "reference_tokens": reference["token_ids"],
                    "speculative_tokens": speculative["token_ids"],
                    "reference_termination": reference["termination"],
                    "speculative_termination": speculative["termination"],
                    "stats": speculative["stats"],
                    "blocks": speculative["blocks"],
                    "target_cache_length": speculative["target_cache_length"],
                    "draft_cache_length": speculative["draft_cache_length"],
                }
                records.append(record)
                print(
                    f"row={index} prompt={len(prompt)} cap={cap} k={draft_k} "
                    f"match={record['match']} "
                    f"acc={speculative['stats']['acceptance_rate']:.3f}",
                    flush=True,
                )
                with args.output.open("w", encoding="utf-8") as handle:
                    for item in records:
                        handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "config": cfg,
        "config_sha256": sha256(args.config),
        "model_config_sha256": sha256(args.model_config),
        "script_sha256": sha256(Path(__file__)),
        "speculative_source_sha256": sha256(
            ROOT / "src/clinical_match/decoding/speculative.py"
        ),
        "reference_backend": "cached_explicit_greedy",
        "generation_backend": "cached_explicit_greedy",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "validation_data_sha256": sha256(examples_path),
        "manifest_sha256": sha256(manifest_path),
        "target_model": cfg["model_id"],
        "target_revision": cfg["model_revision"],
        "adapter_sha256": hash_directory(adapter_path) if adapter_path else None,
        "draft_model": draft_cfg["id"],
        "draft_revision": draft_cfg["revision"],
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "git": git_revision(),
        "rows": len(prepared),
        "caps": args.caps,
        "draft_k": args.draft_k,
        "cases": len(records),
        "matches": sum(record["match"] for record in records),
        "all_match": all(record["match"] for record in records),
        "adapter_pending": adapter_path is None,
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.output} and {args.summary}")
    if not summary["all_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

