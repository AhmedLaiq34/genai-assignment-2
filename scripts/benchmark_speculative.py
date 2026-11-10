"""Same-device latency, memory and acceptance benchmark for greedy speculation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
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


def timed_call(fn, repeats: int, warmups: int, device: torch.device) -> dict:
    for _ in range(warmups):
        fn()
    times = []
    peaks = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
        else:
            started = time.perf_counter()
        result = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - started)
        peaks.append(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
    return {
        "times_s": times,
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "peak_allocated_bytes": max(peaks) if peaks and peaks[0] is not None else None,
        "last_result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--caps", type=int, nargs="+", default=[16])
    parser.add_argument("--draft-k", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/speculative/benchmark.json"))
    parser.add_argument("--csv", type=Path, default=Path("results/speculative/benchmark.csv"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    if args.rows <= 0 or args.repeats <= 0 or args.warmups < 0:
        raise ValueError("rows and repeats must be positive; warmups nonnegative")
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
    load_started = time.perf_counter()
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
    load_seconds = time.perf_counter() - load_started
    weight_bytes = sum(
        parameter.numel() * parameter.element_size()
        for model in (target, draft)
        for parameter in model.parameters()
    )

    prepared = prepare_prompts(tokenizer, examples, cfg)
    rows = []
    for prompt_index, prompt_row in enumerate(prepared):
        prompt = prompt_row["input_ids"]
        for cap in args.caps:
            reference = cached_explicit_greedy(
                target,
                prompt,
                max_new_tokens=cap,
                eos_token_id=tokenizer.eos_token_id,
                device=device,
            )
            modes = [("cached", None)] + [("speculative", k) for k in args.draft_k]
            for repeat_index in range(args.repeats + args.warmups):
                order = modes[repeat_index % len(modes):] + modes[: repeat_index % len(modes)]
                for mode, draft_k in order:
                    if mode == "cached":
                        result = timed_call(
                            lambda: cached_explicit_greedy(
                                target,
                                prompt,
                                max_new_tokens=cap,
                                eos_token_id=tokenizer.eos_token_id,
                                device=device,
                            ),
                            repeats=1,
                            warmups=0,
                            device=device,
                        )
                    else:
                        result = timed_call(
                            lambda: speculative_greedy(
                                target,
                                draft,
                                prompt,
                                max_new_tokens=cap,
                                eos_token_id=tokenizer.eos_token_id,
                                draft_k=draft_k,
                                device=device,
                            ),
                            repeats=1,
                            warmups=0,
                            device=device,
                        )
                    payload = result["last_result"]
                    if payload["token_ids"] != reference["token_ids"]:
                        raise RuntimeError(
                            f"Token mismatch for mode={mode} k={draft_k} cap={cap}"
                        )
                    if repeat_index < args.warmups:
                        continue
                    existing = next(
                        (
                            row
                            for row in rows
                            if row["prompt_index"] == prompt_index
                            and row["cap"] == cap
                            and row["mode"] == mode
                            and row["draft_k"] == draft_k
                        ),
                        None,
                    )
                    if existing is None:
                        existing = {
                            "prompt_index": prompt_index,
                            "prompt_tokens": len(prompt),
                            "cap": cap,
                            "mode": mode,
                            "draft_k": draft_k,
                            "times_s": [],
                            "peak_allocated_bytes": [],
                            "stats": None,
                            "token_match": True,
                        }
                        rows.append(existing)
                    existing["times_s"].append(result["median_s"])
                    existing["peak_allocated_bytes"].append(result["peak_allocated_bytes"])
                    existing["stats"] = payload.get("stats")

    cached_medians = {
        (row["prompt_index"], row["cap"]): statistics.median(row["times_s"])
        for row in rows
        if row["mode"] == "cached"
    }
    for row in rows:
        row["median_s"] = statistics.median(row["times_s"])
        row["min_s"] = min(row["times_s"])
        row["max_s"] = max(row["times_s"])
        row["peak_allocated_bytes"] = max(
            value for value in row["peak_allocated_bytes"] if value is not None
        ) if row["peak_allocated_bytes"] and row["peak_allocated_bytes"][0] is not None else None
        row["speedup"] = cached_medians[(row["prompt_index"], row["cap"])] / row["median_s"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": cfg,
        "config_sha256": sha256(args.config),
        "model_config_sha256": sha256(args.model_config),
        "script_sha256": sha256(Path(__file__)),
        "speculative_source_sha256": sha256(ROOT / "src/clinical_match/decoding/speculative.py"),
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
        "dtype": str(resolve_dtype(cfg.get("dtype", "float16"), device)),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "reference_backend": "cached_explicit_greedy",
        "generation_backend": "cached_explicit_greedy",
        "git": git_revision(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "model_load_seconds": load_seconds,
        "loaded_weight_bytes": weight_bytes,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "protocol": (
            "Same device and process, both models resident, load excluded from timers. "
            "Each timed run starts after CUDA synchronize and empty_cache; peak memory "
            "includes both model weights and KV caches. Token equality is checked against "
            "the target cached-greedy reference."
        ),
        "adapter_pending": adapter_path is None,
        "rows": rows,
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "prompt_index",
            "prompt_tokens",
            "cap",
            "mode",
            "draft_k",
            "median_s",
            "speedup",
            "peak_allocated_bytes",
            "acceptance_rate",
            "proposed_tokens",
            "accepted_tokens",
            "corrections",
            "bonuses",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            stats = row["stats"] or {}
            writer.writerow(
                {
                    "prompt_index": row["prompt_index"],
                    "prompt_tokens": row["prompt_tokens"],
                    "cap": row["cap"],
                    "mode": row["mode"],
                    "draft_k": row["draft_k"],
                    "median_s": row["median_s"],
                    "speedup": row["speedup"],
                    "peak_allocated_bytes": row["peak_allocated_bytes"],
                    "acceptance_rate": stats.get("acceptance_rate", ""),
                    "proposed_tokens": stats.get("proposed_tokens", ""),
                    "accepted_tokens": stats.get("accepted_tokens", ""),
                    "corrections": stats.get("correction_tokens", ""),
                    "bonuses": stats.get("bonus_tokens", ""),
                }
            )
    print(f"wrote {args.output} and {args.csv}")


if __name__ == "__main__":
    main()

