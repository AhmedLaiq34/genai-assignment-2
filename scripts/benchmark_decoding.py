"""Benchmark explicit uncached and KV-cached greedy decoding on development prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.clinical_match.decoding import cached_explicit_greedy, uncached_greedy
from src.clinical_match.evaluation import hash_directory, load_eval_model
from src.clinical_match.prompts import build_prompt


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return revision, dirty
    except Exception:  # noqa: BLE001 - provenance is best effort
        return None, None


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_run(fn, device: torch.device) -> tuple[dict, float, int | None]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    started = time.perf_counter()
    result = fn()
    sync(device)
    elapsed = time.perf_counter() - started
    peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    return result, elapsed, peak


def summarize(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval.yaml")
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/processed/development/validation.jsonl"
    )
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cap", type=int, default=16)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument(
        "--modes", nargs="+", choices=["uncached", "cached_explicit"], default=None
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "results/cache/benchmark.json")
    args = parser.parse_args()
    modes = args.modes or ["uncached", "cached_explicit"]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    rows = read_jsonl(args.data)[: args.rows]
    if not rows:
        raise ValueError("No development rows found")
    adapter_path = args.adapter
    if adapter_path is not None and not adapter_path.is_absolute():
        adapter_path = ROOT / adapter_path
    tokenizer, model = load_eval_model(cfg, device, adapter_path=adapter_path)
    revision, dirty = git_revision()
    provenance = {
        "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "tokenizer_revision": cfg.get("tokenizer_revision", cfg["model_revision"]),
        "prompt_revision": cfg.get("prompt_revision"),
        "config_sha256": sha256_file(args.config),
        "dataset_sha256": sha256_file(args.data),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "dtype": cfg.get("dtype"),
        "attention_implementation": cfg.get("attn_implementation", "sdpa"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "eos_token_id": tokenizer.eos_token_id,
        "adapter_sha256": hash_directory(adapter_path) if adapter_path else None,
        "git_revision": revision,
        "git_dirty": dirty,
    }
    records: list[dict] = []
    for row in rows:
        prompt = build_prompt(
            tokenizer,
            row["input"],
            max_total_tokens=int(cfg["max_total_tokens"]),
            max_new_tokens=int(cfg["max_new_tokens"]),
        )
        prompt_hash = hashlib.sha256(
            json.dumps(prompt["input_ids"], separators=(",", ":")).encode()
        ).hexdigest()
        for mode in modes:
            if mode == "uncached":
                fn = lambda: uncached_greedy(
                    model,
                    prompt["input_ids"],
                    max_new_tokens=args.cap,
                    eos_token_id=tokenizer.eos_token_id,
                    device=device,
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                fn = lambda: cached_explicit_greedy(
                    model,
                    prompt["input_ids"],
                    max_new_tokens=args.cap,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                    measure_timing=True,
                )
            for _ in range(args.warmups):
                timed_run(fn, device)
            repetitions = []
            last_result = None
            for _ in range(args.repeats):
                result, elapsed, peak = timed_run(fn, device)
                last_result = result
                timings = result.get("timings")
                generated_count = len(result["token_ids"])
                repetitions.append(
                    {
                        "seconds": elapsed,
                        "peak_allocated_bytes": peak,
                        "timings": timings,
                        "ttft_seconds": timings["prefill_seconds"] if timings else None,
                        "decode_tokens_per_second": (
                            max(0, generated_count - 1) / timings["decode_seconds"]
                            if timings and timings["decode_seconds"] > 0
                            else None
                        ),
                        "end_to_end_tokens_per_second": (
                            generated_count / elapsed if elapsed > 0 else None
                        ),
                    }
                )
            hit_cap = bool(last_result.get("hit_token_cap", False))
            records.append(
                {
                    "annotation_id": row["provenance"]["annotation_id"],
                    "mode": mode,
                    "prompt_tokens": len(prompt["input_ids"]),
                    "prompt_sha256": prompt_hash,
                    "max_new_tokens": args.cap,
                    "generated_tokens": len(last_result["token_ids"]),
                    "generated_token_ids": last_result["token_ids"],
                    "hit_token_cap": hit_cap,
                    "termination": "cap" if hit_cap else "eos",
                    "warmups": args.warmups,
                    "repetitions": repetitions,
                    "latency_seconds": summarize(
                        [item["seconds"] for item in repetitions]
                    ),
                    "ttft_seconds": summarize(
                        [
                            item["ttft_seconds"]
                            for item in repetitions
                            if item["ttft_seconds"] is not None
                        ]
                    )
                    if any(item["ttft_seconds"] is not None for item in repetitions)
                    else None,
                    "end_to_end_tokens_per_second": summarize(
                        [
                            item["end_to_end_tokens_per_second"]
                            for item in repetitions
                            if item["end_to_end_tokens_per_second"] is not None
                        ]
                    ),
                    **provenance,
                }
            )
            print(
                f"{mode} annotation={row['provenance']['annotation_id']} "
                f"prompt={len(prompt['input_ids'])} cap={args.cap} "
                f"mean={records[-1]['latency_seconds']['mean']:.4f}s"
            )
    grouped: dict[tuple, dict] = {}
    for record in records:
        key = (
            record["annotation_id"],
            record["prompt_sha256"],
            record["max_new_tokens"],
        )
        grouped.setdefault(key, {})[record["mode"]] = record
    comparisons = []
    for (annotation_id, prompt_hash, cap), pair in grouped.items():
        if "uncached" in pair and "cached_explicit" in pair:
            slow = pair["uncached"]["latency_seconds"]["mean"]
            fast = pair["cached_explicit"]["latency_seconds"]["mean"]
            comparisons.append(
                {
                    "annotation_id": annotation_id,
                    "prompt_sha256": prompt_hash,
                    "prompt_tokens": pair["uncached"]["prompt_tokens"],
                    "max_new_tokens": cap,
                    "uncached_mean_seconds": slow,
                    "cached_explicit_mean_seconds": fast,
                    "speedup": slow / fast if fast else None,
                }
            )
    report = {
        "provenance": provenance,
        "summary": {"comparisons": comparisons},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
