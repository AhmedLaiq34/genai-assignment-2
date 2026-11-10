"""Compare explicit uncached and explicitly cached greedy token IDs on dev prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
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


def first_mismatch(left: list[int], right: list[int]) -> dict | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return {"index": index, "reference": a, "cached": b}
    if len(left) != len(right):
        index = min(len(left), len(right))
        return {
            "index": index,
            "reference": left[index] if index < len(left) else None,
            "cached": right[index] if index < len(right) else None,
        }
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval.yaml")
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/processed/development/validation.jsonl"
    )
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=None,
        help="Optional synthetic prompt lengths built by tiling/truncating a development prompt.",
    )
    parser.add_argument("--caps", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/cache/correctness.jsonl"
    )
    args = parser.parse_args()
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    cases: list[tuple[dict, dict, list[int], str]] = []
    for row in rows:
        prompt = build_prompt(
            tokenizer,
            row["input"],
            max_total_tokens=int(cfg["max_total_tokens"]),
            max_new_tokens=int(cfg["max_new_tokens"]),
        )
        if args.lengths:
            base = prompt["input_ids"]
            for target in args.lengths:
                repeated = (base * ((target + len(base) - 1) // len(base)))[:target]
                cases.append((row, prompt, repeated, "synthetic_tiled"))
        else:
            cases.append((row, prompt, prompt["input_ids"], "development_prompt"))
    with args.output.open("w", encoding="utf-8") as stream:
        for row, prompt, input_ids, prompt_source in cases:
            prompt_hash = hashlib.sha256(
                json.dumps(input_ids, separators=(",", ":")).encode()
            ).hexdigest()
            for cap in args.caps:
                record = {
                    "annotation_id": row["provenance"]["annotation_id"],
                    "prompt_source": prompt_source,
                    "synthetic_truncated": prompt_source == "synthetic_tiled",
                    "prompt_tokens": len(input_ids),
                    "prompt_sha256": prompt_hash,
                    "prompt_truncated": prompt["truncated"],
                    "max_new_tokens": cap,
                    **provenance,
                }
                try:
                    reference = uncached_greedy(
                        model,
                        input_ids,
                        max_new_tokens=cap,
                        eos_token_id=tokenizer.eos_token_id,
                        device=device,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    cached = cached_explicit_greedy(
                        model,
                        input_ids,
                        max_new_tokens=cap,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        device=device,
                    )
                    mismatch = first_mismatch(
                        reference["token_ids"], cached["token_ids"]
                    )
                    record.update(
                        {
                            "status": "ok",
                            "reference_token_ids": reference["token_ids"],
                            "cached_token_ids": cached["token_ids"],
                            "exact_match": mismatch is None,
                            "first_mismatch": mismatch,
                            "reference_hit_cap": reference["hit_token_cap"],
                            "cached_hit_cap": cached["hit_token_cap"],
                            "termination": cached["termination"],
                            "generated_tokens": len(cached["token_ids"]),
                            "cache_lengths": cached["cache_lengths"],
                            "final_cache_length": cached["final_cache_length"],
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - a skip must remain visible
                    record.update(
                        {
                            "status": "error",
                            "exact_match": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(
                    f"annotation={record['annotation_id']} prompt={record['prompt_tokens']} "
                    f"cap={cap} status={record['status']} match={record.get('exact_match')}"
                )
    ok = [record for record in records if record["status"] == "ok"]
    matches = [record for record in ok if record["exact_match"]]
    summary = {
        "provenance": provenance,
        "records": len(records),
        "completed": len(ok),
        "errors": len(records) - len(ok),
        "exact_matches": len(matches),
        "all_completed_match": len(ok) == len(records) and len(matches) == len(records),
        "distinct_prompt_lengths": sorted({record["prompt_tokens"] for record in ok}),
        "distinct_caps": sorted({record["max_new_tokens"] for record in ok}),
    }
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_completed_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
