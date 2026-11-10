"""Shared generation/evaluation helpers used by the CLI harness and trainer."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import torch

from .decoding import cached_library_greedy
from .output_parser import parse_output
from .prompts import build_prompt


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_directory(path: Path) -> str:
    """Stable SHA-256 over sorted relative paths, sizes and file hashes."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(file_path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float16" and device.type == "cuda":
        return torch.float16
    if name == "bfloat16" and device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def load_eval_model(
    cfg: dict[str, Any],
    device: torch.device,
    *,
    adapter_path: Path | str | None = None,
) -> tuple[Any, Any]:
    """Load the pinned base model and optionally a PEFT adapter for inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], use_fast=True
    )
    dtype = resolve_dtype(cfg.get("dtype", "float16"), device)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        revision=cfg["model_revision"],
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    ).to(device)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        model.to(device)
    model.eval()
    return tokenizer, model


def prepare_prompts(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        build_prompt(
            tokenizer,
            row["input"],
            max_total_tokens=cfg["max_total_tokens"],
            max_new_tokens=cfg["max_new_tokens"],
        )
        for row in rows
    ]


def generate_predictions(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: torch.device,
    *,
    prompts: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Run the frozen greedy path over rows and retain every failure."""
    if prompts is None:
        prompts = prepare_prompts(tokenizer, rows, cfg)
    if len(prompts) != len(rows):
        raise ValueError("Prompt and row counts differ")
    predictions: list[dict[str, Any]] = []
    total_seconds = 0.0
    for row, prompt in zip(rows, prompts):
        started = time.perf_counter()
        runtime_status = "ok"
        try:
            generated = cached_library_greedy(
                model,
                prompt["input_ids"],
                max_new_tokens=cfg["max_new_tokens"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            raw = tokenizer.decode(generated["token_ids"], skip_special_tokens=True)
            truncated = generated["hit_token_cap"]
        except Exception as exc:  # Preserve failures as rows, never silently drop them.
            runtime_status = "error"
            raw = ""
            truncated = False
            generated = {"token_ids": [], "error_type": type(exc).__name__, "error": str(exc)}
        elapsed = time.perf_counter() - started
        total_seconds += elapsed
        parsed = parse_output(
            raw,
            row["input"]["criterion_type"],
            set(prompt["kept_sentence_ids"]),
        )
        predictions.append(
            {
                "annotation_id": row["provenance"]["annotation_id"],
                "patient_id": row["provenance"]["patient_id"],
                "trial_id": row["provenance"]["trial_id"],
                "criterion_type": row["input"]["criterion_type"],
                "criterion_text": row["input"]["criterion_text"],
                "gold_label": row["target"]["label"],
                "gold_evidence_ids": row["target"]["evidence_sentence_ids"],
                "prompt_token_ids": prompt["input_ids"],
                "prompt_token_count": len(prompt["input_ids"]),
                "prompt_truncated": prompt["truncated"],
                "removed_sentence_ids": prompt["removed_sentence_ids"],
                "generated_token_ids": generated["token_ids"],
                **parsed,
                "truncated": truncated,
                "runtime_status": runtime_status,
                "runtime_error": generated.get("error") if runtime_status == "error" else None,
                "timing_seconds": elapsed,
                "model_revision": cfg["model_revision"],
                "prompt_revision": cfg["prompt_revision"],
            }
        )
    return predictions, total_seconds
