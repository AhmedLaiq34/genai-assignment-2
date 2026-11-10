"""Resumable training state and verified checkpoint bundles."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import random
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..checkpointing import create_verified_bundle, restore_verified_bundle, verify_bundle
from ..evaluation import hash_directory


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes(root: Path, config: dict[str, Any]) -> dict[str, str]:
    root = Path(root)
    paths = {
        "train_jsonl": root / "data/processed/development/train.jsonl",
        "validation_jsonl": root / "data/processed/development/validation.jsonl",
        "split_manifest": root / config["split_manifest"],
        "model_config": root / config["model_config"],
        "prompt_config": root / config["prompt_config"],
        "data_config": root / config["data_config"],
        "train_config": root / config.get("train_config_path", "configs/train.yaml"),
    }
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        if path.is_file():
            hashes[name] = sha256_file(path)
        else:
            hashes[name] = "missing"
    sidecar = root / config["manifest_sha256_file"]
    hashes["manifest_sidecar"] = sha256_file(sidecar) if sidecar.is_file() else "missing"
    return hashes


def verify_manifest_sidecar(root: Path, config: dict[str, Any]) -> str:
    root = Path(root)
    manifest = root / config["split_manifest"]
    sidecar = root / config["manifest_sha256_file"]
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(manifest)
    if actual != expected:
        raise ValueError(
            f"Split manifest checksum mismatch: expected {expected}, found {actual}"
        )
    return actual


def environment_report(device: torch.device, precision: str) -> dict[str, Any]:
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "vram_total_bytes": (
            torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else None
        ),
        "precision": precision,
    }


def _package_version(name: str) -> str | None:
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:
        return None


def save_resume_state(
    state_dir: Path,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    trainer_state: dict[str, Any],
    resolved_config: dict[str, Any],
    environment: dict[str, Any],
    data_hashes: dict[str, str],
    tokenizer: Any | None = None,
) -> Path:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(state_dir / "adapter")
    if tokenizer is not None:
        tokenizer.save_pretrained(state_dir / "tokenizer")
    torch.save(optimizer.state_dict(), state_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), state_dir / "scheduler.pt")
    if scaler is not None:
        torch.save(scaler.state_dict(), state_dir / "scaler.pt")
    with (state_dir / "rng.pkl").open("wb") as stream:
        pickle.dump(
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            stream,
        )
    (state_dir / "trainer_state.json").write_text(
        json.dumps(trainer_state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (state_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (state_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (state_dir / "data_hashes.json").write_text(
        json.dumps(data_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return state_dir


def load_resume_state(
    state_dir: Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    state_dir = Path(state_dir)
    trainer_state = json.loads(
        (state_dir / "trainer_state.json").read_text(encoding="utf-8")
    )
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(state_dir / "optimizer.pt", weights_only=False))
    if scheduler is not None and (state_dir / "scheduler.pt").is_file():
        scheduler.load_state_dict(torch.load(state_dir / "scheduler.pt", weights_only=False))
    if scaler is not None and (state_dir / "scaler.pt").is_file():
        scaler.load_state_dict(torch.load(state_dir / "scaler.pt", weights_only=False))
    if restore_rng and (state_dir / "rng.pkl").is_file():
        with (state_dir / "rng.pkl").open("rb") as stream:
            state = pickle.load(stream)
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda"):
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    return trainer_state


def resolve_resume_dir(path: Path, working_root: Path) -> Path:
    """Return a usable state directory from a raw dir or a verified bundle."""
    path = Path(path)
    if (path / "COMPLETE.json").is_file():
        verify_bundle(path)
        destination = Path(
            tempfile.mkdtemp(prefix=f".{path.name}.restore-", dir=working_root)
        )
        shutil.rmtree(destination)
        return restore_verified_bundle(path, destination)
    if (path / "trainer_state.json").is_file():
        return path
    raise ValueError(f"Resume path is neither a verified bundle nor a state directory: {path}")


def make_bundle(
    state_dir: Path,
    bundle_root: Path,
    bundle_name: str,
    metadata: dict[str, Any],
) -> Path:
    return create_verified_bundle(state_dir, bundle_root, bundle_name, metadata)


def publish_bundle(bundle: Path, drive_root: Path) -> Path:
    """Copy a verified local bundle to persistent storage and verify the copy."""
    bundle = Path(bundle)
    drive_root = Path(drive_root)
    drive_root.mkdir(parents=True, exist_ok=True)
    destination = drive_root / bundle.name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite persistent bundle: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{bundle.name}.publish-", dir=drive_root))
    shutil.rmtree(staging)
    try:
        shutil.copytree(bundle, staging)
        verify_bundle(staging)
        os.replace(staging, destination)
        verify_bundle(destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def prune_resume_bundles(bundle_root: Path, keep: int = 2) -> list[Path]:
    """Remove only old project-owned resume bundles; never touch the best adapter."""
    bundle_root = Path(bundle_root)
    if keep < 1 or not bundle_root.is_dir():
        return []
    candidates = sorted(
        path
        for path in bundle_root.iterdir()
        if path.is_dir() and path.name.startswith("resume-step-")
    )
    removed: list[Path] = []
    for path in candidates[:-keep]:
        shutil.rmtree(path)
        removed.append(path)
    return removed
