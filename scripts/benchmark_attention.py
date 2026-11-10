from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.clinical_match.attention import causal_mask, dense_attention, tiled_attention


def _dtype(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {name}")
    return value


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _git_revision() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _benchmark(fn, warmups: int, repeats: int, device: torch.device) -> dict:
    for _ in range(warmups):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    import time

    times = []
    peaks = []
    increments = []
    for _ in range(repeats):
        baseline = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
        start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if start is not None:
            start.record()
        else:
            cpu_start = time.perf_counter()
        fn()
        if start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end) / 1000.0)
            peak = torch.cuda.max_memory_allocated(device)
            peaks.append(peak)
            increments.append(peak - baseline)
        else:
            times.append(time.perf_counter() - cpu_start)
            peaks.append(None)
            increments.append(None)
    cuda_peaks = [p for p in peaks if p is not None]
    cuda_increments = [p for p in increments if p is not None]
    return {
        "times_s": times,
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "peak_allocated_bytes_per_repeat": peaks,
        "peak_incremental_bytes_per_repeat": increments,
        "peak_allocated_bytes": max(cuda_peaks) if cuda_peaks else None,
        "peak_incremental_bytes": max(cuda_increments) if cuda_increments else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dense vs online-softmax tiled attention")
    parser.add_argument("--config", type=Path, default=Path("configs/attention_bench.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/attention/benchmark.json"))
    parser.add_argument("--csv", type=Path, default=Path("results/attention/benchmark.csv"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = _resolve_device(config.get("device", "auto"))
    dtype = _dtype(config.get("dtype", "float32"))
    batch = int(config["batch_size"])
    heads = int(config["heads"])
    kv_heads = int(config["kv_heads"])
    head_dim = int(config["head_dim"])
    lengths = [int(x) for x in config["lengths"]]
    tiles = [int(x) for x in config["tiles"]]
    warmups = int(config.get("warmups", 2))
    repeats = int(config.get("repeats", 5))
    seed = int(config.get("seed", 2026))
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads:
        raise ValueError("heads must be positive and divisible by kv_heads")

    generator = torch.Generator(device=device).manual_seed(seed)
    rows = []
    for length in lengths:
        q = torch.randn(batch, heads, length, head_dim, generator=generator, dtype=dtype, device=device)
        k = torch.randn(batch, kv_heads, length, head_dim, generator=generator, dtype=dtype, device=device)
        v = torch.randn(batch, kv_heads, length, head_dim, generator=generator, dtype=dtype, device=device)
        mask = causal_mask(length, length, device=device).unsqueeze(0).unsqueeze(0)
        full_bytes = batch * heads * length * length * torch.empty((), dtype=dtype).element_size()
        try:
            dense = _benchmark(
                lambda: dense_attention(q, k, v, mask), warmups, repeats, device
            )
            dense_result = dense_attention(q, k, v, mask)
            dense_ok = True
            dense_error = None
        except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            dense_result = None
            dense = None
            dense_ok = False
            dense_error = {"type": type(exc).__name__, "message": str(exc)}
        for tile in tiles:
            tile_bytes = batch * heads * tile * tile * torch.empty((), dtype=dtype).element_size()
            try:
                tiled = _benchmark(
                    lambda: tiled_attention(q, k, v, mask, query_tile=tile, key_tile=tile),
                    warmups,
                    repeats,
                    device,
                )
                tiled_result = tiled_attention(
                    q, k, v, mask, query_tile=tile, key_tile=tile
                )
                tiled_ok = True
                tiled_error = None
            except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                tiled_result = None
                tiled = None
                tiled_ok = False
                tiled_error = {"type": type(exc).__name__, "message": str(exc)}
            error = {}
            if dense_ok and tiled_ok:
                diff = (tiled_result.float() - dense_result.float()).abs()
                relative = diff / dense_result.float().abs()
                relative = relative[dense_result.float().abs() > 1e-3]
                error = {
                    "max_abs_error": float(diff.max()),
                    "mean_abs_error": float(diff.mean()),
                    "max_relative_error_above_1e-3": (
                        float(relative.max()) if relative.numel() else None
                    ),
                }
            row = {
                "length": length,
                "batch": batch,
                "heads": heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "dtype": str(dtype),
                "device": str(device),
                "tile": tile,
                "dense": dense,
                "dense_error": dense_error,
                "tiled": tiled,
                "tiled_error": tiled_error,
                "error": error,
                "full_score_matrix_bytes": full_bytes,
                "max_tile_score_bytes": tile_bytes,
                "tiled_full_score_matrix_construction": False,
            }
            if dense_ok and tiled_ok:
                row["speedup"] = dense["median_s"] / tiled["median_s"]
                row["peak_delta_bytes"] = (
                    tiled["peak_incremental_bytes"] - dense["peak_incremental_bytes"]
                    if tiled["peak_incremental_bytes"] is not None
                    and dense["peak_incremental_bytes"] is not None
                    else None
                )
            rows.append(row)
            print(
                f"length={length} tile={tile} dense={row.get('speedup')} "
                f"ok={dense_ok and tiled_ok}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": config,
        "config_sha256": _sha256(args.config),
        "script_sha256": _sha256(Path(__file__)),
        "source_sha256": {
            "dense": _sha256(ROOT / "src/clinical_match/attention/dense.py"),
            "tiled": _sha256(ROOT / "src/clinical_match/attention/tiled.py"),
            "masks": _sha256(ROOT / "src/clinical_match/attention/masks.py"),
        },
        "command": [sys.executable, *sys.argv],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "git": _git_revision(),
        "score_matrix_note": (
            "Dense materializes [B,H,L,L]; tiled allocates [B,H,Tq,Tk] score tiles "
            "plus tile-sized temporaries. Peak memory includes inputs and allocator state; "
            "incremental peaks subtract the pre-call allocated baseline."
        ),
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "length",
            "tile",
            "dtype",
            "device",
            "dense_median_s",
            "tiled_median_s",
            "speedup",
            "dense_peak_allocated_bytes",
            "tiled_peak_allocated_bytes",
            "dense_peak_incremental_bytes",
            "tiled_peak_incremental_bytes",
            "max_abs_error",
            "mean_abs_error",
            "full_score_matrix_bytes",
            "max_tile_score_bytes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "length": row["length"],
                    "tile": row["tile"],
                    "dtype": row["dtype"],
                    "device": row["device"],
                    "dense_median_s": row["dense"]["median_s"] if row["dense"] else "",
                    "tiled_median_s": row["tiled"]["median_s"] if row["tiled"] else "",
                    "speedup": row.get("speedup", ""),
                    "dense_peak_allocated_bytes": row["dense"]["peak_allocated_bytes"] if row["dense"] else "",
                    "tiled_peak_allocated_bytes": row["tiled"]["peak_allocated_bytes"] if row["tiled"] else "",
                    "dense_peak_incremental_bytes": row["dense"]["peak_incremental_bytes"] if row["dense"] else "",
                    "tiled_peak_incremental_bytes": row["tiled"]["peak_incremental_bytes"] if row["tiled"] else "",
                    "max_abs_error": row["error"].get("max_abs_error", ""),
                    "mean_abs_error": row["error"].get("mean_abs_error", ""),
                    "full_score_matrix_bytes": row["full_score_matrix_bytes"],
                    "max_tile_score_bytes": row["max_tile_score_bytes"],
                }
            )
    print(f"wrote {args.output} and {args.csv}")


if __name__ == "__main__":
    main()
