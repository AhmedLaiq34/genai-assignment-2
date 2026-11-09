"""Record machine and ML-runtime facts; optionally run a small CUDA smoke test."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "huggingface-hub",
    "numpy",
    "scikit-learn",
    "gradio",
)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def nvidia_smi_report() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found on PATH"}
    result = subprocess.run(
        [
            executable,
            "--query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        return {
            "available": False,
            "returncode": result.returncode,
            "stderr": result.stderr.strip(),
        }
    devices = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 7:
            devices.append(
                {
                    "name": fields[0],
                    "driver_version": fields[1],
                    "memory_total_mib": fields[2],
                    "memory_free_mib": fields[3],
                    "temperature_c": fields[4],
                    "power_draw_w": fields[5],
                    "power_limit_w": fields[6],
                }
            )
    return {"available": bool(devices), "devices": devices}


def memory_report() -> dict[str, int | None]:
    try:
        import psutil
    except ImportError:
        return {"total_bytes": None, "available_bytes": None}
    memory = psutil.virtual_memory()
    return {
        "total_bytes": int(memory.total),
        "available_bytes": int(memory.available),
    }


def cuda_smoke_test() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "error": "PyTorch is not installed"}

    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        return report

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    report["device"] = {
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "bfloat16_supported": torch.cuda.is_bf16_supported(),
    }
    torch.cuda.reset_peak_memory_stats(device)
    left = torch.randn((128, 128), device=device, dtype=torch.float32)
    product = left @ left
    torch.cuda.synchronize(device)
    report["matrix_multiply"] = {
        "passed": bool(torch.isfinite(product).all().item()),
        "shape": list(product.shape),
    }

    layer = torch.nn.TransformerEncoderLayer(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        batch_first=True,
        dropout=0.0,
    ).to(device)
    layer.eval()
    tokens = torch.randn((1, 8, 32), device=device)
    with torch.inference_mode():
        output = layer(tokens)
    torch.cuda.synchronize(device)
    report["tiny_forward"] = {
        "passed": bool(torch.isfinite(output).all().item()),
        "shape": list(output.shape),
    }
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    report["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return report


def collect_report(run_smoke_test: bool) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    storage: dict[str, dict[str, int]] = {}
    storage_roots = [project_root]
    if os.name == "nt":
        storage_roots = [
            Path(f"{drive}:\\")
            for drive in ("C", "D")
            if Path(f"{drive}:\\").exists()
        ]
    for path in storage_roots:
        usage = shutil.disk_usage(path)
        key = "project" if path == project_root else (path.drive or str(path))
        storage[key] = {
            "total_bytes": usage.total,
            "free_bytes": usage.free,
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "profile": os.environ.get("PROJECT_RUNTIME_PROFILE", "local"),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "windows_version": platform.win32_ver(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "cpu_memory": memory_report(),
            "project_storage_path": str(project_root),
            "storage": storage,
        },
        "packages": package_versions(),
        "nvidia_smi": nvidia_smi_report(),
        "environment_variables": {
            name: os.environ.get(name)
            for name in ("HF_HOME", "HF_HUB_CACHE", "UV_CACHE_DIR", "PIP_CACHE_DIR")
        },
    }
    if run_smoke_test:
        report["cuda_smoke_test"] = cuda_smoke_test()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the report as JSON.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run an actual CUDA matrix multiplication and tiny Transformer forward pass.",
    )
    args = parser.parse_args()
    report = collect_report(args.smoke_test)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote environment report to {args.output.resolve()}")
    smoke = report.get("cuda_smoke_test", {})
    if args.smoke_test and smoke.get("cuda_available"):
        tests = (smoke.get("matrix_multiply", {}), smoke.get("tiny_forward", {}))
        return 0 if all(test.get("passed") for test in tests) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
