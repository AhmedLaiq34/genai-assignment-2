"""Run the local demo in the checked launcher process to avoid a second CUDA context."""
from __future__ import annotations
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "artifacts/phase04/phase04_lora_colab_cli_20261110T1650Z_bundle/selected_adapter"

# Limit CPU worker pools before importing PyTorch. This also avoids oversubscribing
# the Windows host when tokenization and CUDA dispatch run together.
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_HUB_CACHE"] = str(ROOT / ".cache/huggingface/hub")
os.environ["PROJECT_RUNTIME_PROFILE"] = "local"


def main() -> int:
    if not ADAPTER.is_dir() or not (ADAPTER / "adapter_model.safetensors").is_file():
        print(f"Selected adapter not found or incomplete: {ADAPTER}", file=sys.stderr)
        return 2
    try:
        import torch
    except Exception as exc:
        print(f"Could not load the project PyTorch runtime: {exc}", file=sys.stderr)
        return 2
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        print("CUDA is unavailable. The demo requires the configured GPU and will not silently fall back to CPU or the base model.", file=sys.stderr)
        return 2
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print("Adapter: selected Phase 4 epoch 3", flush=True)
    os.chdir(ROOT)
    runpy.run_path(str(ROOT / "app/main.py"), run_name="__main__")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
