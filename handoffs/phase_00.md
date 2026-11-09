# Phase 0 handoff

## Status

**Phase 0 setup and acceptance checks: complete.** The local runtime and a
real Colab T4 runtime were inspected, tested, and recorded. A checksummed
result bundle was transferred from Colab to D: and restored successfully.

## Implemented

- Project-local Python 3.11.17 venv at `.venv`.
- CUDA-enabled PyTorch 2.11.0+cu128 and pinned shared requirements in
  `requirements-base.txt`; `requirements-local.lock` captures the tested
  Windows environment.
- D:-resident Hugging Face, pip and uv cache paths configured by local
  activation/setup scripts.
- Runtime profiles in `configs/runtime_local.yaml` and
  `configs/runtime_colab.yaml`.
- `scripts/doctor.py` records OS, Python/packages, CPU RAM, disk, NVIDIA
  device/driver/VRAM, PyTorch CUDA build, capability and BF16 support.
- Local CUDA matrix multiply and tiny Transformer forward pass recorded in
  `results/environments/local.json`; both passed on the RTX 3050 6GB laptop.
- `src/clinical_match/checkpointing.py` creates atomic checksum-verified
  bundles. Synthetic save/verify/restore and corruption/overwrite tests pass.
- `scripts/package_colab_source.py` creates a source-only archive and SHA-256
  sidecar. It excludes caches, model weights, raw Parquet and prepared JSONL.
- Thin interactive Colab launcher at `notebooks/colab_runner.ipynb`.
- Official `googlecolab/google-colab-cli` installed in an isolated Ubuntu WSL
  environment; authenticated interactively by the account owner.
- Standard Tesla T4 session: 16 GiB VRAM, Colab Python 3.13.16,
  PyTorch 2.14.1+cu130; CUDA matrix multiply and tiny Transformer passed.
- Colab report saved at
  `results/environments/colab_20261109T110002Z.json`; summary pointer saved
  in `results/environment.json`.
- Five report files were checksum-bundled on the Colab VM, restore-tested,
  downloaded with the CLI, then verified and restored locally to
  `results/colab_verified_bundle`. Archive SHA-256:
  `a7494de262ae44edcd55c2cc08cd7bfd9eeed42aa6a91331c7f2fcb296b7f214`.

## Acceptance evidence

```text
Python: 3.11.17
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
Driver: 610.74
PyTorch: 2.11.0+cu128 (CUDA build 12.8)
CUDA available: true
Matrix multiply: pass
Tiny Transformer forward: pass
Available D: storage at latest report time: 94,251,409,408 bytes
Synthetic checkpoint round-trip/checksum tests: pass (8 project tests total)
```

Current free memory/disk/temperature are snapshots, not guarantees. A single
Colab T4 session was allocated under the current account access. The CLI
reported a 0.00 compute-unit balance and 1.15 compute-units/hour while it was
active, so the session was stopped after completing the pilots. A subsequent
check showed no active sessions, 0.00 compute-units/hour, and 0 active
assignments. Do not assume future GPU usage is free or allocate another
runtime without checking current usage and user authorization. The latest
local environment report captured 54 C GPU temperature; current readings can
change.
The user must verify the exact Classroom deadline time and live-demo
arrangements separately.

No additional Phase 0 action is required. The Colab notebook remains available
for future runs, but check compute-unit usage before allocating a new session.
