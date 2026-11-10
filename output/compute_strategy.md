# Compute strategy: Colab research and local application

Updated 10 November 2026 (Asia/Karachi). The user selected Google Colab for
Phase 4 training after Kaggle did not provide an enabled GPU accelerator. The
laptop remains the development, artifact verification, and final-application
environment. See the [Colab training guide](../docs/colab_manual_setup.md),
[Phase 4 Colab handoff](../handoffs/phase_04_colab_training_handoff.md), and
[runner notebook](../notebooks/colab_runner.ipynb).

## Active route

1. Develop and run portable CPU/toy checks locally.
2. Build a source-only ZIP with a SHA-256 sidecar and upload both, plus the
   current runner notebook, to Google Drive.
3. In Colab select **Runtime > Change runtime type > GPU**. Run the environment
   doctor and require `cuda_available: true` plus a finite CUDA operation.
4. Run the notebook's data/hash verification, test suite, overfit gate, and
   interrupted/resumed rehearsal before full training.
5. Run the configured three-epoch FP16 LoRA job on the approved training rows,
   validating and selecting only on the validation split. Persist verified
   checkpoints to Drive.
6. Download the selected export and checksums to D:, verify the bundle, reload
   it locally, and compare it to the unchanged base on all 206 validation rows.

Colab GPU allocation, runtime duration, storage, and package environment can
change. Record the actual device and software versions for each run. If CUDA
is unavailable, stop; do not train on CPU, retry full training locally, or
silently switch precision/model. Preserve the sealed test split until Phase 9.

## Training and persistence boundary

The Colab runner invokes repository scripts rather than duplicating model or
training logic. Its ordered gates cover pinned data regeneration and manifest
verification, the complete local test suite, overfit smoke, resume rehearsal,
full training with per-epoch validation, selected export, and bundle checks.
The training configuration remains `configs/train.yaml`.

The source archive excludes raw Parquet, prepared patient JSONL, model weights,
caches, and credentials. Colab `/content` is temporary; persist resumable
bundles under `MyDrive/GenAI_Assignment_2_checkpoints/<run>/`. Verify a copied
bundle before pruning or resuming from it. Keep only verified bundles and
selected exports; notebook output alone is not durable evidence.

## Comparisons and local application

Keep Colab selection metrics separate from fresh local reload metrics. Use the
same prompt, parser, budgets, base revision, and all 206 validation rows for
the local before/after comparison. Never compare unlike-device timings as a
speedup. The final app runs locally without a Colab mount, active session,
cloud authorization, or cloud GPU.

## Historical Kaggle attempt

Kaggle CLI authentication succeeded, but the GPU-enabled private probe
received `torch 2.11.0+cpu` and `cuda_available: false`; the notebook UI also
showed GPU T4 x2 disabled pending phone verification. No Kaggle training was
submitted. Preserve its logs and probe outputs as historical evidence. The
[Kaggle guide](../docs/kaggle_setup.md) and
[Kaggle-era handoff](../handoffs/phase_04_training_resume_handoff.md) are
archival records, not current run instructions.
