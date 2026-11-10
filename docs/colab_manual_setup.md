# Google Colab training workflow

**Status: Phase 4 training completed on Colab (10 November 2026).** Colab was
selected after Kaggle's GPU accelerator was unavailable to this account.
The laptop remains the development, artifact-verification, and final-app
environment. Kaggle CLI attempts and probe outputs are retained as historical
evidence; do not use them to infer that Kaggle training ran. The Phase 4
training contract and acceptance criteria remain in
[`phase_04_training_resume_handoff.md`](../handoffs/phase_04_training_resume_handoff.md).

This workflow uses the repository's existing
[`notebooks/colab_runner.ipynb`](../notebooks/colab_runner.ipynb), which
contains environment checks, data regeneration and hash verification, the
test suite, model/overfit/resume gates, full training, and verified export.
Colab GPU availability and session duration are not guaranteed. Require a real
CUDA operation before starting GPU-dependent cells; stop if the runtime is
CPU-only. Do not retry full FP16 training on the laptop or switch silently to
4-bit/QLoRA.

## 1. Prepare a source archive on D:

From `D:\Gen_AI_Assignment_2`, create a uniquely named archive so existing
source evidence is preserved:

```powershell
.\.venv\Scripts\python.exe scripts\package_colab_source.py `
  --output artifacts\colab_phase04_source_20261110.zip
```

This produces the ZIP and `colab_phase04_source_20261110.zip.sha256`. The
archive is source-only: it excludes raw Parquet, prepared patient JSONL, model
weights, caches, and credentials. Upload both files to
`MyDrive/GenAI_Assignment_2_source/`, naming them `colab_source.zip` and
`colab_source.zip.sha256` so the notebook's expected paths match. Upload the
current `notebooks/colab_runner.ipynb` to Google Drive or open it in Colab with
**File > Upload notebook**.

## 2. Start and verify the Colab runtime

1. Open the notebook while signed into the Google account with the source and
   output Drive folders.
2. Choose **Runtime > Change runtime type**, select **GPU** if available, and
   connect. A T4 is not guaranteed.
3. Run the bootstrap cell. Review the Drive authorization request, then let it
   verify the source ZIP against its SHA-256 sidecar and extract the versioned
   source copy.
4. Run the environment/data bootstrap cell. Confirm `cuda_available: true`,
   GPU name, VRAM, Python/PyTorch versions, and adequate runtime disk. If CUDA
   is false, stop before GPU-dependent work.
5. Run cells in notebook order. Keep Colab's preinstalled PyTorch build; the
   dependency step installs the pinned project packages without replacing
   Torch. If Colab asks for a runtime restart, restart and rerun cells from the
   top.

## 3. Run gates and full Phase 4 training

The completed run regenerated the pinned dataset and verified the canonical
split manifest, ran the full test suite, and passed the verification and
overfit/resume gates before training. It used the unchanged
configuration in `configs/train.yaml`: 602 training rows, validation after
each epoch on 206 rows, up to 3 epochs, and selection by validation macro-F1,
then evidence micro-F1, then validation loss. The sealed 206-row test split is
not loaded or inspected.

The run published verified checkpoints under
`MyDrive/GenAI_Assignment_2_checkpoints/phase04_lora_colab_cli_20261110T1650Z/`.
Its selected export is validation epoch 3; see the completed run record for
metrics, checksums, and gate results. For future runs, resume only from a
downloaded or Drive-persisted checkpoint bundle after verifying its manifest,
hashes, and run identity. Do not assume notebook output is a checkpoint.

## 4. Download and verify the result

The final notebook cells verify and restore the selected export, then write a
ZIP and SHA-256 sidecar to the Drive source/output folder. Download both files
to a new directory under `D:\Gen_AI_Assignment_2\artifacts\phase04\`. Check
the sidecar hash, extract to a fresh folder, and verify the bundle with
`src.clinical_match.checkpointing.verify_bundle` as described in the Phase 4
handoff. Require `selection_source == "best_validation"`, a selected epoch,
all three epoch validation records, matching adapter hash, and a complete
bundle manifest.

The downloaded bundle passed local `verify_bundle`. Fresh Colab and local
processes each loaded the export and scored all 206 validation rows. The local
RTX 3050 evaluation used the canonical config and completed in 212.26 seconds;
accuracy and classification macro-F1 match Colab, while evidence micro-F1
differs by 0.001342. The old local mismatch came from an invalid preflight:
raw argmax omitted Qwen's configured `repetition_penalty=1.1`. The uncached
reference now recomputes with `generate(use_cache=False)` and matches cached
generation on all three checked examples. This evaluator-only correction was
made after training; it did not alter the selected weights, cached generation
path, training source archive, or cached validation metrics. The previous
native exit `3221225477` did not recur.

See the [Phase 4 Colab handoff](../handoffs/phase_04_colab_training_handoff.md)
for run metrics and the local comparison. Do not open test data until Phase 9's
frozen manifest and correctness gates are complete.

## 5. Security, storage, and failure handling

- Never place Google, Kaggle, or Hugging Face credentials in the repository,
  notebook source, logs, handoff, or chat. The current public pinned model and
  dataset downloads do not require a token.
- The notebook requests Drive mount access to read the source archive and
  persist verified checkpoints/exports. Review the notebook and requested
  scopes before granting access.
- Colab `/content` is temporary. Drive persistence can be slower or quota
  limited; verify each copied bundle before pruning any checkpoint.
- Run only one GPU workload at a time. Record the actual GPU/session,
  environment report, source hash, run name, start/end times, gates, epoch
  metrics, selection, and transfer hashes.
- If Colab provides no GPU, preserve the environment report and stop. Do not
  train on CPU or silently change precision/model. Continue only after a
  verified GPU runtime is available.
- If the UI says **"No backend with GPU available"**, decline the offered
  no-accelerator runtime and stop the allocation attempt. This is a provider
  capacity result, not a project gate failure. Do not launch GPU-dependent
  cells or accept a CPU run; retry the GPU allocation later and verify CUDA
  before continuing. The CLI showing no active session is consistent with
  this failure and does not mean training is running in the background.
