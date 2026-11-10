# Historical Kaggle CLI setup attempt

**Status: historical; superseded 10 November 2026.** The user has switched
Phase 4 training to Google Colab after Kaggle's GPU accelerator was disabled
for this account and a GPU-enabled probe received CPU-only PyTorch. No Kaggle
training ran. Use the [active Colab training guide](colab_manual_setup.md) and
[Phase 4 Colab handoff](../handoffs/phase_04_colab_training_handoff.md).
The remaining sections preserve the Kaggle attempt's setup and evidence only.

The earlier Kaggle attempt on 9 November was a Phase 2/GPU-probe attempt during
a service incident. It is not evidence that training ran, and it does not
justify retrying the old CPU probes. This page replaces the archived setup
note. The active implementation uses a private Kaggle kernel prepared from the
current source archive.

## 1. Credentials and access

The user supplies the Kaggle username and API key when needed. The preferred
setup is environment variables in the shell that runs the agent:

```powershell
$env:KAGGLE_USERNAME = "<username>"
$env:KAGGLE_KEY = "<api-key>"
```

The alternative is the standard private file
`%USERPROFILE%\.kaggle\kaggle.json` containing the username and key. Do not:
- commit `kaggle.json` or any token;
- paste the key into a notebook, kernel source, log, handoff, prompt, or chat;
- put credentials in the source archive or Kaggle kernel metadata;
- print environment variables while debugging.

Use a least-privilege/revocable API token. If a token is exposed, revoke and
replace it. The Kaggle kernel itself needs no Kaggle credentials; it uses the
public pinned Hugging Face models and dataset and writes results under
`/kaggle/working/output_bundle`.

Verify authentication without echoing the key:

```powershell
kaggle --version
kaggle kernels list -m
```

Record the CLI version. A successful private-kernel list confirms that the CLI
can authenticate. Do not claim GPU access from authentication alone.

## 2. GPU allocation probe

Before a training submission, run the small private GPU probe from
`scripts/prepare_kaggle_gpu_probe.py`:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_kaggle_gpu_probe.py --username <username>
kaggle kernels push -p artifacts\kaggle_gpu_probe
kaggle kernels status <username>/trialgpt-gpu-probe
kaggle kernels output <username>/trialgpt-gpu-probe -p artifacts\kaggle_gpu_probe_pulled
```

Accept the probe only when the downloaded `gpu_probe.json` has
`"passed": true`, `cuda_available: true`, a real GPU name, and a finite CUDA
matrix multiplication. If Kaggle returns a CPU-only PyTorch build or no device,
stop; do not train on CPU and do not silently reinstall a different framework.
Record the actual GPU/VRAM in the run evidence.

The old probe result in `results/kaggle_gpu_probe_v1.json` is historical and
failed (`torch` CPU-only). A new probe is required for the active route.

## 3. Prepare the Phase 4 training job

Rebuild the source archive and generate a private Kaggle training kernel:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_kaggle_training_job.py `
  --username <username> `
  --run-name phase04_lora_r8_kaggle_<YYYYMMDD> `
  --kernel-slug trialgpt-phase-4-<YYYYMMDD>
```

This writes `artifacts\kaggle_phase04_job\` with:
- `source.zip` and `source.zip.sha256`;
- the self-contained `train_phase04.py` kernel;
- `kernel-metadata.json` with `enable_gpu: true`, `enable_internet: true`, and
  no credentials or private dataset sources.

The generated kernel:
1. extracts the hash-verified source into scratch storage;
2. keeps Hugging Face/model cache outside the persisted output;
3. regenerates the pinned dataset and verifies the canonical manifest;
4. runs the full local test suite;
5. requires `torch.cuda.is_available()`;
6. runs the real-model overfit gate and interrupted/resumed rehearsal;
7. runs full 3-epoch LoRA training with validation selection and checkpoints;
8. writes results, selected export, checksums, and a download ZIP under
   `/kaggle/working/output_bundle`.

The run name must be unique and recorded. Do not reuse a name whose output or
resume bundles already exist. A clean run is the default; do not resume the
failed local partial run across devices unless the user explicitly asks for
that separate experiment.

## 4. Submit, monitor, and download

From the repository root:

```powershell
kaggle kernels push -p artifacts\kaggle_phase04_job
kaggle kernels status <username>/trialgpt-phase-4-<YYYYMMDD>
```

`kaggle kernels push` submits/updates the private kernel. Use
`kaggle kernels status` to check the state; do not start another GPU job while
one is running. For live logs/open the kernel page in the Kaggle UI if needed.
Do not invent a live log command or assume that a queued job is running.

When the status is complete, download the output:

```powershell
kaggle kernels output <username>/trialgpt-phase-4-<YYYYMMDD> `
  -p artifacts\kaggle_phase04_output
```

Then inspect:
- `output_bundle\SUCCESS.txt` or `output_bundle\ERROR.txt`;
- `output_bundle\resume_rehearsal.json` (must have `passed: true`);
- `output_bundle\results\training\<run-name>\run_summary.json`;
- per-epoch `validation/metrics.json` and `training_history.json`;
- `output_bundle\persistent\exports\<run-name>\phase04-export-*\`
  with `COMPLETE.json` and `bundle_manifest.json`;
- `output_bundle\phase04-<run-name>.zip` and `.sha256`.

Do not treat notebook/kernel output text as the artifact. The selected adapter,
manifest, logs, and checksums must be downloaded and verified locally.

If a job fails:
- read `ERROR.txt` and the kernel page;
- preserve the failed job/output as evidence;
- do not repeatedly resubmit during a Kaggle incident;
- if no usable checkpoint survives, use a fresh run name and restart cleanly;
- if a checkpoint does survive, resume only after documenting the Kaggle
  dataset/output transfer and verifying the bundle identity.

## 5. Local verification and adapter evaluation

Verify the downloaded ZIP against its sidecar, extract to a new directory, and
verify the bundle with `src.clinical_match.checkpointing.verify_bundle`. Check
`selected_manifest.json` for:

- `selection_source == "best_validation"`;
- a selected epoch and selected adapter hash;
- `epochs_completed == 3` for the canonical run (or a documented early-stop);
- pinned target revision, prompt revision, data hashes, and environment.

Then run the local 206-row validation comparison with the pinned base model:

```powershell
$env:HF_HOME='D:\Gen_AI_Assignment_2\.cache\huggingface'
$env:HUGGINGFACE_HUB_CACHE='D:\Gen_AI_Assignment_2\.cache\huggingface\hub'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:TEMP='D:\Gen_AI_Assignment_2\.cache\tmp'
$env:TMP='D:\Gen_AI_Assignment_2\.cache\tmp'
$env:PYTHONIOENCODING='utf-8'

& .\.venv\Scripts\python.exe scripts\evaluate.py `
  --config configs/eval.yaml `
  --split validation `
  --adapter "artifacts\phase04\<run-name>\selected_adapter" `
  --output-dir "results/training\<run-name>\local_validation"
```

The local base-model baseline was produced on the RTX 3050, so the local
adapter reload is the fair delivery comparison. Keep cloud selection metrics
and local reload metrics separate. Do not mix Kaggle and laptop timings as a
speedup. Do not open the sealed test split before Phase 9's frozen manifest.

## 6. Data, compute, and retention rules

- The source archive excludes raw Parquet, prepared JSONL, model weights, and
  caches. The Kaggle kernel must download/regenerate the pinned data and verify
  its SHA-256; do not substitute local prepared files.
- Use `/kaggle/temp` (or another scratch path) for the extracted project and
  model cache so `/kaggle/working/output_bundle` contains only logs, metrics,
  checkpoints/export, and checksums.
- Run GPU jobs sequentially. Do not run training, validation, or benchmarks
  concurrently.
- Keep model/adapter weights out of Git. Large artifacts belong under
  `artifacts/` or the downloaded Kaggle output directory.
- Never use the test split for checkpoint, prompt, threshold, or hyperparameter
  selection.
- Record the Kaggle username, kernel slug/version, GPU type/VRAM, CLI version,
  source SHA-256, run name, start/end time, and command in the acceptance
  record.

## 7. Historical record

The 9 November 2026 Kaggle attempt used a private CPU Phase 2 verification job
and a GPU probe. CLI authentication and a quota lookup succeeded, but the
execution showed a CPU-only PyTorch build and no CUDA device. Subsequent jobs
queued during a user-reported Kaggle startup/connectivity incident. Those
artifacts remain at:
- `results/kaggle_phase02_submission.json`;
- `results/kaggle_gpu_probe_v1.json`;
- `results/kaggle_resubmission_status.json`.

They are historical evidence only. No Kaggle model training or clinical result
exists yet.

