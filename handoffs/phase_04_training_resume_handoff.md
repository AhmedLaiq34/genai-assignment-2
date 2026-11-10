> **Status update (11 November 2026):** Training is complete through the Google Colab route. The validation-selected epoch 3 adapter was downloaded and verified locally; adapted screening, decoder correctness, the frozen evaluation, and the IEEE report have since been completed. This file records the earlier continuation plan and is historical. For current artifacts, see [Phase 4 Colab training handoff](phase_04_colab_training_handoff.md), [Phase 5](phase_05.md), [Phase 6](phase_06.md), [Phase 8](phase_08.md), and [submission guide](../SUBMISSION.md). No new training run is required.

# Historical Phase 4 Kaggle training handoff

> **Archived 10 November 2026:** the user switched the active training route
> to Google Colab after Kaggle's GPU accelerator was disabled and the corrected
> probe reported CPU-only PyTorch. No Kaggle training ran. Use the current
> [Colab training handoff](phase_04_colab_training_handoff.md) and
> [Colab setup guide](../docs/colab_manual_setup.md). The details below remain
> as provenance and must not be followed as current run instructions.

**Date:** 10 November 2026  
**Purpose:** hand off the training work to another agent so it can obtain the
real validation-selected LoRA adapter, verify it locally, and unblock Phases
5-9.  
**Current state:** Phase 4 code and local plumbing gates are complete. The full
FP16 local run did **not** complete. No validation-selected adapter exists yet.
Phases 5-8 are implemented against the unchanged base model; adapter-specific
reruns are pending.

This document is the operational handoff for the next training agent. Read it
before running anything. It supersedes the operational details in
`handoffs/phase_04.md` where they differ.

## 1. Immediate objective

Run full LoRA training on Kaggle with the official Kaggle CLI using the
approved 602-row TrialGPT training split, validate after each epoch on the
206-row validation split, select the best checkpoint by validation metrics,
export a verified adapter bundle, download it to this laptop, verify its
hashes, reload it in a fresh local process, and compare it with the unchanged
base model on the same validation split.

Do **not** start Phase 9 final test evaluation. The protected test split stays
sealed until the adapter is selected and a complete frozen manifest exists.

## 2. Non-negotiable constraints

1. **Use D: for local work and caches.** Do not fill C: or delete unrelated
   files. Local cache roots are under `D:\Gen_AI_Assignment_2\.cache\`.
2. **Do not retry full FP16 training on the laptop.** The 6 GiB RTX 3050
   Laptop GPU failed with memory exhaustion. Local smoke, overfit, and resume
   tests already passed; full training belongs on Kaggle.
3. **Do not switch to 4-bit/QLoRA silently.** No 4-bit path is implemented in
   the current trainer. A 4-bit run is a new experiment with different
   numerical behavior and requires an explicit user/decision-log choice,
   config support, and a matching evaluation identity.
4. **Use Kaggle through the official Kaggle CLI.** The user supplies the Kaggle
   username/API key through a secure environment or private `kaggle.json`.
   Never put a token in the repository, notebook, kernel source, prompt, logs,
   handoff, or chat. The active setup guide is
   [`docs/kaggle_setup.md`](../docs/kaggle_setup.md). Manual Colab is historical
   fallback only.
5. **Train only on the 602 approved training rows and validate only on the 206
   validation rows.** The test split is 206 rows across 8 patients and is
   sealed. Do not open it, inspect it, or use it for selection.
6. **Do not change the frozen inference contract during training.** Keep the
   `v1-provisional` prompt, pinned model revisions, prompt/output budgets,
   parser, and greedy generation settings. If any of these change, the baseline
   and adapter comparison must be rebuilt.
7. **Do not call a partial adapter a trained model.** The local
   `best_adapter` directory is not a completed Phase 4 artifact.
8. **Do not commit adapter weights or large artifacts to Git.** Keep them under
   `artifacts/` or another ignored local path.
9. **Run GPU jobs sequentially.** Do not run validation, benchmarking, or both
   models concurrently on the laptop or on Kaggle.
10. **Preserve provenance.** Save source hashes, configs, environment reports,
    logs, selection evidence, and checksums with every accepted artifact.

## 3. What is already complete

### Phases 0-3

- Environment, model selection, tokenizer checks, data preparation, split
  manifest, baseline harness, parser, metrics, and unchanged-model baseline
  are complete.
- Dataset: `ncbi/TrialGPT-Criterion-Annotations`, revision
  `1cfcafde94a1560a33b4addc1638664fe28fc059`, source SHA-256
  `14c547184fac9b013cd17441fd028845b40150c76b022e58334f5494077e5234`.
- Split: 37/8/8 patients, seed 42, 602 train rows, 206 validation rows, 206
  test rows. Training annotation `883` is explicitly quarantined because its
  criterion text is missing.
- Split manifest SHA-256:
  `309ea164a836f9a3f6d6c315578be990370971990412bbb75ef77cde707e063b`.
- Target: `Qwen/Qwen2.5-1.5B-Instruct`, revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Draft: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Baseline validation metrics
  (`results/baseline/base_greedy_v1_validation/metrics.json`):
  - accuracy `0.13106796116504854`
  - classification macro-F1 `0.12405091068685538`
  - evidence micro-F1 `0.15492957746478872`
  - 4 invalid predictions
  - generation backend `transformers_generate_greedy_cache`
  - 206/206 validation rows, test not opened
- Baseline was produced on the local RTX 3050 with FP16. Use the local baseline
  for the fair local before/after comparison.

### Phase 4 implementation and local gates

Implemented files:

- `src/clinical_match/training/data.py`: exact prompt rendering,
  assistant-only labels, template/EOS validation, deterministic epoch order,
  padding.
- `src/clinical_match/training/trainer.py`: LoRA wrapping, AdamW/scheduler/
  scaler, gradient accumulation, gradient checkpointing, validation, selection,
  checkpointing, export.
- `src/clinical_match/training/checkpoints.py`: verified bundles,
  optimizer/scheduler/scaler/RNG state, resume restoration, persistent
  publishing, pruning.
- `src/clinical_match/training/validation.py`: full 206-row validation
  generation and scoring.
- `scripts/train.py`: training entry point, smoke/overfit/resume/stop options.
- `scripts/rehearse_resume.py`: uninterrupted control versus
  interrupted/resumed comparison.
- `scripts/evaluate.py`: base or adapter evaluation on train/validation only;
  refuses test.
- `scripts/prepare_kaggle_training_job.py`: builds the active private Kaggle
  Phase 4 training kernel from the current source archive.
- `docs/kaggle_setup.md`: active Kaggle CLI credential, probe, submission,
  monitoring, download, and verification guide.

Local evidence that passed:

| Gate | Evidence | Result |
|---|---|---|
| Automated tests | current full suite | 75 passed, 1 warning, 111 subtests passed |
| Real target smoke | `results/training/phase04_real_smoke/` | finite loss `0.5740`, grad norm `1.494`, ~4.07 s/step |
| Real target overfit | `results/training/phase04_real_overfit/overfit_report.json` | 12 examples, 40 steps, loss `0.4534 -> 0.1174`, exact target matches `1 -> 5`, finite |
| Real resume rehearsal | `results/training/resume_rehearsal.json` | step/epoch/cursor/order identical; max loss delta `0.0` |
| Full local attempt | `results/training/phase04_local_gate.json` | blocked: OOM; not accepted |

The trainer has not been changed since these local tests except for unrelated
downstream Phase 7/8 work. Re-run the suite before a new training run.

### Phases 5-8 status

- Phase 5 screening service and evidence/error pipeline are implemented and
  locally tested against the base model. The development error ledger is
  base-model evidence only and must be regenerated with the adapter.
- Phase 6 explicit cached/uncached decoding is implemented and exactly
  token-equivalent on the base model. Its scripts currently load the base model
  only; adapter support must be added for an adapter rerun.
- Phase 7 tiled attention is standalone and adapter-independent. It is an
  educational comparison, not integrated into the Qwen model.
- Phase 8 deterministic greedy speculative decoding is implemented and exactly
  token-equivalent on the base model. Its scripts currently load the base target
  only; adapter support must be added for an adapter rerun.
- Phase 9 is blocked until the selected adapter and frozen manifest exist.

## 4. Training contract

Current `configs/train.yaml`:

- LoRA rank `8`, alpha `16`, dropout `0.05`, target modules `q_proj` and
  `v_proj`.
- AdamW, learning rate `2e-4`, weight decay `0.01`, betas `[0.9, 0.999]`.
- Linear schedule, warmup ratio `0.05`, gradient norm cap `1.0`.
- Microbatch `1`, gradient accumulation `8`, maximum `3` epochs, seed `42`.
- FP16, SDPA attention, gradient checkpointing, training KV cache off.
- `max_total_tokens=1024`, `max_new_tokens=96`, `max_seq_len=1024`.
- Validation after each epoch, same prompt/parser/greedy settings as the
  baseline.
- Checkpoint every 25 optimizer steps or 300 seconds, plus every epoch; keep
  two recent resume bundles.
- Selection: classification macro-F1, then evidence micro-F1, then lower
  validation loss.
- Export: `best_adapter` copied to `selected_adapter`, plus
  `selected_manifest.json`, configs, environment, data hashes, history, log,
  and a verified export bundle.

Step arithmetic with 602 examples and accumulation 8:

- `ceil(602 / 8) = 76` optimizer steps per epoch.
- 3 epochs = 228 planned optimizer steps.
- Warmup = `int(228 * 0.05) = 11` steps.
- At the local measured ~3.7 s/step, training is about 5 minutes per epoch;
  full 206-row validation was about 6 minutes locally. Budget roughly 30-45
  minutes for the full run on a comparable GPU, but measure on the actual
  Kaggle GPU and do not rely on this estimate.

### Important implementation behavior

- Only assistant answer tokens are supervised; prompt and padding labels are
  `-100`.
- The assistant answer includes the template end-of-turn token.
- Training data order is deterministic per epoch:
  `random.Random(seed + epoch).shuffle`.
- Checkpoints include adapter, optimizer, scheduler, scaler, Python/NumPy/
  PyTorch RNG, epoch/global step, example cursor, data order, resolved config,
  environment, and input hashes.
- `resolve_resume_dir` verifies a bundle and restores it before resuming.
- `publish_bundle` refuses to overwrite an existing persistent bundle of the
  same name. Use a unique run name on a fresh Kaggle run.
- The trainer validates after each epoch and writes
  `epoch_N/validation/metrics.json`.
- The final export only exists after the run finalizes. A process that dies
  after a checkpoint does not produce `export/`.
- If validation is skipped, `_finalize` falls back to
  `last_state_no_validation_selection`; this is **not** a valid Phase 4
  selection and must not be accepted.

## 5. Existing local partial run: read this before touching artifacts

`results/training/phase04_lora_r8/` contains diagnostic evidence from the failed
local run. It is **not** the selected adapter.

Facts:

- The run reached optimizer step 115, which is mid-epoch 2.
- Epoch 1 completed at step 76 and validation metrics **do** exist:
  - `results/training/phase04_lora_r8/epoch_1/validation/metrics.json`
  - accuracy `0.7184466019417476`
  - macro-F1 `0.3457388939998816`
  - evidence micro-F1 `0.27993779160186627`
  - validation loss `0.15450268192668853`
  - adapter hash
    `56411c794a94efebcbfa4a413b44b7f35854db77d731697625f8bea06b0f217a`
- An earlier version of `handoffs/phase_04.md` said no epoch-1 metrics were
  saved. That was incorrect; the metrics file exists and the old handoff has
  been corrected.
- `best_adapter/` is a copy of that epoch-1 adapter and is not a finished run
  selection.
- The local run log contains duplicated step 77-115 records from a retry; do
  not treat it as a clean resume lineage.
- Peak allocated memory was `4,498,746,880` bytes (about `4.19 GiB`); peak
  reserved was `8,873,050,112` bytes (about `8.26 GiB`) on a 6 GiB GPU. The run
  ended in OOM.
- Do not use `phase04_lora_r8/best_adapter`, `epoch_1/adapter`, or the local
  partial checkpoints as the Phase 4 deliverable.
- Do not resume the local partial run on Kaggle unless the user explicitly
  requests a cross-device resumed experiment. A clean Kaggle run is the default
  and gives a complete 3-epoch history and unambiguous selection evidence.

## 6. Pre-flight on the laptop

Run from `D:\Gen_AI_Assignment_2`.

1. Inspect the working tree and preserve all current changes. Do not use
   `git clean`, `git reset --hard`, or delete checkpoints/artifacts.
2. Run the full test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: 75 passed, 1 PEFT warning, 111 subtests passed. If this changes,
understand why before training.

3. Confirm Kaggle CLI authentication without printing the token:

```powershell
kaggle --version
kaggle kernels list -m
```

4. Run the private GPU probe before training. See
   [`docs/kaggle_setup.md`](../docs/kaggle_setup.md). Require
   `cuda_available: true` and a finite CUDA operation. Do not train on CPU.

5. Rebuild the source archive and prepare the training job:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_kaggle_training_job.py `
  --username <username> `
  --run-name phase04_lora_r8_kaggle_<YYYYMMDD> `
  --kernel-slug trialgpt-phase-4-<YYYYMMDD>
```

This writes `artifacts\kaggle_phase04_job\` with `source.zip`,
`source.zip.sha256`, `train_phase04.py`, and `kernel-metadata.json`. Record the
printed source SHA-256 and the kernel slug. Do not put credentials in any of
these files.

6. Inspect the generated job locally:

```powershell
Get-Content artifacts\kaggle_phase04_job\kernel-metadata.json
Get-Content artifacts\kaggle_phase04_job\source.zip.sha256
```

## 7. Submit and monitor the Kaggle job

```powershell
kaggle kernels push -p artifacts\kaggle_phase04_job
kaggle kernels status <username>/trialgpt-phase-4-<YYYYMMDD>
```

Run only one GPU job at a time. The generated kernel performs the data
regeneration, full test suite, CUDA check, overfit gate, resume rehearsal, and
full training in order. It writes evidence under `/kaggle/working/output_bundle`
and keeps the project/model cache in Kaggle scratch storage.

Use the Kaggle UI for live logs if needed. Do not invent a streaming log command
or assume a queued kernel is running. Record the kernel URL/version, actual GPU
name/VRAM, CLI version, source SHA-256, run name, and start/end times.

If the job fails:

- preserve `output_bundle/ERROR.txt` and the kernel page;
- do not repeatedly resubmit during a service incident;
- download any retained checkpoint/export before starting another attempt;
- if no usable checkpoint survived, use a fresh run name and restart cleanly;
- if a checkpoint survived, resume only after a documented, hash-verified
  transfer and identity check.

## 8. Download the output

```powershell
$out = "artifacts\kaggle_phase04_output"
if (Test-Path -LiteralPath $out) { throw "Refusing to overwrite existing Kaggle output evidence: $out" }
kaggle kernels output <username>/trialgpt-phase-4-<YYYYMMDD> -p $out
```

Inspect:

```text
artifacts\kaggle_phase04_output\output_bundle\
  SUCCESS.txt or ERROR.txt
  environment.json
  resume_rehearsal.json
  phase02_runtime_handoff_kaggle.json
  tokenization_audit.json
  source_sha256.txt
  results\training\<run-name>\
  persistent\checkpoints\<run-name>\
  persistent\exports\<run-name>\phase04-export-*\
  phase04-<run-name>.zip
  phase04-<run-name>.zip.sha256
```

Do not treat kernel output text as the artifact. The selected adapter,
manifest, logs, and checksums must be downloaded and verified locally.

## 9. Verify and extract the export

Download the output before ending/ignoring the Kaggle job. On the laptop:

```powershell
$zip = "D:\Gen_AI_Assignment_2\artifacts\kaggle_phase04_output\output_bundle\phase04-<RUN_NAME>.zip"
Get-FileHash -Algorithm SHA256 -LiteralPath $zip
Get-Content "D:\Gen_AI_Assignment_2\artifacts\kaggle_phase04_output\output_bundle\phase04-<RUN_NAME>.zip.sha256"
```

The two SHA-256 values must match. Extract to a new directory:

```powershell
$dest = "D:\Gen_AI_Assignment_2\artifacts\phase04\<RUN_NAME>"
if (Test-Path -LiteralPath $dest) { throw "Destination already exists: $dest" }
Expand-Archive -LiteralPath $zip -DestinationPath $dest
```

The extracted folder should contain `COMPLETE.json`, `bundle_manifest.json`,
`selected_adapter/`, `selected_manifest.json`, and the exported logs/configs.
Verify the bundle:

```powershell
$env:PYTHONIOENCODING='utf-8'
& .\.venv\Scripts\python.exe -c "from pathlib import Path; from src.clinical_match.checkpointing import verify_bundle; b=Path(r'artifacts\phase04\<RUN_NAME>'); m=verify_bundle(b); print(m['bundle_name'], len(m['files']))"
```

Inspect the selection manifest before evaluating:

```powershell
$env:PYTHONIOENCODING='utf-8'
& .\.venv\Scripts\python.exe -c "import json; from pathlib import Path; p=Path(r'artifacts\phase04\<RUN_NAME>/selected_manifest.json'); m=json.loads(p.read_text()); print(json.dumps({k:m.get(k) for k in ['run_name','selection_source','selected','global_step','epochs_completed','model_id','model_revision','prompt_revision','selected_adapter_sha256']}, indent=2))"
```

Acceptance requirements:

- `selection_source == "best_validation"`, not
  `last_state_no_validation_selection`.
- `selected` is present and names an epoch.
- `selected_adapter_sha256` matches the extracted `selected_adapter/`.
- `epochs_completed == 3` for the canonical run, or an explicitly documented
  reason for stopping earlier.
- The manifest identifies the pinned target, prompt revision, data hashes, and
  environment.
- All bundle files verify.
- The local adapter loads in a fresh process.

If `selection_source` is `last_state_no_validation_selection`, stop. That
artifact is not a valid Phase 4 selection.

## 10. Local adapter reload and fair comparison

The current `scripts/evaluate.py` uses `--adapter`, not the older
`--model adapted` form.

Run the full 206-row validation evaluation with the local pinned base model plus
the extracted adapter:

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
  --adapter "artifacts\phase04\<RUN_NAME>\selected_adapter" `
  --output-dir "results/training/<RUN_NAME>/local_validation"
```

This writes:

```text
results/training/<RUN_NAME>/local_validation/
  validation_predictions.jsonl
  metrics.json
  run_config.json
```

Use the local run for the fair deployment comparison because the Phase 3
baseline was produced on the same local RTX 3050 with the same prompt and
generation backend. Do not compare a Kaggle adapter metric to a local base
metric and call the difference a speed or quality result. Keep cloud selection
metrics and local reload metrics separate.

Compare these fields with
`results/baseline/base_greedy_v1_validation/metrics.json`:

| Field | Base | Adapter | Delta |
|---|---:|---:|---:|
| classification accuracy | 0.1311 | record | record |
| classification macro-F1 | 0.1241 | record | record |
| evidence micro-F1 | 0.1549 | record | record |
| invalid predictions | 4 | record | record |
| parse failures | record | record | record |
| output-cap truncations | 0 | record | record |

Also check:

- `scope.split == "validation"`, `scope.n == 206`,
  `scope.test_data_opened == false`.
- `artifacts.adapter_sha256` matches the selected adapter hash.
- `artifacts.prompt_config_sha256`, `split_manifest_sha256`,
  `input_jsonl_sha256`, and model revision match the baseline.
- Predictions are scored from all 206 rows; no filtering of hard cases.
- A regression is a valid result. Do not hide it or substitute the base model.

If local reload produces category/parse behavior that differs materially from
the Kaggle validation metrics, investigate before proceeding. Exact token
sequences can differ across devices; parsed-output and metric drift must be
explained. Never reselect the epoch using test data.

## 11. When training is accepted

A Phase 4 acceptance record must include:

- Full Kaggle command, kernel slug/version, run name, actual GPU/session,
  start/end time, and software versions.
- Complete logs and `training_history.json`.
- Per-epoch validation metrics and selected epoch.
- Verified resume bundles and evidence that at least one checkpoint was
  published and restored.
- Export bundle and SHA-256.
- Download transfer hash.
- Local fresh-process adapter reload command and result.
- Base-versus-adapter validation comparison with absolute metrics and deltas.
- Explicit statement that the test split was not opened.
- Updated `handoffs/phase_04.md`, `research/decision_log.md`, and
  `output/deliverables_checklist.md`.

Recommended canonical local adapter location after verification:

```text
D:\Gen_AI_Assignment_2\artifacts\adapters\phase04\<run_name>\
  selected_adapter/
  selected_manifest.json
  metrics.json
  environment.json
  SHA256SUMS.txt
```

Keep the original downloaded bundle unchanged as the provenance copy. Do not
commit adapter weights.

## 12. Downstream work after the adapter exists

Do not start these until Phase 4 is accepted.

1. **Phase 5:** rerun screening and rebuild `results/errors/development.md` with
   the adapter. Use `scripts/screen_example.py --adapter <path>` and preserve
   the same policy/schema. The old error ledger is base-model evidence only.
2. **Phase 6:** add adapter support to
   `scripts/check_decoding_correctness.py` and
   `scripts/benchmark_decoding.py`; they currently call
   `load_eval_model(cfg, device)` without an adapter. Rerun exact
   cached/uncached equality and benchmarks. Do not change the decoder contract.
3. **Phase 7:** attention is adapter-independent. No adapter rerun is required;
   keep it standalone and continue to report the honest slowdown.
4. **Phase 8:** add adapter support to the target load in
   `scripts/check_speculative_correctness.py` and
   `scripts/benchmark_speculative.py`; the draft remains the pinned 0.5B model.
   Rerun correctness and the k sweep with the selected target adapter. Keep
   draft/target tokenizer compatibility checks.
5. **Phase 9:** only after the adapter, all correctness tests, and the frozen
   manifest exist. Do not open the sealed test split before that point.

## 13. Known traps and stale documentation

- An earlier revision of `handoffs/phase_04.md` said no epoch-1 validation
  metrics were saved. The metrics exist and are diagnostic only; the current
  file is corrected.
- Older notes use `--model adapted`; the implemented evaluator uses
  `--adapter <path>`.
- `output/phase_4_6_implementation_plan.md` and parts of `output/master_plan.md`
  were written before the trainer existed. Treat this handoff and the code as
  current; update those documents when they conflict.
- The local `phase04_lora_r8/best_adapter` is not a final selection.
- The generated Kaggle kernel uses an explicit `--run-name`; record it. A
  retry with a new name creates a new output tree.
- The source archive excludes prepared JSONL and raw Parquet. Kaggle must
  regenerate data from the pinned source; do not bypass its hash checks.
- Kaggle scratch storage is not the same as downloaded evidence. Download and
  verify `output_bundle`; kernel output text alone is not an artifact.
- `publish_bundle` refuses to overwrite a same-name persistent bundle. Use a
  unique run name or resume the original run.
- `scripts/evaluate.py` intentionally rejects any split other than `train` or
  `validation`.
- The local `train_log.jsonl` contains duplicated step 77-115 records from a
  retry; do not use it as a clean resume sequence.
- The local partial run peaked at `4,498,746,880` bytes allocated (about
  `4.19 GiB`) and `8,873,050,112` bytes reserved (about `8.26 GiB`) on a 6 GiB
  GPU. The Kaggle run should be monitored, but do not copy the local OOM
  workaround into config without documenting it.
- `best_adapter`/`selected_adapter` are PEFT adapter directories, not full model
  weights. Always load with the pinned base model revision.
- Do not use `--skip-validation` for the accepted full run.
- The Kaggle training kernel is a private job and must never contain the
  Kaggle API key or the Hugging Face token.

## 14. Quick checklist for the training agent

- [ ] Full local test suite passes.
- [ ] Kaggle CLI authenticates without printing the token.
- [ ] Private GPU probe passes with CUDA available.
- [ ] Kaggle training job prepared with a unique run name and kernel slug.
- [ ] Job submitted and monitored; only one GPU job at a time.
- [ ] Data regenerated and manifest hash verified.
- [ ] Real overfit gate passes.
- [ ] Interrupted/resumed rehearsal passes.
- [ ] Full 3-epoch run finalizes and validation metrics exist for each completed
      epoch.
- [ ] Selection is `best_validation`; selected epoch and metrics are recorded.
- [ ] Verified export bundle exists in Kaggle output and downloaded to
      `artifacts/phase04/`.
- [ ] ZIP SHA-256 and bundle inventory verified locally.
- [ ] Adapter reloads in a fresh local process.
- [ ] Local base-versus-adapter validation comparison saved.
- [ ] Test split remains unopened.
- [ ] Phase 4 handoff, decision log, and checklist updated.
