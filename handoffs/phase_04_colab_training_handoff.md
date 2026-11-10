# Phase 4 Colab training handoff

**Updated:** 10 November 2026 (Asia/Karachi)  
**Status:** Colab training and selection completed; a full local 206-row adapter evaluation now completes successfully. The generation-reference bug is fixed.
Kaggle's GPU accelerator was disabled for this account, and two corrected
GPU-probe attempts still received CPU-only PyTorch. No Kaggle training ran.
The user has switched the project back to Colab. See the
[Colab setup guide](../docs/colab_manual_setup.md) and
[Colab runner](../notebooks/colab_runner.ipynb).

**Completion update (10 November 2026):** Colab CLI allocated a Tesla T4 and
the configured FP16 LoRA run completed. All pretraining gates passed, three
epoch validations completed, the epoch-3 best-validation export was verified,
and the full validation split was evaluated from a fresh Colab CLI process.
The sealed test split remained unopened. Follow-up found that the uncached
reference used raw argmax and omitted Qwen's pinned `repetition_penalty=1.1`;
this made the cached-versus-reference comparison invalid. The reference now
uses `generate(use_cache=False)`, retaining the same generation processors
while recomputing the prefix. The preflight now matches exactly for all three
examples, and a fresh local RTX 3050 process completed all 206 validation rows
in 212.26 seconds without reproducing native status `3221225477`.
This was an evaluator-reference-only correction made after training. It did
not change the training source archive, selected weights, cached generation
path, training selection, or cached validation metrics.

## Completed run record

- Run: `phase04_lora_colab_cli_20261110T1650Z`; 3 epochs, 228/228 optimizer
  steps, 602 training rows, 206 validation rows; unchanged `configs/train.yaml`
  (LoRA r=8, alpha=16, dropout .05, q/v projections, AdamW 2e-4, microbatch 1,
  accumulation 8, FP16, 11 warmup steps).
- Source archive SHA-256:
  `ca38d541ab3e51ea27f8b224cb38ae5360d90165f81354866fde9ea97d3ce861`.
- Runtime: Tesla T4, 15 GiB; Python 3.13.16, PyTorch 2.14.1+cu130 / CUDA
  13.0, Transformers 4.57.1, PEFT 0.17.1. Environment report was recorded
  at 2026-11-10 16:46:45 UTC.
- Run window: CLI job submitted 2026-11-10 16:46:27 UTC; verified export
  completed 2026-11-10 17:03:24 UTC. See
  [`run_timing.txt`](../results/phase04/phase04_lora_colab_cli_20261110T1650Z_colab_fresh_eval/run_timing.txt).
- Gates: environment doctor/CUDA smoke passed; tokenizer compatibility and
  handoff verification passed; pytest 75 passed, 1 PEFT warning, 111 subtests
  passed; model pair and synthetic 10-step fit pilots passed; resume rehearsal
  passed (same sample order, max loss delta 0.0012943); 40-step real-model
  overfit smoke passed (loss 0.4532 to 0.1161, target matches 1 to 5).
- Epoch validation (all 206 rows):

  | Epoch | Validation loss | Accuracy | Macro-F1 | Evidence micro-F1 |
  |---:|---:|---:|---:|---:|
  | 1 | 0.154210 | 0.718447 | 0.345739 | 0.280374 |
  | 2 | 0.141998 | 0.728155 | 0.465736 | 0.307167 |
  | 3 | 0.135959 | 0.737864 | 0.472929 | 0.344961 |

- Selection: `best_validation`, epoch 3, 228 steps; adapter SHA-256
  `b823bc3f7e979be9fca7b2f9543305a5b5562721f4342af5487dc3d2941151fb`.
- Export ZIP SHA-256:
  `86d594f59453e45fa3f997c8b1a565c44860c65d58c8671734f112a1f658a41a`;
  sidecar matched, and local `verify_bundle` passed. Downloaded bundle:
  [`phase04_lora_colab_cli_20261110T1650Z_bundle`](../artifacts/phase04/phase04_lora_colab_cli_20261110T1650Z_bundle).
- Fresh Colab-process full validation: accuracy 0.737864, macro-F1 0.472929,
  evidence micro-F1 0.344961. The temporary evaluation copy disabled only the
  three-example cached-vs-uncached preflight after the Windows diagnostic
  failed; all generation settings/backend stayed unchanged. Its evaluation
  config hash is `2140094c342208bf0679c8d3ec1bb21c118a95ab56c2d9984b9263ec5e6a0416`
  rather than the canonical config hash
  `e5d169b68ce549d0ffc06892a3884d22403b323779c70345f76d49c0b576bb95` because
  of that disabled preflight. Result:
  [`metrics.json`](../results/phase04/phase04_lora_colab_cli_20261110T1650Z_colab_fresh_eval/metrics.json).
- Fresh local RTX 3050 full validation (canonical config and cached backend,
  206/206 rows): accuracy `0.737864`, macro-F1 `0.472929`, evidence micro-F1
  `0.346304`; elapsed generation time 212.26 seconds. Accuracy and
  classification macro-F1 match the Colab run exactly; evidence micro-F1
  differs by 0.001342. Compared with the same-device base validation, the
  local adapter deltas are +0.606796 accuracy, +0.348878 classification
  macro-F1, and +0.191374 evidence micro-F1. See
  [`local metrics`](../results/phase04/local_eval_fixed_full/metrics.json).
- Existing base validation: accuracy 0.131068, macro-F1 0.124051, evidence
  micro-F1 0.154930. Adapter deltas on the same validation IDs are +0.606796,
  +0.348878, and +0.190032 respectively. This compares the Colab T4 adapter
  run with the existing local RTX 3050 baseline, so treat deltas as descriptive
  until a same-device local full-split evaluation succeeds. The pinned model
  ID/revision, prompt, validation IDs, and generation backend match, but this
  is not a controlled same-device or test-set result.
- Data audit records canonical manifest hash
  `309ea164a836f9a3f6d6c315578be990370971990412bbb75ef77cde707e063b`,
  disjoint patient splits, 602 train rows, 206 validation rows, and
  `test_data_opened: false`.

The three epoch validation JSONs, run summary, data audit, environment report,
resolved configuration, and fresh-process metrics are retained under
[`results/phase04/phase04_lora_colab_cli_20261110T1650Z_colab_fresh_eval`](../results/phase04/phase04_lora_colab_cli_20261110T1650Z_colab_fresh_eval).

## Objective and boundaries

Complete a clean, full-precision FP16 LoRA run on Colab using the approved 602
training rows; validate after each epoch on the 206 validation rows; select
the best epoch using the configured validation rule; export and verify the
adapter; download it to D:; reload it in a fresh local process; and compare it
with the unchanged base model on the same validation split.

Do not open, inspect, or use the sealed 206-row test split. Do not begin Phase
9 final evaluation until the selected adapter and frozen manifest exist. Keep
the `v1-provisional` prompt, pinned model revisions, token budgets, parser,
and greedy generation settings unchanged. Do not resume the contaminated
local partial run or silently switch to QLoRA/4-bit.

## Existing state and evidence

- Training implementation and local gates are complete; the full laptop run
  failed from OOM and did not finalize an artifact.
- Local suite: 75 passed, 1 PEFT warning, 111 subtests passed.
- The local partial run at `results/training/phase04_lora_r8/` and its
  `best_adapter` are diagnostic only; do not use them as the selected model.
- The prior Kaggle probe artifacts are preserved under
  `artifacts/kaggle_gpu_probe_output_20261110*`; the corrected probe reported
  `torch 2.11.0+cpu` and `cuda_available: false`.
- Colab previously provided a Tesla T4 16 GiB for Phase 0/1 synthetic and
  feasibility work. This is historical evidence only; verify the current
  Colab session before training.
- The canonical split is 602 train, 206 validation, 206 test; manifest SHA-256
  is `309ea164a836f9a3f6d6c315578be990370971990412bbb75ef77cde707e063b`.
- The dataset source and model revisions/hashes are recorded in the earlier
  sections of the [Kaggle-era handoff](phase_04_training_resume_handoff.md)
  and in the split manifest. The Kaggle operational instructions in that
  historical file are superseded by this document.

## Run procedure

1. Follow [`docs/colab_manual_setup.md`](../docs/colab_manual_setup.md). Build
   a uniquely named current source archive with
   `scripts/package_colab_source.py`; upload the ZIP and SHA-256 sidecar plus
   the current `notebooks/colab_runner.ipynb` to Drive/Colab.
2. In Colab choose **Runtime > Change runtime type > GPU**, connect, and run
   the notebook from the top. Review the Drive mount request. Confirm the
   report says `cuda_available: true`, and record GPU name/VRAM and software
   versions. Stop if CUDA is unavailable.
3. Run all notebook verification, data/hash, test, overfit, and resume gates
   in order. Do not proceed to full training if a gate fails.
4. Run the notebook's full Phase 4 training cell with a unique run name. Use
   `configs/train.yaml` unchanged: LoRA r=8, alpha=16, dropout=.05, q/v
   projections; AdamW 2e-4; microbatch 1, accumulation 8; FP16; three epochs;
   validation each epoch. There are 76 optimizer steps per epoch (228 planned)
   and 11 warmup steps. Checkpoints publish to
   `MyDrive/GenAI_Assignment_2_checkpoints/<run>/`.
5. Keep GPU work sequential. If Colab disconnects, preserve the session report
   and checkpoint files. Resume only after checking the bundle manifest,
   checksums, run/config/data identity, and cursor. Never call a partial
   adapter a completed model.
6. Download the notebook's verified export ZIP and sidecar into a new local
   directory under `artifacts/phase04/`, compare SHA-256 values, extract to a
   new path, and run `verify_bundle`.
7. Require `selection_source == "best_validation"`, a selected epoch,
   `epochs_completed == 3`, a selected adapter hash matching the files, all
   three per-epoch validation metrics, pinned model/prompt/data identity, and
   complete bundle verification. A regression is acceptable as a result; do
   not reselect using test data.
8. In a fresh process run
   `scripts/evaluate.py --config configs/eval.yaml --split validation
   --adapter <selected_adapter> --output-dir <new-run-dir>` against all 206
   rows. Confirm test data was not opened and compare absolute metrics/deltas
   with `results/baseline/base_greedy_v1_validation/metrics.json`. The fresh
   Both the fresh Colab and local RTX 3050 processes completed the full
   comparison.

## Acceptance record

After successful verification, update this handoff, `handoffs/phase_04.md`,
`research/decision_log.md`, and `output/deliverables_checklist.md` with the
Colab command/cell, run name, source SHA-256, start/end time, actual GPU and
software versions, gate results, epoch metrics and selected epoch, export and
transfer checksums, fresh-process reload result, base/adapter validation
metrics and deltas, and an explicit statement that test data remained sealed.
Keep adapter weights and large artifacts out of Git.

## After Phase 4 acceptance

- Rerun Phase 5 screening/error analysis with the adapter.
- Add adapter loading to Phase 6 correctness/benchmark scripts and rerun exact
  cached/uncached comparisons.
- Keep Phase 7 standalone; no adapter-specific rerun is needed.
- Add adapter loading to Phase 8 target correctness/benchmark scripts and
  rerun speculative correctness and the `k` sweep.
- Only then create the frozen Phase 9 manifest and evaluate the sealed test
  split.
