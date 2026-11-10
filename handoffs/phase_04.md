# Phase 4 handoff: LoRA training and selection

**Operational update (10 November 2026):** the full configured training run is complete on Colab. The Kaggle CLI handoff and failed local partial run are historical/diagnostic only. See the [completed Colab training record](phase_04_colab_training_handoff.md). The evaluator flag is `--adapter <path>`.

**Status: training and validation selection complete; full local reload completed.** The cached-versus-uncached preflight bug was fixed: raw argmax had omitted Qwen's configured repetition penalty. The uncached reference now calls Transformers generation with `use_cache=False`, matching generation processors. The local RTX 3050 run completed the canonical 206-row validation in 212.26 seconds, and all three preflight examples matched exactly. The prior native status `3221225477` did not recur.

The selected run is `phase04_lora_colab_cli_20261110T1650Z`, using the unchanged FP16 config. It completed 228/228 optimizer steps across three epochs and selected epoch 3 by `best_validation`. The validation macro-F1 is 0.472929, evidence micro-F1 0.344961, and accuracy 0.737864. The adapter is at `artifacts/phase04/phase04_lora_colab_cli_20261110T1650Z_bundle/selected_adapter`; its hash is `b823bc3f7e979be9fca7b2f9543305a5b5562721f4342af5487dc3d2941151fb`. The export ZIP SHA-256 is `86d594f59453e45fa3f997c8b1a565c44860c65d58c8671734f112a1f658a41a`.

## What passed locally

| Gate | Command | Result |
|---|---|---|
| Automated tests | `python -m pytest -q` | 28 passed, 1 PEFT warning |
| Real target smoke | `python scripts/train.py --config configs/train.yaml --run-name phase04_real_smoke --smoke --device cuda` | finite loss 0.5740, grad norm 1.494, 4.07 s/step |
| Real target overfit | 12 training examples, 40 optimizer steps | loss 0.4534 -> 0.1174; exact target matches 1 -> 5 of 12; finite |
| Real resume rehearsal | 4 steps, interrupt at 2 | step/epoch/cursor/order identical; max absolute loss delta 0.0 |
| Partial full run | all 602 usable training rows | reached optimizer step 115; epoch 1 completed at step 76 with diagnostic validation metrics; peak allocated 4,498,746,880 bytes; peak reserved 8,873,050,112 bytes; run not finalized |

Evidence: [local gate report](../results/training/phase04_local_gate.json), [resume report](../results/training/resume_rehearsal.json), [overfit report](../results/training/phase04_real_overfit/overfit_report.json), [smoke summary](../results/training/phase04_real_smoke/export/selected_manifest.json).

## Historical local training failure

The 6 GiB GPU cannot reliably complete this FP16 training/validation configuration. Windows reported a larger reserved allocation than physical VRAM, and the user reported an out-of-memory failure. The partial run is retained only for diagnosis:

- `checkpoints/phase04_lora_r8/bundles/resume-step-000076-epoch`
- `checkpoints/phase04_lora_r8/bundles/resume-step-000100-periodic`
- `results/training/phase04_lora_r8/`

Do **not** use `results/training/phase04_lora_r8/best_adapter` or `epoch_1/adapter` as the selected Phase 4 artifact. Epoch-1 validation finished and remains diagnostic only. The completed selected artifact is the Colab export above.

## Completed Colab run and evaluation

The full job ran through the [Colab runner](../notebooks/colab_runner.ipynb) workflow using the connected T4 session via Colab CLI. The effective training command was:

```powershell
python scripts/train.py --config configs/train.yaml --run-name phase04_lora_colab_cli_20261110T1650Z --drive-dir /content/drive/MyDrive/GenAI_Assignment_2_checkpoints
```

The run used all 602 training rows and validated all 206 validation rows after each epoch. The fresh Colab CLI process reloaded the Drive-exported adapter and completed the full validation split. It reported macro-F1 0.472929, accuracy 0.737864, and evidence micro-F1 0.344961. A fresh local RTX 3050 process also completed 206 rows with the canonical config and cached backend: macro-F1 0.472929, accuracy 0.737864, evidence micro-F1 0.346304. Against the same-device local base validation (0.124051 macro-F1, 0.131068 accuracy, 0.154930 evidence micro-F1), the local deltas are +0.348878, +0.606796, and +0.191374. The small evidence metric difference from Colab is consistent with cross-device generation variation. Results are in `results/phase04/local_eval_fixed_full/metrics.json` and `results/phase04/phase04_lora_colab_cli_20261110T1650Z_colab_fresh_eval/`.

The selected export was downloaded and verified locally with `verify_bundle`. The ZIP and individual files pass integrity verification, and the adapter loaded successfully for the full local evaluation.

```powershell
python scripts/evaluate.py --split validation --config configs/eval.yaml --adapter artifacts/phase04/<run>/selected_adapter --output-dir results/training/<run>/local_validation
```

The run's source archive SHA-256 is `ca38d541ab3e51ea27f8b224cb38ae5360d90165f81354866fde9ea97d3ce861`; manifest SHA-256 is `309ea164a836f9a3f6d6c315578be990370971990412bbb75ef77cde707e063b`. The audit confirms disjoint patient splits and `test_data_opened: false`. Do not open the protected test split until Phase 9's frozen manifest is complete and all correctness tests pass.

## What is implemented

- `src/clinical_match/training/data.py`: exact prompt rendering, assistant-only labels, template/EOS validation, patient/example ordering and padding.
- `src/clinical_match/training/trainer.py`: LoRA wrapping, optimizer/scheduler/scaler, deterministic ordering, periodic/epoch checkpoints, epoch validation, selection and export.
- `src/clinical_match/training/checkpoints.py`: verified atomic bundles, RNG/optimizer/scheduler state, resume restoration and pruning.
- `src/clinical_match/training/validation.py`: full 206-row validation generation/scoring.
- `scripts/train.py`: smoke, overfit, resume, stop-after, tiny-model and drive-root options.
- `scripts/rehearse_resume.py`: control vs interrupted/resumed comparison.

## Next phase dependency

Phase 5 is implemented with the unchanged base model as a contract/integration test. Its result objects are adapter-ready, but adapter-specific screening/error evidence must be rerun after Phase 4 completes. Phase 6 base-model cache correctness is independent of the adapter; Phase 8's selected-target speculative run must be repeated after Phase 4.

