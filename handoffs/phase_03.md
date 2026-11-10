# Phase 3 handoff: evaluation harness and unchanged-model baseline

**Status: complete.** The parser, fixed-class/evidence metric implementation, uncached reference decoder and validation evaluator are in place. The unchanged target has been evaluated on all 206 validation annotations. No test rows were loaded or inspected.

## What is now available

- [Prompt builder](../src/clinical_match/prompts.py) serializes only the allowed model input, preserves original sentence IDs, applies the existing leading-whole-sentence token budget, and records the exact prompt token IDs with every prediction.
- [Output parser](../src/clinical_match/output_parser.py) preserves raw text and malformed output, enforces the JSON shape and criterion-specific labels, flags malformed evidence lists, and scores invalid sentence IDs as false positives. It never converts a bad answer into abstention.
- [Metrics](../src/clinical_match/metrics.py) report fixed six-class macro-F1 and per-class values, criterion-specific four-class results, an extra invalid-prediction confusion column, evidence micro/macro scores, nonempty-gold results, and parse, invalid-ID, runtime and truncation counts.
- [Decoder references](../src/clinical_match/decoding.py) include explicit uncached greedy forward passes and the library's cached greedy generation path. The latter is used for the full evaluation to make the local GPU run practical; it is not the custom KV-cache implementation required later in Phase 6.
- [Evaluator](../scripts/evaluate.py) accepts only train/validation, checks the selected annotations against the canonical manifest, freezes model/prompt/generation settings in its artifacts, and records all rows even when runtime errors occur.
- [Metric fixture](../tests/test_metrics.py) checks hand-calculated values and malformed output. [Baseline verifier](../scripts/verify_baseline.py) checks one-to-one validation coverage and reproduces the saved metrics from predictions.

## Baseline run

The baseline uses `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, the Phase 2 prompt (`v1-provisional`, carried forward unchanged), greedy decoding, FP16 and the local NVIDIA GeForce RTX 3050 6GB Laptop GPU. The exact config, prompt, model-config, split-manifest and input-data SHA-256 values are in [metrics.json](../results/baseline/base_greedy_v1_validation/metrics.json); the resolved settings are in [run_config.json](../results/baseline/base_greedy_v1_validation/run_config.json). Predictions are saved locally at `results/baseline/base_greedy_v1_validation/validation_predictions.jsonl`; this file is ignored by Git because it contains source text and model outputs. The prediction file has 206 unique rows, matching the validation manifest exactly.

| Measure | Result |
|---|---:|
| Classification accuracy | 0.1311 |
| Six-class macro-F1 | 0.1241 |
| Evidence micro-F1 | 0.1549 |
| Evidence micro-F1 on 90 nonempty-gold records | 0.3379 |
| Valid JSON / legal output shape | 202/206 |
| Runtime failures / output-cap truncations | 0 / 0 |
| Test data opened | No |

The output parser found four JSON failures. The scores are an unchanged-model reference point for a later same-settings adapter comparison. They do not establish clinical accuracy, safety or suitability.

## Cached-path check and interpretation

On the first three validation examples, the library cached generation path was compared with the explicit uncached decoder. Generated token sequences differed. In all three cases, strict parsing returned the same label, evidence IDs and parse state. The evaluator records both `exact_token_match` and `parsed_output_match` in the metric artifact, and refuses the cached run if parsed predictions differ. This Phase 3 convenience check is not a claim of token-level equivalence; Phase 6 still requires exact generated-token equality for the custom cached decoder.

## Verification and next step

Verified with `python -m pytest -q tests/test_metrics.py tests/test_phase2_contracts.py` (6 passed), `python scripts/verify_metrics_fixture.py`, and `python scripts/verify_baseline.py` (206 IDs covered; saved metrics reproduced).

Phase 4 training is the next implementation phase and is intentionally outside this handoff. The active route is Google Colab using the existing runner notebook. The Phase 4 trainer, deterministic patient-group batches, verified resumable checkpoints and validation-only selection are implemented and locally tested; the real adapter run remains outstanding. The prior synthetic LoRA feasibility pilot is not task training or quality evidence. Keep the held-out test sealed. See the [Colab guide](../docs/colab_manual_setup.md) and [training handoff](phase_04_colab_training_handoff.md).

