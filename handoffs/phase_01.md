# Phase 1 handoff

## Status

**Phase 1 feasibility gates: complete.** The pinned tokenizer pair matches;
the target fits locally and on the Colab T4; target+draft fit the local
application envelope; and a synthetic 10-step LoRA pilot passed on T4.
The pair is selected for subsequent implementation, with quality validation
still pending.

## Implemented

- Pinned candidate IDs and revisions in `configs/model.yaml`.
- Candidate architecture/memory comparison in
  `research/model_comparison.md`.
- Source summaries with claims, cautions and design consequences in
  `research/sources.md`.
- Alternatives and evidence status in `research/decision_log.md`.
- Full target/draft tokenizer inspection in `scripts/check_tokenizers.py`.
- Local tokenizer report in `results/tokenizer_compatibility_local.json`.
- Synthetic local inference/co-residency script and measured report in
  `scripts/smoke_models.py` and `results/feasibility_local.json`.
- Synthetic 10-step gradient-accumulated LoRA pilot in
  `scripts/fit_pilot.py` and T4 evidence in `results/fit_pilot_colab.json`.
- Colab target inference report in `results/feasibility_colab.json`.
- Full tokenizer-equivalence report in
  `results/tokenizer_compatibility_colab.json`.

## Verified local facts

- Model revisions:
  - Target `Qwen/Qwen2.5-1.5B-Instruct`:
    `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
  - Draft `Qwen/Qwen2.5-0.5B-Instruct`:
    `7ae557604adf67be50417f59c2c2f167def9a775`.
- Both pinned configs report vocabulary size 151,936 and 32,768 maximum
  positions.
- Local full-vocabulary maps, added vocabularies, special-token IDs, chat
  templates, and synthetic prompt token IDs all match.
- On the local RTX 3050 6GB, target FP16 forward and one-token cached decode
  passed at exactly 256 and 1,024 prompt tokens. Target+draft co-residency
  and cached decode passed at both lengths.
- Target peak reserved VRAM at 1,024 tokens was 3,676,307,456 bytes; pair
  peak reserved VRAM at 1,024 tokens was 4,685,037,568 bytes. These are
  measured snapshots, not guaranteed headroom.
- On the Tesla T4, the target passed both prompt lengths and target+draft
  passed the 256-token inference/co-residency pilot.
- The T4 LoRA pilot completed 10 FP16 optimizer steps, gradient accumulation
  8, total sequence length 1,024, synthetic-only. All losses were finite;
  peak allocated/reserved memory was 5,406,132,736 / 6,106,906,624 bytes;
  mean step duration was 4.87 seconds.
- The CLI's execution request timed out waiting for the final Jupyter reply
  after the remote subprocess reported return code 0. The output JSON was
  subsequently read from the VM, checked for `passed: true`, and included in
  the checksum-verified downloaded bundle.

## Scope boundary

These are execution, tokenizer, and resource-feasibility results, not
evidence of clinical performance. The toy synthetic loss is not a training
metric for the assignment dataset. Training on the prescribed data, model
quality comparisons, and held-out evaluation belong to later phases. The
Colab T4 session was stopped after result transfer; its CLI usage display
showed 0.00 balance and 1.15 compute-units/hour while it was running. After
stopping, a fresh check showed 0 active sessions, 0 active assignments, and
0.00 compute-units/hour. Check current usage before requesting more compute.
