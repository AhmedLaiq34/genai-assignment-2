# Phase 6 handoff: explicit KV-cached greedy generation

## Status

Implemented with a shared greedy logits policy, tested on the base model, and
adapter-enabled. Adapter-backed explicit-cache correctness matched the uncached
reference in 6/6 cases (three validation prompts, caps 16 and 32); raw evidence
is `results/cache/correctness_adapter.jsonl`. Adapter timing is recorded in `results/cache/benchmark_adapter.json`; explicit cache is measured against uncached generation on two validation prompts, with five repetitions per mode.

## Implementation

- `src/clinical_match/decoding/` package:
  - `reference.py`: explicit uncached reference and library cached oracle.
  - `cache.py`: `ExplicitKVCache` with length, layer-shape, crop and dtype checks.
  - `cached.py`: prefill-once explicit greedy loop with absolute positions,
    growing mask, post-forward cache-length assertions and optional timing.
  - `__init__.py`: preserves the old `from .decoding import ...` API.
- `scripts/check_decoding_correctness.py`: real development prompts and a
  128/512/1024 by 16/64/128 synthetic grid; writes raw JSONL plus summary.
- `scripts/benchmark_decoding.py`: same-device warm/repeated timing, CUDA
  synchronize, peak allocated memory, TTFT, throughput, termination and speedups.
- `scripts/screen_example.py --decoder explicit_cached`: integration check.
- `docs/decoding_cache.md`: prefill/update/position/mask/growth/first-token note.

## Verification

- `pytest -q`: **55 passed, 3 subtests passed**.
- Exact token equality:
  - `results/cache/correctness.jsonl`: 6/6 on development prompts of 335/341/356
    tokens, caps 16/32.
  - `results/cache/correctness_grid.jsonl`: 9/9 on the complete 128/512/1024 by
    16/64/128 grid.
- Cache invariants verified in tests and real artifacts:
  - prefill cache length equals prompt length;
  - each decode step adds one token;
  - final length equals prompt length plus `max(0, generated - 1)`;
  - EOS is not fed back;
  - two tiny random Qwen2 layers produce expected layer shapes and crop correctly.
- Real cap-64 runs terminate at EOS after 29/30 tokens, so mid-generation EOS is
  covered, not only forced caps.
- Screening integration: `results/dev_checks/synthetic_screening_explicit_cache.json`
  validates against `schemas/screening_result.schema.json`; aggregate remains
  `review_needed` because the base model labels every criterion `not applicable`.

## Local benchmark

`results/cache/benchmark.json`: fixed cap 16, two development prompts, one warmup,
five timed repetitions, RTX 3050 Laptop GPU, FP16.

- 356-token prompt: uncached 1.9593 s vs cache 0.7295 s; **2.69x**.
- 335-token prompt: uncached 1.8617 s vs cache 0.7137 s; **2.61x**.

`results/cache/benchmark_eos.json`: normal EOS-terminated application prompts,
cap 64, same protocol.

- 356-token prompt: 29 tokens, uncached 3.6193 s vs cache 1.2432 s; **2.91x**.
- 335-token prompt: 30 tokens, uncached 3.8198 s vs cache 1.2923 s; **2.96x**.
- Cached TTFT 0.127-0.144 s; EOS-run throughput about 25.6-26.9 tokens/s.
- Peak allocated: about 3.33 GB vs 3.22 GB, dominated by model weights.

These are local unchanged-model numbers, not cloud or adapter results.

## Provenance

Correctness and benchmark records include model/tokenizer revisions, config and
dataset SHA-256, dtype, attention implementation, device name, PyTorch and
Transformers versions, EOS token ID, git revision and dirty flag. Raw
per-repetition timings and memory are retained. `results/cache/SHA256SUMS.txt`
contains artifact checksums.

- Model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Eval config SHA-256: `e5d169b68ce549d0ffc06892a3884d22403b323779c70345f76d49c0b576bb95`.
- Validation data SHA-256: `d064084797472e9bc48a54880344f36fe33381437856134664e20355467eb686`.
- Torch `2.11.0+cu128`; Transformers `4.57.1`; attention `sdpa`.

## Commands

```powershell
& .venv\Scripts\python.exe -m pytest -q
& .venv\Scripts\python.exe scripts\check_decoding_correctness.py --config configs\eval.yaml --rows 3 --caps 16 32 --output results\cache\correctness.jsonl
& .venv\Scripts\python.exe scripts\check_decoding_correctness.py --config configs\eval.yaml --rows 1 --lengths 128 512 1024 --caps 16 64 128 --output results\cache\correctness_grid.jsonl
& .venv\Scripts\python.exe scripts\benchmark_decoding.py --config configs\eval.yaml --rows 2 --cap 16 --warmups 1 --repeats 5 --output results\cache\benchmark.json
& .venv\Scripts\python.exe scripts\benchmark_decoding.py --config configs\eval.yaml --rows 2 --cap 64 --warmups 1 --repeats 5 --output results\cache\benchmark_eos.json
```

## Limitations and next inputs

- The library cached `generate` path is an oracle only; it differed from the
  explicit path in the synthetic screening example while labels matched. That is
  not a Phase 6 equality failure because the explicit uncached reference is the
  standard.
- The synthetic grid uses one source development prompt tiled/truncated to the
  target lengths; it is not a clinical sample at those lengths.
- No test data was opened.
- Phase 7 consumes `ExplicitKVCache` and the same prefill/decode contract. Phase 8
  will use cache crop/rollback for speculative blocks.



## Adapter timing update

The adapted target measured 2.94x (356-token prompt) and 2.19x (335-token prompt) cached-versus-uncached mean latency at cap 16 on the local RTX 3050. See `results/cache/benchmark_adapter.json`; the record includes raw repetitions and provenance.
