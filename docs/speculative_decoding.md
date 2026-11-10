# Greedy speculative decoding

This is deterministic target-greedy matching, not stochastic speculative sampling.
The local implementation uses the unchanged Qwen2.5-1.5B target and
Qwen2.5-0.5B draft. The final target must be the validation-selected adapter;
adapter reruns remain pending.

## Invariant

At the start of an iteration, both caches contain exactly the committed prefix
`x`. Both models also have next-token logits after `x`. Caches may have different
internal shapes, but their token IDs and length agree. Mutations are explicit:

```text
start: target cache = x, target logits = f_target(x)
       draft  cache = x, draft  logits = f_draft(x)

draft d1..dk through the draft:
  cache = x + d1..dk; logits after dk are valid

verify d1..dk in one target pass:
  saved target logits after x       -> predicts d1
  target logits after d1..di        -> predicts d(i+1)
  target logits after dk            -> optional bonus

accept the matching prefix; on mismatch:
  crop both caches to len(x) + accepted
  commit target correction c
  process c through both models

on full acceptance:
  retain caches through dk
  optionally commit target bonus b
  process b through both models if the budget remains
```

The one-token alignment is the important part: verification logits after `d_i`
predict `d_(i+1)`, not `d_i`. The saved prefix logits predict `d1`.

## Algorithm

1. Prefill target and draft once on the prompt.
2. Draft up to `k = min(draft_k, remaining budget)` target-greedy tokens through
   the draft cache.
3. Verify the whole draft block in one target forward pass using the target
   prefix cache.
4. Compare draft tokens with target argmax predictions until the first mismatch.
   Commit accepted tokens plus one target correction token.
5. Crop both caches to the committed prefix, then process the correction through
   both models unless it is EOS.
6. If every draft token matches, optionally commit the target bonus token and
   process it through both caches.
7. Stop immediately on EOS; never accept a token after an EOS. Crop both caches
   to the same length, excluding the unfed EOS, so their terminal descriptors
   remain equal. Respect the same output cap as ordinary greedy decoding.

No bonus is permitted if the output budget is exhausted. The implementation
does not fabricate a speedup; on the local 6 GB laptop the extra target and draft
work was slower than target-only cached greedy.

## State-machine tests

`tests/test_speculative.py` covers:

- full acceptance and bonus at the cap;
- partial acceptance, correction, and repeated rejection;
- rejection followed by full acceptance, checking that the committed prefix is
  not lost;
- zero accepted tokens;
- one-token remaining budget;
- correction EOS, draft EOS and bonus EOS;
- exact target-greedy equality for `k=1,2,4` on tiny randomly initialized real-architecture target/draft models;
- EOS equality on a tiny real model pair.

The suite reports **10 tests and 3 subtests passing** for speculative decoding.
Full repository tests report **75 passed, 1 warning, 111 subtests passed**.

## Real-pair correctness

`results/speculative/correctness.jsonl` contains 36 cases from three validation
prompts (356/335/341 tokens), caps 8/16/32 and `k=1/2/4/8`. All **36/36** token
sequences exactly equal the target cached-greedy reference from
`cached_explicit_greedy`. `adapter_pending` is recorded because the target is
still the unchanged base model.

Aggregate acceptance over those cases:

| k | Accepted / proposed | Acceptance rate |
|---:|---:|---:|
| 1 | 70 / 95 | 0.737 |
| 2 | 95 / 137 | 0.693 |
| 4 | 117 / 201 | 0.582 |
| 8 | 126 / 294 | 0.429 |

These are aggregate ratios over the 36 cases. The rate is lower at larger blocks
on these three prompts because a mismatch discards the remainder of the block.
This is a base-model/draft result and may change after adapter selection.

## Local benchmark

`results/speculative/benchmark.json`: one validation prompt of 356 tokens, cap
16, both models resident, one warmup, five measured runs, rotated/round-robin
mode order, CUDA synchronization, empty cache before each timed run, load
excluded from timers. Same RTX 3050 Laptop GPU. This is a single-prompt local timing, not a representative latency estimate.

| Mode | Median s | Speedup vs target cached | Acceptance | Peak allocated MiB |
|---|---:|---:|---:|---:|
| target cached | 0.725 | 1.000x | - | 4,028 |
| speculative k=1 | 1.429 | 0.507x | 0.889 | 4,135 |
| speculative k=2 | 1.329 | 0.546x | 0.714 | 4,135 |
| speculative k=4 | 1.089 | 0.666x | 0.750 | 4,135 |
| speculative k=8 | 1.459 | 0.497x | 0.481 | 4,135 |

Loaded weights are 3.796 GiB. Peak memory includes both models, both KV caches
and allocator state, not incremental attention workspace. The cached baseline
also has both models resident, so this is a marginal-cost comparison rather than
target-only latency. The local result is a slowdown, not a speedup; report it
honestly. A Kaggle T4 sweep may differ and must be recorded separately on that device and session.

## Commands

```powershell
& .venv\Scripts\python.exe -m pytest tests\test_speculative.py -q
& .venv\Scripts\python.exe scripts\check_speculative_correctness.py --rows 3 --caps 8 16 32 --draft-k 1 2 4 8
& .venv\Scripts\python.exe scripts\benchmark_speculative.py --rows 1 --caps 16 --draft-k 1 2 4 8 --warmups 1 --repeats 5
```

## Limitations

- The target is not yet adapter-selected; all local results are base-target
  evidence and must be rerun after adapter selection.
- The local GPU is 6 GB, so both models plus caches fit tightly; larger blocks
  can be slower from draft overhead and correction/verification costs.
- No stochastic acceptance, sampling correction, tree speculation or cloud
  measurement is claimed.
- The real-pair test uses three prompts and fixed caps, not a full clinical
  quality evaluation.


