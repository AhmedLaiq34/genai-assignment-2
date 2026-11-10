# Phase 8 handoff: greedy speculative decoding

## Status

Complete. The adapter-backed target matched canonical greedy in 36/36 cases over three validation prompts, caps 8/16/32, and k=1/2/4/8. The adapter-backed paired benchmark is `results/speculative/benchmark_adapter.json`; speculation was slower for every measured k. No speedup is claimed.

## Implementation

- `src/clinical_match/decoding/speculative.py`: deterministic greedy block
  drafting, one-pass target verification, accepted-prefix plus correction,
  explicit cache crop/reconciliation, EOS and bonus handling.
- `src/clinical_match/decoding/__init__.py`: exports `speculative_greedy`.
- `tests/test_speculative.py`: mock-logit state-machine tests plus tiny
  target/draft model equality tests.
- `scripts/check_speculative_correctness.py`: real validation-prompt token
  equality runner with acceptance traces.
- `scripts/benchmark_speculative.py`: same-device cached vs speculative timing,
  peak memory and acceptance measurements.
- `docs/speculative_decoding.md`: invariant, alignment, algorithm, results and
  limitations.

## Verification

- `tests/test_speculative.py`: **10 passed, 3 subtests passed**.
- Full suite: **75 passed, 1 warning, 111 subtests passed**.
- State-machine coverage: zero acceptance, partial acceptance, repeated
  rejection, rejection then full acceptance, full acceptance with bonus, bonus
  EOS, draft EOS with reconciled cache lengths, correction EOS, one-token budget.
- Real target/draft correctness: **36/36 exact token matches** on three
  validation prompts (356/335/341 tokens), caps 8/16/32 and k=1/2/4/8.
- Every benchmark mode also asserted exact equality against target cached greedy.

## Real-pair correctness artifact

`results/speculative/correctness.jsonl`; summary in
`results/speculative/correctness_summary.json`.

Aggregate acceptance:

| k | Accepted / proposed | Rate |
|---:|---:|---:|
| 1 | 70 / 95 | 0.737 |
| 2 | 95 / 137 | 0.693 |
| 4 | 117 / 201 | 0.582 |
| 8 | 126 / 294 | 0.429 |

## Local benchmark

One 356-token validation prompt, cap 16, both models resident, one warmup, five
measured runs, rotated/round-robin mode order, CUDA synchronization, empty cache
before each timed run, load excluded.

| Mode | Median s | Speedup | Acceptance | Peak MiB |
|---|---:|---:|---:|---:|
| target cached | 0.725 | 1.000x | - | 4,028 |
| speculative k=1 | 1.429 | 0.507x | 0.889 | 4,135 |
| speculative k=2 | 1.329 | 0.546x | 0.714 | 4,135 |
| speculative k=4 | 1.089 | 0.666x | 0.750 | 4,135 |
| speculative k=8 | 1.459 | 0.497x | 0.481 | 4,135 |

Loaded weights: 3.796 GiB. The cached baseline also has both models resident, so
this is a marginal-cost comparison. The local result is a slowdown. Do not
present it as a speedup or compare it with cloud timings.

## Artifacts

- `results/speculative/correctness.jsonl`: token equality plus acceptance blocks.
- `results/speculative/correctness_summary.json`: case counts/provenance.
- `results/speculative/benchmark.json`: all repeated timings, peaks and stats.
- `results/speculative/benchmark.csv`: machine-readable benchmark table.
- `results/speculative/SHA256SUMS.txt`: artifact checksums.

## Commands

```powershell
& .venv\Scripts\python.exe -m pytest tests\test_speculative.py -q
& .venv\Scripts\python.exe scripts\check_speculative_correctness.py --rows 3 --caps 8 16 32 --draft-k 1 2 4 8
& .venv\Scripts\python.exe scripts\benchmark_speculative.py --rows 1 --caps 16 --draft-k 1 2 4 8 --warmups 1 --repeats 5
```

## Adapter-backed update

- Correctness: `results/speculative/correctness_adapter.jsonl` has **36/36**
  exact matches over the same three validation prompts, caps 8/16/32, and
  k=1/2/4/8. The summary records the target adapter and unchanged draft.
- Local benchmark: `results/speculative/benchmark_adapter.json` and `.csv`
  record the same-device paired run (one prompt, cap 16, one warmup, five
  repetitions). All benchmark paths assert exact target-token equality.
- Keep this adapter-backed local timing distinct from the historical base
  timing and do not describe speculative decoding as a general speedup.

## Limitations

- One local benchmark prompt; correctness uses three prompts.
- Peak memory includes both complete models and KV caches.
- No stochastic acceptance or tree speculation.



