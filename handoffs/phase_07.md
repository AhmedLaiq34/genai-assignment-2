# Phase 7 handoff: tiled attention with online softmax

## Status

Complete as a standalone educational implementation on the unchanged local
environment. It is not integrated into the Qwen2 model and makes no production
FlashAttention claim.

## Implementation

- `src/clinical_match/attention/dense.py`: stable dense scaled dot-product
  reference with explicit scaling, mask, max-subtracted softmax and FP32
  accumulation; fully masked rows return zero.
- `src/clinical_match/attention/tiled.py`: query/key tiling with running maximum,
  normalizer and weighted-value accumulator; only score tiles are allocated.
- `src/clinical_match/attention/masks.py`: explicit absolute-position causal
  masks, including rectangular cached-offset queries.
- `src/clinical_match/attention/__init__.py`: package exports.
- `tests/test_attention.py`: shape/mask/GQA/numerical/state tests.
- `scripts/benchmark_attention.py`: warmed, synchronized repeated timing and
  peak-memory comparison.
- `scripts/plot_attention_results.py`: dependency-free SVG plots.
- `configs/attention_bench.yaml`: benchmark shape and protocol.

## Verification

- `pytest -q`: **65 passed, 1 warning, 108 subtests passed**.
- Attention suite alone: **10 passed, 105 subtests passed**.
- All-masked rows are zero in both paths; no NaN/inf observed.
- FP32 tolerance `atol=1e-5, rtol=1e-4`; FP16 tolerance `atol=2e-3, rtol=2e-3`
  against FP32.
- Dense path compared with an FP64 oracle.
- Rectangular cached-offset test changes a future key/value and confirms output
  is unchanged for queries that cannot see it.
- Maximum absolute error across benchmark cells: about `1.1e-6`.

## Benchmark

`configs/attention_bench.yaml`: batch 1, 8 query heads, 2 KV heads, head dim 64,
FP32, causal, lengths 128/512/1,024/2,048, tiles 16/32/64, 2 warmups, 5 measured
runs, RTX 3050 Laptop GPU.

| Length | Dense ms | Tiled-64 ms | Speedup | Dense MB | Tiled-64 MB |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.76 | 2.76 | 0.277x | 11.0 | 11.2 |
| 512 | 2.80 | 38.45 | 0.073x | 44.9 | 17.8 |
| 1,024 | 8.60 | 149.52 | 0.058x | 148.2 | 27.0 |
| 2,048 | 29.80 | 613.66 | 0.049x | 546.3 | 47.0 |

The tiled path is slower due to Python-level tile loops and kernel launch
overhead. Its peak allocation is lower at larger lengths; at 2,048 tokens the
incremental peaks are 520.1 MB (dense) and 12.9 MB (tiled-64). Report this as a
slowdown, not a speedup.

## Artifacts

- `results/attention/benchmark.json`: all timings, peaks and errors.
- `results/attention/benchmark.csv`: machine-readable table.
- `results/attention/error_by_length.svg`: numerical error plot.
- `results/attention/runtime_memory.svg`: runtime and memory plot.
- `results/attention/SHA256SUMS.txt`: artifact checksums.
- `docs/tiled_attention.md`: formulas, evidence and limitations.

## Commands

```powershell
& .venv\Scripts\python.exe -m pytest tests\test_attention.py -q
& .venv\Scripts\python.exe scripts\benchmark_attention.py --config configs\attention_bench.yaml
& .venv\Scripts\python.exe scripts\plot_attention_results.py
```

## Limitations and next inputs

- Not integrated into model attention; keep the standalone/educational label.
- Single device, FP32, causal, batch 1; no dropout, backward pass or training.
- Python implementation is not a fused kernel.
- Phase 8 speculative decoding does not require this module's attention path; it
  uses the Phase 6 explicit KV cache. Phase 7 remains independent evidence for
  E1-E3.

