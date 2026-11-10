# Tiled attention with online softmax

This is a standalone educational implementation, not a fused FlashAttention
kernel and not the path used by the Qwen2 model in the application.

## Contract

- Query `Q`: `[batch, heads, query_length, head_dim]`.
- Key/value `K`, `V`: `[batch, key_heads, key_length, head_dim]`.
- `key_heads` may be smaller than `heads`; each key/value head is shared by
  `heads / key_heads` consecutive query heads (GQA/MQA).
- A boolean mask is `[query, key]`, `[batch, query, key]` or
  `[batch, heads, query, key]`; `True` means visible.
- Default scale is `1 / sqrt(head_dim)`.
- Dense and tiled paths accumulate in FP32 and cast the result to the requested
  output dtype.

## Dense reference

`src/clinical_match/attention/dense.py` materializes the score matrix, applies
the mask with `-inf`, subtracts the row maximum, masks weights, normalizes and
multiplies by `V`. Fully masked rows return zero rather than NaN.

## Tiled recurrence

`src/clinical_match/attention/tiled.py` processes query blocks and key/value
tiles. For each row it maintains running maximum `m`, normalizer `l` and
unnormalized weighted-value accumulator `A`, all in FP32:

```text
S = Q_tile K_tile^T / sqrt(d), with mask applied
m_new = max(m, row_max(S))
alpha = exp(m - m_new), or 0 when both maxima are -inf
P = exp(S - m_new), or 0 where masked
l_new = alpha * l + row_sum(P)
A_new = alpha * A + P V_tile
m, l, A = m_new, l_new, A_new
output = A / l, or 0 when l == 0
```

Only `[batch, heads, query_tile, key_tile]` score tiles and tile-sized
temporaries are allocated. The full `[batch, heads, query_length, key_length]`
matrix is never created by the tiled path.

Causal masking uses absolute positions. For a cached rectangular shape with
`query_length < key_length`, `causal_mask(query_length, key_length,
query_offset=key_length - query_length)` makes the newest queries see the full
prefix without relying on a backend's default causal alignment.

## Correctness evidence

`tests/test_attention.py` covers:

- lengths 1, 17, 64, 129 and 256;
- head dimensions 32, 64 and 128;
- 4 query heads with 1, 2 or 4 key/value heads;
- causal and fully visible masks;
- query/key tiles 16, 32 and 64, including non-divisible lengths;
- large logits (Q scaled by 50);
- fully masked rows;
- rectangular cached-offset masks and a future-key perturbation test;
- FP16 inputs with FP32 accumulation;
- invalid GQA grouping and invalid mask shapes.

The suite reports **10 tests and 105 subtests passing**. FP32 comparisons use
`atol=1e-5, rtol=1e-4`; FP16 uses `atol=2e-3, rtol=2e-3` against an FP32
reference. The dense implementation is also compared with an FP64 oracle.

## Benchmark

`configs/attention_bench.yaml` uses batch 1, 8 query heads, 2 key/value heads,
head dimension 64, FP32, causal masking, lengths 128/512/1,024/2,048, tiles
16/32/64, two warmups and five measured runs on the RTX 3050 Laptop GPU.

| Length | Dense ms | Tiled-64 ms | Speedup | Peak dense MB | Peak tiled-64 MB | Full scores MB | Tile scores MB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.76 | 2.76 | 0.277x | 11.0 | 11.2 | 0.50 | 0.125 |
| 512 | 2.80 | 38.45 | 0.073x | 44.9 | 17.8 | 8.0 | 0.125 |
| 1,024 | 8.60 | 149.52 | 0.058x | 148.2 | 27.0 | 32.0 | 0.125 |
| 2,048 | 29.80 | 613.66 | 0.049x | 546.3 | 47.0 | 128.0 | 0.125 |

The tiled path is much slower because this Python loop launches many small
kernels and does not fuse scheduling or data movement. Its peak allocation is
lower at larger lengths; total peak memory still includes dense input tensors
and allocator state. Incremental peaks subtract the pre-call baseline: at 2,048
tokens dense uses 520.1 MB versus 12.9 MB for tiled-64. This is an honest
educational comparison, not a speedup claim. The maximum tiled-vs-dense absolute
error across all length/tile cells was about `1.1e-6`.

`results/attention/benchmark.json` retains every per-repeat timing and peak
sample plus the derived errors; `results/attention/benchmark.csv` is the table source.
`results/attention/error_by_length.svg` and
`results/attention/runtime_memory.svg` are generated from those logs.

## Commands

```powershell
& .venv\Scripts\python.exe -m pytest tests\test_attention.py -q
& .venv\Scripts\python.exe scripts\benchmark_attention.py --config configs\attention_bench.yaml
& .venv\Scripts\python.exe scripts\plot_attention_results.py
```

## Limitations

- No integration into the Transformers Qwen2 attention module; the model still
  uses its own SDPA path.
- The benchmark is single-device, FP32, causal, batch 1, and does not cover
  dropout, backward passes or training.
- The Python implementation is not a production FlashAttention substitute.
- Adapter-independent only; Phase 7 does not require or claim an adapter.


