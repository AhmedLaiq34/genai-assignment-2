# Explicit KV-cached greedy decoding

## Why the implementation is explicit

`model.generate(use_cache=True)` hides the cache lifecycle. The project therefore
uses an explicit batch-one loop in `src/clinical_match/decoding/cached.py`:

1. **Prefill once.** The complete unpadded prompt is passed with an empty
   `DynamicCache`, a full attention mask and absolute positions `0..L-1`.
2. **Select the first token.** Greedy argmax is taken from the last valid prompt
   position. That token is *not yet* in the cache.
3. **Incremental decode.** If the token is not EOS and the cap is not reached,
   feed only that one token with the returned cache. The attention mask length is
   cache length plus one, and the absolute position is the cache length.
4. **Update and check.** The cache must grow by exactly the number of processed
   tokens. `ExplicitKVCache.assert_length` runs after every forward, and the cache
   can be cropped for later speculative rollback.
5. **Stop.** EOS is never fed back. At a cap, the last generated token is also
   not fed back.

Terminal cache length is therefore:

```text
prompt_length + max(0, generated_tokens - 1)
```

This convention is deliberate and recorded in every result. It avoids pretending
that the last sampled token has been consumed by the model.

## Mask and position rules

- The attention mask covers the complete cached prefix plus the new chunk.
- `cache_position` and `position_ids` are absolute, starting at the current cache
  length.
- Only batch size one and unpadded prompts are supported in this phase.
- `pad_token_id` is intentionally ignored; it is irrelevant without padding.
- Cache length is asserted after each forward; layer count is checked in tests.
- Shape metadata is recorded rather than assumed in production output.

## Correctness

`uncached_greedy` recomputes the full prefix at every step with
`use_cache=False`. This is the reference. The library `generate` path is a third
oracle only; it is not used as the correctness standard.

All saved comparisons are exact token matches:

- `results/cache/correctness.jsonl`: 6/6 comparisons on development prompts of
  335, 341 and 356 tokens, caps 16 and 32.
- `results/cache/correctness_grid.jsonl`: 9/9 comparisons covering the complete
  128/512/1024 prompt-token by 16/64/128 output-cap grid.
- Synthetic 128/512/1024 prompts are deterministic token prompts built by
  tiling/truncating a development prompt; they are not new clinical records.
- Real-model cap-32/cap-64 cases include mid-generation EOS, so EOS termination
  is exercised in addition to cap termination.
- No test-split content was used. All runs use the unchanged pinned base model.

## Local benchmark

Same RTX 3050 Laptop GPU, pinned Qwen2.5-1.5B-Instruct FP16, one loaded model,
two validation development prompts, one warmup and five timed repetitions.
CUDA is synchronized before and after each timed region. Cached per-step timing
adds extra synchronization, which biases the cached path against the speedup.

### Fixed-cap microbenchmark

| Prompt | Mode | Mean latency | Peak allocated | Relative speed |
|---:|---|---:|---:|---:|
| 356 tokens | uncached | 1.9593 s | 3,326,245,376 B | 1.00x |
| 356 tokens | explicit cache | 0.7295 s | 3,224,567,296 B | 2.69x |
| 335 tokens | uncached | 1.8617 s | 3,313,662,464 B | 1.00x |
| 335 tokens | explicit cache | 0.7137 s | 3,217,340,928 B | 2.61x |

### EOS-terminated application prompts

| Prompt | Mode | Generated | Mean latency | Peak allocated | Relative speed |
|---:|---|---:|---:|---:|---:|
| 356 tokens | uncached | 29, EOS | 3.6193 s | 3,333,576,704 B | 1.00x |
| 356 tokens | explicit cache | 29, EOS | 1.2432 s | 3,224,567,296 B | 2.91x |
| 335 tokens | uncached | 30, EOS | 3.8198 s | 3,322,051,072 B | 1.00x |
| 335 tokens | explicit cache | 30, EOS | 1.2923 s | 3,217,340,928 B | 2.96x |

Cached prefill/TTFT was 0.127-0.144 s. End-to-end decode throughput was about
25.6-26.9 tokens/s in the EOS runs. Peak allocated memory is dominated by the
1.5B model weights, so this is not a memory-savings claim. The local speedups are
same-device measurements only; they are not cloud results.

## Provenance

Both correctness and benchmark records include model/tokenizer revisions, config
and dataset SHA-256, dtype, attention implementation, device name, PyTorch and
Transformers versions, EOS token ID, and the git revision plus dirty flag. Raw
per-repetition timings and peak memory are retained.

## Limitations

- No adapter exists yet, so all measurements are unchanged-model results.
- The synthetic 128/512/1024 token prompts test decoder mechanics, not clinical
  behaviour.
- Python orchestration dominates short runs; do not generalize these speedups to
  other hardware, longer sequences or other models.
- The library `generate` oracle can differ from the explicit paths at token level;
  only the explicit uncached reference defines Phase 6 correctness.
- The 128/512/1024 grid uses one source development prompt and does not cover
  every possible EOS position or model backend.
