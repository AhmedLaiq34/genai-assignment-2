# Candidate model comparison

The pinned target/draft pair passed the phase-1 tokenizer and inference-fit
gates and is selected for subsequent implementation. This is a feasibility
choice, not a claim about task quality; development/validation quality checks
and final held-out evaluation remain future work.

| Property | Target candidate | Draft / smaller-target candidate |
|---|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-0.5B-Instruct` |
| Pinned revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Parameter count | 1,543,714,304 | 494,032,768 |
| Architecture | `Qwen2ForCausalLM`; 28 layers; hidden 1,536; 12 query heads; 2 KV heads | `Qwen2ForCausalLM`; 24 layers; hidden 896; 14 query heads; 2 KV heads |
| Config vocabulary/context | 151,936 / 32,768 | 151,936 / 32,768 |
| Config dtype / license | BF16 checkpoint / Apache-2.0 | BF16 checkpoint / Apache-2.0 |
| Raw FP16 weight lower bound | about 2.87 GiB | about 0.92 GiB |
| Proposed role | Task target; must fit local app and research runtime | Draft for speculation; also fallback target if task/fit results favor it |

The raw weight estimate is parameter count times two bytes. It is only a lower
bound: inference also uses KV cache, activations, logits, allocator and kernel
workspace. The two-model pair is roughly 3.79 GiB before those costs, so the
RTX 3050's 6 GiB VRAM makes co-residency plausible but not proven.

Both use the same reported vocabulary size and model family, but this does not
prove that every token maps to the same ID or that special tokens/templates
match. `scripts/check_tokenizers.py` compares full vocabularies, added tokens,
special IDs, chat templates, and synthetic prompt encodings before declaring
the pair compatible.

## Why not an encoder-only model?

An encoder such as BERT is designed to represent an input sequence, not
autoregressively generate the required JSON answer one token at a time. It
could be used for a separate classifier, but it does not directly serve the
assignment's cached greedy and speculative token-generation experiments.

## Decision gate

1. Verify model/tokenizer metadata at the pinned revisions.
2. Measure target inference at 256- and 1,024-token synthetic prompts locally.
3. Measure whether target and draft coexist on the local GPU.
4. A 10-step LoRA fit pilot on the allocated T4 passed with finite synthetic
   losses; the result and resource measurements are saved locally.
5. Freeze a target only if its research fit and local application fit are both
   demonstrated. Both selected candidates passed the measured local inference
   envelope; no deployment quantization is currently required.

No phase-1 model-quality comparison or retained/training adapter has been
claimed. The short synthetic LoRA pilot is only an execution and memory check.

## Local feasibility evidence

On the RTX 3050 6GB laptop, the pinned target passed synthetic FP16 forward
and one-token cached-decode checks at exactly 256 and 1,024 prompt tokens.
Target and draft also passed co-residency and cached-decode checks at 256
and 1,024 tokens. The target's 1,024-token peak reserved memory was
3,676,307,456 bytes; the pair's 1,024-token peak reserved memory was
4,685,037,568 bytes. On the Tesla T4, target inference passed at both prompt
lengths and the pair fit check passed at 256 tokens; the synthetic LoRA pilot
also passed. These are one-run measurements, not guaranteed runtime headroom
or model-quality evidence.

The local pair-at-1,024 result closes the local application-envelope gate.
The cloud GPU produced 5,406,132,736 peak allocated bytes and
6,106,906,624 peak reserved bytes during the 10-step FP16 LoRA pilot, with
finite losses and 4.87 seconds mean per optimizer step. Full outputs are in
`results/feasibility_local.json`, `results/feasibility_colab.json`, and
`results/fit_pilot_colab.json`.
