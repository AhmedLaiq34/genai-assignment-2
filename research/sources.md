# Research source notes

These are concise starting notes, not a substitute for reading and citing the
sources in the final paper. Claims below are paraphrased; performance numbers
from publications are not project results.

| Source | Useful claim | Limitation / caution | Design consequence |
|---|---|---|---|
| Jin et al., TrialGPT, *Nature Communications* (2024), [paper](https://www.nature.com/articles/s41467-024-53081-z), [NCBI repository](https://github.com/ncbi-nlp/TrialGPT) | Decomposes trial matching around criteria and evidence; supports criterion-level evidence as a useful inspection unit. | Their study/system, data protocol, models and evaluation are not this assignment's fixed patient split. Published scores cannot be claimed as ours. A citation is not proof of clinical correctness. | Use the prescribed criterion-annotation data, preserve uncertainty and evidence IDs, and compare our own held-out outputs. |
| Qwen team, [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671), plus pinned [1.5B config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/config.json) and [0.5B config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/config.json) | Establishes the Qwen2-family decoder design; individual configs report model dimensions, vocabulary and context. | A family report is not a substitute for exact checkpoint metadata. Parameter count or context length alone does not demonstrate local memory fit or task quality. | Pin revisions, compare exact tokenizer mappings, use the provided chat template, and run local inference/co-residency smoke tests. |
| Hu et al., [LoRA](https://arxiv.org/abs/2106.09685) | Freeze pretrained weights and train low-rank updates, reducing trainable parameters and memory compared with updating all weights. | Results depend on task, rank, modules and setup. LoRA does not eliminate activations, optimizer state for adapters, logits, or inference memory. It does not promise a clinical-quality gain. | Start with the plan's LoRA pilot, log trainable parameter counts, and decide from measured fit and validation—not the paper's reported results. |
| Dettmers et al., [QLoRA](https://arxiv.org/abs/2305.14314) | Shows an approach to adapting quantized base weights while training low-rank adapters. | Quantization may change numeric behavior and adds hardware/package constraints; availability of a recipe does not prove it works in our Windows or Colab environment. | Keep 4-bit adaptation as a measured fallback, not an assumed default. Treat a precision change as a separately identified run. |
| Hugging Face, [Transformers cache explanation](https://huggingface.co/docs/transformers/main/en/cache_explanation) | KV pairs from prior tokens can be stored and reused during autoregressive inference; cache shape, mask and positions must agree. | Cache APIs vary by library version and cache strategy. Training-time cache use is not equivalent and can cause errors. | Pin runtime versions, inspect actual cache APIs, and implement/test the explicit cache loop and cache growth. |
| Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) | Tiling and IO-aware scheduling can compute exact attention while reducing high-bandwidth-memory traffic. | A simple Python tiled/online-softmax educational implementation is not the fused CUDA FlashAttention kernel and may be slower. | Implement a clear dense reference and tiled algorithm; measure runtime and memory without claiming production kernel performance. |
| Rabe and Staats, [Self-attention Does Not Need O(n^2) Memory](https://arxiv.org/abs/2112.05682) | Blockwise online accumulation can avoid materializing a full attention-score matrix. | Numerical correctness depends on stable rescaling, masking, all-masked rows, and accumulation precision. | Test non-divisible tiles, causal offsets, fully masked rows, and numerical error against a reference. |
| Leviathan et al., [Speculative Decoding](https://arxiv.org/abs/2211.17192) | A smaller draft can propose multiple tokens for a target to verify, enabling exact output-preserving acceleration under an appropriate algorithm. | Published speedups are hardware-, model-, and algorithm-dependent; their sampling procedure is not identical to this assignment's greedy target-equivalence specialization. | Implement draft/verify/reject/correct/cache-crop explicitly; prove token equality and report slowdown if that is what measurements show. |
| PyTorch, [official installation selector](https://pytorch.org/get-started/locally/) | Selects wheels by OS, package manager, Python and CUDA runtime. The CUDA version printed by the NVIDIA driver is not proof that a local CUDA toolkit is installed. | Driver capability does not mean every wheel/kernel combination works on a given device. | Install an official CUDA-enabled Torch wheel in the project venv and verify a real CUDA tensor operation. |
| Kaggle, [API and CLI documentation](https://www.kaggle.com/docs/api) | The official CLI can authenticate, push private kernels, check status, and download kernel output. | GPU type, quota, queue state, internet access, and output retention are not guaranteed. CLI commands and kernel behavior can change. Credentials must be handled securely. | Use the private Kaggle training kernel as research compute; verify GPU/CUDA, download output, and keep the final app local. |
| Google, [Colab FAQ](https://research.google.com/colaboratory/faq.html) | Active training route: free runtimes have variable availability, limits and temporary local storage. | A GPU type, session duration, weekly quota, or uninterrupted training is not guaranteed. | Use Colab only after verifying the current runtime allocation. |

## Live facts verified for this setup

- Dataset metadata/API was pinned to revision
  `1cfcafde94a1560a33b4addc1638664fe28fc059`.
- Target model revision:
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Draft model revision:
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Both model configs report `Qwen2ForCausalLM`, vocabulary size 151,936,
  context 32,768, and two KV heads; target has hidden size 1,536 and 28
  layers; draft has hidden size 896 and 24 layers. Tokenizer-map equality
  was checked locally and in historical Colab; the active Colab run must
  re-verify it in its own environment.
