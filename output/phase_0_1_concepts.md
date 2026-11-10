# Phase 0–1: the important ideas, in plain language

> **Compute-route update (10 November 2026):** the active training route is **Google Colab**, with the laptop for development and the final app. See [Colab training setup](../docs/colab_manual_setup.md). Verify CUDA in the current runtime.

This is a short conceptual companion to the [master plan](master_plan.md).
It explains why the setup and model-choice work matters and distinguishes
feasibility evidence from model-quality evidence.

## The project in one picture

The model will read a patient description and one typed trial criterion. It
will generate a small structured answer: a criterion label plus IDs of the
original patient sentences that support it. The program, not the model,
retrieves and displays the quoted sentence text.

Inclusion and exclusion labels are not interchangeable. An inclusion criterion
asks whether a required feature is present; an exclusion criterion asks
whether a disqualifying feature applies. Missing information is a legitimate
uncertain answer, not evidence that a condition is absent.

## Phase 0: why set up environments before doing ML?

**A Python environment is a reproducible toolbox.** The system Python on this
machine is newer than many ML packages have traditionally supported. A
project-local Python 3.11 environment keeps compatible libraries separate
from Windows and from other projects. The CUDA-enabled PyTorch build also
belongs in this environment, not in the system Python.

**A GPU name is not a fit result.** The RTX 3050 has 6 GiB of VRAM. A model's
weights are only part of its memory use: attention cache, activations, logits,
temporary buffers and the desktop GPU all consume memory. The environment
doctor records the actual state; the model smoke test measures whether the
intended prompt and model fit. A failed fit is evidence too.

**Local app and cloud research are different jobs.** Google Colab offers a
different GPU through the official CLI, but availability is variable and its
scratch files are temporary. We can use it to research/train, while the
evaluator-facing app must work on this laptop without Drive or a running
kernel. Any cloud checkpoint therefore needs a verified downloaded copy and a
tested local reload.

**A checkpoint must be resumable, not just present.** A completion marker and
file hashes help detect an incomplete or corrupted copy. The small synthetic
test checks the transfer/verification mechanism; it does not yet prove that a
model's optimizer state can resume training. That is a later Phase 4 test.

## Phase 1: why this particular model pair?

The proposed **target** is Qwen2.5-1.5B-Instruct. It generates the screening
answer. The smaller **draft** is Qwen2.5-0.5B-Instruct. In speculative
decoding, the draft suggests several next tokens and the target verifies them.
The method is correct only if the final tokens exactly match ordinary greedy
decoding from the target.

Both candidates are decoder-only language models: they can generate the
answer token by token. An encoder-only classifier might classify text, but
would not directly support the required autoregressive JSON generation,
KV-cache loop, and draft/verify experiment.

The target has about 1.54 billion parameters; the draft about 0.49 billion.
At two bytes per FP16 parameter, their weights alone need about 2.87 GiB and
0.92 GiB. Together that is about 3.79 GiB, before runtime overhead. These are
**lower-bound estimates**, not evidence that either model fits in 6 GiB VRAM.

Their configs both report a vocabulary of 151,936 and 32,768 maximum
positions. That is not enough to prove the draft can propose token IDs for the
target: we still compare the complete token-to-ID vocabularies, added tokens,
special-token IDs and real prompt encodings at the pinned revisions.

## Why LoRA is the proposed first training method

Full fine-tuning updates every model weight and needs optimizer states and
gradients for all those weights. LoRA freezes the large pretrained model and
learns small low-rank update matrices instead. This reduces trainable state,
but does **not** make training free: forward activations, gradients through
the adapters, attention workspaces and output logits still consume memory.

The initial plan is an ordinary 16-bit LoRA run on a verified Colab GPU. If
that cannot fit, a tested 4-bit method is a fallback. Quantization changes
the experiment and software requirements; it should not be selected simply
because it sounds smaller. The measured 10-step T4 pilot used one synthetic
example, gradient accumulation 8 and 1,024-token sequences; it fit with finite
losses. This checks execution and memory, not whether the model learned the
assignment task.

## What makes the choice defensible?

Separate **facts**, **estimates**, and **measurements**:

- **Facts from pinned model configs:** model dimensions, vocabulary, context,
  checkpoint revisions, and license metadata.
- **Estimates:** weight-only memory arithmetic.
- **Measurements:** tokenizer equivalence, actual GPU memory, forward-pass
  success at 256/1,024 tokens, target/draft co-residency, and LoRA fit/time.

Do not promote a candidate to “selected” until its tokenizer and local serving
fit pass. Cloud training fit alone is insufficient if the final local app
cannot run the target. Both pinned models passed target/draft inference and
cached-decode checks on the laptop at 256 and 1,024 tokens; the target and
LoRA pilot also passed their cloud checks. This supports feasibility, not
clinical quality. Published benchmark speedups are not predictions for this
laptop.

The local 1,024-token target run peaked at about 3.42 GiB reserved, and both
models co-resident at 1,024 tokens peaked at about 4.36 GiB reserved. The T4
10-step LoRA pilot peaked at about 5.04 GiB allocated / 5.69 GiB reserved.
These are single-run feasibility measurements; they do not prove task quality,
full-training readiness, or guaranteed future Colab access.

## What you should be able to explain

1. Why a patient-level split and held-out test are needed to prevent leakage.
2. Why model weights are only a lower bound on GPU memory.
3. The difference between the target and speculative draft.
4. Why shared vocabulary size does not prove identical token IDs.
5. What LoRA freezes and what still uses memory.
6. Why Colab is a temporary research environment, not the required app host.

If you can answer these, you understand the choices Phase 0–1 are meant to
establish. Read the source and model comparison notes for citations and exact
revision identifiers.
