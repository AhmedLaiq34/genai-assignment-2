> **Compute-route update (10 November 2026):** this file records historical Colab evidence. The active route is **Google Colab**; see [Colab training setup](../docs/colab_manual_setup.md). This file records earlier Colab feasibility evidence; verify CUDA in the current training session.

# Collective handoff: Phase 0 and Phase 1

**Status:** Phase 0 setup/verification and Phase 1 research-feasibility gates
are complete. The model pair is selected for subsequent implementation.
This does **not** claim clinical task quality, dataset training, a completed
application, or final held-out evaluation.

**Prepared:** 9 November 2026  
**Workspace:** `D:\Gen_AI_Assignment_2`

## 1. Executive summary

- A reproducible Windows/Python/CUDA development environment and an actual
  Colab research runtime were inspected and smoke-tested.
- Pinned target/draft tokenizers have identical vocabulary mappings, added
  tokens, special-token IDs, chat templates, and tested synthetic prompt IDs.
- The target passed its synthetic inference/cache checks locally and on the
  Colab T4. The target and draft both fit together on the local RTX 3050 at
  the tested 256- and 1,024-token prompt lengths.
- A 10-step FP16 LoRA feasibility pilot completed on the T4 using synthetic
  text only. It tests that the training path runs and fits; it is not task
  training and does not establish useful learning.
- Colab evidence was collected into a checksum-verified bundle, downloaded,
  restored, and verified locally.
- No additional Phase 0 or Phase 1 feasibility work is open. Later phases
  still need to train/evaluate on the prescribed dataset, assess task quality,
  implement the remaining system, and run final held-out evaluation.

## 2. Phase 0: environment and project contract

### Completed

- Project-local Python **3.11.17** environment under `.venv`; system Python
  was left untouched.
- Local CUDA PyTorch **2.11.0+cu128** and pinned shared dependencies.
- D:-resident model, uv, and pip caches; local and Colab runtime profiles.
- Environment doctor records OS, Python/packages, CPU RAM, disk, GPU, CUDA
  and dtype support, and can execute a real matrix multiply and tiny
  Transformer forward pass.
- Local CUDA smoke passed on the **RTX 3050 Laptop GPU, 6 GiB VRAM**.
- Colab CUDA smoke passed on a **Tesla T4, 16 GiB VRAM**. Colab used Python
  3.13.16 and PyTorch 2.14.1+cu130; Colab's preinstalled Torch was retained.
- Atomic checksum-verified bundle creation and restore helpers are covered
  by synthetic round-trip, corruption, and overwrite-refusal tests.
- A source-only Colab archive is generated with a SHA-256 sidecar; raw
  dataset rows, caches, model weights, and prepared JSONL are excluded.
- The Colab notebook remains a thin interactive runner; the final app is
  intended to run locally without a notebook or Drive dependency.
- The official `googlecolab/google-colab-cli` v0.7.5 was installed outside
  the project venv in the existing Ubuntu 24.04 WSL distribution. Account
  sign-in was performed by the owner; credentials are not stored in the
  project or handoff.
- Five cloud report files were bundled and restore-tested on the T4,
  transferred using the official CLI, checksum-verified, restored locally,
  and JSON-validated.

### Evidence

- Local runtime: [`../results/environments/local.json`](../results/environments/local.json)
- Colab runtime: [`../results/environments/colab_20261109T110002Z.json`](../results/environments/colab_20261109T110002Z.json)
- Primary-environment pointer: [`../results/environment.json`](../results/environment.json)
- Checkpoint helper: [`../src/clinical_match/checkpointing.py`](../src/clinical_match/checkpointing.py)
- Checkpoint tests: [`../tests/test_checkpointing.py`](../tests/test_checkpointing.py)
- Colab runner: [`../notebooks/colab_runner.ipynb`](../notebooks/colab_runner.ipynb)
- Setup and commands: [`../README.md`](../README.md)
- Phase-specific handoff: [`phase_00.md`](phase_00.md)

### Cloud account and cost caution

The T4 session was explicitly stopped after evidence transfer. There are no
active Colab sessions. During the run the CLI showed a zero compute-unit
balance and a nonzero usage rate; after stopping, it reported zero active
assignments and a zero usage rate. This is not a promise of free future
compute. Check current usage and obtain user approval before allocating any
future runtime.

## 3. Phase 1: research and model feasibility decision

### Selected pinned candidates

| Role | Model | Revision | Phase 1 decision |
|---|---|---|---|
| Target | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Selected for subsequent implementation |
| Speculative draft | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | Selected as the smaller draft; task utility remains to be evaluated |

Both use the Qwen2 decoder architecture, report vocabulary size **151,936**
and maximum position length **32,768**, and report Apache-2.0 metadata. The
tokenizer check compared complete mappings rather than relying on vocabulary
size alone.

### Acceptance evidence

- Target and draft tokenizer checks passed locally and in Colab:
  vocabulary mappings, added vocabulary, special IDs, chat templates, and
  synthetic prompt token IDs all match.
- Local RTX 3050 FP16 target inference passed at **256 and 1,024 prompt
  tokens**. One-token cached decode also passed.
- Both models co-resided on the local RTX 3050 and passed inference/cache
  checks at **256 and 1,024 prompt tokens**. At 1,024 tokens, pair peak
  reserved memory was **4,685,037,568 bytes** (about 4.36 GiB).
- On the Colab T4, target inference passed at **256 and 1,024 prompt tokens**.
  The T4 target+draft co-residency script was run at 256 tokens and passed;
  a T4 pair-at-1,024 result was not collected and is not required for the
  local deployment gate.
- The T4 LoRA pilot used FP16, 10 optimizer steps, accumulation 8, sequence
  length 1,024, and synthetic text only. All losses were finite; peak
  allocated/reserved memory was **5,406,132,736 / 6,106,906,624 bytes**;
  mean optimizer-step time was **4.87 seconds**.
- The LoRA command's CLI request timed out waiting for the final Jupyter
  response after the remote subprocess reported return code 0. The persisted
  result was independently read from the VM, verified as `passed: true`,
  downloaded, and included in the hash-verified bundle.
- The target/draft pair is selected on feasibility grounds only. This phase
  does not measure clinical accuracy or speculative-decoding speedup.

### Evidence

- Model settings: [`../configs/model.yaml`](../configs/model.yaml)
- Tokenizer implementation: [`../scripts/check_tokenizers.py`](../scripts/check_tokenizers.py)
- Local tokenizer report: [`../results/tokenizer_compatibility_local.json`](../results/tokenizer_compatibility_local.json)
- Colab tokenizer report: [`../results/tokenizer_compatibility_colab.json`](../results/tokenizer_compatibility_colab.json)
- Local inference report: [`../results/feasibility_local.json`](../results/feasibility_local.json)
- Colab inference report: [`../results/feasibility_colab.json`](../results/feasibility_colab.json)
- LoRA pilot implementation: [`../scripts/fit_pilot.py`](../scripts/fit_pilot.py)
- Colab LoRA report: [`../results/fit_pilot_colab.json`](../results/fit_pilot_colab.json)
- Research source ledger: [`../research/sources.md`](../research/sources.md)
- Candidate comparison: [`../research/model_comparison.md`](../research/model_comparison.md)
- Decision log: [`../research/decision_log.md`](../research/decision_log.md)
- Phase-specific handoff: [`phase_01.md`](phase_01.md)

## 4. Colab result transfer and integrity

The downloaded archive is [`../artifacts/phase01-colab-verified-bundle.zip`](../artifacts/phase01-colab-verified-bundle.zip).

```text
SHA-256:
a7494de262ae44edcd55c2cc08cd7bfd9eeed42aa6a91331c7f2fcb296b7f214

Locally restored bundle:
results/colab_verified_bundle/

Verified report files:
environment.json
environments/colab_20261109T110002Z.json
feasibility_colab.json
fit_pilot_colab.json
tokenizer_compatibility_colab.json
```

The refreshed source-only archive is
[`../artifacts/colab_source.zip`](../artifacts/colab_source.zip), with
[`../artifacts/colab_source.zip.sha256`](../artifacts/colab_source.zip.sha256)
as its checksum sidecar.

## 5. Verification performed

- `python -m unittest discover -s tests -v`: **8 tests passed**.
- Python source compilation passed.
- `uv pip check`: all installed packages compatible.
- Notebook JSON and code cells, runtime YAML, and result JSON parsed.
- Refreshed source ZIP SHA-256 matched its sidecar; expected scripts and
  configuration were included and raw/cache/model artifacts excluded.
- Local model smoke was rerun after extending it to check target+draft
  co-residency at both 256 and 1,024 tokens; both passed.
- Colab VM was stopped and a final CLI check reported no active sessions.

## 6. Explicit boundaries for the next agent

Do not treat any Phase 0/1 feasibility result as evidence that the model is
clinically useful. In particular:

1. The LoRA run was a synthetic mechanics/memory pilot, **not** training on
   the TrialGPT dataset.
2. No patient examples, protected test examples, or held-out labels were used
   for these model-selection smoke tests.
3. There is no trained adapter to resume or deploy from this pilot.
4. Do not change the pinned model revisions silently. Any replacement must
   repeat tokenizer, local inference-envelope, and research-fit checks.
5. Any next checklist should cite the master plan and this handoff, mark only
   evidence-backed items complete, and retain later-phase work as open rather
   than expanding this handoff's completion claim.

## 7. Recommended next use

Use this collective handoff with the full
[`../output/master_plan.md`](../output/master_plan.md) to create a detailed,
traceable checklist for Phase 0 and Phase 1. Treat the two phase-specific
handoffs and the linked reports/scripts as supporting evidence. The following
phases start with the actual dataset/training pipeline and must maintain the
patient-level split and held-out-test protections described in the master
plan.
