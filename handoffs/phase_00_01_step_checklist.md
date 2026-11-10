> **Compute-route update (10 November 2026):** this file records historical Colab evidence. The active route is **Google Colab**; see [Colab training setup](../docs/colab_manual_setup.md). This checklist records earlier environment evidence; verify CUDA in the current training session.

# Phase 0 and Phase 1: detailed step checklist

**Audit basis:** the complete [master plan](../output/master_plan.md), especially its Phase 0 and Phase 1 sections; the [collective handoff](phase_00_01_collective.md); the [Phase 0 handoff](phase_00.md); the [Phase 1 handoff](phase_01.md); and the linked saved artifacts. The collective handoff describes the environment and model-feasibility work as complete. The item-level statuses below apply the plan's more detailed requirements and identify missing or narrower evidence; they do not silently promote a summary claim to full acceptance.

**Status key:** `[x]` Complete = directly evidenced; `[~]` Partial = some evidence, with a stated gap; `[ ]` Open = not done or not evidenced; `[N/A]` Not applicable = conditional instruction did not apply, with a reason. IDs are stable within each original step; output, verification and gate IDs use `OUT`, `VER` and `GATE`.

**Scope:** the T4 LoRA pilot and inference checks used synthetic data and establish execution/memory feasibility only. They are not TrialGPT training, clinical performance, a trained deployment adapter, or final held-out evaluation. Later-phase deliverables remain outside this checklist and are never inferred from these pilots.

## Phase 0 - Establish the environment and project contract

### Step 1 - Deadline, demo and rubric

- [ ] **P0-01a** Confirm the exact Google Classroom deadline time and timezone. The owner must check Classroom; [requirements](../research/requirements.md) records only the assignment date and the unresolved confirmation.
- [ ] **P0-01b** Confirm live-demo arrangements and attendance with the relevant course channel. The [Phase 0 handoff](phase_00.md) explicitly leaves this to the user.
- [x] **P0-01c** Record the interpretation of the rubric's 100-versus-135-mark discrepancy. [Requirements](../research/requirements.md) lists the category totals and does not assume normalization.

### Step 2 - Project structure, ignore rules and repository

- [x] **P0-02a** Create the Phase 0/1 project scaffold and supporting directories. Evidence: [README](../README.md), [configs](../configs/), [scripts](../scripts/), [source package](../src/clinical_match/) and [handoffs](./). Later-phase application/report directories are not claimed complete.
- [x] **P0-02b** Ignore the project environment and caches. [`.gitignore`](../.gitignore) covers `.venv`, `.cache` and Python/tool caches.
- [~] **P0-02c** Ignore raw/processed data, model weights and large temporary artifacts. [`.gitignore`](../.gitignore) covers specific Parquet/JSONL/CSV paths, artifact/checkpoint directories and weight extensions; it does not explicitly cover every possible raw file format or generic temporary-file location.
- [~] **P0-02d** Protect private inputs and credentials from publication. [`.gitignore`](../.gitignore) excludes `.env` and selected data files, and [README](../README.md) instructs users not to expose tokens; no explicit broad patient-upload/private-input rule or publication audit is evidenced.
- [x] **P0-02e** Initialize the local Git repository if needed. The current workspace has `.git`; the [collective handoff](phase_00_01_collective.md) and local project files document the repository scaffold. No remote publication is implied.

### Step 3 - Local and Colab runtime setup

- [x] **P0-03a** Create an isolated Python 3.11 environment on D:. [Local environment report](../results/environments/local.json) records Python 3.11.17 at `D:\Gen_AI_Assignment_2\.venv`.
- [x] **P0-03b** Put local model, pip and uv caches on D:. [Local environment report](../results/environments/local.json) records all three cache locations.
- [x] **P0-03c** Install a local CUDA-enabled PyTorch build and exercise the GPU. [Local environment report](../results/environments/local.json) records PyTorch 2.11.0+cu128 and passed CUDA smoke tests.
- [x] **P0-03d** Select an actual Colab GPU runtime and record its allocation, Python and PyTorch. [Colab environment report](../results/environments/colab_20261109T110002Z.json) records a Tesla T4, Python 3.13.16 and PyTorch 2.14.1+cu130.
- [x] **P0-03e** Retain compatible preinstalled Colab Torch. [Colab requirements](../requirements-colab.txt) does not reinstall Torch; the [Phase 0 handoff](phase_00.md) records that the existing build was retained.
- [N/A] **P0-03f** Time-box unavailable-GPU troubleshooting to 20-30 minutes. A T4 was allocated and tested; no unavailable-GPU troubleshooting episode is evidenced in the [collective handoff](phase_00_01_collective.md).
- [~] **P0-03g** Retain a local fallback route if Colab becomes unavailable. [Local runtime profile](../configs/runtime_local.yaml), [setup](../scripts/setup_local.cmd) and [local inference report](../results/feasibility_local.json) show a local route; a full local training-fit/resume run is not evidenced.

### Step 4 - Portable environment doctor and actual smoke tests

- [x] **P0-04a** Implement the portable environment doctor. [`doctor.py`](../scripts/doctor.py) and the two saved environment reports show the implementation and use.
- [x] **P0-04b** Record OS, Python and package versions in each environment. [Local report](../results/environments/local.json); [Colab report](../results/environments/colab_20261109T110002Z.json).
- [x] **P0-04c** Record CPU RAM and disk capacity/free space in each environment. [Local report](../results/environments/local.json); [Colab report](../results/environments/colab_20261109T110002Z.json). Values are snapshots.
- [x] **P0-04d** Record GPU identity, VRAM and compute capability. [Local report](../results/environments/local.json); [Colab report](../results/environments/colab_20261109T110002Z.json).
- [x] **P0-04e** Record CUDA build and dtype capability information. [Local report](../results/environments/local.json); [Colab report](../results/environments/colab_20261109T110002Z.json). The reported BF16 flag is a runtime-reported value, not a tested native-BF16 training result.
- [x] **P0-04f** Run a GPU matrix multiplication in the local environment. [Local report](../results/environments/local.json) records `matrix_multiply.passed: true`.
- [x] **P0-04g** Run a tiny Transformer forward pass in the local environment. [Local report](../results/environments/local.json) records `tiny_forward.passed: true`.
- [x] **P0-04h** Run a GPU matrix multiplication in Colab. [Colab report](../results/environments/colab_20261109T110002Z.json) records `matrix_multiply.passed: true`.
- [x] **P0-04i** Run a tiny Transformer forward pass in Colab. [Colab report](../results/environments/colab_20261109T110002Z.json) records `tiny_forward.passed: true`.
- [x] **P0-04j** Save separate local and run-specific Colab environment reports. [Local](../results/environments/local.json); [Colab](../results/environments/colab_20261109T110002Z.json).
- [x] **P0-04k** Save the primary research environment pointer and local summary. [Environment pointer](../results/environment.json) links both reports.

### Step 5 - Dependency locks, runtime profiles and cache API

- [x] **P0-05a** Pin shared library versions. [Shared requirements](../requirements-base.txt) pins Transformers, PEFT, NumPy and other packages; [environment reports](../results/environment.json) identify the tested runtimes.
- [x] **P0-05b** Save the tested Windows environment lock. [Local lock](../requirements-local.lock); [Phase 0 handoff](phase_00.md).
- [~] **P0-05c** Save a tested Colab/Linux environment-specific lock. [Colab requirements](../requirements-colab.txt) imports shared pins and deliberately retains preinstalled Torch; [Colab report](../results/environments/colab_20261109T110002Z.json) records selected package versions, but a complete tested Linux lock is not present.
- [x] **P0-05d** Create the Colab runtime profile. [Colab profile](../configs/runtime_colab.yaml) records `/content`, cache and artifact locations. Its `gpu_allocation_verified: false` is stale relative to the saved T4 report and should not be mistaken for current session availability.
- [x] **P0-05e** Create the local runtime profile. [Local profile](../configs/runtime_local.yaml) records local paths, CUDA, precision and batch settings.
- [~] **P0-05f** Make all source paths configurable. [Runtime profiles](../configs/) provide configurable paths, but the [Colab notebook](../notebooks/colab_runner.ipynb) hardcodes paths, including a requirements path inconsistent with its hash-suffixed active directory.
- [~] **P0-05g** Inspect the installed cache API. [Model smoke code](../scripts/smoke_models.py) uses returned `past_key_values` and `get_seq_length()`; a separate documented inspection of the installed cache class/API is not evidenced. Full cache-flow implementation remains Phase 6.
- [x] **P0-05h** Keep Windows CUDA-wheel assumptions out of Colab. [Local setup](../scripts/setup_local.cmd) uses a Windows CUDA wheel source; [Colab requirements](../requirements-colab.txt) retains the existing Torch installation.
- [x] **P0-05i** Use the actual Colab Python instead of forcing local Python 3.11. [Colab report](../results/environments/colab_20261109T110002Z.json) records Python 3.13.16; [local report](../results/environments/local.json) records 3.11.17.

### Step 6 - Colab runner, persistence and synthetic recovery

- [x] **P0-06a** Create a thin interactive Colab notebook using repository scripts rather than embedding the training implementation. [Colab runner](../notebooks/colab_runner.ipynb) calls the doctor, tokenizer check, model smoke and fit pilot scripts.
- [x] **P0-06b** Verify the source archive before extracting the project. [Colab runner](../notebooks/colab_runner.ipynb) checks the SHA-256 sidecar and source archive paths; [source archive sidecar](../artifacts/colab_source.zip.sha256) records the archive hash.
- [~] **P0-06c** Provide a functioning notebook bootstrap/dependency-install sequence. The current [Colab runner](../notebooks/colab_runner.ipynb) copies the verified source into its hash-suffixed `ACTIVE_DIR` and installs requirements from that directory. The full notebook path still needs a fresh end-to-end run.
- [x] **P0-06d** Include a doctor cell. [Colab runner](../notebooks/colab_runner.ipynb) invokes [`doctor.py`](../scripts/doctor.py) and writes a run-specific report.
- [x] **P0-06e** Include training runner cells. [Colab runner](../notebooks/colab_runner.ipynb) now invokes the full Phase 4 trainer and includes verification, overfit/resume rehearsal, training and export cells. Downstream adapter-specific evaluation and benchmark reruns remain later-phase work.
- [x] **P0-06f** Stage active code and model cache under `/content`. [Colab runner](../notebooks/colab_runner.ipynb), [Colab profile](../configs/runtime_colab.yaml) and [Colab environment report](../results/environments/colab_20261109T110002Z.json) show active workspace/cache paths.
- [~] **P0-06g** Persist verified notebook bundles to a chosen Drive/project folder. [Colab runner](../notebooks/colab_runner.ipynb) contains a Drive-backed bundle cell; the [collective handoff](phase_00_01_collective.md) evidences a CLI-downloaded bundle restored locally, not an end-to-end execution of that notebook cell against mounted Drive.
- [x] **P0-06h** Implement generic checksum-verified snapshot/export helpers. [Checkpoint helpers](../src/clinical_match/checkpointing.py) create, verify and restore bundles; [tests](../tests/test_checkpointing.py) cover recovery and integrity checks.
- [x] **P0-06i** Include a snapshot/restore cell. [Colab runner](../notebooks/colab_runner.ipynb) creates a verified bundle and checks a restored copy; execution of this exact notebook path remains unverified as noted in P0-06g.
- [x] **P0-06j** Run a synthetic save/verify/restore test. [Checkpoint tests](../tests/test_checkpointing.py) contain a synthetic round trip; the [collective handoff](phase_00_01_collective.md) records the tests passing and the cloud report bundle being verified/restored locally.
- [~] **P0-06k** Integrate the 25-step/5-minute checkpoint schedule with actual task training. The trainer implements periodic and epoch checkpoints, and local resume rehearsal passes; a full TrialGPT Colab run remains outstanding.
- [~] **P0-06l** Keep secrets out of source and notebook output. [Colab runner](../notebooks/colab_runner.ipynb) avoids requesting/printing secrets and [README](../README.md) warns against committing tokens; no comprehensive secret scan or published-repository audit is evidenced.

### Step 7 - Windows launch design and source archive

- [x] **P0-07a** Provide a local Windows setup launcher. [Setup_Local.cmd](../Setup_Local.cmd) delegates to [local setup](../scripts/setup_local.cmd); [README](../README.md) documents it.
- [x] **P0-07b** Design a local evaluator-facing `.cmd` app launcher without requiring Colab. [Start_App.cmd](../Start_App.cmd) targets a local Python/app path; [README](../README.md) states it is a Phase 10 placeholder.
- [ ] **P0-07c** Verify that an evaluator can actually double-click and use the browser app without Colab. [Start_App.cmd](../Start_App.cmd) explicitly reports that the app is not yet implemented; the working-app test belongs to Phase 10.
- [x] **P0-07d** Record a source archive hash while publication is not ready. [Source archive](../artifacts/colab_source.zip) and [SHA-256 sidecar](../artifacts/colab_source.zip.sha256) agree; the [collective handoff](phase_00_01_collective.md) records the archive verification.

### Phase 0 outputs

- [x] **P0-OUTa** Shared dependency pins: [requirements-base.txt](../requirements-base.txt).
- [~] **P0-OUTb** Two tested environment locks/profiles: [local lock](../requirements-local.lock), [local profile](../configs/runtime_local.yaml), [Colab profile](../configs/runtime_colab.yaml), [Colab requirements](../requirements-colab.txt); the full Colab-specific lock remains missing.
- [~] **P0-OUTc** `.gitignore` and source skeleton: [ignore rules](../.gitignore), [source](../src/clinical_match/), [scripts](../scripts/); broad private-input and generic temporary-file rules are not evidenced.
- [x] **P0-OUTd** Colab launcher and Phase 4 training workflow: [notebook](../notebooks/colab_runner.ipynb) includes the current bootstrap and full training/export sequence. Its full mounted-Drive path has not yet been rerun after recent edits; see the Phase 4 Colab handoff.
- [x] **P0-OUTe** Local/Colab environment reports and summary: [local](../results/environments/local.json), [Colab](../results/environments/colab_20261109T110002Z.json), [pointer](../results/environment.json).
- [x] **P0-OUTf** Generic checkpoint/export helpers and synthetic test: [helpers](../src/clinical_match/checkpointing.py), [tests](../tests/test_checkpointing.py), [restored bundle](../results/colab_verified_bundle/).
- [x] **P0-OUTg** Setup notes: [README](../README.md).
- [x] **P0-OUTh** Phase-specific handoff: [phase_00.md](phase_00.md).

### Phase 0 verification criteria

- [x] **P0-VERa** Run the doctor locally and save a report. [Local report](../results/environments/local.json) records both CUDA smoke tests; [README](../README.md) documents the `--smoke-test` command.
- [x] **P0-VERb** Run the doctor in Colab and save a report. [Colab report](../results/environments/colab_20261109T110002Z.json) records both CUDA smoke tests; [collective handoff](phase_00_01_collective.md) identifies the saved run.
- [x] **P0-VERc** Confirm local caches are on D:. [Local report](../results/environments/local.json) records the Hugging Face, pip and uv paths.
- [x] **P0-VERd** Confirm Colab active files/cache are under `/content`. [Colab report](../results/environments/colab_20261109T110002Z.json), [Colab profile](../configs/runtime_colab.yaml) and [notebook](../notebooks/colab_runner.ipynb).
- [x] **P0-VERe** Restore a synthetic persistent bundle with matching hashes. [Checkpoint tests](../tests/test_checkpointing.py) and [collective handoff](phase_00_01_collective.md); the [restored cloud bundle](../results/colab_verified_bundle/) supplies separate transferred-report evidence.
- [N/A] **P0-VERf** Record an unavailable cloud GPU if encountered. The saved [Colab report](../results/environments/colab_20261109T110002Z.json) shows an available T4; this conditional failure case was not encountered in the recorded run.
- [x] **P0-VERg** Treat doctor checks as setup evidence, not training fit. [Phase 0 handoff](phase_00.md) describes GPU smoke separately from the synthetic model pilot in [Phase 1](phase_01.md).

### Phase 0 done-when gate and summary

- [~] **P0-GATEa** Local setup is reproducible. [Local setup script](../scripts/setup_local.cmd), [local lock](../requirements-local.lock) and [local report](../results/environments/local.json) document a working installation; a separate fresh-machine/fresh-environment reproduction is not recorded.
- [~] **P0-GATEb** Colab bootstrap and persistence are tested when available. [Colab report](../results/environments/colab_20261109T110002Z.json) and [verified bundle](../results/colab_verified_bundle/) show CLI-based runtime and transfer; the notebook's full bootstrap/Drive-cell path has not been verified and contains the path mismatch above.
- [x] **P0-GATEc** State the primary research route and local fallback. [Decision log](../research/decision_log.md) selects Colab research and local app; [local runtime profile](../configs/runtime_local.yaml) records the fallback environment. Full local training feasibility remains separately partial under P0-03g.
- [N/A] **P0-GATEd** Apply the denied-cloud-allocation exception if necessary. The recorded [Colab T4 report](../results/environments/colab_20261109T110002Z.json) shows an allocation, so the exception was not invoked.
- [~] **P0-GATEe** Complete Phase 0 without depending on a future module. Core doctor/persistence evidence exists, but the notebook launcher and tested environment-lock gaps above remain; the actual browser app is explicitly a Phase 10 deliverable.

**Phase 0 status:** The collective handoff supports completed local/Colab GPU and synthetic-bundle feasibility checks. Under the full step-by-step plan, the exact deadline and live-demo arrangements are **open**, and the cloud lock, notebook path/coverage, broad ignore coverage and fresh setup reproduction remain **partial**. The saved runtime is historical; it does not guarantee a future free allocation. No training or application-completion claim is made.

## Phase 1 - Research and choose the model pair

### Step 1 - Research and source summaries

- [~] **P1-01a** Read at least five credible technical/scholarly sources. [Source notes](../research/sources.md) summarize more than five cited sources, but they explicitly say the notes are not a substitute for reading the sources; independent complete reading by the student is not evidenced.
- [x] **P1-01b** Summarize at least five sources in the project's own words. [Source notes](../research/sources.md) contains paraphrased summaries and source links across the required topics.
- [x] **P1-01c** Cover clinical matching and criterion/evidence context. [Source notes](../research/sources.md) includes the TrialGPT paper and official repository.
- [x] **P1-01d** Cover exact model cards/configuration and decoder architecture. [Source notes](../research/sources.md) links the Qwen technical report and pinned configs; [candidate comparison](../research/model_comparison.md) records both checkpoints.
- [x] **P1-01e** Cover LoRA/quantized adaptation. [Source notes](../research/sources.md) summarizes LoRA and QLoRA with limitations and design implications.
- [x] **P1-01f** Cover KV caching. [Source notes](../research/sources.md) links the Transformers cache documentation and its application to this project.
- [x] **P1-01g** Cover tiled/online-softmax attention and FlashAttention. [Source notes](../research/sources.md) discusses Dao et al. and Rabe/Staats.
- [x] **P1-01h** Cover speculative decoding. [Source notes](../research/sources.md) cites Leviathan et al. and distinguishes the proposed greedy experiment.
- [x] **P1-01i** Record each listed source's claim, relevance to the project, caution/limitation, implementation consequence and citation. [Source notes](../research/sources.md) contains those elements in its source, claim, caution and design-consequence columns.

### Step 2 - Compare the two decoder candidates

- [x] **P1-02a** Identify both open-weight decoder candidates and their exact revisions. [Candidate comparison](../research/model_comparison.md); [model config](../configs/model.yaml).
- [x] **P1-02b** Compare architectures and parameter counts. [Candidate comparison](../research/model_comparison.md) records layer/head/hidden dimensions and counts for both.
- [x] **P1-02c** Compare tokenizer information. [Local tokenizer report](../results/tokenizer_compatibility_local.json) records tokenizer classes and equality checks; [candidate comparison](../research/model_comparison.md) explains the requirement.
- [x] **P1-02d** Compare model-specific context lengths. [Candidate comparison](../research/model_comparison.md) reports 32,768 positions for both; [local tokenizer report](../results/tokenizer_compatibility_local.json) preserves config metadata.
- [x] **P1-02e** Record licenses for both candidates. [Candidate comparison](../research/model_comparison.md) identifies Apache-2.0 metadata for each.
- [x] **P1-02f** Estimate expected model memory and state overhead limitations. [Candidate comparison](../research/model_comparison.md) records raw FP16 lower bounds and the additional cache/activation/workspace costs.
- [x] **P1-02g** Explain why an encoder-only BERT-style model does not directly satisfy the required autoregressive generation experiments. [Candidate comparison](../research/model_comparison.md) has a dedicated explanation.

### Step 3 - Pinned tokenizer/config compatibility

- [x] **P1-03a** Fetch pinned target and draft configs/tokenizers in the local D: cache. [Local tokenizer report](../results/tokenizer_compatibility_local.json) records revisions; [local environment report](../results/environments/local.json) records the D: Hugging Face cache.
- [x] **P1-03b** Fetch pinned target and draft configs/tokenizers in Colab's `/content` cache. [Colab tokenizer report](../results/tokenizer_compatibility_colab.json) records revisions; [Colab environment report](../results/environments/colab_20261109T110002Z.json) records the cache path.
- [x] **P1-03c** Compare complete vocabulary-to-ID mappings. Both [local](../results/tokenizer_compatibility_local.json) and [Colab](../results/tokenizer_compatibility_colab.json) report `vocabulary_mapping_equal: true`; [comparison code](../scripts/check_tokenizers.py) hashes the sorted full mappings.
- [x] **P1-03d** Compare added vocabularies. [Local](../results/tokenizer_compatibility_local.json) and [Colab](../results/tokenizer_compatibility_colab.json) reports record equality.
- [x] **P1-03e** Compare special-token IDs. [Local](../results/tokenizer_compatibility_local.json) and [Colab](../results/tokenizer_compatibility_colab.json) reports record equality.
- [x] **P1-03f** Compare representative synthetic encodings and chat templates. [Local](../results/tokenizer_compatibility_local.json) and [Colab](../results/tokenizer_compatibility_colab.json) reports record both checks passing.
- [~] **P1-03g** Use identical target-formatted prompt IDs for the actual target/draft speculative workflow after compatibility is proven. [Tokenizer comparison](../scripts/check_tokenizers.py) verifies matching synthetic prompt encodings, but the actual shared-prompt speculative workflow is a Phase 8 implementation and has not been demonstrated here.

### Step 4 - Research-GPU inference and LoRA fit pilot

- [x] **P1-04a** Run target greedy-next-token inference at 256 input tokens on the research GPU. [Colab feasibility report](../results/feasibility_colab.json) records the T4 forward and one-token cached decode passing.
- [x] **P1-04b** Run target greedy-next-token inference at 1,024 input tokens on the research GPU. [Colab feasibility report](../results/feasibility_colab.json) records the T4 check passing.
- [x] **P1-04c** Run a 10-optimizer-step LoRA fit pilot on the research GPU. [T4 pilot report](../results/fit_pilot_colab.json) records `steps: 10` and `passed: true`.
- [x] **P1-04d** Start with FP16 target weights. [T4 pilot report](../results/fit_pilot_colab.json) records `torch.float16`; [pilot code](../scripts/fit_pilot.py) specifies FP16.
- [x] **P1-04e** Use microbatch 1 and accumulation 8. [Pilot code](../scripts/fit_pilot.py) creates a batch-one tensor; [T4 pilot report](../results/fit_pilot_colab.json) records `gradient_accumulation: 8`.
- [~] **P1-04f** Exercise the proposed 1,024-token training envelope. [Pilot report](../results/fit_pilot_colab.json) records a 1,024-position tensor, but [pilot code](../scripts/fit_pilot.py) pads one short synthetic example to that length. This is not a measured fit for a full 1,024-token non-padding patient/criterion example.
- [x] **P1-04g** Enable gradient checkpointing for the pilot. [Pilot code](../scripts/fit_pilot.py) calls `gradient_checkpointing_enable()`; [T4 pilot report](../results/fit_pilot_colab.json) records its successful run.
- [x] **P1-04h** Record measured VRAM, optimizer-step timing and finite losses. [T4 pilot report](../results/fit_pilot_colab.json) contains allocated/reserved peaks, step times and `all_losses_finite: true`.
- [x] **P1-04i** Avoid enabling BF16 without a supported/verified path. [Pilot code](../scripts/fit_pilot.py) uses FP16, and [T4 pilot report](../results/fit_pilot_colab.json) confirms the run dtype. No native-BF16 claim is needed for this pilot.
- [~] **P1-04j** Estimate the eventual run cost/time from measured pilot throughput. [T4 pilot report](../results/fit_pilot_colab.json) records approximately 4.87 seconds per step, but no saved total training-time or monetary-cost estimate for the real dataset is evidenced.
- [x] **P1-04k** Use synthetic/development fixtures only for the fit test. [T4 pilot report](../results/fit_pilot_colab.json) states `synthetic_only: true`; [pilot code](../scripts/fit_pilot.py) constructs a fictional example.

### Step 5 - Laptop inference, pair co-residency and separate fit records

- [x] **P1-05a** Test target-only inference locally. [Local feasibility report](../results/feasibility_local.json) records passed synthetic checks at 256 and 1,024 input tokens.
- [x] **P1-05b** Test simultaneous target/draft GPU residency locally. [Local feasibility report](../results/feasibility_local.json) records `pair_fit_passed: true` at both input lengths.
- [~] **P1-05c** Test the complete intended app-generation envelope locally. [Local feasibility report](../results/feasibility_local.json) records target/draft inference and a one-token cached decode up to 1,024 input tokens; the [local profile](../configs/runtime_local.yaml) proposes up to 96 new tokens, and a full app/speculative run at that output length is not evidenced.
- [x] **P1-05d** Distinguish target-only training from target-plus-draft speculative inference. [Decision log](../research/decision_log.md) identifies the target, smaller draft and separate training/inference roles; [pilot code](../scripts/fit_pilot.py) loads the target only.
- [x] **P1-05e** Record local and cloud fit separately. [Local feasibility](../results/feasibility_local.json), [Colab feasibility](../results/feasibility_colab.json) and [T4 pilot](../results/fit_pilot_colab.json) are separate reports.
- [x] **P1-05f** Record the actual cloud pair test's limited scope. [Colab feasibility](../results/feasibility_colab.json) and [Phase 1 handoff](phase_01.md) show pair co-residency at 256 tokens; no cloud pair-at-1,024 result is claimed.
- [N/A] **P1-05g** Choose a revised model or validated deployment variant if the tested pair fits in cloud but fails locally. [Local feasibility report](../results/feasibility_local.json) records no local OOM for the tested one-token-decode envelope; the untested full app envelope remains P1-05c.

### Step 6 - Freeze the provisional design and document alternatives

- [x] **P1-06a** Record the selected pinned target/draft and FP16 precision. [Model config](../configs/model.yaml) identifies both revisions, roles and precision.
- [~] **P1-06b** Freeze the provisional prompt/schema. [Model config](../configs/model.yaml) records `v1-provisional`, label subsets and evidence-ID output; an exact versioned/frozen clinical prompt template is not present. [Pilot code](../scripts/fit_pilot.py) contains only a synthetic prompt.
- [~] **P1-06c** Freeze the source ledger. [Source notes](../research/sources.md) exists with citations, but no immutable commit/hash for that ledger is recorded; its final line still says tokenizer equality must be run, while [local tokenizer evidence](../results/tokenizer_compatibility_local.json) shows it was run.
- [x] **P1-06d** Record alternatives and rejected/selected options. [Decision log](../research/decision_log.md) compares compute routes, model families, adaptation methods and hosting choices.
- [x] **P1-06e** Record observed issues and Colab/free-compute constraints. [Collective handoff](phase_00_01_collective.md) documents the pilot command timeout, saved-result verification, stopped session and usage uncertainty; [decision log](../research/decision_log.md) records the compute-unit observation.

### Phase 1 outputs

- [x] **P1-OUTa** Research source ledger: [research/sources.md](../research/sources.md).
- [x] **P1-OUTb** Model comparison: [research/model_comparison.md](../research/model_comparison.md). The document contains an older "not proven" sentence alongside later measured fit evidence; rely on its later evidence and the reports for the tested envelope.
- [x] **P1-OUTc** Technical decision log: [research/decision_log.md](../research/decision_log.md).
- [x] **P1-OUTd** Pinned model and provisional schema configuration: [configs/model.yaml](../configs/model.yaml); exact prompt-template freeze remains partial under P1-06b.
- [x] **P1-OUTe** Tokenizer compatibility evidence: [local report](../results/tokenizer_compatibility_local.json), [Colab report](../results/tokenizer_compatibility_colab.json), [comparison script](../scripts/check_tokenizers.py).
- [~] **P1-OUTf** Feasibility report at the planned `results/feasibility.json` path. Separate [local](../results/feasibility_local.json), [Colab](../results/feasibility_colab.json) and [fit-pilot](../results/fit_pilot_colab.json) reports exist, but the specifically named aggregate file is absent.
- [x] **P1-OUTg** Phase-specific handoff: [phase_01.md](phase_01.md).

### Phase 1 verification criteria

- [x] **P1-VERa** Verify selected target forward logits. [Smoke code](../scripts/smoke_models.py) reads target logits; [local](../results/feasibility_local.json) and [Colab](../results/feasibility_colab.json) reports show passing checks.
- [x] **P1-VERb** Verify model forward accepts and returns a KV cache. [Smoke code](../scripts/smoke_models.py) passes `past_key_values`, reads returned cache and checks its length; [local report](../results/feasibility_local.json) records the one-token results. The full custom cache decoder remains Phase 6.
- [x] **P1-VERc** Verify the target supports a LoRA training path. [Pilot code](../scripts/fit_pilot.py) attaches LoRA and runs optimization; [T4 pilot report](../results/fit_pilot_colab.json) records success on synthetic input.
- [x] **P1-VERd** Verify the draft is strictly smaller. [Candidate comparison](../research/model_comparison.md) lists 494,032,768 draft versus 1,543,714,304 target base parameters.
- [x] **P1-VERe** Verify token compatibility rather than relying on matching vocabulary size. [Local](../results/tokenizer_compatibility_local.json) and [Colab](../results/tokenizer_compatibility_colab.json) reports record mapping, added-token, special-ID and synthetic-encoding checks.
- [~] **P1-VERf** Verify the chosen runtimes fit the measured envelope. [Local](../results/feasibility_local.json), [Colab](../results/feasibility_colab.json) and [T4 pilot](../results/fit_pilot_colab.json) support their tested inputs and synthetic workloads; full 96-token app generation and a full non-padding 1,024-token training example remain unmeasured.
- [x] **P1-VERg** Treat the model pair as selected only after the recorded compatibility and fit checks, and limit the claim to the measured envelope. [Model config](../configs/model.yaml), [Phase 1 handoff](phase_01.md) and the linked feasibility reports document that scope.

### Phase 1 done-when gate and summary

- [x] **P1-GATEa** Support the model choice with recorded tokenizer, local/cloud inference and synthetic LoRA feasibility evidence. [Phase 1 handoff](phase_01.md); [model config](../configs/model.yaml); [local feasibility](../results/feasibility_local.json); [T4 pilot](../results/fit_pilot_colab.json).
- [x] **P1-GATEb** Support the selected precision with evidence. [Model config](../configs/model.yaml), [local feasibility](../results/feasibility_local.json) and [T4 pilot](../results/fit_pilot_colab.json) record FP16.
- [x] **P1-GATEc** Support the selected research and local-application compute route with recorded environments. [Decision log](../research/decision_log.md); [environment pointer](../results/environment.json); [local feasibility](../results/feasibility_local.json).
- [~] **P1-GATEd** Establish that no paid service is necessary. The [collective handoff](phase_00_01_collective.md) records a T4 pilot with zero reported balance but a nonzero displayed usage rate and cautions against assuming future free access. No paid API is required by the selected design, but durable availability and a complete demonstrated no-paid-service training route are not established.
- [x] **P1-GATEe** Avoid held-out patient cases during research and model smoke tests. [Collective handoff](phase_00_01_collective.md) explicitly records no patient/protected test examples in these pilots; [pilot code](../scripts/fit_pilot.py) constructs a synthetic example.
- [~] **P1-GATEf** Satisfy every detailed Phase 1 requirement. The saved synthetic feasibility gates support model selection, but the individual reading confirmation, exact prompt freeze, total pilot-derived run estimate, planned aggregate report, full-length non-padding training fit and complete local app output envelope are not fully evidenced above.

**Phase 1 status:** The collective handoff's **synthetic model-selection feasibility** claim is supported for the recorded T4 and RTX 3050 checks. Detailed acceptance remains **partial** for the documented gaps. No TrialGPT supervised training, quality assessment, trained adapter, final test result, full app or speculative speedup is claimed by this checklist.
