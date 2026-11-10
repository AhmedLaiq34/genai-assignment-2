# Implementation Plan: Phases 4-6

**Purpose:** a stand-alone execution brief for implementing Phases 4, 5 and 6 of the GenAI Assignment #2 project. It is written so a coding agent can complete one phase at a time using this file and the named inputs, without relying on later phases. The assignment PDF is authoritative; where this plan makes a project-level choice, it identifies the evidence and limits.

**Current state (updated 10 November 2026):** Phase 3 is complete and the unchanged-model validation baseline is recorded. The Phase 4 trainer, checkpoint/resume path, validation selection and Colab runner is implemented and locally tested; full local training OOMs and the validation-selected Colab adapter is still outstanding. Phase 5 screening and Phase 6 explicit KV decoding are implemented and tested against the base model. Adapter-dependent reruns remain pending.

**Execution route:** implement and test portable code on the laptop; use **Google Colab** for Phase 4 training and GPU measurements. If the active Colab session has no usable CUDA, preserve its report and stop; record that real training and GPU measurements remain pending. A phase that requires a real model artifact or GPU result cannot be declared complete without that evidence.

## How to use this plan

Complete the phases in order because Phase 5 consumes the exported model selected in Phase 4, and Phase 6 should benchmark the inference path the application will actually use. Each phase has a bounded scope, exact inputs, outputs, checks and a gate. Do not begin the next phase by assuming a failed or missing gate passed. Record material changes to defaults in the decision log, along with their reason, evidence and limitations.

The three phases target these assignment rubric items:

| Phase | Rubric | What earns evidence |
|---|---|---|
| 4: fine-tune and select | B3-B4; A3 | Executable supervised adaptation, reproducible/resumable training, validated checkpoint, same-model before/after comparison and documented choices. |
| 5: traceable screening | C4; G2 groundwork | Criterion-level workflow, exact source excerpts, conservative explanations, preliminary aggregation and an honest development error ledger. |
| 6: KV-cached generation | D1-D3 | Explicit cache-aware decoder, greedy-token equivalence against the uncached reference, and a controlled timing/memory comparison. |

The assignment's total marks discrepancy (heading says 100; detailed criteria total 135) is already recorded in the project. This plan does not guess how the instructor will normalize marks.

## Fixed facts and constraints for all three phases

- Dataset: `ncbi/TrialGPT-Criterion-Annotations`, at the pinned revision and source hash in [`configs/data.yaml`](../configs/data.yaml) and [`handoffs/phase_02.md`](../handoffs/phase_02.md).
- Preserve the canonical patient-grouped 37/8/8 split from [`data/splits/manifest.json`](../data/splits/manifest.json), seed 42. The source contains 1,015 annotations; the assigned split counts are 603/206/206. The approved malformed training annotation 883 has no criterion text and is excluded from supervised inputs; therefore the usable training set has 602 rows. Preserve its audit/quarantine record. Do not recover its text from labels, previous GPT output or trial metadata.
- The model may see only patient note/sentence map, criterion type and criterion text. Expert category and expert evidence IDs are targets only. Never use GPT-generated prediction, explanation, eligibility or sentence fields as input or target.
- Keep all six expert categories, with the type-specific allowed sets. Inclusion and exclusion meanings are not interchangeable. Preserve original patient sentence IDs.
- Validation is for development and checkpoint selection. The 206 test rows remain sealed until the explicitly frozen final evaluation phase (Phase 9 in the master plan). Do not open test text, labels or evidence to debug, choose a model, prompt, threshold or implementation.
- The baseline prompt is `v1-provisional`; the Phase 3 base run uses greedy decoding and `max_new_tokens=96`. Any prompt or generation change requires new hashes and a comparable baseline before using that change for selection. Keep base and adapted comparisons on the same validation rows, prompt, token budget, parser, metrics and generation settings.
- Target hypothesis: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; LoRA is the planned method. Treat model and precision as pinned starting choices, not proof of task quality. See [`configs/model.yaml`](../configs/model.yaml) and [`handoffs/phase_01.md`](../handoffs/phase_01.md).
- Laptop is Windows 11 with RTX 3050 Laptop GPU (6 GiB VRAM) and limited available system RAM at inspection. Colab GPU access is conditional; no accelerator or uninterrupted runtime is guaranteed. Use the actual runtime report and keep local and Colab performance results separate. Do not merge unlike-device timings into a speedup claim.
- Phase 3 macro-F1 0.1241 and evidence micro-F1 0.1549 are unchanged-model validation baseline results, not clinical performance claims. The baseline used a library cached `generate` path. On three examples its generated token sequences differed from the explicit uncached reference even though parsed outputs matched. This is not Phase 6 cache-equivalence evidence.
- Synthetic LoRA results show only that a short optimizer run was technically feasible on one historical Colab T4 session. They are not TrialGPT training, useful learning, validation improvement or clinical-performance evidence.

## Phase 4 - Fine-tune and select a validated checkpoint

### Objective and gate

Train the selected target on the 602 usable rows in the fixed training split, select a checkpoint using validation only, compare it fairly with the unchanged target, and export an adapter that can be reloaded locally without a live Colab session. This phase provides the trained artifact needed by Phase 5.

**Inputs:** source tree and split manifest; prepared train/validation data; Phase 3 baseline predictions/metrics/config; model and prompt configs; current environment report; active Colab workflow in [`docs/colab_manual_setup.md`](../docs/colab_manual_setup.md) and the [Phase 4 training handoff](../handoffs/phase_04_colab_training_handoff.md). The Colab runner invokes the implemented trainer.

**Out of scope:** test evaluation, app UI, Phase 6 cache implementation, speculative decoding and attention experiments. Do not let this phase expand into a hyperparameter sweep.

### Decisions and why

1. **Use LoRA as the first method.** It fits the existing 1.5B target feasibility evidence and is a bounded way to adapt with limited hardware. It is not intrinsically superior, and the assignment permits other methods. Record actual trainable parameter count and memory. If the proposed settings cannot fit or train correctly, follow the fallback ladder below and record the changed run as a separate experiment.
2. **Train only on the canonical training patients.** Validation is used to select among epochs/configurations, not to update weights. Test remains sealed. This preserves the assignment protocol and makes the baseline comparison meaningful.
3. **Use the structured label-plus-sentence-ID target.** The source does not provide expert-authored explanations. Train only on expert labels and expert evidence IDs. Explanations are constructed later from cited original text.
4. **Keep an honest fallback.** If cloud allocation fails, continue to implement and verify the trainer locally. A feasibility failure may block completing a real training artifact, but does not justify calling synthetic work a trained model. Preserve the baseline and report the blocker rather than switching to a different dataset or claiming a fabricated result.

### Ordered implementation

#### 4.1 Freeze the run contract before writing the trainer

Create `configs/train.yaml` with explicit values for model ID/revision, tokenizer revision, prompt schema/revision and hash, train/validation manifest hashes, seed, optimizer, scheduler, LoRA modules/rank/alpha/dropout, precision, maximum sequence length, output cap, effective batch, epoch cap, checkpoint interval, validation generation settings and output directories. A run must serialize its fully resolved config, not just the YAML path.

Use the plan's starting point: LoRA rank 8, alpha 16, dropout 0.05, `q_proj`/`v_proj`, AdamW, learning rate 2e-4, weight decay 0.01, warmup ratio 0.05, gradient norm cap 1, per-device microbatch 1, gradient accumulation 8, up to 3 epochs, seed 42, gradient checkpointing on, and training KV cache off. These are hypotheses. Use FP16 with a scaler unless BF16 support and finite behavior are demonstrated on that runtime. Never silently switch precision, attention backend, sequence length or optimizer.

Before a real run, assert at startup that loaded IDs/hashes match the manifest, the training loader contains only the 602 usable training annotations, and no patient ID appears in validation. Log counts by criterion type and label so omissions are visible.

#### 4.2 Implement the executable training path

Implement the core trainer in importable project code and expose `scripts/train.py`; keep the notebook as a thin caller. Prefer explicit training code over opaque convenience wrappers where that makes labels, masks, optimizer steps and resume state easier to inspect.

For each example, apply the exact tokenizer chat template used for inference. Append a compact assistant JSON target containing the type-appropriate expert label and sorted unique original evidence IDs. Include the correct end-of-turn/end-of-sequence target token for the model template. Construct labels so every system/user/prompt token and padding token is `-100`; only assistant target tokens contribute causal next-token cross-entropy. Verify the causal shift and mask on a decoded example and assert that at least one assistant target token is supervised. Do not add a second label loss unless it is separately justified, logged and treated as a distinct run.

Implement deterministic patient/example ordering from the fixed seed, dynamic or documented padding, gradient accumulation with optimizer-step accounting, gradient clipping, warmup/scheduler accounting, periodic validation, structured logs and graceful interruption at optimizer-step boundaries. Confirm only intended LoRA parameters have `requires_grad=True`; save trainable and total parameter counts before training. Record token counts, loss, learning rate, gradient norms, optimizer step, epoch/step duration, allocated/reserved GPU memory, precision and skipped/overflow steps.

Smoke interface: `python scripts/train.py --config configs/train.yaml --smoke`. It must validate inputs and masks and exercise forward, loss, backward and one optimizer update on a small fixture without modifying the canonical full-run artifacts. A separate `--overfit-smoke` mode can use a fixed 8-16-row subset of the training split.

#### 4.3 Prove learning behavior before spending the full run

Overfit 8-16 deterministic training examples. Require finite loss across steps, a generally downward loss trend, and clear improvement in exact target reproduction on this tiny subset. Define and save the starting/end loss and exact structured-target match counts. Do not require perfect fit if the model's decoding/format is the limiting factor, but investigate any failure to learn. Stop before full training if labels are masked incorrectly, gradients are absent/nonfinite, generated targets do not match the serialization, or the update fails to improve the subset.

Measure seconds per optimizer step, peak memory and validation generation time. Estimate total time using the actual number of optimizer steps and measured validation/checkpoint overhead. Proceed only if there is time to finish a full run and persist/export its artifacts. Colab availability can change; verify the GPU in the active session and do not let a queued/stuck kernel consume repeated attempts.

#### 4.4 Build durable, verifiable resume before full training

Checkpoint on the earlier of every 25 optimizer steps or 5 minutes, and at epoch ends. Write each bundle atomically (temporary local directory, complete-file manifest, then finalized directory); a partial Drive copy must never be considered valid. A resumable bundle contains adapter/model-trainable state, optimizer, scheduler, scaler when used, global step/epoch, batch sampler permutation and cursor, Python/NumPy/PyTorch CPU RNG, CUDA RNG per device, resolved config, software/runtime snapshot, and hashes of all source inputs.

Stage the bundle under `/content`, calculate per-file and bundle SHA-256 values, copy it to the configured Drive artifact folder, then re-read and verify the copy. Keep the best validation adapter plus the two most recent verified resumable bundles; pruning may delete only project-owned checkpoints with a verified replacement. Never write credentials into source or notebook output.

Before full training, run a short interruption/resume rehearsal: train to a checkpoint, force a normal stop, restore from the durable Drive copy in a fresh process, and continue. Compare with an uninterrupted control run of the same tiny fixture/steps. The expected behavior is the same resumed data order and compatible optimizer/scheduler/global-step progression; document any nondeterminism and its source. A successful `save_pretrained` of adapter weights alone is not proof of resumability.

#### 4.5 Train, validate and select without test leakage

Run the full usable training split. At each epoch boundary save a candidate and generate validation outputs with the same frozen prompt, parser and greedy settings as the base run. Retain failed/malformed predictions; do not filter hard examples. Select using validation six-class macro-F1 as the primary criterion and evidence F1 as tie-breaker; also publish inclusion/exclusion metrics, per-class scores, confusion matrices, validation loss, parse failures, invalid IDs, truncations and runtime failures. State the ordering rule before reviewing results. If two candidates tie on both criteria, use lower validation loss; if still tied, choose the earlier/simpler checkpoint and record the rule.

Perform at most one pre-declared competing adaptation experiment if the measured budget allows (for example rank 8 vs rank 16 at the same optimizer-step/epoch budget, or learning rate 1e-4 vs 2e-4). Record it as a separate config/run; do not compare a tiny run to a full run as if it isolated one variable. Base-versus-adapter is compulsory regardless of this optional competitor. Do not continue an unbounded search.

Evaluate the selected adapter against the unchanged target on exactly the same 206 validation annotation IDs, with identical prompt IDs/settings wherever possible. Report absolute metrics and differences, not just the better score. A regression is a valid result. The system still needs the trained adapter artifact; do not quietly substitute the base model as the claimed fine-tuned system.

#### 4.6 Export and verify the selected artifact

Export the selected adapter separately from resumable optimizer bundles. Include an artifact manifest with model/tokenizer/prompt/dataset/split/config hashes, selected epoch/global step, selection reason and metric artifact hashes; library versions, actual Colab GPU/RAM/disk/precision; and checksums for every file. Download the bundle to the laptop and verify all hashes. In a fresh local process, load the pinned base and exported adapter without a cloud mount, run validation development inference, and compare predictions/metrics to the saved selected run. Document any numerical variation; investigate any parsing/category mismatch or unexplained result drift.

### Phase 4 outputs

- `configs/train.yaml` and resolved per-run configs.
- `scripts/train.py` and importable training/data-collation/checkpoint modules.
- Notebook cells that invoke the script and persist verified bundles, without duplicate training logic.
- Unit/integration tests for answer masks, causal target shifts, label constraints, deterministic ordering, intended trainable parameters, checkpoint integrity and resume.
- TrialGPT training history, validation prediction JSONL and reproducible metrics for each candidate; plotted loss/validation curves generated from the saved logs.
- Best adapter plus two verified recent resumable bundles (where the completed run produced them), exported model manifest and SHA-256 sidecars.
- Local fresh-process reload/parity report and completed `handoffs/phase_04.md`.

### Phase 4 verification and done gate

Required checks: training smoke passes; overfit smoke demonstrates learning; tests prevent prompt-label leakage and validation/test optimizer access; interrupted Drive checkpoint restore passes; full run records all expected steps/epochs or a documented non-recoverable failure; validation selection is reproducible from saved predictions; local artifact checksum/load/parity check passes. Use the existing `scripts/evaluate.py` and `scripts/verify_baseline.py` patterns or extend them without weakening existing protections.

**Done only when:** a real adapter trained on the permitted TrialGPT training examples exists locally and is checksummed; at least one validation-selected checkpoint is reloadable; base-versus-adapter validation comparison and training history are saved; resume behavior has evidence; the handoff names exact commands, hashes, software and runtime. If the Colab run cannot proceed, or the selected adapter cannot be downloaded/reloaded, report Phase 4 as blocked/incomplete and list what the user must run; do not mark the synthetic pilot as completion.

## Phase 5 - Make evidence and explanations traceable

### Objective and gate

Build a UI-independent criterion screening service that produces a stable result, resolves cited IDs to verbatim source sentences, explains only what the record supports, and calculates a plainly labeled preliminary patient-trial status. This is the core service that a later application UI will call.

**Inputs:** Phase 4 selected adapter manifest and local reload test; Phase 2 prepared schemas, sentence maps, prompt/parser and data contracts; Phase 3 metrics and validation predictions; current inference result schema. If the adapter artifact is not available, implement and test the service contract with a clearly marked fake model fixture or base-model development run. Such tests do not satisfy the Phase 4 dependency or constitute final adapter integration evidence.

**Out of scope:** app UI/export layout, final test analysis, medical advice, automatic clinical decisioning, and changes to training targets. The assignment needs a screening research prototype; aggregate output must not be described as ground-truth patient-trial eligibility.

### Decisions and why

1. **The model emits IDs; software retrieves quotes.** This guarantees displayed evidence is byte-for-byte sourced from the input record and avoids fabricated or paraphrased quotations.
2. **Separate prediction from operational quality flags.** `not enough information` and `not applicable` are valid expert categories. A JSON parse failure, invalid ID, truncation or runtime exception is not another medical category and must remain visible.
3. **Explain evidence limits explicitly.** A sentence ID provides traceability, not proof that the sentence entails the criterion. For empty citations, say no source sentence was cited and the rationale cannot be verified from citations. Do not invent a clinical explanation.
4. **Aggregation is a transparent policy layer.** Encode explicit precedence and retain per-criterion results. It is preliminary triage logic, not a separately labeled benchmark result or enrollment recommendation.

### Ordered implementation

#### 5.1 Define and lock the service interface

Create a service callable from Python without importing UI code. Input should include patient text (or original sentence records), one criterion's type and text, inference mode/model identity, and optional request metadata. Validate non-empty input, allowed criterion type, bounded text length, no unsupported structured fields, and safe handling of malformed sentence numbering. Do not change IDs by re-numbering. Use the existing prompt builder and deterministic tokenizer/template. Keep the same v1 prompt unless an intentional validation-only revision is documented and its baseline rerun.

Return the existing inference-result contract (extend/version it deliberately if needed) with at minimum: request/criterion identity; criterion type/text; raw generated text; parsed label; raw cited IDs; deduplicated valid IDs; parse status; evidence validity/duplicate/missing-ID flags; exact excerpts; truncation status and omitted sentence IDs/count; model and adapter revisions; prompt revision/hash; generation mode; timing; and operational error fields. No gold labels/evidence are included in a live service result. Preserve raw output even if parsing fails.

Treat model loading as an injected/owned service dependency with a single loaded model and serialized requests by default on the 6 GiB GPU. The service should load locally from configured artifact paths and never require Drive, cloud kernel globals or scratch paths. Avoid reloading weights for each criterion.

#### 5.2 Resolve evidence deterministically

Use the exact sentence map built from the original patient record. For each predicted ID, return the stored source text exactly, preserving punctuation, negation and original wording. Invalid or missing IDs must produce a visible validation flag and no excerpt. Duplicate IDs may be deduplicated for scoring/display but preserve raw occurrences and flag the duplication. Never let an ID from another patient or another criterion resolve against a global sentence map.

Keep the model's predicted category for evaluation even when its evidence IDs are invalid. The service can label the result as requiring review, but must not silently convert the category to unknown or a runtime failure. Store raw and validated evidence-ID lists separately.

#### 5.3 Add constrained explanation templates

Generate explanation text deterministically from the parsed result, category and resolved source sentences; do not ask the model to write an unconstrained rationale. Show the category and each exact excerpt. Use neutral wording such as “The model cites this sentence as evidence.” For `not enough information`, state that the provided record does not establish the criterion. For `not applicable`, identify that output without asserting a clinical reason absent from the record. For an empty citation set, explicitly state that no sentence was cited and the result cannot be verified from citations. If the record contains contradiction, show both cited statements and mark review rather than fabricating resolution.

Keep an explicit educational-prototype notice in service metadata so the eventual app can display it. Do not write advice, enrollment instructions or claims of medical suitability.

#### 5.4 Define preliminary aggregation in configuration

Add a versioned `configs/aggregation.yaml` and a pure aggregation function. For a set of criterion-level results:

1. No criteria supplied -> `no_assessment`.
2. Any inclusion `not included` or exclusion `excluded` -> `preliminary_criterion_failure` (while retaining any operational flags on that criterion).
3. Otherwise, any `not enough information`, `not applicable`, invalid category/evidence that prevents traceability, unresolved contradiction, truncation, parse failure or runtime failure -> `review_needed`.
4. Otherwise, if every required inclusion is `included` and every exclusion is `not excluded`, return `preliminary_match`.
5. Any unrecognized or internally inconsistent state -> `review_needed` and record the reason.

Clarify in config/docs how `not applicable` is handled. Because no external applicability rule is supplied, `not applicable` remains visible but forces `review_needed`; treating it as a pass would convert an uncertain or misused category into an unsupported match. Aggregation must preserve every underlying criterion result and reasons. It must never relabel this output as a gold patient-trial outcome or “eligible for enrollment.” Add tests for priority when a failure coexists with an unknown/invalid result; the result may remain failure while the caveat stays visible.

#### 5.5 Test fixtures and validation-only error analysis

Create deterministic fixtures for age, negation, family history versus patient history, temporal criteria, missing lab data, contradictory statements, exclusion-label inversion, invalid/unsupported citations, duplicate IDs, empty citations and malformed/truncated JSON. Include an “unseen note” fixture not used in training, but label it synthetic. Test exact quote identity, no invented text, category semantics, operational-status separation and aggregation precedence.

Build `results/errors/development.md` from actual Phase 3/4 validation outputs, not imagined failures. Review all validation predictions under a documented error taxonomy, then select at least eight distinct observed error cases if available, including unsupported evidence and missing-information failures. Record validation annotation ID, criterion type, gold/predicted label, gold/predicted evidence IDs, exact cited/gold excerpts where appropriate, parse/truncation/validity status, error category, likely cause and consequence. Do not claim a causal explanation as established fact; mark it as a hypothesis. If fewer than eight distinct observed validation errors exist, say so and list them all; supplement functional tests with clearly synthetic stress cases but do not count synthetic cases toward “observed model errors.” No test examples may be used for this ledger or service refinement.

### Phase 5 outputs

- Importable `src/clinical_match/screening.py` (or a deliberate module equivalent) and deterministic evidence lookup.
- Versioned inference-result contract and aggregation config/function.
- Explanation templates and an educational prototype notice string/field.
- Unit/integration fixtures for evidence lookup, parsing, edge cases and aggregation.
- `results/errors/development.md` based only on development predictions, with observed/synthetic cases distinguished.
- Completed `handoffs/phase_05.md` with exact commands, output hashes, edge cases and unresolved limits.

### Phase 5 verification and done gate

Call the service on a synthetic unseen note from a clean local Python process. Assert excerpts equal exact source substrings for every valid ID; invalid IDs produce no quote; raw output and all flags survive; inclusion/exclusion semantics remain distinct; unknown exclusion cannot become a preliminary match; contradictions are visible; empty input produces a controlled validation error; model runtime errors do not masquerade as labels. Run all project tests and verify imported service has no cloud/Drive dependency.

**Done only when:** the UI-independent service returns a stable, validated criterion result; citations are exact and patient-scoped; explanation and aggregation policies are tested and plainly limited; the development error ledger is evidence-based; the selected adapter can be loaded locally if Phase 4 passed. If real model access is pending, service-contract tests may pass but Phase 5 integration is explicitly incomplete.

## Phase 6 - Implement explicit KV-cached greedy generation

### Objective and gate

Implement a custom batch-one autoregressive greedy path that explicitly reuses and updates the model's key/value cache, compare it token-by-token with the uncached Phase 3 reference, and measure latency and memory under controlled conditions. This addresses the assignment requirement that merely enabling a library cache is insufficient.

**Inputs:** pinned target model/tokenizer and, after Phase 4, the selected local adapter; Phase 3 uncached reference; exact saved development prompt IDs; installed Transformers version/API; accepted local and/or verified Colab runtime. Never load test prompts before the test-frozen phase.

**Out of scope:** tiled attention, speculative decoding, batching/padding optimization, sampling and production-kernel claims. Keep cache code usable by later phases without implementing their algorithms here.

### Decisions and why

1. **One model, one prompt, one greedy contract per comparison.** Differences in weights, dtype, attention backend, tokenizer, logits processors, prompt IDs, EOS or token cap would confound correctness claims.
2. **Explicit cache state over a `generate(use_cache=True)` switch.** The implementation itself must demonstrate prefill, cache update, position/mask management and incremental input. Library generation is only an optional oracle.
3. **Token identity is the correctness criterion.** Similar decoded text or equal parsed JSON is insufficient. Numerical logit differences may explain a mismatch but cannot be rebranded as token equivalence.
4. **A speedup is not required.** Correctness and honest measurements matter. Python-level orchestration may add overhead, and latency/memory results are hardware-specific.

### Ordered implementation

#### 6.1 Inspect the exact installed model/cache API

Before coding, record Transformers, PyTorch, model revision, cache class and tensor shapes. Determine whether the returned cache is mutable, how its sequence length is reported, and whether it can be cloned/cropped for rollback. Build a small cache adapter with explicit operations: create empty cache, obtain sequence length, pass/update cache through model forward, and crop/restore when supported. Add assertions for layer count, batch/head/head-dimension and sequence axis; do not hard-code shapes without checking the model output.

If Phase 3's `src/clinical_match/decoding.py` needs to evolve into `src/clinical_match/decoding/`, preserve the Phase 3 imports via a compatibility wrapper or update all callers and tests in one change. Keep one canonical implementation and document the path chosen; avoid duplicating decoder logic.

#### 6.2 Implement cached greedy decoding with correct token timing

At generation start, pass the complete prompt exactly once with an empty cache, correct attention mask and positions. Use the last valid prompt-position logits to choose the first greedy token. That sampled first token is not yet represented in the cache: on the next iteration, feed only that newly committed token with the returned cache. Each forward pass advances the cache by the number of tokens actually processed. Stop on the same EOS/end-turn rule and at the same generated-token cap as the uncached reference.

For each call, pass an attention mask whose key length covers all cached positions plus the new input. Align `position_ids` or `cache_position` with the absolute token positions expected by the model and RoPE implementation. Initially support batch size one, unpadded prompts only. Assert cache length equals prompt length after prefill and grows exactly by the number of subsequently processed tokens. Record the terminal cache-length convention so off-by-one accounting is clear.

Keep raw generated token IDs and decode only after generation. Never compare text strings as a proxy for IDs. Preserve EOS behavior and distinguish prompt tokens, generated tokens, and processed/cache-resident tokens in logs.

#### 6.3 Match the reference decoding conditions exactly

The cached and uncached paths must share the same model/adapter object or same verified weights, tokenizer revision, prompt token IDs, dtype, evaluation mode, attention implementation, device, greedy argmax, EOS criteria and maximum new tokens. Disable dropout, sampling and unconfigured logits processors. Verify that no implicit BOS token or chat template is applied a second time. Save a comparison manifest with all of these settings and hashes.

Use the explicit Phase 3 no-cache forward-pass decoder as the reference. A library `generate()` result can be recorded as a third check but cannot replace either path. Because the existing Phase 3 library-cached output differed at token level on a small sample, investigate rather than treating it as ground truth.

#### 6.4 Verify correctness before benchmarking

Unit-test cache adapter create/update/length/crop operations on a tiny deterministic model or small controlled fixture. Then compare cached and uncached generated IDs on fixed prompts with lengths in the required envelope: test prompt lengths 128, 512 and 1,024 tokens crossed with output caps 16, 64 and 128 where supported by model context and actual resources. Include real train/validation development prompts, deterministic synthetic token prompts, early EOS, and forced max-length termination. Do not inspect or generate from held-out test cases.

For every comparison, save prompt hash/length, output cap, reference IDs, cached IDs, exact-match flag, first mismatch position and token IDs, top-logit margin at the mismatch, observed cache lengths, EOS/termination reason and model/runtime hashes. On mismatch, debug cache positions, masks, RoPE offsets, first-token handling, mutable-cache reuse and backend numerical differences. Compare logits on a small FP32 fixture and use the same backend where possible. Do not silently weaken equality to decoded text or JSON equality. If the specified envelope cannot fit locally, run correctness subset on the available device and report any untested cells; do not claim full D2 coverage.

Acceptance requires exact token equality across at least three distinct prompt/output length combinations, including short and longer output, with no unexplained mismatch in the claimed supported envelope. The full 3x3 grid is the planned target; document any resource-limited cells precisely. Unresolved differences mean the decoder is not accepted for that condition.

#### 6.5 Benchmark only after correctness is stable

Benchmark uncached and cached paths on the same device and loaded model/adapter using identical fixed prompt IDs and output conditions. Separate controlled fixed-token-count microbenchmarks (for comparable decode work) from normal EOS-terminated application prompts. Warm up until initialization and kernels stabilize, synchronize CUDA immediately before and after timed regions, and collect at least five timed repetitions per case. Reset and record peak allocated/reserved memory per run where supported.

Report prompt length, actual generated tokens, EOS or cap termination, prefill/TTFT, decode time, end-to-end latency, decode tokens/second, peak allocated/reserved memory, mean and spread (at least standard deviation or min/median/max), GPU/runtime, precision and backend. Define whether first-token latency is included in prefill/TTFT and exclude it consistently from decode throughput. Do not compare CPU to GPU, base to adapter, or different runtimes as a speedup. Show slowdowns if observed.

Run the app's development request through both modes after correctness passes, and ensure the service can expose mode selection without changing scoring or evidence logic. Keep this as a decoder integration check; the later app phase will provide the full UI/benchmark view.

### Phase 6 outputs

- Explicit uncached and cached greedy decoder modules plus cache adapter, preserving current project API compatibility.
- Tests for cache API/state growth, masks/positions, first-token handling, EOS/cap termination and cached-vs-uncached token identity.
- `results/cache/correctness.jsonl` or a versioned equivalent containing every raw comparison and resolved mismatch evidence.
- Benchmark script and raw CSV/JSON plus summary table with repeated timing/memory records, separated by environment.
- Cache diagram or concise implementation note explaining prefill, incremental update, absolute positions, mask length, cache growth and first sampled token timing.
- Completed `handoffs/phase_06.md` with hashes, actual device/software, commands, correctness coverage, measurements and limits.

### Phase 6 verification and done gate

Run targeted cached-decoder tests and the whole existing test suite. Run the quick development benchmark only after equivalence checks pass. Confirm that uncached and cached output IDs are equal under the supported deterministic envelope, cache length invariants hold, and all reported timing rows came from saved measurements. Verify benchmark artifacts contain no test data and that results clearly separate local and Colab environments.

**Done only when:** the implementation visibly performs model-forward cache reuse/update; the correctness artifact establishes exact greedy token agreement for the declared supported cases; timing and memory are synchronized, repeated and same-device; and explanation/handoff accurately state any unsupported cases. A library caching flag, speed estimate, three parsed-output matches or screenshot is not sufficient evidence.

## Cross-phase acceptance ledger

| Requirement | Evidence expected by the end of the relevant phase | What does not count |
|---|---|---|
| Real fine-tuning (B3) | Training logs/curves, executable script, adapter trained on fixed TrialGPT train rows, configuration and checkpoint hashes | Synthetic-only pilot; base-model inference; notebook that only installs packages |
| Before/after validation (B4) | Same 206 validation IDs, same model family/prompt/generation/metrics, saved predictions and absolute/delta metrics | Comparing different prompts/settings or quoting only the best score |
| Evidence traceability (C4/G2 groundwork) | Exact source excerpts from ID lookup, visible invalid/missing/contradictory cases, evidence-backed development errors | Model-written quotations, synthetic cases counted as observed model errors, claim of entailment from citation alone |
| Cache implementation/correctness (D1-D2) | Explicit forward/cache loop and token-ID equality report over multiple lengths | `use_cache=True`, text/JSON equivalence alone, no-cache vs library-only comparison |
| Cache benchmark (D3) | Synchronized warm/repeated same-device prefill/decode/throughput/memory records | Published speedup copied from a paper, unlike-device numbers combined as one speedup |
| Test isolation | Hash-checked manifests/logs showing only train/validation used before freeze | Opening test examples for “debugging” or checkpoint/prompt selection |

## Required phase handoff format

At the end of each phase, write `handoffs/phase_04.md`, `handoffs/phase_05.md` or `handoffs/phase_06.md`. Each handoff must state: status and exact scope; source/config/data/model hashes; commands and environment versions; outputs and checksums; tests/acceptance evidence; actual failures and resource limits; decisions added to [`research/decision_log.md`](../research/decision_log.md); unresolved actions assigned to the user versus code; and the next phase's explicit inputs. A phase is not complete merely because a handoff says “complete.” Link every completion claim to a saved artifact or command report.

## Existing references to read before implementation

- [Assignment PDF](../GenAI_Assignment%232.pdf) (authoritative requirements; particularly Tasks 1-3 and rubric B3-B4, C4, D1-D3).
- [Master plan](master_plan.md), Phase 4-6 sections (full project-wide intent and later phase boundary).
- [Phase 2 handoff](../handoffs/phase_02.md), [Phase 3 handoff](../handoffs/phase_03.md), [Phase 0/1/2 conceptual guide](phase_0_1_2_conceptual_guide.md).
- [Data preparation and leakage contracts](../configs/data.yaml), [model choice](../configs/model.yaml), [prompt contract](../configs/prompt.yaml), [evaluation config](../configs/eval.yaml), [prepared schema](../schemas/prepared_example.schema.json), and [inference result schema](../schemas/inference_result.schema.json).
- [Colab workflow](../docs/colab_manual_setup.md), [runtime profile](../configs/runtime_colab.yaml), [Phase 3 baseline metrics](../results/baseline/base_greedy_v1_validation/metrics.json), and [decision log](../research/decision_log.md).

The lower-level implementer should first verify that these paths and hashes still match the repository. If code has changed since this plan was written, preserve accepted Phase 0-3 contracts and update this plan or the phase handoff with a dated reason before proceeding.
