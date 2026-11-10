# Remaining implementation plan

**Prepared:** 10 November 2026 (Asia/Karachi)  
**Purpose:** finish the adapted-model evidence, working local demo, and hand-in
package in a safe order. The assignment brief remains authoritative.

## Current position (verified 11 November 2026)

| Area | State | Evidence / remaining |
|---|---|---|
| Phases 0-4 | Completed; Colab T4 LoRA training selected epoch 3. | Bundle and training/validation evidence are preserved under `artifacts/phase04/` and `results/phase04/`. |
| Phase 5 | Adapter-backed service, synthetic smoke, validation ledger and post-freeze test error ledger completed. | `results/dev_checks/app_smoke_summary.json`, `results/errors/development_adapter.md`, `results/errors/final_test_adapter.md`. |
| Phase 6 | Shared generation policy, exact adapter cache correctness (6/6), base and adapter timing completed. | `results/cache/correctness_adapter.jsonl`, `results/cache/benchmark_adapter.json`. |
| Phase 7 | Dense and tiled online attention and raw benchmark evidence completed. | `results/attention/benchmark.json`; figures and source CSV are in `report/`. |
| Phase 8 | Adapter speculative correctness (36/36) and paired benchmark completed. | `results/speculative/correctness_adapter.jsonl`, `results/speculative/benchmark_adapter.json`; local result is slower. |
| Phase 9 | Frozen manifest and guarded baseline/adapted test evaluations completed once on the exact 206 IDs. | `results/frozen_manifest.json`, sidecar SHA, predictions/metrics in `results/final_test/`. No tuning followed. |
| Phase 10 | Local Gradio app and double-click launcher implemented; adapter-backed service smoke passed. | `app/main.py`, `Start_App.cmd`, `scripts/launch_app.py`. Actual UI screenshot and full browser rehearsal remain outstanding because the computer-use browser host was unavailable. |
| Phase 11 | IEEE paper compiled and rendered; submission guide created. | `report/paper.pdf`, `report/main.tex`, `SUBMISSION.md`. Actual app screenshot, 5-7 minute recording, publishing the current working tree/model link, and Classroom receipt remain to be done. |

The earlier execution plan below describes the intended gates and is now historical. Those gates remain evidence requirements, not instructions to rerun test evaluation or training.

## Historical execution order and gates (completed work; do not rerun the held-out evaluation)

### 0. Confirm the delivery window and lock the demo scope

1. Check the exact Classroom due time, timezone, submission format, and live-demo
   arrangements. The assignment document states 10 November 2026 but does not
   state a clock time.
2. Agree a minimal demo that can run locally with the selected 1.5B adapter and
   one loaded target model. Do not make cloud access or a second loaded model a
   requirement for the core screening workflow.
3. Preserve the current training, validation, and benchmark evidence. Keep the
   sealed test split unopened.

**Gate:** a written delivery checklist ordered by actual deadline, with enough
time left for a launch rehearsal and upload verification.

### 1. Align decoding behavior and adapter loading

This is a prerequisite for trustworthy adapter-backed Phase 5, 6, and 8 results
and for the demo's comparison panel.

1. Define one explicit generation contract from the pinned model and current
   evaluator configuration: greedy decoding, one beam, configured output cap,
   EOS and padding IDs, the pinned Qwen `repetition_penalty=1.1`, and no sampling.
   The recently corrected uncached reference calls
   `generate(use_cache=False)` and serves as the no-cache oracle.
2. Implement a shared logits-processor policy for manual token loops. Apply
   history-dependent processors (especially repetition penalty) to the full
   generated history before selecting each token. Ensure the library cached,
   explicit-cache, uncached-reference, and speculative target paths interpret
   the same prompt IDs, settings, stop IDs, and output cap.
3. In speculation, apply the draft model's frozen processor policy when it
   drafts and the target policy when it verifies/corrects. The accepted stream
   must match the target's canonical cached greedy output exactly.
4. Add `--adapter` to `scripts/check_decoding_correctness.py`,
   `scripts/benchmark_decoding.py`, `scripts/check_speculative_correctness.py`,
   and `scripts/benchmark_speculative.py`. Load the adapter only on the target;
   keep the pinned 0.5B draft unchanged. Record adapter, target/draft revisions,
   configs, hardware and generation policy in each output.
5. Rerun the base and adapted correctness records affected by the generation
   policy. Save token-level mismatches, first divergence and logits margins if
   any case still fails. Do not proceed to speed measurements until correctness
   passes on the real target.
6. Use a single adapter-fingerprint helper in the evaluator, training evidence,
   and live screening result. The user-facing report should display model and
   adapter identity using the same function.

**Gate:** adapter loads in a fresh local process; cached, explicit and uncached
target generation produce exact token IDs on at least three real validation
prompts at multiple output caps; the speculative adapted-target stream matches
canonical target greedy on the correctness matrix. Existing Phase 6/8 timing
numbers remain labeled base-only until refreshed.

**Commands after the CLI options are added:**

```powershell
.\.venv\Scripts\python.exe scripts\check_decoding_correctness.py --config configs\eval.yaml --adapter artifacts\phase04\phase04_lora_colab_cli_20261110T1650Z_bundle\selected_adapter --rows 3 --caps 16 32 --output results\cache\correctness_adapter.jsonl
.\.venv\Scripts\python.exe scripts\check_speculative_correctness.py --config configs\eval.yaml --adapter artifacts\phase04\phase04_lora_colab_cli_20261110T1650Z_bundle\selected_adapter --rows 3 --caps 8 16 32 --draft-k 1 2 4 8 --output results\speculative\correctness_adapter.jsonl
```

### 2. Complete adapted screening evidence (Phase 5)

1. Refresh `handoffs/phase_05.md` so it says the selected adapter exists and
   distinguishes base-only historical output from adapted evidence.
2. Run `scripts/screen_example.py` on the bundled synthetic case with the
   selected adapter. Add an option to pass named synthetic edge cases from
   `examples/screening_edge_cases.json` without reformatting them manually.
3. Confirm result objects preserve the raw response and separate category,
   parse, evidence, truncation and runtime fields. Validate every result against
   `schemas/screening_result.schema.json`. Assert each excerpt is an exact slice
   of its sentence and each cited ID came from the input.
4. Regenerate `results/errors/development.md` with the adapter on validation
   rows only. Include annotation ID, gold/predicted category, predicted/gold
   evidence IDs, exact snippets, observed failure type and likely cause. Keep
   synthetic stress cases in their own section and label them synthetic.
5. Preserve the existing aggregation policy. Explain that the output is a
   preliminary rule-based screening signal, not a learned patient-trial label.

**Gate:** adapter-backed synthetic run and validation error ledger complete;
schema and exact-source checks pass; no test example used for prompt or policy
changes.

### 3. Refresh adapted model comparisons (Phases 6 and 8)

1. Rerun explicit-cache correctness on the selected target using the shared
   generation contract. Cover real validation prompts plus the existing
   128/512/1,024-token prompt by 16/64/128 output grid where memory allows.
2. Rerun cache performance on the local RTX 3050 with synchronized warmups and
   at least five repetitions. Report median, spread, actual generated tokens,
   TTFT, throughput, peak allocated/reserved memory and GPU. Keep base and
   adapter tables distinct.
3. Rerun adapted speculative correctness on three or more real validation
   prompts. Include k=1/2/4/8, early EOS, full acceptance, rejection/correction,
   bonus token, and cap boundary cases. Verify the draft tokenizer is identical
   to the target tokenizer before generation.
4. Benchmark cached target against speculative modes with target and draft both
   resident, identical prompts/caps, randomized or rotated mode order, warmup,
   CUDA synchronization, repeated runs and memory reset. Show the actual result
   even if speculation is slower. Keep local measurements separate from Colab.
5. Update `handoffs/phase_06.md`, `handoffs/phase_08.md`, and the Phase 6/8 rows
   in `output/deliverables_checklist.md` with adapter-specific evidence.

**Gate:** exact target token equality is demonstrated for the saved matrix;
benchmarks reproduce from raw logs, use the same local device per comparison,
and make no speedup claim unless measured.

### 4. Build the demo application before opening test data (Phase 10 MVP)

The core screening UI is the deadline-critical deliverable. Use a lightweight
Gradio app that delegates inference to `ScreeningService`; keep callbacks thin.

1. Create `app/main.py` and an app module for configuration/model loading.
   Load the target and adapter once, serialize GPU requests, show a clear
   startup error if weights or CUDA are unavailable, and do not silently switch
   to another model or precision.
2. Provide a bundled synthetic example and fields for a patient note and a
   multi-row criterion table (type + text). Deterministically split the note
   into numbered sentences, show the numbered text for review, and allow edits
   before screening. Do not use protected test records as app examples.
3. Display each criterion's category, exact cited excerpt, short evidence-led
   explanation, uncertainty/parse/evidence/truncation/runtime flags, and the
   preliminary aggregate outcome. Show the exact notice: **Educational
   screening prototype; not a clinical enrollment decision.**
4. Start with canonical library-cached inference as the default mode. Add
   explicit-cache and speculative modes only after Phase 6/8 adapter gates
   pass. Show actual per-request latency, output tokens, tokens/s and speculative
   acceptance where applicable; make token agreement visible.
5. Add dense/tiled attention comparison as a small separate educational panel
   using synthetic tensors and saved measured ranges. Do not imply the tiled
   routine powers Qwen inference.
6. Add input recovery for blank note, empty criteria, malformed criteria rows,
   oversized inputs, conflicting sentence IDs, model exceptions and output
   truncation. A failed request must not poison the next valid request.
7. Export a self-contained readable HTML report (JSON optional) with the
   current result, evidence, criterion inputs, aggregate rule, model/prompt and
   decoder provenance, timestamp, and timings. Do not retain submitted notes by
   default.
8. Replace the `Start_App.cmd` placeholder with a launcher that prepares the
   local environment if needed, starts the app, waits for localhost readiness,
   opens the browser, handles paths with spaces, and reports startup failures.
   Document first-time model download and setup separately from normal launch.

**Gate:** with cloud sessions closed, launch by double-click; run the bundled
synthetic case, one custom criterion, one unknown/insufficient case, one invalid
input followed by a successful request, compare available modes, export and
open the HTML report, and restart the app successfully. Confirm no cloud login
or notebook is needed.

### 5. Freeze and perform final held-out evaluation (Phase 9)

This stage is one-way: first finish all validation-driven changes and demo
smokes. Then record an immutable release configuration and evaluate test once.

1. Create `results/frozen_manifest.json` containing code revision, data and
   split hashes, model/tokenizer and selected adapter, generation-policy
   version, prompt, truncation, parser, aggregation, precision/backend, metric
   definitions, runtime and outputs. Freeze both `base` and `adapted` model
   identities as variants under the same shared prompt/split/metric contract;
   the base variant has no adapter, while the adapted variant points to the
   selected adapter. Capture the manifest's own checksum.
2. Implement a protected test command in `scripts/evaluate.py` (or a separate
   `scripts/evaluate_frozen.py`) that requires both a valid frozen manifest and
   an explicit `--allow-final-test --variant base|adapted`. Extend the current
   test-access guard so it validates the chosen variant's model and adapter
   identity (base variant explicitly has no adapter; adapted variant requires
   the selected adapter), while both share the same frozen split/prompt/metric
   contract. Verify actual config file hashes and resolved field values against
   the manifest, not just required-key presence. It must verify exact test
   IDs/counts, reject overrides of frozen fields, write one prediction per
   expected ID, and refuse partial or duplicate coverage. Normal development
   CLI must continue to reject test split access.
3. Evaluate baseline and selected adapter against the same 206 test rows in
   the chosen primary runtime, with identical preprocessing and decoding
   settings. Save raw per-row predictions before computing aggregate scores.
   Keep the existing validation results unchanged.
4. Add `scripts/summarize_results.py` to derive aggregate and by-criterion
   classification/evidence metrics, parse/invalid/truncation counts, confusion
   matrices, tables and plots only from frozen manifests and saved predictions.
5. Analyze at least eight genuine errors across validation and test if
   available. Label each split. Include error cause, consequence and exact
   evidence; if there are fewer genuine cases, state the count and add clearly
   labeled synthetic stress cases separately. Do not tune after reading test
   outcomes.
6. Generate training curves, base/adapter metric tables, per-type confusion
   matrices, evidence charts, cache and speculative timing/memory plots, and
   tiled-attention error/time/memory figures from machine-readable files.
   Preserve CSV/JSON sources with each figure.

**Gate:** all expected test annotation IDs appear exactly once, manifest and
configs match, metrics recompute from saved outputs, no test-driven change is
made, and every plot/table traces back to raw artifacts.

### 6. Write the research paper and hand-in package (Phase 11)

1. Create `report/main.tex`, references and figures in IEEE format. Cover data
   and split, leakage controls, model and LoRA settings, prompt/output contract,
   generation policy, metrics, all results, custom attention recurrence, KV
   cache design, speculative acceptance/rejection, app architecture and limits.
2. Interpret each result in text. Report the local RTX 3050 and Colab GPU
   separately. State the local speculative slowdown and standalone tiled
   attention slowdown honestly. Include sample size, label imbalance, small
   patient count, missingness, evidence limitations, and educational-only use.
3. Include at least eight real error analyses, app screenshots, training
   curves and benchmark/error plots generated from saved artifacts. Add the
   required short AI-use appendix.
4. Create `SUBMISSION.md` with repository and paper links, selected model
   download/setup, checksums, launch instructions, tested environment, exact
   reproduction commands, demo/video link and access requirements.
5. Verify the adapter download method from a clean directory and verify files;
   keep large weights outside Git. Check licenses/attribution and ensure source,
   configs, tests and run instructions are committed. Inspect `git status` and
   confirm no credentials, raw sensitive notes, cache directories or large
   model files are staged.
6. Record a 5-7 minute video using the actual local application with cloud
   disconnected. Suggested run: problem/resources (30 s), unseen case/evidence
   (100 s), compare inference modes and show attention view (80 s), export and
   invalid-input recovery (50 s), results/limitations (60 s), repo/reproduction
   (30 s). Confirm the link opens for the intended audience.
7. Rehearse the live defense: explain the six labels and inclusion/exclusion
   polarity; trace an evidence ID; explain cache positions and masks; derive
   online-softmax state; walk a speculative correction; explain actual timing
   results; modify or inspect one small component; reload and run a new case.
8. Submit before the confirmed deadline, verify all attachments/links with a
   separate browser or account where possible, save the receipt, and confirm
   live-demo attendance arrangements.

**Gate:** clean local launch and report compile succeed; video is 5-7 minutes;
all links and artifacts are accessible; no required output exists only in a
Colab session or on the author's machine.

## Priorities if time is tight

1. **Must have for the demo:** app startup, adapter-backed screening, exact
   evidence excerpts, aggregate result, clear educational notice, HTML export,
   and recovery after invalid input.
2. **Must have before final test:** shared generation semantics, adapter-backed
   correctness for the modes being compared, frozen manifest, app smoke, and
   guarded test execution.
3. **Must have for submission:** paper, repository/setup instructions,
   reproducible metrics/figures, short video, and `SUBMISSION.md`.
4. Keep Phase 7 unchanged; do not spend time optimizing the educational tiled
   Python implementation. Drop optional cross-device sweeps, extra UI polish,
   and live trial-search integration before cutting correctness, the app, paper,
   video, or upload checks.

## Definition of done

- Adapter-backed screening and all offered decoding modes use a shared,
  documented generation policy.
- The local app works without an active cloud service and can be operated by an
  evaluator without an IDE or terminal.
- The test split has been opened only after freeze and is not used for tuning.
- Paper, video, repository, artifact links and Classroom receipt are complete.
- The live demonstration can be run from the laptop with the cloud disconnected.
