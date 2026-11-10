# Full-credit deliverables checklist


Check off each item only after reviewing the actual evidence. See `master_plan.md` for the complete phased implementation guide. Each subcriterion is worth 5 marks in the detailed rubric, totaling 135. Filenames below are proposed paths to create; check them off only when the actual evidence exists.

**Phase 4 update (10 November 2026):** the configured FP16 LoRA run completed on a Colab T4 and epoch 3 was selected from validation. The verified export bundle, per-epoch metrics, data audit, fresh Colab evaluation, and a fresh local RTX 3050 206-row evaluation are recorded under `artifacts/phase04/` and `results/phase04/`. Investigation found the old uncached reference omitted Qwen's `repetition_penalty=1.1`; it now uses `generate(use_cache=False)` and matches the canonical cached path on all three preflight examples. The previous Windows native process failure did not recur. The sealed test split was not opened.

| Code | Required evidence | Completion artifact |
|---|---|---|
| A1 | At least five credible sources; critical comparison and limitations connected to design | `research/sources.md`, paper related work and bibliography |
| A2 | Two candidates; accurate architecture/license; measured cloud-training and local-inference feasibility | model comparison, `results/environments/`, cloud/local fit logs |
| A3 | Alternatives, rejected options, observed failures and evidence-based choices | `research/decision_log.md`, experiment comparisons, paper discussion |
| B1 | Prescribed data; exact 37/8/8 seed-42 patient split; IDs/counts and leakage checks | `data/splits/manifest.json`, data hashes, tests |
| B2 | Patient/type/criterion inputs; expert labels and sentence-ID targets; no GPT leakage | preparation script, schemas, prompt audit, preprocessing notes |
| B3 | Executable training, reproducible config, verified resume/export, adapters and training logs/curves | [selected Colab adapter bundle](../artifacts/phase04/phase04_lora_colab_cli_20261110T1650Z_bundle), [run and epoch metrics](../results/phase04/phase04_lora_colab_cli_20261110T1650Z_colab_fresh_eval), [local reload metrics](../results/phase04/local_eval_fixed_full/metrics.json) |
| B4 | Same unchanged/adapted model comparison on identical validation/test examples and decoding settings | Same-device local 206-row validation metrics plus guarded, frozen paired [base test](../results/final_test/base/metrics.json) and [adapter test](../results/final_test/adapted/metrics.json); manifest: [frozen manifest](../results/frozen_manifest.json) |
| C1 | Inclusion/exclusion labels including unknown/NA; held-out accuracy/macro-F1 | [overall and per-type test summary](../report/generated/final_test_summary.json) |
| C2 | Per-class precision/recall/F1 and confusion matrices, overall and by type | [metric JSON/CSV and confusion-matrix figures](../report/generated/) |
| C3 | Original sentence-ID retrieval; micro/macro evidence metrics and empty-set rules | evidence scorer/tests, saved predictions, metric tables |
| C4 | Exact evidence-linked explanations; at least eight distinct genuine errors with missing/unsupported cases | screening output, error ledger, paper analysis |
| D1 | Explicit forward/cache reuse/update, masks, positions and cache growth | `decoding/cached.py`, cache notes/diagram |
| D2 | Greedy token equivalence across at least three input/output lengths; mismatch investigation | test suite and token-level correctness logs |
| D3 | Warmed/synchronized repeated timing and peak memory; same-device mode comparisons | raw cache CSV/JSON, GPU/session metadata and plots |
| E1 | Correct stable scaled dot-product reference with scaling/masking | `attention/dense.py`, numerical tests |
| E2 | Actual blocked attention and online maximum/normalizer/value accumulation | `attention/tiled.py`, recurrence in report |
| E3 | Multiple shapes/causal masks; max/mean error; honest speed/memory comparison | attention tests, raw logs and figures |
| F1 | Smaller compatible draft/target; tokenizer check, settings and memory justification | model manifest, compatibility test, feasibility record |
| F2 | Own drafting, block verification, accepted-prefix/rejection correction and cache synchronization | `decoding/speculative.py`, state-machine tests |
| F3 | Target-greedy equality, acceptance/latency/memory, within-device speedup or slowdown | cloud benchmark traces, local correctness/fit log, paper analysis |
| G1 | Standalone local browser UI, startup/model loading and recovery; independent of cloud | app source, launcher, offline/disconnected-Colab verification |
| G2 | Unseen note + selected/entered criteria; decisions, exact evidence, uncertainty and preliminary aggregate | working app, integration cases, video |
| G3 | Live meaningful inference comparison and experiment visibility | compare action, attention view, displayed token agreement/timing |
| G4 | Readable UI and downloadable screening report with provenance; invalid-input recovery | exported HTML report, screenshots, recovery evidence |
| H1 | Complete IEEE LaTeX research paper; interpreted methods/results/plots, limitations and citations | compiled paper PDF, LaTeX source project link |
| H2 | GitHub source, both runtime profiles, research launcher/resume guide, model/split/logs and video | repo/model URLs, environment locks, transfer hashes, guide, 5-7 minute video |
| H3 | Individual understanding; explain, execute or modify components and handle unseen cases live | defense notes, functioning local app, live-demo attendance |

### Final submission bundle

- [ ] IEEE research-paper PDF with AI-use appendix: [paper](../report/paper.pdf), [LaTeX source](../report/main.tex).
- [ ] GitHub URL containing complete source, environment lock, data preparation, training/evaluation scripts, all three decoders, both attention implementations, app and correctness tests.
- [ ] Trained adapter/weights link, exact base-model/tokenizer revisions, license notices, checksums and tested download/loading method.
- [ ] Published split manifest with patient IDs, annotation membership/counts, seed/procedure and leakage checks.
- [ ] Paired baseline/adapted validation and held-out predictions, complete classification/evidence results, training logs/curves and error analysis.
- [ ] Raw benchmark logs, numerical/token correctness evidence, machine-readable summary tables and interpreted figures.
- [ ] Clear setup/reproduction guide and double-click browser-app launcher; sample downloadable screening report with provenance and educational notice.
- [ ] Colab training notebook/config, tested local/Colab environment profiles, resumable-checkpoint instructions, durable artifact/export hashes and a successful local adapter reload. These support the rubric's reproducibility requirements; they are workflow additions, not extra instructor-assigned marks.
- [ ] Actual GPU/session labels on research benchmarks, within-device speedup calculations, and independent local app/live-comparison checks with no active cloud session.
- [ ] Accessible 5-7 minute recorded demonstration link, correct sharing permissions and confirmed live-demo attendance.
- [ ] `SUBMISSION.md` linking every artifact, known limitations and the exact release commit; Classroom submission receipt.


## Status update (11 November 2026)

- H1 paper artifact is available at `report/paper.pdf`, with source at `report/main.tex`.
- Paired frozen test metrics and exact prediction coverage are recorded under `results/final_test/`; the manifest SHA-256 sidecar is `results/frozen_manifest.sha256`.
- Adapter-backed cache/speculative correctness and local timing artifacts are under `results/cache/` and `results/speculative/`.
- Remaining external delivery items: capture the actual UI screenshot, record the 5-7 minute demo, publish the reviewed source and adapter download, add their links to `SUBMISSION.md`, and submit through Classroom.
