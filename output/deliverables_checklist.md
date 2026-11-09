# Full-credit deliverables checklist


Check off each item only after reviewing the actual evidence. See `master_plan.md` for the complete phased implementation guide. Each subcriterion is worth 5 marks in the detailed rubric, totaling 135. Filenames below are proposed paths to create; check them off only when the actual evidence exists.

| Code | Required evidence | Completion artifact |
|---|---|---|
| A1 | At least five credible sources; critical comparison and limitations connected to design | `research/sources.md`, paper related work and bibliography |
| A2 | Two candidates; accurate architecture/license; measured cloud-training and local-inference feasibility | model comparison, `results/environments/`, cloud/local fit logs |
| A3 | Alternatives, rejected options, observed failures and evidence-based choices | `research/decision_log.md`, experiment comparisons, paper discussion |
| B1 | Prescribed data; exact 37/8/8 seed-42 patient split; IDs/counts and leakage checks | `data/splits/manifest.json`, data hashes, tests |
| B2 | Patient/type/criterion inputs; expert labels and sentence-ID targets; no GPT leakage | preparation script, schemas, prompt audit, preprocessing notes |
| B3 | Executable training, reproducible config, verified resume/export, adapters and training logs/curves | script/config, Colab runner, resumable bundles, manifest, downloaded adapter |
| B4 | Same unchanged/adapted model comparison on identical validation/test examples and decoding settings | paired predictions, metric tables, manifests |
| C1 | Inclusion/exclusion labels including unknown/NA; held-out accuracy/macro-F1 | overall and per-type test summary |
| C2 | Per-class precision/recall/F1 and confusion matrices, overall and by type | metric JSON/CSV, confusion-matrix figures |
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
| G1 | Standalone local browser UI, startup/model loading and recovery; independent of Colab | app source, launcher, offline/disconnected-Colab verification |
| G2 | Unseen note + selected/entered criteria; decisions, exact evidence, uncertainty and preliminary aggregate | working app, integration cases, video |
| G3 | Live meaningful inference comparison and experiment visibility | compare action, attention view, displayed token agreement/timing |
| G4 | Readable UI and downloadable screening report with provenance; invalid-input recovery | exported HTML report, screenshots, recovery evidence |
| H1 | Complete IEEE LaTeX research paper; interpreted methods/results/plots, limitations and citations | compiled paper PDF, LaTeX source project link |
| H2 | GitHub source, both runtime profiles, research launcher/resume guide, model/split/logs and video | repo/model URLs, environment locks, transfer hashes, guide, 5-7 minute video |
| H3 | Individual understanding; explain, execute or modify components and handle unseen cases live | defense notes, functioning local app, live-demo attendance |

### Final submission bundle

- [ ] IEEE research-paper PDF with AI-use appendix and linked accessible LaTeX source project.
- [ ] GitHub URL containing complete source, environment lock, data preparation, training/evaluation scripts, all three decoders, both attention implementations, app and correctness tests.
- [ ] Trained adapter/weights link, exact base-model/tokenizer revisions, license notices, checksums and tested download/loading method.
- [ ] Published split manifest with patient IDs, annotation membership/counts, seed/procedure and leakage checks.
- [ ] Paired baseline/adapted validation and held-out predictions, complete classification/evidence results, training logs/curves and error analysis.
- [ ] Raw benchmark logs, numerical/token correctness evidence, machine-readable summary tables and interpreted figures.
- [ ] Clear setup/reproduction guide and double-click browser-app launcher; sample downloadable screening report with provenance and educational notice.
- [ ] Thin Colab research notebook, tested local/cloud environment profiles, resumable-checkpoint instructions, durable artifact/export hashes and a successful local adapter reload. These support the rubric's reproducibility requirements; they are workflow additions, not extra instructor-assigned marks.
- [ ] Actual GPU/session labels on research benchmarks, within-device speedup calculations, and independent local app/live-comparison checks with Colab disconnected.
- [ ] Accessible 5-7 minute recorded demonstration link, correct sharing permissions and confirmed live-demo attendance.
- [ ] `SUBMISSION.md` linking every artifact, known limitations and the exact release commit; Classroom submission receipt.
