# Understanding our work so far: Phases 0 through 3

> **Compute-route update (10 November 2026):** the active training route is **Google Colab**, with the laptop for development and the final app. See [Colab training setup](../docs/colab_manual_setup.md). Verify CUDA in the current session.

This guide explains what we have done, why we did it, what the results mean, and what comes next. It is written for understanding rather than as a command-by-command implementation manual. For exact files and checks, see the [Phase 0/1 collective handoff](../handoffs/phase_00_01_collective.md), [Phase 0/1 step checklist](../handoffs/phase_00_01_step_checklist.md), [Phase 2 handoff](../handoffs/phase_02.md), and [Phase 3 handoff](../handoffs/phase_03.md).

**Our current position:** We have prepared the working environments, selected a model pair, organized the data, measured the original untrained model on validation, and implemented Phases 5-8. We have **not** completed the real TrialGPT adapter training or built the final app. The baseline measures assignment metrics, not clinical safety or usefulness. The student decided to exclude one training record with a missing criterion; its effect on model quality is unknown.

## 1. The assignment in one example

Imagine a trial says that participants must be at least 18 years old. A fictional patient's note contains these numbered sentences:

> 0. The patient is 24 years old.  
> 1. The patient reports no previous heart surgery.

We want our system to read the patient's note, the kind of requirement, and the actual requirement. It should return a category such as `included` and identify sentence `0` as evidence. For an exclusion requirement, the wording and meaning differ: `excluded` means the exclusion applies. Sometimes the note does not give enough information, and sometimes a requirement is not applicable.

The assignment is about **individual criteria**. A summary for an entire trial can be constructed with explicit rules, but the dataset labels individual criteria, not a separate overall trial verdict. The final application will be an educational screening prototype; it will not make a clinical enrollment decision.

## 2. Phase 0: where will the work run?

### What we needed to solve

Training and running a language model requires software, memory, and a suitable computer. Your laptop has a 6 GB NVIDIA GPU. That is useful for inference, but it limits how much model data and temporary computation can fit at once. Drive D: also has more available storage than drive C: did during the initial inspection.

We therefore chose two roles for our computers:

1. **Google Colab for research computation.** Use the project runner notebook and verify that the live session has CUDA before training. GPU availability is not guaranteed.
2. **Your laptop for development and the final browser application.** Copies of important results and the trained adapter must be available locally, so the evaluator does not need an active Colab session.

A Python **environment** is an isolated collection of Python and the libraries needed by a project. The local environment is under `.venv` on D:. It has its own Python and packages, so project changes do not require replacing system Python. We have local environment reports and historical Colab evidence; the current Colab GPU needs a fresh probe and report.

### What was actually checked

The environment checker ran small GPU calculations and a small Transformer forward pass on both the laptop's RTX 3050 and a Colab Tesla T4. A **forward pass** means passing an input through a model to obtain an output; it is an execution check, not proof that the model answers clinical questions correctly.

We also have tools that bundle files, check their hashes, and restore them. A **hash** is a fingerprint calculated from a file's contents. If the fingerprint after copying matches the original, it is strong evidence that the copy has not changed. One historical set of Colab reports was transferred and checked locally. Those checks are not evidence of a current Colab allocation.

### What remains to be careful about

The current training route is Colab. Confirm CUDA is actually available in the live session before training, and do not train on CPU. The historical T4 pilot and reports are snapshots, not promises of current availability. The exact Classroom deadline time and live-demo arrangements still need confirmation by you.

**What to remember:** Phase 0 proved that specific small calculations ran and that report files could be recovered. It did not train the assignment model or prove a future Colab GPU will be available.

## 3. Phase 1: why did we choose two models?

The main model we selected is `Qwen2.5-1.5B-Instruct`. It is a pre-trained Transformer that will be adapted to the criterion-matching task. The proposed smaller helper is `Qwen2.5-0.5B-Instruct`. Later, that smaller model will suggest possible next tokens for the main model to check during speculative decoding.

A **token** is a piece of text represented by a number. One word may be one token or several. For the two-model experiment, a token number proposed by the helper must mean the same thing to the main model. The saved checks compared their vocabularies, added tokens, special-token IDs, chat templates, and synthetic prompt encodings. They matched for the pinned model versions.

We also tested memory fit. The laptop ran the target model alone and the target plus draft together using synthetic inputs of 256 and 1,024 tokens. These checks included a one-token cached decode. They are useful evidence that the models fit the tested setup, but they are not a completed browser application or proof that every 96-token application request will fit.

On a historical Colab T4, a short **LoRA** pilot ran for ten optimizer steps on a fictional example. LoRA is a way to train relatively small additional model components while keeping most of the original model fixed. The pilot showed that this particular execution path ran with finite losses on the T4. Its input was padded to a length of 1,024 tokens, so it does not prove the memory requirements of a real, fully populated 1,024-token clinical example. It also did not produce the trained assignment adapter.

**What to remember:** We have a plausible, tested model pair. At this stage, we did not know whether the main model was good at the assignment's matching task. Phase 3 now gives us the first measured starting point; training is still needed to see whether it can learn from the expert examples.

## 4. Phase 2: what exactly is the dataset?

The assignment specifies the `ncbi/TrialGPT-Criterion-Annotations` dataset. A row represents one patient's note paired with one trial criterion, along with an expert's category and the sentence IDs the expert identified as relevant. A patient can have many such rows because the same note can be checked against many criteria.

The source snapshot used here contains **1,015 annotations for 53 patients**. We pinned a particular dataset revision and recorded a SHA-256 fingerprint for the original Parquet file. A pinned revision and fingerprint make it possible to verify that someone else is preparing the same source instead of an unnoticed later version.

The source also contains past GPT-generated explanations, sentence selections, and predictions. Those are not permitted model inputs. Our preparation code constructs a small, explicit list of input fields: the criterion type, criterion text, and original numbered patient sentences. The expert's label and sentence IDs are stored separately as the **target**, which is the answer we use for supervised learning or scoring. Other IDs and the original evidence annotation string are stored as **provenance**, meaning information used to trace the example back to its source.

Keeping these sections separate helps prevent **data leakage**: accidentally showing the model an answer, or information derived from an answer, that it should have had to predict.

## 5. Why do we split by patient?

If the model sees one row about a patient during training, and another row about the same patient appears in the test set, the test would not be a clean check on an unseen patient. The assignment therefore requires all rows for each patient to stay together.

The specified procedure sorts the 53 patient IDs, shuffles them with NumPy's `default_rng(42)`, then assigns the first 37 patients to training, the next 8 to validation, and the final 8 to test. The **seed** `42` makes the same shuffle reproducible when the same procedure and inputs are used.

| Group | Patients | Source annotations | Purpose |
|---|---:|---:|---|
| Training | 37 | 603 | Teach the model using available, valid examples. |
| Validation | 8 | 206 | Compare development choices, prompts, and checkpoints. |
| Test | 8 | 206 | Provide the final held-out evaluation after choices are frozen. |

The [split manifest](../data/splits/manifest.json) records the patient and annotation IDs, counts, source fingerprint and method. The preparation tests check that patients do not appear in more than one group and that every source annotation is accounted for. The same trial may still occur across patient groups; that is a limitation of the required patient-based split, not a reason to change it.

## 6. Why are there 602 usable training examples, not 603?

One training annotation, ID **883**, has no criterion text in the pinned source. The model's task needs that text: without it, we do not know what requirement the patient is being checked against. Filling it in from the expert label or a previous model response would leak the answer, and guessing would manufacture data.

We kept the row in the original snapshot and split manifest, and recorded it in [quarantine](../data/processed/quarantine.jsonl) with its reason. Quarantine means it is accounted for but not made into an invalid supervised example. The result is:

- 603 training source rows: **602 usable and 1 quarantined**.
- 206 usable validation rows.
- 206 protected test rows.

The preparation command now reports the exclusion and records the approved decision in the manifest. The loader rejects unexpected exclusions. This prevents a future script from quietly presenting the training set as if all 603 rows were usable.

The assignment restricts public trial metadata to supporting the application, so we have not substituted another source to repair the supervised example. The decision is in the [decision log](../research/decision_log.md). Excluding 1/603 rows is a small fraction. This particular row belongs to a common training group: it was an exclusion criterion labeled `not excluded`, and 291 of 292 rows in that group remain. The impact on model performance and instructor grading treatment are still unknown.

## 7. Why preserve sentence numbers?

The model must support its decisions using the patient's record. If the expert labels sentence `3` as evidence, the program needs to keep sentence `3` associated with its original text. We parse the numbered lines, retain their original IDs, and check that each expert evidence ID actually exists in that patient's note.

We also preserve wrapped continuation lines as part of their original sentence. A decimal such as `3.5` must not be mistaken for a new sentence just because it contains a period. Evidence IDs are parsed safely as a list, sorted, and deduplicated; an empty list is allowed when no evidence sentence is annotated.

For example, if a patient note has sentence IDs `0`, `2`, and `5`, we keep those IDs. We do not rename them `0`, `1`, and `2` merely to make them consecutive. Otherwise the evidence references could point at the wrong sentence.

## 8. What are the prompt and the output contract?

A **prompt** is the instruction and information we give the model. Our provisional prompt tells the model its educational screening role, supplies the patient sentences and one criterion as data, lists the categories appropriate to the criterion type, and asks for a compact structured answer.

For a made-up inclusion example, the desired shape is:

```json
{"label":"included","evidence_sentence_ids":[0]}
```

This is the expected format, not a result from our trained model. The prompt is built from an allowlist: criterion type, criterion text and sentence ID/text pairs. It does not serialize the expert target or earlier GPT predictions. The answer format uses consistent field order and sorted IDs.

We also defined an **inference result schema**: a description of what a future prediction record must contain, such as the raw model output, parsed category, evidence IDs, exact evidence excerpts, error status, truncation flag, model version, and timing. It deliberately has no expert target field. The schema is a contract; the actual prediction and explanation pipeline is still future work.

The prompt began as **provisional**. For the Phase 3 baseline, we carried it forward unchanged and froze it under the recorded `v1-provisional` revision. This gives us a fair before-training reference. A later change must be documented, audited again for token length, and scored as a new experiment.

## 9. What did the token-length audit show?

A model has a limit on how much input plus output it can handle in one request. We started with a total budget of **1,024 tokens**, reserving **96 tokens** for the answer. If a record is too long, the policy keeps the earliest whole sentences that fit; it does not use the expert's evidence labels to decide which sentences to keep. The criterion and instructions must remain present.

Using the pinned target tokenizer and the current provisional prompt, we measured only the usable training and validation examples:

| Group | Maximum prompt length | Examples shortened | Expert evidence IDs retained |
|---|---:|---:|---:|
| Training | 595 tokens | 0 of 602 | 418 of 418 |
| Validation | 436 tokens | 0 of 206 | 115 of 115 |

All development answers also fit within the proposed 96-token answer reserve. The complete numbers and fingerprints are in [tokenization audit](../results/tokenization_audit.json).

**What this means:** For these development examples and this exact prompt, our current limit does not remove patient sentences or their annotated evidence. It does **not** mean the model can retrieve the evidence correctly. It also says nothing about test-set lengths, and it is not a performance score. New user-entered notes may be longer, and a changed prompt must be measured again.

## 10. How is the test set protected?

The preparation script mechanically constructs the protected test file and checks data integrity. We did not use the test text, labels, or evidence to pick the prompt, compute development statistics, select a model, or tune a threshold.

The provided data loader refuses ordinary test access. To load it through that interface, later evaluation must explicitly request final-test access and provide a frozen experiment manifest with a matching fingerprint and the required identity fields. The loader checks the frozen file's checksum, dataset revision and split-manifest fingerprint. Future evaluation code must additionally verify that the actual model, adapter, prompt and evaluation settings match the recorded identities.

This is a guard against accidental misuse, not a security wall. The files and raw dataset still exist on disk, so responsible use and clear records remain necessary. We must not make development decisions from held-out test cases.

## 11. What did we verify?

We reran preparation and confirmed that the split-manifest fingerprint stayed the same. The updated preparation command records the approved exclusion and keeps the row accounted for in quarantine.

The automated tests cover the deterministic split, data accounting, note parsing, evidence IDs, exclusion of prior AI fields from model inputs, structured schemas, sentence-preserving truncation, and attempts to load protected test data without the required authorization. The latest result is in the [verification report](../results/phase02_verification.json).

Tests can show that specific checks passed. They do not prove clinical safety, model quality, or that the missing training criterion is resolved.

## 12. Phase 3: how did we measure the original model?

Before teaching the model, we ran it on all 206 validation examples. Validation is the practice set we are allowed to use for development decisions; the separate 206-example test set stays sealed for the final evaluation. Each example gives the model one criterion and the patient's numbered sentences. The model must return one label and sentence IDs as evidence.

We also built a strict checker. It keeps the model's raw answer, even if the answer is broken JSON or uses an invalid category. A broken answer counts as a failure; it is not quietly changed to “not enough information.” Evidence is measured by comparing the sentence IDs the model chose with the expert IDs, one example at a time.

On the RTX 3050 laptop GPU, the original model achieved **0.124 macro-F1** across the six categories and **0.155 evidence micro-F1**. It produced valid structured output on 202 of 206 examples. These numbers look low, but that is exactly why we establish the baseline before training: they show what the untouched model does under our fixed prompt and scoring rules. They are not a medical-quality or safety measure.

To finish in a practical time, the complete run used Transformers' built-in key/value cache, which reuses calculations from earlier text instead of repeating them for every next token. We compared its parsed answers with the slow, straightforward decoder on three examples. The parsed answers matched, though token sequences did not exactly match. That is enough for this baseline's practical score, but a later phase still has to implement its own cache loop and prove exact token agreement.

**What to remember:** We now know the pre-training score and have saved every validation answer. This is the “before” picture that a later trained model must be compared against. No training has happened yet.

## 13. What happens next?

**Phase 4: fine-tune.** We will teach the selected model using the 602 usable training examples and report the approved exclusion. We will use validation performance to select a saved training version and compare it with the original model under the same conditions. The synthetic Phase 1 pilot is not a substitute for this training. The active Colab runner now invokes the implemented trainer; the real adapter run is still outstanding. The baseline itself was completed locally.

**Later phases:** We will build the evidence/explanation pipeline, implement and test caching, tiled attention and speculative decoding, run final evaluation after freezing choices, create the local browser app, and write the research paper and demonstration. The [master plan](master_plan.md) gives the detailed acceptance criteria for each phase.

The exact Classroom deadline and live-demo arrangements still need confirmation. Continue with the 602 valid training rows and transparently report the 602/603 coverage. The student's exclusion decision does not determine how the instructor will grade it.

