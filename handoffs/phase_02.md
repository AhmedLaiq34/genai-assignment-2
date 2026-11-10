# Phase 2 handoff: dataset preparation and contracts

**Current compute route (updated 10 November):** use **Google Colab** for Phase 4 training and GPU research. See the [Colab guide](../docs/colab_manual_setup.md), [compute strategy](../output/compute_strategy.md), and [Phase 4 handoff](phase_04_colab_training_handoff.md). Kaggle CLI attempts are historical evidence only.

**Historical Kaggle attempt (9 November):** CLI authentication and quota lookup succeeded, and a private CPU verification job was submitted, but execution and output recovery were not verified. A separate private GPU probe ran and its result reported a CPU-only PyTorch build and no CUDA availability; it did not establish whether physical GPU hardware was attached. See the [submission snapshot](../results/kaggle_phase02_submission.json), [probe report](../results/kaggle_gpu_probe_v1.json) and the historical section of the [Kaggle guide](../docs/kaggle_setup.md). These are neither model-fit nor clinical-evaluation evidence.

**Historical outage note (9 November):** the user reported a Kaggle banner about notebook startup/connectivity; two recreated jobs remained queued and CLI deletion requests stalled. This records the earlier incident; the active route is now Google Colab. Verify CUDA in the live Colab session before training.

## Status and boundary

The preparation pipeline, fixed split, audited exclusion, prompt contracts, development-only tokenization audit and automated checks are implemented. All 1,015 original annotation IDs are accounted for; the source contains **one training row without criterion text**. On 9 November 2026 the student decided to exclude that row from training. The original is retained in the source and audit; the usable training set contains 602 of 603 assigned rows. The row's influence on model quality and the instructor's grading treatment remain unknown.

No model was trained or evaluated for clinical quality in this phase. The protected test file was mechanically constructed/validated for integrity; its text, labels and evidence were not used to tune prompts or to generate development statistics.

## Source and split

- Dataset: `ncbi/TrialGPT-Criterion-Annotations`, pinned revision `1cfcafde94a1560a33b4addc1638664fe28fc059`.
- Raw Parquet SHA-256: `14c547184fac9b013cd17441fd028845b40150c76b022e58334f5494077e5234`. The source is unchanged and accompanied by [license](../data/raw/LICENSE) and [snapshot notes](../data/raw/README.md).
- Split: sorted patient IDs, `numpy.random.default_rng(42).shuffle`, then 37/8/8 patients. [Manifest](../data/splits/manifest.json) records 603/206/206 source annotations, all patient/annotation IDs, and approved excluded ID 883. The current manifest SHA-256 is in [manifest.sha256](../data/splits/manifest.sha256).
- Usable outputs: train 602, validation 206, protected test 206; quarantine 1. The missing training criterion is annotation 883. [Audit](../results/data_audit.json) and [quarantine record](../data/processed/quarantine.jsonl) explain the reason without manufacturing a criterion.
- [Training-only excluded-row analysis](../results/excluded_training_row_analysis.json) records the row's type/category as exclusion/`not excluded`: 291 of 292 training-source examples in that type/category remain. No patient text or test data is included; model-performance effect remains unmeasured.
- The assignment allows public trial metadata only for application support. Do not recover a training criterion from trial metadata or previous GPT output. The student's exclusion decision and its limitations are recorded in the [decision log](../research/decision_log.md).

## What is implemented

- [Preparation script](../scripts/prepare_data.py) validates the pinned hash, required fields, patient-note consistency, unique IDs, original numbered sentence IDs, safe evidence parsing, six allowed expert categories and patient-disjoint split. It retains the original expert-evidence string in provenance and excludes GPT predictions and the legacy training flag from model inputs.
- [Data config](../configs/data.yaml) fixes dataset/revision/hash, seed, split and exclusion policy. Missing criteria other than the approved training annotation 883 cause failure. The approved exclusion is counted, retained in quarantine and reported.
- [Prepared example schema](../schemas/prepared_example.schema.json) preserves original IDs and type-specific label sets. [Inference result schema](../schemas/inference_result.schema.json) keeps raw output, status, excerpts and provenance without gold labels.
- [Prompt module](../src/clinical_match/prompts.py) uses a concrete v1-provisional chat prompt, a strict input allowlist, deterministic JSON output targets, separate allowed categories, and a fixed leading-whole-sentence truncation policy that does not inspect gold evidence. [Prompt config](../configs/prompt.yaml) reserves 96 of 1,024 tokens for output. Phase 3 may revise the prompt using validation only; changing it requires rerunning the audit and recording hashes before the final freeze.
- [Data access guard](../src/clinical_match/data_access.py) allows the approved exclusion and refuses unexpected quarantined rows. It requires an explicit final-test flag plus a checksum-verified complete frozen-manifest identity and matching dataset/split hash for test loading. It is an accidental-misuse guard, not a security boundary; later evaluation must verify actual model/adapter/prompt/metric/generation hashes against the frozen manifest.
- [Data README](../data/README.md) documents environment, commands, source anomaly, contracts and guards. [Dev requirements](../requirements-dev.txt) add pytest/jsonschema; original Phase 0 lock has not been overwritten.
- [Colab source packaging](../scripts/package_colab_source.py) includes the canonical split manifest/hash and data/development requirements, without raw Parquet or prepared JSONL. The bundle is uploaded to the active Colab runner.
- The [Colab notebook](../notebooks/colab_runner.ipynb) is the active Phase 4 workflow. The [runtime handoff verifier](../scripts/verify_colab_handoff.py) regenerates the data, compares the canonical manifest hash, and audits only development data. It passed a [local rehearsal](../results/phase02_runtime_handoff_local.json).
- Manifest serialization uses LF line endings and [.gitattributes](../.gitattributes) preserves those bytes across Windows and Linux. Earlier manifest hashes recorded in historical handoffs are superseded by the current sidecar because the approved exclusion and portable serialization changed the manifest; patient assignments did not change.

## Development-only token audit

Run [token audit](../scripts/audit_tokenization.py) against the pinned cached tokenizer. The saved [report](../results/tokenization_audit.json) includes prompt-source/config/manifest hashes, and opens only train/validation data. For the current prompt and 1,024/96 budget:

| Split | Valid examples | Maximum full prompt tokens | Truncated examples | Gold evidence IDs retained |
|---|---:|---:|---:|---:|
| Train | 602 | 595 | 0 | 418/418 |
| Validation | 206 | 436 | 0 | 115/115 |

These are retained annotated evidence IDs under preprocessing, **not** model evidence-retrieval scores. No claim is made about test-set lengths or model accuracy. All development answer targets fit the 96-token output reserve.

## Commands and verification

From the repository root, with the Phase 0 `.venv` and the target tokenizer already cached:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/prepare_data.py --config configs/data.yaml
.\.venv\Scripts\python.exe scripts/audit_tokenization.py
.\.venv\Scripts\python.exe -m pytest -q
```

Observed: the updated preparation command records the approved exclusion; the test suite checks patient disjointness, source row accounting, decimals/multiline/duplicate IDs, evidence parsing, leak-free prompt serialization, budget truncation, schemas and protected split access. The [verification report](../results/phase02_verification.json) records current hashes and test results.

## Next phase and outstanding items

Phase 3 is complete; see the [Phase 3 handoff](phase_03.md). The parser, metric harness, hand-checked fixture and unchanged-model baseline are saved. All 206 validation outputs are present and metrics reproduce; test content was not opened. The prompt is carried forward unchanged from Phase 2 under its explicit `v1-provisional` revision. The score is a baseline, not clinical-performance evidence. Phase 4 training is the next phase and uses the [Colab runner](../notebooks/colab_runner.ipynb); the adapter is still outstanding. Use the [Colab guide](../docs/colab_manual_setup.md) and [training handoff](phase_04_colab_training_handoff.md). Phase 2's training exclusion remains 602 usable rows from 603 assigned, with its effect unmeasured. Earlier checklist issues such as user-confirmed deadline/demo remain open.


