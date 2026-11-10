# Phase 2 data preparation

The prescribed source is `ncbi/TrialGPT-Criterion-Annotations`, pinned to revision `1cfcafde94a1560a33b4addc1638664fe28fc059`. Its unchanged Parquet file is stored under `data/raw/`; its SHA-256 and public-domain license are recorded in `data/splits/manifest.json` and `data/raw/LICENSE`. The source `train` partition contains the entire dataset; its legacy `training` column is not the assignment split.

## Setup and execution

From the project root on Windows, use the Phase 0 Python environment:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/prepare_data.py --config configs/data.yaml
```

The student has approved excluding source annotation 883, whose criterion is missing. The command reports the exclusion and prepares the 602 valid training examples. Any additional missing criterion fails; review `results/data_audit.json` and the [decision log](../research/decision_log.md).

```powershell
.\.venv\Scripts\python.exe scripts/audit_tokenization.py
.\.venv\Scripts\python.exe -m pytest tests/test_data.py tests/test_prepare_data.py tests/test_phase2_contracts.py -q
```

The tokenizer audit needs the pinned target tokenizer in the configured local Hugging Face cache. Phase 1 already downloaded it under D:. In the active Kaggle training kernel, set `HF_HUB_CACHE` to the scratch cache; the generated kernel handles this automatically. The archived Colab notebook used `/content/.cache/huggingface/hub`. The audit never loads the test JSONL.

## Produced files and split rules

- `data/splits/manifest.json` and `manifest.sha256` contain exact patient/annotation IDs and accounting, with 37/8/8 patients and 603/206/206 source rows.
- `data/processed/development/train.jsonl` contains 602 valid examples; `validation.jsonl` contains 206. `data/processed/quarantine.jsonl` accounts for the remaining training row.
- `data/processed/protected_test/test.jsonl` contains 206 test examples and is reserved for final evaluation. Preparation mechanically validates source integrity and writes this file; development audits, prompt decisions and metric calculations must not inspect it.
- `results/data_audit.json` records source integrity, split accounting, leakage prevention and quarantine. `results/tokenization_audit.json` records training/validation-only lengths, truncation and evidence retention.
- `schemas/prepared_example.schema.json` defines the provenance, model input and expert target sections. Original expert sentence annotation strings are retained **only in provenance** for audit; prompt construction uses the explicit input allowlist.
- `schemas/inference_result.schema.json` describes a prediction without a gold target. It is a contract, not an implemented predictor.

Use `src.clinical_match.data_access.load_examples` for split access. Training loads the approved 602-row set; the loader rejects any unapproved quarantine. Loading test requires `allow_final_test=True`, an explicitly supplied frozen experiment manifest and its SHA-256. The loader verifies the frozen file, required identity fields, pinned dataset revision and split-manifest hash; future evaluation code must also verify that the model, adapter, prompt, generation and metric artifacts actually match the hashes in the frozen manifest. This is a guard against accidental misuse, not filesystem security: the raw snapshot and files still exist on disk.

## Known source issue

Pinned training annotation `883` for patient `sigir-201519` and trial `NCT02213809` has no `criterion_text`. The student chose to exclude it from training. It remains in the source snapshot and manifest and is explicitly listed in quarantine; no criterion is invented. The assignment restricts public trial metadata to application support, so do not reconstruct a supervised input from another dataset, trial metadata, the expert label or previous GPT output. Report the 602/603 coverage and the unmeasured impact on model performance; instructor grading treatment is unknown.

Validation/test records with missing criteria are fatal errors because dropping them would invalidate the required held-out comparison. No test-derived prompt or threshold tuning is permitted.

