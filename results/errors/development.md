# Development error ledger: unchanged-model validation

**Source:** the 206 saved Phase 3 validation predictions under `results/baseline/base_greedy_v1_validation/`.
**Model:** unadapted `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
**Scope:** validation only. No protected-test case was opened for this ledger.
**Status:** baseline evidence; it must be repeated for the selected adapter after Phase 4 completes.

The cases below are real observed validation errors, not synthetic stress examples. "Likely cause" is a diagnostic hypothesis, not an established causal explanation. Excerpts are copied from `results/errors/development_candidates.json`.

| # | Annotation | Type | Criterion | Gold | Prediction | Observed output / evidence | Likely cause / consequence |
|---:|---:|---|---|---|---|---|---|
| 1 | 542 | inclusion | New claudication patient with Rutherford stage 2 and Fontaine stage 2a symptoms | not enough information | no parsed label (`json_error`) | Raw: `label: not applicable` / `evidence_sentence_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8]`. Gold excerpt: "A 72-year-old man complains of increasing calf pain when walking uphill." | Format failure hides an uncertain case. The parser keeps it invalid instead of converting it to abstention. |
| 2 | 553 | inclusion | Males or females older than 18 years of age | included | no parsed label (`json_error`) | Same YAML-like output. Gold excerpt: "A 72-year-old man complains of increasing calf pain when walking uphill." | Output-contract failure on an easy age criterion. This is not evidence of medical reasoning. |
| 3 | 563 | exclusion | Any illness or significant laboratory abnormality that could interfere with compliance, safety or interpretation | not enough information | no parsed label (`json_error`) | Raw: `label: excluded` / `evidence_sentence_ids: [0, 1, 2, 3, 4, 5, 6, 7]`; no gold evidence. | Invalid format for a broad exclusion. No usable evidence despite the ID list. |
| 4 | 569 | exclusion | Recent myocardial infarction within 3 months prior to screening | not excluded | no parsed label (`json_error`) | Raw: `label: excluded` / `evidence_sentence_ids: [2, 3, 4, 5, 6, 7, 8]`. Gold excerpt: "The patient had an uncomplicated myocardial infarction 2 years earlier and a transient ischemic attack 6 months ago." | The model appears to conflate a two-year-old infarction with the three-month window, but the malformed answer cannot be scored as a valid exclusion. |
| 5 | 7 | exclusion | Myocardial infarction in < 6 months | not excluded | excluded | Cited `[0,1,2,3,4,5,6,7,8]`, including "A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain..." and "The EKG shows nonspecific changes." | Hypothesis: chest pain was treated as infarction and the whole note was cited. Produces false exclusion and evidence precision loss. |
| 6 | 10 | exclusion | Severe anemia (Hb 7G/dl) | not excluded | excluded | Cited all nine sentences; the note contains no hemoglobin value. | Hypothesis: label prior or keyword matching without laboratory evidence. Produces an unsupported exclusion. |
| 7 | 11 | exclusion | Cardiac arrhythmias or II or III degree AV block | not excluded | excluded | Cited `[3,4,5,6,7,8]`, including "She is known to have hypertension and obesity" and "The EKG shows nonspecific changes." | Hypothesis: nonspecific ECG changes were over-read as the exclusion. |
| 8 | 5 | exclusion | Hypertension of > 170/100 mm of Hg | not excluded | not applicable | Cited `[3,4,5]`, including "She is known to have hypertension and obesity"; no blood-pressure reading above the threshold. | Hypothesis: `not applicable` was used when the criterion was not clearly met. Category semantics are wrong. |
| 9 | 0 | inclusion | Chronic stable angina with abnormal Exercise Myocardial Perfusion SPECT scan with reversible changes | not enough information | included | Cited `[0..8]`; gold evidence `[0,1,2]` describes chest pain but not the required scan. | Hypothesis: chest-pain language was treated as the imaging criterion. Premature inclusion and unsupported evidence. |
| 10 | 375 | inclusion | Trauma patients undergoing DC/OA management for traumatic injury | not enough information | included | Cited `[0..6]`, including the car accident and fractures but not the required DC/OA management. | Hypothesis: trauma context was treated as the management criterion. |
| 11 | 502 | inclusion | WHO performance status ≤1 | not enough information | included | Cited `[0..5]`; no performance-status assessment. | Hypothesis: general fitness language substituted for an explicit WHO score. |
| 12 | 1 | inclusion | Male and female | included | not applicable | Cited `[0..8]`; the note explicitly describes a 58-year-old woman. Gold evidence `[0]`. | Hypothesis: `not applicable` is being misused for a criterion that is satisfied. False negative and category error. |
| 13 | 2 | inclusion | Age 25 to 65 years | included | not applicable | Cited `[0..8]`; the first sentence says "A 58-year-old African-American woman." Gold evidence `[0]`. | Hypothesis: same `not applicable` category misuse; the age is explicit. |
| 14 | 22 | inclusion | Clinical diagnosis of one or more risk factors, such as diabetes, smoking, obesity, or ECG abnormality | included | not applicable | Cited `[3..8]`, including "She is known to have hypertension and obesity" and "The EKG shows nonspecific changes." Gold evidence `[3,4]`. | Hypothesis: risk-factor evidence was present but the model abstained with the wrong category. |
| 15 | 934 | inclusion | Taking anti-tuberculosis medicines during the time sputa are provided | not included | included | Cited `[0,1,2,3]`, including "A 65-year-old male with a history of tuberculosis..." but not current anti-tuberculosis treatment. | Hypothesis: topic overlap ("tuberculosis") substituted for the treatment condition. False inclusion. |

## Summary by failure mode

- **Output-contract failures:** cases `542`, `553`, `563`, `569`; all are JSON-format failures, not missing-information abstentions. Case `569` appears in the candidate extract under `empty_evidence_nonempty_gold`; its raw output is still a format failure.
- **Exclusion false positives:** cases `7`, `10`, `11`.
- **Unknown-versus-positive confusion:** cases `0`, `375`, `502`.
- **False negative / `not applicable` misuse:** cases `1`, `2`, `5`, `22`.
- **False inclusion from topical overlap:** case `934`.
- **Unsupported whole-note citation:** common across several cases; every sentence is cited even when only one or two are relevant.
- **No invalid sentence IDs were observed in the baseline** (`invalid_evidence_value_count = 0`). This is a property of this saved run, not a guarantee for the adapter.
- The service now also flags possible lexical contradictions. That detector was not part of the baseline run; the baseline did instrument output-cap truncation and recorded zero such cases.

## Consequences for Phase 5 and the application

- Keep the raw output and parse status visible; never convert a malformed answer into `not enough information`.
- Show exact cited text and explicitly warn that citation is traceability, not clinical entailment.
- Treat `not applicable` as a distinct category that forces review in preliminary aggregation; do not silently map it to success.
- Preserve `review_needed` when evidence is missing, unsupported, contradictory, truncated or operationally invalid.
- Rebuild this ledger after adapter selection; do not present baseline errors as post-training performance.
