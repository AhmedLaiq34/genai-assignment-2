# Phase 5 handoff: criterion screening and development errors

## Status

Complete for the planned Phase 5 gates. The adapter-backed synthetic smoke and validation error ledger are present; the frozen test ledger records eight distinct patients. Results remain criterion-level screening signals, not patient-trial ground truth.

## Implemented

- `src/clinical_match/screening.py`
  - Validates sentence IDs, text, criterion type and configured limits.
  - Rejects extra sentence fields such as gold labels.
  - Builds the frozen `v1-provisional` prompt.
  - Resolves model evidence IDs to exact source text.
  - Keeps prediction, parse, evidence and runtime states separate.
  - Preserves malformed output; never converts it to `not enough information`.
  - Flags output-cap truncation separately from prompt truncation.
  - Records runtime exception type/message.
  - Flags conservative lexical contradiction candidates.
  - Records prompt hash, adapter path/hash when supplied, and omitted-sentence count.
  - Applies a versioned, data-driven aggregation policy.
- `configs/aggregation.yaml`
  - Version `v1`; criterion effects and review flags are consumed by code.
  - `not applicable` remains visible but forces `review_needed`. This is a recorded safety decision, not a claim that the expert category is wrong.
  - Definite criterion failure takes precedence while flags remain attached.
- `schemas/screening_result.schema.json`
  - Strict live-result schema; no gold fields.
- `tests/test_screening.py`
  - 20 tests covering exact excerpts, no gold leakage, invalid/duplicate/non-integer IDs, malformed JSON, illegal category, shape errors, runtime failures, missing information, output-cap and prompt truncation, contradiction flag, policy precedence, and input poisoning.
- `scripts/screen_example.py` + `examples/synthetic_screening_case.json`
  - Runs an unseen synthetic note through the real base model. Saved result: `results/dev_checks/synthetic_screening_v2.json`. Adapter-backed synthetic run: `results/dev_checks/synthetic_screening_adapter.json`.
- `examples/screening_edge_cases.json`
  - Synthetic fixtures for negation, family history, temporal qualifiers, contradiction and exclusion inversion.
- `results/errors/development.md` and `results/errors/development_adapter.md`
  - Base and selected-adapter validation ledgers with exact observed outputs/excerpts; the adapted ledger records eight distinct validation patients. Final test analysis is separately recorded in `results/errors/final_test_adapter.md` after freeze.

## Verification

- Full local suite: **51 passed**.
- Real base-model synthetic run completed; aggregate is `review_needed` because all three criteria were labelled `not applicable`. The application correctly refuses to turn that into a preliminary match.
- Schema validation is covered in `test_valid_result_has_exact_source_text_and_no_gold`.

## Decisions and limitations

- `not applicable` is deliberately review-forcing in preliminary aggregation. The original label is preserved. See the Phase 5 decision-log entry.
- Contradiction detection is a conservative lexical aid, not clinical reasoning.
- Evidence IDs removed by prompt truncation are treated as invalid because the model could not have seen those sentences.
- Adapter identity is recorded in the adapter-backed synthetic artifact. The adapter-backed validation ledger is current; frozen test errors were analyzed after the one-time guarded evaluation and no tuning followed.

## Next

See [SUBMISSION.md](../SUBMISSION.md) for final release items. No test-driven model, prompt, parser, or aggregation changes followed the frozen evaluation.


