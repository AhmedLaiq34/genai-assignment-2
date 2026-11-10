# Assignment requirements and project interpretation

The assignment PDF is authoritative. The master plan and this file are
implementation aids, not substitutes for the brief.

## Whole-project objective

Build an individual Transformer-based system that evaluates patient
descriptions against individual clinical-trial inclusion/exclusion criteria.
The output includes a category and source sentence IDs. Separately implement
and measure KV-cached generation, tiled efficient attention, and speculative
decoding. Deliver a local application, source/reproduction bundle, IEEE-format
LaTeX paper, recorded demo, and live demonstration.

This is educational screening software, not a medical enrollment decision.

## Phase 0/1 implications

- Phase 0 establishes reproducible local and Kaggle environments, captures
  actual hardware/runtime information, and tests checkpoint persistence.
- Phase 1 supports rubric A1-A3 and F1: research sources, candidate metadata,
  tokenizer compatibility, local/cloud memory fit, and documented choices.
- Qwen2.5-1.5B-Instruct and Qwen2.5-0.5B-Instruct are hypotheses from the
  plan until the tokenizer, forward, fit, and local co-residency pilots pass.
- Do not claim a Kaggle GPU, speedup, successful LoRA fit, or selected model
  until the corresponding run is recorded.
- The final application must work locally without requiring Kaggle or any
  active cloud session.

## Rubric arithmetic discrepancy

The assignment heading says 100 marks. Its detailed categories are A=15,
B=20, C=20, D=15, E=15, F=15, G=20, H=15, totaling 135. The project does not
assume a normalization rule; request clarification from the instructor.

## Deadline and demo

The assignment lists 10 November 2026 but no clock time or timezone. Confirm
the Classroom deadline timestamp and live-demo arrangements. A two-hour grace
period is stated; missing the live demo means no marks. The separate recording
must be 5-7 minutes and does not replace live attendance.
