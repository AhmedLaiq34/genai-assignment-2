# 6-minute demo recording guide

The local demo is running at <http://127.0.0.1:7860>. Record the browser and your narration, keeping all inputs synthetic. A natural 5–7 minute recording is enough; this is a run sheet, not a script to read word for word.

## Before recording

- Confirm the app page loads and the GPU is available.
- Use the synthetic case below. Do not use identifiable patient notes.
- Keep the repository open in VS Code for the final reproduction segment.

Synthetic note:

> A 55-year-old patient has no active malignancy. ECOG performance status is 1.

Criteria:

- Inclusion: Age 18 years or older
- Exclusion: History of active malignancy

## Recording outline

| Time | Show and explain |
|---|---|
| 0:00–0:35 | State the problem: reviewing trial eligibility criteria against patient-note evidence. Point out the educational-only notice and explain that predictions are not enrollment decisions. |
| 0:35–1:15 | Show the local app and synthetic note/criteria. Explain that sentence IDs are generated so evidence can be traced to the original text. |
| 1:15–2:20 | Run screening. Inspect each criterion, prediction, exact cited sentence, parse/evidence flags, and the rule-based aggregate. Mention that citations aid review but do not prove clinical entailment. |
| 2:20–3:05 | Change the note to an unknown case, such as “The note does not state the patient's age or cancer history.” Run it and inspect how the system handles missing facts and review flags. The model's exact prediction can vary. |
| 3:05–3:40 | Clear the note and submit once to show validation. Restore the synthetic note and run again to demonstrate recovery after invalid input. |
| 3:40–4:35 | Show the decoder choices. Run **Explicit KV cache** on the synthetic case if time permits. Keep the standard library-cached mode as the default; skip speculative mode if memory is tight. |
| 4:35–5:15 | Open the Attention demonstration tab and run it. Explain that it compares dense and tiled attention on synthetic tensors and does not replace the model's production attention. |
| 5:15–5:45 | Create the HTML download and open it. Point out the criteria, evidence excerpts, aggregation policy, model and adapter provenance, prompt identity, timestamp, timings, and educational notice. |
| 5:45–6:25 | Show the report PDF and repository. Summarize the adapter's held-out metrics and one limitation, then point to `SUBMISSION.md` for setup and reproduction. |

## Final check

Play the recording once to confirm that the text is readable, narration is audible, the video is between 5 and 7 minutes, and the share link is accessible to the instructor. Add that link to the submission form and, if desired, to `SUBMISSION.md`.
