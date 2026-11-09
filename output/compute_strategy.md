# Compute strategy: free Colab research, local application

Decision updated on **9 November 2026**, Asia/Karachi. This supplements the complete implementation plan in `master_plan.md`; it records the reasoning behind its revised runtime choices. No Colab account allocation, training run or speed benchmark has been performed for this analysis.

## Recommendation

**Use free Colab first for fine-tuning, quality evaluation and the main inference experiments. Keep development, verified artifact copies and the final browser application on the laptop.** Retain a tested local training fallback.

The laptop's inspected 6 GiB GPU and low free system RAM make training fit less forgiving. A cloud GPU with more usable memory would provide room for activations, logits and temporary buffers. Its actual speed and memory advantage must be established with a pilot. Keep the proposed Qwen2.5-1.5B target and 0.5B draft because the final local application still has a 6 GiB GPU constraint.

Free Colab allocation/session limits vary, runtime storage is temporary, and separate web-UI use is restricted. Treat it as research compute rather than dependable app hosting. [Google's official FAQ](https://research.google.com/colaboratory/faq.html)

## Opportunity assessment

| Concern | Local-only route | Revised route and consequence |
|---|---|---|
| Training memory | Tight 6 GiB envelope; possible need for quantization | Prefer 16-bit LoRA if the actual cloud allocation fits. This reduces dependence on quantization-specific code. |
| Laptop RAM/storage pressure | Training, environment and model loading compete with desktop apps | Move training workspace to Colab; retain D: caches and sufficient local RAM for serving. |
| GPU performance | Limited by this laptop's device, power and thermals | Benchmark the actual cloud accelerator. A speed advantage is plausible, not established. |
| Availability | Owned hardware is immediately controllable | Time-box cloud setup; preserve local fallback instead of waiting indefinitely. |
| Reproducibility | One OS/runtime is simpler | Use one source tree, shared scientific configs, two tested environment profiles and hashes. |
| Lost work | Local storage survives a browser session ending | Persist verified checkpoints, logs and exports throughout cloud jobs. |
| Demo reliability | Local localhost app can be tested offline | Keep this advantage; no cloud connection or notebook is needed by the evaluator. |
| Deadline | Large local runs may be slow | Prioritize a short pilot and early export. Cloud compute does not shorten the coding/report workload. |

The assignment already permits free Colab and research notebooks. This revision changes execution placement and reproducibility artifacts; the dataset, exact patient split, required algorithms, metrics, paper and standalone app requirements remain intact. No additional instructor marks are assumed for the added workflow artifacts.

## Allocation gate

At the start of the first research session, record `nvidia-smi`, total/free VRAM, GPU capability, CPU RAM, disk, Python and package versions. Do not assume a particular GPU model or memory size.

1. If at least 12 GiB usable VRAM is measured, try the 1.5B model with ordinary 16-bit LoRA, microbatch 1, accumulation 8, gradient checkpointing and 1,024 total tokens.
2. With 6 to under 12 GiB usable VRAM, run the same short fit pilot before committing. Reduce unnecessary allocations; use verified 4-bit adaptation only if needed.
3. With no usable GPU, stop allocation/setup troubleshooting after 20-30 minutes. Continue local CPU development and use the measured local training fallback.
4. Independently verify the target and target+draft inference envelope on the laptop. Cloud training success alone does not satisfy this gate.

These thresholds and time limits are project decisions. They do not predict what a free account will receive. Enable BF16 only when the actual device supports it; otherwise use tested FP16. Select context length from development data and measured fit, not from unused cloud capacity.

## Where each phase runs

| Phase | Primary execution | Required local or persistent handoff |
|---|---|---|
| 0: environment | Local scaffold plus interactive Colab bootstrap | Two environment reports/profiles; tested persistence and restore. |
| 1: research/choice | Colab training-fit pilot; local inference-fit pilot | Pin target/draft/tokenizer; prove both research and delivery feasibility. |
| 2: dataset | CPU locally or in Colab | One canonical split manifest/hash copied unchanged. |
| 3: evaluation/baseline | CPU metric tests; primary GPU for generation | Saved predictions and exact decoding/runtime configs. |
| 4: adaptation | Colab primary, local fallback | Persistent resumable checkpoints, selected adapter, logs and verified local reload. |
| 5: evidence service | Local implementation; optional cloud development batches | No notebook-global state, Drive mounts or hardcoded cloud paths. |
| 6: KV caching | CPU/toy tests locally; main GPU grid in Colab | Full correctness logs plus small real-model local equivalence checks. |
| 7: attention | CPU correctness locally; GPU comparison in Colab | Numerical/performance logs with actual hardware and backend labels. |
| 8: speculation | Local state tests; Colab sweep | Local target/draft fit and token agreement; verified cloud traces. |
| 9: final experiments | One recorded primary research environment | Frozen configs, complete paired held-out predictions, logs and figures downloaded. |
| 10: application | Laptop localhost | All live modes work with Colab disconnected and cached weights offline. |
| 11: paper/release | Local packaging and recording | Accessible repository/model/video links and cloud/local reproduction instructions. |

## Portable repository and notebook contract

Implement `notebooks/colab_runner.ipynb` as a short control notebook calling the same `scripts/` and `src/` modules used locally. It bootstraps a pinned commit/source archive, checks dependencies, runs the doctor, chooses a config, launches a job and persists/resumes its outputs. The custom attention, cache and speculative algorithms remain reviewable Python source.

Use `configs/runtime_colab.yaml` and `configs/runtime_local.yaml`. Shared experiment definitions hold the split, prompt, model, adaptation and decoding choices. Separate tested locks describe the Linux/Windows installations; preserve compatible shared package versions without assuming identical CUDA wheel builds. Use Colab's detected Python and compatible preinstalled torch rather than forcing the local interpreter onto it. See the [official Transformers installation guidance](https://huggingface.co/docs/transformers/installation).

Stage active work in `/content`. Persist verified project bundles in an explicitly chosen durable location. After each productive job, download important outputs to D: as well. Setup/login/Drive authorization is a research action, never a required action for the app evaluator. This task updates plans only; no cloud service has been connected.

## Interruption and export protocol

- Save at 25 optimizer steps or five minutes, whichever is earlier, and at epoch boundaries. Include adapter, optimizer, scheduler/scaler, RNG, progress and data-order/cursor state.
- Assemble locally, copy to a uniquely named persistent bundle, verify SHA-256, then mark complete and advance the latest pointer. Keep the last two complete resumable bundles plus the best adapter.
- Simulate a restore before a full run. Verify the next training step, selected checkpoint, optimizer and data position. Record if a different GPU changes floating-point behavior.
- Restore optimizer state only for a compatible training representation. A move from 16-bit to 4-bit adaptation starts a documented new stage/run with compatible state, not a claimed exact continuation.
- Export adapter/config/tokenizer references, source commit, environment report, split manifest, predictions and raw logs. Avoid repeatedly copying downloadable pinned base weights.
- Verify the downloaded bundle on the laptop and reload the selected adapter. Use development fixtures to check schema, evidence and generation before the final test freeze.

When resuming evaluation, join completed outputs by annotation ID and the full frozen configuration/model hashes. Missing rows must be completed; incompatible variants must not be merged. Gold test labels must remain unavailable for tuning.

## Measurement rules

Measure ordinary greedy and speculative modes on the same device/checkpoint/precision/prompts. Likewise compare dense and tiled attention on the same device/dtype/shapes. Record session, GPU, software, backend, warm-up, repeated times and memory. Report local application measurements separately from cloud research measurements.

A changed cloud GPU starts a new comparison group: rerun every compared mode. Do not compute a speedup from a laptop denominator and a cloud numerator. Keep checkpoint/Drive transfer and dependency installation outside model timers, while reporting those costs separately as workflow overhead.

Cross-device equality is a development diagnostic, not a promise of bitwise reproducibility. Within each runtime, the required deterministic decoder token-equivalence tests still apply. Any quantized local deployment variant must be named and validated separately; do not automatically attribute cloud evaluation scores to it.

## Immediate next action

Implement phase 0's thin runner and doctor, inspect an actual allocation, run the small fit pilot, and prove local target/draft inference fit before spending a long session. Do not upscale the model. With the 10 November deadline approaching, prioritize complete required implementations, early artifact export and a working local demo over larger sweeps or hosting work.

All phase contracts, acceptance gates and the full-credit mapping are updated in `master_plan.md`; the standalone checklist and PDF are generated from that source.
