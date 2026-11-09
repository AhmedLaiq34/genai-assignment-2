# Compute strategy: Kaggle research and local application

Updated 9 November 2026. The user chose Kaggle after repeated Colab runtime allocation stalls. The laptop remains the development and final application environment. Earlier Colab T4 synthetic pilots are historical feasibility evidence, not Kaggle results or clinical performance.

Kaggle can run private jobs through its [official CLI](https://github.com/Kaggle/kaggle-cli); [kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md) controls visibility, accelerator and internet settings. Account access and quota were checked; internet, GPU and dependencies remain unverified. Do not assume fixed free resources. See [Kaggle setup](../docs/kaggle_setup.md).

The RTX 3050 laptop has 6 GiB VRAM, so a suitable cloud GPU might make training easier, but that is not a measured speedup. Retain the chosen 1.5B target and smaller 0.5B draft subject to actual fit. Record hardware, memory and a short pilot on the allocated Kaggle device. If no cloud GPU is available, time-box troubleshooting and verify a local training fallback.

Run preparation and metric tests on CPU; use the Kaggle GPU for training and research benchmarks when verified. Compare methods on the same hardware, checkpoint and precision. The final app must run independently on the laptop after downloading its files.

Kaggle working storage is temporary. Download and verify completed job outputs. Before long training, implement and test a way to preserve resumable checkpoints during an active run, or use short completed jobs with verified output recovery. Completed outputs do not prove interrupted-job recovery. Save adapter, optimizer, scheduler, random states and data position; verify restoration and hashes. Never tune on held-out test data. See the [master plan](master_plan.md) and [decision log](../research/decision_log.md).
