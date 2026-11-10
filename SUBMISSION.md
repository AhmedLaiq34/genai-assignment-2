# Assignment 2 submission guide

## Release state

The working tree contains the implementation, a compiled six-page IEEE paper, frozen base and adapted test predictions, metrics and plots, and the selected adapter bundle. The Git remote is configured as `origin` (`https://github.com/AhmedLaiq34/genai-assignment-2.git`), and the current source release is published on `main`. The adapter bundle and checksum, as well as the demo recording, are linked below; submit the artifacts through the course portal and save the receipt.

## Deliverables checklist

| Item | Artifact / status |
|---|---|
| IEEE report PDF and LaTeX | [`report/paper.pdf`](report/paper.pdf), [`report/main.tex`](report/main.tex), [`report/references.bib`](report/references.bib); AI-use appendix is included in the PDF. |
| GitHub URL | <https://github.com/AhmedLaiq34/genai-assignment-2> (public `main` branch). |
| Run and reproduction instructions | This guide, `README.md`, and the commands below. |
| Model weights / download | [Selected adapter ZIP on Google Drive](https://drive.google.com/file/d/1MvDikzqD8oGyZhP__d-DU08bkMs5pCx0/view?usp=drivesdk), with [SHA-256 sidecar](https://drive.google.com/file/d/1X68LYEoKLNkA0jXwbyVxR4JJWefgdZN1/view?usp=drivesdk). Both Drive files allow anyone with the link to view and download them. Base model downloads from Hugging Face: <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>. |
| Split manifest | `results/frozen_manifest.json` and `results/frozen_manifest.sha256`. |
| Benchmark logs | Raw JSON/CSV under `results/cache/`, `results/speculative/`, and `results/attention/`. |
| AI-use appendix | Included in `report/paper.pdf` and `report/main.tex`. |
| 5–7 minute demo video | [YouTube demo](https://youtu.be/ZSvL8BiK-Cg) (confirm it is viewable by course staff). |

## Run the local demo

1. On Windows with the project environment installed, double-click `Start_App.cmd`.
2. The first model load requires network access unless the pinned model is already cached. The selected adapter bundle must exist at `artifacts/phase04/phase04_lora_colab_cli_20261110T1650Z_bundle/selected_adapter`.
3. Use a synthetic or de-identified note, add inclusion/exclusion criteria, run the default library-cached decoder, and inspect criterion labels, exact cited sentences, parse/evidence status, review flags, and aggregate result.
4. The Attention demonstration tab uses synthetic tensors. Explicit KV cache and speculative `k=4` are optional decoder choices; speculation also loads the pinned 0.5B draft.
5. JSON/HTML exports are created only when requested and are held in a temporary directory for the app process lifetime.

The exact notice is **Educational screening prototype; not a clinical enrollment decision.** Do not use this prototype for clinical enrollment decisions.

## Model and adapter identity

- Target: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (Apache-2.0).
- Draft: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0).
- Adapter: Phase 4 run `phase04_lora_colab_cli_20261110T1650Z`, validation-selected epoch 3, global step 228. Full bundle is locally preserved under `artifacts/phase04/phase04_lora_colab_cli_20261110T1650Z_bundle/`; large model weights are git-ignored.
- Bundle manifest: `artifacts/phase04/phase04_lora_colab_cli_20261110T1650Z_bundle/bundle_manifest.json`. It checks the SHA-256 and byte size of each bundle file. Verify with `python scripts/verify_training_bundle.py`; it checks all 10 bundle files against size and SHA-256 in the bundle manifest. The adapter tensor file SHA-256 is recorded by that command.
- The adapter bundle and SHA-256 sidecar are shared as viewer-accessible links above. Do not add `.safetensors` files to Git.

## Frozen evaluation and evidence

The held-out set contains 206 criterion examples from eight patients. Adapted test results: accuracy 0.704, macro-F1 0.362, and sentence-evidence micro-F1 0.296. Base results: accuracy 0.146, macro-F1 0.137, and evidence micro-F1 0.165.

- Frozen manifest: `results/frozen_manifest.json`; SHA-256 sidecar: `results/frozen_manifest.sha256` (`65924bda7fe02e4ec640029758bb283bf00b72c58fede55cfa697de04cd61f4f`).
- The guarded base and adapted evaluations each cover the same 206 test annotations exactly once. Run configs and aggregate metrics are under `results/final_test/base/` and `results/final_test/adapted/`. Full per-row prediction JSONLs remain in the local workspace for audit and are excluded from the public GitHub release.
- Summary tables and confusion plots are under `report/generated/`; eight distinct held-out error examples are included in the report and `results/errors/final_test_adapter.md`.
- The paper is `report/paper.pdf`; the IEEE LaTeX source is `report/main.tex`, with bibliography and figures beside it. Regenerate benchmark figures with `python report/build_additional_figures.py`, then build the paper with `python report/build_paper.py` from the repository root; Tectonic must be installed or supplied at the ignored workspace-local tool path. The checked-in PDF was rendered and inspected locally.
- Adapter-backed explicit-cache correctness: 6/6 exact token matches. Adapter-backed speculative correctness: 36/36. Local speculative timing is slower than cached target decoding; tiled attention is also slower on the measured local GPU. See raw JSON/CSV records under `results/cache/`, `results/speculative/`, and `results/attention/`.

## Reproduction environment

- Verify the selected training bundle: `python scripts/verify_training_bundle.py`.
- Rebuild the guarded final summary from saved outputs: `python scripts/summarize_results.py --manifest-sha256 65924bda7fe02e4ec640029758bb283bf00b72c58fede55cfa697de04cd61f4f`. This validates the manifest, prediction checksums, complete ID coverage, and recomputed metrics; it does not run inference. This audit command requires the local per-row JSONLs, which are not part of the public release.

- Training: Google Colab Tesla T4, Python 3.13.16, PyTorch 2.14.1+cu130, Transformers 4.57.1, FP16.
- Local evaluation/demo: Windows, project `.venv`, NVIDIA GeForce RTX 3050 Laptop GPU (6 GiB), PyTorch 2.11.0+cu128, Transformers 4.57.1, FP16.
- Local setup: `scripts/setup_local.cmd`; normal application launch: `Start_App.cmd`.
- Source archive and Colab notebook are available under `artifacts/colab_source.zip` and `notebooks/colab_runner.ipynb`.

## Final hand-in items to add

- Published repository URL: https://github.com/AhmedLaiq34/genai-assignment-2 (main branch).
- Adapter ZIP and SHA-256 sidecar: shared as Anyone with the link (Viewer).
- 5-7 minute demonstration: https://youtu.be/ZSvL8BiK-Cg (confirm course-staff access).
- Screenshot of the actual running local browser UI.
- Classroom submission receipt and live-demo attendance confirmation.

Before handing in, verify the YouTube link in a signed-out browser, include a screenshot of the running app if required, submit through the course portal, and save the Classroom receipt.
