# Kaggle setup and Phase 2 verification

Kaggle is the chosen research provider. CLI authentication and a quota query
succeeded on 9 November 2026. The account reported 30 GPU hours remaining at
that moment; availability and limits can change. Internet, actual accelerator,
job execution and output retrieval are unverified. The local app remains
independent of Kaggle. The prior Colab pilots remain historical evidence only.

1. Sign in at [Kaggle settings](https://www.kaggle.com/settings/api), create an
   API token, and follow Kaggle's on-screen instructions to save credentials
   locally. Do not send credentials in chat or add them to this repository.
   If Kaggle asks for account or phone verification, complete it in the UI.
2. The official CLI is installed as `kaggle` (version 2.2.5, with `six`); run
   `kaggle --version`, then `kaggle quota` to check account access and quota.
   Command availability can be checked with `--help`.
3. Build a private CPU verification job, replacing the placeholder with your
   Kaggle username:

   ```powershell
   .\.venv\Scripts\python.exe scripts\prepare_kaggle_job.py --username YOUR_USERNAME
   kaggle kernels push -p artifacts\kaggle_phase02_job
   kaggle kernels status YOUR_USERNAME/trialgpt-phase-2-verification
   kaggle kernels output YOUR_USERNAME/trialgpt-phase-2-verification -p artifacts\kaggle_phase02_output
   ```

   The generated script embeds a checksummed source archive; it does not embed
   credentials, raw patient data, processed JSONL, or model weights. The job
   downloads the pinned public dataset and tokenizer and installs pinned
   dependencies, so **internet access is required**. Inspect the actual run
   logs if internet is blocked; use Kaggle's notebook UI to enable internet if
   the account permits, or arrange verified offline inputs. Do not claim a pass
   from a submitted or still-running job.
4. Confirm `output_bundle/SUCCESS.txt`, environment report, Phase 2 handoff
   report and token audit are present in the downloaded output. The manifest
   hash must equal `data/splits/manifest.sha256`; verify outputs before citing
   them. Failures appear in `ERROR.txt` or the run log. This run does not train
   a model or evaluate the held-out test set.
5. For later GPU work, check `kaggle quota` and submit a separately configured
   GPU job after inspecting its metadata and actual allocation. A GPU is not
   guaranteed; do not assume a fixed weekly allowance or accelerator type.
   Record `nvidia-smi`, RAM, packages and pilot results on that actual device.

Kaggle `/kaggle/working` is session storage. A completed notebook version can
expose saved outputs for CLI download, but interrupted jobs must not be assumed
to persist their last checkpoint. Before long training, implement and test a
durable checkpoint upload and restore method that works *during* a run, or use
bounded jobs that finish and publish outputs, then download and verify them.
No resume mechanism for Kaggle training has yet been verified. Keep clinical
data and model artifacts private; verify privacy and licenses before uploading.

Official references: [Kaggle CLI](https://github.com/Kaggle/kaggle-cli),
[kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md),
and [authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md).
