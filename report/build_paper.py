"""Build the IEEE PDF from frozen result artifacts and checked-in figures."""
from pathlib import Path
import hashlib, json, subprocess, shutil
root = Path(__file__).resolve().parents[1]
report = root / "report"
manifest = (root / "results/frozen_manifest.json").read_bytes()
expected = (root / "results/frozen_manifest.sha256").read_text(encoding="utf-8").split()[0]
actual = hashlib.sha256(manifest).hexdigest()
if actual != expected:
    raise SystemExit("Frozen manifest checksum mismatch")
for rel in ["generated/results.tex", "generated/per_class_tables.tex", "generated/confusion_tables.tex", "generated/error_examples.tex", "references.bib", "figures/system_architecture.pdf", "figures/training_loss.pdf", "figures/test_metrics.pdf", "figures/decoder_benchmarks.pdf", "figures/attention_runtime.pdf", "figures/cache_benchmarks.pdf", "figures/speculative_memory.pdf", "figures/attention_error.pdf", "figures/attention_memory.pdf"]:
    if not (report / rel).is_file():
        raise SystemExit(f"Missing report input: {rel}")
exe = shutil.which("tectonic") or str(root / "tmp/tools/tectonic-0.17.0/tectonic.exe")
if not Path(exe).is_file():
    raise SystemExit("Tectonic is required; install it or place the Windows executable at tmp/tools/tectonic-0.17.0/tectonic.exe")
run = subprocess.run([str(exe), "-X", "compile", "main.tex", "--outdir", ".", "--keep-logs"], cwd=report, check=False)
if run.returncode == 0:
    shutil.copy2(report / "main.pdf", report / "paper.pdf")
raise SystemExit(run.returncode)
