"""Build a private, self-contained Kaggle Phase 2 verification job locally."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from package_colab_source import build_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username', required=True, help='Public Kaggle username, never a credential')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/kaggle_phase02_job')
    parser.add_argument('--gpu', action='store_true', help='Request a GPU; CPU is sufficient for Phase 2')
    args = parser.parse_args()
    if not args.username or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.username.lower()):
        parser.error('Use a Kaggle username containing letters, digits or hyphens.')
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / 'source.zip'
    digest = build_archive(ROOT, archive)
    payload = base64.b64encode(archive.read_bytes()).decode('ascii')
    script = '''import base64, hashlib, os, pathlib, subprocess, sys, traceback, zipfile
root = pathlib.Path('/kaggle/working')
archive = root / 'source.zip'
data = base64.b64decode(PAYLOAD)
assert hashlib.sha256(data).hexdigest() == DIGEST, 'Source archive checksum mismatch'
archive.write_bytes(data)
with zipfile.ZipFile(archive) as z:
    z.extractall(root)
project = root / 'Gen_AI_Assignment_2'
os.chdir(project)
os.environ['PROJECT_RUNTIME_PROFILE'] = 'kaggle'
os.environ['HF_HOME'] = '/kaggle/working/.cache/huggingface'
os.environ['HF_HUB_CACHE'] = '/kaggle/working/.cache/huggingface/hub'
out = root / 'output_bundle'
out.mkdir(exist_ok=True)
try:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-kaggle.txt', '-r', 'requirements-data.txt', '-r', 'requirements-dev.txt'], check=True)
    subprocess.run([sys.executable, 'scripts/doctor.py', '--smoke-test', '--output', str(out / 'environment.json')], check=True)
    subprocess.run([sys.executable, 'scripts/verify_colab_handoff.py'], check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q'], check=True)
    for name in ['phase02_runtime_handoff_kaggle.json', 'tokenization_audit.json']:
        (out / name).write_bytes((project / 'results' / name).read_bytes())
    (out / 'source_sha256.txt').write_text(DIGEST + '\\n')
    (out / 'SUCCESS.txt').write_text('Phase 2 runtime verification passed. No model training was performed.\\n')
except Exception:
    (out / 'ERROR.txt').write_text(traceback.format_exc())
    raise
'''
    script = script.replace('PAYLOAD', repr(payload)).replace('DIGEST', repr(digest))
    (args.output / 'phase02_verify.py').write_text(script, encoding='utf-8', newline='\n')
    metadata = {'id': f'{args.username.lower()}/trialgpt-phase-2-verification', 'title': 'TrialGPT Phase 2 verification',
                'code_file': 'phase02_verify.py', 'language': 'python', 'kernel_type': 'script',
                'is_private': 'true', 'enable_gpu': str(args.gpu).lower(), 'enable_internet': 'true',
                'dataset_sources': [], 'competition_sources': [], 'kernel_sources': [], 'model_sources': []}
    (args.output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'Job: {args.output.resolve()}\nSource SHA-256: {digest}\nKernel: {metadata["id"]}')


if __name__ == '__main__':
    main()
