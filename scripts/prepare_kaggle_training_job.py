"""Build a private, self-contained Kaggle Phase 4 LoRA training job locally."""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from package_colab_source import build_archive


KERNEL_TEMPLATE = '''import base64
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import traceback
import zipfile

ROOT = pathlib.Path('/kaggle/working')
SCRATCH = pathlib.Path('/kaggle/temp') if pathlib.Path('/kaggle/temp').is_dir() else pathlib.Path('/tmp/trialgpt_kaggle')
OUT = ROOT / 'output_bundle'
RUN_NAME = __RUN_NAME__

SCRATCH.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
payload = base64.b64decode(__PAYLOAD__)
digest = hashlib.sha256(payload).hexdigest()
if digest != __DIGEST__:
    raise RuntimeError('Source archive checksum mismatch')
archive = SCRATCH / 'source.zip'
archive.write_bytes(payload)
project = SCRATCH / 'Gen_AI_Assignment_2'
if project.exists():
    shutil.rmtree(project)
with zipfile.ZipFile(archive) as z:
    z.extractall(SCRATCH)

os.chdir(project)
os.environ['PROJECT_RUNTIME_PROFILE'] = 'kaggle'
os.environ['HF_HOME'] = str(SCRATCH / '.cache' / 'huggingface')
os.environ['HF_HUB_CACHE'] = str(SCRATCH / '.cache' / 'huggingface' / 'hub')
os.environ['TMPDIR'] = str(SCRATCH / 'tmp')
os.environ['TEMP'] = str(SCRATCH / 'tmp')
os.environ['TMP'] = str(SCRATCH / 'tmp')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONIOENCODING'] = 'utf-8'
pathlib.Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)


def run(args):
    command = [sys.executable, *args]
    print('RUN:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=project, check=True)


def copy_file(source, destination):
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def copy_tree(source, destination):
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)


try:
    run(['-m', 'pip', 'install', '-r', 'requirements-kaggle.txt', '-r', 'requirements-data.txt', '-r', 'requirements-dev.txt'])
    run(['scripts/doctor.py', '--smoke-test', '--output', str(OUT / 'environment.json')])
    run(['-c', 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)'])
    run(['scripts/verify_colab_handoff.py'])
    run(['-m', 'pytest', '-q'])

    run(['scripts/train.py', '--run-name', f'{RUN_NAME}_overfit', '--overfit-smoke', '--overfit-examples', '12', '--max-steps', '40', '--device', 'cuda'])
    run(['scripts/rehearse_resume.py', '--max-steps', '4', '--interrupt-at', '2', '--device', 'cuda', '--output', str(OUT / 'resume_rehearsal.json')])
    run(['scripts/train.py', '--run-name', RUN_NAME, '--device', 'cuda', '--drive-dir', str(OUT / 'persistent')])

    copy_tree(project / 'results' / 'training' / RUN_NAME, OUT / 'results' / RUN_NAME)
    copy_tree(project / 'results' / 'training' / f'{RUN_NAME}_overfit', OUT / 'results' / f'{RUN_NAME}_overfit')
    copy_file(project / 'results' / 'training' / 'resume_rehearsal.json', OUT / 'resume_rehearsal.json')
    copy_file(project / 'results' / 'phase02_runtime_handoff_kaggle.json', OUT / 'phase02_runtime_handoff_kaggle.json')
    copy_file(project / 'results' / 'tokenization_audit.json', OUT / 'tokenization_audit.json')

    exports = sorted((OUT / 'persistent' / 'exports' / RUN_NAME).glob('phase04-export-*'))
    if not exports:
        raise RuntimeError('No export bundle was produced')
    bundle = exports[-1]
    zip_path = OUT / f'phase04-{RUN_NAME}.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(bundle).as_posix())
    zip_digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (OUT / f'phase04-{RUN_NAME}.zip.sha256').write_text(f'{zip_digest}  {zip_path.name}\\n', encoding='utf-8')
    (OUT / 'source_sha256.txt').write_text(digest + '\\n', encoding='utf-8')
    (OUT / 'SUCCESS.txt').write_text(
        json.dumps({'run_name': RUN_NAME, 'source_sha256': digest, 'export_zip_sha256': zip_digest}, indent=2, sort_keys=True) + '\\n',
        encoding='utf-8',
    )
except Exception:
    (OUT / 'ERROR.txt').write_text(traceback.format_exc(), encoding='utf-8')
    raise
'''


def validate_run_name(value: str) -> str:
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{2,60}', value):
        raise ValueError('Run name must be 3-61 lowercase letters, digits, underscores or hyphens')
    return value


def validate_kernel_slug(value: str) -> str:
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,49}', value):
        raise ValueError('Kernel slug must be 3-50 lowercase letters, digits or hyphens')
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username', required=True, help='Public Kaggle username, never a credential')
    parser.add_argument('--run-name', required=True, help='Unique Phase 4 run name')
    parser.add_argument('--kernel-slug', default='trialgpt-phase-4-training', help='Unique private kernel slug')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/kaggle_phase04_job')
    args = parser.parse_args()
    username = args.username.lower()
    if not username or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in username):
        parser.error('Use a Kaggle username containing letters, digits or hyphens.')
    try:
        run_name = validate_run_name(args.run_name)
        kernel_slug = validate_kernel_slug(args.kernel_slug)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / 'source.zip'
    digest = build_archive(ROOT, archive)
    payload = base64.b64encode(archive.read_bytes()).decode('ascii')
    script = (
        KERNEL_TEMPLATE.replace('__RUN_NAME__', repr(run_name))
        .replace('__PAYLOAD__', repr(payload))
        .replace('__DIGEST__', repr(digest))
    )
    (args.output / 'train_phase04.py').write_text(script, encoding='utf-8', newline='\n')
    metadata = {
        'id': f'{username}/{kernel_slug}',
        'title': 'TrialGPT Phase 4 LoRA training',
        'code_file': 'train_phase04.py',
        'language': 'python',
        'kernel_type': 'script',
        'is_private': 'true',
        'enable_gpu': 'true',
        'enable_internet': 'true',
        'dataset_sources': [],
        'competition_sources': [],
        'kernel_sources': [],
        'model_sources': [],
    }
    (args.output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'Job: {args.output.resolve()}\nSource SHA-256: {digest}\nKernel: {metadata["id"]}\nRun name: {run_name}')


if __name__ == '__main__':
    main()
