"""Prepare a small private Kaggle GPU allocation and arithmetic probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/kaggle_gpu_probe')
    args = parser.parse_args()
    username = args.username.lower()
    if not username or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in username):
        parser.error('Invalid Kaggle username')
    args.output.mkdir(parents=True, exist_ok=True)
    code = '''import json, platform, subprocess, sys, traceback
from datetime import datetime, timezone
from pathlib import Path
report = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
          'python': sys.version, 'synthetic_only': True, 'passed': False}
try:
    try:
        smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15, check=False)
        report['nvidia_smi_returncode'] = smi.returncode
        report['nvidia_smi_stdout'] = smi.stdout
        report['nvidia_smi_stderr'] = smi.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        report['nvidia_smi_unavailable'] = str(exc)
    import torch
    report['torch'] = torch.__version__
    report['cuda_available'] = torch.cuda.is_available()
    if not report['cuda_available']:
        raise RuntimeError('Kaggle did not provide a CUDA device')
    report['device_count'] = torch.cuda.device_count()
    report['gpu_name'] = torch.cuda.get_device_name(0)
    report['gpu_total_bytes'] = torch.cuda.get_device_properties(0).total_memory
    report['compute_capability'] = list(torch.cuda.get_device_capability(0))
    free, total = torch.cuda.mem_get_info(0)
    report['gpu_free_bytes'] = free
    a = torch.arange(64, device='cuda', dtype=torch.float32).reshape(8, 8)
    b = a @ a
    torch.cuda.synchronize()
    report['finite_matrix_multiply'] = bool(torch.isfinite(b).all().item())
    if not report['finite_matrix_multiply']:
        raise RuntimeError('Nonfinite CUDA matrix multiplication')
    report['passed'] = True
except Exception:
    report['error'] = traceback.format_exc()
finally:
    Path('/kaggle/working/gpu_probe.json').write_text(json.dumps(report, indent=2) + '\\n')
    print(json.dumps(report, indent=2))
if not report['passed']:
    raise SystemExit(1)
'''
    (args.output / 'gpu_probe.py').write_text(code, encoding='utf-8', newline='\n')
    metadata = {'id': f'{username}/trialgpt-gpu-probe', 'title': 'TrialGPT GPU probe',
                'code_file': 'gpu_probe.py', 'language': 'python', 'kernel_type': 'script',
                'is_private': 'true', 'enable_gpu': 'true', 'enable_internet': 'false',
                'dataset_sources': [], 'competition_sources': [], 'kernel_sources': [], 'model_sources': []}
    (args.output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {metadata["id"]} at {args.output.resolve()}')


if __name__ == '__main__':
    main()
