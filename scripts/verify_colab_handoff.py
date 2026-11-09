"""Verify packaged canonical split and regenerate Phase 2 data in a runtime.

This checks preparation integrity, never evaluates or reports test content.
Run from the extracted project; the canonical manifest must already be present.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = ROOT / 'data/splits/manifest.json'
    sidecar = ROOT / 'data/splits/manifest.sha256'
    expected = sidecar.read_text(encoding='utf-8').split()[0]
    if sha(manifest) != expected:
        raise ValueError('Packaged canonical manifest does not match its SHA-256')
    subprocess.run([sys.executable, 'scripts/prepare_data.py', '--config', 'configs/data.yaml'],
                   cwd=ROOT, check=True)
    if sha(manifest) != expected or sidecar.read_text(encoding='utf-8').split()[0] != expected:
        raise ValueError('Regenerated manifest differs from the packaged canonical split')
    subprocess.run([sys.executable, 'scripts/audit_tokenization.py'], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    token_report = json.loads((ROOT / 'results/tokenization_audit.json').read_text(encoding='utf-8'))
    if token_report['split_manifest_sha256'] != expected or set(token_report['splits']) != {'train', 'validation'}:
        raise ValueError('Token audit is not tied to the canonical development splits')
    profile = os.environ.get('PROJECT_RUNTIME_PROFILE', 'local')
    if profile not in {'local', 'colab', 'kaggle'}:
        raise ValueError('Unknown runtime profile')
    report = {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'platform': sys.platform,
        'runtime_profile': profile,
        'packaged_manifest_sha256': expected,
        'regenerated_manifest_sha256': sha(manifest),
        'canonical_manifest_matches': True,
        'token_audit_scope': token_report['scope'],
        'raw_or_prepared_test_content_reported': False,
        'training_excluded_annotation_ids': [883],
    }
    output = ROOT / 'results' / f'phase02_runtime_handoff_{profile}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
