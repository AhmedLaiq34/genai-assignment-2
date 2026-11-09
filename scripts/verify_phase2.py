"""Verify phase-2 artifacts without opening held-out examples and save a report."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest_path = ROOT / 'data/splits/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest_sha = digest(manifest_path)
    assert manifest_sha == (ROOT / 'data/splits/manifest.sha256').read_text().split()[0]
    raw_path = ROOT / 'data/raw/train-00000-of-00001.parquet'
    assert digest(raw_path) == manifest['source_sha256']
    audit = json.loads((ROOT / 'results/tokenization_audit.json').read_text(encoding='utf-8'))
    assert audit['split_manifest_sha256'] == manifest_sha
    assert audit['prompt_source_sha256'] == digest(ROOT / 'src/clinical_match/prompts.py')
    assert audit['prompt_config_sha256'] == digest(ROOT / 'configs/prompt.yaml')
    assert set(audit['splits']) == {'train', 'validation'}
    assert all(not v['failures'] and not v['targets_exceeding_output_reserve_ids'] for v in audit['splits'].values())
    archive_path = ROOT / 'artifacts/colab_source.zip'
    archive_sha = digest(archive_path)
    assert archive_sha == (ROOT / 'artifacts/colab_source.zip.sha256').read_text().split()[0]
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert archive.read('Gen_AI_Assignment_2/data/splits/manifest.json') == manifest_path.read_bytes()
        assert 'Gen_AI_Assignment_2/data/splits/manifest.sha256' in names
        assert not any(name.endswith(('.jsonl', '.parquet')) for name in names)
    notebook = json.loads((ROOT / 'notebooks/colab_runner.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'colab-notebook-cell', 'exec')
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError('Test suite failed:\n' + result.stdout + result.stderr)
    report = {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'manifest_sha256': manifest_sha,
        'source_sha256': manifest['source_sha256'], 'source_archive_sha256': archive_sha,
        'row_counts': manifest['row_counts'], 'patient_counts': {k: len(v) for k, v in manifest['patient_ids'].items()},
        'test_suite_command': f'{sys.executable} -m pytest -q',
        'test_suite_output': result.stdout.strip(),
        'test_suite_exit_code': result.returncode,
        'canonical_manifest_in_archive_matches': True,
        'source_archive_contains_no_parquet_or_jsonl': True,
        'notebook_code_cells_compile_locally': True,
        'colab_bootstrap_rerun': False,
        'token_audit_scope': audit['scope'],
        'protected_test_read_by_this_verifier': False,
        'quarantined_training_rows': manifest['row_counts']['train']['quarantined'],
    }
    path = ROOT / 'results/phase02_verification.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
