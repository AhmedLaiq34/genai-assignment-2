"""Verify each saved file in the downloaded Phase 4 bundle."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=Path('artifacts/phase04/phase04_lora_colab_cli_20261110T1650Z_bundle'))
    args = parser.parse_args()
    root = args.bundle.resolve()
    manifest_path = root / 'bundle_manifest.json'
    if not manifest_path.is_file():
        raise SystemExit(f'Missing {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    failures=[]
    for rel, expected in manifest.get('files', {}).items():
        file=(root / rel).resolve()
        if root not in file.parents or not file.is_file():
            failures.append(f'missing or out-of-bundle: {rel}'); continue
        data=file.read_bytes(); digest=hashlib.sha256(data).hexdigest()
        if len(data)!=expected['size_bytes'] or digest!=expected['sha256']:
            failures.append(f'checksum/size mismatch: {rel}')
    if failures:
        print('\n'.join(failures)); return 1
    print(f"Verified {len(manifest['files'])} files from {manifest.get('bundle_name')}")
    print(f"Adapter tensor SHA-256: {manifest['files']['selected_adapter/adapter_model.safetensors']['sha256']}")
    return 0
if __name__=='__main__': raise SystemExit(main())
