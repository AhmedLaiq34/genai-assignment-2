"""Describe the approved missing row using training labels only, without note text."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest_path = ROOT / 'data/splits/manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    train_ids = manifest['annotation_ids']['train']
    raw = ROOT / 'data/raw/train-00000-of-00001.parquet'
    if hashlib.sha256(raw.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise ValueError('Pinned source hash mismatch')
    rows = pd.read_parquet(raw, columns=['annotation_id', 'criterion_type', 'expert_eligibility'],
                           filters=[('annotation_id', 'in', train_ids)])
    excluded = rows.loc[rows['annotation_id'] == 883]
    if len(excluded) != 1 or len(rows) != manifest['row_counts']['train']['source']:
        raise ValueError('Unexpected training population or missing excluded annotation')
    kind, label = excluded.iloc[0][['criterion_type', 'expert_eligibility']]
    same_category = int(((rows['criterion_type'] == kind) & (rows['expert_eligibility'] == label)).sum())
    result = {
        'scope': 'training annotations only; no patient text, test records or model outputs',
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'excluded_annotation_id': 883,
        'criterion_type': kind,
        'expert_category': label,
        'training_source_rows': len(rows),
        'excluded_rows': 1,
        'same_type_and_category_before_exclusion': same_category,
        'same_type_and_category_after_exclusion': same_category - 1,
        'effect_on_model_performance': 'not measured',
    }
    output = ROOT / 'results/excluded_training_row_analysis.json'
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
