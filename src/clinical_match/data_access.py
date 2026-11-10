"""Split access guard; prevents accidental test use, not filesystem security."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_examples(root: Path, split: str, *, allow_final_test: bool = False,
                  frozen_manifest: Path | None = None, frozen_sha256: str | None = None,
                  allow_quarantined_training: bool = False,
                  variant: str | None = None) -> list[dict[str, Any]]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unknown split")
    root = Path(root)
    if split == "test":
        if not allow_final_test or frozen_manifest is None or not frozen_sha256:
            raise PermissionError("Test access requires explicit final-test flag and frozen manifest SHA-256")
        contents = Path(frozen_manifest).read_bytes()
        if hashlib.sha256(contents).hexdigest() != frozen_sha256:
            raise ValueError("Frozen manifest checksum mismatch")
        frozen = json.loads(contents)
        required = ("dataset_revision", "split_manifest_sha256")
        if frozen.get("state") != "frozen" or any(not frozen.get(k) for k in required):
            raise ValueError("Frozen manifest lacks required experiment identity")
        if frozen.get("variants"):
            if variant not in frozen["variants"]:
                raise ValueError("A frozen manifest variant must be selected for test access")
            selected = frozen["variants"][variant]
            if not selected.get("model_revision"):
                raise ValueError("Selected frozen variant lacks a model revision")
            if variant == "base" and selected.get("adapter_sha256") is not None:
                raise ValueError("The frozen base variant must not use an adapter")
            if variant == "adapted" and not selected.get("adapter_sha256"):
                raise ValueError("The frozen adapted variant must identify its adapter")
        else:
            # Compatibility with the original single-variant freeze contract.
            legacy = ("model_revision", "adapter_sha256", "prompt_sha256",
                      "generation_config_sha256", "metric_config_sha256")
            if any(not frozen.get(k) for k in legacy):
                raise ValueError("Frozen manifest lacks required experiment identity")
        manifest_path = root / "data/splits/manifest.json"
        split_bytes = manifest_path.read_bytes()
        if hashlib.sha256(split_bytes).hexdigest() != frozen["split_manifest_sha256"]:
            raise ValueError("Split manifest checksum mismatch")
        if json.loads(split_bytes)["dataset_revision"] != frozen["dataset_revision"]:
            raise ValueError("Dataset revision mismatch")
        path = root / "data/processed/protected_test/test.jsonl"
    else:
        if split == "train":
            manifest = json.loads((root / "data/splits/manifest.json").read_text(encoding="utf-8"))
            quarantined = manifest["row_counts"]["train"]["quarantined"]
            approved = manifest.get("excluded_training_annotation_ids")
            if quarantined and (quarantined != 1 or approved != [883]):
                raise ValueError("Unapproved quarantined training records; review before loading")
        path = root / "data/processed/development" / f"{split}.jsonl"
    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if split == "test" and frozen.get("variants"):
        expected_ids = frozen.get("expected_test_ids", [])
        observed_ids = [row["provenance"]["annotation_id"] for row in rows]
        if len(rows) != frozen.get("expected_test_count") or sorted(observed_ids) != sorted(expected_ids):
            raise ValueError("Protected test rows do not match the frozen IDs and count")
    return rows
