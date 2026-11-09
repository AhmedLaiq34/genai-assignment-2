"""Create and verify durable, checksum-validated checkpoint bundles."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_BUNDLE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
_MANIFEST = "bundle_manifest.json"
_COMPLETE = "COMPLETE.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as file:
        temporary = Path(file.name)
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def verify_bundle(bundle_path: Path) -> dict[str, Any]:
    bundle_path = bundle_path.resolve()
    complete_path = bundle_path / _COMPLETE
    manifest_path = bundle_path / _MANIFEST
    if not bundle_path.is_dir() or not complete_path.is_file():
        raise ValueError(f"Bundle is missing its completion marker: {bundle_path}")
    if not manifest_path.is_file():
        raise ValueError(f"Bundle is missing its manifest: {bundle_path}")

    completion = json.loads(complete_path.read_text(encoding="utf-8"))
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if completion.get("manifest_sha256") != manifest_hash:
        raise ValueError(f"Bundle manifest checksum mismatch: {bundle_path}")
    manifest = json.loads(manifest_bytes)
    listed_files = manifest.get("files")
    if not isinstance(listed_files, dict) or not listed_files:
        raise ValueError(f"Bundle manifest has no file inventory: {bundle_path}")

    actual_files = {
        file.relative_to(bundle_path).as_posix()
        for file in bundle_path.rglob("*")
        if file.is_file() and file.name not in {_MANIFEST, _COMPLETE}
    }
    if actual_files != set(listed_files):
        raise ValueError(f"Bundle file inventory mismatch: {bundle_path}")
    for relative_path, expected in listed_files.items():
        file_path = bundle_path / Path(relative_path)
        if file_path.is_symlink() or not file_path.is_file():
            raise ValueError(f"Bundle member is not a regular file: {relative_path}")
        if file_path.stat().st_size != expected.get("size_bytes"):
            raise ValueError(f"Bundle member size mismatch: {relative_path}")
        if _sha256(file_path) != expected.get("sha256"):
            raise ValueError(f"Bundle member checksum mismatch: {relative_path}")
    return manifest


def create_verified_bundle(
    source_dir: Path,
    destination_root: Path,
    bundle_name: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    source_dir = source_dir.resolve()
    destination_root = destination_root.resolve()
    if not source_dir.is_dir():
        raise ValueError(f"Checkpoint source is not a directory: {source_dir}")
    if not _BUNDLE_NAME.fullmatch(bundle_name):
        raise ValueError(f"Invalid checkpoint bundle name: {bundle_name!r}")

    source_files = sorted(source_dir.rglob("*"))
    regular_files = [path for path in source_files if path.is_file()]
    if not regular_files:
        raise ValueError(f"Checkpoint source contains no files: {source_dir}")
    if any(path.is_symlink() for path in source_files):
        raise ValueError("Checkpoint source must not contain symbolic links")

    destination_root.mkdir(parents=True, exist_ok=True)
    final_path = destination_root / bundle_name
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint bundle: {final_path}")

    staging = Path(tempfile.mkdtemp(prefix=f".{bundle_name}.staging-", dir=destination_root))
    try:
        inventory: dict[str, dict[str, Any]] = {}
        for source_file in regular_files:
            relative = source_file.relative_to(source_dir)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
            inventory[relative.as_posix()] = {
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }

        manifest = {
            "schema_version": 1,
            "bundle_name": bundle_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "files": inventory,
        }
        manifest_path = staging / _MANIFEST
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_hash = _sha256(manifest_path)
        if _verify_staging(staging, manifest_hash):
            _atomic_json(staging / _COMPLETE, {"manifest_sha256": manifest_hash})
        else:
            raise ValueError("Staged checkpoint bundle failed validation")
        os.replace(staging, final_path)
        verify_bundle(final_path)
        _atomic_json(
            destination_root / "latest.json",
            {"bundle_name": bundle_name, "manifest_sha256": manifest_hash},
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return final_path


def restore_verified_bundle(bundle_path: Path, destination: Path) -> Path:
    bundle_path = bundle_path.resolve()
    destination = destination.resolve()
    if bundle_path == destination or bundle_path in destination.parents:
        raise ValueError("Restore destination must be outside the source bundle")
    if destination in bundle_path.parents:
        raise ValueError("Restore destination cannot contain the source bundle")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite restore destination: {destination}")

    verify_bundle(bundle_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.restore-", dir=destination.parent)
    )
    try:
        shutil.copytree(
            bundle_path,
            staging,
            dirs_exist_ok=True,
            copy_function=shutil.copy2,
            symlinks=True,
        )
        verify_bundle(staging)
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def _verify_staging(staging: Path, expected_manifest_hash: str) -> bool:
    manifest_path = staging / _MANIFEST
    if _sha256(manifest_path) != expected_manifest_hash:
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative_path, expected in manifest["files"].items():
        file_path = staging / Path(relative_path)
        if (
            not file_path.is_file()
            or file_path.stat().st_size != expected["size_bytes"]
            or _sha256(file_path) != expected["sha256"]
        ):
            return False
    return True
