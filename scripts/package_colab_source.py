"""Build a deterministic source-only ZIP and SHA-256 sidecar for Colab."""

from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from pathlib import Path


ROOT_FILES = (
    ".gitignore",
    ".gitattributes",
    "README.md",
    "Setup_Local.cmd",
    "Start_App.cmd",
    "requirements-base.txt",
    "requirements-colab.txt",
    "requirements-kaggle.txt",
    "requirements-data.txt",
    "requirements-dev.txt",
    "checkpoints/README.md",
)
SOURCE_DIRS = (
    "configs",
    "docs",
    "examples",
    "handoffs",
    "notebooks",
    "research",
    "schemas",
    "scripts",
    "src",
    "tests",
)
DATA_FILES = (
    "data/README.md",
    "data/raw/README.md",
    "data/raw/LICENSE",
    "data/splits/manifest.json",
    "data/splits/manifest.sha256",
)
PROJECT_GUIDES = (
    "output/master_plan.md",
    "output/phase_0_1_concepts.md",
    "output/phase_0_1_2_conceptual_guide.md",
    "output/phase_4_6_implementation_plan.md",
)
BASELINE_FILES = (
    "results/baseline/base_greedy_v1_validation/metrics.json",
    "results/baseline/base_greedy_v1_validation/run_config.json",
    "results/baseline/base_greedy_v1_validation/equivalence_diagnostic.json",
)
EXCLUDED_PARTS = {
    ".cache",
    ".git",
    ".venv",
    "__pycache__",
}


def source_files(project_root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for relative in ROOT_FILES + DATA_FILES + PROJECT_GUIDES + BASELINE_FILES:
        path = project_root / relative
        if path.is_file():
            candidates.add(path)
    for relative_dir in SOURCE_DIRS:
        directory = project_root / relative_dir
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if (
                path.is_file()
                and not any(part in EXCLUDED_PARTS for part in path.parts)
                and path.suffix.lower()
                not in {".parquet", ".jsonl", ".safetensors", ".pt", ".pth", ".bin"}
            ):
                candidates.add(path)
    return sorted(candidates)


def build_archive(project_root: Path, destination: Path) -> str:
    project_root = project_root.resolve()
    destination = destination.resolve()
    files = source_files(project_root)
    if not files:
        raise ValueError(f"No source files found under {project_root}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in files:
                relative = path.relative_to(project_root).as_posix()
                info = zipfile.ZipInfo(
                    f"Gen_AI_Assignment_2/{relative}",
                    date_time=(2026, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100644 & 0xFFFF) << 16
                archive.writestr(info, path.read_bytes())
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    sidecar.write_text(
        f"{digest}  {destination.name}\n",
        encoding="utf-8",
    )
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/colab_source.zip"),
    )
    args = parser.parse_args()
    output = args.output
    if not output.is_absolute():
        output = args.project_root / output
    digest = build_archive(args.project_root, output)
    print(f"Source archive: {output.resolve()}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
