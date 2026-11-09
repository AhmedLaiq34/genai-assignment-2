"""Download, validate, split, and prepare the TrialGPT criterion dataset."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"
DATASET_REVISION = "1cfcafde94a1560a33b4addc1638664fe28fc059"
DATASET_FILE = "data/train-00000-of-00001.parquet"
EXPECTED_PARQUET_SHA256 = "14c547184fac9b013cd17441fd028845b40150c76b022e58334f5494077e5234"
EXPECTED_ROWS = 1015
EXPECTED_PATIENTS = 53
SPLIT_SEED = 42

REQUIRED_COLUMNS = {
    "annotation_id",
    "patient_id",
    "note",
    "trial_id",
    "criterion_type",
    "criterion_text",
    "expert_sentences",
    "expert_eligibility",
}
LABELS_BY_TYPE = {
    "inclusion": {
        "included",
        "not included",
        "not enough information",
        "not applicable",
    },
    "exclusion": {
        "excluded",
        "not excluded",
        "not enough information",
        "not applicable",
    },
}
LEAKAGE_FIELDS = {
    "gpt4_explanation",
    "gpt4_sentences",
    "gpt4_eligibility",
    "explanation_correctness",
    "training",
}
SENTENCE_LINE = re.compile(r"^(\d+)\.[ \t]+(.*)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://huggingface.co/datasets/"
        f"{DATASET_ID}/resolve/{DATASET_REVISION}/{DATASET_FILE}"
    )
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        urllib.request.urlretrieve(url, temporary)
        actual_hash = sha256_file(temporary)
        if actual_hash != EXPECTED_PARQUET_SHA256:
            raise ValueError(
                "Downloaded parquet checksum does not match the pinned snapshot: "
                f"{actual_hash}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_note(note: str) -> list[dict[str, Any]]:
    if not isinstance(note, str) or not note.strip():
        raise ValueError("Patient note is empty or is not text")

    sentences: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for line_number, line in enumerate(note.splitlines(), start=1):
        if not line.strip():
            continue
        match = SENTENCE_LINE.match(line)
        if match is None:
            if not sentences:
                raise ValueError(f"Patient note line {line_number} has no numbered sentence prefix")
            # A wrapped continuation belongs to the previous original sentence.
            sentences[-1]["text"] += "\n" + line
            continue
        sentence_id = int(match.group(1))
        sentence_text = match.group(2).strip()
        if sentence_id in seen_ids:
            raise ValueError(f"Duplicate sentence ID {sentence_id}")
        if not sentence_text:
            raise ValueError(f"Sentence {sentence_id} has no text")
        seen_ids.add(sentence_id)
        sentences.append({"id": sentence_id, "text": sentence_text})

    if not sentences:
        raise ValueError("Patient note contains no numbered sentences")
    return sentences


def parse_evidence(value: object, sentence_ids: set[int]) -> list[int]:
    if not isinstance(value, str):
        raise ValueError("expert_sentences is not a string")
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Malformed expert_sentences value: {value!r}") from error
    if not isinstance(parsed, list) or any(type(item) is not int for item in parsed):
        raise ValueError("expert_sentences must be a list of integer sentence IDs")
    evidence_ids = sorted(set(parsed))
    invalid_ids = set(evidence_ids) - sentence_ids
    if invalid_ids:
        raise ValueError(
            f"Evidence IDs do not exist in the patient note: {sorted(invalid_ids)}"
        )
    return evidence_ids


def split_patients(patient_ids: list[str], seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    ordered_ids = sorted(patient_ids)
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("Patient IDs must be unique before splitting")
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered_ids)
    return {
        "train": ordered_ids[:37],
        "validation": ordered_ids[37:45],
        "test": ordered_ids[45:],
    }


def build_example(row: dict[str, Any]) -> dict[str, Any]:
    sentences = parse_note(row["note"])
    sentence_ids = {sentence["id"] for sentence in sentences}
    evidence_ids = parse_evidence(row["expert_sentences"], sentence_ids)
    criterion_type = row["criterion_type"]
    label = row["expert_eligibility"]
    if criterion_type not in LABELS_BY_TYPE:
        raise ValueError(f"Unknown criterion type: {criterion_type!r}")
    if label not in LABELS_BY_TYPE[criterion_type]:
        raise ValueError(
            f"Label {label!r} is not valid for {criterion_type!r} criteria"
        )

    return {
        "provenance": {
            "annotation_id": int(row["annotation_id"]),
            "patient_id": str(row["patient_id"]),
            "trial_id": str(row["trial_id"]),
            "source_expert_sentences": row["expert_sentences"],
        },
        "input": {
            "criterion_type": criterion_type,
            "criterion_text": row["criterion_text"].strip(),
            "sentences": sentences,
        },
        "target": {
            "label": label,
            "evidence_sentence_ids": evidence_ids,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def prepare_data(project_root: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    if config is not None:
        expected = {"dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
                    "source_sha256": EXPECTED_PARQUET_SHA256, "seed": SPLIT_SEED}
        if (any(config.get(key) != value for key, value in expected.items())
                or config.get("split") != [37, 8, 8]
                or config.get("malformed_training_policy") != "exclude_known_missing_training_883"
                or config.get("approved_excluded_training_annotation_ids") != [883]
                or config.get("malformed_validation_or_test_policy") != "fail"):
            raise ValueError("Data config does not match the required pinned source and split")
    raw_path = project_root / "data" / "raw" / Path(DATASET_FILE).name
    if not raw_path.exists():
        download_source(raw_path)
    raw_hash = sha256_file(raw_path)
    if raw_hash != EXPECTED_PARQUET_SHA256:
        raise ValueError(
            f"Raw dataset checksum mismatch for {raw_path}: {raw_hash}"
        )

    frame = pd.read_parquet(raw_path)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    leaked = LEAKAGE_FIELDS.intersection(frame.columns)
    if not LEAKAGE_FIELDS.issubset(frame.columns):
        raise ValueError(
            f"Expected source leakage fields are missing: "
            f"{sorted(LEAKAGE_FIELDS - set(frame.columns))}"
        )
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows; found {len(frame)}")
    if frame["annotation_id"].isna().any() or frame["annotation_id"].duplicated().any():
        raise ValueError("Annotation IDs must be present and unique")
    if frame[["patient_id", "note", "trial_id", "criterion_type",
              "expert_sentences", "expert_eligibility"]].isna().any().any():
        raise ValueError("A required non-criterion field is missing")

    notes_per_patient = frame.groupby("patient_id")["note"].nunique()
    inconsistent_patients = notes_per_patient[notes_per_patient != 1]
    if not inconsistent_patients.empty:
        raise ValueError(
            "Patients have inconsistent notes: "
            f"{inconsistent_patients.index.tolist()}"
        )

    patient_ids = frame["patient_id"].astype(str).unique().tolist()
    if len(patient_ids) != EXPECTED_PATIENTS:
        raise ValueError(
            f"Expected {EXPECTED_PATIENTS} unique patients; found {len(patient_ids)}"
        )
    splits = split_patients(patient_ids)
    if tuple(len(splits[name]) for name in ("train", "validation", "test")) != (
        37,
        8,
        8,
    ):
        raise AssertionError("Patient split sizes do not match the assignment")
    if set(splits["train"]) & set(splits["validation"]) or (
        set(splits["train"]) & set(splits["test"])
    ) or set(splits["validation"]) & set(splits["test"]):
        raise AssertionError("Patient leakage detected between splits")

    patient_to_split = {
        patient_id: split_name
        for split_name, split_patient_ids in splits.items()
        for patient_id in split_patient_ids
    }
    examples: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    annotation_ids: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    quarantine: list[dict[str, Any]] = []

    for row in frame.to_dict(orient="records"):
        annotation_id = int(row["annotation_id"])
        patient_id = str(row["patient_id"])
        split_name = patient_to_split[patient_id]
        annotation_ids[split_name].append(annotation_id)

        criterion_text = row["criterion_text"]
        if pd.isna(criterion_text) or not str(criterion_text).strip():
            quarantine.append(
                {
                    "annotation_id": annotation_id,
                    "patient_id": patient_id,
                    "trial_id": str(row["trial_id"]),
                    "split": split_name,
                    "reason": "missing criterion_text in pinned source dataset",
                }
            )
            continue

        examples[split_name].append(build_example(row))

    raw_counts = {
        split_name: len(annotation_ids[split_name]) for split_name in splits
    }
    if raw_counts != {"train": 603, "validation": 206, "test": 206}:
        raise ValueError(f"Unexpected per-split source row counts: {raw_counts}")
    if sum(raw_counts.values()) != len(frame):
        raise AssertionError("Split manifest does not account for every source row")
    if sum(map(len, examples.values())) + len(quarantine) != len(frame):
        raise AssertionError("Prepared and quarantined rows do not cover the source")
    if any(item["split"] != "train" for item in quarantine):
        raise ValueError("A validation/test row has missing criterion text; evaluation coverage would be incomplete")
    if [item["annotation_id"] for item in quarantine] != [883]:
        raise ValueError("Unexpected missing training criteria: only source annotation 883 is approved for exclusion")

    data_root = project_root / "data"
    dev_dir = data_root / "processed" / "development"
    test_dir = data_root / "processed" / "protected_test"
    _write_jsonl(dev_dir / "train.jsonl", examples["train"])
    _write_jsonl(dev_dir / "validation.jsonl", examples["validation"])
    _write_jsonl(test_dir / "test.jsonl", examples["test"])
    _write_jsonl(data_root / "processed" / "quarantine.jsonl", quarantine)

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_file": DATASET_FILE,
        "source_sha256": raw_hash,
        "source_license": "public-domain; see data/raw/LICENSE",
        "source_row_count": len(frame),
        "excluded_training_annotation_ids": [883],
        "exclusion_decision": "User-approved exclusion of the known missing criterion; see research/decision_log.md",
        "split_method": (
            "Sort unique patient IDs, then shuffle with "
            "numpy.random.default_rng(seed); assign 37/8/8."
        ),
        "seed": SPLIT_SEED,
        "patient_ids": splits,
        "annotation_ids": {
            name: sorted(ids) for name, ids in annotation_ids.items()
        },
        "row_counts": {
            name: {
                "source": raw_counts[name],
                "prepared": len(examples[name]),
                "quarantined": sum(
                    item["split"] == name for item in quarantine
                ),
            }
            for name in splits
        },
    }
    manifest_path = data_root / "splits" / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    with (data_root / "splits" / "manifest.sha256").open("w", encoding="utf-8", newline="\n") as file:
        file.write(f"{manifest_hash}  manifest.json\n")

    audit = {
        "source_row_count": len(frame),
        "patient_count": len(patient_ids),
        "annotation_ids_unique": True,
        "patient_notes_consistent": True,
        "patient_splits_disjoint": True,
        "all_source_rows_accounted_for": True,
        "valid_evidence_references_checked": True,
        "missing_criterion_policy": "Exclude only user-approved training annotation 883 from supervised examples; preserve source and quarantine record. Any other missing criterion is a fatal error.",
        "test_content_policy": "Protected test examples are constructed and checked for integrity, but no test text, labels, or evidence are included in development statistics or tuning.",
        "trial_overlap_note": "The assignment groups by patient, not trial; trial IDs may occur across splits.",
        "leakage_columns_present_in_source_but_omitted_from_prepared_examples": sorted(
            leaked
        ),
        "quarantined_rows": quarantine,
        "prepared_counts": {
            name: len(examples[name]) for name in examples
        },
        "manifest_sha256": manifest_hash,
    }
    _write_json(project_root / "results" / "data_audit.json", audit)
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_hash,
        "audit": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing data/ (defaults to this script's parent).",
    )
    parser.add_argument("--config", type=Path, default=None, help="Pinned dataset/split configuration")
    parser.add_argument("--allow-quarantined-training", action="store_true",
                        help="Legacy compatibility flag; exclusion of source annotation 883 is already recorded")
    args = parser.parse_args()
    if args.config:
        import yaml
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    else:
        config = None
    result = prepare_data(args.project_root.resolve(), config)
    counts = result["manifest"]["row_counts"]
    print(f"Prepared data at {args.project_root.resolve() / 'data'}")
    print(f"Split manifest SHA-256: {result['manifest_sha256']}")
    print("Rows by split (source/prepared/quarantined):")
    for split_name, values in counts.items():
        print(
            f"  {split_name}: {values['source']}/"
            f"{values['prepared']}/{values['quarantined']}"
        )
    if result["audit"]["quarantined_rows"]:
        print("WARNING: malformed training rows quarantined; see results/data_audit.json. Do not claim all training rows are usable.")
        print("Decision recorded: exclude the known source annotation 883 from training; the original record remains auditable.")


if __name__ == "__main__":
    main()
