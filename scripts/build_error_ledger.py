"""Build a human-readable error ledger from saved development predictions only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.clinical_match.data_access import load_examples


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def excerpt(sentences: dict[int, str], ids: list[int]) -> str:
    if not ids:
        return "(no cited sentence IDs)"
    return " ".join(f"[{idx}] {sentences[idx]}" for idx in ids if idx in sentences)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/frozen_manifest.json")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--variant", choices=["base", "adapted"], default="adapted")
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive")

    predictions = read_jsonl(args.predictions)
    if args.split == "test":
        if not args.manifest_sha256:
            raise PermissionError("Test error analysis requires the frozen manifest SHA-256")
        manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
        rows = load_examples(ROOT, "test", allow_final_test=True,
                             frozen_manifest=manifest, frozen_sha256=args.manifest_sha256,
                             variant=args.variant)
    else:
        rows = read_jsonl(ROOT / "data/processed/development" / f"{args.split}.jsonl")
    by_id = {row["provenance"]["annotation_id"]: row for row in rows}
    if len(by_id) != len(rows) or {p["annotation_id"] for p in predictions} != set(by_id):
        raise ValueError("Prediction IDs must cover the chosen development split exactly once")
    errors = []
    for pred in predictions:
        ann_id = pred["annotation_id"]
        example = by_id[ann_id]
        gold = pred["gold_label"]
        predicted = pred.get("label_for_scoring")
        parse_status = pred.get("parse_status", "unknown")
        invalid = pred.get("invalid_evidence_values", [])
        evidence_gold = list(pred.get("gold_evidence_ids", []))
        evidence_pred = list(pred.get("evidence_ids_validated", []))
        if predicted != gold or parse_status != "valid" or invalid or set(evidence_pred) != set(evidence_gold):
            errors.append((pred, example))

    # Prioritize parse/runtime errors, then label errors, then evidence mismatches.
    errors.sort(key=lambda pair: (
        pair[0].get("parse_status") == "valid",
        pair[0].get("label_for_scoring") == pair[0].get("gold_label"),
        pair[0]["annotation_id"],
    ))
    distinct, seen_patients = [], set()
    for pred, example in errors:
        patient = pred.get("patient_id", example["provenance"].get("patient_id", pred["annotation_id"]))
        if patient in seen_patients:
            continue
        seen_patients.add(patient)
        distinct.append((pred, example))
    lines = [
        f"# {args.split.title()} error ledger: {args.model_label}", "",
        f"**Source predictions:** `{args.predictions.as_posix()}`.",
        ("**Scope:** validation only; protected test data was not opened."
         if args.split != "test" else
         "**Scope:** protected test data after the frozen one-time evaluation; no tuning followed."),
        f"**Observed:** {len(errors)} rows with label, parse, invalid-ID, or evidence-set discrepancy out of {len(predictions)}.",
        "**Interpretation:** issues are directly observed; consequence text describes the operational effect, not an established model rationale. One example per patient is shown.", "",
        "| # | Annotation | Type | Criterion | Gold | Prediction | Parse | Gold evidence excerpts | Predicted excerpts | Observed issue and consequence | Raw output |",
        "|---:|---:|---|---|---|---|---|---|---|---|---|",
    ]
    for n, (pred, example) in enumerate(distinct[:args.limit], 1):
        inp = example["input"]
        sentence_map = {int(item["id"]): item["text"] for item in inp["sentences"]}
        gold_text = excerpt(sentence_map, [int(i) for i in pred.get("gold_evidence_ids", [])])
        pred_text = excerpt(sentence_map, [int(i) for i in pred.get("evidence_ids_validated", [])])
        criterion = str(pred.get("criterion_text", "")).replace("|", "\\|")
        raw = str(pred.get("raw_output", "")).replace("|", "\\|").replace("\n", " ")
        parse_status = pred.get("parse_status", "unknown")
        predicted = pred.get("label_for_scoring")
        gold = pred.get("gold_label")
        invalid = pred.get("invalid_evidence_values", [])
        problems = []
        if parse_status != "valid":
            problems.append("invalid structured output; prediction is not scorable and requires review")
        if predicted != gold:
            problems.append("label mismatch; eligibility category is incorrect for this annotation")
        if set(evidence_pred) != set(evidence_gold):
            problems.append("evidence-set mismatch; source localization is incomplete or overinclusive")
        if invalid:
            problems.append(f"invalid evidence IDs {invalid}; citation cannot be resolved to the source note")
        issue = "; ".join(problems).replace("|", "\\|")
        gold_markdown = gold_text.replace("|", "\\|")
        pred_markdown = pred_text.replace("|", "\\|")
        lines.append(
            f"| {n} | {pred['annotation_id']} | {pred.get('criterion_type')} | {criterion} "
            f"| {pred.get('gold_label')} | {pred.get('label_for_scoring')} | {pred.get('parse_status')} "
            f"| {gold_markdown} | {pred_markdown} | {issue} | `{raw}` |"
        )
    if not errors:
        lines.extend(["", "No errors were found by the label/parse/evidence-set criteria above."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"split": args.split, "predictions": len(predictions), "error_count": len(errors),
                      "distinct_patients_with_errors": len(seen_patients),
                      "written_cases": min(len(distinct), args.limit), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
