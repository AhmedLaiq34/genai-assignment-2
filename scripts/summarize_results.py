"""Summarize only saved predictions and the checksum-verified frozen manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import math
import xml.sax.saxutils as xml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clinical_match.metrics import score_predictions

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_variant(name: str, manifest_hash: str) -> dict:
    folder = ROOT / "results/final_test" / name
    metrics_path = folder / "metrics.json"
    predictions_path = folder / "test_predictions.jsonl"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("final_evaluation", {}).get("manifest_sha256") != manifest_hash:
        raise ValueError(f"{name} result does not match the current frozen manifest")
    if metrics["final_evaluation"].get("predictions_sha256") != sha256(predictions_path):
        raise ValueError(f"{name} prediction file checksum mismatch")
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (metrics.get("scope", {}).get("split") != "test"
            or metrics.get("scope", {}).get("n") != 206
            or metrics.get("scope", {}).get("expected_n") != 206
            or metrics.get("scope", {}).get("full_split") is not True
            or metrics.get("final_evaluation", {}).get("variant") != name
            or metrics.get("n") != len(predictions)):
        raise ValueError(f"{name} metrics do not describe a complete final test run")
    recomputed = score_predictions(predictions)
    def same(left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
        return left == right
    for key, value in recomputed.items():
        if key not in metrics or not same(value, metrics[key]):
            raise ValueError(f"{name} saved metrics do not recompute from predictions at {key}")
    return {"metrics": metrics, "predictions": predictions}


def bar_svg(values: dict[str, float], title: str, filename: str) -> str:
    width, height = 760, 360
    entries = list(values.items())
    max_value = max([v for _, v in entries] + [0.01])
    bar_width = min(120, 500 / max(1, len(entries)))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="30" y="34" font-size="20" font-family="Arial">{xml.escape(title)}</text>',
             '<line x1="80" y1="290" x2="700" y2="290" stroke="#334155"/>']
    for i, (label, value) in enumerate(entries):
        x = 110 + i * (bar_width + 60)
        h = 210 * value / max_value if max_value else 0
        parts.append(f'<rect x="{x:.1f}" y="{290-h:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="#4778c7" rx="5"/>')
        parts.append(f'<text x="{x+bar_width/2:.1f}" y="{280-h:.1f}" text-anchor="middle" font-size="13" font-family="Arial">{value:.3f}</text>')
        parts.append(f'<text x="{x+bar_width/2:.1f}" y="315" text-anchor="middle" font-size="13" font-family="Arial">{xml.escape(label)}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def confusion_svg(classification: dict, title: str) -> str:
    labels, matrix = classification["confusion_matrix_labels"], classification["confusion_matrix"]
    n, cell = len(labels), 42
    left, top = 210, 85
    width, height = left + n * cell + 20, top + n * cell + 55
    maximum = max((max(row) for row in matrix), default=1) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="20" y="30" font-size="18" font-family="Arial">{xml.escape(title)}</text>',
             f'<text x="20" y="58" font-size="12" font-family="Arial">Rows: gold · columns: predicted</text>']
    for i, label in enumerate(labels):
        short = label.replace("not enough information", "not enough info").replace("__invalid__", "invalid")
        parts.append(f'<text x="{left-8}" y="{top+i*cell+25}" text-anchor="end" font-size="10" font-family="Arial">{xml.escape(short)}</text>')
        parts.append(f'<text x="{left+i*cell+20}" y="{top-8}" transform="rotate(-45 {left+i*cell+20} {top-8})" text-anchor="start" font-size="10" font-family="Arial">{xml.escape(short)}</text>')
        for j, value in enumerate(matrix[i]):
            intensity = int(245 - 130 * value / maximum)
            color = f"#{intensity:02x}{intensity+3:02x}ff"
            x, y = left + j * cell, top + i * cell
            parts.extend([f'<rect x="{x}" y="{y}" width="{cell-1}" height="{cell-1}" fill="{color}"/>',
                          f'<text x="{x+cell/2}" y="{y+25}" text-anchor="middle" font-size="11" font-family="Arial">{value}</text>'])
    parts.append('</svg>')
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/frozen_manifest.json")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report/generated")
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_hash != args.manifest_sha256:
        raise ValueError("Frozen manifest checksum mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("state") != "frozen":
        raise ValueError("Only a frozen experiment can be summarized")
    variants = {name: load_variant(name, manifest_hash) for name in ("base", "adapted")}
    expected = set(manifest["expected_test_ids"])
    for name, data in variants.items():
        got = [p["annotation_id"] for p in data["predictions"]]
        if len(got) != len(set(got)) or set(got) != expected:
            raise ValueError(f"{name} predictions do not cover frozen test IDs exactly")

    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = {"manifest_sha256": manifest_hash, "variants": {}}
    rows = []
    for name, payload in variants.items():
        metrics = payload["metrics"]
        summary["variants"][name] = {
            "n": metrics["n"], "classification": metrics["classification"],
            "by_criterion_type": metrics["by_criterion_type"], "evidence": metrics["evidence"],
            "parse_status_counts": metrics["parse_status_counts"],
            "invalid_evidence_value_count": metrics["invalid_evidence_value_count"],
            "truncated_count": metrics["truncated_count"],
            "runtime_failure_count": metrics["runtime_failure_count"],
        }
        rows.append({"variant": name, "n": metrics["n"], "accuracy": metrics["classification"]["accuracy"],
                     "macro_f1": metrics["classification"]["macro_f1"],
                     "evidence_micro_f1": metrics["evidence"]["micro"]["f1"],
                     "parse_failures": sum(v for k, v in metrics["parse_status_counts"].items() if k != "valid"),
                     "invalid_evidence_values": metrics["invalid_evidence_value_count"],
                     "truncated": metrics["truncated_count"], "runtime_failures": metrics["runtime_failure_count"]})
        for kind, block in metrics["by_criterion_type"].items():
            (out / f"confusion_{name}_{kind}.svg").write_text(confusion_svg(block["classification"], f"{name}: {kind} confusion"), encoding="utf-8")
    summary_path = out / "final_test_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for metric in ("macro_f1", "evidence_micro_f1", "accuracy"):
        (out / f"{metric}.svg").write_text(bar_svg({r["variant"]: float(r[metric]) for r in rows}, metric.replace("_", " ").title(), f"{metric}.svg"), encoding="utf-8")

    # Export an epoch-level training trace from the checksummed run artifact.
    history_path = ROOT / "artifacts/phase04/phase04_lora_colab_cli_20261110T1650Z_bundle/training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    losses = history.get("losses", [])
    if losses:
        points = " ".join(f"{40+i*680/max(1,len(losses)-1):.1f},{270-210*float(v)/(max(losses) or 1):.1f}" for i,v in enumerate(losses))
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="760" height="330"><rect width="100%" height="100%" fill="white"/>'
               '<text x="30" y="30" font-family="Arial" font-size="20">Training loss (optimizer steps)</text>'
               f'<polyline fill="none" stroke="#4778c7" stroke-width="2" points="{points}"/></svg>')
        (out / "training_loss.svg").write_text(svg, encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "artifacts": sorted(p.name for p in out.iterdir())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
