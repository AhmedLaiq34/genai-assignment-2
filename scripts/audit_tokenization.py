"""Audit train/validation prompt lengths without loading protected test data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml
from transformers import AutoTokenizer
from src.clinical_match.data_access import load_examples
from src.clinical_match.prompts import build_prompt, target_json


def stats(values):
    if not values:
        return None
    a = np.asarray(values)
    return {"min": int(a.min()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": int(a.max())}


def run(root: Path, config_path: Path, output_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured_cache = os.environ.get("HF_HUB_CACHE")
    local_cache = root / ".cache/huggingface/hub"
    cache_dir = configured_cache or (str(local_cache) if local_cache.exists() else None)
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"],
                                              cache_dir=cache_dir, local_files_only=True, use_fast=True)
    prompt_sha = hashlib.sha256((root / "src/clinical_match/prompts.py").read_bytes()).hexdigest()
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256((root / "data/splits/manifest.json").read_bytes()).hexdigest()
    report = {"scope": "training and validation only; no protected test examples loaded",
              "model_id": config["model_id"], "model_revision": config["model_revision"],
              "prompt_version": config["version"], "prompt_source_sha256": prompt_sha,
              "prompt_config_sha256": config_sha, "split_manifest_sha256": manifest_sha,
              "max_total_tokens": config["max_total_tokens"], "max_new_tokens": config["max_new_tokens"],
              "selection_policy": "leading whole sentences only, independent of gold evidence",
              "splits": {}}
    for split in ("train", "validation"):
        records = load_examples(root, split, allow_quarantined_training=(split == "train"))
        full, kept, answer, removed = [], [], [], []
        failures, truncated = [], []
        gold_total = gold_retained = nonempty_gold = all_gold_retained = 0
        answer_over_budget = []
        for row in records:
            ident = row["provenance"]["annotation_id"]
            try:
                result = build_prompt(tokenizer, row["input"], max_total_tokens=config["max_total_tokens"],
                                      max_new_tokens=config["max_new_tokens"])
            except ValueError as error:
                failures.append({"annotation_id": ident, "reason": str(error)})
                continue
            full.append(result["full_prompt_tokens"])
            kept.append(len(result["input_ids"]))
            removed.append(len(result["removed_sentence_ids"]))
            if result["truncated"]:
                truncated.append(ident)
            gold = set(row["target"]["evidence_sentence_ids"])
            kept_ids = set(result["kept_sentence_ids"])
            gold_total += len(gold)
            gold_retained += len(gold & kept_ids)
            nonempty_gold += bool(gold)
            all_gold_retained += bool(gold) and gold <= kept_ids
            ans = len(tokenizer.encode(target_json(row["target"]), add_special_tokens=False)) + 1
            answer.append(ans)
            if ans > config["max_new_tokens"]:
                answer_over_budget.append(ident)
        report["splits"][split] = {
            "records": len(records), "successfully_budgeted": len(kept), "failures": failures,
            "full_prompt_token_lengths": stats(full), "kept_prompt_token_lengths": stats(kept),
            "target_token_lengths_including_eos": stats(answer), "targets_exceeding_output_reserve_ids": answer_over_budget,
            "truncated_count": len(truncated), "truncated_annotation_ids": truncated,
            "removed_sentences_total": sum(removed), "removed_sentences_per_record": stats(removed),
            "gold_evidence_ids_total": gold_total, "gold_evidence_ids_retained": gold_retained,
            "gold_evidence_id_retention": gold_retained / gold_total if gold_total else None,
            "records_with_nonempty_gold": nonempty_gold, "records_with_all_gold_retained": all_gold_retained,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/prompt.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tokenization_audit.json")
    args = parser.parse_args()
    report = run(ROOT, args.config, args.output)
    print(json.dumps(report, indent=2))
    if any(v["failures"] or v["targets_exceeding_output_reserve_ids"] for v in report["splits"].values()):
        raise SystemExit("Prompt budget failures require review before training")


if __name__ == "__main__":
    main()
