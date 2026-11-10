"""Screen a synthetic note and criteria with the local model; no test data is read."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import transformers
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.clinical_match.decoding import cached_explicit_greedy, cached_library_greedy
from src.clinical_match.evaluation import hash_directory, load_eval_model
from src.clinical_match.screening import ScreeningService, aggregate_results

DEFAULT_CASE = ROOT / "examples/synthetic_screening_case.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval.yaml")
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--decoder",
        choices=["library", "explicit_cached"],
        default="library",
        help="Library cached oracle or the explicit KV-cache decoder.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.device:
        cfg["device"] = args.device
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    case = json.loads(args.case.read_text(encoding="utf-8"))
    if args.adapter is not None:
        cfg["adapter_path"] = str(args.adapter)
        cfg["adapter_sha256"] = hash_directory(args.adapter)
    tokenizer, model = load_eval_model(cfg, device, adapter_path=args.adapter)
    generate_fn = (
        cached_explicit_greedy
        if args.decoder == "explicit_cached"
        else cached_library_greedy
    )
    service = ScreeningService(
        tokenizer,
        model,
        cfg,
        device,
        generate_fn=generate_fn,
    )
    results = [
        service.screen(
            case["patient_sentences"],
            criterion["criterion_type"],
            criterion["criterion_text"],
            generation_mode=args.decoder,
        )
        for criterion in case["criteria"]
    ]
    report = {
        "notice": case.get("notice"),
        "runtime": {
            "decoder": args.decoder,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "model_id": cfg["model_id"],
            "model_revision": cfg["model_revision"],
            "adapter_path": str(args.adapter) if args.adapter is not None else None,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "criteria": results,
        "aggregate": aggregate_results(results),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
