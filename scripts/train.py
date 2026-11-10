"""Train a Phase 4 LoRA adapter on the permitted TrialGPT training split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.clinical_match.training.trainer import Trainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    parser.add_argument("--run-name", default=None, help="Unique run/inventory directory name")
    parser.add_argument("--resume", type=Path, default=None, help="Verified resume bundle or state directory")
    parser.add_argument("--max-steps", type=int, default=None, help="Planned total optimizer steps")
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        default=None,
        help="Stop this run early while keeping the scheduler plan from --max-steps",
    )
    parser.add_argument("--smoke", action="store_true", help="One optimizer step on two examples; no validation")
    parser.add_argument("--overfit-smoke", action="store_true", help="Fit a small training subset and check learning")
    parser.add_argument("--overfit-examples", type=int, default=12)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--tiny-model", action="store_true", help="Use a tiny random Qwen2 model for local plumbing tests")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--drive-dir", type=Path, default=None, help="Persistent bundle root, e.g. a Drive folder")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.device:
        cfg["device"] = args.device
    if args.tiny_model:
        cfg["tiny_model"] = True
    cfg["train_config_path"] = str(
        args.config.resolve().relative_to(ROOT)
        if args.config.resolve().is_relative_to(ROOT)
        else args.config.resolve()
    )

    max_steps = args.max_steps
    if args.smoke:
        max_steps = max_steps or 1
        args.skip_validation = True
        args.overfit_smoke = False
    if args.overfit_smoke:
        max_steps = max_steps or 40

    run_name = args.run_name or cfg["run_name"]
    if args.smoke and not args.run_name:
        run_name = f"{run_name}_smoke"
    elif args.overfit_smoke and not args.run_name:
        run_name = f"{run_name}_overfit"

    trainer = Trainer(
        cfg,
        ROOT,
        run_name=run_name,
        tiny_model=args.tiny_model,
        drive_dir=args.drive_dir,
    )
    summary = trainer.run(
        resume=args.resume,
        max_steps=max_steps,
        stop_after_steps=args.stop_after_steps,
        skip_validation=args.skip_validation,
        overfit=args.overfit_smoke,
        overfit_examples=args.overfit_examples,
    )
    print(f"Training summary written for {summary['run_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
