"""Rehearse an interrupted/resumed training run against an uninterrupted control."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_train(*args: str) -> None:
    command = [sys.executable, str(ROOT / "scripts/train.py"), *args]
    print("RUN:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def latest_state(run_name: str, config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bundle_root = ROOT / cfg["checkpoint"]["local_root"] / run_name / "bundles"
    candidates = sorted(
        path for path in bundle_root.iterdir() if path.is_dir() and path.name.startswith("resume-step-")
    )
    if not candidates:
        raise FileNotFoundError(f"No resume bundle under {bundle_root}")
    latest = candidates[-1]
    return json.loads((latest / "trainer_state.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    parser.add_argument("--overfit-examples", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--interrupt-at", type=int, default=2)
    parser.add_argument("--tiny-model", action="store_true")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/training/resume_rehearsal.json")
    args = parser.parse_args()
    if args.interrupt_at <= 0 or args.interrupt_at >= args.max_steps:
        raise ValueError("interrupt-at must be positive and smaller than max-steps")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    control = f"rehearsal_control_{stamp}"
    interrupted = f"rehearsal_interrupted_{stamp}"
    resumed = f"rehearsal_resumed_{stamp}"
    common = [
        "--config", str(args.config),
        "--overfit-smoke",
        "--overfit-examples", str(args.overfit_examples),
    ]
    if args.tiny_model:
        common.append("--tiny-model")
    if args.device:
        common.extend(["--device", args.device])

    run_train(*common, "--run-name", control, "--max-steps", str(args.max_steps))
    run_train(
        *common,
        "--run-name", interrupted,
        "--max-steps", str(args.max_steps),
        "--stop-after-steps", str(args.interrupt_at),
    )

    interrupted_checks = ROOT / "checkpoints" / interrupted / "bundles"
    bundles = sorted(
        path for path in interrupted_checks.iterdir() if path.is_dir() and path.name.startswith("resume-step-")
    )
    resume_bundle = bundles[-1]
    run_train(
        *common,
        "--run-name", resumed,
        "--resume", str(resume_bundle),
        "--max-steps", str(args.max_steps),
    )

    control_state = latest_state(control, args.config)
    resumed_state = latest_state(resumed, args.config)
    control_history = json.loads(
        (ROOT / "results/training" / control / "training_history.json").read_text(encoding="utf-8")
    )
    resumed_history = json.loads(
        (ROOT / "results/training" / resumed / "training_history.json").read_text(encoding="utf-8")
    )
    control_losses = control_history["losses"]
    resumed_losses = resumed_history["losses"]
    max_loss_delta = max(
        (abs(a - b) for a, b in zip(control_losses, resumed_losses)), default=None
    )
    report = {
        "control_run": control,
        "interrupted_run": interrupted,
        "resumed_run": resumed,
        "resume_bundle": str(resume_bundle),
        "max_steps": args.max_steps,
        "interrupt_at": args.interrupt_at,
        "control_global_step": control_state["global_step"],
        "resumed_global_step": resumed_state["global_step"],
        "control_epoch": control_state["epoch"],
        "resumed_epoch": resumed_state["epoch"],
        "control_example_cursor": control_state["example_cursor"],
        "resumed_example_cursor": resumed_state["example_cursor"],
        "orders_equal": control_state["order"] == resumed_state["order"],
        "control_loss_count": len(control_losses),
        "resumed_loss_count": len(resumed_losses),
        "max_abs_loss_delta": max_loss_delta,
        "losses_finite": all(math.isfinite(value) for value in control_losses + resumed_losses),
    }
    report["passed"] = bool(
        report["control_global_step"] == report["resumed_global_step"]
        and report["control_epoch"] == report["resumed_epoch"]
        and report["control_example_cursor"] == report["resumed_example_cursor"]
        and report["orders_equal"]
        and report["control_loss_count"] == report["resumed_loss_count"]
        and report["losses_finite"]
        and (max_loss_delta is None or max_loss_delta <= 1e-2)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
