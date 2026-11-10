"""Run one clearly synthetic app-service request and save a text-free smoke record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.main import screen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/dev_checks/app_smoke_summary.json")
    args = parser.parse_args()
    note = "Synthetic demo: A 42-year-old patient has asthma. No cancer history."
    rows = [["inclusion", "Age above 18"]]
    panel, report, _raw_report, status = screen(note, rows, "Library cached greedy")
    passed = bool(report and report.get("criteria") and "<div class='error'>" not in panel
                  and all(item.get("runtime_error") is None for item in report["criteria"]))
    summary = {
        "passed": passed,
        "synthetic_input_only": True,
        "adapter_sha256": report.get("runtime", {}).get("adapter_sha256") if report else None,
        "decoder": report.get("runtime", {}).get("decoder") if report else None,
        "criterion_count": len(report.get("criteria", [])) if report else 0,
        "evidence_statuses": [item.get("evidence_status") for item in report.get("criteria", [])] if report else [],
        "aggregate_outcome": report.get("aggregate", {}).get("outcome") if report else None,
        "no_raw_note_or_excerpts_saved": True,
        "message": status,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
