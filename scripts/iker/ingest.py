#!/usr/bin/env python3
"""Gate one VLM response for an IKER configuration (design spec §6). No Isaac needed.

Reads <run-dir>/attempt_NN/response.md, writes attempt_NN/gate.json, and on a pass writes
<run-dir>/interaction_vlm.json. Exit code 0 only when the gate passes.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm python3 scripts/iker/ingest.py \
        --run-dir iker_runs/shoe_place/config_00 --attempt 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openarm.agnostic.modules.iker import gate, run_files
from openarm.agnostic.tasks.iker_shoe import layout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate one VLM response.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--meta", type=Path, default=layout.SHOE_META_PATH)
    args = parser.parse_args(argv)
    run_dir = args.run_dir if args.run_dir.is_absolute() else layout.HDGP_ROOT / args.run_dir
    attempt_dir = run_dir / f"attempt_{args.attempt:02d}"
    response_path = attempt_dir / "response.md"
    if not response_path.is_file():
        raise FileNotFoundError(f"no VLM response at {response_path}")
    doc = run_files.read_json(run_dir / "keypoints.json")
    meta = run_files.load_shoe_meta(args.meta)
    movables = run_files.movable_objects(doc, meta, layout.supports())
    interaction, report = gate.gate_response(
        response_path.read_text(encoding="utf-8"),
        run_files.vlm_keypoints(doc),
        movables,
        run_files.static_keypoint_ids(doc),
        layout.gate_config(),
    )
    run_files.write_json(attempt_dir / "gate.json", {"schema": run_files.SCHEMA_VERSION, **run_files.gate_document(report)})
    print(f"attempt {args.attempt:02d}: gate passed={report.passed} failures={list(report.failures)}")
    if report.passed and interaction is not None:
        out = run_files.write_json(
            run_dir / "interaction_vlm.json",
            run_files.interaction_document("vlm", interaction, report, extra={"attempt": args.attempt}),
        )
        print(f"wrote {out}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
