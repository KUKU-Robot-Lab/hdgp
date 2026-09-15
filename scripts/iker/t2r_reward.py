#!/usr/bin/env python3
"""IKER stage-1 t2r round files (spec 2026-09-15-iker-stage1-t2r §3): render, ingest, reflect.

    python3 scripts/iker/t2r_reward.py render  --iter-dir <loop>/stage1_t2r/iter_00
    ../IsaacLab/_isaac_sim/python.sh scripts/iker/t2r_reward.py ingest --iter-dir <iter>   # Isaac python: the cuda dry run runs
    python3 scripts/iker/t2r_reward.py reflect --prev-dir <iter_00> --next-dir <iter_01> --events <run>/summaries/events.out.tfevents.*

ingest exits 1 when the generated reward fails validation (validation.json holds the errors).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "openarm"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from openarm.agnostic.tasks.iker_shoe.t2r import pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render")
    render.add_argument("--iter-dir", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--iter-dir", required=True)
    reflect = commands.add_parser("reflect")
    reflect.add_argument("--prev-dir", required=True)
    reflect.add_argument("--next-dir", required=True)
    reflect.add_argument("--events", required=True)
    reflect.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    if args.command == "render":
        print(f"[t2r] prompt -> {pipeline.render(Path(args.iter_dir))}")
        return 0
    if args.command == "ingest":
        report = pipeline.ingest(Path(args.iter_dir))
        print(json.dumps({key: report[key] for key in ("ok", "errors", "warnings", "fields_used")}, ensure_ascii=False, indent=1))
        return 0 if report["ok"] else 1
    from parse_tfevents import load_tfevents

    notes = Path(args.notes).read_text(encoding="utf-8") if args.notes else None
    result = pipeline.reflect(Path(args.prev_dir), Path(args.next_dir), load_tfevents(args.events), notes=notes)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
