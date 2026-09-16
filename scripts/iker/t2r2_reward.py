#!/usr/bin/env python3
"""IKER stage-2 t2r reward round files (spec 2026-09-16-iker-stage2-t2r §7, task 3): render, ingest.

The stage-2 round is driven by hand (spec §0: no loop/round machinery here), so this CLI takes plain file
paths rather than an iteration directory.

    python3 scripts/iker/t2r2_reward.py render --out prompt.md [--previous compute_reward.py] \\
        [--feedback events.json] [--notes notes.md]
    ../IsaacLab/_isaac_sim/python.sh scripts/iker/t2r2_reward.py ingest \\
        --response response.md --out compute_reward.py --report validation.json

--feedback names a JSON file of {tag: [values, ...]} (already extracted from TFEvents), rendered through
render_feedback_table. --notes names a plain-text file of VERIFIED observations (PromptSpec.user_notes): things
the logged tags cannot say, such as a probe's measurement of a predicate the environment does not log. Never put
a hypothesis there — the generator reads it as fact (see memory t2r-notes-verified-facts-only). ingest exits 1
when the generated reward fails validation (the report also holds the errors).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "openarm"))

from openarm.agnostic.tasks.iker_shoe.place_stage import PlaceRewardCfg  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.t2r2 import prompts as P  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.t2r2 import validator as V  # noqa: E402

FENCE = re.compile(r"```python\s*\n(.*?)\n```", re.S)


def extract_code(response: str) -> str | None:
    """The last python block (text2reward: reasoning first, final code last); None without one."""
    blocks = FENCE.findall(response)
    return blocks[-1].strip() + "\n" if blocks else None


def cmd_render(args: argparse.Namespace) -> int:
    previous_code = Path(args.previous).read_text(encoding="utf-8") if args.previous else None
    feedback = None
    if args.feedback:
        series = json.loads(Path(args.feedback).read_text(encoding="utf-8"))
        feedback = P.render_feedback_table(series)
    notes = Path(args.notes).read_text(encoding="utf-8") if args.notes else None
    cfg = PlaceRewardCfg()
    spec = P.PromptSpec(task=P.task_text(cfg), previous_code=previous_code, feedback=feedback, user_notes=notes)
    Path(args.out).write_text(P.render_prompt(spec, cfg=cfg), encoding="utf-8")
    print(f"[t2r2] prompt -> {args.out}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    response = Path(args.response).read_text(encoding="utf-8")
    code = extract_code(response)
    if code is None:
        report = {"ok": False, "errors": ["the response holds no ```python block"], "warnings": [],
                  "term_stats": {}, "total_stats": {}, "fields_used": []}
    else:
        Path(args.out).write_text(code, encoding="utf-8")
        report = V.validate(args.out).as_dict()
    Path(args.report).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("ok", "errors", "warnings", "fields_used")}, ensure_ascii=False, indent=1))
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    render = commands.add_parser("render")
    render.add_argument("--out", required=True)
    render.add_argument("--previous")
    render.add_argument("--feedback")
    render.add_argument("--notes")

    ingest = commands.add_parser("ingest")
    ingest.add_argument("--response", required=True)
    ingest.add_argument("--out", required=True)
    ingest.add_argument("--report", required=True)

    args = parser.parse_args(argv)
    if args.command == "render":
        return cmd_render(args)
    return cmd_ingest(args)


if __name__ == "__main__":
    raise SystemExit(main())
