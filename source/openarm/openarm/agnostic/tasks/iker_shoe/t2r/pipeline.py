"""File steps of one stage-1 t2r round (spec 2026-09-15-iker-stage1-t2r §3, §8): render, ingest and reflect.

Pure torch (no Isaac); scripts/iker/t2r_reward.py is its CLI. An iteration directory holds prompt.md, response.md,
compute_reward.py, validation.json and, after its round, feedback.md.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping, Sequence

from .. import grasp_stage as gs
from . import prompts as P
from . import validator as V

PROMPT, RESPONSE, CODE, VALIDATION, FEEDBACK = "prompt.md", "response.md", "compute_reward.py", "validation.json", "feedback.md"
FENCE = re.compile(r"```python\s*\n(.*?)\n```", re.S)


def extract_code(response: str) -> str | None:
    """The last python block (text2reward: reasoning first, final code last); None without one."""
    blocks = FENCE.findall(response)
    return blocks[-1].strip() + "\n" if blocks else None


def render(iter_dir: Path, *, previous_code: str | None = None, feedback: str | None = None, notes: str | None = None) -> Path:
    iter_dir = Path(iter_dir)
    path = iter_dir / PROMPT
    if path.exists():
        raise FileExistsError(f"{path} exists; a round's prompt is written once")
    iter_dir.mkdir(parents=True, exist_ok=True)
    spec = P.PromptSpec(task=P.task_text(gs.Stage1RewardCfg()), previous_code=previous_code, feedback=feedback, user_notes=notes)
    path.write_text(P.render_prompt(spec), encoding="utf-8")
    return path


def ingest(iter_dir: Path, devices: tuple[str, ...] | None = None) -> dict:
    """response.md -> compute_reward.py + validation.json; returns the report (``ok`` False without a python block)."""
    iter_dir = Path(iter_dir)
    code = extract_code((iter_dir / RESPONSE).read_text(encoding="utf-8"))
    if code is None:
        report = {"ok": False, "errors": ["the response holds no ```python block"], "warnings": [], "term_stats": {},
                  "total_stats": {}, "fields_used": []}
    else:
        (iter_dir / CODE).write_text(code, encoding="utf-8")
        report = V.validate(str(iter_dir / CODE), devices=devices).as_dict()
        report["reward_sha256"] = hashlib.sha256(code.encode("utf-8")).hexdigest()
    (iter_dir / VALIDATION).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def feedback_series(events: Mapping[str, Sequence[tuple[int, float]]]) -> dict[str, list[float]]:
    """TFEvents series -> feedback tags: ``Episode/`` and ``/iter`` are cut, ``/time`` and ``/step`` duplicates are dropped."""
    series = {}
    for raw, points in events.items():
        tag = raw.removeprefix("Episode/")
        if tag.endswith(("/time", "/step")):
            continue
        tag = tag.removesuffix("/iter")
        if any(tag.startswith(prefix) for prefix in P.FEEDBACK_TAG_PREFIXES):
            series[tag] = [float(value) for _, value in points]
    return series


def reflect(prev_dir: Path, next_dir: Path, events: Mapping[str, Sequence[tuple[int, float]]], *, notes: str | None = None,
            n_points: int = 10) -> dict:
    """feedback.md in the finished round's folder and the next round's prompt with its code and that feedback."""
    prev_dir, next_dir = Path(prev_dir), Path(next_dir)
    series = feedback_series(events)
    if not series:
        raise ValueError("no feedback tag in the training events")
    table = P.render_feedback_table(series, n_points=n_points)
    (prev_dir / FEEDBACK).write_text(table, encoding="utf-8")
    render(next_dir, previous_code=(prev_dir / CODE).read_text(encoding="utf-8"), feedback=table, notes=notes)
    return {"tags": len(series), "feedback": str(prev_dir / FEEDBACK), "prompt": str(next_dir / PROMPT)}
