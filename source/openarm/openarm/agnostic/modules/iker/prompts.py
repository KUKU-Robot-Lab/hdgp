"""IKER VLM prompts (paper Appendix C), stored as text files in ``prompt_text/``.

The files transcribe the PDF: two-column line wraps are joined, typographic quotes become ASCII
apostrophes, and each sentence group starts on its own line as the paper sets it. Every line was
checked against the text extracted from the PDF (2026-09-13). Tests pin the file hashes, so any
edit to a prompt is deliberate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROMPT_DIR = Path(__file__).resolve().parent / "prompt_text"
SINGLE_STEP = "single_step"
MULTI_STEP = "multi_step"
TASK_MARKER = "[TASK]"
IMAGE_MARKER = "[IMAGE_WITH_KEYPOINTS]"
QUERY_HEADER = "## Query"


@dataclass(frozen=True)
class StageRecord:
    """One earlier stage of a multi-step task: where its image goes and the code the VLM wrote."""

    image_marker: str
    code: str


def load_template(kind: str) -> str:
    if kind not in (SINGLE_STEP, MULTI_STEP):
        raise ValueError(f"unknown prompt kind {kind!r}")
    text = (PROMPT_DIR / f"{kind}.txt").read_text(encoding="utf-8")
    for marker in (TASK_MARKER, IMAGE_MARKER, QUERY_HEADER):
        if text.count(marker) != 1:
            raise ValueError(f"{kind} prompt must contain {marker!r} exactly once")
    return text


def fill_single_step(task: str) -> str:
    _check_task(task)
    return load_template(SINGLE_STEP).replace(TASK_MARKER, task)


def fill_multi_step(task: str, history: Sequence[StageRecord]) -> str:
    """Multi-step prompt; earlier stages are listed in a ``## History`` section before the query."""
    _check_task(task)
    text = load_template(MULTI_STEP).replace(TASK_MARKER, task)
    if not history:
        return text
    lines = ["## History"]
    for number, stage in enumerate(history, start=1):
        lines.append(f"Stage {number} image: {stage.image_marker}")
        lines.append(f"Stage {number} code:\n```python\n{stage.code.strip()}\n```")
    return text.replace(QUERY_HEADER, "\n".join(lines) + "\n" + QUERY_HEADER)


def _check_task(task: str) -> None:
    if not isinstance(task, str) or not task.strip() or "\n" in task:
        raise ValueError("task must be a non-empty single line")
