"""t2r steps of the IKER auto loop (spec 2026-09-15-iker-stage1-t2r §3.4): the isolated reward generator's brief and record,
and the names of a round's files and runs. No torch: scripts/iker/loop.py runs on system python.

The file names mirror tasks/iker_shoe/t2r/pipeline.py (which needs torch); test_loop_cli checks they agree.
"""

from __future__ import annotations

from typing import Mapping

from . import loop_vlm, run_files

GENERATOR_AGENT = "iker-t2r-generator"  # .claude/agents/iker-t2r-generator.md
GENERATOR_TOOLS = ("Read", "Write")
STAGE_DIR = "stage1_t2r"
PROMPT, RESPONSE, CODE, VALIDATION, SMOKE, FEEDBACK, GENERATOR = (
    "prompt.md", "response.md", "compute_reward.py", "validation.json", "smoke.json", "feedback.md", "generator.json",
)


def iter_dir_name(iteration: int) -> str:
    return f"iter_{iteration:02d}"


def run_label(prefix: str, iteration: int) -> str:
    return f"{prefix}_i{iteration:02d}"


def failed_attempt_names(k: int) -> dict[str, str]:
    """Where a failed ingest or a failed round smoke moves its files, so the next decision sees no response and asks the
    generator again (§14, finding 1: a failed smoke is not a crash, so its files are moved aside the same way)."""
    return {RESPONSE: f"response_attempt_{k}.md", VALIDATION: f"validation_attempt_{k}.json",
            CODE: f"compute_reward_attempt_{k}.py", SMOKE: f"smoke_attempt_{k}.json"}


def generator_brief(prompt_path, response_path) -> str:
    """The whole message a fresh generator agent receives: the prompt file and the response path, nothing else."""
    return loop_vlm.generator_brief(prompt_path, response_path, {})


def generator_record(request: Mapping, request_count: int, requested: str) -> dict:
    """generator.json next to the response: which agent was asked, with which tools and brief, and how often."""
    return {
        "schema": run_files.SCHEMA_VERSION, "agent_type": GENERATOR_AGENT, "tools": list(GENERATOR_TOOLS), "request": request_count,
        "requested": requested, "prompt": request["prompt"], "response": request["response"], "brief": request["brief"],
        "note": loop_vlm.GENERATOR_NOTE,
    }
