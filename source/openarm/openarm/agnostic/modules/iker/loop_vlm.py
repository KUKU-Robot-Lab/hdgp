"""VLM steps of the IKER auto loop (design 2026-09-14-iker-auto-loop §6).

The generator is a fresh Claude Code agent that receives only ``generator_brief``: the prompt file, the images and the
response path. The first stage uses the single-step prompt; the re-query after execution uses the multi-step prompt
with the first stage's code as history and is only parsed: ``None`` from the function means done.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import interaction, prompts, run_files

STAGE_IMAGE_MARKER = "[STAGE_1_IMAGE]"
GENERATOR_AGENT = "iker-vlm-generator"  # .claude/agents/iker-vlm-generator.md
GENERATOR_TOOLS = ("Read", "Write")
GENERATOR_NOTE = (
    "The agent definition limits the generator to the Read and Write tools and to the files its brief names; "
    "Claude Code may still inject harness-level context (CLAUDE.md files, auto-memory) into the agent."
)


def target_prompt(task: str) -> str:
    return prompts.fill_single_step(task)


def requery_prompt(task: str, stage_response: str) -> str:
    """Multi-step prompt whose history holds the code of the response that set the stage-1 target."""
    code = interaction.extract_code_block(stage_response)
    return prompts.fill_multi_step(task, [prompts.StageRecord(STAGE_IMAGE_MARKER, code)])


def requery_result(response: str, keypoints: Mapping[int, Sequence[float]]) -> dict:
    """{"done": bool, "detail": str}: done only when ``get_interaction_data`` returns None (``done=True``)."""
    try:
        result = interaction.run_interaction(interaction.extract_code_block(response), keypoints)
    except interaction.InteractionError as exc:
        return {"done": False, "detail": f"G1: {exc}"}
    if result.done:
        return {"done": True, "detail": "get_interaction_data returned None (done=True)"}
    return {"done": False, "detail": f"new stage: move {result.object_name} keypoints {list(result.keypoint_ids)}"}


def generator_record(request: Mapping, request_count: int, requested: str) -> dict:
    """generator.json next to the response: which agent was asked, with which tools and brief, and how often (§6)."""
    return {
        "schema": run_files.SCHEMA_VERSION, "agent_type": GENERATOR_AGENT, "tools": list(GENERATOR_TOOLS), "request": request_count,
        "requested": requested, "prompt": request["prompt"], "images": dict(request["images"]), "response": request["response"],
        "brief": request["brief"], "note": GENERATOR_NOTE,
    }


def generator_brief(prompt_path, response_path, images: Mapping[str, str]) -> str:
    """The whole message a fresh generator agent receives (§6 generator isolation)."""
    lines = [f"Read the prompt file {prompt_path} and answer it exactly as it instructs."]
    lines += [f"The image for {marker} is {path}; open it with the Read tool." for marker, path in images.items()]
    lines += [
        f"Write your complete answer, with the single python code block the prompt asks for, to {response_path}.",
        "Read only these files. Do not open, list or search any other file or directory, and do not run commands.",
    ]
    return "\n".join(lines)
