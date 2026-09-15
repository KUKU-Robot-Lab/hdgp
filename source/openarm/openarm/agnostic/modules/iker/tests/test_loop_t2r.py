"""loop_t2r — the t2r generator request and round names of the IKER auto loop (spec 2026-09-15-iker-stage1-t2r §3; no Isaac, no torch)."""

import os
import subprocess
import sys

from openarm.agnostic.modules.iker import loop_t2r
from openarm.agnostic.tasks.iker_shoe import layout


def test_names_of_a_round():
    assert loop_t2r.iter_dir_name(3) == "iter_03" and loop_t2r.run_label("iker_grasp_c00_t2r", 3) == "iker_grasp_c00_t2r_i03"
    assert loop_t2r.failed_attempt_names(2) == {"response.md": "response_attempt_2.md", "validation.json": "validation_attempt_2.json",
                                                "compute_reward.py": "compute_reward_attempt_2.py", "smoke.json": "smoke_attempt_2.json"}
    assert (loop_t2r.PROMPT, loop_t2r.RESPONSE, loop_t2r.CODE, loop_t2r.VALIDATION, loop_t2r.SMOKE, loop_t2r.FEEDBACK, loop_t2r.GENERATOR) == (
        "prompt.md", "response.md", "compute_reward.py", "validation.json", "smoke.json", "feedback.md", "generator.json")


def test_the_generator_brief_names_only_the_prompt_and_the_response():
    brief = loop_t2r.generator_brief("/l/iter_00/prompt.md", "/l/iter_00/response.md")
    assert brief.startswith("Read the prompt file /l/iter_00/prompt.md") and "/l/iter_00/response.md" in brief
    assert "image" not in brief.lower() and "do not run commands" in brief
    request = {"prompt": "/l/p.md", "response": "/l/r.md", "brief": brief}
    record = loop_t2r.generator_record(request, 2, "2026-09-15T23:00:00")
    assert (record["agent_type"], record["tools"], record["request"], record["brief"]) == ("iker-t2r-generator", ["Read", "Write"], 2, brief)


def test_the_module_does_not_import_torch():
    code = "import sys; import openarm.agnostic.modules.iker.loop_t2r; print('torch' in sys.modules)"
    env = {**os.environ, "PYTHONPATH": str(layout.HDGP_ROOT / "source" / "openarm")}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "False", done.stdout + done.stderr
