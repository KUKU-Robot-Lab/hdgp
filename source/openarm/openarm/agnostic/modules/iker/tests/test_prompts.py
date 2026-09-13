"""prompts — paper Appendix C text and filling (no Isaac)."""

import hashlib

import pytest

from openarm.agnostic.modules.iker import prompts

TASK = "Place the shoe on the rack next to the other shoe."
# Pinned transcriptions; a change here must be a deliberate prompt edit.
PROMPT_SHA256 = {
    prompts.SINGLE_STEP: "ddcb3beeb3db0bc54edbe2036eb78a486a976fd8d404e7b33151d69a5aae597b",
    prompts.MULTI_STEP: "f6335691c1f28c0a0ddfb99320d91111f44506f490e333748c098386e53597e0",
}


def test_prompt_files_are_the_pinned_appendix_c_transcription():
    for kind, digest in PROMPT_SHA256.items():
        assert hashlib.sha256((prompts.PROMPT_DIR / f"{kind}.txt").read_bytes()).hexdigest() == digest


def test_single_step_fill_inserts_the_task_and_keeps_the_image_marker():
    text = prompts.fill_single_step(TASK)
    assert text.startswith("## Instructions\nYour job is to help with moving rigid objects in real-world by writing code in python.\n")
    assert f"Query Task: '{TASK}'" in text
    assert prompts.TASK_MARKER not in text
    assert text.count(prompts.IMAGE_MARKER) == 1


def test_multi_step_prompt_differs_where_the_paper_differs():
    single = prompts.load_template(prompts.SINGLE_STEP)
    multi = prompts.load_template(prompts.MULTI_STEP)
    assert "Generally, big objects can only be pushed" in multi and "Generally, big objects" not in single
    assert "done = ?" in multi and "done = ?" not in single


def test_multi_step_history_is_placed_before_the_query():
    text = prompts.fill_multi_step(TASK, [prompts.StageRecord("[IMAGE_STAGE_1]", "def f():\n    return\n")])
    assert text.index("## History") < text.index("## Query")
    assert "Stage 1 image: [IMAGE_STAGE_1]\nStage 1 code:\n```python\ndef f():\n    return\n```" in text


@pytest.mark.parametrize("task", ["", "   ", "two\nlines"])
def test_task_must_be_a_single_non_empty_line(task):
    with pytest.raises(ValueError, match="single line"):
        prompts.fill_single_step(task)
