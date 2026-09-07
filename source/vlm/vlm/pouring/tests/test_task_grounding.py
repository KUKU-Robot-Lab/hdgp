from __future__ import annotations

import pytest

from vlm.pouring.task_grounding import parse_task_specification

VALID = """```json
{"task":"pour","source_id":"cup_2","target_id":"cup_5",
"nominal_plan":["grasp_lift","pre_pour_bridge","bimanual_pour"],
"allowed_skills":["grasp_lift","pre_pour_bridge","bimanual_pour","recovery"]}
```"""


def test_parse_task_specification_accepts_one_fenced_json_object() -> None:
    result = parse_task_specification(VALID)

    assert result.source_id == "cup_2"
    assert result.target_id == "cup_5"


def test_parse_task_specification_accepts_one_plain_json_object() -> None:
    result = parse_task_specification(VALID.removeprefix("```json\n").removesuffix("\n```"))

    assert result.task == "pour"


@pytest.mark.parametrize(
    "text",
    [
        "no json",
        "{} {}",
        '{"task":"pour","source_id":"a","target_id":"b",'
        '"nominal_plan":["grasp_lift"],"allowed_skills":["grasp_lift"],"joint_command":[1]}',
        '{"task":"pour","source_id":"a","target_id":"b",'
        '"nominal_plan":"grasp_lift","allowed_skills":["grasp_lift"]}',
    ],
)
def test_parse_task_specification_rejects_missing_multiple_or_extra_control_fields(text: str) -> None:
    with pytest.raises(ValueError):
        parse_task_specification(text)
