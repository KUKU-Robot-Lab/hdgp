from __future__ import annotations

import pytest

from vlm.pouring.contracts import (
    ControlMode,
    HighLevelDecision,
    SkillCommand,
    SkillId,
    TaskSpecification,
)


def test_task_specification_accepts_ordered_allowed_plan() -> None:
    spec = TaskSpecification(
        task="pour",
        source_id="right_cup",
        target_id="left_cup",
        nominal_plan=("grasp_lift", "pre_pour_bridge", "bimanual_pour"),
        allowed_skills=("grasp_lift", "pre_pour_bridge", "bimanual_pour", "recovery"),
    )

    assert spec.nominal_plan[0] == "grasp_lift"


@pytest.mark.parametrize(
    "changes",
    [
        {"task": "pick"},
        {"source_id": ""},
        {"target_id": ""},
        {"nominal_plan": ()},
        {"allowed_skills": ("joint_command",)},
        {"nominal_plan": ("bimanual_pour",), "allowed_skills": ("grasp_lift",)},
    ],
)
def test_task_specification_rejects_invalid_or_disallowed_values(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "task": "pour",
        "source_id": "source",
        "target_id": "target",
        "nominal_plan": ("grasp_lift",),
        "allowed_skills": ("grasp_lift", "recovery"),
    }
    values.update(changes)

    with pytest.raises(ValueError):
        TaskSpecification(**values)  # type: ignore[arg-type]


def test_high_level_decision_rejects_conflicting_retry_and_recover() -> None:
    with pytest.raises(ValueError, match="retry and recover"):
        HighLevelDecision(SkillId.RECOVERY, retry=True, recover=True, reason="invalid")


def test_high_level_decision_copies_transition_parameters() -> None:
    parameters = {"height": 0.1}
    decision = HighLevelDecision(SkillId.APPROACH, transition_parameters=parameters)
    parameters["height"] = 0.2

    assert decision.transition_parameters["height"] == 0.1
    with pytest.raises(TypeError):
        decision.transition_parameters["height"] = 0.3  # type: ignore[index]


def test_skill_command_keeps_control_mode_explicit() -> None:
    command = SkillCommand(ControlMode.TASK_SPACE_POSE, (0.1, 0.2, 0.3), "approach")

    assert command.control_mode is ControlMode.TASK_SPACE_POSE
