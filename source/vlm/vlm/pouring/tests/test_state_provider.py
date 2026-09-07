from __future__ import annotations

import math

import pytest

from vlm.pouring.contracts import SkillId
from vlm.pouring.state_provider import SemanticState


def valid_state(**changes: object) -> SemanticState:
    values: dict[str, object] = {
        "source_pose": (0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0),
        "target_pose": (0.2, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0),
        "source_velocity": (0.0,) * 6,
        "target_velocity": (0.0,) * 6,
        "left_arm_joint_pos": (0.0,) * 7,
        "left_arm_joint_vel": (0.0,) * 7,
        "right_arm_joint_pos": (0.0,) * 7,
        "right_arm_joint_vel": (0.0,) * 7,
        "left_hand_joint_pos": (0.0,) * 20,
        "left_hand_joint_vel": (0.0,) * 20,
        "right_hand_joint_pos": (0.0,) * 20,
        "right_hand_joint_vel": (0.0,) * 20,
        "left_ee_pose": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        "right_ee_pose": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    }
    values.update(changes)
    return SemanticState(**values)  # type: ignore[arg-type]


def test_semantic_state_accepts_fixed_sim_neutral_contract() -> None:
    state = valid_state()

    assert state.current_skill is SkillId.WAIT_FOR_TASK
    assert state.source_confidence == 1.0


def test_semantic_state_rejects_bad_shape_and_non_finite_vector() -> None:
    with pytest.raises(ValueError, match="source_pose"):
        valid_state(source_pose=(0.0,) * 6)
    with pytest.raises(ValueError, match="finite"):
        valid_state(right_arm_joint_pos=(math.nan,) + (0.0,) * 6)


@pytest.mark.parametrize(
    "changes",
    [
        {"source_confidence": -0.1},
        {"target_confidence": 1.1},
        {"joint_limit_margin": math.inf},
        {"contact_count": -1},
        {"skill_elapsed_steps": -1},
    ],
)
def test_semantic_state_rejects_invalid_scalar_state(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        valid_state(**changes)
