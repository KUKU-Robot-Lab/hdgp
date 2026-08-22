from __future__ import annotations

import math

import pytest

from vlm.pouring.contracts import SkillId
from vlm.pouring.isaac_state import FabricStateProvider


class FakeView:
    def __init__(self, num_envs: int, palm_z: float = 0.42, object_z: float = 0.30) -> None:
        self._n = num_envs
        self.palm = [[0.30, -0.20, palm_z, math.pi / 2, 0.0, math.pi / 2]] * num_envs
        self.object = [[0.30, -0.20, object_z, 1.0, 0.0, 0.0, 0.0]] * num_envs

    @property
    def num_envs(self) -> int:
        return self._n

    def palm_pose_zyx(self):
        return self.palm

    def object_pose(self):
        return self.object

    def object_velocity(self):
        return [[0.0] * 6] * self._n

    def arm_joint_pos(self):
        return [[0.1] * 7] * self._n

    def arm_joint_vel(self):
        return [[0.0] * 7] * self._n

    def hand_joint_pos(self):
        return [[0.2] * 20] * self._n

    def hand_joint_vel(self):
        return [[0.0] * 20] * self._n


class FakeRouting:
    def __init__(self, skills, elapsed=None) -> None:
        self.current_skills = list(skills)
        self.elapsed_steps = list(elapsed or [0] * len(self.current_skills))


def test_states_carry_scene_and_routing_data() -> None:
    provider = FabricStateProvider(
        FakeView(2),
        FakeRouting([SkillId.WAIT_FOR_TASK, SkillId.APPROACH], elapsed=[0, 7]),
        approach_offset=(0.0, 0.0, 0.12),
    )

    states = provider.get_states()

    assert len(states) == 2
    assert states[0].current_skill is SkillId.WAIT_FOR_TASK
    assert states[1].current_skill is SkillId.APPROACH
    assert states[1].skill_elapsed_steps == 7
    assert states[1].right_arm_joint_pos == (0.1,) * 7
    assert states[1].left_arm_joint_pos == (0.0,) * 7
    assert states[1].source_pose[:3] == (0.30, -0.20, 0.30)
    # right_ee_pose round-trips the palm euler_zyx into a quaternion.
    assert states[1].right_ee_pose[:3] == (0.30, -0.20, 0.42)
    assert sum(v * v for v in states[1].right_ee_pose[3:]) == pytest.approx(1.0)


def test_approach_success_fires_only_within_tolerance_and_only_for_approach() -> None:
    # palm exactly at object + offset -> success for APPROACH, not WAIT_FOR_TASK.
    view = FakeView(2, palm_z=0.42, object_z=0.30)
    provider = FabricStateProvider(
        view,
        FakeRouting([SkillId.APPROACH, SkillId.WAIT_FOR_TASK]),
        approach_offset=(0.0, 0.0, 0.12),
        approach_tolerance=0.03,
    )
    states = provider.get_states()
    assert states[0].current_skill_success is True
    assert states[1].current_skill_success is False

    far = FabricStateProvider(
        FakeView(1, palm_z=0.60, object_z=0.30),
        FakeRouting([SkillId.APPROACH]),
        approach_offset=(0.0, 0.0, 0.12),
        approach_tolerance=0.03,
    )
    assert far.get_states()[0].current_skill_success is False


def test_left_side_mounts_data_on_left_slots() -> None:
    provider = FabricStateProvider(
        FakeView(1), FakeRouting([SkillId.APPROACH]), controlled_side="left"
    )
    state = provider.get_states()[0]
    assert state.left_arm_joint_pos == (0.1,) * 7
    assert state.right_arm_joint_pos == (0.0,) * 7


def test_batch_size_mismatch_is_rejected() -> None:
    provider = FabricStateProvider(FakeView(2), FakeRouting([SkillId.APPROACH]))
    with pytest.raises(ValueError, match="routing tracks 1"):
        provider.get_states()
