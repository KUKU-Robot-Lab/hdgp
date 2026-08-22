from __future__ import annotations

import math

import pytest

from vlm.pouring.contracts import ControlMode, SkillCommand
from vlm.pouring.fabric_bridge import (
    PalmActionSpace,
    command_to_action,
    euler_zyx_to_quat_wxyz,
    pose_command_to_palm_pose,
    quat_wxyz_to_euler_zyx,
)

_HOME = (0.36, -0.02, 0.42, math.pi / 2, 0.0, math.pi / 2)
_SPACE = PalmActionSpace(
    home=_HOME,
    low=(0.20, -0.55, 0.20, math.pi / 4, -math.pi / 4, math.pi / 4),
    high=(0.65, 0.22, 0.65, 3 * math.pi / 4, math.pi / 4, 3 * math.pi / 4),
)


def test_euler_quaternion_round_trip_matches_home_convention() -> None:
    quat = euler_zyx_to_quat_wxyz(_HOME[3:6])
    yaw, pitch, roll = quat_wxyz_to_euler_zyx(quat)
    assert yaw == pytest.approx(_HOME[3])
    assert pitch == pytest.approx(_HOME[4])
    assert roll == pytest.approx(_HOME[5])


def test_encode_home_is_zero_action() -> None:
    assert command_to_action(
        SkillCommand(
            ControlMode.TASK_SPACE_POSE,
            (*_HOME[:3], *euler_zyx_to_quat_wxyz(_HOME[3:6])),
            "approach",
        ),
        _SPACE,
        hand_dim=3,
        hold_pose=_HOME,
    ) == pytest.approx((0.0,) * 6 + (-1.0,) * 3, abs=1e-9)


def test_encode_inverts_the_env_symmetric_decode() -> None:
    scale = _SPACE.scale
    pose = tuple(_HOME[axis] + 0.4 * scale[axis] for axis in range(6))
    action = _SPACE.encode(pose)
    assert action == pytest.approx((0.4,) * 6)
    decoded = tuple(
        min(max(_HOME[axis] + action[axis] * scale[axis], _SPACE.low[axis]), _SPACE.high[axis])
        for axis in range(6)
    )
    assert decoded == pytest.approx(pose)


def test_encode_clamps_targets_beyond_the_box() -> None:
    pose = (10.0, -10.0, _HOME[2], *_HOME[3:6])
    action = _SPACE.encode(pose)
    assert action[0] == 1.0
    assert action[1] == -1.0
    assert action[2:] == pytest.approx((0.0,) * 4)


def test_encode_unwraps_angles_toward_home() -> None:
    # yaw of -3*pi/2 is the same rotation as +pi/2 (home) — must encode as 0.
    pose = (*_HOME[:3], _HOME[3] - 2 * math.pi, _HOME[4], _HOME[5])
    assert _SPACE.encode(pose) == pytest.approx((0.0,) * 6, abs=1e-9)


def test_policy_action_passes_through_with_dimension_check() -> None:
    values = tuple(0.1 * i for i in range(9))
    command = SkillCommand(ControlMode.POLICY_ACTION, values, "grasp_lift")
    assert command_to_action(command, _SPACE, hand_dim=3, hold_pose=_HOME) == values
    with pytest.raises(ValueError, match="policy_action must be 9D"):
        command_to_action(
            SkillCommand(ControlMode.POLICY_ACTION, values[:5], "grasp_lift"),
            _SPACE,
            hand_dim=3,
            hold_pose=_HOME,
        )


def test_safe_stop_and_no_op_hold_the_current_pose() -> None:
    hold = tuple(_HOME[axis] + (0.02 if axis < 3 else 0.1) for axis in range(6))
    expected = (*_SPACE.encode(hold), -1.0, -1.0)
    for mode in (ControlMode.SAFE_STOP, ControlMode.NO_OP):
        action = command_to_action(
            SkillCommand(mode, (), "abort"),
            _SPACE,
            hand_dim=2,
            hold_pose=hold,
        )
        assert action == pytest.approx(expected)


def test_hold_latch_pins_the_pose_at_hold_entry() -> None:
    from vlm.pouring.fabric_bridge import HoldPoseLatch

    latch = HoldPoseLatch(num_envs=1)
    move = SkillCommand(ControlMode.TASK_SPACE_POSE, (0.0,) * 7, "approach")
    hold = SkillCommand(ControlMode.NO_OP, (), "done")

    assert latch.resolve(0, move, _HOME) == _HOME
    first_hold = tuple(_HOME[axis] + 0.01 for axis in range(6))
    assert latch.resolve(0, hold, first_hold) == first_hold
    # The pose keeps drifting, but the latch must keep returning the entry pose.
    drifted = tuple(_HOME[axis] - 0.05 for axis in range(6))
    assert latch.resolve(0, hold, drifted) == first_hold
    # A non-hold command releases the latch.
    assert latch.resolve(0, move, drifted) == drifted
    assert latch.resolve(0, hold, drifted) == drifted


def test_pose_command_requires_seven_values() -> None:
    with pytest.raises(ValueError, match="7D"):
        pose_command_to_palm_pose(
            SkillCommand(ControlMode.TASK_SPACE_POSE, (0.0, 0.0, 0.0), "approach")
        )


def test_home_outside_box_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside box"):
        PalmActionSpace(home=(1.0,) + _HOME[1:], low=_SPACE.low, high=_SPACE.high)
