from __future__ import annotations

import math

import pytest

from vlm.pouring.contracts import ChannelCommand, ControlMode, SkillCommand
from vlm.pouring.fabric_bridge import (
    HoldPoseLatch,
    PalmActionSpace,
    arm_channel_to_action,
    command_to_action,
    euler_zyx_to_quat_wxyz,
    hand_channel_to_action,
    pose_channel_to_palm_pose,
    quat_wxyz_to_euler_zyx,
)

_HOME = (0.36, -0.02, 0.42, math.pi / 2, 0.0, math.pi / 2)
_SPACE = PalmActionSpace(
    home=_HOME,
    low=(0.20, -0.55, 0.20, math.pi / 4, -math.pi / 4, math.pi / 4),
    high=(0.65, 0.22, 0.65, 3 * math.pi / 4, math.pi / 4, 3 * math.pi / 4),
)


def _pose_channel(pose6: tuple[float, ...]) -> ChannelCommand:
    quat = euler_zyx_to_quat_wxyz((pose6[3], pose6[4], pose6[5]))
    return ChannelCommand(ControlMode.TASK_SPACE_POSE, (*pose6[:3], *quat))


def test_euler_quaternion_round_trip_matches_home_convention() -> None:
    quat = euler_zyx_to_quat_wxyz(_HOME[3:6])
    yaw, pitch, roll = quat_wxyz_to_euler_zyx(quat)
    assert yaw == pytest.approx(_HOME[3])
    assert pitch == pytest.approx(_HOME[4])
    assert roll == pytest.approx(_HOME[5])


def test_command_composes_independent_arm_and_hand_channels() -> None:
    command = SkillCommand(
        _pose_channel(_HOME),
        ChannelCommand(ControlMode.HAND_JOINT_TARGETS, (-1.0, 0.5, 2.0)),
        "approach",
    )
    action = command_to_action(command, _SPACE, hand_dim=3, hold_pose=_HOME)
    # arm: home pose encodes to zeros; hand: clamped normalized passthrough.
    assert action == pytest.approx((0.0,) * 6 + (-1.0, 0.5, 1.0), abs=1e-9)


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
    pose = (*_HOME[:3], _HOME[3] - 2 * math.pi, _HOME[4], _HOME[5])
    assert _SPACE.encode(pose) == pytest.approx((0.0,) * 6, abs=1e-9)


def test_arm_policy_slice_is_clamped_and_dimension_checked() -> None:
    channel = ChannelCommand(ControlMode.POLICY_ACTION, (0.1, -0.2, 0.3, 2.0, -2.0, 0.0))
    assert arm_channel_to_action(channel, _SPACE, hold_pose=_HOME) == pytest.approx(
        (0.1, -0.2, 0.3, 1.0, -1.0, 0.0)
    )
    with pytest.raises(ValueError, match="6D"):
        arm_channel_to_action(
            ChannelCommand(ControlMode.POLICY_ACTION, (0.1,)), _SPACE, hold_pose=_HOME
        )


def test_hand_channel_validates_width_for_all_target_modes() -> None:
    for mode in (
        ControlMode.HAND_JOINT_TARGETS,
        ControlMode.HAND_TIP_TARGETS,
        ControlMode.POLICY_ACTION,
    ):
        assert hand_channel_to_action(
            ChannelCommand(mode, (0.5, -0.5)), hand_dim=2
        ) == pytest.approx((0.5, -0.5))
        with pytest.raises(ValueError, match="2D"):
            hand_channel_to_action(ChannelCommand(mode, (0.5,)), hand_dim=2)
    # arm-only mode is rejected on the hand channel.
    with pytest.raises(ValueError, match="hand channel"):
        hand_channel_to_action(
            ChannelCommand(ControlMode.TASK_SPACE_POSE, (0.0,) * 7), hand_dim=7
        )


def test_hold_channels_hold_pose_and_hand() -> None:
    hold = tuple(_HOME[axis] + (0.02 if axis < 3 else 0.1) for axis in range(6))
    for mode in (ControlMode.SAFE_STOP, ControlMode.NO_OP):
        command = SkillCommand(ChannelCommand(mode), ChannelCommand(mode), "abort")
        action = command_to_action(
            command, _SPACE, hand_dim=2, hold_pose=hold, hand_hold=(0.7, 0.7)
        )
        assert action == pytest.approx((*_SPACE.encode(hold), 0.7, 0.7))
    # default hand hold is fully open.
    action = command_to_action(
        SkillCommand.no_op("done"), _SPACE, hand_dim=2, hold_pose=hold
    )
    assert action[6:] == pytest.approx((-1.0, -1.0))


def test_hold_latch_pins_the_arm_pose_at_hold_entry() -> None:
    latch = HoldPoseLatch(num_envs=1)
    move = _pose_channel(_HOME)
    hold = ChannelCommand(ControlMode.NO_OP)

    assert latch.resolve(0, move, _HOME) == _HOME
    first_hold = tuple(_HOME[axis] + 0.01 for axis in range(6))
    assert latch.resolve(0, hold, first_hold) == first_hold
    drifted = tuple(_HOME[axis] - 0.05 for axis in range(6))
    assert latch.resolve(0, hold, drifted) == first_hold
    assert latch.resolve(0, move, drifted) == drifted
    assert latch.resolve(0, hold, drifted) == drifted


def test_pose_channel_requires_seven_values() -> None:
    with pytest.raises(ValueError, match="7D"):
        pose_channel_to_palm_pose(
            ChannelCommand(ControlMode.TASK_SPACE_POSE, (0.0, 0.0, 0.0))
        )


def test_home_outside_box_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside box"):
        PalmActionSpace(home=(1.0,) + _HOME[1:], low=_SPACE.low, high=_SPACE.high)
