"""Map skill commands onto the fabric tasks' absolute action space.

The `agnostic` fabric tasks (grasp_lift_fabric / pour_fabric) take absolute
actions: `a[:6]` encodes a palm 6D pose (env-local xyz + euler_zyx) relative
to the profile home inside the workspace box, and `a[6:]` are absolute hand
joint targets over the full range (`a=-1` fully open, `a=+1` fully closed).

This module is the pure-math half of the Isaac integration boundary. It never
imports Isaac or torch, so the CPU test suite covers it completely. The env's
decode is `desired = clamp(home + a * scale, lo, hi)` with one symmetric
scale per axis; `PalmActionSpace.encode` is its exact inverse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import ControlMode, SkillCommand

_TWO_PI = 2.0 * math.pi


def quat_wxyz_to_euler_zyx(quat: tuple[float, ...]) -> tuple[float, float, float]:
    """Quaternion (w, x, y, z) -> (yaw, pitch, roll) with R = Rz·Ry·Rx.

    Matches Isaac Lab's `euler_xyz_from_quat` reordered as the fabric's
    `euler_zyx` convention (`[yaw, pitch, roll]`).
    """
    if len(quat) != 4:
        raise ValueError(f"quaternion must be wxyz (4 values), got {len(quat)}")
    w, x, y, z = (float(value) for value in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - x * z))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (yaw, pitch, roll)


def euler_zyx_to_quat_wxyz(euler_zyx: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """(yaw, pitch, roll) with R = Rz·Ry·Rx -> quaternion (w, x, y, z)."""
    yaw, pitch, roll = (float(value) for value in euler_zyx)
    if not all(math.isfinite(value) for value in (yaw, pitch, roll)):
        raise ValueError("euler angles must be finite")
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    return (
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    )


def _unwrap_angle(angle: float, reference: float) -> float:
    """Shift `angle` by multiples of 2*pi to land nearest `reference`."""
    return angle + _TWO_PI * round((reference - angle) / _TWO_PI)


@dataclass(frozen=True)
class PalmActionSpace:
    """One robot profile's palm action mapping (identical across envs)."""

    home: tuple[float, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("home", "low", "high"):
            values = getattr(self, name)
            if len(values) != 6:
                raise ValueError(f"{name} must have 6 values, got {len(values)}")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} must contain only finite values")
        for axis in range(6):
            if not self.low[axis] <= self.home[axis] <= self.high[axis]:
                raise ValueError(
                    f"home[{axis}]={self.home[axis]} outside box "
                    f"[{self.low[axis]}, {self.high[axis]}]"
                )

    @property
    def scale(self) -> tuple[float, ...]:
        """Symmetric per-axis scale: max(high-home, home-low), floored at eps."""
        return tuple(
            max(self.high[axis] - self.home[axis], self.home[axis] - self.low[axis], 1e-6)
            for axis in range(6)
        )

    def encode(self, pose: tuple[float, ...]) -> tuple[float, ...]:
        """Absolute palm pose (env-local xyz + euler_zyx) -> action in [-1, 1].

        Angles are first unwrapped toward home so a +/-pi-equivalent target
        never encodes as a full-turn command.
        """
        if len(pose) != 6:
            raise ValueError(f"palm pose must have 6 values, got {len(pose)}")
        if not all(math.isfinite(float(value)) for value in pose):
            raise ValueError("palm pose must contain only finite values")
        scale = self.scale
        action = []
        for axis in range(6):
            value = float(pose[axis])
            if axis >= 3:
                value = _unwrap_angle(value, self.home[axis])
            action.append(
                max(-1.0, min(1.0, (value - self.home[axis]) / scale[axis]))
            )
        return tuple(action)


class HoldPoseLatch:
    """Pin the palm pose at the moment a hold command starts.

    Re-reading the current pose on every tick turns any tracking error into a
    ratchet drift toward the fabric rest pose (measured in the sim demo: a
    DONE hold walked ~180mm back to home over 44 ticks). The latch stores the
    pose once when an env enters SAFE_STOP/NO_OP and releases it on the next
    non-hold command.
    """

    def __init__(self, num_envs: int) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self._held: list[tuple[float, ...] | None] = [None] * num_envs

    def resolve(
        self,
        env_id: int,
        command: SkillCommand,
        current_pose: tuple[float, ...],
    ) -> tuple[float, ...]:
        """Return the hold pose to use for this env on this tick."""
        if not 0 <= env_id < len(self._held):
            raise IndexError(f"env_id {env_id} out of range 0..{len(self._held) - 1}")
        if command.control_mode in (ControlMode.SAFE_STOP, ControlMode.NO_OP):
            held = self._held[env_id]
            if held is None:
                held = tuple(float(value) for value in current_pose)
                self._held[env_id] = held
            return held
        self._held[env_id] = None
        return tuple(float(value) for value in current_pose)


def pose_command_to_palm_pose(command: SkillCommand) -> tuple[float, ...]:
    """TASK_SPACE_POSE values (xyz + quat wxyz) -> palm pose (xyz + euler_zyx)."""
    if command.control_mode is not ControlMode.TASK_SPACE_POSE:
        raise ValueError(f"expected task_space_pose command, got {command.control_mode.value}")
    if len(command.values) != 7:
        raise ValueError(f"task_space_pose values must be 7D (xyz + quat wxyz), got {len(command.values)}")
    position = command.values[:3]
    yaw, pitch, roll = quat_wxyz_to_euler_zyx(command.values[3:])
    return (*position, yaw, pitch, roll)


def command_to_action(
    command: SkillCommand,
    space: PalmActionSpace,
    *,
    hand_dim: int,
    hold_pose: tuple[float, ...],
    hand_action: float = -1.0,
) -> tuple[float, ...]:
    """One skill command -> one fabric-task action row (6 + hand_dim).

    - TASK_SPACE_POSE: encode the target palm pose; hand set to `hand_action`.
    - POLICY_ACTION: pass through unchanged (dimension checked).
    - SAFE_STOP / NO_OP: hold the current palm pose (`hold_pose`), hand held
      at `hand_action` — with absolute actions "hold" is re-commanding the
      current pose, never zeroing the action.
    """
    if hand_dim <= 0:
        raise ValueError("hand_dim must be positive")
    if len(hold_pose) != 6 or not all(math.isfinite(float(v)) for v in hold_pose):
        raise ValueError("hold_pose must be 6 finite values")
    if not math.isfinite(hand_action) or not -1.0 <= hand_action <= 1.0:
        raise ValueError("hand_action must be finite and within [-1, 1]")

    if command.control_mode is ControlMode.POLICY_ACTION:
        expected = 6 + hand_dim
        if len(command.values) != expected:
            raise ValueError(
                f"policy_action must be {expected}D for this profile, got {len(command.values)}"
            )
        return tuple(float(value) for value in command.values)

    if command.control_mode is ControlMode.TASK_SPACE_POSE:
        palm = space.encode(pose_command_to_palm_pose(command))
    elif command.control_mode in (ControlMode.SAFE_STOP, ControlMode.NO_OP):
        palm = space.encode(hold_pose)
    else:  # pragma: no cover - enum is closed, guards future additions
        raise ValueError(f"unsupported control mode: {command.control_mode.value}")
    return (*palm, *((hand_action,) * hand_dim))
