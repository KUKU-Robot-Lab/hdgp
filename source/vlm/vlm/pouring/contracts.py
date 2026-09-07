from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


class SkillId(str, Enum):
    """Stable skill identifiers shared by grounding, policy, and execution."""

    WAIT_FOR_TASK = "wait_for_task"
    APPROACH = "approach"
    PRE_GRASP_BRIDGE = "pre_grasp_bridge"
    GRASP_LIFT = "grasp_lift"
    PRE_POUR_BRIDGE = "pre_pour_bridge"
    BIMANUAL_POUR = "bimanual_pour"
    RECOVERY = "recovery"
    ABORT = "abort"
    DONE = "done"


class ControlMode(str, Enum):
    """Command representations that must never be blended implicitly.

    Arm and hand are separate articulations with separate fabric controllers,
    so a skill command carries one mode per channel (user decision, 08.23):
    - arm: TASK_SPACE_POSE (xyz + quat wxyz), POLICY_ACTION (normalized palm
      slice of a trained policy's output), SAFE_STOP, NO_OP.
    - hand: HAND_JOINT_TARGETS (normalized [-1,1] absolute joint targets,
      -1 open / +1 closed), HAND_TIP_TARGETS (normalized [-1,1] palm-relative
      fingertip targets — the fabric tip-IK convention), POLICY_ACTION,
      SAFE_STOP, NO_OP.
    """

    TASK_SPACE_POSE = "task_space_pose"
    POLICY_ACTION = "policy_action"
    HAND_JOINT_TARGETS = "hand_joint_targets"
    HAND_TIP_TARGETS = "hand_tip_targets"
    SAFE_STOP = "safe_stop"
    NO_OP = "no_op"


ARM_MODES = frozenset({
    ControlMode.TASK_SPACE_POSE,
    ControlMode.POLICY_ACTION,
    ControlMode.SAFE_STOP,
    ControlMode.NO_OP,
})
HAND_MODES = frozenset({
    ControlMode.HAND_JOINT_TARGETS,
    ControlMode.HAND_TIP_TARGETS,
    ControlMode.POLICY_ACTION,
    ControlMode.SAFE_STOP,
    ControlMode.NO_OP,
})
HOLD_MODES = frozenset({ControlMode.SAFE_STOP, ControlMode.NO_OP})


@dataclass(frozen=True)
class TaskSpecification:
    """Validated low-frequency task metadata emitted by the VLM."""

    task: str
    source_id: str
    target_id: str
    nominal_plan: tuple[str, ...]
    allowed_skills: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.task != "pour":
            raise ValueError("v1 supports only task='pour'")
        if not self.source_id or not self.target_id:
            raise ValueError("source_id and target_id must be non-empty")
        if not self.nominal_plan:
            raise ValueError("nominal_plan must be non-empty")
        valid = {item.value for item in SkillId}
        allowed = set(self.allowed_skills)
        if not allowed <= valid:
            raise ValueError("allowed_skills contains an unknown skill")
        if not set(self.nominal_plan) <= allowed:
            raise ValueError("nominal_plan must be a subset of allowed_skills")


@dataclass(frozen=True)
class HighLevelDecision:
    """One high-level decision for one environment."""

    skill_id: SkillId
    terminate_current_skill: bool = False
    retry: bool = False
    recover: bool = False
    transition_parameters: Mapping[str, float] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        if self.retry and self.recover:
            raise ValueError("retry and recover cannot both be true")
        copied = {str(key): float(value) for key, value in self.transition_parameters.items()}
        object.__setattr__(self, "transition_parameters", MappingProxyType(copied))


@dataclass(frozen=True)
class ChannelCommand:
    """One articulation channel's command (arm or hand)."""

    control_mode: ControlMode
    values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        copied = tuple(float(value) for value in self.values)
        object.__setattr__(self, "values", copied)
        if self.control_mode in HOLD_MODES and copied:
            raise ValueError(f"{self.control_mode.value} carries no values, got {len(copied)}")


@dataclass(frozen=True)
class SkillCommand:
    """A command with an explicit, skill-owned representation per channel.

    The arm and hand are separate articulations with separate fabric
    controllers — one skill commands both channels independently (e.g.
    approach moves the arm while the hand opens; a hold freezes one channel
    while the other keeps acting).
    """

    arm: ChannelCommand
    hand: ChannelCommand
    source: str

    def __post_init__(self) -> None:
        if self.arm.control_mode not in ARM_MODES:
            raise ValueError(f"arm channel cannot use {self.arm.control_mode.value}")
        if self.hand.control_mode not in HAND_MODES:
            raise ValueError(f"hand channel cannot use {self.hand.control_mode.value}")

    @classmethod
    def no_op(cls, source: str) -> SkillCommand:
        return cls(ChannelCommand(ControlMode.NO_OP), ChannelCommand(ControlMode.NO_OP), source)

    @classmethod
    def safe_stop(cls, source: str) -> SkillCommand:
        return cls(
            ChannelCommand(ControlMode.SAFE_STOP), ChannelCommand(ControlMode.SAFE_STOP), source
        )


@dataclass(frozen=True)
class TransitionRecord:
    """Auditable result of a requested skill transition."""

    env_id: int
    previous_skill: SkillId
    requested_skill: SkillId
    accepted_skill: SkillId
    accepted: bool
    reason: str
    step_index: int
