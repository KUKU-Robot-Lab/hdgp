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
    """Command representations that must never be blended implicitly."""

    TASK_SPACE_POSE = "task_space_pose"
    POLICY_ACTION = "policy_action"
    SAFE_STOP = "safe_stop"
    NO_OP = "no_op"


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
class SkillCommand:
    """A command with an explicit, skill-owned representation."""

    control_mode: ControlMode
    values: tuple[float, ...]
    source: str


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
