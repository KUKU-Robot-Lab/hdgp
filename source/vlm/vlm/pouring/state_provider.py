from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from .contracts import SkillId


def _validate_vector(name: str, values: tuple[float, ...], size: int) -> None:
    if len(values) != size:
        raise ValueError(f"{name} must have length {size}, got {len(values)}")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{name} must contain only finite values")


def _validate_unit_interval(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")


@dataclass(frozen=True)
class SemanticState:
    """Simulator-neutral state. Poses use xyz + quaternion wxyz."""

    source_pose: tuple[float, ...]
    target_pose: tuple[float, ...]
    source_velocity: tuple[float, ...]
    target_velocity: tuple[float, ...]
    left_arm_joint_pos: tuple[float, ...]
    left_arm_joint_vel: tuple[float, ...]
    right_arm_joint_pos: tuple[float, ...]
    right_arm_joint_vel: tuple[float, ...]
    left_hand_joint_pos: tuple[float, ...]
    left_hand_joint_vel: tuple[float, ...]
    right_hand_joint_pos: tuple[float, ...]
    right_hand_joint_vel: tuple[float, ...]
    left_ee_pose: tuple[float, ...]
    right_ee_pose: tuple[float, ...]
    source_confidence: float = 1.0
    target_confidence: float = 1.0
    contact_count: int = 0
    tactile_summary: tuple[float, ...] = ()
    source_grasped: bool = False
    source_lifted: bool = False
    source_upright_score: float = 1.0
    cup_drop: bool = False
    pregrasp_ready: bool = False
    warm_state_valid: bool = False
    pour_complete: bool = False
    workspace_valid: bool = True
    joint_limit_margin: float = 1.0
    current_skill: SkillId = SkillId.WAIT_FOR_TASK
    skill_elapsed_steps: int = 0
    current_skill_success: bool = False
    current_skill_failed: bool = False

    def __post_init__(self) -> None:
        for name in ("source_pose", "target_pose", "left_ee_pose", "right_ee_pose"):
            _validate_vector(name, getattr(self, name), 7)
        for name in ("source_velocity", "target_velocity"):
            _validate_vector(name, getattr(self, name), 6)
        for name in (
            "left_arm_joint_pos",
            "left_arm_joint_vel",
            "right_arm_joint_pos",
            "right_arm_joint_vel",
        ):
            _validate_vector(name, getattr(self, name), 7)
        for name in (
            "left_hand_joint_pos",
            "left_hand_joint_vel",
            "right_hand_joint_pos",
            "right_hand_joint_vel",
        ):
            _validate_vector(name, getattr(self, name), 20)
        if not all(math.isfinite(float(value)) for value in self.tactile_summary):
            raise ValueError("tactile_summary must contain only finite values")
        _validate_unit_interval("source_confidence", self.source_confidence)
        _validate_unit_interval("target_confidence", self.target_confidence)
        _validate_unit_interval("source_upright_score", self.source_upright_score)
        if not math.isfinite(self.joint_limit_margin):
            raise ValueError("joint_limit_margin must be finite")
        if self.contact_count < 0:
            raise ValueError("contact_count cannot be negative")
        if self.skill_elapsed_steps < 0:
            raise ValueError("skill_elapsed_steps cannot be negative")


class StateProvider(Protocol):
    """Source-agnostic batch state provider."""

    def get_states(self) -> tuple[SemanticState, ...]: ...
