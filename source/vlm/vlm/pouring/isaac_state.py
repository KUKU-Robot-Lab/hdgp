"""Semantic-state assembly for the fabric grasp/pour scenes.

`FabricSceneView` is the narrow, batched read interface the Isaac side must
implement (a thin adapter over the fabric env's buffers). `FabricStateProvider`
turns one view snapshot plus the skill manager's routing state into validated
`SemanticState` tuples, so the CPU test suite can drive it with a fake view.

Skill success here is deliberately minimal for the v1 demo: only APPROACH is
scored (palm within `approach_tolerance` of the source-relative target). The
learned skills report success through their own env buffers once the RL
inference backends land (roadmap step 5).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from .contracts import SkillId
from .fabric_bridge import euler_zyx_to_quat_wxyz
from .state_provider import SemanticState, StateProvider

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
_ZEROS7 = (0.0,) * 7
_ZEROS20 = (0.0,) * 20


class FabricSceneView(Protocol):
    """Batched, env-local reads from one fabric task scene."""

    @property
    def num_envs(self) -> int: ...

    def palm_pose_zyx(self) -> Sequence[Sequence[float]]:
        """(N, 6) palm pose: env-local xyz + euler_zyx."""
        ...

    def object_pose(self) -> Sequence[Sequence[float]]:
        """(N, 7) object pose: env-local xyz + quat wxyz."""
        ...

    def object_velocity(self) -> Sequence[Sequence[float]]:
        """(N, 6) object linear + angular velocity."""
        ...

    def arm_joint_pos(self) -> Sequence[Sequence[float]]:
        """(N, 7) controlled-arm joint positions."""
        ...

    def arm_joint_vel(self) -> Sequence[Sequence[float]]:
        """(N, 7) controlled-arm joint velocities."""
        ...

    def hand_joint_pos(self) -> Sequence[Sequence[float]]:
        """(N, 20) hand joint positions."""
        ...

    def hand_joint_vel(self) -> Sequence[Sequence[float]]:
        """(N, 20) hand joint velocities."""
        ...


class SkillRoutingView(Protocol):
    """What the provider needs from the skill manager, and nothing more."""

    current_skills: list[SkillId]
    elapsed_steps: list[int]


class FabricStateProvider(StateProvider):
    """Assemble one `SemanticState` per environment from a fabric scene."""

    def __init__(
        self,
        view: FabricSceneView,
        routing: SkillRoutingView,
        *,
        controlled_side: str = "right",
        approach_offset: tuple[float, float, float] = (0.0, 0.0, 0.10),
        approach_tolerance: float = 0.03,
    ) -> None:
        if controlled_side not in ("right", "left"):
            raise ValueError(f"controlled_side must be 'right' or 'left', got {controlled_side!r}")
        if approach_tolerance <= 0.0:
            raise ValueError("approach_tolerance must be positive")
        if not all(math.isfinite(float(v)) for v in approach_offset):
            raise ValueError("approach_offset must be finite")
        self.view = view
        self.routing = routing
        self.controlled_side = controlled_side
        self.approach_offset = approach_offset
        self.approach_tolerance = float(approach_tolerance)

    def get_states(self) -> tuple[SemanticState, ...]:
        num_envs = self.view.num_envs
        palm = self.view.palm_pose_zyx()
        source = self.view.object_pose()
        source_vel = self.view.object_velocity()
        arm_pos = self.view.arm_joint_pos()
        arm_vel = self.view.arm_joint_vel()
        hand_pos = self.view.hand_joint_pos()
        hand_vel = self.view.hand_joint_vel()
        lengths = {len(palm), len(source), len(source_vel), len(arm_pos),
                   len(arm_vel), len(hand_pos), len(hand_vel)}
        if lengths != {num_envs}:
            raise ValueError(f"view batch sizes {sorted(lengths)} != num_envs {num_envs}")
        if len(self.routing.current_skills) != num_envs:
            raise ValueError(
                f"routing tracks {len(self.routing.current_skills)} envs, view has {num_envs}"
            )
        return tuple(
            self._one(
                palm=tuple(float(v) for v in palm[env_id]),
                source=tuple(float(v) for v in source[env_id]),
                source_vel=tuple(float(v) for v in source_vel[env_id]),
                arm_pos=tuple(float(v) for v in arm_pos[env_id]),
                arm_vel=tuple(float(v) for v in arm_vel[env_id]),
                hand_pos=tuple(float(v) for v in hand_pos[env_id]),
                hand_vel=tuple(float(v) for v in hand_vel[env_id]),
                skill=self.routing.current_skills[env_id],
                elapsed=self.routing.elapsed_steps[env_id],
            )
            for env_id in range(num_envs)
        )

    def _one(
        self,
        *,
        palm: tuple[float, ...],
        source: tuple[float, ...],
        source_vel: tuple[float, ...],
        arm_pos: tuple[float, ...],
        arm_vel: tuple[float, ...],
        hand_pos: tuple[float, ...],
        hand_vel: tuple[float, ...],
        skill: SkillId,
        elapsed: int,
    ) -> SemanticState:
        ee_pose = (*palm[:3], *euler_zyx_to_quat_wxyz((palm[3], palm[4], palm[5])))
        success = skill is SkillId.APPROACH and self._approach_reached(palm, source)
        side_kwargs = {
            f"{self.controlled_side}_arm_joint_pos": arm_pos,
            f"{self.controlled_side}_arm_joint_vel": arm_vel,
            f"{self.controlled_side}_hand_joint_pos": hand_pos,
            f"{self.controlled_side}_hand_joint_vel": hand_vel,
            f"{self.controlled_side}_ee_pose": ee_pose,
        }
        other = "left" if self.controlled_side == "right" else "right"
        side_kwargs.update({
            f"{other}_arm_joint_pos": _ZEROS7,
            f"{other}_arm_joint_vel": _ZEROS7,
            f"{other}_hand_joint_pos": _ZEROS20,
            f"{other}_hand_joint_vel": _ZEROS20,
            f"{other}_ee_pose": (0.0, 0.0, 0.0, *_IDENTITY_QUAT),
        })
        return SemanticState(
            source_pose=source,
            target_pose=(0.0, 0.0, 0.0, *_IDENTITY_QUAT),
            source_velocity=source_vel,
            target_velocity=(0.0,) * 6,
            current_skill=skill,
            skill_elapsed_steps=max(0, int(elapsed)),
            current_skill_success=success,
            **side_kwargs,
        )

    def _approach_reached(self, palm: tuple[float, ...], source: tuple[float, ...]) -> bool:
        target = tuple(source[axis] + self.approach_offset[axis] for axis in range(3))
        distance = math.sqrt(sum((palm[axis] - target[axis]) ** 2 for axis in range(3)))
        return distance <= self.approach_tolerance
