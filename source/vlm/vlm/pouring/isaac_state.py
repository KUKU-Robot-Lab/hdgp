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
        grasp_lift_success_z: float | None = None,
        grasp_lift_hold_ticks: int = 1,
    ) -> None:
        if controlled_side not in ("right", "left"):
            raise ValueError(f"controlled_side must be 'right' or 'left', got {controlled_side!r}")
        if approach_tolerance <= 0.0:
            raise ValueError("approach_tolerance must be positive")
        if not all(math.isfinite(float(v)) for v in approach_offset):
            raise ValueError("approach_offset must be finite")
        if grasp_lift_success_z is not None and not math.isfinite(grasp_lift_success_z):
            raise ValueError("grasp_lift_success_z must be finite when given")
        if grasp_lift_hold_ticks < 1:
            raise ValueError("grasp_lift_hold_ticks must be >= 1")
        self.view = view
        self.routing = routing
        self.controlled_side = controlled_side
        self.approach_offset = approach_offset
        self.approach_tolerance = float(approach_tolerance)
        # ★absolute-z judgement: the caller must derive this from the *resting
        #   object origin* (e.g. baseline z after settling + lift height) — a
        #   surface-height guess poisons the baseline (repo-wide trap).
        self.grasp_lift_success_z = grasp_lift_success_z
        # ★순간 z 스파이크(쳐올림·발리스틱 토스)도 임계를 넘는다 — 실측: 정책이
        #   컵을 띄운 순간 DONE 이 발화하고 hold 동결 후 컵이 떨어졌다.
        #   성공은 연속 hold_ticks 틱 동안 유지됐을 때만 인정한다.
        self.grasp_lift_hold_ticks = int(grasp_lift_hold_ticks)
        self._lift_streaks: list[int] = []

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
        if len(self._lift_streaks) != num_envs:
            self._lift_streaks = [0] * num_envs
        return tuple(
            self._one(
                env_id=env_id,
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
        env_id: int,
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
        lifted = (
            self.grasp_lift_success_z is not None
            and source[2] >= self.grasp_lift_success_z
        )
        if skill is SkillId.GRASP_LIFT and lifted:
            self._lift_streaks[env_id] += 1
        else:
            self._lift_streaks[env_id] = 0
        if skill is SkillId.APPROACH:
            success = self._approach_reached(palm, source)
        elif skill is SkillId.GRASP_LIFT:
            success = self._lift_streaks[env_id] >= self.grasp_lift_hold_ticks
        else:
            success = False
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
            source_lifted=lifted,
            current_skill=skill,
            skill_elapsed_steps=max(0, int(elapsed)),
            current_skill_success=success,
            **side_kwargs,
        )

    def _approach_reached(self, palm: tuple[float, ...], source: tuple[float, ...]) -> bool:
        target = tuple(source[axis] + self.approach_offset[axis] for axis in range(3))
        distance = math.sqrt(sum((palm[axis] - target[axis]) ** 2 for axis in range(3)))
        return distance <= self.approach_tolerance
