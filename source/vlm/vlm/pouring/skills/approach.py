from __future__ import annotations

import math

from ..contracts import HAND_MODES, ChannelCommand, ControlMode, SkillCommand, SkillId
from ..state_provider import SemanticState


class ApproachSkill:
    """Rule-based source-relative pregrasp target generator.

    Arm and hand are separate channels: the arm gets a task-space pose, and
    the hand gets an explicit normalized target (e.g. fully open) in the
    convention of the running fabric task — joint targets or tip targets.
    The default hand channel is NO_OP (leave the hand alone).
    """

    skill_id = SkillId.APPROACH

    def __init__(
        self,
        *,
        offset: tuple[float, float, float],
        orientation_wxyz: tuple[float, float, float, float],
        hand_mode: ControlMode = ControlMode.NO_OP,
        hand_targets: tuple[float, ...] = (),
    ) -> None:
        if not all(math.isfinite(value) for value in (*offset, *orientation_wxyz)):
            raise ValueError("approach target configuration must be finite")
        if hand_mode not in HAND_MODES:
            raise ValueError(f"hand channel cannot use {hand_mode.value}")
        self.offset = offset
        self.orientation_wxyz = orientation_wxyz
        self._hand = ChannelCommand(hand_mode, hand_targets)

    def reset(self, env_ids: tuple[int, ...]) -> None:
        del env_ids

    def infer(
        self,
        env_ids: tuple[int, ...],
        states: tuple[SemanticState, ...],
    ) -> tuple[SkillCommand, ...]:
        del env_ids
        commands = []
        for state in states:
            position = tuple(state.source_pose[index] + self.offset[index] for index in range(3))
            commands.append(
                SkillCommand(
                    ChannelCommand(
                        ControlMode.TASK_SPACE_POSE, position + self.orientation_wxyz
                    ),
                    self._hand,
                    self.skill_id.value,
                )
            )
        return tuple(commands)
