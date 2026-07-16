from __future__ import annotations

import math

from ..contracts import ControlMode, SkillCommand, SkillId
from ..state_provider import SemanticState


class ApproachSkill:
    """Rule-based source-relative pregrasp target generator."""

    skill_id = SkillId.APPROACH

    def __init__(
        self,
        *,
        offset: tuple[float, float, float],
        orientation_wxyz: tuple[float, float, float, float],
    ) -> None:
        if not all(math.isfinite(value) for value in (*offset, *orientation_wxyz)):
            raise ValueError("approach target configuration must be finite")
        self.offset = offset
        self.orientation_wxyz = orientation_wxyz

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
                    ControlMode.TASK_SPACE_POSE,
                    position + self.orientation_wxyz,
                    self.skill_id.value,
                )
            )
        return tuple(commands)
