from __future__ import annotations

from ..contracts import ControlMode, SkillCommand, SkillId
from ..state_provider import SemanticState


class PreGraspBridgeSkill:
    """Bounded task-space alignment before entering grasp_v1."""

    skill_id = SkillId.PRE_GRASP_BRIDGE

    def __init__(self, *, max_position_step: float) -> None:
        if max_position_step <= 0.0:
            raise ValueError("max_position_step must be positive")
        self.max_position_step = float(max_position_step)

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
            if state.pregrasp_ready:
                commands.append(SkillCommand(ControlMode.NO_OP, (), self.skill_id.value))
                continue
            current = state.right_ee_pose[:3]
            desired = state.source_pose[:3]
            bounded = tuple(
                current[index]
                + max(-self.max_position_step, min(self.max_position_step, desired[index] - current[index]))
                for index in range(3)
            )
            commands.append(
                SkillCommand(
                    ControlMode.TASK_SPACE_POSE,
                    bounded + state.right_ee_pose[3:7],
                    self.skill_id.value,
                )
            )
        return tuple(commands)
