from __future__ import annotations

from ..contracts import SkillCommand, SkillId
from ..state_provider import SemanticState


class RecoverySkill:
    """V1 recovery is an explicit safe stop, not an untrained motion."""

    skill_id = SkillId.RECOVERY

    def reset(self, env_ids: tuple[int, ...]) -> None:
        del env_ids

    def infer(
        self,
        env_ids: tuple[int, ...],
        states: tuple[SemanticState, ...],
    ) -> tuple[SkillCommand, ...]:
        del env_ids
        return tuple(SkillCommand.safe_stop(self.skill_id.value) for _ in states)
