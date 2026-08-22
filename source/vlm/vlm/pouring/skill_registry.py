from __future__ import annotations

from collections.abc import Iterable

from .contracts import SkillId
from .execution import Skill


class SkillRegistry:
    """Explicit mapping from stable skill IDs to runtime adapters."""

    def __init__(self, skills: Iterable[Skill]) -> None:
        self._skills: dict[SkillId, Skill] = {}
        for skill in skills:
            if skill.skill_id in self._skills:
                raise ValueError(f"duplicate skill registration: {skill.skill_id.value}")
            self._skills[skill.skill_id] = skill

    def get(self, skill_id: SkillId) -> Skill:
        try:
            return self._skills[skill_id]
        except KeyError as exc:
            raise KeyError(f"skill is not registered: {skill_id.value}") from exc

    def contains(self, skill_id: SkillId) -> bool:
        return skill_id in self._skills
