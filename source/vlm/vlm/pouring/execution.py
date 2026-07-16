from __future__ import annotations

from typing import Protocol

from .contracts import SkillCommand, SkillId
from .state_provider import SemanticState


class Skill(Protocol):
    """A batched low-level skill selected by the hard router."""

    skill_id: SkillId

    def reset(self, env_ids: tuple[int, ...]) -> None: ...

    def infer(
        self,
        env_ids: tuple[int, ...],
        states: tuple[SemanticState, ...],
    ) -> tuple[SkillCommand, ...]: ...


class PolicyInferenceBackend(Protocol):
    """Injected boundary for an RL-Games or test inference implementation."""

    def infer(
        self,
        observations: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[float, ...], ...]: ...

    def reset(self, env_ids: tuple[int, ...]) -> None: ...
