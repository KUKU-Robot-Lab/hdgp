from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import SkillCommand, SkillId
from ..state_provider import SemanticState


class PrePourBridgeSkill:
    """Gate pour entry on the existing warm-state loader and semantic readiness."""

    skill_id = SkillId.PRE_POUR_BRIDGE

    def __init__(
        self,
        bridge: Any,
        warm_state_path: Path,
        *,
        expected_object_spawn_z: float,
    ) -> None:
        self.bridge = bridge
        self.warm_state_path = warm_state_path
        self.expected_object_spawn_z = expected_object_spawn_z
        self.bank: Any | None = None

    def validate(self) -> Any:
        self.bank = self.bridge.load(
            self.warm_state_path,
            expected_object_spawn_z=self.expected_object_spawn_z,
        )
        return self.bank

    def reset(self, env_ids: tuple[int, ...]) -> None:
        del env_ids

    def infer(
        self,
        env_ids: tuple[int, ...],
        states: tuple[SemanticState, ...],
    ) -> tuple[SkillCommand, ...]:
        del env_ids
        if self.bank is None:
            raise RuntimeError("pre-pour warm-state bridge must validate before inference")
        return tuple(
            (SkillCommand.no_op if state.warm_state_valid else SkillCommand.safe_stop)(
                self.skill_id.value
            )
            for state in states
        )
