from __future__ import annotations

from dataclasses import dataclass

from .contracts import HighLevelDecision, SkillCommand, TaskSpecification, TransitionRecord
from .high_level_policy import HighLevelPolicy
from .skill_manager import SkillManager
from .state_provider import SemanticState, StateProvider


@dataclass(frozen=True)
class TickResult:
    """Auditable result of one low-frequency hierarchical tick."""

    states: tuple[SemanticState, ...]
    decisions: tuple[HighLevelDecision, ...]
    commands: tuple[SkillCommand, ...]
    transitions: tuple[TransitionRecord, ...]


class PouringPipeline:
    """Closed-loop composition without owning low-level control timing."""

    def __init__(
        self,
        *,
        task: TaskSpecification,
        state_provider: StateProvider,
        high_level_policy: HighLevelPolicy,
        skill_manager: SkillManager,
    ) -> None:
        self.task = task
        self.state_provider = state_provider
        self.high_level_policy = high_level_policy
        self.skill_manager = skill_manager

    def tick(self) -> TickResult:
        states = self.state_provider.get_states()
        decisions = self.high_level_policy.decide(self.task, states)
        commands, transitions = self.skill_manager.step(self.task, states, decisions)
        return TickResult(states, decisions, commands, transitions)
