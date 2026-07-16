from __future__ import annotations

from collections.abc import Callable
import math
from typing import Protocol

from .checkpoint_resolver import PolicyArtifacts
from .contracts import ControlMode, SkillCommand, SkillId
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


ObservationBuilder = Callable[
    [tuple[SemanticState, ...]],
    tuple[tuple[float, ...], ...],
]


class ReferencedPolicySkill:
    """Dimension-safe adapter around an injected existing-policy backend."""

    skill_id: SkillId
    observation_dim: int
    action_dim: int

    def __init__(
        self,
        artifacts: PolicyArtifacts,
        *,
        observation_builder: ObservationBuilder,
        backend: PolicyInferenceBackend,
    ) -> None:
        self.artifacts = artifacts
        self.observation_builder = observation_builder
        self.backend = backend

    def reset(self, env_ids: tuple[int, ...]) -> None:
        self.backend.reset(env_ids)

    def infer(
        self,
        env_ids: tuple[int, ...],
        states: tuple[SemanticState, ...],
    ) -> tuple[SkillCommand, ...]:
        observations = self.observation_builder(states)
        if len(observations) != len(states):
            raise ValueError("observation builder must return one observation per environment")
        for observation in observations:
            if len(observation) != self.observation_dim:
                raise ValueError(
                    f"{self.skill_id.value} observation must be {self.observation_dim}D, got {len(observation)}"
                )
            if not all(math.isfinite(float(value)) for value in observation):
                raise ValueError(f"{self.skill_id.value} observation must be finite")
        actions = self.backend.infer(observations)
        if len(actions) != len(states):
            raise ValueError("policy backend must return one action per environment")
        for action in actions:
            if len(action) != self.action_dim:
                raise ValueError(f"{self.skill_id.value} action must be {self.action_dim}D, got {len(action)}")
        return tuple(
            SkillCommand(ControlMode.POLICY_ACTION, tuple(float(value) for value in action), self.skill_id.value)
            for action in actions
        )
