from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

from .checkpoint_resolver import PolicyArtifacts, read_policy_contract
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
    """Injected boundary for an RL-Games or test inference implementation.

    `env_ids` names which environment each observation row belongs to —
    a recurrent policy keeps one hidden state per environment, and the hard
    router only sends the subset currently assigned to this skill.
    """

    def infer(
        self,
        env_ids: tuple[int, ...],
        observations: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[float, ...], ...]: ...

    def reset(self, env_ids: tuple[int, ...]) -> None: ...


# env_ids come first so a builder that reads live env buffers (the common
# Isaac case — the policy obs is the env's own observation vector) can pick
# exactly the routed rows; builders that derive obs from SemanticState alone
# may ignore them.
ObservationBuilder = Callable[
    [tuple[int, ...], tuple[SemanticState, ...]],
    tuple[tuple[float, ...], ...],
]


class ReferencedPolicySkill:
    """Dimension-safe adapter around an injected existing-policy backend.

    Dimensions are never hardcoded per skill: they come from the run's own
    `params/env.yaml` (what the checkpoint was actually trained with), or an
    explicit override for tests and special cases. Retrained tracks ship
    different contracts, so a class-level constant would silently rot.
    """

    skill_id: SkillId

    def __init__(
        self,
        artifacts: PolicyArtifacts,
        *,
        observation_builder: ObservationBuilder,
        backend: PolicyInferenceBackend,
        observation_dim: int | None = None,
        action_dim: int | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.observation_builder = observation_builder
        self.backend = backend
        if observation_dim is None or action_dim is None:
            contract = read_policy_contract(artifacts.env_yaml)
            observation_dim = observation_dim if observation_dim is not None else contract.observation_dim
            action_dim = action_dim if action_dim is not None else contract.action_dim
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError(
                f"{self.skill_id.value} dims must be positive, got "
                f"obs={observation_dim} act={action_dim}"
            )
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

    def reset(self, env_ids: tuple[int, ...]) -> None:
        self.backend.reset(env_ids)

    def infer(
        self,
        env_ids: tuple[int, ...],
        states: tuple[SemanticState, ...],
    ) -> tuple[SkillCommand, ...]:
        observations = self.observation_builder(env_ids, states)
        if len(observations) != len(states):
            raise ValueError("observation builder must return one observation per environment")
        for observation in observations:
            if len(observation) != self.observation_dim:
                raise ValueError(
                    f"{self.skill_id.value} observation must be {self.observation_dim}D, got {len(observation)}"
                )
            if not all(math.isfinite(float(value)) for value in observation):
                raise ValueError(f"{self.skill_id.value} observation must be finite")
        actions = self.backend.infer(env_ids, observations)
        if len(actions) != len(states):
            raise ValueError("policy backend must return one action per environment")
        for action in actions:
            if len(action) != self.action_dim:
                raise ValueError(f"{self.skill_id.value} action must be {self.action_dim}D, got {len(action)}")
        return tuple(
            SkillCommand(ControlMode.POLICY_ACTION, tuple(float(value) for value in action), self.skill_id.value)
            for action in actions
        )
