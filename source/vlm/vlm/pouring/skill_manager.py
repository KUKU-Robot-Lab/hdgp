from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from .contracts import (
    HighLevelDecision,
    SkillCommand,
    SkillId,
    TaskSpecification,
    TransitionRecord,
)
from .safety import SafetySupervisor
from .skill_registry import SkillRegistry
from .state_provider import SemanticState

_TERMINAL_SKILLS = {SkillId.ABORT, SkillId.DONE}


class SkillManager:
    """Guard and execute one hard-routed skill per environment."""

    def __init__(
        self,
        *,
        registry: SkillRegistry,
        num_envs: int,
        initial_skills: tuple[SkillId, ...] | None = None,
        minimum_steps: Mapping[SkillId, int] | None = None,
        safety: SafetySupervisor | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if initial_skills is not None and len(initial_skills) != num_envs:
            raise ValueError("initial_skills length must equal num_envs")
        self.registry = registry
        self.num_envs = num_envs
        self.current_skills = list(initial_skills or (SkillId.WAIT_FOR_TASK,) * num_envs)
        self.elapsed_steps = [0] * num_envs
        self.minimum_steps = {skill: max(0, int(steps)) for skill, steps in (minimum_steps or {}).items()}
        self.safety = safety or SafetySupervisor()
        self.step_index = 0

    def step(
        self,
        task: TaskSpecification,
        states: tuple[SemanticState, ...],
        decisions: tuple[HighLevelDecision, ...],
    ) -> tuple[tuple[SkillCommand, ...], tuple[TransitionRecord, ...]]:
        if len(states) != self.num_envs or len(decisions) != self.num_envs:
            raise ValueError("states and decisions must contain one item per environment")

        accepted_skills: list[SkillId] = []
        records: list[TransitionRecord] = []
        switched: dict[SkillId, list[int]] = defaultdict(list)

        for env_id, (state, decision) in enumerate(zip(states, decisions, strict=True)):
            previous = self.current_skills[env_id]
            accepted, reason = self._guard(
                task,
                previous,
                max(self.elapsed_steps[env_id], state.skill_elapsed_steps),
                decision,
            )
            selected = decision.skill_id if accepted else previous
            if accepted and selected != previous:
                switched[selected].append(env_id)
                self.elapsed_steps[env_id] = 0
            else:
                self.elapsed_steps[env_id] += 1
            self.current_skills[env_id] = selected
            accepted_skills.append(selected)
            records.append(
                TransitionRecord(
                    env_id=env_id,
                    previous_skill=previous,
                    requested_skill=decision.skill_id,
                    accepted_skill=selected,
                    accepted=accepted,
                    reason=reason,
                    step_index=self.step_index,
                )
            )

        for skill_id, env_ids in switched.items():
            if self.registry.contains(skill_id):
                self.registry.get(skill_id).reset(tuple(env_ids))

        commands: list[SkillCommand | None] = [None] * self.num_envs
        grouped: dict[SkillId, list[int]] = defaultdict(list)
        for env_id, skill_id in enumerate(accepted_skills):
            grouped[skill_id].append(env_id)

        for skill_id, env_ids in grouped.items():
            if skill_id is SkillId.DONE or skill_id is SkillId.WAIT_FOR_TASK:
                for env_id in env_ids:
                    commands[env_id] = SkillCommand.no_op(skill_id.value)
                continue
            if skill_id is SkillId.ABORT:
                for env_id in env_ids:
                    commands[env_id] = SkillCommand.safe_stop(skill_id.value)
                continue
            skill = self.registry.get(skill_id)
            env_tuple = tuple(env_ids)
            state_batch = tuple(states[env_id] for env_id in env_ids)
            outputs = skill.infer(env_tuple, state_batch)
            if len(outputs) != len(env_ids):
                raise ValueError(f"{skill_id.value} returned {len(outputs)} commands for {len(env_ids)} environments")
            for env_id, output in zip(env_ids, outputs, strict=True):
                commands[env_id] = self.safety.validate(output)

        self.step_index += 1
        if any(command is None for command in commands):
            raise RuntimeError("skill routing did not produce a command for every environment")
        return tuple(command for command in commands if command is not None), tuple(records)

    def _guard(
        self,
        task: TaskSpecification,
        previous: SkillId,
        elapsed_steps: int,
        decision: HighLevelDecision,
    ) -> tuple[bool, str]:
        requested = decision.skill_id
        if requested not in _TERMINAL_SKILLS and requested.value not in task.allowed_skills:
            return False, "skill_not_allowed"
        if requested == previous:
            return True, "continue_current_skill"
        required = self.minimum_steps.get(previous, 0)
        if previous is not SkillId.WAIT_FOR_TASK and elapsed_steps < required:
            return False, "minimum_duration"
        if (
            previous is not SkillId.WAIT_FOR_TASK
            and requested not in _TERMINAL_SKILLS
            and not decision.terminate_current_skill
            and not decision.retry
            and not decision.recover
        ):
            return False, "termination_required"
        return True, decision.reason or "transition_accepted"
