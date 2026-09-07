from __future__ import annotations

from typing import Protocol

from .contracts import HighLevelDecision, SkillId, TaskSpecification
from .state_provider import SemanticState


class HighLevelPolicy(Protocol):
    """Replaceable deterministic or learned high-level policy."""

    def decide(
        self,
        task: TaskSpecification,
        states: tuple[SemanticState, ...],
    ) -> tuple[HighLevelDecision, ...]: ...


class DeterministicHighLevelPolicy:
    """V1 policy that advances only on explicit semantic success."""

    def decide(
        self,
        task: TaskSpecification,
        states: tuple[SemanticState, ...],
    ) -> tuple[HighLevelDecision, ...]:
        return tuple(self._decide_one(task, state) for state in states)

    @staticmethod
    def _decide_one(task: TaskSpecification, state: SemanticState) -> HighLevelDecision:
        if state.cup_drop or not state.workspace_valid or state.joint_limit_margin <= 0.0:
            return HighLevelDecision(
                SkillId.ABORT,
                terminate_current_skill=True,
                reason="safety_violation",
            )
        if state.current_skill_failed:
            recovery_allowed = SkillId.RECOVERY.value in task.allowed_skills
            next_skill = SkillId.RECOVERY if recovery_allowed else SkillId.ABORT
            return HighLevelDecision(
                next_skill,
                terminate_current_skill=True,
                recover=recovery_allowed,
                reason="skill_failed",
            )

        plan = tuple(SkillId(item) for item in task.nominal_plan)
        if state.current_skill is SkillId.WAIT_FOR_TASK:
            return HighLevelDecision(plan[0], reason="task_started")
        if not state.current_skill_success:
            return HighLevelDecision(state.current_skill, reason="continue_current_skill")
        if state.current_skill not in plan:
            return HighLevelDecision(
                SkillId.ABORT,
                terminate_current_skill=True,
                reason="skill_not_in_plan",
            )

        index = plan.index(state.current_skill)
        if index + 1 == len(plan):
            return HighLevelDecision(
                SkillId.DONE,
                terminate_current_skill=True,
                reason="plan_complete",
            )
        return HighLevelDecision(
            plan[index + 1],
            terminate_current_skill=True,
            reason="skill_complete",
        )
