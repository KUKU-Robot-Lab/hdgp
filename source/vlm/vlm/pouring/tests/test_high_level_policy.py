from __future__ import annotations

from dataclasses import replace

from vlm.pouring.contracts import SkillId, TaskSpecification
from vlm.pouring.high_level_policy import DeterministicHighLevelPolicy
from vlm.pouring.tests.test_state_provider import valid_state


TASK = TaskSpecification(
    task="pour",
    source_id="source",
    target_id="target",
    nominal_plan=("grasp_lift", "pre_pour_bridge", "bimanual_pour"),
    allowed_skills=("grasp_lift", "pre_pour_bridge", "bimanual_pour", "recovery"),
)


def test_policy_enters_first_nominal_skill() -> None:
    decision = DeterministicHighLevelPolicy().decide(TASK, (valid_state(),))[0]

    assert decision.skill_id is SkillId.GRASP_LIFT
    assert decision.reason == "task_started"


def test_policy_keeps_current_skill_until_success() -> None:
    state = valid_state(current_skill=SkillId.GRASP_LIFT, skill_elapsed_steps=3)

    decision = DeterministicHighLevelPolicy().decide(TASK, (state,))[0]

    assert decision.skill_id is SkillId.GRASP_LIFT
    assert not decision.terminate_current_skill


def test_policy_advances_and_finishes_nominal_plan() -> None:
    policy = DeterministicHighLevelPolicy()
    grasp_done = valid_state(current_skill=SkillId.GRASP_LIFT, current_skill_success=True)
    pour_done = valid_state(current_skill=SkillId.BIMANUAL_POUR, current_skill_success=True)

    assert policy.decide(TASK, (grasp_done,))[0].skill_id is SkillId.PRE_POUR_BRIDGE
    assert policy.decide(TASK, (pour_done,))[0].skill_id is SkillId.DONE


def test_policy_recovers_on_skill_failure_and_aborts_on_safety_failure() -> None:
    policy = DeterministicHighLevelPolicy()
    failed = valid_state(current_skill=SkillId.GRASP_LIFT, current_skill_failed=True)
    dropped = replace(failed, cup_drop=True)

    recovery = policy.decide(TASK, (failed,))[0]
    abort = policy.decide(TASK, (dropped,))[0]

    assert recovery.skill_id is SkillId.RECOVERY
    assert recovery.recover
    assert abort.skill_id is SkillId.ABORT


def test_policy_emits_one_decision_per_environment() -> None:
    states = (valid_state(), valid_state(current_skill=SkillId.GRASP_LIFT))

    decisions = DeterministicHighLevelPolicy().decide(TASK, states)

    assert len(decisions) == 2


def test_policy_aborts_when_current_skill_is_not_in_plan() -> None:
    state = valid_state(current_skill=SkillId.APPROACH, current_skill_success=True)

    decision = DeterministicHighLevelPolicy().decide(TASK, (state,))[0]

    assert decision.skill_id is SkillId.ABORT
    assert decision.reason == "skill_not_in_plan"
