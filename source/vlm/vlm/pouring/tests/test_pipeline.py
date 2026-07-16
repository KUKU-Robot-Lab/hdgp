from __future__ import annotations

from dataclasses import dataclass

from vlm.pouring.contracts import ControlMode, SkillId
from vlm.pouring.high_level_policy import DeterministicHighLevelPolicy
from vlm.pouring.pipeline import PouringPipeline
from vlm.pouring.skill_manager import SkillManager
from vlm.pouring.skill_registry import SkillRegistry
from vlm.pouring.state_provider import SemanticState
from vlm.pouring.tests.test_skill_manager import FakeSkill, make_task
from vlm.pouring.tests.test_state_provider import valid_state


@dataclass
class FakeStateProvider:
    states: tuple[SemanticState, ...]

    def get_states(self) -> tuple[SemanticState, ...]:
        return self.states


def test_pipeline_routes_grounded_task_without_loading_qwen_or_isaac() -> None:
    grasp = FakeSkill(SkillId.GRASP_LIFT, 1.0, 11)
    manager = SkillManager(registry=SkillRegistry((grasp,)), num_envs=1)
    pipeline = PouringPipeline(
        task=make_task(),
        state_provider=FakeStateProvider((valid_state(),)),
        high_level_policy=DeterministicHighLevelPolicy(),
        skill_manager=manager,
    )

    result = pipeline.tick()

    assert result.decisions[0].skill_id is SkillId.GRASP_LIFT
    assert result.commands[0].source == "grasp_lift"
    assert result.transitions[0].accepted


def test_pipeline_routes_two_environments_to_different_skills() -> None:
    grasp = FakeSkill(SkillId.GRASP_LIFT, 1.0, 11)
    pour = FakeSkill(SkillId.BIMANUAL_POUR, 2.0, 12)
    manager = SkillManager(
        registry=SkillRegistry((grasp, pour)),
        num_envs=2,
        initial_skills=(SkillId.WAIT_FOR_TASK, SkillId.BIMANUAL_POUR),
    )
    states = (
        valid_state(),
        valid_state(current_skill=SkillId.BIMANUAL_POUR),
    )
    pipeline = PouringPipeline(
        task=make_task(),
        state_provider=FakeStateProvider(states),
        high_level_policy=DeterministicHighLevelPolicy(),
        skill_manager=manager,
    )

    result = pipeline.tick()

    assert tuple(command.source for command in result.commands) == ("grasp_lift", "bimanual_pour")
    assert tuple(len(command.values) for command in result.commands) == (11, 12)


def test_pipeline_turns_safety_violation_into_abort_safe_stop() -> None:
    grasp = FakeSkill(SkillId.GRASP_LIFT, 1.0, 11)
    pipeline = PouringPipeline(
        task=make_task(),
        state_provider=FakeStateProvider(
            (valid_state(current_skill=SkillId.GRASP_LIFT, cup_drop=True),)
        ),
        high_level_policy=DeterministicHighLevelPolicy(),
        skill_manager=SkillManager(
            registry=SkillRegistry((grasp,)),
            num_envs=1,
            initial_skills=(SkillId.GRASP_LIFT,),
        ),
    )

    result = pipeline.tick()

    assert result.decisions[0].skill_id is SkillId.ABORT
    assert result.commands[0].control_mode is ControlMode.SAFE_STOP
