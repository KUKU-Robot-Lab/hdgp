from __future__ import annotations

from dataclasses import dataclass, field

from vlm.pouring.contracts import (
    ControlMode,
    HighLevelDecision,
    SkillCommand,
    SkillId,
    TaskSpecification,
)
from vlm.pouring.safety import SafetySupervisor
from vlm.pouring.skill_manager import SkillManager
from vlm.pouring.skill_registry import SkillRegistry
from vlm.pouring.state_provider import SemanticState
from vlm.pouring.tests.test_state_provider import valid_state


@dataclass
class FakeSkill:
    skill_id: SkillId
    value: float
    action_dim: int
    calls: list[tuple[int, ...]] = field(default_factory=list)
    resets: list[tuple[int, ...]] = field(default_factory=list)

    def reset(self, env_ids: tuple[int, ...]) -> None:
        self.resets.append(env_ids)

    def infer(
        self,
        env_ids: tuple[int, ...],
        states: tuple[SemanticState, ...],
    ) -> tuple[SkillCommand, ...]:
        self.calls.append(env_ids)
        return tuple(
            SkillCommand(ControlMode.POLICY_ACTION, (self.value,) * self.action_dim, self.skill_id.value)
            for _ in states
        )


def make_task(allowed: tuple[str, ...] | None = None) -> TaskSpecification:
    default = ("grasp_lift", "bimanual_pour", "recovery")
    return TaskSpecification(
        "pour",
        "source",
        "target",
        ("grasp_lift", "bimanual_pour"),
        allowed or default,
    )


def make_manager(
    *,
    num_envs: int = 2,
    initial_skills: tuple[SkillId, ...] | None = None,
    minimum_steps: dict[SkillId, int] | None = None,
) -> tuple[SkillManager, FakeSkill, FakeSkill]:
    grasp = FakeSkill(SkillId.GRASP_LIFT, 1.0, 11)
    pour = FakeSkill(SkillId.BIMANUAL_POUR, 2.0, 12)
    manager = SkillManager(
        registry=SkillRegistry((grasp, pour)),
        num_envs=num_envs,
        initial_skills=initial_skills,
        minimum_steps=minimum_steps,
    )
    return manager, grasp, pour


def test_manager_routes_each_environment_without_action_blending() -> None:
    manager, grasp, pour = make_manager()
    states = (valid_state(), valid_state())

    commands, records = manager.step(
        make_task(),
        states,
        (
            HighLevelDecision(SkillId.GRASP_LIFT, reason="grasp"),
            HighLevelDecision(SkillId.BIMANUAL_POUR, reason="pour"),
        ),
    )

    assert commands[0].values == (1.0,) * 11
    assert commands[1].values == (2.0,) * 12
    assert grasp.calls == [(0,)]
    assert pour.calls == [(1,)]
    assert all(record.accepted for record in records)


def test_manager_rejects_disallowed_transition() -> None:
    manager, grasp, pour = make_manager(
        num_envs=1,
        initial_skills=(SkillId.GRASP_LIFT,),
    )
    task = make_task(allowed=("grasp_lift", "recovery"))

    commands, records = manager.step(
        task,
        (valid_state(current_skill=SkillId.GRASP_LIFT),),
        (HighLevelDecision(SkillId.BIMANUAL_POUR, terminate_current_skill=True),),
    )

    assert not records[0].accepted
    assert records[0].accepted_skill is SkillId.GRASP_LIFT
    assert commands[0].values == (1.0,) * 11
    assert pour.calls == []


def test_manager_enforces_current_skill_minimum_duration() -> None:
    manager, _, pour = make_manager(
        num_envs=1,
        initial_skills=(SkillId.GRASP_LIFT,),
        minimum_steps={SkillId.GRASP_LIFT: 5},
    )

    _, records = manager.step(
        make_task(),
        (valid_state(current_skill=SkillId.GRASP_LIFT, skill_elapsed_steps=1),),
        (HighLevelDecision(SkillId.BIMANUAL_POUR, terminate_current_skill=True),),
    )

    assert not records[0].accepted
    assert records[0].reason == "minimum_duration"
    assert pour.resets == []


def test_manager_resets_only_environment_that_switches() -> None:
    manager, _, pour = make_manager(
        initial_skills=(SkillId.GRASP_LIFT, SkillId.BIMANUAL_POUR),
    )
    states = (
        valid_state(current_skill=SkillId.GRASP_LIFT, skill_elapsed_steps=8),
        valid_state(current_skill=SkillId.BIMANUAL_POUR, skill_elapsed_steps=8),
    )

    manager.step(
        make_task(),
        states,
        (
            HighLevelDecision(SkillId.BIMANUAL_POUR, terminate_current_skill=True),
            HighLevelDecision(SkillId.BIMANUAL_POUR),
        ),
    )

    assert pour.resets == [(0,)]


def test_safety_supervisor_replaces_non_finite_command() -> None:
    unsafe = SkillCommand(ControlMode.POLICY_ACTION, (float("nan"),), "bad_skill")

    safe = SafetySupervisor().validate(unsafe)

    assert safe.control_mode is ControlMode.SAFE_STOP
    assert safe.values == ()


def test_manager_maps_terminal_skills_without_registry_entries() -> None:
    manager, _, _ = make_manager(
        initial_skills=(SkillId.BIMANUAL_POUR, SkillId.GRASP_LIFT),
    )

    commands, _ = manager.step(
        make_task(),
        (valid_state(), valid_state()),
        (
            HighLevelDecision(SkillId.DONE, terminate_current_skill=True),
            HighLevelDecision(SkillId.ABORT, terminate_current_skill=True),
        ),
    )

    assert commands[0].control_mode is ControlMode.NO_OP
    assert commands[1].control_mode is ControlMode.SAFE_STOP
