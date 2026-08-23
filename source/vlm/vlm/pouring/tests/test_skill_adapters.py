from __future__ import annotations

from pathlib import Path

import pytest

from vlm.pouring.checkpoint_resolver import PolicyArtifacts
from vlm.pouring.contracts import ControlMode, SkillId
from vlm.pouring.skills.approach import ApproachSkill
from vlm.pouring.skills.bimanual_pour import BimanualPourSkill
from vlm.pouring.skills.grasp_lift import GraspLiftSkill
from vlm.pouring.skills.pre_grasp_bridge import PreGraspBridgeSkill
from vlm.pouring.skills.pre_pour_bridge import PrePourBridgeSkill
from vlm.pouring.skills.recovery import RecoverySkill
from vlm.pouring.tests.test_state_provider import valid_state


class FakeBackend:
    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim
        self.env_ids: tuple[int, ...] = ()
        self.observations: tuple[tuple[float, ...], ...] = ()
        self.resets: list[tuple[int, ...]] = []

    def infer(self, env_ids, observations):
        self.env_ids = env_ids
        self.observations = observations
        return tuple((0.25,) * self.action_dim for _ in observations)

    def reset(self, env_ids: tuple[int, ...]) -> None:
        self.resets.append(env_ids)


def artifacts(task_id: str) -> PolicyArtifacts:
    return PolicyArtifacts(task_id, Path("run"), Path("model.pth"), Path("agent.yaml"), Path("env.yaml"))


def test_approach_skill_builds_source_relative_task_space_target() -> None:
    skill = ApproachSkill(offset=(0.0, -0.1, 0.05), orientation_wxyz=(1.0, 0.0, 0.0, 0.0))

    command = skill.infer((0,), (valid_state(),))[0]

    assert skill.skill_id is SkillId.APPROACH
    assert command.control_mode is ControlMode.TASK_SPACE_POSE
    assert command.values == (0.2, -0.2, 0.35, 1.0, 0.0, 0.0, 0.0)


def test_pre_grasp_bridge_noops_when_ready_and_bounds_position_correction() -> None:
    skill = PreGraspBridgeSkill(max_position_step=0.02)

    ready, correcting = skill.infer(
        (0, 1),
        (valid_state(pregrasp_ready=True), valid_state()),
    )

    assert ready.control_mode is ControlMode.NO_OP
    assert correcting.control_mode is ControlMode.TASK_SPACE_POSE
    assert max(abs(value) for value in correcting.values[:3]) <= 0.02


def test_grasp_adapter_validates_106d_observation_and_11d_action() -> None:
    backend = FakeBackend(action_dim=11)
    skill = GraspLiftSkill(
        artifacts("open-tesol_r_grasp_v1-lstm"),
        observation_builder=lambda env_ids, states: tuple((0.0,) * 106 for _ in states),
        backend=backend,
        observation_dim=106,
        action_dim=11,
    )

    command = skill.infer((3,), (valid_state(),))[0]
    skill.reset((3,))

    assert len(backend.observations[0]) == 106
    assert len(command.values) == 11
    assert backend.resets == [(3,)]


def test_adapter_reads_dimensions_from_the_run_env_yaml(tmp_path: Path) -> None:
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text(
        "seed: 42\nobservation_space: 51\nnum_observations: 51\n"
        "action_space: 15\nnum_actions: 15\n"
    )
    backend = FakeBackend(action_dim=15)
    skill = BimanualPourSkill(
        PolicyArtifacts("open-tesol_b_pour_v1-lstm", tmp_path, tmp_path / "model.pth",
                        tmp_path / "agent.yaml", env_yaml),
        observation_builder=lambda env_ids, states: tuple((0.0,) * 51 for _ in states),
        backend=backend,
    )

    command = skill.infer((0,), (valid_state(),))[0]

    assert (skill.observation_dim, skill.action_dim) == (51, 15)
    assert len(backend.observations[0]) == 51
    assert len(command.values) == 15


def test_adapter_rejects_env_yaml_without_contract_keys(tmp_path: Path) -> None:
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text("scene: {}\n")
    with pytest.raises(ValueError, match="contract keys"):
        BimanualPourSkill(
            PolicyArtifacts("open-tesol_b_pour_v1-lstm", tmp_path, tmp_path / "model.pth",
                            tmp_path / "agent.yaml", env_yaml),
            observation_builder=lambda env_ids, states: ((0.0,) * 51,),
            backend=FakeBackend(15),
        )


@pytest.mark.parametrize(
    ("skill_factory", "message"),
    [
        (
            lambda: GraspLiftSkill(
                artifacts("open-tesol_r_grasp_v1-lstm"),
                observation_builder=lambda env_ids, states: ((0.0,) * 105,),
                backend=FakeBackend(11),
                observation_dim=106,
                action_dim=11,
            ),
            "observation",
        ),
        (
            lambda: BimanualPourSkill(
                artifacts("open-tesol_b_pour_v1-lstm"),
                observation_builder=lambda env_ids, states: ((0.0,) * 55,),
                backend=FakeBackend(11),
                observation_dim=55,
                action_dim=12,
            ),
            "action",
        ),
    ],
)
def test_policy_adapters_reject_dimension_mismatch(skill_factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        skill_factory().infer((0,), (valid_state(),))


def test_pre_pour_bridge_requires_validation_and_semantic_readiness(tmp_path: Path) -> None:
    class FakeBridge:
        def load(self, path, **kwargs):
            return object()

    skill = PrePourBridgeSkill(FakeBridge(), tmp_path / "warm.hdf5", expected_object_spawn_z=0.297)
    with pytest.raises(RuntimeError, match="validate"):
        skill.infer((0,), (valid_state(warm_state_valid=True),))

    skill.validate()
    ready, waiting = skill.infer(
        (0, 1),
        (valid_state(warm_state_valid=True), valid_state(warm_state_valid=False)),
    )

    assert ready.control_mode is ControlMode.NO_OP
    assert waiting.control_mode is ControlMode.SAFE_STOP


def test_recovery_skill_always_safe_stops() -> None:
    command = RecoverySkill().infer((0,), (valid_state(),))[0]

    assert command.control_mode is ControlMode.SAFE_STOP
