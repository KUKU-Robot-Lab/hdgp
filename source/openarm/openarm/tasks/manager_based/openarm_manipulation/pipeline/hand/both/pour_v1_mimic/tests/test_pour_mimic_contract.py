from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parents[1]
_INIT = _ROOT / "__init__.py"
_ENV = _ROOT / "pour_mimic_env.py"
_CFG = _ROOT / "pour_mimic_env_cfg.py"
_MATH = _ROOT / "pour_mimic_math.py"
_SUBTASK = _ROOT / "pour_mimic_subtask.py"


def _load_math_module():
    spec = importlib.util.spec_from_file_location("pour_mimic_math", _MATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pour_mimic_package_registers_eval_and_generation_tasks() -> None:
    init_text = _INIT.read_text(encoding="utf-8")

    assert '_register("Pour-Mimic"' in init_text
    assert '_register("Pour-Mimic-Mimic"' in init_text
    assert ".pour_mimic_managed_env:PourMimicManagedEnv" in init_text
    assert 'f"{__name__}:PourMimicManagedEnvCfg"' in init_text
    assert 'f"{__name__}:PourMimicManagedMimicEnvCfg"' in init_text


def test_pour_mimic_cfg_matches_phase_1_contract() -> None:
    cfg_text = _CFG.read_text(encoding="utf-8")

    assert "class PourMimicEnvCfg(PourEnvCfg)" in cfg_text
    assert "class PourMimicMimicEnvCfg(PourMimicEnvCfg, MimicEnvCfg)" in cfg_text
    assert "num_actions: int = NUM_ACTIONS" in cfg_text
    assert "num_observations: int = ACTOR_OBSERVATION_DIM" in cfg_text
    assert "episode_length_s: float = 10.0" in cfg_text
    assert "generation_num_trials = 1000" in cfg_text
    for signal in ("grasp_done", "lift_done", "align_done", "pour_done"):
        assert signal in cfg_text


def test_pour_mimic_env_exposes_required_isaaclab_mimic_methods() -> None:
    env_text = _ENV.read_text(encoding="utf-8")

    assert "class PourMimicEnv(PourEnv, ManagerBasedRLMimicEnv)" in env_text
    for method in (
        "def get_robot_eef_pose(",
        "def target_eef_pose_to_action(",
        "def action_to_target_eef_pose(",
        "def actions_to_gripper_actions(",
        "def get_object_poses(",
        "def get_subtask_term_signals(",
    ):
        assert method in env_text


def test_pour_mimic_subtask_module_defines_phase_signal_contract() -> None:
    subtask_text = _SUBTASK.read_text(encoding="utf-8")

    assert "RIGHT_EEF_NAME = \"right\"" in subtask_text
    assert "LEFT_EEF_NAME = \"left\"" in subtask_text
    assert "SUBTASK_TERM_SIGNALS" in subtask_text
    assert "validate_subtask_signals" in subtask_text
    for signal in ("grasp_done", "lift_done", "align_done", "pour_done"):
        assert signal in subtask_text


def test_action_pose_round_trip_preserves_delta_and_gripper_contract() -> None:
    math_mod = _load_math_module()

    current_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    action = torch.zeros(2, 18, dtype=torch.float32)
    action[:, :3] = torch.tensor([[0.10, -0.02, 0.03], [-0.04, 0.05, 0.06]])
    action[:, 3:6] = torch.tensor([[0.0, 0.0, 0.20], [0.10, 0.0, 0.0]])
    action[:, 6:11] = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0], [0.25] * 5])

    target_pose = math_mod.action_to_target_pose(current_pose, action)
    recovered = math_mod.target_pose_to_action(current_pose, target_pose, action[:, 6:11])

    assert target_pose.shape == (2, 4, 4)
    assert recovered.shape == (2, 18)
    torch.testing.assert_close(recovered[:, :11], action[:, :11], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(recovered[:, 11:], torch.zeros(2, 7), atol=0.0, rtol=0.0)


def test_action_pose_mapping_applies_env_delta_scale() -> None:
    math_mod = _load_math_module()

    current_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    action = torch.zeros(1, 18, dtype=torch.float32)
    action[0, 0] = 1.0
    action[0, 5] = 1.0

    target_pose = math_mod.action_to_target_pose(current_pose, action)

    assert abs(target_pose[0, 0, 3].item() - math_mod.RIGHT_PALM_DELTA_XYZ_SCALE) < 1e-6
    recovered = math_mod.target_pose_to_action(current_pose, target_pose, action[:, 6:11])
    torch.testing.assert_close(recovered[:, :6], action[:, :6], atol=1e-5, rtol=1e-5)


def test_actions_to_gripper_actions_keeps_batch_or_sequence_shape() -> None:
    math_mod = _load_math_module()

    batched = torch.arange(36, dtype=torch.float32).reshape(2, 18)
    sequenced = torch.arange(3 * 2 * 18, dtype=torch.float32).reshape(3, 2, 18)

    torch.testing.assert_close(math_mod.actions_to_gripper_actions(batched), batched[:, 6:11])
    torch.testing.assert_close(math_mod.actions_to_gripper_actions(sequenced), sequenced[:, :, 6:11])
