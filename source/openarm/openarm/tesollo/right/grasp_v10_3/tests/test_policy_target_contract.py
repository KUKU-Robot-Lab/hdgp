from pathlib import Path

import torch

from openarm.tesollo.right.grasp_v10_3.finger_action_utils import (
    compute_preset_residual_finger_targets,
)
from openarm.tesollo.right.grasp_v10_3.grasp_right_constants import (
    FINGER_ACTION_SLICE,
    NUM_ACTIONS,
    PALM_POS_ACTION_SLICE,
    PALM_QUAT_ACTION_SLICE,
)
from openarm.tesollo.right.grasp_v10_3.grasp_right_preset import HAND_GRASP_POSE


_ROOT = Path(__file__).resolve().parents[1]


def test_action_contract_is_27d_target_action() -> None:
    assert NUM_ACTIONS == 27
    assert PALM_POS_ACTION_SLICE == slice(0, 3)
    assert PALM_QUAT_ACTION_SLICE == slice(3, 7)
    assert FINGER_ACTION_SLICE == slice(7, 27)


def test_quaternion_action_normalizes_to_unit_norm() -> None:
    quat_action = torch.tensor([[0.2, -0.4, 0.1, 0.5], [0.0, 0.0, 0.0, 2.0]])
    quat = torch.nn.functional.normalize(quat_action, dim=-1, eps=1e-6)
    assert torch.allclose(quat.norm(dim=-1), torch.ones(2), atol=1e-6)


def test_finger_action_is_small_residual_around_preset() -> None:
    preset = torch.tensor([0.0, 1.0, 3.0])
    lower = torch.tensor([-1.0, 0.0, 2.0])
    upper = torch.tensor([1.0, 2.0, 4.0])
    action = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 0.5]])
    target = compute_preset_residual_finger_targets(
        preset,
        action,
        lower,
        upper,
        residual_scale=0.2,
    )
    assert torch.all(target >= lower.unsqueeze(0))
    assert torch.all(target <= upper.unsqueeze(0))
    assert torch.allclose(target[0], preset)
    assert torch.allclose(target[1], torch.tensor([0.2, 0.8, 3.1]))


def test_hand_residual_mask_keeps_fixed_joints_at_preset() -> None:
    preset = torch.tensor(HAND_GRASP_POSE, dtype=torch.float32)
    lower = torch.full_like(preset, -4.0)
    upper = torch.full_like(preset, 4.0)
    action = torch.ones(2, preset.numel(), dtype=torch.float32)
    action[1] = -1.0
    mask = torch.ones_like(preset)
    mask[[0, 4, 8, 12, 16, 17]] = 0.0

    target = compute_preset_residual_finger_targets(
        preset,
        action,
        lower,
        upper,
        residual_scale=0.15,
        residual_mask=mask,
    )

    fixed = torch.tensor([0, 4, 8, 12, 16, 17])
    assert torch.allclose(target[:, fixed], preset[fixed].unsqueeze(0).expand(2, -1))
    assert torch.allclose(target[0, mask.bool()], preset[mask.bool()] + 0.15)
    assert torch.allclose(target[1, mask.bool()], preset[mask.bool()] - 0.15)


def test_live_fabrics_uses_quaternion_target_without_scripted_lift() -> None:
    env = (_ROOT / "grasp_right_env.py").read_text(encoding="utf-8")
    assert '"quaternion"' in env
    assert "self.palm_pose_targets = torch.zeros(self.num_envs, 7" in env
    assert "compute_lift_stabilize_palm_targets" not in env
    assert "LIFT_START_STEP" not in env
    assert "lift_finger_pos_buf" not in env
    assert "compute_preset_residual_finger_targets" in env


def test_palm_contact_sensor_uses_palm_link_net_force_without_actor_obs_growth() -> None:
    cfg = (_ROOT / "grasp_right_env_cfg.py").read_text(encoding="utf-8")
    env = (_ROOT / "grasp_right_env.py").read_text(encoding="utf-8")
    constants = (_ROOT / "grasp_right_constants.py").read_text(encoding="utf-8")

    assert "palm_sensor_cfg" in cfg
    palm_cfg = cfg.split("palm_sensor_cfg", 1)[1]
    assert 'prim_path="/World/envs/env_.*/Robot/rl_dg_palm"' in palm_cfg
    assert "filter_prim_paths_expr" not in palm_cfg.split("history_length=1", 1)[0]
    assert "self._palm_sensor.data.net_forces_w[:, 0, :]" in env
    assert "quat_apply_inverse" in env
    assert "palm_force = torch.relu(-palm_force_local[:, 0])" in env
    assert "NUM_OBSERVATIONS = 134" in constants


def test_reward_gate_success_contract_uses_tip5_and_palm_not_middle_or_pose_filters() -> None:
    env = (_ROOT / "grasp_right_env.py").read_text(encoding="utf-8")

    assert "grasp_ready_now = full_tip_contact_bool & palm_contact_bool" in env
    latch_block = env.split("grasp_ready_now =", 1)[1].split("self.grasp_ready_hold_buf", 1)[0]
    assert "middle" not in latch_block
    assert "cup_tilt" not in latch_block
    assert "cup_xy_displacement" not in latch_block

    assert "* lift_gate\n            * envelope_grasp" in env
    assert "success_now = in_or_past_lift & lifted & full_tip_contact & palm_contact & upright_success" in env
    assert "finger_depth =" not in env
    assert "middle_contact_ready" not in env
    assert 'self.extras["contact/middle_count"]' not in env
