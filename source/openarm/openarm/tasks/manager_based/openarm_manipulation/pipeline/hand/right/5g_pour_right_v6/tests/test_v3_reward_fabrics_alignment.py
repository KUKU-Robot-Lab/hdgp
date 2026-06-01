from __future__ import annotations

import re
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (TASK_DIR / path).read_text()


def _cfg_value(source: str, name: str) -> str:
    match = re.search(rf"^\s*{name}:\s*[^=]+=\s*(.+?)(?:\s+#.*)?$", source, flags=re.MULTILINE)
    assert match is not None, f"{name} not found"
    return match.group(1).strip()


def _method_body(source: str, name: str) -> str:
    marker = f"    def {name}("
    start = source.index(marker)
    next_method = source.find("\n    def ", start + len(marker))
    if next_method == -1:
        return source[start:]
    return source[start:next_method]


def _assignment_block(source: str, target: str) -> str:
    start = source.index(target)
    end = source.index("\n        )", start)
    return source[start:end]


def test_v6_keeps_diffusion_observation_contract() -> None:
    constants = _read("pour_right_constants.py")
    env = _read("pour_right_env.py")

    assert "NUM_OBSERVATIONS = 52" in constants
    assert "DiffusionActor 52D obs" in env
    assert "left_arm_joint_pos" in env
    assert "right_cup_pos_rel_palm" in env


def test_reward_config_matches_v3_public_values() -> None:
    cfg = _read("pour_right_env_cfg.py")

    expected = {
        "source_outer_radius": "0.045",
        "success_target_fill_ratio": "0.50",
        "success_spill_max": "0.40",
        "weight_palm_pose": "10.0",
        "z_window_lower_ramp": "0.01",
        "z_window_upper_end": "0.08",
        "z_window_upper_ramp": "0.03",
        "curriculum_pour_warmup_steps": "40000",
        "curriculum_bead_warmup_start": "0",
        "curriculum_bead_warmup_steps": "60000",
        "weight_action_rate_palm": "0.02",
        "weight_action_rate_finger": "0.005",
        "pour_tilt_sharpness": "4.0",
        "pour_binary_xy_thresh": "0.20",
    }
    for name, value in expected.items():
        assert _cfg_value(cfg, name) == value


def test_fabrics_control_uses_v3_mouth_gate_rim_pivot_and_nullspace() -> None:
    env = _read("pour_right_env.py")
    pre_physics = _method_body(env, "_pre_physics_step")

    assert "self._mouth_xy_distance" in pre_physics
    assert "self._cup_center_xy_dist) / gate_den" not in pre_physics
    assert "rim_env = self._source_pour_point_w - self.scene.env_origins" in pre_physics
    assert "pour_point_target_xy = rim_env[:, :2] + delta[:, :2]" in pre_physics
    assert "palm_pose[:, :2] = pour_point_target_xy - expected_offset_xy" in pre_physics
    assert "_null_cfg[:, 3] = _null_cfg[:, 3] * 0.95 + 1.84 * 0.05" in pre_physics
    assert "_null_cfg[:, 4] = _null_cfg[:, 4] * 0.90 + (-1.16) * 0.10" in pre_physics


def test_intermediate_geometry_uses_v3_lowest_rim_pour_point() -> None:
    env = _read("pour_right_env.py")
    intermediate = _method_body(env, "_compute_intermediate_values")

    assert "_rim_center_w = self.cup.data.root_pos_w + quat_apply(" in intermediate
    assert "_gravity_perp_hat = _gravity_perp / _gravity_perp_norm" in intermediate
    assert "self._source_pour_point_w = _rim_center_w + self.cfg.source_outer_radius * _gravity_perp_hat" in intermediate


def test_reward_total_uses_v3_terms_without_v6_extra_costs() -> None:
    env = _read("pour_right_env.py")
    rewards = _method_body(env, "_get_rewards")
    total = _assignment_block(rewards, "total = (")

    required_terms = [
        "r_hold",
        "r_dist_to_target",
        "r_palm_pose",
        'demo_terms["r_demo_arm_pose"]',
        "r_pour_dist",
        "r_pour_stage",
        "r_source_drain",
        "self.cfg.weight_success * r_success",
        "spill_weight * spill_cost",
    ]
    for term in required_terms:
        assert term in total

    forbidden_terms = [
        "r_demo_palm_pose",
        "weight_premature_tilt",
        "weight_grasp_loss",
        "weight_cup_collision",
        "action_rate_penalty",
        "arm_vel_cost",
        "cost_demo_smooth",
        "cost_thumb_grip",
        "simple_reward_terms",
    ]
    for term in forbidden_terms:
        assert term not in total
