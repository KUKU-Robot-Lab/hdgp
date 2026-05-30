from __future__ import annotations

import re
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (TASK_DIR / path).read_text()


def _int_constant(source: str, name: str) -> int:
    match = re.search(rf"^{name}\s*=\s*(.+?)(?:\s+#.*)?$", source, flags=re.MULTILINE)
    assert match is not None, f"{name} constant not found"
    expr = match.group(1).strip()
    if expr.isdigit():
        return int(expr)
    total = 0
    for term in (part.strip() for part in expr.split("+")):
        total += _int_constant(source, term)
    return total


def test_v5_lstm_actor_uses_pour_flow_observation_contract() -> None:
    constants = _read("pour_right_constants.py")
    env = _read("pour_right_env.py")

    assert _int_constant(constants, "NUM_OBSERVATIONS") == 60
    assert _int_constant(constants, "NUM_CRITIC_OBSERVATIONS") == 143
    assert "finger_grasp_progress" in env
    assert "flow_summary" in env

    actor_block = env.split("actor_obs = torch.cat([", maxsplit=1)[1].split("], dim=-1)   #", maxsplit=1)[0]
    assert "finger_joint_vel" not in actor_block
    assert "binary_contact" not in actor_block
    assert "tip_force_norm" not in actor_block
    assert "last_actions" not in actor_block
    assert "last_palm_actions" in actor_block


def test_v5_lstm_config_encodes_then_recurrs() -> None:
    cfg = _read("config/agents/rl_games_ppo_cfg.yaml")

    assert "units: [256]" in cfg
    assert "units: 512" in cfg
    assert "before_mlp: False" in cfg
    assert "concat_input: False" in cfg
    assert "concat_output: False" in cfg
    assert "seq_length: 32" in cfg
    assert "minibatch_size: 8192" in cfg
    assert "bc_seq_len: 32" in cfg


def test_v5_real_demo_bc_buffer_matches_actor_observation_contract() -> None:
    demo = _read("demo_bc_buffer.py")

    obs_block = demo.split("obs = torch.cat([", maxsplit=1)[1].split("], dim=-1)", maxsplit=1)[0]
    assert "finger_grasp_progress" in obs_block
    assert "last_palm_actions" in obs_block
    assert "flow_zero" in obs_block
    assert "binary_contact" not in obs_block
    assert "tip_norm" not in obs_block


def test_v5_real_demo_bc_uses_warm_aligned_full_trajectory() -> None:
    cfg = _read("config/agents/rl_games_ppo_cfg.yaml")
    demo = _read("demo_bc_buffer.py")

    assert "real_demo_offline_bc_enable: False" in cfg
    assert "real_demo_bc_start_mode: warm_state_match" in cfg
    assert "real_demo_pour_sample_ratio: 0.0" in cfg
    assert "real_demo_time_bin_weights: [0.20, 0.20, 0.25, 0.35]" in cfg
    assert 'start_mode: str = "warm_state_match"' in demo
    assert "start_index" in demo
    assert "time_bin_weights" in demo


def test_v5_real_demo_bc_is_rollout_conditioned_by_default() -> None:
    cfg = _read("config/agents/rl_games_ppo_cfg.yaml")
    agent = _read("lstm_bc_agent.py")
    env = _read("pour_right_env.py")
    env_cfg = _read("pour_right_env_cfg.py")

    assert "rollout-conditioned real-demo teacher loss" in cfg
    assert "real_demo_teacher_palm_weight: 1.0" in cfg
    assert "real_demo_teacher_finger_weight: 0.0" in cfg
    assert 'res_dict["demo_teacher_actions"]' in agent
    assert 'input_dict.get("demo_teacher_actions"' in agent
    assert "get_demo_teacher_actions" in env
    assert "_demo_pose_id_valid" in env
    assert "weight_demo_arm_pose: float = 1.0" in env_cfg
    assert "weight_demo_palm_pose: float = 1.0" in env_cfg
    assert "weight_demo_smooth: float = 0.02" in env_cfg


def test_v5_non_demo_reward_settings_match_v3() -> None:
    env_cfg = _read("pour_right_env_cfg.py")

    expected = [
        "success_target_fill_ratio: float = 0.50",
        "success_spill_max: float = 0.40",
        "weight_grasp_maintain: float = 0.50",
        "weight_contact_maintain: float = 0.50",
        "weight_force_balance: float = 0.30",
        "weight_finger_curl: float = 0.50",
        "weight_dist_to_target: float = 10.0",
        "weight_pour_dist: float = 12.0",
        "weight_tilt: float = 40.0",
        "weight_align: float = 6.0",
        "weight_bead_progressive: float = 200.0",
        "weight_bead_entry_delta: float = 300.0",
        "weight_source_drain: float = 20.0",
        "curriculum_pour_warmup_steps: int = 40000",
        "curriculum_bead_warmup_start: int = 0",
        "weight_spill: float = 40.0",
        "weight_j0_ext_rot: float = 3.0",
        "weight_premature_tilt: float = 1.00",
        "weight_action_rate_palm: float = 0.02",
        "weight_action_rate_finger: float = 0.005",
        '"spill_weight": (1.0, 15.0)',
        "spill_adr_trigger_threshold: float = 0.10",
        "pour_tilt_sharpness: float = 4.0",
        "pour_binary_xy_thresh: float = 0.20",
    ]
    for text in expected:
        assert text in env_cfg

    disabled_v5_only_terms = [
        "weight_pour_xy: float = 0.0",
        "weight_capture_spill: float = 0.0",
        "weight_simple_spill: float = 0.0",
        "weight_all_beads_bonus: float = 0.0",
        "weight_cup_collision: float = 0.0",
        "weight_arm_joint_vel: float = 0.0",
        "weight_arm_joint_acc: float = 0.0",
        "weight_arm_joint_vel_approach: float = 0.0",
        "weight_arm_joint_jerk: float = 0.0",
    ]
    for text in disabled_v5_only_terms:
        assert text in env_cfg
