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
