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


def test_v6_diffusion_actor_uses_fk_observation_contract() -> None:
    constants = _read("pour_right_constants.py")
    env = _read("pour_right_env.py")

    assert _int_constant(constants, "NUM_OBSERVATIONS") == 52
    assert _int_constant(constants, "NUM_CRITIC_OBSERVATIONS") == 143
    assert "finger_grasp_progress" in env
    assert "DiffusionActor 52D obs" in env

    actor_block = env.split("actor_obs = torch.cat([", maxsplit=1)[1].split("], dim=-1)   #", maxsplit=1)[0]
    assert "finger_joint_vel" not in actor_block
    assert "binary_contact" not in actor_block
    assert "tip_force_norm" not in actor_block
    assert "last_actions" not in actor_block
    assert "last_palm_actions" in actor_block
    assert "left_arm_joint_pos" in actor_block
    assert "left_arm_joint_vel" in actor_block
    assert "right_palm_quat_xyzw" in actor_block


def test_v6_diffusion_config_uses_diffusion_actor_and_asymmetric_value() -> None:
    cfg = _read("config/agents/skrl_diffusion_ppo_cfg.yaml")

    assert "5g_pour_right-v6-diffusion" in cfg
    assert "DiffusionActor" in cfg
    assert "bc_checkpoint_path" in cfg
    assert "input: STATES" in cfg
    assert "discount_factor: 0.998" in cfg


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

    assert "real_demo_bc_start_mode: warm_state_match" in cfg
    assert "real_demo_pour_sample_ratio: 0.0" in cfg
    assert 'start_mode: str = "warm_state_match"' in demo
    assert "start_index" in demo
