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


def test_v3_lstm_actor_uses_pour_flow_observation_contract() -> None:
    constants = _read("pour_right_constants.py")
    env = _read("pour_right_env.py")

    assert _int_constant(constants, "NUM_OBSERVATIONS") == 60
    assert _int_constant(constants, "NUM_CRITIC_OBSERVATIONS") == 140
    assert "finger_grasp_progress" in env
    assert "flow_summary" in env

    actor_block = env.split("actor_obs = torch.cat([", maxsplit=1)[1].split("], dim=-1)   #", maxsplit=1)[0]
    assert "finger_joint_vel" not in actor_block
    assert "binary_contact" not in actor_block
    assert "tip_force_norm" not in actor_block
    assert "last_actions" not in actor_block
    assert "last_palm_actions" in actor_block


def test_v3_lstm_config_encodes_then_recurrs() -> None:
    cfg = _read("config/agents/rl_games_ppo_lstm_cfg.yaml")

    assert "units: [256]" in cfg
    assert "units: 512" in cfg
    assert "before_mlp: False" in cfg
    assert "concat_input: False" in cfg
    assert "concat_output: False" in cfg
    assert "seq_length: 8" in cfg
    assert "minibatch_size: 8192" in cfg
