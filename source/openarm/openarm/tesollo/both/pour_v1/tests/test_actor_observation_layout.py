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


def test_lstm_actor_uses_51d_sim2real_observation_contract() -> None:
    """★[both/pour_v1] 55 → 51 / critic 144 → 140.

    구 pour_sensor 는 왼팔 obs 로 arm7+그리퍼2=9 를 넣었다. pour_v1 왼손은 DG-5FS 20관절이고
    warm 파지자세로 **동결**되므로(정책 미제어) 관측에 넣을 행동 정보가 없다 → arm 7관절만 쓴다.
    같은 코드를 그대로 두면 왼손 20관절이 흘러들어 actor obs 가 91 로 터진다(실행으로 확인).
    """
    constants = _read("pour_right_constants.py")
    env = _read("pour_right_env.py")

    assert _int_constant(constants, "NUM_OBSERVATIONS") == 51
    assert _int_constant(constants, "NUM_CRITIC_OBSERVATIONS") == 140
    assert "finger_grasp_progress" in env

    actor_block = env.split("actor_obs = torch.cat([", maxsplit=1)[1].split("], dim=-1)   #", maxsplit=1)[0]
    assert "left_arm_joint_pos" in actor_block
    assert "left_arm_joint_vel" in actor_block
    assert "finger_joint_vel" not in actor_block
    assert "binary_contact" not in actor_block
    assert "tip_force_norm" not in actor_block
    assert "last_actions" not in actor_block
    assert "last_palm_actions" in actor_block
    assert "pour_point_to_opening" in actor_block
    assert "source_pour_axis_clean" in actor_block
    assert "source_up_axis_clean" in actor_block
    assert "target_up_axis_clean" in actor_block
    assert "right_cup_pos_rel_palm" not in actor_block
    assert "left_cup_pos_rel_palm" not in actor_block
    assert "right_cup_quat_clean" not in actor_block
    assert "transport_summary" not in actor_block
    assert "flow_summary" not in actor_block
    assert "_bead_in_source_fraction" not in actor_block
    assert "_bead_in_target_fraction" not in actor_block
    assert "_bead_cross_fraction" not in actor_block
    assert "_spill_ratio" not in actor_block


def test_v3_lstm_config_encodes_then_recurrs() -> None:
    cfg = _read("config/agents/rl_games_ppo_lstm_cfg.yaml")

    assert "units: [256]" in cfg
    assert "units: 512" in cfg
    assert "before_mlp: False" in cfg
    assert "concat_input: False" in cfg
    assert "concat_output: False" in cfg
    assert "seq_length: 8" in cfg
    assert "minibatch_size: 8192" in cfg
