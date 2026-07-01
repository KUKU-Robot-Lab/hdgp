"""inspire_r_pour_v1 = pour_v6 구조 RH56F1 이식본 (정적 계약 테스트).

RH56F1(6-DOF 손)으로 이식 시 변경된 계약:
  - action:  12D (6 palm + 1 nullspace + 5 per-finger lerp)
  - actor obs: 51D
  - critic obs: 112D
  - registry id: open-rh56f1_r_pour_v1
  - entry point: openarm.rh56f1.right.pour_v1.pour_right_env:PourRightEnv
  - fabric attribute: self.fabric (not self.open_tesollo_fabric)
  - demo reward return: {"r_demo_arm_pose": zero, "demo_arm_joint_err": zero}

ablation flag 구조(nullspace_baseline / enable_demo_pose_reward)는
tesollo pour_v6과 동일하게 유지.
"""
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


def test_action_obs_contract_unchanged() -> None:
    """RH56F1 이식 차원 계약: 12D action, 51D actor obs, 112D critic obs."""
    constants = _read("pour_right_constants.py")
    assert _int_constant(constants, "NUM_ACTIONS") == 12
    assert _int_constant(constants, "NUM_OBSERVATIONS") == 51
    assert _int_constant(constants, "NUM_CRITIC_OBSERVATIONS") == 112


def test_config_registers_rh56f1_id_and_entry_point() -> None:
    """gym 등록이 rh56f1 모듈/id로 갱신되어야 한다."""
    cfg = _read("config/__init__.py")
    assert 'id="open-rh56f1_r_pour_v1"' in cfg
    assert "openarm.rh56f1.right.pour_v1.pour_right_env:PourRightEnv" in cfg


def test_two_orthogonal_ablation_flags_exist() -> None:
    """직교 flag 2개: nullspace_baseline(str) + enable_demo_pose_reward(bool)."""
    cfg = _read("pour_right_env_cfg.py")
    m = re.search(r'^\s*nullspace_baseline\s*:\s*str\s*=\s*"(\w+)"', cfg, flags=re.MULTILINE)
    assert m is not None, "nullspace_baseline str flag 없음"
    assert m.group(1) == "demo", "[stage1/2] 기본값=demo (FK 검증: j5 roll deep tilt 주역). flag·분기 구조는 유지."
    assert re.search(r"^\s*enable_demo_pose_reward\s*:\s*bool\s*=", cfg, flags=re.MULTILINE)


def test_nullspace_block_branches_on_flag() -> None:
    """nullspace baseline 블록이 cfg.nullspace_baseline로 분기.

    demo 분기는 demo 구조(j1-4 + ready j5)를, robot_start는 중립을 써야 한다.
    분기 밖의 self-motion 식(baseline + scale·α·offset)은 공통(불변)이어야 한다.
    """
    env = _read("pour_right_env.py")
    # RH56F1: self.fabric.default_config (not self.open_tesollo_fabric.default_config)
    block = env.split("_null_cfg = self.fabric_q.detach().clone()", maxsplit=2)[-1]
    block = block.split("self.fabric.default_config.copy_(_null_cfg)", maxsplit=1)[0]

    assert 'self.cfg.nullspace_baseline == "demo"' in block, "flag 분기 없음"
    # demo 구조는 분기 안에서만 적용
    assert "self._demo_pour_arm_pose[:4]" in block
    assert "self._pour_ready_latched" in block
    # 공통 self-motion 식은 분기와 무관하게 유지
    assert "_coeff * self._nullspace_offset_arm" in block
    assert "self._arm_joint_max" in block and "self._arm_joint_min" in block


def test_demo_reward_runtime_gated_off_by_default() -> None:
    """demo reward는 flag off일 때 정확히 0 (ablation 청결성).

    RH56F1: _get_demo_pose_reward_terms()는 메서드로 존재하지만 _get_rewards에서
    호출되지 않으므로 enable_demo_pose_reward=False(기본값)일 때 항상 0.
    j5 demo reward 없음 → return dict는 {r_demo_arm_pose, demo_arm_joint_err}.
    """
    env = _read("pour_right_env.py")
    # flag는 env에 존재해야 (bank 로딩 조건으로 사용)
    assert "enable_demo_pose_reward" in env
    # _get_demo_pose_reward_terms 메서드가 존재하고 early-exit 경로를 갖추어야 함
    assert "_get_demo_pose_reward_terms" in env
    # RH56F1 이식: r_demo_j5 없음 (tesollo 고유 reward)
    assert 'return {"r_demo_arm_pose": zero, "demo_arm_joint_err": zero}' in env
    # demo reward는 _get_rewards 총합에 포함되지 않음 (기본 비활성 보장)
    assert "+ r_demo_arm_pose" not in env
