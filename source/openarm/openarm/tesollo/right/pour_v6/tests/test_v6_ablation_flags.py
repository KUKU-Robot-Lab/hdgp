"""pour_v6 = 논문용 ablation 통합 env (정적 계약 테스트).

v6는 v4(=v5 구조 + demo reward 기계 + demo nullspace)를 복사하되,
nullspace baseline을 cfg flag로 전환해 4셀 ablation을 단일 env에서 재현한다.

  | cell          | nullspace_baseline | enable_demo_pose_reward | = 기존 |
  | 순수 DRL      | robot_start        | False                   | v5     |
  | demo nullspace| demo               | False                   | v4     |
  | demo reward   | robot_start        | True                    | 신규   |
  | both          | demo               | True                    | -      |

obs/action 차원은 v4/v5와 동일(불변). 새 변수는 두 직교 flag뿐.
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
    """v6 차원 계약 = v4/v5와 동일 (7-D action, 144 critic obs)."""
    constants = _read("pour_right_constants.py")
    assert _int_constant(constants, "NUM_ACTIONS") == 7
    assert _int_constant(constants, "NUM_OBSERVATIONS") == 55
    assert _int_constant(constants, "NUM_CRITIC_OBSERVATIONS") == 144


def test_config_registers_v6_id_and_entry_point() -> None:
    """gym 등록이 pour_v6 모듈/id로 갱신되어야 한다 (복사 직후엔 v4)."""
    cfg = _read("config/__init__.py")
    assert 'id="open-tesol_r_pour_v6"' in cfg
    assert "openarm.tesollo.right.pour_v6.pour_right_env:PourRightEnv" in cfg
    assert "pour_v4" not in cfg, "v4 참조 잔존 → 잘못된 모듈로 entry"
    assert 'id="open-tesol_r_pour_v4"' not in cfg


def test_two_orthogonal_ablation_flags_exist() -> None:
    """직교 flag 2개: nullspace_baseline(str) + enable_demo_pose_reward(bool)."""
    cfg = _read("pour_right_env_cfg.py")
    m = re.search(r'^\s*nullspace_baseline\s*:\s*str\s*=\s*"(\w+)"', cfg, flags=re.MULTILINE)
    assert m is not None, "nullspace_baseline str flag 없음"
    assert m.group(1) == "demo", "[stage1/2] 기본값=demo (FK 검증: j5 roll deep tilt 주역, robot_start 순수DRL은 미수렴 회귀). flag·분기 구조는 유지."
    assert re.search(r"^\s*enable_demo_pose_reward\s*:\s*bool\s*=", cfg, flags=re.MULTILINE)


def test_nullspace_block_branches_on_flag() -> None:
    """nullspace baseline 블록이 cfg.nullspace_baseline로 분기.

    demo 분기는 demo 구조(j1-4 + ready j5)를, robot_start는 중립을 써야 한다.
    분기 밖의 self-motion 식(baseline + scale·α·offset)은 공통(불변)이어야 한다.
    """
    env = _read("pour_right_env.py")
    block = env.split("_null_cfg = self.fabric_q.detach().clone()", maxsplit=2)[-1]
    block = block.split("self.open_tesollo_fabric.default_config.copy_(_null_cfg)", maxsplit=1)[0]

    assert 'self.cfg.nullspace_baseline == "demo"' in block, "flag 분기 없음"
    # demo 구조는 분기 안에서만 적용
    assert "self._demo_pour_arm_pose[:4]" in block
    assert "self._pour_ready_latched" in block
    # 공통 self-motion 식은 분기와 무관하게 유지
    assert "_coeff * self._nullspace_offset_arm" in block
    assert "self._arm_joint_max" in block and "self._arm_joint_min" in block


def test_demo_reward_runtime_gated_off_by_default() -> None:
    """demo reward는 flag off일 때 정확히 0 (ablation 청결성)."""
    env = _read("pour_right_env.py")
    assert "not self.cfg.enable_demo_pose_reward" in env
    assert 'return {"r_demo_arm_pose": zero, "r_demo_j5": zero' in env
