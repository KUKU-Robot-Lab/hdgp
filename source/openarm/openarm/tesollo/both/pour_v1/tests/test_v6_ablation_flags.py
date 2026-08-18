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


def test_action_obs_contract_both_pour_v1() -> None:
    """[both/pour_v1] action = 오른팔 12D + 왼팔 TCP 3D(DiffIK) = 15D (불변).

    obs 는 왼손 동결로 51/140 (구 pour_sensor 55/144). action 차원만 불변이면
    frozen→learned 단계형 체크포인트 인계가 성립한다.
    """
    constants = _read("pour_right_constants.py")
    assert _int_constant(constants, "NUM_LEFT_TCP_ACTION") == 3
    assert _int_constant(constants, "NUM_ACTIONS") == 15
    assert _int_constant(constants, "NUM_OBSERVATIONS") == 51
    assert _int_constant(constants, "NUM_CRITIC_OBSERVATIONS") == 140


def test_config_registers_v6_id_and_entry_point() -> None:
    """gym 등록이 pour_v6 모듈/id로 갱신되어야 한다 (복사 직후엔 v4)."""
    cfg = _read("config/__init__.py")
    assert 'id="open-tesol_b_pour_v1"' in cfg
    assert "openarm.tesollo.both.pour_v1.pour_right_env:PourRightEnv" in cfg
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


def test_receiver_control_mode_flag_exists_default_learned() -> None:
    """[RA-L] receiver_control_mode(str) flag 존재 + 기본값 learned(M4, 기존 학습 무영향)."""
    cfg = _read("pour_right_env_cfg.py")
    m = re.search(r'^\s*receiver_control_mode\s*:\s*str\s*=\s*"(\w+)"', cfg, flags=re.MULTILINE)
    assert m is not None, "receiver_control_mode str flag 없음"
    assert m.group(1) == "learned", "기본값=learned (M4). frozen/scripted는 override로만."
    # EXP-2 necessity + scripted 파라미터도 존재
    assert re.search(r"^\s*receiver_action_scale\s*:\s*float\s*=", cfg, flags=re.MULTILINE)
    assert re.search(r"^\s*receiver_action_delay_steps\s*:\s*int\s*=", cfg, flags=re.MULTILINE)
    assert re.search(r"^\s*scripted_receiver_clearance\s*:\s*float\s*=", cfg, flags=re.MULTILINE)


def test_receiver_mode_branches_all_three() -> None:
    """env가 frozen/scripted/learned 3분기 모두 처리하고, learned가 기본(else)이어야 한다."""
    env = _read("pour_right_env.py")
    assert '_recv_mode = self.cfg.receiver_control_mode' in env
    assert '_recv_mode == "frozen"' in env, "M0 frozen 분기 없음"
    assert '_recv_mode == "scripted"' in env, "M2 scripted 분기 없음"
    # scripted는 source pour-point(base frame)를 추종
    assert "self._source_pour_point_w" in env
    # EXP-2 scale/delay는 learned 경로에만
    assert "self.cfg.receiver_action_scale" in env
    assert "self.cfg.receiver_action_delay_steps" in env


def test_left_tcp_z_down_allows_lowering_receiver() -> None:
    """receiver TCP z 하강 — **pour_v1 에서 계약이 뒤집혔다** (2026-08-18).

    구 pour_sensor 계약은 `left_tcp_z_down_m == 0.0` 이었다. 이유는 receiver 컵이
    `kinematic-follow` 라 테이블과 물리충돌이 없어, 하강을 허용하면 컵이 테이블을
    **관통**했기 때문이다(실물 불가 → s2r 붕괴).

    pour_v1 의 왼컵은 dynamic rigid body 를 왼손이 실제로 쥔다 → 물리가 관통을 막으므로
    그 근거가 사라졌다. 반대로 왼손이 컵을 **들고** 있어 receiver 가 pour_sensor 대비
    7.4cm 높고(z 0.291 → 0.365), source 도 z 0.367 이라 두 컵이 같은 높이에서 시작한다.
    붓기는 원리상 source 가 위여야 하므로 **하강 없이는 과제가 성립하지 않는다.**
    실측: 하강 금지 상태의 A-E1-frozen 은 2442 epoch 동안 bead_in_target 0.000 평탄
    (mouth_xy_dist 0.2255→0.2265, mouth_z_clearance −0.009).

    구조(z 하한만 별도 clamp)는 그대로 유지한다 — 바뀐 것은 값의 근거뿐이다.
    """
    cfg = _read("pour_right_env_cfg.py")
    m = re.search(r"^\s*left_tcp_z_down_m\s*:\s*float\s*=\s*([0-9.]+)", cfg, flags=re.MULTILINE)
    assert m is not None, "left_tcp_z_down_m flag 없음"
    assert float(m.group(1)) >= 0.06, (
        f"z 하강 허용치 {m.group(1)} — receiver 를 내릴 수 없어 붓기가 불가능하다 "
        "(필요 하강 약 6.5cm = pour_sensor 기하 복원)"
    )
    env = _read("pour_right_env.py")
    assert "self.cfg.left_tcp_z_down_m" in env, "env가 z-down 캡 미적용"
    assert "_wr_min" in env and "_wr_min[0, 2]" in env, "z 하한만 별도 clamp 아님"
