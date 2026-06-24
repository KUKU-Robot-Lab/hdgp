"""[v4 joint-PD] arm_control_mode="joint" 배선 검증 (정적, Isaac 불필요).

배경: v4 = v5/v6과 동일 reward + joint-position PD 제어(Fabrics IK 우회).
7 action → 7 arm joint target = DEMO_POUR_ARM_POSE + scale·action, joint-limit clamp.
PD 추종은 _apply_action의 set_joint_position_target(articulation drive).
j1-3(어깨)를 정책이 직접 제어 → deep tilt 시 어깨 협응 학습 가능.
"""

from __future__ import annotations

import re
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (TASK_DIR / name).read_text(encoding="utf-8")


def test_arm_control_mode_joint_default() -> None:
    cfg = _read("pour_right_env_cfg.py")
    m = re.search(r'arm_control_mode\s*:\s*str\s*=\s*"(\w+)"', cfg)
    assert m is not None, "arm_control_mode flag 없음"
    assert m.group(1) == "joint", "v4 기본값=joint(joint-PD)"


def test_joint_action_scale_defined() -> None:
    cfg = _read("pour_right_env_cfg.py")
    m = re.search(r'joint_action_scale\s*:\s*float\s*=\s*([0-9.]+)', cfg)
    assert m is not None, "joint_action_scale 없음"
    # START→DEMO 축 gain: ≥1이어야 DEMO(pour) 도달 + 그 너머 deep tilt 여유
    assert 1.0 <= float(m.group(1)) <= 2.0, "축 gain은 [1,2](DEMO 도달+deep 여유)"


def test_pre_physics_dispatch_to_joint() -> None:
    """_pre_physics_step이 joint 모드일 때 _pre_physics_step_joint로 분기·early return."""
    env = _read("pour_right_env.py")
    assert 'self.cfg.arm_control_mode == "joint"' in env, "joint 분기 없음"
    assert "self._pre_physics_step_joint(actions)" in env, "joint 메서드 dispatch 없음"
    assert "def _pre_physics_step_joint" in env, "_pre_physics_step_joint 미정의"


def test_joint_target_start_to_demo_axis_and_clamped() -> None:
    """target = START + gain·EMA(action)·(DEMO−START), joint-limit clamp.

    action=0=START(hold 안전)이 핵심 — DEMO 중심이면 hold서 컵 튕김.
    """
    env = _read("pour_right_env.py")
    block = env.split("def _pre_physics_step_joint", 1)[1].split("def _pre_physics_step(", 1)[0]
    assert "self.robot_start_joint_pos[:, :NUM_ARM_DOF]" in block, "START 중심(축 기준) 아님"
    assert "self._joint_demo_offset" in block, "DEMO−START 축 offset 미적용"
    assert "self.cfg.joint_action_scale" in block, "축 gain 미적용"
    assert "self._ema_arm_action" in block, "EMA smoothing 미적용"
    assert "self._arm_joint_max" in block and "self._arm_joint_min" in block, "joint-limit clamp 없음"
    assert "self.fabric_q[:, :NUM_ARM_DOF] = arm_target" in block, "arm target 미반영"


def test_joint_offset_is_demo_minus_start() -> None:
    """_joint_demo_offset = DEMO−START = NULLSPACE_OFFSET_ARM (action=0→START 보장)."""
    env = _read("pour_right_env.py")
    assert "self._joint_demo_offset = torch.tensor(" in env, "_joint_demo_offset 버퍼 없음"
    assert "NULLSPACE_OFFSET_ARM" in env.split("_joint_demo_offset = torch.tensor(", 1)[1][:80], (
        "_joint_demo_offset가 DEMO−START(NULLSPACE_OFFSET_ARM) 아님"
    )


def test_apply_action_is_position_pd() -> None:
    """제어 PD는 set_joint_position_target(articulation drive) — 그대로 유지."""
    env = _read("pour_right_env.py")
    assert "set_joint_position_target" in env, "joint-position PD 미사용"


def test_ema_arm_action_buffer_and_reset() -> None:
    env = _read("pour_right_env.py")
    assert "self._ema_arm_action = torch.zeros" in env, "_ema_arm_action 버퍼 없음"
    assert env.count("self._ema_arm_action[env_ids] = 0.0") >= 2, "reset 초기화 누락"


def test_reward_identical_to_v5_markers() -> None:
    """reward는 v5/v6과 동일 — Phase3 재설계 마커(outcome/rim/pose) 존재 확인."""
    env = _read("pour_right_env.py")
    cfg = _read("pour_right_env_cfg.py")
    assert "outcome_adr" in env, "outcome ADR(v5 reward) 미존재"
    assert "_rim_antiparallel" in env, "rim_antiparallel tilt(v5 reward) 미존재"
    assert "_pose_success_now" in env, "pose_success(v5 reward) 미존재"
    assert "weight_pour_bead" in cfg and "weight_tilt_delta" in cfg, "v5 reward 가중치 미존재"
