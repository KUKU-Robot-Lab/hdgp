"""[stage2] n_demo nullspace α offset 검증 (정적, Isaac 불필요).

배경: 기존 α offset=(demo−start)는 palm pose를 보존하지 않는 'tilt 슬라이더'라
정책이 drift 회피로 α를 낮춰 deep tilt 미달. stage2는 demo 자세 palm 6D Jacobian의
진짜 nullspace(elbow-swivel) n_demo로 교체 → α가 tilt 안 망치고 잉여 1-DOF만 조절.
"""

from __future__ import annotations

import importlib
import math
import re
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (TASK_DIR / name).read_text(encoding="utf-8")


def _load_constants():
    return importlib.import_module(
        "openarm.tesollo.right.pour_v5.pour_right_constants"
    )


def test_n_demo_offset_defined_shape() -> None:
    c = _load_constants()
    n = c.N_DEMO_NULLSPACE_OFFSET
    assert len(n) == 7, "n_demo는 7-DOF arm 축이어야 함"


def test_n_demo_j4_component_zero() -> None:
    """elbow-swivel은 j4(elbow)를 안 건드림 — FK 검증 결과 j4 성분=0."""
    c = _load_constants()
    assert abs(c.N_DEMO_NULLSPACE_OFFSET[3]) < 1e-2, "n_demo j4 성분은 ~0이어야 함"


def test_n_demo_unit_norm() -> None:
    c = _load_constants()
    norm = math.sqrt(sum(x * x for x in c.N_DEMO_NULLSPACE_OFFSET))
    assert abs(norm - 1.0) < 1e-2, "n_demo는 단위벡터(SVD null vec)여야 함"


def test_n_demo_orthogonal_to_tilt_axis() -> None:
    """n_demo는 tilt 축(demo−start)과 거의 직교(cos<0.1) — palm pose 보존 근거."""
    c = _load_constants()
    n = c.N_DEMO_NULLSPACE_OFFSET
    ds = c.NULLSPACE_OFFSET_ARM
    dot = sum(a * b for a, b in zip(n, ds))
    nn = math.sqrt(sum(a * a for a in n))
    nd = math.sqrt(sum(b * b for b in ds))
    cos = abs(dot) / (nn * nd)
    assert cos < 0.1, f"n_demo가 tilt축과 직교해야 palm 보존 (cos={cos:.3f})"


def test_cfg_offset_mode_true_nullspace() -> None:
    cfg = _read("pour_right_env_cfg.py")
    m = re.search(
        r'^\s*nullspace_offset_mode\s*:\s*str\s*=\s*"(\w+)"', cfg, flags=re.MULTILINE
    )
    assert m is not None, "nullspace_offset_mode flag 없음"
    assert m.group(1) == "true_nullspace", "stage2 기본값=true_nullspace"


def test_env_selects_n_demo_when_true_nullspace() -> None:
    env = _read("pour_right_env.py")
    assert "N_DEMO_NULLSPACE_OFFSET" in env, "env가 n_demo offset 미사용"
    assert 'nullspace_offset_mode' in env, "env가 offset_mode 분기 없음"


def test_post_ik_manipulation_disabled() -> None:
    """[β 수정] post-IK 관절 클램프/override 전면 비활성화 (pour-point 보존).

    구 stage3 clamp·j5 override는 IK 후 단일관절을 덮어써 end-effector(주둥이) 포즈를 깸.
    deep tilt는 β-setpoint→rim-pivot 회전으로 구동하므로 post-IK 조작 불필요.
    """
    cfg = _read("pour_right_env_cfg.py")
    assert re.search(r'pour_phase_clamp_enable\s*:\s*bool\s*=\s*False', cfg), (
        "pour_phase_clamp_enable=False(post-IK 조작 off)이어야"
    )


def test_pour_phase_clamp_gated_on_ready() -> None:
    """클램프가 _pour_ready_latched(pour 단계)로 게이트 — approach 무영향."""
    env = _read("pour_right_env.py")
    assert "pour_phase_clamp_enable" in env, "env가 클램프 미적용"
    assert "_pour_ready_latched" in env and "_pour_clamp_lo" in env, "ready 게이트/band 미사용"


def test_b_trajectory_mode_cfg() -> None:
    """[B-trajectory] action 모드·β 채널 cfg."""
    cfg = _read("pour_right_env_cfg.py")
    m = re.search(r'pour_action_mode\s*:\s*str\s*=\s*"(\w+)"', cfg)
    assert m and m.group(1) == "b_trajectory", "pour_action_mode=b_trajectory 기본값"
    assert re.search(r'beta_action_index\s*:\s*int\s*=\s*\d', cfg), "beta_action_index 없음"


def test_beta_setpoint_wiring() -> None:
    """[β 수정] β=tilt setpoint: delta[:,4](tilt_toward)를 목표 tilt_amount까지 피드백 구동.

    구 j5 하드구동(_rbeta_arm_lookup post-IK overwrite)은 제거 — pour-point를 깸.
    """
    env = _read("pour_right_env.py")
    cfg = _read("pour_right_env_cfg.py")
    # β-setpoint cfg
    assert re.search(r'beta_target_tilt_amount\s*:\s*float', cfg), "beta_target_tilt_amount 없음"
    assert re.search(r'beta_tilt_kp\s*:\s*float', cfg), "beta_tilt_kp 없음"
    assert re.search(r'beta_tilt_max_step\s*:\s*float', cfg), "beta_tilt_max_step 없음"
    # env: β→tilt_toward setpoint 구동
    assert "_beta_cmd" in env and 'pour_action_mode == "b_trajectory"' in env, "b_trajectory 분기 없음"
    assert "self.cfg.beta_target_tilt_amount" in env, "β→목표 tilt_amount 매핑 없음"
    assert "self.cfg.beta_tilt_kp" in env and "delta[:, 4] = _tilt_step" in env, "tilt setpoint 구동 없음"
    # 구 j5 post-IK override는 제거되어야 (pour-point 보존)
    assert "_arm_q[_ready, 4] = _rbeta" not in env, "j5 post-IK override가 남아있음(pour-point 깸)"


def test_b_light_orient_release_wiring() -> None:
    """[B-light] orientation 풀기 + 주둥이 hold + β→cspace j5 구동."""
    cfg = _read("pour_right_env_cfg.py")
    env = _read("pour_right_env.py")
    assert re.search(r'pour_orient_release\s*:\s*bool\s*=\s*True', cfg), "pour_orient_release flag 없음"
    # orientation 풀기: palm 방향 target=현재
    assert "self.cfg.pour_orient_release" in env, "B-light 분기 없음"
    assert "_spout_offset_body" in env, "주둥이 body offset 동결 버퍼 없음"
    assert "quat_apply_inverse" in env, "body frame offset 계산 없음"
    assert "_R_cur_xyzw" in env, "orientation=현재(released) 계산 없음"
    # tilt 단계(ready)만 풀기 — approach는 조준
    assert "torch.where(_rm, _R_cur_xyzw, palm_pose[:, 3:7])" in env, "orientation 풀기가 ready 게이트 아님"
    # β→cspace j5: j5 baseline = β·demo_j5
    # [B-full] j5를 β무관 full demo로 강제 (v5는 β-graded)
    assert "_baseline_arm[_ready_pour, 4] = self._demo_pour_arm_pose[4]" in env, "B-full forced j5 없음"
