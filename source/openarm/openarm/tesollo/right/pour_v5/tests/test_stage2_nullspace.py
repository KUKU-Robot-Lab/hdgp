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


def test_pour_phase_clamp_cfg() -> None:
    """[stage3] phase별 차등 클램프 cfg: j5 상한 0(거꾸로 roll 금지), j6 [-0.2,0.2](leak 차단)."""
    cfg = _read("pour_right_env_cfg.py")
    assert re.search(r'pour_phase_clamp_enable\s*:\s*bool\s*=\s*True', cfg), "phase clamp flag 없음"
    # j5(idx4) 상한 0.0, j6(idx5) [-0.2,0.2]
    mlo = re.search(r'pour_phase_arm_lo\s*:\s*tuple\s*=\s*\(([^)]*)\)', cfg)
    mhi = re.search(r'pour_phase_arm_hi\s*:\s*tuple\s*=\s*\(([^)]*)\)', cfg)
    assert mlo and mhi, "pour_phase_arm_lo/hi 없음"
    lo = [float(x) for x in mlo.group(1).split(",")]
    hi = [float(x) for x in mhi.group(1).split(",")]
    assert len(lo) == 7 and len(hi) == 7, "7-DOF band 아님"
    assert hi[4] == 0.0, "j5 상한=0(거꾸로 roll 금지)이어야"
    assert lo[5] <= -0.25 and hi[5] >= 0.30, "j6 밴드=demo 자연범위(±0.3, 문서검증). 구 [-0.2,0.2] 폐기"


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


def test_b_trajectory_env_wiring() -> None:
    """env: R(β) import·lookup·β구동·j5 하드구동."""
    env = _read("pour_right_env.py")
    assert "RBETA_ARM" in env and "_rbeta_arm_lookup" in env, "R(β) lookup 미구현"
    assert "_beta_cmd" in env and 'pour_action_mode == "b_trajectory"' in env, "b_trajectory 분기 없음"
    assert "_rbeta_arm_lookup(self._beta_cmd)" in env, "j5 하드구동(R(β)) 없음"


def test_j6_band_natural_range() -> None:
    """j6 밴드는 demo 자연범위(±0.3) — 문서검증 후 Stage3 [-0.2,0.2] 폐기."""
    cfg = _read("pour_right_env_cfg.py")
    mlo = re.search(r'pour_phase_arm_lo\s*:\s*tuple\s*=\s*\(([^)]*)\)', cfg)
    mhi = re.search(r'pour_phase_arm_hi\s*:\s*tuple\s*=\s*\(([^)]*)\)', cfg)
    lo = [float(x) for x in mlo.group(1).split(",")]
    hi = [float(x) for x in mhi.group(1).split(",")]
    assert lo[5] <= -0.25 and hi[5] >= 0.30, "j6 밴드가 demo 자연범위(±0.3) 반영해야"
