# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0
"""source 컵 질량 정합 계약 테스트 (정적, Isaac 불필요).

배경(08.16): grasp_v1은 cup_big 4종 질량을 스케일과 무관하게 0.134kg로 고정한다.
pour가 MultiAsset으로 스케일 컵을 스폰하면 USD density 파생 질량이 s³로 변해
(0.082/0.134/0.204/0.294) 수집↔소비 force-ratio가 최대 2.2배 어긋난다.
이 테스트는 두 프로젝트의 질량 기준이 코드 레벨에서 계속 일치하도록 잠근다.

동시에, scale_set을 쓰지 않는 **기본 경로의 물리는 건드리지 않았음**을 보장한다
(M4 등 기존 체크포인트와의 대조 무결성).
"""

import re
from pathlib import Path

_POUR_CFG = Path(__file__).resolve().parents[1] / "pour_right_env_cfg.py"
_POUR_ENV = Path(__file__).resolve().parents[1] / "pour_right_env.py"
_GRASP_CFG = (
    Path(__file__).resolve().parents[4] / "tesollo" / "right" / "grasp_v1" / "grasp_right_env_cfg.py"
)


def test_fixed_mass_matches_grasp_v1_cup_mass():
    """pour의 고정 질량 기본값이 grasp_v1 cup_big 질량과 같아야 수집↔소비가 정합한다."""
    pour_cfg = _POUR_CFG.read_text(encoding="utf-8")
    m = re.search(r"source_cup_fixed_mass:\s*float\s*\|\s*None\s*=\s*([0-9.]+)", pour_cfg)
    assert m, "source_cup_fixed_mass 기본값을 찾지 못했다."
    pour_mass = float(m.group(1))

    grasp_cfg = _GRASP_CFG.read_text(encoding="utf-8")
    # 질량은 리터럴로도, 명명 상수(_BASE_OBJECT_MASS)로도 쓰인다 — 둘 다 수집한다.
    grasp_masses = {float(v) for v in re.findall(r'"mass":\s*([0-9.]+)', grasp_cfg)}
    for name in set(re.findall(r'"mass":\s*(_[A-Z_]+)', grasp_cfg)):
        m_const = re.search(rf"^{name}\s*:\s*float\s*=\s*([0-9.]+)", grasp_cfg, re.M)
        if m_const:
            grasp_masses.add(float(m_const.group(1)))
    assert grasp_masses, "grasp_v1 물체 스펙에서 mass 값을 찾지 못했다."
    assert pour_mass in grasp_masses, (
        f"pour 고정 질량({pour_mass})이 grasp_v1 컵 질량{sorted(grasp_masses)}과 다르다 — "
        "warm 파지의 force-ratio가 어긋난다."
    )


def test_fixed_mass_applied_only_in_scale_set_branch():
    """기본(단일 컵) 경로의 질량은 손대지 않아야 기존 런과 물리가 동일하다."""
    env_src = _POUR_ENV.read_text(encoding="utf-8")
    assert "source_cup_fixed_mass" in env_src
    # scale_set 분기 안에서만 MassPropertiesCfg를 주입한다
    branch = env_src.split("_src_set = tuple(getattr(cfg, \"source_cup_scale_set\"")[1].split(
        "super().__init__"
    )[0]
    assert "MassPropertiesCfg" in branch, "scale_set 분기에서 질량을 고정해야 한다."

    pour_cfg = _POUR_CFG.read_text(encoding="utf-8")
    cup_cfg_block = pour_cfg.split("cup_cfg: RigidObjectCfg = RigidObjectCfg(")[1].split(
        "left_target_cup_cfg"
    )[0]
    assert "mass_props" not in cup_cfg_block, (
        "기본 cup_cfg에 질량을 명시하면 기존 학습(M4 등)의 물리가 바뀐다."
    )


def test_done_reason_logging_present():
    """조기종료 원인 규명을 위한 종료 사유별 로깅이 있어야 한다(08.16 분석에서 결여 확인)."""
    env_src = _POUR_ENV.read_text(encoding="utf-8")
    assert 'extras[f"done/{_name}"]' in env_src
    for reason in ("out_x", "dropped_by_force", "source_drained", "bead_fallen"):
        assert f'("{reason}"' in env_src, f"종료 사유 {reason} 로깅 누락"
