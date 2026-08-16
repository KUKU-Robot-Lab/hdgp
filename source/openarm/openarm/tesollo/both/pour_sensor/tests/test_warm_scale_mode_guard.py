# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0
"""warm 파지 재사용 모드 계약 테스트 (정적, Isaac 불필요).

배경(08.16): source_cup_scale_set 활성 시 warm 파지를 스케일 컵에 맞추는 방법이 두 가지다.
  - spec-matched: grasp_v1이 그 스케일 컵에서 실제로 잡은 파지를 그대로 사용 (정공법)
  - radial-fix : nominal 파지를 컵 반경 차만큼 밀어 근사 (미태깅 구캐시용 fallback)
둘을 동시에 적용하면 이미 맞는 파지를 다시 밀어내 어긋난다(이중보정). 또한 rollout
수집분도 per-env 스케일 컵에서 모은 것이라 보정 대상이 아니다.
이 테스트는 보정이 fallback 전용으로만 걸리도록 소스 계약을 고정한다.
"""

from pathlib import Path

_ENV_PY = Path(__file__).resolve().parents[1] / "pour_right_env.py"
_CFG_PY = Path(__file__).resolve().parents[1] / "pour_right_env_cfg.py"


def _env_source() -> str:
    return _ENV_PY.read_text(encoding="utf-8")


def test_radial_fix_is_flag_gated_not_scale_set_gated():
    """보정 가드가 scale_set 활성 여부(_src_spec_env)로 걸리면 안 된다 — 회귀 방지 핵심."""
    src = _env_source()
    reset_fn = src.split("def _reset_from_warmstart_cache")[1].split("\n    def ")[0]
    assert "if self._warm_radial_fix_active:" in reset_fn, (
        "반경 기하보정은 _warm_radial_fix_active 플래그로만 게이팅되어야 한다."
    )
    assert "if self._src_spec_env is not None:" not in reset_fn, (
        "scale_set 활성만으로 보정을 걸면 spec-matched/rollout 경로에서 이중보정이 발생한다."
    )


def test_radial_fix_flag_lifecycle():
    """플래그가 False 초기화 → 미태깅 분기에서만 True → 리셋에서 참조되는 3지점을 갖는다."""
    src = _env_source()
    assert "self._warm_radial_fix_active: bool = False" in src, "__init__ 기본 False 초기화 필요"
    assert src.count("self._warm_radial_fix_active = True") == 1, (
        "True 대입은 미태깅 캐시 분기 1곳뿐이어야 한다."
    )
    # spec 풀 구성(정공법) 직후에 보정을 켜면 안 된다
    after_pools = src.split("self._warm_spec_pools = pools")[1].split("def ")[0]
    assert "_warm_radial_fix_active = True" not in after_pools, (
        "spec-matched 경로에서 보정을 켜면 안 된다."
    )


def test_warm_mode_is_observable():
    """런 사후에 어느 warm 경로로 학습했는지 TB로 판정 가능해야 한다."""
    src = _env_source()
    assert "self._warm_mode_code" in src
    assert 'extras["log/warm_mode"]' in src, "warm 모드를 TB 스칼라로 남겨야 사후 판정이 가능하다."


def test_source_scale_set_has_spec_map_and_fixed_mass():
    """스케일별 실파지 매칭과 질량 정합 cfg가 함께 존재해야 한다."""
    cfg = _CFG_PY.read_text(encoding="utf-8")
    assert "source_cup_scale_set" in cfg
    assert "source_warm_spec_map" in cfg
    assert "source_cup_fixed_mass" in cfg
