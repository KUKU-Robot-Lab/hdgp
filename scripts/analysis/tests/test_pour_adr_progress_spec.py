# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0
"""ADR 초기 레벨 고정(cfg 스칼라 → 카운터) 검증.

체크포인트 재개마다 쓰는 경로라 조용히 틀리면 학습 며칠을 날린다.
isaaclab 없이 돌도록 pour_adr 모듈만 직접 로드한다.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest

_MOD = (Path(__file__).resolve().parents[3]
        / "source/openarm/openarm/tesollo/right/pour_sensor/pour_adr.py")
_spec = importlib.util.spec_from_file_location("pour_adr_under_test", _MOD)
pour_adr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pour_adr)

collect = pour_adr.collect_adr_progress_pins


@dataclass
class _Cfg:
    """cfg 의 관련 필드만 흉내낸다 (기본값 = 고정 안 함)."""
    adr_initial_progress_success: float = -1.0
    adr_initial_progress_outcome: float = -1.0
    adr_initial_progress_noise: float = -1.0
    adr_initial_progress_spill: float = -1.0


def test_defaults_pin_nothing():
    assert collect(_Cfg()) == {}


def test_collects_only_non_negative_keys():
    cfg = _Cfg(adr_initial_progress_success=1.0, adr_initial_progress_noise=0.15)

    assert collect(cfg) == {"success": 1.0, "noise": 0.15}


def test_zero_is_a_real_pin_not_a_skip():
    """0.0 = '0 레벨로 고정'. 음수 = '건드리지 않음'. 둘은 달라야 한다."""
    assert collect(_Cfg(adr_initial_progress_outcome=0.0)) == {"outcome": 0.0}


def test_progress_above_one_raises_instead_of_clamping():
    with pytest.raises(ValueError, match="0~1"):
        collect(_Cfg(adr_initial_progress_outcome=8.0))


def test_missing_field_is_treated_as_unpinned():
    class _Bare:
        pass

    assert collect(_Bare()) == {}


def test_full_progress_pins_adr_to_max_increment():
    adr = pour_adr.PourADR(custom_cfg={"outcome": {"w": (0.0, 50.0)}}, num_increments=8)
    frac = collect(_Cfg(adr_initial_progress_outcome=1.0))["outcome"]

    adr.set_increment(int(round(adr.num_increments * frac)))

    assert adr.increment_counter == 8
    assert adr.get_param("outcome", "w") == 50.0


def test_fractional_progress_rounds_to_nearest_increment():
    adr = pour_adr.PourADR(custom_cfg={"outcome": {"w": (0.0, 8.0)}}, num_increments=8)
    frac = collect(_Cfg(adr_initial_progress_outcome=0.625))["outcome"]

    adr.set_increment(int(round(adr.num_increments * frac)))

    assert adr.increment_counter == 5
    assert adr.get_param("outcome", "w") == 5.0
