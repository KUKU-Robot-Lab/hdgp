# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0
"""PourADR 단계형 학습 인계(initial_increment) 단위 테스트.

배경(08.16): 컵 스케일 DR 학습에서 outcome_adr이 pose_success 게이트(0.80)를 한 번도
넘지 못해 weight_pour_bead가 6,000+ iter 내내 0 = pour 보상 부재로 학습이 불가능했다.
체크포인트에는 ADR 카운터가 저장되지 않아 fine-tune도 같은 데드락을 재현했다.
단계형 커리큘럼에서 이전 단계 레벨을 인계하는 것이 이 테스트의 대상이다.

PourADR은 torch/Isaac 의존이 없는 순수 파이썬이라 직접 로드해 검증한다.
"""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "pour_adr.py"


def _load_pour_adr():
    spec = importlib.util.spec_from_file_location("_pour_adr_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PourADR


PourADR = _load_pour_adr()

_CFG = {"outcome": {"weight_pour_bead": (0.0, 50.0)}}


def test_default_is_zero_unchanged():
    """기본값은 0 — 기존 학습 동작이 그대로여야 한다."""
    adr = PourADR(custom_cfg=_CFG, num_increments=4)
    assert adr.increment_counter == 0
    assert adr.progress == 0.0
    assert adr.get_param("outcome", "weight_pour_bead") == 0.0


def test_initial_increment_sets_level_and_param():
    """인계 레벨이 progress와 보간 파라미터에 즉시 반영된다."""
    adr = PourADR(custom_cfg=_CFG, num_increments=4, initial_increment=4)
    assert adr.increment_counter == 4
    assert adr.progress == 1.0
    # max 레벨이면 pour 보상 가중치가 처음부터 최종값 — 데드락 회피의 핵심
    assert adr.get_param("outcome", "weight_pour_bead") == 50.0

    half = PourADR(custom_cfg=_CFG, num_increments=4, initial_increment=2)
    assert half.progress == 0.5
    assert half.get_param("outcome", "weight_pour_bead") == pytest.approx(25.0)


def test_initial_increment_is_clamped():
    """과대/음수 입력은 [0, num_increments]로 클램프된다."""
    assert PourADR(custom_cfg=_CFG, num_increments=4, initial_increment=99).increment_counter == 4
    assert PourADR(custom_cfg=_CFG, num_increments=4, initial_increment=-5).increment_counter == 0


def test_increment_continues_from_initial_level():
    """인계 후에도 maybe_increment가 그 레벨에서 이어서 오른다(타이머는 새로 시작)."""
    adr = PourADR(
        custom_cfg=_CFG, num_increments=4, increment_interval=2,
        trigger_threshold=0.5, initial_increment=2,
    )
    adr.maybe_increment(0.9)          # step 1 — 간격 미도달
    assert adr.increment_counter == 2
    adr.maybe_increment(0.9)          # step 2 — 간격 도달, 임계 통과
    assert adr.increment_counter == 3


def test_below_threshold_does_not_increment():
    """임계 미달이면 인계 레벨을 유지만 하고 오르지 않는다."""
    adr = PourADR(
        custom_cfg=_CFG, num_increments=4, increment_interval=1,
        trigger_threshold=0.8, initial_increment=1,
    )
    for _ in range(5):
        adr.maybe_increment(0.1)
    assert adr.increment_counter == 1
