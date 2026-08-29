# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GraspADR 물리 DR 확장 계약 (08.16 grasp_v2 이식).

질량/마찰이 ADR increment 에 따라 중립(초기) → terminal 로 선형 확장되는지 검증.
Isaac 의존 없이 EventManager 를 최소 스텁으로 대체한다.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ADR_PATH = Path(__file__).resolve().parents[1] / "grasp_adr.py"
_spec = importlib.util.spec_from_file_location("_grasp_adr_v1", _ADR_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GraspADR = _mod.GraspADR


class _TermStub:
    def __init__(self, params: dict):
        self.params = params


class _EventManagerStub:
    """get_term_cfg 만 제공하는 EventManager 대체물."""

    def __init__(self, terms: dict):
        self._terms = {k: _TermStub(v) for k, v in terms.items()}

    def get_term_cfg(self, name: str) -> _TermStub:
        return self._terms[name]


def _make(num_increments: int = 10):
    em = _EventManagerStub({
        "object_scale_mass": {"mass_distribution_params": (1.0, 1.0)},
        "object_physics_material": {"dynamic_friction_range": (1.0, 1.0)},
    })
    adr = GraspADR(
        custom_cfg={"spawn": {"xy_range": (0.02, 0.08)}},
        num_increments=num_increments,
        increment_interval=1,
        trigger_threshold=0.5,
        event_manager=em,
        physics_cfg={
            "object_scale_mass": {"mass_distribution_params": (0.5, 3.0)},
            "object_physics_material": {"dynamic_friction_range": (0.3, 1.0)},
        },
    )
    return adr, em


def _mass_range(em):
    return em.get_term_cfg("object_scale_mass").params["mass_distribution_params"]


def test_starts_neutral_before_any_increment():
    """ADR 0 = 중립 범위(원 질량/마찰) — 커리큘럼 시작점."""
    _, em = _make()

    assert _mass_range(em) == (1.0, 1.0)
    assert em.get_term_cfg("object_physics_material").params["dynamic_friction_range"] == (1.0, 1.0)


def test_reaches_terminal_range_at_max_increment():
    """만렙에서 정확히 terminal 범위에 도달한다."""
    adr, em = _make(num_increments=10)

    adr.set_increment(10)

    lo, hi = _mass_range(em)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(3.0)


def test_expands_linearly_at_half_progress():
    """중간 진행도에서 initial↔terminal 의 선형 보간값을 갖는다."""
    adr, em = _make(num_increments=10)

    adr.set_increment(5)

    lo, hi = _mass_range(em)
    assert lo == pytest.approx(1.0 + (0.5 - 1.0) * 0.5)   # 0.75
    assert hi == pytest.approx(1.0 + (3.0 - 1.0) * 0.5)   # 2.0


def test_maybe_increment_expands_physics_when_metric_passes():
    """interval 경과 후 성능 게이트를 넘으면 increment 와 함께 물리 범위가 넓어진다."""
    adr, em = _make(num_increments=10)  # interval=1

    assert adr.maybe_increment(0.9) is False   # 아직 interval 미경과
    assert adr.maybe_increment(0.9) is True

    lo, hi = _mass_range(em)
    assert lo < 1.0 and hi > 1.0


def test_metric_below_threshold_keeps_ranges_frozen():
    """성능이 임계 미만이면 난이도를 올리지 않는다(붕괴 방지 게이트)."""
    adr, em = _make(num_increments=10)

    for _ in range(5):
        assert adr.maybe_increment(0.1) is False
    assert _mass_range(em) == (1.0, 1.0)


def test_recovers_immediately_after_low_metric_period():
    """임계 미만 구간을 지나 회복하면 다음 스텝에 바로 증분한다.

    구 구현(고정 주기 `% interval`)은 검사 시점에 메트릭이 낮으면 interval 을 통째로
    더 기다렸다. 순간 메트릭에서는 증분 타이밍이 운에 좌우되므로 이 동작을 제거했다.
    """
    adr, em = _make(num_increments=10)  # interval=1

    adr.maybe_increment(0.9)            # interval 충족용 첫 호출
    for _ in range(3):                  # 성능 저하 구간 — 대기만 함
        assert adr.maybe_increment(0.0) is False
    assert adr.maybe_increment(0.9) is True   # 회복 즉시 증분(추가 대기 없음)


def test_interval_enforced_between_consecutive_increments():
    """증분 사이에는 interval 이 강제된다(연속 폭주 방지)."""
    adr, _ = _make(num_increments=10)
    adr.increment_interval = 3

    fired = [adr.maybe_increment(0.9) for _ in range(8)]

    # 3스텝마다 한 번씩만 True
    assert fired == [False, False, False, True, False, False, False, True]


def test_works_without_event_manager():
    """event_manager 미주입(구 동작)에서도 custom_cfg 보간은 그대로 동작."""
    adr = GraspADR(
        custom_cfg={"object_wrench": {"max_linear_accel": (0.0, 15.0)}},
        num_increments=10,
        increment_interval=1,
        trigger_threshold=0.5,
    )

    adr.set_increment(10)

    assert adr.get_param("object_wrench", "max_linear_accel") == pytest.approx(15.0)
