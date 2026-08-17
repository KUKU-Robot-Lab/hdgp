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

"""Phase 4 질량 커리큘럼(deform_water) 계약 테스트.

isaaclab 없이 검증하기 위해 GraspRightEnv의 헬퍼 3종을 순수 로직으로 재현하지 않고,
실제 메서드를 언바운드로 끌어와 최소 스텁(self)에 바인딩해 호출한다. 따라서 env.py의
로직이 바뀌면 이 테스트가 같이 깨진다(계약 테스트로서 의도된 동작).
"""

from types import SimpleNamespace

import pytest

from openarm.tesollo.right.grasp_adapt.grasp_adr import GraspADR


# ---------------------------------------------------------------------------
# env.py 헬퍼를 isaaclab 임포트 없이 가져오기 위한 소스 추출
# ---------------------------------------------------------------------------
def _load_mass_helpers():
    """grasp_right_env.py에서 질량 커리큘럼 헬퍼 4종만 추출해 실행한다."""
    import ast
    import pathlib

    src_path = (
        pathlib.Path(__file__).resolve().parents[1] / "grasp_right_env.py"
    )
    tree = ast.parse(src_path.read_text(encoding="utf-8"))

    wanted = {
        "_mass_adr_increment",
        "_mass_shift_active",
        "_resolve_reset_bead_cap",
        "_resolve_shift_target_count",
    }
    funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {}
            exec(compile(mod, "<helpers>", "exec"), ns)  # noqa: S102
            funcs[node.name] = ns[node.name]
    missing = wanted - set(funcs)
    assert not missing, f"env.py에서 헬퍼를 찾지 못함: {missing}"
    return funcs


HELPERS = _load_mass_helpers()


def _make_env(*, use_mass_adr, increment, adr=None, **cfg_over):
    """헬퍼 호출용 최소 self 스텁."""
    cfg = SimpleNamespace(
        use_mass_adr=use_mass_adr,
        bead_count_min=0,
        bead_count_max=30,
        num_beads=30,
        mass_shift_target_bead_count=30,
        mass_shift_adr_start=25,
        mass_shift_reset_bead_cap=10,
    )
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    if adr is None and use_mass_adr:
        adr = GraspADR(
            custom_cfg={
                "mass": {
                    "bead_count_max": (0.0, 30.0),
                    "shift_target_count": (0.0, 30.0),
                }
            },
            num_increments=50,
        )
    if adr is not None:
        adr.set_increment(increment)
    env = SimpleNamespace(cfg=cfg, grasp_adr=adr)
    # 헬퍼끼리 self로 상호 호출하므로(예: _resolve_reset_bead_cap → _mass_shift_active)
    # 스텁에 바인딩해 둔다.
    for name, fn in HELPERS.items():
        setattr(env, name, fn.__get__(env, type(env)))
    return env


def _call(name, env):
    return HELPERS[name](env)


# ---------------------------------------------------------------------------
# 기존 태스크 무영향 (회귀 방지)
# ---------------------------------------------------------------------------
def test_mass_adr_off_keeps_static_bead_cap():
    """use_mass_adr=False면 cfg 고정값을 그대로 쓴다(기존 태스크 동작 불변)."""
    env = _make_env(use_mass_adr=False, increment=0, adr=None)
    assert _call("_resolve_reset_bead_cap", env) == 30


def test_mass_adr_off_keeps_shift_always_active():
    """use_mass_adr=False면 shift는 항상 활성 — 기존 massshift 태스크 동작 유지."""
    env = _make_env(use_mass_adr=False, increment=0, adr=None)
    assert _call("_mass_shift_active", env) is True


def test_mass_adr_off_uses_fixed_shift_target():
    env = _make_env(use_mass_adr=False, increment=0, adr=None)
    assert _call("_resolve_shift_target_count", env) == 30


# ---------------------------------------------------------------------------
# 2단계 게이팅 — 전반부는 정적 수위만
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("increment", [0, 10, 24])
def test_shift_inactive_before_adr_start(increment):
    """increment < mass_shift_adr_start(25)면 동적 물 추가는 꺼져 있다."""
    env = _make_env(use_mass_adr=True, increment=increment)
    assert _call("_mass_shift_active", env) is False


@pytest.mark.parametrize("increment", [25, 40, 50])
def test_shift_active_from_adr_start(increment):
    env = _make_env(use_mass_adr=True, increment=increment)
    assert _call("_mass_shift_active", env) is True


def test_static_water_ramps_up_before_gate():
    """전반부에는 정적 수위가 0에서 단조 증가한다."""
    caps = [
        _call("_resolve_reset_bead_cap", _make_env(use_mass_adr=True, increment=i))
        for i in (0, 6, 12, 24)
    ]
    assert caps[0] == 0, "increment 0에서는 빈 컵으로 시작해야 한다"
    assert caps == sorted(caps), f"정적 수위가 단조 증가하지 않음: {caps}"
    assert caps[-1] > caps[0]


def test_reset_becomes_light_when_shift_activates():
    """동적 shift가 켜지면 리셋은 가벼운 컵으로 바뀐다(들고 나서 차오름 성립)."""
    before = _call("_resolve_reset_bead_cap", _make_env(use_mass_adr=True, increment=24))
    after = _call("_resolve_reset_bead_cap", _make_env(use_mass_adr=True, increment=25))
    assert after == 10
    assert after < before, "게이트 통과 후 리셋 수위가 낮아져야 한다"


def test_shift_target_never_below_reset_cap():
    """추가 목표가 리셋 수위보다 낮으면 무게가 줄어드는 셈 → 하한을 건다."""
    for inc in (25, 30, 40, 50):
        env = _make_env(use_mass_adr=True, increment=inc)
        target = _call("_resolve_shift_target_count", env)
        assert target >= 10, f"increment={inc}에서 target={target} < reset_cap"


def test_shift_target_ramps_to_full():
    """후반부에는 추가 목표가 가득(30)까지 오른다."""
    assert _call(
        "_resolve_shift_target_count", _make_env(use_mass_adr=True, increment=50)
    ) == 30


# ---------------------------------------------------------------------------
# 폴백 — "mass" 그룹이 없는 커스텀 ADR 설정
# ---------------------------------------------------------------------------
def test_missing_mass_group_falls_back_to_cfg():
    """adr_custom_cfg에 mass 그룹이 없어도 KeyError 없이 고정값으로 폴백한다."""
    adr = GraspADR(custom_cfg={"spawn": {"object_spawn_xy_range": (0.01, 0.06)}})
    env = _make_env(use_mass_adr=True, increment=10, adr=adr)
    assert _call("_resolve_reset_bead_cap", env) == 30
    assert _call("_resolve_shift_target_count", env) == 30


def test_no_adr_object_falls_back():
    """ADR 비활성(enable_adr=False)이면 grasp_adr가 None — 고정값 폴백."""
    env = _make_env(use_mass_adr=True, increment=0, adr=None)
    env.grasp_adr = None
    assert _call("_mass_adr_increment", env) == 0
    assert _call("_resolve_reset_bead_cap", env) == 30
