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

"""grasp_sensor_v2 정적 계약 — Isaac 불필요.

각 계약은 v1 이 실측으로 태운 함정 하나, 또는 v2 설계의 근거식 하나에 대응한다.
통과가 목적이 아니라 **재발 방지**가 목적이다.
"""

import math
import os
from pathlib import Path

from openarm.gripper.left.grasp_sensor import grasp_left_preset as V1
from openarm.gripper.left.grasp_sensor_v2 import v2_preset as P

_PKG = Path(__file__).resolve().parents[1]
_EPS = 1e-9


def _src(name: str) -> str:
    return (_PKG / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Lee 의 거리 shaping — 스케일에 의미가 있다는 것이 이 설계의 전제다
# ---------------------------------------------------------------------------
def _d(dist: float, s: float, tau: float = 0.0) -> float:
    """`v2_stages.d_shape` 의 순수 파이썬 복제. Isaac 없이 성질을 검증하기 위함."""
    if tau > 0.0 and dist < tau:
        return 1.0
    q = math.tanh(dist * (P.D_SHAPE_K / s))
    return 1.0 - q * q


def test_d_shape_decays_to_five_percent_at_scale():
    """★`s` 의 정의: **보상이 0.05 로 떨어지는 거리**.

    v1 은 `1 − tanh(d/std)` 의 `std` 가 무엇을 뜻하는지 정의돼 있지 않아 감으로 골랐고
    두 번 실패했다(coarse 0.3→0.15 기각 t63 · fine 0.05→0.12 역행 t64).
    이 계약이 깨지면 `s` 를 그런 감각적 손잡이로 되돌린 것이다.
    """
    for s in (0.10, 0.15, 0.20, 0.30):
        assert abs(_d(s, s) - 0.05) < 1e-9, f"s={s} 에서 D 가 0.05 가 아니다"


def test_d_shape_is_one_at_zero_and_monotone():
    """0 에서 1, 그리고 단조 감소. 어긋나면 gradient 부호가 뒤집힌다."""
    assert abs(_d(0.0, 0.20) - 1.0) < _EPS
    prev = 1.0
    for mm in range(0, 400, 10):
        cur = _d(mm / 1000.0, 0.20)
        assert cur <= prev + _EPS, f"{mm} mm 에서 증가했다"
        prev = cur


def test_d_shape_tolerance_saturates():
    """`τ` 안에서는 1 — Lee 의 tolerance. 목표 반경 안에서 미세 진동을 벌하지 않는다."""
    assert _d(0.005, P.TRANSPORT_S, P.TRANSPORT_TAU) == 1.0
    assert _d(0.05, P.TRANSPORT_S, P.TRANSPORT_TAU) < 1.0


# ---------------------------------------------------------------------------
# ★★채점점 정렬 — v2 의 존재 이유 중 하나
# ---------------------------------------------------------------------------
def test_goal_box_is_shifted_by_measured_grasp_offset():
    """목표 상자는 v1 상자를 **실측 파지 오프셋만큼 평행이동**한 것이어야 한다.

    v1 결함: goal 보상은 TCP 로 재는데 합격은 컵으로 쟀다. t79 best 결정론 프로브
    (64env × 300 step) 리프트 후 실측 `컵 − TCP` = (−35, +4, +12) mm, 3D 37.2 mm.
    ⇒ TCP 가 목표에 완벽히 도달해도 컵은 37 mm 남아 합격 예산 57 mm 의 65% 를 먹었다.

    상자를 통째로 옮기면 컵이 새 상자에 놓일 때 TCP 는 **옛 상자(TCP 제약 IK 로 도달성이
    검증된 자리)** 에 있다 — 검증을 버리지 않고 채점만 정렬한다.
    """
    for i in range(3):
        assert abs(P.GOAL_POINT_V2[i] - (V1.GOAL_POINT[i] + P.GRASP_OFFSET_ROOT[i])) < _EPS
    assert P.GOAL_JITTER_V2 == V1.GOAL_JITTER, "상자 크기는 바꾸지 않는다 — 옮기기만 한다"


def test_grasp_offset_matches_probe_measurement():
    """오프셋 크기가 **v2 정책 실측 46.1 mm** 와 일치해야 한다 (R3, 08.29 갱신).

    `probe_v2_transport_diag.py` (64 env × 300 step, 결정론) 두 체크포인트 모두
    ‖컵−TCP‖ = 46.1 mm. z 만 유의하게 달랐다(+28~35 mm vs 구 값 +12 mm)라 z 만 고쳤다.

    ⚠ 이 값은 파지 자세에 의존한다. 자세가 크게 달라지면 프로브를 다시 돌려 갱신할 것 —
      갱신 없이 자세만 바뀌면 채점이 조용히 어긋난다.
    """
    mag = math.sqrt(sum(v * v for v in P.GRASP_OFFSET_ROOT))
    assert abs(mag - 0.0461) < 0.003, f"v2 실측 46.1 mm 와 어긋난다: {mag * 1000:.1f} mm"


def test_r3_only_changes_z():
    """R3 는 **z 만** 고친다 — x·y 는 실측 산포가 부호까지 갈려 유의하지 않았다.

    BEST (−33.9, +5.7, +27.7) mm · EP1700 (−24.4, −11.8, +35.3) mm.
    x 는 preset(−35)과 ±10 mm 안, y 는 부호가 갈린다. z 만 둘 다 구 값(+12)보다 크다.
    """
    assert P.GRASP_OFFSET_ROOT[:2] == P.GRASP_OFFSET_ROOT_V1[:2], "x·y 는 건드리지 않는다"
    assert P.GRASP_OFFSET_ROOT[2] > P.GRASP_OFFSET_ROOT_V1[2]
    assert 0.024 <= P.GRASP_OFFSET_ROOT[2] <= 0.036, "실측 +28~35 mm 범위 안"


# ---------------------------------------------------------------------------
# ★★08.29 라운드 3 — 계단을 연속·단조로 재구성한 것의 계약
# ---------------------------------------------------------------------------
# 이 절이 잠그는 것은 설계의 **세 원칙**이다:
#   ① 연속 — 문턱에서 v_k → 1 이고 v_{k+1} → 0
#   ② 단조 — 각 단계 안에서 목표로 갈수록 v_k 증가
#   ③ 고원 없음 — 각 단계 안 어디서도 ∂r/∂(진행) ≠ 0
# 라운드 1 의 R1 은 reward-audit 5 체크를 통과하고도 **정반대로** 작동했다(제자리 왕복이
# 순변위 0 을 만들어 점수를 벌었다). 그 오류는 보상식 격자 스캔만으로 잡힌다 —
# 아래 순수 파이썬 복제가 그 스캔이고, `probe_v2_reward_terrain.py` 가 같은 식을 쓴다.


def _smoothstep(x: float, far: float, near: float) -> float:
    """`v2_stages.smoothstep` 의 순수 파이썬 복제."""
    t = min(1.0, max(0.0, (x - far) / (near - far)))
    return t * t * (3.0 - 2.0 * t)


def _move_up(cup_z: float) -> float:
    t = (cup_z - V1.LIFT_RAMP_ZERO_Z) / (V1.MINIMAL_LIFT_HEIGHT - V1.LIFT_RAMP_ZERO_Z)
    return min(1.0, max(0.0, t))


def _reward(dist: float, cup_z: float, speed: float, upright: float = 1.0,
            r_close: float = 1.0, r_grasp: float = 0.9) -> tuple[float, int, float]:
    """`v2_stages.all_stages` + `v2_rewards._stage_index/_stage_value` 의 복제.

    ⚠ 원본과 어긋나면 이 절 전체가 거짓이 된다 —
      `test_reward_replica_matches_the_source` 가 식의 형태를 대조한다.
    """
    r_lift = r_close * _move_up(cup_z)
    r_transport = r_lift * _d(dist, P.TRANSPORT_S)
    ok3 = float(dist < P.SETTLE_RADIUS and r_close > P.STAGE3_GRASP_MIN)
    idx = 0
    for k, q in ((1, r_lift), (2, r_transport), (3, ok3)):
        if q > P.STAGE_THRESHOLD:
            idx = k
    v = (r_grasp, r_lift, r_transport,
         _smoothstep(speed, *P.P_STILL_BAND) * _smoothstep(upright, *P.P_UPRIGHT_BAND))[idx]
    return (idx + v) / P.N_STAGES, idx, v


# 스폰 → 목표 직선 경로. `GOAL_JITTER` 중심을 쓴다.
_SPAWN = (V1.CUP_SPAWN_X_CENTER, V1.CUP_SPAWN_Y_CENTER, V1.CUP_SPAWN_Z)
_GOAL = tuple(P.GOAL_POINT[i] + P.GRASP_OFFSET_ROOT[i] for i in range(3))


def _path(t: float) -> tuple[float, float]:
    p = [_SPAWN[i] + t * (_GOAL[i] - _SPAWN[i]) for i in range(3)]
    d = math.sqrt(sum((p[i] - _GOAL[i]) ** 2 for i in range(3)))
    return d, p[2]


def test_smoothstep_is_zero_at_far_and_one_at_near():
    """밴드 밖에서 **정확히** 0/1 이어야 한다 — 그래야 "조건을 만족했다"를 표현할 수 있고
    계단 값이 다음 문턱에서 정확히 1 이 된다(`d_shape` 는 꼬리가 남아 불가능하다)."""
    for far, near in (P.P_DIST_BAND, P.P_STILL_BAND, P.P_UPRIGHT_BAND):
        assert _smoothstep(far, far, near) == 0.0
        assert _smoothstep(near, far, near) == 1.0
        # 밴드 밖으로 더 나가도 포화
        assert _smoothstep(far + (far - near), far, near) == 0.0
        assert _smoothstep(near + (near - far), far, near) == 1.0
        mid = 0.5 * (far + near)
        assert 0.0 < _smoothstep(mid, far, near) < 1.0, "사이는 단조 보간"


def test_stage_boundaries_are_continuous():
    """★원칙 ① — 계단 경계의 점프가 작아야 한다.

    run 0 실측 점프: 0→1 +0.050 · 1→2 +0.099 · **2→3 +0.250**(계단 하나 통째).
    불연속은 value 함수가 계단을 맞춰야 한다는 뜻이고, 경계 근처 advantage 가
    "어느 쪽에 떨어졌는가"에 지배된다 = 시드 의존의 직접 원인이다.
    """
    prev, seen = None, {}
    N = 20000
    for i in range(N + 1):
        dist, z = _path(i / N)
        r, idx, _ = _reward(dist, z, speed=0.0)
        if prev is not None and idx != prev[1]:
            seen[(prev[1], idx)] = r - prev[0]
        prev = (r, idx)
    assert seen, "경로에서 단계 전이가 한 번도 안 일어났다 — 경로 설정이 틀렸다"
    # ★★라운드 5 — "완전 연속"은 **포기한 요구**다. 라운드 3 이 그걸 얻으려고 `v_1` 을
    #   문턱에서 1 로 정규화했다가 stage 1 천장을 0.426 → 0.500 으로 올렸고, 라운드 4 는
    #   `v_2` 를 거리만으로 깎아 신호 고갈(σ 1.3 · 결정론 파지 0.190)을 불렀다.
    #   실제로 필요한 불변식은 **하향 점프가 없을 것** 하나다 — 위쪽 점프는 전진에
    #   대한 보너스이지 결함이 아니다(baseline 은 0→1 +0.050 · 1→2 +0.099 로도
    #   결정론 도달 93.7% 를 낸다).
    for (a, b), jump in seen.items():
        assert jump > -_EPS, f"stage {a}→{b} 에서 보상이 떨어진다 ({jump:+.4f})"
    # 2→3 은 stage 3 바닥(0.75) ≥ stage 2 천장(0.75) 이라 구조적으로 하향 불가
    assert seen.get((2, 3), 0.0) >= -_EPS


def test_no_plateau_in_stage2_and_stage3():
    """★stage 2·3 안에 ∂r/∂(진행) = 0 인 구간이 없어야 한다.

    ★★**D1 진단 정정 (08.29).** 나는 "stage 1 안에 목표 거리가 없어 고원이고, 그게
      이송 실패의 원인"이라고 6 판을 진단했다. **틀렸다.** stage 1 의 값은 baseline 도
      `r_lift` 라 램프 포화 후 평탄한데, 그 baseline 이 **결정론 도달 93.7%** 를 낸다
      (컵–목표 최저 거리 중앙 28.1 mm). 고원은 병목이 아니었다 — stage 2 진입 문턱이
      스폰 거리 근처라 정책이 금방 stage 2 로 올라가기 때문이다.
      내가 `atgoal`(스텝 비율)을 도달률로 오독해 없는 병을 고치려 했던 것이다.

    ⇒ 요구는 **실제로 최적화가 일어나는 단계**로 좁힌다: stage 2(거리)와 stage 3
      (속도·직립). stage 0·1 은 다른 축(TCP 접근·상승)이 구동한다.
    """
    N = 400
    flat = 0
    prev = None
    for i in range(N + 1):
        dist, z = _path(i / N)
        r, idx, _ = _reward(dist, z, speed=0.0)
        # ⚠ stage 3 은 제외한다 — `v_3` 는 **설계상 거리에 무관**하다(속도·직립만 본다).
        #   합격 기준이 `dist < SETTLE_RADIUS` 라 반경 안에서 중심을 요구하지 않는다.
        #   그 구간의 gradient 는 아래에서 속도·직립 축으로 따로 검사한다.
        if idx != 2:
            prev = None
            continue
        if prev is not None and abs(r - prev) < 1e-9:
            flat += 1
        prev = r
    assert flat == 0, f"stage 2 경로에 평탄 구간 {flat} 개 — 거리 신호가 죽었다"
    # stage 3 안: 속도·직립 양쪽에 gradient 가 살아 있어야 한다
    z_goal = _path(1.0)[1]
    v_slow = _reward(0.03, z_goal, speed=0.01, upright=0.99)[2]
    v_fast = _reward(0.03, z_goal, speed=0.12, upright=0.99)[2]
    v_up = _reward(0.03, z_goal, speed=0.01, upright=0.995)[2]
    v_tilt = _reward(0.03, z_goal, speed=0.01, upright=0.975)[2]
    assert v_slow - v_fast > 0.1, f"stage 3 에 감속 gradient 가 없다 ({v_fast:.3f}→{v_slow:.3f})"
    assert v_up - v_tilt > 0.1, f"stage 3 에 직립 gradient 가 없다 ({v_tilt:.3f}→{v_up:.3f})"


def test_reward_is_monotone_toward_the_goal():
    """★원칙 ② — 목표로 갈수록 보상이 커져야 한다(정지 상태 기준)."""
    prev = None
    for i in range(0, 401):
        dist, z = _path(i / 400)
        r, idx, _ = _reward(dist, z, speed=0.0)
        if idx != 2:         # stage 0·1 은 접근·상승이, stage 3 은 속도·직립이 구동한다
            prev = None
            continue
        if prev is not None:
            assert r >= prev - _EPS, f"경로 {i/400:.3f} 에서 보상이 감소한다"
        prev = r


def test_stage1_stall_needs_the_cup_to_stay_near_spawn():
    """★`v_1` 에 목표 거리가 들어간 결과 — **stage 1 정체가 구조적으로 어렵다**.

    `r_lift` 를 최대로 올려도 `r_transport = r_lift·D(dist) > 0.1` 이 되는 순간 승급하므로,
    stage 1 에 머물려면 컵이 스폰 근처에 있어야 한다. 붕괴 정책이 주차하던 자리
    (146~170 mm)가 정확히 이 경계다.
    """
    z_top = V1.MINIMAL_LIFT_HEIGHT
    stall_min = None
    for i in range(3001):
        d = i * 0.0001
        _, idx, _ = _reward(d, z_top, speed=0.0)
        if idx <= 1:
            stall_min = d
            break
    spawn_dist = _path(0.0)[0]
    assert stall_min is not None
    assert stall_min < spawn_dist, (
        f"스폰({spawn_dist*1e3:.0f} mm)에서 이미 stage 2 여야 정체가 불가능하다 — "
        f"정체 최소 거리 {stall_min*1e3:.0f} mm")
    margin = spawn_dist - stall_min
    assert margin < 0.05, (
        f"여유 {margin*1e3:.0f} mm — 스폰 근처 정체 구간이 너무 넓다")


def test_stopping_short_of_the_goal_is_not_worth_it():
    """★reward-audit Check 2 의 계약화 — 라운드 1 R1 실패의 재발 방지.

    R1 은 순변위 `still` 을 넣었는데 **거리와 무관**해서 제자리 왕복이 점수를 벌었다
    (A_best 지령 반전 0.670 · 직진 효율 0.215 · 리미터 포화 0.925, baseline 보다 나쁨).
    새 설계는 `P_still` 을 `P_dist` 와 **곱한다** — 멀리서 멈추면 곱이 무너진다.
    """
    r_goal, _, _ = _reward(0.02, _path(1.0)[1], speed=0.0)
    for dmm in (150, 130, 110, 90, 70):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if _path(mid)[0] > dmm / 1000.0 else (lo, mid)
        dist, z = _path(0.5 * (lo + hi))
        r_stop, _, _ = _reward(dist, z, speed=0.0)
        assert r_stop < 0.95 * r_goal, (
            f"목표 밖 {dmm} mm 정지가 도달의 {r_stop/r_goal:.0%} — 해킹면이다")


def test_grasp_is_required_by_both_the_value_and_the_gate():
    """★reward-audit Check 3 의 REVISE — 목표 근처에서 **놓아도** 값이 유지되면 안 된다.

    초안의 `v_2 = P_dist·P_still·P_upright` 에는 파지 항이 없었다. `r_close` 를 값과
    승급 조건 **양쪽**에 넣어 막는다.
    """
    z_goal = _path(1.0)[1]
    held, _, _ = _reward(0.03, z_goal, speed=0.0, r_close=1.0)
    dropped, idx_d, _ = _reward(0.03, z_goal, speed=0.0, r_close=0.0)
    assert dropped < held, "파지를 놓았는데 보상이 그대로면 단계가 안 내려간 것"
    assert idx_d < 3, "파지 없이 stage 3 로 승급하면 승급 조건에 파지가 없는 것"
    src = _src("v2_stages.py")
    # v_2 = r_transport = r_close·move_up·D 라 파지가 이미 곱해져 있다(baseline 원본)
    assert "r_transport = r_lift * d_shape" in src
    assert "r_lift = r_close * move_up" in src
    assert "(r_close > P.STAGE3_GRASP_MIN)" in src, "승급 조건에 파지가 있어야 한다"


def test_grasp_gate_is_binary_so_the_top_boundary_stays_continuous():
    """`stage_close` 는 `grasp_ok` 면 정확히 1.0, 아니면 `0.5·align·enclose·closure` ≤ 0.5.

    ⇒ `r_close > 0.5` ⟺ `r_close == 1.0`. 이 성질 덕분에 stage 3 승급 순간
      `v_2 = 1·P_dist·P_still·P_upright = 1` 이 되어 2→3 경계가 **완전 연속**이다.
      문턱을 0.5 가 아닌 값으로 바꾸면 이 논증이 깨진다.
    """
    src = _src("v2_stages.py")
    assert "partial = 0.5 * align * enclose * closure" in src
    assert "torch.where(ok > 0.5, torch.ones_like(partial), partial)" in src
    assert P.STAGE3_GRASP_MIN == 0.5


def test_stage3_gate_is_reach_and_grasp_only():
    """★★라운드 4 — 승급 문턱에서 **속도·직립을 뺐다**(값으로 옮겼다).

    라운드 3 은 넷을 다 문턱에 넣었고 실측 `stage3_ok` 가 최대 0.0024 였다. 문턱에
    두면 "정지·직립을 못 하면 4 층 자체가 안 열려" 개선 방향을 못 배운다 — 구 설계
    D3(네 인자 곱)와 같은 절벽이다. 값에 두면 4 층 안에서 셋 다 gradient 가 산다.
    """
    z_goal = _path(1.0)[1]
    ok = dict(dist=0.03, cup_z=z_goal, speed=0.0, upright=1.0, r_close=1.0)
    assert _reward(**ok)[1] == 3
    # 속도·직립이 나빠도 **승급은 된다** (값이 낮아질 뿐)
    fast = _reward(**{**ok, "speed": 1.0}); tilt = _reward(**{**ok, "upright": 0.5})
    assert fast[1] == 3 and tilt[1] == 3, "속도·직립은 더 이상 승급을 막지 않는다"
    assert fast[2] < ok_v(ok) and tilt[2] < ok_v(ok), "대신 **값**이 낮아져야 한다"
    # 도달·파지는 여전히 필수
    for bad in (dict(dist=P.SETTLE_RADIUS + 0.001), dict(r_close=0.5)):
        assert _reward(**{**ok, **bad})[1] < 3, f"{bad} 인데 승급했다"
    src = _src("v2_stages.py")
    assert "def stage3_ok" in src and "def success_ok" in src


def ok_v(kw) -> float:
    return _reward(**kw)[2]


def test_success_definition_keeps_all_four_conditions():
    """★합격선은 안 낮춘다 — 지형만 부드럽게 하고 과제 정의는 그대로.

    승급 문턱에서 뺀 정지·직립은 `success_ok`(진단 전용)에 그대로 남는다. 지형과
    합격선을 같이 낮추면 "쉬워져서 올랐다"를 "고쳐서 올랐다"로 오독한다.
    """
    src = _src("v2_stages.py")
    body = src[src.index("def success_ok"):]
    # ★★08.31 라운드 15 — 합격에서 **속도 조건을 뺐다**(사용자 지시).
    #   진동체가 반환점마다 속도 0 을 지나 통과했다: H best 목표 안 평균 속도
    #   중앙 0.145 m/s(합격선 3배)인데 ⑤ 49.2%, 최장 연속 합격 1 스텝.
    #   합격은 **콜라이더 체류**로 잰다 — 체류는 hold 프리미엄이 센다.
    for cond in ("dist < P.SETTLE_RADIUS",
                 "upright_cos > P.STAGE3_UPRIGHT_MIN", "r_close > P.STAGE3_GRASP_MIN"):
        assert cond in body, f"합격 판정에서 {cond} 이 사라졌다"
    assert "speed < P.STAGE3_SPEED_MAX" not in body, \
        "속도 조건이 되살아났다 — 진동체가 반환점마다 통과한다"
    cfg = _src("v2_env_cfg.py")
    assert "diag_success" in cfg, "합격 지표가 등록돼야 한다"


def test_stage2_value_is_invariant_to_speed_and_upright():
    """★★★**stage 2 값은 속도·직립에 불변이어야 한다.**

    라운드 3 은 `v_2 = r_close·P_dist·P_still·P_upright` 였다. 설계 시 지형 스캔이
    `upright=1·speed=0` 인 **이상적 슬라이스**만 봐서 이송 유인이 +0.250 으로 읽혔는데,
    실측 운전점(`P_still` 0.43 · `P_upright` 0.71)에서는 곱이 0.31 배로 축소돼
    **+0.077** 이었다 — 구 설계(+0.207)의 2.7 배 약화. ep1053 까지 `atgoal` 정점
    0.046(baseline 0.285)으로 실측 확인했다.

    ⇒ 정책이 제어하기 어려운 인자로 **거리 신호를 곱해서 깎으면 안 된다.**
      이 계약이 그 재발을 막는다.
    """
    z = _path(0.5)[1]
    base = None
    for spd in (0.0, 0.05, 0.15, 0.30, 1.0):
        for up in (0.3, 0.6, 0.9, 1.0):
            r, idx, v = _reward(0.120, z, spd, up)
            assert idx == 2, "이 지점은 stage 2 여야 한다"
            if base is None:
                base = v
            assert abs(v - base) < _EPS, (
                f"stage 2 값이 speed={spd}·upright={up} 에 따라 변한다 ({v} vs {base})")


def test_transport_incentive_survives_the_realized_operating_point():
    """★라운드 3 실패의 정량 계약 — 실측 운전점에서 이송 유인이 살아 있어야 한다.

    유인 = r(목표 반경 도달) − r(stage 1 천장). 운전점 4 종(이상 · A 실측 · B 실측 ·
    A 최악 구간)에서 편차가 없어야 한다. 라운드 3 은 0.250 → 0.032 로 흔들렸다.
    """
    OP = [(1.000, 1.000), (0.433, 0.712), (0.762, 0.407), (0.473, 0.270)]

    def inv(y, far, near):
        lo, hi = min(far, near), max(far, near)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if (_smoothstep(mid, far, near) < y) == (
                _smoothstep(lo, far, near) < y) else (lo, mid)
        return 0.5 * (lo + hi)

    def at(d):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if _path(mid)[0] > d else (lo, mid)
        return _path(0.5 * (lo + hi))

    # stage 1 천장 = 경로 위 1→2 전이 직전
    prev, top = None, None
    for i in range(20001):
        dist, z = _path(i / 20000)
        r, idx, _ = _reward(dist, z, speed=0.0)
        if prev is not None and prev[1] == 1 and idx == 2:
            top = prev[0]; break
        prev = (r, idx)
    assert top is not None
    gains = []
    for ps, pu in OP:
        d50, z50 = at(P.SETTLE_RADIUS)
        gains.append(_reward(d50, z50, inv(ps, *P.P_STILL_BAND),
                             inv(pu, *P.P_UPRIGHT_BAND))[0] - top)
    assert max(gains) - min(gains) < _EPS, (
        f"운전점에 따라 이송 유인이 흔들린다: {[round(g,4) for g in gains]}")
    assert min(gains) > 0.20, f"이송 유인이 너무 작다: {min(gains):.4f}"


def test_stage_values_stay_in_unit_range():
    """모든 `v_k ∈ [0,1]` — 벗어나면 `(idx+v)/N` 이 계단 순서를 깬다."""
    for i in range(0, 201):
        dist, z = _path(i / 200)
        for spd in (0.0, 0.05, 0.25, 1.0):
            for up in (0.5, 0.95, 1.0):
                for rc in (0.0, 0.3, 1.0):
                    r, idx, v = _reward(dist, z, spd, up, rc)
                    assert -_EPS <= v <= 1.0 + _EPS, f"v={v} 가 범위 밖"
                    assert idx / P.N_STAGES - _EPS <= r <= (idx + 1) / P.N_STAGES + _EPS


def test_reward_replica_matches_the_source():
    """이 절의 순수 파이썬 복제가 원본 식과 같은지 **줄 단위 정확 일치**로 대조한다.

    복제가 낡으면 위 계약이 전부 거짓이 된다 — 실제로 `diag_r_*` 4 함수가 통째로
    사라졌는데 계약 31 건이 전부 통과한 사고가 있었다(런타임에서만 터졌다).

    ⚠ **부분 문자열로 대조하면 안 된다.** 초판이 `"v2 = r_close * p_dist" in src` 였는데,
      원본을 `v2 = r_close * p_dist * p_still * p_upright` 로 되돌려도 접두사라서
      통과했다(가드 반증에서 발각). 항이 **추가**되는 회귀를 못 잡는다.
    """
    lines = {ln.strip() for ln in _src("v2_stages.py").splitlines()}
    for frag in ("p_still = smoothstep(speed, *P.P_STILL_BAND)",
                 "p_upright = smoothstep(upright_cos, *P.P_UPRIGHT_BAND)",
                 "p_center = smoothstep(dist, *P.CENTER_BAND)",
                 "v3 = p_center * p_upright",
                 "return pos, (r_grasp, r_lift, r_transport, v3)",
                 "ok3 = stage3_ok(dist, r_close)"):
        assert frag in lines, f"원본의 그 줄이 정확히 일치하지 않는다: {frag!r}"


def test_speed_input_is_the_only_ab_difference():
    """arm A ↔ arm B 의 유일한 차이는 **속도 입력**이어야 한다.

    식이 갈리면(라운드 1 처럼 `if net_speed is None:` 로 두 벌을 두면) 단일 변수가
    아니게 된다. 속도는 한 번 계산해 `P_still` 과 승급 조건 **양쪽**에 같은 값이 간다.
    """
    src = _src("v2_stages.py")
    flat = " ".join(src.split())
    assert "speed = (torch.norm(obj.data.root_lin_vel_w, dim=1) " \
           "if net_speed is None else net_speed)" in flat, \
        "속도를 한 번만 계산해 두 곳에 같은 값을 보내야 한다"
    assert "return stages, stages" not in src, "run 0 분기가 남아 있으면 식이 두 벌이다"
    body = src[src.index("def all_stages"):]
    assert body.count("stage3_ok(") == 1, "승급 조건은 한 곳에서만 계산해야 한다"


def test_net_speed_window_outlives_a_vibration_cycle():
    """순변위 창은 왕복 **한 주기보다 길어야** 상쇄가 일어난다.

    실측 지령 방향 반전 비율이 최대 0.456 = 평균 주기 약 4 스텝. 창 20 스텝이면 5 주기를
    덮는다. 창이 짧으면 순변위도 진동을 정지로 오독하기 시작한다.
    """
    assert P.NET_SPEED_WINDOW >= 10, "왕복 한 주기(≈4 스텝)의 배수 이상이어야"


def test_still_band_has_gradient_at_the_pass_line():
    """합격선(50 mm/s) 부근에 gradient 가 살아 있어야 한다.

    ★라운드 5 로 전제가 바뀌었다. `P_still` 은 이제 `v_3` 의 인자라 **목표 반경 안**
      에서만 쓰인다. 그 구간의 실측 속도는 p10 0.0128 · 중앙 0.0399 · p90 0.1485 로,
      이송 중 속도(150~230 mm/s)가 아니다. far 를 그 위에 두면 밴드가 포화한다 —
      구 far 0.30 에서 운전점 v_3 = 0.982 였다.
    """
    far, near = P.P_STILL_BAND
    assert far >= 0.14, "목표 안 실측 p90(0.1485)을 밴드가 덮어야 한다"
    assert far <= 0.20, "far 가 너무 크면 운전점에서 포화한다(구 0.30 의 실패)"
    assert near <= P.STAGE3_SPEED_MAX, "합격선에서 만점 이상이어야"
    assert 0.0 < _smoothstep(0.0399, far, near) < 1.0, "운전점 중앙값에 gradient 가 있어야"


def test_net_speed_is_updated_once_per_step():
    """순변위 트래커는 스텝당 **한 번만** 전진해야 한다.

    진단 항이 `all_stages` 를 다시 부르므로, 캐시가 없으면 창 길이가 사실상 1/N 로 줄어
    순변위가 순간속도에 가까워진다 = arm B 가 arm A 가 된다.
    """
    st = _src("v2_stages.py")
    assert "common_step_counter" in st, "트래커가 스텝 카운터로 중복 갱신을 막아야 한다"
    rw = _src("v2_rewards.py")
    assert "_staircase(env)" in rw, "진단은 같은 인스턴스의 캐시를 써야 한다"
    assert "_cache_step" in rw


# ---------------------------------------------------------------------------
# 액션 파일 — v1 과 **공유**한다. v1 불변이 유일한 관심사다.
# ---------------------------------------------------------------------------
# ★A1(이송 국면 리미터 1/4)은 라운드 1 에서 실측 기각됐다("멈추기"는 고쳐졌으나
#   도달 env 가 60 → 29). v2 preset 의 상수는 제거했지만 액션 파일의 cfg 필드는
#   기본값 `None` 으로 **남겨 둔다** — v1 거동이 불변이고 나중에 다시 쓸 수 있다.
def test_fine_limiter_is_off_by_default_so_v1_is_untouched():
    """A1 은 cfg 기본값이 `None` 이라 **v1 트랙 거동이 바뀌지 않아야** 한다.

    v1(`grasp_sensor`)은 챔피언이자 폴백이라 동결이다. 액션 파일은 v1 과 **공유**하므로
    이 기본값이 v1 불변의 유일한 보증이다.
    """
    fa = (_PKG.parent / "grasp_sensor" / "grasp_left_fabric_action.py").read_text(
        encoding="utf-8")
    assert "fine_cmd_rate_limit: float | None = None" in fa
    assert "fine_latch_cup_z: float | None = None" in fa
    # 꺼져 있으면 상수 경로로 **즉시 반환**해야 한다(물체 조회조차 하지 않는다)
    assert "if fine is None or latch_z is None:" in fa
    assert "return base" in fa


# ---------------------------------------------------------------------------
# 실험 스위치 — 단일 변수 대조가 성립하는가
# ---------------------------------------------------------------------------
def test_still_switch_defaults_to_instantaneous_speed():
    """스위치 기본값 off = arm A(순간속도). 켜는 것은 환경변수의 명시적 행위여야 한다."""
    src = _src("v2_env_cfg.py")
    assert 'os.environ.get("HDGP_V2_STILL_NET", "0") == "1"' in src.replace("_os.", "os.")
    # 값이 cfg 필드에 박혀야 env.yaml dump 에 남는다(학습 전 확인 규칙 ②)
    assert "v2_still_net: bool" in src
    assert '"still_net": self.v2_still_net' in src, "cfg 필드가 보상 항으로 전달돼야 한다"


def test_rejected_round12_switches_are_gone():
    """라운드 1·2 의 기각된 처방이 코드에 남아 있으면 안 된다.

    남겨 두면 다음 사람이 "켜 보자"를 반복한다. 기각 사유는 `v2_preset.py` 하단 주석에
    글로 남긴다 — 코드가 아니라 기록으로.
    """
    def _code(src: str) -> str:
        """주석·docstring 을 뺀 코드만. 기각 사유는 **글로 남기는 것이 옳다** —
        주석까지 금지하면 왜 기각됐는지가 사라져 같은 실수를 반복한다."""
        return "\n".join(ln.split("#")[0] for ln in src.splitlines()
                          if not ln.lstrip().startswith("#"))

    cfg = _code(_src("v2_env_cfg.py"))
    pre = _code(_src("v2_preset.py"))
    st = _code(_src("v2_stages.py"))
    for gone in ("HDGP_V2_REWARD_FIX", "HDGP_V2_ACTION_FIX",
                 "v2_reward_fix", "v2_action_fix"):  # noqa: E501 — R3 스위치는 부활했다
        assert gone not in cfg, f"{gone} 이 아직 코드에 살아 있다"
    for gone in ("SPEED_GATE_NEAR", "SPEED_S_NEAR", "SPEED_S_FAR",
                 "FINE_CMD_RATE_LIMIT", "FINE_LATCH_CUP_Z"):
        assert f"\n{gone} =" not in pre, f"{gone} 상수가 아직 정의돼 있다"
        assert not hasattr(P, gone), f"P.{gone} 이 아직 임포트 가능하다"
    assert "def speed_scale_from_dist" not in st, "R2 함수가 아직 살아 있다"


def test_goal_offset_is_switchable_for_the_baseline_comparison():
    """★★라운드 4 — R3(목표 상자 이동)를 **다시 스위치로** 되돌렸다.

    라운드 3 에서 상시 적용한 결과 baseline(run0-s43)과 목표 z 가 18 mm 달라져
    (0.415~0.515 vs 0.397~0.497) 7 판 치 비교가 통째로 근사치가 됐다. `cupd`
    124 mm vs 89 mm 격차의 최대 절반이 이 교란이다.

    끌 수 있어야 baseline 과 **같은 상자**에서 보상 효과를 단일 변수로 잴 수 있다.
    기본값은 실측 오프셋(True) — 물리적으로 옳은 쪽이 기본이어야 한다.
    """
    src = _src("v2_env_cfg.py")
    assert 'os.environ.get("HDGP_V2_GOAL_MEASURED", "1") == "1"' in src.replace("_os.", "os.")
    assert "v2_goal_measured: bool" in src, "값이 env.yaml 에 남아야 증명 가능하다"
    assert "P.GRASP_OFFSET_ROOT if self.v2_goal_measured else P.GRASP_OFFSET_ROOT_V1" in src
    # 두 값의 차가 실제로 18 mm 인지 — 대조 해석의 근거
    dz = P.GRASP_OFFSET_ROOT[2] - P.GRASP_OFFSET_ROOT_V1[2]
    assert abs(dz - 0.018) < 1e-9, f"목표 z 차이가 18 mm 가 아니다: {dz*1000:.1f} mm"


# ---------------------------------------------------------------------------
# ★★조건부 추종을 강제하는 지시함수
# ---------------------------------------------------------------------------
def test_settle_radius_leaves_room_for_conditional_advantage():
    """고정점 전략이 지시함수를 만족하는 비율이 **1 보다 충분히 작아야** 한다.

    이것이 v2 설계의 핵심 논거다. v1 은 전 항이 연속 shaping 이라 목표 분포 중심에
    고정점을 찍는 것으로 대부분을 회수할 수 있었고, 실제로 목표→지령 기울기가 0 근처
    (일부는 음수)였다. 반경 ε 지시함수는 **완전 추종해야만 100%** 가 된다.

    상자 ±(50,70,50) mm 균일, 반경 50 mm 구가 상자 안에 완전히 들어가므로
        P = (4/3)π r³ / (2jx · 2jy · 2jz) ≈ 37.4%
    ⇒ 조건부 추종에 약 2.7 배 이득.
    """
    jx, jy, jz = P.GOAL_JITTER_V2
    r = P.SETTLE_RADIUS
    assert r <= min(jx, jy, jz) + _EPS, "구가 상자를 벗어나면 아래 부피 계산이 틀린다"
    p_fixed = (4.0 / 3.0 * math.pi * r**3) / (8.0 * jx * jy * jz)
    assert 0.2 < p_fixed < 0.6, f"고정점 만족률 {p_fixed:.3f} — 조건부 이득이 사라진다"


def test_settle_radius_is_tighter_than_pass_line():
    """지시함수 반경이 합격선(57 mm)보다 좁아야 한다 — 만족하면 합격이 보장된다."""
    assert P.SETTLE_RADIUS < 0.057


# ---------------------------------------------------------------------------
# 계단 구조
# ---------------------------------------------------------------------------
def test_staircase_advancing_a_stage_always_pays():
    """단계 전진은 **항상** 이득이어야 한다: `(k+1+0)/N ≥ (k+1)/N` 경계 도약 ≥ 0.

    v1 은 가산형이라 "높이 들고 가만히"가 충분히 이득이었고, 36 점이 게이트 하나를
    공유해 이동이 도박이었다(프로브 실측 이동 중 0.944 vs 정지 1.000).
    계단은 상위 단계가 하위 단계를 곱해 만들어지므로 임계에서 도약이 음수가 될 수 없다.
    """
    n = P.N_STAGES
    thr = P.STAGE_THRESHOLD
    for k in range(n - 1):
        below = (k + 1.0) / n            # 하위 단계 만점
        above = (k + 1 + thr) / n        # 상위 단계 진입 직후
        assert above >= below - _EPS, f"단계 {k}→{k+1} 경계에서 보상이 떨어진다"


def test_stage_count_matches_implementation():
    """단계 수가 구현과 맞아야 한다 — 어긋나면 계단 최댓값이 1 이 아니게 된다."""
    assert P.N_STAGES == 4
    src = _src("v2_stages.py")
    for name in ("r_grasp", "r_lift", "r_transport", "ok3"):
        assert name in src, f"{name} 이 계단 판정에 없다"
    rw = _src("v2_rewards.py")
    assert "def _stage_value(idx, v0, v1, v2, v3)" in rw, "값은 진행도 네 벌이다"


def test_situation_removal_is_off_by_default():
    """`1_SR`(Hundt)은 기본 off. 원문은 이산 primitive 태스크지만 우리는 50 Hz
    연속제어라 경계 근처에서 매 스텝 보상을 0 으로 만들 위험이 있다 — 단일 변수로 켠다."""
    assert '"use_sr": False' in _src("v2_env_cfg.py")


# ---------------------------------------------------------------------------
# v1 에서 태운 함정의 재발 방지
# ---------------------------------------------------------------------------
def test_transport_is_scored_on_cup_not_tcp():
    """이송 거리는 **컵 원점**으로 잰다(문서 6.3 `R_transport`). TCP 로 재면 v1 의
    37 mm 계통 오프셋이 되살아난다."""
    src = _src("v2_stages.py")
    assert "obj.data.root_pos_w" in src
    assert "def cup_goal_distance" in src
    # 이송·정지 단계가 TCP 프레임을 쓰지 않는지
    seg = src[src.index("def stage_transport"):]
    assert "target_pos_w" not in seg, "이송 이후 단계가 TCP 를 쓰고 있다"


def test_action_penalty_has_no_curriculum():
    """`action_rate` 커리큘럼을 걸지 않는다.

    t73/t75 가 **정확히 발동 시점(36000 step ÷ horizon 24 = ep1500)** 에 꺾였다
    (t75 fine 0.320→0.156 · t73 rew 124→92). 그리고 이 항이 재는 것의 대부분은
    정책의 거칢이 아니라 σ 다(σ≈1·6 차원이면 독립 샘플 차분 기댓값 2σ²×6 = 12).
    """
    src = _src("v2_env_cfg.py")
    assert "self.curriculum.action_rate = None" in src
    assert "self.curriculum.joint_vel = None" in src


def test_action_penalties_are_mean_not_sum():
    """합(sum)은 상시 포화해 gradient 가 0 인 죽은 항이 된다(자매 트랙 실측).
    평균이면 액션 차원이 바뀌어도 스케일이 불변이다."""
    src = _src("v2_rewards.py")
    assert "torch.mean(torch.square(env.action_manager.action)" in src
    assert src.count("torch.mean") >= 2


def test_v1_reward_terms_are_cleared():
    """부모(v1)의 10 항 + 게이트 스택을 **전부** 지운 뒤 계단을 얹어야 한다.
    하나라도 남으면 가산형 병리가 그만큼 되살아난다."""
    src = _src("v2_env_cfg.py")
    assert "setattr(self.rewards, _name, None)" in src


def test_diagnostics_cover_the_pass_metric_and_conditionality():
    """계단은 보상 항이 하나라 진단이 없으면 안이 안 보인다. 최소한 합격 지표와
    조건부 추종 지표는 있어야 한다."""
    src = _src("v2_env_cfg.py")
    for term in ("diag_cup_goal_dist", "diag_at_goal", "diag_stage",
                 "diag_r_transport", "diag_r_settle"):
        assert term in src, f"{term} 진단이 없다"
    # 축 포화는 절대 태스크공간 액션의 상설 감시 지표다(v1 y 99.1% 사망 이력)
    for ax in ("x", "y", "z"):
        assert f'"{ax}"' in src or f"'{ax}'" in src


def test_skeleton_is_inherited_not_rewritten():
    """액션·홈·스폰·자산·물리는 v1 을 **상속**해야 한다 — lift 레시피가 단순한 제어로도
    학습되는 이유라 하나라도 다시 쓰면 그 장점이 사라진다."""
    src = _src("v2_env_cfg.py")
    assert "GraspLeftGripperFabEnvCfg" in src
    # ★라운드 8 예외: `terminations.object_dropping = None` 은 골격 재작성이 아니라
    #   재소환 스위치의 필연이다(같은 문턱이라 종료가 먼저 발화 — 끄지 않으면 재소환이
    #   영영 안 일어난다). 그 외의 종료 항 수정은 여전히 금지.
    # ★라운드 17 예외: `terminations.goal_dwell` — 과제 정의가 바뀌었다(사용자 지시).
    #   "목표로 이송하면 끝"이라 종료 조건이 과제의 일부다. 스위치로 켜야만 생기고
    #   반드시 truncation 이어야 한다(별도 계약 test_dwell_end_is_truncation…).
    _t_lines = [ln for ln in src.splitlines()
                if "self.terminations." in ln
                and "object_dropping = None" not in ln
                and "goal_dwell" not in ln]
    assert not _t_lines, f"골격을 다시 쓰고 있다: {_t_lines}"
    for forbidden in ("self.actions.arm_action =", "self.scene.robot =",
                      "self.events.reset_object_position"):
        assert forbidden not in src, f"골격을 다시 쓰고 있다: {forbidden}"


def test_gym_ids_are_registered_with_v2_entry_points():
    """등록 실패는 `except ImportError: pass` 로 조용히 삼켜진다 — 문자열로 고정한다."""
    src = (_PKG / "config" / "__init__.py").read_text(encoding="utf-8")
    assert 'id="open-grip_l_grasp_sensor_v2"' in src
    assert 'id="open-grip_l_grasp_sensor_v2-play"' in src
    assert "v2_env_cfg:GraspLeftV2EnvCfg" in src
    assert "rl_games_ppo_v2_cfg.yaml" in src


def test_ppo_keeps_the_two_confirmed_fixes():
    """08.28 에 확정된 두 처방을 v2 도 이어받아야 한다 — bounds 2e-2(축 포화 99.1%→0.7%),
    entropy 0.001(0.005 는 t82 에서 기각)."""
    yaml = (_PKG / "config" / "agents" / "rl_games_ppo_v2_cfg.yaml").read_text(encoding="utf-8")
    assert "bounds_loss_coef: 0.02" in yaml
    assert "entropy_coef: 0.001" in yaml
    assert "name: open-grip_l_grasp_sensor_v2" in yaml


def test_every_reward_func_referenced_by_cfg_exists():
    """★`v2_env_cfg` 가 `R.<name>` 으로 거는 함수는 **전부 실재해야** 한다.

    08.29 실측 사고: `_stage_pick` 을 재작성하면서 `diag_r_grasp/lift/transport/settle`
    네 개를 통째로 날렸는데 계약 31 건이 전부 통과했다. 런타임에서만
    `AttributeError: module ... has no attribute 'diag_r_grasp'` 로 터졌다.
    이 테스트가 그 구멍이다 — Isaac 없이 소스 텍스트로 검증한다.
    """
    import re

    cfg = _src("v2_env_cfg.py")
    rw = _src("v2_rewards.py")
    referenced = set(re.findall(r"\bR\.([A-Za-z_][A-Za-z0-9_]*)", cfg))
    assert referenced, "cfg 가 R.* 를 하나도 안 건다면 파싱이 틀린 것"
    defined = set(re.findall(r"^def ([A-Za-z_][A-Za-z0-9_]*)", rw, re.M))
    defined |= set(re.findall(r"^class ([A-Za-z_][A-Za-z0-9_]*)", rw, re.M))
    # v1 에서 재수출하는 것들
    defined |= set(re.findall(r"^\s+([a-z_][A-Za-z0-9_]*),\s*$", rw, re.M))
    missing = referenced - defined
    assert not missing, f"cfg 가 거는데 v2_rewards 에 없는 함수: {sorted(missing)}"


def test_every_obs_func_referenced_by_cfg_exists():
    """관측도 같다 — `obs.<name>` 이 실재하는지."""
    import re

    cfg = _src("v2_env_cfg.py")
    ob = _src("v2_observations.py")
    referenced = set(re.findall(r"\bobs\.([A-Za-z_][A-Za-z0-9_]*)", cfg))
    defined = set(re.findall(r"^def ([A-Za-z_][A-Za-z0-9_]*)", ob, re.M))
    defined |= set(re.findall(r"^\s+([a-z_][A-Za-z0-9_]*),\s*$", ob, re.M))
    missing = referenced - defined
    assert not missing, f"cfg 가 거는데 v2_observations 에 없는 함수: {sorted(missing)}"


def test_net_speed_tracker_never_exceeds_path_length():
    """★순변위 속도는 **경로길이 속도를 넘을 수 없다** — 리셋 텔레포트를 읽으면 넘는다.

    08.29 런타임 스모크 실측 사고: `diag_cup_net_speed` 2.10 m/s vs 순간속도 0.025 m/s
    (**84 배**). 원인은 링버퍼 인덱싱이었다 — `buf[_ptr]` 은 "가장 오래된 슬롯"이지
    "span 스텝 전"이 아니라, 버퍼가 안 찬 리셋 직후 env 가 **직전 에피소드의 위치**를
    읽었다. 계약 33 건이 전부 통과한 채로 넘어갈 뻔했다.

    여기서는 트래커 로직을 순수 파이썬으로 복제해 불변식을 검증한다(Isaac 불필요).
    """
    import random

    W, N, dt = P.NET_SPEED_WINDOW, 3, 0.02
    buf = [[[0.0, 0.0, 0.0] for _ in range(N)] for _ in range(W)]
    n = [0] * N
    ptr = 0
    rng = random.Random(0)
    pos = [[0.0, 0.0, 0.0] for _ in range(N)]
    worst = 0.0

    def norm(a, b):
        return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5

    for t in range(120):
        if t == 47:                       # env 1 리셋(텔레포트)
            pos[1] = [p + 5.0 for p in pos[1]]
            n[1] = 0
        for e in range(N):                # 스텝 이동은 최대 0.01 m
            pos[e] = [pos[e][i] + rng.uniform(-0.01, 0.01) for i in range(3)]
        for e in range(N):
            span = min(n[e], W)
            idx = (ptr - span) % W
            disp = norm(pos[e], buf[idx][e])
            val = disp / max(span, 1) / dt if span > 0 else 0.0
            if t > W + 30:                # 버퍼가 충분히 찬 뒤에만 판정
                worst = max(worst, val)
        for e in range(N):
            buf[ptr][e] = list(pos[e])
            n[e] = min(n[e] + 1, W)
        ptr = (ptr + 1) % W

    # 스텝 이동 상한 0.01·√3 m ⇒ 경로길이 속도 상한 ≈ 0.87 m/s. 순변위는 그 이하여야.
    assert worst < 0.9, f"순변위가 경로길이 상한을 넘었다: {worst:.3f} m/s — 리셋 오염"


def test_net_speed_tracker_indexes_by_span_not_pointer():
    """구현이 `(_ptr − span) % w` 로 **env 별** 슬롯을 봐야 한다.

    `buf[_ptr]` 을 전 env 공통으로 쓰면 리셋 직후 env 가 옛 에피소드를 읽는다.
    """
    src = _src("v2_stages.py")
    assert "(self._ptr - span) % self._w" in src, "span 기준 인덱싱이어야 한다"
    assert "torch.arange" in src, "env 별로 다른 슬롯을 gather 해야 한다"


# ---------------------------------------------------------------------------
# fabric params A/B (08.29) — 이송을 막는 것이 앵커라는 가설의 검증 수단
# ---------------------------------------------------------------------------
def test_fabric_params_default_is_the_original_file():
    """기본값은 **원본** 이어야 한다 — 배선만으로는 어떤 런의 거동도 바뀌면 안 된다."""
    assert V1.FABRIC_PARAMS_FILENAME == "openarm_gripper_left_pose_params.yaml"


def test_fabric_params_variants_change_only_the_two_measured_values():
    """F1/F2 는 자매 트랙과의 **실측 차이 두 개**만 바꿔야 한다.

    자매(`openarm_tesollo_sensor_pose_params.yaml`) 대조 실측:
      · `cspace_attractor.conical_gain`  우리 3.0 vs 자매 1.0  ← 홈으로 당기는 힘 3 배
      · `palm_attractor.damping`         우리 100. vs 자매 50. ← 목표로 가는 감쇠 2 배
    다른 값이 같이 바뀌면 A/B 가 무효가 된다.
    """
    import yaml

    root = _PKG.parents[4] / "FABRICS" / "src" / "fabrics_sim" / "fabric_params"
    if not root.exists():                      # 저장소 배치가 다르면 경로로 재탐색
        for cand in _PKG.parents:
            hit = list(cand.glob("**/fabric_params/openarm_gripper_left_pose_params.yaml"))
            if hit:
                root = hit[0].parent
                break
    def flat(d, p=""):
        out = {}
        for k, v in (d or {}).items():
            key = f"{p}{k}"
            if isinstance(v, dict):
                out.update(flat(v, key + "."))
            elif isinstance(v, (int, float, bool, str)):
                out[key] = v
        return out

    base = flat(yaml.safe_load((root / "openarm_gripper_left_pose_params.yaml").read_text()))
    exp = {
        "_f1": {"fabric_params.cspace_attractor.conical_gain": 1.0},
        "_f2": {"fabric_params.cspace_attractor.conical_gain": 1.0,
                "fabric_params.palm_attractor.damping": 50.0},
    }
    for suf, want in exp.items():
        f = root / f"openarm_gripper_left_pose_params{suf}.yaml"
        assert f.exists(), f"{f.name} 이 없다"
        v = flat(yaml.safe_load(f.read_text()))
        assert set(v) == set(base), f"{suf}: 키 집합이 원본과 달라졌다"
        diff = {k: v[k] for k in base if base[k] != v[k]}
        assert diff == want, f"{suf}: 의도치 않은 차이 {diff}"


def test_fabric_params_selector_rejects_typos():
    """오타는 **즉시 죽어야** 한다 — 조용한 폴백은 '실험을 안 한 것'을 '했다'고 믿게 만든다."""
    src = (_PKG.parent / "grasp_sensor" / "grasp_left_preset.py").read_text(encoding="utf-8")
    assert "_FP_ALLOWED" in src and "raise ValueError" in src


def test_fabric_params_is_actually_passed_to_the_fabric():
    """preset 상수가 fabric 생성자까지 **실제로 전달**되는지 — 상수만 만들고 안 쓰면 무효다."""
    fa = (_PKG.parent / "grasp_sensor" / "grasp_left_fabric_action.py").read_text(encoding="utf-8")
    assert "fabric_params_filename=P.FABRIC_PARAMS_FILENAME" in fa


# ---------------------------------------------------------------------------
# ★★08.29 라운드 5 — baseline 복구 + ⑤ 만 더한다
# ---------------------------------------------------------------------------
def test_stage0_to_2_are_the_baseline_originals():
    """★라운드 3·4 의 재설계를 **되돌렸는지** 잠근다.

    env 별 결말 프로브(결정론 1024 env)가 밝힌 것: baseline 은 ①~④ 를 이미 푼다 —
    파지 실패 0.6% · **도달 93.7%** · 컵–목표 최저 거리 중앙 28.1 mm. 남은 실패는
    ⑤ 정지(성공 0.0%) 하나뿐이다. 그러니 0~2 단계는 건드리면 안 된다.

    ⚠ 라운드 3 은 `v_1 = r_transport/0.1`, 라운드 4 는 `v_2 = r_close·P_dist` 로 바꿨고
      둘 다 신호를 얇게 만들어 σ 를 1.3 까지 팽창시켰다(결정론 파지 0.190).
    """
    lines = {ln.strip() for ln in _src("v2_stages.py").splitlines()}
    assert "return pos, (r_grasp, r_lift, r_transport, v3)" in lines, (
        "stage 0~2 값이 baseline 원본(r_grasp · r_lift · r_transport)이어야 한다")
    src = _src("v2_stages.py")
    for gone in ("v1 = (r_transport / P.STAGE_THRESHOLD)", "v2 = r_close * p_dist",
                 "v2 = r_close * p_dist * p_still * p_upright"):
        assert gone not in src, f"라운드 3·4 회귀가 남아 있다: {gone}"


def test_v3_equals_one_implies_pass():
    """`v_3 = 1` 이면 합격이어야 한다 — 밴드 near 가 합격선보다 **엄격**해야 성립한다.

    지형은 부드럽게 하되 합격선(`success_ok`)은 안 낮춘다. 둘을 같이 낮추면
    "쉬워져서 올랐다"를 "고쳐서 올랐다"로 오독한다.
    """
    assert P.P_STILL_BAND[1] <= P.STAGE3_SPEED_MAX, "속도 near 가 합격선보다 느슨하다"
    assert P.P_UPRIGHT_BAND[1] >= P.STAGE3_UPRIGHT_MIN, "직립 near 가 합격선보다 느슨하다"
    # 합격선 정확히에서 v_3 < 1 (아직 여지가 남아야 gradient 가 산다)
    v_pass = (_smoothstep(P.STAGE3_SPEED_MAX, *P.P_STILL_BAND)
              * _smoothstep(P.STAGE3_UPRIGHT_MIN, *P.P_UPRIGHT_BAND))
    assert 0.5 < v_pass < 1.0, f"합격선에서 v_3 = {v_pass:.3f}"


def test_bands_are_calibrated_to_the_measured_operating_point():
    """★★밴드는 "합격선"이 아니라 **"현재 행동 → 합격선" 구간**을 덮어야 한다.

    baseline best 결정론 1024 env 실측(목표 반경 안):
        속도  p10 0.0128 · 중앙 0.0399 · p90 0.1485   (합격 < 0.05)
        직립  p10 0.9747 · 중앙 0.9829 · p90 0.9933   (합격 > 0.99)
    구 밴드 (0.30,0.05)/(0.90,0.99) 는 이 운전점에서 v_3 = 0.982 로 **포화**해
    운전점→합격 gradient 가 +0.0044 뿐이었다. 재보정으로 +0.0710(16 배).
    """
    OP_SPD, OP_COS = 0.0399, 0.9829
    v_op = (_smoothstep(OP_SPD, *P.P_STILL_BAND)
            * _smoothstep(OP_COS, *P.P_UPRIGHT_BAND))
    assert 0.2 < v_op < 0.8, (
        f"운전점에서 v_3 = {v_op:.3f} — 밴드가 포화(>0.8)하거나 죽어(<0.2) 있다")
    v_pass = (_smoothstep(P.STAGE3_SPEED_MAX, *P.P_STILL_BAND)
              * _smoothstep(P.STAGE3_UPRIGHT_MIN, *P.P_UPRIGHT_BAND))
    grad = (v_pass - v_op) / P.N_STAGES
    assert grad >= 0.05, f"운전점→합격 gradient {grad:+.4f} — 구 밴드(+0.0044)와 다를 바 없다"
    # p10/p90 이 밴드 안에 들어와야 분포 전체에 gradient 가 산다
    for spd in (0.0128, 0.1485):
        assert 0.0 <= _smoothstep(spd, *P.P_STILL_BAND) <= 1.0
    assert _smoothstep(0.9747, *P.P_UPRIGHT_BAND) < 0.5, "p10 직립이 밴드 하단에 있어야"


# ---------------------------------------------------------------------------
# ★★08.29 라운드 6 — 제어 층(회전 자유도 · env 별 앵커)
# ---------------------------------------------------------------------------
def test_control_switches_default_off_so_v1_is_untouched():
    """두 스위치 모두 cfg 기본값이 `None` 이라 **v1 거동이 안 바뀐다.**

    액션 파일은 v1(`grasp_sensor`)과 **공유**한다. v1 은 챔피언이자 폴백이라 동결이고,
    이 기본값이 그 불변의 유일한 보증이다.
    """
    fa = (_PKG.parent / "grasp_sensor" / "grasp_left_fabric_action.py").read_text(
        encoding="utf-8")
    assert "palm_max_pose_angle: float | None = None" in fa
    assert "anchor_relatch_cup_z: float | None = None" in fa
    # 켜지 않으면 preset 값을 그대로 쓴다
    assert "P.PALM_MAX_POSE_ANGLE if self.cfg.palm_max_pose_angle is None" in fa
    # 재앵커는 조건부 블록 안에서만 일어난다
    assert "if self.cfg.anchor_relatch_cup_z is not None:" in fa
    src = _src("v2_env_cfg.py")
    for k, d in (("HDGP_V2_ROT_WIDE", "0"), ("HDGP_V2_ANCHOR_RELATCH", "0")):
        assert f'os.environ.get("{k}", "{d}") == "1"' in src.replace("_os.", "os.")
    for k in ("v2_rot_wide: bool", "v2_anchor_relatch: bool"):
        assert k in src, f"{k} 가 cfg 필드로 안 박혀 있으면 env.yaml 로 증명 못 한다"


def test_rotation_box_is_widened_past_the_measured_span():
    """★회전 박스는 실측이 요구하는 스팬을 덮어야 한다.

    baseline best 결정론 1024 env, 축별 부호 실측:
        ez  전 국면 −1 (0.944 / 0.613)          → 중심 오류
        ey  접근 +0.594  ·  이송 −0.885         → **부호가 뒤집힌다 = 40° 스팬 필요**
        ex  리프트 후 −0.886                     → 리프트 후 −20° 를 넘는다
    구 박스는 ±20° = 스팬 40° 로 `ey` 요구와 정확히 같아 여유가 0 이었다.
    """
    span_old = 2 * P.PALM_MAX_POSE_ANGLE
    span_new = 2 * P.PALM_MAX_POSE_ANGLE_WIDE
    assert abs(span_old - math.radians(40.0)) < 1e-6, "구 박스 스팬이 40° 가 아니다"
    assert span_new >= math.radians(100.0), (
        f"새 스팬 {math.degrees(span_new):.0f}° — ey 요구 40° 에 여유가 부족하다")
    # 중심 오류(ez)도 흡수돼야 한다: 구 박스 한계를 넘어설 수 있어야
    assert P.PALM_MAX_POSE_ANGLE_WIDE > 2 * P.PALM_MAX_POSE_ANGLE


def test_anchor_relatch_is_monotone_and_reset_restores_home():
    """★재앵커는 **단조 래치**이고 리셋에서 홈으로 되돌아와야 한다.

    깜빡이면 널스페이스 목표가 흔들려 지령이 더 떨고, 안 되돌리면 직전 에피소드의
    파지 자세가 다음 에피소드의 앵커가 되어 **리셋 오염**이 된다(이 트랙에서 리셋
    오염에 이미 세 번 당했다).
    """
    fa = (_PKG.parent / "grasp_sensor" / "grasp_left_fabric_action.py").read_text(
        encoding="utf-8")
    assert "self._anchor_latched |= _new" in fa, "단조 래치(OR 누적)여야 한다"
    assert "(~self._anchor_latched) &" in fa, "이미 래치된 env 는 다시 안 쓴다"
    assert "self._anchor_home" in fa, "홈 앵커를 보존해야 되돌릴 수 있다"
    assert "self._anchor_latched[env_ids] = False" in fa
    assert "self._fabric.default_config[env_ids] = self._anchor_home[env_ids]" in fa
    assert P.ANCHOR_RELATCH_CUP_Z == P.MINIMAL_LIFT_HEIGHT


def test_rotation_axis_diagnostics_are_registered():
    """★회전 축(3·4·5) 진단 — 라운드 6 까지 **계측 공백**이었다.

    액션은 6 차원(위치 3 + 회전 3)인데 `diag_act_*` 는 위치 축 0·1·2 만 등록돼 있어
    회전 포화를 한 번도 못 봤다. v1 에서 y 축 포화 99.1% 로 죽은 이력이 있는데
    회전에는 같은 감시가 없었고, 실제로 euler 박스가 98% 포화 중이었다.
    """
    src = _src("v2_env_cfg.py")
    for ax, i in (("ez", 3), ("ey", 4), ("ex", 5)):
        assert f'("{ax}", {i})' in src, f"회전 축 {ax}(={i}) 진단이 없다"
    assert '("x", 0), ("y", 1), ("z", 2)' in src, "위치 축 진단은 유지돼야 한다"


# ---------------------------------------------------------------------------
# ★★08.29 라운드 7 — 지속 정착 프리미엄 (hold)
# ---------------------------------------------------------------------------
def test_hold_premium_defaults_off_and_is_cfg_visible():
    """기본 off — 켜는 것은 환경변수의 명시적 행위이고 값은 env.yaml 에 남아야 한다."""
    src = _src("v2_env_cfg.py")
    assert 'os.environ.get("HDGP_V2_HOLD", "0") == "1"' in src.replace("_os.", "os.")
    assert "v2_hold_premium: bool" in src
    assert '"hold_weight": P.HOLD_WEIGHT if self.v2_hold_premium else 0.0' in src
    rw = _src("v2_rewards.py")
    assert "hold_weight: float = 0.0" in rw, "파라미터 기본값이 0 이어야 기존 판이 불변이다"


def test_hold_counter_resets_on_break_and_on_env_reset():
    """★카운터는 **단조가 아니다** — 조건이 깨지면 0. 그리고 에피소드 리셋에서도 0.

    단조로 두면 한 번 정착했다 떠나도 프리미엄이 계속 나와 '찍고 튀기'가 생긴다.
    리셋을 안 하면 직전 에피소드의 hold 가 새 에피소드 첫 스텝에 지급된다(리셋 오염).
    """
    rw = _src("v2_rewards.py")
    assert "torch.where(succ > 0.5, self._hold + 1.0," in rw
    assert "torch.zeros_like(self._hold))" in rw, "깨지면 0 으로 — 감쇠나 유지가 아니라"
    assert "self._hold[env_ids] = 0.0" in rw and "self._hold[:] = 0.0" in rw


def test_hold_uses_instantaneous_speed_not_net():
    """★★hold 판정은 **순간속도**여야 한다.

    제자리 진동은 순변위 ≈ 0 이라, 순변위 조건이면 흔들기가 success_ok 를 세워
    프리미엄을 받는다 — 라운드 1 R1 이 정확히 이 함정으로 실패했다. 이 프리미엄의
    존재 이유가 '흔들기와 정착을 가르는 것'이므로 여기가 뚫리면 전부 무효다.
    """
    st = _src("v2_stages.py")
    body = st[st.index("def settle_success"):st.index("def success_ok")]
    assert "torch.norm(obj.data.root_lin_vel_w, dim=1)" in body, "순간속도여야 한다"
    assert "net_speed" not in body, "순변위가 섞이면 흔들기가 프리미엄을 받는다"


def test_hold_premium_flips_the_break_even():
    """★프리미엄 크기가 기대값 구조를 실제로 바꾸는지 산술로 잠근다.

    실측: 흔들기 0.814/step · 정착(프리미엄 전) 1.0/step · 낙하 손실 ≈ 잔여 100 step.
    프리미엄 전 손익분기 실패율 22.9% — 실측 전도율 51% 아래라 흔들기가 이겼다.
    프리미엄 후(램프 완주 +0.5) 손익분기가 40% 를 넘어야 정착 도박이 성립한다.
    """
    wob, settle, horizon = 0.814, 1.0, 100.0
    gain_old = settle - wob
    be_old = (gain_old * horizon) / (wob * horizon)
    assert be_old < 0.25, "전제 확인 — 프리미엄 전 손익분기가 낮아야 이 처방이 성립한다"
    gain_new = settle + P.HOLD_WEIGHT - wob
    be_new = (gain_new * horizon) / (wob * horizon)
    assert be_new > 0.40, f"프리미엄 후 손익분기 {be_new:.1%} — 기대값을 못 뒤집는다"
    # ★08.31 라운드 15 갱신 — 구 계약은 램프 10~60(0.2~1.2 s)을 강제했다. 그건 합격에
    #   **속도 조건이 있던** 시절 값이다(연속 합격 중앙 1 스텝). 속도를 빼고 재니 실측
    #   체류가 중앙 61 · p90 191 스텝이라 60 이면 대부분 만점이라 "더 오래"에 gradient
    #   가 없다. 상한을 에피소드의 60%까지 연다(유계는 유지 — 스케일 폭주 금지).
    assert 10 <= P.HOLD_RAMP_STEPS <= 150


# ---------------------------------------------------------------------------
# ★★08.29 라운드 8 — 낙하/전도 재소환
# ---------------------------------------------------------------------------
def test_respawn_defaults_off_and_disables_drop_termination_together():
    """기본 off. 켜면 **종료 항 제거와 이벤트 등록이 한 몸**이어야 한다.

    재소환 문턱과 종료 문턱이 같은 높이(`OBJECT_DROP_HEIGHT`)라, 종료를 안 끄면
    종료가 먼저 발화해 재소환이 영영 안 일어난다 — 조용히 "실험을 안 한 것"이 된다.
    """
    src = _src("v2_env_cfg.py")
    assert 'os.environ.get("HDGP_V2_RESPAWN", "0") == "1"' in src.replace("_os.", "os.")
    assert "v2_respawn: bool" in src
    blk = src[src.index("if self.v2_respawn:"):]
    blk = blk[:blk.index("\n\n")]
    assert "self.terminations.object_dropping = None" in blk
    assert "respawn_cup = EventTermCfg" in blk and 'mode="interval"' in blk


def test_respawn_keeps_clearance_and_lands_upright():
    """재소환은 **TCP 여유 · 직립 · 속도 0** 를 보장해야 한다.

    여유(리젝션 샘플링)가 없으면 팔 위에 컵이 떨어져 물리 폭발, 직립이 아니면
    곧바로 다시 전도 판정, 속도를 안 지우면 낙하 관성이 이어진다.
    """
    ev = _src("v2_events.py")
    assert "d >= P.RESPAWN_TCP_CLEARANCE" in ev, "리젝션 조건이 없다"
    # ★★폴백은 **보류**여야 한다. 초판의 "가장 먼 후보" 폴백은 스모크 실측에서
    #   TCP 19 mm 옆에 컵을 놓았다(스폰 상자 40×40 mm 라 팔이 위에 있으면 어떤
    #   후보도 여유 120 mm 를 못 채운다) — 그리퍼 안 텔레포트는 물리 폭발 위험이다.
    assert "has_ok = ok.any(dim=1)" in ev and '_STATS["defer"]' in ev, (
        "여유 미달 시 이번 스텝 보류(다음 스텝 재시도)여야 한다")
    assert "torch.argmax(d, dim=1)" not in ev, "가장 먼 후보 폴백이 되살아났다"
    assert "pose[:, 3] = 1.0" in ev, "직립(항등 quat)으로 놓아야 한다"
    assert "write_root_velocity_to_sim(torch.zeros" in ev, "속도를 지워야 한다"
    # 여유가 스폰 상자 대각(≈57 mm)보다 커야 리젝션이 의미 있다
    import math as _m
    diag = _m.sqrt((2 * V1.CUP_SPAWN_X_RANGE) ** 2 + (2 * V1.CUP_SPAWN_Y_RANGE) ** 2)
    assert P.RESPAWN_TCP_CLEARANCE > diag, "여유가 상자 대각 이하면 조건이 항상 참이다"


def test_respawn_thresholds_reuse_task_constants():
    """재소환 판정은 기존 과제 상수를 **재사용**해야 한다 — 새 문턱을 만들면
    종료·재소환·판정이 서로 다른 높이를 보게 된다."""
    ev = _src("v2_events.py")
    assert "P.OBJECT_DROP_HEIGHT" in ev
    assert "P.RESPAWN_TIPPED_COS" in ev
    assert 0.3 <= P.RESPAWN_TIPPED_COS <= 0.7, "전도 판정 60° 근방이어야"
    assert "P.CUP_SPAWN_X_CENTER" in ev and "P.CUP_SPAWN_Z" in ev, "스폰 상자 재사용"


# ---------------------------------------------------------------------------
# ★★08.30 라운드 9 — DR + ADR 사다리
# ---------------------------------------------------------------------------
def test_dr_defaults_off_and_level0_equals_current_task():
    """기본 off. 켜도 **레벨 0 은 현행 과제와 완전히 동일**해야 한다 — 그래야
    HR(챔피언 후보)과의 대조가 성립하고, 사다리가 진짜 사다리가 된다."""
    src = _src("v2_env_cfg.py")
    assert 'os.environ.get("HDGP_V2_DR", "0") == "1"' in src.replace("_os.", "os.")
    assert "v2_dr: bool" in src
    assert '"mass_distribution_params": (1.0, 1.0)' in src, "레벨 0 질량 = 무변경"
    cur = _src("v2_curriculum.py")
    # 레벨 0 에서 보간 f=0 → 현행 상수 그대로
    assert "f = self._level / (P.ADR_LEVELS - 1)" in cur
    assert "P.GOAL_JITTER_V2[1] + f *" in cur
    # ★08.30 라운드 13 — 스폰은 **절대 상자 보간**으로 바뀌었다(구 대칭 오프셋 폐기)
    assert "b0[i] + f * (bm[i] - b0[i])" in cur


def test_adr_promotion_needs_success_and_spacing_and_halves_ema():
    """승급 = 성공 EMA ≥ 게이트 **AND** 최소 간격. 그리고 승급 시 EMA 반토막.

    반토막이 없으면 이전(쉬운) 레벨의 성공 잔고로 연쇄 승급해 사다리가 계단이 아니게
    된다. 간격이 없으면 한 배치의 우연으로 두 칸을 뛴다(fab_test19 의
    ADR_MIN_EPOCHS_BETWEEN 과 같은 장치).
    """
    cur = _src("v2_curriculum.py")
    assert "self._ema >= P.ADR_SUCCESS_GATE" in cur
    assert "step - self._last_promo >= P.ADR_MIN_STEPS_BETWEEN" in cur
    # ★08.30 — 승급 신호는 상수로만 지정한다(하드코딩된 항 이름 금지).
    #   `diag_success` 는 σ 포함 per-step 4조건 플래그라 승급 신호로 못 쓴다(실측 정점
    #   0.086 초 vs 게이트 1.5 초 = 17 배). 척도를 바꿔도 계약이 따라오도록 상수 경유.
    assert "P.ADR_SIGNAL_TERM" in cur, "승급 신호가 상수를 경유하지 않는다"
    assert '"diag_success"' not in cur, "승급 신호에 diag_success 하드코딩 금지"
    assert "self._ema *= 0.5" in cur, "승급 시 EMA 반토막이 없다"
    assert "self._level < P.ADR_LEVELS - 1" in cur, "만렙 초과 방지"
    # off-by-one 재발 방지: 만렙(4)이 실제로 도달 가능해야 한다(ADR 분모 메모리)
    assert "P.ADR_LEVELS - 1" in cur


def test_upright_shaping_is_additive_gated_and_cannot_skip_a_stage():
    """★라운드 11 처방 A — 이송 구간 직립 셰이핑의 불변식 넷.

    ① **가산**이어야 한다. 라운드 3 은 stage 2 값에 직립·정지를 **곱해** 운전점에서
       유인을 +0.250 → +0.077 로 2.7 배 약화시켰고 σ 1.3 팽창으로 실패했다.
    ② 가중치 < 계단 한 칸(1/N_STAGES = 0.25). 보너스가 단계를 건너뛰면 계단의
       "전진이 항상 유리" 성질이 깨진다.
    ③ `idx >= UPRIGHT_MIN_STAGE`(이송) 게이팅 — 안 옮기고 세워만 두는 해킹 차단.
    ④ stage 2·3 에 **동일하게** 붙어야 2→3 무점프가 보존된다(둘 다 게이트 통과).
    """
    import openarm.gripper.left.grasp_sensor_v2.v2_preset as P
    rew = _src("v2_rewards.py")
    assert "r = r + upright_weight * S.upright_shaped(env, object_cfg) * gate" in rew, \
        "직립 셰이핑이 가산항이 아니다(곱하면 라운드 3 실패 재현)"
    assert "(idx >= P.UPRIGHT_MIN_STAGE)" in rew, "이송 게이팅이 없다"
    assert P.UPRIGHT_WEIGHT < 1.0 / P.N_STAGES, \
        f"가중치 {P.UPRIGHT_WEIGHT} 가 계단 한 칸 {1.0/P.N_STAGES} 이상 — 단계 건너뛰기 가능"
    assert P.UPRIGHT_MIN_STAGE >= 2.0, "이송 이전 단계에서 지급되면 안 된다"
    # 같은 밴드를 써야 이송 중 직립이 stage 3 의 p_upright 로 이어진다
    st = _src("v2_stages.py")
    assert "def upright_shaped(" in st
    assert "smoothstep(_cup_upright_cos(env, object_cfg), *P.P_UPRIGHT_BAND)" in st, \
        "셰이핑이 v_3 과 다른 밴드를 쓴다(두 신호가 싸운다)"


def test_hold_ramp_matches_measured_dwell():
    """★08.31 라운드 15 — hold 는 "오래 머물수록" 커져야 한다(사용자 지시).

    새 기준(속도 제거)에서 실측 체류가 중앙 61 · p90 191 스텝이라 램프 30 은
    대부분 만점 → gradient 소멸. 램프가 실측 중앙값보다 충분히 커야 한다.
    ⚠ 무한 증가는 금지 — 계단 최대 1.0 대비 스케일이 터지면 앞 단계가 죽는다.
    """
    import openarm.gripper.left.grasp_sensor_v2.v2_preset as P
    assert P.HOLD_RAMP_STEPS >= 100, "램프가 짧아 대부분 만점 — 더 오래에 유인이 없다"
    rw = _src("v2_rewards.py")
    assert "(self._hold / P.HOLD_RAMP_STEPS).clamp(max=1.0)" in rw, \
        "hold 가 유계가 아니다(스케일 폭주)"


def test_still_at_goal_is_gated_by_the_collider_not_a_band():
    """★★08.31 라운드 16 — 목표 안 안정화 셰이핑의 불변식.

    ① 게이트가 **콜라이더 자체**여야 한다. D 판은 근접 밴드(0.16 m)를 써서 평균 컵–목표
       125 mm 에 멈춰 보상을 받았다(`at_goal` 0.066 vs 기준선 0.235). 밖에서 못 벌게 한다.
    ② 밴드는 **실측 운전점**(목표 안 평균 속도 중앙 0.144 · p10 0.104)을 덮어야 한다.
    ③ 가산항이고 가중치 < 계단 한 칸(0.25).
    ④ 합격 판정에는 들어가지 않는다 — 속도를 다시 게이트로 쓰면 진동체가 통과한다.
    """
    import openarm.gripper.left.grasp_sensor_v2.v2_preset as P
    st = _src("v2_stages.py")
    body = st.split("def still_at_goal(")[1].split("\ndef ")[0]
    assert "(dist < P.SETTLE_RADIUS)" in body, "게이트가 콜라이더가 아니다"
    assert "P.P_DIST_BAND" not in body and "P.STILL_NEAR_BAND" not in body, \
        "근접 밴드를 쓰면 도착 전에 멈춰 벌 수 있다(D 판 실패)"
    lo, hi = P.STILL_GOAL_BAND[1], P.STILL_GOAL_BAND[0]
    assert lo < 0.104 < hi, f"밴드 {P.STILL_GOAL_BAND} 가 실측 운전점(p10 0.104)을 안 덮는다"
    assert lo < 0.144 < hi, f"밴드 {P.STILL_GOAL_BAND} 가 실측 중앙(0.144)을 안 덮는다"
    assert P.STILL_GOAL_WEIGHT < 1.0 / P.N_STAGES
    rw = _src("v2_rewards.py")
    assert "r = r + still_goal_weight * S.still_at_goal(" in rw, "가산항이 아니다"
    su = st.split("def success_ok(")[1].split("\ndef ")[0]
    assert "STILL_GOAL" not in su, "합격 판정에 속도가 되살아났다"


def test_obs_noise_is_gated_on_grasp():
    """★★08.31 라운드 15 (사용자 지적) — 스텝 잡음은 **파지 전에만** 얹는다.

    실기에서 컵 좌표의 출처가 국면마다 다르다: 파지 전은 `/cup_pose` 인식,
    파지 후는 **TCP FK + 파지 오프셋**(엔코더 정밀도). 파지 후에도 흔들면 절대 위치
    지령이 그대로 떨고 fabric 이 충실히 따라가며, 학습 내내 그러면 정책이 고이득
    반응성을 배운다(추론 때만 꺼도 안 사라진다 — ablation 으로 확인).
    """
    ob = _src("v2_observations.py")
    assert "held = _S.stage_close(env, jaw_cfg, object_cfg) > 0.5" in ob, \
        "파지 게이팅이 없다 — 잡은 뒤에도 관측이 떤다"
    assert "noise * (~held)" in ob
    assert '"jaw_cfg": _jaw()' in _src("v2_env_cfg.py"), \
        "jaw_cfg 를 주입하지 않으면 게이팅이 조용히 꺼진다(기본값 None)"


def test_still_shaping_mirrors_upright_shaping_invariants():
    """★라운드 14 정지 처방 — 처방 A 와 **같은 불변식 넷**을 지켜야 한다.

    ① 가산(곱하면 라운드 3 실패 재현) ② 가중치 < 계단 한 칸 ③ 이송 게이팅
    ④ `p_still` 은 `v_3` 과 같은 밴드 · `p_near` 는 `P_DIST_BAND`(150 mm 밖 0)
       — ④ 가 없으면 "멀리서 멈춰 점수 벌기"가 열린다.
    ⑤ 속도는 **순간속도**여야 한다(순변위면 제자리 진동이 만점 — 라운드 1 함정).
    """
    import openarm.gripper.left.grasp_sensor_v2.v2_preset as P
    rew = _src("v2_rewards.py")
    assert "r = r + still_weight * S.still_shaped(" in rew, \
        "감속 셰이핑이 가산항이 아니다"
    assert "(idx >= P.STILL_MIN_STAGE)" in rew, "이송 게이팅이 없다"
    assert P.STILL_WEIGHT < 1.0 / P.N_STAGES, \
        f"가중치 {P.STILL_WEIGHT} 가 계단 한 칸 이상 — 단계 건너뛰기 가능"
    assert P.STILL_MIN_STAGE >= 2.0
    st = _src("v2_stages.py")
    assert "def still_shaped(" in st
    assert "smoothstep(speed, *P.P_STILL_BAND) * smoothstep(dist, *P.STILL_NEAR_BAND)" in st, \
        "밴드가 v_3 과 다르거나 거리 게이트가 없다"
    # ★★08.31 — 감속 보상은 **직립을 곱해야** 한다. 안 그러면 "눕혀서 정지"가 열린다
    #   (E 판 ep2150: at_goal 은 올랐는데 전도 14.1% → 85.0%, ⑤ 41.4% → 25.3%).
    body = st.split("def still_shaped(")[1].split("\ndef ")[0]
    assert "_cup_upright_cos(env, object_cfg), *P.P_UPRIGHT_BAND" in body and "* upright" in body, \
        "감속 보상에 직립 인자가 없다 — 눕힌 채 정지해도 보상을 받는다"
    # ★08.30 D 판 실측 — `P_DIST_BAND`(0.16) 재사용은 "도착 전 멈추기"에 돈을 준다.
    #   평균 컵–목표 125 mm 가 그 밴드 한가운데였고 at_goal 이 기준선의 1/3.5 로 무너졌다.
    #   근접 게이트는 **운전 거리보다 확실히 안쪽**이어야 한다.
    assert P.STILL_NEAR_BAND[0] <= 0.10, \
        f"근접 게이트 {P.STILL_NEAR_BAND[0]} 가 넓다 — 목표 밖에 멈춰도 보상을 받는다"
    assert P.STILL_NEAR_BAND[0] > P.STILL_NEAR_BAND[1] >= 0.0
    assert "torch.norm(obj.data.root_lin_vel_w, dim=1)" in st.split("def still_shaped(")[1] \
        .split("def ")[0], "감속 판정이 순간속도가 아니다"


def test_still_shaping_is_off_by_default():
    cfg = _src("v2_env_cfg.py")
    assert '_V2_STILL_SHAPE = _os.environ.get("HDGP_V2_STILL_SHAPE", "0") == "1"' in cfg
    assert "if self.v2_still_shaping else 0.0" in cfg


def test_upright_shaping_is_off_by_default():
    """스위치는 기본 off — 대조군 조건이 코드 기본값으로 남아야 한다."""
    cfg = _src("v2_env_cfg.py")
    assert '_V2_UPRIGHT = _os.environ.get("HDGP_V2_UPRIGHT", "0") == "1"' in cfg
    assert "P.UPRIGHT_WEIGHT\n" in cfg or "P.UPRIGHT_WEIGHT" in cfg
    assert "if self.v2_upright_shaping else 0.0" in cfg


def test_adr_spawn_box_stays_inside_measured_envelope():
    """★★08.30 라운드 13 — 모든 레벨의 스폰 상자가 **P1 실측 봉투** 안이어야 한다.

    라운드 12 는 `SPAWN_X_SAFE_MIN`(구 홈의 관통 경계) 하한만 잘랐는데, P1 정책 스윕이
    방향이 반대임을 보였다 — x 0.344 는 ① 파지 실패 0~7%(멀쩡), x 0.417 은 46~70%,
    x 0.436 은 **100%**. 벽은 −x 가 아니라 **+x(≈0.41)** 다.
    ⇒ 봉투는 중심 비대칭이라 ± 오프셋으로 표현 불가. **절대 상자 보간**으로만 간다.
    """
    import openarm.gripper.left.grasp_sensor_v2.v2_preset as P
    cur = _src("v2_curriculum.py")
    assert "P.ADR_SPAWN_BOX_L0" in cur and "P.ADR_SPAWN_BOX_MAX" in cur, \
        "스폰 사다리가 절대 상자 보간이 아니다"
    assert "P.SPAWN_X_SAFE_MIN" not in cur, \
        "폐기된 관통 경계를 아직 참조한다(구 홈 기준이라 무효)"
    assert 'pr["x"] = (xlo - P.CUP_SPAWN_X_CENTER' in cur, \
        "절대 경계를 중심 기준 오프셋으로 변환하지 않았다(부호 함정)"
    xl, xh, yl, yh = P.ADR_SPAWN_BOX_MAX
    # ★08.31 재실측 — +x 벽은 y 에 의존한다. y 0.295 까지는 x 0.390 이 ① 0%,
    #   y 0.338 에서는 x 0.390 이 ① 53%. 사각 상자는 안전한 쪽(0.390)으로 잡는다.
    assert xh <= 0.395, f"만렙 스폰 상한 {xh} — 실측 벽 밖(y 상단에서 ① 53%)"
    assert yh <= 0.300, f"만렙 스폰 y 상한 {yh} — x 상한과 양립 불가(대각 벽)"
    for lvl in range(P.ADR_LEVELS):
        f = lvl / (P.ADR_LEVELS - 1)
        b = [P.ADR_SPAWN_BOX_L0[i] + f * (P.ADR_SPAWN_BOX_MAX[i] - P.ADR_SPAWN_BOX_L0[i])
             for i in range(4)]
        assert b[0] >= xl - 1e-9 and b[1] <= max(xh, P.ADR_SPAWN_BOX_L0[1]) + 1e-9
        assert b[2] >= yl - 1e-9 and b[3] <= yh + 1e-9


def test_adr_goal_and_bias_max_match_measurement():
    """★08.30 P1 — 목표 만렙은 1.4 배(⑤ 72.2%), obs bias 는 실기 오차 41 mm 를 덮는다."""
    import openarm.gripper.left.grasp_sensor_v2.v2_preset as P
    # ★08.31 — y 만렙은 `HDGP_V2_GOAL_Y_MAX` 로 오버라이드한다(호스트별 단일 변수).
    #   기본값은 현행 0.100 이어야 대조군 조건이 코드 기본값으로 남는다.
    assert '_os.environ.get("HDGP_V2_GOAL_Y_MAX", "0.100")' in _src("v2_preset.py")
    assert P.ADR_GOAL_JITTER_MAX == (0.050, 0.100, 0.065)
    # ★bias 상한은 합격 반경의 절반이다(계약 test_obs_noise_... 가 강제).
    #   실기 41 mm 는 DR 로 덮을 수 없다 — 관측이 치우치면 그만큼 실제 컵이 벗어난다.
    assert 0.020 < P.ADR_OBS_BIAS_MAX <= P.SETTLE_RADIUS * 0.5


def test_adr_ladder_can_demote_with_hysteresis():
    """★★08.30 라운드 12 — 사다리는 **양방향**이어야 한다.

    한 방향뿐이라 소화 못 하는 레벨에 영구 고착됐다(레벨 2 승급 후 승급 신호
    2.50 → 0.62 초로 1540 epoch 단조 하락, 대응 없음). 강등 문턱은 승급선보다
    낮아야 진동하지 않는다(히스테리시스).
    """
    import openarm.gripper.left.grasp_sensor_v2.v2_preset as P
    cur = _src("v2_curriculum.py")
    assert "self._level -= 1" in cur, "강등 경로가 없다"
    assert "self._ema < P.ADR_DEMOTE_GATE" in cur
    assert P.ADR_DEMOTE_GATE < P.ADR_SUCCESS_GATE, \
        "강등 문턱이 승급선 이상 — 승급↔강등 진동이 난다"
    assert "step - self._last_promo >= P.ADR_MIN_STEPS_BETWEEN" in cur


def test_adr_min_spacing_raised_after_level3_collapse():
    """★08.30 라운드 11 — 승급 간격 150 → 300 epoch.

    150 에서 ep653/800/970 연속 승급 → 레벨 3 붕괴(succ 400 epoch 째 0.000).
    """
    import openarm.gripper.left.grasp_sensor_v2.v2_preset as P
    assert P.ADR_MIN_STEPS_BETWEEN >= 300 * 24


def test_adr_touches_only_the_four_declared_knobs():
    """ADR 이 조작하는 것은 목표 ranges · 스폰 pose_range · 질량 params ·
    **obs bias params** 넷뿐이어야 한다. 보상·액션·마찰(startup 버킷)은 금지."""
    cur = _src("v2_curriculum.py")
    assert 'get_term("object_pose")' in cur
    assert 'get_term_cfg("reset_object_position")' in cur
    assert 'get_term_cfg("dr_cup_mass")' in cur
    assert 'get_term_cfg("dr_obs_bias")' in cur
    for forbidden in ("rewards", "actions.", "friction", "P_STILL", "P_UPRIGHT"):
        assert forbidden not in cur, f"ADR 이 선언 밖을 만진다: {forbidden}"


def test_adr_goal_x_is_frozen_and_bounds_are_verified():
    """목표 x 는 안 넓힌다(상한 0.46 = 판 앞모서리 10 mm 앞) · z 상한은 도달 지도
    검증 구간("0.47 이상 21/21") 안이어야 한다."""
    assert P.ADR_GOAL_JITTER_MAX[0] == P.GOAL_JITTER_V2[0], "x 는 동결"
    assert P.ADR_GOAL_JITTER_MAX[1] > P.GOAL_JITTER_V2[1], "y 가 주 확장축(pour 요구)"
    # z 상한: GOAL_MEASURED=0(구 오프셋) 기준 중심 0.447 + jz
    z_hi = (V1.GOAL_POINT[2] + P.GRASP_OFFSET_ROOT_V1[2]) + P.ADR_GOAL_JITTER_MAX[2]
    assert z_hi <= 0.52, f"목표 z 상한 {z_hi:.3f} — 검증 구간을 크게 벗어난다"
    # ★08.30 라운드 13 — 스폰은 ± 오프셋이 아니라 **절대 상자**다(P1 실측 봉투).
    #   구 계약("±0.02 만 검증됨")은 실측 전 가정이었고, 이제 봉투가 근거다.
    xl, xh, yl, yh = P.ADR_SPAWN_BOX_MAX
    assert xh <= 0.410, f"만렙 스폰 상한 {xh} — 실측 벽(x≈0.41, ① 100%) 밖"
    # ★08.31 갱신 — 구 계약은 하한 0.340 이었다(당시 격자가 0.344 부터라 미검증).
    #   재실측 격자는 x0.330 을 포함하고 ① 파지실패 0%(y 0.122~0.338 전 구간)라
    #   근거가 생겼다. 격자 하단(0.320)까지만 허용한다 — 그 밖은 여전히 미검증.
    assert xl >= 0.320, f"만렙 스폰 하한 {xl} — 실측 격자 밖(미검증 구간)"
    # ★08.31 갱신 — 재실측 격자는 y 0.100~0.360 을 덮고, y 0.122 행이 x 0.330~0.390
    #   전 구간 ① 0% 다. 구 하한 0.150 은 좁은 격자 시절의 값이라 폐기한다.
    assert yl >= 0.100 and yh <= 0.300, f"y 봉투 [{yl},{yh}] 가 실측 격자 밖"


def test_friction_dr_is_static_and_consistent():
    """마찰은 startup 정적 DR — `make_consistent` 로 동마찰 ≤ 정마찰을 보장해야
    물리가 유효하다. 현재 기본값이 조용한 0.5 라는 것이 이 DR 의 존재 이유다."""
    src = _src("v2_env_cfg.py")
    blk = src[src.index("dr_cup_friction"):src.index("dr_cup_mass")]
    assert 'mode="startup"' in blk
    assert '"make_consistent": True' in blk
    assert P.DR_FRICTION_DYNAMIC[1] <= P.DR_FRICTION_STATIC[1]
    assert P.DR_FRICTION_STATIC[0] <= 0.5 <= P.DR_FRICTION_STATIC[1], (
        "범위가 현행 기본값(0.5)을 포함해야 분포 이동이 아니라 확장이 된다")


def test_obs_noise_is_policy_eye_only_and_bias_is_episodic():
    """★obs 노이즈는 **정책의 눈에만** 낀다 — 보상·판정은 ground truth.

    그리고 bias 는 에피소드 고정(리셋 이벤트 재샘플)이어야 한다 — 실기 /cup_pose
    캘리브 오차(41 mm 실측)는 한 판 안에서 안 변하는 성질이라 Unoise(스텝 독립)로는
    표현이 안 된다(IsaacLab 기본 항 메모리).
    """
    ob = _src("v2_observations.py")
    body = ob[ob.index("def object_position_noisy"):ob.index("def goal_minus_cup")]
    assert "_v2_cup_obs_bias" in body and "step_noise" in body
    ev = _src("v2_events.py")
    assert "def resample_obs_bias" in ev and 'mode="reset"' in _src("v2_env_cfg.py")
    st = _src("v2_stages.py")
    assert "_v2_cup_obs_bias" not in st, "보상 경로에 obs bias 가 새면 채점이 오염된다"
    rw = _src("v2_rewards.py")
    assert "_v2_cup_obs_bias" not in rw
    # 레벨 0 = bias 0 (현행 동일), 만렙 bias 는 목표 반경의 절반 이하(과제 성립선)
    src = _src("v2_env_cfg.py")
    assert '"bias_range": 0.0' in src
    assert P.ADR_OBS_BIAS_MAX <= P.SETTLE_RADIUS * 0.5


# ---------------------------------------------------------------------------
# 라운드 17 — 홈 교체(J1·J4·J7) + 이송에서 끝나는 에피소드
# ---------------------------------------------------------------------------
def test_home_j147_uses_only_three_joints():
    """사용자 지시: j2·j3·j5·j6 은 정확히 0 이어야 한다."""
    home = P.LEFT_ARM_HOME_J147
    assert set(home) == {f"l_aj_{i}" for i in range(1, 8)}
    for jn in ("l_aj_2", "l_aj_3", "l_aj_5", "l_aj_6"):
        assert home[jn] == 0.0, f"{jn} 는 0 이어야 한다 (실제 {home[jn]})"
    for jn in ("l_aj_1", "l_aj_4", "l_aj_7"):
        assert home[jn] != 0.0, f"{jn} 는 자세를 만드는 축이라 0 이면 안 된다"


def test_home_j147_keeps_action_range_slack():
    """액션 반범위 0.5 rad 를 온전히 쓰려면 한계에서 0.5 rad 여유가 필요하다.

    ★j1 은 상한 1.3963 이라 0.2484 에서 여유 1.148 — 충분하다. 사용자가 처음 제안한
      0.9 는 여유 0.496 으로 아슬아슬했고, 도달성에서 기각됐다.
    """
    import xml.etree.ElementTree as ET
    urdf = _PKG.resolve().parents[5] / (
        "assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.urdf")
    if not urdf.exists():          # 자산 미배포 호스트에서는 건너뛴다
        return
    lim = {}
    for j in ET.parse(urdf).getroot().iter("joint"):
        n = j.get("name") or ""
        el = j.find("limit")
        if n.startswith("l_aj_") and el is not None:
            lim[n] = (float(el.get("lower")), float(el.get("upper")))
    for jn in ("l_aj_1", "l_aj_4", "l_aj_7"):
        q = P.LEFT_ARM_HOME_J147[jn]
        lo, hi = lim[jn]
        assert min(q - lo, hi - q) >= 0.5, (
            f"{jn}={q} 한계여유 {min(q - lo, hi - q):.4f} < 0.5 — 액션 범위가 잘린다")


def test_home_j147_j2_clipping_is_known_and_accepted():
    """★j2 를 0 으로 두면 **액션 범위가 위쪽으로 잘린다** — 알려진 대가다.

    `l_aj_2` 한계는 (-3.3161, +0.1745) 라 q=0 에서 위쪽 여유가 0.1745 rad 뿐이다.
    액션 반범위 0.5 의 35% 다. 현재 홈은 j2=-0.3757 로 이 한계를 피해 여유 0.550 을
    확보하고 있었다.

    ⚠ 그래도 이 홈을 쓰는 근거: 도달성 실측이 **이 클리핑을 포함한 상태**에서
      나왔다(프로브가 표본을 한계로 clamp 한다). 컵 30 mm + 물기자세 0.07% 로
      현재 홈(0.02%)의 3.5 배다. 즉 잘린 채로도 현재보다 낫다.
    """
    import xml.etree.ElementTree as ET
    urdf = _PKG.resolve().parents[5] / (
        "assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.urdf")
    if not urdf.exists():
        return
    for j in ET.parse(urdf).getroot().iter("joint"):
        if (j.get("name") or "") == "l_aj_2":
            el = j.find("limit")
            hi = float(el.get("upper"))
            assert abs(hi - 0.1745) < 1e-3, "j2 상한이 바뀌었다 — 클리핑 계산 재확인"
            assert P.LEFT_ARM_HOME_J147["l_aj_2"] == 0.0
            return


def test_dwell_end_is_truncation_not_termination():
    """★★이 계약이 깨지면 정책이 성공을 **회피**하도록 학습한다.

    진짜 종료(terminated)면 가치 부트스트랩이 끊겨, 목표 도달이 곧 남은 스텝 보상을
    포기하는 행위가 된다 — 목표 밖을 맴돌며 stage 3 보상을 빠는 쪽이 유리해진다.
    """
    src = _src("v2_env_cfg.py")
    i = src.index("self.terminations.goal_dwell")
    seg = src[i:i + 400]
    assert "time_out=True" in seg, "goal_dwell 은 반드시 truncation 이어야 한다"


def test_dwell_threshold_is_short_and_hold_ramp_can_follow():
    """이송에서 끝내는 과제라 체류 문턱은 짧아야 하고, hold 램프가 그보다 길면
    정책은 프리미엄 만점을 **한 번도 못 받는다**(램프가 종료보다 늦게 찬다)."""
    assert 1 <= P.EPISODE_DWELL_STEPS <= 30
    src = _src("v2_preset.py")
    assert "HDGP_V2_HOLD_RAMP" in src, "램프를 종료 문턱에 맞출 수 있어야 한다"
    assert "HDGP_V2_DWELL_STEPS" in src


def test_goal_dwell_counter_resets_on_break():
    """연속이어야 한다 — 반경을 들락거리는 진동이 문턱에 닿으면 안 된다."""
    src = _src("v2_terminations.py")
    assert "torch.zeros_like(self._dwell)" in src, "체류가 끊기면 0 으로 되돌려야 한다"
    assert "def reset" in src, "에피소드 경계에서 카운터를 비워야 한다"
    assert "settle_success" in src, "합격 판정은 보상과 같은 정의를 써야 한다"


def test_home_switches_are_off_by_default():
    """기존 체크포인트 재생을 위해 새 스위치는 기본 off 여야 한다."""
    src = _src("v2_env_cfg.py")
    for var in ("HDGP_V2_HOME_J147", "HDGP_V2_DWELL_END"):
        assert f'_os.environ.get("{var}", "0")' in src, f"{var} 기본값이 off 가 아니다"


def test_truncation_is_actually_bootstrapped():
    """★★`time_out=True` 만으로는 부족하다 — 알고리즘이 보정해야 의미가 있다.

    rl_games 는 `if self.value_bootstrap and 'time_outs' in infos` 일 때만
    `shaped_rewards += gamma * values * time_outs` 를 더한다(a2c_common:777).
    IsaacLab 래퍼는 `extras["time_outs"] = truncated` 를 항상 넘긴다(rl_games.py:284).
    따라서 `value_bootstrap: False` 면 truncation 이 **진짜 종료처럼** 취급되어
    목표 도달이 남은 스텝 보상을 버리는 행위가 된다.

    실측(08.31 라운드 17 A): stage 3 스텝당 최대 0.9 인데 100 스텝에서 끊기면
    150×0.9 ≈ 135 를 포기한다 — 에피소드 총보상(63~77)보다 크다.
    `diag_at_goal` 이 0.215 → 0.002 로 무너진 것이 이 구조와 정합했다.
    """
    yaml_txt = (_PKG / "config" / "agents" / "rl_games_ppo_v2_cfg.yaml").read_text(
        encoding="utf-8")
    line = [ln for ln in yaml_txt.splitlines()
            if ln.strip().startswith("value_bootstrap:")]
    assert line, "value_bootstrap 설정이 없다"
    assert line[0].split(":")[1].strip().lower() == "true", (
        "goal_dwell 종료를 쓰는 한 value_bootstrap 은 True 여야 한다 — "
        "끄면 정책이 성공을 회피하도록 학습한다")


# ---------------------------------------------------------------------------
# 라운드 18 — 경로 z 마진 (좌팔 보정 문서 §2-2)
# ---------------------------------------------------------------------------
def test_tip_floor_is_compatible_with_grasping():
    """★파지와 역방향인 항이라, 하한이 파지 대역 **안**에 있어야 양립한다.

    컵 파지 대역은 판 위 `GRASP_HEIGHT_BAND` = (10, 85) mm 다. 하한을 대역 상단
    가까이 올리면 잡을 수 있는 높이가 사라진다. 20 mm 면 하단 10 mm 만 포기하고
    65 mm 가 남는다(대역의 76%).
    """
    lo, hi = V1.GRASP_HEIGHT_BAND
    assert lo < P.TIP_FLOOR_MARGIN < hi, (
        f"하한 {P.TIP_FLOOR_MARGIN} 이 파지 대역 ({lo}, {hi}) 밖이다")
    remain = hi - P.TIP_FLOOR_MARGIN
    assert remain >= 0.5 * (hi - lo), (
        f"파지 대역이 {remain * 1e3:.0f} mm 만 남는다 — 절반 미만이면 파지가 위태롭다")


def test_tip_floor_weight_stays_under_one_stair():
    """계단 밖 가산·감산항은 한 칸(0.25)보다 작아야 계단 골격이 안 깨진다."""
    ratio = abs(P.TIP_FLOOR_WEIGHT) / P.STAIRCASE_WEIGHT
    assert ratio < 0.25, f"벌점 최대 기여 {ratio:.3f} 가 계단 한 칸 0.25 이상이다"
    assert P.TIP_FLOOR_WEIGHT < 0.0, "마진 위반은 벌점이어야 한다"


def test_tip_floor_is_zero_above_margin():
    """하한 위에서 정확히 0 이어야 파지 구간에 gradient 를 안 만든다."""
    src = _src("v2_rewards.py")
    i = src.index("def tip_floor_penalty")
    seg = src[i:i + 900]
    assert ".clamp(0.0, 1.0)" in seg, "힌지는 [0,1] 로 잘려야 한다(위=0, 아래=포화)"
    assert "TIP_FLOOR_MARGIN - h" in seg, "마진 위에서 음수가 되어 0 으로 잘려야 한다"


def test_tip_floor_uses_lowest_point_not_one_body():
    """긁는 것은 **가장 낮은** 부위다 — 한 점만 보면 다른 부위가 긁어도 못 잡는다."""
    src = _src("v2_rewards.py")
    i = src.index("def _tip_height")
    seg = src[i:i + 700]
    assert "min(dim=1)" in seg and "minimum" in seg, "턱 링크들과 TCP 의 최소를 써야 한다"


# ---------------------------------------------------------------------------
# 라운드 20 — 접근 자세 제약 (형상 비의존)
# ---------------------------------------------------------------------------
def test_approach_dir_peaks_at_horizontal_not_at_vertical():
    """★순수 벌점판이 실패한 지점 — 90° 이하가 전부 같은 값이면 정책은 **위로 세워서**
    제약을 피하고 파지를 포기한다(F 실측: 접근각 17.5° · lift 0.002).
    보상은 **수평(a_z=0)에서 최대**여야 하고 위로 세우면 BASE 만 받아야 한다.
    """
    src = _src("v2_rewards.py")
    i = src.index("def approach_dir_bonus")
    seg = src[i:i + 1400]
    assert "torch.exp(-((az / P.APPROACH_DIR_SIGMA) ** 2))" in seg, "수평에서 최대인 봉우리여야 한다"
    assert "az < 0.0" in seg, "아래로 기울면(90° 초과) 0 이어야 한다"
    assert P.APPROACH_DIR_BASE < 0.05, (
        f"BASE {P.APPROACH_DIR_BASE} 가 크면 '위로 세우기'가 다시 값어치를 갖는다")


def test_approach_dir_is_multiplied_by_reach_progress():
    """★"수평으로 서서 안 잡기"를 막는 유일한 장치다 — 접근 진행도를 곱해야
    컵 근처까지 가야만 보너스가 생긴다."""
    src = _src("v2_rewards.py")
    i = src.index("def approach_dir_bonus")
    seg = src[i:i + 1400]
    assert "S.stage_reach(" in seg, "접근 진행도를 곱하지 않으면 제자리 해킹이 성립한다"
    assert "~held" in seg, "파지 후에는 0 이어야 한다(무는 동작·리프트 면제)"


def test_approach_dir_weight_is_positive_and_bounded():
    assert P.APPROACH_DIR_WEIGHT > 0.0, "방향은 벌점이 아니라 보너스여야 한다"
    assert abs(P.APPROACH_DIR_WEIGHT) / P.STAIRCASE_WEIGHT < 0.25


def test_approach_penalty_is_zero_at_or_above_horizontal():
    """90° 이하(수평·위쪽)에서 **정확히 0** 이어야 한다.

    그래야 "수평으로 접근하라"가 아니라 "아래로 기울지 마라"가 된다 — 수평 접근에
    gradient 를 주면 정책이 필요 이상으로 손목을 세워 파지가 어려워진다.
    """
    src = _src("v2_rewards.py")
    i = src.index("def approach_tilt_penalty")
    seg = src[i:i + 800]
    assert "(-_approach_az(env)).clamp(0.0, 1.0)" in seg, (
        "a_z ≥ 0 구간이 0 으로 잘려야 한다")


def test_approach_penalty_is_gated_off_after_grasp():
    """무는 동작(j7 을 드는 것)과 리프트·이송은 면제되어야 한다."""
    src = _src("v2_rewards.py")
    i = src.index("def approach_tilt_penalty")
    seg = src[i:i + 800]
    assert "stage_close" in seg and "~held" in seg, "파지 후에는 0 이어야 한다"


def test_approach_weight_stays_under_one_stair():
    ratio = abs(P.APPROACH_AZ_WEIGHT) / P.STAIRCASE_WEIGHT
    assert ratio < 0.25, f"최대 기여 {ratio:.3f} 가 계단 한 칸 0.25 이상이다"
    assert P.APPROACH_AZ_WEIGHT < 0.0, "아래로 기욺은 벌점이어야 한다"


def test_approach_constraint_is_shape_agnostic():
    """★파지 대역(`GRASP_HEIGHT_BAND`)은 컵이 바뀌면 무의미해지는 **형상 상수**다.
    접근 자세 제약은 거기 의존하면 안 된다 — 다물체로 갈 때 같이 죽는다."""
    src = _src("v2_rewards.py")
    i = src.index("def _approach_az")
    seg = src[i:src.index("def diag_approach_down")]
    for forbidden in ("GRASP_HEIGHT_BAND", "TABLE_SURFACE_Z", "CUP_"):
        assert forbidden not in seg, f"{forbidden} 에 의존하면 형상 비의존이 깨진다"


def test_approach_dir_is_pbrs_not_level():
    """★레벨 지급은 실패했다 — 각도는 90.1° 로 내려갔지만 정책이 그 자세로 250 스텝 중
    **239 스텝을 떠 있기만** 하고 안 잡았다(⑤ 0.0%). 에피소드가 목표 도달 10 스텝에
    끝나므로 **버티기가 완수와 값이 같아진다.** 차분으로 줘야 그 경로가 사라진다.
    """
    src = _src("v2_rewards.py")
    assert "class ApproachDirPBRS" in src, "차분 지급 클래스가 있어야 한다"
    i = src.index("class ApproachDirPBRS")
    seg = src[i:]
    assert "out = phi - self._prev" in seg, "레벨이 아니라 차분을 지급해야 한다"
    assert "def reset" in seg, "에피소드 경계에서 Φ 를 비워야 한다"
    assert "torch.where(held, self._prev, phi)" in seg, (
        "파지 후 Φ 동결 — 게이트로 0 을 만들면 파지하는 순간 −Φ 절벽이 생겨 파지를 벌한다")


def test_approach_weight_below_stall_breakeven():
    """★정체 손익분기를 넘기면 "수평으로 떠서 버티기"가 완수와 값이 같아진다.

        버티기 = 250 스텝 × w × dt = 5w
        완수   = 약 60 스텝 × 계단 0.6 ≈ 36   (목표 체류 10 스텝에 에피소드가 끝난다)
        ⇒ 손익분기 w ≈ 7.2.  w=7.0 으로 돌렸다가 ⑤ 0.0% 를 봤다(G 실측).
    """
    breakeven = 36.0 / (250 * 0.02)
    assert P.APPROACH_DIR_WEIGHT < 0.5 * breakeven, (
        f"w={P.APPROACH_DIR_WEIGHT} 가 손익분기 {breakeven:.1f} 에 너무 가깝다")


def test_pbrs_variant_is_kept_but_not_wired():
    """★차분판은 **정의상 최적 정책을 못 바꾼다**(policy invariance). 실측도 그랬다 —
    H(PBRS) 접근 각도 118.7° = B(무개입) 117.5°. 행동을 바꾸려면 레벨이어야 한다.
    코드는 근거로 남기되 배선하지 않는다."""
    src = _src("v2_rewards.py")
    assert "class ApproachDirPBRS" in src, "기각 근거로 남겨둔다"
    cfg = _src("v2_env_cfg.py")
    i = cfg.index("self.rewards.approach_dir")
    assert "R.approach_dir_bonus" in cfg[i:i + 300], "레벨판이 배선돼 있어야 한다"


# ===========================================================================
# 라운드 22 · Part 1 — 곱셈형 방향 (`HDGP_V2_DIRMUL`)
# ===========================================================================
def test_dirq_has_gradient_below_horizontal():
    """★★I2(라운드 21) 실패의 회귀 방지 — **가장 중요한 계약**.

    지시받은 모양("90° 초과면 0")을 그대로 쓰면 `az<0` 이 값도 기울기도 0 인 평지가
    된다. 정책이 110° 에 자리잡은 순간 되돌아올 신호가 사라져 **1000 epoch 동안
    각도가 미동도 안 했다**(수령률 상한의 0.05%). 하향 구간은 반드시 0 이 아니어야
    한다.
    """
    import math
    sg = P.APPROACH_DIR_SIGMA_DN
    q = lambda deg: math.exp(-((math.cos(math.radians(deg)) / sg) ** 2))
    for deg, lo in ((103.9, 0.30), (110.0, 0.15), (117.5, 0.05)):
        assert q(deg) > lo, (
            f"{deg}° 에서 방향 품질 {q(deg):.3f} — 기울기가 사실상 죽었다(I2 재현)")
    assert q(103.9) > q(110.0) > q(117.5), "아래로 갈수록 단조 감소해야 한다"


def test_dirq_is_full_marks_at_or_above_horizontal():
    """사용자 지시의 핵심 — **90° 이하는 만점**이다. 위로 더 세운다고 더 받지 않는다
    (그러면 '위로 세우고 안 잡기'가 값어치를 갖는다 — F 실측의 실패)."""
    src = _src("v2_rewards.py")
    i = src.index("def _approach_dirq")
    seg = src[i:i + 1400]
    assert "torch.where(az >= 0.0, torch.ones_like(az), below)" in seg, (
        "90° 이하는 평지 1.0 이어야 한다")


def test_dirmul_multiplier_never_creates_a_stall():
    """★G(w=7.0) 가 만든 함정의 구조적 차단.

    배수가 1 미만이면 stage 0 천장(0.25)이 stage 1 바닥(mult×0.25)보다 커져
    **나쁜 각도의 파지가 손해**가 된다 — 정확히 G 의 −11.1 이다.
    가산 배수(1 이상)여야 경계에 절벽이 안 생긴다.
    """
    assert P.DIRMUL_GAIN > 0.0
    src = _src("v2_rewards.py")
    i = src.index("if dirmul_gain > 0.0:")
    seg = src[i:i + 800]
    assert "r * (1.0 + dirmul_gain * self.dir_quality())" in seg, (
        "배수는 (1 + gain×품질) 이어야 한다 — 감액형은 파지 절벽을 만든다")
    stage0_ceiling = 1.0 / P.N_STAGES
    stage1_floor = (1.0 + P.DIRMUL_GAIN * 0.0) * (1.0 / P.N_STAGES)
    assert stage1_floor >= stage0_ceiling, "파지 순간에 보상이 떨어지면 안 된다"


def test_dirmul_is_paid_only_after_grasp():
    """머물기로는 못 딴다 — stage 0 에서는 배수가 안 붙어야 정체 유인이 안 생긴다.
    (레벨 지급 G 는 파지 전에 지급해서 250 스텝 정체를 만들었다.)"""
    src = _src("v2_rewards.py")
    i = src.index("if dirmul_gain > 0.0:")
    seg = src[i:i + 800]
    assert "pre = idx < 1.0" in seg
    assert "torch.where(pre, r, r *" in seg, "파지 전에는 원래 계단 그대로여야 한다"


def test_dirmul_uses_pre_grasp_average_not_instant_value():
    """★순간값이면 '파지 직전 한 스텝만 수평'으로 만점을 딸 수 있다 — 접근 내내
    테이블을 긁고도 보너스를 챙기는 해킹면이다. 접근 구간 **평균**이어야 한다."""
    src = _src("v2_rewards.py")
    i = src.index("if dirmul_gain > 0.0:")
    seg = src[i:i + 800]
    assert "self._dir_sum + _approach_dirq(env)" in seg
    assert "self._dir_cnt + 1.0" in seg
    src2 = src[src.index("def dir_quality"):][:400]
    assert "self._dir_sum / self._dir_cnt.clamp(min=1.0)" in src2


def test_dirmul_state_resets_per_episode():
    """에피소드 경계에서 안 지우면 이전 에피소드의 접근 품질이 새 판에 새어든다."""
    src = _src("v2_rewards.py")
    i = src.index("    def reset(self, env_ids=None):")
    seg = src[i:i + 900]
    for tok in ("self._dir_sum[:] = 0.0", "self._dir_cnt[:] = 0.0",
                "self._dir_sum[env_ids] = 0.0", "self._dir_cnt[env_ids] = 0.0"):
        assert tok in seg, f"reset 에 {tok} 이 없다"


def test_level_home_is_env_gated_and_fresh_only():
    """★홈 교체는 액션 상자를 통째로 옮긴다 — 기본 꺼짐이어야 기존 체크포인트
    (B ep800 · E19 ep1800)의 재생 경로가 살아 있다."""
    src = _src("v2_env_cfg.py")
    assert '_os.environ.get("HDGP_V2_HOME_LEVEL", "0") == "1"' in src, "기본 꺼짐"
    assert "P.LEFT_ARM_HOME_LEVEL" in src


def test_home_low_is_lower_and_less_tilted_than_home_level():
    """★A94(HOME_LEVEL) 선정 실패의 회귀 방지.

    각도만 보고 홈을 고르면 TCP 가 파지점보다 80mm 위로 떠서, 정책이 잡으러
    내려가며 접근각이 도로 107° 로 돌아간다(K22 실측). B100 은 **높이도** 낮다.
    """
    assert set(P.LEFT_ARM_HOME_LOW) == set(P.LEFT_ARM_HOME_LEVEL)
    src = _src("v2_env_cfg.py")
    assert '_os.environ.get("HDGP_V2_HOME_LOW", "0") == "1"' in src, "기본 꺼짐"
    assert "P.LEFT_ARM_HOME_LOW" in src
    # j4(팔꿈치)를 더 접고 j7(손목)을 더 젖혀 TCP 를 낮춘다
    assert P.LEFT_ARM_HOME_LOW["l_aj_4"] < P.LEFT_ARM_HOME_LEVEL["l_aj_4"]
    assert P.LEFT_ARM_HOME_LOW["l_aj_7"] < P.LEFT_ARM_HOME_LEVEL["l_aj_7"]


def test_home_mid_keeps_the_working_home_height():
    """★B100 실패의 회귀 방지 — 홈을 낮추면 컵이 램프(+6mm)에 못 닿아 리프트가
    통째로 죽는다(P22: grasp_ok 0.765 인데 컵 +4.9mm, ② 파지만 99.2%).
    C96 은 구 홈 높이를 지키므로 j4 가 B100 보다 커야 한다(팔을 덜 접는다)."""
    assert P.LEFT_ARM_HOME_MID["l_aj_4"] > P.LEFT_ARM_HOME_LOW["l_aj_4"]
    src = _src("v2_env_cfg.py")
    assert '_os.environ.get("HDGP_V2_HOME_MID", "0") == "1"' in src, "기본 꺼짐"
    assert "P.LEFT_ARM_HOME_MID" in src


def test_only_one_home_switch_is_meant_to_be_on():
    """홈 스위치 4 종은 배타적이다 — 프로브·재생 때 학습과 같은 것만 켜야 한다."""
    src = _src("v2_env_cfg.py")
    for name in ("HDGP_V2_HOME_J147", "HDGP_V2_HOME_LEVEL",
                 "HDGP_V2_HOME_LOW", "HDGP_V2_HOME_MID"):
        assert f'_os.environ.get("{name}", "0") == "1"' in src


def test_lift_ramp_low_is_env_gated_and_still_needs_a_real_grasp():
    """★P22 의 1.1mm 함정을 겨냥한 스위치. 기본 꺼짐이어야 기존 판과 비교가 선다.

    그리고 `stage_lift = stage_close × move_up` 구조가 유지돼야 한다 — 안 그러면
    "안 잡고 컵만 기울이기"가 리프트로 지급되는 해킹면이 열린다.
    """
    src = _src("v2_preset.py")
    assert '_os.environ.get("HDGP_V2_LIFT_RAMP_LOW", "0") == "1"' in src, "기본 꺼짐"
    st = _src("v2_stages.py")
    i = st.index("def stage_lift")
    assert "stage_close(env, jaw_cfg, object_cfg) * move_up" in st[i:i + 900], (
        "리프트는 폐쇄 성공을 곱해야 한다")


def test_action_rate_weight_is_env_overridable_and_defaults_small():
    """★기본값은 작게 유지한다(과제 형성 중에는 억제가 탐색을 죽인다).
    다만 env 로 열어 둬 **과제가 선 뒤 웜스타트에서** 키울 수 있어야 한다 —
    R22 실측으로 σ 오염 근거가 무효화됐기 때문이다(원값 0.080, σ≈1 이면 2.0)."""
    src = _src("v2_preset.py")
    assert '_os.environ.get("HDGP_V2_ARATE_W", "")' in src
    assert "ACTION_RATE_WEIGHT = -abs(float(_arate_w))" in src, "부호는 항상 음수여야 한다"
    import os as _o
    assert not _o.environ.get("HDGP_V2_ARATE_W"), "테스트 환경에선 기본값이어야 한다"
    assert P.ACTION_RATE_WEIGHT == -0.002, "기본값은 작게"


def test_action_rate_curriculum_is_a_step_at_a_given_epoch():
    """사용자 설계 — 처음엔 약하게, 지정 epoch 에서 목표치로 올린다.

    ★`common_step_counter` 는 env 스텝이므로 **epoch × horizon_length** 여야 한다.
    (라운드 10 에 같은 단위 오독으로 ADR 게이트를 17배로 잡은 전례가 있다.)
    """
    assert P.ACTION_RATE_AT_STEPS == P.ACTION_RATE_AT_EPOCH * P.RL_HORIZON_LENGTH
    assert P.RL_HORIZON_LENGTH == 24, "yaml 의 horizon_length 와 일치해야 한다"
    assert P.ACTION_RATE_TARGET < 0.0, "억제 항이므로 음수"
    assert abs(P.ACTION_RATE_TARGET) > abs(P.ACTION_RATE_WEIGHT), "올라가야 한다"
    src = _src("v2_env_cfg.py")
    i = src.index("if P.ACTION_RATE_CURR:")
    seg = src[i:i + 600]
    assert "mdp.modify_reward_weight" in seg
    assert '"term_name": "action_rate"' in seg
    assert "P.ACTION_RATE_AT_STEPS" in seg


def test_vendor_gains_match_the_vendor_yaml_and_are_env_gated():
    """★실기 벤더 파일이 진실이다(R2S §1). 사본 3곳 md5 일치를 확인했고, 그 값과
    코드 상수가 어긋나면 sim 이 다른 로봇을 흉내내게 된다."""
    assert P.LEFT_ARM_VENDOR_STIFFNESS == {
        "l_aj_1": 70.0, "l_aj_2": 70.0, "l_aj_3": 70.0, "l_aj_4": 60.0,
        "l_aj_5": 10.0, "l_aj_6": 10.0, "l_aj_7": 10.0}
    assert P.LEFT_ARM_VENDOR_DAMPING == {
        "l_aj_1": 2.75, "l_aj_2": 2.50, "l_aj_3": 2.00, "l_aj_4": 2.00,
        "l_aj_5": 0.70, "l_aj_6": 0.60, "l_aj_7": 0.50}
    src = _src("v2_env_cfg.py")
    assert '_os.environ.get("HDGP_V2_VENDOR_GAINS", "0") == "1"' in src, "기본 꺼짐"
    assert 'actuators["left_arm"].stiffness' in src


def test_vendor_gains_are_much_softer_than_the_sim_default():
    """★이 격차가 '제어기가 정책 진동을 그대로 따라가는' 원인이다 —
    기록으로 남겨 다음 사람이 배율을 다시 재지 않게 한다."""
    from openarm.gripper.left.grasp_sensor import grasp_left_preset as V1
    assert V1.ARM_IK_STIFFNESS["l_aj_[1-4]"] / P.LEFT_ARM_VENDOR_STIFFNESS["l_aj_1"] > 4.0
    assert V1.ARM_IK_DAMPING["l_aj_7"] / P.LEFT_ARM_VENDOR_DAMPING["l_aj_7"] > 20.0


def test_kd_bracket_requires_vendor_gains_and_straddles_it():
    """★좌팔 여진 측정이 없어 kd 진짜 값을 모른다 — 벤더 kd(덜 감쇠)와 R2S 산출
    kd(과감쇠)로 **양 끝을 잡는다**. 두 판의 차이가 작으면 그 자체가 결론이다."""
    src = _src("v2_env_cfg.py")
    assert '_os.environ.get("HDGP_V2_KD_R2S", "0") == "1"' in src, "기본 꺼짐"
    i = src.index("if self.v2_vendor_gains:")
    seg = src[i:i + 900]
    assert "if self.v2_kd_r2s:" in seg, "VENDOR_GAINS 블록 안에 있어야 한다(kp 는 벤더 고정)"
    # j1~j5 는 R2S 쪽이 더 크고(과감쇠), 손목 j6·j7 은 더 작다 — 배율이 제각각이라는 기록
    for j in ("l_aj_1", "l_aj_2", "l_aj_3", "l_aj_4", "l_aj_5"):
        assert P.LEFT_ARM_R2S_DAMPING[j] > P.LEFT_ARM_VENDOR_DAMPING[j]
    for j in ("l_aj_6", "l_aj_7"):
        assert P.LEFT_ARM_R2S_DAMPING[j] < P.LEFT_ARM_VENDOR_DAMPING[j]


def test_tip_floor_margin_and_weight_are_env_overridable():
    """★Y24 실측 — 이 항이 작동점에서 죽어 있었다(기여 0.67%, 힌지 원값 0.017).
    마진이 병목이므로 env 로 열어 스윕할 수 있어야 한다. 기본값은 유지."""
    src = _src("v2_preset.py")
    assert '_os.environ.get("HDGP_V2_TIP_MARGIN", "")' in src
    assert '_os.environ.get("HDGP_V2_TIP_W", "")' in src
    assert "TIP_FLOOR_WEIGHT = -abs(float(_tf_w))" in src, "부호는 항상 음수"
    import os as _o
    if not _o.environ.get("HDGP_V2_TIP_MARGIN"):
        assert P.TIP_FLOOR_MARGIN == 0.020, "기본 마진 유지"
    if not _o.environ.get("HDGP_V2_TIP_W"):
        assert P.TIP_FLOOR_WEIGHT == -6.0, "기본 가중치 유지"


def test_tip_floor_margin_stays_within_the_measured_graspable_band():
    """★이 항은 파지와 역방향이다. 실측상 손끝 중앙 27.3mm(R22, ⑤100%)·
    31.8mm(N22, ⑤97.1%)는 파지와 양립했다 — 마진을 그 범위 밖(≥40mm)으로 올리면
    파지 자체를 벌하게 된다. 상한을 계약으로 고정한다."""
    assert P.TIP_FLOOR_MARGIN <= 0.035, (
        f"마진 {P.TIP_FLOOR_MARGIN*1000:.0f}mm 는 실측 파지 대역(27~32mm)을 넘는다")
