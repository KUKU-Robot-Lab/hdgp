#!/usr/bin/env python3
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

"""v2 계단 보상의 **지형**을 Isaac 없이 스캔한다 — 학습 기동 전 필수 관문.

★★왜 이 도구가 필요한가. 라운드 1 의 R1(순변위 `still`)은 reward-audit 5 체크를
  통과했는데도 **정반대로 작동했다** — 제자리 왕복이 순변위 0 을 만들어 점수를 벌었고,
  결정론 프로브에서 지령 반전 0.670 · 직진 효율 0.215 로 baseline 보다 더 심하게
  떨었다. 4000 epoch × 2 GPU 를 쓰고 나서야 알았다.
  그 오류는 **보상식만 격자로 스캔해도** 학습 없이 드러난다. 이 스크립트가 그 관문이다.

무엇을 재는가:
  ① 연속성 — 세 계단 경계에서 `r` 의 점프
  ② 단조성 — 목표로 접근할 때 `r` 이 단조 증가하는가
  ③ 고원   — `∂r/∂dist ≈ 0` 인 구간이 있는가 (라운드 1·2 붕괴의 직접 원인)
  ④ 해킹면 — "목표 밖에서 멈추기" 가 "목표 도달" 보다 싼가
  ⑤ 구·신 지형 나란히 — 무엇이 실제로 달라졌는지

⚠ 씬이 필요 없다. `v2_stages` 의 순수 함수(`d_shape`·`smoothstep`)만 쓰고 나머지
  (`r_close`·`move_up`)는 물리 상수에서 해석적으로 만든다. Isaac 이 없는 호스트에서도
  돈다 — `pytest` 계약과 이 스크립트가 학습 전 방어선의 전부다.

사용:
    python3 scripts/probes/probe_v2_reward_terrain.py
    python3 scripts/probes/probe_v2_reward_terrain.py --still-net   # arm B 조건
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "source" / "openarm"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openarm.gripper.left.grasp_sensor_v2 import v2_preset as P  # noqa: E402

# ---------------------------------------------------------------------------
# 순수 파이썬 복제 — Isaac 없이 돌리기 위함. 계약 테스트가 원본과의 일치를 잠근다.
# ---------------------------------------------------------------------------
K = P.D_SHAPE_K


def d_shape(d: float, s: float, tau: float = 0.0) -> float:
    if tau > 0.0 and d < tau:
        return 1.0
    q = math.tanh(d * K / s)
    return 1.0 - q * q


def smoothstep(x: float, far: float, near: float) -> float:
    t = (x - far) / (near - far)
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def move_up(cup_z: float) -> float:
    """리프트 램프. 스폰 +6 mm 에서 0, +40 mm 에서 1."""
    t = (cup_z - P.LIFT_RAMP_ZERO_Z) / (P.MINIMAL_LIFT_HEIGHT - P.LIFT_RAMP_ZERO_Z)
    return min(1.0, max(0.0, t))


def reward_new(dist: float, cup_z: float, speed: float,
               upright: float = 1.0, r_close: float = 1.0,
               r_grasp: float = 0.9) -> tuple[float, int, float]:
    """라운드 3 지형. `(r, idx, v)`."""
    r_lift = r_close * move_up(cup_z)
    r_transport = r_lift * d_shape(dist, P.TRANSPORT_S, P.TRANSPORT_TAU)
    ok3 = float(dist < P.SETTLE_RADIUS and r_close > P.STAGE3_GRASP_MIN)

    idx = 0
    if r_lift > P.STAGE_THRESHOLD:
        idx = 1
    if r_transport > P.STAGE_THRESHOLD:
        idx = 2
    if ok3 > P.STAGE_THRESHOLD:
        idx = 3

    v3 = smoothstep(speed, *P.P_STILL_BAND) * smoothstep(upright, *P.P_UPRIGHT_BAND)
    v = (r_grasp, r_lift, r_transport, v3)[idx]
    return (idx + v) / P.N_STAGES, idx, v


def reward_old(dist: float, cup_z: float, speed: float,
               upright: float = 1.0, r_close: float = 1.0,
               r_grasp: float = 0.9) -> tuple[float, int, float]:
    """run 0 지형 (붕괴한 쪽). 비교 대조용."""
    r_lift = r_close * move_up(cup_z)
    r_transport = r_lift * d_shape(dist, P.TRANSPORT_S, P.TRANSPORT_TAU)
    at_goal = float(dist < P.SETTLE_RADIUS)
    still = d_shape(speed, 0.15)                       # 구 SETTLE_VEL_S
    r_settle = r_transport * at_goal * still * 0.5 * (1.0 + upright)

    idx = 0
    for k, q in ((1, r_lift), (2, r_transport), (3, r_settle)):
        if q > P.STAGE_THRESHOLD:
            idx = k
    v = (r_grasp, r_lift, r_transport, r_settle)[idx]
    return (idx + v) / P.N_STAGES, idx, v


# ---------------------------------------------------------------------------
# 기하 — 스폰에서 목표까지의 실제 경로
# ---------------------------------------------------------------------------
SPAWN = (P.CUP_SPAWN_X_CENTER, P.CUP_SPAWN_Y_CENTER, P.CUP_SPAWN_Z)
GOAL = tuple(P.GOAL_POINT[i] + P.GRASP_OFFSET_ROOT[i] for i in range(3))


def path_point(t: float) -> tuple[float, float]:
    """스폰(t=0) → 목표(t=1) 직선 위의 점 ⇒ `(dist, cup_z)`."""
    p = [SPAWN[i] + t * (GOAL[i] - SPAWN[i]) for i in range(3)]
    return math.dist(p, GOAL), p[2]


def _fmt(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


# ---------------------------------------------------------------------------
def section_geometry() -> None:
    print("\n" + "=" * 78)
    print("① 기하 — 이 과제의 '이송'이 무엇인가")
    print("=" * 78)
    d = [GOAL[i] - SPAWN[i] for i in range(3)]
    tot = math.dist(SPAWN, GOAL)
    print(f"  스폰 {tuple(round(x, 4) for x in SPAWN)} → 목표 {tuple(round(x, 4) for x in GOAL)}")
    print(f"  성분 dx {d[0]*1000:+6.1f} · dy {d[1]*1000:+6.1f} · dz {d[2]*1000:+6.1f} mm"
          f"   총 {tot*1000:.1f} mm")
    print(f"  z 비중 {abs(d[2])/tot*100:.1f}%  ⇒ 이 과제의 이송은 사실상 **수직**이다")
    span = P.MINIMAL_LIFT_HEIGHT - P.LIFT_RAMP_ZERO_Z
    print(f"  리프트 램프 span {span*1000:.1f} mm = 요구 z 의 {span/abs(d[2])*100:.1f}%"
          f"   ⇒ 램프만으로는 나머지 {(abs(d[2])-span)*1000:.0f} mm 에 신호가 없다")


def section_continuity(fn, name: str) -> list[str]:
    """② 연속성 — 경계를 미세하게 넘나들며 점프를 잰다."""
    print("\n" + "=" * 78)
    print(f"② 연속성 — 계단 경계의 점프  [{name}]")
    print("=" * 78)
    fails = []
    # 경로를 촘촘히 훑으며 idx 가 바뀌는 지점의 r 차이를 본다.
    N = 20000
    prev = None
    for i in range(N + 1):
        t = i / N
        dist, z = path_point(t)
        r, idx, _ = fn(dist, z, speed=0.0)
        if prev is not None and idx != prev[1]:
            jump = r - prev[0]
            tag = "OK " if jump > -1e-9 else "★하향"
            print(f"  stage {prev[1]} → {idx}  @ dist {dist*1000:6.1f} mm  "
                  f"z {z:.4f}   r {_fmt(prev[0])} → {_fmt(r)}   점프 {jump:+.4f}  {tag}")
            if jump <= -1e-9:
                fails.append(f"{name}: stage {prev[1]}→{idx} **하향** 점프 {jump:+.4f}")
        prev = (r, idx)
    if not fails:
        print("  ⇒ 하향 점프 없음 ✓  (위쪽 점프는 전진 보너스이지 결함이 아니다 —")
        print("     baseline 은 +0.050 / +0.099 점프로도 결정론 도달 93.7% 를 낸다)")
    return fails


def section_plateau(fn, name: str) -> list[str]:
    """③ 고원 — 경로를 따라가며 ∂r/∂진행 이 0 인 구간을 찾는다.

    ⚠ **stage 0 은 제외한다.** 그 단계의 값은 `r_grasp = r_reach × bonus` 이고, 구동
      변수는 TCP↔파지점 거리이지 컵의 경로 좌표가 아니다. 여기서는 컵을 이미 쥔 채
      경로를 훑으므로 stage 0 구간이 평탄해 보이는 것은 **시뮬레이션의 인공물**이다.
      고치려는 결함(D1)은 stage 1·2 안의 고원이다.
    """
    print("\n" + "=" * 78)
    print(f"③ 고원 탐지 — ∂r/∂(경로 진행) ≈ 0 인 구간  [{name}]")
    print("   ★stage 2 만 본다. stage 0·1 은 TCP 접근·상승이, stage 3 은 속도·직립이")
    print("     구동한다 — 컵 경로 좌표가 그 단계의 변수가 아니다.")
    print("   ★★D1 정정: 'stage 1 고원이 이송 실패의 원인'은 오진이었다. baseline 도")
    print("     같은 고원을 갖는데 결정론 도달 93.7% 를 낸다.")
    print("=" * 78)
    N = 400
    flat_runs, cur, start = [], 0, 0.0
    prev_r = None
    for i in range(N + 1):
        t = i / N
        dist, z = path_point(t)
        r, idx, _ = fn(dist, z, speed=0.0)
        if idx != 2:                              # ★stage 2 만 컵 경로가 구동한다
            prev_r, cur = None, 0
            continue
        if prev_r is not None:
            if abs(r - prev_r) < 1e-6:
                if cur == 0:
                    start = t
                cur += 1
            else:
                if cur >= 5:                      # 경로의 1% 이상 평탄
                    flat_runs.append((start, t))
                cur = 0
        prev_r = r
    if cur >= 5:
        flat_runs.append((start, 1.0))

    fails = []
    if not flat_runs:
        print("  고원 없음 — 경로 전 구간에 gradient 가 산다 ✓")
    for a, b in flat_runs:
        da, _ = path_point(a)
        db, _ = path_point(b)
        print(f"  ★고원  경로 {a:.2f}~{b:.2f}  (컵–목표 {da*1000:6.1f} → {db*1000:6.1f} mm)"
              f"   길이 {(b-a)*100:.0f}%")
        fails.append(f"{name}: 고원 {a:.2f}~{b:.2f}")
    return fails


def section_profile(fn, name: str) -> None:
    """경로 프로파일 표 — 눈으로 지형을 본다."""
    print("\n" + "=" * 78)
    print(f"④ 경로 프로파일 (스폰 → 목표, 정지 상태 가정)  [{name}]")
    print("=" * 78)
    print(f"  {'t':>5} {'dist_mm':>8} {'cup_z':>7} {'idx':>4} {'v':>7} {'r':>7} {'Δr':>8}")
    prev = None
    for i in range(0, 21):
        t = i / 20
        dist, z = path_point(t)
        r, idx, v = fn(dist, z, speed=0.0)
        d = "" if prev is None else f"{r - prev:+8.4f}"
        print(f"  {t:>5.2f} {dist*1000:>8.1f} {z:>7.4f} {idx:>4} {v:>7.4f} {r:>7.4f} {d:>8}")
        prev = r


def section_hacking(fn, name: str) -> list[str]:
    """⑤ 해킹면 — '목표 밖에서 멈추기' 가 '목표 도달' 보다 싼가."""
    print("\n" + "=" * 78)
    print(f"⑤ 해킹면 — 목표 밖 정지 vs 목표 도달  [{name}]")
    print("=" * 78)
    _, z_goal = path_point(1.0)
    r_reach, _, _ = fn(0.02, z_goal, speed=0.0)
    print(f"  {'상황':<34}{'r':>8}   {'목표도달 대비':>12}")
    print(f"  {'목표 도달 + 정지 (dist 20 mm)':<34}{r_reach:>8.4f}   {'—':>12}")
    fails = []
    for dmm in (150, 130, 110, 90, 70):
        t = 1.0
        # 그 거리에 해당하는 경로 지점을 찾는다(z 도 함께 따라간다)
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if path_point(mid)[0] > dmm / 1000.0:
                lo = mid
            else:
                hi = mid
        t = 0.5 * (lo + hi)
        dist, z = path_point(t)
        r, _, _ = fn(dist, z, speed=0.0)
        ratio = r / r_reach if r_reach > 0 else float("inf")
        tag = "" if ratio < 0.95 else "  ★해킹면"
        print(f"  {'목표 밖 ' + str(dmm) + ' mm 에서 완전 정지':<34}{r:>8.4f}   {ratio:>11.1%}{tag}")
        if ratio >= 0.95:
            fails.append(f"{name}: {dmm}mm 정지가 도달의 {ratio:.0%}")

    # 진동 해킹 — 왕복해도 순변위는 작다. 순간속도 판은 반환점에서 0 이 된다.
    print()
    dist, z = path_point(0.0)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if path_point(mid)[0] > 0.12:
            lo = mid
        else:
            hi = mid
    dist, z = path_point(0.5 * (lo + hi))
    vals = {}
    for spd, lab in ((0.0, "정지(또는 왕복 반환점)"), (0.25, "이동 중(0.25 m/s)")):
        r, idx, v = fn(dist, z, speed=spd)
        vals[lab] = r
        print(f"  dist 120 mm · {lab:<24} idx {idx}  v {v:.4f}  r {r:.4f}")
    vib = vals["정지(또는 왕복 반환점)"] - vals["이동 중(0.25 m/s)"]
    trans = r_reach - vals["이동 중(0.25 m/s)"]
    print(f"  진동 이득 {vib:+.4f}  vs  이송 이득 {trans:+.4f}"
          f"   (이송이 {trans/vib:.1f} 배)" if abs(vib) > 1e-9 else
          f"  진동 이득 {vib:+.4f} (속도 항 없음)")
    print("  ★A/B 예측: 이 '진동 이득' 은 **순간속도**(arm A)에서만 실현된다 — 왕복의")
    print("     반환점에서 순간속도가 0 이기 때문이다. **순변위**(arm B)는 왕복을 상쇄하므로")
    print("     같은 행동이 큰 speed 로 측정돼 이득이 사라진다. 이것이 A↔B 의 가설이다.")
    return fails


def section_stage_ladder(fn, name: str) -> list[str]:
    """⑥ 전진 유인 — 현재 계단을 완성하는 것 vs 다음 계단으로 올라가는 것."""
    print("\n" + "=" * 78)
    print(f"⑥ 전진 유인 — stage 1 정체가 가능한가  [{name}]")
    print("=" * 78)
    fails = []
    # r_lift 를 최대로 올렸을 때 stage 1 에 머물 수 있는 **최소** 거리.
    # ★0 에서 위로 훑는다 — 위에서 아래로 훑고 첫 idx≤1 에 멈추면 스캔 상한이 그대로
    #   답으로 나오는 버그가 된다(초판에서 300 mm 라는 무의미한 값이 나왔다).
    z_top = P.MINIMAL_LIFT_HEIGHT
    best_stay = None
    for i in range(3001):
        dist = i * 0.0001                          # 0 ~ 300 mm, 0.1 mm 눈금
        _, idx, _ = fn(dist, z_top, speed=0.0)
        if idx <= 1:
            best_stay = dist
            break
    spawn_dist = path_point(0.0)[0]
    if best_stay is None:
        print("  r_lift 최대에서는 어느 거리에서도 stage 1 에 못 머문다")
    else:
        print(f"  r_lift 최대(램프 포화)로 stage 1 에 머물려면 컵–목표 ≥ "
              f"{best_stay*1000:.1f} mm 여야 한다")
        print(f"  스폰 거리 = {spawn_dist*1000:.1f} mm"
              f"  ⇒ 여유 {(spawn_dist - best_stay)*1000:+.1f} mm")
        if best_stay < spawn_dist:
            print(f"     즉 컵을 {(spawn_dist-best_stay)*1000:.0f} mm 만 목표 쪽으로 옮기면"
                  f" 승급 — stage 1 정체가 구조적으로 어렵다 ✓")
    return fails


# ---------------------------------------------------------------------------
# ★★⑦ 실측 운전점 스윕 — 라운드 3 실패를 놓친 결함의 처방
# ---------------------------------------------------------------------------
# 라운드 3 설계 때 이 스크립트는 `upright=1.0 · speed=0` 인 **이상적 슬라이스**만
# 훑었다. 그 슬라이스에서 이송 유인은 +0.250 으로 읽혔는데, 실제 정책의 운전점
# (`P_still` 0.43 · `P_upright` 0.71)에서는 **+0.077** 이었다 — 구 설계(+0.207)의
# 2.7 배 약화. 4000 GPU-epoch 를 쓰고 나서야 로그로 알았다.
# ⇒ 이제 **학습 로그에서 실측한 운전점**을 반드시 함께 훑는다.
#   운전점은 tfevents 의 `diag_p_still`·`diag_p_upright` 100 epoch 구간 평균이다.
# ★라운드 5 — 운전점을 **baseline 결정론 실측**(목표 반경 안, 1024 env)으로 갱신.
#   속도 p10 0.0128 · 중앙 0.0399 · p90 0.1485  /  직립 p10 0.9747 · 중앙 0.9829 · p90 0.9933
OPERATING_POINTS = [
    ("이상 (설계가 보던 슬라이스)", 1.000, 1.000),
    ("baseline p10 (나쁜 쪽)",     smoothstep(0.1485, *P.P_STILL_BAND),
                                   smoothstep(0.9747, *P.P_UPRIGHT_BAND)),
    ("baseline 중앙",              smoothstep(0.0399, *P.P_STILL_BAND),
                                   smoothstep(0.9829, *P.P_UPRIGHT_BAND)),
    ("baseline p90 (좋은 쪽)",     smoothstep(0.0128, *P.P_STILL_BAND),
                                   smoothstep(0.9933, *P.P_UPRIGHT_BAND)),
]


def section_settle_incentive(fn, name: str) -> list[str]:
    """★⑤ 정지 유인 — 목표 반경 안에서 "지금 행동 → 합격"의 보상 차이.

    라운드 5 의 유일한 신규 항이 `v_3 = P_still·P_upright` 다. 이 값이 실측 운전점에서
    포화하면(구 밴드가 그랬다: v_3 = 0.982) 개선 여지가 없다.
    """
    print("\n" + "=" * 78)
    print(f"⑦ ⑤ 정지 유인 — 목표 반경 안 (baseline 결정론 실측 운전점)  [{name}]")
    print("=" * 78)
    _, z_goal = _dist_to_path(0.030)
    print(f"   {'상황':>26} {'속도':>7} {'직립':>8} {'v_3':>7} {'r':>7}")
    rows = [("진입(빠름)", 0.1485, 0.9747), ("baseline 중앙", 0.0399, 0.9829),
            ("합격선 정확히", P.STAGE3_SPEED_MAX, P.STAGE3_UPRIGHT_MIN),
            ("이상", 0.010, 0.999)]
    vals = {}
    for lab, sp, up in rows:
        r, idx, v = fn(0.030, z_goal, sp, up)
        vals[lab] = r
        print(f"   {lab:>26} {sp:>7.4f} {up:>8.4f} {v:>7.4f} {r:>7.4f}")
    grad = vals["합격선 정확히"] - vals["baseline 중앙"]
    print(f"\n   운전점 → 합격 gradient  {grad:+.4f}")
    fails = []
    if grad < 0.05:
        print("   ★FAIL — 운전점에서 이미 포화했다. 밴드를 실측 분포에 재보정해야 한다.")
        fails.append(f"{name}: 정지 유인 {grad:+.4f} < 0.05")
    else:
        print("   ⇒ 정책이 개선할 여지가 있다 ✓  (구 밴드는 +0.0044 였다)")
    return fails


def _unused_operating_points(fn, name: str) -> list[str]:
    """구 절 — stage 2 가 거리만이던 시절의 검사. 보존만 한다."""
    print("\n" + "=" * 78)
    print(f"⑦ 실측 운전점 스윕 — 이송 유인이 운전점에 얼마나 흔들리는가  [{name}]")
    print("=" * 78)
    print("   유인 = r(목표 반경 50 mm 도달) − r(stage 1 천장). 클수록 이송이 이득이다.")
    print(f"   {'운전점':>24} {'st1 천장':>9} {'120mm':>8} {'50mm':>8} {'유인':>8}")
    fails, vals = [], []
    for lab, ps, pu in OPERATING_POINTS:
        # P_still / P_upright 를 그 운전점에 고정하려면 역산한 speed·upright 를 넣는다.
        far_s, near_s = P.P_STILL_BAND
        far_u, near_u = P.P_UPRIGHT_BAND
        spd = _invert_smoothstep(ps, far_s, near_s)
        upr = _invert_smoothstep(pu, far_u, near_u)
        # stage 1 천장 = 경로 위 1→2 전이 직전의 r (설계 규약에 의존하지 않게)
        top = _stage1_top_on_path(fn)
        _, z50 = _dist_to_path(0.050)
        _, z120 = _dist_to_path(0.120)
        r50 = fn(0.050, z50, spd, upr)[0]
        r120 = fn(0.120, z120, spd, upr)[0]
        gain = r50 - top
        vals.append(gain)
        print(f"   {lab:>24} {top:>9.4f} {r120:>8.4f} {r50:>8.4f} {gain:>8.4f}")
    spread = max(vals) - min(vals)
    print(f"\n   유인 편차(최대−최소) = {spread:.4f}")
    if spread > 0.05:
        print("   ★FAIL — 운전점에 따라 이송 유인이 크게 흔들린다. 정책이 제어하기 어려운")
        print("      인자(속도·직립)가 거리 신호를 곱으로 깎고 있다는 뜻이다.")
        fails.append(f"{name}: 운전점 편차 {spread:.4f}")
    else:
        print("   ⇒ 운전점 무관 — 거리 신호가 다른 인자에 안 깎인다 ✓")
    # 구 설계(run 0) 기준선 — **같은 경로·같은 규약**으로 잰다.
    #   ⚠ 천장을 `r_lift` 가정값으로, 값을 경로 실제값으로 재면 섞인다(초판 버그).
    #     둘 다 경로에서 직접 뽑는다: 천장 = 1→2 전이 직전의 r.
    old_top = _stage1_top_on_path(reward_old)
    old_gain = reward_old(*_dist_to_path(0.050), 0.0)[0] - old_top
    print(f"   참고: 구 설계(run 0) — st1 천장 {old_top:.4f} · 유인 {old_gain:.4f}"
          f"  (속도·직립 무관)")
    # ⚠ 기준을 "구 설계를 무조건 넘어라"로 두면 과설계를 부른다. 실제로 라운드 3 을
    #   죽인 것은 2.7 배(유인 0.207 → 0.077) 규모였다. 0.9 배 안이면 지배적 결함이
    #   아니므로 통과시키고, 차이의 출처를 명시해 다음 판단 재료로 남긴다.
    ratio = min(vals) / old_gain if old_gain else float("inf")
    if ratio < 0.9:
        print(f"   ★FAIL — 최악 운전점 유인이 구 설계의 {ratio:.0%} 다.")
        fails.append(f"{name}: 최악 운전점 {min(vals):.4f} = 구 설계의 {ratio:.0%}")
    else:
        print(f"   ⇒ 최악 운전점 유인 = 구 설계의 {ratio:.0%} — 지배적 결함 아님 ✓")
        if ratio < 1.0:
            new_top = _stage1_top_on_path(fn)
            print(f"      (차이의 출처: stage 1 천장 {new_top:.4f} vs 구 {old_top:.4f}"
                  f" — `v_1` 이 문턱에서 1 이 되도록 정규화한 대가다.")
            print(f"       천장을 낮추면 1→2 경계에 그만큼 위쪽 점프가 생긴다 — 상충하는")
            print(f"       두 성질이라, 지배적 결함을 고친 뒤 따로 판단한다.)")
    return fails


def _stage1_top_on_path(fn) -> float:
    """경로 위에서 stage 1 → 2 전이 **직전**의 r. 설계마다 규약이 달라 직접 잰다."""
    prev = None
    N = 20000
    for i in range(N + 1):
        dist, z = path_point(i / N)
        r, idx, _ = fn(dist, z, speed=0.0)
        if prev is not None and prev[1] == 1 and idx == 2:
            return prev[0]
        prev = (r, idx)
    return float("nan")


def _invert_smoothstep(y: float, far: float, near: float) -> float:
    """`smoothstep(x; far, near) = y` 인 x. 이분법(단조라 안전)."""
    lo, hi = min(far, near), max(far, near)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        v = smoothstep(mid, far, near)
        if (v < y) == (smoothstep(lo, far, near) < y):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _dist_to_path(target_d: float) -> tuple[float, float]:
    """경로 위에서 컵–목표 거리가 `target_d` 인 지점의 `(dist, cup_z)`."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if path_point(mid)[0] > target_d else (lo, mid)
    return path_point(0.5 * (lo + hi))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--still-net", action="store_true",
                    help="arm B 조건 표기용(식은 같고 속도 입력만 다르다)")
    ap.add_argument("--old-only", action="store_true", help="run 0 지형만 본다")
    args = ap.parse_args()

    print("=" * 78)
    print("v2 계단 보상 지형 스캔 — Isaac 불필요")
    print(f"  속도 입력: {'순변위 (arm B)' if args.still_net else '순간속도 (arm A)'}")
    print("  ⚠ 지형 자체는 속도 입력의 *값*에만 의존한다. 여기서는 속도를 직접 넣으므로"
          " 두 arm 의 지형은 같고, 실제 차이는 같은 상황에서 speed 가 얼마로 측정되는가다.")
    print("=" * 78)

    section_geometry()

    fails: list[str] = []
    variants = [(reward_old, "run 0 (붕괴한 지형)")]
    if not args.old_only:
        variants.append((reward_new, "라운드 5 (baseline 복구 + v_3)"))

    for fn, name in variants:
        is_new = fn is reward_new
        c = section_continuity(fn, name)
        p = section_plateau(fn, name)
        section_profile(fn, name)
        h = section_hacking(fn, name)
        section_stage_ladder(fn, name)
        o = section_settle_incentive(fn, name) if is_new else []
        if is_new:                     # 구 지형의 결함은 이미 알려진 것 — 실패로 세지 않는다
            fails += c + p + h + o

    print("\n" + "=" * 78)
    if fails:
        print(f"★FAILED — 라운드 5 지형에 결함 {len(fails)} 건")
        for f in fails:
            print(f"   · {f}")
        return 1
    print("PASSED — 라운드 5 지형: 하향 점프 없음 · stage 2 고원 없음 · 해킹면 없음")
    print("          · ⑤ 정지 유인 살아 있음")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
