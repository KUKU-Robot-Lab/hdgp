#!/usr/bin/env python3
"""[P0-2] 좌팔 2지 그리퍼 측면 파지 자세 도달성 + 기준자세 도출 (numpy/scipy, Isaac 불필요).

무엇을 결정하나
---------------
"컵 스폰 박스 전 구간에서 **jaw 가 수평인** 파지 자세를 팔이 실제로 낼 수 있는가"를 학습 전에
못 박고, 낼 수 있다면 **전 구간 공통 기준자세(θ*, φ*)** 를 뽑아 preset 상수로 넘긴다.

왜 이 방법인가 (실패한 접근 기록)
---------------------------------
처음에는 파지 자세를 "접근축도 수평(φ=0)"으로 **고정**해 놓고 IK 를 풀었다. 전 구간 실패했고
DLS 솔버를 고쳐도 잔차 8~24° 가 남았다. 원인은 솔버가 아니라 **자세를 과잉구속한 것**이었다:
손목 j6 가 ±45° 뿐이라 이 팔의 달성 가능 자세 집합은 (방위 × 기울기) 2-D 패치가 아니라
**곡선**이다. 격자로 (θ, φ) 를 찍어 맞히려 하면 거의 다 빗나간다.

그래서 순서를 뒤집었다:
  1. **위치만** IK 로 푼다(항상 잘 풀린다) → 같은 위치를 내는 해가 여럿 나온다(7 DOF 잉여).
  2. 그 해들의 **실제 자세**를 보고 jaw 기울기 < JAW_TILT_TOL 인 것만 남긴다.
  3. 남은 자세들의 (θ=jaw 방위, φ=접근 기울기)를 격자점끼리 **교집합** 내어 공통 기준자세를 고른다.
즉 "원하는 자세를 팔에 요구"하는 게 아니라 "팔이 낼 수 있는 자세 중 파지에 유효한 것"을 고른다.

파지에 정말 필요한 조건은 접근축 방향이 아니라 **jaw 축이 수평**인 것뿐이다 —
그래야 두 접촉점이 컵 단면의 지름 양끝(대향)에 놓인다. 접근 기울기는 자유롭게 둔다.

홈 분기 제한
------------
Fabrics 는 현재 q 에서 연속 이동하므로 팔이 통째로 뒤집힌 분기(j1≈-3.4rad, 어깨 195° 회전)는
실제로 쓸 수 없다. 시드와 bound 를 홈에서 관절당 HOME_BRANCH_DEV 이내로 묶는다.
(이 제한을 안 걸면 "잔차 0mm" 해가 나오지만 전부 뒤집힌 분기라 무의미하다 — 실측으로 확인)

사용: python3 scripts/probes/probe_left_gripper_reach.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HDGP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HDGP / "scripts/assets_tools"))
from generate_left_fabric_urdf import axis_angle, parse_urdf, rpy_to_mat  # noqa: E402

FABRIC_URDF = (
    HDGP / "source/FABRICS/src/fabrics_sim/models/robots/urdf"
    / "openarm_tesollo_sensor_left_gripper/openarm_tesollo_sensor_left_gripper.urdf"
)
ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]

# ── 씬 기하 ────────────────────────────────────────────────────────
TABLE_Z = 0.2082               # right/grasp_sensor 와 동일 테이블 표면
CUP_BOTTOM_TO_ORIGIN = 0.092090  # shaker_closed_rl.usd 메시 bottom → 원점 (probe_gripper_opening 실측)
CUP_CENTER_X = 0.25            # ★우측(0.30)의 단순 미러가 아니다 — 아래 "왜 x=0.25" 참조
CUP_CENTER_Y = 0.20
SPAWN_HALF_X = 0.03
SPAWN_HALF_Y = 0.10
# 테이블 기준 파지 중심 높이. ★P0-1 통과대역(shaker: 10~85mm) 안에서 **스윕해 정한다** —
# 그리퍼 여유와 팔 도달성이 반대 방향이라 높이가 곧 자유 파라미터다.
# 전 격자점 최소 관절여유 실측: h=55 → 0.101 / h=65 → **0.238** / h=75 → 0.005 / h=85 → 공통해 없음.
GRASP_HEIGHT = 0.065

# 좌팔 홈 = grasp_sensor 프리셋 LEFT_ARM_REST_JOINT_POS 실측값
Q_HOME = np.array([-0.0431, -0.6706, -0.0961, 0.7342, -0.3750, -0.5678, -0.6709])
HOME_BRANCH_DEV = 1.8          # rad, 홈 분기 폭

# ── 게이트 ─────────────────────────────────────────────────────────
JAW_TILT_TOL = 5.0     # ° jaw 축이 수평에서 벗어나도 되는 한도
POSE_MATCH_TOL = 8.0   # ° 격자점 간 공통 기준자세로 인정할 각도 근접도
MARGIN_GATE = 0.10     # rad 관절 한계 여유
N_SAMPLES = 400        # 격자점당 위치 IK 재시작 수


# ── FK (체인 1회 컴파일) ───────────────────────────────────────────
def _compile_chain(joints: dict, tip: str = "palm_link"):
    by_child = {j["child"]: (n, j) for n, j in joints.items()}
    chain, cur = [], tip
    while cur in by_child:
        chain.append(by_child[cur])
        cur = by_child[cur][1]["parent"]
    chain.reverse()
    steps = []
    for _name, j in chain:
        T = np.eye(4)
        T[:3, :3] = rpy_to_mat(*j["rpy"])
        T[:3, 3] = j["xyz"]
        steps.append((T, j["axis"] if j["type"] == "revolute" else None))
    return steps


def _fk(steps, q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    qi = 0
    for Tf, ax in steps:
        T = T @ Tf
        if ax is not None:
            Rq = np.eye(4)
            Rq[:3, :3] = axis_angle(ax, q[qi])
            qi += 1
            T = T @ Rq
    return T


def sample_jaw_horizontal_poses(steps, limits, p_goal, seed):
    """위치만 만족하는 해를 다수 구한 뒤 jaw 수평인 것만 남겨 (θ, φ, margin) 반환."""
    lo = np.maximum(limits[:, 0], Q_HOME - HOME_BRANCH_DEV)
    hi = np.minimum(limits[:, 1], Q_HOME + HOME_BRANCH_DEV)
    rng = np.random.default_rng(seed)
    out = []
    for k in range(N_SAMPLES):
        q0 = Q_HOME.copy() if k == 0 else rng.uniform(lo, hi)
        sol = least_squares(lambda q: p_goal - _fk(steps, q)[:3, 3], q0,
                            bounds=(lo, hi), xtol=1e-10, ftol=1e-10, max_nfev=120)
        if np.linalg.norm(p_goal - _fk(steps, sol.x)[:3, 3]) > 1e-4:
            continue
        R = _fk(steps, sol.x)[:3, :3]
        jaw, approach = R[:, 1], R[:, 2]
        jaw_tilt = math.degrees(math.asin(min(1.0, abs(jaw[2]))))
        if jaw_tilt > JAW_TILT_TOL:
            continue
        theta = math.degrees(math.atan2(-jaw[0], jaw[1]))
        theta = theta - 180.0 if theta > 90 else (theta + 180.0 if theta < -90 else theta)
        phi = math.degrees(math.asin(max(-1.0, min(1.0, -approach[2]))))
        margin = float(np.min(np.minimum(sol.x - limits[:, 0], limits[:, 1] - sol.x)))
        out.append((theta, phi, margin))
    return out


def main() -> int:
    if not FABRIC_URDF.is_file():
        print(f"[FAIL] fabric URDF 없음: {FABRIC_URDF}\n"
              f"       먼저 scripts/assets_tools/generate_sensor_left_gripper_fabric_urdf.py 실행")
        return 1
    joints = parse_urdf(FABRIC_URDF)
    steps = _compile_chain(joints)
    limits = np.array([joints[n]["limits"] for n in ARM_JOINTS])

    grasp_z = TABLE_Z + GRASP_HEIGHT
    print("=== 설정 ===")
    print(f"  테이블 z={TABLE_Z:.4f}  파지 z={grasp_z:.4f} (테이블 위 {GRASP_HEIGHT*1000:.0f} mm)")
    print(f"  스폰 박스 x={CUP_CENTER_X:.2f}±{SPAWN_HALF_X:.2f}  y={CUP_CENTER_Y:+.2f}±{SPAWN_HALF_Y:.2f}")
    print(f"  게이트: jaw 기울기<{JAW_TILT_TOL:.0f}°, 관절여유>{MARGIN_GATE:.2f}rad, "
          f"전 격자점 공통자세 ±{POSE_MATCH_TOL:.0f}°")

    grid = [(x, y)
            for x in (CUP_CENTER_X - SPAWN_HALF_X, CUP_CENTER_X, CUP_CENTER_X + SPAWN_HALF_X)
            for y in (CUP_CENTER_Y - SPAWN_HALF_Y, CUP_CENTER_Y, CUP_CENTER_Y + SPAWN_HALF_Y)]

    print("\n=== 격자점별 jaw-수평 자세 집합 ===")
    print(f"  {'x':>5} {'y':>6} | {'해수':>4}  {'θ 범위[°]':>16}  {'φ 범위[°]':>16}  {'최대여유':>8}")
    sets: dict[tuple[float, float], list] = {}
    for i, (x, y) in enumerate(grid):
        s = sample_jaw_horizontal_poses(steps, limits, np.array([x, y, grasp_z]), seed=i)
        sets[(x, y)] = s
        if not s:
            print(f"  {x:5.2f} {y:+6.2f} |    0  {'—':>16}  {'—':>16}  {'—':>8}")
            continue
        th = np.array([a[0] for a in s]); ph = np.array([a[1] for a in s])
        mg = np.array([a[2] for a in s])
        print(f"  {x:5.2f} {y:+6.2f} | {len(s):4d}  {th.min():+7.1f}~{th.max():+7.1f}  "
              f"{ph.min():+7.1f}~{ph.max():+7.1f}  {mg.max():8.3f}")

    print("\n=== 전 격자점 공통 기준자세 탐색 ===")
    best = None
    for th0 in range(-90, 91, 5):
        for ph0 in range(0, 86, 5):
            worst = 9.0
            for s in sets.values():
                cand = [m for t, p_, m in s
                        if abs(t - th0) <= POSE_MATCH_TOL and abs(p_ - ph0) <= POSE_MATCH_TOL]
                if not cand:
                    worst = -1.0
                    break
                worst = min(worst, max(cand))
            if worst >= 0 and (best is None or worst > best[0]):
                best = (worst, th0, ph0)

    print("\n=== 판정 ===")
    if best is None or best[0] < MARGIN_GATE:
        got = "없음" if best is None else f"최소여유 {best[0]:.3f} < {MARGIN_GATE}"
        print(f"  [FAIL] 전 격자점 공통 jaw-수평 기준자세 {got}")
        print("         → 파지 높이/스폰 박스/컵 위치 재조정 필요")
        return 1
    _, th_star, ph_star = best
    print(f"  [PASS] 공통 기준자세 존재 — 전 격자점 최소 관절여유 {best[0]:.3f} rad")
    print(f"  → preset 상수:")
    print(f"       GRASP_JAW_AZIMUTH_DEG  = {th_star}   # jaw 축 = (-sinθ, cosθ, 0), 수평")
    print(f"       GRASP_APPROACH_TILT_DEG = {ph_star}   # 접근축을 수평에서 아래로 φ")
    print(f"       GRASP_HEIGHT_ABOVE_TABLE = {GRASP_HEIGHT:.3f}")
    print(f"       CUP_SPAWN_CENTER = ({CUP_CENTER_X:.2f}, {CUP_CENTER_Y:+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
