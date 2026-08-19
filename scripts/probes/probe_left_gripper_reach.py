#!/usr/bin/env python3
"""[P0-2] 좌팔 2지 그리퍼 자세 다양체 **탐색용** 지도 (numpy/scipy, Isaac 불필요).

⚠ 기준자세·홈의 **정식 도출기는 probe_left_gripper_home.py** 다. 이 프로브는 보조다.
  여기서 쓰는 "±POSE_MATCH_TOL 근방에 해가 있으면 통과" 기준은 **실제로 명령하는 정확
  자세의 도달성을 보장하지 않는다** — 그 착각으로 한 번 실패했다(Isaac 에서 자세 28° 오차).
  이 프로브는 "어느 (θ, φ) 대역을 들여다볼지" 감을 잡는 용도로만 쓸 것.


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
CUP_CENTER_X = 0.26            # ★우측(0.30)의 단순 미러가 아니다 — 아래 주석 참조
# ★스폰 박스 = x 0.26±0.04, y 0.30±0.06 (연속법 IK 실측).
#   이 영역에서 θ=30°/φ=15° 정확 파지자세의 관절여유가 전 구간 0.20~0.27 로 넉넉하다.
#   더 안쪽(y<=0.20)은 팔이 몸쪽으로 접혀 자세 자체가 안 나오고, y=0.24 대역은 되긴 하나
#   여유가 0.11 로 절반이다. 바깥(y>=0.40)·앞(x>=0.34)은 다시 한계에 붙는다.
#   우측 컵이 y=-0.20 이므로 좌우 분리 0.50 — 양팔 겹침 관점에서도 여유롭다.
CUP_CENTER_Y = 0.30
SPAWN_HALF_X = 0.04
SPAWN_HALF_Y = 0.06
# 테이블 기준 파지 중심 높이. ★P0-1 통과대역(shaker: 10~85mm) 안에서 **스윕해 정한다** —
# 그리퍼 여유와 팔 도달성이 반대 방향이라 높이가 곧 자유 파라미터다.
# 전 격자점 최소 관절여유 실측: h=55 → 0.101 / h=65 → **0.238** / h=75 → 0.005 / h=85 → 공통해 없음.
GRASP_HEIGHT = 0.065

# 좌팔 홈 = probe_left_gripper_home.py 가 도출한 그리퍼 전용 홈.
# ★구 값(우팔 DG-5F 홈의 미러)은 파지 자세군 밖이라 Fabrics 가 못 따라갔다.
Q_HOME = np.array([0.0844, -1.3476, 1.2701, 1.7705, 1.2631, -0.4643, 1.2345])
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
    """위치만 만족하는 해를 다수 구한 뒤 jaw 수평인 것만 남겨 (θ, φ, margin, dq) 반환.

    dq = 홈에서의 최대 관절 변위. ★이게 왜 필요한가: Fabrics 는 IK 솔버가 아니라 홈에서
    출발하는 **기울기 흐름**이라, 기구학적으로 도달 가능해도 흐름이 닿지 않는 자세가 있다.
    실제로 기구학 여유만 보고 고른 자세(θ=-15/φ=35)는 Isaac 에서 j5 가 한계에 붙어
    자세 오차 28° 로 실패했다. 홈에서 가까운 해일수록 흐름이 실제로 도달한다.
    """
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
        dq = float(np.abs(sol.x - Q_HOME).max())
        out.append((theta, phi, margin, dq))
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
    print(f"  {'x':>5} {'y':>6} | {'해수':>4}  {'θ 범위[°]':>16}  {'φ 범위[°]':>16}  "
          f"{'최대여유':>8}  {'최소dq':>7}")
    sets: dict[tuple[float, float], list] = {}
    for i, (x, y) in enumerate(grid):
        s = sample_jaw_horizontal_poses(steps, limits, np.array([x, y, grasp_z]), seed=i)
        sets[(x, y)] = s
        if not s:
            print(f"  {x:5.2f} {y:+6.2f} |    0  {'—':>16}  {'—':>16}  {'—':>8}")
            continue
        th = np.array([a[0] for a in s]); ph = np.array([a[1] for a in s])
        mg = np.array([a[2] for a in s]); dq = np.array([a[3] for a in s])
        print(f"  {x:5.2f} {y:+6.2f} | {len(s):4d}  {th.min():+7.1f}~{th.max():+7.1f}  "
              f"{ph.min():+7.1f}~{ph.max():+7.1f}  {mg.max():8.3f}  {dq.min():7.2f}")

    # ★선택 기준 = "관절여유가 게이트를 넘는 것들 중 **홈에서 가장 가까운** 자세".
    #   구 기준(여유 최대화)은 기구학적으로만 좋은 자세를 골라 Fabrics 가 못 따라갔다.
    print("\n=== 전 격자점 공통 기준자세 탐색 (여유 게이트 통과 + 홈 변위 최소) ===")
    best = None
    for th0 in range(-90, 91, 5):
        for ph0 in range(0, 86, 5):
            worst_margin, worst_dq = 9.0, 0.0
            for s in sets.values():
                cand = [(m, d) for t, p_, m, d in s
                        if abs(t - th0) <= POSE_MATCH_TOL and abs(p_ - ph0) <= POSE_MATCH_TOL
                        and m >= MARGIN_GATE]
                if not cand:
                    worst_margin = -1.0
                    break
                m_best, d_best = min(cand, key=lambda md: md[1])   # 홈에서 가장 가까운 해
                worst_margin = min(worst_margin, m_best)
                worst_dq = max(worst_dq, d_best)
            if worst_margin >= 0 and (best is None or worst_dq < best[3]):
                best = (worst_margin, th0, ph0, worst_dq)

    print("\n=== 판정 ===")
    if best is None or best[0] < MARGIN_GATE:
        got = "없음" if best is None else f"최소여유 {best[0]:.3f} < {MARGIN_GATE}"
        print(f"  [FAIL] 전 격자점 공통 jaw-수평 기준자세 {got}")
        print("         → 파지 높이/스폰 박스/컵 위치 재조정 필요")
        return 1
    _, th_star, ph_star, dq_star = best
    print(f"  [PASS] 공통 기준자세 존재 — 최소 관절여유 {best[0]:.3f} rad, "
          f"홈에서 최대 관절 변위 {dq_star:.3f} rad")
    print("  ⚠ 이 프로브는 **기구학** 도달성만 본다. Fabrics 는 홈에서 출발하는 기울기 흐름이라")
    print("    여기서 PASS 여도 실제로는 다른 평형에 갇힐 수 있다 →")
    print("    scripts/probes/probe_gripper_grip_force.py 로 반드시 확인할 것.")
    print(f"  → preset 상수:")
    print(f"       GRASP_JAW_AZIMUTH_DEG  = {th_star}   # jaw 축 = (-sinθ, cosθ, 0), 수평")
    print(f"       GRASP_APPROACH_TILT_DEG = {ph_star}   # 접근축을 수평에서 아래로 φ")
    print(f"       GRASP_HEIGHT_ABOVE_TABLE = {GRASP_HEIGHT:.3f}")
    print(f"       CUP_SPAWN_CENTER = ({CUP_CENTER_X:.2f}, {CUP_CENTER_Y:+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
