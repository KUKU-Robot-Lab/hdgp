#!/usr/bin/env python3
"""[P0-2b] 좌팔 그리퍼 **기준 파지자세 + 홈 자세** 도출 (numpy/scipy, Isaac 불필요).

왜 홈을 새로 잡아야 하나
------------------------
좌팔 홈을 처음에는 right/grasp_sensor 의 유휴 왼팔 rest(= 우팔 DG-5F 홈의 부호 미러)로 썼다.
그 값은 **20관절 손 기준**으로 잡힌 자세라 2지 그리퍼의 파지 자세군과 손목이 ~100° 어긋난다.
Isaac 실측: Fabrics 가 파지 자세를 못 내고 jaw 가 28.5° 기울어 수평 파지 불성립, j5 는
한계에 붙었다. 유효 IK 해들의 홈 변위가 전부 1.73~1.80 rad(분기 폭 경계)였다.

Fabrics 는 IK 솔버가 아니라 홈에서 출발하는 **기울기 흐름**이다. 그러니 홈은
"파지 자세군 한가운데"에 있어야 한다 — 그래야 흐름이 손목을 재배치하지 않고 도달한다.

방법
----
1. (θ, φ) 후보마다 스폰 박스 9점에서 **정확 파지 자세** IK. 전 점 성공만 통과.
2. 통과한 (θ, φ) 마다 홈 = 9개 해의 **관절공간 중심**에서 물러난 자세(여러 후보 탐색).
3. 새 홈 기준으로 9점 재검증: 도달 + 관절여유 + 홈 변위.

두 가지 수치 함정 (둘 다 실측으로 밟음)
---------------------------------------
① **180° 뒤집힌 해**: 비-ok 해를 위치오차만으로 순위매기면 자세가 180° 뒤집힌 해가
   pos 0mm 로 1등이 된다. 아래는 위치와 자세를 항상 함께 본다.
② **얇은 자세 다양체**: 손목 j6 가 ±45° 뿐이라 도달 가능 자세 집합이 얇다. 무작위 재시작만
   쓰면 시드에 따라 해를 놓쳐 같은 점의 판정이 뒤집힌다. 격자점끼리는 가까우므로 이웃 해를
   시드로 쓰는 **연속법**(solve_grid_by_continuation)이 훨씬 안정적이다.

사용: python3 scripts/probes/probe_left_gripper_home.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HDGP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HDGP / "scripts/probes"))
sys.path.insert(0, str(HDGP / "scripts/assets_tools"))
from generate_left_fabric_urdf import parse_urdf  # noqa: E402
from probe_left_gripper_reach import (  # noqa: E402
    ARM_JOINTS,
    CUP_CENTER_X,
    CUP_CENTER_Y,
    FABRIC_URDF,
    GRASP_HEIGHT,
    SPAWN_HALF_X,
    SPAWN_HALF_Y,
    TABLE_Z,
    _compile_chain,
    _fk,
)

GRASP_DEPTH = 0.02          # preset 과 동일 — 컵 축보다 접근축 방향으로 밀어넣는 양
ROT_WEIGHT = 0.10           # 잔차에서 회전 1 rad 을 10 cm 로 환산
POS_TOL = 0.005             # m
# ★자세 허용오차 5°: 파지에 필요한 건 jaw 가 수평인 것이고, 5° 기울기면 두 접촉점이
#   여전히 컵 단면 지름 근처에 놓인다(reach 프로브의 JAW_TILT_TOL 과 같은 기준).
ROT_TOL = 5.0               # °
MARGIN_GATE = 0.10          # rad, 파지 자세의 관절 한계 여유
# 홈은 매 에피소드 시작 자세이자 유휴 자세라 파지 자세보다 넉넉해야 한다.
# 여유가 0.1 근처면 리셋 직후부터 관절한계 반발이 걸려 Fabrics 출발이 흔들린다.
HOME_MARGIN_GATE = 0.20     # rad
HOME_BRANCH_DEV = 1.8       # rad, 홈에서 연속 이동 가능하다고 보는 분기 폭

# 홈 위치 후보 = 파지 중심에서 (접근축 반대, +z, +y(바깥)) 조합으로 물러난 지점.
# ★단일 조합을 고정하면 안 된다 — 접근축이 대부분 +x 라 그 반대로만 물리면 TCP 가 몸통
#   쪽(x≈0.13)으로 파고들어 IK 가 깨진다(실측 pos 11mm / rot 16.6°).
HOME_OFFSET_CANDIDATES = [
    (0.06, 0.12, 0.04), (0.06, 0.16, 0.06), (0.08, 0.12, 0.06),
    (0.10, 0.10, 0.04), (0.10, 0.15, 0.08), (0.12, 0.08, 0.02),
    (0.04, 0.18, 0.08), (0.08, 0.18, 0.10), (0.12, 0.14, 0.06),
]


def rot_of(theta_deg: float, phi_deg: float) -> np.ndarray:
    """jaw 수평 파지자세 R = [핑거폭, jaw, 접근] (preset.grasp_axes 와 동일 정의)."""
    t, f = math.radians(theta_deg), math.radians(phi_deg)
    jaw = np.array([-math.sin(t), math.cos(t), 0.0])
    approach = np.array([math.cos(t) * math.cos(f), math.sin(t) * math.cos(f), -math.sin(f)])
    return np.column_stack([np.cross(jaw, approach), jaw, approach])


def _err(steps, q, p_goal, R_goal, limits):
    T = _fk(steps, q)
    pos = float(np.linalg.norm(p_goal - T[:3, 3]))
    dR = R_goal @ T[:3, :3].T
    rot = math.degrees(math.acos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
    margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
    return pos, rot, margin


def _ok(pos, rot, margin) -> bool:
    return (pos < POS_TOL) and (rot < ROT_TOL) and (margin > MARGIN_GATE)


def solve_pose(steps, limits, p_goal, R_goal, bounds, seed, restarts=30, prefer=None,
               seeds=()):
    """정확 6-DOF 자세 IK. prefer 가 있으면 그 자세에 가까운 해, 없으면 관절여유 최대."""
    lo, hi = bounds
    rng = np.random.default_rng(seed)

    def resid(q):
        T = _fk(steps, q)
        dR = R_goal @ T[:3, :3].T
        er = 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
        return np.concatenate([p_goal - T[:3, 3], ROT_WEIGHT * er])

    starts = [np.clip(np.asarray(s, dtype=float), lo, hi) for s in seeds]
    starts.append((lo + hi) / 2.0)
    starts += [rng.uniform(lo, hi) for _ in range(restarts)]

    best = None
    for q0 in starts:
        sol = least_squares(resid, q0, bounds=(lo, hi), xtol=1e-11, ftol=1e-11, max_nfev=400)
        pos, rot, margin = _err(steps, sol.x, p_goal, R_goal, limits)
        dq = 0.0 if prefer is None else float(np.abs(sol.x - prefer).max())
        # ★위치와 자세를 항상 함께 본다. 위치만 보면 180° 뒤집힌 해가 1등이 된다.
        key = ((0, dq if prefer is not None else -margin) if _ok(pos, rot, margin)
               else (1, pos + ROT_WEIGHT * math.radians(rot)))
        if best is None or key < best[0]:
            best = (key, pos, rot, margin, dq, sol.x)
    return best[1:]


def solve_grid_by_continuation(steps, limits, grid, grasp_z, R_goal, bounds,
                               prefer=None, extra_seeds=(), restarts=30):
    """중심점부터 풀고 이웃 해를 시드로 바깥으로 전파.

    returns (성공여부, 격자 순서대로의 해, 최악 관절여유, 최대 prefer 변위)
    """
    cx = sum(p[0] for p in grid) / len(grid)
    cy = sum(p[1] for p in grid) / len(grid)
    order = sorted(range(len(grid)),
                   key=lambda i: (grid[i][0] - cx) ** 2 + (grid[i][1] - cy) ** 2)
    solved: dict[int, np.ndarray] = {}
    worst_mg, worst_dq = 9.0, 0.0
    for i in order:
        x, y = grid[i]
        p = np.array([x, y, grasp_z]) + GRASP_DEPTH * R_goal[:, 2]
        near = sorted(solved, key=lambda j: (grid[j][0] - x) ** 2 + (grid[j][1] - y) ** 2)[:3]
        pos, rot, mg, dq, q = solve_pose(
            steps, limits, p, R_goal, bounds, seed=i + 1, prefer=prefer,
            restarts=restarts, seeds=[solved[j] for j in near] + list(extra_seeds),
        )
        if not _ok(pos, rot, mg):
            return False, [], -1.0, -1.0
        solved[i] = q
        worst_mg = min(worst_mg, mg)
        worst_dq = max(worst_dq, dq)
    return True, [solved[i] for i in range(len(grid))], worst_mg, worst_dq


def main() -> int:
    joints = parse_urdf(FABRIC_URDF)
    steps = _compile_chain(joints)
    limits = np.array([joints[n]["limits"] for n in ARM_JOINTS])
    full = (limits[:, 0], limits[:, 1])

    grasp_z = TABLE_Z + GRASP_HEIGHT
    grid = [(x, y)
            for x in (CUP_CENTER_X - SPAWN_HALF_X, CUP_CENTER_X, CUP_CENTER_X + SPAWN_HALF_X)
            for y in (CUP_CENTER_Y - SPAWN_HALF_Y, CUP_CENTER_Y, CUP_CENTER_Y + SPAWN_HALF_Y)]

    print("=== 설정 ===")
    print(f"  스폰 박스 x={CUP_CENTER_X:.2f}±{SPAWN_HALF_X:.2f} "
          f"y={CUP_CENTER_Y:.2f}±{SPAWN_HALF_Y:.2f}  파지 z={grasp_z:.4f}")
    print(f"  게이트: pos<{POS_TOL*1000:.0f}mm, rot<{ROT_TOL:.0f}°, "
          f"관절여유>{MARGIN_GATE:.2f}rad")

    print("\n=== 1. (θ, φ) 후보 — 스폰 박스 9점 정확 파지자세 도달 ===")
    print(f"  {'θ':>5} {'φ':>4} | {'최소 관절여유':>13}")
    survivors = []
    # 탐색 범위는 실측 지도(θ 30~50 / φ 15~30 대역이 유효)에 맞춰 좁혔다.
    # 연속법이 시드를 대부분 해결하므로 survey 단계는 무작위 재시작을 줄인다.
    for th in range(20, 61, 10):
        for ph in (15, 30):
            R_goal = rot_of(th, ph)
            ok, sols, worst_mg, _ = solve_grid_by_continuation(
                steps, limits, grid, grasp_z, R_goal, full, restarts=8)
            if ok:
                survivors.append((th, ph, np.array(sols), worst_mg))
                print(f"  {th:5d} {ph:4d} | {worst_mg:13.3f}")
    if not survivors:
        print("\n  [FAIL] 전 격자점에서 정확 자세가 가능한 (θ, φ) 가 없다")
        print("         → 스폰 박스 축소 또는 컵 위치 재조정 필요")
        return 1

    print("\n=== 2. 홈 후보 (파지 해 중심에서 후퇴) ===")
    print(f"  {'θ':>5} {'φ':>4} {'후퇴/상승/바깥':>17} | {'홈여유':>7} {'최대홈변위':>10} "
          f"{'홈 TCP':>24} {'박스밖':>6}")
    best = None
    for th, ph, sols, _mg in survivors:
        R_goal = rot_of(th, ph)
        centroid = sols.mean(axis=0)
        c_tcp = _fk(steps, centroid)[:3, 3]
        for back, lift, out in HOME_OFFSET_CANDIDATES:
            home_tcp = c_tcp - back * R_goal[:, 2] + np.array([0.0, out, lift])
            pos, rot, mg, _dq, q_home = solve_pose(
                steps, limits, home_tcp, R_goal, full, seed=99, prefer=centroid,
                seeds=[centroid, *sols],
            )
            label = f"{back:.2f}/{lift:.2f}/{out:.2f}"
            if not _ok(pos, rot, mg):
                # ★조용히 넘기지 않는다 — 어느 조건에서 걸렸는지 보여야 다음 후보를 고른다.
                print(f"  {th:5d} {ph:4d} {label:>17} | 홈 IK 실패 "
                      f"(pos {pos*1000:.1f}mm, rot {rot:.1f}°, 여유 {mg:.3f})")
                continue
            lo = np.maximum(limits[:, 0], q_home - HOME_BRANCH_DEV)
            hi = np.minimum(limits[:, 1], q_home + HOME_BRANCH_DEV)
            ok2, _s2, mg2, dq2 = solve_grid_by_continuation(
                steps, limits, grid, grasp_z, R_goal, (lo, hi),
                prefer=q_home, extra_seeds=[q_home, *sols])
            if not ok2:
                print(f"  {th:5d} {ph:4d} {label:>17} | 홈여유 {mg:.3f} 이지만 "
                      f"이 홈에서 9점 재검증 실패(분기 ±{HOME_BRANCH_DEV})")
                continue
            t_home = _fk(steps, q_home)[:3, 3]
            outside = (abs(t_home[0] - CUP_CENTER_X) > SPAWN_HALF_X + 0.04
                       or abs(t_home[1] - CUP_CENTER_Y) > SPAWN_HALF_Y + 0.04
                       or t_home[2] > grasp_z + 0.08)
            print(f"  {th:5d} {ph:4d} {label:>17} | {mg:7.3f} {dq2:10.3f} "
                  f"{str([round(float(v), 3) for v in t_home]):>24} "
                  f"{'예' if outside else '아니오':>6}")
            if outside and mg >= HOME_MARGIN_GATE and (best is None or dq2 < best[0]):
                best = (dq2, th, ph, q_home, mg2, t_home)

    print("\n=== 판정 ===")
    if best is None:
        print("  [FAIL] 박스 밖 + 9점 도달 + 여유를 모두 만족하는 홈이 없다")
        return 1
    dq, th, ph, q_home, mg, t_home = best
    print(f"  [PASS] θ={th}°, φ={ph}° / 홈→파지 최대 관절변위 {dq:.3f} rad "
          f"(구 홈은 1.73~1.80 = 분기 경계였다) / 파지 최소여유 {mg:.3f} rad")
    print(f"  홈 TCP = {[round(float(v), 4) for v in t_home]}")
    print("\n  → preset 상수:")
    print(f"       GRASP_JAW_AZIMUTH_DEG   = {th}.0")
    print(f"       GRASP_APPROACH_TILT_DEG = {ph}.0")
    print("       LEFT_ARM_HOME_JOINT_POS = {")
    for i, v in enumerate(q_home):
        print(f'           "l_aj_{i + 1}": {v:+.4f},')
    print("       }")
    print("  → fabric 클래스 _GRIPPER_LEFT_DEFAULT_CONFIG 도 같은 값으로 갱신할 것")
    return 0


if __name__ == "__main__":
    sys.exit(main())
