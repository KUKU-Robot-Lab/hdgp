#!/usr/bin/env python3
"""5g_grasp_right pregrasp ARM_START_POSE IK 계산 도구.

목표: cup center가 rj_dg_1_1 (thumb base, z=0.087m from palm_link)와
     rl_dg_3_4 (middle fingertip, z=0.248m from palm_link) 사이 중점에 오도록
     최적 ARM_START_POSE (7 DOF) 계산.

원리:
  - palm_link +Z (손가락 방향) ≈ world +X  → cup.x = palm_link.x + GRASP_MIDPOINT
  - palm_link +X (손바닥 법선) ≈ world +Y  → 접근 방향
  - 컵 root z + 0.04 = palm_link z         → 파지 높이 일치

사용법:
  python3 find_pregrasp_pose.py
  python3 find_pregrasp_pose.py --cup 0.40 -0.15 0.215
  python3 find_pregrasp_pose.py --cup 0.40 -0.15 0.215 --pregrasp-y -0.09
  python3 find_pregrasp_pose.py --restarts 10
"""

import sys
import argparse
import math
import numpy as np
from scipy.optimize import minimize

# openarm_fk.py import (numpy-only FK 도구)
sys.path.insert(0, "/home/user/rl_ws/scripts/tools")
from openarm_fk import arm_fk_raw, arm_fk, T_PALM_LINK, _OFFSET   # noqa: E402

# ---------------------------------------------------------------------------
# 손가락 기하 상수 (palm_link frame 기준, 손 펼친 자세 open pose)
# URDF: openarm_tesollo.urdf
# ---------------------------------------------------------------------------

# rj_dg_1_1 (thumb base joint) origin in palm_link frame: xyz=[-0.0162, 0.019, 0.0866]
# palm_link +Z 방향 오프셋 (손가락 방향 = 컵 방향)
THUMB_BASE_Z_FROM_PALM   = 0.0866

# rj_dg_3_4 (middle finger DIP joint) palm_link +Z 방향 총 길이 (open pose 기준):
#   rj_dg_3_1: 0.1439  (palm_link → MCP base)
#   rj_dg_3_2: 0.0265  (MCP base → MCP tip)
#   rj_dg_3_3: 0.0388  (PIP)
#   rj_dg_3_4: 0.0388  (DIP)
MIDDLE_TIP_Z_FROM_PALM   = 0.1439 + 0.0265 + 0.0388 + 0.0388   # = 0.248 m

# 파지 중점: 컵이 위치할 palm_link +Z 오프셋
GRASP_MIDPOINT_Z         = (THUMB_BASE_Z_FROM_PALM + MIDDLE_TIP_Z_FROM_PALM) / 2.0

# cup_grasp_z_offset (grasp_right_env_cfg.py 기준)
CUP_GRASP_Z_OFFSET       = 0.04

# ---------------------------------------------------------------------------
# 목표 방향 (grasp_right_preset.py: ez=0°, ey=90°, ex=0° 기준)
# ---------------------------------------------------------------------------
TARGET_PALM_Z_DIR  = np.array([1.0, 0.0, 0.0])   # 손가락 → world +X
TARGET_PALM_X_DIR  = np.array([0.0, 1.0, 0.0])   # 손바닥 법선 → world +Y

# ---------------------------------------------------------------------------
# OpenArm 관절 한계 (rad)
# ---------------------------------------------------------------------------
JOINT_LIMITS = [
    (-3.14, 3.14),  # j1 (torso rotation)
    (-1.57, 1.57),  # j2
    (-3.14, 3.14),  # j3
    (-1.57, 2.50),  # j4
    (-3.14, 3.14),  # j5
    (-1.57, 1.57),  # j6
    (-3.14, 3.14),  # j7
]

# 초기 추정값 (현재 grasp_right_preset.py ARM_START_POSE)
Q0 = [0.5, 0.1, 0.4, 0.60, -0.2, 0.0, 0.0]


# ---------------------------------------------------------------------------
# palm_link 위치·방향 계산
# ---------------------------------------------------------------------------

def get_palm_link_pose(q, apply_offset=True):
    """palm_link origin 위치와 방향벡터 반환.

    Returns:
        pos   : (3,) palm_link origin, sim 좌표 (calibrated)
        x_dir : (3,) palm_link +X 단위벡터 (손바닥 법선)
        z_dir : (3,) palm_link +Z 단위벡터 (손가락 방향)
    """
    T_world, _, _ = arm_fk_raw(q)
    T_palm = T_world @ T_PALM_LINK
    pos_raw = T_palm[:3, 3]
    x_dir   = T_palm[:3, 0]
    z_dir   = T_palm[:3, 2]
    pos = pos_raw - _OFFSET if apply_offset else pos_raw
    return pos, x_dir, z_dir


# ---------------------------------------------------------------------------
# 목표 palm_link 위치 계산
# ---------------------------------------------------------------------------

def compute_target_palm_link(cup_pos, pregrasp_offset_y=-0.09):
    """컵 위치로부터 목표 palm_link 위치 계산.

    palm_link +Z = world +X (손가락 → 컵 방향) 이므로:
      palm_link.x = cup.x - GRASP_MIDPOINT_Z   (손가락 중점이 컵 center와 일치)

    palm_link +X = world +Y (손바닥 법선) 이므로:
      palm_link.y = cup.y + pregrasp_offset_y   (옆(-Y)에서 접근)

    Z:
      palm_link.z = cup_root_z + CUP_GRASP_Z_OFFSET   (파지 높이)
    """
    cup_x, cup_y, cup_z = cup_pos
    target_x = cup_x - GRASP_MIDPOINT_Z
    target_y = cup_y + pregrasp_offset_y
    target_z = cup_z + CUP_GRASP_Z_OFFSET
    return np.array([target_x, target_y, target_z])


# ---------------------------------------------------------------------------
# IK 목적함수
# ---------------------------------------------------------------------------

def ik_objective(q, target_pos, w_pos=200.0, w_ori=20.0, w_reg=0.05):
    """IK 비용함수.

    Args:
        q          : (7,) 관절각 (rad)
        target_pos : (3,) 목표 palm_link 위치
        w_pos      : 위치 오차 가중치
        w_ori      : 방향 오차 가중치  (1 - cos 기반)
        w_reg      : 초기 자세 편차 패널티
    """
    pos, x_dir, z_dir = get_palm_link_pose(q)

    pos_err   = np.sum((pos - target_pos) ** 2)
    ori_err_z = 1.0 - float(np.dot(z_dir, TARGET_PALM_Z_DIR))
    ori_err_x = 1.0 - float(np.dot(x_dir, TARGET_PALM_X_DIR))
    reg_err   = float(np.sum((q - np.array(Q0)) ** 2))

    return w_pos * pos_err + w_ori * (ori_err_z + ori_err_x) + w_reg * reg_err


# ---------------------------------------------------------------------------
# IK 풀기
# ---------------------------------------------------------------------------

def solve_ik(cup_pos, pregrasp_offset_y=-0.09, n_restarts=8, verbose=True):
    """L-BFGS-B로 IK 풀기 (다중 초기값 시도).

    Args:
        cup_pos          : (3,) 컵 root 위치 (world, sim 좌표)
        pregrasp_offset_y: palm y 접근 오프셋 (음수 = 컵 -Y 방향에서 접근)
        n_restarts       : 다중 초기값 시도 횟수
        verbose          : 상세 출력

    Returns:
        q_opt      : (7,) 최적 관절각
        target_pos : (3,) 목표 palm_link 위치
    """
    target_pos = compute_target_palm_link(cup_pos, pregrasp_offset_y)

    if verbose:
        print(f"\n[IK 입력]")
        print(f"  cup_pos            = {np.array(cup_pos).round(4).tolist()}")
        print(f"  pregrasp_offset_y  = {pregrasp_offset_y}")
        print(f"  GRASP_MIDPOINT_Z   = {GRASP_MIDPOINT_Z:.4f} m  "
              f"(thumb_base={THUMB_BASE_Z_FROM_PALM:.4f}, middle_tip={MIDDLE_TIP_Z_FROM_PALM:.4f})")
        print(f"  target palm_link   = {target_pos.round(4).tolist()}")
        print(f"  target palm_z_dir  = {TARGET_PALM_Z_DIR.tolist()}  (손가락 → world +X)")
        print(f"  target palm_x_dir  = {TARGET_PALM_X_DIR.tolist()}  (손바닥 법선 → world +Y)")

    best_q   = None
    best_val = float("inf")

    # 초기값 후보 생성
    rng = np.random.default_rng(42)
    q0_candidates = [np.array(Q0, dtype=float)]
    for _ in range(n_restarts - 1):
        noise = rng.uniform(-0.4, 0.4, 7)
        q0_candidates.append(np.array(Q0, dtype=float) + noise)

    for i, q0 in enumerate(q0_candidates):
        result = minimize(
            ik_objective,
            q0,
            args=(target_pos,),
            method="L-BFGS-B",
            bounds=JOINT_LIMITS,
            options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-9},
        )
        if verbose:
            pos_check, _, _ = get_palm_link_pose(result.x)
            pos_err_mm = np.linalg.norm(pos_check - target_pos) * 1000
            print(f"  restart {i:2d}: cost={result.fun:.6f}, pos_err={pos_err_mm:.1f} mm")
        if result.fun < best_val:
            best_val = result.fun
            best_q   = result.x.copy()

    return best_q, target_pos


# ---------------------------------------------------------------------------
# 결과 출력
# ---------------------------------------------------------------------------

def print_results(q_opt, cup_pos, pregrasp_offset_y):
    pos, x_dir, z_dir = get_palm_link_pose(q_opt)
    target_pos = compute_target_palm_link(cup_pos, pregrasp_offset_y)

    pos_err_m  = np.linalg.norm(pos - target_pos)
    ori_err_z  = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(z_dir, TARGET_PALM_Z_DIR))))))
    ori_err_x  = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(x_dir, TARGET_PALM_X_DIR))))))

    fk = arm_fk(q_opt)

    print(f"\n{'='*60}")
    print(f"[IK 결과]")
    print(f"  ARM_START_POSE = {[round(v, 4) for v in q_opt.tolist()]}")
    print(f"")
    print(f"  palm_link pos   = {pos.round(4).tolist()}")
    print(f"  target pos      = {target_pos.round(4).tolist()}")
    print(f"  위치 오차        = {pos_err_m*1000:.1f} mm")
    print(f"")
    print(f"  palm_z_dir      = {z_dir.round(3).tolist()}  (손가락, 목표 [1,0,0])")
    print(f"  palm_x_dir      = {x_dir.round(3).tolist()}  (손바닥, 목표 [0,1,0])")
    print(f"  손가락 방향 오차  = {ori_err_z:.1f}°")
    print(f"  손바닥 방향 오차  = {ori_err_x:.1f}°")
    print(f"")
    print(f"  palm_center     = {fk['palm_center'].round(4).tolist()}  (참고용)")

    # 손가락 위치 검증
    thumb_world  = pos + z_dir * THUMB_BASE_Z_FROM_PALM
    middle_world = pos + z_dir * MIDDLE_TIP_Z_FROM_PALM
    midpt_world  = (thumb_world + middle_world) / 2.0
    cup_center   = np.array([cup_pos[0], cup_pos[1], cup_pos[2] + CUP_GRASP_Z_OFFSET])

    print(f"")
    print(f"[손가락-컵 위치 검증]")
    print(f"  cup grasp center = {cup_center.round(4).tolist()}")
    print(f"  thumb base world = {thumb_world.round(4).tolist()}  "
          f"(ΔX={thumb_world[0]-cup_center[0]:+.3f} m)")
    print(f"  middle tip world = {middle_world.round(4).tolist()}  "
          f"(ΔX={middle_world[0]-cup_center[0]:+.3f} m)")
    print(f"  finger midpoint  x={midpt_world[0]:.4f}  "
          f"cup x={cup_center[0]:.4f}  diff={midpt_world[0]-cup_center[0]:+.4f} m")

    # 코드 출력
    q_r = [round(v, 4) for v in q_opt.tolist()]
    print(f"")
    print(f"{'='*60}")
    print(f"[grasp_right_preset.py 수정]")
    print(f"RIGHT_ARM_START_POSE = {q_r}")
    print(f"")
    print(f"[grasp_right_env_cfg.py init_state 수정]")
    for i, v in enumerate(q_r, 1):
        print(f'    "openarm_right_joint{i}": {v},')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="5g_grasp_right pregrasp ARM_START_POSE IK 계산",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cup", nargs=3, type=float, default=[0.40, -0.15, 0.215],
        metavar=("X", "Y", "Z"),
        help="컵 root 위치 (default: 0.40 -0.15 0.215)",
    )
    parser.add_argument(
        "--pregrasp-y", type=float, default=-0.09,
        help="pregrasp y 접근 오프셋 (default: -0.09, 음수=컵 -Y에서 접근)",
    )
    parser.add_argument(
        "--restarts", type=int, default=8,
        help="IK 다중 초기값 시도 횟수 (default: 8)",
    )
    parser.add_argument("--quiet", action="store_true", help="간략 출력")
    args = parser.parse_args()

    print(f"5g_grasp_right pregrasp ARM_START_POSE 최적화")
    print(f"{'='*60}")
    print(f"  thumb base  : {THUMB_BASE_Z_FROM_PALM:.4f} m from palm_link (+Z)")
    print(f"  middle tip  : {MIDDLE_TIP_Z_FROM_PALM:.4f} m from palm_link (+Z)")
    print(f"  grasp midpt : {GRASP_MIDPOINT_Z:.4f} m from palm_link (+Z)")

    q_opt, _ = solve_ik(
        cup_pos=args.cup,
        pregrasp_offset_y=args.pregrasp_y,
        n_restarts=args.restarts,
        verbose=not args.quiet,
    )

    print_results(q_opt, args.cup, args.pregrasp_y)


if __name__ == "__main__":
    main()
