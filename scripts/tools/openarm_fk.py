#!/usr/bin/env python3
"""OpenArm + Teosllo FK 계산 도구

URDF 기반 순운동학(Forward Kinematics) — 외부 라이브러리 불필요 (numpy only).
palm_center, palm_x 위치를 시뮬레이션 좌표계(sim)로 출력.

URDF: /home/user/rl_ws/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_tesollo/openarm_tesollo.urdf

캘리브레이션:
  시뮬레이션에서 실측한 기준점으로 FK_raw → sim 좌표 보정.
  기준: q=[0.5, 0.5, -0.6, 0.7, 0, 0, 1.0] → sim_pos=[0.281, -0.270, 0.424]
  offset = FK_raw(q_ref) - sim_pos_ref (x,y,z 각각 일정값으로 보정)

사용법:
  python3 openarm_fk.py
  python3 openarm_fk.py --q "0.5 0.1 0.4 1.3 -0.2 0.0 0.0"
  python3 openarm_fk.py --q "0.5 0.1 0.4 1.3 -0.2 0.0 0.0" --no-offset
  python3 openarm_fk.py --scan-j4          # j4 스캔 (z 높이 확인용)
  python3 openarm_fk.py --scan-j1j2        # j1,j2 스캔 (reach 확인용)
"""

import argparse
import math
import numpy as np


# ---------------------------------------------------------------------------
# SE(3) 유틸리티
# ---------------------------------------------------------------------------

def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

def rpy_to_rot(r: float, p: float, y: float) -> np.ndarray:
    """URDF RPY 순서: Rz(y) @ Ry(p) @ Rx(r)"""
    return rot_z(y) @ rot_y(p) @ rot_x(r)

def make_T(xyz, rpy) -> np.ndarray:
    """4×4 homogeneous transform from URDF <origin xyz rpy>"""
    T = np.eye(4)
    T[:3, :3] = rpy_to_rot(*rpy)
    T[:3, 3] = xyz
    return T

def rot_axis(axis: np.ndarray, q: float) -> np.ndarray:
    """단위 axis 벡터 + 각도 → 3×3 회전행렬 (Rodrigues)"""
    c, s = math.cos(q), math.sin(q)
    K = np.array([
        [    0, -axis[2],  axis[1]],
        [ axis[2],    0, -axis[0]],
        [-axis[1],  axis[0],    0],
    ])
    return np.eye(3) + s * K + (1 - c) * (K @ K)

def joint_T(xyz, rpy, axis, q: float) -> np.ndarray:
    """URDF joint 하나의 변환: T_origin × R_axis(q)"""
    T_origin = make_T(xyz, rpy)
    R = rot_axis(np.array(axis, dtype=float), q)
    T_rot = np.eye(4)
    T_rot[:3, :3] = R
    return T_origin @ T_rot


# ---------------------------------------------------------------------------
# OpenArm 7-DOF arm URDF 파라미터
# (openarm_tesollo.urdf 기준, 관절 1~7)
# ---------------------------------------------------------------------------

ARM_JOINTS = [
    # (xyz,                      rpy,                           axis)
    ([0.,   -0.0935,  0.6979998], [1.5708,    0.,       0.    ], [0, 0, 1]),  # j1
    ([-0.0301, 0.,    0.06     ], [-1.5707963, 0., 3.1415927  ], [1, 0, 0]),  # j2
    ([-0.0301, 0.,    0.06625  ], [0.,         0., 3.1415927  ], [0, 0, 1]),  # j3
    ([0.,    0.0315,  0.15375  ], [0.,         0.,       0.   ], [0, 1, 0]),  # j4
    ([0.,   -0.0315,  0.0955   ], [0.,         0.,       0.   ], [0, 0, 1]),  # j5
    ([0.0375, 0.,     0.1205   ], [0.,         0.,       0.   ], [1, 0, 0]),  # j6
    ([-0.0375, 0.,    0.       ], [0.,         0.,       0.   ], [0, 1, 0]),  # j7
]

# palm_link: fixed joint from link7
# xyz=[0, 0.0000003, 0.1333695], rpy=[0, 0, -pi/2]
T_PALM_LINK = make_T([0., 0.0000003, 0.1333695], [0., 0., -1.5707964])

# palm_center: fixed joint from palm_link
# xyz=[0, 0.03, 0.04], rpy=[0, 0, 0]
T_PALM_CENTER = make_T([0., 0.03, 0.04], [0., 0., 0.])

# palm_x: fixed joint from palm_link — 손바닥 법선 방향 마커 (+X = 손바닥)
# xyz=[0.25, 0, 0], rpy=[0, 0, 0]
T_PALM_X = make_T([0.25, 0., 0.], [0., 0., 0.])


# ---------------------------------------------------------------------------
# 캘리브레이션 오프셋
# ---------------------------------------------------------------------------
# 시뮬레이션 실측 기준점:
#   q_ref = [0.5, 0.5, -0.6, 0.7, 0, 0, 1.0]
#   sim_pos_ref = [0.281, -0.270, 0.424]  (palm_center 실측)
# offset = FK_raw(q_ref) - sim_pos_ref
#
# 계산된 오프셋 (스크립트 최초 실행 시 자동 계산됨)
# 필요 시 --calibrate 로 재계산 가능.
Q_REF      = [0.5, 0.5, -0.6, 0.7, 0., 0., 1.0]
SIM_POS_REF = np.array([0.281, -0.270, 0.424])   # palm_center 시뮬 실측값


def arm_fk_raw(q: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    7-DOF arm FK (보정 없음, URDF 원시 좌표).

    Returns:
        T_world: (4,4) link7 끝단 변환
        palm_center_raw: (3,) palm_center 위치
        palm_x_raw: (3,) palm_x 마커 위치 (손바닥 법선 끝점)
    """
    T = np.eye(4)
    for i, (xyz, rpy, axis) in enumerate(ARM_JOINTS):
        T = T @ joint_T(xyz, rpy, axis, q[i])

    T_world = T  # openarm_right_link7 frame

    T_palm = T_world @ T_PALM_LINK
    palm_center_raw = (T_palm @ T_PALM_CENTER)[:3, 3]
    palm_x_raw      = (T_palm @ T_PALM_X)[:3, 3]

    return T_world, palm_center_raw, palm_x_raw


def compute_offset() -> np.ndarray:
    """FK_raw(Q_REF) - SIM_POS_REF 로 오프셋 계산."""
    _, palm_center_raw, _ = arm_fk_raw(Q_REF)
    return palm_center_raw - SIM_POS_REF


# 모듈 로드 시 오프셋 사전 계산
_OFFSET = compute_offset()


def arm_fk(q: list[float], apply_offset: bool = True) -> dict:
    """
    7-DOF arm FK (시뮬레이션 좌표 보정 포함).

    Args:
        q: [j1..j7] 관절 각도 (rad), 7개 값
        apply_offset: True → sim 좌표 보정 적용

    Returns:
        dict with:
          palm_center: (3,) palm_center 시뮬 위치
          palm_x:      (3,) palm_x 마커 시뮬 위치 (손바닥 법선 끝)
          palm_x_dir:  (3,) 손바닥 법선 단위 방향 벡터
          palm_z_dir:  (3,) 손가락 방향 단위 벡터 (palm_link +Z)
          link7_R:     (3,3) link7 회전행렬
          raw_palm_center: (3,) 보정 전 원시 좌표
    """
    if len(q) != 7:
        raise ValueError(f"q must have 7 elements, got {len(q)}")

    T_world, palm_center_raw, palm_x_raw = arm_fk_raw(q)

    # palm_link 변환 (link7 @ T_PALM_LINK)
    T_palm = T_world @ T_PALM_LINK
    # palm_link +X(손바닥 법선), +Z(손가락) — palm_link 좌표계 축
    palm_x_dir_raw = T_palm[:3, 0]  # palm_link +X = 손바닥 법선
    palm_z_dir_raw = T_palm[:3, 2]  # palm_link +Z = 손가락 방향

    if apply_offset:
        palm_center = palm_center_raw - _OFFSET
        palm_x      = palm_x_raw - _OFFSET        # 마커도 동일 오프셋
    else:
        palm_center = palm_center_raw
        palm_x      = palm_x_raw

    palm_x_dir = palm_x_dir_raw                  # 방향 벡터는 오프셋 무관
    palm_z_dir = palm_z_dir_raw

    return {
        "palm_center":     palm_center,
        "palm_x":          palm_x,
        "palm_x_dir":      palm_x_dir,
        "palm_z_dir":      palm_z_dir,
        "link7_R":         T_world[:3, :3],
        "raw_palm_center": palm_center_raw,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_fk(q: list[float], apply_offset: bool = True) -> None:
    r = arm_fk(q, apply_offset)
    tag = "" if apply_offset else " [RAW, no offset]"
    print(f"\nq = {[round(v, 4) for v in q]}{tag}")
    print(f"  palm_center : [{r['palm_center'][0]:.4f}, {r['palm_center'][1]:.4f}, {r['palm_center'][2]:.4f}]")
    print(f"  palm_x_dir  : [{r['palm_x_dir'][0]:.3f}, {r['palm_x_dir'][1]:.3f}, {r['palm_x_dir'][2]:.3f}]  (손바닥 법선)")
    print(f"  palm_z_dir  : [{r['palm_z_dir'][0]:.3f}, {r['palm_z_dir'][1]:.3f}, {r['palm_z_dir'][2]:.3f}]  (손가락 방향)")


def scan_j4(q_base: list[float], j4_range=(0.0, 2.0), steps=20, apply_offset=True) -> None:
    print(f"\n[j4 scan]  base_q={q_base}  offset={'on' if apply_offset else 'off'}")
    print(f"{'j4':>6}  {'x':>7}  {'y':>7}  {'z':>7}  palm_x_dir")
    j4_vals = np.linspace(*j4_range, steps)
    for j4 in j4_vals:
        q = q_base.copy()
        q[3] = j4
        r = arm_fk(q, apply_offset)
        pc = r["palm_center"]
        xd = r["palm_x_dir"]
        print(f"{j4:6.3f}  {pc[0]:7.4f}  {pc[1]:7.4f}  {pc[2]:7.4f}  "
              f"[{xd[0]:.3f},{xd[1]:.3f},{xd[2]:.3f}]")


def scan_j1j2(q_base: list[float], j1_vals=None, j2_vals=None, apply_offset=True) -> None:
    if j1_vals is None:
        j1_vals = np.linspace(0.0, 1.5, 5)
    if j2_vals is None:
        j2_vals = np.linspace(-0.3, 0.3, 5)
    print(f"\n[j1,j2 scan]  base_q={q_base}")
    print(f"{'j1':>5}  {'j2':>5}  {'x':>7}  {'y':>7}  {'z':>7}")
    for j1 in j1_vals:
        for j2 in j2_vals:
            q = q_base.copy()
            q[0] = j1
            q[1] = j2
            r = arm_fk(q, apply_offset)
            pc = r["palm_center"]
            print(f"{j1:5.2f}  {j2:5.2f}  {pc[0]:7.4f}  {pc[1]:7.4f}  {pc[2]:7.4f}")


def main():
    parser = argparse.ArgumentParser(description="OpenArm FK 계산 도구")
    parser.add_argument("--q", type=str, default=None,
                        help='관절 값 (7개, 공백 구분). 예: "0.5 0.1 0.4 1.3 -0.2 0.0 0.0"')
    parser.add_argument("--no-offset", action="store_true",
                        help="시뮬레이션 캘리브레이션 오프셋 미적용 (URDF raw 좌표)")
    parser.add_argument("--scan-j4", action="store_true",
                        help="j4 범위 스캔 (z 높이 확인)")
    parser.add_argument("--scan-j1j2", action="store_true",
                        help="j1,j2 범위 스캔 (reach 확인)")
    parser.add_argument("--calibrate", action="store_true",
                        help="현재 캘리브레이션 오프셋 출력")
    args = parser.parse_args()

    apply_offset = not args.no_offset

    # 캘리브레이션 정보 출력
    if args.calibrate:
        print(f"\n[캘리브레이션]")
        print(f"  Q_REF      = {Q_REF}")
        print(f"  SIM_POS_REF= {SIM_POS_REF.tolist()}")
        _, raw, _ = arm_fk_raw(Q_REF)
        print(f"  FK_raw     = [{raw[0]:.6f}, {raw[1]:.6f}, {raw[2]:.6f}]")
        print(f"  offset     = [{_OFFSET[0]:.6f}, {_OFFSET[1]:.6f}, {_OFFSET[2]:.6f}]")
        return

    # 기본 q (ARM_START_POSE 현재값)
    DEFAULT_Q = [0.5, 0.1, 0.4, 1.3, -0.2, 0.0, 0.0]

    if args.q:
        q = list(map(float, args.q.split()))
        print_fk(q, apply_offset)
    elif args.scan_j4:
        q_base = DEFAULT_Q.copy()
        scan_j4(q_base, apply_offset=apply_offset)
    elif args.scan_j1j2:
        q_base = DEFAULT_Q.copy()
        scan_j1j2(q_base, apply_offset=apply_offset)
    else:
        # 기본: 주요 포즈들 출력
        poses = {
            "ARM_START_POSE (j4=1.3, current)": [0.5, 0.1, 0.4, 1.3, -0.2, 0.0, 0.0],
            "test7 pose     (j4=0.8, old)    ": [0.5, 0.1, 0.4, 0.8, -0.2, 0.0, 0.0],
            "DEXTRAH ref    (j4=0.7, FK base)": [0.5, 0.5, -0.6, 0.7, 0.0, 0.0, 1.0],
            "working pose   (j1=1.0, j2=-0.1)": [1.0, -0.1, 0.0, 0.5, 0.0, 0.0, 0.0],
        }
        print(f"\n캘리브레이션 오프셋: {_OFFSET.round(4).tolist()}")
        for label, q in poses.items():
            r = arm_fk(q, apply_offset)
            pc = r["palm_center"]
            xd = r["palm_x_dir"]
            zd = r["palm_z_dir"]
            print(f"\n[{label}]  q={q}")
            print(f"  palm_center: [{pc[0]:.4f}, {pc[1]:.4f}, {pc[2]:.4f}]")
            print(f"  palm_x_dir : [{xd[0]:.3f}, {xd[1]:.3f}, {xd[2]:.3f}]  (손바닥 법선, 목표: [0,1,0])")
            print(f"  palm_z_dir : [{zd[0]:.3f}, {zd[1]:.3f}, {zd[2]:.3f}]  (손가락 방향, 목표: [1,0,0])")


if __name__ == "__main__":
    main()
