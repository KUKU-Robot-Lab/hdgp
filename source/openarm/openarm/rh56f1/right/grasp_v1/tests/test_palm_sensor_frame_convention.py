"""palm_sensor 프레임 규약 수치 검증 (렌더 없이 자세 정확성 확정).

fabric IK 를 palm_link(Tesollo 가상) → r_hl_palm_sensor(실제) 로 정합한 뒤,
env 의 palm pose euler 규약이 손 배치를 보존하고 손바닥(+z)이 컵을 향하는지 검증한다.

핵심 사실 (URDF FK 로 유도, verify_palm_sensor_fk.py 와 정합):
  - palm_link → palm_sensor 상대회전 R_ls = Rx(90°).
  - euler_zyx 의 마지막 회전이 Rx 이므로 R(ez,ey,ex)@Rx(90) = R(ez,ey,ex+90).
    → palm_link 규약 → palm_sensor 규약 = ex 에 +90° (선형).
  - 기존 pregrasp (ez,ey,ex)=(90,0,90)[palm_link] → palm_sensor (90,0,180).
"""
from __future__ import annotations

import math

import numpy as np

from openarm.rh56f1.right.grasp_v1.grasp_right_preset import (
    palm_pose_mins,
    palm_pose_maxs,
)


def _Rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _Ry(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _Rx(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


R_LS = _Rx(math.radians(90.0))  # palm_link → palm_sensor (URDF FK 유도)
PREGRASP_SENSOR_EULER = (math.radians(90.0), 0.0, math.radians(180.0))  # (ez, ey, ex)


def test_pregrasp_palm_faces_cup_downward():
    """새 규약 palm_sensor target (90,0,180) → palm_sensor +z_world 가 아래(컵)."""
    ez, ey, ex = PREGRASP_SENSOR_EULER
    R_sensor = _Rz(ez) @ _Ry(ey) @ _Rx(ex)
    z_world = R_sensor[:, 2]
    assert np.allclose(z_world, [0.0, 0.0, -1.0], atol=1e-9), z_world


def test_ex_plus_90_preserves_hand_placement():
    """ex+90(=R_ls 적용)이 기존 palm_link 배치를 정확히 보존."""
    # 기존 pregrasp 는 palm_LINK 를 (90,0,90) 자세로 만들었다.
    R_link_old = _Rz(math.radians(90.0)) @ _Ry(0.0) @ _Rx(math.radians(90.0))
    # 그 손배치의 palm_sensor 자세 = R_link_old @ R_ls
    R_sensor_from_old = R_link_old @ R_LS
    # 새 규약은 palm_sensor 를 직접 (90,0,180) 으로 지정
    ez, ey, ex = PREGRASP_SENSOR_EULER
    R_sensor_new = _Rz(ez) @ _Ry(ey) @ _Rx(ex)
    assert np.allclose(R_sensor_from_old, R_sensor_new, atol=1e-9)


def test_preset_palm_pose_mins_maxs_ex_centered_at_180():
    """preset palm_pose_mins/maxs 의 ex(index 5) 중심이 palm_sensor 규약(180°)."""
    max_angle = 30.0
    mins = palm_pose_mins(max_angle)
    maxs = palm_pose_maxs(max_angle)
    ex_center = math.degrees((mins[5] + maxs[5]) / 2.0)
    assert abs(ex_center - 180.0) < 1e-6, ex_center
    # ez(index 3) 는 여전히 90° 중심(R_ls 는 ex 축만 영향).
    ez_center = math.degrees((mins[3] + maxs[3]) / 2.0)
    assert abs(ez_center - 90.0) < 1e-6, ez_center
