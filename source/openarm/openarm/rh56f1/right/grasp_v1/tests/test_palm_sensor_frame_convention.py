"""palm_sensor 프레임 규약 수치 검증 (렌더 없이 자세 정확성 확정).

fabric IK 가 실제 r_hl_palm_sensor 를 직접 제어한다. side grasp 이므로 손바닥
법선(palm_sensor +z)이 컵(수평 +x 방향)을 향해야 한다.

핵심 사실:
  - 컵은 palm 옆·같은 높이(pregrasp_offset z≈0)에 있으므로 palm 법선은 수평이어야 한다.
    palm 이 아래(테이블)를 향하면 손날로 컵을 미는 파지가 된다(관측된 버그).
  - euler_zyx (ez,ey,ex)=(90,0,90) → palm_sensor +z_world = Rz(90)·Rx(90)·[0,0,1] = (+1,0,0)=+x(컵).
  - ex=180 은 palm_sensor +z 를 (0,0,-1)=테이블로 돌려 손날 파지를 유발한 버그였다.
    rh56f1 은 Tesollo palm_link 가상프레임(실제 palm 대비 Rx90 어긋남)을 이식받아
    이를 (90,0,90)으로 제어했고, 그 결과 실제 palm_sensor 는 (90,0,180)=아래를 향했다.
    → 재설계 이전부터 줄곧 손날 파지였으며, 올바른 값은 palm_sensor ex=90 이다.
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


PREGRASP_SENSOR_EULER = (math.radians(90.0), 0.0, math.radians(90.0))  # (ez, ey, ex)


def test_pregrasp_palm_faces_cup_horizontally():
    """pregrasp palm_sensor (90,0,90) → palm_sensor +z_world 가 수평 +x(컵)."""
    ez, ey, ex = PREGRASP_SENSOR_EULER
    R_sensor = _Rz(ez) @ _Ry(ey) @ _Rx(ex)
    z_world = R_sensor[:, 2]
    assert np.allclose(z_world, [1.0, 0.0, 0.0], atol=1e-9), z_world


def test_ex_180_would_face_table_blade_grasp():
    """ex=180(구 버그)은 palm_sensor +z 를 테이블(-z)로 돌려 손날 파지를 유발함을 명시."""
    R_bug = _Rz(math.radians(90.0)) @ _Ry(0.0) @ _Rx(math.radians(180.0))
    z_world = R_bug[:, 2]
    assert np.allclose(z_world, [0.0, 0.0, -1.0], atol=1e-9), z_world


def test_preset_palm_pose_mins_maxs_ex_centered_at_90():
    """preset palm_pose_mins/maxs 의 ex(index 5) 중심이 컵-향 규약(90°)."""
    max_angle = 30.0
    mins = palm_pose_mins(max_angle)
    maxs = palm_pose_maxs(max_angle)
    ex_center = math.degrees((mins[5] + maxs[5]) / 2.0)
    assert abs(ex_center - 90.0) < 1e-6, ex_center
    # ez(index 3) 도 90° 중심.
    ez_center = math.degrees((mins[3] + maxs[3]) / 2.0)
    assert abs(ez_center - 90.0) < 1e-6, ez_center
