"""camera: look-at 자세 계산 (warp 스텁으로 로드)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "camera.py"


def _load():
    warp = types.ModuleType("warp")
    warp.mat44f = lambda: np.zeros((4, 4), dtype=np.float32)
    sys.modules.setdefault("warp", warp)

    spec = importlib.util.spec_from_file_location("_camera_under_test", _SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


camera = _load()


def _rotate(quat, vec):
    """(w,x,y,z) 쿼터니언으로 벡터를 회전."""
    w, x, y, z = quat
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return rot @ np.asarray(vec, dtype=float)


def test_camera_z_axis_points_at_target():
    # ROS 규약: +z = 시선. 회전된 +z 가 pos→target 방향과 같아야 한다.
    pos, target = [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]

    quat = camera.look_at_quat(pos, target)

    expected = np.array(target) - np.array(pos)
    expected /= np.linalg.norm(expected)
    assert np.allclose(_rotate(quat, [0, 0, 1]), expected, atol=1e-6)


def test_camera_y_axis_points_down():
    # ROS 규약: +y = 아래. 수평 시선이면 회전된 +y 의 z 성분이 음수여야 한다.
    quat = camera.look_at_quat([1.0, 0.0, 0.5], [0.0, 0.0, 0.5])

    assert _rotate(quat, [0, 1, 0])[2] < 0


def test_quaternion_is_unit_norm():
    quat = camera.look_at_quat([1.05, -0.10, 0.75], [0.30, -0.10, 0.32])

    assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-6)


def test_left_right_mirror_shares_rotation_when_gaze_lies_in_xz_plane():
    # 카메라와 목표의 y 가 같으면 시선이 x-z 평면 안에 있어 y 미러가 회전을 안 바꾼다.
    # preset 의 좌우 CAMERA_ROT 가 같은 값인 이유 — 실측 캘리브레이션 후엔 성립하지 않는다.
    right = camera.look_at_quat([1.05, -0.10, 0.75], [0.30, -0.10, 0.32])
    left = camera.look_at_quat([1.05, 0.10, 0.75], [0.30, 0.10, 0.32])

    assert np.allclose(right, left, atol=1e-9)


def test_gaze_parallel_to_up_is_rejected():
    # 바로 위에서 수직으로 내려다보면 roll 이 정의되지 않는다 → 조용히 이상한 자세를
    # 내놓는 대신 터뜨린다
    with pytest.raises(ValueError, match="up 축과 평행"):
        camera.look_at_quat([0.3, -0.1, 1.5], [0.3, -0.1, 0.3])


def test_zero_length_gaze_is_rejected():
    with pytest.raises(ValueError, match="같다"):
        camera.look_at_quat([0.3, -0.1, 0.5], [0.3, -0.1, 0.5])
