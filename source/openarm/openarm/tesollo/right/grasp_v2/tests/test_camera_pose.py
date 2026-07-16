"""정면-사선 카메라 extrinsics 검증 (top-down 가림 회피 배치).

카메라가 (a) 물체를 실제로 바라보고 (b) 물체 앞(+x)에서 (c) 완만한 하향각으로
(d) D435i 최소거리(0.3m) 밖에 있어야 top-down 하강 시 물체 가림이 최소화된다.
좌우가 y-미러(위치 y 반전, 회전 동일)인지도 고정한다 — 미러가 깨지면 한쪽 뷰가
어긋난다.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
LEFT_PKG = PKG.parents[1] / "left" / "grasp_v2"

# 물체 중심(env_cfg object_spawn_*_center; isaaclab 없이 로드 불가라 설계값을 명시).
OBJ_R = (0.27, -0.10, 0.31)
OBJ_L = (0.27, 0.10, 0.31)

# 정면 하향각 허용 범위(deg): DEXTRAH-like 배치 27.1°. over-shoulder 65° 는 배제.
DOWN_MIN, DOWN_MAX = 20.0, 58.0


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


r = _load("_cam_r_preset", PKG / "grasp_right_preset.py")
l = _load("_cam_l_preset", LEFT_PKG / "grasp_left_preset.py")


def _forward(quat):
    """ROS 카메라 시선 = 로컬 +z 의 월드 방향 (회전행렬 3번째 열)."""
    w, x, y, z = quat
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def _check(pos, quat, obj):
    fwd = _forward(quat)
    to_obj = [obj[i] - pos[i] for i in range(3)]
    d = math.sqrt(sum(v * v for v in to_obj))
    unit = [v / d for v in to_obj]
    cos = sum(fwd[i] * unit[i] for i in range(3))
    horiz = math.hypot(to_obj[0], to_obj[1])
    down = math.degrees(math.atan2(-to_obj[2], horiz))
    return d, cos, down


# 단일 중앙 카메라 → 워크스페이스 중앙 정조준(각 팔 물체 사이 y=0).
CENTER = (0.27, 0.0, 0.31)


def test_camera_looks_at_workspace_center():
    d, cos, down = _check(r.CAMERA_POS, r.CAMERA_ROT, CENTER)
    assert cos > 0.999, f"시선이 중앙을 향하지 않음 (cos={cos:.4f})"
    assert d >= 0.3, f"D435i 최소거리 미만 (dist={d:.3f})"
    assert DOWN_MIN <= down <= DOWN_MAX, f"하향각 범위 벗어남 ({down:.1f}deg)"
    assert r.CAMERA_POS[0] > CENTER[0], "카메라가 워크스페이스 앞(+x)에 있지 않음"


def test_left_right_camera_identical():
    # 실물 카메라 1대 → 좌우 grasp_v2 가 완전히 같은 POS·ROT 를 써야 한다(미러 아님).
    assert list(l.CAMERA_POS) == list(r.CAMERA_POS)
    assert list(l.CAMERA_ROT) == list(r.CAMERA_ROT)


def test_each_arm_object_within_fov():
    # 중앙 카메라라도 각 팔 물체(y=∓0.10)가 광각 87° 프레임 안(HFOV/2=43.5°)에 들어와야 한다.
    hfov_half = math.degrees(math.atan(r.CAMERA_HORIZONTAL_APERTURE / (2 * r.CAMERA_FOCAL_LENGTH)))
    for obj in (OBJ_R, OBJ_L):
        _, cos, _ = _check(r.CAMERA_POS, r.CAMERA_ROT, obj)
        off = math.degrees(math.acos(min(1.0, cos)))
        assert off < hfov_half, f"물체가 FOV 밖 ({off:.1f}deg > {hfov_half:.1f})"


def test_intrinsics_d435i_aspect():
    # 16:9 유지 + D435i 최소거리 0.3m (sim2real 갭 방지).
    assert r.CAMERA_IMG_WIDTH == 320 and r.CAMERA_IMG_HEIGHT == 180
    assert r.CAMERA_CLIPPING_RANGE[0] == 0.3


def test_depth_band_covers_dextrah_distance():
    # DEXTRAH-like ~0.9m 거리 + 광각이라 depth far 를 2.0m 로 올려야 워크스페이스가 밴드 안에 든다.
    # 좌우 동일해야 미러 student 가 같은 depth 통계를 본다.
    assert r.CAMERA_D_MAX == 2.0 and l.CAMERA_D_MAX == 2.0
    assert r.CAMERA_D_MIN == 0.3 and l.CAMERA_D_MIN == 0.3
