"""SP2 좌표변환·intrinsics 순수 함수 (numpy only, Isaac 무관).

설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp2-camera-design.md §4.3
"""
from __future__ import annotations

import numpy as np


def validate_T(T: np.ndarray) -> None:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"invalid transform: shape={T.shape}, finite={np.isfinite(T).all()}")
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
        raise ValueError("rotation block not orthonormal")


def compose_local_pose(T_local_cam: np.ndarray, T_cam_obj: np.ndarray) -> np.ndarray:
    """env-local 물체 위치 = (T_local_cam @ T_cam_obj) 평행이동부."""
    validate_T(T_local_cam)
    validate_T(T_cam_obj)
    return (np.asarray(T_local_cam, dtype=float) @ np.asarray(T_cam_obj, dtype=float))[:3, 3].copy()


def k_from_pinhole(focal_length: float, horizontal_aperture: float,
                   width: int, height: int) -> np.ndarray:
    """Isaac PinholeCameraCfg → K. Isaac은 정방픽셀(수직 aperture는 종횡비 유도)."""
    fx = float(focal_length) / float(horizontal_aperture) * float(width)
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fx, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def pinhole_from_k(K: np.ndarray, width: int, height: int) -> tuple[float, float]:
    """실기 camera_info K → (focal_length, horizontal_aperture). fx 기준(정방픽셀 근사)."""
    K = np.asarray(K, dtype=float)
    if K.shape != (3, 3) or K[0, 0] <= 0:
        raise ValueError(f"invalid K: {K}")
    # focal/aperture 는 비율만 의미 → aperture 를 관례값 20.955(mm)로 두고 focal 역산
    aperture = 20.955
    focal = K[0, 0] / float(width) * aperture
    return focal, aperture
