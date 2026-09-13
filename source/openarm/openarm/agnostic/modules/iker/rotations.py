"""Rotation helpers for the IKER modules (numpy only).

Quaternions are ``wxyz`` (Isaac Lab order). Matrices rotate column vectors.
"""

from __future__ import annotations

import numpy as np

_ORTHONORMAL_TOL = 1e-6


def finite_array(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    """``value`` as a float array of ``shape``; raises unless every entry is finite."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def quat_wxyz_to_matrix(quat) -> np.ndarray:
    q = finite_array("quat", quat, (4,))
    norm = float(np.linalg.norm(q))
    if not norm > 1e-9:
        raise ValueError("quat has zero norm")
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_wxyz(matrix) -> np.ndarray:
    """Unit quaternion with ``w >= 0`` (Shepperd's method)."""
    m = finite_array("matrix", matrix, (3, 3))
    if not (np.allclose(m.T @ m, np.eye(3), atol=_ORTHONORMAL_TOL) and np.linalg.det(m) > 0.0):
        raise ValueError("matrix is not a proper rotation")
    trace = float(np.trace(m))
    k = int(np.argmax([trace, m[0, 0], m[1, 1], m[2, 2]]))
    if k == 0:
        w = np.sqrt(1.0 + trace) / 2.0
        q = [w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)]
    elif k == 1:
        x = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) / 2.0
        q = [(m[2, 1] - m[1, 2]) / (4 * x), x, (m[0, 1] + m[1, 0]) / (4 * x), (m[0, 2] + m[2, 0]) / (4 * x)]
    elif k == 2:
        y = np.sqrt(1.0 - m[0, 0] + m[1, 1] - m[2, 2]) / 2.0
        q = [(m[0, 2] - m[2, 0]) / (4 * y), (m[0, 1] + m[1, 0]) / (4 * y), y, (m[1, 2] + m[2, 1]) / (4 * y)]
    else:
        z = np.sqrt(1.0 - m[0, 0] - m[1, 1] + m[2, 2]) / 2.0
        q = [(m[1, 0] - m[0, 1]) / (4 * z), (m[0, 2] + m[2, 0]) / (4 * z), (m[1, 2] + m[2, 1]) / (4 * z), z]
    quat = np.asarray(q, dtype=float)
    return quat if quat[0] >= 0.0 else -quat


def rpy_xyz_to_matrix(rpy) -> np.ndarray:
    """Extrinsic x-y-z Euler angles (scipy ``from_euler("xyz")``), the convention of the IKER assets."""
    roll, pitch, yaw = finite_array("rpy", rpy, (3,))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def yaw_matrix(yaw_rad: float) -> np.ndarray:
    return rpy_xyz_to_matrix((0.0, 0.0, yaw_rad))
