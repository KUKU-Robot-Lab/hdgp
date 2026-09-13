"""Head-camera geometry (design spec §5).

The camera pose is the ``head_camera`` link pose composed with the calibrated optical offset
(``sim2real/config/head_camera_sim.json``, ROS optical frame: +x right, +y down, +z forward).
Isaac Lab's ``Camera.data.pos_w`` / ``quat_w_ros`` stay zero/NaN for a camera attached after the
scene is built, so the pose is always computed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .rotations import finite_array, quat_wxyz_to_matrix

MIN_DEPTH_M = 1e-6


@dataclass(frozen=True)
class CameraPose:
    position: np.ndarray  # (3,) world
    rotation: np.ndarray  # (3, 3) columns: optical x, y, z axes in world


def camera_pose_from_link(link_pos, link_quat_wxyz, offset_pos, offset_quat_wxyz) -> CameraPose:
    link_rotation = quat_wxyz_to_matrix(link_quat_wxyz)
    offset = finite_array("offset_pos", offset_pos, (3,))
    position = finite_array("link_pos", link_pos, (3,)) + link_rotation @ offset
    return CameraPose(position=position, rotation=link_rotation @ quat_wxyz_to_matrix(offset_quat_wxyz))


def project_points(points_w, camera: CameraPose, intrinsic) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (N, 2) and optical depth (N,) of world points."""
    points = np.asarray(points_w, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points_w must be a finite (N, 3) array")
    k = finite_array("intrinsic", intrinsic, (3, 3))
    position = finite_array("camera.position", camera.position, (3,))
    rotation = finite_array("camera.rotation", camera.rotation, (3, 3))
    cam = (points - position) @ rotation
    depth = cam[:, 2]
    if not np.all(depth > MIN_DEPTH_M):
        raise ValueError("a keypoint lies behind the camera")
    u = k[0, 0] * cam[:, 0] / depth + k[0, 2]
    v = k[1, 1] * cam[:, 1] / depth + k[1, 2]
    return np.stack([u, v], axis=1), depth


def derotation_deg(anchor_w, camera: CameraPose, intrinsic, step_m: float = 0.1) -> float:
    """Counter-clockwise image rotation (PIL ``Image.rotate``) that makes projected world +x point up."""
    anchor = finite_array("anchor_w", anchor_w, (3,))
    uv, _ = project_points(np.stack([anchor, anchor + [step_m, 0.0, 0.0]]), camera, intrinsic)
    du, dv = uv[1] - uv[0]
    if not math.hypot(du, dv) > 1e-6:
        raise ValueError("world +x projects to a single pixel at the anchor")
    return 90.0 - math.degrees(math.atan2(-dv, du))


def rotate_image_points(uv, theta_deg: float, size: tuple[int, int]) -> np.ndarray:
    """Where pixels land after ``Image.rotate(theta_deg, expand=False)`` about the image centre."""
    points = np.asarray(uv, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("uv must be a finite (N, 2) array")
    if not math.isfinite(theta_deg):
        raise ValueError("theta_deg must be finite")
    width, height = size
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    x = points[:, 0] - width / 2.0
    y = -(points[:, 1] - height / 2.0)
    return np.stack([width / 2.0 + (c * x - s * y), height / 2.0 - (s * x + c * y)], axis=1)


def image_margin(uv, size: tuple[int, int]) -> np.ndarray:
    """Distance (px) from each point to the nearest image border; negative outside the image."""
    points = np.asarray(uv, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("uv must be a finite (N, 2) array")
    width, height = size
    sides = np.stack([points[:, 0], width - points[:, 0], points[:, 1], height - points[:, 1]], axis=1)
    return sides.min(axis=1)
