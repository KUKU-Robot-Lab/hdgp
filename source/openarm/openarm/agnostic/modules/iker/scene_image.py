"""Head-camera frame -> numbered IKER keypoint image (design spec §4-5), independent of the simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from . import annotate, keypoints, projection
from .projection import CameraPose

AXIS_STEP_M = 0.1


@dataclass(frozen=True)
class PointGroup:
    object_name: str
    is_static: bool
    points_w: np.ndarray  # (K, 3) world, in the object's keypoint order


@dataclass(frozen=True)
class KeypointRecord:
    label: int
    object_name: str
    is_static: bool
    world: tuple[float, float, float]
    uv_raw: tuple[float, float]
    uv: tuple[float, float]
    depth: float
    render_depth: float | None
    visible: bool
    kept: bool


@dataclass(frozen=True)
class AnnotatedSnapshot:
    image: Image.Image
    derotation_deg: float
    records: tuple[KeypointRecord, ...]  # sorted by label
    axes: Mapping[str, tuple[float, float]]
    min_margin_px: float
    min_gap_px: float | None


def annotate_snapshot(
    rgb,
    depth,
    camera: CameraPose,
    intrinsic,
    groups: Sequence[PointGroup],
    anchor_w,
    overlap_min_px: float,
    occlusion_tol_m: float = 0.01,
) -> AnnotatedSnapshot:
    """Project, label, filter and draw the keypoints of one head-camera frame.

    ``depth`` is the renderer's distance to the image plane. A keypoint is occluded when the rendered
    surface is more than ``occlusion_tol_m`` in front of it. Raises if the overlap filter would remove
    a movable keypoint, because the reward tracks every one of them.
    """
    pixels = np.asarray(rgb)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("rgb must be (H, W, 3)")
    height, width = pixels.shape[:2]
    depth_image = np.asarray(depth, dtype=float)
    if depth_image.shape != (height, width):
        raise ValueError(f"depth must have shape {(height, width)}, got {depth_image.shape}")
    names, static, points_w = _flatten(groups)
    uv_raw, cam_depth = projection.project_points(points_w, camera, intrinsic)
    render_depth = _sample_depth(depth_image, uv_raw)
    visible = [d is not None and d + occlusion_tol_m >= z for d, z in zip(render_depth, cam_depth)]
    theta = projection.derotation_deg(anchor_w, camera, intrinsic)
    uv = projection.rotate_image_points(uv_raw, theta, (width, height))
    labels = keypoints.assign_labels(names, static, uv[:, 0])
    kept = keypoints.overlap_keep_mask(uv, labels, static, overlap_min_px)
    dropped = sorted(int(labels[i]) for i in range(len(labels)) if not static[i] and not kept[i])
    if dropped:
        raise ValueError(f"overlap filter removed movable keypoints {dropped}; every movable keypoint is needed")
    axes = _axes(anchor_w, camera, intrinsic, theta, (width, height))
    markers = _markers(names, static, labels, uv, visible, kept)
    image = annotate.draw_markers(annotate.rotate_image(pixels, theta), markers, axes)
    records = tuple(
        sorted(
            (
                KeypointRecord(
                    label=int(labels[i]),
                    object_name=names[i],
                    is_static=static[i],
                    world=tuple(float(c) for c in points_w[i]),
                    uv_raw=(float(uv_raw[i, 0]), float(uv_raw[i, 1])),
                    uv=(float(uv[i, 0]), float(uv[i, 1])),
                    depth=float(cam_depth[i]),
                    render_depth=render_depth[i],
                    visible=bool(visible[i]),
                    kept=bool(kept[i]),
                )
                for i in range(len(names))
            ),
            key=lambda record: record.label,
        )
    )
    kept_uv = uv[kept]
    gaps = [float(np.hypot(*(a - b))) for i, a in enumerate(kept_uv) for b in kept_uv[i + 1 :]]
    margin = float(projection.image_margin(kept_uv, (width, height)).min())
    return AnnotatedSnapshot(image, theta, records, axes, margin, min(gaps) if gaps else None)


def _flatten(groups: Sequence[PointGroup]) -> tuple[list[str], list[bool], np.ndarray]:
    if not groups:
        raise ValueError("no keypoint groups")
    names: list[str] = []
    static: list[bool] = []
    blocks = []
    for group in groups:
        points = np.asarray(group.points_w, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(f"{group.object_name}: points_w must be (K, 3) with K >= 1")
        names += [group.object_name] * len(points)
        static += [bool(group.is_static)] * len(points)
        blocks.append(points)
    return names, static, np.concatenate(blocks)


def _sample_depth(depth_image: np.ndarray, uv: np.ndarray) -> list[float | None]:
    height, width = depth_image.shape
    samples: list[float | None] = []
    for u, v in uv:
        col, row = math.floor(u), math.floor(v)
        inside = 0 <= col < width and 0 <= row < height
        value = float(depth_image[row, col]) if inside else math.nan
        samples.append(value if math.isfinite(value) else None)
    return samples


def _axes(anchor_w, camera: CameraPose, intrinsic, theta: float, size: tuple[int, int]) -> dict[str, tuple[float, float]]:
    anchor = np.asarray(anchor_w, dtype=float)
    probe = np.stack([anchor, anchor + [AXIS_STEP_M, 0.0, 0.0], anchor + [0.0, AXIS_STEP_M, 0.0]])
    uv, _ = projection.project_points(probe, camera, intrinsic)
    rotated = projection.rotate_image_points(uv, theta, size)
    dx, dy = rotated[1] - rotated[0], rotated[2] - rotated[0]
    return {"+x": (float(dx[0]), float(dx[1])), "+y": (float(dy[0]), float(dy[1]))}


def _markers(names, static, labels, uv, visible, kept) -> list[annotate.Marker]:
    movable_order: list[str] = []
    for i in np.argsort(labels):
        if not static[i] and names[i] not in movable_order:
            movable_order.append(names[i])
    markers = []
    for i in range(len(names)):
        if not kept[i]:
            continue
        color = annotate.STATIC_COLOR if static[i] else annotate.movable_color(movable_order.index(names[i]))
        markers.append(annotate.Marker(int(labels[i]), (float(uv[i, 0]), float(uv[i, 1])), color, bool(visible[i])))
    return markers
