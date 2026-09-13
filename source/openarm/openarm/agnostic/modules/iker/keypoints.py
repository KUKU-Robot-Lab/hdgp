"""IKER candidate keypoints (paper §III-A, design spec §4).

Movable objects get the mesh extremities on the two local axes that are horizontal at rest; static
objects get a uniform grid on their top surface. Labels are grouped by object, movable objects
first from left to right in the derotated image. The overlap filter follows the paper: static
points near movable ones are removed first, and among movable points the lower label survives.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

HORIZONTAL_TOL = 1e-6


def snap_rotation(matrix) -> np.ndarray:
    """Nearest rotation whose columns are signed world axes (removes a scan's residual tilt)."""
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError("matrix must be a finite 3x3 array")
    snapped = np.zeros((3, 3))
    for col in range(3):
        row = int(np.argmax(np.abs(m[:, col])))
        snapped[row, col] = np.sign(m[row, col])
    is_permutation = np.allclose(np.abs(snapped).sum(axis=1), 1.0)
    if not (is_permutation and np.linalg.det(snapped) > 0.0):
        raise ValueError("matrix does not snap to an axis-aligned proper rotation")
    return snapped


def horizontal_axes(rest_rotation, vertices) -> tuple[int, int]:
    """The two local axes that are horizontal at rest, longest mesh extent first.

    Sorting all three axes by size would pick the shoe's height (0.104 m) over its width (0.096 m).
    """
    rotation = np.asarray(rest_rotation, dtype=float)
    points = _points(vertices, "vertices")
    horizontal = [axis for axis in range(3) if abs(rotation[2, axis]) < HORIZONTAL_TOL]
    if len(horizontal) != 2:
        raise ValueError(f"rest rotation must keep exactly two local axes horizontal, got {horizontal}")
    extent = points.max(axis=0) - points.min(axis=0)
    first, second = sorted(horizontal, key=lambda axis: -extent[axis])
    return first, second


def extremity_offsets(vertices, axes: tuple[int, int]) -> np.ndarray:
    """Mesh extremities ``(+a0, -a0, +a1, -a1)`` on the given local axes, shape (4, 3)."""
    points = _points(vertices, "vertices")
    if len(set(axes)) != 2 or not all(axis in (0, 1, 2) for axis in axes):
        raise ValueError(f"axes must be two distinct local axes, got {axes}")
    offsets = np.zeros((4, 3))
    for i, axis in enumerate(axes):
        offsets[2 * i, axis] = points[:, axis].max()
        offsets[2 * i + 1, axis] = points[:, axis].min()
    return offsets


def surface_grid(x_range, y_range, z: float, inset: float, nx: int, ny: int) -> np.ndarray:
    """``nx x ny`` grid on a horizontal rectangle, ``inset`` from its edges, x-major order."""
    (x0, x1), (y0, y1) = x_range, y_range
    if not (x1 - x0 > 2 * inset and y1 - y0 > 2 * inset and nx >= 1 and ny >= 1):
        raise ValueError("rectangle is too small for the inset or the grid is empty")
    xs = np.linspace(x0 + inset, x1 - inset, nx)
    ys = np.linspace(y0 + inset, y1 - inset, ny)
    return np.array([[x, y, z] for x in xs for y in ys], dtype=float)


def assign_labels(object_names: Sequence[str], is_static: Sequence[bool], u: Sequence[float]) -> np.ndarray:
    """1-based labels grouped by object: movable objects by mean image ``u`` (left first), then static ones.

    Within an object the labels follow the input order, so they run in the object's keypoint order.
    """
    names = list(object_names)
    static = [bool(s) for s in is_static]
    columns = np.asarray(u, dtype=float)
    if not (len(names) == len(static) == len(columns)):
        raise ValueError("object_names, is_static and u must have the same length")
    if not np.all(np.isfinite(columns)):
        raise ValueError("u contains non-finite values")
    kind: dict[str, bool] = {}
    for name, flag in zip(names, static):
        if kind.setdefault(name, flag) != flag:
            raise ValueError(f"object {name!r} mixes static and movable keypoints")
    members = {name: [i for i, n in enumerate(names) if n == name] for name in kind}
    movable = sorted((n for n in kind if not kind[n]), key=lambda n: float(columns[members[n]].mean()))
    labels = np.zeros(len(names), dtype=int)
    next_label = 1
    for name in movable + [n for n in kind if kind[n]]:
        for i in members[name]:
            labels[i] = next_label
            next_label += 1
    return labels


def overlap_keep_mask(uv, labels, is_static, min_px: float) -> np.ndarray:
    """Greedy paper rule: movable before static, lower label first; a point is kept only if it is
    at least ``min_px`` from every point already kept."""
    points = np.asarray(uv, dtype=float)
    label_arr = np.asarray(labels, dtype=int)
    static = [bool(s) for s in is_static]
    if points.ndim != 2 or points.shape[1] != 2 or not (len(points) == len(label_arr) == len(static)):
        raise ValueError("uv must be (N, 2) and match labels and is_static")
    if not np.all(np.isfinite(points)):
        raise ValueError("uv contains non-finite values")
    keep = np.zeros(len(points), dtype=bool)
    kept: list[int] = []
    for i in sorted(range(len(points)), key=lambda j: (static[j], int(label_arr[j]))):
        if all(np.hypot(*(points[i] - points[j])) >= min_px for j in kept):
            keep[i] = True
            kept.append(i)
    return keep


def _points(value, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError(f"{name} must be (N, 3) with N >= 2")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} contains non-finite values")
    return points
