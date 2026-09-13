"""Human baseline target (design spec §6): the public IKER target relation, refitted to our keypoints.

The public ``target.json`` places the moving shoe's four keypoints relative to the other shoe, but
its spacing (0.150 / 0.100 m) matches no rigid shoe pose, so copying it would make the baseline
unreachable. The relation is carried into our scene through the other shoe's frame, and the moving
shoe is placed where it best matches that relation while keeping the attitude it settled into at
the snapshot (a scanned sole is not flat; the settled attitude is the one it will rest in again):
only its heading and position change, and it is set down on the support under it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .gate import Support, kabsch
from .rotations import finite_array, rpy_xyz_to_matrix, yaw_matrix

# Our offsets are ordered (+long, -long, +width, -width). The legacy targets use the same long pair,
# but their width pair can be in either order; the pairing that keeps the object upright is used.
WIDTH_PAIRINGS = ((0, 1, 2, 3), (0, 1, 3, 2))


@dataclass(frozen=True)
class HumanTarget:
    keypoints: np.ndarray  # (4, 3) world, in the object's keypoint order
    rotation: np.ndarray  # (3, 3)
    center: np.ndarray  # (3,)
    relation_residual_m: np.ndarray  # (4,) free rigid-fit residual against the carried relation
    support_name: str
    width_pair_swapped: bool


def legacy_relation_targets(legacy_other_pos, legacy_other_rpy, legacy_targets, other_pos, other_rotation) -> np.ndarray:
    """Legacy target keypoints expressed in the legacy other shoe's frame, placed on our other shoe."""
    legacy_rotation = rpy_xyz_to_matrix(legacy_other_rpy)
    legacy = finite_array("legacy_targets", legacy_targets, (4, 3))
    in_other_frame = (legacy - finite_array("legacy_other_pos", legacy_other_pos, (3,))) @ legacy_rotation
    rotation = finite_array("other_rotation", other_rotation, (3, 3))
    return finite_array("other_pos", other_pos, (3,)) + in_other_frame @ rotation.T


def heading(rotation, long_axis: int) -> float:
    """World yaw (rad) of the object's long local axis."""
    direction = finite_array("rotation", rotation, (3, 3))[:, long_axis]
    if not math.hypot(direction[0], direction[1]) > 1e-6:
        raise ValueError("the long axis is vertical; heading is undefined")
    return math.atan2(direction[1], direction[0])


def human_target(
    relation_targets, local_offsets, hull_local, start_rotation, long_axis: int, supports: Sequence[Support]
) -> HumanTarget:
    relation = finite_array("relation_targets", relation_targets, (4, 3))
    offsets = finite_array("local_offsets", local_offsets, (4, 3))
    hull = np.asarray(hull_local, dtype=float)
    start = finite_array("start_rotation", start_rotation, (3, 3))
    vertical_axis = int(np.argmax(np.abs(start[2])))
    for pairing in WIDTH_PAIRINGS:
        ordered = relation[list(pairing)]
        fit_rotation, fit_center = kabsch(offsets, ordered)
        if np.sign(fit_rotation[2, vertical_axis]) == np.sign(start[2, vertical_axis]):
            break
    else:
        raise ValueError("no width pairing keeps the object upright")
    residual = np.linalg.norm(offsets @ fit_rotation.T + fit_center - ordered, axis=1)
    rotation = yaw_matrix(heading(fit_rotation, long_axis) - heading(start, long_axis)) @ start
    center = ordered.mean(axis=0) - (offsets @ rotation.T).mean(axis=0)  # least squares for a fixed rotation
    support = next((s for s in supports if s.contains_xy(center[0], center[1])), None)
    if support is None:
        raise ValueError(f"target centre {np.round(center, 4).tolist()} is above no support")
    center[2] += support.top_z - float((hull @ rotation.T + center)[:, 2].min())
    return HumanTarget(
        keypoints=offsets @ rotation.T + center,
        rotation=rotation,
        center=center,
        relation_residual_m=residual,
        support_name=support.name,
        width_pair_swapped=pairing != WIDTH_PAIRINGS[0],
    )
