"""keypoints — mesh extremities, rack grid, labels, overlap filter (no Isaac)."""

import numpy as np
import pytest

from openarm.agnostic.modules.iker import keypoints as kp
from openarm.agnostic.modules.iker.rotations import rpy_xyz_to_matrix

SHOE_REST = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
# A box with the shoe's extents: width x 0.096, height y 0.104, length z 0.250.
SHOE_BOX = np.array([[sx * 0.048, sy * 0.052, sz * 0.125] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def test_snap_rotation_removes_a_small_scan_tilt():
    tilted = SHOE_REST @ rpy_xyz_to_matrix((0.05, -0.03, 0.04))
    assert np.array_equal(kp.snap_rotation(tilted), SHOE_REST)


def test_snap_rotation_rejects_a_reflection():
    with pytest.raises(ValueError, match="axis-aligned"):
        kp.snap_rotation(np.diag([1.0, 1.0, -1.0]))


def test_horizontal_axes_are_length_then_width_not_height():
    # Sorting all three axes by size would return (2, 1): the height beats the width.
    assert kp.horizontal_axes(SHOE_REST, SHOE_BOX) == (2, 0)


def test_extremity_offsets_follow_axis_order_and_keep_asymmetry():
    offsets = kp.extremity_offsets(SHOE_BOX + [0.0, 0.0, 0.01], (2, 0))
    assert np.allclose(offsets, [[0, 0, 0.135], [0, 0, -0.115], [0.048, 0, 0], [-0.048, 0, 0]])


def test_surface_grid_is_inset_and_x_major():
    grid = kp.surface_grid((0.11, 0.43), (-0.33, 0.03), 0.325, 0.05, 3, 4)
    assert grid.shape == (12, 3)
    assert np.allclose(grid[0], [0.16, -0.28, 0.325])
    assert np.allclose(grid[3], [0.16, -0.02, 0.325])
    assert np.allclose(grid[11], [0.38, -0.02, 0.325])


def test_labels_number_movable_objects_left_to_right_then_static():
    names = ["shoe_move"] * 4 + ["shoe_other"] * 4 + ["rack"] * 2
    static = [False] * 8 + [True] * 2
    u = [300, 310, 320, 330, 100, 110, 120, 130, 50, 60]
    assert kp.assign_labels(names, static, u).tolist() == [5, 6, 7, 8, 1, 2, 3, 4, 9, 10]


def test_labels_reject_an_object_with_mixed_kinds():
    with pytest.raises(ValueError, match="mixes"):
        kp.assign_labels(["a", "a"], [False, True], [0.0, 1.0])


def test_overlap_filter_drops_static_first_then_the_higher_label():
    uv = np.array([[100.0, 100.0], [105.0, 100.0], [300.0, 300.0], [306.0, 300.0]])
    keep = kp.overlap_keep_mask(uv, [10, 3, 1, 2], [True, False, False, False], 15.0)
    assert keep.tolist() == [False, True, True, False]
