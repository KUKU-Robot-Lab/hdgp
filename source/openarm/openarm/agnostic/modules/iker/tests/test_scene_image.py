"""scene_image + annotate — labelled keypoint image from a synthetic camera (no Isaac)."""

import numpy as np
import pytest
from PIL import Image

from openarm.agnostic.modules.iker import annotate
from openarm.agnostic.modules.iker.projection import CameraPose
from openarm.agnostic.modules.iker.scene_image import PointGroup, annotate_snapshot

DOWN = CameraPose(position=np.array([0.0, 0.0, 1.0]), rotation=np.diag([1.0, -1.0, -1.0]))
K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
RGB = np.full((480, 640, 3), 40, dtype=np.uint8)
DEPTH = np.full((480, 640), 1.0)  # the plane z = 0 is 1 m below the camera


def _shoe(y: float) -> np.ndarray:
    return np.array([[0.12, y, 0.0], [-0.12, y, 0.0], [0.0, y - 0.05, 0.0], [0.0, y + 0.05, 0.0]])


def _groups(rack_points) -> list[PointGroup]:
    return [
        PointGroup("shoe_move", False, _shoe(0.15)),
        PointGroup("shoe_other", False, _shoe(-0.15)),
        PointGroup("rack", True, np.asarray(rack_points, dtype=float)),
    ]


def test_left_object_gets_the_first_labels_and_static_points_come_last():
    snap = annotate_snapshot(RGB, DEPTH, DOWN, K, _groups([[0.1, -0.35, 0.0], [-0.1, -0.35, 0.0]]), [0, 0, 0], 15.0)
    by_object: dict[str, list[int]] = {}
    for record in snap.records:
        by_object.setdefault(record.object_name, []).append(record.label)
    assert by_object == {"shoe_move": [1, 2, 3, 4], "shoe_other": [5, 6, 7, 8], "rack": [9, 10]}
    assert snap.derotation_deg == pytest.approx(90.0)


def test_static_point_on_a_shoe_keypoint_is_dropped_and_occlusion_is_recorded():
    depth = DEPTH.copy()
    depth[240 - 90, 320 + 72] = 0.5  # raw pixel of shoe_move's first keypoint (x 0.12, y 0.15)
    snap = annotate_snapshot(RGB, depth, DOWN, K, _groups([[0.12, -0.15, 0.0], [-0.1, -0.35, 0.0]]), [0, 0, 0], 15.0)
    assert [r.kept for r in snap.records if r.object_name == "rack"] == [False, True]
    first = next(r for r in snap.records if r.label == 1)
    assert first.visible is False and first.render_depth == pytest.approx(0.5)
    assert all(r.visible for r in snap.records if r.label != 1)


def test_overlapping_movable_keypoints_abort_the_snapshot():
    groups = [PointGroup("shoe_move", False, _shoe(0.0)), PointGroup("shoe_other", False, _shoe(0.0) + [0.0, 0.004, 0.0])]
    with pytest.raises(ValueError, match="removed movable"):
        annotate_snapshot(RGB, DEPTH, DOWN, K, groups, [0, 0, 0], 15.0)


def test_markers_and_legend_are_drawn_on_the_derotated_image():
    snap = annotate_snapshot(RGB, DEPTH, DOWN, K, _groups([[-0.1, -0.35, 0.0]]), [0, 0, 0], 15.0)
    assert snap.image.size == (640, 480)
    for label, color in ((1, annotate.MOVABLE_COLORS[0]), (5, annotate.MOVABLE_COLORS[1]), (9, annotate.STATIC_COLOR)):
        u, v = next(r for r in snap.records if r.label == label).uv
        assert snap.image.getpixel((round(u - 8), round(v))) == color
    assert snap.axes["+x"][1] < 0.0 and abs(snap.axes["+x"][0]) < 1e-6
    assert snap.axes["+y"][0] < 0.0 and abs(snap.axes["+y"][1]) < 1e-6
    assert snap.min_margin_px > 15.0 and snap.min_gap_px >= 15.0


def test_draw_markers_refuses_non_finite_coordinates():
    marker = annotate.Marker(1, (float("nan"), 3.0), (255, 0, 0), True)
    with pytest.raises(ValueError, match="non-finite"):
        annotate.draw_markers(Image.new("RGB", (64, 48)), [marker], {})
