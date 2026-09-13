"""projection — camera pose, pinhole projection, derotation (no Isaac)."""

import numpy as np
import pytest
from PIL import Image

from openarm.agnostic.modules.iker import projection
from openarm.agnostic.modules.iker.projection import CameraPose

# Looking straight down from 1 m: optical x = world +x, optical y = world -y, optical z = world -z.
DOWN = CameraPose(position=np.array([0.0, 0.0, 1.0]), rotation=np.diag([1.0, -1.0, -1.0]))
K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def test_camera_pose_composes_link_pose_and_optical_offset():
    link_quat = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]  # 90 deg about z
    pose = projection.camera_pose_from_link([1.0, 2.0, 3.0], link_quat, [0.1, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(pose.position, [1.0, 2.1, 3.0])
    assert np.allclose(pose.rotation[:, 0], [0.0, 1.0, 0.0])


def test_projection_reproduces_pinhole_pixels_and_depth():
    uv, depth = projection.project_points([[0.1, 0.0, 0.0], [0.0, 0.1, 0.5]], DOWN, K)
    assert np.allclose(uv, [[380.0, 240.0], [320.0, 120.0]])
    assert np.allclose(depth, [1.0, 0.5])


def test_points_behind_the_camera_raise():
    with pytest.raises(ValueError, match="behind"):
        projection.project_points([[0.0, 0.0, 2.0]], DOWN, K)


def test_non_finite_camera_raises_instead_of_projecting_nan():
    bad = CameraPose(position=np.array([0.0, np.nan, 1.0]), rotation=np.eye(3))
    with pytest.raises(ValueError, match="non-finite"):
        projection.project_points([[0.0, 0.0, 0.0]], bad, K)


def test_derotation_turns_world_x_up_and_world_y_left():
    theta = projection.derotation_deg([0.0, 0.0, 0.0], DOWN, K)
    assert theta == pytest.approx(90.0)
    uv, _ = projection.project_points([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], DOWN, K)
    rotated = projection.rotate_image_points(uv, theta, (640, 480))
    x_dir, y_dir = rotated[1] - rotated[0], rotated[2] - rotated[0]
    assert x_dir[1] < 0.0 and abs(x_dir[0]) < 1e-6
    assert y_dir[0] < 0.0 and abs(y_dir[1]) < 1e-6


def test_rotate_image_points_matches_pil_rotation():
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    pixels[95:106, 495:506] = 255
    theta = 30.0
    rotated = np.asarray(Image.fromarray(pixels).rotate(theta, resample=Image.Resampling.BICUBIC, expand=False))
    rows, cols = np.nonzero(rotated[:, :, 0] > 127)
    expected = projection.rotate_image_points([[500.5, 100.5]], theta, (640, 480))[0]
    assert abs(cols.mean() + 0.5 - expected[0]) < 1.0
    assert abs(rows.mean() + 0.5 - expected[1]) < 1.0


def test_image_margin_is_negative_outside_the_image():
    assert np.allclose(projection.image_margin([[10.0, 240.0], [-5.0, 100.0]], (640, 480)), [10.0, -5.0])
