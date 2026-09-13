"""rotations — quaternion/matrix/Euler helpers (no Isaac)."""

import numpy as np
import pytest

from openarm.agnostic.modules.iker import rotations as rot

SHOE_REST = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def test_quat_matrix_round_trip_keeps_w_non_negative():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        back = rot.matrix_to_quat_wxyz(rot.quat_wxyz_to_matrix(q))
        assert back[0] >= 0.0
        assert np.allclose(back, q if q[0] >= 0.0 else -q, atol=1e-9)


def test_shoe_rest_rotation_has_the_expected_quaternion():
    assert np.allclose(rot.matrix_to_quat_wxyz(SHOE_REST), [0.5, -0.5, 0.5, -0.5])


def test_rpy_matches_scipy_extrinsic_xyz():
    rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    rpy = (-1.5636626780948757, 0.028418572882054294, -1.6710602751696029)
    assert np.allclose(rot.rpy_xyz_to_matrix(rpy), rotation.from_euler("xyz", rpy).as_matrix(), atol=1e-12)


def test_non_finite_and_improper_inputs_raise():
    with pytest.raises(ValueError, match="non-finite"):
        rot.quat_wxyz_to_matrix([np.nan, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="proper rotation"):
        rot.matrix_to_quat_wxyz(np.diag([1.0, 1.0, -1.0]))
