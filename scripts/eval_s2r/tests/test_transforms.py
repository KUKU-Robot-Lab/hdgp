import numpy as np
import pytest

from scripts.eval_s2r.transforms import (
    compose_local_pose, k_from_pinhole, pinhole_from_k, validate_T,
)


def _rt(rot, trans):
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


class TestCompose:
    def test_identity_camera(self):
        # 카메라가 local 원점·무회전이면 T_cam_obj 평행이동이 그대로 local 위치
        p = compose_local_pose(np.eye(4), _rt(np.eye(3), [0.1, 0.2, 0.3]))
        assert np.allclose(p, [0.1, 0.2, 0.3])

    def test_rotated_camera_roundtrip(self):
        # 합성 검증: 알려진 local 물체 위치 → cam frame으로 보낸 뒤 복원
        Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        T_local_cam = _rt(Rz, [0.5, 0.0, 0.7])
        p_local_gt = np.array([0.3, -0.1, 0.05])
        T_cam_local = np.linalg.inv(T_local_cam)
        p_cam = (T_cam_local @ np.append(p_local_gt, 1.0))[:3]
        p = compose_local_pose(T_local_cam, _rt(np.eye(3), p_cam))
        assert np.allclose(p, p_local_gt, atol=1e-9)

    def test_invalid_T_raises(self):
        bad = np.eye(4); bad[0, 0] = 2.0  # 비직교 회전부
        with pytest.raises(ValueError):
            compose_local_pose(bad, np.eye(4))
        with pytest.raises(ValueError):
            compose_local_pose(np.eye(4), np.full((4, 4), np.nan))


class TestIntrinsics:
    def test_k_from_pinhole_center(self):
        K = k_from_pinhole(18.14756, 37.9586, 320, 180)
        assert K[0, 2] == pytest.approx(160.0) and K[1, 2] == pytest.approx(90.0)
        assert K[0, 0] == pytest.approx(18.14756 / 37.9586 * 320)
        assert K[1, 1] == pytest.approx(K[0, 0])  # 정방 픽셀

    def test_roundtrip(self):
        K = k_from_pinhole(20.0, 30.0, 640, 360)
        f, ap = pinhole_from_k(K, 640, 360)
        K2 = k_from_pinhole(f, ap, 640, 360)
        assert np.allclose(K, K2)
