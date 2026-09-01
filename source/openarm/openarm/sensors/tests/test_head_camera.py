"""실기 head 카메라 이식 사양 테스트 (Isaac 불필요)."""

import json

import pytest

from openarm.sensors.head_camera import (
    DEFAULT_SPEC_JSON,
    NECK_LINK,
    load_spec,
    urdf_head_angles,
)

GOOD = {
    "link": "head_camera",
    "pos": [0.051162, 0.052617, 0.009042],
    "quat_wxyz": [-0.008293, -0.700528, 0.713562, 0.004465],
    "width": 640, "height": 480,
    "intrinsic_matrix": [606.604, 0.0, 320.02, 0.0, 605.652, 240.574, 0.0, 0.0, 1.0],
    "clipping_range": [0.01, 10.0],
}


def _write(tmp_path, data):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_shipped_calibration_exists_and_loads():
    """★배선의 전제 — 캘리브 파일이 실제로 거기 있어야 한다."""
    assert DEFAULT_SPEC_JSON.is_file(), f"캘리브 없음: {DEFAULT_SPEC_JSON}"
    spec = load_spec()
    assert spec.link == NECK_LINK
    assert (spec.width, spec.height) == (640, 480)


def test_camera_is_on_the_tilt_link_not_cam_view():
    """★`head_cam_view` 는 실제와 59.5 mm 어긋난다 — 거기 붙이면 안 된다."""
    assert load_spec().link == "head_camera" != "head_cam_view"


def test_load_spec_reads_all_fields(tmp_path):
    s = load_spec(_write(tmp_path, GOOD))
    assert s.pos == pytest.approx(GOOD["pos"])
    assert s.quat_wxyz == pytest.approx(GOOD["quat_wxyz"])
    assert s.intrinsic_matrix == pytest.approx(GOOD["intrinsic_matrix"])


def test_load_spec_rejects_missing_field(tmp_path):
    bad = {k: v for k, v in GOOD.items() if k != "intrinsic_matrix"}
    with pytest.raises(ValueError, match="intrinsic_matrix"):
        load_spec(_write(tmp_path, bad))


def test_pan_sign_is_inverted():
    """URDF pan 축이 (0,0,-1) 이라 인코더와 반대다."""
    assert urdf_head_angles(10.0, -20.0) == pytest.approx((-10.0, -20.0))


def test_tilt_sign_is_preserved():
    assert urdf_head_angles(0.0, -20.0)[1] == pytest.approx(-20.0)


def test_intrinsics_match_measured_realsense():
    """실측 K 를 그대로 넘겨야 한다 — 우리가 미리 변환하지 않는다."""
    K = load_spec().intrinsic_matrix
    assert K[0] == pytest.approx(606.604, abs=0.01)     # fx
    assert K[4] == pytest.approx(605.652, abs=0.01)     # fy
    assert K[2] == pytest.approx(320.02, abs=0.01)      # cx
