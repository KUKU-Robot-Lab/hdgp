"""The committed shoe meta file matches the measured IKER mesh geometry (no Isaac)."""

import numpy as np
import pytest

from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.tasks.iker_shoe import layout


@pytest.fixture(scope="module")
def meta() -> dict:
    return run_files.load_shoe_meta(layout.SHOE_META_PATH)


def test_shoes_map_to_the_iker_objects(meta):
    assert meta["objects"][layout.MOVING_SHOE]["source"] == "object_0"
    assert meta["objects"][layout.OTHER_SHOE]["source"] == "object_1"
    assert {"object_0_mesh.obj", "object_1_mesh.obj", "object_pose.json", "target.json"} <= set(meta["source"]["sha256"])


@pytest.mark.parametrize("name, half_length, half_width, rest_height", [
    (layout.MOVING_SHOE, 0.1248, 0.0481, 0.05214),
    (layout.OTHER_SHOE, 0.1261, 0.0485, 0.05502),
])
def test_keypoints_are_length_and_width_extremities_at_rest(meta, name, half_length, half_width, rest_height):
    obj = meta["objects"][name]
    assert obj["horizontal_axes"] == [2, 0]
    expected = [[0, 0, half_length], [0, 0, -half_length], [half_width, 0, 0], [-half_width, 0, 0]]
    assert np.allclose(obj["keypoint_offsets"], expected, atol=5e-4)
    assert obj["rest_height"] == pytest.approx(rest_height, abs=5e-4)
    assert np.allclose(obj["rest_quat_wxyz"], [0.5, -0.5, 0.5, -0.5])
    assert len(obj["hull_local"]) > 100


def test_legacy_relation_is_recorded(meta):
    assert meta["legacy"]["z_lift"] == 1.0
    assert len(meta["legacy"]["target_position"]) == 4
    assert set(meta["legacy"]["object_pose"]) == {"object_0", "object_1"}
