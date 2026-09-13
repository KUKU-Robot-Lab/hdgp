"""baseline — public IKER relation carried to our scene and refitted (no Isaac)."""

import math

import numpy as np
import pytest

from openarm.agnostic.modules.iker import baseline
from openarm.agnostic.modules.iker.gate import Support
from openarm.agnostic.modules.iker.rotations import rpy_xyz_to_matrix, yaw_matrix

REST = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
LONG_AXIS = 2
MOVE_OFFSETS = np.array([[0.0, 0.0, 0.1248], [0.0, 0.0, -0.1248], [0.0481, 0.0, 0.0], [-0.0481, 0.0, 0.0]])
MOVE_HULL = np.array([[sx * 0.0481, sy * 0.05214, sz * 0.1248] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
SUPPORTS = (Support("rack", 0.325, (0.11, 0.43), (-0.33, 0.03)), Support("table", 0.205))
OTHER_POS = np.array([0.27, -0.15, 0.325 + 0.05502])
# IKER envs/shoe_place: object_pose.json object_1 (lifted +1.0 m as the legacy task spawns it) and target.json.
LEGACY_OTHER_POS = np.array([0.6329157289252264, -0.02441016504, 0.17804389875793722 + 1.0])
LEGACY_OTHER_RPY = (-1.5636626780948757, 0.028418572882054294, -1.6710602751696029)
LEGACY_TARGETS = np.array(
    [
        [0.7075356401032895, 0.09688803021587822, 1.2085786984839113],
        [0.5582958177471633, 0.11193229987877429, 1.207509099031963],
        [0.6279129482614393, 0.05468136477793894, 1.2066231614852063],
        [0.6379185095890135, 0.15413896531671356, 1.2094646360306682],
    ]
)
EXPECTED_TARGET = np.array(
    [[0.3821, -0.021, 0.3771], [0.1325, -0.021, 0.3771], [0.2573, -0.0691, 0.3771], [0.2573, 0.0271, 0.3771]]
)


def _relation() -> np.ndarray:
    return baseline.legacy_relation_targets(LEGACY_OTHER_POS, LEGACY_OTHER_RPY, LEGACY_TARGETS, OTHER_POS, REST)


def test_relation_puts_the_target_beside_the_other_shoe_and_keeps_the_public_spacing():
    relation = _relation()
    assert np.allclose(relation.mean(axis=0) - OTHER_POS, [-0.0127, 0.129, 0.0264], atol=5e-4)
    assert np.linalg.norm(relation[0] - relation[1]) == pytest.approx(0.150, abs=1e-3)
    assert np.linalg.norm(relation[2] - relation[3]) == pytest.approx(0.100, abs=1e-3)


def test_human_target_is_rigid_upright_and_resting_on_the_rack():
    target = baseline.human_target(_relation(), MOVE_OFFSETS, MOVE_HULL, REST, LONG_AXIS, SUPPORTS)
    assert np.allclose(target.keypoints, EXPECTED_TARGET, atol=5e-4)
    assert target.support_name == "rack" and not target.width_pair_swapped
    assert np.linalg.norm(target.keypoints[0] - target.keypoints[1]) == pytest.approx(2 * 0.1248)
    assert np.allclose(target.relation_residual_m, [0.0498, 0.0498, 0.0019, 0.0019], atol=5e-4)
    assert np.allclose(target.rotation, REST, atol=1e-6)


def test_mirrored_width_order_is_detected_and_gives_the_same_target():
    target = baseline.human_target(_relation()[[0, 1, 3, 2]], MOVE_OFFSETS, MOVE_HULL, REST, LONG_AXIS, SUPPORTS)
    assert target.width_pair_swapped
    assert np.allclose(target.keypoints, EXPECTED_TARGET, atol=5e-4)


def test_settled_attitude_is_kept_and_only_the_heading_changes():
    # The shoe settled on the table 8 deg off its nominal heading and rolled 4 deg on its uneven sole.
    start = yaw_matrix(math.radians(8.0)) @ REST @ rpy_xyz_to_matrix((0.0, 0.0, math.radians(4.0)))
    target = baseline.human_target(_relation(), MOVE_OFFSETS, MOVE_HULL, start, LONG_AXIS, SUPPORTS)
    assert baseline.heading(target.rotation, LONG_AXIS) == pytest.approx(baseline.heading(REST, LONG_AXIS), abs=1e-3)
    assert np.allclose(target.rotation[2], start[2], atol=1e-9)  # same tilt as at the start
    assert float((MOVE_HULL @ target.rotation.T + target.center)[:, 2].min()) == pytest.approx(0.325)
    assert np.allclose(target.keypoints[:, :2].mean(axis=0), EXPECTED_TARGET[:, :2].mean(axis=0), atol=5e-4)
