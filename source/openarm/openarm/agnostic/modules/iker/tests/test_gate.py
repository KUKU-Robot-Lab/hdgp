"""gate — G1-G7 on the nominal shoe scene (no Isaac)."""

import numpy as np
import pytest

from openarm.agnostic.modules.iker import gate
from openarm.agnostic.modules.iker.interaction import Interaction

REST = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
MOVE_OFFSETS = np.array([[0.0, 0.0, 0.1248], [0.0, 0.0, -0.1248], [0.0481, 0.0, 0.0], [-0.0481, 0.0, 0.0]])
OTHER_OFFSETS = np.array([[0.0, 0.0, 0.1261], [0.0, 0.0, -0.1261], [0.0485, 0.0, 0.0], [-0.0485, 0.0, 0.0]])
MOVE_HULL = np.array([[sx * 0.0481, sy * 0.05214, sz * 0.1248] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
OTHER_HULL = np.array([[sx * 0.0485, sy * 0.05502, sz * 0.1261] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
TABLE_Z, RACK_Z = 0.205, 0.325
MOVE_START = np.array([0.27, 0.14, TABLE_Z + 0.05214])
OTHER_START = np.array([0.27, -0.15, RACK_Z + 0.05502])
SUPPORTS = (gate.Support("rack", RACK_Z, (0.11, 0.43), (-0.33, 0.03)), gate.Support("table", TABLE_Z))
CFG = gate.GateConfig(supports=SUPPORTS, palm_box_min=(0.20, -0.55, 0.20), palm_box_max=(0.55, 0.22, 0.70))
MOVABLES = (
    gate.MovableObject("shoe_move", (1, 2, 3, 4), MOVE_OFFSETS, MOVE_HULL, MOVE_START + MOVE_OFFSETS @ REST.T, TABLE_Z),
    gate.MovableObject("shoe_other", (5, 6, 7, 8), OTHER_OFFSETS, OTHER_HULL, OTHER_START + OTHER_OFFSETS @ REST.T, RACK_Z),
)
STATIC_IDS = tuple(range(9, 21))
BESIDE_OTHER = [0.2573, -0.021, RACK_Z + 0.05214]
# The public IKER target relation carried beside the other shoe without refitting (spacing 0.150 / 0.100 m).
PUBLIC_SPACING_TARGET = np.array(
    [[0.3323, -0.021, 0.4065], [0.1823, -0.021, 0.4065], [0.2573, -0.071, 0.4065], [0.2573, 0.029, 0.4065]]
)


def _rigid(center) -> np.ndarray:
    return np.asarray(center, dtype=float) + MOVE_OFFSETS @ REST.T


def _interaction(targets, ids=(1, 2, 3, 4), grasp=True) -> Interaction:
    coordinates = {k: tuple(p) for k, p in zip((1, 2, 3, 4), np.asarray(targets, dtype=float).tolist())}
    return Interaction("shoe", tuple(ids), grasp, coordinates, False)


def _codes(report: gate.GateReport) -> list[str]:
    return [failure[:2] for failure in report.failures]


def test_kabsch_recovers_a_rigid_motion():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    r, t = gate.kabsch(MOVE_OFFSETS, MOVE_OFFSETS @ rotation.T + [0.1, 0.2, 0.3])
    assert np.allclose(r, rotation) and np.allclose(t, [0.1, 0.2, 0.3])


def test_rigid_target_resting_beside_the_other_shoe_passes():
    report = gate.evaluate_gate(_interaction(_rigid(BESIDE_OTHER)), MOVABLES, STATIC_IDS, CFG)
    assert report.passed, report.failures
    assert report.object_name == "shoe_move" and report.metrics["support"] == "rack"
    assert report.metrics["mean_distance_m"] == pytest.approx(0.0, abs=1e-9)
    assert report.metrics["hull_min_z"] == pytest.approx(RACK_Z, abs=1e-9)
    assert np.allclose(report.target_keypoints, _rigid(BESIDE_OTHER))


def test_done_fails_g1():
    report = gate.evaluate_gate(Interaction("", (), False, {}, True), MOVABLES, STATIC_IDS, CFG)
    assert _codes(report) == ["G1"]


@pytest.mark.parametrize("ids, fragment", [((1, 99), "unknown"), ((9, 10), "static"), ((1, 5), "span")])
def test_keypoint_ids_must_belong_to_one_movable_object(ids, fragment):
    report = gate.evaluate_gate(_interaction(_rigid(BESIDE_OTHER), ids=ids), MOVABLES, STATIC_IDS, CFG)
    assert _codes(report) == ["G2"] and fragment in report.failures[0]


def test_missing_or_non_finite_targets_fail_g3():
    targets = _rigid(BESIDE_OTHER)
    targets[2, 0] = np.nan
    assert _codes(gate.evaluate_gate(_interaction(targets), MOVABLES, STATIC_IDS, CFG)) == ["G3"]
    partial = Interaction("shoe", (1, 2, 3, 4), True, {1: (0.3, 0.0, 0.3)}, False)
    report = gate.evaluate_gate(partial, MOVABLES, STATIC_IDS, CFG)
    assert _codes(report) == ["G3"] and "[2, 3, 4]" in report.failures[0]


def test_public_iker_target_spacing_fails_only_the_normalized_bound_of_g4():
    report = gate.evaluate_gate(_interaction(PUBLIC_SPACING_TARGET), MOVABLES, STATIC_IDS, CFG)
    assert _codes(report) == ["G4"] and "normalized" in report.failures[0]
    assert report.metrics["mean_distance_m"] == pytest.approx(0.0258, abs=1e-3)
    assert report.metrics["normalized_error"] == pytest.approx(0.1147, abs=2e-3)


def test_target_sunk_into_the_rack_fails_g5():
    sunk = [BESIDE_OTHER[0], BESIDE_OTHER[1], BESIDE_OTHER[2] - 0.03]
    assert _codes(gate.evaluate_gate(_interaction(_rigid(sunk)), MOVABLES, STATIC_IDS, CFG)) == ["G5"]


def test_target_outside_the_palm_box_fails_g6():
    far = [0.60, 0.10, TABLE_Z + 0.05214]
    assert _codes(gate.evaluate_gate(_interaction(_rigid(far)), MOVABLES, STATIC_IDS, CFG)) == ["G6"]


def test_push_cannot_lift_the_shoe_onto_the_rack_g7():
    report = gate.evaluate_gate(_interaction(_rigid(BESIDE_OTHER), grasp=False), MOVABLES, STATIC_IDS, CFG)
    assert _codes(report) == ["G7"]


def test_push_along_the_table_passes():
    report = gate.evaluate_gate(_interaction(_rigid([0.27, 0.05, TABLE_Z + 0.05214]), grasp=False), MOVABLES, STATIC_IDS, CFG)
    assert report.passed, report.failures


RELATIVE_RESPONSE = '''```python
import numpy as np

def get_interaction_data(keypoint_coordinates):
    """Place the shoe to the left of the other shoe, on the rack."""
    object_to_interact = "right shoe"
    keypoint_indices_to_interact = [1, 2, 3, 4]
    grasp_mode = True
    for moving, reference in (("1", "5"), ("2", "6"), ("3", "7"), ("4", "8")):
        keypoint_coordinates[moving] = keypoint_coordinates[reference] + np.array([0.0, 0.129, -0.0029])
    return object_to_interact, keypoint_indices_to_interact, grasp_mode, keypoint_coordinates
```'''


def _all_keypoints() -> dict[int, tuple[float, float, float]]:
    points = list(MOVABLES[0].init_keypoints) + list(MOVABLES[1].init_keypoints)
    points += [(x, y, RACK_Z) for x in (0.16, 0.27, 0.38) for y in (-0.28, -0.1933, -0.1067, -0.02)]
    return {label: tuple(float(c) for c in p) for label, p in enumerate(points, start=1)}


def test_gate_response_runs_and_gates_a_relative_response():
    interaction, report = gate.gate_response(RELATIVE_RESPONSE, _all_keypoints(), MOVABLES, STATIC_IDS, CFG)
    assert interaction is not None and interaction.object_name == "right shoe"
    assert report.passed, report.failures


def test_gate_response_turns_unusable_text_into_g1():
    interaction, report = gate.gate_response("I would move the shoe.", _all_keypoints(), MOVABLES, STATIC_IDS, CFG)
    assert interaction is None and _codes(report) == ["G1"] and "code block" in report.failures[0]
