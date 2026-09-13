"""run_files — keypoints.json / shoe_meta.json / interaction documents (no Isaac)."""

import copy
import json

import numpy as np
import pytest

from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.modules.iker.gate import GateReport, Support
from openarm.agnostic.modules.iker.interaction import Interaction
from openarm.agnostic.modules.iker.projection import CameraPose
from openarm.agnostic.modules.iker.scene_image import PointGroup, annotate_snapshot

REST = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
OFFSETS = [[0.0, 0.0, 0.1248], [0.0, 0.0, -0.1248], [0.0481, 0.0, 0.0], [-0.0481, 0.0, 0.0]]
HULL = [[sx * 0.0481, sy * 0.05214, sz * 0.1248] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
SUPPORTS = (Support("rack", 0.325, (0.11, 0.43), (-0.33, 0.03)), Support("table", 0.205))
MOVE_POS, OTHER_POS = (0.27, 0.14, 0.257), (0.27, -0.15, 0.377)
CAMERA = CameraPose(position=np.array([0.27, 0.0, 1.2]), rotation=np.diag([1.0, -1.0, -1.0]))
K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
META_OBJECT = {
    "source": "object_0", "rest_rotation": REST.tolist(), "rest_quat_wxyz": [0.5, -0.5, 0.5, -0.5],
    "horizontal_axes": [2, 0], "keypoint_offsets": OFFSETS, "rest_height": 0.05214, "hull_local": HULL,
}
META = {"schema": 1, "legacy": {}, "objects": {"shoe_move": META_OBJECT, "shoe_other": META_OBJECT}}


def _doc() -> dict:
    groups = [
        PointGroup("shoe_move", False, np.add(MOVE_POS, np.asarray(OFFSETS) @ REST.T)),
        PointGroup("shoe_other", False, np.add(OTHER_POS, np.asarray(OFFSETS) @ REST.T)),
        PointGroup("rack", True, np.array([[0.16, -0.28, 0.325], [0.38, -0.02, 0.325]])),
    ]
    snap = annotate_snapshot(
        np.zeros((480, 640, 3), np.uint8), np.full((480, 640), 5.0), CAMERA, K, groups, [0.27, 0.0, 0.2], 15.0
    )
    return run_files.keypoints_document(
        scene_config={"index": 0},
        camera=CAMERA,
        intrinsic=K,
        snapshot=snap,
        image_size=(640, 480),
        object_poses={"shoe_move": (MOVE_POS, (0.5, -0.5, 0.5, -0.5)), "shoe_other": (OTHER_POS, (0.5, -0.5, 0.5, -0.5))},
        head={"pan_cmd_deg": -20.0},
        checks={"settle_disp_m": 0.001},
    )


def test_keypoints_document_round_trips_through_json(tmp_path):
    doc = _doc()
    path = run_files.write_json(tmp_path / "run" / "keypoints.json", doc)
    assert run_files.read_json(path) == json.loads(json.dumps(doc))
    assert not (tmp_path / "run" / "keypoints.json.tmp").exists()


def test_write_rejects_nan_and_read_checks_the_schema(tmp_path):
    with pytest.raises(ValueError):
        run_files.write_json(tmp_path / "bad.json", {"schema": 1, "x": float("nan")})
    (tmp_path / "old.json").write_text('{"schema": 0}')
    with pytest.raises(ValueError, match="schema"):
        run_files.read_json(tmp_path / "old.json")
    with pytest.raises(FileNotFoundError):
        run_files.read_json(tmp_path / "missing.json")


def test_movable_objects_follow_labels_and_start_supports():
    doc = _doc()
    move, other = run_files.movable_objects(doc, META, SUPPORTS)
    assert move.name == "shoe_move" and move.keypoint_ids == (1, 2, 3, 4) and move.start_support_z == 0.205
    assert other.keypoint_ids == (5, 6, 7, 8) and other.start_support_z == 0.325
    assert np.allclose(move.init_keypoints, np.add(MOVE_POS, np.asarray(OFFSETS) @ REST.T))
    assert set(run_files.vlm_keypoints(doc)) == {r["label"] for r in doc["keypoints"] if r["kept"]}
    assert run_files.static_keypoint_ids(doc) == (9, 10)


def test_movable_objects_refuse_a_removed_keypoint():
    doc = copy.deepcopy(_doc())
    next(r for r in doc["keypoints"] if r["label"] == 2)["kept"] = False
    with pytest.raises(ValueError, match="removed"):
        run_files.movable_objects(doc, META, SUPPORTS)


def test_load_shoe_meta_names_the_missing_field(tmp_path):
    broken = copy.deepcopy(META)
    del broken["objects"]["shoe_move"]["hull_local"]
    run_files.write_json(tmp_path / "meta.json", broken)
    with pytest.raises(ValueError, match="hull_local"):
        run_files.load_shoe_meta(tmp_path / "meta.json")


def test_interaction_document_carries_targets_and_gate():
    interaction = Interaction("shoe", (1, 2, 3, 4), True, {}, False)
    report = GateReport(True, (), {"mean_distance_m": 0.0}, "shoe_move", ((0.1, 0.2, 0.3),) * 4)
    doc = run_files.interaction_document("human", interaction, report, extra={"note": "x"})
    assert doc["object"] == "shoe_move" and doc["target_keypoints"][0] == [0.1, 0.2, 0.3]
    assert doc["gate"]["passed"] is True and doc["extra"] == {"note": "x"}
    with pytest.raises(ValueError, match="source"):
        run_files.interaction_document("gpt", interaction, report)
