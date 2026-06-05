from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "warm_state_bank.py"
SPEC = importlib.util.spec_from_file_location("warm_state_bank_v5", MODULE_PATH)
warm_state_bank = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = warm_state_bank
SPEC.loader.exec_module(warm_state_bank)

PourWarmStateBank = warm_state_bank.PourWarmStateBank


def _write_warm_state(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = 2
        h5.attrs["count"] = 1
        h5.attrs["meta/object_spawn_z"] = 0.297
        h5.attrs["meta/export_mode"] = "demo0_arm_palm_actual_grasp"
        grp = h5.create_group("warm_states")
        grp.create_dataset("arm_joint_pos", data=np.ones((1, 7), dtype=np.float32))
        grp.create_dataset("hand_joint_pos", data=np.ones((1, 20), dtype=np.float32))
        grp.create_dataset("palm_pose_quat_xyzw", data=np.zeros((1, 7), dtype=np.float32))
        grp.create_dataset("palm_pose_euler_zyx", data=np.zeros((1, 6), dtype=np.float32))
        grp.create_dataset("cup_pos_local", data=np.asarray([[0.24, -0.08, 0.34]], dtype=np.float32))
        grp.create_dataset("cup_quat_wxyz", data=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        grp.create_dataset("num_contacts", data=np.asarray([4], dtype=np.float32))
        grp.create_dataset("demo_file_idx", data=np.asarray([3], dtype=np.int64))
        grp.create_dataset("per_finger_contact", data=np.asarray([[1, 1, 1, 1, 0]], dtype=np.uint8))
        grp.create_dataset("stable_contact_steps", data=np.asarray([12], dtype=np.int64))


def test_warm_state_bank_loads_schema_v2_optional_quality_fields(tmp_path: Path) -> None:
    path = tmp_path / "warm.hdf5"
    _write_warm_state(path)

    bank = PourWarmStateBank.from_hdf5_paths(
        [path],
        device="cpu",
        expected_object_spawn_z=0.297,
    )

    assert bank.num_states == 1
    assert bank.source_meta["export_mode"] == "demo0_arm_palm_actual_grasp"
    assert bank.demo_file_idx is not None
    assert bank.demo_file_idx.tolist() == [3]
    assert bank.per_finger_contact is not None
    assert bank.per_finger_contact.tolist() == [[True, True, True, True, False]]
    assert bank.stable_contact_steps is not None
    assert bank.stable_contact_steps.tolist() == [12]
