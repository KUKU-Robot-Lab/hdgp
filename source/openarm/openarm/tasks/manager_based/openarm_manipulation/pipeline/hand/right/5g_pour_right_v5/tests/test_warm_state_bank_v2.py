from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


RIGHT_DIR = Path(__file__).resolve().parents[2]


def _load_bank_module(task_dir: str):
    module_path = RIGHT_DIR / task_dir / "warm_state_bank.py"
    spec = importlib.util.spec_from_file_location(f"warm_state_bank_{task_dir}", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_base_datasets(group: h5py.Group, n: int) -> None:
    group.create_dataset("arm_joint_pos", data=np.zeros((n, 7), dtype=np.float32))
    group.create_dataset("hand_joint_pos", data=np.zeros((n, 20), dtype=np.float32))
    group.create_dataset("palm_pose_quat_xyzw", data=np.zeros((n, 7), dtype=np.float32))
    group.create_dataset("palm_pose_euler_zyx", data=np.zeros((n, 6), dtype=np.float32))
    group.create_dataset("cup_pos_local", data=np.zeros((n, 3), dtype=np.float32))
    group.create_dataset("cup_quat_wxyz", data=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32))
    group.create_dataset("num_contacts", data=np.ones((n,), dtype=np.float32) * 5.0)


def _write_v1(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        _write_base_datasets(h5.create_group("warm_states"), 2)
        h5.attrs["meta/object_spawn_z"] = 0.297


def _write_v2(path: Path) -> None:
    n = 2
    num_beads = 30
    with h5py.File(path, "w") as h5:
        group = h5.create_group("warm_states")
        _write_base_datasets(group, n)
        group.create_dataset("bead_state_local", data=np.zeros((n, num_beads, 13), dtype=np.float32))
        group.create_dataset("bead_count_initial", data=np.array([30, 20], dtype=np.int64))
        group.create_dataset("bead_count_current", data=np.array([30, 20], dtype=np.int64))
        group.create_dataset("bead_count_target", data=np.array([30, 20], dtype=np.int64))
        group.create_dataset("dynamic_bead_spawned", data=np.array([False, True], dtype=np.bool_))
        group.create_dataset("cup_friction_static", data=np.array([0.3, 0.4], dtype=np.float32))
        group.create_dataset("object_goal_local", data=np.zeros((n, 3), dtype=np.float32))
        h5.attrs["meta/schema_version"] = 2
        h5.attrs["meta/object_spawn_z"] = 0.297
        h5.attrs["meta/num_beads"] = num_beads
        h5.attrs["meta/bead_single_mass"] = 0.01
        h5.attrs["meta/source_task"] = "5g_grasp_right-v11"


@pytest.mark.parametrize("task_dir", ["5g_pour_right_v3", "5g_pour_right_v5"])
def test_warm_state_bank_loads_legacy_v1(task_dir: str, tmp_path: Path) -> None:
    module = _load_bank_module(task_dir)
    path = tmp_path / "warm_v1.hdf5"
    _write_v1(path)

    bank = module.PourWarmStateBank.from_hdf5_paths(
        [path],
        device="cpu",
        expected_object_spawn_z=0.297,
    )

    assert bank.schema_version == 1
    assert not bank.has_bead_state
    assert bank.num_states == 2


@pytest.mark.parametrize("task_dir", ["5g_pour_right_v3", "5g_pour_right_v5"])
def test_warm_state_bank_loads_v11_bead_schema(task_dir: str, tmp_path: Path) -> None:
    module = _load_bank_module(task_dir)
    path = tmp_path / "warm_v2.hdf5"
    _write_v2(path)

    bank = module.PourWarmStateBank.from_hdf5_paths(
        [path],
        device="cpu",
        expected_object_spawn_z=0.297,
        expected_num_beads=30,
        expected_bead_single_mass=0.01,
    )

    assert bank.schema_version == 2
    assert bank.has_bead_state
    assert bank.bead_state_local.shape == (2, 30, 13)
    assert bank.bead_count_current.tolist() == [30, 20]
    assert bank.source_meta["source_task"] == "5g_grasp_right-v11"


@pytest.mark.parametrize("task_dir", ["5g_pour_right_v3", "5g_pour_right_v5"])
def test_warm_state_bank_rejects_v11_bead_meta_mismatch(task_dir: str, tmp_path: Path) -> None:
    module = _load_bank_module(task_dir)
    path = tmp_path / "warm_v2.hdf5"
    _write_v2(path)

    with pytest.raises(ValueError, match="num_beads mismatch"):
        module.PourWarmStateBank.from_hdf5_paths(
            [path],
            device="cpu",
            expected_object_spawn_z=0.297,
            expected_num_beads=20,
            expected_bead_single_mass=0.01,
        )
