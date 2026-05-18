from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "demo_pose_reference.py"
SPEC = importlib.util.spec_from_file_location("demo_pose_reference", MODULE_PATH)
demo_pose_reference = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = demo_pose_reference
SPEC.loader.exec_module(demo_pose_reference)

DemoPoseReferenceBank = demo_pose_reference.DemoPoseReferenceBank

PRESET_PATH = Path(__file__).resolve().parents[1] / "pour_right_preset.py"
PRESET_SPEC = importlib.util.spec_from_file_location("pour_right_preset", PRESET_PATH)
pour_right_preset = importlib.util.module_from_spec(PRESET_SPEC)
assert PRESET_SPEC.loader is not None
sys.modules[PRESET_SPEC.name] = pour_right_preset
PRESET_SPEC.loader.exec_module(pour_right_preset)


def test_demo_pose_reference_loads_a11_to_a20_contract() -> None:
    paths = [f"/home/user/rl_ws/datasets/pour_v1_a{i}.hdf5" for i in range(11, 21)]

    bank = DemoPoseReferenceBank.from_hdf5_paths(paths, phase="pour", device="cpu")

    assert bank.num_frames > 0
    assert bank.arm_joint_pos.shape[1] == 7
    assert bank.hand_joint_pos.shape[1] == 20
    assert bank.hand_reference_joint_pos.shape[1] == 20
    assert bank.palm_pose.shape[1] == 7
    assert bank.target_palm_pose.shape[1] == 7
    for tensor in (
        bank.arm_joint_pos,
        bank.hand_joint_pos,
        bank.hand_reference_joint_pos,
        bank.palm_pose,
        bank.target_palm_pose,
        bank.arm_joint_mean,
        bank.arm_joint_std,
        bank.palm_pos_mean,
        bank.palm_pos_std,
        bank.palm_quat_mean,
        bank.thumb_joint_mean,
        bank.thumb_joint_std,
        bank.arm_vel_l2_p95,
        bank.arm_jerk_l2_p95,
    ):
        assert torch.isfinite(tensor).all()


def test_demo_pose_reference_all_phase_expands_beyond_pour_segment() -> None:
    paths = [f"/home/user/rl_ws/datasets/pour_v1_a{i}.hdf5" for i in range(11, 21)]

    pour_bank = DemoPoseReferenceBank.from_hdf5_paths(paths, phase="pour", device="cpu")
    all_bank = DemoPoseReferenceBank.from_hdf5_paths(paths, phase="all", device="cpu")

    assert all_bank.num_frames > pour_bank.num_frames
    assert all_bank.arm_joint_pos.shape[1] == 7
    assert all_bank.palm_pose.shape[1] == 7


def test_demo_thumb_cost_is_compatible_with_v5_grip_presets() -> None:
    paths = [f"/home/user/rl_ws/datasets/pour_v1_a{i}.hdf5" for i in range(11, 21)]
    bank = DemoPoseReferenceBank.from_hdf5_paths(paths, phase="pour", device="cpu")
    hand_open = torch.tensor(pour_right_preset.HAND_APPROACH_POSE, dtype=torch.float32)
    hand_grasp = torch.tensor(pour_right_preset.HAND_GRASP_POSE, dtype=torch.float32)

    def weighted_thumb_cost(hand_pose: torch.Tensor) -> torch.Tensor:
        unweighted = torch.norm((hand_pose[:4] - bank.thumb_joint_mean) / bank.thumb_joint_std) / 2.0
        return 0.5 * unweighted

    assert weighted_thumb_cost(hand_open) < 1.0
    assert weighted_thumb_cost(hand_grasp) < 1.0


def test_demo_pose_reference_missing_key_error_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "bad_demo.hdf5"
    with h5py.File(path, "w") as h5:
        demo = h5.create_group("data/demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("right_arm_joint_pos", data=np.zeros((2, 7), dtype=np.float32))

    with pytest.raises(KeyError, match="missing required key"):
        DemoPoseReferenceBank.from_hdf5_paths([path], phase="pour", device="cpu")


def test_demo_pose_reference_resolves_dataset_dir_env(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_dir = Path("/home/user/rl_ws/datasets")
    monkeypatch.setenv("POUR_V1_DATASET_DIR", str(dataset_dir))

    bank = DemoPoseReferenceBank.from_hdf5_paths(
        ["/missing/datasets/pour_v1_a11.hdf5"],
        phase="pour",
        device="cpu",
    )

    assert bank.num_frames > 0
    assert bank.source_paths == (str(dataset_dir / "pour_v1_a11.hdf5"),)
