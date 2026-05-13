from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "demo_grasp_reset.py"
SPEC = importlib.util.spec_from_file_location("demo_grasp_reset", MODULE_PATH)
demo_grasp_reset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = demo_grasp_reset
SPEC.loader.exec_module(demo_grasp_reset)

DemoGraspResetBank = demo_grasp_reset.DemoGraspResetBank
compute_demo_cup_spawn_local = demo_grasp_reset.compute_demo_cup_spawn_local


def _write_demo(path: Path, *, pour_start_frame: int | None, palm_offset: float = 0.0) -> None:
    n = 5
    with h5py.File(path, "w") as h5:
        demo = h5.create_group("data/demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset(
            "right_arm_joint_pos",
            data=np.arange(n * 7, dtype=np.float32).reshape(n, 7) + palm_offset,
        )
        obs.create_dataset(
            "right_hand_joint_pos",
            data=np.arange(n * 20, dtype=np.float32).reshape(n, 20) * 0.01 + palm_offset,
        )

        datagen = obs.create_group("datagen_info")
        eef_pose = datagen.create_group("eef_pose")
        mats = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], n, axis=0)
        mats[:, 0, 3] = np.linspace(0.1, 0.5, n, dtype=np.float32) + palm_offset
        mats[:, 1, 3] = np.linspace(-0.2, 0.2, n, dtype=np.float32)
        mats[:, 2, 3] = np.linspace(0.3, 0.7, n, dtype=np.float32)
        eef_pose.create_dataset("right", data=mats)

        starts = datagen.create_group("subtask_start_signals")
        pour_start = np.zeros((n,), dtype=bool)
        if pour_start_frame is not None:
            pour_start[pour_start_frame] = True
        starts.create_dataset("pour_start", data=pour_start)


def test_demo_grasp_reset_bank_selects_start_and_pour_start_lift_frame(tmp_path: Path) -> None:
    path = tmp_path / "demo.hdf5"
    _write_demo(path, pour_start_frame=3)

    bank = DemoGraspResetBank.from_hdf5_paths([path], device="cpu")

    assert bank.num_demos == 1
    assert int(bank.start_frames[0]) == 0
    assert int(bank.lift_frames[0]) == 3
    assert torch.allclose(bank.start_arm_joint_pos[0], torch.arange(7, dtype=torch.float32))
    assert torch.allclose(bank.lift_arm_joint_pos[0], torch.arange(21, 28, dtype=torch.float32))
    assert bank.lift_medoid_arm_joint_pos.shape == (7,)
    assert torch.isfinite(bank.lift_medoid_arm_joint_pos).all()


def test_demo_grasp_reset_bank_falls_back_to_last_frame_without_pour_start(tmp_path: Path) -> None:
    path = tmp_path / "demo.hdf5"
    _write_demo(path, pour_start_frame=None)

    bank = DemoGraspResetBank.from_hdf5_paths([path], device="cpu")

    assert int(bank.lift_frames[0]) == 4
    assert torch.allclose(bank.lift_arm_joint_pos[0], torch.arange(28, 35, dtype=torch.float32))


def test_lift_medoid_uses_palm_position_closest_to_mean(tmp_path: Path) -> None:
    path_a = tmp_path / "a.hdf5"
    path_b = tmp_path / "b.hdf5"
    _write_demo(path_a, pour_start_frame=2, palm_offset=0.0)
    _write_demo(path_b, pour_start_frame=2, palm_offset=10.0)

    bank = DemoGraspResetBank.from_hdf5_paths([path_a, path_b], device="cpu")

    assert bank.lift_medoid_arm_joint_pos.shape == (7,)
    assert torch.isfinite(bank.lift_medoid_arm_joint_pos).all()
    assert any(torch.allclose(bank.lift_medoid_arm_joint_pos, arm) for arm in bank.lift_arm_joint_pos)


def test_compute_demo_cup_spawn_local_uses_palm_xy_minus_offset_and_cfg_z() -> None:
    palm = torch.tensor([[0.42, -0.18, 0.33, 0.0, 0.0, 0.0]], dtype=torch.float32)
    offset_xy = torch.tensor([-0.06, -0.07], dtype=torch.float32)
    noise_xy = torch.tensor([[0.01, -0.02]], dtype=torch.float32)

    cup = compute_demo_cup_spawn_local(palm, offset_xy, 0.297, noise_xy)

    assert cup.shape == (1, 3)
    assert cup[0, 0] == pytest.approx(0.49)
    assert cup[0, 1] == pytest.approx(-0.13)
    assert cup[0, 2] == pytest.approx(0.297)
