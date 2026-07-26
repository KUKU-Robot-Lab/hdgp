from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import torch


def _install_isaaclab_math_stub() -> None:
    isaaclab = types.ModuleType("isaaclab")
    utils = types.ModuleType("isaaclab.utils")
    math_mod = types.ModuleType("isaaclab.utils.math")

    def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        half = angle * 0.5
        quat = torch.zeros(angle.shape[0], 4, dtype=angle.dtype, device=angle.device)
        quat[:, 0] = torch.cos(half)
        quat[:, 1:] = axis * torch.sin(half).unsqueeze(1)
        return quat

    def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    math_mod.quat_from_angle_axis = quat_from_angle_axis
    math_mod.quat_mul = quat_mul
    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.utils"] = utils
    sys.modules["isaaclab.utils.math"] = math_mod


_install_isaaclab_math_stub()

MODULE_PATH = Path(__file__).resolve().parents[1] / "warm_state_cache.py"
SPEC = importlib.util.spec_from_file_location("warm_state_cache", MODULE_PATH)
warm_state_cache = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = warm_state_cache
SPEC.loader.exec_module(warm_state_cache)

GraspWarmStateCache = warm_state_cache.GraspWarmStateCache
compute_arm_joint_match = warm_state_cache.compute_arm_joint_match


def test_warm_state_cache_persists_demo0_export_quality_fields(tmp_path: Path) -> None:
    cache = GraspWarmStateCache(
        capacity=2,
        device="cpu",
        source_meta={
            "object_spawn_z": 0.297,
            "export_mode": "left_grip_lift_wait_actual_grasp",
            "warm_min_contacts": 2,
            "warm_contact_stable_steps": 1,
        },
    )

    added = cache.append(
        arm=torch.ones(2, 7),
        hand=torch.ones(2, 20) * 2.0,
        palm_euler=torch.zeros(2, 6),
        cup_pos_local=torch.tensor([[0.24, -0.08, 0.34], [0.25, -0.07, 0.35]]),
        cup_quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        num_contacts=torch.tensor([4, 5]),
        per_finger_contact=torch.tensor(
            [[True, True, True, True, False], [True, True, True, True, True]]
        ),
        stable_contact_steps=torch.tensor([12, 18]),
        demo_file_idx=torch.tensor([0, 1]),
    )

    assert added == 2
    path = tmp_path / "warm.hdf5"
    cache.save_hdf5(path)

    with h5py.File(path, "r") as h5:
        assert h5.attrs["schema_version"] == 2
        assert h5.attrs["meta/export_mode"] == "left_grip_lift_wait_actual_grasp"
        grp = h5["warm_states"]
        assert np.asarray(grp["per_finger_contact"]).tolist() == [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ]
        assert np.asarray(grp["stable_contact_steps"]).tolist() == [12, 18]
        assert np.asarray(grp["num_contacts"]).tolist() == [4.0, 5.0]


def test_compute_arm_joint_match_tracks_tolerance_and_hold_steps() -> None:
    actual = torch.tensor(
        [
            [0.0, 0.01, -0.02, 0.03, 0.0, 0.0, 0.034],
            [0.0, 0.01, -0.02, 0.03, 0.0, 0.0, 0.036],
            [0.0, 0.01, -0.02, 0.03, 0.0, 0.0, 0.034],
        ]
    )
    target = torch.zeros(3, 7)
    previous_hold = torch.tensor([0, 3, 1])

    matched, hold = compute_arm_joint_match(
        actual,
        target,
        tol=0.035,
        previous_hold_steps=previous_hold,
        required_hold_steps=2,
    )

    assert hold.tolist() == [1, 0, 2]
    assert matched.tolist() == [False, False, True]
