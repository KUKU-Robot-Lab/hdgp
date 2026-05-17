from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parents[1]


def _load_as_pkg(module_stem: str):
    pkg_name = "pour_right_v5_testpkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(BASE_DIR)]
        sys.modules[pkg_name] = pkg

    full_name = f"{pkg_name}.{module_stem}"
    spec = importlib.util.spec_from_file_location(full_name, BASE_DIR / f"{module_stem}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def _rot_z(yaw_rad: float) -> np.ndarray:
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _to_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def _write_demo_pose_reference_file(path: Path, local_traj: np.ndarray) -> None:
    T = local_traj.shape[0]
    timestamps = (np.arange(T, dtype=np.int64) * 16_666_667)

    spawns = [
        (np.array([0.10, -0.15, 0.20], dtype=np.float32), 0.0),
        (np.array([-0.25, 0.35, 0.05], dtype=np.float32), np.deg2rad(60.0)),
    ]

    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        for idx, (cup_t, yaw) in enumerate(spawns):
            demo = data.create_group(f"demo_{idx}")
            R_cup = _rot_z(float(yaw))

            cup_pose = np.stack([_to_pose(R_cup, cup_t) for _ in range(T)], axis=0)
            eef_pose = np.stack(
                [_to_pose(R_cup, cup_t + (R_cup @ local_traj[t])) for t in range(T)],
                axis=0,
            )

            demo.create_dataset("obs/right_arm_joint_pos", data=np.zeros((T, 7), dtype=np.float32))
            demo.create_dataset("obs/right_hand_joint_pos", data=np.zeros((T, 20), dtype=np.float32))
            demo.create_dataset("obs/right_hand_reference_joint_pos", data=np.zeros((T, 20), dtype=np.float32))
            demo.create_dataset("obs/datagen_info/eef_pose/right", data=eef_pose)
            demo.create_dataset("obs/datagen_info/target_eef_pose/right", data=eef_pose)
            demo.create_dataset("obs/datagen_info/object_pose/source_cup", data=cup_pose)
            demo.create_dataset("timestamps_ns", data=timestamps)


def _apply_world_transform(pose_seq: np.ndarray, Rg: np.ndarray, tg: np.ndarray) -> np.ndarray:
    out = np.empty_like(pose_seq)
    out[:, :3, :3] = np.einsum("ij,tjk->tik", Rg, pose_seq[:, :3, :3])
    out[:, :3, 3] = np.einsum("ij,tj->ti", Rg, pose_seq[:, :3, 3]) + tg
    out[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return out


def _write_bc_demo_file(path: Path, *, transformed: bool) -> None:
    T = 5
    timestamps = (np.arange(T, dtype=np.int64) * 16_666_667)

    R_src = _rot_z(np.deg2rad(20.0))
    t_src = np.array([0.20, -0.10, 0.22], dtype=np.float32)
    src_pose = np.stack([_to_pose(R_src, t_src) for _ in range(T)], axis=0)

    R_tgt = _rot_z(np.deg2rad(-10.0))
    t_tgt = np.array([-0.15, 0.25, 0.20], dtype=np.float32)
    tgt_cup_pose = np.stack([_to_pose(R_tgt, t_tgt) for _ in range(T)], axis=0)

    base_local = np.array([0.02, -0.01, 0.03], dtype=np.float32)
    base_world = t_src + (R_src @ base_local)
    base_pose = _to_pose(R_src, base_world)
    eef_pose = np.stack([base_pose for _ in range(T)], axis=0)

    # cup-local target trajectory that should stay invariant under global spawn transform
    target_local_deltas = np.array(
        [
            [0.01, 0.00, 0.00],
            [0.02, -0.01, 0.00],
            [0.03, -0.01, 0.01],
            [0.01, 0.02, 0.00],
            [0.00, 0.00, -0.01],
        ],
        dtype=np.float32,
    )
    tgt_pose = np.stack(
        [_to_pose(R_src, base_world + (R_src @ d)) for d in target_local_deltas],
        axis=0,
    )

    if transformed:
        Rg = _rot_z(np.deg2rad(75.0))
        tg = np.array([0.40, -0.35, 0.15], dtype=np.float32)
        src_pose = _apply_world_transform(src_pose, Rg, tg)
        tgt_cup_pose = _apply_world_transform(tgt_cup_pose, Rg, tg)
        eef_pose = _apply_world_transform(eef_pose, Rg, tg)
        tgt_pose = _apply_world_transform(tgt_pose, Rg, tg)

    with h5py.File(path, "w") as h5:
        demo = h5.create_group("data").create_group("demo_0")
        demo.create_dataset("obs/right_arm_joint_pos", data=np.zeros((T, 7), dtype=np.float32))
        demo.create_dataset("obs/right_hand_joint_pos", data=np.zeros((T, 20), dtype=np.float32))
        demo.create_dataset("obs/right_joint_vel", data=np.zeros((T, 27), dtype=np.float32))
        demo.create_dataset("obs/tip_force_norm", data=np.zeros((T, 5), dtype=np.float32))
        demo.create_dataset("obs/datagen_info/eef_pose/right", data=eef_pose)
        demo.create_dataset("obs/datagen_info/target_eef_pose/right", data=tgt_pose)
        demo.create_dataset("obs/datagen_info/object_pose/source_cup", data=src_pose)
        demo.create_dataset("obs/datagen_info/object_pose/target_cup", data=tgt_cup_pose)
        demo.create_dataset("obs/datagen_info/subtask_start_signals/pour_start", data=np.zeros(T, dtype=np.bool_))
        demo.create_dataset("timestamps_ns", data=timestamps)


def test_demo_pose_reference_palm_in_cup_is_spawn_invariant(tmp_path: Path) -> None:
    demo_pose_reference = _load_as_pkg("demo_pose_reference")

    local_traj = np.array(
        [
            [0.03, -0.02, 0.05],
            [0.04, -0.01, 0.04],
            [0.02, 0.00, 0.06],
        ],
        dtype=np.float32,
    )
    h5_path = tmp_path / "synthetic_pose_ref.hdf5"
    _write_demo_pose_reference_file(h5_path, local_traj)

    bank = demo_pose_reference.DemoPoseReferenceBank.from_hdf5_paths([h5_path], phase="all", device="cpu")

    expected = np.concatenate([local_traj, local_traj], axis=0)
    assert bank.palm_in_cup_pos.shape == (expected.shape[0], 3)
    assert torch.allclose(bank.palm_in_cup_pos, torch.as_tensor(expected), atol=1e-5)


def test_bc_pos_action_is_spawn_invariant_for_translation_and_yaw(tmp_path: Path) -> None:
    demo_bc_buffer = _load_as_pkg("demo_bc_buffer")

    base_path = tmp_path / "demo_base.hdf5"
    transformed_path = tmp_path / "demo_transformed.hdf5"
    _write_bc_demo_file(base_path, transformed=False)
    _write_bc_demo_file(transformed_path, transformed=True)

    ep_base = demo_bc_buffer._load_episode(base_path, stride=1, device=torch.device("cpu"))
    ep_tf = demo_bc_buffer._load_episode(transformed_path, stride=1, device=torch.device("cpu"))

    assert ep_base.actions.shape == ep_tf.actions.shape
    assert torch.any(torch.abs(ep_base.actions[:, :3]) > 1e-6)
    assert torch.allclose(ep_base.actions[:, :3], ep_tf.actions[:, :3], atol=1e-5)


def test_pos_delta_mapping_uses_cup_frame_except_warmstart_collect() -> None:
    pour_right_utils = _load_as_pkg("pour_right_utils")

    delta_xyz = torch.tensor([[0.10, 0.00, 0.00]], dtype=torch.float32)
    yaw_90_quat_wxyz = torch.tensor([[0.70710677, 0.0, 0.0, 0.70710677]], dtype=torch.float32)

    world_delta_normal = pour_right_utils.map_delta_pos_to_world(
        delta_xyz,
        yaw_90_quat_wxyz,
        warmstart_collect_mode=False,
    )
    world_delta_warmstart = pour_right_utils.map_delta_pos_to_world(
        delta_xyz,
        yaw_90_quat_wxyz,
        warmstart_collect_mode=True,
    )

    assert torch.allclose(world_delta_normal, torch.tensor([[0.0, 0.10, 0.0]]), atol=1e-5)
    assert torch.allclose(world_delta_warmstart, delta_xyz, atol=1e-6)
