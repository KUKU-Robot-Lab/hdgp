"""IL2 Demo Loader: EEF 궤적 + 손가락 관절 로드 (Phase 1 EEF 추종용).

cup 데이터(source_cup, target_cup)는 a11-a20에서 더미값이므로 로드하지 않음.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class IL2Trajectory:
    eef_pos: np.ndarray        # (T, 3)  right palm pos (env-local frame)
    eef_quat_xyzw: np.ndarray  # (T, 4)  right palm quaternion xyzw
    hand_joints: np.ndarray    # (T, 20) right_hand_joint_pos
    arm_joints: np.ndarray     # (T, 7)  right_arm_joint_pos


def load_il2_trajectory(path: Path, stride: int = 1) -> IL2Trajectory:
    """HDF5 demo 파일 → IL2Trajectory."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        demo = h5["data"]["demo_0"]
        s = slice(0, None, stride)
        eef_mat = np.array(
            demo["obs"]["datagen_info"]["eef_pose"]["right"][s], dtype=np.float64
        )  # (T, 4, 4)
        hand_joints = np.array(
            demo["obs"]["right_hand_joint_pos"][s], dtype=np.float32
        )  # (T, 20)
        arm_joints = np.array(
            demo["obs"]["right_arm_joint_pos"][s], dtype=np.float32
        )  # (T, 7)

    eef_pos = eef_mat[:, :3, 3].astype(np.float32)  # (T, 3)
    eef_quat_xyzw = Rotation.from_matrix(eef_mat[:, :3, :3]).as_quat().astype(np.float32)  # (T, 4)

    return IL2Trajectory(
        eef_pos=eef_pos,
        eef_quat_xyzw=eef_quat_xyzw,
        hand_joints=hand_joints,
        arm_joints=arm_joints,
    )
