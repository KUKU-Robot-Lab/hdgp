"""IL Demo Loader — a11-a20 HDF5에서 팔/손가락 궤적 로드.

Isaac Sim 의존 없음. IL 환경에서 joint tracking reward 계산에 사용.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class ILTrajectory:
    arm_joints: np.ndarray   # (T, 7)  right_arm_joint_pos
    finger_curl: np.ndarray  # (T, 5)  right_hand_curl (실제 tesollo 센서)
    hand_joints: np.ndarray  # (T, 20) right_hand_joint_pos (관절 추적용)
    pour_mask: np.ndarray    # (T,) bool  pour 단계 마스크


def load_il_trajectory(path: Path, stride: int = 1) -> ILTrajectory:
    """HDF5 demo 파일 → ILTrajectory.

    buffer.py의 finger_act = np.ones 버그 없이 실제 right_hand_curl 로드.
    """
    path = Path(path)
    with h5py.File(path, "r") as h5:
        demo = h5["data"]["demo_0"]
        s = slice(0, None, stride)

        arm_joints  = np.array(demo["obs"]["right_arm_joint_pos"][s], dtype=np.float32)    # (T, 7)
        finger_curl = np.array(demo["obs"]["right_hand_curl"][s], dtype=np.float32)         # (T, 5)
        hand_joints = np.array(demo["obs"]["right_hand_joint_pos"][s], dtype=np.float32)   # (T, 20)

        pour_start_raw = demo["obs"]["datagen_info"]["subtask_start_signals"]["pour_start"][s]
        pour_start = np.array(pour_start_raw, dtype=bool)

    T = arm_joints.shape[0]

    # pour_mask: pour_start 시그널이 있으면 그 이후를 전부 True
    if pour_start.any():
        first_pour = int(np.argmax(pour_start))
        pour_mask = np.zeros(T, dtype=bool)
        pour_mask[first_pour:] = True
    else:
        pour_mask = np.zeros(T, dtype=bool)

    return ILTrajectory(
        arm_joints=arm_joints,
        finger_curl=finger_curl,
        hand_joints=hand_joints,
        pour_mask=pour_mask,
    )
