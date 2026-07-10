"""실물 HDF5 → canonical 정렬된 tracking track.

teleop HDF5의 joint_names attr는 JSON 문자열이고 legacy 이름을 담는다.
따라서 읽는 즉시 manifest로 canonical 이름으로 정규화한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from r2s_autotune.config import RealTrackConfig
from r2s_autotune.joint_contract import JointContractError, Manifest, normalize_joint_names


@dataclass(frozen=True)
class RealTrack:
    """실제 로봇의 명령/응답. 모든 배열은 [T, J], 단위는 rad."""

    time: np.ndarray
    q_cmd: np.ndarray
    q_real: np.ndarray
    dq_real: np.ndarray | None
    joint_names: tuple[str, ...]
    dt: float

    @property
    def num_steps(self) -> int:
        return int(self.q_cmd.shape[0])

    def select(self, joints: Sequence[str]) -> "RealTrack":
        """관심 관절만 남긴 새 track을 돌려준다 (원본 불변)."""
        missing = [j for j in joints if j not in self.joint_names]
        if missing:
            raise JointContractError(f"track has no such joints: {missing}")
        idx = [self.joint_names.index(j) for j in joints]
        return RealTrack(
            time=self.time,
            q_cmd=self.q_cmd[:, idx],
            q_real=self.q_real[:, idx],
            dq_real=None if self.dq_real is None else self.dq_real[:, idx],
            joint_names=tuple(joints),
            dt=self.dt,
        )


def _joint_names(dataset: h5py.Dataset) -> tuple[str, ...]:
    raw = dataset.attrs.get("joint_names")
    if raw is None:
        raise JointContractError(f"dataset missing 'joint_names' attr: {dataset.name}")
    if isinstance(raw, bytes):
        raw = raw.decode()
    if isinstance(raw, str):
        return tuple(json.loads(raw))
    return tuple(x.decode() if isinstance(x, bytes) else str(x) for x in raw)


def _read(demo: h5py.Group, key: str) -> tuple[np.ndarray, tuple[str, ...]]:
    if key not in demo:
        raise KeyError(f"dataset not found in demo: {key}")
    dataset = demo[key]
    return np.asarray(dataset[()], dtype=np.float64), _joint_names(dataset)


def _finite_diff(q: np.ndarray, dt: float) -> np.ndarray:
    if q.shape[0] < 2:
        return np.zeros_like(q)
    return np.gradient(q, dt, axis=0)


def load_real_track(config: RealTrackConfig, manifest: Manifest) -> RealTrack:
    path = Path(config.hdf5)
    if not path.is_file():
        raise FileNotFoundError(f"real track HDF5 not found: {path}")

    with h5py.File(path, "r") as handle:
        root = handle["data"] if "data" in handle else handle
        if config.demo_key not in root:
            raise KeyError(f"demo '{config.demo_key}' not in {path}")
        demo = root[config.demo_key]

        q_cmd, cmd_names = _read(demo, config.command_dataset)
        q_real, real_names = _read(demo, config.measured_dataset)

        dq_real: np.ndarray | None = None
        if config.velocity_dataset:
            dq_raw, vel_names = _read(demo, config.velocity_dataset)
            if normalize_joint_names(vel_names, manifest) != normalize_joint_names(real_names, manifest):
                raise JointContractError("velocity dataset joint order differs from measured")
            dq_real = dq_raw

        time = (
            np.asarray(demo["timestamps_ns"][()], dtype=np.float64) * 1e-9
            if "timestamps_ns" in demo
            else np.arange(q_cmd.shape[0], dtype=np.float64) * config.dt
        )
        time = time - time[0]

    cmd_canonical = normalize_joint_names(cmd_names, manifest)
    real_canonical = normalize_joint_names(real_names, manifest)
    if cmd_canonical != real_canonical:
        raise JointContractError(
            "command and measured datasets have different joint order after normalization"
        )
    if q_cmd.shape != q_real.shape:
        raise ValueError(f"shape mismatch: q_cmd {q_cmd.shape} vs q_real {q_real.shape}")

    if dq_real is None:
        dq_real = _finite_diff(q_real, config.dt)

    return RealTrack(
        time=time,
        q_cmd=q_cmd,
        q_real=q_real,
        dq_real=dq_real,
        joint_names=cmd_canonical,
        dt=config.dt,
    )
