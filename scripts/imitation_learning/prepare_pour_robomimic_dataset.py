#!/usr/bin/env python3
"""Prepare pour_v1 teleop HDF5 files for Robomimic BC training.

The source demonstrations contain object/cup pose fields that may be dummy
values.  This builder exports robot/tactile observations and appends the fixed
sim spawn cup poses so the policy observation contract matches sim rollout.
For physically moving cup observations, use replay_pour_demos_to_robomimic_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


DEFAULT_INPUT_DIR = Path("/home/user/rl_ws/datasets")
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "pour_v1_robot_sensor_bc_robomimic.hdf5"
DEFAULT_DEMO_IDS = tuple(range(1, 21))
DEFAULT_ENV_NAME = "Pour-Mimic"
DEFAULT_SOURCE_CUP_POSE = np.array([0.27, -0.10, 0.277, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_TARGET_CUP_POSE = np.array([0.27, 0.10, 0.277, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
VALID_CONTROL_SCOPES = ("full", "left", "right")

_SOURCE_OBS = (
    ("actor_obs", "actor_obs", 91),
    ("right_joint_pos", "right_joint_pos", 27),
    ("right_joint_vel", "right_joint_vel", 27),
    ("left_joint_pos", "left_joint_pos", 7),
    ("left_joint_vel", "left_joint_vel", 7),
    ("tip_force_norm", "tip_force_norm", 5),
    ("prev_actions", "prev_actions", 18),
)


@dataclass
class BuildReport:
    kept: int = 0
    skipped: int = 0
    missing_demo_ids: list[int] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, *, source: Path, demo: str, status: str, reason: str, length: int = 0) -> None:
        if status == "kept":
            self.kept += 1
        else:
            self.skipped += 1
        self.entries.append(
            {
                "source": source.name,
                "demo": demo,
                "status": status,
                "reason": reason,
                "length": int(length),
            }
        )


def discover_demo_paths(input_dir: Path, demo_ids: Iterable[int] = DEFAULT_DEMO_IDS) -> tuple[list[Path], list[int]]:
    """Return existing pour_v1 demo paths in numeric order and missing ids."""
    paths: list[Path] = []
    missing: list[int] = []
    for demo_id in demo_ids:
        path = input_dir / f"pour_v1_a{demo_id}.hdf5"
        if path.exists():
            paths.append(path)
        else:
            missing.append(int(demo_id))
    return paths, missing


def _demo_sort_key(name: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"demo_(\d+)", name)
    if match:
        return (0, int(match.group(1)))
    return (1, name)


def _validate_demo(demo: h5py.Group) -> tuple[bool, str, int]:
    if "actions" not in demo or "obs" not in demo:
        return False, "missing_actions_or_obs", 0
    actions = demo["actions"]
    if len(actions.shape) != 2 or actions.shape[1] != 18:
        return False, "bad_action_shape", int(actions.shape[0]) if actions.shape else 0
    if actions.shape[0] < 2:
        return False, "too_short", int(actions.shape[0])

    obs = demo["obs"]
    for _, src_name, dim in _SOURCE_OBS:
        if src_name not in obs:
            return False, "missing_robot_sensor_obs", int(actions.shape[0])
        dataset = obs[src_name]
        if len(dataset.shape) != 2 or dataset.shape[0] != actions.shape[0] or dataset.shape[1] != dim:
            return False, f"bad_obs_shape_{src_name}", int(actions.shape[0])
    return True, "ok", int(actions.shape[0])


def _fixed_pose_block(pose: np.ndarray, length: int) -> np.ndarray:
    return np.repeat(pose.reshape(1, 7), length, axis=0).astype(np.float32)


def _write_shifted_obs(
    src_obs: h5py.Group,
    dst_obs: h5py.Group,
    start: int,
    stop: int,
    *,
    source_cup_pose: np.ndarray,
    target_cup_pose: np.ndarray,
) -> None:
    length = stop - start
    source_pose = _fixed_pose_block(source_cup_pose, length)
    target_pose = _fixed_pose_block(target_cup_pose, length)

    actor_obs = np.asarray(src_obs["actor_obs"][start:stop], dtype=np.float32)
    policy = np.concatenate([actor_obs, source_pose, target_pose], axis=-1)
    dst_obs.create_dataset("policy", data=policy, compression="gzip")

    for dst_name, src_name, _ in _SOURCE_OBS[1:]:
        dst_obs.create_dataset(dst_name, data=src_obs[src_name][start:stop], compression="gzip")
    dst_obs.create_dataset("source_cup_pose", data=source_pose, compression="gzip")
    dst_obs.create_dataset("target_cup_pose", data=target_pose, compression="gzip")


def _mask_actions(actions: np.ndarray, control_scope: str) -> np.ndarray:
    if control_scope not in VALID_CONTROL_SCOPES:
        raise ValueError(f"control_scope must be one of {VALID_CONTROL_SCOPES}, got {control_scope!r}")
    masked = np.asarray(actions, dtype=np.float32).copy()
    if control_scope == "left":
        masked[:, :11] = 0.0
    elif control_scope == "right":
        masked[:, 11:18] = 0.0
    return masked


def build_dataset(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    input_file: Path | None = None,
    demo_ids: Iterable[int] = DEFAULT_DEMO_IDS,
    env_name: str = DEFAULT_ENV_NAME,
    control_scope: str = "full",
    source_cup_pose: np.ndarray = DEFAULT_SOURCE_CUP_POSE,
    target_cup_pose: np.ndarray = DEFAULT_TARGET_CUP_POSE,
) -> BuildReport:
    """Build a Robomimic-compatible low-dimensional BC dataset.

    Source arrays have T actions and T observations.  Robomimic samples
    transitions, so the output stores T-1 actions, obs[0:T-1], and
    next_obs[1:T].
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    if control_scope not in VALID_CONTROL_SCOPES:
        raise ValueError(f"control_scope must be one of {VALID_CONTROL_SCOPES}, got {control_scope!r}")

    if input_file is not None:
        paths = [Path(input_file)]
        missing = []
    else:
        paths, missing = discover_demo_paths(input_dir, demo_ids)
    report = BuildReport(missing_demo_ids=missing)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as out:
        data_group = out.create_group("data")
        data_group.attrs["total"] = 0
        data_group.attrs["env_args"] = json.dumps({"env_name": env_name, "type": 2, "env_kwargs": {}})
        out.create_group("mask")
        train_demo_names: list[str] = []
        demo_index = 0

        for path in paths:
            with h5py.File(path, "r") as src:
                source_data = src.get("data")
                if source_data is None:
                    report.add(source=path, demo="", status="skipped", reason="missing_data_group")
                    continue

                for source_demo_name in sorted(source_data.keys(), key=_demo_sort_key):
                    source_demo = source_data[source_demo_name]
                    ok, reason, length = _validate_demo(source_demo)
                    if not ok:
                        report.add(source=path, demo=source_demo_name, status="skipped", reason=reason, length=length)
                        continue

                    out_length = length - 1
                    demo_name = f"demo_{demo_index}"
                    dst_demo = data_group.create_group(demo_name)
                    dst_demo.attrs["num_samples"] = out_length
                    dst_demo.attrs["source_file"] = path.name
                    dst_demo.attrs["source_demo"] = source_demo_name
                    dst_demo.attrs["success"] = bool(source_demo.attrs.get("success", True))
                    dst_demo.attrs["control_scope"] = control_scope

                    actions = _mask_actions(source_demo["actions"][:out_length], control_scope)
                    dst_demo.create_dataset("actions", data=actions, compression="gzip")
                    dst_demo.create_dataset("rewards", data=np.zeros(out_length, dtype=np.float32), compression="gzip")
                    dones = np.zeros(out_length, dtype=np.bool_)
                    dones[-1] = True
                    dst_demo.create_dataset("dones", data=dones, compression="gzip")

                    _write_shifted_obs(
                        source_demo["obs"],
                        dst_demo.create_group("obs"),
                        0,
                        out_length,
                        source_cup_pose=source_cup_pose,
                        target_cup_pose=target_cup_pose,
                    )
                    _write_shifted_obs(
                        source_demo["obs"],
                        dst_demo.create_group("next_obs"),
                        1,
                        length,
                        source_cup_pose=source_cup_pose,
                        target_cup_pose=target_cup_pose,
                    )

                    data_group.attrs["total"] += out_length
                    train_demo_names.append(demo_name)
                    report.add(source=path, demo=source_demo_name, status="kept", reason="ok", length=out_length)
                    demo_index += 1

        string_dtype = h5py.string_dtype(encoding="utf-8")
        out["mask"].create_dataset("train", data=np.asarray(train_demo_names, dtype=object), dtype=string_dtype)
        data_group.attrs["num_demos"] = report.kept

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--demo-id", type=int, action="append", dest="demo_ids")
    parser.add_argument("--control-scope", choices=VALID_CONTROL_SCOPES, default="full")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_dataset(
        args.input_dir,
        args.output,
        input_file=args.input_file,
        demo_ids=tuple(args.demo_ids) if args.demo_ids else DEFAULT_DEMO_IDS,
        env_name=args.env_name,
        control_scope=args.control_scope,
    )
    print(f"kept={report.kept} skipped={report.skipped} missing={report.missing_demo_ids}")
    for entry in report.entries:
        print("{source}:{demo} {status} {reason} length={length}".format(**entry))


if __name__ == "__main__":
    main()
