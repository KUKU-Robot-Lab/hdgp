#!/usr/bin/env python3
"""Replay pour_v1 demos in Isaac Sim and export Robomimic BC data.

This is the preferred builder when the policy should observe cups from sim.
Each demo resets the robot to the first recorded joint frame, leaves source and
target cups at the fixed scene spawn poses, replays the full action trajectory,
and records the environment's live ``obs["policy"]`` before and after each step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from prepare_pour_robomimic_dataset import (  # noqa: E402
    DEFAULT_DEMO_IDS,
    DEFAULT_ENV_NAME,
    DEFAULT_INPUT_DIR,
    VALID_CONTROL_SCOPES,
    _mask_actions,
    discover_demo_paths,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_ENV_NAME)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INPUT_DIR / "pour_v1_sim_replay_robot_object_bc_robomimic.hdf5",
    )
    parser.add_argument("--demo-id", type=int, action="append", dest="demo_ids")
    parser.add_argument("--control-scope", choices=VALID_CONTROL_SCOPES, default="full")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_OPENARM_SRC = Path("/home/user/rl_ws/hdgp/source/openarm")
if str(_OPENARM_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENARM_SRC))

import openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.both.pour_v1_mimic  # noqa: E402,F401


def _policy(obs: dict) -> torch.Tensor:
    return obs["policy"] if isinstance(obs, dict) else obs[0]["policy"]


def _set_joints(robot, pattern: str, values: torch.Tensor, joint_pos: torch.Tensor) -> None:
    joint_ids, _ = robot.find_joints([pattern])
    if len(joint_ids) != values.shape[-1]:
        raise RuntimeError(f"{pattern} matched {len(joint_ids)} joints, expected {values.shape[-1]}")
    joint_pos[:, joint_ids] = values


def _asset_init_pose(env, asset_name: str) -> torch.Tensor:
    asset_cfg = getattr(env.cfg.scene, asset_name)
    pos = torch.tensor(asset_cfg.init_state.pos, device=env.device, dtype=torch.float32).reshape(1, 3)
    rot = torch.tensor(asset_cfg.init_state.rot, device=env.device, dtype=torch.float32).reshape(1, 4)
    if hasattr(env.scene, "env_origins"):
        pos = pos + env.scene.env_origins[:1]
    return torch.cat([pos, rot], dim=-1)


def _reset_cup_to_init(env, asset_name: str) -> None:
    cup = env.scene[asset_name]
    cup.write_root_pose_to_sim(_asset_init_pose(env, asset_name))
    cup.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device, dtype=torch.float32))


def _passive_cups_for_scope(control_scope: str) -> tuple[str, ...]:
    if control_scope == "right":
        return ("target_cup",)
    if control_scope == "left":
        return ("source_cup",)
    return ()


def _reset_from_demo_first_frame(env, demo: h5py.Group) -> dict:
    """Reset fixed scene spawn, then set robot joints and cups deterministically."""
    env.scene.reset()

    robot = env.scene["robot"]
    obs = demo["obs"]
    device = env.device
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)

    right_arm_joint_pos = torch.as_tensor(obs["right_arm_joint_pos"][0], device=device, dtype=torch.float32).unsqueeze(0)
    right_hand_joint_pos = torch.as_tensor(obs["right_hand_joint_pos"][0], device=device, dtype=torch.float32).unsqueeze(0)
    left_joint_pos = torch.as_tensor(obs["left_joint_pos"][0], device=device, dtype=torch.float32).unsqueeze(0)

    _set_joints(robot, "openarm_right_joint[1-7]", right_arm_joint_pos, joint_pos)
    _set_joints(robot, "rj_dg_[1-5]_[1-4]", right_hand_joint_pos, joint_pos)
    _set_joints(robot, "openarm_left_joint[1-7]", left_joint_pos, joint_pos)

    if "left_gripper_joint_pos" in obs:
        left_gripper = torch.as_tensor(obs["left_gripper_joint_pos"][0], device=device, dtype=torch.float32).unsqueeze(0)
        _set_joints(robot, "openarm_left_finger_joint[1-2]", left_gripper, joint_pos)

    robot.write_joint_state_to_sim(position=joint_pos, velocity=joint_vel)
    _reset_cup_to_init(env, "source_cup")
    _reset_cup_to_init(env, "target_cup")
    env.sim.step()
    env.obs_buf = env.observation_manager.compute(update_history=True)
    return env.obs_buf


def _write_demo(dst_demo: h5py.Group, obs: np.ndarray, next_obs: np.ndarray, actions: np.ndarray) -> None:
    length = actions.shape[0]
    dst_demo.attrs["num_samples"] = length
    dst_demo.create_dataset("actions", data=actions, compression="gzip")
    dst_demo.create_dataset("rewards", data=np.zeros(length, dtype=np.float32), compression="gzip")
    dones = np.zeros(length, dtype=np.bool_)
    dones[-1] = True
    dst_demo.create_dataset("dones", data=dones, compression="gzip")
    dst_demo.create_group("obs").create_dataset("policy", data=obs, compression="gzip")
    dst_demo.create_group("next_obs").create_dataset("policy", data=next_obs, compression="gzip")


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.headless)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    demo_ids = tuple(args_cli.demo_ids) if args_cli.demo_ids else DEFAULT_DEMO_IDS
    paths, missing = discover_demo_paths(args_cli.input_dir, demo_ids)
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args_cli.output, "w") as out:
        data_group = out.create_group("data")
        data_group.attrs["total"] = 0
        data_group.attrs["num_demos"] = 0
        data_group.attrs["env_args"] = json.dumps({"env_name": args_cli.task, "type": 2, "env_kwargs": {}})
        data_group.attrs["missing_demo_ids"] = json.dumps(missing)
        mask_group = out.create_group("mask")
        train_names: list[str] = []

        demo_index = 0
        for path in paths:
            with h5py.File(path, "r") as src:
                for source_demo_name in sorted(src["data"].keys()):
                    source_demo = src["data"][source_demo_name]
                    obs_dict = _reset_from_demo_first_frame(env, source_demo)
                    source_actions = np.asarray(source_demo["actions"], dtype=np.float32)
                    actions = _mask_actions(source_actions, args_cli.control_scope)
                    passive_cups = _passive_cups_for_scope(args_cli.control_scope)

                    obs_frames: list[np.ndarray] = []
                    next_obs_frames: list[np.ndarray] = []
                    for action in actions:
                        obs_frames.append(_policy(obs_dict).detach().cpu().numpy()[0].astype(np.float32))
                        action_tensor = torch.as_tensor(action, device=env.device, dtype=torch.float32).reshape(1, -1)
                        obs_dict, _, _, _, _ = env.step(action_tensor)
                        for cup_name in passive_cups:
                            _reset_cup_to_init(env, cup_name)
                        if passive_cups:
                            env.obs_buf = env.observation_manager.compute(update_history=True)
                            obs_dict = env.obs_buf
                        next_obs_frames.append(_policy(obs_dict).detach().cpu().numpy()[0].astype(np.float32))

                    demo_name = f"demo_{demo_index}"
                    dst_demo = data_group.create_group(demo_name)
                    dst_demo.attrs["source_file"] = path.name
                    dst_demo.attrs["source_demo"] = source_demo_name
                    dst_demo.attrs["control_scope"] = args_cli.control_scope
                    _write_demo(dst_demo, np.asarray(obs_frames), np.asarray(next_obs_frames), actions)
                    data_group.attrs["total"] += actions.shape[0]
                    data_group.attrs["num_demos"] += 1
                    train_names.append(demo_name)
                    print(f"replayed {path.name}:{source_demo_name} -> {demo_name} len={actions.shape[0]}", flush=True)
                    demo_index += 1

        string_dtype = h5py.string_dtype(encoding="utf-8")
        mask_group.create_dataset("train", data=np.asarray(train_names, dtype=object), dtype=string_dtype)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
