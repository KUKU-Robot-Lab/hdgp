#!/usr/bin/env python3
# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export a single USD that contains the task’s robot + table + objects."""

import argparse
import json
import os
import sys
import torch

import torch

from isaaclab.app import AppLauncher

# helper to parse CLI
# helper to parse CLI
parser = argparse.ArgumentParser(description="Export a task scene as USD.")
parser.add_argument("--task", required=True, help="Name of the Isaac Lab task to export.")
parser.add_argument(
    "--agent",
    default="rsl_rl_cfg_entry_point",
    help="Agent config used to resolve Hydra (usually matches the training entry point).",
)
parser.add_argument(
    "--output",
    default=None,
    help="Path to write the combined USD. Defaults to <task>_scene.usd in the current directory.",
)
parser.add_argument(
    "--joint-input",
    default=None,
    help="Optional joint-state JSON to apply before exporting the stage.",
)
parser.add_argument(
    "--joint-output",
    default=None,
    help="Path to write the robot joint dictionary (defaults to <task>_joint_state.json).",
)
# allow overriding simulator settings
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# launch Omniverse application before importing modules requiring omni/pxr
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import openarm  # noqa: F401 registers tasks
import isaaclab_tasks  # noqa: F401 registers dependencies
from isaaclab_tasks.utils.hydra import hydra_task_config

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    """Build the specified task and export its stage as USD."""

    # make sure we only spawn one env to keep things lightweight
    if hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = 1
    if hasattr(env_cfg.scene, "env_spacing"):
        env_cfg.scene.env_spacing = 0.5

    if joint_input_path := args_cli.joint_input:
        with open(joint_input_path, "r", encoding="utf-8") as joint_file:
            desired_joint_state = json.load(joint_file)
        # OVERWRITE the initial state in the config BEFORE the environment is created.
        # This is a much more reliable way to set the initial pose.
        if desired_joint_state and hasattr(env_cfg.scene, "robot"):
            print("[INFO] Overwriting robot's initial joint positions from the provided JSON file.")
            # Define fixed joints that should not have their position set from the JSON,
            # as they have a joint range of zero and are likely for defining frames.
            fixed_joints = [
                "openarm_left_hand",
                "openarm_right_hand",
                "openarm_left_ee_tcp_joint",
                "openarm_right_ee_tcp_joint",
            ]
            # Filter out the fixed joints from the desired state dictionary
            movable_joint_state = {k: v for k, v in desired_joint_state.items() if k not in fixed_joints}
            print(f"[INFO] Applying positions for {len(movable_joint_state)} movable joints (fixed joints were excluded).")
            # Overwrite the config's joint positions with only the movable joints
            env_cfg.scene.robot.init_state.joint_pos = movable_joint_state
    else:
        desired_joint_state = None

    # Create the environment with the (possibly modified) configuration
    print("[INFO] Creating environment...")
    env = gym.make(args_cli.task, cfg=env_cfg)
    # The reset call will apply the initial state, including our modified joint positions.
    print("[INFO] Resetting environment to apply the initial state...")
    env.reset()

    # Unwrap the environment to access the base environment's properties like 'device'.
    core_env = env.unwrapped if hasattr(env, "unwrapped") else env

    # Stepping simulation once to finalize the scene:
    # Convert numpy array action to torch tensor for the environment's step function.
    print("[INFO] Stepping simulation once to finalize the scene...")
    action_numpy = env.action_space.sample()
    action_tensor = torch.from_numpy(action_numpy).to(core_env.device)  # Use core_env.device
    env.step(action_tensor)

    stage = core_env.scene.stage
    output_path = os.path.abspath(args_cli.output or f"{args_cli.task}_scene.usd")
    stage.Export(output_path)
    print(f"[INFO] Exported combined USD to: {output_path}")

    # export initial joint targets if available in configuration
    joint_state = {}
    robot_cfg = getattr(env_cfg.scene, "robot", None)
    if robot_cfg and getattr(robot_cfg, "init_state", None):
        joint_state = getattr(robot_cfg.init_state, "joint_pos", {}) or {}

    robot_in_scene = core_env.scene["robot"]
    final_joint_state = {
        name: float(robot_in_scene.data.joint_pos[0, idx])
        for idx, name in enumerate(robot_in_scene.joint_names)
    }
    joint_output_path = os.path.abspath(args_cli.joint_output or f"{args_cli.task}_joint_state.json")
    os.makedirs(os.path.dirname(joint_output_path) or ".", exist_ok=True)
    with open(joint_output_path, "w", encoding="utf-8") as joints_file:
        json.dump(final_joint_state, joints_file, indent=2)
    print(f"[INFO] Saved joint dictionary to: {joint_output_path}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
