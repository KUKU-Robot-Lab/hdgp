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

"""Print observation layout and command slice offsets for a task."""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Print observation term layout and command slice offsets.")
parser.add_argument("--task", required=True, help="Gym task id, e.g. reach2g-v1.")
parser.add_argument("--group", default="policy", help="Observation group name (default: policy).")
parser.add_argument(
    "--command-terms",
    default="left_pose_command,right_pose_command,pose_command,ee_pose,object_pose",
    help="Comma-separated observation term names to treat as commands.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to create.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric and use USD I/O operations.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

if args_cli.enable_cameras:
    args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _prod(shape: tuple[int, ...]) -> int:
    if not shape:
        return 0
    return int(math.prod(shape))


def main():
    import gymnasium as gym

    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import openarm.tasks  # noqa: F401

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs_mgr = env.unwrapped.observation_manager

    print(obs_mgr)

    group = args_cli.group
    term_names = obs_mgr.active_terms.get(group, [])
    term_dims = obs_mgr.group_obs_term_dim.get(group, [])
    if not term_names:
        raise RuntimeError(f"Observation group '{group}' not found.")

    term_sizes = [_prod(shape) for shape in term_dims]

    offsets = []
    running = 0
    for size in term_sizes:
        offsets.append(running)
        running += size

    command_terms = [name.strip() for name in args_cli.command_terms.split(",") if name.strip()]
    command_indices = [i for i, name in enumerate(term_names) if name in command_terms]

    if not command_indices:
        print("[WARN] No command terms found in observation group.")
        return

    slices = []
    for idx in command_indices:
        start = offsets[idx]
        end = start + term_sizes[idx]
        slices.append((term_names[idx], start, end))

    print("Command term slices:")
    for name, start, end in slices:
        print(f"  - {name}: [{start}, {end})")

    contiguous = (
        len(command_indices) == 1
        or all(command_indices[i + 1] == command_indices[i] + 1 for i in range(len(command_indices) - 1))
    )
    if contiguous:
        start = slices[0][1]
        end = slices[-1][2]
        print(f"command_slice=[{start}, {end}]")
    else:
        print("[WARN] Command terms are not contiguous. Consider reorganizing observations.")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
