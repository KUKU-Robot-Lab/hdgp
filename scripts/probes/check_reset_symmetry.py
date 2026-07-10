#!/usr/bin/env python3
"""Check left/right object reset symmetry in robot root frame."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser():
    parser = argparse.ArgumentParser(description="Check reset symmetry for bimanual tasks.")
    parser.add_argument("--task", type=str, default="grasp2g-v0")
    parser.add_argument("--num_resets", type=int, default=50)
    parser.add_argument("--num_envs", type=int, default=None, help="Override env num_envs if supported")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    app_launcher = AppLauncher(args)

    import gymnasium as gym
    import torch

    # Ensure hdgp/source is on path for task registration.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root / "source"))

    # Register OpenArm tasks
    import openarm.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from openarm.tasks.manager_based.openarm_manipulation.pipeline.gripper.both.grasp_2g import mdp as grasp_mdp
    from isaaclab.managers import SceneEntityCfg

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs or 1,
        use_fabric=not args.disable_fabric if hasattr(args, "disable_fabric") else True,
    )

    env = gym.make(args.task, cfg=env_cfg)
    core_env = env.unwrapped

    robot_cfg = SceneEntityCfg("robot")
    obj_cfg = SceneEntityCfg("object")
    obj2_cfg = SceneEntityCfg("object2")

    diffs = []
    for _ in range(args.num_resets):
        env.reset()
        left = grasp_mdp.object_position_in_robot_root_frame(core_env, robot_cfg, obj_cfg)
        right = grasp_mdp.object_position_in_robot_root_frame(core_env, robot_cfg, obj2_cfg)
        # mirror right across Y (robot frame)
        right_mirror = right.clone()
        right_mirror[:, 1] *= -1.0
        diff = left - right_mirror
        diffs.append(diff)

    diff_all = torch.cat(diffs, dim=0)
    abs_diff = torch.abs(diff_all)
    mean_xyz = abs_diff.mean(dim=0)
    max_xyz = abs_diff.max(dim=0).values
    mean_norm = torch.norm(diff_all, dim=1).mean().item()
    max_norm = torch.norm(diff_all, dim=1).max().item()

    print("Robot-frame symmetry (left vs mirrored right):")
    print(f"  mean abs diff (x,y,z): {mean_xyz.tolist()}")
    print(f"  max  abs diff (x,y,z): {max_xyz.tolist()}")
    print(f"  mean L2 diff: {mean_norm:.6f}")
    print(f"  max  L2 diff: {max_norm:.6f}")

    env.close()


if __name__ == "__main__":
    main()
