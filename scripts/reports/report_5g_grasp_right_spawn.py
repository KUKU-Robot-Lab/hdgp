#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report spawn-time palm/cup world positions and relative vector for 5g_grasp_right-v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report initial spawn states for 5g_grasp_right-v1."
    )
    parser.add_argument("--task", type=str, default="5g_grasp_right-v1")
    parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--env_id", type=int, default=0, help="Environment index to report.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path. If omitted, only print.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _to_list(x):
    return [float(v) for v in x.detach().cpu().tolist()]


def _resolve_palm_center_world(core_env, env_id: int):
    """Resolve palm_center world position robustly across task implementations."""
    robot = core_env.scene["robot"]
    env_origin = core_env.scene.env_origins[env_id]

    # 1) Prefer direct body pose if body exists in physics articulation.
    if "palm_center" in robot.data.body_names:
        body_idx = robot.data.body_names.index("palm_center")
        return robot.data.body_pos_w[env_id, body_idx], "robot.data.body_pos_w[palm_center]"

    # 2) Fallback: task-maintained local palm position (local + env_origin -> world).
    if hasattr(core_env, "palm_center_pos"):
        return core_env.palm_center_pos[env_id] + env_origin, "core_env.palm_center_pos + env_origin"

    # 3) Fallback: hand_pos[:,0] is palm in this task.
    if hasattr(core_env, "hand_pos"):
        return core_env.hand_pos[env_id, 0] + env_origin, "core_env.hand_pos[:,0] + env_origin"

    raise RuntimeError("Could not resolve palm_center position from environment.")


def _joint_dict(robot, env_id: int):
    return {
        name: float(robot.data.joint_pos[env_id, idx].item())
        for idx, name in enumerate(robot.joint_names)
    }


def main() -> int:
    parser = build_parser()
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Ensure local task package import works when executed from IsaacLab root.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "source"))

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.hydra import hydra_task_config

    # Registers task ids like "5g_grasp_right-v1".
    import openarm.tasks  # noqa: F401

    @hydra_task_config(args.task, args.agent)
    def _run(env_cfg, _agent_cfg):
        if hasattr(env_cfg.scene, "num_envs"):
            env_cfg.scene.num_envs = args.num_envs

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        obs, _ = env.reset()
        _ = obs

        core_env = env.unwrapped if hasattr(env, "unwrapped") else env
        if args.env_id < 0 or args.env_id >= core_env.num_envs:
            raise ValueError(
                f"env_id={args.env_id} is out of range for num_envs={core_env.num_envs}"
            )

        # In this task, reset sets spawn states directly in _reset_idx().
        # One sim step is used to ensure state buffers are fully synchronized.
        zero_action = torch.zeros((core_env.num_envs, core_env.cfg.num_actions), device=core_env.device)
        env.step(zero_action)

        env_id = args.env_id
        robot = core_env.scene["robot"]
        cup = core_env.scene["cup"] if "cup" in core_env.scene.keys() else None
        if cup is None:
            raise RuntimeError("Scene has no 'cup'. Check task config enable_cup=True.")

        palm_world, palm_source = _resolve_palm_center_world(core_env, env_id)
        cup_world = cup.data.root_pos_w[env_id]
        rel_cup_minus_palm = cup_world - palm_world
        env_origin = core_env.scene.env_origins[env_id]

        joint_pos_all = _joint_dict(robot, env_id)
        right_arm_joint_pos = {
            name: joint_pos_all[name]
            for name in robot.joint_names
            if name.startswith("openarm_right_joint")
        }

        result = {
            "task": args.task,
            "env_id": int(env_id),
            "palm_center_world_pos": _to_list(palm_world),
            "palm_center_world_source": palm_source,
            "cup_world_pos": _to_list(cup_world),
            "relative_vector_cup_minus_palm": _to_list(rel_cup_minus_palm),
            "env_origin_world": _to_list(env_origin),
            "robot_right_arm_joint_pos": right_arm_joint_pos,
            "robot_all_joint_pos": joint_pos_all,
        }

        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[INFO] Saved JSON: {output_path}")

        env.close()

    try:
        _run()
    finally:
        if hasattr(simulation_app, "close"):
            simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
