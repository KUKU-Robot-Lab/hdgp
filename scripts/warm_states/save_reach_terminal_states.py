#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Save reach policy terminal states for curriculum learning.

This script runs the reach policy and saves successful terminal states
to be used as initial states for grasp training (roll-out reset).

Usage:
    cd /home/user/rl_ws/IsaacLab

    ./isaaclab.sh -p ../hdgp/scripts/warm_states/save_reach_terminal_states.py \
        --task Reach-v1 \
        --checkpoint /path/to/reach/model.pt \
        --num_episodes 1000 \
        --output ../hdgp/data/reach_terminal_states.pt \
        --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save reach policy terminal states for grasp curriculum learning."
    )
    parser.add_argument("--task", type=str, required=True, help="Reach task name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to reach policy checkpoint")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes to collect")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--output", type=str, default="reach_terminal_states.pt", help="Output file path")
    parser.add_argument("--success_threshold", type=float, default=0.05, help="Distance threshold for success (m)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _unwrap_actions(action):
    """Unwrap nested action dictionaries."""
    if isinstance(action, dict):
        for key in ("actions", "action", "actions_mean", "policy"):
            if key in action:
                return _unwrap_actions(action[key])
        if len(action) == 1:
            return _unwrap_actions(next(iter(action.values())))
    if isinstance(action, (tuple, list)) and len(action) == 1:
        return _unwrap_actions(action[0])
    return action


def main() -> int:
    parser = build_parser()
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Ensure SkillBlender source is on path
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "source"))

    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    import openarm.tasks  # noqa: F401

    @hydra_task_config(args.task, args.agent)
    def _run(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
        env_cfg.scene.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # Load policy
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        # Get robot and objects
        robot = env.unwrapped.scene["robot"]

        # Check for cup/cup2 (grasp style) or object/object2 (reach style)
        if "cup" in env.unwrapped.scene.keys():
            left_obj = env.unwrapped.scene["cup"]
            obj_key = "cup"
        elif "object" in env.unwrapped.scene.keys():
            left_obj = env.unwrapped.scene["object"]
            obj_key = "object"
        else:
            left_obj = None
            obj_key = None

        if "cup2" in env.unwrapped.scene.keys():
            right_obj = env.unwrapped.scene["cup2"]
            obj2_key = "cup2"
        elif "object2" in env.unwrapped.scene.keys():
            right_obj = env.unwrapped.scene["object2"]
            obj2_key = "object2"
        else:
            right_obj = None
            obj2_key = None

        # Find EEF link indices
        left_eef_names = ["openarm_left_ee_tcp", "openarm_left_eef_link", "left_ee_link", "lj_dg_mid"]
        right_eef_names = ["openarm_right_ee_tcp", "openarm_right_eef_link", "right_ee_link", "rj_dg_mid"]

        left_eef_idx = None
        right_eef_idx = None

        for name in left_eef_names:
            if name in robot.data.body_names:
                left_eef_idx = robot.data.body_names.index(name)
                print(f"[INFO] Found left EEF: {name} at index {left_eef_idx}")
                break

        for name in right_eef_names:
            if name in robot.data.body_names:
                right_eef_idx = robot.data.body_names.index(name)
                print(f"[INFO] Found right EEF: {name} at index {right_eef_idx}")
                break

        # Storage for terminal states
        terminal_states = {
            "joint_pos": [],
            "joint_vel": [],
        }

        if left_obj is not None:
            terminal_states["object_pos"] = []
            terminal_states["object_quat"] = []

        if right_obj is not None:
            terminal_states["object2_pos"] = []
            terminal_states["object2_quat"] = []

        if left_eef_idx is not None:
            terminal_states["left_eef_pos"] = []
        if right_eef_idx is not None:
            terminal_states["right_eef_pos"] = []

        # Episode tracking
        num_envs = env.unwrapped.num_envs
        episode_count = 0
        success_count = 0

        obs = env.get_observations()
        next_report = 100

        print(f"[INFO] Collecting terminal states from {args.num_episodes} episodes...")
        print(f"[INFO] Success threshold: {args.success_threshold}m")
        print(f"[INFO] Using {num_envs} parallel environments")

        # IMPORTANT: Buffer to store states BEFORE reset happens
        # Isaac Lab resets the environment inside step() when done=True,
        # so we need to capture the state from the PREVIOUS step
        prev_joint_pos = robot.data.joint_pos.clone()
        prev_joint_vel = robot.data.joint_vel.clone()
        prev_left_obj_pos = left_obj.data.root_pos_w.clone() if left_obj is not None else None
        prev_left_obj_quat = left_obj.data.root_quat_w.clone() if left_obj is not None else None
        prev_right_obj_pos = right_obj.data.root_pos_w.clone() if right_obj is not None else None
        prev_right_obj_quat = right_obj.data.root_quat_w.clone() if right_obj is not None else None
        prev_left_eef_pos = robot.data.body_pos_w[:, left_eef_idx].clone() if left_eef_idx is not None else None
        prev_right_eef_pos = robot.data.body_pos_w[:, right_eef_idx].clone() if right_eef_idx is not None else None

        while episode_count < args.num_episodes:
            with torch.inference_mode():
                actions = _unwrap_actions(policy(obs))
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

            if torch.any(dones):
                done_ids = torch.nonzero(dones).squeeze(-1)

                for idx in done_ids.tolist():
                    if episode_count >= args.num_episodes:
                        break

                    # Check success using PREVIOUS step's state (before reset)
                    is_success = True

                    if left_eef_idx is not None and left_obj is not None:
                        left_eef_pos = prev_left_eef_pos[idx]
                        left_obj_pos = prev_left_obj_pos[idx]
                        left_dist = torch.norm(left_eef_pos - left_obj_pos)
                        is_success = is_success and (left_dist < args.success_threshold)

                    if right_eef_idx is not None and right_obj is not None:
                        right_eef_pos = prev_right_eef_pos[idx]
                        right_obj_pos = prev_right_obj_pos[idx]
                        right_dist = torch.norm(right_eef_pos - right_obj_pos)
                        is_success = is_success and (right_dist < args.success_threshold)

                    episode_count += 1

                    if is_success:
                        success_count += 1

                        # Save terminal state from PREVIOUS step (before reset)
                        terminal_states["joint_pos"].append(
                            prev_joint_pos[idx].cpu().clone()
                        )
                        terminal_states["joint_vel"].append(
                            prev_joint_vel[idx].cpu().clone()
                        )

                        if left_obj is not None:
                            # Save local position (subtract env_origin)
                            local_pos = prev_left_obj_pos[idx] - env.unwrapped.scene.env_origins[idx]
                            terminal_states["object_pos"].append(
                                local_pos.cpu().clone()
                            )
                            terminal_states["object_quat"].append(
                                prev_left_obj_quat[idx].cpu().clone()
                            )

                        if right_obj is not None:
                            # Save local position (subtract env_origin)
                            local_pos2 = prev_right_obj_pos[idx] - env.unwrapped.scene.env_origins[idx]
                            terminal_states["object2_pos"].append(
                                local_pos2.cpu().clone()
                            )
                            terminal_states["object2_quat"].append(
                                prev_right_obj_quat[idx].cpu().clone()
                            )

                        if left_eef_idx is not None:
                            # Save local position (subtract env_origin) for consistency
                            left_eef_local = prev_left_eef_pos[idx] - env.unwrapped.scene.env_origins[idx]
                            terminal_states["left_eef_pos"].append(
                                left_eef_local.cpu().clone()
                            )
                        if right_eef_idx is not None:
                            # Save local position (subtract env_origin) for consistency
                            right_eef_local = prev_right_eef_pos[idx] - env.unwrapped.scene.env_origins[idx]
                            terminal_states["right_eef_pos"].append(
                                right_eef_local.cpu().clone()
                            )

                    if episode_count >= next_report:
                        print(f"[INFO] Episodes: {episode_count}/{args.num_episodes}, "
                              f"Success: {success_count} ({100*success_count/episode_count:.1f}%)")
                        next_report += 100

            # Update buffers with current state for next iteration
            prev_joint_pos = robot.data.joint_pos.clone()
            prev_joint_vel = robot.data.joint_vel.clone()
            if left_obj is not None:
                prev_left_obj_pos = left_obj.data.root_pos_w.clone()
                prev_left_obj_quat = left_obj.data.root_quat_w.clone()
            if right_obj is not None:
                prev_right_obj_pos = right_obj.data.root_pos_w.clone()
                prev_right_obj_quat = right_obj.data.root_quat_w.clone()
            if left_eef_idx is not None:
                prev_left_eef_pos = robot.data.body_pos_w[:, left_eef_idx].clone()
            if right_eef_idx is not None:
                prev_right_eef_pos = robot.data.body_pos_w[:, right_eef_idx].clone()

        # Convert lists to tensors
        for key in terminal_states:
            if terminal_states[key]:
                terminal_states[key] = torch.stack(terminal_states[key])
            else:
                terminal_states[key] = torch.tensor([])

        # Save to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(terminal_states, output_path)

        print("\n" + "=" * 60)
        print("Collection Summary")
        print("=" * 60)
        print(f"  Total episodes: {episode_count}")
        print(f"  Successful episodes: {success_count}")
        print(f"  Success rate: {100*success_count/episode_count:.1f}%")
        print(f"  Saved states: {len(terminal_states['joint_pos'])}")
        print(f"  Output file: {output_path.absolute()}")
        print("=" * 60)

        env.close()

    _run()

    if hasattr(simulation_app, "close"):
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
