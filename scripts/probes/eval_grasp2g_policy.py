#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Evaluate grasp2g policy with left/right success metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate grasp2g policy success metrics.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--goal_threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _unwrap_actions(action):
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
    # clear out sys.argv for Hydra
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ensure SkillBlender source is on path
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "source"))

    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    import openarm.tasks  # noqa: F401
    from openarm.tasks.manager_based.openarm_manipulation.pipeline.gripper.both.grasp_2g import mdp as grasp_mdp

    @hydra_task_config(args.task, args.agent)
    def _run(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
        env_cfg.scene.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # load policy
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        # reward params for thresholds
        rewards_cfg = env.unwrapped.cfg.rewards
        lift_height = rewards_cfg.left_lifting_object.params.get("lift_height", 0.1)
        goal_params = rewards_cfg.left_object_goal_tracking.params
        hold_duration = goal_params["phase_params"].get("hold_duration", 2.0)
        minimal_height = goal_params.get("minimal_height", 0.04)

        left_obj_cfg = SceneEntityCfg("object")
        right_obj_cfg = SceneEntityCfg("object2")
        left_ee_cfg = SceneEntityCfg("left_ee_frame")
        right_ee_cfg = SceneEntityCfg("right_ee_frame")

        # episode trackers
        num_envs = env.unwrapped.num_envs
        episode_count = 0
        lift_left = []
        lift_right = []
        hold_left = []
        hold_right = []
        goal_left = []
        goal_right = []

        # per-env state
        lift_flag_l = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)
        lift_flag_r = torch.zeros_like(lift_flag_l)
        hold_flag_l = torch.zeros_like(lift_flag_l)
        hold_flag_r = torch.zeros_like(lift_flag_l)
        goal_flag_l = torch.zeros_like(lift_flag_l)
        goal_flag_r = torch.zeros_like(lift_flag_l)
        hold_time_l = torch.zeros(num_envs, device=env.unwrapped.device)
        hold_time_r = torch.zeros_like(hold_time_l)

        obs = env.get_observations()
        next_report = 10
        while episode_count < args.num_episodes:
            with torch.inference_mode():
                actions = _unwrap_actions(policy(obs))
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

                # compute lift
                lift_l = grasp_mdp.object_is_lifted(env.unwrapped, lift_height, left_obj_cfg) > 0.5
                lift_r = grasp_mdp.object_is_lifted(env.unwrapped, lift_height, right_obj_cfg) > 0.5

                # goal tracking success
                goal_l = (
                    grasp_mdp.object_goal_distance_with_ee(
                        env.unwrapped,
                        std=goal_params["std"],
                        minimal_height=minimal_height,
                        command_name=goal_params["command_name"],
                        object_cfg=left_obj_cfg,
                        ee_frame_cfg=left_ee_cfg,
                        reach_std=goal_params["reach_std"],
                    )
                    > args.goal_threshold
                )
                goal_r = (
                    grasp_mdp.object_goal_distance_with_ee(
                        env.unwrapped,
                        std=goal_params["std"],
                        minimal_height=minimal_height,
                        command_name=rewards_cfg.right_object_goal_tracking.params["command_name"],
                        object_cfg=right_obj_cfg,
                        ee_frame_cfg=right_ee_cfg,
                        reach_std=goal_params["reach_std"],
                    )
                    > args.goal_threshold
                )

            # update trackers outside inference mode
            lift_flag_l |= lift_l
            lift_flag_r |= lift_r

            hold_time_l = torch.where(lift_l, hold_time_l + env.unwrapped.step_dt, torch.zeros_like(hold_time_l))
            hold_time_r = torch.where(lift_r, hold_time_r + env.unwrapped.step_dt, torch.zeros_like(hold_time_r))
            hold_flag_l |= hold_time_l >= hold_duration
            hold_flag_r |= hold_time_r >= hold_duration

            goal_flag_l |= goal_l
            goal_flag_r |= goal_r

            if torch.any(dones):
                done_ids = torch.nonzero(dones).squeeze(-1)
                for idx in done_ids.tolist():
                    if episode_count >= args.num_episodes:
                        break
                    lift_left.append(bool(lift_flag_l[idx].item()))
                    lift_right.append(bool(lift_flag_r[idx].item()))
                    hold_left.append(bool(hold_flag_l[idx].item()))
                    hold_right.append(bool(hold_flag_r[idx].item()))
                    goal_left.append(bool(goal_flag_l[idx].item()))
                    goal_right.append(bool(goal_flag_r[idx].item()))
                    episode_count += 1
                    if episode_count >= next_report:
                        print(f"[INFO] Completed {episode_count}/{args.num_episodes} episodes")
                        next_report += 10

                # reset per-env trackers for done envs
                lift_flag_l[done_ids] = False
                lift_flag_r[done_ids] = False
                hold_flag_l[done_ids] = False
                hold_flag_r[done_ids] = False
                goal_flag_l[done_ids] = False
                goal_flag_r[done_ids] = False
                hold_time_l[done_ids] = 0.0
                hold_time_r[done_ids] = 0.0

        def _mean(x):
            return sum(x) / len(x) if x else 0.0

        print("Evaluation summary (episodes={}):".format(args.num_episodes))
        print(f"  Lift success   L/R: {_mean(lift_left):.3f} / {_mean(lift_right):.3f}")
        print(f"  Hold success   L/R: {_mean(hold_left):.3f} / {_mean(hold_right):.3f}")
        print(f"  Goal success   L/R: {_mean(goal_left):.3f} / {_mean(goal_right):.3f}")
        print(f"  Goal threshold: {args.goal_threshold:.2f}")
        print(f"  Lift height: {lift_height:.3f}, Hold duration: {hold_duration:.2f}s, Min height: {minimal_height:.3f}")

        env.close()

    _run()
    if hasattr(simulation_app, "close"):
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
