#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Reach -> Grasp -> Transfer -> Pour policies sequentially.

This script loads each policy checkpoint and runs it for a fixed number of steps.
At stage boundaries, it copies the simulator state (robot joints + object/bead roots)
into the next task environment so the sequence continues in the same physical state.

Example:
  ./isaaclab.sh -p ../hdgp/scripts/probes/pipeline_play.py \
    --reach-ckpt /path/to/reach.pt \
    --grasp-ckpt /path/to/grasp.pt \
    --transfer-ckpt /path/to/transfer.pt \
    --pour-ckpt /path/to/pour.pt \
    --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sequential skill pipeline (Reach->Grasp->Transfer->Pour).")
    parser.add_argument("--reach-task", type=str, default="ReachIK-v0")
    parser.add_argument("--grasp-task", type=str, default="GraspIK-v0")
    parser.add_argument("--transfer-task", type=str, default="TransferIK-v0")
    parser.add_argument("--pour-task", type=str, default="PourIK-v0")
    parser.add_argument("--agent", type=str, default="rsl_rl_dual_cfg_entry_point")
    parser.add_argument("--reach-ckpt", type=str, required=True)
    parser.add_argument("--grasp-ckpt", type=str, required=True)
    parser.add_argument("--transfer-ckpt", type=str, required=True)
    parser.add_argument("--pour-ckpt", type=str, required=True)
    parser.add_argument("--reach-steps", type=int, default=600)
    parser.add_argument("--grasp-steps", type=int, default=600)
    parser.add_argument("--transfer-steps", type=int, default=600)
    parser.add_argument("--pour-steps", type=int, default=600)
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


def _capture_state(env):
    scene = env.scene
    robot = scene["robot"]

    state = {
        "joint_pos": robot.data.joint_pos.clone(),
        "joint_vel": robot.data.joint_vel.clone(),
        "root_pos": robot.data.root_pos_w.clone(),
        "root_quat": robot.data.root_quat_w.clone(),
    }

    for name in ("object", "object2", "bead"):
        if name in scene:
            obj = scene[name]
            state[f"{name}_root_pos"] = obj.data.root_pos_w.clone()
            state[f"{name}_root_quat"] = obj.data.root_quat_w.clone()
            state[f"{name}_root_lin_vel"] = obj.data.root_lin_vel_w.clone()
            state[f"{name}_root_ang_vel"] = obj.data.root_ang_vel_w.clone()

    return state


def _apply_state(env, state):
    import torch

    scene = env.scene
    robot = scene["robot"]
    num_envs = env.num_envs
    device = env.device
    env_ids = None

    # robot joints
    robot.write_joint_state_to_sim(state["joint_pos"], state["joint_vel"], env_ids=env_ids)

    # robot root (if floating base in future)
    if "root_pos" in state and "root_quat" in state:
        root_pose = state["root_pos"].clone()
        root_quat = state["root_quat"].clone()
        if root_pose.shape[0] == num_envs:
            robot.write_root_pose_to_sim(torch.cat([root_pose, root_quat], dim=-1), env_ids=env_ids)

    # objects / bead
    for name in ("object", "object2", "bead"):
        key = f"{name}_root_pos"
        if key in state and name in scene:
            pos = state[f"{name}_root_pos"]
            quat = state[f"{name}_root_quat"]
            lin = state[f"{name}_root_lin_vel"]
            ang = state[f"{name}_root_ang_vel"]
            root_state = torch.cat([pos, quat, lin, ang], dim=-1)
            scene[name].write_root_state_to_sim(root_state, env_ids=env_ids)


def _run_stage(task_name, checkpoint, agent_entry, steps, prev_state=None, seed=None):
    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    @hydra_task_config(task_name, agent_entry)
    def _inner(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
        env_cfg.scene.num_envs = 1
        if seed is not None:
            env_cfg.seed = seed
        # disable corruption for play
        if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.enable_corruption = False

        env = gym.make(task_name, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        if prev_state is not None:
            _apply_state(env.unwrapped, prev_state)

        obs = env.get_observations()
        for _ in range(steps):
            with torch.inference_mode():
                actions = _unwrap_actions(policy(obs))
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

        return _capture_state(env.unwrapped)

    return _inner()


def main() -> int:
    parser = build_parser()
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args)
    _ = app_launcher.app

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "source"))

    import openarm.tasks  # noqa: F401

    # Stage 1: Reach
    state = _run_stage(args.reach_task, args.reach_ckpt, args.agent, args.reach_steps, None, args.seed)
    # Stage 2: Grasp
    state = _run_stage(args.grasp_task, args.grasp_ckpt, args.agent, args.grasp_steps, state, args.seed)
    # Stage 3: Transfer
    state = _run_stage(args.transfer_task, args.transfer_ckpt, args.agent, args.transfer_steps, state, args.seed)
    # Stage 4: Pour
    _run_stage(args.pour_task, args.pour_ckpt, args.agent, args.pour_steps, state, args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
