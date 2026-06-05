#!/usr/bin/env python3
"""Replay action sequences in Isaac Lab env and record policy observations.

This script fills step-2 for BC dataset building:
- input: npz containing `actions` [T, act_dim]
- output: npz containing `observations` [K, obs_dim] + `actions` [K, act_dim]

Notes:
- Uses current environment observation pipeline (exact policy input).
- Stops early if episode terminates unless `--reset-on-done` is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect grasp policy observations by replaying actions.")
    parser.add_argument("--task", type=str, default="5g_grasp_left-v1")
    parser.add_argument("--input-actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--reset-on-done", action="store_true", default=False)
    parser.add_argument("--disable-obs-noise", action="store_true", default=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def _extract_policy_obs(obs):
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        # fallback: first value
        return next(iter(obs.values()))
    return obs


def main() -> None:
    data = np.load(args_cli.input_actions, allow_pickle=True)
    if "actions" not in data:
        raise KeyError("Input dataset must contain 'actions'")
    actions_np = data["actions"].astype(np.float32)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if getattr(env_cfg, "observations", None) is not None and getattr(env_cfg.observations, "policy", None) is not None:
        env_cfg.observations.policy.concatenate_terms = True
        if args_cli.disable_obs_noise:
            env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    obs, _ = env.reset()
    policy_obs = _extract_policy_obs(obs)

    obs_buffer = []
    act_buffer = []

    with torch.inference_mode():
        for t in range(actions_np.shape[0]):
            obs_t = policy_obs[0].detach().cpu().numpy()
            act_t = actions_np[t]

            action_tensor = torch.from_numpy(act_t).to(env.device).unsqueeze(0)
            obs_buffer.append(obs_t)
            act_buffer.append(act_t)

            obs, _, terminated, truncated, _ = env.step(action_tensor)
            policy_obs = _extract_policy_obs(obs)

            done = bool((terminated | truncated).any().item()) if isinstance(terminated, torch.Tensor) else bool(terminated)
            if done:
                if args_cli.reset_on_done:
                    obs, _ = env.reset()
                    policy_obs = _extract_policy_obs(obs)
                else:
                    print(f"[INFO] Episode terminated at step {t}. Stopping collection.")
                    break

    env.close()

    obs_arr = np.asarray(obs_buffer, dtype=np.float32)
    act_arr = np.asarray(act_buffer, dtype=np.float32)

    meta = {
        "task": args_cli.task,
        "input_actions": str(args_cli.input_actions),
        "num_steps_collected": int(len(obs_arr)),
        "obs_dim": int(obs_arr.shape[1]) if obs_arr.ndim == 2 else None,
        "action_dim": int(act_arr.shape[1]) if act_arr.ndim == 2 else None,
        "reset_on_done": bool(args_cli.reset_on_done),
        "disable_obs_noise": bool(args_cli.disable_obs_noise),
    }

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args_cli.output,
        observations=obs_arr,
        actions=act_arr,
        meta_json=json.dumps(meta),
    )

    print(f"Saved: {args_cli.output}")
    print(f"observations={obs_arr.shape}, actions={act_arr.shape}")


if __name__ == "__main__":
    main()
    simulation_app.close()
