#!/usr/bin/env python3
"""Phase B3 rollout validation for 5g_grasp_right-v2.

Checks:
- Observation shape reflects Phase B expansion (expected 255).
- Reference and object-pc metrics are present in `extras`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Phase B metrics with short rollout.")
    parser.add_argument("--task", type=str, default="5g_grasp_right-v2")
    parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output", type=str, default=None, help="Optional output json path.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _to_float(v):
    if hasattr(v, "detach"):
        t = v.detach().float().mean()
        return float(t.cpu().item())
    return float(v)


def main() -> int:
    parser = build_parser()
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "source"))

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.hydra import hydra_task_config
    import openarm.tasks  # noqa: F401

    @hydra_task_config(args.task, args.agent)
    def _run(env_cfg, _agent_cfg):
        if hasattr(env_cfg.scene, "num_envs"):
            env_cfg.scene.num_envs = args.num_envs

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        obs, _ = env.reset()
        core_env = env.unwrapped if hasattr(env, "unwrapped") else env

        if isinstance(obs, dict) and "policy" in obs:
            policy_obs = obs["policy"]
        else:
            policy_obs = obs

        obs_dim = int(policy_obs.shape[-1])
        expected_obs = int(core_env.cfg.observation_space)

        tracked_keys = [
            "reference_palm_tracking_error",
            "reference_pca_saturation_ratio",
            "reference_palm_saturation_ratio",
            "object_pc_feature_norm",
            "object_pc_clip_ratio",
            "object_pc_invalid_ratio",
        ]
        tracked = {k: [] for k in tracked_keys}

        for _ in range(args.steps):
            zero_action = torch.zeros((core_env.num_envs, core_env.cfg.num_actions), device=core_env.device)
            env.step(zero_action)
            extras = getattr(core_env, "extras", {})
            for k in tracked_keys:
                if k in extras:
                    tracked[k].append(_to_float(extras[k]))

        summary = {
            "task": args.task,
            "num_envs": int(core_env.num_envs),
            "steps": int(args.steps),
            "obs_dim": obs_dim,
            "expected_obs_dim": expected_obs,
            "obs_dim_match": bool(obs_dim == expected_obs),
            "tracked_metrics_present": {k: bool(len(v) > 0) for k, v in tracked.items()},
            "tracked_metrics_last": {k: (v[-1] if v else None) for k, v in tracked.items()},
            "tracked_metrics_mean": {k: (sum(v) / len(v) if v else None) for k, v in tracked.items()},
        }

        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if args.output:
            out = Path(args.output).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[INFO] Saved report: {out}")

        env.close()

    try:
        _run()
    finally:
        if hasattr(simulation_app, "close"):
            simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

