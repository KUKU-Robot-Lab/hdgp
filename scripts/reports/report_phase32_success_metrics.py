#!/usr/bin/env python3
"""Check Phase 3.2 success/contact metrics in 5g_grasp_right-v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report success/contact metrics.")
    parser.add_argument("--task", type=str, default="5g_grasp_right-v2")
    parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output", type=str, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _to_float(v):
    if hasattr(v, "detach"):
        return float(v.detach().float().mean().cpu().item())
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
        env.reset()
        core_env = env.unwrapped if hasattr(env, "unwrapped") else env

        keys = [
            "grasp_success_contact_rate",
            "grasp_success",
            "lift_success",
            "final_success",
            "table_contact_penalty",
            "object_impact_penalty",
            "self_collision_penalty",
            "self_collision_rate",
            "tip_object_contact_rate",
            "tip_table_contact_rate",
            "palm_object_contact_rate",
            "palm_table_contact_rate",
        ]
        tracked = {k: [] for k in keys}

        for _ in range(args.steps):
            zero_action = torch.zeros((core_env.num_envs, core_env.cfg.num_actions), device=core_env.device)
            env.step(zero_action)
            extras = getattr(core_env, "extras", {})
            for k in keys:
                if k in extras:
                    tracked[k].append(_to_float(extras[k]))

        summary = {
            "task": args.task,
            "num_envs": int(core_env.num_envs),
            "steps": int(args.steps),
            "metrics_present": {k: bool(len(v) > 0) for k, v in tracked.items()},
            "metrics_last": {k: (v[-1] if v else None) for k, v in tracked.items()},
            "metrics_mean": {k: (sum(v) / len(v) if v else None) for k, v in tracked.items()},
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
