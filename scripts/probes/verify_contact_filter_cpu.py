#!/usr/bin/env python3
"""Verify object/table contact-filter separation on CPU for 5g_grasp_right-v2.

Scenarios (env0):
1) baseline: no forced contact
2) object_contact_forced: cup moved to fingertip, table moved away
3) table_contact_forced: table moved near fingertip, cup moved away
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify contact filter behavior on CPU.")
    parser.add_argument("--task", type=str, default="5g_grasp_right-v2")
    parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--force_sim_cpu", action="store_true", default=True)
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
        if args.force_sim_cpu and hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
            env_cfg.sim.device = "cpu"

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        core_env = env.unwrapped if hasattr(env, "unwrapped") else env
        device = core_env.device

        def step_n(n: int):
            for _ in range(n):
                zero_action = torch.zeros((core_env.num_envs, core_env.cfg.num_actions), device=device)
                env.step(zero_action)

        def read_metrics(tag: str):
            extras = getattr(core_env, "extras", {})
            return {
                "tag": tag,
                "tip_object_contact_rate": _to_float(extras.get("tip_object_contact_rate", 0.0)),
                "tip_table_contact_rate": _to_float(extras.get("tip_table_contact_rate", 0.0)),
                "tip_object_contact_force": _to_float(extras.get("tip_object_contact_force", 0.0)),
                "tip_table_contact_force": _to_float(extras.get("tip_table_contact_force", 0.0)),
                "tip_contact_force": _to_float(extras.get("tip_contact_force", 0.0)),
            }

        def write_pose(obj, pos_w: torch.Tensor, write_vel: bool = True):
            pose = torch.zeros((1, 7), device=device)
            pose[:, :3] = pos_w.view(1, 3)
            pose[:, 3] = 1.0  # wxyz identity
            obj.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=device, dtype=torch.long))
            if write_vel:
                vel = torch.zeros((1, 6), device=device)
                obj.write_root_velocity_to_sim(vel, env_ids=torch.tensor([0], device=device, dtype=torch.long))

        env.reset()
        step_n(4)

        results = []
        results.append(read_metrics("baseline"))

        # Use physics body position of first distal link as contact anchor.
        robot = core_env.scene["robot"]
        link_name = core_env.cfg.right_tip_contact_links[0]
        if link_name not in robot.data.body_names:
            raise RuntimeError(f"Missing contact anchor body: {link_name}")
        link_idx = robot.data.body_names.index(link_name)
        tip0_w = robot.data.body_pos_w[0, link_idx]

        # Scenario A: force object contact
        if core_env.cup is not None:
            write_pose(core_env.cup, tip0_w)
        # Move table away to reduce accidental table contact
        if core_env.table is not None:
            write_pose(core_env.table, torch.tensor([0.5725, 0.003, -2.0], device=device), write_vel=False)
        step_n(args.steps)
        results.append(read_metrics("object_contact_forced"))

        # Scenario B: force table contact
        if core_env.cup is not None:
            write_pose(core_env.cup, torch.tensor([0.2, 0.2, 1.5], device=device))
        if core_env.table is not None:
            # Bring table root near fingertip so table volume intersects hand links.
            write_pose(core_env.table, tip0_w, write_vel=False)
        step_n(args.steps)
        results.append(read_metrics("table_contact_forced"))

        # Sensor tensor availability
        sensor_diag = {
            "num_tip_object_sensors": len(getattr(core_env, "_tip_object_sensors", [])),
            "num_tip_table_sensors": len(getattr(core_env, "_tip_table_sensors", [])),
            "object_force_matrix_available": [],
            "table_force_matrix_available": [],
            "object_force_matrix_shapes": [],
            "table_force_matrix_shapes": [],
            "object_net_force_shapes": [],
            "table_net_force_shapes": [],
        }
        for s in getattr(core_env, "_tip_object_sensors", []):
            fm = getattr(s.data, "force_matrix_w", None)
            sensor_diag["object_force_matrix_available"].append(fm is not None)
            sensor_diag["object_force_matrix_shapes"].append(list(fm.shape) if fm is not None else None)
            sensor_diag["object_net_force_shapes"].append(list(s.data.net_forces_w.shape))
        for s in getattr(core_env, "_tip_table_sensors", []):
            fm = getattr(s.data, "force_matrix_w", None)
            sensor_diag["table_force_matrix_available"].append(fm is not None)
            sensor_diag["table_force_matrix_shapes"].append(list(fm.shape) if fm is not None else None)
            sensor_diag["table_net_force_shapes"].append(list(s.data.net_forces_w.shape))

        summary = {
            "task": args.task,
            "device": str(device),
            "results": results,
            "sensor_diag": sensor_diag,
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
