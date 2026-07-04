#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""리셋 후 실제 palm_sensor 자세 vs pregrasp 타깃 실측 (fabric IK 수렴 진단).

palm_sensor +z(손바닥 법선) world 방향이 타깃(+x, 컵-향 수평)과 일치하는지,
아니면 초기 아래-향 자세에 머무는지(IK 미수렴)를 수치로 확정한다.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Probe palm_sensor orientation after reset.")
    p.add_argument("--task", type=str, default="open-rh56f1_r_grasp_v1")
    p.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=1, help="reset 후 zero-action step 수")
    p.add_argument("--fabric_steps", type=int, default=-1, help=">0 이면 pregrasp_fabric_steps 오버라이드")
    p.add_argument("--no_cache", action="store_true", help="cache_pregrasp_reset 비활성(매 reset fabric rollout)")
    AppLauncher.add_app_launcher_args(p)
    return p


def _euler_zyx_z_axis(ez, ey, ex):
    """euler_to_matrix(=Rz.Ry.Rx) 의 3열(=palm +z 타깃 world)."""
    cz, sz = math.cos(ez), math.sin(ez)
    cy, sy = math.cos(ey), math.sin(ey)
    cx, sx = math.cos(ex), math.sin(ex)
    return [
        cz * sy * cx + sz * sx,
        sz * sy * cx - cz * sx,
        cy * cx,
    ]


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
    from isaaclab.utils.math import quat_apply
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import openarm.tasks  # noqa: F401

    @hydra_task_config(args.task, args.agent)
    def _run(env_cfg, _agent_cfg):
        if hasattr(env_cfg.scene, "num_envs"):
            env_cfg.scene.num_envs = args.num_envs
        if args.fabric_steps > 0:
            env_cfg.pregrasp_fabric_steps = args.fabric_steps
        if args.no_cache:
            env_cfg.cache_pregrasp_reset = False

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        env.reset()
        core = env.unwrapped if hasattr(env, "unwrapped") else env

        zero = torch.zeros((core.num_envs, core.cfg.num_actions), device=core.device)
        for _ in range(max(1, args.steps)):
            env.step(zero)

        robot = core.scene["robot"]
        origins = core.scene.env_origins
        palm_idx = core.palm_body_index
        z_local = torch.zeros(core.num_envs, 3, device=core.device)
        z_local[:, 2] = 1.0

        palm_quat = robot.data.body_quat_w[:, palm_idx]        # (N,4) wxyz
        palm_pos = robot.data.body_pos_w[:, palm_idx] - origins
        palm_z_world = quat_apply(palm_quat, z_local)          # 실제 palm +z

        cup = core.scene["cup"]
        cup_pos = cup.data.root_pos_w - origins

        tgt = core.pregrasp_palm_pose_buf                       # (N,6) pos+euler_zyx

        print("=" * 70)
        print("task:", args.task, "| num_envs:", core.num_envs)
        for i in range(core.num_envs):
            ez, ey, ex = [float(v) for v in tgt[i, 3:6].tolist()]
            tz = _euler_zyx_z_axis(ez, ey, ex)                 # 타깃 palm +z
            az = [float(v) for v in palm_z_world[i].tolist()]  # 실제 palm +z
            cp = [float(v) for v in cup_pos[i].tolist()]
            pp = [float(v) for v in palm_pos[i].tolist()]
            to_cup = [cp[0] - pp[0], cp[1] - pp[1], cp[2] - pp[2]]
            n = math.sqrt(sum(c * c for c in to_cup)) + 1e-9
            to_cup_u = [c / n for c in to_cup]
            dot_down = az[2]                                    # 실제 +z 와 world -z: az·(0,0,-1) = -az[2]
            dot_cup = sum(a * b for a, b in zip(az, to_cup_u))  # 실제 +z 와 컵방향 정렬
            ang_from_horiz = math.degrees(math.asin(max(-1.0, min(1.0, abs(az[2])))))
            print(f"\n[env {i}]")
            print(f"  target euler_zyx (deg): ez={math.degrees(ez):7.2f} ey={math.degrees(ey):7.2f} ex={math.degrees(ex):7.2f}")
            print(f"  target palm +z (world): [{tz[0]:+.3f} {tz[1]:+.3f} {tz[2]:+.3f}]")
            print(f"  ACTUAL palm +z (world): [{az[0]:+.3f} {az[1]:+.3f} {az[2]:+.3f}]")
            print(f"  palm_pos: [{pp[0]:+.3f} {pp[1]:+.3f} {pp[2]:+.3f}]  cup_pos: [{cp[0]:+.3f} {cp[1]:+.3f} {cp[2]:+.3f}]")
            print(f"  palm→cup dir: [{to_cup_u[0]:+.3f} {to_cup_u[1]:+.3f} {to_cup_u[2]:+.3f}]  dist={n:.3f}")
            print(f"  actual +z · (down) = {-dot_down:+.3f}  (|+1|=바닥향)")
            print(f"  actual +z · (컵방향) = {dot_cup:+.3f}  (|+1|=컵을 정확히 향함)")
            print(f"  actual +z 의 수평면 대비 각 = {ang_from_horiz:5.1f}° (0=완전수평, 90=수직)")
        print("=" * 70)
        env.close()

    try:
        _run()
    finally:
        if hasattr(simulation_app, "close"):
            simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
