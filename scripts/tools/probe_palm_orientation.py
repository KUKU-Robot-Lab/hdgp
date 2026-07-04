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
    p.add_argument("--reset_damping", type=float, default=-1.0, help=">0 이면 리셋 fabric cspace damping 오버라이드")
    p.add_argument("--dt", type=float, default=-1.0, help=">0 이면 fabric timestep 오버라이드")
    p.add_argument("--offx", type=float, default=None, help="pregrasp_offset_x 오버라이드")
    p.add_argument("--offy", type=float, default=None, help="pregrasp_offset_y 오버라이드")
    p.add_argument("--offz", type=float, default=None, help="pregrasp_offset_z 오버라이드")
    p.add_argument("--laj1", type=float, default=None, help="왼팔 l_aj_1 오버라이드 (left palm y 튜닝)")
    p.add_argument("--laj_idx", type=int, default=0, help="오버라이드할 왼팔 관절 index (0=l_aj_1)")
    p.add_argument("--raj7_bias", type=float, default=None, help="pregrasp cache 의 r_aj_7 을 이만큼 빼서 palm 을 낮춤")
    p.add_argument("--thumb1", type=float, default=None, help="approach thumb_1(abduction) 오버라이드 (컵 clearance 튜닝)")
    p.add_argument("--grip_steps", type=int, default=0, help=">0 이면 손가락 닫는 액션으로 N step 진행 후 wrap 측정")
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
        if args.offx is not None:
            env_cfg.pregrasp_offset_x = args.offx
        if args.offy is not None:
            env_cfg.pregrasp_offset_y = args.offy
        if args.offz is not None:
            env_cfg.pregrasp_offset_z = args.offz

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        core = env.unwrapped if hasattr(env, "unwrapped") else env
        # 캐시는 첫 reset 에서 빌드되므로, reset 전에 damping/dt 를 덮어써야 반영된다.
        if args.reset_damping > 0 and hasattr(core, "_reset_damping"):
            core._reset_damping.fill_(args.reset_damping)
        if args.dt > 0 and hasattr(core, "timestep"):
            core.timestep = args.dt
        if args.laj1 is not None and hasattr(core, "left_arm_zero_pos"):
            core.left_arm_zero_pos[:, args.laj_idx] = args.laj1
        if args.thumb1 is not None and hasattr(core, "hand_approach_pose"):
            core.hand_approach_pose[0] = args.thumb1
        env.reset()
        core0 = env.unwrapped if hasattr(env, "unwrapped") else env
        # r_aj_7 bias: 첫 reset 에서 캐시가 빌드된 뒤, 캐시의 r_aj_7(arm index 6)을 낮추고 재리셋.
        if args.raj7_bias is not None and hasattr(core0, "_cache_q_arm"):
            core0._cache_q_arm[..., 6] -= args.raj7_bias
            env.reset()

        zero = torch.zeros((core.num_envs, core.cfg.num_actions), device=core.device)
        for _ in range(max(1, args.steps)):
            env.step(zero)

        # 손가락 닫는 액션(palm 유지, finger idx6:12 = +1)으로 진행 → grasp phase 진입 후 wrap 측정
        if args.grip_steps > 0:
            grip = torch.zeros((core.num_envs, core.cfg.num_actions), device=core.device)
            grip[:, 6:core.cfg.num_actions] = 1.0
            for _ in range(args.grip_steps):
                env.step(grip)

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

        # wrap 측정 링크: 5 손끝 + 근위(envelope signature) + 엄지 중간마디.
        wrap_links = (
            "r_hl_thumb_4", "r_hl_index_2", "r_hl_middle_2", "r_hl_ring_2", "r_hl_pinky_2",
            "r_hl_index_1", "r_hl_middle_1", "r_hl_ring_1", "r_hl_pinky_1", "r_hl_thumb_3",
        )
        thumb_pos = {}
        for tn in wrap_links:
            if tn in robot.data.body_names:
                ti = robot.data.body_names.index(tn)
                thumb_pos[tn] = robot.data.body_pos_w[:, ti] - origins

        # 왼손 palm_sensor 위치(있으면)
        left_palm_pos = None
        if "l_hl_palm_sensor" in robot.data.body_names:
            lidx = robot.data.body_names.index("l_hl_palm_sensor")
            left_palm_pos = robot.data.body_pos_w[:, lidx] - origins

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
            if left_palm_pos is not None:
                lp = [float(v) for v in left_palm_pos[i].tolist()]
                print(f"  LEFT palm_sensor pos: [{lp[0]:+.3f} {lp[1]:+.3f} {lp[2]:+.3f}]")
            # wrap 분류: 컵축(xy) 거리 vs 반경0.035, 링크두께~0.01 감안.
            # 컵 z 범위(0.205~0.345) 밖이면 '높이밖'. dxy<0.03=관통, 0.03~0.05=감쌈(접촉), >0.05=벌어짐.
            cup_r = 0.035
            for tn, tp in thumb_pos.items():
                t = [float(v) for v in tp[i].tolist()]
                dxy = math.sqrt((t[0] - cp[0]) ** 2 + (t[1] - cp[1]) ** 2)
                in_z = 0.205 < t[2] < 0.345
                if not in_z:
                    cls = "높이밖"
                elif dxy < 0.03:
                    cls = "◄관통"
                elif dxy <= 0.05:
                    cls = "◄감쌈(접촉)"
                else:
                    cls = "벌어짐"
                print(f"  {tn}: [{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}] 컵축거리={dxy:.3f} {cls}")
        # 오른팔 관절 포화 확인 (env 0)
        arm_idx = core.arm_dof_indices
        jp = robot.data.joint_pos[0]
        jl = robot.data.joint_limits[0] if hasattr(robot.data, "joint_limits") else robot.data.soft_joint_pos_limits[0]
        print("  [env0] 오른팔 관절 (값 / [min,max] / 포화%):")
        for k, ai in enumerate(arm_idx):
            v = float(jp[ai]); lo = float(jl[ai, 0]); hi = float(jl[ai, 1])
            frac = (v - lo) / (hi - lo + 1e-9)
            sat = "◄SAT" if (frac < 0.05 or frac > 0.95) else ""
            print(f"    {robot.joint_names[ai]:20s} {v:+.3f} / [{lo:+.2f},{hi:+.2f}] / {frac*100:5.1f}% {sat}")
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
