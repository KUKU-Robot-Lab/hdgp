# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""**동역학** 도달 실험 — 정책 없이 palm 지령만 주고 실제로 갈 수 있는지 잰다.

질문(사용자 08.27): "제자리에 멈추는 것까지는 되는데 이송을 못한다. 학습을 멈추고
  실험으로 원인을 파악하자."

제약 IK 프로브는 **기구학**만 본다(해가 존재하는가). 여기서는 실제 PD·fabric 으로
지령을 주고 **정착 오차**를 재서, 그 자세를 실제로 유지할 수 있는지 본다.
정책·체크포인트를 쓰지 않으므로 학습 상태와 무관하다.

측정: 목표 지령 → 정착 palm 위치 오차 · 관절별 추종오차 · 포화 관절.
액추에이터 조건은 인자로 바꾼다: --wrist_effort 50 --arm_kp 200
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-grip_l_grasp_sensor_fab")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--settle", type=int, default=200, help="각 목표에서 정착 대기 스텝")
parser.add_argument("--wrist_effort", type=float, default=-1.0)
parser.add_argument("--arm_effort", type=float, default=-1.0)
parser.add_argument("--arm_kp", type=float, default=-1.0)
parser.add_argument("--arm_kd", type=float, default=-1.0)
parser.add_argument("--right_style", action="store_true",
                    help="오른팔 실물 캘리브 세팅을 왼팔에 적용"
                         " (kp 300/100/50/25 · kd 45/20/15/15 · effort 300)")
parser.add_argument("--targets", type=str,
                    default="0.39,0.24,0.50;0.39,0.24,0.45;0.39,0.24,0.55;"
                            "0.34,0.16,0.50;0.44,0.32,0.50;0.39,0.24,0.35",
                    help="palm 지령 목표들 (env-local, 세미콜론 구분)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym                    # noqa: E402
import torch                               # noqa: E402
import openarm.tasks                       # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P   # noqa: E402

cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
act_cfg = cfg.scene.robot.actuators["left_arm"]
if args.wrist_effort > 0 or args.arm_effort > 0:
    eff = dict(P.ARM_EFFORT_LIMIT)
    if args.arm_effort > 0:
        eff["l_aj_[1-2]"] = args.arm_effort
        eff["l_aj_[3-4]"] = args.arm_effort
    if args.wrist_effort > 0:
        eff["l_aj_[5-7]"] = args.wrist_effort
    act_cfg.effort_limit_sim = eff
if args.right_style:
    # ★오른팔(r_aj_*)은 07.29 실물 캘리브로 kp 테이퍼 + effort 300 을 쓴다.
    #   왼팔은 effort 를 아예 적지 않아 URDF 기본값(40/27/7)이 조용히 쓰이고 있었다.
    act_cfg.stiffness = {"l_aj_[1-4]": 300.0, "l_aj_5": 100.0,
                         "l_aj_6": 50.0, "l_aj_7": 25.0}
    act_cfg.damping = {"l_aj_[1-4]": 45.0, "l_aj_5": 20.0,
                       "l_aj_6": 15.0, "l_aj_7": 15.0}
    act_cfg.effort_limit_sim = 300.0
if args.arm_kp > 0:
    act_cfg.stiffness = args.arm_kp
if args.arm_kd > 0:
    act_cfg.damping = args.arm_kd
cfg.episode_length_s = 1e9        # 정착을 봐야 하므로 리셋을 막는다(정책이 없어 무해)

env = gym.make(args.task, cfg=cfg).unwrapped
robot = env.scene["robot"]
ee = env.scene["ee_frame"]
origins = env.scene.env_origins
env.reset()

at = env.action_manager.get_term("arm_action")
aid = at._arm_joint_ids
center, half = at._box_center, at._box_half
adim = env.action_manager.total_action_dim
# ★포화 기준은 **현재 설정**에서 파생시킨다. 하드코딩하면 effort 를 바꿔도
#   표시가 안 바뀌어 "포화율 그대로"라는 오독을 만든다(실제로 그랬다).
def _per_joint(v, default):
    if isinstance(v, dict):
        out = []
        for j in range(1, 8):
            hit = default
            for k, val in v.items():
                import re as _re
                if _re.fullmatch(k.replace("l_aj_", "l_aj_"), f"l_aj_{j}") or \
                   _re.fullmatch(k, f"l_aj_{j}"):
                    hit = val
            out.append(hit)
        return out
    return [float(v)] * 7

_kp = _per_joint(act_cfg.stiffness, 400.0)
_ef = _per_joint(act_cfg.effort_limit_sim, 40.0)
lim = [_ef[j] / _kp[j] for j in range(7)]

print("\n" + "=" * 92)
print("동역학 도달 실험 — 정책 없이 palm 지령만 준다")
print(f"  effort={act_cfg.effort_limit_sim}  kp={act_cfg.stiffness}  kd={act_cfg.damping}")
print(f"  박스 x{P.PALM_BOX_X} y{P.PALM_BOX_Y} z{P.PALM_BOX_Z}")
print("=" * 92)
print(f"  {'목표 지령':<22}{'정착 palm':<24}{'오차':>9}{'포화 관절(대비)':>28}")

for spec in args.targets.split(";"):
    tx, ty, tz = (float(v) for v in spec.split(","))
    tgt = torch.tensor([tx, ty, tz], device=env.device)
    a_pos = ((tgt - center) / half).clamp(-1.0, 1.0)
    act = torch.zeros(env.num_envs, adim, device=env.device)
    act[:, :3] = a_pos
    act[:, -1] = -1.0                       # 그리퍼 닫힘 유지
    for _ in range(args.settle):
        env.step(act)
    tcp = (ee.data.target_pos_w[:, 0, :] - origins).mean(0)
    err = (robot.data.joint_pos_target[:, aid] - robot.data.joint_pos[:, aid]).abs().mean(0)
    cmd = at._palm_pose_target[:, :3].mean(0)
    # palm 지령이 실제로 목표가 됐는지도 함께 본다(클램프·리미터 확인)
    d = float((tcp - tgt).norm()) * 1e3
    hot = [f"j{j+1} {float(err[j])/lim[j]:.0%}" for j in range(7) if float(err[j]) > lim[j]]
    print(f"  [{tx:.2f},{ty:.2f},{tz:.2f}] cmd[{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f}]"
          f"  tcp[{tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}]{d:8.0f}mm"
          f"   {' '.join(hot) if hot else '-'}")

print("\n  → 오차가 작으면 그 자세는 **실제로 유지 가능**하다(정책·보상 문제).")
print("     오차가 크고 특정 관절이 포화하면 그 관절이 물리 병목이다.")
print("  ⚠ cmd 가 목표와 다르면 박스 clamp 또는 리미터가 막은 것이다.")
env.close()
app.close()
