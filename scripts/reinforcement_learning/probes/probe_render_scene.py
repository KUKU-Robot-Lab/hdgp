"""씬 렌더 probe — 환경 세팅을 눈으로 확인한다.

여러 각도에서 프레임을 뽑아 PNG 로 저장한다. 학습된 정책이 없어도 되며,
zero-action(=홈 유지)으로 세워 둔 상태를 찍는다.

사용:
    isaaclab.sh -p scripts/reinforcement_learning/probes/probe_render_scene.py \
        --task open-bis_r_grasp_lift_fab --enable_cameras --out /tmp/scene
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--out", default="/tmp/scene")
parser.add_argument("--settle", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app = AppLauncher(args).app

import gymnasium as gym          # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks             # noqa: E402,F401

# 카메라 각도들 (env-local 좌표)
VIEWS = {
    "01_overview":  ((1.30, -1.10, 0.95), (0.25, -0.15, 0.25)),
    "02_front":     ((1.40,  0.00, 0.55), (0.10,  0.00, 0.20)),
    "03_side":      ((0.30, -1.30, 0.55), (0.30, -0.15, 0.25)),
    "04_top":       ((0.35, -0.18, 1.40), (0.35, -0.18, 0.20)),
    "05_workspace": ((0.85, -0.75, 0.62), (0.28, -0.22, 0.30)),
}

os.makedirs(args.out, exist_ok=True)
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env_cfg.viewer.origin_type = "env"
env_cfg.viewer.env_index = 0
env_cfg.viewer.resolution = (1600, 1000)

first = next(iter(VIEWS.values()))
env_cfg.viewer.eye, env_cfg.viewer.lookat = first

env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
env.reset()
zero = torch.zeros(1, env.unwrapped.cfg.action_space, device=env.unwrapped.device)
for _ in range(args.settle):
    env.step(zero)

import cv2                       # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

sim = SimulationContext.instance()
# ★첫 프레임은 렌더 파이프라인 워밍업 전이라 검게 나온다(실측 밝기 0). 버린다.
for _ in range(4):
    env.step(zero)
    env.render()

for name, (eye, lookat) in VIEWS.items():
    sim.set_camera_view(eye=eye, target=lookat)
    for _ in range(6):           # 카메라 이동 반영 + 렌더 안정화
        env.step(zero)
    rgb = env.render()
    if rgb is None:
        print(f"  {name}: render() 가 None — 스킵", flush=True)
        continue
    path = os.path.join(args.out, f"{name}.png")
    cv2.imwrite(path, cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
    print(f"  저장 {path}  {np.asarray(rgb).shape}", flush=True)

env.close()
app.close()
