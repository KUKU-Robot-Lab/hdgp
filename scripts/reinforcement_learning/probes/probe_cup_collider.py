"""컵 콜라이더가 **SDF(속 빈)** 인가 **convexHull(속 찬)** 인가 — 결정적 판정.

메모리 함정: visdex 컵은 `physics:approximation="sdf"` 를 적어놓고도 apiSchemas 에
PhysxSDFMeshCollisionAPI 가 없어 PhysX 가 **convexHull 로 폴백**한 이력이 있다.
cup_big_rl 은 sdfResolution=64 가 있지만 apiSchemas 에는 여전히 그 API 가 없다.

판정법: 작은 구를 컵 **공동 안**(축 위, 림 아래)에 떨어뜨린다.
  · SDF(속 빈)  → 구가 컵 바닥까지 내려간다
  · convexHull  → 구가 컵 윗면에 얹힌다

리프트가 안 되는 원인 후보다 — 속이 찼다면 손가락이 '표면 안쪽'에 있다는 건
진짜 관통이고, 그 접촉은 마찰이 아니라 밀어내기라 물체를 들 수 없다.

    isaaclab.sh -p .../probe_cup_collider.py
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--steps", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks         # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=4)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()
zero = torch.zeros(4, env.cfg.action_space, device=env.device)
for _ in range(10):
    env.step(zero)

obj = env.object.data.root_pos_w - env.scene.env_origins
spec = env.bank.specs[0]; sc = float(spec.scale[2])
OFF, H = 0.0773 * sc, 0.1776 * sc
top_z = obj[0, 2].item() - OFF + H            # 컵 상단(림)
bot_z = obj[0, 2].item() - OFF                # 컵 바닥
print(f"\n컵: 바닥 z={bot_z:.4f} · 림 z={top_z:.4f} · 원점 z={obj[0,2].item():.4f}")

# 로봇을 치우고(홈 유지) 물체를 살짝 띄운 뒤, '컵 축 위 림 아래' 지점에 물체를
# 하나 더 놓을 수는 없으므로 — 컵 자신을 뒤집어 떨어뜨려 내부 접근성을 본다.
# 대신 더 단순한 판정: 컵을 **거꾸로** 놓고 안착 높이를 본다.
#   속 빈 컵을 뒤집으면 림이 바닥에 닿아 원점이 높게 뜬다.
#   속 찬(convexHull) 이면 원뿔 옆면이 닿아 다르게 안착한다.
root = torch.zeros(4, 13, device=env.device)
root[:, :3] = env.object.data.root_pos_w
root[:, 2] = env.scene.env_origins[:, 2] + float(env.profile.surface_z) + 0.25
# 180도 뒤집기 (x축 회전)
root[:, 3] = 0.0; root[:, 4] = 1.0
env.object.write_root_state_to_sim(root)
for _ in range(args.steps):
    env.step(zero)
inv = (env.object.data.root_pos_w - env.scene.env_origins)[:, 2]
print(f"\n거꾸로 놓았을 때 안착 원점 z = {inv.mean().item():.4f}")
print(f"  속 빈(림 착지) 예상 z ≈ {float(env.profile.surface_z) + (H - OFF):.4f}")
print(f"  정상(바닥 착지) 기준 z = {float(env.profile.surface_z) + OFF:.4f}")
d_hollow = abs(inv.mean().item() - (float(env.profile.surface_z) + (H - OFF)))
print(f"\n{'→ 속 빈 컵(SDF) 로 동작' if d_hollow < 0.02 else '→ ★속이 찬 것처럼 동작(convexHull 폴백 의심)'}")
env.close()
app.close()
