"""이 손이 이 컵을 **애초에 들 수 있는가** — 정책 없이 스크립트로 판정.

fab_test5 실측: gate 0.958 · grip 0.857 · 손가락 4.3개 접촉 · 힘 10~20N 인데
dz 가 정확히 0. 컵 무게는 1.3N 뿐이므로 힘은 충분하다.
남는 가설은 "접촉이 마찰이 아니라 관통/밀어내기라 들 수 없다".

절차: 컵을 손 안에 놓고 → 손을 닫고 → palm 목표 z 를 +15cm 램프 → 컵이 따라오는가.

    isaaclab.sh -p .../probe_lift_capability.py
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--close", type=int, default=80, help="폐합 스텝")
parser.add_argument("--lift", type=int, default=240, help="상승 스텝")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks         # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()
N, A = args.num_envs, env.cfg.action_space
act = torch.zeros(N, A, device=env.device)
env.step(act)

# 컵을 palm 과 손끝 사이로
palm0 = env.robot.data.body_pos_w[:, env.palm_idx]
tips0 = env.robot.data.body_pos_w[:, env._tip_t].mean(dim=1)
root = torch.zeros(N, 13, device=env.device)
root[:, :3] = 0.5 * (palm0 + tips0)
root[:, 3] = 1.0
env.object.write_root_state_to_sim(root)
z0 = (env.object.data.root_pos_w - env.scene.env_origins)[:, 2].clone()
print(f"\n컵을 손 안으로. 시작 z = {z0.mean():.4f}")

print("\n[1] 손 폐합")
for i in range(args.close):
    act[:, 6:] = min(1.0, i / (args.close * 0.5))
    env.step(act)
f, _, _, _ = env._contact()
print(f"  접촉력 env별최대 평균 {f.max(dim=1).values.mean():.2f} N · >1N 손가락 "
      f"{(f > 1.0).float().sum(dim=1).mean():.2f}")

print("\n[2] palm 목표 z 를 +0.15 램프 (손은 닫은 채)")
# a[2] 는 z. a=0 → 홈. 상한까지 선형이므로 목표 z 를 직접 계산해 액션으로 역산.
home_z = env.home_palm[0, 2].item(); hi_z = env.palm_hi[0, 2].item()
for i in range(args.lift):
    frac = min(1.0, i / (args.lift * 0.5))
    tgt_z = home_z + 0.15 * frac
    act[:, 2] = min(1.0, (tgt_z - home_z) / max(hi_z - home_z, 1e-6))
    env.step(act)
    if i in (0, 40, 120, args.lift - 1):
        z = (env.object.data.root_pos_w - env.scene.env_origins)[:, 2]
        pz = (env.robot.data.body_pos_w[:, env.palm_idx] - env.scene.env_origins)[:, 2]
        ff, _, _, _ = env._contact()
        print(f"  step {i:3d} palm z {pz.mean():.4f} (+{pz.mean()-palm0[:,2].mean()+env.scene.env_origins[0,2]:.4f})"
              f" | 컵 dz {(z - z0).mean():+.4f} m | 힘 {ff.max(dim=1).values.mean():5.2f} N")

z = (env.object.data.root_pos_w - env.scene.env_origins)[:, 2]
dz = (z - z0)
print("\n" + "=" * 58)
print(f"컵 상승량: 평균 {dz.mean():+.4f} m · 최대 {dz.max():+.4f} m")
print(f"  5cm 이상 든 env: {(dz > 0.05).float().mean()*100:.1f}%")
if dz.mean() > 0.05:
    print("PASS — 손이 컵을 들 수 있다. 리프트 실패는 **정책/보상** 문제다.")
elif dz.mean() > 0.01:
    print("부분 — 조금 들리지만 미끄러진다. 파지 안정성 문제.")
else:
    print("★FAIL — 스크립트로 닫고 올려도 안 들린다. **물리/접촉** 문제다.")
env.close()
app.close()
