"""홈/유휴 자세에서 로봇이 **스스로 겹치는지** 링크 쌍 거리로 판정한다.

self-collision 을 켰더니 zero-action 인데 컵이 5cm 밀리고 Fabrics 추종이 10배 악화됐다.
유휴 팔의 tuck 자세가 self-collision 이 꺼진 상태에서 정해졌다면, 켜는 순간
스스로 관통하며 큰 반발력을 낸다 — 그 가설을 검사한다.

    isaaclab.sh -p .../probe_self_collision.py
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=4)
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
zero = torch.zeros(args.num_envs, env.cfg.action_space, device=env.device)
for _ in range(5):
    env.step(zero)

names = env.robot.body_names
pos = env.robot.data.body_pos_w[0] - env.scene.env_origins[0]     # (B,3)
D = torch.cdist(pos, pos)
B = len(names)

# 인접 링크는 당연히 가깝다 — 이름 접두가 다른(=다른 체인) 쌍만 본다.
def chain(n):
    for p in ("r_aj", "r_al", "r_hl", "l_aj", "l_al", "l_hl", "head", "body"):
        if n.startswith(p): return p
    return "other"

ch = [chain(n) for n in names]
print(f"\n링크 {B}개 · 체인별 개수:", {c: ch.count(c) for c in set(ch)})
print("\n=== 서로 다른 체인 간 최소거리 (홈/유휴 자세) ===")
rows = []
seen = set()
for i in range(B):
    for j in range(i + 1, B):
        if ch[i] == ch[j]:
            continue
        rows.append((D[i, j].item(), names[i], names[j], ch[i], ch[j]))
rows.sort()
for d, a, b, ca, cb in rows[:12]:
    flag = " ★겹침 의심" if d < 0.02 else ""
    print(f"  {d*1000:7.1f}mm  {a:22s} ↔ {b:22s}{flag}")

# 체인쌍별 최소
print("\n=== 체인쌍별 최소거리 ===")
best = {}
for d, a, b, ca, cb in rows:
    k = tuple(sorted((ca, cb)))
    if k not in best or d < best[k][0]:
        best[k] = (d, a, b)
for k in sorted(best, key=lambda x: best[x][0]):
    d, a, b = best[k]
    print(f"  {str(k):26s} {d*1000:7.1f}mm   {a} ↔ {b}")

# ── 활성 손 내부 (손가락끼리) — a=0 이 '범위 중앙'이라 반쯤 굽은 자세다 ──────
side = env.profile.side
hidx = [i for i, n in enumerate(names) if n.startswith(f"{side}_hl_")]
def fin(n):
    for f in env.profile.fingers:
        if f"_{f}_" in n: return f
    return None
print(f"\n=== 활성 손 내부 (a=0 = 관절범위 중앙) ===")
pairs = []
for a_ in range(len(hidx)):
    for b_ in range(a_ + 1, len(hidx)):
        i, j = hidx[a_], hidx[b_]
        fa, fb = fin(names[i]), fin(names[j])
        if fa is None or fb is None or fa == fb:
            continue
        pairs.append((D[i, j].item(), names[i], names[j]))
pairs.sort()
if pairs:
    print(f"  다른 손가락 링크 쌍 {len(pairs)}개 · 최소 {pairs[0][0]*1000:.1f}mm")
    for d, a_, b_ in pairs[:6]:
        print(f"    {d*1000:7.1f}mm  {a_:20s} ↔ {b_}")
    n_close = sum(1 for d, _, _ in pairs if d < 0.020)
    print(f"  20mm 미만 쌍: {n_close}/{len(pairs)}")

# a=-1(완전 개방) 과 비교
_open = torch.zeros(args.num_envs, env.cfg.action_space, device=env.device)
_open[:, 6:] = -1.0
for _ in range(40):
    env.step(_open)
pos2 = env.robot.data.body_pos_w[0] - env.scene.env_origins[0]
D2 = torch.cdist(pos2, pos2)
p2 = []
for a_ in range(len(hidx)):
    for b_ in range(a_ + 1, len(hidx)):
        i, j = hidx[a_], hidx[b_]
        fa, fb = fin(names[i]), fin(names[j])
        if fa is None or fb is None or fa == fb: continue
        p2.append(D2[i, j].item())
if p2:
    print(f"\n  a=-1(완전 개방) 최소 {min(p2)*1000:.1f}mm · 20mm 미만 "
          f"{sum(1 for d in p2 if d < 0.020)}/{len(p2)}")
    print("  → 개방이 중앙보다 여유롭다면 a=0 을 '개방'으로 매핑하는 편이 낫다.")

print("\n" + "=" * 60)
worst = rows[0]
if worst[0] < 0.02:
    print(f"★홈/유휴 자세에서 이미 겹친다: {worst[1]} ↔ {worst[2]} ({worst[0]*1000:.1f}mm)")
    print("  self-collision 을 켜면 이 쌍이 상시 반발력을 낸다 → 자세를 고쳐야 한다.")
else:
    print(f"홈/유휴 자세는 자기충돌 없음 (최소 {worst[0]*1000:.1f}mm).")
env.close()
app.close()
