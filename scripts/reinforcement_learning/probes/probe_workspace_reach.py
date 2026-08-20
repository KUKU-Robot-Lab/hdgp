"""palm 워크스페이스 박스의 **도달성 지도**.

가설: 정책이 명령하는 목표가 박스 안이지만 **팔이 못 닿는 곳**이라, Fabrics 가
타협점에 주차하고 정책은 차등 피드백을 못 받아 겨냥을 못 배운다.
(실측: 목표거리 41cm · palm거리 17cm · 추종오차 38cm — palm 이 자기 목표보다
 컵에 훨씬 가깝다.)

박스를 격자로 훑어 각 점의 정상상태 오차를 재고, "도달 가능(<3cm)" 비율을 낸다.
박스는 grasp_v1(sensor_rl)에서 물려받았는데 bi_s 는 palm 링크가 54.8mm 다르다.

    isaaclab.sh -p .../probe_workspace_reach.py --grid 4 --settle 150
"""
from __future__ import annotations

import argparse
import itertools

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--grid", type=int, default=4, help="축당 격자 수 → grid^3 점")
parser.add_argument("--settle", type=int, default=150, help="점마다 정착 스텝")
parser.add_argument("--box", type=str, default=None,
                    help="후보 박스 'xlo,xhi,ylo,yhi,zlo,zhi' (미지정 시 cfg 박스)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks         # noqa: E402,F401

G = args.grid
pts = list(itertools.product(range(G), repeat=3))
N = len(pts)
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=N)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()
zero = torch.zeros(N, env.cfg.action_space, device=env.device)
env.step(zero)

if args.box:
    v = [float(x) for x in args.box.split(",")]
    lo = torch.tensor(v[0::2], device=env.device)
    hi = torch.tensor(v[1::2], device=env.device)
    print(f"[후보 박스] x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}]")
else:
    lo, hi = env.palm_lo[0, :3], env.palm_hi[0, :3]
# env i 마다 박스 안 격자점 하나를 목표로 준다(자세는 홈 자세 고정)
tgt = env.home_palm.clone()
for i, (a, b, c) in enumerate(pts):
    for k, idx in enumerate((a, b, c)):
        tgt[i, k] = lo[k] + (hi[k] - lo[k]) * (idx / (G - 1) if G > 1 else 0.5)

print(f"\n격자 {G}^3 = {N} 점 · 정착 {args.settle} 스텝", flush=True)
print(f"박스 x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}]",
      flush=True)

for _ in range(args.settle):
    env.palm_targets = tgt
    env.fabric.set_features(env._fabric_hand_cmd, env.palm_targets, "euler_zyx",
                            env.fabric_q.detach(), env.fabric_qd.detach(),
                            env._world_ids, env._world_indicator, env._fabric_damping)
    env._step_fabric()
    for _ in range(env.cfg.decimation):
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

palm = env.robot.data.body_pos_w[:, env.palm_idx] - env.scene.env_origins
err = (tgt[:, :3] - palm).norm(dim=-1)

print("\n=== 도달성 ===", flush=True)
for thr in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10):
    # ★m → mm 는 ×1000 이다. 전에 ×100 으로 적어 라벨이 10배 어긋났고,
    #   그 틀린 숫자로 "도달 93.6%" 라고 보고했다(실제 <10mm 는 58.4%).
    print(f"  오차 < {thr*1000:5.0f}mm : {(err < thr).float().mean()*100:5.1f} %")
print(f"  오차 중앙값 {err.median()*1000:.1f}mm · p90 {torch.quantile(err,0.9)*1000:.1f}mm"
      f" · 최대 {err.max()*1000:.1f}mm")

print("\n=== 축별 (오차 중앙값) ===", flush=True)
for k, name in enumerate("xyz"):
    print(f"  {name}:", end="")
    for idx in range(G):
        m = torch.tensor([p[k] == idx for p in pts], device=err.device)
        v = lo[k] + (hi[k] - lo[k]) * (idx / (G - 1) if G > 1 else 0.5)
        print(f"  {v:+.2f}→{err[m].median():.3f}", end="")
    print()

# 홈이 박스 안인지 (a=0 매핑의 전제)
h = env.home_palm[0, :3]
print(f"\n홈 palm {[f'{v:.3f}' for v in h.tolist()]} 이 박스 안인가: "
      f"{bool(((h >= lo) & (h <= hi)).all())}")

print("\n=== 컵 주변은 닿는가 (스폰 중심 위) ===", flush=True)
obj = env.object.data.root_pos_w - env.scene.env_origins
d_obj = (tgt[:, :3] - obj).norm(dim=-1)
near = d_obj < 0.15
if near.any():
    print(f"  컵 15cm 이내 격자점 {int(near.sum())}개 · 오차 중앙값 {err[near].median():.4f} m")
else:
    print("  ★컵 15cm 이내에 격자점이 없다 — 격자를 촘촘히 하거나 박스를 확인할 것")
# ── 합격 게이트 ─────────────────────────────────────────────────────
frac30 = float((err < 0.03).float().mean())
med = float(err.median())
p90 = float(torch.quantile(err, 0.9))
ok = frac30 >= 0.85 and med <= 0.02 and p90 <= 0.06
print("\n" + "=" * 56)
print("게이트: 오차<30mm 85% 이상 & 중앙값 20mm 이하 & p90 60mm 이하")
print(f"  실측  <30mm {frac30*100:.1f}% · 중앙값 {med*1000:.1f}mm · p90 {p90*1000:.1f}mm"
      f"  →  {'PASS' if ok else '★FAIL'}")
if not ok:
    print("  박스가 팔 도달범위보다 크다. 정책 액션이 포화하면 주로 못 닿는 곳을")
    print("  명령하게 되고, 액션을 바꿔도 결과가 안 바뀌어 겨냥을 배울 수 없다.")
env.close()
app.close()
