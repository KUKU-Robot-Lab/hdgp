"""손가락이 컵을 **뚫었는지** 기하로 판정한다.

접촉력 스파이크(전형 13~20N vs 최대 7218N)가 관통 신호인지 확인.
컵을 원기둥으로 근사해, 손끝/마디가 그 **내부**에 들어간 깊이를 잰다.
표면 접촉이면 깊이 ≈ 0, 관통이면 음수(=반경보다 안쪽)로 크게 나온다.

    isaaclab.sh -p .../probe_penetration.py --checkpoint <path>
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--checkpoint", default=None, help="없으면 손을 강제 폐합")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200)
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
env.step(zero)

# 컵 기하(cup_big scale 1.0, bbox 실측): 반경 0.045, 원점은 바닥+0.0773, 높이 0.1776
R, OFF, H = 0.045, 0.0773, 0.1776
spec = env.bank.specs[0]
sc = float(spec.scale[2])
R, OFF, H = R * sc, OFF * sc, H * sc
print(f"\n컵 근사: 반경 {R*1000:.1f}mm · 높이 {H*1000:.1f}mm · 원점은 바닥+{OFF*1000:.1f}mm")
print("판정: 손끝이 표면에 닿으면 반경거리 ≈ R. 반경거리 < R 이면 **관통**.\n")

# ★컵을 손 안으로 옮긴다. 안 그러면 손이 컵에서 15cm 떨어진 홈 자세를 재게 되어
#   "관통 없음" 이라는 무의미한 결과가 나온다(1차 실행이 그랬다).
_palm = env.robot.data.body_pos_w[:, env.palm_idx]
_tips = env.robot.data.body_pos_w[:, env._tip_t].mean(dim=1)
_root = torch.zeros(args.num_envs, 13, device=env.device)
_root[:, :3] = 0.5 * (_palm + _tips)
_root[:, 3] = 1.0
env.object.write_root_state_to_sim(_root)

act = torch.zeros(args.num_envs, env.cfg.action_space, device=env.device)
for i in range(args.steps):
    act[:, 6:] = min(1.0, i / (args.steps * 0.3))     # 손만 서서히 폐합
    env.step(act)

obj = env.object.data.root_pos_w - env.scene.env_origins
# 손끝 + 감쌈 마디 전부
names, ids = [], []
for f in env.profile.fingers:
    for b in env.profile.finger_tip_bodies[f] + env.profile.finger_wrap_bodies.get(f, ()):
        bid, _ = env.robot.find_bodies(b)
        names.append(b); ids.append(bid[0])
bt = torch.tensor(ids, device=env.device)
pos = env.robot.data.body_pos_w[:, bt] - env.scene.env_origins[:, None, :]

rel = pos - obj[:, None, :]
radial = rel[:, :, :2].norm(dim=-1)                    # 컵 축까지 수평거리
zrel = rel[:, :, 2]
inside_z = (zrel > -OFF) & (zrel < H - OFF)            # 컵 높이 구간 안
depth = (R - radial) * inside_z.float()                # 양수 = 표면 안쪽으로 들어감

print(f"{'body':22s} {'반경거리':>9s} {'관통깊이':>9s} {'관통 env%':>10s}")
worst = []
for k, n in enumerate(names):
    r_ = radial[:, k].mean().item()
    d_ = depth[:, k].clamp(min=0)
    frac = (d_ > 0.002).float().mean().item() * 100
    print(f"  {n:20s} {r_*1000:8.1f}mm {d_.max().item()*1000:8.1f}mm {frac:9.1f}%")
    worst.append((d_.max().item(), n))
worst.sort(reverse=True)

# ── 손가락 상호 관통 ────────────────────────────────────────────────
# self-collision 을 껐으므로(Fabrics 가 팔만 제어, 손은 직접 PD) 손가락끼리
# 막는 것이 아무것도 없다. 다른 손가락 링크 사이 최소거리로 판정한다.
print("\n=== 손가락 상호 관통 ===")
fing_of = []
for f in env.profile.fingers:
    for _ in env.profile.finger_tip_bodies[f] + env.profile.finger_wrap_bodies.get(f, ()):
        fing_of.append(f)
fi = torch.tensor([env.profile.fingers.index(x) for x in fing_of], device=env.device)
D = torch.cdist(pos, pos)                                   # (N, L, L)
diff = fi[:, None] != fi[None, :]                           # 다른 손가락 쌍만
D = D.masked_fill(~diff.unsqueeze(0), float("inf"))
mind, idx = D.view(D.shape[0], -1).min(dim=1)
LINK_R = 0.010          # 마디 반경 근사 [m] — 두 마디 중심거리가 2R 보다 작으면 겹침
print(f"  마디 반경 근사 {LINK_R*1000:.0f}mm → 중심거리 < {2*LINK_R*1000:.0f}mm 이면 겹침")
print(f"  다른 손가락 링크 간 **최소** 거리: 평균 {mind.mean()*1000:.1f}mm"
      f" · 최소 {mind.min()*1000:.1f}mm")
for thr in (0.010, 0.015, 0.020):
    print(f"    < {thr*1000:.0f}mm 인 env 비율: {(mind < thr).float().mean()*100:5.1f}%")
_worst_env = int(mind.argmin())
_a, _b = divmod(int(idx[_worst_env]), len(fing_of))
print(f"  최악 쌍: {names[_a]} ↔ {names[_b]}  ({mind.min()*1000:.1f}mm)")

print("\n" + "=" * 60)
mx, who = worst[0]
if mx > 0.010:
    print(f"★관통 확인 — {who} 이 컵 표면 안쪽 {mx*1000:.1f}mm 까지 들어갔다.")
    print("  콜라이더(convexHull vs SDF)·depenetration·접촉 offset 을 확인할 것.")
elif mx > 0.002:
    print(f"경미한 침투 {mx*1000:.1f}mm ({who}) — 접촉 해석상 정상 범위일 수 있다.")
else:
    print(f"관통 없음 (최대 {mx*1000:.1f}mm).")
env.close()
app.close()
