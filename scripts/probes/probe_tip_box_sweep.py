"""손끝 액션 박스 스윕 — **어떤 (span_frac, quantile) 이 실제 도달 자세를 담는가**.

08.25 실측: 현행 span_frac=0.3 / q=0.98 에서 손 관절 전범위 FK 표본 32,768 개 중
**박스 안에 5 손끝이 모두 들어오는 표본이 0 개(0.00%)** 였다. 정책이 지시할 수 있는
영역과 손이 실제로 갈 수 있는 영역이 거의 만나지 않는다는 뜻이고, 엄지 접촉력이
좌·우 전 런에서 정확히 0.00N 이었던 이유다.

원인 가설: 박스를 **축별 독립 백분위**로 잡는다. 손끝 5×3=15 축은 서로 강하게
결합돼 있어, 각 축의 중앙 구간을 **동시에** 만족하는 실제 자세는 존재하지 않는다.
박스는 15D 직육면체인데 도달 영역은 그 안을 지나는 얇은 다양체다.

그래서 span 을 키우는 것만으로 되는지, 아니면 유도 방식 자체를 바꿔야 하는지를
여기서 가른다. FK 표본을 **한 번만** 모으고 (f,q) 조합을 전부 그 표본으로 평가한다.

읽는 법:
    포함률   박스 안에 5 손끝이 **동시에** 들어오는 표본 비율. 이게 0 이면 그 박스로
             지시 가능한 자세가 없다는 뜻이다.
    대향최소 박스 안 표본에서 엄지↔4지 최소거리. 컵 지름 35~60mm 보다 작아야 파지 가능.
    중심포함 파지중심(approach 의 목표점)이 각 손끝 박스 안에 있는가 — 밖이면 그
             목표는 **지시할 수 없는 좌표**다.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--rounds", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch                                       # noqa: E402
import gymnasium as gym                            # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg     # noqa: E402

import openarm.tasks                               # noqa: E402,F401
from openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env_cfg import (  # noqa: E402
    resolve_cfg,
)

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.episode_length_s = 1.0e6
cfg.enable_adr = False
cfg.enable_physics_dr = False
resolve_cfg(cfg)
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
dev, N = env.device, args.num_envs
FING = list(env.profile.fingers)
IA, IB = env._grp_a.tolist(), env._grp_b.tolist()
n_arm = env.profile.num_arm_joints
q0 = env.fabric_q.clone()
lo_j, hi_j = env._fab_hand_lo.unsqueeze(0), env._fab_hand_hi.unsqueeze(0)
HOME, GC = env._tip_home, env._grasp_center_local

# ── FK 표본 수집(한 번만) ────────────────────────────────────────────────
acc = []
for r in range(args.rounds):
    q = q0.clone()
    u = torch.rand(N, hi_j.shape[1], device=dev)
    q[:, n_arm:] = lo_j + u * (hi_j - lo_j)
    o, R = env._palm_frame(q)
    tips, _ = env.fabric._fingertip_taskmap(q, None)
    tips = tips.reshape(N, -1, 3)
    acc.append(torch.einsum("bij,bkj->bki", R.transpose(1, 2), tips - o[:, None, :]))
    print(f"  수집 {r+1}/{args.rounds}", flush=True)
REL = torch.cat(acc, dim=0)                    # (S,F,3) palm 상대
S = REL.shape[0]
D = (REL[:, IA][:, :, None, :] - REL[:, IB][:, None, :, :]).norm(dim=-1).min(dim=1).values
DMIN = D.min(dim=1).values                     # (S,) 표본별 엄지↔4지 최소거리
print(f"\n표본 {S} · 전범위 대향 최소 {float(DMIN.min())*1000:.1f}mm", flush=True)

# 대향이 성립하는(≤60mm) 표본이 실제로 어디에 있는지 — 박스를 그쪽으로 열어야 한다.
OPP = DMIN <= 0.060
print(f"대향 성립(≤60mm) 표본 {int(OPP.sum())}/{S} = {100*float(OPP.float().mean()):.2f}%")


def box(f: float, q: float):
    rmin = torch.quantile(REL, 1.0 - q, dim=0)
    rmax = torch.quantile(REL, q, dim=0)
    lo = HOME - (HOME - rmin).clamp(min=0.0) * f
    hi = HOME + (rmax - HOME).clamp(min=0.0) * f
    return lo, hi


print("\n" + "=" * 96)
print("박스 스윕 — 축별 독립 백분위(현행 방식)")
print("=" * 96)
print(f"{'span_f':>7}{'quant':>7}{'포함률%':>10}{'대향최소mm':>12}"
      f"{'대향가능표본%':>14}{'중심포함':>10}{'박스z겹침mm':>13}")
for f in (0.3, 0.5, 0.7, 0.85, 1.0):
    for qq in (0.98, 1.0):
        lo, hi = box(f, qq)
        inside = ((REL >= lo[None]) & (REL <= hi[None])).all(-1).all(-1)
        frac = 100.0 * float(inside.float().mean())
        dm = float(DMIN[inside].min()) * 1000 if inside.any() else float("nan")
        opp = 100.0 * float((inside & OPP).float().mean())
        gc_in = bool((((GC[None] >= lo) & (GC[None] <= hi)).all(-1)).all())
        gap = float(lo[IB, 2].min() - hi[IA, 2].max()) * 1000
        print(f"{f:>7.2f}{qq:>7.2f}{frac:>10.3f}{dm:>12.1f}{opp:>14.3f}"
              f"{('예' if gc_in else '✗아니오'):>10}{gap:>13.1f}")

# ── 대안: 홈 중심 **반경**(결합) 방식 ────────────────────────────────────
print("\n" + "=" * 96)
print("대안 — 홈에서의 거리 백분위로 반경을 잡는 결합 방식(축 독립 아님)")
print("=" * 96)
rad = (REL - HOME[None]).norm(dim=-1)          # (S,F)
print(f"{'quant':>7}{'반경mm(최대)':>14}{'포함률%':>10}{'대향최소mm':>12}{'대향가능표본%':>14}")
for qq in (0.5, 0.7, 0.85, 0.95, 0.99):
    rr = torch.quantile(rad, qq, dim=0)         # (F,)
    inside = (rad <= rr[None]).all(-1)
    frac = 100.0 * float(inside.float().mean())
    dm = float(DMIN[inside].min()) * 1000 if inside.any() else float("nan")
    opp = 100.0 * float((inside & OPP).float().mean())
    print(f"{qq:>7.2f}{float(rr.max())*1000:>14.1f}{frac:>10.3f}{dm:>12.1f}{opp:>14.3f}")

print("\n파지중심 palm+({:.0f},{:.0f},{:.0f})mm — 대향 성립 표본들의 실제 파지점과 비교:"
      .format(*(GC * 1000).tolist()))
if OPP.any():
    mid = 0.5 * (REL[OPP][:, IA].mean(1) + REL[OPP][:, IB].mean(1))     # (M,3)
    m = mid.mean(0) * 1000
    sd = mid.std(0) * 1000
    print(f"  대향 성립 시 파지중심 평균 palm+({m[0]:.0f},{m[1]:.0f},{m[2]:.0f})mm "
          f"· 표준편차 ({sd[0]:.0f},{sd[1]:.0f},{sd[2]:.0f})")
    print(f"  현행 파지중심과의 거리: {float((mid.mean(0)-GC).norm())*1000:.1f}mm")
print("=" * 96)
app.close()
