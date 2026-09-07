"""DG-5F-S 손끝 액션 박스 중 **겹치는 영역**이 얼마인지 통계를 낸다.

왜: 손끝 15D 액션은 손가락마다 독립 3D 목표라, 정책이 서로 교차하는 지시를 낼 수 있다.
KUKA 는 PCA 5D 라 교차 자세가 액션 공간에 **존재하지 않는다**. 우리는 그 층이 비어
있어 후보가 둘이다 — ①fabric repulsion 으로 계획에서 거르기 ②액션 박스를 잘라내기.

②를 하려면 "박스의 얼마가 겹침인가"를 알아야 한다. 그런데 손가락 하나는 4-DOF 로 3D
목표(여유자유도 1)라 **같은 손끝 위치에 교차하는 해와 안 하는 해가 함께 있다** —
손끝 좌표만으로는 겹침이 결정되지 않는다. 그래서 좌표가 아니라 **fabric 이 실제로 푼
관절각**에서 마디 거리를 잰다.

측정:
    액션 박스 균일 샘플 → fabric IK 수렴 → 손가락 쌍 wrap 마디 최소 중심거리
    → 겹침 비율 · 손가락 쌍별 분해 · repulsion ON/OFF 대비

판정(사전 등록):
    겹침 < 5%   → repulsion 만으로 충분, 박스 유지
    5~20%       → repulsion + 감시 지표
    > 20%       → 박스 축소 / 외전 고정 검토

사용:
    isaaclab.sh -p scripts/probes/probe_finger_overlap_workspace.py --repulsion off
    isaaclab.sh -p scripts/probes/probe_finger_overlap_workspace.py --repulsion on
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--rounds", type=int, default=8, help="샘플 라운드(각 num_envs 개)")
parser.add_argument("--settle", type=int, default=40, help="라운드당 IK 수렴 스텝")
parser.add_argument("--repulsion", choices=["on", "off"], default="off")
parser.add_argument("--contact_mm", type=float, default=18.0,
                    help="마디 반경 9mm×2 = 물리적 접촉선. 이보다 작으면 겹친 것.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym                            # noqa: E402
import torch                                       # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg     # noqa: E402

import openarm.tasks                               # noqa: E402,F401
from openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env_cfg import (  # noqa: E402
    resolve_cfg,
)

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.use_body_repulsion_pairs = (args.repulsion == "on")
cfg.enable_gravity = False
cfg.episode_length_s = 1.0e6      # ★리셋 오염 차단(프로브 필수 규약)
resolve_cfg(cfg)
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
dev, N = env.device, args.num_envs

# 손가락별 wrap 마디 인덱스 (F, P)
WRAP = env._wrap_t
F, P = WRAP.shape
FING = list(env.profile.fingers)
FID = torch.arange(F, device=dev).repeat_interleave(P)
SAME = (FID[:, None] == FID[None, :])

# ★물체를 치운다 — 컵이 손가락 사이에 끼면 그게 벌린 것을 "안 겹쳤다"로 오독한다.
_far = torch.zeros(N, 13, device=dev)
_far[:, 0] = 50.0
_far[:, 3] = 1.0
env.object.write_root_state_to_sim(_far)

thr = args.contact_mm / 1000.0
tot = 0
n_overlap = 0
pair_hits = torch.zeros(F, F, device=dev)
dmins = []
# 겹친 샘플의 액션(손끝 성분)을 모아 "박스 어디서" 를 본다
ov_actions = []

for r in range(args.rounds):
    # 팔은 홈 유지(a=0 → 박스 중앙), 손끝만 균일 샘플
    a = torch.zeros(N, env.cfg.action_space, device=dev)
    a[:, 6:] = torch.rand(N, env.cfg.action_space - 6, device=dev) * 2.0 - 1.0
    for _ in range(args.settle):
        env.step(a)

    pos = env.robot.data.body_pos_w[:, WRAP.reshape(-1)]        # (N, F*P, 3)
    d = torch.cdist(pos, pos).masked_fill(SAME.unsqueeze(0), float("inf"))
    dmin = d.reshape(N, -1).min(dim=1).values                   # (N,)
    dmins.append(dmin)

    ov = dmin < thr
    n_overlap += int(ov.sum())
    tot += N
    if ov.any():
        ov_actions.append(a[ov, 6:].clone())
        # 어느 손가락 쌍이 겹쳤나 — 쌍 단위 최소거리
        dp = d.reshape(N, F, P, F, P).amin(dim=(2, 4))          # (N, F, F)
        pair_hits += ((dp < thr) & ov[:, None, None]).float().sum(0)
    print(f"  round {r+1}/{args.rounds}: 누적 겹침 {n_overlap}/{tot} "
          f"({100*n_overlap/tot:.1f}%)", flush=True)

dmin_all = torch.cat(dmins)
frac = n_overlap / tot

print("\n" + "=" * 88)
print(f"DG-5F-S 손끝 액션 박스 겹침 통계 — repulsion {args.repulsion.upper()} "
      f"· span_frac={env.cfg.tip_action_span_frac} · 샘플 {tot}")
print("=" * 88)
print(f"  접촉선 {args.contact_mm:.0f}mm (마디 반경 9mm × 2)")
print(f"  손가락 쌍 최소거리  중앙값 {dmin_all.median()*1e3:6.1f}mm · "
      f"p10 {dmin_all.quantile(0.10)*1e3:6.1f}mm · 최소 {dmin_all.min()*1e3:6.1f}mm")
print(f"  ★겹침 비율  {frac*100:.1f}%   ({n_overlap}/{tot})")

if n_overlap:
    print("\n  손가락 쌍별 겹침 횟수 (상위):")
    pairs = [(float(pair_hits[i, j]), FING[i], FING[j])
             for i in range(F) for j in range(i + 1, F)]
    for c, x, y in sorted(pairs, reverse=True)[:6]:
        if c > 0:
            print(f"     {x:7s} ↔ {y:7s}  {int(c):5d}  ({100*c/n_overlap:.1f}%)")
    A = torch.cat(ov_actions)          # (M, 15)
    print("\n  겹친 샘플의 액션 분포 (손가락별 xyz 평균, −1..+1):")
    for i, f in enumerate(FING):
        m = A[:, 3*i:3*i+3].mean(0)
        print(f"     {f:7s}  x {m[0]:+.2f}  y {m[1]:+.2f}  z {m[2]:+.2f}")

print("\n  판정:", "repulsion 만으로 충분 (박스 유지)" if frac < 0.05
      else "repulsion + 감시 지표" if frac < 0.20
      else "★박스 축소 / 외전 고정 검토 필요")
print("=" * 88)
app.close()
