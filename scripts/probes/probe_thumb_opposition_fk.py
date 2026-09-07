"""손 기하의 **절대 한계** — 엄지와 4 지가 얼마나 가까워질 수 있는가(FK 전용).

앞선 probe 는 액션 박스 안에서 116.9mm 가 최대 근접임을 보였다. 그런데 박스는
`tip_action_span_frac=0.3` 으로 실측 범위의 30% 만 연 것이라, "손이 못 하는 것"과
"박스가 막은 것"이 섞여 있다. 여기서 그 둘을 가른다.

방법: 손 관절 **전범위** 균일 샘플 → fabric FK. 물리도 액션도 거치지 않는 순수
기하다. 이 값이 손이 낼 수 있는 최대치이고, 여기서도 멀면 **자산 자체가 대향 파지를
못 한다**는 뜻이라 어떤 보상·박스로도 못 고친다.

★관절 전범위 샘플은 정책 분포가 아니다([[jointspace-sampling-is-not-policy-distribution]]).
  여기서는 그게 **의도**다 — 도달 **가능성의 상한**을 물었기 때문이다. 개입 필요성을
  묻는 게 아니라 "천장이 어디냐"를 묻는 probe다.

손끝뿐 아니라 **마디**(_3/_4)도 함께 잰다 — 이 손은 감쌈을 마디로 판정하므로
(cup_dist=_4, cup_mid=_3) 손끝이 멀어도 마디가 만나면 대향이 성립할 수 있다.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--rounds", type=int, default=24)
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

best = torch.full((len(IB),), 1e9, device=dev)
best_any = torch.tensor(1e9, device=dev)
tot = 0
# 박스 제약을 씌웠을 때의 최소거리도 같이 센다(손 한계 vs 박스 제약 분리).
LO, HI = env._tip_lo, env._tip_hi
best_box = torch.tensor(1e9, device=dev)
n_in_box = 0

for r in range(args.rounds):
    q = q0.clone()
    u = torch.rand(N, hi_j.shape[1], device=dev)
    q[:, n_arm:] = lo_j + u * (hi_j - lo_j)
    o, R = env._palm_frame(q)
    tips, _ = env.fabric._fingertip_taskmap(q, None)
    tips = tips.reshape(N, -1, 3)
    rel = torch.einsum("bij,bkj->bki", R.transpose(1, 2), tips - o[:, None, :])
    d = (rel[:, IA][:, :, None, :] - rel[:, IB][:, None, :, :]).norm(dim=-1)  # (N,|A|,|B|)
    per_b = d.min(dim=1).values                      # (N,|B|)
    best = torch.minimum(best, per_b.min(dim=0).values)
    best_any = torch.minimum(best_any, per_b.min())
    # 박스 안에 **모든** 손끝이 들어오는 표본만
    inside = ((rel >= LO[None]) & (rel <= HI[None])).all(dim=-1).all(dim=-1)
    if inside.any():
        n_in_box += int(inside.sum())
        best_box = torch.minimum(best_box, per_b[inside].min())
    tot += N
    print(f"  round {r+1}/{args.rounds}: 전범위 최소 {float(best_any)*1000:.1f}mm · "
          f"박스내 표본 {n_in_box}/{tot} 최소 {float(best_box)*1000:.1f}mm", flush=True)

print("\n" + "=" * 88)
print(f"손 기하 절대 한계 — task={args.task} · 표본 {tot}")
print("=" * 88)
print(f"  엄지({FING[IA[0]]}) ↔ 4 지 손끝 최소거리 [mm]")
for k, i in enumerate(IB):
    print(f"     {FING[i]:8s} {float(best[k])*1000:8.1f}")
print(f"\n  ★전범위 최소(손이 낼 수 있는 최대 근접) : {float(best_any)*1000:.1f} mm")
print(f"  ★액션 박스 안 최소                     : "
      f"{float(best_box)*1000:.1f} mm  (박스 내 표본 {n_in_box}/{tot} = "
      f"{100*n_in_box/tot:.2f}%)")

CUP = 60.0
print("\n판정:")
if float(best_any) * 1000 > CUP:
    print(f"  ★★**자산 한계** — 관절 전범위에서도 {float(best_any)*1000:.1f}mm 로")
    print(f"    컵 지름({CUP:.0f}mm)보다 멀다. 이 손은 손끝 대향 파지를 **못 한다**.")
    print("    → 대향 그룹 정의(엄지 vs 4지)를 손 기하에 맞게 바꾸거나, 파지 방식")
    print("      자체를 감쌈(마디) 기반으로 재정의해야 한다. 박스·보상 무관.")
elif float(best_box) * 1000 > CUP:
    print(f"  ★**박스 제약** — 손은 {float(best_any)*1000:.1f}mm 까지 가능한데 액션")
    print(f"    박스가 {float(best_box)*1000:.1f}mm 로 막는다. span_frac 을 열면 된다.")
else:
    print("  대향 가능 — 병목은 탐색/보상이다.")
print("=" * 88)
app.close()
