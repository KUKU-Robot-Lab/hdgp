"""**팔만으로** 파지중심을 컵에 얼마나 붙일 수 있는가 — λ 임계 유도용.

역할 분리(08.26 사용자 확정): **접근은 팔이 담당, 파지부터가 손가락 담당.**
그래서 접근 중 손끝은 홈에 고정되고, λ 게이트(파지중심↔컵)는 **손가락 도움 없이**
팔만으로 도달 가능한 거리여야 한다. 자매가 자기 포화값(83mm)에 맞춰 λ=120mm 를
고른 것과 같은 방법을, 우리 포화값에 적용한다.

★왜 필요한가: 파지중심을 palm 부착으로 바꾸고 손가락을 홈에 고정하자 `d_gc` 가
  88mm(손가락 자유) → **195~222mm** 로 벌어져 λ(120mm)가 영영 안 열렸다.
  두 변경이 서로를 무력화한 것이고, 충돌하는 것은 **임계값 하나**뿐이다.

측정: 손끝을 홈에 고정한 채 **팔 액션 박스 전역**을 균일 샘플 → fabric → 물리.
      각 표본에서 palm 부착 파지중심 ↔ 컵중심 거리를 재고 분포를 낸다.
      ★액션 → fabric → 물리 파이프라인을 그대로 통과시킨다
      ([[jointspace-sampling-is-not-policy-distribution]] 규약).

사용:
    isaaclab.sh -p scripts/probes/probe_arm_only_reach.py --task open-bis_r_grasp_lift_fab
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--rounds", type=int, default=10)
parser.add_argument("--settle", type=int, default=45,
                    help="지시 후 물리 스텝. fabric τ≈0.78s 이므로 넉넉히.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch                                       # noqa: E402
import gymnasium as gym                            # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg     # noqa: E402
from isaaclab.utils.math import matrix_from_quat   # noqa: E402

import openarm.tasks                               # noqa: E402,F401
from openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env_cfg import (  # noqa: E402
    resolve_cfg,
)

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.episode_length_s = 1.0e6          # ★리셋 오염 차단(프로브 규약)
cfg.enable_adr = False
cfg.enable_physics_dr = False
cfg.enable_tip_markers = False
resolve_cfg(cfg)
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
dev, N, A = env.device, args.num_envs, cfg.action_space

# 손끝 액션 = 홈(정규화 좌표) — 접근 중 실제 배선과 같은 상태로 둔다.
span = (env._tip_hi - env._tip_lo).clamp(min=1e-6)
u_home = (2.0 * (env._tip_home - env._tip_lo) / span - 1.0).clamp(-1.0, 1.0)

best = torch.full((N,), 1e9, device=dev)
alls = []
for r in range(args.rounds):
    a = torch.zeros(N, A, device=dev)
    a[:, :6] = torch.rand(N, 6, device=dev) * 2.0 - 1.0        # 팔만 무작위
    a[:, 6:] = u_home.reshape(-1)[None, :].expand(N, -1)       # 손은 홈 고정
    env.reset()
    for _ in range(args.settle):
        env.step(a)
    pw = env.robot.data.body_pos_w[:, env.palm_idx]
    Rw = matrix_from_quat(env.robot.data.body_quat_w[:, env.palm_idx])
    gc = pw + torch.einsum("bij,j->bi", Rw, env._grasp_center_local)
    d = (gc - env.object.data.root_pos_w).norm(dim=-1)
    best = torch.minimum(best, d)
    alls.append(d.clone())
    print(f"  round {r+1}/{args.rounds}: 이번 최소 {float(d.min())*1000:.1f}mm · "
          f"누적 최소 {float(best.min())*1000:.1f}mm", flush=True)

D = torch.cat(alls) * 1000.0
q = lambda p: float(torch.quantile(D, p))
print("\n" + "=" * 84)
print(f"팔만으로 도달하는 파지중심↔컵 거리 — task={args.task} · 표본 {D.numel()}")
print("=" * 84)
print(f"  최소   {float(D.min()):7.1f} mm      ← 팔이 낼 수 있는 최선")
print(f"  p1     {q(0.01):7.1f} mm")
print(f"  p5     {q(0.05):7.1f} mm")
print(f"  p25    {q(0.25):7.1f} mm")
print(f"  중앙   {q(0.50):7.1f} mm")
print(f"  평균   {float(D.mean()):7.1f} mm")
_cur = float(env.cfg.stage_gate_approach_m) * 1000.0
_pass = float((D < _cur).float().mean()) * 100.0
print(f"\n  현행 λ 임계 {_cur:.0f}mm → 무작위 팔 액션의 {_pass:.2f}% 만 통과")
print("\n판정:")
if float(D.min()) > _cur:
    print(f"  ★★λ 가 **영영 안 열린다** — 팔 최선 {float(D.min()):.0f}mm > 임계 {_cur:.0f}mm.")
    print(f"    임계를 최소값보다 크게 잡아야 한다. 권장 = p5 {q(0.05):.0f}mm 부근")
    print("    (무작위 탐색으로도 5% 는 통과 = 학습이 신호를 만난다).")
elif _pass < 0.5:
    print(f"  ★희소하다({_pass:.2f}%) — 열리긴 하나 무작위로는 거의 못 만난다.")
    print(f"    권장 = p5 {q(0.05):.0f}mm 로 완화하거나 학습 초기에만 넓힌다.")
else:
    print(f"  현행 임계로 충분하다({_pass:.2f}% 통과).")
print("=" * 84)
app.close()
