"""Fabrics 가 palm 을 목표로 못 데려가는 이유 — 축별 분해 + 관절 한계 + 프레임 정합.

가설 3개를 한 번에 가른다:
  (1) palm 프레임 불일치 — sim USD r_hl_palm 과 fabric attractor 기준점이 다르면
      목표와 무관하게 일정한 오프셋이 남는다 (모든 목표에서 같은 방향/크기).
  (2) attractor gain 부족 — 목표 방향으로 가긴 하는데 덜 간다 (오차가 목표 거리에 비례).
  (3) 도달 불가 자세 / 관절 한계 — 특정 목표에서만 크게 벗어나고 관절이 limit 에 붙는다.

사용:
  ./isaaclab.sh -p scripts/probes/probe_fabric_diagnose.py --task open-tesol_r_grasp_v2-lstm
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_OUT = open("/tmp/probe_fabric_diag.txt", "w")
_p = print


def print(*a, **kw):  # noqa: A001
    _p(*a, **kw, flush=True)
    _p(*a, **kw, file=_OUT, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

zero = torch.zeros(env.num_envs, env.cfg.num_actions, device=env.device)
for _ in range(int(env.cfg.settle_steps) + 2):
    env.step(zero)

origins = env.scene.env_origins
n = env.num_envs
obj0_local = env.object_pos.clone()   # 이미 env-local

offsets = [0.16, 0.12, 0.08, 0.06, 0.04]
per = max(1, n // len(offsets))
labels = [offsets[min(i // per, len(offsets) - 1)] for i in range(n)]
off_t = torch.tensor(labels, device=env.device)

tgt = torch.zeros(n, 6, device=env.device)
tgt[:, :3] = obj0_local
tgt[:, 2] += off_t
tgt[:, 5] = math.pi
tgt = torch.max(torch.min(tgt, env.palm_maxs_env), env.palm_mins_env)

lo, hi = env.palm_mins_env, env.palm_maxs_env
a = torch.zeros(n, env.cfg.num_actions, device=env.device)
a[:, :6] = (2.0 * (tgt - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)
a[:, 6:] = -1.0

for _ in range(args.steps):
    env.step(a)

# ---- 측정 ----
palm_sim = env.palm_center_pos                                 # 이미 env-local
fab_q = env.fabric_q[:, :7]                                    # fabric 이 푼 팔 관절
sim_q = env.robot.data.joint_pos[:, env.arm_dof_indices]       # 실제 로봇 팔 관절
err = palm_sim - tgt[:, :3]                                    # 축별 오차

# fabric 이 스스로 생각하는 palm 위치 (FK, env-local) — 프레임 정합 확인
_pts, _ = env.open_tesollo_fabric.get_taskmap("palm")(
    env.robot.data.joint_pos[:, env.actuated_dof_indices], None
)
fab_palm = _pts[:, :3]

lim = env.robot.data.soft_joint_pos_limits[0, env.arm_dof_indices]   # (7,2)

print("\n" + "=" * 84)
print("Fabrics 미도달 원인 진단 — %s   (%d step 유지)" % (args.task, args.steps))
print("=" * 84)

print("\n[A] 축별 오차 (실제 palm - 목표), env-local")
print("  %-9s %-9s %-9s %-9s %-9s" % ("목표(위)", "err_x", "err_y", "err_z", "‖err‖"))
seen = set()
for i in range(n):
    h = labels[i]
    if h in seen:
        continue
    seen.add(h)
    idx = [j for j in range(n) if labels[j] == h]
    e = err[idx].mean(dim=0)
    print("  %-9.2f %+-9.3f %+-9.3f %+-9.3f %-9.3f" % (
        h, e[0], e[1], e[2], err[idx].norm(dim=-1).mean()))
print("  → 모든 목표에서 오차가 같은 방향·크기면 (1) 프레임 오프셋")
print("  → 목표를 낮출수록 err_z 가 커지면 (3) 도달 불가/한계")

if fab_palm is not None:
    d = (fab_palm - palm_sim).norm(dim=-1)
    print("\n[B] 프레임 정합: fabric FK palm vs sim USD palm")
    print("  평균 %.4f m  최대 %.4f m" % (d.mean(), d.max()))
    print("  → 3cm 이상이면 fabric 이 다른 지점을 palm 으로 알고 있다 (프레임 불일치)")
else:
    print("\n[B] fabric FK palm 을 읽지 못함 (env.hand_pos 없음)")

print("\n[C] fabric 이 푼 관절 vs 실제 로봇 관절 (추종 오차)")
dq = (fab_q - sim_q).abs()
print("  관절별 평균 오차(rad):", " ".join("%.3f" % v for v in dq.mean(dim=0)))
print("  → 크면 fabric 은 풀었는데 로봇이 못 따라간다 (PD gain / 물리 문제)")

print("\n[D] 관절이 한계에 붙어 있는가 (fabric 해 기준)")
for j in range(7):
    q = fab_q[:, j]
    lo_j, hi_j = lim[j, 0].item(), lim[j, 1].item()
    at_lo = (q < lo_j + 0.05).float().mean().item()
    at_hi = (q > hi_j - 0.05).float().mean().item()
    flag = " ←한계" if max(at_lo, at_hi) > 0.3 else ""
    print("  j%d  범위[%+.2f,%+.2f]  평균 %+.2f  하한붙음 %3.0f%%  상한붙음 %3.0f%%%s" % (
        j + 1, lo_j, hi_j, q.mean().item(), at_lo * 100, at_hi * 100, flag))
print("  → 한계에 붙어 있으면 (3) 팔이 그 자세를 물리적으로 못 만든다")

_OUT.close()
env.close()
app.close()
