"""Fabrics IK 가 palm 을 물체 근처로 실제로 데려가는가?

정책이 palm target 을 물체 쪽으로 줘도 fabric 이 못 데려가면, 보상은 안 오고
curl 페널티만 먹으므로 "가만히 있기"가 유일한 합리적 선택이 된다.
이 probe 는 정책 없이 palm target 을 물체 바로 위로 강제로 주고,
  (a) 실제 palm 이 목표에 도달하는지 (IK 오차)
  (b) 손끝이 물체에 닿는 거리까지 가는지
를 측정한다. 도달하지 못하면 문제는 reward 가 아니라 제어(fabric)다.

사용:
  ./isaaclab.sh -p scripts/probes/probe_fabric_reach_object.py --task open-tesol_r_grasp_v2-lstm
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200, help="target 유지 스텝 수")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_OUT = open("/tmp/probe_fabric_reach.txt", "w")
_p = print


def print(*a, **kw):  # noqa: A001
    _p(*a, **kw, flush=True)
    _p(*a, **kw, file=_OUT, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

zero = torch.zeros(env.num_envs, env.cfg.num_actions, device=env.device)
for _ in range(int(env.cfg.settle_steps) + 2):   # 물체 안착
    env.step(zero)

obj0 = env.object_pos.clone()
origins = env.scene.env_origins
n = env.num_envs

# 물체 위 여러 높이를 palm target 으로 강제 지정 (env 별로 다른 높이)
offsets = [0.16, 0.13, 0.10, 0.08, 0.06, 0.04, 0.02, 0.00]
per = max(1, n // len(offsets))
labels = [offsets[min(i // per, len(offsets) - 1)] for i in range(n)]
off_t = torch.tensor(labels, device=env.device)

# palm 목표: 물체 바로 위 (G 규약 top-down 자세)
tgt = torch.zeros(n, 6, device=env.device)
tgt[:, :3] = obj0 - origins
tgt[:, 2] += off_t
tgt[:, 3] = 0.0
tgt[:, 4] = 0.0
tgt[:, 5] = math.pi          # ex=180° → 법선 -Z

# 박스 안으로 클램프 (실제 학습과 동일한 안전 경계)
tgt = torch.max(torch.min(tgt, env.palm_maxs_env), env.palm_mins_env)

# palm action 은 절대 pose 이므로 역산 가능:
#   palm_pose = scale(a, mins, maxs) = 0.5*(a+1)*(maxs-mins) + mins
#   → a = 2*(target - mins)/(maxs - mins) - 1
# 즉 "정책이 물체 위로 가라고 지령한" 상황을 그대로 재현한다(실제 학습 경로와 동일).
lo, hi = env.palm_mins_env, env.palm_maxs_env
a_palm = 2.0 * (tgt - lo) / (hi - lo + 1e-9) - 1.0
a_palm = a_palm.clamp(-1.0, 1.0)

act = torch.zeros(n, env.cfg.num_actions, device=env.device)
act[:, :6] = a_palm            # palm: 물체 위로
act[:, 6:] = -1.0              # 손가락: 완전히 편 채 (접근만 본다)

for _ in range(args.steps):
    env.step(act)

palm = env.palm_center_pos - origins
tips = env.fingertip_pos
obj = env.object_pos
ik_err = (palm - tgt[:, :3]).norm(dim=-1)
tip_d = (tips - obj.unsqueeze(1)).norm(dim=-1).min(dim=1).values
palm_d = (env.palm_center_pos - obj).norm(dim=-1)
obj_moved = (obj - obj0).norm(dim=-1)

print("\n" + "=" * 78)
print("Fabrics IK 가 palm 을 물체로 데려가는가 — %s" % args.task)
print("  palm target 을 물체 위 N cm 로 강제 지정하고 %d step 유지" % args.steps)
print("=" * 78)
print("\n  %-10s %-12s %-11s %-12s %-11s %-8s" % (
    "목표(물체위)", "실제 palm z", "IK 오차", "손끝~물체", "palm~물체", "물체이동"))
seen = set()
for i in range(n):
    h = labels[i]
    if h in seen:
        continue
    seen.add(h)
    idx = [j for j in range(n) if labels[j] == h]
    print("  %-10.2f %-12.3f %-11.3f %-12.3f %-11.3f %-8.3f" % (
        h,
        palm[idx, 2].mean().item(),
        ik_err[idx].mean().item(),
        tip_d[idx].mean().item(),
        palm_d[idx].mean().item(),
        obj_moved[idx].mean().item(),
    ))

print("\n  IK 오차 < 3cm  = fabric 이 목표에 도달 (제어 정상)")
print("  IK 오차 큼     = fabric 이 못 감 → reward 가 아니라 제어 문제")
print("  손끝~물체가 0 에 가까우면 손가락만 굽히면 잡을 수 있다는 뜻")
_OUT.close()
env.close()
app.close()
