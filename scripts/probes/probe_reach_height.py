"""palm 이 실제로 어느 높이까지 도달하는가 (Fabrics IK 기준).

top-down 파지에서는 palm 이 물체보다 (clearance + 여유) 만큼 위에 있어야 한다.
따라서 물체를 goal(z=0.65) 로 옮기려면 palm 이 0.72~0.81 까지 올라가야 하는데,
palm workspace 박스 z 상한이 0.65 라 물체 절반이 goal 에 도달할 수 없다.
박스를 올리기 전에, 팔이 물리적으로 그 높이에 닿는지 먼저 확인한다.

사용:
  ./isaaclab.sh -p scripts/probes/probe_reach_height.py --task open-tesol_r_grasp_v2-lstm
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_OUT = open("/tmp/probe_reach.txt", "w")
_p = print


def print(*a, **kw):  # noqa: A001
    _p(*a, **kw, flush=True)
    _p(*a, **kw, file=_OUT, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

# goal 의 xy 위에서 여러 높이를 목표로 주고 Fabrics IK 를 돌린다
goal = list(env.cfg.object_goal_pos)
heights = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
n = env.num_envs
per = max(1, n // len(heights))

targets = torch.zeros(n, 6, device=env.device)
labels = []
for i in range(n):
    h = heights[min(i // per, len(heights) - 1)]
    labels.append(h)
    targets[i, 0] = goal[0]
    targets[i, 1] = goal[1]
    targets[i, 2] = h
    # top-down 자세 (G 규약: ez=0, ey=0, ex=180°)
    targets[i, 3] = 0.0
    targets[i, 4] = 0.0
    targets[i, 5] = math.pi

env_ids = torch.arange(n, device=env.device)
q0 = env.robot_start_joint_pos[env_ids].clone()
q = env._run_reset_fabric(env_ids, targets, q0)

# FK 로 실제 palm 위치 확인 — 로봇에 적용 후 읽는다
env.robot.write_joint_state_to_sim(
    q, torch.zeros_like(q), joint_ids=env.actuated_dof_indices, env_ids=env_ids
)
env.scene.write_data_to_sim()
env.sim.step(render=False)
env.scene.update(dt=env.cfg.sim.dt)

palm = env.robot.data.body_pos_w[:, env.palm_body_index] - env.scene.env_origins

print("\n" + "=" * 64)
print("palm 도달 높이 검증 — %s" % args.task)
print("  goal xy = (%.2f, %.2f),  현재 palm 박스 z 상한 = 0.65" % (goal[0], goal[1]))
print("=" * 64)
print("\n  %-10s %-12s %-10s %-8s" % ("목표 z", "실제 palm z", "오차", "판정"))
seen = {}
for i in range(n):
    h = labels[i]
    if h in seen:
        continue
    seen[h] = True
    idx = [j for j in range(n) if labels[j] == h]
    z = palm[idx, 2].mean().item()
    err = abs(z - h)
    ok = "도달" if err < 0.03 else ("근접" if err < 0.08 else "미달")
    print("  %-10.2f %-12.3f %-10.3f %-8s" % (h, z, err, ok))

print("\n  (오차 < 3cm = 도달. 미달이면 그 높이는 팔이 물리적으로 못 간다)")
_OUT.close()
env.close()
app.close()
