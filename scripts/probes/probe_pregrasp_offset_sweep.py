"""pregrasp 의 palm x offset 을 얼마나 당겨야 물체를 잡는가?

확정된 기하 (probe_pregrasp_symmetry 실측):
  palm 원점은 물체 바로 위에 정확히 놓인다 (d_palm x = +0.018).
  그런데 손끝은 물체에서 +X 로 20cm 앞에 있다 — tesollo 는 palm 원점(손목)에서
  손끝까지 x 로 20cm 이기 때문이다. Allegro 는 palm 원점이 grasp center 에 가까워
  "palm 을 물체 위에" = "잡는 지점을 물체 위에" 였지만, tesollo 에 그 관례를 그대로
  옮기면 잡는 지점이 20cm 빗나간다.

  손을 완전히 굽혀도 손끝은 palm 쪽으로 ~13cm 당겨질 뿐이라(probe_curl_local),
  20cm 밖에서 시작하면 물체에 닿지 못한다.

여기서 재는 것: palm 을 -X 로 dx 만큼 당긴 뒤 손을 닫고 들어올리면 물체가 따라오는가.
정책 없이 순수 기구/물리로만 확인한다.

사용:
  ./isaaclab.sh -p scripts/probes/probe_pregrasp_offset_sweep.py --task open-tesol_r_grasp_v2-lstm
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=128)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_OUT = open("/tmp/probe_pregrasp_offset_sweep.txt", "w")
_p = print


def print(*a, **kw):  # noqa: A001
    _p(*a, **kw, flush=True)
    _p(*a, **kw, file=_OUT, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

n = env.num_envs
IS_LEFT = "_l_" in args.task
SIDE = "LEFT" if IS_LEFT else "RIGHT"


def trial(dx: float, dz: float = 0.10):
    """palm 을 물체에서 (-dx, ·, +dz) 에 top-down 으로 두고 → 손 폐쇄 → 20cm 상승."""
    env.reset()
    zero = torch.zeros(n, env.cfg.num_actions, device=env.device)
    for _ in range(int(env.cfg.settle_steps) + 2):
        env.step(zero)

    obj0 = env.object_pos.clone()

    tgt = torch.zeros(n, 6, device=env.device)
    tgt[:, 0] = obj0[:, 0] + dx          # dx < 0 이면 물체 뒤(-X)로 당긴다
    tgt[:, 1] = obj0[:, 1]
    tgt[:, 2] = obj0[:, 2] + dz
    tgt[:, 5] = math.pi                   # G 규약 top-down
    tgt = torch.max(torch.min(tgt, env.palm_maxs_env), env.palm_mins_env)

    lo, hi = env.palm_mins_env, env.palm_maxs_env
    act = torch.zeros(n, env.cfg.num_actions, device=env.device)
    act[:, :6] = (2.0 * (tgt - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)
    act[:, 6:11] = -1.0                   # 손 개방
    for _ in range(90):
        env.step(act)

    tips_open = env.fingertip_pos.clone()
    d_open = (tips_open - obj0.unsqueeze(1)).norm(dim=-1)       # (n,5)

    act[:, 6:11] = 1.0                    # 손 폐쇄
    for _ in range(120):
        env.step(act)

    tips_cl = env.fingertip_pos.clone()
    d_close = (tips_cl - obj0.unsqueeze(1)).norm(dim=-1)
    grip = (
        env.binary_contact_buf | env.middle_binary_contact_buf | env.distal_binary_contact_buf
    ).sum(dim=-1).float()

    # 리프트: palm 을 20cm 올린다
    tgt_up = tgt.clone()
    tgt_up[:, 2] = torch.clamp(tgt[:, 2] + 0.20, max=env.palm_maxs_env[:, 2])
    act[:, :6] = (2.0 * (tgt_up - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)
    for _ in range(120):
        env.step(act)

    lift = (env.object_pos[:, 2] - obj0[:, 2]).mean() * 100     # cm
    return d_open.min(dim=1).values.mean(), d_close.min(dim=1).values.mean(), grip.mean(), lift


print("=" * 88)
print("pregrasp palm x offset 스윕 — %s (%s)" % (args.task, SIDE))
print("  palm 을 물체에서 x 로 dx, z 로 +0.10 에 두고 → 손 폐쇄 → 20cm 상승")
print("  현행 pregrasp 는 dx = 0 (palm 원점을 물체 바로 위) → 손끝이 물체 앞 20cm")
print("=" * 88)
print("\n  %-8s %12s %12s %8s %10s" % ("dx", "손끝~물체", "폐쇄후", "grip", "리프트(cm)"))
print("  %-8s %12s %12s %8s %10s" % ("", "(개방, m)", "(m)", "", ""))

for dx in (0.00, -0.04, -0.06, -0.08, -0.09, -0.10, -0.12, -0.15):
    do, dc, g, lf = trial(dx)
    mark = "  ← 잡힘" if lf > 3.0 else ""
    print("  %-8.2f %12.4f %12.4f %8.2f %10.1f%s" % (dx, do, dc, g, lf, mark))

print("\n  현행(dx=0)에서 리프트가 0 이고, 특정 dx 에서 물체가 들리면")
print("  → pregrasp 기준점을 그만큼 당겨야 한다는 뜻이다 (PREGRASP_TOPDOWN_XY 의 x).")

_OUT.close()
env.close()
app.close()
