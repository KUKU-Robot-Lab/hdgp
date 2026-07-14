"""RIGHT world 의 왼팔영역 반발체가 top-down 접근을 막는가?

사용자 관찰(핵심 단서):
    side-to-side : RIGHT 성공, LEFT 실패
    top-down     : LEFT  성공, RIGHT 실패
접근 방향을 바꾸니 성공하는 팔이 뒤집혔다. 이는 팔 자체가 아니라 **접근 경로**에
방향 의존적 방해물이 있다는 뜻이다.

그리고 side 시절 LEFT 실패의 원인은 이미 밝혀져 있었다 — right world 의 왼팔영역
반발체(left_arm_body sphere y=+0.20 r=0.15 z=0.55, left_target_cup box y=+0.10
z 0.20~0.42)를 left 가 그대로 쓰는 바람에 왼손이 자기 workspace 에서 밀려났다.
미러 world 를 만들어 고쳤다.

가설: 지금 RIGHT 를 죽이는 것도 같은 반발체다. side 는 팔을 낮게 옆으로 뻗어
무관했지만, top-down 은 팔을 물체 위(z 0.4~0.7)로 들어올려야 하는데 반발체가
바로 그 높이에 있다.

측정: 같은 top-down 파지 시퀀스를 (a) 현행 world (b) 반발체 뺀 world 로 돌려
리프트를 비교한다. 동시에 팔 링크가 반발체에 얼마나 접근하는지 잰다.

사용:
  ./isaaclab.sh -p scripts/probes/probe_world_obstacle.py --task open-tesol_r_grasp_v2-lstm
  ./isaaclab.sh -p scripts/probes/probe_world_obstacle.py --task open-tesol_r_grasp_v2-lstm --world noarm
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--world", type=str, default="", help="fabric world 덮어쓰기 (예: noarm)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

# --- fabric world 를 env 생성 전에 갈아끼운다 (preset 상수를 직접 덮어쓴다) ---
_side = "left" if "_l_" in args.task else "right"
if args.world:
    import importlib
    _pmod = importlib.import_module(
        "openarm.tesollo.%s.grasp_v2.grasp_%s_preset" % (_side, _side)
    )
    _new = "open_tesollo_boxes_%s" % args.world
    _pmod.FABRIC_WORLD_FILENAME = _new
    # env 모듈이 from-import 로 이미 바인딩했을 수 있으므로 그쪽도 갈아끼운다
    _emod = importlib.import_module(
        "openarm.tesollo.%s.grasp_v2.grasp_%s_env" % (_side, _side)
    )
    _emod.FABRIC_WORLD_FILENAME = _new

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_OUT = open("/tmp/probe_world_obstacle.txt", "a")
_p = print


def print(*a, **kw):  # noqa: A001
    _p(*a, **kw, flush=True)
    _p(*a, **kw, file=_OUT, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

n = env.num_envs
D = env.device

# 반발체 위치 (RIGHT world 기준; LEFT 는 y 미러)
_sgn = -1.0 if _side == "left" else 1.0
OBST = {
    "left_arm_body(sphere r=0.15)": torch.tensor([0.25, 0.20 * _sgn, 0.55], device=D),
    "left_target_cup(box)":         torch.tensor([0.268, 0.100 * _sgn, 0.310], device=D),
}


def trial(dx: float, a1: float):
    """top-down 파지: palm 을 물체 위 (dx,0,+0.10) → PC1 폐쇄 → 20cm 상승."""
    env.reset()
    zero = torch.zeros(n, env.cfg.num_actions, device=D)
    for _ in range(int(env.cfg.settle_steps) + 2):
        env.step(zero)

    obj0 = env.object_pos.clone()

    tgt = torch.zeros(n, 6, device=D)
    tgt[:, 0] = obj0[:, 0] + dx
    tgt[:, 1] = obj0[:, 1]
    tgt[:, 2] = obj0[:, 2] + 0.10
    tgt[:, 5] = math.pi
    tgt = torch.max(torch.min(tgt, env.palm_maxs_env), env.palm_mins_env)

    lo, hi = env.palm_mins_env, env.palm_maxs_env
    act = torch.zeros(n, env.cfg.num_actions, device=D)
    act[:, :6] = (2.0 * (tgt - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)
    act[:, 6:11] = -1.0
    for _ in range(90):
        env.step(act)

    # 접근 완료 시점: palm 추종 오차 + 팔 링크가 반발체에 얼마나 가까운가
    palm_err = (env.palm_center_pos - tgt[:, :3]).norm(dim=-1).mean()
    arm_pos = env.robot.data.body_pos_w[:, :, :] - env.scene.env_origins.unsqueeze(1)  # (n,B,3)
    near = {}
    for name, c in OBST.items():
        d = (arm_pos - c.view(1, 1, 3)).norm(dim=-1)       # (n,B)
        near[name] = d.min(dim=1).values.mean()             # 가장 가까운 링크까지

    act[:, 6] = a1
    for _ in range(120):
        env.step(act)

    tgt_up = tgt.clone()
    tgt_up[:, 2] = torch.clamp(tgt[:, 2] + 0.20, max=env.palm_maxs_env[:, 2])
    act[:, :6] = (2.0 * (tgt_up - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)
    for _ in range(120):
        env.step(act)

    lift = (env.object_pos[:, 2] - obj0[:, 2]).mean() * 100
    return palm_err, near, lift


W = args.world if args.world else "현행(no_table)"
print("\n" + "=" * 88)
print("world 반발체가 top-down 접근을 막는가 — %s / world=%s" % (args.task, W))
print("=" * 88)
print("\n  %-8s %-8s %12s %10s" % ("dx", "PC1", "palm추종오차", "리프트cm"))
for dx in (0.00, -0.08):
    for a1 in (0.0, 0.5):
        pe, near, lf = trial(dx, a1)
        mark = " *" if lf > 3.0 else ""
        print("  %-8.2f %-8.1f %12.4f %10.1f%s" % (dx, a1, pe, lf, mark))

print("\n  [팔 링크 ↔ 반발체 최근접 거리]  (마지막 trial 기준)")
for name, d in near.items():
    r = 0.15 if "sphere" in name else 0.11        # sphere 반지름 / box 대각 절반 근사
    hit = "  ← 반발 영역 안!" if float(d) < r else ""
    print("    %-30s %.4f m  (반발반경 ~%.2f)%s" % (name, d, r, hit))

_OUT.close()
env.close()
app.close()
