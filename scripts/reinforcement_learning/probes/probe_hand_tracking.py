"""손이 **정책 목표를 따라가는가** — 제어 경로별 순수 추종 능력.

왜 컵 없이 재는가: 컵이 손 안에 있으면 "막혀서 못 가는 것"과 "제어가 약해서 못 가는
것"이 섞인다. 먼저 방해 없는 조건에서 추종 자체를 확인해야 그 다음 판정이 선다.

두 층을 나눠 잰다 — 어디서 막히는지 이 둘이 가른다:
  cmd_err   정책 관절 목표 → fabric attractor 가 만든 `fabric_q` (fabric 내부)
  track_err `fabric_q` → PhysX PD 가 실현한 실제 관절 (물리)
`pd` 모드는 fabric 을 거치지 않으므로 track_err 만 의미가 있다.

배경: 손끝 IK(tip)가 정확히 cmd_err 층에서 막혔다(추종오차 85mm, 게이트 23,400스텝
0.000). 관절 direct 로 바꾼 뒤 게이트는 열렸지만 물체를 들지 못한다(dz 0.0014 vs
직접 PD 0.180) — 접촉력은 오히려 크므로(11.3N vs 7.3N) 힘 부족은 아니다.

    isaaclab.sh -p .../probe_hand_tracking.py --hand_control fabric --gain 400
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--hand_control", default="fabric", choices=["fabric", "pd"])
parser.add_argument("--gain", type=float, default=None, help="hand_attractor conical_gain")
parser.add_argument("--steps", type=int, default=180, help="지령 유지 스텝")
parser.add_argument("--ramp", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks         # noqa: E402,F401
from openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env_cfg import (  # noqa: E402
    resolve_cfg,
)

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env_cfg.enable_self_collisions = True
env_cfg.enable_gravity = True
env_cfg.gravity_compensation = 1.0
env_cfg.hand_control = args.hand_control
env_cfg.use_tip_fabric = False
if args.gain is not None:
    env_cfg.hand_attractor_gain = args.gain
resolve_cfg(env_cfg)
env = gym.make(args.task, cfg=env_cfg).unwrapped
N, A = args.num_envs, env.cfg.action_space
dev = env.device
n_arm = env.profile.num_arm_joints


# ★물체를 인위적으로 치우지 않는다. z 아래로 보내면 `fallen` 종료가 매 스텝
#   발동해 env 가 계속 리셋되고 손 이동이 정확히 0 으로 나온다(1차 시도의 결함).
#   팔 액션 0(홈)에서 손끝은 컵에서 ~7cm 떨어져 있어 방해가 없다.
def measure(level: float) -> tuple[float, float, float, float]:
    """level: 손 액션 값(-1=개방 … +1=폐합). 목표 대비 실제가 얼마나 가는가."""
    env.reset()
    act = torch.zeros(N, A, device=dev)
    act[:, 6:] = -1.0
    for _ in range(20):
        env.step(act)
    q0 = env.robot.data.joint_pos[:, env._hand_t].clone()

    for i in range(args.ramp + args.steps):
        act[:, 6:] = -1.0 + (level + 1.0) * min(1.0, (i + 1) / args.ramp)
        env.step(act)

    q1 = env.robot.data.joint_pos[:, env._hand_t]
    # 정책이 지시한 관절 목표(자유 관절만) — env 가 액션을 매핑한 것과 같은 식
    u = 0.5 * (act[:, 6:] + 1.0)
    want_free = env._hand_lo + u * (env._hand_hi - env._hand_lo)
    free_cols = [env._hand_t.tolist().index(int(j)) for j in env._hand_free_t.tolist()]
    got_free = q1[:, free_cols]
    goal_err = (got_free - want_free).abs().mean()
    moved = (q1 - q0).abs().mean()
    cmd_err = float(env.extras.get("hand/cmd_err_rad", torch.tensor(float("nan"))))
    track_err = float(env.extras.get("hand/track_err_rad", torch.tensor(float("nan"))))
    return float(goal_err), float(moved), cmd_err, track_err


tag = f"{args.hand_control}" + (f"/gain{int(args.gain)}" if args.gain else "")
print(f"\n=== 손 추종 · {tag} · {N}env (컵 제거) ===", flush=True)
print(f"{'액션':>6s} {'목표오차[rad]':>13s} {'실제이동[rad]':>13s} "
      f"{'cmd_err':>9s} {'track_err':>10s}", flush=True)
for lv in [0.0, 0.5, 1.0]:
    ge, mv, ce, te = measure(lv)
    print(f"{lv:6.1f} {ge:13.4f} {mv:13.4f} {ce:9.4f} {te:10.4f}", flush=True)

env.close()
app.close()
