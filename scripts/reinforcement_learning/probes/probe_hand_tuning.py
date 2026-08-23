"""손 액추에이터 재튜닝 — **파지력이 지령에 따라 변하는가**를 잰다.

fab_test9 실측: `torque_mean 0.82 / limit 1.5`(55%) · `torque_max 1.500` 전 구간 상시.
stiffness 5.0 · effort 1.5 면 포화 시작 각도가 1.5/5.0 = 0.3rad(17°) 뿐이라,
손가락이 컵 표면에서 멈추는 순간 토크가 상한에 붙는다. 힘이 **부족한** 게 아니라
**조절이 안 되는** 상태다(08.16 "손 토크 포화가 파지력 제어를 무효화"와 같은 구조).

판정: 폐합 지령을 20~100% 로 올렸을 때 접촉력이 따라 오르면 제어 가능, 평평하면 불가.
      기울기(force/command)가 클수록 정책이 파지력을 배울 여지가 크다.

★런타임 게인 변경은 PhysX 에는 반영되지만 actuator 장부는 갱신되지 않는다(저장소 기록).
  그래서 여기서는 **접촉 센서**와 **PhysX actuation force** 만 신뢰한다
  (`robot.data.applied_torque` 는 쓰지 않는다).

    isaaclab.sh -p .../probe_hand_tuning.py --num_envs 32
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--settle", type=int, default=70, help="지령 단계마다 정착 스텝")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks         # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
from openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env_cfg import resolve_cfg  # noqa: E402
env_cfg.enable_self_collisions = True      # 손가락 상호관통이 있으면 접촉력이 허수다
env_cfg.enable_gravity = True
env_cfg.gravity_compensation = 1.0
resolve_cfg(env_cfg)
env = gym.make(args.task, cfg=env_cfg).unwrapped
N, A = args.num_envs, env.cfg.action_space
dev = env.device
hand_t = env._hand_free_t                       # 정책이 제어하는 손 관절

# (stiffness, damping, effort_limit)
COMBOS = [
    ("현재",        5.0, 2.0,  1.5),
    ("effort↑",     5.0, 2.0,  5.0),
    ("k↓",          1.5, 0.6,  1.5),
    ("k↓ effort↑",  2.0, 0.8,  3.0),
    ("k↑ effort↑", 10.0, 3.0, 10.0),
]
# ★지령 단계마다 **새로 배치**한다. 한 번 배치하고 지령만 올리면 낮은 지령에서 컵이
#   떨어져 이후 측정이 전부 0 이 된다(첫 시도에서 그렇게 나왔다).
# ★접촉이 성립한 **이후** 구간만 본다. 접촉 전 힘 0 은 제어 문제가 아니라 정상이다.
# ★접촉이 시작되는 지점 이후만 본다. 첫 시도에서 60~80% 가 전부 0 이었던 것은
#   "제어 불가"가 아니라 **컵이 떨어져서**였다(접촉 손가락 0.73개).
LEVELS = [0.80, 0.85, 0.90, 0.95, 1.00]


def measure(level: float, eff: float) -> tuple[float, float, float]:
    """리셋 → 손 열기 → 컵 **공간 고정** → level 까지 램프 → 정착.

    ★컵을 매 스텝 같은 위치로 다시 써서 **운동학적으로 고정**한다. 그러지 않으면
      중력으로 떨어져서 낮은 지령에서 접촉이 아예 없고(첫 시도 실측), 그러면
      "액추에이터 변조성"과 "파지 안정성"이 섞여 액추에이터를 못 가른다.
      여기서 답할 질문은 **"접촉한 뒤 더 밀면 힘이 오르는가"** 하나다.
    """
    env.reset()
    act = torch.zeros(N, A, device=dev)
    act[:, 6:] = -1.0
    for _ in range(25):
        env.step(act)
    palm = env.robot.data.body_pos_w[:, env.palm_idx]
    tips = env.robot.data.body_pos_w[:, env._tip_t].mean(dim=1)
    hold = torch.zeros(N, 13, device=dev)
    hold[:, :3] = 0.5 * (palm + tips); hold[:, 3] = 1.0

    tgt = -1.0 + 2.0 * level
    ramp = 30
    for i in range(ramp + args.settle):
        env.object.write_root_state_to_sim(hold)          # 매 스텝 고정
        act[:, 6:] = -1.0 + (tgt + 1.0) * min(1.0, (i + 1) / ramp)
        env.step(act)
    env.object.write_root_state_to_sim(hold)
    f, _, _, _ = env._contact()
    tau = env.robot.root_physx_view.get_dof_actuation_forces()[:, hand_t].abs()
    return (f.sum(dim=1).mean().item(),
            (tau >= 0.99 * eff).float().mean().item(),
            (f > 1.0).float().sum(dim=1).mean().item())


print("\n" + "=" * 100)
print(f"{'설정':<12s}{'k':>5s}{'kd':>5s}{'eff':>6s}  | 접촉력합[N] " +
      " ".join(f"{int(l*100):>5d}%" for l in LEVELS) + "  | 증가  포화%  손가락")
print("=" * 100)

for name, k, kd, eff in COMBOS:
    ks = torch.full((N, len(hand_t)), k, device=dev)
    ds = torch.full((N, len(hand_t)), kd, device=dev)
    es = torch.full((N, len(hand_t)), eff, device=dev)
    env.robot.write_joint_stiffness_to_sim(ks, joint_ids=hand_t.tolist())
    env.robot.write_joint_damping_to_sim(ds, joint_ids=hand_t.tolist())
    env.robot.write_joint_effort_limit_to_sim(es, joint_ids=hand_t.tolist())

    res = [measure(lv, eff) for lv in LEVELS]
    forces = [r[0] for r in res]
    sat = 100.0 * sum(r[1] for r in res) / len(res)
    nfing = sum(r[2] for r in res) / len(res)
    # ★상대 증가율은 분모가 0 에 가까우면 폭발한다(첫 시도에서 8.3e6 이 나왔다). 절대차로.
    gain = forces[-1] - forces[0]
    print(f"{name:<12s}{k:5.1f}{kd:5.1f}{eff:6.1f}  |            " +
          " ".join(f"{v:6.2f}" for v in forces) +
          f"  |{gain:+6.2f} {sat:5.1f} {nfing:6.2f}")

print("=" * 100)
print("증가 = 100% 지령 힘 - 60% 지령 힘 [N].  0 근처면 **파지력 제어 불가**.")
print("포화% = 손 관절이 effort 상한에 붙어 있던 비율.  손가락 = 1N 초과 접촉 손가락 수.")
env.close()
app.close()
