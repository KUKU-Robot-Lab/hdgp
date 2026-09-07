#!/usr/bin/env python3
"""policy/critic 관측의 **항별 크기**를 재서 `clip_observations` 에 걸리는지 본다.

왜
--
kuka 정합에서 두 가지를 동시에 했다:
  ① `clip_observations` 100.0 → **5.0** (rl_games 가 raw obs 를 ±5 로 자른다)
  ② obs 에 `fabric_qdd`(관절 **가속도**) · `hand_vel`(6D body 속도, 각속도 포함) 추가
가속도와 각속도는 단위가 rad/s², rad/s 라 ±5 를 우습게 넘는다. 그러면 그 차원들은
**상시 포화**해 정보가 통째로 사라지고, 정책은 상수를 보는 것과 같아진다.
에러가 없고 값도 "그럴듯"해서 조용히 죽는 부류다 — 이 트랙이 이미 세 번 당했다.

무엇을 재나
----------
무작위 액션으로 굴리며 항마다: |x| 의 평균·99 분위·최대, 그리고 **±5 를 넘는 비율**.
넘는 비율이 크면 그 항은 현재 clip 아래에서 쓸모가 없다.

사용:
  TERM=xterm PYTHONUNBUFFERED=1 PYTHONPATH=<hdgp>/source/openarm \\
    ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_obs_scale.py --num_envs 256
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--clip", type=float, default=5.0, help="rl_games clip_observations")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.gripper.left.grasp_sensor  # noqa: F401,E402

TASK = "open-grip_l_grasp_sensor_fab"


def main() -> None:
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()
    om = env.observation_manager
    dev, n = env.device, env.num_envs

    stats = {}
    for _ in range(args.steps):
        a = (torch.rand(n, env.action_manager.total_action_dim, device=dev) * 2 - 1)
        env.step(a)
        for grp in om.active_terms:
            for name, term in zip(om.active_terms[grp], om._group_obs_term_cfgs[grp]):
                v = term.func(env, **term.params)
                v = v.reshape(n, -1).abs()
                k = (grp, name)
                cur = stats.setdefault(k, [0.0, 0.0, 0.0, 0, 0])
                cur[0] += float(v.mean()) ; cur[1] = max(cur[1], float(v.max()))
                cur[2] += float((v > args.clip).float().mean())
                cur[3] += 1 ; cur[4] = v.shape[1]

    print(f"\nclip_observations = {args.clip}  ·  {args.steps} 스텝 × {n} env\n")
    print(f"{'그룹':<8}{'항':<24}{'차원':>5}{'|x| 평균':>10}{'|x| 최대':>10}{'>clip 비율':>12}")
    for (grp, name), (msum, mx, csum, cnt, dim) in sorted(stats.items()):
        frac = csum / cnt
        flag = "  ★포화" if frac > 0.02 else ("  ⚠" if mx > args.clip else "")
        print(f"{grp:<8}{name:<24}{dim:>5}{msum/cnt:>10.3f}{mx:>10.2f}{frac*100:>11.2f}%{flag}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
