# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""관측 채널 위생 진단 — **항별로** 값 범위·분산·NaN 을 잰다.

★★fab_test68/69 가 이 프로브를 만든 이유: obs 를 +15 한 판 둘이 똑같이 죽었는데
   설정 diff 는 의도한 것뿐이었다. 남은 가능성은 **채널 값 자체가 병든 것**이고,
   그건 학습 로그에는 안 찍힌다(메모리: "지표 정확히 0.0000이면 센서 의심").

판정:
  · NaN/Inf 가 하나라도 있으면 그 항이 범인이다.
  · |값| 이 수십을 넘으면 `clip_observations`(100) 근처라 정규화가 망가진다.
  · std 가 0 이면 죽은 채널(정보 없음), std 가 다른 항보다 100 배 크면 지배 채널.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="open-grip_l_grasp_sensor_fab")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    # ★관측 위생은 **노이즈 없이** 봐야 한다. 손상은 그 위에 얹히는 것이다.
    cfg.observations.policy.enable_corruption = False
    # ★항별로 보려면 concatenate 를 끈다 — 붙여 놓으면 어느 항이 병들었는지 못 가린다.
    cfg.observations.policy.concatenate_terms = False
    env = gym.make(args.task, cfg=cfg).unwrapped

    stats: dict[str, list[torch.Tensor]] = {}
    env.reset()
    zero = torch.zeros(env.num_envs, env.action_space.shape[1], device=env.device)
    for _ in range(args.steps):
        obs, *_ = env.step(zero)
        for name, val in obs["policy"].items():
            stats.setdefault(name, []).append(val.detach().float().cpu())

    print("\n=== 관측 채널 위생 (zero-action, 노이즈 OFF) ===")
    print(f"{'항':<24}{'차원':>5}{'min':>10}{'max':>10}{'mean':>10}"
          f"{'std':>10}{'NaN':>6}{'Inf':>6}")
    total = 0
    for name, chunks in stats.items():
        v = torch.cat(chunks)
        total += v.shape[-1]
        nan = int(torch.isnan(v).sum())
        inf = int(torch.isinf(v).sum())
        fin = v[torch.isfinite(v)]
        lo = float(fin.min()) if fin.numel() else float("nan")
        hi = float(fin.max()) if fin.numel() else float("nan")
        flag = "  ← ★" if (nan or inf or abs(lo) > 30 or abs(hi) > 30) else ""
        print(f"{name:<24}{v.shape[-1]:5d}{lo:10.3f}{hi:10.3f}"
              f"{float(fin.mean()):10.3f}{float(fin.std()):10.3f}"
              f"{nan:6d}{inf:6d}{flag}")
    print(f"{'합계':<24}{total:5d}")

    # 채널별 std 가 0 인 곳(죽은 채널)을 항 안에서 집어낸다.
    print("\n=== 항 안의 죽은/지배 채널 ===")
    for name, chunks in stats.items():
        v = torch.cat(chunks)
        s = v.std(dim=0)
        dead = torch.nonzero(s < 1e-6).flatten().tolist()
        big = torch.nonzero(s > 10.0).flatten().tolist()
        if dead or big:
            print(f"  {name}: std≈0 채널 {dead} · std>10 채널 {big}")
    print("  (없으면 전 채널 정상)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
