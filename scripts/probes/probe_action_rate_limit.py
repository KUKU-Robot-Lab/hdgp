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

"""액션 변화율 제한기가 **실제로** 제한하는지 잰다.

코드를 믿지 않고 확인한다. 매 스텝 ±1 사이 난수 액션(= 제한기가 없으면 관절 목표가
한 스텝에 최대 1.0 rad 도약)을 넣고, 실제로 적용된 관절 목표의 스텝간 변화를 잰다.

통과 기준:
  1. 어떤 관절도 `velocity_limit × dt` 를 넘지 않는다.
  2. 포화 액션이 계속 들어오면 상한에 **닿아야** 한다(제한기가 그냥 액션을 죽인 게 아니다).
  3. 리셋 직후 첫 스텝도 상한을 넘지 않는다(제한기 상태가 리셋돼야 한다).

실행:
    PYTHONUNBUFFERED=1 ../IsaacLab/isaaclab.sh -p scripts/probes/probe_action_rate_limit.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg("open-grip_l_grasp_sensor", num_envs=args.num_envs)
    env = gym.make("open-grip_l_grasp_sensor", cfg=cfg).unwrapped
    env.reset()

    term = env.action_manager.get_term("arm_action")
    names = term._joint_names
    joint_ids = term._joint_ids
    dt = env.step_dt
    limit = term._max_step_delta[0].clone()          # [rad] per control step

    robot = env.scene["robot"]
    # ★리셋 직후 `joint_pos_target` 버퍼는 stale 하다(아직 아무도 안 썼다). 그걸 기준으로
    #   첫 스텝 변화를 재면 **홈 자세 전체가 한 번에 튄 것처럼** 보인다 — 실제로 그렇게 봐서
    #   제한기가 새는 줄 알았다(j4 에서 0.977 rad = 홈 0.9336 + 상한 0.0435). 한 스텝 뒤 잡는다.
    n_act0 = env.action_manager.total_action_dim
    env.step(torch.zeros(env.num_envs, n_act0, device=env.device))
    prev = robot.data.joint_pos_target[:, joint_ids].clone()
    worst = torch.zeros(len(names), device=env.device)
    reached = torch.zeros(len(names), device=env.device)
    first_step_after_reset = torch.zeros(len(names), device=env.device)
    n_reset = 0

    n_act = env.action_manager.total_action_dim
    for i in range(args.steps):
        # 포화 액션: 매 스텝 부호를 뒤집는 ±1. 제한기가 없으면 목표가 ±0.5 rad 씩 튄다.
        sign = 1.0 if i % 2 == 0 else -1.0
        act = sign * torch.ones(env.num_envs, n_act, device=env.device)
        _, _, terminated, truncated, _ = env.step(act)
        cur = robot.data.joint_pos_target[:, joint_ids]
        # ★리셋된 env 는 제외한다. 리셋은 홈 자세로 **텔레포트**하는 것이라 목표가 크게
        #   튀는 게 정상이고, 그걸 세면 제한기가 새는 것처럼 보인다(실제로 그렇게 봤다).
        alive = ~(terminated | truncated)
        if bool(alive.any()):
            d = (cur - prev)[alive].abs().amax(dim=0)
            worst = torch.maximum(worst, d)
        prev = cur.clone()
        n_reset += int((~alive).sum())

    # 리셋 직후 검사: 강제로 전부 리셋한 뒤 한 스텝
    env.reset()
    # 리셋 뒤 제한기의 기준은 **기본 자세**여야 한다. stale 버퍼가 아니라 그것과 비교한다.
    base = robot.data.default_joint_pos[:, joint_ids]
    env.step(torch.ones(env.num_envs, n_act, device=env.device))
    first_step_after_reset = (
        robot.data.joint_pos_target[:, joint_ids] - base
    ).abs().amax(dim=0)

    print(f"\n제어 스텝 dt = {dt:.4f} s   (도중 리셋된 env-step {n_reset} 개는 제외)")
    print(f"{'관절':<10}{'상한[rad]':>12}{'실측최대':>12}{'상한대비':>10}{'리셋후1스텝':>14}")
    ok = True
    for i, n in enumerate(names):
        lim = float(limit[i]); got = float(worst[i]); rst = float(first_step_after_reset[i])
        ratio = got / lim if lim else float("nan")
        flag = ""
        if got > lim * 1.001:
            flag = "  ← 상한 초과!"; ok = False
        if rst > lim * 1.001:
            flag += "  ← 리셋후 초과!"; ok = False
        if ratio < 0.99:
            flag += "  ← 상한에 못 닿음(제한기가 과하게 죽였나?)"; ok = False
        print(f"{n:<10}{lim:12.5f}{got:12.5f}{ratio:9.1%}{rst:14.5f}{flag}")

    unlimited = 1.0 * 0.5 * 2  # ±1 부호 반전 × scale
    print(f"\n제한기가 없었다면 스텝간 변화는 {unlimited:.3f} rad "
          f"(= {unlimited/dt:.1f} rad/s). 관절 속도 한계는 "
          f"{min(P.ARM_VELOCITY_LIMIT.values()):.3f}~{max(P.ARM_VELOCITY_LIMIT.values()):.3f} rad/s.")
    print("\n판정:", "PASS" if ok else "FAIL")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
