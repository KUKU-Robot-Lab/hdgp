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

"""유휴 오른팔 rest 자세 후보 평가 (gripper/left/grasp_sensor).

왜 필요한가: `right/grasp_sensor` 의 우팔 q_home 을 그대로 가져왔더니 오른손이 테이블에
얹혀 자세가 무너졌다(관절 오차 최대 25°). effort_limit_sim 을 1000 까지 올려도 남는 오차라
원인은 토크 부족이 아니라 **자세가 이 씬의 테이블과 충돌**하는 것이다 — 그 태스크와
테이블·컵 배치가 다르다.

여기서 재는 것(후보를 env 에 나눠 배정해 **한 번의 롤아웃**으로 전수 평가):
  · 지령 자세를 실제로 지키는가 (관절 오차)
  · 어떤 링크도 테이블 상면 아래로 내려가지 않는가
  · 왼팔 작업 공간(컵 스폰 박스)과 겹치지 않는가

실행:
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_idle_right_arm_rest.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import openarm.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P

TASK = "open-grip_l_grasp_sensor"

# 후보 자세. j2 = 어깨 pitch, j4 = 팔꿈치.
# 테이블은 x∈[0.210, 0.935] 이므로, 팔을 몸 쪽(x 작은 쪽)으로 접거나 위로 들면 판을 피한다.
CANDIDATES: dict[str, dict[str, float]] = {
    "현행(right/grasp_sensor q_home)": dict(P.RIGHT_ARM_REST_JOINT_POS),
    "차렷(전부 0)": {f"r_aj_{i}": 0.0 for i in range(1, 8)},
    "몸쪽으로 접음": {
        "r_aj_1": 0.0, "r_aj_2": -0.5, "r_aj_3": 0.0,
        "r_aj_4": 1.6, "r_aj_5": 0.0, "r_aj_6": 0.0, "r_aj_7": 0.0,
    },
    "옆으로 벌려 접음": {
        "r_aj_1": -0.6, "r_aj_2": -0.3, "r_aj_3": 0.0,
        "r_aj_4": 1.8, "r_aj_5": 0.0, "r_aj_6": 0.0, "r_aj_7": 0.0,
    },
    "앞으로 낮게 접음": {
        "r_aj_1": 0.0, "r_aj_2": 0.3, "r_aj_3": 0.0,
        "r_aj_4": 2.0, "r_aj_5": 0.0, "r_aj_6": 0.0, "r_aj_7": 0.0,
    },
}


def main() -> None:
    names = list(CANDIDATES)
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=len(names))
    env = gym.make(TASK, cfg=env_cfg).unwrapped

    env.reset()
    robot = env.scene["robot"]
    origins = env.scene.env_origins
    arm_ids, arm_names = robot.find_joints([f"r_aj_{i}" for i in range(1, 8)], preserve_order=True)

    # env k 에 후보 k 를 넣는다. default_joint_pos 를 바꿔야 리셋 후에도 유지된다.
    target = robot.data.default_joint_pos.clone()
    for k, nm in enumerate(names):
        for j, jn in enumerate(arm_names):
            target[k, arm_ids[j]] = CANDIDATES[nm][jn]
    robot.data.default_joint_pos[:] = target
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))

    # ★관절 **상태**만 바꾸면 소용없다. 오른팔은 액션 대상이 아니라 아무도 target 을 갱신하지
    #   않으므로, PD 목표는 리셋 때 잡힌 값(= 원래 자세)에 머문 채 팔이 그리로 돌아가려 한다.
    #   그렇게 재서 "후보 자세를 못 지킨다"고 한 번 오판했다. 매 스텝 target 을 명시한다.
    arm_target = target[:, arm_ids].clone()
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(args.steps):
        robot.set_joint_position_target(arm_target, joint_ids=arm_ids)
        env.step(zero.clone())

    right_bodies = [
        (i, n) for i, n in enumerate(robot.body_names)
        if n.startswith("r_hl_") or n.startswith("r_al_")
    ]
    body_idx = [i for i, _ in right_bodies]

    print("\n=== 유휴 오른팔 rest 자세 후보 ===")
    print(f"  테이블 상면 {P.TABLE_SURFACE_Z:.3f}, 판 x∈[0.210, 0.935]")
    print(f"  컵 스폰 박스 x∈[{P.CUP_SPAWN_X_CENTER - P.CUP_SPAWN_X_RANGE:.2f}, "
          f"{P.CUP_SPAWN_X_CENTER + P.CUP_SPAWN_X_RANGE:.2f}] "
          f"y∈[{P.CUP_SPAWN_Y_CENTER - P.CUP_SPAWN_Y_RANGE:.2f}, "
          f"{P.CUP_SPAWN_Y_CENTER + P.CUP_SPAWN_Y_RANGE:.2f}]")
    print(f"\n{'후보':<32} {'관절오차max':>10} {'최저 z':>9} {'그 링크':<22} {'컵까지 최소':>10}")

    cup = env.scene["object"].data.root_pos_w - origins
    for k, nm in enumerate(names):
        want = torch.tensor([CANDIDATES[nm][jn] for jn in arm_names], device=env.device)
        err = torch.rad2deg((robot.data.joint_pos[k, arm_ids] - want).abs()).max().item()
        zs = robot.data.body_pos_w[k, body_idx, 2] - origins[k, 2]
        lo_i = int(zs.argmin())
        lo_z = float(zs.min())
        pos = robot.data.body_pos_w[k, body_idx, :] - origins[k]
        d_cup = float((pos - cup[k]).norm(dim=-1).min())
        flags = []
        if err > 5.0:
            flags.append("자세못지킴")
        if lo_z < P.TABLE_SURFACE_Z + 0.02:
            flags.append("테이블에닿음")
        if d_cup < 0.15:
            flags.append("컵에가까움")
        mark = "  ← " + "/".join(flags) if flags else "  ✓"
        print(f"{nm:<32} {err:9.2f}° {lo_z:8.4f} {right_bodies[lo_i][1]:<22} "
              f"{d_cup * 1e3:8.1f}mm{mark}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
