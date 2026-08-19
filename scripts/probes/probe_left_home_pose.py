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

"""왼팔 홈(= 액션 0) 자세 진단·개선.

lift 레시피에서 초기 자세는 단순한 초기값이 아니다. `use_default_offset=True` 라
**액션 0 이 곧 이 자세**이고 정책은 그 주변 ±0.5 rad 를 국소 탐색한다. 초기 자세가
파지에 유리할수록 학습이 쉽다.

현재 홈은 컵 스폰이 x=0.30 이던 시절 IK 로 뽑은 값인데, 스폰을 x=0.36 으로 옮긴 뒤에도
그대로 쓰고 있다. 게다가 렌더에서 "왼손 시작 자세가 이상하다, j7 이 꺾여 보인다"는 관찰이
나왔다(l_aj_7 = 1.3563 rad = 77.7°).

여기서 재는 것:
  · 현재 홈의 jaw 수평도 / TCP–컵 거리 / 링크-테이블 여유
  · **j7(손목 롤) 스윕** — j7 은 롤이라 jaw 방향을 직접 돌린다. TCP 위치를 거의 바꾸지
    않으면서 jaw 를 수평으로 만들 수 있는지 본다.

실행:
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_left_home_pose.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math

import gymnasium as gym
import torch

import openarm.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P

TASK = "open-grip_l_grasp_sensor"

# j7(손목 롤) 후보. 현재 홈은 +1.3563.
J7_CANDIDATES = [round(v, 4) for v in torch.linspace(-0.4, 1.6, 21).tolist()]


def main() -> None:
    n = len(J7_CANDIDATES)
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=n)
    # 컵을 스폰 중심에 고정해 비교를 깨끗하게 (랜덤화가 있으면 거리 비교가 흔들린다)
    env_cfg.events.reset_object_position.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)
    }
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    obj = env.scene["object"]
    ee = env.scene["ee_frame"]
    origins = env.scene.env_origins
    arm_ids, arm_names = robot.find_joints(P.LEFT_ARM_JOINT_NAMES, preserve_order=True)
    base_i = robot.body_names.index(P.GRIPPER_BASE_BODY)
    left_idx = [i for i, nm in enumerate(robot.body_names) if nm.startswith(("l_hl_", "l_al_"))]

    # env k 에 j7 후보 k 를 넣는다. default 를 바꿔야 액션 0 의 기준점도 함께 움직인다.
    target = robot.data.default_joint_pos.clone()
    for k, j7 in enumerate(J7_CANDIDATES):
        for j, jn in enumerate(arm_names):
            target[k, arm_ids[j]] = (
                j7 if jn == "l_aj_7" else P.LEFT_ARM_HOME_JOINT_POS[jn]
            )
    robot.data.default_joint_pos[:] = target
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))

    # ★★`default_joint_pos` 만 바꾸면 소용없다. `JointPositionAction` 은 생성 시점에
    #   default 를 `_offset` 으로 **복사해 두고** 매 스텝 `target = offset + scale*action`
    #   을 쓴다. 그래서 액션 0 은 여전히 옛 홈을 가리키고, 팔이 그리로 돌아간다.
    #   (그렇게 재서 "j7 을 바꿔도 jaw 가 안 변한다"고 한 번 오판했다.)
    arm_term = env.action_manager.get_term("arm_action")
    arm_term._offset[:] = target[:, arm_ids]

    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    arm_target = target[:, arm_ids].clone()
    for _ in range(args.steps):
        env.step(zero.clone())

    cup = obj.data.root_pos_w - origins
    tcp = ee.data.target_pos_w[:, 0, :] - origins
    q = robot.data.body_quat_w[:, base_i, :]
    w, x, y, z = q.unbind(-1)
    jaw_z = 2.0 * (y * z + w * x)                     # base y 축의 world z 성분
    appr_z = 1.0 - 2.0 * (x * x + y * y)              # base z 축의 world z 성분
    jaw_deg = torch.rad2deg(torch.asin(jaw_z.abs().clamp(max=1.0)))
    appr_deg = torch.rad2deg(torch.asin(appr_z.abs().clamp(max=1.0)))
    d_cup = (tcp - cup).norm(dim=-1)
    low_z = (robot.data.body_pos_w[:, left_idx, 2] - origins[:, 2:3]).min(dim=1).values
    settle = torch.rad2deg((robot.data.joint_pos[:, arm_ids] - arm_target).abs().max(dim=1).values)

    print("\n=== 왼팔 홈: j7(손목 롤) 스윕 ===")
    print(f"  컵 중심 {cup[0].tolist()}  (스폰 고정)")
    print(f"  현재 홈 j7 = {P.LEFT_ARM_HOME_JOINT_POS['l_aj_7']:+.4f} rad "
          f"({math.degrees(P.LEFT_ARM_HOME_JOINT_POS['l_aj_7']):.1f}°)")
    print(f"\n{'j7(rad)':>9} {'jaw수평이탈':>11} {'접근pitch':>10} {'TCP-컵(mm)':>11} "
          f"{'최저링크z':>10} {'정착오차':>9}")
    best = None
    for k, j7 in enumerate(J7_CANDIDATES):
        mark = ""
        if abs(j7 - P.LEFT_ARM_HOME_JOINT_POS["l_aj_7"]) < 1e-3:
            mark = "  ← 현재"
        ok = low_z[k] > P.TABLE_SURFACE_Z + 0.01
        score = float(jaw_deg[k]) + 0.1 * float(d_cup[k]) * 1e3
        if ok and (best is None or score < best[0]):
            best = (score, j7, k)
        print(f"{j7:9.4f} {jaw_deg[k]:10.1f}° {appr_deg[k]:9.1f}° {d_cup[k] * 1e3:10.1f} "
              f"{low_z[k]:10.4f} {settle[k]:8.1f}°{mark}")
    if best is not None:
        _, j7, k = best
        print(f"\n  → jaw 수평에 가장 가까운 후보: j7 = {j7:+.4f} rad "
              f"(jaw {jaw_deg[k]:.1f}°, TCP–컵 {d_cup[k] * 1e3:.1f} mm)")
    print("  ※ jaw 수평 이탈 0° = 두 손가락이 같은 높이 = 컵 지름 양끝을 무는 자세")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
