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

"""왼팔 홈(= 액션 0) 후보 탐색.

lift 레시피에서 홈은 단순 초기값이 아니다. `use_default_offset=True` 라 **액션 0 이 곧 홈**
이고 정책은 홈 ±0.5 rad(= scale) 만 국소 탐색한다. 그래서 홈이 만족해야 할 조건은:

  1. **관절 한계에서 0.5 rad 여유** — 없으면 정책의 탐색 범위가 그쪽으로 잘린다.
     ★현재 홈의 l_aj_7 = 1.3563(77.7°)인데 한계가 ±90° 라 위쪽으로 12.3° 뿐이다.
       렌더에서 "j7 이 꺾여 보인다"는 관찰이 이것이고, 실질 문제는 **탐색 범위 손실**이다.
  2. jaw 축(= gripper_base 의 y 축)이 **수평** — 두 접촉점이 컵 지름 양끝에 놓인다.
  3. 접근축(base +z)이 되도록 수평 — 원통을 옆에서 무는 자세.
  4. TCP 가 컵에서 적당히 앞 — 너무 가까우면 스폰 시 관통, 너무 멀면 도달 부담.
  5. 어떤 링크도 테이블에 닿지 않음.

l_aj_7 은 축이 (0,-1,0) 인 **pitch** 다(롤이 아니다). 그래서 j7 이 접근 각도를 지배하고,
그것을 낮추면 TCP 가 멀어지므로 팔꿈치 j4 로 보정한다. 두 축을 격자로 훑는다.

실행:
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_left_home_search.py
"""

from __future__ import annotations

import argparse
import itertools

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed_value", type=int, default=0)
parser.add_argument("--refine", type=str, default="",
                    help="쉼표로 구분한 7 개 값 주변 ±--refine_span 을 재탐색")
parser.add_argument("--refine_span", type=float, default=0.15)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import torch

import openarm.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from openarm import OPENARM_ROOT_DIR
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P

TASK = "open-grip_l_grasp_sensor"
ACTION_HALF_RANGE = 0.5          # scale=0.5 → 액션 ±1 이 ±0.5 rad

# ★탐색 박스: 이 안의 값은 **전부** 관절 한계에서 0.5 rad 이상 떨어져 있다.
#   현재 홈이 조건을 못 지키는 이유는 j7 이 아니라 `l_aj_6`(홈 -0.6695, 한계 ±0.7854 →
#   여유 0.116)이다. j3(여유 0.164)·j5(0.481)도 빠듯하다. Fabrics 시절 IK 가 "정확한
#   파지 자세"를 풀면서 관절을 한계까지 밀어붙인 결과라, lift 방식에는 맞지 않는다.
SEARCH_BOX = {
    "l_aj_1": (-0.60, 0.30),
    "l_aj_2": (-1.20, -0.35),
    "l_aj_3": (-1.00, 1.00),
    "l_aj_4": (0.60, 1.90),
    "l_aj_5": (-1.00, 1.00),
    "l_aj_6": (-0.28, 0.28),
    "l_aj_7": (-1.00, 1.00),
}


def _joint_limits() -> dict[str, tuple[float, float]]:
    urdf = Path(OPENARM_ROOT_DIR).resolve().parents[2] / (
        "assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.urdf"
    )
    out = {}
    for j in ET.parse(urdf).getroot().iter("joint"):
        name = j.get("name") or ""
        lim = j.find("limit")
        if name.startswith("l_aj_") and lim is not None:
            out[name] = (float(lim.get("lower", "0")), float(lim.get("upper", "0")))
    return out


def main() -> None:
    n_samples = args.num_envs
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=n_samples)
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
    limits = _joint_limits()

    gen = torch.Generator(device="cpu").manual_seed(args.seed_value)
    target = robot.data.default_joint_pos.clone()
    samples = torch.zeros(n_samples, len(arm_names))
    seed_pose = [float(v) for v in args.refine.split(",")] if args.refine else None
    for j, jn in enumerate(arm_names):
        lo, hi = SEARCH_BOX[jn]
        if seed_pose is not None:
            # 정제 모드: 시드 주변만. 단 탐색 박스(=한계여유 보장) 밖으로는 안 나간다.
            lo = max(lo, seed_pose[j] - args.refine_span)
            hi = min(hi, seed_pose[j] + args.refine_span)
        samples[:, j] = torch.rand(n_samples, generator=gen) * (hi - lo) + lo
    # env 0 은 대조군으로 현재 홈을 그대로 둔다
    for j, jn in enumerate(arm_names):
        samples[0, j] = (seed_pose[j] if seed_pose is not None
                         else P.LEFT_ARM_HOME_JOINT_POS[jn])
    for k in range(n_samples):
        for j in range(len(arm_names)):
            target[k, arm_ids[j]] = float(samples[k, j])
    robot.data.default_joint_pos[:] = target
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    # ★ActionTerm 이 생성 시점의 default 를 offset 으로 캐시해 두므로 함께 갱신해야 한다.
    env.action_manager.get_term("arm_action")._offset[:] = target[:, arm_ids]

    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    # ★1 스텝만 굴려 **아직 밀리지 않은** 상태에서 기하를 잰다. 그 뒤 계속 굴려 관통 여부를
    #   본다. 처음에 60 스텝 후 거리를 재는 바람에, 관통해서 이미 튕겨나간 컵까지의 거리를
    #   "여유 있는 배치"로 오독했다(추천 홈이 실제로는 스폰 박스 전체를 관통시켰다).
    env.step(zero.clone())
    cup0 = (obj.data.root_pos_w - origins).clone()
    tcp = (ee.data.target_pos_w[:, 0, :] - origins).clone()
    q0 = robot.data.body_quat_w[:, base_i, :].clone()
    for _ in range(args.steps):
        env.step(zero.clone())

    cup = obj.data.root_pos_w - origins
    cup_moved = (cup[:, :2] - cup0[:, :2]).norm(dim=-1)
    cw, cx, cy, cz = obj.data.root_quat_w.unbind(-1)
    cup_tilt = torch.rad2deg(torch.acos((1 - 2 * (cx * cx + cy * cy)).clamp(-1.0, 1.0)))
    w, x, y, z = q0.unbind(-1)
    jaw_deg = torch.rad2deg(torch.asin((2.0 * (y * z + w * x)).abs().clamp(max=1.0)))
    appr_deg = torch.rad2deg(torch.asin((1.0 - 2.0 * (x * x + y * y)).abs().clamp(max=1.0)))
    d_cup = (tcp - cup0).norm(dim=-1)
    low_z = (robot.data.body_pos_w[:, left_idx, 2] - origins[:, 2:3]).min(dim=1).values

    print("\n=== 왼팔 홈 후보 탐색 (한계여유 0.5 rad 박스에서 랜덤) ===")
    print(f"  컵 {cup[0].tolist()}  |  테이블 상면 {P.TABLE_SURFACE_Z}")
    print(f"  현재 홈: j7={P.LEFT_ARM_HOME_JOINT_POS['l_aj_7']:.4f}  "
          f"j4={P.LEFT_ARM_HOME_JOINT_POS['l_aj_4']:.4f}")
    ranked = []
    for k in range(n_samples):
        # 한계 여유: 모든 관절이 액션 범위(±0.5 rad)를 온전히 쓸 수 있는가
        slack = min(
            min(float(target[k, arm_ids[j]]) - limits[jn][0],
                limits[jn][1] - float(target[k, arm_ids[j]]))
            for j, jn in enumerate(arm_names)
        )
        table_ok = float(low_z[k]) > P.TABLE_SURFACE_Z + 0.01
        # ★관통 검사가 핵심이다. 홈이 컵 자리를 점유하면 스폰 즉시 컵이 튕겨나간다.
        quiet = float(cup_moved[k]) < 0.002 and float(cup_tilt[k]) < 1.0
        good = (
            quiet
            and slack >= ACTION_HALF_RANGE
            and float(jaw_deg[k]) < 10.0
            and float(appr_deg[k]) < 35.0
            and 0.09 < float(d_cup[k]) < 0.17
            # TCP 가 컵 파지 대역 높이 근처여야 한다(테이블 위 10~85 mm = 절대 0.225~0.30).
            and 0.23 < float(tcp[k, 2]) < 0.34
            and table_ok
        )
        if k == 0:
            print(f"  [대조군] jaw {jaw_deg[k]:.1f}° 접근 {appr_deg[k]:.1f}° "
                  f"TCP-컵 {d_cup[k] * 1e3:.1f} mm 최저z {low_z[k]:.4f} 한계여유 {slack:.3f} "
                  f"TCP z={tcp[k, 2]:.4f} 컵이동 {cup_moved[k] * 1e3:.2f} mm"
                  f"{'' if quiet else '  ← 관통!'}")
            continue
        if good:
            # 점수: 접근이 수평에 가깝고 jaw 가 수평이며 한계 여유가 클수록 좋다
            ranked.append((float(appr_deg[k]) + float(jaw_deg[k]) - 20.0 * slack, k, slack))

    print(f"\n  조건 통과 {len(ranked)}/{n_samples - 1} 후보")
    if ranked:
        ranked.sort()
        print(f"\n{'순위':>4} {'jaw':>7} {'접근':>7} {'TCP-컵':>9} {'최저z':>8} {'여유':>7}")
        for rank, (_score, k, slack) in enumerate(ranked[:5], 1):
            print(f"{rank:4d} {jaw_deg[k]:6.1f}° {appr_deg[k]:6.1f}° {d_cup[k] * 1e3:8.1f} "
                  f"{low_z[k]:8.4f} {slack:7.3f}   TCP z={tcp[k, 2]:.4f} "
                  f"컵이동 {cup_moved[k] * 1e3:.2f} mm")
            print("       " + "  ".join(
                f"{jn}={float(target[k, arm_ids[j]]):+.4f}" for j, jn in enumerate(arm_names)))
    else:
        print("\n  → 조건을 모두 만족하는 후보 없음. 박스나 조건을 완화할 것.")
    print(f"\n  ※ 한계여유 ≥ {ACTION_HALF_RANGE} 여야 정책이 액션 범위를 온전히 쓴다.")
    print(f"  ※ 현재 홈의 l_aj_7 여유 = "
          f"{min(P.LEFT_ARM_HOME_JOINT_POS['l_aj_7'] - limits['l_aj_7'][0], limits['l_aj_7'][1] - P.LEFT_ARM_HOME_JOINT_POS['l_aj_7']):.3f} rad "
          f"({math.degrees(limits['l_aj_7'][1] - P.LEFT_ARM_HOME_JOINT_POS['l_aj_7']):.1f}°)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
