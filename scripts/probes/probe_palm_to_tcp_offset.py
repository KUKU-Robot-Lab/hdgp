#!/usr/bin/env python3
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

"""팜 지령 프레임 ↔ 실제 TCP/턱 중점의 **강체 오프셋**을 실측한다.

왜 필요한가 — 지금 제어와 보상이 다른 프레임을 쓴다:

    [Isaac 실제]                               [fabric 모델]
    l_al_7 ──0.1001──▶ l_hl_gripper_base       link7 ──0.1801──▶ palm_link
                        ├─0.015─▶ 턱 두 개                       ↑ fabric attractor
                        └─0.080─▶ l_hl_gripper_tcp                 = 액션이 미는 곳

fab_test41 실측: 팜 지령 ↔ 실제 턱 중점이 **35 mm** 어긋나 있다(x −33 · y +4 · z +13).
정책이 그 오프셋을 스스로 학습해야 하고, sim2real 브리지도 같은 변환을 다시 해야 한다.

**액션의 의미를 TCP 로 바꾸면** fabric 을 건드리지 않고 해소된다:

    palm_cmd = tcp_target − R(palm_cmd_quat) · d

여기서 `d` 는 **팜 프레임에서 본 팜→TCP 벡터**이고 상수여야 한다. 이 프로브가 그 `d` 가
정말 상수인지(= 강체인지) 여러 지령 자세에서 확인하고 값을 낸다.

⚠ 두 URDF 의 팜 자세 규약이 다르다(fabric 기본 palm euler_zyx = π/2, 0, π/2).
  **눈대중으로 상수를 넣지 않는다** — 이 트랙은 그렇게 여러 번 당했다.

사용:
    ./isaaclab.sh -p scripts/probes/probe_palm_to_tcp_offset.py [--num_envs 32] [--settle 120]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--settle", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.gripper.left.grasp_sensor  # noqa: F401,E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P  # noqa: E402

TASK = "open-grip_l_grasp_sensor_fab"


def main() -> None:
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    # ★리셋 오염 방지 — 이 트랙이 네 번 당했다. 프로브는 절대 리셋되면 안 된다.
    cfg.episode_length_s = 1.0e9
    for t in ("time_out", "object_dropping", "object_out_of_workspace", "object_tipped"):
        if hasattr(cfg.terminations, t):
            setattr(cfg.terminations, t, None)
    cfg.curriculum.adr = None
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()
    dev, n = env.device, env.num_envs
    act = env.action_manager.get_term("arm_action")

    # 컵은 치운다 — 프레임 기하를 재는 것이지 파지를 재는 것이 아니다.
    obj = env.scene["object"]
    st = obj.data.default_root_state.clone()
    st[:, :3] = env.scene.env_origins + torch.tensor([0.0, 0.0, -5.0], device=dev)
    obj.write_root_pose_to_sim(st[:, :7])
    obj.write_root_velocity_to_sim(torch.zeros_like(st[:, 7:]))

    robot = env.scene["robot"]
    jaw_ids = [robot.body_names.index(b) for b in P.GRIPPER_FINGER_BODIES]
    base_id = robot.body_names.index(P.GRIPPER_BASE_BODY)

    def measured():
        """(TCP, 턱중점) env 로컬 위치. TCP = base + z·0.08 (URDF `l_hl_gripper_tcp`)."""
        org = env.scene.env_origins
        bq = robot.data.body_quat_w[:, base_id, :]
        bz = matrix_from_quat(bq)[:, :, 2]
        tcp = robot.data.body_pos_w[:, base_id, :] + bz * P.TCP_OFFSET_IN_BASE_Z - org
        jp = robot.data.body_pos_w[:, jaw_ids, :]
        ap = matrix_from_quat(robot.data.body_quat_w[:, jaw_ids[0], :])[:, :, 2]
        jaw = (jp + (ap * P.JAW_PAD_OFFSET).unsqueeze(1)).mean(dim=1) - org
        return tcp, jaw

    # ── env 마다 **다른 지령 자세**를 준다. 오프셋이 상수면 강체다 ──────────
    # 위치는 박스 안에서 고루, 회전은 중심 ± 절반까지 흔든다.
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.zeros(n, env.action_manager.total_action_dim, device=dev)
    a[:, :6] = (torch.rand(n, 6, generator=g).to(dev) * 2.0 - 1.0) * 0.5
    a[:, 6:] = -1.0            # 그리퍼는 열어 둔다
    for _ in range(args.settle):
        env.step(a)

    cmd_pos = act._palm_pose_target[:, :3]          # 팜 지령 위치 (env 로컬)
    ez, ey, ex = act._palm_pose_target[:, 3], act._palm_pose_target[:, 4], act._palm_pose_target[:, 5]
    # euler_zyx(ez,ey,ex) = Rz·Ry·Rx
    cz, sz = torch.cos(ez), torch.sin(ez)
    cy, sy = torch.cos(ey), torch.sin(ey)
    cx, sx = torch.cos(ex), torch.sin(ex)
    R = torch.stack([
        torch.stack([cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx], -1),
        torch.stack([sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx], -1),
        torch.stack([-sy, cy * sx, cy * cx], -1),
    ], dim=1)                                        # (n,3,3) 팜→world

    tcp, jaw = measured()
    err_cmd = (act._fabric_q - robot.data.joint_pos[:, act._arm_joint_ids]).abs().max(dim=-1).values

    def report(name: str, target: torch.Tensor) -> None:
        delta_w = target - cmd_pos                       # world 에서 본 팜→대상
        d_palm = torch.einsum("nji,nj->ni", R, delta_w)  # Rᵀ·Δ = 팜 프레임에서 본 값
        ok = err_cmd < 0.05                              # fabric 이 수렴한 env 만
        d = d_palm[ok]
        print(f"\n── 팜 → {name}  (팜 프레임, mm)   수렴 env {int(ok.sum())}/{n}")
        if d.numel() == 0:
            print("   수렴한 env 가 없다 — settle 을 늘릴 것")
            return
        mean, std = d.mean(0) * 1000, d.std(0) * 1000
        print(f"   평균  x {mean[0]:+8.2f}  y {mean[1]:+8.2f}  z {mean[2]:+8.2f}")
        print(f"   표준편차 {std[0]:8.2f} {std[1]:8.2f} {std[2]:8.2f}")
        rigid = bool((std < 3.0).all())
        print(f"   → {'**강체 오프셋 확인**' if rigid else '★상수가 아니다 — 강체 가정 실패'}"
              f" (판정: 축별 표준편차 < 3 mm)")
        if rigid:
            print(f"   상수: ({mean[0]/1000:.5f}, {mean[1]/1000:.5f}, {mean[2]/1000:.5f})  # m, 팜 프레임")

    print(f"\n지령 자세 {n} 종 · settle {args.settle} 스텝 · 관절 추종오차 "
          f"중앙값 {float(err_cmd.median()):.4f} rad")
    report("TCP (l_hl_gripper_tcp)", tcp)
    report("턱 중점 (패드 보정)", jaw)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
