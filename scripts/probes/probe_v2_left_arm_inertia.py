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

"""좌팔 **관절 유효관성 `J_sim`** 측정 → R2S 식으로 kd 를 역산한다.

R2S 원칙(`sim2real/docs/R2S_FRAMEWORK.md` §1)은 **kp 는 벤더값 고정, kd 만
`2ζ√(kp·J_sim)` 으로 계산**이다. 그런데 문서에 실린 kd 숫자
(7.053/4.182/…)는 **테솔로 손 1.763 kg 이 달린 우팔의 `J_sim`** 으로 나온 값이라
2지 그리퍼 좌팔에 그대로 못 쓴다.

여기서는 **좌팔 자신의 `J_sim`** 을 재서 같은 식에 넣는다. `ζ` 는 좌팔 여진 측정이
아직 없으므로 우팔 실측 ζ 를 목표로 삼는다 — 숫자를 옮기는 게 아니라 **방법을
적용**하는 것이다.

`J_sim` 은 일반화 질량행렬의 대각성분이고 **자세 의존**이라, 실제 운전 자세
(리셋 홈)에서 잰다.

실행:
    PYTHONUNBUFFERED=1 python -u scripts/probes/probe_v2_left_arm_inertia.py
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from openarm.gripper.left.grasp_sensor_v2 import v2_preset as P  # noqa: E402

TASK = "open-grip_l_grasp_sensor_v2"
# 우팔 실측 ζ (R2S_FRAMEWORK.md §확정 파라미터)
ZETA_RIGHT = [0.372, 0.579, 0.163, 0.292, 0.071, 0.012, 0.069]
OMEGA_RIGHT = [1.45, 2.58, 1.46, 1.24, 1.40, 2.36, 1.39]   # Hz


def main() -> None:
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    names = P.LEFT_ARM_JOINT_NAMES
    ids, order = robot.find_joints(names, preserve_order=True)

    dt = env.sim.get_physics_dt()
    for _ in range(args.steps):
        env.sim.step(render=False)
        env.scene.update(dt)

    M = robot.root_physx_view.get_generalized_mass_matrices()[0]     # (nD, nD)
    print(f"[mass matrix] shape {tuple(M.shape)}  · dof {robot.num_joints}")
    print(f"[home] " + " ".join(f"{k}={v:+.4f}" for k, v in
                                P.LEFT_ARM_HOME_JOINT_POS.items()))

    kp_v = P.LEFT_ARM_VENDOR_STIFFNESS
    kd_v = P.LEFT_ARM_VENDOR_DAMPING
    print(f"\n{'관절':>8} {'J_sim':>9} {'kp':>6} {'벤더kd':>8} {'ζ_목표':>7} "
          f"{'계산kd':>8} {'배율':>6}   {'벤더ζ':>7} {'ωn[Hz]':>7}")
    out = {}
    for j, (jn, gi) in enumerate(zip(order, ids)):
        J = float(M[gi, gi])
        kp = kp_v[jn]
        z = ZETA_RIGHT[j]
        kd_calc = 2.0 * z * math.sqrt(kp * J)
        # 벤더 kd 를 그대로 쓸 때 실제로 나오는 ζ
        z_vendor = kd_v[jn] / (2.0 * math.sqrt(kp * J)) if J > 0 else float("nan")
        wn = math.sqrt(kp / J) / (2.0 * math.pi) if J > 0 else float("nan")
        out[jn] = kd_calc
        print(f"{jn:>8} {J:>9.5f} {kp:>6.1f} {kd_v[jn]:>8.3f} {z:>7.3f} "
              f"{kd_calc:>8.3f} {kd_calc / kd_v[jn]:>6.2f} {z_vendor:>7.3f} {wn:>7.2f}")

    print("\n★해석")
    print("  · '벤더ζ' = 벤더 kd 를 sim 에 그대로 넣었을 때 나오는 감쇠비.")
    print("    우팔 실측 ζ 와 비슷하면 벤더 kd 로 충분하다는 뜻이고,")
    print("    훨씬 작으면 sim 이 실기보다 **덜 감쇠**돼 진동이 과장된다.")
    print("  · ωn 은 sim 고유진동수 — 우팔 실측 " +
          "/".join(f"{w:.2f}" for w in OMEGA_RIGHT) + " Hz 와 비교한다.")
    print("\nLEFT_ARM_CALC_DAMPING = {")
    for jn, v in out.items():
        print(f"    \"{jn}\": {v:.3f},")
    print("}")
    print("INERTIA_DONE")
    env.close()


main()
simulation_app.close()
