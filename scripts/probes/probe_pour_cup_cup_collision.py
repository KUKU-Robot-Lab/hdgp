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

"""받는 컵과 소스 컵이 **실제로 충돌하는지** 강제로 겹쳐서 확인한다.

왜. 09.02 사용자가 영상에서 "컵이 컵을 통과한다"를 발견했다. `contact_offset` 음수는
고쳤고(dump 로 0.02/0.0 확인), 자산도 `collisionEnabled=True` · SDF 근사로 정상이다.
그런데도 통과한다면 **kinematic 강체 + SDF 조합**에서 PhysX 가 접촉을 만들지 않는
엔진 수준 문제다. 설정만 읽어서는 판별이 안 되므로 물리로 직접 묻는다.

방법. 정책을 쓰지 않고 좌팔 TCP 목표를 **소스 컵 쪽으로 직진**시켜 두 컵을 겹친다.
받는 컵은 kinematic 이라 밀리지 않으므로, 충돌이 살아 있다면 **소스 컵이 밀려나야**
한다. 그래서 관측량은 두 개다:

  ① 컵-컵 중심거리        — 얼마나 겹쳤나 (반경 합 아래로 내려가면 기하 침투)
  ② 소스 컵의 손 기준 이탈 — 밀렸나. 안 밀리면 유령이다.

②는 `grasp_broken` 판정과 같은 양(팜 기준 상대 위치)이라, 파지가 버티는지도 같이 보인다.

실행:
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_pour_cup_cup_collision.py \\
        --num_envs 4 --steps 200 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_pour_sensor-play-lstm")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--every", type=int, default=20)
parser.add_argument("--bank", type=str, default="")
parser.add_argument("--arm_gains", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402
from pathlib import Path  # noqa: E402

_HDGP = Path(__file__).resolve().parents[2]
for _p in (str(_HDGP / "scripts/tools"), str(_HDGP / "source/openarm")):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
for _n in [m for m in sys.modules if m == "openarm" or m.startswith("openarm.")]:
    del sys.modules[_n]

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm  # noqa: E402
_EXPECTED = str((_HDGP / "source/openarm/openarm").resolve())
if not str(Path(openarm.__file__).resolve()).startswith(_EXPECTED + os.sep):
    raise SystemExit(f"openarm 이 저장소 밖에서 왔다: {openarm.__file__}")
import openarm.tasks  # noqa: F401,E402


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    cfg.left_arm_action_enable = True
    if args_cli.bank:
        cfg.warm_state_paths = (str(Path(args_cli.bank).expanduser().resolve()),)
    if args_cli.arm_gains:
        cfg.arm_gain_profile = args_cli.arm_gains
    cfg.finalize_after_overrides()

    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    cp = env.cfg.left_target_cup_cfg.spawn.collision_props
    print(f"[COL] 받는 컵 — kinematic={env.cfg.left_target_cup_cfg.spawn.rigid_props.kinematic_enabled} · "
          f"contact_offset={cp.contact_offset} · rest_offset={cp.rest_offset}", flush=True)

    env.reset()
    act = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)

    # 소스 컵 방향 단위벡터를 base 프레임에서 구해, 좌팔 TCP 를 그쪽으로 직진시킨다.
    # 액션은 [-1,1] 이 delta 배율이므로 방향만 넣으면 매 스텝 최대속도로 접근한다.
    from isaaclab.utils.math import subtract_frame_transforms
    src_b, _ = subtract_frame_transforms(
        env.robot.data.root_pos_w, env.robot.data.root_quat_w,
        env.cup.data.root_pos_w, env.cup.data.root_quat_w)
    dirv = src_b - env.left_tcp_target_pos_b
    dirv = dirv / dirv.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    d0 = (env.left_target_cup.data.root_pos_w - env.cup.data.root_pos_w).norm(dim=-1)
    # 소스 컵의 팜 기준 상대 위치(밀렸는지 판정) — grasp_broken 과 같은 양.
    palm_w = env.robot.data.body_pos_w[:, env._left_hand_body_index] * 0  # placeholder
    rel0 = (env.cup.data.root_pos_w - env.robot.data.root_pos_w).clone()
    print(f"[COL] 시작 컵-컵 거리 {[round(float(v)*1000, 1) for v in d0]} mm", flush=True)

    for step in range(args_cli.steps + 1):
        if step % args_cli.every == 0:
            d = (env.left_target_cup.data.root_pos_w - env.cup.data.root_pos_w).norm(dim=-1)
            rel = (env.cup.data.root_pos_w - env.robot.data.root_pos_w)
            push = (rel - rel0).norm(dim=-1)
            tcp_off = (env.left_tcp_target_pos_b - env._left_tcp_rest_pos_b).norm(dim=-1)
            print(f"[COL] step {step:>3} · 컵-컵 {float(d.min())*1000:6.1f}mm(최소) "
                  f"{float(d.mean())*1000:6.1f}mm(평균) · 소스컵 이탈 "
                  f"{float(push.max())*1000:6.1f}mm(최대) · TCP 이탈 "
                  f"{float(tcp_off.mean())*1000:6.1f}mm", flush=True)
        if step < args_cli.steps:
            act[:, 12:15] = dirv          # 좌팔만 구동, 우팔·손은 0(warm 자세 유지)
            env.step(act)

    d = (env.left_target_cup.data.root_pos_w - env.cup.data.root_pos_w).norm(dim=-1)
    push = ((env.cup.data.root_pos_w - env.robot.data.root_pos_w) - rel0).norm(dim=-1)
    print("\n[COL] 판정", flush=True)
    print(f"  최종 컵-컵 최소거리 {float(d.min())*1000:.1f}mm "
          f"(두 컵 외경 합 ≈ 88mm — 이보다 작으면 기하 침투)", flush=True)
    print(f"  소스 컵 최대 이탈  {float(push.max())*1000:.1f}mm "
          f"(침투했는데 이 값이 ~0 이면 **유령** = 접촉이 생성되지 않는다)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
