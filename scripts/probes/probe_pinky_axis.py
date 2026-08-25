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

"""pinky 의 굴곡축은 어느 관절인가 — 관절별 손끝 이동을 실측한다.

왜 필요한가: 학습 영상에서 **새끼손가락만 안 말린다**(사용자 관찰). 로그도 같은 말을
한다 — `syn_close/pinky` 0.459 로 index(0.402)보다 **더** 닫으라는 명령이 가는데
`touch/pinky` 는 0.003 이다.

자세표를 보면 pinky 만 `_2` 가 개방·파지 양쪽 0.0 이다:
    index  open (0,0,0,0)  grip (0, 1.9, 1.8, 1.8)   ← _2(MCP) 가 큰 운동
    pinky  open (0,0,0,0)  grip (0, 0.0, 1.8, 1.8)   ← _2 가 죽어 있다
즉 밑마디가 안 접히고 끝마디만 말려 손끝이 컵에 도달하지 못한다.

프로필 주석 두 곳이 서로 모순된다:
    :154  "pinky_1 = Z-flex, 외전 아님"
    :195  "_2(외전)는 안 쓰고 _3 가 curl 역할"
**추론으로 고치면 안 된다.** 여기서 관절 하나씩 훑어 실측한다.

재는 것 (env 를 관절×각도에 나눠 배정해 한 번의 롤아웃으로 전수):
  · 손끝 이동 거리 — 그 관절이 실제로 손끝을 옮기는가
  · **파지중심 방향 성분** — 옮기되 컵을 잡는 쪽으로 옮기는가(굴곡) 아니면
    옆으로 벌리는가(외전). 부호가 판별자다
  · index `_2` 를 같은 방식으로 재서 기준으로 삼는다

★리셋 배제를 위해 `episode_length_s` 를 크게 잡는다.

실행:
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_pinky_axis.py
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-sens_r_grasp_sensor")
parser.add_argument("--steps", type=int, default=260, help="목표 각도까지 정착시킬 스텝")
parser.add_argument("--seed", type=int, default=12345)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.tasks  # noqa: E402,F401

# (손가락, 관절번호) — pinky 4축 전부 + 기준으로 index 4축
CASES = [("pinky", j) for j in (1, 2, 3, 4)] + [("index", j) for j in (1, 2, 3, 4)]
ANGLES = (0.6, 1.2)          # rad. 부호 판별과 크기를 같이 본다
SIGNS = (+1.0, -1.0)


def main() -> None:
    n_cond = len(CASES) * len(ANGLES) * len(SIGNS)
    env_cfg = parse_env_cfg(args_cli.task, num_envs=n_cond)
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 1.0e6          # ★리셋 배제
    env_cfg.synergy_contact_freeze = False    # 접촉 동결이 스윕을 막지 않도록

    env = gym.make(args_cli.task, cfg=env_cfg)
    core = env.unwrapped
    torch.manual_seed(args_cli.seed)
    core.reset()
    dev = core.device

    names = list(core.robot.data.joint_names)
    tips = {f: core.robot.find_bodies(f"r_hl_{f}_tip")[0][0] for f in ("pinky", "index")}

    # env 마다 (손가락, 관절, 목표각) 하나씩 배정
    plan = [(f, j, s * a) for (f, j) in CASES for a in ANGLES for s in SIGNS]
    jidx = torch.tensor([names.index(f"r_hj_{f}_{j}") for f, j, _ in plan], device=dev)
    targ = torch.tensor([t for _, _, t in plan], device=dev)

    # 기준 자세 = 전 관절 0 (완전 개방). 팔은 홈에 둔다.
    q = core.robot.data.joint_pos.clone()
    hand_ids = torch.tensor(
        [names.index(n) for n in core.profile.hand_joint_names], device=dev)
    q[:, hand_ids] = 0.0
    core.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    core.robot.set_joint_position_target(q)
    core.scene.write_data_to_sim()
    for _ in range(30):
        core.sim.step(render=False)
        core.scene.update(core.sim.get_physics_dt())

    def tip_pos(finger: str) -> torch.Tensor:
        return core._env_local(core.robot.data.body_pos_w[:, tips[finger]])

    base = {f: tip_pos(f).clone() for f in tips}
    # 파지중심 방향 = 손끝에서 파지중심으로 가는 단위벡터(굴곡이면 이쪽으로 간다)
    qa = core.robot.data.joint_pos[:, core._fab_t].contiguous()
    _po, _pR = core._tip_palm_frame(qa)
    gc = _po + torch.einsum("bij,j->bi", _pR, core._gc_local) + core._fab_to_env

    # 지정 관절만 목표각으로 램프
    tgt = q.clone()
    rows = torch.arange(n_cond, device=dev)
    for t in range(args_cli.steps):
        f = min(1.0, (t + 1) / (args_cli.steps * 0.5))
        cmd = tgt.clone()
        cmd[rows, jidx] = targ * f
        core.robot.set_joint_position_target(cmd)
        core.scene.write_data_to_sim()
        core.sim.step(render=False)
        core.scene.update(core.sim.get_physics_dt())

    q_end = core.robot.data.joint_pos
    print("\n" + "=" * 88, flush=True)
    print("pinky 굴곡축 실측 — 관절 하나씩 구동했을 때 손끝이 어디로 가는가", flush=True)
    print(f"파지중심 palm-local {[round(float(v)*1000) for v in core._gc_local]}mm · "
          f"{args_cli.steps} 스텝 정착 · env {n_cond}", flush=True)
    print("-" * 88, flush=True)
    print(f"{'관절':<14}{'지령°':>8}{'실제°':>8}{'추종%':>8}"
          f"{'손끝이동mm':>12}{'→파지중심mm':>13}  판정", flush=True)

    for i, (f, j, t) in enumerate(plan):
        d = tip_pos(f)[i] - base[f][i]
        moved = float(torch.norm(d)) * 1000.0
        # 파지중심 방향 성분(+ = 컵을 잡는 쪽으로 접힘)
        u = torch.nn.functional.normalize(gc[i] - base[f][i], dim=-1)
        toward = float((d * u).sum()) * 1000.0
        act = float(q_end[i, jidx[i]])
        track = abs(act / t) * 100.0 if abs(t) > 1e-6 else 0.0
        if track < 30.0:
            verdict = "관절이 안 움직임"
        elif moved < 8.0:
            verdict = "이동 미미"
        elif toward > 0.6 * moved:
            verdict = "★굴곡(파지 유효)"
        elif toward < -0.3 * moved:
            verdict = "역굴곡"
        else:
            verdict = "외전/측방"
        print(f"{f + '_' + str(j):<14}{torch.rad2deg(torch.tensor(t)):>8.1f}"
              f"{torch.rad2deg(torch.tensor(act)):>8.1f}{track:>7.0f}%"
              f"{moved:>12.1f}{toward:>13.1f}  {verdict}", flush=True)

    print("=" * 88, flush=True)
    print("해석: index_2 가 MCP 굴곡의 기준값이다. pinky 축 중 '★굴곡' 이 나오는 것을", flush=True)
    print("      `hand_grip_pose` 에서 구동해야 새끼손가락이 실제로 말린다.", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
