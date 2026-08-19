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

"""학습된 정책이 컵을 **무엇으로** 들고 있는지 측정한다.

왜 필요한가: test3(1500 epoch)에서 `lifting_object` 가 에피소드의 91% 를 차지하는데
`reaching_object` 는 평탄했다(TCP–컵 약 19 cm). 즉 리프트 판정은 계속 참인데 **그리퍼는
컵 근처에 없다**. `mdp.object_is_lifted` 는 파지를 요구하지 않고 z 만 보므로, 팔뚝·손등처럼
그리퍼가 아닌 부위로 떠받쳐도 만점이 나온다.

여기서는 컵에 가장 가까운 **링크가 무엇인지**를 시계열로 세어 그 가설을 확정하거나 기각한다.
"그리퍼로 잡았다"면 최근접 링크가 손가락이어야 하고, "얹었다"면 팔뚝/손등이 나온다.

실행:
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_lift_left_policy_contact.py \
        --checkpoint <path.pth>
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import openarm.tasks  # noqa: F401
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

TASK = "open-grip_l_grasp_sensor"


def main() -> None:
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(TASK, "rl_games_cfg_entry_point")

    env = gym.make(TASK, cfg=env_cfg)
    raw = env.unwrapped
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math_inf := float("inf"))
    clip_act = agent_cfg["params"]["env"].get("clip_actions", math_inf)
    wrapped = RlGamesVecEnvWrapper(env, args.device, clip_obs, clip_act)

    vecenv.register("IsaacRlgWrapper", lambda cfg_name, n, **kw: RlGamesGpuEnv(cfg_name, n, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})

    agent_cfg["params"]["config"]["env_info"] = wrapped.get_number_of_agents and {
        "observation_space": wrapped.observation_space,
        "action_space": wrapped.action_space,
        "agents": 1,
    }
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(args.checkpoint)
    agent.reset()

    robot = raw.scene["robot"]
    obj = raw.scene["object"]
    ee = raw.scene["ee_frame"]
    origins = raw.scene.env_origins
    left = [(i, n) for i, n in enumerate(robot.body_names) if n.startswith(("l_hl_", "l_al_"))]
    idx = [i for i, _ in left]
    names = [n for _, n in left]
    grip_ids, _ = robot.find_joints(P.GRIPPER_JOINT_NAMES, preserve_order=True)

    def _tensor(o):
        # RlGamesVecEnvWrapper 는 {'obs': tensor} 를 준다. player 는 텐서를 기대한다.
        return o["obs"] if isinstance(o, dict) else o

    obs = _tensor(wrapped.reset())
    # ★play.py 와 같은 준비 절차. 이게 없으면 player 가 배치를 1개로 보고
    #   (1, num_envs*obs_dim) 로 flatten 해 행렬곱이 깨진다.
    agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()
    nearest_count: dict[str, int] = {n: 0 for n in names}
    lifted_steps = 0
    held_steps = 0
    total = 0
    grip_open_when_lifted = []
    tcp_when_lifted = []

    for _ in range(args.steps):
        with torch.inference_mode():
            act = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
        obs, _, _, _ = wrapped.step(act)
        obs = _tensor(obs)

        cup = obj.data.root_pos_w - origins
        lifted = cup[:, 2] > P.MINIMAL_LIFT_HEIGHT
        tcp_w = ee.data.target_pos_w[:, 0, :] - origins
        held = lifted & ((tcp_w - cup).norm(dim=-1) < P.GRASP_MAX_EE_DISTANCE)
        held_steps += int(held.sum())
        total += int(lifted.numel())
        lifted_steps += int(lifted.sum())
        if not bool(lifted.any()):
            continue
        pos = robot.data.body_pos_w[:, idx, :] - origins.unsqueeze(1)
        d = (pos - cup.unsqueeze(1)).norm(dim=-1)          # (E, L)
        near = d.argmin(dim=-1)
        for e in torch.nonzero(lifted).flatten().tolist():
            nearest_count[names[int(near[e])]] += 1
        tcp = ee.data.target_pos_w[:, 0, :] - origins
        tcp_when_lifted.append(float((tcp - cup).norm(dim=-1)[lifted].mean()))
        grip_open_when_lifted.append(float(robot.data.joint_pos[:, grip_ids[0]][lifted].mean()))

    print("\n=== 리프트 판정 중 컵에 가장 가까운 링크 ===")
    print(f"  z 만 보는 판정(레퍼런스): {lifted_steps / max(total, 1):.1%}")
    print(f"  쥐고 있음까지 요구(신규):   {held_steps / max(total, 1):.1%}"
          f"   ← 이 정책의 처내기가 새 게이트로 얼마나 무효화되는가")
    ranked = sorted(nearest_count.items(), key=lambda kv: -kv[1])
    shown = sum(v for _, v in ranked) or 1
    for n, c in ranked[:8]:
        if c == 0:
            continue
        kind = "그리퍼" if "gripper" in n else "팔"
        print(f"  {n:<28} {c / shown:6.1%}  ({kind})")
    if tcp_when_lifted:
        print(f"\n  리프트 중 TCP–컵 거리 평균 {sum(tcp_when_lifted) / len(tcp_when_lifted) * 1e3:.1f} mm")
        print(f"  리프트 중 그리퍼 개도 평균 {sum(grip_open_when_lifted) / len(grip_open_when_lifted) * 1e3:.1f} mm "
              f"(닫힘 0 ~ 열림 {P.GRIPPER_OPEN_POS * 1e3:.0f})")
        print("  → 최근접이 손가락이 아니고 TCP 가 멀면 **그리퍼가 아닌 부위로 떠받친 것**이다.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
