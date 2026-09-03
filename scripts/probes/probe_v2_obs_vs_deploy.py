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

"""sim env 의 actor obs 와 **배포 코어**(`sim2real/scripts/left_policy_core`)가 만든
obs 를 같은 상태에서 항목별로 대조한다.

★★왜 필요한가 (09.03). 실기에서 정책이 컵을 수평으로 쓸고 지나갔다. 실기 추종
  지연을 뺀 **완전추종 오프라인 롤아웃도 똑같이** 쓸고 지나갔으므로(수평 136 mm
  vs 수직 43 mm, 게이트 0회), 원인은 플랜트가 아니라 **정책이 받는 입력**이다.
  sim 은 같은 체크포인트로 ⑤ 99.6% 를 낸다 — 그러면 두 obs 가 다르다는 뜻이다.

  49D 를 통째로 비교하면 "다르다"까지밖에 못 간다. 학습 obs 는 10개 항의 연결이라
  **항 경계로 잘라** 어느 항이 어긋나는지까지 짚는다. 슬롯 지도는 `env.yaml` 의
  observations.policy 순서가 진실원천이다.

실행:
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_v2_obs_vs_deploy.py \
        --checkpoint <path.pth> --steps 3
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="open-grip_l_grasp_sensor_v2")
parser.add_argument("--steps", type=int, default=3, help="대조할 스텝 수")
parser.add_argument("--height_only", action="store_true",
                    help="obs 대조 대신 sim 의 TCP·손끝 높이만 스텝마다 찍는다 "
                         "— 정책이 판을 긁는지 직접 본다")
parser.add_argument("--run", type=str,
                    default="/home/user/rl_ws/sim2real/logs/policy/left_v2E29",
                    help="배포 코어가 읽는 런 dump 디렉터리")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.num_envs = 1

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

SIM2REAL = Path("/home/user/rl_ws/sim2real")
sys.path.insert(0, str(SIM2REAL / "scripts"))

# ★슬롯 지도는 **런 dump 에서** 만든다 — 트랙마다 항이 다르다(v2 49D · fab 45D).


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg)
    raw = env.unwrapped

    inf = float("inf")
    wrapped = RlGamesVecEnvWrapper(
        env, args.device,
        agent_cfg["params"]["env"].get("clip_observations", inf),
        agent_cfg["params"]["env"].get("clip_actions", inf))
    vecenv.register("IsaacRlgWrapper",
                    lambda cfg_name, n, **kw: RlGamesGpuEnv(cfg_name, n, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["config"]["env_info"] = {
        "observation_space": wrapped.observation_space,
        "action_space": wrapped.action_space, "agents": 1}
    hz = int(agent_cfg["params"]["config"].get("horizon_length", 24))
    agent_cfg["params"]["config"]["minibatch_size"] = 1 * hz

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(args.checkpoint)
    agent.reset()

    robot = raw.scene["robot"]
    obj = raw.scene["object"]
    ee_cfg = SceneEntityCfg("ee_frame")
    ee_cfg.resolve(raw.scene)

    from left_policy_core import LeftPolicyCore, LeftSensors  # noqa: E402

    def _t(o):
        return o["obs"] if isinstance(o, dict) else o

    obs = _t(wrapped.reset())
    agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    origin = raw.scene.env_origins[0]
    goal_cmd = raw.command_manager.get_command("object_pose")[0].detach().cpu().numpy()
    print(f"[probe] goal 명령 (root 기준) {np.round(goal_cmd, 4).tolist()}", flush=True)

    from left_obs_builder import segments_from_run  # noqa: E402
    seg = segments_from_run(Path(args.run) / "params/env.yaml")
    slots, off = [], 0
    for nm, dim in seg:
        slots.append((nm, off, off + dim))
        off += dim
    print(f"[probe] 레이아웃 {off}D · {[n for n, _ in seg]}", flush=True)

    core = LeftPolicyCore(
        policy=lambda o: np.zeros(7), fabric=None,
        run_env_yaml=Path(args.run) / "params/env.yaml",
        run_agent_yaml=Path(args.run) / "params/agent.yaml",
        goal7=np.asarray(goal_cmd, dtype=np.float64),
        urdf_path=Path("/home/user/rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf"))
    # fabric 은 obs 조립에 관여하지 않는다 — 팔 목표만 만든다. 항등으로 둔다.
    core.fabric = lambda palm6, n=0: np.asarray(core.home, dtype=np.float64)

    names = list(robot.joint_names)
    arm_i = [names.index(f"l_aj_{k}") for k in range(1, 8)]
    grip_i = names.index("l_hj_gripper_1")

    with torch.inference_mode():
        for step in range(args.steps):
            q = robot.data.joint_pos[0].detach().cpu().numpy()
            qd = robot.data.joint_vel[0].detach().cpu().numpy()
            cup = (obj.data.root_pos_w[0] - origin).detach().cpu().numpy()
            cupq = obj.data.root_quat_w[0].detach().cpu().numpy()

            mine = core.step(LeftSensors(
                arm_q=q[arm_i].astype(np.float64), arm_qd=qd[arm_i].astype(np.float64),
                grip_q=float(q[grip_i]), grip_qd=float(qd[grip_i]),
                cup_pos=cup.astype(np.float64), cup_quat=cupq.astype(np.float64))).obs
            sim = obs[0].detach().cpu().numpy()

            if args.height_only:
                pz = core.fk.poses(q[arm_i].astype(np.float64),
                                   float(q[grip_i]), float(q[grip_i]))
                fmin = min(pz.finger_l_pos[2], pz.finger_r_pos[2])
                print(f" [{step:3d}] TCP z {pz.tcp_pos[2]:.3f} (판 위 "
                      f"{(pz.tcp_pos[2] - 0.200) * 1000:6.1f} mm) · 손끝 최저 "
                      f"{(fmin - 0.200) * 1000:6.1f} mm · 컵 z "
                      f"{cup[2]:.3f}", flush=True)
            else:
                print(f"\n=== step {step} ===  컵 {np.round(cup, 4).tolist()}", flush=True)
                print(f"{'항':>22} {'슬롯':>9}  {'최대차':>9}   sim / 배포 (최대차 슬롯)")
                for nm, a, b in slots:
                    d = np.abs(sim[a:b] - mine[a:b])
                    k = int(d.argmax())
                    flag = "  ★" if d.max() > 1e-3 else ""
                    print(f"{nm:>22} [{a:2d}:{b:2d}] {d.max():9.5f}   "
                          f"{sim[a+k]:+.4f} / {mine[a+k]:+.4f}{flag}")

            act = agent.get_action(obs, is_deterministic=True)
            obs, _, _, _ = wrapped.step(act)
            obs = _t(obs)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
