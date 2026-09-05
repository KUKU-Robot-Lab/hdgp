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

"""좌 그리퍼 파지 정책(v2E29)이 **컵을 어떻게 무는지**를 실측해 뽑는다.

왜. `right/pour_sensor` 의 받는 컵은 kinematic 텔레포트라 컵이 컵을 통과했다(09.02).
실물 강체로 바꾸려면 그리퍼가 실제로 쥐어야 하는데, 압착량을 임의로 정하면 근거가 없다.
검증된 파지 정책이 쓰는 값을 그대로 가져오는 것이 맞다(사용자 지시).

뽑는 것 — 리프트가 성립한 프레임에서:
  ① 그리퍼 관절값 `l_hj_gripper_1/2`  = 압착량. pour 의 `left_gripper_rest` 로 쓴다.
  ② 컵 pose 를 **그리퍼 base 로컬**로 변환 = 컵이 손 어디에 물리는가.
     pour 의 `left_cup_follow_local_z`(및 회전)의 근거가 된다.
  ③ 손가락 간격과 그 높이의 컵 지름 — 파지 대역 안인지 교차검증.

★환경변수는 `runE29.sh` 와 **같아야** 한다(특히 `HDGP_V2_VENDOR_GAINS=1` — 없으면
  그리퍼가 안 닫힌다). 러너 스크립트에서 export 한 뒤 이 파일을 부를 것.

실행:
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_extract_left_grasp_pose.py \\
        --checkpoint log/checkpoints_keep/v2E29_band80_ep3550.pth --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-grip_l_grasp_sensor_v2-play")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--lift_min_m", type=float, default=0.05,
                    help="테이블 위 이만큼 올라간 프레임만 '파지 성립'으로 본다")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math  # noqa: E402
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

from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_conjugate  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import openarm  # noqa: E402
_EXPECTED = str((_HDGP / "source/openarm/openarm").resolve())
if not str(Path(openarm.__file__).resolve()).startswith(_EXPECTED + os.sep):
    raise SystemExit(f"openarm 이 저장소 밖에서 왔다: {openarm.__file__}")
import openarm.tasks  # noqa: F401,E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    ck = Path(args_cli.checkpoint).expanduser().resolve()
    if not ck.is_file():
        raise SystemExit(f"체크포인트가 없다: {ck}")
    env_cfg.scene.num_envs = args_cli.num_envs

    device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_act = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RlGamesVecEnvWrapper(env, device, clip_obs, clip_act,
                               agent_cfg["params"]["env"].get("obs_groups"),
                               agent_cfg["params"]["env"].get("concate_obs_groups", True))
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(ck)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(str(ck))
    agent.reset()

    raw = env.unwrapped
    while hasattr(raw, "env"):
        raw = raw.env.unwrapped
    robot = raw.scene["robot"]
    obj = raw.scene["object"]

    names = list(robot.data.joint_names)
    g_idx = [names.index(n) for n in names if n.startswith("l_hj_gripper_")]
    bodies = list(robot.data.body_names)
    base_i = bodies.index("l_hl_gripper_base")
    fing = [bodies.index(b) for b in bodies if "gripper" in b and "finger" in b]
    print(f"[EXT] 그리퍼 관절 {[names[i] for i in g_idx]} · base body 'l_hl_gripper_base' · "
          f"손가락 {[bodies[i] for i in fing]}", flush=True)

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    best = []
    z0 = None                      # ★리프트는 **에피소드 시작 z 대비**로 판정한다.
    for step in range(args_cli.steps):
        with torch.inference_mode():
            action = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            obs, _, dones, _ = env.step(action)
            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                for h in agent.states:
                    h[:, dones, :] = 0.0

        # ★리프트 성립 = 시작 z 대비 상승 + **그리퍼가 실제로 닫혀 있음**.
        #   09.02 실패: 절대 z 문턱(0.25)을 쓰니 테이블에 놓인 컵(z≈0.292)이 전부 통과해
        #   완전개방(q=0.044) 프레임이 뽑혔다. 컵이 손에서 108mm 떨어져 있었다.
        z = obj.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
        if z0 is None:
            z0 = z.clone()
        gq_all = robot.data.joint_pos[:, g_idx].mean(dim=-1)
        near_hand = torch.norm(
            obj.data.root_pos_w - robot.data.body_pos_w[:, base_i], dim=-1)
        lifted = ((z - z0) > args_cli.lift_min_m) & (gq_all < 0.040) & (near_hand < 0.12)
        if not bool(lifted.any()):
            continue
        sel = torch.nonzero(lifted, as_tuple=False).flatten()
        bpos = robot.data.body_pos_w[sel, base_i]
        bquat = robot.data.body_quat_w[sel, base_i]
        cup_rel_pos = quat_apply_inverse(bquat, obj.data.root_pos_w[sel] - bpos)
        cup_rel_quat = quat_mul(quat_conjugate(bquat), obj.data.root_quat_w[sel])
        gq = robot.data.joint_pos[sel][:, g_idx]
        fgap = torch.norm(robot.data.body_pos_w[sel, fing[0]]
                          - robot.data.body_pos_w[sel, fing[1]], dim=-1)
        for k in range(sel.numel()):
            best.append((float(gq[k, 0]), float(gq[k].mean()), float(fgap[k]),
                         cup_rel_pos[k].tolist(), cup_rel_quat[k].tolist()))
        if len(best) > 400:
            break

    if not best:
        print("[EXT] 리프트 성립 프레임이 없다 — 환경변수(HDGP_V2_VENDOR_GAINS 등) 확인", flush=True)
        env.close()
        return

    import statistics as st
    q1 = st.median(b[0] for b in best)
    gap = st.median(b[2] for b in best)
    px = st.median(b[3][0] for b in best)
    py = st.median(b[3][1] for b in best)
    pz = st.median(b[3][2] for b in best)
    print(f"\n[EXT] 파지 성립 프레임 {len(best)}개 (중앙값)", flush=True)
    print(f"  ① 그리퍼 관절 q        = {q1:.5f} m   (0=닫힘 · 0.044=완전개방)", flush=True)
    print(f"  ③ 손가락 body 간격     = {gap*1000:.1f} mm", flush=True)
    print(f"  ② 컵 pos (base 로컬)   = ({px:+.4f}, {py:+.4f}, {pz:+.4f})", flush=True)
    print(f"     → pour 의 left_cup_follow_local_z 대응값 = {pz:.4f}", flush=True)
    _q = [st.median(b[4][i] for b in best) for i in range(4)]
    print(f"  ② 컵 quat(base 로컬)   = {[round(v, 4) for v in _q]}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
