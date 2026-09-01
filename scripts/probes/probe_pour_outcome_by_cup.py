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

"""pour 결과를 **컵 종류별로** 가른다. 학습을 끊지 않고 체크포인트로 잰다.

왜 필요한가. TFEvents 의 `outcome/*_at_done` 은 전 env 평균이라 "어떤 컵이 못 하는지"
를 못 본다. 뱅크 실측상 파지 품질이 컵마다 크게 다르다(≥4지 s115/s120/shaker 100%
vs s100 40%). 전량 이송(bead_at_done → 1.0)이 목표라면 **어느 컵이 깎아먹는지**가
바로 다음 행동을 정한다.

`_warm_env_spec`(env → 컵 스펙 인덱스, 스폰과 같은 배정)으로 갈라 done 시점 값을 모은다.

실행:
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_pour_outcome_by_cup.py \\
        --checkpoint <run>/nn/open-tesol_r_pour_sensor-lstm.pth \\
        --num_envs 64 --episodes 8 --headless
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_pour_sensor-play-lstm")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=64, help="컵 종류 수의 배수로 둘 것")
parser.add_argument("--episodes", type=int, default=8, help="env 당 목표 에피소드 수")
parser.add_argument("--max_steps", type=int, default=6000)
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
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import openarm  # noqa: E402
_EXPECTED = str((_HDGP / "source/openarm/openarm").resolve())
if not str(Path(openarm.__file__).resolve()).startswith(_EXPECTED + os.sep):
    raise SystemExit(f"openarm 이 저장소 밖에서 왔다: {openarm.__file__}")
import openarm.tasks  # noqa: F401,E402

from run_cfg_restore import restore_run_cfg_if_available  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    resume = Path(args_cli.checkpoint).expanduser().resolve()
    if not resume.is_file():
        raise SystemExit(f"체크포인트가 없다: {resume}")
    agent_cfg = restore_run_cfg_if_available(
        env_cfg, agent_cfg, resume_path=str(resume), workspace_root=str(_HDGP.parent))
    env_cfg.seed = agent_cfg["params"]["seed"]
    env_cfg.scene.num_envs = args_cli.num_envs   # ★복원 뒤에 강제

    device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_act = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RlGamesVecEnvWrapper(env, device, clip_obs, clip_act,
                               agent_cfg["params"]["env"].get("obs_groups"),
                               agent_cfg["params"]["env"].get("concate_obs_groups", True))
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(resume)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(str(resume))
    agent.reset()

    raw = env.unwrapped
    while hasattr(raw, "env"):
        raw = raw.env.unwrapped

    from openarm.agnostic.modules import object_bank as _ob
    bank = _ob.get(raw.cfg.object_bank)
    names = [sp.id for sp in bank.specs]
    n_cup = len(names)

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    # env → 컵 스펙 (스폰과 같은 배정). warm 경로가 채운 값을 그대로 쓴다.
    spec = raw._warm_env_spec
    if spec is None:
        spec = torch.tensor(bank.assign_indices(raw.num_envs), device=raw.device)
    spec = spec.long()
    print(f"[CUP] env→컵 배정: {[names[i] for i in spec[:n_cup].tolist()]} … (반복)", flush=True)

    acc = {k: torch.zeros(n_cup, device=raw.device) for k in ("bead", "spill", "mouth", "n")}
    ep_done = torch.zeros(raw.num_envs, device=raw.device)

    for step in range(args_cli.max_steps):
        with torch.inference_mode():
            action = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            obs, _, dones, _ = env.step(action)
            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                for h in agent.states:
                    h[:, dones, :] = 0.0
        d = dones.to(raw.device).bool().reshape(-1)
        if d.any():
            ids = d.nonzero(as_tuple=False).reshape(-1)
            s = spec[ids]
            acc["bead"].index_add_(0, s, raw._last_done_bead[ids])
            acc["spill"].index_add_(0, s, raw._last_done_spill[ids])
            acc["mouth"].index_add_(0, s, raw._last_done_mouth_xy[ids])
            acc["n"].index_add_(0, s, torch.ones_like(s, dtype=torch.float))
            ep_done[ids] += 1
        if step % 600 == 0:
            print(f"[CUP] step {step}/{args_cli.max_steps} · 에피소드 "
                  f"{int(ep_done.min())}~{int(ep_done.max())} / {args_cli.episodes}", flush=True)
        if float(ep_done.min()) >= args_cli.episodes:
            break

    n = acc["n"].clamp(min=1.0)
    bead, spill, mouth = acc["bead"] / n, acc["spill"] / n, acc["mouth"] / n
    print(f"\n[CUP] 컵별 결과 (에피소드 {int(acc['n'].min())}~{int(acc['n'].max())}개씩)", flush=True)
    print(f"  {'컵':16s}{'n':>5s}{'bead':>8s}{'spill':>8s}{'잔량':>8s}{'mouth_xy mm':>13s}")
    order = sorted(range(n_cup), key=lambda i: -float(bead[i]))
    for i in order:
        rest = 1.0 - float(bead[i]) - float(spill[i])
        print(f"  {names[i]:16s}{int(acc['n'][i]):5d}{float(bead[i]):8.3f}"
              f"{float(spill[i]):8.3f}{rest:8.3f}{float(mouth[i])*1e3:13.1f}")
    tot = acc["n"].sum().clamp(min=1.0)
    print(f"  {'전체':16s}{int(acc['n'].sum()):5d}"
          f"{float((acc['bead'].sum()/tot)):8.3f}{float((acc['spill'].sum()/tot)):8.3f}"
          f"{1.0-float(acc['bead'].sum()/tot)-float(acc['spill'].sum()/tot):8.3f}"
          f"{float(acc['mouth'].sum()/tot)*1e3:13.1f}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
