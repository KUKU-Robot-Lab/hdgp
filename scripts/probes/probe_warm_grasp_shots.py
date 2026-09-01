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

"""warm 뱅크가 담는 **파지 자세를 컵 종류별로 촬영**한다.

왜 필요한가. 뱅크의 숫자(접촉 수·팁힘·컵-손 거리)는 "몇 개가 닿았나"는 말해주지만
**어떻게 잡았나**는 말해주지 않는다. 손가락이 감쌌는지, 끝만 걸쳤는지, 컵이 손가락
사이에 끼워져만 있는지는 그림으로 봐야 갈린다(09.01 pour 사고가 그랬다 —
지표는 멀쩡한데 영상에서 손가락이 벌어져 있었다).

`num_envs` 를 뱅크 크기와 같게 두면 `env_id % N` 배정이 **env i = 물체 i** 를 만든다.
정책을 굴려 파지가 안정된 뒤 env 별 카메라를 그 컵에 붙여 한 장씩 찍는다.

실행:
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_warm_grasp_shots.py \\
        --task open-sens_r_grasp_s2r-play-lstm --checkpoint <ckpt> \\
        --out /tmp/d3 --steps 250 --headless --enable_cameras
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-sens_r_grasp_s2r-play-lstm")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--out", type=str, required=True, help="출력 PNG 접두사")
parser.add_argument("--steps", type=int, default=250, help="촬영 전 정책 스텝 수")
parser.add_argument("--num_envs", type=int, default=8, help="뱅크 물체 수와 같게 둘 것")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=960)
parser.add_argument(
    "--cam_offset", type=str, default="0.34,0.30,0.16",
    help="컵 기준 카메라 위치 오프셋 'dx,dy,dz' [m]. 컵을 바라본다.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# ★경로는 isaaclab import 보다 먼저 — `import isaaclab_tasks` 가 확장 진입점을 훑으며
#   openarm 을 먼저 import 하면 다른 트리가 sys.modules 를 선점한다(수집기와 같은 규약).
import os  # noqa: E402
from pathlib import Path  # noqa: E402

_HDGP = Path(__file__).resolve().parents[2]
for _p in (str(_HDGP / "scripts/tools"), str(_HDGP / "source/openarm")):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
for _name in [m for m in sys.modules if m == "openarm" or m.startswith("openarm.")]:
    del sys.modules[_name]

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import openarm  # noqa: E402
_EXPECTED = str((_HDGP / "source/openarm/openarm").resolve())
if not str(Path(openarm.__file__).resolve()).startswith(_EXPECTED + os.sep):
    raise SystemExit(f"openarm 이 저장소 밖에서 왔다: {openarm.__file__}")
import openarm.tasks  # noqa: F401,E402

from run_cfg_restore import restore_run_cfg_if_available  # noqa: E402


def _save(rgb: np.ndarray, path: str) -> None:
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except ImportError:
        import imageio.v3 as iio
        iio.imwrite(path, rgb)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    resume = Path(args_cli.checkpoint).expanduser().resolve()
    if not resume.is_file():
        raise SystemExit(f"체크포인트가 없다: {resume}")

    agent_cfg = restore_run_cfg_if_available(
        env_cfg, agent_cfg, resume_path=str(resume), workspace_root=str(_HDGP.parent))
    env_cfg.seed = agent_cfg["params"]["seed"]
    # ★복원 **뒤에** 강제 — 덤프의 num_envs 가 되살아난다(수집기와 같은 함정).
    env_cfg.scene.num_envs = args_cli.num_envs

    # ★카메라는 씬 cfg 에 넣어 만든다. 씬 밖에서 Camera(...) 로 만들면 sim 이 이미
    #   play 중이라 초기화 콜백이 다시 안 돌고 `_ALL_INDICES` 가 없다며 죽는다.
    #   `{ENV_REGEX_NS}` 라 env 마다 한 대 → 8종을 한 번에 찍는다.
    env_cfg.scene.shot_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/shot_cam",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=32.0, clipping_range=(0.02, 20.0)),
    )

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
    names = [bank.specs[k].id for k in bank.assign_indices(raw.num_envs)]
    print(f"[SHOT] env→물체 배정: {list(enumerate(names))}", flush=True)

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    for step in range(args_cli.steps):
        with torch.inference_mode():
            action = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            obs, _, dones, _ = env.step(action)
            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                for h in agent.states:
                    h[:, dones, :] = 0.0

    # ---- 촬영 ----
    cam = raw.scene["shot_cam"]
    cup_w = raw.object.data.root_pos_w.clone()                    # (N, 3) world
    off = torch.tensor([float(v) for v in args_cli.cam_offset.split(",")],
                       device=raw.device, dtype=cup_w.dtype)
    eye = cup_w + off.unsqueeze(0)
    cam.set_world_poses_from_view(eye, cup_w)
    # 카메라를 옮긴 뒤 렌더가 따라오도록 몇 프레임 돌린다(1 프레임이면 이전 포즈가 남는다).
    for _ in range(6):
        raw.sim.render()
        cam.update(dt=0.0)

    tip = raw.contact_force_raw if hasattr(raw, "contact_force_raw") else None
    lift = (raw.object.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
            - float(raw.cfg.object_spawn_z))
    rgb_all = cam.data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)
    for i in range(raw.num_envs):
        path = f"{args_cli.out}_{i}_{names[i]}.png"
        _save(rgb_all[i], path)
        extra = ""
        if tip is not None:
            n_c = int((tip[i] > 0.1).sum())
            extra = f" · 접촉 {n_c} · 팁힘max {float(tip[i].max()):.2f}N"
        print(f"  저장 {path}  리프트 {float(lift[i])*1e3:+.0f}mm{extra}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
