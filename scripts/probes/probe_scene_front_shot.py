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

"""임의의 DirectRLEnv 태스크 씬을 **고정 카메라**로 찍는다 — 두 트랙 배치 대조용.

카메라 위치를 하드코딩해 두 태스크를 같은 눈높이에서 찍는다. 다른 각도로 찍은
사진 두 장은 "배치가 같은지" 를 판정할 수 없다.

실행:
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_scene_front_shot.py \\
        --task open-sens_r_grasp_s2r-play-lstm --out /tmp/s2r
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--out", type=str, required=True, help="출력 PNG 접두사")
parser.add_argument("--settle", type=int, default=30, help="찍기 전 스텝 수")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
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

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm  # noqa: E402
_EXPECTED = str((_HDGP / "source/openarm/openarm").resolve())
if not str(Path(openarm.__file__).resolve()).startswith(_EXPECTED + os.sep):
    raise SystemExit(f"openarm 이 저장소 밖에서 왔다: {openarm.__file__}")
import openarm.tasks  # noqa: F401,E402

# (이름, eye, lookat) — env 원점 기준. **두 태스크에 동일하게 적용**한다.
VIEWS = [
    ("front", (1.90, 0.00, 0.55), (0.10, 0.00, 0.28)),
    ("side",  (0.15, 1.70, 0.60), (0.15, 0.00, 0.28)),
    ("iso",   (1.40, 1.20, 1.00), (0.10, 0.00, 0.25)),
    ("top",   (0.20, 0.00, 1.90), (0.20, 0.00, 0.20)),
]


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, num_envs=1)
    cfg.scene.env_spacing = 6.0        # 옆 env 가 화면에 안 들어오게
    cfg.scene.shot_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/shot_cam", update_period=0.0,
        height=960, width=1280, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.0,
                                         clipping_range=(0.02, 30.0)),
    )
    env = gym.make(args_cli.task, cfg=cfg).unwrapped

    tcfg = getattr(env.cfg, "table_cfg", None)
    if tcfg is not None:
        print(f"[SHOT] 작업면 usd={str(tcfg.spawn.usd_path).split('/')[-1]} "
              f"pos={list(tcfg.init_state.pos)}", flush=True)
    print(f"[SHOT] 로봇 base pos={list(env.cfg.robot_cfg.init_state.pos)}", flush=True)

    env.reset()
    n_act = int(env.cfg.action_space)
    for _ in range(args_cli.settle):
        env.step(torch.zeros(env.num_envs, n_act, device=env.device))

    cam = env.scene["shot_cam"]
    org = env.scene.env_origins[0]
    from PIL import Image
    for name, eye, look in VIEWS:
        e = torch.tensor(eye, device=env.device).unsqueeze(0) + org.unsqueeze(0)
        t = torch.tensor(look, device=env.device).unsqueeze(0) + org.unsqueeze(0)
        cam.set_world_poses_from_view(e, t)
        for _ in range(6):
            env.sim.render()
            cam.update(dt=0.0)
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        path = f"{args_cli.out}_{name}.png"
        Image.fromarray(rgb).save(path)
        print(f"  저장 {path}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
