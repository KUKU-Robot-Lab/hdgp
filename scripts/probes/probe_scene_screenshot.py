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

"""세팅된 씬을 여러 각도에서 PNG 로 뽑는다 (학습 재개 전 눈으로 확인용).

env 1 개만 만들고, 카메라를 지정한 위치들에 놓아 리셋 직후 상태를 찍는다.

실행:
    PYTHONUNBUFFERED=1 ../IsaacLab/isaaclab.sh -p scripts/probes/probe_scene_screenshot.py \
        --out /tmp/scene
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-grip_l_grasp_sensor_ik")
parser.add_argument("--out", type=str, required=True, help="출력 PNG 접두사")
parser.add_argument("--settle", type=int, default=30, help="찍기 전 스텝 수")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import openarm.tasks  # noqa: F401,E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# (이름, eye, lookat) — 로봇 base link 원점 기준 world 좌표
VIEWS = [
    ("front", (2.30, 0.00, 0.75), (0.10, 0.05, 0.30)),
    ("iso", (1.80, 1.50, 1.25), (0.10, 0.05, 0.25)),
    ("side", (0.15, 2.10, 0.75), (0.15, 0.00, 0.30)),
    ("top", (0.22, 0.05, 2.10), (0.22, 0.05, 0.20)),
    ("closeup", (0.95, 0.35, 0.55), (0.36, 0.18, 0.31)),
]


def main() -> None:
    cfg = parse_env_cfg(args.task, num_envs=1)
    # ★카메라는 **씬 cfg 에 넣어** 만든다. 씬 밖에서 Camera(...) 로 만들면 sim 이 이미
    #   play 중이라 초기화 콜백이 다시 안 돌고 `_ALL_INDICES` 가 없다며 죽는다.
    cfg.scene.shot_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/shot_cam",
        update_period=0.0,
        height=1080,
        width=1920,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 20.0)),
    )
    # 디버그 마커(목표 pose·EE 프레임 축)가 씬을 가린다. 세팅 확인용 사진에서는 끈다.
    cfg.commands.object_pose.debug_vis = False
    if hasattr(cfg.scene, "ee_frame"):
        cfg.scene.ee_frame.debug_vis = False
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    cam = env.scene["shot_cam"]
    n_act = env.action_manager.total_action_dim
    for _ in range(args.settle):
        env.step(torch.zeros(env.num_envs, n_act, device=env.device))

    try:
        from PIL import Image
    except ImportError:
        Image = None
        import imageio.v3 as iio

    for name, eye, look in VIEWS:
        cam.set_world_poses_from_view(
            torch.tensor([eye], device=env.device),
            torch.tensor([look], device=env.device),
        )
        # 카메라를 옮긴 뒤 렌더가 따라오도록 몇 프레임 돌린다(1 프레임이면 이전 포즈가 남는다).
        for _ in range(4):
            env.sim.render()
            cam.update(dt=0.0)
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        path = f"{args.out}_{name}.png"
        if Image is not None:
            Image.fromarray(rgb).save(path)
        else:
            iio.imwrite(path, rgb)
        print(f"  저장 {path}  {rgb.shape[1]}x{rgb.shape[0]}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
