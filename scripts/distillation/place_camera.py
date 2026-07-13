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

"""증류 카메라를 GUI 에서 직접 옮겨 배치를 정한다 (대화형).

Isaac Sim 창이 뜨면 Stage 트리에서 `/World/envs/env_0/Camera` 를 고르고
이동/회전 기즈모(W / E 키)로 옮긴다. 움직일 때마다 터미널에 현재 pose 가
preset 에 붙여넣을 형식으로 찍히고, 같은 값이 파일로도 저장된다.

카메라가 실제로 보는 화면을 보려면 Isaac Sim 에서 뷰포트를 하나 더 열고
(Window → Viewport → Viewport 2) 그 뷰포트의 카메라를 위 Camera prim 으로 바꾼다.

시각 DR(텍스처)은 꺼지므로 textures.zip 없이 돈다.

사용:
  ./isaaclab.sh -p ../hdgp/scripts/distillation/place_camera.py \
      --task open-tesol_r_grasp_v2-distill

배치가 정해지면 저장된 CAMERA_POS / CAMERA_ROT 를 preset 에 붙여넣는다:
  source/openarm/openarm/tesollo/right/grasp_v2/grasp_right_preset.py
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Place the distillation camera interactively.")
parser.add_argument("--task", type=str, required=True, help="…-distill task id.")
parser.add_argument("--out", type=str, default="docs/camera_preview/camera_pose.txt",
                    help="pose 저장 파일 (hdgp 기준 상대경로).")
parser.add_argument("--steps_per_poll", type=int, default=10,
                    help="pose 를 다시 읽는 주기 (물리 스텝 수).")
parser.add_argument(
    "--mode", choices=["viewport", "gizmo"], default="viewport",
    help="viewport(기본): 뷰포트를 날아다니면 그 시점이 곧 카메라 배치가 된다. "
         "gizmo: Stage 트리에서 Camera prim 을 직접 기즈모로 옮긴다 "
         "(sim 이 도는 동안 이동 기즈모가 안 먹는 경우가 있다).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# GUI 로 띄운다 — 이 스크립트의 존재 이유다
args_cli.headless = False
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import pathlib  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

_HDGP_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDGP_ROOT / "source" / "openarm"))

import openarm.tasks  # noqa: F401,E402
from isaaclab.utils.math import (  # noqa: E402
    convert_camera_frame_orientation_convention,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

PLACE_NUM_ENVS = 1
# 이만큼 움직여야 "옮겼다"고 보고 다시 찍는다 (미세 떨림으로 로그가 도배되지 않게)
POS_EPS = 2e-3      # 2 mm
QUAT_EPS = 2e-3


def _format_pose(pos, rot) -> str:
    return (
        f"CAMERA_POS = [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]\n"
        f"CAMERA_ROT = [{rot[0]:.7f}, {rot[1]:.7f}, {rot[2]:.7f}, {rot[3]:.7f}]"
    )


def _viewport_pose_w(device):
    """활성 뷰포트 카메라의 월드 pose → (pos, quat_ros).

    USD 카메라는 -Z 를 바라보는 opengl 규약이라 ros 로 변환해야 한다.
    """
    import omni.usd
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf, UsdGeom

    viewport = get_active_viewport()
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(str(viewport.camera_path))
    if not prim or not prim.IsValid():
        raise RuntimeError(f"뷰포트 카메라 prim 을 찾을 수 없다: {viewport.camera_path}")

    xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
    translation = xform.ExtractTranslation()
    quat: Gf.Quatd = xform.ExtractRotationQuat()
    imaginary = quat.GetImaginary()

    pos = torch.tensor(
        [[translation[0], translation[1], translation[2]]],
        dtype=torch.float32, device=device,
    )
    quat_opengl = torch.tensor(
        [[quat.GetReal(), imaginary[0], imaginary[1], imaginary[2]]],
        dtype=torch.float32, device=device,
    )
    quat_ros = convert_camera_frame_orientation_convention(
        quat_opengl, origin="opengl", target="ros"
    )
    return pos, quat_ros


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, num_envs=PLACE_NUM_ENVS)

    if not env_cfg.distillation:
        raise ValueError(f"'{args_cli.task}' 는 증류 태스크가 아니다 (카메라가 없다).")

    env_cfg.enable_visual_dr = False       # 텍스처 없이 배치만 본다
    env_cfg.scene.num_envs = PLACE_NUM_ENVS
    # 이게 없으면 카메라 pose 를 초기화 때 한 번만 읽는다 (기본 False).
    # GUI 에서 prim 을 옮겨도 data.pos_w 가 preset 값 그대로라 이 스크립트가 무용지물이 된다.
    env_cfg.tiled_camera_cfg.update_latest_camera_pose = True

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env.reset()

    unwrapped = env.unwrapped
    camera = unwrapped._tiled_camera
    env_origin = unwrapped.scene.env_origins[0]

    out_path = _HDGP_ROOT / args_cli.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    if args_cli.mode == "viewport":
        print("뷰포트를 날아다니면 그 시점이 곧 카메라 배치가 된다.", flush=True)
        print("  - 마우스 우클릭 드래그 + WASD 로 이동, 우클릭 드래그로 시선 회전", flush=True)
        print("  - 지금 보고 있는 화면이 그대로 student 가 볼 화면이다", flush=True)
        print("  - 원하는 시점에서 멈추면 그 pose 가 아래에 찍힌다", flush=True)
    else:
        print("Stage 트리에서 /World/envs/env_0/Camera 를 골라 기즈모로 옮긴다.", flush=True)
        print("  W(이동) / E(회전). sim 이 도는 동안 이동이 안 먹으면 --mode viewport 를 쓸 것.",
              flush=True)
    print(f"움직일 때마다 pose 가 찍히고 {out_path} 에 저장된다.", flush=True)
    print("=" * 70, flush=True)

    zero_action = torch.zeros(
        PLACE_NUM_ENVS, unwrapped.num_actions, device=unwrapped.device
    )
    last_pos = None
    last_rot = None
    step = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(zero_action)
        step += 1

        if step % args_cli.steps_per_poll:
            continue

        if args_cli.mode == "viewport":
            # 뷰포트 시점을 그대로 증류 카메라에 심는다 → 보이는 화면 = student 가 볼 화면
            pos_w, rot_w = _viewport_pose_w(unwrapped.device)
            camera.set_world_poses(
                positions=pos_w, orientations=rot_w, convention="ros"
            )
            pos = (pos_w[0] - env_origin).cpu()
            rot = rot_w[0].cpu()
        else:
            # 기즈모로 prim 을 옮기면 TiledCamera 가 매 업데이트마다 pose 를 다시 읽는다
            pos = (camera.data.pos_w[0] - env_origin).cpu()   # env 로컬 좌표
            rot = camera.data.quat_w_ros[0].cpu()             # (w, x, y, z), ros

        # q 와 -q 는 같은 회전이지만 preset 표기를 w>=0 으로 통일한다
        if rot[0] < 0:
            rot = -rot

        moved = (
            last_pos is None
            or torch.norm(pos - last_pos) > POS_EPS
            or torch.norm(rot - last_rot) > QUAT_EPS
        )
        if not moved:
            continue

        last_pos, last_rot = pos, rot
        pose_text = _format_pose(pos.tolist(), rot.tolist())
        print(pose_text, flush=True)
        print("-" * 70, flush=True)
        out_path.write_text(
            f"# {args_cli.task} — GUI 배치 결과 (env 로컬 좌표, ros 규약)\n"
            f"# preset 의 CAMERA_POS / CAMERA_ROT 를 이 값으로 교체할 것\n"
            f"{pose_text}\n"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
