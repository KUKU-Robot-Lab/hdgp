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

"""증류 카메라 배치 프리뷰 — student 가 실제로 보게 될 화면을 뽑는다.

실물 D435i 를 달기 전에 시뮬에서 시점을 먼저 정하기 위한 도구다.
pose 를 CLI 로 바꿔가며 렌더해 보고, 마음에 드는 값이 나오면 출력된
CAMERA_POS / CAMERA_ROT 를 preset 에 붙여넣으면 된다.

시각 DR(텍스처)은 끄고 돈다 — 배치를 보는 데 textures.zip 8.4GB 가 필요하진 않다.

사용:
  # preset 의 현재(placeholder) 배치 확인
  ./isaaclab.sh -p ../hdgp/scripts/distillation/preview_camera.py \
      --task open-tesol_r_grasp_v2-distill

  # 위치를 바꿔가며 탐색 (--look 을 주면 그 점을 바라보는 자세를 자동 계산)
  ./isaaclab.sh -p ../hdgp/scripts/distillation/preview_camera.py \
      --task open-tesol_r_grasp_v2-distill \
      --pos 0.9 -0.45 0.8 --look 0.27 -0.10 0.32
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Preview the distillation camera view.")
parser.add_argument("--task", type=str, required=True, help="…-distill task id.")
parser.add_argument("--pos", type=float, nargs=3, default=None,
                    help="카메라 위치 x y z (env 로컬 좌표). 생략 시 preset 값.")
parser.add_argument("--look", type=float, nargs=3, default=None,
                    help="바라볼 점 x y z. 주면 자세를 look-at 으로 계산한다.")
parser.add_argument("--rot", type=float, nargs=4, default=None,
                    help="카메라 자세 w x y z (ros). --look 과 함께 쓰지 말 것.")
parser.add_argument("--steps", type=int, default=30,
                    help="렌더 전 스텝 수 (물체가 안착할 시간).")
parser.add_argument("--out", type=str, default="docs/camera_preview",
                    help="이미지 출력 디렉토리 (hdgp 기준 상대경로).")
parser.add_argument(
    "--sweep", type=str, nargs="+", default=None,
    help="여러 배치를 한 번의 기동으로 비교. "
         "형식: '<name>=x,y,z:lx,ly,lz' (Isaac 기동에 1분 걸리므로 스윕이 훨씬 빠르다). "
         "예: A=1.05,-0.1,0.75:0.3,-0.1,0.32 B=0.8,-0.1,0.62:0.27,-0.1,0.36",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = True   # 카메라를 보려고 띄우는 스크립트다
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
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from openarm.distillation.camera import look_at_quat  # noqa: E402

PREVIEW_NUM_ENVS = 1


def _save_png(tensor_hwc, path: pathlib.Path) -> None:
    from PIL import Image

    array = (tensor_hwc.clamp(0.0, 1.0) * 255).to(torch.uint8).cpu().numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    Image.fromarray(array).save(path)


def _parse_sweep(specs: list[str]) -> list[tuple[str, list[float], list[float]]]:
    """'<name>=x,y,z:lx,ly,lz' → (name, pos, rot)."""
    parsed = []
    for spec in specs:
        try:
            name, geometry = spec.split("=", 1)
            pos_str, look_str = geometry.split(":", 1)
            pos = [float(v) for v in pos_str.split(",")]
            look = [float(v) for v in look_str.split(",")]
        except ValueError as exc:
            raise ValueError(
                f"--sweep 형식 오류: '{spec}' "
                "(기대: '<name>=x,y,z:lx,ly,lz')"
            ) from exc
        if len(pos) != 3 or len(look) != 3:
            raise ValueError(f"--sweep 좌표는 3개씩이어야 한다: '{spec}'")
        parsed.append((name, pos, look_at_quat(pos, look)))
    return parsed


def _project_object(env) -> tuple[float, float, bool]:
    """물체 중심을 이미지 픽셀로 투영 → (u, v, 프레임 안인가).

    "물체가 잘 보이나"를 눈대중하지 않기 위한 진단. 중앙(160, 90)에 가까울수록 좋다.
    """
    from isaaclab.utils.math import quat_apply_inverse

    unwrapped = env.unwrapped
    camera = unwrapped._tiled_camera

    obj_w = unwrapped.object_pos + unwrapped.scene.env_origins   # 월드 좌표
    cam_pos = camera.data.pos_w
    cam_quat = camera.data.quat_w_ros                            # ros: +z 전방, +y 아래

    obj_cam = quat_apply_inverse(cam_quat, obj_w - cam_pos)[0]   # 카메라 프레임
    depth = float(obj_cam[2])
    if depth <= 1e-6:
        return float("nan"), float("nan"), False                 # 카메라 뒤

    intrinsics = camera.data.intrinsic_matrices[0]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    u = fx * float(obj_cam[0]) / depth + cx
    v = fy * float(obj_cam[1]) / depth + cy

    inside = 0 <= u < env.unwrapped.cfg.img_width and 0 <= v < env.unwrapped.cfg.img_height
    return u, v, inside


def _capture(env, env_cfg, out_dir, name: str, pos, rot) -> None:
    """카메라를 옮기고 렌더 → PNG 저장 + 지표 출력."""
    camera = env.unwrapped._tiled_camera
    camera.set_world_poses(
        positions=torch.tensor([pos], device=env.unwrapped.device, dtype=torch.float32)
        + env.unwrapped.scene.env_origins,
        orientations=torch.tensor([rot], device=env.unwrapped.device, dtype=torch.float32),
        convention="ros",
    )

    obs, _ = env.reset()
    for _ in range(args_cli.steps):
        zero_action = torch.zeros(
            PREVIEW_NUM_ENVS, env.unwrapped.num_actions, device=env.unwrapped.device
        )
        obs, *_ = env.step(zero_action)

    rgb = obs["rgb"][0].permute(1, 2, 0)                    # (H, W, 3), 0~1
    depth = obs["img"][0].permute(1, 2, 0)                  # (H, W, 1), meters
    d_min, d_max = env_cfg.d_min, env_cfg.d_max
    depth_vis = (depth - d_min) / max(d_max - d_min, 1e-6)

    _save_png(rgb, out_dir / f"{name}_rgb.png")
    _save_png(depth_vis, out_dir / f"{name}_depth.png")

    valid = (depth > 0).float().mean().item()
    u, v, inside = _project_object(env)
    print(f"[{name}] pos={[round(v_, 3) for v_ in pos]}  "
          f"depth 유효 픽셀 {valid:.1%}  "
          f"물체 픽셀 ({u:.0f}, {v:.0f})/{env_cfg.img_width}x{env_cfg.img_height} "
          f"{'프레임 안' if inside else '⚠ 프레임 밖'}", flush=True)
    if valid < 0.2:
        print(f"  ⚠ [{name}] 유효 픽셀 부족 — 작업공간을 못 담거나 밴드가 안 맞는다.",
              flush=True)
    print(f"  CAMERA_POS = [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]", flush=True)
    print("  CAMERA_ROT = [{:.7f}, {:.7f}, {:.7f}, {:.7f}]".format(*rot), flush=True)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, num_envs=PREVIEW_NUM_ENVS)

    if not env_cfg.distillation:
        raise ValueError(f"'{args_cli.task}' 는 증류 태스크가 아니다 (카메라가 없다).")

    # 텍스처 없이 배치만 본다
    env_cfg.enable_visual_dr = False
    env_cfg.scene.num_envs = PREVIEW_NUM_ENVS
    # 기본 False 면 카메라 pose 를 초기화 때 한 번만 읽는다 → set_world_poses 로 옮겨도
    # camera.data 는 첫 배치를 그대로 반환하고, 물체 투영 진단이 전부 같은 값이 된다.
    env_cfg.tiled_camera_cfg.update_latest_camera_pose = True

    if args_cli.rot is not None and args_cli.look is not None:
        raise ValueError("--rot 과 --look 은 함께 쓸 수 없다 (자세를 두 번 정하게 된다).")

    if args_cli.sweep:
        placements = _parse_sweep(args_cli.sweep)
    else:
        pos = (args_cli.pos if args_cli.pos is not None
               else list(env_cfg.tiled_camera_cfg.offset.pos))
        if args_cli.look is not None:
            rot = look_at_quat(pos, args_cli.look)
        elif args_cli.rot is not None:
            rot = args_cli.rot
        else:
            rot = list(env_cfg.tiled_camera_cfg.offset.rot)
        placements = [(args_cli.task.replace("open-", ""), list(pos), list(rot))]

    # 첫 배치를 cfg 에 심어 카메라를 생성한다 (이후엔 set_world_poses 로 옮긴다)
    env_cfg.tiled_camera_cfg.offset.pos = list(placements[0][1])
    env_cfg.tiled_camera_cfg.offset.rot = list(placements[0][2])

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    out_dir = _HDGP_ROOT / args_cli.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print(f"해상도 {env_cfg.img_width}x{env_cfg.img_height} (D435i depth 16:9), "
          f"depth 밴드 {env_cfg.d_min}~{env_cfg.d_max} m", flush=True)
    for name, pos, rot in placements:
        _capture(env, env_cfg, out_dir, name, pos, rot)
    print("=" * 60, flush=True)
    print(f"이미지: {out_dir}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
