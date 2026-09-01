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

"""왼팔이 작업면과 겹치는지, 받는 컵이 어디에 오는지 잰다.

배치를 grasp_s2r 기준(작업면 원점)으로 되돌리면 왼팔이 테이블에 박힐 수 있다.
왼팔을 올리면 **받는 컵도 같이 움직인다** — 받는 컵 자세는 왼팔 FK 로 정해지기
때문이다(`_get_left_cup_fk_pose`). 그래서 둘을 **같이** 본다:

  ① 왼팔 각 링크의 최저 z (작업면 상면 z=0.2 대비 여유)
  ② 받는 컵 입구 위치와, 우팔 warm 파지 주둥이에서 본 거리
  ③ `--l_aj` 로 대안 자세를 즉시 시험 (재학습 없이 기하만 확인)

실행:
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_pour_left_arm_clear.py --headless
    ... --l_aj " -0.30,-0.60,0.20,0.90,0.50,-0.60,-0.90"
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_pour_sensor-play-lstm")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--l_aj", type=str, default="",
                    help="시험할 왼팔 7관절 'a,b,c,d,e,f,g' (rad). 비우면 현재 cfg 값")
parser.add_argument("--out", type=str, default="", help="주면 PNG 저장")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.out:
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


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    if args_cli.l_aj:
        vals = [float(v) for v in args_cli.l_aj.split(",")]
        if len(vals) != 7:
            raise SystemExit(f"--l_aj 는 7개여야 한다: {vals}")
        for i, v in enumerate(vals, start=1):
            cfg.robot_cfg.init_state.joint_pos[f"l_aj_{i}"] = v
        # ★리셋이 쓰는 좌팔 자세는 cfg 필드가 아니라 **모듈 상수**다
        #   (`pour_right_env.py` 가 `from .pour_right_preset import LEFT_ARM_REST_JOINT_POS`
        #   로 값을 이미 바인딩했으므로, preset 이 아니라 **env 모듈의 이름**을 갈아야 한다).
        import openarm.tesollo.right.pour_sensor.pour_right_env as _pe
        _pe.LEFT_ARM_REST_JOINT_POS = {
            **_pe.LEFT_ARM_REST_JOINT_POS,
            **{f"l_aj_{i}": v for i, v in enumerate(vals, start=1)},
        }
    if args_cli.out:
        cfg.scene.shot_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/shot_cam", update_period=0.0,
            height=960, width=1280, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0,
                                             clipping_range=(0.02, 20.0)),
        )
    env = gym.make(args_cli.task, cfg=cfg).unwrapped

    print(f"[LEFT] 작업면 usd={str(env.cfg.table_cfg.spawn.usd_path).split('/')[-1]} "
          f"pos={list(env.cfg.table_cfg.init_state.pos)} · 상면 z={env.cfg.table_surface_z}",
          flush=True)
    print(f"[LEFT] 붓는 컵 스폰 중심=({env.cfg.object_spawn_x_center}, "
          f"{env.cfg.object_spawn_y_center}) z={env.cfg.object_spawn_z:.4f}", flush=True)

    env.reset()
    zero = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)
    for _ in range(args_cli.steps):
        env.step(zero)

    names = list(env.robot.data.body_names)
    left = [(i, n) for i, n in enumerate(names) if n.startswith("l_")]
    pos = env.robot.data.body_pos_w - env.scene.env_origins.unsqueeze(1)   # env-local
    surf = float(env.cfg.table_surface_z)

    print(f"\n[LEFT] 왼팔 링크 최저 z (env-local) — 상면 {surf:.3f} 대비", flush=True)
    print(f"  {'링크':26s}{'z_min':>9s}{'여유mm':>9s}")
    worst = (1e9, "")
    for i, n in left:
        z = float(pos[:, i, 2].min())
        clr = (z - surf) * 1e3
        if clr < worst[0]:
            worst = (clr, n)
        flag = "  ← 관통" if clr < 0 else ""
        print(f"  {n:26s}{z:9.4f}{clr:9.1f}{flag}")
    print(f"  ★최저 여유 {worst[0]:.1f} mm ({worst[1]})", flush=True)

    # ★받는 컵은 로봇 링크가 아니라 별도 물체다 — 링크 여유만 보면 이 겹침을 놓친다.
    #   기울어져 있으면 최저점이 중심-오프셋보다 더 내려가므로 **회전을 적용해서** 잰다.
    from isaaclab.utils.math import quat_apply as _qa
    from openarm.agnostic.modules import object_bank as _ob
    _tgt_spec = _ob.spec_by_id(env.cfg.left_target_cup_spec)
    _lcq = env.left_target_cup.data.root_quat_w
    _down = torch.tensor([0.0, 0.0, -_tgt_spec.origin_offset_z], device=env.device)
    _bot_w = env.left_target_cup.data.root_pos_w + _qa(_lcq, _down.unsqueeze(0).expand(env.num_envs, -1))
    _bot = (_bot_w - env.scene.env_origins)[:, 2]
    print(f"\n[LEFT] 받는 컵 바닥 z {float(_bot.min()):.4f}~{float(_bot.max()):.4f} · "
          f"상면 {surf:.3f} 대비 여유 {float((_bot.min()-surf)*1e3):+.1f} mm"
          f"{'  ← 관통' if float(_bot.min()) < surf else ''}", flush=True)
    _tilt = torch.rad2deg(torch.acos(_qa(_lcq, torch.tensor([0.,0.,1.], device=env.device)
                                         .unsqueeze(0).expand(env.num_envs,-1))[:, 2].clamp(-1,1)))
    print(f"[LEFT] 받는 컵 기울기 {float(_tilt.mean()):.1f}° (0=직립)", flush=True)

    lc = env.left_target_cup.data.root_pos_w - env.scene.env_origins
    op = env._target_opening_w - env.scene.env_origins
    sp = env._source_pour_point_w - env.scene.env_origins
    print(f"\n[LEFT] 받는 컵 중심 {[round(float(v),4) for v in lc[0]]} · "
          f"입구 {[round(float(v),4) for v in op[0]]}", flush=True)
    d_xy = torch.norm((op - sp)[:, :2], dim=-1) * 1e3
    d_z = (sp[:, 2] - op[:, 2]) * 1e3
    print(f"[LEFT] 붓기점→입구  xy {float(d_xy.mean()):.1f} mm "
          f"({float(d_xy.min()):.1f}~{float(d_xy.max()):.1f}) · "
          f"z여유 {float(d_z.mean()):+.1f} mm (양수 = 소스가 입구 위)", flush=True)

    if args_cli.out:
        cam = env.scene["shot_cam"]
        eye = lc.clone(); eye[:, 0] += 1.1; eye[:, 1] += 0.9; eye[:, 2] += 0.7
        cam.set_world_poses_from_view(eye + env.scene.env_origins,
                                      lc + env.scene.env_origins)
        for _ in range(6):
            env.sim.render(); cam.update(dt=0.0)
        rgb = cam.data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)
        from PIL import Image
        for i in range(min(2, env.num_envs)):
            Image.fromarray(rgb[i]).save(f"{args_cli.out}_{i}.png")
            print(f"  저장 {args_cli.out}_{i}.png", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
