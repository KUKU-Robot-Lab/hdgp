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

"""좌팔 모드에서 **받는 컵이 어디에 놓이는지**를 고정 모드와 나란히 잰다.

왜. 09.02 실측: `left_arm_action_enable=True` 인 두 런 모두 target 컵 입구가
(0.042, 0.224, 0.674) 로, 12D 런의 (0.298, 0.022, 0.406) 에서 3D 로 ~40cm 어긋났다.
우팔이 닿을 수 없어 palm 클램프가 55% 포화되고 접근을 아예 못 배운다.
포트 코드(추종 수식·body 인덱스·DOF 인덱스·IK 호출)는 both/pour_sensor 와 동일하고
preset 상수도 자기일관적이므로, 어긋남은 **런타임**에서 생긴다.

그래서 정적 비교를 멈추고 층별로 찍는다:
  ① rest FK 로 계산한 TCP 목표 (base)      — 상수. 맞아야 정상
  ② 실제 왼손 body 의 pose (base)          — ①과 벌어지면 IK 가 못 따라간 것
  ③ 실제 좌팔 관절 vs LEFT_ARM_REST         — 벌어지면 리셋/지령이 안 먹은 것
  ④ 컵 pose (world, env0 origin 뺀 값)     — preset 고정배치와 대조
스텝을 진행시키며 찍어 **정착 과도**인지 **정상상태 오차**인지 구분한다.

실행:
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_pour_left_cup_place.py \\
        --num_envs 4 --steps 60 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_pour_sensor-play-lstm")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--every", type=int, default=10, help="이 스텝마다 한 줄 찍는다")
parser.add_argument("--bank", type=str, default="", help="warm 뱅크 HDF5 (비우면 cfg 기본)")
parser.add_argument("--arm_gains", type=str, default="", help="우팔 게인 프로필 kuka|r2s")
parser.add_argument("--left_arm", type=str, default="on", help="on|off — 좌팔 액션 모드")
parser.add_argument("--out", type=str, default="", help="주면 좌팔/컵 근접 PNG 저장")
parser.add_argument("--grip_q", type=float, default=-1.0,
                    help="좌 그리퍼 관절 목표[m]. 0=닫힘 · 0.044=완전개방(=rest). "
                         "음수면 cfg rest 그대로. 손가락 간격 = 2*(0.006+q).")
parser.add_argument("--drive_left", action="store_true",
                    help="좌팔 TCP 를 워크스페이스 한쪽으로 밀어 **컵이 따라오는지** 본다. "
                         "실물 컵(kinematic 해제)은 손가락 접촉으로만 붙잡히므로, "
                         "팔을 움직여 봐야 붙어오는지 미끄러지는지 알 수 있다.")
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

import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm  # noqa: E402
_EXPECTED = str((_HDGP / "source/openarm/openarm").resolve())
if not str(Path(openarm.__file__).resolve()).startswith(_EXPECTED + os.sep):
    raise SystemExit(f"openarm 이 저장소 밖에서 왔다: {openarm.__file__}")
import openarm.tasks  # noqa: F401,E402


def _fmt(v) -> str:
    return "(" + ", ".join(f"{float(x):+.4f}" for x in v) + ")"


def main() -> None:
    # ★`parse_env_cfg` 는 hydra `env.x=` 를 안 받는다 — 바꿀 것은 명시 인자로만.
    cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    cfg.left_arm_action_enable = (args_cli.left_arm.lower() == "on")
    if args_cli.bank:
        cfg.warm_state_paths = (str(Path(args_cli.bank).expanduser().resolve()),)
    if args_cli.arm_gains:
        cfg.arm_gain_profile = args_cli.arm_gains
    if args_cli.out:
        cfg.scene.shot_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/shot_cam", update_period=0.0,
            height=900, width=1200, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=42.0,
                                             clipping_range=(0.02, 20.0)))
    cfg.finalize_after_overrides()
    print(f"[LEFT] left_arm_action_enable={cfg.left_arm_action_enable} · "
          f"action_space={cfg.action_space} · local_z={cfg.left_cup_follow_local_z}", flush=True)

    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    if args_cli.grip_q >= 0.0:
        # ★그리퍼를 실제로 **닫는다**. rest(0.044)는 완전 개방이라 컵을 못 문다 —
        #   실물 컵이 팔을 따라오려면 손가락이 컵 몸통을 압착해야 한다(09.02).
        env.left_gripper_rest = torch.full_like(env.left_gripper_rest, args_cli.grip_q)
        _gap = 2.0 * (0.006 + args_cli.grip_q) * 1000.0
        print(f"[LEFT] 그리퍼 목표 q={args_cli.grip_q:.4f} → 손가락 간격 {_gap:.1f}mm "
              f"(개구 한계 84.5mm · 접촉높이 컵 지름 ≈68~78mm)", flush=True)

    from openarm.tesollo.right.pour_sensor.pour_right_preset import (
        LEFT_ARM_REST_JOINT_POS, LEFT_TARGET_CUP_POS_ENV_LOCAL)

    preset_cup = np.asarray(LEFT_TARGET_CUP_POS_ENV_LOCAL, dtype=float)
    print(f"[LEFT] preset 고정배치 컵(env-local) = {_fmt(preset_cup)}", flush=True)
    if getattr(env, "_left_ik", None) is not None:
        print(f"[LEFT] ① rest TCP 목표(base)      = "
              f"{_fmt(env._left_tcp_rest_pos_b[0])}", flush=True)
        print(f"[LEFT]   워크스페이스 min/max      = "
              f"{_fmt(env._left_tcp_min[0])} ~ {_fmt(env._left_tcp_max[0])}", flush=True)

    # 좌팔 OFF(12D)에서는 15D 전용 인덱스가 없다 — 팔 전체 인덱스로 대체한다.
    l_idx = getattr(env, "left_arm_only_dof_indices", None) or [
        i for i in env.left_arm_dof_indices
        if env.robot.joint_names[i].startswith("l_aj_")]
    rest_q = np.array([LEFT_ARM_REST_JOINT_POS.get(env.robot.joint_names[i], 0.0)
                       for i in l_idx], dtype=float)
    print(f"[LEFT] 좌팔 관절 {[env.robot.joint_names[i] for i in l_idx]}", flush=True)
    print(f"[LEFT] rest q = {_fmt(rest_q)}", flush=True)

    env.reset()
    act = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)
    if args_cli.drive_left and env.cfg.action_space > 12:
        act[:, 12:15] = torch.tensor([1.0, -1.0, 1.0], device=env.device)  # 대각으로 밀기
    hand0 = None

    from isaaclab.utils.math import subtract_frame_transforms
    for step in range(args_cli.steps + 1):
        # ★리셋 직후 몇 스텝은 매 스텝 찍는다 — 튕김이 어느 스텝에 생기는지 봐야
        #   "스폰 겹침"인지 "그리퍼가 닫히며 친 것"인지 갈린다(09.02).
        if step <= 12 or step % args_cli.every == 0:
            origin = env.scene.env_origins[0]
            hand_pos_b, _ = subtract_frame_transforms(
                env.robot.data.root_pos_w, env.robot.data.root_quat_w,
                env.robot.data.body_pos_w[:, env._left_hand_body_index],
                env.robot.data.body_quat_w[:, env._left_hand_body_index])
            q_now = env.robot.data.joint_pos[0, l_idx].cpu().numpy()
            cup_local = (env.left_target_cup.data.root_pos_w[0] - origin).cpu().numpy()
            dq = np.abs(q_now - rest_q)
            # ★컵이 손을 따라오는지 = 손 이동량 대비 컵 이동량. 붙어 있으면 두 값이 같다.
            if hand0 is None:
                hand0 = hand_pos_b[0].clone()
                cup0 = cup_local.copy()
            d_hand = float(torch.norm(hand_pos_b[0] - hand0)) * 1000.0
            d_cup = float(np.linalg.norm(cup_local - cup0)) * 1000.0
            # ★env 전체 분포. 4 env 로는 안 보이던 이탈이 2048 env 에서 나왔다(09.02):
            #   학습 로그 mouth_xy 4.37m = 받는 컵이 씬 밖. 소수 env 만 튕겨도 평균이 깨진다.
            _hand_w = env.robot.data.body_pos_w[:, env._left_hand_body_index]
            _d_all = torch.norm(env.left_target_cup.data.root_pos_w - _hand_w, dim=-1)
            _v_all = torch.norm(env.left_target_cup.data.root_lin_vel_w, dim=-1)
            _bad = int((_d_all > 0.20).sum())
            _gq = env.robot.data.joint_pos[:, env.left_gripper_dof_indices]
            _fL = env.robot.data.body_pos_w[:, env.robot.data.body_names.index(
                "l_hl_gripper_left_finger")]
            _fR = env.robot.data.body_pos_w[:, env.robot.data.body_names.index(
                "l_hl_gripper_right_finger")]
            _gap = torch.norm(_fL - _fR, dim=-1)
            # ★반드시 **그리퍼 로컬 프레임**에서 잰다. 월드 XY 로 재면 그리퍼가 기울어진
            #   만큼 투영돼, 완벽히 중앙 정렬된 컵도 수십 mm 어긋나 보인다(09.02 오진).
            #   턱 사이 축은 손가락을 잇는 방향(±y_local)이므로, 그 축 성분이 0 이어야
            #   대칭 파지다. z_local 은 물리는 깊이라 0 이 아니어도 된다.
            _bq = env.robot.data.body_quat_w[:, env._left_hand_body_index]
            _bp = env.robot.data.body_pos_w[:, env._left_hand_body_index]
            _cup_b = quat_apply_inverse(_bq, env.left_target_cup.data.root_pos_w - _bp)
            _mid_b = quat_apply_inverse(_bq, 0.5 * (_fL + _fR) - _bp)
            _jaw_b = quat_apply_inverse(_bq, _fL - _fR)          # 턱 축 방향(로컬)
            _rel = _cup_b - _mid_b
            _along_jaw = (_rel * (_jaw_b / _jaw_b.norm(dim=-1, keepdim=True))).sum(-1)
            print(f"[GRIP] step {step:>3} · q {float(_gq.median()):.5f} · "
                  f"간격 {float(_gap.median())*1000:5.1f}mm · "
                  f"컵(base로컬) ({float(_cup_b[:,0].median())*1000:+6.1f}, "
                  f"{float(_cup_b[:,1].median())*1000:+6.1f}, "
                  f"{float(_cup_b[:,2].median())*1000:+6.1f})mm · "
                  f"**턱축 어긋남 {float(_along_jaw.median())*1000:+6.1f}mm**", flush=True)
            print(f"[LEFT] step {step:>3} · 컵-손 거리 중앙 {float(_d_all.median())*1000:6.1f}mm · "
                  f"최대 {float(_d_all.max())*1000:8.1f}mm · **이탈(>200mm) {_bad}/{env.num_envs}** · "
                  f"컵속도 최대 {float(_v_all.max()):6.2f} m/s", flush=True)
            print(f"[LEFT]       ② 왼손(base) {_fmt(hand_pos_b[0])} · "
                  f"④ 컵(env-local) {_fmt(cup_local)} · "
                  f"손 이동 {d_hand:6.1f}mm · 컵 이동 {d_cup:6.1f}mm · "
                  f"**미끄럼 {abs(d_hand - d_cup):5.1f}mm** · |q-rest| {dq.max():.4f}rad",
                  flush=True)
        if args_cli.out and step in (0, 1, 5, args_cli.steps):
            import numpy as _np, imageio.v2 as _iio
            cam = env.scene["shot_cam"]
            tgt = env.left_target_cup.data.root_pos_w.clone()
            eye = tgt.clone()
            eye[:, 0] += 0.28
            eye[:, 1] += 0.22
            eye[:, 2] += 0.14
            cam.set_world_poses_from_view(eye, tgt)
            env.sim.render()
            cam.update(dt=0.0)
            rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(_np.uint8)
            _f = f"{args_cli.out}_step{step:03d}.png"
            _iio.imwrite(_f, rgb)
            print(f"[LEFT]   사진 저장 {_f}", flush=True)
        if step < args_cli.steps:
            env.step(act)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
