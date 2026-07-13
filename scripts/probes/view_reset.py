"""에피소드 시작 상태(로봇 pregrasp + 물체 spawn)를 GUI 로 관찰한다.

정책 없이 무행동(action=0)으로 돌린다 → reset 자세가 그대로 유지되므로
"물체가 손 앞에 소환되는가" 를 눈으로 확인할 수 있다.

사용:
  # 기본 (여러 물체가 섞여 스폰)
  ./isaaclab.sh -p scripts/probes/view_reset.py --task open-tesol_r_grasp_v2-lstm --num_envs 9

  # 단일 물체만 (cup 의 side 접근 확인 등)
  ./isaaclab.sh -p scripts/probes/view_reset.py --task open-tesol_r_grasp_v2-lstm --object cup
  ./isaaclab.sh -p scripts/probes/view_reset.py --task open-tesol_r_grasp_v2-lstm --object small_8_cuboid

  # N 스텝마다 강제 리셋해 스폰 순간을 반복 관찰
  ./isaaclab.sh -p scripts/probes/view_reset.py --task open-tesol_r_grasp_v2-lstm --reset_every 120
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=9)
parser.add_argument("--object", type=str, default=None, help="이 물체만 스폰 (예: cup)")
parser.add_argument("--reset_every", type=int, default=0, help="N 스텝마다 강제 리셋 (0=안 함)")
parser.add_argument("--zoom", type=float, default=0.55, help="카메라 거리 (m)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)

# ---- 단일 물체 모드: MultiAsset 목록을 하나로 줄인다 ----
if args.object is not None:
    names = list(env_cfg.active_object_names)
    if args.object not in names:
        raise SystemExit(f"물체 '{args.object}' 없음. 예: {names[:8]} …")
    ti = names.index(args.object)
    env_cfg.cup_cfg.spawn.assets_cfg = [env_cfg.cup_cfg.spawn.assets_cfg[ti]]

# ---- 카메라: env_0 의 물체 spawn 지점을 클로즈업 ----
cx = env_cfg.object_spawn_x_center
cy = env_cfg.object_spawn_y_center
cz = env_cfg.object_spawn_z
z = float(args.zoom)
env_cfg.viewer.origin_type = "env"
env_cfg.viewer.env_index = 0
env_cfg.viewer.eye = (cx + z * 0.8, cy - z * 0.9, cz + z * 0.5)
env_cfg.viewer.lookat = (cx, cy, cz)

env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array").unwrapped
obs, _ = env.reset()

names = list(env.cfg.active_object_names)
print("\n" + "=" * 64)
print("에피소드 시작 상태 관찰 — 무행동(action=0)")
print("  task      :", args.task)
print("  envs      :", env.num_envs)
if args.object:
    print("  물체      :", args.object, "(단일)")
print("  카메라    : env_0 물체 spawn 지점 클로즈업")
print("  Ctrl+C 로 종료")
print("=" * 64 + "\n")

zero = torch.zeros(env.num_envs, env.cfg.num_actions, device=env.device)
step = 0
logged = False

while app.is_running():
    with torch.inference_mode():
        env.step(zero)
    step += 1

    # settle 직후 1회 수치 출력 (물체가 안착한 시점)
    if not logged and step == int(env.cfg.settle_steps) + 3:
        logged = True
        obj = env.object_pos
        palm = env.palm_center_pos
        tips = env.fingertip_pos
        d_palm = (palm - obj).norm(dim=-1)
        d_tip = (tips - obj.unsqueeze(1)).norm(dim=-1).min(dim=1).values
        goal = env.pregrasp_palm_pose_buf[:, :3] + env.scene.env_origins
        ik_err = (palm - goal).norm(dim=-1)
        print("%-4s %-16s %-9s %10s %10s %10s" % (
            "env", "물체", "접근", "palm~물체", "손끝~물체", "IK오차"))
        for i in range(min(env.num_envs, 12)):
            nm = names[int(env.object_idx[i].item())] if not args.object else args.object
            pose = "top-down" if int(env.palm_pose_id[i].item()) == 1 else "side"
            print("%-4d %-16s %-9s %8.1fcm %8.1fcm %8.1fcm" % (
                i, nm[:16], pose, d_palm[i] * 100, d_tip[i] * 100, ik_err[i] * 100))
        print("\n(IK오차 = pregrasp 목표 대비 실제 palm 위치. 크면 Fabrics IK 미도달)\n")

    if args.reset_every and step % args.reset_every == 0:
        env.reset()
        logged = False
        step = 0

env.close()
app.close()
