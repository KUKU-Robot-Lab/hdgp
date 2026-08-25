#!/usr/bin/env python3
"""팜 지령 z → **실제 턱 z** 전달함수. 파지 높이를 내려면 지령이 얼마여야 하는가.

왜
--
보상 지형 프로브(probe_reward_landscape)가 팜 지령 z ≈ 0.42 아래에서 턱-컵 거리가
188 → 419 mm 로 튀는 것을 보였다. 그리고 파지 지령을 정확히 줘도 턱이 144 mm 떨어져
있었다. 지령대로 안 간다는 뜻이고, 그러면 보상을 어떻게 고쳐도 소용이 없다.

액션 박스 z 바닥은 0.22 다. 파지에 필요한 지령이 그보다 낮으면 **파지 자세가 액션
공간 밖**이고, 정책은 원리적으로 도달할 수 없다.

⚠ 박스를 프로브용으로 넓혀서 잰다 — 현재 박스로만 재면 바닥에서 잘려 전달함수의
  아래쪽을 못 본다.
⚠ 각 z 를 **독립적으로** 잰다(env 하나에 하나씩, 같은 초기 상태에서 정착).
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=48)
parser.add_argument("--settle", type=int, default=80)
parser.add_argument("--z_lo", type=float, default=0.08)
parser.add_argument("--z_hi", type=float, default=0.52)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.gripper.left.grasp_sensor  # noqa: F401,E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P  # noqa: E402

TASK = "open-grip_l_grasp_sensor_fab"
PROBE_BOX = ((0.10, 0.70), (0.00, 0.55), (0.05, 0.60))


def main() -> None:
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.episode_length_s = 1.0e9
    for t in ("time_out", "object_dropping", "object_out_of_workspace"):
        setattr(cfg.terminations, t, None)
    cfg.curriculum.adr = None
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()
    dev, n = env.device, env.num_envs
    act = env.action_manager.get_term("arm_action")
    lo = torch.tensor([b[0] for b in PROBE_BOX], device=dev)
    hi = torch.tensor([b[1] for b in PROBE_BOX], device=dev)
    act._box_center, act._box_half = 0.5 * (lo + hi), 0.5 * (hi - lo)

    # 컵은 치운다 — 팔의 도달 능력을 재는 것이다
    obj = env.scene["object"]
    st = obj.data.default_root_state.clone()
    st[:, :3] = env.scene.env_origins + torch.tensor([0.0, 0.0, -5.0], device=dev)
    obj.write_root_pose_to_sim(st[:, :7])
    obj.write_root_velocity_to_sim(torch.zeros_like(st[:, 7:]))

    robot = env.scene["robot"]
    ids = [robot.body_names.index(b) for b in P.GRIPPER_FINGER_BODIES]

    def jaw_z():
        p = robot.data.body_pos_w[:, ids, :]
        ap = matrix_from_quat(robot.data.body_quat_w[:, ids[0], :])[:, :, 2]
        p = p + (ap * P.JAW_PAD_OFFSET).unsqueeze(1)
        return (p.mean(dim=1) - env.scene.env_origins)[:, 2]

    zs = torch.linspace(args.z_hi, args.z_lo, n, device=dev)
    pts = torch.stack([torch.full((n,), P.CUP_SPAWN_X_CENTER, device=dev),
                       torch.full((n,), P.CUP_SPAWN_Y_CENTER, device=dev), zs], dim=-1)
    a = torch.zeros(n, env.action_manager.total_action_dim, device=dev)
    a[:, :3] = ((pts - act._box_center) / act._box_half).clamp(-1.0, 1.0)
    a[:, 6:] = -1.0
    for _ in range(args.settle):
        env.step(a)
    jz = jaw_z()

    target = P.GRASP_TARGET_Z
    box_lo, box_hi = P.PALM_BOX_Z
    print(f"\n파지 목표 턱 z = {target:.4f} · 현재 액션 박스 z = ({box_lo}, {box_hi})\n")
    print(f"{'지령 z':>9}{'실제 턱 z':>11}{'오차(mm)':>10}   박스 안?")
    best = None
    for i in range(n):
        cz, j = float(zs[i]), float(jz[i])
        inbox = box_lo <= cz <= box_hi
        err = (j - target) * 1000
        if best is None or abs(err) < abs(best[2]):
            best = (cz, j, err, inbox)
        print(f"{cz:>9.3f}{j:>11.4f}{err:>10.1f}   {'O' if inbox else '★밖'}")
    cz, j, err, inbox = best
    print(f"\n★파지 높이에 가장 가까운 지령 z = {cz:.3f} (턱 z {j:.4f}, 오차 {err:+.1f} mm)")
    print(f"  현재 박스 안인가: {'예' if inbox else '**아니다** — 파지 자세가 액션 공간 밖이다'}")
    if not inbox:
        print(f"  필요한 박스 하한 ≤ {cz:.3f} · 현재 {box_lo} → **{(box_lo-cz)*1000:.0f} mm 부족**")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
