#!/usr/bin/env python3
"""Tesollo 시너지(eigengrasp) basis 렌더 검증.

학습 전 시각 검증용: 각 PC 스윕이 만드는 손 자세를 카메라로 캡처해 PNG 저장.
env 의 action→진행도→관절 target 경로(compute_synergy_progress_targets +
open↔FULL_GRIP lerp)를 그대로 재현하므로 학습에서 손이 취할 자세와 동일하다.

실행 (server):
  ./isaaclab.sh -p ../hdgp/scripts/pca/render_synergy_sweep.py \
      --headless --enable_cameras --out /tmp/synergy_sweep
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default="/tmp/synergy_sweep")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg


def _force_local_openarm() -> None:
    hdgp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src = os.path.join(hdgp, "source", "openarm")
    if src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)


_force_local_openarm()

from openarm.tesollo.right.grasp_v2.grasp_right_env_cfg import GraspRightEnvCfg  # noqa: E402
from openarm.tesollo.right.grasp_v2.finger_action_utils import (  # noqa: E402
    compute_synergy_progress_targets,
)
from openarm.tesollo.right.grasp_v2.tesollo_hand_synergy import (  # noqa: E402
    HAND_SYNERGY_BASIS,
    HAND_SYNERGY_ANCHOR,
    HAND_SYNERGY_COEFF_MINS,
    HAND_SYNERGY_COEFF_MAXS,
)
from openarm.tesollo.right.grasp_v2.grasp_right_preset import (  # noqa: E402
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
    RIGHT_ARM_START_POSE,
    HAND_APPROACH_POSE,
    HAND_FULL_GRIP_POSE,
)


def main() -> None:
    os.makedirs(args_cli.out, exist_ok=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60.0, device="cuda:0"))

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

    env_cfg = GraspRightEnvCfg()
    robot_cfg = env_cfg.robot_cfg.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    cam = Camera(CameraCfg(
        prim_path="/World/cam",
        width=960, height=720,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(),
    ))

    sim.reset()

    arm_ids = [robot.joint_names.index(n) for n in RIGHT_ARM_JOINT_NAMES]
    hand_ids = [robot.joint_names.index(n) for n in RIGHT_HAND_JOINT_NAMES]
    dev = robot.device

    B = torch.tensor(HAND_SYNERGY_BASIS, dtype=torch.float32, device=dev)
    A = torch.tensor(HAND_SYNERGY_ANCHOR, dtype=torch.float32, device=dev)
    mn = torch.tensor(HAND_SYNERGY_COEFF_MINS, dtype=torch.float32, device=dev)
    mx = torch.tensor(HAND_SYNERGY_COEFF_MAXS, dtype=torch.float32, device=dev)
    open_pose = torch.tensor(HAND_APPROACH_POSE, dtype=torch.float32, device=dev)
    grip_pose = torch.tensor(HAND_FULL_GRIP_POSE, dtype=torch.float32, device=dev)
    lims = robot.data.soft_joint_pos_limits[0, hand_ids]

    def hand_q(action5: list) -> torch.Tensor:
        a = torch.tensor([action5], dtype=torch.float32, device=dev)
        p = compute_synergy_progress_targets(a, B, A, mn, mx, open_pose, grip_pose)
        q = torch.lerp(open_pose.unsqueeze(0), grip_pose.unsqueeze(0), p)
        return q.clamp(lims[:, 0].unsqueeze(0), lims[:, 1].unsqueeze(0))[0]

    def set_pose(hand20: torch.Tensor) -> None:
        q = robot.data.default_joint_pos.clone()
        q[0, arm_ids] = torch.tensor(RIGHT_ARM_START_POSE, dtype=torch.float32, device=dev)
        q[0, hand_ids] = hand20
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        # PD target 도 동일하게 유지 — write_data_to_sim 없으면 sim에 미반영!
        robot.set_joint_position_target(q)
        robot.write_data_to_sim()

    # 카메라: 한 스텝 돌려 palm 위치 확보 후 손 정면·측면에서 촬영
    set_pose(open_pose)
    sim.step()
    robot.update(sim.get_physics_dt())
    palm_idx = robot.body_names.index("r_hl_palm")
    palm = robot.data.body_pos_w[0, palm_idx].cpu().numpy()

    poses = [
        ("00_open_action_all_-1", [-1, -1, -1, -1, -1]),
        ("01_neutral_action_0", [0, 0, 0, 0, 0]),
        ("02_PC1_power_+1", [1, -1, -1, -1, -1]),
        ("03_PC2_distal_+1", [-1, 1, -1, -1, -1]),
        ("04_PC3_reshape_+1", [-1, -1, 1, -1, -1]),
        ("05_PC4_thumb_+1", [-1, -1, -1, 1, -1]),
        ("06_PC5_fine_+1", [-1, -1, -1, -1, 1]),
        ("07_full_close_all_+1", [1, 1, 1, 1, 1]),
        ("08_PC1_half_+0", [0, -1, -1, -1, -1]),
    ]
    views = {
        "front": palm + np.array([0.45, 0.0, 0.1]),
        "side":  palm + np.array([0.05, -0.45, 0.15]),
        "top":   palm + np.array([0.05, -0.05, 0.5]),
    }

    try:
        from PIL import Image
    except ImportError:
        Image = None

    for name, act in poses:
        q = hand_q(act)
        set_pose(q)
        for _ in range(30):   # 안정화 + 렌더 워밍업
            set_pose(q)
            sim.step()
            robot.update(sim.get_physics_dt())
        for vname, eye in views.items():
            cam.set_world_poses_from_view(
                torch.tensor([eye], dtype=torch.float32, device=dev),
                torch.tensor([palm], dtype=torch.float32, device=dev),
            )
            for _ in range(4):
                sim.step()
                cam.update(sim.get_physics_dt())
            rgb = cam.data.output["rgb"][0].cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            path = os.path.join(args_cli.out, f"{name}_{vname}.png")
            if Image is not None:
                Image.fromarray(rgb[..., :3]).save(path)
            else:
                np.save(path.replace(".png", ".npy"), rgb)
        print(f"[SWEEP] {name} 저장 완료")

    print(f"[DONE] {args_cli.out}")
    simulation_app.close()


if __name__ == "__main__":
    main()
