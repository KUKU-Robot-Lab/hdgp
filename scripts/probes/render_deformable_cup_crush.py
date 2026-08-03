"""deformable cup 껍데기(shell)가 찌그러지는 과정을 렌더링해 영상으로 저장.

안쪽 반경 힘을 0→최대→0으로 램프해 패널이 접혀 들어갔다(변형/좌굴) 스프링으로
복원하는 과정을 카메라로 캡처, mp4/gif로 저장. 학습이 아니라 **자산 변형 메커니즘의
시각 시연**(fragile 컵이 실제로 찌그러짐을 눈으로 확인)이 목적이라, 잘 보이게 소프트 K 사용.

사용(로컬 GPU):
  cd /home/user/rl_ws/hdgp
  /home/user/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/render_deformable_cup_crush.py \
      --stiffness 0.3 --push-force 1.0 --out /path/out
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", type=str,
                    default="/home/user/rl_ws/hdgp/assets/cup/deformable_cup.usd")
parser.add_argument("--stiffness", type=float, default=0.3,
                    help="패널 스프링 강성(낮을수록 극적으로 찌그러짐)")
parser.add_argument("--damping", type=float, default=0.03)
parser.add_argument("--armature", type=float, default=1.0e-3)
parser.add_argument("--push-force", type=float, default=1.0, help="패널당 최대 안쪽 힘(N)")
parser.add_argument("--frames", type=int, default=180, help="총 프레임 수")
parser.add_argument("--out", type=str,
                    default="/tmp/claude-1000/-home-user-rl-ws/21fefa30-5de6-40a8-a088-f4d81dfb2222/scratchpad/cup_crush")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(
        dt=1.0 / 120.0, device=args.device, gravity=(0.0, 0.0, 0.0)))

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
    sim_utils.SphereLightCfg(intensity=30000.0, radius=0.1).func(
        "/World/KeyLight", sim_utils.SphereLightCfg(intensity=30000.0, radius=0.1),
        translation=(0.3, 0.3, 0.5))

    cup_cfg = ArticulationCfg(
        prim_path="/World/Cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args.usd,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.10)),
        actuators={
            "panels": ImplicitActuatorCfg(
                joint_names_expr=[".*"],  # single-hinge(revolute_*)·ring(edge_*) 모두 커버
                stiffness=args.stiffness,
                damping=args.damping,
                armature=args.armature,
                effort_limit=1.0e6,
            ),
        },
    )
    cup = Articulation(cup_cfg)

    cam_cfg = CameraCfg(
        prim_path="/World/Camera",
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0),
        offset=CameraCfg.OffsetCfg(
            pos=(0.28, 0.28, 0.28), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
    )
    cam = Camera(cam_cfg)
    # 컵 중심을 바라보게 카메라 배치
    cam_pos = torch.tensor([[0.30, 0.30, 0.26]], device=args.device)
    cam_target = torch.tensor([[0.0, 0.0, 0.13]], device=args.device)

    sim.reset()
    cam.set_world_poses_from_view(cam_pos, cam_target)

    n_bodies = cup.num_bodies
    body_names = cup.body_names
    panel_ids = [i for i, n in enumerate(body_names) if n.startswith("panel_")]
    forces = torch.zeros((1, n_bodies, 3), device=cup.device)
    torques = torch.zeros((1, n_bodies, 3), device=cup.device)

    os.makedirs(args.out, exist_ok=True)
    rgb_frames = []
    F = args.frames

    def apply_inward(scale: float) -> None:
        forces.zero_()
        pos_w = cup.data.body_pos_w[0]
        root_xy = cup.data.root_pos_w[0, :2]
        for i in panel_ids:
            radial = pos_w[i, :2] - root_xy
            nrm = torch.norm(radial)
            if nrm > 1e-6:
                inward = -radial / nrm
                forces[0, i, 0] = inward[0] * args.push_force * scale
                forces[0, i, 1] = inward[1] * args.push_force * scale

    for f in range(F):
        # 0~0.35: 램프 업(찌그러짐), 0.35~0.6: 유지(최대 crush), 0.6~1.0: 해제(복원)
        t = f / F
        if t < 0.35:
            scale = t / 0.35
        elif t < 0.6:
            scale = 1.0
        else:
            scale = max(0.0, 1.0 - (t - 0.6) / 0.4)
        apply_inward(scale)
        cup.set_external_force_and_torque(forces, torques)
        cup.write_data_to_sim()
        sim.step()
        cup.update(sim.get_physics_dt())
        cam.update(sim.get_physics_dt())
        rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        rgb_frames.append(rgb)
        if f % 30 == 0:
            q = cup.data.joint_pos.abs().max().item() * 180.0 / math.pi
            print(f"[render] frame {f}/{F} scale={scale:.2f} max|angle|={q:.1f}deg",
                  flush=True)

    mp4 = os.path.join(args.out, "cup_crush.mp4")
    gif = os.path.join(args.out, "cup_crush.gif")
    try:
        imageio.mimsave(mp4, rgb_frames, fps=30)
        print(f"[render] saved {mp4}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[render] mp4 실패({e}), gif로", flush=True)
    imageio.mimsave(gif, rgb_frames[::2], fps=15)
    print(f"[render] saved {gif}  ({len(rgb_frames)} frames)", flush=True)
    app.close()


if __name__ == "__main__":
    main()
