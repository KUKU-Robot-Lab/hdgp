"""maximal-coordinate 닫힌 링 컵을 양쪽에서 구(손가락)로 눌러 cohesive 변형 + 루프 안정성 확인.

닫힌 링은 articulation(트리) 미지원 → 강체+조인트(maximal). Articulation 래퍼를 못 쓰므로
raw 스폰 + 2개 kinematic 구를 안쪽으로 이동시켜 접촉으로 링을 눌러 crumple 시킨다.
camera로 렌더 → 루프가 터지지 않고(안정) 형상 전체가 cohesive하게 찌그러졌다 복원하는지 확인.

  cd /home/user/rl_ws/hdgp
  /home/user/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/render_ring_squeeze.py
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", type=str,
                    default="/home/user/rl_ws/hdgp/assets/cup/deformable_cup_ring.usd")
parser.add_argument("--frames", type=int, default=200)
parser.add_argument("--out", type=str,
                    default="/tmp/claude-1000/-home-user-rl-ws/21fefa30-5de6-40a8-a088-f4d81dfb2222/scratchpad/ring_sq")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import os  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def _presser(prim, pos):
    # 면(paddle) presser: 실제 손가락/손처럼 벽을 넓게(높이·둘레) 눌러 하중 분산.
    # 얇게(x=누름방향)·넓게(y=접선)·높게(z=벽높이).
    return RigidObject(RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.CuboidCfg(
            size=(0.012, 0.06, 0.12),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    ))


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))

    # 닫힌 링 컵 raw 스폰 (base 정적, 패널 동적, edge 루프 조인트)
    cfg = sim_utils.UsdFileCfg(usd_path=args.usd, activate_contact_sensors=False)
    cfg.func("/World/Cup", cfg, translation=(0.0, 0.0, 0.10))

    z_mid = 0.10 - 0.047 + 0.0735   # 컵 벽 중간 높이(월드)
    x0 = 0.085                       # 시작 x(벽 밖)
    left = _presser("/World/PressL", (-x0, 0.0, z_mid))
    right = _presser("/World/PressR", (x0, 0.0, z_mid))

    cam_cfg = CameraCfg(prim_path="/World/Camera", height=480, width=640,
                        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=24.0),
                        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0),
                                                   rot=(0.0, 0.0, 0.0, 1.0), convention="world"))
    cam = Camera(cam_cfg)

    sim.reset()
    cam.set_world_poses_from_view(
        torch.tensor([[0.32, 0.30, 0.30]], device=args.device),
        torch.tensor([[0.0, 0.0, 0.14]], device=args.device))

    os.makedirs(args.out, exist_ok=True)
    rgb_frames = []
    F = args.frames
    x_crush = 0.030   # 벽 안쪽으로(R=0.045 지나 눌러 crush)

    for f in range(F):
        t = f / F
        if t < 0.4:
            xr = x0 - (x0 - x_crush) * (t / 0.4)     # 안으로
        elif t < 0.6:
            xr = x_crush                             # 유지
        else:
            xr = x_crush + (x0 - x_crush) * ((t - 0.6) / 0.4)  # 복원
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=args.device)
        left.write_root_pose_to_sim(torch.tensor([[-xr, 0.0, z_mid, 1, 0, 0, 0]], device=args.device))
        right.write_root_pose_to_sim(torch.tensor([[xr, 0.0, z_mid, 1, 0, 0, 0]], device=args.device))
        left.write_data_to_sim(); right.write_data_to_sim()
        sim.step()
        left.update(sim.get_physics_dt()); right.update(sim.get_physics_dt())
        cam.update(sim.get_physics_dt())
        rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)  # PIL 'tile' 인코딩 버그(비연속 배열) 회피
        rgb_frames.append(rgb)
        if f % 40 == 0:
            nan = bool(np.isnan(rgb).any())
            print(f"[ring_sq] frame {f}/{F} xr={xr:.3f} nan_rgb={nan}", flush=True)

    # 핵심 PNG 먼저(인코딩 무관, 항상 성공): rest·peak(최대 crush)
    imageio.imwrite(os.path.join(args.out, "rest.png"), rgb_frames[3])
    imageio.imwrite(os.path.join(args.out, "peak.png"), rgb_frames[len(rgb_frames) // 2])
    print(f"[ring_sq] saved rest.png peak.png", flush=True)
    for name, seq, fps in (("ring_squeeze.mp4", rgb_frames, 30),
                           ("ring_squeeze.gif", rgb_frames[::2], 15)):
        try:
            imageio.mimsave(os.path.join(args.out, name), seq, fps=fps)
            print(f"[ring_sq] saved {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[ring_sq] {name} fail: {e}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
