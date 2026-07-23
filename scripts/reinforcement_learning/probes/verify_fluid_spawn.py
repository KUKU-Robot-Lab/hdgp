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

"""[pour_sensor PBD 검증 — Phase 1] 유체 소환/붓기 프로토타입 (정책 없음).

목적: 실제 pour 정책을 붙이기 전에, source cup(cup_big_sdf.usd) 안을 PhysX PBD
유체(물)로 채우고 컵을 kinematic하게 기울여
  (a) 유체가 컵 안에 새지 않고 담기는가 (SDF 벽 충돌)
  (b) 기울이면 rim(+0.100) 넘어 target cup으로 쏟아지는가
를 렌더로 시각 검증한다. num_envs=1 단일 스테이지.

실행(GPU 필요):
  # 로컬/서버 GUI 로 직접 관찰
  ./isaaclab.sh -p scripts/reinforcement_learning/probes/verify_fluid_spawn.py
  # 헤드리스 + 뷰포트 캡처(증거 PNG)
  ./isaaclab.sh -p scripts/reinforcement_learning/probes/verify_fluid_spawn.py \
      --headless --capture_dir /tmp/fluid_verify
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PBD fluid spawn/pour prototype for pour_sensor cup.")
parser.add_argument("--fill_height", type=float, default=0.055, help="컵 내부를 채울 유체 높이(m, bottom 기준).")
parser.add_argument("--particle_contact_offset", type=float, default=0.006, help="PBD particle contact offset(m).")
parser.add_argument("--settle_steps", type=int, default=200, help="붓기 전 안착 스텝.")
parser.add_argument("--pour_steps", type=int, default=300, help="0→최대 tilt 까지 기울이는 스텝.")
parser.add_argument("--hold_steps", type=int, default=300, help="붓기 후 유지 스텝.")
parser.add_argument("--max_tilt_deg", type=float, default=115.0, help="최대 기울임 각(deg, Y축).")
parser.add_argument("--capture_dir", type=str, default=None, help="지정 시 뷰포트를 주기적으로 PNG 캡처.")
parser.add_argument("--capture_every", type=int, default=20, help="캡처 주기(스텝).")
parser.add_argument("--drop_gap", type=float, default=0.0, help="[진단] 유체를 바닥에서 이만큼 띄워 스폰(중력 낙하 확인).")
parser.add_argument("--freeze_cup", action="store_true", default=False, help="[진단] 컵을 안 움직임(kinematic write 영향 격리).")
parser.add_argument("--app_update", action="store_true", default=False, help="[진단] sim.step() 대신 simulation_app.update()로 스텝(순수 Isaac Sim 방식).")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Fabric 비활성 → PhysX 결과가 USD로 write-back 되어 파티클 위치 readback 가능.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils.math import quat_from_angle_axis

from pxr import Gf, Sdf, UsdGeom, UsdPhysics
from omni.physx.scripts import particleUtils

# --- pour_sensor cup 지오메트리 (cup_big_sdf.usd 로컬 프레임 기준) ---
_HDGP_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
_CUP_USD = os.path.join(_HDGP_ROOT, "assets", "cup", "cup_big_sdf.usd")
_CUP_INNER_R = 0.041     # 내부 반경
_CUP_BOTTOM_Z = -0.077   # 바닥 (로컬)
_CUP_RIM_Z = 0.100       # 림 (로컬)

_SOURCE_POS = (0.0, 0.0, 0.30)       # source cup 초기 world 위치 (붓는 컵)
_TARGET_POS = (0.16, 0.0, 0.10)      # target cup world 위치 (받는 컵)


def _find_physics_scene_path(stage) -> str:
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            return prim.GetPath().pathString
    raise RuntimeError("PhysicsScene 를 찾지 못했습니다.")


def _make_cup_cfg(prim_path: str, pos: tuple[float, float, float]) -> RigidObjectCfg:
    """kinematic 컵 — 포즈를 매 스텝 우리가 직접 명령한다 (물리 낙하 없음)."""
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(pos), rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=_CUP_USD,
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
        ),
    )


def _build_fluid(stage, sim_owner: str, contact_offset: float):
    """PBD particle system + 물 material 을 스테이지에 생성하고 바인딩한다."""
    from pxr import UsdShade
    from omni.physx.scripts import physicsUtils

    system_path = "/World/fluid/particleSystem"
    material_path = "/World/fluid/waterMaterial"

    fluid_rest_offset = 0.99 * 0.6 * contact_offset  # 데모 공식과 동일
    particleUtils.add_physx_particle_system(
        stage,
        system_path,
        simulation_owner=sim_owner,
        particle_contact_offset=contact_offset,
        fluid_rest_offset=fluid_rest_offset,
        solid_rest_offset=fluid_rest_offset,
        max_velocity=50.0,
        enable_ccd=True,
        solver_position_iterations=16,
    )

    # 물 material: 경로에 UsdShade.Material 생성 후 PBD water 프리셋 적용 → 시스템에 바인딩
    UsdShade.Material.Define(stage, material_path)
    particleUtils.AddPBDMaterialWater(stage.GetPrimAtPath(material_path))
    physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(system_path), material_path)
    return system_path, fluid_rest_offset


def _fill_positions(center_w, fill_height: float, spacing: float, drop_gap: float = 0.0):
    """source cup 내부(원기둥)를 격자 유체로 채운 world 좌표 리스트."""
    cx, cy, cz = center_w
    r_max = _CUP_INNER_R - spacing        # 벽 여유
    z0 = _CUP_BOTTOM_Z + spacing + drop_gap
    z1 = _CUP_BOTTOM_Z + fill_height + drop_gap
    positions = []
    z = z0
    while z <= z1:
        x = -r_max
        while x <= r_max:
            y = -r_max
            while y <= r_max:
                if x * x + y * y <= r_max * r_max:
                    positions.append(Gf.Vec3f(cx + x, cy + y, cz + z))
                y += spacing
            x += spacing
        z += spacing
    return positions


def _look_at_quat_world(eye, target) -> tuple[float, float, float, float]:
    """isaaclab 'world' 관례(+X=forward, +Z=up)의 look-at 쿼터니언 (w,x,y,z)."""
    import numpy as np
    e = np.asarray(eye, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    x = t - e
    x /= np.linalg.norm(x)
    up = np.array([0.0, 0.0, 1.0])
    z = up - np.dot(up, x) * x
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    r = np.column_stack([x, y, z])  # 카메라축(열)을 world 로
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        qx = (r[2, 1] - r[1, 2]) * s
        qy = (r[0, 2] - r[2, 0]) * s
        qz = (r[1, 0] - r[0, 1]) * s
    else:
        # 대각 최대 성분 기준 분기 (수치 안정)
        i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
        if i == 0:
            s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            w = (r[2, 1] - r[1, 2]) / s; qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s; qz = (r[0, 2] + r[2, 0]) / s
        elif i == 1:
            s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
            w = (r[0, 2] - r[2, 0]) / s; qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s; qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
            w = (r[1, 0] - r[0, 1]) / s; qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s; qz = 0.25 * s
    return (w, qx, qy, qz)


def _place_camera_prim(stage, prim_path: str, eye, target) -> None:
    """USD Xformable 로 카메라 prim 의 world 포즈를 직접 지정 (USD 카메라는 -Z 방향을 봄)."""
    import numpy as np
    e = np.asarray(eye, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    fwd = t - e
    fwd /= np.linalg.norm(fwd)
    zc = -fwd                                   # camera +Z
    up = np.array([0.0, 0.0, 1.0])
    xc = np.cross(up, zc); xc /= np.linalg.norm(xc)   # right (+X)
    yc = np.cross(zc, xc)                        # up (+Y)
    m = Gf.Matrix4d(
        float(xc[0]), float(xc[1]), float(xc[2]), 0.0,
        float(yc[0]), float(yc[1]), float(yc[2]), 0.0,
        float(zc[0]), float(zc[1]), float(zc[2]), 0.0,
        float(e[0]),  float(e[1]),  float(e[2]),  1.0,
    )
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)


def _save_rgb(camera, path: str) -> None:
    """카메라 RGB(torch)를 PNG로 저장."""
    try:
        from PIL import Image
        rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        Image.fromarray(rgb.astype("uint8")).save(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[verify_fluid] RGB 저장 실패: {exc}", flush=True)


def _particle_z_stats(stage) -> str:
    """파티클 world z 분포 요약 — 컵에 담김/쏟아짐을 렌더 없이 정량 확인."""
    try:
        prim = stage.GetPrimAtPath("/World/fluid/waterParticles")
        pos = UsdGeom.Points(prim).GetPointsAttr().Get()
        if not pos:
            return "particles=?"
        zs = [p[2] for p in pos]
        n = len(zs)
        z_mean = sum(zs) / n
        z_min = min(zs)
        z_max = max(zs)
        cup_bottom_w = _SOURCE_POS[2] + _CUP_BOTTOM_Z
        n_on_ground = sum(1 for z in zs if z < 0.05)                 # 지면 근처(관통 낙하)
        n_below_src = sum(1 for z in zs if z < cup_bottom_w - 0.02)  # source 바닥 아래로 빠짐
        return (f"n={n} z[min={z_min:.3f} mean={z_mean:.3f} max={z_max:.3f}] "
                f"on_ground={n_on_ground} below_src={n_below_src}")
    except Exception as exc:  # noqa: BLE001
        return f"z_stats_err={exc}"


def main():
    sim = SimulationContext(
        SimulationCfg(dt=1.0 / 120.0, device=args_cli.device, use_fabric=not args_cli.disable_fabric)
    )
    stage = sim.stage

    # ground + 조명 (DistantLight — DomeLight 안개 방지)
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)).func(
        "/World/light", sim_utils.DistantLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    )

    # 컵 두 개 (붓는 컵 / 받는 컵) — 둘 다 kinematic
    source_cup = RigidObject(_make_cup_cfg("/World/SourceCup", _SOURCE_POS))
    target_cup = RigidObject(_make_cup_cfg("/World/TargetCup", _TARGET_POS))

    # [진단] 동적 rigid 공 — 물리가 도는지 확인용 (z=0.5에서 낙하해야 함).
    test_ball = RigidObject(RigidObjectCfg(
        prim_path="/World/TestBall",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, -0.2, 0.5]),
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
        ),
    ))

    # 유체 시스템
    sim_owner = _find_physics_scene_path(stage)
    # PBD 파티클은 GPU dynamics 필수 → 씬에 강제 활성화 (isaaclab 기본이 안 켤 수 있음).
    from pxr import PhysxSchema
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(sim_owner))
    scene_api.CreateEnableGPUDynamicsAttr().Set(True)
    scene_api.CreateBroadphaseTypeAttr().Set("GPU")
    print(f"[verify_fluid] scene GPUDynamics={scene_api.GetEnableGPUDynamicsAttr().Get()} "
          f"broadphase={scene_api.GetBroadphaseTypeAttr().Get()}", flush=True)
    system_path, fluid_rest_offset = _build_fluid(stage, sim_owner, args_cli.particle_contact_offset)

    # 유체 격자 소환 (source cup 초기 포즈 기준 world 좌표)
    # 공식 데모 방식 = UsdGeom.Points (points/velocities/widths). PointInstancer 는
    # fabric 'mismatched prototypes' 동기화 버그로 시뮬 결과가 렌더에 반영되지 않았음.
    spacing = 2.0 * fluid_rest_offset
    positions = _fill_positions(_SOURCE_POS, args_cli.fill_height, spacing, args_cli.drop_gap)
    velocities = [Gf.Vec3f(0.0, 0.0, 0.0)] * len(positions)
    widths = [2.0 * fluid_rest_offset] * len(positions)
    points = particleUtils.add_physx_particleset_points(
        stage,
        Sdf.Path("/World/fluid/waterParticles"),
        positions,
        velocities,
        widths,
        system_path,
        self_collision=True,
        fluid=True,
        particle_group=0,
        particle_mass=0.0,
        density=1000.0,
    )
    points.GetDisplayColorAttr().Set([Gf.Vec3f(0.35, 0.70, 1.0)])  # 하늘색
    print(f"[verify_fluid] 유체 파티클 {len(positions)}개 소환 (UsdGeom.Points, "
          f"spacing={spacing*1000:.1f}mm, rest_offset={fluid_rest_offset*1000:.1f}mm)")

    # RGB 증거용 카메라 (--capture_dir 지정 시). 헤드리스 렌더 → PNG 저장.
    # 두 컵을 옆·앞에서 비스듬히 내려다보게 look-at pose 를 스폰 offset 으로 선언.
    camera = None
    if args_cli.capture_dir:
        os.makedirs(args_cli.capture_dir, exist_ok=True)
        cam_eye = (0.55, -0.75, 0.55)
        cam_target = (0.08, 0.0, 0.20)
        cam_quat = _look_at_quat_world(cam_eye, cam_target)
        camera = Camera(CameraCfg(
            prim_path="/World/verify_cam",
            update_period=0,
            height=720,
            width=1280,
            data_types=["rgb"],
            offset=CameraCfg.OffsetCfg(pos=cam_eye, rot=cam_quat, convention="world"),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 1.0e4)
            ),
        ))

    sim.reset()

    if camera is not None:
        # OffsetCfg 가 반영 안 되는 케이스 → USD xform 으로 직접 배치.
        _place_camera_prim(stage, "/World/verify_cam", cam_eye, cam_target)
        for _ in range(3):  # 렌더 워밍업
            sim.step()
            camera.update(sim.get_physics_dt())
        print(f"[verify_fluid] camera pos_w={camera.data.pos_w[0].tolist()}", flush=True)

    src_pos_t = torch.tensor([[*_SOURCE_POS]], device=sim.device, dtype=torch.float32)
    y_axis = torch.tensor([[0.0, 1.0, 0.0]], device=sim.device, dtype=torch.float32)
    max_tilt = math.radians(args_cli.max_tilt_deg)

    total = args_cli.settle_steps + args_cli.pour_steps + args_cli.hold_steps
    print(f"[verify_fluid] 시뮬레이션 루프 시작 (총 {total} step)", flush=True)
    step = 0
    try:
        while simulation_app.is_running() and step < total:
            # source cup tilt 스케줄: settle→pour(0→max)→hold(max)
            if step < args_cli.settle_steps:
                frac = 0.0
            elif step < args_cli.settle_steps + args_cli.pour_steps:
                frac = (step - args_cli.settle_steps) / max(1, args_cli.pour_steps)
            else:
                frac = 1.0
            if not args_cli.freeze_cup:
                angle = torch.tensor([frac * max_tilt], device=sim.device, dtype=torch.float32)
                quat = quat_from_angle_axis(angle, y_axis)  # (1,4) wxyz
                pose = torch.cat([src_pos_t, quat], dim=-1)
                source_cup.write_root_pose_to_sim(pose)
                source_cup.write_data_to_sim()
                target_cup.write_data_to_sim()

            if args_cli.app_update:
                simulation_app.update()          # 순수 Isaac Sim 방식 (timeline 재생 중)
            else:
                sim.step()
            source_cup.update(sim.get_physics_dt())
            target_cup.update(sim.get_physics_dt())
            test_ball.update(sim.get_physics_dt())

            if camera is not None:
                camera.update(sim.get_physics_dt())
                if step % args_cli.capture_every == 0:
                    _save_rgb(camera, os.path.join(args_cli.capture_dir, f"frame_{step:05d}.png"))

            if step % 30 == 0:
                z_stats = _particle_z_stats(stage)
                ball_z = float(test_ball.data.root_pos_w[0, 2])
                print(f"[verify_fluid] step {step}/{total}  tilt={math.degrees(frac*max_tilt):.0f}°  "
                      f"ball_z={ball_z:.3f}  {z_stats}", flush=True)
            step += 1
    except Exception:  # noqa: BLE001
        import traceback
        print("[verify_fluid] 루프 예외:\n" + traceback.format_exc(), flush=True)

    print(f"[verify_fluid] 붓기 시퀀스 완료 (실행 step={step}/{total}).", flush=True)

    # GUI 모드: 창을 열어둔 채 사용자가 관찰/orbit 하도록 유지 (창 닫으면 종료).
    if not args_cli.headless:
        print("[verify_fluid] GUI 유지 중 — 뷰포트에서 관찰하세요. 창을 닫으면 종료합니다.", flush=True)
        try:
            while simulation_app.is_running():
                sim.step()
        except Exception:  # noqa: BLE001
            pass

    # PBD 파티클이 있으면 simulation_app.close() 가 teardown 에서 hang → GPU 좀비 누수.
    # close() 를 데몬 스레드로 시도하되, 짧은 유예 후 프로세스를 강제 종료해 GPU 를 확실히 반환한다.
    import threading
    import time as _time
    threading.Thread(target=simulation_app.close, daemon=True).start()
    _time.sleep(3.0)
    print("[verify_fluid] 프로세스 강제 종료(GPU 반환).", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
