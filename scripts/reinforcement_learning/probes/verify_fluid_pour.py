# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""[pour_sensor PBD 검증 — Phase 1, raw-app + cm 스케일] 유체 소환/붓기 (정책 없음).

핵심(공식 물 데모와 동일하게 맞춤):
- isaacsim/isaaclab `SimulationContext` 는 PBD 파티클 시뮬을 끈다 → raw omni.timeline+app.update.
- PBD 물은 **metersPerUnit=0.01(cm)** 에서만 안정적으로 물처럼 거동한다(솔버가 order-1 단위 크기 가정,
  물성 mpu³ 스케일). mpu=1.0 에선 물방울이 극소 단위라 불안정/튕김 → 이 스크립트는 cm 스케일로 동작.
- 물성은 **기본값**, spacing=2.0×fluid_rest, Smoothing/Anisotropy/Isosurface 로 연속 수면 렌더.

내부 단위는 cm(1 unit = 0.01 m). CLI 위치/크기 인자는 미터로 받아 ×100 한다.

실행:
  ./IsaacLab/isaaclab.sh -p hdgp/scripts/reinforcement_learning/probes/verify_fluid_pour.py
"""

import argparse

parser = argparse.ArgumentParser(description="Raw-app cm-scale PBD water pour prototype.")
parser.add_argument("--headless", action="store_true", default=False, help="헤드리스(창 없음).")
parser.add_argument("--fill_height", type=float, default=0.045, help="컵 내부 유체 높이(m). 개수↓.")
parser.add_argument("--particle_contact", type=float, default=0.008, help="particle contact offset(m). 0.8cm.")
parser.add_argument("--no_isosurface", action="store_true", default=False, help="isosurface 끄고 물방울 구슬로 렌더.")
parser.add_argument("--sdf_cup", action="store_true", default=False, help="SDF 컵 USD 사용(기본=원통 벽 컵, 안 샘).")
parser.add_argument("--cup_segments", type=int, default=28, help="원통 벽 컵 세그먼트 수.")
parser.add_argument("--spawn_batches", type=int, default=20, help="유체를 이 수만큼 층으로 나눠 점진 스폰(초기 폭발 방지).")
parser.add_argument("--spawn_interval", type=int, default=18, help="배치 사이 스텝 간격.")
parser.add_argument("--settle_steps", type=int, default=420, help="붓기 전 안착 스텝(점진 스폰 완료 후 여유 포함).")
parser.add_argument("--pour_steps", type=int, default=350, help="0→최대 tilt 스텝.")
parser.add_argument("--hold_steps", type=int, default=250, help="붓기 후 유지 스텝.")
parser.add_argument("--max_tilt_deg", type=float, default=120.0, help="최대 기울임 각(deg, Y축).")
parser.add_argument("--source_z", type=float, default=0.30, help="붓는 컵 높이(m).")
parser.add_argument("--target_x", type=float, default=0.13, help="받는 컵 x(m).")
parser.add_argument("--target_z", type=float, default=0.12, help="받는 컵 높이(m).")
parser.add_argument("--capture_dir", type=str, default=None, help="replicator 프레임 캡처(헤드리스).")
parser.add_argument("--capture_every", type=int, default=50, help="캡처 주기(스텝).")
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": args.headless})

# ---------------------------------------------------------------------------
import math
import os

import omni.kit.commands
import omni.timeline
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
from omni.physx.scripts import particleUtils, physicsUtils

_M = 100.0  # 미터 → cm 단위 (metersPerUnit=0.01)

# cup_big_sdf.usd (미터 저작) 를 cm 스케일에 얹으려면 ×100.
_HDGP_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
# cm 스케일 컵(정점 ×100 구움) — xform 스케일 없이 참조해야 SDF 충돌이 유효.
_CUP_USD = os.path.join(_HDGP_ROOT, "assets", "cup", "cup_big_sdf_cm.usd")
_CUP_INNER_R = 0.041 * _M     # 4.1 cm
_CUP_BOTTOM_Z = -0.077 * _M   # -7.7 cm
_CUP_RIM_Z = 0.100 * _M       # 10 cm

_SOURCE_POS = (0.0, 0.0, args.source_z * _M)
_TARGET_POS = (args.target_x * _M, 0.0, args.target_z * _M)


def _quat_about_y(a: float) -> Gf.Quatf:
    return Gf.Quatf(math.cos(a / 2.0), Gf.Vec3f(0.0, math.sin(a / 2.0), 0.0))


def _place_camera(prim, eye, target):
    import numpy as np
    e = np.asarray(eye, float); t = np.asarray(target, float)
    fwd = t - e; fwd /= np.linalg.norm(fwd)
    zc = -fwd
    xc = np.cross(np.array([0.0, 0.0, 1.0]), zc); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    m = Gf.Matrix4d(
        float(xc[0]), float(xc[1]), float(xc[2]), 0.0,
        float(yc[0]), float(yc[1]), float(yc[2]), 0.0,
        float(zc[0]), float(zc[1]), float(zc[2]), 0.0,
        float(e[0]), float(e[1]), float(e[2]), 1.0,
    )
    xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder(); xf.AddTransformOp().Set(m)


def _add_cup(stage, path, pos):
    """cup_big_sdf.usd 참조(×100) + kinematic. 충돌은 USD 내장 SDF."""
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(_CUP_USD)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    rb = UsdPhysics.RigidBodyAPI(prim)
    rb.CreateKinematicEnabledAttr().Set(True)
    PhysxSchema.PhysxRigidBodyAPI(prim).CreateDisableGravityAttr().Set(True)
    return prim


def _add_cylinder_cup(stage, path, pos, n_seg, wall=0.4, visual=True):
    """비주얼 컵 USD + 내부 얇은 원통 벽(충돌). 실제 컵 모양 + 안 새는 볼록 충돌.

    - visual=True: 원래 cm 컵 USD 를 참조(비주얼). 그 SDF 충돌은 끔.
    - 내부에 바닥 디스크 + 벽(박스 링)을 충돌용으로 추가 → 유체는 이 원통 안에 담김.
    로컬 지오메트리는 컵 상수(bottom/rim/inner_r, cm)와 동일.
    """
    r_in = _CUP_INNER_R
    z_bot = _CUP_BOTTOM_Z
    z_top = _CUP_RIM_Z
    h = z_top - z_bot

    prim = stage.DefinePrim(path, "Xform")
    if visual:
        prim.GetReferences().AddReference(_CUP_USD)   # 비주얼 컵(cm)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr().Set(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(True)
    if visual:
        # 참조된 컵 메시의 SDF 충돌 끔 → 비주얼 전용(충돌은 원통 벽이 담당)
        vmesh = stage.GetPrimAtPath(path + "/cup_big")
        if vmesh and vmesh.IsValid():
            UsdPhysics.CollisionAPI.Apply(vmesh).CreateCollisionEnabledAttr().Set(False)

    # 바닥 디스크 (top 면이 컵 바닥 z_bot)
    bottom = UsdGeom.Cylinder.Define(stage, path + "/bottom")
    bottom.CreateRadiusAttr().Set(r_in + wall)
    bottom.CreateHeightAttr().Set(wall)
    bottom.CreateAxisAttr().Set("Z")
    physicsUtils.set_or_add_translate_op(bottom, Gf.Vec3f(0.0, 0.0, z_bot - wall * 0.5))
    UsdPhysics.CollisionAPI.Apply(bottom.GetPrim())

    # 벽: 박스 링 (각 박스 로컬 X=반경방향 두께, Y=접선방향 폭, Z=높이)
    R = r_in + wall * 0.5
    seg_w = 2.0 * math.pi * R / n_seg * 1.3   # 접선 폭 (겹침 여유)
    for k in range(n_seg):
        th = 2.0 * math.pi * k / n_seg
        b = UsdGeom.Cube.Define(stage, f"{path}/wall_{k:02d}")
        b.CreateSizeAttr().Set(1.0)
        xb = UsdGeom.Xformable(b)
        xb.ClearXformOpOrder()
        xb.AddTranslateOp().Set(Gf.Vec3d(R * math.cos(th), R * math.sin(th), z_bot + h * 0.5))
        xb.AddOrientOp().Set(Gf.Quatf(math.cos(th / 2.0), Gf.Vec3f(0.0, 0.0, math.sin(th / 2.0))))
        xb.AddScaleOp().Set(Gf.Vec3f(wall, seg_w, h))
        UsdPhysics.CollisionAPI.Apply(b.GetPrim())
    return prim


def _set_cup_orient(prim, quat):
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            op.Set(quat); return


def _fill_grid(center, fill_height_cm, spacing):
    cx, cy, cz = center
    r_max = _CUP_INNER_R - 1.2 * spacing
    z0 = _CUP_BOTTOM_Z + 2.0 * spacing
    z1 = _CUP_BOTTOM_Z + 2.0 * spacing + fill_height_cm
    out = []
    z = z0
    while z <= z1:
        x = -r_max
        while x <= r_max:
            y = -r_max
            while y <= r_max:
                if x * x + y * y <= r_max * r_max:
                    out.append(Gf.Vec3f(cx + x, cy + y, cz + z))
                y += spacing
            x += spacing
        z += spacing
    return out


def _extend(attr, elems):
    cur = attr.Get()
    attr.Set(list(cur) + list(elems) if cur else list(elems))


def _append_particles(stage, particleset_path, positions, use_smoothing):
    """점진 스폰: 기존 particle set(point instancer)에 배치를 추가."""
    prim = stage.GetPrimAtPath(particleset_path)
    pi = UsdGeom.PointInstancer(prim)
    if use_smoothing:
        sp = PhysxSchema.PhysxParticleSetAPI(prim).GetSimulationPointsAttr()
        if not sp.HasAuthoredValue():
            sp.Set(Vt.Vec3fArray([]))
        _extend(sp, positions)
    _extend(pi.GetPositionsAttr(), positions)
    _extend(pi.GetVelocitiesAttr(), [Gf.Vec3f(0.0)] * len(positions))
    _extend(pi.GetProtoIndicesAttr(), [0] * len(positions))
    _extend(pi.GetOrientationsAttr(), [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * len(positions))
    _extend(pi.GetScalesAttr(), [Gf.Vec3f(1.0)] * len(positions))


def _in_cup(p, cx, cy, cz):
    return ((p[0] - cx) ** 2 + (p[1] - cy) ** 2 < (_CUP_INNER_R + 1.0) ** 2
            and (cz + _CUP_BOTTOM_Z - 1.0) < p[2] < (cz + _CUP_RIM_Z))


def main():
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)   # cm 스케일 (물 데모와 동일)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81 * _M)   # cm/s^2
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    scene_api.CreateEnableGPUDynamicsAttr().Set(True)
    scene_api.CreateBroadphaseTypeAttr().Set("GPU")
    # 물 데모가 켜는 핵심 설정 — 유체에 외력(중력)을 매 솔버 반복 적용 → 안정적 물 거동.
    scene_api.CreateEnableExternalForcesEveryIterationAttr().Set(True)

    UsdLux.DistantLight.Define(stage, "/World/light").CreateIntensityAttr().Set(3000.0)

    # 지면 (두꺼운 박스, top z=0)
    ground = UsdGeom.Cube.Define(stage, "/World/ground")
    ground.CreateSizeAttr().Set(1.0)
    physicsUtils.set_or_add_scale_op(ground, Gf.Vec3f(4.0 * _M, 4.0 * _M, 0.4 * _M))
    physicsUtils.set_or_add_translate_op(ground, Gf.Vec3f(0.0, 0.0, -0.2 * _M))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    gcapi = PhysxSchema.PhysxCollisionAPI.Apply(ground.GetPrim())
    gcapi.CreateContactOffsetAttr().Set(2.0)   # 관통 방지 (cm)
    gcapi.CreateRestOffsetAttr().Set(1.0)

    if args.sdf_cup:
        source_cup = _add_cup(stage, "/World/SourceCup", _SOURCE_POS)
        _add_cup(stage, "/World/TargetCup", _TARGET_POS)
        # SDF 컵: 스케일 지오메트리에 rest offset 확대.
        for cup in ("SourceCup", "TargetCup"):
            mesh = stage.GetPrimAtPath(f"/World/{cup}/cup_big")
            if mesh and mesh.IsValid():
                capi = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
                capi.CreateContactOffsetAttr().Set(1.2)
                capi.CreateRestOffsetAttr().Set(0.6)
    else:
        # 기본: 원통 벽 컵 (볼록 프리미티브 → 안 샘)
        source_cup = _add_cylinder_cup(stage, "/World/SourceCup", _SOURCE_POS, args.cup_segments)
        _add_cylinder_cup(stage, "/World/TargetCup", _TARGET_POS, args.cup_segments)

    # --- 파티클 시스템 (물 데모 레시피, cm) ---
    system_path = Sdf.Path("/World/particleSystem")
    particle_system = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
    particle_system.CreateSimulationOwnerRel().SetTargets([scene.GetPath()])
    particle_contact_offset = args.particle_contact * _M     # 예: 0.006 m → 0.6 cm
    particle_system.CreateParticleContactOffsetAttr().Set(particle_contact_offset)
    particle_system.CreateMaxVelocityAttr().Set(60.0)      # 낮춰 SDF 짜임·바닥 관통 완화
    particle_system.CreateEnableCCDAttr().Set(True)         # 바닥 관통 방지
    particle_system.CreateSolverPositionIterationCountAttr().Set(24)

    # 물 렌더: smoothing + anisotropy + isosurface (연속 수면)
    PhysxSchema.PhysxParticleSmoothingAPI.Apply(particle_system.GetPrim())
    PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.GetPrim())
    use_iso = not args.no_isosurface
    if use_iso:
        PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(particle_system.GetPrim())
        ani = PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.GetPrim())
        ani.CreateScaleAttr().Set(5.0)
        ani.CreateMinAttr().Set(1.0)
        ani.CreateMaxAttr().Set(2.0)

    # 렌더 material (OmniPBR, 하늘색) — isosurface 는 이 material 로 렌더됨
    mtl_created = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary", mdl_name="OmniPBR.mdl", mtl_name="OmniPBR",
        mtl_created_list=mtl_created, bind_selected_prims=False, select_new_prim=False,
    )
    render_material_path = mtl_created[0]
    shader = UsdShade.Shader.Get(stage, render_material_path + "/Shader")
    if shader:
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.4, 0.9))
    # SDF 컵은 박스와 달리 미세 틈이 있어, 무점성 물이 짜여 나감 → 약간의 점성/감쇠로 방지.
    particleUtils.add_pbd_particle_material(
        stage, render_material_path,
        viscosity=5.0, cohesion=0.5, friction=0.5, surface_tension=0.01, damping=0.9,
    )
    omni.kit.commands.execute("BindMaterialCommand", prim_path=str(system_path),
                              material_path=render_material_path, strength=None)

    fluid_rest = 0.99 * 0.6 * particle_contact_offset
    spacing = 2.0 * fluid_rest
    all_pos = _fill_grid(_SOURCE_POS, args.fill_height * _M, spacing)
    all_pos.sort(key=lambda p: p[2])   # 바닥층부터 (아래→위)
    # 층별 배치로 분할 → 점진 스폰(초기 폭발 방지)
    nb = max(1, args.spawn_batches)
    batch_sz = max(1, math.ceil(len(all_pos) / nb))
    batches = [all_pos[i:i + batch_sz] for i in range(0, len(all_pos), batch_sz)]

    particleset_path = Sdf.Path("/World/waterParticles")
    b0 = batches[0]
    particleUtils.add_physx_particleset_pointinstancer(
        stage, particleset_path, Vt.Vec3fArray(b0), Vt.Vec3fArray([Gf.Vec3f(0.0)] * len(b0)),
        system_path, self_collision=True, fluid=True, particle_group=0,
        particle_mass=0.001, density=0.0,
    )
    proto = UsdGeom.Sphere.Get(stage, particleset_path.AppendChild("particlePrototype0"))
    if proto:
        proto.CreateRadiusAttr().Set(fluid_rest)
        omni.kit.commands.execute("BindMaterialCommand", prim_path=str(proto.GetPath()),
                                  material_path=render_material_path, strength=None)
    pending_batches = batches[1:]   # 루프에서 순차 추가
    print(f"[pour] 유체 {len(all_pos)}개 ({len(batches)}배치 점진 스폰, cm 스케일, "
          f"contact={particle_contact_offset:.2f}cm fluid_rest={fluid_rest:.2f}cm, isosurface={use_iso})", flush=True)

    # replicator 캡처
    capture = None
    if args.capture_dir:
        os.makedirs(args.capture_dir, exist_ok=True)
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.core")
        import omni.replicator.core as rep
        cam = UsdGeom.Camera.Define(stage, "/World/capCam")
        _place_camera(cam.GetPrim(), (0.42 * _M, -0.48 * _M, 0.46 * _M), (0.06 * _M, 0.0, 0.24 * _M))
        rp = rep.create.render_product("/World/capCam", (1280, 720))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach(rp)
        capture = annot

    def save_frame(path):
        try:
            from PIL import Image
            data = capture.get_data()
            if data is not None and len(data) > 0:
                Image.fromarray(data[..., :3]).save(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[pour] 캡처 실패: {exc}", flush=True)

    def stat():
        pi = UsdGeom.PointInstancer.Get(stage, particleset_path)
        pos = pi.GetPositionsAttr().Get()
        if not pos:
            return "n=0"
        n_src = sum(1 for p in pos if _in_cup(p, *_SOURCE_POS))
        n_tgt = sum(1 for p in pos if _in_cup(p, *_TARGET_POS))
        n_gnd = sum(1 for p in pos if p[2] < 3.0)
        return f"n={len(pos)} in_src={n_src} in_tgt={n_tgt} on_ground={n_gnd} zmin={min(q[2] for q in pos):.1f}"

    tl = omni.timeline.get_timeline_interface()
    tl.play()

    max_tilt = math.radians(args.max_tilt_deg)
    total = args.settle_steps + args.pour_steps + args.hold_steps
    print(f"[pour] 루프 시작 (총 {total} step, cm 스케일)", flush=True)
    bi = 0   # 다음에 추가할 pending 배치 인덱스
    for step in range(total):
        # 점진 스폰: spawn_interval 마다 다음 층 배치 추가 (초기 폭발 방지)
        if bi < len(pending_batches) and step > 0 and step % args.spawn_interval == 0:
            _append_particles(stage, particleset_path, pending_batches[bi], use_smoothing=True)
            bi += 1
        if step < args.settle_steps:
            frac = 0.0
        elif step < args.settle_steps + args.pour_steps:
            frac = (step - args.settle_steps) / max(1, args.pour_steps)
        else:
            frac = 1.0
        _set_cup_orient(source_cup, _quat_about_y(frac * max_tilt))
        app.update()
        if capture is not None and step % args.capture_every == 0:
            save_frame(os.path.join(args.capture_dir, f"frame_{step:05d}.png"))
        if step % 50 == 0:
            print(f"[pour] step {step}/{total} tilt={math.degrees(frac*max_tilt):.0f}°  {stat()}", flush=True)

    print(f"[pour] 붓기 완료. {stat()}", flush=True)

    if not args.headless:
        print("[pour] GUI 유지 중 — 창을 닫으면 종료.", flush=True)
        while app.is_running():
            app.update()

    import threading
    import time as _time
    threading.Thread(target=app.close, daemon=True).start()
    _time.sleep(3.0)
    os._exit(0)


if __name__ == "__main__":
    main()
