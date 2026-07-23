"""P1b — PBD 유체 스케일 안정성 스모크 (live_policy_fluid_plan 대안 B 확정용).

질문: raw-app(SimulationContext 없음) 에서 PBD 물 파티클이 **meter 스케일(mpu=1.0)** 에서 안정한가.
P1 결론(로봇 contact grasp 는 meter 에서만 작동, cm Xform-scale 붕괴) → 전체 meter(대안 B) 가려면
PBD 도 meter 에서 안정해야 함. 메모리엔 "mpu=1.0 불안정, 재검증 필요" 로 남아있어 이를 실측 판정한다.

방법: 정적 컵(kinematic) 에 물 파티클을 채우고 중력 하 안착 → 폭발/누수/동결 없이 컵 안 기둥으로
안정하는지 측정. replay_pour_fluid.py 의 파티클 시스템·컵·fill 을 스케일 파라미터화(_M)해 재사용.

실행:
  ./IsaacLab/isaaclab.sh -p hdgp/scripts/reinforcement_learning/probes/p1b_pbd_scale_smoke.py --headless
  ./IsaacLab/isaaclab.sh -p ... --headless --cm     # cm(=기존 replay 스케일) 대조
"""

import argparse
import math
import os

import numpy as np

_HDGP_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))

parser = argparse.ArgumentParser(description="P1b PBD scale stability smoke")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--cm", action="store_true", default=False, help="cm 스케일(mpu=0.01) 대조.")
parser.add_argument("--particle_contact", type=float, default=0.008, help="particle contact offset(m).")
parser.add_argument("--fill_height", type=float, default=0.045, help="컵 유체 높이(m).")
parser.add_argument("--settle_steps", type=int, default=400, help="안착 스텝.")
parser.add_argument("--check_tail", type=int, default=120, help="말미 안정성(진동) 측정 스텝.")
parser.add_argument("--no_isosurface", action="store_true", default=False)
parser.add_argument("--cup_segments", type=int, default=28)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless})

import carb  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402
from omni.physx.scripts import particleUtils, physicsUtils  # noqa: E402

_cs = carb.settings.get_settings()
_cs.set_bool("/app/useFabricSceneDelegate", False)
_cs.set_bool("/physics/fabricUpdateTransformations", False)
_cs.set_bool("/physics/updateToUsd", True)
_cs.set_bool("/physics/updateParticlesToUsd", True)

_M = 100.0 if args.cm else 1.0
_MPU = 0.01 if args.cm else 1.0
_CUP_USD = os.path.join(_HDGP_ROOT, "assets", "cup", "cup_big_sdf_cm.usd")
_CUP_INNER_R = 0.041 * _M
_CUP_BOTTOM_Z = -0.077 * _M
_CUP_RIM_Z = 0.100 * _M
# 컵을 바닥 위로 띄워 배치(바닥=z0). 컵 중심 z 오프셋으로 rim/bottom 을 양수 영역에.
_CUP_CENTER_Z = 0.20 * _M


def _add_cylinder_cup(stage, path, n_seg, wall=0.004 * _M):
    r_in, z_bot, z_top = _CUP_INNER_R, _CUP_BOTTOM_Z, _CUP_RIM_Z
    h = z_top - z_bot
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(_CUP_USD)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, _CUP_CENTER_Z))
    xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr().Set(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(True)
    vmesh = stage.GetPrimAtPath(path + "/cup_big")
    if vmesh and vmesh.IsValid():
        UsdPhysics.CollisionAPI.Apply(vmesh).CreateCollisionEnabledAttr().Set(False)

    bottom = UsdGeom.Cylinder.Define(stage, path + "/bottom")
    bottom.CreateRadiusAttr().Set(r_in + wall)
    bottom.CreateHeightAttr().Set(wall)
    bottom.CreateAxisAttr().Set("Z")
    physicsUtils.set_or_add_translate_op(bottom, Gf.Vec3f(0.0, 0.0, z_bot - wall * 0.5))
    UsdPhysics.CollisionAPI.Apply(bottom.GetPrim())

    R = r_in + wall * 0.5
    seg_w = 2.0 * math.pi * R / n_seg * 1.3
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


def _build_particle_system(stage, scene):
    system_path = Sdf.Path("/World/particleSystem")
    ps = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
    ps.CreateSimulationOwnerRel().SetTargets([scene.GetPath()])
    pco = args.particle_contact * _M
    ps.CreateParticleContactOffsetAttr().Set(pco)
    ps.CreateMaxVelocityAttr().Set(60.0 * _M)   # 물리속도 클램프도 스케일
    ps.CreateEnableCCDAttr().Set(True)
    ps.CreateSolverPositionIterationCountAttr().Set(24)
    PhysxSchema.PhysxParticleSmoothingAPI.Apply(ps.GetPrim())
    if not args.no_isosurface:
        PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(ps.GetPrim())
        ani = PhysxSchema.PhysxParticleAnisotropyAPI.Apply(ps.GetPrim())
        ani.CreateScaleAttr().Set(5.0); ani.CreateMinAttr().Set(1.0); ani.CreateMaxAttr().Set(2.0)

    mtl_created = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary", mdl_name="OmniPBR.mdl", mtl_name="OmniPBR",
        mtl_created_list=mtl_created, bind_selected_prims=False, select_new_prim=False,
    )
    mat = mtl_created[0]
    particleUtils.add_pbd_particle_material(
        stage, mat, viscosity=0.3, cohesion=0.01, friction=0.2, surface_tension=0.0072, damping=0.05,
    )
    omni.kit.commands.execute("BindMaterialCommand", prim_path=str(system_path), material_path=mat, strength=None)
    fluid_rest = 0.99 * 0.6 * pco
    return system_path, mat, fluid_rest


def _fill_local(fill_height, spacing):
    r_max = _CUP_INNER_R - 1.2 * spacing
    z0 = _CUP_BOTTOM_Z + 2.0 * spacing
    z1 = z0 + fill_height
    out = []
    z = z0
    while z <= z1:
        x = -r_max
        while x <= r_max:
            y = -r_max
            while y <= r_max:
                if x * x + y * y <= r_max * r_max:
                    out.append((x, y, z))
                y += spacing
            x += spacing
        z += spacing
    return out


def _spawn_particles(stage, system_path, positions_local):
    path = "/World/waterParticles"
    pts = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]) + _CUP_CENTER_Z) for p in positions_local]
    particleUtils.add_physx_particleset_pointinstancer(
        stage, Sdf.Path(path), Vt.Vec3fArray(pts), Vt.Vec3fArray([Gf.Vec3f(0.0)] * len(pts)),
        system_path, self_collision=True, fluid=True, particle_group=0,
        particle_mass=0.0, density=1000.0,
    )
    return path


def _read_particles(stage):
    pi = UsdGeom.PointInstancer.Get(stage, "/World/waterParticles")
    if not pi:
        return None
    pos = pi.GetPositionsAttr().Get()
    if pos is None:
        return None
    return np.array([[p[0], p[1], p[2]] for p in pos], dtype=np.float64)


def _frac_in_cup(parts):
    """컵 로컬(중심 _CUP_CENTER_Z) 기준 내부 비율. 컵 upright 고정이므로 z 오프셋만."""
    if parts is None or len(parts) == 0:
        return 0.0, 0
    n_in = 0
    for p in parts:
        r2 = p[0] * p[0] + p[1] * p[1]
        lz = p[2] - _CUP_CENTER_Z
        if r2 < (_CUP_INNER_R + 0.5 * _M / 100.0) ** 2 and (_CUP_BOTTOM_Z - 1.0) < lz < (_CUP_RIM_Z + 1.0):
            n_in += 1
    return n_in / len(parts), n_in


def main():
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, _MPU)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81 * _M)
    sapi = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    sapi.CreateEnableGPUDynamicsAttr().Set(True)
    sapi.CreateBroadphaseTypeAttr().Set("GPU")
    sapi.CreateSolverTypeAttr().Set("TGS")

    UsdLux.DistantLight.Define(stage, "/World/light").CreateIntensityAttr().Set(3000.0)
    ground = UsdGeom.Cube.Define(stage, "/World/ground")
    ground.CreateSizeAttr().Set(1.0)
    physicsUtils.set_or_add_scale_op(ground, Gf.Vec3f(4.0 * _M, 4.0 * _M, 0.4 * _M))
    physicsUtils.set_or_add_translate_op(ground, Gf.Vec3f(0.0, 0.0, -0.2 * _M))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    _add_cylinder_cup(stage, "/World/Cup", args.cup_segments)
    system_path, _mat, fluid_rest = _build_particle_system(stage, scene)
    spacing = 2.0 * fluid_rest
    local = _fill_local(args.fill_height * _M, spacing)
    _spawn_particles(stage, system_path, local)
    n0 = len(local)
    print(f"[P1b] scale={'cm' if args.cm else 'meter'} (mpu={_MPU}) | "
          f"pco={args.particle_contact*_M:.4g} | 파티클 {n0}개 스폰", flush=True)

    tl = omni.timeline.get_timeline_interface()
    tl.play()

    # 안착
    for _ in range(args.settle_steps):
        app.update()
    parts_mid = _read_particles(stage)

    # 말미 안정성(진동/폭발) 측정: check_tail 동안 이동량
    ref = parts_mid.copy() if parts_mid is not None else None
    for _ in range(args.check_tail):
        app.update()
    parts_end = _read_particles(stage)

    # ---- 판정 지표 ----
    if parts_end is None:
        print("[P1b] 파티클 읽기 실패 → PBD 미작동(SimulationContext 오염 의심)", flush=True)
        app.close(); return

    finite = np.isfinite(parts_end).all(axis=1)
    n_finite = int(finite.sum())
    pe = parts_end[finite] / _M   # meter
    bbox = (pe.max(axis=0) - pe.min(axis=0)) if len(pe) else np.zeros(3)
    frac_in, n_in = _frac_in_cup(parts_end)
    # 말미 이동(안정=작음). 파티클 순서 보존 가정(PointInstancer 인덱스 고정).
    if ref is not None and parts_end.shape == ref.shape:
        move = np.linalg.norm((parts_end[finite] - ref[finite]) / _M, axis=1)
        tail_move_mean = float(np.mean(move)); tail_move_max = float(np.max(move))
    else:
        tail_move_mean = tail_move_max = float("nan")
    cup_extent = (2 * _CUP_INNER_R + _CUP_RIM_Z - _CUP_BOTTOM_Z) / _M  # 대략 컵 크기(m)

    exploded = (n_finite < n0) or (bbox.max() > 5.0 * cup_extent)
    contained = frac_in > 0.85
    settled = (tail_move_mean < 0.01) if not math.isnan(tail_move_mean) else False

    print(f"[P1b] finite {n_finite}/{n0} | bbox(m)=[{bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f}] "
          f"(컵~{cup_extent:.3f}m) | frac_in_cup={frac_in:.2f} ({n_in}) | "
          f"tail_move mean={tail_move_mean*1000:.2f}mm max={tail_move_max*1000:.2f}mm", flush=True)
    verdict = "STABLE" if (not exploded and contained and settled) else "UNSTABLE"
    print(f"[P1b] ===== 판정: {verdict} | scale={'cm' if args.cm else 'meter'} "
          f"(exploded={exploded}, contained={contained}, settled={settled}) =====", flush=True)
    print("[P1b] STABLE(meter) → 대안 B 성립: 전체 meter 로 P2/P3/P4 진행 가능", flush=True)

    app.close()


if __name__ == "__main__":
    main()
