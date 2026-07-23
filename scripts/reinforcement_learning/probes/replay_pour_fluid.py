# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""[pour_sensor PBD 검증 — Phase 2] 정책 궤적 재생 + 실제 유체 이송 성공률 측정.

record_pour_traj.py 가 저장한 hdf5(컵 포즈 + 로봇 관절각) 를 로드해,
SimulationContext 없는 raw-app cm 스케일 유체 씬에서 두 컵을 기록 궤적대로 kinematic
구동하며 PBD 물을 붓고, **target 컵 안에 담긴 파티클 비율**을 측정한다.
bead 대리지표가 아닌 진짜 유체 이송률로 정책을 평가한다.

- 컵: 비주얼 컵 USD + 내부 얇은 원통 벽(볼록 충돌, 안 샘). verify_fluid_pour.py 와 동일 원리.
- SimulationContext 는 PBD 시뮬을 끄므로 절대 사용 금지 → omni.timeline + app.update.
- 로봇 재생(--with_robot)은 비주얼 목적의 best-effort (관절 kinematic).

실행(GPU 필요, 사용자 요청 시):
  ./IsaacLab/isaaclab.sh -p hdgp/scripts/reinforcement_learning/probes/replay_pour_fluid.py \
      --traj <log_dir>/pour_traj_<stamp>.hdf5 --episodes all --headless
"""

import argparse

parser = argparse.ArgumentParser(description="Replay pour policy trajectory with PBD fluid and measure transfer.")
parser.add_argument("--traj", type=str, required=True, help="record_pour_traj.py 가 저장한 hdf5 경로.")
parser.add_argument("--episodes", type=str, default="all", help="'all' 또는 콤마구분 인덱스(예: 0,3,5).")
parser.add_argument("--headless", action="store_true", default=False, help="헤드리스(창 없음).")
parser.add_argument("--no_isosurface", action="store_true", default=False, help="isosurface 끄고 물방울 구슬 렌더.")
parser.add_argument("--particle_contact", type=float, default=0.008, help="particle contact offset(m).")
parser.add_argument("--fill_height", type=float, default=0.045, help="컵 유체 높이(m).")
parser.add_argument("--spawn_batches", type=int, default=20, help="점진 스폰 배치 수.")
parser.add_argument("--spawn_interval", type=int, default=18, help="배치 간격(스텝).")
parser.add_argument("--fill_frames", type=int, default=380, help="유체 채우는 초기 프레임 수(궤적 시작부, 컵이 대체로 정지인 구간).")
parser.add_argument("--tail_settle", type=int, default=120, help="궤적 종료 후 추가 안착 스텝(측정 전).")
parser.add_argument("--cup_segments", type=int, default=28, help="원통 벽 세그먼트.")
parser.add_argument("--success_frac", type=float, default=0.5, help="유체 이송 성공 임계(target 내부 비율).")
parser.add_argument("--with_robot", action="store_true", default=False, help="로봇 USD 를 관절 궤적으로 비주얼 재생(best-effort).")
parser.add_argument("--weld_cup", action="store_true", default=False, help="[실험] 컵을 palm 에 붙여 손 따라가게(drive 추종 부정확 시 유체 샘 — 기본 off).")
parser.add_argument("--capture_dir", type=str, default=None, help="replicator 프레임 캡처 폴더.")
parser.add_argument("--capture_every", type=int, default=40, help="캡처 주기.")
parser.add_argument("--report_out", type=str, default=None, help="이송 성공률 리포트(md) 저장 경로.")
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": args.headless})

# ---------------------------------------------------------------------------
import math
import os

import h5py
import numpy as np

import omni.kit.commands
import omni.timeline
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
from omni.physx.scripts import particleUtils, physicsUtils

_M = 100.0  # 미터 → cm (metersPerUnit=0.01)

_HDGP_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
_CUP_USD = os.path.join(_HDGP_ROOT, "assets", "cup", "cup_big_sdf_cm.usd")
_CUP_INNER_R = 0.041 * _M
_CUP_BOTTOM_Z = -0.077 * _M
_CUP_RIM_Z = 0.100 * _M


# ---------------------------------------------------------------------------
# 수학 유틸 (wxyz quat)
# ---------------------------------------------------------------------------
def _quat_rotate(q, v):
    """wxyz quat 으로 벡터 회전 (numpy)."""
    w, x, y, z = q
    u = np.array([x, y, z], dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    return (2.0 * np.dot(u, vv) * u
            + (w * w - np.dot(u, u)) * vv
            + 2.0 * w * np.cross(u, vv))


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _pose_to_cm(pose7):
    """기록 pose(env-rel meters, pos3+quat_wxyz4) → (translate_cm np3, Gf.Quatf)."""
    pos_cm = np.asarray(pose7[:3], dtype=np.float64) * _M
    q = np.asarray(pose7[3:7], dtype=np.float64)
    return pos_cm, Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3])))


# ---------------------------------------------------------------------------
# 씬 구성
# ---------------------------------------------------------------------------
def _add_cylinder_cup(stage, path, n_seg, wall=0.4, visual=True):
    """비주얼 컵 USD + 내부 얇은 원통 벽(충돌). verify_fluid_pour.py 와 동일."""
    r_in, z_bot, z_top = _CUP_INNER_R, _CUP_BOTTOM_Z, _CUP_RIM_Z
    h = z_top - z_bot
    prim = stage.DefinePrim(path, "Xform")
    if visual:
        prim.GetReferences().AddReference(_CUP_USD)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr().Set(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(True)
    if visual:
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


def _set_pose(prim, pos_cm, quat):
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(float(pos_cm[0]), float(pos_cm[1]), float(pos_cm[2])))
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            op.Set(quat)


def _setup_robot_drives(stage, robot_prim, joint_names):
    """로봇 관절 drive 를 고강성으로 구성 → (targetPos_attr, scale, col_idx) 목록.
    revolute: target=deg(rad×180/π), prismatic: target=cm(m×_M). fluid 에 안 닿으므로 비주얼 추종용."""
    name_to_col = {n: i for i, n in enumerate(joint_names)}
    drives = []
    for p in Usd.PrimRange(robot_prim):
        is_rev = p.IsA(UsdPhysics.RevoluteJoint)
        is_prism = p.IsA(UsdPhysics.PrismaticJoint)
        if not (is_rev or is_prism):
            continue
        col = name_to_col.get(p.GetName())
        if col is None:
            continue
        dtype = "angular" if is_rev else "linear"
        drive = UsdPhysics.DriveAPI.Apply(p, dtype)
        drive.CreateTypeAttr().Set("force")
        # ×100 스케일 시 관성 폭증 → 타이트 추종 위해 게인 대폭 상향 + maxforce 실질 무한.
        drive.CreateStiffnessAttr().Set(1.0e11)
        drive.CreateDampingAttr().Set(1.0e9)
        drive.CreateMaxForceAttr().Set(1.0e16)
        scale = 180.0 / math.pi if is_rev else _M
        drives.append((drive.CreateTargetPositionAttr(), scale, col))
    return drives


def _drive_robot(drives, jrow):
    for attr, scale, col in drives:
        attr.Set(float(jrow[col]) * scale)


def _find_prim_by_name(root_prim, name):
    for p in Usd.PrimRange(root_prim):
        if p.GetName() == name:
            return p
    return None


def _prim_world_matrix(prim):
    """prim 의 local→world 4x4 (cm, ×100 스케일 포함)."""
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _matrix_to_pose(m):
    """Gf.Matrix4d → (pos_cm np3, Gf.Quatf wxyz). 스케일 제거."""
    xf = Gf.Transform(m)
    t = xf.GetTranslation()
    q = xf.GetRotation().GetQuat()   # Gf.Quatd
    im = q.GetImaginary()
    return (np.array([t[0], t[1], t[2]], dtype=np.float64),
            Gf.Quatf(float(q.GetReal()), Gf.Vec3f(float(im[0]), float(im[1]), float(im[2]))))


def _build_particle_system(stage, scene):
    system_path = Sdf.Path("/World/particleSystem")
    ps = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
    ps.CreateSimulationOwnerRel().SetTargets([scene.GetPath()])
    pco = args.particle_contact * _M
    ps.CreateParticleContactOffsetAttr().Set(pco)
    ps.CreateMaxVelocityAttr().Set(60.0)
    ps.CreateEnableCCDAttr().Set(True)
    ps.CreateSolverPositionIterationCountAttr().Set(24)
    PhysxSchema.PhysxParticleSmoothingAPI.Apply(ps.GetPrim())
    PhysxSchema.PhysxParticleAnisotropyAPI.Apply(ps.GetPrim())
    use_iso = not args.no_isosurface
    if use_iso:
        PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(ps.GetPrim())
        ani = PhysxSchema.PhysxParticleAnisotropyAPI.Apply(ps.GetPrim())
        ani.CreateScaleAttr().Set(5.0); ani.CreateMinAttr().Set(1.0); ani.CreateMaxAttr().Set(2.0)

    mtl_created = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary", mdl_name="OmniPBR.mdl", mtl_name="OmniPBR",
        mtl_created_list=mtl_created, bind_selected_prims=False, select_new_prim=False,
    )
    mat = mtl_created[0]
    shader = UsdShade.Shader.Get(stage, mat + "/Shader")
    if shader:
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.4, 0.9))
    # 물처럼: 낮은 점성/응집/댐핑 → 94° tilt 로도 흘러나옴. cylinder 벽이라 누수 없음.
    particleUtils.add_pbd_particle_material(
        stage, mat, viscosity=0.3, cohesion=0.01, friction=0.2, surface_tension=0.0072, damping=0.05,
    )
    omni.kit.commands.execute("BindMaterialCommand", prim_path=str(system_path), material_path=mat, strength=None)
    fluid_rest = 0.99 * 0.6 * pco
    return system_path, mat, fluid_rest, use_iso


def _fill_local(fill_height_cm, spacing):
    """컵 로컬 프레임(cm) 유체 격자."""
    r_max = _CUP_INNER_R - 1.2 * spacing
    z0 = _CUP_BOTTOM_Z + 2.0 * spacing
    z1 = z0 + fill_height_cm
    out = []
    z = z0
    while z <= z1:
        x = -r_max
        while x <= r_max:
            y = -r_max
            while y <= r_max:
                if x * x + y * y <= r_max * r_max:
                    out.append(np.array([x, y, z], dtype=np.float64))
                y += spacing
            x += spacing
        z += spacing
    return out


def _extend(attr, elems):
    cur = attr.Get()
    attr.Set(list(cur) + list(elems) if cur else list(elems))


def _append_particles(stage, path, positions):
    prim = stage.GetPrimAtPath(path)
    pi = UsdGeom.PointInstancer(prim)
    sp = PhysxSchema.PhysxParticleSetAPI(prim).GetSimulationPointsAttr()
    if not sp.HasAuthoredValue():
        sp.Set(Vt.Vec3fArray([]))
    _extend(sp, positions)
    _extend(pi.GetPositionsAttr(), positions)
    _extend(pi.GetVelocitiesAttr(), [Gf.Vec3f(0.0)] * len(positions))
    _extend(pi.GetProtoIndicesAttr(), [0] * len(positions))
    _extend(pi.GetOrientationsAttr(), [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * len(positions))
    _extend(pi.GetScalesAttr(), [Gf.Vec3f(1.0)] * len(positions))


# ---------------------------------------------------------------------------
# 측정: target 컵 로컬 프레임 기준 내부 파티클 비율
# ---------------------------------------------------------------------------
def _fraction_in_cup(part_world_cm, cup_pos_cm, cup_quat_wxyz):
    if part_world_cm is None or len(part_world_cm) == 0:
        return 0.0, 0
    qc = _quat_conj(cup_quat_wxyz)
    n_in = 0
    for p in part_world_cm:
        local = _quat_rotate(qc, np.asarray(p, dtype=np.float64) - cup_pos_cm)
        r2 = local[0] * local[0] + local[1] * local[1]
        if r2 < (_CUP_INNER_R + 0.5) ** 2 and (_CUP_BOTTOM_Z - 0.5) < local[2] < _CUP_RIM_Z:
            n_in += 1
    return n_in / len(part_world_cm), n_in


def main():
    if not os.path.exists(args.traj):
        raise SystemExit(f"[REPLAY] 궤적 파일 없음: {args.traj}")

    f = h5py.File(args.traj, "r")
    n_ep = int(f.attrs.get("n_episodes", len(list(f.keys()))))
    if args.episodes == "all":
        ep_ids = list(range(n_ep))
    else:
        ep_ids = [int(x) for x in args.episodes.split(",") if x.strip() != ""]
    print(f"[REPLAY] {args.traj}  에피소드 {ep_ids} / 총 {n_ep}", flush=True)

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81 * _M)
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    scene_api.CreateEnableGPUDynamicsAttr().Set(True)
    scene_api.CreateBroadphaseTypeAttr().Set("GPU")
    scene_api.CreateEnableExternalForcesEveryIterationAttr().Set(True)

    _dl = UsdLux.DistantLight.Define(stage, "/World/light")
    _dl.CreateIntensityAttr().Set(8000.0)
    _dl.CreateAngleAttr().Set(1.0)
    UsdGeom.Xformable(_dl.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 20.0))
    _dome = UsdLux.DomeLight.Define(stage, "/World/domeLight")
    _dome.CreateIntensityAttr().Set(1200.0)

    ground = UsdGeom.Cube.Define(stage, "/World/ground")
    ground.CreateSizeAttr().Set(1.0)
    physicsUtils.set_or_add_scale_op(ground, Gf.Vec3f(4.0 * _M, 4.0 * _M, 0.4 * _M))
    physicsUtils.set_or_add_translate_op(ground, Gf.Vec3f(0.0, 0.0, -0.2 * _M))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    gc = PhysxSchema.PhysxCollisionAPI.Apply(ground.GetPrim())
    gc.CreateContactOffsetAttr().Set(2.0); gc.CreateRestOffsetAttr().Set(1.0)

    source_cup = _add_cylinder_cup(stage, "/World/SourceCup", args.cup_segments)
    target_cup = _add_cylinder_cup(stage, "/World/TargetCup", args.cup_segments)

    system_path, render_mat, fluid_rest, use_iso = _build_particle_system(stage, scene)
    spacing = 2.0 * fluid_rest
    local_fill = _fill_local(args.fill_height * _M, spacing)

    # 선택: 로봇 (best-effort) — cm 씬(×100)에 physics articulation 로드 + 고강성 drive 로 기록 joint 추종.
    robot_prim = None
    palm_prim = None
    grip_offset = None   # cup 을 palm 에 붙이는 frame0 오프셋 (Gf.Matrix4d)
    robot_drives = []   # [(drive_attr, scale, col_idx)] col_idx = joint_names 인덱스
    robot_joint_names = [b.decode() if isinstance(b, bytes) else str(b) for b in f.attrs.get("joint_names", [])]
    if args.with_robot:
        robot_usd = f.attrs.get("robot_usd", "")
        robot_usd = robot_usd.decode() if isinstance(robot_usd, bytes) else str(robot_usd)
        if robot_usd and os.path.exists(robot_usd):
            # 배치/스케일은 깨끗한 부모 Xform 에, 관절은 자식 참조 로봇에.
            # (참조 로봇 루트는 quatd orient 를 이미 가져 직접 xform op 추가 시 precision 충돌)
            root_xform = stage.DefinePrim("/World/RobotRoot", "Xform")
            rroot = np.asarray(f.attrs["robot_root"], dtype=np.float64)
            rp, rq = _pose_to_cm(rroot)
            xr = UsdGeom.Xformable(root_xform)
            xr.ClearXformOpOrder()
            xr.AddTranslateOp().Set(Gf.Vec3d(float(rp[0]), float(rp[1]), float(rp[2])))
            xr.AddOrientOp().Set(rq)
            xr.AddScaleOp().Set(Gf.Vec3f(_M, _M, _M))
            robot_prim = stage.DefinePrim("/World/RobotRoot/Robot", "Xform")
            robot_prim.GetReferences().AddReference(robot_usd)
            robot_drives = _setup_robot_drives(stage, robot_prim, robot_joint_names)
            # 로봇은 비주얼 전용 — 충돌 끄기(유체 간섭 방지) + 중력 끄기(스케일된 중력이 추종 방해).
            _ncol = 0
            for _p in Usd.PrimRange(robot_prim):
                if _p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(_p).CreateCollisionEnabledAttr().Set(False)
                    _ncol += 1
                if _p.HasAPI(UsdPhysics.RigidBodyAPI):
                    PhysxSchema.PhysxRigidBodyAPI.Apply(_p).CreateDisableGravityAttr().Set(True)
            palm_prim = _find_prim_by_name(robot_prim, "r_hl_palm")
            print(f"[REPLAY] 로봇 로드: {robot_usd}  드라이브 {len(robot_drives)}/{len(robot_joint_names)}개 구성"
                  f"  palm={'OK' if palm_prim else 'MISSING'}", flush=True)
        else:
            print(f"[REPLAY] 로봇 USD 없음/미기록 → 로봇 생략: '{robot_usd}'", flush=True)

    tl = omni.timeline.get_timeline_interface()
    tl.play()

    capture = None
    if args.capture_dir:
        os.makedirs(args.capture_dir, exist_ok=True)
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.core")
        import omni.replicator.core as rep
        cam = UsdGeom.Camera.Define(stage, "/World/capCam")
        cam.CreateFocalLengthAttr(24.0)
        cam.CreateHorizontalApertureAttr(20.955)
        # 손+컵 그립이 잘 보이도록 3/4 전면 클로즈업 (손은 대략 (30,-8,47), 소스컵 (34,-5,50)).
        _eye = Gf.Vec3d(78.0, -66.0, 60.0)      # cm
        _center = Gf.Vec3d(31.0, -6.0, 44.0)
        _up = Gf.Vec3d(0.0, 0.0, 1.0)
        _fwd = (_center - _eye).GetNormalized()          # 카메라 -Z = 시선
        _right = Gf.Cross(_fwd, _up).GetNormalized()
        _tup = Gf.Cross(_right, _fwd).GetNormalized()
        _m = Gf.Matrix4d(
            _right[0], _right[1], _right[2], 0.0,
            _tup[0], _tup[1], _tup[2], 0.0,
            -_fwd[0], -_fwd[1], -_fwd[2], 0.0,
            _eye[0], _eye[1], _eye[2], 1.0,
        )
        _cxf = UsdGeom.Xformable(cam.GetPrim())
        _cxf.ClearXformOpOrder()
        _cxf.AddTransformOp().Set(_m)
        rp_ = rep.create.render_product("/World/capCam", (1280, 720))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach(rp_)
        capture = annot
        # 캡처 시 컵 벽 콜라이더 숨김 → 내부 유체(입자)가 보이게 (측정/물리 무관, 시각용).
        # (반투명은 헤드리스 RTX 에서 투명 렌더 안 되어 오히려 유체 가림 → 숨김이 최선.)
        for _cup_path in ("/World/SourceCup", "/World/TargetCup"):
            _cp = stage.GetPrimAtPath(_cup_path)
            if _cp:
                for _q in Usd.PrimRange(_cp):
                    _qn = _q.GetName()
                    if _qn.startswith("wall_") or _qn == "bottom":
                        UsdGeom.Imageable(_q).MakeInvisible()

    def read_particle_world():
        pi = UsdGeom.PointInstancer.Get(stage, "/World/waterParticles")
        if not pi:
            return []
        pos = pi.GetPositionsAttr().Get()
        return [np.array([p[0], p[1], p[2]], dtype=np.float64) for p in pos] if pos else []

    results = []   # (ep_id, bead_frac, fluid_frac, n_part, n_in)

    for ep in ep_ids:
        g = f[f"ep_{ep:03d}"]
        src = np.asarray(g["source_pose"])   # [T,7] env-rel meters
        tgt = np.asarray(g["target_pose"])
        jpos = np.asarray(g["joint_pos"]) if "joint_pos" in g else None
        bead_frac = float(g.attrs.get("bead_frac", -1))
        T = src.shape[0]

        # 이전 에피소드 파티클 제거 후 재생성 (배치0 스폰)
        if stage.GetPrimAtPath("/World/waterParticles"):
            stage.RemovePrim("/World/waterParticles")

        # 컵 초기 포즈 설정
        sp0, sq0 = _pose_to_cm(src[0]); _set_pose(source_cup, sp0, sq0)
        tp0, tq0 = _pose_to_cm(tgt[0]); _set_pose(target_cup, tp0, tq0)
        # 로봇 초기 관절 타깃(프레임0) — 큰 초기 점프/폭발 방지 위해 미리 세팅 후 정착
        if robot_drives and jpos is not None:
            _drive_robot(robot_drives, jpos[0])
            for _ in range(30):
                app.update()
        for _ in range(2):
            app.update()
        # weld: frame0 에서 cup 을 palm 에 고정하는 오프셋 계산 → 이후 cup 은 손을 따라감(grasp 시각화)
        grip_offset = None
        if args.weld_cup and palm_prim is not None and robot_drives:
            palm_m0 = _prim_world_matrix(palm_prim)
            cup_m0 = _prim_world_matrix(source_cup)
            grip_offset = cup_m0 * palm_m0.GetInverse()   # cup = grip_offset * palm

        # 유체 격자를 source 컵 프레임0 world 로 변환 → 층별 배치 준비
        def world_fill(pose7):
            pos_cm, q = _pose_to_cm(pose7)
            qn = np.asarray(pose7[3:7], dtype=np.float64)
            return [Gf.Vec3f(*(_quat_rotate(qn, lp) + pos_cm).astype(np.float32).tolist()) for lp in local_fill]

        all_w = world_fill(src[0])
        all_w_sorted = sorted(range(len(all_w)), key=lambda i: local_fill[i][2])  # 바닥층부터
        order = all_w_sorted
        nb = max(1, args.spawn_batches)
        bsz = max(1, math.ceil(len(order) / nb))
        idx_batches = [order[i:i + bsz] for i in range(0, len(order), bsz)]

        # 배치0 로 particle set 생성
        b0 = [world_fill(src[0])[i] for i in idx_batches[0]]
        particleUtils.add_physx_particleset_pointinstancer(
            stage, Sdf.Path("/World/waterParticles"),
            Vt.Vec3fArray(b0), Vt.Vec3fArray([Gf.Vec3f(0.0)] * len(b0)),
            system_path, self_collision=True, fluid=True, particle_group=0,
            particle_mass=0.001, density=0.0,
        )
        proto = UsdGeom.Sphere.Get(stage, "/World/waterParticles/particlePrototype0")
        if proto:
            proto.CreateRadiusAttr().Set(fluid_rest)
            omni.kit.commands.execute("BindMaterialCommand", prim_path=str(proto.GetPath()),
                                      material_path=render_mat, strength=None)
        pending = idx_batches[1:]

        print(f"[REPLAY] ep {ep}: T={T} 프레임, 유체 {len(order)}개, bead_frac={bead_frac:.2f}", flush=True)

        # --- 재생 루프 ---
        # 채우기: fill_frames 동안 배치를 현재 source 컵 프레임으로 변환해 추가.
        # 그 뒤 궤적을 그대로 재생(컵이 움직이며 붓기).
        bi = 0
        fill_span = min(args.fill_frames, T)
        total_frames = T + args.tail_settle
        for step in range(total_frames):
            fi = min(step, T - 1)   # 궤적 프레임 인덱스 (tail 은 마지막 유지)
            if robot_drives and jpos is not None:
                _drive_robot(robot_drives, jpos[fi])
            # 소스 컵: weld 모드면 손(palm)을 따라가고, 아니면 기록 포즈로 kinematic 구동.
            # 유체 fill 도 동일한 '현재 컵 포즈'(cur_src7)를 기준으로 해야 어긋나지 않음.
            if grip_offset is not None and palm_prim is not None:
                cup_m = grip_offset * _prim_world_matrix(palm_prim)
                sp, sq = _matrix_to_pose(cup_m)
                _im = sq.GetImaginary()
                cur_src7 = np.array([sp[0] / _M, sp[1] / _M, sp[2] / _M,
                                     float(sq.GetReal()), float(_im[0]), float(_im[1]), float(_im[2])],
                                    dtype=np.float64)
            else:
                cur_src7 = src[fi]
                sp, sq = _pose_to_cm(cur_src7)
            _set_pose(source_cup, sp, sq)
            tp, tq = _pose_to_cm(tgt[fi]); _set_pose(target_cup, tp, tq)

            # 점진 스폰 (채우기 구간, 현재 source 프레임으로 변환)
            if bi < len(pending) and step > 0 and step < fill_span and step % args.spawn_interval == 0:
                wf = world_fill(cur_src7)
                _append_particles(stage, "/World/waterParticles", [wf[i] for i in pending[bi]])
                bi += 1

            app.update()
            if capture is not None and step % args.capture_every == 0:
                _save_frame(capture, os.path.join(args.capture_dir, f"ep{ep:02d}_{step:05d}.png"))

            # per-step 진단: 유체가 어느 컵/바닥에 있는지 추적 (원인 국소화용)
            if step % 50 == 0 or step == total_frames - 1:
                _dp = read_particle_world()
                _spc, _ = _pose_to_cm(src[fi]); _sqn = np.asarray(src[fi][3:7], dtype=np.float64)
                _tpc, _ = _pose_to_cm(tgt[fi]); _tqn = np.asarray(tgt[fi][3:7], dtype=np.float64)
                _, _nsrc = _fraction_in_cup(_dp, _spc, _sqn)
                _, _ntgt = _fraction_in_cup(_dp, _tpc, _tqn)
                _zmin = min((float(p[2]) for p in _dp), default=0.0)
                _palm_str = ""
                if palm_prim is not None:
                    _pmw, _ = _matrix_to_pose(_prim_world_matrix(palm_prim))
                    _dsrc = float(np.linalg.norm(np.asarray(_pmw) - _spc))
                    _dtgt = float(np.linalg.norm(np.asarray(_pmw) - _tpc))
                    _palm_str = (f" palm=({_pmw[0]:.0f},{_pmw[1]:.0f},{_pmw[2]:.0f}) "
                                 f"src=({_spc[0]:.0f},{_spc[1]:.0f},{_spc[2]:.0f}) "
                                 f"tgt=({_tpc[0]:.0f},{_tpc[1]:.0f},{_tpc[2]:.0f}) "
                                 f"d_src={_dsrc:.1f} d_tgt={_dtgt:.1f}")
                print(f"[REPLAY]   ep{ep} step {step}/{total_frames} n={len(_dp)} "
                      f"in_src={_nsrc} in_tgt={_ntgt} zmin={_zmin:.1f}{_palm_str}", flush=True)

        # --- 측정: 최종 target 컵 프레임 기준 내부 비율 ---
        parts = read_particle_world()
        tpf, tqf = _pose_to_cm(tgt[T - 1])
        tqn = np.asarray(tgt[T - 1][3:7], dtype=np.float64)
        fluid_frac, n_in = _fraction_in_cup(parts, tpf, tqn)
        results.append((ep, bead_frac, fluid_frac, len(parts), n_in))
        print(f"[REPLAY] ep {ep}: 유체 이송 {n_in}/{len(parts)} = {fluid_frac*100:.1f}%  "
              f"(bead {bead_frac*100:.0f}%)", flush=True)

    f.close()

    # --- 집계 리포트 ---
    _report(results)

    import threading
    import time as _time
    threading.Thread(target=app.close, daemon=True).start()
    _time.sleep(3.0)
    os._exit(0)


def _save_frame(annot, path):
    try:
        from PIL import Image
        data = annot.get_data()
        if data is not None and len(data) > 0:
            Image.fromarray(data[..., :3]).save(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[REPLAY] 캡처 실패: {exc}", flush=True)


def _report(results):
    if not results:
        print("[REPLAY] 결과 없음.", flush=True)
        return
    fl = [r[2] for r in results]
    succ = sum(1 for x in fl if x >= args.success_frac)
    lines = []
    lines.append("# Pour 정책 실제 유체 이송 평가 (PBD replay)")
    lines.append("")
    lines.append(f"- 궤적: `{args.traj}`")
    lines.append(f"- 에피소드 수: {len(results)}  |  성공 임계: 유체 이송률 ≥ {args.success_frac:.2f}")
    lines.append(f"- **유체 이송 성공률**: **{succ/len(results)*100:.1f}%** ({succ}/{len(results)})")
    lines.append(f"- **평균 유체 이송률**: **{np.mean(fl)*100:.1f}%**  (범위 {min(fl)*100:.0f}~{max(fl)*100:.0f}%)")
    lines.append("")
    lines.append("| ep | bead_frac | 유체 이송률 | target 내부 | 총 파티클 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for ep, bf, ff, n, nin in results:
        lines.append(f"| {ep} | {bf*100:.0f}% | {ff*100:.1f}% | {nin} | {n} |")
    report = "\n".join(lines)
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w") as fh:
            fh.write(report)
        print(f"[REPLAY] 리포트 저장: {args.report_out}", flush=True)


if __name__ == "__main__":
    main()
