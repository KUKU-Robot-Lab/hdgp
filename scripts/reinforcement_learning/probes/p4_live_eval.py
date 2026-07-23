"""P4 — 정책 궤적 구동 + 물리 파지 + PBD 유체 평가 (live_policy_fluid_plan 최종 실현).

목표: meter 스케일 raw-app(SimulationContext 없음)에서 **정책의 genuine 관절 궤적**(pour_traj.hdf5,
lstm_test2 롤아웃)으로 로봇을 구동하되, 소스 컵을 **kinematic 이 아니라 물리 파지**(dynamic + 손가락
contact, P1) 로 들고 붓고, 컵 안에 **PBD 유체**(P1b) → target 컵 내부 유체 비율 η_ft 측정.

record-replay(kinematic 컵) 대비 업그레이드 = **물리 접촉 파지**. 정책이 유체 미관측이므로 이 궤적은
라이브 실행과 행동 동일(확정 사실). 계획의 궁극 질문 답: 물리 파지가 full pour tilt 를 견디는가 + η_ft.

전제(검증 완료): P1(meter 파지 10/10), P1b(meter PBD STABLE), P2(obs), P3.0(Fabrics), P3.1(정책).

실행:
  ./IsaacLab/isaaclab.sh -p hdgp/scripts/reinforcement_learning/probes/p4_live_eval.py --headless \
      --traj hdgp/log/rl_games/open-tesol/right/pour-v1/lstm_test2/pour_traj.hdf5 --episode 0
"""

import argparse
import math
import os

import numpy as np

_HDGP = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))

parser = argparse.ArgumentParser(description="P4 정책궤적+물리파지+PBD 유체 평가")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--traj", type=str,
                    default=os.path.join(_HDGP, "log/rl_games/open-tesol/right/pour-v1/lstm_test2/pour_traj.hdf5"))
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--robot_usd", type=str,
                    default=os.path.join(_HDGP, "assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd"))
parser.add_argument("--cup_usd", type=str, default=os.path.join(_HDGP, "assets/cup/cup_big_sdf.usd"))
parser.add_argument("--particle_contact", type=float, default=0.006, help="particle contact offset(m).")
parser.add_argument("--fill_height", type=float, default=0.05, help="소스 컵 유체 높이(m).")
parser.add_argument("--settle_steps", type=int, default=40, help="파지+유체 안착(구동 전).")
parser.add_argument("--friction", type=float, default=1.4)
parser.add_argument("--arm_stiffness", type=float, default=1.0e6)
parser.add_argument("--hand_stiffness", type=float, default=3.0e4)
parser.add_argument("--kinematic_cup", action="store_true", default=False,
                    help="[대조] 소스 컵 kinematic(기존 replay 방식) — 물리 파지 대신.")
parser.add_argument("--capture_dir", type=str, default="")
parser.add_argument("--capture_every", type=int, default=6, help="캡처 주기(스텝).")
parser.add_argument("--max_drive", type=int, default=0, help=">0 이면 구동 스텝 제한(조명/디버그용 빠른 실행).")
parser.add_argument("--gui", action="store_true", default=False,
                    help="실시간 GUI 관찰용 플래그(carb 뷰포트 설정 적용).")
parser.add_argument("--cup_style", type=str, default="glass",
                    choices=["glass", "frosted", "opaque_src", "opaque_both"],
                    help="컵 렌더: glass=투명유리(기존), frosted=서리유리(팔 비침 완화), opaque_src=소스 불투명+타겟 유리.")
parser.add_argument("--dome_gray", type=float, default=0.7,
                    help="배경/IBL 밝기(0~1). 1.0=회색 스튜디오(밝게 잘 보임), 낮추면 배경 어두워지되 조명도 약해짐.")
parser.add_argument("--video_out", type=str, default="", help="mp4 출력(미지정 시 capture_dir/p4_ep<N>.mp4).")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless})

import carb  # noqa: E402
import h5py  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from omni.physx.scripts import particleUtils, physicsUtils  # noqa: E402

_cs = carb.settings.get_settings()
_cs.set_bool("/app/useFabricSceneDelegate", False)
_cs.set_bool("/physics/fabricUpdateTransformations", False)
_cs.set_bool("/physics/updateToUsd", True)
_cs.set_bool("/physics/updateParticlesToUsd", True)
# RTX auto-exposure 끄기 + 고정 노출 → pour 중 밝은 유체에 노출이 흔들려 화면이 검어지는 것 방지.
_cs.set_bool("/rtx/post/histogram/enabled", False)
_cs.set_bool("/rtx/post/tonemap/enabled", True)
_cs.set_int("/rtx/post/tonemap/op", 1)
_cs.set_float("/rtx/post/tonemap/cameraShutter", 30.0)
_cs.set_float("/rtx/post/tonemap/fNumber", 4.0)
_cs.set_float("/rtx/post/tonemap/isoValue", 200.0)
# 뷰포트 기본 조명 리그(회색 스튜디오 돔) 끄고 스테이지 라이트만 사용 → 돔 제거.
# GUI·headless 캡처 양쪽 다 적용해야 함(캡처서 미적용 시 리그가 살아 이중 조명→과다노출/하얗게 날아감).
_cs.set_bool("/rtx/useViewLightingMode", False)

# meter 스케일 컵 기하 (cup_big_sdf.usd 기준)
CUP_INNER_R = 0.041
CUP_BOTTOM_Z = -0.077
CUP_RIM_Z = 0.100
_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_PREFIX = tuple(f"r_hl_{fg}" for fg in _FINGERS)


def _quat_apply(q, v):  # wxyz
    q = np.asarray(q, float); v = np.asarray(v, float)
    w, u = q[0], q[1:]
    return v + 2.0 * (w * np.cross(u, v) + np.cross(u, np.cross(u, v)))


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _set_pose(prim, pos, quat_wxyz):
    x = UsdGeom.Xformable(prim); x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    x.AddOrientOp().Set(Gf.Quatf(float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])))


def _prim_pose(prim):
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation(); q = m.ExtractRotationQuat(); im = q.GetImaginary()
    return np.array([t[0], t[1], t[2]]), np.array([q.GetReal(), im[0], im[1], im[2]])


def _friction_mat(stage, path, mu):
    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr().Set(mu); pm.CreateDynamicFrictionAttr().Set(mu)
    pm.CreateRestitutionAttr().Set(0.0)
    return mat.GetPrim()


def _bind_mat(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material(mat), bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")


def _add_cylinder_walls(stage, cup_path, mu, n_seg=28, wall=0.004):
    """replay 식 얇은 실린더 벽 collider (유체 격납). SDF 컵보다 유출 없음.
    cup_path 자식으로 붙어 컵과 함께 이동. 소스 컵은 이 벽을 손가락이 grip."""
    r_in, z_bot, z_top = CUP_INNER_R, CUP_BOTTOM_Z, CUP_RIM_Z
    h = z_top - z_bot
    bottom = UsdGeom.Cylinder.Define(stage, cup_path + "/fbottom")
    bottom.CreateRadiusAttr().Set(r_in + wall); bottom.CreateHeightAttr().Set(wall); bottom.CreateAxisAttr().Set("Z")
    physicsUtils.set_or_add_translate_op(bottom, Gf.Vec3f(0.0, 0.0, z_bot - wall * 0.5))
    UsdPhysics.CollisionAPI.Apply(bottom.GetPrim()); _bind_mat(bottom.GetPrim(), mu)
    R = r_in + wall * 0.5
    seg_w = 2.0 * math.pi * R / n_seg * 1.3
    for k in range(n_seg):
        th = 2.0 * math.pi * k / n_seg
        b = UsdGeom.Cube.Define(stage, f"{cup_path}/fwall_{k:02d}")
        b.CreateSizeAttr().Set(1.0)
        xb = UsdGeom.Xformable(b); xb.ClearXformOpOrder()
        xb.AddTranslateOp().Set(Gf.Vec3d(R * math.cos(th), R * math.sin(th), z_bot + h * 0.5))
        xb.AddOrientOp().Set(Gf.Quatf(math.cos(th / 2.0), Gf.Vec3f(0.0, 0.0, math.sin(th / 2.0))))
        xb.AddScaleOp().Set(Gf.Vec3f(wall, seg_w, h))
        UsdPhysics.CollisionAPI.Apply(b.GetPrim()); _bind_mat(b.GetPrim(), mu)


def _disable_sdf_collision(cup_prim):
    for p in Usd.PrimRange(cup_prim):
        if p.GetName() == "cup_big" and p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr().Set(False)


def _build_particle_system(stage, scene):
    sp = Sdf.Path("/World/particleSystem")
    ps = PhysxSchema.PhysxParticleSystem.Define(stage, sp)
    ps.CreateSimulationOwnerRel().SetTargets([scene.GetPath()])
    pco = args.particle_contact
    ps.CreateParticleContactOffsetAttr().Set(pco)
    ps.CreateMaxVelocityAttr().Set(50.0)
    ps.CreateEnableCCDAttr().Set(True)
    ps.CreateSolverPositionIterationCountAttr().Set(24)
    PhysxSchema.PhysxParticleSmoothingAPI.Apply(ps.GetPrim())
    if args.capture_dir or args.gui:   # 렌더용 isosurface(유체 표면 시각화). 측정은 파티클 위치 기반이라 무관.
        PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(ps.GetPrim())
        ani = PhysxSchema.PhysxParticleAnisotropyAPI.Apply(ps.GetPrim())
        ani.CreateScaleAttr().Set(5.0); ani.CreateMinAttr().Set(1.0); ani.CreateMaxAttr().Set(2.0)
    mtl = []
    omni.kit.commands.execute("CreateAndBindMdlMaterialFromLibrary", mdl_name="OmniPBR.mdl",
                              mtl_name="OmniPBR", mtl_created_list=mtl, bind_selected_prims=False,
                              select_new_prim=False)
    _sh = UsdShade.Shader.Get(stage, mtl[0] + "/Shader")
    if _sh:
        _sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.4, 0.9))
    particleUtils.add_pbd_particle_material(stage, mtl[0], viscosity=0.3, cohesion=0.01,
                                            friction=0.2, surface_tension=0.0072, damping=0.05)
    omni.kit.commands.execute("BindMaterialCommand", prim_path=str(sp), material_path=mtl[0], strength=None)
    return sp, 0.99 * 0.6 * pco


def _fill_local(fill_h, spacing):
    r = CUP_INNER_R - 1.2 * spacing
    z0 = CUP_BOTTOM_Z + 2.0 * spacing
    out, z = [], z0
    while z <= z0 + fill_h:
        x = -r
        while x <= r:
            y = -r
            while y <= r:
                if x * x + y * y <= r * r:
                    out.append((x, y, z))
                y += spacing
            x += spacing
        z += spacing
    return out


def _spawn_fluid(stage, sp, pts_world, rest):
    path = "/World/waterParticles"
    particleUtils.add_physx_particleset_pointinstancer(
        stage, Sdf.Path(path), Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts_world]),
        Vt.Vec3fArray([Gf.Vec3f(0.0)] * len(pts_world)), sp, self_collision=True, fluid=True,
        particle_group=0, particle_mass=0.0, density=1000.0)
    # 프로토타입 Sphere 는 반경 미지정 시 USD 기본값 1.0m → meter 씬에서 입자마다 거대한 구로 렌더됨.
    # rest offset 크기로 축소(측정은 PointInstancer positions 기반이라 렌더 반경과 무관).
    proto = UsdGeom.Sphere.Get(stage, Sdf.Path(path + "/particlePrototype0"))
    if proto:
        proto.CreateRadiusAttr().Set(float(rest))
    return path


def _read_particles(stage):
    pi = UsdGeom.PointInstancer.Get(stage, "/World/waterParticles")
    if not pi:
        return None
    pos = pi.GetPositionsAttr().Get()
    return None if pos is None else np.array([[p[0], p[1], p[2]] for p in pos])


def _frac_in_cup(parts, cup_pos, cup_quat):
    if parts is None or len(parts) == 0:
        return 0.0, 0
    qc = _quat_conj(cup_quat); n = 0
    for p in parts:
        loc = _quat_apply(qc, np.asarray(p) - cup_pos)
        if loc[0]**2 + loc[1]**2 < (CUP_INNER_R + 0.005)**2 and (CUP_BOTTOM_Z - 0.005) < loc[2] < CUP_RIM_Z:
            n += 1
    return n / len(parts), n


def main():
    f = h5py.File(args.traj, "r")
    ep = f[f"ep_{args.episode:03d}"]
    joint_names = [b.decode() for b in f.attrs["joint_names"]]
    jpos = np.asarray(ep["joint_pos"])       # (T, 36)
    src_pose = np.asarray(ep["source_pose"])  # (T,7) env-rel meter wxyz
    tgt_pose = np.asarray(ep["target_pose"])
    T = jpos.shape[0]
    frac_ref = float(ep.attrs.get("bead_frac", -1))
    print(f"[P4] ep{args.episode} | {T} step | joints {len(joint_names)} | "
          f"bead_frac(record)={frac_ref:.3f} | {'KINEMATIC' if args.kinematic_cup else '물리파지'} cup", flush=True)

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    sapi = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    sapi.CreateEnableGPUDynamicsAttr().Set(True); sapi.CreateBroadphaseTypeAttr().Set("GPU")
    sapi.CreateSolverTypeAttr().Set("TGS")
    # ---- 조명: dome(안정적 균일 밝기) + sphere key(입체감). 단, 두 광원 모두 **그림자 캐스팅 OFF**
    #   → 밝기는 안정적이면서 배경에 하드 그림자가 안 생김(dome-only 는 RTX 에서 불안정해 어두워짐).
    _dome = UsdLux.DomeLight.Define(stage, "/World/dome")
    _dome.CreateIntensityAttr().Set(2000.0)
    _c = float(args.dome_gray)   # 배경/IBL 톤(1.0=밝은 회색).
    _dome.CreateColorAttr().Set(Gf.Vec3f(_c, _c, _c))
    UsdLux.ShadowAPI.Apply(_dome.GetPrim()).CreateShadowEnableAttr().Set(False)
    if args.capture_dir:
        # 완만한 fill(상단=로봇 머리 쪽에서 아래로). GUI(dome-only)와 노출을 맞추기 위해 약하게.
        _sl = UsdLux.SphereLight.Define(stage, "/World/keyLight")
        _sl.CreateIntensityAttr().Set(3000.0); _sl.CreateRadiusAttr().Set(0.3)
        physicsUtils.set_or_add_translate_op(_sl, Gf.Vec3f(0.35, -0.30, 1.1))
        UsdLux.ShadowAPI.Apply(_sl.GetPrim()).CreateShadowEnableAttr().Set(False)  # 그림자 없음
    ground = UsdGeom.Cube.Define(stage, "/World/ground"); ground.CreateSizeAttr().Set(1.0)
    physicsUtils.set_or_add_scale_op(ground, Gf.Vec3f(4.0, 4.0, 0.4))
    physicsUtils.set_or_add_translate_op(ground, Gf.Vec3f(0.0, 0.0, -0.2))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    mu = _friction_mat(stage, "/World/frictionMat", args.friction)

    # 로봇 (meter, 물리 collision 유지)
    add_reference_to_stage(args.robot_usd, "/World/Robot")
    robot = stage.GetPrimAtPath("/World/Robot")
    drives = {}
    for p in Usd.PrimRange(robot):
        jn = p.GetName()
        if jn in joint_names and (p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)):
            is_rev = p.IsA(UsdPhysics.RevoluteJoint)
            stiff = args.hand_stiffness if jn.startswith("r_hj_") else args.arm_stiffness
            dr = UsdPhysics.DriveAPI.Apply(p, "angular" if is_rev else "linear")
            dr.CreateTypeAttr().Set("force"); dr.CreateStiffnessAttr().Set(stiff)
            dr.CreateDampingAttr().Set(stiff * 0.02); dr.CreateMaxForceAttr().Set(1e9)
            js = PhysxSchema.JointStateAPI.Apply(p, "angular" if is_rev else "linear")
            drives[jn] = (dr.CreateTargetPositionAttr(), js.CreatePositionAttr(), is_rev)
        if jn.startswith(FINGER_PREFIX) and p.HasAPI(UsdPhysics.CollisionAPI):
            _bind_mat(p, mu)

    def drive_to(row, set_init=False):
        for jn, val in zip(joint_names, row):
            if jn not in drives:
                continue
            tgt, ini, is_rev = drives[jn]
            v = float(val) * (180.0 / math.pi if is_rev else 1.0)
            tgt.Set(v)
            if set_init:
                ini.Set(v)

    # 소스 컵: SDF collision off + 실린더 벽(유체 격납, replay식). 손가락은 실린더 벽을 grip.
    add_reference_to_stage(args.cup_usd, "/World/SourceCup")
    src_cup = stage.GetPrimAtPath("/World/SourceCup")
    UsdPhysics.RigidBodyAPI.Apply(src_cup)
    PhysxSchema.PhysxRigidBodyAPI.Apply(src_cup).CreateDisableGravityAttr().Set(False)
    UsdPhysics.MassAPI.Apply(src_cup).CreateMassAttr().Set(0.05)
    if args.kinematic_cup:
        UsdPhysics.RigidBodyAPI(src_cup).CreateKinematicEnabledAttr().Set(True)
    _disable_sdf_collision(src_cup)
    _add_cylinder_walls(stage, "/World/SourceCup", mu)
    # 타겟 컵 (kinematic, 궤적 추종 — 받는 컵). SDF off + 실린더 벽.
    add_reference_to_stage(args.cup_usd, "/World/TargetCup")
    tgt_cup = stage.GetPrimAtPath("/World/TargetCup")
    UsdPhysics.RigidBodyAPI.Apply(tgt_cup).CreateKinematicEnabledAttr().Set(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(tgt_cup).CreateDisableGravityAttr().Set(True)
    _disable_sdf_collision(tgt_cup)
    _add_cylinder_walls(stage, "/World/TargetCup", mu)

    # 시각 정리(렌더/GUI 공통): 컵 본체=투명 유리, collision 박스(fwall/fbottom)=숨김
    #   → 유리 컵 형태 + 내부/이송 유체 동시 가시. (없으면 불투명 컵+회색 박스가 지저분)
    if args.capture_dir or args.gui:
        def _mk_glass(name):   # 투명 유리(OmniGlass — 클리어일 때만 사용, 이 입력들은 검증됨)
            m = []
            omni.kit.commands.execute("CreateAndBindMdlMaterialFromLibrary", mdl_name="OmniGlass.mdl",
                                      mtl_name=name, mtl_created_list=m,
                                      bind_selected_prims=False, select_new_prim=False)
            sh = UsdShade.Shader.Get(stage, m[0] + "/Shader")
            if sh:
                sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.85, 0.9, 0.95))
                sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.05)
                sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
            return stage.GetPrimAtPath(m[0])

        def _mk_preview(name, color, opacity, rough=0.55):
            # 표준 UsdPreviewSurface — diffuseColor/opacity 입력이 버전 무관 확실히 작동(빨강 fallback 없음).
            mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
            sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/S")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            if opacity < 1.0:
                sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            return mat.GetPrim()

        _tint = Gf.Vec3f(0.70, 0.80, 0.92)   # 연청색
        if args.cup_style == "opaque_both":      # 양 컵 불투명 → 팔이 어디서도 안 비침(유리 없음)
            _src_mat = _tgt_mat = None            # None = 컵 기본(불투명) 재질 유지
        elif args.cup_style == "opaque_src":     # 소스 불투명, 타겟 투명유리(타겟 뒤 팔은 비침)
            _src_mat = None
            _tgt_mat = _mk_glass("TgtGlass")
        elif args.cup_style == "frosted":        # 양 컵 반투명(서리)
            _src_mat = _tgt_mat = _mk_preview("Frosted", _tint, 0.4, rough=0.7)
        else:                                    # glass: 양 컵 투명(헤드리스서 OmniGlass MDL 은 SdrShaderNode
            #   컴파일 실패→빨강 폴백. UsdPreviewSurface opacity 는 헤드리스/GUI 모두 확실히 작동.
            _src_mat = _tgt_mat = _mk_preview("Glass", _tint, 0.25, rough=0.12)

        for cup_path, _mat in (("/World/SourceCup", _src_mat), ("/World/TargetCup", _tgt_mat)):
            cp = stage.GetPrimAtPath(cup_path)
            if not cp:
                continue
            for q in Usd.PrimRange(cp):
                nm = q.GetName()
                if nm.startswith("fwall_") or nm == "fbottom":
                    UsdGeom.Imageable(q).MakeInvisible()
                elif nm == "cup_big" and _mat is not None:
                    UsdShade.MaterialBindingAPI.Apply(q).Bind(
                        UsdShade.Material(_mat), bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    # PBD 유체
    sysp, rest = _build_particle_system(stage, scene)
    spacing = 2.0 * rest

    # 초기화: 관절/컵을 궤적 첫 프레임으로
    drive_to(jpos[0], set_init=True)
    _set_pose(src_cup, src_pose[0, :3], src_pose[0, 3:7])
    _set_pose(tgt_cup, tgt_pose[0, :3], tgt_pose[0, 3:7])

    # ---- 영상 캡처 설정 (replicator rgb) ----
    capture = None
    if args.capture_dir:
        os.makedirs(args.capture_dir, exist_ok=True)
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.core")
        import omni.replicator.core as rep
        cam = UsdGeom.Camera.Define(stage, "/World/capCam")
        cam.CreateFocalLengthAttr(24.0); cam.CreateHorizontalApertureAttr(20.955)
        # 준정면(+x 기준 약간 우측 오프셋): 완전 dead-front 는 손이 컵과 시선축상 겹쳐 "손가락이 컵에 박힌"
        #   원근 착시가 생김 → 소폭 각을 줘 depth 를 분리(손-컵이 안 겹침)하면서도 정면 느낌 유지.
        ctr = Gf.Vec3d(0.29, -0.02, 0.42)
        eye = Gf.Vec3d(1.55, -0.55, 0.52)
        up = Gf.Vec3d(0, 0, 1)
        fwd = (ctr - eye); fwd = fwd / fwd.GetLength()
        right = Gf.Cross(fwd, up); right = right / right.GetLength()
        tup = Gf.Cross(right, fwd)
        m = Gf.Matrix4d(right[0], right[1], right[2], 0, tup[0], tup[1], tup[2], 0,
                        -fwd[0], -fwd[1], -fwd[2], 0, eye[0], eye[1], eye[2], 1)
        cxf = UsdGeom.Xformable(cam.GetPrim()); cxf.ClearXformOpOrder(); cxf.AddTransformOp().Set(m)
        rp = rep.create.render_product("/World/capCam", (1280, 720))
        capture = rep.AnnotatorRegistry.get_annotator("rgb"); capture.attach(rp)

    def _save_frame(idx):
        if capture is None:
            return
        try:
            from PIL import Image
            tl.pause()   # 물리 정지 → 정적 씬에서 RTX 시간누적(모션 중 언더샘플 방지)
            arr = None
            for _attempt in range(8):   # 충분히 밝아질 때까지 누적(미완성/검은 프레임 방지)
                for _ in range(12):
                    app.update()
                d = capture.get_data()
                if d is None or len(d) == 0:
                    continue
                arr = np.asarray(d)
                if float(arr[..., :3].mean()) > 20.0:   # 전체 평균 밝기 기준(고정 노출이라 안정)
                    break
            if arr is not None:
                Image.fromarray(arr[..., :3]).save(os.path.join(args.capture_dir, f"f{idx:05d}.png"))
            tl.play()   # 물리 재개
        except Exception as exc:
            print(f"[P4] 캡처 실패: {exc}", flush=True)

    tl = omni.timeline.get_timeline_interface()
    tl.stop(); tl.play()

    _fi = [0]

    def _tick():
        app.update()
        if capture is not None and _fi[0] % args.capture_every == 0:
            _save_frame(_fi[0] // args.capture_every)
        _fi[0] += 1

    # 파지 안착 (유체 스폰 전, 컵이 손에 물리적으로 안정)
    for _ in range(args.settle_steps):
        _tick()
    scp0, scq0 = _prim_pose(src_cup)

    # 유체 스폰 (소스 컵 현재 pose 로컬 → world)
    local = _fill_local(args.fill_height, spacing)
    pts_w = [scp0 + _quat_apply(scq0, np.array(p)) for p in local]
    _spawn_fluid(stage, sysp, pts_w, rest)
    n0 = len(pts_w)
    for _ in range(30):   # 유체 컵 안 안착
        _tick()

    # 궤적 구동
    _Tend = min(T, args.max_drive) if args.max_drive > 0 else T
    for t in range(1, _Tend):
        drive_to(jpos[t])
        if args.kinematic_cup:
            _set_pose(src_cup, src_pose[t, :3], src_pose[t, 3:7])
        _set_pose(tgt_cup, tgt_pose[t, :3], tgt_pose[t, 3:7])
        _tick()

    for _ in range(120):   # 말미 안착
        _tick()

    parts = _read_particles(stage)
    tcp, tcq = _prim_pose(tgt_cup)
    scp, scq = _prim_pose(src_cup)
    eta_ft, n_in = _frac_in_cup(parts, tcp, tcq)
    src_frac, _ = _frac_in_cup(parts, scp, scq)
    finite = int(np.isfinite(parts).all(axis=1).sum()) if parts is not None else 0
    # 파지 유지 진단: 소스 컵이 손 근처(궤적 예상)에 있나 vs 바닥 낙하
    cup_dropped = scp[2] < 0.05
    print(f"[P4] --- 결과 (ep{args.episode}) ---", flush=True)
    print(f"[P4] 유체 {finite}/{n0} finite | η_ft(target 내부)={eta_ft:.3f} ({n_in}) | "
          f"소스 잔류={src_frac:.3f} | 소스컵 최종 z={scp[2]:.3f}m {'(낙하!)' if cup_dropped else '(파지 유지)'}", flush=True)
    print(f"[P4] record-replay 기준 η_ft≈0.335 (pour_v1) | record bead_frac(this ep)={frac_ref:.3f}", flush=True)
    ok = (not cup_dropped) and eta_ft > 0.05
    print(f"[P4] ===== {'물리파지 pour 성공 — 유체 이송 발생' if ok else '점검 필요 (파지 낙하 또는 이송 0)'} =====", flush=True)

    # ---- 영상 합성 (ffmpeg) ----
    if args.capture_dir and _fi[0] > 0:
        import glob
        vout = args.video_out or os.path.join(args.capture_dir, f"p4_ep{args.episode}.mp4")
        frames = sorted(glob.glob(os.path.join(args.capture_dir, "f*.png")))
        try:
            import imageio.v2 as iio
            w = iio.get_writer(vout, fps=20, codec="libx264", quality=8, macro_block_size=None)
            for fr in frames:
                w.append_data(iio.imread(fr))
            w.close()
            print(f"[P4] 영상 저장: {vout} ({len(frames)} 프레임)", flush=True)
        except Exception as exc:
            print(f"[P4] 영상 합성 실패({exc}); PNG 프레임은 {args.capture_dir}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
