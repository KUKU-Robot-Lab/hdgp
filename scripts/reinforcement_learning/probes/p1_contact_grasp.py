"""P1 — raw-app contact grasp 검증 (live_policy_fluid_plan P1).

목표: SimulationContext(텐서 파이프라인) 없이 raw Isaac Sim(omni.timeline + app.update)에서
로봇 손가락이 **dynamic 컵을 실제 contact + friction 으로 잡고 유지/리프트**하는지 검증한다.
grasp_warm_tesollo.hdf5 의 성공 파지 상태(arm 7D + hand 20D + cup pose)를 그대로 재구성한다.

왜 meter 기본인가:
- P1 은 grasp 단독(유체 없음) → PBD 불필요 → cm(×100) 불필요.
- cm 는 PhysX articulation contact 에서 알려진 불안정 요인(contact offset/mass/inertia/drive gain 이
  Xform scale 을 정확히 안 따름). meter(로봇 네이티브)에서 먼저 검증해 **텐서 파이프라인 효과**를
  스케일 문제와 분리한다. --cm 으로 cm 도 시험 가능(대안 A/B 결정용).

좌표계: pour_v1 env 는 로봇 베이스를 pos=[0,0,0]·rot=identity 로 스폰(cfg 676-678) →
env 프레임 = 월드 프레임 → warm `cup_pos_local` 이 그대로 월드 컵 위치.

실행: ./IsaacLab/isaaclab.sh -p hdgp/scripts/reinforcement_learning/probes/p1_contact_grasp.py --headless
"""

import argparse
import math
import os

import numpy as np

_HDGP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

parser = argparse.ArgumentParser(description="P1 raw-app contact grasp 검증")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--warm", type=str,
                    default=os.path.join(_HDGP_ROOT, "data", "grasp_warm_tesollo.hdf5"))
parser.add_argument("--robot_usd", type=str,
                    default=os.path.join(_HDGP_ROOT, "assets", "robot",
                                         "openarm_tesollo_sensor_rl", "openarm_tesollo_sensor_rl.usd"))
parser.add_argument("--cup_usd", type=str, default="",
                    help="컵 USD (미지정 시 meter=cup_big_sdf.usd / cm=cup_big_sdf_cm.usd 자동).")
parser.add_argument("--states", type=str, default="0,1,2,3,4",
                    help="검증할 warm state 인덱스(콤마) 또는 'random:N'.")
parser.add_argument("--cm", action="store_true", default=False, help="cm 스케일(×100, mpu=0.01).")
parser.add_argument("--settle_steps", type=int, default=90, help="파지 안착 스텝(gravity 전).")
parser.add_argument("--hold_steps", type=int, default=300, help="gravity 하 유지 측정 스텝.")
parser.add_argument("--lift", action="store_true", default=False, help="유지 후 팔 리프트(r_aj_2) 시험.")
parser.add_argument("--lift_delta", type=float, default=0.15, help="리프트 시 r_aj_2 목표 증분(rad).")
parser.add_argument("--friction", type=float, default=1.2, help="컵/손 static·dynamic friction.")
parser.add_argument("--arm_stiffness", type=float, default=1.0e6)
parser.add_argument("--arm_damping", type=float, default=1.0e4)
parser.add_argument("--hand_stiffness", type=float, default=2.0e4)
parser.add_argument("--hand_damping", type=float, default=4.0e2)
parser.add_argument("--capture_dir", type=str, default="")
parser.add_argument("--control_drop", action="store_true", default=False,
                    help="[판독 검증] 컵을 grasp 위치보다 +20cm 위에 배치(접촉 없음) → 낙하해야 정상.")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless})

import carb  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402

# physics 결과를 USD stage 로 write-back 강제 → ComputeLocalToWorldTransform 이 live pose 읽음.
# Fabric(scene delegate)이 켜지면 physics 는 Fabric 에만 쓰고 USD stage xformOp 은 stale 해진다.
# → Fabric transform 경로 끄고 physics→USD 직접 기록.
_cs = carb.settings.get_settings()
_cs.set_bool("/app/useFabricSceneDelegate", False)
_cs.set_bool("/physics/fabricUpdateTransformations", False)
_cs.set_bool("/physics/updateToUsd", True)
_cs.set_bool("/physics/updateVelocitiesToUsd", True)
_cs.set_bool("/physics/updateParticlesToUsd", True)
import h5py  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402

_M = 100.0 if args.cm else 1.0            # 길이 스케일(월드 단위/미터)
_MPU = 0.01 if args.cm else 1.0           # metersPerUnit

ARM_JOINTS = [f"r_aj_{i}" for i in range(1, 8)]
_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
HAND_JOINTS = [f"r_hj_{fg}_{j}" for fg in _FINGERS for j in range(1, 5)]  # warm hand 20 순서
FINGER_LINK_PREFIXES = tuple(f"r_hl_{fg}" for fg in _FINGERS)             # collision 유지 대상


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _quat_wxyz_to_gf(q):
    return Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _set_pose(prim, pos_world, quat_wxyz):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(float(pos_world[0]), float(pos_world[1]), float(pos_world[2])))
    x.AddOrientOp().Set(_quat_wxyz_to_gf(quat_wxyz))


def _prim_world_pose(prim):
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    im = q.GetImaginary()
    return (np.array([t[0], t[1], t[2]], dtype=np.float64),
            np.array([q.GetReal(), im[0], im[1], im[2]], dtype=np.float64))


def _set_kinematic(prim, kin):
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr().Set(bool(kin))


def _find_prim_by_name(root, name):
    for p in Usd.PrimRange(root):
        if p.GetName() == name:
            return p
    return None


def _make_friction_material(stage, path, mu):
    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr().Set(float(mu))
    pm.CreateDynamicFrictionAttr().Set(float(mu))
    pm.CreateRestitutionAttr().Set(0.0)
    return mat.GetPrim()


def _bind_material(prim, mat_prim):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material(mat_prim), bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics")


def _setup_drives(stage, robot_prim):
    """관절 이름 → DriveAPI (고강성). 반환: {joint_name: (targetAttr, is_revolute)}."""
    drives = {}
    for p in Usd.PrimRange(robot_prim):
        jname = p.GetName()
        is_rev = p.IsA(UsdPhysics.RevoluteJoint)
        is_prism = p.IsA(UsdPhysics.PrismaticJoint)
        if not (is_rev or is_prism):
            continue
        if jname in ARM_JOINTS:
            stiff, damp = args.arm_stiffness, args.arm_damping
        elif jname in HAND_JOINTS:
            stiff, damp = args.hand_stiffness, args.hand_damping
        else:
            stiff, damp = args.arm_stiffness, args.arm_damping  # 왼팔 등 고정
        dtype = "angular" if is_rev else "linear"
        drive = UsdPhysics.DriveAPI.Apply(p, dtype)
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(float(stiff))
        drive.CreateDampingAttr().Set(float(damp))
        drive.CreateMaxForceAttr().Set(1.0e10)
        jstate = PhysxSchema.JointStateAPI.Apply(p, dtype)  # 초기 관절 위치 지정용
        drives[jname] = (drive.CreateTargetPositionAttr(), is_rev,
                         jstate.CreatePositionAttr())
    return drives


def _drive_to(drives, joint_pos_dict, set_init=False):
    """drive target 설정. set_init=True 면 초기 관절 위치(JointStateAPI)도 같이 지정
    → t=0 부터 해당 자세로 시작(침투 없는 valid grasp 재구성)."""
    for jname, val in joint_pos_dict.items():
        if jname not in drives:
            continue
        attr, is_rev, init_attr = drives[jname]
        v = float(val) * (180.0 / math.pi if is_rev else _M)  # revolute=degree
        attr.Set(v)
        if set_init:
            init_attr.Set(v)


# ----------------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------------
def build_scene():
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
    sapi.CreateEnableCCDAttr().Set(True)

    dl = UsdLux.DistantLight.Define(stage, "/World/light")
    dl.CreateIntensityAttr().Set(3000.0)
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr().Set(1000.0)

    ground = UsdGeom.Cube.Define(stage, "/World/ground")
    ground.CreateSizeAttr().Set(1.0)
    from omni.physx.scripts import physicsUtils
    physicsUtils.set_or_add_scale_op(ground, Gf.Vec3f(4.0 * _M, 4.0 * _M, 0.4 * _M))
    physicsUtils.set_or_add_translate_op(ground, Gf.Vec3f(0.0, 0.0, -0.2 * _M))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    return stage


def load_robot(stage):
    # replay 패턴: 스케일/배치는 깨끗한 부모 Xform 에, 관절 로봇은 자식 참조로.
    # (참조 앵커에 직접 scale op 추가 시 기존 double3 precision 과 충돌.)
    root = stage.DefinePrim("/World/RobotRoot", "Xform")
    xr = UsdGeom.Xformable(root)
    xr.ClearXformOpOrder()
    if args.cm:
        xr.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(_M, _M, _M))
    add_reference_to_stage(args.robot_usd, "/World/RobotRoot/Robot")
    robot_prim = stage.GetPrimAtPath("/World/RobotRoot/Robot")
    # 손가락 링크 collision 은 USD 기본값 유지(replay 처럼 끄지 않음). 왼팔/몸통 등 나머지는
    # 유지해도 무방(컵과 무관). 로봇 gravity 는 켠 채로 두되 arm 은 고강성 drive 가 자세 유지.
    palm = _find_prim_by_name(robot_prim, "r_hl_palm")
    return robot_prim, palm


def load_cup(stage, mu_mat):
    cup_usd = args.cup_usd or os.path.join(
        _HDGP_ROOT, "assets", "cup", "cup_big_sdf_cm.usd" if args.cm else "cup_big_sdf.usd")
    add_reference_to_stage(cup_usd, "/World/Cup")
    # cup USD default prim(/cup_big Xform)이 RigidBodyAPI 보유 → 참조 앵커 /World/Cup 이 rigid body.
    # 자식 mesh(/World/Cup/cup_big)는 collider. rigid body 는 /World/Cup 하나만(중첩 금지).
    cup_prim = stage.GetPrimAtPath("/World/Cup")
    if not cup_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(cup_prim)
    # dynamic + gravity ON (replay 는 DisableGravity=True 로 float — 여기선 반대).
    PhysxSchema.PhysxRigidBodyAPI.Apply(cup_prim).CreateDisableGravityAttr().Set(False)
    # 질량 명시(자동계산 불안정 방지) — 가벼운 컵 ~50g.
    UsdPhysics.MassAPI.Apply(cup_prim).CreateMassAttr().Set(0.05)
    for p in Usd.PrimRange(cup_prim):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            _bind_material(p, mu_mat)
    return cup_prim


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse_states(f):
    n = int(f["warm_states/arm_joint_pos"].shape[0])
    if args.states.startswith("random:"):
        k = int(args.states.split(":")[1])
        rng = np.random.default_rng(0)
        return sorted(rng.choice(n, size=min(k, n), replace=False).tolist())
    return [int(x) for x in args.states.split(",") if x.strip() != "" and int(x) < n]


def main():
    f = h5py.File(args.warm, "r")
    ws = f["warm_states"]
    state_ids = parse_states(f)
    print(f"[P1] scale={'cm(×100)' if args.cm else 'meter'} | warm states {state_ids} | "
          f"friction={args.friction}", flush=True)

    stage = build_scene()
    mu_mat = _make_friction_material(stage, "/World/frictionMat", args.friction)
    robot_prim, palm = load_robot(stage)
    drives = _setup_drives(stage, robot_prim)
    # 손가락 링크 collision 에도 마찰 재질 바인딩
    for p in Usd.PrimRange(robot_prim):
        if p.GetName().startswith(FINGER_LINK_PREFIXES) and p.HasAPI(UsdPhysics.CollisionAPI):
            _bind_material(p, mu_mat)
    cup = load_cup(stage, mu_mat)
    print(f"[P1] 로봇 drive {len(drives)}개 | palm={'OK' if palm else 'MISSING'}", flush=True)

    tl = omni.timeline.get_timeline_interface()

    results = []
    for sid in state_ids:
        arm = np.asarray(ws["arm_joint_pos"][sid], dtype=np.float64)      # (7,)
        hand = np.asarray(ws["hand_joint_pos"][sid], dtype=np.float64)    # (20,)
        cup_pos = np.asarray(ws["cup_pos_local"][sid], dtype=np.float64)  # (3,) env=world 프레임(m)
        cup_quat = np.asarray(ws["cup_quat_wxyz"][sid], dtype=np.float64)  # (4,) wxyz
        n_contact = float(ws["num_contacts"][sid])

        jd = {ARM_JOINTS[i]: arm[i] for i in range(7)}
        jd.update({HAND_JOINTS[i]: hand[i] for i in range(20)})
        cup_place = cup_pos.copy()
        if args.control_drop:
            cup_place[2] += 0.20   # 손 위 20cm — 접촉 없음, 낙하해야 함(read 검증)

        tl.stop()
        # arm+hand 를 warm grasp 자세로 **초기화**(t=0 부터 valid 파지, 침투/드래그 없음).
        _drive_to(drives, jd, set_init=True)
        _set_pose(cup, cup_place * _M, cup_quat)   # 컵 dynamic + gravity(load_cup 설정)
        tl.play()
        p_cup_placed, _ = _prim_world_pose(cup)    # 배치 직후 기준(z)

        # (1) 짧은 안착: contact 등록(자세 이미 grasp 라 컵 거의 안 움직임).
        for _ in range(args.settle_steps):
            app.update()
        p_palm0, _ = _prim_world_pose(palm) if palm else (np.zeros(3), None)
        p_cup0, q_cup0 = _prim_world_pose(cup)
        rel0 = (p_cup0 - p_palm0) / _M

        # (2) 유지: 손가락 contact+friction 만으로 컵을 잡고 있나.
        for _ in range(args.hold_steps):
            app.update()
        p_palm1, _ = _prim_world_pose(palm) if palm else (np.zeros(3), None)
        p_cup1, q_cup1 = _prim_world_pose(cup)
        rel1 = (p_cup1 - p_palm1) / _M
        slip = float(np.linalg.norm(rel1 - rel0))              # palm 기준 컵 이동(m)
        world_drop = float((p_cup_placed[2] - p_cup1[2]) / _M)  # 배치→최종 총 z 낙하(m)

        # (3) 선택 리프트: r_aj_2 를 올려 컵이 따라 들리나.
        lift_slip = None
        if args.lift:
            jd2 = dict(jd)
            jd2["r_aj_2"] = arm[1] - args.lift_delta   # shoulder pitch 올림(부호는 실측 조정)
            _drive_to(drives, jd2)
            for _ in range(args.hold_steps):
                app.update()
            p_palm2, _ = _prim_world_pose(palm)
            p_cup2, _ = _prim_world_pose(cup)
            rel2 = (p_cup2 - p_palm2) / _M
            lift_slip = float(np.linalg.norm(rel2 - rel0))

        held = slip < 0.03 and world_drop < 0.05
        results.append(dict(sid=sid, n_contact=n_contact, rel0=rel0.tolist(),
                            slip=slip, world_drop=world_drop, lift_slip=lift_slip,
                            held=held, cup_z_final=float(p_cup1[2] / _M)))
        print(f"[P1] state {sid:4d} | contacts(warm)={n_contact:.0f} | "
              f"slip={slip*1000:6.1f}mm | world_drop={world_drop*1000:6.1f}mm | "
              f"lift_slip={'-' if lift_slip is None else f'{lift_slip*1000:.1f}mm'} | "
              f"{'HELD' if held else 'DROP/SLIP'}", flush=True)

    n_held = sum(1 for r in results if r["held"])
    print(f"\n[P1] ===== 결과: {n_held}/{len(results)} held "
          f"(slip<30mm & drop<50mm) | scale={'cm' if args.cm else 'meter'} =====", flush=True)
    print("[P1] 판정: 다수 held → 대안 %s 후보 / 다수 DROP → 파이프라인·마찰·drive 재검토" %
          ("B(meter+PBD)" if not args.cm else "A(cm 재베이킹)"), flush=True)

    app.close()


if __name__ == "__main__":
    main()
