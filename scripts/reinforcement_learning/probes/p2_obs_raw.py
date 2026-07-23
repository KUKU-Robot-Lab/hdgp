"""P2 Level-2 raw-app USD 읽기 검증.

raw Isaac Sim(SimulationContext 없음, meter 스케일=대안 B)에서 덤프 상태(p2_ref.npz)로 로봇 관절·
두 컵을 배치한 뒤, **USD/physx 로 상태를 읽어** actor obs 55D 를 재조립 → 덤프 obs 와 비교.
P3 라이브 루프의 핵심 과제 "raw-app 관절 상태 읽기" 를 함께 검증한다.

joint 읽기: physx 아티큘레이션 뷰(omni.physics.tensors) — SimulationContext 없이 뷰만 생성(PBD 무해,
P2 엔 PBD 없음). 실패 시 JointStateAPI fallback.

실행:
  ./IsaacLab/isaaclab.sh -p hdgp/scripts/reinforcement_learning/probes/p2_obs_raw.py --headless \
      --ref hdgp/docs/eval/p2_ref.npz --env_idx 0
"""

import argparse
import math
import os
import sys

import numpy as np

_HDGP_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.dirname(__file__))

parser = argparse.ArgumentParser(description="P2 raw-app obs read validation")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--ref", type=str, default=os.path.join(_HDGP_ROOT, "docs", "eval", "p2_ref.npz"))
parser.add_argument("--env_idx", type=int, default=0, help="검증할 덤프 env 인덱스.")
parser.add_argument("--robot_usd", type=str,
                    default=os.path.join(_HDGP_ROOT, "assets", "robot",
                                         "openarm_tesollo_sensor_rl", "openarm_tesollo_sensor_rl.usd"))
parser.add_argument("--cup_usd", type=str, default=os.path.join(_HDGP_ROOT, "assets", "cup", "cup_big_sdf.usd"))
parser.add_argument("--settle_steps", type=int, default=20)
parser.add_argument("--no_collision", action="store_true", default=False,
                    help="로봇 collision off → 접촉 confound 제거, 읽기+조립 파이프라인 순수 검증.")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless})

import carb  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from omni.physx.scripts import physicsUtils  # noqa: E402

import p2_reconstruct as R  # geometry/assembly 재사용  # noqa: E402

_cs = carb.settings.get_settings()
_cs.set_bool("/app/useFabricSceneDelegate", False)
_cs.set_bool("/physics/fabricUpdateTransformations", False)
_cs.set_bool("/physics/updateToUsd", True)

ARM = [f"r_aj_{i}" for i in range(1, 8)]
_FG = ["thumb", "index", "middle", "ring", "pinky"]
HAND = [f"r_hj_{fg}_{j}" for fg in _FG for j in range(1, 5)]
LARM = [f"l_aj_{i}" for i in range(1, 8)] + ["l_hj_gripper_1", "l_hj_gripper_2"]


def _set_pose(prim, pos, quat_wxyz):
    x = UsdGeom.Xformable(prim); x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    x.AddOrientOp().Set(Gf.Quatf(float(quat_wxyz[0]), float(quat_wxyz[1]),
                                 float(quat_wxyz[2]), float(quat_wxyz[3])))


def _prim_quat_pos(prim):
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation(); q = m.ExtractRotationQuat(); im = q.GetImaginary()
    return (np.array([t[0], t[1], t[2]]), np.array([q.GetReal(), im[0], im[1], im[2]]))


def main():
    d = {k: v for k, v in np.load(os.path.abspath(args.ref)).items()}
    i = args.env_idx
    origin = d["env_origins"][i]

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)   # meter (대안 B)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    scene.CreateGravityMagnitudeAttr().Set(0.0)   # obs 읽기만 — 중력 불필요(상태 고정)
    sapi = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    sapi.CreateEnableGPUDynamicsAttr().Set(True); sapi.CreateBroadphaseTypeAttr().Set("GPU")
    UsdLux.DistantLight.Define(stage, "/World/light").CreateIntensityAttr().Set(3000.0)

    # 로봇: 베이스 origin(=env origin 로컬 프레임). 관절을 덤프값으로 초기화 + drive 고정.
    add_reference_to_stage(args.robot_usd, "/World/Robot")
    robot = stage.GetPrimAtPath("/World/Robot")
    if args.no_collision:
        # 접촉 물리 분리: collision 끄면 gravity=0 + 고강성 drive 로 관절이 정확히 target 유지
        #   → 읽기+조립 파이프라인만 순수 검증(손가락 접촉 평형 confound 제거).
        for p in Usd.PrimRange(robot):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr().Set(False)
    jd = {}
    for k, names, arr in (("a", ARM, d["arm_joint_pos"][i]),
                          ("h", HAND, d["finger_joint_pos"][i]),
                          ("l", LARM, d["left_arm_joint_pos"][i])):
        for jn, val in zip(names, arr):
            jd[jn] = float(val)
    joint_prims = {}
    for p in Usd.PrimRange(robot):
        jn = p.GetName()
        if jn in jd and (p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)):
            is_rev = p.IsA(UsdPhysics.RevoluteJoint)
            unit = 180.0 / math.pi if is_rev else 1.0
            drive = UsdPhysics.DriveAPI.Apply(p, "angular" if is_rev else "linear")
            drive.CreateTypeAttr().Set("force")
            drive.CreateStiffnessAttr().Set(1.0e7); drive.CreateDampingAttr().Set(1.0e5)
            drive.CreateTargetPositionAttr().Set(jd[jn] * unit)
            js = PhysxSchema.JointStateAPI.Apply(p, "angular" if is_rev else "linear")
            js.CreatePositionAttr().Set(jd[jn] * unit)
            joint_prims[jn] = (p, is_rev, js)

    # 두 컵: 덤프 world pose 를 env-local(원점 로봇 베이스)로 이동해 배치(회전·상대위치 보존).
    add_reference_to_stage(args.cup_usd, "/World/SourceCup")
    add_reference_to_stage(args.cup_usd, "/World/TargetCup")
    src_cup = stage.GetPrimAtPath("/World/SourceCup")
    tgt_cup = stage.GetPrimAtPath("/World/TargetCup")
    _set_pose(src_cup, d["cup_pos_w"][i] - origin, d["cup_quat_w"][i])
    _set_pose(tgt_cup, d["left_cup_pos_w"][i] - origin, d["left_cup_quat_w"][i])
    for c in (src_cup, tgt_cup):
        UsdPhysics.RigidBodyAPI.Apply(c).CreateKinematicEnabledAttr().Set(True)  # 고정(상태 읽기용)

    tl = omni.timeline.get_timeline_interface()
    tl.play()
    for _ in range(args.settle_steps):
        app.update()

    # ---- USD 읽기 ----
    # joint pos: JointStateAPI position (updateToUsd 로 physics 값 반영되는지 확인)
    def read_joints(names):
        out = []
        for jn in names:
            p, is_rev, js = joint_prims[jn]
            val = js.GetPositionAttr().Get()
            unit = math.pi / 180.0 if is_rev else 1.0
            out.append(float(val) * unit)
        return np.array(out)[None, :]

    arm_pos = read_joints(ARM); fin_pos = read_joints(HAND); larm_pos = read_joints(LARM)
    # 컵 pose USD 읽기
    scp, scq = _prim_quat_pos(src_cup); tcp, tcq = _prim_quat_pos(tgt_cup)

    # 재구성기 입력 구성 (1-env). vel=0(정적), origin 되더해 world 로.
    dd = dict(
        arm_joint_pos=arm_pos, arm_joint_vel=np.zeros((1, 7)),
        finger_joint_pos=fin_pos, left_arm_joint_pos=larm_pos, left_arm_joint_vel=np.zeros((1, 9)),
        cup_pos_w=(scp + origin)[None, :], cup_quat_w=scq[None, :],
        left_cup_pos_w=(tcp + origin)[None, :], left_cup_quat_w=tcq[None, :],
        hand_open_pose=d["hand_open_pose"], hand_grasp_pose=d["hand_grasp_pose"],
        source_cup_pour_point_pos_b=d["source_cup_pour_point_pos_b"],
        source_cup_pour_axis_b=d["source_cup_pour_axis_b"],
        source_cup_up_axis_b=d["source_cup_up_axis_b"],
        target_cup_opening_pos_b=d["target_cup_opening_pos_b"],
        target_cup_up_axis_b=d["target_cup_up_axis_b"],
        source_outer_radius=d["source_outer_radius"],
        pour_point_dyn_lo=d["pour_point_dyn_lo"], pour_point_dyn_hi=d["pour_point_dyn_hi"],
    )
    geo = R.compute_geometry(dd)
    rec = R.assemble_actor_obs(dd, geo)[0]
    ref = d["actor_obs"][i]

    # ---- 진단: 읽기 정확도 ----
    print(f"[P2-raw] env {i} | joint 읽기 오차: arm={np.abs(arm_pos[0]-d['arm_joint_pos'][i]).max():.3e} "
          f"hand={np.abs(fin_pos[0]-d['finger_joint_pos'][i]).max():.3e} "
          f"larm={np.abs(larm_pos[0]-d['left_arm_joint_pos'][i]).max():.3e}", flush=True)
    _he = np.abs(fin_pos[0] - d["finger_joint_pos"][i])
    _bad = np.argsort(_he)[::-1][:5]
    print("[P2-raw]   hand 관절별 오차 top5: " +
          ", ".join(f"{HAND[j]}={_he[j]:.3e}(set {jd[HAND[j]]:+.3f}→read {fin_pos[0][j]:+.3f})" for j in _bad),
          flush=True)
    print(f"[P2-raw] 컵 pose 읽기 오차: src_pos={np.abs(scp-(d['cup_pos_w'][i]-origin)).max():.3e} "
          f"src_quat={np.abs(scq-d['cup_quat_w'][i]).max():.3e} "
          f"tgt_pos={np.abs(tcp-(d['left_cup_pos_w'][i]-origin)).max():.3e}", flush=True)
    err = np.abs(rec - ref)
    seg = [("arm_pos",0,7),("arm_vel",7,14),("fgp",14,19),("larm_pos",19,28),
           ("larm_vel",28,37),("pp2open",37,40),("src_pour_ax",40,43),
           ("src_up_ax",43,46),("tgt_up_ax",46,49),("last_palm",49,55)]
    print(f"[P2-raw] --- actor obs 55D (raw-app USD 읽기 재구성 vs 덤프) ---", flush=True)
    for name, s, e in seg:
        print(f"[P2-raw]   {name:14s}[{s:2d}:{e:2d}] max|err|={err[s:e].max():.3e}", flush=True)
    # 판정 이원화: porting 로직(비-손가락) 은 exact 여야 하고, 손가락(fgp)은 raw-app 물리 평형
    #   차이(drive 모델≠isaaclab ImplicitActuator)로 obs_noise 이내면 허용.
    #   라이브 루프는 hand 를 grasp_hold 로 drive-freeze 하므로 이 차이는 무의미.
    fgp_err = err[14:19].max()
    other = np.concatenate([err[:14], err[19:]]).max()   # fgp 제외 전체
    TOL_LOGIC = 2e-3     # porting 로직 정확도
    TOL_FGP = 1.0e-2     # obs_noise_joint_pos(0.01rad) 이내 → 정책 견딤
    ok = (other < TOL_LOGIC) and (fgp_err < TOL_FGP)
    print(f"[P2-raw] ===== 판정: {'PASS' if ok else 'FAIL'} | "
          f"porting(비-fgp) max|err|={other:.3e}(<{TOL_LOGIC:.0e}) | "
          f"fgp max|err|={fgp_err:.3e}(<{TOL_FGP:.0e}, 물리평형·noise이내) =====", flush=True)
    app.close()


if __name__ == "__main__":
    main()
