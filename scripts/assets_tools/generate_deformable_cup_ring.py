"""Edge-hinged 닫힌 12각형 링 컵 (진짜 cohesive 변형 프로토타입).

기존 generate_deformable_cup.py는 패널이 각자 바닥에만 힌지 → 독립적으로 움직여
"따로 노는 판"이 됨. 이 버전은 **인접 패널을 공유 수직 모서리에서 revolute(Z축)+스프링으로
서로 연결** → 닫힌 링. 한쪽을 누르면 모서리 힌지로 전파돼 단면이 타원으로 눌리며
**형상 전체가 cohesive하게 찌그러졌다 복원**한다(실제 종이컵 ovalization).

구조(articulation): base(root, rigid 디스크) —fixed— panel_0 —edge(Z)— panel_1 — ... —
panel_11, 그리고 panel_11 —edge(Z)— panel_0 로 **루프 폐쇄**(PhysX가 loop joint=constraint로 처리).
edge 힌지 = 수직축 → 12각형 내각 변화 = ovalization. 스프링이 형상 복원.

⚠️ 루프+12스프링이라 안정성(NaN/발산) 미검증 → probe로 먼저 확인하는 프로토타입.

실행(pxr 독립):
  cd /home/user/rl_ws/hdgp
  python3 scripts/assets_tools/generate_deformable_cup_ring.py
"""
from __future__ import annotations

import argparse
import math
import os

from pxr import Gf, Usd, UsdGeom, UsdPhysics

DEFAULT_RADIUS = 0.045
DEFAULT_Z_BOTTOM = -0.077
DEFAULT_Z_TOP = 0.100
DEFAULT_BASE_HEIGHT = 0.030
DEFAULT_WALL_THICKNESS = 0.0025
DEFAULT_BASE_MASS = 0.006
DEFAULT_PANEL_MASS = 0.0006


def _rot_z_quat(deg: float) -> Gf.Quatf:
    r = math.radians(deg)
    return Gf.Quatf(math.cos(r / 2.0), 0.0, 0.0, math.sin(r / 2.0))


def _body_xform(stage, path, translate, rot_z_deg):
    xf = UsdGeom.Xform.Define(stage, path)
    api = UsdGeom.XformCommonAPI(xf)
    api.SetTranslate(translate)
    if rot_z_deg != 0.0:
        api.SetRotate(Gf.Vec3f(0.0, 0.0, rot_z_deg))
    return xf


def _rigid(prim, mass):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)


def _box(stage, path, size, translate):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    api = UsdGeom.XformCommonAPI(cube)
    api.SetTranslate(translate)
    api.SetScale(Gf.Vec3f(size[0], size[1], size[2]))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _cyl(stage, path, radius, height, translate):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr(UsdGeom.Tokens.z)
    c.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2.0),
                        Gf.Vec3f(radius, radius, height / 2.0)])
    UsdGeom.XformCommonAPI(c).SetTranslate(translate)
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())


def _fixed_joint(stage, path, body0, body1, lpos0, lrot0):
    j = UsdPhysics.FixedJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([body0])
    j.CreateBody1Rel().SetTargets([body1])
    j.CreateLocalPos0Attr(lpos0)
    j.CreateLocalRot0Attr(lrot0)
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def _edge_joint(stage, path, body0, body1, lpos0, lpos1,
                stiffness, damping, limit_deg):
    """수직(Z) 모서리 힌지 + 복원 스프링. 두 패널의 Z축 = 월드 Z(동일) → localRot 단위."""
    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([body0])
    j.CreateBody1Rel().SetTargets([body1])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(lpos0)
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(lpos1)
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-limit_deg)
    j.CreateUpperLimitAttr(limit_deg)
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    d.CreateTypeAttr("force")
    d.CreateTargetPositionAttr(0.0)
    d.CreateStiffnessAttr(stiffness)
    d.CreateDampingAttr(damping)
    d.CreateMaxForceAttr(1.0e6)


def generate(out_path, *, panels, radius, z_bottom, z_top, base_height,
             wall_thickness, base_mass, panel_mass, stiffness, damping, limit_deg):
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_path = "/deformable_cup_ring"
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    # ★ ArticulationRootAPI 없음: 닫힌 링(루프)은 articulation(트리) 미지원 →
    #   maximal-coordinate 강체+조인트로 루프 시뮬. base는 정적(static) 앵커.

    base_top = z_bottom + base_height
    wall_h = z_top - base_top
    half_w = radius * math.tan(math.pi / panels)   # 패널 반폭(외접, 모서리 맞닿음)
    panel_w = 2.0 * half_w

    # base = 정적 콜라이더(RigidBody 없음) — 링을 세계에 앵커. 패널만 동적.
    base_path = f"{root_path}/base"
    _body_xform(stage, base_path, Gf.Vec3d(0.0, 0.0, 0.0), 0.0)
    _cyl(stage, f"{base_path}/base_geo", radius=radius, height=base_height,
         translate=Gf.Vec3d(0.0, 0.0, z_bottom + base_height / 2.0))

    # 패널 12개 (벽 세그먼트, Rz(θ+90): local X=접선폭, Y=방사두께, Z=위)
    UsdGeom.Scope.Define(stage, f"{root_path}/joints")
    for i in range(panels):
        theta = 360.0 * i / panels
        th = math.radians(theta)
        origin = Gf.Vec3d(radius * math.cos(th), radius * math.sin(th), base_top)
        p_path = f"{root_path}/panel_{i:02d}"
        _body_xform(stage, p_path, origin, theta + 90.0)
        _rigid(stage.GetPrimAtPath(p_path), panel_mass)
        _box(stage, f"{p_path}/panel_geo",
             size=Gf.Vec3f(panel_w, wall_thickness, wall_h),
             translate=Gf.Vec3d(0.0, 0.0, wall_h / 2.0))

    # base —fixed— panel_0 (링을 base에 고정, 한 점 앵커)
    _fixed_joint(stage, f"{root_path}/joints/base_fix",
                 base_path, f"{root_path}/panel_00",
                 Gf.Vec3f(float(radius), 0.0, float(base_top)), _rot_z_quat(90.0))

    # 인접 패널 모서리 힌지(수직 Z) — 닫힌 링(panel_11 → panel_0 루프 폐쇄)
    zc = wall_h / 2.0
    for i in range(panels):
        j = (i + 1) % panels
        _edge_joint(
            stage, f"{root_path}/joints/edge_{i:02d}",
            f"{root_path}/panel_{i:02d}", f"{root_path}/panel_{j:02d}",
            lpos0=Gf.Vec3f(float(half_w), 0.0, float(zc)),    # panel_i 우측 모서리
            lpos1=Gf.Vec3f(float(-half_w), 0.0, float(zc)),   # panel_j 좌측 모서리
            stiffness=stiffness, damping=damping, limit_deg=limit_deg,
        )

    stage.GetRootLayer().Save()
    print(f"[ring] saved {out_path}")
    print(f"  panels={panels} radius={radius} wall_h={wall_h:.4f} panel_w={panel_w:.4f}")
    print(f"  edge springs: stiffness={stiffness} damping={damping} limit=±{limit_deg}deg")
    print(f"  panel mass total={panels * panel_mass:.4f}kg ({panels}panel), base=static anchor")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "assets", "cup", "deformable_cup_ring.usd")
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--panels", type=int, default=12)
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    ap.add_argument("--z-bottom", type=float, default=DEFAULT_Z_BOTTOM)
    ap.add_argument("--z-top", type=float, default=DEFAULT_Z_TOP)
    ap.add_argument("--base-height", type=float, default=DEFAULT_BASE_HEIGHT)
    ap.add_argument("--wall-thickness", type=float, default=DEFAULT_WALL_THICKNESS)
    ap.add_argument("--base-mass", type=float, default=DEFAULT_BASE_MASS)
    ap.add_argument("--panel-mass", type=float, default=DEFAULT_PANEL_MASS)
    ap.add_argument("--stiffness", type=float, default=0.3)
    ap.add_argument("--damping", type=float, default=0.05)
    ap.add_argument("--limit-deg", type=float, default=40.0)
    args = ap.parse_args()
    if args.panels < 3:
        ap.error("--panels >= 3")
    generate(args.out, panels=args.panels, radius=args.radius, z_bottom=args.z_bottom,
             z_top=args.z_top, base_height=args.base_height,
             wall_thickness=args.wall_thickness, base_mass=args.base_mass,
             panel_mass=args.panel_mass, stiffness=args.stiffness,
             damping=args.damping, limit_deg=args.limit_deg)


if __name__ == "__main__":
    main()
