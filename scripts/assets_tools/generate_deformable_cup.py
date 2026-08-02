#!/usr/bin/env python3
"""Segmented-shell deformable cup USD 생성 (grasp_adapt Phase 2, Gate A).

원리
----
FEM 소프트바디는 2048-env 실시간 학습에 비현실적이라, "변형"을 **articulation 근사**로 만든다.
- **base**: rigid 바닥 디스크 = articulation root(자유부유). bead가 얹히는 컵 바닥.
- **N 패널**: 원통 벽을 각도분할한 얇은 rigid 세그먼트. 각 패널은 base 링 하단 모서리에
  **접선(tangential) 수평 힌지(revolute)**로 연결된다.
    * 힌지각 0 = 벽이 수직(온전).
    * 손가락이 안쪽으로 누르면 패널이 안쪽으로 기울어짐 = 힌지각 발생 = **변형(deformation)**.
    * revolute drive(target 0, stiffness K, damping D) = **복원 스프링**.
    * 각이 buckle 임계 초과 = 좌굴(env가 실패로 판정).

주의: 계획서 초안의 "세로 hinge"는 tangential swing이라 반경 denting을 못 만든다.
      물리적으로 올바른 것은 **접선 수평축 힌지(바닥 hinged flap)** — 이 스크립트가 채택.

출력
----
assets/cup/deformable_cup.usd (ArticulationRootAPI, 원본 cup_big_sdf.usd 치수 정합).

실행 (pxr 독립 — Isaac 런타임 불필요, 저작만)
-----------------------------------------------
  cd /home/user/rl_ws/hdgp
  python3 scripts/assets_tools/generate_deformable_cup.py \
      --panels 12 --stiffness 0.5 --damping 0.05 --limit-deg 45

Gate A 물리 검증(패널 눌림/복원/리셋)은 별도 probe(Isaac 런타임)에서 수행한다.
"""
from __future__ import annotations

import argparse
import math
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# 원본 cup_big_sdf.usd 치수(미터, metersPerUnit=1.0): 반경 ~0.045, z[-0.077, 0.100].
DEFAULT_RADIUS = 0.045
DEFAULT_Z_BOTTOM = -0.077
DEFAULT_Z_TOP = 0.100
DEFAULT_BASE_HEIGHT = 0.030          # 바닥 rigid 디스크 두께
DEFAULT_WALL_THICKNESS = 0.0025      # 패널 반경방향 두께(종이컵 벽)

# 질량(kg): 실제 종이컵 ~0.010kg. 물(=bead)은 별도 하중.
DEFAULT_BASE_MASS = 0.006
DEFAULT_PANEL_MASS = 0.0004          # 패널당 (×12 = 0.0048)


def _make_body_xform(stage: Usd.Stage, path: str, translate: Gf.Vec3d,
                     rot_z_deg: float) -> UsdGeom.Xform:
    """RigidBody가 붙을 body Xform (translate + Z회전만; scale 금지)."""
    xf = UsdGeom.Xform.Define(stage, path)
    api = UsdGeom.XformCommonAPI(xf)
    api.SetTranslate(translate)
    if rot_z_deg != 0.0:
        api.SetRotate(Gf.Vec3f(0.0, 0.0, rot_z_deg))
    return xf


def _apply_rigid_body(prim: Usd.Prim, mass: float) -> None:
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)


def _make_box_collider(stage: Usd.Stage, path: str, size: Gf.Vec3f,
                       translate: Gf.Vec3d) -> None:
    """body 자식으로 box gprim + CollisionAPI (visual+collision 겸용)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)  # 단위 큐브(±0.5) → scale로 실치수
    api = UsdGeom.XformCommonAPI(cube)
    api.SetTranslate(translate)
    api.SetScale(Gf.Vec3f(size[0], size[1], size[2]))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _make_cylinder_collider(stage: Usd.Stage, path: str, radius: float,
                            height: float, translate: Gf.Vec3d) -> None:
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(UsdGeom.Tokens.z)
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2.0),
                          Gf.Vec3f(radius, radius, height / 2.0)])
    UsdGeom.XformCommonAPI(cyl).SetTranslate(translate)
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())


def _make_revolute_spring_joint(
    stage: Usd.Stage, path: str, base_path: str, panel_path: str,
    hinge_world: Gf.Vec3f, rot_z_deg: float,
    stiffness: float, damping: float, limit_deg: float,
) -> None:
    """base↔panel 접선축 revolute + 복원 스프링 drive.

    joint frame X축 = 접선(=패널 안쪽 눌림축). base frame은 무회전(월드 정렬),
    panel frame은 Rz(θ) 회전됨:
      - localRot0(base) = Rz(θ) → base 기준 joint X = 접선.
      - localRot1(panel) = identity → panel-local X = 접선(이미 Rz(θ) 적용).
      - localPos0 = hinge(월드), localPos1 = (0,0,0)(panel 원점=hinge).
    """
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([base_path])
    joint.CreateBody1Rel().SetTargets([panel_path])
    joint.CreateAxisAttr("X")

    theta = math.radians(rot_z_deg)
    quat_z = Gf.Quatf(
        math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)
    )  # Rz(θ)
    joint.CreateLocalPos0Attr(hinge_world)
    joint.CreateLocalRot0Attr(quat_z)
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # 대칭 한계(안/밖). 변형은 |angle|로 측정.
    joint.CreateLowerLimitAttr(-limit_deg)
    joint.CreateUpperLimitAttr(limit_deg)

    # 복원 스프링: position drive target 0.
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    drive.CreateMaxForceAttr(1.0e6)


def generate(out_path: str, *, panels: int, radius: float, z_bottom: float,
             z_top: float, base_height: float, wall_thickness: float,
             base_mass: float, panel_mass: float,
             stiffness: float, damping: float, limit_deg: float) -> None:
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_path = "/deformable_cup"
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    base_top = z_bottom + base_height
    wall_h = z_top - base_top
    if wall_h <= 0:
        raise ValueError(f"wall height <= 0 (base_top={base_top}, z_top={z_top})")

    # --- base(바닥 디스크) = articulation root body -----------------------
    base_path = f"{root_path}/base"
    _make_body_xform(stage, base_path, Gf.Vec3d(0.0, 0.0, 0.0), 0.0)
    _apply_rigid_body(stage.GetPrimAtPath(base_path), base_mass)
    _make_cylinder_collider(
        stage, f"{base_path}/base_geo", radius=radius, height=base_height,
        translate=Gf.Vec3d(0.0, 0.0, z_bottom + base_height / 2.0),
    )

    # --- N 패널(벽 세그먼트) + 접선 힌지 스프링 ---------------------------
    panel_w = 2.0 * radius * math.sin(math.pi / panels)  # 접선 폭(현)
    joints_scope = UsdGeom.Scope.Define(stage, f"{root_path}/joints")
    _ = joints_scope

    for i in range(panels):
        theta_deg = 360.0 * i / panels
        theta = math.radians(theta_deg)
        hinge = Gf.Vec3f(radius * math.cos(theta), radius * math.sin(theta),
                         base_top)
        panel_path = f"{root_path}/panel_{i:02d}"
        # 패널 body: 원점=hinge, Rz(θ) 회전(local X=접선, Y=반경밖, Z=위).
        _make_body_xform(stage, panel_path,
                         Gf.Vec3d(float(hinge[0]), float(hinge[1]), float(hinge[2])),
                         theta_deg)
        _apply_rigid_body(stage.GetPrimAtPath(panel_path), panel_mass)
        # 패널 geo: 안쪽(-Y)으로 두께 절반, 위로 wall_h/2.
        _make_box_collider(
            stage, f"{panel_path}/panel_geo",
            size=Gf.Vec3f(panel_w, wall_thickness, wall_h),
            translate=Gf.Vec3d(0.0, -wall_thickness / 2.0, wall_h / 2.0),
        )
        _make_revolute_spring_joint(
            stage, f"{root_path}/joints/revolute_{i:02d}",
            base_path, panel_path, hinge, theta_deg,
            stiffness=stiffness, damping=damping, limit_deg=limit_deg,
        )

    stage.GetRootLayer().Save()

    total_mass = base_mass + panels * panel_mass
    print(f"[generate_deformable_cup] 저장: {out_path}")
    print(f"  panels={panels} radius={radius} z[{z_bottom},{z_top}] "
          f"base_h={base_height} wall_h={wall_h:.4f} panel_w={panel_w:.4f}")
    print(f"  mass: base={base_mass} panel×{panels}={panels*panel_mass:.4f} "
          f"total={total_mass:.4f}kg")
    print(f"  spring: stiffness={stiffness} damping={damping} "
          f"limit=±{limit_deg}deg")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "assets", "cup", "deformable_cup.usd",
    )
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--panels", type=int, default=12)
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    ap.add_argument("--z-bottom", type=float, default=DEFAULT_Z_BOTTOM)
    ap.add_argument("--z-top", type=float, default=DEFAULT_Z_TOP)
    ap.add_argument("--base-height", type=float, default=DEFAULT_BASE_HEIGHT)
    ap.add_argument("--wall-thickness", type=float, default=DEFAULT_WALL_THICKNESS)
    ap.add_argument("--base-mass", type=float, default=DEFAULT_BASE_MASS)
    ap.add_argument("--panel-mass", type=float, default=DEFAULT_PANEL_MASS)
    ap.add_argument("--stiffness", type=float, default=0.5,
                    help="revolute drive stiffness(복원). 커리큘럼 초기엔 크게(rigid-like).")
    ap.add_argument("--damping", type=float, default=0.05)
    ap.add_argument("--limit-deg", type=float, default=45.0)
    args = ap.parse_args()

    if args.panels < 3:
        ap.error("--panels >= 3")
    generate(
        args.out, panels=args.panels, radius=args.radius,
        z_bottom=args.z_bottom, z_top=args.z_top, base_height=args.base_height,
        wall_thickness=args.wall_thickness, base_mass=args.base_mass,
        panel_mass=args.panel_mass, stiffness=args.stiffness,
        damping=args.damping, limit_deg=args.limit_deg,
    )


if __name__ == "__main__":
    main()
