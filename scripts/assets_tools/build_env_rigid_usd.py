#!/usr/bin/env python3
"""`env.usd` → `env_rigid.usd` — 정적 작업면에 **kinematic RigidBodyAPI** 를 붙인 사본.

왜 필요한가
-----------
`grasp_s2r` 의 작업면(`env.usd`)은 `/Env` Xform 아래 충돌 Mesh 8개뿐이고
RigidBodyAPI 가 **없다**. 그래서 `_setup_scene` 이 `spawn.func` 로 원시 프림을 박고,
`InteractiveScene` 은 그것을 자산으로 추적하지 못한다.

`replicate_physics=True` 에서는 `clone_environments` 의 `enable_env_ids` 가 env 간
충돌을 격리해 주므로 증상이 없다. 그러나 **다물체(MultiAsset)는 `replicate_physics=False`
가 필수**이고, 그때는 그 격리가 사라져 전 env 의 작업면이 한 충돌 그룹에 남는다.

08.29 분리 실측(1024 env · 22 iter · 단일 컵으로 고정하고 플래그만 뒤집음):

    replicate_physics=True   abnormal 0.0000 · joint_err 0.058 rad · palm_to_cup 0.104
    replicate_physics=False  abnormal 0.849  · joint_err 0.74  rad · palm_to_cup 0.52

팔이 41° 어긋난 채 고착된다. 자매 트랙 `tesollo/grasp_v2` 는 같은 조건에서 정상인데,
거기 테이블은 `RigidObject(table_cfg)` = **씬 자산**이다. 그것이 유일한 구조 차이다.

`UsdFileCfg.rigid_props` 로는 못 고친다 — 그 경로는 기존 API 를 **수정만** 하지
적용하지 않아 `RigidObject` 가 부팅에서 fail-loud 한다
(`Failed to find a rigid body when resolving '/World/envs/env_.*/Table'`).
그래서 USD 자체에 저작해야 한다.

무엇을 하는가
-------------
`/Env`(defaultPrim)에 `UsdPhysics.RigidBodyAPI` 를 적용하고 `kinematicEnabled=True`
로 둔다. kinematic 강체는 중력·외력을 무시하고 제자리에 있으므로 **정적 충돌체와
물리적으로 동등**하다. 자식 Mesh 의 `PhysicsCollisionAPI` 는 건드리지 않는다 —
그대로 이 강체의 충돌 형상이 된다.

★원본은 **읽기만** 한다. `env.usd` 는 자매 트랙도 쓰므로 절대 덮어쓰지 않는다.

사용
----
    python3 scripts/assets_tools/build_env_rigid_usd.py            # 기본 경로
    python3 scripts/assets_tools/build_env_rigid_usd.py --force    # 기존 산출물 덮어쓰기
    python3 scripts/assets_tools/build_env_rigid_usd.py --verify-only
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_HDGP_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_ENV_USD_DIR = os.path.join(_HDGP_ROOT, "assets", "env", "usd")

DEFAULT_SRC = os.path.join(_ENV_USD_DIR, "env.usd")
DEFAULT_DST = os.path.join(_ENV_USD_DIR, "env_rigid.usd")


def _fail(msg: str) -> None:
    print(f"[build_env_rigid] ✗ {msg}", flush=True)
    sys.exit(1)


def _collision_meshes(stage, root_path: str) -> list:
    """`root_path` 아래에서 PhysicsCollisionAPI 가 붙은 Mesh 경로."""
    from pxr import UsdPhysics

    out = []
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if not p.startswith(root_path + "/"):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            out.append(p)
    return sorted(out)


def verify(path: str) -> None:
    """산출물이 기대한 저작을 갖췄는지 확인 — 조용히 틀린 자산을 배포하지 않는다."""
    from pxr import Usd, UsdPhysics

    if not os.path.isfile(path):
        _fail(f"산출물이 없다: {path}")
    stage = Usd.Stage.Open(path)
    if stage is None:
        _fail(f"USD 를 열 수 없다: {path}")
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        _fail(f"defaultPrim 이 없다: {path}")

    if not root.HasAPI(UsdPhysics.RigidBodyAPI):
        _fail(f"{root.GetPath()} 에 RigidBodyAPI 가 없다 — RigidObject 가 못 붙는다")
    rb = UsdPhysics.RigidBodyAPI(root)
    kin = rb.GetKinematicEnabledAttr()
    if not kin or not bool(kin.Get()):
        _fail(f"{root.GetPath()} 이 kinematic 이 아니다 — 작업면이 중력에 떨어진다")

    meshes = _collision_meshes(stage, str(root.GetPath()))
    if not meshes:
        _fail(f"{root.GetPath()} 아래 충돌 메시가 하나도 없다 — 작업면이 통과된다")

    print(f"[build_env_rigid] ✓ 검증 통과: {path}")
    print(f"    defaultPrim = {root.GetPath()} (RigidBodyAPI · kinematic=True)")
    print(f"    충돌 메시 {len(meshes)}개: {[m.rsplit('/', 1)[1] for m in meshes]}")


def build(src: str, dst: str, force: bool) -> None:
    from pxr import Usd, UsdPhysics

    if not os.path.isfile(src):
        _fail(f"원본이 없다: {src}")
    if os.path.abspath(src) == os.path.abspath(dst):
        _fail("원본과 산출물 경로가 같다 — `env.usd` 는 자매 트랙도 쓰므로 덮어쓰지 않는다")
    if os.path.exists(dst) and not force:
        _fail(f"이미 있다(덮어쓰려면 --force): {dst}")

    stage = Usd.Stage.Open(src)
    if stage is None:
        _fail(f"USD 를 열 수 없다: {src}")
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        _fail(f"defaultPrim 이 없다: {src}")

    before = _collision_meshes(stage, str(root.GetPath()))
    if not before:
        _fail(f"{root.GetPath()} 아래 충돌 메시가 없다 — 잘못된 원본이다")
    if root.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"[build_env_rigid] ⚠ 원본에 이미 RigidBodyAPI 가 있다: {root.GetPath()}")

    # ★kinematic 강체 = 중력·외력 무시, 제자리 고정 → 정적 충돌체와 물리적으로 동등.
    #   자식 Mesh 의 PhysicsCollisionAPI 는 건드리지 않는다(그대로 충돌 형상이 된다).
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    rb.CreateRigidBodyEnabledAttr(True)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not stage.GetRootLayer().Export(dst):
        _fail(f"내보내기 실패: {dst}")

    after = _collision_meshes(Usd.Stage.Open(dst), str(root.GetPath()))
    if after != before:
        _fail(f"충돌 메시가 바뀌었다: {before} → {after}")

    print(f"[build_env_rigid] 원본 {src}")
    print(f"[build_env_rigid] 산출 {dst}")
    verify(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--force", action="store_true", help="기존 산출물 덮어쓰기")
    ap.add_argument("--verify-only", action="store_true", help="빌드 없이 산출물만 검증")
    args = ap.parse_args()

    try:
        import pxr  # noqa: F401
    except ImportError:
        _fail("pxr 를 import 할 수 없다 — Isaac 파이썬으로 실행할 것 "
              "(예: ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/assets_tools/build_env_rigid_usd.py)")

    if args.verify_only:
        verify(args.dst)
        return
    build(args.src, args.dst, args.force)


if __name__ == "__main__":
    main()
