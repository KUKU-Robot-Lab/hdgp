#!/usr/bin/env python3
"""컵 USD의 collision 방식을 Convex Decomposition → SDF로 변환.

Isaac Sim Python 환경에서 실행:
  cd /home/user/rl_ws/IsaacLab
  ./isaaclab.sh -p ../hdgp/scripts/tools/convert_cup_collision_to_sdf.py

결과: assets/cup/cup_big_sdf.usd (원본 유지, 새 파일 생성)
"""

import argparse
import os
import shutil

# -------------------------------------------------------------------
# USD / PhysX schema import (Isaac Sim Python 환경 필요)
# -------------------------------------------------------------------
from pxr import Usd, UsdPhysics, PhysxSchema


def convert_mesh_to_sdf(
    stage: Usd.Stage,
    sdf_resolution: int = 256,
    sdf_subgrid_resolution: int = 6,
    sdf_margin: float = 0.005,
    sdf_narrow_band_thickness: float = 0.01,
) -> int:
    """stage 내 모든 Mesh prim의 collision을 SDF로 변환.

    Returns:
        변환된 prim 수
    """
    count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.CollisionAPI):
            # CollisionAPI가 없는 prim은 건너뜀
            continue

        mesh_api = UsdPhysics.MeshCollisionAPI.Get(stage, prim.GetPath())
        if not mesh_api:
            # MeshCollisionAPI가 없으면 Apply
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)

        # approximation을 sdf로 설정
        mesh_api.GetApproximationAttr().Set("sdf")

        # 기존 ConvexDecompositionCollisionAPI 제거 (있으면)
        if prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
            prim.RemoveAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI)

        # SDFMeshCollisionAPI 적용 및 파라미터 설정
        if not prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
            sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
        else:
            sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Get(stage, prim.GetPath())

        sdf_api.GetSdfResolutionAttr().Set(sdf_resolution)
        sdf_api.GetSdfSubgridResolutionAttr().Set(sdf_subgrid_resolution)
        sdf_api.GetSdfMarginAttr().Set(sdf_margin)
        sdf_api.GetSdfNarrowBandThicknessAttr().Set(sdf_narrow_band_thickness)

        print(f"  SDF 적용: {prim.GetPath()}")
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="컵 USD collision → SDF 변환")
    parser.add_argument(
        "--input", default="/home/user/rl_ws/hdgp/assets/cup/cup_big.usd",
        help="입력 USD 경로",
    )
    parser.add_argument(
        "--output", default="/home/user/rl_ws/hdgp/assets/cup/cup_big_sdf.usd",
        help="출력 USD 경로 (원본 보존)",
    )
    parser.add_argument("--sdf-resolution", type=int, default=256, help="SDF 해상도 (default=256)")
    parser.add_argument("--sdf-subgrid", type=int, default=6, help="SDF 서브그리드 해상도 (default=6)")
    args = parser.parse_args()

    # 원본 복사
    print(f"복사: {args.input} → {args.output}")
    shutil.copy2(args.input, args.output)

    # stage 열기 및 변환
    stage = Usd.Stage.Open(args.output)
    if not stage:
        raise RuntimeError(f"USD 열기 실패: {args.output}")

    print(f"SDF 변환 중 (resolution={args.sdf_resolution}, subgrid={args.sdf_subgrid})...")
    count = convert_mesh_to_sdf(
        stage,
        sdf_resolution=args.sdf_resolution,
        sdf_subgrid_resolution=args.sdf_subgrid,
    )

    stage.Save()
    print(f"완료: {count}개 prim 변환 → {args.output}")


if __name__ == "__main__":
    main()
