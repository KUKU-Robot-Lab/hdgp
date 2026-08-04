"""grasp_v1 sim2real 평가 SP2 — Task 6: USD→obj 메쉬 추출 도구(standalone).

FoundationPose는 CAD 모델(.obj)이 필요하다 — Isaac USD 물체 자산을 그대로 쓸 수 없어
render_pass.py가 만드는 object_map.json(Pass 1 산출물, 스키마 {"<obj_idx>": {"id",
"usd_path", "scale"}})을 읽어 각 물체를 독립 .obj로 추출한다(fp_batch.py의
mesh_path_for()가 `<mesh_dir>/<id>.obj`로 찾으므로 파일명은 반드시 id 기준).

USD Mesh prim의 points/faceVertexIndices/faceVertexCounts를 모아
1) 팬 삼각화(quad/ngon 대응, USD 메쉬는 삼각형만 있다고 가정할 수 없음)
2) prim의 accumulated world transform(ComputeLocalToWorldTransform) 적용
   (멀티메쉬 자산이 하나의 좌표계로 정합 병합되도록)
3) metersPerUnit 반영(스테이지 고유 단위 → 미터)
4) object_map의 spec scale(비균일 포함) 베이크(Isaac 쪽 UsdFileCfg(scale=...)와 동일 연산 —
   grasp_right_env_cfg.py:83 `_object_usd_cfg`가 spec["scale"]을 그대로 UsdFileCfg에 넘기는
   것과 대응)
를 거쳐 하나의 obj(v/f 라인만, 법선/UV 없음)로 병합 저장한다. trimesh 등 외부 의존성 없음.

같은 USD가 다른 스케일로 여러 번 나타나는 경우(cup_big×4, grasp_right_env_cfg.py:66-69)도
object_map의 각 entry(=obj_idx)마다 독립적으로 열고 스케일을 베이크하므로 별도 obj로 저장된다.

**실행 환경**: pxr는 Isaac Python 환경(`isaaclab.sh -p`)에서만 import 가능하다 — 이 저장소
안에서 직접 실행할 수 없다(게이트, 검증은 문법 파싱까지만). 상단 import는 json/argparse/
pathlib/numpy뿐이라 이 repo 환경에서도 `ast.parse`로 문법 검증은 가능하다. pxr import는
main() 안에서 지연 수행한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def fan_triangulate(indices):
    """면 정점 인덱스(다각형 순서)를 팬(fan) 방식으로 삼각화.

    v0을 고정 앵커로 (v0, vi, vi+1) 삼각형들을 만든다. USD 메쉬는 faceVertexCounts에
    3(삼각형)뿐 아니라 4(사각형) 이상도 흔하므로 필수(quad export가 드물지 않음).
    len(indices) < 3이면 퇴화면이라 빈 리스트를 반환한다(스킵, 크래시하지 않음).
    """
    if len(indices) < 3:
        return []
    return [(indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)]


def bbox_extents(points):
    """정점 리스트(N,3)의 축정렬 bbox (min, max, extents) — numpy만 사용, pxr 무관.

    points가 비어 있으면 전부 0으로 채운 값을 반환한다(호출자가 이후 예외를 던질 수 있게
    여기서는 조용히 실패하지 않는다 — 순수 계산 함수라 입력 검증은 호출자 책임).
    """
    if not points:
        zero = (0.0, 0.0, 0.0)
        return zero, zero, zero
    arr = np.asarray(points, dtype=float)
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    return tuple(mn.tolist()), tuple(mx.tolist()), tuple((mx - mn).tolist())


def write_obj(path, points, faces) -> None:
    """v/f 라인만 기록하는 최소 obj writer(법선/UV 없음, trimesh 미사용). f는 1-based."""
    with open(path, "w") as f:
        f.write("# exported by scripts/eval_s2r/export_meshes.py (grasp_v1 s2r eval SP2 Task 6)\n")
        for x, y, z in points:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for i, j, k in faces:
            f.write(f"f {i + 1} {j + 1} {k + 1}\n")


def _extract_mesh_prim(prim, Usd, UsdGeom, Gf):
    """단일 UsdGeom.Mesh prim에서 (world-space points, 0-based 삼각형 face) 추출.

    ComputeLocalToWorldTransform은 이 usd 파일 *자체* 스테이지(defaultPrim 루트) 기준
    world다 — 멀티메쉬 자산(여러 mesh prim이 각자 다른 위치/회전을 가짐)을 하나의
    좌표계로 병합하려면 이 변환이 반드시 필요하다(안 하면 병합 시 메쉬 조각들이
    엉뚱한 상대 위치에 놓인다).
    """
    mesh = UsdGeom.Mesh(prim)
    points_attr = mesh.GetPointsAttr().Get()
    fvc_attr = mesh.GetFaceVertexCountsAttr().Get()
    fvi_attr = mesh.GetFaceVertexIndicesAttr().Get()
    if not points_attr or not fvc_attr or not fvi_attr:
        return [], []

    xform_matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_points = [
        tuple(xform_matrix.Transform(Gf.Vec3d(p[0], p[1], p[2]))) for p in points_attr
    ]

    faces = []
    cursor = 0
    for count in fvc_attr:
        face_indices = list(fvi_attr[cursor:cursor + count])
        cursor += count
        faces.extend(fan_triangulate(face_indices))
    return world_points, faces


def _export_object(spec, Usd, UsdGeom, Gf):
    """object_map.json entry 하나(usd_path+scale)를 (병합 points, 병합 faces)로 변환.

    stage 안의 모든 UsdGeom.Mesh prim을 순회·병합한 뒤 metersPerUnit → spec scale 순서로
    베이크한다(순서 근거: world 변환은 스테이지 고유 단위계에서 계산되므로, 그 뒤에 곱하는
    균일/비균일 스칼라는 순서를 바꿔도 최종 좌표값엔 영향이 없다 — 둘 다 단순 축별 스칼라
    곱이라 교환 가능. metersPerUnit을 world 변환 *이전*에 곱하면 안 되는 이유는 없지만,
    이후에 곱하는 편이 "world 좌표 → 미터 → spec 스케일" 순으로 읽기 쉽다).
    """
    usd_path = spec["usd_path"]
    scale = spec["scale"]
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"USD 열기 실패: {usd_path}")
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)

    all_points = []
    all_faces = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        pts, faces = _extract_mesh_prim(prim, Usd, UsdGeom, Gf)
        if not pts:
            continue
        offset = len(all_points)
        all_points.extend(pts)
        all_faces.extend((i + offset, j + offset, k + offset) for i, j, k in faces)

    if not all_points:
        raise RuntimeError(f"Mesh prim을 찾지 못함(빈 자산?): {usd_path}")

    sx, sy, sz = scale
    baked = [
        (x * meters_per_unit * sx, y * meters_per_unit * sy, z * meters_per_unit * sz)
        for x, y, z in all_points
    ]
    return baked, all_faces


def _try_import_pxr(use_kit: bool):
    """pxr import(지연) — 실패 시 --kit로 AppLauncher(headless) 부트스트랩 후 재시도 안내.

    plain import를 기본으로 먼저 시도한다: Kit 앱 생성은 수 초~수십 초가 걸리는데, 여러
    probe 스크립트(scripts/tools/probe_cup_assets.py 등)가 SimulationApp을 항상 띄우는 건
    "USD stage를 읽으려면 Kit이 반드시 필요해서"가 아니라 보수적 습관에 가깝다 — pxr는
    순수 USD 파싱 라이브러리라 Isaac python env(`isaaclab.sh -p`)에 sys.path가 잡혀 있으면
    Kit 없이도 import되는 경우가 많다. 그래도 일부 환경(Kit 플러그인 초기화가 import 시점에
    필요한 경우)은 plain import가 실패할 수 있어 --kit 폴백을 제공한다.

    반환: (Usd, UsdGeom, Gf, app_or_None) — app_or_None은 --kit 경로에서 생성된
    simulation app 인스턴스(호출자가 종료 시 close()해야 함), plain 경로면 None.
    """
    app = None
    if use_kit:
        try:
            from isaaclab.app import AppLauncher
        except ImportError as e:
            print(
                f"[ERROR] --kit 지정했으나 isaaclab.app.AppLauncher import 실패: {e}\n"
                f"        isaaclab.sh -p 로 실행 중인지 확인하라.",
                file=sys.stderr,
            )
            sys.exit(1)
        app = AppLauncher(headless=True).app

    try:
        from pxr import Usd, UsdGeom, Gf
        return Usd, UsdGeom, Gf, app
    except ImportError as e:
        print(
            "[ERROR] pxr import 실패 — 이 스크립트는 Isaac Python 환경에서만 동작한다.\n"
            "        `isaaclab.sh -p scripts/eval_s2r/export_meshes.py --object_map ... --out ...`\n"
            "        로 실행하라. (plain import가 안 되는 환경이면 --kit 플래그를 추가해 "
            "AppLauncher를 먼저 띄운 뒤 재시도할 수 있다.)\n"
            f"        원인: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="grasp_v1 s2r 평가 SP2 — Task 6: object_map.json의 USD 물체들을 .obj로 추출"
    )
    parser.add_argument("--object_map", required=True, help="render_pass.py가 만든 object_map.json 경로")
    parser.add_argument("--out", required=True, help="obj 출력 디렉토리(<id>.obj로 저장)")
    parser.add_argument(
        "--kit", action="store_true",
        help="plain pxr import가 실패하는 환경에서 AppLauncher(headless=True)를 먼저 띄운 뒤 pxr를 import",
    )
    args = parser.parse_args()

    Usd, UsdGeom, Gf, app = _try_import_pxr(args.kit)

    with open(args.object_map) as f:
        object_map = json.load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in sorted(object_map.keys(), key=int):
        spec = object_map[key]
        points, faces = _export_object(spec, Usd, UsdGeom, Gf)
        obj_path = out_dir / f"{spec['id']}.obj"
        write_obj(obj_path, points, faces)
        _mn, _mx, extents = bbox_extents(points)
        # 단위 검증용 sanity note: cup_big_s100(scale 1.0)은 실물 컵 지름 ~7cm 근방이어야
        # 한다 — bbox_extents_m의 x/y가 이와 크게 어긋나면 metersPerUnit·scale 베이크
        # 순서/값이 잘못됐다는 신호다(§요구사항 3).
        print(
            f"[obj_idx={key}] id={spec['id']} -> {obj_path} "
            f"verts={len(points)} faces={len(faces)} "
            f"bbox_extents_m=({extents[0]:.4f},{extents[1]:.4f},{extents[2]:.4f})"
        )

    if app is not None:
        app.close()


if __name__ == "__main__":
    main()
