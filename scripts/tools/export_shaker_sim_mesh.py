#!/usr/bin/env python3
"""sim shaker 자산의 메시를 FP++ 용 vertex-color PLY 로 내보낸다 (원점·단위 = sim 그대로).

배경(2026-09-03 분석)
--------------------
FP++ 에 넣던 CAD(`shaker_full.obj`, 뚜껑+캡 포함 238mm, 원점=바닥)와 학습 자산
(`assets/cup/shaker_closed_rl.usd`, 몸통만 175mm, 원점=바닥+92.1mm)이 다른 물체였다.
cup_big 은 CAD≡sim 메시라 relay 의 cad_to_body 가 회전만이면 됐지만, shaker 는
원점·높이·형상이 전부 달라 z 편향·x 떨림·상하 반전이 났다.

이 스크립트는 sim 콜라이더 메시를 **좌표 변환 없이** PLY 로 쓴다. 그러면 FP++ 출력
프레임 == sim body 프레임이 되어 relay 의 cad_to_body 는 항등이면 된다.

FP++ 는 텍스처 없는 OBJ 를 죽이므로(`make_mesh_tensors` 가 material.image 요구)
vertex color 를 넣는다. 색은 실물(파란 무광 shaker)에 맞춰 인자로 준다.

실행:
  python3 scripts/tools/export_shaker_sim_mesh.py --out /tmp/shaker_sim.ply --rgb 30 50 120
  (pxr + numpy 만 사용. Isaac 불필요)
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from pxr import Usd

_HDGP_ROOT = Path(__file__).resolve().parents[2]
_SRC = _HDGP_ROOT / "assets" / "cup" / "shaker_closed_rl.usd"
_MESH_PRIM = "/object_shaker_body/baseLink/collisions"


def _triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """USD faceVertexCounts/Indices → (M,3) 삼각형. 팬 분할(볼록 면 가정)."""
    tris, cursor = [], 0
    for n in counts:
        face = indices[cursor:cursor + n]
        cursor += n
        for k in range(1, n - 1):
            tris.append((face[0], face[k], face[k + 1]))
    return np.asarray(tris, dtype=np.int32)


def load_sim_mesh(src: Path = _SRC) -> tuple[np.ndarray, np.ndarray]:
    stage = Usd.Stage.Open(str(src))
    prim = stage.GetPrimAtPath(_MESH_PRIM)
    if not prim.IsValid():
        raise RuntimeError(f"prim 없음: {_MESH_PRIM} in {src}")
    verts = np.asarray(prim.GetAttribute("points").Get(), dtype=np.float32)
    counts = np.asarray(prim.GetAttribute("faceVertexCounts").Get(), dtype=np.int32)
    indices = np.asarray(prim.GetAttribute("faceVertexIndices").Get(), dtype=np.int32)
    return verts, _triangulate(counts, indices)


def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray, rgb: tuple[int, int, int]) -> None:
    header = "\n".join([
        "ply", "format binary_little_endian 1.0",
        "comment sim shaker_closed_rl.usd collisions mesh, origin = sim body frame, meters",
        f"element vertex {len(verts)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header", ""])
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        color = bytes(rgb)
        for v in verts:
            fh.write(struct.pack("<fff", *v) + color)
        for f in faces:
            fh.write(struct.pack("<Biii", 3, *f))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rgb", type=int, nargs=3, default=(30, 50, 120),
                    help="vertex color (실물 색과 비슷하게)")
    args = ap.parse_args()
    verts, faces = load_sim_mesh()
    write_ply(args.out, verts, faces, tuple(args.rgb))
    lo, hi = verts.min(0), verts.max(0)
    print(f"saved {args.out}: {len(verts)} verts / {len(faces)} tris, "
          f"z {lo[2]:+.4f}..{hi[2]:+.4f} m (origin = sim body frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
