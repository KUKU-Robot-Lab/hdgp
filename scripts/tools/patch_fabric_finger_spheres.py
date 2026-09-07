"""Fabrics 손가락 충돌 구를 실측 형상에 맞게 재배치한다.

목적: PhysX self-collision 을 끄려면 손가락 관통을 `body_repulsion` 이 계획 단계에서
막아야 한다. 그런데 현행 구 배치로는 두 가지 이유로 불가능하다.

  ① **반경이 2.5배 과대** — 20mm 인데 링크 STL 실측 단면이 16.1 x 19.6mm(반경 8~10mm).
     정상 파지의 마디 간 최소거리가 20.9mm 인데 두 구 반경 합이 40mm 라 **상시 충돌**로
     잡힌다(실측: 다른 손가락 구쌍 최소여유 −31.3mm).
  ② **마디를 따라가지 않는다** — 오프셋이 (0,0,0.02) 인데 마디 방향은 x 축이다.
     구가 손가락 옆으로 20mm 튀어나가 있다.
  ③ 감쌈 담당 `_2`·`_4` 마디에는 구가 **아예 없다**. 관통 실측 3쌍 중 둘이 `_4` 다.

반경 근거(세 갈래가 한 대역으로 모인다):
    링크 STL 단면        16.1 x 19.6mm  → 반경 8~10mm
    정상 파지 최소거리   20.9mm         → 2r < 20.9 이어야 오탐 없음
    관통 최대거리        16.9mm         → 2r > 16.9 이어야 놓치지 않음
  → **r = 9mm** (합 18mm)

관통 실측(self-collision OFF, 완전 폐합, 160쌍 중 3쌍만):
    thumb_4 ↔ index_3  13.7mm   ring_4 ↔ pinky_1  15.9mm   ring_3 ↔ pinky_1  16.9mm

★생성 파이프라인 주의: 구 배치의 원본은 `rl_ws/urdf/eef/fabric_templates/` 이고
  `rl_ws/urdf/tools/gen_fabric_urdfs.py` 가 그것을 찍어낸다. 이 스크립트는 **찍혀 나온
  결과를 후처리**한다 — fabric URDF 를 재생성하면 다시 돌려야 한다.
  영구 해법은 템플릿에 반영하는 것이다(rl_ws/urdf 는 사용자 관리 영역).

    python scripts/tools/patch_fabric_finger_spheres.py --dry-run
    python scripts/tools/patch_fabric_finger_spheres.py
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

FABRIC_URDF_DIR = Path("source/FABRICS/src/fabrics_sim/models/robots/urdf")
PARAMS_DIR = Path("source/FABRICS/src/fabrics_sim/fabric_params")

RADIUS_M = 0.009            # 실측 근거는 모듈 docstring 참조
FINGERS = (1, 2, 3, 4, 5)
SEGMENTS = ("1", "2", "3", "4")
# 구 개수는 **마디 길이에서 자동 산출**한다 — 고정값(2개)으로는 엄지 근위(38.1mm)와
# 새끼 뿌리(39.2mm)에 틈이 생긴다. n = ceil(길이 / 지름) 이면 구가 이어 붙는다.
MIN_SPHERES_PER_SEG = 1


def _joint_origin(txt: str, parent: str, child: str) -> tuple[float, float, float] | None:
    """parent → child 고정/회전 조인트의 origin xyz. 마디 벡터가 된다."""
    for m in re.finditer(r"<joint\b[^>]*>(.*?)</joint>", txt, re.S):
        body = m.group(1)
        p = re.search(r'<parent\s+link="([^"]+)"', body)
        c = re.search(r'<child\s+link="([^"]+)"', body)
        if not (p and c) or p.group(1) != parent or c.group(1) != child:
            continue
        o = re.search(r'<origin\s+xyz="([^"]+)"', body)
        if not o:
            return (0.0, 0.0, 0.0)
        return tuple(float(v) for v in o.group(1).split())
    return None


def _link_names(txt: str, fi: int) -> tuple[str, str]:
    """(마디 링크 접두사, tip 링크명) — 자산마다 접두사가 다르다."""
    seg = re.search(rf'link name="([\w]*rl_dg_{fi}_1)"', txt)
    tip = re.search(rf'link name="([\w]*rl_dg_{fi}_tip)"', txt)
    if not (seg and tip):
        raise RuntimeError(f"손가락 {fi} 링크를 못 찾았다")
    return seg.group(1)[:-1], tip.group(1)      # "..._dg_2_", "rl_dg_2_tip"


def patch_urdf(path: Path, dry: bool) -> list[tuple[str, float]]:
    """손가락 구 프레임을 재생성하고 (프레임명, 반경) 목록을 돌려준다."""
    txt = path.read_text()

    # 기존 손가락 구 링크/조인트 제거 (팔·몸통 구는 건드리지 않는다)
    txt = re.sub(r"\s*<link name=\"[\w]*rl_dg_\d+_\d+_sphere\d+\">.*?</link>", "", txt, flags=re.S)
    txt = re.sub(r"\s*<joint name=\"[\w]*rl_dg_\d+_\d+_joint_sphere\d+\"[^>]*>.*?</joint>",
                 "", txt, flags=re.S)

    blocks, frames = [], []
    for fi in FINGERS:
        pre, tip = _link_names(txt, fi)
        for si, s in enumerate(SEGMENTS):
            parent = f"{pre}{s}"
            child = f"{pre}{SEGMENTS[si + 1]}" if si + 1 < len(SEGMENTS) else tip
            vec = _joint_origin(txt, parent, child)
            if vec is None:
                raise RuntimeError(f"마디 {parent}->{child} 조인트를 못 찾았다")
            length = math.sqrt(sum(v * v for v in vec))
            n = max(MIN_SPHERES_PER_SEG, math.ceil(length / (2 * RADIUS_M)))
            for k in range(n):
                # 마디 **방향을 따라** 균등 배치. 기존 (0,0,0.02) 는 마디와 무관한 축이라
                # 구가 손가락 옆으로 튀어나가 있었다.
                t = (k + 0.5) / n
                name = f"{parent}_sph{k + 1}"
                blocks.append(
                    f'  <link name="{name}">\n'
                    f"    <inertial>\n"
                    f'      <origin xyz="0. 0. 0." rpy="0. 0. 0." />\n'
                    f'      <mass value="0.001" />\n'
                    f'      <inertia ixx="1e-6" ixy="0." ixz="0." iyy="1e-6" iyz="0." izz="1e-6" />\n'
                    f"    </inertial>\n"
                    f"  </link>\n"
                    f'  <joint name="{name}_joint" type="fixed">\n'
                    f'    <origin xyz="{vec[0]*t:.6f} {vec[1]*t:.6f} {vec[2]*t:.6f}" '
                    f'rpy="0. 0. 0." />\n'
                    f'    <parent link="{parent}" />\n'
                    f'    <child link="{name}" />\n'
                    f"  </joint>\n"
                )
                frames.append((name, RADIUS_M))
            assert length <= 2 * RADIUS_M * n + 1e-9, (
                f"{parent}: 길이 {length*1000:.1f}mm 를 구 {n}개로 못 덮는다")

    txt = txt.replace("</robot>", "".join(blocks) + "</robot>")
    if not dry:
        path.write_text(txt)
    print(f"{path.name}: 손가락 구 {len(frames)}개 재생성"
          f"{' (dry-run)' if dry else ''}")
    return frames


def patch_params(path: Path, frames: list[tuple[str, float]], dry: bool) -> None:
    """yaml 의 collision_sphere_frames/radii 를 갈아끼우고 손가락 쌍을 추가한다."""
    txt = path.read_text()
    old_frames = re.search(r"collision_sphere_frames:\s*\n(\s*\[.*?\])", txt, re.S)
    old_radii = re.search(r"collision_sphere_radii:\s*(\[.*?\])", txt, re.S)
    if not (old_frames and old_radii):
        raise RuntimeError(f"{path.name}: collision_sphere_* 블록을 못 찾았다")

    keep = [f for f in re.findall(r'"([^"]+)"', old_frames.group(1)) if "rl_dg_" not in f]
    keep_r = [float(v) for v in re.findall(r"[\d.]+", old_radii.group(1))]
    old_all = re.findall(r'"([^"]+)"', old_frames.group(1))
    keep_r = [keep_r[i] for i, n in enumerate(old_all) if "rl_dg_" not in n]

    names = keep + [n for n, _ in frames]
    radii = keep_r + [r for _, r in frames]
    f_txt = "          [" + ",\n           ".join(f'"{n}"' for n in names) + "]"
    r_txt = "[" + ", ".join(f"{r}" for r in radii) + "]"
    txt = txt[:old_frames.start(1)] + f_txt + txt[old_frames.end(1):]
    old_radii = re.search(r"collision_sphere_radii:\s*(\[.*?\])", txt, re.S)
    txt = txt[:old_radii.start(1)] + r_txt + txt[old_radii.end(1):]

    # 손가락↔손가락 prefix 쌍 — 같은 손가락끼리는 넣지 않는다(캡슐이라 항상 겹친다).
    pre = frames[0][0][: frames[0][0].index("rl_dg_")]
    pairs = []
    for i, fa in enumerate(FINGERS):
        for fb in FINGERS[i + 1:]:
            for sa in SEGMENTS:
                for sb in SEGMENTS:
                    pairs.append(f'["{pre}rl_dg_{fa}_{sa}_sph", "{pre}rl_dg_{fb}_{sb}_sph"]')
    marker = "collision_link_prefix_pairs: ["
    idx = txt.index(marker) + len(marker)
    add = ("\n                                    # ★손가락↔손가락 — self-collision 을 끄기 위한 쌍.\n"
           "                                    #   반경 9mm(2r=18mm)라 정상 파지(20.9mm)는 안 걸리고\n"
           "                                    #   관통(13.7~16.9mm)만 잡는다.\n                                    "
           + ",\n                                    ".join(pairs) + ",")
    txt = txt[:idx] + add + txt[idx:]
    if not dry:
        path.write_text(txt)
    print(f"{path.name}: 구 {len(names)}개(손가락 {len(frames)}) · 손가락쌍 {len(pairs)}개 추가"
          f"{' (dry-run)' if dry else ''}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="openarm_tesollo_bi_s")
    ap.add_argument("--params", default="openarm_tesollo_pose_params.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-params", action="store_true",
                    help="yaml 은 건드리지 않는다 — 좌우가 같은 params 를 공유하므로\n                         두 번째 자산부터는 URDF 만 패치한다")
    a = ap.parse_args()

    urdf = FABRIC_URDF_DIR / a.robot / f"{a.robot}.urdf"
    if not urdf.exists():
        print(f"URDF 없음: {urdf}")
        return 1
    frames = patch_urdf(urdf, a.dry_run)
    if a.skip_params:
        print("(yaml 은 건너뛴다 — 좌우가 같은 프레임명·같은 params 를 공유한다)")
    else:
        patch_params(PARAMS_DIR / a.params, frames, a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
