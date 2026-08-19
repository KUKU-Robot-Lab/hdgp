#!/usr/bin/env python3
"""[P0-1] 좌팔 2지 그리퍼 개구 vs 컵 지름 실측 (numpy + pxr, Isaac 불필요).

왜 필요한가
-----------
openarm_tesollo_sensor_rl 왼손은 스트로크 0~0.044 m 프리즈매틱 2지 그리퍼다.
두 가지를 눈대중으로 넘기면 태스크가 통째로 성립하지 않는다:

  1. **개구를 조인트 origin(∓0.006)으로 계산하면 안 된다.** 패드 실면은 조인트 프레임이
     아니라 핑거 메시가 정한다. 실측 결과 최대 개구는 이론치 100 mm 가 아니라 84.5 mm 다.
     충돌 근사가 convexHull 이므로 핑거 안쪽 오목부는 메워지고, 통과 가능 폭은
     **가장 안쪽 점(핑거 팁)** 이 지배한다.
  2. **컵이 원통이 아니다.** cup_big 은 원뿔형, shaker 는 계단형 원뿔이다.
     bbox 반경은 **림(최대치)** 을 재는 값이라 몸통 실제 지름과 크게 다르다:
       cup_big : 하단 62~71 / 중단 83~86 / 림 93 mm  → 파지 가능 h = 35~60 mm
       shaker  : 하단 58 / 68 / 78 / 상단 88 mm       → 파지 가능 h = 10~80 mm
     즉 파지 높이를 고르면 스케일 축소 없이 잡을 수 있다. bbox 만 보고
     "scale 을 줄여야 한다"고 결론내면 실물 크기를 불필요하게 버린다.

따라서 이 프로브는 "개구 vs bbox 지름"이 아니라 **"개구 vs 파지 높이별 실제 단면 지름"**을 잰다.

계산
----
핑거: 메시 정점 v[mm] → 링크 프레임  p = S·v + collision_origin  (S = diag(0.001, ±0.001, 0.001))
      링크 → gripper_base:  p += (0, joint_origin_y + axis_y·q, 0.015)
컵:   USD 메시 정점 → (r, z). 테이블 기준 높이 h = z - z_bottom.
      그리퍼가 파지 높이 h 를 겨눌 때 핑거는 x 방향 ±HALF_WIDTH 를 덮으므로,
      그 구간에서의 **최대** 지름이 통과해야 할 폭이다.

사용:
  python3 scripts/probes/probe_gripper_opening.py                        # 기본 = shaker_closed
  python3 scripts/probes/probe_gripper_opening.py --cup cup_big_rl.usd
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HDGP = Path(__file__).resolve().parents[2]
FINGER_STL = Path(
    "/home/user/rl_ws/urdf/vendor/openarm_description/meshes/ee/openarm_hand/collision/finger.stl"
)
DEFAULT_CUP = "shaker_closed_rl.usd"

# URDF openarm_tesollo_sensor_rl.urdf 실값 (링크 880-915, 조인트 1294-1308)
MESH_SCALE_MM = 0.001
STROKE_MAX = 0.044          # l_hj_gripper_1/2 limit upper [m]
JOINT_Z = 0.015
FINGERS = {
    # l_hj_gripper_2 → l_hl_gripper_left_finger (mimic multiplier 1)
    "left": dict(origin=np.array([0.0, -0.05, -0.673001]), sy=+1.0, j_y=+0.006, ax_y=+1.0),
    # l_hj_gripper_1 → l_hl_gripper_right_finger (구동 관절)
    "right": dict(origin=np.array([0.0, +0.05, -0.673001]), sy=-1.0, j_y=-0.006, ax_y=-1.0),
}

# 편측 여유 판정 임계 [m] — 계획 P0-1 게이트
CLEARANCE_GATE = 0.008


def load_stl_vertices(path: Path) -> np.ndarray:
    """binary/ascii STL → (N,3) 정점 배열 [mesh 단위, 여기서는 mm]."""
    raw = path.read_bytes()
    if raw[:5].lower().lstrip() == b"solid" and b"facet normal" in raw[:2048]:
        verts = [
            [float(x) for x in line.split()[1:4]]
            for line in raw.decode("ascii", "replace").splitlines()
            if line.strip().startswith("vertex")
        ]
        return np.asarray(verts, dtype=float)

    (n_tri,) = struct.unpack("<I", raw[80:84])
    expected = 84 + n_tri * 50
    if len(raw) != expected:
        raise ValueError(f"binary STL 크기 불일치: {len(raw)} != {expected} (tri={n_tri})")
    data = np.frombuffer(raw[84:], dtype=np.uint8).reshape(n_tri, 50)
    floats = data[:, :48].copy().view(np.float32).reshape(n_tri, 12)
    return floats[:, 3:].reshape(-1, 3).astype(float)


def finger_points_in_base(name: str, q: float, verts_mm: np.ndarray) -> np.ndarray:
    """핑거 메시 정점을 gripper_base 프레임으로 변환."""
    f = FINGERS[name]
    scale = np.array([MESH_SCALE_MM, f["sy"] * MESH_SCALE_MM, MESH_SCALE_MM])
    return verts_mm * scale + f["origin"] + np.array([0.0, f["j_y"] + f["ax_y"] * q, JOINT_Z])


def load_cup_points(cup_usd: Path) -> np.ndarray:
    from pxr import Usd, UsdGeom  # noqa: PLC0415  (pxr 는 선택적 의존)

    stage = Usd.Stage.Open(str(cup_usd))
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if pts is not None and len(pts):
            return np.asarray(pts, dtype=float)
    raise ValueError(f"{cup_usd} 에서 메시 정점을 찾지 못했다")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cup", default=DEFAULT_CUP, help="assets/cup/ 아래 USD 파일명")
    args = ap.parse_args()
    cup_usd = HDGP / "assets/cup" / args.cup
    if not cup_usd.is_file():
        print(f"[FAIL] 컵 자산 없음: {cup_usd}")
        return 1
    if not FINGER_STL.is_file():
        print(f"[FAIL] 핑거 메시 없음: {FINGER_STL}")
        return 1

    # ── 1. 핑거 기하 ────────────────────────────────────────────────
    fv = load_stl_vertices(FINGER_STL)
    p_left_open = finger_points_in_base("left", STROKE_MAX, fv)
    p_right_open = finger_points_in_base("right", STROKE_MAX, fv)
    p_left_closed = finger_points_in_base("left", 0.0, fv)

    open_gap = p_left_open[:, 1].min() - p_right_open[:, 1].max()
    closed_gap = p_left_closed[:, 1].min() - finger_points_in_base("right", 0.0, fv)[:, 1].max()
    half_width = float(max(abs(p_left_open[:, 0].min()), abs(p_left_open[:, 0].max())))
    finger_len = float(p_left_open[:, 2].max() - p_left_open[:, 2].min())

    print("=== 1. 핑거 기하 (gripper_base 프레임) ===")
    print(f"  메시 정점 {len(fv)}개, 충돌 근사 = convexHull (physics USD 확인)")
    print(f"  최대 개구(팁 통과폭) {open_gap*1000:7.2f} mm   완전 폐쇄 {closed_gap*1000:+.2f} mm")
    print(f"  핑거 길이(접근축 z) {finger_len*1000:.1f} mm   폭 반값(x) ±{half_width*1000:.1f} mm")
    print("  ※ 측면 파지에서 x=수직 → 핑거가 컵 높이 방향으로 ±%.1f mm 를 덮는다."
          % (half_width * 1000))

    # ── 2. 컵 프로파일 ──────────────────────────────────────────────
    cp = load_cup_points(cup_usd)
    cz, cr = cp[:, 2], np.hypot(cp[:, 0], cp[:, 1])
    z_bottom = float(cz.min())
    cup_height = float(cz.max() - z_bottom)
    print(f"\n=== 2. {args.cup} 프로파일 (scale 1.0) ===")
    print(f"  원점 기준 bottom z = {z_bottom:+.6f} m  (CUP_BOTTOM_TO_ORIGIN = {-z_bottom:.6f})")
    print(f"  정점 {len(cp)}개, 높이 {cup_height*1000:.1f} mm, 최대 지름 {2*cr.max()*1000:.2f} mm (림)")

    # ── 3. 파지 높이별 통과 폭 ──────────────────────────────────────
    print(f"\n=== 3. 파지 높이별 통과 폭 (게이트: 편측 여유 >= {CLEARANCE_GATE*1000:.0f} mm) ===")
    print("   h[mm]는 테이블 표면 기준 파지 중심 높이. 지름은 핑거가 덮는 ±%.0f mm 구간의 최대값."
          % (half_width * 1000))
    feasible: list[tuple[float, float, float]] = []
    for h in np.arange(0.010, cup_height - 0.005, 0.005):
        zc = z_bottom + h
        mask = (cz >= zc - half_width) & (cz <= zc + half_width)
        if not mask.any():
            continue
        d = float(2 * cr[mask].max())
        clearance = (open_gap - d) / 2.0
        ok = clearance >= CLEARANCE_GATE
        if ok:
            feasible.append((float(h), d, clearance))
        print(f"   h={h*1000:6.1f}  통과지름 {d*1000:6.2f} mm  편측 여유 {clearance*1000:+7.2f} mm  "
              f"{'PASS' if ok else 'FAIL'}")

    # ── 4. 판정 ────────────────────────────────────────────────────
    print("\n=== 4. 판정 ===")
    if not feasible:
        print("  [FAIL] scale 1.0 에서 통과 가능한 파지 높이가 없다 → 컵 스케일 축소 필요.")
        return 1
    widest = max(feasible, key=lambda t: t[2])
    highest = feasible[-1]
    h_lo, h_hi = feasible[0][0], feasible[-1][0]
    print(f"  [PASS] {args.cup} **scale 1.0 유지 가능**. 파지 가능 높이대 "
          f"h = {h_lo*1000:.0f}~{h_hi*1000:.0f} mm (테이블 기준)")
    print(f"  여유 최대 h = {widest[0]*1000:.0f} mm "
          f"(통과지름 {widest[1]*1000:.2f} mm, 편측 여유 {widest[2]*1000:.2f} mm)")
    # ★실제 채택은 보통 **대역의 최상단**이다. 낮은 파지점일수록 팔이 못 미치기 때문
    #   (probe_left_gripper_reach 실측: h=45mm 잔차 21mm vs h=100mm 잔차 0.4mm).
    #   그리퍼 여유와 팔 도달성이 반대 방향이라, 여유가 게이트를 넘는 선에서 가장 높게 잡는다.
    print(f"  최상단 통과 h = {highest[0]*1000:.0f} mm "
          f"(통과지름 {highest[1]*1000:.2f} mm, 편측 여유 {highest[2]*1000:.2f} mm) "
          f"← 팔 도달성 때문에 보통 이쪽을 채택")
    print(f"  → 다음: probe_left_gripper_reach.py 의 GRASP_HEIGHT 를 이 대역에서 정하고 재실행")
    return 0


if __name__ == "__main__":
    sys.exit(main())
