"""손바닥 중심(palm_ee) 오프셋 — 자산 URDF 의 fixed 조인트에서 읽는다(순수 xml, Isaac 불요).

★09.14 사용자 "손바닥의 중심쪽은 palm_ee xform". env 의 `palm_idx` 는 프로필 `palm_body`(`r_hl_palm` = 손바닥 링크 원점,
  손목 쪽)라서 생성 보상에 넘기던 `ctx.palm_pos` 가 손바닥 중심보다 손가락 반대쪽으로 40 mm · 손바닥 면 안쪽으로 28 mm 떨어진 점이었다.
  palm_ee 는 질량 없는 fixed 링크라 articulation body 로 남는다는 보장이 없어 URDF 오프셋으로 계산한다.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def palm_center_offset(urdf_path: str | Path, palm_body: str) -> tuple[float, float, float]:
    """`palm_body` → `<palm_body>_ee` fixed 조인트의 xyz [m] (palm_body 프레임).

    회전이 0 이 아니면 raise — palm_ee 축이 palm_body 축과 달라져 `palm_normal` 도 바꿔야 한다.
    """
    child = f"{palm_body}_ee"
    root = ET.parse(urdf_path).getroot()
    for joint in root.iter("joint"):
        parent, ch = joint.find("parent"), joint.find("child")
        if parent is None or ch is None or parent.get("link") != palm_body or ch.get("link") != child:
            continue
        if joint.get("type") != "fixed":
            raise ValueError(f"{urdf_path}: {palm_body} → {child} 조인트가 fixed 가 아니다({joint.get('type')})")
        origin = joint.find("origin")
        xyz = tuple(float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split())
        rpy = tuple(float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split())
        if len(xyz) != 3 or len(rpy) != 3:
            raise ValueError(f"{urdf_path}: {joint.get('name')} origin 형식이 틀렸다 xyz={xyz} rpy={rpy}")
        if any(abs(a) > 1e-9 for a in rpy):
            raise ValueError(f"{urdf_path}: {joint.get('name')} rpy={rpy} — palm_ee 축이 palm_body 와 달라 법선도 바꿔야 한다")
        return xyz
    raise ValueError(f"{urdf_path}: {palm_body} → {child} fixed 조인트가 없다")
