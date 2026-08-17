#!/usr/bin/env python3
"""Fabrics URDF 를 DG-5F → DG-5FS 기구학으로 갱신한다.

배경(2026-08-17):
  로봇 자산이 openarm_tesollo_bi_rl(DG-5F) → openarm_tesollo_bi_s_rl(DG-5FS)로 바뀌었다.
  조인트/링크 **이름은 동일**하지만 기구학이 전면 재정의됐다:
    · 회전축   0 1 0 / 1 0 0  →  0 0 1 (전부, origin rpy 도입)
    · 마디 길이 PIP/DIP 0.0388 → 0.0334 m (14% 단축), tip 0.0255 → 0.018
    · palm 오프셋 0.0698 → 0.015 m
    · 조인트 한계 20개 중 10개 변경
  Fabrics 는 **별도 URDF**(models/robots/urdf/openarm_tesollo*)를 쓰고, 그 값이 구 DG-5F 와
  완전히 일치함을 확인했다. 갱신하지 않으면 palm pose IK 가 존재하지 않는 손을 푼다.

네이밍 매핑(검증됨 — 토폴로지 일치 확인):
  fabric rj_dg_{f}_{j}  ←  asset {p}_hj_{name}_{j}   (f: 1=thumb 2=index 3=middle 4=ring 5=pinky)
  fabric rl_dg_{f}_{j}  ←  asset {p}_hl_{name}_{j}

무엇을 바꾸나:
  ① 손 revolute 20개: origin(xyz/rpy) · axis · limit
  ② palm_link_joint: palm 오프셋 변화분만 **델타 적용**(fabric 의 mount 규약을 보존하기 위해
     절대값 재계산이 아니라 차분으로 옮긴다 — fabric mount 0.0595695 vs asset 0.0495 처럼
     두 URDF 의 손목 기준이 다르기 때문)
  ③ tip 고정조인트 5개: 새 tip 오프셋
  ④ 충돌 구 프레임: 마디 길이 비율로 스케일(근사 — 검증 대상)

검증: --verify 로 fabric/asset 양쪽 FK 를 돌려 손끝 위치 오차를 비교한다.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np

FINGER = {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "pinky"}


# ---------------------------------------------------------------------------
# URDF 파싱 (정규식 — 두 파일 모두 속성 순서가 달라 rpy/xyz 를 개별로 잡는다)
# ---------------------------------------------------------------------------
def _attr(body: str, tag: str, name: str) -> str | None:
    m = re.search(rf'<{tag}[^>]*\b{name}="([^"]*)"', body)
    return m.group(1) if m else None


def parse_joints(text: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for m in re.finditer(r'<joint name="([^"]+)"[^>]*type="([^"]+)"[^>]*>(.*?)</joint>',
                         text, re.S):
        n, ty, body = m.group(1), m.group(2), m.group(3)
        out[n] = {
            "type": ty,
            "body": body,
            "span": m.span(),
            "xyz": _attr(body, "origin", "xyz"),
            "rpy": _attr(body, "origin", "rpy"),
            "axis": _attr(body, "axis", "xyz"),
            "lower": _attr(body, "limit", "lower"),
            "upper": _attr(body, "limit", "upper"),
            "effort": _attr(body, "limit", "effort"),
            "velocity": _attr(body, "limit", "velocity"),
            "parent": _attr(body, "parent", "link"),
            "child": _attr(body, "child", "link"),
        }
    return out


def _v(s: str | None) -> np.ndarray:
    return np.zeros(3) if not s else np.array([float(x) for x in s.replace(",", " ").split()])


def rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_R(axis: np.ndarray, q: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * (K @ K)


def fk_chain(joints: dict[str, dict], order: list[str], q: dict[str, float]):
    """루트부터 order 순으로 누적 변환. 각 조인트 child 프레임의 (R, t) 를 돌려준다."""
    R, t = np.eye(3), np.zeros(3)
    frames = {}
    for jn in order:
        j = joints[jn]
        Rj = rpy_to_R(_v(j["rpy"]))
        tj = _v(j["xyz"])
        t = t + R @ tj
        R = R @ Rj
        if j["type"] == "revolute":
            R = R @ axis_R(_v(j["axis"]), q.get(jn, 0.0))
        frames[jn] = (R.copy(), t.copy())
    return frames


# ---------------------------------------------------------------------------
# 생성
# ---------------------------------------------------------------------------
def _sub_joint(text: str, name: str, *, xyz=None, rpy=None, axis=None,
               lower=None, upper=None) -> str:
    """지정 조인트 블록 안의 origin/axis/limit 속성만 외과적으로 치환."""
    m = re.search(rf'(<joint name="{re.escape(name)}"[^>]*>)(.*?)(</joint>)', text, re.S)
    if not m:
        raise KeyError(f"fabric URDF 에 조인트 없음: {name}")
    head, body, tail = m.group(1), m.group(2), m.group(3)

    if xyz is not None or rpy is not None:
        om = re.search(r'<origin[^>]*/>', body)
        if not om:
            raise KeyError(f"{name}: origin 태그 없음")
        o = om.group(0)
        if xyz is not None:
            o = (re.sub(r'\bxyz="[^"]*"', f'xyz="{xyz}"', o) if 'xyz="' in o
                 else o.replace("/>", f' xyz="{xyz}" />'))
        if rpy is not None:
            o = (re.sub(r'\brpy="[^"]*"', f'rpy="{rpy}"', o) if 'rpy="' in o
                 else o.replace("/>", f' rpy="{rpy}" />'))
        body = body[:om.start()] + o + body[om.end():]

    if axis is not None:
        body = re.sub(r'(<axis[^>]*\bxyz=")[^"]*(")', rf'\g<1>{axis}\g<2>', body)
    if lower is not None:
        body = re.sub(r'(<limit[^>]*\blower=")[^"]*(")', rf'\g<1>{lower}\g<2>', body)
    if upper is not None:
        body = re.sub(r'(<limit[^>]*\bupper=")[^"]*(")', rf'\g<1>{upper}\g<2>', body)

    return text[:m.start()] + head + body + tail + text[m.end():]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="assets/robot/openarm_tesollo_bi_s_rl/"
                                       "openarm_tesollo_bi_s_rl.urdf")
    ap.add_argument("--fabric", required=True, help="갱신할 fabric URDF (템플릿이자 출력)")
    ap.add_argument("--prefix", default="r", choices=["r", "l"], help="자산에서 읽을 팔")
    ap.add_argument("--out", default="", help="비우면 --fabric 을 제자리 갱신")
    ap.add_argument("--verify", action="store_true", help="FK 대조만 수행(쓰기 없음)")
    a = ap.parse_args()

    asset = parse_joints(Path(a.asset).read_text(encoding="utf-8"))
    fab_text = Path(a.fabric).read_text(encoding="utf-8")
    fab = parse_joints(fab_text)
    p = a.prefix

    # ---- 매핑 확인 -------------------------------------------------------
    pairs: list[tuple[str, str]] = []
    for f, nm in FINGER.items():
        for j in (1, 2, 3, 4):
            fk, ak = f"rj_dg_{f}_{j}", f"{p}_hj_{nm}_{j}"
            if fk not in fab:
                raise KeyError(f"fabric 에 {fk} 없음")
            if ak not in asset:
                raise KeyError(f"asset 에 {ak} 없음")
            pairs.append((fk, ak))

    if a.verify:
        _verify(fab, asset, p)
        return

    out = fab_text
    changed = []

    # ① 손 revolute 20개
    for fk, ak in pairs:
        s = asset[ak]
        out = _sub_joint(out, fk, xyz=s["xyz"], rpy=s["rpy"], axis=s["axis"],
                         lower=s["lower"], upper=s["upper"])
        changed.append(fk)

    # ② palm_link_joint — 손목→palm 체인을 자산에서 **절대 재계산**한다.
    #    처음엔 palm 오프셋 변화분만 옮기려 했으나, 자산 재생성으로 mount 도 함께 바뀌었다
    #    (0.0595695 → 0.0495, −10.07mm). fabric 은 구 mount 로 접혀 있어 DG-5FS 전환과 무관하게
    #    **이미 10mm 어긋난 상태**였다. 델타만 옮기면 그 오차가 그대로 남는다.
    #    체인: {p}_hj_mount → adapter → base → palm. 중간 회전은 yaw 뿐이라 z 는 단순 합산.
    chain = [f"{p}_hj_mount", f"{p}_hj_adapter", f"{p}_hj_base", f"{p}_hj_palm"]
    zs = [_v(asset[c]["xyz"])[2] for c in chain]
    z_new = float(sum(zs))
    pj = _v(fab["palm_link_joint"]["xyz"])
    out = _sub_joint(out, "palm_link_joint",
                     xyz=f"{pj[0]:.7g} {pj[1]:.7g} {z_new:.7g}")
    changed.append(
        f"palm_link_joint (z {pj[2]:.7g} → {z_new:.7g} = "
        + " + ".join(f"{c.split('_hj_')[1]}({z:.7g})" for c, z in zip(chain, zs)) + ")"
    )

    # ③ tip 고정조인트 5개 — 변형마다 이름이 다르다(openarm_tesollo* 는 rl_dg_{f}_tip_joint,
    #    openarm_tesollo_sensor 는 rj_dg_{f}_tip). 둘 다 시도하고 없으면 명시적으로 보고한다.
    for f, nm in FINGER.items():
        ak = f"{p}_hj_{nm}_tip"
        if ak not in asset:
            continue
        s = asset[ak]
        for cand in (f"rl_dg_{f}_tip_joint", f"rj_dg_{f}_tip"):
            if cand in fab:
                out = _sub_joint(out, cand, xyz=s["xyz"], rpy=s["rpy"])
                changed.append(cand)
                break
        else:
            raise KeyError(f"tip 조인트를 못 찾음(finger {f}): "
                           f"rl_dg_{f}_tip_joint / rj_dg_{f}_tip 모두 부재")

    dst = Path(a.out or a.fabric)
    dst.write_text(out, encoding="utf-8")
    print(f"[gen] {dst} 갱신 — 조인트 {len(changed)}개")
    for c in changed:
        print(f"       {c}")
    print("[gen] ⚠️ 충돌 구 프레임(tesollo_*_sphere2)은 미변경 — 마디 단축(0.0388→0.0334)에")
    print("       맞춰 별도 검토 필요. FK 정확도에는 영향 없음(충돌 근사 전용).")


def _verify(fab: dict, asset: dict, p: str) -> None:
    """같은 관절각에서 fabric/asset 손끝 위치가 일치하는지 대조."""
    rng = np.random.default_rng(0)
    worst = 0.0
    n_cmp = 0          # ★비교 0건이면 "오차 0"이 되어 거짓 통과한다 — 반드시 세어야 한다
    for trial in range(5):
        q = {}
        qa = {}
        for f, nm in FINGER.items():
            for j in (1, 2, 3, 4):
                lo = float(asset[f"{p}_hj_{nm}_{j}"]["lower"])
                hi = float(asset[f"{p}_hj_{nm}_{j}"]["upper"])
                v = float(rng.uniform(lo, hi))
                q[f"rj_dg_{f}_{j}"] = v
                qa[f"{p}_hj_{nm}_{j}"] = v
        for f, nm in FINGER.items():
            tipf = next((c for c in (f"rl_dg_{f}_tip_joint", f"rj_dg_{f}_tip") if c in fab), None)
            fo = [f"rj_dg_{f}_{j}" for j in (1, 2, 3, 4)] + ([tipf] if tipf else [])
            ao = [f"{p}_hj_{nm}_{j}" for j in (1, 2, 3, 4)] + [f"{p}_hj_{nm}_tip"]
            if tipf is None or not all(k in fab for k in fo) or not all(k in asset for k in ao):
                print(f"  ★건너뜀 {nm}: fabric/asset 프레임 부재 (tip={tipf})")
                continue
            ff = fk_chain(fab, fo, q)[fo[-1]][1]
            aa = fk_chain(asset, ao, qa)[ao[-1]][1]
            err = float(np.linalg.norm(ff - aa))
            worst = max(worst, err)
            n_cmp += 1
            print(f"  trial{trial} {nm:7s} fabric={np.round(ff, 5)} asset={np.round(aa, 5)} "
                  f"오차={err*1000:.3f} mm")
    if n_cmp == 0:
        raise SystemExit("[verify] ★비교 0건 — 프레임 이름이 안 맞는다. "
                         "'오차 0 = 일치'로 읽으면 거짓 통과다.")
    ok = worst < 1e-4
    print(f"\n[verify] 비교 {n_cmp}건 | 최대 손끝 오차 = {worst*1000:.3f} mm "
          f"({'일치' if ok else '★불일치 — 매핑/치환 재검토'})")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
