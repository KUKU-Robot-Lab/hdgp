#!/usr/bin/env python3
"""Allegro eigengrasp PCA(5×16) → Tesollo DG-5F(5×20) 리타겟.

DEXTRAH kuka_allegro_pose_fabric.py 하드코딩 PCA 행렬을 관절 의미 대응으로
tesollo 20관절 공간에 이식한다. numpy 전용(오프라인) — 학습 코드와 무관.

관절 대응 (의미 기준):
  Allegro finger [j0=벌림, j1=MCP, j2=PIP, j3=DIP]
  Tesollo index/middle/ring [_1=벌림, _2=MCP, _3=PIP, _4=DIP]  → 1:1
  Tesollo pinky  [_1=Z-flex(미사용), _2=벌림, _3=MCP curl, _4=DIP]
                 ← Allegro ring 복제 (_2←j0, _3←j1, _4←(j2+j3)/2)
  Tesollo thumb  [_1=벌림, _2=대향 Z-curl(감김=음수), _3=PIP, _4=DIP]
                 ← Allegro thumb (_1←j1, _2←−j0, _3←j2, _4←j3)

스케일: 관절별 (tesollo 가동범위 / allegro 가동범위) × 감김방향 부호.
후처리: Gram-Schmidt 직교정규화(원 순서 유지) → 계수 범위는 open pose(0)와
HAND_FULL_GRIP_POSE 투영으로 산출.

출력: hdgp/data/tesollo_hand_pca.npz
  basis (5,20) 직교정규 행벡터 / coeff_open (5,) / coeff_grip (5,)
  coeff_mins/maxs (5,) / joint_names (20,)
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_HDGP = os.path.normpath(os.path.join(_HERE, "..", ".."))
_FABRICS_URDF = os.path.join(
    _HDGP, "source", "FABRICS", "src", "fabrics_sim", "models", "robots", "urdf"
)

# ---------------------------------------------------------------------------
# DEXTRAH 원본 Allegro PCA (kuka_allegro_pose_fabric.py add_hand_fabric 하드코딩)
# 열 순서 = URDF 정의 순: index_0..3, middle_0..3, ring_0..3, thumb_0..3
# ---------------------------------------------------------------------------
ALLEGRO_PCA = np.array([
    [-3.8872e-02,  3.7917e-01,  4.4703e-01,  7.1016e-03,  2.1159e-03,
      3.2014e-01,  4.4660e-01,  5.2108e-02,  5.6869e-05,  2.9845e-01,
      3.8575e-01,  7.5774e-03, -1.4790e-02,  9.8163e-02,  4.3551e-02,
      3.1699e-01],
    [-5.1148e-02, -1.3007e-01,  5.7727e-02,  5.7914e-01,  1.0156e-02,
     -1.8469e-01,  5.3809e-02,  5.4888e-01,  1.3351e-04, -1.7747e-01,
      2.7809e-02,  4.8187e-01,  2.9753e-02,  2.6149e-02,  6.6994e-02,
      1.8117e-01],
    [-5.7137e-02, -3.4707e-01,  3.3365e-01, -1.8029e-01, -4.3560e-02,
     -4.7666e-01,  3.2517e-01, -1.5208e-01, -5.9691e-05, -4.5790e-01,
      3.6536e-01, -1.3916e-01,  2.3925e-03,  3.7238e-02, -1.0124e-01,
     -1.7442e-02],
    [ 2.2795e-02, -3.4090e-02,  3.4366e-02, -2.6531e-02,  2.3471e-02,
      4.6123e-02,  9.8059e-02, -1.2619e-03, -1.6452e-04, -1.3741e-02,
      1.3813e-01,  2.8677e-02,  2.2661e-01, -5.9911e-01,  7.0257e-01,
     -2.4525e-01],
    [-4.4911e-02, -4.7156e-01,  9.3124e-02,  2.3135e-01, -2.4607e-03,
      9.5564e-02,  1.2470e-01,  3.6613e-02,  1.3821e-04,  4.6072e-01,
      9.9315e-02, -8.1080e-02, -4.7617e-01, -2.7734e-01, -2.3989e-01,
     -3.1222e-01],
])

ALLEGRO_JOINTS = [
    f"{f}_joint_{j}" for f in ("index", "middle", "ring", "thumb") for j in range(4)
]

# Tesollo 프로젝트 관절 순서 (grasp_right_preset finger-major)
TESOLLO_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
TESOLLO_JOINTS = [f"r_hj_{f}_{j}" for f in TESOLLO_FINGERS for j in range(1, 5)]
# fabric URDF 이름 (dg1=thumb .. dg5=pinky)
_DG = {"thumb": 1, "index": 2, "middle": 3, "ring": 4, "pinky": 5}

# HAND_FULL_GRIP_POSE (grasp_right_preset) — 감김 방향 부호의 근거
FULL_GRIP = np.array([
    0.0, -1.57, 1.8, 1.8,   # thumb
    0.0,  1.9,  1.8, 1.8,   # index
    0.0,  1.9,  1.8, 1.8,   # middle
    0.0,  1.9,  1.8, 1.8,   # ring
    0.0,  0.0,  1.8, 1.8,   # pinky
])
# 앵커 = HAND_APPROACH_POSE. Allegro PCA가 "엄지 이미 대향된 rest" 주변에서
# 동작하듯(allegro thumb_0 범위가 상시 대향), tesollo도 접근 자세(thumb_2=-1.57
# 사전 대향)를 원점으로 둬야 PC1(전지 감김)에 대향 생성 부담이 없다.
ANCHOR = np.array([
    0.0, -1.57, -0.5, 0.0,  # thumb (approach: 대향 + PIP 사전 굽힘)
    0.0,  0.0,   0.0, 0.0,  # index
    0.0,  0.0,   0.0, 0.0,  # middle
    0.0,  0.0,   0.0, 0.0,  # ring
    0.0,  0.0,   0.0, 0.0,  # pinky
])


def parse_limits(urdf_path: str, name_filter: str) -> dict[str, tuple[float, float]]:
    root = ET.parse(urdf_path).getroot()
    out = {}
    for j in root.iter("joint"):
        n = j.get("name", "")
        if j.get("type") != "revolute" or not re.match(name_filter, n):
            continue
        lim = j.find("limit")
        out[n] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def build_retarget_map() -> list[tuple[int, int, float]]:
    """(tesollo_idx, allegro_idx, sign) 대응 목록. (j2+j3)/2 병합은 두 항으로."""
    pairs: list[tuple[int, int, float]] = []
    a_idx = {n: i for i, n in enumerate(ALLEGRO_JOINTS)}
    t_idx = {n: i for i, n in enumerate(TESOLLO_JOINTS)}

    for f in ("index", "middle", "ring"):
        for tj, aj in ((1, 0), (2, 1), (3, 2), (4, 3)):
            pairs.append((t_idx[f"r_hj_{f}_{tj}"], a_idx[f"{f}_joint_{aj}"], 1.0))
    # pinky ← ring 복제 (_2←j0, _3←j1, _4←(j2+j3)/2)
    pairs.append((t_idx["r_hj_pinky_2"], a_idx["ring_joint_0"], 1.0))
    pairs.append((t_idx["r_hj_pinky_3"], a_idx["ring_joint_1"], 1.0))
    pairs.append((t_idx["r_hj_pinky_4"], a_idx["ring_joint_2"], 0.5))
    pairs.append((t_idx["r_hj_pinky_4"], a_idx["ring_joint_3"], 0.5))
    # thumb: _1←j1(벌림), _2←−j0(대향: tesollo 감김=음수), _3←j2, _4←j3
    pairs.append((t_idx["r_hj_thumb_1"], a_idx["thumb_joint_1"], 1.0))
    pairs.append((t_idx["r_hj_thumb_2"], a_idx["thumb_joint_0"], -1.0))
    pairs.append((t_idx["r_hj_thumb_3"], a_idx["thumb_joint_2"], 1.0))
    pairs.append((t_idx["r_hj_thumb_4"], a_idx["thumb_joint_3"], 1.0))
    return pairs


def main() -> None:
    allegro_lim = parse_limits(
        os.path.join(_FABRICS_URDF, "kuka_allegro", "kuka_allegro.urdf"),
        r"^(index|middle|ring|thumb)_joint_\d$",
    )
    tesollo_raw = parse_limits(
        os.path.join(_FABRICS_URDF, "openarm_tesollo_sensor", "openarm_tesollo_sensor.urdf"),
        r"^rj_dg_\d_\d$",
    )
    # rj_dg_{dg}_{j} → r_hj_{finger}_{j}
    tesollo_lim = {}
    for f in TESOLLO_FINGERS:
        for j in range(1, 5):
            tesollo_lim[f"r_hj_{f}_{j}"] = tesollo_raw[f"rj_dg_{_DG[f]}_{j}"]

    a_range = np.array([allegro_lim[n][1] - allegro_lim[n][0] for n in ALLEGRO_JOINTS])
    t_range = np.array([tesollo_lim[n][1] - tesollo_lim[n][0] for n in TESOLLO_JOINTS])
    t_lower = np.array([tesollo_lim[n][0] for n in TESOLLO_JOINTS])
    t_upper = np.array([tesollo_lim[n][1] for n in TESOLLO_JOINTS])

    # 5×20 리타겟: 열 대응 + (t_range/a_range) 스케일 + 부호
    basis = np.zeros((5, 20))
    for t_i, a_i, sgn in build_retarget_map():
        basis[:, t_i] += ALLEGRO_PCA[:, a_i] * sgn * (t_range[t_i] / a_range[a_i])

    # Gram-Schmidt 직교정규화 (PC1→PC5 순서 유지)
    ortho = []
    for v in basis:
        w = v.copy()
        for u in ortho:
            w -= (w @ u) * u
        n = np.linalg.norm(w)
        assert n > 1e-6, "직교화 중 rank 붕괴"
        ortho.append(w / n)
    B = np.stack(ortho)   # (5,20) 행 직교정규

    # PC1 부호 정렬: 감김 방향(FULL_GRIP-OPEN)과 양의 내적이 되도록
    close_dir = FULL_GRIP - ANCHOR
    for k in range(5):
        if B[k] @ close_dir < 0 and k == 0:
            B[k] = -B[k]

    # 계수 범위: open(0)·FULL_GRIP 투영 (open 기준 좌표)
    c_open = B @ (ANCHOR - ANCHOR)   # = 0 (앵커 기준 좌표)
    c_grip = B @ (FULL_GRIP - ANCHOR)
    margin = 0.35 * np.abs(c_grip).max()
    c_min = np.minimum(c_open, c_grip) - margin
    c_max = np.maximum(c_open, c_grip) + margin

    out = os.path.join(_HDGP, "data", "tesollo_hand_pca.npz")
    np.savez(
        out,
        basis=B, coeff_open=c_open, coeff_grip=c_grip,
        coeff_mins=c_min, coeff_maxs=c_max,
        joint_names=np.array(TESOLLO_JOINTS), anchor=ANCHOR,
        joint_lower=t_lower, joint_upper=t_upper,
    )
    print(f"저장: {out}\n")

    # ---- 학습 코드용 파이썬 모듈 생성 (data/ 는 gitignore — 상수로 버전 관리) ----
    def _fmt(arr, indent):
        rows = [
            "[" + ", ".join(f"{v: .6f}" for v in row) + "]"
            for row in np.atleast_2d(arr)
        ]
        sep = ",\n" + " " * indent
        return sep.join(rows)

    mod = os.path.join(
        _HDGP, "source", "openarm", "openarm", "tesollo", "right", "grasp_v2",
        "tesollo_hand_synergy.py",
    )
    with open(mod, "w", encoding="utf-8") as fh:
        fh.write(
            '"""Tesollo hand synergy basis — retarget_allegro_pca_to_tesollo.py 생성물.\n\n'
            "DEXTRAH Allegro eigengrasp PCA(5×16)를 관절 의미 대응으로 이식한 5×20\n"
            "직교정규 basis. 수동 편집 금지 — 스크립트 재실행으로 갱신.\n"
            "행 순서: PC1=전지 감김(파워), PC2=말단 감김, PC3=파지형 재편,\n"
            "PC4=엄지 배치, PC5=미세 분화. 열 순서 = finger-major\n"
            "[thumb,index,middle,ring,pinky]×[_1.._4] (grasp_right_preset 동일).\n"
            '"""\n\n'
            f"HAND_SYNERGY_BASIS = [\n    {_fmt(B, 4)},\n]\n\n"
            f"HAND_SYNERGY_ANCHOR = {_fmt(ANCHOR, 0)}  # = HAND_APPROACH_POSE\n\n"
            f"HAND_SYNERGY_COEFF_MINS = {_fmt(c_min, 0)}\n"
            f"HAND_SYNERGY_COEFF_MAXS = {_fmt(c_max, 0)}\n"
            f"HAND_SYNERGY_COEFF_GRIP = {_fmt(c_grip, 0)}  # FULL_GRIP 투영치(참고)\n"
        )
    print(f"모듈 생성: {mod}\n")

    # ---- 검증 스윕 ----
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("계수 범위 (open=0 기준):")
    for k in range(5):
        print(f"  PC{k+1}: grip투영 {c_grip[k]:7.3f}  범위 [{c_min[k]:7.3f}, {c_max[k]:7.3f}]")
    print()
    for k in range(5):
        q = np.clip(ANCHOR + c_grip[k] * B[k], t_lower, t_upper)
        moved = [
            f"{TESOLLO_JOINTS[i].replace('r_hj_', '')}:{q[i]:+.2f}"
            for i in np.argsort(-np.abs(q))[:8] if abs(q[i]) > 0.05
        ]
        print(f"PC{k+1} @grip투영치 → {moved}")
    # 전체 조합: 각 PC를 grip 투영값으로 모두 밟았을 때 = FULL_GRIP 재구성 오차
    q_full = np.clip(ANCHOR + (c_grip[None, :] @ B).ravel(), t_lower, t_upper)
    err = np.abs(q_full - np.clip(FULL_GRIP, t_lower, t_upper))
    print(f"\nFULL_GRIP 재구성 오차: max {err.max():.3f} rad @ "
          f"{TESOLLO_JOINTS[int(err.argmax())]}, mean {err.mean():.3f}")
    # 5지 감김 확인: PC1만으로 각 손가락 대표 curl 관절이 얼마나 감기나
    q1 = np.clip(ANCHOR + c_grip[0] * B[0], t_lower, t_upper)
    reps = ["thumb_2", "index_2", "middle_2", "ring_2", "pinky_3"]
    print("\nPC1 단독(전지 감김 축) 대표 관절:")
    for r in reps:
        i = TESOLLO_JOINTS.index(f"r_hj_{r}")
        print(f"  {r:9s}: {q1[i]:+.3f} rad (grip 목표 {FULL_GRIP[i]:+.2f})")


if __name__ == "__main__":
    main()
