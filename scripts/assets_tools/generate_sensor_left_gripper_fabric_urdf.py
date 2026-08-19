#!/usr/bin/env python3
"""좌팔 2지 그리퍼 fabric URDF 생성 + FK 교차검증 (numpy only).

무엇을 만드나
-------------
`openarm_tesollo_sensor_rl` 의 **왼팔 7 DOF + 2지 그리퍼**를 Fabrics 가 쓸 수 있는 URDF 로 만든다.
출력: source/FABRICS/.../urdf/openarm_tesollo_sensor_left_gripper/openarm_tesollo_sensor_left_gripper.urdf

왜 우측 URDF 를 템플릿으로 편집하나 (from-scratch 가 아니라)
------------------------------------------------------------
`OpenArmTeoslloPoseFabric` 은 fabric_params 파일명이 하드코딩돼 있고
(`openarm_tesollo_pose_params.yaml`), 그 안의 `collision_sphere_frames` 는 **프레임 이름 리스트**다.
게다가 palm attractor 는 `palm_link`/`palm_x`/... 7점, fingertip taskmap 은 `rl_dg_1..5_tip` 을
이름으로 찾는다. 따라서 **링크/조인트 이름을 우측과 한 글자도 다르게 두면 안 된다**
(기존 generate_left_fabric_urdf.py 와 동일한 제약).
우측 URDF 를 복사해 값만 바꾸면 이름·메시참조·inertial 이 자동으로 보존된다.

무엇을 바꾸나
-------------
1. 팔 7관절: origin/axis/limit 를 **sensor_rl 의 `l_aj_1..7` 실값**으로 교체.
   미러 추정이 아니라 실제 좌팔 체인을 그대로 쓴다 → 생성 URDF FK == USD FK (0 오차).
   ⚠ 우측 fabric URDF(openarm_tesollo)는 sensor_rl 보다 base z 가 +8 mm 어긋나 있다
     (매니페스트 "robot origin sits at the mount plate TOP (vendor origin +8mm)").
     기존 우측 태스크는 그 편차를 안고 학습됐지만, 신설 좌측은 정합을 맞춘다.
2. 손 20관절(`rj_dg_*`): revolute → **fixed** 로 굳힌다.
   Fabrics 는 revolute 만 cspace 로 세므로(`fabrics/fabric.py:683-688`) cspace = 팔 7 DOF 가 되고,
   그리퍼 1 DOF 는 Fabrics 가 아니라 RL 액션이 직접 제어한다.
   굳힌 프레임들은 그리퍼 실제 부피를 대표하도록 재배치한다(충돌구가 엉뚱한 데 있으면
   회피가 거짓으로 작동한다). DG-5F 메시는 제거 — 위치가 무의미해지므로 남기면 오히려 해롭다.
3. `palm_link` = **그리퍼 TCP** (`l_al_7` + z 0.1001(mount) + z 0.08(tcp) = 0.1801), rpy 0.
   palm 축 = 그리퍼 고유축: **+z = 접근(툴)축, +y = jaw 개폐축, +x = 핑거 폭(측면 파지 시 수직)**.
4. palm helper 프레임(`palm_x` 등 ±0.25)은 **손대지 않는다** — 지오메트리가 아니라
   fabric 코드가 자세를 유도하는 규약이다(generate_left_fabric_urdf.py:132-139 실증).

검증
----
생성 URDF `palm_link` FK  ==  sensor_rl `l_hl_gripper_tcp` FK  (동일 q, 랜덤 20회, < 1e-9 m)

사용: python3 scripts/assets_tools/generate_sensor_left_gripper_fabric_urdf.py [--check-only]
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_left_fabric_urdf import fk_link, mat_to_rpy, parse_urdf  # noqa: E402

HDGP = Path(__file__).resolve().parents[2]
FABRIC_ROOT = HDGP / "source/FABRICS/src/fabrics_sim/models/robots/urdf"
RIGHT_FABRIC_URDF = FABRIC_ROOT / "openarm_tesollo/openarm_tesollo.urdf"
SENSOR_URDF = HDGP / "assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.urdf"
OUT_DIR = FABRIC_ROOT / "openarm_tesollo_sensor_left_gripper"
OUT_URDF = OUT_DIR / "openarm_tesollo_sensor_left_gripper.urdf"

ROBOT_NAME = "openarm_tesollo_sensor_left_gripper"

# 그리퍼 EE 기하 (sensor_rl URDF 실값)
GRIPPER_MOUNT_Z = 0.1001   # l_hj_gripper_mount: l_al_7 → l_hl_gripper_base
GRIPPER_TCP_Z = 0.08       # l_hj_gripper_tcp:   gripper_base → tcp
PALM_Z_IN_LINK7 = GRIPPER_MOUNT_Z + GRIPPER_TCP_Z   # 0.1801

# palm(=TCP) 프레임에서 본 그리퍼 부피 (probe_gripper_opening.py 실측 기반, 개방 자세)
#   핑거: gripper_base 기준 z 0.0005~0.0954, 폭 x ±0.0305, 개방 시 중심 y ±0.055
#   → palm(TCP, gripper_base z 0.08) 기준으로 z 를 -0.08 만큼 평행이동
_FY = 0.055     # 개방 시 핑거 중심 y
_FX = 0.025     # 핑거 폭 방향 대표점
_HAND_Z = -0.085  # 손 본체(gripper_base 근방)
DG_FRAME_TARGETS: dict[str, tuple[float, float, float]] = {
    # 1번 = 왼쪽 핑거 (jaw +y)
    "tesollo_right_rl_dg_1_1": (0.0, +_FY, -0.055),
    "tesollo_right_rl_dg_1_2": (0.0, +_FY, -0.030),
    "tesollo_right_rl_dg_1_3": (0.0, +_FY, -0.005),
    "tesollo_right_rl_dg_1_4": (0.0, +0.050, +0.010),
    "rl_dg_1_tip": (0.0, +0.045, +0.015),
    # 2번 = 오른쪽 핑거 (jaw -y)
    "tesollo_right_rl_dg_2_1": (0.0, -_FY, -0.055),
    "tesollo_right_rl_dg_2_2": (0.0, -_FY, -0.030),
    "tesollo_right_rl_dg_2_3": (0.0, -_FY, -0.005),
    "tesollo_right_rl_dg_2_4": (0.0, -0.050, +0.010),
    "rl_dg_2_tip": (0.0, -0.045, +0.015),
    # 3번 = 손 본체 (앞/뒤 두 점)
    "tesollo_right_rl_dg_3_1": (0.0, 0.0, -0.060),
    "tesollo_right_rl_dg_3_2": (0.0, 0.0, -0.072),
    "tesollo_right_rl_dg_3_3": (0.0, 0.0, -0.100),
    "tesollo_right_rl_dg_3_4": (0.0, 0.0, -0.100),
    "rl_dg_3_tip": (0.0, 0.0, _HAND_Z),
    # 4·5번 = 핑거 폭(x) 양끝 — 중복 구 대신 부피를 넓게 덮는다
    "tesollo_right_rl_dg_4_1": (+_FX, +_FY, -0.030),
    "tesollo_right_rl_dg_4_2": (+_FX, +_FY, -0.030),
    "tesollo_right_rl_dg_4_3": (-_FX, +_FY, -0.030),
    "tesollo_right_rl_dg_4_4": (-_FX, +_FY, -0.030),
    "rl_dg_4_tip": (0.0, +0.045, +0.015),
    "tesollo_right_rl_dg_5_1": (+_FX, -_FY, -0.030),
    "tesollo_right_rl_dg_5_2": (+_FX, -_FY, -0.030),
    "tesollo_right_rl_dg_5_3": (-_FX, -_FY, -0.030),
    "tesollo_right_rl_dg_5_4": (-_FX, -_FY, -0.030),
    "rl_dg_5_tip": (0.0, -0.045, +0.015),
}
# palm_link 자식 구: 우측은 TCP 앞 0.02 였다. 그리퍼는 손 본체를 덮게 뒤로 보낸다.
PALM_SPHERE2_XYZ = (0.0, 0.0, -0.100)
# dg 링크에 달린 보조 구(*_sphere2): 핑거를 따라 손 쪽으로 15 mm
DG_SPHERE2_XYZ = (0.0, 0.0, -0.015)

PALM_HELPER_JOINTS = {
    "palm_x_joint", "palm_x_neg_joint",
    "palm_y_joint", "palm_y_neg_joint",
    "palm_z_joint", "palm_z_neg_joint",
}


def fmt(v) -> str:
    return " ".join(f"{x:.9g}" if abs(x) > 1e-12 else "0" for x in v)


def set_origin(joint: ET.Element, xyz, rpy=(0.0, 0.0, 0.0)) -> None:
    o = joint.find("origin")
    if o is None:
        o = ET.SubElement(joint, "origin")
    o.set("xyz", fmt(xyz))
    o.set("rpy", fmt(rpy))


def freeze(joint: ET.Element) -> None:
    """revolute → fixed. axis/limit 제거(남으면 파서가 관절로 오인할 여지)."""
    joint.set("type", "fixed")
    for tag in ("axis", "limit", "dynamics", "safety_controller", "mimic"):
        el = joint.find(tag)
        if el is not None:
            joint.remove(el)


def build(sensor_joints: dict) -> None:
    tree = ET.parse(RIGHT_FABRIC_URDF)
    root = tree.getroot()
    root.set("name", ROBOT_NAME)

    joints = {j.get("name"): j for j in root.iter("joint")}
    links = {l.get("name"): l for l in root.iter("link")}

    # ── 1. 팔 7관절: sensor_rl 좌팔 실값 ────────────────────────────
    # joint1 origin = body_root → l_al_1 합성(q=0). joint2~7 은 l_aj_i origin 그대로.
    T1 = fk_link(sensor_joints, "l_al_1", {})
    set_origin(joints["openarm_right_joint1"], T1[:3, 3], mat_to_rpy(T1[:3, :3]))
    for i in range(1, 8):
        src = sensor_joints[f"l_aj_{i}"]
        dst = joints[f"openarm_right_joint{i}"]
        if i > 1:
            set_origin(dst, src["xyz"], src["rpy"])
        dst.find("axis").set("xyz", fmt(src["axis"]))
        lim = dst.find("limit")
        lim.set("lower", f"{src['limits'][0]:.9g}")
        lim.set("upper", f"{src['limits'][1]:.9g}")

    # ── 2. palm_link = 그리퍼 TCP (회전 없음: 그리퍼 고유축 그대로) ──
    set_origin(joints["palm_link_joint"], (0.0, 0.0, PALM_Z_IN_LINK7), (0.0, 0.0, 0.0))
    set_origin(joints["palm_link_joint_sphere2"], PALM_SPHERE2_XYZ)

    # ── 3. 손 관절 동결 + 프레임 재배치 ──────────────────────────────
    by_child = {j.find("child").get("link"): j for j in root.iter("joint")}
    for link_name, target in DG_FRAME_TARGETS.items():
        j = by_child[link_name]
        freeze(j)
        parent = j.find("parent").get("link")
        p_target = np.zeros(3) if parent == "palm_link" else np.asarray(DG_FRAME_TARGETS[parent])
        set_origin(j, np.asarray(target) - p_target)
    for name, j in joints.items():
        if name.endswith("_joint_sphere2") and "rl_dg_" in name:
            set_origin(j, DG_SPHERE2_XYZ)

    # ── 4. dg 링크의 DG-5F 메시 제거 (재배치로 위치가 무의미해졌다) ──
    for link_name in DG_FRAME_TARGETS:
        link = links.get(link_name)
        if link is None:
            continue
        for tag in ("visual", "collision"):
            for el in list(link.findall(tag)):
                link.remove(el)

    # ── 5. helper 프레임은 불변 (코드 규약) ─────────────────────────
    for name in PALM_HELPER_JOINTS:
        assert joints[name].find("origin") is not None, name

    # ── 6. 메시 경로: 우측 폴더 상대참조 ────────────────────────────
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn and fn.startswith("meshes/"):
            mesh.set("filename", f"../openarm_tesollo/{fn}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = (
        f"<!-- generated by scripts/assets_tools/{Path(__file__).name} -->\n"
        "<!-- openarm_tesollo_sensor_rl 좌팔 7 DOF + 2지 그리퍼 fabric URDF. -->\n"
        "<!-- 손 20관절은 fixed 로 동결(cspace = 팔 7 DOF), 그리퍼는 RL 액션이 직접 제어. -->\n"
        "<!-- 링크/조인트 이름은 우측(openarm_tesollo)과 동일 유지 — fabric_params 프레임 리스트 재사용. -->\n"
    )
    OUT_URDF.write_text(
        '<?xml version="1.0" ?>\n' + header + ET.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )
    print(f"생성: {OUT_URDF.relative_to(HDGP)}")


def validate(sensor_joints: dict, seed: int = 11) -> float:
    """생성 URDF palm_link FK vs sensor_rl l_hl_gripper_tcp FK."""
    gj = parse_urdf(OUT_URDF)
    rng = np.random.default_rng(seed)
    worst = 0.0
    worst_rot = 0.0
    for _ in range(20):
        qg, qs = {}, {}
        for i in range(1, 8):
            lo, hi = gj[f"openarm_right_joint{i}"]["limits"]
            v = float(rng.uniform(lo, hi))
            qg[f"openarm_right_joint{i}"] = v
            qs[f"l_aj_{i}"] = v
        Tg = fk_link(gj, "palm_link", qg)
        Ts = fk_link(sensor_joints, "l_hl_gripper_tcp", qs)
        worst = max(worst, float(np.linalg.norm(Tg[:3, 3] - Ts[:3, 3])))
        worst_rot = max(worst_rot, float(np.abs(Tg[:3, :3] - Ts[:3, :3]).max()))
    print(f"검증: palm_link vs l_hl_gripper_tcp — 위치 최대오차 {worst*1e6:.3f} µm, "
          f"회전 최대성분오차 {worst_rot:.3e}")
    return max(worst, worst_rot)


def report_cspace() -> None:
    gj = parse_urdf(OUT_URDF)
    rev = [n for n, j in gj.items() if j["type"] == "revolute"]
    print(f"cspace: revolute {len(rev)}개 = {rev}")
    assert len(rev) == 7, f"팔 7 DOF 여야 한다 (실제 {len(rev)})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="생성 없이 기존 URDF 만 검증")
    args = ap.parse_args()

    for f in (RIGHT_FABRIC_URDF, SENSOR_URDF):
        if not f.is_file():
            print(f"[FAIL] 없음: {f}")
            return 1
    sensor_joints = parse_urdf(SENSOR_URDF)

    if not args.check_only:
        build(sensor_joints)
    if not OUT_URDF.is_file():
        print(f"[FAIL] 생성물 없음: {OUT_URDF}")
        return 1

    report_cspace()
    err = validate(sensor_joints)
    # 허용오차 1 µm: URDF 는 origin 을 9 유효숫자 텍스트로 쓰므로 왕복 오차가 ~1e-9 남는다.
    # 이는 생성 로직 오류가 아니라 직렬화 정밀도이고, 물리적으로 무의미한 크기다.
    if err > 1e-6:
        print(f"[FAIL] FK 불일치 {err:.3e} — 생성 로직 점검 필요")
        return 1
    print("[PASS] 좌팔 그리퍼 fabric URDF 가 sensor_rl 좌팔과 FK 일치")
    return 0


if __name__ == "__main__":
    sys.exit(main())
