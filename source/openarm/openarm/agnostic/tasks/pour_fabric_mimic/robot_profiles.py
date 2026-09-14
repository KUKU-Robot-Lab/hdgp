"""pour_fabric_mimic 로봇 프로필 — RH56F1 양손 **Fabrics 판**.

`modules/robot_profiles.RH56F1_RIGHT` 는 Track B(`grasp_fj_rh`) 전용이라 `fabric_class=None`
이다(그 파일 주석). 이 트랙은 Fabrics 팔이 필요하므로 **여기서** fabric 을 켠 변형을 만든다.
모듈 레지스트리(`PROFILES`)에는 넣지 않는다 — grasp 계열 config 가 그 dict 를 훑어 gym id 를
찍어 내므로, 거기 추가하면 이 트랙과 무관한 등록이 생긴다.

★Fabrics: `OpenArmRh56f1PoseFabric`(우) / `OpenArmRh56f1LeftPoseFabric`(좌) — 둘 다 **양팔 26 DOF**
  cspace [r_arm7, r_hand6, l_arm7, l_hand6] 하나를 쓰고 side 로 palm 제어점만 바꾼다
  (`FABRICS/src/fabrics_sim/fabrics/openarm_rh56f1_pose_fabric.py`). 그래서 `fabric_joint_order`
  는 26개 전부이고, 팔 하나(SideRig)는 자기 슬라이스만 읽고 쓴다(`bimanual.fabric_slots`).

★좌손 미러(09.14 FK 실측, `assets/robot/openarm_rh56f1_bi_rl/*.urdf` 순수 FK):
  팔 부호 (−1,−1,−1,+1,−1,−1,−1) · 손 6관절 **전부 +1**(좌 URDF 가 엄지 축을 이미 뒤집어 두었고
  4지는 축·한계가 같다). 같은 q 에서 4지 손끝 |l − mirror(r)| = 0.00 mm, 엄지 5.7 mm ·
  palm_sensor 2.7 mm 는 벤더 자산의 좌우 비대칭(엄지 마운트 rpy)이지 부호 문제가 아니다
  (부호가 틀리면 수 cm 가 난다).
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace

from openarm.agnostic.modules import robot_profiles as _rp

_RH_FLEX = ("index", "middle", "ring", "pinky")
_ARM_SIGN_L = (-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0)

# 26 DOF fabric cspace 순서(fabric 클래스 docstring 그대로).
_FAB_HAND = ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "pinky_1")
FABRIC_JOINT_ORDER: tuple[str, ...] = (
    *tuple(f"r_aj_{i}" for i in range(1, 8)),
    *tuple(f"r_hj_{n}" for n in _FAB_HAND),
    *tuple(f"l_aj_{i}" for i in range(1, 8)),
    *tuple(f"l_hj_{n}" for n in _FAB_HAND),
)

_BASE = _rp.RH56F1_RIGHT

RH56F1_RIGHT_FAB = _dc_replace(
    _BASE,
    name="rh56f1_right_fab",
    fabric_class="OpenArmRh56f1PoseFabric",
    fabric_robot_dir="openarm_rh56f1",
    fabric_params_filename=None,            # 클래스가 자기 params 를 고정 로드한다
    fabric_joint_order=FABRIC_JOINT_ORDER,
    # ★palm 박스(Track B 값 x≥0.20·z≤0.65)는 붓기 델타 박스(x −0.15 / z +0.25)를 자른다(09.14 부팅 실측
    #   "잘리는 축 ['x','z']"). 붓기는 컵을 들어 올려야 하므로 z 상한 0.70, x 하한 0.10 으로 연다.
    #   z 하한 0.26 은 손 최하단−테이블 실측(09.02)이라 유지. palm_box_verified=False — probe 후 승격.
    palm_box_min=(0.10, _BASE.palm_box_min[1], _BASE.palm_box_min[2]),
    palm_box_max=(_BASE.palm_box_max[0], _BASE.palm_box_max[1], 0.70),
    # ★붓기 트랙 spawn: Track B 값(0.38, −0.16)을 그대로 시작점으로 쓴다 — probe 로 재확인.
)


def _mirror_init(init: dict) -> dict:
    """우 활성 홈 → 좌 활성 홈. 팔은 부호 미러, 손은 그대로(+1). 유휴 우측은 fabric rest."""
    out = dict(init)
    for i, s in enumerate(_ARM_SIGN_L, start=1):
        out[f"l_aj_{i}"] = s * init[f"r_aj_{i}"]
    for n in ("thumb_1", "thumb_2", "thumb_3", "thumb_4",
              *(f"{f}_1" for f in _RH_FLEX), *(f"{f}_2" for f in _RH_FLEX)):
        out[f"l_hj_{n}"] = init[f"r_hj_{n}"]
    # 유휴 우팔 = fabric `_ARM_REST_R`(쌍 합성에서 버려지지만 단독 부팅 정합용).
    rest_r = (0.315, 0.290, -0.400, 0.513, -0.666, 0.729, 0.957)
    for i, v in enumerate(rest_r, start=1):
        out[f"r_aj_{i}"] = v
    for n in ("thumb_1", "thumb_2", "thumb_3", "thumb_4",
              *(f"{f}_1" for f in _RH_FLEX), *(f"{f}_2" for f in _RH_FLEX)):
        out[f"r_hj_{n}"] = 0.0
    return out


def _l(names):
    return tuple(n.replace("r_h", "l_h", 1) for n in names)


RH56F1_LEFT_FAB = _dc_replace(
    RH56F1_RIGHT_FAB,
    name="rh56f1_left_fab",
    arm_joint_regex="l_aj_[1-7]",
    hand_joint_regex="l_hj_(thumb_[12]|index_1|middle_1|ring_1|pinky_1)",
    palm_body="l_hl_palm_sensor",
    fabric_class="OpenArmRh56f1LeftPoseFabric",
    hand_joint_names=_l(_BASE.hand_joint_names),
    finger_sensor_bodies={f: _l(b) for f, b in _BASE.finger_sensor_bodies.items()},
    fingertip_bodies=_l(_BASE.fingertip_bodies),
    # palm 박스: y 미러.
    palm_box_min=(RH56F1_RIGHT_FAB.palm_box_min[0], -RH56F1_RIGHT_FAB.palm_box_max[1], RH56F1_RIGHT_FAB.palm_box_min[2]),
    palm_box_max=(RH56F1_RIGHT_FAB.palm_box_max[0], -RH56F1_RIGHT_FAB.palm_box_min[1], RH56F1_RIGHT_FAB.palm_box_max[2]),
    palm_box_verified=False,
    init_joint_pos=_mirror_init(_BASE.init_joint_pos),
    object_spawn_center=(_BASE.object_spawn_center[0], -_BASE.object_spawn_center[1]),
)

PROFILES: dict[str, _rp.RobotProfile] = {p.name: p for p in (RH56F1_RIGHT_FAB, RH56F1_LEFT_FAB)}
