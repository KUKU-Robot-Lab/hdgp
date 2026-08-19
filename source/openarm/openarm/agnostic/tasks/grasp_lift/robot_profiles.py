"""로봇 프로필 — robot-agnostic grasp-lift 태스크에서 **로봇 종속 정보가 모이는 유일한 곳**.

설계 목표(2026-08-20 사용자): 잘 설계된 보상함수와 환경 세팅만으로, assets/robot 의
어떤 로봇을 소환해도 태스크가 성공해야 한다. 태스크 코드(env/reward/curriculum)는
이 프로필의 필드만 참조하고, 조인트/바디 **이름**을 하드코딩하지 않는다.

새 로봇 추가 = 이 파일에 프로필 1개 추가가 전부여야 한다(합격 조건).

의도적으로 isaaclab 을 import 하지 않는다(순수 데이터) — 테스트가 Isaac 앱 없이
프로필 계약을 검증할 수 있어야 한다. ArticulationCfg 조립은 env_cfg 쪽에서 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RobotProfile:
    name: str
    # assets/robot/ 기준 상대 경로
    usd_relpath: str

    # 제어 차원(공간 크기 계산용 — env 부팅 시 regex 해석 결과와 대조해 fail-loud 검증)
    num_arm_joints: int = 7
    num_hand_joints: int = 0

    # ---- 제어 대상 (regex 는 Articulation.find_joints 로 해석) ----------------
    arm_joint_regex: str = ""
    hand_joint_regex: str = ""

    # ---- diff IK ---------------------------------------------------------------
    palm_body: str = ""                  # IK 가 추종하는 EE body
    tcp_offset_pos: tuple = (0.0, 0.0, 0.0)

    # ---- 접촉 (보상의 대향 게이트) ----------------------------------------------
    # finger 이름 → body 이름 튜플. body 마다 ContactSensor 를 **개별** 생성해
    # 코드에서 합산한다 — 다중 body 단일 센서는 force_matrix_w 가 0 을 반환한다
    # (grasp_sensor 실측 함정, env 주석 참조).
    finger_sensor_bodies: dict = field(default_factory=dict)
    # 대향 그룹: A(엄지/조1) AND B(나머지/조2) 동시 접촉 = 파지 성립.
    # dexsuite 의 "thumb AND (index|middle|ring)" 게이트의 일반화 — 2지 그리퍼는
    # A=jaw1, B=jaw2 로 같은 코드가 동작한다.
    contact_group_a: tuple = ()
    contact_group_b: tuple = ()

    # ---- 관측/보상용 손끝 body (reaching 은 max 거리 = 전 손가락 유도) -----------
    fingertip_bodies: tuple = ()

    # ---- 초기 상태 / 액추에이터 -------------------------------------------------
    init_joint_pos: dict = field(default_factory=dict)
    # 그룹명 → ImplicitActuatorCfg kwargs (env_cfg 가 조립). 전 DOF 커버 필수 —
    # 커버리지 누락 관절은 조용히 무구동 자유회전한다(adf0b24 교훈).
    actuator_specs: dict = field(default_factory=dict)

    # ---- 씬 배치 (로봇 서 있는 쪽에 따라 다름) -----------------------------------
    object_spawn_center: tuple = (0.30, -0.20)   # env-local (x, y)
    object_spawn_z: float = 0.297                # 테이블 위 안착 높이(cup_big)


# =============================================================================
# tesollo_right — a1 s2r 자산(openarm_tesollo_sensor_rl): DG-5F 우손 20 DOF
# 게인/effort 근거: grasp_sensor 실측 캘리브 승계 —
#   팔 400/80 + friction(0.213/0.493/0.151, real2sim 07.29)
#   손 k5/kd2(08.16 S1~S4 스윕: 구 400/60 은 토크 포화 레짐) + effort 1.5 N·m
#   (08.19 A4: 7.5 레짐은 전 관절 3~5 N·m 상시 + thumb_1 하드스톱 밖 -0.94rad).
# ★구 태스크와 달리 손 20관절 전부 정책 제어(고정관절 없음) — thumb_1 을 약한 PD 로
#   0 에 고정해 back-drive 당하던 구조 자체를 없앤다(dexsuite 방식).
# =============================================================================
_FINGERS = ("thumb", "index", "middle", "ring", "pinky")

TESOLLO_RIGHT = RobotProfile(
    name="tesollo_right",
    usd_relpath="robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd",
    num_arm_joints=7,
    num_hand_joints=20,
    arm_joint_regex="r_aj_[1-7]",
    hand_joint_regex="r_hj_(thumb|index|middle|ring|pinky)_[1-4]",
    palm_body="r_hl_palm",
    # 원위마디(_4)와 센서팁 둘 다 — 인벨롭(마디 접촉)과 핀치(팁 접촉) 어느 쪽이든
    # 접촉으로 인정(tesollo 팁은 인벨롭에서 안 닿기 쉬움, 07.29 교훈).
    finger_sensor_bodies={f: (f"r_hl_{f}_4", f"r_hl_{f}_tip") for f in _FINGERS},
    contact_group_a=("thumb",),
    contact_group_b=("index", "middle", "ring", "pinky"),
    fingertip_bodies=tuple(f"r_hl_{f}_tip" for f in _FINGERS),
    init_joint_pos={
        # 팔: 컵 쪽을 향한 자연스러운 준비 자세(grasp_sensor init 승계)
        "r_aj_1": 0.5, "r_aj_2": 0.1, "r_aj_3": 0.4, "r_aj_4": 0.60,
        "r_aj_5": -0.2, "r_aj_6": 0.0, "r_aj_7": 0.0,
        # 손: 엄지 대향 + 나머지 폄
        "r_hj_thumb_1": 0.0, "r_hj_thumb_2": -1.57, "r_hj_thumb_3": -0.5, "r_hj_thumb_4": 0.0,
        **{f"r_hj_{f}_{j}": 0.0 for f in ("index", "middle", "ring", "pinky") for j in (1, 2, 3, 4)},
        # 유휴 좌팔(파지 팔 홈의 부호 미러, DG-5F IK 실측 — grasp_sensor preset 승계)
        "l_aj_1": -0.0431, "l_aj_2": -0.6706, "l_aj_3": -0.0961, "l_aj_4": 0.7342,
        "l_aj_5": -0.3750, "l_aj_6": -0.5678, "l_aj_7": -0.6709,
        "l_hj_gripper_1": 0.044, "l_hj_gripper_2": 0.044,
        "head_j_pan": 0.0, "head_j_tilt": 0.0,
    },
    actuator_specs={
        "right_arm_proximal": dict(joint_names_expr=["r_aj_[1-3]"], stiffness=400.0, damping=80.0, friction=0.213),
        "right_arm_elbow":    dict(joint_names_expr=["r_aj_4"],     stiffness=400.0, damping=80.0, friction=0.493),
        "right_arm_wrist":    dict(joint_names_expr=["r_aj_[5-7]"], stiffness=400.0, damping=80.0, friction=0.151),
        "hand":               dict(joint_names_expr=["r_hj_[a-z]+_[1-4]"], stiffness=5.0, damping=2.0,
                                   effort_limit_sim=1.5),
        "left_arm":           dict(joint_names_expr=["l_aj_[1-7]"], stiffness=400.0, damping=80.0),
        "left_gripper":       dict(joint_names_expr=["l_hj_gripper_[1-2]"], stiffness=400.0, damping=80.0),
        "head":               dict(joint_names_expr=["head_j_(pan|tilt)"], stiffness=400.0, damping=80.0),
    },
    object_spawn_center=(0.30, -0.20),
    object_spawn_z=0.297,
)


# =============================================================================
# gripper_left — 같은 자산의 좌팔 2-DOF 평행 그리퍼. agnosticism 검증용(Phase 2):
# 이 프로필 추가 외에 태스크 코드 수정이 0 이어야 합격.
# 대향 그룹 = jaw1 / jaw2. l_hj_gripper_2 는 USD PhysX mimic(gearing=-1).
# =============================================================================
GRIPPER_LEFT = RobotProfile(
    name="gripper_left",
    usd_relpath="robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd",
    num_arm_joints=7,
    num_hand_joints=1,
    arm_joint_regex="l_aj_[1-7]",
    hand_joint_regex="l_hj_gripper_1",   # mimic(_2)은 제어 대상에서 제외
    palm_body="l_hl_gripper_base" ,      # Phase 2 에서 실제 body 이름 검증 후 확정
    finger_sensor_bodies={
        "jaw1": ("l_hl_gripper_1",),
        "jaw2": ("l_hl_gripper_2",),
    },
    contact_group_a=("jaw1",),
    contact_group_b=("jaw2",),
    fingertip_bodies=("l_hl_gripper_1", "l_hl_gripper_2"),
    init_joint_pos={
        "l_aj_1": 0.0431, "l_aj_2": 0.6706, "l_aj_3": 0.0961, "l_aj_4": 0.7342,
        "l_aj_5": 0.3750, "l_aj_6": 0.5678, "l_aj_7": 0.6709,
        "l_hj_gripper_1": 0.044, "l_hj_gripper_2": 0.044,
        # 유휴 우팔+손
        "r_aj_1": -0.0431, "r_aj_2": -0.6706, "r_aj_3": -0.0961, "r_aj_4": 0.7342,
        "r_aj_5": -0.3750, "r_aj_6": -0.5678, "r_aj_7": -0.6709,
        "r_hj_thumb_2": -1.57,
        **{f"r_hj_{f}_{j}": 0.0 for f in _FINGERS for j in (1, 2, 3, 4) if not (f == "thumb" and j == 2)},
        "head_j_pan": 0.0, "head_j_tilt": 0.0,
    },
    actuator_specs={
        "left_arm_proximal": dict(joint_names_expr=["l_aj_[1-3]"], stiffness=400.0, damping=80.0, friction=0.213),
        "left_arm_elbow":    dict(joint_names_expr=["l_aj_4"],     stiffness=400.0, damping=80.0, friction=0.493),
        "left_arm_wrist":    dict(joint_names_expr=["l_aj_[5-7]"], stiffness=400.0, damping=80.0, friction=0.151),
        "left_gripper":      dict(joint_names_expr=["l_hj_gripper_[1-2]"], stiffness=400.0, damping=80.0),
        "right_arm":         dict(joint_names_expr=["r_aj_[1-7]"], stiffness=400.0, damping=80.0),
        "right_hand":        dict(joint_names_expr=["r_hj_[a-z]+_[1-4]"], stiffness=5.0, damping=2.0,
                                  effort_limit_sim=1.5),
        "head":              dict(joint_names_expr=["head_j_(pan|tilt)"], stiffness=400.0, damping=80.0),
    },
    object_spawn_center=(0.30, 0.20),    # 좌측 미러
    object_spawn_z=0.297,
)


PROFILES: dict[str, RobotProfile] = {
    p.name: p for p in (TESOLLO_RIGHT, GRIPPER_LEFT)
}
