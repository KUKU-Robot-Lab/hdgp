"""로봇 프로필 — robot-agnostic grasp-sensor 태스크에서 **로봇 종속 정보가 모이는 유일한 곳**.

설계 목표(2026-08-20 사용자): 잘 설계된 보상함수와 환경 세팅만으로, assets/robot 의
어떤 로봇을 소환해도 태스크가 성공해야 한다. 태스크 코드(env/reward/curriculum)는
이 프로필의 필드만 참조하고, 조인트/바디 **이름**을 하드코딩하지 않는다.

새 로봇 추가 = 이 파일에 프로필 1개 추가가 전부여야 한다(합격 조건).

의도적으로 isaaclab 을 import 하지 않는다(순수 데이터) — 테스트가 Isaac 앱 없이
프로필 계약을 검증할 수 있어야 한다. ArticulationCfg 조립은 env_cfg 쪽에서 한다.
"""

from __future__ import annotations

import os as _os
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
    # 손 관절 중 **정책이 건드리지 않고 홈 값으로 고정**할 것 (PD 가 잡고 있는다).
    # grasp_v2 방식 — 외전(_1)을 자유화하면 자기충돌이 꺼진 상태에서 손가락이
    # 서로 벌어져 겹치고 파지 평면이 무너진다(08.20 사용자 지시).
    # 빈 문자열이면 전 손관절이 정책 제어.
    hand_locked_joint_regex: str = ""
    num_locked_hand_joints: int = 0      # 공간 계산용(regex 해석 결과와 대조 검증)

    # ---- 팔 제어: Fabrics ---------------------------------------------------------
    palm_body: str = ""                  # Fabrics palm attractor 가 추종하는 EE body
    # fabrics_sim 클래스 **이름**(문자열). None 이면 이 태스크로 못 띄운다(fail-loud).
    fabric_class: str | None = None
    # FABRICS/models/robots/urdf/<dir>/<dir>.urdf — robot_dir_name·robot_name 양쪽에 쓰인다.
    fabric_robot_dir: str | None = None
    # FABRICS/fabric_params/<file> — None 이면 fabric 클래스 기본값.
    # 충돌 구 목록이 URDF 마다 다르면(마디 길이로 개수가 정해진다) 전용 파일이 필요하다.
    fabric_params_filename: str | None = None
    # ★articulation 은 depth-major(index_1, middle_1, …), fabric URDF 는 finger-major
    #   (thumb_1..4, index_1..4, …) 다. cat([arm, hand]) 로 만들면 손 20관절이 통째로
    #   어긋나 fabric 이 없는 자기충돌을 피하려 팔을 민다(병행 트랙 실측: palm 이 2초에
    #   61mm 만 움직이면서 관절속도 20 rad/s 포화). 이 순서가 유일한 방어선이다.
    fabric_joint_order: tuple = ()
    # ★fabric cspace 가 **양팔 전체**일 때 쓰는 슬롯(RH56F1 fabric 은 26 DOF 양팔이다).
    #   비우면 [0:num_arm] = 팔, [num_arm:num_arm+num_hand] = 손 이라는 단일팔 규약을
    #   그대로 쓴다(tesollo 항등). 비어 있지 않으면 이 인덱스가 유일한 진실이다 —
    #   `fabric_joint_order` 는 cspace **전체**를 덮어야 하고(길이 == fabric.num_joints),
    #   슬롯은 그중 이 태스크가 제어하는 팔/손을 가리킨다.
    fabric_arm_slot: tuple = ()
    fabric_hand_slot: tuple = ()
    # fabric 클래스 생성자에 **추가로** 넘길 인자(로봇별). 예: RH56F1 은 양팔 클래스라
    # `{"side": "right"}` 로 어느 팔의 palm IK 를 살릴지 정한다.
    # ★생성자 시그니처는 클래스마다 다르다 — 태스크가 넘기는 공용 인자 중 그 클래스가
    #   안 받는 것은 부팅에서 걸러지고, **중립값이 아니면 죽는다**(조용한 기능 상실 차단).
    fabric_kwargs: dict = field(default_factory=dict)
    # palm 목표 워크스페이스(env-local 절대). 액션 누산 결과를 여기로 clamp 한다.
    palm_box_min: tuple = (0.0, 0.0, 0.0)
    palm_box_max: tuple = (0.0, 0.0, 0.0)
    palm_rot_center_deg: tuple = (90.0, 0.0, 90.0)   # euler_zyx (ez, ey, ex)
    palm_rot_half_deg: float = 45.0
    palm_box_verified: bool = False      # probe 로 도달성 확인했는가
    # ★★palm_body 국소축 → 태스크 표준축 회전(3x3, 행 우선). 비우면 항등.
    #   태스크 표준: **열0 = 손바닥 법선**, 열1 = 손바닥면 가로, 열2 = 손가락 방향.
    #   보상의 `palm_normal_dist`/`palm_lateral_dist` 와 z 데드밴드가 이 규약에 묶여
    #   있어서(`_palm_ee_R()` 열 0 이 법선), 프레임이 다른 로봇을 그냥 꽂으면 법선과
    #   가로축이 바뀐 채 **조용히** 학습된다.
    #   ★`palm_body` 자체는 고를 수 없다 — fabric 제어점과 같아야 부팅 FK 게이트
    #     (5mm/2°)를 통과한다. 그래서 프레임 차이는 여기서 흡수한다.
    #   RH56F1 실측: `r_hl_palm_sensor` 는 base 대비 rpy(90°,0,90°) 라 법선이 국소 +z 다.
    palm_frame_remap: tuple = ()

    # ---- 시너지 그립 (hand_control="synergy") -----------------------------------
    # ★손끝 IK(tip_cyl)가 파워그립을 **만들 수 없음**이 실측으로 확정돼(08.25) 도입한
    #   관절공간 경로. r 을 86→14mm 로 전 범위 훑어도 검지 MCP 가 0.03→0.18 rad 에
    #   그쳤다(파워그립 기준 1.90 의 1/10). 관절 목표를 직접 보간하면 말아 쥐는 것이
    #   구조적으로 보장된다 — grasp_v1(단일컵 98%)의 검증된 방식.
    # 두 자세는 **손 관절 이름 순서**(`hand_joint_names`)에 대응하는 값 목록이다.
    #   articulation 순서와 다를 수 있으므로 **슬라이스 금지, 이름으로 매핑**한다.
    hand_joint_names: tuple = ()
    hand_open_pose: tuple = ()        # 폐쇄도 0 (접근 자세)
    hand_grip_pose: tuple = ()        # 폐쇄도 1 (완전 파지) — 관절한계 초과분은 런타임 clamp
    # 폐쇄 채널 → 관절 대응. 관절 이름 접미사별로 어느 채널이 그 관절을 몰지 지정한다.
    #   예 tesollo: {"1": 0, "2": 1, "3": 2, "4": 2} = [외전, MCP, PIP·DIP 공통]
    #   ★채널을 나누는 이유: 손가락당 스칼라 하나를 4관절에 복사하면 관절 목표가
    #     open→grip 직선 하나 위에만 존재해 **진짜 인벨롭 자세가 액션 공간에 없다**.
    hand_channel_of_joint: dict = field(default_factory=dict)
    # ---- 손가락별 레이아웃 (cfg.hand_layout="per_finger", 08.29 O 라운드) ---------
    # finger 이름 → {관절 접미사: **전역 액션 슬롯**}. 빠진 접미사는 **고정**(폐쇄도 0).
    # ★사용자 확정 설계: 엄지 2슬롯(근위/원위 분리) · 검/중/약 각 1슬롯(j2/3/4 공통)
    #   · 소지 1슬롯(j3/4 — j2 는 실측 가동폭 0 이라 자연 고정) · 외전(_1) 전부 고정.
    #   L1 의 완전 해제(12ch)와 달리 손 차원 15→6 **축소** 재편이다.
    hand_finger_channels: dict = field(default_factory=dict)
    # 접촉 시 동결할 관절 접미사 — 그 손가락의 원위∨팁 접촉이 성립하면 진행을 멈춘다.
    #   ★이것이 감쌈 생성 메커니즘이다. 풀면 손가락이 컵 반경보다 작게 말려 손끝만
    #     닿는 핀치가 된다(grasp_v1 실증: full_envelope 0.176→0.035).
    hand_freeze_suffixes: tuple = ()
    # 그중 **중간마디**에 해당하는 접미사 — 중간마디 접촉으로 얼리는 관절을 고른다.
    #   ★09.02 이전에는 `"3"` 이 control 코드에 **리터럴로 박혀** 있었다. tesollo 는
    #     손가락이 `_1.._4` 4마디라 `_3` 이 중간이지만, RH56F1 은 구동관절이 손가락당
    #     하나(`_1`)뿐이라 그 리터럴이 아무 관절에도 안 걸린다 — 즉 중간마디 동결이
    #     조용히 전부 꺼진다. 로봇 사실이므로 프로필이 들고 있어야 한다.
    #   빈 값이면 `hand_freeze_suffixes` 전부를 원위로 본다(마디가 하나뿐인 손).
    hand_freeze_mid_suffixes: tuple = ()

    # ---- 언더액추에이션(mimic) -------------------------------------------------
    # 종속관절 이름 → (구동관절 이름, 배율, 오프셋). URDF `<mimic>` 을 그대로 옮긴다.
    # ★왜 필요한가: RH56F1 은 물리 12관절 중 6개만 구동이고 나머지는 하드웨어로 결합돼
    #   있다. 그런데 **USD 에 PhysxMimicJoint 가 없으면** 종속관절은 지령을 못 받아
    #   init 각도에 굳은 강체 막대가 된다 — 실기의 결합 굴곡과 다른 손을 학습한다.
    #   (선행 rh56f1 트랙이 강성 400/60 을 줘서 "흐물"만 막아둔 상태였다.)
    # 체인 허용: `thumb_4 → thumb_3 → thumb_2` 처럼 종속관절이 다른 종속관절을 물면
    #   부팅에서 구동관절까지 전개해 (배율·오프셋)을 합성한다.
    hand_mimic: dict = field(default_factory=dict)
    # "software" = 우리가 매 스텝 종속목표를 계산해 써넣는다(USD 에 mimic 이 없을 때).
    # "physx"    = USD PhysxMimicJoint 가 처리하므로 **건드리지 않는다**.
    # "none"     = 결합 없음(tesollo 처럼 전 관절 독립).
    hand_mimic_mode: str = "none"

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
    # ---- 인벨롭 손가락 (감쌈 판정·d_side 분모) ----------------------------------
    # envelope_frac 의 분모와 d_side 의 wrap 그룹 평균에 들어가는 손가락만.
    # ★tesollo pinky 는 제대로 된 굴곡축이 없어(pinky_1 손끝이동 12mm vs index_2 42mm,
    #   메모리 tesollo-pinky-joint-kinematics) 분모에 넣으면 상한이 0.8 로 깎인다 — 제외.
    envelope_fingers: tuple = ()
    # ---- 손바닥면 법선 (감쌈이 **손바닥 접촉인지** 판정) -------------------------
    # finger 이름 → 그 손가락 마디 링크의 국소 좌표에서 손바닥이 향하는 축.
    # ★필요한 이유(08.23 실측): 접촉센서는 링크에 붙어 있어 손바닥면이든 **손등**이든
    #   똑같이 힘을 낸다. lstm_test3 ep5000 에서 middle_4 는 접촉 시간의 **100%** 를
    #   손등으로 접촉했는데 envelope_frac 은 그걸 감쌈으로 셌다(정직한 값 0.55 vs
    #   계상값 0.746, 성공 임계 0.6 을 허수로 통과). 손등 파지는 force-closure 가
    #   아니라서 pour 의 손목 회전에서 그대로 빠진다.
    # 유도법: URDF 에서 그 마디를 움직이는 관절의 **굴곡축**과 **장축**(자식 관절
    #   origin 방향)을 읽어 cross(굴곡축, 장축) — 굴곡이 향하는 쪽이 손바닥이다.
    #   tesollo 실측: 네 손가락 굴곡축 국소+y·장축+z → 손바닥 +x. 엄지만 축이 달라 +z.
    palmar_axis_local: dict = field(default_factory=dict)

    # ---- 관측/보상용 손끝 body (reaching 은 max 거리 = 전 손가락 유도) -----------
    fingertip_bodies: tuple = ()

    # ---- 초기 상태 / 액추에이터 -------------------------------------------------
    init_joint_pos: dict = field(default_factory=dict)
    # 그룹명 → ImplicitActuatorCfg kwargs (env_cfg 가 조립). 전 DOF 커버 필수 —
    # 커버리지 누락 관절은 조용히 무구동 자유회전한다(adf0b24 교훈).
    actuator_specs: dict = field(default_factory=dict)

    # ---- 씬 배치 (로봇 서 있는 쪽에 따라 다름) -----------------------------------
    object_spawn_center: tuple = (0.30, -0.20)   # env-local (x, y)
    # ★object_spawn_z 는 여기 없다 — 높이는 cfg(table_surface_z + origin_offset + pad)
    #   한 곳에서만 파생한다. 프로필이 완성값을 들고 env 가 패딩을 또 더해 9.7mm
    #   어긋났던 이중 패딩(08.21)의 구조적 재발 차단.


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
    # ★★08.25 `_armhull`(팔 23 hull + 손 27 decomposition) → `_hull`(**50개 전부 hull**).
    #   사용자 결정: DEXTRAH 는 전면 convexHull + 효율 위주 물리로 teacher→student 를
    #   끝까지 성공시켰고 우리도 distillation 까지 가야 한다. 실제로 Kuka-Allegro 자산은
    #   콜라이더 26개가 **손 17개 포함 전부 convexHull** 이다(SDF 0개).
    #   자산은 physics 레이어만 교체한 얇은 변형(44KB, base 는 원본 108MB 심볼릭 링크).
    #   ★★반대 실측이 하나 있다 — 되돌릴 때 근거가 되므로 지우지 않는다.
    #     arm5080 A/B(08.23): 팔만 hull = 처리량 +13.7%, 접촉력 36.2→32.8N(편차 안).
    #     그러나 **손까지 hull 로 하면 접촉력 133N** 으로 4배 뛰었다.
    #     재현되면 촉각 obs 가 죽는다: env 가 `contact = (force/5).clamp(max=4)` 라
    #     20N 에서 포화하므로 133N 이면 5채널이 전부 상수 4.0 이 되고, 감쌈 판정 임계
    #     `stage_contact_threshold=0.1N` 도 모든 접촉에서 참이 된다.
    #   ★학습 로그에서 볼 것: `task/contact_*` 포화 · `task/wrap4` 가 1.0 에 붙는지 ·
    #     `task/deep4` 와의 괴리. 붙으면 이 줄을 `_armhull` 로 되돌린다(1줄).
    usd_relpath="robot/openarm_tesollo_sensor_rl_hull/openarm_tesollo_sensor_rl.usd",
    num_arm_joints=7,
    num_hand_joints=20,
    arm_joint_regex="r_aj_[1-7]",
    hand_joint_regex="r_hj_(thumb|index|middle|ring|pinky)_[1-4]",
    # index/middle/ring 의 _1 = 외전. grasp_v2 도 이 축들을 정책에서 뺐다.
    # ★thumb_1(대향 벌림)과 pinky_1(= Z-flex, 외전 아님 — tesollo pinky 운동학 메모)은
    #   파지에 필수라 자유 유지.
    hand_locked_joint_regex="r_hj_(index|middle|ring)_1",
    num_locked_hand_joints=3,
    palm_body="r_hl_palm",
    # ---- Fabrics (DG-5F 계보) ----
    fabric_class="OpenArmTeoslloPoseFabric",
    # ★FK 게이트 0.0um 로 sensor_rl 에서 재생성한 자산(08.22). 레거시 openarm_tesollo /
    #   openarm_tesollo_sensor 는 같은 DG-5F 손이지만 팔 베이스가 +8mm 어긋나
    #   RL URDF 대비 worst 17.93mm 였다.
    fabric_robot_dir="openarm_tesollo_sensor_right",
    # ★전용 params — 공유 openarm_tesollo_pose_params.yaml 을 쓸 수 없다(08.23).
    #   자매 트랙이 손가락 충돌 구를 실측 형상으로 재배치(반경 9mm·마디 방향)했는데,
    #   구 개수는 **마디 길이 ÷ 지름**으로 자동 산출되고 sensor_rl 자산은 bi_s 와
    #   링크 길이가 달라 39개 vs 52개로 갈린다. 공유 yaml 을 그대로 쓰면 우리 URDF 에
    #   없는 `dg_1_2_sph3` 를 찾다 KeyError 로 부팅이 죽는다(실측). 반대로 공유 yaml 을
    #   덮어쓰면 bi_s 트랙이 깨진다 — 그래서 frames/radii 만 우리 것으로 바꾼 사본을 쓴다.
    #   쌍(collision_link_prefix_pairs)은 **접두사 매칭**이라 구 개수와 무관해 그대로다.
    fabric_params_filename="openarm_tesollo_sensor_pose_params.yaml",
    # 팔 7 + 손 20, **finger-major**(생성기 FINGERS 순서 = thumb,index,middle,ring,pinky)
    fabric_joint_order=(
        tuple(f"r_aj_{i}" for i in range(1, 8))
        + tuple(f"r_hj_{f}_{j}" for f in _FINGERS for j in range(1, 5))
    ),
    # ---- 시너지 그립 (grasp_v1 검증값 이식) --------------------------------------
    # 순서는 아래 hand_joint_names 와 1:1. articulation 은 관절번호-major 라 다르므로
    # env 가 **이름으로** 매핑한다(슬라이스 금지).
    hand_joint_names=tuple(f"r_hj_{f}_{j}" for f in _FINGERS for j in range(1, 5)),
    #                 _1 외전  _2 MCP   _3 PIP  _4 DIP
    hand_open_pose=(
        0.0, -1.57, -0.5, 0.0,    # thumb — _2 는 opposition 으로 고정(양 자세 동일),
        0.0,  0.0,   0.0, 0.0,    #         _3 −0.5 pre-curl(밑마디가 먼저 닿는 것 방지)
        0.0,  0.0,   0.0, 0.0,    # index / middle / ring / pinky 는 완전 개방
        0.0,  0.0,   0.0, 0.0,
        0.0,  0.0,   0.0, 0.0,
    ),
    hand_grip_pose=(
        0.0, -1.57, 1.8, 1.8,     # thumb  — _2 불변(대향 유지)
        0.0,  1.9,  1.8, 1.8,     # index
        0.0,  1.9,  1.8, 1.8,     # middle
        0.0,  1.9,  1.8, 1.8,     # ring
        0.0,  0.0,  1.8, 1.8,     # pinky  — _2(외전)는 안 쓰고 _3 가 curl 역할
    ),
    # ★1.8 은 관절한계(±1.571) 초과 과지령이며 런타임 soft limit 으로 흡수된다 —
    #   목표를 한계에 정확히 두면 PD 가 한계 직전에서 힘을 못 낸다(grasp_v1 규약).
    # ★★grasp_v1 실제 경로와 동일 — 손가락당 **채널 3개** `[ch0, ch1, ch2, ch2]`.
    #   (`grasp_v1/grasp_right_env.py:1063`, `NUM_ACTIONS = 6 + 15 = 21`)
    #   ★08.25 한 번 1채널로 줄였다가 되돌렸다. 근거로 삼은
    #   `finger_action_utils.compute_absolute_finger_targets`(손가락당 스칼라 1개,
    #   `repeat_interleave(4)`)는 **import 만 되고 호출되지 않는 죽은 코드**다.
    #   실제 grasp_v1 은 채널 3개이고, PIP/DIP 분리가 접촉 동결과 한 묶음으로
    #   "닿은 마디부터 순차 정지 = 컵 형상에 드리워짐"을 만든다(그쪽 주석: "PIP/DIP
    #   분리가 의미를 가지려면 절대 폐쇄도 전환이 필수 — 둘은 한 묶음").
    #   액션 = palm 6 + 손가락 5×3 = 21D.
    hand_channel_of_joint={"1": 0, "2": 1, "3": 2, "4": 2},
    hand_finger_channels={
        # 엄지 j2 는 실측 가동폭 0 → "j2/j3·4" 의도를 "j3/j4"(근위/원위 분리)로 구현.
        "thumb": {"3": 0, "4": 1},
        "index": {"2": 2, "3": 2, "4": 2},
        "middle": {"2": 3, "3": 3, "4": 3},
        "ring": {"2": 4, "3": 4, "4": 4},
        "pinky": {"3": 5, "4": 5},
    },
    hand_freeze_suffixes=("3", "4"),
    # ★리터럴 축출 전 코드의 `s == "3"` 과 동일 — tesollo 항등 보존.
    hand_freeze_mid_suffixes=("3",),
    # grasp_sensor 프리셋(같은 DG-5F 자산에서 검증된 palm workspace) 승계.
    # ★modules/robots.py 의 _BOX_R 은 bi_s(DG-5FS) 실측이라 palm 이 54.8mm 달라 못 쓴다.
    palm_box_min=(0.20, -0.55, 0.20),
    palm_box_max=(0.55, 0.22, 0.70),
    palm_rot_center_deg=(90.0, 0.0, 90.0),
    palm_rot_half_deg=45.0,
    # ★★08.25 P-2 도달성 실측 완료(현 제어층 — 절대 매핑 + 속도 피드포워드 1.0).
    #   박스 전체 3×3×3 격자: 27점 중 지령오차 <10mm 는 4점뿐이고 최악 코너는 304mm
    #   벗어난다. 즉 **박스는 워크스페이스보다 크다**. 그러나 이는 결함이 아니다:
    #     · 축별 단조성 54구간 **위반 0건** — gradient 가 죽은 구간이 없다
    #     · 유효 이득은 공칭의 0.37~1.00배로 압축될 뿐 뒤집히지 않는다
    #       (x 65~175 · y 200~363 · z 123~250 mm/unit)
    #   Kuka 원본도 박스(1.68 m³)를 워크스페이스보다 훨씬 크게 잡고 초과분을 fabric
    #   attractor 의 소프트 포화에 맡긴다 — 같은 구조다.
    #   ★정작 중요한 **과제 영역**(컵 스폰 xy=(0.30,−0.20))에서는 거의 이상적이다:
    #     · a_z ≥ −0.5 구간 palm z 오차 0.0~0.2mm
    #     · 파지중심 z 바닥 0.2639 < 컵 원점 0.2823 (여유 18.4mm) — 파지 높이 도달 가능
    #     · 파지중심이 컵 ±20mm 에 드는 구간 a_z ∈ [−1.0, −0.6]
    #     · 바닥 포화로 낭비되는 z 액션 4.8% (z_min 0.20 이 도달 바닥 0.2727 보다 낮음)
    #   z_min 을 0.27 로 올려 4.8% 를 회수하는 안은 **기각** — 박스 중심이 움직여 a=0 의
    #   의미가 바뀌고 그 위에서 잰 상수가 전부 무효가 된다. 이득 대비 비용이 크다.
    #   또한 바닥 근처 이득 압축(19 mm/unit)은 파지 높이에서 **정밀 제어**로 유리하다.
    #   ★x span(0.35)이 y span(0.77)의 절반인 것은 사고가 아니라 팔 도달 한계다
    #     (반경 방향은 리치와 베이스에 양쪽으로 잘리고, y 는 스윕 방향이라 넓다).
    palm_box_verified=True,              # P-2 통과 (probe_boxreach / probe_taskreach)
    # 중간마디(_3)·원위마디(_4)·센서팁 — 감쌈(마디 접촉)과 핀치(팁 접촉) 모두 인정.
    # ★_3 추가(08.22): 직경 72~90mm 컵에 우월한 감쌈 자세가 _4/_tip 만으로는 게이트를
    #   못 켰다. grasp_v1 도 _4 와 _3 두 곳에 센서를 단다. 손가락별 합산이라 obs 차원 불변.
    finger_sensor_bodies={
        f: (f"r_hl_{f}_3", f"r_hl_{f}_4", f"r_hl_{f}_tip") for f in _FINGERS
    },
    contact_group_a=("thumb",),
    contact_group_b=("index", "middle", "ring", "pinky"),
    # ★08.24 pinky 포함으로 복귀(사용자 지시). 구 규약 "pinky 굴곡축 부재 → 분모 제외"는
    #   **관절공간 액션 시절** 근거다. tip_cyl(손가락별 손끝 IK + 원통 (r,z))에서는
    #   probe 실측으로 pinky 도 손바닥면 100% 감쌈했다 — 액션 구조가 바뀌면 도달성도 바뀐다.
    envelope_fingers=("thumb", "index", "middle", "ring", "pinky"),
    # URDF(openarm_tesollo_sensor_right) 실측 유도 — 필드 주석의 cross(굴곡축, 장축):
    #   네 손가락 rj_dg_{2..5}_{3,4} 축 (0,1,0) · 장축 (0,0,1) → 손바닥 (1,0,0)
    #   엄지     rj_dg_1_{3,4}      축 (1,0,0) · 장축 (0,1,0) → 손바닥 (0,0,1)
    palmar_axis_local={
        "thumb": (0.0, 0.0, 1.0),
        **{f: (1.0, 0.0, 0.0) for f in ("index", "middle", "ring", "pinky")},
    },
    fingertip_bodies=tuple(f"r_hl_{f}_tip" for f in _FINGERS),
    init_joint_pos={
        # 팔: grasp_v1 의 실제 런타임 고정 홈 = reset_home_palm_pose
        #   (0.28,-0.38,0.42 / ez90·ey0·ex90) 를 sensor_rl 에서 IK 역산한 관절값
        #   (probe_solve_v1_home 08.20: 오차 2.2mm/0.6°, 손끝 z 0.37~0.44 테이블 위).
        # ★grasp_v1 의 cfg init joint 값(0.5,0.1,...)을 복사하면 안 된다 — 그 값은
        #   시작 시 IK 로 덮어써지는 자리표시자이고, sensor_rl 에선 손이 스폰 박스를
        #   점유해 컵을 리셋 즉시 밀어낸다(팔 홈은 관절값이 아니라 palm 포즈가 정의).
        "r_aj_1": 0.0380, "r_aj_2": 0.4012, "r_aj_3": 0.6015, "r_aj_4": 0.9643,
        "r_aj_5": 0.0294, "r_aj_6": 0.7060, "r_aj_7": 0.4213,
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
        # ★★08.25 DEXTRAH Kuka(`assets/kuka_allegro/kuka_allegro.py`) 게인으로 전환.
        #   Kuka 는 팔 게인을 **원위로 갈수록 낮춘다**(kp 300→25, kd 45→15) — 손목이
        #   부드러워 접촉 시 팔이 물체를 밀어내지 않는다. 우리는 400/80 균일이었다.
        #   ★이 전환은 real2sim 07.29 실측(friction 0.213/0.493/0.151, 직접 토크 식별로
        #     실물 우팔 kp ≤13% 오차 검증)을 **덮어쓴다**. 사용자 지시("모두 KUKA
        #     SETTING으로")에 따른 것이며, 실기 배포 시에는 재검토가 필요하다.
        #     되돌리려면 이 블록만 아래 구 값으로 복원하면 된다:
        #       [1-3] 400/80 f0.213 · 4 400/80 f0.493 · [5-7] 400/80 f0.151 · 손 5.0/2.0 e1.5
        # ★★08.31 실기 배포 — 위 주석이 예고한 "재검토" 시점이 왔다.
        #   `HDGP_S2R_REAL_GAINS=1` 이면 **실기 실측 게인**으로 바꾼다(기본값은 KUKA 유지).
        #   근거: 07.29 계단 실측 kp 74.7/75.1/69.5/60.9/10.8/14.5/10.5(벤더 스펙과 ≤4%)
        #   + autotune damping 6.376/5.635/2.154(벤더 kd 가 2.6~3.6배 부족했다는 발견).
        #   ★이 자산은 **로봇 중력이 꺼져 있다**(env_cfg:118 disable_gravity=True). 실기에
        #     중력보상을 켜면 같은 조건이 된다 — 08.31 실기에서 그 조합으로 추종오차
        #     RMSE 0.94° 를 얻었다. 즉 게인만 맞추면 sim↔실기가 같은 물리가 된다.
        #   ⚠기본값을 바꾸지 않는 이유: 현 배포 정책 b1_ep10800 이 KUKA 게인에서 학습됐다.
        #     게인을 바꾸면 재학습이 필요하다.
        # ★★09.01 갱신 — R3 자세 여진(r2s collect ×3 → fit)으로 **관절별** 재식별.
        #   07.29 값(kp 73.1/60.9/11.9 · kd 6.376/5.635/2.154, 3그룹)을 대체한다.
        #   그때는 **손이 없거나 다른 조건**의 정적 계단이었고, 지금은 테솔로 손
        #   1.763 kg 을 단 채 동적으로 쟀다. 손목 관성이 10~12배 달라지므로 손목
        #   게인이 다른 것이 당연하다 — 07.29 를 기준으로 삼지 않는다(사용자 판단).
        #   근거 파일: sim2real/logs/r2s/right_R3_s065_fit.json (holdout 1런 보유)
        #   ★자기정합: fit 이 낸 kd/√kp 하위 셋(j5 0.14·j7 0.21·j6 0.30)이 실측
        #     오버슈트 상위 셋(×1.47·×1.58·×2.07)과 정확히 일치한다. 나머지 넷은
        #     kd/√kp ≥ 0.55 이고 오버슈트가 전부 ≈1.00 이다.
        # ★★09.01 2차 모델 fit 으로 재계산 — `sim2real/scripts/fit_excite_model.py`.
        #   `robotctl r2s fit` 은 모델에 **armature 가 없어** kp 를 부풀려 맞춘다. 여진
        #   응답에 관절별 2차계 `J q̈ + kd q̇ + kp q = kp q_des` 를 직접 맞추면 (ωn, ζ)
        #   가 kp 와 무관하게 결정되고, kp 를 주면 J = kp/ωn² · kd = 2ζωn J ·
        #   armature = J − J_link 가 따라온다. holdout 런 RMSE 가 fit 과 거의 같다
        #   (0.43~2.05° vs 0.41~1.90°) — 과적합이 아니다.
        # ★★★09.01 최종본 — **kp 는 벤더 실기값, kd 만 맞춘다.**
        #   kp 를 ωn 에서 역산해 바꿨더니(j2 70→40.8) preset 궤적 재생에서
        #   **j2 RMSE 10.77° · max 42.80°** 로 무너졌다(실기는 RMSE 0.94°). 여진은
        #   R3 한 자세에서 ±3~9° 만 흔드는데 preset 은 ±50° 를 움직인다 — 관성이
        #   자세에 따라 변하므로 **작은 진폭에서 맞춘 kp 는 큰 움직임에서 안 맞는다.**
        #   ⇒ kp 는 실기 그대로(70/70/70/60/10/10/10) 두고, kd 로만 ζ 를 맞춘다:
        #     `kd = 2ζ·√(kp·J_sim)` · 손목은 여기에 스윕 배율(j5 ×5.0 · j6 ×4.0 · j7 ×0.5).
        #   armature 는 0 — sim 자산의 관성이 이미 대체로 옳다(sim 관성 + 실기 ωn 으로
        #   역산한 kp 가 벤더 값과 6~12 % 안: j1 75.8/70 · j5 10.6/10 · j6 8.8/10).
        #
        # ※폐기한 시도(sim 관성 기준으로 kp 까지 바꾸는 것):
        #   여기 kp/kd 는 **실기 하드웨어 게인이 아니다.** sim 로봇의 관성으로 실기가
        #   보인 (ωn, ζ)를 내도록 푼 값이다: kp = ωn²·J_sim · kd = 2ζ·ωn·J_sim.
        #   실기 하드웨어 게인은 따로 있고 확인됐다 —
        #   `openarm_description/config/arm/v10/control_gains.yaml`(사본 8개 동일):
        #   kp 70/70/70/60/**10/10/10** · kd 2.75/2.5/2.0/2.0/0.7/0.6/0.5,
        #   `v10_simple_hardware.cpp:65-71,276` 이 이걸 읽어 모터에 그대로 보낸다.
        #   ★둘이 가까운 것이 검증이다: 위 kp 필요값 75.8/40.8/84.8/91.7/**10.6/8.8**/17.4
        #     vs 벤더 70/70/70/60/**10/10/10** — j1·j5·j6 은 6~12% 안에 든다.
        #     즉 **sim 자산의 관성이 대체로 옳다.**
        #   ★★armature 는 넣지 않는다(0). 세션 중반에 0.8~1.0 을 넣었던 것은
        #     링크 관성을 point-mass(`Σm·d²`)로 어림해 **링크 자체 회전관성을 빼먹은**
        #     탓이다. sim 실측 관성은 그 어림의 4.5~5.6배였다(j1 0.914 vs 0.203).
        #     "반사관성"이라고 불렀던 것의 정체가 그 빠뜨린 항이다.
        # ★★09.01 손목 kd 는 **병렬 스윕으로 관절별 최적화**했다
        #   (`probe_excite_sim_replay.py --num-envs 16 --kd-scale`). `Articulation` 이
        #   env_ids 를 받으므로(`articulation.py:640`) 6분 한 번에 16개 조합을 본다.
        #   점수는 **lock-in 주파수 응답**(0.7/1.3/2.1/3.7 Hz)이다 — ptp 비(최대−최소)로
        #   재면 kd 를 3배 키워도 12 % 밖에 안 변해 최적화가 서지 않는다.
        #   관절별 최적 배율 j5 ×5.0 · j6 ×4.0 · j7 ×0.5 → 오차 0.689→0.486.
        #   ⚠j7 은 스윕 **범위 끝**에서 최적이라 더 낮출 여지가 있다.
        #   ⚠팔(j1~j4)은 오차 0.051 로 이미 맞아 스윕에서 제외했다.
        #   ⚠아직 sim 재생 검증 전이다. 직전 라운드 기록:
        #     KUKA 0.429 · 실측게인+손목armature 0.185 · 실기kp+armature 0.326.
        #
        # ※이전 라운드 주석(kp 를 파일에서 확인한 경위):
        #   `v10_simple_hardware.cpp:65-71,276` 이 URDF hardware_parameters 의
        #   kp1..7/kd1..7 을 읽어 모터에 그대로 보내고, 그 출처인
        #   `openarm_description/config/arm/v10/control_gains.yaml` 은 워크스페이스
        #   **사본 8개가 전부 동일**하다: kp 70/70/70/60/**10/10/10** ·
        #   kd 2.75/2.5/2.0/2.0/0.7/0.6/0.5. 오버라이드 없음.
        #   ⇒ `r2s fit` 이 낸 kp(96~190)는 **틀렸다**. armature 가 없는 모델이라
        #     관성을 kp 로 흡수해 j6 을 19배 부풀렸다.
        #   ★★검증: kp=10 으로 풀면 j6 의 J = 10/(2π·2.36)² = **0.0456** 이고 URDF
        #     링크 관성이 **0.0426** 이다 — armature 0.003, 즉 거의 0 이고 소수점
        #     셋째 자리까지 맞는다. 우연일 수 없다.
        #   ⇒ **손목 관성의 93 %가 손(1.763 kg)이다**(손 없으면 0.0040). 사용자가
        #     처음부터 말한 "손 무게 때문"이 옳았다. 세션 중간에 "모터 반사관성이
        #     지배한다"고 뒤집었던 것은 잘못된 kp 를 믿은 결과다.
        #   ⚠kd 는 fit(손목 0.43/0.15/0.37)과 벤더 설정(0.7/0.6/0.5)이 같은 자릿수지만
        #     j6 만 4배 차이나고, 팔은 fit 이 2~3배 크다. 아직 정합하지 않는다.
        #   ⚠아래 값은 **sim 재생 검증을 아직 못 했다**(세션 종료). 다음 세션에서
        #     `probe_excite_sim_replay.py` 로 오버슈트를 대조할 것.
        #     직전 검증본(kp 96~190 전제, 오차 0.185)은 git 이력에 있다.
        #
        # ※이전 라운드 기록: 아래 조합이 실기 오버슈트를 가장 잘 재현했다(오차 0.185,
        #   현행 KUKA 0.429 대비 2.3배). 5개 조합을 sim 재생으로 실측 대조한 결과다:
        #     KUKA(현행 기본값)            0.429
        #     실측 게인만                  0.341
        #     ★실측 게인 + **손목만** armature  0.185   ← 이것
        #     2차모델 fit(kd 재계산)        0.211
        #     2차모델 + 마찰 분리           0.216 (손목 0.148 로 최선이나 팔이 0.266 로 악화)
        #   ⚠**sim 의 friction 은 작동하지 않는다.** friction 을 1.019→0 으로 바꿔도
        #     재생 결과가 소수점까지 동일했다(09.01 실측). 감쇠는 kd 로만 들어간다 —
        #     아래 friction 값은 기록용이지 sim 동특성에 기여하지 않는다.
        #   ⚠남는 오차: j6 1.45 vs 실기 2.01. j5·j6 armature 미세조정이 다음 과제다.
        #
        #   ※참고: 2차 모델 fit 은 `sim2real/scripts/fit_excite_model.py` 에 남아 있고
        #     결과는 logs/r2s/excite_model_fit{,_fric}.json 이다. 모델에 Coulomb 마찰을
        #     넣으면 빼면 그 감쇠가 ζ 로 흡수되고, sim 에
        #     kd 와 friction 을 둘 다 넣게 되어 감쇠가 이중 계산된다 — 실제로 그렇게
        #     했더니 sim 오버슈트가 실기보다 작았다(j6 1.51 vs 2.01). 마찰을 넣자
        #     잔차가 7관절 전부에서 줄었다(j3 0.847→0.727 · j5 0.860→0.711).
        #   측정된 ωn[Hz]/ζ: j1 1.45/0.372 · j2 2.58/0.579 · j3 1.46/0.163 ·
        #     j4 1.24/0.292 · **j5 1.40/0.071 · j6 2.36/0.012 · j7 1.39/0.069**
        #   ★손목은 거의 무감쇠이고 저항의 대부분이 마찰이다. armature 는 마찰을
        #     넣든 빼든 0.82±0.01 로 거의 안 변한다 — 그만큼 강건한 추정이다.
        #   ⚠속도 피드포워드 모델(kd·q̇_des 항 포함)도 시험했으나 팔에서 잔차가 크고
        #     (j1 0.81 vs 0.52) 지연을 57~149 ms 로 잡아 비현실적이다. 컨트롤러가
        #     `interpolation_method: "none"` 이라 속도 지령이 서지 않는 것과 정합한다.
        #
        # ★★armature(모터 반사관성 = 기어비²×회전자). 없으면 sim 은
        #   실기 공진을 **재현하지 못한다**. 근거: 공진에서 역산한 등가 관성
        #   I_eq = kp/ωn² 이 손목 3관절에서 0.853/1.088/1.006 으로 모이는데,
        #   URDF 링크 관성은 0.027/0.043/0.051 로 서로 2배 차이다. 같은 모터면
        #   반사관성이 같다는 것과 정합하고, 손목이 느끼는 관성의 95~97%가 모터다.
        #   ⇒ 손 1.763 kg 는 등가 관성의 3~5% 에 불과하다(오버슈트의 주원인이 아니다).
        #   실측 대조(여진 오버슈트 실기 j5 1.46·j6 2.01·j7 1.56):
        #     armature 없이 KUKA 게인   → 1.24·0.74·0.54 (손목 평균오차 0.843)
        #     armature 없이 실측 게인   → 1.24·1.01·1.10 (0.564)
        #   ★j1~j4 는 armature 를 넣지 않는다. 과감쇠라 공진 피크가 없어 fn 추정이
        #     거칠었고, 거기서 나온 값(j4 2.726)을 넣자 sim 이 0.7 Hz 에서 공진해
        #     실기 1.00 대비 1.50 으로 **악화**했다(09.01 실측). 팔은 원래 오버슈트가
        #     없는 관절이므로 링크 관성만으로 충분하다.
        #   ⚠friction 은 중력보상을 켠 채로 잰 값이라 잔여 중력을 일부 흡수했다
        #     (fit 경고: "standing load has landed in bias"). j2·j3 가 큰 이유일 수
        #     있다 — 보상 OFF 대조군을 받으면 갈라진다.
        **(
            {
                "right_arm_j1": dict(joint_names_expr=["r_aj_1"], stiffness=70.0,
                                     damping=7.053, friction=0.0,
                                     effort_limit_sim=300.0),
                "right_arm_j2": dict(joint_names_expr=["r_aj_2"], stiffness=70.0,
                                     damping=4.182, friction=0.0,
                                     effort_limit_sim=300.0),
                "right_arm_j3": dict(joint_names_expr=["r_aj_3"], stiffness=70.0,
                                     damping=7.804, friction=0.0,
                                     effort_limit_sim=300.0),
                "right_arm_j4": dict(joint_names_expr=["r_aj_4"], stiffness=60.0,
                                     damping=6.531, friction=0.0,
                                     effort_limit_sim=300.0),
                "right_arm_j5": dict(joint_names_expr=["r_aj_5"], stiffness=10.0,
                                     damping=2.236, friction=0.0,
                                     effort_limit_sim=300.0),
                "right_arm_j6": dict(joint_names_expr=["r_aj_6"], stiffness=10.0,
                                     damping=0.580, friction=0.0,
                                     effort_limit_sim=300.0),
                "right_arm_j7": dict(joint_names_expr=["r_aj_7"], stiffness=10.0,
                                     damping=0.242, friction=0.0,
                                     effort_limit_sim=300.0),
            }
            if _os.environ.get("HDGP_S2R_REAL_GAINS") == "1"
            else {
                "right_arm_proximal": dict(joint_names_expr=["r_aj_[1-4]"], stiffness=300.0,
                                           damping=45.0, effort_limit_sim=300.0),
                "right_arm_j5":       dict(joint_names_expr=["r_aj_5"], stiffness=100.0,
                                           damping=20.0, effort_limit_sim=300.0),
                "right_arm_j6":       dict(joint_names_expr=["r_aj_6"], stiffness=50.0,
                                           damping=15.0, effort_limit_sim=300.0),
                "right_arm_j7":       dict(joint_names_expr=["r_aj_7"], stiffness=25.0,
                                           damping=15.0, effort_limit_sim=300.0),
            }
        ),
        # ★★손 게인은 **Tesollo 실측**으로 간다(Kuka Allegro 값 3.0/0.1/0.5 로 덮였던 것을
        #   되돌림). grasp_v1 에 남아 있는 이 손의 kd 스윕이 근거다:
        #     kd 6.71 → 포화 20.5%(감쇠항 자체가 토크를 포화) · kd ≤ 0.5 → 정착속도 2배(채터)
        #     · **kd 2.0 이 포화 0.8% + 최저 채터로 양쪽 최적**
        #   kd 0.1 은 그 스윕이 기각한 채터 영역이었다. Allegro 는 다른 손이고,
        #   "KUKA 충실은 1순위 기준이 아니다"가 이 트랙의 확정 사항이다.
        # ★effort 0.5 는 유지 가능한 위치오차가 0.5/3.0 = 9.5° 뿐이라 컵을 눌러 감쌀 힘이
        #   안 났다(08.27: wrap_frac 이 전 런에서 0.000). 1.5/5.0 = 17.2° 로 회복.
        #   ⚠실기 d=0.0 이므로 이 damping 은 기계마찰의 sim 대역품 —
        #   r2s 복구 후 armature/joint friction 실측치로 교체할 것(grasp_v1 규약).
        "hand":               dict(joint_names_expr=["r_hj_[a-z]+_[1-4]"], stiffness=5.0, damping=2.0,
                                   effort_limit_sim=1.5),
        "left_arm":           dict(joint_names_expr=["l_aj_[1-7]"], stiffness=400.0, damping=80.0),
        "left_gripper":       dict(joint_names_expr=["l_hj_gripper_[1-2]"], stiffness=400.0, damping=80.0),
        "head":               dict(joint_names_expr=["head_j_(pan|tilt)"], stiffness=400.0, damping=80.0),
    },
    # ★★08.27 grasp_ua: 구 (0.30, −0.20) → (0.362, −0.16).
    #   부팅 실측(`_report_home_cage`)으로 홈 케이지 중심이 (0.3623, −0.3137, 0.4212),
    #   반경 120mm 임을 확인했다. 구 스폰에서는 케이지−컵 = (+62, −114, +114)mm 라
    #     · x +62mm : 케이지가 컵을 **지나쳐** 있어 정책이 후진 후 재접근해야 했다
    #                 (y 이동과 겹쳐 3D 대각선 — 사용자 GUI 관찰)
    #     · 엄지가 컵에 걸린 채 지령이 계속 아래를 향해, 풀리는 순간 손이 테이블까지
    #       내려갔다(같은 관찰)
    #   x 를 케이지에 정렬하면 접근이 y-z 평면 2D 로 단순해진다. y 간격은 케이지
    #   반경(120mm)보다 크게 잡아(154mm) 컵이 홈 케이지 안에 들어간 채 시작하지
    #   않도록 한다 — 그러면 리셋 순간 손가락 메시가 컵을 관통한다.
    #   ★좌팔 그리퍼 트랙 결론과 같은 처방이다: "컵을 앞에 둔다"와 "홈을 뒤로 물린다"는
    #     로봇 기준 상대 배치가 같아 물리적으로 동등하다.
    #   ⚠이 값은 **홈에 종속**이다. 홈을 바꾸면 `_report_home_cage` 로 다시 재라.
    object_spawn_center=(0.362, -0.16),
)


_RH_FLEX = ("index", "middle", "ring", "pinky")

# =============================================================================
# rh56f1_right — Inspire RH56F1 우손. **물리 12관절 중 구동 6**(언더액추에이션).
#
# ★이 프로필이 grasp_ua 트랙이 grasp_s2r 에서 갈라진 이유다. 자유도가 낮아서가
#   아니라 **제어 자유도(6) ≠ 물리 자유도(12)** 라서다:
#     r_hj_thumb_3 = thumb_2 × 1.1425 · thumb_4 = thumb_3 × 0.7508
#     r_hj_{index,middle,ring,pinky}_2 = 같은 손가락 _1 × 1.1169
#   URDF 에는 <mimic> 이 있지만 **USD 로 오면서 사라진다**(현 자산 실측: PhysxMimicJoint
#   0건). 선행 rh56f1 트랙은 원위 6개에 강성 400/60 만 주고 지령을 안 보내
#   init 각도에 굳혀 뒀다 — sim 의 손가락이 사실상 1자유도 강체 막대가 된다.
#   s2r 트랙에서 그건 "실기에 없는 손"을 학습하는 것이라 `hand_mimic` 으로 명시한다.
#   ★PhysxMimicJoint 가 들어간 USD 가 준비되면 `hand_mimic_mode="physx"` 로만 바꾼다
#     (`hand_mimic` 표는 그대로 둔다 — 부팅 검증에 쓰인다).
#
# 출처: assets/robot/openarm_bi_rh56f1_rl/openarm_bi_rh56f1_rl.urdf 실측(축·한계·mimic)
#       + 선행 트랙 openarm/rh56f1/right/grasp_v2 의 검증된 자세·홈.
# =============================================================================
RH56F1_RIGHT = RobotProfile(
    name="rh56f1_right",
    # ★tesollo 처럼 `_armhull` 변형 자산이 아직 없다. 손 접촉력 왜곡을 피하려면
    #   (손 decomposition + 팔 hull) 변형이 필요하다 — 만들기 전까지 원본을 쓴다.
    usd_relpath="robot/openarm_bi_rh56f1_rl/openarm_bi_rh56f1_rl.usd",
    num_arm_joints=7,
    num_hand_joints=6,
    arm_joint_regex="r_aj_[1-7]",
    # ★구동 6개만. 종속 6개(_2 · thumb_[34])는 `hand_mimic` 이 따로 몬다.
    hand_joint_regex="r_hj_(thumb_[12]|index_1|middle_1|ring_1|pinky_1)",
    # ★palm_body 는 **fabric 제어점과 같아야 한다** — 부팅 FK 게이트가 5mm/2° 로
    #   대조한다(OpenArmRh56f1PoseFabric._palm_frame = "r_hl_palm_sensor").
    palm_body="r_hl_palm_sensor",
    fabric_class="OpenArmRh56f1PoseFabric",
    fabric_robot_dir="openarm_rh56f1",
    fabric_params_filename="openarm_rh56f1_pose_params.yaml",
    # ★이 fabric 은 **양팔 26 DOF** 다(tesollo 는 우측 27 단독). cspace 전체를 적고
    #   제어 대상은 슬롯으로 가리킨다. 순서는 FABRICS urdf 실측(revolute 26, _rl 정의순).
    fabric_joint_order=(
        tuple(f"r_aj_{i}" for i in range(1, 8))
        + ("r_hj_thumb_1", "r_hj_thumb_2")
        + tuple(f"r_hj_{f}_1" for f in _RH_FLEX)
        + tuple(f"l_aj_{i}" for i in range(1, 8))
        + ("l_hj_thumb_1", "l_hj_thumb_2")
        + tuple(f"l_hj_{f}_1" for f in _RH_FLEX)
    ),
    fabric_arm_slot=tuple(range(0, 7)),
    fabric_hand_slot=tuple(range(7, 13)),
    # ★이 클래스는 양팔 하나로 좌우를 다 들고 있어 어느 팔의 palm IK 를 살릴지
    #   생성자가 정한다(비활성 팔은 cspace target 으로만 유지된다).
    fabric_kwargs={"side": "right"},
    # ---- 시너지 그립 (선행 트랙 검증 자세 승계) ----------------------------------
    hand_joint_names=(
        "r_hj_thumb_1", "r_hj_thumb_2",
        *tuple(f"r_hj_{f}_1" for f in _RH_FLEX),
    ),
    # ★thumb_1 은 **외전(대향)** 이지 굴곡이 아니다. open 1.57 = opposition 으로 벌려
    #   컵이 들어올 gap 을 만들고, grip 1.20 으로 내려 컵을 받친다(선행 트랙 실측:
    #   1.57 고정이면 정책이 엄지를 못 쓰고 gap 에 핀만 박힌다).
    #   ★그래서 open→grip 이 **감소**한다 — 시너지 lerp 는 부호에 무관하다.
    hand_open_pose=(1.57, 0.00, 0.00, 0.00, 0.00, 0.00),
    hand_grip_pose=(1.20, 0.24, 1.08, 1.08, 0.85, 0.85),
    # 접미사 → 채널. RH56F1 은 접미사가 `1`(엄지 외전 / 4지 굴곡)과 `2`(엄지 굴곡)뿐이다.
    # ★같은 `_1` 이 엄지에선 외전, 4지에선 굴곡이지만 채널은 **손가락별로** 전개되므로
    #   (a 는 (N, 손가락, 채널)) 섞이지 않는다.
    hand_channel_of_joint={"1": 0, "2": 1},
    # ★기본 레이아웃은 per_finger 다(cfg 에서 지정) — 6슬롯이 구동 6관절과 1:1.
    #   채널 레이아웃(2×5=10)을 쓰면 4지의 채널1 이 아무 관절에도 안 걸려 액션 4개가
    #   그냥 죽는다.
    hand_finger_channels={
        "thumb": {"1": 0, "2": 1},
        "index": {"1": 2},
        "middle": {"1": 3},
        "ring": {"1": 4},
        "pinky": {"1": 5},
    },
    # ★손가락당 구동관절이 하나라 "닿은 마디만 멈추고 원위가 계속 감는다"는 tesollo 식
    #   드레이프가 **구조적으로 불가능**하다. 감쌈은 하드웨어 결합(_2 = _1 × 1.1169)이
    #   만든다. 그래서 접촉 동결은 손가락 단위이고 중간마디 접미사는 비운다.
    #   → 권장 운전점은 `synergy_hold_mode="blocked"`(토크 포화 기반, 형상 무관)다.
    hand_freeze_suffixes=("1", "2"),
    hand_freeze_mid_suffixes=(),
    # ---- 언더액추에이션 표 (URDF <mimic> 그대로) ---------------------------------
    hand_mimic={
        "r_hj_thumb_3": ("r_hj_thumb_2", 1.1425, 0.0),
        "r_hj_thumb_4": ("r_hj_thumb_3", 0.7508, 0.0),   # ★체인 — 부팅에서 전개된다
        **{f"r_hj_{f}_2": (f"r_hj_{f}_1", 1.1169, 0.0) for f in _RH_FLEX},
    },
    # ★★09.02 physx 유지. software 로 되돌려 봤으나 **더 나빴다** — USD 의
    #   PhysxMimicJoint 제약이 그대로 남아 있어 우리 PD 와 싸운다(a=0 에서도
    #   mimic_err 32.8 rad). 자산에 제약이 박힌 이상 physx 가 유일한 경로다.
    hand_mimic_mode="physx",
    # ---- palm 워크스페이스 -------------------------------------------------------
    # 선행 rh56f1 트랙 `grasp_right_preset.palm_pose_mins/maxs` 승계(같은 OpenArm 팔).
    # ★★단 **z 하한만 0.20 → 0.30 으로 올렸다**(09.02 실측). 이 손은 palm 원점에서
    #   최하단 링크까지 114mm 다(홈에서 palm z 0.419 · hand_z_min 0.305). 즉 palm z 가
    #   0.314 밑이면 손 링크가 테이블(z 0.200) 안으로 들어간다. 앵커 z(0.329~0.356)에서
    #   델타 −0.10 은 0.23~0.26 이라 그 한참 아래고, 실제로 그 점을 지령하면 fabric 이
    #   **테이블 아래로 발산**했다(케이지 z 0.019 · 오차 367mm, probe_ua_home_calib).
    #   → z 축은 박스가 **일부러 자른다**. 부팅이 "잘리는 축 z" 를 찍는 게 정상이다.
    #   ⚠델타 z 를 줄여 자름을 없애면 안 된다 — 이송 목표가 +0.12 라 위쪽 여유가
    #     그대로 필요하다(델타는 ±0.10 유지, 하한만 박스로 막는다).
    palm_box_min=(0.20, -0.55, 0.30),
    palm_box_max=(0.65, 0.22, 0.65),
    # ★★09.02 캘리브로 확정. 태스크가 기대하는 기준 자세는 tesollo 홈에서 읽는다 —
    #   **손바닥 법선 +y(컵 쪽) · 손가락 +x(수평)**. 그걸 이 프레임으로 옮기면
    #   R_ps = palm_frame_remap 이 되는데, 그 오일러가 정확히 **짐벌락(ey=−90°)** 이다.
    #   그래서 손가락을 25° 아래로 기울여 특이점에서 떼어냈다: ez 0 · ey −65 · ex −90.
    #     · 손바닥 법선은 여전히 정확히 +y (컵을 향한다)
    #     · 손가락이 25° 아래를 향해 테이블 위 컵으로 내려가기 자연스럽다
    #     · 짐벌락 자세(ey −90°)가 ±45° 박스 **안**이라 정책이 수평 파지도 낼 수 있다
    #   ⚠선행 트랙 프리셋 (180, 0, 90) 을 그대로 쓰면 안 된다 — 그건 손가락이 **위**를
    #     향하는 자세이고, 실측에서 fabric 이 도달 못 해 130mm/51° 로 포화했다.
    palm_rot_center_deg=(0.0, -65.0, -90.0),
    palm_rot_half_deg=45.0,
    # ★홈은 이 중심과 **같다**(위 init_joint_pos 가 그 자세로 수렴한 값이다) —
    #   `_palm_anchor` 가 회전 성분을 홈 그대로 두므로, 홈 자세 = 액션 회전 원점이다.
    palm_box_verified=False,      # ← 델타박스 도달성 격자 통과 후 True 로 승격
    # ★★URDF 실측 유도: r_hj_palm_sensor 의 rpy=(90°,0,90°) → 회전행렬 R(ps→base) 의
    #   열이 (base+y, base+z, base+x). 손바닥 법선은 base +x 이므로 palm_sensor 국소
    #   **+z** 가 법선이다(선행 트랙 주석 "palm_sensor +z(법선)"과 일치).
    #   태스크 표준(열0=법선, 열1=가로, 열2=손가락방향)으로 맞추는 행렬:
    #     표준x = 국소(0,0,1) · 표준y = 국소(1,0,0) · 표준z = 국소(0,1,0)
    #   ★base 프레임에서 보면 tesollo 와 완전히 같은 규약이다(손가락 +z, 법선 +x) —
    #     어긋나는 건 palm_sensor 프레임 하나뿐이라 여기서만 흡수한다.
    palm_frame_remap=((0.0, 1.0, 0.0),
                      (0.0, 0.0, 1.0),
                      (1.0, 0.0, 0.0)),
    # ---- 접촉 ---------------------------------------------------------------------
    # 규약: (중간, 원위, 팁). RH56F1 4지는 2마디라 (_1, _2, _tip), 엄지는 원위 2마디가
    # mimic 이라 (_3, _4, _tip). 전 링크에 collision 이 있어 센서가 실제로 잡는다(URDF 실측).
    finger_sensor_bodies={
        "thumb": ("r_hl_thumb_3", "r_hl_thumb_4", "r_hl_thumb_tip"),
        **{f: (f"r_hl_{f}_1", f"r_hl_{f}_2", f"r_hl_{f}_tip") for f in _RH_FLEX},
    },
    contact_group_a=("thumb",),
    contact_group_b=_RH_FLEX,
    envelope_fingers=("thumb", *_RH_FLEX),
    # ★URDF 실측 유도 — 필드 주석의 cross(굴곡축, 장축), **마디 링크 국소 프레임**:
    #   4지: r_hj_{f}_{1,2} 축 (0,0,-1) · 장축(자식 origin) ≈ +y → cross = (+1,0,0)
    #   엄지: r_hj_thumb_{3,4} 축 (0,0,-1) · 장축 ≈ (−0.6,+0.8,0) 정규화 → cross ≈ 그 수직
    #        thumb_3 origin (−0.029155, 0.03869, −0.0032) → 장축 (−0.602, 0.799, 0)
    #        cross((0,0,-1), (−0.602,0.799,0)) = (0.799, 0.602, 0)
    palmar_axis_local={
        "thumb": (0.799, 0.602, 0.0),
        **{f: (1.0, 0.0, 0.0) for f in _RH_FLEX},
    },
    fingertip_bodies=("r_hl_thumb_tip", *tuple(f"r_hl_{f}_tip" for f in _RH_FLEX)),
    init_joint_pos={
        # ★★09.02 캘리브 실측 — `scripts/probes/probe_ua_home_calib.py` 로 목표 palm
        #   포즈(0.310, −0.300, 0.419 / ez 0° ey −65° ex −90°)를 fabric 으로 추종시켜
        #   수렴한 관절값이다(오차 0.14mm / 0.07°).
        #   ★위치는 **홈 케이지가 컵과 x 정렬**되도록 역산했다 — 케이지는 palm 에 강체로
        #     붙어 있고 이 자세에서 world 오프셋이 실측 (+0.070, +0.035, −0.017) m 이라
        #     home_x = 스폰_x − 0.070 이면 접근이 y-z 평면 2D 가 된다(tesollo 와 같은 처방).
        #   ⚠선행 트랙 값(0.5, 0.1, 0.4, 0.60, −0.2, 0, 0)을 그대로 쓰면 안 된다 —
        #     같은 팔이라도 손이 바뀌면 palm 프레임이 달라져 같은 관절값이 전혀 다른
        #     자세를 낸다(그 값은 이 프레임에서 ez −9.6° ey −62.7° ex −99.1° 를 냈고,
        #     부팅 게이트가 "홈이 워크스페이스 박스 밖"으로 막았다).
        "r_aj_1": -0.2069, "r_aj_2": 0.4203, "r_aj_3": 0.3880, "r_aj_4": 1.5941,
        "r_aj_5": 0.4569, "r_aj_6": 0.3426, "r_aj_7": -0.4174,
        # 손 구동 6 = hand_open_pose
        "r_hj_thumb_1": 1.57, "r_hj_thumb_2": 0.0,
        **{f"r_hj_{f}_1": 0.0 for f in _RH_FLEX},
        # ★종속 6 = 구동 × 배율. 결합값으로 시작하지 않으면 첫 스텝에 snap 이 난다.
        "r_hj_thumb_3": 0.0, "r_hj_thumb_4": 0.0,
        **{f"r_hj_{f}_2": 0.0 for f in _RH_FLEX},
        # 유휴 좌팔 — fabric default_config 의 _ARM_REST_L 과 같은 값(정합 유지).
        "l_aj_1": -0.315, "l_aj_2": -0.290, "l_aj_3": 0.400, "l_aj_4": 0.513,
        "l_aj_5": 0.666, "l_aj_6": -0.729, "l_aj_7": -0.957,
        "l_hj_thumb_1": 0.0, "l_hj_thumb_2": 0.0,
        **{f"l_hj_{f}_1": 0.0 for f in _RH_FLEX},
        "l_hj_thumb_3": 0.0, "l_hj_thumb_4": 0.0,
        **{f"l_hj_{f}_2": 0.0 for f in _RH_FLEX},
        "head_j_pan": 0.0, "head_j_tilt": 0.0,
    },
    actuator_specs={
        # ★★★종속 6 — **위치 PD 는 안 걸고 감쇠만** 건다(kp=0).
        #   위치 목표를 주면 PhysX mimic 제약과 싸운다(09.02 실측: software 로 PD 를
        #   걸었더니 a=0 에서도 mimic_err 32.8 rad). 그렇다고 아무것도 안 걸면 그
        #   관절엔 **감쇠가 0** 이라 stiff 제약(naturalFrequency 500, dt 1/120 에서
        #   ω·dt≈4.2)만 작용해 링잉이 발산한다.
        #   09.02 파탄 순간 실측(random 액션 step 20): 팔은 |qd| 2.8 rad/s 로 멀쩡한데
        #     hand_drive |qd| 101 · hand_mimic |qd| **181 rad/s** — 위치는 아직 범위
        #     안(0.38 rad)이고 **속도만** 폭주했다. 감쇠 부재가 유일한 설명이다.
        #   kp=0 이면 위치항이 없어 제약과 안 싸우고, kd 만 에너지를 뽑는다.
        #   ★effort 상한을 따로 올린다. 기본(USD 값 1 N·m)이면 감쇠 토크가 1 rad/s
        #     에서 이미 포화해 **에너지를 못 뽑는다** — 09.02 실측에서 감쇠를 넣고도
        #     종속 |qd| 가 87 rad/s 였다. kp=0 이라 이 상한은 **소산에만** 쓰이고
        #     에너지를 주입할 수 없다(파지력과 무관하다 — 파지력은 구동관절 몫).
        "right_hand_mimic_damp": dict(
            joint_names_expr=["r_hj_(thumb_[34]|index_2|middle_2|ring_2|pinky_2)"],
            stiffness=0.0, damping=2.0, effort_limit_sim=20.0),
        **({
            # ★★09.02 실기 벤더 게인 — `control_gains.yaml` 이 진실원천이고, 같은 값이
            #   이번 자산 갱신으로 USD DriveAPI 에도 실렸다(단위 변환된 형태로 확인).
            #   팔은 tesollo 와 **같은 OpenArm** 이라 kp 가 그 트랙 실기 분기와 동일하다.
            #   ⚠kd 는 벤더값이다 — tesollo 는 손 1.763 kg 을 단 채 r2s 로 **재식별**한
            #     kd 를 쓰는데, RH56F1 손은 질량이 달라 그 값을 쓸 수 없다. RH56F1 로
            #     r2s 를 돌리기 전까지 이건 **출발점이지 정합값이 아니다**.
            "right_arm_j1": dict(joint_names_expr=["r_aj_1"], stiffness=70.0, damping=2.75),
            "right_arm_j2": dict(joint_names_expr=["r_aj_2"], stiffness=70.0, damping=2.50),
            "right_arm_j3": dict(joint_names_expr=["r_aj_3"], stiffness=70.0, damping=2.00),
            "right_arm_j4": dict(joint_names_expr=["r_aj_4"], stiffness=60.0, damping=2.00),
            "right_arm_j5": dict(joint_names_expr=["r_aj_5"], stiffness=10.0, damping=0.70),
            "right_arm_j6": dict(joint_names_expr=["r_aj_6"], stiffness=10.0, damping=0.60),
            "right_arm_j7": dict(joint_names_expr=["r_aj_7"], stiffness=10.0, damping=0.50),
        } if _os.environ.get("HDGP_S2R_REAL_GAINS") == "1" else {
            # KUKA 기본(DEXTRAH) — 원위로 갈수록 낮춘다. tesollo 트랙과 같은 값.
            "right_arm_proximal": dict(joint_names_expr=["r_aj_[1-4]"], stiffness=300.0, damping=45.0),
            "right_arm_j5":       dict(joint_names_expr=["r_aj_5"], stiffness=100.0, damping=25.0),
            "right_arm_j6":       dict(joint_names_expr=["r_aj_6"], stiffness=50.0, damping=20.0),
            "right_arm_j7":       dict(joint_names_expr=["r_aj_7"], stiffness=25.0, damping=15.0),
        }),
        # 손 구동 6 — 선행 rh56f1 트랙 값(굴곡 400/60 · 엄지 외전 200/35).
        # ⚠이 자산의 손 effort 는 **1 N·m** 다(벤더 URDF 값, tesollo 1.5~7.5 대비 낮다).
        #   게인 400 은 그 상한에서 포화할 수 있다 — 파지력 실측 전까지 미지수다
        #   (선례: tesollo 400/60 이 effort 에서 포화해 k=5/kd=2 로 재설정됐다).
        # ★★★09.02 게인 400/60 · 200/35 → **5/2**. effort 가 1 N·m 인데 kp 400 이면
        #   오차 0.0025 rad 에서 포화해 사실상 **정토크원**이 된다 — 접촉이 들어오면
        #   관절이 그대로 역구동돼 한계 밖으로 밀린다(rh_a1_fresh iteration 1 실측:
        #   `hand_joint_err_max` 1.94 rad > 가동범위 1.53 rad). 그 상태에서 stiff
        #   mimic 제약이 에너지를 주입해 씬 전체가 발산했다.
        #   ★tesollo 가 08.16 에 겪은 것과 **같은 함정**이고 처방도 같다(400/60 → 5/2).
        #   ⚠파지력은 안 줄어든다 — 파지력은 effort(1 N·m ≈ 손가락당 25 N, 5지 12 kg)가
        #     정하고 kp 는 그 토크에 도달하는 **오차 폭**만 정한다. kp 5 면 0.2 rad 에서
        #     effort 에 닿는다(선형 구간 확보).
        "right_hand_flex":  dict(joint_names_expr=["r_hj_(thumb_2|index_1|middle_1|ring_1|pinky_1)"],
                                 stiffness=5.0, damping=2.0),
        "right_hand_abd":   dict(joint_names_expr=["r_hj_thumb_1"], stiffness=5.0, damping=2.0),
        # 유휴 좌팔·좌손 (hold). ★좌손도 구동 6만 — 좌측 종속 6도 DriveAPI 가 없다.
        "left_arm":        dict(joint_names_expr=["l_aj_[1-7]"], stiffness=400.0, damping=80.0),
        "left_hand":       dict(joint_names_expr=["l_hj_[a-z]+_[1-4]"], stiffness=30.0, damping=5.0),
        "head":            dict(joint_names_expr=["head_j_(pan|tilt)"], stiffness=400.0, damping=80.0),
    },
    # ★★09.02 캘리브. 두 제약을 동시에 만족하는 값이다:
    #   ① 액션 앵커(= 스폰 + `palm_anchor_offset_xyz`) ± 델타 0.10 이 palm 박스 안 —
    #      x 하한 0.20 때문에 스폰 x 는 0.367 이상이어야 한다(0.38 채택).
    #   ② 홈 케이지(0.467, −0.280, 0.445)와 컵이 충분히 떨어져 리셋 순간 손가락
    #      메시가 컵을 관통하지 않는다 — 실측 거리 204mm ≫ 케이지 반경 54mm.
    object_spawn_center=(0.38, -0.16),
)

# =============================================================================
# gripper_left — 같은 자산의 좌팔 2-DOF 평행 그리퍼. agnosticism 검증용(Phase 2):
# 이 프로필 추가 외에 태스크 코드 수정이 0 이어야 합격.
# 대향 그룹 = jaw1 / jaw2. l_hj_gripper_2 는 USD PhysX mimic(gearing=-1).
# =============================================================================
# =============================================================================
# gripper_left — Phase 2(agnosticism 검증)용. ★fabric_class=None:
#   sensor_left_gripper fabric 자산은 존재하지만 그 URDF 의 손은 2지 그리퍼가 아니라
#   DG-5F 이고, 그리퍼 트랙은 Fabrics 로 jaw 수평(손목 ±45°·effort 7N·m)을 못 내
#   자세오차 28° 로 ABORTED 된 이력이 있다. 조용히 폴백하지 말고 env 부팅에서 죽인다.
#   → 이 프로필로 Phase 2 를 하려면 전용 fabric 자산부터 만들어야 한다.
# =============================================================================
GRIPPER_LEFT = RobotProfile(
    name="gripper_left",
    # ★손 27개 링크는 convexDecomposition 유지, 나머지 23개(팔·몸통·헤드)만 convexHull.
    #   실측(arm5080): 처리량 +13.7%, 접촉력은 오히려 소폭 감소(36.2→32.8N, 반복측정
    #   편차 8% 안) = 촉각 obs 손실 없음. 컵에 닿는 건 손뿐이고 팔 자기충돌은
    #   Fabrics body_repulsion 이 계획 단계에서 이미 회피하므로 팔은 껍질로 충분하다.
    #   ★손까지 hull 로 하면 접촉력이 4배(133N) → 촉각 왜곡으로 s2r 이 깨진다. 금지.
    #   자산은 physics 레이어만 교체한 얇은 변형(40KB, base 는 원본 심볼릭 링크).
    usd_relpath="robot/openarm_tesollo_sensor_rl_armhull/openarm_tesollo_sensor_rl.usd",
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
    envelope_fingers=("jaw1", "jaw2"),   # 2지 그리퍼는 양 jaw 접촉이 곧 감쌈
    # ★미정의 — 실측 전까지 비워 둔다. jaw 링크 국소 프레임에서 "무는 면"이 어느 축인지
    #   확인되지 않았고, 추측값을 넣으면 판정이 **조용히 뒤집힌다**(손등을 손바닥으로).
    #   이 프로필은 fabric_class=None 이라 어차피 등록에서 SKIPPED 되지만, Phase 2 에서
    #   되살릴 때 반드시 실측할 것: jaw1/jaw2 body_quat 을 읽고 서로를 향하는 축을 본다.
    #   env 부팅이 fail-loud 로 막는다(_palmar_axes).
    palmar_axis_local={},
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
)


PROFILES: dict[str, RobotProfile] = {
    p.name: p for p in (TESOLLO_RIGHT, RH56F1_RIGHT, GRIPPER_LEFT)
}
