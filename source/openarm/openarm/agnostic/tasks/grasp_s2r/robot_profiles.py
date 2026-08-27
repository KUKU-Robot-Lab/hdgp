"""로봇 프로필 — robot-agnostic grasp-sensor 태스크에서 **로봇 종속 정보가 모이는 유일한 곳**.

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
    # palm 목표 워크스페이스(env-local 절대). 액션 누산 결과를 여기로 clamp 한다.
    palm_box_min: tuple = (0.0, 0.0, 0.0)
    palm_box_max: tuple = (0.0, 0.0, 0.0)
    palm_rot_center_deg: tuple = (90.0, 0.0, 90.0)   # euler_zyx (ez, ey, ex)
    palm_rot_half_deg: float = 45.0
    palm_box_verified: bool = False      # probe 로 도달성 확인했는가

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
    # 접촉 시 동결할 관절 접미사 — 그 손가락의 원위∨팁 접촉이 성립하면 진행을 멈춘다.
    #   ★이것이 감쌈 생성 메커니즘이다. 풀면 손가락이 컵 반경보다 작게 말려 손끝만
    #     닿는 핀치가 된다(grasp_v1 실증: full_envelope 0.176→0.035).
    hand_freeze_suffixes: tuple = ()

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
    hand_freeze_suffixes=("3", "4"),
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
        "right_arm_proximal": dict(joint_names_expr=["r_aj_[1-4]"], stiffness=300.0, damping=45.0,
                                   effort_limit_sim=300.0),
        "right_arm_j5":       dict(joint_names_expr=["r_aj_5"],     stiffness=100.0, damping=20.0,
                                   effort_limit_sim=300.0),
        "right_arm_j6":       dict(joint_names_expr=["r_aj_6"],     stiffness=50.0,  damping=15.0,
                                   effort_limit_sim=300.0),
        "right_arm_j7":       dict(joint_names_expr=["r_aj_7"],     stiffness=25.0,  damping=15.0,
                                   effort_limit_sim=300.0),
        # 손: Kuka allegro kp 3.0 / kd 0.1 / effort 0.5. 우리 구 값은 5.0/2.0/1.5 로
        #   kd/kp 가 0.40 vs Kuka 0.033 = **12배 과감쇠**였다. 오늘 "닫는 속도가 컵을
        #   쳐낸다"를 명령 변화율(synergy_close_speed)로 맞췄는데, 원본은 게인으로 만든다.
        "hand":               dict(joint_names_expr=["r_hj_[a-z]+_[1-4]"], stiffness=3.0, damping=0.1,
                                   effort_limit_sim=0.5),
        "left_arm":           dict(joint_names_expr=["l_aj_[1-7]"], stiffness=400.0, damping=80.0),
        "left_gripper":       dict(joint_names_expr=["l_hj_gripper_[1-2]"], stiffness=400.0, damping=80.0),
        "head":               dict(joint_names_expr=["head_j_(pan|tilt)"], stiffness=400.0, damping=80.0),
    },
    # ★★08.27 grasp_s2r: 구 (0.30, −0.20) → (0.362, −0.16).
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
    p.name: p for p in (TESOLLO_RIGHT, GRIPPER_LEFT)
}
