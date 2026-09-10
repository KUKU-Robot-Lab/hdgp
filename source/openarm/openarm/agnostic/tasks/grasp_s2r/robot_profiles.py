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
from dataclasses import replace as _dc_replace

from openarm.agnostic.modules import vendor_gains as _vg


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
    # ★09.08 **full-joint 손**(grasp_fj `hand_direct`)의 관절별 액션한계 override: 정규식 → (lo, hi).
    #   env 는 articulation soft limit 과 **교집합**만 취한다(좁히기만, 넓히기 불가). None = 그 끝은 그대로.
    #   왜 필요한가: SimToolReal 은 액션을 URDF 한계에 선형 매핑한다. SHARPA 는 PIP/DIP 하한이 0 이라
    #   그게 안전했지만 테솔로 `_3/_4` 는 ±1.571 **대칭**이라 a=−1 이 손등 −90° 를 지령한다 — 접촉
    #   항이 0개인 보상에서 손등 갈고리 파지가 정상 파지와 같은 점수를 받는다(08.23 실측 exploit).
    #   그래서 "URDF 한계 + `_3/_4` 하한 0" 이 사용자 확정값이다(09.08). 정규식이 아무 관절도 못
    #   잡으면 부팅이 죽고, 리셋 자세(init_joint_pos)가 범위 밖이어도 부팅이 죽는다.
    hand_action_limit_override: dict = field(default_factory=dict)

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
# tesollo_right — s2r 자산(openarm_dg5f-m_bi_rl, 09.05 라인업): DG-5F-M 우손 20 DOF
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
    # ★★2026-09-06 자산 교체: `openarm_tesollo_sensor_rl_hull` → `openarm_dg5f-m_bi_rl`.
    #   09.05 자산 4종 재생성으로 구 자산은 트리에서 사라졌다(경로가 죽어 부팅 불가였다).
    #   새 자산은 **좌우 모두 DG-5F-M 20 DOF** 라 좌팔이 그리퍼가 아니다 — 유휴 좌측
    #   actuator/init 이 손 20관절로 바뀐다(아래).
    #   ★콜라이더는 manifest 가 정한다: 팔·몸통·헤드 = convexHull, **손 = convexDecomposition**.
    #     이게 08.23 실측이 지목한 안전한 쪽이다 — 팔만 hull 로 처리량 +13.7%, 접촉력
    #     36.2→32.8N. **손까지 hull 로 하면 접촉력 133N** 으로 4배 뛰고 촉각 obs 가 죽는다
    #     (env 가 `contact = (force/5).clamp(max=4)` 라 20N 에서 포화 → 5채널 상수 4.0,
    #      감쌈 임계 `stage_contact_threshold=0.1N` 도 항상 참). 자산을 다시 만들 때 지키자.
    #   ★학습 로그에서 볼 것: `task/contact_*` 포화 · `task/wrap4` 가 1.0 에 붙는지 ·
    #     `task/deep4` 와의 괴리.
    usd_relpath="robot/openarm_dg5f-m_bi_rl/openarm_dg5f-m_bi_rl.usd",
    num_arm_joints=7,
    num_hand_joints=20,
    arm_joint_regex="r_aj_[1-7]",
    hand_joint_regex="r_hj_(thumb|index|middle|ring|pinky)_[1-4]",
    # index/middle/ring 의 _1 = 외전. grasp_v2 도 이 축들을 정책에서 뺐다.
    # ★thumb_1(대향 벌림)과 pinky_1(= Z-flex, 외전 아님 — tesollo pinky 운동학 메모)은
    #   파지에 필수라 자유 유지.
    # ★09.08 아래 두 필드는 **소비자 없음**(grep) — full-joint(hand_direct) 에서는 전 20관절이 정책 제어라 무효. 이력용.
    hand_locked_joint_regex="r_hj_(index|middle|ring)_1",
    num_locked_hand_joints=3,
    # 엄지 대향은 손가락열에서 떨어져 있다 — 침범 원인이 아니므로 넓게 연다.
    # ★09.08 full-joint 손 액션한계(사용자 확정 "URDF 한계 + _3/_4 하한 0"). 다섯 손가락 `_3/_4` 10개 전부
    #   굴곡 전용 [0, 1.571] — 엄지 `_3` 도 0 이다(사용자 재확정 09.08: "−0.5 면 꺾이는 자세"). 시너지 시절 open
    #   자세 −0.5(pre-curl)는 A/grasp_s2r 것이라 프로필 init 은 그대로 두고, **B 는 리셋에서 손 관절 상태를 액션한계로
    #   clamp 해 심는다**(grasp_fj `_reset_idx`, 이동량 상한 cfg `hand_reset_clamp_max_rad`) — 리셋 자세와 범위가
    #   어긋나 첫 스텝에 튀는 일이 없다. 외전 `_1`·`_2`(MCP) 는 URDF 전폭(외전 잠금 해제 = 09.08 사용자 결정).
    #   `thumb_2` 는 외전이 아니라 **대향**(URDF [−π, 0] 180°) — 사용자 확정(09.08): URDF 실제 범위 그대로.
    # ★★09.10 **관절 의미별** 액션 범위(사용자 확정). 자산 한계는 매뉴얼 = 실기 진실이고,
    #   여기서 정하는 것은 "정책이 명령할 수 있는 범위"다. 두 층은 다르다.
    #     thumb_1  엄지 자체를 **돌리는** 축(대향이 아니다) → 매뉴얼 전폭
    #     thumb_2  **대향** 관절 — 손바닥을 움직이는 것과 같다. 인벨롭의 핵심 → 매뉴얼 전폭
    #     thumb_3/4 조이는 관절 → 굴곡 전용(하한 0)
    #     index~ring _1  외전 — **열수록 손가락끼리 충돌한다**(사용자 반복 확정) → ±0.16 유지
    #     pinky_1  손바닥 쪽으로 마는 관절(사실상 손바닥 관절) → 매뉴얼 전폭. 인벨롭에 쓰인다.
    #              ⚠구 코드가 이걸 외전과 한 규칙으로 묶어 ±9.2° 로 잠가 뒀다(주석에 의심만 적혀
    #                있었다). 손바닥 오목을 만드는 자유도를 통째로 막고 있었던 셈이다.
    #     pinky_2  손가락을 **벌리는** 관절(검~약지 _1 과 같은 성격) → 같은 이유로 ±0.16
    #     pinky_3/4 다른 손가락과 같은 맥락 → 굴곡 전용
    # ★★09.10 인벨롭 파지 자세의 **부호 규약**을 액션 한계로 박는다. 이상한 자세가
    #   애초에 도달 불가능해야 한다(사용자 확정). 출처는 Isaac Sim lula kinematics 실측
    #   이고 우리 자산 URDF FK 로 재확인했다(엄지끝↔검지·중지·약지끝 거리로 판정).
    #
    #     우손:  _1 = 0 유지 · _2 thumb − / index·middle·ring + / pinky 0~+ · _3 전부 + · _4 전부 +
    #     좌손:  _1 = 0 유지 · _2 thumb + / index·middle·ring + / pinky −~0 · _3 thumb − 나머지 + · _4 동일
    #
    #   ⚠**좌우 부호 규약이 다르다.** 미러는 링크 origin 의 rpy 에 있고 관절 **축 벡터는
    #     좌우가 같다** — `_3/_4` 는 한계도 ±90° 대칭이라 축과 한계표만 보면 좌우가 같아
    #     보인다. 부호를 표에서 추론하지 말고 FK 로 재라(09.10 에 그렇게 추론했다가 우손
    #     엄지 부호를 반대로 판정했다). 검증 도구는 `urdf/tools/` 의 FK 대조를 쓴다.
    #   ★`_1` 을 정확히 0 으로 묶으면 폭 0 액션 칸이라 부팅 가드가 죽인다 — ±0.01 rad
    #     (0.57°)로 둔다. 리셋 자세가 전부 0.0 이라 clamp 이동은 없다.
    hand_action_limit_override={
        r"r_hj_(thumb|index|middle|ring|pinky)_[34]$": (0.0, None),
        r"r_hj_(thumb|index|middle|ring|pinky)_1$": (-0.01, 0.01),
        r"r_hj_pinky_2$": (0.0, None),
    },
    palm_body="r_hl_palm",
    # ---- Fabrics (DG-5F 계보) ----
    fabric_class="OpenArmTeoslloPoseFabric",
    # ★FK 게이트 0.0um 로 sensor_rl 에서 재생성한 자산(08.22). 레거시 openarm_tesollo /
    #   openarm_tesollo_sensor 는 같은 DG-5F 손이지만 팔 베이스가 +8mm 어긋나
    #   RL URDF 대비 worst 17.93mm 였다.
    # ★09.06 dg5f-m 자산으로 이관. 구 `openarm_tesollo_sensor_right` 와 **완전 드롭인**:
    #   링크 이름 104개·구동관절 27개 순서·body_repulsion 충돌구 프레임 65개가 전부 일치한다
    #   (부팅 전 대조 완료). 그래서 `fabric_joint_order` 와 아래 매핑은 손대지 않는다.
    fabric_robot_dir="openarm_dg5f-m_bi_right",
    # ★전용 params — 공유 openarm_tesollo_pose_params.yaml 을 쓸 수 없다(08.23).
    #   자매 트랙이 손가락 충돌 구를 실측 형상으로 재배치(반경 9mm·마디 방향)했는데,
    #   구 개수는 **마디 길이 ÷ 지름**으로 자동 산출되고 sensor_rl 자산은 bi_s 와
    #   링크 길이가 달라 39개 vs 52개로 갈린다. 공유 yaml 을 그대로 쓰면 우리 URDF 에
    #   없는 `dg_1_2_sph3` 를 찾다 KeyError 로 부팅이 죽는다(실측). 반대로 공유 yaml 을
    #   덮어쓰면 bi_s 트랙이 깨진다 — 그래서 frames/radii 만 우리 것으로 바꾼 사본을 쓴다.
    #   쌍(collision_link_prefix_pairs)은 **접두사 매칭**이라 구 개수와 무관해 그대로다.
    fabric_params_filename="openarm_dg5f-m_right_pose_params.yaml",
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
        0.0, -1.57, 0.0, 0.0,     # thumb — _2 는 opposition 으로 고정(양 자세 동일),
        0.0,  0.0,   0.0, 0.0,    #   ★09.09 구 `_3 −0.5` pre-curl 폐기 — 음수는 손등 과신전
                                  #   방향이라 자산 한계 [0,1.05] 밖이다(도달 불가).
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
        # ★09.09 `thumb_3` −0.5 → 0.0. 자산 한계가 [0, 1.05] 로 좁혀져(음수 = 손등 과신전)
        #   −0.5 는 **도달 불가능**해졌고, IsaacLab 이 부팅에서 기본자세를 한계와 대조해
        #   거부한다(fj_e 3런 동시 크래시: "default positions out of the limits").
        #   구 값의 의도(open pre-curl, 밑마디가 먼저 닿는 것 방지)는 음수 방향이라 애초에
        #   물리적으로 불가능한 자세였다. B 트랙은 이미 런타임에서 0 으로 clamp 해 심고 있었다.
        "r_hj_thumb_1": 0.0, "r_hj_thumb_2": -1.57, "r_hj_thumb_3": 0.0, "r_hj_thumb_4": 0.0,
        **{f"r_hj_{f}_{j}": 0.0 for f in ("index", "middle", "ring", "pinky") for j in (1, 2, 3, 4)},
        # 유휴 좌팔(파지 팔 홈의 부호 미러, DG-5F IK 실측 — grasp_sensor preset 승계)
        "l_aj_1": -0.0431, "l_aj_2": -0.6706, "l_aj_3": -0.0961, "l_aj_4": 0.7342,
        "l_aj_5": -0.3750, "l_aj_6": -0.5678, "l_aj_7": -0.6709,
        # 유휴 좌손(dg5f-m). ★엄지 대향은 **좌우 부호가 반대**다 — 자산 실측:
        #   `r_hj_thumb_2` [-3.142, 0] vs `l_hj_thumb_2` [0, +3.142]. 우 -1.57 의 거울은 +1.57 이고
        #   0.0 을 넣으면 관절 하한에 붙는다. 나머지 19관절은 0.0 이 전부 범위 안이다.
        "l_hj_thumb_2": 1.57,
        **{f"l_hj_{f}_{j}": 0.0 for f in _FINGERS for j in (1, 2, 3, 4)
           if not (f == "thumb" and j == 2)},
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
        # ══ 2026-09-06 사용자 확정: 팔 PD 게인은 **벤더값만** ═══════════════════
        #   숫자는 `modules/vendor_gains` 하나에서만 온다(벤더 control_gains.yaml).
        #   아래 히스토리 주석(KUKA 전환·r2s 재식별·HDGP_S2R_REAL_GAINS 스위치)은
        #   그 결정으로 **전부 대체됐다** — 기록으로만 남긴다. 게인을 바꾸려면 벤더
        #   yaml 을 고친다. ⚠동특성이 달라지므로 기존 체크포인트와 비호환(FRESH 전용).
        # ★09.06 `effort_limit_sim=300.0` 삭제. 벤더 한계는 40/40/27/27/7/7/7 N·m 라
        #   손목에서 **42배** 넘는 토크로 학습하고 있었다. 자산 URDF/USD 가 이미 벤더값을
        #   담고 있으므로(joint limit effort) **지정하지 않는 것이 곧 벤더값**이다.
        #   ⚠우 j7 은 정적 자세만으로 3.17 N·m(한계 7 의 45%)를 연속으로 문 이력이 있다.
        **_vg.arm_actuators("right_arm", "r", friction=0.0),
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
        # 손 게인 = DG-5F 벤더 PID(2026-09-06).
        # ★★09.08 `effort_limit_sim=1.5` **삭제** → USD 드라이브의 **벤더 7.5 N·m** 이 산다
        #   (`<limit effort="7.5">`, CAD 릴리스·드라이버 사본 둘 다 20관절 전부).
        #   1.5 는 08.19 A4 가 "7.5 레짐은 thumb_1 하드스톱 밖 −0.94rad" 를 보고 **증상을
        #   덮으려고** 내린 값이었다. 덮기에 실패했다는 것이 09.08 계측으로 확정됐다 —
        #   같은 관절이 접촉 중 **−4.075 rad**(한계 밖 3.69 rad)까지 밀려난다. 원인은 토크
        #   상한이 아니라 자산 쪽(손 collider 가 벤더 `convex_hull` 이 아니라 decomposition)이라,
        #   증상을 덮던 비벤더 값을 먼저 걷어내고 진짜 원인을 고친다.
        #   계측: `scripts/analysis/fj_joint_limit_viol.py`(baseline max_viol 3.74 ± 0.09 rad).
        **_vg.hand_actuator("hand", ["r_hj_[a-z]+_[1-4]"]),
        **_vg.arm_actuators("left_arm", "l"),          # 유휴측도 벤더 게인(같은 로봇이다)
        # 유휴 좌손도 같은 로봇이므로 벤더 DG-5F PID 를 쓴다(자산에 그리퍼는 없다).
        **_vg.hand_actuator("left_hand", ["l_hj_[a-z]+_[1-4]"]),
        # head 는 Dynamixel 이라 벤더 팔 파일에 없다(`vendor_gains.NO_VENDOR_PD`). 현행값 유지.
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
# tesollo_right_short — DG-5F **short base** 판(openarm_dg5f-m-short_bi_rl).
#
# 같은 DG-5F 손인데 mount/base 구간이 짧다: `*j_dg_palm` z 0.0698 → 0.022, 즉
# **손바닥이 플랜지 쪽으로 47.8mm 당겨졌다.** 손가락 체인은 이름·원점·축이
# TESOLLO_RIGHT 와 완전히 동일하고 manifest 의 control_joint_order(54)·link_order(84)·
# source_to_canonical_joints 도 바이트 동일하다. 그래서 이 프로필은 `_dc_replace` 로
# **바뀐 것만** 덮어쓴다 — 손 관련 필드를 복제하면 두 벌이 조용히 갈라진다.
#
# ★자산 출처가 dg5f-m 과 다르다: 이쪽은 `vendor/tesollo_model/dg5f`(CAD 릴리스, 사용자
#   지시)이고 dg5f-m 은 `vendor/delto_m_ros2`(드라이버 사본)다. 결과로 **손 질량이
#   1.763 → 1.570 kg**(CAD 는 질량·관성이 정합, 드라이버 사본은 라운드 임시값)이고
#   **9개 관절의 한계가 다르다**(예: `_hj_thumb_1` 상한 0.890 → 1.344 rad,
#   `_hj_pinky_2` 상한 0.611 → 1.571 rad). 아래 `hand_open_pose`/`hand_grip_pose`/
#   `init_joint_pos` 값은 전부 새 한계 안이라 그대로 승계된다(확인 완료).
#   ⚠실기 첫 구동 전 그 9개 관절의 실제 가동범위를 확인할 것 — 드라이버가 배포하는
#     description 보다 넓게 지령할 수 있다.
#
# ★팔 홈은 **재산출했다**(복사 아님). 손이 47.8mm 짧으므로 dg5f-m 의 관절값을 그대로
#   쓰면 palm 이 47.3mm 어긋난다. dg5f-m 홈의 palm 월드 포즈를 목표로 IK 를 풀었고
#   (수렴 0.0002mm / 0.0000°, 관절한계 여유 0.136 rad), 그 결과 **palm 이후 손 전 프레임이
#   dg5f-m 홈과 0.023mm 이내로 일치**한다(손끝 10개 실측). 그래서 손 기하에 종속인
#   `object_spawn_center`(홈 케이지)·`finger_sensor_bodies`·`palmar_axis_local` 은
#   재측정 없이 유효하다. 달라지는 것은 `*_hl_base` 뿐(47.80mm).
#
# ⚠`palm_box` 는 **미검증**이다. palm 이 손목에 가까워졌으므로 같은 palm 목표를 내려면
#   팔이 그만큼 더 뻗어야 한다 — 박스의 도달 가능 영역이 줄어든다. 방향(palm_rot_center)
#   은 IK 가 자세까지 맞췄으므로 그대로 유효하다. 학습 전 probe_boxreach / probe_taskreach
#   로 재확인하고 `palm_box_verified=True` 로 승격할 것.
# =============================================================================
TESOLLO_RIGHT_SHORT = _dc_replace(
    TESOLLO_RIGHT,
    name="tesollo_right_short",
    usd_relpath="robot/openarm_dg5f-m-short_bi_rl/openarm_dg5f-m-short_bi_rl.usd",
    fabric_robot_dir="openarm_dg5f-m-short_bi_right",
    fabric_params_filename="openarm_dg5f-m-short_right_pose_params.yaml",
    palm_box_verified=False,
    init_joint_pos={
        **TESOLLO_RIGHT.init_joint_pos,
        # 우팔: dg5f-m 홈 palm 포즈를 목표로 재산출(IK 오차 0.0002mm / 0.0000°)
        "r_aj_1": 0.2667, "r_aj_2": 0.4487, "r_aj_3": 0.4923, "r_aj_4": 0.7184,
        "r_aj_5": -0.0460, "r_aj_6": 0.6496, "r_aj_7": 0.4762,
        # 유휴 좌팔도 같은 방식으로 재산출(IK 오차 0.0000mm / 0.0000°)
        "l_aj_1": -0.3610, "l_aj_2": -0.6357, "l_aj_3": 0.0322, "l_aj_4": 0.4330,
        "l_aj_5": -0.2661, "l_aj_6": -0.5842, "l_aj_7": -0.7274,
    },
)


# =============================================================================
# rh56f1_right — Inspire RH56F1 우손. **물리 12관절 중 구동 6**(언더액추에이션).
#
# ★이 프로필은 **Track B(`grasp_fj_rh`) 전용**이다. fabric 을 쓰는 트랙
#   (grasp_s2r / grasp_kp)으로는 못 띄운다 — `fabric_class=None` 이라 부팅에서 죽는다.
#   여기 두는 이유는 이 파일이 로봇 레지스트리의 단일 출처이기 때문이다(파일 상단 설계
#   목표: "새 로봇 추가 = 이 파일에 프로필 1개 추가"). 부모 cfg 의
#   `finalize_after_overrides` 가 이 모듈의 `PROFILES` 를 읽으므로 다른 곳에 두면
#   조용한 KeyError 가 된다.
#
# ★언더액추에이션은 **자산이 들고 있다**(09.05 재생성본 USD 실측):
#     구동 6 = thumb_1(외전) · thumb_2(굴곡) · {index,middle,ring,pinky}_1
#              → PhysicsDriveAPI 보유
#     종속 6 = thumb_3(=thumb_2×1.1425) · thumb_4(=thumb_3×0.7508)
#              · {index,middle,ring,pinky}_2(=같은 손가락 _1 × 1.1169)
#              → **DriveAPI 없음** · physxMimicJoint(naturalFrequency 200 · dampingRatio 1.0)
#   09.02 씬 폭발(1024 env 중 400개 동시 위반)의 원인이던 "종속 한계가 좁아 mimic 제약과
#   관절한계가 동시에 만족 불가"는 자산 쪽에서 고쳐졌다 — 종속 한계가 리더 오버슈트
#   0.5 rad 여유로 넓혀져 있다(thumb_3 −0.571~+1.113 rad). 그래서 software mimic 을
#   이식하지 않는다. 종속관절 액추에이터는 **0/0** 으로 덮어 제약과 싸우지 않게 한다
#   (IsaacLab 은 전 관절 액추에이터 커버리지를 요구한다 — articulation.py:1769).
#
# 출처: assets/robot/openarm_rh56f1_bi_rl/*(URDF·USD 실측) + 09.02~09.03 grasp_ua 캘리브.
# =============================================================================
_RH_FLEX = ("index", "middle", "ring", "pinky")

RH56F1_RIGHT = RobotProfile(
    name="rh56f1_right",
    usd_relpath="robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.usd",
    num_arm_joints=7,
    num_hand_joints=6,
    arm_joint_regex="r_aj_[1-7]",
    # ★구동 6개만. 종속 6개는 정책이 보지도 몰지도 않는다(PhysX mimic 이 몬다).
    hand_joint_regex="r_hj_(thumb_[12]|index_1|middle_1|ring_1|pinky_1)",
    palm_body="r_hl_palm_sensor",
    # ---- Fabrics 없음(Track B 전용) ----
    fabric_class=None,
    fabric_joint_order=(),          # 비우면 env 가 cat([arm, hand]) 순서를 쓴다
    # ---- 시너지 그립 ------------------------------------------------------------
    hand_joint_names=(
        "r_hj_thumb_1", "r_hj_thumb_2",
        *tuple(f"r_hj_{f}_1" for f in _RH_FLEX),
    ),
    # ★thumb_1 은 **외전(대향)** 이지 굴곡이 아니다. open 1.57 = 벌려서 컵이 들어올 gap 을
    #   만들고, grip 1.20 으로 내려 컵을 받친다 — open→grip 이 **감소**한다(lerp 는 부호 무관).
    #   09.03 실측: 엄지-4지 법선 간극은 1.57 에서 83.7mm 로 최대이고 2.09 로 더 벌리면
    #   73.9mm 로 오히려 줄어든다.
    hand_open_pose=(1.57, 0.00, 0.00, 0.00, 0.00, 0.00),
    hand_grip_pose=(1.20, 0.24, 1.08, 1.08, 0.85, 0.85),
    # 접미사 → 채널. RH56F1 은 접미사가 `1`(엄지 외전 / 4지 굴곡)과 `2`(엄지 굴곡)뿐이다.
    hand_channel_of_joint={"1": 0, "2": 1},
    # ★per_finger 레이아웃 전용 — 6슬롯이 구동 6관절과 1:1. 채널 레이아웃(2×5=10)을 쓰면
    #   4지의 채널1 이 아무 관절에도 안 걸려 액션 4개가 그냥 죽는다.
    hand_finger_channels={
        "thumb": {"1": 0, "2": 1},
        "index": {"1": 2},
        "middle": {"1": 3},
        "ring": {"1": 4},
        "pinky": {"1": 5},
    },
    # 손가락당 구동관절이 하나라 tesollo 식 "닿은 마디만 정지" 드레이프가 구조적으로
    # 불가능하다 — 감쌈은 하드웨어 결합(_2 = _1 × 1.1169)이 만든다.
    hand_freeze_suffixes=("1", "2"),
    # ---- palm 워크스페이스 (Track B 는 제어에 안 쓴다 — 홈 검사·앵커 부트스트랩용) ----
    # ★z 하한 0.26: palm 원점~최하단 링크가 114mm 라 palm z 가 0.314 밑이면 손이 테이블
    #   안이다(09.02). 0.30 은 케이지가 컵 높이에 못 내려가 0.26 으로 내렸다(09.03).
    palm_box_min=(0.20, -0.55, 0.26),
    palm_box_max=(0.65, 0.22, 0.65),
    # ★09.02 캘리브: 태스크 기준자세(손바닥 +y · 손가락 +x)가 이 프레임에선 짐벌락(ey −90°)
    #   이라 손가락을 25° 내려 특이점에서 뗐다. 선행 트랙 프리셋 (180,0,90)은 손가락이
    #   **위**를 향해 도달 불가였다(130mm/51° 포화).
    palm_rot_center_deg=(0.0, -65.0, -90.0),
    palm_rot_half_deg=45.0,
    palm_box_verified=False,
    # ---- 접촉 링크 (Track B 는 센서를 만들지 않는다 — 손가락 목록·손끝으로만 쓰인다) ----
    # 규약: (중간, 원위, 팁). RH56F1 4지는 2마디라 (_1, _2, _tip), 엄지는 원위 2마디가
    # 종속이라 (_3, _4, _tip).
    finger_sensor_bodies={
        "thumb": ("r_hl_thumb_3", "r_hl_thumb_4", "r_hl_thumb_tip"),
        **{f: (f"r_hl_{f}_1", f"r_hl_{f}_2", f"r_hl_{f}_tip") for f in _RH_FLEX},
    },
    contact_group_a=("thumb",),
    contact_group_b=_RH_FLEX,
    envelope_fingers=("thumb", *_RH_FLEX),
    # URDF 실측 유도 — cross(굴곡축, 장축), 마디 링크 국소 프레임:
    #   4지: 축 (0,0,-1) · 장축 ≈ +y → (+1,0,0)
    #   엄지: 축 (0,0,-1) · 장축 (−0.602, 0.799, 0) → (0.799, 0.602, 0)
    palmar_axis_local={
        "thumb": (0.799, 0.602, 0.0),
        **{f: (1.0, 0.0, 0.0) for f in _RH_FLEX},
    },
    fingertip_bodies=("r_hl_thumb_tip", *tuple(f"r_hl_{f}_tip" for f in _RH_FLEX)),
    init_joint_pos={
        # ★★09.02 캘리브(probe_ua_home_calib) — 목표 palm (0.310, −0.300, 0.419 /
        #   ez 0° ey −65° ex −90°) 로 수렴한 관절값. 09.07 재확인: 새 자산에서도 그대로 나온다.
        # ★★★09.08 되돌림 — 접근거리를 케이지 반경의 2.7배 → 1.6배로 당겼던 3차·4차 홈을
        #   **폐기**한다. 근거(무액션 실측): 그 홈에서는 **아무 액션 없이도** 물체가 21° 기울고
        #   7~23mm 떠오른다 — 열린 손가락이 케이지 중심에서 53mm 밖으로 뻗어 있어, 케이지가
        #   62mm 앞에 서면 손끝이 물체 반경(28.6mm) 안으로 들어간다(62 − 53 = 9mm < 28.6).
        #   즉 리셋 순간 손과 물체가 겹친다. 부팅 검사는 "xy 간격 > 케이지 반경"만 보고
        #   **물체 반경을 안 본다** — 그래서 조용히 통과했다(영상에서 셰이커가 상판에 박혀 보였다).
        #   구 홈은 xy 간격 106mm 라 106 − 53 = 53mm > 28.6mm 로 여유가 있다.
        #   ⇒ 접근거리 비율(tesollo 1.57)을 맞추려던 시도는 **기각**. 손이 작으면 손가락이
        #     상대적으로 더 튀어나오므로 같은 비율이 같은 여유를 뜻하지 않는다.
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
        # 유휴 좌손 — 좌우 한계가 같아(둘 다 하한 0) 0 이 전부 범위 안이다(URDF 실측).
        "l_hj_thumb_1": 0.0, "l_hj_thumb_2": 0.0,
        **{f"l_hj_{f}_1": 0.0 for f in _RH_FLEX},
        "l_hj_thumb_3": 0.0, "l_hj_thumb_4": 0.0,
        **{f"l_hj_{f}_2": 0.0 for f in _RH_FLEX},
        "head_j_pan": 0.0, "head_j_tilt": 0.0,
    },
    actuator_specs={
        **_vg.arm_actuators("right_arm", "r"),
        **_vg.arm_actuators("left_arm", "l"),         # 유휴측도 벤더 게인(같은 로봇이다)
        # ★★손 게인 5.0/2.0 — 벤더 PD 가 **없다**(`vendor_gains.NO_VENDOR_PD["rh56f1_hand"]`:
        #   RS-485 위치 서보라 PD 개념이 없다). 자산 USD 는 fallback 100/1 이지만
        #   이 관절들의 effort 는 1 N·m 라 kp 100 이면 오차 0.01 rad 에서 포화해
        #   사실상 정토크원이 된다 — 접촉이 들어오면 관절이 역구동돼 한계 밖으로 밀린다
        #   (09.02 실측 hand_joint_err_max 1.94 rad > 가동범위 1.53 rad → 씬 발산).
        #   kp 5 면 0.2 rad 에서 effort 에 닿아 선형 구간이 남는다. 파지력은 effort 가
        #   정하므로 줄지 않는다.
        "right_hand_drive": dict(
            joint_names_expr=["r_hj_(thumb_[12]|index_1|middle_1|ring_1|pinky_1)"],
            stiffness=5.0, damping=2.0),
        "left_hand_drive": dict(
            joint_names_expr=["l_hj_(thumb_[12]|index_1|middle_1|ring_1|pinky_1)"],
            stiffness=5.0, damping=2.0),
        # ★★종속 12 = **0/0**. 이 관절들은 USD 에 DriveAPI 가 없고 PhysX mimic 제약이
        #   위치를 정한다. 액추에이터를 두는 이유는 IsaacLab 이 전 관절 커버리지를
        #   요구하기 때문이고(articulation.py:1769), 게인을 0 으로 두는 이유는 드라이브가
        #   살아나면 mimic 제약과 싸우기 때문이다.
        "right_hand_mimic": dict(
            joint_names_expr=["r_hj_(thumb_[34]|index_2|middle_2|ring_2|pinky_2)"],
            stiffness=0.0, damping=0.0),
        "left_hand_mimic": dict(
            joint_names_expr=["l_hj_(thumb_[34]|index_2|middle_2|ring_2|pinky_2)"],
            stiffness=0.0, damping=0.0),
        "head": dict(joint_names_expr=["head_j_(pan|tilt)"], stiffness=400.0, damping=80.0),
    },
    # ★09.02 캘리브: 홈 케이지가 컵과 x 정렬되도록 역산(케이지는 palm 에 강체).
    #   ⚠자산 재생성 후 `probe_fjrh_calib.py` 로 재확인한다.
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
    # ★★2026-09-06 이 프로필은 **09.05 자산 라인업에서 고아가 됐다.** 필요한 것은
    #   "좌 2지 그리퍼 + 우 DG-5F 손" 혼합 로봇인데 새 4종에는 그런 자산이 없다
    #   (dg5f-m = 좌우 손, gripper = 좌우 그리퍼). 그나마 가까운 자산을 가리켜 두되,
    #   **되살리려면 아래 3가지를 새로 실측해야 한다**:
    #     ① 그리퍼 관절명: 새 자산은 `l_hj_gripper_[1-2]` 가 맞는지 확인 필요
    #     ② 조 body 명: 새 자산은 `l_hl_gripper_{left,right}_finger` 다(구 `_1/_2` 아님)
    #     ③ 유휴 우측: gripper 자산의 우측은 그리퍼라 `r_hj_[a-z]+_[1-4]` 가 없다
    #   `fabric_class=None` 이라 등록에서 SKIPPED 되므로 지금은 스폰되지 않는다.
    #   env 부팅의 regex 해석 대조가 fail-loud 로 다시 막는다.
    usd_relpath="robot/openarm_gripper_bi_rl/openarm_gripper_bi_rl.usd",
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
        # 게인=벤더값, friction=r2s 07.29 캘리브(마찰은 PD 게인이 아니라 벤더 규칙 밖).
        **_vg.arm_actuators("left_arm", "l", friction=_vg.R2S_FRICTION),
        "left_gripper":      dict(joint_names_expr=["l_hj_gripper_[1-2]"], stiffness=400.0, damping=80.0),
        **_vg.arm_actuators("right_arm", "r"),         # 유휴측도 벤더 게인(같은 로봇이다)
        # 손 게인 = DG-5F 벤더 PID(2026-09-06). effort 는 게인이 아니라 유지.
        **_vg.hand_actuator("right_hand", ["r_hj_[a-z]+_[1-4]"], effort_limit_sim=1.5),
        "head":              dict(joint_names_expr=["head_j_(pan|tilt)"], stiffness=400.0, damping=80.0),
    },
    object_spawn_center=(0.30, 0.20),    # 좌측 미러
)


# ★09.07 오른팔 전용 변형 — TESOLLO_RIGHT 와 **자산 경로만** 다르다.
#   왜: `replicate_physics=False`(다물체 필수)는 env 마다 articulation 을 통째로 PhysX 가
#       파싱한다. 우리는 오른팔만 학습하는데 왼팔 39링크·39관절이 env 마다 파싱된다.
#       16,384 env 실측에서 PhysX 가 요구한 접촉이 24,777,720(env 당 1,512)이었고 그게
#       기동 실패의 직접 원인이었다(버퍼를 키우자 요구치도 7.6M→24.8M 로 같이 늘었다).
#   자산은 base USD 를 subLayer 로 물고 왼팔 subtree 를 `active=false` 로 끄는 **오버레이**다.
#   메시를 다시 만들지 않는다. 실측 감소: prim 1,477→799 · 충돌형상 75→40 · 가동관절 56→29.
#   ★관절 이름·게인·palm 박스는 전부 TESOLLO_RIGHT 와 같다 — 오른쪽은 손대지 않았다.
#   ★체크포인트 비호환: articulation 관절 수가 바뀌므로 FRESH 학습이 필요하다.
def _drop_left(profile: RobotProfile, name: str, usd_relpath: str) -> RobotProfile:
    """좌측 관절을 `init_joint_pos`·`actuator_specs` 에서 뺀 변형을 만든다.

    ★자산에서만 빼면 안 된다 — 두 dict 에 남은 좌측 항목의 정규식이 매칭에 실패해
      IsaacLab 이 `Not all regular expressions are matched!` 로 부팅에서 죽는다(실측).
    head 는 자산에 남아 있으므로 그대로 둔다.
    """
    ijp = {k: v for k, v in profile.init_joint_pos.items() if not k.startswith("l_")}
    acts = {k: v for k, v in profile.actuator_specs.items()
            if not any(str(e).startswith("l_") for e in v.get("joint_names_expr", []))}
    return _dc_replace(profile, name=name, usd_relpath=usd_relpath,
                       init_joint_pos=ijp, actuator_specs=acts)


TESOLLO_RIGHT_ONLY = _drop_left(
    TESOLLO_RIGHT,
    "tesollo_right_only",
    "robot/openarm_dg5f-m_bi_rl/openarm_dg5f-m_bi_rl_right.usda",
)

# ★오른팔 전용 오버레이 판 — 왼팔 subtree 를 합성에서 끈 자산(`*_right.usda`).
#   dg5f-m 2,048 env 실측에서 처리량 +63%(24,155 → 39,448 fps)이고 물리 지표는 동일했다.
#   RH56F1 은 왼쪽 링크 35·관절 35 를 끈다(`scripts/tools/make_right_only_overlay.py`).
#   ★체크포인트 비호환 — articulation 관절 수가 바뀌므로 FRESH 학습이 필요하다.
RH56F1_RIGHT_ONLY = _drop_left(
    RH56F1_RIGHT,
    "rh56f1_right_only",
    "robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl_right.usda",
)

PROFILES: dict[str, RobotProfile] = {
    p.name: p for p in (TESOLLO_RIGHT, TESOLLO_RIGHT_ONLY, TESOLLO_RIGHT_SHORT,
                        RH56F1_RIGHT, RH56F1_RIGHT_ONLY, GRIPPER_LEFT)
}
