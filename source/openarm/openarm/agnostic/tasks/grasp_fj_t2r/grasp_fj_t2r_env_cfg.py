"""grasp_fj_t2r cfg — Track B leaf(`GraspFJTesolloRightShortEnvCfg`: short · thumb_1 용접)에
생성 보상 코드 경로만 더한다.

관측·액션·종료·성공 판정·커리큘럼·자산은 전부 B leaf 그대로다. B 의 보상 계수(`rw_*`, g(q))는
에피소드 상태(lifted 래치·최단거리 추적기)를 만드는 데만 쓰이고 **총보상에는 쓰이지 않는다** —
총보상은 `reward_code_path` 의 `compute_reward(ctx)` 가 낸다.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ..grasp_fj.grasp_fj_env_cfg import GraspFJTesolloRightShortEnvCfg


@configclass
class GraspFJT2RRightShortEnvCfg(GraspFJTesolloRightShortEnvCfg):
    """text2reward 생성 보상판. 보상 코드는 런처가 `env.reward_code_path=<절대경로>` 로 준다."""

    #: 생성 보상 코드(`compute_reward(ctx)`) 절대경로. 비면 **영 보상** — 부팅·무작위 롤아웃 전용.
    #:   env 가 `__init__` 에서 읽는다(런타임) — hydra override 가 `__post_init__` 에 구워지는 함정이 없다.
    reward_code_path: str = ""
    #: ★09.15 사용자 "시작 상태 커리큘럼 + env 고정" — 리셋 중 이 비율을 컵 옆 IK 자세에서 시작한다(0 = 끔). reach leaf 만 켠다.
    near_start_frac: float = 0.0
    #: 공통 스텝이 이 값 이하일 때는 가까운 출발을 쓰지 않는다 — B 부팅 시작 거리 가드(`common_step_counter <= 4`)가 먼 출발만 본다.
    near_start_after_common_steps: int = 4
    #: 가까운 출발 팔 관절(r_aj_1..7) 표 — 행 순서 = `near_start_species` = 물체 뱅크 순서(env_id % N). env 가 부팅에서 대조한다.
    near_start_species: tuple = ()
    near_start_arm_q: tuple = ()


@configclass
class GraspFJT2RReachEnvCfg(GraspFJT2RRightShortEnvCfg):
    """★최종 목표판(09.14 사용자): 팔·손을 **테이블 앞 가장자리 위**에서 시작해 **여러 모양의 컵**으로 접근 → 파지 → 들기.

    위 판과 다른 것 넷 — 전부 사용자 결정(09.14):
      · 시작 자세 — palm (0.05, −0.30, 0.45): 상판 앞 가장자리(x 0.07) 바깥, 상판 위 0.245 m. 회전은 short_tl 시작 자세와 같다.
        `urdf/tools/solve_arm_reset_pose.py` 의 Urdf·잔차로 short-tl URDF 에서 6D IK — 위치·회전 오차 0 · 관절한계 여유 11.4° ·
        |q−홈| 1.464 rad(r_aj_1, 검증기 1.5 이내) · 손 링크 최저 z 0.40 · palm→컵 중심 0.369–0.379 m.
        ★손가락이 +x 로 뻗어 손끝이 상판 위 x 0.25 까지 걸친다. palm 을 x −0.05/−0.10 으로 물리면 |q−홈| 1.66/1.53 rad(가드 초과),
          −0.15 는 IK 미수렴이라 같은 손 방향으로는 손 전체를 바깥에 둘 수 없다 — 사용자 확인 후 x 0.05 채택.
      · 팔 속도 2배 — k_arm 0.05 · dofSpeedScale 3.0 · slew 0.3 rad/s(검증기가 셋을 같이 대조한다). 컵까지 거리가 0.16→0.37 m.
      · 에피소드 15 s(900 스텝) — 0.15 rad/s·10 s 로는 이동만 ≥452 스텝이었다.
      · 물체 `cup_family` — 입구 열린 컵 cup_big 7크기(파지 반경 52.7–80.6 mm) + 닫힌 shaker_closed(44 mm). 전부 `baseLink`
        강체라 `_cup_contact_filter` 가 `Object/baseLink` 를 찾는다(셰이커 `shaker.usda` 는 루트가 강체라 섞을 수 없다).
    """

    arm_reset_joint_pos_override: tuple = (-1.1974, 0.6707, 0.1866, 1.7310, 0.6920, 0.0416, 0.9460)
    #: palm→컵 중심 IK 0.369–0.379 m(원점 높이가 다른 컵별) — 부팅 가드는 리셋 직후 env 평균을 본다.
    start_palm_dist_band_m: tuple = (0.30, 0.46)
    k_arm: float = 0.05
    arm_dof_speed_scale: float = 3.0
    arm_slew_rad_s: float = 0.3
    episode_length_s: float = 15.0
    object_bank: str = "cup_family"
    #: ★09.15 23:2x 사용자 "시작 상태 커리큘럼 + env 고정 · 50 % · C자 사전파지" — 두 번째 에피소드부터 절반을 컵 옆에서 시작한다.
    #:   자세 = `urdf/tools/solve_arm_reset_pose.py`(short-tl URDF, 시작 자세와 같은 회전, rot-weight 0.3) 6D IK — 위치 오차 7종 0 ·
    #:   s085 0.2 mm(팔꿈치 r_aj_4 하한) · 회전 오차 ≤ 0.01° · 나머지 관절한계 여유 17.6–39.5°. palm_ee 목표 = 스폰 중심
    #:   (0.362, −0.16) 에서 x −(R+0.025) · y −(R+0.045) · z = 0.205 + 원점 오프셋 + 0.8·반높이 → C자 완료
    #:   (손바닥면 −1~+2 cm · 컵 축 R−0.5~R+2 cm)보다 손바닥면 4.5 cm · 컵 축 2.5 cm 앞.
    #:   ★09.15 서버 스모크: 손바닥면 3 cm 판은 컵 스폰 −2 cm 흔들림에 1 cm 가 돼 32 중 4 env 가 첫 스텝에 공짜 접근 래치 → 4.5 cm
    #:   (흔들려도 ≥ 2.5 cm, 창 밖). 컵 축 ≥ R+0.5 cm 라 엄지와 겹치지 않는다. 띠 중심 높이(0.5 H 이하)는 작은 컵에서 팔꿈치 한계로
    #:   IK 미수렴(최대 29 mm)이라 0.8 H. `tests/test_grasp_gates.py` 가 FK 로 대조한다.
    #:   ★09.16 사용자 "가까운 출발 = 접근 완료로 시작" — env 가 리셋에서 접근 래치를 세운다(2단계부터). 창 밖 4.5 cm 는 겹침 여유로만 남는다.
    near_start_frac: float = 0.5
    near_start_after_common_steps: int = 4
    near_start_species: tuple = ("cup_big_s085", "cup_big_s100", "cup_big_s115", "cup_big_s130", "shaker_closed",
                                 "cup_big_s090", "cup_big_s105", "cup_big_s120")
    near_start_arm_q: tuple = (
        (0.3906, 0.3075, -0.5595, 0.0000, 0.5611, 0.3050, 1.1696),      # cup_big_s085
        (0.1150, 0.4313, -0.2859, 0.5365, 0.4627, 0.2277, 0.9178),      # cup_big_s100
        (-0.0054, 0.5243, -0.3017, 0.7565, 0.5714, 0.1857, 0.8356),     # cup_big_s115
        (-0.0932, 0.6387, -0.3641, 0.9209, 0.7127, 0.1307, 0.8049),     # cup_big_s130
        (0.1320, 0.5151, -0.5643, 0.7116, 0.7427, 0.0664, 0.8426),      # shaker_closed
        (0.2411, 0.3814, -0.3960, 0.3075, 0.4886, 0.2463, 1.0271),      # cup_big_s090
        (0.0693, 0.4596, -0.2809, 0.6197, 0.4912, 0.2157, 0.8828),      # cup_big_s105
        (-0.0301, 0.5778, -0.3668, 0.8160, 0.6580, 0.1532, 0.8317),     # cup_big_s120
    )

    #: ★09.16 사용자 "테이블과 접촉하는 정책은 사용불가" — 상판 관통 종료를 부모 3 cm 에서 5 mm 로 조인다.
    #:   왜 0 이 아닌가: `floor_hit = hand_z_min < table_surface_z − depth` 라 0 은 PhysX 정상 접촉의 미세 관통에도
    #:   에피소드를 끊어 접근 자체를 죽인다(cfg 주석의 "0 이면 끔"은 옆 필드 `palm_box_min_z_override` 설명이다).
    #:   근거: i05 가 허용 슬랙 안쪽 2.8 cm 에 눌러앉았고, 관통이 라운드 최소(1.2 mm)였던 e950 에 리프트가 최고(0.296)였다.
    hand_floor_terminate_depth: float = 0.005
    #: ★09.16 사용자 "Task 성공 세팅 및 지표를 제대로 수정" — 감쌈 전제조건을 되살린다.
    #:   부모가 0.0(= 끔)이라 i05 의 성공에는 파지 조건이 **전혀** 없었다(성공 = kp_dist ≤ tol 10스텝 누적뿐).
    #:   사다리는 `prev_episode_successes ≥ 2.0` 에서만 750 프레임마다 ×1.10 오르므로, 성공이 희박한 동안은
    #:   임계가 고정이라 fj_h2 의 "0.627 잠김"은 재발하지 않는다. 관측 wrap_frac 0.59 > 0.15 라 즉시 붕괴도 없다.
    grasp_wrap_start: float = 0.15
