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
    #:   자세 = `urdf/tools/solve_arm_reset_pose.py`(short-tl URDF, 시작 자세와 같은 회전, rot-weight 0.3) 6D IK — 컵 8종 전부
    #:   위치·회전 오차 0 · 관절한계 여유 15.7–42.6°. palm_ee 목표 = 스폰 중심 (0.362, −0.16) 에서 x −(R+0.025) · y −(R+0.03) ·
    #:   z = 0.205 + 원점 오프셋 + 0.8·반높이 → C자 완료(손바닥면 ≤ 2 cm · 컵 축 R−0.5~R+2 cm)보다 3 cm · 2.5 cm 앞.
    #:   컵 스폰 ±2 cm 에도 손바닥면 ≥ 1 cm · 컵 축 ≥ R+0.5 cm 라 손이 컵과 겹치지 않는다. 띠 중심 높이(0.5 H 이하)는 작은 컵에서
    #:   팔꿈치 한계로 IK 미수렴(최대 29 mm)이라 0.8 H. `tests/test_grasp_gates.py` 가 FK 로 대조한다.
    near_start_frac: float = 0.5
    near_start_after_common_steps: int = 4
    near_start_species: tuple = ("cup_big_s085", "cup_big_s100", "cup_big_s115", "cup_big_s130", "shaker_closed",
                                 "cup_big_s090", "cup_big_s105", "cup_big_s120")
    near_start_arm_q: tuple = (
        (0.2721, 0.3501, -0.5729, 0.2741, 0.6373, 0.1909, 1.0545),      # cup_big_s085
        (0.0933, 0.4461, -0.4102, 0.6141, 0.5813, 0.1431, 0.9039),      # cup_big_s100
        (0.0043, 0.5752, -0.5059, 0.8202, 0.7443, 0.0710, 0.8600),      # cup_big_s115
        (-0.0815, 0.6752, -0.5054, 0.9801, 0.8190, 0.0314, 0.8282),     # cup_big_s130
        (0.1214, 0.5081, -0.6055, 0.7659, 0.7699, 0.0031, 0.8329),      # shaker_closed
        (0.1950, 0.3850, -0.4638, 0.4207, 0.5715, 0.1717, 0.9857),      # cup_big_s090
        (0.0684, 0.5050, -0.4985, 0.6907, 0.6879, 0.1038, 0.8946),      # cup_big_s105
        (-0.0453, 0.5760, -0.4159, 0.8776, 0.6923, 0.0847, 0.8217),     # cup_big_s120
    )
