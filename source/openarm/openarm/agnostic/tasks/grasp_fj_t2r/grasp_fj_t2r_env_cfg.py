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
