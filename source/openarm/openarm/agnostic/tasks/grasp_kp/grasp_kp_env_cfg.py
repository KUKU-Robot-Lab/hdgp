"""grasp_kp cfg — `GraspS2REnvCfg` 상속 + SimToolReal 식 목표열·progress 보상 필드 (Track A).

설계 원본은 `DESIGN.md` §2~§8. 이 파일이 하는 일 세 가지:
- 기존 필드 5개 덮어쓰기(respawn OFF · blocked 홀드 · 접촉동결 OFF · 코히런트 노이즈 · ADR OFF).
- 신설 필드 전부(단일 출처). env 는 `goal_seq_cfg()` / `progress_reward_cfg()` /
  `tolerance_curriculum_kwargs()` / `keypoint_half_height()` 로만 읽는다 — 모듈 dataclass 를
  만드는 자리는 이 한 곳이다.
- 목표 박스(env-local 절대)와 obs/state 차원을 `finalize_after_overrides` 에서 파생한다
  (hydra 오버라이드는 `__post_init__` 을 다시 부르지 않으므로 여기가 유일한 반영 지점).

접촉 센서 관련 부모 필드(`contact_force_*`, `envelope_*`, `latch_mode` …)는 남아 있지만
이 트랙의 env 는 **어느 것도 소비하지 않는다** — 계약 테스트가 grep 으로 잠근다.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ...modules.keypoint_goal import NUM_KEYPOINTS, GoalSeqCfg
from ...modules.progress_reward import ProgressRewardCfg
from ..grasp_s2r.grasp_s2r_env_cfg import GraspS2REnvCfg
from .robot_profiles import PROFILES

# 키포인트 4개 × xyz — actor obs 의 kp_rel_palm / kp_rel_goal 각각의 폭(=12).
_KP_DIM = 3 * NUM_KEYPOINTS


@configclass
class GraspKPEnvCfg(GraspS2REnvCfg):
    """SimToolReal 식 트랙 A: fabric palm 6D 델타 + 시너지 15D, 접촉 항 0개.

    actor 129 / critic 153 (tesollo_right). 물체 정체성·크기는 obs 에 없다 — 키포인트
    박스는 `keypoint_fixed_height` 로 **고정**이다(SimToolReal 은 물체별 크기인데,
    저장소 규칙 "obs 에 물체 정체성 없음"을 우선한다).
    """

    # ---- 기존 필드 덮어쓰기 (DESIGN §8) ----------------------------------------------
    # 왜 OFF: SimToolReal 처럼 낙하는 리셋이다. 재소환은 접촉 래치를 되감는 구판 규약이라
    #   높이 래치와 맞지 않는다(env 가 부팅에서 fail-loud 로 재확인한다).
    respawn_on_fail: bool = False
    # 왜 blocked: 지령↔실측 정체로 막힘을 잰다 — 접촉 센서 없이 성립하는 유일한 홀드.
    synergy_hold_mode: str = "blocked"
    synergy_contact_freeze: bool = False
    # 왜 True: 물체 파생 obs 3종(quat·kp_rel_palm·kp_rel_goal)을 **한 자세**(지연+노이즈)에서
    #   뽑는다. 항마다 따로 뽑으면 평균으로 노이즈를 상쇄하는 우회가 생긴다.
    obs_object_noise_coherent: bool = True
    enable_adr: bool = False
    # 왜 0.1: SimToolReal 관절속도 노이즈 값(실측 운동 구간 0.045 의 2배로 상향).
    obs_noise_qvel: float = 0.1
    # 왜 같이 올리나: 부모 `_assert_adr_monotonic` 은 ADR OFF 여도 base ≤ max 를 요구한다 —
    #   부모 max(0.05) < base(0.1) 이면 부팅에서 죽는다. = base → 폭 0(ADR OFF 라 무의미).
    adr_obs_noise_qvel_max: float = 0.1
    # 왜 z 0.35: 목표 박스 z 상한(정착고+0.30)+뱅크 정착고 편차(0.035)를 앵커(정착고+0.085)에서
    #   덮어야 한다. 부모 ±0.10 이면 목표의 위쪽 2/3 가 지령 불가라 목표열이 조용히 멈춘다.
    #   아래쪽은 `palm_box_min_z_override` 가 잘라낸다(a_z<−0.5 사구간 — 앵커를 올리면 a=0 이
    #   "컵에서 도망"이 되는 Phase 0 함정이 재발하므로 델타 확장을 택했다). xy 는 부모 그대로.
    #   env `_assert_goal_box_in_arm_reach` 가 이 셋(델타·박스·목표)을 부팅에서 대조한다.
    palm_delta_xyz: tuple[float, float, float] = (0.10, 0.10, 0.35)

    # ---- 키포인트 (DESIGN §2) -----------------------------------------------------------
    keypoint_scale: float = 1.5
    keypoint_fixed_height: float = 0.12       # s = 0.5·1.5·0.12 = 0.09 m (고정 박스)

    # ---- 목표열 (DESIGN §2) -------------------------------------------------------------
    goal_first_xy_range: float = 0.05
    # 왜 (0.16, 0.24): REWARD_AUDIT Check 2 — near_goal(tol 0.06) ⇒ dz ≥ 0.10 = 래치 높이.
    #   더 낮으면 래치를 안 넘고도 goal_bonus 를 받는 구멍이 생긴다.
    goal_first_z_range: tuple[float, float] = (0.16, 0.24)
    goal_delta_distance: float = 0.08
    goal_delta_rotation_deg: float = 0.0      # 0 = 직립 유지. 붓기 확장 시 올린다.
    # 왜 0.08(SimToolReal 0.15 아님): 프로필 palm 박스 x 하한 0.20 + 앵커 오프셋 x −0.066 이면
    #   목표 x ≥ 스폰 −0.096 만 지령 가능하다 — 델타를 키워도 못 넘는 **물리 한계**. 여유 포함 0.08.
    goal_box_xy_halfwidth: float = 0.08       # 스폰 중심 ±
    goal_box_z_range: tuple[float, float] = (0.10, 0.30)   # 정착고 기준(하한 0.10 = 래치)
    goal_success_steps: int = 10
    goal_max: int = 50
    # ★09.07 True 로 올렸다가 kp_a8 실측으로 **되돌렸다**. a6 와의 유일한 활성 차이가 이 한 줄이었고,
    #   그것만으로 리프트가 사라졌다(e50 lifted 0.108 → e200 0.0000, succ 4.25 → 0.000, kp_dist_min 0.109 → 0.201).
    #   기전(REWARD_AUDIT Check 1 그 자체): 연속이면 성공이 거의 안 나 **목표가 전진하지 않는다** →
    #   closest_kp 가 에피소드 최소에 고정돼 keypoint_progress 가 마르고, goal_bonus 는 tol 안 스텝에만 나온다.
    #   그런데 리프트는 `lift`(1.0/step 상수)와 `fingertip_progress` 를 끈다 → 목표로 수렴 못 하는 리프트 env 의
    #   수입 ≈ 0/step < 안 든 env 의 1.0/step. **리프트가 순손실이 되어** 정책이 되돌렸다.
    #   SimToolReal 이 연속을 쓸 수 있는 건 시작 공차가 0.075×1.5 = 0.1125 m 로 우리(0.06)의 2배이기 때문이다 —
    #   술어만 베끼고 공차를 안 맞춘 것이 오류였다. 되살리려면 tol_start 를 같이 올려야 한다.
    #   ★안정 파지는 성공 술어가 아니라 `rw_cmd_rate_scale` 벌점이 담당한다(완료 판정과 매끄러움을 섞지 않는다).
    goal_force_consecutive: bool = False
    # finalize 파생(단일 소스) — env-local 절대 박스. 직접 쓰지 말 것.
    goal_box_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_box_max: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ---- 허용오차 커리큘럼 (DESIGN §2) --------------------------------------------------
    tol_start: float = 0.06
    tol_floor: float = 0.015
    tol_factor: float = 0.9
    tol_interval: int = 3000
    tol_success_threshold: float = 2.0
    # 왜: 커리큘럼 상태는 프로세스 로컬(체크포인트에 없다) — play 는 tol 0.06 에서 다시 시작해
    #   커리큘럼이 굴러가고 성공수가 런마다 비교 불가였다. > 0 이면 **고정 tol**(갱신 없음).
    #   0 = 커리큘럼(학습). `-play` id 는 등록부가 tol_floor 로 둔다(hydra `env.tol_eval=` 가능).
    tol_eval: float = 0.0

    # ---- 보상 (DESIGN §3 — ProgressRewardCfg 필드 그대로, 접두사 rw_) -----------------
    rw_ft_scale: float = 50.0
    rw_lift_scale: float = 20.0
    rw_lift_base: float = 0.05
    rw_lift_clip: float = 0.5
    rw_lift_bonus: float = 300.0
    rw_lift_latch_height: float = 0.10
    rw_kp_scale: float = 200.0
    rw_goal_bonus: float = 1000.0
    rw_arm_vel_scale: float = 0.03
    rw_hand_vel_scale: float = 0.003
    rw_hand_floor_penalty: float = 10.0
    # 왜 rw_ 접두사가 붙은 별도 필드인가: 부모 `hand_floor_z/penalty/_max` 는 부모 보상 전용이라
    #   이 트랙에서는 **미소비**다. 보상·진단(`task/hand_floor_depth_max`) 기준은 이 값이다.
    rw_hand_floor_z: float = 0.215
    rw_hand_floor_max: float = 5.0
    # ★09.07 A-v: 리프트 후 지령 변화 벌점. 측도(env `_cmd_rate`) = 리미터 전 원지령 변화 / 리미터 상한(위치·회전
    #   평균), 비유계. a2/a6 작동점 step_raw 0.20 m = 10×(회전은 미계측, 벽↔벽 80° 면 27×) → 평균 ≈ 10~15.
    #   Check 1(do-nothing): 리프트 전 do-nothing 가치 = 1.0/(1−0.99) = 100. 리프트 후 벌점 −0.1×12.5 = −1.25/step
    #   할인 합 −125 → 리프트 가치 300 − 125 = 175 > 100. 리프트는 여전히 이득(여유 1.75×). 0.2 면 50 < 100 로 뒤집힌다.
    #   이송 세금: 리미터에 딱 맞게 지령하며 전속 이송 −0.1/step vs keypoint_progress +4/step(2.5%).
    #   정지 상태에서는 goal_bonus 100 이 상수라 이 항이 **유일한 변동 신호** — 거기서 학습된다.
    rw_cmd_rate_scale: float = 0.1
    # ★09.07 A-vi(kp_a7 붕괴 후): 크기(0.1)는 맞았는데 **누구에게 붙었나**가 틀렸다. per-env lifted 만으로 게이트하니
    #   e25 의 "lifted"(우연히 튕겨 올라가 상판에 다시 놓인 컵, sticky 래치·done/fell 사망)에 −1.35/step 이 500 스텝
    #   붙어 "컵을 건드리지 말자"를 배웠다(e25 close 0.37 → e50 0.007, a2/a6 는 같은 시점 0.15/0.38).
    #   ① 전역 래치: 매 스텝 lifted_frac 의 EMA(α=arm_ema, 창 ≈500 스텝 ≈ 31 epoch)가 arm_lifted_frac 이상이면
    #      sticky 로 arm. 임계는 성공 런이 실제 도달한 값(a6 e130 ≈ 0.30, 최종 0.79; a2 0.66)에서 잡는다
    #      (suppression-terms-need-task-first: "게이트는 시점만 정한다 — 크기는 Check 1 로 따로").
    #   ② hold 게이트: lifted 이고 dz > hold_dz(= drop_frac 판정선 0.03)일 때만. 떨어뜨린 컵을 다시 쥐러 가는 이동은
    #      벌하지 않는다 — 벌하면 재파지 대신 회피를 배운다.
    #   체크포인트 재개 시 arm 상태는 없다(tol 커리큘럼과 같은 프로세스 로컬) — FRESH 전제.
    rw_cmd_rate_hold_dz: float = 0.03
    rw_cmd_rate_arm_lifted_frac: float = 0.30
    rw_cmd_rate_arm_ema: float = 0.002

    # ---- 지연·지각 노이즈 (DESIGN §4·§5) ------------------------------------------------
    obs_delay_steps: int = 3                  # 큐 길이 L(1 = 지연 없음), 매 스텝 인덱스 재추첨
    action_delay_steps: int = 3
    object_delay_steps: int = 10
    obs_object_xyz_std: float = 0.01
    obs_object_rot_deg: float = 5.0

    # ---- 외란 (DESIGN §5) — lifted 일 때만, 질량 정규화 ---------------------------------
    wrench_force_scale: float = 20.0          # N / kg
    wrench_torque_scale: float = 2.0          # N·m / kg
    wrench_prob_range: tuple[float, float] = (0.001, 0.1)   # env 별 발화확률 logU

    # ---- 종료·박스·액션 (DESIGN §5·§8) --------------------------------------------------
    hand_floor_terminate_depth: float = 0.03  # 손 링크가 상판보다 이만큼 아래 → 종료
    # 왜 0.27: 손 최하단이 palm 원점 −57 mm, 상판 0.205 → 관통 방지(09.06 "a=0 에서 49 mm 뚫림").
    #   0 이면 끔. env 가 `_palm_lo[2]`·`_box_lo[2]` 를 **올리기만** 한다(낮추지 않음).
    # ★09.07 A-ii: 증분 매핑의 **복원력**. `a=0` 이 "지금 지령 유지"가 되면 정책이 표류해도
    #   돌아올 힘이 없다 — kp_a3 실측 epoch 437 에 리프트 0.0, 보상 1.038(= 아무것도 안 할 때
    #   주는 상수), ft_dist 0.313(a2 는 같은 시점 0.101). 그래서 매 스텝 앵커 쪽으로 조금 끈다.
    #     pull = clamp(palm_cmd_anchor_pull · (앵커 − 이전지령), 노름 ≤ 리미터·leak_reserve)
    #   ★리미터 예산을 액션과 **나눠 쓴다**. 안 나누면 pull+step 이 리미터를 넘어 A-i 가
    #     되살리려던 포화가 되돌아온다. 액션 게인은 남은 예산으로 고정 계산해 정책이 보는
    #     기울기가 위치에 따라 변하지 않게 한다.
    # ★09.07 A-iv: 팔 매핑 스위치. 기본은 **절대**다 — 증분은 세 번 다 과제를 못 배웠다.
    #   kp_a3(복원 없음) 523 · kp_a4(복원 0.010) 2,161 · kp_a5(복원 0.004) 476 epoch,
    #   전부 lifted 0.0000 · reward ≈ 1.0(do-nothing 상수) · close_gate ≤ 0.03.
    #   같은 보상·같은 박스의 kp_a2(절대)는 e476 에 lifted 0.658 · close_gate 0.907 이다.
    #   A-iii 로 목줄 반경을 0.065 → 0.1225 m 로 넓히자 **포화는 예측대로 내려갔지만**
    #   (act_sat_arm0 0.752 → 0.429, box_sat_x 0.919 → 0.643) 리프트는 정확히 0 이었다.
    #   즉 목줄은 실재했으나 묶는 제약이 아니었다. 아직 설명되지 않은 것: 증분 쪽은 학습 이전인
    #   e100 에서 이미 ft_dist 0.31(a2 는 0.17)로 손이 2배 멀고, syn_close 가 0.02(a2 0.40)다.
    #   ★증분 경로와 그 테스트는 지운 게 아니라 이 스위치 뒤에 남겨 둔다 — 리미터에 기대지 않는
    #     지령이라는 목표 자체는 유효하고, 작동하는 정책 위에서 다시 풀 문제다.
    palm_cmd_incremental: bool = False

    # ↓ 아래 두 값은 palm_cmd_incremental=True 일 때만 쓰인다.
    # ★09.07 A-iii: 복원력 0.01 → 0.004, 복원예산 0.2 → 0.05. 0.01 은 **탐색을 죽였다**.
    #   증분+복원은 OU 과정이라 무작위 탐색이 도는 반경이 걸음/√(2·pull) 로 고정된다.
    #   손이 좁혀야 할 거리는 ft_dist 0.19 → 0.07 ≈ **0.12 m**(a2 수렴값)이므로 반경을 거기 맞춘다.
    #     이전 (0.2, 0.010): 걸음 0.009228 → σ 0.0653 m — 과제 거리의 **절반**. 도달확률 6.6%
    #     지금 (0.05, 0.004): 걸음 0.010959 → σ 0.1225 m — 과제 거리와 일치. 도달확률 **32.7%**
    #   ★두 값은 **같이** 정해야 한다. 복원이 약해지면 복원에 떼어둔 예산이 남아 걸음이 19% 커지고,
    #     그러면 σ 도 같이 커진다. reserve 만 0.05 로 내리고 pull 을 0.003 에 두면 σ 0.1415 로
    #     과제 거리를 16% 넘겨 xy 박스(±0.10)를 계속 때린다.
    #   상관시간 1/pull 은 100 → 250 스텝. 에피소드 600 보다 짧아 kp_a3(복원 0) 식 표류가 아니다.
    #   ★부수 효과: 변위 d 를 **유지**하는 평형 액션은 μ = pull·d/걸음 이라 0.12 m 유지에
    #     0.130 → 0.044 로 작아진다. kp_a4 실측 box_sat 0.97 · act_sat_arm0 0.92 —
    #     박스 벽에 붙어 있던 원인이 같이 내려간다.
    #   예산: 걸음 대각 0.010959·√3 + 복원상한 0.02·0.05 = 0.01998 ≤ 리미터 0.02 ✔
    palm_cmd_anchor_pull: float = 0.004     # 반감기 ln2/0.004 ≈ 173 스텝 ≈ 2.9 s
    palm_cmd_leak_reserve: float = 0.05     # 리미터의 5% 를 복원에 배정(실제 소요 0.00048 m 의 2.1배)
    palm_box_min_z_override: float = 0.27
    arm_cmd_dim: int = 6                      # obs cmd_state 폭: A = palm_targets−anchor(6)

    # ------------------------------------------------------------------
    # 모듈 dataclass 조립 — env 가 읽는 유일한 통로
    # ------------------------------------------------------------------
    def keypoint_half_height(self) -> float:
        """s = 0.5 · keypoint_scale · fixed_height (m)."""
        return 0.5 * float(self.keypoint_scale) * float(self.keypoint_fixed_height)

    def goal_seq_cfg(self) -> GoalSeqCfg:
        # first_tilt_deg 는 0 고정 — 첫 목표 자세 = 정착 자세(직립). 붓기 확장 시 delta 와 함께 연다.
        return GoalSeqCfg(
            first_xy_range=float(self.goal_first_xy_range),
            first_z_range=tuple(float(v) for v in self.goal_first_z_range),
            first_tilt_deg=0.0,
            delta_distance=float(self.goal_delta_distance),
            delta_rotation_deg=float(self.goal_delta_rotation_deg),
            box_min=tuple(float(v) for v in self.goal_box_min),
            box_max=tuple(float(v) for v in self.goal_box_max),
            success_steps=int(self.goal_success_steps),
            force_consecutive=bool(self.goal_force_consecutive),
            max_goals=int(self.goal_max),
        )

    def progress_reward_cfg(self) -> ProgressRewardCfg:
        # success_steps 는 `goal_success_steps` 하나에서 온다(두 필드로 갈리면 goal_bonus/step 이 어긋난다).
        return ProgressRewardCfg(
            ft_scale=float(self.rw_ft_scale),
            lift_scale=float(self.rw_lift_scale),
            lift_base=float(self.rw_lift_base),
            lift_clip=float(self.rw_lift_clip),
            lift_bonus=float(self.rw_lift_bonus),
            lift_latch_height=float(self.rw_lift_latch_height),
            kp_scale=float(self.rw_kp_scale),
            goal_bonus=float(self.rw_goal_bonus),
            success_steps=int(self.goal_success_steps),
            arm_vel_scale=float(self.rw_arm_vel_scale),
            hand_vel_scale=float(self.rw_hand_vel_scale),
            hand_floor_penalty=float(self.rw_hand_floor_penalty),
            hand_floor_z=float(self.rw_hand_floor_z),
            hand_floor_max=float(self.rw_hand_floor_max),
            cmd_rate_scale=float(self.rw_cmd_rate_scale),
            cmd_rate_hold_dz=float(self.rw_cmd_rate_hold_dz),
        )

    def tolerance_curriculum_kwargs(self) -> dict:
        return dict(
            start=float(self.tol_start), floor=float(self.tol_floor), factor=float(self.tol_factor),
            interval=int(self.tol_interval), success_threshold=float(self.tol_success_threshold),
        )

    # ------------------------------------------------------------------
    # 파생 (멱등 — env `__init__` 이 super() 전에 다시 부른다)
    # ------------------------------------------------------------------
    def finalize_after_overrides(self) -> None:
        super().finalize_after_overrides()          # robot_cfg · events · 뱅크 · 스폰고 · _derive_spaces(이 클래스 판)
        self._validate_kp_fields()
        self._derive_goal_box(PROFILES[self.profile_name])
        # 모듈 dataclass 를 지금 만들어 본다 — 범위 위반은 부팅이 아니라 cfg 단계에서 죽는다.
        self.goal_seq_cfg()
        self.progress_reward_cfg()

    def _validate_kp_fields(self) -> None:
        """신설 필드의 범위를 한 번에 검사한다(조용한 클램프·브로드캐스트 금지)."""
        errs = []
        for name in ("obs_delay_steps", "action_delay_steps", "object_delay_steps"):
            if int(getattr(self, name)) < 1:
                errs.append(f"{name} 은 ≥ 1 (1 = 지연 없음), got {getattr(self, name)}")
        if float(self.keypoint_scale) <= 0.0 or float(self.keypoint_fixed_height) <= 0.0:
            errs.append("keypoint_scale / keypoint_fixed_height 는 > 0")
        if not (0.0 < float(self.tol_floor) <= float(self.tol_start)):
            errs.append(f"0 < tol_floor ≤ tol_start 여야 한다: {self.tol_floor}/{self.tol_start}")
        if float(self.tol_eval) < 0.0:
            errs.append(f"tol_eval 은 ≥ 0 (0 = 커리큘럼), got {self.tol_eval}")
        _fz, _bz = self.goal_first_z_range, self.goal_box_z_range
        if _fz[0] > _fz[1] or _bz[0] > _bz[1]:
            errs.append(f"z 범위 lo ≤ hi 위반: first {_fz} box {_bz}")
        # 첫 목표가 박스에 잘리면 분포가 조용히 바뀐다 — 첫 목표 범위는 박스 안이어야 한다.
        if _fz[0] < _bz[0] or _fz[1] > _bz[1]:
            errs.append(f"goal_first_z_range {_fz} 가 goal_box_z_range {_bz} 밖이다")
        if float(self.spawn_range) + float(self.goal_first_xy_range) > float(self.goal_box_xy_halfwidth):
            errs.append("spawn_range + goal_first_xy_range 가 goal_box_xy_halfwidth 를 넘어 첫 목표가 잘린다")
        _lo, _hi = (float(v) for v in self.wrench_prob_range)
        if not (0.0 < _lo <= _hi <= 1.0):
            errs.append(f"wrench_prob_range 는 0 < lo ≤ hi ≤ 1: {self.wrench_prob_range}")
        if float(self.hand_floor_terminate_depth) < 0.0 or float(self.palm_box_min_z_override) < 0.0:
            errs.append("hand_floor_terminate_depth / palm_box_min_z_override 는 ≥ 0")
        if not (0.0 <= float(self.palm_cmd_anchor_pull) < 1.0):
            errs.append(f"palm_cmd_anchor_pull 은 [0,1): {self.palm_cmd_anchor_pull}")
        if not (0.0 <= float(self.palm_cmd_leak_reserve) < 1.0):
            errs.append(f"palm_cmd_leak_reserve 는 [0,1): {self.palm_cmd_leak_reserve}")
        if float(self.rw_cmd_rate_scale) < 0.0:
            errs.append(f"rw_cmd_rate_scale 는 ≥ 0 (0 = 끔), got {self.rw_cmd_rate_scale}")
        # 정규화 분모가 두 리미터다 — 벌점을 켜 둔 채 리미터를 끄면 측도가 정의되지 않는다.
        if float(self.rw_cmd_rate_scale) > 0.0 and (
                float(self.palm_cmd_rate_limit_m) <= 0.0 or float(self.palm_cmd_rate_limit_rot_deg) <= 0.0):
            errs.append("rw_cmd_rate_scale > 0 이면 palm_cmd_rate_limit_m 와 palm_cmd_rate_limit_rot_deg 둘 다 > 0 이어야 한다")
        if not (0.0 <= float(self.rw_cmd_rate_arm_lifted_frac) <= 1.0):
            errs.append(f"rw_cmd_rate_arm_lifted_frac 는 [0,1] (0 = 즉시 arm): {self.rw_cmd_rate_arm_lifted_frac}")
        if not (0.0 < float(self.rw_cmd_rate_arm_ema) <= 1.0):
            errs.append(f"rw_cmd_rate_arm_ema 는 (0,1]: {self.rw_cmd_rate_arm_ema}")
        if float(self.rw_cmd_rate_hold_dz) < 0.0:
            errs.append(f"rw_cmd_rate_hold_dz 는 ≥ 0: {self.rw_cmd_rate_hold_dz}")
        if int(self.arm_cmd_dim) < 1:
            errs.append(f"arm_cmd_dim ≥ 1, got {self.arm_cmd_dim}")
        if errs:
            raise RuntimeError("[grasp_kp cfg] " + " · ".join(errs))

    def _derive_goal_box(self, profile) -> None:
        """목표 박스(env-local 절대) = 스폰 중심 ± xy 반폭, z = 정착고 + goal_box_z_range."""
        cx, cy = (float(v) for v in profile.object_spawn_center)
        h = float(self.goal_box_xy_halfwidth)
        # 왜 뱅크 최댓값인가: `object_origin_offset_z` 는 `_apply_object_bank` 가 뱅크 **최댓값**으로
        #   굳힌 값이고 env 별 정착고는 런타임 `_obj_origin_off` 가 준다. 박스 z 는 클램프일 뿐이라
        #   최댓값 기준이면 작은 물체에서 하한이 몇 mm 높아질 뿐(안전한 쪽)이다.
        z0 = float(self.table_surface_z) + float(self.object_origin_offset_z)
        self.goal_box_min = (cx - h, cy - h, z0 + float(self.goal_box_z_range[0]))
        self.goal_box_max = (cx + h, cy + h, z0 + float(self.goal_box_z_range[1]))

    def _arm_action_dim(self, profile) -> int:
        """액션의 팔 구간 폭. A = palm 6D. Track B 는 이 훅만 덮어써 `num_arm_joints` 를 준다."""
        return 6

    def _derive_spaces(self, profile) -> None:
        """액션/관측 차원 파생 — 부모와 같은 자리(finalize)에서 불린다."""
        n_arm = profile.num_arm_joints
        n_hand = profile.num_hand_joints
        num_tips = len(profile.fingertip_bodies)
        # `finger_sensor_bodies` 는 여기서 **손가락 목록**으로만 쓴다 — 센서는 만들지 않는다.
        num_fingers = len(profile.finger_sensor_bodies)
        if str(self.hand_layout) == "per_finger":
            _slots = [s for m in profile.hand_finger_channels.values() for s in m.values()]
            if not _slots:
                raise RuntimeError(f"[{profile.name}] hand_layout=per_finger 인데 hand_finger_channels 가 비어 있다")
            if sorted(set(_slots)) != list(range(max(_slots) + 1)):
                raise RuntimeError(f"[{profile.name}] 액션 슬롯이 연속이 아니다: {sorted(set(_slots))}")
            self.action_space = self._arm_action_dim(profile) + max(_slots) + 1
        else:
            n_ch = len(set(profile.hand_channel_of_joint.values()))
            self.action_space = self._arm_action_dim(profile) + n_ch * num_fingers   # 6 + 3·5 = 21

        # actor obs (DESIGN §4 순서 — env `_get_observations` 의 cat 순서와 **정확히** 같아야 한다):
        #   arm q/qd(2·n_arm) + hand q/qd(2·n_hand) + palm_pos(3) + palm_ax(6) + tips_rel_palm(3·nt)
        #   + cmd_state(arm_cmd_dim) + kp_rel_palm(12) + kp_rel_goal(12) + last_action
        # ★물체 onehot·치수·질량·클래스는 넣지 않는다 — 배포 시 알 수 없는 정보다.
        # ★물체 쿼터니언도 넣지 않는다 — 키포인트가 위치·기울기를 이미 담고, 남는 정보는 yaw 와
        #   q≡−q 부호뿐인데 축대칭 물체의 실기 yaw 는 임의(FP++)라 배포 시 분포 밖 채널이 된다.
        self.observation_space = (
            2 * n_arm + 2 * n_hand + 3 + 6 + 3 * num_tips + int(self.arm_cmd_dim)
            + _KP_DIM + _KP_DIM + self.action_space
        )
        # critic = obs(clean) + 물체 선/각속도(6) + palm 선/각속도(6) + d*_kp(1) + d*_ft(nt)
        #          + lifted(1) + progress(1) + successes(1) + reward·0.01(1) + dz(1) + d_kp(1)
        self.state_space = (
            self.observation_space + 6 + 6 + 1 + num_tips + 1 + 1 + 1 + 1 + 1 + 1)


@configclass
class GraspKPTesolloRightEnvCfg(GraspKPEnvCfg):
    profile_name: str = "tesollo_right"
