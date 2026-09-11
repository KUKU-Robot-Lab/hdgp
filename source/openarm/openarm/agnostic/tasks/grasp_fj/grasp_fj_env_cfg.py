"""grasp_fj cfg — `FJKeypointEnvCfg` 상속, Track B(팔 7D 관절 증분 + EMA, Fabrics 없음).

DESIGN.md §1 B 열. 목표열·보상·관측·종료·DR 의 **정의는 A 에서 유래**하지만,
★09.10 부터 코드를 공유하지 않는다(사용자 확정 "s2r 하고 fj 는 공유 금지").
부모는 이 디렉터리 안의 포크본(`fj_kp_cfg` → `fj_core_cfg`)이며, A 트랙을 고쳐도
여기는 따라오지 않는다. A 대비 바뀌는 것은 팔 액션 어댑터 세 값이다:
- `arm_cmd_dim = n_arm`(obs 의 cmd_state = 직전 팔 목표 q*_{t-1}),
- `k_arm`(rad/step per unit action), `arm_ema`(α).
차원은 A 의 `_derive_spaces` 공식을 그대로 쓰고 `_arm_action_dim` 훅만 n_arm 으로 바꾼다
(tesollo_right: action 22 · actor 131 · critic 155).
"""

from __future__ import annotations

from dataclasses import fields

from isaaclab.utils import configclass

from .fj_kp_cfg import FJKeypointEnvCfg
from .fj_reward import FJRewardCfg
from .robot_profiles import PROFILES


@configclass
class GraspFJEnvCfg(FJKeypointEnvCfg):
    """SimToolReal 식 트랙 B: 팔 관절 7D 증분+EMA(위치 목표만) + 시너지 15D, 접촉 항 0개.

    왜 B 인가: 실기 배포에서 fabric 노드를 빼고(4노드→3노드) 정책 출력을 pd 노드에 바로
    먹이기 위해서다. 팔 속도 목표는 주지 않는다 — 실기 JTC 가 velocity 를 쓰지 않는다.
    """

    # obs cmd_state 폭 = 직전 팔 목표 q*_{t-1}(n_arm). finalize 가 프로필과 대조한다.
    arm_cmd_dim: int = 7
    # ★09.08 k_arm 은 SimToolReal 의 `dof_speed_scale · dt` 를 **상수 하나로 접은 값**이다:
    #     그들: arm_raw = prev + dof_speed_scale · dt · a   (action_utils.py:52, dt=1/60,
    #           controlFrequencyInv 1 → 정책 스텝 1/60 s) ⇒ 계수 = 1.5 × 1/60 = 0.025
    #     우리: arm_raw = prev + k_arm · a,  정책 스텝 = sim.dt(1/120) × decimation 2 = 1/60 s
    #   ⇒ k_arm ≡ dof_speed_scale × 정책_dt. 구 0.167 은 dof_speed_scale 10.0 = 그들의 6.7배였다.
    #   ★이 등식은 정책 dt 가 양쪽 다 1/60 이라서 성립한다 — decimation·sim.dt 를 건드리면 다시
    #     환산해야 하므로 `_validate_fj_fields` 가 k_arm/dt ≈ 1.5 를 대조한다.
    #   "1.0 rad/s 브리지 상한"이라는 구 근거는 **현재 설정에 없다**(09.08 확인): 실기 팔 캡은
    #   reduced 0.25 / full 2.0(pd 노드) · 브리지 기본 0.5 이고, 1.0 은 **손** 캡이었다.
    #   0.15 rad/s 는 가장 보수적인 단계 캡 안에 들어가지만 1.0 은 그것을 4배 넘는다.
    k_arm: float = 0.025
    # ★09.08 위 환산의 **선언값**. `arm_slew_rad_s` 와 같은 역할이다 — 상수를 코드에 박으면
    #   의도적 이탈(팔 속도 대조군)까지 막히고, 안 박으면 dt 변경 시 조용히 어긋난다.
    #   검증기가 `k_arm ≈ arm_dof_speed_scale × 정책_dt` 를 대조하므로, 대조군은 이 값을
    #   **같이** 선언해야 부팅한다(예: 10.0 + k_arm 0.167 + slew 1.0).
    arm_dof_speed_scale: float = 1.5      # SimToolReal.yaml:26 dofSpeedScale
    # 왜 0.1: 목표 EMA(SimToolReal α). q*_t = α·q_raw + (1−α)·q*_{t-1} — 실효 slew 는 α·k_arm/step.
    arm_ema: float = 0.1
    # 선언된 포화 slew(rad/s). finalize 가 α·k_arm/policy_dt 와 대조한다 — 문구와 실효값이 못 갈린다.
    # 0.1 × 0.025 / (1/60) = 0.150 = SimToolReal(armMovingAverage 0.1 · dofSpeedScale 1.5).
    arm_slew_rad_s: float = 0.15
    # ★09.07 홈 주변 **관절 목표 상자**(rad). 0 = 끔(구 거동 = 관절 한계까지 자유).
    #   왜 필요할 수 있나: 증분 매핑은 적분기라 복원력이 없다. a 의 부호가 잠깐 치우치면 목표가
    #   그 방향으로 계속 쌓여 관절 한계에 붙고, 반사 경계 때문에 정상분포가 **관절 범위 전체**가
    #   된다 — 손이 물체에서 멀어진 채 돌아오지 못한다(RH56F1 실측 `ctrl/arm_limit_sat` 0.49,
    #   `ft_dist` 0.13 → 0.69). 상자는 A 의 palm 박스와 같은 역할을 관절공간에서 한다.
    #   tesollo B(fj_b1)는 이것 없이 학습됐으므로 기본은 0 이다 — 켜는 트랙만 켠다.
    arm_target_box_rad: float = 0.0
    # ★09.07 B-v: 리프트 후 팔 액션 **반전** 벌점. 측도 = 1차 차분 RMS/2 ∈ [0,1](A 는 비유계·작동점 ≈10× 라 0.1).
    #   최악(매 스텝 ±1 반전) −1.0/step, γ 0.99 할인 합 −100 < 리프트 보너스 300 → 리프트가 여전히 이득(Check 1).
    #   전속 이송(a 일정)은 차분 0 이라 세금이 없다 — 수준 |a| 를 벌하면 이송까지 벌한다.
    rw_cmd_rate_scale: float = 1.0
    # ★★09.09 감쌈 보상 계수. **기본 0.0 = 끔** — 켜지 않은 런은 현행과 비트 동일하다.
    #   이게 곧 형제 트랙(`grasp_fj_rh`) 방화벽이다(leaf 가 아니라 base 에 두되 기본값이 off).
    #   근거·설계는 `fj_reward.FJRewardCfg.wrap_scale` 주석. reward-audit 09.09: REVISE→조건부 ACCEPT
    #   (근접 게이트 + 실측 기준 + 상한 2.0 + A(kp 5.0) 선행이 조건).
    # ★★09.11 **인벨롭 그립을 성공의 전제조건으로** (사용자 확정 + reward-audit REVISE).
    #   왜: goal_bonus 가 총점의 93.2% 를 독점하는데 성공 술어가 `kp_dist ≤ tol 연속 10회`
    #   뿐이라 **손이 물체를 어떻게 잡았는지가 전혀 안 들어간다**. 그래서 정책이 손끝을
    #   물체 표면에서 ~50mm 띄운 채(ft_dist 90mm, 설계 파지 대비 굴곡 41%) 성공을 받는
    #   해에 수렴했고, 보상은 epoch 300 이후 평평하다. wrap 항은 지급 0.0002 로 사실상 0.
    #
    #   ★임계를 설계 파지(0.83)로 **바로 걸면 안 된다** — 현재 0.34 라 성공이 즉시 0 이 되고
    #     총점의 93% 가 사라진다(reward-audit Check 4 파괴). 그래서 커리큘럼으로 올린다:
    #     현재값 바로 위에서 시작해, 직전 에피소드 성공이 임계를 넘을 때만 한 칸 조인다.
    #     `tol` 커리큘럼(0.1125 → 0.0598)이 같은 규약으로 이미 작동하고 있다.
    #   ★0.0 이면 **끔**(전제조건 없음) = 09.11 이전과 비트 동일.
    # ★★09.11 — `hand_curl` 정의가 바뀌었다(고정 칸 thumb_2·pinky_2 제외, 10칸 → 8칸).
    #   옛 정의의 실측 0.348 중 두 고정 칸이 실어 나르던 몫이 빠지므로 자연값이 내려간다.
    #   0.35 로 두면 시작부터 성공을 막는다 — 커리큘럼의 자동정지 가드는 **상승**만 막고
    #   시작값은 못 막는다. 새 정의의 추정 자연값 근처에서 출발시킨다.
    # 오프라인 계산(R 36.5mm · H 50mm · τxy 20mm · τz 30mm, 마디 8개):
    #   리셋 자세(손이 16cm 밖) 0.01~0.03 · **평평한 손 캐리(영상 실패형) 0.230** ·
    #   인벨롭 0.886. 0.15 는 평평한 손 캐리를 **통과시키고** 리셋 자세는 막는다 —
    #   시작부터 성공을 0 으로 만들면 goal_bonus 가 사라져 부트스트랩이 죽는다.
    grasp_wrap_start: float = 0.15
    # 프로필 `hand_grip_pose` 를 새 정의(8칸)로 계산한 값 = 0.9854.
    #   옛 0.83 은 10칸 정의의 값이었다(thumb_2 0.420 · pinky_2 0.000 이 평균을 끌어내렸다).
    # 인벨롭 실측 추정 0.886 바로 아래. 1.0 은 모든 마디가 표면 안에 있어야 해서
    #   손가락 두께·관절 한계상 도달 불가다 — 닿을 수 없는 천장은 커리큘럼을 멈춘 채 둔다.
    grasp_wrap_max: float = 0.85
    # ×1.10 — fj_g2 가 같은 배속을 따라왔다(hand_curl +0.048/100ep, successes 0.96 유지).
    #   0.15 → 0.85 은 18.2 승급 × 46.9 epoch = **853 epoch**.
    grasp_wrap_factor: float = 1.10
    # ★★09.11 — 3000 → 750. 3000 프레임 ÷ horizon_length 16 = **187.5 epoch/승급**이라
    #   fj_g1/g2 가 300 epoch 동안 각각 **1회**만 승급했다(tol 커리큘럼도 같이 1회).
    #   0.35 → 0.83 은 ×1.05 로 17.7 승급 = 3,320 epoch, ×1.10 로 9.1 승급 = 1,700 epoch.
    #   판단 주기보다 한 자릿수 느려서 `curl_tol` 0.3675 가 정책의 자연값(0.335~0.348)과
    #   같아 **구속력이 없었다** — 전과 같이 goal_bonus 만 먹고 수렴했다(보상 증가분의 91%).
    #   750 → 46.9 epoch/승급. 안전장치는 이미 있다: prev_ep_successes < threshold 면
    #   `RisingCurriculum` 이 승급을 멈춘다(현재 4.66 vs 2.0).
    grasp_wrap_interval: int = 750       # tol 커리큘럼과 같은 프레임 간격(tol 은 그대로)
    grasp_wrap_threshold: float = 2.0    # tol 과 같은 입력(prev_ep_successes_mean)
    # ★09.11 — 기본값을 0.0 → 2.0 으로. fj_g1/g2 는 이 값을 CLI 로만 넘겨 돌았고
    #   (`env.rw_wrap_scale=2.0`), 런처가 빠지면 조용히 감쌈 항이 사라진다. git 에 고정한다.
    #   ★★09.11 진행형으로 바뀌어 척도가 달라졌다 — 스텝당이 아니라 **에피소드 총량**이
    #     scale × (최고 포위도 − 리셋 포위도) ≈ scale × 0.64 다. ft_scale(50) 과 같은 진행형
    #     척도로 두면 총량 ≈ 32 — kp_progress(~40) · lift_bonus(300) 보다 작다(reward-audit Check 1).
    rw_wrap_scale: float = 50.0
    rw_wrap_tau_xy: float = 0.02
    rw_wrap_tau_z: float = 0.03

    # ★09.08 손 20관절 **독립** 지령 스위치. 기본 False.
    #   왜 기본이 False 인가: (1) 기존 계약 22/131/155 와 fj_b9 체크포인트를 그대로 둔다.
    #   (2) `grasp_fj_rh`(RH56F1)가 이 cfg 를 상속한다 — 기본이 True 면 남의 진행 중 트랙 차원이
    #       조용히 바뀐다. 켜는 것은 런처의 `env.hand_direct=true` 뿐이다.
    #   근거(09.08 영상+로그): 시너지의 실효 지령은 15 가 아니라 **4개**다 — `_1` 관절 5개는
    #   open==grip 이라 무효이고, `couple_four_fingers=True`·`finger_residual_scale=0.0` 이라
    #   엄지 외 네 손가락이 채널 평균 하나로 묶인다. 14개 가동관절을 4개가 움직인다.
    #   실측 fj_b9: syn_close 0.43(절반도 안 닫힘)인데 act_sat_hand 0.71 — 정책은 최대로 지르는데
    #   시너지 램프가 그 이상을 못 만든다. 옆에서 접근하면 손 모양을 못 바꿔 파지가 실패한다.
    hand_direct: bool = False

    # ★09.08 **full-joint 손의 관절 목표 EMA**(SimToolReal `handMovingAverage` 0.1). `hand_direct`
    #   전용이다 — 시너지 경로(A·`grasp_fj_rh`)는 안 읽는다. 법칙은 팔과 같은 꼴이다:
    #     raw = lo + ½(a+1)(hi−lo)  →  q*_t = α·raw + (1−α)·q*_{t-1}  →  clamp(lo, hi)
    #   여기서 [lo, hi] 는 articulation soft limit ∩ 프로필 `hand_action_limit_override`
    #   (테솔로: `_3/_4` 하한 0 — 손등 과신전 차단). 폐쇄도·램프·close_gate·blocked 는 **없다**
    #   (09.08 사용자 확정: "SimToolReal 처럼 풀 조인트"). 1.0 = 평활 없음, (0,1] 만 허용.
    hand_ema: float = 0.1
    # ★09.08 hand_direct 리셋: 프로필 리셋 자세(시너지 open, 엄지 `_3` −0.5 pre-curl)가 액션한계 밖이면 B 는 그
    #   관절을 **한계로 clamp 해 심는다**(관절 상태 + EMA 시드 둘 다). 이 값보다 크게 움직여야 하면 리셋 자세가
    #   범위와 아예 다른 자세라는 뜻이라 부팅을 죽인다. 0.5(엄지 `_3` −0.5 → 0) 가 현재 유일한 사례.
    hand_reset_clamp_max_rad: float = 0.6
    # ★09.08 성공마다 에피소드 시계를 이 값으로 되돌린다(SimToolReal env.py:2437-2439 의
    #   `progress_buf[is_success > 0] = 0`) = 스텝 예산이 **에피소드당**이 아니라 **목표당**이 된다.
    #   −1 = 끔 = 현행. 왜 0 이 아니라 2 를 권하나(leaf 참조): episode_length_buf 가 0/1 일 때
    #   반응하는 소비자가 넷 있다 — 액션·관측·**물체**(10스텝) 지연 flush 가 목표마다 터지고
    #   (물체 참값을 훔쳐보게 된다), `_fresh <= 1` 리셋 진단(grasp_s2r_env:1528)이 영구 오염되어
    #   `reset/arm_q_dev_max` 가 fail-loud 가드 기능을 잃는다. 2 면 넷 다 피하고 비용은 600→597.
    goal_clock_restart_step: int = -1

    #: ★09.10 시작 거리 가드의 대역 — **손바닥 중심 ↔ 물체 중심**(m). 손끝 평균이 아니다:
    #:   손끝은 손가락 굽힘에 따라 같은 팔 자세에서도 98.8→67.3→82.5 mm 로 요동친다.
    #:   손바닥은 손가락 관절의 상류라 자세와 무관하다(같은 조건 150.4 mm 불변, 09.10 FK).
    #:   대역은 자산·물체가 바뀌면 다시 재야 한다 — 물체 반경을 모르는 지표이므로
    #:   "이 값이면 감쌀 공간이 있다"를 보장하지 않는다. 그건 감쌈 여유 검사가 따로 한다.
    start_palm_dist_band_m: tuple = (0.10, 0.26)

    def _arm_action_dim(self, profile) -> int:
        """액션의 팔 구간 폭 = 관절 수(B). A 의 `_derive_spaces` 가 이 훅으로 22 를 만든다."""
        return int(profile.num_arm_joints)

    def _hand_action_dim(self, profile) -> int:
        """`hand_direct` 면 손 폭 = 관절 수(20). 아니면 A 공식 그대로."""
        if bool(self.hand_direct):
            return int(profile.num_hand_joints)
        return super()._hand_action_dim(profile)

    def finalize_after_overrides(self) -> None:
        super().finalize_after_overrides()          # A: 박스·kp 필드 검증·차원(이 클래스의 훅으로)
        self._validate_fj_fields(PROFILES[self.profile_name])

    def _supports_per_finger_hand(self) -> bool:
        """손가락별 액션 슬롯(per_finger)을 쓸 수 있는 트랙인가.

        기본 False — mixin 의 팔 폭 기본값은 palm 6D 라 B 의 팔 7 과 어긋난다.
        env 가 `_arm_slot_width()` 를 `_arm_action_dim` 으로 덮은 서브클래스만 True.
        """
        return False

    def _validate_fj_fields(self, profile) -> None:
        """B 신설 필드 범위 + A 와 갈릴 수 있는 조합을 cfg 단계에서 죽인다."""
        errs = []
        if int(self.arm_cmd_dim) != int(profile.num_arm_joints):
            errs.append(f"arm_cmd_dim {self.arm_cmd_dim} ≠ num_arm_joints {profile.num_arm_joints}")
        if float(self.k_arm) <= 0.0:
            errs.append(f"k_arm 은 > 0, got {self.k_arm}")
        if float(self.arm_target_box_rad) < 0.0:
            errs.append(f"arm_target_box_rad 는 ≥ 0 (0 = 끔), got {self.arm_target_box_rad}")
        if not (0.0 < float(self.arm_ema) <= 1.0):
            errs.append(f"arm_ema 는 (0, 1], got {self.arm_ema}")
        _dt = float(self.sim.dt) * int(self.decimation)                 # 정책 스텝
        _slew = float(self.arm_ema) * float(self.k_arm) / _dt          # 실효 포화 slew(rad/s)
        if abs(_slew - float(self.arm_slew_rad_s)) > 0.02 * float(self.arm_slew_rad_s):
            errs.append(f"실효 slew α·k_arm/dt = {_slew:.3f} rad/s ≠ arm_slew_rad_s {self.arm_slew_rad_s}")
        # per_finger 는 팔 폭을 mixin `_arm_slot_width()` 훅에서 읽는다(기본 6 = A 의 palm).
        # B 에서 쓰려면 env 가 그 훅을 `num_arm_joints` 로 덮어야 하므로, 덮은 서브클래스만
        # `_supports_per_finger_hand()` 로 선언한다(tesollo B 는 채널 레이아웃이라 False).
        if str(self.hand_layout) == "per_finger" and not self._supports_per_finger_hand():
            errs.append("hand_layout=per_finger 는 팔 폭 훅(_arm_slot_width)을 덮은 트랙만 쓴다")
        if bool(self.hand_direct):
            # full-joint 손: 관절 목표 EMA 하나가 법칙의 전부다. 범위 밖이면 평활이 발산하거나(>1)
            # 목표가 영원히 안 움직인다(≤0).
            if not (0.0 < float(self.hand_ema) <= 1.0):
                errs.append(f"hand_ema 는 (0, 1], got {self.hand_ema}")
            if float(self.hand_reset_clamp_max_rad) < 0.0:
                errs.append(f"hand_reset_clamp_max_rad 는 ≥ 0, got {self.hand_reset_clamp_max_rad}")
            # ★속도 피드포워드 금지: 램프가 없으므로 `_syn_vel` = 목표 차분/dt 가 최대
            #   (hi−lo)·α/dt ≈ 3.14×0.1×60 ≈ 19 rad/s 까지 뛴다. SimToolReal 은 위치 목표만 준다.
            if float(self.hand_velocity_ff_scale) != 0.0:
                errs.append(f"hand_direct 는 hand_velocity_ff_scale = 0 을 요구한다(위치 목표만): "
                            f"{self.hand_velocity_ff_scale}")
        # ★09.08 k_arm 은 dof_speed_scale·dt 를 접은 값이다(위 필드 주석). 정책 dt 가 바뀌면
        #   k_arm 을 다시 접어야 하는데 그걸 지켜주는 것이 없었다. 선언값과 대조한다 —
        #   의도적 이탈(대조군)은 선언을 같이 바꾸면 통과하고, dt 만 바뀐 실수는 죽는다.
        _implied_speed_scale = float(self.k_arm) / _dt
        _declared = float(self.arm_dof_speed_scale)
        if _declared <= 0.0:
            errs.append(f"arm_dof_speed_scale 은 > 0, got {_declared}")
        elif abs(_implied_speed_scale - _declared) > 0.02 * _declared:
            errs.append(f"k_arm/정책_dt = {_implied_speed_scale:.3f} ≠ 선언 dofSpeedScale {_declared} "
                        f"(k_arm {self.k_arm} · dt {_dt:.5f}) — dt 를 바꿨으면 k_arm 도 환산해야 한다")
        # ★09.10 시작 자세는 **프로필**이 소유한다(`arm_reset_joint_pos`, 절대 관절값).
        #   구 `arm_reset_offset_rad`(태스크 cfg 의 홈 기준 델타)는 자산 간 이식이 불가능해 폐기했다.
        _rq = tuple(profile.arm_reset_joint_pos)
        if _rq and len(_rq) != int(profile.num_arm_joints):
            errs.append(f"{profile.name}.arm_reset_joint_pos 길이 {len(_rq)} "
                        f"≠ num_arm_joints {profile.num_arm_joints}")
        if _rq:
            import re as _re
            _pat = _re.compile(str(profile.arm_joint_regex))
            _home = [v for k, v in profile.init_joint_pos.items() if _pat.fullmatch(k)]
            if len(_home) != len(_rq):
                errs.append(f"{profile.name}: arm_joint_regex 로 뽑은 홈 팔관절 {len(_home)}개 "
                            f"≠ arm_reset_joint_pos {len(_rq)}개")
                _home = list(_rq)
            _d = max(abs(a - float(b)) for a, b in zip(_rq, _home))
            if _d > 1.5:
                errs.append(f"{profile.name}.arm_reset_joint_pos 가 홈에서 {_d:.2f} rad 떨어져 있다 "
                            f"(|1.5| 초과) — 시작 자세가 아니라 다른 홈이다")
        _max_steps = int(round(float(self.episode_length_s) / _dt))
        _r = int(self.goal_clock_restart_step)
        # 0/1 은 지연 flush·`_fresh` 리셋 진단·`ctrl/start_ft_dist`(≤1 을 '리셋 직후' 로 읽는다)를 오염시킨다.
        if _r < -1 or _r in (0, 1) or _r >= _max_steps - 1:
            errs.append(f"goal_clock_restart_step 는 −1(끔) 또는 [2, {_max_steps - 2}], got {_r}")
        # ★kp_a8 잠금쌍: 연속 판정은 SimToolReal 공차(0.075 × keypointScale 1.5 = 0.1125)와
        #   **같이만** 켠다. 술어만 베끼고 공차를 안 맞춘 것이 리프트를 죽였다.
        if bool(self.goal_force_consecutive) and float(self.tol_start) < 0.10:
            errs.append(f"goal_force_consecutive=True 는 tol_start ≥ 0.10 을 요구한다"
                        f"(kp_a8): {self.tol_start}")
        # ★tol ↔ 첫 목표 z 짝. `goal_bonus` 는 lifted 게이트가 없다 — near_goal 이 리프트 래치보다
        #   낮은 곳에서 켜지면 물체를 안 들고도 목표 보너스를 받는다(REWARD_AUDIT Check 2).
        #   현행 A: 0.16 − 0.06 = 0.10 = 래치. 공차를 올리면 이 짝을 같이 올려야 한다.
        _zfloor = float(self.goal_first_z_range[0]) - float(self.tol_start)
        if _zfloor < float(self.rw_lift_latch_height) - 1e-9:
            errs.append(f"goal_first_z_range[0] − tol_start = {_zfloor:.4f} < 리프트 래치 "
                        f"{self.rw_lift_latch_height} — 물체를 안 들고도 goal_bonus 가 나간다")
        # ★D1-a 짝: 커리큘럼은 mean(prev_episode_successes) ≥ tol_success_threshold 에서만 전진하는데
        #   성공 수는 goal_max 에서 잘린다. goal_max ≤ 게이트면 tol 이 영원히 못 조여진다(조용한 교착).
        if int(self.goal_max) <= float(self.tol_success_threshold):
            errs.append(f"goal_max {self.goal_max} ≤ tol_success_threshold {self.tol_success_threshold} — "
                        f"커리큘럼이 영원히 안 전진한다")
        if float(self.goal_delta_distance) < 0.0:
            errs.append(f"goal_delta_distance 는 ≥ 0 (0 = 제자리 유지), got {self.goal_delta_distance}")
        if errs:
            raise RuntimeError("[grasp_fj cfg] " + " · ".join(errs))

    def progress_reward_cfg(self) -> FJRewardCfg:
        """★B 전용 보상 cfg. A 와 제어 방식이 달라 모듈을 공유하지 않는다(09.08 사용자 확정).

        필드는 A 와 1:1 이고, 갈리는 것은 `goal_bonus` 의 **지급 방식**뿐이다 —
        A 는 near_goal 스텝마다 `goal_bonus/success_steps`, B 는 성공 순간 1회 전액
        (SimToolReal env.py:2656-2659 의 forceConsecutive 분기). 총액은 같다.
        """
        a = super().progress_reward_cfg()
        # ★지급 방식은 성공 술어에서 **유도**한다 — 둘이 갈리면 형제 트랙(`grasp_fj_rh`)이
        #   "누적 판정 + 1회성 보너스" 를 조용히 받는다(09.08 감사에서 발견).
        # ★`a` 는 **부모** cfg 라 B 고유 필드(wrap_*)를 안 갖는다 — 명시적으로 넘긴다.
        #   여기 빠뜨리면 계수를 hydra 로 올려도 조용히 0 이 되어 실험이 no-op 이 된다.
        return FJRewardCfg(goal_one_shot=bool(self.goal_force_consecutive),
                           wrap_scale=float(self.rw_wrap_scale),
                           wrap_tau_xy=float(self.rw_wrap_tau_xy),
                           wrap_tau_z=float(self.rw_wrap_tau_z),
                           **{f.name: getattr(a, f.name) for f in fields(a)})


@configclass
class GraspFJTesolloRightEnvCfg(GraspFJEnvCfg):
    """SimToolReal **과제 의미** 정합(09.08 사용자 확정).

    ★왜 이 값들이 base 가 아니라 leaf 에 있나 — **포획 방화벽**이다.
    형제 트랙 `grasp_fj_rh`(RH56F1, 진행 중·미추적)가 `GraspFJEnvCfg` 를 상속한다.
    base 에 두면 남의 트랙의 성공 술어·공차·예산·외란이 조용히 바뀐다. 등록부가
    인스턴스화하는 것은 이 leaf 하나뿐이므로(`config/__init__.py`), 여기 두면 안 잡힌다.
    `k_arm`/`arm_slew_rad_s` 만 base 에 있는데, fj_rh 가 이미 동일 값으로 고정·assert
    하고 있어(그쪽 `_validate`) base 변경이 no-op 이기 때문이다.

    사용자 지시(09.08): "팔매핑 simtooreal과 동일 / 팔실효 slew도 / 손속도 제한도 /
    성공목표에피소드 동일성유지 / **목표델타는 제외** / 외란은 줄일 필요가 있음".
    """

    profile_name: str = "tesollo_right"
    #: ★09.10 물체를 **신규 셰이커**로 통일한다(사용자 지시). `assets/simulation_setting/shaker`
    #:   의 usda 하나를 scale 0.80~1.20(0.05 단위 9종)로만 바꿔 소환한다.
    #:   구 기본값은 `grasp_s2r` 에서 상속한 `cup_family`(cup_big 7종 + 구 shaker 1종)였다 —
    #:   A 트랙(`grasp_kp`)은 이미 `shaker_sweep` 인데 B 만 컵을 보고 있어서 두 트랙의
    #:   비교가 성립하지 않았다.
    #:   ⚠물체가 바뀌면 원점 규약도 바뀐다: cup_big 은 바닥→원점 0.0773, 신규 셰이커는
    #:     **바운딩박스 중심** 0.0875. 시작 자세·여유는 반드시 다시 재야 한다.
    object_bank: str = "shaker_sweep"

    # ── 자기충돌 ON (09.09 사용자 확정) ──────────────────────────────────────
    # ★왜 이제 켤 수 있나. 09.01 에 끈 이유는 "손 hull 초기 겹침 × 자기충돌 = 폭주"였고,
    #   그 전제가 09.09 자산 정비로 사라졌다: 손 collider 를 벤더 convex_hull 로 굽고
    #   (shape 731→75), 관절한계를 자기관통이 없는 범위로 좁혔다(`_1` ±0.16 ·
    #   `thumb_3/4` ≤1.05 · `_3/4` 하한 0). **리셋(영) 자세 감사 PASS, FAIL 0** 이 조건이다.
    # ★왜 켜야 하나. 꺼두면 손가락이 서로와 손바닥을 **통과**한다 — 실기에 없는 자세를
    #   정책이 학습한다. c 시리즈 영상의 "손가락이 엉킨다"가 그것이고, 엄지가 손바닥을
    #   뚫고 −4.07 rad 까지 돌아간 것도 기하가 안 막은 것이 한 축이었다.
    # ⚠완전 폐쇄에서는 손가락이 손바닥·서로에 **닿는다**(실제 손도 그렇다). 그건 결함이
    #   아니라 자기충돌이 막아야 할 바로 그 접촉이다 — `audit_poses.yaml` 은 "task home"
    #   용이므로 폐쇄 자세를 거기 등록하면 정상 접촉을 결함으로 신고하게 된다(09.09 시행착오).
    # ⚠첫 스모크에서 `done/abnormal` 과 `ctrl/joint_err_max` 를 볼 것 — 08.29 식 폭주가
    #   재발하면 여기서 바로 드러난다.
    enable_self_collisions: bool = True

    # ── 손: 7+20 DOF 직접 제어 ────────────────────────────────────────────────
    # 사용자 지시: "simtooreal 하고 동일성을 유지한다면 시너지그립도 제거하고 7+20dof 제어".
    hand_direct: bool = True

    # ★09.08 손 법칙은 SimToolReal 과 **같은 순수 full-joint** 다(사용자 확정: "보정 open→grip 매핑,
    #   close_gate+blocked 홀드 — 이걸 안 하려고 했음"). 액션 20칸이 각 관절의 액션한계
    #   [lo, hi] 에 선형 매핑되고 관절 목표 EMA(`hand_ema` 0.1, base)만 걸린다. 폐쇄도·램프·
    #   게이트·blocked·외전 반폭 같은 것은 없다. [lo, hi] = soft limit ∩ 프로필
    #   `hand_action_limit_override`(테솔로: `_3/_4` 하한 0 = 손등 과신전 차단, 나머지 URDF 전폭).
    #   08.25 폐쇄속도 스윕(빠를수록 감쌈 악화)은 결합 시너지에서 잰 값이라 적용 범위 밖이고,
    #   위험은 런을 태우는 대신 계기로 관찰한다: ctrl/hand_joint_err_max · ctrl/hand_blocked_frac
    #   (진단 전용) · diag/obj_speed_lifted · 영상(손등 접근 여부 — 하한 0 이 유일한 방어선).
    # ★위치 목표만 준다(SimToolReal 원본). 램프가 없어 `_syn_vel` 이 최대 ≈19 rad/s 까지 뛴다 —
    #   cfg 검증기가 hand_direct 에서 0 을 강제한다. `grasp_fj_rh` 도 같은 이유로 0 이다.
    hand_velocity_ff_scale: float = 0.0

    # ── 성공·목표·에피소드 의미 ───────────────────────────────────────────────
    # successTolerance 0.075 × keypointScale 1.5 = 0.1125 (SimToolReal.yaml:84,88)
    tol_start: float = 0.1125
    # ★tol 과 **짝**이다 — 0.2125 − 0.1125 = 0.10 = 리프트 래치. 안 올리면 물체가 4.75cm 만
    #   떠도 goal_bonus 가 나간다(REWARD_AUDIT Check 2). goal_box_z_range (0.10, 0.30) 안.
    goal_first_z_range: tuple[float, float] = (0.2125, 0.28)
    # ★09.08 리뷰 정정: 커리큘럼 게이트 = mean(prev_episode_successes) ≥ threshold 인데 successes 는 goal_max 에서
    #   잘린다. SimToolReal 3/50 = 6%, A 2/50 = 4% 였고, goal_max 5 에 3.0 을 그대로 쓰면 **60%** — 10배 엄격.
    #   D1-a 는 결과가 0/5 ↔ 5/5 로 이분되므로 "env 60% 가 완주해야 tol 이 처음 조여진다" 가 된다. 2.0(40%)으로
    #   낮춘다 — 6%(0.3)는 완주 env 6% 에 조여져 kp_a8 형 붕괴(연속 판정 + 좁은 공차) 위험. 게이트 입력은
    #   `ctrl/prev_ep_successes_mean` 으로 본다.
    tol_success_threshold: float = 2.0
    goal_force_consecutive: bool = True         # 연속 10회 — cfg 가 tol_start 와 짝을 대조한다
    goal_clock_restart_step: int = 2            # 목표당 스텝 예산(왜 2 인지는 base 필드 주석)

    # ── 외란: **끔**(사용자 확정 09.10) ─────────────────────────────────────
    # ★base 가 0.0 으로 껐는데 이 leaf 가 같은 이름을 재선언해 2.7 로 되살리고 있었다.
    #   fj_sh1/fj_sh2 가 extF 켜진 채로 돌아간 원인이다. leaf 에서도 명시적으로 끈다.
    #   되살릴 때의 근거값은 지운 게 아니라 여기 남긴다:
    #   Kuka 어깨 300 N·m vs OpenArm 어깨 40 N·m = 7.5배 → 20.0/7.5 = 2.67 ≈ 2.7 · 2.0/7.5 = 0.27
    wrench_force_scale: float = 0.0
    wrench_torque_scale: float = 0.0

    # ★09.08 D1-a(사용자 확정): 과제 목적이 **grasp-lift 만**이라 목표열을 "제자리 유지(dwell)" 로 바꾼다.
    #   첫 목표는 그대로 리프트 높이(dz ∈ goal_first_z_range), 그 다음 목표는 이전 목표와 **같은 자리**
    #   (Δ 0) — 연속 10스텝 판정과 합쳐지면 "들어올린 채 정지" 가 곧 성공이다. 목표 5개 = 50스텝(0.83 s)
    #   외란 아래 유지 후 truncated(값 부트스트랩) → 재접근 연습. 회전 델타는 A 값(0°) 상속.
    #   ★goal_max 는 커리큘럼 게이트(tol_success_threshold 2.0)보다 **커야** 한다 — 같거나 작으면
    #     tol 이 영원히 안 조여진다(검증기가 죽인다). 게이트 2.0 = cap 의 40%(리뷰 정정, 아래 필드 주석).
    #   REWARD_AUDIT.md 09.08 D1-a 절: 목표당 keypoint_progress 잔여 ≤ 200×tol = 22.5 ≪ goal_bonus 1000.
    goal_delta_distance: float = 0.0
    goal_max: int = 5


@configclass
class GraspFJTesolloRightShortEnvCfg(GraspFJTesolloRightEnvCfg):
    """DG-5F short base 판 — 프로필만 다르고 과제 정의는 GraspFJTesolloRightEnvCfg 와 동일하다.

    ★손가락 체인·관절 이름·액션 공간이 dg5f-m 과 같고 홈 palm 포즈도 IK 로 맞췄으므로
      과제 상수는 전부 그대로 유효하다(손 프레임 일치 오차 0.023mm). 달라지는 것은
      자산·fabric variant·팔 홈 관절값이며 전부 프로필이 들고 있다.
    ⚠`palm_box` 는 미검증이다 — 부팅 시 경고가 뜬다. probe 후 승격할 것.
    """

    # ★★09.10 사용자 확정 "잠그고 재진행" — `thumb_1` 을 자산에서 용접한 변종으로 옮긴다.
    #   액션 한계(±0.01)·솔버(8/0↔32/1)·벤더 게인 33배·자기충돌 OFF 가 전부 실패했고,
    #   드라이브가 27배 포화라 어떤 게인으로도 못 잡는다는 것이 계산으로 확정됐다.
    #   트랙 A 는 `tesollo_right_short`(용접 없음)를 그대로 쓴다 — 자산이 갈린다.
    profile_name: str = "tesollo_right_short_tl"
