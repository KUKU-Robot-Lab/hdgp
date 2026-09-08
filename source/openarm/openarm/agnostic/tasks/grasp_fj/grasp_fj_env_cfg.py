"""grasp_fj cfg — `GraspKPEnvCfg` 상속, Track B(팔 7D 관절 증분 + EMA, Fabrics 없음).

DESIGN.md §1 B 열. 목표열·보상·관측·종료·DR 은 전부 A(`grasp_kp`)와 **같은 cfg** 를
쓴다 — 여기서 바뀌는 것은 팔 액션 어댑터 세 값뿐이다:
- `arm_cmd_dim = n_arm`(obs 의 cmd_state = 직전 팔 목표 q*_{t-1}),
- `k_arm`(rad/step per unit action), `arm_ema`(α).
차원은 A 의 `_derive_spaces` 공식을 그대로 쓰고 `_arm_action_dim` 훅만 n_arm 으로 바꾼다
(tesollo_right: action 22 · actor 131 · critic 155).
"""

from __future__ import annotations

from dataclasses import fields

from isaaclab.utils import configclass

from ..grasp_kp.grasp_kp_env_cfg import GraspKPEnvCfg
from .fj_reward import FJRewardCfg
from .robot_profiles import PROFILES


@configclass
class GraspFJEnvCfg(GraspKPEnvCfg):
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

    # ★09.08 손 폐쇄 EMA(SimToolReal `handMovingAverage` 0.1). 0 = 끔 = 현행 램프.
    #   > 0 이면 폐쇄도 증분을 α 배로 쓴다 — tgt = lerp(open, grip, c) 가 c 에 **아핀**이라
    #   c ← α·cmd + (1−α)·c 는 관절공간 EMA 와 항등이다. 범위만 원시 관절한계가 아니라
    #   보정된 open→grip 인데 이건 **의도한 divergence** 다: `_3`/`_4` 가 ±90° 대칭이라
    #   원시 한계로 매핑하면 a=−1 이 손등 −90° 를 지령하고, 접촉 항이 **0개**인 이 보상에서
    #   손등 갈고리 파지가 정상 파지와 같은 점수를 받는다(tesollo-distal-hyperextension-exploit).
    #   대응책이던 require_palmar_contact 는 이 트랙에서 계약 금지다(ContactSensor 없음).
    #   ★기본 0 인 이유: `grasp_fj_rh` 가 이 cfg 를 상속한다(그쪽은 `_hand_command` 를 통째로
    #   덮어 이 필드를 안 읽지만, 기본값으로 남의 트랙 의미를 바꾸지 않는다는 규칙은 지킨다).
    synergy_close_ema: float = 0.0
    # ★09.08 폐쇄도 [0,1] 이 매핑되는 **관절 끝점**을 고르는 스위치.
    #   "synergy"  = 현행 open→grip.
    #   "per_role" = 시너지가 **폭 0 으로 묶어둔 관절만** 관절 한계로 풀어준다.
    #   왜 그것만 푸나(실측 09.08): 굴곡 관절(_2/_3/_4)의 open→grip 은 이미 관절 한계와
    #   실질적으로 같다 — index_3 은 [0, 1.8] 을 지령하고 soft limit 1.571 이 흡수하니 도달집합이
    #   [max(0,lo), hi] 와 같다. 반면 **폭 0 인 6개**(다섯 손가락 _1 외전 + pinky_2)는 하드웨어가
    #   ±0.4~0.6 rad 를 낼 수 있는데 액션 슬롯이 죽어 있다 — 20칸 중 실제 가동은 14칸뿐이었다.
    #   손이 좌우로 못 벌어지고, 이것이 09.08 영상의 "옆에서 파지 실패 · 손가락이 묶여 보임" 과
    #   직접 대응하는 자유도다.
    #   ★굴곡을 raw 한계로 풀지 **않는** 이유: _3/_4 가 ±1.571 대칭이라 a=−1 이 손등 −90° 를
    #     지령하고, 접촉 항이 0개인 이 보상에서 손등 갈고리 파지가 정상 파지와 같은 점수를 받는다.
    #     대책이던 require_palmar_contact 는 이 트랙에서 계약 금지다(ContactSensor 없음).
    #   ★판별은 **이름이 아니라 폭**으로 한다 — 로봇 관절명을 env 에 박지 않는다(계약).
    hand_range_mode: str = "synergy"
    # ★09.08 per_role 에서 **풀린 외전의 홈 기준 반폭**(rad). 왜 한계까지 안 푸나:
    #   우리 sim 은 `enable_self_collisions=False` 다(08.29 다물체 무한 리셋 대책) — 손가락이
    #   서로 **관통한다**. 접촉 항도 0개라 벌점도 없다. 즉 외전을 크게 열면 정책이 실기에서
    #   재현 불가한 "손가락 교차" 자세로 물체를 가둘 수 있다(원위 과신전 익스플로잇과 같은 형태).
    #   실측 기하(09.08): 인접 손끝 간격 24.8mm · 손가락 길이 130.8mm ⇒ 서로 마주 돌 때
    #   끝단이 닿기 시작하는 각 ≈ 0.084 rad(4.8°). 손가락 두께를 무시한 값이라 **상한**이다.
    #   0.035 는 SimToolReal 이 URDF 에서 AA 를 자른 값과 같고 그 상한의 2.4배 안쪽이다.
    hand_shape_span_rad: float = 0.035
    # 프로필 `hand_wide_shape_joint_regex` 에 걸린 외전의 반폭 — 손가락열에서 떨어진 것(엄지).
    # 0.35 ≈ SimToolReal 이 엄지 CMC_AA 에 남긴 폭(0.48 rad)의 절반 반폭.
    hand_wide_span_rad: float = 0.35
    # ★09.08 성공마다 에피소드 시계를 이 값으로 되돌린다(SimToolReal env.py:2437-2439 의
    #   `progress_buf[is_success > 0] = 0`) = 스텝 예산이 **에피소드당**이 아니라 **목표당**이 된다.
    #   −1 = 끔 = 현행. 왜 0 이 아니라 2 를 권하나(leaf 참조): episode_length_buf 가 0/1 일 때
    #   반응하는 소비자가 넷 있다 — 액션·관측·**물체**(10스텝) 지연 flush 가 목표마다 터지고
    #   (물체 참값을 훔쳐보게 된다), `_fresh <= 1` 리셋 진단(grasp_s2r_env:1528)이 영구 오염되어
    #   `reset/arm_q_dev_max` 가 fail-loud 가드 기능을 잃는다. 2 면 넷 다 피하고 비용은 600→597.
    goal_clock_restart_step: int = -1
    # ★09.08 리셋 시 팔 관절을 홈에서 이만큼 **고정** 오프셋한다(rad, 7개). 빈 튜플 = 끔 = 홈 그대로.
    #   왜: SimToolReal 의 팔 속도(0.15 rad/s)는 **손이 물체 바로 옆에서 시작**하는 것과 한 묶음이다.
    #   실측(09.08) — 그들 초기 손끝→물체 104mm(KUKA FK), 우리 250mm. 같은 속도로 우리 거리를
    #   가면 무작위 정책이 600스텝(1 에피소드)에 203mm 중 **14.4mm** 밖에 못 좁힌다
    #   (256환경 실측, 60mm 안에 든 env 0%). 속도만 베끼고 거리를 안 옮기면 아무도 근거를 못 댄
    #   조합이 된다. 그래서 시작 거리도 같이 맞춘다.
    #   ★★**컵 상대가 아니라 고정값**이어야 한다: 08.18 에 컵 참값 pregrasp 텔레포트를 버렸고
    #   (실기에서 컵 위치는 지각 결과라 재현 불가) 고정 홈 리셋으로 전환했다. 여기서도 스폰
    #   분포에 대한 **평균**이 100mm 가 되는 고정 자세 하나를 쓴다 — 그들 방식과 같은 구조다.
    #   값은 damped least squares IK 로 구했다(손바닥 회전 드리프트를 최소노름으로 억제).
    arm_reset_offset_rad: tuple = ()

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
            # 직접 지령은 시너지의 **폐쇄도 상태**(_syn_close·open/grip 자세)를 그대로 재사용한다.
            # 채널 전개만 건너뛰므로 레이아웃과 무관하지만, 아래 둘은 전제라 꺼져 있으면 죽인다.
            if float(self.synergy_close_speed) <= 0.0:
                errs.append(f"hand_direct 는 synergy_close_speed > 0 이 필요하다: {self.synergy_close_speed}")
            if str(self.synergy_hold_mode) != "blocked":
                errs.append(f"hand_direct 는 synergy_hold_mode='blocked' 를 전제한다(접촉 항 0개인 이 보상에서 "
                            f"감쌈을 만드는 유일한 장치): {self.synergy_hold_mode}")
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
        if str(self.hand_range_mode) not in ("synergy", "per_role"):
            errs.append(f"hand_range_mode 는 'synergy' | 'per_role', got {self.hand_range_mode}")
        _ro = tuple(self.arm_reset_offset_rad)
        if _ro and len(_ro) != int(profile.num_arm_joints):
            errs.append(f"arm_reset_offset_rad 길이 {len(_ro)} ≠ num_arm_joints {profile.num_arm_joints}")
        if _ro and max(abs(float(v)) for v in _ro) > 1.5:
            errs.append(f"arm_reset_offset_rad 이 |1.5| rad 를 넘는다 — 홈에서 그렇게 멀면 "
                        f"고정 자세가 아니라 다른 홈이다: {_ro}")
        for _f in ("hand_shape_span_rad", "hand_wide_span_rad"):
            if float(getattr(self, _f)) < 0.0:
                errs.append(f"{_f} 는 ≥ 0, got {getattr(self, _f)}")
        # ★자기충돌이 꺼진 채로 외전을 크게 열면 손가락이 관통한다(실측 침범각 0.084 rad).
        if str(self.hand_range_mode) == "per_role" and float(self.hand_shape_span_rad) > 0.08:
            errs.append(f"hand_shape_span_rad {self.hand_shape_span_rad} > 0.08 — 실측 침범각"
                        f"(0.084 rad, 손가락 두께 무시한 상한)을 넘는다. self-collision 이 꺼져 "
                        f"있어 물리도 보상도 이걸 못 막는다")
        if str(self.hand_range_mode) == "per_role" and not bool(self.hand_direct):
            errs.append("per_role 범위는 hand_direct 전용이다 — 결합 상태에서는 여러 관절이 채널 하나를 "
                        "공유해 외전만 따로 풀 수가 없다")
        if not (0.0 <= float(self.synergy_close_ema) <= 1.0):
            errs.append(f"synergy_close_ema 는 [0, 1] (0 = 끔), got {self.synergy_close_ema}")
        # EMA 를 켜면 램프는 **증명 가능하게** 무력이어야 한다 — 폐쇄도가 [0,1] 이라 rate ≥ 1 이면
        # clamp 가 항등이다. 둘이 동시에 걸리면 실효 속도가 어느 쪽인지 어떤 지표로도 못 가른다.
        if float(self.synergy_close_ema) > 0.0 and float(self.synergy_close_speed) < 1.0:
            errs.append(f"synergy_close_ema > 0 은 synergy_close_speed ≥ 1.0 을 요구한다"
                        f"(램프 무력화 증명): {self.synergy_close_speed}")
        _max_steps = int(round(float(self.episode_length_s) / _dt))
        _r = int(self.goal_clock_restart_step)
        if _r < -1 or _r >= _max_steps - 1:
            errs.append(f"goal_clock_restart_step 는 −1(끔) 또는 [0, {_max_steps - 2}], got {_r}")
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
        return FJRewardCfg(goal_one_shot=bool(self.goal_force_consecutive),
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

    # ── 손: 7+20 DOF 직접 제어 ────────────────────────────────────────────────
    # 사용자 지시: "simtooreal 하고 동일성을 유지한다면 시너지그립도 제거하고 7+20dof 제어".
    hand_direct: bool = True
    # ★09.08 시작 거리를 SimToolReal 과 맞춘다 — 손끝→물체 평균 **104.7 mm**(그들 104 mm).
    #   홈은 248.9 mm 였고, 그 거리에서 그들 팔 속도(0.15 rad/s)로는 무작위 정책이 한 에피소드
    #   (600스텝)에 203 mm 중 **14.4 mm** 밖에 못 좁힌다(256환경 실측, 60 mm 안에 든 env 0%).
    #   속도만 베끼고 거리를 안 옮기면 아무도 근거를 못 댄 조합이 된다.
    #   ★자세는 안 건드렸다 — 홈의 손 자세가 이미 옆면(side) 접근에 맞다. 6D IK 로 회전을
    #   고정하고 TCP 위치만 옮겼다: 손바닥 이동 [−50.8, +150.7, −94.5] mm · **회전 드리프트 0.01°**.
    #   ★관절 포화도 확인했다 — 한계 여유 최소 11.6°(홈 4.5°)로 **홈보다 2.6배 낫다**.
    #   증분 매핑은 적분기라 한계에 붙으면 반사 경계가 되어 정책이 못 돌아온다(RH56F1 실측
    #   `arm_limit_sat` 0.49 · `ft_dist` 0.13→0.69). 손끝 z 최저 0.273 m > 테이블 0.215 m.
    arm_reset_offset_rad: tuple = (-0.1967, -0.3729, -0.2159, -0.0179, -0.2384, -0.3813, 0.3810)

    # ★09.08 폭 0 으로 묶여 있던 6개(다섯 _1 외전 + pinky_2)를 관절 한계로 푼다.
    #   이걸 안 하면 액션 20칸 중 14칸만 움직이고 손이 좌우로 못 벌어진다 —
    #   "20관절 독립"이 아니라 "14관절 독립"이었다. 굴곡은 이미 한계와 같아 안 건드린다.
    hand_range_mode: str = "per_role"
    # ★손 속도는 SimToolReal 동일성(handMovingAverage 0.1). 램프는 clamp 항등으로 무력화한다.
    #   08.25 스윕(빠를수록 감쌈 악화, 단조)은 **결합된 시너지**에서 잰 값이라 적용 범위 밖이고,
    #   위험은 런을 태우는 대신 계기로 관찰한다:
    #     ctrl/hand_blocked_frac · ctrl/hand_joint_err_max · diag/obj_speed_lifted
    synergy_close_speed: float = 1.0
    synergy_close_ema: float = 0.1
    # ★위치 목표만 준다(SimToolReal 원본). 램프를 풀면 `_syn_vel` 피드포워드가 최대 20배로
    #   뛴다(≈10.8 rad/s). `grasp_fj_rh` 도 같은 이유로 0 이다. 3팔이 이 축에서 같아야 한다.
    hand_velocity_ff_scale: float = 0.0

    # ── 성공·목표·에피소드 의미 ───────────────────────────────────────────────
    # successTolerance 0.075 × keypointScale 1.5 = 0.1125 (SimToolReal.yaml:84,88)
    tol_start: float = 0.1125
    # ★tol 과 **짝**이다 — 0.2125 − 0.1125 = 0.10 = 리프트 래치. 안 올리면 물체가 4.75cm 만
    #   떠도 goal_bonus 가 나간다(REWARD_AUDIT Check 2). goal_box_z_range (0.10, 0.30) 안.
    goal_first_z_range: tuple[float, float] = (0.2125, 0.28)
    tol_success_threshold: float = 3.0          # utils.py:238 하드코딩
    goal_force_consecutive: bool = True         # 연속 10회 — cfg 가 tol_start 와 짝을 대조한다
    goal_clock_restart_step: int = 2            # 목표당 스텝 예산(왜 2 인지는 base 필드 주석)

    # ── 외란: 의도적 divergence(사용자 확정 "외란이 arm 이 못버티는 양임") ────
    # Kuka 어깨 300 N·m vs OpenArm 어깨 40 N·m = 7.5배 → 20.0/7.5 = 2.67 ≈ 2.7 · 2.0/7.5 = 0.27
    wrench_force_scale: float = 2.7
    wrench_torque_scale: float = 0.27

    # ★`goal_delta_distance`(0.08) / `goal_delta_rotation_deg`(0.0) 는 **일부러 없다** —
    #   사용자가 "목표델타는 제외" 라고 명시했다. A 값을 그대로 상속한다.


@configclass
class GraspFJTesolloRightShortEnvCfg(GraspFJTesolloRightEnvCfg):
    """DG-5F short base 판 — 프로필만 다르고 과제 정의는 GraspFJTesolloRightEnvCfg 와 동일하다.

    ★손가락 체인·관절 이름·액션 공간이 dg5f-m 과 같고 홈 palm 포즈도 IK 로 맞췄으므로
      과제 상수는 전부 그대로 유효하다(손 프레임 일치 오차 0.023mm). 달라지는 것은
      자산·fabric variant·팔 홈 관절값이며 전부 프로필이 들고 있다.
    ⚠`palm_box` 는 미검증이다 — 부팅 시 경고가 뜬다. probe 후 승격할 것.
    """

    profile_name: str = "tesollo_right_short"
    # ★09.08 팔 리셋 오프셋은 **상속하면 안 된다** — 부모의 값은 `tesollo_right` 홈에서
    #   6D IK 로 푼 **델타**라, 홈 관절값이 다른 이 프로필에 얹으면 TCP 가 어디로 갈지 모른다
    #   (부모 값의 근거였던 "손끝→물체 104.7mm" 보장이 그대로 깨진다).
    #   이 판을 SimToolReal 시작 거리에 맞추려면 **이 팔로 IK 를 다시 풀어** 값을 넣어야 한다.
    #   그때까지는 끔 = 자기 홈 그대로다(09.08 이전 거동).
    #   ⚠단, 끈 상태에서는 팔 속도(0.15 rad/s)와 시작 거리(≈250mm)가 서로 안 맞는 조합이 된다 —
    #   실측(256환경): 그 조합에서 무작위 정책은 600스텝에 203mm 중 14.4mm 밖에 못 좁힌다.
    arm_reset_offset_rad: tuple = ()
