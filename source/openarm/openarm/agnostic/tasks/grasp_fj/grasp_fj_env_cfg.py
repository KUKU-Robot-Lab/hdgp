"""grasp_fj cfg — `GraspKPEnvCfg` 상속, Track B(팔 7D 관절 증분 + EMA, Fabrics 없음).

DESIGN.md §1 B 열. 목표열·보상·관측·종료·DR 은 전부 A(`grasp_kp`)와 **같은 cfg** 를
쓴다 — 여기서 바뀌는 것은 팔 액션 어댑터 세 값뿐이다:
- `arm_cmd_dim = n_arm`(obs 의 cmd_state = 직전 팔 목표 q*_{t-1}),
- `k_arm`(rad/step per unit action), `arm_ema`(α).
차원은 A 의 `_derive_spaces` 공식을 그대로 쓰고 `_arm_action_dim` 훅만 n_arm 으로 바꾼다
(tesollo_right: action 22 · actor 131 · critic 155).
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ..grasp_kp.grasp_kp_env_cfg import GraspKPEnvCfg
from .robot_profiles import PROFILES


@configclass
class GraspFJEnvCfg(GraspKPEnvCfg):
    """SimToolReal 식 트랙 B: 팔 관절 7D 증분+EMA(위치 목표만) + 시너지 15D, 접촉 항 0개.

    왜 B 인가: 실기 배포에서 fabric 노드를 빼고(4노드→3노드) 정책 출력을 pd 노드에 바로
    먹이기 위해서다. 팔 속도 목표는 주지 않는다 — 실기 JTC 가 velocity 를 쓰지 않는다.
    """

    # obs cmd_state 폭 = 직전 팔 목표 q*_{t-1}(n_arm). finalize 가 프로필과 대조한다.
    arm_cmd_dim: int = 7
    # 왜 0.167: EMA 가 누적 목표에 걸리므로 스텝당 변화는 정확히 α·k_arm·a 다 — |a|=1 에서
    #   0.1·0.167 = 0.0167 rad/step = 1.0 rad/s(브리지 상한). 구 0.0167 은 실효 0.1 rad/s 로
    #   설계 문구와 10배 어긋났다(09.06 리뷰). A 의 palm 리미터(1.2 m/s)와 같은 자릿수.
    k_arm: float = 0.167
    # 왜 0.1: 목표 EMA(SimToolReal α). q*_t = α·q_raw + (1−α)·q*_{t-1} — 실효 slew 는 α·k_arm/step.
    arm_ema: float = 0.1
    # 선언된 포화 slew(rad/s). finalize 가 α·k_arm/policy_dt 와 대조한다 — 문구와 실효값이 못 갈린다.
    arm_slew_rad_s: float = 1.0
    # ★09.07 B-v: 리프트 후 팔 액션 **반전** 벌점. 측도 = 1차 차분 RMS/2 ∈ [0,1](A 는 비유계·작동점 ≈10× 라 0.1).
    #   최악(매 스텝 ±1 반전) −1.0/step, γ 0.99 할인 합 −100 < 리프트 보너스 300 → 리프트가 여전히 이득(Check 1).
    #   전속 이송(a 일정)은 차분 0 이라 세금이 없다 — 수준 |a| 를 벌하면 이송까지 벌한다.
    rw_cmd_rate_scale: float = 1.0

    def _arm_action_dim(self, profile) -> int:
        """액션의 팔 구간 폭 = 관절 수(B). A 의 `_derive_spaces` 가 이 훅으로 22 를 만든다."""
        return int(profile.num_arm_joints)

    def finalize_after_overrides(self) -> None:
        super().finalize_after_overrides()          # A: 박스·kp 필드 검증·차원(이 클래스의 훅으로)
        self._validate_fj_fields(PROFILES[self.profile_name])

    def _validate_fj_fields(self, profile) -> None:
        """B 신설 필드 범위 + A 와 갈릴 수 있는 조합을 cfg 단계에서 죽인다."""
        errs = []
        if int(self.arm_cmd_dim) != int(profile.num_arm_joints):
            errs.append(f"arm_cmd_dim {self.arm_cmd_dim} ≠ num_arm_joints {profile.num_arm_joints}")
        if float(self.k_arm) <= 0.0:
            errs.append(f"k_arm 은 > 0, got {self.k_arm}")
        if not (0.0 < float(self.arm_ema) <= 1.0):
            errs.append(f"arm_ema 는 (0, 1], got {self.arm_ema}")
        _dt = float(self.sim.dt) * int(self.decimation)                 # 정책 스텝
        _slew = float(self.arm_ema) * float(self.k_arm) / _dt          # 실효 포화 slew(rad/s)
        if abs(_slew - float(self.arm_slew_rad_s)) > 0.02 * float(self.arm_slew_rad_s):
            errs.append(f"실효 slew α·k_arm/dt = {_slew:.3f} rad/s ≠ arm_slew_rad_s {self.arm_slew_rad_s}")
        # 부모 mixin `_setup_synergy` 의 per_finger 검사가 팔 폭을 6 으로 박아 두었다 — B 와 맞지 않는다.
        if str(self.hand_layout) == "per_finger":
            errs.append("hand_layout=per_finger 는 B 에서 미지원(mixin 이 팔 폭 6 을 가정)")
        if errs:
            raise RuntimeError("[grasp_fj cfg] " + " · ".join(errs))


@configclass
class GraspFJTesolloRightEnvCfg(GraspFJEnvCfg):
    profile_name: str = "tesollo_right"
