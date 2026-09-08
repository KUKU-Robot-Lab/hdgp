"""grasp_fj_rh cfg — `GraspFJEnvCfg`(Track B) 상속, 로봇만 RH56F1 로 바꾼다.

Track B(팔 7D 관절 증분+EMA · Fabrics 없음 · 접촉 센서 0)의 **로봇 교체판**이다.
목표열·보상·관측·종료·지연·외란은 A(`grasp_kp`)/B(`grasp_fj`)와 같은 cfg 를 쓰고,
여기서 바뀌는 것은 **손이 언더액추에이션이라서 달라지는 값들**뿐이다.

왜 fabric 트랙(grasp_s2r/grasp_kp)이 아니라 B 를 베이스로 하나
--------------------------------------------------------------
RH56F1 의 fabric 은 양팔 26 DOF 클래스라 슬롯·전용 params·손끝 FK 게이트가 전부 따라온다
(09.02 grasp_ua 가 그 길을 갔고 일반화 축 5개를 새로 만들어야 했다). B 는 팔이 관절공간
증분이라 그 배선이 통째로 필요 없다 — 실기 배포도 3노드(정책 → pd → 드라이버)로 같다.

★★★09.08 사용자 확정: **SimToolReal 방식을 그대로 쓰고 로봇만 RH56F1 로 바꾼다.**
원본(`repo/simtoolreal/isaacgymenvs/tasks/simtoolreal/env.py:3801-3846`, `cfg/task/SimToolReal.yaml`)
과 대조해 우리 쪽이 갈라져 있던 두 곳을 되돌린다.

| 항목 | SimToolReal 원본 | 구 우리 값 | 지금 |
|---|---|---|---|
| 팔 스텝 | `dofSpeedScale 1.5 × dt(1/60)` = 0.025 rad, EMA 0.1 → **0.15 rad/s** | k_arm 0.167 → **1.0 rad/s** | k_arm 0.025 → 0.15 rad/s |
| 손 매핑 | `scale(a, 관절하한, 관절상한)` **절대** + EMA 0.1, 게이트·속도제한 없음 | 시너지 폐쇄도 0.005/step × close_gate | 절대 + EMA 0.1 |

★팔 슬루를 1.0 rad/s 로 올린 것은 09.06 에 "배포 브리지 상한에 맞춘다"는 이유였는데, 원본은
  상한 근처가 아니라 그 1/6.7 지점에서 돈다. 실측도 같은 방향이다 — 접근 속도를 낮출수록
  전도가 611 → 71 회로 줄고 폐쇄가 33배 올랐다(probe_fjrh_scripted).
★`close_gate` 는 원본에 **없다**(`gate`·`freeze`·`blocked` 에 해당하는 손 제어가 전무). 우리
  `grasp_s2r` 계보의 장치이고, 케이지 반경이 53mm 인 이 손에서는 17,312 epoch 내내 0.001 이라
  손이 구조적으로 못 닫혔다. 끈다.

RH56F1 이라서 바뀌는 것 (전부 실측 근거를 단다)
----------------------------------------------
1. `hand_layout="per_finger"` — 구동 6관절과 액션 6슬롯이 1:1. 채널 레이아웃(2×5=10)이면
   4지의 채널1 이 아무 관절에도 안 걸려 액션 4개가 죽는다.
2. `oppose_grip_delta_rad=0.0` — 기본 −0.6 은 **tesollo 엄지 대향(_2)** 기준이다. RH56F1 의
   ch1 은 엄지 **굴곡**(`thumb_2`, 한계 0~0.475)이라 grip 이 −0.6 으로 잘려 엄지가 300스텝
   폐쇄 지령에도 0.0001 rad 였다(09.02). 지표 어디에도 이유가 안 나오는 실패다.
3. `blocked_err_thr_rad` — 기본 1.00 은 가동폭 1.53 rad 인 이 손에서 판정이 안 선다.
4. `object_bank="shaker_small"` — 이 손의 엄지-4지 간극은 열린 자세에서 **105.5mm**,
   완전 폐쇄에서 **46.6mm** 다(09.07 probe_fjrh_calib, 중력보상 적용). cup_family(105~161mm)는 물리적으로
   안 들어간다.
★검토했다가 **채택하지 않은 것** 두 가지(09.07 실측, `our_source/rh56f1_fj_calib/`):
  · `max_depenetration_velocity` 1000 → 1 (09.02 대책): 고정 원통 폐쇄 스트레스에서
    종속 |qd|max 87.7 → 91.3 · 종속 한계위반 0.0009 → 0.0111 rad 로 **오히려 조금 나빴다**.
    둘 다 안정(NaN 0)이라 부모 기본값을 그대로 둔다 — 근거 없는 분기를 만들지 않는다.
  · 종속관절 damping 0 → 0.1: 종속 |qd|max **387** · 한계위반 4.59 rad · 구동관절이 하한
    483 rad 밖으로 나가 완전 발산했다. mimic 제약이 위치를 정하는 관절에 드라이브를 켜면
    싸운다는 09.02 기록이 실측으로 재확인됐다. **종속은 0/0 이 계약이다.**
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ..grasp_fj.grasp_fj_env_cfg import GraspFJEnvCfg
from .robot_profiles import PROFILES

#: 손 구동관절의 **토크 포화 오차**(rad) = effort / kp = 1.0 N·m / 5.0.
#: effort 는 자산 URDF 실측(모든 손 관절 1 N·m), kp 는 프로필 `right_hand_drive` 값이다.
#: 이 오차를 넘으면 관절은 더 못 움직인다 — 막힘 판정 임계는 반드시 이보다 위여야 한다.
_HAND_TORQUE_SATURATION_RAD = 1.0 / 5.0


@configclass
class GraspFJRHEnvCfg(GraspFJEnvCfg):
    """Track B · RH56F1 우손(구동 6 · 종속 6 PhysX mimic).

    차원: action 13 = 팔 7 + 손 6 · actor 94 · critic 118 (`_derive_spaces` 공식 그대로).
    """

    profile_name: str = "rh56f1_right"

    # ---- 팔 (SimToolReal 원본 값) ---------------------------------------------------
    # 원본: targets = prev + dofSpeedScale(1.5) · dt(1/60) · a = 0.025 rad · a → EMA 0.1
    #   ⇒ 스텝당 0.0025 rad = **0.15 rad/s**. 배포 브리지 상한 1.0 rad/s 를 한참 밑돈다.
    k_arm: float = 0.025
    arm_slew_rad_s: float = 0.15

    # ---- 손 (SimToolReal 원본 방식: 절대 매핑) ---------------------------------------
    # 원본은 손 액션을 관절 범위로 **직접** 스케일한다(`scale(a, lower, upper)`) + EMA.
    # 속도 제한도 게이트도 없다. `hand_ema` 는 원본 `handMovingAverage` 다.
    hand_ema: float = 0.1
    # 원본은 위치 목표만 쓴다(`set_dof_position_target_tensor`) — 속도 피드포워드 없음.
    hand_velocity_ff_scale: float = 0.0
    # ★close_gate 는 원본에 없다. 이 손(케이지 53mm)에서는 열리지 않아 폐쇄를 막기만 했다.
    close_gate_enabled: bool = False

    # ---- 손 레이아웃 -----------------------------------------------------------------
    # 구동 6관절 ↔ 액션 6슬롯 1:1. env 가 `_arm_slot_width()` 를 팔 관절 수로 덮어야
    # mixin 의 per_finger 검사가 선다(`_supports_per_finger_hand` 로 선언한다).
    hand_layout: str = "per_finger"
    # ★기본 −0.6 은 tesollo 의 엄지 **대향** 관절 기준이다 — RH56F1 에 그대로 쓰면 엄지
    #   굴곡 grip 이 한계 밖(−0.6)으로 잘려 엄지가 아예 안 접힌다(09.02 실측).
    #   이 손의 엄지 자세는 프로필의 open/grip 표가 이미 담고 있다(1.57→1.20 · 0.0→0.24).
    oppose_grip_delta_rad: float = 0.0
    # ★★09.07 두 번 고쳤다. `synergy_hold_mode="blocked"` 는 |목표−실측| > 이 값이면
    #   "외부에 막혔다"로 보고 **폐쇄 진행을 멈춘다**.
    #     · 자유 공간 정상 오차 ≤ 0.095 rad (grip 자세 정착 실측)
    #     · 토크 포화 오차 = effort/kp = 1.0/5.0 = **0.20 rad** ← 접촉하면 곧바로 여기 도달한다
    #     · 물체에 막힌 관절은 목표가 계속 전진하므로 오차가 0.9 rad 까지 커진다
    #   1차 시도 0.25 는 포화 오차 바로 위라 **첫 접촉에서 즉시 막힘**으로 판정됐다 —
    #   대본 파지 실측에서 150 스텝 완전폐쇄 지령에도 `syn_close` 가 0.01 에 얼어붙었다
    #   (손이 물체에 닿기만 하고 쥐지 못한다 → 리프트·목표가 전부 도달 불가).
    #   막힘 판정은 "닿았다"가 아니라 "더 못 조인다"를 잡아야 한다.
    #   ⇒ 0.6 = 포화의 3배 · 굴곡 가동폭(open→grip 1.08 rad)의 55%. 접촉 후에도 목표가
    #     계속 전진해 effort 한계(1 N·m)까지 조이고, 그 뒤에야 진행이 멈춘다.
    blocked_err_thr_rad: float = 0.6

    # ---- 언더액추 백스톱 ----------------------------------------------------------------
    # ★★09.07 실측(rh_b1 epoch 13~222): 손 최하단이 테이블(0.205)을 3~5cm 파고든 epoch 에서만
    #   `ctrl/mimic_err_max` 가 0.15 → 400~790 rad 로 폭주했다(정상 epoch 은 손 최하단 ≥ 0.21).
    #   메커니즘은 09.02 와 같다: 접촉이 **구동관절을 하한 밑으로 역구동**하면 mimic 이 요구하는
    #   종속 위치가 종속 관절한계 밖이 되어 두 제약이 동시에 만족 불가가 되고, 솔버가 에너지를
    #   주입한다. 자산은 리더 오버슈트 0.5 rad 여유로 한계를 넓혀 두었지만 테이블 반력은 그보다
    #   더 민다.
    #   ⇒ 종속관절 한계를 부팅에서 이만큼 **더 넓힌다**. 물리적으로 공짜다 — 종속 위치를 정하는
    #     권한은 mimic 제약이고 관절한계는 백스톱일 뿐이다(09.02 결론).
    #   0 이면 자산 값 그대로(비교 실험용).
    mimic_dep_limit_margin_rad: float = 1.5

    # ---- 팔 목표 상자 --------------------------------------------------------------------
    # ★★09.07 실측 3연속(rh_b3·b4·b6): 팔 적분기가 e25 안에 관절 한계까지 표류하고
    #   (`ctrl/arm_limit_sat` 0 → 0.27 → 0.49) 손이 물체에서 멀어진 채(`ft_dist` 0.13 → 0.69)
    #   돌아오지 않는다. 증분 매핑에는 복원력이 없어 반사 경계의 정상분포가 관절 범위 전체다.
    #   ±0.5 rad 이면 과제가 요구하는 협응(케이지 83mm 이동 ≈ 0.28 rad)을 덮으면서 표류를 막는다.
    #   ⚠형제 tesollo B(fj_b1)는 이 상자 없이 학습됐다 — 부모 기본값은 0(끔)이다.
    #   ⚠09.08 재판정: 이 값을 정한 실측(rh_b3·b4·b6 의 표류)은 **물체가 잘못 소환된 상태**에서
    #     나온 것이라 근거가 무효다. 형제 tesollo B(fj_b1)는 상자 없이 학습됐으므로 **끈다**.
    #     물체를 고친 뒤 표류가 다시 보이면 그때 근거를 새로 만들어 켠다.
    arm_target_box_rad: float = 0.0

    # ---- 물체 -------------------------------------------------------------------------
    # ★★09.08 사용자 지시 "셰이커로 하고" — **단일** 셰이커(0.65 · 지름 57mm · 높이 114mm).
    #   크기 8종(shaker_small)은 파지 창에 다 들어오지만 다물체는 `replicate_physics=False` 를
    #   강제하고, 첫 과제 성립을 보는 단계에서 원인을 섞는다.
    object_bank: str = "shaker_one"
    # ⚠09.08 재판정: 아래 "전도 대책" 실측은 전부 **잘못 소환된 물체**(셰이커 높이에 놓인 컵)
    #   위에서 잰 것이라 무효다. 파지 높이는 부모 기본값으로 되돌린다 — 형제와 같은 값이다.
    # (구 주석) 전도 대책 ①: 파지 높이를 물체 **원점**으로 내린다(부모 0.03).
    #   셰이커는 종횡비 1.99(지름 57 · 높이 114mm)라 원점보다 30mm 위를 밀면 전도 모멘트가 커진다.
    #   대본 파지 실측에서 손가락이 닿는 즉시 60°(`tilt_reset_deg`)를 넘어 `done/tipped` 로
    #   종료됐고, 그 때문에 `ep_len` 이 6~17 스텝에 머물러 폐쇄가 0.014 이상 못 올라갔다.
    #   원점은 바닥+60mm = 높이의 53% 라 무게중심에 가깝다.
    object_grasp_z_offset: float = 0.03
    # ★★09.08 전도 대책 ②: 물체 각감쇠(0 = 부모 기본). 07.22 선행 rh56f1 트랙이 같은 증상에
    #   질량 0.15→0.20 + 각감쇠 10 으로 대응한 전례가 있다. 0 이면 아무것도 안 바꾼다.
    object_angular_damping: float = 0.0
    # ★★09.08 전도 대책 ③: 물체 질량(0 = 뱅크 값 0.134 그대로). 무거울수록 손가락 한쪽이
    #   먼저 닿아도 덜 밀린다. 07.22 선행 트랙 전례는 0.20.
    object_mass_override: float = 0.0

    # ★★★09.08 전도 종료 제거 — reward-audit ACCEPT.
    #   실측: 대본 probe 4회의 리셋 사유가 **100% `tipped`**(16/16 · 17/17 · 16/16 · 18/18).
    #   `tipped` 은 truncation 이 아니라 `terminated` 라 value_bootstrap 이 V(s')=0 을 넣는다.
    #   작동점 1.095/step · γ 0.99 · 잔여 500스텝이면 전도 1회의 가치 손실이 ≈109 인데,
    #   리프트 보너스 300 은 10cm 를 들어야 나오는 일회성이다. 즉 "닿으면 확정 −109,
    #   잡으면 어쩌면 +300" 이라 정책이 **손을 편 채 안 닿는 법**을 배웠다(실측: 4지 굴곡
    #   지령이 150스텝 안에 0.000 으로 포화한 뒤 에피소드 끝까지 유지).
    #   ★참조에는 이 종료가 없다 — SimToolReal `_compute_resets` 는 낙하(z<0.1)·에피소드
    #   길이·hand_far_from_object·(옵션)테이블 힘뿐이다. 우리가 덧붙인 항이다.
    #   손안 판정(팔·접근·리셋 배제)에서 엄지 1.57·4지 0.90 자세가 8env 중 7개를 중력에
    #   대해 유지했으므로 하드웨어는 무죄이고, 막힌 것은 접촉 전이가 잘려나가는 것이다.
    #   ⚠부모 기본값(60.0)은 건드리지 않는다 — 형제 트랙 불변.
    #   ⚠판정에 `tilt_deg` 를 쓰지 말 것. 이 변경 후 오르는 것이 정상이다.
    tilt_reset_deg: float = 179.0

    # ------------------------------------------------------------------
    # 파생 (멱등 — env `__init__` 이 super() 전에 다시 부른다)
    # ------------------------------------------------------------------
    def _supports_per_finger_hand(self) -> bool:
        """이 트랙의 env 가 `_arm_slot_width()` 를 팔 관절 수로 덮는다."""
        return True

    def finalize_after_overrides(self) -> None:
        super().finalize_after_overrides()      # robot_cfg 재조립 · 차원 파생 · B 필드 검증
        # ★물체 spawn 은 뱅크 적용에서 새로 조립되므로 여기서 덮어야 한다(`__post_init__` 은 늦다).
        _d = float(self.object_angular_damping)
        _m = float(self.object_mass_override)
        if _d > 0.0 or _m > 0.0:
            from isaaclab.sim.schemas import MassPropertiesCfg as _MP
            _spawns = getattr(self.object_cfg.spawn, "assets_cfg", None) or [self.object_cfg.spawn]
            for _sp in _spawns:
                if _d > 0.0 and getattr(_sp, "rigid_props", None) is not None:
                    _sp.rigid_props.angular_damping = _d
                if _m > 0.0:
                    _sp.mass_props = _MP(mass=_m)
        self._validate_rh_fields(PROFILES[self.profile_name])

    def _validate_rh_fields(self, profile) -> None:
        """언더액추 손에서 **조용히 무효가 되는 조합**을 cfg 단계에서 죽인다."""
        errs = []
        if str(self.hand_layout) != "per_finger":
            errs.append(
                f"hand_layout 은 per_finger 여야 한다(구동 {profile.num_hand_joints}관절 1:1), "
                f"got {self.hand_layout}")
        # ★09.02 실패의 재발 차단: 노브가 grip 을 open 과 같게(=안 움직이는 관절) 만들거나
        #   관절 한계 밖으로 밀면 지표에 아무 흔적 없이 그 손가락이 죽는다.
        if float(self.oppose_grip_delta_rad) != 0.0:
            errs.append(
                "oppose_grip_delta_rad 는 tesollo 엄지 대향(_2) 전용이다 — RH56F1 의 ch1 은 "
                "엄지 굴곡이라 grip 이 한계 밖으로 잘린다(0.0 으로 두고 프로필 자세표를 쓴다)")
        if float(self.mimic_dep_limit_margin_rad) < 0.0:
            errs.append("mimic_dep_limit_margin_rad 는 ≥ 0 (0 = 자산 한계 그대로)")
        # ★★09.07 정정 — 이 검사는 처음에 **반대로** 걸려 있었다("0.5 이상이면 판정이 안 선다").
        #   대본 파지 실측이 뒤집었다: 임계가 토크 포화 오차(effort/kp = 1.0/5.0 = 0.20 rad)에
        #   가까우면 **첫 접촉에서 곧바로 막힘으로 판정**되어 폐쇄가 syn_close 0.01 에서 얼어붙는다
        #   (150 스텝 완전폐쇄 지령에도 손이 안 쥐어진다). 막힘 판정은 "더 못 조인다"를 잡아야지
        #   "닿았다"를 잡으면 안 된다.
        #   ⇒ 임계는 포화 오차의 **2배 이상**이어야 하고, 관절 가동폭(open→grip 최대 1.08 rad)을
        #     넘으면 영영 안 걸리므로 그 아래여야 한다.
        # ★SimToolReal 정합 — 원본과 갈리면 안 되는 값들.
        if abs(float(self.k_arm) - 0.025) > 1e-9 or abs(float(self.arm_ema) - 0.1) > 1e-9:
            errs.append(
                f"팔 스텝이 원본과 다르다: k_arm {self.k_arm}·EMA {self.arm_ema} "
                "(원본 dofSpeedScale 1.5 × dt 1/60 = 0.025 · armMovingAverage 0.1)")
        if abs(float(self.hand_ema) - 0.1) > 1e-9:
            errs.append(f"hand_ema {self.hand_ema} ≠ 원본 handMovingAverage 0.1")
        if bool(self.close_gate_enabled):
            errs.append("close_gate 는 원본에 없다 — 이 손에서는 열리지 않아 폐쇄를 막는다")
        if float(self.hand_velocity_ff_scale) != 0.0:
            errs.append("원본은 위치 목표만 준다 — hand_velocity_ff_scale 은 0 이어야 한다")
        # `blocked_err_thr_rad`·`synergy_close_speed`·`hand_finger_channels` 는 절대 매핑에서
        # **소비되지 않는다**(시너지 경로를 안 탄다). 값은 부모 기본값 그대로 둔다.
        # ★종속관절은 **0/0** 이어야 한다 — 09.07 실측: damping 0.1 만 줘도 mimic 제약과 싸워
        #   구동관절이 하한 483 rad 밖으로 나가며 발산했다.
        for _n in ("right_hand_mimic", "left_hand_mimic"):
            _spec = profile.actuator_specs.get(_n)
            if _spec is not None and (_spec.get("stiffness"), _spec.get("damping")) != (0.0, 0.0):
                errs.append(f"{_n} 게인이 0/0 이 아니다 — PhysX mimic 제약과 싸운다")
        if errs:
            raise RuntimeError("[grasp_fj_rh cfg] " + " · ".join(errs))


@configclass
class GraspFJRH56F1RightEnvCfg(GraspFJRHEnvCfg):
    profile_name: str = "rh56f1_right"
