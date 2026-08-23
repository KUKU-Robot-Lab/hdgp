"""grasp_lift_fabric 보상 — 7항.

**항 범주는 grasp_v2(`common/grasp_reward_core.py`)에서, 게이트 구조는 새로 짰다.**

왜 grasp_lift(diffIK 트랙)의 5항을 쓰지 않는가:
    그쪽 success 는 "물체가 goal 위치에 있으면 끝"이라 **파지 품질·자세·유지에
    gradient 가 하나도 없다**(보상에 회전 항이 아예 없다). 그래서 엄지가 컵 입구를
    가로지르는 평면 핀치 같은 축퇴를 구조적으로 허용한다.

★귀속 정정(08.22): 아래 ①②의 래치/5중첩은 **grasp_v1 과 grasp_reward_core(코어)** 의
구조다. 활성 grasp_v2 는 코어를 호출하지 않는다(4항 인라인, 래치 없음) — 조사로 확정.
    ① 래치(v1)가 걸리면 `grasp`(가중 12.0)가 즉시 꺼져 기여가 0.0362 → 0.0358 로
       **평탄**했다. "만렙 후 감쌈 침식"의 직접 원인. → **래치를 쓰지 않는다.**
    ② 코어의 `stabilize` 는 게이트 5중첩(lift × lifted × contact × upright × action)이라
       하나가 0이면 전부 0 → 원인 분리 불가. → **곱셈은 최대 2단**, 자세는 0 이 아니라
       **0.5~1.0 완화 곱수**로 넣는다.
    ③ v1 은 밀기 페널티가 `approach` 안에 묻혀 pre-lift 게이트와 함께 꺼졌다.
       (v2 는 밀기 페널티 자체를 삭제 — object_to_goal 과 충돌해서.)

08.22 TEST1 (reward-audit ACCEPT):
    · **A. 대향 파지점 approach** — 손끝 목표를 중심 → 대향 파지점(v1/v2 이식).
      중심 기준은 전 손끝을 같은 점으로 당기는 shaping + 기하 상한 0.57 이었다.
    · **B. persistence 재도입** — credit 0.5/0.3/0.2 (v1/v2 도 0.25 로 유지하는 항).
    · **C/D. push·tilt 페널티 제거** — 실측 기여 −0.004/−0.12 무용, 종료(0.35m/60°)가
      상한 담당. up_mul 은 유지.
    · **E. 참여 임계 0.1N 분리** — 게이트는 1.0N 유지(스침 success 차단).

reward-audit 판정 이력 (REVISE → ACCEPT):
    · `upright` 독립 항은 컵이 애초에 서 있어 사실상 공짜였다(접촉 보너스 중복)
      → 독립 항 제거, lift/success 의 완화 곱수로만 사용 + tilt_penalty 신설(→08.22 재제거).
    · penalty 가 무한대라 멀리 밀린 뒤 회복 불가 → **상한 클램프**(접근 회피 방지,
      agn_test2 의 종료-회피와 같은 실패 축).
    · `persistence` 는 래치를 없앤 뒤 중복 → 제거(envelope 0.6 / grip 0.4).

정체(테이블 위 감쌈 유지) 4.0 : 추가(리프트+성공) 12.0 = 1:3.0.
"""

from __future__ import annotations

import torch


def _cfg(cfg: object, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def contact_gate(
    group_a_force: torch.Tensor, group_b_force: torch.Tensor, threshold: float
) -> torch.Tensor:
    """대향 접촉 = 파지 성립. 5지면 엄지 AND 나머지, 2지면 조1 AND 조2 — 같은 코드."""
    a = (group_a_force > threshold).any(dim=-1)
    b = (group_b_force > threshold).any(dim=-1)
    return a & b


def upright_quality(object_tilt_deg: torch.Tensor, exponent: float = 4.0) -> torch.Tensor:
    """1(수직) → 0(90° 기움). `cos(tilt)^exponent`.

    ★08.23 선형(1−tilt/max_deg) 에서 교체. 자매 트랙 grasp_sensor 가 같은 문제로
      먼저 도달한 규약이다 — cos 는 소각에서 평평해 15~30° 판별력이 없어서 지수를 준다.
      선형은 전 구간 도당 기울기가 상수(1/60=0.0167)라 "이미 많이 기운" 구간에서
      개선 압력이 약했다. cos^4 는 20~30° 에서 0.020~0.023 으로 더 가파르다.
      실측 tilt 38.5° 대역에서 1.2 배.
    """
    c = torch.cos(torch.deg2rad(object_tilt_deg)).clamp(0.0, 1.0)
    return c ** float(exponent)


def compute_rewards(
    *,
    palm_to_object: torch.Tensor,     # (N,)  palm ↔ 물체 거리
    tip_side_dist: torch.Tensor,      # (N,)  손끝 평균 ↔ 물체 거리
    envelope_frac: torch.Tensor,      # (N,)  감쌈(중간·원위 마디) 접촉 손가락 비율
    grip_frac: torch.Tensor,          # (N,)  아무 마디든 접촉한 손가락 비율
    persistence: torch.Tensor,        # (N,)  대향 게이트 연속 유지 비율 (0..1)
    object_height_delta: torch.Tensor,
    object_to_goal: torch.Tensor,
    object_xy_displacement: torch.Tensor,
    object_tilt_deg: torch.Tensor,
    group_a_force: torch.Tensor,
    group_b_force: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    cfg: object,
) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
    """Returns: (total, terms, gate, upright_q)."""
    gate = contact_gate(
        group_a_force, group_b_force, _cfg(cfg, "contact_force_threshold", 1.0)
    )
    gate_f = gate.float()
    up_q = upright_quality(object_tilt_deg, _cfg(cfg, "upright_exponent", 4.0))
    # ★08.23 `up_mul`(lift/success 에 곱하던 0.5~1.0 자세 완화 곱수) 제거.
    #   자세 압력을 곱수와 독립항에 이중으로 걸면 원인 분리가 안 되고, 곱수는
    #   최대 2 배 차이라 실측에서 tilt 38.5°→10° 개선 이득이 +0.09 에 그쳤다.
    #   자세는 `upright` 항이 **전담**한다(자매 트랙 grasp_sensor 와 동일 구조).

    # 1. approach — 게이트 없음(접촉 전 유일한 gradient)
    approach = _cfg(cfg, "approach_weight", 1.0) * torch.exp(
        -_cfg(cfg, "approach_sharpness", 4.0) * (palm_to_object + tip_side_dist)
    )

    # 2. grasp_quality — ★래치 없이 **상시** 켜져 있다. 리프트 후 감쌈이 풀리면
    #    바로 깎이므로 "만렙 후 감쌈 침식"에 반대 gradient 가 생긴다.
    #    persistence = 대향 게이트 연속 유지 비율(0..1) — 잡았다-놓기 축퇴 억제.
    _env_credit = _cfg(cfg, "grasp_envelope_credit", 0.5)
    _grip_credit = _cfg(cfg, "grasp_grip_credit", 0.3)
    _per_credit = _cfg(cfg, "grasp_persist_credit", 0.2)
    grasp_quality = (
        _cfg(cfg, "grasp_quality_weight", 3.0)
        * gate_f
        * (_env_credit * envelope_frac.clamp(0.0, 1.0)
           + _grip_credit * grip_frac.clamp(0.0, 1.0)
           + _per_credit * persistence.clamp(0.0, 1.0))
    )

    # ★★08.22 우선순위 재설계 (사용자 지시): ①인벨롭 그립 ②컵 똑바로 들기 ③이송.
    #   구 설계는 정확히 반대였다 — envelope 를 0.50→1.00 으로 **완성**해도 이득이 +0.59
    #   인데 "대충 잡고 들기"만으로 lift 가 이미 0.79 였다(우팔 ep950 실측 분해).
    #   그래서 정책은 감쌈을 만들 이유가 없었다.
    #
    #   env_mul: 감쌈 성립도. lift/이송은 **인벨롭 그립 위에서만** 온전히 열린다.
    #   ★하한 0.3 — 0 으로 두면 초기(envelope≈0)에 리프트 신호가 통째로 죽어
    #     "들기"를 못 배우고, 들지 못하면 감쌈도 못 배우는 폐루프가 된다(audit Check 4).
    _env_ref = max(_cfg(cfg, "envelope_reference_frac", 0.8), 1e-6)
    env_mul = _cfg(cfg, "envelope_mul_floor", 0.3) + (
        1.0 - _cfg(cfg, "envelope_mul_floor", 0.3)
    ) * (envelope_frac / _env_ref).clamp(0.0, 1.0)

    # 3. upright — ★신설 독립항(우선순위 ②). 곱수만으로는 압력이 없었다:
    #    up_mul 은 0.5~1.0 이라 tilt 31°→10° 개선의 이득이 +0.09 에 불과했다.
    #    ★lifted 곱수 필수 — 없으면 테이블 위 컵(tilt 0°)을 감싸기만 해도 만점이다
    #      (audit Check 2: 들지 않고 보상을 최대화하는 경로).
    lifted = (object_height_delta / max(_cfg(cfg, "upright_lift_ref", 0.05), 1e-6)).clamp(0.0, 1.0)
    upright = _cfg(cfg, "upright_weight", 3.0) * gate_f * env_mul * up_q * lifted

    # 4. lift — ★dz 0.10 이상이 전 구간 만점이라 **보상 평지**였다(실측 dz 0.27 표류).
    #    goal 높이(0.15)+여유(0.05) 를 넘으면 감쇠시켜 "필요 이상으로 올리지 마라"를 준다.
    #    0.10 미만 구간은 구 설계와 **동일**하다(파지·리프트 초기 학습 불변).
    height_q = (
        object_height_delta / max(_cfg(cfg, "lift_success_height", 0.10), 1e-6)
    ).clamp(0.0, 1.0)
    _over = (object_height_delta - _cfg(cfg, "lift_overshoot_start", 0.20)).clamp(min=0.0)
    height_q = height_q * torch.exp(-_over / max(_cfg(cfg, "lift_overshoot_scale", 0.06), 1e-6))
    lift = _cfg(cfg, "lift_weight", 2.0) * gate_f * env_mul * height_q

    # 5. tracking — ★08.23 신설. success 의 `tanh(d/0.05)²` 는 **5cm 밖에서 사실상 0**
    #    이다(d=8cm 계수 0.006). dz 가 0.09~0.15 로 진동하는 것만으로 goal 에서 벗어나
    #    이송 신호가 끊겼고, 실측에서 success 가 0.24 까지 갔다가 **0.000** 으로 소멸했다.
    #    완만한 유도(std 0.10)와 날카로운 성공(std 0.05)을 **분리**한다 — 자매 트랙
    #    grasp_sensor 가 tracking 2.0 / success 10.0 으로 쓰는 구조와 같다.
    tracking = (
        _cfg(cfg, "tracking_weight", 2.0)
        * (1.0 - torch.tanh(object_to_goal / _cfg(cfg, "tracking_std", 0.10)))
        * gate_f
        * env_mul
    )

    # 6. success (이송 성공)
    success = (
        _cfg(cfg, "success_weight", 10.0)
        * (1.0 - torch.tanh(object_to_goal / _cfg(cfg, "success_pos_std", 0.05))) ** 2
        * gate_f
        * env_mul
    )

    # (구 5·6 push/tilt 페널티는 08.22 제거 — 0.35m 이탈·60° 전도가 **종료**로 승격되어
    #  상한 압력을 종료가 담당한다. 실측 기여도 −0.004/−0.12 로 무용했다.)

    # 5. action_rate — ★**mean** 이고 clamp 이 없다. 둘 다 실측 근거가 있다.
    #   · sum 이면 액션 차원에 비례한다(같은 지터에서 26D=12.0 / 18D=8.3 / 7D=3.2).
    #     robot-agnostic 트랙에서 같은 weight 가 로봇마다 다른 압력이 되므로 mean 을 쓴다.
    #   · clamp(max=1.0) 은 실측 sum 12.0 에서 **12배 포화**해 페널티를 상수로 만들었다.
    #     지터가 12→1 로 줄 때까지 gradient 가 0 이었다(fab_test2 iter15 실측).
    #     grasp_v2 의 래치가 grasp 항을 죽인 것과 같은 부류의 결함이다.
    #   mean((Δa)²) 는 자연 상한 4.0(매 스텝 완전 반전) 이라 clamp 가 필요 없다.
    action_rate = _cfg(cfg, "action_rate_weight", -0.3) * torch.mean(
        (actions - prev_actions) ** 2, dim=-1
    )

    terms = {
        "approach": approach,
        "grasp_quality": grasp_quality,
        "upright": upright,
        "lift": lift,
        "tracking": tracking,
        "success": success,
        "action_rate": action_rate,
    }
    total = torch.nan_to_num(sum(terms.values()), nan=0.0, posinf=0.0, neginf=0.0)
    return total, terms, gate, up_q
