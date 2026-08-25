"""tip 모드(손끝 IK) 전용 보상 — 08.24 단계형 재구성(사용자 확정). 로봇 무관 텐서 함수.

★별도 파일인 이유: `grasp_sensor/rewards.py` 는 자매 트랙(타 세션 소유)과 공유하고
그쪽 계약 테스트가 파일 전체의 `cfg.X` 키를 자기 cfg 와 대조한다. 공용 소부품만 가져온다.

단계형 구조(가중치가 곧 단계 상한 — 다음 단계가 항상 더 크다):

    접근 1  →  파지 2  →  감쌈 3  ‖  리프트 5  →  이송 8  →  성공 12
    └─────── 무게이트 ───────┘     └──── 대향접촉 게이트 ────┘

  · 접근  = 파지중심(palm 고정점, 부팅 실측 palm+(35,∓2,147)mm)을 물체에 붙인다.
            충돌 페널티 없음 — 쓰러뜨려도 리셋되니 빠르게 접근하는 것부터 배운다.
  · 파지  = wrap 마디(_3/_4)를 물체 쪽으로. 손가락별 **mean** 이라 각자 자기 gradient.
            그 손가락이 닿는 순간 off(압입 차단) — 팁이 아니라 **마디**를 당기는 이유는
            팁을 당기면 최적이 핀치가 되고 팁 압입(자매 실측 28~46N)이 되기 때문.
  · 감쌈  = 손바닥면 접촉 마디 부분점수 5 지 mean(pinky 포함).
  · 파지력은 보상이 지정하지 않는다 — 놓치면 리프트·이송·성공을 잃으므로 무게·마찰에
    맞게 정책이 스스로 정한다.
  · 형상 비의존 — 보상이 쓰는 값은 obs 로 계산 가능한 것뿐(pose·링크 위치·접촉력).
    물체 반경·높이 상수를 쓰던 자세 3 항(radial/band/spread)은 이 원칙 위반으로 제거.
    유일한 길이 상수 grip_dist_floor(9mm)는 **손 기하**(팁 반경)다.

이전 반복의 교훈(회귀 방지용 기록):
  · approach 기준을 palm 원점으로 두면 최적점 d=0 = "손바닥으로 컵 관통"(유효 파지는
    ~150mm 뒤). max(palm+손끝) 커널은 palm 이 argmax 를 52.7% 점유해 접근을 잘할수록
    손가락 gradient 가 소멸(자매 lstm_test5: gate 2,820ep 내내 0.000).
  · action_l2 에 손을 넣으면 tip 모드에서 a=0=펴진 손이라 "손을 펴라"는 역압이 된다.
  · 감쌈 판정이 방향 무관이면 손등 접촉을 감쌈으로 센다(자매 실측 middle_4 손등 100%,
    env_frac 0.746 vs 정직 0.55) — 호출부(env)가 palmar 필터를 적용해 넘긴다.
"""

from __future__ import annotations

import torch

from ..grasp_sensor.rewards import (
    action_l2_clamped,
    action_rate_l2_clamped,
    contact_gate,
)


def envelope_fraction_graded(
    mid_force: torch.Tensor,   # (N, E) envelope 손가락별 중간마디(_3) 접촉력
    dist_force: torch.Tensor,  # (N, E) envelope 손가락별 원위마디(_4) 접촉력
    threshold: float,
) -> torch.Tensor:
    """감싼 **마디** 비율: mean_f 0.5·(1[mid] + 1[dist]) ∈ [0,1].

    공유 `envelope_fraction` 은 손가락별 OR 이라 **_3 만 닿아도 그 손가락을 1 로 센다**
    (사용자 지적). 받치기와 감쌈이 같은 점수가 되고, 실측으로 같은 정책이 느슨(OR)
    0.50 · 엄격(전 마디 AND) 0.069 로 7 배 벌어진다.

    AND 로 가지 않는 이유: 손가락마다 닿는 마디가 다르다(grasp_v1 실측 — 정책에 따라
    엄지가 tip 0.907/_4 0.249 이거나 tip 0.183/_4 0.808). AND 는 유효한 파지도 0 으로
    세고 0.069 대역이라 초기 gradient 가 없다. 부분 점수는 0 → 0.5 → 1.0 사다리를
    만들어 받치기와 감쌈을 구분하면서 gradient 를 남긴다.

    rim-hook 반증(구 v2 공식 `0.5·(mid평균+dist평균)` 이 0.56 을 준 건)은 **접촉 깊이**를
    쟀기 때문이다. 이 함수는 이진 접촉의 평균이라 깊이로 올릴 수 없다:
    검지만 두 마디 = 0.2 · 두 손가락 = 0.4 · 5 지 전 마디 = 1.0.
    팁 접촉은 여기 없다 — 핀치(팁만)로는 이 항이 0 이다.
    """
    graded = 0.5 * ((mid_force > threshold).float() + (dist_force > threshold).float())
    return graded.mean(dim=-1)


def compute_tip_rewards(
    palm_pos: torch.Tensor,           # (N, 3) env-local (진단용 — 보상 항에는 안 쓴다)
    grasp_center_pos: torch.Tensor,   # (N, 3) env-local — palm 에 고정된 파지중심(손끝 홈 평균)
    fingertip_pos: torch.Tensor,      # (N, T, 3) env-local (진단용)
    object_pos: torch.Tensor,         # (N, 3) env-local
    goal_pos: torch.Tensor,           # (N, 3) env-local
    object_up: torch.Tensor,          # (N,) 물체 local +z · world +z = cos(기울기)
    tcp_normal_z: torch.Tensor,       # (N,) 손 TCP(palm_ee) +x 축의 **world z 성분**
    group_a_force: torch.Tensor,      # (N, Fa) 손가락별(합산) 접촉력
    group_b_force: torch.Tensor,      # (N, Fb)
    env_mid_force: torch.Tensor,      # (N, E) ★호출부가 palmar 필터를 이미 적용한 값
    env_dist_force: torch.Tensor,     # (N, E) ★동
    wrap_body_pos: torch.Tensor,      # (N, F, P, 3) wrap 마디 위치 env-local (P=마디 수)
    finger_touch: torch.Tensor,       # (N, F) bool — 방향 **무관** 접촉(참여 임계)
    actions: torch.Tensor,            # (N, 6+3T)
    prev_actions: torch.Tensor,
    n_arm_actions: int,               # 팔 액션 폭(=6). 손 성분은 action_l2 에서 제외.
    cfg: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """returns (total, terms, gate, envelope_frac) — 구 함수와 동일 계약.

    finger_touch 가 방향 무관인 것은 **의도된 비대칭**(자매 규약): 손등으로 밀어도
    파지 항의 당김은 멈추지만(압입 방지) 감쌈 점수는 손바닥면만 — 자세를 고쳐야
    점수가 오른다.
    """
    thr = float(cfg.contact_force_threshold)
    g = contact_gate(group_a_force, group_b_force, thr)
    gf = g.float()
    env_frac = envelope_fraction_graded(env_mid_force, env_dist_force, thr)

    # ── 무게이트 3 단계 ──────────────────────────────────────────────────────
    # ① 접근(1) — 파지중심 → 물체. d=0 = 컵이 손 한가운데(관통 아님).
    d_grasp = torch.norm(grasp_center_pos - object_pos, dim=-1)
    approach = torch.exp(-float(cfg.approach_sharpness) * d_grasp)

    # ①-b 접근 **정렬**(0.5) — 손바닥 법선이 컵 축과 **수직**이어야 측면 파지가 된다.
    #   ★컵 축을 추정하지 않는다: 컵은 중력 반대로 서 있어야 하므로(과제 요구이고
    #     upright 항이 그 압력을 준다) cup_z ≡ world +z 다. 그러면 조건이
    #     |dot(n_tcp, ẑ)| = |n_tcp.z| → 0 으로 줄어 **회전 추정이 불필요**하다.
    #     정책도 같은 값을 obs(TCP 자세 x축의 z 성분)로 직접 본다 — 부분관측이 아니다.
    #   ★거리 커널을 곱하는 이유: 안 곱하면 컵에서 멀리 떨어져 자세만 맞추는 것이
    #     만점이 된다(reward-audit Check 2). approach 와 같은 커널을 써서 "가까이 가되
    #     바른 자세로" 를 하나의 표면으로 만든다.
    #   ★approach 에 곱수로 넣지 않는다 — 유일하게 작동 중인 초기 신호를 깎는다.
    #     floor 도 두지 않는다(08.22 envelope_mul_floor 0.3 이 30% 유출을 만든 선례).
    align_cos = tcp_normal_z.abs().clamp(0.0, 1.0)      # 0 = 수직(목표) · 1 = 평행
    align = (1.0 - align_cos) * approach

    # ② 파지(2) — wrap 마디를 물체 쪽으로, 닿으면 그 손가락 off.
    #    floor(팁 반경) 아래는 보상 증가 없음 — 접촉 감지가 실패해도 압입이
    #    무한보상이 되지 않는다.
    d_wrap = torch.norm(
        wrap_body_pos - object_pos[:, None, None, :], dim=-1).mean(dim=-1)   # (N, F)
    open_f = (~finger_touch).float()
    grip_kernel = torch.exp(
        -float(cfg.grip_sharpness)
        * (d_wrap - float(cfg.grip_dist_floor)).clamp(min=0.0))
    grip = (open_f * grip_kernel).mean(dim=-1)

    # ③ 감쌈(3) — 손바닥면 마디 부분점수(위에서 계산).

    # ── 대향접촉 게이트 3 단계 ───────────────────────────────────────────────
    d_goal = torch.norm(object_pos - goal_pos, dim=-1)
    theta = torch.acos(object_up.clamp(-1.0, 1.0))
    # ── 감쌈 램프 — 분모는 판정 임계(0.6)가 아니라 saturation(0.85). 판정과 같으면
    #    3 지(부분점수 0.6)를 넘는 순간 4·5 지째 유인이 소멸한다(자매 실측: env 0.65
    #    에 2,500 에폭 고착 = 정확히 그 포화점). 판정 0.6 은 env 쪽에서 불변.
    env_gate = (env_frac / max(float(cfg.envelope_gate_saturation), 1e-6)).clamp(0.0, 1.0)
    # ★★08.25 리프트·이송 게이트에 감쌈 램프를 **곱한다**. 대향 게이트(gf)만으로는
    #   2~3 지 핀치에서도 0.83 으로 열려, 리프트 5 + 이송 8 이 파지 2 + 감쌈 3 을
    #   압도했다 — 실측 기여 6.60 대 0.745(8.9 배)로 "대충 잡고 들기"가 압도적
    #   이득이었고, 실제로 grip 이 0.350 → 0.167 로 반토막 나며 후퇴했다.
    #   ★★하한(floor)을 두지 않는다. 자매 트랙이 하한 0.3 으로 정확히 같은 실패를
    #   냈다 — 감쌈 0.21 에 고착한 채 이송만 학습(계약 test_envelope_has_no_gate_floor).
    #   하한은 "감쌈 없이도 흐르는 보상"이라 지금 고치려는 그 구멍을 다시 뚫는다.
    #   탐색이 죽지 않는 이유: 램프는 gf 와 **함께** 오른다. 대향 접촉이 성립하면
    #   이미 마디가 닿아 env_frac > 0 이라(실측 초기 구간 0.107) 신호가 남는다.
    gf_env = gf * env_gate
    # ④ 리프트(5) — |z 오차| 대칭 커널(과주행 자동 벌점) × 게이트 × 감쌈 램프.
    lift = torch.exp(-float(cfg.lift_sharpness)
                     * (goal_pos[:, 2] - object_pos[:, 2]).abs()) * gf_env
    # ⑤ 이송(8)
    tracking = (1.0 - torch.tanh(d_goal / float(cfg.tracking_std))) * gf_env
    # ⑥ 성공(12) — 목표 근접 × 직립(직립 독립항은 여기 흡수) × 게이트 × 감쌈 램프.
    #    ★성공만은 floor 없이 **순수 램프**다 — 성공 판정은 진짜 감쌈을 요구한다.
    success = ((1.0 - torch.tanh(d_goal / float(cfg.success_std)))
               * (1.0 - torch.tanh(theta / float(cfg.success_rot_std)))
               * gf * env_gate)

    # ── 정규화(단계 밖) — action_l2 는 팔만(손은 파지 자세가 곧 큰 액션),
    #    변화율은 전체(손끝 목표 채터링은 자세와 무관하게 벌한다).
    a_arm = actions[:, :n_arm_actions]
    terms = {
        "approach": float(cfg.approach_weight) * approach,
        "align": float(cfg.align_weight) * align,
        "grip": float(cfg.grip_weight) * grip,
        "envelope": float(cfg.envelope_weight) * env_frac,
        "lift": float(cfg.lift_weight) * lift,
        "tracking": float(cfg.tracking_weight) * tracking,
        "success": float(cfg.success_weight) * success,
        "action_l2": float(cfg.action_l2_weight) * action_l2_clamped(a_arm),
        "action_rate_l2": float(cfg.action_rate_l2_weight)
        * action_rate_l2_clamped(actions, prev_actions),
    }
    total = torch.nan_to_num(
        sum(v for k, v in terms.items() if not k.startswith("_")),
        nan=0.0, posinf=0.0, neginf=0.0)
    terms["_d_grasp"] = d_grasp
    # ★원값을 남긴다 — 보상은 가중치가 곱해져 있어 "정렬이 실제로 좋아졌는가"를
    #   못 읽는다(08.22 clamp 교훈: 포화한 항이 "작지만 0 아닌 값"으로 오독됨).
    terms["_align_cos"] = align_cos
    terms["_d_wrap"] = d_wrap.mean(dim=-1)
    # ★손가락별 원값 — 집계 평균만 보면 "안 닿는 손가락"의 원인을 못 가른다.
    #   d_wrap 이 크면 못 간 것(도달/액션 박스), 작은데 wrap 이 0 이면 손등이다.
    terms["_d_wrap_per"] = d_wrap                       # (N, F)
    terms["_grip_per"] = open_f * grip_kernel           # (N, F)
    terms["_env_gate"] = env_gate
    return total, terms, g, env_frac
