"""tip_cyl 전용 보상 — 08.24 2차 총 재설계(형상 비의존 6단계). 로봇 무관 텐서 함수.

★별도 파일인 이유(공유 세션 분리): `rewards.py` 는 grasp_lift_fabric(타 세션 트랙)이
공유하고, 그쪽 계약 테스트가 파일 전체의 `cfg.X` 키를 자기 cfg 와 대조한다.
tip_cyl 상수(stage_*)는 이 트랙 전용이므로 파일을 가르는 것이 경계다.
공용 소부품(contact_gate, action_l2 계열)만 rewards 에서 가져온다.
"""

from __future__ import annotations

import torch

from .rewards import (
    action_l2_clamped,
    action_rate_l2_clamped,
    contact_gate,
)


def compute_tip_cyl_rewards(
    grasp_center_pos: torch.Tensor,   # (N, 3) env-local · palm 부착 파지중심(손가락 무관)
    wrap_dist: torch.Tensor,          # (N, F) 손가락별 min(‖mid−obj‖, ‖dist−obj‖)
    palmar_wrap: torch.Tensor,        # (N, F) bool · 손바닥면 감쌈 c_i
    any_touch: torch.Tensor,          # (N, F) bool · 방향무관 접촉 a_i
    object_pos: torch.Tensor,         # (N, 3) env-local
    goal_pos: torch.Tensor,           # (N, 3) env-local
    object_up: torch.Tensor,          # (N,) 물체 local +z · world +z = cos(기울기)
    group_a_force: torch.Tensor,      # (N, Fa) 손가락별(합산) 접촉력
    group_b_force: torch.Tensor,      # (N, Fb)
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    cfg: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """6단계 · 단계마다 상한이 커진다 — 접근→파지→감쌈→리프트→이송→성공.

    ★1차안(max 커널)이 실패한 이유(lstm_test5 실측, gate 2,820ep 내내 0.000):
      `approach = exp(−k·max_i‖pᵢ−obj‖)` 의 점 목록에 **palm 을 넣은 것**이 원인이었다.
      학습된 정책에서 argmax 지목률 palm 52.7% · index 32.2% · mid 10.3% · thumb 4.8%
      · **ring 0.0% · pinky 0.0%**. palm 은 완전 파지에서도 컵에서 130.3mm(구조적)라
      **가까워질수록 argmax 점유가 52.7%→100% 로 커진다** = 접근을 잘할수록 손가락
      gradient 가 사라지는 자기소멸 동역학. "가장 먼 손끝이 지배 → 5지 참여 강제"라는
      DEXTRAH 의도는 palm 을 점 목록에 넣는 순간 깨진다(손끝 도달 53~85mm vs palm 130mm).

    설계 원칙 4개:
      ①**액션 그룹당 항 하나** — ①은 palm 6D(팔), ②는 (r,z)×5(손가락). 한 항이 두
        그룹을 겸하면 위 자기소멸이 재발한다.
      ②**max 아니라 mean** — 손가락별로 액션이 독립이므로 mean 이라야 각 손가락이
        자기 편미분을 받는다. max 는 5개 액션 차원을 스칼라로 접어 4개를 굶긴다.
      ③**형상 비의존** — 보상이 쓰는 양은 obs 로 계산 가능한 것뿐(물체 pose, 링크
        위치, 접촉력). 물체 CAD·반지름·높이를 쓰지 않는다. 조임 깊이도 지정하지
        않는다(②가 접촉 시 꺼지므로) — 무게·마찰에 맞는 조임은 ④~⑥ 상실 위험이 정한다.
      ④**단계 상한 증가**(1<2<3<5<8<12) — 다음 단계가 항상 더 크므로 주차하지 않는다.
        1차안은 무게이트가 approach 2.0 하나뿐이고 나머지 15가 전부 게이트 뒤라
        1→2 사이가 절벽이었다. ①②③을 무게이트로 두는 것이 부트스트랩의 핵심.

    ★②가 **팁이 아니라 mid/dist 마디**를 당기는 이유(reward-audit Check2 지적):
      감쌈 판정 c_i 는 중간(_3)·원위(_4) 마디 접촉이고 팁 접촉은 명시적으로 제외된다
      ("핀치(팁만)로는 이 항이 0"). 팁을 당기면 ②의 최적이 **핀치**가 되어 ③=0 인
      자세를 가르치고, 핀치는 c_i=0 이라 `(1−a_i)` 가 안 꺼져 **팁을 계속 압입**한다
      (lstm_test5 의 force_max 28~46N 과 같은 동역학). ③이 판정하는 그 마디를 당겨야
      ②→③ 이 연속이다.

    ★`a_i`(방향 무관)로 ②를 끄고 `c_i`(손바닥면)로 ③을 주는 비대칭이 의도적이다 —
      손등으로 밀어도 당김은 멈추지만(압입 방지) 이득은 0 이라, 자세를 고쳐야만 점수가 오른다.

    returns (total, terms, gate, envelope_frac) — 구 함수와 동일 계약(env 배선 호환).
    """
    thr = float(cfg.contact_force_threshold)
    g = contact_gate(group_a_force, group_b_force, thr)
    gf = g.float()

    # ① 접근(팔) — 파지중심을 물체로. 무게이트. palm 원점이 아니라 **파지중심**이라
    #    d=0 이 "컵이 손 한가운데"이고, 손바닥으로 관통하라는 지시가 되지 않는다.
    d_gc = torch.norm(grasp_center_pos - object_pos, dim=-1)
    approach = torch.exp(-float(cfg.stage_approach_sharpness) * d_gc)

    # ② 파지 접근(손가락) — 아직 안 닿은 손가락만 그 마디를 물체로 당긴다. 무게이트.
    #    바닥값은 **손 기하 상수**(팁 반경)라 물체 형상을 끌어들이지 않는다.
    open_i = (~any_touch).float()
    grip = (open_i * torch.exp(
        -float(cfg.stage_grip_sharpness)
        * wrap_dist.clamp(min=float(cfg.stage_grip_dist_floor)))).mean(dim=-1)

    # ③ 감쌈 — 손바닥면 접촉 비율(소지 포함 5지). 무게이트.
    env_frac = palmar_wrap.float().mean(dim=-1)

    # ④⑤⑥ — 대향 접촉 게이트(곱셈 게이트는 이 하나뿐)
    d_goal = torch.norm(object_pos - goal_pos, dim=-1)
    theta = torch.acos(object_up.clamp(-1.0, 1.0))
    lift = torch.exp(-float(cfg.stage_lift_sharpness)
                     * (goal_pos[:, 2] - object_pos[:, 2]).abs()) * gf
    tracking = (1.0 - torch.tanh(d_goal / float(cfg.stage_tracking_std))) * gf
    upright = (1.0 - torch.tanh(theta / float(cfg.stage_upright_std))) * gf
    success = ((1.0 - torch.tanh(d_goal / float(cfg.stage_success_pos_std)))
               * (1.0 - torch.tanh(theta / float(cfg.stage_success_rot_std)))) * gf

    terms = {
        "approach": float(cfg.stage_approach_weight) * approach,
        "grip": float(cfg.stage_grip_weight) * grip,
        "envelope": float(cfg.stage_envelope_weight) * env_frac,
        "lift": float(cfg.stage_lift_weight) * lift,
        "tracking": float(cfg.stage_tracking_weight) * tracking,
        "upright": float(cfg.stage_upright_weight) * upright,
        "success": float(cfg.stage_success_weight) * success,
        "action_l2": float(cfg.action_l2_weight) * action_l2_clamped(actions),
        "action_rate_l2": float(cfg.action_rate_l2_weight)
        * action_rate_l2_clamped(actions, prev_actions),
    }
    total = torch.nan_to_num(sum(terms.values()), nan=0.0, posinf=0.0, neginf=0.0)

    # 진단용(보상 아님) — 로깅에서 pop 한다.
    terms["_d_gc"] = d_gc
    terms["_grip_dist"] = wrap_dist.mean(dim=-1)
    terms["_touch_frac"] = any_touch.float().mean(dim=-1)
    return total, terms, g, env_frac
