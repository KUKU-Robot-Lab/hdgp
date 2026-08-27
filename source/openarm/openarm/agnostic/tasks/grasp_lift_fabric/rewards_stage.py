"""계층 게이트 λ→μ→ν→ρ — **진단 전용**(08.27 부터 보상이 아니다).

보상은 `tesollo/right/grasp_v1/grasp_reward.py` 이식본이 낸다(사용자 사양 08.27).
이 게이트는 "사슬이 어디서 끊기는가"를 읽는 판정 지표로만 남는다.

★왜 보상에서 뺐나: μ 가 **접촉 손가락 수**만 세고 감쌈을 보지 않아서, 세 손가락을
  닿게만 해도 열렸다. h7 실측이 그 결과다 — 좌팔 μ hit 0.153 인데 엄격 감쌈은
  index 0.050·middle 0.073·ring 0.072, `envelope_strict` 0.100. 정책은 엄지와
  손바닥으로 받쳐 들었고 컵이 13.4° 기울었다. 게이트 hit 을 성공으로 읽은 것이
  오독이었다. grasp_v1 보상은 grasp 항의 55% 를 **엄격 감쌈**에 걸어 이 경로를 막는다.
"""

from __future__ import annotations

import torch


def compute_stage_gates_only(
    *,
    grasp_center_pos: torch.Tensor,
    object_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    touch_c: torch.Tensor,
    wrap_c: torch.Tensor,
    height_delta: torch.Tensor,
    corridor_ok: torch.Tensor,
    cfg: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(λμνρ 순간 게이트 float (N,4), 느슨한 감쌈비율 (N,)).

    이진 **누적 곱** — 한 칸이 막히면 아래가 전부 0 이다(자매 규약 유지).
    """
    d_gc = (grasp_center_pos - object_pos).norm(dim=-1)
    lam = (d_gc < float(cfg.stage_gate_approach_m)).float()
    mu = lam * (touch_c.float().sum(dim=-1)
                >= float(cfg.stage_gate_contact_n)).float()
    nu = mu * (height_delta >= float(cfg.stage_gate_lift_m)).float() * corridor_ok
    rho = nu * ((object_pos - goal_pos).norm(dim=-1)
                < float(cfg.stage_gate_transfer_m)).float()
    envelope_frac = wrap_c.float().mean(dim=-1)
    return torch.stack([lam, mu, nu, rho], dim=-1), envelope_frac
