"""2단계 놓기 성공 판정 (스펙 2026-09-16-iker-stage2-t2r §3).

네 조건(`placed` · `released` · `resting` · `still`)이 **최근 window_steps 스텝 중 stable_steps 스텝 이상**
만족하면 성공이다.

판정 변경 이력:
  1. 기존 IKER 2단계의 success_count 는 누적이라 흩어진 스텝으로도 성공이 됐다.
  2. 그래서 **연속** 카운터로 바꿨다(조건이 깨지면 0 으로 리셋).
  3. 2026-09-17, 연속 방식이 노이즈에 부서지는 것을 실측했다 — 같은 ep250 정책을 노이즈 off/on 으로 재면
     조건 충족률(`ok` 점유율·도달 env)은 거의 같은데 20스텝 완주율만 0.391 → 0.0078 로 50배 무너졌다.
     한 스텝만 흔들려도 카운터가 0 이 되기 때문이다. 학습 지표가 실제 실력(39 %)이 아니라 취약성(0.8 %)을
     찍고 있었다. 사용자 결정으로 **창 방식**으로 바꾼다: 흐트러짐은 걸러내되 노이즈에는 견딘다.
     같은 결정에서 `place_tolerance` 도 0.05 → 0.03 으로 조인다 — 설계상 두 신발 사이 틈이 2.8 cm 인데
     허용오차가 5 cm 여서, 판정상 성공해도 눈으로는 옆 신발에 붙지 않은 것으로 보였다(실측 틈 중앙값 6.1 cm).

`stable_count` 는 이름을 유지하되 의미가 바뀐다 — 연속 횟수가 아니라 **창 안의 충족 스텝 수**다.
소비처(t2r2 컨텍스트·프롬프트·프로브)가 그대로 쓰도록 이름과 모양을 바꾸지 않았다.

``shoe_ang_speed`` 를 넘기면 선속도와 같은 ``cfg.still_speed`` 문턱으로 각속도도 함께 본다(제자리에서 도는
신발을 "정지"로 세지 않기 위해). 생략하면 선속도만 본다.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import layout


@dataclass
class PlaceRewardCfg:
    place_tolerance: float = 0.03      # 키포인트 4개 평균 거리 [m] (2026-09-17: 0.05 → 0.03)
    release_radius: float = 0.15       # palm-shoe 거리가 이 값을 넘어야 "손을 뗐다" [m]
    rack_top_z: float = layout.RACK_TOP_Z
    resting_tol: float = 0.01          # hull 최저점이 받침 상면에서 벗어날 수 있는 폭 [m]
    still_speed: float = 0.05          # 신발 속도 [m/s]
    stable_steps: int = 20             # 창 안에서 만족해야 하는 스텝 수
    window_steps: int = 30             # 창의 길이 — 최근 이만큼의 스텝을 본다


@dataclass(frozen=True)
class PlaceStep:
    placed: torch.Tensor
    released: torch.Tensor
    resting: torch.Tensor
    still: torch.Tensor
    ok: torch.Tensor
    window: torch.Tensor       # (N, window_steps) 이번 스텝까지의 창 내용
    stable_count: torch.Tensor  # (N,) 창 안에서 네 조건이 모두 성립한 스텝 수
    success: torch.Tensor


def new_window(num_envs: int, cfg: PlaceRewardCfg, device: torch.device | str) -> torch.Tensor:
    """리셋 직후의 빈 창. 환경이 버퍼를 만들 때와 `_reset_idx` 에서 지울 때 모두 이걸 쓴다."""
    return torch.zeros(num_envs, int(cfg.window_steps), device=device)


def place_step(
    keypoint_dist: torch.Tensor,
    palm_shoe_dist: torch.Tensor,
    shoe_bottom_z: torch.Tensor,
    shoe_speed: torch.Tensor,
    window: torch.Tensor,
    cfg: PlaceRewardCfg,
    shoe_ang_speed: torch.Tensor | None = None,
) -> PlaceStep:
    placed = keypoint_dist <= cfg.place_tolerance
    released = palm_shoe_dist > cfg.release_radius
    resting = (shoe_bottom_z - cfg.rack_top_z).abs() <= cfg.resting_tol
    still = shoe_speed < cfg.still_speed
    if shoe_ang_speed is not None:
        still = still & (shoe_ang_speed < cfg.still_speed)
    ok = placed & released & resting & still
    # 창을 한 칸 밀고 이번 스텝 결과를 끝에 넣는다(가장 오래된 스텝이 빠진다).
    rolled = torch.cat([window[:, 1:], ok[:, None].float()], dim=1)
    count = rolled.sum(dim=1)
    return PlaceStep(placed, released, resting, still, ok, rolled, count, count >= float(cfg.stable_steps))
