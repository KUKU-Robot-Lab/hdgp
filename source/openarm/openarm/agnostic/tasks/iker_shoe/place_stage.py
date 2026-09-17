"""2단계 놓기 성공 판정 (스펙 2026-09-16-iker-stage2-t2r §3).

다섯 조건(`placed` · `released` · `resting` · `still` · `home`)이 **최근 window_steps 스텝 중 stable_steps 스텝
이상** 만족하면 성공이다.

판정 변경 이력:
  1. 기존 IKER 2단계의 success_count 는 누적이라 흩어진 스텝으로도 성공이 됐다.
  2. 그래서 **연속** 카운터로 바꿨다(조건이 깨지면 0 으로 리셋).
  3. 2026-09-17, 연속 방식이 노이즈에 부서지는 것을 실측했다 — 같은 ep250 정책을 노이즈 off/on 으로 재면
     조건 충족률(`ok` 점유율·도달 env)은 거의 같은데 20스텝 완주율만 0.391 → 0.0078 로 50배 무너졌다.
     사용자 결정으로 **창 방식**으로 바꾸고, `place_tolerance` 를 0.05 → 0.03 으로 조였다(설계상 두 신발 틈
     2.8 cm 인데 허용오차 5 cm 라 판정상 성공도 붙지 않아 보였다, 실측 틈 중앙값 6.1 cm).
  4. 2026-09-17, iter_02 ep350 영상에서 신발을 놓은 뒤 팔을 들어 올려 "만세" 자세가 됐다(사용자 관찰).
     원인: 놓은 뒤 손을 어디 둘지 알려주는 신호가 없었다 — 보상 컨텍스트에 복귀 위치 필드가 없었고,
     `released` 는 방향 무관 거리라 위로 드는 것이 가장 빠른 길이었다(`place/retreated` 0.79 → 0.05).
     사용자 결정으로 **`home`(로봇 기본 자세로 복귀)을 다섯 번째 조건**으로 넣는다.
     관절값이 아니라 **홈 자세일 때의 손바닥 위치**로 잰다: 게이트 프로브에서 손바닥 IK 명령으로 홈 손바닥
     위치에는 9스텝 만에 5 cm 안에 들어가지만(최종 오차 중앙값 1.0 cm, p90 3.2 cm), 그때도 관절 최대 오차가
     중앙값 1.01 rad · p90 1.52 rad 였다. 액션이 손바닥 6자유도라 7축 팔의 여유 자유도를 정책이 제어할 수
     없으므로, 관절값 조건은 학습으로 도달할 수 없다.

`stable_count` 는 이름을 유지하되 **창 안의 충족 스텝 수**다(연속 횟수가 아니다).

``shoe_ang_speed`` 를 넘기면 선속도와 같은 ``cfg.still_speed`` 문턱으로 각속도도 함께 본다.
``palm_home_dist`` 는 키워드 전용 필수 인자다 — 빠뜨리면 조건이 조용히 참이 되는 기본값을 두지 않는다.
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
    home_radius: float = 0.05          # 손바닥이 홈 자세 손바닥 위치에서 이 안이면 "복귀" [m] (2026-09-17)


@dataclass(frozen=True)
class PlaceStep:
    placed: torch.Tensor
    released: torch.Tensor
    resting: torch.Tensor
    still: torch.Tensor
    home: torch.Tensor
    ok: torch.Tensor
    window: torch.Tensor       # (N, window_steps) 이번 스텝까지의 창 내용
    stable_count: torch.Tensor  # (N,) 창 안에서 다섯 조건이 모두 성립한 스텝 수
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
    *,
    palm_home_dist: torch.Tensor,
) -> PlaceStep:
    placed = keypoint_dist <= cfg.place_tolerance
    released = palm_shoe_dist > cfg.release_radius
    resting = (shoe_bottom_z - cfg.rack_top_z).abs() <= cfg.resting_tol
    still = shoe_speed < cfg.still_speed
    if shoe_ang_speed is not None:
        still = still & (shoe_ang_speed < cfg.still_speed)
    home = palm_home_dist <= cfg.home_radius
    ok = placed & released & resting & still & home
    # 창을 한 칸 밀고 이번 스텝 결과를 끝에 넣는다(가장 오래된 스텝이 빠진다).
    rolled = torch.cat([window[:, 1:], ok[:, None].float()], dim=1)
    count = rolled.sum(dim=1)
    return PlaceStep(
        placed=placed, released=released, resting=resting, still=still, home=home, ok=ok,
        window=rolled, stable_count=count, success=count >= float(cfg.stable_steps),
    )
