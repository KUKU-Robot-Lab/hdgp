"""2단계 놓기 성공 판정 + 스크립트 home 복귀 (스펙 2026-09-16-iker-stage2-t2r §3).

다섯 조건(`placed` · `released` · `resting` · `still` · `home`)이 **최근 window_steps 스텝 중 stable_steps 스텝
이상** 만족하면 성공이다.

판정 변경 이력:
  1. 기존 IKER 2단계의 success_count 는 누적이라 흩어진 스텝으로도 성공이 됐다.
  2. 그래서 **연속** 카운터로 바꿨다(조건이 깨지면 0 으로 리셋).
  3. 2026-09-17, 연속 방식이 노이즈에 부서지는 것을 실측했다(20스텝 완주율 노이즈 off 0.391 → on 0.0078).
     **창 방식**으로 바꾸고 `place_tolerance` 를 0.05 → 0.03 으로 조였다(설계 틈 2.8 cm < 허용오차 5 cm).
  4. 2026-09-17, 놓은 뒤 팔을 들어 올리는 "만세"(사용자 관찰). 손바닥 위치 기반 `home` 을 넣었으나,
     학습된 정책은 손바닥만 그 점에 두고 팔 모양이 home 과 관절 최대 약 62°, 손바닥 방향 126° 달랐다
     (iter_04 ep500 측정). 액션이 손바닥 6자유도라 7축 팔의 여유 자유도를 정책이 제어할 수 없다.
  5. 2026-09-17, 사용자 결정 — **복귀는 학습하지 않고 정해진 궤적으로 한다.** 정책은 신발을 놓고 손을 펴는 데까지만
     책임진다. `placed & resting & still` 이고 그립 개방도가 `retract_open_min` 이상인 첫 스텝에 환경이 팔을 넘겨받아,
     손을 완전히 편 채 팔 관절을 home 관절값까지 `retract_steps` 스텝 동안 smoothstep 으로 보간한다. 이후 정책의
     액션은 무시된다. `home` 은 이제 **팔 관절이 home 관절값에서 `home_joint_tol` 이내**다.
     게이트 프로브(iter_04 ep500, 64 env, 20스텝): 트리거 65.6 %, 트리거된 env 100 % 성공, 신발 이동 최대 0.9 mm,
     손바닥-신발 최소 8 cm, 종료 관절 오차 0.13°.
     최종 목표는 기술들(집기·놓기·역방향)을 home 에서 시작해 home 에서 끝나게 이어 반복하는 것이다.

`stable_count` 는 이름을 유지하되 **창 안의 충족 스텝 수**다(연속 횟수가 아니다).
``arm_home_err`` 는 키워드 전용 필수 인자다 — 빠뜨리면 조건이 조용히 참이 되는 기본값을 두지 않는다.
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
    home_joint_tol: float = 0.035      # 팔 관절 최대 오차가 이 안이면 home [rad] (약 2°; 스크립트 복귀 종료 오차 0.13°)
    retract_steps: int = 20            # 스크립트 복귀에 쓰는 스텝 수 (2 s)
    retract_open_min: float = 0.9      # 복귀를 넘겨받는 그립 개방도 문턱 (0 = 쥔 자세, 1 = 편 자세)


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
    arm_home_err: torch.Tensor,
) -> PlaceStep:
    placed = keypoint_dist <= cfg.place_tolerance
    released = palm_shoe_dist > cfg.release_radius
    resting = (shoe_bottom_z - cfg.rack_top_z).abs() <= cfg.resting_tol
    still = shoe_speed < cfg.still_speed
    if shoe_ang_speed is not None:
        still = still & (shoe_ang_speed < cfg.still_speed)
    home = arm_home_err <= cfg.home_joint_tol
    ok = placed & released & resting & still & home
    # 창을 한 칸 밀고 이번 스텝 결과를 끝에 넣는다(가장 오래된 스텝이 빠진다).
    rolled = torch.cat([window[:, 1:], ok[:, None].float()], dim=1)
    count = rolled.sum(dim=1)
    return PlaceStep(
        placed=placed, released=released, resting=resting, still=still, home=home, ok=ok,
        window=rolled, stable_count=count, success=count >= float(cfg.stable_steps),
    )


def retract_trigger(
    placed: torch.Tensor, resting: torch.Tensor, still: torch.Tensor, open_frac: torch.Tensor,
    retracting: torch.Tensor, cfg: PlaceRewardCfg,
) -> torch.Tensor:
    """(N,) bool — 이번 스텝에 새로 복귀를 넘겨받을 env. 이미 복귀 중인 env 는 다시 트리거하지 않는다."""
    return ~retracting & placed & resting & still & (open_frac >= cfg.retract_open_min)


def retract_targets(q_start: torch.Tensor, q_home: torch.Tensor, k: torch.Tensor, cfg: PlaceRewardCfg) -> torch.Tensor:
    """(M, J) 팔 관절 목표 — 넘겨받은 뒤 k 번째 스텝(0 부터)의 smoothstep 보간. k >= retract_steps 면 home 에 머문다."""
    s = ((k.float() + 1.0) / float(cfg.retract_steps)).clamp(0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)
    return q_start + (q_home - q_start) * s[:, None]
