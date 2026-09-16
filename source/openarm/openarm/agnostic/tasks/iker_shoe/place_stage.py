"""2단계 놓기 성공 판정 (스펙 2026-09-16-iker-stage2-t2r §3).

네 조건을 연속 stable_steps 스텝 만족하면 성공이다. 조건이 깨지면 카운터는 0 으로 돌아간다 —
기존 IKER 2단계의 success_count 는 누적이라 흩어진 스텝으로도 성공이 됐다.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import layout


@dataclass
class PlaceRewardCfg:
    place_tolerance: float = 0.05      # 키포인트 4개 평균 거리 [m]
    release_radius: float = 0.15       # palm-shoe 거리가 이 값을 넘어야 "손을 뗐다" [m]
    rack_top_z: float = layout.RACK_TOP_Z
    resting_tol: float = 0.01          # hull 최저점이 받침 상면에서 벗어날 수 있는 폭 [m]
    still_speed: float = 0.05          # 신발 속도 [m/s]
    stable_steps: int = 20             # 연속 만족 스텝 수


@dataclass(frozen=True)
class PlaceStep:
    placed: torch.Tensor
    released: torch.Tensor
    resting: torch.Tensor
    still: torch.Tensor
    ok: torch.Tensor
    stable_count: torch.Tensor
    success: torch.Tensor


def place_step(
    keypoint_dist: torch.Tensor,
    palm_shoe_dist: torch.Tensor,
    shoe_bottom_z: torch.Tensor,
    shoe_speed: torch.Tensor,
    stable_count: torch.Tensor,
    cfg: PlaceRewardCfg,
) -> PlaceStep:
    placed = keypoint_dist <= cfg.place_tolerance
    released = palm_shoe_dist > cfg.release_radius
    resting = (shoe_bottom_z - cfg.rack_top_z).abs() <= cfg.resting_tol
    still = shoe_speed < cfg.still_speed
    ok = placed & released & resting & still
    count = torch.where(ok, stable_count + 1.0, torch.zeros_like(stable_count))
    return PlaceStep(placed, released, resting, still, ok, count, count >= float(cfg.stable_steps))
