"""2단계 놓기 성공 판정 (스펙 2026-09-16-iker-stage2-t2r §3).

네 조건을 연속 stable_steps 스텝 만족하면 성공이다. 조건이 깨지면 카운터는 0 으로 돌아간다 —
기존 IKER 2단계의 success_count 는 누적이라 흩어진 스텝으로도 성공이 됐다.

fix round 3 (final review, small item): ``still`` 은 기본적으로 선속도만 본다(``shoe_speed``) — 제자리에서
도는 신발도 "정지"로 셌다. ``shoe_ang_speed`` 를 넘기면 그것도 같은 ``cfg.still_speed`` 문턱으로 함께
본다(별도 각속도 문턱을 새로 만들지 않는다 — 0.05 rad/s 는 0.05 m/s 와 같은 정신으로 "아주 느림"). 생략하면
(기존 ``tests/test_place_stage.py`` 의 6-위치인자 호출들처럼) 이전과 동일하게 선속도만 본다 — 그 테스트
파일은 이 라운드에서도 건드리지 않는다.
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
    shoe_ang_speed: torch.Tensor | None = None,
) -> PlaceStep:
    placed = keypoint_dist <= cfg.place_tolerance
    released = palm_shoe_dist > cfg.release_radius
    resting = (shoe_bottom_z - cfg.rack_top_z).abs() <= cfg.resting_tol
    still = shoe_speed < cfg.still_speed
    if shoe_ang_speed is not None:
        still = still & (shoe_ang_speed < cfg.still_speed)
    ok = placed & released & resting & still
    count = torch.where(ok, stable_count + 1.0, torch.zeros_like(stable_count))
    return PlaceStep(placed, released, resting, still, ok, count, count >= float(cfg.stable_steps))
