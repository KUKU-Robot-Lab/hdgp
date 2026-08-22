"""비드 분류 — both/pour_v1 `_compute_bead_flags`(pour_right_env.py:1322) 의 순수 함수 이식.

env 메서드 → 순수 함수: 상태(prev z·crossed 마스크)는 인자로 받고 결과로 돌려준다.
판정 기하(내부 반경·z 대역·mouth z)는 cfg 가 준다 — 컵 자산 실측값(.usd 기준):
    bottom=-0.077, rim=+0.100, inner_r=0.041.

분류 규약(원본 그대로):
  in_target / in_source : 컵 로컬 프레임에서 xy 반경 + z 대역 안
  crossed(mouth)        : 직전 스텝 로컬 z 가 mouth 위였다가 아래로 내려온 이력(латch 아님 —
                          이건 물리 사건 기록이라 보상 래치 금지 계약과 무관하다)
  spilled               : source 밖 + target 로컬 z < z_min = 영구 손실
                          (공중 이동(transit) 비드는 z > z_min 이라 spill 이 아니다)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.utils.math import quat_apply_inverse


@dataclass(frozen=True)
class BeadGeometry:
    """컵 판정 기하(로컬 프레임, scale 적용 후 [m])."""
    inner_radius: float
    inside_z_min: float
    inside_z_max: float
    mouth_z: float


@dataclass
class BeadFlagResult:
    in_source_frac: torch.Tensor      # (N,)
    in_target_frac: torch.Tensor      # (N,)
    crossed_frac: torch.Tensor        # (N,)  mouth 통과 이력 비율
    spill_frac: torch.Tensor          # (N,)
    centroid_w: torch.Tensor          # (N,3) 비드 무게중심(world)
    # 다음 스텝에 되넘길 상태
    target_local_z: torch.Tensor      # (N,k)
    crossed_mask: torch.Tensor        # (N,k) bool


def _local(bead_pos_w: torch.Tensor, cup_pos_w: torch.Tensor,
           cup_quat_w: torch.Tensor) -> torch.Tensor:
    """비드를 컵 로컬 프레임으로. (N,k,3)"""
    n, k = bead_pos_w.shape[:2]
    quat = cup_quat_w.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
    rel = (bead_pos_w - cup_pos_w.unsqueeze(1)).reshape(-1, 3)
    return quat_apply_inverse(quat, rel).reshape(n, k, 3)


def compute_bead_flags(
    *,
    bead_pos_w: torch.Tensor,          # (N,k,3)
    source_pos_w: torch.Tensor,        # (N,3)
    source_quat_w: torch.Tensor,       # (N,4) wxyz
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    geom_source: BeadGeometry,
    geom_target: BeadGeometry,
    prev_target_local_z: torch.Tensor,  # (N,k)
    crossed_mask: torch.Tensor,         # (N,k) bool — 리셋 시 False 로 초기화할 것
) -> BeadFlagResult:
    p_tgt = _local(bead_pos_w, target_pos_w, target_quat_w)
    xy_tgt = p_tgt[..., :2].norm(dim=-1)
    in_target = (
        (xy_tgt <= geom_target.inner_radius)
        & (p_tgt[..., 2] >= geom_target.inside_z_min)
        & (p_tgt[..., 2] <= geom_target.inside_z_max)
    )

    p_src = _local(bead_pos_w, source_pos_w, source_quat_w)
    xy_src = p_src[..., :2].norm(dim=-1)
    in_source = (
        (xy_src <= geom_source.inner_radius)
        & (p_src[..., 2] >= geom_source.inside_z_min)
        & (p_src[..., 2] <= geom_source.inside_z_max)
    )

    crossed_now = (
        (xy_tgt <= geom_target.inner_radius)
        & (prev_target_local_z > geom_target.mouth_z)
        & (p_tgt[..., 2] <= geom_target.mouth_z)
    )
    crossed = crossed_mask | crossed_now

    # source 밖 + target 로컬 z<z_min = 영구 손실. transit(공중) 비드는 제외된다.
    spilled = (~in_source) & (p_tgt[..., 2] < geom_target.inside_z_min)

    return BeadFlagResult(
        in_source_frac=in_source.float().mean(dim=-1),
        in_target_frac=in_target.float().mean(dim=-1),
        crossed_frac=crossed.float().mean(dim=-1),
        spill_frac=spilled.float().mean(dim=-1),
        centroid_w=bead_pos_w.mean(dim=1),
        target_local_z=p_tgt[..., 2],
        crossed_mask=crossed,
    )
