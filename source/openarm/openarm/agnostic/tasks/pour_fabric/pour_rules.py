"""pour_fabric 순수 규칙(torch 만, isaaclab 무관) — 09.16 비드 부피 DR · 조준 전 틸트 래치.

env 는 여기 함수를 부르기만 한다. isaaclab 이 없는 로컬에서 테스트되는 유일한 층(`tests/test_pour_rules.py`).

배경(사용자 결정 09.16 "권장 방식으로 진행"):
  - 컵 내부 177 mm × r 41 mm 에 12 mm 비드 20개는 3 % 채움이라 "접근 전 틸트 = 흘림" 이 물리로 안 생긴다.
    비드를 30 mm 로 키우고(스폰 개수 고정) 에피소드마다 활성 개수를 뽑아 부피를 DR 한다. 비활성 비드는
    env 마다 테이블 뒤 지면 위 격자에 눕힌다(한 점에 모으면 브로드페이즈 페어 폭발 — repfalse 이력).
  - 채움 정도 0~1 은 실측(정착 후 활성 비드 평균 높이)으로 정한다 — 개수·크기와 무관한 부피량이라
    실기에서는 사람이 어림잡아 넣는 명령 입력과 뜻이 같다.
  - 조준 전 틸트: 소스 입구가 리시버 입구에서 xy 로 멀리 있는데 소스가 한계 각도를 넘으면 성공 무효
    (i07 계측: 입구 거리 0.236 m 에서 30° 돌파, 실제 붓기 중 입구 거리 중앙값 0.021 m).
"""
from __future__ import annotations

import math

import torch

_LAYER_TWIST_RAD = 0.35      # 층마다 비틀어 수직 정렬을 깬다(bead_assets 검증 배치와 같은 값)
_WALL_MARGIN_M = 0.003       # 링 반지름 = 안쪽 반지름 − 비드 반지름 − 여유
_SPAWN_GAP_M = 0.002         # 이웃 비드 사이 여유(소환 겹침 → 벽 관통 사고 08.17 재발 방지)
_PARK_GAP_M = 0.010          # 파킹 격자 간격 여유


def premature_tilt_now(src_tilt: torch.Tensor, lip_xy: torch.Tensor, *,
                       tilt_max_deg: float, lip_xy_min_m: float) -> torch.Tensor:
    """이 스텝의 순간 판정 (N,) bool — 래치(에피소드 누적)는 env 가 한다."""
    return (src_tilt > math.radians(float(tilt_max_deg))) & (lip_xy > float(lip_xy_min_m))


def fill_level_from_local_z(z_local: torch.Tensor, active: torch.Tensor, *,
                            bottom_z: float, top_z: float) -> torch.Tensor:
    """활성 비드 평균 높이의 2배를 컵 내부 높이로 나눈 [0,1] (N,).

    균일 기둥이면 평균 높이 = 바닥 + h/2 라 정확하고, 한 알이 튀어 올라도 최고점 방식처럼 흔들리지 않는다.
    활성이 0개면 0.
    """
    a = active.to(z_local.dtype)
    n = a.sum(dim=1)
    mean_z = (z_local * a).sum(dim=1) / n.clamp(min=1.0)
    h = 2.0 * (mean_z - float(bottom_z))
    level = (h / (float(top_z) - float(bottom_z))).clamp(0.0, 1.0)
    return torch.where(n > 0, level, torch.zeros_like(level))


def sample_active_mask(n_envs: int, *, k: int, lo: int, hi: int, device) -> torch.Tensor:
    """에피소드별 활성 비드 수 n ~ U{lo..hi} 를 뽑아 앞쪽 n 개만 True 인 (N,k) bool.

    배치가 아래층부터라 앞쪽 = 낮은 층 → 활성 비드가 컵 바닥에 모인다.
    """
    lo, hi, k = int(lo), int(hi), int(k)
    if not 1 <= lo <= hi <= k:
        raise ValueError(f"활성 범위는 1 ≤ lo ≤ hi ≤ k 여야 한다: lo={lo} hi={hi} k={k}")
    n_active = torch.randint(lo, hi + 1, (int(n_envs),), device=device)
    return torch.arange(k, device=device).unsqueeze(0) < n_active.unsqueeze(1)


def bead_layout_in_cup(n: int, *, diameter: float, inner_radius: float, bottom_z: float) -> list[list[float]]:
    """컵 원점 기준 소환 offset (n,3) — 지름에서 층당 개수·링 반지름·층 간격을 유도한다.

    bead_assets.bead_offsets_in_cup(12 mm 20개 검증 배치)의 일반화. 이웃 중심 거리 ≥ 지름 + 여유를
    보장하고 바닥부터 쌓는다(활성 prefix 규약). 소환 기둥이 림 위로 솟는 것은 허용 — 정착 대기 동안 떨어진다.
    """
    d = float(diameter)
    r_ring = float(inner_radius) - d / 2.0 - _WALL_MARGIN_M
    ratio = (d + _SPAWN_GAP_M) / (2.0 * r_ring) if r_ring > 0.0 else 1.0
    per_layer = 1 if ratio >= 1.0 else max(1, int(math.floor(math.pi / math.asin(ratio))))
    radius = r_ring if per_layer > 1 else 0.0
    dz = d + _SPAWN_GAP_M
    z0 = float(bottom_z) + d / 2.0 + _SPAWN_GAP_M
    out: list[list[float]] = []
    for i in range(int(n)):
        layer, slot = divmod(i, per_layer)
        angle = 2.0 * math.pi * slot / per_layer + _LAYER_TWIST_RAD * layer
        out.append([radius * math.cos(angle), radius * math.sin(angle), z0 + dz * layer])
    return out


def park_offsets(k: int, *, diameter: float, origin_xy: tuple[float, float], z: float,
                 per_row: int) -> list[list[float]]:
    """비활성 비드 파킹 격자 (k,3), env-local. 행은 −x 로 물러나며 늘어난다(로봇·테이블은 +x 쪽)."""
    step = float(diameter) + _PARK_GAP_M
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    out: list[list[float]] = []
    for i in range(int(k)):
        row, col = divmod(i, int(per_row))
        out.append([ox - step * row, oy + step * col, float(z)])
    return out
