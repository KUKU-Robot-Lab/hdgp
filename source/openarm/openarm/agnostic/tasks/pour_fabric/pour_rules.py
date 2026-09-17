"""pour_fabric 순수 규칙(torch 만, isaaclab 무관) — 09.16 비드 부피 DR · 조준 전 틸트 래치.

env 는 여기 함수를 부르기만 한다. isaaclab 이 없는 로컬에서 테스트되는 유일한 층(`tests/test_pour_rules.py`).

배경(사용자 결정 09.16 "권장 방식으로 진행"):
  - 컵 내부 177 mm × r 41 mm 에 12 mm 비드 20개는 3 % 채움이라 "접근 전 틸트 = 흘림" 이 물리로 안 생긴다.
    비드를 30 mm 로 키우고(스폰 개수 고정) 에피소드마다 활성 개수를 뽑아 부피를 DR 한다. 비활성 비드는
    env 마다 테이블 뒤 지면 위 격자에 눕힌다(한 점에 모으면 브로드페이즈 페어 폭발 — repfalse 이력).
  - 채움 정도 0~1 은 실측(정착 후 활성 비드 평균 높이)으로 정한다 — 개수·크기와 무관한 부피량이라
    실기에서는 사람이 어림잡아 넣는 명령 입력과 뜻이 같다.
  - 조준 전 틸트: 붓는 쪽 림 점이 리시버 입구에서 xy 로 멀리 있는데 소스가 한계 각도를 넘으면 성공 무효
    (i07 계측: 입구 거리 0.236 m 에서 30° 돌파, 실제 붓기 중 입구 거리 중앙값 0.021 m).
    ★09.18 한계 각도는 고정 30° 가 아니라 채움별 유출각 − 여유, 거리는 입구 중심이 아니라 붓는 쪽 림 점 기준.
"""
from __future__ import annotations

import math

import torch

_LAYER_TWIST_RAD = 0.35      # 층마다 비틀어 수직 정렬을 깬다(bead_assets 검증 배치와 같은 값)
_WALL_MARGIN_M = 0.003       # 링 반지름 = 안쪽 반지름 − 비드 반지름 − 여유
_SPAWN_GAP_M = 0.002         # 이웃 비드 사이 여유(소환 겹침 → 벽 관통 사고 08.17 재발 방지)
_PARK_GAP_M = 0.010          # 파킹 격자 간격 여유


def premature_tilt_limit_rad(fill: torch.Tensor, *, release_full_deg: float, release_span_deg: float,
                             margin_deg: float) -> torch.Tensor:
    """조준 없이 허용되는 소스 기울기 상한 [rad] (N,) = 첫 비드 유출 각도(채움 의존) − 여유.

    ★09.18 사용자 지적 "비드 양에 따라 기울이는 각도가 다르고 틸팅을 시도할 거리도 다르다": 고정 30° 를 버린다.
    프로브 실측: 가득 찬 컵은 약 70° 에서, 몇 알 든 컵은 약 90° 에서 첫 비드가 나간다 →
    release = full + span·(1 − fill). 유출각 − margin 까지는 어디서든(접근 중에도) 미리 기울여도 된다.
    """
    release = math.radians(float(release_full_deg)) + math.radians(float(release_span_deg)) * (
        1.0 - fill.clamp(0.0, 1.0))
    return release - math.radians(float(margin_deg))


def pour_dir_update(prev_dir: torch.Tensor, src_cup_xy: torch.Tensor, rcv_mouth_xy: torch.Tensor, *,
                    min_sep_m: float) -> torch.Tensor:
    """붓는 방향 d̂ (N,2) = 소스 컵 원점 → 리시버 입구 수평 단위벡터.

    컵 자세와 무관해서 직립(기울기 0)에서도 정의되고, 컵을 어느 쪽으로 기울였든 같은 쪽을 가리킨다
    (실제 림 최저점은 직립 근처에서 방향이 튀고, 반대로 기울이면 리시버 반대편을 가리킨다 — 사용자 지적 09.18).
    수평거리가 min_sep_m 보다 짧으면 방향이 흔들리므로 직전 값을 유지한다.
    """
    v = rcv_mouth_xy - src_cup_xy
    n = v.norm(dim=-1, keepdim=True)
    return torch.where(n >= float(min_sep_m), v / n.clamp(min=1e-6), prev_dir)


def pour_lip(src_mouth: torch.Tensor, src_up: torch.Tensor, pour_dir: torch.Tensor, *,
             rim_radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    """붓는 쪽 림 점 (N,3) 과 리시버 쪽 부호 있는 기울기 θ [rad] (N,).

    θ = atan2(up·d̂, up_z) — (d̂, z) 수직면 안의 기울기만 본다(옆으로 기운 성분은 0, 반대로 기울면 음수).
    림 점 = 입구 중심 + r·(cos θ·d̂ − sin θ·ẑ): 직립이면 리시버를 향한 쪽 림, 기울일수록 아래로 내려가고
    90° 를 넘으면 컵 밑으로 말려 들어간다(실제 유출 지점과 같은 거동). 입구 중심 자체가 컵 높이·sin θ 만큼
    d̂ 쪽으로 나가므로, 많이 기울여야 하는(비드 적은) 컵일수록 컵 몸통이 더 멀리서 조준이 성립한다.
    """
    u_d = (src_up[:, :2] * pour_dir).sum(dim=-1)
    theta = torch.atan2(u_d, src_up[:, 2])
    r = float(rim_radius)
    lip = src_mouth.clone()
    lip[:, :2] = lip[:, :2] + (r * torch.cos(theta)).unsqueeze(-1) * pour_dir
    lip[:, 2] = lip[:, 2] - r * torch.sin(theta)
    return lip, theta


def premature_tilt_now(src_tilt: torch.Tensor, tilt_limit: torch.Tensor, lip_dist_xy: torch.Tensor,
                       grasped: torch.Tensor, *, lip_max_m: float) -> torch.Tensor:
    """이 스텝의 순간 판정 (N,) bool — 래치(에피소드 누적)는 env 가 한다.

    **잡은** 소스 컵의 전체 기울기(방향 무관)가 채움별 상한(`premature_tilt_limit_rad`)을 넘었는데
    붓는 쪽 림 점(`pour_lip`)이 리시버 입구 중심에서 xy 로 lip_max_m 보다 멀면 조준 전 틸트.
    방향 무관이라 리시버 반대쪽·옆으로 크게 기울여도 걸린다.

    ★09.17 i08: 파지 조건 없이 걸었더니 첫 epoch 래치 19.8 % 가 전부 파지 0 에서 나왔다(탐색이 컵을 쳐서 넘어뜨림).
    래치는 영구라 정책이 소스 컵을 아예 피했다(파지 0.000, 이물 접촉 15→2 %). 리시버 규칙(파지·들기 이후에만)과
    같은 원칙으로 **잡은 채** 기울인 경우만 잡는다.
    """
    return (src_tilt > tilt_limit) & (lip_dist_xy > float(lip_max_m)) & grasped.to(torch.bool)


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
