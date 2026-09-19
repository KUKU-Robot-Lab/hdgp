"""pour_fabric 순수 규칙(torch 만) — 조준 전 틸트 래치 · 채움 정도 · 활성 비드 · 지름별 배치.

isaaclab 없이 로컬에서 돈다. 09.16 사용자 결정: 비드 30 mm × 개수 DR(파킹) · 채움 정도 0~1 obs ·
입구 xy 거리 > 0.10 m 에서 30° 넘으면 성공 무효.
"""
import math

import pytest
import torch

from openarm.agnostic.tasks.pour_fabric import pour_rules as R


# ---------------------------------------------------------------- 조준 전 틸트 래치
def test_premature_tilt_only_when_grasped_far_and_tilted():
    # 09.17 i08 실측: 첫 epoch 래치 19.8 % 가 소스 파지 0 에서 나왔다(탐색이 컵을 쳐서 넘어뜨림) → 정책이 소스 컵 회피.
    #   래치는 리시버 규칙과 같은 원칙으로 **파지 성립 시에만** — 잡은 채 테이블 위에서 기울이는 i07 식 행동만 잡는다.
    # 09.18: 상한은 고정 30° 가 아니라 채움별 텐서, 거리는 입구 중심이 아니라 붓는 쪽 림 점 기준.
    tilt = torch.tensor([math.radians(a) for a in (10.0, 60.0, 60.0, 51.0, 60.0)])
    limit = torch.full((5,), math.radians(52.0))
    lip = torch.tensor([0.30, 0.30, 0.05, 0.30, 0.30])
    grasped = torch.tensor([True, True, True, True, False])
    out = R.premature_tilt_now(tilt, limit, lip, grasped, lip_max_m=0.081)
    assert out.tolist() == [False, True, False, False, False]   # 마지막: 잡지 않은 채 넘어진 컵은 래치 아님


def test_tilt_limit_follows_fill():
    # 가득 72° − 20° = 52°, 빈 컵 72 + 34 − 20 = 86°, 채움 0.5 는 69°. 범위 밖 채움은 잘린다.
    fill = torch.tensor([1.0, 0.5, 0.0, 1.7, -0.3])
    lim = torch.rad2deg(R.premature_tilt_limit_rad(fill, release_full_deg=72.0, release_span_deg=34.0,
                                                   margin_deg=20.0))
    assert lim.tolist() == pytest.approx([52.0, 69.0, 86.0, 52.0, 86.0], abs=1e-4)


def test_pour_dir_holds_previous_when_cups_overlap():
    prev = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    src = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    rcv = torch.tensor([[0.3, 0.0], [0.005, 0.0]])          # 둘째: 수평거리 5 mm < min_sep → 직전 값 유지
    d = R.pour_dir_update(prev, src, rcv, min_sep_m=0.02)
    assert d[0].tolist() == pytest.approx([1.0, 0.0])
    assert d[1].tolist() == pytest.approx([0.0, 1.0])


def test_pour_lip_ignores_wrong_way_and_sideways_tilt():
    # 사용자 지적 09.18: 림 최저점을 쓰면 반대/옆으로 기울였을 때 엉뚱한 쪽을 가리킨다 → 방향은 d̂ 고정.
    r = 0.041
    mouth = torch.zeros(4, 3)
    d = torch.tensor([[1.0, 0.0]]).repeat(4, 1)
    a = math.radians(40.0)
    up = torch.tensor([
        [0.0, 0.0, 1.0],                       # 직립
        [math.sin(a), 0.0, math.cos(a)],       # 리시버 쪽 40°
        [-math.sin(a), 0.0, math.cos(a)],      # 반대쪽 40°
        [0.0, math.sin(a), math.cos(a)],       # 옆으로 40°
    ])
    lip, theta = R.pour_lip(mouth, up, d, rim_radius=r)
    assert torch.rad2deg(theta).tolist() == pytest.approx([0.0, 40.0, -40.0, 0.0], abs=1e-4)
    assert lip[0].tolist() == pytest.approx([r, 0.0, 0.0], abs=1e-6)                       # 직립에서도 정의됨
    assert lip[1].tolist() == pytest.approx([r * math.cos(a), 0.0, -r * math.sin(a)], abs=1e-6)
    assert lip[2, 0] > 0 and lip[2, 2] > 0     # 반대로 기울여도 림 점은 리시버 쪽에 남는다(위로 올라갈 뿐)
    assert lip[3].tolist() == pytest.approx([r, 0.0, 0.0], abs=1e-6)                       # 옆 기울기는 무시


# ---------------------------------------------------------------- 채움 정도
def test_fill_level_is_twice_mean_height_over_cup_height():
    bottom, top = -0.077, 0.100                       # 컵 내부 177 mm
    h = 0.120                                         # 120 mm 채움
    z = torch.linspace(bottom, bottom + h, 24).unsqueeze(0)   # 균일 기둥 → 평균 = bottom + h/2
    active = torch.ones(1, 24, dtype=torch.bool)
    lvl = R.fill_level_from_local_z(z, active, bottom_z=bottom, top_z=top)
    assert lvl.shape == (1,)
    assert abs(float(lvl[0]) - h / (top - bottom)) < 1e-3


def test_fill_level_ignores_parked_beads_and_clamps():
    bottom, top = -0.077, 0.100
    z = torch.full((2, 6), -5.0)                      # 파킹(멀리 아래)
    z[0, :3] = torch.tensor([bottom + 0.01, bottom + 0.03, bottom + 0.05])
    active = torch.zeros(2, 6, dtype=torch.bool)
    active[0, :3] = True
    lvl = R.fill_level_from_local_z(z, active, bottom_z=bottom, top_z=top)
    assert abs(float(lvl[0]) - (2 * 0.03) / (top - bottom)) < 1e-4
    assert float(lvl[1]) == 0.0                       # 활성 0개 → 0
    z_over = torch.full((1, 4), top + 1.0)
    assert float(R.fill_level_from_local_z(z_over, torch.ones(1, 4, dtype=torch.bool),
                                           bottom_z=bottom, top_z=top)[0]) == 1.0


# ---------------------------------------------------------------- 활성 비드 마스크
def test_active_mask_is_prefix_with_count_in_range():
    torch.manual_seed(0)
    m = R.sample_active_mask(64, k=26, lo=6, hi=26, device="cpu")
    assert m.shape == (64, 26) and m.dtype == torch.bool
    n = m.sum(dim=1)
    assert int(n.min()) >= 6 and int(n.max()) <= 26
    assert int(n.min()) < int(n.max())                # 실제로 다양하다
    idx = torch.arange(26).unsqueeze(0)
    assert torch.equal(m, idx < n.unsqueeze(1))       # 앞쪽 n 개만 True(아래층 우선)


def test_active_mask_fixed_when_lo_equals_hi():
    m = R.sample_active_mask(8, k=10, lo=10, hi=10, device="cpu")
    assert bool(m.all())


def test_active_mask_rejects_bad_range():
    with pytest.raises(ValueError):
        R.sample_active_mask(4, k=10, lo=0, hi=10, device="cpu")
    with pytest.raises(ValueError):
        R.sample_active_mask(4, k=10, lo=5, hi=11, device="cpu")


# ---------------------------------------------------------------- 지름별 컵 안 배치
@pytest.mark.parametrize("d,n", [(0.012, 20), (0.030, 26), (0.024, 30)])
def test_layout_fits_inside_cup_and_never_overlaps(d, n):
    r_in, bottom = 0.041, -0.077
    pts = torch.tensor(R.bead_layout_in_cup(n, diameter=d, inner_radius=r_in, bottom_z=bottom))
    assert pts.shape == (n, 3)
    assert bool((pts[:, :2].norm(dim=-1) + d / 2 <= r_in).all())         # 벽 안
    assert bool((pts[:, 2] - d / 2 >= bottom).all())                     # 바닥 위
    dist = torch.cdist(pts, pts) + torch.eye(n) * 10.0
    assert float(dist.min()) >= d + 0.0015                               # 소환 겹침 없음(08.17 사고 재발 방지)
    assert bool((pts[1:, 2] >= pts[:-1, 2]).all())                       # 아래층부터(활성 prefix 규약)


# ---------------------------------------------------------------- 파킹 격자
def test_park_offsets_are_spread_not_piled():
    d = 0.030
    pts = torch.tensor(R.park_offsets(26, diameter=d, origin_xy=(-1.0, -0.6), z=-0.083, per_row=13))
    assert pts.shape == (26, 3)
    dist = torch.cdist(pts, pts) + torch.eye(26) * 10.0
    assert float(dist.min()) >= d + 0.005                                # 한 점에 모으면 브로드페이즈 폭발
    assert bool((pts[:, 0] <= -0.5).all())                               # 로봇·테이블(+x) 뒤쪽
    assert bool((pts[:, 2] == -0.083).all())


# ---------------------------------------------------------------- 09.20 i16 영상 지적 4종(사용자) 판정
def test_pour_dir_ok_rejects_outward_pour():
    # i16 계측: 기울인 구간 d̂ 평균 (0.90, 0.35) — 몸 바깥(+x)으로 부었다. 허용 = 바깥 성분 ≤ sin 30°.
    d = torch.tensor([[0.90, 0.35], [0.0, 1.0], [-0.6, 0.8], [0.49, 0.87], [0.51, 0.86]])
    ok = R.pour_dir_ok(d, max_outward=0.5)
    assert ok.tolist() == [False, True, True, True, False]


def test_rcv_side_ok_keeps_receiver_on_its_own_side():
    # i16 계측: 붓는 동안 리시버 컵 y 중앙값 −0.13(스폰 +0.16) — 왼팔이 중심선을 13 cm 넘었다.
    y = torch.tensor([-0.13, -0.03, 0.0, 0.16])
    assert R.rcv_side_ok(y, side_sign=1.0, min_side_m=-0.03).tolist() == [False, True, True, True]
    # 좌우가 바뀐 배치(리시버가 −y 쪽)에서도 같은 규칙
    assert R.rcv_side_ok(-y, side_sign=-1.0, min_side_m=-0.03).tolist() == [False, True, True, True]


def test_wrap_count_ignores_tip_only_contact():
    # i16 계측: 리시버 손 4지는 팁만 닿았다(중간·원위 0 N) — 팁만 닿은 손가락은 감싼 것으로 세지 않는다.
    mid = torch.tensor([[8.7, 0.0, 0.0, 0.0, 0.0], [4.6, 0.0, 0.0, 0.0, 0.0]])
    dist = torch.tensor([[0.9, 0.0, 0.0, 0.3, 0.0], [3.0, 0.0, 3.8, 2.9, 0.0]])
    n = R.wrap_count(mid, dist, thr=1.0)
    assert n.tolist() == [1.0, 3.0]


def test_cup_hit_now_is_off_during_hold():
    f = torch.tensor([0.0, 0.9, 1.1, 5.0])
    hold = torch.tensor([False, False, False, True])
    assert R.cup_hit_now(f, hold, thr=1.0).tolist() == [False, False, True, False]
