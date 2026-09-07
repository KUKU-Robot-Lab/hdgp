"""keypoint_goal 단위 테스트 (Isaac 불필요, 순수 torch). DESIGN.md §7-1 게이트."""

from __future__ import annotations

import math

import pytest
import torch

from openarm.agnostic.modules import keypoint_goal as KG

DEV = "cpu"
S = 0.09  # half_height (keypoint_scale 1.5 · fixed_height 0.12 / 2)


def _ident(n: int) -> torch.Tensor:
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return q


def _axis_quat(n: int, angle: float, axis) -> torch.Tensor:
    return KG.quat_from_angle_axis(torch.full((n,), angle), torch.tensor(axis, dtype=torch.float32).expand(n, 3))


# =============================================================================
# 쿼터니언 헬퍼 규약
# =============================================================================
def test_quat_apply_rotates_x_to_y_about_z():
    q = _axis_quat(1, math.pi / 2, (0.0, 0.0, 1.0))
    out = KG.quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(out, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_quat_mul_matches_composition():
    """quat_mul(a,b) 는 b 먼저·a 나중 (isaaclab 규약)."""
    a = _axis_quat(1, 0.7, (0.0, 0.0, 1.0))
    b = _axis_quat(1, 0.4, (1.0, 0.0, 0.0))
    v = torch.tensor([[0.3, -0.2, 0.5]])
    assert torch.allclose(KG.quat_apply(KG.quat_mul(a, b), v), KG.quat_apply(a, KG.quat_apply(b, v)), atol=1e-6)


def test_random_small_rotation_identity_and_unit_norm():
    assert torch.equal(KG.random_small_rotation(3, 0.0, DEV), _ident(3))
    q = KG.random_small_rotation(64, 15.0, DEV)
    assert torch.allclose(q.norm(dim=-1), torch.ones(64), atol=1e-6)
    assert (q[:, 0] >= math.cos(math.radians(7.5)) - 1e-6).all()  # |angle| ≤ 15°


# =============================================================================
# (1)(2)(3) 키포인트 불변성·단조성
# =============================================================================
def test_yaw_rotation_gives_zero_keypoint_dist():
    off = KG.keypoint_offsets(S, DEV)
    pos = torch.tensor([[0.3, -0.1, 0.25]]).expand(8, 3)
    yaw = KG.quat_from_angle_axis(torch.linspace(-math.pi, math.pi, 8), torch.tensor([[0.0, 0.0, 1.0]]).expand(8, 3))
    d = KG.keypoint_max_dist(KG.keypoints_world(pos, yaw, off), KG.keypoints_world(pos, _ident(8), off))
    assert torch.allclose(d, torch.zeros(8), atol=1e-6)


def test_pure_translation_dist_equals_norm():
    off = KG.keypoint_offsets(S, DEV)
    n = 16
    torch.manual_seed(0)
    pos = torch.randn(n, 3)
    t = torch.randn(n, 3) * 0.1
    q = KG.random_small_rotation(n, 30.0, DEV)
    d = KG.keypoint_max_dist(KG.keypoints_world(pos + t, q, off), KG.keypoints_world(pos, q, off))
    assert torch.allclose(d, t.norm(dim=-1), atol=1e-5)


def test_tilt_about_x_matches_chord_and_is_monotonic():
    off = KG.keypoint_offsets(S, DEV)
    thetas = torch.linspace(0.0, math.pi, 13)
    n = thetas.numel()
    pos = torch.zeros(n, 3)
    q = KG.quat_from_angle_axis(thetas, torch.tensor([[1.0, 0.0, 0.0]]).expand(n, 3))
    d = KG.keypoint_max_dist(KG.keypoints_world(pos, q, off), KG.keypoints_world(pos, _ident(n), off))
    expected = 2.0 * S * torch.sin(thetas / 2)  # 팁 키포인트(반경 s)의 현 길이
    assert torch.allclose(d, expected, atol=1e-5)
    assert (d[1:] > d[:-1]).all()


def test_keypoint_shape_errors_are_loud():
    off = KG.keypoint_offsets(S, DEV)
    with pytest.raises(ValueError):
        KG.keypoints_world(torch.zeros(2, 3), torch.zeros(3, 4), off)
    with pytest.raises(ValueError):
        KG.keypoints_world(torch.zeros(2, 3), _ident(2), off[:3])
    with pytest.raises(ValueError):
        KG.keypoint_max_dist(torch.zeros(2, 4, 3), torch.zeros(2, 3, 3))
    with pytest.raises(ValueError):
        KG.keypoint_offsets(0.0, DEV)


# =============================================================================
# (4) progress_delta 센티널
# =============================================================================
def test_progress_delta_sentinel_1d():
    curr = torch.tensor([0.5, 0.3, 0.2])
    closest = torch.tensor([-1.0, 0.4, 0.1])
    delta, new = KG.progress_delta(curr, closest)
    assert torch.allclose(delta, torch.tensor([0.0, 0.1, 0.0]))   # 센티널 0 · 전진 0.1 · 후퇴 0
    assert torch.allclose(new, torch.tensor([0.5, 0.3, 0.1]))     # 센티널→curr · min


def test_progress_delta_sentinel_2d():
    curr = torch.tensor([[0.5, 0.2], [0.3, 0.3]])
    closest = torch.tensor([[-1.0, 0.4], [0.1, -1.0]])
    delta, new = KG.progress_delta(curr, closest)
    assert torch.allclose(delta, torch.tensor([[0.0, 0.2], [0.0, 0.0]]))
    assert torch.allclose(new, torch.tensor([[0.5, 0.2], [0.1, 0.3]]))
    with pytest.raises(ValueError):
        KG.progress_delta(curr, closest[:, :1])


# =============================================================================
# (5) update_near_goal 누적 vs 연속
# =============================================================================
def _run_near(force_consecutive: bool, pattern):
    cfg = KG.GoalSeqCfg(success_steps=3, force_consecutive=force_consecutive)
    tr = KG.GoalTrackers(1, 5, DEV)
    succ = []
    for near in pattern:
        d = torch.tensor([0.0 if near else 1.0])
        ng, ok = KG.update_near_goal(d, 0.05, tr, cfg)
        assert bool(ng[0]) is near
        succ.append(bool(ok[0]))
    return succ, int(tr.near_goal_steps[0])


def test_update_near_goal_cumulative_ignores_gaps():
    succ, steps = _run_near(False, [True, False, True, False, True])
    assert succ == [False, False, False, False, True] and steps == 3


def test_update_near_goal_consecutive_resets_on_gap():
    succ, steps = _run_near(True, [True, False, True, False, True])
    assert succ == [False] * 5 and steps == 1
    succ, steps = _run_near(True, [True, True, True])
    assert succ == [False, False, True] and steps == 3


# =============================================================================
# (6) GoalTrackers 장부
# =============================================================================
def test_trackers_clear_goal_and_full_reset_bookkeeping():
    tr = KG.GoalTrackers(3, 2, DEV)
    assert (tr.closest_kp == -1).all() and (tr.closest_ft == -1).all()
    tr.closest_kp[:] = 0.2
    tr.closest_ft[:] = 0.1
    tr.near_goal_steps[:] = 4
    tr.successes[:] = torch.tensor([2, 5, 7])
    ids = torch.tensor([0, 2])
    tr.clear_goal(ids)
    assert (tr.closest_kp[ids] == -1).all() and (tr.closest_ft[ids] == -1).all() and (tr.near_goal_steps[ids] == 0).all()
    assert tr.closest_kp[1] == pytest.approx(0.2) and tr.near_goal_steps[1] == 4
    assert torch.equal(tr.successes, torch.tensor([2, 5, 7]))  # clear_goal 은 성공수를 건드리지 않는다
    tr.full_reset(ids)
    assert torch.equal(tr.prev_episode_successes, torch.tensor([2, 0, 7]))
    assert torch.equal(tr.successes, torch.tensor([0, 5, 0]))
    tr.successes[0] = 9
    tr.full_reset(torch.tensor([0]))
    assert int(tr.prev_episode_successes[0]) == 9 and int(tr.successes[0]) == 0


# =============================================================================
# (7)(8) 목표 샘플러
# =============================================================================
def test_sample_first_goal_ranges_clamp_and_quat():
    torch.manual_seed(1)
    n = 512
    settled_pos = torch.tensor([[0.36, -0.16, 0.30]]).expand(n, 3)
    settled_quat = _axis_quat(n, 0.3, (0.0, 0.0, 1.0))
    cfg = KG.GoalSeqCfg(first_xy_range=0.05, first_z_range=(0.12, 0.20), first_tilt_deg=0.0)
    pos, quat = KG.sample_first_goal(settled_pos, settled_quat, cfg)
    delta = pos - settled_pos
    assert (delta[:, :2].abs() <= 0.05 + 1e-6).all()
    assert (delta[:, 2] >= 0.12 - 1e-6).all() and (delta[:, 2] <= 0.20 + 1e-6).all()
    assert delta[:, :2].abs().max() > 0.04 and delta[:, 2].min() < 0.13 and delta[:, 2].max() > 0.19  # 범위를 실제로 채운다
    assert torch.equal(quat, settled_quat)  # tilt 0 → 정착 자세 그대로
    # 박스 클램프: xy 상한을 settled 로 잡으면 +방향 샘플이 전부 잘린다
    cfg_box = KG.GoalSeqCfg(box_min=(-1.0, -1.0, 0.0), box_max=(0.36, -0.16, 0.45))
    pos_b, _ = KG.sample_first_goal(settled_pos, settled_quat, cfg_box)
    assert (pos_b[:, 0] <= 0.36 + 1e-6).all() and (pos_b[:, 1] <= -0.16 + 1e-6).all() and (pos_b[:, 2] <= 0.45).all()
    assert (pos_b[:, 0] == 0.36).sum() > n // 4  # 클램프가 실제로 작동


def test_sample_delta_goal_within_box_and_rotation_identity():
    torch.manual_seed(2)
    n = 512
    prev_pos = torch.tensor([[0.50, 0.0, 0.28]]).expand(n, 3)
    prev_quat = _axis_quat(n, 0.2, (1.0, 0.0, 0.0))
    cfg = KG.GoalSeqCfg(delta_distance=0.08, delta_rotation_deg=0.0,
                        box_min=(0.21, -0.15, 0.285), box_max=(0.51, 0.15, 0.505))
    pos, quat = KG.sample_delta_goal(prev_pos, prev_quat, cfg)
    lo, hi = torch.tensor(cfg.box_min), torch.tensor(cfg.box_max)
    assert (pos >= lo - 1e-6).all() and (pos <= hi + 1e-6).all()
    assert ((pos - prev_pos).abs() <= 0.08 + 1e-6).all()
    assert (pos[:, 0] == 0.51).any() and (pos[:, 2] == 0.285).any()  # 두 축에서 클램프가 작동
    assert torch.equal(quat, prev_quat)
    _, quat_r = KG.sample_delta_goal(prev_pos, prev_quat, KG.GoalSeqCfg(delta_rotation_deg=10.0))
    assert not torch.allclose(quat_r, prev_quat) and torch.allclose(quat_r.norm(dim=-1), torch.ones(n), atol=1e-6)


def test_goal_seq_cfg_validates():
    with pytest.raises(ValueError):
        KG.GoalSeqCfg(first_z_range=(0.2, 0.1))
    with pytest.raises(ValueError):
        KG.GoalSeqCfg(box_min=(1.0, 0.0, 0.0), box_max=(0.0, 0.0, 0.0))


# =============================================================================
# (9) ToleranceCurriculum
# =============================================================================
def test_tolerance_curriculum_gates_on_interval_and_threshold_and_floor():
    cur = KG.ToleranceCurriculum(start=0.06, floor=0.015, factor=0.9, interval=5, success_threshold=2.0)
    good = torch.tensor([3, 2, 1])   # mean 2.0 ≥ threshold
    bad = torch.tensor([1, 1, 1])    # mean 1.0 < threshold
    assert [cur.update(good) for _ in range(4)] == [False] * 4      # interval 미달
    assert cur.tol == pytest.approx(0.06)
    assert cur.update(bad) is False and cur.tol == pytest.approx(0.06)  # interval 충족·지표 미달
    for _ in range(4):
        assert cur.update(good) is False
    assert cur.update(good) is True and cur.tol == pytest.approx(0.054)  # 둘 다 충족
    for _ in range(200):
        cur.update(good)
    assert cur.tol == pytest.approx(0.015)
    assert all(cur.update(good) is False for _ in range(10))  # floor 에서는 더 줄지도, 보고하지도 않는다


def test_tolerance_curriculum_floor_is_never_crossed_and_stops_reporting():
    cur = KG.ToleranceCurriculum(start=0.02, floor=0.015, factor=0.5, interval=1)
    good = torch.tensor([5.0])
    assert cur.update(good) is True and cur.tol == pytest.approx(0.015)
    assert cur.update(good) is False and cur.tol == pytest.approx(0.015)
    with pytest.raises(ValueError):
        KG.ToleranceCurriculum(start=0.01, floor=0.015)
