"""perception_delay 단위 테스트 (Isaac 불필요, cpu)."""

from __future__ import annotations

import math

import pytest
import torch

from openarm.agnostic.modules import perception_delay as PD

DEV = "cpu"


# =============================================================================
# DelayQueue
# =============================================================================
def test_max_delay_one_returns_exactly_pushed_values():
    q = PD.DelayQueue(8, 1, 3, DEV)
    flush = torch.zeros(8, dtype=torch.bool)
    for _ in range(5):
        v = torch.randn(8, 3)
        assert torch.equal(q.push(v, flush), v)


def test_flush_fills_all_slots():
    q = PD.DelayQueue(4, 5, 2, DEV)
    for _ in range(6):
        q.push(torch.randn(4, 2), torch.zeros(4, dtype=torch.bool))
    v = torch.randn(4, 2)
    flush = torch.tensor([True, False, True, False])
    out = q.push(v, flush)
    # flush env 는 전 슬롯이 v
    assert torch.equal(q.buf[flush], v[flush].unsqueeze(1).expand(-1, 5, -1))
    assert torch.equal(out[flush], v[flush])
    # 비flush env 는 방금 값이 슬롯 0 에만 (이전 값은 남아있다)
    assert torch.equal(q.buf[~flush, 0], v[~flush])
    # 이후 어떤 인덱스가 뽑혀도 flush env 는 v 를 돌려준다
    for _ in range(20):
        out = q.push(v, torch.zeros(4, dtype=torch.bool))
        assert torch.equal(out[flush], v[flush])


def test_sample_is_one_of_last_L_pushes_per_env():
    torch.manual_seed(0)
    n, L, d = 16, 4, 3
    q = PD.DelayQueue(n, L, d, DEV)
    history: list[torch.Tensor] = []
    for t in range(30):
        v = torch.randn(n, d)
        history.append(v)
        # 왜 첫 push 만 flush: 리셋 직후(episode_length_buf==0) 규약 — 0 초기값이 새지 않는다
        out = q.push(v, torch.full((n,), t == 0, dtype=torch.bool))
        recent = torch.stack(history[-L:], dim=1)                       # (n, ≤L, d)
        match = (recent == out.unsqueeze(1)).all(-1).any(-1)            # (n,)
        assert bool(match.all()), "sample not among last L pushes"


def test_sample_covers_all_delays():
    torch.manual_seed(1)
    n, L = 64, 3
    q = PD.DelayQueue(n, L, 1, DEV)
    seen = set()
    for t in range(1, 40):
        out = q.push(torch.full((n, 1), float(t)), torch.zeros(n, dtype=torch.bool))
        if t > L:
            seen.update((t - out[:, 0]).long().tolist())
    assert seen == set(range(L))


def test_reset_zeroes():
    q = PD.DelayQueue(4, 3, 2, DEV)
    q.push(torch.ones(4, 2), torch.ones(4, dtype=torch.bool))
    q.reset(torch.tensor([1, 3]))
    assert torch.equal(q.buf[[1, 3]], torch.zeros(2, 3, 2))
    assert torch.equal(q.buf[[0, 2]], torch.ones(2, 3, 2))


def test_shape_and_type_checks_raise():
    q = PD.DelayQueue(4, 3, 2, DEV)
    with pytest.raises(ValueError):
        q.push(torch.zeros(4, 3), torch.zeros(4, dtype=torch.bool))
    with pytest.raises(ValueError):
        q.push(torch.zeros(3, 2), torch.zeros(3, dtype=torch.bool))
    with pytest.raises(ValueError):
        q.push(torch.zeros(4, 2), torch.zeros(5, dtype=torch.bool))
    with pytest.raises(TypeError):
        q.push(torch.zeros(4, 2), torch.zeros(4))
    with pytest.raises(ValueError):
        PD.DelayQueue(4, 0, 2, DEV)


# =============================================================================
# perturb_quat / noisy_pose
# =============================================================================
def _rand_unit_quat(n: int) -> torch.Tensor:
    q = torch.randn(n, 4)
    return q / q.norm(dim=-1, keepdim=True)


def _angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dot = (a * b).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def test_perturb_quat_zero_is_identity():
    q = _rand_unit_quat(10)
    assert torch.equal(PD.perturb_quat(q, 0.0), q)


def test_perturb_quat_unit_norm_and_bounded_angle():
    torch.manual_seed(2)
    q = _rand_unit_quat(2000)
    out = PD.perturb_quat(q, 5.0)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2000), atol=1e-5)
    ang = _angle_between(out, q)
    assert float(ang.max()) <= math.radians(5.0) + 1e-4
    assert float(ang.max()) > math.radians(4.0)        # 범위 상단 근처까지 실제로 쓴다
    assert not torch.allclose(out, q)


def test_perturb_quat_rejects_bad_input():
    with pytest.raises(ValueError):
        PD.perturb_quat(torch.zeros(3, 3), 5.0)
    with pytest.raises(ValueError):
        PD.perturb_quat(_rand_unit_quat(3), -1.0)


def test_noisy_pose_std_matches():
    torch.manual_seed(3)
    n = 200_000
    pos = torch.zeros(n, 3)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, 4).contiguous()
    p, q = PD.noisy_pose(pos, quat, 0.01, 5.0)
    assert p.std(dim=0).allclose(torch.full((3,), 0.01), rtol=0.05)
    assert torch.allclose(p.mean(dim=0), torch.zeros(3), atol=1e-3)
    assert torch.allclose(q.norm(dim=-1), torch.ones(n), atol=1e-5)
    # 각도는 U(-5°,5°) 의 절댓값 → 평균 2.5°
    ang = torch.rad2deg(_angle_between(q, quat))
    assert abs(float(ang.mean()) - 2.5) < 0.1


def test_noisy_pose_zero_noise_is_identity():
    pos, quat = torch.randn(5, 3), _rand_unit_quat(5)
    p, q = PD.noisy_pose(pos, quat, 0.0, 0.0)
    assert torch.equal(p, pos) and torch.equal(q, quat)


def test_noisy_pose_rejects_mismatch():
    with pytest.raises(ValueError):
        PD.noisy_pose(torch.zeros(4, 3), _rand_unit_quat(3), 0.01, 5.0)
    with pytest.raises(ValueError):
        PD.noisy_pose(torch.zeros(4, 2), _rand_unit_quat(4), 0.01, 5.0)
