"""object_wrench 단위 테스트 (Isaac 불필요, cpu)."""

from __future__ import annotations

import pytest
import torch

from openarm.agnostic.modules import object_wrench as OW

DEV = "cpu"


def test_sample_log_uniform_range_and_shape():
    torch.manual_seed(0)
    s = OW.sample_log_uniform(0.001, 0.1, 10_000, DEV)
    assert s.shape == (10_000,)
    assert float(s.min()) >= 0.001 and float(s.max()) <= 0.1
    # log 균일: 중앙값 ≈ 기하평균 0.01
    assert abs(float(s.log().median().exp()) - 0.01) < 0.002
    with pytest.raises(ValueError):
        OW.sample_log_uniform(0.0, 0.1, 3, DEV)
    with pytest.raises(ValueError):
        OW.sample_log_uniform(0.2, 0.1, 3, DEV)


def test_probabilities_within_range_after_reset():
    w = OW.WrenchDR(512, DEV, prob_range=(0.001, 0.1))
    for p in (w.p_force, w.p_torque):
        assert p.shape == (512,)
        assert float(p.min()) >= 0.001 and float(p.max()) <= 0.1
    w.reset(torch.arange(512))
    for p in (w.p_force, w.p_torque):
        assert float(p.min()) >= 0.001 and float(p.max()) <= 0.1
    assert w.forces.shape == (512, 1, 3) and w.torques.shape == (512, 1, 3)


def test_zero_wrench_when_not_lifted():
    w = OW.WrenchDR(256, DEV, prob_range=(1.0, 1.0))      # 항상 발화하려 해도
    mass = torch.full((256,), 0.3)
    for _ in range(5):
        f, t = w.step(mass, torch.zeros(256, dtype=torch.bool))
        assert torch.equal(f, torch.zeros(256, 1, 3))
        assert torch.equal(t, torch.zeros(256, 1, 3))


def test_lifted_gate_per_env():
    w = OW.WrenchDR(8, DEV, prob_range=(1.0, 1.0))
    lifted = torch.tensor([True, False] * 4)
    f, t = w.step(torch.ones(8), lifted)
    assert bool((f[~lifted] == 0).all()) and bool((t[~lifted] == 0).all())
    assert bool((f[lifted] != 0).any(-1).all()) and bool((t[lifted] != 0).any(-1).all())


def test_magnitude_scales_linearly_with_mass():
    n = 64
    lifted = torch.ones(n, dtype=torch.bool)
    w1 = OW.WrenchDR(n, DEV, prob_range=(1.0, 1.0), force_scale=20.0, torque_scale=2.0)
    torch.manual_seed(7)
    f1, t1 = w1.step(torch.full((n,), 0.2), lifted)
    w2 = OW.WrenchDR(n, DEV, prob_range=(1.0, 1.0), force_scale=20.0, torque_scale=2.0)
    torch.manual_seed(7)
    f2, t2 = w2.step(torch.full((n,), 0.6), lifted)
    assert torch.allclose(f2, 3.0 * f1) and torch.allclose(t2, 3.0 * t1)
    # 스케일 확인: std ≈ mass·scale
    assert abs(float(f1.std()) - 0.2 * 20.0) < 0.2 * 20.0 * 0.3
    assert abs(float(t1.std()) - 0.2 * 2.0) < 0.2 * 2.0 * 0.3


def test_forces_zeroed_each_step_before_resample():
    torch.manual_seed(1)
    w = OW.WrenchDR(2000, DEV, prob_range=(0.05, 0.05))
    lifted = torch.ones(2000, dtype=torch.bool)
    mass = torch.ones(2000)
    w.step(mass, lifted)
    f, t = w.step(mass, lifted)
    # decay 0: 이번 스텝에 발화하지 않은 env 는 이전 값을 잇지 않고 0
    nz_f = (f != 0).any(-1).any(-1).float().mean()
    nz_t = (t != 0).any(-1).any(-1).float().mean()
    assert abs(float(nz_f) - 0.05) < 0.02 and abs(float(nz_t) - 0.05) < 0.02
    # reset 은 wrench 를 0 으로
    w.reset(torch.arange(2000))
    assert torch.equal(w.forces, torch.zeros(2000, 1, 3))


def test_step_shape_checks_raise():
    w = OW.WrenchDR(4, DEV)
    with pytest.raises(ValueError):
        w.step(torch.ones(3), torch.ones(4, dtype=torch.bool))
    with pytest.raises(ValueError):
        w.step(torch.ones(4), torch.ones(4))
    with pytest.raises(ValueError):
        OW.WrenchDR(4, DEV, prob_range=(0.5,))
