import numpy as np
import pytest

from r2s_autotune.compute_tracking_error import (
    compute_tracking_error,
    estimate_lag_steps,
)
from r2s_autotune.config import ErrorWeights

WEIGHTS = ErrorWeights()
DT = 0.01


def _ramp(steps=100, joints=2):
    t = np.linspace(0.0, 1.0, steps)[:, None]
    return np.tile(t, (1, joints))


def test_perfect_match_yields_zero_error():
    q = _ramp()
    dq = np.gradient(q, DT, axis=0)

    errors = compute_tracking_error(q, q, dq, q[None], dq[None], WEIGHTS, DT)

    assert errors.total[0] == pytest.approx(0.0, abs=1e-12)


def test_best_index_selects_the_closest_candidate():
    q_cmd = _ramp()
    q_real = q_cmd.copy()
    dq_real = np.gradient(q_real, DT, axis=0)

    q_sim = np.stack([q_real + 0.5, q_real + 0.01, q_real + 0.2])
    dq_sim = np.stack([dq_real] * 3)

    errors = compute_tracking_error(q_cmd, q_real, dq_real, q_sim, dq_sim, WEIGHTS, DT)

    assert errors.best_index == 1


def test_velocity_error_is_weighted_below_position_error():
    q_cmd = _ramp()
    q_real = q_cmd.copy()
    dq_real = np.gradient(q_real, DT, axis=0)

    position_off = compute_tracking_error(
        q_cmd, q_real, dq_real, (q_real + 0.1)[None], dq_real[None], WEIGHTS, DT
    )
    velocity_off = compute_tracking_error(
        q_cmd, q_real, dq_real, q_real[None], (dq_real + 0.1)[None], WEIGHTS, DT
    )

    assert velocity_off.total[0] < position_off.total[0]


def test_spread_is_near_zero_when_candidates_are_identical():
    q_cmd = _ramp()
    dq = np.gradient(q_cmd, DT, axis=0)
    q_sim = np.stack([q_cmd + 0.1] * 4)

    errors = compute_tracking_error(q_cmd, q_cmd, dq, q_sim, np.stack([dq] * 4), WEIGHTS, DT)

    assert errors.spread() == pytest.approx(0.0, abs=1e-12)


def test_spread_is_positive_when_candidates_differ():
    q_cmd = _ramp()
    dq = np.gradient(q_cmd, DT, axis=0)
    q_sim = np.stack([q_cmd + 0.01, q_cmd + 0.5])

    errors = compute_tracking_error(q_cmd, q_cmd, dq, q_sim, np.stack([dq] * 2), WEIGHTS, DT)

    assert errors.spread() > 0.1


def test_lag_estimation_recovers_a_known_shift():
    command = np.zeros((60, 1))
    command[20:] = 1.0
    measured = np.zeros((60, 1))
    measured[25:] = 1.0

    assert estimate_lag_steps(command, measured, max_lag=20) == 5


def test_lag_estimation_returns_zero_for_a_constant_signal():
    constant = np.ones((30, 1))

    assert estimate_lag_steps(constant, constant, max_lag=10) == 0


def test_shape_mismatch_between_sim_and_real_is_rejected():
    q = _ramp()
    dq = np.gradient(q, DT, axis=0)

    with pytest.raises(ValueError, match="sim/real shape mismatch"):
        compute_tracking_error(q, q, dq, q[None, :, :1], dq[None, :, :1], WEIGHTS, DT)


def test_two_dimensional_sim_input_is_rejected():
    q = _ramp()
    dq = np.gradient(q, DT, axis=0)

    with pytest.raises(ValueError, match=r"\[K, T, J\]"):
        compute_tracking_error(q, q, dq, q, dq, WEIGHTS, DT)
