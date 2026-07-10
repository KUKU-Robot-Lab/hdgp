import numpy as np
import pytest

from r2s_autotune.excitation import (
    ExcitationSpec,
    build_excitation,
    interior_neutral,
    is_saturated,
)


def _bounds(num_joints: int = 3):
    neutral = np.zeros(num_joints)
    lower = np.full(num_joints, -1.0)
    upper = np.full(num_joints, 1.0)
    return neutral, lower, upper


def test_sequence_starts_and_ends_at_neutral_hold():
    neutral, lower, upper = _bounds()
    spec = ExcitationSpec(dt=0.01, hold_sec=0.5)

    _, q_cmd = build_excitation(neutral, lower, upper, spec)

    np.testing.assert_allclose(q_cmd[0], neutral)
    np.testing.assert_allclose(q_cmd[-1], neutral)


def test_time_matches_dt_and_length():
    neutral, lower, upper = _bounds()
    spec = ExcitationSpec(dt=0.02)

    time, q_cmd = build_excitation(neutral, lower, upper, spec)

    assert time.shape[0] == q_cmd.shape[0]
    np.testing.assert_allclose(np.diff(time), spec.dt)


def test_command_never_exceeds_limits_minus_margin():
    neutral, lower, upper = _bounds()
    spec = ExcitationSpec(step_rad=5.0, sine_amp_rad=5.0, limit_margin=0.05)

    _, q_cmd = build_excitation(neutral, lower, upper, spec)

    assert np.all(q_cmd >= lower + spec.limit_margin - 1e-9)
    assert np.all(q_cmd <= upper - spec.limit_margin + 1e-9)


def test_degenerate_limits_collapse_to_midpoint_instead_of_inverting():
    neutral = np.array([0.0])
    lower, upper = np.array([-0.01]), np.array([0.01])
    spec = ExcitationSpec(limit_margin=0.05)

    _, q_cmd = build_excitation(neutral, lower, upper, spec)

    np.testing.assert_allclose(q_cmd, 0.0)


def test_sequence_actually_excites_the_joint():
    """모든 후보가 같은 오차를 내는 것을 막으려면 명령이 실제로 움직여야 한다."""
    neutral, lower, upper = _bounds()

    _, q_cmd = build_excitation(neutral, lower, upper, ExcitationSpec())

    assert float(np.max(np.abs(q_cmd - neutral))) > 0.1


def test_interior_neutral_moves_a_neutral_that_sits_on_a_limit():
    """Tesollo curl 관절: default=0 인데 하한도 0. 그대로 두면 관절이 한계를 뚫는다."""
    spec = ExcitationSpec(step_rad=0.15, sine_amp_rad=0.20, limit_margin=0.05)

    neutral = interior_neutral(
        default=np.array([0.0]), lower=np.array([0.0]), upper=np.array([2.007]), spec=spec
    )

    assert neutral[0] == pytest.approx(0.25)  # 0 + margin 0.05 + amplitude 0.20


def test_interior_neutral_leaves_a_comfortable_neutral_untouched():
    spec = ExcitationSpec()
    neutral = interior_neutral(np.array([0.5]), np.array([-2.0]), np.array([2.0]), spec)

    assert neutral[0] == pytest.approx(0.5)


def test_interior_neutral_collapses_to_midpoint_when_range_is_too_narrow():
    spec = ExcitationSpec()
    neutral = interior_neutral(np.array([0.0]), np.array([-0.01]), np.array([0.01]), spec)

    assert neutral[0] == pytest.approx(0.0)


def test_excitation_from_interior_neutral_is_never_clamped():
    spec = ExcitationSpec()
    lower, upper = np.array([0.0]), np.array([2.007])
    neutral = interior_neutral(np.array([0.0]), lower, upper, spec)

    assert not is_saturated(neutral, lower, upper, spec)[0]


def test_saturation_is_detected_when_neutral_sits_on_the_limit():
    spec = ExcitationSpec()
    lower, upper = np.array([0.0]), np.array([2.007])

    assert is_saturated(np.array([0.0]), lower, upper, spec)[0]


def test_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="lower bound exceeds upper bound"):
        build_excitation(np.zeros(1), np.array([1.0]), np.array([-1.0]))


def test_rejects_non_positive_dt():
    with pytest.raises(ValueError, match="dt must be positive"):
        ExcitationSpec(dt=0.0)
