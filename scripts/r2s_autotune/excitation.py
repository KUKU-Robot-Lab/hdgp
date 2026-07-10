"""Identification용 여기(excitation) 명령 시퀀스.

sim replay와 실물 수집이 **같은 시퀀스**를 써야 하므로 여기 한 곳에만 정의한다.
teleop demo는 stiffness/damping/delay를 분리 식별하기에 부족하다 (가이드 §4).

실물에 그대로 나가는 명령이므로 관절 한계를 넘지 않도록 항상 clamp한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_BOUND_TOL = 1e-9


@dataclass(frozen=True)
class ExcitationSpec:
    """가이드 §7.2의 Tesollo 초기 시퀀스를 파라미터화한 것."""

    dt: float = 0.01
    hold_sec: float = 2.0
    step_rad: float = 0.15
    ramp_sec: float = 3.0
    sine_amp_rad: float = 0.20
    sine_freq_hz: float = 0.25
    sine_cycles: float = 2.0
    limit_margin: float = 0.05  # 관절 한계에서 남겨둘 여유 [rad]

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.sine_freq_hz <= 0.0:
            raise ValueError("sine_freq_hz must be positive")


def _steps(seconds: float, dt: float) -> int:
    return max(1, int(round(seconds / dt)))


def interior_neutral(
    default: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    spec: ExcitationSpec = ExcitationSpec(),
) -> np.ndarray:
    """excitation 전 구간이 관절 한계 안에 들어오도록 neutral을 범위 안쪽으로 민다.

    Tesollo curl 관절(r_hj_*_2)은 default_joint_pos가 0인데 하한도 정확히 0이다.
    그 자리에서 구동하면 PD가 한계 구속과 싸우다 관절이 한계를 뚫고 나간다.
    한계에 얹힌 neutral에서는 tracking 오차가 actuator 특성이 아니라 구속을 반영하므로
    식별이 불가능하다.
    """
    default = np.asarray(default, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    amplitude = max(spec.step_rad, spec.sine_amp_rad)
    safe_lower = lower + spec.limit_margin + amplitude
    safe_upper = upper - spec.limit_margin - amplitude

    midpoint = 0.5 * (lower + upper)
    degenerate = safe_lower > safe_upper
    safe_lower = np.where(degenerate, midpoint, safe_lower)
    safe_upper = np.where(degenerate, midpoint, safe_upper)
    return np.clip(default, safe_lower, safe_upper)


def is_saturated(
    neutral: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    spec: ExcitationSpec = ExcitationSpec(),
) -> np.ndarray:
    """관절별로 excitation이 한계에 잘렸는지. True면 그 관절은 식별성이 떨어진다.

    한계를 스치는 것(경계에 정확히 닿음)은 잘림이 아니므로, clamp 전 원형과 비교한다.
    """
    raw = _raw_profile(np.asarray(neutral, dtype=np.float64), spec)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    # 경계를 정확히 스치는 것은 잘림이 아니다. 반올림 오차(1 ulp)를 잘림으로 보지 않는다.
    below = (raw < lower + spec.limit_margin - _BOUND_TOL).any(axis=0)
    above = (raw > upper - spec.limit_margin + _BOUND_TOL).any(axis=0)
    return below | above


def build_excitation(
    neutral: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    spec: ExcitationSpec = ExcitationSpec(),
) -> tuple[np.ndarray, np.ndarray]:
    """neutral hold → step → hold → return → ramp → sine 시퀀스를 만든다.

    Returns (time [T], q_cmd [T, J]) — 모두 rad, 관절 한계 안으로 clamp됨.
    """
    neutral = np.asarray(neutral, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if not (neutral.shape == lower.shape == upper.shape) or neutral.ndim != 1:
        raise ValueError("neutral, lower, upper must be 1-D arrays of equal length")
    if np.any(lower > upper):
        raise ValueError("lower bound exceeds upper bound")

    q_cmd = _clamp(_raw_profile(neutral, spec), lower, upper, spec.limit_margin)
    time = np.arange(q_cmd.shape[0], dtype=np.float64) * spec.dt
    return time, q_cmd


def _raw_profile(neutral: np.ndarray, spec: ExcitationSpec) -> np.ndarray:
    """관절 한계를 무시한 원형 시퀀스. clamp 여부 판정에도 쓰인다."""
    return np.concatenate(
        [
            _hold(neutral, _steps(spec.hold_sec, spec.dt)),
            _step(neutral, spec.step_rad, _steps(spec.hold_sec, spec.dt)),
            _ramp(neutral, spec),
            _sine(neutral, spec),
            _hold(neutral, _steps(spec.hold_sec, spec.dt)),
        ],
        axis=0,
    )


def _hold(neutral: np.ndarray, steps: int) -> np.ndarray:
    return np.tile(neutral, (steps, 1))


def _step(neutral: np.ndarray, amplitude: float, hold_steps: int) -> np.ndarray:
    """step 인가 후 hold, 그리고 neutral 복귀 후 hold."""
    stepped = _hold(neutral + amplitude, hold_steps)
    returned = _hold(neutral, hold_steps)
    return np.concatenate([stepped, returned], axis=0)


def _ramp(neutral: np.ndarray, spec: ExcitationSpec) -> np.ndarray:
    steps = _steps(spec.ramp_sec, spec.dt)
    up = np.linspace(0.0, spec.step_rad, steps)
    profile = np.concatenate([up, up[::-1]])
    return neutral[None, :] + profile[:, None]


def _sine(neutral: np.ndarray, spec: ExcitationSpec) -> np.ndarray:
    duration = spec.sine_cycles / spec.sine_freq_hz
    steps = _steps(duration, spec.dt)
    t = np.arange(steps, dtype=np.float64) * spec.dt
    profile = spec.sine_amp_rad * np.sin(2.0 * np.pi * spec.sine_freq_hz * t)
    return neutral[None, :] + profile[:, None]


def _clamp(
    q_cmd: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    margin: float,
) -> np.ndarray:
    safe_lower = lower + margin
    safe_upper = upper - margin
    # 한계 폭이 margin 두 배보다 좁으면 중점으로 붕괴시킨다 (실물 안전 우선).
    midpoint = 0.5 * (lower + upper)
    degenerate = safe_lower > safe_upper
    safe_lower = np.where(degenerate, midpoint, safe_lower)
    safe_upper = np.where(degenerate, midpoint, safe_upper)
    return np.clip(q_cmd, safe_lower, safe_upper)
