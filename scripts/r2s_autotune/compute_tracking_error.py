"""Real/sim tracking 오차. 논문 Algorithm 1의 error 항.

    error = w_q * MSE(q_sim, q_real)
          + w_dq * MSE(dq_sim, dq_real)
          + w_delay * |lag_sim - lag_real|

contact/FT sensor는 관절 동역학이 맞은 뒤(phase 2)에 추가한다 (가이드 §9.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from r2s_autotune.config import ErrorWeights

DEFAULT_MAX_LAG_STEPS = 25


@dataclass(frozen=True)
class ErrorBreakdown:
    """후보별 오차 분해. total만 보고 판단하면 원인을 놓친다."""

    total: np.ndarray  # [K]
    mse_q: np.ndarray
    mse_dq: np.ndarray
    delay_penalty: np.ndarray

    @property
    def best_index(self) -> int:
        return int(np.argmin(self.total))

    def spread(self) -> float:
        """후보 간 total 오차의 상대 산포. 0에 가까우면 excitation이 약하다 (가이드 §11.2)."""
        mean = float(np.mean(self.total))
        if mean <= 0.0:
            return 0.0
        return float(np.std(self.total) / mean)


def estimate_lag_steps(command: np.ndarray, measured: np.ndarray, max_lag: int) -> int:
    """command 대비 measured의 지연 스텝 수를 상호상관으로 추정한다.

    정지 구간이 오래 이어지면 상관이 평평해지므로 신호의 변화량(1차 차분)을 쓴다.
    """
    d_cmd = np.diff(command, axis=0).ravel()
    d_meas = np.diff(measured, axis=0).ravel()
    if d_cmd.size == 0 or np.allclose(d_cmd, 0.0) or np.allclose(d_meas, 0.0):
        return 0

    best_lag, best_score = 0, -np.inf
    for lag in range(0, min(max_lag, d_cmd.size - 1) + 1):
        shifted = d_meas[lag:]
        reference = d_cmd[: shifted.size]
        if shifted.size < 2:
            break
        score = float(np.dot(reference, shifted))
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag


def compute_tracking_error(
    q_cmd: np.ndarray,
    q_real: np.ndarray,
    dq_real: np.ndarray,
    q_sim: np.ndarray,
    dq_sim: np.ndarray,
    weights: ErrorWeights,
    dt: float,
    max_lag_steps: int = DEFAULT_MAX_LAG_STEPS,
) -> ErrorBreakdown:
    """q_sim/dq_sim은 [K, T, J], 나머지는 [T, J]."""
    if q_sim.ndim != 3 or dq_sim.ndim != 3:
        raise ValueError("q_sim and dq_sim must be [K, T, J]")
    if q_sim.shape[1:] != q_real.shape:
        raise ValueError(f"sim/real shape mismatch: {q_sim.shape[1:]} vs {q_real.shape}")
    if q_sim.shape != dq_sim.shape:
        raise ValueError("q_sim and dq_sim must have the same shape")

    mse_q = np.mean(np.square(q_sim - q_real[None, ...]), axis=(1, 2))
    mse_dq = np.mean(np.square(dq_sim - dq_real[None, ...]), axis=(1, 2))

    real_lag = estimate_lag_steps(q_cmd, q_real, max_lag_steps)
    delay = np.array(
        [
            abs(estimate_lag_steps(q_cmd, q_sim[k], max_lag_steps) - real_lag) * dt
            for k in range(q_sim.shape[0])
        ],
        dtype=np.float64,
    )

    total = weights.q * mse_q + weights.dq * mse_dq + weights.delay * delay
    return ErrorBreakdown(total=total, mse_q=mse_q, mse_dq=mse_dq, delay_penalty=delay)
