"""Best candidate를 schema_version 1 calibration JSON으로 내보낸다.

재현성은 세 가지로 보장한다 (가이드 §6): calibration 경로, source_dataset, fit_error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from r2s_autotune.calibration_io import Calibration, write_calibration
from r2s_autotune.compute_tracking_error import ErrorBreakdown
from r2s_autotune.sample_candidates import Candidate


def export_best_calibration(
    output_path: str | Path,
    asset: str,
    source_dataset: str,
    candidates: Sequence[Candidate],
    errors: ErrorBreakdown,
    tune_groups: Sequence[str],
) -> Calibration:
    if len(candidates) != errors.total.size:
        raise ValueError("candidate count does not match error count")

    best = candidates[errors.best_index]
    seed_total = float(errors.total[0])
    best_total = float(errors.total[errors.best_index])

    calibration = Calibration(
        robot_asset=asset,
        source_dataset=str(source_dataset),
        groups=dict(best.groups),
        fit_error={
            "total": best_total,
            "mse_q": float(errors.mse_q[errors.best_index]),
            "mse_dq": float(errors.mse_dq[errors.best_index]),
            "delay_penalty": float(errors.delay_penalty[errors.best_index]),
        },
    )

    write_calibration(
        output_path,
        calibration,
        extra={
            "autotune": {
                "best_candidate_index": int(errors.best_index),
                "population_size": len(candidates),
                "tune_groups": list(tune_groups),
                "seed_total_error": seed_total,
                "improvement_ratio": (
                    0.0 if seed_total <= 0.0 else 1.0 - best_total / seed_total
                ),
                "error_spread": errors.spread(),
            }
        },
    )
    return calibration
