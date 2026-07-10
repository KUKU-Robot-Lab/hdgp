"""논문 Algorithm 1의 parameter search space 샘플링.

seed calibration을 중심으로 group 단위 scale을 뽑는다.
joint별 독립 탐색은 하지 않는다 — 식별 가능성과 탐색 비용이 맞지 않는다 (가이드 §9.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from r2s_autotune.calibration_io import Calibration, GroupCalibration
from r2s_autotune.config import AutotuneConfig, GroupConfig, ScaleRange


@dataclass(frozen=True)
class Candidate:
    """후보 하나. index는 병렬 replay에서의 env_id와 같다."""

    index: int
    groups: Mapping[str, GroupCalibration]


def _sample(rng: np.random.Generator, scale: ScaleRange | None, count: int) -> np.ndarray:
    if scale is None:
        return np.ones(count)
    return rng.uniform(scale.low, scale.high, size=count)


def sample_candidates(
    config: AutotuneConfig,
    seed_calibration: Calibration,
) -> tuple[Candidate, ...]:
    """population_size개의 후보를 만든다. candidate 0은 항상 seed 자신이다.

    seed를 포함시켜야 "탐색이 seed보다 나아졌는가"를 같은 실행 안에서 판정할 수 있다.
    """
    missing = [name for name in config.tune_groups if name not in seed_calibration.groups]
    if missing:
        raise ValueError(f"seed calibration lacks tune groups: {missing}")

    rng = np.random.default_rng(config.random_seed)
    count = config.population_size

    scales: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in config.tune_groups:
        group: GroupConfig = config.groups[name]
        scales[name] = (
            _sample(rng, group.stiffness_scale, count),
            _sample(rng, group.damping_scale, count),
            _sample(rng, group.friction_scale, count),
        )

    candidates: list[Candidate] = []
    for index in range(count):
        groups = dict(seed_calibration.groups)
        for name in config.tune_groups:
            stiffness, damping, friction = scales[name]
            if index == 0:
                continue  # candidate 0 == seed
            groups[name] = seed_calibration.groups[name].scaled(
                stiffness=float(stiffness[index]),
                damping=float(damping[index]),
                friction=float(friction[index]),
            )
        candidates.append(Candidate(index=index, groups=groups))

    return tuple(candidates)
