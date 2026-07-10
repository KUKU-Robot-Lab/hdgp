"""후보별 gain을 [K, num_joints] 행렬로 편다.

group에 속하지 않은 관절이 0으로 남으면 actuator가 죽으므로,
반드시 config 기본값으로 전체를 먼저 채운 뒤 후보 값으로 덮어쓴다.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from r2s_autotune.config import AutotuneConfig
from r2s_autotune.sample_candidates import Candidate

_FIELDS = ("stiffness", "damping", "joint_friction")


def build_gain_matrices(
    config: AutotuneConfig,
    candidates: Sequence[Candidate],
    group_indices: Mapping[str, Sequence[int]],
    num_joints: int,
) -> dict[str, np.ndarray]:
    """Returns {"stiffness": [K, J], "damping": [K, J], "joint_friction": [K, J]}"""
    num_candidates = len(candidates)
    matrices = {
        field: np.zeros((num_candidates, num_joints), dtype=np.float64) for field in _FIELDS
    }

    # 1) config 기본값으로 전체를 채운다.
    for group_name, joint_indices in group_indices.items():
        group = config.groups[group_name]
        columns = list(joint_indices)
        for field in _FIELDS:
            matrices[field][:, columns] = getattr(group, field)

    # 2) 후보가 값을 가진 group만 덮어쓴다.
    for candidate in candidates:
        for group_name, calibration in candidate.groups.items():
            joint_indices = group_indices.get(group_name)
            if joint_indices is None:
                continue
            columns = list(joint_indices)
            for field in _FIELDS:
                matrices[field][candidate.index, columns] = getattr(calibration, field)

    return matrices
