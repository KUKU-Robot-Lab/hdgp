from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "grasp_left_utils.py"
SPEC = importlib.util.spec_from_file_location("grasp_left_utils", MODULE_PATH)
grasp_left_utils = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = grasp_left_utils
SPEC.loader.exec_module(grasp_left_utils)

compute_joint7_lift_wait_target = grasp_left_utils.compute_joint7_lift_wait_target


def test_compute_joint7_lift_wait_target_only_changes_joint7() -> None:
    actual_arm = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 1.4],
        ]
    )

    target = compute_joint7_lift_wait_target(
        actual_arm,
        joint7_delta=0.31,
        joint7_min=0.2,
        joint7_max=1.5,
    )

    assert torch.allclose(target[:, :6], actual_arm[:, :6])
    assert torch.allclose(target[:, 6], torch.tensor([1.01, 1.5]))
