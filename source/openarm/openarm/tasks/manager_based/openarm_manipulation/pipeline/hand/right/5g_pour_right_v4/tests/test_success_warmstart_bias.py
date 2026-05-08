"""Static checks for V4 trajectory-buffer-biased pouring configuration.

These tests stay Isaac-Sim free and verify that V4 preserves its LSTM+BC
trajectory capture path instead of the removed success-cache reset path.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch


_TASK_DIR = Path(__file__).parent.parent
_CFG_TEXT = (_TASK_DIR / "pour_right_env_cfg.py").read_text()


def _parse_value(name: str, cast: type[float] = float):
    match = re.search(rf"^\s*{name}\s*:\s*(?:float|int|bool)\s*=\s*([^\n#]+)", _CFG_TEXT, re.MULTILINE)
    assert match is not None, f"{name} not found"
    raw = match.group(1).strip()
    if cast is bool:
        return raw == "True"
    return cast(raw)


def _trajectory_score(bead_in_target: torch.Tensor, spill_ratio: torch.Tensor, active_bead_count: torch.Tensor) -> torch.Tensor:
    return (bead_in_target - 0.5 * spill_ratio) * active_bead_count


class TestSuccessWarmstartConfig:
    def test_trajectory_capture_is_enabled_for_bc(self):
        assert _parse_value("enable_trajectory_capture", bool) is True
        assert _parse_value("trajectory_capture_window", int) == 200
        assert _parse_value("trajectory_buffer_capacity", int) == 256
        assert _parse_value("trajectory_min_steps", int) == 60

    def test_trajectory_success_thresholds_match_v4_policy(self):
        assert _parse_value("trajectory_success_bead_threshold") == pytest.approx(0.50)
        assert _parse_value("trajectory_success_spill_max") == pytest.approx(0.10)
        assert _parse_value("bc_min_buffer_size", int) == 20
        assert _parse_value("grasp_warmstart_reset_ratio") == pytest.approx(1.0)

    def test_reward_weights_preserve_v4_pour_capture_bias(self):
        assert _parse_value("weight_transport_progress") == pytest.approx(12.0)
        assert _parse_value("weight_prepour_dir") == pytest.approx(2.5)
        assert _parse_value("weight_prepour_align") == pytest.approx(2.0)
        assert _parse_value("weight_cross") == pytest.approx(80.0)
        assert _parse_value("weight_capture") == pytest.approx(160.0)
        assert _parse_value("weight_first_capture_bonus") == pytest.approx(40.0)
        assert _parse_value("weight_terminal_pour") == pytest.approx(60.0)
        assert _parse_value("weight_terminal_capture") == pytest.approx(200.0)
        assert _parse_value("weight_success_per_bead") == pytest.approx(20.0)

    def test_tilt_and_gate_match_current_v4_neighborhood(self):
        assert _parse_value("tilt_action_gate_xy_near") == pytest.approx(0.04)
        assert _parse_value("tilt_action_gate_xy_far") == pytest.approx(0.20)
        assert _parse_value("tilt_onset_dist_threshold") == pytest.approx(0.08)
        assert _parse_value("reward_gate_xy_scale") == pytest.approx(5.0)
        assert _parse_value("reward_approach_xy_scale") == pytest.approx(6.0)
        assert _parse_value("weight_premature_tilt") == pytest.approx(1.5)


class TestTrajectoryScore:
    def test_capture_dominates_low_spill_with_same_bead_count(self):
        score = _trajectory_score(
            bead_in_target=torch.tensor([0.6, 0.5]),
            spill_ratio=torch.tensor([0.1, 0.0]),
            active_bead_count=torch.tensor([5.0, 5.0]),
        )
        assert float(score[0]) > float(score[1])

    def test_later_adr_stage_wins_when_fraction_quality_matches(self):
        score = _trajectory_score(
            bead_in_target=torch.tensor([0.6, 0.6]),
            spill_ratio=torch.tensor([0.1, 0.1]),
            active_bead_count=torch.tensor([1.0, 5.0]),
        )
        assert float(score[1]) > float(score[0])
