"""Static checks for success-warmstart-biased pouring configuration.

These tests stay Isaac-Sim free and verify that v3 is intentionally biased
toward replaying successful pour states rather than fresh exploration.
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


def _success_cache_score(bead_in_target: torch.Tensor, bead_cross: torch.Tensor, mouth_xy: torch.Tensor) -> torch.Tensor:
    return 1000.0 * bead_in_target + 100.0 * bead_cross - mouth_xy


class TestSuccessWarmstartConfig:
    def test_success_warmstart_bias_is_enabled(self):
        assert _parse_value("enable_success_warmstart_reset", bool) is True
        assert _parse_value("success_warmstart_cache_size", int) == 512
        assert _parse_value("success_reset_hold_steps", int) == 20
        assert _parse_value("success_reset_palm_delta_xyz") == pytest.approx(0.05)

    def test_reset_mix_prefers_success_cache(self):
        success_ratio = _parse_value("success_cache_reset_ratio")
        grasp_ratio = _parse_value("grasp_warmstart_reset_ratio")
        assert success_ratio == pytest.approx(0.70)
        assert grasp_ratio == pytest.approx(0.25)
        assert success_ratio + grasp_ratio == pytest.approx(0.95)

    def test_reward_weights_bias_toward_replaying_successful_pours(self):
        assert _parse_value("weight_transport_progress") == pytest.approx(24.0)
        assert _parse_value("weight_prepour_dir") == pytest.approx(1.5)
        assert _parse_value("weight_prepour_align") == pytest.approx(1.0)
        assert _parse_value("weight_cross") == pytest.approx(80.0)
        assert _parse_value("weight_capture") == pytest.approx(160.0)
        assert _parse_value("weight_first_capture_bonus") == pytest.approx(40.0)
        assert _parse_value("weight_terminal_pour") == pytest.approx(60.0)
        assert _parse_value("weight_success_per_bead") == pytest.approx(20.0)

    def test_tilt_and_gate_are_tightened_around_success_neighborhood(self):
        assert _parse_value("tilt_action_gate_xy_near") == pytest.approx(0.04)
        assert _parse_value("tilt_action_gate_xy_far") == pytest.approx(0.12)
        assert _parse_value("tilt_onset_dist_threshold") == pytest.approx(0.08)
        assert _parse_value("reward_gate_xy_scale") == pytest.approx(10.0)
        assert _parse_value("reward_approach_xy_scale") == pytest.approx(6.0)
        assert _parse_value("weight_premature_tilt") == pytest.approx(1.5)


class TestSuccessCacheScore:
    def test_capture_dominates_cross_and_distance(self):
        score = _success_cache_score(
            bead_in_target=torch.tensor([0.05, 0.0]),
            bead_cross=torch.tensor([0.0, 0.4]),
            mouth_xy=torch.tensor([0.08, 0.01]),
        )
        assert float(score[0]) > float(score[1])

    def test_closer_state_wins_when_capture_and_cross_match(self):
        score = _success_cache_score(
            bead_in_target=torch.tensor([0.1, 0.1]),
            bead_cross=torch.tensor([0.2, 0.2]),
            mouth_xy=torch.tensor([0.03, 0.08]),
        )
        assert float(score[0]) > float(score[1])
