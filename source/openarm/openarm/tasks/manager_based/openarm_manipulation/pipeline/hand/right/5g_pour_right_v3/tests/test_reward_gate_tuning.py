"""TDD: test3 reward gate tuning and terminal pour bonus verification.

This test stays Isaac-Sim free and verifies:
1. Config thresholds were tightened as planned.
2. The narrowed approach gate keeps gradient alive until 6 cm.
3. Pour gate now requires both close XY and >90 deg tilt.
4. Terminal pour bonus exists and only activates on the last step in pour pose.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch


_TASK_DIR = Path(__file__).parent.parent
_CFG_TEXT = (_TASK_DIR / "pour_right_env_cfg.py").read_text()
_ENV_TEXT = (_TASK_DIR / "pour_right_env.py").read_text()


def _parse_float_constant(name: str, text: str) -> float:
    match = re.search(rf"^\s*{name}\s*:\s*float\s*=\s*([-+]?\d+(?:\.\d+)?)", text, re.MULTILINE)
    assert match is not None, f"{name} not found"
    return float(match.group(1))


def _approach_gate(cup_center_xy_dist: torch.Tensor, near: float, far: float) -> torch.Tensor:
    den = max(far - near, 1e-6)
    return torch.clamp((cup_center_xy_dist - near) / den, min=0.0, max=1.0)


def _gate_pour_binary(cup_center_xy_dist: torch.Tensor, source_up_dot: torch.Tensor, xy_thresh: float, tilt_thresh: float) -> torch.Tensor:
    return ((cup_center_xy_dist < xy_thresh) & (source_up_dot < tilt_thresh)).float()


def _terminal_pour_bonus(
    episode_length_buf: torch.Tensor,
    max_episode_length: int,
    cup_center_xy_dist: torch.Tensor,
    source_up_dot: torch.Tensor,
    xy_thresh: float,
    tilt_thresh: float,
    weight: float,
) -> torch.Tensor:
    is_last_step = episode_length_buf >= (max_episode_length - 1)
    in_pour_pose = (cup_center_xy_dist < xy_thresh) & (source_up_dot < tilt_thresh)
    return weight * is_last_step.float() * in_pour_pose.float()


class TestRewardGateConfig:
    def test_phase_a_pour_gate_is_tightened(self):
        assert _parse_float_constant("pour_binary_xy_thresh", _CFG_TEXT) == pytest.approx(0.10)
        assert _parse_float_constant("pour_binary_tilt_thresh", _CFG_TEXT) == pytest.approx(0.0)

    def test_phase_b_approach_dead_zone_is_shrunk(self):
        assert _parse_float_constant("approach_xy_off_near", _CFG_TEXT) == pytest.approx(0.02)
        assert _parse_float_constant("approach_xy_off_far", _CFG_TEXT) == pytest.approx(0.06)

    def test_phase_c_tilt_onset_bridge_is_stronger(self):
        assert _parse_float_constant("tilt_onset_dot_threshold", _CFG_TEXT) == pytest.approx(0.17)
        assert _parse_float_constant("weight_tilt_onset_bonus", _CFG_TEXT) == pytest.approx(10.0)

    def test_phase_d_terminal_pour_params_exist(self):
        assert _parse_float_constant("weight_terminal_pour", _CFG_TEXT) == pytest.approx(30.0)
        assert _parse_float_constant("terminal_pour_tilt_thresh", _CFG_TEXT) == pytest.approx(0.0)
        assert _parse_float_constant("terminal_pour_xy_thresh", _CFG_TEXT) == pytest.approx(0.12)


class TestRewardGateBehavior:
    def test_approach_gate_keeps_gradient_until_six_cm(self):
        dists = torch.tensor([0.015, 0.03, 0.05, 0.07])
        gate = _approach_gate(dists, near=0.02, far=0.06)
        assert gate.tolist() == pytest.approx([0.0, 0.25, 0.75, 1.0])

    def test_pour_gate_requires_closeness_and_over_ninety_deg_tilt(self):
        xy = torch.tensor([0.09, 0.09, 0.11])
        tilt = torch.tensor([0.10, -0.05, -0.05])
        gate = _gate_pour_binary(xy, tilt, xy_thresh=0.10, tilt_thresh=0.0)
        assert gate.tolist() == pytest.approx([0.0, 1.0, 0.0])

    def test_terminal_bonus_only_pays_on_last_step_in_pour_pose(self):
        bonus = _terminal_pour_bonus(
            episode_length_buf=torch.tensor([498, 499, 499]),
            max_episode_length=500,
            cup_center_xy_dist=torch.tensor([0.08, 0.08, 0.13]),
            source_up_dot=torch.tensor([-0.1, -0.1, -0.1]),
            xy_thresh=0.12,
            tilt_thresh=0.0,
            weight=30.0,
        )
        assert bonus.tolist() == pytest.approx([0.0, 30.0, 0.0])


class TestRewardImplementationHooks:
    def test_env_computes_terminal_pour_reward(self):
        assert "r_terminal_pour" in _ENV_TEXT
        assert "weight_terminal_pour" in _ENV_TEXT
        assert "terminal_pour_xy_thresh" in _ENV_TEXT
        assert "terminal_pour_tilt_thresh" in _ENV_TEXT

    def test_env_logs_terminal_pour_metric(self):
        assert 'self.extras["r_terminal_pour"]' in _ENV_TEXT
