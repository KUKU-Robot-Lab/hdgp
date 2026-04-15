"""TDD: test3 reward gate tuning and terminal pour bonus verification.

This test stays Isaac-Sim free and verifies:
1. Config thresholds were tightened as planned.
2. The narrowed approach gate keeps gradient alive until 6 cm.
3. Pour gate now requires mouth-to-mouth alignment plus >90 deg tilt.
4. First capture / success are also guarded by the same pour pose.
5. Terminal pour bonus exists and only activates on the last step in pour pose.
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


def _approach_gate(mouth_xy_dist: torch.Tensor, near: float, far: float) -> torch.Tensor:
    den = max(far - near, 1e-6)
    return torch.clamp((mouth_xy_dist - near) / den, min=0.0, max=1.0)


def _dynamic_rim_point(mouth_center: torch.Tensor, cup_up: torch.Tensor, fallback_axis: torch.Tensor, radius: float) -> torch.Tensor:
    gravity_down = torch.tensor([0.0, 0.0, -1.0], dtype=mouth_center.dtype).expand_as(cup_up)
    downhill = gravity_down - (gravity_down * cup_up).sum(dim=-1, keepdim=True) * cup_up
    downhill_norm = downhill.norm(dim=-1, keepdim=True)
    fallback_unit = fallback_axis / fallback_axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    downhill_unit = torch.where(downhill_norm > 1e-6, downhill / downhill_norm.clamp(min=1e-6), fallback_unit)
    return mouth_center + radius * downhill_unit


def _gate_pour_binary(
    mouth_xy_dist: torch.Tensor,
    mouth_z_clearance: torch.Tensor,
    source_up_dot: torch.Tensor,
    xy_thresh: float,
    z_min: float,
    z_max: float,
    tilt_thresh: float,
) -> torch.Tensor:
    return (
        (mouth_xy_dist < xy_thresh)
        & (mouth_z_clearance > z_min)
        & (mouth_z_clearance < z_max)
        & (source_up_dot < tilt_thresh)
    ).float()


def _terminal_pour_bonus(
    episode_length_buf: torch.Tensor,
    max_episode_length: int,
    mouth_xy_dist: torch.Tensor,
    mouth_z_clearance: torch.Tensor,
    source_up_dot: torch.Tensor,
    xy_thresh: float,
    z_min: float,
    z_max: float,
    tilt_thresh: float,
    weight: float,
) -> torch.Tensor:
    is_last_step = episode_length_buf >= (max_episode_length - 1)
    in_pour_pose = (
        (mouth_xy_dist < xy_thresh)
        & (mouth_z_clearance > z_min)
        & (mouth_z_clearance < z_max)
        & (source_up_dot < tilt_thresh)
    )
    return weight * is_last_step.float() * in_pour_pose.float()


def _first_capture_bonus(
    gate_pour_binary: torch.Tensor,
    bead_in_target_fraction: torch.Tensor,
    bonus_paid: torch.Tensor,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_capture = gate_pour_binary.bool() & (bead_in_target_fraction > 0.0) & (~bonus_paid)
    return weight * first_capture.float(), (bonus_paid | first_capture)


def _premature_tilt_cost(g_ready: torch.Tensor, source_up_dot: torch.Tensor, tilt_onset_dot_threshold: float) -> torch.Tensor:
    significant_tilt = torch.clamp(tilt_onset_dot_threshold - source_up_dot, min=0.0)
    return (1.0 - g_ready) * significant_tilt


def _success_by_pose_and_fill(
    gate_pour_binary: torch.Tensor,
    bead_in_target_fraction: torch.Tensor,
    spill_ratio: torch.Tensor,
    success_fill_ratio: float,
    success_spill_max: float,
) -> torch.Tensor:
    return gate_pour_binary.bool() & (bead_in_target_fraction >= success_fill_ratio) & (spill_ratio <= success_spill_max)


def _tilt_action_gate(mouth_xy_dist: torch.Tensor, near: float, far: float) -> torch.Tensor:
    den = max(far - near, 1e-6)
    return torch.clamp((far - mouth_xy_dist) / den, min=0.0, max=1.0)


class TestRewardGateConfig:
    def test_phase_a_pour_gate_threshold(self):
        # test1 분석: policy plateau 11cm에서 0.03m gate 도달 불가 → 0.08m으로 완화
        assert _parse_float_constant("pour_binary_mouth_xy_thresh", _CFG_TEXT) == pytest.approx(0.08)
        assert _parse_float_constant("pour_binary_mouth_z_min", _CFG_TEXT) == pytest.approx(-0.01)
        assert _parse_float_constant("pour_binary_mouth_z_max", _CFG_TEXT) == pytest.approx(0.03)
        assert _parse_float_constant("pour_binary_tilt_thresh", _CFG_TEXT) == pytest.approx(0.0)

    def test_phase_b_approach_dead_zone_is_shrunk(self):
        # approach_xy_off_far=0.03: pour gate threshold(0.08)보다 작아 gate 안쪽에서만 annealing
        assert _parse_float_constant("approach_xy_off_near", _CFG_TEXT) == pytest.approx(0.02)
        assert _parse_float_constant("approach_xy_off_far", _CFG_TEXT) == pytest.approx(0.03)

    def test_phase_c_tilt_onset_bridge_is_stronger(self):
        assert _parse_float_constant("tilt_onset_dot_threshold", _CFG_TEXT) == pytest.approx(0.17)
        assert _parse_float_constant("weight_tilt_onset_bonus", _CFG_TEXT) == pytest.approx(10.0)

    def test_phase_d_terminal_pour_params_exist(self):
        assert _parse_float_constant("weight_terminal_pour", _CFG_TEXT) == pytest.approx(30.0)
        assert _parse_float_constant("terminal_pour_tilt_thresh", _CFG_TEXT) == pytest.approx(0.0)
        assert _parse_float_constant("terminal_pour_mouth_xy_thresh", _CFG_TEXT) == pytest.approx(0.03)
        assert _parse_float_constant("terminal_pour_mouth_z_min", _CFG_TEXT) == pytest.approx(-0.01)
        assert _parse_float_constant("terminal_pour_mouth_z_max", _CFG_TEXT) == pytest.approx(0.03)
        assert _parse_float_constant("weight_spill", _CFG_TEXT) == pytest.approx(3.0)

    def test_strict_success_metrics_exist(self):
        assert _parse_float_constant("final_success_target_fill_ratio", _CFG_TEXT) == pytest.approx(0.95)
        assert _parse_float_constant("final_success_spill_max", _CFG_TEXT) == pytest.approx(0.05)


class TestRewardGateBehavior:
    def test_approach_gate_keeps_gradient_until_six_cm(self):
        dists = torch.tensor([0.015, 0.03, 0.05, 0.07])
        gate = _approach_gate(dists, near=0.02, far=0.06)
        assert gate.tolist() == pytest.approx([0.0, 0.25, 0.75, 1.0])

    def test_dynamic_rim_point_moves_to_downhill_edge_when_tilted(self):
        mouth_center = torch.tensor([[0.0, 0.0, 0.1]])
        cup_up = torch.tensor([[1.0, 0.0, 0.0]])
        fallback_axis = torch.tensor([[1.0, 0.0, 0.0]])
        point = _dynamic_rim_point(mouth_center, cup_up, fallback_axis, radius=0.041)
        assert point[0].tolist() == pytest.approx([0.0, 0.0, 0.059], abs=1e-3)

    def test_dynamic_rim_point_falls_back_when_upright(self):
        mouth_center = torch.tensor([[0.0, 0.0, 0.1]])
        cup_up = torch.tensor([[0.0, 0.0, 1.0]])
        fallback_axis = torch.tensor([[1.0, 0.0, 0.0]])
        point = _dynamic_rim_point(mouth_center, cup_up, fallback_axis, radius=0.041)
        assert point[0].tolist() == pytest.approx([0.041, 0.0, 0.1], abs=1e-6)

    def test_pour_gate_requires_mouth_alignment_and_over_ninety_deg_tilt(self):
        # xy_thresh=0.08 (test1 완화값): 8cm 이내 + z 범위 + >90° tilt 동시 충족
        mouth_xy = torch.tensor([0.07, 0.07, 0.09, 0.07, 0.07])
        mouth_z = torch.tensor([0.02, 0.05, 0.02, -0.02, 0.02])
        tilt = torch.tensor([-0.05, -0.05, -0.05, -0.05, 0.10])
        gate = _gate_pour_binary(
            mouth_xy,
            mouth_z,
            tilt,
            xy_thresh=0.08,
            z_min=-0.01,
            z_max=0.03,
            tilt_thresh=0.0,
        )
        assert gate.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])

    def test_terminal_bonus_only_pays_on_last_step_in_pour_pose(self):
        bonus = _terminal_pour_bonus(
            episode_length_buf=torch.tensor([498, 499, 499, 499]),
            max_episode_length=500,
            mouth_xy_dist=torch.tensor([0.02, 0.02, 0.04, 0.02]),
            mouth_z_clearance=torch.tensor([0.02, 0.02, 0.02, 0.05]),
            source_up_dot=torch.tensor([-0.1, -0.1, -0.1, -0.1]),
            xy_thresh=0.03,
            z_min=-0.01,
            z_max=0.03,
            tilt_thresh=0.0,
            weight=30.0,
        )
        assert bonus.tolist() == pytest.approx([0.0, 30.0, 0.0, 0.0])

    def test_first_capture_bonus_only_latches_inside_pour_gate(self):
        reward, bonus_paid = _first_capture_bonus(
            gate_pour_binary=torch.tensor([0.0, 1.0, 1.0]),
            bead_in_target_fraction=torch.tensor([0.2, 0.2, 0.2]),
            bonus_paid=torch.tensor([False, False, True]),
            weight=20.0,
        )
        assert reward.tolist() == pytest.approx([0.0, 20.0, 0.0])
        assert bonus_paid.tolist() == [False, True, True]

    def test_success_requires_valid_pour_pose_and_fill(self):
        success = _success_by_pose_and_fill(
            gate_pour_binary=torch.tensor([0.0, 1.0, 1.0]),
            bead_in_target_fraction=torch.tensor([0.8, 0.8, 0.4]),
            spill_ratio=torch.tensor([0.0, 0.1, 0.1]),
            success_fill_ratio=0.5,
            success_spill_max=0.2,
        )
        assert success.tolist() == [False, True, False]

    def test_premature_tilt_cost_penalizes_far_tilt_not_upright_pose(self):
        cost = _premature_tilt_cost(
            g_ready=torch.tensor([0.1, 0.1, 0.9]),
            source_up_dot=torch.tensor([-0.2, 1.0, -0.2]),
            tilt_onset_dot_threshold=0.17,
        )
        assert cost.tolist() == pytest.approx([0.333, 0.0, 0.037], abs=1e-3)

    def test_tilt_action_gate_is_based_on_mouth_distance(self):
        gate = _tilt_action_gate(torch.tensor([0.30, 0.20, 0.06, 0.03]), near=0.06, far=0.25)
        assert gate.tolist() == pytest.approx([0.0, 0.2631579, 1.0, 1.0], abs=1e-6)


class TestRewardImplementationHooks:
    def test_env_computes_terminal_pour_reward(self):
        assert "r_terminal_pour" in _ENV_TEXT
        assert "weight_terminal_pour" in _ENV_TEXT
        assert "terminal_pour_mouth_xy_thresh" in _ENV_TEXT
        assert "terminal_pour_mouth_z_min" in _ENV_TEXT
        assert "terminal_pour_mouth_z_max" in _ENV_TEXT
        assert "terminal_pour_tilt_thresh" in _ENV_TEXT

    def test_env_logs_terminal_pour_metric(self):
        assert 'self.extras["r_terminal_pour"]' in _ENV_TEXT
        assert 'self.extras["final_success_strict"]' in _ENV_TEXT

    def test_env_uses_dynamic_rim_point_and_mouth_based_shaping(self):
        assert "def _compute_dynamic_source_pour_point_w" in _ENV_TEXT
        assert "self._source_pour_point_w = self._compute_dynamic_source_pour_point_w" in _ENV_TEXT
        assert "self._g_align_xy = torch.exp(-self.cfg.reward_gate_xy_scale * self._mouth_xy_distance)" in _ENV_TEXT
        assert "r_approach_xy = 1.0 - torch.tanh(self.cfg.reward_approach_xy_scale * self._mouth_xy_distance)" in _ENV_TEXT
        assert "(self._mouth_xy_distance < self.cfg.tilt_onset_dist_threshold)" in _ENV_TEXT
        assert "(self.cfg.tilt_action_gate_xy_far - self._mouth_xy_distance)" in _ENV_TEXT

    def test_env_wires_success_and_first_capture_to_pose_gate(self):
        assert "first_capture = gate_pour_binary.bool() &" in _ENV_TEXT
        assert "success_by_pose_and_fill = gate_pour_binary.bool() & success_now" in _ENV_TEXT
        assert "self.success_flag.copy_(success_by_pose_and_fill)" in _ENV_TEXT

    def test_env_uses_far_tilt_penalty_not_upright_penalty(self):
        assert "significant_tilt = torch.clamp(" in _ENV_TEXT
        assert "self.cfg.tilt_onset_dot_threshold - self._source_up_dot_world" in _ENV_TEXT
