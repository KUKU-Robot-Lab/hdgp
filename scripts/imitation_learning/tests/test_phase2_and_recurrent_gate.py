"""Tests for Phase 2 obs alignment and RecurrentGate epoch-based schedule."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import pytest

# ---------------------------------------------------------------------------
# sys.path setup (task 디렉토리 직접 import)
# ---------------------------------------------------------------------------
_HDGP_ROOT = Path(__file__).resolve().parents[3]
_TASK_DIR = (
    _HDGP_ROOT
    / "source/openarm/openarm/tasks/manager_based/openarm_manipulation"
    / "pipeline/hand/right/5g_pour_right_v4"
)
sys.path.insert(0, str(_TASK_DIR))

from real_demo_bc import (
    _compute_pour_geometry,
    _remap_hdf5_obs_to_sim_layout,
    _BINARY_CONTACT_NORM_THRESHOLD,
    SIM_OBS_DIM,
    REAL_DEMO_OBS_DIM,
)
from recurrent_gate import RecurrentGateState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_cup_mat(T: int = 8) -> torch.Tensor:
    """(T, 4, 4) identity transforms."""
    m = torch.eye(4).unsqueeze(0).expand(T, -1, -1).clone()
    return m


def _translated_cup_mat(pos: list[float], T: int = 8) -> torch.Tensor:
    m = _identity_cup_mat(T)
    m[:, :3, 3] = torch.tensor(pos, dtype=torch.float32)
    return m


def _tilted_cup_mat(angle_deg: float, T: int = 8) -> torch.Tensor:
    """컵을 X축 기준으로 angle_deg 회전 (pour 방향으로 기울임)."""
    a = math.radians(angle_deg)
    R = torch.tensor([
        [1.0,        0.0,         0.0, 0.0],
        [0.0,  math.cos(a), -math.sin(a), 0.0],
        [0.0,  math.sin(a),  math.cos(a), 0.0],
        [0.0,        0.0,         0.0, 1.0],
    ])
    return R.unsqueeze(0).expand(T, -1, -1).clone()


def _random_hdf5_obs(T: int = 16) -> torch.Tensor:
    obs = torch.zeros(T, REAL_DEMO_OBS_DIM)
    obs[:, 68:73] = torch.rand(T, 5) * 0.5  # tip_force_norm
    obs[:, 79:90] = torch.rand(T, 11) * 0.8  # prev_actions
    return obs


# ---------------------------------------------------------------------------
# RecurrentGate epoch-based schedule
# ---------------------------------------------------------------------------

class TestRecurrentGateEpochBased:
    def test_alpha_zero_before_ramp_start(self):
        gate = RecurrentGateState(ramp_start_epoch=100, ramp_epochs=400)
        alpha, _, _, _ = gate.update(epoch=99, traj_size=0.0, success_rate=0.0)
        assert alpha == pytest.approx(0.0)

    def test_alpha_zero_at_ramp_start(self):
        gate = RecurrentGateState(ramp_start_epoch=100, ramp_epochs=400)
        alpha, _, _, _ = gate.update(epoch=100, traj_size=0.0, success_rate=0.0)
        assert alpha == pytest.approx(0.0)

    def test_alpha_half_at_midpoint(self):
        gate = RecurrentGateState(ramp_start_epoch=100, ramp_epochs=400)
        alpha, _, _, _ = gate.update(epoch=300, traj_size=0.0, success_rate=0.0)
        assert alpha == pytest.approx(0.5, abs=1e-5)

    def test_alpha_one_after_ramp_end(self):
        gate = RecurrentGateState(ramp_start_epoch=100, ramp_epochs=400)
        alpha, _, _, _ = gate.update(epoch=500, traj_size=0.0, success_rate=0.0)
        assert alpha == pytest.approx(1.0)

    def test_alpha_clamped_above_one(self):
        gate = RecurrentGateState(ramp_start_epoch=0, ramp_epochs=100)
        alpha, _, _, _ = gate.update(epoch=9999, traj_size=0.0, success_rate=0.0)
        assert alpha == pytest.approx(1.0)

    def test_epoch_based_ignores_success_and_traj(self):
        """성공/궤적이 0이어도 에포크 기반이면 alpha 상승."""
        gate = RecurrentGateState(ramp_start_epoch=0, ramp_epochs=200)
        alpha, _, _, _ = gate.update(epoch=200, traj_size=0.0, success_rate=0.0)
        assert alpha == pytest.approx(1.0)

    def test_backward_compat_original_mode(self):
        """ramp_start_epoch=-1이면 원래 success+traj 기반 동작."""
        gate = RecurrentGateState(
            ramp_start_epoch=-1,
            success_threshold=0.01,
            traj_threshold=5.0,
            ramp_epochs=100,
        )
        # traj/success 부족 → alpha=0
        alpha, _, _, _ = gate.update(epoch=1000, traj_size=1.0, success_rate=0.0)
        assert alpha == pytest.approx(0.0)
        # traj/success 충분 → alpha 상승 시작
        alpha, _, _, _ = gate.update(epoch=1001, traj_size=10.0, success_rate=1.0)
        assert alpha > 0.0

    def test_success_ema_still_tracked_in_epoch_mode(self):
        """에포크 기반이어도 success_ema는 계속 업데이트 (로깅용)."""
        gate = RecurrentGateState(ramp_start_epoch=0, ramp_epochs=100, success_ema_tau=0.0)
        _, _, _, ema = gate.update(epoch=10, traj_size=0.0, success_rate=0.5)
        assert ema == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _compute_pour_geometry
# ---------------------------------------------------------------------------

class TestComputePourGeometry:
    def test_output_shapes(self):
        T = 16
        sc = _identity_cup_mat(T)
        tc = _translated_cup_mat([0.3, 0.3, 0.0], T)
        ppt, pax, uax, tsm = _compute_pour_geometry(sc, tc)
        assert ppt.shape == (T, 3)
        assert pax.shape == (T, 3)
        assert uax.shape == (T, 3)
        assert tsm.shape == (T, 8)

    def test_source_up_axis_identity(self):
        """컵이 직립(identity)이면 up_axis = world +Z."""
        sc = _identity_cup_mat(4)
        tc = _translated_cup_mat([0.3, 0.3, 0.0], 4)
        _, _, uax, _ = _compute_pour_geometry(sc, tc)
        assert torch.allclose(uax, torch.tensor([[0.0, 0.0, 1.0]]).expand(4, -1), atol=1e-5)

    def test_source_pour_axis_identity(self):
        """컵이 직립(identity)이면 pour_axis = world +X."""
        sc = _identity_cup_mat(4)
        tc = _translated_cup_mat([0.3, 0.3, 0.0], 4)
        _, pax, _, _ = _compute_pour_geometry(sc, tc)
        assert torch.allclose(pax, torch.tensor([[1.0, 0.0, 0.0]]).expand(4, -1), atol=1e-5)

    def test_transport_summary_finite(self):
        sc = _identity_cup_mat(8)
        tc = _translated_cup_mat([0.3, 0.3, 0.0], 8)
        _, _, _, tsm = _compute_pour_geometry(sc, tc)
        assert torch.isfinite(tsm).all()

    def test_mouth_distance_decreases_when_cups_closer(self):
        """컵이 가까울수록 mouth_distance가 감소."""
        sc = _identity_cup_mat(1)
        tc_far  = _translated_cup_mat([0.5, 0.5, 0.0], 1)
        tc_near = _translated_cup_mat([0.2, 0.2, 0.0], 1)
        _, _, _, tsm_far  = _compute_pour_geometry(sc, tc_far)
        _, _, _, tsm_near = _compute_pour_geometry(sc, tc_near)
        assert tsm_near[0, 0] < tsm_far[0, 0]  # mouth_distance 감소

    def test_g_ready_zero_when_far(self):
        """컵이 멀리 있으면 g_ready ≈ 0."""
        sc = _identity_cup_mat(4)
        tc = _translated_cup_mat([2.0, 2.0, 0.0], 4)
        _, _, _, tsm = _compute_pour_geometry(sc, tc)
        g_ready = tsm[:, 7]
        assert (g_ready < 0.01).all()

    def test_source_up_dot_world_between_neg1_pos1(self):
        sc = _tilted_cup_mat(45.0, 8)
        tc = _translated_cup_mat([0.3, 0.3, 0.0], 8)
        _, _, _, tsm = _compute_pour_geometry(sc, tc)
        up_dot = tsm[:, 4]  # source_up_dot_world
        assert (up_dot >= -1.0).all() and (up_dot <= 1.0).all()

    def test_directional_tilt_cos_range(self):
        sc = _identity_cup_mat(8)
        tc = _translated_cup_mat([0.3, 0.3, 0.0], 8)
        _, _, _, tsm = _compute_pour_geometry(sc, tc)
        cos_val = tsm[:, 5]
        assert (cos_val >= -1.0 - 1e-5).all() and (cos_val <= 1.0 + 1e-5).all()


# ---------------------------------------------------------------------------
# _remap_hdf5_obs_to_sim_layout  (Phase 2)
# ---------------------------------------------------------------------------

class TestRemapHdf5ObsPhase2:
    def test_output_shape_with_cup_mats(self):
        obs = _random_hdf5_obs(16)
        sc = _identity_cup_mat(16)
        tc = _translated_cup_mat([0.3, 0.3, 0.0], 16)
        out = _remap_hdf5_obs_to_sim_layout(obs, sc, tc)
        assert out.shape == (16, SIM_OBS_DIM)

    def test_pour_geometry_nonzero_with_cup_mats(self):
        """컵 포즈가 있으면 [68:85]이 0이 아님."""
        obs = _random_hdf5_obs(8)
        sc = _identity_cup_mat(8)
        tc = _translated_cup_mat([0.3, 0.3, 0.0], 8)
        out = _remap_hdf5_obs_to_sim_layout(obs, sc, tc)
        assert not torch.allclose(out[:, 68:85], torch.zeros(8, 17))

    def test_pour_geometry_zero_without_cup_mats(self):
        """컵 포즈 없으면 [68:85] = 0 (하위 호환)."""
        obs = _random_hdf5_obs(8)
        out = _remap_hdf5_obs_to_sim_layout(obs)
        assert torch.allclose(out[:, 68:85], torch.zeros(8, 17))

    def test_binary_contact_from_tip_force(self):
        obs = _random_hdf5_obs(10)
        obs[:5, 68:73] = 0.001   # below threshold
        obs[5:, 68:73] = 0.5     # above threshold
        out = _remap_hdf5_obs_to_sim_layout(obs)
        assert torch.all(out[:5, 85:90] == 0.0)
        assert torch.all(out[5:, 85:90] == 1.0)

    def test_tip_force_norm_preserved(self):
        obs = _random_hdf5_obs(8)
        out = _remap_hdf5_obs_to_sim_layout(obs)
        assert torch.allclose(out[:, 90:95], obs[:, 68:73])

    def test_last_actions_preserved(self):
        obs = _random_hdf5_obs(8)
        out = _remap_hdf5_obs_to_sim_layout(obs)
        assert torch.allclose(out[:, 95:106], obs[:, 79:90])

    def test_bead_in_source_fraction_one(self):
        obs = _random_hdf5_obs(8)
        out = _remap_hdf5_obs_to_sim_layout(obs)
        assert torch.all(out[:, 106] == 1.0)

    def test_bead_target_cross_spill_zero(self):
        obs = _random_hdf5_obs(8)
        sc = _identity_cup_mat(8)
        tc = _translated_cup_mat([0.3, 0.3, 0.0], 8)
        out = _remap_hdf5_obs_to_sim_layout(obs, sc, tc)
        assert torch.all(out[:, 107:110] == 0.0)

    def test_binary_contact_threshold_value(self):
        """임계값 경계 테스트 (0.01 = CONTACT_FORCE_THRESHOLD / CONTACT_FORCE_MAX)."""
        assert _BINARY_CONTACT_NORM_THRESHOLD == pytest.approx(0.01)
        obs = _random_hdf5_obs(4)
        obs[:, 68:73] = _BINARY_CONTACT_NORM_THRESHOLD + 1e-5
        out = _remap_hdf5_obs_to_sim_layout(obs)
        assert torch.all(out[:, 85:90] == 1.0)

    def test_pour_geometry_consistency_with_cup_local_basis(self):
        """_compute_pour_geometry와 _build_cup_local_basis가 같은 컵 정보를 사용."""
        from real_demo_bc import _build_cup_local_basis, _quat_xyzw_from_matrix
        T = 4
        sc = _identity_cup_mat(T)
        tc = _translated_cup_mat([0.3, 0.3, 0.0], T)
        ppt, pax, uax, _ = _compute_pour_geometry(sc, tc)

        # up_axis는 basis의 spin_axis와 동일해야 함
        spin_ax, _, _ = _build_cup_local_basis(sc, tc)
        assert torch.allclose(uax, spin_ax, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
