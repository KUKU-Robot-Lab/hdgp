"""Unit tests for Phase-1 cup-local rotation label re-mapping.

Tests verify that:
1. _build_cup_local_basis produces an orthonormal basis matching env semantics.
2. Projecting a world-frame rotvec onto the basis and reconstructing via
   the env's formula recovers the original rotvec (roundtrip).
3. target_pose_to_pose_action_cup_local produces valid shape/range and
   zero action for zero delta.

Run with:
    cd /home/user/rl_ws/hdgp
    python -m pytest scripts/imitation_learning/tests/test_real_demo_bc_cuplocal.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# Mirror the sys.path setup used by train_pour_real_demo_bc.py.
_HDGP_ROOT = Path(__file__).resolve().parents[3]
_TASK_DIR = (
    _HDGP_ROOT
    / "source/openarm/openarm/tasks/manager_based/openarm_manipulation"
    / "pipeline/hand/right/5g_pour_right_v4"
)
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from real_demo_bc import (  # noqa: E402
    _build_cup_local_basis,
    _default_delta_bounds,
    _quat_apply_xyzw,
    _quat_xyzw_from_matrix,
    target_pose_to_pose_action_cup_local,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_4x4(T: int = 1) -> torch.Tensor:
    return torch.eye(4).unsqueeze(0).expand(T, -1, -1).clone().float()


def _homogeneous(pos: list[float], rotvec: list[float], T: int = 1) -> torch.Tensor:
    """Build a (T, 4, 4) homogeneous transform from position + axis-angle."""
    rv = torch.tensor(rotvec, dtype=torch.float32)
    angle = rv.norm()
    if angle < 1e-8:
        R = torch.eye(3, dtype=torch.float32)
    else:
        axis = rv / angle
        K = torch.tensor([
            [0.0,      -axis[2],  axis[1]],
            [axis[2],   0.0,     -axis[0]],
            [-axis[1],  axis[0],  0.0],
        ])
        R = torch.eye(3) + math.sin(float(angle)) * K + (1 - math.cos(float(angle))) * (K @ K)
    mat = torch.eye(4, dtype=torch.float32)
    mat[:3, :3] = R
    mat[:3, 3]  = torch.tensor(pos, dtype=torch.float32)
    return mat.unsqueeze(0).expand(T, -1, -1).clone()


def _env_reconstruct_rotvec(
    delta_local: torch.Tensor,
    source_cup_mat: torch.Tensor,
    target_cup_mat: torch.Tensor,
) -> torch.Tensor:
    """Mirror env._build_cup_local_tilt_rotvec using only BC-side helpers."""
    spin_ax, tilt_ax, ortho_ax = _build_cup_local_basis(source_cup_mat, target_cup_mat)
    return (
        delta_local[:, 0:1] * spin_ax
        + delta_local[:, 1:2] * tilt_ax
        + delta_local[:, 2:3] * ortho_ax
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCupLocalBasis:
    def test_orthonormality_upright_cup(self):
        src = _identity_4x4()
        tgt = _homogeneous([0.3, -0.2, 0.1], [0.0, 0.0, 0.0])
        s, tt, to = _build_cup_local_basis(src, tgt)

        assert s.shape == (1, 3)
        assert torch.allclose(s.norm(dim=-1), torch.ones(1), atol=1e-5), "spin_axis not unit"
        assert torch.allclose(tt.norm(dim=-1), torch.ones(1), atol=1e-5), "tilt_toward not unit"
        assert torch.allclose(to.norm(dim=-1), torch.ones(1), atol=1e-5), "tilt_ortho not unit"
        assert abs((s * tt).sum().item()) < 1e-5, "spin ⊥ tilt_toward failed"
        assert abs((s * to).sum().item()) < 1e-5, "spin ⊥ tilt_ortho failed"
        assert abs((tt * to).sum().item()) < 1e-5, "tilt_toward ⊥ tilt_ortho failed"

    def test_orthonormality_tilted_cup(self):
        src = _homogeneous([0.5, -0.1, 0.3], [math.pi / 2, 0.0, 0.0])
        tgt = _homogeneous([0.3, -0.2, 0.1], [0.0, 0.0, 0.0])
        s, tt, to = _build_cup_local_basis(src, tgt)

        assert torch.allclose(s.norm(dim=-1), torch.ones(1), atol=1e-5)
        assert torch.allclose(tt.norm(dim=-1), torch.ones(1), atol=1e-5)
        assert torch.allclose(to.norm(dim=-1), torch.ones(1), atol=1e-5)
        assert abs((s * tt).sum().item()) < 1e-5
        assert abs((s * to).sum().item()) < 1e-5
        assert abs((tt * to).sum().item()) < 1e-5

    def test_spin_axis_equals_cup_up_axis(self):
        """Upright cup: spin_axis must equal world +Z (cup local +Z)."""
        src = _homogeneous([0.0, 0.0, 0.0], [0.0, 0.0, math.pi / 4])
        tgt = _identity_4x4()
        s, _, _ = _build_cup_local_basis(src, tgt)
        assert torch.allclose(s, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-5)

    def test_batch_independent(self):
        T = 8
        src = _identity_4x4(T)
        tgt = _identity_4x4(T)
        s, tt, to = _build_cup_local_basis(src, tgt)
        assert torch.allclose(s[0], s[-1], atol=1e-6)
        assert torch.allclose(tt[0], tt[-1], atol=1e-6)


class TestRotvecRoundtrip:
    """Project world-frame rotvec → local → reconstruct → must match original."""

    @pytest.mark.parametrize("rotvec", [
        [0.3, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.7],
        [0.2, -0.3, 0.4],
        [0.0, 0.0, 0.0],
    ])
    def test_upright_cup(self, rotvec):
        src = _identity_4x4()
        tgt = _homogeneous([0.3, -0.2, 0.1], [0.0, 0.0, 0.0])
        rv = torch.tensor([rotvec], dtype=torch.float32)

        spin_ax, tilt_ax, ortho_ax = _build_cup_local_basis(src, tgt)
        spin        = (rv * spin_ax).sum(-1, keepdim=True)
        tilt_toward = (rv * tilt_ax).sum(-1, keepdim=True)
        tilt_ortho  = (rv * ortho_ax).sum(-1, keepdim=True)
        local = torch.cat([spin, tilt_toward, tilt_ortho], dim=-1)

        rv_rec = _env_reconstruct_rotvec(local, src, tgt)
        assert torch.allclose(rv_rec, rv, atol=1e-5), \
            f"rotvec={rotvec}: got {rv_rec.tolist()}"

    @pytest.mark.parametrize("cup_rotvec", [
        [0.0, 0.0, 0.0],
        [math.pi / 4, 0.0, 0.0],
        [0.0, math.pi / 3, 0.0],
    ])
    def test_tilted_cup(self, cup_rotvec):
        src = _homogeneous([0.5, -0.1, 0.3], cup_rotvec)
        tgt = _homogeneous([0.3, -0.2, 0.1], [0.0, 0.0, 0.0])
        rv = torch.tensor([[0.15, -0.2, 0.3]], dtype=torch.float32)

        spin_ax, tilt_ax, ortho_ax = _build_cup_local_basis(src, tgt)
        spin        = (rv * spin_ax).sum(-1, keepdim=True)
        tilt_toward = (rv * tilt_ax).sum(-1, keepdim=True)
        tilt_ortho  = (rv * ortho_ax).sum(-1, keepdim=True)
        local = torch.cat([spin, tilt_toward, tilt_ortho], dim=-1)

        rv_rec = _env_reconstruct_rotvec(local, src, tgt)
        assert torch.allclose(rv_rec, rv, atol=1e-5), \
            f"cup_rotvec={cup_rotvec}: got {rv_rec.tolist()}"


class TestPoseActionCupLocal:
    def _base_pose(self, T: int = 1) -> torch.Tensor:
        return torch.tensor([[0.5, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0]]).expand(T, -1).clone()

    def test_output_shape_and_range(self):
        T = 16
        src = _identity_4x4(T)
        tgt = _homogeneous([0.3, -0.2, 0.1], [0.0, 0.0, 0.0], T)
        base = self._base_pose(T)
        target = base.clone()
        target[:, :3] += 0.05 * torch.randn(T, 3)

        d_min, d_max = _default_delta_bounds()
        action = target_pose_to_pose_action_cup_local(target, base, src, tgt, d_min, d_max)

        assert action.shape == (T, 6)
        assert action.abs().max().item() <= 1.0 + 1e-6

    def test_zero_delta_gives_zero_action(self):
        """target == base → action must be all zeros (zero delta, zero normalized action)."""
        T = 4
        src = _identity_4x4(T)
        tgt = _homogeneous([0.3, -0.2, 0.1], [0.0, 0.0, 0.0], T)
        base = self._base_pose(T)

        d_min, d_max = _default_delta_bounds()
        action = target_pose_to_pose_action_cup_local(base, base, src, tgt, d_min, d_max)
        assert torch.allclose(action, torch.zeros_like(action), atol=1e-5), \
            f"zero-delta should give all-zero action, got {action[0].tolist()}"

    def test_finger_constant_one(self):
        """Finger labels stored in actions must be constant 1.0."""
        T = 8
        src = _identity_4x4(T)
        tgt = _homogeneous([0.3, -0.2, 0.1], [0.0, 0.0, 0.0], T)
        base = self._base_pose(T)
        d_min, d_max = _default_delta_bounds()
        palm = target_pose_to_pose_action_cup_local(base, base, src, tgt, d_min, d_max)
        finger = torch.ones(T, 5)
        actions = torch.cat([palm, finger], dim=-1)
        assert actions.shape == (T, 11)
        assert torch.all(actions[:, 6:] == 1.0)
