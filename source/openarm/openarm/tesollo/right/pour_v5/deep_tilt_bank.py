"""Deep-tilt start-state bank (sparse-reward bootstrap, analysis.md lstm_test1 §6).

During training the policy reaches a deep-tilt, over-target, beads-in-source
state (e.g. lstm_test1's plateau: cup ~80°, pour-point on the target, beads
still in source) but never discovers the final push into a pour, so the
200-weight capture reward never fires. This bank captures those *real* near-pour
frames as full physics snapshots and restores a fraction (``f_boot``) of resets
to them, giving the policy many cheap attempts at the final tilt so the sparse
reward becomes reachable. ``f_boot`` anneals to 0 so the learned value transfers
back to upright starts (the policy already reaches ~80° on its own).

Unlike the grasp warmstart bank — which stores the cup *upright* and *re-samples*
beads, causing a teleport bounce when the joints are tilted — this bank stores
the *real* cup orientation and the captured bead state and restores them
atomically, so the configuration stays physically consistent.

Pure torch only (no Isaac Sim) so it is unit-testable on CPU. The env wires the
mask/anneal in and feeds env-local full-state snapshots to the bank.
"""

from __future__ import annotations

import torch


def capture_mask(
    tilt_amount: torch.Tensor,
    bead_in_source_frac: torch.Tensor,
    mouth_xy_dist: torch.Tensor,
    *,
    tilt_min: float,
    src_min: float,
    mouth_max: float,
) -> torch.Tensor:
    """Boolean mask (per env) of states that are productive deep-tilt seeds.

    A state qualifies only if it is simultaneously:
      - deep enough to be near pour: ``tilt_amount >= tilt_min``
        (``tilt_amount = (1 - cos θ) / 2``);
      - still holding its payload: ``bead_in_source_frac >= src_min``
        (rejects already-pouring frames → no free-pour bootstrap, audit Check 2);
      - aimed at the target: ``mouth_xy_dist <= mouth_max`` (pour-point over the
        target opening, so the final tilt lands beads in the target).

    All inputs are 1-D tensors of equal length; returns a bool tensor.
    """
    return (
        (tilt_amount >= tilt_min)
        & (bead_in_source_frac >= src_min)
        & (mouth_xy_dist <= mouth_max)
    )


def f_boot_ratio(progress: float, *, f_start: float, f_end: float) -> float:
    """Linearly anneal the bootstrap fraction from ``f_start`` to ``f_end``.

    ``progress`` in [0, 1] (clamped) is an external anneal signal (e.g. bead
    curriculum advance or windowed success rate). Returns the fraction of reset
    envs that should restore from the deep-tilt bank this step.
    """
    p = min(max(progress, 0.0), 1.0)
    return f_start + (f_end - f_start) * p


class DeepTiltStateBank:
    """Ring buffer of full-state deep-tilt snapshots (all env-local).

    Stores robot arm/hand joint positions, the palm control reference pose, the
    cup pose (pos + *real* wxyz quat), and the captured bead state
    ``(num_beads, 13)`` so a restore reproduces the exact physical configuration
    and a consistent control reference. Once full, the oldest entry is
    overwritten. Field order mirrors the env restore tuple
    ``(arm, hand, palm_pose, cup_pose, bead_state)``.
    """

    def __init__(
        self,
        capacity: int,
        num_arm: int,
        num_hand: int,
        num_beads: int,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device
        self._count = 0   # number of valid (written) entries
        self._head = 0    # next write position (ring index)
        self.arm = torch.zeros(self.capacity, num_arm, device=device)
        self.hand = torch.zeros(self.capacity, num_hand, device=device)
        self.palm_pose = torch.zeros(self.capacity, 7, device=device)  # pos3 + quat_xyzw4 (control ref)
        self.cup_pose = torch.zeros(self.capacity, 7, device=device)  # pos3 + quat_wxyz4 (env-local)
        self.bead_state = torch.zeros(self.capacity, num_beads, 13, device=device)

    @property
    def count(self) -> int:
        return self._count

    def is_ready(self, min_count: int) -> bool:
        return self._count >= int(min_count)

    def store(
        self,
        arm: torch.Tensor,
        hand: torch.Tensor,
        palm_pose: torch.Tensor,
        cup_pose: torch.Tensor,
        bead_state: torch.Tensor,
    ) -> None:
        """Append a batch of ``m`` snapshots, wrapping around when full."""
        m = int(arm.shape[0])
        if m == 0:
            return
        idx = (self._head + torch.arange(m, device=self.device)) % self.capacity
        self.arm[idx] = arm.to(self.arm.dtype)
        self.hand[idx] = hand.to(self.hand.dtype)
        self.palm_pose[idx] = palm_pose.to(self.palm_pose.dtype)
        self.cup_pose[idx] = cup_pose.to(self.cup_pose.dtype)
        self.bead_state[idx] = bead_state.to(self.bead_state.dtype)
        self._head = (self._head + m) % self.capacity
        self._count = min(self._count + m, self.capacity)

    def sample(self, n: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return ``(n,)`` indices sampled uniformly from the valid entries.

        Only indices ``< count`` are returned, so uninitialized slots are never
        restored.
        """
        if self._count <= 0:
            raise RuntimeError("DeepTiltStateBank.sample called on an empty bank")
        return torch.randint(self._count, (n,), device=self.device, generator=generator)
