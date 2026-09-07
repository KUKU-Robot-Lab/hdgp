"""리프트 후 질량정규화 힘/토크 외란 (순수 torch, isaaclab 금지).

DESIGN.md §5·§8. SimToolReal 동일: env 별 발화확률 p ~ logU(prob_range) 를 리셋마다
재추첨하고, 매 스텝 rand<p 인 env 에 randn·mass·scale 을 새로 뽑는다. decay 는 0 이라
이전 값은 매 스텝 소거된다(발화 안 한 스텝은 외란 0). **lifted 일 때만** 적용한다.
호출측이 `object.set_external_force_and_torque(forces, torques, is_global=True)` 로 넘긴다.
"""

from __future__ import annotations

import math

import torch


def sample_log_uniform(lo: float, hi: float, n: int, device) -> torch.Tensor:
    """(n,) ~ exp(U(log lo, log hi)). lo>0, hi≥lo."""
    lo, hi = float(lo), float(hi)
    if lo <= 0.0 or hi < lo:
        raise ValueError(f"log-uniform needs 0 < lo ≤ hi, got ({lo}, {hi})")
    u = torch.rand(int(n), device=device)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


class WrenchDR:
    """forces/torques (N,1,3) — 두 번째 축은 body 수(물체 1개)."""

    def __init__(
        self,
        num_envs: int,
        device,
        *,
        force_scale: float = 20.0,
        torque_scale: float = 2.0,
        prob_range: tuple[float, float] = (0.001, 0.1),
    ) -> None:
        if int(num_envs) < 1:
            raise ValueError(f"num_envs must be ≥ 1, got {num_envs}")
        if len(prob_range) != 2:
            raise ValueError(f"prob_range must be (lo, hi), got {prob_range}")
        self.num_envs = int(num_envs)
        self.device = device
        self.force_scale = float(force_scale)
        self.torque_scale = float(torque_scale)
        self.prob_range = (float(prob_range[0]), float(prob_range[1]))
        self.forces = torch.zeros(self.num_envs, 1, 3, device=device)
        self.torques = torch.zeros(self.num_envs, 1, 3, device=device)
        self.p_force = torch.zeros(self.num_envs, device=device)
        self.p_torque = torch.zeros(self.num_envs, device=device)
        self.reset(torch.arange(self.num_envs, device=device))

    def reset(self, ids) -> None:
        """확률 재추첨(log-uniform) · wrench 0."""
        n = int(self.p_force[ids].numel())
        lo, hi = self.prob_range
        self.p_force[ids] = sample_log_uniform(lo, hi, n, self.device)
        self.p_torque[ids] = sample_log_uniform(lo, hi, n, self.device)
        self.forces[ids] = 0.0
        self.torques[ids] = 0.0

    def step(self, mass: torch.Tensor, lifted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """mass (N,) · lifted (N,) bool → (forces (N,1,3), torques (N,1,3))."""
        n = self.num_envs
        if tuple(mass.shape) != (n,):
            raise ValueError(f"mass: expected ({n},), got {tuple(mass.shape)}")
        if tuple(lifted.shape) != (n,) or lifted.dtype != torch.bool:
            raise ValueError(f"lifted: expected bool ({n},), got {tuple(lifted.shape)} {lifted.dtype}")
        dev = self.forces.device
        # 왜 매번 새 텐서: decay 0 = 이전 값 소거. 발화 안 한 env 는 0 이어야 한다.
        gate = lifted.unsqueeze(-1)
        fire_f = (torch.rand(n, device=dev) < self.p_force).unsqueeze(-1) & gate
        new_f = torch.randn(n, 3, device=dev) * mass.unsqueeze(-1) * self.force_scale
        fire_t = (torch.rand(n, device=dev) < self.p_torque).unsqueeze(-1) & gate
        new_t = torch.randn(n, 3, device=dev) * mass.unsqueeze(-1) * self.torque_scale
        self.forces = torch.where(fire_f, new_f, torch.zeros_like(new_f)).unsqueeze(1)
        self.torques = torch.where(fire_t, new_t, torch.zeros_like(new_t)).unsqueeze(1)
        return self.forces, self.torques
