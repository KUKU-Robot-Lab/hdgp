"""DiffusionActor — skrl GaussianMixin Actor (Phase 2 RL fine-tune).

skrl PPO에서 policy 모델로 사용.
BC checkpoint을 로드해 "이미 pour하는" prior에서 RL 시작.

YAML config에서:
  class: openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.right.5g_pour_right_v6.diffusion_actor_skrl:DiffusionActor
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_V6_DIR = str(Path(__file__).parent)
if _V6_DIR not in sys.path:
    sys.path.insert(0, _V6_DIR)

from diffusion_bc.net import DiffusionBCNet


def _find_hdgp_root() -> Optional[Path]:
    """__file__ 에서 위로 올라가며 'hdgp' 디렉터리를 찾는다."""
    p = Path(__file__).resolve()
    while p != p.parent:
        if p.name == "hdgp":
            return p
        p = p.parent
    return None


_DEFAULT_BC_CKPT = ""
_hdgp = _find_hdgp_root()
if _hdgp is not None:
    _DEFAULT_BC_CKPT = str(_hdgp / "log" / "diffusion_bc" / "v6" / "bc_5000.pt")

try:
    from skrl.models.torch import GaussianMixin, Model
except ImportError as e:
    raise ImportError(f"skrl not installed: {e}. pip install skrl>=1.4.3")


class DiffusionActor(GaussianMixin, Model):
    """Diffusion-prior Gaussian Actor.

    compute() = DDIM K-step denoising → mu + learnable log_std.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        bc_checkpoint_path: Optional[str] = _DEFAULT_BC_CKPT,
        n_ddim_steps: int = 5,
        freeze_bc: bool = False,
        min_log_std: float = -4.0,
        max_log_std: float = 0.0,
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        self.diffusion_bc  = DiffusionBCNet().to(device)
        self.n_ddim_steps  = n_ddim_steps

        resolved_path = bc_checkpoint_path or _DEFAULT_BC_CKPT
        if resolved_path and Path(resolved_path).exists():
            ckpt = torch.load(resolved_path, map_location=device)
            self.diffusion_bc.load_state_dict(ckpt.get("model", ckpt))
            print(f"[DiffusionActor] loaded BC checkpoint: {resolved_path}")
        else:
            if resolved_path:
                print(f"[DiffusionActor] WARNING: checkpoint not found: {resolved_path}")
            print("[DiffusionActor] starting from random weights")

        if freeze_bc:
            for p in self.diffusion_bc.parameters():
                p.requires_grad_(False)

        num_actions = action_space.shape[0]
        self.log_std_parameter = nn.Parameter(
            torch.full((num_actions,), -2.0, device=device)
        )

    def compute(self, inputs, role: str):
        obs = inputs["states"]  # (B, 52)
        if self.training:
            mu = self._ddim_with_grad(obs)
        else:
            mu = self.diffusion_bc.sample(obs, n_steps=self.n_ddim_steps)
        return mu, self.log_std_parameter, {}

    def _ddim_with_grad(self, obs: torch.Tensor) -> torch.Tensor:
        """Gradient를 허용하는 DDIM 샘플링 (PPO 업데이트용)."""
        net = self.diffusion_bc
        B, device = obs.shape[0], obs.device

        step_ratio = max(1, net.ddpm_steps // self.n_ddim_steps)
        timesteps  = list(reversed(range(0, net.ddpm_steps, step_ratio)))[:self.n_ddim_steps]

        x = torch.randn(B, net.chunk_size, net.action_dim, device=device)

        for i, ti in enumerate(timesteps):
            t_vec    = torch.full((B,), ti, device=device, dtype=torch.long)
            cond     = net._cond(obs, t_vec)
            pred_eps = net.denoiser(x.flatten(1), cond).view(B, net.chunk_size, net.action_dim)

            ac_t    = net.alphas_cumprod[ti]
            pred_x0 = ((x - (1.0 - ac_t).sqrt() * pred_eps) / ac_t.sqrt()).clamp(-1.0, 1.0)

            ac_next = net.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 \
                      else torch.ones(1, device=device)

            x = ac_next.sqrt() * pred_x0 + (1.0 - ac_next).sqrt() * pred_eps

        return x[:, 0, :]

    def bc_loss(
        self,
        obs: torch.Tensor,
        action_chunk: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.diffusion_bc.compute_loss(obs, action_chunk, mask)
