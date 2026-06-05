"""DiffusionBCNet — 외부 라이브러리 없는 자립형 구현.

의존성: torch만 필요. diffusers / diffusion_policy 레포 불필요.
"""
from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
BC_OBS_DIM    = 52
BC_ACTION_DIM = 11
BC_CHUNK_SIZE = 16


# ---------------------------------------------------------------------------
# EMA (외부 레포 대체)
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.averaged_model = copy.deepcopy(model)
        self.decay = decay

    def step(self, online: nn.Module) -> None:
        with torch.no_grad():
            for ema_p, p in zip(self.averaged_model.parameters(), online.parameters()):
                ema_p.data.lerp_(p.data, 1.0 - self.decay)


# ---------------------------------------------------------------------------
# 노이즈 스케줄 (cosine, DDPM/DDIM 공용)
# ---------------------------------------------------------------------------

def _cosine_alphas_cumprod(T: int) -> torch.Tensor:
    """cosine schedule → alpha_bar (T,)  [t=1..T]"""
    s = 0.008
    t = torch.linspace(0, T, T + 1)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    return ac[1:].clamp(1e-6, 1.0 - 1e-6)


# ---------------------------------------------------------------------------
# 네트워크 빌딩 블록
# ---------------------------------------------------------------------------

class _SinPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class _FiLMBlock(nn.Module):
    """FiLM 조건부 residual 블록."""

    def __init__(self, d: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc   = nn.Linear(d, d)
        self.film = nn.Linear(cond_dim, 2 * d)
        self.act  = nn.Mish()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1.0 + scale) + shift
        return x + self.act(self.fc(h))


class _ActionDenoiser(nn.Module):
    """flatten(action_chunk) → 노이즈 예측."""

    def __init__(self, flat_dim: int, cond_dim: int, hidden: int = 512, depth: int = 6) -> None:
        super().__init__()
        self.in_proj  = nn.Linear(flat_dim, hidden)
        self.blocks   = nn.ModuleList([_FiLMBlock(hidden, cond_dim) for _ in range(depth)])
        self.out_proj = nn.Linear(hidden, flat_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# 메인 모델
# ---------------------------------------------------------------------------

class DiffusionBCNet(nn.Module):
    """DDPM 학습 + DDIM inference Diffusion BC.

    obs(52D) → cond(256+128D) → ActionDenoiser → action_chunk(16×11D)
    """

    def __init__(
        self,
        obs_dim:    int = BC_OBS_DIM,
        action_dim: int = BC_ACTION_DIM,
        chunk_size: int = BC_CHUNK_SIZE,
        ddpm_steps: int = 100,
    ) -> None:
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.ddpm_steps = ddpm_steps
        self.flat_dim   = action_dim * chunk_size

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Mish(), nn.Linear(256, 256)
        )
        self.t_emb = nn.Sequential(
            _SinPosEmb(128),
            nn.Linear(128, 128), nn.Mish(), nn.Linear(128, 128),
        )
        cond_dim = 256 + 128
        self.denoiser = _ActionDenoiser(self.flat_dim, cond_dim)

        self.register_buffer("alphas_cumprod", _cosine_alphas_cumprod(ddpm_steps))

    # ------------------------------------------------------------------
    def _cond(self, obs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.obs_encoder(obs), self.t_emb(t)], dim=-1)

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        obs:          torch.Tensor,           # (B, obs_dim)
        action_chunk: torch.Tensor,           # (B, chunk_size, action_dim)
        mask:         Optional[torch.Tensor] = None,  # (B, chunk_size) bool
    ) -> torch.Tensor:
        B = obs.shape[0]
        t  = torch.randint(0, self.ddpm_steps, (B,), device=obs.device)
        ac = self.alphas_cumprod[t].view(B, 1, 1)

        noise = torch.randn_like(action_chunk)
        noisy = ac.sqrt() * action_chunk + (1.0 - ac).sqrt() * noise

        cond = self._cond(obs, t)
        pred = self.denoiser(noisy.flatten(1), cond).view(B, self.chunk_size, self.action_dim)

        if mask is None:
            return F.mse_loss(pred, noise)
        m = mask.unsqueeze(-1).float()
        return F.mse_loss(pred * m, noise * m, reduction="sum") / m.sum().clamp(1.0)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, obs: torch.Tensor, n_steps: int = 5) -> torch.Tensor:
        """DDIM K-step 추론 → 첫 action (B, action_dim)."""
        B, device = obs.shape[0], obs.device

        step_ratio = max(1, self.ddpm_steps // n_steps)
        timesteps  = list(reversed(range(0, self.ddpm_steps, step_ratio)))[:n_steps]

        x = torch.randn(B, self.chunk_size, self.action_dim, device=device)

        for i, ti in enumerate(timesteps):
            t_vec     = torch.full((B,), ti, device=device, dtype=torch.long)
            cond      = self._cond(obs, t_vec)
            pred_eps  = self.denoiser(x.flatten(1), cond).view(B, self.chunk_size, self.action_dim)

            ac_t  = self.alphas_cumprod[ti]
            pred_x0 = ((x - (1.0 - ac_t).sqrt() * pred_eps) / ac_t.sqrt()).clamp(-1.0, 1.0)

            if i < len(timesteps) - 1:
                ac_next = self.alphas_cumprod[timesteps[i + 1]]
            else:
                ac_next = torch.ones(1, device=device)

            x = ac_next.sqrt() * pred_x0 + (1.0 - ac_next).sqrt() * pred_eps

        return x[:, 0, :]
