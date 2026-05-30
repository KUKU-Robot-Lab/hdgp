# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PourChunkBCAgent: LSTM PPO + Step BC + Chunk BC.

기존 PourLstmBCAgent (step-by-step BC) 에 Chunk BC를 추가한다.

Chunk BC:
  obs[t] → MLP → LSTM(zero h₀) → chunk_head → action_chunk(11×H)
  loss = MSE(chunk_pred, demo_action[t:t+H])

chunk_head는 별도 optimizer로 업데이트하며 model(LSTM+encoder)은 PPO가 학습한다.
LSTM hidden이 BC forward를 통해서도 gradient를 받지 않으므로 PPO 학습이 안정적이다.

등록:
  v6 __init__.py 가 'pour_chunk_bc' 알고리즘을 algo_factory에 등록한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lstm_bc_agent import PourLstmBCAgent, _bc_weight
from .pour_chunk_bc_buffer import PourChunkBCBuffer
from .pour_right_constants import NUM_ACTIONS


class PourChunkBCAgent(PourLstmBCAgent):
    """LSTM PPO + step BC (부모) + chunk BC (신규)."""

    def __init__(self, base_name: str, params: dict) -> None:
        super().__init__(base_name, params)
        cfg = params.get("config", {})

        self._chunk_size     = int(cfg.get("chunk_size",              32))
        self._chunk_warmup   = int(cfg.get("chunk_bc_warmup_epochs",   0))
        self._chunk_decay    = int(cfg.get("chunk_bc_decay_epochs", 5000))
        self._chunk_w_init   = float(cfg.get("chunk_bc_weight_init",  10.0))
        self._chunk_w_final  = float(cfg.get("chunk_bc_weight_final",  1.0))
        self._chunk_bc_batch = int(cfg.get("chunk_bc_batch_size", self._bc_batch))

        lstm_units = int(params.get("network", {}).get("rnn", {}).get("units", 512))

        # chunk head: LSTM output (lstm_units) → action chunk (NUM_ACTIONS × H)
        self.chunk_head = nn.Sequential(
            nn.LayerNorm(lstm_units),
            nn.Linear(lstm_units, lstm_units),
            nn.ELU(),
            nn.Linear(lstm_units, NUM_ACTIONS * self._chunk_size),
        ).to(self.ppo_device)

        self.chunk_optimizer = torch.optim.Adam(
            self.chunk_head.parameters(),
            lr=float(cfg.get("chunk_bc_lr", 1e-4)),
        )

        # LSTM output 포착용 forward hook
        self._lstm_feat: torch.Tensor | None = None
        self._register_lstm_hook()

        self._chunk_buf: PourChunkBCBuffer | None = None
        self._last_bc_chunk_loss = 0.0

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def _register_lstm_hook(self) -> None:
        rnn_wrapper = getattr(self.model.a2c_network, "rnn", None)
        if rnn_wrapper is None:
            raise RuntimeError(
                "PourChunkBCAgent: model.a2c_network.rnn 을 찾을 수 없음 — "
                "YAML에 rnn 섹션이 있는지 확인"
            )

        def _hook(module: nn.Module, input: tuple, output: tuple) -> None:
            out = output[0]             # (seq_len, B, hidden)
            self._lstm_feat = out[-1]   # (B, hidden), 마지막 step

        rnn_wrapper.register_forward_hook(_hook)

    # ------------------------------------------------------------------
    # Chunk buffer lazy init
    # ------------------------------------------------------------------

    def _get_chunk_buffer(self, demo_buf) -> PourChunkBCBuffer | None:
        if demo_buf is None:
            return None
        if self._chunk_buf is None:
            self._chunk_buf = PourChunkBCBuffer(demo_buf, self._chunk_size)
        return self._chunk_buf

    # ------------------------------------------------------------------
    # calc_gradients override
    # ------------------------------------------------------------------

    def calc_gradients(self, input_dict: dict) -> None:
        # 1. PPO + step-by-step BC (부모 메서드가 전체 backward 수행)
        super().calc_gradients(input_dict)

        # 2. Chunk BC (chunk_head 전용 별도 backward)
        chunk_w = _bc_weight(
            self.epoch_num, self._chunk_warmup, self._chunk_decay,
            self._chunk_w_init, self._chunk_w_final,
        )
        if chunk_w < 1e-6:
            return

        demo_buf = self._resolve_demo_buffer()
        chunk_buf = self._get_chunk_buffer(demo_buf)
        if chunk_buf is None or not chunk_buf.is_warm(self._demo_min_buf):
            return

        obs_t, demo_chunk, mask = chunk_buf.sample(self._chunk_bc_batch)
        B = obs_t.shape[0]

        obs_prep = self._preproc_obs(obs_t)                          # (B, obs_dim)
        zero_rnn = self._make_zero_rnn_states(B, self.ppo_device)

        # 새 계산 그래프 — 부모 backward 이후이므로 기존 graph는 소멸
        with torch.enable_grad():
            bc_fwd = {
                "is_train":   True,
                "obs":        obs_prep,
                "rnn_states": zero_rnn,
                "seq_length": 1,
            }
            self.model(bc_fwd)           # hook → self._lstm_feat = (B, lstm_units)

            lstm_feat = self._lstm_feat  # (B, lstm_units), gradient path intact

            chunk_pred = self.chunk_head(lstm_feat)                         # (B, H*11)
            chunk_pred = chunk_pred.reshape(B, self._chunk_size, NUM_ACTIONS)  # (B, H, 11)

            mask_f = mask.unsqueeze(-1).float()                             # (B, H, 1)
            diff = (chunk_pred - demo_chunk).pow(2) * mask_f                # (B, H, 11)
            denom = (mask_f.sum() * NUM_ACTIONS).clamp(min=1.0)
            chunk_loss = diff.sum() / denom
            total_chunk = chunk_w * chunk_loss

        self.chunk_optimizer.zero_grad()
        total_chunk.backward()
        nn.utils.clip_grad_norm_(self.chunk_head.parameters(), self.grad_norm)
        self.chunk_optimizer.step()

        self._last_bc_chunk_loss = float(chunk_loss.detach().item())

    # ------------------------------------------------------------------
    # write_stats override
    # ------------------------------------------------------------------

    def write_stats(self, *args, **kwargs) -> None:
        super().write_stats(*args, **kwargs)
        frame = args[11] if len(args) > 11 else kwargs.get("frame", 0)
        if not (hasattr(self, "writer") and self.writer is not None):
            return
        chunk_w = _bc_weight(
            self.epoch_num, self._chunk_warmup, self._chunk_decay,
            self._chunk_w_init, self._chunk_w_final,
        )
        self.writer.add_scalar("bc/chunk_loss",   self._last_bc_chunk_loss, frame)
        self.writer.add_scalar("bc/chunk_weight", chunk_w,                  frame)
