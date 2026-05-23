# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""시뮬 성공 궤적 ring buffer (BC aux loss 학습용).

에피소드가 성공으로 끝난 env의 (obs, action) 시퀀스를 저장한다.
용량 초과 시 score 가 가장 낮은 슬롯을 교체한다.
"""

from __future__ import annotations

import torch
from torch import Tensor


class SuccessTrajBuffer:
    """성공 에피소드 궤적 ring buffer.

    capacity 초과 시 min-score 슬롯 교체 전략을 사용한다.
    LSTM BC 학습을 위해 시간 순서가 보존된 contiguous seq 를 샘플링한다.

    메모리 (기본 설정):
        256 × 200 × (110+11) × 4B ≈ 23 MB
    """

    def __init__(
        self,
        capacity: int,
        max_len: int,
        obs_dim: int,
        act_dim: int,
        device: str,
    ) -> None:
        self.capacity = capacity
        self.max_len  = max_len
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim
        self.device   = device
        D = obs_dim + act_dim

        self._data    = torch.zeros(capacity, max_len, D, device=device)
        self._lengths = torch.zeros(capacity, dtype=torch.long, device=device)
        self._scores  = torch.full((capacity,), -1.0e9, device=device)
        self._count   = 0

    def store(self, obs_seq: Tensor, act_seq: Tensor, score: float) -> bool:
        """(T, obs_dim), (T, act_dim) 궤적을 저장.

        Returns:
            True  — 저장 성공
            False — score 미달로 skip
        """
        T_obs = int(obs_seq.shape[0])
        T_act = int(act_seq.shape[0])
        T = min(T_obs, T_act, self.max_len)
        if T == 0:
            return False

        if self._count < self.capacity:
            slot = self._count
            self._count += 1
        else:
            min_score, min_idx = self._scores[: self._count].min(dim=0)
            if score <= float(min_score.item()):
                return False
            slot = int(min_idx.item())

        traj = torch.cat([obs_seq[:T], act_seq[:T]], dim=-1)   # (T, D)
        self._data[slot, :T].copy_(traj)
        if T < self.max_len:
            self._data[slot, T:].zero_()
        self._lengths[slot] = T
        self._scores[slot]  = score
        return True

    def sample(self, batch_size: int, seq_len: int) -> dict | None:
        """BC 학습용 (B, seq_len) 배치 샘플.

        Returns:
            {"obs": (B, seq_len, obs_dim), "actions": (B, seq_len, act_dim), "mask": (B, seq_len)}
            버퍼가 비어있으면 None.
        """
        if self._count == 0:
            return None

        dev = self._data.device
        traj_idx = torch.randint(self._count, (batch_size,), device=dev)
        lengths  = self._lengths[traj_idx]                            # (B,)

        max_start = (lengths - seq_len).clamp(min=0)
        start = (torch.rand(batch_size, device=dev) * (max_start.float() + 1.0)).long().clamp_(max=max_start)

        t_idx = (start.unsqueeze(1) + torch.arange(seq_len, device=dev).unsqueeze(0)).clamp_(max=self.max_len - 1)
        batch = self._data[traj_idx[:, None], t_idx]    # (B, seq_len, D)

        step_pos = start.unsqueeze(1) + torch.arange(seq_len, device=dev).unsqueeze(0)
        mask = step_pos < lengths.unsqueeze(1)           # (B, seq_len)

        return {
            "obs":     batch[..., : self.obs_dim],
            "actions": batch[..., self.obs_dim :],
            "mask":    mask,
        }

    def __len__(self) -> int:
        return self._count

    def is_warm(self, min_count: int = 10) -> bool:
        return self._count >= min_count
