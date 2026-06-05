# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PourChunkBCBuffer: DemoBCBuffer 래퍼 — (obs[t], action_chunk[t:t+H]) 배치 반환.

DemoBCBuffer.sample(B, H)이 반환하는 (B, H, obs_dim) / (B, H, act_dim) 중
첫 step obs 만 취해 Chunk BC 학습용 배치를 만든다.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .demo_bc_buffer import DemoBCBuffer


class PourChunkBCBuffer:
    """DemoBCBuffer를 Chunk BC 형식으로 감싸는 얇은 래퍼."""

    def __init__(self, demo_buf: DemoBCBuffer, chunk_size: int = 32) -> None:
        self._buf = demo_buf
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        return len(self._buf)

    def is_warm(self, min_count: int = 1) -> bool:
        return self._buf.is_warm(min_count)

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        """(obs_t, action_chunk, mask) 배치 샘플.

        Returns:
            obs_t:        (B, obs_dim)  — 창 첫 step obs
            action_chunk: (B, H, act_dim) — 창 전체 demo action
            mask:         (B, H) bool   — 유효 step 여부
        """
        raw = self._buf.sample(batch_size, self.chunk_size)
        obs_t = raw["obs"][:, 0, :]   # (B, obs_dim)
        actions = raw["actions"]       # (B, H, act_dim)
        mask = raw["mask"]             # (B, H)
        return obs_t, actions, mask
