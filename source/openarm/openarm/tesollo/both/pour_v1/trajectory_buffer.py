# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""궤적 캡처 및 성공 궤적 버퍼.

TrajectoryCapture       : env 별 rolling window 기록 (GPU circular buffer)
SuccessTrajectoryBuffer : 성공 에피소드 궤적 저장 (PourLstmBCAgent BC loss 학습용)

설계 원칙:
- 모든 텐서는 GPU에 상주 (CPU 전송 없음 → hot-path 오버헤드 최소화)
- TrajectoryCapture.append() 는 매 policy step 호출 — 연산량 최소화 (scatter write 1회)
- SuccessTrajectoryBuffer.store() 는 에피소드 종료 시만 호출 — 빈도 낮음
- sample() 은 BC minibatch 생성; LSTM BPTT 에 맞는 contiguous seq 반환
"""

from __future__ import annotations

import torch
from torch import Tensor


class TrajectoryCapture:
    """env 별 rolling window 기록 (circular buffer).

    매 step append() 로 (obs_actor, action) 을 저장.
    에피소드 종료 시 get_trajectory() 로 시간 순서 궤적 추출 후 reset_envs() 로 초기화.

    메모리:
        2048 envs × 200 window × 117 data_dim × 4B ≈ 192 MB
        (window=200 @ 60Hz ≈ 마지막 3.3초)
    """

    def __init__(
        self,
        num_envs: int,
        window: int,
        obs_dim: int,
        act_dim: int,
        device: str,
    ) -> None:
        self.num_envs = num_envs
        self.window = window
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        data_dim = obs_dim + act_dim

        # (num_envs, window, data_dim) — circular buffer
        self._data  = torch.zeros(num_envs, window, data_dim, device=device)
        self._ptr   = torch.zeros(num_envs, dtype=torch.long, device=device)   # 다음 쓰기 위치
        self._count = torch.zeros(num_envs, dtype=torch.long, device=device)   # 저장된 스텝 수 (≤window)
        self._env_idx = torch.arange(num_envs, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    def append(self, obs: Tensor, act: Tensor) -> None:
        """모든 env 에 (obs, act) 한 스텝 기록.

        Args:
            obs: (num_envs, obs_dim) — actor 관측
            act: (num_envs, act_dim) — 정책 액션
        """
        slot = self._ptr % self.window                                     # (N,)
        self._data[self._env_idx, slot] = torch.cat([obs, act], dim=-1)   # (N, data_dim)
        self._ptr += 1
        self._count = (self._count + 1).clamp_(max=self.window)

    # ------------------------------------------------------------------
    def get_trajectory(self, env_id: int) -> Tensor:
        """env_id 의 궤적을 시간 순서로 반환: (T, data_dim).

        T = min(captured_steps, window).
        버퍼가 비어 있으면 (0, data_dim) 반환.
        """
        count = int(self._count[env_id].item())
        if count == 0:
            return torch.zeros(0, self.obs_dim + self.act_dim, device=self.device)
        ptr = int(self._ptr[env_id].item())
        if count < self.window:
            # 아직 버퍼 미포화 — 앞부분만 유효
            return self._data[env_id, :count].clone()
        # 포화: ptr 가 wrap-around 한 상태 — 시간 순서 재조립
        start = ptr % self.window
        return torch.cat([
            self._data[env_id, start:],
            self._data[env_id, :start],
        ], dim=0).clone()

    def get_count(self, env_id: int) -> int:
        """env_id 에 현재 저장된 스텝 수."""
        return int(self._count[env_id].item())

    # ------------------------------------------------------------------
    def reset_envs(self, env_ids: Tensor) -> None:
        """에피소드 종료 env 의 버퍼를 초기화.

        Args:
            env_ids: (K,) long tensor — 리셋할 env 인덱스
        """
        if env_ids.numel() == 0:
            return
        self._data[env_ids]  = 0.0
        self._ptr[env_ids]   = 0
        self._count[env_ids] = 0


class SuccessTrajectoryBuffer:
    """성공 에피소드 궤적 ring buffer (BC loss 학습용).

    score 기반 우선순위: 용량 초과 시 가장 낮은 score 궤적을 교체.
    LSTM BC 학습을 위해 contiguous seq_len 시퀀스를 샘플링.

    메모리:
        256 capacity × 200 max_len × 117 data_dim × 4B ≈ 23 MB
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
        data_dim = obs_dim + act_dim

        self._data    = torch.zeros(capacity, max_len, data_dim, device=device)
        self._lengths = torch.zeros(capacity, dtype=torch.long, device=device)
        self._scores  = torch.full((capacity,), -1.0e9, device=device)
        self._count   = 0   # 현재 유효 슬롯 수

    # ------------------------------------------------------------------
    def store(self, trajectory: Tensor, score: float) -> bool:
        """궤적 (T, data_dim) 을 score 와 함께 저장.

        - 용량 미달이면 빈 슬롯에 저장.
        - 용량 포화이면 score 가 가장 낮은 슬롯을 교체 (score 가 낮으면 skip).

        Returns:
            True  — 저장 성공
            False — score 미달로 skip
        """
        T = min(int(trajectory.shape[0]), self.max_len)
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

        self._data[slot, :T]  = trajectory[:T]
        if T < self.max_len:
            self._data[slot, T:] = 0.0          # 패딩 초기화
        self._lengths[slot]   = T
        self._scores[slot]    = score
        return True

    # ------------------------------------------------------------------
    def sample(self, batch_size: int, seq_len: int) -> dict | None:
        """BC 학습용 (B, seq_len) 시퀀스 배치 샘플.

        Returns dict 또는 버퍼가 비어있으면 None:
            "obs"     : (B, seq_len, obs_dim)   — actor 관측
            "actions" : (B, seq_len, act_dim)   — 정책 액션
            "mask"    : (B, seq_len) bool        — 유효 스텝 (패딩=False)
        """
        if self._count == 0:
            return None

        dev = self._data.device

        # 랜덤 궤적 선택 (복원 추출)
        traj_idx = torch.randint(self._count, (batch_size,), device=dev)
        lengths  = self._lengths[traj_idx]                              # (B,)

        # 각 궤적에서 유효 시작점 균등 샘플
        max_start = (lengths - seq_len).clamp(min=0)                   # (B,)
        start = (
            torch.rand(batch_size, device=dev) * (max_start.float() + 1.0)
        ).long().clamp_(max=max_start)                                  # (B,)

        # 시퀀스 인덱스: (B, seq_len)
        t_idx = (
            start.unsqueeze(1) + torch.arange(seq_len, device=dev).unsqueeze(0)
        ).clamp_(max=self.max_len - 1)

        # 데이터 수집: (B, seq_len, data_dim)  — advanced indexing
        batch = self._data[traj_idx[:, None], t_idx]

        # 유효 마스크: 패딩 위치는 False
        step_pos = start.unsqueeze(1) + torch.arange(seq_len, device=dev).unsqueeze(0)
        mask = step_pos < lengths.unsqueeze(1)                          # (B, seq_len)

        return {
            "obs":     batch[..., : self.obs_dim],    # (B, T, obs_dim)
            "actions": batch[..., self.obs_dim :],    # (B, T, act_dim)
            "mask":    mask,                           # (B, T)
        }

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._count

    def is_warm(self, min_count: int = 10) -> bool:
        """BC 학습 시작 가능 여부 판단."""
        return self._count >= min_count
