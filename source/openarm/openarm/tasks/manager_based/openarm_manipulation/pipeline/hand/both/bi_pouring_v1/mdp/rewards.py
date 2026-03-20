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

"""DexPour Transport → Pour 보상 함수 (bi_pouring_v1).

Stage 분기 (ρ-trigger):
  Transport (pour_stage_active=False): 컵 이동 + 직립 유지
  Pour      (pour_stage_active=True):  45° 틸팅 + pour axis 정렬
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..bi_pouring_env import BiPouringEnv


# ---------------------------------------------------------------------------
# Transport stage (ρ=0)
# ---------------------------------------------------------------------------

def transport_cup_distance(env: BiPouringEnv) -> torch.Tensor:
    """pour point → target opening 거리 감소 보상.

    Transport/Pour 단계 모두 항상 활성.
    exp(-2 * dist): dist=0 → 1.0, dist=0.5 → 0.37
    """
    return env._r_cup_dist


def transport_upright_penalty(env: BiPouringEnv) -> torch.Tensor:
    """이송 중 컵 기울기 패널티 (Transport 단계 한정).

    source_up · world_up 가 1에 가까울수록 직립 → 패널티 0.
    반환값은 양수 (cfg에서 음수 weight 부여).
    """
    return env._p_upright * (~env._pour_stage_active).float()


# ---------------------------------------------------------------------------
# Pour stage (ρ=1)
# ---------------------------------------------------------------------------

def pour_tilt(env: BiPouringEnv) -> torch.Tensor:
    """45° 틸팅 Gaussian 보상 (Pour 단계 한정).

    cos(45°) ≈ 0.707을 목표로 하는 Gaussian.
    """
    return env._r_tilt * env._pour_stage_active.float()


def pour_align(env: BiPouringEnv) -> torch.Tensor:
    """pour axis → target opening 방향 정렬 보상 (Pour 단계 한정).

    0.5*(1 + cos): 완전 정렬 → 1.0, 반대 → 0.0.
    """
    return env._r_align * env._pour_stage_active.float()


# ---------------------------------------------------------------------------
# Bead dynamics (항상 활성)
# ---------------------------------------------------------------------------

def bead_entry(env: BiPouringEnv) -> torch.Tensor:
    """target cup에 새로 진입한 bead 개수 (이벤트성)."""
    return env._bead_entry_reward


def bead_stable_retention(env: BiPouringEnv) -> torch.Tensor:
    """target cup 안에서 bead가 안정적으로 머무를 때 연속 보상."""
    return env._stable_retention_reward


def bead_spill_penalty(env: BiPouringEnv) -> torch.Tensor:
    """흘린 bead 비율 패널티.

    반환값은 양수 (cfg에서 음수 weight 부여).
    """
    return env._spill_penalty


# ---------------------------------------------------------------------------
# Safety / smoothness
# ---------------------------------------------------------------------------

def collision_penalty(env: BiPouringEnv) -> torch.Tensor:
    """rim / EE 근접 충돌 패널티.

    반환값은 양수 (cfg에서 음수 weight 부여).
    """
    return env._collision_penalty


def action_smoothness_penalty(env: BiPouringEnv) -> torch.Tensor:
    """연속 action 변화량 L2 패널티.

    반환값은 양수 (cfg에서 음수 weight 부여).
    """
    return env._smoothness_penalty
