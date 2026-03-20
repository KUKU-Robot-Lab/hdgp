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

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..bi_pouring_env import BiPouringEnv


def transport_goal(env: BiPouringEnv) -> torch.Tensor:
    return env._r_transport_goal


def palm_to_goal(env: BiPouringEnv) -> torch.Tensor:
    return env._r_palm_to_goal


def tilt(env: BiPouringEnv) -> torch.Tensor:
    return env._r_tilt


def directional_tilt(env: BiPouringEnv) -> torch.Tensor:
    return env._r_directional_tilt


def height(env: BiPouringEnv) -> torch.Tensor:
    return env._r_height


def mouth_alignment(env: BiPouringEnv) -> torch.Tensor:
    return env._r_mouth_alignment


def bead_spill_penalty(env: BiPouringEnv) -> torch.Tensor:
    return env._spill_penalty


def collision_penalty(env: BiPouringEnv) -> torch.Tensor:
    return env._collision_penalty


def action_smoothness_penalty(env: BiPouringEnv) -> torch.Tensor:
    return env._smoothness_penalty
