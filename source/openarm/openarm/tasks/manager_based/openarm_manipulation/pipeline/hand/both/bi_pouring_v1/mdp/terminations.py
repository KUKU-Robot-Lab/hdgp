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

"""Termination functions for bi_pouring_v1."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..bi_pouring_env import BiPouringEnv


def major_spill(env: BiPouringEnv) -> torch.Tensor:
    """bead가 양쪽 컵 밖으로 크게 흘렸을 때 종료."""
    return env._major_spill_flag


def invalid_state(env: BiPouringEnv) -> torch.Tensor:
    """obs가 NaN이거나 컵/bead가 작업 공간 밖으로 나갔을 때 종료."""
    return env._invalid_state_flag
