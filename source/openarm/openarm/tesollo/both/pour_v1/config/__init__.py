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

import gymnasium as gym

from . import agents
from ..pour_env_cfg import PourEnvCfg


class PourEnvCfgPlay(PourEnvCfg):
    """플레이용 설정 (소규모 환경)."""

    def __post_init__(self):
        self.scene.num_envs = 128
        self.scene.env_spacing = 2.5


# Backward-compatible aliases for older task wiring.
PourRightEnvCfg_PLAY = PourEnvCfgPlay


# ── MLP PPO ──────────────────────────────────────────────────────────────
gym.register(
    id="open-tesol_b_pour_v1",
    entry_point=(
        "openarm.tesollo.both.pour_v1.pour_env:PourEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:PourEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_b_pour_v1-play",
    entry_point=(
        "openarm.tesollo.both.pour_v1.pour_env:PourEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:PourEnvCfgPlay",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# ── LSTM + BC Aux Loss (PourLstmBCAgent) ────────────────────────────────
gym.register(
    id="open-tesol_b_pour_v1-lstm-bc",
    entry_point=(
        "openarm.tesollo.both.pour_v1.pour_env:PourEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:PourEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_bc_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_b_pour_v1-play-lstm-bc",
    entry_point=(
        "openarm.tesollo.both.pour_v1.pour_env:PourEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:PourEnvCfgPlay",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_bc_cfg.yaml",
    },
)
