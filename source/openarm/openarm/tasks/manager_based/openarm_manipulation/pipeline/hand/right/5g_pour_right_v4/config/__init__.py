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
from ..pour_right_env_cfg import PourRightEnvCfg


class PourRightEnvCfg_PLAY(PourRightEnvCfg):
    """플레이용 설정 (소규모 환경)."""

    def __post_init__(self):
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.warmstart_cache_size = 1
        self.warmstart_max_rollout_steps = 6000


# ── MLP PPO (기존 방식, 하위 호환) ──────────────────────────────────────
gym.register(
    id="5g_pour_right-v4",
    entry_point=(
        "openarm.tasks.manager_based.openarm_manipulation"
        ".pipeline.hand.right.5g_pour_right_v4"
        ".pour_right_env:PourRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:PourRightEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="5g_pour_right-play-v4",
    entry_point=(
        "openarm.tasks.manager_based.openarm_manipulation"
        ".pipeline.hand.right.5g_pour_right_v4"
        ".pour_right_env:PourRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:PourRightEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# ── LSTM + BC Aux Loss (PourLstmBCAgent) ────────────────────────────────
gym.register(
    id="5g_pour_right-v4-lstm-bc",
    entry_point=(
        "openarm.tasks.manager_based.openarm_manipulation"
        ".pipeline.hand.right.5g_pour_right_v4"
        ".pour_right_env:PourRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:PourRightEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_bc_cfg.yaml",
    },
)

gym.register(
    id="5g_pour_right-play-v4-lstm-bc",
    entry_point=(
        "openarm.tasks.manager_based.openarm_manipulation"
        ".pipeline.hand.right.5g_pour_right_v4"
        ".pour_right_env:PourRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:PourRightEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_bc_cfg.yaml",
    },
)
