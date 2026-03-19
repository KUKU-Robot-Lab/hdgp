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
from ..bi_pouring_env_cfg import BiPouringEnvCfg, BiPouringEnvCfg_PLAY


gym.register(
    id="bi_pouring-v1",
    entry_point=(
        "openarm.tasks.manager_based.openarm_manipulation"
        ".pipeline.hand.both.bi_pouring_v1.bi_pouring_env:BiPouringEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:BiPouringEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="bi_pouring-play-v1",
    entry_point=(
        "openarm.tasks.manager_based.openarm_manipulation"
        ".pipeline.hand.both.bi_pouring_v1.bi_pouring_env:BiPouringEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:BiPouringEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
