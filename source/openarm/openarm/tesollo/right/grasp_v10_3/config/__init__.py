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
from ..grasp_right_env_cfg import GraspRightEnvCfg, GraspRightEnvCfgNoActorMass


class GraspRightEnvCfg_PLAY(GraspRightEnvCfg):
    """플레이용 설정 (소규모 환경)."""

    def __post_init__(self):
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5


class GraspRightEnvCfgNoActorMass_PLAY(GraspRightEnvCfgNoActorMass):
    """플레이용 oracle-mass-free actor 설정."""

    def __post_init__(self):
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5


gym.register(
    id="open-tesol_r_grasp_v10_3",
    entry_point=(
        "openarm.tesollo.right.grasp_v10_3.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_r_grasp_v10_3-lstm",
    entry_point=(
        "openarm.tesollo.right.grasp_v10_3.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgNoActorMass",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_r_grasp_v10_3-play",
    entry_point=(
        "openarm.tesollo.right.grasp_v10_3.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_r_grasp_v10_3-play-lstm",
    entry_point=(
        "openarm.tesollo.right.grasp_v10_3.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgNoActorMass_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)
