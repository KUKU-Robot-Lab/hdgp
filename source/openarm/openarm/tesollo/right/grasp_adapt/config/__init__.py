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
from ..grasp_right_env_cfg import (
    GraspRightEnvCfg,
    GraspRightEnvCfgDeformable,
    GraspRightEnvCfgDeformableWater,
    GraspRightEnvCfgMassShift,
    GraspRightEnvCfgNoActorMass,
)


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


class GraspRightEnvCfgDeformable_PLAY(GraspRightEnvCfgDeformable):
    """플레이/스모크용 deformable cup 설정 (소규모 환경)."""

    def __post_init__(self):
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5


gym.register(
    id="open-tesol_r_grasp_adapt",
    entry_point=(
        "openarm.tesollo.right.grasp_adapt.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgNoActorMass",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_r_grasp_adapt-lstm",
    entry_point=(
        "openarm.tesollo.right.grasp_adapt.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgNoActorMass",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_r_grasp_adapt_massshift-lstm",
    entry_point=(
        "openarm.tesollo.right.grasp_adapt.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgMassShift",
        # Phase3 fine-tune 전용: actor LR 1e-4 (fine-tune 붕괴 방지)
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_massshift_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_r_grasp_adapt_deform-lstm",
    entry_point=(
        "openarm.tesollo.right.grasp_adapt.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgDeformable",
        # fresh 재학습(rigid 비전이). fine-tune 아니지만 안정 위해 lstm 표준 yaml.
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_r_grasp_adapt_deform_ft-lstm",
    entry_point=(
        "openarm.tesollo.right.grasp_adapt.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgDeformable",
        # warm-start fine-tune 전용: actor LR 1e-4(붕괴 방지) + minibatch 65536(num_envs 8192).
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_deform_ft_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_r_grasp_adapt_deform_water-lstm",
    entry_point=(
        "openarm.tesollo.right.grasp_adapt.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        # Phase 4: 변형 종이컵 + 물(정적 수위 + 동적 추가). 질량 ADR 2단계 게이팅.
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgDeformableWater",
        # deform_ft와 동일 yaml — actor LR 1e-4. 07.30 실증(LR 3e-4는 수렴 정책을
        # ep~273에 붕괴시킴)에 따라 fresh가 아니라 저LR fine-tune으로 시작한다.
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_deform_ft_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_r_grasp_adapt_deform-play-lstm",
    entry_point=(
        "openarm.tesollo.right.grasp_adapt.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgDeformable_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_r_grasp_adapt-play",
    entry_point=(
        "openarm.tesollo.right.grasp_adapt.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgNoActorMass_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_r_grasp_adapt-play-lstm",
    entry_point=(
        "openarm.tesollo.right.grasp_adapt.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfgNoActorMass_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)
