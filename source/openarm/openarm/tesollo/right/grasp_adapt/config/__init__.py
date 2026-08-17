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
        # ★08.17 fresh→warmstart 전환(재정정). fresh(rl_games_ppo_lstm_cfg, LR 3e-4)로
        # 실제 학습을 돌려본 결과 ep1부터 reward -1e14로 즉시 붕괴. 격리 실험으로 근본원인
        # 확인: 12패널 spring-articulated 종이컵이 랜덤(미학습) 액션에 물리적으로 못 버팀
        # (rigid 태스크는 동일 USD·동일 랜덤액션에서 40스텝 정상 — 매니페스트/USD 무죄,
        # deform_ft도 동일 폭발 재현 — 내 mass 변경과도 무관, 순수 콜드스타트 취약성).
        # → test25(이미 gentle하게 잡는 정책) warmstart로 이 상황 자체를 피한다.
        # actor LR 1e-4(07.30 실증: 수렴 정책에 3e-4 쓰면 massshift1처럼 ep~273 붕괴).
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
