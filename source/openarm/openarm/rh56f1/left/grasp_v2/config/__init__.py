# Copyleft 2025 Enactic, Inc.
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
from ..grasp_left_env_cfg import GraspLeftEnvCfg

# entry_point 모듈 경로 (grasp_v2 = DEXTRAH 구조, tesollo grasp_v2 매칭 / RH56F1 6-DOF 손).
_ENTRY = (
    "openarm.rh56f1.left.grasp_v2.grasp_left_env:GraspLeftEnv"
)


class GraspLeftEnvCfg_PLAY(GraspLeftEnvCfg):
    """플레이용 설정 (소규모 환경)."""

    def __post_init__(self):
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5


class GraspLeftEnvCfg_DISTILL(GraspLeftEnvCfg):
    """Distillation 설정 — D435i(RGB+depth) 활성, student obs(116). right 규약 동일."""

    DISTILL_EXCLUDED_OBJECT_NAMES: tuple[str, ...] = ()

    def __post_init__(self):
        self.distillation = True
        self.scene.num_envs = 256
        self.aux_coeff = 10.0
        self.img_aug_type = "rgb"
        # ★env 를 teacher 작동점(ADR 50, left lstm_test3 만렙)에 고정.
        self.starting_adr_increments = 50
        self.distill_excluded_object_names = self.DISTILL_EXCLUDED_OBJECT_NAMES


gym.register(
    id="open-rh56f1_l_grasp_v2",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspLeftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="open-rh56f1_l_grasp_v2-play",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspLeftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="open-rh56f1_l_grasp_v2-lstm",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspLeftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

# play-lstm: lstm 체크포인트 rollout(warm-state 수집)용. PLAY 설정(소규모 env) + lstm 네트워크.
gym.register(
    id="open-rh56f1_l_grasp_v2-play-lstm",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspLeftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

# distill: teacher(lstm) → vision student(D435i mono RGB + object_pos aux) DAgger 증류.
gym.register(
    id="open-rh56f1_l_grasp_v2-distill",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspLeftEnvCfg_DISTILL",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
        "student_cfg_entry_point": (
            f"{agents.__name__}:rl_games_student_mono_transformer.yaml"
        ),
    },
)
