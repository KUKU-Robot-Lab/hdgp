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
from ..grasp_left_env_cfg import GraspLeftEnvCfg, _grasp_object_spawn_for
from ..grasp_left_utils import kept_object_names_and_indices


class GraspLeftEnvCfg_PLAY(GraspLeftEnvCfg):
    """플레이용 설정 (소규모 환경)."""

    def __post_init__(self):
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5


class GraspLeftEnvCfg_DISTILL(GraspLeftEnvCfg):
    """Distillation 설정 — D435i TiledCamera 활성, student obs.

    env 수를 teacher(4096)보다 크게 줄인다: env 당 320x180 RGB-D 렌더 타깃이
    붙어 GPU 메모리가 teacher 규모를 감당하지 못한다.
    """

    # teacher 완료 후 실패물체 이름을 여기 주입한다(예: ("small_5_cyl", "small_8_cyl")).
    # onehot 은 active_object_names(153) 유지 → teacher 체크포인트 호환. 빈 튜플이면 제외 없음.
    # right 와 동일 목록을 쓴다(합집합, 좌우 대칭).
    DISTILL_EXCLUDED_OBJECT_NAMES: tuple[str, ...] = ()

    def __post_init__(self):
        self.distillation = True
        # DEXTRAH 증류 레시피: env.num_envs=256 (타일 렌더는 제곱수가 유리), aux_coeff=10.
        # aux(object_pos 회귀)를 크게 걸어야 인코더가 "물체가 어디 있나"를 먼저 배운다.
        self.scene.num_envs = 256
        self.aux_coeff = 10.0
        # student depth 입력(D435i, sim2real 갭↓). student network use_depth=True 와 반드시 일치
        # (dagger 가 불일치를 막는다). depth 증강(aug_depth)이 이 경로에서 켜진다.
        self.img_aug_type = "depth"
        # 실패물체 제외: onehot 은 153 유지, 스포너만 kept 로 교체(env 는 object_idx 를
        # 원본 슬롯으로 remap). teacher 학습 env 는 이 경로를 타지 않는다(distillation=False).
        self.distill_excluded_object_names = self.DISTILL_EXCLUDED_OBJECT_NAMES
        if self.distill_excluded_object_names:
            kept, _ = kept_object_names_and_indices(
                list(self.active_object_names), self.distill_excluded_object_names
            )
            self.cup_cfg.spawn = _grasp_object_spawn_for(kept)


gym.register(
    id="open-tesol_l_grasp_v2",
    entry_point=(
        "openarm.tesollo.left.grasp_v2.grasp_left_env:GraspLeftEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspLeftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_l_grasp_v2-play",
    entry_point=(
        "openarm.tesollo.left.grasp_v2.grasp_left_env:GraspLeftEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspLeftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_l_grasp_v2-lstm",
    entry_point=(
        "openarm.tesollo.left.grasp_v2.grasp_left_env:GraspLeftEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspLeftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

# play-lstm: lstm 체크포인트 rollout(warm-state 수집)용. rh56f1 의
# open-rh56f1_l_grasp_v1-play-lstm 과 대칭. PLAY 설정(소규모 env) + lstm 네트워크.
gym.register(
    id="open-tesol_l_grasp_v2-play-lstm",
    entry_point=(
        "openarm.tesollo.left.grasp_v2.grasp_left_env:GraspLeftEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspLeftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

# distill: teacher(lstm) → vision student(D435i mono RGB-D) DAgger 증류.
# rl_games_cfg_entry_point 는 teacher cfg 를 가리킨다 — Dagger 가 teacher 를
# 이 cfg 로 빌드하고, student cfg 는 run_distillation.py 가 따로 넘긴다.
gym.register(
    id="open-tesol_l_grasp_v2-distill",
    entry_point=(
        "openarm.tesollo.left.grasp_v2.grasp_left_env:GraspLeftEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspLeftEnvCfg_DISTILL",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
        "student_cfg_entry_point": (
            f"{agents.__name__}:rl_games_student_mono_transformer.yaml"
        ),
    },
)
