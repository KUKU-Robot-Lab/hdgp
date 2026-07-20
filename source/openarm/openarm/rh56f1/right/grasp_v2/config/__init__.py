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
from ..grasp_right_env_cfg import GraspRightEnvCfg

# entry_point 모듈 경로 (grasp_v2 = DEXTRAH 구조, tesollo grasp_v2 매칭 / RH56F1 6-DOF 손).
_ENTRY = (
    "openarm.rh56f1.right.grasp_v2.grasp_right_env:GraspRightEnv"
)


class GraspRightEnvCfg_PLAY(GraspRightEnvCfg):
    """플레이용 설정 (소규모 환경)."""

    def __post_init__(self):
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5


class GraspRightEnvCfg_DISTILL(GraspRightEnvCfg):
    """Distillation 설정 — D435i TiledCamera(RGB+depth) 활성, student obs(116).

    RGB 입력 + object_pos aux(현재 dagger; depth 재구성 head 는 상류 재추가 시). env 당
    320x180 렌더 타깃이 붙어 GPU 메모리가 teacher(4096) 규모를 감당 못하므로 env 축소.
    """

    # teacher 완료 후 제외물체 이름 주입(예: ("cup", "cup_big") — 파지 실패 확정).
    # 빈 튜플이면 제외 없음(전 148종 증류). onehot 은 유지 → teacher 체크포인트 호환.
    DISTILL_EXCLUDED_OBJECT_NAMES: tuple[str, ...] = ()

    def __post_init__(self):
        self.distillation = True
        # DEXTRAH 증류 레시피: env 축소(타일 렌더 제곱수 유리) + aux 강하게(물체 위치 우선 학습).
        self.scene.num_envs = 256
        self.aux_coeff = 10.0
        # student 입력 = RGB (use_depth=False 네트워크와 일치). depth 는 obs 에 함께 렌더.
        self.img_aug_type = "rgb"
        # ★env 를 teacher 작동점(ADR 47, right lstm_test1 최종)에 고정 — ADR 0 의 시연 왜곡 방지.
        self.starting_adr_increments = 47
        # 제외물체(스폰 축소) — 현재 헬퍼 없이 빈 튜플 기본. 지정 시 스포너 교체는 후속.
        self.distill_excluded_object_names = self.DISTILL_EXCLUDED_OBJECT_NAMES


gym.register(
    id="open-rh56f1_r_grasp_v2",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="open-rh56f1_r_grasp_v2-play",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="open-rh56f1_r_grasp_v2-lstm",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

# play-lstm: lstm 체크포인트 rollout(warm-state 수집)용. PLAY 설정(소규모 env) + lstm 네트워크.
gym.register(
    id="open-rh56f1_r_grasp_v2-play-lstm",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

# distill: teacher(lstm) → vision student(D435i mono RGB + object_pos aux) DAgger 증류.
# rl_games_cfg_entry_point 는 teacher cfg — Dagger 가 teacher 를 이 cfg 로 빌드하고,
# student cfg 는 run_distillation.py 가 student_cfg_entry_point 로 따로 넘긴다.
gym.register(
    id="open-rh56f1_r_grasp_v2-distill",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg_DISTILL",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
        "student_cfg_entry_point": (
            f"{agents.__name__}:rl_games_student_mono_transformer.yaml"
        ),
    },
)
