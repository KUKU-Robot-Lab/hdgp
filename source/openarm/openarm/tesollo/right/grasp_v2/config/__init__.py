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
from ..grasp_right_env_cfg import GraspRightEnvCfg, _grasp_object_spawn_for
from ..grasp_right_utils import kept_object_names_and_indices


class GraspRightEnvCfg_PLAY(GraspRightEnvCfg):
    """플레이용 설정 (소규모 환경)."""

    def __post_init__(self):
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5


class GraspRightEnvCfg_DISTILL(GraspRightEnvCfg):
    """Distillation 설정 — D435i TiledCamera 활성, student obs.

    env 수를 teacher(4096)보다 크게 줄인다: env 당 320x180 RGB-D 렌더 타깃이
    붙어 GPU 메모리가 teacher 규모를 감당하지 못한다.
    """

    # teacher 완료 후 실패물체 이름을 여기 주입한다.
    # onehot 은 active_object_names(148) 유지 → teacher 체크포인트 호환. 빈 튜플이면 제외 없음.
    # [07-23] lstm_test3 teacher(ep_13000, ADR28) clean eval 하위: cup=0.39(2지 핀치,
    # envelope 와 다른 파지모드) · 80597=0.25 · 2dvafvp8=0.32 (teacher 거의 실패).
    # 나쁜 시연 모방 방지 위해 제외 (사용자 지시 cup+teacher<0.35).
    DISTILL_EXCLUDED_OBJECT_NAMES: tuple[str, ...] = ("cup", "80597", "2dvafvp8")

    def __post_init__(self):
        self.distillation = True
        # DEXTRAH 증류 레시피: env.num_envs=256 (타일 렌더는 제곱수가 유리), aux_coeff=10.
        # aux(object_pos 회귀)를 크게 걸어야 인코더가 "물체가 어디 있나"를 먼저 배운다.
        self.scene.num_envs = 256
        self.aux_coeff = 10.0
        # [07-23] ①손가락 손실가중 3.0 + ②action EMA 0.3 은 baseline 대비 무효(in_success·
        # obj_drift 불변) → 격리 위해 off(base 기본 1.0/0.0). 다음 가설=RGB crop(지각정밀도,
        # grasp_right_env.py). 필요 시 아래 재활성:
        # self.finger_loss_weight = 3.0
        # self.action_ema_alpha = 0.3
        # ★env 를 teacher 실제 작동점(ADR 28)에 고정 — teacher(lstm_test3 ep_13000)는
        # ADR 28 에서 포화(게이트 0.4 미달)돼 그 이상은 학습 안 됨. 만렙(50)에 고정하면
        # teacher 가 미학습 난이도에서 굴러 시연이 왜곡된다(deterministic 작동점 = 28,
        # clean in_success 0.417). ADR 0 도 안 됨(dt1 고원 0.19 = 스폰/abduction 왜곡).
        self.starting_adr_increments = 28
        # student RGB 입력(D435i, 실물 RGB 정합 + visual DR). student network use_depth=False
        # (img_aug_type="rgb") 와 반드시 일치 — dagger modality 가드가 불일치를 막는다.
        # depth 는 인코더 입력이 아니라 aux 재구성 대상(base aug_depth=False 유지).
        self.img_aug_type = "rgb"
        # 실패물체 제외: onehot 은 153 유지, 스포너만 kept 로 교체(env 는 object_idx 를
        # 원본 슬롯으로 remap). teacher 학습 env 는 이 경로를 타지 않는다(distillation=False).
        self.distill_excluded_object_names = self.DISTILL_EXCLUDED_OBJECT_NAMES
        if self.distill_excluded_object_names:
            kept, _ = kept_object_names_and_indices(
                list(self.active_object_names), self.distill_excluded_object_names
            )
            self.cup_cfg.spawn = _grasp_object_spawn_for(kept)


gym.register(
    id="open-tesol_r_grasp_v2",
    entry_point=(
        "openarm.tesollo.right.grasp_v2.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_r_grasp_v2-play",
    entry_point=(
        "openarm.tesollo.right.grasp_v2.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="open-tesol_r_grasp_v2-lstm",
    entry_point=(
        "openarm.tesollo.right.grasp_v2.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

# play-lstm: lstm 체크포인트 rollout(warm-state 수집)용. rh56f1 의
# open-rh56f1_r_grasp_v1-play-lstm 과 대칭. PLAY 설정(소규모 env) + lstm 네트워크.
gym.register(
    id="open-tesol_r_grasp_v2-play-lstm",
    entry_point=(
        "openarm.tesollo.right.grasp_v2.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

# distill: teacher(lstm) → vision student(D435i mono RGB-D) DAgger 증류.
# rl_games_cfg_entry_point 는 teacher cfg 를 가리킨다 — Dagger 가 teacher 를
# 이 cfg 로 빌드하고, student cfg 는 run_distillation.py 가 따로 넘긴다.
gym.register(
    id="open-tesol_r_grasp_v2-distill",
    entry_point=(
        "openarm.tesollo.right.grasp_v2.grasp_right_env:GraspRightEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:GraspRightEnvCfg_DISTILL",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
        "student_cfg_entry_point": (
            f"{agents.__name__}:rl_games_student_mono_transformer.yaml"
        ),
    },
)
