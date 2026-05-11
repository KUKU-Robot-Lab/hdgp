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
        # warmstart: BC pour policy는 컵을 쥔 상태에서만 동작
        # play용: rollout steps를 충분히 늘려 4 env로도 안정적 수집
        # grasp 품질 기준은 유지 (컵이 떨어지지 않으려면 solid grasp 필요)
        self.warmstart_cache_size = 16            # 훈련용 256보다 작되 충분한 다양성
        self.warmstart_max_rollout_steps = 40000  # 4 env × 40000 steps → 충분
        self.warmstart_cache_min_contacts = 4     # 5→4: 약간 완화하되 solid grasp 유지
        self.warmstart_stable_hold_steps = 15     # 30→15: 안정 확인 단축 (0.25s)
        self.warmstart_cache_min_lift_height = 0.10  # 0.15→0.10m: 확실히 들린 상태
        # BC 체크포인트 play 호환: 훈련 시 zeros였던 obs 슬롯 마스킹
        self.zero_bc_missing_obs = True
        # grasp 안착 유예: warmstart 후 물리 안정화에 더 여유를 줌
        self.drop_force_hold_steps = 90           # 10→90: 1.5s 유예 (물리 안정화)


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
