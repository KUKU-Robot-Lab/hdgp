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

"""gym 등록. `openarm/tasks/__init__.py` 가 `*/*/*/config/__init__.py` 를 glob 임포트한다.

⚠ 그 임포트는 `except (ModuleNotFoundError, ImportError): pass` 로 감싸여 있어
  **에러가 조용히 삼켜진다**. 등록 실패는 "task not found" 로만 보이므로,
  이 패키지를 건드린 뒤에는 항상 아래를 실행해 확인할 것:
      python3 -c "import openarm.tasks, gymnasium as gym; print(gym.spec('open-grip_l_grasp_sensor'))"

id 규약: open-{robot}_{l|r|b}_{task}. 로그 경로는 train.py 가
  open-grip_l_grasp_sensor → log/rl_games/open-grip/left/grasp-sensor/ 로 해석한다.
"""

import gymnasium as gym

from . import agents
from ..grasp_left_env_cfg import GraspLeftGripperEnvCfg

_ENTRY = "openarm.gripper.left.grasp_sensor.grasp_left_env:GraspLeftGripperEnv"


class GraspLeftGripperEnvCfg_PLAY(GraspLeftGripperEnvCfg):
    """플레이용 설정 (소규모 환경)."""

    def __post_init__(self):
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5


for _suffix, _cfg, _yaml in (
    ("", "GraspLeftGripperEnvCfg", "rl_games_ppo_cfg.yaml"),
    ("-play", "GraspLeftGripperEnvCfg_PLAY", "rl_games_ppo_cfg.yaml"),
    ("-lstm", "GraspLeftGripperEnvCfg", "rl_games_ppo_lstm_cfg.yaml"),
    ("-play-lstm", "GraspLeftGripperEnvCfg_PLAY", "rl_games_ppo_lstm_cfg.yaml"),
):
    gym.register(
        id=f"open-grip_l_grasp_sensor{_suffix}",
        entry_point=_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:{_cfg}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:{_yaml}",
        },
    )
