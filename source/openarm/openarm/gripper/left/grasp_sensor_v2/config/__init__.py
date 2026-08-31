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

⚠ 그 임포트는 `except (ModuleNotFoundError, ImportError): pass` 로 감싸여 있어 **에러가
  조용히 삼켜진다**. 등록 실패는 "task not found" 로만 보이므로 건드린 뒤 반드시 확인할 것:
      python3 -c "import openarm.tasks, gymnasium as gym; print(gym.spec('open-grip_l_grasp_sensor_v2'))"

id 규약: open-{robot}_{l|r|b}_{task}. train.py 가
  open-grip_l_grasp_sensor_v2 → log/rl_games/open-grip/left/grasp-sensor-v2/ 로 해석한다.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="open-grip_l_grasp_sensor_v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "openarm.gripper.left.grasp_sensor_v2.v2_env_cfg:GraspLeftV2EnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_v2_cfg.yaml",
    },
)

gym.register(
    id="open-grip_l_grasp_sensor_v2-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "openarm.gripper.left.grasp_sensor_v2.v2_env_cfg:GraspLeftV2EnvCfg_PLAY"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_v2_cfg.yaml",
    },
)
