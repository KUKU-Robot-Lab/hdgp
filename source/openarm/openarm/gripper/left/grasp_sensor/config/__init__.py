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
  조용히 삼켜진다**. 등록 실패는 "task not found" 로만 보이므로 건드린 뒤에는 항상 확인할 것:
      python3 -c "import openarm.tasks, gymnasium as gym; print(gym.spec('open-grip_l_grasp_sensor'))"

entry_point 가 커스텀 env 클래스가 아니라 **`isaaclab.envs:ManagerBasedRLEnv`** 다 —
IsaacLab lift 레시피를 그대로 쓰므로 env 코드를 따로 둘 이유가 없다.

id 규약: open-{robot}_{l|r|b}_{task}. train.py 가
  open-grip_l_grasp_sensor → log/rl_games/open-grip/left/grasp-sensor/ 로 해석한다.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="open-grip_l_grasp_sensor",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "openarm.gripper.left.grasp_sensor.grasp_left_env_cfg:GraspLeftGripperEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="open-grip_l_grasp_sensor-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "openarm.gripper.left.grasp_sensor.grasp_left_env_cfg:GraspLeftGripperEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# ── 태스크공간(diff-IK) 변형 ─────────────────────────────────────────────
# 팔 액션만 TCP 상대 pose 6D 로 바꾼 것. 보상·씬·목표는 위와 동일해서 같은 조건에서
# 관절공간판과 머리를 맞댈 수 있다. 근거는 `grasp_left_ik_env_cfg.py` docstring.
gym.register(
    id="open-grip_l_grasp_sensor_ik",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "openarm.gripper.left.grasp_sensor.grasp_left_ik_env_cfg:GraspLeftGripperIKEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="open-grip_l_grasp_sensor_ik-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "openarm.gripper.left.grasp_sensor.grasp_left_ik_env_cfg:GraspLeftGripperIKEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# ── Fabrics 변형 ─────────────────────────────────────────────────────────
# 팔 액션만 절대 palm 6D(Fabrics)로 바꾼 것. 보상·씬·목표는 관절공간판과 동일해서
# test17 과 제어기만 다른 직접 비교가 성립한다. 근거는 `grasp_left_fab_env_cfg.py`.
gym.register(
    id="open-grip_l_grasp_sensor_fab",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "openarm.gripper.left.grasp_sensor.grasp_left_fab_env_cfg:GraspLeftGripperFabEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_fab_cfg.yaml",
    },
)

gym.register(
    id="open-grip_l_grasp_sensor_fab-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "openarm.gripper.left.grasp_sensor.grasp_left_fab_env_cfg:GraspLeftGripperFabEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_fab_cfg.yaml",
    },
)

