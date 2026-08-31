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

"""grasp_sensor_v2 종료 항 — **이송에서 끝나는 에피소드**(08.31 라운드 17).

과제를 "컵을 목표로 옮기고 끝"으로 재정의한다. 목표 반경 안에 연속
`P.EPISODE_DWELL_STEPS` 스텝 머물면 그 자리에서 에피소드를 끊는다. 그 뒤 자세
정리는 정책이 아니라 IK 가 맡으므로 "오래 버티기"를 학습시킬 이유가 없다.

★★이 항은 반드시 `DoneTerm(..., time_out=True)` 로 등록해야 한다.
  진짜 종료(terminated)로 두면 가치 부트스트랩이 끊긴다. 그러면 성공은 남은
  스텝의 보상을 **포기하는 행위**가 되고, 정책은 목표 반경 밖을 맴돌며 stage 3
  보상을 계속 빠는 쪽이 유리해진다 — 성공을 회피하도록 학습한다.
  계약 테스트가 이 플래그를 고정한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from . import v2_preset as P
from . import v2_stages as S

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GoalDwellDone(ManagerTermBase):
    """목표 반경 안 **연속** 체류가 문턱에 닿으면 True.

    자체 카운터를 들고 있다 — 보상의 `Staircase._hold` 를 재사용하지 않는다.
    매니저 실행 순서(보상 → 종료)에 판정이 묶이면 한 스텝 밀리거나, 보상 항을
    끄는 순간 종료가 조용히 죽는다.

    체류가 끊기면 0 으로 되돌린다(단조 아님). 그래서 진동으로 반경을 들락거리는
    정책은 문턱에 못 닿고, 실제로 자리를 잡아야만 끝난다.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._dwell = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._dwell[:] = 0.0
        else:
            self._dwell[env_ids] = 0.0

    def __call__(self, env: "ManagerBasedRLEnv",
                 command_name: str,
                 robot_cfg: SceneEntityCfg,
                 jaw_cfg: SceneEntityCfg,
                 object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
                 dwell_steps: int | None = None) -> torch.Tensor:
        thr = float(P.EPISODE_DWELL_STEPS if dwell_steps is None else dwell_steps)
        ok = S.settle_success(env, command_name, robot_cfg, jaw_cfg, object_cfg)
        self._dwell = torch.where(ok > 0.5, self._dwell + 1.0,
                                  torch.zeros_like(self._dwell))
        return self._dwell >= thr

    def dwell_steps(self) -> torch.Tensor:
        """진단용 — 현재 연속 체류 스텝(리셋 포함)."""
        return self._dwell


def diag_goal_dwell(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """등록된 `GoalDwellDone` 의 연속 체류 스텝. 항이 없으면 0."""
    try:
        term = env.termination_manager.get_term_cfg("goal_dwell").func
    except (AttributeError, ValueError, KeyError):
        return torch.zeros(env.num_envs, device=env.device)
    if not isinstance(term, GoalDwellDone):
        return torch.zeros(env.num_envs, device=env.device)
    return term.dwell_steps()
