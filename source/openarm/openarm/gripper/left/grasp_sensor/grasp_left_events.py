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

"""이 태스크 전용 이벤트 term.

lift 레퍼런스는 로봇의 **모든** 관절이 액션 대상이라 필요 없던 것이, 비대칭 양팔 로봇에서는
필요해진다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def hold_joints_at_target(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    joint_targets: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """액션이 건드리지 않는 관절의 **PD 위치 목표**를 지정 자세로 고정한다.

    ★왜 필요한가. `ArticulationCfg.init_state.joint_pos` 는 관절의 **상태**만 정한다.
      PD 목표(`joint_pos_target`)는 정하지 않으며, 그 버퍼는 0 으로 시작한다. 액션 대상
      관절은 매 스텝 ActionTerm 이 목표를 써 주지만, **액션 대상이 아닌 관절은 아무도 쓰지
      않는다** — 목표가 0 인 채로 남아 팔이 "차렷"으로 내려간다.

      이 로봇은 왼팔 7 + 그리퍼 1 만 액션 대상이고 오른팔 7 + 오른손 20 + 헤드 2 는 아니다.
      그래서 유휴 오른팔이 중력이 아니라 **0 을 향한 PD 지령** 때문에 내려가 테이블·바닥에
      닿았다(렌더 관찰). effort_limit 을 아무리 올려도 안 고쳐지는 종류의 문제다.

      실측(프로브): 목표를 명시하지 않으면 관절 오차 최대 25.4°, 명시하면 2.1°.

    리셋 때 한 번만 써 주면 된다 — 목표 버퍼는 다음 리셋까지 유지된다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, joint_names = asset.find_joints(list(joint_targets), preserve_order=True)
    target = torch.tensor(
        [joint_targets[n] for n in joint_names], device=asset.device, dtype=torch.float32
    )
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    asset.set_joint_position_target(
        target.unsqueeze(0).expand(len(env_ids), -1), joint_ids=joint_ids, env_ids=env_ids
    )
