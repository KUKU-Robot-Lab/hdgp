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

"""Custom curriculum functions for 5g_lift_left tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class linear_reward_weight(ManagerTermBase):
    """Curriculum that linearly interpolates a reward weight over a step range.

    Unlike ``modify_reward_weight`` (hard step-function), this term smoothly
    ramps the weight from ``start_weight`` to ``end_weight`` between
    ``start_step`` and ``end_step``.

    Usage example in CurriculumCfg::

        action_rate = CurrTerm(
            func=mdp.linear_reward_weight,
            params={
                "term_name":    "action_rate",
                "start_weight": -5e-4,   # 현재 weight와 동일하게 맞춰서 시작
                "end_weight":   -5e-3,   # 목표 weight
                "start_step":   0,
                "end_step":     50000,
            },
        )

    Notes:
        - ``start_step`` 이전에는 ``start_weight``가 그대로 유지됩니다.
        - ``end_step`` 이후에는 ``end_weight``가 고정됩니다.
        - 토글 방법: CurriculumCfg 필드를 주석 처리/해제하면 됩니다.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        start_weight: float,
        end_weight: float,
        start_step: int,
        end_step: int,
    ) -> float:
        t = env.common_step_counter

        if t <= start_step:
            w = start_weight
        elif t >= end_step:
            w = end_weight
        else:
            alpha = (t - start_step) / (end_step - start_step)
            w = start_weight + alpha * (end_weight - start_weight)

        self._term_cfg.weight = w
        env.reward_manager.set_term_cfg(term_name, self._term_cfg)
        return w
