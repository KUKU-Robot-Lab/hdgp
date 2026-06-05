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

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class Lift5gHoldEnv(ManagerBasedRLEnv):
    """Lift env with 2g-style inactive-arm action masking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_inactive_arm = getattr(self.cfg, "mask_inactive_arm_actions", True)
        self._setup_curriculum_masking()

    def _setup_curriculum_masking(self) -> None:
        self._action_slices: dict[str, slice] = {}
        idx = 0
        for name, term in self.action_manager._terms.items():
            dim = term.action_dim
            self._action_slices[name] = slice(idx, idx + dim)
            idx += dim

        if "left_arm_action" in self._action_slices:
            sl = self._action_slices["left_arm_action"]
            self._left_arm_hold = torch.zeros(self.num_envs, sl.stop - sl.start, device=self.device)
        if "right_arm_action" in self._action_slices:
            sl = self._action_slices["right_arm_action"]
            self._right_arm_hold = torch.zeros(self.num_envs, sl.stop - sl.start, device=self.device)
        if "left_hand_action" in self._action_slices:
            sl = self._action_slices["left_hand_action"]
            # FingerSynergyAction maps -1 to fully open and +1 to fully close.
            # Use -1 for inactive hand so it stays open instead of half-closed.
            self._left_hand_hold = -torch.ones(self.num_envs, sl.stop - sl.start, device=self.device)
        if "right_hand_action" in self._action_slices:
            sl = self._action_slices["right_hand_action"]
            self._right_hand_hold = -torch.ones(self.num_envs, sl.stop - sl.start, device=self.device)
        if "left_thumb_action" in self._action_slices:
            sl = self._action_slices["left_thumb_action"]
            self._left_thumb_hold = torch.zeros(self.num_envs, sl.stop - sl.start, device=self.device)
        if "right_thumb_action" in self._action_slices:
            sl = self._action_slices["right_thumb_action"]
            self._right_thumb_hold = torch.zeros(self.num_envs, sl.stop - sl.start, device=self.device)

    def _get_curriculum_stage(self) -> int:
        return int(getattr(self.cfg, "curriculum_stage", 2))

    def step(self, action: torch.Tensor):
        action = action.to(self.device)

        if self.mask_inactive_arm:
            stage = self._get_curriculum_stage()

            if stage == 0:
                if "right_arm_action" in self._action_slices:
                    action[:, self._action_slices["right_arm_action"]] = self._right_arm_hold
                if "right_hand_action" in self._action_slices:
                    action[:, self._action_slices["right_hand_action"]] = self._right_hand_hold
                if "right_thumb_action" in self._action_slices:
                    action[:, self._action_slices["right_thumb_action"]] = self._right_thumb_hold

            elif stage == 1:
                if "left_arm_action" in self._action_slices:
                    action[:, self._action_slices["left_arm_action"]] = self._left_arm_hold
                if "left_hand_action" in self._action_slices:
                    action[:, self._action_slices["left_hand_action"]] = self._left_hand_hold
                if "left_thumb_action" in self._action_slices:
                    action[:, self._action_slices["left_thumb_action"]] = self._left_thumb_hold

        return super().step(action)
