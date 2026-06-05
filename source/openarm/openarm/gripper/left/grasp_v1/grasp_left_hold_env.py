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


class Grasp2gHoldEnv(ManagerBasedRLEnv):
    """Grasp env that holds gripper closed once grasp phase begins.

    V2 Addition: Role-separated curriculum with action masking.
    - Stage 0 (LEFT_ONLY): Right arm holds initial pose
    - Stage 1 (RIGHT_ONLY): Left arm holds initial pose
    - Stage 2 (BIMANUAL): Both arms active
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_gripper_hold = getattr(self.cfg, "enable_gripper_hold", False)
        self.mask_inactive_arm = getattr(self.cfg, "mask_inactive_arm_actions", True)
        self._setup_gripper_hold()
        self._setup_curriculum_masking()
        if getattr(self.cfg, "debug_obs_split", False):
            self._log_obs_split_debug()
        if getattr(self.cfg, "debug_reward_mapping", False):
            self._check_left_right_reward_mapping()

    def _setup_gripper_hold(self) -> None:
        self._action_slices: dict[str, slice] = {}
        idx = 0
        for name, term in self.action_manager._terms.items():
            dim = term.action_dim
            self._action_slices[name] = slice(idx, idx + dim)
            idx += dim

        self._left_hand_term = self.action_manager.get_term("left_hand_action")
        self._right_hand_term = self.action_manager.get_term("right_hand_action")

        self._left_close_raw = self._compute_close_raw(self._left_hand_term)
        self._right_close_raw = self._compute_close_raw(self._right_hand_term)

    def _setup_curriculum_masking(self) -> None:
        """Setup initial pose actions for curriculum stage masking."""
        # Left arm initial actions (zero = hold current offset position)
        if "left_arm_action" in self._action_slices:
            sl = self._action_slices["left_arm_action"]
            dim = sl.stop - sl.start
            self._left_arm_hold = torch.zeros(self.num_envs, dim, device=self.device)

        # Right arm initial actions
        if "right_arm_action" in self._action_slices:
            sl = self._action_slices["right_arm_action"]
            dim = sl.stop - sl.start
            self._right_arm_hold = torch.zeros(self.num_envs, dim, device=self.device)

        # Left hand open position (BinaryJointPositionAction uses 0=open, 1=close)
        if "left_hand_action" in self._action_slices:
            sl = self._action_slices["left_hand_action"]
            dim = sl.stop - sl.start
            # For BinaryJointPositionAction, action=0 means open
            self._left_hand_open = torch.zeros(self.num_envs, dim, device=self.device)

        # Right hand open position
        if "right_hand_action" in self._action_slices:
            sl = self._action_slices["right_hand_action"]
            dim = sl.stop - sl.start
            # For BinaryJointPositionAction, action=0 means open
            self._right_hand_open = torch.zeros(self.num_envs, dim, device=self.device)

    def _get_curriculum_stage(self) -> int:
        """Get current curriculum stage from config."""
        return int(getattr(self.cfg, "curriculum_stage", 2))

    def _compute_close_raw(self, term) -> torch.Tensor:
        if hasattr(term, "_close_command"):
            return term._close_command.unsqueeze(0).expand(self.num_envs, -1)

        joint_ids = term._joint_ids
        joint_limits = term._asset.data.joint_pos_limits
        target = joint_limits[:, joint_ids, 0]

        if isinstance(term._offset, torch.Tensor):
            offset = term._offset
        else:
            offset = torch.full_like(target, float(term._offset))

        if isinstance(term._scale, torch.Tensor):
            scale = term._scale
        else:
            scale = torch.full_like(target, float(term._scale))

        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        return (target - offset) / scale

    def _get_phase(self, attr_name: str) -> torch.Tensor:
        if hasattr(self, attr_name):
            phase = getattr(self, attr_name)
            if isinstance(phase, torch.Tensor):
                return phase
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def step(self, action: torch.Tensor):
        action = action.to(self.device)

        # ===== Curriculum Stage Action Masking =====
        # Stage 0 (LEFT_ONLY): Right arm holds initial pose, left arm active
        # Stage 1 (RIGHT_ONLY): Left arm holds initial pose, right arm active
        # Stage 2 (BIMANUAL): Both arms active
        if self.mask_inactive_arm:
            stage = self._get_curriculum_stage()

            if stage == 0:  # LEFT_ONLY: freeze right arm
                if "right_arm_action" in self._action_slices:
                    sl = self._action_slices["right_arm_action"]
                    action[:, sl] = self._right_arm_hold
                if "right_hand_action" in self._action_slices:
                    sl = self._action_slices["right_hand_action"]
                    action[:, sl] = self._right_hand_open

            elif stage == 1:  # RIGHT_ONLY: freeze left arm
                if "left_arm_action" in self._action_slices:
                    sl = self._action_slices["left_arm_action"]
                    action[:, sl] = self._left_arm_hold
                if "left_hand_action" in self._action_slices:
                    sl = self._action_slices["left_hand_action"]
                    action[:, sl] = self._left_hand_open

            # stage == 2 (BIMANUAL): no masking, both arms active

        # ===== Gripper Hold (phase-based) =====
        if self.enable_gripper_hold:
            left_phase = self._get_phase("grasp2g_phase_left")
            right_phase = self._get_phase("grasp2g_phase_right")

            if "left_hand_action" in self._action_slices:
                left_mask = left_phase >= 1
                if torch.any(left_mask):
                    sl = self._action_slices["left_hand_action"]
                    action[left_mask, sl] = self._left_close_raw[left_mask]

            if "right_hand_action" in self._action_slices:
                right_mask = right_phase >= 1
                if torch.any(right_mask):
                    sl = self._action_slices["right_hand_action"]
                    action[right_mask, sl] = self._right_close_raw[right_mask]

        return super().step(action)

    def set_gripper_hold(self, enabled: bool) -> None:
        """Toggle gripper hold behavior at runtime."""
        self.enable_gripper_hold = bool(enabled)

    def _log_obs_split_debug(self) -> None:
        if not hasattr(self, "observation_manager"):
            return
        obs_mgr = self.observation_manager
        if "policy" not in obs_mgr.active_terms:
            return
        term_names = list(obs_mgr.active_terms["policy"])
        term_dims = list(obs_mgr.group_obs_term_dim["policy"])
        sizes = []
        for dims in term_dims:
            sizes.append(int(torch.prod(torch.tensor(dims)).item()))
        total = sum(sizes)
        split = getattr(self.cfg, "actor_obs_split_index", None)
        # infer left block size by ordering (until first right_* term)
        inferred_split = 0
        for name, size in zip(term_names, sizes):
            if name.startswith("right_") or name.startswith("cup2_to_hand_right"):
                break
            inferred_split += size
        print("[OBS_SPLIT] policy terms:")
        for name, size in zip(term_names, sizes):
            print(f"[OBS_SPLIT]   {name}: {size}")
        print(f"[OBS_SPLIT] total={total} split_index={split} inferred_left_size={inferred_split}")

    def _check_left_right_reward_mapping(self) -> None:
        rewards_cfg = getattr(self.cfg, "rewards", None)
        if rewards_cfg is None:
            return
        mismatches = []
        for name, term in vars(rewards_cfg).items():
            if term is None or not hasattr(term, "params"):
                continue
            params = term.params
            if not isinstance(params, dict):
                continue
            object_cfg = params.get("object_cfg", None)
            eef_link_name = params.get("eef_link_name", "")
            if "left" in name:
                if object_cfg is not None and getattr(object_cfg, "name", None) != "cup":
                    mismatches.append((name, "object_cfg", getattr(object_cfg, "name", None)))
                if eef_link_name and "left" not in eef_link_name:
                    mismatches.append((name, "eef_link_name", eef_link_name))
            if "right" in name:
                if object_cfg is not None and getattr(object_cfg, "name", None) != "cup2":
                    mismatches.append((name, "object_cfg", getattr(object_cfg, "name", None)))
                if eef_link_name and "right" not in eef_link_name:
                    mismatches.append((name, "eef_link_name", eef_link_name))

        if mismatches:
            print("[REWARD_MAP] mismatches detected:")
            for name, key, val in mismatches:
                print(f"[REWARD_MAP]   {name}: {key}={val}")
        else:
            print("[REWARD_MAP] left/right reward mapping OK")
