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

"""Initial DirectRLEnv skeleton for bi_pouring_v1.

This task intentionally preserves the structural style of 5g_grasp_right_v5 while
leaving scene/reset/observation/reward/bead details as explicit TODOs.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

from .bi_pouring_constants import NUM_ACTIONS, NUM_OBSERVATIONS
from .bi_pouring_env_cfg import BiPouringEnvCfg
from .bi_pouring_preset import (
    BEAD_SPAWN_POS_SOURCE_CUP_B,
    BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ,
    LEFT_HOLDER_FIXED_JOINT_POS,
    RIGHT_HAND_GRASP_JOINT_POS,
    RIGHT_HAND_JOINT_NAMES,
    RIGHT_ARM_HOME_POSE,
)


class BiPouringEnv(DirectRLEnv):
    """Right-arm pouring with a fixed left-side holder for the initial v1 scope."""

    cfg: BiPouringEnvCfg

    def __init__(self, cfg: BiPouringEnvCfg, render_mode: str | None = None, **kwargs):
        self._right_arm_joint_ids: list[int] = []
        self._right_hand_joint_ids: list[int] = []
        self._left_holder_joint_ids: list[int] = []

        self._right_arm_home = None
        self._right_hand_grasp = None
        self._left_holder_home = None
        self._last_actions = None
        self._joint_pos_target = None
        self._obs_buf = None
        self._state_buf = None
        self._right_source_cup_attach_pos_b = None
        self._right_source_cup_attach_quat_b = None
        self._left_target_cup_attach_pos_b = None
        self._left_target_cup_attach_quat_b = None
        self._bead_spawn_pos_source_cup_b = None
        self._bead_spawn_quat_source_cup = None
        self._right_source_cup_body_id = -1
        self._left_target_cup_body_id = -1
        self._source_cup_pour_point_pos_b = None
        self._target_cup_opening_pos_b = None
        self._source_cup_pour_axis_b = None
        self._source_cup_up_axis_b = None
        self._target_cup_up_axis_b = None
        self._world_up_axis = None
        self._stable_alignment_steps = None
        self._stable_retention_steps = None
        self._prev_actions = None
        self._prev_bead_in_target_flag = None
        self._bead_has_entered_target_flag = None
        self._bead_exited_target_after_entry_flag = None
        self._alignment_reward = None
        self._controlled_tilt_reward = None
        self._bead_entry_reward = None
        self._stable_retention_reward = None
        self._spill_penalty = None
        self._collision_penalty = None
        self._smoothness_penalty = None
        self._major_spill_flag = None
        self._invalid_state_flag = None
        self._bead_in_target_flag = None
        self._bead_in_source_flag = None
        self._bead_spilled_flag = None
        self._success_flag = None

        super().__init__(cfg, render_mode, **kwargs)

        for name in cfg.policy_arm_joint_names:
            self._right_arm_joint_ids.append(self.robot.joint_names.index(name))
        for name in RIGHT_HAND_JOINT_NAMES:
            self._right_hand_joint_ids.append(self.robot.joint_names.index(name))
        for name in cfg.left_holder_joint_names:
            self._left_holder_joint_ids.append(self.robot.joint_names.index(name))

        self._right_arm_home = torch.tensor(RIGHT_ARM_HOME_POSE, dtype=torch.float32, device=self.device)
        right_hand_grasp = [RIGHT_HAND_GRASP_JOINT_POS[name] for name in RIGHT_HAND_JOINT_NAMES]
        self._right_hand_grasp = torch.tensor(right_hand_grasp, dtype=torch.float32, device=self.device)
        left_holder_home = [
            LEFT_HOLDER_FIXED_JOINT_POS[name]
            for name in cfg.left_holder_joint_names
        ]
        self._left_holder_home = torch.tensor(left_holder_home, dtype=torch.float32, device=self.device)
        self._right_source_cup_attach_pos_b = torch.tensor(
            cfg.right_source_cup_attach_pos_b, dtype=torch.float32, device=self.device
        )
        self._right_source_cup_attach_quat_b = torch.tensor(
            cfg.right_source_cup_attach_quat_wxyz_b, dtype=torch.float32, device=self.device
        )
        self._left_target_cup_attach_pos_b = torch.tensor(
            cfg.left_target_cup_attach_pos_b, dtype=torch.float32, device=self.device
        )
        self._left_target_cup_attach_quat_b = torch.tensor(
            cfg.left_target_cup_attach_quat_wxyz_b, dtype=torch.float32, device=self.device
        )
        self._bead_spawn_pos_source_cup_b = torch.tensor(
            getattr(cfg, "bead_spawn_pos_source_cup_b", BEAD_SPAWN_POS_SOURCE_CUP_B),
            dtype=torch.float32,
            device=self.device,
        )
        self._bead_spawn_quat_source_cup = torch.tensor(
            getattr(cfg, "bead_spawn_quat_source_cup_wxyz", BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ),
            dtype=torch.float32,
            device=self.device,
        )
        self._right_source_cup_body_id = self.robot.data.body_names.index(cfg.right_source_cup_attach_frame_name)
        self._left_target_cup_body_id = self.robot.data.body_names.index(cfg.left_target_cup_attach_frame_name)
        self._source_cup_pour_point_pos_b = torch.tensor(
            cfg.source_cup_pour_point_pos_b, dtype=torch.float32, device=self.device
        )
        self._target_cup_opening_pos_b = torch.tensor(
            cfg.target_cup_opening_pos_b, dtype=torch.float32, device=self.device
        )
        self._source_cup_pour_axis_b = torch.tensor(cfg.source_cup_pour_axis_b, dtype=torch.float32, device=self.device)
        self._source_cup_up_axis_b = torch.tensor(cfg.source_cup_up_axis_b, dtype=torch.float32, device=self.device)
        self._target_cup_up_axis_b = torch.tensor(cfg.target_cup_up_axis_b, dtype=torch.float32, device=self.device)
        self._world_up_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).unsqueeze(0)

        self._prev_actions = torch.zeros(self.num_envs, NUM_ACTIONS, device=self.device)
        self._last_actions = torch.zeros(self.num_envs, NUM_ACTIONS, device=self.device)
        self._joint_pos_target = self._right_arm_home.unsqueeze(0).repeat(self.num_envs, 1).clone()
        self._obs_buf = torch.zeros(self.num_envs, NUM_OBSERVATIONS, device=self.device)
        self._state_buf = torch.zeros(self.num_envs, self.cfg.num_states, device=self.device)
        self._stable_alignment_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._stable_retention_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_bead_in_target_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._bead_has_entered_target_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._bead_exited_target_after_entry_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._alignment_reward = torch.zeros(self.num_envs, device=self.device)
        self._controlled_tilt_reward = torch.zeros(self.num_envs, device=self.device)
        self._bead_entry_reward = torch.zeros(self.num_envs, device=self.device)
        self._stable_retention_reward = torch.zeros(self.num_envs, device=self.device)
        self._spill_penalty = torch.zeros(self.num_envs, device=self.device)
        self._collision_penalty = torch.zeros(self.num_envs, device=self.device)
        self._smoothness_penalty = torch.zeros(self.num_envs, device=self.device)
        self._major_spill_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._invalid_state_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._bead_in_target_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._bead_in_source_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._bead_spilled_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.right_source_cup = RigidObject(self.cfg.right_source_cup_cfg)
        self.left_target_cup = RigidObject(self.cfg.left_target_cup_cfg)
        self.bead = RigidObject(self.cfg.bead_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["right_source_cup"] = self.right_source_cup
        self.scene.rigid_objects["left_target_cup"] = self.left_target_cup
        self.scene.rigid_objects["bead"] = self.bead

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.scene.clone_environments(copy_from_source=True)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions.copy_(self._last_actions)
        self._last_actions = actions.clamp(-1.0, 1.0)
        self._joint_pos_target = self._right_arm_home.unsqueeze(0) + self.cfg.action_scale * self._last_actions

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_pos_target, joint_ids=self._right_arm_joint_ids)
        right_hand_target = self._right_hand_grasp.unsqueeze(0).expand(self.num_envs, -1)
        self.robot.set_joint_position_target(right_hand_target, joint_ids=self._right_hand_joint_ids)
        left_holder_target = self._left_holder_home.unsqueeze(0).expand(self.num_envs, -1)
        self.robot.set_joint_position_target(left_holder_target, joint_ids=self._left_holder_joint_ids)
        self._update_attached_cups_from_ee()

    def _get_observations(self) -> dict:
        right_cup_pos_w = self.right_source_cup.data.root_pos_w
        right_cup_quat_w = self.right_source_cup.data.root_quat_w
        left_cup_pos_w = self.left_target_cup.data.root_pos_w
        left_cup_quat_w = self.left_target_cup.data.root_quat_w
        arm_joint_pos = self.robot.data.joint_pos[:, self._right_arm_joint_ids]
        arm_joint_vel = self.robot.data.joint_vel[:, self._right_arm_joint_ids]
        rel_cup_pos_source, rel_cup_quat_source = subtract_frame_transforms(
            right_cup_pos_w, right_cup_quat_w, left_cup_pos_w, left_cup_quat_w
        )

        source_pour_point_w = right_cup_pos_w + quat_apply(
            right_cup_quat_w,
            self._source_cup_pour_point_pos_b.unsqueeze(0).expand(self.num_envs, -1),
        )
        target_opening_w = left_cup_pos_w + quat_apply(
            left_cup_quat_w,
            self._target_cup_opening_pos_b.unsqueeze(0).expand(self.num_envs, -1),
        )
        target_opening_in_source, _ = subtract_frame_transforms(
            right_cup_pos_w, right_cup_quat_w, target_opening_w
        )
        pour_point_to_target_opening = target_opening_in_source - self._source_cup_pour_point_pos_b.unsqueeze(0)

        source_lin_speed = torch.norm(self.right_source_cup.data.root_lin_vel_w, dim=-1, keepdim=True)
        source_ang_speed = torch.norm(self.right_source_cup.data.root_ang_vel_w, dim=-1, keepdim=True)
        source_velocity_summary = torch.cat([source_lin_speed, source_ang_speed], dim=-1)

        source_up_axis_w = quat_apply(
            right_cup_quat_w, self._source_cup_up_axis_b.unsqueeze(0).expand(self.num_envs, -1)
        )
        source_pour_axis_w = quat_apply(
            right_cup_quat_w, self._source_cup_pour_axis_b.unsqueeze(0).expand(self.num_envs, -1)
        )
        target_up_axis_w = quat_apply(
            left_cup_quat_w, self._target_cup_up_axis_b.unsqueeze(0).expand(self.num_envs, -1)
        )
        source_to_target_dir_w = self._safe_normalize(target_opening_w - source_pour_point_w)
        tilt_alignment_summary = torch.cat(
            [
                torch.sum(source_up_axis_w * self._world_up_axis.expand(self.num_envs, -1), dim=-1, keepdim=True),
                torch.sum(source_pour_axis_w * source_to_target_dir_w, dim=-1, keepdim=True),
                torch.sum(source_up_axis_w * target_up_axis_w, dim=-1, keepdim=True),
            ],
            dim=-1,
        )

        actor_obs = torch.cat(
            [
                arm_joint_pos,
                arm_joint_vel,
                rel_cup_pos_source,
                rel_cup_quat_source,
                pour_point_to_target_opening,
                source_velocity_summary,
                tilt_alignment_summary,
                self._last_actions,
            ],
            dim=-1,
        )
        if actor_obs.shape[1] != NUM_OBSERVATIONS:
            raise RuntimeError(f"bi_pouring_v1 actor obs dim mismatch: {actor_obs.shape[1]} != {NUM_OBSERVATIONS}")

        bead_pos_env = self.bead.data.root_pos_w - self.scene.env_origins
        bead_lin_vel_w = self.bead.data.root_lin_vel_w
        bead_pos_in_target, _ = subtract_frame_transforms(
            left_cup_pos_w, left_cup_quat_w, self.bead.data.root_pos_w, self.bead.data.root_quat_w
        )
        bead_pos_in_source, _ = subtract_frame_transforms(
            right_cup_pos_w, right_cup_quat_w, self.bead.data.root_pos_w, self.bead.data.root_quat_w
        )
        bead_speed = torch.norm(bead_lin_vel_w, dim=-1)
        right_ee_pos_w = self.robot.data.body_pos_w[:, self._right_source_cup_body_id]
        left_ee_pos_w = self.robot.data.body_pos_w[:, self._left_target_cup_body_id]
        pour_point_to_target_opening_w = target_opening_w - source_pour_point_w
        cup_xy = torch.norm(pour_point_to_target_opening_w[:, :2], dim=-1)
        cup_z = source_pour_point_w[:, 2] - target_opening_w[:, 2]
        stable_alignment = (
            (cup_xy <= self.cfg.stable_alignment_xy_threshold)
            & (cup_z >= self.cfg.stable_alignment_z_min)
            & (cup_z <= self.cfg.stable_alignment_z_max)
            & (tilt_alignment_summary[:, 1] >= 0.5)
        )
        self._stable_alignment_steps = torch.where(
            stable_alignment, self._stable_alignment_steps + 1, torch.zeros_like(self._stable_alignment_steps)
        )
        bead_target_xy = torch.norm(bead_pos_in_target[:, :2], dim=-1)
        bead_source_xy = torch.norm(bead_pos_in_source[:, :2], dim=-1)
        self._bead_in_target_flag = (
            (bead_target_xy <= self.cfg.target_inner_radius)
            & (bead_pos_in_target[:, 2] >= self.cfg.target_inside_z_min)
            & (bead_pos_in_target[:, 2] <= self.cfg.target_entry_z_max)
        )
        self._bead_in_source_flag = (
            (bead_source_xy <= self.cfg.source_inner_radius)
            & (bead_pos_in_source[:, 2] >= self.cfg.source_inside_z_min)
            & (bead_pos_in_source[:, 2] <= self.cfg.source_inside_z_max)
        )
        bead_entry_event = self._bead_in_target_flag & (~self._prev_bead_in_target_flag) & (~self._bead_in_source_flag)
        self._bead_has_entered_target_flag |= bead_entry_event
        bead_exit_after_entry = (
            self._prev_bead_in_target_flag
            & (~self._bead_in_target_flag)
            & self._bead_has_entered_target_flag
        )
        self._bead_exited_target_after_entry_flag |= bead_exit_after_entry
        stable_retention = (
            self._bead_in_target_flag
            & (~self._bead_exited_target_after_entry_flag)
            & (bead_speed <= self.cfg.stable_retention_speed_threshold)
        )
        self._stable_retention_steps = torch.where(
            stable_retention, self._stable_retention_steps + 1, torch.zeros_like(self._stable_retention_steps)
        )
        bead_xy_to_target = torch.norm((self.bead.data.root_pos_w - target_opening_w)[:, :2], dim=-1)
        bead_xy_to_source = torch.norm((self.bead.data.root_pos_w - source_pour_point_w)[:, :2], dim=-1)
        bead_below_target = self.bead.data.root_pos_w[:, 2] <= (target_opening_w[:, 2] + self.cfg.major_spill_z_margin)
        self._major_spill_flag = (
            (~self._bead_in_target_flag)
            & (~self._bead_in_source_flag)
            & bead_below_target
            & (bead_xy_to_target >= self.cfg.major_spill_xy_radius)
            & (bead_xy_to_source >= self.cfg.major_spill_xy_radius)
        )
        self._bead_spilled_flag = self._major_spill_flag | (
            (self.bead.data.root_pos_w[:, 2] <= self.cfg.bead_spill_z_threshold)
            & (~self._bead_in_target_flag)
            & (~self._bead_in_source_flag)
        )
        self._success_flag = (
            self._bead_has_entered_target_flag
            & (~self._bead_exited_target_after_entry_flag)
            & self._bead_in_target_flag
            & (self._stable_retention_steps >= self.cfg.success_retention_steps)
        )

        alignment_xy_term = torch.exp(-torch.square(cup_xy / max(self.cfg.alignment_xy_scale, 1.0e-6)))
        alignment_z_term = torch.exp(
            -torch.square((cup_z - self.cfg.alignment_z_target) / max(self.cfg.alignment_z_scale, 1.0e-6))
        )
        self._alignment_reward = alignment_xy_term * alignment_z_term

        source_up_dot_world = tilt_alignment_summary[:, 0].clamp(-1.0, 1.0)
        pour_axis_to_target = tilt_alignment_summary[:, 1].clamp(-1.0, 1.0)
        upright_progress = torch.clamp(
            (source_up_dot_world - self.cfg.approach_upright_min_cos)
            / max(1.0 - self.cfg.approach_upright_min_cos, 1.0e-6),
            0.0,
            1.0,
        )
        tilt_window = torch.exp(
            -torch.square(
                (source_up_dot_world - self.cfg.controlled_tilt_target_cos)
                / max(self.cfg.controlled_tilt_cos_scale, 1.0e-6)
            )
        )
        pour_progress = torch.clamp(0.5 * (pour_axis_to_target + 1.0), 0.0, 1.0)
        alignment_gate = self._alignment_reward.detach()
        self._controlled_tilt_reward = ((1.0 - alignment_gate) * upright_progress) + (
            alignment_gate * tilt_window * pour_progress
        )
        self._bead_entry_reward = bead_entry_event.float()
        self._stable_retention_reward = stable_retention.float() * (
            0.5 + 0.5 * torch.clamp(1.0 - bead_speed / max(self.cfg.stable_retention_speed_threshold, 1.0e-6), 0.0, 1.0)
        )
        self._spill_penalty = self._bead_spilled_flag.float()
        rim_xy_clearance = torch.norm((source_pour_point_w - target_opening_w)[:, :2], dim=-1)
        rim_vertical_clearance = source_pour_point_w[:, 2] - target_opening_w[:, 2]
        rim_scrape_penalty = self._proximity_penalty(rim_xy_clearance, self.cfg.rim_clearance_threshold) * torch.clamp(
            (self.cfg.stable_alignment_z_min - rim_vertical_clearance) / max(self.cfg.stable_alignment_z_min, 1.0e-6),
            min=0.0,
            max=1.0,
        )
        ee_clearance_penalty = self._proximity_penalty(
            torch.norm(right_ee_pos_w - left_ee_pos_w, dim=-1),
            self.cfg.ee_clearance_threshold,
        )
        cross_cup_ee_penalty = torch.maximum(
            self._proximity_penalty(
                torch.norm(right_cup_pos_w - left_ee_pos_w, dim=-1),
                self.cfg.cup_to_opposite_ee_clearance_threshold,
            ),
            self._proximity_penalty(
                torch.norm(left_cup_pos_w - right_ee_pos_w, dim=-1),
                self.cfg.cup_to_opposite_ee_clearance_threshold,
            ),
        )
        self._collision_penalty = torch.maximum(rim_scrape_penalty, torch.maximum(ee_clearance_penalty, cross_cup_ee_penalty))
        self._smoothness_penalty = torch.mean(torch.square(self._last_actions - self._prev_actions), dim=-1)
        finite_mask = (
            torch.isfinite(actor_obs).all(dim=-1)
            & torch.isfinite(bead_pos_env).all(dim=-1)
            & torch.isfinite(bead_lin_vel_w).all(dim=-1)
        )
        bead_dropped_to_table_or_floor = (
            (~self._bead_in_target_flag)
            & (~self._bead_in_source_flag)
            & (
                (bead_pos_env[:, 2] <= self.cfg.invalid_bead_drop_z_threshold)
                | (bead_pos_env[:, 2] <= self.cfg.invalid_bead_floor_z_threshold)
            )
        )
        bead_out_of_workspace = torch.norm(bead_pos_env[:, :2], dim=-1) >= self.cfg.invalid_bead_xy_threshold
        self._invalid_state_flag = (
            (~finite_mask)
            | (cup_xy >= self.cfg.invalid_cup_xy_threshold)
            | (torch.abs(cup_z) >= self.cfg.invalid_cup_z_threshold)
            | bead_dropped_to_table_or_floor
            | bead_out_of_workspace
        )
        self._prev_bead_in_target_flag.copy_(self._bead_in_target_flag)

        task_flags = torch.stack(
            [
                stable_alignment.float(),
                self._bead_has_entered_target_flag.float(),
                self._success_flag.float(),
            ],
            dim=-1,
        )
        stable_steps = (
            self._stable_alignment_steps.float().unsqueeze(-1) / float(max(1, self.max_episode_length))
        ).clamp(0.0, 1.0)
        spill_flags = torch.stack(
            [
                self._bead_spilled_flag.float(),
                self._bead_exited_target_after_entry_flag.float(),
            ],
            dim=-1,
        )
        critic_obs = torch.cat(
            [
                actor_obs,
                bead_pos_env,
                bead_lin_vel_w,
                task_flags,
                stable_steps,
                spill_flags,
            ],
            dim=-1,
        )
        if critic_obs.shape[1] != self.cfg.num_states:
            raise RuntimeError(f"bi_pouring_v1 critic obs dim mismatch: {critic_obs.shape[1]} != {self.cfg.num_states}")

        self._obs_buf.copy_(actor_obs)
        self._state_buf.copy_(critic_obs)
        return {"policy": self._obs_buf, "critic": self._state_buf}

    def _get_rewards(self) -> torch.Tensor:
        rewards = (
            self.cfg.reward_alignment_weight * self._alignment_reward
            + self.cfg.reward_controlled_tilt_weight * self._controlled_tilt_reward
            + self.cfg.reward_bead_entry_weight * self._bead_entry_reward
            + self.cfg.reward_stable_retention_weight * self._stable_retention_reward
            - self.cfg.penalty_spill_weight * self._spill_penalty
            - self.cfg.penalty_collision_weight * self._collision_penalty
            - self.cfg.penalty_action_smoothness_weight * self._smoothness_penalty
        )
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._major_spill_flag | self._invalid_state_flag
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        if self.cfg.bead_count != 1:
            raise ValueError(f"bi_pouring_v1 requires bead_count == 1, got {self.cfg.bead_count}")

        num_envs = len(env_ids)
        reset_plan = self._build_reset_plan(env_ids)
        full_pos = self.robot.data.default_joint_pos[env_ids].clone()
        full_vel = torch.zeros(num_envs, self.robot.num_joints, device=self.device)

        full_pos[:, self._right_arm_joint_ids] = reset_plan["right_arm_joint_pos"]
        full_pos[:, self._right_hand_joint_ids] = self._right_hand_grasp.unsqueeze(0).expand(num_envs, -1)
        full_pos[:, self._left_holder_joint_ids] = reset_plan["left_holder_joint_pos"]
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(
            full_pos[:, self._right_arm_joint_ids], joint_ids=self._right_arm_joint_ids, env_ids=env_ids
        )
        self.robot.set_joint_position_target(
            full_pos[:, self._right_hand_joint_ids], joint_ids=self._right_hand_joint_ids, env_ids=env_ids
        )
        self.robot.set_joint_position_target(
            full_pos[:, self._left_holder_joint_ids], joint_ids=self._left_holder_joint_ids, env_ids=env_ids
        )
        if hasattr(self.robot, "write_data_to_sim"):
            self.robot.write_data_to_sim()
        if hasattr(self.robot, "update"):
            self.robot.update(0.0)

        object_reset_plan = self._build_object_reset_plan(env_ids)
        right_pose = object_reset_plan["right_source_cup_pose"]
        left_pose = object_reset_plan["left_target_cup_pose"]
        bead_pose = object_reset_plan["bead_pose"]
        zero_vel = torch.zeros(num_envs, 6, device=self.device)
        self.right_source_cup.write_root_pose_to_sim(right_pose, env_ids=env_ids)
        self.right_source_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        self.left_target_cup.write_root_pose_to_sim(left_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        self.bead.write_root_pose_to_sim(bead_pose, env_ids=env_ids)
        self.bead.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        self._last_actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._joint_pos_target[env_ids] = reset_plan["right_arm_joint_pos"]
        self._stable_alignment_steps[env_ids] = 0
        self._stable_retention_steps[env_ids] = 0
        self._prev_bead_in_target_flag[env_ids] = False
        self._bead_has_entered_target_flag[env_ids] = False
        self._bead_exited_target_after_entry_flag[env_ids] = False
        self._alignment_reward[env_ids] = 0.0
        self._controlled_tilt_reward[env_ids] = 0.0
        self._bead_entry_reward[env_ids] = 0.0
        self._stable_retention_reward[env_ids] = 0.0
        self._spill_penalty[env_ids] = 0.0
        self._collision_penalty[env_ids] = 0.0
        self._smoothness_penalty[env_ids] = 0.0
        self._major_spill_flag[env_ids] = False
        self._invalid_state_flag[env_ids] = False
        self._bead_in_target_flag[env_ids] = False
        self._bead_in_source_flag[env_ids] = False
        self._bead_spilled_flag[env_ids] = False
        self._success_flag[env_ids] = False

        # TODO: if left-arm FABRICS reset is enabled later, keep this flow and only
        # replace the left-holder sampler upstream so the cup/bead staging stays shared.

    def _build_reset_plan(self, env_ids: Sequence[int]) -> dict[str, torch.Tensor]:
        right_arm_joint_pos = self._sample_right_arm_init_joint_pos(env_ids)
        left_holder_joint_pos = self._sample_left_holder_init_joint_pos(env_ids)

        return {
            "right_arm_joint_pos": right_arm_joint_pos,
            "left_holder_joint_pos": left_holder_joint_pos,
        }

    def _build_object_reset_plan(self, env_ids: Sequence[int]) -> dict[str, torch.Tensor]:
        right_source_cup_pose = self._sample_right_source_cup_pose(env_ids)
        left_target_cup_pose = self._sample_left_holder_cup_pose(env_ids)
        bead_pose = self._sample_bead_pose_inside_source_cup(right_source_cup_pose)
        return {
            "right_source_cup_pose": right_source_cup_pose,
            "left_target_cup_pose": left_target_cup_pose,
            "bead_pose": bead_pose,
        }

    def _sample_right_arm_init_joint_pos(self, env_ids: Sequence[int]) -> torch.Tensor:
        num_envs = len(env_ids)
        joint_pos = self._right_arm_home.unsqueeze(0).repeat(num_envs, 1)
        if not self.cfg.enable_right_arm_init_noise:
            return joint_pos

        noise_abs = torch.tensor(self.cfg.right_arm_init_joint_noise_abs, dtype=torch.float32, device=self.device)
        noise = (2.0 * torch.rand(num_envs, noise_abs.shape[0], device=self.device) - 1.0) * noise_abs.unsqueeze(0)
        return joint_pos + noise

    def _sample_left_holder_init_joint_pos(self, env_ids: Sequence[int]) -> torch.Tensor:
        num_envs = len(env_ids)
        # Match the 5g_grasp_right_v5 reset-fabric seam naming, but keep the fixed
        # holder path until a reusable left-side FABRICS pose sampler is available.
        if self.cfg.use_left_holder_reset_fabric:
            raise NotImplementedError(
                "bi_pouring_v1 does not yet provide a left-side FABRICS reset sampler; keep use_left_holder_reset_fabric=False."
            )
        # TODO: future curriculum should randomize both left-arm pose and left-gripper
        # aperture jointly so the receiving cup pose remains robust to perception noise.
        return self._left_holder_home.unsqueeze(0).repeat(num_envs, 1)

    def _sample_right_source_cup_pose(self, env_ids: Sequence[int]) -> torch.Tensor:
        return self._compute_attached_root_pose(
            body_id=self._right_source_cup_body_id,
            attach_pos_b=self._right_source_cup_attach_pos_b,
            attach_quat_b=self._right_source_cup_attach_quat_b,
            env_ids=env_ids,
        )

    def _sample_left_holder_cup_pose(self, env_ids: Sequence[int]) -> torch.Tensor:
        # Keep the static attachment definition separate from future holder pose generation.
        # When curriculum randomization is enabled, this method should keep using the same
        # attachment transform while the left-holder joint sampler changes upstream.
        return self._compute_attached_root_pose(
            body_id=self._left_target_cup_body_id,
            attach_pos_b=self._left_target_cup_attach_pos_b,
            attach_quat_b=self._left_target_cup_attach_quat_b,
            env_ids=env_ids,
        )

    def _sample_bead_pose_inside_source_cup(self, source_cup_pose: torch.Tensor) -> torch.Tensor:
        source_cup_pos_w = source_cup_pose[:, :3]
        source_cup_quat_w = source_cup_pose[:, 3:7]
        bead_pos_w = source_cup_pos_w + quat_apply(
            source_cup_quat_w,
            self._bead_spawn_pos_source_cup_b.unsqueeze(0).expand_as(source_cup_pos_w),
        )
        bead_quat_w = quat_mul(
            source_cup_quat_w,
            self._bead_spawn_quat_source_cup.unsqueeze(0).expand(source_cup_quat_w.shape[0], -1),
        )
        return torch.cat([bead_pos_w, bead_quat_w], dim=-1)

    def _compute_attached_root_pose(
        self,
        body_id: int,
        attach_pos_b: torch.Tensor,
        attach_quat_b: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if env_ids is None:
            body_pos_w = self.robot.data.body_pos_w[:, body_id]
            body_quat_w = self.robot.data.body_quat_w[:, body_id]
        else:
            body_pos_w = self.robot.data.body_pos_w[env_ids, body_id]
            body_quat_w = self.robot.data.body_quat_w[env_ids, body_id]

        attach_pos_w = body_pos_w + quat_apply(body_quat_w, attach_pos_b.unsqueeze(0).expand_as(body_pos_w))
        attach_quat_w = quat_mul(body_quat_w, attach_quat_b.unsqueeze(0).expand(body_quat_w.shape[0], -1))
        return torch.cat([attach_pos_w, attach_quat_w], dim=-1)

    def _update_attached_cups_from_ee(self, env_ids: Sequence[int] | None = None) -> None:
        right_pose = self._compute_attached_root_pose(
            body_id=self._right_source_cup_body_id,
            attach_pos_b=self._right_source_cup_attach_pos_b,
            attach_quat_b=self._right_source_cup_attach_quat_b,
            env_ids=env_ids,
        )
        left_pose = self._compute_attached_root_pose(
            body_id=self._left_target_cup_body_id,
            attach_pos_b=self._left_target_cup_attach_pos_b,
            attach_quat_b=self._left_target_cup_attach_quat_b,
            env_ids=env_ids,
        )

        zero_vel = torch.zeros(right_pose.shape[0], 6, device=self.device)
        self.right_source_cup.write_root_pose_to_sim(right_pose, env_ids=env_ids)
        self.right_source_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        self.left_target_cup.write_root_pose_to_sim(left_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        # TODO: if the future left-holder curriculum introduces transient pre-reset motion,
        # move the sampling logic into a dedicated pose generator but keep this attachment
        # writer unchanged so cups still follow the selected tool frames.

    @staticmethod
    def _safe_normalize(vec: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        return vec / torch.clamp(torch.norm(vec, dim=-1, keepdim=True), min=eps)

    @staticmethod
    def _proximity_penalty(distance: torch.Tensor, threshold: float) -> torch.Tensor:
        return torch.clamp((threshold - distance) / max(threshold, 1.0e-6), min=0.0, max=1.0)
