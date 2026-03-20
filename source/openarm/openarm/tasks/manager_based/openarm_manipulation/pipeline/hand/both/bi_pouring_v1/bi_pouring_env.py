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

"""DexPour 계층적 리워드를 적용한 bi_pouring_v1 환경 (PD 직접 관절 제어).

Stage 3 (Transport, ρ=0): 컵 이동 + 직립 유지
Stage 4 (Pour, ρ=1): 45° 틸팅 + pour axis 정렬

컵은 EE에 이미 부착되어 있으므로 Stage 1/2(Approaching/Grasping)는 Skip.
오른팔 7개 관절을 PD 제어기로 직접 제어한다 (FABRICS 제거).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

from .bi_pouring_constants import NUM_ACTIONS, NUM_OBSERVATIONS, NUM_ARM_DOF
from .bi_pouring_env_cfg import BiPouringEnvCfg
from .bi_pouring_preset import (
    BEAD_SPAWN_POS_SOURCE_CUP_B,
    BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ,
    LEFT_HOLDER_FIXED_JOINT_POS,
    RIGHT_HAND_GRASP_JOINT_POS,
    RIGHT_HAND_JOINT_NAMES,
    RIGHT_ARM_POUR_READY_POSE,
)


class BiPouringEnv(DirectRLEnv):
    """DexPour 계층적 리워드(Transport + Pour)를 적용한 포어링 환경.

    컵이 EE에 이미 부착되어 있으므로 Transport(Stage 3) + Pour(Stage 4)만 구현.
    ρ trigger로 두 단계를 전환하며, multi-bead APA를 지원한다.
    오른팔 7D 관절 델타 액션을 PD 제어기로 직접 적용한다.
    """

    cfg: BiPouringEnvCfg

    def __init__(self, cfg: BiPouringEnvCfg, render_mode: str | None = None, **kwargs):
        self._right_arm_joint_ids: list[int] = []
        self._right_hand_joint_ids: list[int] = []
        self._left_holder_joint_ids: list[int] = []

        self._right_hand_grasp = None
        self._left_holder_home = None
        self._arm_default_pos = None   # RIGHT_ARM_POUR_READY_POSE tensor
        self._arm_target = None        # 현재 arm 관절 목표 (누적 delta)
        self._last_actions = None
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

        # DexPour stage buffers
        self._pour_trigger_steps = None
        self._pour_stage_active = None

        # Multi-bead 상태 추적 (N, K)
        self._prev_bead_in_target_flags = None
        self._bead_has_entered_target_flags = None
        self._bead_exited_target_after_entry_flags = None

        # Aggregated bead 상태 (N,)
        self._stable_retention_steps = None
        self._prev_actions = None

        # Reward component buffers
        self._transport_reward = None
        self._pour_reward = None
        self._bead_entry_reward = None
        self._stable_retention_reward = None
        self._spill_penalty = None
        self._collision_penalty = None
        self._smoothness_penalty = None

        # State flags
        self._major_spill_flag = None
        self._invalid_state_flag = None
        self._bead_in_target_flags = None
        self._bead_in_source_flags = None
        self._bead_spilled_flags = None
        self._success_flag = None

        super().__init__(cfg, render_mode, **kwargs)

        for name in cfg.policy_arm_joint_names:
            self._right_arm_joint_ids.append(self.robot.joint_names.index(name))
        for name in RIGHT_HAND_JOINT_NAMES:
            self._right_hand_joint_ids.append(self.robot.joint_names.index(name))
        for name in cfg.left_holder_joint_names:
            self._left_holder_joint_ids.append(self.robot.joint_names.index(name))

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

        K = cfg.bead_count

        self._prev_actions = torch.zeros(self.num_envs, NUM_ACTIONS, device=self.device)
        self._last_actions = torch.zeros(self.num_envs, NUM_ACTIONS, device=self.device)
        self._obs_buf = torch.zeros(self.num_envs, NUM_OBSERVATIONS, device=self.device)
        self._state_buf = torch.zeros(self.num_envs, self.cfg.num_states, device=self.device)

        # PD 관절 제어 버퍼
        self._arm_default_pos = torch.tensor(
            RIGHT_ARM_POUR_READY_POSE, dtype=torch.float32, device=self.device
        )
        self._arm_target = self._arm_default_pos.unsqueeze(0).expand(self.num_envs, -1).clone()

        # DexPour stage
        self._pour_trigger_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._pour_stage_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Stable retention
        self._stable_retention_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Per-bead 상태 (N, K)
        self._prev_bead_in_target_flags = torch.zeros(self.num_envs, K, dtype=torch.bool, device=self.device)
        self._bead_has_entered_target_flags = torch.zeros(self.num_envs, K, dtype=torch.bool, device=self.device)
        self._bead_exited_target_after_entry_flags = torch.zeros(self.num_envs, K, dtype=torch.bool, device=self.device)
        self._bead_in_target_flags = torch.zeros(self.num_envs, K, dtype=torch.bool, device=self.device)
        self._bead_in_source_flags = torch.zeros(self.num_envs, K, dtype=torch.bool, device=self.device)
        self._bead_spilled_flags = torch.zeros(self.num_envs, K, dtype=torch.bool, device=self.device)

        # Reward buffers
        self._transport_reward = torch.zeros(self.num_envs, device=self.device)
        self._pour_reward = torch.zeros(self.num_envs, device=self.device)
        self._bead_entry_reward = torch.zeros(self.num_envs, device=self.device)
        self._stable_retention_reward = torch.zeros(self.num_envs, device=self.device)
        self._spill_penalty = torch.zeros(self.num_envs, device=self.device)
        self._collision_penalty = torch.zeros(self.num_envs, device=self.device)
        self._smoothness_penalty = torch.zeros(self.num_envs, device=self.device)
        self._major_spill_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._invalid_state_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.right_source_cup = RigidObject(self.cfg.right_source_cup_cfg)
        self.left_target_cup = RigidObject(self.cfg.left_target_cup_cfg)

        # Multi-bead: bead_count 만큼 RigidObject 생성
        self.beads: list[RigidObject] = []
        for i in range(self.cfg.bead_count):
            bead_cfg_i = copy.deepcopy(self.cfg.bead_cfg)
            bead_cfg_i.prim_path = f"/World/envs/env_.*/Bead{i}"
            bead_obj = RigidObject(bead_cfg_i)
            self.beads.append(bead_obj)
            self.scene.rigid_objects[f"bead{i}"] = bead_obj

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["right_source_cup"] = self.right_source_cup
        self.scene.rigid_objects["left_target_cup"] = self.left_target_cup

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.scene.clone_environments(copy_from_source=True)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions.copy_(self._last_actions)
        self._last_actions = actions.clamp(-1.0, 1.0)

        # PD 제어: 누적 delta joint position
        # target += action * scale (관절 한계는 USD soft_joint_pos_limit_factor로 제한)
        self._arm_target = self._arm_target + self._last_actions * self.cfg.action_scale

    def _apply_action(self) -> None:
        # 오른팔: PD 목표 관절 위치 적용
        self.robot.set_joint_position_target(self._arm_target, joint_ids=self._right_arm_joint_ids)

        # 오른손: 파지 자세 고정 (write_joint_state_to_sim으로 직접 고정)
        zero_vel_hand = torch.zeros(self.num_envs, len(self._right_hand_joint_ids), device=self.device)
        zero_vel_left = torch.zeros(self.num_envs, len(self._left_holder_joint_ids), device=self.device)
        self.robot.write_joint_state_to_sim(
            self._right_hand_grasp.unsqueeze(0).expand(self.num_envs, -1),
            zero_vel_hand,
            joint_ids=self._right_hand_joint_ids,
        )
        self.robot.write_joint_state_to_sim(
            self._left_holder_home.unsqueeze(0).expand(self.num_envs, -1),
            zero_vel_left,
            joint_ids=self._left_holder_joint_ids,
        )
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

        # ---- Actor obs (36D) ----
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

        # ---- DexPour ρ trigger: Transport → Pour 단계 전환 ----
        pour_point_to_target_opening_w = target_opening_w - source_pour_point_w
        dist_cup = torch.norm(pour_point_to_target_opening_w, dim=-1)
        at_pour = dist_cup < self.cfg.pour_trigger_dist
        self._pour_trigger_steps = torch.where(
            at_pour,
            self._pour_trigger_steps + 1,
            torch.zeros_like(self._pour_trigger_steps),
        )
        # Persistent condition: 컵이 target에서 멀어지면 transport mode로 복귀
        self._pour_stage_active = self._pour_trigger_steps >= self.cfg.pour_trigger_hold_steps

        source_up_dot_world = tilt_alignment_summary[:, 0].clamp(-1.0, 1.0)

        # ---- Proximity reward: stage 무관 항상 활성 ----
        r_cup_dist = torch.exp(-2.0 * dist_cup)

        # ---- Transport stage 한정: 이송 중 기울기 패널티 ----
        p_upright = (1.0 - source_up_dot_world).clamp(0.0, 1.0)
        transport_mask = (~self._pour_stage_active).float()
        self._transport_reward = (
            self.cfg.reward_cup_dist_weight * r_cup_dist
            - transport_mask * self.cfg.penalty_transport_tilt_weight * p_upright
        )

        # ---- Pour stage reward (ρ=1): tilt + align ----
        r_tilt = torch.exp(
            -((source_up_dot_world - self.cfg.pour_tilt_target_cos) / max(self.cfg.pour_tilt_cos_scale, 1.0e-6)) ** 2
        )
        pour_axis_to_target_cos = tilt_alignment_summary[:, 1].clamp(-1.0, 1.0)
        r_align = 0.5 * (1.0 + pour_axis_to_target_cos)
        pour_mask = self._pour_stage_active.float()
        self._pour_reward = pour_mask * (
            self.cfg.reward_tilt_weight * r_tilt
            + self.cfg.reward_align_weight * r_align
        )

        # ---- Multi-bead dynamics ----
        K = self.cfg.bead_count

        all_bead_pos_w = torch.stack([b.data.root_pos_w for b in self.beads], dim=1)
        all_bead_vel_w = torch.stack([b.data.root_lin_vel_w for b in self.beads], dim=1)
        bead_centroid_pos_w = all_bead_pos_w.mean(dim=1)
        bead_centroid_vel_w = all_bead_vel_w.mean(dim=1)

        bead_pos_in_target_list = []
        bead_pos_in_source_list = []
        for i in range(K):
            bead_pos_w_i = self.beads[i].data.root_pos_w
            bead_quat_w_i = self.beads[i].data.root_quat_w
            pos_in_t, _ = subtract_frame_transforms(left_cup_pos_w, left_cup_quat_w, bead_pos_w_i, bead_quat_w_i)
            pos_in_s, _ = subtract_frame_transforms(right_cup_pos_w, right_cup_quat_w, bead_pos_w_i, bead_quat_w_i)
            bead_pos_in_target_list.append(pos_in_t)
            bead_pos_in_source_list.append(pos_in_s)
        bead_pos_in_target = torch.stack(bead_pos_in_target_list, dim=1)
        bead_pos_in_source = torch.stack(bead_pos_in_source_list, dim=1)

        bead_target_xy = torch.norm(bead_pos_in_target[..., :2], dim=-1)
        bead_source_xy = torch.norm(bead_pos_in_source[..., :2], dim=-1)

        self._bead_in_target_flags = (
            (bead_target_xy <= self.cfg.target_inner_radius)
            & (bead_pos_in_target[..., 2] >= self.cfg.target_inside_z_min)
            & (bead_pos_in_target[..., 2] <= self.cfg.target_entry_z_max)
        )
        self._bead_in_source_flags = (
            (bead_source_xy <= self.cfg.source_inner_radius)
            & (bead_pos_in_source[..., 2] >= self.cfg.source_inside_z_min)
            & (bead_pos_in_source[..., 2] <= self.cfg.source_inside_z_max)
        )

        bead_entry_event = (
            self._bead_in_target_flags
            & (~self._prev_bead_in_target_flags)
            & (~self._bead_in_source_flags)
        )
        self._bead_has_entered_target_flags = self._bead_has_entered_target_flags | bead_entry_event
        bead_exit_after_entry = (
            self._prev_bead_in_target_flags
            & (~self._bead_in_target_flags)
            & self._bead_has_entered_target_flags
        )
        self._bead_exited_target_after_entry_flags = self._bead_exited_target_after_entry_flags | bead_exit_after_entry

        bead_new_entry_count = bead_entry_event.sum(dim=-1).float()
        self._bead_entry_reward = bead_new_entry_count

        bead_in_target_count = self._bead_in_target_flags.sum(dim=-1).float()
        bead_in_target_ratio = bead_in_target_count / float(K)

        bead_below_target = all_bead_pos_w[..., 2] <= (
            target_opening_w.unsqueeze(1)[..., 2] + self.cfg.major_spill_z_margin
        )
        bead_xy_to_target = torch.norm(
            (all_bead_pos_w - target_opening_w.unsqueeze(1))[..., :2], dim=-1
        )
        bead_xy_to_source = torch.norm(
            (all_bead_pos_w - source_pour_point_w.unsqueeze(1))[..., :2], dim=-1
        )
        bead_pos_env = all_bead_pos_w - self.scene.env_origins.unsqueeze(1)

        major_spill_per_bead = (
            (~self._bead_in_target_flags)
            & (~self._bead_in_source_flags)
            & bead_below_target
            & (bead_xy_to_target >= self.cfg.major_spill_xy_radius)
            & (bead_xy_to_source >= self.cfg.major_spill_xy_radius)
        )
        self._bead_spilled_flags = major_spill_per_bead | (
            (all_bead_pos_w[..., 2] <= self.cfg.bead_spill_z_threshold)
            & (~self._bead_in_target_flags)
            & (~self._bead_in_source_flags)
        )

        bead_spilled_count = self._bead_spilled_flags.sum(dim=-1).float()
        bead_spill_ratio = bead_spilled_count / float(K)
        self._major_spill_flag = major_spill_per_bead.any(dim=-1)

        centroid_speed = torch.norm(bead_centroid_vel_w, dim=-1)
        any_in_target = self._bead_in_target_flags.any(dim=-1)
        any_exited = self._bead_exited_target_after_entry_flags.any(dim=-1)
        stable_retention = (
            any_in_target
            & (~any_exited)
            & (centroid_speed <= self.cfg.stable_retention_speed_threshold)
        )
        self._stable_retention_steps = torch.where(
            stable_retention,
            self._stable_retention_steps + 1,
            torch.zeros_like(self._stable_retention_steps),
        )
        self._stable_retention_reward = stable_retention.float() * (
            0.5 + 0.5 * torch.clamp(
                1.0 - centroid_speed / max(self.cfg.stable_retention_speed_threshold, 1.0e-6),
                0.0, 1.0,
            )
        )
        self._spill_penalty = bead_spill_ratio

        any_has_entered = self._bead_has_entered_target_flags.any(dim=-1)
        self._success_flag = (
            any_has_entered
            & (~any_exited)
            & any_in_target
            & (self._stable_retention_steps >= self.cfg.success_retention_steps)
        )

        right_ee_pos_w = self.robot.data.body_pos_w[:, self._right_source_cup_body_id]
        left_ee_pos_w  = self.robot.data.body_pos_w[:, self._left_target_cup_body_id]
        rim_xy_clearance = torch.norm((source_pour_point_w - target_opening_w)[:, :2], dim=-1)
        rim_vertical_clearance = source_pour_point_w[:, 2] - target_opening_w[:, 2]
        rim_scrape_penalty = self._proximity_penalty(rim_xy_clearance, self.cfg.rim_clearance_threshold) * torch.clamp(
            (self.cfg.target_inside_z_min - rim_vertical_clearance)
            / max(abs(self.cfg.target_inside_z_min), 1.0e-6),
            min=0.0, max=1.0,
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
        self._collision_penalty = torch.maximum(
            rim_scrape_penalty, torch.maximum(ee_clearance_penalty, cross_cup_ee_penalty)
        )

        self._smoothness_penalty = torch.mean(torch.square(self._last_actions - self._prev_actions), dim=-1)

        # Validity
        cup_xy = torch.norm(pour_point_to_target_opening_w[:, :2], dim=-1)
        cup_z  = torch.abs(pour_point_to_target_opening_w[:, 2])
        finite_mask = torch.isfinite(actor_obs).all(dim=-1) & torch.isfinite(bead_centroid_pos_w).all(dim=-1)
        bead_dropped = (
            (~self._bead_in_target_flags)
            & (~self._bead_in_source_flags)
            & (
                (bead_pos_env[..., 2] <= self.cfg.invalid_bead_drop_z_threshold)
                | (bead_pos_env[..., 2] <= self.cfg.invalid_bead_floor_z_threshold)
            )
        ).any(dim=-1)
        bead_out_of_workspace = (
            torch.norm(bead_pos_env[..., :2], dim=-1) >= self.cfg.invalid_bead_xy_threshold
        ).any(dim=-1)
        self._invalid_state_flag = (
            (~finite_mask)
            | (cup_xy >= self.cfg.invalid_cup_xy_threshold)
            | (cup_z >= self.cfg.invalid_cup_z_threshold)
            | bead_dropped
            | bead_out_of_workspace
        )

        # DEBUG: 첫 번째 env의 termination 원인 주기적 출력
        if (self.episode_length_buf[0].item() % 500 == 1) or (
            self._invalid_state_flag[0].item() and self.episode_length_buf[0].item() < 10
        ):
            print(
                f"[DBG ep={self.episode_length_buf[0].item()}] "
                f"cup_xy={cup_xy[0]:.3f}(th={self.cfg.invalid_cup_xy_threshold}) "
                f"cup_z={cup_z[0]:.3f}(th={self.cfg.invalid_cup_z_threshold}) "
                f"dist={dist_cup[0]:.3f} pour={self._pour_stage_active[0].item()} "
                f"finite={finite_mask[0].item()} invalid={self._invalid_state_flag[0].item()}"
            )

        self._prev_bead_in_target_flags.copy_(self._bead_in_target_flags)

        # ---- Critic obs (51D) ----
        bead_centroid_pos_env = bead_centroid_pos_w - self.scene.env_origins
        task_flags = torch.stack(
            [
                any_has_entered.float(),
                self._success_flag.float(),
                any_exited.float(),
            ],
            dim=-1,
        )
        stable_steps = (
            self._stable_retention_steps.float().unsqueeze(-1) / float(max(1, self.max_episode_length))
        ).clamp(0.0, 1.0)
        spill_flags = torch.stack(
            [
                self._bead_spilled_flags.any(dim=-1).float(),
                self._major_spill_flag.float(),
            ],
            dim=-1,
        )
        spill_ratio = bead_spill_ratio.unsqueeze(-1)

        critic_obs = torch.cat(
            [
                actor_obs,                            # 36
                bead_centroid_pos_env,                # 3
                bead_centroid_vel_w,                  # 3
                bead_in_target_ratio.unsqueeze(-1),   # 1
                self._pour_stage_active.float().unsqueeze(-1),  # 1
                task_flags,                           # 3
                stable_steps,                         # 1
                spill_flags,                          # 2
                spill_ratio,                          # 1
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
            self._transport_reward
            + self._pour_reward
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

        num_envs = len(env_ids)
        full_pos = self.robot.data.default_joint_pos[env_ids].clone()
        full_vel = torch.zeros(num_envs, self.robot.num_joints, device=self.device)

        arm_start = self._arm_default_pos
        full_pos[:, self._right_arm_joint_ids]   = arm_start.unsqueeze(0).expand(num_envs, -1)
        full_pos[:, self._right_hand_joint_ids]  = self._right_hand_grasp.unsqueeze(0).expand(num_envs, -1)
        full_pos[:, self._left_holder_joint_ids] = self._left_holder_home.unsqueeze(0).expand(num_envs, -1)
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        # PD arm target 리셋
        self._arm_target[env_ids] = arm_start.unsqueeze(0).expand(num_envs, -1)

        # 컵 포즈 리셋
        right_pose = self._sample_right_source_cup_pose(env_ids)
        left_pose  = self._sample_left_holder_cup_pose(env_ids)
        zero_vel   = torch.zeros(num_envs, 6, device=self.device)
        self.right_source_cup.write_root_pose_to_sim(right_pose, env_ids=env_ids)
        self.right_source_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        self.left_target_cup.write_root_pose_to_sim(left_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        # Multi-bead 리셋
        for i, bead in enumerate(self.beads):
            jitter = torch.zeros(num_envs, 3, device=self.device)
            if self.cfg.bead_count > 1:
                jitter[:, :2] = (
                    torch.rand(num_envs, 2, device=self.device) - 0.5
                ) * 2.0 * self.cfg.bead_spawn_jitter_xy
            bead_pose = self._sample_bead_pose_inside_source_cup(right_pose, offset=jitter)
            bead.write_root_pose_to_sim(bead_pose, env_ids=env_ids)
            bead.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        # 액션 버퍼 리셋
        self._last_actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # DexPour stage 리셋
        self._pour_trigger_steps[env_ids] = 0
        self._pour_stage_active[env_ids]  = False

        # Stable retention 리셋
        self._stable_retention_steps[env_ids] = 0

        # Per-bead 상태 리셋
        self._prev_bead_in_target_flags[env_ids]             = False
        self._bead_has_entered_target_flags[env_ids]          = False
        self._bead_exited_target_after_entry_flags[env_ids]   = False
        self._bead_in_target_flags[env_ids]                   = False
        self._bead_in_source_flags[env_ids]                   = False
        self._bead_spilled_flags[env_ids]                     = False

        # Reward/flag 리셋
        self._transport_reward[env_ids]      = 0.0
        self._pour_reward[env_ids]           = 0.0
        self._bead_entry_reward[env_ids]     = 0.0
        self._stable_retention_reward[env_ids] = 0.0
        self._spill_penalty[env_ids]         = 0.0
        self._collision_penalty[env_ids]     = 0.0
        self._smoothness_penalty[env_ids]    = 0.0
        self._major_spill_flag[env_ids]      = False
        self._invalid_state_flag[env_ids]    = False
        self._success_flag[env_ids]          = False

    def _sample_right_source_cup_pose(self, env_ids: Sequence[int]) -> torch.Tensor:
        return self._compute_attached_root_pose(
            body_id=self._right_source_cup_body_id,
            attach_pos_b=self._right_source_cup_attach_pos_b,
            attach_quat_b=self._right_source_cup_attach_quat_b,
            env_ids=env_ids,
        )

    def _sample_left_holder_cup_pose(self, env_ids: Sequence[int]) -> torch.Tensor:
        return self._compute_attached_root_pose(
            body_id=self._left_target_cup_body_id,
            attach_pos_b=self._left_target_cup_attach_pos_b,
            attach_quat_b=self._left_target_cup_attach_quat_b,
            env_ids=env_ids,
        )

    def _sample_bead_pose_inside_source_cup(
        self,
        source_cup_pose: torch.Tensor,
        offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source_cup_pos_w  = source_cup_pose[:, :3]
        source_cup_quat_w = source_cup_pose[:, 3:7]
        spawn_offset = self._bead_spawn_pos_source_cup_b.unsqueeze(0).expand_as(source_cup_pos_w)
        if offset is not None:
            spawn_offset = spawn_offset + offset
        bead_pos_w = source_cup_pos_w + quat_apply(source_cup_quat_w, spawn_offset)
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
            body_pos_w  = self.robot.data.body_pos_w[:, body_id]
            body_quat_w = self.robot.data.body_quat_w[:, body_id]
        else:
            body_pos_w  = self.robot.data.body_pos_w[env_ids, body_id]
            body_quat_w = self.robot.data.body_quat_w[env_ids, body_id]

        attach_pos_w  = body_pos_w + quat_apply(body_quat_w, attach_pos_b.unsqueeze(0).expand_as(body_pos_w))
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

    @staticmethod
    def _safe_normalize(vec: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        return vec / torch.clamp(torch.norm(vec, dim=-1, keepdim=True), min=eps)

    @staticmethod
    def _proximity_penalty(distance: torch.Tensor, threshold: float) -> torch.Tensor:
        return torch.clamp((threshold - distance) / max(threshold, 1.0e-6), min=0.0, max=1.0)
