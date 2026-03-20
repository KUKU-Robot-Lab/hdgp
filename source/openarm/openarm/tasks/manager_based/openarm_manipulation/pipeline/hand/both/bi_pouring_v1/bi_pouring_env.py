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

"""bi_pouring_v1 환경 (DirectRLEnv 기반, 5g_grasp_right_v5 패턴).

Stage 3 (Transport, ρ=0): 컵 이동 + 직립 유지
Stage 4 (Pour,      ρ=1): 45° 틸팅 + pour axis 정렬

제어 방식:
  - 오른팔 7D: set_joint_position_target (default_pos + clamp(action) * scale)
  - 오른손 / 왼팔+그리퍼: _apply_action() 에서 write_joint_state_to_sim 으로 강제 고정

컵 부착:
  - right source cup  → rl_dg_ee (오른손 EE)에 kinematic attach
  - left target cup   → ll_dg_ee (왼손 EE)에 kinematic attach
  - 매 physics sub-step마다 EE body pose에서 cup pose 역산하여 write_root_pose_to_sim

DexPour ρ-trigger 및 bead 상태는 env attribute로 관리하고,
mdp/rewards.py, mdp/terminations.py 함수들이 이를 참조한다.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

from .bi_pouring_preset import (
    BEAD_SPAWN_POS_SOURCE_CUP_B,
    BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ,
    LEFT_HOLDER_FIXED_JOINT_POS,
    RIGHT_HAND_GRASP_JOINT_POS,
    RIGHT_HAND_JOINT_NAMES,
    RIGHT_ARM_POUR_READY_POSE,
)


class BiPouringEnv(DirectRLEnv):
    """DexPour Transport+Pour 포어링 환경 (DirectRLEnv).

    초기화 순서:
      super().__init__() 내부:
        1. _setup_scene()  ← Articulation/RigidObject 생성 + scene 등록
        2. sim.reset() + scene 초기화
      __init__() 계속:
        3. _setup_env_refs()  ← joint/body ID, tensor 파라미터 설정
        4. state buffer 초기화
    """

    cfg: "BiPouringEnvCfg"  # noqa: F821

    # ------------------------------------------------------------------
    # Scene 구성 (DirectRLEnv 필수 override)
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        """Articulation/RigidObject 생성 후 scene에 등록."""
        self.robot = Articulation(self.cfg.robot_cfg)
        self.right_source_cup = RigidObject(self.cfg.right_source_cup_cfg)
        self.left_target_cup = RigidObject(self.cfg.left_target_cup_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.beads: list[RigidObject] = [
            RigidObject(self.cfg.bead_cfg) for _ in range(self.cfg.bead_count)
        ]

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["right_source_cup"] = self.right_source_cup
        self.scene.rigid_objects["left_target_cup"] = self.left_target_cup
        self.scene.rigid_objects["table"] = self.table
        for i, bead in enumerate(self.beads):
            self.scene.rigid_objects[f"bead{i}"] = bead

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ------------------------------------------------------------------
    # 초기화
    # ------------------------------------------------------------------

    def __init__(self, cfg: "BiPouringEnvCfg", render_mode: str | None = None, **kwargs):  # noqa: F821
        # super().__init__() 내부:
        #   1. scene = InteractiveScene(cfg.scene)
        #   2. self._setup_scene()
        #   3. sim.reset() + 초기 physics step → robot.data 사용 가능
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # super() 이후: robot.data.body_names, robot.joint_names 등 사용 가능
        self._setup_env_refs()

        K = cfg.bead_count
        N = self.num_envs

        # ---- lab_test1-style rho gate ----
        self._rho = torch.zeros(N, device=self.device)
        self._reset_hold_steps_left = torch.zeros(N, dtype=torch.long, device=self.device)

        # ---- Stable retention ----
        self._stable_retention_steps = torch.zeros(N, dtype=torch.long, device=self.device)

        # ---- Per-bead 상태 (N, K) ----
        self._prev_bead_in_target_flags = torch.zeros(N, K, dtype=torch.bool, device=self.device)
        self._bead_has_entered_target_flags = torch.zeros(N, K, dtype=torch.bool, device=self.device)
        self._bead_exited_target_after_entry_flags = torch.zeros(N, K, dtype=torch.bool, device=self.device)
        self._bead_in_target_flags = torch.zeros(N, K, dtype=torch.bool, device=self.device)
        self._bead_in_source_flags = torch.zeros(N, K, dtype=torch.bool, device=self.device)
        self._bead_spilled_flags = torch.zeros(N, K, dtype=torch.bool, device=self.device)

        # ---- Reward 중간값 (mdp/rewards.py 참조용) ----
        self._r_cup_dist = torch.zeros(N, device=self.device)
        self._prev_dist_cup = torch.zeros(N, device=self.device)
        self._p_upright = torch.zeros(N, device=self.device)
        self._r_tilt = torch.zeros(N, device=self.device)
        self._r_align = torch.zeros(N, device=self.device)
        self._bead_entry_reward = torch.zeros(N, device=self.device)
        self._stable_retention_reward = torch.zeros(N, device=self.device)
        self._spill_penalty = torch.zeros(N, device=self.device)
        self._collision_penalty = torch.zeros(N, device=self.device)
        self._smoothness_penalty = torch.zeros(N, device=self.device)

        # ---- 종료 flag (mdp/terminations.py 참조용) ----
        self._major_spill_flag = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._invalid_state_flag = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._success_flag = torch.zeros(N, dtype=torch.bool, device=self.device)

        # ---- action 버퍼 ----
        num_arm_dof = len(cfg.policy_arm_joint_names)
        self._actions = torch.zeros(N, num_arm_dof, device=self.device)
        self._prev_actions = torch.zeros(N, num_arm_dof, device=self.device)

    # ------------------------------------------------------------------
    # scene refs / tensor 파라미터 설정 (_setup_scene() 이후 호출)
    # ------------------------------------------------------------------

    def _setup_env_refs(self) -> None:
        """joint/body ID 해석 및 tensor 파라미터 초기화."""
        cfg = self.cfg

        # ---- 관절 ID 해석 ----
        self._right_arm_joint_ids: list[int] = [
            self.robot.joint_names.index(n) for n in cfg.policy_arm_joint_names
        ]
        self._right_hand_joint_ids: list[int] = [
            self.robot.joint_names.index(n) for n in RIGHT_HAND_JOINT_NAMES
        ]
        self._left_holder_joint_ids: list[int] = [
            self.robot.joint_names.index(n) for n in cfg.left_holder_joint_names
        ]

        # ---- 고정 자세 tensor ----
        self._right_hand_grasp = torch.tensor(
            [RIGHT_HAND_GRASP_JOINT_POS[n] for n in RIGHT_HAND_JOINT_NAMES],
            dtype=torch.float32, device=self.device,
        )
        self._left_holder_home = torch.tensor(
            [LEFT_HOLDER_FIXED_JOINT_POS[n] for n in cfg.left_holder_joint_names],
            dtype=torch.float32, device=self.device,
        )
        self._right_arm_home = torch.tensor(
            RIGHT_ARM_POUR_READY_POSE,
            dtype=torch.float32,
            device=self.device,
        )

        # ---- 컵 부착 파라미터 ----
        attach_pos_r = torch.tensor(cfg.right_source_cup_attach_pos_b, dtype=torch.float32, device=self.device)
        attach_pos_l = torch.tensor(cfg.left_target_cup_attach_pos_b, dtype=torch.float32, device=self.device)
        self._right_source_cup_attach_quat_b = torch.tensor(
            cfg.right_source_cup_attach_quat_wxyz_b, dtype=torch.float32, device=self.device
        )
        self._left_target_cup_attach_quat_b = torch.tensor(
            cfg.left_target_cup_attach_quat_wxyz_b, dtype=torch.float32, device=self.device
        )

        # ---- 컵 attachment body/frame ----
        self._right_source_cup_body_id, self._right_source_cup_attach_pos_b = self._resolve_attachment_body(
            cfg.right_source_cup_attach_frame_name, attach_pos_r,
        )
        self._left_target_cup_body_id, self._left_target_cup_attach_pos_b = self._resolve_attachment_body(
            cfg.left_target_cup_attach_frame_name, attach_pos_l,
        )
        self._right_palm_body_id, _ = self._resolve_attachment_body(
            "palm_ee", torch.zeros(3, dtype=torch.float32, device=self.device)
        )

        # ---- pour 포인트/축 (obs 함수에서 참조) ----
        self._source_cup_pour_point_pos_b = torch.tensor(
            cfg.source_cup_pour_point_pos_b, dtype=torch.float32, device=self.device
        )
        self._target_cup_opening_pos_b = torch.tensor(
            cfg.target_cup_opening_pos_b, dtype=torch.float32, device=self.device
        )
        self._source_cup_pour_axis_b = torch.tensor(
            cfg.source_cup_pour_axis_b, dtype=torch.float32, device=self.device
        )
        self._source_cup_up_axis_b = torch.tensor(
            cfg.source_cup_up_axis_b, dtype=torch.float32, device=self.device
        )
        self._target_cup_up_axis_b = torch.tensor(
            cfg.target_cup_up_axis_b, dtype=torch.float32, device=self.device
        )
        self._world_up_axis = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # ---- bead spawn ----
        self._bead_spawn_pos_source_cup_b = torch.tensor(
            getattr(cfg, "bead_spawn_pos_source_cup_b", BEAD_SPAWN_POS_SOURCE_CUP_B),
            dtype=torch.float32, device=self.device,
        )
        self._bead_spawn_quat_source_cup = torch.tensor(
            getattr(cfg, "bead_spawn_quat_source_cup_wxyz", BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ),
            dtype=torch.float32, device=self.device,
        )
        self._transport_locked_action_indices = tuple(
            int(i) for i in getattr(cfg, "transport_locked_action_indices", ())
            if 0 <= int(i) < len(self._right_arm_joint_ids)
        )

    # ------------------------------------------------------------------
    # Physics step overrides (DirectRLEnv)
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """이전 action 저장 후 현재 action 갱신."""
        self._prev_actions.copy_(self._actions)
        self._actions.copy_(actions)

    def _apply_action(self) -> None:
        """오른팔 joint target 적용 + 손/왼팔 강제 고정 + 컵 부착."""
        effective_actions = torch.clamp(self._actions, -1.0, 1.0).clone()

        hold_mask = self._reset_hold_steps_left > 0
        if hold_mask.any():
            effective_actions[hold_mask] = 0.0

        transport_mask = (self._rho < 0.5) & (~hold_mask)
        if transport_mask.any() and self._transport_locked_action_indices:
            for action_idx in self._transport_locked_action_indices:
                effective_actions[transport_mask, action_idx] = 0.0

        # 오른팔: reset/home pose 기준으로 action delta를 더한다.
        # default_joint_pos를 쓰면 리셋 직후에도 내부 기본자세로 끌려가며 원치 않는 회전이 생긴다.
        arm_default = self._right_arm_home.unsqueeze(0).expand(self.num_envs, -1)
        arm_target = arm_default + effective_actions * self.cfg.action_scale
        self.robot.set_joint_position_target(arm_target, joint_ids=self._right_arm_joint_ids)

        # 오른손 / 왼팔 강제 고정
        zero_vel_hand = torch.zeros(
            self.num_envs, len(self._right_hand_joint_ids), device=self.device
        )
        zero_vel_left = torch.zeros(
            self.num_envs, len(self._left_holder_joint_ids), device=self.device
        )
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
        self._reset_hold_steps_left = torch.clamp(self._reset_hold_steps_left - 1, min=0)

    # ------------------------------------------------------------------
    # Observations / Rewards / Dones (DirectRLEnv 필수 overrides)
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        """중간값 계산 후 36D policy observation 반환."""
        self._compute_intermediate_values()
        from .mdp import observations as O
        obs = torch.cat(
            [
                O.arm_joint_pos(self),                 # 7
                O.arm_joint_vel(self),                 # 7
                O.cup_relative_pose(self),             # 7
                O.pour_point_to_opening(self),         # 3
                O.source_cup_velocity_summary(self),   # 2
                O.tilt_alignment_summary(self),        # 3
                self._prev_actions,                    # 7
            ],
            dim=-1,
        )  # total 36D
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """각 stage별 reward / penalty 합산."""
        from .mdp import rewards as R
        cfg = self.cfg
        return (
            cfg.reward_cup_dist_weight * R.transport_cup_distance(self)
            - cfg.penalty_transport_tilt_weight * R.transport_upright_penalty(self)
            + cfg.reward_tilt_weight * R.pour_tilt(self)
            + cfg.reward_align_weight * R.pour_align(self)
            + cfg.reward_bead_entry_weight * R.bead_entry(self)
            + cfg.reward_stable_retention_weight * R.bead_stable_retention(self)
            - cfg.penalty_spill_weight * R.bead_spill_penalty(self)
            - cfg.penalty_collision_weight * R.collision_penalty(self)
            - cfg.penalty_action_smoothness_weight * R.action_smoothness_penalty(self)
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """종료 조건: (terminated, time_out)."""
        from .mdp import terminations as T
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = T.major_spill(self) | T.invalid_state(self) | self._success_flag
        return terminated, time_out

    # ------------------------------------------------------------------
    # DexPour 중간값 계산 (reward / termination 함수들이 참조)
    # ------------------------------------------------------------------

    def _compute_intermediate_values(self) -> None:
        """매 step 한 번: ρ-trigger, bead 상태, reward 중간값을 갱신."""
        cfg = self.cfg
        N = self.num_envs

        right_cup_pos_w = self.right_source_cup.data.root_pos_w
        right_cup_quat_w = self.right_source_cup.data.root_quat_w
        left_cup_pos_w = self.left_target_cup.data.root_pos_w
        left_cup_quat_w = self.left_target_cup.data.root_quat_w

        source_pour_point_w = right_cup_pos_w + quat_apply(
            right_cup_quat_w,
            self._source_cup_pour_point_pos_b.unsqueeze(0).expand(N, -1),
        )
        target_opening_w = left_cup_pos_w + quat_apply(
            left_cup_quat_w,
            self._target_cup_opening_pos_b.unsqueeze(0).expand(N, -1),
        )
        source_up_w = quat_apply(
            right_cup_quat_w,
            self._source_cup_up_axis_b.unsqueeze(0).expand(N, -1),
        )
        source_pour_axis_w = quat_apply(
            right_cup_quat_w,
            self._source_cup_pour_axis_b.unsqueeze(0).expand(N, -1),
        )

        pour_point_to_target_w = target_opening_w - source_pour_point_w
        dist_cup = torch.norm(pour_point_to_target_w, dim=-1)

        # lab_test1-style rho: 현재 컵 기하 상태로 매 step 직접 계산
        cup_xy = torch.norm(pour_point_to_target_w[:, :2], dim=-1)
        cup_z_clearance = source_pour_point_w[:, 2] - target_opening_w[:, 2]
        self._rho = (
            (cup_xy <= cfg.rho_xy_threshold)
            & (cup_z_clearance > cfg.rho_z_min)
            & (cup_z_clearance < cfg.rho_z_max)
        ).float()

        # source_up · world_up
        world_up = self._world_up_axis.expand(N, -1)
        source_up_dot_world = torch.sum(source_up_w * world_up, dim=-1).clamp(-1.0, 1.0)

        # Transport 중간값: 절대 거리 기반 shaping.
        # 멀수록 약하고, 가까워질수록 즉시 보상이 증가해 리셋 직후부터 이송을 유도한다.
        self._r_cup_dist = torch.exp(-cfg.transport_dist_temperature * dist_cup)
        self._prev_dist_cup = dist_cup.clone()
        self._p_upright = (1.0 - source_up_dot_world).clamp(0.0, 1.0)

        # Pour 중간값: pour_axis(컵 X축)가 아래(-Z)를 향할수록 reward
        # lab_test1 방식: (1 + dot(pour_axis, target_dir)) / 2
        # 직립 시 pour_axis ≈ world +X → z성분=0 → reward=0.5
        # 완전 기울이면 pour_axis → world -Z → reward=1.0
        pour_axis_down_dot = (-source_pour_axis_w[:, 2]).clamp(-1.0, 1.0)
        self._r_tilt = (1.0 + pour_axis_down_dot) * 0.5
        to_target_dir = pour_point_to_target_w / torch.clamp(
            torch.norm(pour_point_to_target_w, dim=-1, keepdim=True), min=1e-6
        )
        pour_axis_cos = torch.sum(source_pour_axis_w * to_target_dir, dim=-1).clamp(-1.0, 1.0)
        self._r_align = 0.5 * (1.0 + pour_axis_cos)

        # ---- Multi-bead dynamics ----
        K = cfg.bead_count
        all_bead_pos_w = torch.stack([b.data.root_pos_w for b in self.beads], dim=1)
        all_bead_vel_w = torch.stack([b.data.root_lin_vel_w for b in self.beads], dim=1)

        bead_pos_in_target_list = []
        bead_pos_in_source_list = []
        for i in range(K):
            bead_pos_w_i = self.beads[i].data.root_pos_w
            bead_quat_w_i = self.beads[i].data.root_quat_w
            pos_in_t, _ = subtract_frame_transforms(
                left_cup_pos_w, left_cup_quat_w, bead_pos_w_i, bead_quat_w_i
            )
            pos_in_s, _ = subtract_frame_transforms(
                right_cup_pos_w, right_cup_quat_w, bead_pos_w_i, bead_quat_w_i
            )
            bead_pos_in_target_list.append(pos_in_t)
            bead_pos_in_source_list.append(pos_in_s)

        bead_pos_in_target = torch.stack(bead_pos_in_target_list, dim=1)
        bead_pos_in_source = torch.stack(bead_pos_in_source_list, dim=1)

        bead_target_xy = torch.norm(bead_pos_in_target[..., :2], dim=-1)
        bead_source_xy = torch.norm(bead_pos_in_source[..., :2], dim=-1)

        self._bead_in_target_flags = (
            (bead_target_xy <= cfg.target_inner_radius)
            & (bead_pos_in_target[..., 2] >= cfg.target_inside_z_min)
            & (bead_pos_in_target[..., 2] <= cfg.target_entry_z_max)
        )
        self._bead_in_source_flags = (
            (bead_source_xy <= cfg.source_inner_radius)
            & (bead_pos_in_source[..., 2] >= cfg.source_inside_z_min)
            & (bead_pos_in_source[..., 2] <= cfg.source_inside_z_max)
        )

        bead_entry_event = (
            self._bead_in_target_flags
            & (~self._prev_bead_in_target_flags)
            & (~self._bead_in_source_flags)
        )
        self._bead_has_entered_target_flags |= bead_entry_event
        bead_exit_after_entry = (
            self._prev_bead_in_target_flags
            & (~self._bead_in_target_flags)
            & self._bead_has_entered_target_flags
        )
        self._bead_exited_target_after_entry_flags |= bead_exit_after_entry

        self._bead_entry_reward = bead_entry_event.sum(dim=-1).float()

        bead_spill_ratio = self._compute_bead_spill(
            all_bead_pos_w, target_opening_w, source_pour_point_w, cfg
        )
        self._spill_penalty = bead_spill_ratio

        # stable retention
        bead_centroid_vel_w = all_bead_vel_w.mean(dim=1)
        centroid_speed = torch.norm(bead_centroid_vel_w, dim=-1)
        any_in_target = self._bead_in_target_flags.any(dim=-1)
        any_exited = self._bead_exited_target_after_entry_flags.any(dim=-1)
        stable_retention = (
            any_in_target
            & (~any_exited)
            & (centroid_speed <= cfg.stable_retention_speed_threshold)
        )
        self._stable_retention_steps = torch.where(
            stable_retention,
            self._stable_retention_steps + 1,
            torch.zeros_like(self._stable_retention_steps),
        )
        self._stable_retention_reward = stable_retention.float() * (
            0.5 + 0.5 * torch.clamp(
                1.0 - centroid_speed / max(cfg.stable_retention_speed_threshold, 1e-6),
                0.0, 1.0,
            )
        )

        any_has_entered = self._bead_has_entered_target_flags.any(dim=-1)
        self._success_flag = (
            any_has_entered
            & (~any_exited)
            & any_in_target
            & (self._stable_retention_steps >= cfg.success_retention_steps)
        )

        # collision penalty
        right_ee_pos_w = self.robot.data.body_pos_w[:, self._right_source_cup_body_id]
        left_ee_pos_w = self.robot.data.body_pos_w[:, self._left_target_cup_body_id]
        rim_xy_clearance = torch.norm(
            (source_pour_point_w - target_opening_w)[:, :2], dim=-1
        )
        rim_vertical_clearance = source_pour_point_w[:, 2] - target_opening_w[:, 2]
        rim_scrape = self._proximity_penalty(
            rim_xy_clearance, cfg.rim_clearance_threshold
        ) * torch.clamp(
            (cfg.target_inside_z_min - rim_vertical_clearance)
            / max(abs(cfg.target_inside_z_min), 1e-6),
            min=0.0, max=1.0,
        )
        ee_pen = self._proximity_penalty(
            torch.norm(right_ee_pos_w - left_ee_pos_w, dim=-1),
            cfg.ee_clearance_threshold,
        )
        cross_pen = torch.maximum(
            self._proximity_penalty(
                torch.norm(right_cup_pos_w - left_ee_pos_w, dim=-1),
                cfg.cup_to_opposite_ee_clearance_threshold,
            ),
            self._proximity_penalty(
                torch.norm(left_cup_pos_w - right_ee_pos_w, dim=-1),
                cfg.cup_to_opposite_ee_clearance_threshold,
            ),
        )
        self._collision_penalty = torch.maximum(rim_scrape, torch.maximum(ee_pen, cross_pen))

        # action smoothness
        self._smoothness_penalty = torch.mean(
            torch.square(self._actions - self._prev_actions), dim=-1
        )

        # invalid state
        bead_pos_env = all_bead_pos_w - self.scene.env_origins.unsqueeze(1)
        cup_z = torch.abs(pour_point_to_target_w[:, 2])
        bead_dropped = (
            (~self._bead_in_target_flags)
            & (~self._bead_in_source_flags)
            & (
                (bead_pos_env[..., 2] <= cfg.invalid_bead_drop_z_threshold)
                | (bead_pos_env[..., 2] <= cfg.invalid_bead_floor_z_threshold)
            )
        ).any(dim=-1)
        bead_out = (
            torch.norm(bead_pos_env[..., :2], dim=-1) >= cfg.invalid_bead_xy_threshold
        ).any(dim=-1)
        self._invalid_state_flag = (
            (cup_xy >= cfg.invalid_cup_xy_threshold)
            | (cup_z >= cfg.invalid_cup_z_threshold)
            | bead_dropped
            | bead_out
        )

        # DEBUG
        if (self.episode_length_buf[0].item() % 500 == 1) or (
            self._invalid_state_flag[0].item() and self.episode_length_buf[0].item() < 10
        ):
            print(
                f"[DBG ep={self.episode_length_buf[0].item()}] "
                f"dist={dist_cup[0]:.3f} rho={self._rho[0].item():.0f} "
                f"cup_xy={cup_xy[0]:.3f} cup_z={cup_z[0]:.3f} "
                f"invalid={self._invalid_state_flag[0].item()}"
            )

        self._prev_bead_in_target_flags.copy_(self._bead_in_target_flags)

    def _compute_bead_spill(
        self,
        all_bead_pos_w: torch.Tensor,
        target_opening_w: torch.Tensor,
        source_pour_point_w: torch.Tensor,
        cfg,
    ) -> torch.Tensor:
        K = cfg.bead_count
        bead_below_target = all_bead_pos_w[..., 2] <= (
            target_opening_w.unsqueeze(1)[..., 2] + cfg.major_spill_z_margin
        )
        bead_xy_to_target = torch.norm(
            (all_bead_pos_w - target_opening_w.unsqueeze(1))[..., :2], dim=-1
        )
        bead_xy_to_source = torch.norm(
            (all_bead_pos_w - source_pour_point_w.unsqueeze(1))[..., :2], dim=-1
        )
        major_spill_per_bead = (
            (~self._bead_in_target_flags)
            & (~self._bead_in_source_flags)
            & bead_below_target
            & (bead_xy_to_target >= cfg.major_spill_xy_radius)
            & (bead_xy_to_source >= cfg.major_spill_xy_radius)
        )
        self._bead_spilled_flags = major_spill_per_bead | (
            (all_bead_pos_w[..., 2] <= cfg.bead_spill_z_threshold)
            & (~self._bead_in_target_flags)
            & (~self._bead_in_source_flags)
        )
        self._major_spill_flag = major_spill_per_bead.any(dim=-1)
        return self._bead_spilled_flags.sum(dim=-1).float() / float(K)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        num_reset = len(env_ids)
        full_pos = self.robot.data.default_joint_pos[env_ids].clone()
        full_vel = torch.zeros(num_reset, self.robot.num_joints, device=self.device)

        arm_start = torch.tensor(
            RIGHT_ARM_POUR_READY_POSE, dtype=torch.float32, device=self.device
        )
        full_pos[:, self._right_arm_joint_ids] = arm_start.unsqueeze(0).expand(num_reset, -1)
        full_pos[:, self._right_hand_joint_ids] = (
            self._right_hand_grasp.unsqueeze(0).expand(num_reset, -1)
        )
        full_pos[:, self._left_holder_joint_ids] = (
            self._left_holder_home.unsqueeze(0).expand(num_reset, -1)
        )
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        # 컵 포즈 리셋
        right_pose = self._compute_attached_root_pose(
            self._right_source_cup_body_id,
            self._right_source_cup_attach_pos_b,
            self._right_source_cup_attach_quat_b,
            env_ids=env_ids,
        )
        left_pose = self._compute_attached_root_pose(
            self._left_target_cup_body_id,
            self._left_target_cup_attach_pos_b,
            self._left_target_cup_attach_quat_b,
            env_ids=env_ids,
        )
        zero_vel = torch.zeros(num_reset, 6, device=self.device)
        self.right_source_cup.write_root_pose_to_sim(right_pose, env_ids=env_ids)
        self.right_source_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        self.left_target_cup.write_root_pose_to_sim(left_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        # bead 리셋
        for bead in self.beads:
            jitter = torch.zeros(num_reset, 3, device=self.device)
            if self.cfg.bead_count > 1:
                jitter[:, :2] = (
                    torch.rand(num_reset, 2, device=self.device) - 0.5
                ) * 2.0 * self.cfg.bead_spawn_jitter_xy
            bead_pose = self._sample_bead_pose_inside_source_cup(right_pose, offset=jitter)
            bead.write_root_pose_to_sim(bead_pose, env_ids=env_ids)
            bead.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        # 상태 버퍼 리셋
        self._rho[env_ids] = 0.0
        self._reset_hold_steps_left[env_ids] = self.cfg.reset_hold_steps
        self._stable_retention_steps[env_ids] = 0
        self._prev_bead_in_target_flags[env_ids] = False
        self._bead_has_entered_target_flags[env_ids] = False
        self._bead_exited_target_after_entry_flags[env_ids] = False
        self._bead_in_target_flags[env_ids] = False
        self._bead_in_source_flags[env_ids] = False
        self._bead_spilled_flags[env_ids] = False
        self._major_spill_flag[env_ids] = False
        self._invalid_state_flag[env_ids] = False
        self._success_flag[env_ids] = False
        self._r_cup_dist[env_ids] = 0.0
        self._prev_dist_cup[env_ids] = 0.0
        self._p_upright[env_ids] = 0.0
        self._r_tilt[env_ids] = 0.0
        self._r_align[env_ids] = 0.0
        self._bead_entry_reward[env_ids] = 0.0
        self._stable_retention_reward[env_ids] = 0.0
        self._spill_penalty[env_ids] = 0.0
        self._collision_penalty[env_ids] = 0.0
        self._smoothness_penalty[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    def _update_attached_cups_from_ee(self, env_ids: Sequence[int] | None = None) -> None:
        right_pose = self._compute_attached_root_pose(
            self._right_source_cup_body_id,
            self._right_source_cup_attach_pos_b,
            self._right_source_cup_attach_quat_b,
            env_ids=env_ids,
        )
        left_pose = self._compute_attached_root_pose(
            self._left_target_cup_body_id,
            self._left_target_cup_attach_pos_b,
            self._left_target_cup_attach_quat_b,
            env_ids=env_ids,
        )
        zero_vel = torch.zeros(right_pose.shape[0], 6, device=self.device)
        self.right_source_cup.write_root_pose_to_sim(right_pose, env_ids=env_ids)
        self.right_source_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        self.left_target_cup.write_root_pose_to_sim(left_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

    def _resolve_attachment_body(
        self,
        requested_body_name: str,
        attach_pos_b: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        """Resolve desired attachment frames to actual simulated bodies."""
        body_names = self.robot.data.body_names
        alias_offsets: dict[str, list[tuple[str, tuple[float, float, float]]]] = {
            # desired-frame origin expressed in the candidate actual body frame.
            "rl_dg_ee": [
                ("rl_dg_ee", (0.0, 0.0, 0.0)),
                ("palm_ee", (0.0, 0.0, 0.0)),
                ("rl_dg_palm", (0.028, 0.0, 0.04)),
            ],
            "palm_ee": [
                ("palm_ee", (0.0, 0.0, 0.0)),
                ("rl_dg_ee", (0.0, 0.0, 0.0)),
                ("rl_dg_palm", (0.028, 0.0, 0.04)),
            ],
            "ll_dg_ee": [
                ("ll_dg_ee", (0.0, 0.0, 0.0)),
                ("openarm_left_hand_tcp", (0.0, 0.0, -0.08)),
                ("openarm_left_hand", (0.0, 0.0, 0.0)),
            ],
            "openarm_left_hand": [
                ("openarm_left_hand", (0.0, 0.0, 0.0)),
                ("openarm_left_hand_tcp", (0.0, 0.0, -0.08)),
                ("ll_dg_ee", (0.0, 0.0, -0.08)),
            ],
        }
        candidates = alias_offsets.get(requested_body_name, [(requested_body_name, (0.0, 0.0, 0.0))])
        for body_name, desired_origin_in_body in candidates:
            if body_name in body_names:
                resolved_pos_b = attach_pos_b + torch.tensor(
                    desired_origin_in_body, dtype=attach_pos_b.dtype, device=attach_pos_b.device
                )
                return body_names.index(body_name), resolved_pos_b
        candidate_names = [name for name, _ in candidates]
        raise ValueError(
            f"Attachment frame '{requested_body_name}' was not found. "
            f"Tried body names: {candidate_names}. Available bodies: {body_names}"
        )

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

        attach_pos_w = body_pos_w + quat_apply(
            body_quat_w, attach_pos_b.unsqueeze(0).expand_as(body_pos_w)
        )
        attach_quat_w = quat_mul(
            body_quat_w, attach_quat_b.unsqueeze(0).expand(body_quat_w.shape[0], -1)
        )
        return torch.cat([attach_pos_w, attach_quat_w], dim=-1)

    def _sample_bead_pose_inside_source_cup(
        self,
        source_cup_pose: torch.Tensor,
        offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source_cup_pos_w = source_cup_pose[:, :3]
        source_cup_quat_w = source_cup_pose[:, 3:7]
        spawn_offset = self._bead_spawn_pos_source_cup_b.unsqueeze(0).expand_as(source_cup_pos_w)
        if offset is not None:
            spawn_offset = spawn_offset + offset
        bead_pos_w = source_cup_pos_w + quat_apply(source_cup_quat_w, spawn_offset)
        bead_quat_w = quat_mul(
            source_cup_quat_w,
            self._bead_spawn_quat_source_cup.unsqueeze(0).expand(
                source_cup_quat_w.shape[0], -1
            ),
        )
        return torch.cat([bead_pos_w, bead_quat_w], dim=-1)

    @staticmethod
    def _proximity_penalty(distance: torch.Tensor, threshold: float) -> torch.Tensor:
        return torch.clamp((threshold - distance) / max(threshold, 1e-6), min=0.0, max=1.0)
