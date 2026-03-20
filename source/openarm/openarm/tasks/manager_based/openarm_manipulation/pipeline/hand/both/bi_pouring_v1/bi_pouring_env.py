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

"""DexPour 계층적 리워드를 적용한 bi_pouring_v1 환경.

Stage 3 (Transport, ρ=0): 컵 이동 + 직립 유지
Stage 4 (Pour, ρ=1): 45° 틸팅 + pour axis 정렬

컵은 EE에 이미 부착되어 있으므로 Stage 1/2(Approaching/Grasping)는 Skip.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from collections.abc import Sequence

import torch

# FABRICS 경로 설정 (hdgp/source/FABRICS/src 우선)
for _parent in Path(__file__).resolve().parents:
    if _parent.name == "source":
        _vendored = _parent / "FABRICS" / "src"
        if _vendored.exists():
            _v = str(_vendored)
            if _v not in sys.path:
                sys.path.insert(0, _v)
        break

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

from .bi_pouring_constants import NUM_ACTIONS, NUM_OBSERVATIONS, NUM_ARM_DOF
from .bi_pouring_env_cfg import BiPouringEnvCfg
from .bi_pouring_preset import (
    ARM_START_POSE,
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
    """

    cfg: BiPouringEnvCfg

    def __init__(self, cfg: BiPouringEnvCfg, render_mode: str | None = None, **kwargs):
        self._right_arm_joint_ids: list[int] = []
        self._right_hand_joint_ids: list[int] = []
        self._left_holder_joint_ids: list[int] = []

        self._right_hand_grasp = None
        self._left_holder_home = None
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
        self._pour_trigger_steps = None       # (N,) 연속 pour 조건 충족 step 수
        self._pour_stage_active = None        # (N,) latch: Transport→Pour 전환

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
        self._bead_in_target_flags = None     # (N, K)
        self._bead_in_source_flags = None     # (N, K)
        self._bead_spilled_flags = None       # (N, K)
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

        # DexPour stage
        self._pour_trigger_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._pour_stage_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Stable retention (bead 기준)
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

        # ---- FABRICS arm 제어 ----
        arm_start = torch.tensor(ARM_START_POSE, dtype=torch.float32, device=self.device)
        self.robot_start_joint_pos = torch.cat(
            [arm_start, self._right_hand_grasp]
        ).unsqueeze(0).repeat(self.num_envs, 1).contiguous()

        self.palm_pose_state = torch.zeros(self.num_envs, 6, device=self.device)
        self._setup_geometric_fabrics()

        _palm_start = self.open_tesollo_fabric.get_palm_pose(
            self.robot_start_joint_pos, "euler_zyx"
        )
        self._init_palm_pose = _palm_start[0].clone()
        self.palm_pose_state.copy_(self._init_palm_pose.unsqueeze(0).expand(self.num_envs, -1))

    def _setup_geometric_fabrics(self) -> None:
        initialize_warp(self.device[-1])
        print("=== BiPouringEnv: Creating Fabrics world ===")
        self.world_model = WorldMeshesModel(
            batch_size=self.num_envs,
            max_objects_per_env=self.cfg.fabrics_max_objects_per_env,
            device=self.device,
            world_filename="open_tesollo_boxes_no_table",
        )
        self.object_ids, self.object_indicator = self.world_model.get_object_ids()
        self.timestep = self.cfg.fabrics_dt
        self.open_tesollo_fabric = OpenArmTeoslloPoseFabric(
            self.num_envs, self.device, self.timestep,
            graph_capturable=False,
            use_hand_fabric=False,
        )
        num_joints = self.open_tesollo_fabric.num_joints  # 27
        self.open_tesollo_integrator = DisplacementIntegrator(self.open_tesollo_fabric)
        self.fabric_q   = self.robot_start_joint_pos.clone().contiguous()
        self.fabric_qd  = torch.zeros(self.num_envs, num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, num_joints, device=self.device)
        self.hand_pca_targets  = torch.zeros(self.num_envs, 5, device=self.device)
        self.palm_pose_targets = torch.zeros(self.num_envs, 6, device=self.device)
        self.fabric_damping_gain = 10.0 * torch.ones(self.num_envs, 1, device=self.device)
        cspace_default = self.open_tesollo_fabric.default_config.clone()
        cspace_default[:, :NUM_ARM_DOF] = torch.tensor(
            ARM_START_POSE, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)
        cspace_default[:, NUM_ARM_DOF:] = self._right_hand_grasp.unsqueeze(0).expand(self.num_envs, -1)
        self.open_tesollo_fabric.default_config.copy_(cspace_default)
        print("=== BiPouringEnv: Fabrics initialized ===")

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

        self.palm_pose_state = self.palm_pose_state + self._last_actions * self.cfg.action_scale
        self.palm_pose_targets.copy_(self.palm_pose_state)
        self.hand_pca_targets.zero_()

        self.open_tesollo_fabric.set_features(
            self.hand_pca_targets,
            self.palm_pose_targets,
            "euler_zyx",
            self.fabric_q.detach(),
            self.fabric_qd.detach(),
            self.object_ids,
            self.object_indicator,
            self.fabric_damping_gain,
        )
        for _ in range(self.cfg.fabric_decimation):
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.open_tesollo_integrator.step(
                self.fabric_q.detach(),
                self.fabric_qd.detach(),
                self.fabric_qdd.detach(),
                self.timestep,
            )

    def _apply_action(self) -> None:
        arm_target = self.fabric_q[:, :NUM_ARM_DOF]
        arm_vel    = self.fabric_qd[:, :NUM_ARM_DOF]
        self.robot.set_joint_position_target(arm_target, joint_ids=self._right_arm_joint_ids)
        self.robot.set_joint_velocity_target(arm_vel,    joint_ids=self._right_arm_joint_ids)

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

        # ---- Actor obs (35D) ----
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
        pour_point_to_target_opening_w = target_opening_w - source_pour_point_w  # (N, 3)
        dist_cup = torch.norm(pour_point_to_target_opening_w, dim=-1)             # (N,)
        at_pour = dist_cup < self.cfg.pour_trigger_dist
        self._pour_trigger_steps = torch.where(
            at_pour,
            self._pour_trigger_steps + 1,
            torch.zeros_like(self._pour_trigger_steps),
        )
        # Persistent condition (래치 없음): 컵이 target에서 멀어지면 transport mode로 복귀
        # 래치(| 누적)를 사용하면 탐색 중 우연히 trigger된 뒤 제자리에서 tilting만 해도
        # pour reward가 계속 주어지는 local optimum이 생긴다.
        self._pour_stage_active = self._pour_trigger_steps >= self.cfg.pour_trigger_hold_steps

        source_up_dot_world = tilt_alignment_summary[:, 0].clamp(-1.0, 1.0)  # (N,)

        # ---- Proximity reward: stage 무관 항상 활성 ----
        # pour stage에서 꺼지면 "제자리 tilting" local optimum 발생.
        # 컵이 target 위에 있을 때만 tilt reward가 의미 있도록 항상 위치 보상 유지.
        r_cup_dist = torch.exp(-2.0 * dist_cup)

        # ---- Transport stage 한정: 이송 중 기울기 패널티 ----
        p_upright = (1.0 - source_up_dot_world).clamp(0.0, 1.0)
        transport_mask = (~self._pour_stage_active).float()
        self._transport_reward = (
            self.cfg.reward_cup_dist_weight * r_cup_dist
            - transport_mask * self.cfg.penalty_transport_tilt_weight * p_upright
        )

        # ---- Pour stage reward (ρ=1): tilt + align ----
        # r_cup_dist는 위에서 항상 포함되므로 여기서는 회전 성분만
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

        # (N, K, 3) 모든 bead 위치/속도 stack
        all_bead_pos_w = torch.stack([b.data.root_pos_w for b in self.beads], dim=1)
        all_bead_vel_w = torch.stack([b.data.root_lin_vel_w for b in self.beads], dim=1)
        bead_centroid_pos_w = all_bead_pos_w.mean(dim=1)   # (N, 3)
        bead_centroid_vel_w = all_bead_vel_w.mean(dim=1)   # (N, 3)

        # 각 bead의 target/source 컵 내 위치 (N, K, 3)
        bead_pos_in_target_list = []
        bead_pos_in_source_list = []
        for i in range(K):
            bead_pos_w_i = self.beads[i].data.root_pos_w       # (N, 3)
            bead_quat_w_i = self.beads[i].data.root_quat_w     # (N, 4)
            pos_in_t, _ = subtract_frame_transforms(left_cup_pos_w, left_cup_quat_w, bead_pos_w_i, bead_quat_w_i)
            pos_in_s, _ = subtract_frame_transforms(right_cup_pos_w, right_cup_quat_w, bead_pos_w_i, bead_quat_w_i)
            bead_pos_in_target_list.append(pos_in_t)
            bead_pos_in_source_list.append(pos_in_s)
        bead_pos_in_target = torch.stack(bead_pos_in_target_list, dim=1)  # (N, K, 3)
        bead_pos_in_source = torch.stack(bead_pos_in_source_list, dim=1)  # (N, K, 3)

        bead_target_xy = torch.norm(bead_pos_in_target[..., :2], dim=-1)  # (N, K)
        bead_source_xy = torch.norm(bead_pos_in_source[..., :2], dim=-1)  # (N, K)

        self._bead_in_target_flags = (
            (bead_target_xy <= self.cfg.target_inner_radius)
            & (bead_pos_in_target[..., 2] >= self.cfg.target_inside_z_min)
            & (bead_pos_in_target[..., 2] <= self.cfg.target_entry_z_max)
        )  # (N, K)
        self._bead_in_source_flags = (
            (bead_source_xy <= self.cfg.source_inner_radius)
            & (bead_pos_in_source[..., 2] >= self.cfg.source_inside_z_min)
            & (bead_pos_in_source[..., 2] <= self.cfg.source_inside_z_max)
        )  # (N, K)

        # 새로 target에 진입한 bead (entry event)
        bead_entry_event = (
            self._bead_in_target_flags
            & (~self._prev_bead_in_target_flags)
            & (~self._bead_in_source_flags)
        )  # (N, K)
        self._bead_has_entered_target_flags = self._bead_has_entered_target_flags | bead_entry_event
        bead_exit_after_entry = (
            self._prev_bead_in_target_flags
            & (~self._bead_in_target_flags)
            & self._bead_has_entered_target_flags
        )
        self._bead_exited_target_after_entry_flags = self._bead_exited_target_after_entry_flags | bead_exit_after_entry

        bead_new_entry_count = bead_entry_event.sum(dim=-1).float()   # (N,)
        self._bead_entry_reward = bead_new_entry_count

        bead_in_target_count = self._bead_in_target_flags.sum(dim=-1).float()  # (N,)
        bead_in_target_ratio = bead_in_target_count / float(K)                 # (N,)

        # Spill 판단
        bead_below_target = all_bead_pos_w[..., 2] <= (
            target_opening_w.unsqueeze(1)[..., 2] + self.cfg.major_spill_z_margin
        )  # (N, K)
        bead_xy_to_target = torch.norm(
            (all_bead_pos_w - target_opening_w.unsqueeze(1))[..., :2], dim=-1
        )  # (N, K)
        bead_xy_to_source = torch.norm(
            (all_bead_pos_w - source_pour_point_w.unsqueeze(1))[..., :2], dim=-1
        )  # (N, K)
        bead_pos_env = all_bead_pos_w - self.scene.env_origins.unsqueeze(1)  # (N, K, 3)

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

        bead_spilled_count = self._bead_spilled_flags.sum(dim=-1).float()   # (N,)
        bead_spill_ratio = bead_spilled_count / float(K)                    # (N,)
        self._major_spill_flag = major_spill_per_bead.any(dim=-1)           # (N,)

        # Stable retention
        centroid_speed = torch.norm(bead_centroid_vel_w, dim=-1)  # (N,)
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

        # Success
        any_has_entered = self._bead_has_entered_target_flags.any(dim=-1)
        self._success_flag = (
            any_has_entered
            & (~any_exited)
            & any_in_target
            & (self._stable_retention_steps >= self.cfg.success_retention_steps)
        )

        # Collision penalty
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

        self._prev_bead_in_target_flags.copy_(self._bead_in_target_flags)

        # ---- Critic obs (50D) ----
        bead_centroid_pos_env = bead_centroid_pos_w - self.scene.env_origins  # (N, 3)
        task_flags = torch.stack(
            [
                any_has_entered.float(),
                self._success_flag.float(),
                any_exited.float(),
            ],
            dim=-1,
        )  # (N, 3)
        stable_steps = (
            self._stable_retention_steps.float().unsqueeze(-1) / float(max(1, self.max_episode_length))
        ).clamp(0.0, 1.0)  # (N, 1)
        spill_flags = torch.stack(
            [
                self._bead_spilled_flags.any(dim=-1).float(),
                self._major_spill_flag.float(),
            ],
            dim=-1,
        )  # (N, 2)
        spill_ratio = bead_spill_ratio.unsqueeze(-1)  # (N, 1)

        critic_obs = torch.cat(
            [
                actor_obs,                            # 35
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

        arm_start = torch.tensor(RIGHT_ARM_POUR_READY_POSE, dtype=torch.float32, device=self.device)
        full_pos[:, self._right_arm_joint_ids]   = arm_start.unsqueeze(0).expand(num_envs, -1)
        full_pos[:, self._right_hand_joint_ids]  = self._right_hand_grasp.unsqueeze(0).expand(num_envs, -1)
        full_pos[:, self._left_holder_joint_ids] = self._left_holder_home.unsqueeze(0).expand(num_envs, -1)
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        self.fabric_q[env_ids]   = self.robot_start_joint_pos[env_ids]
        self.fabric_qd[env_ids].zero_()
        self.fabric_qdd[env_ids].zero_()
        self.palm_pose_state[env_ids] = self._init_palm_pose.unsqueeze(0).expand(num_envs, -1)

        # 컵 포즈 리셋
        right_pose = self._sample_right_source_cup_pose(env_ids)
        left_pose  = self._sample_left_holder_cup_pose(env_ids)
        zero_vel   = torch.zeros(num_envs, 6, device=self.device)
        self.right_source_cup.write_root_pose_to_sim(right_pose, env_ids=env_ids)
        self.right_source_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        self.left_target_cup.write_root_pose_to_sim(left_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        # Multi-bead 리셋: bead 0은 중심, 나머지는 XY jitter 적용
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
