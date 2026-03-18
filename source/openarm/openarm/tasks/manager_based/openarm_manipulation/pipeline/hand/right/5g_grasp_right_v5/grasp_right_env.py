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

"""환경 클래스: 5g_grasp_right_v5

v5: FABRICS pre-grasp reset + 정책 손가락 grasp formation + Scripted lift checker.

핵심 차이 (v4 대비):
  - Action: 8D = 5D per-finger lerp + 3D palm xyz residual (orientation 고정)
  - Pre-grasp: reset에서 FABRICS로 cup-relative 위치 형성
  - Contact: 물리 ContactSensor (fingertip 5개) 기반
  - Episode: 6s = 5s grasp phase + 1s scripted lift checker
  - Reward: contact-rich (contact_reward, contact_delta, enclosure, opposition)
  - ADR: contact_delta_weight (3→1), enclosure_weight (2→3)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from collections.abc import Sequence

import torch

# Fabrics 경로 설정 (hdgp/source/FABRICS/src 우선)
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
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap

from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR
from .grasp_right_env_cfg import GraspRightEnvCfg
from .grasp_adr import GraspADR
from .grasp_right_constants import (
    NUM_ARM_DOF,
    NUM_HAND_DOF,
    NUM_FINGERTIPS,
    NUM_OBSERVATIONS,
    NUM_DISTAL_SENSORS,
    NUM_MIDDLE_SENSORS,
    NUM_CRITIC_OBSERVATIONS,
    GRASP_PHASE_STEPS,
    LIFT_PHASE_STEPS,
    LIFT_Z_DELTA,
    CONTACT_FORCE_THRESHOLD,
    CONTACT_FORCE_MAX,
    MIN_CONTACTS_FOR_SUCCESS,
    PREGRASP_FABRICS_STEPS,
    ARM_START_POSE,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
)
from .grasp_right_preset import (
    FABRIC_HAND_BODY_NAMES,
    LEFT_ARM_REST_JOINT_POS,
    RIGHT_ACTUATED_JOINT_NAMES,
    HAND_START_POSE,
    HAND_APPROACH_POSE,
    HAND_GRASP_POSE,
    OBJECT_GOAL_POS,
)
from .grasp_right_utils import scale, to_torch


class GraspRightEnv(DirectRLEnv):
    """OpenArm+Teosllo 오른손 파지 환경 (v5).

    Action: 5D
      [0:5] per-finger lerp (thumb, index, middle, ring, pinky)
             action=0  → HAND_GRASP_POSE
             action=+1 → HAND_GRASP + (GRASP - APPROACH) (더 닫힘)
             action=-1 → HAND_APPROACH_POSE (완전 열림)

    Pre-grasp reset:
      FABRICS를 pregrasp_fabric_steps 동안 실행 → arm이 cup-relative pre-grasp 위치로 이동.
      pregrasp arm joint 위치를 저장. 추가로 LIFT_Z_DELTA만큼 높인 palm target으로
      prelift arm joint 위치를 계산해 저장.

    Episode structure:
      step 0 ~ GRASP_PHASE_STEPS-1: Grasp Phase
        - arm joints = pregrasp 고정 (set_joint_position_target)
        - finger joints = 정책 제어 (5D per-finger lerp)
      step GRASP_PHASE_STEPS ~ 끝:  Lift Phase
        - arm joints = pregrasp → prelift 선형 보간
        - finger joints = Grasp Phase 종료 시점 포지션으로 고정

    Success: Lift Phase 이후 cup_z > spawn_z + lift_success_height AND num_contacts >= 2
    """

    cfg: GraspRightEnvCfg

    def __init__(self, cfg: GraspRightEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ----------------------------------------------------------------
        # DOF index 설정
        # ----------------------------------------------------------------
        self.actuated_dof_indices: list[int] = []
        for name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.robot.joint_names.index(name))

        self.left_arm_dof_indices: list[int] = []
        for name in cfg.left_arm_joint_names:
            if name in self.robot.joint_names:
                self.left_arm_dof_indices.append(self.robot.joint_names.index(name))

        # arm/hand 분리 (robot.data.joint_pos 인덱스)
        self.arm_dof_indices  = self.actuated_dof_indices[:NUM_ARM_DOF]   # list[int]
        self.hand_dof_indices = self.actuated_dof_indices[NUM_ARM_DOF:]   # list[int]

        # ----------------------------------------------------------------
        # Palm pose 워크스페이스
        # ----------------------------------------------------------------
        self.palm_mins = to_torch(PALM_POSE_MINS_FUNC(cfg.max_pose_angle), device=self.device)
        self.palm_maxs = to_torch(PALM_POSE_MAXS_FUNC(cfg.max_pose_angle), device=self.device)

        # ----------------------------------------------------------------
        # Hand poses (per-finger lerp용)
        # hand_open_pose = HAND_APPROACH_POSE: thumb _2=-1.57, 나머지=0
        #   lerp(t=0) = approach pose  →  lerp(t=1) = grasp pose
        # ----------------------------------------------------------------
        self.hand_open_pose    = to_torch(HAND_APPROACH_POSE, device=self.device)  # (20,) lerp 시작
        self.hand_grasp_pose   = to_torch(HAND_GRASP_POSE,    device=self.device)  # (20,) lerp 끝
        self.hand_approach_buf = to_torch(HAND_APPROACH_POSE, device=self.device)  # reset 전용 참조

        # ----------------------------------------------------------------
        # 로봇 시작 자세 (arm: ARM_START_POSE, hand: HAND_APPROACH_POSE)
        # ----------------------------------------------------------------
        arm_start  = to_torch(ARM_START_POSE,     device=self.device)   # (7,)
        hand_start = to_torch(HAND_APPROACH_POSE, device=self.device)   # (20,)
        robot_start = torch.cat([arm_start, hand_start], dim=0)          # (27,)
        self.robot_start_joint_pos = (
            robot_start.unsqueeze(0).repeat(self.num_envs, 1).contiguous()
        )

        # ----------------------------------------------------------------
        # 왼팔 고정 자세
        # ----------------------------------------------------------------
        left_vals = [
            LEFT_ARM_REST_JOINT_POS.get(self.robot.joint_names[idx], 0.0)
            for idx in self.left_arm_dof_indices
        ]
        self.left_arm_zero_pos = (
            to_torch(left_vals, device=self.device)
            .unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.left_arm_zero_vel = torch.zeros(
            self.num_envs, len(self.left_arm_dof_indices), device=self.device
        )

        # ----------------------------------------------------------------
        # 목표 위치
        # ----------------------------------------------------------------
        self.object_goal = (
            to_torch(OBJECT_GOAL_POS, device=self.device)
            .unsqueeze(0).repeat(self.num_envs, 1)
        )

        # Pregrasp offset (cup 기준 palm target offset)
        self.pregrasp_offset = to_torch(
            [cfg.pregrasp_offset_x, cfg.pregrasp_offset_y, cfg.pregrasp_offset_z],
            device=self.device,
        )

        # ----------------------------------------------------------------
        # 중간값 버퍼
        # ----------------------------------------------------------------
        self.object_pos      = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_rot      = torch.zeros(self.num_envs, 4, device=self.device)
        self.object_init_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.palm_center_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.fingertip_pos   = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.actions         = torch.zeros(self.num_envs, cfg.num_actions, device=self.device)

        # ----------------------------------------------------------------
        # Pregrasp / Lift 버퍼 (reset에서 계산)
        # ----------------------------------------------------------------
        self.pregrasp_palm_pos_buf    = torch.zeros(self.num_envs, 3, device=self.device)
        self.pregrasp_palm_orient_buf = torch.zeros(self.num_envs, 3, device=self.device)
        # arm joint: pregrasp 고정값 (Grasp phase 내내), prelift 목표값 (Lift phase 끝)
        self.pregrasp_arm_pos_buf = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        self.prelift_arm_pos_buf  = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        # finger joint: Lift phase 진입 시점에 캡처 → Lift phase 동안 고정
        self.lift_finger_pos_buf  = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        # 현재 phase 플래그 (_pre_physics_step → _apply_action 전달)
        self.is_lift_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ----------------------------------------------------------------
        # Hand joint targets (per-finger lerp 결과)
        # ----------------------------------------------------------------
        self.hand_joint_targets = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)

        # ----------------------------------------------------------------
        # 접촉 상태 버퍼 (actor: fingertip)
        # get_contact_force_matrix() 해당: Cup-only pair-wise filtered force
        # ----------------------------------------------------------------
        self.contact_force_xyz_raw = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)  # (N,5,3) fx,fy,fz
        self.contact_force_raw     = torch.zeros(self.num_envs, NUM_FINGERTIPS, device=self.device)      # (N,5) norm
        self.binary_contact_buf    = torch.zeros(self.num_envs, NUM_FINGERTIPS, dtype=torch.bool, device=self.device)
        self.num_contacts_buf      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_contact_count_buf = torch.zeros(self.num_envs, device=self.device)

        # ----------------------------------------------------------------
        # 접촉 상태 버퍼 (critic privileged: distal/middle)
        # ----------------------------------------------------------------
        self.distal_contact_force_raw  = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, device=self.device)
        self.distal_binary_contact_buf = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, dtype=torch.bool, device=self.device)
        self.middle_contact_force_raw  = torch.zeros(self.num_envs, NUM_MIDDLE_SENSORS, device=self.device)
        self.middle_binary_contact_buf = torch.zeros(self.num_envs, NUM_MIDDLE_SENSORS, dtype=torch.bool, device=self.device)

        # ----------------------------------------------------------------
        # 성공 플래그 (terminal reward 판정용)
        # ----------------------------------------------------------------
        self.success_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 컵 기울기 한계
        self._cup_tipping_cos = math.cos(math.radians(cfg.cup_tipping_max_deg))

        # ----------------------------------------------------------------
        # ADR
        # ----------------------------------------------------------------
        if cfg.enable_adr:
            self.grasp_adr = GraspADR(
                custom_cfg=cfg.adr_custom_cfg,
                num_increments=cfg.adr_num_increments,
                increment_interval=cfg.adr_increment_interval,
                trigger_threshold=cfg.adr_trigger_threshold,
            )
        else:
            self.grasp_adr = None

        # ----------------------------------------------------------------
        # Fabrics 초기화
        # ----------------------------------------------------------------
        self._setup_geometric_fabrics()

    # ------------------------------------------------------------------
    # Scene 설정
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cup   = RigidObject(self.cfg.cup_cfg)
        self.table = RigidObject(self.cfg.table_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cup"]   = self.cup
        self.scene.rigid_objects["table"] = self.table

        # Actor: Per-fingertip ContactSensor (rl_dg_1_tip ~ rl_dg_5_tip)
        self._tip_sensors: list[ContactSensor] = []
        for link_name in self.cfg.right_tip_contact_links:
            sensor_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link_name}",
                filter_prim_paths_expr=["/World/envs/env_.*/Cup"],
                history_length=1,
                track_air_time=False,
            )
            sensor = ContactSensor(sensor_cfg)
            self._tip_sensors.append(sensor)
            self.scene.sensors[f"tip_sensor_{link_name}"] = sensor

        # Critic privileged: Distal ContactSensor (rl_dg_*_4)
        self._distal_sensors: list[ContactSensor] = []
        for link_name in self.cfg.right_distal_contact_links:
            sensor_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link_name}",
                filter_prim_paths_expr=["/World/envs/env_.*/Cup"],
                history_length=1,
                track_air_time=False,
            )
            sensor = ContactSensor(sensor_cfg)
            self._distal_sensors.append(sensor)
            self.scene.sensors[f"distal_sensor_{link_name}"] = sensor

        # Critic privileged: Middle ContactSensor (rl_dg_*_3, thumb/index/middle)
        self._middle_sensors: list[ContactSensor] = []
        for link_name in self.cfg.right_middle_contact_links:
            sensor_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link_name}",
                filter_prim_paths_expr=["/World/envs/env_.*/Cup"],
                history_length=1,
                track_air_time=False,
            )
            sensor = ContactSensor(sensor_cfg)
            self._middle_sensors.append(sensor)
            self.scene.sensors[f"middle_sensor_{link_name}"] = sensor

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.scene.clone_environments(copy_from_source=True)

    # ------------------------------------------------------------------
    # Geometric Fabrics 초기화
    # ------------------------------------------------------------------
    def _setup_geometric_fabrics(self) -> None:
        warp_cache_dir = self.device[-1]
        initialize_warp(warp_cache_dir)

        print("=== GraspRightEnv v5: Creating Fabrics world ===")
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
            use_hand_fabric=False,   # hand는 직접 PD로 제어
        )
        num_joints = self.open_tesollo_fabric.num_joints   # 27

        self.open_tesollo_integrator = DisplacementIntegrator(self.open_tesollo_fabric)

        # Fabric 상태 버퍼
        self.fabric_q   = self.robot_start_joint_pos.clone().contiguous()
        self.fabric_qd  = torch.zeros(self.num_envs, num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, num_joints, device=self.device)

        # Fabric input 버퍼 (hand PCA 고정 0)
        self.hand_pca_targets  = torch.zeros(self.num_envs, 5, device=self.device)
        self.palm_pose_targets = torch.zeros(self.num_envs, 6, device=self.device)

        # cspace attractor: hand는 grasp pose 방향
        cspace_default = self.open_tesollo_fabric.default_config.clone()
        cspace_default[:, NUM_ARM_DOF:] = self.hand_grasp_pose.unsqueeze(0).expand(self.num_envs, -1)
        self.open_tesollo_fabric.default_config.copy_(cspace_default)

        self.fabric_damping_gain = 10.0 * torch.ones(self.num_envs, 1, device=self.device)

        # Hand FK taskmap (센서 URDF 기준, 7 bodies)
        # [0]=palm_link (Fabrics attractor 기준점), [1]=palm_x, [2:7]=rl_dg_*_tip
        robot_dir_name = "openarm_tesollo_sensor"
        robot_name     = "openarm_tesollo_sensor"
        urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)
        self.hand_points_taskmap = RobotFrameOriginsTaskMap(
            urdf_path, FABRIC_HAND_BODY_NAMES, self.num_envs, self.device
        )

        print("=== GraspRightEnv v5: Fabrics initialized ===")

    # ------------------------------------------------------------------
    # Contact force reading — MD Section 10 PhysX API 대응
    # ------------------------------------------------------------------
    def _get_cup_contact_force_xyz(self, sensor: ContactSensor) -> torch.Tensor:
        """net_forces_w 기반 contact force (N, 3).

        force_matrix_w(Cup-only filtered)가 항상 0을 반환하는 문제로 인해
        net_forces_w(전체 contact 합산)를 사용.
        self_collision=False이고 tip이 table에 닿는 경우는 극히 드물어 실질적으로 Cup-only.

        net_forces_w shape: (N, B, 3)  B=1(single body sensor)
        """
        nf = sensor.data.net_forces_w   # (N, 1, 3)
        if nf is not None and nf.numel() > 0:
            return nf[:, 0, :]           # (N, 3)
        return torch.zeros(self.num_envs, 3, device=self.device)

    def _get_cup_contact_force_norm(self, sensor: ContactSensor) -> torch.Tensor:
        """Contact force magnitude (N,)."""
        return self._get_cup_contact_force_xyz(sensor).norm(dim=-1)

    def _update_contact_forces(self) -> None:
        """Actor/critic 접촉력 업데이트 (net_forces_w 기반)."""
        # ---- Actor: fingertip xyz force (get_contact_force_matrix 해당) ----
        tip_xyz = torch.stack(
            [self._get_cup_contact_force_xyz(s) for s in self._tip_sensors], dim=1
        )  # (N, 5, 3)
        tip_norms = tip_xyz.norm(dim=-1)  # (N, 5)

        self.contact_force_xyz_raw.copy_(tip_xyz)
        self.contact_force_raw.copy_(tip_norms)
        self.binary_contact_buf.copy_(tip_norms > CONTACT_FORCE_THRESHOLD)
        self.num_contacts_buf.copy_(self.binary_contact_buf.sum(dim=-1).long())

        # ---- Critic: distal (rl_dg_*_4) ----
        per_distal = torch.stack(
            [self._get_cup_contact_force_norm(s) for s in self._distal_sensors], dim=-1
        )  # (N, 5)
        self.distal_contact_force_raw.copy_(per_distal)
        self.distal_binary_contact_buf.copy_(per_distal > CONTACT_FORCE_THRESHOLD)

        # ---- Critic: middle (rl_dg_*_3) ----
        per_middle = torch.stack(
            [self._get_cup_contact_force_norm(s) for s in self._middle_sensors], dim=-1
        )  # (N, 3)
        self.middle_contact_force_raw.copy_(per_middle)
        self.middle_binary_contact_buf.copy_(per_middle > CONTACT_FORCE_THRESHOLD)

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

        # ---- 1. Finger control: per-finger delta 5D → 20D ----
        # action=0  → HAND_GRASP_POSE
        # action=+1 → HAND_GRASP + (GRASP - APPROACH) (더 닫힘)
        # action=-1 → HAND_APPROACH_POSE (완전 열림)
        finger_action = actions[:, :5]   # (N, 5) ∈ [-1, 1]

        hand_target = self.hand_grasp_pose.unsqueeze(0).expand(self.num_envs, -1).clone()  # (N, 20)
        for i in range(5):
            s = i * 4
            e = s + 4
            a_i    = finger_action[:, i:i+1]                                  # (N, 1)
            delta  = self.hand_grasp_pose[s:e] - self.hand_open_pose[s:e]     # (4,) GRASP-APPROACH
            hand_target[:, s:e] = (
                self.hand_grasp_pose[s:e].unsqueeze(0) + a_i * delta.unsqueeze(0)
            )
        self.hand_joint_targets.copy_(hand_target)

        # ---- 2. Phase 판정 ----
        is_lift = (self.episode_length_buf >= GRASP_PHASE_STEPS)   # (N,) bool
        self.is_lift_phase.copy_(is_lift)

        # ---- 3. Lift phase 진입 시 finger joint 포지션 캡처 ----
        # episode_length_buf == GRASP_PHASE_STEPS인 첫 번째 스텝에 캡처
        # (이때 robot.data.joint_pos = Grasp phase 마지막 스텝의 실제 값)
        just_entering_lift = (self.episode_length_buf == GRASP_PHASE_STEPS)
        if just_entering_lift.any():
            ids = just_entering_lift.nonzero(as_tuple=True)[0]
            self.lift_finger_pos_buf[ids] = (
                self.robot.data.joint_pos[ids][:, self.hand_dof_indices].clone()
            )

    def _apply_action(self) -> None:
        is_lift = self.is_lift_phase   # (N,) bool

        # ---- 오른팔 ----
        # Grasp phase: pregrasp arm joint 위치 고정
        # Lift phase:  pregrasp → prelift 선형 보간 (reset 시 Fabrics로 미리 계산된 값)
        lift_progress = (
            (self.episode_length_buf - GRASP_PHASE_STEPS).clamp(min=0).float()
            / LIFT_PHASE_STEPS
        ).clamp(max=1.0).unsqueeze(1)   # (N, 1) ∈ [0, 1]

        arm_target_lift = (
            self.pregrasp_arm_pos_buf * (1.0 - lift_progress)
            + self.prelift_arm_pos_buf * lift_progress
        )
        arm_target = torch.where(
            is_lift.unsqueeze(1),
            arm_target_lift,
            self.pregrasp_arm_pos_buf,
        )

        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(arm_target), joint_ids=self.arm_dof_indices
        )

        # ---- 오른손 ----
        # Grasp phase: 정책 action → per-finger lerp target
        # Lift phase:  Grasp phase 종료 시점 캡처 포지션으로 고정
        finger_target = torch.where(
            is_lift.unsqueeze(1),
            self.lift_finger_pos_buf,
            self.hand_joint_targets,
        )

        self.robot.set_joint_position_target(finger_target, joint_ids=self.hand_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(finger_target), joint_ids=self.hand_dof_indices
        )

        # ---- 왼팔: 고정 자세 ----
        self.robot.write_joint_state_to_sim(
            self.left_arm_zero_pos,
            self.left_arm_zero_vel,
            joint_ids=self.left_arm_dof_indices,
        )

    # ------------------------------------------------------------------
    # Intermediate values
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self) -> None:
        # fabric_q 동기화 (episode 중 Fabrics를 사용하지 않으므로 robot 실제값으로 갱신)
        # hand_points_taskmap FK에서 arm + hand DOF 모두 필요
        self.fabric_q[:, :NUM_ARM_DOF] = self.robot.data.joint_pos[:, self.arm_dof_indices]
        self.fabric_q[:, NUM_ARM_DOF:] = self.robot.data.joint_pos[:, self.hand_dof_indices]

        # 물체 위치
        self.object_pos = self.cup.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.cup.data.root_quat_w

        # Hand FK (센서 URDF 기준)
        # [0]=palm_link, [1]=palm_x, [2:7]=rl_dg_*_tip
        hand_pos_flat, _ = self.hand_points_taskmap(self.fabric_q, None)  # (N, 21)
        all_pos = hand_pos_flat.view(self.num_envs, 7, 3)

        # palm 위치: rl_dg_palm USD body 우선, 없으면 Fabrics FK palm_link
        if self.cfg.right_palm_contact_link in self.robot.data.body_names:
            palm_body_id = self.robot.data.body_names.index(self.cfg.right_palm_contact_link)
            self.palm_center_pos = (
                self.robot.data.body_pos_w[:, palm_body_id] - self.scene.env_origins
            )
        else:
            self.palm_center_pos = all_pos[:, 0, :]

        self.fingertip_pos = all_pos[:, 2:, :]   # (N, 5, 3)

        # 접촉력 업데이트
        self._update_contact_forces()

    # ------------------------------------------------------------------
    # Observations: Actor 104D | Critic 128D (actor + 24D privileged)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # ==== Actor obs (104D) — real-compatible ====

        # 1. palm → cup 상대 위치 (3D)
        palm_to_cup = self.object_pos - self.palm_center_pos   # (N, 3)

        # 2. cup 회전 (4D)
        cup_rot = self.object_rot   # (N, 4)

        # 3. fingertip → cup 상대 위치 (15D)
        fingertip_to_cup = (
            self.fingertip_pos - self.object_pos.unsqueeze(1)
        ).view(self.num_envs, -1)   # (N, 15)

        # 4. finger joint pos/vel (20D each)
        finger_joint_pos = self.robot.data.joint_pos[:, self.hand_dof_indices]   # (N, 20)
        finger_joint_vel = self.robot.data.joint_vel[:, self.hand_dof_indices]   # (N, 20)

        # 5. arm joint pos/vel (7D each)
        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_dof_indices]   # (N, 7)
        arm_joint_vel = self.robot.data.joint_vel[:, self.arm_dof_indices]   # (N, 7)

        # 6. fingertip binary contact (5D) — real Teosllo FT sensor 기반
        binary_contact = self.binary_contact_buf.float()   # (N, 5)

        # 7. fingertip contact force xyz (15D) — net_forces_w 기반, 정규화 [-1,1]
        fingertip_force_xyz = (
            self.contact_force_xyz_raw / CONTACT_FORCE_MAX
        ).clamp(-1.0, 1.0).view(self.num_envs, -1)   # (N, 15)

        # 8. last actions (5D)
        last_actions = self.actions   # (N, 5)

        actor_obs = torch.cat([
            palm_to_cup,           # 3
            cup_rot,               # 4
            fingertip_to_cup,      # 15
            finger_joint_pos,      # 20
            finger_joint_vel,      # 20
            arm_joint_pos,         # 7
            arm_joint_vel,         # 7
            binary_contact,        # 5
            fingertip_force_xyz,   # 15  (fx,fy,fz × 5 tips, Cup-only filtered)
            last_actions,          # 5
        ], dim=-1)   # 104D

        if actor_obs.shape[1] != NUM_OBSERVATIONS:
            raise RuntimeError(
                f"[v5] Actor obs dim mismatch: {actor_obs.shape[1]} != {NUM_OBSERVATIONS}"
            )

        # ==== Critic privileged extras (24D) — sim-only ====

        # cup velocity (6D)
        cup_lin_vel = self.cup.data.root_lin_vel_w   # (N, 3)
        cup_ang_vel = self.cup.data.root_ang_vel_w   # (N, 3)

        # distal link contact (5D binary + 5D force_norm)
        distal_binary    = self.distal_binary_contact_buf.float()   # (N, 5)
        distal_force_norm = (self.distal_contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)  # (N, 5)

        # middle link contact — thumb/index/middle (3D binary + 3D force_norm)
        middle_binary    = self.middle_binary_contact_buf.float()   # (N, 3)
        middle_force_norm = (self.middle_contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)  # (N, 3)

        # scripted lift phase flag (1D)
        lift_flag = (self.episode_length_buf >= GRASP_PHASE_STEPS).float().unsqueeze(1)  # (N, 1)

        # cup height delta from spawn (1D)
        cup_height_delta = (
            self.object_pos[:, 2] - self.object_init_pos[:, 2]
        ).unsqueeze(1)   # (N, 1)

        critic_obs = torch.cat([
            actor_obs,            # 94
            cup_lin_vel,          # 3
            cup_ang_vel,          # 3
            distal_binary,        # 5
            distal_force_norm,    # 5
            middle_binary,        # 3
            middle_force_norm,    # 3
            lift_flag,            # 1
            cup_height_delta,     # 1
        ], dim=-1)   # 118D

        if critic_obs.shape[1] != NUM_CRITIC_OBSERVATIONS:
            raise RuntimeError(
                f"[v5] Critic obs dim mismatch: {critic_obs.shape[1]} != {NUM_CRITIC_OBSERVATIONS}"
            )

        return {"policy": actor_obs, "critic": critic_obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # ---- ADR: contact_delta_weight (3→1), enclosure_weight (2→3) ----
        if self.grasp_adr is not None:
            contact_delta_weight = self.grasp_adr.get_param("reward_weights", "contact_delta_weight")
            enclosure_weight     = self.grasp_adr.get_param("reward_weights", "enclosure_weight")
        else:
            contact_delta_weight = self.cfg.contact_delta_weight
            enclosure_weight     = self.cfg.enclosure_weight

        # ---- 1. contact_reward: 유지 보상 ----
        num_contacts = self.num_contacts_buf.float()   # (N,)
        contact_reward = self.cfg.contact_reward_weight * (num_contacts / NUM_FINGERTIPS)

        # ---- 2. contact_delta: 증가 보상 / 감소 패널티 ----
        delta_contacts = (num_contacts - self.prev_contact_count_buf)   # (N,)
        contact_delta_reward = contact_delta_weight * delta_contacts
        self.prev_contact_count_buf.copy_(num_contacts)

        # ---- 3. enclosure: fingertip → cup 평균 거리 기반 ----
        grasp_center = self.object_pos.clone()
        grasp_center[:, 2] += self.cfg.cup_grasp_z_offset
        fingertip_to_cup_dist = (
            self.fingertip_pos - grasp_center.unsqueeze(1)
        ).norm(dim=-1)  # (N, 5)
        enclosure_reward = enclosure_weight * torch.exp(
            -self.cfg.enclosure_sharpness * fingertip_to_cup_dist.mean(dim=-1)
        )

        # ---- 4. opposition: thumb + 다른 손가락 동시 접촉 ----
        thumb_contact  = self.binary_contact_buf[:, 0].float()            # (N,)
        other_contact  = self.binary_contact_buf[:, 1:].any(dim=-1).float()  # (N,)
        opposition_reward = self.cfg.opposition_weight * (thumb_contact * other_contact)

        # ---- 5. action_reg: action 크기 패널티 ----
        action_reg = self.cfg.action_reg_weight * self.actions.pow(2).sum(dim=-1)

        # ---- 6. terminal rewards ----
        is_terminal = self.reset_buf   # (N,) True when episode ends (terminated|truncated)
        terminal_success_reward = self.cfg.terminal_success_weight * self.success_flag.float()
        terminal_fail_reward    = self.cfg.terminal_fail_weight * (
            is_terminal.float() * (~self.success_flag).float()
        )

        # ---- 합산 ----
        total = (
            contact_reward
            + contact_delta_reward
            + enclosure_reward
            + opposition_reward
            + action_reg
            + terminal_success_reward
            + terminal_fail_reward
        )

        # ---- ADR increment ----
        if self.grasp_adr is not None:
            # 성공 비율로 ADR 진행
            lift_success_ratio = self.success_flag.float().mean()
            self.grasp_adr.maybe_increment(lift_success_ratio)

        # ---- 로깅 ----
        self.extras["contact_reward"]       = contact_reward.mean()
        self.extras["contact_delta_reward"] = contact_delta_reward.mean()
        self.extras["enclosure_reward"]     = enclosure_reward.mean()
        self.extras["opposition_reward"]    = opposition_reward.mean()
        self.extras["action_reg"]           = action_reg.mean()
        self.extras["terminal_success"]     = terminal_success_reward.mean()
        self.extras["terminal_fail"]        = terminal_fail_reward.mean()
        self.extras["num_contacts"]         = num_contacts.mean()
        self.extras["cup_z"]                = self.object_pos[:, 2].mean()
        self.extras["success_rate"]         = self.success_flag.float().mean()
        self.extras["adr_contact_delta_w"]  = torch.tensor(contact_delta_weight, device=self.device)
        self.extras["adr_enclosure_w"]      = torch.tensor(enclosure_weight, device=self.device)
        if self.grasp_adr is not None:
            self.extras["adr_progress"] = torch.tensor(self.grasp_adr.progress, device=self.device)

        return total

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # 물체 범위 이탈
        out_x = (
            (self.object_pos[:, 0] < self.cfg.obj_out_x_min) |
            (self.object_pos[:, 0] > self.cfg.obj_out_x_max)
        )
        out_y = (
            (self.object_pos[:, 1] < self.cfg.obj_out_y_min) |
            (self.object_pos[:, 1] > self.cfg.obj_out_y_max)
        )
        fallen = self.object_pos[:, 2] < self.cfg.obj_fallen_z

        # 컵 기울기 초과
        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cup_z_world = quat_apply(self.object_rot, z_local)
        tipped = cup_z_world[:, 2] < self._cup_tipping_cos

        # ---- success: lift phase 이후 컵 들린 상태 + 접촉 유지 ----
        # GRASP_PHASE_STEPS 이후(lift phase)에만 success 판정
        in_or_past_lift = (self.episode_length_buf >= GRASP_PHASE_STEPS)
        lifted  = self.object_pos[:, 2] > (self.object_init_pos[:, 2] + self.cfg.lift_success_height)
        grasped = (self.num_contacts_buf >= MIN_CONTACTS_FOR_SUCCESS)
        self.success_flag.copy_(in_or_past_lift & lifted & grasped)

        terminated = out_x | out_y | fallen | tipped | self.success_flag
        truncated  = self.episode_length_buf >= self.max_episode_length - 1

        # 로깅
        self.extras["grasp_success"] = self.success_flag.float().mean()
        self.extras["object_z"]      = self.object_pos[:, 2].mean()

        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        n = len(env_ids)

        # ---- DEBUG: 에피소드 종료 시점 위치 출력 (env 0 기준) ----
        if 0 in env_ids.tolist():
            body_names = list(self.robot.data.body_names)

            def _get_local(body_name: str) -> torch.Tensor:
                if body_name in body_names:
                    bid = body_names.index(body_name)
                    return self.robot.data.body_pos_w[0, bid] - self.scene.env_origins[0]
                return torch.full((3,), float("nan"), device=self.device)

            palm_dbg  = _get_local("rl_dg_palm")
            thumb_dbg = _get_local("rl_dg_1_1")
            cup_dbg   = self.cup.data.root_pos_w[0] - self.scene.env_origins[0]

            print(
                f"\n[DEBUG episode-end env0]\n"
                f"  rl_dg_palm : x={palm_dbg[0]:.4f}  y={palm_dbg[1]:.4f}  z={palm_dbg[2]:.4f}\n"
                f"  rl_dg_1_1  : x={thumb_dbg[0]:.4f}  y={thumb_dbg[1]:.4f}  z={thumb_dbg[2]:.4f}\n"
                f"  cup        : x={cup_dbg[0]:.4f}  y={cup_dbg[1]:.4f}  z={cup_dbg[2]:.4f}\n"
                f"  palm→cup   : dx={cup_dbg[0]-palm_dbg[0]:.4f}"
                f"  dy={cup_dbg[1]-palm_dbg[1]:.4f}"
                f"  dz={cup_dbg[2]-palm_dbg[2]:.4f}"
            )

        # ---- 1. 로봇 관절 상태 리셋 ----
        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)

        for k, idx in enumerate(self.actuated_dof_indices):
            full_pos[:, idx] = self.robot_start_joint_pos[0, k]
        for k, idx in enumerate(self.left_arm_dof_indices):
            full_pos[:, idx] = self.left_arm_zero_pos[0, k]

        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        # ---- 2. Fabrics 상태 리셋 ----
        self.fabric_q[env_ids]   = self.robot_start_joint_pos[env_ids]
        self.fabric_qd[env_ids].zero_()
        self.fabric_qdd[env_ids].zero_()

        # ---- 3. 컵 spawn 위치 계산 ----
        obj_x = self.cfg.object_spawn_x_center + (
            torch.rand(n, device=self.device) - 0.5
        ) * 2.0 * self.cfg.object_spawn_xy_range
        obj_y = self.cfg.object_spawn_y_center + (
            torch.rand(n, device=self.device) - 0.5
        ) * 2.0 * self.cfg.object_spawn_xy_range
        obj_pos_local = torch.stack(
            [obj_x, obj_y, torch.full((n,), self.cfg.object_spawn_z, device=self.device)], dim=1
        )
        self.object_init_pos[env_ids] = obj_pos_local

        # ---- 4. FABRICS pregrasp: cup 기준 palm 위치 이동 ----
        noise = torch.stack([
            (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_x,
            (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_y,
            (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_z,
        ], dim=1)
        pregrasp_pos = obj_pos_local + self.pregrasp_offset.unsqueeze(0) + noise   # (n, 3)

        pregrasp_palm_pose = torch.zeros(n, 6, device=self.device)
        pregrasp_palm_pose[:, :3] = pregrasp_pos
        pregrasp_palm_pose[:, 3] = math.radians(90.0)   # ez: +90° (palm +X → world +Y, 손바닥이 컵 방향)
        pregrasp_palm_pose[:, 4] = math.radians(0.0)    # ey:   0° (side-approach, v1과 동일)
        pregrasp_palm_pose[:, 5] = math.radians(90.0)   # ex: +90° (palm +Z → world +X, 손가락이 +X 방향)

        # workspace clamp (전체 6D)
        pregrasp_palm_pose = torch.max(
            torch.min(pregrasp_palm_pose, self.palm_maxs.unsqueeze(0)),
            self.palm_mins.unsqueeze(0),
        )

        # pregrasp orientation 저장 (clamp 적용 후 값)
        self.pregrasp_palm_orient_buf[env_ids] = pregrasp_palm_pose[:, 3:6]

        # Fabrics pregrasp rollout (full-batch, env_ids 슬롯만 실제 변경)
        self.palm_pose_targets[env_ids] = pregrasp_palm_pose

        fabric_q_full   = self.fabric_q.clone()
        fabric_qd_full  = self.fabric_qd.clone()
        fabric_qdd_full = self.fabric_qdd.clone()

        inputs_pregrasp = [
            self.hand_pca_targets,
            self.palm_pose_targets.clone(),
            "euler_zyx",
            fabric_q_full.detach(),
            fabric_qd_full.detach(),
            self.object_ids,
            self.object_indicator,
            self.fabric_damping_gain,
        ]
        self.open_tesollo_fabric.set_features(*inputs_pregrasp)

        for _ in range(self.cfg.pregrasp_fabric_steps):
            fabric_q_full, fabric_qd_full, fabric_qdd_full = self.open_tesollo_integrator.step(
                fabric_q_full.detach(),
                fabric_qd_full.detach(),
                fabric_qdd_full.detach(),
                self.timestep,
            )

        self.fabric_q[env_ids]   = fabric_q_full[env_ids]
        self.fabric_qd[env_ids]  = fabric_qd_full[env_ids]
        self.fabric_qdd[env_ids] = fabric_qdd_full[env_ids]

        # hand DOF를 APPROACH_POSE로 고정 (arm DOF만 Fabrics 결과 사용)
        # pregrasp rollout 중 cspace attractor가 손가락을 당겼을 수 있으므로 열린 자세로 복원
        approach_hand = self.hand_approach_buf.unsqueeze(0).expand(n, -1)
        self.fabric_q[env_ids, NUM_ARM_DOF:]   = approach_hand
        self.fabric_qd[env_ids, NUM_ARM_DOF:].zero_()
        self.fabric_qdd[env_ids, NUM_ARM_DOF:].zero_()

        # ---- 5. pregrasp palm FK 위치 기록 ----
        hand_pos_flat, _ = self.hand_points_taskmap(self.fabric_q, None)   # (N, 21)
        all_pos_pg = hand_pos_flat.view(self.num_envs, 7, 3)
        self.pregrasp_palm_pos_buf[env_ids] = all_pos_pg[env_ids, 0, :]   # palm_link

        # ---- 5b. pregrasp arm joint 위치 저장 ----
        # Grasp phase 전 기간 동안 arm joint target으로 사용 (고정)
        self.pregrasp_arm_pos_buf[env_ids] = self.fabric_q[env_ids, :NUM_ARM_DOF]

        # ---- 5c. prelift arm joint 위치 계산 ----
        # lifted palm target (pregrasp + LIFT_Z_DELTA) 으로 Fabrics rollout →
        # Lift phase에서 선형 보간의 끝점으로 사용
        lifted_palm_pose = self.palm_pose_targets.clone()
        lifted_palm_pose[env_ids, 2] += LIFT_Z_DELTA   # z만 LIFT_Z_DELTA 올림

        fabric_q_lift   = self.fabric_q.clone()
        fabric_qd_lift  = torch.zeros_like(fabric_q_lift)
        fabric_qdd_lift = torch.zeros_like(fabric_q_lift)

        self.open_tesollo_fabric.set_features(
            self.hand_pca_targets,
            lifted_palm_pose,
            "euler_zyx",
            fabric_q_lift.detach(),
            fabric_qd_lift.detach(),
            self.object_ids,
            self.object_indicator,
            self.fabric_damping_gain,
        )
        for _ in range(self.cfg.pregrasp_fabric_steps):
            fabric_q_lift, fabric_qd_lift, fabric_qdd_lift = self.open_tesollo_integrator.step(
                fabric_q_lift.detach(),
                fabric_qd_lift.detach(),
                fabric_qdd_lift.detach(),
                self.timestep,
            )
        self.prelift_arm_pos_buf[env_ids] = fabric_q_lift[env_ids, :NUM_ARM_DOF]

        # ---- 5d. lift_finger_pos_buf 초기화 (approach pose) ----
        # _pre_physics_step에서 GRASP_PHASE_STEPS 진입 시 실제 값으로 덮어쓴다
        self.lift_finger_pos_buf[env_ids] = approach_hand

        # ---- 5e. pregrasp 완료 위치 디버그 (env0만) ----
        if 0 in env_ids.tolist():
            pg_palm = self.pregrasp_palm_pos_buf[0]   # Fabrics FK palm_link 위치
            pg_cup  = obj_pos_local[0] if 0 < n else obj_pos_local[0]
            print(
                f"\n[DEBUG pregrasp-start env0]\n"
                f"  palm_link (Fabrics FK): x={pg_palm[0]:.4f}  y={pg_palm[1]:.4f}  z={pg_palm[2]:.4f}\n"
                f"  cup target            : x={pg_cup[0]:.4f}  y={pg_cup[1]:.4f}  z={pg_cup[2]:.4f}\n"
                f"  palm→cup dx={pg_cup[0]-pg_palm[0]:.4f}"
                f"  dy={pg_cup[1]-pg_palm[1]:.4f}"
                f"  dz={pg_cup[2]-pg_palm[2]:.4f}"
                f"  (기대 dx≈+0.167, dy≈+0.09, dz≈-0.04)"
            )

        # ---- 6. 로봇 관절을 Fabrics pregrasp 결과로 업데이트 ----
        pregrasp_full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        pregrasp_full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)

        for k, idx in enumerate(self.arm_dof_indices):
            pregrasp_full_pos[:, idx] = self.fabric_q[env_ids, k]
        for k_hand, idx in enumerate(self.hand_dof_indices):
            pregrasp_full_pos[:, idx] = HAND_APPROACH_POSE[k_hand]   # 열린 자세: 침투 없음
        for k, idx in enumerate(self.left_arm_dof_indices):
            pregrasp_full_pos[:, idx] = self.left_arm_zero_pos[0, k]

        self.robot.write_joint_state_to_sim(pregrasp_full_pos, pregrasp_full_vel, env_ids=env_ids)

        # ---- 7. 컵 spawn ----
        obj_pos_world = obj_pos_local + self.scene.env_origins[env_ids]
        upright_rot = torch.zeros(n, 4, device=self.device)
        upright_rot[:, 0] = 1.0
        zero_vel = torch.zeros(n, 6, device=self.device)
        cup_root_state = torch.cat([obj_pos_world, upright_rot, zero_vel], dim=-1)
        self.cup.write_root_state_to_sim(cup_root_state, env_ids=env_ids)

        # ---- 8. hand joint targets 리셋 (open → episode 첫 스텝에서 action=0이 GRASP_POSE로 이동) ----
        self.hand_joint_targets[env_ids] = self.hand_open_pose.unsqueeze(0).expand(n, -1)

        # ---- 9. 접촉 상태 리셋 (actor + critic) ----
        self.contact_force_raw[env_ids].zero_()
        self.binary_contact_buf[env_ids] = False
        self.num_contacts_buf[env_ids]   = 0
        self.prev_contact_count_buf[env_ids].zero_()

        self.distal_contact_force_raw[env_ids].zero_()
        self.distal_binary_contact_buf[env_ids] = False
        self.middle_contact_force_raw[env_ids].zero_()
        self.middle_binary_contact_buf[env_ids] = False

        # ---- 10. 성공 플래그 리셋 ----
        self.success_flag[env_ids] = False

        # ---- 11. actions 리셋 (손가락: open 상태에서 시작) ----
        self.actions[env_ids].fill_(-1.0)  # 손가락 action=-1 → APPROACH_POSE(open) 유지
