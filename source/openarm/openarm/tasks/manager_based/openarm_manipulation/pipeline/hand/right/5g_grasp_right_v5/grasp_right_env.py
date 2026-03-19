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
  - Action: 5D per-finger lerp (orientation 고정)
  - Pre-grasp: reset에서 FABRICS로 cup-relative 위치 형성
  - Contact: 물리 ContactSensor (fingertip 5개) 기반
  - Episode: 10s = 8s grasp phase + 2s scripted lift checker
  - Reward: Lift-phase conditioned (방향 A)
      Grasp phase: dense reward × grasp_shaping_scale (0.05)  ← 누적 지배 방지
      Lift  phase: dense reward × 1.0 + lift_reward           ← 파지 유지하며 리프트
  - ADR: contact_delta_weight (2→0.5), enclosure_weight (4→8), trigger 2%
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
        - finger joints = Grasp Phase 종료 시점 포지션으로 고정 (마지막 action hold)

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

        # fingertip body indices (body_pos_w 직접 참조용, taskmap 대체)
        _tip_names = [f"rl_dg_{i}_tip" for i in range(1, 6)]
        self.fingertip_body_indices: list[int] = [
            self.robot.data.body_names.index(name)
            for name in _tip_names
        ]
        _palm_name = "rl_dg_palm"
        self.palm_body_index: int = (
            self.robot.data.body_names.index(_palm_name)
            if _palm_name in self.robot.data.body_names
            else -1
        )

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
        # net_forces_w: (N, 5, 3) — body별 합산 접촉력
        # ----------------------------------------------------------------
        self.contact_force_xyz_raw = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)  # (N,5,3) fx,fy,fz
        self.contact_force_raw     = torch.zeros(self.num_envs, NUM_FINGERTIPS, device=self.device)      # (N,5) norm
        self.binary_contact_buf    = torch.zeros(self.num_envs, NUM_FINGERTIPS, dtype=torch.bool, device=self.device)
        self.num_contacts_buf      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_contact_count_buf = torch.zeros(self.num_envs, device=self.device)

        # ----------------------------------------------------------------
        # 접촉 상태 버퍼 (critic privileged: distal)
        # ----------------------------------------------------------------
        self.distal_contact_force_raw  = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, device=self.device)
        self.distal_binary_contact_buf = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, dtype=torch.bool, device=self.device)

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

        # Actor: fingertip 5개 개별 ContactSensor (USD ContactSensor + filter)
        # force_matrix_w: (N, 1, 1, 3) — Cup-only 접촉력
        _CUP_FILTER = ["/World/envs/env_.*/Cup"]
        self._tip_sensors: list[ContactSensor] = []
        for link_name in self.cfg.right_tip_contact_links:
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link_name}",
                filter_prim_paths_expr=_CUP_FILTER,
                history_length=1,
                track_air_time=False,
            ))
            self._tip_sensors.append(sensor)
            self.scene.sensors[f"tip_sensor_{link_name}"] = sensor

        # Critic privileged: distal 5개 통합 ContactSensor (USD ContactSensor 없음)
        # net_forces_w: (N, 5, 3) — filter 없이 body별 합산 접촉력
        self._distal_sensor = ContactSensor(self.cfg.distal_sensor_cfg)
        self.scene.sensors["distal_sensor"] = self._distal_sensor

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

        # Reset 전용 소형 Fabrics (env_ids만 실행하여 full-batch 낭비 제거)
        # 매 iter reset env 수 ≈ num_envs × horizon / episode_length ≈ 56개
        # MAX_RESET_CHUNK보다 많으면 chunk로 나눠 처리
        self._reset_chunk = self.cfg.reset_fabric_chunk_size
        self._reset_fabric = OpenArmTeoslloPoseFabric(
            self._reset_chunk, self.device, self.timestep,
            graph_capturable=False,
            use_hand_fabric=False,
        )
        self._reset_integrator = DisplacementIntegrator(self._reset_fabric)

        reset_cspace = self._reset_fabric.default_config.clone()
        reset_cspace[:, NUM_ARM_DOF:] = self.hand_grasp_pose.unsqueeze(0).expand(self._reset_chunk, -1)
        self._reset_fabric.default_config.copy_(reset_cspace)

        # Reset Fabrics 고정 입력 버퍼
        self._reset_pca    = torch.zeros(self._reset_chunk, 5, device=self.device)
        self._reset_damping = 10.0 * torch.ones(self._reset_chunk, 1, device=self.device)
        # WorldMeshesModel은 batch_size 고정이므로 reset 전용 더미 생성
        self._reset_world = WorldMeshesModel(
            batch_size=self._reset_chunk,
            max_objects_per_env=self.cfg.fabrics_max_objects_per_env,
            device=self.device,
            world_filename="open_tesollo_boxes_no_table",
        )
        self._reset_obj_ids, self._reset_obj_indicator = self._reset_world.get_object_ids()

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
    # Reset 전용 Fabrics rollout (env_ids chunk만 실행)
    # ------------------------------------------------------------------
    def _run_reset_fabric(
        self,
        env_ids: torch.Tensor,
        palm_pose: torch.Tensor,
        q_init: torch.Tensor,
    ) -> torch.Tensor:
        """env_ids(n개)만 Fabrics rollout해서 arm joint 위치 반환.

        Args:
            env_ids: reset 대상 env 인덱스 (n,)
            palm_pose: 목표 palm pose (n, 6)
            q_init: 시작 joint pos (n, 27)

        Returns:
            q_out: 수렴된 joint pos (n, 27)
        """
        n = len(env_ids)
        C = self._reset_chunk   # chunk 크기 (128)
        q_out = torch.zeros_like(q_init)   # (n, 27)

        for start in range(0, n, C):
            end = min(start + C, n)
            m   = end - start   # 실제 envs 수

            # chunk 슬라이스 (m ≤ C)
            pp = palm_pose[start:end]       # (m, 6)
            qi = q_init[start:end]          # (m, 27)

            # m < C 이면 마지막 env로 패딩 (Fabrics batch_size=C 유지)
            if m < C:
                pad = C - m
                pp = torch.cat([pp, pp[-1:].expand(pad, -1)], dim=0)
                qi = torch.cat([qi, qi[-1:].expand(pad, -1)], dim=0)

            fq   = qi.clone().contiguous()
            fqd  = torch.zeros(C, qi.shape[1], device=self.device)
            fqdd = torch.zeros(C, qi.shape[1], device=self.device)

            self._reset_fabric.set_features(
                self._reset_pca,
                pp,
                "euler_zyx",
                fq.detach(),
                fqd.detach(),
                self._reset_obj_ids,
                self._reset_obj_indicator,
                self._reset_damping,
            )
            for _ in range(self.cfg.pregrasp_fabric_steps):
                fq, fqd, fqdd = self._reset_integrator.step(
                    fq.detach(), fqd.detach(), fqdd.detach(), self.timestep
                )

            q_out[start:end] = fq[:m]

        return q_out

    def _update_contact_forces(self) -> None:
        """Actor/critic 접촉력 업데이트.

        tip: 개별 센서 force_matrix_w (N,1,1,3) → Cup-only 접촉력 (USD ContactSensor 기반)
        distal: 통합 센서 net_forces_w (N,5,3) → 합산 접촉력 (critic only, sim-only)
        """
        # ---- Actor: fingertip 개별 센서 (Cup-only, force_matrix_w) ----
        tip_xyz = torch.stack([
            s.data.force_matrix_w[:, 0, 0, :] for s in self._tip_sensors
        ], dim=1)                          # (N, 5, 3)
        tip_norms = tip_xyz.norm(dim=-1)   # (N, 5)

        self.contact_force_xyz_raw.copy_(tip_xyz)
        self.contact_force_raw.copy_(tip_norms)
        self.binary_contact_buf.copy_(tip_norms > CONTACT_FORCE_THRESHOLD)
        self.num_contacts_buf.copy_(self.binary_contact_buf.sum(dim=-1).long())

        # ---- Critic: distal 통합 센서 (net_forces_w, filter 없음) ----
        per_distal = self._distal_sensor.data.net_forces_w.norm(dim=-1)  # (N, 5)

        self.distal_contact_force_raw.copy_(per_distal)
        self.distal_binary_contact_buf.copy_(per_distal > CONTACT_FORCE_THRESHOLD)

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

        # per-finger lerp 벡터화: action (N,5) → joint target (N,20)
        # delta (20,): GRASP - APPROACH per joint
        # action_exp (N,20): 각 finger action을 4 joints에 반복 적용
        delta_20    = self.hand_grasp_pose - self.hand_open_pose               # (20,)
        action_exp  = finger_action.repeat_interleave(4, dim=1)                # (N, 20)
        hand_target = self.hand_grasp_pose.unsqueeze(0) + action_exp * delta_20.unsqueeze(0)
        self.hand_joint_targets.copy_(hand_target)

        # ---- 2. Phase 판정 ----
        is_lift = (self.episode_length_buf >= GRASP_PHASE_STEPS)   # (N,) bool
        self.is_lift_phase.copy_(is_lift)

        # ---- 3. Lift phase 진입 시 finger joint 포지션 캡처 (마지막 action hold) ----
        # torch.where로 CPU sync 없이 순수 GPU 연산으로 처리
        just_entering_lift = (self.episode_length_buf == GRASP_PHASE_STEPS)  # (N,) bool
        self.lift_finger_pos_buf = torch.where(
            just_entering_lift.unsqueeze(1),
            self.robot.data.joint_pos[:, self.hand_dof_indices],
            self.lift_finger_pos_buf,
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
        # Lift phase:  마지막 action hold (Grasp phase 종료 시점 캡처값 고정)
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
        # 물체 위치
        self.object_pos = self.cup.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.cup.data.root_quat_w

        # palm / fingertip 위치 — body_pos_w (순수 GPU tensor, taskmap 불필요)
        env_origins = self.scene.env_origins   # (N, 3)
        if self.palm_body_index >= 0:
            self.palm_center_pos = (
                self.robot.data.body_pos_w[:, self.palm_body_index, :] - env_origins
            )
        self.fingertip_pos = (
            self.robot.data.body_pos_w[:, self.fingertip_body_indices, :] - env_origins.unsqueeze(1)
        )  # (N, 5, 3)

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

        # 7. fingertip contact force xyz (15D) — force_matrix_w Cup-only, 정규화 [-1,1]
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
            fingertip_force_xyz,   # 15  (fx,fy,fz × 5 tips, force_matrix_w Cup-only)
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
        distal_binary     = self.distal_binary_contact_buf.float()
        distal_force_norm = (self.distal_contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)

        # scripted lift phase flag (1D)
        lift_flag = (self.episode_length_buf >= GRASP_PHASE_STEPS).float().unsqueeze(1)  # (N, 1)

        # cup height delta from spawn (1D)
        cup_height_delta = (
            self.object_pos[:, 2] - self.object_init_pos[:, 2]
        ).unsqueeze(1)   # (N, 1)

        critic_obs = torch.cat([
            actor_obs,            # 101
            cup_lin_vel,          # 3
            cup_ang_vel,          # 3
            distal_binary,        # 5
            distal_force_norm,    # 5
            lift_flag,            # 1
            cup_height_delta,     # 1
        ], dim=-1)   # 119D

        if critic_obs.shape[1] != NUM_CRITIC_OBSERVATIONS:
            raise RuntimeError(
                f"[v5] Critic obs dim mismatch: {critic_obs.shape[1]} != {NUM_CRITIC_OBSERVATIONS}"
            )

        return {"policy": actor_obs, "critic": critic_obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # ---- ADR ----
        if self.grasp_adr is not None:
            contact_delta_weight = self.grasp_adr.get_param("reward_weights", "contact_delta_weight")
            enclosure_weight     = self.grasp_adr.get_param("reward_weights", "enclosure_weight")
        else:
            contact_delta_weight = self.cfg.contact_delta_weight
            enclosure_weight     = self.cfg.enclosure_weight

        # ---- Phase flag ----
        is_lift_flag = self.is_lift_phase.float()   # (N,)

        # ---- Phase-conditional dense reward scale (방향 A) ----
        # Grasp phase: dense × grasp_shaping_scale  (방향 안내만, 누적 지배 방지)
        # Lift  phase: dense × 1.0                  (파지 유지 + 리프트 동시 달성 시 풀 보상)
        dense_scale = (
            self.cfg.grasp_shaping_scale
            + (1.0 - self.cfg.grasp_shaping_scale) * is_lift_flag
        )   # (N,) ∈ [grasp_shaping_scale, 1.0]

        # ---- 1. contact_reward ----
        num_contacts = self.num_contacts_buf.float()   # (N,)
        contact_reward = self.cfg.contact_reward_weight * (num_contacts / NUM_FINGERTIPS)

        # ---- 2. contact_delta ----
        delta_contacts = (num_contacts - self.prev_contact_count_buf)   # (N,)
        contact_delta_reward = contact_delta_weight * delta_contacts
        self.prev_contact_count_buf.copy_(num_contacts)

        # ---- 3. enclosure ----
        grasp_center = self.object_pos.clone()
        grasp_center[:, 2] += self.cfg.cup_grasp_z_offset
        fingertip_to_cup_dist = (
            self.fingertip_pos - grasp_center.unsqueeze(1)
        ).norm(dim=-1)  # (N, 5)
        enclosure_reward = enclosure_weight * torch.exp(
            -self.cfg.enclosure_sharpness * fingertip_to_cup_dist.mean(dim=-1)
        )

        # ---- 4. opposition ----
        thumb_contact = self.binary_contact_buf[:, 0].float()             # (N,)
        other_contact = self.binary_contact_buf[:, 1:].any(dim=-1).float()  # (N,)
        opposition_reward = self.cfg.opposition_weight * (thumb_contact * other_contact)

        # ---- dense reward에 phase scale 적용 ----
        dense_reward = dense_scale * (
            contact_reward
            + contact_delta_reward
            + enclosure_reward
            + opposition_reward
        )

        # ---- 5. action_reg (scale 미적용, 항상 full) ----
        action_reg = self.cfg.action_reg_weight * self.actions.pow(2).sum(dim=-1)

        # ---- 6. lift_reward ----
        cup_height_delta = (
            self.object_pos[:, 2] - self.object_init_pos[:, 2]
        ).clamp(min=0.0)   # (N,) ≥ 0
        lift_reward = self.cfg.lift_reward_weight * is_lift_flag * cup_height_delta

        # ---- 7. terminal rewards ----
        is_terminal = self.reset_buf
        terminal_success_reward = self.cfg.terminal_success_weight * self.success_flag.float()
        terminal_fail_reward    = self.cfg.terminal_fail_weight * (
            is_terminal.float() * (~self.success_flag).float()
        )

        # ---- 합산 ----
        total = (
            dense_reward
            + action_reg
            + lift_reward
            + terminal_success_reward
            + terminal_fail_reward
        )

        # ---- ADR increment ----
        if self.grasp_adr is not None:
            self.grasp_adr.maybe_increment(self.success_flag.float().mean())

        # ---- 로깅 ----
        self.extras["contact_reward"]       = contact_reward.mean()
        self.extras["contact_delta_reward"] = contact_delta_reward.mean()
        self.extras["enclosure_reward"]     = enclosure_reward.mean()
        self.extras["opposition_reward"]    = opposition_reward.mean()
        self.extras["dense_scale"]          = dense_scale.mean()   # 모니터링: 0.05→1.0 전환 확인
        self.extras["action_reg"]           = action_reg.mean()
        self.extras["lift_reward"]          = lift_reward.mean()
        self.extras["cup_height_delta"]     = cup_height_delta.mean()
        self.extras["num_contacts"]         = num_contacts.mean()
        self.extras["success_rate"]         = self.success_flag.float().mean()
        self.extras["adr_contact_delta_w"]  = torch.tensor(contact_delta_weight, device=self.device)
        self.extras["adr_enclosure_w"]      = torch.tensor(enclosure_weight, device=self.device)
        # ---- tip / distal 접촉 상태 개별 모니터링 ----
        self.extras["tip_num_contacts"]     = self.binary_contact_buf.float().sum(dim=-1).mean()
        self.extras["tip_force_mean"]       = self.contact_force_raw.mean()
        self.extras["distal_num_contacts"]  = self.distal_binary_contact_buf.float().sum(dim=-1).mean()
        self.extras["distal_force_mean"]    = self.distal_contact_force_raw.mean()
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
        # grasp_success = success_rate (동일값, _get_rewards에서 로깅)
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

        # ---- 1. 로봇 관절 상태 리셋 ----
        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)

        full_pos[:, self.actuated_dof_indices] = self.robot_start_joint_pos[0]
        full_pos[:, self.left_arm_dof_indices] = self.left_arm_zero_pos[0]

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

        # ---- Pregrasp IK: env_ids 전용 소형 Fabrics (full-batch 낭비 제거) ----
        q_init_pregrasp = self.fabric_q[env_ids].clone()   # (n, 27) 현재 joint 상태
        q_pregrasp = self._run_reset_fabric(
            env_ids, pregrasp_palm_pose, q_init_pregrasp
        )   # (n, 27)

        self.fabric_q[env_ids]   = q_pregrasp
        self.fabric_qd[env_ids].zero_()
        self.fabric_qdd[env_ids].zero_()

        # hand DOF를 APPROACH_POSE로 고정 (arm DOF만 Fabrics 결과 사용)
        approach_hand = self.hand_approach_buf.unsqueeze(0).expand(n, -1)
        self.fabric_q[env_ids, NUM_ARM_DOF:]   = approach_hand
        self.fabric_qd[env_ids, NUM_ARM_DOF:].zero_()
        self.fabric_qdd[env_ids, NUM_ARM_DOF:].zero_()

        # ---- 5b. pregrasp arm joint 위치 저장 ----
        self.pregrasp_arm_pos_buf[env_ids] = self.fabric_q[env_ids, :NUM_ARM_DOF]

        # ---- 5c. prelift arm joint 위치 계산 ----
        lifted_palm_pose = pregrasp_palm_pose.clone()
        lifted_palm_pose[:, 2] += LIFT_Z_DELTA   # z만 LIFT_Z_DELTA 올림

        q_init_prelift = self.fabric_q[env_ids].clone()   # pregrasp 수렴점에서 시작
        q_prelift = self._run_reset_fabric(
            env_ids, lifted_palm_pose, q_init_prelift
        )   # (n, 27)
        self.prelift_arm_pos_buf[env_ids] = q_prelift[:, :NUM_ARM_DOF]

        # ---- 5d. lift_finger_pos_buf 초기화 (approach pose) ----
        # _pre_physics_step에서 GRASP_PHASE_STEPS 진입 시 실제 값으로 덮어쓴다
        self.lift_finger_pos_buf[env_ids] = approach_hand

        # ---- 6. 로봇 관절을 Fabrics pregrasp 결과로 업데이트 ----
        pregrasp_full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        pregrasp_full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)

        pregrasp_full_pos[:, self.arm_dof_indices]  = self.fabric_q[env_ids, :NUM_ARM_DOF]
        pregrasp_full_pos[:, self.hand_dof_indices] = self.hand_approach_buf.unsqueeze(0).expand(n, -1)
        pregrasp_full_pos[:, self.left_arm_dof_indices] = self.left_arm_zero_pos[0]

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

        # ---- 10. 성공 플래그 리셋 ----
        self.success_flag[env_ids] = False

        # ---- 11. actions 리셋 (손가락: open 상태에서 시작) ----
        self.actions[env_ids].fill_(-1.0)  # 손가락 action=-1 → APPROACH_POSE(open) 유지
