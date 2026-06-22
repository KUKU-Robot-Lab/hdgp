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

"""환경 클래스: open-rh56f1_r_grasp_v1.

Action (12D):
  [0:6]  6D palm pose → Fabrics IK → arm 7 DOF
  [6:12] 6D absolute hand synergy target
         grasp:     APPROACH_POSE(-1) ~ GRASP_POSE(+1)
         post-grasp: GRASP_POSE(-1) ~ FULL_GRIP_POSE(+1)

Episode (10s @ 60Hz):
  Grasp     phase (0~419):    Fabrics arm + absolute finger target
  Lift      phase (420~539):  policy lift + full-grip target refinement
  Stabilize phase (540~599):  hold/re-grip stabilization
  Transport phase: disabled
"""

from __future__ import annotations

import math
import os
import sys
from collections import deque
from pathlib import Path
from collections.abc import Sequence

import h5py
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
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

from openarm.common.grasp_logging import (
    joint_state_scalars,
    per_finger_contact_scalars,
)
from openarm.common.grasp_reward_core import compute_grasp_reward_terms
from openarm.common.grasp_v2_contract import (
    compute_action_delta_norm,
    compute_grasp_v2_stability,
    compute_stationary_grasp_success,
    log_grasp_v2_common_scalars,
)
from fabrics_sim.fabrics.openarm_rh56f1_pose_fabric import OpenArmRh56f1PoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

from .grasp_right_env_cfg import GraspRightEnvCfg
from .grasp_adr import GraspADR
from .grasp_right_constants import (
    NUM_ARM_DOF,
    NUM_HAND_DOF,
    NUM_ACTIONS,
    NUM_FINGERTIPS,
    NUM_TIP_SENSORS,
    NUM_CRITIC_OBSERVATIONS,
    GRASP_PHASE_STEPS,
    LIFT_PHASE_STEPS,
    LIFT_START_STEP,
    STABILIZE_START_STEP,
    STABILIZE_PHASE_STEPS,
    EPISODE_STEPS,
    CONTACT_FORCE_THRESHOLD,
    CONTACT_FORCE_MAX,
    CUP_RADIUS_APPROX,
    ARM_START_POSE,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
)
from .grasp_right_preset import (
    LEFT_ARM_REST_JOINT_POS,
    HAND_APPROACH_POSE,
    HAND_GRASP_POSE,
    HAND_FULL_GRIP_POSE,
)
from .finger_action_utils import (
    compute_grasp_phase_finger_targets,
    compute_lift_finger_targets,
)
from .grasp_reward_utils import (
    compute_lift_readiness,
    compute_late_grasp_full_grip_mask,
    compute_middle_contact_gate,
    compute_slip_proxy,
    compute_upright_success_mask,
)
from .grasp_right_utils import scale, to_torch
from .demo_grasp_reset import DemoGraspResetBank, compute_demo_cup_spawn_local


class GraspRightEnv(DirectRLEnv):
    """OpenArm + RH56F1 오른손 파지 환경 (inspire_r_grasp_v1).

    Action: 12D
      [0:6]  palm pose (x,y,z,ez,ey,ex), 정규화 [-1,1] → Fabrics IK
      [6:12] 6D absolute hand synergy target (RH56F1 drive 6)
             grasp: APPROACH_POSE(-1) ~ GRASP_POSE(+1)
             post-grasp: GRASP_POSE(-1) ~ FULL_GRIP_POSE(+1)

    Episode:
      Grasp     phase (step 0~419):    Fabrics arm + absolute finger target
      Lift      phase (step 420~539):  policy lift + full-grip target refinement
      Stabilize phase (step 540~599):  hold/re-grip stabilization
      Transport phase: disabled
    """

    cfg: GraspRightEnvCfg

    _PALM_SENSOR_OFFSET_IN_FABRIC_PALM = (0.0, 0.03, 0.04)

    @staticmethod
    def _quat_xyzw_from_euler_zyx(euler_zyx: torch.Tensor) -> torch.Tensor:
        """Convert ZYX Euler angles to quaternion ordered as (x, y, z, w)."""
        batch = euler_zyx.shape[0]
        unit_x = euler_zyx.new_tensor([1.0, 0.0, 0.0]).expand(batch, -1)
        unit_y = euler_zyx.new_tensor([0.0, 1.0, 0.0]).expand(batch, -1)
        unit_z = euler_zyx.new_tensor([0.0, 0.0, 1.0]).expand(batch, -1)
        qz = quat_from_angle_axis(euler_zyx[:, 0], unit_z)
        qy = quat_from_angle_axis(euler_zyx[:, 1], unit_y)
        qx = quat_from_angle_axis(euler_zyx[:, 2], unit_x)
        quat_wxyz = quat_mul(quat_mul(qz, qy), qx)
        return quat_wxyz[:, [1, 2, 3, 0]]

    def _fabric_palm_pose_from_sensor_target(self, palm_sensor_pose: torch.Tensor) -> torch.Tensor:
        """Convert desired palm sensor pose to the Fabric palm_link pose target."""
        batch = palm_sensor_pose.shape[0]
        unit_x = palm_sensor_pose.new_tensor([1.0, 0.0, 0.0]).expand(batch, -1)
        unit_y = palm_sensor_pose.new_tensor([0.0, 1.0, 0.0]).expand(batch, -1)
        unit_z = palm_sensor_pose.new_tensor([0.0, 0.0, 1.0]).expand(batch, -1)
        qz = quat_from_angle_axis(palm_sensor_pose[:, 3], unit_z)
        qy = quat_from_angle_axis(palm_sensor_pose[:, 4], unit_y)
        qx = quat_from_angle_axis(palm_sensor_pose[:, 5], unit_x)
        palm_quat_wxyz = quat_mul(quat_mul(qz, qy), qx)
        sensor_offset = palm_sensor_pose.new_tensor(self._PALM_SENSOR_OFFSET_IN_FABRIC_PALM).expand(batch, -1)

        fabric_palm_pose = palm_sensor_pose.clone()
        fabric_palm_pose[:, :3] = palm_sensor_pose[:, :3] - quat_apply(palm_quat_wxyz, sensor_offset)
        return fabric_palm_pose

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

        self.arm_dof_indices  = self.actuated_dof_indices[:NUM_ARM_DOF]
        self.hand_dof_indices = self.actuated_dof_indices[NUM_ARM_DOF:]
        # real2sim DR: 소비처(_apply_real2sim_actuator_randomization)는 .values() 의 joint_ids 만 사용.
        # RH56F1 손은 drive 6관절을 단일 그룹으로 묶는다 (Tesollo 의 abduction/curl/pip/dip 분할 불필요).
        self.real2sim_actuator_group_indices = {
            "openarm_right_arm": self.arm_dof_indices,
            "rh56f1_right_drive": self.hand_dof_indices,
            "openarm_left_arm": self.left_arm_dof_indices,
        }

        # body indices (RH56F1: 말단 손가락 링크 = fingertip = distal). cfg.right_tip_contact_links 사용.
        _tip_names = list(self.cfg.right_tip_contact_links)  # thumb_4, index_2, middle_2, ring_2, little_2
        self.fingertip_body_indices: list[int] = [
            self.robot.data.body_names.index(name) for name in _tip_names
        ]
        _palm_name = self.cfg.hand_body_names[0]
        self.palm_body_index: int = (
            self.robot.data.body_names.index(_palm_name)
            if _palm_name in self.robot.data.body_names
            else -1
        )
        # distal4 = fingertip (RH56F1 는 말단 링크가 곧 distal)
        self.distal4_body_indices: list[int] = list(self.fingertip_body_indices)

        # middle phalanx: RH56F1 손가락은 2 링크뿐 → 중간 phalanx 대신 proximal 링크(_1) 사용.
        # (middle_to_cup obs 의 기하 의미 유지: 손가락 밑마디→컵 벡터, FK 로 sim2real 가능)
        _proximal_names = [
            "rh56f1_right_right_thumb_1", "rh56f1_right_right_index_1",
            "rh56f1_right_right_middle_1", "rh56f1_right_right_ring_1",
            "rh56f1_right_right_little_1",
        ]
        self.middle3_body_indices: list[int] = [
            self.robot.data.body_names.index(name)
            for name in _proximal_names
            if name in self.robot.data.body_names
        ]

        # ----------------------------------------------------------------
        # Palm pose workspace (안전 한계)
        # ----------------------------------------------------------------
        self.palm_mins = to_torch(PALM_POSE_MINS_FUNC(cfg.max_pose_angle), device=self.device)
        self.palm_maxs = to_torch(PALM_POSE_MAXS_FUNC(cfg.max_pose_angle), device=self.device)

        # Delta palm action 범위
        _delta_rad = math.radians(cfg.palm_delta_rot_deg)
        self.delta_mins = to_torch([
            -cfg.palm_delta_xyz, -cfg.palm_delta_xyz, -cfg.palm_delta_xyz,
            -_delta_rad, -_delta_rad, -_delta_rad,
        ], device=self.device)
        self.delta_maxs = to_torch([
            cfg.palm_delta_xyz, cfg.palm_delta_xyz, cfg.palm_delta_xyz,
            _delta_rad, _delta_rad, _delta_rad,
        ], device=self.device)
        _lift_delta_rad = math.radians(cfg.lift_palm_delta_rot_deg)
        self.lift_delta_mins = to_torch([
            -cfg.lift_palm_delta_xyz, -cfg.lift_palm_delta_xyz, -cfg.lift_palm_delta_xyz,
            -_lift_delta_rad, -_lift_delta_rad, -_lift_delta_rad,
        ], device=self.device)
        self.lift_delta_maxs = to_torch([
            cfg.lift_palm_delta_xyz, cfg.lift_palm_delta_xyz, cfg.lift_palm_delta_xyz,
            _lift_delta_rad, _lift_delta_rad, _lift_delta_rad,
        ], device=self.device)

        # pregrasp palm pose 버퍼
        self.pregrasp_palm_pose_buf   = torch.zeros(self.num_envs, 6, device=self.device)
        self.grasp_anchor_palm_pose_buf = torch.zeros(self.num_envs, 6, device=self.device)
        self.lift_palm_start_pose_buf = torch.zeros(self.num_envs, 6, device=self.device)
        self.demo_lift_palm_target_buf = torch.zeros(self.num_envs, 6, device=self.device)
        self.demo_grasp_reset_bank = (
            DemoGraspResetBank.from_hdf5_paths(cfg.demo_grasp_pose_paths, device=self.device)
            if cfg.enable_demo_grasp_reset
            else None
        )

        # ----------------------------------------------------------------
        # Hand 관절 한계 (absolute synergy target 클램프용)
        # soft_joint_pos_limits: (num_envs, num_joints, 2) — [lower, upper]
        # ----------------------------------------------------------------
        hand_limits = self.robot.data.soft_joint_pos_limits[0, self.hand_dof_indices, :]  # (6, 2)
        self.hand_joint_lower_limits = hand_limits[:, 0].contiguous()  # (6,)
        self.hand_joint_upper_limits = hand_limits[:, 1].contiguous()  # (6,)

        # RH56F1 6D 직접 제어: 모든 drive 관절(thumb_1,thumb_2,index_1,middle_1,ring_1,little_1)
        # 을 RL 이 제어한다. (Tesollo 의 abduction 고정 마스크 불필요)
        # ----------------------------------------------------------------
        # 접근/파지 자세 (reset 및 reference pose 용). RH56F1 6D.
        # ----------------------------------------------------------------
        self.hand_approach_pose   = to_torch(HAND_APPROACH_POSE,   device=self.device)  # (6,)
        self.hand_grasp_pose      = to_torch(HAND_GRASP_POSE,      device=self.device)  # (6,)
        self.hand_full_grip_pose  = to_torch(HAND_FULL_GRIP_POSE,  device=self.device)  # (6,)
        # 6D action 내 thumb 관절 인덱스: [thumb_1(abduction)=0, thumb_2(flexion)=1]
        self.thumb_joint_indices = torch.tensor([0, 1], dtype=torch.long, device=self.device)
        # 관절 한계: USD soft_joint_pos_limits (RH56F1 raw 한계) 를 그대로 사용.
        # (Tesollo 의 approach 기반 closure 재조정 제거 — KISS, 추후 튜닝)
        self.hand_joint_lower_limits = self.hand_joint_lower_limits.contiguous()
        self.hand_joint_upper_limits = self.hand_joint_upper_limits.contiguous()

        # ----------------------------------------------------------------
        # 로봇 시작 자세 (arm: ARM_START_POSE, hand: HAND_APPROACH_POSE)
        # 열린 approach 자세에서 시작해 컵 spawn 시 손가락 관통을 피한다.
        # ----------------------------------------------------------------
        arm_start   = to_torch(ARM_START_POSE,   device=self.device)
        hand_start  = to_torch(HAND_APPROACH_POSE,  device=self.device)
        robot_start = torch.cat([arm_start, hand_start], dim=0)
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
        self.distal4_pos     = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.middle3_pos     = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.actions         = torch.zeros(self.num_envs, cfg.num_actions, device=self.device)
        self.prev_actions    = torch.full((self.num_envs, cfg.num_actions), 0.0, device=self.device)
        self._palm_target_delta_buf = torch.zeros(self.num_envs, 6, device=self.device)
        self._ema_palm_action = torch.zeros(self.num_envs, 6, device=self.device)

        # ----------------------------------------------------------------
        # Pregrasp / Lift 버퍼
        # ----------------------------------------------------------------
        self.pregrasp_arm_pos_buf = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        self.lift_finger_pos_buf  = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        self.is_grasp_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_lift_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_stabilize_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_post_grasp_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ----------------------------------------------------------------
        # Hand joint targets (absolute synergy 결과)
        # ----------------------------------------------------------------
        self.hand_joint_targets = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)

        # ----------------------------------------------------------------
        # 접촉 상태 버퍼
        # ----------------------------------------------------------------
        self.contact_force_xyz_raw   = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.contact_force_raw       = torch.zeros(self.num_envs, NUM_FINGERTIPS, device=self.device)
        self.binary_contact_buf    = torch.zeros(self.num_envs, NUM_FINGERTIPS, dtype=torch.bool, device=self.device)
        self.num_contacts_buf      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.distal_contact_force_raw  = torch.zeros(self.num_envs, NUM_TIP_SENSORS, device=self.device)
        self.distal_binary_contact_buf = torch.zeros(self.num_envs, NUM_TIP_SENSORS, dtype=torch.bool, device=self.device)

        self.middle_contact_force_raw  = torch.zeros(self.num_envs, NUM_TIP_SENSORS, device=self.device)
        self.middle_binary_contact_buf = torch.zeros(self.num_envs, NUM_TIP_SENSORS, dtype=torch.bool, device=self.device)

        self.palm_contact_force_raw  = torch.zeros(self.num_envs, device=self.device)
        self.palm_binary_contact_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.distal_contact_force_xyz = torch.zeros(self.num_envs, NUM_TIP_SENSORS, 3, device=self.device)
        self.middle_contact_force_xyz = torch.zeros(self.num_envs, NUM_TIP_SENSORS, 3, device=self.device)
        self.palm_contact_force_xyz   = torch.zeros(self.num_envs, 3,                  device=self.device)

        self._prev_total_grip_force_buf = torch.zeros(self.num_envs, device=self.device)
        self._prev_num_contacts_buf = torch.zeros(self.num_envs, device=self.device)
        self._prev_middle_contacts_buf = torch.zeros(self.num_envs, device=self.device)
        self._prev_cup_tilt_deg_buf = torch.zeros(self.num_envs, device=self.device)
        self._contact_persistence_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._lift_contact_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grasp_started_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_anchor_set_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._approach_ready_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._approach_timeout_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_from_timeout_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lift_contact_ready_latched_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lift_started_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lift_start_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._contacts_at_lift_start_buf = torch.zeros(self.num_envs, device=self.device)
        self._palm_at_lift_start_buf = torch.zeros(self.num_envs, device=self.device)
        self._grasp_tilt_at_lift_start_buf = torch.zeros(self.num_envs, device=self.device)
        self._force_ratio_at_lift_start_buf = torch.zeros(self.num_envs, device=self.device)
        self._full_grip_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._lift_success_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._full_grip_ready_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._full_grip_ready_latched_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stabilize_started_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stabilize_start_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grip_ready_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grip_ready_latched_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ----------------------------------------------------------------
        # 기타 버퍼
        # ----------------------------------------------------------------
        self.success_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cup_tipping_cos = math.cos(math.radians(cfg.cup_tipping_max_deg))
        self.episode_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_lift_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_stabilize_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lift_success_latched_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stabilize_success_latched_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._phase_curriculum_stage = min(max(int(cfg.phase_curriculum_initial_stage), 0), 1)
        self._episode_curriculum_stage_buf = torch.full(
            (self.num_envs,),
            1 if not cfg.enable_phase_curriculum else self._phase_curriculum_stage,
            dtype=torch.long,
            device=self.device,
        )
        self._total_episodes: int = 0
        self._successful_episodes: int = 0

        # ----------------------------------------------------------------
        # Eval 로깅
        # ----------------------------------------------------------------
        self._eval_grip_at_lift = torch.zeros(self.num_envs, device=self.device)
        self._eval_finger_actions_at_lift = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        self._last_grasp_finger_action    = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        self._eval_episode_started        = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._eval_grasp_action_sum       = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        self._eval_grasp_action_sq_sum    = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        self._eval_grasp_action_min       = torch.full((self.num_envs, NUM_HAND_DOF), float("inf"), device=self.device)
        self._eval_grasp_action_max       = torch.full((self.num_envs, NUM_HAND_DOF), float("-inf"), device=self.device)
        self._eval_grasp_action_count     = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._eval_lift_action_sum        = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        self._eval_lift_action_sq_sum     = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        self._eval_lift_force_delta_sum   = torch.zeros(self.num_envs, device=self.device)
        self._eval_lift_contact_delta_sum = torch.zeros(self.num_envs, device=self.device)
        self._eval_lift_action_count      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._eval_lift_snapshot_valid    = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._eval_records: list[dict] = []

        # ----------------------------------------------------------------
        # Bead 무게 도메인 랜덤화
        # ----------------------------------------------------------------
        beads_per_layer = 5
        _bead_offsets = []
        for i in range(cfg.num_beads):
            layer = i // beads_per_layer
            slot  = i % beads_per_layer
            angle  = (2 * math.pi * slot / beads_per_layer) + 0.35 * layer
            radius = 0.014 + 0.004 * (layer % 2)
            z      = cfg.bead_spawn_z_offset + 0.006 + 0.014 * layer
            _bead_offsets.append([radius * math.cos(angle), radius * math.sin(angle), z])
        self._bead_offsets_b = torch.tensor(_bead_offsets, dtype=torch.float32, device=self.device)

        hidden_cols = 5
        hidden_spacing = 0.03
        hidden_base_x = -1.20
        hidden_base_y = -0.60
        hidden_z = 0.02
        _hidden_bead_offsets = []
        for i in range(cfg.num_beads):
            row = i // hidden_cols
            col = i % hidden_cols
            _hidden_bead_offsets.append([
                hidden_base_x - hidden_spacing * col,
                hidden_base_y - hidden_spacing * row,
                hidden_z,
            ])
        self._hidden_bead_offsets_b = torch.tensor(
            _hidden_bead_offsets, dtype=torch.float32, device=self.device
        )

        self._bead_mass_normalized = torch.zeros(self.num_envs, device=self.device)
        self._bead_count_initial = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._bead_count_current = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._bead_count_target = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._dynamic_bead_add_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._dynamic_bead_spawned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cup_friction_static = torch.zeros(self.num_envs, device=self.device)

        self._warm_state_export: dict[str, torch.Tensor] | None = None
        self._warm_state_export_count = 0
        self._warm_state_export_written = False
        self._warm_state_export_path = Path(str(cfg.warm_state_export_path)).expanduser()
        if cfg.enable_warm_state_export:
            self._init_warm_state_export()

        # 6.2: moving-window ADR trigger (최근 N 에피소드 성공률)
        _win = cfg.adr_window_size if cfg.adr_window_size > 0 else 500
        self._success_window: deque = deque(maxlen=_win)
        self._lift_success_window: deque[int] = deque(maxlen=_win)
        self._stabilize_success_window: deque[int] = deque(maxlen=_win)

        # 6.3: per-bin 에피소드 성공 카운터 (bead level 0~3: 0/10/20/30 beads)
        self._total_episodes_bin: list[int] = [0, 0, 0, 0]
        self._successful_episodes_bin: list[int] = [0, 0, 0, 0]

        # ----------------------------------------------------------------
        # ADR — contact curriculum (threshold=0.1, 먼저 진행)
        # ----------------------------------------------------------------
        if cfg.enable_contact_adr:
            self.contact_adr = GraspADR(
                custom_cfg=cfg.contact_adr_custom_cfg,
                num_increments=cfg.contact_adr_num_increments,
                increment_interval=cfg.contact_adr_increment_interval,
                trigger_threshold=cfg.contact_adr_trigger_threshold,
            )
        else:
            self.contact_adr = None

        # ----------------------------------------------------------------
        # ADR — 난이도 (threshold=0.8, contact ADR 이후 진행)
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

        # cspace attractor: reset warm-start hand pose와 일치
        cspace_default = self.fabric.default_config.clone()
        cspace_default[:, NUM_ARM_DOF:] = self.hand_approach_pose.unsqueeze(0).expand(self.num_envs, -1)
        self.fabric.default_config.copy_(cspace_default)

        # 초기 액션: 0 → palm pose workspace 중심, finger target은 approach/grasp 중간점
        self.actions.zero_()

    # ------------------------------------------------------------------
    # Scene 설정
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cup   = RigidObject(self.cfg.cup_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.beads = RigidObjectCollection(self.cfg.beads_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cup"]   = self.cup
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_object_collections["beads"] = self.beads

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

        # RH56F1: 별도 distal/middle phalanx 접촉 센서 없음. reward에서는 fingertip/palm만 사용.
        # tip 센서(위 _tip_sensors)가 force_sensor 패드 접촉을 포착한다.
        self._palm_sensor = ContactSensor(self.cfg.palm_sensor_cfg)
        self.scene.sensors["palm_sensor"] = self._palm_sensor

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

        self.world_model = WorldMeshesModel(
            batch_size=self.num_envs,
            max_objects_per_env=self.cfg.fabrics_max_objects_per_env,
            device=self.device,
            world_filename="open_tesollo_boxes_no_table",
        )
        self.object_ids, self.object_indicator = self.world_model.get_object_ids()

        self.timestep = self.cfg.fabrics_dt

        self.fabric = OpenArmRh56f1PoseFabric(
            self.num_envs, self.device, self.timestep,
            graph_capturable=False,
            use_hand_fabric=False,
        )
        num_joints = self.fabric.num_joints

        self.fabric_integrator = DisplacementIntegrator(self.fabric)

        self.fabric_q   = self.robot_start_joint_pos.clone().contiguous()
        self.fabric_qd  = torch.zeros(self.num_envs, num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, num_joints, device=self.device)

        # use_hand_fabric=False → set_features 에서 무시됨. 차원만 NUM_HAND_DOF(6) 로 유지.
        self.hand_pca_targets  = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        self.palm_pose_targets = torch.zeros(self.num_envs, 6, device=self.device)
        self.fabric_damping_gain = self.cfg.fabrics_damping_gain * torch.ones(self.num_envs, 1, device=self.device)

        self._reset_chunk = self.cfg.reset_fabric_chunk_size
        self._reset_fabric = OpenArmRh56f1PoseFabric(
            self._reset_chunk, self.device, self.timestep,
            graph_capturable=False,
            use_hand_fabric=False,
        )
        self._reset_integrator = DisplacementIntegrator(self._reset_fabric)

        reset_cspace = self._reset_fabric.default_config.clone()
        reset_cspace[:, NUM_ARM_DOF:] = self.hand_approach_pose.unsqueeze(0).expand(self._reset_chunk, -1)
        self._reset_fabric.default_config.copy_(reset_cspace)

        self._reset_pca     = torch.zeros(self._reset_chunk, NUM_HAND_DOF, device=self.device)
        self._reset_damping = 10.0 * torch.ones(self._reset_chunk, 1, device=self.device)
        self._reset_world   = WorldMeshesModel(
            batch_size=self._reset_chunk,
            max_objects_per_env=self.cfg.fabrics_max_objects_per_env,
            device=self.device,
            world_filename="open_tesollo_boxes_no_table",
        )
        self._reset_obj_ids, self._reset_obj_indicator = self._reset_world.get_object_ids()

        if self.cfg.cache_pregrasp_reset and self.demo_grasp_reset_bank is None:
            self._build_pregrasp_cache()

    # ------------------------------------------------------------------
    # Pregrasp grid 캐시
    # ------------------------------------------------------------------
    def _build_pregrasp_cache(self) -> None:
        _N = 13
        xs = torch.linspace(
            self.cfg.object_spawn_x_center - self.cfg.object_spawn_xy_range,
            self.cfg.object_spawn_x_center + self.cfg.object_spawn_xy_range,
            _N, device=self.device,
        )
        ys = torch.linspace(
            self.cfg.object_spawn_y_center - self.cfg.object_spawn_xy_range,
            self.cfg.object_spawn_y_center + self.cfg.object_spawn_xy_range,
            _N, device=self.device,
        )
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        flat_x, flat_y = gx.flatten(), gy.flatten()
        M = flat_x.shape[0]

        palm_sensor = torch.zeros(M, 6, device=self.device)
        palm_sensor[:, 0] = flat_x + self.cfg.pregrasp_offset_x
        palm_sensor[:, 1] = flat_y + self.cfg.pregrasp_offset_y
        palm_sensor[:, 2] = self.cfg.object_spawn_z + self.cfg.pregrasp_offset_z
        palm_sensor[:, 3] = math.radians(90.0)
        palm_sensor[:, 4] = math.radians(0.0)
        palm_sensor[:, 5] = math.radians(90.0)
        palm = self._fabric_palm_pose_from_sensor_target(palm_sensor)
        palm = torch.max(
            torch.min(palm, self.palm_maxs.unsqueeze(0)),
            self.palm_mins.unsqueeze(0),
        )

        q_init = self.robot_start_joint_pos[0].unsqueeze(0).expand(M, -1).contiguous()
        dummy  = torch.arange(M, device=self.device)
        q_out  = self._run_reset_fabric(dummy, palm, q_init.clone())

        self._cache_q_arm = q_out[:, :NUM_ARM_DOF].view(_N, _N, NUM_ARM_DOF).contiguous()
        self._cache_xs    = xs
        self._cache_ys    = ys
        self._cache_n     = _N

    # ------------------------------------------------------------------
    # Reset 전용 Fabrics rollout
    # ------------------------------------------------------------------
    def _run_reset_fabric(
        self,
        env_ids: torch.Tensor,
        palm_pose: torch.Tensor,
        q_init: torch.Tensor,
    ) -> torch.Tensor:
        n = len(env_ids)
        C = self._reset_chunk
        q_out = torch.zeros_like(q_init)

        for start in range(0, n, C):
            end = min(start + C, n)
            m   = end - start

            pp = palm_pose[start:end]
            qi = q_init[start:end]

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

    def _sample_bead_counts(self, n: int, min_count: int, max_count: int) -> torch.Tensor:
        low = min(max(int(min_count), 0), int(self.cfg.num_beads))
        high = min(max(int(max_count), low), int(self.cfg.num_beads))
        return torch.randint(low, high + 1, (n,), device=self.device)

    def _init_warm_state_export(self) -> None:
        target_count = int(self.cfg.warm_state_target_count)
        if target_count <= 0:
            raise ValueError(
                f"warm_state_target_count must be positive, got {target_count}"
            )
        success_source = str(self.cfg.warm_state_success_source)
        if success_source not in ("stage", "lift", "stabilize"):
            raise ValueError(
                "warm_state_success_source must be one of "
                "'stage', 'lift', 'stabilize'"
            )

        self._warm_state_export = {
            "arm_joint_pos": torch.empty(target_count, NUM_ARM_DOF, dtype=torch.float32),
            "hand_joint_pos": torch.empty(target_count, NUM_HAND_DOF, dtype=torch.float32),
            "palm_pose_quat_xyzw": torch.empty(target_count, 7, dtype=torch.float32),
            "palm_pose_euler_zyx": torch.empty(target_count, 6, dtype=torch.float32),
            "cup_pos_local": torch.empty(target_count, 3, dtype=torch.float32),
            "cup_quat_wxyz": torch.empty(target_count, 4, dtype=torch.float32),
            "num_contacts": torch.empty(target_count, dtype=torch.float32),
            "bead_state_local": torch.empty(
                target_count, int(self.cfg.num_beads), 13, dtype=torch.float32
            ),
            "bead_count_initial": torch.empty(target_count, dtype=torch.int64),
            "bead_count_current": torch.empty(target_count, dtype=torch.int64),
            "bead_count_target": torch.empty(target_count, dtype=torch.int64),
            "dynamic_bead_spawned": torch.empty(target_count, dtype=torch.bool),
            "cup_friction_static": torch.empty(target_count, dtype=torch.float32),
        }

    def _select_warm_state_export_success(
        self, env_ids: torch.Tensor, started_mask: torch.Tensor
    ) -> torch.Tensor:
        source = str(self.cfg.warm_state_success_source)
        if source == "stage":
            success = self.episode_success_buf[env_ids]
        elif source == "lift":
            success = self.episode_lift_success_buf[env_ids]
        else:
            success = self.episode_stabilize_success_buf[env_ids]
        return success & started_mask.to(device=self.device, dtype=torch.bool)

    def _maybe_export_warm_states(
        self, env_ids: torch.Tensor, started_mask: torch.Tensor
    ) -> None:
        if self._warm_state_export is None or self._warm_state_export_written:
            return
        target_count = self._warm_state_export["arm_joint_pos"].shape[0]
        if self._warm_state_export_count >= target_count:
            return

        success_mask = self._select_warm_state_export_success(env_ids, started_mask)
        if not success_mask.any():
            return

        success_env_ids = env_ids[success_mask]
        remaining = target_count - self._warm_state_export_count
        success_env_ids = success_env_ids[:remaining]
        count = int(success_env_ids.numel())
        if count == 0:
            return

        start = self._warm_state_export_count
        end = start + count
        export = self._warm_state_export

        palm_euler = self.palm_pose_targets[success_env_ids]
        palm_quat = torch.zeros(count, 7, device=self.device)
        palm_quat[:, :3] = palm_euler[:, :3]
        palm_quat[:, 3:7] = self._quat_xyzw_from_euler_zyx(palm_euler[:, 3:6])

        bead_state_local = self.beads.data.object_state_w[success_env_ids].clone()
        bead_state_local[..., :3] -= self.scene.env_origins[success_env_ids].unsqueeze(1)

        export["arm_joint_pos"][start:end] = (
            self.robot.data.joint_pos[success_env_ids][:, self.arm_dof_indices].detach().cpu()
        )
        export["hand_joint_pos"][start:end] = (
            self.robot.data.joint_pos[success_env_ids][:, self.hand_dof_indices].detach().cpu()
        )
        export["palm_pose_quat_xyzw"][start:end] = palm_quat.detach().cpu()
        export["palm_pose_euler_zyx"][start:end] = palm_euler.detach().cpu()
        export["cup_pos_local"][start:end] = (
            (self.cup.data.root_pos_w[success_env_ids] - self.scene.env_origins[success_env_ids])
            .detach()
            .cpu()
        )
        export["cup_quat_wxyz"][start:end] = self.cup.data.root_quat_w[success_env_ids].detach().cpu()
        export["num_contacts"][start:end] = self.num_contacts_buf[success_env_ids].float().detach().cpu()
        export["bead_state_local"][start:end] = bead_state_local.detach().cpu()
        export["bead_count_initial"][start:end] = self._bead_count_initial[success_env_ids].detach().cpu()
        export["bead_count_current"][start:end] = self._bead_count_current[success_env_ids].detach().cpu()
        export["bead_count_target"][start:end] = self._bead_count_target[success_env_ids].detach().cpu()
        export["dynamic_bead_spawned"][start:end] = self._dynamic_bead_spawned[success_env_ids].detach().cpu()
        export["cup_friction_static"][start:end] = self._cup_friction_static[success_env_ids].detach().cpu()

        self._warm_state_export_count = end
        if self._warm_state_export_count >= target_count:
            self._write_warm_state_export_file()

    def _write_warm_state_export_file(self) -> None:
        if self._warm_state_export is None or self._warm_state_export_written:
            return
        count = self._warm_state_export_count
        if count <= 0:
            return

        path = self._warm_state_export_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")

        with h5py.File(tmp_path, "w") as h5:
            grp = h5.create_group("warm_states")
            for key, value in self._warm_state_export.items():
                grp.create_dataset(key, data=value[:count].numpy())

            attrs = h5.attrs
            attrs["meta/schema_version"] = 2
            attrs["meta/source_task"] = "5g_grasp_right-v11"
            attrs["meta/success_source"] = str(self.cfg.warm_state_success_source)
            attrs["meta/object_spawn_z"] = float(self.cfg.object_spawn_z)
            attrs["meta/object_spawn_xy_range"] = float(self.cfg.object_spawn_xy_range)
            attrs["meta/num_beads"] = int(self.cfg.num_beads)
            attrs["meta/bead_single_mass"] = float(self.cfg.bead_single_mass)
            attrs["meta/bead_scale"] = float(self.cfg.bead_scale)
            attrs["meta/cup_base_mass"] = float(self.cfg.cup_base_mass)
            attrs["meta/palm_min_x"] = float(self.palm_mins[0])
            attrs["meta/palm_min_y"] = float(self.palm_mins[1])
            attrs["meta/palm_min_z"] = float(self.palm_mins[2])
            attrs["meta/palm_max_x"] = float(self.palm_maxs[0])
            attrs["meta/palm_max_y"] = float(self.palm_maxs[1])
            attrs["meta/palm_max_z"] = float(self.palm_maxs[2])

        os.replace(tmp_path, path)
        self._warm_state_export_written = True
        print(
            f"[5g_grasp_right_v11] exported {count} pour warm states to {path}",
            flush=True,
        )

    @staticmethod
    def _window_success_rate(window: deque[int]) -> float:
        if len(window) == 0:
            return 0.0
        return float(sum(window)) / float(len(window))

    def _curriculum_success_rate(self) -> tuple[float, int]:
        if not self.cfg.enable_phase_curriculum:
            if len(self._success_window) > 0:
                return self._window_success_rate(self._success_window), len(self._success_window)
            return self._successful_episodes / max(self._total_episodes, 1), 0

        if self._phase_curriculum_stage <= 0:
            window = self._lift_success_window
        elif self._phase_curriculum_stage == 1:
            window = self._stabilize_success_window
        else:
            window = self._success_window

        if len(window) > 0:
            return self._window_success_rate(window), len(window)
        return 0.0, 0

    def _maybe_update_phase_curriculum(self) -> None:
        if not self.cfg.enable_phase_curriculum:
            return

        min_episodes = max(int(self.cfg.phase_curriculum_min_episodes), 1)
        if STABILIZE_PHASE_STEPS <= 0:
            return
        if (
            self._phase_curriculum_stage <= 0
            and len(self._lift_success_window) >= min_episodes
            and self._window_success_rate(self._lift_success_window)
            >= float(self.cfg.phase_curriculum_lift_success_threshold)
        ):
            self._phase_curriculum_stage = 1
            self._stabilize_success_window.clear()
            self._success_window.clear()
            return


    def _spawn_dynamic_beads(self, env_mask: torch.Tensor) -> None:
        if not self.cfg.physical_beads_enabled:
            return

        env_ids = env_mask.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return

        current_count = self._bead_count_current[env_ids]
        target_count = self._bead_count_target[env_ids].clamp(max=int(self.cfg.num_beads))
        cup_pos_w = self.cup.data.root_pos_w[env_ids]
        cup_quat_w = self.cup.data.root_quat_w[env_ids]

        for bead_index in range(int(self.cfg.num_beads)):
            activate = (current_count <= bead_index) & (target_count > bead_index)
            if not activate.any():
                continue

            active_env_ids = env_ids[activate]
            local_offset = self._bead_offsets_b[bead_index].unsqueeze(0).expand(active_env_ids.numel(), -1)
            bead_pos = cup_pos_w[activate] + quat_apply(cup_quat_w[activate], local_offset)

            bead_state = torch.zeros(active_env_ids.numel(), 1, 13, device=self.device)
            bead_state[:, 0, :3] = bead_pos
            bead_state[:, 0, 3] = 1.0

            object_ids = torch.tensor([bead_index], dtype=torch.long, device=self.device)
            self.beads.write_object_state_to_sim(
                bead_state,
                env_ids=active_env_ids,
                object_ids=object_ids,
            )

        self._bead_count_current[env_ids] = target_count
        self._bead_mass_normalized[env_ids] = target_count.float() / max(float(self.cfg.num_beads), 1.0)
        self._dynamic_bead_spawned[env_ids] = True

    # ------------------------------------------------------------------
    # 접촉력 업데이트
    # ------------------------------------------------------------------
    def _update_contact_forces(self) -> None:
        tip_xyz = torch.stack([
            s.data.force_matrix_w[:, 0, 0, :] for s in self._tip_sensors
        ], dim=1)
        tip_xyz = torch.nan_to_num(tip_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        tip_norms = tip_xyz.norm(dim=-1)

        self.contact_force_xyz_raw.copy_(tip_xyz)
        self.contact_force_raw.copy_(tip_norms)
        self.binary_contact_buf.copy_(tip_norms > CONTACT_FORCE_THRESHOLD)
        self.num_contacts_buf.copy_(self.binary_contact_buf.sum(dim=-1).long())

        # friction_forces_w: track_friction_forces 비활성화로 미사용
        # (get_friction_data()의 buffer overflow 문제로 제거)

        # RH56F1: 별도 distal phalanx 센서 없음. tip과 중복 계산하지 않는다.
        self.distal_contact_force_xyz.zero_()
        self.distal_contact_force_raw.zero_()
        self.distal_binary_contact_buf.zero_()

        # RH56F1: middle phalanx 센서 없음 → 항상 zeros (reward 의 middle 항목은 무기여).
        self.middle_contact_force_xyz.zero_()
        self.middle_contact_force_raw.zero_()
        self.middle_binary_contact_buf.zero_()

        palm_xyz = torch.nan_to_num(self._palm_sensor.data.net_forces_w[:, 0, :], nan=0.0, posinf=0.0, neginf=0.0)
        per_palm = palm_xyz.norm(dim=-1)
        self.palm_contact_force_xyz.copy_(palm_xyz)
        self.palm_contact_force_raw.copy_(per_palm)
        self.palm_binary_contact_buf.copy_(per_palm > CONTACT_FORCE_THRESHOLD)

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone()

        palm_action   = actions[:, :6]    # (N, 6) ∈ [-1, 1]
        finger_action = actions[:, 6:NUM_ACTIONS]  # (N, 6) ∈ [-1, 1] — absolute synergy target
        self._ema_palm_action.copy_(
            float(self.cfg.ema_action_alpha) * palm_action
            + (1.0 - float(self.cfg.ema_action_alpha)) * self._ema_palm_action
        )
        fabric_palm_action = self._ema_palm_action

        # ---- Phase 판정 ----
        stabilize_curriculum_enabled = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if not self.cfg.enable_phase_curriculum
            else self._episode_curriculum_stage_buf >= 1
        )

        just_entering_lift = self._lift_contact_ready_latched_buf & (~self._lift_started_buf)
        if just_entering_lift.any():
            prev_finger_action = self._last_grasp_finger_action[just_entering_lift]
            self._eval_grip_at_lift[just_entering_lift] = (
                0.5 * (prev_finger_action + 1.0)
            ).mean(dim=-1)
            self._eval_finger_actions_at_lift[just_entering_lift] = prev_finger_action
            self._eval_lift_snapshot_valid[just_entering_lift] = True
            self._lift_start_step_buf[just_entering_lift] = self.episode_length_buf[just_entering_lift]

        self.lift_finger_pos_buf = torch.where(
            just_entering_lift.unsqueeze(1),
            self.robot.data.joint_pos[:, self.hand_dof_indices],
            self.lift_finger_pos_buf,
        )
        self.lift_palm_start_pose_buf = torch.where(
            just_entering_lift.unsqueeze(1),
            self.palm_pose_targets,
            self.lift_palm_start_pose_buf,
        )
        self._lift_started_buf |= just_entering_lift

        just_entering_stabilize = (
            (STABILIZE_PHASE_STEPS > 0)
            & stabilize_curriculum_enabled
            & self._lift_success_latched_buf
            & (~self._stabilize_started_buf)
        )
        if just_entering_stabilize.any():
            self._stabilize_start_step_buf[just_entering_stabilize] = (
                self.episode_length_buf[just_entering_stabilize]
            )
        self._stabilize_started_buf |= just_entering_stabilize

        is_stabilize = self._stabilize_started_buf
        is_lift = self._lift_started_buf & (~is_stabilize)
        is_grasp = ~self._lift_started_buf
        is_post_grasp = self._lift_started_buf
        approach_elapsed_ok = self.episode_length_buf >= int(self.cfg.approach_min_steps)
        cup_quat_inv = self.object_rot.clone()
        cup_quat_inv[:, 1:] = -cup_quat_inv[:, 1:]
        approach_palm_local = quat_apply(
            cup_quat_inv,
            self.palm_center_pos - self.object_pos,
        )
        approach_palm_radial = approach_palm_local[:, :2].norm(dim=-1)
        approach_palm_local_z = approach_palm_local[:, 2]
        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cup_z_world = quat_apply(self.object_rot, z_local)
        approach_upright = compute_upright_success_mask(
            cup_z_world[:, 2],
            self.cfg.approach_upright_max_deg,
        )
        approach_ready = (
            approach_elapsed_ok
            & (approach_palm_radial >= self.cfg.approach_palm_radial_min)
            & (approach_palm_radial <= self.cfg.approach_palm_radial_max)
            & (approach_palm_local_z >= self.cfg.approach_palm_local_z_min)
            & (approach_palm_local_z <= self.cfg.approach_palm_local_z_max)
            & (self.num_contacts_buf <= int(self.cfg.approach_max_tip_contacts))
            & approach_upright
        )
        approach_timeout = (
            approach_elapsed_ok
            & (int(self.cfg.approach_timeout_steps) > 0)
            & (self.episode_length_buf >= int(self.cfg.approach_timeout_steps))
        )
        self._approach_ready_buf.copy_(approach_ready)
        self._approach_timeout_buf.copy_(approach_timeout)
        just_entering_close_grasp = (
            is_grasp
            & (~self._grasp_started_buf)
            & (approach_ready | approach_timeout)
        )
        just_entering_timeout_grasp = just_entering_close_grasp & approach_timeout & (~approach_ready)
        self._grasp_started_buf |= just_entering_close_grasp
        self._grasp_from_timeout_buf |= just_entering_timeout_grasp

        self.is_grasp_phase.copy_(is_grasp)
        self.is_lift_phase.copy_(is_lift)
        self.is_stabilize_phase.copy_(is_stabilize)
        self.is_post_grasp_phase.copy_(is_post_grasp)

        just_entering_close_grasp = just_entering_close_grasp & (~self._grasp_anchor_set_buf)
        if just_entering_close_grasp.any():
            self.grasp_anchor_palm_pose_buf[just_entering_close_grasp] = (
                self.palm_pose_targets[just_entering_close_grasp]
            )
            self._grasp_anchor_set_buf[just_entering_close_grasp] = True

        if self.cfg.physical_beads_enabled and self.cfg.dynamic_bead_spawn_enabled:
            stabilize_elapsed = (
                self.episode_length_buf - self._stabilize_start_step_buf
            ).clamp(min=0)
            dynamic_bead_delay = max(int(self.cfg.dynamic_bead_spawn_step) - STABILIZE_START_STEP, 0)
            dynamic_bead_mask = (
                (stabilize_elapsed == dynamic_bead_delay)
                & is_stabilize
                & self._lift_success_latched_buf
                & (~self._dynamic_bead_spawned)
            )
            if dynamic_bead_mask.any():
                self._spawn_dynamic_beads(dynamic_bead_mask)

        # Eval: grasp phase 동안 finger action 버퍼링
        grasp_mask = is_grasp
        if grasp_mask.any():
            self._eval_episode_started[grasp_mask] = True
            self._last_grasp_finger_action[grasp_mask] = finger_action[grasp_mask]
            self._eval_grasp_action_sum[grasp_mask] += finger_action[grasp_mask]
            self._eval_grasp_action_sq_sum[grasp_mask] += finger_action[grasp_mask].square()
            self._eval_grasp_action_min[grasp_mask] = torch.minimum(
                self._eval_grasp_action_min[grasp_mask], finger_action[grasp_mask]
            )
            self._eval_grasp_action_max[grasp_mask] = torch.maximum(
                self._eval_grasp_action_max[grasp_mask], finger_action[grasp_mask]
            )
            self._eval_grasp_action_count[grasp_mask] += 1

        lift_mask = is_post_grasp
        if lift_mask.any():
            self._eval_lift_action_sum[lift_mask] += finger_action[lift_mask]
            self._eval_lift_action_sq_sum[lift_mask] += finger_action[lift_mask].square()
            self._eval_lift_action_count[lift_mask] += 1

        # ---- Palm pose 계산 ----
        approach_mask = is_grasp & (~self._grasp_started_buf)
        delta = scale(fabric_palm_action, self.delta_mins, self.delta_maxs)
        approach_palm_pose = self.pregrasp_palm_pose_buf + delta
        palm_mins = torch.minimum(self.palm_mins.unsqueeze(0), self.pregrasp_palm_pose_buf)
        palm_maxs = torch.maximum(self.palm_maxs.unsqueeze(0), self.pregrasp_palm_pose_buf)
        approach_palm_pose = torch.max(torch.min(approach_palm_pose, palm_maxs), palm_mins)

        grasp_delta = delta * float(self.cfg.grasp_palm_delta_scale)
        grasp_palm_pose = self.grasp_anchor_palm_pose_buf + grasp_delta
        cup_inward_xy = self.object_pos[:, :2] - self.grasp_anchor_palm_pose_buf[:, :2]
        cup_inward_xy = cup_inward_xy / cup_inward_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        grasp_palm_pose[:, :2] = (
            grasp_palm_pose[:, :2]
            + cup_inward_xy * float(self.cfg.grasp_palm_inward_offset)
        )
        grasp_palm_pose = torch.max(torch.min(grasp_palm_pose, palm_maxs), palm_mins)

        lift_elapsed_steps = torch.where(
            self._lift_started_buf,
            (self.episode_length_buf - self._lift_start_step_buf).clamp(min=0),
            torch.zeros_like(self.episode_length_buf),
        )
        lift_progress = (
            lift_elapsed_steps.float() / LIFT_PHASE_STEPS
        ).clamp(max=1.0).unsqueeze(1)
        if self.cfg.eval_mass_shift_enabled:
            shift_mask = self._lift_started_buf & (
                lift_elapsed_steps == int(self.cfg.eval_mass_shift_step)
            )
            if shift_mask.any():
                target_count = min(
                    max(int(self.cfg.eval_mass_shift_target_bead_count), 0),
                    int(self.cfg.num_beads),
                )
                self._bead_mass_normalized[shift_mask] = (
                    float(target_count) / max(float(self.cfg.num_beads), 1.0)
                )
                self._bead_count_current[shift_mask] = target_count
        lift_policy_delta = scale(fabric_palm_action, self.lift_delta_mins, self.lift_delta_maxs)
        # Phase B: stabilize에서 palm freeze — policy delta를 무시해 latch된 위치에 고정.
        # orientation은 아래 upright correction이 처리, 손가락 residual은 그대로(파지 강화 허용).
        lift_policy_delta = torch.where(
            is_stabilize.unsqueeze(1),
            torch.zeros_like(lift_policy_delta),
            lift_policy_delta,
        )
        lift_palm_pose = self.lift_palm_start_pose_buf + lift_policy_delta
        if self.demo_grasp_reset_bank is not None:
            demo_lift_palm_pose = torch.lerp(
                self.lift_palm_start_pose_buf,
                self.demo_lift_palm_target_buf,
                lift_progress,
            )
            demo_lift_palm_pose[:, 3:] = self.lift_palm_start_pose_buf[:, 3:]
            lift_palm_pose = torch.lerp(
                lift_palm_pose,
                demo_lift_palm_pose,
                lift_progress,
            )
        lift_palm_pose = torch.max(torch.min(lift_palm_pose, self.palm_maxs), self.palm_mins)

        upright_blend_steps = max(int(self.cfg.stabilize_upright_orientation_blend_steps), 1)
        stabilize_elapsed_steps = (
            self.episode_length_buf - self._stabilize_start_step_buf
        ).clamp(min=0)
        stabilize_upright_progress = (
            stabilize_elapsed_steps.float() / float(upright_blend_steps)
        ).clamp(max=1.0).unsqueeze(1)
        if self.cfg.stabilize_upright_orientation_enabled:
            lift_palm_pose = self._apply_upright_palm_orientation_correction(
                lift_palm_pose,
                is_stabilize,
                stabilize_upright_progress,
            )

        palm_pose = torch.where(
            is_post_grasp.unsqueeze(1),
            lift_palm_pose,
            torch.where(approach_mask.unsqueeze(1), approach_palm_pose, grasp_palm_pose),
        )
        self._palm_target_delta_buf.copy_(palm_pose - self.palm_pose_targets)
        self.palm_pose_targets.copy_(palm_pose)
        self.hand_pca_targets.zero_()

        self.fabric.set_features(
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
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.fabric_integrator.step(
                self.fabric_q.detach(),
                self.fabric_qd.detach(),
                self.fabric_qdd.detach(),
                self.timestep,
            )

        # ---- Grasp phase absolute synergy target ----
        late_grasp_full_grip_mask = (
            compute_late_grasp_full_grip_mask(
                num_contacts=self.num_contacts_buf,
                is_grasp_phase=self.is_grasp_phase,
                episode_length_buf=self.episode_length_buf,
                grasp_phase_steps=GRASP_PHASE_STEPS,
                contact_threshold=self.cfg.grasp_phase_full_grip_contact_threshold,
                progress_threshold=self.cfg.grasp_phase_full_grip_progress_threshold,
            )
            if self.cfg.enable_grasp_phase_full_grip_blend
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        close_hand_target = compute_grasp_phase_finger_targets(
            finger_action=finger_action,
            approach_pose=self.hand_approach_pose,
            grasp_pose=self.hand_grasp_pose,
            full_grip_pose=self.hand_full_grip_pose,
            lower_limits=self.hand_joint_lower_limits,
            upper_limits=self.hand_joint_upper_limits,
            late_grasp_mask=late_grasp_full_grip_mask,
        )
        approach_hand_target = compute_lift_finger_targets(
            finger_action=finger_action,
            grasp_pose=torch.zeros_like(self.hand_approach_pose),
            full_grip_pose=self.hand_approach_pose,
            lower_limits=self.hand_joint_lower_limits,
            upper_limits=self.hand_joint_upper_limits,
        )
        hand_target = torch.where(
            approach_mask.unsqueeze(1),
            approach_hand_target,
            close_hand_target,
        )
        self.hand_joint_targets.copy_(hand_target)

        # fabric_q hand 부분 동기화
        self.fabric_q[:, NUM_ARM_DOF:] = hand_target
        self.fabric_qd[:, NUM_ARM_DOF:].zero_()

    def _apply_action(self) -> None:
        is_post_grasp = self.is_post_grasp_phase

        # ---- 오른팔 ----
        self.robot.set_joint_position_target(self.fabric_q[:, :NUM_ARM_DOF], joint_ids=self.arm_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(self.fabric_q[:, :NUM_ARM_DOF]), joint_ids=self.arm_dof_indices
        )

        # ---- 오른손 ----
        # Lift/Stabilize/Transport에서는 grasp -> full-grip absolute synergy target 사용
        lift_finger_target = compute_lift_finger_targets(
            finger_action=self.actions[:, 6:NUM_ACTIONS],
            grasp_pose=self.hand_grasp_pose,
            full_grip_pose=self.hand_full_grip_pose,
            lower_limits=self.hand_joint_lower_limits,
            upper_limits=self.hand_joint_upper_limits,
        )
        finger_target = torch.where(
            is_post_grasp.unsqueeze(1),
            lift_finger_target,
            self.hand_joint_targets,
        )
        self.robot.set_joint_position_target(finger_target, joint_ids=self.hand_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(finger_target), joint_ids=self.hand_dof_indices
        )

        # ---- 왼팔: 고정 자세 ----
        self.robot.set_joint_position_target(
            self.left_arm_zero_pos, joint_ids=self.left_arm_dof_indices
        )

    def _apply_upright_palm_orientation_correction(
        self,
        palm_pose: torch.Tensor,
        phase_mask: torch.Tensor,
        phase_progress: torch.Tensor,
    ) -> torch.Tensor:
        if not phase_mask.any():
            return palm_pose

        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cup_z_world = quat_apply(self.object_rot, z_local)
        max_correction = math.radians(float(self.cfg.stabilize_upright_orientation_max_deg))
        gain = float(self.cfg.stabilize_upright_orientation_gain)
        pitch_roll_correction = torch.stack(
            [
                -cup_z_world[:, 0],
                cup_z_world[:, 1],
            ],
            dim=1,
        )
        pitch_roll_correction = (
            gain * pitch_roll_correction
        ).clamp(min=-max_correction, max=max_correction)
        pitch_roll_correction = pitch_roll_correction * phase_progress

        corrected = palm_pose.clone()
        corrected[:, 4] = corrected[:, 4] + pitch_roll_correction[:, 0]
        corrected[:, 5] = corrected[:, 5] + pitch_roll_correction[:, 1]
        corrected = torch.max(torch.min(corrected, self.palm_maxs), self.palm_mins)
        return torch.where(phase_mask.unsqueeze(1), corrected, palm_pose)

    # ------------------------------------------------------------------
    # Intermediate values
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self) -> None:
        self.object_pos = self.cup.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.cup.data.root_quat_w

        env_origins = self.scene.env_origins

        if self.palm_body_index >= 0:
            self.palm_center_pos = (
                self.robot.data.body_pos_w[:, self.palm_body_index, :] - env_origins
            )

        self.fingertip_pos = (
            self.robot.data.body_pos_w[:, self.fingertip_body_indices, :] - env_origins.unsqueeze(1)
        )

        if len(self.distal4_body_indices) == NUM_FINGERTIPS:
            self.distal4_pos = (
                self.robot.data.body_pos_w[:, self.distal4_body_indices, :] - env_origins.unsqueeze(1)
            )

        if len(self.middle3_body_indices) == NUM_FINGERTIPS:
            self.middle3_pos = (
                self.robot.data.body_pos_w[:, self.middle3_body_indices, :] - env_origins.unsqueeze(1)
            )

        self._update_contact_forces()

    # ------------------------------------------------------------------
    # Observations: Actor 96D (with oracle mass 97) | Critic 114D
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # ==== 공통 clean state (critic용) ====
        arm_joint_pos_clean    = self.robot.data.joint_pos[:, self.arm_dof_indices]
        arm_joint_vel_clean    = self.robot.data.joint_vel[:, self.arm_dof_indices]
        finger_joint_pos_clean = self.robot.data.joint_pos[:, self.hand_dof_indices]
        finger_joint_vel_clean = self.robot.data.joint_vel[:, self.hand_dof_indices]
        palm_center_pos_clean  = self.palm_center_pos
        fingertip_pos_clean    = self.fingertip_pos
        cup_pos_clean          = self.object_pos

        # ==== Actor obs용 noisy state ====
        σ_qp = self.cfg.obs_noise_joint_pos
        σ_qv = self.cfg.obs_noise_joint_vel
        σ_bp = self.cfg.obs_noise_body_pos
        σ_cp = (
            self.grasp_adr.get_param("noise", "obs_noise_cup_pos")
            if self.grasp_adr is not None
            else self.cfg.obs_noise_cup_pos
        )

        arm_joint_pos    = arm_joint_pos_clean    + torch.randn_like(arm_joint_pos_clean)    * σ_qp
        arm_joint_vel    = arm_joint_vel_clean    + torch.randn_like(arm_joint_vel_clean)    * σ_qv
        finger_joint_pos = finger_joint_pos_clean + torch.randn_like(finger_joint_pos_clean) * σ_qp
        finger_joint_vel = finger_joint_vel_clean + torch.randn_like(finger_joint_vel_clean) * σ_qv
        palm_center_pos  = palm_center_pos_clean  + torch.randn_like(palm_center_pos_clean)  * σ_bp
        fingertip_pos    = fingertip_pos_clean    + torch.randn_like(fingertip_pos_clean)    * σ_bp
        cup_pos_noisy    = cup_pos_clean          + torch.randn_like(cup_pos_clean)          * σ_cp

        fingertip_pos_rel_palm = (
            fingertip_pos - palm_center_pos.unsqueeze(1)
        ).view(self.num_envs, -1)

        palm_to_cup = cup_pos_noisy - palm_center_pos

        # middle phalanx → cup 벡터 (FK 기반, sim2real 가능)
        middle3_pos_noisy = self.middle3_pos + torch.randn_like(self.middle3_pos) * σ_bp
        middle_to_cup = (
            middle3_pos_noisy - cup_pos_noisy.unsqueeze(1)
        ).view(self.num_envs, -1)   # (N, 15)

        last_actions = self.actions  # (N, 12)

        # tip force: 3D 법선 방향 벡터 (5 × 3D = 15D)
        tip_force_xyz_norm = (
            self.contact_force_xyz_raw / CONTACT_FORCE_MAX
        ).clamp(-1.0, 1.0).view(self.num_envs, -1)  # (N, 15)

        phase_step_ratio = (
            self.episode_length_buf.float() / EPISODE_STEPS
        ).unsqueeze(1)

        cup_lin_vel  = self.cup.data.root_lin_vel_w
        cup_ang_vel  = self.cup.data.root_ang_vel_w
        cup_rot      = self.object_rot
        palm_binary_obs = self.palm_binary_contact_buf.float().unsqueeze(-1)
        palm_force_obs = (
            self.palm_contact_force_raw / CONTACT_FORCE_MAX
        ).clamp(0.0, 1.0).unsqueeze(-1)

        actor_obs_parts = [
            arm_joint_pos,          # 7
            arm_joint_vel,          # 7
            finger_joint_pos,       # 6
            finger_joint_vel,       # 6
            palm_center_pos,        # 3
            fingertip_pos_rel_palm, # 15
            palm_to_cup,            # 3
            cup_rot,                # 4
            last_actions,           # 12
        ]
        if self.cfg.actor_observe_bead_mass:
            actor_obs_parts.append(self._bead_mass_normalized.unsqueeze(-1))  # 1 (oracle mass -> 100D)
        actor_obs_parts.extend([
            tip_force_xyz_norm,     # 15 (실 fingertip 힘센서)
            middle_to_cup,          # 15
            phase_step_ratio,       # 1
            palm_binary_obs,        # 1 (실 palm 힘센서)
            palm_force_obs,         # 1
        ])
        actor_obs = torch.cat(actor_obs_parts, dim=-1)

        actor_obs = torch.nan_to_num(actor_obs, nan=0.0, posinf=5.0, neginf=-5.0)

        if actor_obs.shape[1] != self.cfg.num_observations:
            raise RuntimeError(
                f"[v11] Actor obs dim mismatch: {actor_obs.shape[1]} != {self.cfg.num_observations}"
            )

        # ==== Critic extra obs (18D, sim-only privileged) ====
        cup_height_delta = (
            cup_pos_clean[:, 2] - self.object_init_pos[:, 2]
        ).unsqueeze(1)

        # tip_contact_binary: fingertip 접촉 flag. 정밀 접촉 신호(privileged).
        tip_contact_binary = self.binary_contact_buf.float()   # (N, 5)

        tip_to_cup_dist = (
            fingertip_pos_clean - cup_pos_clean.unsqueeze(1)
        ).norm(dim=-1)
        fingertip_signed_dist = (tip_to_cup_dist - CUP_RADIUS_APPROX).unsqueeze(-1).squeeze(-1)  # (N, 5)

        middle_to_cup_clean = (
            self.middle3_pos - cup_pos_clean.unsqueeze(1)
        ).view(self.num_envs, -1)   # (N, 15)

        # actor 와 동일 구성(noise 없는 clean) — 96D
        actor_obs_clean = torch.cat([
            arm_joint_pos_clean,
            arm_joint_vel_clean,
            finger_joint_pos_clean,   # 6
            finger_joint_vel_clean,   # 6
            palm_center_pos_clean,
            (fingertip_pos_clean - palm_center_pos_clean.unsqueeze(1)).view(self.num_envs, -1),
            cup_pos_clean - palm_center_pos_clean,
            cup_rot,
            last_actions,             # 12
            tip_force_xyz_norm,       # 15 (실 fingertip 힘센서)
            middle_to_cup_clean,      # 15
            phase_step_ratio,
            palm_binary_obs,          # 1
            palm_force_obs,           # 1
        ], dim=-1)   # 96D

        critic_obs = torch.cat([
            actor_obs_clean,                            # 96
            self._bead_mass_normalized.unsqueeze(-1),   # 1  (oracle mass)
            cup_lin_vel,                                # 3
            cup_ang_vel,                                # 3  (critic-only privileged)
            cup_height_delta,                           # 1
            tip_contact_binary,                         # 5
            fingertip_signed_dist,                      # 5
        ], dim=-1)   # 114D

        critic_obs = torch.nan_to_num(critic_obs, nan=0.0, posinf=5.0, neginf=-5.0)

        if critic_obs.shape[1] != NUM_CRITIC_OBSERVATIONS:
            raise RuntimeError(
                f"[v11] Critic obs dim mismatch: {critic_obs.shape[1]} != {NUM_CRITIC_OBSERVATIONS}"
            )

        return {"policy": actor_obs, "critic": critic_obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        self.extras.clear()

        # ---- 접촉력 / 질량 ----
        total_grip_force = self.contact_force_raw.sum(dim=-1)
        effective_mass = (
            self.cfg.cup_base_mass
            + self._bead_mass_normalized * self.cfg.num_beads * self.cfg.bead_single_mass
        )
        mg = effective_mass * 9.81
        force_ratio = total_grip_force / (mg + 1e-4)

        # ---- 컵 자세 ----
        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cup_z_world = quat_apply(self.object_rot, z_local)
        cup_tilt_deg = torch.rad2deg(
            torch.acos(cup_z_world[:, 2].clamp(min=-1.0, max=1.0))
        )

        # ---- slip proxy ----
        cup_horiz_vel = torch.nan_to_num(
            self.cup.data.root_lin_vel_w[:, :2].norm(dim=-1), nan=0.0
        )
        contact_delta_abs = (self.num_contacts_buf.float() - self._prev_num_contacts_buf).abs()
        middle_contact_delta_abs = (
            self.middle_binary_contact_buf.float().sum(dim=-1) - self._prev_middle_contacts_buf
        ).abs()
        cup_tilt_delta_abs = (cup_tilt_deg - self._prev_cup_tilt_deg_buf).abs()
        slip_proxy = compute_slip_proxy(
            cup_xy_velocity=cup_horiz_vel,
            cup_tilt_delta_deg=cup_tilt_delta_abs,
            contact_delta_abs=contact_delta_abs,
            middle_contact_delta_abs=middle_contact_delta_abs,
            xy_velocity_scale=self.cfg.stabilize_cup_lin_vel_threshold,
            tilt_delta_scale=self.cfg.slip_proxy_tilt_delta_scale,
            contact_delta_scale=self.cfg.stabilize_contact_delta_threshold,
            middle_contact_delta_scale=self.cfg.stabilize_contact_delta_threshold,
            contact_delta_weight=self.cfg.slip_proxy_contact_delta_weight,
            middle_contact_delta_weight=self.cfg.slip_proxy_middle_contact_delta_weight,
            tilt_delta_weight=self.cfg.slip_proxy_tilt_delta_weight,
        )

        approach_mask = self.is_grasp_phase & (~self._grasp_started_buf)
        close_grasp_mask = self.is_grasp_phase & self._grasp_started_buf

        # ---- Phase A: contact_adr 동적 min_contacts (3→4→5 lift 진입 허들) ----
        _adr_min_contacts = (
            int(round(self.contact_adr.get_param("contact", "min_contacts")))
            if self.contact_adr is not None
            else int(self.cfg.stage0_lift_start_min_contacts)
        )

        # ---- lift contact hold 추적 ----
        lift_contact_phase = close_grasp_mask
        self._lift_contact_hold_count, lift_contact_ready_now, self._lift_contact_ready_latched_buf = compute_lift_readiness(
            num_contacts=self.num_contacts_buf,
            is_grasp_phase=lift_contact_phase,
            previous_hold_count=self._lift_contact_hold_count,
            previous_latched=self._lift_contact_ready_latched_buf,
            min_contacts=_adr_min_contacts,
            hold_steps=self.cfg.stage0_lift_start_hold_steps,
        )
        lift_contact_ready_gate = self._lift_contact_ready_latched_buf.float()
        lift_started_now = self._lift_started_buf & (
            self.episode_length_buf == self._lift_start_step_buf
        )
        if lift_started_now.any():
            self._contacts_at_lift_start_buf[lift_started_now] = (
                self.num_contacts_buf[lift_started_now].float()
            )
            self._palm_at_lift_start_buf[lift_started_now] = (
                self.palm_binary_contact_buf[lift_started_now].float()
            )
            self._grasp_tilt_at_lift_start_buf[lift_started_now] = cup_tilt_deg[lift_started_now]
            self._force_ratio_at_lift_start_buf[lift_started_now] = force_ratio[lift_started_now]

        # ---- contact persistence 추적 ----
        has_5_contact_bool = self.num_contacts_buf >= NUM_FINGERTIPS
        late_grasp_full_grip_mask = (
            compute_late_grasp_full_grip_mask(
                num_contacts=self.num_contacts_buf,
                is_grasp_phase=self.is_grasp_phase,
                episode_length_buf=self.episode_length_buf,
                grasp_phase_steps=GRASP_PHASE_STEPS,
                contact_threshold=self.cfg.grasp_phase_full_grip_contact_threshold,
                progress_threshold=self.cfg.grasp_phase_full_grip_progress_threshold,
            )
            if self.cfg.enable_grasp_phase_full_grip_blend
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        middle_envelope_gate = compute_middle_contact_gate(
            self.middle_binary_contact_buf,
            self.cfg.min_middle_contacts_for_success,
        ).float()
        full_tip_middle_contact = has_5_contact_bool & middle_envelope_gate.bool()
        self._contact_persistence_buf = torch.where(
            full_tip_middle_contact,
            self._contact_persistence_buf + 1,
            torch.zeros_like(self._contact_persistence_buf),
        )

        # ---- full grip ready 추적 ----
        upright_success_for_grip = compute_upright_success_mask(
            cup_z_world[:, 2], self.cfg.success_upright_max_deg,
        )
        lifted_for_full_grip = (
            self.object_pos[:, 2] > (self.object_init_pos[:, 2] + self.cfg.lift_success_height)
        )
        full_grip_ready_now = (
            self._lift_started_buf
            & has_5_contact_bool
            & lifted_for_full_grip
            & upright_success_for_grip
        )
        self._full_grip_ready_buf.copy_(full_grip_ready_now)
        self._full_grip_ready_latched_buf |= full_grip_ready_now
        self._grip_ready_latched_buf.copy_(self._full_grip_ready_latched_buf)
        grip_ready_now = full_grip_ready_now
        self._grip_ready_hold_count = torch.where(
            grip_ready_now,
            self._grip_ready_hold_count + 1,
            torch.zeros_like(self._grip_ready_hold_count),
        )
        full_grip_ready_gate = self._full_grip_ready_buf.float()
        grip_ready_gate = self._full_grip_ready_latched_buf.float()

        # ---- ADR (min_contacts는 위 lift 게이트에서 이미 계산됨) ----
        self._maybe_update_phase_curriculum()
        _ep_success_rate, _ep_success_window_len = self._curriculum_success_rate()
        if self.contact_adr is not None:
            # Phase A: contact ADR을 lift_started_rate로 트리거.
            # success_rate 기반은 lift 진입이 안 되면 영원히 0이라 허들 상승이 막힌다.
            self.contact_adr.maybe_increment(self._lift_started_buf.float().mean())
        if self.grasp_adr is not None:
            self.grasp_adr.maybe_increment(_ep_success_rate)

        cup_height_delta = (self.object_pos[:, 2] - self.object_init_pos[:, 2]).clamp(min=0.0)
        tip_contact_progress = (
            self.num_contacts_buf.float() / float(NUM_FINGERTIPS)
        ).clamp(max=1.0)
        contact_persistence_progress = (
            self._lift_contact_hold_count.float()
            / max(float(self.cfg.grasp_contact_persistence_reward_steps), 1.0)
        ).clamp(max=1.0)
        five_tip_contact = has_5_contact_bool.float()
        stabilize_upright_quality = torch.exp(
            -cup_tilt_deg / max(float(self.cfg.stabilize_upright_reward_scale_deg), 1e-6)
        )
        spawn_xy_dist = (self.object_pos[:, :2] - self.object_init_pos[:, :2]).norm(dim=-1)

        # ---- prev buffer 갱신 ----
        self._prev_total_grip_force_buf.copy_(total_grip_force)
        self._prev_num_contacts_buf.copy_(self.num_contacts_buf.float())
        self._prev_middle_contacts_buf.copy_(self.middle_binary_contact_buf.float().sum(dim=-1))
        self._prev_cup_tilt_deg_buf.copy_(cup_tilt_deg)

        # ---- Curated TensorBoard diagnostics ----
        zero = torch.zeros((), device=self.device)

        def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return values[mask].mean() if mask.any() else zero

        self.extras["cup/height_delta"] = cup_height_delta.mean()
        self.extras["cup/tilt_deg"] = cup_tilt_deg.mean()
        self.extras["cup/tilt_grasp_deg"] = _masked_mean(cup_tilt_deg, self.is_grasp_phase)
        self.extras["cup/tilt_lift_deg"] = _masked_mean(cup_tilt_deg, self.is_lift_phase)
        self.extras["cup/tilt_stabilize_deg"] = _masked_mean(
            cup_tilt_deg, self.is_stabilize_phase
        )

        self.extras["task/force_ratio"] = force_ratio.mean()
        self.extras["task/grip_ready_rate"] = grip_ready_gate.mean()
        self.extras["task/lift_contact_ready_rate"] = lift_contact_ready_gate.mean()
        self.extras["task/lift_started_rate"] = self._lift_started_buf.float().mean()
        self.extras["task/full_grip_ready_rate"] = full_grip_ready_gate.mean()
        self.extras["task/five_tip_contact_rate"] = five_tip_contact.mean()
        self.extras["task/prelift_five_tip_contact_rate"] = _masked_mean(
            five_tip_contact, self.is_grasp_phase
        )
        self.extras["task/lift_five_tip_contact_rate"] = _masked_mean(
            five_tip_contact, self.is_lift_phase
        )

        # ---- Phase C: per-finger contact + joint-state diagnostics ----
        # RH56F1 has no distal/middle phalanx sensors (those buffers stay zero),
        # so only fingertip + palm scalars are exposed.
        self.extras.update(
            per_finger_contact_scalars(
                tip_force=self.contact_force_raw,
                tip_binary=self.binary_contact_buf,
                palm_force=self.palm_contact_force_raw,
            )
        )
        self.extras.update(
            joint_state_scalars(
                arm_pos=self.robot.data.joint_pos[:, self.arm_dof_indices],
                arm_vel=self.robot.data.joint_vel[:, self.arm_dof_indices],
                finger_pos=self.robot.data.joint_pos[:, self.hand_dof_indices],
                finger_vel=self.robot.data.joint_vel[:, self.hand_dof_indices],
            )
        )
        self.extras["task/pre_lift_full_contact_rate"] = (
            full_tip_middle_contact & self.is_grasp_phase
        ).float().mean()
        self.extras["task/slip_proxy"] = slip_proxy.mean()
        self.extras["task/lift_contact_hold"] = self._lift_contact_hold_count.float().mean()
        self.extras["task/contact_persistence"] = self._contact_persistence_buf.float().mean()
        self.extras["task/grip_ready_hold"] = self._grip_ready_hold_count.float().mean()
        self.extras["task/late_grasp_full_grip_mode_rate"] = (
            late_grasp_full_grip_mask.float().mean()
        )
        transition_mask = (
            late_grasp_full_grip_mask
            & self.is_grasp_phase
            & (self.num_contacts_buf >= self.cfg.grasp_phase_full_grip_contact_threshold)
        )
        self.extras["task/contact_to_full_grip_transition_rate"] = (
            transition_mask.float().mean()
        )
        meaningful_contact = self.num_contacts_buf > 0
        lifted_bool = cup_height_delta >= self.cfg.lift_success_height
        # NOTE: phase/{approach,grasp,lift,stabilize} are emitted canonically via
        # log_grasp_v2_common_scalars() below (it clear()s then re-sets the common
        # tag set), so assigning them here is dead — the helper overwrites them.
        # Removed to keep the phase/* tags single-sourced and identical to Teosllo.
        self.extras["task/phase_grasp"] = self.is_grasp_phase.float().mean()
        self.extras["task/phase_approach"] = approach_mask.float().mean()
        self.extras["task/phase_close_grasp"] = close_grasp_mask.float().mean()
        self.extras["task/approach_ready_rate"] = self._approach_ready_buf.float().mean()
        self.extras["task/approach_timeout_rate"] = self._approach_timeout_buf.float().mean()
        self.extras["task/grasp_from_timeout_rate"] = self._grasp_from_timeout_buf.float().mean()
        cup_quat_inv_log = self.object_rot.clone()
        cup_quat_inv_log[:, 1:] = -cup_quat_inv_log[:, 1:]
        palm_local_log = quat_apply(cup_quat_inv_log, self.palm_center_pos - self.object_pos)
        palm_local_radial_log = palm_local_log[:, :2].norm(dim=-1)
        self.extras["task/approach_palm_local_radial"] = _masked_mean(
            palm_local_radial_log, approach_mask
        )
        self.extras["task/approach_palm_local_z"] = _masked_mean(
            palm_local_log[:, 2], approach_mask
        )
        self.extras["task/phase_lift"] = self.is_lift_phase.float().mean()
        self.extras["task/phase_stabilize"] = self.is_stabilize_phase.float().mean()
        self.extras["task/curriculum_stage"] = torch.tensor(
            float(1 if not self.cfg.enable_phase_curriculum else self._phase_curriculum_stage),
            device=self.device,
        )
        self.extras["task/curriculum_window_len"] = torch.tensor(
            float(_ep_success_window_len), device=self.device
        )
        self.extras["task/lift_success_rate"] = torch.tensor(
            self._window_success_rate(self._lift_success_window), device=self.device
        )
        self.extras["task/stabilize_success_rate"] = torch.tensor(
            self._window_success_rate(self._stabilize_success_window), device=self.device
        )
        self.extras["task/curriculum_success_rate"] = torch.tensor(
            _ep_success_rate, device=self.device
        )
        self.extras["task/success_rate"] = torch.tensor(
            self._window_success_rate(self._success_window), device=self.device
        )
        cup_center = self.object_pos
        palm_to_cup_dist = (self.palm_center_pos - cup_center).norm(dim=-1)
        middle_to_cup_dist = (
            self.middle3_pos - cup_center.unsqueeze(1)
        ).norm(dim=-1).mean(dim=-1)
        tip_to_cup_dist_top3 = (
            self.fingertip_pos - cup_center.unsqueeze(1)
        ).norm(dim=-1).topk(k=min(3, NUM_FINGERTIPS), dim=-1, largest=False).values.mean(dim=-1)
        self.extras["task/palm_to_cup_dist"] = palm_to_cup_dist.mean()
        self.extras["task/middle_to_cup_dist"] = middle_to_cup_dist.mean()
        self.extras["task/tip_to_cup_dist_top3"] = tip_to_cup_dist_top3.mean()
        lift_start_mask = self._eval_lift_snapshot_valid
        if lift_start_mask.any():
            self.extras["task/contacts_at_lift_start"] = self._contacts_at_lift_start_buf[
                lift_start_mask
            ].mean()
            self.extras["task/force_ratio_at_lift_start"] = self._force_ratio_at_lift_start_buf[
                lift_start_mask
            ].mean()
        else:
            self.extras["task/contacts_at_lift_start"] = zero
            self.extras["task/force_ratio_at_lift_start"] = zero
        if self.contact_adr is not None:
            self.extras["task/adr_min_contacts"] = torch.tensor(
                float(_adr_min_contacts), device=self.device
            )
        if self.grasp_adr is not None:
            self.extras["task/adr_difficulty_progress"] = torch.tensor(
                self.grasp_adr.progress, device=self.device
            )

        cup_to_palm_xy = self.palm_center_pos[:, :2] - cup_center[:, :2]
        approach_dir_xy = cup_to_palm_xy / cup_to_palm_xy.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-6)
        perp_dir_xy = torch.stack(
            [-approach_dir_xy[:, 1], approach_dir_xy[:, 0]], dim=1
        )
        enclosure_axis = torch.zeros(self.num_envs, 3, device=self.device)
        enclosure_axis[:, :2] = perp_dir_xy

        radius = float(self.cfg.cup_radius_approx)
        thumb_target = cup_center + enclosure_axis * radius
        others_target = cup_center - enclosure_axis * radius

        thumb_dist = (self.fingertip_pos[:, 0, :] - thumb_target).norm(dim=-1)
        others_dist_per_finger = (
            self.fingertip_pos[:, 1:, :] - others_target.unsqueeze(1)
        ).norm(dim=-1)
        others_dist = others_dist_per_finger.mean(dim=-1)
        thumb_weight = float(self.cfg.enclosure_thumb_weight)
        fingertip_side_dist = thumb_weight * thumb_dist + (1.0 - thumb_weight) * others_dist

        action_delta_norm = compute_action_delta_norm(self.actions, self.prev_actions)
        stability = compute_grasp_v2_stability(
            cup_lin_vel=self.cup.data.root_lin_vel_w,
            cup_ang_vel=self.cup.data.root_ang_vel_w,
            contact_delta=contact_delta_abs,
            action_delta_norm=action_delta_norm,
            cfg=self.cfg,
        )
        total, reward_terms, reward_gates = compute_grasp_reward_terms(
            num_tip_contacts=self.num_contacts_buf,
            tip_contact_frac=tip_contact_progress,
            full_tip_contact=five_tip_contact,
            contact_persistence_frac=contact_persistence_progress,
            palm_to_cup_dist=palm_to_cup_dist,
            fingertip_side_dist=fingertip_side_dist,
            cup_height_delta=cup_height_delta,
            cup_xy_displacement=spawn_xy_dist,
            cup_tilt_deg=cup_tilt_deg,
            upright_quality=stabilize_upright_quality,
            lift_latched=self._lift_started_buf,
            action_delta_norm=action_delta_norm,
            stabilize_reward_gate=self.is_stabilize_phase,
            success_now=self.success_flag,
            stable=stability.stable,
            stability_quality=stability.quality,
            cfg=self.cfg,
        )

        lift_tilt_for_common_log = cup_tilt_deg[self._lift_started_buf]
        success_held = self._success_hold_count >= int(self.cfg.success_hold_steps)
        log_grasp_v2_common_scalars(
            self.extras,
            {
                "phase/approach": ((~self._lift_started_buf) & (~meaningful_contact)).float().mean(),
                "phase/grasp": ((~self._lift_started_buf) & meaningful_contact).float().mean(),
                "phase/lift": self.is_lift_phase.float().mean(),
                "phase/stabilize": self.is_stabilize_phase.float().mean(),
                "reward/approach": reward_terms["approach"].mean(),
                "reward/grasp": reward_terms["grasp"].mean(),
                "reward/lift": reward_terms["lift"].mean(),
                "reward/stabilize": reward_terms["stabilize"].mean(),
                "reward/success_bonus": reward_terms["success_bonus"].mean(),
                "reward/post_lift_contact_loss": reward_terms["post_lift_contact_loss"].mean(),
                "reward/action_smooth": reward_terms["action_smooth"].mean(),
                "reward/stability": reward_terms["stability"].mean(),
                "reward/total": total.mean(),
                "contact/count": self.num_contacts_buf.float().mean(),
                "contact/full_contact_rate": five_tip_contact.mean(),
                "contact/palm": self.palm_binary_contact_buf.float().mean(),
                "contact/palm_force": self.palm_contact_force_raw.mean(),
                "contact/grasp_ready_hold": self._lift_contact_hold_count.float().mean(),
                "contact/contacts_at_lift_start": self._contacts_at_lift_start_buf.mean(),
                "contact/palm_at_lift_start": self._palm_at_lift_start_buf.mean(),
                "cup/height_delta": cup_height_delta.mean(),
                "cup/tilt_deg": cup_tilt_deg.mean(),
                "cup/upright_quality": stabilize_upright_quality.mean(),
                "cup/grasp_tilt_deg": self._grasp_tilt_at_lift_start_buf.mean(),
                "cup/lift_tilt_deg": (
                    lift_tilt_for_common_log.mean()
                    if lift_tilt_for_common_log.numel() > 0
                    else cup_tilt_deg.mean()
                ),
                "cup/xy_displacement": spawn_xy_dist.mean(),
                "task/lifted_rate": lifted_bool.float().mean(),
                "task/upright_rate": reward_gates["final_upright_success"].mean(),
                "task/stable_rate": stability.stable.float().mean(),
                "task/cup_lin_vel": stability.cup_lin_vel_norm.mean(),
                "task/cup_ang_vel": stability.cup_ang_vel_norm.mean(),
                "task/action_delta_norm": stability.action_delta_norm.mean(),
                "task/contact_delta": stability.contact_delta.mean(),
                "task/grasp_ready_rate": (
                    self.num_contacts_buf >= self.cfg.stage0_lift_start_min_contacts
                ).float().mean(),
                "task/five_tip_contact_rate": five_tip_contact.mean(),
                "task/prelift_five_tip_contact_rate": _masked_mean(
                    five_tip_contact, self.is_grasp_phase
                ),
                "task/lift_five_tip_contact_rate": _masked_mean(
                    five_tip_contact, self.is_lift_phase
                ),
                "task/lift_started_rate": self._lift_started_buf.float().mean(),
                "task/lift_success_now": self._lift_success_latched_buf.float().mean(),
                "task/stabilize_success_now": self._stabilize_success_latched_buf.float().mean(),
                "task/lift_success_rate": torch.tensor(
                    self._window_success_rate(self._lift_success_window),
                    device=self.device,
                ),
                "task/stabilize_success_rate": torch.tensor(
                    self._window_success_rate(self._stabilize_success_window),
                    device=self.device,
                ),
                "task/success_rate": torch.tensor(_ep_success_rate, device=self.device),
                "task/common_success_now": reward_gates["success_now"].mean(),
                "task/success_held_rate": success_held.float().mean(),
                "task/success_hold_count": self._success_hold_count.float().mean(),
            },
        )
        self.extras["debug/rh56f1/task/force_ratio"] = force_ratio.mean()
        self.extras["debug/rh56f1/task/slip_proxy"] = slip_proxy.mean()
        self.extras["debug/rh56f1/task/prelift_force_ratio"] = (
            force_ratio * self.is_grasp_phase.float()
        ).sum() / self.is_grasp_phase.float().sum().clamp(min=1.0)
        return total

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        out_x = (
            (self.object_pos[:, 0] < self.cfg.obj_out_x_min) |
            (self.object_pos[:, 0] > self.cfg.obj_out_x_max)
        )
        out_y = (
            (self.object_pos[:, 1] < self.cfg.obj_out_y_min) |
            (self.object_pos[:, 1] > self.cfg.obj_out_y_max)
        )
        fallen = self.object_pos[:, 2] < self.cfg.obj_fallen_z

        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cup_z_world = quat_apply(self.object_rot, z_local)
        tipped = cup_z_world[:, 2] < self._cup_tipping_cos

        lift_elapsed_steps = torch.where(
            self._lift_started_buf,
            (self.episode_length_buf - self._lift_start_step_buf).clamp(min=0),
            torch.zeros_like(self.episode_length_buf),
        )
        stabilize_elapsed_steps = torch.where(
            self._stabilize_started_buf,
            (self.episode_length_buf - self._stabilize_start_step_buf).clamp(min=0),
            torch.zeros_like(self.episode_length_buf),
        )

        in_or_past_lift = self._lift_started_buf
        in_stabilize = self.is_stabilize_phase
        if self.cfg.enable_phase_curriculum:
            stage0_lift_only = self._episode_curriculum_stage_buf <= 0
            stage1_stabilize_only = self._episode_curriculum_stage_buf == 1
        else:
            stage0_lift_only = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            stage1_stabilize_only = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        lifted  = self.object_pos[:, 2] > (self.object_init_pos[:, 2] + self.cfg.lift_success_height)
        _lift_min_contacts = NUM_FINGERTIPS
        lift_grasped = self.num_contacts_buf >= _lift_min_contacts
        contact_grasped = self.num_contacts_buf >= NUM_FINGERTIPS
        upright_success = compute_upright_success_mask(
            cup_z_world[:, 2],
            self.cfg.success_upright_max_deg,
        )
        full_grip_ready_now = (
            in_or_past_lift
            & contact_grasped
            & upright_success
            & lifted
        )
        self._full_grip_ready_buf.copy_(full_grip_ready_now)
        self._full_grip_ready_latched_buf |= full_grip_ready_now
        lift_success_candidate = in_or_past_lift & lifted & lift_grasped & upright_success
        self._lift_success_hold_count = torch.where(
            lift_success_candidate,
            self._lift_success_hold_count + 1,
            torch.zeros_like(self._lift_success_hold_count),
        )
        lift_success_now = self._lift_success_hold_count >= int(self.cfg.full_grip_hold_steps)
        cup_tilt_deg = torch.rad2deg(
            torch.acos(cup_z_world[:, 2].clamp(min=-1.0, max=1.0))
        )
        action_delta_norm = compute_action_delta_norm(self.actions, self.prev_actions)
        contact_delta_abs = (self.num_contacts_buf.float() - self._prev_num_contacts_buf).abs()
        stability = compute_grasp_v2_stability(
            cup_lin_vel=self.cup.data.root_lin_vel_w,
            cup_ang_vel=self.cup.data.root_ang_vel_w,
            contact_delta=contact_delta_abs,
            action_delta_norm=action_delta_norm,
            cfg=self.cfg,
        )
        stationary_success = compute_stationary_grasp_success(
            stabilize_started=in_stabilize,
            cup_height_delta=self.object_pos[:, 2] - self.object_init_pos[:, 2],
            full_contact=contact_grasped,
            cup_tilt_deg=cup_tilt_deg,
            stable=stability.stable,
            previous_success_hold_count=self._success_hold_count,
            cfg=self.cfg,
        )
        success_now = stationary_success.success_now
        stabilize_success_now = (
            in_stabilize
            & lifted
            & contact_grasped
            & upright_success
            & stability.stable
        )
        self._lift_success_latched_buf |= lift_success_now
        self._stabilize_success_latched_buf |= stabilize_success_now

        self.success_flag.copy_(success_now)
        self.episode_success_buf |= success_now
        self.episode_lift_success_buf |= lift_success_now
        self.episode_stabilize_success_buf |= stabilize_success_now

        self._success_hold_count = stationary_success.hold_count
        success_held = stationary_success.success_held

        if self.cfg.terminate_on_lift_failure:
            grasp_timeout_failed = (
                (self.episode_length_buf >= LIFT_START_STEP)
                & (~self._lift_contact_ready_latched_buf)
                & (~self._lift_started_buf)
            )
            lift_failed = (
                self._lift_started_buf
                & (lift_elapsed_steps >= LIFT_PHASE_STEPS)
                & (~self._lift_success_latched_buf)
            ) | grasp_timeout_failed
        else:
            grasp_timeout_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            lift_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        curriculum_lift_horizon = (
            stage0_lift_only
            & self._lift_started_buf
            & (lift_elapsed_steps >= LIFT_PHASE_STEPS)
        )
        curriculum_stabilize_horizon = (
            (STABILIZE_PHASE_STEPS > 0)
            & stage1_stabilize_only
            & self._stabilize_started_buf
            & (stabilize_elapsed_steps >= STABILIZE_PHASE_STEPS)
        )

        terminated = out_x | out_y | fallen | tipped | success_held | lift_failed
        truncated  = (
            (self.episode_length_buf >= self.max_episode_length - 1)
            | curriculum_lift_horizon
            | curriculum_stabilize_horizon
        )

        self.extras["object_stat/obj_z"] = self.object_pos[:, 2].mean()
        self.extras["cup/obj_z"] = self.extras["object_stat/obj_z"]
        self.extras["task/lift_success_now"] = lift_success_now.float().mean()
        self.extras["task/stabilize_success_now"] = stabilize_success_now.float().mean()
        self.extras["debug/rh56f1/task/curriculum_lift_only"] = stage0_lift_only.float().mean()
        self.extras["debug/rh56f1/task/curriculum_stabilize_only"] = (
            stage1_stabilize_only.float().mean()
        )
        self.extras["debug/rh56f1/task/grasp_timeout_fail_rate"] = (
            grasp_timeout_failed.float().mean()
        )

        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        had_started = self._eval_episode_started[env_ids].clone()

        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        self._apply_real2sim_actuator_randomization(env_ids)

        n = len(env_ids)
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        started_n = int(had_started.sum().item())

        # ---- episode 성공 집계 ----
        self._total_episodes += started_n
        if started_n > 0:
            self._successful_episodes += int(
                (self.episode_success_buf[env_ids] & had_started).sum().item()
            )

        # 6.2 & 6.3: moving window + per-bin 업데이트
        for i, env_id in enumerate(env_ids):
            if not bool(had_started[i].item()):
                continue
            stage_val = (
                int(self._episode_curriculum_stage_buf[env_id].item())
                if self.cfg.enable_phase_curriculum
                else 1
            )
            lift_success_val = int(bool(self.episode_lift_success_buf[env_id].item()))
            stabilize_success_val = int(bool(self.episode_stabilize_success_buf[env_id].item()))
            success_val = int(bool(self.episode_success_buf[env_id].item()))
            self._lift_success_window.append(lift_success_val)
            if stage_val >= 1:
                self._stabilize_success_window.append(stabilize_success_val)
            if stage_val >= 1:
                self._success_window.append(success_val)
            # 6.3: bead level (0~3) 판별 — _bead_mass_normalized는 아직 이전 에피소드 값
            lvl = int(round(self._bead_mass_normalized[env_id].item() * 3.0))
            lvl = min(max(lvl, 0), 3)
            self._total_episodes_bin[lvl] += 1
            self._successful_episodes_bin[lvl] += success_val

        # Eval 기록 저장
        for i, env_id in enumerate(env_ids):
            if not bool(had_started[i].item()):
                continue

            grasp_count = max(int(self._eval_grasp_action_count[env_id].item()), 1)
            grasp_mean  = self._eval_grasp_action_sum[env_id] / grasp_count           # (20,)
            grasp_var   = (
                self._eval_grasp_action_sq_sum[env_id] / grasp_count - grasp_mean.square()
            ).clamp_min(0.0)
            grasp_std = torch.sqrt(grasp_var)

            if bool(self._eval_lift_snapshot_valid[env_id].item()):
                finger_action_at_lift = self._eval_finger_actions_at_lift[env_id]
                grip_at_lift = self._eval_grip_at_lift[env_id].item()
            else:
                finger_action_at_lift = self._last_grasp_finger_action[env_id]
                grip_at_lift = finger_action_at_lift.abs().mean().item()

            bead_count = int(self._bead_count_current[env_id].item())
            bead_initial_count = int(self._bead_count_initial[env_id].item())
            dynamic_bead_added = max(bead_count - bead_initial_count, 0)
            lift_count = max(int(self._eval_lift_action_count[env_id].item()), 1)
            lift_mean = self._eval_lift_action_sum[env_id] / lift_count
            lift_var = (
                self._eval_lift_action_sq_sum[env_id] / lift_count - lift_mean.square()
            ).clamp_min(0.0)
            lift_std = torch.sqrt(lift_var)

            # RH56F1 6D action layout: [thumb_1, thumb_2, index_1, middle_1, ring_1, little_1]
            # 손가락별 curl 관절 인덱스: thumb→thumb_2(1), index→2, middle→3, ring→4, little→5
            def _curl_idx(finger: int) -> int:  # finger 0-indexed (0=thumb..4=little)
                return finger + 1

            self._eval_records.append({
                "bead_count": bead_count,
                "bead_mass": self._bead_mass_normalized[env_id].item(),
                "bead_count_initial": bead_initial_count,
                "dynamic_bead_added": dynamic_bead_added,
                "cup_friction_static": self._cup_friction_static[env_id].item(),
                "curriculum_stage": int(self._episode_curriculum_stage_buf[env_id].item()),
                "lift_success": self.episode_lift_success_buf[env_id].item(),
                "stabilize_success": self.episode_stabilize_success_buf[env_id].item(),
                "grip":      grip_at_lift,
                "grasp_steps": grasp_count,
                "grasp_action_mean": grasp_mean.mean().item(),
                "grasp_action_std":  grasp_std.mean().item(),
                "grasp_action_min":  self._eval_grasp_action_min[env_id].mean().item(),
                "grasp_action_max":  self._eval_grasp_action_max[env_id].mean().item(),
                "lift_steps": lift_count,
                "lift_action_mean": lift_mean.mean().item(),
                "lift_action_std": lift_std.mean().item(),
                "lift_action_abs_mean": lift_mean.abs().mean().item(),
                "lift_force_delta_mean": (
                    self._eval_lift_force_delta_sum[env_id] / lift_count
                ).item(),
                "lift_contact_delta_mean": (
                    self._eval_lift_contact_delta_sum[env_id] / lift_count
                ).item(),
                # curl joint actions at lift (per finger _2 joint)
                "thumb_action":  finger_action_at_lift[_curl_idx(0)].item(),
                "index_action":  finger_action_at_lift[_curl_idx(1)].item(),
                "middle_action": finger_action_at_lift[_curl_idx(2)].item(),
                "ring_action":   finger_action_at_lift[_curl_idx(3)].item(),
                "pinky_action":  finger_action_at_lift[_curl_idx(4)].item(),
                "thumb_mean_action":  grasp_mean[_curl_idx(0)].item(),
                "index_mean_action":  grasp_mean[_curl_idx(1)].item(),
                "middle_mean_action": grasp_mean[_curl_idx(2)].item(),
                "ring_mean_action":   grasp_mean[_curl_idx(3)].item(),
                "pinky_mean_action":  grasp_mean[_curl_idx(4)].item(),
                "thumb_std_action":  grasp_std[_curl_idx(0)].item(),
                "index_std_action":  grasp_std[_curl_idx(1)].item(),
                "middle_std_action": grasp_std[_curl_idx(2)].item(),
                "ring_std_action":   grasp_std[_curl_idx(3)].item(),
                "pinky_std_action":  grasp_std[_curl_idx(4)].item(),
                "curriculum_success": self.episode_success_buf[env_id].item(),
                "success": self.episode_success_buf[env_id].item(),
            })

        self._maybe_export_warm_states(env_ids_tensor, had_started)

        self.episode_success_buf[env_ids] = False
        self.episode_lift_success_buf[env_ids] = False
        self.episode_stabilize_success_buf[env_ids] = False
        self._episode_curriculum_stage_buf[env_ids] = (
            2 if not self.cfg.enable_phase_curriculum else self._phase_curriculum_stage
        )

        # ---- 1. Reset source 선택 ----
        if self.demo_grasp_reset_bank is not None:
            demo_indices = torch.randint(
                self.demo_grasp_reset_bank.num_demos,
                (n,),
                device=self.device,
            )
            start_arm = self.demo_grasp_reset_bank.start_arm_joint_pos[demo_indices]
            start_hand = self.demo_grasp_reset_bank.start_hand_joint_pos[demo_indices]
            pregrasp_palm_pose = self.demo_grasp_reset_bank.start_palm_pose_euler_zyx[demo_indices].clone()
            pregrasp_palm_pose[:, 2] = self.cfg.object_spawn_z + self.cfg.pregrasp_offset_z
            demo_lift_target = self.demo_grasp_reset_bank.lift_palm_pose_euler_zyx[demo_indices].clone()
            demo_lift_target[:, 2] = torch.minimum(
                demo_lift_target[:, 2],
                pregrasp_palm_pose[:, 2] + float(self.cfg.lift_target_z_delta),
            )
            self.demo_lift_palm_target_buf[env_ids] = demo_lift_target

            q_pregrasp = torch.cat([start_arm, start_hand], dim=1)
            approach_hand = start_hand

            noise_xy = torch.stack([
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_x,
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_y,
            ], dim=1)
            obj_pos_local = compute_demo_cup_spawn_local(
                pregrasp_palm_pose,
                self.pregrasp_offset[:2],
                self.cfg.object_spawn_z,
                noise_xy,
            )
        else:
            q_pregrasp = self.robot_start_joint_pos[env_ids].clone()
            approach_hand = self.hand_approach_pose.unsqueeze(0).expand(n, -1)

            _xy_range = (
                self.grasp_adr.get_param("spawn", "object_spawn_xy_range")
                if self.grasp_adr is not None
                else self.cfg.object_spawn_xy_range
            )
            obj_x = self.cfg.object_spawn_x_center + (
                torch.rand(n, device=self.device) - 0.5
            ) * 2.0 * _xy_range
            obj_y = self.cfg.object_spawn_y_center + (
                torch.rand(n, device=self.device) - 0.5
            ) * 2.0 * _xy_range
            obj_pos_local = torch.stack(
                [obj_x, obj_y, torch.full((n,), self.cfg.object_spawn_z, device=self.device)], dim=1
            )

            noise = torch.stack([
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_x,
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_y,
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_z,
            ], dim=1)
            pregrasp_sensor_pos = obj_pos_local + self.pregrasp_offset.unsqueeze(0) + noise

            pregrasp_sensor_pose = torch.zeros(n, 6, device=self.device)
            pregrasp_sensor_pose[:, :3] = pregrasp_sensor_pos
            pregrasp_sensor_pose[:, 3] = math.radians(90.0)
            pregrasp_sensor_pose[:, 4] = math.radians(0.0)
            pregrasp_sensor_pose[:, 5] = math.radians(90.0)
            pregrasp_palm_pose = self._fabric_palm_pose_from_sensor_target(pregrasp_sensor_pose)
            pregrasp_palm_pose = torch.max(
                torch.min(pregrasp_palm_pose, self.palm_maxs.unsqueeze(0)),
                self.palm_mins.unsqueeze(0),
            )

            if self.cfg.cache_pregrasp_reset:
                xi = ((obj_x - self._cache_xs[0]) / (self._cache_xs[1] - self._cache_xs[0])).round().long().clamp(0, self._cache_n - 1)
                yi = ((obj_y - self._cache_ys[0]) / (self._cache_ys[1] - self._cache_ys[0])).round().long().clamp(0, self._cache_n - 1)
                q_pregrasp[:, :NUM_ARM_DOF] = self._cache_q_arm[xi, yi]
            else:
                q_pregrasp = self._run_reset_fabric(env_ids, pregrasp_palm_pose, q_pregrasp)
            q_pregrasp[:, NUM_ARM_DOF:] = approach_hand
            self.demo_lift_palm_target_buf[env_ids] = pregrasp_palm_pose

        # ---- 2. 로봇/Fabrics 상태 리셋 ----
        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_pos[:, self.actuated_dof_indices] = q_pregrasp
        full_pos[:, self.left_arm_dof_indices] = self.left_arm_zero_pos[0]
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        self.fabric_q[env_ids] = q_pregrasp
        self.fabric_qd[env_ids].zero_()
        self.fabric_qdd[env_ids].zero_()
        self.object_init_pos[env_ids] = obj_pos_local

        # ---- 5. pregrasp 버퍼 저장 ----
        self.pregrasp_arm_pos_buf[env_ids] = q_pregrasp[:, :NUM_ARM_DOF]
        self.palm_pose_targets[env_ids]    = pregrasp_palm_pose
        self.pregrasp_palm_pose_buf[env_ids] = pregrasp_palm_pose
        self.grasp_anchor_palm_pose_buf[env_ids] = pregrasp_palm_pose

        self.fabric.default_config[env_ids, :NUM_ARM_DOF] = q_pregrasp[:, :NUM_ARM_DOF]
        self.lift_finger_pos_buf[env_ids] = approach_hand

        # ---- 7. 컵 spawn ----
        obj_pos_world = obj_pos_local + self.scene.env_origins[env_ids]
        upright_rot = torch.zeros(n, 4, device=self.device)
        upright_rot[:, 0] = 1.0
        zero_vel = torch.zeros(n, 6, device=self.device)
        cup_root_state = torch.cat([obj_pos_world, upright_rot, zero_vel], dim=-1)
        self.cup.write_root_state_to_sim(cup_root_state, env_ids=env_ids)

        # ---- 7a. 컵 마찰계수 DR ----
        # μ_static ~ Uniform[cup_friction_min, cup_friction_max]
        # μ_dynamic = μ_static × 0.9
        # materials shape: (num_envs, num_shapes, 3) — [static, dynamic, restitution]
        # 6.4: cup_friction_fixed >= 0이면 고정값 사용 (friction ablation)
        if self.cfg.cup_friction_fixed >= 0.0:
            _friction_vals = torch.full((n,), self.cfg.cup_friction_fixed, device=self.device).cpu()
        else:
            _friction_vals = (
                torch.rand(n, device=self.device)
                * (self.cfg.cup_friction_max - self.cfg.cup_friction_min)
                + self.cfg.cup_friction_min
            ).cpu()
        _env_ids_cpu = (
            env_ids.cpu().int() if isinstance(env_ids, torch.Tensor)
            else torch.tensor(list(env_ids), dtype=torch.int32)
        )
        _materials = self.cup.root_physx_view.get_material_properties().clone()  # (N_envs, shapes, 3)
        _materials[_env_ids_cpu, :, 0] = _friction_vals.unsqueeze(-1)          # static friction
        _materials[_env_ids_cpu, :, 1] = (_friction_vals * 0.9).unsqueeze(-1)  # dynamic friction
        self.cup.root_physx_view.set_material_properties(_materials, _env_ids_cpu)
        self._cup_friction_static[env_ids] = _friction_vals.to(self.device)

        # ---- 7b. Bead 스폰 ----
        if not self.cfg.physical_beads_enabled:
            bead_count = torch.zeros(n, dtype=torch.long, device=self.device)
            dynamic_add_count = torch.zeros(n, dtype=torch.long, device=self.device)
            target_bead_count = torch.zeros(n, dtype=torch.long, device=self.device)
        elif self.cfg.dynamic_bead_spawn_enabled:
            bead_count = self._sample_bead_counts(
                n,
                self.cfg.bead_initial_count_min,
                self.cfg.bead_initial_count_max,
            )
            dynamic_add_count = self._sample_bead_counts(
                n,
                self.cfg.dynamic_bead_add_count_min,
                self.cfg.dynamic_bead_add_count_max,
            )
            target_bead_count = (bead_count + dynamic_add_count).clamp(max=int(self.cfg.num_beads))
        else:
            # 기존 이산 4단계: {0, 10, 20, 30}개 × 10g = {0, 100, 200, 300}g 추가 질량
            min_level = min(max(int(self.cfg.bead_count_min) // 10, 0), 3)
            max_level = min(max(int(self.cfg.bead_count_max) // 10, min_level), 3)
            _bead_lvl = torch.randint(min_level, max_level + 1, (n,), device=self.device)  # 0~3
            bead_count = _bead_lvl * 10  # {0, 10, 20, 30}
            dynamic_add_count = torch.zeros(n, dtype=torch.long, device=self.device)
            target_bead_count = bead_count

        bead_state = torch.zeros(n, self.cfg.num_beads, 13, device=self.device)
        hidden_pos = self.scene.env_origins[env_ids].unsqueeze(1) + self._hidden_bead_offsets_b.unsqueeze(0)
        bead_state[..., :3] = hidden_pos
        bead_state[..., 3] = 1.0

        for bi in range(self.cfg.num_beads):
            active = bead_count > bi
            if self.cfg.physical_beads_enabled and active.any():
                bead_pos = obj_pos_world + self._bead_offsets_b[bi].unsqueeze(0)
                bead_state[active, bi, :3] = bead_pos[active]
                bead_state[active, bi, 3]  = 1.0

        self.beads.write_object_state_to_sim(bead_state, env_ids=env_ids)
        self._bead_count_initial[env_ids] = bead_count
        self._bead_count_current[env_ids] = bead_count
        self._dynamic_bead_add_count[env_ids] = dynamic_add_count
        self._bead_count_target[env_ids] = target_bead_count
        self._dynamic_bead_spawned[env_ids] = False
        self._bead_mass_normalized[env_ids] = bead_count.float() / self.cfg.num_beads

        # ---- 8. 버퍼 리셋 ----
        self.hand_joint_targets[env_ids] = approach_hand
        self.contact_force_raw[env_ids] = 0.0
        self.binary_contact_buf[env_ids] = False
        self.num_contacts_buf[env_ids]   = 0
        self.distal_contact_force_raw[env_ids] = 0.0
        self.distal_binary_contact_buf[env_ids] = False
        self.middle_contact_force_raw[env_ids] = 0.0
        self.middle_binary_contact_buf[env_ids] = False
        self._prev_total_grip_force_buf[env_ids] = 0.0
        self._prev_num_contacts_buf[env_ids] = 0.0
        self._prev_middle_contacts_buf[env_ids] = 0.0
        self._prev_cup_tilt_deg_buf[env_ids] = 0.0
        self._contact_persistence_buf[env_ids] = 0
        self._lift_contact_hold_count[env_ids] = 0
        self._grasp_started_buf[env_ids] = False
        self._grasp_anchor_set_buf[env_ids] = False
        self._approach_ready_buf[env_ids] = False
        self._approach_timeout_buf[env_ids] = False
        self._grasp_from_timeout_buf[env_ids] = False
        self._lift_contact_ready_latched_buf[env_ids] = False
        self._lift_started_buf[env_ids] = False
        self._lift_start_step_buf[env_ids] = 0
        self._contacts_at_lift_start_buf[env_ids] = 0.0
        self._palm_at_lift_start_buf[env_ids] = 0.0
        self._grasp_tilt_at_lift_start_buf[env_ids] = 0.0
        self._force_ratio_at_lift_start_buf[env_ids] = 0.0
        self._full_grip_hold_count[env_ids] = 0
        self._lift_success_hold_count[env_ids] = 0
        self._full_grip_ready_buf[env_ids] = False
        self._full_grip_ready_latched_buf[env_ids] = False
        self._stabilize_started_buf[env_ids] = False
        self._stabilize_start_step_buf[env_ids] = 0
        self._grip_ready_hold_count[env_ids] = 0
        self._grip_ready_latched_buf[env_ids] = False
        self._lift_success_latched_buf[env_ids] = False
        self._stabilize_success_latched_buf[env_ids] = False
        self.success_flag[env_ids] = False
        self._success_hold_count[env_ids] = 0
        self.is_grasp_phase[env_ids] = False
        self.is_lift_phase[env_ids] = False
        self.is_stabilize_phase[env_ids] = False
        self.is_post_grasp_phase[env_ids] = False
        self._eval_grip_at_lift[env_ids] = 0.0
        self._eval_finger_actions_at_lift[env_ids] = 0.0
        self._last_grasp_finger_action[env_ids] = 0.0
        # ---- Fabrics 상태 초기화 ----
        # 리셋 시 실제 로봇 상태로 동기화하여 첫 프레임 튐 방지
        # fabric_q(27D) = arm(7D) + hand(20D)
        arm_pos  = self.robot.data.joint_pos[env_ids][:, self.arm_dof_indices]
        hand_pos = self.robot.data.joint_pos[env_ids][:, self.hand_dof_indices]
        self.fabric_q[env_ids] = torch.cat([arm_pos, hand_pos], dim=-1)
        
        self.fabric_qd[env_ids] = 0.0
        self.fabric_qdd[env_ids] = 0.0
        
        self.hand_joint_targets[env_ids] = hand_pos
        self.palm_pose_targets[env_ids]  = self.pregrasp_palm_pose_buf[env_ids]

        self._eval_episode_started[env_ids] = False
        self._eval_grasp_action_sum[env_ids] = 0.0
        self._eval_grasp_action_sq_sum[env_ids] = 0.0
        self._eval_grasp_action_min[env_ids] = float("inf")
        self._eval_grasp_action_max[env_ids] = float("-inf")
        self._eval_grasp_action_count[env_ids] = 0
        self._eval_lift_action_sum[env_ids] = 0.0
        self._eval_lift_action_sq_sum[env_ids] = 0.0
        self._eval_lift_force_delta_sum[env_ids] = 0.0
        self._eval_lift_contact_delta_sum[env_ids] = 0.0
        self._eval_lift_action_count[env_ids] = 0
        self._eval_lift_snapshot_valid[env_ids] = False

        # actions 리셋: absolute synergy action=0 = approach/grasp 중간값
        self.actions[env_ids, :6] = 0.0
        self.actions[env_ids, 6:] = 0.0
        self.prev_actions[env_ids, :6] = 0.0
        self.prev_actions[env_ids, 6:] = 0.0
        self._palm_target_delta_buf[env_ids] = 0.0
        self._ema_palm_action[env_ids] = 0.0
        self.lift_palm_start_pose_buf[env_ids] = self.pregrasp_palm_pose_buf[env_ids]

    def _uniform_scale(self, shape: tuple[int, int], value_range: tuple[float, float]) -> torch.Tensor:
        low, high = value_range
        return torch.empty(shape, device=self.device).uniform_(float(low), float(high))

    def _apply_real2sim_actuator_randomization(self, env_ids: Sequence[int]) -> None:
        if not self.cfg.real2sim_actuator_randomization_enabled:
            return

        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids_tensor.numel() == 0:
            return

        for joint_ids in self.real2sim_actuator_group_indices.values():
            if not joint_ids:
                continue
            shape = (int(env_ids_tensor.numel()), len(joint_ids))
            default_stiffness = self.robot.data.default_joint_stiffness[env_ids_tensor][:, joint_ids]
            default_damping = self.robot.data.default_joint_damping[env_ids_tensor][:, joint_ids]
            default_friction = self.robot.data.default_joint_friction_coeff[env_ids_tensor][:, joint_ids]

            stiffness = default_stiffness * self._uniform_scale(
                shape, self.cfg.real2sim_stiffness_scale_range
            )
            damping = default_damping * self._uniform_scale(
                shape, self.cfg.real2sim_damping_scale_range
            )
            self.robot.write_joint_stiffness_to_sim(
                stiffness,
                joint_ids=joint_ids,
                env_ids=env_ids_tensor,
            )
            self.robot.write_joint_damping_to_sim(
                damping,
                joint_ids=joint_ids,
                env_ids=env_ids_tensor,
            )
            if torch.any(default_friction != 0.0):
                self.robot.write_joint_friction_coefficient_to_sim(
                    default_friction * self._uniform_scale(shape, self.cfg.real2sim_friction_scale_range),
                    joint_ids=joint_ids,
                    env_ids=env_ids_tensor,
                )
