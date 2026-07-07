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

import json
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
    action_policy_scalars,
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
    NUM_ROBOT_DOF,  # 우측 한 팔 DOF(13) = fabric cspace 우측 슬라이스 [0:13]
    NUM_ACTIONS,
    NUM_PALM_ACTION,
    NUM_HAND_PCA,
    HAND_PCA_MINS,
    HAND_PCA_MAXS,
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
    RIGHT_HAND_MIMIC_JOINT_NAMES,
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
from .grasp_right_utils import scale, tensor_clamp, to_torch
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
        """fabric IK 가 r_hl_palm_sensor 를 직접 제어하므로 항등(변환 불필요).

        기존 Tesollo palm_link offset (0,0.03,0.04)는 실제 palm_sensor 와 위치 3.4cm
        어긋난 오차였다(관측 palm_center_pos 는 실 palm_sensor, 제어 target 은 3.4cm 벗어난
        palm_link → palm-first 안착 실패 원인). palm_sensor 직접 제어로 관측과 IK 제어점이
        같은 프레임이 되어 정합한다. 회전 규약은 env 가 palm_sensor 기준(euler_zyx)으로
        지정한다 — palm_link 대비 ex 축 +90°(R_ls = Rx(90°)).
        """
        return palm_sensor_pose

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
        # mimic 원위 관절 인덱스 — warm-state export 에 실제 mimic 물리값을 저장해
        # pour warmstart 가 손 전체 자세(drive+mimic)를 정확히 재현하도록 한다.
        self._hand_mimic_dof_indices = [
            self.robot.joint_names.index(name) for name in RIGHT_HAND_MIMIC_JOINT_NAMES
        ]
        # 원위(mimic) 능동 curl: PhysxMimicJoint 미결합으로 원위가 안 닫혀 fingertip pinch가 됨.
        # _apply_action 에서 mimic = drive×mult 로 구동해 원위가 컵을 감싸는 envelope grip 유도.
        # drive(finger_target) 순서 [thumb_1,thumb_2,index_1,middle_1,ring_1,pinky_1] 기준 src/mult.
        self._hand_mimic_src_idx = torch.tensor([1, 1, 2, 3, 4, 5], device=self.device)
        self._hand_mimic_mult = torch.tensor(
            [1.1425, 1.1425 * 0.7508, 1.1169, 1.1169, 1.1169, 1.1169], device=self.device
        )
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
            "r_hl_thumb_1", "r_hl_index_1",
            "r_hl_middle_1", "r_hl_ring_1",
            "r_hl_pinky_1",
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
        # palm-first freeze: thumb_1(idx0) 열만 True 인 (1,6) 마스크. approach 중 palm 근접 전까지
        # 이 열만 approach 값으로 덮어써 엄지 opposition 통로를 유지한다.
        self.thumb_freeze_col_mask = torch.zeros(
            1, NUM_HAND_DOF, dtype=torch.bool, device=self.device
        )
        self.thumb_freeze_col_mask[0, 0] = True
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

        # -------- DEXTRAH goal-driven 버퍼 (grasp_v2) --------
        # object_goal: 물체를 옮겨 들어올릴 목표 위치(고정, Phase4 에서 dynamic 샘플 고려).
        self.object_goal = to_torch(
            list(self.cfg.object_goal_pos), device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)          # (N, 3)
        # curled_q: finger_curl_reg 타깃(감싸기 자세). rh56f1 6-drive grasp pose 로 초기화.
        self.curled_q = self.hand_grasp_pose.unsqueeze(0).repeat(self.num_envs, 1)  # (N, 6)
        # PCA action 범위(uncentered 투영). action[-1,1] → [mins,maxs] scale.
        self.hand_pca_mins = to_torch(HAND_PCA_MINS, device=self.device)  # (5,)
        self.hand_pca_maxs = to_torch(HAND_PCA_MAXS, device=self.device)  # (5,)
        # DEXTRAH reward 중간값
        self.hand_to_object_pos_error = torch.ones(self.num_envs, device=self.device)
        self.object_to_goal_pos_error = torch.zeros(self.num_envs, device=self.device)
        self.object_vertical_error    = torch.zeros(self.num_envs, device=self.device)

        # 실험3b: apply_object_wrench 외란 버퍼 (firm grip)
        self.object_applied_force  = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.object_applied_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._object_mass = None   # get_masses() 1회 캐시(매 step CPU 조회 방지 → fps)

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
        # palm-first envelope: approach 중 thumb_1 이 palm 근접까지 고정된 env (진단 로깅용)
        self._thumb_frozen_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
        # 매 step force_ratio(grip force / mg). success 게이트에서 grip force 하중충족 판정.
        self._force_ratio_buf = torch.zeros(self.num_envs, device=self.device)
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
        # leaky hold_count(float): miss 시 success_hold_miss_decay 감쇠 → flicker 허용
        self._success_hold_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
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

        # cspace attractor: reset warm-start hand pose와 일치 (오른손[7:13]만; 좌측[13:26] 중립 유지)
        cspace_default = self.fabric.default_config.clone()  # 26D
        cspace_default[:, NUM_ARM_DOF:NUM_ROBOT_DOF] = \
            self.hand_approach_pose.unsqueeze(0).expand(self.num_envs, -1)
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
                # grasp_v2: MultiAsset(replicate_physics=False)는 GPU contact filter 미지원 →
                # filter 제거, net_forces_w 사용(물체 구분 없이 접촉력).
                history_length=1,
                track_air_time=False,
            ))
            self._tip_sensors.append(sensor)
            self.scene.sensors[f"tip_sensor_{link_name}"] = sensor

        # 근위(proximal) 마디 접촉 센서 — sim-only(실물엔 tip 힘센서만). envelope 그립 여부를
        # 계측하고 critic privileged obs 로 노출한다(테솔로 grasp middle-contact 방식과 동일).
        self._middle_sensors: list[ContactSensor] = []
        for link_name in self.cfg.right_middle_contact_links:
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link_name}",
                # grasp_v2: MultiAsset(replicate_physics=False)는 GPU contact filter 미지원 →
                # filter 제거, net_forces_w 사용(물체 구분 없이 접촉력).
                history_length=1,
                track_air_time=False,
            ))
            self._middle_sensors.append(sensor)
            self.scene.sensors[f"middle_sensor_{link_name}"] = sensor

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
            use_hand_fabric=self.cfg.use_hand_fabric,   # grasp_v2: True → fabric 이 손도 제어
            hand_mode=self.cfg.hand_mode,               # "pca": 5D PCA action
        )
        num_joints = self.fabric.num_joints

        self.fabric_integrator = DisplacementIntegrator(self.fabric)

        # fabric cspace = 26D [r_arm7, r_hand6, l_arm7, l_hand6].
        # 좌측[13:26]은 fabric default(중립) 유지(로봇 왼팔은 left_arm_zero_pos 로 별도 구동 —
        # fabric 좌측은 nullspace 입력용). 우측[0:13]만 robot_start(arm_start+hand_approach)로 초기화.
        self.fabric_q   = self.fabric.default_config.clone().contiguous()
        self.fabric_q[:, :NUM_ROBOT_DOF] = self.robot_start_joint_pos
        self.fabric_qd  = torch.zeros(self.num_envs, num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, num_joints, device=self.device)

        # 실험1 direct: hand_target 은 6D drive 관절 목표(hand_mode="direct").
        # (PCA 복귀 시 NUM_HAND_PCA=5). use_hand_fabric=True → fabric 이 손 제어.
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

        reset_cspace = self._reset_fabric.default_config.clone()  # 26D, 좌측 중립 유지
        reset_cspace[:, NUM_ARM_DOF:NUM_ROBOT_DOF] = \
            self.hand_approach_pose.unsqueeze(0).expand(self._reset_chunk, -1)
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
        # euler_zyx (ez, ey, ex). side grasp: palm_sensor +z(법선)가 컵(-y 접근 → +y)을 향한다.
        # (ez,ey,ex)=(180,0,90) → palm_sensor +z = (0,+1,0)=+y. 팔의 자연 수평자세와 일치
        # (ez=90 은 +x 목표라 도달 못 하고 47° 기울어 정착하는 문제였음).
        palm_sensor[:, 3] = math.radians(180.0)
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

        # reset_fabric 은 26D cspace. q_init 이 우측 13D 로 오면 좌측 중립(default[13:26])을
        # 붙여 26D 로 확장하고, 반환은 다시 우측 13D 로 슬라이스한다(호출부 계약 유지).
        right_only = q_init.shape[1] == NUM_ROBOT_DOF
        if right_only:
            left_neutral = self._reset_fabric.default_config[0, NUM_ROBOT_DOF:].unsqueeze(0)
            q_init = torch.cat([q_init, left_neutral.expand(q_init.shape[0], -1)], dim=1)

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

        return q_out[:, :NUM_ROBOT_DOF] if right_only else q_out

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
            "hand_mimic_pos": torch.empty(target_count, len(RIGHT_HAND_MIMIC_JOINT_NAMES), dtype=torch.float32),
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
        self._write_warm_export_progress(
            count=0, target=target_count, added=0, status="running"
        )

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
        export["hand_mimic_pos"][start:end] = (
            self.robot.data.joint_pos[success_env_ids][:, self._hand_mimic_dof_indices].detach().cpu()
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
        self._write_warm_export_progress(
            count=self._warm_state_export_count,
            target=target_count,
            added=count,
            status="running",
        )
        print(
            f"[warm_state_export] {self._warm_state_export_count}/{target_count} "
            f"성공 상태 수집 (+{count})",
            flush=True,
        )
        if self._warm_state_export_count >= target_count:
            self._write_warm_state_export_file()

    def _write_warm_export_progress(
        self,
        *,
        count: int,
        target: int,
        added: int,
        status: str,
    ) -> None:
        """수집 진행 상황을 ``<out>.progress.json`` 으로 원자적 기록 (외부 폴링용)."""
        path = self._warm_state_export_path.with_name(
            self._warm_state_export_path.name + ".progress.json"
        )
        tmp = path.with_suffix(path.suffix + ".tmp")
        data = {
            "count": int(count),
            "target": int(target),
            "added": int(added),
            "status": status,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, sort_keys=True)
                f.write("\n")
            tmp.replace(path)
        except OSError:
            pass

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
        self._write_warm_export_progress(
            count=count,
            target=int(self.cfg.warm_state_target_count),
            added=0,
            status="complete",
        )
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
            s.data.net_forces_w[:, 0, :] for s in self._tip_sensors
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

        # 근위(proximal) 마디 접촉 = envelope signature. sim-only 센서에서 컵 접촉력을 읽어
        # 채운다(계측 + critic privileged). reward 는 slip weight 0 으로 중립(reward-neutral).
        mid_xyz = torch.stack([
            s.data.net_forces_w[:, 0, :] for s in self._middle_sensors
        ], dim=1)
        mid_xyz = torch.nan_to_num(mid_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        mid_norms = mid_xyz.norm(dim=-1)
        self.middle_contact_force_xyz.copy_(mid_xyz)
        self.middle_contact_force_raw.copy_(mid_norms)
        self.middle_binary_contact_buf.copy_(mid_norms > CONTACT_FORCE_THRESHOLD)

        palm_xyz = torch.nan_to_num(self._palm_sensor.data.net_forces_w[:, 0, :], nan=0.0, posinf=0.0, neginf=0.0)
        per_palm = palm_xyz.norm(dim=-1)
        self.palm_contact_force_xyz.copy_(palm_xyz)
        self.palm_contact_force_raw.copy_(per_palm)
        self.palm_binary_contact_buf.copy_(per_palm > CONTACT_FORCE_THRESHOLD)

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """DEXTRAH goal-driven: action(palm 6 + PCA 5) -> fabric IK. phase/synergy 없음."""
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone()

        # ---- action -> palm pose target + hand PCA target (absolute scale) ----
        palm_actions = actions[:, :NUM_PALM_ACTION]                # (N, 6) in [-1,1]
        hand_actions = actions[:, NUM_PALM_ACTION:NUM_ACTIONS]     # (N, 5) in [-1,1]

        # settle: episode 초기 settle_steps 동안 손을 열린 자세(action=-1 -> pca_mins)로 고정 ->
        # 물체 낙하 안착까지 파지 억제(tesollo grasp_v2 방식). settle 후 정책이 PCA 자유 제어.
        if int(self.cfg.settle_steps) > 0:
            in_settle = (self.episode_length_buf < int(self.cfg.settle_steps)).unsqueeze(1)
            hand_actions = torch.where(in_settle, -torch.ones_like(hand_actions), hand_actions)

        palm_pose = tensor_clamp(
            scale(palm_actions, self.palm_mins, self.palm_maxs),
            self.palm_mins, self.palm_maxs,
        )
        self._palm_target_delta_buf.copy_(palm_pose - self.palm_pose_targets)
        self.palm_pose_targets.copy_(palm_pose)

        # 실험1 direct: hand action(6) → 관절 한계로 scale → drive 6 목표.
        # (PCA 복귀 시 scale 대상을 self.hand_pca_mins/maxs 로.)
        hand_tgt = tensor_clamp(
            scale(hand_actions, self.hand_joint_lower_limits, self.hand_joint_upper_limits),
            self.hand_joint_lower_limits, self.hand_joint_upper_limits,
        )
        self.hand_pca_targets.copy_(hand_tgt)

        # ---- fabric: palm pose + PCA hand -> arm+hand cspace 통합 IK ----
        # use_hand_fabric=True 라 fabric integrator 가 손(drive 6)까지 업데이트(수동 동기화 불필요).
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

        # 실험3b: 파지 시 랜덤 외란(firm grip 강제, DEXTRAH apply_object_wrench)
        self._apply_object_wrench()

    def _apply_object_wrench(self) -> None:
        """손이 물체를 잡았을 때(hand_to_object<threshold) 랜덤 force/torque 외란 →
        정책이 물체를 goal 로 옮기려면 견디며 꽉 잡아야 함(firm grip 간접학습). 외란 크기는
        ADR(object_wrench.max_linear_accel)로 0→강 점증. DEXTRAH apply_object_wrench 이식."""
        if not self.cfg.enable_object_wrench:
            return
        # ADR increment 0(외란 크기 0) 단계에선 set/write_data_to_sim 스킵 → fps 정상 유지.
        # lifted_rate 가 threshold 넘어 increment 시작되면 외란 켜짐(firm grip 단계).
        if self.grasp_adr is None or self.grasp_adr.increment_counter == 0:
            return
        # object mass (N,1) — MultiAsset 이라 env 마다 다름. 물체 고정이라 1회만 조회·캐시.
        if self._object_mass is None:
            self._object_mass = self.cup.root_physx_view.get_masses().to(self.device)[:, 0:1]
        object_mass = self._object_mass
        max_accel = (
            self.grasp_adr.get_param("object_wrench", "max_linear_accel")
            if self.grasp_adr is not None else 0.0
        )
        linear_accel = max_accel * torch.rand(self.num_envs, 1, device=self.device)
        max_force  = (linear_accel * object_mass).unsqueeze(2)                          # (N,1,1)
        max_torque = (object_mass * linear_accel * self.cfg.torsional_radius).unsqueeze(2)

        def _rand_dir() -> torch.Tensor:
            v = torch.randn(self.num_envs, 1, 3, device=self.device)
            return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        forces  = max_force  * _rand_dir()
        torques = max_torque * _rand_dir()

        # wrench_trigger_every step 마다 새 방향, 그 외엔 유지
        new_dir = (self.episode_length_buf.view(-1, 1, 1) % int(self.cfg.wrench_trigger_every)) == 0
        self.object_applied_force  = torch.where(new_dir, forces,  self.object_applied_force)
        self.object_applied_torque = torch.where(new_dir, torques, self.object_applied_torque)

        # 손이 물체를 잡았을 때만 외란 적용(아니면 0)
        applied = (self.hand_to_object_pos_error <= self.cfg.hand_to_object_dist_threshold)[:, None, None]
        self.object_applied_force  = torch.where(applied, self.object_applied_force,  torch.zeros_like(self.object_applied_force))
        self.object_applied_torque = torch.where(applied, self.object_applied_torque, torch.zeros_like(self.object_applied_torque))

        self.cup.set_external_force_and_torque(
            forces=self.object_applied_force, torques=self.object_applied_torque
        )
        self.cup.write_data_to_sim()

    def _apply_action(self) -> None:
        """fabric cspace(arm 7 + hand 6) -> 로봇 관절 target. mimic 원위는 drive 추종."""
        # 오른팔
        self.robot.set_joint_position_target(
            self.fabric_q[:, :NUM_ARM_DOF], joint_ids=self.arm_dof_indices
        )
        self.robot.set_joint_velocity_target(
            torch.zeros_like(self.fabric_q[:, :NUM_ARM_DOF]), joint_ids=self.arm_dof_indices
        )
        # 오른손 drive 6 (fabric PCA IK 결과)
        hand_target = self.fabric_q[:, NUM_ARM_DOF:NUM_ROBOT_DOF]
        self.robot.set_joint_position_target(hand_target, joint_ids=self.hand_dof_indices)
        # 원위(mimic) 마디: drive x mult 로 능동 curl (하드웨어 결합 유지)
        _mimic_target = hand_target[:, self._hand_mimic_src_idx] * self._hand_mimic_mult.unsqueeze(0)
        self.robot.set_joint_position_target(_mimic_target, joint_ids=self._hand_mimic_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(hand_target), joint_ids=self.hand_dof_indices
        )
        # 왼팔 고정
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
    # Observations: Actor 96D (with oracle mass 97) | Critic 119D
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
            self.object_goal,       # 3  (DEXTRAH goal)
            last_actions,           # 11
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
            self.object_goal,         # 3 (DEXTRAH goal)
            last_actions,             # 11
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
            self.middle_binary_contact_buf.float(),     # 5 (근위 접촉 = envelope, privileged)
        ], dim=-1)   # 119D

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
        """DEXTRAH: hand_to_object + object_to_goal + finger_curl_reg + lift."""
        self.extras.clear()

        # ---- 중간값 ----
        # hand keypoints: fingertip 5 + palm_center 1 -> object 중심까지 MAX 거리(모든 손점 밀착 강제 -> 감싸기).
        hand_pos = torch.cat(
            [self.fingertip_pos, self.palm_center_pos.unsqueeze(1)], dim=1
        )  # (N, 6, 3)
        self.hand_to_object_pos_error = torch.norm(
            hand_pos - self.object_pos.unsqueeze(1), dim=-1
        ).max(dim=-1).values
        self.object_to_goal_pos_error = torch.norm(self.object_pos - self.object_goal, dim=-1)
        self.object_vertical_error = torch.abs(self.object_goal[:, 2] - self.object_pos[:, 2])

        # fabric hand cspace(drive 6) - finger_curl_reg 용
        hand_dof_pos = self.fabric_q[:, NUM_ARM_DOF:NUM_ROBOT_DOF]

        # ---- reward terms ----
        hand_to_object_reward = self.cfg.hand_to_object_weight * torch.exp(
            -self.cfg.hand_to_object_sharpness * self.hand_to_object_pos_error
        )
        object_to_goal_reward = self.cfg.object_to_goal_weight * torch.exp(
            self.cfg.object_to_goal_sharpness * self.object_to_goal_pos_error
        )
        finger_curl_dist = (hand_dof_pos - self.curled_q).norm(p=2, dim=-1)
        finger_curl_reg = self.cfg.finger_curl_reg_weight * finger_curl_dist ** 2
        lift_reward = self.cfg.lift_weight * torch.exp(
            -self.cfg.lift_sharpness * self.object_vertical_error
        )

        total_reward = (
            hand_to_object_reward + object_to_goal_reward + finger_curl_reg + lift_reward
        )

        in_success = self.object_to_goal_pos_error < self.cfg.object_goal_tol

        # ---- 로깅(grasp_v1 형식: "group/key") ----
        self.extras["reward/hand_to_object"] = hand_to_object_reward.mean()
        self.extras["reward/object_to_goal"] = object_to_goal_reward.mean()
        self.extras["reward/finger_curl_reg"] = finger_curl_reg.mean()
        self.extras["reward/lift"] = lift_reward.mean()
        self.extras["reward/total"] = total_reward.mean()
        self.extras["metric/hand_to_object_err"] = self.hand_to_object_pos_error.mean()
        self.extras["metric/object_to_goal_err"] = self.object_to_goal_pos_error.mean()
        self.extras["metric/object_height"] = self.object_pos[:, 2].mean()
        self.extras["metric/in_success_rate"] = in_success.float().mean()

        # 실험3b: wrench ADR 커리큘럼 — lifted_rate 가 threshold 넘으면 외란 난이도 점증(firm grip 강화)
        lifted_rate = (self.object_pos[:, 2] > self.cfg.wrench_lifted_z).float().mean()
        self.extras["metric/lifted_rate"] = lifted_rate
        if self.grasp_adr is not None:
            self.grasp_adr.maybe_increment(lifted_rate)
            self.extras["adr/increment"] = float(self.grasp_adr.increment_counter)
            self.extras["adr/wrench_accel"] = self.grasp_adr.get_param("object_wrench", "max_linear_accel")

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """DEXTRAH: 물체가 작업공간(xy 범위)을 벗어나거나 낙하(z) 하면 종료."""
        self._compute_intermediate_values()

        out_x = (
            (self.object_pos[:, 0] < self.cfg.obj_out_x_min)
            | (self.object_pos[:, 0] > self.cfg.obj_out_x_max)
        )
        out_y = (
            (self.object_pos[:, 1] < self.cfg.obj_out_y_min)
            | (self.object_pos[:, 1] > self.cfg.obj_out_y_max)
        )
        fallen = self.object_pos[:, 2] < self.cfg.obj_fallen_z
        out_of_reach = out_x | out_y | fallen

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return out_of_reach, time_out

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
            # side grasp: palm_sensor +z(법선)가 컵(-y 접근 → +y)을 향하도록 (180,0,90). cache 와 일치.
            pregrasp_sensor_pose[:, 3] = math.radians(180.0)
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
            # r_aj_7(손목, arm index 6)을 낮춰 palm 을 컵 높이로 내림. fabric 은 +y 수평 유지
            # 위해 r_aj_7 을 높게 잡아 palm 이 컵 rim 에 뜨므로(probe 확정), bias 로 끌어내린다.
            # bias 후 실제 palm(FK)로 anchor 를 정합해 정책 시작 시 palm 튐 방지.
            if self.cfg.pregrasp_r_aj7_bias != 0.0:
                q_pregrasp[:, 6] = q_pregrasp[:, 6] - self.cfg.pregrasp_r_aj7_bias
                fq_fk = self.fabric.default_config.clone()
                fq_fk[env_ids, :NUM_ROBOT_DOF] = q_pregrasp
                pregrasp_palm_pose = self.fabric.get_palm_pose(fq_fk, "euler_zyx")[env_ids]
            self.demo_lift_palm_target_buf[env_ids] = pregrasp_palm_pose

        # ---- 2. 로봇/Fabrics 상태 리셋 ----
        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_pos[:, self.actuated_dof_indices] = q_pregrasp
        full_pos[:, self.left_arm_dof_indices] = self.left_arm_zero_pos[0]
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        self.fabric_q[env_ids, :NUM_ROBOT_DOF] = q_pregrasp  # 좌측[13:26]은 중립 유지
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
            # 가상질량(hidden-mass): 물리 bead 없이 bead_count 만 랜덤화 → effective_mass/
            # force_ratio/critic oracle 에 반영(물리 스폰은 아래 physical_beads_enabled 가드로 차단).
            # bead_count_max=0 이면 {0} 고정으로 하위호환. {0,10,20,30}개 → {170,270,370,470}g.
            _min_lvl = min(max(int(self.cfg.bead_count_min) // 10, 0), 3)
            _max_lvl = min(max(int(self.cfg.bead_count_max) // 10, _min_lvl), 3)
            bead_count = torch.randint(_min_lvl, _max_lvl + 1, (n,), device=self.device) * 10
            dynamic_add_count = torch.zeros(n, dtype=torch.long, device=self.device)
            target_bead_count = bead_count
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
        self._thumb_frozen_buf[env_ids] = False
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
        # fabric_q 우측[0:13] = 실 로봇 오른팔(7)+오른손(6); 좌측[13:26]은 중립 유지.
        arm_pos  = self.robot.data.joint_pos[env_ids][:, self.arm_dof_indices]
        hand_pos = self.robot.data.joint_pos[env_ids][:, self.hand_dof_indices]
        self.fabric_q[env_ids, :NUM_ROBOT_DOF] = torch.cat([arm_pos, hand_pos], dim=-1)
        
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
