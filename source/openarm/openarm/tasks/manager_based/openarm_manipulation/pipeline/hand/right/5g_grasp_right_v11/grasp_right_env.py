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

"""환경 클래스: 5g_grasp_right_v11

v10: v9 기반 버그 수정
- Fix 1: rj_dg_1_1 (thumb abduction) = 0.0 고정 (v9: -0.283 → 엄지 치우침 수정)
- Fix 2: MIN_CONTACTS_FOR_SUCCESS = 4, ADR과 분리 (v9: 2접촉 success 오판정 수정)
- Fix 3: has_5_contact = num_contacts>=5 고정 (v9: has_4_contact와 동일 식 버그 수정)

Action (26D):
  [0:6]  6D palm pose → Fabrics IK → arm 7 DOF
  [6:26] 20D per-joint finger delta: reference_pose + action × finger_delta_scale [rad]

Episode (18s @ 60Hz):
  Grasp     phase (0~479):    Fabrics arm + per-joint finger delta
  Lift      phase (480~719):  goal-pose lift + micro-delta hand
  Stabilize phase (720~839):  hold/re-grip stabilization
  Transport phase (840~1079): goal-pose transport + grasp maintenance
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

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

from .grasp_right_env_cfg import GraspRightEnvCfg
from .grasp_adr import GraspADR
from .grasp_right_constants import (
    NUM_ARM_DOF,
    NUM_HAND_DOF,
    NUM_FINGERTIPS,
    NUM_DISTAL_SENSORS,
    NUM_MIDDLE_SENSORS,
    NUM_PALM_SENSORS,
    NUM_CRITIC_OBSERVATIONS,
    GRASP_PHASE_STEPS,
    LIFT_PHASE_STEPS,
    LIFT_START_STEP,
    STABILIZE_START_STEP,
    STABILIZE_PHASE_STEPS,
    TRANSPORT_START_STEP,
    TRANSPORT_PHASE_STEPS,
    EPISODE_STEPS,
    PRELOAD_START_STEP,
    CONTACT_FORCE_THRESHOLD,
    CONTACT_FORCE_MAX,
    MIN_CONTACTS_FOR_SUCCESS,
    PREGRASP_FABRICS_STEPS,
    CUP_RADIUS_APPROX,
    ARM_START_POSE,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
)
from .grasp_right_preset import (
    LEFT_ARM_REST_JOINT_POS,
    RIGHT_ACTUATED_JOINT_NAMES,
    HAND_APPROACH_POSE,
    HAND_GRASP_POSE,
    HAND_FULL_GRIP_POSE,
)
from .finger_action_utils import (
    compute_grasp_finger_targets,
    compute_lift_finger_targets,
    resolve_grasp_delta_scale,
)
from .grasp_reward_utils import (
    compute_middle_contact_gate,
    compute_slip_proxy,
    compute_transport_success_mask,
    compute_upright_success_mask,
)
from .grasp_right_utils import scale, to_torch
from .demo_grasp_reset import DemoGraspResetBank, compute_demo_cup_spawn_local


class GraspRightEnv(DirectRLEnv):
    """OpenArm+Teosllo 오른손 파지 환경 v11.

    Action: 26D
      [0:6]  palm pose (x,y,z,ez,ey,ex), 정규화 [-1,1] → Fabrics IK
      [6:26] 20D per-joint finger delta: reference_pose + action × finger_delta_scale [rad]

    Episode:
      Grasp     phase (step 0~479):    Fabrics arm + per-joint finger delta
      Lift      phase (step 480~719):  goal-pose lift + micro-delta hand
      Stabilize phase (step 720~839):  hold/re-grip stabilization
      Transport phase (step 840~1079): goal-pose transport + grasp maintenance
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
        self.real2sim_actuator_group_indices = {
            "openarm_right_arm": self.arm_dof_indices,
            "tesollo_hand_abduction": [
                idx for idx in self.hand_dof_indices if self.robot.joint_names[idx].endswith("_1")
            ],
            "tesollo_hand_curl": [
                idx for idx in self.hand_dof_indices if self.robot.joint_names[idx].endswith("_2")
            ],
            "tesollo_hand_pip": [
                idx for idx in self.hand_dof_indices if self.robot.joint_names[idx].endswith("_3")
            ],
            "tesollo_hand_dip": [
                idx for idx in self.hand_dof_indices if self.robot.joint_names[idx].endswith("_4")
            ],
            "openarm_left_arm": self.left_arm_dof_indices,
        }

        # body indices
        _tip_names = [f"rl_dg_{i}_tip" for i in range(1, 6)]
        self.fingertip_body_indices: list[int] = [
            self.robot.data.body_names.index(name) for name in _tip_names
        ]
        _palm_name = "rl_dg_palm"
        self.palm_body_index: int = (
            self.robot.data.body_names.index(_palm_name)
            if _palm_name in self.robot.data.body_names
            else -1
        )
        _distal4_names = [f"rl_dg_{i}_4" for i in range(1, 6)]
        self.distal4_body_indices: list[int] = [
            self.robot.data.body_names.index(name)
            for name in _distal4_names
            if name in self.robot.data.body_names
        ]

        # middle phalanx (PIP, _3 link) body indices — FK 기반 actor obs용
        # sim2real 가능: joint encoder FK로 실기에서도 계산 가능
        _middle3_names = [f"rl_dg_{i}_3" for i in range(1, 6)]
        self.middle3_body_indices: list[int] = [
            self.robot.data.body_names.index(name)
            for name in _middle3_names
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

        # pregrasp palm pose 버퍼
        self.pregrasp_palm_pose_buf   = torch.zeros(self.num_envs, 6, device=self.device)
        self.lift_palm_start_pose_buf = torch.zeros(self.num_envs, 6, device=self.device)
        self.demo_lift_palm_target_buf = torch.zeros(self.num_envs, 6, device=self.device)
        self.transport_palm_start_pose_buf = torch.zeros(self.num_envs, 6, device=self.device)
        self.transport_palm_target_pose_buf = torch.zeros(self.num_envs, 6, device=self.device)
        self.transport_object_start_pos_buf = torch.zeros(self.num_envs, 3, device=self.device)
        self.demo_grasp_reset_bank = (
            DemoGraspResetBank.from_hdf5_paths(cfg.demo_grasp_pose_paths, device=self.device)
            if cfg.enable_demo_grasp_reset
            else None
        )

        # ----------------------------------------------------------------
        # Hand 관절 한계 (per-joint delta 클램프용)
        # soft_joint_pos_limits: (num_envs, num_joints, 2) — [lower, upper]
        # ----------------------------------------------------------------
        hand_limits = self.robot.data.soft_joint_pos_limits[0, self.hand_dof_indices, :]  # (20, 2)
        self.hand_joint_lower_limits = hand_limits[:, 0].contiguous()  # (20,)
        self.hand_joint_upper_limits = hand_limits[:, 1].contiguous()  # (20,)

        # 외전/내전 관절(abduction) delta scale 마스크 — 0으로 설정 시 사실상 고정
        # RIGHT_HAND_JOINT_NAMES = [rj_dg_{f}_{j} for f in 1~5, j in 1~4]
        # rj_*_1 abduction은 HAND_GRASP_POSE 기준으로 고정해 closure 탐색을 curl 관절에 집중한다.
        # index 17 (pinky Z-flex)도 기존처럼 고정한다.
        self.finger_delta_mask = torch.ones(NUM_HAND_DOF, device=self.device)
        self.finger_delta_mask[[0, 4, 8, 12, 16, 17]] = 0.0

        # ----------------------------------------------------------------
        # 접근 자세 (reset 및 Fabrics null-space용)
        # v9는 lerp를 쓰지 않으나 reset 초기화와 Fabrics attractor에서는 계속 필요.
        # ----------------------------------------------------------------
        self.hand_approach_pose   = to_torch(HAND_APPROACH_POSE,   device=self.device)  # (20,)
        self.hand_grasp_pose      = to_torch(HAND_GRASP_POSE,      device=self.device)  # (20,)
        self.hand_full_grip_pose  = to_torch(HAND_FULL_GRIP_POSE,  device=self.device)  # (20,)
        self.thumb_joint_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=self.device)
        self.thumb_curl_index = 1
        self.shape_anchor_mask = torch.zeros(NUM_HAND_DOF, device=self.device)
        self.shape_anchor_mask[[0, 4, 8, 12, 13, 14, 15, 16, 17, 18, 19]] = 1.0

        # ----------------------------------------------------------------
        # approach_pose 기준 관절 한계 재조정 — 반대 방향 휘어짐 방지
        # curl 양수 관절: lower = max(original, approach)  → approach보다 더 열리는 것 차단
        # curl 음수 관절 (thumb_2, 20D index 1): upper = min(original, approach) → approach보다 더 펴지는 것 차단
        # full_grip_pose는 imitation target이 아니라 "grasp에서 약간 더 닫히는" closure 상한으로만 사용
        # ----------------------------------------------------------------
        _approach = self.hand_approach_pose  # (20,)
        _new_lower = torch.max(self.hand_joint_lower_limits, _approach)
        _new_upper = self.hand_joint_upper_limits.clone()
        _new_lower[1] = self.hand_joint_lower_limits[1]                          # thumb_2: lower는 원래값 유지
        _new_upper[1] = torch.min(self.hand_joint_upper_limits[1], _approach[1]) # thumb_2: approach 이상으로 펴지는 것 차단
        _closing_up = self.hand_full_grip_pose > self.hand_grasp_pose
        _new_upper = torch.where(
            _closing_up,
            torch.minimum(_new_upper, self.hand_full_grip_pose),
            _new_upper,
        )
        _closing_down = self.hand_full_grip_pose < self.hand_grasp_pose
        _new_lower = torch.where(
            _closing_down,
            torch.maximum(_new_lower, self.hand_full_grip_pose),
            _new_lower,
        )
        self.hand_joint_lower_limits = _new_lower.contiguous()
        self.hand_joint_upper_limits = _new_upper.contiguous()

        # ----------------------------------------------------------------
        # 로봇 시작 자세 (arm: ARM_START_POSE, hand: HAND_GRASP_POSE)
        # HAND_GRASP_POSE에서 시작 → 20D delta 탐색 공간 축소 (파지 포즈 근처 미세 조정)
        # ----------------------------------------------------------------
        arm_start   = to_torch(ARM_START_POSE,   device=self.device)
        hand_start  = to_torch(HAND_GRASP_POSE,  device=self.device)
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

        # ----------------------------------------------------------------
        # 목표 위치
        # ----------------------------------------------------------------
        _goal_mid = [
            0.5 * (cfg.transport_goal_x_range[0] + cfg.transport_goal_x_range[1]),
            0.5 * (cfg.transport_goal_y_range[0] + cfg.transport_goal_y_range[1]),
            0.5 * (cfg.transport_goal_z_range[0] + cfg.transport_goal_z_range[1]),
        ]
        self.object_goal = (
            to_torch(_goal_mid, device=self.device)
            .unsqueeze(0).repeat(self.num_envs, 1)
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

        # ----------------------------------------------------------------
        # Pregrasp / Lift 버퍼
        # ----------------------------------------------------------------
        self.pregrasp_arm_pos_buf = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        self.lift_finger_pos_buf  = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        self.is_grasp_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_lift_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_stabilize_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_transport_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_post_grasp_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ----------------------------------------------------------------
        # Hand joint targets (per-joint delta 결과)
        # ----------------------------------------------------------------
        self.hand_joint_targets = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)

        # ----------------------------------------------------------------
        # 접촉 상태 버퍼
        # ----------------------------------------------------------------
        self.contact_force_xyz_raw   = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.contact_force_raw       = torch.zeros(self.num_envs, NUM_FINGERTIPS, device=self.device)
        self.binary_contact_buf    = torch.zeros(self.num_envs, NUM_FINGERTIPS, dtype=torch.bool, device=self.device)
        self.num_contacts_buf      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.distal_contact_force_raw  = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, device=self.device)
        self.distal_binary_contact_buf = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, dtype=torch.bool, device=self.device)

        self.middle_contact_force_raw  = torch.zeros(self.num_envs, NUM_MIDDLE_SENSORS, device=self.device)
        self.middle_binary_contact_buf = torch.zeros(self.num_envs, NUM_MIDDLE_SENSORS, dtype=torch.bool, device=self.device)

        self.palm_contact_force_raw  = torch.zeros(self.num_envs, device=self.device)
        self.palm_binary_contact_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.distal_contact_force_xyz = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, 3, device=self.device)
        self.middle_contact_force_xyz = torch.zeros(self.num_envs, NUM_MIDDLE_SENSORS, 3, device=self.device)
        self.palm_contact_force_xyz   = torch.zeros(self.num_envs, 3,                  device=self.device)

        self._prev_total_grip_force_buf = torch.zeros(self.num_envs, device=self.device)
        self._prev_tip_force_buf    = torch.zeros(self.num_envs, NUM_FINGERTIPS,     device=self.device)
        self._prev_distal_force_buf = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, device=self.device)
        self._prev_middle_force_buf = torch.zeros(self.num_envs, NUM_MIDDLE_SENSORS, device=self.device)
        self._prev_palm_force_buf   = torch.zeros(self.num_envs,                     device=self.device)
        self._prev_num_contacts_buf = torch.zeros(self.num_envs, device=self.device)
        self._prev_middle_contacts_buf = torch.zeros(self.num_envs, device=self.device)
        self._prev_cup_tilt_deg_buf = torch.zeros(self.num_envs, device=self.device)
        self._contact_persistence_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._lift_contact_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._lift_contact_ready_latched_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lift_started_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lift_start_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._full_grip_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._full_grip_ready_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._full_grip_ready_latched_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stabilize_started_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stabilize_start_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._transport_started_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._transport_start_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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
        self.episode_transport_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lift_success_latched_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stabilize_success_latched_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._phase_curriculum_stage = min(max(int(cfg.phase_curriculum_initial_stage), 0), 2)
        self._episode_curriculum_stage_buf = torch.full(
            (self.num_envs,),
            2 if not cfg.enable_phase_curriculum else self._phase_curriculum_stage,
            dtype=torch.long,
            device=self.device,
        )
        self._total_episodes: int = 0
        self._successful_episodes: int = 0

        # ----------------------------------------------------------------
        # Eval 로깅 (v9: 20D finger action)
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

        # cspace attractor: hand는 grasp pose 방향
        cspace_default = self.open_tesollo_fabric.default_config.clone()
        cspace_default[:, NUM_ARM_DOF:] = self.hand_grasp_pose.unsqueeze(0).expand(self.num_envs, -1)
        self.open_tesollo_fabric.default_config.copy_(cspace_default)

        # 초기 액션: 0 → palm pose workspace 중심, finger delta = 0
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

        self._distal_sensor = ContactSensor(self.cfg.distal_sensor_cfg)
        self.scene.sensors["distal_sensor"] = self._distal_sensor

        self._middle_sensor = ContactSensor(self.cfg.middle_sensor_cfg)
        self.scene.sensors["middle_sensor"] = self._middle_sensor

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

        self.open_tesollo_fabric = OpenArmTeoslloPoseFabric(
            self.num_envs, self.device, self.timestep,
            graph_capturable=False,
            use_hand_fabric=False,
        )
        num_joints = self.open_tesollo_fabric.num_joints

        self.open_tesollo_integrator = DisplacementIntegrator(self.open_tesollo_fabric)

        self.fabric_q   = self.robot_start_joint_pos.clone().contiguous()
        self.fabric_qd  = torch.zeros(self.num_envs, num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, num_joints, device=self.device)

        self.hand_pca_targets  = torch.zeros(self.num_envs, 5, device=self.device)
        self.palm_pose_targets = torch.zeros(self.num_envs, 6, device=self.device)
        self.fabric_damping_gain = self.cfg.fabrics_damping_gain * torch.ones(self.num_envs, 1, device=self.device)

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

        self._reset_pca     = torch.zeros(self._reset_chunk, 5, device=self.device)
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

        palm = torch.zeros(M, 6, device=self.device)
        palm[:, 0] = flat_x + self.cfg.pregrasp_offset_x
        palm[:, 1] = flat_y + self.cfg.pregrasp_offset_y
        palm[:, 2] = self.cfg.object_spawn_z + self.cfg.pregrasp_offset_z
        palm[:, 3] = math.radians(90.0)
        palm[:, 4] = math.radians(0.0)
        palm[:, 5] = math.radians(90.0)
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

    def _sample_transport_goals(self, n: int) -> torch.Tensor:
        def _axis(value_range: tuple[float, float]) -> torch.Tensor:
            low, high = float(value_range[0]), float(value_range[1])
            if low == high:
                return torch.full((n,), low, device=self.device)
            return torch.empty(n, device=self.device).uniform_(low, high)

        x = _axis(self.cfg.transport_goal_x_range)
        y = _axis(self.cfg.transport_goal_y_range)
        z = _axis(self.cfg.transport_goal_z_range)
        return torch.stack([x, y, z], dim=1)

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
        if success_source not in ("transport", "stage", "lift", "stabilize"):
            raise ValueError(
                "warm_state_success_source must be one of "
                "'transport', 'stage', 'lift', 'stabilize'"
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
            "object_goal_local": torch.empty(target_count, 3, dtype=torch.float32),
        }

    def _select_warm_state_export_success(
        self, env_ids: torch.Tensor, started_mask: torch.Tensor
    ) -> torch.Tensor:
        source = str(self.cfg.warm_state_success_source)
        if source == "transport":
            success = self.episode_transport_success_buf[env_ids]
        elif source == "stage":
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
        export["object_goal_local"][start:end] = self.object_goal[success_env_ids].detach().cpu()

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
            attrs["meta/bead_scale"] = 0.5
            attrs["meta/cup_base_mass"] = float(self.cfg.cup_base_mass)
            attrs["meta/palm_min_x"] = float(self.palm_mins[0])
            attrs["meta/palm_min_y"] = float(self.palm_mins[1])
            attrs["meta/palm_min_z"] = float(self.palm_mins[2])
            attrs["meta/palm_max_x"] = float(self.palm_maxs[0])
            attrs["meta/palm_max_y"] = float(self.palm_maxs[1])
            attrs["meta/palm_max_z"] = float(self.palm_maxs[2])
            attrs["meta/transport_goal_x_low"] = float(self.cfg.transport_goal_x_range[0])
            attrs["meta/transport_goal_x_high"] = float(self.cfg.transport_goal_x_range[1])
            attrs["meta/transport_goal_y_low"] = float(self.cfg.transport_goal_y_range[0])
            attrs["meta/transport_goal_y_high"] = float(self.cfg.transport_goal_y_range[1])
            attrs["meta/transport_goal_z_low"] = float(self.cfg.transport_goal_z_range[0])
            attrs["meta/transport_goal_z_high"] = float(self.cfg.transport_goal_z_range[1])

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

        if (
            self._phase_curriculum_stage == 1
            and len(self._stabilize_success_window) >= min_episodes
            and self._window_success_rate(self._stabilize_success_window)
            >= float(self.cfg.phase_curriculum_stabilize_success_threshold)
        ):
            self._phase_curriculum_stage = 2
            self._success_window.clear()

    def _spawn_dynamic_beads(self, env_mask: torch.Tensor) -> None:
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

        distal_xyz = torch.nan_to_num(self._distal_sensor.data.net_forces_w, nan=0.0, posinf=0.0, neginf=0.0)
        per_distal = distal_xyz.norm(dim=-1)
        self.distal_contact_force_xyz.copy_(distal_xyz)
        self.distal_contact_force_raw.copy_(per_distal)
        self.distal_binary_contact_buf.copy_(per_distal > CONTACT_FORCE_THRESHOLD)

        middle_xyz = torch.nan_to_num(self._middle_sensor.data.net_forces_w, nan=0.0, posinf=0.0, neginf=0.0)
        per_middle = middle_xyz.norm(dim=-1)
        self.middle_contact_force_xyz.copy_(middle_xyz)
        self.middle_contact_force_raw.copy_(per_middle)
        self.middle_binary_contact_buf.copy_(per_middle > CONTACT_FORCE_THRESHOLD)

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
        finger_action = actions[:, 6:26]  # (N, 20) ∈ [-1, 1] — per-joint delta

        # ---- Phase 판정 ----
        stabilize_curriculum_enabled = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if not self.cfg.enable_phase_curriculum
            else self._episode_curriculum_stage_buf >= 1
        )
        transport_curriculum_enabled = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if not self.cfg.enable_phase_curriculum
            else self._episode_curriculum_stage_buf >= 2
        )

        time_lift_ready = self.episode_length_buf >= LIFT_START_STEP
        just_entering_lift = time_lift_ready & (~self._lift_started_buf)
        if just_entering_lift.any():
            prev_finger_action = self._last_grasp_finger_action[just_entering_lift]
            self._eval_grip_at_lift[just_entering_lift] = prev_finger_action.abs().mean(dim=-1)
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
            stabilize_curriculum_enabled
            & self._lift_success_latched_buf
            & self._full_grip_ready_latched_buf
            & (~self._stabilize_started_buf)
        )
        if just_entering_stabilize.any():
            self._stabilize_start_step_buf[just_entering_stabilize] = (
                self.episode_length_buf[just_entering_stabilize]
            )
        self._stabilize_started_buf |= just_entering_stabilize

        just_entering_transport = (
            transport_curriculum_enabled
            & self._stabilize_success_latched_buf
            & self._full_grip_ready_buf
            & (~self._transport_started_buf)
        )
        if just_entering_transport.any():
            self._transport_start_step_buf[just_entering_transport] = (
                self.episode_length_buf[just_entering_transport]
            )
            current_palm = self.palm_pose_targets[just_entering_transport]
            current_object = self.object_pos[just_entering_transport]
            goal_delta = self.object_goal[just_entering_transport] - current_object
            transport_target = current_palm.clone()
            transport_target[:, :3] = transport_target[:, :3] + goal_delta
            transport_target[:, 3:] = current_palm[:, 3:]
            transport_target = torch.max(
                torch.min(transport_target, self.palm_maxs.unsqueeze(0)),
                self.palm_mins.unsqueeze(0),
            )
            self.transport_palm_start_pose_buf[just_entering_transport] = current_palm
            self.transport_palm_target_pose_buf[just_entering_transport] = transport_target
            self.transport_object_start_pos_buf[just_entering_transport] = current_object
        self._transport_started_buf |= just_entering_transport

        is_transport = self._transport_started_buf
        is_stabilize = self._stabilize_started_buf & (~is_transport)
        is_lift = self._lift_started_buf & (~self._stabilize_started_buf) & (~is_transport)
        is_grasp = ~self._lift_started_buf
        is_post_grasp = self._lift_started_buf
        self.is_grasp_phase.copy_(is_grasp)
        self.is_lift_phase.copy_(is_lift)
        self.is_stabilize_phase.copy_(is_stabilize)
        self.is_transport_phase.copy_(is_transport)
        self.is_post_grasp_phase.copy_(is_post_grasp)

        if self.cfg.dynamic_bead_spawn_enabled:
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
        delta = scale(palm_action, self.delta_mins, self.delta_maxs)
        grasp_palm_pose = self.pregrasp_palm_pose_buf + delta
        palm_mins = torch.minimum(self.palm_mins.unsqueeze(0), self.pregrasp_palm_pose_buf)
        palm_maxs = torch.maximum(self.palm_maxs.unsqueeze(0), self.pregrasp_palm_pose_buf)
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
        if self.demo_grasp_reset_bank is not None:
            lift_palm_pose = torch.lerp(
                self.lift_palm_start_pose_buf,
                self.demo_lift_palm_target_buf,
                lift_progress,
            )
            lift_palm_pose[:, 3:] = self.lift_palm_start_pose_buf[:, 3:]
        else:
            lift_palm_pose = self.lift_palm_start_pose_buf.clone()
            lift_palm_pose[:, 2] = (
                lift_palm_pose[:, 2]
                + float(self.cfg.lift_target_z_delta) * lift_progress.squeeze(1)
            )
        lift_palm_pose = torch.max(torch.min(lift_palm_pose, self.palm_maxs), self.palm_mins)

        transport_elapsed_steps = torch.where(
            self._transport_started_buf,
            (self.episode_length_buf - self._transport_start_step_buf).clamp(min=0),
            torch.zeros_like(self.episode_length_buf),
        )
        transport_progress = (
            transport_elapsed_steps.float() / TRANSPORT_PHASE_STEPS
        ).clamp(max=1.0).unsqueeze(1)
        transport_palm_pose = torch.lerp(
            self.transport_palm_start_pose_buf,
            self.transport_palm_target_pose_buf,
            transport_progress,
        )
        transport_palm_pose[:, 3:] = self.transport_palm_start_pose_buf[:, 3:]
        transport_palm_pose = torch.max(
            torch.min(transport_palm_pose, self.palm_maxs),
            self.palm_mins,
        )

        palm_pose = torch.where(
            is_transport.unsqueeze(1),
            transport_palm_pose,
            torch.where(is_post_grasp.unsqueeze(1), lift_palm_pose, grasp_palm_pose),
        )
        self.palm_pose_targets.copy_(palm_pose)
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

        # ---- Grasp phase finger delta target ----
        # grasp/lift 모두 "reference pose + bounded delta" semantics를 사용한다.
        current_finger_pos = self.robot.data.joint_pos[:, self.hand_dof_indices]
        grasp_delta_scale = resolve_grasp_delta_scale(
            default_scale=self.cfg.finger_delta_scale,
            adr_delta_scale=(
                self.grasp_adr.get_param("finger", "delta_scale")
                if self.grasp_adr is not None
                else None
            ),
        )
        hand_target = compute_grasp_finger_targets(
            current_pos=current_finger_pos,
            finger_action=finger_action,
            lower_limits=self.hand_joint_lower_limits,
            upper_limits=self.hand_joint_upper_limits,
            delta_scale=grasp_delta_scale,
            delta_mask=self.finger_delta_mask,
            thumb_curl_index=self.thumb_curl_index,
            thumb_downward_action_scale=self.cfg.thumb_curl_downward_action_scale,
            thumb_anchor_pose=self.hand_grasp_pose,
            thumb_curl_max_downward_delta=self.cfg.thumb_curl_max_downward_delta,
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
        # Grasp/Lift 모두 reference pose + bounded delta semantics 사용
        lift_finger_target = compute_lift_finger_targets(
            lift_reference_pos=self.lift_finger_pos_buf,
            finger_action=self.actions[:, 6:26],
            lower_limits=self.hand_joint_lower_limits,
            upper_limits=self.hand_joint_upper_limits,
            delta_scale=self.cfg.lift_finger_delta_scale,
            delta_mask=self.finger_delta_mask,
            thumb_curl_index=self.thumb_curl_index,
            thumb_downward_action_scale=self.cfg.thumb_curl_downward_action_scale,
            thumb_anchor_pose=self.hand_grasp_pose,
            thumb_curl_max_downward_delta=self.cfg.thumb_curl_max_downward_delta,
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
    # Observations: Actor 144D | Critic 174D
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
        cup_to_goal = self.object_goal - cup_pos_noisy

        # middle phalanx → cup 벡터 (FK 기반, sim2real 가능)
        middle3_pos_noisy = self.middle3_pos + torch.randn_like(self.middle3_pos) * σ_bp
        middle_to_cup = (
            middle3_pos_noisy - cup_pos_noisy.unsqueeze(1)
        ).view(self.num_envs, -1)   # (N, 15)

        last_actions = self.actions  # (N, 26)

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
            finger_joint_pos,       # 20
            finger_joint_vel,       # 20
            palm_center_pos,        # 3
            fingertip_pos_rel_palm, # 15
            palm_to_cup,            # 3
            cup_to_goal,            # 3
            cup_ang_vel,            # 3
            cup_rot,                # 4
            last_actions,           # 26
        ]
        if self.cfg.actor_observe_bead_mass:
            actor_obs_parts.append(self._bead_mass_normalized.unsqueeze(-1))  # 1
        actor_obs_parts.extend([
            tip_force_xyz_norm,     # 15
            middle_to_cup,          # 15
            phase_step_ratio,       # 1
            palm_binary_obs,        # 1
            palm_force_obs,         # 1
        ])
        actor_obs = torch.cat(actor_obs_parts, dim=-1)

        actor_obs = torch.nan_to_num(actor_obs, nan=0.0, posinf=5.0, neginf=-5.0)

        if actor_obs.shape[1] != self.cfg.num_observations:
            raise RuntimeError(
                f"[v11] Actor obs dim mismatch: {actor_obs.shape[1]} != {self.cfg.num_observations}"
            )

        # ==== Critic extra obs (30D) ====
        cup_height_delta = (
            cup_pos_clean[:, 2] - self.object_init_pos[:, 2]
        ).unsqueeze(1)

        distal_binary     = self.distal_binary_contact_buf.float()
        distal_force_norm = (self.distal_contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)
        middle_binary     = self.middle_binary_contact_buf.float()
        middle_force_norm = (self.middle_contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)

        tip_to_cup_dist = (
            fingertip_pos_clean - cup_pos_clean.unsqueeze(1)
        ).norm(dim=-1)
        fingertip_signed_dist = (tip_to_cup_dist - CUP_RADIUS_APPROX).unsqueeze(-1).squeeze(-1)

        middle_to_cup_clean = (
            self.middle3_pos - cup_pos_clean.unsqueeze(1)
        ).view(self.num_envs, -1)   # (N, 15)
        cup_to_goal_clean = self.object_goal - cup_pos_clean

        actor_obs_clean = torch.cat([
            arm_joint_pos_clean,
            arm_joint_vel_clean,
            finger_joint_pos_clean,
            finger_joint_vel_clean,
            palm_center_pos_clean,
            (fingertip_pos_clean - palm_center_pos_clean.unsqueeze(1)).view(self.num_envs, -1),
            cup_pos_clean - palm_center_pos_clean,
            cup_to_goal_clean,
            cup_ang_vel,
            cup_rot,
            last_actions,
            tip_force_xyz_norm,     # 15D
            middle_to_cup_clean,    # 15D
            phase_step_ratio,
            palm_binary_obs,        # 1D
            palm_force_obs,         # 1D
        ], dim=-1)   # 144D

        critic_obs = torch.cat([
            actor_obs_clean,        # 144
            self._bead_mass_normalized.unsqueeze(-1),  # 1
            cup_lin_vel,            # 3
            cup_height_delta,       # 1
            distal_binary,          # 5
            distal_force_norm,      # 5
            middle_binary,          # 5
            middle_force_norm,      # 5
            fingertip_signed_dist,  # 5
        ], dim=-1)   # 174D

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
        # ---- 접촉력 / 질량 ----
        total_grip_force = self.contact_force_raw.sum(dim=-1)
        effective_mass = (
            self.cfg.cup_base_mass
            + self._bead_mass_normalized * self.cfg.num_beads * self.cfg.bead_single_mass
        )
        mg = effective_mass * 9.81
        force_ratio = total_grip_force / (mg + 1e-4)
        force_delta_abs_for_ready = (total_grip_force - self._prev_total_grip_force_buf).abs()
        force_delta_ratio_abs_for_ready = force_delta_abs_for_ready / (mg + 1e-4)

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
        no_slip_gate = (slip_proxy <= self.cfg.slip_proxy_threshold).float()

        # ---- lift contact hold 추적 ----
        lift_contact_now = self.num_contacts_buf >= MIN_CONTACTS_FOR_SUCCESS
        lift_contact_phase = self.is_grasp_phase | self.is_lift_phase
        self._lift_contact_hold_count = torch.where(
            lift_contact_now & lift_contact_phase,
            self._lift_contact_hold_count + 1,
            torch.where(
                self._lift_contact_ready_latched_buf,
                self._lift_contact_hold_count,
                torch.zeros_like(self._lift_contact_hold_count),
            ),
        )
        lift_contact_ready_now = (
            self._lift_contact_hold_count >= int(self.cfg.lift_contact_hold_steps)
        )
        self._lift_contact_ready_latched_buf |= lift_contact_ready_now
        lift_contact_ready_gate = self._lift_contact_ready_latched_buf.float()

        # ---- contact persistence 추적 ----
        has_5_contact_bool = self.num_contacts_buf >= NUM_FINGERTIPS
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
        full_grip_ready_now = (
            self._lift_started_buf
            & has_5_contact_bool
            & middle_envelope_gate.bool()
            & no_slip_gate.bool()
            & upright_success_for_grip
            & (force_ratio >= self.cfg.lift_min_force_ratio)
            & (force_delta_ratio_abs_for_ready <= self.cfg.stabilize_force_delta_threshold)
            & (contact_delta_abs <= self.cfg.stabilize_contact_delta_threshold)
            & (middle_contact_delta_abs <= self.cfg.stabilize_contact_delta_threshold)
        )
        self._full_grip_hold_count = torch.where(
            full_grip_ready_now,
            self._full_grip_hold_count + 1,
            torch.zeros_like(self._full_grip_hold_count),
        )
        full_grip_ready_held = self._full_grip_hold_count >= int(self.cfg.full_grip_hold_steps)
        self._full_grip_ready_buf.copy_(full_grip_ready_now)
        self._full_grip_ready_latched_buf |= full_grip_ready_held
        self._grip_ready_latched_buf.copy_(self._full_grip_ready_latched_buf)
        grip_ready_now = full_grip_ready_now
        self._grip_ready_hold_count = torch.where(
            grip_ready_now,
            self._grip_ready_hold_count + 1,
            torch.zeros_like(self._grip_ready_hold_count),
        )
        full_grip_ready_gate = self._full_grip_ready_buf.float()
        grip_ready_gate = self._full_grip_ready_latched_buf.float()

        # ---- ADR ----
        _adr_min_contacts = (
            int(round(self.contact_adr.get_param("contact", "min_contacts")))
            if self.contact_adr is not None
            else 2
        )
        self._maybe_update_phase_curriculum()
        _ep_success_rate, _ep_success_window_len = self._curriculum_success_rate()
        if self.contact_adr is not None:
            self.contact_adr.maybe_increment(_ep_success_rate)
        if self.grasp_adr is not None:
            self.grasp_adr.maybe_increment(_ep_success_rate)

        # ---- prev buffer 갱신 ----
        self._prev_total_grip_force_buf.copy_(total_grip_force)
        self._prev_num_contacts_buf.copy_(self.num_contacts_buf.float())
        self._prev_middle_contacts_buf.copy_(self.middle_binary_contact_buf.float().sum(dim=-1))
        self._prev_cup_tilt_deg_buf.copy_(cup_tilt_deg)

        # ---- 상태 로깅 ----
        self.extras["f_ratio"] = force_ratio.mean()
        light_mask = self._bead_mass_normalized < 0.5
        heavy_mask = self._bead_mass_normalized > 0.5
        if light_mask.any() and heavy_mask.any():
            self.extras["f_ratio_delta"] = (
                force_ratio[heavy_mask].mean() - force_ratio[light_mask].mean()
            )
        self.extras["stat_num_contacts"] = self.num_contacts_buf.float().mean()
        self.extras["stat_middle_contacts"] = (
            self.middle_binary_contact_buf.float().sum(dim=-1).mean()
        )
        self.extras["stat_grip_ready_rate"] = grip_ready_gate.mean()
        self.extras["stat_lift_contact_ready_rate"] = lift_contact_ready_gate.mean()
        self.extras["stat_lift_started_rate"] = self._lift_started_buf.float().mean()
        self.extras["stat_full_grip_ready_rate"] = full_grip_ready_gate.mean()
        self.extras["stat_pre_lift_full_contact_rate"] = (
            full_tip_middle_contact & self.is_grasp_phase
        ).float().mean()
        self.extras["stat_slip_proxy"] = slip_proxy.mean()
        self.extras["stat_lift_contact_hold"] = self._lift_contact_hold_count.float().mean()
        self.extras["stat_contact_persistence"] = self._contact_persistence_buf.float().mean()
        self.extras["stat_grip_ready_hold"] = self._grip_ready_hold_count.float().mean()
        self.extras["stat_cup_uprightness"] = cup_z_world[:, 2].clamp(min=0.0).mean()
        self.extras["stat_cup_tilt_deg"] = cup_tilt_deg.mean()
        self.extras["stat_phase_grasp"] = self.is_grasp_phase.float().mean()
        self.extras["stat_phase_lift"] = self.is_lift_phase.float().mean()
        self.extras["stat_phase_stabilize"] = self.is_stabilize_phase.float().mean()
        self.extras["stat_phase_transport"] = self.is_transport_phase.float().mean()
        self.extras["stat_curriculum_stage"] = torch.tensor(
            float(2 if not self.cfg.enable_phase_curriculum else self._phase_curriculum_stage),
            device=self.device,
        )
        self.extras["stat_curriculum_window_len"] = torch.tensor(
            float(_ep_success_window_len), device=self.device
        )
        self.extras["stat_lift_success_rate"] = torch.tensor(
            self._window_success_rate(self._lift_success_window), device=self.device
        )
        self.extras["stat_stabilize_success_rate"] = torch.tensor(
            self._window_success_rate(self._stabilize_success_window), device=self.device
        )
        self.extras["stat_transport_success_rate"] = torch.tensor(
            self._window_success_rate(self._success_window), device=self.device
        )
        self.extras["stat_success_rate"] = torch.tensor(_ep_success_rate, device=self.device)
        self.extras["stat_dynamic_bead_added"] = (
            self._bead_count_current - self._bead_count_initial
        ).clamp_min(0).float().mean()
        self.extras["stat_bead_count_initial"] = self._bead_count_initial.float().mean()
        self.extras["stat_bead_count_current"] = self._bead_count_current.float().mean()
        self.extras["stat_cup_friction"] = self._cup_friction_static.mean()
        if self.contact_adr is not None:
            self.extras["adr_min_contacts"] = torch.tensor(
                float(_adr_min_contacts), device=self.device
            )
        if self.grasp_adr is not None:
            self.extras["adr_difficulty_progress"] = torch.tensor(
                self.grasp_adr.progress, device=self.device
            )
        _bin_defs = [
            ("0b",  self._bead_mass_normalized < 0.17),
            ("10b", (self._bead_mass_normalized >= 0.17) & (self._bead_mass_normalized < 0.50)),
            ("20b", (self._bead_mass_normalized >= 0.50) & (self._bead_mass_normalized < 0.84)),
            ("30b", self._bead_mass_normalized >= 0.84),
        ]
        for _lvl, (_tag, _mask) in enumerate(_bin_defs):
            self.extras[f"bin_{_tag}_sr"] = torch.tensor(
                self._successful_episodes_bin[_lvl]
                / max(self._total_episodes_bin[_lvl], 1),
                device=self.device,
            )
            if _mask.any():
                self.extras[f"bin_{_tag}_f_ratio"] = force_ratio[_mask].mean()
                self.extras[f"bin_{_tag}_contacts"] = self.num_contacts_buf[_mask].float().mean()
            else:
                _zero = torch.zeros((), device=self.device)
                self.extras[f"bin_{_tag}_f_ratio"] = _zero
                self.extras[f"bin_{_tag}_contacts"] = _zero

        # ================================================================
        # 7-term Mass-Adaptive Enveloping Grip Reward
        # ================================================================

        # ---- r_height: exp(-α_h * (z_cup - z*)²), lift phase에서만 ----
        cup_height_delta = self.object_pos[:, 2] - self.object_init_pos[:, 2]
        r_height = torch.exp(
            -self.cfg.r_height_sharpness * (cup_height_delta - self.cfg.lift_target_z_delta) ** 2
        ) * self._lift_started_buf.float()

        # ---- r_ori: exp(-α_R * tilt_rad²) ----
        tilt_rad = torch.acos(cup_z_world[:, 2].clamp(-1.0, 1.0))
        r_ori = torch.exp(-self.cfg.r_ori_sharpness * tilt_rad ** 2)

        # ---- r_slip: -w_s * Σᵢ∈C 1_{cᵢ} * ||v_rel,i||²  (HTML exact formula) ----
        # v_rel,i = (v_cup + ω_cup × r_i) - v_finger_i
        cup_lin_vel_w = self.cup.data.root_lin_vel_w          # (N, 3)
        cup_ang_vel_w = self.cup.data.root_ang_vel_w          # (N, 3)

        def _tangential_sq(v_rel: torch.Tensor, f_xyz: torch.Tensor) -> torch.Tensor:
            """||v_t||² = ||v_rel||² - (v_rel·n̂)²,  n̂ = f_xyz/||f_xyz||"""
            n_hat = f_xyz / (f_xyz.norm(dim=-1, keepdim=True) + 1e-6)
            v_dot_n = (v_rel * n_hat).sum(dim=-1)
            return (v_rel.pow(2).sum(dim=-1) - v_dot_n ** 2).clamp(min=0.0)

        # tip
        r_tip = self.fingertip_pos - self.object_pos.unsqueeze(1)   # (N, 5, 3)
        cup_surf_vel_tip = (
            cup_lin_vel_w.unsqueeze(1)
            + torch.linalg.cross(cup_ang_vel_w.unsqueeze(1).expand(-1, NUM_FINGERTIPS, -1), r_tip)
        )
        finger_tip_vel = self.robot.data.body_lin_vel_w[:, self.fingertip_body_indices, :]
        v_rel_tip = cup_surf_vel_tip - finger_tip_vel                # (N, 5, 3)
        tip_slip = (self.binary_contact_buf.float()
                    * _tangential_sq(v_rel_tip, self.contact_force_xyz_raw)).sum(dim=-1)

        # distal
        if len(self.distal4_body_indices) == NUM_FINGERTIPS:
            r_distal = self.distal4_pos - self.object_pos.unsqueeze(1)
            cup_surf_vel_distal = (
                cup_lin_vel_w.unsqueeze(1)
                + torch.linalg.cross(cup_ang_vel_w.unsqueeze(1).expand(-1, NUM_FINGERTIPS, -1), r_distal)
            )
            finger_distal_vel = self.robot.data.body_lin_vel_w[:, self.distal4_body_indices, :]
            v_rel_distal = cup_surf_vel_distal - finger_distal_vel
            distal_slip = (self.distal_binary_contact_buf.float()
                           * _tangential_sq(v_rel_distal, self.distal_contact_force_xyz)).sum(dim=-1)
        else:
            distal_slip = torch.zeros(self.num_envs, device=self.device)

        # middle
        if len(self.middle3_body_indices) == NUM_FINGERTIPS:
            r_middle = self.middle3_pos - self.object_pos.unsqueeze(1)
            cup_surf_vel_middle = (
                cup_lin_vel_w.unsqueeze(1)
                + torch.linalg.cross(cup_ang_vel_w.unsqueeze(1).expand(-1, NUM_FINGERTIPS, -1), r_middle)
            )
            finger_middle_vel = self.robot.data.body_lin_vel_w[:, self.middle3_body_indices, :]
            v_rel_middle = cup_surf_vel_middle - finger_middle_vel
            middle_slip = (self.middle_binary_contact_buf.float()
                           * _tangential_sq(v_rel_middle, self.middle_contact_force_xyz)).sum(dim=-1)
        else:
            middle_slip = torch.zeros(self.num_envs, device=self.device)

        # palm
        if self.palm_body_index >= 0:
            r_palm_vec = self.palm_center_pos - self.object_pos
            cup_surf_vel_palm = cup_lin_vel_w + torch.linalg.cross(cup_ang_vel_w, r_palm_vec)
            finger_palm_vel = self.robot.data.body_lin_vel_w[:, self.palm_body_index, :]
            v_rel_palm = cup_surf_vel_palm - finger_palm_vel
            palm_slip = (
                self.palm_binary_contact_buf.float()
                * _tangential_sq(v_rel_palm, self.palm_contact_force_xyz)
            )
        else:
            palm_slip = torch.zeros(self.num_envs, device=self.device)

        r_slip = -self.cfg.r_slip_weight * (tip_slip + distal_slip + middle_slip + palm_slip)

        # ---- r_margin: -w_m * [max(0, s·mg - μ·ΣFn)]², lift phase에서만 ----
        total_fn = (
            (self.binary_contact_buf.float() * self.contact_force_raw).sum(dim=-1)
            + (self.distal_binary_contact_buf.float() * self.distal_contact_force_raw).sum(dim=-1)
            + (self.middle_binary_contact_buf.float() * self.middle_contact_force_raw).sum(dim=-1)
            + self.palm_binary_contact_buf.float() * self.palm_contact_force_raw
        )
        cup_az = self.cup.data.body_com_acc_w[:, 0, 2].clamp(min=0.0)
        friction_support = self._cup_friction_static * total_fn
        required_support = self.cfg.friction_safety_factor * effective_mass * (9.81 + cup_az)
        margin_deficit = torch.relu(required_support - friction_support)
        r_margin = (
            -self.cfg.r_margin_weight * margin_deficit ** 2
            * self._lift_started_buf.float()
        )

        # ---- r_contact: w_tip·Σtip + w_phalanx·Σphalanx ----
        tip_count     = self.binary_contact_buf.float().sum(dim=-1)
        phalanx_count = (
            self.distal_binary_contact_buf.float().sum(dim=-1)
            + self.middle_binary_contact_buf.float().sum(dim=-1)
        )
        r_contact = (
            self.cfg.r_contact_tip_weight * tip_count
            + self.cfg.r_contact_phalanx_weight * phalanx_count
            + self.cfg.r_contact_palm_weight * self.palm_binary_contact_buf.float()
        )

        # ---- r_force: -w_f · Σ fn²  (과도 grip force 억제) ----
        tip_fn_sq    = (self.binary_contact_buf.float()        * self.contact_force_raw ** 2).sum(dim=-1)
        distal_fn_sq = (self.distal_binary_contact_buf.float() * self.distal_contact_force_raw ** 2).sum(dim=-1)
        middle_fn_sq = (self.middle_binary_contact_buf.float() * self.middle_contact_force_raw ** 2).sum(dim=-1)
        palm_fn_sq   = self.palm_binary_contact_buf.float() * self.palm_contact_force_raw ** 2
        r_force = -self.cfg.r_force_weight * (tip_fn_sq + distal_fn_sq + middle_fn_sq + palm_fn_sq)

        # ---- r_deltaf: -w_Δf · Σ (fn,t - fn,t-1)²  (급격한 force 변화 억제) ----
        tip_delta_sq = (
            self.binary_contact_buf.float()
            * (self.contact_force_raw    - self._prev_tip_force_buf) ** 2
        ).sum(dim=-1)
        distal_delta_sq = (
            self.distal_binary_contact_buf.float()
            * (self.distal_contact_force_raw - self._prev_distal_force_buf) ** 2
        ).sum(dim=-1)
        middle_delta_sq = (
            self.middle_binary_contact_buf.float()
            * (self.middle_contact_force_raw - self._prev_middle_force_buf) ** 2
        ).sum(dim=-1)
        palm_delta_sq = (
            self.palm_binary_contact_buf.float()
            * (self.palm_contact_force_raw - self._prev_palm_force_buf) ** 2
        )
        r_deltaf = -self.cfg.r_deltaf_weight * (tip_delta_sq + distal_delta_sq + middle_delta_sq + palm_delta_sq)

        # ---- prev per-link force 갱신 ----
        self._prev_tip_force_buf.copy_(self.contact_force_raw)
        self._prev_distal_force_buf.copy_(self.distal_contact_force_raw)
        self._prev_middle_force_buf.copy_(self.middle_contact_force_raw)
        self._prev_palm_force_buf.copy_(self.palm_contact_force_raw)

        # ---- 리워드 로깅 ----
        self.extras["rew_height"]  = r_height.mean()
        self.extras["rew_ori"]     = r_ori.mean()
        self.extras["rew_slip"]    = r_slip.mean()
        self.extras["rew_margin"]  = r_margin.mean()
        self.extras["rew_contact"] = r_contact.mean()
        self.extras["rew_force"]   = r_force.mean()
        self.extras["rew_deltaf"]  = r_deltaf.mean()
        self.extras["stat_friction_support"] = friction_support.mean()
        self.extras["stat_required_support"] = required_support.mean()
        self.extras["stat_margin_deficit"]   = margin_deficit.mean()
        self.extras["stat_cup_height_delta"] = cup_height_delta.mean()
        self.extras["stat_tilt_deg"]         = torch.rad2deg(tilt_rad).mean()

        reward = (
            self.cfg.r_height_weight  * r_height
            + self.cfg.r_ori_weight   * r_ori
            + r_slip
            + r_margin
            + r_contact
            + r_force
            + r_deltaf
        )
        return reward

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
        in_transport = self._transport_started_buf
        if self.cfg.enable_phase_curriculum:
            stage0_lift_only = self._episode_curriculum_stage_buf <= 0
            stage1_stabilize_only = self._episode_curriculum_stage_buf == 1
            stage2_transport = self._episode_curriculum_stage_buf >= 2
        else:
            stage0_lift_only = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            stage1_stabilize_only = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            stage2_transport = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        lifted  = self.object_pos[:, 2] > (self.object_init_pos[:, 2] + self.cfg.lift_success_height)
        _lift_min_contacts = MIN_CONTACTS_FOR_SUCCESS
        lift_grasped = self.num_contacts_buf >= _lift_min_contacts
        contact_grasped = self.num_contacts_buf >= NUM_FINGERTIPS
        middle_grasped = compute_middle_contact_gate(
            self.middle_binary_contact_buf,
            self.cfg.min_middle_contacts_for_success,
        )
        upright_success = compute_upright_success_mask(
            cup_z_world[:, 2],
            self.cfg.success_upright_max_deg,
        )
        total_grip_force = self.contact_force_raw.sum(dim=-1)
        effective_mass = (
            self.cfg.cup_base_mass
            + self._bead_mass_normalized * self.cfg.num_beads * self.cfg.bead_single_mass
        )
        mg = effective_mass * 9.81
        force_ratio = total_grip_force / (mg + 1e-4)
        force_delta_abs = (total_grip_force - self._prev_total_grip_force_buf).abs()
        force_delta_ratio_abs = force_delta_abs / (mg + 1e-4)
        contact_delta_abs = (self.num_contacts_buf.float() - self._prev_num_contacts_buf).abs()
        middle_contact_delta_abs = (
            self.middle_binary_contact_buf.float().sum(dim=-1) - self._prev_middle_contacts_buf
        ).abs()
        cup_horiz_vel = torch.nan_to_num(self.cup.data.root_lin_vel_w[:, :2].norm(dim=-1), nan=0.0)
        cup_ang_speed = torch.nan_to_num(self.cup.data.root_ang_vel_w.norm(dim=-1), nan=0.0)
        no_slip = (
            (cup_horiz_vel <= self.cfg.stabilize_cup_lin_vel_threshold)
            & (cup_ang_speed <= self.cfg.stabilize_cup_ang_vel_threshold)
            & (contact_delta_abs <= self.cfg.stabilize_contact_delta_threshold)
            & (middle_contact_delta_abs <= self.cfg.stabilize_contact_delta_threshold)
        )
        force_stable = force_delta_ratio_abs <= self.cfg.stabilize_force_delta_threshold
        full_grip_ready_now = (
            in_or_past_lift
            & contact_grasped
            & middle_grasped
            & no_slip
            & upright_success
            & (force_ratio >= self.cfg.lift_min_force_ratio)
            & force_stable
        )
        self._full_grip_ready_buf.copy_(full_grip_ready_now)
        stable_grasped = full_grip_ready_now
        lift_success_now = in_or_past_lift & lifted & lift_grasped & upright_success
        goal_dist = (self.object_pos - self.object_goal).norm(dim=-1)
        transport_success = compute_transport_success_mask(
            goal_dist=goal_dist,
            upright_success=upright_success,
            contact_grasped=contact_grasped,
            middle_grasped=middle_grasped,
            no_slip=no_slip,
            goal_dist_threshold=self.cfg.transport_goal_dist_threshold,
        ) & full_grip_ready_now
        success_now = in_transport & lifted & transport_success
        stabilize_eval_delay = (
            max(int(self.cfg.dynamic_bead_spawn_step) - STABILIZE_START_STEP, 0)
            if self.cfg.dynamic_bead_spawn_enabled
            else 0
        )
        stabilize_success_now = (
            self._stabilize_started_buf
            & (stabilize_elapsed_steps >= stabilize_eval_delay)
            & lifted
            & stable_grasped
        )
        self._lift_success_latched_buf |= lift_success_now
        self._stabilize_success_latched_buf |= stabilize_success_now

        stage_success_now = torch.where(
            stage0_lift_only,
            lift_success_now,
            torch.where(stage1_stabilize_only, stabilize_success_now, success_now),
        )
        self.success_flag.copy_(stage_success_now)
        self.episode_success_buf |= stage_success_now
        self.episode_lift_success_buf |= lift_success_now
        self.episode_stabilize_success_buf |= stabilize_success_now
        self.episode_transport_success_buf |= success_now

        self._success_hold_count = torch.where(
            stage_success_now,
            self._success_hold_count + 1,
            torch.zeros_like(self._success_hold_count),
        )
        hold_steps = torch.where(
            stage2_transport,
            torch.full_like(self._success_hold_count, int(self.cfg.transport_success_hold_steps)),
            torch.full_like(self._success_hold_count, int(self.cfg.success_hold_steps)),
        )
        success_held = self._success_hold_count >= hold_steps

        if self.cfg.terminate_on_lift_failure:
            lift_wait_failed = (
                (self.episode_length_buf >= STABILIZE_START_STEP)
                & (~self._lift_started_buf)
            )
            lift_failed = (
                self._lift_started_buf
                & (lift_elapsed_steps >= LIFT_PHASE_STEPS)
                & (~self._lift_success_latched_buf)
            ) | lift_wait_failed
        else:
            lift_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        curriculum_lift_horizon = (
            stage0_lift_only
            & self._lift_started_buf
            & (lift_elapsed_steps >= LIFT_PHASE_STEPS)
        )
        curriculum_stabilize_horizon = (
            stage1_stabilize_only
            & self._stabilize_started_buf
            & (stabilize_elapsed_steps >= STABILIZE_PHASE_STEPS)
        )

        terminated = out_x | out_y | fallen | tipped | success_held | lift_failed
        truncated  = (
            (self.episode_length_buf >= self.max_episode_length - 1)
            | curriculum_lift_horizon
            | curriculum_stabilize_horizon
        )

        self.extras["stat_obj_z"] = self.object_pos[:, 2].mean()
        self.extras["stat_lift_success_now"] = lift_success_now.float().mean()
        self.extras["stat_stabilize_success_now"] = stabilize_success_now.float().mean()
        self.extras["stat_transport_success_now"] = success_now.float().mean()
        self.extras["stat_curriculum_success_now"] = stage_success_now.float().mean()

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
            self._successful_episodes += int((self.episode_success_buf[env_ids] & had_started).sum().item())

        # 6.2 & 6.3: moving window + per-bin 업데이트
        for i, env_id in enumerate(env_ids):
            if not bool(had_started[i].item()):
                continue
            stage_val = (
                int(self._episode_curriculum_stage_buf[env_id].item())
                if self.cfg.enable_phase_curriculum
                else 2
            )
            lift_success_val = int(bool(self.episode_lift_success_buf[env_id].item()))
            stabilize_success_val = int(bool(self.episode_stabilize_success_buf[env_id].item()))
            transport_success_val = int(bool(self.episode_transport_success_buf[env_id].item()))
            success_val = int(bool(self.episode_success_buf[env_id].item()))
            self._lift_success_window.append(lift_success_val)
            if stage_val >= 1:
                self._stabilize_success_window.append(stabilize_success_val)
            if stage_val >= 2:
                self._success_window.append(transport_success_val)
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

            # curl joint indices per finger (rj_dg_X_2 = index 4*(f-1)+1 for f=1..4, pinky _1 = 16)
            # finger layout: [f1_1,f1_2,f1_3,f1_4, f2_1,..., f5_1,f5_2,f5_3,f5_4]
            def _curl_idx(finger: int) -> int:  # finger 0-indexed
                return finger * 4 + 1  # _2 joint

            self._eval_records.append({
                "bead_count": bead_count,
                "bead_mass": self._bead_mass_normalized[env_id].item(),
                "bead_count_initial": bead_initial_count,
                "dynamic_bead_added": dynamic_bead_added,
                "cup_friction_static": self._cup_friction_static[env_id].item(),
                "goal_x": self.object_goal[env_id, 0].item(),
                "goal_y": self.object_goal[env_id, 1].item(),
                "goal_z": self.object_goal[env_id, 2].item(),
                "curriculum_stage": int(self._episode_curriculum_stage_buf[env_id].item()),
                "lift_success": self.episode_lift_success_buf[env_id].item(),
                "stabilize_success": self.episode_stabilize_success_buf[env_id].item(),
                "transport_success": self.episode_transport_success_buf[env_id].item(),
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
                "success":   self.episode_success_buf[env_id].item(),
            })

        self._maybe_export_warm_states(env_ids_tensor, had_started)

        self.episode_success_buf[env_ids] = False
        self.episode_lift_success_buf[env_ids] = False
        self.episode_stabilize_success_buf[env_ids] = False
        self.episode_transport_success_buf[env_ids] = False
        self._episode_curriculum_stage_buf[env_ids] = (
            2 if not self.cfg.enable_phase_curriculum else self._phase_curriculum_stage
        )
        self.object_goal[env_ids] = self._sample_transport_goals(n)

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
            pregrasp_pos = obj_pos_local + self.pregrasp_offset.unsqueeze(0) + noise

            pregrasp_palm_pose = torch.zeros(n, 6, device=self.device)
            pregrasp_palm_pose[:, :3] = pregrasp_pos
            pregrasp_palm_pose[:, 3] = math.radians(90.0)
            pregrasp_palm_pose[:, 4] = math.radians(0.0)
            pregrasp_palm_pose[:, 5] = math.radians(90.0)
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

        if self.cfg.enable_phase_curriculum:
            transport_disabled = self._episode_curriculum_stage_buf[env_ids] < 2
            if transport_disabled.any():
                self.object_goal[env_ids_tensor[transport_disabled]] = obj_pos_local[transport_disabled]

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
        self.transport_palm_start_pose_buf[env_ids] = pregrasp_palm_pose
        self.transport_palm_target_pose_buf[env_ids] = pregrasp_palm_pose
        self.transport_object_start_pos_buf[env_ids] = obj_pos_local

        self.open_tesollo_fabric.default_config[env_ids, :NUM_ARM_DOF] = q_pregrasp[:, :NUM_ARM_DOF]
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
        if self.cfg.dynamic_bead_spawn_enabled:
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
            if active.any():
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
        self._prev_tip_force_buf[env_ids]         = 0.0
        self._prev_distal_force_buf[env_ids]      = 0.0
        self._prev_middle_force_buf[env_ids]      = 0.0
        self._prev_palm_force_buf[env_ids]        = 0.0
        self._prev_num_contacts_buf[env_ids] = 0.0
        self._prev_middle_contacts_buf[env_ids] = 0.0
        self._prev_cup_tilt_deg_buf[env_ids] = 0.0
        self._contact_persistence_buf[env_ids] = 0
        self._lift_contact_hold_count[env_ids] = 0
        self._lift_contact_ready_latched_buf[env_ids] = False
        self._lift_started_buf[env_ids] = False
        self._lift_start_step_buf[env_ids] = 0
        self._full_grip_hold_count[env_ids] = 0
        self._full_grip_ready_buf[env_ids] = False
        self._full_grip_ready_latched_buf[env_ids] = False
        self._stabilize_started_buf[env_ids] = False
        self._stabilize_start_step_buf[env_ids] = 0
        self._transport_started_buf[env_ids] = False
        self._transport_start_step_buf[env_ids] = 0
        self._grip_ready_hold_count[env_ids] = 0
        self._grip_ready_latched_buf[env_ids] = False
        self._lift_success_latched_buf[env_ids] = False
        self._stabilize_success_latched_buf[env_ids] = False
        self.success_flag[env_ids] = False
        self._success_hold_count[env_ids] = 0
        self.is_grasp_phase[env_ids] = False
        self.is_lift_phase[env_ids] = False
        self.is_stabilize_phase[env_ids] = False
        self.is_transport_phase[env_ids] = False
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

        # actions 리셋: delta=0 = 현재 자세 유지
        self.actions[env_ids, :6] = 0.0
        self.actions[env_ids, 6:] = 0.0
        self.prev_actions[env_ids, :6] = 0.0
        self.prev_actions[env_ids, 6:] = 0.0
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

            self.robot.write_joint_stiffness_to_sim(
                default_stiffness * self._uniform_scale(shape, self.cfg.real2sim_stiffness_scale_range),
                joint_ids=joint_ids,
                env_ids=env_ids_tensor,
            )
            self.robot.write_joint_damping_to_sim(
                default_damping * self._uniform_scale(shape, self.cfg.real2sim_damping_scale_range),
                joint_ids=joint_ids,
                env_ids=env_ids_tensor,
            )
            if torch.any(default_friction != 0.0):
                self.robot.write_joint_friction_coefficient_to_sim(
                    default_friction * self._uniform_scale(shape, self.cfg.real2sim_friction_scale_range),
                    joint_ids=joint_ids,
                    env_ids=env_ids_tensor,
                )
