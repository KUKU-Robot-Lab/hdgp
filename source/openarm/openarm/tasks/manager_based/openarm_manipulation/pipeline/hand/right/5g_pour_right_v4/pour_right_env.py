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

"""환경 클래스: 5g_pour_right_v7

v7: Fabrics 팔 학습 + per-finger lerp 5D + Contact sensor 없는 FK 기반 근접도 리워드

핵심 개선 (v1/v6 대비):
  - v1 문제: fabric_q/qd obs → sim2real 불가, palm_dist 기반 자동 닫힘 → 충돌 충격
  - v6 문제: 팔 고정 → cup 위치 오차 대응 불가, per-finger 5D 협응 학습 부족

Action (11D):
  [0:6]  6D palm pose → Fabrics IK → arm 7 DOF (학습, cup 위치 오차 대응)
  [6:11] 5D per-finger lerp: -1 → HAND_APPROACH_POSE, +1 → HAND_GRASP_POSE

Episode (10s @ 60Hz):
  Grasp phase (0~479): Fabrics arm + per-finger 정책
  Lift  phase (480~599): scripted arm prelift + frozen hand
"""

from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path
from collections.abc import Sequence

import torch
import torch.nn as nn

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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_angle_axis, quat_mul

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

from .pour_right_env_cfg import PourRightEnvCfg
from .trajectory_buffer import TrajectoryCapture, SuccessTrajectoryBuffer
from .pour_right_constants import (
    NUM_ARM_DOF,
    NUM_HAND_DOF,
    NUM_FINGER_ACTION,
    NUM_FINGERTIPS,
    NUM_OBSERVATIONS,
    NUM_DISTAL_SENSORS,
    NUM_MIDDLE_SENSORS,
    NUM_CRITIC_OBSERVATIONS,
    EPISODE_STEPS,
    CONTACT_FORCE_THRESHOLD,
    CONTACT_FORCE_MAX,
    MIN_CONTACTS_FOR_SUCCESS,
    PREGRASP_FABRICS_STEPS,
    CUP_RADIUS_APPROX,
    ARM_START_POSE,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
)
from .pour_adr import PourADR
from .pour_adr import PourADR as GraspADR
from .pour_right_preset import (
    BEAD_SPAWN_POS_SOURCE_CUP_B,
    BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ,
    LEFT_ARM_REST_JOINT_POS,
    LEFT_TARGET_CUP_ATTACH_FRAME_NAME,
    LEFT_TARGET_CUP_ATTACH_POS_B,
    LEFT_TARGET_CUP_ATTACH_QUAT_WXYZ_B,
    RIGHT_ACTUATED_JOINT_NAMES,
    HAND_APPROACH_POSE,
    HAND_GRASP_POSE,
    HAND_FULL_GRIP_POSE,
    OBJECT_GOAL_POS,
)
from .pour_right_utils import scale, to_torch




class _WarmstartPolicy(nn.Module):
    def __init__(self, checkpoint_path: str, device: str):
        super().__init__()
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)["model"]
        self.register_buffer("obs_mean", state["running_mean_std.running_mean"].float())
        self.register_buffer("obs_var", state["running_mean_std.running_var"].float())
        self.register_buffer("actor_l1_w", state["a2c_network.actor_mlp.0.weight"].float())
        self.register_buffer("actor_l1_b", state["a2c_network.actor_mlp.0.bias"].float())
        self.register_buffer("actor_l2_w", state["a2c_network.actor_mlp.2.weight"].float())
        self.register_buffer("actor_l2_b", state["a2c_network.actor_mlp.2.bias"].float())
        self.register_buffer("actor_l3_w", state["a2c_network.actor_mlp.4.weight"].float())
        self.register_buffer("actor_l3_b", state["a2c_network.actor_mlp.4.bias"].float())
        self.register_buffer("mu_w", state["a2c_network.mu.weight"].float())
        self.register_buffer("mu_b", state["a2c_network.mu.bias"].float())
        self.obs_dim = int(self.obs_mean.shape[0])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5)
        x = torch.clamp(x, -5.0, 5.0)
        x = torch.nn.functional.elu(torch.nn.functional.linear(x, self.actor_l1_w, self.actor_l1_b))
        x = torch.nn.functional.elu(torch.nn.functional.linear(x, self.actor_l2_w, self.actor_l2_b))
        x = torch.nn.functional.elu(torch.nn.functional.linear(x, self.actor_l3_w, self.actor_l3_b))
        return torch.nn.functional.linear(x, self.mu_w, self.mu_b)


class PourRightEnv(DirectRLEnv):
    """OpenArm+Teosllo 오른손 물붓기 환경 v1.

    Warmstart(v7 grasp policy)로 컵이 이미 파지·들린 상태로 시작.
    Policy는 transport → tilt → pour 모션을 학습.

    Action: 11D
      [0:6]  palm pose (x,y,z,ez,ey,ex), 정규화 [-1,1] → Fabrics IK
      [6:11] per-finger lerp (freeze_grasp=True → 항상 grasp_hold 유지)

    Episode (6s @ 60Hz = 360 steps):
      Pour phase (step 0~359): Fabrics arm policy + frozen hand
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

    @staticmethod
    def _compose_world_delta_quat_xyzw(base_quat_xyzw: torch.Tensor, delta_rotvec: torch.Tensor) -> torch.Tensor:
        """Apply a world-frame axis-angle delta to a base quaternion."""
        delta_angle = torch.norm(delta_rotvec, dim=-1)
        delta_axis = torch.zeros_like(delta_rotvec)
        nonzero = delta_angle > 1e-8
        if torch.any(nonzero):
            delta_axis[nonzero] = delta_rotvec[nonzero] / delta_angle[nonzero].unsqueeze(-1)
        delta_axis[~nonzero, 0] = 1.0
        delta_quat_wxyz = quat_from_angle_axis(delta_angle, delta_axis)
        base_quat_wxyz = base_quat_xyzw[:, [3, 0, 1, 2]]
        # World-frame delta uses left-multiplication.
        target_quat_wxyz = quat_mul(delta_quat_wxyz, base_quat_wxyz)
        target_quat_xyzw = target_quat_wxyz[:, [1, 2, 3, 0]]
        return torch.nn.functional.normalize(target_quat_xyzw, dim=-1)

    @staticmethod
    def _safe_normalize(vec: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(vec, dim=-1, keepdim=True)
        fallback_norm = torch.norm(fallback, dim=-1, keepdim=True).clamp(min=1e-6)
        fallback_unit = fallback / fallback_norm
        return torch.where(norm > 1e-6, vec / norm.clamp(min=1e-6), fallback_unit)

    def _compute_dynamic_source_pour_point_w(
        self,
        cup_pos_w: torch.Tensor,
        cup_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        """Return the downhill rim edge instead of the mouth-center point.

        The old mouth-center point matches the cup axis but not the actual bead exit point.
        For a tilted cylindrical cup, the relevant pour point is the lowest rim edge.
        """
        n = cup_pos_w.shape[0]
        mouth_center_w = cup_pos_w + quat_apply(
            cup_quat_w,
            self._source_cup_pour_point_pos_b.unsqueeze(0).expand(n, -1),
        )
        cup_up_axis_w = quat_apply(
            cup_quat_w,
            self._source_cup_up_axis_b.unsqueeze(0).expand(n, -1),
        )
        cup_pour_axis_w = quat_apply(
            cup_quat_w,
            self._source_cup_pour_axis_b.unsqueeze(0).expand(n, -1),
        )
        gravity_down = cup_up_axis_w.new_tensor([0.0, 0.0, -1.0]).expand_as(cup_up_axis_w)
        downhill_rim_dir = gravity_down - (gravity_down * cup_up_axis_w).sum(dim=-1, keepdim=True) * cup_up_axis_w
        downhill_rim_dir = self._safe_normalize(downhill_rim_dir, cup_pour_axis_w)
        return mouth_center_w + self.cfg.source_inner_radius * downhill_rim_dir

    def _build_cup_local_tilt_rotvec(self, delta_local: torch.Tensor) -> torch.Tensor:
        """Map local tilt commands to a world-frame rotvec.

        delta_local[:, 0]: spin around current cup up-axis
        delta_local[:, 1]: tilt toward target opening
        delta_local[:, 2]: tilt in the orthogonal in-plane direction
        """
        n = delta_local.shape[0]
        cup_quat_w = self.cup.data.root_quat_w
        left_target_pos_w = self.left_target_cup.data.root_pos_w
        left_target_quat_w = self.left_target_cup.data.root_quat_w

        source_pour_point_w = self._compute_dynamic_source_pour_point_w(self.cup.data.root_pos_w, cup_quat_w)
        target_opening_w = left_target_pos_w + quat_apply(
            left_target_quat_w,
            self._target_cup_opening_pos_b.unsqueeze(0).expand(n, -1),
        )
        cup_up_axis_w = quat_apply(
            cup_quat_w,
            self._source_cup_up_axis_b.unsqueeze(0).expand(n, -1),
        )
        cup_pour_axis_w = quat_apply(
            cup_quat_w,
            self._source_cup_pour_axis_b.unsqueeze(0).expand(n, -1),
        )

        mouth_delta = target_opening_w - source_pour_point_w
        target_dir = mouth_delta - (mouth_delta * cup_up_axis_w).sum(dim=-1, keepdim=True) * cup_up_axis_w
        pour_axis_plane = cup_pour_axis_w - (cup_pour_axis_w * cup_up_axis_w).sum(dim=-1, keepdim=True) * cup_up_axis_w
        target_dir = self._safe_normalize(target_dir, pour_axis_plane)

        tilt_toward_axis = self._safe_normalize(
            torch.cross(target_dir, cup_up_axis_w, dim=-1),
            torch.cross(pour_axis_plane, cup_up_axis_w, dim=-1),
        )
        tilt_ortho_axis = self._safe_normalize(
            torch.cross(cup_up_axis_w, tilt_toward_axis, dim=-1),
            pour_axis_plane,
        )
        spin_axis = self._safe_normalize(cup_up_axis_w, cup_up_axis_w.new_tensor([0.0, 0.0, 1.0]).expand_as(cup_up_axis_w))

        return (
            delta_local[:, 0:1] * spin_axis
            + delta_local[:, 1:2] * tilt_toward_axis
            + delta_local[:, 2:3] * tilt_ortho_axis
        )

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

        self.arm_dof_indices  = self.actuated_dof_indices[:NUM_ARM_DOF]    # list[int]
        self.hand_dof_indices = self.actuated_dof_indices[NUM_ARM_DOF:]    # list[int]

        # body indices (robot.data.body_pos_w 참조용)
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
        # distal phalanx body indices (rl_dg_*_4)
        _distal4_names = [f"rl_dg_{i}_4" for i in range(1, 6)]
        self.distal4_body_indices: list[int] = [
            self.robot.data.body_names.index(name)
            for name in _distal4_names
            if name in self.robot.data.body_names
        ]

        # ----------------------------------------------------------------
        # Palm pose 절대 workspace (안전 한계 클램프용)
        # ----------------------------------------------------------------
        self.palm_mins = to_torch(PALM_POSE_MINS_FUNC(cfg.max_pose_angle), device=self.device)
        self.palm_maxs = to_torch(PALM_POSE_MAXS_FUNC(cfg.max_pose_angle), device=self.device)

        # ----------------------------------------------------------------
        # Delta palm action 범위 (pregrasp 기준 상대 오프셋)
        # action=0 → pregrasp 위치 유지, action=±1 → ±delta 이동
        # scale(0) = pregrasp 이므로 초기 정책(출력≈0) = 안정된 pregrasp 위치
        # ----------------------------------------------------------------
        _delta_rad = math.radians(cfg.palm_delta_rot_deg)
        self.delta_mins = to_torch([
            -cfg.palm_delta_xyz, -cfg.palm_delta_xyz, -cfg.palm_delta_xyz,
            -_delta_rad, -_delta_rad, -_delta_rad,
        ], device=self.device)
        self.delta_maxs = to_torch([
            cfg.palm_delta_xyz, cfg.palm_delta_xyz, cfg.palm_delta_xyz,
            _delta_rad, _delta_rad, _delta_rad,
        ], device=self.device)
        self.delta_mins_warmstart_collect = to_torch([
            -cfg.warmstart_collect_palm_delta_xyz,
            -cfg.warmstart_collect_palm_delta_xyz,
            -cfg.warmstart_collect_palm_delta_xyz,
            -_delta_rad, -_delta_rad, -_delta_rad,
        ], device=self.device)
        self.delta_maxs_warmstart_collect = to_torch([
            cfg.warmstart_collect_palm_delta_xyz,
            cfg.warmstart_collect_palm_delta_xyz,
            cfg.warmstart_collect_palm_delta_xyz,
            _delta_rad, _delta_rad, _delta_rad,
        ], device=self.device)
        # pregrasp palm pose 버퍼 (에피소드별 delta action 기준점)
        # [x, y, z, qx, qy, qz, qw]
        self.pregrasp_palm_pose_buf = torch.zeros(self.num_envs, 7, device=self.device)

        # ----------------------------------------------------------------
        # Hand poses (per-finger lerp용)
        # open_pose = HAND_APPROACH_POSE (action=-1), grasp_pose = HAND_GRASP_POSE (action=+1)
        # ----------------------------------------------------------------
        self.hand_open_pose       = to_torch(HAND_APPROACH_POSE,  device=self.device)  # (20,)
        self.hand_grasp_pose      = to_torch(HAND_GRASP_POSE,     device=self.device)  # (20,)
        self.hand_full_grip_pose  = to_torch(HAND_FULL_GRIP_POSE, device=self.device)  # (20,)

        # ----------------------------------------------------------------
        # 로봇 시작 자세 (arm: ARM_START_POSE, hand: HAND_APPROACH_POSE)
        # ----------------------------------------------------------------
        arm_start  = to_torch(ARM_START_POSE,    device=self.device)  # (7,)
        hand_start = to_torch(HAND_APPROACH_POSE, device=self.device)  # (20,)
        robot_start = torch.cat([arm_start, hand_start], dim=0)         # (27,)
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
        self._grasp_rel_palm_to_cup_init = torch.zeros(self.num_envs, 3, device=self.device)
        self._grasp_cup_pos_palm_local_init = torch.zeros(self.num_envs, 3, device=self.device)  # palm local frame
        self._needs_grasp_init_update = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_cup_height_init = torch.zeros(self.num_envs, device=self.device)
        self.palm_center_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.fingertip_pos   = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.distal4_pos     = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.actions         = torch.zeros(self.num_envs, cfg.num_actions, device=self.device)
        self.prev_actions    = torch.full((self.num_envs, cfg.num_actions), 0.0, device=self.device)
        self._prev_mouth_xy_distance = torch.zeros(self.num_envs, device=self.device)
        # Phase-0 진단용: arm joint velocity/acceleration 추적 버퍼
        self._prev_arm_joint_vel = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        # [Phase-1 Step 6] jerk 계산용: 이전 step acc 벡터
        self._prev_arm_joint_acc = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        # [Phase-1 Step 7] EMA palm action 버퍼 (Fabrics IK 입력 smoothing)
        self._ema_palm_action = torch.zeros(self.num_envs, 6, device=self.device)

        # ----------------------------------------------------------------
        # ADR schedulers (spill penalty / noise scaling)
        # ----------------------------------------------------------------
        self.spill_adr = (
            PourADR(
                custom_cfg=cfg.spill_adr_custom_cfg,
                num_increments=cfg.spill_adr_num_increments,
                increment_interval=cfg.spill_adr_increment_interval,
                trigger_threshold=cfg.spill_adr_trigger_threshold,
            )
            if cfg.enable_spill_adr
            else None
        )

        self.noise_adr = (
            PourADR(
                custom_cfg=cfg.noise_adr_custom_cfg,
                num_increments=cfg.noise_adr_num_increments,
                increment_interval=cfg.noise_adr_increment_interval,
                trigger_threshold=cfg.noise_adr_trigger_threshold,
            )
            if cfg.enable_noise_adr
            else None
        )

        self.success_adr = (
            PourADR(
                custom_cfg=cfg.success_adr_custom_cfg,
                num_increments=cfg.success_adr_num_increments,
                increment_interval=cfg.success_adr_increment_interval,
                trigger_threshold=cfg.success_adr_trigger_threshold,
            )
            if cfg.enable_success_adr
            else None
        )

        self._noise_base_joint_pos = cfg.obs_noise_joint_pos
        self._noise_base_joint_vel = cfg.obs_noise_joint_vel
        self._noise_base_body_pos  = cfg.obs_noise_body_pos
        self._noise_base_cup_pos   = cfg.obs_noise_cup_pos

        # ----------------------------------------------------------------
        # Pregrasp / Lift 버퍼 (reset에서 계산)
        # ----------------------------------------------------------------
        self.pregrasp_arm_pos_buf  = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        self.grasp_hold_hand_pos_buf = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)

        # ----------------------------------------------------------------
        # Hand joint targets (per-finger lerp 결과)
        # ----------------------------------------------------------------
        self.hand_joint_targets = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        # warmstart reset으로 시작한 env는 초기 grasp보다 열리는 방향(action 감소)을 금지한다.
        self._warmstart_finger_action_floor = torch.full(
            (self.num_envs, NUM_FINGER_ACTION), -1.0, device=self.device
        )
        self._warmstart_only_close = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # warmstart 텔레포트 직후 컵 절대 고정 좌표 (world frame)
        # hold 기간 동안 이 위치로 컵을 고정하여 PhysX 접촉력 재정립 시간을 확보
        self._cup_hold_pos_w  = torch.zeros(self.num_envs, 3, device=self.device)
        self._cup_hold_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._cup_hold_quat_w[:, 0] = 1.0  # identity quaternion

        # ----------------------------------------------------------------
        # 접촉 상태 버퍼
        # ----------------------------------------------------------------
        self.contact_force_xyz_raw = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.contact_force_raw     = torch.zeros(self.num_envs, NUM_FINGERTIPS, device=self.device)
        self.binary_contact_buf    = torch.zeros(self.num_envs, NUM_FINGERTIPS, dtype=torch.bool, device=self.device)
        self.num_contacts_buf      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.distal_contact_force_raw  = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, device=self.device)
        self.distal_binary_contact_buf = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, dtype=torch.bool, device=self.device)

        self.middle_contact_force_raw  = torch.zeros(self.num_envs, NUM_MIDDLE_SENSORS, device=self.device)
        self.middle_binary_contact_buf = torch.zeros(self.num_envs, NUM_MIDDLE_SENSORS, dtype=torch.bool, device=self.device)

        # ----------------------------------------------------------------
        # 기타 버퍼
        # ----------------------------------------------------------------
        self._approach_dir_buf = torch.zeros(self.num_envs, 3, device=self.device)
        self.success_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # episode-level 성공 추적 (per-step average 허수 문제 해결)
        self.episode_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 슬라이딩 윈도우 성공률 (누적 방식 대체: stage 전환 시 이전 성공이 오염되는 문제 해소)
        _window_size = cfg.bead_count_adr_window_size if cfg.enable_bead_count_adr else 1000
        self._success_window: deque[int] = deque(maxlen=_window_size)

        # ----------------------------------------------------------------
        # Fabrics 초기화
        # ----------------------------------------------------------------
        self._setup_geometric_fabrics()

        # cspace attractor: hand는 grasp pose 방향
        cspace_default = self.open_tesollo_fabric.default_config.clone()
        cspace_default[:, NUM_ARM_DOF:] = self.hand_grasp_pose.unsqueeze(0).expand(self.num_envs, -1)
        self.open_tesollo_fabric.default_config.copy_(cspace_default)

        # 초기 액션: 0 → palm pose workspace 중심 (접근 자세 유지)
        self.actions.zero_()

        # Left target cup는 reset 시점 attach pose를 스냅샷으로 저장하고,
        # 에피소드 중에는 그 월드 pose를 고정 유지한다.
        self._left_target_cup_attach_pos_b = to_torch(
            self.cfg.left_target_cup_attach_pos_b, device=self.device
        )
        self._left_target_cup_attach_quat_b = to_torch(
            self.cfg.left_target_cup_attach_quat_wxyz_b, device=self.device
        )
        self._left_target_cup_body_id, self._left_target_cup_attach_pos_b = self._resolve_attachment_body(
            self.cfg.left_target_cup_attach_frame_name,
            self._left_target_cup_attach_pos_b,
        )
        self._left_target_cup_fixed_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
        self._bead_spawn_pos_source_cup_b = to_torch(self.cfg.bead_spawn_pos_source_cup_b, device=self.device)
        self._bead_spawn_quat_source_cup = to_torch(self.cfg.bead_spawn_quat_source_cup_wxyz, device=self.device)
        self._source_cup_pour_point_pos_b = to_torch(self.cfg.source_cup_pour_point_pos_b, device=self.device)
        self._target_cup_opening_pos_b = to_torch(self.cfg.target_cup_opening_pos_b, device=self.device)
        self._source_cup_pour_axis_b = to_torch(self.cfg.source_cup_pour_axis_b, device=self.device)
        self._source_cup_up_axis_b = to_torch(self.cfg.source_cup_up_axis_b, device=self.device)
        self._target_cup_up_axis_b = to_torch(self.cfg.target_cup_up_axis_b, device=self.device)
        self.num_beads = int(self.cfg.bead_count)

        # ---- Hidden parking grid (비활성 bead를 env 외부로 숨김) ----
        _hidden_offsets = []
        for _i in range(self.num_beads):
            _col = _i % cfg.bead_hidden_cols
            _row = _i // cfg.bead_hidden_cols
            _hidden_offsets.append([
                cfg.bead_hidden_base_x + _col * cfg.bead_hidden_spacing,
                cfg.bead_hidden_base_y + _row * cfg.bead_hidden_spacing,
                cfg.bead_hidden_z,
            ])
        self._hidden_bead_offsets_b = torch.tensor(_hidden_offsets, dtype=torch.float32, device=self.device)

        # ---- Bead count ADR ----
        self._bead_count_stages: list[int] = (
            list(cfg.bead_count_stages) if cfg.enable_bead_count_adr else [self.num_beads]
        )
        self._bead_adr_stage_idx: int = 0
        _initial_count = self._bead_count_stages[0]
        self._active_bead_count = torch.full(
            (self.num_envs,), _initial_count, dtype=torch.long, device=self.device
        )
        # active mask: 첫 k개 bead가 활성 (나머지는 parking grid에 숨김)
        self._active_bead_mask = torch.zeros(
            (self.num_envs, self.num_beads), dtype=torch.bool, device=self.device
        )
        self._active_bead_mask[:, :_initial_count] = True

        _bead_offsets = []
        beads_per_layer = 5
        for i in range(self.num_beads):
            layer = i // beads_per_layer
            slot = i % beads_per_layer
            angle = (2.0 * math.pi * slot / beads_per_layer) + (0.35 * layer)
            radius = 0.014 + 0.004 * (layer % 2)
            z = 0.006 + 0.014 * layer
            _bead_offsets.append([radius * math.cos(angle), radius * math.sin(angle), z])
        self._bead_offsets_source_cup_b = torch.tensor(_bead_offsets, device=self.device)
        self._source_pour_point_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_opening_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._source_pour_axis_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._source_up_axis_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_up_axis_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._mouth_delta = torch.zeros(self.num_envs, 3, device=self.device)
        self._mouth_distance = torch.zeros(self.num_envs, device=self.device)

        # ---- Pour 중간값 버퍼 ----
        self._mouth_xy_distance = torch.zeros(self.num_envs, device=self.device)
        self._cup_center_xy_dist = torch.zeros(self.num_envs, device=self.device)
        self._mouth_z_clearance = torch.zeros(self.num_envs, device=self.device)
        self._source_up_dot_world = torch.zeros(self.num_envs, device=self.device)
        self._directional_tilt_cos = torch.zeros(self.num_envs, device=self.device)
        self._mouth_alignment_cos = torch.zeros(self.num_envs, device=self.device)
        self._g_align_xy = torch.zeros(self.num_envs, device=self.device)
        self._g_clear = torch.zeros(self.num_envs, device=self.device)
        self._g_tilt = torch.zeros(self.num_envs, device=self.device)
        self._g_ready = torch.zeros(self.num_envs, device=self.device)
        self._g_pour = torch.zeros(self.num_envs, device=self.device)
        self._bead_in_target = torch.zeros(self.num_envs, self.num_beads, dtype=torch.bool, device=self.device)
        self._bead_in_source = torch.zeros(self.num_envs, self.num_beads, dtype=torch.bool, device=self.device)
        self._bead_ever_in_target = torch.zeros(
            self.num_envs, self.num_beads, dtype=torch.bool, device=self.device
        )
        self._bead_crossed_target_mouth = torch.zeros(
            self.num_envs, self.num_beads, dtype=torch.bool, device=self.device
        )
        self._prev_bead_target_local_z = torch.full(
            (self.num_envs, self.num_beads), 10.0, device=self.device
        )
        self._bead_cross_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._bead_cross_fraction = torch.zeros(self.num_envs, device=self.device)
        self._prev_bead_ever_in_target_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._bead_in_target_fraction = torch.zeros(self.num_envs, device=self.device)
        self._bead_in_source_fraction = torch.zeros(self.num_envs, device=self.device)
        self._bead_centroid_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._spill_ratio = torch.zeros(self.num_envs, device=self.device)
        self._all_beads_bonus_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._first_capture_bonus_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tilt_onset_bonus_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pre_pour_ready_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._no_tip_force_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._source_empty_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._world_up = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)

        self._warmstart_collect_mode = False
        self._warmstart_policy = None
        self._warmstart_cache_count = 0
        self._warmstart_reset_debug_printed = False
        cache_size = max(int(self.cfg.warmstart_cache_size), 1)
        self._warmstart_arm_pos = torch.zeros(cache_size, NUM_ARM_DOF, device=self.device)
        self._warmstart_hand_pos = torch.zeros(cache_size, NUM_HAND_DOF, device=self.device)
        self._warmstart_palm_pose = torch.zeros(cache_size, 7, device=self.device)
        self._warmstart_cup_pose = torch.zeros(cache_size, 7, device=self.device)
        # 연속 안정 hold 카운터 (collect mode 에서만 사용)
        self._warmstart_stable_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # GUI target visualization: source pour point (red) + target opening (blue)
        # 비활성화 가능 (cfg.enable_visual_markers)
        if cfg.enable_visual_markers:
            self._vis_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/FiveGPourRightMarkers",
                    markers={
                        "source_pour": sim_utils.SphereCfg(
                            radius=0.018,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                        ),
                        "target_opening": sim_utils.SphereCfg(
                            radius=0.018,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 1.0)),
                        ),
                    },
                )
            )
        else:
            self._vis_markers = None

        # [v4] _reset_idx 에서 참조되므로 warmstart 캐시 빌드 전에 None 으로 초기화
        self._trajectory_capture = None
        self.success_trajectory_buffer = None

        self._build_warmstart_reset_cache()

        # [v4] Trajectory capture + Success trajectory buffer (BC loss용)
        if cfg.enable_trajectory_capture:
            self._trajectory_capture = TrajectoryCapture(
                num_envs=self.num_envs,
                window=cfg.trajectory_capture_window,
                obs_dim=cfg.num_observations,
                act_dim=cfg.num_actions,
                device=self.device,
            )
            self.success_trajectory_buffer = SuccessTrajectoryBuffer(
                capacity=cfg.trajectory_buffer_capacity,
                max_len=cfg.trajectory_capture_window,
                obs_dim=cfg.num_observations,
                act_dim=cfg.num_actions,
                device=self.device,
            )
        else:
            self._trajectory_capture = None
            self.success_trajectory_buffer = None

    # ------------------------------------------------------------------
    # Scene 설정
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cup = RigidObject(self.cfg.cup_cfg)
        self.left_target_cup = RigidObject(self.cfg.left_target_cup_cfg)
        self.beads = RigidObjectCollection(self.cfg.beads_cfg)
        self.table = RigidObject(self.cfg.table_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cup"] = self.cup
        self.scene.rigid_objects["left_target_cup"] = self.left_target_cup
        self.scene.rigid_object_collections["beads"] = self.beads
        self.scene.rigid_objects["table"] = self.table

        # Actor: fingertip 개별 ContactSensor (Cup-only, real FT sensor 대응)
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

        # Critic: distal phalanx ContactSensor (sim-only)
        self._distal_sensor = ContactSensor(self.cfg.distal_sensor_cfg)
        self.scene.sensors["distal_sensor"] = self._distal_sensor

        # Critic: middle phalanx ContactSensor (sim-only)
        self._middle_sensor = ContactSensor(self.cfg.middle_sensor_cfg)
        self.scene.sensors["middle_sensor"] = self._middle_sensor

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

        # Main fabric (arm 제어용, graph_capturable=False)
        self.open_tesollo_fabric = OpenArmTeoslloPoseFabric(
            self.num_envs, self.device, self.timestep,
            graph_capturable=False,
            use_hand_fabric=False,
        )
        num_joints = self.open_tesollo_fabric.num_joints   # 27

        self.open_tesollo_integrator = DisplacementIntegrator(self.open_tesollo_fabric)

        # Fabric 상태 버퍼
        self.fabric_q   = self.robot_start_joint_pos.clone().contiguous()
        self.fabric_qd  = torch.zeros(self.num_envs, num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, num_joints, device=self.device)

        # Fabric input 버퍼
        self.hand_pca_targets  = torch.zeros(self.num_envs, 5, device=self.device)
        self.palm_pose_targets = torch.zeros(self.num_envs, 7, device=self.device)
        self.fabric_damping_gain = self.cfg.fabrics_damping_gain * torch.ones(self.num_envs, 1, device=self.device)

        # Reset 전용 소형 Fabrics (chunk 단위)
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



        # Pregrasp IK 캐시 사전 계산 (spawn grid 전체)
        self._build_pregrasp_cache()

    # ------------------------------------------------------------------
    # Pregrasp grid 캐시 빌드 (startup 1회)
    # ------------------------------------------------------------------
    def _build_pregrasp_cache(self) -> None:
        """spawn 위치 13×13 grid에 대해 Fabrics IK를 startup에서 일괄 계산.

        reset 시 nearest-neighbor lookup → Fabrics rollout 생략 → 대폭 속도 향상.
        1cm 간격 grid이므로 실제 spawn 위치와 최대 ~0.7cm 오차 → Fabrics가 첫 몇 스텝에서 보정.
        """
        _N = 13  # 1cm 간격, ±6cm 범위
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
        M = flat_x.shape[0]  # 169

        palm_euler = torch.zeros(M, 6, device=self.device)
        palm_euler[:, 0] = flat_x + self.cfg.pregrasp_offset_x
        palm_euler[:, 1] = flat_y + self.cfg.pregrasp_offset_y
        palm_euler[:, 2] = self.cfg.object_spawn_z + self.cfg.pregrasp_offset_z
        palm_euler[:, 3] = math.radians(90.0)
        palm_euler[:, 4] = math.radians(0.0)
        palm_euler[:, 5] = math.radians(90.0)
        palm_euler = torch.max(
            torch.min(palm_euler, self.palm_maxs.unsqueeze(0)),
            self.palm_mins.unsqueeze(0),
        )
        palm = torch.zeros(M, 7, device=self.device)
        palm[:, :3] = palm_euler[:, :3]
        palm[:, 3:7] = self._quat_xyzw_from_euler_zyx(palm_euler[:, 3:6])

        q_init = self.robot_start_joint_pos[0].unsqueeze(0).expand(M, -1).contiguous()
        dummy  = torch.arange(M, device=self.device)
        q_out  = self._run_reset_fabric(dummy, palm, q_init.clone())

        # (13, 13, 7): arm joints only
        self._cache_q_arm = q_out[:, :NUM_ARM_DOF].view(_N, _N, NUM_ARM_DOF).contiguous()
        self._cache_xs    = xs
        self._cache_ys    = ys
        self._cache_n     = _N


    # ------------------------------------------------------------------
    # Reset 전용 Fabrics rollout (chunk 단위)
    # ------------------------------------------------------------------
    def _run_reset_fabric(
        self,
        env_ids: torch.Tensor,
        palm_pose: torch.Tensor,
        q_init: torch.Tensor,
    ) -> torch.Tensor:
        """env_ids(n개)만 Fabrics rollout해서 arm joint 위치 반환."""
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
                "quaternion",
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

    # ------------------------------------------------------------------
    # 접촉력 업데이트
    # ------------------------------------------------------------------
    def _update_contact_forces(self) -> None:
        # Actor: fingertip 개별 센서 (Cup-only)
        tip_xyz = torch.stack([
            s.data.force_matrix_w[:, 0, 0, :] for s in self._tip_sensors
        ], dim=1)   # (N, 5, 3)
        tip_norms = tip_xyz.norm(dim=-1)   # (N, 5)

        self.contact_force_xyz_raw.copy_(tip_xyz)
        self.contact_force_raw.copy_(tip_norms)
        self.binary_contact_buf.copy_(tip_norms > CONTACT_FORCE_THRESHOLD)
        self.num_contacts_buf.copy_(self.binary_contact_buf.sum(dim=-1).long())

        # Critic: distal
        per_distal = self._distal_sensor.data.net_forces_w.norm(dim=-1)   # (N, 5)
        self.distal_contact_force_raw.copy_(per_distal)
        self.distal_binary_contact_buf.copy_(per_distal > CONTACT_FORCE_THRESHOLD)

        # Critic: middle
        per_middle = self._middle_sensor.data.net_forces_w.norm(dim=-1)   # (N, 5)
        self.middle_contact_force_raw.copy_(per_middle)
        self.middle_binary_contact_buf.copy_(per_middle > CONTACT_FORCE_THRESHOLD)

    def _compute_bead_flags(self) -> None:
        """beads가 source/target cup 내부 또는 target mouth를 통과했는지 계산."""
        bead_pos_w = self.beads.data.object_pos_w
        self._bead_centroid_w.copy_(bead_pos_w.mean(dim=1))

        n = bead_pos_w.shape[0]
        k = bead_pos_w.shape[1]

        left_cup_quat_w = self.left_target_cup.data.root_quat_w
        left_cup_pos_w = self.left_target_cup.data.root_pos_w
        left_quat_flat = left_cup_quat_w.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
        left_rel_flat = (bead_pos_w - left_cup_pos_w.unsqueeze(1)).reshape(-1, 3)
        pos_in_target = quat_apply_inverse(left_quat_flat, left_rel_flat).reshape(n, k, 3)
        bead_xy_to_target = torch.norm(pos_in_target[..., :2], dim=-1)
        bead_in_target = (
            (bead_xy_to_target <= self.cfg.target_inner_radius)
            & (pos_in_target[..., 2] >= self.cfg.target_inside_z_min)
            & (pos_in_target[..., 2] <= self.cfg.target_inside_z_max)
        )
        self._bead_in_target.copy_(bead_in_target)
        self._bead_ever_in_target |= bead_in_target

        cup_quat_w = self.cup.data.root_quat_w
        cup_pos_w = self.cup.data.root_pos_w
        cup_quat_flat = cup_quat_w.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
        cup_rel_flat = (bead_pos_w - cup_pos_w.unsqueeze(1)).reshape(-1, 3)
        pos_in_source = quat_apply_inverse(cup_quat_flat, cup_rel_flat).reshape(n, k, 3)
        bead_xy_to_source = torch.norm(pos_in_source[..., :2], dim=-1)
        bead_in_source = (
            (bead_xy_to_source <= self.cfg.source_inner_radius)
            & (pos_in_source[..., 2] >= self.cfg.source_inside_z_min)
            & (pos_in_source[..., 2] <= self.cfg.source_inside_z_max)
        )
        self._bead_in_source.copy_(bead_in_source)

        mouth_crossed_now = (
            (bead_xy_to_target <= self.cfg.target_inner_radius)
            & (self._prev_bead_target_local_z > self.cfg.target_mouth_z)
            & (pos_in_target[..., 2] <= self.cfg.target_mouth_z)
        )
        self._bead_crossed_target_mouth |= mouth_crossed_now
        # active mask 기반 분모: inactive bead는 집계에서 제외
        _active_count_f = self._active_bead_count.clamp(min=1).float()  # (N,)
        _mask = self._active_bead_mask  # (N, num_beads)

        self._bead_cross_count.copy_(
            (self._bead_crossed_target_mouth & _mask).sum(dim=-1).long()
        )
        self._bead_cross_fraction.copy_(
            (self._bead_crossed_target_mouth & _mask).float().sum(dim=-1) / _active_count_f
        )
        self._bead_in_target_fraction.copy_(
            (self._bead_in_target & _mask).float().sum(dim=-1) / _active_count_f
        )
        self._bead_in_source_fraction.copy_(
            (self._bead_in_source & _mask).float().sum(dim=-1) / _active_count_f
        )

        bead_env_z = bead_pos_w[..., 2] - self.scene.env_origins[:, 2].unsqueeze(1)
        bead_spilled = (
            (~self._bead_in_target)
            & (~self._bead_in_source)
            & (bead_env_z < 0.230)
        )
        self._spill_ratio.copy_(
            (bead_spilled & _mask).float().sum(dim=-1) / _active_count_f
        )
        self._prev_bead_target_local_z.copy_(pos_in_target[..., 2])

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone()

        palm_action   = actions[:, :6]    # (N, 6) ∈ [-1, 1]
        finger_action = actions[:, 6:11]  # (N, 5) ∈ [-1, 1]

        # ---- Pour phase: Fabrics arm 제어 ----
        # Delta action: action=0 → pregrasp 유지, action=±1 → pregrasp ± delta
        # 절대 workspace(palm_mins/maxs)로 클램프하여 안전 영역 보장

        # 에피소드 시작 직후 N스텝: warmstart pose를 강제 유지 (물리 안착)
        # warmstart 캐시에서 텔레포트한 직후 contact force가 안정화되기 전에
        # 랜덤 action이 arm/hand를 움직이면 손가락이 먼저 풀리거나 컵이 낙하함.
        if self.cfg.episode_hold_steps > 0:
            hold_mask = (self.episode_length_buf < self.cfg.episode_hold_steps).unsqueeze(1)
            palm_action = torch.where(hold_mask, torch.zeros_like(palm_action), palm_action)
            if not self._warmstart_collect_mode:
                finger_action = torch.where(hold_mask, torch.ones_like(finger_action), finger_action)
        if not self._warmstart_collect_mode:
            close_only = self._warmstart_only_close.unsqueeze(1)
            finger_action = torch.where(
                close_only,
                torch.maximum(finger_action, self._warmstart_finger_action_floor),
                finger_action,
            )
            self.actions[:, 6:11] = finger_action

        # [Phase-1 Step 7] EMA palm action smoothing: Fabrics에 smooth 궤적 전달
        # action_rate_penalty는 raw self.actions 기반 유지 (training gradient 보존)
        self._ema_palm_action.copy_(
            self.cfg.ema_action_alpha * palm_action
            + (1.0 - self.cfg.ema_action_alpha) * self._ema_palm_action
        )

        if self._warmstart_collect_mode:
            delta = scale(self._ema_palm_action, self.delta_mins_warmstart_collect, self.delta_maxs_warmstart_collect)
        else:
            delta = scale(self._ema_palm_action, self.delta_mins, self.delta_maxs)   # (N, 6)
            # 멀리 있을 때는 회전/tilt action을 억제해서 "원거리 tilt"를 방지한다.
            gate_den = max(self.cfg.tilt_action_gate_xy_far - self.cfg.tilt_action_gate_xy_near, 1e-6)
            tilt_gate = torch.clamp(
                (self.cfg.tilt_action_gate_xy_far - self._mouth_xy_distance) / gate_den,
                0.0,
                1.0,
            )
            delta[:, 3:6] = delta[:, 3:6] * tilt_gate.unsqueeze(1)
        # Rotation action is interpreted in a cup-local basis:
        # [spin around cup-up, tilt toward target opening, orthogonal tilt].
        delta_rotvec_world = self._build_cup_local_tilt_rotvec(delta[:, 3:6])
        palm_pose = torch.zeros_like(self.pregrasp_palm_pose_buf)
        palm_pose[:, :3] = self.pregrasp_palm_pose_buf[:, :3] + delta[:, :3]
        palm_pose[:, :3] = torch.max(
            torch.min(palm_pose[:, :3], self.palm_maxs[:3].unsqueeze(0)),
            self.palm_mins[:3].unsqueeze(0),
        )
        palm_pose[:, 3:7] = self._compose_world_delta_quat_xyzw(
            self.pregrasp_palm_pose_buf[:, 3:7],
            delta_rotvec_world,
        )
        self.palm_pose_targets.copy_(palm_pose)
        self.hand_pca_targets.zero_()

        # null-space attractor를 현재 관절 위치로 추적:
        # default_config가 warmstart grasp pose로 고정되면 pour transport 방향으로
        # 이동할 때 매 step 파지 위치로 당기는 저항이 발생함.
        # 현재 fabric_q를 default_config로 덮어써서 null-space 당김 제거.
        self.open_tesollo_fabric.default_config.copy_(self.fabric_q.detach())

        self.open_tesollo_fabric.set_features(
            self.hand_pca_targets,
            self.palm_pose_targets,
            "quaternion",
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

        # ---- 오른손 파지 유지 (pour 중 항상 grasp pose freeze) ----
        if self.cfg.freeze_grasp_hand_during_episode and (not self._warmstart_collect_mode):
            self.actions[:, 6:] = 1.0
            hand_target = self.grasp_hold_hand_pos_buf
        else:
            delta_20   = self.hand_grasp_pose - self.hand_open_pose                # (20,)
            t          = (finger_action + 1.0) / 2.0                               # (N,5) ∈ [0,1]
            t_expanded = t.repeat_interleave(4, dim=1)                             # (N,20)
            hand_target = self.hand_open_pose.unsqueeze(0) + t_expanded * delta_20.unsqueeze(0)
        self.hand_joint_targets.copy_(hand_target)

        # fabric_q hand 부분 동기화 (FK 계산에 활용)
        self.fabric_q[:, NUM_ARM_DOF:] = hand_target
        self.fabric_qd[:, NUM_ARM_DOF:].zero_()

    def _apply_action(self) -> None:
        # ---- 오른팔: Fabrics arm target (pour phase 전체) ----
        arm_target = self.fabric_q[:, :NUM_ARM_DOF]
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(arm_target), joint_ids=self.arm_dof_indices
        )

        # ---- 오른손: grasp_hold 유지 ----
        self.robot.set_joint_position_target(self.hand_joint_targets, joint_ids=self.hand_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(self.hand_joint_targets), joint_ids=self.hand_dof_indices
        )

        # ---- 왼팔: 고정 자세 ----
        self.robot.set_joint_position_target(
            self.left_arm_zero_pos, joint_ids=self.left_arm_dof_indices
        )
        self.robot.set_joint_velocity_target(
            self.left_arm_zero_vel, joint_ids=self.left_arm_dof_indices
        )

        left_cup_pose = self._get_left_target_cup_fixed_pose()
        zero_cup_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self.left_target_cup.write_root_pose_to_sim(left_cup_pose)
        self.left_target_cup.write_root_velocity_to_sim(zero_cup_vel)

        # cup pinning 비활성화: warmstart(grasp) 캐시 리셋에서 컵을 고정하면
        # 손가락(position-control)과 충돌하여 PhysX가 관통을 해소 못해 컵이 팅겨져 나감.
        # v3와 동일하게 physics가 자연스럽게 접촉을 재정립하도록 허용.

    # ------------------------------------------------------------------
    # Intermediate values
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self) -> None:
        # 물체 위치
        self.object_pos = self.cup.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.cup.data.root_quat_w
        left_target_pos_w = self.left_target_cup.data.root_pos_w
        left_target_quat_w = self.left_target_cup.data.root_quat_w

        # body_pos_w 기반 위치 (실제 sim 위치, Fabrics FK보다 정확)
        env_origins = self.scene.env_origins

        if self.palm_body_index >= 0:
            self.palm_center_pos = (
                self.robot.data.body_pos_w[:, self.palm_body_index, :] - env_origins
            )

        self.fingertip_pos = (
            self.robot.data.body_pos_w[:, self.fingertip_body_indices, :] - env_origins.unsqueeze(1)
        )   # (N, 5, 3)

        if len(self.distal4_body_indices) == NUM_FINGERTIPS:
            self.distal4_pos = (
                self.robot.data.body_pos_w[:, self.distal4_body_indices, :] - env_origins.unsqueeze(1)
            )   # (N, 5, 3)

        n = self.num_envs
        self._source_pour_point_w = self._compute_dynamic_source_pour_point_w(
            self.cup.data.root_pos_w,
            self.cup.data.root_quat_w,
        )
        self._target_opening_w = left_target_pos_w + quat_apply(
            left_target_quat_w,
            self._target_cup_opening_pos_b.unsqueeze(0).expand(n, -1),
        )
        self._source_pour_axis_w = quat_apply(
            self.cup.data.root_quat_w,
            self._source_cup_pour_axis_b.unsqueeze(0).expand(n, -1),
        )
        self._source_up_axis_w = quat_apply(
            self.cup.data.root_quat_w,
            self._source_cup_up_axis_b.unsqueeze(0).expand(n, -1),
        )
        self._target_up_axis_w = quat_apply(
            left_target_quat_w,
            self._target_cup_up_axis_b.unsqueeze(0).expand(n, -1),
        )
        self._mouth_delta = self._target_opening_w - self._source_pour_point_w
        self._mouth_distance = torch.norm(self._mouth_delta, dim=-1)

        # ---- Pour 기하학 계산 (bi_pouring_v1 패턴) ----
        self._mouth_xy_distance = torch.norm(self._mouth_delta[:, :2], dim=-1)
        self._mouth_z_clearance = self._source_pour_point_w[:, 2] - self._target_opening_w[:, 2]
        self._source_up_dot_world = self._source_up_axis_w[:, 2].clamp(-1.0, 1.0)

        # Directional tilt: 원통 컵의 실제 pouring side는 cup root +Z 축이 기울어진 반대편 림이다.
        # 따라서 낮아지는 림 방향 = -project(source_up_axis, XY) 로 정의하고,
        # 그 방향이 target opening의 XY 방향을 향하도록 보상한다.
        _mouth_delta_xy = self._mouth_delta[:, :2]   # (N, 2): target - source XY
        _mouth_dir_xy = _mouth_delta_xy / (_mouth_delta_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        # cup up axis XY = mouth 방향 XY. 올바른 pour = mouth가 target 방향 → 양수
        _mouth_tilt_dir_xy = self._source_up_axis_w[:, :2]
        _mouth_tilt_dir_xy = _mouth_tilt_dir_xy / (_mouth_tilt_dir_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        self._directional_tilt_cos = (_mouth_tilt_dir_xy * _mouth_dir_xy).sum(dim=-1).clamp(-1.0, 1.0)

        # Mouth alignment: a cylindrical cup has no meaningful yaw axis at the rim.
        # If we align a fixed local axis (e.g. [-1, 0, 0]) to the target, the policy can exploit
        # wrist yaw / joint7-only spin without improving actual pouring geometry.
        # Therefore use only the XY heading induced by the current tilt. When nearly upright,
        # there is no valid pour heading, so keep the alignment cosine at 0 (neutral / no reward).
        _mouth_dir = self._mouth_delta / self._mouth_distance.unsqueeze(1).clamp(min=1e-6)
        _pour_heading_xy = self._source_up_axis_w[:, :2]
        _pour_heading_xy_norm = _pour_heading_xy.norm(dim=-1, keepdim=True)
        _effective_heading_xy = torch.where(
            _pour_heading_xy_norm > 1e-4,
            _pour_heading_xy / _pour_heading_xy_norm.clamp(min=1e-6),
            torch.zeros_like(_pour_heading_xy),
        )
        _effective_pour_heading = torch.cat(
            [_effective_heading_xy, torch.zeros(n, 1, device=self.device)], dim=-1
        )
        self._mouth_alignment_cos = (_effective_pour_heading * _mouth_dir).sum(dim=-1).clamp(-1.0, 1.0)

        # dense shaping도 mouth geometry를 직접 따른다. 그렇지 않으면 root 정렬만 학습한다.
        self._cup_center_xy_dist = torch.norm(
            self.cup.data.root_pos_w[:, :2] - self._target_opening_w[:, :2], dim=-1
        )
        self._g_align_xy = torch.exp(-self.cfg.reward_gate_xy_scale * self._mouth_xy_distance)
        self._g_clear = torch.sigmoid(
            self.cfg.reward_gate_clear_scale
            * (self._mouth_z_clearance - self.cfg.reward_clearance_min)
        )
        self._g_tilt = torch.sigmoid(
            self.cfg.reward_gate_tilt_scale
            * (self._directional_tilt_cos - self.cfg.reward_tilt_cos_min)
        )
        self._g_ready = self._g_align_xy * self._g_clear
        self._g_pour = self._g_ready * self._g_tilt

        # Bead flags & spill
        self._compute_bead_flags()

        if self._vis_markers is not None:
            _all_pts = torch.cat([self._source_pour_point_w, self._target_opening_w], dim=0)
            _marker_idx = torch.zeros(2 * n, dtype=torch.long, device=self.device)
            _marker_idx[n:] = 1
            self._vis_markers.visualize(translations=_all_pts, marker_indices=_marker_idx)

        # 접촉력 업데이트
        self._update_contact_forces()

    # ------------------------------------------------------------------
    # Observations: Actor 105D | Critic 155D
    # ------------------------------------------------------------------
    def _get_legacy_warmstart_policy_obs(self) -> torch.Tensor:
        """Build the legacy actor observation expected by the warmstart checkpoint.

        Supported warmstart actor observation layouts:
          106D: v7 grasp checkpoint
          107D: v8 grasp checkpoint (106D + bead_mass_normalized)
          112D: future-proof fallback (107D + tip_force_norm)
          113D: v8 grasp checkpoint (112D + phase_step_ratio)
        """

        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_dof_indices]
        arm_joint_vel = self.robot.data.joint_vel[:, self.arm_dof_indices]
        finger_joint_pos = self.robot.data.joint_pos[:, self.hand_dof_indices]
        finger_joint_vel = self.robot.data.joint_vel[:, self.hand_dof_indices]
        palm_center_pos = self.palm_center_pos
        fingertip_pos = self.fingertip_pos
        cup_pos = self.object_pos
        binary_contact = self.binary_contact_buf.float()
        last_actions = self.actions

        fingertip_pos_rel_palm = (fingertip_pos - palm_center_pos.unsqueeze(1)).view(self.num_envs, -1)
        palm_to_cup = cup_pos - palm_center_pos
        cup_to_fingertip = (fingertip_pos - cup_pos.unsqueeze(1)).view(self.num_envs, -1)

        warmstart_obs_106 = torch.cat([
            arm_joint_pos,
            arm_joint_vel,
            finger_joint_pos,
            finger_joint_vel,
            palm_center_pos,
            fingertip_pos_rel_palm,
            palm_to_cup,
            cup_to_fingertip,
            binary_contact,
            last_actions,
        ], dim=-1)

        if warmstart_obs_106.shape[1] != 106:
            raise RuntimeError(f"[warmstart] Legacy obs dim mismatch: {warmstart_obs_106.shape[1]} != 106")

        if self._warmstart_policy is None:
            return warmstart_obs_106

        obs_dim = self._warmstart_policy.obs_dim
        if obs_dim == 106:
            return warmstart_obs_106

        bead_mass_normalized = torch.ones(self.num_envs, 1, device=self.device)
        warmstart_obs_107 = torch.cat([warmstart_obs_106, bead_mass_normalized], dim=-1)
        if obs_dim == 107:
            return warmstart_obs_107

        tip_force_norm = (self.contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)
        warmstart_obs_112 = torch.cat([warmstart_obs_107, tip_force_norm], dim=-1)
        if obs_dim == 112:
            return warmstart_obs_112

        # 5g_pour_right_v8 normalizes phase by its 720-step grasp horizon.
        warmstart_grasp_episode_steps = 720.0
        phase_step_ratio = (
            self.episode_length_buf.float() / warmstart_grasp_episode_steps
        ).unsqueeze(1)
        warmstart_obs_113 = torch.cat([warmstart_obs_112, phase_step_ratio], dim=-1)
        if obs_dim == 113:
            return warmstart_obs_113

        raise RuntimeError(
            f"[warmstart] Unsupported checkpoint obs dim: {obs_dim}. "
            "Expected one of {106, 107, 112, 113}."
        )

    def _get_observations(self) -> dict:
        # ==== 공통 clean state (critic용, 물리 정확값) ====
        arm_joint_pos_clean = self.robot.data.joint_pos[:, self.arm_dof_indices]
        arm_joint_vel_clean = self.robot.data.joint_vel[:, self.arm_dof_indices]
        finger_joint_pos_clean = self.robot.data.joint_pos[:, self.hand_dof_indices]
        finger_joint_vel_clean = self.robot.data.joint_vel[:, self.hand_dof_indices]
        left_arm_joint_pos_clean = self.robot.data.joint_pos[:, self.left_arm_dof_indices]
        left_arm_joint_vel_clean = self.robot.data.joint_vel[:, self.left_arm_dof_indices]
        palm_center_pos_clean = self.palm_center_pos

        right_cup_pos_clean = self.cup.data.root_pos_w
        right_cup_quat_clean = self.cup.data.root_quat_w
        left_cup_pos_clean = self.left_target_cup.data.root_pos_w
        left_cup_quat_clean = self.left_target_cup.data.root_quat_w

        source_pour_point_clean = self._source_pour_point_w
        target_opening_clean = self._target_opening_w
        source_pour_axis_clean = self._source_pour_axis_w
        source_up_axis_clean = self._source_up_axis_w
        target_up_axis_clean = self._target_up_axis_w
        bead_pos_clean = self._bead_centroid_w  # (미사용, 하위 호환 보존)

        # ==== Actor obs용 noisy state (sim2real domain randomization) ====
        if self._warmstart_collect_mode:
            σ_qp = σ_qv = σ_bp = σ_cp = 0.0
        elif self.noise_adr is not None:
            σ_qp = self.noise_adr.get_param("noise", "obs_noise_joint_pos")
            σ_qv = self.noise_adr.get_param("noise", "obs_noise_joint_vel")
            σ_bp = self.noise_adr.get_param("noise", "obs_noise_body_pos")
            σ_cp = self.noise_adr.get_param("noise", "obs_noise_cup_pos")
        else:
            σ_qp = self.cfg.obs_noise_joint_pos
            σ_qv = self.cfg.obs_noise_joint_vel
            σ_bp = self.cfg.obs_noise_body_pos
            σ_cp = self.cfg.obs_noise_cup_pos

        arm_joint_pos = arm_joint_pos_clean + torch.randn_like(arm_joint_pos_clean) * σ_qp
        arm_joint_vel = arm_joint_vel_clean + torch.randn_like(arm_joint_vel_clean) * σ_qv
        finger_joint_pos = finger_joint_pos_clean + torch.randn_like(finger_joint_pos_clean) * σ_qp
        finger_joint_vel = finger_joint_vel_clean + torch.randn_like(finger_joint_vel_clean) * σ_qv
        palm_center_pos = palm_center_pos_clean + torch.randn_like(palm_center_pos_clean) * σ_bp
        right_cup_pos = right_cup_pos_clean + torch.randn_like(right_cup_pos_clean) * σ_cp
        left_cup_pos = left_cup_pos_clean + torch.randn_like(left_cup_pos_clean) * σ_cp
        source_pour_point = source_pour_point_clean + torch.randn_like(source_pour_point_clean) * σ_cp
        target_opening = target_opening_clean + torch.randn_like(target_opening_clean) * σ_cp

        right_cup_pos_rel_palm = right_cup_pos - palm_center_pos
        left_cup_pos_rel_palm = left_cup_pos - palm_center_pos
        pour_point_to_opening = target_opening - source_pour_point

        binary_contact = self.binary_contact_buf.float()
        last_actions = self.actions

        # transport_summary (8D): pour 기하학 + stage readiness
        transport_summary = torch.stack([
            self._mouth_distance,
            self._mouth_xy_distance,       # pour point 기반 (pour phase 정밀도)
            self._cup_center_xy_dist,      # cup center 기반 (approach 기준, tilt-invariant)
            self._mouth_z_clearance,
            self._source_up_dot_world,
            self._directional_tilt_cos,
            self._mouth_alignment_cos,
            self._g_ready,
        ], dim=-1)   # (N, 8)

        tip_force_norm = (self.contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)

        actor_obs = torch.cat([
            arm_joint_pos,              # 7
            arm_joint_vel,              # 7
            finger_joint_pos,           # 20
            finger_joint_vel,           # 20
            right_cup_pos_rel_palm,     # 3
            right_cup_quat_clean,       # 4
            left_cup_pos_rel_palm,      # 3
            left_cup_quat_clean,        # 4
            pour_point_to_opening,      # 3
            source_pour_axis_clean,     # 3
            source_up_axis_clean,       # 3
            # target_up_axis 제거: 타겟 컵은 항상 직립 → 항상 [0,0,1], 정보 없음
            transport_summary,          # 8
            binary_contact,             # 5
            tip_force_norm,             # 5 (fingertip force, sim2real 가능)
            last_actions,               # 11
            # bead 상태: actor가 pour 결과를 직접 관측 (reward 항목 대응)
            self._bead_in_source_fraction.unsqueeze(1),  # 1 (소스 잔량)
            self._bead_in_target_fraction.unsqueeze(1),  # 1 (타겟 유입량, r_capture 대응)
            self._bead_cross_fraction.unsqueeze(1),      # 1 (mouth 통과율, r_cross weight=20 대응)
            self._spill_ratio.unsqueeze(1),              # 1 (유출율, spill_cost weight=10 대응)
        ], dim=-1)   # 110D

        if actor_obs.shape[1] != NUM_OBSERVATIONS:
            raise RuntimeError(
                f"[pour_v3] Actor obs dim mismatch: {actor_obs.shape[1]} != {NUM_OBSERVATIONS}"
            )

        # ==== Critic extra obs (50D) ====
        cup_height_delta = (right_cup_pos_clean[:, 2] - self.object_init_pos[:, 2]).unsqueeze(1)

        distal_binary     = self.distal_binary_contact_buf.float()
        distal_force_norm = (self.distal_contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)


        # critic actor_obs_clean (106D) — clean state 재조합, actor_obs 구조와 동일
        actor_obs_clean = torch.cat([
            arm_joint_pos_clean,                                      # 7
            arm_joint_vel_clean,                                      # 7
            finger_joint_pos_clean,                                   # 20
            finger_joint_vel_clean,                                   # 20
            right_cup_pos_clean - palm_center_pos_clean,              # 3
            right_cup_quat_clean,                                     # 4
            left_cup_pos_clean - palm_center_pos_clean,               # 3
            left_cup_quat_clean,                                      # 4
            target_opening_clean - source_pour_point_clean,           # 3
            source_pour_axis_clean,                                   # 3
            source_up_axis_clean,                                     # 3
            torch.stack([                                             # 8
                self._mouth_distance,
                self._mouth_xy_distance,
                self._cup_center_xy_dist,
                self._mouth_z_clearance,
                self._source_up_dot_world,
                self._directional_tilt_cos,
                self._mouth_alignment_cos,
                self._g_ready,
            ], dim=-1),
            binary_contact,                                           # 5
            tip_force_norm,                                           # 5 (v8처럼, sim2real)
            last_actions,                                             # 11
            self._bead_in_source_fraction.unsqueeze(1),               # 1
            self._bead_in_target_fraction.unsqueeze(1),               # 1
            self._bead_cross_fraction.unsqueeze(1),                   # 1
            self._spill_ratio.unsqueeze(1),                           # 1
        ], dim=-1)   # 110D

        critic_obs = torch.cat([
            actor_obs_clean,                                    # 110
            left_arm_joint_pos_clean,                          # 9
            left_arm_joint_vel_clean,                          # 9
            distal_binary,                                     # 5
            distal_force_norm,                                 # 5
            cup_height_delta,                                  # 1
            self._mouth_distance.unsqueeze(1),                 # 1
            self._mouth_xy_distance.unsqueeze(1),              # 1
            self._cup_center_xy_dist.unsqueeze(1),             # 1
            self._mouth_z_clearance.unsqueeze(1),              # 1
            self._source_up_dot_world.unsqueeze(1),            # 1
            self._directional_tilt_cos.unsqueeze(1),           # 1
            self._mouth_alignment_cos.unsqueeze(1),            # 1
            self._g_align_xy.unsqueeze(1),                     # 1
            self._g_clear.unsqueeze(1),                        # 1
            self._g_tilt.unsqueeze(1),                         # 1
            self._g_ready.unsqueeze(1),                        # 1
            self._g_pour.unsqueeze(1),                         # 1
            self._bead_cross_fraction.unsqueeze(1),            # 1
            self._bead_in_source_fraction.unsqueeze(1),        # 1
            self._bead_in_target_fraction.unsqueeze(1),        # 1
            self._spill_ratio.unsqueeze(1),                    # 1
        ], dim=-1)   # 155D

        if critic_obs.shape[1] != NUM_CRITIC_OBSERVATIONS:
            raise RuntimeError(
                f"[pour_v3] Critic obs dim mismatch: {critic_obs.shape[1]} != {NUM_CRITIC_OBSERVATIONS}"
            )

        # [v4] 궤적 캡처: (obs_t, action_t) 기록
        # action은 _pre_physics_step 에서 self.actions 에 저장됨
        if self._trajectory_capture is not None:
            self._trajectory_capture.append(actor_obs.detach(), self.actions.detach())

        return {"policy": actor_obs, "critic": critic_obs}

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        # ---- 1. Grasp hold: full-grip-pose L2 유지 ----
        # warmstart에서 이미 파지 완료 → full-grip-pose 유지만으로 충분
        hand_q = self.robot.data.joint_pos[:, self.hand_dof_indices]
        grip_err = (hand_q - self.hand_full_grip_pose).norm(dim=-1)
        r_hold = self.cfg.weight_grip_pose * torch.exp(-self.cfg.grip_pose_sharpness * grip_err)

        # ---- Stage B. Transport (Approach) - Always active with smooth gradient ----
        cup_height_delta = (self.object_pos[:, 2] - self.object_init_pos[:, 2]).clamp(
            min=0.0, max=self.cfg.lift_height_cap
        )
        r_lift = self.cfg.weight_approach_z * cup_height_delta
        
        # Smooth approach reward using tanh (0 at infinity, 1 at 0)
        # dense reward도 실제 붓는 점을 따라가야 mouth-to-mouth로 수렴한다.
        r_approach_xy = 1.0 - torch.tanh(self.cfg.reward_approach_xy_scale * self._mouth_xy_distance)
        r_transport_progress = torch.clamp(
            self._prev_mouth_xy_distance - self._mouth_xy_distance,
            min=0.0,
        )
        # 가까이 다가오면(≈3cm) 접근 보상을 끄고 pour 신호를 강조
        approach_gate_den = max(self.cfg.approach_xy_off_far - self.cfg.approach_xy_off_near, 1e-6)
        approach_gate = torch.clamp(
            (self._mouth_xy_distance - self.cfg.approach_xy_off_near) / approach_gate_den,
            min=0.0,
            max=1.0,
        )
        r_approach = approach_gate * (
            self.cfg.weight_approach_xy * r_approach_xy
            + self.cfg.weight_transport_progress * r_transport_progress
        )

        # ---- Stage C. Pre-pour (using soft g_ready gate) ----
        target_tilt_cos = math.cos(math.radians(self.cfg.pour_tilt_target_deg))
        r_tilt = torch.exp(
            -self.cfg.pour_tilt_sharpness * torch.abs(self._source_up_dot_world - target_tilt_cos)
        )
        r_align = 0.5 * (self._mouth_alignment_cos + 1.0)
        
        # Pre-pour signals: cup center 기반 gate (pour point 이동에 무관하게 stable gradient)
        # g_ready(mouth_xy 기반)는 pour binary gate에서만 사용; 여기선 cup center로 대체
        g_ready_prepour = (
            torch.exp(-self.cfg.reward_gate_xy_scale * self._cup_center_xy_dist)
            * self._g_clear
        )
        r_prepour_stage = g_ready_prepour * (
            self.cfg.weight_prepour_dir * r_tilt
            + self.cfg.weight_prepour_align * r_align
        )

        # ---- Stage D. Pour (mouth-to-mouth binary gate) ----
        # 컵 중심 근접만으로는 위에서 멀리 붓는 정책이 남으므로, 입구 정렬과 rim 높이를 직접 gate한다.
        r_cross = self._bead_cross_fraction
        r_capture = self._bead_in_target_fraction

        gate_pour_binary = (
            (self._mouth_xy_distance < self.cfg.pour_binary_mouth_xy_thresh)
            & (self._mouth_z_clearance > self.cfg.pour_binary_mouth_z_min)
            & (self._mouth_z_clearance < self.cfg.pour_binary_mouth_z_max)
            & (self._source_up_dot_world < self.cfg.pour_binary_tilt_thresh)
        ).float()

        r_pour_align = 0.5 * (self._mouth_alignment_cos + 1.0)
        r_pour_stage = gate_pour_binary * (
            self.cfg.weight_cross * r_cross
            + self.cfg.weight_capture * r_capture
            + self.cfg.weight_pour_align * r_pour_align
        )

        # 첫 비드 유입 시 1회성 보너스로 탐색을 유도
        first_capture = gate_pour_binary.bool() & (self._bead_in_target_fraction > 0.0) & (~self._first_capture_bonus_paid)
        r_first_capture = self.cfg.weight_first_capture_bonus * first_capture.float()
        self._first_capture_bonus_paid |= first_capture

        # tilt onset bonus: >60° 기울기 + 근접 조건 첫 달성 시 1회 bridge reward
        # "안전한 직립 유지" 로컬 최적을 벗어나 tilt 탐색을 유도
        tilt_onset = (
            (self._source_up_dot_world < self.cfg.tilt_onset_dot_threshold)
            & (self._mouth_xy_distance < self.cfg.tilt_onset_dist_threshold)
            & (~self._tilt_onset_bonus_paid)
        )
        r_tilt_onset = self.cfg.weight_tilt_onset_bonus * tilt_onset.float()
        self._tilt_onset_bonus_paid |= tilt_onset

        # 마지막 step까지 pour pose를 유지하면 종료 시 명확한 자세 신호를 준다.
        is_last_step = self.episode_length_buf >= (self.max_episode_length - 1)
        in_terminal_pour_pose = (
            (self._mouth_xy_distance < self.cfg.terminal_pour_mouth_xy_thresh)
            & (self._mouth_z_clearance > self.cfg.terminal_pour_mouth_z_min)
            & (self._mouth_z_clearance < self.cfg.terminal_pour_mouth_z_max)
            & (self._source_up_dot_world < self.cfg.terminal_pour_tilt_thresh)
        )
        r_terminal_pour = self.cfg.weight_terminal_pour * is_last_step.float() * in_terminal_pour_pose.float()

        # [v4 신규] 에피소드 종료(truncated 또는 source about to drain) 시 최종 capture 보너스
        # success_flag로 즉시 종료를 제거했으므로 에피소드 끝까지 채울수록 더 큰 보상을 받는다.
        # _source_empty_steps는 _get_dones에서 업데이트되므로 여기서는 hold_steps-1로 검출
        is_source_ending = self._source_empty_steps >= (self.cfg.source_empty_hold_steps - 1)
        is_episode_ending = is_last_step | is_source_ending
        r_terminal_capture = (
            self.cfg.weight_terminal_capture
            * self._bead_in_target_fraction
            * is_episode_ending.float()
        )

        # ---- Outcome and costs ----
        # ADR: success 기준을 낮은 fill_ratio에서 시작해 점진적으로 상향
        success_fill_ratio = (
            self.success_adr.get_param("success", "fill_ratio")
            if self.success_adr is not None
            else self.cfg.success_target_fill_ratio
        )

        success_now = (
            (self._bead_in_target_fraction >= success_fill_ratio)
            & (self._spill_ratio <= self.cfg.success_spill_max)
        )
        success_by_pose_and_fill = gate_pour_binary.bool() & success_now

        # 기본 성공 보상(바이너리)
        r_success = success_by_pose_and_fill.float()

        final_success_strict = (
            gate_pour_binary.bool()
            & (self._bead_in_target_fraction >= self.cfg.final_success_target_fill_ratio)
            & (self._spill_ratio <= self.cfg.final_success_spill_max)
        )

        # 성공 기준을 넘은 양에 비례한 오버필 보너스 (옵션)
        overfill_bonus = 0.0
        if self.cfg.weight_success_overfill > 0.0:
            overfill = torch.clamp(
                (self._bead_in_target_fraction - success_fill_ratio)
                / (1.0 - success_fill_ratio + 1e-6),
                min=0.0,
                max=1.0,
            )
            overfill_bonus = self.cfg.weight_success_overfill * overfill
        spill_cost = self._spill_ratio
        # bead count ADR: success/spill 가중치를 active bead 수에 비례 스케일
        if self.cfg.enable_bead_count_adr:
            _active_f = self._active_bead_count.float()  # (N,)
            spill_weight = self.cfg.weight_spill_per_bead * _active_f
            _success_weight = self.cfg.weight_success_per_bead * _active_f
        else:
            spill_weight = (
                self.spill_adr.get_param("reward", "spill_weight")
                if self.spill_adr is not None
                else self.cfg.weight_spill
            )
            _success_weight = self.cfg.weight_success
        
        significant_tilt = torch.clamp(
            self.cfg.tilt_onset_dot_threshold - self._source_up_dot_world,
            min=0.0,
        )
        premature_tilt_cost = (1.0 - self._g_ready) * significant_tilt
        palm_delta   = (self.actions[:, :6] - self.prev_actions[:, :6]).pow(2).sum(dim=-1)
        finger_delta = (self.actions[:, 6:] - self.prev_actions[:, 6:]).pow(2).sum(dim=-1)
        action_rate_penalty = (
            self.cfg.weight_action_rate_palm   * palm_delta
            + self.cfg.weight_action_rate_finger * finger_delta
        )

        arm_qd = self.robot.data.joint_vel[:, self.arm_dof_indices]
        arm_qd_l2 = arm_qd.norm(dim=-1)                          # (num_envs,) L2 norm
        arm_qd_max = arm_qd.abs().max(dim=-1).values              # (num_envs,) per-joint max
        arm_qacc = (arm_qd - self._prev_arm_joint_vel).norm(dim=-1)  # (num_envs,) acc proxy
        # pouring phase에서의 arm vel (cup이 가까울 때)
        in_tilt_phase = (self._mouth_xy_distance < self.cfg.tilt_action_gate_xy_far).float()
        tilt_phase_arm_vel = (arm_qd_l2 * in_tilt_phase).sum() / (in_tilt_phase.sum() + 1e-6)

        # [Phase-1 Step 4] arm joint vel^2 sum penalty (clipped)
        arm_qd_sq_sum = arm_qd.pow(2).sum(dim=-1).clamp_max(self.cfg.arm_joint_vel_sq_clip)
        arm_acc_vec = arm_qd - self._prev_arm_joint_vel          # 현재 acc 벡터 (N, DOF)
        arm_qacc_sq_sum = arm_acc_vec.pow(2).sum(dim=-1)

        # [Phase-1 Step 6] jerk = d(acc)/dt (acc 벡터 변화량)
        arm_jerk_vec = arm_acc_vec - self._prev_arm_joint_acc    # (N, DOF)
        arm_jerk_sq_sum = arm_jerk_vec.pow(2).sum(dim=-1)

        approach_only = 1.0 - in_tilt_phase   # approach 구간 (cup 멀리 있을 때)
        if self.cfg.arm_vel_tilt_gate_only:
            arm_vel_cost = (
                self.cfg.weight_arm_joint_vel  * arm_qd_sq_sum
                + self.cfg.weight_arm_joint_acc  * arm_qacc_sq_sum
                + self.cfg.weight_arm_joint_jerk * arm_jerk_sq_sum
            ) * in_tilt_phase
        else:
            arm_vel_cost = (
                self.cfg.weight_arm_joint_vel  * arm_qd_sq_sum
                + self.cfg.weight_arm_joint_acc  * arm_qacc_sq_sum
                + self.cfg.weight_arm_joint_jerk * arm_jerk_sq_sum
            ) * in_tilt_phase + self.cfg.weight_arm_joint_vel_approach * arm_qd_sq_sum * approach_only

        total = (
            r_hold
            + r_lift
            + r_approach
            + r_prepour_stage
            + r_pour_stage
            + r_first_capture
            + r_tilt_onset
            + r_terminal_pour
            + r_terminal_capture         # [v4 신규] 에피소드 종료 시 capture 보너스
            + _success_weight * r_success
            + overfill_bonus
            - spill_weight * spill_cost
            - self.cfg.weight_premature_tilt * premature_tilt_cost
            - action_rate_penalty
            - arm_vel_cost
        )

        self._prev_mouth_xy_distance.copy_(self._mouth_xy_distance)
        self._prev_arm_joint_vel.copy_(arm_qd)
        self._prev_arm_joint_acc.copy_(arm_acc_vec)   # [Step 6] jerk 계산용

        # ---- ADR increment (슬라이딩 윈도우 성공률 기반) ----
        _win_len = len(self._success_window)
        _ep_success_rate = sum(self._success_window) / max(_win_len, 1)

        # bead count ADR: 80% 성공 시 stage 진급 (윈도우가 충분히 찼을 때만 판정)
        if (
            self.cfg.enable_bead_count_adr
            and _win_len >= min(50, self.cfg.bead_count_adr_window_size)
            and _ep_success_rate >= self.cfg.bead_count_adr_trigger_threshold
            and self._bead_adr_stage_idx < len(self._bead_count_stages) - 1
        ):
            self._bead_adr_stage_idx += 1
            self._success_window.clear()  # 새 stage 기준으로 재측정

        if self.spill_adr is not None:
            self.spill_adr.maybe_increment(_ep_success_rate)
        if self.noise_adr is not None:
            self.noise_adr.maybe_increment(_ep_success_rate)
        if self.success_adr is not None:
            self.success_adr.maybe_increment(_ep_success_rate)

        # ---- Logging to TensorBoard ----
        # ── Reward stages ──────────────────────────────────────────────────
        self.extras["r_hold"]     = r_hold.mean()
        self.extras["r_approach"] = r_approach.mean()
        self.extras["r_prepour"]  = r_prepour_stage.mean()
        self.extras["r_pour"]     = r_pour_stage.mean()
        self.extras["r_terminal_capture"] = r_terminal_capture.mean()

        # ── Pour quality ────────────────────────────────────────────────────
        self.extras["gate_pour_binary"] = gate_pour_binary.mean()   # pour gate 활성 환경 비율
        self.extras["mouth_xy_dist"]    = self._mouth_xy_distance.mean()
        self.extras["spill_ratio"]      = self._spill_ratio.mean()
        self.extras["g_ready"]          = self._g_ready.mean()
        self.extras["env_frac_near"]    = (self._mouth_xy_distance < 0.08).float().mean()

        # ── 핵심 bead 성과 지표 ─────────────────────────────────────────────
        # bead_score = Σ(성공 환경의 active_bead_count) / 전체 환경 수
        # 해석: 평균적으로 환경당 몇 개의 bead가 성공적으로 들어가 있는가
        bead_score = (self.success_flag.float() * self._active_bead_count.float()).mean()
        self.extras["bead_score"] = bead_score

        # ── ADR 상태 ────────────────────────────────────────────────────────
        self.extras["bead_adr_stage"]        = torch.tensor(float(self._bead_adr_stage_idx), device=self.device)
        self.extras["bead_adr_success_rate"] = torch.tensor(_ep_success_rate, device=self.device)
        if self.success_adr is not None:
            self.extras["success_fill_ratio"] = torch.tensor(float(success_fill_ratio), device=self.device)

        # ── [v4] 궤적 버퍼 상태 ──────────────
        if self.success_trajectory_buffer is not None:
            self.extras["traj_buffer_size"] = torch.tensor(
                float(len(self.success_trajectory_buffer)), device=self.device
            )
            self.extras["traj_buffer_warm"] = torch.tensor(
                float(self.success_trajectory_buffer.is_warm(self.cfg.bc_min_buffer_size)),
                device=self.device,
            )

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
        fallen = self.object_pos[:, 2] < self.cfg.obj_fallen_z  # 컵이 테이블 아래로 낙하
        no_tip_force = self.contact_force_raw.max(dim=-1).values <= CONTACT_FORCE_THRESHOLD
        drop_force_active = (
            (~torch.full_like(no_tip_force, self._warmstart_collect_mode, dtype=torch.bool))
            & (self.episode_length_buf >= self.cfg.episode_hold_steps)
        )
        no_tip_force = no_tip_force & drop_force_active
        self._no_tip_force_steps = torch.where(
            no_tip_force,
            self._no_tip_force_steps + 1,
            torch.zeros_like(self._no_tip_force_steps),
        )
        dropped_by_force = drop_force_active & (self._no_tip_force_steps >= self.cfg.drop_force_hold_steps)

        # 소스 컵이 비어있는 상태가 source_empty_hold_steps 연속 지속되면 종료.
        # hold 버퍼를 두는 이유: 비드가 공중에 있는 동안 종료하면 타겟 컵 착지 전에 에피소드가
        # 끝나 capture 집계가 누락될 수 있음 → 30 steps(0.5s) 대기 후 최종 판정.
        source_empty_now = (
            (self._bead_in_source_fraction < 0.05)
            & (self.episode_length_buf >= self.cfg.episode_hold_steps)
            & (~torch.full_like(self.success_flag, self._warmstart_collect_mode))
        )
        self._source_empty_steps = torch.where(
            source_empty_now,
            self._source_empty_steps + 1,
            torch.zeros_like(self._source_empty_steps),
        )
        source_drained = source_empty_now & (self._source_empty_steps >= self.cfg.source_empty_hold_steps)

        success_fill_ratio = (
            self.success_adr.get_param("success", "fill_ratio")
            if self.success_adr is not None
            else self.cfg.success_target_fill_ratio
        )

        success_now = (
            (self._bead_in_target_fraction >= success_fill_ratio)
            & (self._spill_ratio <= self.cfg.success_spill_max)
        )
        gate_pour_binary = (
            (self._mouth_xy_distance < self.cfg.pour_binary_mouth_xy_thresh)
            & (self._mouth_z_clearance > self.cfg.pour_binary_mouth_z_min)
            & (self._mouth_z_clearance < self.cfg.pour_binary_mouth_z_max)
            & (self._source_up_dot_world < self.cfg.pour_binary_tilt_thresh)
        )
        # [v3 수정] 안정적 파지(grasped) 상태일 때만 성공으로 인정하여 우연한 낙하 성공 배제
        grasped = self.num_contacts_buf >= MIN_CONTACTS_FOR_SUCCESS
        success_by_pose_and_fill = gate_pour_binary.bool() & success_now & grasped

        self.success_flag.copy_(success_by_pose_and_fill)
        self.episode_success_buf |= self.success_flag   # 에피소드 중 한 번이라도 성공 시 True
        self._maybe_store_warmstart_successes()

        # [v4 수정] success_flag를 terminated에서 제거: 성공해도 에피소드를 계속 진행하여
        # 더 많은 bead를 넣을 기회를 준다. source_drained (bead 모두 소스에서 나감) 또는
        # truncated (시간 초과)로만 종료. terminal capture bonus가 "많이 넣을수록 좋다"를 보상.
        terminated = out_x | out_y | fallen | dropped_by_force | source_drained
        truncated  = self.episode_length_buf >= self.max_episode_length - 1

        # [v4 Phase2] 종료 원인별 로깅
        self.extras["term_frac_success_flag"]   = self.success_flag.float().mean()   # 성공 조건 달성 비율 (종료 안 함, bead_score와 함께 해석)
        self.extras["term_frac_source_drained"] = source_drained.float().mean()      # source 고갈 종료 비율 (pour 완료)
        self.extras["term_frac_dropped"]        = dropped_by_force.float().mean()    # grasp 실패 종료 비율
        # mean_episode_length 제거: rl_games가 episode_lengths/iter로 동일 값을 자동 로깅함

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

        # [v4] 에피소드 종료 env 의 궤적을 성공 버퍼에 저장 (리셋 전에 수행)
        if self._trajectory_capture is not None:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._finalize_episode_trajectories(env_ids_t)
            self._trajectory_capture.reset_envs(env_ids_t)

        n = len(env_ids)

        # ---- episode 성공 집계 후 클리어 (슬라이딩 윈도우) ----
        for _s in self.episode_success_buf[env_ids].tolist():
            self._success_window.append(int(_s))
        self.episode_success_buf[env_ids] = False
        self._warmstart_only_close[env_ids] = False
        self._warmstart_finger_action_floor[env_ids] = -1.0
        # collect mode 안정 카운터 리셋 (에피소드 종료 = 새 grasp 시작)
        self._warmstart_stable_steps[env_ids] = 0
        if not self._warmstart_collect_mode:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            reset_draw = torch.rand(len(env_ids_t), device=self.device)
            grasp_mask = torch.zeros(len(env_ids_t), dtype=torch.bool, device=self.device)

            if self._warmstart_cache_count > 0:
                grasp_mask = reset_draw < self.cfg.grasp_warmstart_reset_ratio

            if torch.any(grasp_mask):
                self._reset_from_warmstart_cache(env_ids_t[grasp_mask])

            env_ids = env_ids_t[~grasp_mask]
            n = len(env_ids)
            if n == 0:
                return

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

        # ---- 3. 컵 spawn 위치 계산 (±0.06m 랜덤) ----
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

        # ---- 4. FABRICS pregrasp rollout ----
        noise = torch.stack([
            (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_x,
            (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_y,
            (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_z,
        ], dim=1)
        pregrasp_pos = obj_pos_local + self.pregrasp_offset.unsqueeze(0) + noise

        pregrasp_palm_pose_euler = torch.zeros(n, 6, device=self.device)
        pregrasp_palm_pose_euler[:, :3] = pregrasp_pos
        pregrasp_palm_pose_euler[:, 3] = math.radians(90.0)
        pregrasp_palm_pose_euler[:, 4] = math.radians(0.0)
        pregrasp_palm_pose_euler[:, 5] = math.radians(90.0)
        pregrasp_palm_pose_euler = torch.max(
            torch.min(pregrasp_palm_pose_euler, self.palm_maxs.unsqueeze(0)),
            self.palm_mins.unsqueeze(0),
        )
        pregrasp_palm_pose = torch.zeros(n, 7, device=self.device)
        pregrasp_palm_pose[:, :3] = pregrasp_palm_pose_euler[:, :3]
        pregrasp_palm_pose[:, 3:7] = self._quat_xyzw_from_euler_zyx(pregrasp_palm_pose_euler[:, 3:6])

        # ---- cache lookup: spawn 위치 → 가장 가까운 grid point arm IK ----
        xi = ((obj_x - self._cache_xs[0]) / (self._cache_xs[1] - self._cache_xs[0])).round().long().clamp(0, self._cache_n - 1)
        yi = ((obj_y - self._cache_ys[0]) / (self._cache_ys[1] - self._cache_ys[0])).round().long().clamp(0, self._cache_n - 1)
        q_pregrasp = self.fabric_q[env_ids].clone()
        q_pregrasp[:, :NUM_ARM_DOF] = self._cache_q_arm[xi, yi]

        self.fabric_q[env_ids] = q_pregrasp
        self.fabric_qd[env_ids].zero_()
        self.fabric_qdd[env_ids].zero_()

        # hand는 APPROACH_POSE로 강제
        approach_hand = self.hand_open_pose.unsqueeze(0).expand(n, -1)
        self.fabric_q[env_ids, NUM_ARM_DOF:] = approach_hand
        self.fabric_qd[env_ids, NUM_ARM_DOF:].zero_()

        # ---- 5. pregrasp / prelift 버퍼 저장 ----
        self.pregrasp_arm_pos_buf[env_ids] = q_pregrasp[:, :NUM_ARM_DOF]

        # palm_pose_targets를 pregrasp로 동기화 (첫 Fabrics 스텝 타겟 일관성)
        self.palm_pose_targets[env_ids] = pregrasp_palm_pose

        # delta action 기준점: action=0 → pregrasp 위치 유지
        self.pregrasp_palm_pose_buf[env_ids] = pregrasp_palm_pose
        self._grasp_rel_palm_to_cup_init[env_ids] = obj_pos_local - pregrasp_palm_pose[:, :3]
        self._grasp_cup_height_init[env_ids] = obj_pos_local[:, 2]

        # Fabrics cspace attractor(null-space)를 pregrasp arm pos로 설정
        # default_config가 ARM_START_POSE이면 null-space 항이 계속 팔을 당겨 초기 흔들림 발생
        # pregrasp arm pos로 설정 → 에피소드 시작 시 null-space 항 ≈ 0 → 안정
        self.open_tesollo_fabric.default_config[env_ids, :NUM_ARM_DOF] = q_pregrasp[:, :NUM_ARM_DOF]

        self.grasp_hold_hand_pos_buf[env_ids] = approach_hand

        # ---- 6. 로봇 pregrasp 자세로 초기화 ----
        pregrasp_full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        pregrasp_full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)
        pregrasp_full_pos[:, self.arm_dof_indices]  = q_pregrasp[:, :NUM_ARM_DOF]
        pregrasp_full_pos[:, self.hand_dof_indices] = approach_hand
        pregrasp_full_pos[:, self.left_arm_dof_indices] = self.left_arm_zero_pos[0]
        self.robot.write_joint_state_to_sim(pregrasp_full_pos, pregrasp_full_vel, env_ids=env_ids)

        # ---- 7. 컵 spawn ----
        obj_pos_world = obj_pos_local + self.scene.env_origins[env_ids]
        upright_rot = torch.zeros(n, 4, device=self.device)
        upright_rot[:, 0] = 1.0
        zero_vel = torch.zeros(n, 6, device=self.device)
        cup_root_state = torch.cat([obj_pos_world, upright_rot, zero_vel], dim=-1)
        self.cup.write_root_state_to_sim(cup_root_state, env_ids=env_ids)

        left_cup_pose = self._compute_attached_root_pose(
            self._left_target_cup_body_id,
            self._left_target_cup_attach_pos_b,
            self._left_target_cup_attach_quat_b,
            env_ids=env_ids,
        )
        left_cup_pose[:, 2] += self.cfg.left_cup_world_z_offset
        self._left_target_cup_fixed_pose_w[env_ids] = left_cup_pose
        self.left_target_cup.write_root_pose_to_sim(left_cup_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        if self._warmstart_collect_mode:
            self._hide_beads(env_ids)
        else:
            # 현재 ADR stage의 active bead count 적용 후 spawn
            _target_count = self._bead_count_stages[self._bead_adr_stage_idx]
            self._active_bead_count[env_ids] = _target_count
            _bead_idx = torch.arange(self.num_beads, device=self.device).unsqueeze(0)
            self._active_bead_mask[env_ids] = _bead_idx < _target_count
            bead_state = self._spawn_beads_with_active_mask(cup_root_state[:, :7], env_ids)
            self.beads.write_object_state_to_sim(bead_state, env_ids=env_ids)

        # ---- 8. 버퍼 리셋 ----
        self.hand_joint_targets[env_ids] = approach_hand
        self.contact_force_raw[env_ids].zero_()
        self.binary_contact_buf[env_ids] = False
        self.num_contacts_buf[env_ids]   = 0
        self.distal_contact_force_raw[env_ids].zero_()
        self.distal_binary_contact_buf[env_ids] = False
        self.middle_contact_force_raw[env_ids].zero_()
        self.middle_binary_contact_buf[env_ids] = False
        self._bead_in_target[env_ids] = False
        self._bead_in_source[env_ids] = False
        self._bead_ever_in_target[env_ids] = False
        self._bead_crossed_target_mouth[env_ids] = False
        self._prev_bead_target_local_z[env_ids].fill_(10.0)
        self._needs_grasp_init_update[env_ids] = True   # 다음 스텝에 palm local init 갱신
        self._bead_cross_count[env_ids] = 0
        self._bead_cross_fraction[env_ids] = 0.0
        self._prev_bead_ever_in_target_count[env_ids] = 0
        self._bead_in_target_fraction[env_ids] = 0.0
        self._bead_in_source_fraction[env_ids] = 0.0
        self._bead_centroid_w[env_ids].zero_()
        self._spill_ratio[env_ids] = 0.0
        self._all_beads_bonus_paid[env_ids] = False
        self._first_capture_bonus_paid[env_ids] = False
        self._tilt_onset_bonus_paid[env_ids] = False
        self._no_tip_force_steps[env_ids] = 0
        self._source_empty_steps[env_ids] = 0
        self.success_flag[env_ids] = False
        self._pre_pour_ready_steps[env_ids] = 0
        self._prev_arm_joint_vel[env_ids].zero_()
        self._prev_arm_joint_acc[env_ids].zero_()   # [Step 6]
        self._ema_palm_action[env_ids].zero_()       # [Step 7]

        # actions 리셋: delta action 방식 → action=0 = pregrasp 위치
        # (역스케일 불필요: scale(0, delta_mins, delta_maxs) = delta=0 → pregrasp 유지)
        self.actions[env_ids, :6] = 0.0
        self.actions[env_ids, 6:] = -1.0
        self.prev_actions[env_ids, :6] = 0.0
        self.prev_actions[env_ids, 6:] = -1.0
        self._prev_mouth_xy_distance[env_ids] = 0.0


    def _resolve_attachment_body(self, requested_body_name: str, attach_pos_b: torch.Tensor) -> tuple[int, torch.Tensor]:
        body_names = self.robot.data.body_names
        alias_offsets: dict[str, list[tuple[str, tuple[float, float, float]]]] = {
            "openarm_left_hand": [
                ("openarm_left_hand", (0.0, 0.0, 0.0)),
                ("openarm_left_hand_tcp", (0.0, 0.0, -0.08)),
                ("ll_dg_ee", (0.0, 0.0, -0.08)),
            ],
            "ll_dg_ee": [
                ("ll_dg_ee", (0.0, 0.0, 0.0)),
                ("openarm_left_hand_tcp", (0.0, 0.0, -0.08)),
                ("openarm_left_hand", (0.0, 0.0, 0.0)),
            ],
        }
        candidates = alias_offsets.get(requested_body_name, [(requested_body_name, (0.0, 0.0, 0.0))])
        for body_name, desired_origin_in_body in candidates:
            if body_name in body_names:
                resolved_pos_b = attach_pos_b + torch.tensor(
                    desired_origin_in_body, dtype=attach_pos_b.dtype, device=attach_pos_b.device
                )
                return body_names.index(body_name), resolved_pos_b
        raise ValueError(f"Attachment frame '{requested_body_name}' was not found.")

    def _get_left_target_cup_fixed_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            return self._left_target_cup_fixed_pose_w
        return self._left_target_cup_fixed_pose_w[env_ids]

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

    def _sample_bead_states_inside_cup(self, cup_pose: torch.Tensor) -> torch.Tensor:
        cup_pos_w = cup_pose[:, :3]
        cup_quat_w = cup_pose[:, 3:7]
        n = cup_pose.shape[0]
        base_offset = self._bead_spawn_pos_source_cup_b.unsqueeze(0).unsqueeze(1).expand(n, self.num_beads, -1)
        local_offsets = base_offset + self._bead_offsets_source_cup_b.unsqueeze(0).expand(n, -1, -1)
        cup_quat_expanded = cup_quat_w.unsqueeze(1).expand(-1, self.num_beads, -1)
        bead_pos_w = cup_pos_w.unsqueeze(1) + quat_apply(
            cup_quat_expanded.reshape(-1, 4),
            local_offsets.reshape(-1, 3),
        ).reshape(n, self.num_beads, 3)
        bead_quat_w = quat_mul(
            cup_quat_expanded.reshape(-1, 4),
            self._bead_spawn_quat_source_cup.unsqueeze(0).unsqueeze(1).expand(n, self.num_beads, -1).reshape(-1, 4),
        ).reshape(n, self.num_beads, 4)
        bead_state = torch.zeros(n, self.num_beads, 13, device=self.device)
        bead_state[..., :3] = bead_pos_w
        bead_state[..., 3:7] = bead_quat_w
        return bead_state

    def _spawn_beads_with_active_mask(
        self, cup_pose: torch.Tensor, env_ids: Sequence[int]
    ) -> torch.Tensor:
        """active bead는 컵 내부에, inactive bead는 hidden parking grid에 배치."""
        n = len(env_ids)
        active_counts = self._active_bead_count[env_ids]  # (n,)

        # 모든 bead의 컵-내부 위치 계산 (n, num_beads, 13)
        all_states = self._sample_bead_states_inside_cup(cup_pose)

        # hidden 위치: env_origin + local offset (n, num_beads, 3)
        hidden_pos_w = (
            self.scene.env_origins[env_ids].unsqueeze(1)      # (n, 1, 3)
            + self._hidden_bead_offsets_b.unsqueeze(0)         # (1, num_beads, 3)
        )

        # inactive mask: bead i >= active_count → hidden으로 교체
        bead_idx = torch.arange(self.num_beads, device=self.device).unsqueeze(0)  # (1, num_beads)
        inactive = bead_idx >= active_counts.unsqueeze(1)                          # (n, num_beads)

        all_states[..., :3] = torch.where(
            inactive.unsqueeze(-1).expand(n, self.num_beads, 3),
            hidden_pos_w,
            all_states[..., :3],
        )
        # inactive bead 속도 0 보장
        all_states[..., 7:] = torch.where(
            inactive.unsqueeze(-1).expand(n, self.num_beads, 6),
            torch.zeros_like(all_states[..., 7:]),
            all_states[..., 7:],
        )
        return all_states

    def _hide_beads(self, env_ids: Sequence[int]) -> None:
        n = len(env_ids)
        bead_state = torch.zeros(n, self.num_beads, 13, device=self.device)
        bead_state[..., 2] = -10.0
        bead_state[..., 3] = 1.0
        self.beads.write_object_state_to_sim(bead_state, env_ids=env_ids)

    def _build_warmstart_reset_cache(self) -> None:
        if not self.cfg.enable_warmstart_reset:
            return

        ckpt = self.cfg.warmstart_checkpoint_path
        if not ckpt:
            return

        try:
            self._warmstart_policy = _WarmstartPolicy(ckpt, self.device).to(self.device)
        except Exception as exc:
            print(f"[5g_pour_right_v3] warmstart policy load failed: {exc}", flush=True)
            self._warmstart_policy = None
            return

        obs_noise_joint_pos = self.cfg.obs_noise_joint_pos
        obs_noise_joint_vel = self.cfg.obs_noise_joint_vel
        obs_noise_body_pos = self.cfg.obs_noise_body_pos
        obs_noise_cup_pos = self.cfg.obs_noise_cup_pos
        self.cfg.obs_noise_joint_pos = 0.0
        self.cfg.obs_noise_joint_vel = 0.0
        self.cfg.obs_noise_body_pos = 0.0
        self.cfg.obs_noise_cup_pos = 0.0
        self._warmstart_collect_mode = True

        try:
            self.reset()
            with torch.no_grad():
                for _ in range(int(self.cfg.warmstart_max_rollout_steps)):
                    if self._warmstart_cache_count >= self._warmstart_arm_pos.shape[0]:
                        break
                    actions = self._warmstart_policy(self._get_legacy_warmstart_policy_obs())
                    self.step(actions)
        finally:
            self._warmstart_collect_mode = False
            self.cfg.obs_noise_joint_pos = obs_noise_joint_pos
            self.cfg.obs_noise_joint_vel = obs_noise_joint_vel
            self.cfg.obs_noise_body_pos = obs_noise_body_pos
            self.cfg.obs_noise_cup_pos = obs_noise_cup_pos

        if self._warmstart_cache_count == 0:
            raise RuntimeError(
                "[5g_pour_right_v3] warmstart cache is empty. "
                "The v7 checkpoint rollout did not produce any lift-success state, so this task cannot start "
                "from the requested play-like grasp state."
            )

        print(
            f"[5g_pour_right_v3] collected {self._warmstart_cache_count} warmstart success states.",
            flush=True,
        )

    def _maybe_store_warmstart_successes(self) -> None:
        """collect mode 에서 '확실하게 lift' 된 env 만 캐시에 저장.

        저장 조건 (모두 충족해야 함):
          1. lift height  >= warmstart_cache_min_lift_height   (기본 0.15m: 확실히 들린 상태)
          2. contacts     >= warmstart_cache_min_contacts       (기본 3개: 견고한 파지)
          3. upright      source up-axis z > 0.85              (컵이 수직 유지)
          4. not_falling  cup z-velocity  >= -0.02 m/s         (낙하 중이 아님)
          5. stable_hold  위 조건을 연속 warmstart_stable_hold_steps 프레임 유지
        """
        if not self._warmstart_collect_mode:
            return
        if self._warmstart_cache_count >= self._warmstart_arm_pos.shape[0]:
            return

        # ── 조건 1~4: 이번 프레임 품질 기준 ────────────────────────────────
        min_lift   = self.cfg.warmstart_cache_min_lift_height
        min_ct     = self.cfg.warmstart_cache_min_contacts
        stable_req = self.cfg.warmstart_stable_hold_steps

        lifted      = self.object_pos[:, 2] > (self.object_init_pos[:, 2] + min_lift)
        grasped     = self.num_contacts_buf >= min_ct
        upright     = self._source_up_axis_w[:, 2] > 0.85
        not_falling = self.cup.data.root_link_lin_vel_w[:, 2] >= -0.02
        quality_ok  = lifted & grasped & upright & not_falling

        # ── 조건 5: 연속 안정 카운터 갱신 ──────────────────────────────────
        self._warmstart_stable_steps = torch.where(
            quality_ok,
            self._warmstart_stable_steps + 1,
            torch.zeros_like(self._warmstart_stable_steps),
        )

        # stable_req 프레임 연속 유지된 env 만 저장 후보
        stable_ok = self._warmstart_stable_steps >= stable_req
        success_env_ids = stable_ok.nonzero(as_tuple=False).squeeze(-1)
        if success_env_ids.numel() == 0:
            return

        # 이미 저장된 env 가 다음 스텝에 또 저장되지 않도록 카운터 리셋
        self._warmstart_stable_steps[success_env_ids] = 0

        remaining = self._warmstart_arm_pos.shape[0] - self._warmstart_cache_count
        success_env_ids = success_env_ids[:remaining]
        count = success_env_ids.numel()
        if count == 0:
            return

        start = self._warmstart_cache_count
        end   = start + count
        self._warmstart_arm_pos[start:end]       = self.robot.data.joint_pos[success_env_ids][:, self.arm_dof_indices]
        self._warmstart_hand_pos[start:end]      = self.robot.data.joint_pos[success_env_ids][:, self.hand_dof_indices]
        self._warmstart_palm_pose[start:end]     = self.palm_pose_targets[success_env_ids]
        self._warmstart_cup_pose[start:end, :3]  = (
            self.cup.data.root_pos_w[success_env_ids] - self.scene.env_origins[success_env_ids]
        )
        self._warmstart_cup_pose[start:end, 3:7] = self.cup.data.root_quat_w[success_env_ids]
        self._warmstart_cache_count = end

    def _reset_from_cached_state(
        self,
        env_ids: Sequence[int],
        arm_pos: torch.Tensor,
        hand_pos: torch.Tensor,
        palm_pose: torch.Tensor,
        cup_pose_local: torch.Tensor,
        *,
        left_cup_pose_local: torch.Tensor | None = None,
        bead_state_local: torch.Tensor | None = None,
        active_bead_count: torch.Tensor | None = None,
        object_init_z_override: float | None = None,
    ) -> None:
        n = len(env_ids)
        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_pos[:, self.arm_dof_indices] = arm_pos
        full_pos[:, self.hand_dof_indices] = hand_pos
        full_pos[:, self.left_arm_dof_indices] = self.left_arm_zero_pos[0]
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        self.fabric_q[env_ids].zero_()
        self.fabric_q[env_ids, :NUM_ARM_DOF] = arm_pos
        self.fabric_q[env_ids, NUM_ARM_DOF:] = hand_pos
        self.fabric_qd[env_ids].zero_()
        self.fabric_qdd[env_ids].zero_()

        self.pregrasp_arm_pos_buf[env_ids] = arm_pos
        self.grasp_hold_hand_pos_buf[env_ids] = hand_pos
        delta_20 = (self.hand_grasp_pose - self.hand_open_pose).unsqueeze(0).expand(n, -1)
        open_20 = self.hand_open_pose.unsqueeze(0).expand(n, -1)
        valid = torch.abs(delta_20) > 1e-6
        t_20 = torch.where(valid, (hand_pos - open_20) / delta_20, torch.zeros_like(hand_pos))
        t_20 = t_20.clamp(0.0, 1.0)
        valid_f = valid.view(n, NUM_FINGER_ACTION, 4).float()
        t_f = t_20.view(n, NUM_FINGER_ACTION, 4)
        valid_count = valid_f.sum(dim=-1)
        t_finger = torch.where(
            valid_count > 0,
            (t_f * valid_f).sum(dim=-1) / valid_count.clamp(min=1.0),
            torch.zeros_like(valid_count),
        )
        self._warmstart_finger_action_floor[env_ids] = (2.0 * t_finger - 1.0).clamp(-1.0, 1.0)
        self._warmstart_only_close[env_ids] = True

        cached_palm_pose = palm_pose.clone()
        cached_palm_pose[:, :3] = torch.max(
            torch.min(cached_palm_pose[:, :3], self.palm_maxs[:3].unsqueeze(0)),
            self.palm_mins[:3].unsqueeze(0),
        )
        self.pregrasp_palm_pose_buf[env_ids] = cached_palm_pose
        self.palm_pose_targets[env_ids] = cached_palm_pose
        self.hand_joint_targets[env_ids] = hand_pos
        self.object_init_pos[env_ids] = cup_pose_local[:, :3]
        if object_init_z_override is not None:
            self.object_init_pos[env_ids, 2] = object_init_z_override
        self._grasp_rel_palm_to_cup_init[env_ids] = cup_pose_local[:, :3] - palm_pose[:, :3]
        self._grasp_cup_height_init[env_ids] = cup_pose_local[:, 2]
        self.open_tesollo_fabric.default_config[env_ids, :NUM_ARM_DOF] = arm_pos

        cup_pose_world = cup_pose_local.clone()
        cup_pose_world[:, :3] += self.scene.env_origins[env_ids]
        zero_vel = torch.zeros(n, 6, device=self.device)
        self.cup.write_root_pose_to_sim(cup_pose_world, env_ids=env_ids)
        self.cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        # hold 기간 컵 고정 좌표 저장 (palm body 추적 오차 없는 절대 좌표)
        self._cup_hold_pos_w[env_ids]  = cup_pose_world[:, :3]
        self._cup_hold_quat_w[env_ids] = cup_pose_world[:, 3:7]

        if left_cup_pose_local is None:
            left_cup_pose = self._compute_attached_root_pose(
                self._left_target_cup_body_id,
                self._left_target_cup_attach_pos_b,
                self._left_target_cup_attach_quat_b,
                env_ids=env_ids,
            )
            left_cup_pose[:, 2] += self.cfg.left_cup_world_z_offset
        else:
            left_cup_pose = left_cup_pose_local.clone()
            left_cup_pose[:, :3] += self.scene.env_origins[env_ids]
        self._left_target_cup_fixed_pose_w[env_ids] = left_cup_pose
        self.left_target_cup.write_root_pose_to_sim(left_cup_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        if active_bead_count is None:
            active_bead_count = torch.full(
                (n,), self._bead_count_stages[self._bead_adr_stage_idx], dtype=torch.long, device=self.device
            )
        self._active_bead_count[env_ids] = active_bead_count
        _bead_idx = torch.arange(self.num_beads, device=self.device).unsqueeze(0)
        self._active_bead_mask[env_ids] = _bead_idx < active_bead_count.unsqueeze(1)

        if bead_state_local is None:
            bead_state = self._spawn_beads_with_active_mask(cup_pose_world, env_ids)
        else:
            bead_state = bead_state_local.clone()
            bead_state[..., :3] += self.scene.env_origins[env_ids].unsqueeze(1)
        self.beads.write_object_state_to_sim(bead_state, env_ids=env_ids)

        self.contact_force_raw[env_ids].zero_()
        self.binary_contact_buf[env_ids] = False
        self.num_contacts_buf[env_ids] = 0
        self.distal_contact_force_raw[env_ids].zero_()
        self.distal_binary_contact_buf[env_ids] = False
        self.middle_contact_force_raw[env_ids].zero_()
        self.middle_binary_contact_buf[env_ids] = False
        self._bead_in_target[env_ids] = False
        self._bead_in_source[env_ids] = False
        self._bead_ever_in_target[env_ids] = False
        self._bead_crossed_target_mouth[env_ids] = False
        self._prev_bead_target_local_z[env_ids].fill_(10.0)
        self._needs_grasp_init_update[env_ids] = True
        self._bead_cross_count[env_ids] = 0
        self._bead_cross_fraction[env_ids] = 0.0
        self._prev_bead_ever_in_target_count[env_ids] = 0
        self._bead_in_target_fraction[env_ids] = 0.0
        self._bead_in_source_fraction[env_ids] = 0.0
        self._bead_centroid_w[env_ids].zero_()
        self._spill_ratio[env_ids] = 0.0
        self._all_beads_bonus_paid[env_ids] = False
        self._first_capture_bonus_paid[env_ids] = False
        self._tilt_onset_bonus_paid[env_ids] = False
        self._no_tip_force_steps[env_ids] = 0
        self._source_empty_steps[env_ids] = 0
        self.success_flag[env_ids] = False
        self._prev_arm_joint_vel[env_ids].zero_()
        self._prev_arm_joint_acc[env_ids].zero_()
        self._ema_palm_action[env_ids].zero_()
        self.actions[env_ids, :6] = 0.0
        self.actions[env_ids, 6:] = 1.0
        self.prev_actions[env_ids, :6] = 0.0
        self.prev_actions[env_ids, 6:] = 1.0
        self._prev_mouth_xy_distance[env_ids] = 0.0
        self._pre_pour_ready_steps[env_ids] = 0
        self.success_flag[env_ids] = False

    # ------------------------------------------------------------------
    # [v4] Trajectory finalize helper
    # ------------------------------------------------------------------
    def _finalize_episode_trajectories(self, env_ids: torch.Tensor) -> None:
        """에피소드 종료 env 의 궤적을 성공 조건 검사 후 success_trajectory_buffer 에 저장.

        저장 기준:
          - bead_in_target_fraction >= cfg.trajectory_success_bead_threshold
          - spill_ratio < cfg.trajectory_success_spill_max
          - 캡처된 스텝 수 >= cfg.trajectory_min_steps
        """
        if self.success_trajectory_buffer is None:
            return
        for env_id in env_ids.tolist():
            count = self._trajectory_capture.get_count(env_id)
            if count < self.cfg.trajectory_min_steps:
                continue
            bead_frac  = float(self._bead_in_target_fraction[env_id].item())
            spill_frac = float(self._spill_ratio[env_id].item())
            if (
                bead_frac  >= self.cfg.trajectory_success_bead_threshold
                and spill_frac <= self.cfg.trajectory_success_spill_max
            ):
                traj  = self._trajectory_capture.get_trajectory(env_id)
                # active bead 수를 곱해 ADR 진행에 따라 score 자동 상향.
                # → 초반 우연성 궤적(bead=1, score≈1)은
                #   ADR 진행 후 실질 궤적(bead=5, score≈3.5)에 밀려남.
                active_count = float(self._active_bead_count[env_id].item())
                score = (bead_frac - 0.5 * spill_frac) * active_count
                self.success_trajectory_buffer.store(traj, score)

    def _reset_from_warmstart_cache(self, env_ids: Sequence[int]) -> None:
        n = len(env_ids)
        pick = torch.randint(self._warmstart_cache_count, (n,), device=self.device)
        arm_pos = self._warmstart_arm_pos[pick]
        hand_pos = self._warmstart_hand_pos[pick]
        palm_pose = self._warmstart_palm_pose[pick]
        cup_pose_local = self._warmstart_cup_pose[pick]
        self._reset_from_cached_state(
            env_ids,
            arm_pos,
            hand_pos,
            palm_pose,
            cup_pose_local,
            object_init_z_override=self.cfg.object_spawn_z,
        )

        if not self._warmstart_reset_debug_printed:
            warmstart_palm_pose = self.palm_pose_targets[env_ids]
            cup_pose_world = cup_pose_local.clone()
            cup_pose_world[:, :3] += self.scene.env_origins[env_ids]
            left_cup_pose = self._left_target_cup_fixed_pose_w[env_ids]
            source_pour_point_w = self._compute_dynamic_source_pour_point_w(
                cup_pose_world[:, :3],
                cup_pose_world[:, 3:7],
            )
            target_opening_w = left_cup_pose[:, :3] + quat_apply(
                left_cup_pose[:, 3:7],
                self._target_cup_opening_pos_b.unsqueeze(0).expand(n, -1),
            )
            mouth_delta = target_opening_w - source_pour_point_w
            mouth_xy_distance = torch.norm(mouth_delta[:, :2], dim=-1)
            mouth_z_clearance = source_pour_point_w[:, 2] - target_opening_w[:, 2]
            cup_z_local = cup_pose_local[:, 2]
            print(
                "[5g_pour_right_v3][warmstart_reset] "
                f"cup_z_local mean={cup_z_local.mean().item():.4f} "
                f"min={cup_z_local.min().item():.4f} max={cup_z_local.max().item():.4f} | "
                f"mouth_xy mean={mouth_xy_distance.mean().item():.4f} "
                f"min={mouth_xy_distance.min().item():.4f} max={mouth_xy_distance.max().item():.4f} | "
                f"mouth_z_clearance mean={mouth_z_clearance.mean().item():.4f} "
                f"min={mouth_z_clearance.min().item():.4f} max={mouth_z_clearance.max().item():.4f} | "
                f"warmstart_palm_z mean={warmstart_palm_pose[:, 2].mean().item():.4f}",
                flush=True,
            )
            self._warmstart_reset_debug_printed = True

