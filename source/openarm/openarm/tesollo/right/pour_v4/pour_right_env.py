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
from .pour_right_constants import (
    NUM_ARM_DOF,
    NUM_HAND_DOF,
    NUM_PALM_ACTION,
    NUM_FINGERTIPS,
    NUM_OBSERVATIONS,
    NUM_DISTAL_SENSORS,
    NUM_MIDDLE_SENSORS,
    NUM_CRITIC_OBSERVATIONS,
    CONTACT_FORCE_THRESHOLD,
    CONTACT_FORCE_MAX,
    MIN_CONTACTS_FOR_SUCCESS,
    ARM_START_POSE,
    DEMO_POUR_ARM_POSE,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
)
from .pour_adr import PourADR
from .pour_adr import PourADR as GraspADR
from .pour_right_preset import (
    BEAD_SPAWN_POS_SOURCE_CUP_B,
    BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ,
    LEFT_ARM_REST_JOINT_POS,
    LEFT_TARGET_CUP_POS_ENV_LOCAL,
    LEFT_TARGET_CUP_QUAT_WXYZ,
    RIGHT_ACTUATED_JOINT_NAMES,
    HAND_APPROACH_POSE,
    HAND_GRASP_POSE,
    OBJECT_GOAL_POS,
)
from .pour_right_utils import pour_corridor_score, scale, to_torch
from .demo_pose_reference import DemoPoseReferenceBank
from .warm_state_bank import PourWarmStateBank




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

    Action: 6D
      [0:6]  palm pose (x,y,z,ez,ey,ex), 정규화 [-1,1] → Fabrics IK

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
    def _quat_conjugate_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
        return torch.cat([quat_wxyz[:, :1], -quat_wxyz[:, 1:]], dim=-1)

    @staticmethod
    def _quat_angle_error_wxyz(quat_a_wxyz: torch.Tensor, quat_b_wxyz: torch.Tensor) -> torch.Tensor:
        quat_a_wxyz = torch.nn.functional.normalize(quat_a_wxyz, dim=-1)
        quat_b_wxyz = torch.nn.functional.normalize(quat_b_wxyz, dim=-1)
        dot = (quat_a_wxyz * quat_b_wxyz).sum(dim=-1).abs().clamp(0.0, 1.0)
        return 2.0 * torch.acos(dot)

    @staticmethod
    def _safe_normalize(vec: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(vec, dim=-1, keepdim=True)
        fallback_norm = torch.norm(fallback, dim=-1, keepdim=True).clamp(min=1e-6)
        fallback_unit = fallback / fallback_norm
        return torch.where(norm > 1e-6, vec / norm.clamp(min=1e-6), fallback_unit)

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

        source_pour_point_w = self.cup.data.root_pos_w + quat_apply(
            cup_quat_w,
            self._source_cup_pour_point_pos_b.unsqueeze(0).expand(n, -1),
        )
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
        # Palm action 범위.
        # xyz는 rim-pivot target의 one-step 이동량, rot는 current palm 기준 incremental 회전량.
        # action=0이면 현재 rim/palm pose를 유지하므로 큰 absolute target을 만들지 않는다.
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
        # warmstart 수집 시 delta는 v7-2 학습 조건과 일치해야 함:
        # palm_delta_rot_deg=20° (pour 120°와 다름)
        _warmstart_delta_rad = math.radians(cfg.warmstart_collect_palm_delta_rot_deg)
        self.delta_mins_warmstart_collect = to_torch([
            -cfg.warmstart_collect_palm_delta_xyz,
            -cfg.warmstart_collect_palm_delta_xyz,
            -cfg.warmstart_collect_palm_delta_xyz,
            -_warmstart_delta_rad, -_warmstart_delta_rad, -_warmstart_delta_rad,
        ], device=self.device)
        self.delta_maxs_warmstart_collect = to_torch([
            cfg.warmstart_collect_palm_delta_xyz,
            cfg.warmstart_collect_palm_delta_xyz,
            cfg.warmstart_collect_palm_delta_xyz,
            _warmstart_delta_rad, _warmstart_delta_rad, _warmstart_delta_rad,
        ], device=self.device)

        # pregrasp palm pose 버퍼 (reset/warmstart 동기화용; normal pour 회전 기준점은 current palm)
        # [x, y, z, qx, qy, qz, qw]
        self.pregrasp_palm_pose_buf = torch.zeros(self.num_envs, 7, device=self.device)
        # warmstart 수집 전용: euler ZYX 형식 (v7-2 학습과 동일한 직접 덧셈 방식)
        # [x, y, z, ez, ey, ex]
        self.pregrasp_palm_pose_buf_euler = torch.zeros(self.num_envs, 6, device=self.device)

        # ----------------------------------------------------------------
        # Hand poses (per-finger lerp용)
        # open_pose = HAND_APPROACH_POSE (action=-1), grasp_pose = HAND_GRASP_POSE (action=+1)
        # ----------------------------------------------------------------
        self.hand_open_pose  = to_torch(HAND_APPROACH_POSE, device=self.device)  # (20,)
        self.hand_grasp_pose = to_torch(HAND_GRASP_POSE,    device=self.device)  # (20,)

        # ----------------------------------------------------------------
        # 로봇 시작 자세 (arm: ARM_START_POSE, hand: HAND_APPROACH_POSE)
        # ----------------------------------------------------------------
        arm_start  = to_torch(ARM_START_POSE,    device=self.device)  # (7,)
        hand_start = to_torch(HAND_APPROACH_POSE, device=self.device)  # (20,)
        robot_start = torch.cat([arm_start, hand_start], dim=0)         # (27,)
        self.robot_start_joint_pos = (
            robot_start.unsqueeze(0).repeat(self.num_envs, 1).contiguous()
        )
        # [pour_v4] demo pour 팔 자세 (nullspace default_config 바이어스용, j1-4만 사용).
        self._demo_pour_arm_pose = to_torch(DEMO_POUR_ARM_POSE, device=self.device)  # (7,)

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
        # Phase-0 진단용: arm joint velocity/acceleration 추적 버퍼
        self._prev_arm_joint_vel = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        # [Phase-1 Step 6] jerk 계산용: 이전 step acc 벡터
        self._prev_arm_joint_acc = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        # [Phase-1 Step 7] EMA palm action 버퍼 (Fabrics IK 입력 smoothing)
        self._ema_palm_action = torch.zeros(self.num_envs, 6, device=self.device)
        # Action/kinematics diagnostics: policy command vs gate/Fabrics result 분리.
        self._raw_palm_action = torch.zeros(self.num_envs, 6, device=self.device)
        self._applied_palm_action = torch.zeros(self.num_envs, 6, device=self.device)
        self._action_tilt_gate = torch.ones(self.num_envs, device=self.device)
        self._cmd_delta_pre_gate = torch.zeros(self.num_envs, 6, device=self.device)
        self._cmd_delta_post_gate = torch.zeros(self.num_envs, 6, device=self.device)
        self._cmd_delta_rotvec_world = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_palm_target_delta = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_tilt_amount_log = torch.zeros(self.num_envs, device=self.device)

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
        # 손은 episode 내내 grasp_hold freeze (6D action = palm pose만) → finger floor 불필요.
        self._beads_spawned = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

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
        self._total_episodes: int = 0
        self._successful_episodes: int = 0

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

        # Demo pose bank — critic privileged obs + (pour_v4) 정책 demo pose reward.
        self.demo_pose_reference = None
        if self.cfg.enable_demo_critic_obs or self.cfg.enable_demo_pose_reward:
            self.demo_pose_reference = DemoPoseReferenceBank.from_hdf5_paths(
                self.cfg.demo_pose_paths,
                phase=self.cfg.demo_pose_phase,
                device=self.device,
            )
            _uses = []
            if self.cfg.enable_demo_critic_obs:
                _uses.append("critic-obs")
            if self.cfg.enable_demo_pose_reward:
                _uses.append("policy-reward")
            print(
                f"[5g_pour_right_v4] loaded demo pose reference bank ({'+'.join(_uses)}): "
                f"{self.demo_pose_reference.num_frames} frames from {len(self.demo_pose_reference.source_paths)} files",
                flush=True,
            )

        # Demo pose reward graduation state (flow EMA로 weight를 floor까지 감쇠).
        self._demo_arm_pose_w: float = self.cfg.weight_demo_arm_pose
        self._demo_j5_w: float = self.cfg.weight_demo_j5
        self._demo_graduate_ema: float = 0.0

        # Left target cup — FK 기반 고정 배치 (LEFT_ARM_REST_JOINT_POS hand local_z=0.04)
        self._left_cup_pos_env_local = to_torch(
            self.cfg.left_target_cup_pos_env_local, device=self.device
        )
        self._left_cup_quat_wxyz = to_torch(
            self.cfg.left_target_cup_quat_wxyz, device=self.device
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
        self._intermediate_values_step = -1

        # ---- Pour 중간값 버퍼 ----
        self._mouth_xy_distance = torch.zeros(self.num_envs, device=self.device)
        self._cup_center_xy_dist = torch.zeros(self.num_envs, device=self.device)
        self._mouth_z_clearance = torch.zeros(self.num_envs, device=self.device)
        # 3a 진단: rim-pivot 해가 workspace 클램프로 깨지는 정도(클램프 전후 palm 차이)
        self._palm_clamp_viol_xy = torch.zeros(self.num_envs, device=self.device)
        self._palm_clamp_viol_z = torch.zeros(self.num_envs, device=self.device)
        # per-axis 클램프 진단: 어느 bound가 rim-pivot 스윙을 막는지 확정 (binding bound 식별)
        #   부호 있는 위반량(clamp 후 - clamp 전): x_max/y_max binding이면 음수, x_min/y_min이면 양수.
        self._palm_clamp_viol_x = torch.zeros(self.num_envs, device=self.device)
        self._palm_clamp_viol_y = torch.zeros(self.num_envs, device=self.device)
        self._source_up_dot_world = torch.zeros(self.num_envs, device=self.device)
        self._directional_tilt_cos = torch.zeros(self.num_envs, device=self.device)
        # [test8] cup-center 앵커 방향 cosine (전달 자세서 안정 → 상충 제거)
        self._directional_tilt_cos_c = torch.zeros(self.num_envs, device=self.device)
        self._mouth_alignment_cos = torch.zeros(self.num_envs, device=self.device)
        self._rim_facing_cos = torch.zeros(self.num_envs, device=self.device)       # [H10b] palm_ee +z · world +x (cos>0=내회전)
        self._internal_rot_gate = torch.zeros(self.num_envs, device=self.device)    # 내회전 게이트
        self._grasp_cup_quat_palm_init = torch.zeros(self.num_envs, 4, device=self.device)
        self._grasp_cup_quat_palm_init[:, 0] = 1.0
        self._palm_target_rot_error_deg = torch.zeros(self.num_envs, device=self.device)
        self._cup_rel_drift_deg = torch.zeros(self.num_envs, device=self.device)
        self._cmd_minus_actual_tilt_deg = torch.zeros(self.num_envs, device=self.device)
        self._rho = torch.zeros(self.num_envs, device=self.device)  # binary pour gate
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
        self._bead_in_source_delta = torch.zeros(self.num_envs, device=self.device)
        self._bead_in_target_delta = torch.zeros(self.num_envs, device=self.device)
        self._bead_cross_delta = torch.zeros(self.num_envs, device=self.device)
        self._spill_delta = torch.zeros(self.num_envs, device=self.device)
        self._bead_centroid_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._spill_ratio = torch.zeros(self.num_envs, device=self.device)
        # [REDESIGN v4] dense bead 보상: 방출된 bead의 target 축 근접 점수 (N,)
        self._bead_near_score = torch.zeros(self.num_envs, device=self.device)
        self._all_beads_bonus_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._first_capture_bonus_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pre_pour_ready_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._no_tip_force_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._source_empty_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._pour_ready_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._world_up = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)

        self._warmstart_collect_mode = False
        self._warmstart_policy = None
        self._warmstart_cache_count = 0
        self._warmstart_reset_debug_printed = False
        self._warmstart_env_captured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        cache_size = max(int(self.cfg.warmstart_cache_size), 1)
        self._warmstart_arm_pos = torch.zeros(cache_size, NUM_ARM_DOF, device=self.device)
        self._warmstart_hand_pos = torch.zeros(cache_size, NUM_HAND_DOF, device=self.device)
        self._warmstart_palm_pose = torch.zeros(cache_size, 7, device=self.device)
        self._warmstart_cup_pose = torch.zeros(cache_size, 7, device=self.device)
        # GUI 시각화: pour_point(빨강)만 표시 (cfg.enable_visual_markers)
        if cfg.enable_visual_markers:
            self._vis_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/FiveGPourRightMarkers",
                    markers={
                        "source_pour": sim_utils.SphereCfg(
                            radius=0.018,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                        ),
                    },
                )
            )
        else:
            self._vis_markers = None

        self._build_warmstart_reset_cache()

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
    # Fabrics collision box 파싱 (시각화 전용)
    # ------------------------------------------------------------------
    def _parse_fabrics_obstacle(self, world_filename: str, object_name: str):
        """Fabrics world yaml에서 obstacle의 (type, center[3], quat_wxyz[4], size[3])를 파싱한다.

        yaml: type = box|sphere, transform = "x y z qx qy qz qw"(xyzw), scaling = "sx sy sz"(full extent).
        return (type:str, center[3] env-local, quat[4] wxyz, size[3]).
        """
        import os
        import yaml as _yaml
        import fabrics_sim

        worlds_dir = os.path.join(os.path.dirname(fabrics_sim.__file__), "worlds")
        yaml_path = os.path.join(worlds_dir, world_filename + ".yaml")
        with open(yaml_path) as f:
            world = _yaml.safe_load(f)
        obj = world[object_name]
        otype = str(obj.get("type", "box"))
        xform = [float(v) for v in str(obj["transform"]).split()]
        scaling = [float(v) for v in str(obj["scaling"]).split()]
        # yaml quat = xyzw → isaac marker = wxyz
        if len(xform) >= 7:
            quat_wxyz = [xform[6], xform[3], xform[4], xform[5]]
        else:
            quat_wxyz = [1.0, 0.0, 0.0, 0.0]
        return otype, xform[:3], quat_wxyz, scaling

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
            world_filename="open_tesollo_boxes_pour_v5",
        )
        self.object_ids, self.object_indicator = self.world_model.get_object_ids()

        self.timestep = self.cfg.fabrics_dt

        # Main fabric (arm 제어용, graph_capturable=False)
        self.open_tesollo_fabric = OpenArmTeoslloPoseFabric(
            self.num_envs, self.device, self.timestep,
            graph_capturable=False,
            use_hand_fabric=False,
        )
        # [lstm_test5] nullspace(cspace) 어트랙터 무게 강화 — demo j1-4(elbow-up) default_config를
        #   palm-pose task에 덜 밀리게 유지. params는 Attractor가 매 step live로 읽음(스칼라 float).
        #   공유 YAML 대신 이 fabric 인스턴스만 per-task override (다른 태스크 영향 없음).
        _csa = self.open_tesollo_fabric.fabric_params['cspace_attractor']
        _csa['min_isotropic_mass'] = self.cfg.cspace_attractor_mass
        _csa['max_isotropic_mass'] = self.cfg.cspace_attractor_mass
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
            world_filename="open_tesollo_boxes_pour_v5",
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
        self._bead_cross_count.copy_(self._bead_crossed_target_mouth.sum(dim=-1).long())
        bead_cross_fraction = self._bead_crossed_target_mouth.float().mean(dim=-1)
        bead_in_target_fraction = self._bead_in_target.float().mean(dim=-1)
        bead_in_source_fraction = self._bead_in_source.float().mean(dim=-1)

        # source 컵 밖 + target 컵 로컬 z < z_min 아래로 떨어진 bead = 영구 손실
        # transit bead (공중 이동 중): target local z > z_min → spill 아님
        bead_spilled = (
            (~self._bead_in_source)
            & (pos_in_target[..., 2] < self.cfg.target_inside_z_min)
        )
        spill_ratio = bead_spilled.float().mean(dim=-1)

        # [REDESIGN v4] dense bead 근접 점수: 소스에서 방출(released)되고 아직 손실되지
        # 않은(바닥 위) bead가 target 축(중심)에 가까울수록 보상. 4.1cm binary가 만드는
        # sparse 0→0 자기참조를 끊는 다리. captured bead(xy≈0)도 높은 점수 → 채움 유지.
        # anti-hacking: 소스 안 bead는 제외(실제로 따라야 점수) + 바닥 아래(spill) 제외.
        _released = ~self._bead_in_source
        _not_lost = pos_in_target[..., 2] >= self.cfg.target_inside_z_min
        _xy_score = torch.exp(-self.cfg.bead_near_scale * bead_xy_to_target)
        _near = _xy_score * (_released & _not_lost).float()
        self._bead_near_score.copy_(_near.mean(dim=-1))

        self._bead_in_source_delta.copy_(bead_in_source_fraction - self._bead_in_source_fraction)
        self._bead_in_target_delta.copy_(bead_in_target_fraction - self._bead_in_target_fraction)
        self._bead_cross_delta.copy_(bead_cross_fraction - self._bead_cross_fraction)
        self._spill_delta.copy_(spill_ratio - self._spill_ratio)
        self._bead_cross_fraction.copy_(bead_cross_fraction)
        self._bead_in_target_fraction.copy_(bead_in_target_fraction)
        self._bead_in_source_fraction.copy_(bead_in_source_fraction)
        self._spill_ratio.copy_(spill_ratio)
        self._prev_bead_target_local_z.copy_(pos_in_target[..., 2])

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone()

        palm_action = actions[:, :6]    # (N, 6) ∈ [-1, 1] — 손은 grasp_hold freeze
        self._raw_palm_action.copy_(palm_action)

        # ---- Pour phase: Fabrics arm 제어 ----
        # xyz: rim-pivot target 이동량, rot: current-palm incremental target.
        # 절대 workspace(palm_mins/maxs)로 클램프하여 안전 영역 보장
        # 에피소드 시작 직후 N스텝: warmstart pose를 강제 유지 (물리 안착)
        # warmstart 캐시에서 텔레포트한 직후 contact force가 안정화되기 전에
        # 랜덤 action이 arm을 움직이면 컵이 낙하하므로 palm_action을 0으로 고정.
        if self.cfg.episode_hold_steps > 0:
            hold_mask = (self.episode_length_buf < self.cfg.episode_hold_steps).unsqueeze(1)
            palm_action = torch.where(hold_mask, torch.zeros_like(palm_action), palm_action)
        self._applied_palm_action.copy_(palm_action)

        # 비드 지연 소환: hold 종료 첫 스텝에 physics 안정화된 컵 위치로 소환
        if self.cfg.episode_hold_steps > 0 and not self._warmstart_collect_mode:
            hold_end = self.cfg.episode_hold_steps
            just_ended_hold = (self.episode_length_buf == hold_end) & ~self._beads_spawned
            spawn_ids_tensor = just_ended_hold.nonzero(as_tuple=False).squeeze(-1)
            if spawn_ids_tensor.numel() > 0:
                cup_pose_now = torch.cat([
                    self.cup.data.root_pos_w[spawn_ids_tensor],
                    self.cup.data.root_quat_w[spawn_ids_tensor],
                ], dim=-1)
                bead_state = self._sample_bead_states_inside_cup(cup_pose_now)
                self.beads.write_object_state_to_sim(bead_state, env_ids=spawn_ids_tensor)
                self._beads_spawned[spawn_ids_tensor] = True

        # EMA palm action smoothing: Fabrics에 smooth 궤적 전달
        # action_rate_penalty는 raw self.actions 기반 유지 (training gradient 보존)
        self._ema_palm_action.copy_(
            self.cfg.ema_action_alpha * palm_action
            + (1.0 - self.cfg.ema_action_alpha) * self._ema_palm_action
        )

        if self._warmstart_collect_mode:
            # v7-2 학습과 동일한 파이프라인: euler 직접 덧셈 + euler_zyx Fabrics
            delta = scale(self._ema_palm_action, self.delta_mins_warmstart_collect, self.delta_maxs_warmstart_collect)
            self._action_tilt_gate.fill_(1.0)
            self._cmd_delta_pre_gate.copy_(delta)
            self._cmd_delta_post_gate.copy_(delta)
            self._cmd_delta_rotvec_world.zero_()
            palm_pose_euler = self.pregrasp_palm_pose_buf_euler + delta   # (N, 6)
            palm_pose_euler = torch.max(
                torch.min(palm_pose_euler, self.palm_maxs.unsqueeze(0)),
                self.palm_mins.unsqueeze(0),
            )
            # palm_pose_targets에는 quaternion으로 변환 저장 (캐시 캡처용)
            palm_pose_quat = torch.zeros_like(self.pregrasp_palm_pose_buf)
            palm_pose_quat[:, :3] = palm_pose_euler[:, :3]
            palm_pose_quat[:, 3:7] = self._quat_xyzw_from_euler_zyx(palm_pose_euler[:, 3:6])
            self.palm_pose_targets.copy_(palm_pose_quat)
            self._cmd_palm_target_delta.copy_(self.palm_pose_targets[:, :3] - self.palm_center_pos)
            self.hand_pca_targets.zero_()
            _null_cfg = self.fabric_q.detach().clone()
            _null_cfg[:, 0] = torch.clamp(_null_cfg[:, 0] * 0.95 + 0.09 * 0.05, min=-0.29, max=0.46)
            _null_cfg[:, 1] = torch.clamp(_null_cfg[:, 1] * 0.95 + 0.39 * 0.05, min=0.00, max=1.05)
            _null_cfg[:, 2] = torch.clamp(_null_cfg[:, 2] * 0.95 + (-0.24) * 0.05, min=-0.74, max=0.38)
            _null_cfg[:, 6] = torch.clamp(_null_cfg[:, 6] * 0.95 + 0.63 * 0.05, min=0.20, max=1.13)
            self.open_tesollo_fabric.default_config.copy_(_null_cfg)
            self.open_tesollo_fabric.set_features(
                self.hand_pca_targets,
                palm_pose_euler,
                "euler_zyx",
                self.fabric_q.detach(),
                self.fabric_qd.detach(),
                self.object_ids,
                self.object_indicator,
                self.fabric_damping_gain,
            )
        else:
            delta_pre_gate = scale(self._ema_palm_action, self.delta_mins, self.delta_maxs)   # (N, 6)
            delta = delta_pre_gate.clone()
            # 멀리 있을 때는 회전/tilt action을 억제해서 "원거리 tilt"를 방지한다.
            # rim-pivot 후에는 cup_center_xy_dist 대신 mouth_xy_distance 사용:
            # tilt 시 cup root는 rim 반대쪽으로 이동하지만 rim(pour point)은 고정되므로
            # cup_center_xy_dist는 tilt 깊어질수록 증가 → gate=0 (tilt 완전 차단 버그).
            # mouth_xy_distance: rim 기반이므로 tilt 중에도 target 근처 유지 → 일관된 gate.
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
            self._action_tilt_gate.copy_(tilt_gate)
            self._cmd_delta_pre_gate.copy_(delta_pre_gate)
            self._cmd_delta_post_gate.copy_(delta)
            self._cmd_delta_rotvec_world.copy_(delta_rotvec_world)
            palm_pose = torch.zeros_like(self.pregrasp_palm_pose_buf)

            # Rim-pivot: rotation pivots around the cup rim (pour point), not the palm.
            # Without this, rotation around the palm moves the rim away from the target.
            _angle = delta_rotvec_world.norm(dim=-1)
            _axis = torch.where(
                _angle.unsqueeze(-1) > 1e-8,
                delta_rotvec_world / _angle.unsqueeze(-1).clamp(min=1e-8),
                delta_rotvec_world.new_tensor([1.0, 0.0, 0.0]).expand(self.num_envs, -1),
            )
            _delta_quat_wxyz = quat_from_angle_axis(_angle, _axis)
            rim_env = self._source_pour_point_w - self.scene.env_origins  # (N,3) env-local
            rim_rel = rim_env - self.palm_center_pos                       # vec: palm→rim

            # Pour-point action: delta.xy = pour_point XY 이동량 (palm XY가 아님).
            # 회전 R 후 pour_point 위치: palm + quat_apply(R, rim_rel)
            # 원하는 pour_point XY = 현재 rim XY + delta.xy
            # → palm.xy = pour_point_target.xy - quat_apply(R, rim_rel).xy
            pour_point_target_xy = rim_env[:, :2] + delta[:, :2]
            expected_offset_xy = quat_apply(_delta_quat_wxyz, rim_rel)[:, :2]

            palm_pose[:, 2] = self.palm_center_pos[:, 2] + delta[:, 2]   # Z: 현재 위치 기준 delta
            palm_pose[:, :2] = pour_point_target_xy - expected_offset_xy  # XY: pour_point 기준
            # 3a 진단: 클램프가 rim-pivot 해를 깨뜨리는지 측정 (클램프 전 palm 보존)
            _palm_xyz_preclamp = palm_pose[:, :3].clone()
            palm_pose[:, :3] = torch.max(
                torch.min(palm_pose[:, :3], self.palm_maxs[:3].unsqueeze(0)),
                self.palm_mins[:3].unsqueeze(0),
            )
            # 클램프로 잘려나간 양 = rim-pivot이 보정하려던 palm 이동이 막힌 정도
            # → 0보다 크면 pour_point가 명령 위치에 안 옴(hinge 기계적 파손)
            self._palm_clamp_viol_xy = torch.norm(
                palm_pose[:, :2] - _palm_xyz_preclamp[:, :2], dim=-1
            )
            self._palm_clamp_viol_z = (palm_pose[:, 2] - _palm_xyz_preclamp[:, 2]).abs()
            # per-axis 부호 있는 위반량: 어느 bound가 binding인지 식별
            #   <0 → x_max/y_max가 잘림(palm이 상한 너머로 가려 함), >0 → x_min/y_min이 잘림
            self._palm_clamp_viol_x = palm_pose[:, 0] - _palm_xyz_preclamp[:, 0]
            self._palm_clamp_viol_y = palm_pose[:, 1] - _palm_xyz_preclamp[:, 1]
            current_palm_quat_xyzw = self.robot.data.body_quat_w[:, self.palm_body_index][:, [1, 2, 3, 0]]
            palm_pose[:, 3:7] = self._compose_world_delta_quat_xyzw(
                current_palm_quat_xyzw,
                delta_rotvec_world,
            )
            self.palm_pose_targets.copy_(palm_pose)
            self._cmd_palm_target_delta.copy_(self.palm_pose_targets[:, :3] - self.palm_center_pos)
            self.hand_pca_targets.zero_()
            # [pour_v4 lstm_test4] nullspace default_config: j1-4 = demo pour 자세, j5-7 = robot_start.
            #   kinematic 검증: 현재 j4=0.60(=robot_start)이 nullspace로 고정돼 deep tilt 시 j6 포화.
            #   demo는 j4=1.87(팔꿈치 up)로 j6를 거의 안 쓰고 붓는다(sat 0.05). j1-4 redundant DOF를
            #   demo로 바이어스하면 palm pose는 target 유지하며 elbow-up 도착 → 틸트 시 j6 이완.
            #   j5-7(손목)은 robot_start 유지 → 정책 palm orientation이 제어(approach 중 cup pre-tilt 방지).
            _null_cfg = self.fabric_q.detach().clone()                       # hand(grasp)는 현재 유지
            _null_cfg[:, :NUM_ARM_DOF] = self.robot_start_joint_pos[:, :NUM_ARM_DOF]
            _null_cfg[:, :4] = self._demo_pour_arm_pose[:4].unsqueeze(0)     # j1-4만 demo (팔꿈치 up)
            self.open_tesollo_fabric.default_config.copy_(_null_cfg)
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

        # ---- 오른손 파지 유지 (pour 중 항상 grasp pose freeze; 6D action엔 손 채널 없음) ----
        hand_target = self.grasp_hold_hand_pos_buf
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

    # ------------------------------------------------------------------
    # Intermediate values
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self) -> None:
        # DirectRLEnv.step() calls _get_dones() before _get_rewards().
        # Keep bead deltas from being recomputed and zeroed within the same env step.
        if self._intermediate_values_step == int(self.common_step_counter):
            return
        self._intermediate_values_step = int(self.common_step_counter)

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
        # rim center (world)
        _rim_center_w = self.cup.data.root_pos_w + quat_apply(
            self.cup.data.root_quat_w,
            self._source_cup_pour_point_pos_b.unsqueeze(0).expand(n, -1),
        )
        self._source_rim_center_w = _rim_center_w  # rim 입구 중심 (approach rim-xy 거리용)
        # cup up axis (world)
        _cup_up_w = quat_apply(
            self.cup.data.root_quat_w,
            self._source_cup_up_axis_b.unsqueeze(0).expand(n, -1),
        )
        # target opening — pour_point xy 방향 계산에 선행 필요 (순서 이동)
        self._target_opening_w = left_target_pos_w + quat_apply(
            left_target_quat_w,
            self._target_cup_opening_pos_b.unsqueeze(0).expand(n, -1),
        )
        # gravity direction perpendicular to cup up → points toward lowest rim (실시간)
        _world_down = _cup_up_w.new_zeros(n, 3)
        _world_down[:, 2] = -1.0
        _dot = (_world_down * _cup_up_w).sum(dim=-1, keepdim=True)
        _gravity_perp = _world_down - _dot * _cup_up_w
        _gravity_perp_norm = _gravity_perp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        _gravity_perp_hat = _gravity_perp / _gravity_perp_norm
        # [(a) pour_point xy 방향 안정화] xy 방향 = 두 컵 위치(target방향, 자세 무관) → wobble 제거.
        #   기존 gravity_perp xy는 자세-민감(직립 근처 ≈0, 16° wobble) → approach 추종 시 mouth_xy 진동
        #   → g_ready(width 0.02) 절벽 붕괴(test5). introt가 컵을 target으로 회전시키므로
        #   gravity_perp≈target방향(올바른 자세선 일치). xy 크기(기울임 깊이=|perp_xy|)와 z는
        #   실시간 gravity_perp 유지(물리 정확). z는 g_ready 미사용·align z_margin 완화로 wobble 영향 작음.
        _perp_xy_mag = _gravity_perp_hat[:, :2].norm(dim=-1, keepdim=True)
        _pour_dir_xy = self._target_opening_w[:, :2] - _rim_center_w[:, :2]
        _pour_dir_hat = _pour_dir_xy / _pour_dir_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        _pp_xy = _rim_center_w[:, :2] + self.cfg.source_outer_radius * _perp_xy_mag * _pour_dir_hat
        _pp_z = (
            _rim_center_w[:, 2] + self.cfg.source_outer_radius * _gravity_perp_hat[:, 2]
        ).unsqueeze(-1)
        self._source_pour_point_w = torch.cat([_pp_xy, _pp_z], dim=-1)
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

        self._mouth_xy_distance = torch.norm(self._mouth_delta[:, :2], dim=-1)
        self._mouth_z_clearance = self._source_pour_point_w[:, 2] - self._target_opening_w[:, 2]
        self._source_up_dot_world = self._source_up_axis_w[:, 2].clamp(-1.0, 1.0)

        # directional_tilt_cos: cup up-axis XY와 target 방향 XY의 cosine
        # 컵이 target 방향으로 기울면 cos>0, 반대면 cos<0
        _mouth_delta_xy = self._mouth_delta[:, :2]
        _mouth_dir_xy = _mouth_delta_xy / (_mouth_delta_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        _mouth_tilt_dir_xy = self._source_up_axis_w[:, :2]
        _mouth_tilt_dir_xy = _mouth_tilt_dir_xy / (_mouth_tilt_dir_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        self._directional_tilt_cos = (_mouth_tilt_dir_xy * _mouth_dir_xy).sum(dim=-1).clamp(-1.0, 1.0)

        # [test8] dir_cos_c: 방향 앵커를 pour_point가 아닌 cup-center로.
        #   pour_point→target은 전달 자세(pour_point가 타겟 위)서 부호가 뒤집혀 방향·위치가 상충하지만,
        #   cup-center→target은 두 컵이 겹칠 수 없어 항상 안정 → 깊은 전달 자세서도 방향이 +로 보상됨.
        _cc_dir_xy = self._target_opening_w[:, :2] - self.cup.data.root_pos_w[:, :2]
        _cc_dir_xy = _cc_dir_xy / (_cc_dir_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        self._directional_tilt_cos_c = (_mouth_tilt_dir_xy * _cc_dir_xy).sum(dim=-1).clamp(-1.0, 1.0)

        # mouth_alignment_cos: tilt XY heading 기반 (고정 로컬축 사용 시 wrist yaw 착취 가능하므로 제외)
        # 직립에 가까우면 heading 불안정 → 0으로 처리
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

        cup_center_w = self.cup.data.root_pos_w
        self._cup_center_xy_dist = torch.norm(
            cup_center_w[:, :2] - self._target_opening_w[:, :2], dim=-1
        )

        # ρ binary gate: cup이 target 근처에 있을 때만 pour 보상 활성 (DexPour-style)
        self._rho = (self._cup_center_xy_dist < self.cfg.pour_binary_xy_thresh).float()

        # [H11] 내회전 게이트: rl_dg_palm +y(손바닥 roll축) vs world +x 각도.
        #   기존 palm+z(손가락축)는 roll을 못 잼 — 손가락이 +x 유지한 채 손바닥만 roll 가능
        #   → palm+z·worldX는 cos≈1 고정, 내회전 드리프트(lstm_test3 0.92→0.60)를 못 막음.
        #   sim 렌더링 확인(사용자): 손바닥이 내회전될 때 palm +y가 world +x와 둔각(90~270°, cos<0).
        #   → palm+y · worldX < 0 = 내회전. (world +x는 env orientation 불변 고정 기준)
        _palm_quat = self.robot.data.body_quat_w[:, self.palm_body_index]
        _palm_y = quat_apply(
            _palm_quat, _palm_quat.new_tensor([0.0, 1.0, 0.0]).expand(n, -1)
        )  # rl_dg_palm +y (손바닥 roll축)
        self._rim_facing_cos = _palm_y[:, 0].clamp(-1.0, 1.0)  # = palm_y · world+x
        # 내회전 = cos<0 (palm+y가 world +x와 둔각) → gate 1. (부호 반전: thresh - cos)
        #   temp=0.1(가파름): drift(cos 음수→0)시 gate 급감 → r_introt 손실 → 내회전 유지 강제.
        self._internal_rot_gate = torch.sigmoid(
            (self.cfg.internal_rot_thresh - self._rim_facing_cos) / max(self.cfg.internal_rot_temp, 1e-6)
        )

        # Bead flags & spill
        self._compute_bead_flags()

        if self._vis_markers is not None:
            # pour_point(빨강)만 마킹 (단일 마커 → marker_indices 기본 0)
            self._vis_markers.visualize(translations=self._source_pour_point_w)

        # 접촉력 업데이트
        self._update_contact_forces()

    # ------------------------------------------------------------------
    # Legacy warmstart policy observations (grasp checkpoint compatibility)
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

    def _finger_grasp_progress(self, finger_joint_pos: torch.Tensor) -> torch.Tensor:
        """Return per-finger progress from approach pose to grasp pose in [0, 1]."""
        delta = self.hand_grasp_pose - self.hand_open_pose
        valid = delta.abs() > 1e-6
        denom = torch.where(valid, delta, torch.ones_like(delta))
        progress_20 = (
            (finger_joint_pos - self.hand_open_pose.unsqueeze(0)) / denom.unsqueeze(0)
        ).clamp(0.0, 1.0)
        progress_20 = progress_20 * valid.unsqueeze(0).to(progress_20.dtype)
        valid_counts = (
            valid.view(NUM_FINGERTIPS, 4).sum(dim=-1).clamp(min=1).to(progress_20.dtype)
        )
        return (
            progress_20.view(-1, NUM_FINGERTIPS, 4).sum(dim=-1)
            / valid_counts.unsqueeze(0)
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
        left_arm_joint_pos = left_arm_joint_pos_clean + torch.randn_like(left_arm_joint_pos_clean) * σ_qp
        left_arm_joint_vel = left_arm_joint_vel_clean + torch.randn_like(left_arm_joint_vel_clean) * σ_qv
        source_pour_point = source_pour_point_clean + torch.randn_like(source_pour_point_clean) * σ_cp
        target_opening = target_opening_clean + torch.randn_like(target_opening_clean) * σ_cp

        pour_point_to_opening = target_opening - source_pour_point

        finger_grasp_progress = self._finger_grasp_progress(finger_joint_pos)
        binary_contact = self.binary_contact_buf.float()
        last_actions = self.actions
        last_palm_actions = self.actions[:, :NUM_PALM_ACTION]

        # tip force (v8처럼, 실로봇 FT 센서 직결, sim2real 가능)
        tip_force_norm = (self.contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)

        # Actor는 실기에서 재현 가능한 proprio/FK/target-relative 입력만 사용한다.
        # Bead outcome, flow delta, handcrafted transport gate는 critic/reward/logging 전용이다.
        actor_obs = torch.cat([
            arm_joint_pos,              # 7
            arm_joint_vel,              # 7
            finger_grasp_progress,      # 5
            left_arm_joint_pos,         # 9
            left_arm_joint_vel,         # 9
            pour_point_to_opening,      # 3
            source_pour_axis_clean,     # 3
            source_up_axis_clean,       # 3
            target_up_axis_clean,       # 3
            last_palm_actions,          # 6
        ], dim=-1)   # 55D

        if actor_obs.shape[1] != NUM_OBSERVATIONS:
            raise RuntimeError(
                f"[pour_v4] Actor obs dim mismatch: {actor_obs.shape[1]} != {NUM_OBSERVATIONS}"
            )

        # ==== Critic extra obs (39D: 기존 30 + demo privileged 9) ====
        cup_height_delta = (right_cup_pos_clean[:, 2] - self.object_init_pos[:, 2]).unsqueeze(1)

        distal_binary     = self.distal_binary_contact_buf.float()
        distal_force_norm = (self.distal_contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)


        # critic base obs (105D) — full-state value estimation, actor LSTM layout과 분리
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
            # target_up_axis 제거: 항상 [0,0,1], 정보 없음
            torch.stack([                                             # 8
                self._mouth_distance,
                self._mouth_xy_distance,
                self._cup_center_xy_dist,
                self._mouth_z_clearance,
                self._source_up_dot_world,
                self._directional_tilt_cos,
                self._mouth_alignment_cos,
                self._rho,
            ], dim=-1),
            binary_contact,                                           # 5
            tip_force_norm,                                           # 5 (v8처럼, sim2real)
            last_actions,                                             # 6 (palm pose action)
            self._bead_in_source_fraction.unsqueeze(1),               # 1
            self._bead_in_target_fraction.unsqueeze(1),               # 1
            self._bead_cross_fraction.unsqueeze(1),                   # 1
            self._spill_ratio.unsqueeze(1),                           # 1
        ], dim=-1)   # 105D

        # critic privileged demo (정책 미관측) — _get_rewards에서 계산, reset 첫 obs 방어 계산
        demo_feat = getattr(self, "_demo_critic_feat", None)
        if demo_feat is None:
            demo_feat = self._get_demo_critic_features()

        critic_obs = torch.cat([
            actor_obs_clean,                                   # 105
            left_arm_joint_pos_clean,                          # 9
            left_arm_joint_vel_clean,                          # 9
            distal_binary,                                     # 5
            distal_force_norm,                                 # 5
            cup_height_delta,                                  # 1
            self._rho.unsqueeze(1),                            # 1 (binary pour gate)
            demo_feat["demo_arm_joint_err"].unsqueeze(1),      # 1 (privileged)
            demo_feat["demo_j5_err"].unsqueeze(1),             # 1 (privileged)
            demo_feat["demo_target_arm_q"],                    # 7 (privileged)
        ], dim=-1)   # 144D

        if critic_obs.shape[1] != NUM_CRITIC_OBSERVATIONS:
            raise RuntimeError(
                f"[pour_v4] Critic obs dim mismatch: {critic_obs.shape[1]} != {NUM_CRITIC_OBSERVATIONS}"
            )

        return {"policy": actor_obs, "critic": critic_obs}

    def _get_demo_critic_features(self) -> dict[str, torch.Tensor]:
        """Critic privileged: 현재 arm 자세 ↔ demo pour 자세 거리 + NN-매칭 목표 arm q.

        정책(actor)은 보지 못한다 — value 추정 가속·초기 탐색 감소용. reward에 사용 안 함.
        NN 매칭은 j1-4(gross 위치)로만 → stuck j5가 frame 선택을 오염시키지 못하게 한다.
        """
        n = self.num_envs
        if self.demo_pose_reference is None:
            zero = torch.zeros(n, device=self.device)
            return {
                "demo_arm_joint_err": zero,
                "demo_j5_err": zero,
                "demo_target_arm_q": torch.zeros(n, NUM_ARM_DOF, device=self.device),
            }
        ref = self.demo_pose_reference
        arm_q = self.robot.data.joint_pos[:, self.arm_dof_indices]  # (N, 7)
        demo_arm = ref.arm_joint_pos  # (T, 7)
        T_demo = demo_arm.shape[0]
        arm_q4 = arm_q[:, :4]
        demo_arm4 = demo_arm[:, :4]
        aa = (arm_q4 * arm_q4).sum(dim=-1, keepdim=True)
        bb = (demo_arm4 * demo_arm4).sum(dim=-1).unsqueeze(0)
        ab = arm_q4 @ demo_arm4.T
        nn_idx = (aa + bb - 2.0 * ab).argmin(dim=-1)
        K = int(self.cfg.demo_nn_lookahead_frames)
        target_idx = (nn_idx + K).clamp(max=T_demo - 1)
        target_arm_q = demo_arm[target_idx]  # (N, 7)
        arm_std4 = ref.arm_joint_std[:4].clamp(min=0.20)
        arm_norm_err = torch.norm((arm_q[:, :4] - target_arm_q[:, :4]) / arm_std4, dim=-1)
        demo_arm_joint_err = arm_norm_err / math.sqrt(4.0)
        j5_std = ref.arm_joint_std[4].clamp(min=0.20)
        demo_j5_err = torch.abs(arm_q[:, 4] - target_arm_q[:, 4]) / j5_std
        return {
            "demo_arm_joint_err": demo_arm_joint_err,
            "demo_j5_err": demo_j5_err,
            "demo_target_arm_q": target_arm_q,
        }

    def _get_demo_pose_reward_terms(self) -> dict[str, torch.Tensor]:
        """v3 이식: a11~a20 pour 분포로 팔 자세(j1-4) + j5(틸트 주역)를 앵커.

        frame 선택은 j1-4(gross 위치)로만 수행 — stuck 손목(j5-7)이 "얕은 틸트 frame"을
        nearest로 골라 현 자세를 강화하는 루프를 막는다. j5 앵커는 ready latch 이후에만
        활성(v5 stage), weight는 _demo_*_w로 감쇠.
        """
        zero = torch.zeros(self.num_envs, device=self.device)
        if self.demo_pose_reference is None or not self.cfg.enable_demo_pose_reward:
            return {"r_demo_arm_pose": zero, "r_demo_j5": zero, "demo_arm_joint_err": zero}

        ref = self.demo_pose_reference
        arm_q = self.robot.data.joint_pos[:, self.arm_dof_indices]  # (N, 7)

        demo_arm = ref.arm_joint_pos  # (T, 7)
        T_demo = demo_arm.shape[0]
        arm_q4 = arm_q[:, :4]
        demo_arm4 = demo_arm[:, :4]
        aa = (arm_q4 * arm_q4).sum(dim=-1, keepdim=True)
        bb = (demo_arm4 * demo_arm4).sum(dim=-1).unsqueeze(0)
        ab = arm_q4 @ demo_arm4.T
        nn_idx = (aa + bb - 2.0 * ab).argmin(dim=-1)
        K = int(self.cfg.demo_nn_lookahead_frames)
        target_idx = (nn_idx + K).clamp(max=T_demo - 1)
        target_arm_q = demo_arm[target_idx]  # (N, 7)

        arm_std4 = ref.arm_joint_std[:4].clamp(min=0.20)
        arm_norm_err = torch.norm((arm_q[:, :4] - target_arm_q[:, :4]) / arm_std4, dim=-1)
        demo_arm_joint_err = arm_norm_err / math.sqrt(4.0)
        r_demo_arm_pose = torch.exp(-demo_arm_joint_err)

        near_gate = torch.exp(
            -torch.square(self._cup_center_xy_dist / max(self.cfg.demo_pose_near_gate_xy, 1e-6))
        )
        warmup_steps = max(int(self.cfg.demo_pose_warmup_steps), 1)
        step_count = float(getattr(self, "common_step_counter", 0))
        warmup = min(step_count / float(warmup_steps), 1.0)
        # [lstm_test3] r_demo_arm_pose도 ready latch 이후로 게이트.
        #   lstm_test1/2: demo arm 앵커(j1-4)가 ready 이전부터 활성 → corridor approach(먼 거리
        #   gradient≈0)와 상쇄 → Stage-A park(corridor 0.02 정지). approach(j1-4 gross 위치)는
        #   corridor 영역이므로 demo는 over-target(ready) 이후 pour joint_state만 유도해야 한다.
        gate = near_gate * warmup * self._pour_ready_latched.float()

        demo_w = getattr(self, "_demo_arm_pose_w", self.cfg.weight_demo_arm_pose)

        # j5(틸트 주역) 앵커 — ready latch 이후만 (v5 stageB 대응), 감쇠 _demo_j5_w.
        j5_std = ref.arm_joint_std[4].clamp(min=0.20)
        j5_err = torch.abs(arm_q[:, 4] - target_arm_q[:, 4]) / j5_std
        j5_w = getattr(self, "_demo_j5_w", 0.0)
        r_demo_j5 = (
            self._pour_ready_latched.float()
            * j5_w
            * torch.exp(-self.cfg.demo_j5_sharpness * j5_err)
        )

        return {
            "r_demo_arm_pose": gate * demo_w * r_demo_arm_pose,
            "r_demo_j5": r_demo_j5,
            "demo_arm_joint_err": demo_arm_joint_err,
        }

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        # critic privileged demo feature (obs에서 재사용; 정책 reward엔 미사용)
        self._demo_critic_feat = self._get_demo_critic_features()

        # ============================================================
        # Stage A — Grasp maintenance (r_hold), tilt-phase aware
        # ============================================================
        palm_quat_w = self.robot.data.body_quat_w[:, self.palm_body_index]
        palm_pos_w = self.robot.data.body_pos_w[:, self.palm_body_index]
        cup_pos_w = self.cup.data.root_pos_w
        cup_quat_w = self.cup.data.root_quat_w
        cup_in_palm_local = quat_apply_inverse(palm_quat_w, cup_pos_w - palm_pos_w)
        cup_quat_in_palm = quat_mul(self._quat_conjugate_wxyz(palm_quat_w), cup_quat_w)
        if self._needs_grasp_init_update.any():
            upd = self._needs_grasp_init_update.nonzero(as_tuple=False).squeeze(-1)
            self._grasp_cup_pos_palm_local_init[upd] = cup_in_palm_local[upd].detach()
            self._grasp_cup_quat_palm_init[upd] = cup_quat_in_palm[upd].detach()
            self._needs_grasp_init_update[upd] = False
        palm_target_quat_w = self.palm_pose_targets[:, 3:7][:, [3, 0, 1, 2]]
        self._palm_target_rot_error_deg.copy_(
            torch.rad2deg(self._quat_angle_error_wxyz(palm_target_quat_w, palm_quat_w))
        )
        self._cup_rel_drift_deg.copy_(
            torch.rad2deg(self._quat_angle_error_wxyz(self._grasp_cup_quat_palm_init, cup_quat_in_palm))
        )
        actual_tilt_rad = torch.acos(self._source_up_dot_world.clamp(-1.0, 1.0))
        self._cmd_minus_actual_tilt_deg.copy_(
            torch.rad2deg(self._cmd_delta_rotvec_world.norm(dim=-1) - actual_tilt_rad)
        )
        slip_dist = torch.norm(cup_in_palm_local - self._grasp_cup_pos_palm_local_init, dim=-1)
        grasp_maintain_reward = torch.exp(-self.cfg.reward_grasp_slip_sharpness * slip_dist)

        thumb_force = self.contact_force_raw[:, 0]
        others_avg_force = self.contact_force_raw[:, 1:].mean(dim=-1)
        others_count = self.binary_contact_buf[:, 1:].sum(dim=-1)
        full_grasp_flag = (
            self.binary_contact_buf[:, 0] & (others_count >= self.cfg.contact_maintain_min_others)
        ).float()
        has_thumb = self.binary_contact_buf[:, 0].float()
        has_others = (others_count >= 1).float()
        balance_gate = has_thumb * has_others
        force_balance_err = (thumb_force - others_avg_force).abs()
        r_force_balance = (
            self.cfg.weight_force_balance
            * balance_gate
            * torch.exp(-self.cfg.force_balance_sharpness * force_balance_err)
        )

        # tilt-phase aware: 직립(tilt=0)일수록 full grip 요구, 깊은 tilt에선 contact 완화
        tilt_amount = ((1.0 - self._source_up_dot_world) / 2.0).clamp(0.0, 1.0)
        contact_gate = (1.0 - 0.7 * tilt_amount)
        upright_gate = (1.0 - tilt_amount).clamp(0.0, 1.0)
        # 손은 freeze(grasp_hold) → finger_curl은 항상 닫힘(상수 weight 가산)
        r_hold = (
            self.cfg.weight_grasp_maintain * grasp_maintain_reward
            + self.cfg.weight_contact_maintain * full_grasp_flag * contact_gate
            + r_force_balance * upright_gate
            + self.cfg.weight_finger_curl
        )

        # ============================================================
        # [H13] Stage A — Approach: tilt로 rim_center↔pour_point blend × anti-parallel
        #   [H12 실패] approach 전체를 pour_point로 바꿨더니 직립 transport가 깨짐(test4 mouth_xy 0.40 고착,
        #   source_up_dot 0.96, g_ready 미점화). 원인: 직립이면 림 수평→"최하단 점" 부재→gravity_perp≈0
        #   →pour_point 방향 불안정(16° wobble 방위로 ±4.5cm 흔들림). pour 전 이송구간 전체가 직립이라
        #   approach가 흔들리는 점을 쫓아 손목 wobble로 착취·이송 실패.
        #   [H11 plateau] 반대로 rim_center만 쓰면 기울임 후 pour_point가 rim+4.5cm 밖에서 포화(8.8cm).
        #   → blend: 직립=rim_center(안정 이송, test3 검증), 기울수록=pour_point(정밀, 8.8cm plateau 회피).
        #   approach penalty: corridor를 찾으면 0, 못 찾으면 음수. positive approach farming 제거.
        # ============================================================
        _tilt_target_approach = (1.0 - math.cos(math.radians(self.cfg.pour_tilt_target_deg))) / 2.0
        _tilt_blend = (tilt_amount / max(_tilt_target_approach, 1e-6)).clamp(0.0, 1.0).unsqueeze(-1)
        corridor_radius = self.cfg.target_inner_radius + self.cfg.pour_corridor_xy_margin
        _approach_pt_w = (
            (1.0 - _tilt_blend) * self._source_rim_center_w
            + _tilt_blend * self._source_pour_point_w
        )
        self._approach_xy_dist = torch.norm(
            _approach_pt_w[:, :2] - self._target_opening_w[:, :2], dim=-1
        )
        _approach_corridor_score = pour_corridor_score(
            _approach_pt_w,
            self._target_opening_w,
            corridor_radius,
            self.cfg.pour_corridor_z_min,
            self.cfg.pour_corridor_z_max,
            self.cfg.pour_corridor_scale,
        )
        self._rim_antiparallel = (
            self._source_up_axis_w * self._target_up_axis_w
        ).sum(dim=-1)
        approach_corridor_miss = (1.0 - _approach_corridor_score).clamp(min=0.0)
        r_approach_pre_ready = -self.cfg.weight_dist_to_target * approach_corridor_miss

        # ============================================================
        # Stage A→B 공간 게이트 (target 입구 corridor + ready latch)
        # ============================================================
        corridor_score = pour_corridor_score(
            self._source_pour_point_w,
            self._target_opening_w,
            corridor_radius,
            self.cfg.pour_corridor_z_min,
            self.cfg.pour_corridor_z_max,
            self.cfg.pour_corridor_scale,
        )
        self._pour_ready_latched |= corridor_score >= self.cfg.ready_latch_threshold
        latched_ready = self._pour_ready_latched.float()
        ready_context = torch.maximum(
            corridor_score,
            latched_ready * self.cfg.ready_latch_floor,
        )
        release_context = torch.maximum(
            corridor_score,
            latched_ready * self.cfg.release_gate_floor_after_ready,
        )
        g_ready = ready_context
        corridor_escape = (1.0 - corridor_score).clamp(min=0.0)
        r_corridor_escape = -self.cfg.weight_corridor_escape_after_ready * corridor_escape
        r_approach = (
            (1.0 - latched_ready) * r_approach_pre_ready
            + latched_ready * r_corridor_escape
        )

        # ============================================================
        # Stage B — tilt / bead. Corridor is phase context, not direct align reward.
        # ============================================================
        # tilt 직접 유도 — v6 ALIGN 실패 교훈: tilt를 직접 보상해 "직립 회피해" 차단
        tilt_target = (1.0 - math.cos(math.radians(self.cfg.pour_tilt_target_deg))) / 2.0
        tilt_progress = (tilt_amount / max(tilt_target, 1e-6)).clamp(0.0, 1.0)
        tilt_amount_delta = tilt_amount - self._prev_tilt_amount_log
        # [H5/H10] tilt에 내회전 방향성 결합: "내회전하면서 기울일 때만" tilt 보상.
        #   rot_dir = floor + (1-floor)·gate. floor=0.0(H10) → rot_dir = internal_rot_gate:
        #   외회전(gate~0)이면 tilt 보상 0, 내회전(gate~1) 가면 100%. 외회전 tilt local min 차단.
        rot_dir = self.cfg.rot_tilt_floor + (1.0 - self.cfg.rot_tilt_floor) * self._internal_rot_gate
        # [tilt 식 교체/test8] 2단 A/B → 0→135° 단일 연속 ramp, always-on.
        #   진단: 구 r_tilt_A는 85°(tilt_pre)서 saturate(grad→0), r_tilt_B는 85° 넘어야 시작 →
        #   82-85° dead spot에서 정책 정지(peak 0.43<0.456) → tilt_progress_B 영구 미발현.
        #   교체: tilt_target(135°)까지 끊김 없는 단일 gradient(=tilt_progress). 85° dead spot 제거.
        # Latch phase gate: once target inlet corridor was reached, deep tilt stays rewarded
        # even if the pour point moves during the physically correct pouring posture.
        tilt_ready_factor = self.cfg.tilt_aim_floor + (1.0 - self.cfg.tilt_aim_floor) * latched_ready
        r_tilt = self.cfg.weight_tilt * tilt_progress * rot_dir * tilt_ready_factor
        # [로깅 전용] 85° 돌파 추적 (보상 미사용). tilt_progress_A=전체 진행도, B=깊은 구간(85→135°)
        tilt_pre = self.cfg.tilt_pre_amount
        tilt_progress_A = tilt_progress
        tilt_progress_B = ((tilt_amount - tilt_pre) / max(tilt_target - tilt_pre, 1e-6)).clamp(0.0, 1.0)
        # [H10] 상시 내회전 유도 (tilt 비종속, Stage A always-on): "내회전이 옳다"를 직접 보상.
        #   r_tilt(곱)는 tilt 전엔 회전 gradient=0 → chicken-and-egg. r_introt가 tilt 없이도,
        #   접근 전(g_ready 무관)부터 내회전 gradient 제공(Stage B는 늦음). w_introt(5)<w_tilt(15)+
        #   r_approach(5) → 회전만 park 아닌 회전+접근+tilt 단계 상승. total에 직접 가산(아래).
        r_introt = self.cfg.weight_introt * self._internal_rot_gate

        # Direct align reward is disabled. Corridor remains as phase/release context only.
        r_align = torch.zeros_like(corridor_score)

        # [r_pour_z 제거] z barrier가 hinge pour와 상충(기울이면 pour_point 자연 하강→페널티→주기적 붕괴).
        #   충돌은 mouth_z_clearance/termination으로 모니터링.
        aim_gate = (
            (self._directional_tilt_cos_c > 0.0) & (tilt_amount > self.cfg.drain_tilt_min)
        ).float()

        # bead — release는 ready 이후 wobble에 hard-block되지 않는 release_context 종속.
        source_release_delta = (-self._bead_in_source_delta).clamp(min=0.0)
        r_source_release = (
            release_context
            * aim_gate
            * self.cfg.weight_source_release
            * source_release_delta
        )
        target_capture_delta = self._bead_in_target_delta.clamp(min=0.0)
        r_target_capture = self.cfg.weight_target_capture_delta * target_capture_delta
        # [release-delta probe] 누적 target 상태 reward는 1-bead park를 만들 수 있어 제거.
        # 실제 배출 유도는 source_release_delta의 transient reward로만 본다.
        r_bead_in = torch.zeros_like(self._bead_in_target_fraction)
        # [release-delta probe] source-empty 누적 상태 reward도 제거.
        r_drain = torch.zeros_like(source_release_delta)

        # [align-off probe] direct align reward is kept as zero for dashboard compatibility.
        r_stageB = r_align


        # ============================================================
        # Outcome
        # ============================================================
        success_fill_ratio = (
            self.success_adr.get_param("success", "fill_ratio")
            if self.success_adr is not None
            else self.cfg.success_target_fill_ratio
        )
        success_now = (
            (self._bead_in_target_fraction >= success_fill_ratio)
            & (self._spill_ratio <= self.cfg.success_spill_max)
            & (self._cup_center_xy_dist < self.cfg.pour_binary_xy_thresh)
        )
        r_success = success_now.float()
        spill_cost = self._spill_ratio.clamp(min=0.0).sqrt()
        spill_weight = (
            self.spill_adr.get_param("reward", "spill_weight")
            if self.spill_adr is not None
            else self.cfg.weight_spill
        )

        # ---- demo pose reward (pour_v4): flow EMA로 weight 감쇠 후 j1-4/j5 앵커 ----
        flow_signal = float(
            (self._bead_cross_fraction + self._bead_in_target_fraction).mean().item()
        )
        a_ema = self.cfg.demo_graduate_ema_alpha
        self._demo_graduate_ema = (1.0 - a_ema) * self._demo_graduate_ema + a_ema * flow_signal
        graduate = 1.0 - min(
            self._demo_graduate_ema / max(self.cfg.demo_graduate_flow_target, 1e-6), 1.0
        )
        self._demo_arm_pose_w = self.cfg.weight_demo_arm_pose_floor + (
            self.cfg.weight_demo_arm_pose - self.cfg.weight_demo_arm_pose_floor
        ) * graduate
        self._demo_j5_w = self.cfg.weight_demo_j5_floor + (
            self.cfg.weight_demo_j5 - self.cfg.weight_demo_j5_floor
        ) * graduate
        demo_terms = self._get_demo_pose_reward_terms()
        r_demo_arm_pose = demo_terms["r_demo_arm_pose"]
        r_demo_j5 = demo_terms["r_demo_j5"]

        total = (
            r_hold
            + r_approach
            + r_introt
            + r_tilt            # latch phase 기반 0→135° 단일 연속 ramp
            + r_stageB
            + r_source_release  # [probe] g_ready 무관, 실제 소스 잔량 감소분만 transient 보상
            + r_target_capture  # [probe] target에 새로 capture된 bead delta만 outcome 보상
            + r_bead_in         # [probe] 누적 bead-in 상태 reward는 0 고정
            + r_demo_arm_pose   # [pour_v4] a11~a20 pour 분포 j1-4 앵커 (감쇠, floor 유지)
            + r_demo_j5         # [pour_v4] j5(틸트 주역) 앵커 — ready latch 이후만
            + self.cfg.weight_success * r_success
            - g_ready * spill_weight * spill_cost   # [H14] g_ready 게이트: target 위(stageB)서만 spill 벌점 → 초기 탐험 보호
        )

        # ---- ADR increment ----
        _ep_success_rate = self._successful_episodes / max(self._total_episodes, 1)
        if self.spill_adr is not None:
            self.spill_adr.maybe_increment(_ep_success_rate)
        if self.noise_adr is not None:
            self.noise_adr.maybe_increment(_ep_success_rate)
        if self.success_adr is not None:
            self.success_adr.maybe_increment(_ep_success_rate)

        # arm vel 추적 버퍼 유지 (진단 호환)
        arm_qd = self.robot.data.joint_vel[:, self.arm_dof_indices]
        self._prev_arm_joint_vel.copy_(arm_qd)

        # ============================================================
        # 복합 로깅
        # ============================================================
        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_dof_indices]
        demo_feat = self._demo_critic_feat
        reward_log: dict = {
            "Reward/hold":     r_hold.mean(),
            "Reward/approach": r_approach.mean(),
            "Reward/introt":   r_introt.mean(),
            "Reward/tilt":     r_tilt.mean(),                # latch phase 기반 0→135° 단일 연속 ramp
            "Reward/source_release":  r_source_release.mean(),
            "Reward/target_capture":  r_target_capture.mean(),
            "Reward/demo_arm_pose":   r_demo_arm_pose.mean(),
            "Reward/demo_j5":         r_demo_j5.mean(),
        }
        reward_w0_log: dict = {
            "Reward_w0/tilt_pre": torch.zeros((), device=self.device),  # [test8] r_tilt_A 폐기 (0 고정, 대시보드 호환)
            "Reward_w0/align":    r_align.mean(),
            "Reward_w0/bead_in":  r_bead_in.mean(),    # [게이트 분리] g_ready 무관 (total 직접 가산값과 일치)
            "Reward_w0/drain":    (g_ready * r_drain).mean(),
            "Reward_w0/success":  (self.cfg.weight_success * r_success).mean(),
        }
        for k, v in reward_log.items():
            self.extras[k] = v
        for k, v in reward_w0_log.items():
            self.extras[k] = v
        action_kinematics_log: dict = {
            "Action_Kinematics/raw_action/x": self._raw_palm_action[:, 0].mean(),
            "Action_Kinematics/raw_action/y": self._raw_palm_action[:, 1].mean(),
            "Action_Kinematics/raw_action/z": self._raw_palm_action[:, 2].mean(),
            "Action_Kinematics/raw_action/spin": self._raw_palm_action[:, 3].mean(),
            "Action_Kinematics/raw_action/tilt_toward": self._raw_palm_action[:, 4].mean(),
            "Action_Kinematics/raw_action/tilt_ortho": self._raw_palm_action[:, 5].mean(),
            "Action_Kinematics/applied_action/x": self._applied_palm_action[:, 0].mean(),
            "Action_Kinematics/applied_action/y": self._applied_palm_action[:, 1].mean(),
            "Action_Kinematics/applied_action/z": self._applied_palm_action[:, 2].mean(),
            "Action_Kinematics/applied_action/spin": self._applied_palm_action[:, 3].mean(),
            "Action_Kinematics/applied_action/tilt_toward": self._applied_palm_action[:, 4].mean(),
            "Action_Kinematics/applied_action/tilt_ortho": self._applied_palm_action[:, 5].mean(),
            "Action_Kinematics/ema_action/x": self._ema_palm_action[:, 0].mean(),
            "Action_Kinematics/ema_action/y": self._ema_palm_action[:, 1].mean(),
            "Action_Kinematics/ema_action/z": self._ema_palm_action[:, 2].mean(),
            "Action_Kinematics/ema_action/spin": self._ema_palm_action[:, 3].mean(),
            "Action_Kinematics/ema_action/tilt_toward": self._ema_palm_action[:, 4].mean(),
            "Action_Kinematics/ema_action/tilt_ortho": self._ema_palm_action[:, 5].mean(),
            "Action_Kinematics/command_pre_gate/x": self._cmd_delta_pre_gate[:, 0].mean(),
            "Action_Kinematics/command_pre_gate/y": self._cmd_delta_pre_gate[:, 1].mean(),
            "Action_Kinematics/command_pre_gate/z": self._cmd_delta_pre_gate[:, 2].mean(),
            "Action_Kinematics/command_pre_gate/spin": self._cmd_delta_pre_gate[:, 3].mean(),
            "Action_Kinematics/command_pre_gate/tilt_toward": self._cmd_delta_pre_gate[:, 4].mean(),
            "Action_Kinematics/command_pre_gate/tilt_ortho": self._cmd_delta_pre_gate[:, 5].mean(),
            "Action_Kinematics/command_post_gate/x": self._cmd_delta_post_gate[:, 0].mean(),
            "Action_Kinematics/command_post_gate/y": self._cmd_delta_post_gate[:, 1].mean(),
            "Action_Kinematics/command_post_gate/z": self._cmd_delta_post_gate[:, 2].mean(),
            "Action_Kinematics/command_post_gate/spin": self._cmd_delta_post_gate[:, 3].mean(),
            "Action_Kinematics/command_post_gate/tilt_toward": self._cmd_delta_post_gate[:, 4].mean(),
            "Action_Kinematics/command_post_gate/tilt_ortho": self._cmd_delta_post_gate[:, 5].mean(),
            "Action_Kinematics/gate/tilt_action": self._action_tilt_gate.mean(),
            "Action_Kinematics/command/rot_norm_pre_gate": self._cmd_delta_pre_gate[:, 3:6].norm(dim=-1).mean(),
            "Action_Kinematics/command/rot_norm": self._cmd_delta_rotvec_world.norm(dim=-1).mean(),
            "Action_Kinematics/command/palm_target_dx": self._cmd_palm_target_delta[:, 0].mean(),
            "Action_Kinematics/command/palm_target_dy": self._cmd_palm_target_delta[:, 1].mean(),
            "Action_Kinematics/command/palm_target_dz": self._cmd_palm_target_delta[:, 2].mean(),
            "Action_Kinematics/tracking/palm_target_rot_error_deg": self._palm_target_rot_error_deg.mean(),
            "Action_Kinematics/tracking/cup_rel_drift_deg": self._cup_rel_drift_deg.mean(),
            "Action_Kinematics/tracking/cmd_minus_actual_tilt_deg": self._cmd_minus_actual_tilt_deg.mean(),
            "Action_Kinematics/kinematics/tilt_amount": tilt_amount.mean(),
            "Action_Kinematics/kinematics/tilt_delta": tilt_amount_delta.mean(),
            "Action_Kinematics/kinematics/mouth_xy_dist": self._mouth_xy_distance.mean(),
            "Action_Kinematics/kinematics/approach_xy_dist": self._approach_xy_dist.mean(),
            "Action_Kinematics/kinematics/corridor_score": corridor_score.mean(),
            "Action_Kinematics/clamp/palm_xy": self._palm_clamp_viol_xy.mean(),
            "Action_Kinematics/clamp/palm_z": self._palm_clamp_viol_z.mean(),
            "Action_Kinematics/clamp/palm_active": (self._palm_clamp_viol_xy + self._palm_clamp_viol_z > 1e-4).float().mean(),
        }
        for k, v in action_kinematics_log.items():
            self.extras[k] = v

        diag: dict = {
            # Stage 게이트
            "log/g_ready":               g_ready.mean(),
            "log/stageB_active":         (g_ready > 0.5).float().mean(),
            "log/aim_gate":              aim_gate.mean(),
            "log/corridor_score":        corridor_score.mean(),
            "log/approach_corridor_score": _approach_corridor_score.mean(),
            "log/approach_corridor_miss": approach_corridor_miss.mean(),
            "log/corridor_miss":         corridor_escape.mean(),
            "log/approach_pre_ready":    r_approach_pre_ready.mean(),
            "log/corridor_escape":       r_corridor_escape.mean(),
            "log/ready_latched":         self._pour_ready_latched.float().mean(),
            # demo pose reward (pour_v4)
            "log/demo_arm_joint_err":    demo_terms["demo_arm_joint_err"].mean(),
            "log/demo_arm_pose_w":       torch.tensor(self._demo_arm_pose_w, device=self.device),
            "log/demo_j5_w":             torch.tensor(self._demo_j5_w, device=self.device),
            "log/demo_graduate_ema":     torch.tensor(self._demo_graduate_ema, device=self.device),
            "log/release_context":       release_context.mean(),
            "log/tilt_ready_factor":     tilt_ready_factor.mean(),
            "log/tilt_latched_phase":    latched_ready.mean(),
            # tilt / 정렬
            "log/tilt_amount":           tilt_amount.mean(),
            "log/tilt_progress_A":       tilt_progress_A.mean(),   # [2단] 0→85° 진행도
            "log/tilt_progress_B":       tilt_progress_B.mean(),   # [2단] 85→135° 진행도
            "log/tilt_frac_90":          (tilt_amount >= 0.5).float().mean(),
            "log/tilt_frac_110":         (tilt_amount >= ((1.0 - math.cos(math.radians(110.0))) / 2.0)).float().mean(),
            "log/tilt_frac_120":         (tilt_amount >= 0.75).float().mean(),
            "log/tilt_frac_135":         (tilt_amount >= ((1.0 - math.cos(math.radians(135.0))) / 2.0)).float().mean(),
            "log/source_up_dot":         self._source_up_dot_world.mean(),
            "log/rim_facing_cos":        self._rim_facing_cos.mean(),  # [H11] palm+y·worldX (음수=내회전)
            "log/internal_rot_gate":     self._internal_rot_gate.mean(),
            "log/tilt_rot_dir":          rot_dir.mean(),
            "log/rim_antiparallel":      self._rim_antiparallel.mean(),  # [H11] source·target rim+z (음수=마주봄)
            # 위치
            "log/approach_xy_dist":      self._approach_xy_dist.mean(),    # [H13] blend(rim_center↔pour_point) 거리 (rim_center_xy 통합)
            "log/cup_center_xy_dist":    self._cup_center_xy_dist.mean(),
            "log/mouth_xy_dist":         self._mouth_xy_distance.mean(),
            "log/mouth_z_clearance":     self._mouth_z_clearance.mean(),
            "log/pour_point_x":          (self._source_pour_point_w[:, 0] - self.scene.env_origins[:, 0]).mean(),
            "log/pour_point_y":          (self._source_pour_point_w[:, 1] - self.scene.env_origins[:, 1]).mean(),
            "log/pour_point_z":          (self._source_pour_point_w[:, 2] - self.scene.env_origins[:, 2]).mean(),
            "log/target_open_x":         (self._target_opening_w[:, 0] - self.scene.env_origins[:, 0]).mean(),
            "log/target_open_y":         (self._target_opening_w[:, 1] - self.scene.env_origins[:, 1]).mean(),
            "log/target_open_z":         (self._target_opening_w[:, 2] - self.scene.env_origins[:, 2]).mean(),
            # joints
            "joint_State/j1": arm_joint_pos[:, 0].mean(),
            "joint_State/j2": arm_joint_pos[:, 1].mean(),
            "joint_State/j3": arm_joint_pos[:, 2].mean(),
            "joint_State/j4": arm_joint_pos[:, 3].mean(),
            "joint_State/j5": arm_joint_pos[:, 4].mean(),
            "joint_State/j6": arm_joint_pos[:, 5].mean(),
            "joint_State/j7": arm_joint_pos[:, 6].mean(),
            # bead flow
            "log/bead_in_target":        self._bead_in_target_fraction.mean(),
            "log/bead_in_source":        self._bead_in_source_fraction.mean(),
            "log/source_release_delta":  source_release_delta.mean(),
            "log/target_capture_delta":  target_capture_delta.mean(),
            "log/spill_ratio":           self._spill_ratio.mean(),
            # [Phase0] rim-pivot hinge 기계적 파손 측정: palm 위치 박스(palm_mins/maxs)가
            #   rim-pivot 보정 palm 이동을 클램프한 양 = pour_point가 명령 위치를 벗어난 정도.
            #   tilt 정지(~83°)와 동시에 상승하면 → 박스가 틸트 벽(reward 아님) 확정.
            "joint_State/palm_clamp_viol_xy": self._palm_clamp_viol_xy.mean(),
            "joint_State/palm_clamp_viol_z":  self._palm_clamp_viol_z.mean(),
            "joint_State/palm_clamp_active":  (self._palm_clamp_viol_xy + self._palm_clamp_viol_z > 1e-4).float().mean(),
            # per-axis binding 식별: 깊은 tilt env에서 어느 bound가 잘리는지 (음수=max bound, 양수=min bound)
            "joint_State/palm_clamp_viol_x_deep": torch.where(
                tilt_amount > 0.4, self._palm_clamp_viol_x, torch.zeros_like(self._palm_clamp_viol_x),
            ).sum() / (tilt_amount > 0.4).float().sum().clamp(min=1.0),
            "joint_State/palm_clamp_viol_y_deep": torch.where(
                tilt_amount > 0.4, self._palm_clamp_viol_y, torch.zeros_like(self._palm_clamp_viol_y),
            ).sum() / (tilt_amount > 0.4).float().sum().clamp(min=1.0),
            # 깊은 tilt(>0.4≈78°) env에서만 본 clamp 위반 (평균 희석 제거 → binding 직접 포착)
            "joint_State/palm_clamp_viol_deep": torch.where(
                tilt_amount > 0.4,
                self._palm_clamp_viol_xy + self._palm_clamp_viol_z,
                torch.zeros_like(self._palm_clamp_viol_xy),
            ).sum() / (tilt_amount > 0.4).float().sum().clamp(min=1.0),
            # j1~j7 한계 포화도: 1.0이면 관절 한계 도달 → arm joint 벽 후보.
            #   부호별 limit 적용(양수=upper, 음수=lower). limits:
            #   j1[-1.40,3.49] j2[-0.17,3.32] j3[±1.57] j4[0,2.44] j5[±1.57] j6[±0.79] j7[±1.57]
            **{
                f"joint_State/j{_i + 1}_sat": torch.maximum(
                    arm_joint_pos[:, _i] / _up, arm_joint_pos[:, _i] / _lo
                ).mean()
                for _i, (_lo, _up) in enumerate([
                    (-1.40, 3.49), (-0.17, 3.32), (-1.57, 1.57), (-1e-6, 2.44),
                    (-1.57, 1.57), (-0.79, 0.79), (-1.57, 1.57),
                ])
            },
        }
        if self.spill_adr is not None:
            diag["log/adr_spill"] = torch.tensor(self.spill_adr.progress, device=self.device)
        if self.noise_adr is not None:
            diag["log/adr_noise"] = torch.tensor(self.noise_adr.progress, device=self.device)
        if self.success_adr is not None:
            diag["log/adr_success"] = torch.tensor(self.success_adr.progress, device=self.device)
            diag["log/success_fill_ratio"] = torch.tensor(float(success_fill_ratio), device=self.device)
        for k, v in diag.items():
            self.extras[k] = v.mean() if isinstance(v, torch.Tensor) and v.dim() > 0 else v

        self._prev_tilt_amount_log.copy_(tilt_amount)
        return total

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

        success_by_fill = (
            (self._bead_in_target_fraction >= success_fill_ratio)
            & (self._spill_ratio <= self.cfg.success_spill_max)
        )
        self.success_flag.copy_(success_by_fill)
        self.episode_success_buf |= self.success_flag   # 에피소드 중 한 번이라도 성공 시 True

        # 비드 이상 감지: 비드 소환 상태에서 z < -0.5 이면 물리 폭발로 판단 → 즉시 종료
        # _hide_beads()는 z=-10.0에 숨기므로, _beads_spawned=False인 env는 체크 제외
        # (월드 z 사용: z방향 env_origin은 모두 0이므로 절대값 유효)
        bead_pos_w = self.beads.data.object_pos_w  # (N, num_beads, 3)
        bead_fallen = self._beads_spawned & (bead_pos_w[..., 2] < -0.5).any(dim=-1)

        # [test11-A] 완료 절벽 제거: source_drained·success는 "실패"가 아니라 "과제 완료/타임아웃"이므로
        #   terminated(V=0)이 아닌 truncated(미래가치 부트스트랩)로 분류. terminated엔 실패 사건만 남김.
        #   → 비드를 다 부으면 보상 스트림이 V=0으로 절단되던 park-farming 유인 제거.
        terminated = (
            out_x | out_y | fallen | dropped_by_force
            | bead_fallen
        )
        truncated  = (
            (self.episode_length_buf >= self.max_episode_length - 1)
            | source_drained | self.success_flag
        )

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

        # ---- episode 성공 집계 후 클리어 ----
        self._total_episodes += n
        self._successful_episodes += int(self.episode_success_buf[env_ids].sum().item())

        # warmstart cache 저장: 에피소드 종료(final state)에만 체크
        self._maybe_store_warmstart_successes(env_ids)
        env_ids_t_reset = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)
        self._warmstart_env_captured[env_ids_t_reset] = False

        self.episode_success_buf[env_ids] = False
        if (not self._warmstart_collect_mode) and self._warmstart_cache_count > 0:
            self._reset_from_warmstart_cache(env_ids)
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

        # reset/warmstart pose cache. normal pour 회전 action은 current palm 기준.
        self.pregrasp_palm_pose_buf[env_ids] = pregrasp_palm_pose
        # warmstart 수집용 euler 버퍼 (v7-2 학습 형식과 동일)
        self.pregrasp_palm_pose_buf_euler[env_ids] = pregrasp_palm_pose_euler
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

        left_cup_pose = self._get_left_cup_fk_pose(env_ids=env_ids)
        self._left_target_cup_fixed_pose_w[env_ids] = left_cup_pose
        self.left_target_cup.write_root_pose_to_sim(left_cup_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        if self._warmstart_collect_mode:
            self._hide_beads(env_ids)
            self._beads_spawned[env_ids] = False
        else:
            bead_state = self._sample_bead_states_inside_cup(cup_root_state[:, :7])
            self.beads.write_object_state_to_sim(bead_state, env_ids=env_ids)
            self._beads_spawned[env_ids] = True

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
        self._prev_bead_target_local_z[env_ids] = 10.0  # [H14] fix: [env_ids].fill_()는 복사본 채우는 no-op였음
        self._needs_grasp_init_update[env_ids] = True   # 다음 스텝에 palm local init 갱신
        self._bead_cross_count[env_ids] = 0
        self._bead_cross_fraction[env_ids] = 0.0
        self._prev_bead_ever_in_target_count[env_ids] = 0
        self._bead_in_target_fraction[env_ids] = 0.0
        self._bead_in_source_fraction[env_ids] = 0.0
        self._bead_in_source_delta[env_ids] = 0.0
        self._bead_in_target_delta[env_ids] = 0.0
        self._bead_cross_delta[env_ids] = 0.0
        self._spill_delta[env_ids] = 0.0
        self._bead_centroid_w[env_ids].zero_()
        self._spill_ratio[env_ids] = 0.0
        self._all_beads_bonus_paid[env_ids] = False
        self._first_capture_bonus_paid[env_ids] = False
        self._no_tip_force_steps[env_ids] = 0
        self._source_empty_steps[env_ids] = 0
        self._pour_ready_latched[env_ids] = False
        self.success_flag[env_ids] = False
        self._pre_pour_ready_steps[env_ids] = 0
        self._prev_arm_joint_vel[env_ids].zero_()
        self._prev_arm_joint_acc[env_ids].zero_()   # [Step 6]
        self._ema_palm_action[env_ids].zero_()       # [Step 7]
        self._raw_palm_action[env_ids].zero_()
        self._applied_palm_action[env_ids].zero_()
        self._action_tilt_gate[env_ids] = 1.0
        self._cmd_delta_pre_gate[env_ids].zero_()
        self._cmd_delta_post_gate[env_ids].zero_()
        self._cmd_delta_rotvec_world[env_ids].zero_()
        self._cmd_palm_target_delta[env_ids].zero_()
        self._grasp_cup_quat_palm_init[env_ids].zero_()
        self._grasp_cup_quat_palm_init[env_ids, 0] = 1.0
        self._palm_target_rot_error_deg[env_ids].zero_()
        self._cup_rel_drift_deg[env_ids].zero_()
        self._cmd_minus_actual_tilt_deg[env_ids].zero_()
        self._prev_tilt_amount_log[env_ids].zero_()

        # actions 리셋: action=0 = current rim/palm pose 유지.
        self.actions[env_ids, :] = 0.0
        self.prev_actions[env_ids, :] = 0.0
        self._intermediate_values_step = -1


    def _get_left_cup_fk_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """FK 상수로 left target cup의 world pose를 반환. stale body_pos_w 불사용."""
        if env_ids is None:
            origins = self.scene.env_origins
        else:
            origins = self.scene.env_origins[env_ids]
        n = origins.shape[0]
        pos_w = origins + self._left_cup_pos_env_local.unsqueeze(0).expand(n, -1)
        quat_w = self._left_cup_quat_wxyz.unsqueeze(0).expand(n, -1)
        return torch.cat([pos_w, quat_w], dim=-1)

    def _get_left_target_cup_fixed_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            return self._left_target_cup_fixed_pose_w
        return self._left_target_cup_fixed_pose_w[env_ids]

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

    def _hide_beads(self, env_ids: Sequence[int]) -> None:
        n = len(env_ids)
        bead_state = torch.zeros(n, self.num_beads, 13, device=self.device)
        bead_state[..., 2] = -10.0
        bead_state[..., 3] = 1.0
        self.beads.write_object_state_to_sim(bead_state, env_ids=env_ids)

    def _load_warmstart_cache_from_disk(self) -> bool:
        """grasp 디스크 캐시를 로드해 _warmstart_* 버퍼를 직접 채운다.

        성공 시 True. 파일 없음/검증 실패 등은 False 를 반환해 호출부가
        rollout 으로 안전하게 degrade 하게 한다 (silent fail 금지: 로그 출력).
        """
        palm_bounds = (
            float(self.palm_mins[0]),
            float(self.palm_mins[1]),
            float(self.palm_mins[2]),
            float(self.palm_maxs[0]),
            float(self.palm_maxs[1]),
            float(self.palm_maxs[2]),
        )
        try:
            bank = PourWarmStateBank.from_hdf5_paths(
                self.cfg.warm_state_paths,
                device=self.device,
                expected_object_spawn_z=self.cfg.object_spawn_z,
                expected_palm_bounds=palm_bounds,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[5g_pour_right_v4] warm-state disk load error: {exc}", flush=True)
            return False

        n = len(bank)
        if n == 0:
            print("[5g_pour_right_v4] warm-state cache is empty on disk.", flush=True)
            return False

        # 버퍼를 디스크 캐시 크기로 재할당 (warmstart_cache_size 무관, 전량 활용)
        self._warmstart_arm_pos = bank.arm_joint_pos.clone()
        self._warmstart_hand_pos = bank.hand_joint_pos.clone()
        self._warmstart_palm_pose = bank.palm_pose_quat_xyzw.clone()  # (n,7) pos+quat_xyzw
        # cup 은 grasp 성공 당시의 실제 자세로 텔레포트한다.
        # upright(identity) 강제는 손-컵 상대 자세를 깨뜨려(손가락이 컵 벽을
        # 파고듦) hold 중 컵 이탈/손가락 끼임을 유발한다. bead 는 hold 종료
        # 후 안정화된 컵 위치에서 소환되므로 upright 강제는 불필요하다.
        cup_pose = torch.zeros(n, 7, device=self.device)
        cup_pose[:, :3] = bank.cup_pos_local
        cup_pose[:, 3:7] = bank.cup_quat_wxyz  # 실제 grasp cup orientation (wxyz)
        self._warmstart_cup_pose = cup_pose
        self._warmstart_cache_count = n

        print(
            f"[5g_pour_right_v4] loaded {n} warmstart states from disk "
            f"({', '.join(bank.source_paths)}).",
            flush=True,
        )
        return True

    def _build_warmstart_reset_cache(self) -> None:
        if not self.cfg.enable_warmstart_reset:
            return

        source = getattr(self.cfg, "warm_state_source", "rollout")

        if source == "preset":
            # 캐시 없이 시작 → _reset_idx 가 일반 pregrasp 경로 사용 (디버그용)
            print(
                "[5g_pour_right_v4] warm_state_source='preset': "
                "skipping warmstart cache (using pregrasp reset).",
                flush=True,
            )
            return

        if source == "disk":
            if self._load_warmstart_cache_from_disk():
                return
            print(
                "[5g_pour_right_v4] disk warm-state load failed; "
                "falling back to checkpoint rollout.",
                flush=True,
            )

        ckpt = self.cfg.warmstart_checkpoint_path
        if not ckpt:
            return

        try:
            self._warmstart_policy = _WarmstartPolicy(ckpt, self.device).to(self.device)
        except Exception as exc:
            print(f"[5g_pour_right_v4] warmstart policy load failed: {exc}", flush=True)
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
                "[5g_pour_right_v4] warmstart cache is empty. "
                "The v7 checkpoint rollout did not produce any lift-success state, so this task cannot start "
                "from the requested play-like grasp state."
            )

        print(
            f"[5g_pour_right_v4] collected {self._warmstart_cache_count} warmstart success states.",
            flush=True,
        )

    def _maybe_store_warmstart_successes(self, env_ids: Sequence[int]) -> None:
        """에피소드 종료 시 v7-2 final state를 warmstart 캐시에 저장.

        _warmstart_env_captured 플래그로 에피소드 내 중복 저장을 방지.
        """
        if not self._warmstart_collect_mode:
            return
        if self._warmstart_cache_count >= self._warmstart_arm_pos.shape[0]:
            return
        if len(env_ids) == 0:
            return

        env_ids_t = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)

        # 이번 에피소드에서 이미 캡처된 env는 제외
        not_captured = ~self._warmstart_env_captured[env_ids_t]
        if not not_captured.any():
            return
        env_ids_t = env_ids_t[not_captured]

        lifted = self.object_pos[env_ids_t, 2] > (self.object_init_pos[env_ids_t, 2] + self.cfg.lift_success_height)
        grasped = self.num_contacts_buf[env_ids_t] >= MIN_CONTACTS_FOR_SUCCESS
        upright = self._source_up_axis_w[env_ids_t, 2] > 0.90
        j7 = self.robot.data.joint_pos[env_ids_t, self.arm_dof_indices[6]]
        j7_in_range = (j7 >= 0.20) & (j7 <= 1.50)
        warmstart_success = lifted & grasped & upright & j7_in_range

        success_local = warmstart_success.nonzero(as_tuple=False).squeeze(-1)
        if success_local.numel() == 0:
            return

        success_env_ids = env_ids_t[success_local]

        remaining = self._warmstart_arm_pos.shape[0] - self._warmstart_cache_count
        success_env_ids = success_env_ids[:remaining]
        count = success_env_ids.numel()
        if count == 0:
            return

        start = self._warmstart_cache_count
        end = start + count
        self._warmstart_arm_pos[start:end] = self.robot.data.joint_pos[success_env_ids][:, self.arm_dof_indices]
        self._warmstart_hand_pos[start:end] = self.robot.data.joint_pos[success_env_ids][:, self.hand_dof_indices]
        self._warmstart_palm_pose[start:end] = self.palm_pose_targets[success_env_ids]
        self._warmstart_cup_pose[start:end, :3] = self.cup.data.root_pos_w[success_env_ids] - self.scene.env_origins[success_env_ids]
        # cup orientation은 upright(identity)로 저장 (기울어진 채 저장 시 bead 소환 위치 오류 방지)
        self._warmstart_cup_pose[start:end, 3] = 1.0   # w=1 (upright)
        self._warmstart_cup_pose[start:end, 4:7] = 0.0  # x,y,z=0
        self._warmstart_cache_count = end
        self._warmstart_env_captured[success_env_ids] = True
        self._warmstart_cache_count = end

    def _reset_from_warmstart_cache(self, env_ids: Sequence[int]) -> None:
        n = len(env_ids)
        pick = torch.randint(self._warmstart_cache_count, (n,), device=self.device)
        arm_pos = self._warmstart_arm_pos[pick]
        hand_pos = self._warmstart_hand_pos[pick]
        palm_pose = self._warmstart_palm_pose[pick]
        cup_pose_local = self._warmstart_cup_pose[pick]

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

        warmstart_palm_pose = palm_pose.clone()
        if self.cfg.warmstart_palm_z_boost > 0.0:
            warmstart_palm_pose[:, 2] = warmstart_palm_pose[:, 2] + self.cfg.warmstart_palm_z_boost
        warmstart_palm_pose[:, :3] = torch.max(
            torch.min(warmstart_palm_pose[:, :3], self.palm_maxs[:3].unsqueeze(0)),
            self.palm_mins[:3].unsqueeze(0),
        )
        # palm pos 가 pour workspace 로 잘리면 palm target ↔ 실제 arm 자세가
        # 괴리될 수 있다. 1회 진단 로그용 최대 클램프량 기록.
        _ws_clamp_delta = (warmstart_palm_pose[:, :3] - palm_pose[:, :3]).abs().max().item()
        self.pregrasp_palm_pose_buf[env_ids] = warmstart_palm_pose
        self.palm_pose_targets[env_ids] = warmstart_palm_pose
        self.hand_joint_targets[env_ids] = hand_pos
        self.object_init_pos[env_ids] = cup_pose_local[:, :3]
        self.object_init_pos[env_ids, 2] = self.cfg.object_spawn_z  # z는 테이블 높이 기준으로 고정 (캐시 lifted z 사용 시 cup_height_delta=0 버그)
        self._grasp_rel_palm_to_cup_init[env_ids] = cup_pose_local[:, :3] - palm_pose[:, :3]
        self._grasp_cup_height_init[env_ids] = cup_pose_local[:, 2]
        self.open_tesollo_fabric.default_config[env_ids, :NUM_ARM_DOF] = arm_pos

        cup_pose_world = cup_pose_local.clone()
        cup_pose_world[:, :3] += self.scene.env_origins[env_ids]
        zero_vel = torch.zeros(n, 6, device=self.device)
        self.cup.write_root_pose_to_sim(cup_pose_world, env_ids=env_ids)
        self.cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        left_cup_pose = self._get_left_cup_fk_pose(env_ids=env_ids)
        self._left_target_cup_fixed_pose_w[env_ids] = left_cup_pose
        self.left_target_cup.write_root_pose_to_sim(left_cup_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        # 비드는 즉시 소환하지 않고 episode_hold_steps 후 물리 안정화된 컵 위치에 소환
        self._hide_beads(env_ids)
        self._beads_spawned[env_ids] = False

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
        self._prev_bead_target_local_z[env_ids] = 10.0  # [H14] fix: [env_ids].fill_()는 복사본 채우는 no-op였음
        self._needs_grasp_init_update[env_ids] = True   # 다음 스텝에 palm local init 갱신
        self._bead_cross_count[env_ids] = 0
        self._bead_cross_fraction[env_ids] = 0.0
        self._prev_bead_ever_in_target_count[env_ids] = 0
        self._bead_in_target_fraction[env_ids] = 0.0
        self._bead_in_source_fraction[env_ids] = 0.0
        self._bead_in_source_delta[env_ids] = 0.0
        self._bead_in_target_delta[env_ids] = 0.0
        self._bead_cross_delta[env_ids] = 0.0
        self._spill_delta[env_ids] = 0.0
        self._bead_centroid_w[env_ids].zero_()
        self._spill_ratio[env_ids] = 0.0
        self._all_beads_bonus_paid[env_ids] = False
        self._first_capture_bonus_paid[env_ids] = False
        self._no_tip_force_steps[env_ids] = 0
        self._source_empty_steps[env_ids] = 0
        self._pour_ready_latched[env_ids] = False
        self.success_flag[env_ids] = False
        self._prev_arm_joint_vel[env_ids].zero_()
        self._prev_arm_joint_acc[env_ids].zero_()   # [Step 6]
        self._ema_palm_action[env_ids].zero_()       # [Step 7]
        self._raw_palm_action[env_ids].zero_()
        self._applied_palm_action[env_ids].zero_()
        self._action_tilt_gate[env_ids] = 1.0
        self._cmd_delta_pre_gate[env_ids].zero_()
        self._cmd_delta_post_gate[env_ids].zero_()
        self._cmd_delta_rotvec_world[env_ids].zero_()
        self._cmd_palm_target_delta[env_ids].zero_()
        self._grasp_cup_quat_palm_init[env_ids].zero_()
        self._grasp_cup_quat_palm_init[env_ids, 0] = 1.0
        self._palm_target_rot_error_deg[env_ids].zero_()
        self._cup_rel_drift_deg[env_ids].zero_()
        self._cmd_minus_actual_tilt_deg[env_ids].zero_()
        self._prev_tilt_amount_log[env_ids].zero_()

        self.actions[env_ids, :] = 0.0
        self.prev_actions[env_ids, :] = 0.0
        self._pre_pour_ready_steps[env_ids] = 0
        self.success_flag[env_ids] = False
        self._intermediate_values_step = -1

        if not self._warmstart_reset_debug_printed:
            source_pour_point_w = cup_pose_world[:, :3] + quat_apply(
                cup_pose_world[:, 3:7],
                self._source_cup_pour_point_pos_b.unsqueeze(0).expand(n, -1),
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
                "[5g_pour_right_v4][warmstart_reset] "
                f"cup_z_local mean={cup_z_local.mean().item():.4f} "
                f"min={cup_z_local.min().item():.4f} max={cup_z_local.max().item():.4f} | "
                f"mouth_xy mean={mouth_xy_distance.mean().item():.4f} "
                f"min={mouth_xy_distance.min().item():.4f} max={mouth_xy_distance.max().item():.4f} | "
                f"mouth_z_clearance mean={mouth_z_clearance.mean().item():.4f} "
                f"min={mouth_z_clearance.min().item():.4f} max={mouth_z_clearance.max().item():.4f} | "
                f"warmstart_palm_z mean={warmstart_palm_pose[:, 2].mean().item():.4f} | "
                f"palm_pos_clamp_max={_ws_clamp_delta:.4f}",
                flush=True,
            )
            if _ws_clamp_delta > 0.02:
                print(
                    "[5g_pour_right_v4][warmstart_reset][WARN] palm pos clamped by "
                    f"{_ws_clamp_delta:.4f}m → palm target may decouple from arm pose. "
                    "Check grasp/pour workspace alignment.",
                    flush=True,
                )
            self._warmstart_reset_debug_printed = True
