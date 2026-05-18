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
    NUM_FINGER_ACTION,
    NUM_FINGERTIPS,
    NUM_OBSERVATIONS,
    NUM_DISTAL_SENSORS,
    NUM_MIDDLE_SENSORS,
    NUM_CRITIC_OBSERVATIONS,
    CONTACT_FORCE_THRESHOLD,
    CONTACT_FORCE_MAX,
    MIN_CONTACTS_FOR_SUCCESS,
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
    LEFT_TARGET_CUP_POS_ENV_LOCAL,
    LEFT_TARGET_CUP_QUAT_WXYZ,
    RIGHT_ACTUATED_JOINT_NAMES,
    HAND_APPROACH_POSE,
    HAND_GRASP_POSE,
    OBJECT_GOAL_POS,
)
from .pour_right_utils import scale, to_torch
from .demo_pose_reference import DemoPoseReferenceBank




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
        # [fix] warmstart 수집 시 grasp v7-2 학습 스케일(xyz=0.15, rot=20°)로 분리
        # 기존: pour env의 _delta_rad(120°) 재사용 → action=0.5가 60° 회전 유발 → 캐시 오염
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

        # pregrasp palm pose 버퍼 (에피소드별 delta action 기준점)
        # [x, y, z, qx, qy, qz, qw]
        self.pregrasp_palm_pose_buf = torch.zeros(self.num_envs, 7, device=self.device)

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
        # warmstart reset 후 비드가 아직 소환되지 않은 env 추적 (hold phase 동안 숨김)
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

        self.demo_pose_reference = None
        if self.cfg.enable_demo_pose_reward:
            self.demo_pose_reference = DemoPoseReferenceBank.from_hdf5_paths(
                self.cfg.demo_pose_paths,
                phase=self.cfg.demo_pose_phase,
                device=self.device,
            )
            print(
                "[5g_pour_right_v3] loaded demo pose reference bank: "
                f"{self.demo_pose_reference.num_frames} frames from {len(self.demo_pose_reference.source_paths)} files",
                flush=True,
            )

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
        self._prev_bead_in_target_fraction = torch.zeros(self.num_envs, device=self.device)
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
        self._bead_cross_count.copy_(self._bead_crossed_target_mouth.sum(dim=-1).long())
        self._bead_cross_fraction.copy_(self._bead_crossed_target_mouth.float().mean(dim=-1))
        self._bead_in_target_fraction.copy_(self._bead_in_target.float().mean(dim=-1))
        self._bead_in_source_fraction.copy_(self._bead_in_source.float().mean(dim=-1))

        # [v5 방식] target cup 로컬 프레임 z 기준으로 spill 판정 (world z threshold 제거)
        # target cup root z=0.323m, 테이블 bead local_z=0.257-0.323=-0.066m
        # threshold=-0.060m: -0.066 < -0.060 → spill ✓ (테이블 낙하 bead 포착)
        # transit bead (공중 낙하): local_z > 0 → spill 아님 (false positive 제거)
        bead_spilled = (
            (~self._bead_in_source)
            & (pos_in_target[..., 2] < self.cfg.target_inside_z_min)
        )
        self._spill_ratio.copy_(bead_spilled.float().mean(dim=-1))
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

        # ---- warmstart hold phase: action freeze + 비드 지연 소환 + action ramp-up ----
        if not self._warmstart_collect_mode:
            hold_end   = self.cfg.episode_hold_steps
            ramp_end   = hold_end + self.cfg.episode_ramp_steps

            # [1] hold: action 완전 억제 (텔레포트 직후 physics 안착)
            if hold_end > 0:
                hold_mask_1d = self.episode_length_buf < hold_end
                hold_mask    = hold_mask_1d.unsqueeze(1)
                palm_action   = torch.where(hold_mask, torch.zeros_like(palm_action),   palm_action)
                finger_action = torch.where(hold_mask, torch.ones_like(finger_action),  finger_action)

            # [2] 비드 지연 소환: hold 종료 첫 스텝에 현재 컵 위치로 소환
            # (physics 안정 후 소환하면 관통/폭발 현상 없음)
            if hold_end > 0:
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

            # [3] ramp-up: hold 종료 후 ramp_steps 동안 warmstart env의 palm action 점진 스케일업
            # untrained policy의 첫 대형 action이 j7을 limit으로 몰아붙이는 것을 방지
            if self.cfg.episode_ramp_steps > 0:
                steps_since_hold = (self.episode_length_buf - hold_end).float()
                ramp_t = (steps_since_hold / self.cfg.episode_ramp_steps).clamp(0.0, 1.0)
                in_ramp = (self.episode_length_buf >= hold_end) & (self.episode_length_buf < ramp_end)
                ramp_scale = torch.where(
                    in_ramp & self._warmstart_only_close,
                    ramp_t,
                    torch.ones_like(ramp_t),
                ).unsqueeze(1)
                palm_action = palm_action * ramp_scale

            # [4] finger: warmstart env는 파지 열기 금지
            close_only = self._warmstart_only_close.unsqueeze(1)
            finger_action = torch.where(
                close_only,
                torch.maximum(finger_action, self._warmstart_finger_action_floor),
                finger_action,
            )
            self.actions[:, 6:11] = finger_action
        else:
            # warmstart collect mode: hold + ramp 없이 단순 적용
            if self.cfg.episode_hold_steps > 0:
                hold_mask = (self.episode_length_buf < self.cfg.episode_hold_steps).unsqueeze(1)
                palm_action = torch.where(hold_mask, torch.zeros_like(palm_action), palm_action)

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
                (self.cfg.tilt_action_gate_xy_far - self._cup_center_xy_dist) / gate_den,
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

        # warmstart hold phase: 저장된 grasp 자세 그대로 유지 (finger_action=1.0→hand_grasp_pose가
        # warmstart 저장 자세보다 열릴 경우 컵이 떨어지는 문제 방지)
        if self.cfg.episode_hold_steps > 0 and not self._warmstart_collect_mode:
            hold_mask_1d = self.episode_length_buf < self.cfg.episode_hold_steps
            warmstart_hold = hold_mask_1d & self._warmstart_only_close
            hand_target = torch.where(
                warmstart_hold.unsqueeze(1),
                self.grasp_hold_hand_pos_buf,
                hand_target,
            )

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
        self._source_pour_point_w = self.cup.data.root_pos_w + quat_apply(
            self.cup.data.root_quat_w,
            self._source_cup_pour_point_pos_b.unsqueeze(0).expand(n, -1),
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
        # [fix] tilt-invariant clearance: pour_point_w_z는 120° tilt 시 target 아래로 내려가 g_clear 붕괴.
        # 컵이 직립했을 때의 rim z (cup_root_z + rim_offset)를 기준으로 계산 → tilting 중에도 g_clear 유지.
        _cup_rim_z_upright = self.cup.data.root_pos_w[:, 2] + self.cfg.source_cup_rim_z_offset
        self._mouth_z_clearance = _cup_rim_z_upright - self._target_opening_w[:, 2]
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

        # cup center XY dist to target (tilt-invariant: pour_point이 기울어져도 변하지 않음)
        cup_center_w = self.cup.data.root_pos_w
        self._cup_center_xy_dist = torch.norm(
            cup_center_w[:, :2] - self._target_opening_w[:, :2], dim=-1
        )

        # g_align_xy: cup center 기반 (tilt 시 pour_point이 9.8cm 이동해도 gate 붕괴 없음)
        self._g_align_xy = torch.exp(-self.cfg.reward_gate_xy_scale * self._cup_center_xy_dist)
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
    # Observations: Actor 110D | Critic 143D
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

        # tip force (v8처럼, 실로봇 FT 센서 직결, sim2real 가능)
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


        # critic actor_obs_clean (110D) — clean state 재조합, actor_obs 구조와 동일
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
            actor_obs_clean,                                    # 110 (transport_summary 8D + bead/spill 4D 포함)
            left_arm_joint_pos_clean,                          # 9
            left_arm_joint_vel_clean,                          # 9
            distal_binary,                                     # 5
            distal_force_norm,                                 # 5
            cup_height_delta,                                  # 1
            self._g_align_xy.unsqueeze(1),                     # 1
            self._g_clear.unsqueeze(1),                        # 1
            self._g_tilt.unsqueeze(1),                         # 1
            self._g_pour.unsqueeze(1),                         # 1
        ], dim=-1)   # 143D

        if critic_obs.shape[1] != NUM_CRITIC_OBSERVATIONS:
            raise RuntimeError(
                f"[pour_v3] Critic obs dim mismatch: {critic_obs.shape[1]} != {NUM_CRITIC_OBSERVATIONS}"
            )

        return {"policy": actor_obs, "critic": critic_obs}

    def _get_demo_pose_reward_terms(
        self,
        *,
        arm_qd_l2: torch.Tensor,
        arm_jerk_l2: torch.Tensor,
        palm_delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = torch.zeros(self.num_envs, device=self.device)
        if self.demo_pose_reference is None:
            return {
                "r_demo_arm_pose": zero,
                "r_demo_palm_pose": zero,
                "cost_demo_smooth": zero,
                "cost_thumb_grip": zero,
                "demo_arm_joint_err": zero,
                "demo_palm_pos_err": zero,
                "demo_palm_rot_err": zero,
            }

        ref = self.demo_pose_reference
        arm_q = self.robot.data.joint_pos[:, self.arm_dof_indices]  # (N, 7)

        # --- Nearest-Neighbor in joint space + look-ahead ---
        # 현재 joint 상태와 가장 가까운 demo 프레임을 찾고 K 프레임 앞 자세를 타겟으로
        # 효율적 L2: ||a-b||^2 = ||a||^2 + ||b||^2 - 2·a·bT (중간 (N,T,7) 텐서 생략)
        demo_arm = ref.arm_joint_pos  # (T, 7)
        T_demo = demo_arm.shape[0]
        aa = (arm_q * arm_q).sum(dim=-1, keepdim=True)          # (N, 1)
        bb = (demo_arm * demo_arm).sum(dim=-1).unsqueeze(0)     # (1, T)
        ab = arm_q @ demo_arm.T                                  # (N, T)
        nn_idx = (aa + bb - 2.0 * ab).argmin(dim=-1)            # (N,)
        K = int(self.cfg.demo_nn_lookahead_frames)
        target_idx = (nn_idx + K).clamp(max=T_demo - 1)         # (N,)
        target_arm_q = demo_arm[target_idx]                      # (N, 7)

        arm_norm_err = torch.norm((arm_q - target_arm_q) / ref.arm_joint_std, dim=-1)
        demo_arm_joint_err = arm_norm_err / math.sqrt(float(NUM_ARM_DOF))
        r_demo_arm_pose = torch.exp(-demo_arm_joint_err)

        # --- Palm: 동일한 target_idx로 일관성 있게 참조 ---
        palm_pos_w = self.robot.data.body_pos_w[:, self.palm_body_index] - self.scene.env_origins
        palm_quat_wxyz = self.robot.data.body_quat_w[:, self.palm_body_index]

        demo_palm = ref.palm_pose  # (T, 7): [x,y,z, qx,qy,qz,qw]
        target_palm = demo_palm[target_idx]                              # (N, 7)
        target_palm_pos = target_palm[:, :3]                             # (N, 3)
        target_palm_quat_xyzw = target_palm[:, 3:7]                     # (N, 4) xyzw
        target_palm_quat_wxyz = torch.cat(
            [target_palm_quat_xyzw[:, 3:4], target_palm_quat_xyzw[:, :3]], dim=-1
        )                                                                # (N, 4) wxyz

        palm_pos_norm_err = torch.norm((palm_pos_w - target_palm_pos) / ref.palm_pos_std, dim=-1)
        quat_dot = torch.abs((palm_quat_wxyz * target_palm_quat_wxyz).sum(dim=-1)).clamp(max=1.0)
        demo_palm_rot_err = 2.0 * torch.acos(quat_dot)
        demo_palm_pos_err = torch.norm(palm_pos_w - target_palm_pos, dim=-1)
        r_demo_palm_pose = torch.exp(-palm_pos_norm_err - demo_palm_rot_err)

        hand_q = self.robot.data.joint_pos[:, self.hand_dof_indices]
        thumb_norm_err = torch.norm((hand_q[:, :4] - ref.thumb_joint_mean) / ref.thumb_joint_std, dim=-1)
        cost_thumb_grip = thumb_norm_err / 2.0

        vel_excess = torch.relu(arm_qd_l2 - ref.arm_vel_l2_p95) / ref.arm_vel_l2_p95
        jerk_excess = torch.relu(arm_jerk_l2 - ref.arm_jerk_l2_p95) / ref.arm_jerk_l2_p95
        cost_demo_smooth = vel_excess.pow(2) + jerk_excess.pow(2) + 0.1 * palm_delta

        near_gate = torch.exp(-torch.square(self._cup_center_xy_dist / max(self.cfg.demo_pose_near_gate_xy, 1e-6)))
        warmup_steps = max(int(self.cfg.demo_pose_warmup_steps), 1)
        step_count = float(getattr(self, "common_step_counter", 0))
        warmup = min(step_count / float(warmup_steps), 1.0)
        gate = near_gate * warmup

        return {
            "r_demo_arm_pose": gate * self.cfg.weight_demo_arm_pose * r_demo_arm_pose,
            "r_demo_palm_pose": gate * self.cfg.weight_demo_palm_pose * r_demo_palm_pose,
            "cost_demo_smooth": gate * self.cfg.weight_demo_smooth * cost_demo_smooth,
            "cost_thumb_grip": gate * self.cfg.weight_thumb_grip_pose * cost_thumb_grip,
            "demo_arm_joint_err": demo_arm_joint_err,
            "demo_palm_pos_err": demo_palm_pos_err,
            "demo_palm_rot_err": demo_palm_rot_err,
        }

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        # ---- 1. Grasp maintenance: cup의 palm local frame 위치 유지 (slip 억제) ----
        palm_quat_w = self.robot.data.body_quat_w[:, self.palm_body_index]
        palm_pos_w  = self.robot.data.body_pos_w[:, self.palm_body_index]
        cup_pos_w   = self.cup.data.root_pos_w
        cup_in_palm_local = quat_apply_inverse(palm_quat_w, cup_pos_w - palm_pos_w)

        if self._needs_grasp_init_update.any():
            upd = self._needs_grasp_init_update.nonzero(as_tuple=False).squeeze(-1)
            self._grasp_cup_pos_palm_local_init[upd] = cup_in_palm_local[upd].detach()
            self._needs_grasp_init_update[upd] = False

        slip_dist = torch.norm(cup_in_palm_local - self._grasp_cup_pos_palm_local_init, dim=-1)
        grasp_maintain_reward = torch.exp(-self.cfg.reward_grasp_slip_sharpness * slip_dist)

        # contact_maintain
        thumb_force = self.contact_force_raw[:, 0]
        others_avg_force = self.contact_force_raw[:, 1:].mean(dim=-1)
        others_count = self.binary_contact_buf[:, 1:].sum(dim=-1)
        full_grasp_flag = (
            self.binary_contact_buf[:, 0] & (others_count >= self.cfg.contact_maintain_min_others)
        ).float()
        
        # force_balance
        has_thumb  = self.binary_contact_buf[:, 0].float()
        has_others = (others_count >= 1).float()
        balance_gate = has_thumb * has_others
        force_balance_err = (thumb_force - others_avg_force).abs()
        r_force_balance = (
            self.cfg.weight_force_balance
            * balance_gate
            * torch.exp(-self.cfg.force_balance_sharpness * force_balance_err)
        )

        # finger_curl (닫힘 유도)
        finger_lerp_min = self.actions[:, 6:11].min(dim=-1).values
        finger_curl_score = (finger_lerp_min + 1.0) / 2.0
        r_finger_curl = self.cfg.weight_finger_curl * finger_curl_score

        # Stage A. Hold (grasp 유지)
        r_hold = (
            self.cfg.weight_grasp_maintain * grasp_maintain_reward
            + self.cfg.weight_contact_maintain * full_grasp_flag
            + r_force_balance
            + r_finger_curl
        )

        # ---- Stage B. Transport (Approach) - Always active with smooth gradient ----
        # [P2] DexPour: "lift reward ceases once cup reaches h_lift"
        # test4: 컵 0.407m 높이에서도 r_lift 수령 → 컵 올리는 행동 지속 유인
        # lift_height_cap(0.05m) 이상은 보상 없음
        cup_height_delta = (self.object_pos[:, 2] - self.object_init_pos[:, 2]).clamp(
            min=0.0, max=self.cfg.lift_height_cap
        )
        r_lift = self.cfg.weight_approach_z * cup_height_delta
        
        # Smooth approach reward using tanh (0 at infinity, 1 at 0)
        # cup center 기반: tilt 시 pour_point이 이동해도 approach reward가 흔들리지 않음
        r_approach_xy = 1.0 - torch.tanh(self.cfg.reward_approach_xy_scale * self._cup_center_xy_dist)
        r_transport_progress = torch.clamp(
            self._prev_mouth_xy_distance - self._mouth_xy_distance,
            min=0.0,
        )
        # 가까이 다가오면(≈3cm) 접근 보상을 끄고 pour 신호를 강조
        approach_gate_den = max(self.cfg.approach_xy_off_far - self.cfg.approach_xy_off_near, 1e-6)
        approach_gate = torch.clamp(
            (self._cup_center_xy_dist - self.cfg.approach_xy_off_near) / approach_gate_den,
            min=0.0,
            max=1.0,
        )
        r_approach = approach_gate * (
            self.cfg.weight_approach_xy * r_approach_xy
            + self.cfg.weight_transport_progress * r_transport_progress
        )

        # [test10] z-height approach: source cup CENTER z (tilt-invariant) 기반으로 수정
        # 버그(test9): _source_pour_point_w[:, 2]는 cup 기울기에 따라 변화 → 120° tilt 시 -5cm
        #   → r_approach_z_rim이 올바른 tilting을 패널티화 (직립 3.0/step, 120° tilt 0.32/step)
        # 수정: cup.data.root_pos_w[:, 2] (cup center, tilt-invariant) 사용
        #   z_cup_center_ideal = target_rim + clearance(4cm) - cup_rim_offset(10cm) = target_rim - 6cm
        z_cup_center_ideal = (
            self._target_opening_w[:, 2]
            + self.cfg.approach_z_rim_target_clearance
            - self.cfg.source_cup_rim_z_offset
        )
        z_rim_err = (self.cup.data.root_pos_w[:, 2] - z_cup_center_ideal).abs()
        r_approach_z_rim = self.cfg.weight_approach_z_rim * torch.exp(
            -self.cfg.approach_z_rim_sharpness * z_rim_err
        )

        # ---- Stage C. Pre-pour (using soft g_ready gate) ----
        target_tilt_cos = math.cos(math.radians(self.cfg.pour_tilt_target_deg))
        r_tilt = torch.exp(
            -self.cfg.pour_tilt_sharpness * torch.abs(self._source_up_dot_world - target_tilt_cos)
        )
        r_align = 0.5 * (self._mouth_alignment_cos + 1.0)
        
        # Pre-pour signals (g_ready gate: tilt reward only active near target)
        r_prepour_stage = self._g_ready * (
            self.cfg.weight_prepour_dir * r_tilt
            + self.cfg.weight_prepour_align * r_align
        )

        # ---- Stage D. Pour (DexPour-style binary gate: ρ) ----
        # [P3] DexPour ρ = (cup_target_dist < d_pour) AND (prior stages complete)
        # 기존 soft g_pour: g_pour=0.128 → pour reward 항상 희미하게 활성 → 멀리서도 bead 흘러도 reward
        # binary gate: cup_center_xy_dist < 0.15m AND source_up_dot < 0.50(>60° tilt) 동시 충족 시에만 활성
        # → "컵 근처에서 기울어야만 pour reward" → transport + tilt 순서 명확히 학습
        r_cross = self._bead_cross_fraction
        r_capture = self._bead_in_target_fraction
        # task progress signal: target 유입량 증가분(전이)을 직접 보상
        r_capture_gain = torch.clamp(
            self._bead_in_target_fraction - self._prev_bead_in_target_fraction,
            min=0.0,
        )

        gate_pour_binary = (
            (self._cup_center_xy_dist < self.cfg.pour_binary_xy_thresh)
            & (self._source_up_dot_world < self.cfg.pour_binary_tilt_thresh)
        ).float()

        r_pour_align = 0.5 * (self._mouth_alignment_cos + 1.0)

        # [fix] r_pour_aim 제거: pour_point XY는 90° tilt에서 target 방향 최대 → 90° local optimum 생성
        # 90°→120° 진행 시 r_pour_aim 감소 → r_prepour tilt gradient 역행 → 에이전트 90°에 정지
        # 진단용 계산은 유지 (logging / weight=0으로 비활성)
        _pour_point_xy_dist = torch.norm(
            self._source_pour_point_w[:, :2] - self._target_opening_w[:, :2], dim=-1
        )
        r_pour_aim = torch.exp(-self.cfg.pour_aim_sharpness * _pour_point_xy_dist)

        # [fix] cup CENTER XY ← target 기반 pour 위치 보상 (각도 무관, monotonic, local max 없음)
        # pour_point(rim) 대신 cup center 사용: tilt 각도에 상관없이 cup이 target 위에 올수록 단조 증가
        # 자연스럽게 소스컵 center가 타겟컵 위로 이동 → 어떤 각도에서도 비드가 흘러 들어가게 됨
        r_cup_center_pour = torch.exp(-self.cfg.pour_center_xy_scale * self._cup_center_xy_dist)

        posture_guidance_scale = self.cfg.pour_posture_guidance_scale
        r_pour_stage = gate_pour_binary * (
            self.cfg.weight_cross * r_cross
            + self.cfg.weight_capture * r_capture
            + self.cfg.weight_capture_flow * r_capture_gain
            + posture_guidance_scale * self.cfg.weight_cup_center_pour * r_cup_center_pour
            + posture_guidance_scale * self.cfg.weight_pour_align * r_pour_align
            + self.cfg.weight_pour_aim * r_pour_aim  # weight=0, 진단용 계산만 유지
        )

        # [fix] Source drain reward: target에 들어간 비율만큼만 보상
        # 기존: (1-bead_in_source_fraction) 사용 → 테이블에 버려도 source_drain=17.5/step 획득
        #       → spill_cost(2.0/step)보다 훨씬 크므로 테이블 덤프가 local optimum
        # 수정: bead_in_target_fraction 기반 → target에 넣어야만 보상 (테이블 덤프 불가)
        r_source_drain = gate_pour_binary * self.cfg.weight_source_drain * r_capture_gain

        # 첫 비드 유입 시 1회성 보너스: target cup 근처(< pour_binary_xy_thresh)에서만 인정
        # [P1] 멀리서 우연히 굴러든 비드 캡처는 보너스에서 제외 → 우연성 방지
        near_for_capture = self._cup_center_xy_dist < self.cfg.pour_binary_xy_thresh
        first_capture = (
            (self._bead_in_target_fraction > 0.0)
            & near_for_capture
            & (~self._first_capture_bonus_paid)
        )
        r_first_capture = self.cfg.weight_first_capture_bonus * first_capture.float()
        self._first_capture_bonus_paid |= first_capture

        # tilt onset bonus: >60° 기울기 + 근접 조건 첫 달성 시 1회 bridge reward
        # "안전한 직립 유지" 로컬 최적을 벗어나 tilt 탐색을 유도
        tilt_onset = (
            (self._source_up_dot_world < self.cfg.tilt_onset_dot_threshold)
            & (self._cup_center_xy_dist < self.cfg.tilt_onset_dist_threshold)
            & (~self._tilt_onset_bonus_paid)
        )
        # [test5] directional onset bonus: 올바른 방향으로 기울어야만 보상 (방향 cos 가중치)
        # dir_cos.clamp(0,1): 0=수직/반대방향(보상 없음), 1=완벽한 target 방향(전액 보상)
        # test3의 방향 무관 onset → 틀린 방향 tilting 문제 수정
        r_tilt_onset = (
            self.cfg.weight_tilt_onset_bonus
            * self._directional_tilt_cos.clamp(0.0, 1.0)
            * tilt_onset.float()
        )
        self._tilt_onset_bonus_paid |= tilt_onset

        # ---- [test4] Directional tilt reward ----
        # cup을 target 방향으로 기울일 때만 보상 → 방향 gradient 제공
        # tilt_strength: 기울기 크기 (0=직립, 1=180°)  — upright일 때 directional_tilt_cos 불안정 방지
        # dir_cos_reward: [0,1]  1=target 방향 완벽 정렬, 0=반대 방향
        # g_ready gate: target 근처에서만 활성 (먼 거리에서 방향 무관 tilting 방지)
        tilt_strength = ((1.0 - self._source_up_dot_world) / 2.0).clamp(0.0, 1.0)
        # cos=+1(target 방향 올바른 기울기)=reward 1, cos<=0(수직/반대방향)=reward 0
        # r_tilt_onset과 동일한 패턴: clamp(0,1) 사용
        dir_cos_reward = self._directional_tilt_cos.clamp(0.0, 1.0)
        r_dir_tilt = self._g_ready * self.cfg.weight_dir_tilt * tilt_strength * dir_cos_reward

        # ---- Outcome and costs ----
        # ADR: success 기준을 낮은 fill_ratio에서 시작해 점진적으로 상향
        success_fill_ratio = (
            self.success_adr.get_param("success", "fill_ratio")
            if self.success_adr is not None
            else self.cfg.success_target_fill_ratio
        )

        # [P1] success도 target cup 근처(< pour_binary_xy_thresh)에서 달성해야 인정
        success_now = (
            (self._bead_in_target_fraction >= success_fill_ratio)
            & (self._spill_ratio <= self.cfg.success_spill_max)
            & (self._cup_center_xy_dist < self.cfg.pour_binary_xy_thresh)
        )

        # 기본 성공 보상(바이너리)
        r_success = success_now.float()

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
        spill_weight = (
            self.spill_adr.get_param("reward", "spill_weight")
            if self.spill_adr is not None
            else self.cfg.weight_spill
        )
        
        # Smooth premature tilt penalty: only active when far from target (g_ready is low)
        # [P0] tilt_amount = (1 - dot) / 2, clamped to [0, 1]
        # 직립(dot=1):   tilt_amount=0 → 페널티 0     (직립은 안전)
        # 60°(dot=0.5):  tilt_amount=0.25
        # 90°(dot=0):    tilt_amount=0.5
        # 100°(dot≈-0.17): tilt_amount=0.585          (pour 각도에도 페널티 부과)
        # 180°(dot=-1):  tilt_amount=1.0 (최대)
        # → 기울기가 클수록, 멀리 있을수록(g_ready 낮을수록) 페널티 증가
        # [test2 분석] 이전 dot.clamp(0,1) 방식: 90° 이상에서 페널티=0 → 멀리서도 마음대로 쏟아붓기 허용
        tilt_amount = ((1.0 - self._source_up_dot_world) / 2.0).clamp(0.0, 1.0)
        premature_tilt_cost = (1.0 - self._g_ready) * tilt_amount
        # grasp quality loss는 hold 직후/원거리에서 즉시 벌점하지 않고,
        # hold 종료 후 지연 + g_ready gate를 통과한 구간에서만 적용한다.
        grasp_loss_cost_raw = 1.0 - full_grasp_flag
        grasp_loss_step_gate = (
            self.episode_length_buf
            >= (self.cfg.episode_hold_steps + self.cfg.grasp_loss_hold_off_steps)
        ).float()
        grasp_ready_den = max(1.0 - self.cfg.grasp_loss_ready_gate_min, 1e-6)
        grasp_loss_ready_gate = torch.clamp(
            (self._g_ready - self.cfg.grasp_loss_ready_gate_min) / grasp_ready_den,
            min=0.0,
            max=1.0,
        )
        grasp_loss_cost = grasp_loss_cost_raw * grasp_loss_step_gate * grasp_loss_ready_gate
        # [Phase-2 Step 9] action_rate: palm(6D) / finger(5D) 분리
        # grasp v9의 action_smoothness_palm/finger 패턴과 동일
        palm_delta   = (self.actions[:, :6] - self.prev_actions[:, :6]).pow(2).sum(dim=-1)
        finger_delta = (self.actions[:, 6:] - self.prev_actions[:, 6:]).pow(2).sum(dim=-1)
        action_rate_penalty = (
            self.cfg.weight_action_rate_palm   * palm_delta
            + self.cfg.weight_action_rate_finger * finger_delta
        )

        # ---- Phase-0 진단 + Phase-1 Step 4: arm joint velocity / acceleration ----
        arm_qd = self.robot.data.joint_vel[:, self.arm_dof_indices]
        arm_qd_l2 = arm_qd.norm(dim=-1)                          # (num_envs,) L2 norm
        arm_qd_max = arm_qd.abs().max(dim=-1).values              # (num_envs,) per-joint max
        arm_qacc = (arm_qd - self._prev_arm_joint_vel).norm(dim=-1)  # (num_envs,) acc proxy
        # pouring phase에서의 arm vel (cup이 가까울 때)
        in_tilt_phase = (self._cup_center_xy_dist < self.cfg.tilt_action_gate_xy_far).float()
        tilt_phase_arm_vel = (arm_qd_l2 * in_tilt_phase).sum() / (in_tilt_phase.sum() + 1e-6)

        # [Phase-1 Step 4] arm joint vel^2 sum penalty (clipped)
        arm_qd_sq_sum = arm_qd.pow(2).sum(dim=-1).clamp_max(self.cfg.arm_joint_vel_sq_clip)
        arm_acc_vec = arm_qd - self._prev_arm_joint_vel          # 현재 acc 벡터 (N, DOF)
        arm_qacc_sq_sum = arm_acc_vec.pow(2).sum(dim=-1)

        # [Phase-1 Step 6] jerk = d(acc)/dt (acc 벡터 변화량)
        arm_jerk_vec = arm_acc_vec - self._prev_arm_joint_acc    # (N, DOF)
        arm_jerk_sq_sum = arm_jerk_vec.pow(2).sum(dim=-1)
        arm_jerk_l2 = arm_jerk_vec.norm(dim=-1)

        demo_terms = self._get_demo_pose_reward_terms(
            arm_qd_l2=arm_qd_l2,
            arm_jerk_l2=arm_jerk_l2,
            palm_delta=palm_delta,
        )

        # [P1] arm vel penalty: approach 구간도 낮은 가중치로 적용 (arm_vel_tilt_gate_only=False)
        # test4: cost_arm_vel=0.001/step → 사실상 0, approach에서 arm이 너무 빠르게 이동
        # approach 구간: weight_arm_joint_vel_approach (tilt 구간의 1/4)
        # tilt 구간: weight_arm_joint_vel (기존 값 유지)
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
            + r_approach_z_rim
            + r_prepour_stage
            + r_pour_stage
            + r_source_drain
            + r_first_capture
            + r_tilt_onset
            + r_dir_tilt
            + demo_terms["r_demo_arm_pose"]
            + demo_terms["r_demo_palm_pose"]
            + self.cfg.weight_success * r_success
            + overfill_bonus
            - spill_weight * spill_cost
            - self.cfg.weight_premature_tilt * premature_tilt_cost
            - self.cfg.weight_grasp_loss * grasp_loss_cost
            - action_rate_penalty
            - arm_vel_cost
            - demo_terms["cost_demo_smooth"]
            - demo_terms["cost_thumb_grip"]
        )

        self._prev_mouth_xy_distance.copy_(self._mouth_xy_distance)
        self._prev_bead_in_target_fraction.copy_(self._bead_in_target_fraction)
        self._prev_arm_joint_vel.copy_(arm_qd)
        self._prev_arm_joint_acc.copy_(arm_acc_vec)   # [Step 6] jerk 계산용

        # ---- ADR increment (success-rate 기반) ----
        _ep_success_rate = self._successful_episodes / max(self._total_episodes, 1)
        if self.spill_adr is not None:
            self.spill_adr.maybe_increment(_ep_success_rate)
        if self.noise_adr is not None:
            self.noise_adr.maybe_increment(_ep_success_rate)
        if self.success_adr is not None:
            self.success_adr.maybe_increment(_ep_success_rate)

        # ---- Logging to TensorBoard ----
        self.extras["r_hold"] = r_hold.mean()
        self.extras["r_lift"] = r_lift.mean()
        self.extras["r_approach"] = r_approach.mean()
        self.extras["r_approach_z_rim"] = r_approach_z_rim.mean()
        self.extras["z_cup_center_err_m"] = z_rim_err.mean()  # cup center z vs ideal (tilt-invariant)
        self.extras["r_prepour"] = r_prepour_stage.mean()
        self.extras["r_pour"] = r_pour_stage.mean()
        self.extras["r_source_drain"] = r_source_drain.mean()
        self.extras["r_capture_gain"] = r_capture_gain.mean()
        self.extras["bead_in_source"] = self._bead_in_source_fraction.mean()
        self.extras["r_pour_align"] = (self.cfg.weight_pour_align * r_pour_align).mean()
        self.extras["r_pour_aim"] = (self.cfg.weight_pour_aim * r_pour_aim).mean()  # weight=0, 진단용
        self.extras["pour_point_xy_dist"] = _pour_point_xy_dist.mean()
        self.extras["r_cup_center_pour"] = (self.cfg.weight_cup_center_pour * r_cup_center_pour * gate_pour_binary).mean()
        self.extras["gate_pour_binary"] = gate_pour_binary.mean()  # [P3] binary gate 활성 비율
        self.extras["source_empty_steps"] = self._source_empty_steps.float().mean()
        self.extras["r_success_weighted"] = (self.cfg.weight_success * r_success).mean()
        self.extras["r_success_overfill"] = (
            overfill_bonus.mean() if isinstance(overfill_bonus, torch.Tensor) else 0.0
        )
        self.extras["cost_spill"] = spill_cost.mean()
        self.extras["spill_weight"] = torch.tensor(float(spill_weight), device=self.device)
        self.extras["cost_premature_tilt"] = premature_tilt_cost.mean()
        self.extras["cost_grasp_loss_raw"] = grasp_loss_cost_raw.mean()
        self.extras["grasp_loss_step_gate"] = grasp_loss_step_gate.mean()
        self.extras["grasp_loss_ready_gate"] = grasp_loss_ready_gate.mean()
        self.extras["cost_grasp_loss"] = grasp_loss_cost.mean()
        self.extras["r_tilt_onset"] = r_tilt_onset.mean()
        self.extras["g_ready"] = self._g_ready.mean()
        self.extras["g_pour"] = self._g_pour.mean()
        self.extras["mouth_xy_dist"] = self._mouth_xy_distance.mean()
        self.extras["cup_center_xy_dist"] = self._cup_center_xy_dist.mean()
        self.extras["bead_in_target"] = self._bead_in_target_fraction.mean()
        self.extras["bead_cross"] = self._bead_cross_fraction.mean()
        self.extras["spill_ratio"] = self._spill_ratio.mean()
        # [test4] directional tilt 진단
        self.extras["r_dir_tilt"] = r_dir_tilt.mean()
        self.extras["directional_tilt_cos"] = self._directional_tilt_cos.mean()
        # Phase-0 진단 메트릭: arm joint velocity / acceleration
        self.extras["arm_joint_vel_l2_mean"] = arm_qd_l2.mean()
        self.extras["arm_joint_vel_max_mean"] = arm_qd_max.mean()
        self.extras["arm_joint_acc_l2_mean"] = arm_acc_vec.norm(dim=-1).mean()
        self.extras["arm_joint_jerk_l2_mean"] = arm_jerk_l2.mean()
        self.extras["tilt_phase_arm_vel"] = tilt_phase_arm_vel
        self.extras["r_demo_arm_pose"] = demo_terms["r_demo_arm_pose"].mean()
        self.extras["r_demo_palm_pose"] = demo_terms["r_demo_palm_pose"].mean()
        self.extras["cost_demo_smooth"] = demo_terms["cost_demo_smooth"].mean()
        self.extras["cost_thumb_grip"] = demo_terms["cost_thumb_grip"].mean()
        self.extras["demo_arm_joint_err"] = demo_terms["demo_arm_joint_err"].mean()
        self.extras["demo_palm_pos_err"] = demo_terms["demo_palm_pos_err"].mean()
        self.extras["demo_palm_rot_err"] = demo_terms["demo_palm_rot_err"].mean()
        # Phase-1 Step 4/5/6: arm vel/acc/jerk cost 로깅 (in_tilt_phase gate 포함)
        self.extras["cost_arm_vel"] = (self.cfg.weight_arm_joint_vel * arm_qd_sq_sum * in_tilt_phase).mean()
        self.extras["cost_arm_acc"] = (self.cfg.weight_arm_joint_acc * arm_qacc_sq_sum * in_tilt_phase).mean()
        self.extras["cost_arm_jerk"] = (self.cfg.weight_arm_joint_jerk * arm_jerk_sq_sum * in_tilt_phase).mean()
        self.extras["cost_arm_vel_approach"] = (self.cfg.weight_arm_joint_vel_approach * arm_qd_sq_sum * approach_only).mean()
        # Phase-2 Step 9: action_rate 분리 로깅
        self.extras["cost_action_rate_palm"] = (self.cfg.weight_action_rate_palm * palm_delta).mean()
        self.extras["cost_action_rate_finger"] = (self.cfg.weight_action_rate_finger * finger_delta).mean()
        # j7 진단 로깅 (warmstart 자세 일관성 확인)
        j7_cur = self.robot.data.joint_pos[:, self.arm_dof_indices[6]]
        self.extras["arm_j7_mean"] = j7_cur.mean()
        self.extras["arm_j7_abs_max"] = j7_cur.abs().max()

        if self.spill_adr is not None:
            self.extras["adr_spill_progress"] = torch.tensor(
                self.spill_adr.progress, device=self.device
            )
        if self.noise_adr is not None:
            self.extras["adr_noise_progress"] = torch.tensor(
                self.noise_adr.progress, device=self.device
            )
        if self.success_adr is not None:
            self.extras["adr_success_progress"] = torch.tensor(
                self.success_adr.progress, device=self.device
            )
            self.extras["success_fill_ratio"] = torch.tensor(
                float(success_fill_ratio), device=self.device
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

        success_by_fill = (
            (self._bead_in_target_fraction >= success_fill_ratio)
            & (self._spill_ratio <= self.cfg.success_spill_max)
        )
        self.success_flag.copy_(success_by_fill)
        self.episode_success_buf |= self.success_flag   # 에피소드 중 한 번이라도 성공 시 True

        terminated = out_x | out_y | fallen | dropped_by_force | self.success_flag | source_drained
        truncated  = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # 에피소드 종료 시점(reset 직전)의 terminal state를 warmstart 캐시에 저장
        self._maybe_store_warmstart_successes(env_ids)

        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        n = len(env_ids)

        # ---- episode 성공 집계 후 클리어 ----
        self._total_episodes += n
        self._successful_episodes += int(self.episode_success_buf[env_ids].sum().item())
        self.episode_success_buf[env_ids] = False
        self._warmstart_only_close[env_ids] = False
        self._warmstart_finger_action_floor[env_ids] = -1.0
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

        left_cup_pose = self._get_left_cup_fk_pose(env_ids=env_ids)
        self._left_target_cup_fixed_pose_w[env_ids] = left_cup_pose
        self.left_target_cup.write_root_pose_to_sim(left_cup_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        if self._warmstart_collect_mode:
            self._hide_beads(env_ids)
        else:
            bead_state = self._sample_bead_states_inside_cup(cup_root_state[:, :7])
            self.beads.write_object_state_to_sim(bead_state, env_ids=env_ids)
        # 일반 reset은 비드를 즉시 소환 (또는 collect mode에서는 숨김)
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
        self._prev_bead_target_local_z[env_ids].fill_(10.0)
        self._needs_grasp_init_update[env_ids] = True   # 다음 스텝에 palm local init 갱신
        self._bead_cross_count[env_ids] = 0
        self._bead_cross_fraction[env_ids] = 0.0
        self._prev_bead_ever_in_target_count[env_ids] = 0
        self._bead_in_target_fraction[env_ids] = 0.0
        self._prev_bead_in_target_fraction[env_ids] = 0.0
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

        # warmstart cache j7 분포 확인 (j7 = arm index 6)
        j7_vals = self._warmstart_arm_pos[:self._warmstart_cache_count, 6]
        print(
            f"[5g_pour_right_v3] collected {self._warmstart_cache_count} warmstart success states. "
            f"j7: min={j7_vals.min().item():.3f} max={j7_vals.max().item():.3f} "
            f"mean={j7_vals.mean().item():.3f} std={j7_vals.std().item():.3f}",
            flush=True,
        )

    def _maybe_store_warmstart_successes(self, env_ids: Sequence[int]) -> None:
        """에피소드 종료 env_ids에 대해 terminal state를 warmstart 캐시에 저장.

        _get_dones() 매 step 호출이 아닌 _reset_idx() 초입(reset 직전)에서만 호출되므로,
        안정적인 hold 상태(에피소드 최종 프레임)만 캐시에 쌓인다.
        """
        if not self._warmstart_collect_mode:
            return
        if self._warmstart_cache_count >= self._warmstart_arm_pos.shape[0]:
            return
        if len(env_ids) == 0:
            return

        if not isinstance(env_ids, torch.Tensor):
            ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            ids = env_ids.long()

        lifted = self.object_pos[ids, 2] > (self.object_init_pos[ids, 2] + self.cfg.lift_success_height)
        grasped = self.num_contacts_buf[ids] >= MIN_CONTACTS_FOR_SUCCESS
        upright = self._source_up_axis_w[ids, 2] > 0.90
        j7 = self.robot.data.joint_pos[ids, self.arm_dof_indices[6]]
        j7_in_range = (j7 >= 0.20) & (j7 <= 1.50)
        warmstart_success = lifted & grasped & upright & j7_in_range

        local_ok = warmstart_success.nonzero(as_tuple=False).squeeze(-1)
        if local_ok.numel() == 0:
            return

        success_env_ids = ids[local_ok]

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
        self._warmstart_cup_pose[start:end, 3:7] = self.cup.data.root_quat_w[success_env_ids]
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

        warmstart_palm_pose = palm_pose.clone()
        warmstart_palm_pose[:, :3] = torch.max(
            torch.min(warmstart_palm_pose[:, :3], self.palm_maxs[:3].unsqueeze(0)),
            self.palm_mins[:3].unsqueeze(0),
        )
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

        # hold phase 동안 비드를 숨겨서 physics 초기화 충돌 방지.
        # 실제 소환은 _pre_physics_step 에서 hold 종료 시점에 수행.
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
        self._prev_bead_target_local_z[env_ids].fill_(10.0)
        self._needs_grasp_init_update[env_ids] = True   # 다음 스텝에 palm local init 갱신
        self._bead_cross_count[env_ids] = 0
        self._bead_cross_fraction[env_ids] = 0.0
        self._prev_bead_ever_in_target_count[env_ids] = 0
        self._bead_in_target_fraction[env_ids] = 0.0
        self._prev_bead_in_target_fraction[env_ids] = 0.0
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
        self._prev_arm_joint_acc[env_ids].zero_()   # [Step 6]
        self._ema_palm_action[env_ids].zero_()       # [Step 7]

        self.actions[env_ids, :6] = 0.0
        self.actions[env_ids, 6:] = 1.0
        self.prev_actions[env_ids, :6] = 0.0
        self.prev_actions[env_ids, 6:] = 1.0
        self._prev_mouth_xy_distance[env_ids] = 0.0
        self._pre_pour_ready_steps[env_ids] = 0
        self.success_flag[env_ids] = False

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
            cup_x_local = cup_pose_local[:, 0]
            cup_y_local = cup_pose_local[:, 1]
            cup_z_local = cup_pose_local[:, 2]
            j7_reset = arm_pos[:, 6]
            print(
                "[5g_pour_right_v3][warmstart_reset] "
                f"cup_x_local mean={cup_x_local.mean().item():.4f} "
                f"cup_y_local mean={cup_y_local.mean().item():.4f} min={cup_y_local.min().item():.4f} max={cup_y_local.max().item():.4f} | "
                f"cup_z_local mean={cup_z_local.mean().item():.4f} "
                f"min={cup_z_local.min().item():.4f} max={cup_z_local.max().item():.4f} | "
                f"mouth_xy mean={mouth_xy_distance.mean().item():.4f} "
                f"min={mouth_xy_distance.min().item():.4f} max={mouth_xy_distance.max().item():.4f} | "
                f"mouth_z_clearance mean={mouth_z_clearance.mean().item():.4f} "
                f"min={mouth_z_clearance.min().item():.4f} max={mouth_z_clearance.max().item():.4f} | "
                f"warmstart_palm_z mean={warmstart_palm_pose[:, 2].mean().item():.4f} | "
                f"j7 min={j7_reset.min().item():.3f} max={j7_reset.max().item():.3f} "
                f"mean={j7_reset.mean().item():.3f} std={j7_reset.std().item():.3f}",
                flush=True,
            )
            self._warmstart_reset_debug_printed = True
