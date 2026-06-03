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

        # pregrasp palm pose 버퍼 (에피소드별 delta action 기준점)
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

        # posture-gated crossover: demo/palm 자세 weight 감쇠 + pour 활성(pour_warmup 대체)을
        # 동시 구동하는 alpha(0→1). demo_arm_joint_err(j1-5)가 threshold 안에 든 env 비율이
        # trigger_rate 이상이면 한 칸 전진(ratchet). success-ADR의 chicken-egg 회피.
        # 래치-후-단조 상태: 래치 전 alpha=0, 래치 후 시간 단조 ramp.
        self._crossover_alpha = 0.0
        self._pour_latched = False
        self._pour_latch_step = 0.0
        self._posture_above_count = 0
        self._last_latch_check = 0.0

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

        # Transport 고정 타겟: demo pour palm pose (a11~a20 평균, env-local 프레임).
        # 정책이 "좋은 arm+palm 자세"에 도달하게 하는 신호. j0~4는 IK가 자동으로 따라옴.
        # quat은 xyzw(cfg) → wxyz(Isaac body_quat_w)로 변환 저장.
        self._transport_palm_pos = to_torch(self.cfg.transport_palm_pos, device=self.device)
        _q_xyzw = to_torch(self.cfg.transport_palm_quat_xyzw, device=self.device)
        self._transport_palm_quat_wxyz = torch.cat([_q_xyzw[3:4], _q_xyzw[:3]], dim=0)

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
        self._all_beads_bonus_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._first_capture_bonus_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pre_pour_ready_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._no_tip_force_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._source_empty_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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

        # [Phase-1 Step 7] EMA palm action smoothing: Fabrics에 smooth 궤적 전달
        # action_rate_penalty는 raw self.actions 기반 유지 (training gradient 보존)
        self._ema_palm_action.copy_(
            self.cfg.ema_action_alpha * palm_action
            + (1.0 - self.cfg.ema_action_alpha) * self._ema_palm_action
        )

        if self._warmstart_collect_mode:
            # v7-2 학습과 동일한 파이프라인: euler 직접 덧셈 + euler_zyx Fabrics
            delta = scale(self._ema_palm_action, self.delta_mins_warmstart_collect, self.delta_maxs_warmstart_collect)
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
            delta = scale(self._ema_palm_action, self.delta_mins, self.delta_maxs)   # (N, 6)
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
            # null-space attractor: j1~j5를 데모(a11-a20) 통계 기반 target으로 당김.
            # j1~j5: arm configuration (pour 자세 branch 결정)
            # j6: 정책이 tilt 제어 (nullspace 제외)
            # j7: tilt 제어, 내회전 방지 (min=0.20 유지)
            _null_cfg = self.fabric_q.detach().clone()
            _null_cfg[:, 0] = torch.clamp(_null_cfg[:, 0] * 0.55 + 0.37 * 0.45, max=0.46)   # j1: alpha 0.45 (min clamp 제거)
            _null_cfg[:, 1] = torch.clamp(_null_cfg[:, 1] * 0.95 + 0.39 * 0.05, min=0.00, max=1.05)
            _null_cfg[:, 2] = torch.clamp(_null_cfg[:, 2] * 0.95 + (-0.24) * 0.05, min=-0.74, max=0.38)
            _null_cfg[:, 3] = _null_cfg[:, 3] * 0.95 + 1.84 * 0.05   # j4: demo mean +1.84
            _null_cfg[:, 4] = _null_cfg[:, 4] * 0.75 + (-1.16) * 0.25  # j5: alpha 0.25 (branch switch 강화)
            _null_cfg[:, 6] = torch.clamp(_null_cfg[:, 6] * 0.95 + 0.63 * 0.05, min=0.20, max=1.13)
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
        # rim center (world)
        _rim_center_w = self.cup.data.root_pos_w + quat_apply(
            self.cup.data.root_quat_w,
            self._source_cup_pour_point_pos_b.unsqueeze(0).expand(n, -1),
        )
        # cup up axis (world)
        _cup_up_w = quat_apply(
            self.cup.data.root_quat_w,
            self._source_cup_up_axis_b.unsqueeze(0).expand(n, -1),
        )
        # gravity direction perpendicular to cup up → points toward lowest rim
        _world_down = _cup_up_w.new_zeros(n, 3)
        _world_down[:, 2] = -1.0
        _dot = (_world_down * _cup_up_w).sum(dim=-1, keepdim=True)
        _gravity_perp = _world_down - _dot * _cup_up_w
        _gravity_perp_norm = _gravity_perp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        _gravity_perp_hat = _gravity_perp / _gravity_perp_norm
        self._source_pour_point_w = _rim_center_w + self.cfg.source_outer_radius * _gravity_perp_hat
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
    # Observations: Actor 110D | Critic 140D
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
            valid.view(NUM_FINGER_ACTION, 4).sum(dim=-1).clamp(min=1).to(progress_20.dtype)
        )
        return (
            progress_20.view(-1, NUM_FINGER_ACTION, 4).sum(dim=-1)
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
        palm_center_pos = palm_center_pos_clean + torch.randn_like(palm_center_pos_clean) * σ_bp
        right_cup_pos = right_cup_pos_clean + torch.randn_like(right_cup_pos_clean) * σ_cp
        left_cup_pos = left_cup_pos_clean + torch.randn_like(left_cup_pos_clean) * σ_cp
        source_pour_point = source_pour_point_clean + torch.randn_like(source_pour_point_clean) * σ_cp
        target_opening = target_opening_clean + torch.randn_like(target_opening_clean) * σ_cp

        right_cup_pos_rel_palm = right_cup_pos - palm_center_pos
        left_cup_pos_rel_palm = left_cup_pos - palm_center_pos
        pour_point_to_opening = target_opening - source_pour_point

        finger_grasp_progress = self._finger_grasp_progress(finger_joint_pos)
        binary_contact = self.binary_contact_buf.float()
        last_actions = self.actions
        last_palm_actions = self.actions[:, :NUM_PALM_ACTION]

        # transport_summary (8D): pour 기하학 + ρ gate
        transport_summary = torch.stack([
            self._mouth_distance,
            self._mouth_xy_distance,       # pour point 기반 (pour phase 정밀도)
            self._cup_center_xy_dist,      # cup center 기반 (approach 기준, tilt-invariant)
            self._mouth_z_clearance,
            self._source_up_dot_world,
            self._directional_tilt_cos,
            self._mouth_alignment_cos,
            self._rho,                     # binary pour gate (cup_center_xy_dist < thresh)
        ], dim=-1)   # (N, 8)

        flow_summary = torch.stack([
            self._bead_in_source_delta,
            self._bead_in_target_delta,
            self._bead_cross_delta,
            self._spill_delta,
        ], dim=-1)   # (N, 4)

        # tip force (v8처럼, 실로봇 FT 센서 직결, sim2real 가능)
        tip_force_norm = (self.contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)

        actor_obs = torch.cat([
            arm_joint_pos,              # 7
            arm_joint_vel,              # 7
            finger_grasp_progress,      # 5
            right_cup_pos_rel_palm,     # 3
            right_cup_quat_clean,       # 4
            left_cup_pos_rel_palm,      # 3
            pour_point_to_opening,      # 3
            source_pour_axis_clean,     # 3
            source_up_axis_clean,       # 3
            # target_up_axis 제거: 타겟 컵은 항상 직립 → 항상 [0,0,1], 정보 없음
            transport_summary,          # 8
            last_palm_actions,          # 6
            # bead 상태: actor가 pour 결과를 직접 관측 (reward 항목 대응)
            self._bead_in_source_fraction.unsqueeze(1),  # 1 (소스 잔량)
            self._bead_in_target_fraction.unsqueeze(1),  # 1 (타겟 유입량, r_capture 대응)
            self._bead_cross_fraction.unsqueeze(1),      # 1 (mouth 통과율, r_cross weight=20 대응)
            self._spill_ratio.unsqueeze(1),              # 1 (유출율, spill_cost weight=10 대응)
            flow_summary,               # 4 (source/target/cross/spill step delta)
        ], dim=-1)   # 60D

        if actor_obs.shape[1] != NUM_OBSERVATIONS:
            raise RuntimeError(
                f"[pour_v3] Actor obs dim mismatch: {actor_obs.shape[1]} != {NUM_OBSERVATIONS}"
            )

        # ==== Critic extra obs (30D) ====
        cup_height_delta = (right_cup_pos_clean[:, 2] - self.object_init_pos[:, 2]).unsqueeze(1)

        distal_binary     = self.distal_binary_contact_buf.float()
        distal_force_norm = (self.distal_contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)


        # critic base obs (110D) — full-state value estimation, actor LSTM layout과 분리
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
            self._rho.unsqueeze(1),                            # 1 (binary pour gate)
        ], dim=-1)   # 140D

        if critic_obs.shape[1] != NUM_CRITIC_OBSERVATIONS:
            raise RuntimeError(
                f"[pour_v3] Critic obs dim mismatch: {critic_obs.shape[1]} != {NUM_CRITIC_OBSERVATIONS}"
            )

        return {"policy": actor_obs, "critic": critic_obs}

    def _get_demo_pose_reward_terms(self) -> dict[str, torch.Tensor]:
        zero = torch.zeros(self.num_envs, device=self.device)
        if self.demo_pose_reference is None:
            return {"r_demo_arm_pose": zero, "demo_arm_joint_err": zero}

        ref = self.demo_pose_reference
        arm_q = self.robot.data.joint_pos[:, self.arm_dof_indices]  # (N, 7)

        # Nearest-Neighbor in joint space + look-ahead
        demo_arm = ref.arm_joint_pos  # (T, 7)
        T_demo = demo_arm.shape[0]
        aa = (arm_q * arm_q).sum(dim=-1, keepdim=True)
        bb = (demo_arm * demo_arm).sum(dim=-1).unsqueeze(0)
        ab = arm_q @ demo_arm.T
        nn_idx = (aa + bb - 2.0 * ab).argmin(dim=-1)
        K = int(self.cfg.demo_nn_lookahead_frames)
        target_idx = (nn_idx + K).clamp(max=T_demo - 1)
        target_arm_q = demo_arm[target_idx]  # (N, 7)

        # j1~j5(idx 0~4)만 branch 결정에 사용. j6/j7은 tilt 자유도.
        arm_std5 = ref.arm_joint_std[:5].clamp(min=0.20)
        arm_norm_err = torch.norm((arm_q[:, :5] - target_arm_q[:, :5]) / arm_std5, dim=-1)
        demo_arm_joint_err = arm_norm_err / math.sqrt(5.0)
        r_demo_arm_pose = torch.exp(-demo_arm_joint_err)

        near_gate = torch.exp(-torch.square(self._cup_center_xy_dist / max(self.cfg.demo_pose_near_gate_xy, 1e-6)))
        warmup_steps = max(int(self.cfg.demo_pose_warmup_steps), 1)
        step_count = float(getattr(self, "common_step_counter", 0))
        warmup = min(step_count / float(warmup_steps), 1.0)
        gate = near_gate * warmup

        demo_w = getattr(self, "_demo_arm_pose_w", self.cfg.weight_demo_arm_pose)
        return {
            "r_demo_arm_pose": gate * demo_w * r_demo_arm_pose,
            "demo_arm_joint_err": demo_arm_joint_err,
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

        thumb_force = self.contact_force_raw[:, 0]
        others_avg_force = self.contact_force_raw[:, 1:].mean(dim=-1)
        others_count = self.binary_contact_buf[:, 1:].sum(dim=-1)
        full_grasp_flag = (
            self.binary_contact_buf[:, 0] & (others_count >= self.cfg.contact_maintain_min_others)
        ).float()
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

        # tilt-phase aware hold:
        # transport(직립) → full grip 요구, active pour(120°) → contact 요구 완화
        # tilt_level: 0=직립(cup up), 1=완전 도립
        tilt_amount = ((1.0 - self._source_up_dot_world) / 2.0).clamp(0.0, 1.0)
        contact_gate = (1.0 - 0.7 * tilt_amount)   # 직립=1.0, 120°tilt≈0.3
        upright_gate = (1.0 - tilt_amount).clamp(0.0, 1.0)

        # Hold: grasp 유지 (tilt-phase aware)
        r_hold = (
            self.cfg.weight_grasp_maintain * grasp_maintain_reward
            + self.cfg.weight_contact_maintain * full_grasp_flag * contact_gate
            + r_force_balance * upright_gate
            + r_finger_curl
        )

        # Curriculum warmup (common_step_counter 기반 선형 증가)
        step_count = float(getattr(self, "common_step_counter", 0))
        # 래치-후-단조: 래치 전 alpha=0(Stage A 자세 학습만), 래치 후 시간 단조 ramp.
        # demo/palm weight는 정적 유지 → j1-5 자세를 Stage B 내내 hold (감쇠 폐기).
        if self.cfg.enable_weight_crossover:
            if self._pour_latched:
                a = min(
                    (step_count - self._pour_latch_step)
                    / max(self.cfg.crossover_monotonic_steps, 1),
                    1.0,
                )
            else:
                a = 0.0
            self._crossover_alpha = a
            self._demo_arm_pose_w = self.cfg.weight_demo_arm_pose
            self._palm_pose_w = self.cfg.weight_palm_pose
            pour_warmup = a
        else:
            self._crossover_alpha = 1.0
            self._demo_arm_pose_w = self.cfg.weight_demo_arm_pose
            self._palm_pose_w = self.cfg.weight_palm_pose
            pour_warmup = min(step_count / max(self.cfg.curriculum_pour_warmup_steps, 1), 1.0)
        bead_warmup = min(
            max(step_count - self.cfg.curriculum_bead_warmup_start, 0.0)
            / max(self.cfg.curriculum_bead_warmup_steps, 1),
            1.0,
        )

        # Transport Stage 1a: Cartesian 근접 (cup_center_xy 기반, 거친 approach gradient)
        _transport_dist = (self._cup_center_xy_dist - self.cfg.cup_transport_saturate_xy).clamp(min=0.0)
        r_dist_to_target = self.cfg.weight_dist_to_target * torch.exp(
            -self.cfg.dist_to_target_exp_scale * _transport_dist
        )

        # Transport Stage 1b: "좋은 arm+palm 자세" — demo pour palm pose 추종.
        # palm pose(pos+rot)가 demo pour 자세에 가까울수록 보상. Fabrics IK가 j0~4를 따라가게 함.
        palm_pos_w = self.robot.data.body_pos_w[:, self.palm_body_index] - self.scene.env_origins
        palm_quat_wxyz = self.robot.data.body_quat_w[:, self.palm_body_index]
        palm_pos_err = torch.norm(palm_pos_w - self._transport_palm_pos, dim=-1)
        _quat_dot = torch.abs(
            (palm_quat_wxyz * self._transport_palm_quat_wxyz).sum(dim=-1)
        ).clamp(max=1.0)
        palm_rot_err = 2.0 * torch.acos(_quat_dot)
        palm_pose_err = palm_pos_err + palm_rot_err
        palm_w = getattr(self, "_palm_pose_w", self.cfg.weight_palm_pose)
        r_palm_pose = palm_w * torch.exp(
            -self.cfg.palm_pose_pos_sharpness * palm_pos_err
            - self.cfg.palm_pose_rot_sharpness * palm_rot_err
        )

        # Pour: Stage 3 (r_pour_dist soft gate로 먼저 계산)
        target_tilt_cos = math.cos(math.radians(self.cfg.pour_tilt_target_deg))
        # pour_aligned_gate: pour_point 정렬도 → r_tilt 증폭 (pour_point pivot 유도)
        pour_aligned_gate = torch.exp(
            -self.cfg.pour_align_gate_scale * self._mouth_xy_distance
        )
        r_tilt = (
            pour_aligned_gate
            * torch.exp(-self.cfg.pour_tilt_sharpness * torch.abs(self._source_up_dot_world - target_tilt_cos))
        )

        # Pour distance: Stage 2 (ρ gate + pour_warmup)
        # initial_tilt_gate: 15° 이상 tilt 후 활성 → r_tilt와 충돌 제거
        # z_window: pour_point Z soft gate — 1~5cm 활성 구역, 정책이 hinge 위치 탐색
        _tilt_threshold_norm = (1.0 - math.cos(math.radians(self.cfg.pour_point_tilt_threshold_deg))) / 2.0
        initial_tilt_gate = (tilt_amount / max(_tilt_threshold_norm, 1e-6)).clamp(0.0, 1.0)
        z_lower = torch.clamp(self._mouth_z_clearance / self.cfg.z_window_lower_ramp, 0.0, 1.0)
        z_upper = torch.clamp(
            (self.cfg.z_window_upper_end - self._mouth_z_clearance) / self.cfg.z_window_upper_ramp,
            0.0, 1.0,
        )
        z_window = z_lower * z_upper
        r_pour_dist = (
            self._rho
            * pour_warmup
            * z_window
            * initial_tilt_gate
            * self.cfg.weight_pour_dist
            * torch.exp(-self.cfg.pour_dist_exp_scale * self._mouth_xy_distance)
        )
        # Stage B z-barrier: pour_point가 림 아래(clearance<margin)로 가라앉는 것만 막는
        # 단방향 penalty. 림 위(clearance≥margin)에서는 0 → 높이는 beads가 결정.
        z_violation = (self.cfg.pour_z_margin - self._mouth_z_clearance).clamp(min=0.0)
        r_pour_z = -(
            self._rho * pour_warmup * self.cfg.weight_pour_z * z_violation
        )

        # DexPour eq.2: 올바른 방향(cos>0)에서 최대, 반대 방향(cos<0)에서 0
        r_align = 0.5 * (1.0 + self._directional_tilt_cos)

        # bead: bead_warmup 별도 스케줄 (bead_warmup_start 이후 점진 활성)
        r_bead_progressive = self.cfg.weight_bead_progressive * (self._bead_in_target_fraction ** 2)
        r_bead_delta = self.cfg.weight_bead_entry_delta * self._bead_in_target_delta.clamp(min=0.0)

        r_pour_stage = self._rho * (
            pour_warmup * (self.cfg.weight_tilt * r_tilt + self.cfg.weight_align * r_align)
            + bead_warmup * (r_bead_progressive + r_bead_delta)
        )
        # directional gate: 타겟 방향으로 배출할 때만 drain 보상 (bead_warmup 연동)
        _drain_dir_gate = 0.5 * (1.0 + self._directional_tilt_cos)
        r_source_drain = (
            self._rho
            * _drain_dir_gate
            * bead_warmup
            * self.cfg.weight_source_drain
            * (1.0 - self._bead_in_source_fraction)
        )

        # Success
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
        overfill_bonus = 0.0
        if self.cfg.weight_success_overfill > 0.0:
            overfill = torch.clamp(
                (self._bead_in_target_fraction - success_fill_ratio)
                / (1.0 - success_fill_ratio + 1e-6),
                min=0.0,
                max=1.0,
            )
            overfill_bonus = self.cfg.weight_success_overfill * overfill

        # Costs
        spill_cost = self._spill_ratio.clamp(min=0.0).sqrt()
        spill_weight = (
            self.spill_adr.get_param("reward", "spill_weight")
            if self.spill_adr is not None
            else self.cfg.weight_spill
        )
        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_dof_indices]

        arm_qd      = self.robot.data.joint_vel[:, self.arm_dof_indices]
        arm_acc_vec  = arm_qd - self._prev_arm_joint_vel
        arm_jerk_vec = arm_acc_vec - self._prev_arm_joint_acc  # noqa: F841

        demo_terms = self._get_demo_pose_reward_terms()

        total = (
            r_hold
            + r_dist_to_target
            + r_palm_pose
            + demo_terms["r_demo_arm_pose"]
            + r_pour_dist
            + r_pour_z
            + r_pour_stage
            + r_source_drain
            + self.cfg.weight_success * r_success
            + overfill_bonus
            - spill_weight * spill_cost
        )

        self._prev_arm_joint_vel.copy_(arm_qd)
        self._prev_arm_joint_acc.copy_(arm_acc_vec)

        # ---- ADR increment ----
        _ep_success_rate = self._successful_episodes / max(self._total_episodes, 1)
        if self.spill_adr is not None:
            self.spill_adr.maybe_increment(_ep_success_rate)
        if self.noise_adr is not None:
            self.noise_adr.maybe_increment(_ep_success_rate)
        if self.success_adr is not None:
            self.success_adr.maybe_increment(_ep_success_rate)
        # 래치 판정: j1-5 자세 진입 비율(posture_rate)이 trigger를 interval 간격으로
        # latch_sustain회 연속 충족하면 래치 → 이후 단조. 래치는 한 번만(전진 재게이팅 없음).
        _posture_rate = (
            (demo_terms["demo_arm_joint_err"] < self.cfg.crossover_posture_threshold)
            .float().mean().item()
        )
        if self.cfg.enable_weight_crossover and not self._pour_latched:
            if step_count - self._last_latch_check >= self.cfg.crossover_increment_interval:
                self._last_latch_check = step_count
                if _posture_rate >= self.cfg.crossover_trigger_rate:
                    self._posture_above_count += 1
                else:
                    self._posture_above_count = 0
                if self._posture_above_count >= self.cfg.crossover_latch_sustain:
                    self._pour_latched = True
                    self._pour_latch_step = step_count

        # ---- TensorBoard logging ----
        # Episode/reward/* — 학습 신호 (가중치 적용된 값)
        reward_log: dict = {
            "reward/hold":             r_hold.mean(),
            "reward/transport":        r_dist_to_target.mean(),
            "reward/palm_pose":        r_palm_pose.mean(),
            "reward/demo_arm_pose":    demo_terms["r_demo_arm_pose"].mean(),
            "reward/pour_dist":        r_pour_dist.mean(),
            "reward/pour_z":           r_pour_z.mean(),
            "reward/pour_tilt":        (pour_warmup * self.cfg.weight_tilt * r_tilt * self._rho).mean(),
            "reward/pour_align":       (pour_warmup * self.cfg.weight_align * r_align * self._rho).mean(),
            "reward/bead_progressive": (self._rho * bead_warmup * r_bead_progressive).mean(),
            "reward/bead_delta":       (self._rho * bead_warmup * r_bead_delta).mean(),
            "reward/source_drain":     r_source_drain.mean(),
            "reward/success":          (self.cfg.weight_success * r_success).mean(),
            "cost/spill":              (spill_weight * spill_cost).mean(),
        }
        self.extras["log"] = reward_log

        # log/* — 진단 지표 (Episode 접두사 없이 log/* 로 기록)
        diag: dict = {
            "log/crossover_alpha":     torch.tensor(self._crossover_alpha, device=self.device),
            "log/posture_rate":        torch.tensor(_posture_rate, device=self.device),
            "log/pour_latched":        torch.tensor(float(self._pour_latched), device=self.device),
            "log/j1":                  arm_joint_pos[:, 0].mean(),
            "log/j2":                  arm_joint_pos[:, 1].mean(),
            "log/j3":                  arm_joint_pos[:, 2].mean(),
            "log/j4":                  arm_joint_pos[:, 3].mean(),
            "log/j5":                  arm_joint_pos[:, 4].mean(),
            "log/j6":                  arm_joint_pos[:, 5].mean(),
            "log/j7":                  arm_joint_pos[:, 6].mean(),
            "log/demo_arm_joint_err":  demo_terms["demo_arm_joint_err"].mean(),
            "log/palm_rot_err":        palm_rot_err.mean(),
            "log/bead_in_target":      self._bead_in_target_fraction.mean(),
            "log/bead_in_source":      self._bead_in_source_fraction.mean(),
            "log/spill_ratio":         self._spill_ratio.mean(),
            "log/cup_center_xy_dist":  self._cup_center_xy_dist.mean(),
            "log/mouth_xy_dist":       self._mouth_xy_distance.mean(),
            "log/mouth_z_clearance":   self._mouth_z_clearance.mean(),
            # pour-point 절대 위치(env-local) — target_opening 대비 어디에 잡고 있는지 직접 관찰
            "log/pour_point_x":        (self._source_pour_point_w[:, 0] - self.scene.env_origins[:, 0]).mean(),
            "log/pour_point_y":        (self._source_pour_point_w[:, 1] - self.scene.env_origins[:, 1]).mean(),
            "log/pour_point_z":        (self._source_pour_point_w[:, 2] - self.scene.env_origins[:, 2]).mean(),
            "log/target_open_x":       (self._target_opening_w[:, 0] - self.scene.env_origins[:, 0]).mean(),
            "log/target_open_y":       (self._target_opening_w[:, 1] - self.scene.env_origins[:, 1]).mean(),
            "log/target_open_z":       (self._target_opening_w[:, 2] - self.scene.env_origins[:, 2]).mean(),
            "log/directional_tilt_cos": self._directional_tilt_cos.mean(),
            "log/rho":                 self._rho.mean(),
            "log/pour_aligned_gate":   pour_aligned_gate.mean(),
            "log/pour_warmup":         torch.tensor(pour_warmup, device=self.device),
            "log/bead_warmup":         torch.tensor(bead_warmup, device=self.device),
        }
        if self.spill_adr is not None:
            diag["log/adr_spill"] = torch.tensor(self.spill_adr.progress, device=self.device)
        if self.noise_adr is not None:
            diag["log/adr_noise"] = torch.tensor(self.noise_adr.progress, device=self.device)
        if self.success_adr is not None:
            diag["log/adr_success"]        = torch.tensor(self.success_adr.progress, device=self.device)
            diag["log/success_fill_ratio"] = torch.tensor(float(success_fill_ratio), device=self.device)
        for k, v in diag.items():
            self.extras[k] = v.mean() if isinstance(v, torch.Tensor) and v.dim() > 0 else v

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

        # 비드 이상 감지: 비드 소환 상태에서 z < -0.5 이면 물리 폭발로 판단 → 즉시 종료
        # _hide_beads()는 z=-10.0에 숨기므로, _beads_spawned=False인 env는 체크 제외
        # (월드 z 사용: z방향 env_origin은 모두 0이므로 절대값 유효)
        bead_pos_w = self.beads.data.object_pos_w  # (N, num_beads, 3)
        bead_fallen = self._beads_spawned & (bead_pos_w[..., 2] < -0.5).any(dim=-1)

        terminated = (
            out_x | out_y | fallen | dropped_by_force
            | self.success_flag | source_drained
            | bead_fallen
        )
        truncated  = self.episode_length_buf >= self.max_episode_length - 1

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
        self._prev_bead_target_local_z[env_ids].fill_(10.0)
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
            print(f"[5g_pour_right_v3] warm-state disk load error: {exc}", flush=True)
            return False

        n = len(bank)
        if n == 0:
            print("[5g_pour_right_v3] warm-state cache is empty on disk.", flush=True)
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
            f"[5g_pour_right_v3] loaded {n} warmstart states from disk "
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
                "[5g_pour_right_v3] warm_state_source='preset': "
                "skipping warmstart cache (using pregrasp reset).",
                flush=True,
            )
            return

        if source == "disk":
            if self._load_warmstart_cache_from_disk():
                return
            print(
                "[5g_pour_right_v3] disk warm-state load failed; "
                "falling back to checkpoint rollout.",
                flush=True,
            )

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
        self._prev_bead_target_local_z[env_ids].fill_(10.0)
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
        self.success_flag[env_ids] = False
        self._prev_arm_joint_vel[env_ids].zero_()
        self._prev_arm_joint_acc[env_ids].zero_()   # [Step 6]
        self._ema_palm_action[env_ids].zero_()       # [Step 7]

        self.actions[env_ids, :6] = 0.0
        self.actions[env_ids, 6:] = 1.0
        self.prev_actions[env_ids, :6] = 0.0
        self.prev_actions[env_ids, 6:] = 1.0
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
            cup_z_local = cup_pose_local[:, 2]
            print(
                "[5g_pour_right_v3][warmstart_reset] "
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
                    "[5g_pour_right_v3][warmstart_reset][WARN] palm pos clamped by "
                    f"{_ws_clamp_delta:.4f}m → palm target may decouple from arm pose. "
                    "Check grasp/pour workspace alignment.",
                    flush=True,
                )
            self._warmstart_reset_debug_printed = True
