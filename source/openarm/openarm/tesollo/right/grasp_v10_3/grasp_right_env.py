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

"""환경 클래스: 5g_grasp_right_v10-3

v10: v9 기반 버그 수정
- Fix 1: rj_dg_1_1 (thumb abduction) = 0.0 고정 (v9: -0.283 → 엄지 치우침 수정)
- Fix 2: MIN_CONTACTS_FOR_SUCCESS = 4, ADR과 분리 (v9: 2접촉 success 오판정 수정)
- Fix 3: has_5_contact = num_contacts>=5 고정 (v9: has_4_contact와 동일 식 버그 수정)

Action (27D):
  [0:3]   normalized palm xyz target around the phase anchor
  [3:7]   palm quaternion target (x, y, z, w)
  [7:27]  20D residual around HAND_GRASP_POSE

Episode (10s @ 60Hz):
  Phase state separates approach, close-grasp anchor, lift, and stabilize.
"""

from __future__ import annotations

import math
import sys
from collections import deque
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
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse

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
    NUM_CRITIC_OBSERVATIONS,
    EPISODE_STEPS,
    CONTACT_FORCE_THRESHOLD,
    CONTACT_FORCE_MAX,
    CUP_RADIUS_APPROX,
    ARM_START_POSE,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
    PALM_POS_ACTION_SLICE,
    PALM_QUAT_ACTION_SLICE,
    FINGER_ACTION_SLICE,
)
from .grasp_right_preset import (
    LEFT_ARM_REST_JOINT_POS,
    HAND_APPROACH_POSE,
    HAND_GRASP_POSE,
    HAND_FULL_GRIP_POSE,
)
from .finger_action_utils import (
    compute_lift_finger_targets,
    compute_preset_residual_finger_targets,
)
from .grasp_reward_utils import (
    compute_tesollo_prelift_lift_readiness,
    compute_upright_success_mask,
)
from .grasp_right_utils import to_torch
from .demo_grasp_reset import DemoGraspResetBank, compute_demo_cup_spawn_local


class GraspRightEnv(DirectRLEnv):
    """OpenArm+Teosllo 오른손 파지 환경 v10.3.

    Action: 27D
      [0:3]  normalized palm xyz target around the phase anchor
      [3:7]  palm quaternion target
      [7:27] 20D residual around HAND_GRASP_POSE

    Phase labels select approach, close-grasp anchor, and lift-anchor palm targets.
    """

    cfg: GraspRightEnvCfg

    @staticmethod
    def _quat_xyzw_from_euler_zyx(euler_zyx: torch.Tensor) -> torch.Tensor:
        z, y, x = euler_zyx[:, 0], euler_zyx[:, 1], euler_zyx[:, 2]
        cz, sz = torch.cos(0.5 * z), torch.sin(0.5 * z)
        cy, sy = torch.cos(0.5 * y), torch.sin(0.5 * y)
        cx, sx = torch.cos(0.5 * x), torch.sin(0.5 * x)
        qw = cz * cy * cx + sz * sy * sx
        qx = cz * cy * sx - sz * sy * cx
        qy = cz * sy * cx + sz * cy * sx
        qz = sz * cy * cx - cz * sy * sx
        return torch.stack([qx, qy, qz, qw], dim=-1)

    def _euler_pose_to_quat_pose(self, palm_euler: torch.Tensor) -> torch.Tensor:
        palm_quat = torch.zeros(palm_euler.shape[0], 7, device=palm_euler.device, dtype=palm_euler.dtype)
        palm_quat[:, :3] = palm_euler[:, :3]
        palm_quat[:, 3:7] = self._quat_xyzw_from_euler_zyx(palm_euler[:, 3:6])
        return palm_quat

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
        # [palm_ee 관측] rl_dg_palm(손바닥 아래쪽) 대신 palm_ee(진짜 손바닥 중심)를 관측/reward 기준으로.
        #   URDF palm_link_to_ee 고정조인트 origin = (0.028, 0, 0.04) 로컬, rpy=0 (orientation 공유).
        #   palm_ee_w = rl_dg_palm_pos_w + R(rl_dg_palm_quat)·offset.
        #   제어 target(palm_pose_targets)은 palm_link 기준 그대로 — 관측/reward만 palm_ee로 이동.
        self._palm_ee_offset_local = torch.tensor(
            [0.028, 0.0, 0.04], device=self.device
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

        # pregrasp palm pose 버퍼
        self.pregrasp_palm_pose_buf = torch.zeros(self.num_envs, 6, device=self.device)
        self.grasp_anchor_palm_pose_buf = torch.zeros(self.num_envs, 7, device=self.device)
        self.lift_palm_start_pose_buf = torch.zeros(self.num_envs, 7, device=self.device)
        self.demo_lift_palm_target_buf = torch.zeros(self.num_envs, 6, device=self.device)
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
        self.shape_anchor_mask[[0, 4, 8, 12, 16, 17]] = 1.0
        self.hand_residual_mask = torch.ones(NUM_HAND_DOF, device=self.device)
        self.hand_residual_mask[[0, 4, 8, 12, 16, 17]] = 0.0

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
        # Pregrasp / phase-gate 버퍼
        # ----------------------------------------------------------------
        self.pregrasp_arm_pos_buf = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        self.lift_finger_pos_buf = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        self.is_approach_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_close_grasp_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_lift_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.grasp_ready_hold_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.lift_ready_latched_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contacts_at_lift_start_buf = torch.zeros(self.num_envs, device=self.device)
        self.palm_at_lift_start_buf = torch.zeros(self.num_envs, device=self.device)
        self.grasp_tilt_at_lift_start_buf = torch.zeros(self.num_envs, device=self.device)
        self.lift_started_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lift_success_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._lift_success_latched_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._eval_mass_shift_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_started_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_anchor_set_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_stabilize_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_stabilize_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ----------------------------------------------------------------
        # Hand joint targets (per-joint delta 결과)
        # ----------------------------------------------------------------
        self.approach_hand_pose_buf = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        self.hand_joint_targets = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)

        # ----------------------------------------------------------------
        # 접촉 상태 버퍼
        # ----------------------------------------------------------------
        self.contact_force_xyz_raw   = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.contact_force_raw       = torch.zeros(self.num_envs, NUM_FINGERTIPS, device=self.device)
        self.contact_friction_xyz_raw = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.binary_contact_buf    = torch.zeros(self.num_envs, NUM_FINGERTIPS, dtype=torch.bool, device=self.device)
        self.num_contacts_buf      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.distal_contact_force_raw  = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, device=self.device)
        self.distal_binary_contact_buf = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, dtype=torch.bool, device=self.device)

        self.middle_contact_force_raw  = torch.zeros(self.num_envs, NUM_MIDDLE_SENSORS, device=self.device)
        self.middle_binary_contact_buf = torch.zeros(self.num_envs, NUM_MIDDLE_SENSORS, dtype=torch.bool, device=self.device)
        self.palm_contact_force_raw = torch.zeros(self.num_envs, device=self.device)
        self.palm_binary_contact_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_total_grip_force_buf = torch.zeros(self.num_envs, device=self.device)
        self._prev_num_contacts_buf = torch.zeros(self.num_envs, device=self.device)
        self._prev_middle_contacts_buf = torch.zeros(self.num_envs, device=self.device)

        # ----------------------------------------------------------------
        # Force-smooth 버퍼 (이전 스텝 총 파지력)
        # ----------------------------------------------------------------
        self._prev_avg_force_buf = torch.zeros(self.num_envs, device=self.device)
        self._force_smooth_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ----------------------------------------------------------------
        # 기타 버퍼
        # ----------------------------------------------------------------
        self._approach_dir_buf = torch.zeros(self.num_envs, 3, device=self.device)
        self.success_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cup_tipping_cos = math.cos(math.radians(cfg.cup_tipping_max_deg))
        self.episode_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_lift_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._total_episodes: int = 0
        self._successful_episodes: int = 0

        # ----------------------------------------------------------------
        # Eval 로깅
        # ----------------------------------------------------------------
        self._eval_episode_started        = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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

        # 6.2: moving-window ADR trigger (최근 N 에피소드 성공률)
        _win = cfg.adr_window_size if cfg.adr_window_size > 0 else 500
        self._success_window: deque = deque(maxlen=_win)

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
        self.palm_pose_targets = torch.zeros(self.num_envs, 7, device=self.device)
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

        per_distal = self._distal_sensor.data.net_forces_w.norm(dim=-1)
        per_distal = torch.nan_to_num(per_distal, nan=0.0, posinf=0.0, neginf=0.0)
        self.distal_contact_force_raw.copy_(per_distal)
        self.distal_binary_contact_buf.copy_(per_distal > CONTACT_FORCE_THRESHOLD)

        per_middle = self._middle_sensor.data.net_forces_w.norm(dim=-1)
        per_middle = torch.nan_to_num(per_middle, nan=0.0, posinf=0.0, neginf=0.0)
        self.middle_contact_force_raw.copy_(per_middle)
        self.middle_binary_contact_buf.copy_(per_middle > CONTACT_FORCE_THRESHOLD)

        palm_xyz = torch.nan_to_num(
            self._palm_sensor.data.net_forces_w[:, 0, :],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        palm_force_local = quat_apply_inverse(
            self.robot.data.body_quat_w[:, self.palm_body_index],
            palm_xyz,
        )
        # Palm +X is the outward face normal; contact on that face pushes the palm along local -X.
        palm_force = torch.relu(-palm_force_local[:, 0])
        self.palm_contact_force_raw.copy_(palm_force)
        self.palm_binary_contact_buf.copy_(palm_force > CONTACT_FORCE_THRESHOLD)

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self._eval_episode_started[:] = True

        palm_pos_action = self.actions[:, PALM_POS_ACTION_SLICE]
        palm_quat_action = self.actions[:, PALM_QUAT_ACTION_SLICE]
        finger_action = self.actions[:, FINGER_ACTION_SLICE]

        cup_quat = self.object_rot
        identity_quat = torch.zeros_like(cup_quat)
        identity_quat[:, 0] = 1.0
        valid_cup_quat = cup_quat.norm(dim=-1, keepdim=True) > 1e-6
        cup_quat = torch.where(valid_cup_quat, cup_quat, identity_quat)
        cup_quat = torch.nn.functional.normalize(cup_quat, dim=-1, eps=1e-6)
        cup_quat_inv = cup_quat.clone()
        cup_quat_inv[:, 1:] = -cup_quat_inv[:, 1:]
        palm_local = quat_apply(cup_quat_inv, self.palm_center_pos - self.object_pos)
        approach_ready = (
            (self.episode_length_buf >= int(self.cfg.approach_min_steps))
            & (palm_local[:, 2] >= float(self.cfg.approach_palm_local_z_min))
            & (palm_local[:, 2] <= float(self.cfg.approach_palm_local_z_max))
            & (~self.lift_ready_latched_buf)
        )
        just_entering_close_grasp = (
            (~self._grasp_started_buf)
            & (~self.lift_ready_latched_buf)
            & approach_ready
        )
        if just_entering_close_grasp.any():
            self.grasp_anchor_palm_pose_buf[just_entering_close_grasp] = (
                self.palm_pose_targets[just_entering_close_grasp]
            )
            self._grasp_anchor_set_buf[just_entering_close_grasp] = True
        self._grasp_started_buf |= just_entering_close_grasp

        just_entering_lift = self.lift_ready_latched_buf & (~self.lift_started_buf)
        if just_entering_lift.any():
            self.lift_palm_start_pose_buf[just_entering_lift] = (
                self.palm_pose_targets[just_entering_lift]
            )
            self.lift_finger_pos_buf[just_entering_lift] = (
                self.robot.data.joint_pos[just_entering_lift][:, self.hand_dof_indices]
            )

        close_grasp_mask = (
            (~self.lift_ready_latched_buf)
            & self._grasp_started_buf
            & self._grasp_anchor_set_buf
        )
        approach_mask = (~self.lift_ready_latched_buf) & (~close_grasp_mask)
        self.is_stabilize_phase.copy_(self._lift_success_latched_buf)
        lift_mask = self.lift_ready_latched_buf & (~self.is_stabilize_phase)
        self.is_approach_phase.copy_(approach_mask)
        self.is_close_grasp_phase.copy_(close_grasp_mask)
        self.is_lift_phase.copy_(lift_mask)

        approach_pos_raw = (
            self.pregrasp_palm_pose_buf[:, :3]
            + palm_pos_action * float(self.cfg.palm_local_workspace_radius)
        )
        approach_pos_raw = torch.max(
            torch.min(approach_pos_raw, self.palm_maxs[:3].unsqueeze(0)),
            self.palm_mins[:3].unsqueeze(0),
        )
        approach_pos_delta = (approach_pos_raw - self.palm_pose_targets[:, :3]).clamp(
            min=-float(self.cfg.palm_target_max_delta),
            max=float(self.cfg.palm_target_max_delta),
        )
        approach_palm_pose = self.palm_pose_targets.clone()
        approach_palm_pose[:, :3] = self.palm_pose_targets[:, :3] + approach_pos_delta
        # B: approach orientation = 고정 pregrasp 자세(컵 향함) 주변 bounded euler residual.
        # grasp-v1 방식 — 정책 절대 quaternion 대신 pregrasp euler ± approach_palm_residual_rot_deg.
        # palm_quat_action 4D 중 앞 3D를 (ez,ey,ex) residual로 사용(나머지 1D inert).
        _palm_residual_rad = math.radians(float(self.cfg.approach_palm_residual_rot_deg))
        approach_euler = (
            self.pregrasp_palm_pose_buf[:, 3:6]
            + palm_quat_action[:, :3] * _palm_residual_rad
        )
        approach_palm_pose[:, 3:7] = self._quat_xyzw_from_euler_zyx(approach_euler)

        grasp_palm_pose = self.grasp_anchor_palm_pose_buf.clone()
        grasp_pos_raw = (
            self.grasp_anchor_palm_pose_buf[:, :3]
            + palm_pos_action
            * float(self.cfg.palm_local_workspace_radius)
            * float(self.cfg.grasp_palm_delta_scale)
        )
        grasp_pos_raw = torch.max(
            torch.min(grasp_pos_raw, self.palm_maxs[:3].unsqueeze(0)),
            self.palm_mins[:3].unsqueeze(0),
        )
        grasp_palm_pose[:, :3] = grasp_pos_raw
        grasp_palm_pose[:, 2] = torch.minimum(
            grasp_palm_pose[:, 2],
            self.grasp_anchor_palm_pose_buf[:, 2]
            + float(self.cfg.prelift_max_cup_height_delta),
        )
        grasp_pos_delta = (grasp_palm_pose[:, :3] - self.palm_pose_targets[:, :3]).clamp(
            min=-float(self.cfg.palm_target_max_delta),
            max=float(self.cfg.palm_target_max_delta),
        )
        grasp_palm_pose[:, :3] = self.palm_pose_targets[:, :3] + grasp_pos_delta
        grasp_palm_pose[:, 3:7] = torch.nn.functional.normalize(
            self.grasp_anchor_palm_pose_buf[:, 3:7],
            dim=-1,
            eps=1e-6,
        )

        # Phase B: stabilize에서 palm freeze — palm action을 무시해 latch된 위치에 고정.
        # 손가락 residual은 그대로 두어 파지 강화(컵 더 잡기)는 허용한다.
        # orientation은 lift_palm_start_pose에서 복사되어 이미 고정이므로 position만 freeze.
        _palm_pos_action_eff = torch.where(
            self.is_stabilize_phase.unsqueeze(1),
            torch.zeros_like(palm_pos_action),
            palm_pos_action,
        )
        lift_palm_pose = self.lift_palm_start_pose_buf.clone()
        lift_pos_raw = (
            self.lift_palm_start_pose_buf[:, :3]
            + _palm_pos_action_eff * float(self.cfg.lift_palm_delta_xyz)
        )
        lift_pos_raw = torch.max(
            torch.min(lift_pos_raw, self.palm_maxs[:3].unsqueeze(0)),
            self.palm_mins[:3].unsqueeze(0),
        )
        lift_pos_delta = (lift_pos_raw - self.palm_pose_targets[:, :3]).clamp(
            min=-float(self.cfg.palm_target_max_delta),
            max=float(self.cfg.palm_target_max_delta),
        )
        lift_palm_pose[:, :3] = self.palm_pose_targets[:, :3] + lift_pos_delta
        lift_palm_pose[:, 3:7] = torch.nn.functional.normalize(
            self.lift_palm_start_pose_buf[:, 3:7], dim=-1, eps=1e-6
        )

        if self.cfg.eval_mass_shift_enabled:
            shift_mask = self.lift_ready_latched_buf & ~self._eval_mass_shift_done
            if shift_mask.any():
                target_count = min(
                    max(int(self.cfg.eval_mass_shift_target_bead_count), 0),
                    int(self.cfg.num_beads),
                )
                self._bead_mass_normalized[shift_mask] = (
                    float(target_count) / max(float(self.cfg.num_beads), 1.0)
                )
                self._eval_mass_shift_done[shift_mask] = True

        palm_pose = torch.where(
            self.is_stabilize_phase.unsqueeze(1),
            lift_palm_pose,
            torch.where(
                lift_mask.unsqueeze(1),
                lift_palm_pose,
                torch.where(
                    close_grasp_mask.unsqueeze(1),
                    grasp_palm_pose,
                    approach_palm_pose,
                ),
            ),
        )
        self.palm_pose_targets.copy_(palm_pose)
        self.hand_pca_targets.zero_()

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

        # close-grasp: envelope grip 유도 — preset을 full_grip(말아쥔 자세)로, residual 확대.
        # 관절 한계가 [approach, full_grip]로 클램프되어 과침투 없이 cup 접촉에서 멈춤.
        close_hand_target = compute_preset_residual_finger_targets(
            preset_pos=self.hand_full_grip_pose,
            finger_action=finger_action,
            lower_limits=self.hand_joint_lower_limits,
            upper_limits=self.hand_joint_upper_limits,
            residual_scale=self.cfg.hand_close_residual_scale,
            residual_mask=self.hand_residual_mask,
        )
        approach_hand_target = self.approach_hand_pose_buf
        lift_hand_target = compute_lift_finger_targets(
            lift_reference_pos=self.lift_finger_pos_buf,
            finger_action=finger_action,
            lower_limits=self.hand_joint_lower_limits,
            upper_limits=self.hand_joint_upper_limits,
            delta_scale=self.cfg.hand_residual_scale,
            delta_mask=self.hand_residual_mask,
            thumb_curl_index=self.thumb_curl_index,
            thumb_downward_action_scale=self.cfg.thumb_curl_downward_action_scale,
            thumb_anchor_pose=self.hand_grasp_pose,
            thumb_curl_max_downward_delta=self.cfg.thumb_curl_max_downward_delta,
        )
        hand_target = torch.where(
            (lift_mask | self.is_stabilize_phase).unsqueeze(1),
            lift_hand_target,
            torch.where(
                approach_mask.unsqueeze(1),
                approach_hand_target,
                close_hand_target,
            ),
        )
        self.hand_joint_targets.copy_(hand_target)

        self.fabric_q[:, NUM_ARM_DOF:] = hand_target
        self.fabric_qd[:, NUM_ARM_DOF:].zero_()

    def _apply_action(self) -> None:
        # ---- 오른팔 ----
        self.robot.set_joint_position_target(self.fabric_q[:, :NUM_ARM_DOF], joint_ids=self.arm_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(self.fabric_q[:, :NUM_ARM_DOF]), joint_ids=self.arm_dof_indices
        )

        # ---- 오른손 ----
        self.robot.set_joint_position_target(self.hand_joint_targets, joint_ids=self.hand_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(self.hand_joint_targets), joint_ids=self.hand_dof_indices
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
            # palm_ee = rl_dg_palm 위치 + R(rl_dg_palm_quat)·offset (진짜 손바닥 중심)
            _palm_quat_w = self.robot.data.body_quat_w[:, self.palm_body_index]
            _palm_ee_w = self.robot.data.body_pos_w[:, self.palm_body_index, :] + quat_apply(
                _palm_quat_w,
                self._palm_ee_offset_local.unsqueeze(0).expand(self.num_envs, -1),
            )
            self.palm_center_pos = _palm_ee_w - env_origins

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
    # Observations: Actor 132D no-mass (133D optional mass/debug) | Critic 169D
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

        last_actions = self.actions  # (N, 27)

        # tip force: 3D 법선 방향 벡터 (5 × 3D = 15D)
        tip_force_xyz_norm = (
            self.contact_force_xyz_raw / CONTACT_FORCE_MAX
        ).clamp(-1.0, 1.0).view(self.num_envs, -1)  # (N, 15)

        phase_step_ratio = (
            self.episode_length_buf.float() / EPISODE_STEPS
        ).unsqueeze(1)

        actor_obs_parts = [
            arm_joint_pos,          # 7
            arm_joint_vel,          # 7
            finger_joint_pos,       # 20
            finger_joint_vel,       # 20
            palm_center_pos,        # 3
            fingertip_pos_rel_palm, # 15
            palm_to_cup,            # 3
            last_actions,           # 27
        ]
        if self.cfg.actor_observe_bead_mass:
            actor_obs_parts.append(self._bead_mass_normalized.unsqueeze(-1))  # 1
        actor_obs_parts.extend([
            tip_force_xyz_norm,     # 15
            middle_to_cup,          # 15
            phase_step_ratio,       # 1
        ])
        actor_obs = torch.cat(actor_obs_parts, dim=-1)

        actor_obs = torch.nan_to_num(actor_obs, nan=0.0, posinf=5.0, neginf=-5.0)

        if actor_obs.shape[1] != self.cfg.num_observations:
            raise RuntimeError(
                f"[v10-3] Actor obs dim mismatch: {actor_obs.shape[1]} != {self.cfg.num_observations}"
            )

        # ==== Critic extra obs (37D, including oracle mass) ====
        cup_lin_vel  = self.cup.data.root_lin_vel_w
        cup_ang_vel  = self.cup.data.root_ang_vel_w
        cup_rot      = self.object_rot
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

        actor_obs_clean = torch.cat([
            arm_joint_pos_clean,
            arm_joint_vel_clean,
            finger_joint_pos_clean,
            finger_joint_vel_clean,
            palm_center_pos_clean,
            (fingertip_pos_clean - palm_center_pos_clean.unsqueeze(1)).view(self.num_envs, -1),
            cup_pos_clean - palm_center_pos_clean,
            last_actions,
            tip_force_xyz_norm,     # 15D (critic도 동일 변환)
            middle_to_cup_clean,    # 15D
            phase_step_ratio,
        ], dim=-1)   # 133D

        critic_obs = torch.cat([
            actor_obs_clean,        # 133
            self._bead_mass_normalized.unsqueeze(-1),  # 1
            cup_lin_vel,            # 3
            cup_ang_vel,            # 3
            cup_rot,                # 4
            cup_height_delta,       # 1
            distal_binary,          # 5
            distal_force_norm,      # 5
            middle_binary,          # 5
            middle_force_norm,      # 5
            fingertip_signed_dist,  # 5
        ], dim=-1)   # 170D

        critic_obs = torch.nan_to_num(critic_obs, nan=0.0, posinf=5.0, neginf=-5.0)

        if critic_obs.shape[1] != NUM_CRITIC_OBSERVATIONS:
            raise RuntimeError(
                f"[v9] Critic obs dim mismatch: {critic_obs.shape[1]} != {NUM_CRITIC_OBSERVATIONS}"
            )

        return {"policy": actor_obs, "critic": critic_obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        cup_height_delta = (self.object_pos[:, 2] - self.object_init_pos[:, 2]).clamp(min=0.0)
        cup_xy_displacement = (self.object_pos[:, :2] - self.object_init_pos[:, :2]).norm(dim=-1)

        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cup_z_world = quat_apply(self.object_rot, z_local)
        cup_tilt_deg = torch.rad2deg(
            torch.acos(cup_z_world[:, 2].clamp(min=-1.0, max=1.0))
        )
        upright_quality = torch.exp(
            -cup_tilt_deg / max(float(self.cfg.stabilize_upright_reward_scale_deg), 1e-6)
        )

        grasp_center = self.object_pos
        palm_to_cup_dist = (self.palm_center_pos - grasp_center).norm(dim=-1)
        cup_to_palm_xy = self.palm_center_pos[:, :2] - grasp_center[:, :2]
        approach_dir_xy = cup_to_palm_xy / cup_to_palm_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        side_dir_xy = torch.stack([-approach_dir_xy[:, 1], approach_dir_xy[:, 0]], dim=1)
        self._approach_dir_buf[:, :2] = side_dir_xy
        self._approach_dir_buf[:, 2] = 0.0

        side_radius = self.cfg.cup_radius_approx
        thumb_target = grasp_center + self._approach_dir_buf * side_radius
        others_target = grasp_center - self._approach_dir_buf * side_radius
        thumb_dist = (self.fingertip_pos[:, 0, :] - thumb_target).norm(dim=-1)
        others_dist = (self.fingertip_pos[:, 1:, :] - others_target.unsqueeze(1)).norm(dim=-1).mean(dim=-1)
        fingertip_side_dist = self.cfg.enclosure_thumb_weight * thumb_dist + (
            1.0 - self.cfg.enclosure_thumb_weight
        ) * others_dist

        num_tip_contacts = self.num_contacts_buf
        tip_contact_frac = num_tip_contacts.float() / float(NUM_FINGERTIPS)
        full_tip_contact_bool = num_tip_contacts >= NUM_FINGERTIPS
        full_tip_contact = full_tip_contact_bool.float()
        palm_contact_bool = self.palm_binary_contact_buf
        palm_contact = palm_contact_bool.float()

        cup_quat = self.object_rot
        identity_quat = torch.zeros_like(cup_quat)
        identity_quat[:, 0] = 1.0
        valid_cup_quat = cup_quat.norm(dim=-1, keepdim=True) > 1e-6
        cup_quat = torch.where(valid_cup_quat, cup_quat, identity_quat)
        cup_quat = torch.nn.functional.normalize(cup_quat, dim=-1, eps=1e-6)
        cup_quat_inv = cup_quat.clone()
        cup_quat_inv[:, 1:] = -cup_quat_inv[:, 1:]
        tip_local = quat_apply(
            cup_quat_inv.unsqueeze(1).expand(-1, NUM_FINGERTIPS, -1).reshape(-1, 4),
            (self.fingertip_pos - self.object_pos.unsqueeze(1)).reshape(-1, 3),
        ).view(self.num_envs, NUM_FINGERTIPS, 3)
        tip_local_z_mean = tip_local[:, :, 2].mean(dim=-1)
        palm_local = quat_apply(
            cup_quat_inv,
            self.palm_center_pos - self.object_pos,
        )
        palm_local_z = palm_local[:, 2]
        prelift_cup_lin_vel = torch.nan_to_num(
            self.cup.data.root_lin_vel_w.norm(dim=-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        # ---- Phase A: contact_adr 동적 min_contacts (3→4→5 lift 진입 허들) ----
        _adr_min_contacts = (
            int(round(self.contact_adr.get_param("contact", "min_contacts")))
            if self.contact_adr is not None
            else int(self.cfg.stage0_lift_start_min_contacts)
        )

        close_grasp_mask = self.is_close_grasp_phase & (~self.lift_ready_latched_buf)
        (
            self.grasp_ready_hold_buf,
            grasp_ready_now,
            lift_ready_latched,
            prelift_readiness_gates,
        ) = compute_tesollo_prelift_lift_readiness(
            num_contacts=num_tip_contacts,
            is_close_grasp_phase=close_grasp_mask,
            tip_local_z_mean=tip_local_z_mean,
            cup_height_delta=cup_height_delta,
            cup_lin_vel_norm=prelift_cup_lin_vel,
            previous_hold_count=self.grasp_ready_hold_buf,
            previous_latched=self.lift_ready_latched_buf,
            min_contacts=_adr_min_contacts,
            hold_steps=self.cfg.grasp_ready_hold_steps,
            body_local_z_min=self.cfg.grasp_body_local_z_min,
            body_local_z_max=self.cfg.grasp_body_local_z_max,
            max_cup_height_delta=self.cfg.prelift_max_cup_height_delta,
            cup_lin_vel_threshold=self.cfg.prelift_cup_lin_vel_threshold,
        )
        just_latched = (
            ~self.lift_ready_latched_buf
            & lift_ready_latched
        )
        if just_latched.any():
            self.contacts_at_lift_start_buf[just_latched] = num_tip_contacts[just_latched].float()
            self.palm_at_lift_start_buf[just_latched] = palm_contact[just_latched]
            self.grasp_tilt_at_lift_start_buf[just_latched] = cup_tilt_deg[just_latched]
            self.lift_palm_start_pose_buf[just_latched] = self.palm_pose_targets[just_latched]
            self.lift_finger_pos_buf[just_latched] = (
                self.robot.data.joint_pos[just_latched][:, self.hand_dof_indices]
            )
        self.lift_ready_latched_buf.copy_(lift_ready_latched)
        self.lift_started_buf |= self.lift_ready_latched_buf

        lifted_gate = (cup_height_delta >= self.cfg.lift_success_height).float()
        meaningful_contact = num_tip_contacts > 0
        self.is_stabilize_phase.copy_(self._lift_success_latched_buf)
        action_delta_norm = compute_action_delta_norm(self.actions, self.prev_actions)
        contact_delta_abs = (num_tip_contacts.float() - self._prev_num_contacts_buf).abs()
        stability = compute_grasp_v2_stability(
            cup_lin_vel=self.cup.data.root_lin_vel_w,
            cup_ang_vel=self.cup.data.root_ang_vel_w,
            contact_delta=contact_delta_abs,
            action_delta_norm=action_delta_norm,
            cfg=self.cfg,
        )
        contact_persistence_frac = (
            self.grasp_ready_hold_buf.float()
            / max(float(self.cfg.grasp_contact_persistence_reward_steps), 1.0)
        ).clamp(max=1.0)
        total, reward_terms, reward_gates = compute_grasp_reward_terms(
            num_tip_contacts=num_tip_contacts,
            tip_contact_frac=tip_contact_frac,
            full_tip_contact=full_tip_contact,
            contact_persistence_frac=contact_persistence_frac,
            palm_to_cup_dist=palm_to_cup_dist,
            fingertip_side_dist=fingertip_side_dist,
            cup_height_delta=cup_height_delta,
            cup_xy_displacement=cup_xy_displacement,
            stabilize_reward_gate=self.is_stabilize_phase,
            success_now=self.success_flag,
            cup_tilt_deg=cup_tilt_deg,
            upright_quality=upright_quality,
            lift_latched=self.lift_ready_latched_buf,
            action_delta_norm=action_delta_norm,
            stable=stability.stable,
            stability_quality=stability.quality,
            cfg=self.cfg,
        )
        prelift_height_excess = torch.relu(
            cup_height_delta - float(self.cfg.prelift_max_cup_height_delta)
        ) / max(float(self.cfg.prelift_max_cup_height_delta), 1e-6)
        prelift_velocity_excess = torch.relu(
            prelift_cup_lin_vel - float(self.cfg.prelift_cup_lin_vel_threshold)
        ) / max(float(self.cfg.prelift_cup_lin_vel_threshold), 1e-6)
        prelift_rim_lift_penalty = (
            -float(self.cfg.prelift_rim_lift_penalty_weight)
            * (~self.lift_ready_latched_buf).float()
            * (prelift_height_excess + prelift_velocity_excess)
        )
        total = torch.nan_to_num(
            total + prelift_rim_lift_penalty,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        r_action_delta = reward_terms["action_smooth"]

        if len(self._success_window) >= 10:
            _ep_success_rate = sum(self._success_window) / len(self._success_window)
        else:
            _ep_success_rate = self._successful_episodes / max(self._total_episodes, 1)

        # Phase A: contact ADR을 lift_started_rate로 트리거.
        # TESOLLO는 success_rate=0이라 success 기반 트리거로는 영원히 허들이 안 오른다.
        if self.contact_adr is not None:
            self.contact_adr.maybe_increment(self.lift_started_buf.float().mean())

        lift_tilt = cup_tilt_deg[self.lift_ready_latched_buf]
        prelift_mask = (~self.lift_ready_latched_buf).float()
        lift_mask = self.lift_ready_latched_buf.float()
        success_held = self._success_hold_count >= int(self.cfg.success_hold_steps)
        log_grasp_v2_common_scalars(
            self.extras,
            {
                "phase/approach": ((~self.lift_ready_latched_buf) & (~meaningful_contact)).float().mean(),
                "phase/grasp": ((~self.lift_ready_latched_buf) & meaningful_contact).float().mean(),
                "phase/lift": self.is_lift_phase.float().mean(),
                "phase/stabilize": self.is_stabilize_phase.float().mean(),
                "reward/approach": reward_terms["approach"].mean(),
                "reward/grasp": reward_terms["grasp"].mean(),
                "reward/lift": reward_terms["lift"].mean(),
                "reward/stabilize": reward_terms["stabilize"].mean(),
                "reward/success_bonus": reward_terms["success_bonus"].mean(),
                "reward/post_lift_contact_loss": reward_terms["post_lift_contact_loss"].mean(),
                "reward/action_smooth": r_action_delta.mean(),
                "reward/stability": reward_terms["stability"].mean(),
                "reward/total": total.mean(),
                "contact/count": num_tip_contacts.float().mean(),
                "contact/full_contact_rate": full_tip_contact.mean(),
                "contact/palm": palm_contact.mean(),
                "contact/palm_force": self.palm_contact_force_raw.mean(),
                "contact/grasp_ready_hold": self.grasp_ready_hold_buf.float().mean(),
                "contact/contacts_at_lift_start": self.contacts_at_lift_start_buf.mean(),
                "contact/palm_at_lift_start": self.palm_at_lift_start_buf.mean(),
                "cup/height_delta": cup_height_delta.mean(),
                "cup/tilt_deg": cup_tilt_deg.mean(),
                "cup/upright_quality": upright_quality.mean(),
                "cup/grasp_tilt_deg": self.grasp_tilt_at_lift_start_buf.mean(),
                "cup/lift_tilt_deg": lift_tilt.mean() if lift_tilt.numel() > 0 else cup_tilt_deg.mean(),
                "cup/xy_displacement": cup_xy_displacement.mean(),
                "task/lifted_rate": lifted_gate.mean(),
                "task/upright_rate": reward_gates["final_upright_success"].mean(),
                "task/stable_rate": stability.stable.float().mean(),
                "task/cup_lin_vel": stability.cup_lin_vel_norm.mean(),
                "task/cup_ang_vel": stability.cup_ang_vel_norm.mean(),
                "task/action_delta_norm": stability.action_delta_norm.mean(),
                "task/contact_delta": stability.contact_delta.mean(),
                "task/grasp_ready_rate": grasp_ready_now.float().mean(),
                "task/five_tip_contact_rate": full_tip_contact.mean(),
                "task/prelift_five_tip_contact_rate": (
                    full_tip_contact * prelift_mask
                ).sum() / prelift_mask.sum().clamp(min=1.0),
                "task/lift_five_tip_contact_rate": (
                    full_tip_contact * lift_mask
                ).sum() / lift_mask.sum().clamp(min=1.0),
                "task/lift_started_rate": self.lift_started_buf.float().mean(),
                "task/lift_success_now": self._lift_success_latched_buf.float().mean(),
                "task/stabilize_success_now": self.episode_stabilize_success_buf.float().mean(),
                "task/lift_success_rate": self._lift_success_latched_buf.float().mean(),
                "task/stabilize_success_rate": self.episode_stabilize_success_buf.float().mean(),
                "task/success_rate": torch.tensor(_ep_success_rate, device=self.device),
                "task/common_success_now": reward_gates["success_now"].mean(),
                "task/success_held_rate": success_held.float().mean(),
                "task/success_hold_count": self._success_hold_count.float().mean(),
            },
        )
        self.extras["debug/tesollo/task/palm_local_z"] = (
            palm_local_z * prelift_mask
        ).sum() / prelift_mask.sum().clamp(min=1.0)
        self.extras["debug/tesollo/task/tip_local_z_mean"] = (
            tip_local_z_mean * prelift_mask
        ).sum() / prelift_mask.sum().clamp(min=1.0)
        self.extras["debug/tesollo/task/rim_contact_proxy"] = (
            prelift_readiness_gates["rim_contact_proxy"] * prelift_mask
        ).sum() / prelift_mask.sum().clamp(min=1.0)
        self.extras["debug/tesollo/task/prelift_cup_height_delta"] = (
            cup_height_delta * prelift_mask
        ).sum() / prelift_mask.sum().clamp(min=1.0)
        self.extras["debug/tesollo/task/prelift_cup_lin_vel"] = (
            prelift_cup_lin_vel * prelift_mask
        ).sum() / prelift_mask.sum().clamp(min=1.0)
        self.extras["debug/tesollo/task/prelift_body_band_rate"] = (
            prelift_readiness_gates["body_band"] * prelift_mask
        ).sum() / prelift_mask.sum().clamp(min=1.0)
        self.extras["debug/tesollo/task/prelift_rim_lift_penalty"] = (
            prelift_rim_lift_penalty * prelift_mask
        ).sum() / prelift_mask.sum().clamp(min=1.0)
        self.extras["debug/tesollo/control/raw_palm_action_norm"] = (
            self.actions[:, :7].norm(dim=-1).mean()
        )
        # palm_pose_targets는 palm_link 기준 제어 target이므로 palm_link 실제 위치와 비교.
        # (palm_center_pos는 이제 palm_ee라 추종 오차 진단에 쓰면 offset만큼 왜곡됨)
        _palm_link_pos = (
            self.robot.data.body_pos_w[:, self.palm_body_index, :] - self.scene.env_origins
        )
        self.extras["debug/tesollo/control/palm_target_position_error"] = (
            self.palm_pose_targets[:, :3] - _palm_link_pos
        ).norm(dim=-1).mean()
        self.extras["object_stat/obj_z"] = self.object_pos[:, 2].mean()
        # Phase A: current contact_adr curriculum hurdle (3→4→5)
        self.extras["task/adr_min_contacts"] = torch.tensor(
            float(_adr_min_contacts), device=self.device
        )

        # ---- Phase C: per-finger contact + joint-state diagnostics ----
        self.extras.update(
            per_finger_contact_scalars(
                tip_force=self.contact_force_raw,
                tip_binary=self.binary_contact_buf,
                distal_force=self.distal_contact_force_raw,
                middle_force=self.middle_contact_force_raw,
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

        self._prev_total_grip_force_buf.copy_(self.contact_force_raw.sum(dim=-1))
        self._prev_num_contacts_buf.copy_(num_tip_contacts.float())
        self._prev_middle_contacts_buf.copy_(self.middle_binary_contact_buf.float().sum(dim=-1))
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

        in_or_past_lift = self.lift_ready_latched_buf
        lifted  = self.object_pos[:, 2] > (self.object_init_pos[:, 2] + self.cfg.lift_success_height)
        full_tip_contact = self.num_contacts_buf >= NUM_FINGERTIPS
        upright_success = compute_upright_success_mask(
            cup_z_world[:, 2],
            self.cfg.success_upright_max_deg,
        )
        lift_success_candidate = in_or_past_lift & lifted & full_tip_contact & upright_success
        self._lift_success_hold_count = torch.where(
            lift_success_candidate,
            self._lift_success_hold_count + 1,
            torch.zeros_like(self._lift_success_hold_count),
        )
        lift_success_now = self._lift_success_hold_count >= int(self.cfg.full_grip_hold_steps)
        self._lift_success_latched_buf |= lift_success_now
        self.episode_lift_success_buf |= lift_success_now

        action_delta_norm = compute_action_delta_norm(self.actions, self.prev_actions)
        contact_delta_abs = (self.num_contacts_buf.float() - self._prev_num_contacts_buf).abs()
        stability = compute_grasp_v2_stability(
            cup_lin_vel=self.cup.data.root_lin_vel_w,
            cup_ang_vel=self.cup.data.root_ang_vel_w,
            contact_delta=contact_delta_abs,
            action_delta_norm=action_delta_norm,
            cfg=self.cfg,
        )
        stabilize_success_now = (
            self.is_stabilize_phase
            & lifted
            & full_tip_contact
            & upright_success
            & stability.stable
        )
        self.episode_stabilize_success_buf |= stabilize_success_now
        stationary_success = compute_stationary_grasp_success(
            stabilize_started=self.is_stabilize_phase,
            cup_height_delta=self.object_pos[:, 2] - self.object_init_pos[:, 2],
            full_contact=full_tip_contact,
            cup_tilt_deg=torch.rad2deg(
                torch.acos(cup_z_world[:, 2].clamp(min=-1.0, max=1.0))
            ),
            stable=stability.stable,
            previous_success_hold_count=self._success_hold_count,
            cfg=self.cfg,
        )
        stabilize_success_now = stationary_success.success_now
        self.success_flag.copy_(stabilize_success_now)
        self.episode_success_buf |= stabilize_success_now
        self.episode_stabilize_success_buf |= stabilize_success_now
        self._success_hold_count = stationary_success.hold_count

        terminated = out_x | out_y | fallen | tipped | stationary_success.success_held
        truncated  = self.episode_length_buf >= self.max_episode_length - 1

        self.extras["object_stat/obj_z"] = self.object_pos[:, 2].mean()
        self.extras["task/lift_success_now"] = lift_success_now.float().mean()
        self.extras["task/stabilize_success_now"] = stabilize_success_now.float().mean()
        self.extras["task/stabilize_success_rate"] = self.episode_stabilize_success_buf.float().mean()

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
            success_val = int(bool(self.episode_success_buf[env_id].item()))
            # 6.2: deque에 추가 (maxlen으로 자동 oldest 제거)
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

            bead_count = int(round(self._bead_mass_normalized[env_id].item() * self.cfg.num_beads))

            self._eval_records.append({
                "bead_count": bead_count,
                "bead_mass": self._bead_mass_normalized[env_id].item(),
                "contacts_at_lift_start": self.contacts_at_lift_start_buf[env_id].item(),
                "palm_at_lift_start": self.palm_at_lift_start_buf[env_id].item(),
                "grasp_cup_tilt_deg": self.grasp_tilt_at_lift_start_buf[env_id].item(),
                "lift_started": self.lift_started_buf[env_id].item(),
                "lift_success": self.episode_lift_success_buf[env_id].item(),
                "stabilize_success": self.episode_stabilize_success_buf[env_id].item(),
                "success": self.episode_success_buf[env_id].item(),
            })

        self.episode_success_buf[env_ids] = False
        self.episode_lift_success_buf[env_ids] = False
        self.episode_stabilize_success_buf[env_ids] = False

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
            self.demo_lift_palm_target_buf[env_ids] = (
                self.demo_grasp_reset_bank.lift_palm_pose_euler_zyx[demo_indices]
            )

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

            # 컵을 palm_ee 앞(palm 프레임 +X, cup_spawn_palm_front_x)에 스폰.
            # 열린 손 손바닥 정면에 정확히 위치 → 엄지팁 충돌 회피. z는 테이블 높이로 고정.
            # _quat_xyzw_from_euler_zyx 는 xyzw 반환 → isaaclab quat_apply(wxyz)용으로 재정렬.
            _pp_quat_xyzw = self._quat_xyzw_from_euler_zyx(pregrasp_palm_pose[:, 3:6])
            _pp_quat_wxyz = torch.cat([_pp_quat_xyzw[:, 3:4], _pp_quat_xyzw[:, :3]], dim=1)
            _palm_ee_pos = pregrasp_palm_pose[:, :3] + quat_apply(
                _pp_quat_wxyz, self._palm_ee_offset_local.unsqueeze(0).expand(n, -1)
            )
            _cup_front_local = torch.zeros(n, 3, device=self.device)
            _cup_front_local[:, 0] = float(self.cfg.cup_spawn_palm_front_x)
            obj_pos_local = _palm_ee_pos + quat_apply(_pp_quat_wxyz, _cup_front_local)
            obj_pos_local[:, 2] = self.cfg.object_spawn_z

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
        reset_palm_pose = self._euler_pose_to_quat_pose(pregrasp_palm_pose)
        self.palm_pose_targets[env_ids] = reset_palm_pose
        self.pregrasp_palm_pose_buf[env_ids] = pregrasp_palm_pose
        self.grasp_anchor_palm_pose_buf[env_ids] = reset_palm_pose
        self.lift_palm_start_pose_buf[env_ids] = reset_palm_pose

        self.open_tesollo_fabric.default_config[env_ids, :NUM_ARM_DOF] = q_pregrasp[:, :NUM_ARM_DOF]

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

        # ---- 7b. Bead 스폰 ----
        # 이산 4단계: {0, 10, 20, 30}개 × 10g = {0, 100, 200, 300}g 추가 질량
        # 총 컵 질량: 170g / 270g / 370g / 470g (1x / 1.6x / 2.2x / 2.8x)
        min_level = min(max(int(self.cfg.bead_count_min) // 10, 0), 3)
        max_level = min(max(int(self.cfg.bead_count_max) // 10, min_level), 3)
        _bead_lvl = torch.randint(min_level, max_level + 1, (n,), device=self.device)  # 0~3
        bead_count = _bead_lvl * 10  # {0, 10, 20, 30}

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
        self._bead_mass_normalized[env_ids] = bead_count.float() / self.cfg.num_beads

        # ---- 8. 버퍼 리셋 ----
        self.approach_hand_pose_buf[env_ids] = approach_hand
        self.hand_joint_targets[env_ids] = approach_hand
        self.lift_finger_pos_buf[env_ids] = approach_hand
        self.contact_force_raw[env_ids] = 0.0
        self.contact_friction_xyz_raw[env_ids] = 0.0
        self.binary_contact_buf[env_ids] = False
        self.num_contacts_buf[env_ids]   = 0
        self.distal_contact_force_raw[env_ids] = 0.0
        self.distal_binary_contact_buf[env_ids] = False
        self.middle_contact_force_raw[env_ids] = 0.0
        self.middle_binary_contact_buf[env_ids] = False
        self.palm_contact_force_raw[env_ids] = 0.0
        self.palm_binary_contact_buf[env_ids] = False
        self._prev_total_grip_force_buf[env_ids] = 0.0
        self._prev_num_contacts_buf[env_ids] = 0.0
        self._prev_middle_contacts_buf[env_ids] = 0.0
        self.success_flag[env_ids] = False
        self._success_hold_count[env_ids] = 0
        self._prev_avg_force_buf[env_ids] = 0.0       # force-smooth 초기화
        self._force_smooth_ready[env_ids] = False
        self.grasp_ready_hold_buf[env_ids] = 0
        self.lift_ready_latched_buf[env_ids] = False
        self.lift_started_buf[env_ids] = False
        self._grasp_started_buf[env_ids] = False
        self._grasp_anchor_set_buf[env_ids] = False
        self._lift_success_hold_count[env_ids] = 0
        self._lift_success_latched_buf[env_ids] = False
        self.is_approach_phase[env_ids] = True
        self.is_close_grasp_phase[env_ids] = False
        self.is_lift_phase[env_ids] = False
        self.is_stabilize_phase[env_ids] = False
        self.episode_stabilize_success_buf[env_ids] = False
        self._eval_mass_shift_done[env_ids] = False
        self.contacts_at_lift_start_buf[env_ids] = 0.0
        self.palm_at_lift_start_buf[env_ids] = 0.0
        self.grasp_tilt_at_lift_start_buf[env_ids] = 0.0
        # ---- Fabrics 상태 초기화 ----
        # 리셋 시 실제 로봇 상태로 동기화하여 첫 프레임 튐 방지
        # fabric_q(27D) = arm(7D) + hand(20D)
        arm_pos  = self.robot.data.joint_pos[env_ids][:, self.arm_dof_indices]
        hand_pos = self.robot.data.joint_pos[env_ids][:, self.hand_dof_indices]
        self.fabric_q[env_ids] = torch.cat([arm_pos, hand_pos], dim=-1)
        
        self.fabric_qd[env_ids] = 0.0
        self.fabric_qdd[env_ids] = 0.0
        
        self.hand_joint_targets[env_ids] = hand_pos
        self.approach_hand_pose_buf[env_ids] = hand_pos
        self.lift_finger_pos_buf[env_ids] = hand_pos
        self.palm_pose_targets[env_ids] = self._euler_pose_to_quat_pose(self.pregrasp_palm_pose_buf[env_ids])
        self.grasp_anchor_palm_pose_buf[env_ids] = self.palm_pose_targets[env_ids]
        self.lift_palm_start_pose_buf[env_ids] = self.palm_pose_targets[env_ids]

        self._eval_episode_started[env_ids] = False

        palm_pos = self.palm_pose_targets[env_ids, :3]
        palm_action = 2.0 * (
            palm_pos - self.palm_mins[:3]
        ) / (self.palm_maxs[:3] - self.palm_mins[:3]).clamp(min=1e-6) - 1.0
        self.actions[env_ids, PALM_POS_ACTION_SLICE] = palm_action.clamp(-1.0, 1.0)
        self.actions[env_ids, PALM_QUAT_ACTION_SLICE] = self.palm_pose_targets[env_ids, 3:7]
        self.actions[env_ids, FINGER_ACTION_SLICE] = 0.0
        self.prev_actions[env_ids] = self.actions[env_ids]

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
