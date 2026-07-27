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

"""환경 클래스: 5g_grasp_adapt

v10: v9 기반 버그 수정
- Fix 1: r_hj_thumb_1 (thumb abduction) = 0.0 고정 (v9: -0.283 → 엄지 치우침 수정)
- Fix 2: MIN_CONTACTS_FOR_SUCCESS = 4, ADR과 분리 (v9: 2접촉 success 오판정 수정)
- Fix 3: has_5_contact = num_contacts>=5 고정 (v9: has_4_contact와 동일 식 버그 수정)

Action (27D):
  [0:3]   palm xyz target → Fabrics IK → arm 7 DOF
  [3:7]   palm quaternion target (x, y, z, w), normalized by env
  [7:27]  20D residual around HAND_GRASP_POSE

Episode (4s @ 60Hz):
  Phase state is latched from contact/lift state for reward/gate/diagnostics only.
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

from openarm.common.grasp_reward_core import compute_grasp_reward_terms
from openarm.common.grasp_adaptive_core import compute_adaptive_grip_terms
from .demo_grasp_reset import DemoGraspResetBank, compute_demo_cup_spawn_local
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
    GRAVITY,
    ARM_START_POSE,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
    PALM_POS_ACTION_SLICE,
    PALM_QUAT_ACTION_SLICE,
    FINGER_ACTION_SLICE,
)
from .grasp_right_preset import (
    LEFT_ARM_REST_JOINT_POS,
    RIGHT_ACTUATED_JOINT_NAMES,
    HAND_APPROACH_POSE,
    HAND_GRASP_POSE,
    HAND_FULL_GRIP_POSE,
)
from .finger_action_utils import (
    compute_preset_residual_finger_targets,
)
from .grasp_reward_utils import compute_upright_success_mask
from .grasp_right_utils import (
    compute_precision_grasp_mask,
    compute_radial_compression,
    to_torch,
)


class GraspRightEnv(DirectRLEnv):
    """OpenArm+Teosllo 오른손 파지 환경 v10.3.

    Action: 27D
      [0:3]   palm xyz target, 정규화 [-1,1] → workspace
      [3:7]   palm quaternion target (x, y, z, w), unit normalize
      [7:27] 20D residual around HAND_GRASP_POSE, 정규화 [-1,1] → ±scale rad

    Phase labels are diagnostic/reward gates; actions are never overridden.
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

    @staticmethod
    def _build_tb_scalar_groups() -> dict[str, tuple[str, str]]:
        groups = {
            "reward/total": ("reward/summary", "total"),
            "reward/approach": ("reward/summary", "approach"),
            "reward/grasp": ("reward/summary", "grasp"),
            "reward/lift": ("reward/summary", "lift"),
            "reward/stabilize": ("reward/summary", "stabilize"),
            "reward/success_bonus": ("reward/summary", "success_bonus"),
            "reward/secure": ("reward/summary", "secure"),
            "reward/efficient": ("reward/summary", "efficient"),
            "reward/drop": ("reward/summary", "drop"),
            "reward/damage": ("reward/summary", "damage"),
            "task/radial_compression": ("task/damage", "radial"),
            "task/radial_compression_hold": ("task/damage", "radial_hold"),
            "task/buckle_rate": ("task/damage", "buckle_rate"),
            "reward/action_smooth": ("reward/summary", "action_smooth"),
            "contact/count": ("task/contact", "tip_count"),
            "task/middle_contact_rate": ("task/contact", "middle_rate"),
            "task/distal_contact_rate": ("task/contact", "distal_rate"),
            "contact/palm": ("task/contact", "palm_rate"),
            "contact/palm_force": ("task/contact", "palm_force"),
            "cup/tilt_deg": ("task/cup", "tilt_deg"),
            "cup/height_delta": ("task/cup", "height_delta"),
            "cup/xy_displacement": ("task/cup", "xy_displacement"),
            "task/precision_grasp_rate": ("task/progress", "precision_grasp_rate"),
            "task/five_tip_contact_rate": ("task/progress", "five_tip_contact_rate"),
            "task/lift_started_rate": ("task/progress", "lift_started_rate"),
            "task/lift_success_rate": ("task/success", "lift_success_rate"),
            "task/stabilize_success_rate": ("task/success", "stabilize_success_rate"),
            "task/success_rate": ("task/success", "success_rate"),
            "task/grip_force_n": ("task/grip", "force_n"),
            "task/grip_ratio": ("task/grip", "ratio"),
            "task/grip_ratio_hold": ("task/grip", "ratio_hold"),
            "task/slip_ratio": ("task/slip_ratio", "mean"),
            "task/slip_ratio_hold": ("task/slip_ratio", "hold"),
            "task/cup_slip_speed": ("task/slip", "cup_slip_speed"),
            "task/secure_quality_hold": ("task/slip", "secure_quality_hold"),
        }
        for bin_idx, bead_count in enumerate((0, 10, 20, 30)):
            tag = f"task/grip_ratio_hold/mass_bin_{bin_idx}"
            groups[tag] = ("task/grip_ratio_hold_by_mass", f"{bead_count}beads")
        for tip_idx in range(NUM_FINGERTIPS):
            tip_tag = f"sensor/tip_{tip_idx + 1}"
            groups[f"{tip_tag}/x"] = (tip_tag, "x")
            groups[f"{tip_tag}/y"] = (tip_tag, "y")
            groups[f"{tip_tag}/z"] = (tip_tag, "z")
        for joint_idx in range(NUM_HAND_DOF):
            finger_idx = joint_idx // 4 + 1
            local_joint_idx = joint_idx % 4 + 1
            joint_tag = f"joint/hand/f{finger_idx}/j{local_joint_idx}"
            groups[joint_tag] = (f"joint/hand/f{finger_idx}", f"j{local_joint_idx}")
        return groups

    def __init__(self, cfg: GraspRightEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.tb_scalar_groups = self._build_tb_scalar_groups()

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

        # body indices (통일 네이밍: r_hl_<finger>_*)
        _FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
        _tip_names = [f"r_hl_{fn}_tip" for fn in _FINGERS]
        self.fingertip_body_indices: list[int] = [
            self.robot.data.body_names.index(name) for name in _tip_names
        ]
        _palm_name = "r_hl_palm"
        self.palm_body_index: int = (
            self.robot.data.body_names.index(_palm_name)
            if _palm_name in self.robot.data.body_names
            else -1
        )
        _distal4_names = [f"r_hl_{fn}_4" for fn in _FINGERS]
        self.distal4_body_indices: list[int] = [
            self.robot.data.body_names.index(name)
            for name in _distal4_names
            if name in self.robot.data.body_names
        ]

        # middle phalanx (PIP, _3 link) body indices — FK 기반 actor obs용
        # sim2real 가능: joint encoder FK로 실기에서도 계산 가능
        _middle3_names = [f"r_hl_{fn}_3" for fn in _FINGERS]
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
        self.pregrasp_palm_pose_buf   = torch.zeros(self.num_envs, 6, device=self.device)
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
        # Stabilize(height-hold) phase 버퍼 (lift 성공 후 10cm 유지 단계)
        self.is_stabilize_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_stabilize_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ----------------------------------------------------------------
        # Hand joint targets (per-joint delta 결과)
        # ----------------------------------------------------------------
        self.hand_joint_targets = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)

        # ----------------------------------------------------------------
        # 접촉 상태 버퍼
        # ----------------------------------------------------------------
        self.contact_force_xyz_raw   = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        # fingertip body-local 힘 (fx,fy=shear, fz=normal) — obs용 sensor-faithful 표현
        self.tip_force_local         = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.contact_force_raw       = torch.zeros(self.num_envs, NUM_FINGERTIPS, device=self.device)
        self.contact_friction_xyz_raw = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.binary_contact_buf    = torch.zeros(self.num_envs, NUM_FINGERTIPS, dtype=torch.bool, device=self.device)
        self.num_contacts_buf      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.distal_contact_force_raw  = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, device=self.device)
        self.distal_binary_contact_buf = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, dtype=torch.bool, device=self.device)

        self.middle_contact_force_raw  = torch.zeros(self.num_envs, NUM_MIDDLE_SENSORS, device=self.device)
        self.middle_binary_contact_buf = torch.zeros(self.num_envs, NUM_MIDDLE_SENSORS, dtype=torch.bool, device=self.device)
        self._radial_compression_buf   = torch.zeros(self.num_envs, device=self.device)
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
        # world-frame tip force → fingertip body-local frame (실 tactile 센서 출력 정합)
        tip_quat = self.robot.data.body_quat_w[:, self.fingertip_body_indices]  # (N, 5, 4)
        tip_force_local = quat_apply_inverse(
            tip_quat.reshape(-1, 4), tip_xyz.reshape(-1, 3)
        ).view(self.num_envs, NUM_FINGERTIPS, 3)
        self.tip_force_local.copy_(torch.nan_to_num(tip_force_local, nan=0.0, posinf=0.0, neginf=0.0))
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

        # ---- Palm target: reset-local policy target with per-step rate limit ----
        palm_pos_raw = (
            self.pregrasp_palm_pose_buf[:, :3]
            + palm_pos_action * float(self.cfg.palm_local_workspace_radius)
        )
        palm_pos_raw = torch.max(
            torch.min(palm_pos_raw, self.palm_maxs[:3].unsqueeze(0)),
            self.palm_mins[:3].unsqueeze(0),
        )
        palm_pos_delta = (palm_pos_raw - self.palm_pose_targets[:, :3]).clamp(
            min=-float(self.cfg.palm_target_max_delta),
            max=float(self.cfg.palm_target_max_delta),
        )
        palm_pos = self.palm_pose_targets[:, :3] + palm_pos_delta
        palm_quat = torch.nn.functional.normalize(palm_quat_action, dim=-1, eps=1e-6)
        fallback_quat = self.palm_pose_targets[:, 3:7]
        fallback_quat = torch.nn.functional.normalize(fallback_quat, dim=-1, eps=1e-6)
        invalid_quat = palm_quat_action.norm(dim=-1, keepdim=True) <= 1e-6
        palm_quat = torch.where(invalid_quat, fallback_quat, palm_quat)

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

        self.palm_pose_targets[:, :3] = palm_pos
        self.palm_pose_targets[:, 3:7] = palm_quat
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

        hand_target = compute_preset_residual_finger_targets(
            preset_pos=self.hand_grasp_pose,
            finger_action=finger_action,
            lower_limits=self.hand_joint_lower_limits,
            upper_limits=self.hand_joint_upper_limits,
            residual_scale=self.cfg.hand_residual_scale,
            residual_mask=self.hand_residual_mask,
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
    # Observations: Actor 133D no-mass (134D optional mass/debug) | Critic 170D
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

        # tip force: fingertip body-local 3D 벡터 (fx,fy=shear, fz=normal; 5 × 3D = 15D)
        tip_force_local_norm = (
            self.tip_force_local / CONTACT_FORCE_MAX
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
            tip_force_local_norm,   # 15 (fingertip body-local: fx,fy shear + fz normal)
            middle_to_cup,          # 15
            phase_step_ratio,       # 1
        ])
        actor_obs = torch.cat(actor_obs_parts, dim=-1)

        actor_obs = torch.nan_to_num(actor_obs, nan=0.0, posinf=5.0, neginf=-5.0)

        if actor_obs.shape[1] != self.cfg.num_observations:
            raise RuntimeError(
                f"[grasp_adapt] Actor obs dim mismatch: {actor_obs.shape[1]} != {self.cfg.num_observations}"
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
            tip_force_local_norm,   # 15D (critic도 동일 fingertip-local 표현)
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
        # Phase 1: 5/5 envelope 강제 제거 → 엄지+대향2지 fingertip precision 파지.
        # 하위 gate(lift latch / adaptive hold / success)가 참조하는 변수명
        # full_tip_contact은 유지하되 의미는 precision 파지로 바뀐다.
        precision_grasp_bool = compute_precision_grasp_mask(
            self.binary_contact_buf, int(self.cfg.precision_min_opposing)
        )
        full_tip_contact = precision_grasp_bool.float()
        # 진짜 5접촉(과압축/envelope 회귀 진단 전용 지표)
        true_five_tip = (num_tip_contacts >= NUM_FINGERTIPS).float()
        palm_contact_bool = self.palm_binary_contact_buf
        palm_contact = palm_contact_bool.float()

        # Phase 1: lift latch 진입도 precision 파지(엄지+대향2지) 형성 기준으로.
        # (기존 num_tip_contacts >= stage0_lift_start_min_contacts 접촉수 게이트 대체)
        grasp_ready_now = precision_grasp_bool
        self.grasp_ready_hold_buf = torch.where(
            grasp_ready_now,
            self.grasp_ready_hold_buf + 1,
            torch.zeros_like(self.grasp_ready_hold_buf),
        )
        just_latched = (
            ~self.lift_ready_latched_buf
            & (self.grasp_ready_hold_buf >= self.cfg.grasp_ready_hold_steps)
        )
        if just_latched.any():
            self.contacts_at_lift_start_buf[just_latched] = num_tip_contacts[just_latched].float()
            self.palm_at_lift_start_buf[just_latched] = palm_contact[just_latched]
            self.grasp_tilt_at_lift_start_buf[just_latched] = cup_tilt_deg[just_latched]
        self.lift_ready_latched_buf |= just_latched
        self.lift_started_buf |= self.lift_ready_latched_buf
        self.is_lift_phase.copy_(self.lift_ready_latched_buf)

        lift_gate = self.lift_ready_latched_buf.float()
        lifted_gate = (cup_height_delta >= self.cfg.lift_success_height).float()
        lifted_bool = lifted_gate > 0.0

        # Stabilize(height-hold) phase: lift off 이후 컵을 세운 채 10cm까지 유지.
        self.is_stabilize_phase.copy_(self.lift_ready_latched_buf & lifted_bool)
        action_delta_norm = torch.nan_to_num((self.actions - self.prev_actions).norm(dim=-1), nan=0.0)
        contact_persistence_frac = (
            self.grasp_ready_hold_buf.float()
            / max(float(self.cfg.grasp_contact_persistence_reward_steps), 1.0)
        ).clamp(max=1.0)
        total, reward_terms, _ = compute_grasp_reward_terms(
            num_tip_contacts=num_tip_contacts,
            tip_contact_frac=tip_contact_frac,
            full_tip_contact=full_tip_contact,
            contact_persistence_frac=contact_persistence_frac,
            palm_to_cup_dist=palm_to_cup_dist,
            fingertip_side_dist=fingertip_side_dist,
            cup_height_delta=cup_height_delta,
            cup_xy_displacement=cup_xy_displacement,
            cup_tilt_deg=cup_tilt_deg,
            upright_quality=upright_quality,
            lift_latched=self.lift_ready_latched_buf,
            action_delta_norm=action_delta_norm,
            cfg=self.cfg,
        )
        hand_residual_magnitude = self.actions[:, FINGER_ACTION_SLICE].pow(2).mean(dim=-1)
        r_action_delta = reward_terms["action_smooth"]
        r_hand_residual_magnitude = (
            self.cfg.hand_residual_magnitude_weight * hand_residual_magnitude
        )
        r_action_smooth = r_action_delta + r_hand_residual_magnitude

        total = torch.nan_to_num(
            total + r_hand_residual_magnitude,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        # ---- 접촉 법선/접선력 분해 (grip force 합 + 진단 KPI) ----
        # 접촉 반력은 컵 반대 방향(cup→fingertip)을 향하므로 grip_axis = normalize(fingertip - cup).
        grip_axis = self.fingertip_pos - self.object_pos.unsqueeze(1)                 # (N,5,3) cup→fingertip
        grip_axis = grip_axis / grip_axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        f_world = self.contact_force_xyz_raw                                          # (N,5,3) world
        f_proj = (f_world * grip_axis).sum(dim=-1)                                    # (N,5) 부호 있는 법선 투영
        f_n = f_proj.clamp(min=0.0)                                                   # (N,5) 압축 법선력
        f_t = (f_world - f_proj.unsqueeze(-1) * grip_axis).norm(dim=-1)               # (N,5) 접선력(shear)
        contact_mask = self.binary_contact_buf.float()
        num_c = contact_mask.sum(dim=-1).clamp(min=1.0)

        # 질량은 reward 전용 privileged 값 (actor obs는 무게 hidden — tactile 추론)
        bead_count = (self._bead_mass_normalized * float(self.cfg.num_beads)).round()
        cup_weight = (
            float(self.cfg.cup_base_mass) + bead_count * float(self.cfg.bead_single_mass)
        ) * GRAVITY
        grip_normal_force = (f_n * contact_mask).sum(dim=-1)                          # ΣF_n
        force_ratio = grip_normal_force / cup_weight.clamp(min=1e-6)                  # KPI

        # slip proxy: 컵-손바닥 상대 속도 (의도된 lift 이동과 slip을 분리 — 같이 움직이면 slip 0)
        cup_lin_vel = self.cup.data.root_lin_vel_w
        palm_lin_vel = self.robot.data.body_lin_vel_w[:, self.palm_body_index]
        cup_slip_speed = (cup_lin_vel - palm_lin_vel).norm(dim=-1)

        # shear severity (진단 KPI 전용, 보상 미사용)
        shear_ratio = (f_t / (f_n + float(self.cfg.slip_shear_eps))).clamp(0.0, 1.0)
        slip_severity = (shear_ratio * contact_mask).sum(dim=-1) / num_c             # [0,1]

        # ---- 목적함수: friction-aware no-slip 최소 힘 (지배항) ----
        # 안 미끄러지는(secure) 최소 힘으로 압박 → mass·friction 적응이 emergent.
        r_objective, adaptive_terms = compute_adaptive_grip_terms(
            grip_normal_force=grip_normal_force,
            cup_weight=cup_weight,
            cup_speed=cup_slip_speed,
            tip_contact_frac=tip_contact_frac,
            lift_gate=lift_gate,
            lifted_gate=lifted_gate,
            full_tip_contact=full_tip_contact,
            cfg=self.cfg,
        )
        hold_gate = adaptive_terms["hold_gate"]

        # 손끝-only 자연 유도(하드웨어 제약): radial 압축 좌굴 fragile damage.
        # 감싸기(사방 마디가 벽 안으로 압박)=radial↑=파손, 손끝 국부 파지=radial↓.
        # Phase 1 geometry envelope penalty는 secure와 상충해 실패 → radial 물리로 대체.
        env_origins = self.scene.env_origins
        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cup_axis = quat_apply(self.object_rot, z_local)              # (N,3) 컵 up축
        cup_center = self.object_pos                                 # (N,3) env-local
        tip_pos = self.fingertip_pos                                 # (N,5,3) env-local
        tip_force = self.contact_force_xyz_raw                       # (N,5,3) world
        tip_mask = self.binary_contact_buf.float()                  # (N,5)
        middle_pos = (
            self.robot.data.body_pos_w[:, self.middle3_body_indices, :]
            - env_origins.unsqueeze(1)
        )                                                            # (N,5,3)
        middle_force = torch.nan_to_num(
            self._middle_sensor.data.net_forces_w, nan=0.0, posinf=0.0, neginf=0.0
        )                                                            # (N,5,3) world
        middle_mask = self.middle_binary_contact_buf.float()        # (N,5)
        contact_pos = torch.cat([tip_pos, middle_pos], dim=1)        # (N,10,3)
        contact_force = torch.cat([tip_force, middle_force], dim=1)  # (N,10,3)
        contact_mask = torch.cat([tip_mask, middle_mask], dim=1)     # (N,10)
        radial_compression = compute_radial_compression(
            contact_pos, contact_force, cup_center, cup_axis, contact_mask
        )                                                            # (N,)
        self._radial_compression_buf.copy_(radial_compression)
        # 순간 penalty: hold 구간 F_safe 초과분
        r_damage = (
            -float(self.cfg.damage_penalty_weight)
            * hold_gate
            * torch.relu(radial_compression - float(self.cfg.f_safe))
        )
        # 진단(로깅용): middle/distal 접촉률
        middle_contact_frac = middle_mask.sum(dim=-1) / float(NUM_MIDDLE_SENSORS)
        distal_contact_frac = (
            self.distal_binary_contact_buf.float().sum(dim=-1) / float(NUM_DISTAL_SENSORS)
        )

        # 평가 bonus (Fork C): 10cm·upright·5접촉 유지 시 보너스. 축소·force-quality 비결합.
        # lift/stabilize는 적응을 "평가"하는 gate일 뿐, 목적이 아니다 (weight는 cfg에서 축소).
        base_success_bonus = reward_terms["success_bonus"]
        reached_target_gate = (cup_height_delta >= float(self.cfg.lift_target_height)).float()
        final_upright = (
            cup_tilt_deg <= float(self.cfg.stabilize_upright_max_deg)
        ).float()
        height_hold_success_bonus = (
            float(self.cfg.success_bonus_weight)
            * lift_gate
            * reached_target_gate
            * full_tip_contact
            * final_upright
        )
        reward_terms["success_bonus"] = height_hold_success_bonus

        total = torch.nan_to_num(
            total - base_success_bonus + height_hold_success_bonus + r_objective + r_damage,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        # 파손(좌굴) 순간 음의 보상: radial이 f_buckle 초과 시 (종료는 _get_dones)
        buckle_now = radial_compression > float(self.cfg.f_buckle)
        total = total - float(self.cfg.buckle_penalty) * buckle_now.float()

        if len(self._success_window) >= 10:
            _ep_success_rate = sum(self._success_window) / len(self._success_window)
        else:
            _ep_success_rate = self._successful_episodes / max(self._total_episodes, 1)

        self.extras["reward/approach"] = reward_terms["approach"].mean()
        self.extras["reward/grasp"] = reward_terms["grasp"].mean()
        self.extras["reward/lift"] = reward_terms["lift"].mean()
        self.extras["reward/stabilize"] = reward_terms["stabilize"].mean()
        self.extras["reward/success_bonus"] = reward_terms["success_bonus"].mean()
        self.extras["reward/secure"] = adaptive_terms["r_secure"].mean()
        self.extras["reward/efficient"] = adaptive_terms["r_efficient"].mean()
        self.extras["reward/drop"] = adaptive_terms["r_drop"].mean()
        self.extras["reward/damage"] = r_damage.mean()
        self.extras["task/radial_compression"] = radial_compression.mean()
        self.extras["task/radial_compression_hold"] = (
            radial_compression * hold_gate
        ).sum() / hold_gate.sum().clamp(min=1.0)
        self.extras["task/middle_contact_rate"] = middle_contact_frac.mean()
        self.extras["task/distal_contact_rate"] = distal_contact_frac.mean()
        # 적응 검증: grip_ratio가 mass bin별로 공통값에 수렴해야 함 (force ∝ mass)
        self.extras["task/grip_force_n"] = grip_normal_force.mean()
        self.extras["task/grip_ratio"] = force_ratio.mean()
        _hold_n = hold_gate.sum().clamp(min=1.0)
        self.extras["task/grip_ratio_hold"] = (force_ratio * hold_gate).sum() / _hold_n
        mass_bin = (self._bead_mass_normalized * 3.0).round().long().clamp(0, 3)
        for bin_idx in range(4):
            bin_hold = hold_gate * (mass_bin == bin_idx).float()
            self.extras[f"task/grip_ratio_hold/mass_bin_{bin_idx}"] = (
                force_ratio * bin_hold
            ).sum() / bin_hold.sum().clamp(min=1.0)
        # no-slip 검증: 주 지표 = 컵-손 상대속도(secure), 보조 = shear severity
        self.extras["task/cup_slip_speed"] = cup_slip_speed.mean()
        self.extras["task/secure_quality_hold"] = (
            adaptive_terms["secure_quality"] * hold_gate
        ).sum() / _hold_n
        self.extras["task/slip_ratio"] = slip_severity.mean()
        self.extras["task/slip_ratio_hold"] = (slip_severity * hold_gate).sum() / _hold_n
        self.extras["reward/action_smooth"] = r_action_smooth.mean()
        self.extras["reward/total"] = total.mean()
        self.extras["contact/count"] = num_tip_contacts.float().mean()
        self.extras["contact/palm"] = palm_contact.mean()
        self.extras["contact/palm_force"] = self.palm_contact_force_raw.mean()
        self.extras["cup/tilt_deg"] = cup_tilt_deg.mean()
        self.extras["cup/height_delta"] = cup_height_delta.mean()
        self.extras["cup/xy_displacement"] = cup_xy_displacement.mean()
        self.extras["task/precision_grasp_rate"] = full_tip_contact.mean()
        self.extras["task/five_tip_contact_rate"] = true_five_tip.mean()
        self.extras["task/lift_started_rate"] = self.lift_started_buf.float().mean()
        self.extras["task/lift_success_rate"] = self._lift_success_latched_buf.float().mean()
        self.extras["task/stabilize_success_rate"] = self.episode_stabilize_success_buf.float().mean()
        self.extras["task/success_rate"] = torch.tensor(_ep_success_rate, device=self.device)

        tip_force_mean = self.tip_force_local.mean(dim=0)
        for tip_idx in range(NUM_FINGERTIPS):
            tip_tag = f"sensor/tip_{tip_idx + 1}"
            self.extras[f"{tip_tag}/x"] = tip_force_mean[tip_idx, 0]
            self.extras[f"{tip_tag}/y"] = tip_force_mean[tip_idx, 1]
            self.extras[f"{tip_tag}/z"] = tip_force_mean[tip_idx, 2]

        hand_joint_pos_mean = self.robot.data.joint_pos[:, self.hand_dof_indices].mean(dim=0)
        for joint_idx in range(NUM_HAND_DOF):
            finger_idx = joint_idx // 4 + 1
            local_joint_idx = joint_idx % 4 + 1
            self.extras[f"joint/hand/f{finger_idx}/j{local_joint_idx}"] = hand_joint_pos_mean[joint_idx]

        self._prev_total_grip_force_buf.copy_(self.contact_force_raw.sum(dim=-1))
        self._prev_num_contacts_buf.copy_(num_tip_contacts.float())
        self._prev_middle_contacts_buf.copy_(self.middle_binary_contact_buf.float().sum(dim=-1))
        return total


    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.extras.clear()
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
        reached_target = self.object_pos[:, 2] > (self.object_init_pos[:, 2] + self.cfg.lift_target_height)
        # Phase 1: 5/5 접촉 → 엄지+대향2지 fingertip precision 파지(lift/success 판정 gate)
        full_tip_contact = compute_precision_grasp_mask(
            self.binary_contact_buf, int(self.cfg.precision_min_opposing)
        )
        upright_success = compute_upright_success_mask(
            cup_z_world[:, 2],
            self.cfg.success_upright_max_deg,
        )
        stabilize_upright_success = compute_upright_success_mask(
            cup_z_world[:, 2],
            self.cfg.stabilize_upright_max_deg,
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
        stabilize_success_now = self._lift_success_latched_buf & lifted & full_tip_contact & upright_success
        self.episode_stabilize_success_buf |= stabilize_success_now

        # 최종 성공: lift 성공 latch 후 컵을 10cm까지 올려 5-tip grasp·엄격 upright로 유지
        final_success_now = (
            self._lift_success_latched_buf
            & reached_target
            & full_tip_contact
            & stabilize_upright_success
        )
        self.success_flag.copy_(final_success_now)
        self.episode_success_buf |= final_success_now
        self._success_hold_count = torch.where(
            final_success_now,
            self._success_hold_count + 1,
            torch.zeros_like(self._success_hold_count),
        )
        final_success_held = self._success_hold_count >= int(self.cfg.success_hold_steps)

        # Phase 2: radial 압축이 f_buckle 초과 → 컵 좌굴(파손) 종료
        buckle = self._radial_compression_buf > float(self.cfg.f_buckle)
        self.extras["task/buckle_rate"] = buckle.float().mean()
        terminated = out_x | out_y | fallen | tipped | final_success_held | buckle
        truncated  = self.episode_length_buf >= self.max_episode_length - 1

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
        self.palm_pose_targets[env_ids] = self._euler_pose_to_quat_pose(pregrasp_palm_pose)
        self.pregrasp_palm_pose_buf[env_ids] = pregrasp_palm_pose

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
        self.hand_joint_targets[env_ids] = approach_hand
        self.contact_force_raw[env_ids] = 0.0
        self.tip_force_local[env_ids] = 0.0
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
        self._lift_success_hold_count[env_ids] = 0
        self._lift_success_latched_buf[env_ids] = False
        self.is_lift_phase[env_ids] = False
        # Stabilize(height-hold) phase 버퍼
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
        self.palm_pose_targets[env_ids] = self._euler_pose_to_quat_pose(self.pregrasp_palm_pose_buf[env_ids])

        self._eval_episode_started[env_ids] = False

        # actions 리셋: palm은 현재 pregrasp target, hand는 preset residual 0.
        palm_pos = self.palm_pose_targets[env_ids, :3]
        palm_action = 2.0 * (palm_pos - self.palm_mins[:3]) / (self.palm_maxs[:3] - self.palm_mins[:3]).clamp(min=1e-6) - 1.0
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
