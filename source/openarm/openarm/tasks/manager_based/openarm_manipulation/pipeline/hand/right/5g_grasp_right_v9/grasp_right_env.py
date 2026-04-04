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

"""환경 클래스: 5g_grasp_right_v9

v9: v8 기반 + 20D 손가락 직접 제어 + Slip/Force-Efficiency/Force-Smooth reward
- 5D synergy lerp 제거 → 20D per-joint position delta (±finger_delta_scale rad/step)
- R_slip: cup 수평 속도 기반 slip proxy (contact gate)
- R_force_efficiency: F_total/mg 기반 최소 충분 파지력 유도
- R_force_smooth: 파지력 변화율 억제 (sim2real 안정성)

Action (26D):
  [0:6]  6D palm pose → Fabrics IK → arm 7 DOF
  [6:26] 20D per-joint finger delta: action × finger_delta_scale [rad]

Episode (10s @ 60Hz):
  Grasp phase (0~479): Fabrics arm + per-joint finger delta
  Lift  phase (480~599): scripted task-space lift + micro-delta hand
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
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

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
    NUM_OBSERVATIONS,
    NUM_DISTAL_SENSORS,
    NUM_MIDDLE_SENSORS,
    NUM_CRITIC_OBSERVATIONS,
    GRASP_PHASE_STEPS,
    LIFT_PHASE_STEPS,
    LIFT_START_STEP,
    EPISODE_STEPS,
    LIFT_Z_DELTA,
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
    OBJECT_GOAL_POS,
)
from .grasp_right_utils import scale, to_torch


class GraspRightEnv(DirectRLEnv):
    """OpenArm+Teosllo 오른손 파지 환경 v9.

    Action: 26D
      [0:6]  palm pose (x,y,z,ez,ey,ex), 정규화 [-1,1] → Fabrics IK
      [6:26] 20D per-joint finger delta: action × finger_delta_scale [rad]

    Episode:
      Grasp phase (step 0~479):  Fabrics arm + per-joint finger delta
      Lift  phase (step 480~599): scripted task-space lift + micro-delta hand
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

        self.arm_dof_indices  = self.actuated_dof_indices[:NUM_ARM_DOF]
        self.hand_dof_indices = self.actuated_dof_indices[NUM_ARM_DOF:]

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

        # ----------------------------------------------------------------
        # Hand 관절 한계 (per-joint delta 클램프용)
        # soft_joint_pos_limits: (num_envs, num_joints, 2) — [lower, upper]
        # ----------------------------------------------------------------
        hand_limits = self.robot.data.soft_joint_pos_limits[0, self.hand_dof_indices, :]  # (20, 2)
        self.hand_joint_lower_limits = hand_limits[:, 0].contiguous()  # (20,)
        self.hand_joint_upper_limits = hand_limits[:, 1].contiguous()  # (20,)

        # 외전/내전 관절(abduction) delta scale 마스크 — 0으로 설정 시 사실상 고정
        # RIGHT_HAND_JOINT_NAMES = [rj_dg_{f}_{j} for f in 1~5, j in 1~4]
        # 고정 인덱스: 0(thm_ab), 4(idx_ab), 8(mid_ab), 12(rng_ab), 16(pnk_z), 17(pnk_ab)
        self.finger_delta_mask = torch.ones(NUM_HAND_DOF, device=self.device)
        self.finger_delta_mask[[0, 4, 8, 12, 16, 17]] = 0.0

        # ----------------------------------------------------------------
        # 접근 자세 (reset 및 Fabrics null-space용)
        # v9는 lerp를 쓰지 않으나 reset 초기화와 Fabrics attractor에서는 계속 필요.
        # ----------------------------------------------------------------
        self.hand_approach_pose   = to_torch(HAND_APPROACH_POSE,   device=self.device)  # (20,)
        self.hand_grasp_pose      = to_torch(HAND_GRASP_POSE,      device=self.device)  # (20,)
        self.hand_full_grip_pose  = to_torch(HAND_FULL_GRIP_POSE,  device=self.device)  # (20,)

        # ----------------------------------------------------------------
        # approach_pose 기준 관절 한계 재조정 — 반대 방향 휘어짐 방지
        # curl 양수 관절: lower = max(original, approach)  → approach보다 더 열리는 것 차단
        # curl 음수 관절 (thumb_2, 20D index 1): upper = min(original, approach) → approach보다 더 펴지는 것 차단
        # ----------------------------------------------------------------
        _approach = self.hand_approach_pose  # (20,)
        _new_lower = torch.max(self.hand_joint_lower_limits, _approach)
        _new_upper = self.hand_joint_upper_limits.clone()
        _new_lower[1] = self.hand_joint_lower_limits[1]                          # thumb_2: lower는 원래값 유지
        _new_upper[1] = torch.min(self.hand_joint_upper_limits[1], _approach[1]) # thumb_2: approach 이상으로 펴지는 것 차단
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
        self.object_goal = (
            to_torch(OBJECT_GOAL_POS, device=self.device)
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
        self.actions         = torch.zeros(self.num_envs, cfg.num_actions, device=self.device)
        self.prev_actions    = torch.full((self.num_envs, cfg.num_actions), 0.0, device=self.device)

        # ----------------------------------------------------------------
        # Pregrasp / Lift 버퍼
        # ----------------------------------------------------------------
        self.pregrasp_arm_pos_buf = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        self.lift_finger_pos_buf  = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        self.is_lift_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ----------------------------------------------------------------
        # Hand joint targets (per-joint delta 결과)
        # ----------------------------------------------------------------
        self.hand_joint_targets = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)

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

        per_distal = self._distal_sensor.data.net_forces_w.norm(dim=-1)
        per_distal = torch.nan_to_num(per_distal, nan=0.0, posinf=0.0, neginf=0.0)
        self.distal_contact_force_raw.copy_(per_distal)
        self.distal_binary_contact_buf.copy_(per_distal > CONTACT_FORCE_THRESHOLD)

        per_middle = self._middle_sensor.data.net_forces_w.norm(dim=-1)
        per_middle = torch.nan_to_num(per_middle, nan=0.0, posinf=0.0, neginf=0.0)
        self.middle_contact_force_raw.copy_(per_middle)
        self.middle_binary_contact_buf.copy_(per_middle > CONTACT_FORCE_THRESHOLD)

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone()

        palm_action   = actions[:, :6]    # (N, 6) ∈ [-1, 1]
        finger_action = actions[:, 6:26]  # (N, 20) ∈ [-1, 1] — per-joint delta

        # ---- Phase 판정 ----
        is_lift = (self.episode_length_buf >= LIFT_START_STEP)
        self.is_lift_phase.copy_(is_lift)

        # Eval: grasp phase 동안 finger action 버퍼링
        grasp_mask = ~is_lift
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

        # ---- Lift 진입 시 기준점 캡처 ----
        just_entering_lift = (self.episode_length_buf == LIFT_START_STEP)

        if just_entering_lift.any():
            prev_finger_action = self._last_grasp_finger_action[just_entering_lift]
            self._eval_grip_at_lift[just_entering_lift] = prev_finger_action.abs().mean(dim=-1)
            self._eval_finger_actions_at_lift[just_entering_lift] = prev_finger_action
            self._eval_lift_snapshot_valid[just_entering_lift] = True

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

        # ---- Palm pose 계산 ----
        delta = scale(palm_action, self.delta_mins, self.delta_maxs)
        grasp_palm_pose = self.pregrasp_palm_pose_buf + delta
        grasp_palm_pose = torch.max(torch.min(grasp_palm_pose, self.palm_maxs), self.palm_mins)

        lift_progress = (
            (self.episode_length_buf - LIFT_START_STEP).clamp(min=0).float()
            / LIFT_PHASE_STEPS
        ).clamp(max=1.0).unsqueeze(1)
        lift_palm_pose = self.lift_palm_start_pose_buf.clone()
        lift_palm_pose[:, 2] = lift_palm_pose[:, 2] + LIFT_Z_DELTA * lift_progress.squeeze(1)
        lift_palm_pose = torch.max(torch.min(lift_palm_pose, self.palm_maxs), self.palm_mins)

        palm_pose = torch.where(is_lift.unsqueeze(1), lift_palm_pose, grasp_palm_pose)
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

        # ---- Per-joint finger absolute offset: Approach + (action * range) ----
        # v9.2: action=0 → APPROACH_POSE, action=1 → FULL_GRIP_POSE (주먹)
        # joint limit clamp이 물리적 한계 보호 (별도 headroom 불필요)
        finger_range = (self.hand_full_grip_pose - self.hand_approach_pose).unsqueeze(0)
        
        hand_target = self.hand_approach_pose.unsqueeze(0) + finger_action * finger_range
        
        hand_target = hand_target.clamp(
            self.hand_joint_lower_limits.unsqueeze(0),
            self.hand_joint_upper_limits.unsqueeze(0),
        )
        self.hand_joint_targets.copy_(hand_target)

        # fabric_q hand 부분 동기화
        self.fabric_q[:, NUM_ARM_DOF:] = hand_target
        self.fabric_qd[:, NUM_ARM_DOF:].zero_()

    def _apply_action(self) -> None:
        is_lift = self.is_lift_phase

        # ---- 오른팔 ----
        self.robot.set_joint_position_target(self.fabric_q[:, :NUM_ARM_DOF], joint_ids=self.arm_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(self.fabric_q[:, :NUM_ARM_DOF]), joint_ids=self.arm_dof_indices
        )

        # ---- 오른손 ----
        # Grasp phase: per-joint delta target
        # Lift  phase: 캡처 자세 + action 기반 micro-delta (±lift_finger_delta_scale rad)
        lift_micro = (
            self.actions[:, 6:26]                         # (N,20)
            * self.cfg.lift_finger_delta_scale
            * self.finger_delta_mask                      # abduction joints 고정
        )
        lift_finger_target = (self.lift_finger_pos_buf + lift_micro).clamp(
            self.hand_joint_lower_limits.unsqueeze(0),
            self.hand_joint_upper_limits.unsqueeze(0),
        )
        finger_target = torch.where(
            is_lift.unsqueeze(1),
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

        self._update_contact_forces()

    # ------------------------------------------------------------------
    # Observations: Actor 128D | Critic 164D
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

        palm_to_cup      = cup_pos_noisy - palm_center_pos
        cup_to_fingertip = (
            fingertip_pos - cup_pos_noisy.unsqueeze(1)
        ).view(self.num_envs, -1)

        binary_contact   = self.binary_contact_buf.float()
        last_actions     = self.actions  # (N, 26)

        tip_force_norm   = (self.contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)

        phase_step_ratio = (
            self.episode_length_buf.float() / EPISODE_STEPS
        ).unsqueeze(1)

        actor_obs = torch.cat([
            arm_joint_pos,          # 7
            arm_joint_vel,          # 7
            finger_joint_pos,       # 20
            finger_joint_vel,       # 20
            palm_center_pos,        # 3
            fingertip_pos_rel_palm, # 15
            palm_to_cup,            # 3
            cup_to_fingertip,       # 15
            binary_contact,         # 5
            last_actions,           # 26 (v9: 11→26)
            self._bead_mass_normalized.unsqueeze(-1),  # 1
            tip_force_norm,         # 5
            phase_step_ratio,       # 1
        ], dim=-1)   # 128D

        if actor_obs.shape[1] != NUM_OBSERVATIONS:
            raise RuntimeError(
                f"[v9] Actor obs dim mismatch: {actor_obs.shape[1]} != {NUM_OBSERVATIONS}"
            )

        # ==== Critic extra obs (36D) ====
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

        actor_obs_clean = torch.cat([
            arm_joint_pos_clean,
            arm_joint_vel_clean,
            finger_joint_pos_clean,
            finger_joint_vel_clean,
            palm_center_pos_clean,
            (fingertip_pos_clean - palm_center_pos_clean.unsqueeze(1)).view(self.num_envs, -1),
            cup_pos_clean - palm_center_pos_clean,
            (fingertip_pos_clean - cup_pos_clean.unsqueeze(1)).view(self.num_envs, -1),
            binary_contact,
            last_actions,
            self._bead_mass_normalized.unsqueeze(-1),
            tip_force_norm,
            phase_step_ratio,
        ], dim=-1)   # 128D

        critic_obs = torch.cat([
            actor_obs_clean,        # 128
            cup_lin_vel,            # 3
            cup_ang_vel,            # 3
            cup_rot,                # 4
            cup_height_delta,       # 1
            distal_binary,          # 5
            distal_force_norm,      # 5
            middle_binary,          # 5
            middle_force_norm,      # 5
            fingertip_signed_dist,  # 5
        ], dim=-1)   # 164D

        if critic_obs.shape[1] != NUM_CRITIC_OBSERVATIONS:
            raise RuntimeError(
                f"[v9] Critic obs dim mismatch: {critic_obs.shape[1]} != {NUM_CRITIC_OBSERVATIONS}"
            )

        return {"policy": actor_obs, "critic": critic_obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        enclosure_weight = self.cfg.enclosure_weight

        cup_height_delta = (
            self.object_pos[:, 2] - self.object_init_pos[:, 2]
        ).clamp(min=0.0)

        # ---- R0. palm_approach ----
        grasp_center_approach = self.object_pos.clone()
        grasp_center_approach[:, 2] += self.cfg.cup_grasp_z_offset
        palm_to_cup_dist = (self.palm_center_pos - grasp_center_approach).norm(dim=-1)
        r0_palm_approach = self.cfg.palm_approach_weight * torch.exp(
            -self.cfg.palm_approach_sharpness * palm_to_cup_dist
        )

        # ---- R1. fingertip_enclosure ----
        grasp_center = grasp_center_approach

        cup_to_palm_xy = self.palm_center_pos[:, :2] - grasp_center[:, :2]
        approach_dir_xy = cup_to_palm_xy / cup_to_palm_xy.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-6)
        perp_dir_xy = torch.stack(
            [-approach_dir_xy[:, 1], approach_dir_xy[:, 0]], dim=1
        )

        self._approach_dir_buf[:, :2] = perp_dir_xy
        self._approach_dir_buf[:, 2]  = 0.0

        r = self.cfg.cup_radius_approx
        thumb_target  = grasp_center + self._approach_dir_buf * r
        others_target = grasp_center - self._approach_dir_buf * r

        thumb_dist  = (self.fingertip_pos[:, 0, :] - thumb_target).norm(dim=-1)
        others_dist = (self.fingertip_pos[:, 1:, :] - others_target.unsqueeze(1)).norm(
            dim=-1
        ).mean(dim=-1)

        tw = self.cfg.enclosure_thumb_weight
        r1_enclosure = enclosure_weight * (
            tw * torch.exp(-self.cfg.enclosure_sharpness * thumb_dist)
            + (1.0 - tw) * torch.exp(-self.cfg.enclosure_sharpness * others_dist)
        )

        # ---- 접촉력 공통 ----
        thumb_force      = self.contact_force_raw[:, 0]
        others_avg_force = self.contact_force_raw[:, 1:].mean(dim=-1)

        # ---- R1b. force_balance ----
        has_thumb_contact  = self.binary_contact_buf[:, 0].float()
        has_others_contact = (self.binary_contact_buf[:, 1:].sum(-1) >= 1).float()
        balance_gate       = has_thumb_contact * has_others_contact
        force_balance_err  = (thumb_force - others_avg_force).abs()
        r1b_force_balance = (
            self.cfg.force_balance_weight
            * balance_gate
            * torch.exp(-self.cfg.force_balance_sharpness * force_balance_err)
        )

        # ---- R1c. multi_phalanx_contact ----
        tip_norm    = (self.contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)
        middle_norm = (self.middle_contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)
        finger_depth = (tip_norm * middle_norm).sqrt()
        r1c_multi_phalanx = self.cfg.multi_phalanx_weight * finger_depth.mean(dim=-1)

        # ---- cup uprightness ----
        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cup_z_world   = quat_apply(self.object_rot, z_local)
        cup_uprightness = cup_z_world[:, 2].clamp(min=0.0)

        # ---- R2. slip_reward ----
        # v9.2: lift phase에서만 활성 — grasp phase에서 "접촉만 해도 보상" local-min 차단
        cup_horiz_vel = self.cup.data.root_lin_vel_w[:, :2].norm(dim=-1)   # (N,)
        cup_horiz_vel = torch.nan_to_num(cup_horiz_vel, nan=0.0)
        has_any_contact  = (self.num_contacts_buf >= 1).float()
        r2_slip = (
            self.cfg.slip_weight
            * self.is_lift_phase.float()
            * has_any_contact
            * torch.exp(-self.cfg.slip_sharpness * cup_horiz_vel)
        )

        # ---- R3. Adaptive Force Reward (v9.3: k 제거, 순수 효율 gradient) ----
        # 역할 분담:
        #   slip        → force 부족 커버 (컵 미끄러지면 penalty)
        #   adaptive_force → force 과도 억제 (force_ratio 클수록 보상 감소)
        # 수식: exp(-decay * force_ratio)
        #   force_ratio = total_grip / mg  (dimensionless)
        #   decay=0.3: ratio=2→0.55, ratio=4→0.30
        # k 없음. slip과의 균형으로 policy가 최적 ratio를 스스로 학습
        total_grip_force = self.contact_force_raw.sum(dim=-1)        # (N,) [N]
        grip_normalized  = (total_grip_force / (CONTACT_FORCE_MAX * NUM_FINGERTIPS)).clamp(0.0, 1.0)

        effective_mass = (
            self.cfg.cup_base_mass
            + self._bead_mass_normalized * self.cfg.bead_count_max * self.cfg.bead_single_mass
        )   # (N,) [kg]
        mg = effective_mass * 9.81                                    # (N,) [N]
        force_ratio = total_grip_force / (mg + 1e-4)                  # (N,) dimensionless

        af_weight = (
            self.grasp_adr.get_param("reward", "adaptive_force_weight")
            if self.grasp_adr is not None
            else self.cfg.adaptive_force_weight
        )

        r3_adaptive_force = (
            af_weight
            * self.is_lift_phase.float()
            * has_any_contact
            * torch.exp(-self.cfg.adaptive_force_decay * force_ratio)
        )

        # ---- R5. force_smooth (v9 신규) ----
        # 파지력 변화율 (mass-normalized) 억제
        force_delta_norm = (total_grip_force - self._prev_avg_force_buf) / (mg + 1e-4)
        # 에피소드 시작 직후(ready=False)에는 현재 force를 기준값으로 세팅하고 패널티를 주지 않는다.
        force_delta_norm = torch.where(
            self._force_smooth_ready,
            force_delta_norm,
            torch.zeros_like(force_delta_norm),
        )
        r5_force_smooth  = -self.cfg.force_smooth_weight * force_delta_norm.pow(2)
        self._force_smooth_ready.fill_(True)
        self._prev_avg_force_buf.copy_(total_grip_force)

        # ---- R6. lift_reward ----
        r6_lift = self.cfg.lift_reward_weight * cup_height_delta * cup_uprightness

        # ---- R8. success_bonus ----
        # lift 성공 조건 유지 중 step당 보너스 (직전 step success_flag 사용, 1 step lag 무방)
        r8_success = self.cfg.success_bonus_weight * self.success_flag.float()

        # ---- R7. action_smoothness ----
        palm_delta   = (self.actions[:, :6] - self.prev_actions[:, :6]).pow(2).sum(dim=-1)
        finger_delta = (self.actions[:, 6:] - self.prev_actions[:, 6:]).pow(2).sum(dim=-1)
        r7_action_smooth = (
            self.cfg.action_smoothness_palm_weight   * palm_delta
            + self.cfg.action_smoothness_finger_weight * finger_delta
        )

        # ---- 합산 ----
        total = (
            r0_palm_approach
            + r1_enclosure
            + r1b_force_balance
            + r1c_multi_phalanx
            + r2_slip
            + r3_adaptive_force
            + r5_force_smooth
            + r6_lift
            + r7_action_smooth
            + r8_success
        )
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- ADR increment ----
        _ep_success_rate = self._successful_episodes / max(self._total_episodes, 1)
        if self.grasp_adr is not None:
            self.grasp_adr.maybe_increment(_ep_success_rate)

        # ---- 로깅 ----
        self.extras["palm_approach_reward"]   = r0_palm_approach.mean()
        self.extras["enclosure_reward"]       = r1_enclosure.mean()
        self.extras["force_balance_reward"]   = r1b_force_balance.mean()
        self.extras["multi_phalanx_reward"]   = r1c_multi_phalanx.mean()
        self.extras["slip_reward"]            = r2_slip.mean()
        self.extras["adaptive_force_reward"]  = r3_adaptive_force.mean()
        self.extras["force_smooth_reward"]    = r5_force_smooth.mean()
        self.extras["lift_reward"]            = r6_lift.mean()
        self.extras["success_bonus"]          = r8_success.mean()
        self.extras["action_smoothness"]      = r7_action_smooth.mean()
        # 진단 지표
        self.extras["force_ratio_mean"]       = force_ratio.mean()
        self.extras["force_balance_err"]      = force_balance_err.mean()
        self.extras["thumb_force_mean"]       = thumb_force.mean()
        self.extras["others_avg_force_mean"]  = others_avg_force.mean()
        self.extras["total_tip_force_norm"]   = grip_normalized.mean()
        self.extras["multi_phalanx_contact"]  = finger_depth.mean()
        self.extras["grip_normalized"]        = grip_normalized.mean()
        self.extras["contact_active_ratio"]   = has_any_contact.mean()
        self.extras["bead_mass_normalized"]   = self._bead_mass_normalized.mean()
        self.extras["num_contacts"]           = self.num_contacts_buf.float().mean()
        self.extras["episode_success_rate"]   = torch.tensor(_ep_success_rate, device=self.device)
        # v9 신규 진단 지표
        self.extras["slip_proxy_cup_vel"]     = cup_horiz_vel.mean()
        self.extras["effective_mass_mean"]    = effective_mass.mean()
        self.extras["adaptive_force_weight"]  = torch.tensor(af_weight, device=self.device)
        # mass별 조건부 지표
        light_mask = (self._bead_mass_normalized < 0.33)
        heavy_mask = (self._bead_mass_normalized > 0.66)
        grip_norm_light = grip_normalized[light_mask].mean() if light_mask.any() else grip_normalized.mean()
        grip_norm_heavy = grip_normalized[heavy_mask].mean() if heavy_mask.any() else grip_normalized.mean()
        if light_mask.any():
            self.extras["grip_norm_light"]      = grip_norm_light
            self.extras["tip_force_norm_light"] = grip_normalized[light_mask].mean()
            self.extras["force_ratio_light"]    = force_ratio[light_mask].mean()
        if heavy_mask.any():
            self.extras["grip_norm_heavy"]      = grip_norm_heavy
            self.extras["tip_force_norm_heavy"] = grip_normalized[heavy_mask].mean()
            self.extras["force_ratio_heavy"]    = force_ratio[heavy_mask].mean()
        self.extras["grip_adaptive_delta"]    = grip_norm_heavy - grip_norm_light
        self.extras["force_ratio_delta"]      = (
            force_ratio[heavy_mask].mean() - force_ratio[light_mask].mean()
            if (light_mask.any() and heavy_mask.any())
            else torch.tensor(0.0, device=self.device)
        )
        if self.grasp_adr is not None:
            self.extras["adr_progress"]            = torch.tensor(self.grasp_adr.progress, device=self.device)
            self.extras["adr_spawn_xy_range"]      = torch.tensor(self.grasp_adr.get_param("spawn",  "object_spawn_xy_range"), device=self.device)
            self.extras["adr_obs_noise_cup_pos"]   = torch.tensor(self.grasp_adr.get_param("noise",  "obs_noise_cup_pos"),     device=self.device)
            self.extras["adr_finger_delta_scale"]  = torch.tensor(self.grasp_adr.get_param("finger", "delta_scale"),           device=self.device)

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

        in_or_past_lift = (self.episode_length_buf >= LIFT_START_STEP)
        lifted  = self.object_pos[:, 2] > (self.object_init_pos[:, 2] + self.cfg.lift_success_height)
        grasped = (self.num_contacts_buf >= MIN_CONTACTS_FOR_SUCCESS)
        success_now = in_or_past_lift & lifted & grasped
        self.success_flag.copy_(success_now)
        self.episode_success_buf |= success_now

        self._success_hold_count = torch.where(
            success_now,
            self._success_hold_count + 1,
            torch.zeros_like(self._success_hold_count),
        )
        success_held = self._success_hold_count >= self.cfg.success_hold_steps

        terminated = out_x | out_y | fallen | tipped | success_held
        truncated  = self.episode_length_buf >= self.max_episode_length - 1

        self.extras["object_z"] = self.object_pos[:, 2].mean()

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

        n = len(env_ids)
        started_n = int(had_started.sum().item())

        # ---- episode 성공 집계 ----
        self._total_episodes += started_n
        if started_n > 0:
            self._successful_episodes += int((self.episode_success_buf[env_ids] & had_started).sum().item())

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

            bead_count = int(round(self._bead_mass_normalized[env_id].item() * self.cfg.bead_count_max))

            # curl joint indices per finger (rj_dg_X_2 = index 4*(f-1)+1 for f=1..4, pinky _1 = 16)
            # finger layout: [f1_1,f1_2,f1_3,f1_4, f2_1,..., f5_1,f5_2,f5_3,f5_4]
            def _curl_idx(finger: int) -> int:  # finger 0-indexed
                return finger * 4 + 1  # _2 joint

            self._eval_records.append({
                "bead_count": bead_count,
                "bead_mass": self._bead_mass_normalized[env_id].item(),
                "grip":      grip_at_lift,
                "grasp_steps": grasp_count,
                "grasp_action_mean": grasp_mean.mean().item(),
                "grasp_action_std":  grasp_std.mean().item(),
                "grasp_action_min":  self._eval_grasp_action_min[env_id].mean().item(),
                "grasp_action_max":  self._eval_grasp_action_max[env_id].mean().item(),
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

        self.episode_success_buf[env_ids] = False

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
        self.object_init_pos[env_ids] = obj_pos_local

        # ---- 4. FABRICS pregrasp rollout ----
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

        # cache lookup
        xi = ((obj_x - self._cache_xs[0]) / (self._cache_xs[1] - self._cache_xs[0])).round().long().clamp(0, self._cache_n - 1)
        yi = ((obj_y - self._cache_ys[0]) / (self._cache_ys[1] - self._cache_ys[0])).round().long().clamp(0, self._cache_n - 1)
        q_pregrasp = self.fabric_q[env_ids].clone()
        q_pregrasp[:, :NUM_ARM_DOF] = self._cache_q_arm[xi, yi]

        self.fabric_q[env_ids] = q_pregrasp
        self.fabric_qd[env_ids].zero_()
        self.fabric_qdd[env_ids].zero_()

        # hand는 APPROACH_POSE로 초기화 (80% 개방형 자세에서 시작)
        approach_hand = self.hand_approach_pose.unsqueeze(0).expand(n, -1)
        self.fabric_q[env_ids, NUM_ARM_DOF:] = approach_hand
        self.fabric_qd[env_ids, NUM_ARM_DOF:].zero_()

        # ---- 5. pregrasp 버퍼 저장 ----
        self.pregrasp_arm_pos_buf[env_ids] = q_pregrasp[:, :NUM_ARM_DOF]
        self.palm_pose_targets[env_ids]    = pregrasp_palm_pose
        self.pregrasp_palm_pose_buf[env_ids] = pregrasp_palm_pose

        self.open_tesollo_fabric.default_config[env_ids, :NUM_ARM_DOF] = q_pregrasp[:, :NUM_ARM_DOF]
        self.lift_finger_pos_buf[env_ids] = approach_hand

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

        # ---- 7b. Bead 스폰 ----
        bead_count = torch.randint(
            self.cfg.bead_count_min, self.cfg.bead_count_max + 1,
            (n,), device=self.device
        )

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
        self._bead_mass_normalized[env_ids] = bead_count.float() / self.cfg.bead_count_max

        # ---- 8. 버퍼 리셋 ----
        self.hand_joint_targets[env_ids] = approach_hand
        self.contact_force_raw[env_ids] = 0.0
        self.binary_contact_buf[env_ids] = False
        self.num_contacts_buf[env_ids]   = 0
        self.distal_contact_force_raw[env_ids] = 0.0
        self.distal_binary_contact_buf[env_ids] = False
        self.middle_contact_force_raw[env_ids] = 0.0
        self.middle_binary_contact_buf[env_ids] = False
        self.success_flag[env_ids] = False
        self._success_hold_count[env_ids] = 0
        self._prev_avg_force_buf[env_ids] = 0.0       # force-smooth 초기화
        self._force_smooth_ready[env_ids] = False
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
        self._eval_lift_snapshot_valid[env_ids] = False

        # actions 리셋: delta=0 = 현재 자세 유지
        self.actions[env_ids, :6] = 0.0
        self.actions[env_ids, 6:] = 0.0
        self.prev_actions[env_ids, :6] = 0.0
        self.prev_actions[env_ids, 6:] = 0.0
        self.lift_palm_start_pose_buf[env_ids] = self.pregrasp_palm_pose_buf[env_ids]
