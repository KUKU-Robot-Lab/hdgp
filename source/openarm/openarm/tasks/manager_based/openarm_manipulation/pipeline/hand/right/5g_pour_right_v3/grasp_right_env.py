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

"""환경 클래스: 5g_grasp_right_v7

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
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

from .grasp_right_env_cfg import GraspRightEnvCfg
from .grasp_right_constants import (
    NUM_ARM_DOF,
    NUM_HAND_DOF,
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
from .grasp_right_preset import (
    BEAD_SPAWN_POS_SOURCE_CUP_B,
    BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ,
    LEFT_ARM_REST_JOINT_POS,
    LEFT_TARGET_CUP_ATTACH_FRAME_NAME,
    LEFT_TARGET_CUP_ATTACH_POS_B,
    LEFT_TARGET_CUP_ATTACH_QUAT_WXYZ_B,
    RIGHT_ACTUATED_JOINT_NAMES,
    HAND_APPROACH_POSE,
    HAND_GRASP_POSE,
    OBJECT_GOAL_POS,
)
from .grasp_right_utils import scale, to_torch




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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5)
        x = torch.clamp(x, -5.0, 5.0)
        x = torch.nn.functional.elu(torch.nn.functional.linear(x, self.actor_l1_w, self.actor_l1_b))
        x = torch.nn.functional.elu(torch.nn.functional.linear(x, self.actor_l2_w, self.actor_l2_b))
        x = torch.nn.functional.elu(torch.nn.functional.linear(x, self.actor_l3_w, self.actor_l3_b))
        return torch.nn.functional.linear(x, self.mu_w, self.mu_b)


class GraspRightEnv(DirectRLEnv):
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

        # pregrasp palm pose 버퍼 (에피소드별 delta action 기준점)
        self.pregrasp_palm_pose_buf = torch.zeros(self.num_envs, 6, device=self.device)

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
        self._grasp_cup_height_init = torch.zeros(self.num_envs, device=self.device)
        self.palm_center_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.fingertip_pos   = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.distal4_pos     = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.actions         = torch.zeros(self.num_envs, cfg.num_actions, device=self.device)
        self.prev_actions    = torch.full((self.num_envs, cfg.num_actions), 0.0, device=self.device)

        # ----------------------------------------------------------------
        # Pregrasp / Lift 버퍼 (reset에서 계산)
        # ----------------------------------------------------------------
        self.pregrasp_arm_pos_buf  = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        self.grasp_hold_hand_pos_buf = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)

        # ----------------------------------------------------------------
        # Hand joint targets (per-finger lerp 결과)
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

        self._left_target_cup_attach_pos_b = to_torch(self.cfg.left_target_cup_attach_pos_b, device=self.device)
        self._left_target_cup_attach_quat_b = to_torch(self.cfg.left_target_cup_attach_quat_wxyz_b, device=self.device)
        self._left_target_cup_body_id, self._left_target_cup_attach_pos_b = self._resolve_attachment_body(
            self.cfg.left_target_cup_attach_frame_name,
            self._left_target_cup_attach_pos_b,
        )
        self._right_source_cup_attach_pos_b = to_torch(self.cfg.right_source_cup_attach_pos_b, device=self.device)
        self._right_source_cup_attach_quat_b = to_torch(self.cfg.right_source_cup_attach_quat_wxyz_b, device=self.device)
        self._right_source_cup_body_id, self._right_source_cup_attach_pos_b = self._resolve_attachment_body(
            self.cfg.right_source_cup_attach_frame_name,
            self._right_source_cup_attach_pos_b,
        )
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
        self._mouth_z_clearance = torch.zeros(self.num_envs, device=self.device)
        self._source_up_dot_world = torch.zeros(self.num_envs, device=self.device)
        self._directional_tilt_cos = torch.zeros(self.num_envs, device=self.device)
        self._mouth_alignment_cos = torch.zeros(self.num_envs, device=self.device)
        self._bead_in_target = torch.zeros(self.num_envs, self.num_beads, dtype=torch.bool, device=self.device)
        self._bead_in_source = torch.zeros(self.num_envs, self.num_beads, dtype=torch.bool, device=self.device)
        self._bead_crossed_target_mouth = torch.zeros(
            self.num_envs, self.num_beads, dtype=torch.bool, device=self.device
        )
        self._prev_bead_target_local_z = torch.full(
            (self.num_envs, self.num_beads), 10.0, device=self.device
        )
        self._bead_cross_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._bead_cross_fraction = torch.zeros(self.num_envs, device=self.device)
        self._bead_in_target_fraction = torch.zeros(self.num_envs, device=self.device)
        self._bead_in_source_fraction = torch.zeros(self.num_envs, device=self.device)
        self._bead_centroid_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._spill_ratio = torch.zeros(self.num_envs, device=self.device)
        self._pre_pour_ready_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._no_tip_force_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._world_up = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)

        self._warmstart_collect_mode = False
        self._warmstart_policy = None
        self._warmstart_cache_count = 0
        cache_size = max(int(self.cfg.warmstart_cache_size), 1)
        self._warmstart_arm_pos = torch.zeros(cache_size, NUM_ARM_DOF, device=self.device)
        self._warmstart_hand_pos = torch.zeros(cache_size, NUM_HAND_DOF, device=self.device)
        self._warmstart_palm_pose = torch.zeros(cache_size, 6, device=self.device)
        self._warmstart_cup_pose = torch.zeros(cache_size, 7, device=self.device)
        # GUI target visualization: source pour point (red) + target opening (blue)
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

        self._tip_sensors = []

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
        self.palm_pose_targets = torch.zeros(self.num_envs, 6, device=self.device)
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
        self.contact_force_xyz_raw.zero_()
        self.contact_force_raw.zero_()
        self.binary_contact_buf.zero_()
        self.num_contacts_buf.zero_()
        self.distal_contact_force_raw.zero_()
        self.distal_binary_contact_buf.zero_()
        self.middle_contact_force_raw.zero_()
        self.middle_binary_contact_buf.zero_()

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

        bead_env_z = bead_pos_w[..., 2] - self.scene.env_origins[:, 2].unsqueeze(1)
        bead_spilled = (
            (~self._bead_in_target)
            & (~self._bead_in_source)
            & (bead_env_z < 0.230)
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

        # ---- Pour phase: Fabrics arm 제어 ----
        # Delta action: action=0 → pregrasp 유지, action=±1 → pregrasp ± delta
        # 절대 workspace(palm_mins/maxs)로 클램프하여 안전 영역 보장

        # 에피소드 시작 직후 N스텝: palm action=0 강제.
        if self.cfg.episode_hold_steps > 0:
            hold_mask = (self.episode_length_buf < self.cfg.episode_hold_steps).unsqueeze(1)
            palm_action = torch.where(hold_mask, torch.zeros_like(palm_action), palm_action)

        delta = scale(palm_action, self.delta_mins, self.delta_maxs)   # (N, 6)
        palm_pose = self.pregrasp_palm_pose_buf + delta
        palm_pose = torch.max(torch.min(palm_pose, self.palm_maxs), self.palm_mins)
        self.palm_pose_targets.copy_(palm_pose)
        self.hand_pca_targets.zero_()

        # 현재 fabric_q를 default_config로 덮어써 null-space 당김을 제거한다.
        self.open_tesollo_fabric.default_config.copy_(self.fabric_q.detach())

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

        # ---- 오른손은 grasp pose 고정 ----
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

        left_cup_pose = self._compute_attached_root_pose(
            self._left_target_cup_body_id,
            self._left_target_cup_attach_pos_b,
            self._left_target_cup_attach_quat_b,
        )
        right_cup_pose = self._compute_attached_root_pose(
            self._right_source_cup_body_id,
            self._right_source_cup_attach_pos_b,
            self._right_source_cup_attach_quat_b,
        )
        zero_cup_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self.cup.write_root_pose_to_sim(right_cup_pose)
        self.cup.write_root_velocity_to_sim(zero_cup_vel)
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
        self._mouth_z_clearance = self._source_pour_point_w[:, 2] - self._target_opening_w[:, 2]
        self._source_up_dot_world = self._source_up_axis_w[:, 2].clamp(-1.0, 1.0)

        # Directional tilt: source cup → target cup 방향(XY)으로 기울도록 유도
        # cup local frame 기반이 아닌 타겟 방향 기반으로 변경
        # (컵 그립 방향에 무관하게 항상 타겟 쪽으로 기울어야 함)
        _tilt_cos = math.cos(math.radians(self.cfg.target_pour_tilt_deg))
        _tilt_sin = math.sin(math.radians(self.cfg.target_pour_tilt_deg))
        _mouth_delta_xy = self._mouth_delta[:, :2]   # (N, 2): target - source XY
        _tilt_dir_xy = _mouth_delta_xy / (_mouth_delta_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        _tilt_dir_3d = torch.cat([_tilt_dir_xy, torch.zeros(n, 1, device=self.device)], dim=-1)
        _ref_up = _tilt_cos * self._world_up.expand(n, -1) + _tilt_sin * _tilt_dir_3d
        _ref_up = _ref_up / _ref_up.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self._directional_tilt_cos = (self._source_up_axis_w * _ref_up).sum(dim=-1).clamp(-1.0, 1.0)

        # Mouth alignment: pour axis → target 방향
        _mouth_dir = self._mouth_delta / self._mouth_distance.unsqueeze(1).clamp(min=1e-6)
        self._mouth_alignment_cos = (self._source_pour_axis_w * _mouth_dir).sum(dim=-1).clamp(-1.0, 1.0)

        # Bead flags & spill
        self._compute_bead_flags()

        # GUI visualization: red = source pour point, blue = target opening
        _all_pts = torch.cat([self._source_pour_point_w, self._target_opening_w], dim=0)
        _marker_idx = torch.zeros(2 * n, dtype=torch.long, device=self.device)
        _marker_idx[n:] = 1
        self._vis_markers.visualize(translations=_all_pts, marker_indices=_marker_idx)

        # 접촉력 업데이트
        self._update_contact_forces()

    # ------------------------------------------------------------------
    # Observations: Actor 102D | Critic 149D
    # ------------------------------------------------------------------
    def _get_legacy_warmstart_policy_obs(self) -> torch.Tensor:
        """Build the original 106D actor observation for the warmstart checkpoint.

        The warmstart policy was trained on the pre-pouring grasp task and must
        keep receiving the legacy actor observation layout even though the main
        training actor observation has changed.
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

        warmstart_obs = torch.cat([
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

        if warmstart_obs.shape[1] != 106:
            raise RuntimeError(f"[warmstart] Legacy obs dim mismatch: {warmstart_obs.shape[1]} != 106")

        return warmstart_obs

    def _get_observations(self) -> dict:
        # ==== 공통 clean state (critic용, 물리 정확값) ====
        arm_joint_pos_clean = self.robot.data.joint_pos[:, self.arm_dof_indices]
        arm_joint_vel_clean = self.robot.data.joint_vel[:, self.arm_dof_indices]
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
        bead_pos_clean = self._bead_centroid_w

        # ==== Actor obs용 noisy state (sim2real domain randomization) ====
        σ_qp = self.cfg.obs_noise_joint_pos
        σ_qv = self.cfg.obs_noise_joint_vel
        σ_bp = self.cfg.obs_noise_body_pos
        σ_cp = self.cfg.obs_noise_cup_pos

        arm_joint_pos = arm_joint_pos_clean + torch.randn_like(arm_joint_pos_clean) * σ_qp
        arm_joint_vel = arm_joint_vel_clean + torch.randn_like(arm_joint_vel_clean) * σ_qv
        palm_center_pos = palm_center_pos_clean + torch.randn_like(palm_center_pos_clean) * σ_bp
        right_cup_pos = right_cup_pos_clean + torch.randn_like(right_cup_pos_clean) * σ_cp
        left_cup_pos = left_cup_pos_clean + torch.randn_like(left_cup_pos_clean) * σ_cp
        source_pour_point = source_pour_point_clean + torch.randn_like(source_pour_point_clean) * σ_cp
        target_opening = target_opening_clean + torch.randn_like(target_opening_clean) * σ_cp

        right_cup_pos_rel_palm = right_cup_pos - palm_center_pos
        left_cup_pos_rel_palm = left_cup_pos - palm_center_pos
        pour_point_to_opening = target_opening - source_pour_point

        last_actions = self.actions

        # transport_summary (5D): pour 기하학 핵심 정보
        transport_summary = torch.stack([
            self._mouth_distance,
            self._mouth_xy_distance,
            self._mouth_z_clearance,
            self._source_up_dot_world,
            self._directional_tilt_cos,
        ], dim=-1)   # (N, 5)

        actor_obs = torch.cat([
            arm_joint_pos,              # 7
            arm_joint_vel,              # 7
            right_cup_pos_rel_palm,     # 3
            right_cup_quat_clean,       # 4
            left_cup_pos_rel_palm,      # 3
            left_cup_quat_clean,        # 4
            pour_point_to_opening,      # 3
            source_pour_axis_clean,     # 3
            source_up_axis_clean,       # 3
            target_up_axis_clean,       # 3
            transport_summary,          # 5
            last_actions,               # 6
        ], dim=-1)   # 51D

        if actor_obs.shape[1] != NUM_OBSERVATIONS:
            raise RuntimeError(
                f"[pour_v1] Actor obs dim mismatch: {actor_obs.shape[1]} != {NUM_OBSERVATIONS}"
            )

        # ==== Critic extra obs (30D) ====

        bead_pos_rel_source_cup = quat_apply_inverse(
            right_cup_quat_clean,
            bead_pos_clean - right_cup_pos_clean,
        )
        bead_pos_rel_target_cup = quat_apply_inverse(
            left_cup_quat_clean,
            bead_pos_clean - left_cup_pos_clean,
        )

        # critic actor_obs_clean (101D) — clean state 재조합
        actor_obs_clean = torch.cat([
            arm_joint_pos_clean,
            arm_joint_vel_clean,
            right_cup_pos_clean - palm_center_pos_clean,
            right_cup_quat_clean,
            left_cup_pos_clean - palm_center_pos_clean,
            left_cup_quat_clean,
            target_opening_clean - source_pour_point_clean,
            source_pour_axis_clean,
            source_up_axis_clean,
            target_up_axis_clean,
            torch.stack([
                self._mouth_distance,
                self._mouth_xy_distance,
                self._mouth_z_clearance,
                self._source_up_dot_world,
                self._directional_tilt_cos,
            ], dim=-1),
            last_actions,
        ], dim=-1)   # 51D

        critic_obs = torch.cat([
            actor_obs_clean,                                    # 51
            left_arm_joint_pos_clean,                          # 9
            left_arm_joint_vel_clean,                          # 9
            bead_pos_rel_source_cup,                           # 3
            bead_pos_rel_target_cup,                           # 3
            self._mouth_distance.unsqueeze(1),                 # 1
            self._mouth_xy_distance.unsqueeze(1),              # 1
            self._mouth_z_clearance.unsqueeze(1),              # 1
            self._source_up_dot_world.unsqueeze(1),            # 1
            self._directional_tilt_cos.unsqueeze(1),           # 1
            self._mouth_alignment_cos.unsqueeze(1),            # 1
            self._bead_cross_fraction.unsqueeze(1),            # 1
        ], dim=-1)   # 81D

        if critic_obs.shape[1] != NUM_CRITIC_OBSERVATIONS:
            raise RuntimeError(
                f"[pour_v1] Critic obs dim mismatch: {critic_obs.shape[1]} != {NUM_CRITIC_OBSERVATIONS}"
            )

        return {"policy": actor_obs, "critic": critic_obs}

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        transport_reward = 1.0 - torch.tanh(self.cfg.reward_transport_scale * self._mouth_xy_distance)
        target_clearance = 0.5 * (self.cfg.success_z_clearance_min + self.cfg.success_z_clearance_max)
        clearance_error = torch.abs(self._mouth_z_clearance - target_clearance)
        clearance_reward = 1.0 - torch.tanh(self.cfg.reward_clearance_scale * clearance_error)

        target_tilt_cos = math.cos(math.radians(self.cfg.target_pour_tilt_deg))
        tilt_error = torch.abs(self._source_up_dot_world - target_tilt_cos)
        tilt_reward = 1.0 - torch.tanh(self.cfg.reward_tilt_scale * tilt_error)
        mouth_alignment_reward = ((self._mouth_alignment_cos + 1.0) * 0.5).pow(self.cfg.reward_mouth_align_scale)
        directional_tilt_reward = ((self._directional_tilt_cos + 1.0) * 0.5).clamp(0.0, 1.0)

        near_target_gate = transport_reward.detach()  # clearance gate 제거: 기울이면 clearance 깨지는 구조 방지
        # tilt_reward(수직 기울기)와 directional_tilt_reward(방향) 결합
        # - tilt_reward: upright=0, 90°=1 (강한 기울기 유도)
        # - directional_tilt_reward: 타겟 방향으로 기울기 (upright=0.5 floor → 방향 보정)
        pour_pose_reward = near_target_gate * (0.5 * tilt_reward + 0.5 * directional_tilt_reward)
        align_reward = near_target_gate * mouth_alignment_reward

        bead_target_reward = self._bead_cross_fraction
        success_reward = self.success_flag.float()
        spill_penalty = self._spill_ratio
        action_rate_penalty = torch.sum((self.actions - self.prev_actions) ** 2, dim=-1)

        total = (
            self.cfg.reward_transport_weight * transport_reward
            + self.cfg.reward_clearance_weight * clearance_reward
            + self.cfg.reward_pour_alignment_weight * align_reward
            + self.cfg.reward_tilt_weight * pour_pose_reward
            + self.cfg.reward_bead_target_weight * bead_target_reward
            + self.cfg.reward_success_weight * success_reward
            - self.cfg.penalty_spill_weight * spill_penalty
            - self.cfg.penalty_action_rate_weight * action_rate_penalty
        )

        self.extras["reward_transport"] = transport_reward.mean()
        self.extras["reward_clearance"] = clearance_reward.mean()
        self.extras["reward_tilt"] = tilt_reward.mean()
        self.extras["reward_directional_tilt"] = directional_tilt_reward.mean()
        self.extras["reward_mouth_alignment"] = mouth_alignment_reward.mean()
        self.extras["penalty_action_rate"] = action_rate_penalty.mean()

        self.extras["mouth_distance"]        = self._mouth_distance.mean()
        self.extras["mouth_xy_distance"]     = self._mouth_xy_distance.mean()
        self.extras["mouth_z_clearance"]     = self._mouth_z_clearance.mean()
        self.extras["source_up_dot"]         = self._source_up_dot_world.mean()
        self.extras["directional_tilt_cos"]  = self._directional_tilt_cos.mean()
        self.extras["mouth_alignment_cos"]   = self._mouth_alignment_cos.mean()
        self.extras["bead_in_source_rate"]   = self._bead_in_source_fraction.mean()
        self.extras["bead_in_target_rate"]   = self._bead_in_target_fraction.mean()
        self.extras["bead_cross_fraction"]   = self._bead_cross_fraction.mean()
        self.extras["bead_cross_count"]      = self._bead_cross_count.float().mean()
        self.extras["spill_ratio"]           = self._spill_ratio.mean()
        self.extras["success_rate"]          = self.success_flag.float().mean()

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

        self.success_flag.copy_(self._bead_cross_count >= self.cfg.success_bead_cross_count)
        self.episode_success_buf |= self.success_flag   # 에피소드 중 한 번이라도 성공 시 True

        terminated = out_x | out_y | fallen | self.success_flag
        truncated  = self.episode_length_buf >= self.max_episode_length - 1

        self.extras["object_z"] = self.object_pos[:, 2].mean()

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
        self.episode_success_buf[env_ids] = False

        # ---- 1. 로봇 관절 상태 리셋 ----
        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)
        fixed_hand = self.hand_grasp_pose.unsqueeze(0).expand(n, -1)
        full_pos[:, self.arm_dof_indices] = self.robot_start_joint_pos[0, :NUM_ARM_DOF]
        full_pos[:, self.hand_dof_indices] = fixed_hand
        full_pos[:, self.left_arm_dof_indices] = self.left_arm_zero_pos[0]
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        # ---- 2. Fabrics 상태 리셋 ----
        self.fabric_q[env_ids] = self.robot_start_joint_pos[env_ids]
        self.fabric_q[env_ids, NUM_ARM_DOF:] = fixed_hand
        self.fabric_qd[env_ids].zero_()
        self.fabric_qdd[env_ids].zero_()

        # ---- 3. action 기준점: 고정된 pour-start palm pose ----
        pregrasp_palm_pose = torch.zeros(n, 6, device=self.device)
        pregrasp_palm_pose[:, 0] = self.cfg.object_spawn_x_center + self.cfg.pregrasp_offset_x
        pregrasp_palm_pose[:, 1] = self.cfg.object_spawn_y_center + self.cfg.pregrasp_offset_y
        pregrasp_palm_pose[:, 2] = self.cfg.object_spawn_z + self.cfg.pregrasp_offset_z
        pregrasp_palm_pose[:, 3] = math.radians(90.0)
        pregrasp_palm_pose[:, 4] = math.radians(0.0)
        pregrasp_palm_pose[:, 5] = math.radians(90.0)
        pregrasp_palm_pose = torch.max(
            torch.min(pregrasp_palm_pose, self.palm_maxs.unsqueeze(0)),
            self.palm_mins.unsqueeze(0),
        )

        # ---- 4. pregrasp / prelift 버퍼 저장 ----
        self.pregrasp_arm_pos_buf[env_ids] = self.robot_start_joint_pos[env_ids, :NUM_ARM_DOF]

        # palm_pose_targets를 pregrasp로 동기화 (첫 Fabrics 스텝 타겟 일관성)
        self.palm_pose_targets[env_ids] = pregrasp_palm_pose

        # delta action 기준점: action=0 → pregrasp 위치 유지
        self.pregrasp_palm_pose_buf[env_ids] = pregrasp_palm_pose
        self._grasp_rel_palm_to_cup_init[env_ids].zero_()
        self._grasp_cup_height_init[env_ids] = self.cfg.object_spawn_z

        self.open_tesollo_fabric.default_config[env_ids, :NUM_ARM_DOF] = self.robot_start_joint_pos[env_ids, :NUM_ARM_DOF]

        self.grasp_hold_hand_pos_buf[env_ids] = fixed_hand

        # ---- 5. source / target cup pose 초기화 ----
        zero_vel = torch.zeros(n, 6, device=self.device)
        right_cup_pose = self._compute_attached_root_pose(
            self._right_source_cup_body_id,
            self._right_source_cup_attach_pos_b,
            self._right_source_cup_attach_quat_b,
            env_ids=env_ids,
        )
        self.cup.write_root_pose_to_sim(right_cup_pose, env_ids=env_ids)
        self.cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        left_cup_pose = self._compute_attached_root_pose(
            self._left_target_cup_body_id,
            self._left_target_cup_attach_pos_b,
            self._left_target_cup_attach_quat_b,
            env_ids=env_ids,
        )
        self.left_target_cup.write_root_pose_to_sim(left_cup_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        bead_state = self._sample_bead_states_inside_cup(right_cup_pose)
        self.beads.write_object_state_to_sim(bead_state, env_ids=env_ids)
        self.object_init_pos[env_ids] = right_cup_pose[:, :3] - self.scene.env_origins[env_ids]

        # ---- 6. 버퍼 리셋 ----
        self.hand_joint_targets[env_ids] = fixed_hand
        self.contact_force_raw[env_ids].zero_()
        self.binary_contact_buf[env_ids] = False
        self.num_contacts_buf[env_ids]   = 0
        self.distal_contact_force_raw[env_ids].zero_()
        self.distal_binary_contact_buf[env_ids] = False
        self.middle_contact_force_raw[env_ids].zero_()
        self.middle_binary_contact_buf[env_ids] = False
        self._bead_in_target[env_ids] = False
        self._bead_in_source[env_ids] = False
        self._bead_crossed_target_mouth[env_ids] = False
        self._prev_bead_target_local_z[env_ids].fill_(10.0)
        self._bead_cross_count[env_ids] = 0
        self._bead_cross_fraction[env_ids] = 0.0
        self._bead_in_target_fraction[env_ids] = 0.0
        self._bead_in_source_fraction[env_ids] = 0.0
        self._bead_centroid_w[env_ids].zero_()
        self._spill_ratio[env_ids] = 0.0
        self._no_tip_force_steps[env_ids] = 0
        self.success_flag[env_ids] = False
        self._pre_pour_ready_steps[env_ids] = 0

        # actions 리셋: delta action 방식 → action=0 = pregrasp 위치
        # (역스케일 불필요: scale(0, delta_mins, delta_maxs) = delta=0 → pregrasp 유지)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0


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
            "openarm_right_hand": [
                ("openarm_right_hand", (0.0, 0.0, 0.0)),
                ("openarm_right_hand_tcp", (0.0, 0.0, -0.08)),
                ("rl_dg_ee", (0.0, 0.0, -0.08)),
            ],
            "rl_dg_ee": [
                ("rl_dg_ee", (0.0, 0.0, 0.0)),
                ("openarm_right_hand_tcp", (0.0, 0.0, -0.08)),
                ("openarm_right_hand", (0.0, 0.0, 0.0)),
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
            print(f"[5g_pour_right_v2] warmstart policy load failed: {exc}", flush=True)
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
                "[5g_pour_right_v2] warmstart cache is empty. "
                "The v7 checkpoint rollout did not produce any lift-success state, so this task cannot start "
                "from the requested play-like grasp state."
            )

        print(
            f"[5g_pour_right_v2] collected {self._warmstart_cache_count} warmstart success states.",
            flush=True,
        )

    def _maybe_store_warmstart_successes(self) -> None:
        if not self._warmstart_collect_mode:
            return
        if self._warmstart_cache_count >= self._warmstart_arm_pos.shape[0]:
            return

        lifted = self.object_pos[:, 2] > (self.object_init_pos[:, 2] + self.cfg.lift_success_height)
        grasped = self.num_contacts_buf >= MIN_CONTACTS_FOR_SUCCESS
        upright = self._source_up_axis_w[:, 2] > 0.7
        warmstart_success = lifted & grasped & upright

        success_env_ids = warmstart_success.nonzero(as_tuple=False).squeeze(-1)
        if success_env_ids.numel() == 0:
            return

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

        # Scripted pre-lift: warmstart cup은 z≈0.34m로 낮아 왼팔 구조체와 충돌 가능.
        # pregrasp Z 기준점을 +0.25m 올려서 episode_hold_steps 동안 팔이 먼저 위로 올라오도록.
        # hold 완료 후 cup z≈0.59m → z_clearance≈0.145m > 0.10m → rho gate 즉시 충족 가능.
        lifted_palm_pose = palm_pose.clone()
        lifted_palm_pose[:, 2] = torch.clamp(
            palm_pose[:, 2] + 0.25,
            self.palm_mins[2],
            self.palm_maxs[2],
        )
        self.pregrasp_palm_pose_buf[env_ids] = lifted_palm_pose
        self.palm_pose_targets[env_ids] = lifted_palm_pose
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

        left_cup_pose = self._compute_attached_root_pose(
            self._left_target_cup_body_id,
            self._left_target_cup_attach_pos_b,
            self._left_target_cup_attach_quat_b,
            env_ids=env_ids,
        )
        self.left_target_cup.write_root_pose_to_sim(left_cup_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        bead_state = self._sample_bead_states_inside_cup(cup_pose_world)
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
        self._bead_crossed_target_mouth[env_ids] = False
        self._prev_bead_target_local_z[env_ids].fill_(10.0)
        self._bead_cross_count[env_ids] = 0
        self._bead_cross_fraction[env_ids] = 0.0
        self._bead_in_target_fraction[env_ids] = 0.0
        self._bead_in_source_fraction[env_ids] = 0.0
        self._bead_centroid_w[env_ids].zero_()
        self._spill_ratio[env_ids] = 0.0
        self._no_tip_force_steps[env_ids] = 0
        self.success_flag[env_ids] = False

        self.actions[env_ids, :6] = 0.0
        self.actions[env_ids, 6:] = 1.0
        self.prev_actions[env_ids, :6] = 0.0
        self.prev_actions[env_ids, 6:] = 1.0
        self._pre_pour_ready_steps[env_ids] = 0
        self.success_flag[env_ids] = False
