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

"""환경 클래스: 5g_pour_right_v3

v3: Fluid Particle System (PhysX 5 PBD) + Cost-Benefit Tradeoff Reward + Warmstart v8

v2 대비 변경:
  - 액체 표현: Rigid Bead → PBD Fluid Particle (200개, 5mm, 1g)
  - 리워드: 복합 shaping → R = w_A * A - w_T * T - w_E * E + 보조 shaping
  - Warmstart: v7 (106D) → v8 (107D, bead_mass_normalized=1.0 고정)

Action: 11D
  [0:6]  palm pose (x,y,z,ez,ey,ex), 정규화 [-1,1] → Fabrics IK
  [6:11] per-finger lerp (freeze_grasp=True → 항상 grasp_hold 유지)

Episode (6s @ 60Hz = 360 steps):
  Pour phase (step 0~359): Fabrics arm policy + frozen hand
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from collections.abc import Sequence

import torch
import torch.nn as nn

# Fabrics 경로 설정
for _parent in Path(__file__).resolve().parents:
    if _parent.name == "source":
        _vendored = _parent / "FABRICS" / "src"
        if _vendored.exists():
            _v = str(_vendored)
            if _v not in sys.path:
                sys.path.insert(0, _v)
        break

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
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
    NUM_LEGACY_WARMSTART_OBS,
    EPISODE_STEPS,
    CONTACT_FORCE_THRESHOLD,
    CONTACT_FORCE_MAX,
    MIN_CONTACTS_FOR_SUCCESS,
    PREGRASP_FABRICS_STEPS,
    CUP_RADIUS_APPROX,
    PARTICLES_PER_ENV,
    PARTICLE_DIAMETER,
    PARTICLE_SYSTEM_PATH,
    PARTICLE_SET_PATTERN,
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
        self.obs_dim = int(self.obs_mean.numel())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5)
        x = torch.clamp(x, -5.0, 5.0)
        x = torch.nn.functional.elu(torch.nn.functional.linear(x, self.actor_l1_w, self.actor_l1_b))
        x = torch.nn.functional.elu(torch.nn.functional.linear(x, self.actor_l2_w, self.actor_l2_b))
        x = torch.nn.functional.elu(torch.nn.functional.linear(x, self.actor_l3_w, self.actor_l3_b))
        return torch.nn.functional.linear(x, self.mu_w, self.mu_b)


class GraspRightEnv(DirectRLEnv):
    """OpenArm+Teosllo 오른손 물붓기 환경 v3.

    Warmstart(v8 grasp policy)로 컵이 이미 파지·들린 상태로 시작.
    Policy는 transport → tilt → pour 모션을 학습.
    액체는 PhysX 5 PBD Fluid Particle (200개)로 시뮬레이션.
    Reward: Cost-Benefit Tradeoff R = w_A*A - w_T*T - w_E*E + 보조 shaping

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
        # Palm pose workspace
        # ----------------------------------------------------------------
        self.palm_mins = to_torch(PALM_POSE_MINS_FUNC(cfg.max_pose_angle), device=self.device)
        self.palm_maxs = to_torch(PALM_POSE_MAXS_FUNC(cfg.max_pose_angle), device=self.device)

        _delta_rad = math.radians(cfg.palm_delta_rot_deg)
        self.delta_mins = to_torch([
            -cfg.palm_delta_xyz, -cfg.palm_delta_xyz, -cfg.palm_delta_xyz,
            -_delta_rad, -_delta_rad, -_delta_rad,
        ], device=self.device)
        self.delta_maxs = to_torch([
            cfg.palm_delta_xyz, cfg.palm_delta_xyz, cfg.palm_delta_xyz,
            _delta_rad, _delta_rad, _delta_rad,
        ], device=self.device)

        self.pregrasp_palm_pose_buf = torch.zeros(self.num_envs, 6, device=self.device)

        # ----------------------------------------------------------------
        # Hand poses
        # ----------------------------------------------------------------
        self.hand_open_pose  = to_torch(HAND_APPROACH_POSE, device=self.device)
        self.hand_grasp_pose = to_torch(HAND_GRASP_POSE,    device=self.device)

        arm_start  = to_torch(ARM_START_POSE,    device=self.device)
        hand_start = to_torch(HAND_APPROACH_POSE, device=self.device)
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
        # 목표 위치 / 오프셋
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
        self._grasp_rel_palm_to_cup_init = torch.zeros(self.num_envs, 3, device=self.device)
        self._grasp_cup_height_init = torch.zeros(self.num_envs, device=self.device)
        self.palm_center_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.fingertip_pos   = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.distal4_pos     = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.actions         = torch.zeros(self.num_envs, cfg.num_actions, device=self.device)
        self.prev_actions    = torch.full((self.num_envs, cfg.num_actions), 0.0, device=self.device)

        # ----------------------------------------------------------------
        # Pregrasp / Grasp hold 버퍼
        # ----------------------------------------------------------------
        self.pregrasp_arm_pos_buf    = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        self.grasp_hold_hand_pos_buf = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        self.hand_joint_targets      = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)

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
        # Pour 기하학 버퍼
        # ----------------------------------------------------------------
        self._approach_dir_buf       = torch.zeros(self.num_envs, 3, device=self.device)
        self._source_pour_point_w    = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_opening_w       = torch.zeros(self.num_envs, 3, device=self.device)
        self._source_pour_axis_w     = torch.zeros(self.num_envs, 3, device=self.device)
        self._source_up_axis_w       = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_up_axis_w       = torch.zeros(self.num_envs, 3, device=self.device)
        self._mouth_delta            = torch.zeros(self.num_envs, 3, device=self.device)
        self._mouth_distance         = torch.zeros(self.num_envs, device=self.device)
        self._mouth_xy_distance      = torch.zeros(self.num_envs, device=self.device)
        self._mouth_z_clearance      = torch.zeros(self.num_envs, device=self.device)
        self._source_up_dot_world    = torch.zeros(self.num_envs, device=self.device)
        self._directional_tilt_cos   = torch.zeros(self.num_envs, device=self.device)
        self._mouth_alignment_cos    = torch.zeros(self.num_envs, device=self.device)
        self._world_up               = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)

        # ----------------------------------------------------------------
        # Fluid particle 버퍼 (bead 버퍼 대체)
        # ----------------------------------------------------------------
        self._fluid_centroid_w         = torch.zeros(self.num_envs, 3, device=self.device)
        self._fluid_accuracy           = torch.zeros(self.num_envs, device=self.device)   # fraction in target
        self._fluid_in_target_fraction = torch.zeros(self.num_envs, device=self.device)
        self._fluid_in_source_fraction = torch.zeros(self.num_envs, device=self.device)
        self._spill_ratio              = torch.zeros(self.num_envs, device=self.device)
        self._pre_pour_ready_steps     = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._no_tip_force_steps       = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Particle view (lazy init after first physics step)
        self._particle_view = None
        self._num_particles_per_env: int = PARTICLES_PER_ENV

        # Particle grid offsets in source cup frame (P, 3)
        self._fluid_grid_offsets_b = self._make_fluid_grid_offsets()

        # ----------------------------------------------------------------
        # 성공 버퍼
        # ----------------------------------------------------------------
        self.success_flag        = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._total_episodes: int = 0
        self._successful_episodes: int = 0

        # ----------------------------------------------------------------
        # Cup 기준점 벡터
        # ----------------------------------------------------------------
        self._left_target_cup_attach_pos_b  = to_torch(self.cfg.left_target_cup_attach_pos_b, device=self.device)
        self._left_target_cup_attach_quat_b = to_torch(self.cfg.left_target_cup_attach_quat_wxyz_b, device=self.device)
        self._left_target_cup_body_id, self._left_target_cup_attach_pos_b = self._resolve_attachment_body(
            self.cfg.left_target_cup_attach_frame_name,
            self._left_target_cup_attach_pos_b,
        )
        self._source_cup_pour_point_pos_b = to_torch(self.cfg.source_cup_pour_point_pos_b, device=self.device)
        self._target_cup_opening_pos_b    = to_torch(self.cfg.target_cup_opening_pos_b, device=self.device)
        self._source_cup_pour_axis_b      = to_torch(self.cfg.source_cup_pour_axis_b, device=self.device)
        self._source_cup_up_axis_b        = to_torch(self.cfg.source_cup_up_axis_b, device=self.device)
        self._target_cup_up_axis_b        = to_torch(self.cfg.target_cup_up_axis_b, device=self.device)
        self._bead_spawn_pos_source_cup_b = to_torch(self.cfg.bead_spawn_pos_source_cup_b, device=self.device)

        # ----------------------------------------------------------------
        # Fabrics 초기화
        # ----------------------------------------------------------------
        self._setup_geometric_fabrics()

        cspace_default = self.open_tesollo_fabric.default_config.clone()
        cspace_default[:, NUM_ARM_DOF:] = self.hand_grasp_pose.unsqueeze(0).expand(self.num_envs, -1)
        self.open_tesollo_fabric.default_config.copy_(cspace_default)

        self.actions.zero_()

        # ----------------------------------------------------------------
        # Warmstart 버퍼
        # ----------------------------------------------------------------
        self._warmstart_collect_mode = False
        self._warmstart_policy = None
        self._warmstart_cache_count = 0
        cache_size = max(int(self.cfg.warmstart_cache_size), 1)
        self._warmstart_arm_pos  = torch.zeros(cache_size, NUM_ARM_DOF, device=self.device)
        self._warmstart_hand_pos = torch.zeros(cache_size, NUM_HAND_DOF, device=self.device)
        self._warmstart_palm_pose = torch.zeros(cache_size, 6, device=self.device)
        self._warmstart_cup_pose  = torch.zeros(cache_size, 7, device=self.device)

        # ----------------------------------------------------------------
        # 시각화 마커
        # ----------------------------------------------------------------
        self._vis_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/FiveGPourRightV3Markers",
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
    # Fluid particle grid 오프셋 생성 (cup local frame)
    # ------------------------------------------------------------------
    def _make_fluid_grid_offsets(self) -> torch.Tensor:
        """4×5×10 grid = 200 particles, cup local frame (단위: m).

        v2 bead spawn offset를 재사용하면 fluid가 컵 중상단에서 시작하므로,
        fluid는 컵 local 좌표에서 직접 바닥 근처 volume을 채운다.

        x: 4열, y: 5열은 컵 내경보다 작은 footprint
        z: 컵 바닥(-0.077m) 위 약 21mm ~ 59mm 범위
        """
        d = PARTICLE_DIAMETER
        z_base = -0.058
        positions = []
        for zi in range(10):
            for yi in range(5):
                for xi in range(4):
                    x = (xi - 1.5) * (0.55 * d)
                    y = (yi - 2.0) * (0.55 * d)
                    z = z_base + zi * (0.85 * d)
                    positions.append([x, y, z])
        return torch.tensor(positions, dtype=torch.float32, device=self.device)  # (200, 3)

    # ------------------------------------------------------------------
    # Scene 설정
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot          = Articulation(self.cfg.robot_cfg)
        self.cup            = RigidObject(self.cfg.cup_cfg)
        self.source_particle_cage = RigidObject(self.cfg.source_particle_cage_cfg)
        self.left_target_cup = RigidObject(self.cfg.left_target_cup_cfg)
        self.table          = RigidObject(self.cfg.table_cfg)

        self.scene.articulations["robot"]          = self.robot
        self.scene.rigid_objects["cup"]            = self.cup
        self.scene.rigid_objects["source_particle_cage"] = self.source_particle_cage
        self.scene.rigid_objects["left_target_cup"] = self.left_target_cup
        self.scene.rigid_objects["table"]          = self.table

        # ContactSensor (Actor: fingertip, Critic: distal/middle)
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

        # Fluid particle system 생성 (cloning 전에 env_0에 template 생성)
        self._create_fluid_particle_system()

        self.scene.clone_environments(copy_from_source=True)

        # cloning 후: 각 env에 고유한 particle_group 할당
        self._fix_particle_groups_after_clone()

    # ------------------------------------------------------------------
    # Fluid particle system 생성
    # ------------------------------------------------------------------
    def _create_fluid_particle_system(self) -> None:
        """PhysX 5 PBD Fluid Particle System과 env_0 template particle set을 생성."""
        try:
            import omni.usd as _omni_usd
            from omni.physx.scripts import particleUtils as _pu
            from pxr import Gf, Sdf, PhysxSchema, UsdGeom, UsdShade, Vt

            stage = _omni_usd.get_context().get_stage()

            # 1. Particle System 생성 (전 env 공유, 단일 PhysX solver)
            _pu.add_physx_particle_system(
                stage=stage,
                particle_system_path=PARTICLE_SYSTEM_PATH,
                contact_offset=self.cfg.particle_contact_offset,
                rest_offset=self.cfg.particle_fluid_rest_offset,
                particle_contact_offset=self.cfg.particle_contact_offset,
                solid_rest_offset=self.cfg.particle_solid_rest_offset,
                fluid_rest_offset=self.cfg.particle_fluid_rest_offset,
                solver_position_iterations=4,
                simulation_owner=Sdf.Path("/physicsScene"),
                global_self_collision_enabled=False,  # 서로 다른 group 간 충돌 없음
            )

            # Global self-collision 비활성화 (다른 env 입자 간 상호작용 제거)
            ps_prim = stage.GetPrimAtPath(PARTICLE_SYSTEM_PATH)
            if ps_prim.IsValid():
                _ps_api = PhysxSchema.PhysxParticleSystem.Get(stage, PARTICLE_SYSTEM_PATH)
                if _ps_api:
                    _ps_api.CreateGlobalSelfCollisionEnabledAttr().Set(False)
                _pu.add_physx_particle_anisotropy(
                    stage,
                    PARTICLE_SYSTEM_PATH,
                    enabled=True,
                    scale=5.0,
                    min=1.0,
                    max=2.0,
                )
                _pu.add_physx_particle_smoothing(
                    stage,
                    PARTICLE_SYSTEM_PATH,
                    enabled=True,
                    strength=0.8,
                )
                _pu.add_physx_particle_isosurface(
                    stage,
                    PARTICLE_SYSTEM_PATH,
                    enabled=True,
                    max_vertices=1_048_576,
                    max_triangles=2_097_152,
                    max_subgrids=4096,
                    grid_spacing=self.cfg.particle_fluid_rest_offset * 1.5,
                    surface_distance=self.cfg.particle_fluid_rest_offset * 1.2,
                    grid_filtering_passes="GSRS",
                    grid_smoothing_radius=self.cfg.particle_fluid_rest_offset * 2.0,
                    num_mesh_smoothing_passes=2,
                    num_mesh_normal_smoothing_passes=2,
                )

            # 2. PBD Particle Material 생성
            _pbd_mat_path = "/World/PBDFluidMaterial"
            _pu.add_pbd_particle_material(
                stage,
                _pbd_mat_path,
                friction=0.1,
                damping=0.0,
                viscosity=self.cfg.particle_viscosity,
                vorticity_confinement=0.0,
                surface_tension=self.cfg.particle_surface_tension,
                cohesion=self.cfg.particle_cohesion,
                adhesion=self.cfg.particle_adhesion,
                particle_friction_scale=0.1,
                adhesion_offset_scale=0.0,
                gravity_scale=1.0,
                lift=0.0,
                drag=0.0,
            )

            # Material → Particle System binding
            mat_api = UsdShade.MaterialBindingAPI.Apply(ps_prim)
            mat_prim = stage.GetPrimAtPath(_pbd_mat_path)
            if mat_prim.IsValid():
                mat = UsdShade.Material(mat_prim)
                mat_api.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")

            # 3. env_0 template particle set 생성 (z=-10: 지하에 숨김)
            N_P = PARTICLES_PER_ENV
            positions = [Gf.Vec3f(0.0, 0.0, -10.0)] * N_P
            velocities = [Gf.Vec3f(0.0, 0.0, 0.0)] * N_P
            widths = [self.cfg.particle_diameter] * N_P

            _pu.add_physx_particleset_points(
                stage=stage,
                path="/World/envs/env_0/FluidParticles",
                positions_list=positions,
                velocities_list=velocities,
                widths_list=widths,
                particle_system_path=PARTICLE_SYSTEM_PATH,
                self_collision=True,
                fluid=True,
                particle_group=0,
                particle_mass=self.cfg.particle_mass,
                density=0.0,
            )

            # 디버그 가시성을 위해 source fluid를 빨간색으로 고정
            fluid_points_prim = stage.GetPrimAtPath("/World/envs/env_0/FluidParticles")
            if fluid_points_prim.IsValid():
                fluid_points = UsdGeom.Points(fluid_points_prim)
                fluid_points.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.1, 0.1)]))

            print(
                f"[5g_pour_right_v3] Fluid particle system created: "
                f"{N_P} particles/env × {self.num_envs} envs",
                flush=True,
            )

        except Exception as exc:
            print(f"[5g_pour_right_v3] WARNING: fluid particle system creation failed: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Particle group 고유 ID 할당 (clone 후)
    # ------------------------------------------------------------------
    def _fix_particle_groups_after_clone(self) -> None:
        """각 env의 FluidParticles에 고유 particle_group 할당.

        동일 particle_group은 서로 충돌 → 다른 env 입자끼리 상호작용 제거.
        global_self_collision=False이므로 group 간 충돌은 없음.
        """
        try:
            import omni.usd as _omni_usd
            from pxr import PhysxSchema

            stage = _omni_usd.get_context().get_stage()
            for env_idx in range(self.num_envs):
                prim_path = f"/World/envs/env_{env_idx}/FluidParticles"
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    api = PhysxSchema.PhysxParticleSetAPI.Apply(prim)
                    api.CreateParticleGroupAttr().Set(env_idx)

            print(
                f"[5g_pour_right_v3] Particle groups assigned: 0~{self.num_envs - 1}",
                flush=True,
            )
        except Exception as exc:
            print(f"[5g_pour_right_v3] WARNING: particle group assignment failed: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Particle stage 경로 lazy 캐시 (첫 호출 시 stage 확인)
    # ------------------------------------------------------------------
    def _init_particle_view_if_needed(self) -> bool:
        """USD stage에서 fluid particle prim 경로를 캐시. 성공 시 True 반환."""
        if self._particle_view is not None:
            return True
        try:
            import omni.usd as _omni_usd
            stage = _omni_usd.get_context().get_stage()
            # env_0 prim이 존재하면 초기화 완료로 간주
            prim = stage.GetPrimAtPath("/World/envs/env_0/FluidParticles")
            if not prim.IsValid():
                return False
            # _particle_view를 stage 자체로 저장 (None이 아님을 표시)
            self._particle_view = stage
            print(
                f"[5g_pour_right_v3] Particle USD view ready: "
                f"{self.num_envs} envs × {self._num_particles_per_env} particles",
                flush=True,
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Particle positions 조회 / 설정 (PhysX SimulationPoints API)
    # ------------------------------------------------------------------
    def _get_particle_positions_w(self) -> torch.Tensor | None:
        """(N, P, 3) 세계좌표 particle positions 반환. 실패 시 None.

        `physxParticle:simulationPoints`는 PhysX schema 상 시뮬레이션 미러 버퍼이며,
        렌더/authoring 기준 위치는 `UsdGeom.Points.points`이다. Isaac Lab의 cloned env는
        각 `/World/envs/env_i` prim translation 아래에 있으므로, stage에 기록된 local
        point를 env origin을 더해 world 좌표로 복원한다.
        """
        if not self._init_particle_view_if_needed():
            return None
        try:
            import omni.usd as _omni_usd
            from pxr import UsdGeom, PhysxSchema
            import numpy as np

            stage = _omni_usd.get_context().get_stage()
            N = self.num_envs
            P = self._num_particles_per_env
            all_pos = torch.zeros(N, P, 3, dtype=torch.float32, device=self.device)

            for env_idx in range(N):
                prim_path = f"/World/envs/env_{env_idx}/FluidParticles"
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue

                points_prim = UsdGeom.Points(prim)
                pts = points_prim.GetPointsAttr().Get()
                if pts is None or len(pts) == 0:
                    particle_api = PhysxSchema.PhysxParticleSetAPI(prim)
                    sim_attr = particle_api.GetSimulationPointsAttr()
                    pts = sim_attr.Get() if sim_attr else None
                if pts is None:
                    continue
                arr = np.array(pts, dtype=np.float32, copy=True)
                if arr.shape[0] != P:
                    continue
                arr += self.scene.env_origins[env_idx].detach().cpu().numpy()
                all_pos[env_idx] = torch.from_numpy(arr).to(self.device)

            return all_pos
        except Exception:
            return None

    def _set_particle_positions_w(self, positions: torch.Tensor) -> None:
        """(N, P, 3) 세계좌표로 particle positions 설정.

        Particle set prim은 각 env prim 하위에 있으므로 local 좌표로 authoring해야 한다.
        PhysX schema 상 `simulationPoints`는 시뮬레이션 상태 mirror이므로, reset 시에는
        `UsdGeom.Points.points`를 canonical source로 갱신하고 `simulationPoints`는
        동일 길이 동기화가 필요한 경우에만 보조적으로 맞춘다.
        """
        if not self._init_particle_view_if_needed():
            return
        try:
            import omni.usd as _omni_usd
            from pxr import UsdGeom, PhysxSchema, Vt
            import numpy as np

            stage = _omni_usd.get_context().get_stage()
            pos_np = positions.detach().cpu().numpy().copy()  # (N, P, 3), world
            N = pos_np.shape[0]

            for env_idx in range(N):
                prim_path = f"/World/envs/env_{env_idx}/FluidParticles"
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue

                local_pts_np = pos_np[env_idx] - self.scene.env_origins[env_idx].detach().cpu().numpy()
                pts = Vt.Vec3fArray.FromNumpy(local_pts_np.astype(np.float32, copy=False))

                points_prim = UsdGeom.Points(prim)
                points_prim.GetPointsAttr().Set(pts)

                # smoothing / parser consistency를 위해 동일 local 좌표를 맞춰 둔다.
                particle_api = PhysxSchema.PhysxParticleSetAPI(prim)
                sim_attr = particle_api.GetSimulationPointsAttr()
                if not sim_attr.HasAuthoredValue():
                    sim_attr = particle_api.CreateSimulationPointsAttr()
                sim_attr.Set(pts)
        except Exception:
            pass

    def _set_particle_velocities_zero(
        self,
        env_ids_tensor: torch.Tensor | None = None,
        linear_velocity_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """particle velocity를 설정. 기본값은 0."""
        if not self._init_particle_view_if_needed():
            return
        try:
            import omni.usd as _omni_usd
            from pxr import UsdGeom, Vt
            import numpy as np

            stage = _omni_usd.get_context().get_stage()
            P = self._num_particles_per_env
            zero_np = np.tile(np.array(linear_velocity_xyz, dtype=np.float32), (P, 1))
            zero_vels = Vt.Vec3fArray.FromNumpy(zero_np)

            if env_ids_tensor is not None:
                env_ids = env_ids_tensor.cpu().tolist()
                if isinstance(env_ids, int):
                    env_ids = [env_ids]
            else:
                env_ids = list(range(self.num_envs))

            for env_idx in env_ids:
                prim_path = f"/World/envs/env_{env_idx}/FluidParticles"
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue
                UsdGeom.Points(prim).GetVelocitiesAttr().Set(zero_vels)
        except Exception:
            pass

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
    # Pregrasp grid 캐시 빌드
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
                self._reset_pca, pp, "euler_zyx",
                fq.detach(), fqd.detach(),
                self._reset_obj_ids, self._reset_obj_indicator,
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
        tip_norms = tip_xyz.norm(dim=-1)

        self.contact_force_xyz_raw.copy_(tip_xyz)
        self.contact_force_raw.copy_(tip_norms)
        self.binary_contact_buf.copy_(tip_norms > CONTACT_FORCE_THRESHOLD)
        self.num_contacts_buf.copy_(self.binary_contact_buf.sum(dim=-1).long())

        per_distal = self._distal_sensor.data.net_forces_w.norm(dim=-1)
        self.distal_contact_force_raw.copy_(per_distal)
        self.distal_binary_contact_buf.copy_(per_distal > CONTACT_FORCE_THRESHOLD)

        per_middle = self._middle_sensor.data.net_forces_w.norm(dim=-1)
        self.middle_contact_force_raw.copy_(per_middle)
        self.middle_binary_contact_buf.copy_(per_middle > CONTACT_FORCE_THRESHOLD)

    # ------------------------------------------------------------------
    # Fluid particle flags 계산 (bead flags 대체)
    # ------------------------------------------------------------------
    def _compute_fluid_flags(self) -> None:
        """각 입자가 source/target cup 내부에 있는지, spill 여부를 계산."""
        particle_pos_w = self._get_particle_positions_w()   # (N, P, 3) or None

        if particle_pos_w is None:
            # Particle view 미초기화 (startup 단계) → 0으로 유지
            return

        n = self.num_envs
        P = particle_pos_w.shape[1]

        # ---- Target cup (left_target_cup) 내부 판정 ----
        left_cup_quat_w = self.left_target_cup.data.root_quat_w   # (N, 4)
        left_cup_pos_w  = self.left_target_cup.data.root_pos_w    # (N, 3)

        left_quat_flat = left_cup_quat_w.unsqueeze(1).expand(-1, P, -1).reshape(-1, 4)
        left_rel_flat  = (particle_pos_w - left_cup_pos_w.unsqueeze(1)).reshape(-1, 3)
        pos_in_target  = quat_apply_inverse(left_quat_flat, left_rel_flat).reshape(n, P, 3)

        xy_to_target = torch.norm(pos_in_target[..., :2], dim=-1)   # (N, P)
        in_target = (
            (xy_to_target <= self.cfg.target_inner_radius)
            & (pos_in_target[..., 2] >= self.cfg.target_inside_z_min)
            & (pos_in_target[..., 2] <= self.cfg.target_inside_z_max)
        )

        # ---- Source cup (cup) 내부 판정 ----
        cup_quat_w = self.cup.data.root_quat_w   # (N, 4)
        cup_pos_w  = self.cup.data.root_pos_w    # (N, 3)

        cup_quat_flat = cup_quat_w.unsqueeze(1).expand(-1, P, -1).reshape(-1, 4)
        cup_rel_flat  = (particle_pos_w - cup_pos_w.unsqueeze(1)).reshape(-1, 3)
        pos_in_source = quat_apply_inverse(cup_quat_flat, cup_rel_flat).reshape(n, P, 3)

        xy_to_source = torch.norm(pos_in_source[..., :2], dim=-1)
        in_source = (
            (xy_to_source <= self.cfg.source_inner_radius)
            & (pos_in_source[..., 2] >= self.cfg.source_inside_z_min)
            & (pos_in_source[..., 2] <= self.cfg.source_inside_z_max)
        )

        # ---- Spill: source/target 모두 아닌 + 낮은 위치 ----
        particle_env_z = particle_pos_w[..., 2] - self.scene.env_origins[:, 2].unsqueeze(1)
        above_ground = particle_env_z > -5.0   # underground (hidden) 입자 제외
        spilled = (~in_target) & (~in_source) & (particle_env_z < 0.230) & above_ground

        # ---- 통계 ----
        self._fluid_in_target_fraction.copy_(in_target.float().mean(dim=-1))
        self._fluid_in_source_fraction.copy_(in_source.float().mean(dim=-1))
        self._fluid_accuracy.copy_(self._fluid_in_target_fraction)

        # above-ground 입자만으로 spill 비율 계산 (underground 숨김 입자 제외)
        above_count = above_ground.float().sum(dim=-1).clamp(min=1.0)
        self._spill_ratio.copy_(spilled.float().sum(dim=-1) / above_count)

        # Centroid: source cup 기준
        mask_source_f = in_source.float().unsqueeze(-1)   # (N, P, 1)
        source_count = in_source.float().sum(dim=-1).clamp(min=1.0)
        self._fluid_centroid_w.copy_(
            (particle_pos_w * mask_source_f).sum(dim=1) / source_count.unsqueeze(1)
        )

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone()

        palm_action   = actions[:, :6]
        finger_action = actions[:, 6:11]

        if self.cfg.episode_hold_steps > 0:
            hold_mask = (self.episode_length_buf < self.cfg.episode_hold_steps).unsqueeze(1)
            palm_action = torch.where(hold_mask, torch.zeros_like(palm_action), palm_action)

        delta = scale(palm_action, self.delta_mins, self.delta_maxs)
        palm_pose = self.pregrasp_palm_pose_buf + delta
        palm_pose = torch.max(torch.min(palm_pose, self.palm_maxs), self.palm_mins)
        self.palm_pose_targets.copy_(palm_pose)
        self.hand_pca_targets.zero_()

        self.open_tesollo_fabric.default_config.copy_(self.fabric_q.detach())

        self.open_tesollo_fabric.set_features(
            self.hand_pca_targets, self.palm_pose_targets, "euler_zyx",
            self.fabric_q.detach(), self.fabric_qd.detach(),
            self.object_ids, self.object_indicator, self.fabric_damping_gain,
        )
        for _ in range(self.cfg.fabric_decimation):
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.open_tesollo_integrator.step(
                self.fabric_q.detach(), self.fabric_qd.detach(), self.fabric_qdd.detach(), self.timestep
            )

        if self.cfg.freeze_grasp_hand_during_episode and (not self._warmstart_collect_mode):
            self.actions[:, 6:] = 1.0
            hand_target = self.grasp_hold_hand_pos_buf
        else:
            delta_20   = self.hand_grasp_pose - self.hand_open_pose
            t          = (finger_action + 1.0) / 2.0
            t_expanded = t.repeat_interleave(4, dim=1)
            hand_target = self.hand_open_pose.unsqueeze(0) + t_expanded * delta_20.unsqueeze(0)
        self.hand_joint_targets.copy_(hand_target)

        self.fabric_q[:, NUM_ARM_DOF:] = hand_target
        self.fabric_qd[:, NUM_ARM_DOF:].zero_()

    def _apply_action(self) -> None:
        arm_target = self.fabric_q[:, :NUM_ARM_DOF]
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(arm_target), joint_ids=self.arm_dof_indices
        )
        self.robot.set_joint_position_target(self.hand_joint_targets, joint_ids=self.hand_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(self.hand_joint_targets), joint_ids=self.hand_dof_indices
        )
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
        zero_cup_vel = torch.zeros(self.num_envs, 6, device=self.device)
        source_cage_pose = torch.cat([self.cup.data.root_pos_w, self.cup.data.root_quat_w], dim=-1)
        self.source_particle_cage.write_root_pose_to_sim(source_cage_pose)
        self.source_particle_cage.write_root_velocity_to_sim(zero_cup_vel)
        self.left_target_cup.write_root_pose_to_sim(left_cup_pose)
        self.left_target_cup.write_root_velocity_to_sim(zero_cup_vel)

    # ------------------------------------------------------------------
    # Intermediate values
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self) -> None:
        self.object_pos = self.cup.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.cup.data.root_quat_w
        left_target_pos_w  = self.left_target_cup.data.root_pos_w
        left_target_quat_w = self.left_target_cup.data.root_quat_w

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
        self._mouth_delta    = self._target_opening_w - self._source_pour_point_w
        self._mouth_distance = torch.norm(self._mouth_delta, dim=-1)

        self._mouth_xy_distance  = torch.norm(self._mouth_delta[:, :2], dim=-1)
        self._mouth_z_clearance  = self._source_pour_point_w[:, 2] - self._target_opening_w[:, 2]
        self._source_up_dot_world = self._source_up_axis_w[:, 2].clamp(-1.0, 1.0)

        _tilt_cos = math.cos(math.radians(self.cfg.target_pour_tilt_deg))
        _tilt_sin = math.sin(math.radians(self.cfg.target_pour_tilt_deg))
        _mouth_delta_xy = self._mouth_delta[:, :2]
        _tilt_dir_xy = _mouth_delta_xy / (_mouth_delta_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        _tilt_dir_3d = torch.cat([_tilt_dir_xy, torch.zeros(n, 1, device=self.device)], dim=-1)
        _ref_up = _tilt_cos * self._world_up.expand(n, -1) + _tilt_sin * _tilt_dir_3d
        _ref_up = _ref_up / _ref_up.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self._directional_tilt_cos = (self._source_up_axis_w * _ref_up).sum(dim=-1).clamp(-1.0, 1.0)

        _mouth_dir = self._mouth_delta / self._mouth_distance.unsqueeze(1).clamp(min=1e-6)
        self._mouth_alignment_cos = (self._source_pour_axis_w * _mouth_dir).sum(dim=-1).clamp(-1.0, 1.0)

        # Fluid particle flags
        self._compute_fluid_flags()

        # GUI 시각화
        _all_pts  = torch.cat([self._source_pour_point_w, self._target_opening_w], dim=0)
        _marker_idx = torch.zeros(2 * n, dtype=torch.long, device=self.device)
        _marker_idx[n:] = 1
        self._vis_markers.visualize(translations=_all_pts, marker_indices=_marker_idx)

        self._update_contact_forces()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_legacy_warmstart_policy_obs(self) -> torch.Tensor:
        """Warmstart checkpoint 차원(106D/107D)에 맞는 legacy obs 구성."""
        arm_joint_pos    = self.robot.data.joint_pos[:, self.arm_dof_indices]
        arm_joint_vel    = self.robot.data.joint_vel[:, self.arm_dof_indices]
        finger_joint_pos = self.robot.data.joint_pos[:, self.hand_dof_indices]
        finger_joint_vel = self.robot.data.joint_vel[:, self.hand_dof_indices]
        palm_center_pos  = self.palm_center_pos
        fingertip_pos    = self.fingertip_pos
        cup_pos          = self.object_pos
        binary_contact   = self.binary_contact_buf.float()
        last_actions     = self.actions

        fingertip_pos_rel_palm = (fingertip_pos - palm_center_pos.unsqueeze(1)).view(self.num_envs, -1)
        palm_to_cup            = cup_pos - palm_center_pos
        cup_to_fingertip       = (fingertip_pos - cup_pos.unsqueeze(1)).view(self.num_envs, -1)

        base_106 = torch.cat([
            arm_joint_pos,           # 7
            arm_joint_vel,           # 7
            finger_joint_pos,        # 20
            finger_joint_vel,        # 20
            palm_center_pos,         # 3
            fingertip_pos_rel_palm,  # 15
            palm_to_cup,             # 3
            cup_to_fingertip,        # 15
            binary_contact,          # 5
            last_actions,            # 11
        ], dim=-1)   # 106D

        policy_obs_dim = int(getattr(self._warmstart_policy, "obs_dim", NUM_LEGACY_WARMSTART_OBS))
        if policy_obs_dim == base_106.shape[1]:
            warmstart_obs = base_106
        elif policy_obs_dim == base_106.shape[1] + 1:
            # bead_mass_normalized = 1.0 고정: 최대 하중 기준 파지 자세 유도
            bead_mass_normalized = torch.ones(self.num_envs, 1, device=self.device)
            warmstart_obs = torch.cat([base_106, bead_mass_normalized], dim=-1)
        else:
            raise RuntimeError(
                f"[warmstart] Unsupported legacy obs dim: {policy_obs_dim}. "
                f"Expected {base_106.shape[1]} or {base_106.shape[1] + 1}."
            )
        return warmstart_obs

    def _get_observations(self) -> dict:
        # ==== Clean state (critic용) ====
        arm_joint_pos_clean    = self.robot.data.joint_pos[:, self.arm_dof_indices]
        arm_joint_vel_clean    = self.robot.data.joint_vel[:, self.arm_dof_indices]
        finger_joint_pos_clean = self.robot.data.joint_pos[:, self.hand_dof_indices]
        finger_joint_vel_clean = self.robot.data.joint_vel[:, self.hand_dof_indices]
        left_arm_joint_pos_clean = self.robot.data.joint_pos[:, self.left_arm_dof_indices]
        left_arm_joint_vel_clean = self.robot.data.joint_vel[:, self.left_arm_dof_indices]
        palm_center_pos_clean  = self.palm_center_pos

        right_cup_pos_clean  = self.cup.data.root_pos_w
        right_cup_quat_clean = self.cup.data.root_quat_w
        left_cup_pos_clean   = self.left_target_cup.data.root_pos_w
        left_cup_quat_clean  = self.left_target_cup.data.root_quat_w

        source_pour_point_clean = self._source_pour_point_w
        target_opening_clean    = self._target_opening_w
        source_pour_axis_clean  = self._source_pour_axis_w
        source_up_axis_clean    = self._source_up_axis_w
        target_up_axis_clean    = self._target_up_axis_w
        fluid_centroid_clean    = self._fluid_centroid_w

        # ==== Noisy state (Actor, sim2real) ====
        σ_qp = self.cfg.obs_noise_joint_pos
        σ_qv = self.cfg.obs_noise_joint_vel
        σ_bp = self.cfg.obs_noise_body_pos
        σ_cp = self.cfg.obs_noise_cup_pos

        arm_joint_pos    = arm_joint_pos_clean    + torch.randn_like(arm_joint_pos_clean)    * σ_qp
        arm_joint_vel    = arm_joint_vel_clean    + torch.randn_like(arm_joint_vel_clean)    * σ_qv
        finger_joint_pos = finger_joint_pos_clean + torch.randn_like(finger_joint_pos_clean) * σ_qp
        finger_joint_vel = finger_joint_vel_clean + torch.randn_like(finger_joint_vel_clean) * σ_qv
        palm_center_pos  = palm_center_pos_clean  + torch.randn_like(palm_center_pos_clean)  * σ_bp
        right_cup_pos    = right_cup_pos_clean    + torch.randn_like(right_cup_pos_clean)    * σ_cp
        left_cup_pos     = left_cup_pos_clean     + torch.randn_like(left_cup_pos_clean)     * σ_cp
        source_pour_point = source_pour_point_clean + torch.randn_like(source_pour_point_clean) * σ_cp
        target_opening    = target_opening_clean    + torch.randn_like(target_opening_clean)    * σ_cp

        right_cup_pos_rel_palm = right_cup_pos - palm_center_pos
        left_cup_pos_rel_palm  = left_cup_pos  - palm_center_pos
        pour_point_to_opening  = target_opening - source_pour_point

        binary_contact = self.binary_contact_buf.float()
        last_actions   = self.actions

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
            finger_joint_pos,           # 20
            finger_joint_vel,           # 20
            right_cup_pos_rel_palm,     # 3
            right_cup_quat_clean,       # 4
            left_cup_pos_rel_palm,      # 3
            left_cup_quat_clean,        # 4
            pour_point_to_opening,      # 3
            source_pour_axis_clean,     # 3
            source_up_axis_clean,       # 3
            target_up_axis_clean,       # 3
            transport_summary,          # 5
            binary_contact,             # 5
            last_actions,               # 11
        ], dim=-1)   # 101D

        if actor_obs.shape[1] != NUM_OBSERVATIONS:
            raise RuntimeError(
                f"[pour_v3] Actor obs dim mismatch: {actor_obs.shape[1]} != {NUM_OBSERVATIONS}"
            )

        # ==== Critic extra obs (42D) ====
        cup_height_delta   = (right_cup_pos_clean[:, 2] - self.object_init_pos[:, 2]).unsqueeze(1)
        distal_binary      = self.distal_binary_contact_buf.float()
        distal_force_norm  = (self.distal_contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)

        fluid_centroid_rel_source = quat_apply_inverse(
            right_cup_quat_clean,
            fluid_centroid_clean - right_cup_pos_clean,
        )
        fluid_centroid_rel_target = quat_apply_inverse(
            left_cup_quat_clean,
            fluid_centroid_clean - left_cup_pos_clean,
        )

        actor_obs_clean = torch.cat([
            arm_joint_pos_clean, arm_joint_vel_clean,
            finger_joint_pos_clean, finger_joint_vel_clean,
            right_cup_pos_clean - palm_center_pos_clean,
            right_cup_quat_clean,
            left_cup_pos_clean - palm_center_pos_clean,
            left_cup_quat_clean,
            target_opening_clean - source_pour_point_clean,
            source_pour_axis_clean, source_up_axis_clean, target_up_axis_clean,
            torch.stack([
                self._mouth_distance, self._mouth_xy_distance, self._mouth_z_clearance,
                self._source_up_dot_world, self._directional_tilt_cos,
            ], dim=-1),
            binary_contact, last_actions,
        ], dim=-1)   # 101D

        critic_obs = torch.cat([
            actor_obs_clean,                               # 101
            left_arm_joint_pos_clean,                      # 9
            left_arm_joint_vel_clean,                      # 9
            distal_binary,                                 # 5
            distal_force_norm,                             # 5
            cup_height_delta,                              # 1
            fluid_centroid_rel_source,                     # 3  (fluid centroid rel source cup)
            fluid_centroid_rel_target,                     # 3  (fluid centroid rel target cup)
            self._mouth_distance.unsqueeze(1),             # 1
            self._mouth_xy_distance.unsqueeze(1),          # 1
            self._mouth_z_clearance.unsqueeze(1),          # 1
            self._source_up_dot_world.unsqueeze(1),        # 1
            self._directional_tilt_cos.unsqueeze(1),       # 1
            self._mouth_alignment_cos.unsqueeze(1),        # 1
            self._fluid_accuracy.unsqueeze(1),             # 1  (fraction in target)
        ], dim=-1)   # 143D

        if critic_obs.shape[1] != NUM_CRITIC_OBSERVATIONS:
            raise RuntimeError(
                f"[pour_v3] Critic obs dim mismatch: {critic_obs.shape[1]} != {NUM_CRITIC_OBSERVATIONS}"
            )

        return {"policy": actor_obs, "critic": critic_obs}

    # ------------------------------------------------------------------
    # Rewards: Cost-Benefit Tradeoff
    # R = w_A * A(t) - w_T * T(t) - w_E * E(t) + 보조 shaping
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        # ---- [편익] A: Accuracy — target cup 내 fluid 비율 [0, 1] ----
        A = self._fluid_accuracy   # (N,)

        # ---- [비용] T: Time — 지수 패널티 exp(alpha*(t - t_max)) ----
        t_frac = self.episode_length_buf.float() / float(EPISODE_STEPS)
        T = torch.exp(self.cfg.alpha_time * (t_frac - 1.0))   # (N,), ∈(0,1] 에피소드 끝에 가장 큼

        # ---- [비용] E: Effort — 팔 관절 토크 제곱합 ----
        arm_torque = self.robot.data.applied_torque[:, self.arm_dof_indices]
        E = (arm_torque ** 2).sum(dim=-1)   # (N,)

        # ---- 주 리워드 ----
        main_reward = self.cfg.w_accuracy * A - self.cfg.w_time * T - self.cfg.w_effort * E

        # ---- 보조 shaping: Transport ----
        transport_reward = 1.0 - torch.tanh(self.cfg.reward_transport_scale * self._mouth_xy_distance)

        # ---- 보조 shaping: Tilt ----
        target_tilt_cos = math.cos(math.radians(self.cfg.target_pour_tilt_deg))
        tilt_error = torch.abs(self._source_up_dot_world - target_tilt_cos)
        tilt_reward = 1.0 - torch.tanh(self.cfg.reward_tilt_scale * tilt_error)
        directional_tilt_reward = ((self._directional_tilt_cos + 1.0) * 0.5).clamp(0.0, 1.0)
        combined_tilt = 0.5 * tilt_reward + 0.5 * directional_tilt_reward

        # ---- 보조 shaping: Force balance (파지 품질) ----
        thumb_force       = self.contact_force_raw[:, 0]
        others_avg_force  = self.contact_force_raw[:, 1:].mean(dim=-1)
        has_thumb         = self.binary_contact_buf[:, 0].float()
        has_others        = (self.binary_contact_buf[:, 1:].sum(dim=-1) >= 1).float()
        force_balance_gate = has_thumb * has_others
        force_balance_err  = (thumb_force - others_avg_force).abs()
        force_balance_reward = force_balance_gate * torch.exp(
            -self.cfg.reward_force_balance_sharpness * force_balance_err
        )

        # ---- 성공 보너스 ----
        success_reward = self.success_flag.float()

        # ---- 패널티: Spill ----
        spill_penalty = self._spill_ratio

        # ---- 패널티: Action rate ----
        action_rate_penalty = torch.sum((self.actions - self.prev_actions) ** 2, dim=-1)

        total = (
            main_reward
            + self.cfg.w_transport    * transport_reward
            + self.cfg.w_tilt         * combined_tilt
            + self.cfg.w_force_balance * force_balance_reward
            + self.cfg.w_success      * success_reward
            - self.cfg.w_spill        * spill_penalty
            - self.cfg.w_action_rate  * action_rate_penalty
        )

        # ---- Extras (TensorBoard 로깅) ----
        self.extras["reward_accuracy"]        = A.mean()
        self.extras["reward_time_cost"]       = T.mean()
        self.extras["reward_effort_cost"]     = E.mean()
        self.extras["reward_main"]            = main_reward.mean()
        self.extras["reward_transport"]       = transport_reward.mean()
        self.extras["reward_tilt"]            = tilt_reward.mean()
        self.extras["reward_directional_tilt"] = directional_tilt_reward.mean()
        self.extras["reward_force_balance"]   = force_balance_reward.mean()
        self.extras["reward_success"]         = success_reward.mean()
        self.extras["penalty_spill"]          = spill_penalty.mean()
        self.extras["penalty_action_rate"]    = action_rate_penalty.mean()
        self.extras["fluid_accuracy"]         = self._fluid_accuracy.mean()
        self.extras["fluid_in_source_rate"]   = self._fluid_in_source_fraction.mean()
        self.extras["fluid_in_target_rate"]   = self._fluid_in_target_fraction.mean()
        self.extras["spill_ratio"]            = self._spill_ratio.mean()
        self.extras["mouth_distance"]         = self._mouth_distance.mean()
        self.extras["mouth_xy_distance"]      = self._mouth_xy_distance.mean()
        self.extras["mouth_z_clearance"]      = self._mouth_z_clearance.mean()
        self.extras["source_up_dot"]          = self._source_up_dot_world.mean()
        self.extras["directional_tilt_cos"]   = self._directional_tilt_cos.mean()
        self.extras["mouth_alignment_cos"]    = self._mouth_alignment_cos.mean()
        self.extras["success_rate"]           = self.success_flag.float().mean()
        self.extras["num_contacts"]           = self.num_contacts_buf.float().mean()
        self.extras["thumb_force_mean"]       = thumb_force.mean()
        self.extras["others_avg_force_mean"]  = others_avg_force.mean()

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

        # 과도한 spillage → 에피소드 종료 (60% 이상 흘리면 의미 없음)
        excessive_spill = self._spill_ratio > self.cfg.spill_termination_ratio

        # 성공: fluid accuracy >= threshold (10% 이상 target cup에 들어감)
        self.success_flag.copy_(self._fluid_accuracy >= self.cfg.success_accuracy_threshold)
        self.episode_success_buf |= self.success_flag
        self._maybe_store_warmstart_successes()

        terminated = out_x | out_y | fallen | dropped_by_force | excessive_spill | self.success_flag
        truncated  = self.episode_length_buf >= self.max_episode_length - 1

        self.extras["object_z"]            = self.object_pos[:, 2].mean()
        self.extras["drop_force_rate"]     = dropped_by_force.float().mean()
        self.extras["no_tip_force_steps"]  = self._no_tip_force_steps.float().mean()
        self.extras["excessive_spill_rate"] = excessive_spill.float().mean()

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

        self._total_episodes      += n
        self._successful_episodes += int(self.episode_success_buf[env_ids].sum().item())
        self.episode_success_buf[env_ids] = False

        if (not self._warmstart_collect_mode) and self._warmstart_cache_count > 0:
            self._reset_from_warmstart_cache(env_ids)
            return

        # ---- 1. 로봇 관절 리셋 ----
        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_pos[:, self.actuated_dof_indices] = self.robot_start_joint_pos[0]
        full_pos[:, self.left_arm_dof_indices] = self.left_arm_zero_pos[0]
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        # ---- 2. Fabrics 리셋 ----
        self.fabric_q[env_ids]   = self.robot_start_joint_pos[env_ids]
        self.fabric_qd[env_ids].zero_()
        self.fabric_qdd[env_ids].zero_()

        # ---- 3. 컵 spawn 위치 ----
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

        # ---- 4. Fabrics pregrasp rollout ----
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

        xi = ((obj_x - self._cache_xs[0]) / (self._cache_xs[1] - self._cache_xs[0])).round().long().clamp(0, self._cache_n - 1)
        yi = ((obj_y - self._cache_ys[0]) / (self._cache_ys[1] - self._cache_ys[0])).round().long().clamp(0, self._cache_n - 1)
        q_pregrasp = self.fabric_q[env_ids].clone()
        q_pregrasp[:, :NUM_ARM_DOF] = self._cache_q_arm[xi, yi]

        self.fabric_q[env_ids] = q_pregrasp
        self.fabric_qd[env_ids].zero_()
        self.fabric_qdd[env_ids].zero_()

        approach_hand = self.hand_open_pose.unsqueeze(0).expand(n, -1)
        self.fabric_q[env_ids, NUM_ARM_DOF:] = approach_hand
        self.fabric_qd[env_ids, NUM_ARM_DOF:].zero_()

        # ---- 5. 버퍼 동기화 ----
        self.pregrasp_arm_pos_buf[env_ids] = q_pregrasp[:, :NUM_ARM_DOF]
        self.palm_pose_targets[env_ids]    = pregrasp_palm_pose
        self.pregrasp_palm_pose_buf[env_ids] = pregrasp_palm_pose
        self._grasp_rel_palm_to_cup_init[env_ids] = obj_pos_local - pregrasp_palm_pose[:, :3]
        self._grasp_cup_height_init[env_ids] = obj_pos_local[:, 2]
        self.open_tesollo_fabric.default_config[env_ids, :NUM_ARM_DOF] = q_pregrasp[:, :NUM_ARM_DOF]
        self.grasp_hold_hand_pos_buf[env_ids] = approach_hand

        # ---- 6. 로봇 초기화 ----
        pregrasp_full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        pregrasp_full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)
        pregrasp_full_pos[:, self.arm_dof_indices]       = q_pregrasp[:, :NUM_ARM_DOF]
        pregrasp_full_pos[:, self.hand_dof_indices]      = approach_hand
        pregrasp_full_pos[:, self.left_arm_dof_indices]  = self.left_arm_zero_pos[0]
        self.robot.write_joint_state_to_sim(pregrasp_full_pos, pregrasp_full_vel, env_ids=env_ids)

        # ---- 7. 컵 spawn ----
        obj_pos_world = obj_pos_local + self.scene.env_origins[env_ids]
        upright_rot = torch.zeros(n, 4, device=self.device)
        upright_rot[:, 0] = 1.0
        zero_vel = torch.zeros(n, 6, device=self.device)
        cup_root_state = torch.cat([obj_pos_world, upright_rot, zero_vel], dim=-1)
        self.cup.write_root_state_to_sim(cup_root_state, env_ids=env_ids)
        self.source_particle_cage.write_root_pose_to_sim(cup_root_state[:, :7], env_ids=env_ids)
        self.source_particle_cage.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        left_cup_pose = self._compute_attached_root_pose(
            self._left_target_cup_body_id,
            self._left_target_cup_attach_pos_b,
            self._left_target_cup_attach_quat_b,
            env_ids=env_ids,
        )
        self.left_target_cup.write_root_pose_to_sim(left_cup_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        # ---- 8. Particle 처리 ----
        if self._warmstart_collect_mode:
            self._hide_particles(list(env_ids))
        else:
            self._reset_particle_state(list(env_ids), cup_root_state[:, :7])

        # ---- 9. 버퍼 리셋 ----
        self.hand_joint_targets[env_ids] = approach_hand
        self.contact_force_raw[env_ids].zero_()
        self.binary_contact_buf[env_ids] = False
        self.num_contacts_buf[env_ids]   = 0
        self.distal_contact_force_raw[env_ids].zero_()
        self.distal_binary_contact_buf[env_ids] = False
        self.middle_contact_force_raw[env_ids].zero_()
        self.middle_binary_contact_buf[env_ids] = False
        self._fluid_accuracy[env_ids]           = 0.0
        self._fluid_in_target_fraction[env_ids] = 0.0
        self._fluid_in_source_fraction[env_ids] = 0.0
        self._fluid_centroid_w[env_ids].zero_()
        self._spill_ratio[env_ids]              = 0.0
        self._no_tip_force_steps[env_ids]       = 0
        self.success_flag[env_ids]              = False
        self._pre_pour_ready_steps[env_ids]     = 0

        self.actions[env_ids, :6] = 0.0
        self.actions[env_ids, 6:] = -1.0
        self.prev_actions[env_ids, :6] = 0.0
        self.prev_actions[env_ids, 6:] = -1.0

    # ------------------------------------------------------------------
    # Particle reset helpers
    # ------------------------------------------------------------------
    def _reset_particle_state(self, env_ids: list[int], cup_pose_world: torch.Tensor) -> None:
        """source cup 내부에 particle grid 배치. cup_pose_world: (n, 7) [pos+quat]."""
        if not self._init_particle_view_if_needed():
            return  # particle view 미초기화 → 다음 reset 시 재시도

        n = len(env_ids)
        P = self._num_particles_per_env

        cup_pos_w  = cup_pose_world[:, :3]   # (n, 3)
        cup_quat_w = cup_pose_world[:, 3:7]  # (n, 4)

        # cup local frame offset → world frame
        offsets = self._fluid_grid_offsets_b  # (P, 3)
        offsets_expanded = offsets.unsqueeze(0).expand(n, -1, -1)   # (n, P, 3)
        cup_quat_flat = cup_quat_w.unsqueeze(1).expand(n, P, 4).reshape(-1, 4)
        offsets_flat  = offsets_expanded.reshape(-1, 3)

        particle_pos_w = (
            cup_pos_w.unsqueeze(1) + quat_apply(cup_quat_flat, offsets_flat).reshape(n, P, 3)
        )   # (n, P, 3)

        # 전체 위치 가져와서 해당 env_ids 슬라이스만 교체
        all_pos = self._get_particle_positions_w()
        if all_pos is None:
            return

        env_ids_t = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        all_pos[env_ids_t] = particle_pos_w
        self._set_particle_positions_w(all_pos)
        self._set_particle_velocities_zero(env_ids_t, linear_velocity_xyz=(0.0, 0.0, -0.05))

    def _hide_particles(self, env_ids: list[int]) -> None:
        """particle을 지하로 이동 (warmstart collect mode)."""
        if not self._init_particle_view_if_needed():
            return  # 초기 상태에서 particle은 이미 -10z에 생성됨

        n = len(env_ids)
        P = self._num_particles_per_env

        all_pos = self._get_particle_positions_w()
        if all_pos is None:
            return

        env_ids_t = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        underground = self.scene.env_origins[env_ids_t].unsqueeze(1).expand(n, P, 3).clone()
        underground[..., 2] = -10.0
        all_pos[env_ids_t] = underground
        self._set_particle_positions_w(all_pos)
        self._set_particle_velocities_zero(env_ids_t)

    # ------------------------------------------------------------------
    # Attachment / pose utilities
    # ------------------------------------------------------------------
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

    def _compute_attached_root_pose(
        self,
        body_id: int,
        attach_pos_b: torch.Tensor,
        attach_quat_b: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if env_ids is None:
            body_pos_w  = self.robot.data.body_pos_w[:, body_id]
            body_quat_w = self.robot.data.body_quat_w[:, body_id]
        else:
            body_pos_w  = self.robot.data.body_pos_w[env_ids, body_id]
            body_quat_w = self.robot.data.body_quat_w[env_ids, body_id]

        attach_pos_w  = body_pos_w + quat_apply(body_quat_w, attach_pos_b.unsqueeze(0).expand_as(body_pos_w))
        attach_quat_w = quat_mul(body_quat_w, attach_quat_b.unsqueeze(0).expand(body_quat_w.shape[0], -1))
        return torch.cat([attach_pos_w, attach_quat_w], dim=-1)

    # ------------------------------------------------------------------
    # Warmstart cache 구축
    # ------------------------------------------------------------------
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
        obs_noise_body_pos  = self.cfg.obs_noise_body_pos
        obs_noise_cup_pos   = self.cfg.obs_noise_cup_pos
        self.cfg.obs_noise_joint_pos = 0.0
        self.cfg.obs_noise_joint_vel = 0.0
        self.cfg.obs_noise_body_pos  = 0.0
        self.cfg.obs_noise_cup_pos   = 0.0
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
            self.cfg.obs_noise_body_pos  = obs_noise_body_pos
            self.cfg.obs_noise_cup_pos   = obs_noise_cup_pos

        if self._warmstart_cache_count == 0:
            raise RuntimeError(
                "[5g_pour_right_v3] warmstart cache is empty. "
                "The v8 checkpoint rollout did not produce any lift-success state."
            )

        print(
            f"[5g_pour_right_v3] collected {self._warmstart_cache_count} warmstart success states.",
            flush=True,
        )

    def _maybe_store_warmstart_successes(self) -> None:
        if not self._warmstart_collect_mode:
            return
        if self._warmstart_cache_count >= self._warmstart_arm_pos.shape[0]:
            return

        lifted  = self.object_pos[:, 2] > (self.object_init_pos[:, 2] + self.cfg.lift_success_height)
        grasped = self.num_contacts_buf >= MIN_CONTACTS_FOR_SUCCESS
        upright = self._source_up_axis_w[:, 2] > 0.7
        warmstart_success = lifted & grasped & upright

        success_env_ids = warmstart_success.nonzero(as_tuple=False).squeeze(-1)
        if success_env_ids.numel() == 0:
            return

        remaining       = self._warmstart_arm_pos.shape[0] - self._warmstart_cache_count
        success_env_ids = success_env_ids[:remaining]
        count = success_env_ids.numel()
        if count == 0:
            return

        start = self._warmstart_cache_count
        end   = start + count
        self._warmstart_arm_pos[start:end]      = self.robot.data.joint_pos[success_env_ids][:, self.arm_dof_indices]
        self._warmstart_hand_pos[start:end]     = self.robot.data.joint_pos[success_env_ids][:, self.hand_dof_indices]
        self._warmstart_palm_pose[start:end]    = self.palm_pose_targets[success_env_ids]
        self._warmstart_cup_pose[start:end, :3] = (
            self.cup.data.root_pos_w[success_env_ids] - self.scene.env_origins[success_env_ids]
        )
        self._warmstart_cup_pose[start:end, 3:7] = self.cup.data.root_quat_w[success_env_ids]
        self._warmstart_cache_count = end

    def _reset_from_warmstart_cache(self, env_ids: Sequence[int]) -> None:
        n = len(env_ids)
        pick = torch.randint(self._warmstart_cache_count, (n,), device=self.device)
        arm_pos      = self._warmstart_arm_pos[pick]
        hand_pos     = self._warmstart_hand_pos[pick]
        palm_pose    = self._warmstart_palm_pose[pick]
        cup_pose_local = self._warmstart_cup_pose[pick]

        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_pos[:, self.arm_dof_indices]      = arm_pos
        full_pos[:, self.hand_dof_indices]     = hand_pos
        full_pos[:, self.left_arm_dof_indices] = self.left_arm_zero_pos[0]
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        self.fabric_q[env_ids].zero_()
        self.fabric_q[env_ids, :NUM_ARM_DOF] = arm_pos
        self.fabric_q[env_ids, NUM_ARM_DOF:] = hand_pos
        self.fabric_qd[env_ids].zero_()
        self.fabric_qdd[env_ids].zero_()

        self.pregrasp_arm_pos_buf[env_ids]    = arm_pos
        self.grasp_hold_hand_pos_buf[env_ids] = hand_pos

        # Scripted pre-lift: warmstart cup z 올리기
        lifted_palm_pose = palm_pose.clone()
        lifted_palm_pose[:, 2] = torch.clamp(
            palm_pose[:, 2] + 0.25,
            self.palm_mins[2],
            self.palm_maxs[2],
        )
        self.pregrasp_palm_pose_buf[env_ids] = lifted_palm_pose
        self.palm_pose_targets[env_ids]       = lifted_palm_pose
        self.hand_joint_targets[env_ids]      = hand_pos
        self.object_init_pos[env_ids]         = cup_pose_local[:, :3]
        self.object_init_pos[env_ids, 2]      = self.cfg.object_spawn_z
        self._grasp_rel_palm_to_cup_init[env_ids] = cup_pose_local[:, :3] - palm_pose[:, :3]
        self._grasp_cup_height_init[env_ids]  = cup_pose_local[:, 2]
        self.open_tesollo_fabric.default_config[env_ids, :NUM_ARM_DOF] = arm_pos

        cup_pose_world = cup_pose_local.clone()
        cup_pose_world[:, :3] += self.scene.env_origins[env_ids]
        zero_vel = torch.zeros(n, 6, device=self.device)
        self.cup.write_root_pose_to_sim(cup_pose_world, env_ids=env_ids)
        self.cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        self.source_particle_cage.write_root_pose_to_sim(cup_pose_world, env_ids=env_ids)
        self.source_particle_cage.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        left_cup_pose = self._compute_attached_root_pose(
            self._left_target_cup_body_id,
            self._left_target_cup_attach_pos_b,
            self._left_target_cup_attach_quat_b,
            env_ids=env_ids,
        )
        self.left_target_cup.write_root_pose_to_sim(left_cup_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        # Particle: source cup 내부에 배치
        self._reset_particle_state(list(env_ids), cup_pose_world)

        # 버퍼 리셋
        self.contact_force_raw[env_ids].zero_()
        self.binary_contact_buf[env_ids]       = False
        self.num_contacts_buf[env_ids]         = 0
        self.distal_contact_force_raw[env_ids].zero_()
        self.distal_binary_contact_buf[env_ids] = False
        self.middle_contact_force_raw[env_ids].zero_()
        self.middle_binary_contact_buf[env_ids] = False
        self._fluid_accuracy[env_ids]           = 0.0
        self._fluid_in_target_fraction[env_ids] = 0.0
        self._fluid_in_source_fraction[env_ids] = 0.0
        self._fluid_centroid_w[env_ids].zero_()
        self._spill_ratio[env_ids]              = 0.0
        self._no_tip_force_steps[env_ids]       = 0
        self.success_flag[env_ids]              = False
        self._pre_pour_ready_steps[env_ids]     = 0

        self.actions[env_ids, :6]      = 0.0
        self.actions[env_ids, 6:]      = 1.0
        self.prev_actions[env_ids, :6] = 0.0
        self.prev_actions[env_ids, 6:] = 1.0
