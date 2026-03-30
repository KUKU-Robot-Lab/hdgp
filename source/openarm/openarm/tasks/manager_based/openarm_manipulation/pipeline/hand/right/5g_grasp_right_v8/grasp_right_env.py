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

"""환경 클래스: 5g_grasp_right_v8

v8: v7 기반 + Bead 무게 도메인 랜덤화
- 에피소드마다 컵 안에 0~10개 bead 랜덤 스폰 (각 10g, 최대 +100g)
- bead_mass_normalized (1D) obs 추가 → 정책이 현재 하중 인지, ADR과 연계

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
    OBJECT_GOAL_POS,
)
from .grasp_right_utils import scale, to_torch


class GraspRightEnv(DirectRLEnv):
    """OpenArm+Teosllo 오른손 파지 환경 v7.

    Action: 11D
      [0:6]  palm pose (x,y,z,ez,ey,ex), 정규화 [-1,1] → Fabrics IK
      [6:11] per-finger lerp (thumb,index,middle,ring,pinky)
             -1 = HAND_APPROACH_POSE, +1 = HAND_GRASP_POSE

    Episode:
      Grasp phase (step 0~479):  Fabrics arm + 정책 손가락
      Lift  phase (step 480~599): scripted arm prelift + frozen hand
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
        # distal phalanx body indices (rl_dg_*_4) — R2 reward용
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
        self.palm_center_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.fingertip_pos   = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.distal4_pos     = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.actions         = torch.zeros(self.num_envs, cfg.num_actions, device=self.device)
        self.prev_actions    = torch.full((self.num_envs, cfg.num_actions), 0.0, device=self.device)

        # ----------------------------------------------------------------
        # Pregrasp / Lift 버퍼 (reset에서 계산)
        # ----------------------------------------------------------------
        self.pregrasp_arm_pos_buf  = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        self.prelift_arm_pos_buf   = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        self.lift_arm_start_buf    = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        self.lift_finger_pos_buf   = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        self.is_lift_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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
        self._cup_tipping_cos = math.cos(math.radians(cfg.cup_tipping_max_deg))
        # episode-level 성공 추적 (per-step average 허수 문제 해결)
        self.episode_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._total_episodes: int = 0
        self._successful_episodes: int = 0

        # ----------------------------------------------------------------
        # Eval 로깅 (bead 무게별 파지 품질 평가)
        # ----------------------------------------------------------------
        # grasp phase 마지막(LIFT_START_STEP)에서의 finger action 강도 기록
        self._eval_grip_at_lift = torch.zeros(self.num_envs, device=self.device)
        # 에피소드별 (bead_mass_normalized, grip_intensity, success) 기록 리스트
        self._eval_records: list[dict] = []

        # ----------------------------------------------------------------
        # Bead 무게 도메인 랜덤화 (v8)
        # ----------------------------------------------------------------
        # 컵 내부 layered spiral 오프셋 사전 계산 (pour_v2 패턴)
        beads_per_layer = 5
        _bead_offsets = []
        for i in range(cfg.num_beads):
            layer = i // beads_per_layer
            slot  = i % beads_per_layer
            angle  = (2 * math.pi * slot / beads_per_layer) + 0.35 * layer
            radius = 0.014 + 0.004 * (layer % 2)
            z      = cfg.bead_spawn_z_offset + 0.006 + 0.014 * layer
            _bead_offsets.append([radius * math.cos(angle), radius * math.sin(angle), z])
        self._bead_offsets_b = torch.tensor(_bead_offsets, dtype=torch.float32, device=self.device)  # (num_beads, 3)

        # obs용 정규화된 bead 무게 버퍼
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

        # 초기 액션: 0 → palm pose workspace 중심 (접근 자세 유지)
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

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone()

        palm_action   = actions[:, :6]    # (N, 6) ∈ [-1, 1]
        finger_action = actions[:, 6:11]  # (N, 5) ∈ [-1, 1]

        # ---- Phase 판정 ----
        is_lift = (self.episode_length_buf >= LIFT_START_STEP)
        self.is_lift_phase.copy_(is_lift)

        # ---- Lift 진입 시 finger/arm joint pos 캡처 ----
        just_entering_lift = (self.episode_length_buf == LIFT_START_STEP)

        # Eval: lift 진입 시점 grip intensity (actions[6:11] 평균) 기록
        if just_entering_lift.any():
            self._eval_grip_at_lift[just_entering_lift] = finger_action[just_entering_lift].mean(dim=-1)

        # Finger: 진입 시점 자세로 고정
        self.lift_finger_pos_buf = torch.where(
            just_entering_lift.unsqueeze(1),
            self.robot.data.joint_pos[:, self.hand_dof_indices],
            self.lift_finger_pos_buf,
        )

        # Arm: 진입 시점 실제 위치 캡처 → lift 보간 시작점으로 사용
        # (pregrasp_arm_pos_buf 대신 실제값 사용: grasp phase에서 Fabrics가 arm을
        #  실제로 이동했으므로 전환 시 불연속 없이 자연스럽게 lift)
        actual_arm_pos = self.robot.data.joint_pos[:, self.arm_dof_indices]
        self.lift_arm_start_buf = torch.where(
            just_entering_lift.unsqueeze(1),
            actual_arm_pos,
            self.lift_arm_start_buf,
        )
        # prelift = 실제 위치에서 j4 +0.31 (lift 방향 고정)
        actual_prelift = actual_arm_pos.clone()
        actual_prelift[:, 3] = (actual_arm_pos[:, 3] + 0.31).clamp(max=3.14)
        self.prelift_arm_pos_buf = torch.where(
            just_entering_lift.unsqueeze(1),
            actual_prelift,
            self.prelift_arm_pos_buf,
        )

        # ---- Grasp phase: Fabrics arm 제어 ----
        # Delta action: action=0 → pregrasp 유지, action=±1 → pregrasp ± delta
        # 절대 workspace(palm_mins/maxs)로 클램프하여 안전 영역 보장
        delta = scale(palm_action, self.delta_mins, self.delta_maxs)   # (N, 6)
        palm_pose = self.pregrasp_palm_pose_buf + delta
        palm_pose = torch.max(torch.min(palm_pose, self.palm_maxs), self.palm_mins)
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

        # ---- Per-finger lerp: action [-1,1] → APPROACH_POSE(-1) ~ GRASP_POSE(+1) ----
        delta_20   = self.hand_grasp_pose - self.hand_open_pose                # (20,)
        t          = (finger_action + 1.0) / 2.0                               # (N,5) ∈ [0,1]
        t_expanded = t.repeat_interleave(4, dim=1)                             # (N,20)
        hand_target = self.hand_open_pose.unsqueeze(0) + t_expanded * delta_20.unsqueeze(0)
        self.hand_joint_targets.copy_(hand_target)

        # fabric_q hand 부분 동기화 (FK 계산에 활용)
        self.fabric_q[:, NUM_ARM_DOF:] = hand_target
        self.fabric_qd[:, NUM_ARM_DOF:].zero_()

        # ---- Lift phase: Fabrics arm 상태 동결 ----
        # lift 중에도 Fabrics integrator가 계속 실행되면 palm target과 실제 arm 위치가
        # 괴리되면서 fabric_qd가 발산 → arm에 전달되지 않더라도 상태 불안정 초래
        # 실제 arm joint 위치로 동기화 + 속도 제로 → lift phase 좌우 흔들림 제거
        if self.is_lift_phase.any():
            lift_mask = self.is_lift_phase
            self.fabric_q[lift_mask, :NUM_ARM_DOF] = (
                self.robot.data.joint_pos[lift_mask][:, self.arm_dof_indices]
            )
            self.fabric_qd[lift_mask, :NUM_ARM_DOF].zero_()
            self.fabric_qdd[lift_mask, :NUM_ARM_DOF].zero_()

    def _apply_action(self) -> None:
        is_lift = self.is_lift_phase   # (N,) bool

        # ---- 오른팔 ----
        # Grasp phase: Fabrics arm target
        # Lift  phase: pregrasp → prelift 선형 보간
        lift_progress = (
            (self.episode_length_buf - LIFT_START_STEP).clamp(min=0).float()
            / LIFT_PHASE_STEPS
        ).clamp(max=1.0).unsqueeze(1)   # (N, 1) ∈ [0, 1]

        arm_target_lift = (
            self.lift_arm_start_buf * (1.0 - lift_progress)
            + self.prelift_arm_pos_buf * lift_progress
        )
        arm_target = torch.where(
            is_lift.unsqueeze(1),
            arm_target_lift,
            self.fabric_q[:, :NUM_ARM_DOF],
        )

        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(arm_target), joint_ids=self.arm_dof_indices
        )

        # ---- 오른손 ----
        # Grasp phase: per-finger lerp target
        # Lift  phase: 진입 시점 캡처값 고정
        finger_target = torch.where(
            is_lift.unsqueeze(1),
            self.lift_finger_pos_buf,
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
        # 물체 위치
        self.object_pos = self.cup.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.cup.data.root_quat_w

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

        # 접촉력 업데이트
        self._update_contact_forces()

    # ------------------------------------------------------------------
    # Observations: Actor 107D | Critic 144D
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # ==== 공통 clean state (critic용, 물리 정확값) ====
        arm_joint_pos_clean    = self.robot.data.joint_pos[:, self.arm_dof_indices]    # (N, 7)
        arm_joint_vel_clean    = self.robot.data.joint_vel[:, self.arm_dof_indices]    # (N, 7)
        finger_joint_pos_clean = self.robot.data.joint_pos[:, self.hand_dof_indices]  # (N, 20)
        finger_joint_vel_clean = self.robot.data.joint_vel[:, self.hand_dof_indices]  # (N, 20)
        palm_center_pos_clean  = self.palm_center_pos                                  # (N, 3)
        fingertip_pos_clean    = self.fingertip_pos                                    # (N, 5, 3)
        cup_pos_clean          = self.object_pos                                       # (N, 3)

        # ==== Actor obs용 noisy state (sim2real domain randomization) ====
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

        # ==== Actor obs 조합 (107D) ====
        # 4. fingertip pos relative to palm (15D)
        fingertip_pos_rel_palm = (
            fingertip_pos - palm_center_pos.unsqueeze(1)
        ).view(self.num_envs, -1)

        # 5. palm to cup vector (3D)
        palm_to_cup = cup_pos_noisy - palm_center_pos

        # 6. cup to fingertip vectors (15D)
        cup_to_fingertip = (
            fingertip_pos - cup_pos_noisy.unsqueeze(1)
        ).view(self.num_envs, -1)

        # 7. fingertip binary contact (5D) — contact 자체에는 noise 없음
        binary_contact = self.binary_contact_buf.float()

        # 8. last actions (11D)
        last_actions = self.actions

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
            last_actions,           # 11
            self._bead_mass_normalized.unsqueeze(-1),  # 1 (0=빈 컵, 1=최대 하중)
        ], dim=-1)   # 107D

        if actor_obs.shape[1] != NUM_OBSERVATIONS:
            raise RuntimeError(
                f"[v8] Actor obs dim mismatch: {actor_obs.shape[1]} != {NUM_OBSERVATIONS}"
            )

        # ==== Critic extra obs (37D) — clean state 사용 ====
        # cup velocity (6D)
        cup_lin_vel = self.cup.data.root_lin_vel_w
        cup_ang_vel = self.cup.data.root_ang_vel_w

        # cup rotation (4D)
        cup_rot = self.object_rot

        # cup height delta (1D) — clean cup pos
        cup_height_delta = (
            cup_pos_clean[:, 2] - self.object_init_pos[:, 2]
        ).unsqueeze(1)

        # distal contact (5D binary + 5D norm)
        distal_binary     = self.distal_binary_contact_buf.float()
        distal_force_norm = (self.distal_contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)

        # middle contact (5D binary + 5D norm)
        middle_binary     = self.middle_binary_contact_buf.float()
        middle_force_norm = (self.middle_contact_force_raw / CONTACT_FORCE_MAX).clamp(0.0, 1.0)

        # phase step ratio (1D)
        phase_step_ratio = (
            self.episode_length_buf.float() / EPISODE_STEPS
        ).unsqueeze(1)

        # fingertip signed dist (5D) — clean positions
        tip_to_cup_dist = (
            fingertip_pos_clean - cup_pos_clean.unsqueeze(1)
        ).norm(dim=-1)
        fingertip_signed_dist = (tip_to_cup_dist - CUP_RADIUS_APPROX).unsqueeze(-1).squeeze(-1)

        # critic actor_obs_clean (107D) — clean state 재조합
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
            self._bead_mass_normalized.unsqueeze(-1),  # 1
        ], dim=-1)   # 107D

        critic_obs = torch.cat([
            actor_obs_clean,        # 107
            cup_lin_vel,            # 3
            cup_ang_vel,            # 3
            cup_rot,                # 4
            cup_height_delta,       # 1
            distal_binary,          # 5
            distal_force_norm,      # 5
            middle_binary,          # 5
            middle_force_norm,      # 5
            phase_step_ratio,       # 1
            fingertip_signed_dist,  # 5
        ], dim=-1)   # 144D

        if critic_obs.shape[1] != NUM_CRITIC_OBSERVATIONS:
            raise RuntimeError(
                f"[v8] Critic obs dim mismatch: {critic_obs.shape[1]} != {NUM_CRITIC_OBSERVATIONS}"
            )

        return {"policy": actor_obs, "critic": critic_obs}

    # ------------------------------------------------------------------
    # Rewards: R1~R5
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # ---- enclosure weight: ADR 대상 아님, 고정값 사용 ----
        enclosure_weight = self.cfg.enclosure_weight

        # ---- 공통 참조 ----
        cup_height_delta = (
            self.object_pos[:, 2] - self.object_init_pos[:, 2]
        ).clamp(min=0.0)   # (N,) ≥ 0

        # ---- R0. palm_to_cup: arm이 컵을 향해 접근하도록 유도 ----
        # 이 항이 없으면 초기 정책(출력≈0) → workspace 중심으로 이동 → cup에서 멀어짐
        # v1의 hand_to_object와 동일 역할: arm approach의 1차 gradient
        grasp_center_approach = self.object_pos.clone()
        grasp_center_approach[:, 2] += self.cfg.cup_grasp_z_offset
        palm_to_cup_dist = (self.palm_center_pos - grasp_center_approach).norm(dim=-1)   # (N,)
        r0_palm_approach = self.cfg.palm_approach_weight * torch.exp(
            -self.cfg.palm_approach_sharpness * palm_to_cup_dist
        )

        # ---- R1. fingertip_enclosure ----
        # 타겟을 접근 방향(approach)의 수직(perpendicular) 축으로 설정:
        #   thumb   → cup 중심에서 perp 방향 (측면)
        #   others  → cup 중심에서 -perp 방향 (반대 측면)
        # 이렇게 하면 엄지↔나머지가 컵을 양옆에서 감싸는 구조 → 토크 균형 → 컵 안 기울어짐
        # (approach-axis 타겟은 near/far 비대칭으로 토크 불균형 유발)
        grasp_center = grasp_center_approach

        cup_to_palm_xy = self.palm_center_pos[:, :2] - grasp_center[:, :2]   # (N, 2)
        approach_dir_xy = cup_to_palm_xy / cup_to_palm_xy.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-6)   # (N, 2)
        # approach에 수직인 방향 (XY 평면에서 90° 회전)
        perp_dir_xy = torch.stack(
            [-approach_dir_xy[:, 1], approach_dir_xy[:, 0]], dim=1
        )   # (N, 2)

        self._approach_dir_buf[:, :2] = perp_dir_xy
        self._approach_dir_buf[:, 2]  = 0.0

        r = self.cfg.cup_radius_approx
        thumb_target  = grasp_center + self._approach_dir_buf * r    # (N, 3)
        others_target = grasp_center - self._approach_dir_buf * r    # (N, 3)

        thumb_dist  = (self.fingertip_pos[:, 0, :] - thumb_target).norm(dim=-1)           # (N,)
        others_dist = (self.fingertip_pos[:, 1:, :] - others_target.unsqueeze(1)).norm(
            dim=-1
        ).mean(dim=-1)   # (N,)

        tw = self.cfg.enclosure_thumb_weight   # 0.6: 엄지 유도 비대칭 강화
        r1_enclosure = enclosure_weight * (
            tw * torch.exp(-self.cfg.enclosure_sharpness * thumb_dist)
            + (1.0 - tw) * torch.exp(-self.cfg.enclosure_sharpness * others_dist)
        )

        # ---- tip contact force (Cup-filtered, 실 Teosllo FT 센서 직접 대응) ----
        thumb_force      = self.contact_force_raw[:, 0]           # (N,) 엄지 tip 힘
        others_avg_force = self.contact_force_raw[:, 1:].mean(dim=-1)  # (N,) 나머지 평균 힘

        # ---- R1b. force_balance_reward: |F_thumb - F_others_avg| → 0 (컵 기울임 방지) ----
        # [contact_bonus 대체] binary count 제거 → 힘 균형 직접 측정
        # 물리적 근거: 컵 기울어짐 = 합력 불균형 → |F_thumb| ≈ |F_others_avg| = 합력 ≈ 0
        # gate: 양쪽 모두 접촉 시에만 활성 (무접촉 err=0 → 오보상 방지)
        has_thumb_contact  = self.binary_contact_buf[:, 0].float()              # (N,)
        has_others_contact = (self.binary_contact_buf[:, 1:].sum(-1) >= 1).float()  # (N,)
        balance_gate       = has_thumb_contact * has_others_contact             # (N,)
        force_balance_err  = (thumb_force - others_avg_force).abs()             # (N,) [N]
        r1b_force_balance = (
            self.cfg.force_balance_weight
            * balance_gate
            * torch.exp(-self.cfg.force_balance_sharpness * force_balance_err)
        )

        # ---- R1c. full_grasp_bonus: Grasp phase per-step ----
        # 조건: 엄지 contact AND 나머지 3개 이상 AND F_thumb >= F_others_avg × ratio_min
        # → 엄지가 0.1N 살짝 닿아도 통과하던 허점 제거 (force adequacy 조건)
        is_grasp_phase = ~self.is_lift_phase                                  # (N,) bool
        others_count = self.binary_contact_buf[:, 1:].sum(dim=-1)             # (N,) 0~4
        thumb_force_adequate = (
            thumb_force >= others_avg_force * self.cfg.thumb_force_ratio_min
        ).float()   # (N,)
        full_grasp_flag = (
            self.binary_contact_buf[:, 0] & (others_count >= 3)
        ).float() * thumb_force_adequate
        r1c_full_grasp = (
            self.cfg.full_grasp_bonus_weight * is_grasp_phase.float() * full_grasp_flag
        )

        # ---- R2. tip_approach_bonus ----
        # tip이 distal(rl_dg_*_4)보다 cup surface에 먼저 닿도록 유도
        if len(self.distal4_body_indices) == NUM_FINGERTIPS:
            # cup surface dist: ||pos - cup_center|| - cup_radius (clamp >= 0)
            tip_surf_dist    = (self.fingertip_pos - grasp_center.unsqueeze(1)).norm(dim=-1) - r
            distal_surf_dist = (self.distal4_pos   - grasp_center.unsqueeze(1)).norm(dim=-1) - r
            tip_lead = (distal_surf_dist - tip_surf_dist).clamp(min=0.0).mean(dim=-1)   # (N,)
            r2_tip_bonus = self.cfg.tip_approach_bonus_weight * tip_lead
        else:
            r2_tip_bonus = torch.zeros(self.num_envs, device=self.device)

        # ---- cup uprightness: 컵 Z축이 세계 Z축과 얼마나 일치하는지 ----
        # 1.0 = 완전히 수직, 0.0 = 수평 → R3/R5에 곱해 기울어진 채 들어올리면 보상 차단
        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cup_z_world = quat_apply(self.object_rot, z_local)
        cup_uprightness = cup_z_world[:, 2].clamp(min=0.0)   # (N,) ∈ [0, 1]

        # ---- R3. lift_reward: 선형 height delta × cup uprightness ----
        # 기울어진 채로 들어올리면 uprightness 낮아져 보상 감소 → 기울어진 파지 억제
        r3_lift = self.cfg.lift_reward_weight * cup_height_delta * cup_uprightness   # (N,)

        # ---- R4. action_smoothness ----
        palm_delta   = (self.actions[:, :6] - self.prev_actions[:, :6]).pow(2).sum(dim=-1)
        finger_delta = (self.actions[:, 6:] - self.prev_actions[:, 6:]).pow(2).sum(dim=-1)
        r4_smooth = (
            self.cfg.action_smoothness_palm_weight   * palm_delta
            + self.cfg.action_smoothness_finger_weight * finger_delta
        )   # (N,) — 음수

        # ---- R5. grasp_quality_lift ----
        # enclosure_quality × cup_height_delta × cup_uprightness → 수직으로 파지하며 들어야 큰 보상
        min_tip_dist = (self.fingertip_pos - grasp_center.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
        enclosure_quality = torch.exp(
            -self.cfg.grasp_quality_lift_sharpness * min_tip_dist
        ).clamp(max=1.0)   # (N,)
        r5_quality_lift = self.cfg.grasp_quality_lift_weight * cup_height_delta * enclosure_quality * cup_uprightness

        # ---- R6. grip_efficiency: bead 무게 대비 과도한 grip 패널티 ----
        # grip_normalized ∈ [0,1]: 0=open, 1=fully closed
        # over_grip > 0 일 때만 패널티 → bead가 많을수록 강한 grip 허용
        grip_normalized = (self.actions[:, 6:].mean(dim=-1) + 1.0) / 2.0   # (N,)
        over_grip = (grip_normalized - self._bead_mass_normalized).clamp(min=0.0)
        r6_grip_eff = -self.cfg.grip_efficiency_weight * over_grip * is_grasp_phase.float()

        # ---- 합산 ----
        total = (
            r0_palm_approach
            + r1_enclosure
            + r1b_force_balance
            + r1c_full_grasp
            + r2_tip_bonus
            + r3_lift
            + r4_smooth
            + r5_quality_lift
            + r6_grip_eff
        )

        # ---- ADR increment ----
        _ep_success_rate = self._successful_episodes / max(self._total_episodes, 1)
        if self.grasp_adr is not None:
            self.grasp_adr.maybe_increment(_ep_success_rate)

        # ---- 로깅 ----
        self.extras["palm_approach_reward"]  = r0_palm_approach.mean()
        self.extras["palm_to_cup_dist"]      = palm_to_cup_dist.mean()
        self.extras["enclosure_reward"]      = r1_enclosure.mean()
        # [force balance 핵심 지표] — 컵 기울어짐 진단
        self.extras["force_balance_reward"]  = r1b_force_balance.mean()
        self.extras["force_balance_err"]     = force_balance_err.mean()       # 0이면 완벽 균형
        self.extras["thumb_force_mean"]      = thumb_force.mean()
        self.extras["others_avg_force_mean"] = others_avg_force.mean()
        self.extras["full_grasp_bonus"]      = r1c_full_grasp.mean()
        self.extras["full_grasp_rate"]       = full_grasp_flag.mean()
        self.extras["thumb_force_adequate"]  = thumb_force_adequate.mean()
        self.extras["tip_approach_bonus"]    = r2_tip_bonus.mean()
        self.extras["grip_efficiency"]       = r6_grip_eff.mean()
        self.extras["grip_normalized"]       = grip_normalized.mean()
        self.extras["bead_mass_normalized"]  = self._bead_mass_normalized.mean()
        self.extras["lift_reward"]           = r3_lift.mean()
        self.extras["action_smoothness"]     = r4_smooth.mean()
        self.extras["grasp_quality_lift"]    = r5_quality_lift.mean()
        self.extras["cup_height_delta"]      = cup_height_delta.mean()
        self.extras["cup_uprightness"]       = cup_uprightness.mean()
        self.extras["num_contacts"]          = self.num_contacts_buf.float().mean()
        self.extras["success_rate"]          = self.success_flag.float().mean()
        self.extras["episode_success_rate"]  = torch.tensor(_ep_success_rate, device=self.device)
        self.extras["thumb_dist"]            = thumb_dist.mean()
        self.extras["others_dist"]           = others_dist.mean()
        if self.grasp_adr is not None:
            self.extras["adr_progress"]          = torch.tensor(self.grasp_adr.progress, device=self.device)
            self.extras["adr_spawn_xy_range"]    = torch.tensor(self.grasp_adr.get_param("spawn",  "object_spawn_xy_range"), device=self.device)
            self.extras["adr_obs_noise_cup_pos"] = torch.tensor(self.grasp_adr.get_param("noise",  "obs_noise_cup_pos"),     device=self.device)
            self.extras["adr_bead_count_max"]    = torch.tensor(self.grasp_adr.get_param("beads",  "bead_count_max"),        device=self.device)

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

        # success: lift phase 이후 + cup 들림 + contact 유지
        in_or_past_lift = (self.episode_length_buf >= LIFT_START_STEP)
        lifted  = self.object_pos[:, 2] > (self.object_init_pos[:, 2] + self.cfg.lift_success_height)
        grasped = (self.num_contacts_buf >= MIN_CONTACTS_FOR_SUCCESS)
        self.success_flag.copy_(in_or_past_lift & lifted & grasped)
        self.episode_success_buf |= self.success_flag   # 에피소드 중 한 번이라도 성공 시 True

        terminated = out_x | out_y | fallen | tipped | self.success_flag
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

        # Eval 기록: success_buf 클리어 전에 per-episode 데이터 저장
        for i, env_id in enumerate(env_ids):
            self._eval_records.append({
                "bead_mass": self._bead_mass_normalized[env_id].item(),
                "grip":      self._eval_grip_at_lift[env_id].item(),
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

        # ---- 3. 컵 spawn 위치 계산 (ADR: ±1cm → ±8cm) ----
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

        # Fabrics cspace attractor(null-space)를 pregrasp arm pos로 설정
        # default_config가 ARM_START_POSE이면 null-space 항이 계속 팔을 당겨 초기 흔들림 발생
        # pregrasp arm pos로 설정 → 에피소드 시작 시 null-space 항 ≈ 0 → 안정
        self.open_tesollo_fabric.default_config[env_ids, :NUM_ARM_DOF] = q_pregrasp[:, :NUM_ARM_DOF]

        prelift_arm = q_pregrasp[:, :NUM_ARM_DOF].clone()
        prelift_arm[:, 3] = (prelift_arm[:, 3] + 0.31).clamp(max=3.14)
        self.prelift_arm_pos_buf[env_ids] = prelift_arm

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

        # ---- 7b. Bead 스폰 (무게 도메인 랜덤화) ----
        _bead_max = int(
            self.grasp_adr.get_param("beads", "bead_count_max")
            if self.grasp_adr is not None
            else self.cfg.bead_count_max
        )
        bead_count = torch.randint(
            self.cfg.bead_count_min, _bead_max + 1,
            (n,), device=self.device
        )  # 각 env당 활성 bead 수 (0 ~ adr_bead_count_max)

        # 비활성 bead 숨김: ground plane(z=0) 위 2cm, 테이블 반대 방향 (x-0.8m)
        # z=-10 사용 시 ground plane에 튕겨 테이블 위에 흩어지는 문제 방지
        hide_pos = self.scene.env_origins[env_ids].clone()  # (n, 3) world
        hide_pos[:, 0] -= 0.8  # 테이블(local x≈+0.57) 반대쪽
        hide_pos[:, 2] = 0.02  # ground plane(z=0) 위 2cm
        bead_state = torch.zeros(n, self.cfg.num_beads, 13, device=self.device)
        bead_state[..., :3] = hide_pos.unsqueeze(1)  # 모든 bead → 숨김 위치
        bead_state[..., 3] = 1.0   # quat w

        # 활성 bead: 컵 내부 layered spiral (컵은 항상 upright이므로 quat_apply 불필요)
        for bi in range(self.cfg.num_beads):
            active = bead_count > bi   # (n,) bool
            if active.any():
                bead_pos = obj_pos_world + self._bead_offsets_b[bi].unsqueeze(0)  # (n, 3)
                bead_state[active, bi, :3] = bead_pos[active]
                bead_state[active, bi, 3]  = 1.0

        self.beads.write_object_state_to_sim(bead_state, env_ids=env_ids)

        # obs용 정규화 무게 업데이트
        self._bead_mass_normalized[env_ids] = bead_count.float() / self.cfg.bead_count_max

        # ---- 8. 버퍼 리셋 ----
        self.hand_joint_targets[env_ids] = approach_hand
        self.contact_force_raw[env_ids].zero_()
        self.binary_contact_buf[env_ids] = False
        self.num_contacts_buf[env_ids]   = 0
        self.distal_contact_force_raw[env_ids].zero_()
        self.distal_binary_contact_buf[env_ids] = False
        self.middle_contact_force_raw[env_ids].zero_()
        self.middle_binary_contact_buf[env_ids] = False
        self.success_flag[env_ids] = False

        # actions 리셋: delta action 방식 → action=0 = pregrasp 위치
        # (역스케일 불필요: scale(0, delta_mins, delta_maxs) = delta=0 → pregrasp 유지)
        self.actions[env_ids, :6] = 0.0
        self.actions[env_ids, 6:] = -1.0
        self.prev_actions[env_ids, :6] = 0.0
        self.prev_actions[env_ids, 6:] = -1.0
