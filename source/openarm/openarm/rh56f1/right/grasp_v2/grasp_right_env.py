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

"""환경 클래스: 5g_grasp_right_v1

v7: Fabrics 팔 학습 + per-finger lerp 5D + Contact sensor 없는 FK 기반 근접도 리워드

핵심 개선 (v1/v6 대비):
  - v1 문제: fabric_q/qd obs → sim2real 불가, palm_dist 기반 자동 닫힘 → 충돌 충격
  - v6 문제: 팔 고정 → cup 위치 오차 대응 불가, per-finger 5D 협응 학습 부족

Action (11D):
  [0:6]  6D palm pose → Fabrics IK → arm 7 DOF (학습, cup 위치 오차 대응)
  [6:11] 5D per-finger lerp: -1 → HAND_APPROACH_POSE, +1 → HAND_GRASP_POSE

Episode (10s @ 60Hz):
  Grasp phase   (0~479): Fabrics arm + per-finger 정책
  Lift-wait phase (480~599): scripted joint7-only lift-wait + frozen hand
"""

from __future__ import annotations

import json
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
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_mul

from openarm.common.grasp_logging import action_policy_scalars

from fabrics_sim.fabrics.openarm_rh56f1_pose_fabric import OpenArmRh56f1PoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

from .grasp_right_env_cfg import GraspRightEnvCfg
from .grasp_adr import GraspADR
from .grasp_right_constants import (
    NUM_ARM_DOF,
    NUM_HAND_DOF,
    NUM_ROBOT_DOF,
    NUM_FINGERTIPS,
    NUM_OBS_BASE,
    NUM_DISTAL_SENSORS,
    NUM_MIDDLE_SENSORS,
    NUM_CRITIC_OBS_BASE,
    GRASP_PHASE_STEPS,
    LIFT_START_STEP,
    CONTACT_FORCE_THRESHOLD,
    CONTACT_FORCE_MAX,
    MIN_CONTACTS_FOR_SUCCESS,
    PREGRASP_FABRICS_STEPS,
    ARM_START_POSE,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
)
from .grasp_right_preset import (
    LEFT_ARM_REST_JOINT_POS,
    RIGHT_ACTUATED_JOINT_NAMES,
    RIGHT_HAND_MIMIC_JOINT_NAMES,
    HAND_APPROACH_POSE,
    HAND_GRASP_POSE,
    HAND_FULL_GRIP_POSE,
)
from .finger_action_utils import compute_grasp_finger_targets, compute_lift_finger_targets
from .grasp_right_utils import (
    compute_joint7_lift_wait_target,
    scale,
    to_torch,
)
from .demo_grasp_reset import DemoGraspResetBank, compute_demo_cup_spawn_local
from .warm_state_cache import GraspWarmStateCache, compute_arm_joint_match


class GraspRightEnv(DirectRLEnv):
    """OpenArm+Teosllo 오른손 다물체 파지 환경 (DEXTRAH 구조).

    Action: 11D
      [0:6]  palm pose (x,y,z,ez,ey,ex), 정규화 [-1,1] → Fabrics IK
      [6:11] per-finger 폐쇄 속도 명령 (thumb,index,middle,ring,pinky)
             접촉-게이트 적응 폐쇄: APPROACH → FULL_GRIP

    Episode (단일 phase, DEXTRAH):
      settle (step 0~24):  물체 drop-settle, 손가락 폐쇄 억제.
                           종료 시 안착점 스냅샷 → object_init_pos/goal 확정
      정책 제어 (step 25~599): Fabrics arm + 손가락 연속 제어.
                           reward = DEXTRAH 4항(hand_to_object/object_to_goal/
                           finger_curl_reg/lift), success = |obj-goal| < tol
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

        # RH56F1 원위(mimic) 관절: PhysxMimicJoint 미결합이라 능동 curl 필요(미구동 시
        # 원위가 approach에 고정 → fingertip pinch만 되고 감싸기 불가, lstm_test1 검증).
        # drive(hand 6) 순서 [thumb_1,thumb_2,index_1,middle_1,ring_1,pinky_1] 기준 src/mult.
        self._hand_mimic_dof_indices = [
            self.robot.joint_names.index(name) for name in RIGHT_HAND_MIMIC_JOINT_NAMES
        ]
        self._hand_mimic_src_idx = torch.tensor([1, 1, 2, 3, 4, 5], device=self.device)
        self._hand_mimic_mult = torch.tensor(
            [1.1425, 1.1425 * 0.7508, 1.1169, 1.1169, 1.1169, 1.1169], device=self.device
        )

        # body indices (robot.data.body_pos_w 참조용). RH56F1: 말단 손가락 링크 = fingertip 센서 body.
        # RH56F1 손은 2-마디 언더액추에이티드(엄지 4마디 + 나머지 _1,_2) → tesollo *_tip 미존재.
        # 말단 링크: thumb_4, {index,middle,ring,pinky}_2 (preset FINGERTIP_SENSOR_BODIES 정합).
        _FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
        _tip_names = ["r_hl_thumb_4", "r_hl_index_2", "r_hl_middle_2", "r_hl_ring_2", "r_hl_pinky_2"]
        self.fingertip_body_indices: list[int] = [
            self.robot.data.body_names.index(name) for name in _tip_names
        ]
        _palm_name = "r_hl_palm_sensor"
        self.palm_body_index: int = (
            self.robot.data.body_names.index(_palm_name)
            if _palm_name in self.robot.data.body_names
            else -1
        )
        # DEXTRAH hand bodies (palm + 5 tips) — critic hand_vel/forces 용
        self._hand_body_indices: list[int] = [self.palm_body_index] + self.fingertip_body_indices
        # distal phalanx body indices — RH56F1은 말단이 곧 tip(더 깊은 마디 없음) → tip 링크 재사용.
        _distal4_names = _tip_names
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
        self.demo_grasp_reset_bank = (
            DemoGraspResetBank.from_hdf5_paths(cfg.demo_grasp_pose_paths, device=self.device)
            if cfg.enable_demo_grasp_reset
            else None
        )
        # env 별로 마지막으로 배정된 demo 인덱스 (-1 = 미배정)
        self._env_assigned_demo_idx = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )

        # ----------------------------------------------------------------
        # Hand poses (per-finger lerp용)
        # open_pose = HAND_APPROACH_POSE (action=-1), grasp_pose = HAND_GRASP_POSE (action=+1)
        # ----------------------------------------------------------------
        self.hand_open_pose      = to_torch(HAND_APPROACH_POSE, device=self.device)
        self.hand_grasp_pose     = to_torch(HAND_GRASP_POSE, device=self.device)
        self.hand_full_grip_pose = to_torch(HAND_FULL_GRIP_POSE, device=self.device)
        hand_limits = self.robot.data.soft_joint_pos_limits[0, self.hand_dof_indices, :]
        self.hand_joint_lower_limits = hand_limits[:, 0].contiguous()
        self.hand_joint_upper_limits = hand_limits[:, 1].contiguous()

        # ----------------------------------------------------------------
        # 로봇 시작 자세 (arm: ARM_START_POSE, hand: HAND_APPROACH_POSE)
        # ----------------------------------------------------------------
        arm_start  = to_torch(ARM_START_POSE,    device=self.device)  # (7,)
        hand_start = to_torch(HAND_APPROACH_POSE, device=self.device)  # (6,)
        robot_start = torch.cat([arm_start, hand_start], dim=0)         # (13,) 우측 팔+손
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
        # 목표 위치 — DEXTRAH식 고정 절대점 (cfg.object_goal_pos, 에피소드 불변)
        # ----------------------------------------------------------------
        self.object_goal = (
            to_torch(list(cfg.object_goal_pos), device=self.device)
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
        # per-object 로깅: MultiAsset(random_choice=False)는 env_id % N 로 물체 배정.
        self._object_names = list(self.cfg.active_object_names)
        self.object_idx = (
            torch.arange(self.num_envs, device=self.device) % len(self._object_names)
        )
        # DEXTRAH 물체 조건화: one-hot object id + scale (obs 구조 원본 동일, distillation 대비)
        self.multi_object_idx_onehot = torch.nn.functional.one_hot(
            self.object_idx, num_classes=len(self._object_names)
        ).float()   # (num_envs, N_obj), reset 불변
        # object_scale: 자리 유지(원본은 spawn 시 랜덤 스케일, 우리는 고정 1.0)
        self.object_scale = torch.ones(self.num_envs, 1, device=self.device)
        # DEXTRAH 관측 노이즈: per-env bias (reset 시 ADR 크기로 재샘플) + per-step uniform
        self.robot_joint_pos_bias = torch.zeros(self.num_envs, NUM_ARM_DOF + NUM_HAND_DOF, device=self.device)
        self.robot_joint_vel_bias = torch.zeros(self.num_envs, NUM_ARM_DOF + NUM_HAND_DOF, device=self.device)
        self.object_pos_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_rot_bias = torch.zeros(self.num_envs, 4, device=self.device)
        # 파지력 확보: 외란 wrench 버퍼 + 물체 질량 (DEXTRAH apply_object_wrench)
        self.object_applied_force  = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.object_applied_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._cup_mass = self.cup.root_physx_view.get_masses().to(self.device)  # (N, 1)
        self.palm_center_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.fingertip_pos   = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.distal4_pos     = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.actions         = torch.zeros(self.num_envs, cfg.num_actions, device=self.device)
        self.prev_actions    = torch.full((self.num_envs, cfg.num_actions), 0.0, device=self.device)

        # ----------------------------------------------------------------
        # Pregrasp / Lift 버퍼 (reset에서 계산)
        # ----------------------------------------------------------------
        self.pregrasp_arm_pos_buf      = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        # prelift: warm-state export(cfg 게이트, 기본 off) 전용 잔존 버퍼
        self.prelift_arm_pos_buf       = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)

        # ----------------------------------------------------------------
        # Hand joint targets (per-finger lerp 결과)
        # ----------------------------------------------------------------
        self.hand_joint_targets = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        # 관절별 접촉-게이트 적응 폐쇄: 관절당 폐쇄 진행도 [0,1] (N,6 RH56F1)
        # (PIP@middle, DIP@distal|tip 동결, MCP 무게이트 full close)
        self.finger_close_buf = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)

        # ----------------------------------------------------------------
        # 접촉 상태 버퍼
        # ----------------------------------------------------------------
        self.contact_force_xyz_raw = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.contact_force_raw     = torch.zeros(self.num_envs, NUM_FINGERTIPS, device=self.device)
        self.binary_contact_buf    = torch.zeros(self.num_envs, NUM_FINGERTIPS, dtype=torch.bool, device=self.device)
        self.num_contacts_buf      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.warm_contact_stable_steps_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.lift_wait_match_hold_steps_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

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
        # DEXTRAH success: 물체가 goal 반경(object_goal_tol) 내 (in_success_region)
        self.in_success_region = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.transfer_entry_grasp_success_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._total_episodes: int = 0
        self._successful_episodes: int = 0
        # 물체별 성공 집계(episode_success_rate/{object}): 어떤 물체가 성공/실패하는지 진단
        _n_obj = len(self._object_names)
        self._obj_total_episodes = torch.zeros(_n_obj, device=self.device)
        self._obj_success_episodes = torch.zeros(_n_obj, device=self.device)
        # warm export diagnostics
        self._warm_diag_step: int = 0
        self._warm_diag_terminated_early: int = 0  # lift-wait phase 중 early termination 횟수

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

        # cspace attractor: 오른손[7:13]만 grasp pose 방향, 좌측[13:26]은 중립 유지
        cspace_default = self.fabric.default_config.clone()  # 26D [r_arm7,r_hand6,l_arm7,l_hand6]
        cspace_default[:, NUM_ARM_DOF:NUM_ROBOT_DOF] = self.hand_grasp_pose.unsqueeze(0).expand(self.num_envs, -1)
        self.fabric.default_config.copy_(cspace_default)

        # 초기 액션: 0 → palm pose workspace 중심 (접근 자세 유지)
        self.actions.zero_()

    # ------------------------------------------------------------------
    # Scene 설정
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cup   = RigidObject(self.cfg.cup_cfg)
        self.table = RigidObject(self.cfg.table_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cup"]   = self.cup
        self.scene.rigid_objects["table"] = self.table

        # Actor: fingertip 개별 ContactSensor.
        # grasp_v2: MultiAsset(replicate_physics=False)에서 filter_prim_paths_expr(force_matrix_w)는
        # GPU 미지원 → contact 0. filter 제거하고 net_forces_w(접촉 여부, 물체 구분 없음, GPU 지원)로
        # 접촉을 판정한다. synergy 게이트는 "닿았나"만 필요하므로 물체 구분 불필요.
        self._tip_sensors: list[ContactSensor] = []
        for link_name in self.cfg.right_tip_contact_links:
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link_name}",
                history_length=1,
                track_air_time=False,
            ))
            self._tip_sensors.append(sensor)
            self.scene.sensors[f"tip_sensor_{link_name}"] = sensor

        # distal/middle 도 손가락별 개별 센서. net_forces_w[:, 0, :] 로 읽는다.
        # RH56F1 2-마디 손: distal = 말단 링크(tip과 동일), middle = 근위 knuckle 링크(envelope).
        #   distal(말단): thumb_4, {index,middle,ring,pinky}_2
        #   middle(근위): thumb_2, {index,middle,ring,pinky}_1
        _DISTAL_LINKS = ["r_hl_thumb_4", "r_hl_index_2", "r_hl_middle_2", "r_hl_ring_2", "r_hl_pinky_2"]
        _MIDDLE_LINKS = ["r_hl_thumb_2", "r_hl_index_1", "r_hl_middle_1", "r_hl_ring_1", "r_hl_pinky_1"]
        self._distal_sensors: list[ContactSensor] = []
        for i, link in enumerate(_DISTAL_LINKS):
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link}",
                history_length=1,
                track_air_time=False,
            ))
            self._distal_sensors.append(sensor)
            self.scene.sensors[f"distal_sensor_{i + 1}"] = sensor

        self._middle_sensors: list[ContactSensor] = []
        for i, link in enumerate(_MIDDLE_LINKS):
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link}",
                history_length=1,
                track_air_time=False,
            ))
            self._middle_sensors.append(sensor)
            self.scene.sensors[f"middle_sensor_{i + 1}"] = sensor

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
        self.fabric = OpenArmRh56f1PoseFabric(
            self.num_envs, self.device, self.timestep,
            graph_capturable=False,
            use_hand_fabric=False,
        )
        num_joints = self.fabric.num_joints   # 26 (bi-arm: r_arm7,r_hand6,l_arm7,l_hand6)

        self.fabric_integrator = DisplacementIntegrator(self.fabric)

        # Fabric 상태 버퍼 (26D). 우측[0:13]=팔+손 시작자세, 좌측[13:26]=default 중립 유지.
        self.fabric_q   = self.fabric.default_config.clone().contiguous()
        self.fabric_q[:, :NUM_ROBOT_DOF] = self.robot_start_joint_pos
        self.fabric_qd  = torch.zeros(self.num_envs, num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, num_joints, device=self.device)

        # Fabric input 버퍼
        self.hand_pca_targets  = torch.zeros(self.num_envs, 5, device=self.device)
        self.palm_pose_targets = torch.zeros(self.num_envs, 6, device=self.device)
        self.fabric_damping_gain = self.cfg.fabrics_damping_gain * torch.ones(self.num_envs, 1, device=self.device)

        # Reset 전용 소형 Fabrics (chunk 단위)
        self._reset_chunk = self.cfg.reset_fabric_chunk_size
        self._reset_fabric = OpenArmRh56f1PoseFabric(
            self._reset_chunk, self.device, self.timestep,
            graph_capturable=False,
            use_hand_fabric=False,
        )
        self._reset_integrator = DisplacementIntegrator(self._reset_fabric)

        reset_cspace = self._reset_fabric.default_config.clone()  # 26D, 좌측 중립 유지
        reset_cspace[:, NUM_ARM_DOF:NUM_ROBOT_DOF] = self.hand_grasp_pose.unsqueeze(0).expand(self._reset_chunk, -1)
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
        if self.cfg.cache_pregrasp_reset and self.demo_grasp_reset_bank is None:
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
        # RH56F1 fabric palm attractor = r_hl_palm_sensor 직접 제어. side grasp:
        # (ez,ey,ex)=(180,0,90) → palm_sensor +z(법선) = world +y (물체 방향). 팔의 자연
        # 수평자세와 일치 (tesollo 규약 ez=90은 +x 목표라 도달 못 하고 47° 기울어 정착).
        palm[:, 3] = math.radians(180.0)
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

        # reset_fabric 은 26D cspace. q_init 이 우측 13D 로 오면 좌측 중립(default[13:26])을
        # 붙여 26D 로 확장하고, 반환은 다시 우측 13D 로 슬라이스한다(호출부 계약 유지).
        right_only = q_init.shape[1] == NUM_ROBOT_DOF
        if right_only:
            left_neutral = self._reset_fabric.default_config[0, NUM_ROBOT_DOF:].unsqueeze(0)
            q_init = torch.cat([q_init, left_neutral.expand(q_init.shape[0], -1)], dim=1)

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

        return q_out[:, :NUM_ROBOT_DOF] if right_only else q_out

    # ------------------------------------------------------------------
    # 접촉력 업데이트
    # ------------------------------------------------------------------
    def _update_contact_forces(self) -> None:
        # Actor: fingertip 개별 센서. net_forces_w[:, 0, :] = 링크 net contact force
        # (filter 없음 → 물체 구분 없이 접촉 여부만; MultiAsset/GPU 호환).
        tip_xyz = torch.stack([
            s.data.net_forces_w[:, 0, :] for s in self._tip_sensors
        ], dim=1)   # (N, 5, 3)
        tip_norms = tip_xyz.norm(dim=-1)   # (N, 5)

        self.contact_force_xyz_raw.copy_(tip_xyz)
        self.contact_force_raw.copy_(tip_norms)
        self.binary_contact_buf.copy_(tip_norms > CONTACT_FORCE_THRESHOLD)
        self.num_contacts_buf.copy_(self.binary_contact_buf.sum(dim=-1).long())

        # Critic: distal (손가락별 개별 센서 net_forces_w[:, 0, :])
        per_distal = torch.stack([
            s.data.net_forces_w[:, 0, :] for s in self._distal_sensors
        ], dim=1).norm(dim=-1)   # (N, 5)
        self.distal_contact_force_raw.copy_(per_distal)
        self.distal_binary_contact_buf.copy_(per_distal > CONTACT_FORCE_THRESHOLD)

        # Critic: middle (손가락별 개별 센서 net_forces_w[:, 0, :])
        per_middle = torch.stack([
            s.data.net_forces_w[:, 0, :] for s in self._middle_sensors
        ], dim=1).norm(dim=-1)   # (N, 5)
        self.middle_contact_force_raw.copy_(per_middle)
        self.middle_binary_contact_buf.copy_(per_middle > CONTACT_FORCE_THRESHOLD)

    # ------------------------------------------------------------------
    # 파지력 확보: 물체 외란 wrench (DEXTRAH apply_object_wrench 이식)
    # ------------------------------------------------------------------
    def _apply_object_wrench(self) -> None:
        # DEXTRAH 원본 정렬: 물체가 goal 도달(in_success_region) 시 외란 인가
        # → 리프트 학습을 방해하지 않으면서 goal 유지 파지력(robust hold)을 학습.
        # 크기는 ADR 커리큘럼 0→10 (원본 object_wrench.max_linear_accel)
        apply = self.in_success_region.view(-1, 1, 1)
        # trigger_every step 마다 새 랜덤 wrench (그 사이 유지)
        new_trig = (
            (self.episode_length_buf % int(self.cfg.wrench_trigger_every)) == 0
        ).view(-1, 1, 1)
        max_accel = (
            self.grasp_adr.get_param("object_wrench", "max_linear_accel")
            if self.grasp_adr is not None else float(self.cfg.wrench_max_accel)
        )
        accel = max_accel * torch.rand(
            self.num_envs, 1, 1, device=self.device
        )
        fmag = accel * self._cup_mass.view(-1, 1, 1)                       # F = m·a
        tmag = fmag * float(self.cfg.wrench_torsional_radius)              # τ = m·a·r

        def _rand_dir() -> torch.Tensor:
            d = torch.randn(self.num_envs, 1, 3, device=self.device)
            return d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        f = fmag * _rand_dir()
        t = tmag * _rand_dir()
        self.object_applied_force = torch.where(new_trig, f, self.object_applied_force)
        self.object_applied_torque = torch.where(new_trig, t, self.object_applied_torque)
        # 파지 전(grip<1) env 는 wrench 0
        self.object_applied_force = torch.where(
            apply, self.object_applied_force, torch.zeros_like(self.object_applied_force)
        )
        self.object_applied_torque = torch.where(
            apply, self.object_applied_torque, torch.zeros_like(self.object_applied_torque)
        )
        self.cup.set_external_force_and_torque(
            self.object_applied_force, self.object_applied_torque
        )

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self.cfg.wrench_enable:
            self._apply_object_wrench()
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone()

        palm_action   = actions[:, :6]    # (N, 6) ∈ [-1, 1]
        finger_action = actions[:, 6:11]  # (N, 5) ∈ [-1, 1]

        # ---- settle 종료 시 안착 스냅샷 → object_height 로깅 baseline 확정 ----
        # (goal 은 DEXTRAH식 고정 절대점이라 갱신 없음)
        snap = self.episode_length_buf == int(self.cfg.settle_steps)
        if snap.any():
            self.object_init_pos[snap] = self.object_pos[snap]

        # ---- Fabrics arm 제어 (전 구간 정책 연속 제어, scripted lift 없음) ----
        # Delta action: action=0 → pregrasp 유지, action=±1 → pregrasp ± delta
        # 절대 workspace(palm_mins/maxs)로 클램프하여 안전 영역 보장
        delta = scale(palm_action, self.delta_mins, self.delta_maxs)   # (N, 6)
        palm_pose = self.pregrasp_palm_pose_buf + delta
        palm_mins = torch.minimum(self.palm_mins.unsqueeze(0), self.pregrasp_palm_pose_buf)
        palm_maxs = torch.maximum(self.palm_maxs.unsqueeze(0), self.pregrasp_palm_pose_buf)
        palm_pose = torch.max(torch.min(palm_pose, palm_maxs), palm_mins)
        self.palm_pose_targets.copy_(palm_pose)
        self.hand_pca_targets.zero_()

        # fabric cspace damping: ADR 커리큘럼 10→20 (DEXTRAH fabric_damping.gain)
        if self.grasp_adr is not None:
            self.fabric_damping_gain.fill_(
                self.grasp_adr.get_param("fabric_damping", "gain")
            )

        self.fabric.set_features(
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
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.fabric_integrator.step(
                self.fabric_q.detach(),
                self.fabric_qd.detach(),
                self.fabric_qdd.detach(),
                self.timestep,
            )

        # ---- 손가락별 접촉-게이트 적응 폐쇄 (RH56F1 6-DOF underactuated wrap) ----
        # RH56F1 손 관절(6): [thumb_1, thumb_2, index_1, middle_1, ring_1, little_1].
        # 5D action = 손가락별 폐쇄 속도 명령[0,1] → 관절 매핑 counts=[2,1,1,1,1](엄지만 2관절).
        # tesollo 4-마디 phalanx gate(_3/_4)는 RH56F1 2-마디엔 부재 → 손가락 단위 gate로 단순화.
        cmd = 0.5 * (finger_action.clamp(-1.0, 1.0) + 1.0)          # (N,5) ∈ [0,1]
        # 다물체 drop-settle: episode 초기 settle_steps 동안 손가락 폐쇄 억제 →
        # 물체(DEXTRAH식 고정 높이 spawn)가 낙하해 테이블에 안착(grasp_v1 정지물체 전제).
        in_settle = (
            self.episode_length_buf < int(self.cfg.settle_steps)
        ).unsqueeze(-1)
        cmd = torch.where(in_settle, torch.zeros_like(cmd), cmd)
        # 손가락 단위 동결(freeze_enable 시): tip|middle 접촉하면 해당 손가락 조임 정지.
        if self.cfg.synergy_freeze_enable:
            finger_gate = (
                self.binary_contact_buf.float()
                + self.middle_binary_contact_buf.float()
            ).clamp(max=1.0)                                        # (N,5)
        else:
            # 동결 제거: 손가락이 물체를 계속 조임(물리 collision이 관통/형상적응 담당) → 파지력.
            finger_gate = torch.zeros_like(cmd)                     # (N,5)
        counts = torch.tensor([2, 1, 1, 1, 1], device=self.device)  # thumb 2관절, 나머지 1관절
        gate6 = finger_gate.repeat_interleave(counts, dim=1)        # (N,6)
        cmd6  = cmd.repeat_interleave(counts, dim=1)                # (N,6) 손가락 명령 → 관절
        advance = float(self.cfg.finger_close_speed) * cmd6 * (1.0 - gate6)
        self.finger_close_buf = (self.finger_close_buf + advance).clamp(0.0, 1.0)  # (N,6)
        hand_target = torch.lerp(
            self.hand_open_pose.unsqueeze(0).expand(self.num_envs, -1),
            self.hand_full_grip_pose.unsqueeze(0).expand(self.num_envs, -1),
            self.finger_close_buf,                                  # (N,6) 관절별 진행도
        ).clamp(
            self.hand_joint_lower_limits.unsqueeze(0),
            self.hand_joint_upper_limits.unsqueeze(0),
        )
        self.hand_joint_targets.copy_(hand_target)

        # fabric_q 오른손[7:13] 부분 동기화 (FK 계산에 활용). 좌측[13:26]은 중립 유지.
        self.fabric_q[:, NUM_ARM_DOF:NUM_ROBOT_DOF] = hand_target
        self.fabric_qd[:, NUM_ARM_DOF:NUM_ROBOT_DOF].zero_()

    def _apply_action(self) -> None:
        # ---- 오른팔: 전 구간 Fabrics arm target (DEXTRAH 단일 phase) ----
        arm_target = self.fabric_q[:, :NUM_ARM_DOF]

        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(arm_target), joint_ids=self.arm_dof_indices
        )

        # ---- 오른손 ----
        # Both phases use policy-controlled absolute synergy targets.
        finger_target = self.hand_joint_targets
        self.robot.set_joint_position_target(finger_target, joint_ids=self.hand_dof_indices)
        # 원위(mimic) 마디: drive × mult 로 능동 curl (하드웨어 결합 재현 — 미구동 시
        # 원위 고정 → 감싸기 불가, RH56F1 전용 plumbing).
        _mimic_target = (
            finger_target[:, self._hand_mimic_src_idx] * self._hand_mimic_mult.unsqueeze(0)
        )
        self.robot.set_joint_position_target(
            _mimic_target, joint_ids=self._hand_mimic_dof_indices
        )
        self.robot.set_joint_velocity_target(
            torch.zeros_like(finger_target), joint_ids=self.hand_dof_indices
        )

        # ---- 왼팔: 고정 자세 ----
        self.robot.set_joint_position_target(
            self.left_arm_zero_pos, joint_ids=self.left_arm_dof_indices
        )

    # ------------------------------------------------------------------
    # Intermediate values (DEXTRAH _compute_intermediate_values 정렬)
    # ------------------------------------------------------------------
    def _adr(self, group: str, name: str, default: float = 0.0) -> float:
        if self.grasp_adr is not None:
            return self.grasp_adr.get_param(group, name)
        return default

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

        # ==== DEXTRAH noisy states (per-step uniform noise + per-episode bias) ====
        self.robot_dof_pos = self.robot.data.joint_pos[:, self.actuated_dof_indices]
        self.robot_dof_vel = self.robot.data.joint_vel[:, self.actuated_dof_indices]
        _pos_w = self._adr("robot_state_noise", "robot_joint_pos_noise")
        _vel_w = self._adr("robot_state_noise", "robot_joint_vel_noise")
        self.robot_dof_pos_noisy = (
            self.robot_dof_pos
            + _pos_w * 2.0 * (torch.rand_like(self.robot_dof_pos) - 0.5)
            + self.robot_joint_pos_bias
        )
        self.robot_dof_vel_noisy = (
            self.robot_dof_vel
            + _vel_w * 2.0 * (torch.rand_like(self.robot_dof_vel) - 0.5)
            + self.robot_joint_vel_bias
        )
        # velocity obs annealing (DEXTRAH teacher: coefficient 0 = vel obs 상시 차단)
        _anneal = self._adr("observation_annealing", "coefficient")
        self.robot_dof_vel_noisy = self.robot_dof_vel_noisy * _anneal

        # hand points (palm + 5 tips): fabric FK on noisy q → noisy hand pos/vel.
        # RH56F1 fabric taskmap 은 26D cspace 입력 필요 → 우측 noisy 13D + 좌측 중립 13D 로 확장.
        # (좌측은 고정이므로 velocity 0. Jacobian 도 26D → vel 도 26D 로 확장해 bmm.)
        _left_neutral = self.fabric.default_config[:, NUM_ROBOT_DOF:]                    # (N, 13)
        _q_noisy_full  = torch.cat([self.robot_dof_pos_noisy, _left_neutral], dim=1)     # (N, 26)
        _qd_noisy_full = torch.cat(
            [self.robot_dof_vel_noisy, torch.zeros_like(_left_neutral)], dim=1
        )                                                                               # (N, 26)
        _palm_pts, _palm_jac = self.fabric.get_taskmap("palm")(_q_noisy_full, None)
        _tip_pts, _tip_jac = self.fabric._fingertip_taskmap(_q_noisy_full, None)
        self.hand_pos_noisy = torch.cat([_palm_pts[:, :3], _tip_pts], dim=-1)  # (N, 18)
        _vel_palm = torch.bmm(
            _palm_jac[:, :3, :], _qd_noisy_full.unsqueeze(2)
        ).squeeze(2)
        _vel_tips = torch.bmm(
            _tip_jac, _qd_noisy_full.unsqueeze(2)
        ).squeeze(2)
        self.hand_vel_noisy = torch.cat([_vel_palm, _vel_tips], dim=-1) * _anneal  # (N, 18)

        # object noisy pose
        _op_w = self._adr("object_state_noise", "object_pos_noise")
        _or_w = self._adr("object_state_noise", "object_rot_noise")
        self.object_pos_noisy = (
            self.object_pos
            + _op_w * 2.0 * (torch.rand_like(self.object_pos) - 0.5)
            + self.object_pos_bias
        )
        self.object_rot_noisy = (
            self.object_rot
            + _or_w * 2.0 * (torch.rand_like(self.object_rot) - 0.5)
            + self.object_rot_bias
        )

        # 접촉력 업데이트
        self._update_contact_forces()

    # ------------------------------------------------------------------
    # Observations: DEXTRAH teacher 구조
    #   policy 193+N_obj | critic 247+N_obj (distillation 대비 원본 동일)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # ==== policy obs (DEXTRAH compute_policy_observations) ====
        actor_obs = torch.cat([
            self.robot_dof_pos_noisy,        # 13 (arm7+hand6)
            self.robot_dof_vel_noisy,        # 13 (annealing=0 → 상시 0)
            self.hand_pos_noisy,             # 18 (fabric FK: palm+5tip)
            self.hand_vel_noisy,             # 18 (0)
            self.object_pos_noisy,           # 3
            self.object_rot_noisy,           # 4
            self.object_goal,                # 3 (고정 절대점)
            self.multi_object_idx_onehot,    # N_obj
            self.object_scale,               # 1
            self.actions,                    # 11
            self.fabric_q[:, :NUM_ROBOT_DOF],    # 13 (우측 팔+손, 좌측 제외)
            self.fabric_qd[:, :NUM_ROBOT_DOF],   # 13
            self.fabric_qdd[:, :NUM_ROBOT_DOF],  # 13
        ], dim=-1)   # 123 + N_obj

        if actor_obs.shape[1] != self.cfg.observation_space:
            raise RuntimeError(
                f"Actor obs dim mismatch: {actor_obs.shape[1]} != {self.cfg.observation_space}"
            )

        # ==== critic obs (DEXTRAH compute_critic_observations, privileged) ====
        hand_pos_clean = torch.cat([
            self.palm_center_pos.unsqueeze(1), self.fingertip_pos
        ], dim=1).view(self.num_envs, -1)                              # 18
        hand_vel_clean = self.robot.data.body_vel_w[
            :, self._hand_body_indices, :
        ].reshape(self.num_envs, -1)                                   # 36
        hand_forces = self.robot.root_physx_view.get_link_incoming_joint_force()[
            :, self._hand_body_indices
        ].reshape(self.num_envs, -1)[:, :3]                            # 3 (DEXTRAH 원본 [:, :3])
        measured_joint_torque = self.robot.root_physx_view.get_dof_projected_joint_forces()[
            :, self.actuated_dof_indices
        ]                                                              # 27

        critic_obs = torch.cat([
            self.robot_dof_pos,              # 13
            self.robot_dof_vel,              # 13
            hand_pos_clean,                  # 18
            hand_vel_clean,                  # 36
            hand_forces,                     # 3
            measured_joint_torque,           # 13
            self.object_pos,                 # 3
            self.object_rot,                 # 4
            self.cup.data.root_vel_w,        # 6
            self.object_goal,                # 3
            self.multi_object_idx_onehot,    # N_obj
            self.object_scale,               # 1
            self.actions,                    # 11
            self.fabric_q[:, :NUM_ROBOT_DOF],    # 13
            self.fabric_qd[:, :NUM_ROBOT_DOF],   # 13
            self.fabric_qdd[:, :NUM_ROBOT_DOF],  # 13
        ], dim=-1)   # 163 + N_obj

        if critic_obs.shape[1] != self.cfg.state_space:
            raise RuntimeError(
                f"Critic obs dim mismatch: {critic_obs.shape[1]} != {self.cfg.state_space}"
            )

        return {"policy": actor_obs, "critic": critic_obs}

    # ------------------------------------------------------------------
    # Rewards: DEXTRAH 4항 (dextrah_kuka_allegro compute_rewards 이식)
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # 1) hand_to_object: palm+5손끝 → 물체중심 MAX 거리 (OpenArm 포팅 규약 .max())
        hand_points = torch.cat(
            [self.palm_center_pos.unsqueeze(1), self.fingertip_pos], dim=1
        )   # (N, 6, 3)
        hand_to_object_err = (
            hand_points - self.object_pos.unsqueeze(1)
        ).norm(dim=-1).max(dim=-1).values
        hand_to_object_reward = float(self.cfg.hand_to_object_weight) * torch.exp(
            -float(self.cfg.hand_to_object_sharpness) * hand_to_object_err
        )

        # ADR reward 스케줄 (DEXTRAH reward_weights): lift shaping 걷어내고 goal 정밀화
        _o2g_sharp = (
            self._adr("reward_weights", "object_to_goal_sharpness")
            if self.grasp_adr is not None else float(self.cfg.object_to_goal_sharpness)
        )
        _lift_w = (
            self._adr("reward_weights", "lift_weight")
            if self.grasp_adr is not None else float(self.cfg.lift_weight)
        )
        _curl_w = (
            self._adr("reward_weights", "finger_curl_reg")
            if self.grasp_adr is not None else float(self.cfg.finger_curl_reg_weight)
        )

        # 2) object_to_goal: 물체 → goal(고정 절대점) 거리
        object_to_goal_err = (self.object_pos - self.object_goal).norm(dim=-1)
        object_to_goal_reward = float(self.cfg.object_to_goal_weight) * torch.exp(
            -_o2g_sharp * object_to_goal_err
        )

        # 3) lift: goal 높이 절대거리 (ADR로 5→0 감쇠 — 후반은 object_to_goal 이 담당)
        object_vertical_err = (self.object_goal[:, 2] - self.object_pos[:, 2]).abs()
        lift_reward = _lift_w * torch.exp(
            -float(self.cfg.lift_sharpness) * object_vertical_err
        )

        # 4) finger_curl_reg (DEXTRAH: -0.01 → -0.005)
        finger_pos = self.robot.data.joint_pos[:, self.hand_dof_indices]
        finger_curl_dist = (
            finger_pos - self.hand_full_grip_pose.unsqueeze(0)
        ).norm(p=2, dim=-1)
        finger_curl_reg = _curl_w * finger_curl_dist ** 2

        # 5) palm orientation: 손바닥 법선(palm 로컬 +X, grasp_v1 규약)이 palm→물체
        #    방향과 정렬되도록. DEXTRAH 4항엔 손목 방향 제약이 없어 손바닥이 임의(천장)
        #    방향으로 수렴하던 것을 side-approach 자세로 유도.
        palm_quat = self.robot.data.body_quat_w[:, self.palm_body_index]   # (N,4) wxyz
        palm_x_local = torch.zeros_like(self.palm_center_pos)
        palm_x_local[:, 0] = 1.0
        palm_normal = quat_apply(palm_quat, palm_x_local)                  # (N,3) world
        palm_to_obj = self.object_pos - self.palm_center_pos
        palm_to_obj = palm_to_obj / (palm_to_obj.norm(dim=-1, keepdim=True) + 1e-8)
        palm_align = (palm_normal * palm_to_obj).sum(dim=-1)               # [-1, 1]
        palm_orient_reward = float(self.cfg.palm_orient_weight) * torch.exp(
            float(self.cfg.palm_orient_sharpness) * (palm_align - 1.0)
        )

        total = (
            hand_to_object_reward + object_to_goal_reward + finger_curl_reg
            + lift_reward + palm_orient_reward
        )

        # success: DEXTRAH in_success_region (goal 도달 = 최소 11cm 리프트 내포)
        self.in_success_region = object_to_goal_err < float(self.cfg.object_goal_tol)

        # contact 진단용 grip(tip|mid|distal 접촉 손가락 수)
        num_grip_fingers = (
            self.binary_contact_buf
            | self.middle_binary_contact_buf
            | self.distal_binary_contact_buf
        ).sum(dim=-1)

        _ep_success_rate = self._successful_episodes / max(self._total_episodes, 1)
        # ADR increment: DEXTRAH 원본 = in_success_region 순간 평균 > success_for_adr(0.4)
        if self.grasp_adr is not None:
            self.grasp_adr.maybe_increment(self.in_success_region.float().mean())

        # ── 로깅: DEXTRAH-g 정렬 ──────────────────────────────────────────
        # DEXTRAH 원본 extras: 4 reward 항 + in_success_region (동일 이름)
        self.extras["hand_to_object_reward"] = hand_to_object_reward.mean()
        self.extras["object_to_goal_reward"] = object_to_goal_reward.mean()
        self.extras["finger_curl_reg"] = finger_curl_reg.mean()
        self.extras["lift_reward"] = lift_reward.mean()
        self.extras["palm_orient_reward"] = palm_orient_reward.mean()
        self.extras["palm_align"] = palm_align.mean()
        self.extras["in_success_region"] = self.in_success_region.float().mean()
        # DEXTRAH 원본 extras: ADR 진행도
        if self.grasp_adr is not None:
            self.extras["num_adr_increases"] = torch.tensor(
                float(self.grasp_adr.increment_counter), device=self.device
            )
        # 해석 보조: 안착점(settle 스냅샷) 대비 실제 리프트 높이 → 렌더와 일치
        self.extras["object_height"] = (
            self.object_pos[:, 2] - self.object_init_pos[:, 2]
        ).mean()
        # contact: 감싸기 노드 그룹별 접촉 손가락 수 (0~5). palm 센서 없음 → 제외.
        self.extras["contact/tip"]    = self.num_contacts_buf.float().mean()
        self.extras["contact/middle"] = self.middle_binary_contact_buf.float().sum(dim=-1).mean()
        self.extras["contact/distal"] = self.distal_binary_contact_buf.float().sum(dim=-1).mean()
        self.extras["contact/grip"]   = num_grip_fingers.float().mean()
        # action policy(palm 6D + finger 5D raw) 로깅 유지
        for k, v in action_policy_scalars(
            action=self.actions, prev_action=self.prev_actions, palm_dims=6,
        ).items():
            self.extras[k] = v
        # episode_success_rate: 물체별 성공률 + 전체 (누적 집계는 _reset_idx 에서 갱신)
        for _i, _name in enumerate(self._object_names):
            _tot = self._obj_total_episodes[_i].item()
            if _tot > 0:
                self.extras[f"episode_success_rate/{_name}"] = torch.tensor(
                    self._obj_success_episodes[_i].item() / _tot, device=self.device
                )
        self.extras["episode_success_rate"] = torch.tensor(
            _ep_success_rate, device=self.device
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
        fallen = self.object_pos[:, 2] < self.cfg.obj_fallen_z

        # 로봇 발산 종료: fabric 폭주로 손이 도달불가 위치로 튕기면 컵-기준 종료가
        # 안 걸려 timeout까지 남던 문제. palm↔물체 거리가 workspace 초과 또는 NaN.
        palm_to_obj_dist = (self.palm_center_pos - self.object_pos).norm(dim=-1)
        robot_diverged = (
            (palm_to_obj_dist > self.cfg.robot_escape_dist)
            | torch.isnan(self.palm_center_pos).any(dim=-1)
            | torch.isnan(self.fabric_q).any(dim=-1)
        )

        # success bookkeeping: DEXTRAH in_success_region (goal 도달) + 유효 물체.
        valid_cup = ~(out_x | out_y | fallen)
        self.success_flag.copy_(self.in_success_region & valid_cup)
        self.transfer_entry_grasp_success_buf |= self.success_flag
        self.episode_success_buf |= self.success_flag

        if self.cfg.enable_warm_state_export:
            z_local = torch.zeros(self.num_envs, 3, device=self.device)
            z_local[:, 2] = 1.0
            cup_z_world = quat_apply(self.object_rot, z_local)
            self._maybe_export_warm_states(cup_z_world[:, 2])

        # DEXTRAH 정렬: 기울기(tipped) 종료 제거 — visdex 153종은 임의 안착 자세라
        # 컵 전용 upright 가정이 부당. 종료 = 물체 이탈/낙하 + timeout.
        terminated = out_x | out_y | fallen | robot_diverged
        truncated  = self.episode_length_buf >= self.max_episode_length - 1

        # warm export 진단: scripted phase 중 (tipped 제외) 조기 종료 추적
        if self.cfg.enable_warm_state_export:
            early_term = (out_x | out_y | fallen) & is_scripted_phase
            if early_term.any():
                self._warm_diag_terminated_early += int(early_term.sum().item())

        return terminated, truncated

    # ------------------------------------------------------------------
    # Warm-state export (grasp 성공 → 디스크 캐시 → pour warmstart)
    # ------------------------------------------------------------------
    def _maybe_export_warm_states(self, _cup_up_z: torch.Tensor) -> None:
        """right-grip lift-wait 상태를 디스크 캐시에 누적, 목표치 도달 시 1회 저장.

        Warm export waits for joint7-only lift-wait arm match plus retained grasp contact.
        """
        if getattr(self, "warm_export_done", False):
            return

        if not hasattr(self, "_warm_export_cache"):
            target_count = int(self.cfg.warm_state_target_count)
            self._warm_export_cache = GraspWarmStateCache(
                capacity=target_count,
                device=self.device,
                source_meta={
                    "object_spawn_z": self.cfg.object_spawn_z,
                    "lift_success_height": self.cfg.lift_success_height,
                    "object_spawn_x_center": self.cfg.object_spawn_x_center,
                    "object_spawn_y_center": self.cfg.object_spawn_y_center,
                    "object_spawn_xy_range": self.cfg.object_spawn_xy_range,
                    "export_mode": "right_grip_lift_wait_actual_grasp",
                    "cup_z_mode": "actual_lifted",
                    "lift_wait_joint7_only": 1.0,
                    "warm_min_contacts": self.cfg.warm_min_contacts,
                    "warm_contact_stable_steps": self.cfg.warm_contact_stable_steps,
                    "warm_lift_wait_arm_tol": self.cfg.warm_lift_wait_arm_tol,
                    "warm_lift_wait_hold_steps": self.cfg.warm_lift_wait_hold_steps,
                    "lift_wait_joint7_delta": self.cfg.lift_wait_joint7_delta,
                    "palm_min_x": float(self.palm_mins[0]),
                    "palm_min_y": float(self.palm_mins[1]),
                    "palm_min_z": float(self.palm_mins[2]),
                    "palm_max_x": float(self.palm_maxs[0]),
                    "palm_max_y": float(self.palm_maxs[1]),
                    "palm_max_z": float(self.palm_maxs[2]),
                },
            )
            self.warm_export_done = False
            self._warm_export_log_interval = max(1, (target_count + 19) // 20)
            self._warm_export_next_log_count = 1
            self._write_warm_export_progress(
                count=0,
                target=target_count,
                added=0,
                status="running",
            )

        warm_min_contacts = max(int(self.cfg.warm_min_contacts), MIN_CONTACTS_FOR_SUCCESS)
        warm_stable_steps = max(int(self.cfg.warm_contact_stable_steps), 1)
        grasped = self.num_contacts_buf >= warm_min_contacts
        self.warm_contact_stable_steps_buf = torch.where(
            grasped,
            self.warm_contact_stable_steps_buf + 1,
            torch.zeros_like(self.warm_contact_stable_steps_buf),
        )
        stable_grasp = self.warm_contact_stable_steps_buf >= warm_stable_steps
        actual_arm_pos_all = self.robot.data.joint_pos[:, self.arm_dof_indices]
        lift_wait_matched, self.lift_wait_match_hold_steps_buf = compute_arm_joint_match(
            actual_arm_pos_all,
            self.prelift_arm_pos_buf,
            tol=self.cfg.warm_lift_wait_arm_tol,
            previous_hold_steps=self.lift_wait_match_hold_steps_buf,
            required_hold_steps=self.cfg.warm_lift_wait_hold_steps,
        )
        warm_ok = (
            lift_wait_matched
            & self.transfer_entry_grasp_success_buf
            & stable_grasp
        )

        # ---- 진단 출력 (매 300 스텝) ----
        self._warm_diag_step += 1
        if self._warm_diag_step % 300 == 0:
            n = self.num_envs
            pct = lambda t: float(t.sum()) / n * 100.0
            print(
                f"[warm_diag step={self._warm_diag_step}] "
                "mode=right_grip_lift_wait "
                f"lift_wait_matched={pct(lift_wait_matched):.1f}% "
                f"entry_grasp={pct(self.transfer_entry_grasp_success_buf):.1f}% "
                f"grasped{warm_min_contacts}+={pct(grasped):.1f}% "
                f"stable{warm_stable_steps}={pct(stable_grasp):.1f}% "
                f"warm_ok={pct(warm_ok):.1f}% "
                f"contacts_mean={self.num_contacts_buf.float().mean():.2f} "
                f"early_term_in_lift_wait={self._warm_diag_terminated_early} "
                f"ep_len_mean={self.episode_length_buf.float().mean():.1f}",
                flush=True,
            )

        ids = warm_ok.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return

        joint_pos = self.robot.data.joint_pos[ids]
        arm_export = joint_pos[:, self.arm_dof_indices].clone()
        cup_pos_export = self.object_pos[ids]
        # arm/hand/cup/contact 모두 실제 sim 의 right-grip lift-wait 상태를 저장한다.
        palm_euler_export = self.palm_pose_targets[ids].clone()
        palm_euler_export[:, :3] = self.palm_center_pos[ids]
        demo_idx = self._env_assigned_demo_idx[ids]
        added = self._warm_export_cache.append(
            arm=arm_export,
            hand=joint_pos[:, self.hand_dof_indices],
            palm_euler=palm_euler_export,
            cup_pos_local=cup_pos_export,
            cup_quat_wxyz=self.object_rot[ids],
            num_contacts=self.num_contacts_buf[ids],
            per_finger_contact=self.binary_contact_buf[ids],
            stable_contact_steps=self.warm_contact_stable_steps_buf[ids],
            demo_file_idx=demo_idx,
        )
        if added <= 0:
            return

        count = len(self._warm_export_cache)
        target = int(self.cfg.warm_state_target_count)
        self._write_warm_export_progress(
            count=count,
            target=target,
            added=added,
            status="running",
        )
        if count >= self._warm_export_next_log_count or self._warm_export_cache.is_full:
            percent = min(100.0, 100.0 * count / max(1, target))
            print(
                f"[5g_grasp_right_v1] warm-state progress: "
                f"mode=right_grip_lift_wait {count}/{target} ({percent:.1f}%, +{added})",
                flush=True,
            )
            interval = self._warm_export_log_interval
            self._warm_export_next_log_count = min(
                target,
                ((count // interval) + 1) * interval,
            )

        if self._warm_export_cache.is_full:
            path = self.cfg.warm_state_export_path
            self._warm_export_cache.save_hdf5(path)
            self.warm_export_done = True
            self._write_warm_export_progress(
                count=len(self._warm_export_cache),
                target=target,
                added=0,
                status="complete",
            )
            print(
                f"[5g_grasp_right_v1] warm-state export complete: "
                f"mode=right_grip_lift_wait {len(self._warm_export_cache)} states → {path}",
                flush=True,
            )

    def _write_warm_export_progress(
        self,
        *,
        count: int,
        target: int,
        added: int,
        status: str,
    ) -> None:
        path = Path(str(self.cfg.warm_state_export_path) + ".progress.json")
        tmp = path.with_suffix(path.suffix + ".tmp")
        data = {
            "count": int(count),
            "target": int(target),
            "added": int(added),
            "status": status,
            "mode": "right_grip_lift_wait",
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, sort_keys=True)
                f.write("\n")
            tmp.replace(path)
        except OSError:
            pass

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
        # 물체별 성공 집계 (episode_success_rate/{object})
        _r_obj = self.object_idx[env_ids]
        _r_succ = self.episode_success_buf[env_ids].float()
        self._obj_total_episodes.index_add_(0, _r_obj, torch.ones_like(_r_succ))
        self._obj_success_episodes.index_add_(0, _r_obj, _r_succ)
        self.episode_success_buf[env_ids] = False

        # ---- 1. Reset source 선택 ----
        if self.demo_grasp_reset_bank is not None:
            # 순환 배정: env_id % num_demos → warmstart 와 demo 간 1:1 대응 보장
            env_ids_t = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)
            demo_indices = env_ids_t % self.demo_grasp_reset_bank.num_demos
            self._env_assigned_demo_idx[env_ids_t] = demo_indices
            start_arm = self.demo_grasp_reset_bank.start_arm_joint_pos[demo_indices]
            start_hand = self.demo_grasp_reset_bank.start_hand_joint_pos[demo_indices]
            start_palm_pose = self.demo_grasp_reset_bank.start_palm_pose_euler_zyx[demo_indices].clone()
            start_palm_pose[:, 2] = self.cfg.object_spawn_z + self.cfg.pregrasp_offset_z
            q_pregrasp = torch.cat([start_arm, start_hand], dim=1)
            approach_hand = start_hand

            noise_xy = torch.stack([
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_x,
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_y,
            ], dim=1)
            obj_pos_local = compute_demo_cup_spawn_local(
                start_palm_pose,
                self.pregrasp_offset[:2],
                self.cfg.object_spawn_z,
                noise_xy,
            )
            pregrasp_palm_pose = start_palm_pose
        else:
            q_pregrasp = self.robot_start_joint_pos[env_ids].clone()
            approach_hand = self.hand_open_pose.unsqueeze(0).expand(n, -1)

            # ---- 컵 spawn 위치: ADR 커리큘럼 반경 0→최대 (DEXTRAH object_spawn) ----
            _xy_range = (
                self._adr("object_spawn", "xy_range")
                if self.grasp_adr is not None else self.cfg.object_spawn_xy_range
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

            # ---- FABRICS pregrasp rollout/cache lookup ----
            noise = torch.stack([
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_x,
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_y,
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_z,
            ], dim=1)
            pregrasp_pos = obj_pos_local + self.pregrasp_offset.unsqueeze(0) + noise

            pregrasp_palm_pose = torch.zeros(n, 6, device=self.device)
            pregrasp_palm_pose[:, :3] = pregrasp_pos
            # RH56F1 side grasp: palm_sensor +z(법선)가 물체(-y 접근 → +y)를 향하도록
            # (ez,ey,ex)=(180,0,90). cache 빌드와 일치 (lstm_test1 검증 규약).
            pregrasp_palm_pose[:, 3] = math.radians(180.0)
            pregrasp_palm_pose[:, 4] = math.radians(0.0)
            pregrasp_palm_pose[:, 5] = math.radians(90.0)
            pregrasp_palm_pose = torch.max(
                torch.min(pregrasp_palm_pose, self.palm_maxs.unsqueeze(0)),
                self.palm_mins.unsqueeze(0),
            )

            if self.cfg.cache_pregrasp_reset:
                # cache lookup: spawn 위치 → 가장 가까운 grid point arm IK
                xi = ((obj_x - self._cache_xs[0]) / (self._cache_xs[1] - self._cache_xs[0])).round().long().clamp(0, self._cache_n - 1)
                yi = ((obj_y - self._cache_ys[0]) / (self._cache_ys[1] - self._cache_ys[0])).round().long().clamp(0, self._cache_n - 1)
                q_pregrasp[:, :NUM_ARM_DOF] = self._cache_q_arm[xi, yi]
            else:
                q_pregrasp = self._run_reset_fabric(env_ids, pregrasp_palm_pose, q_pregrasp)

            # hand는 APPROACH_POSE로 강제
            q_pregrasp[:, NUM_ARM_DOF:] = approach_hand

            # r_aj_7(손목, arm index 6)을 낮춰 palm을 물체 높이로 내림. fabric은 +y 수평
            # 유지 위해 r_aj_7을 높게 잡아 palm이 물체 rim에 뜨므로(probe 확정) bias로
            # 끌어내린다. bias 후 실제 palm(FK)로 anchor를 정합해 정책 시작 시 palm 튐 방지.
            if self.cfg.pregrasp_r_aj7_bias != 0.0:
                q_pregrasp[:, 6] = q_pregrasp[:, 6] - self.cfg.pregrasp_r_aj7_bias
                fq_fk = self.fabric.default_config.clone()
                fq_fk[env_ids, :NUM_ROBOT_DOF] = q_pregrasp
                pregrasp_palm_pose = self.fabric.get_palm_pose(fq_fk, "euler_zyx")[env_ids]

        # ---- 2. 로봇/Fabrics 상태 리셋 ----
        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_pos[:, self.actuated_dof_indices] = q_pregrasp
        full_pos[:, self.left_arm_dof_indices] = self.left_arm_zero_pos[0]
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        self.fabric_q[env_ids, :NUM_ROBOT_DOF] = q_pregrasp   # 우측 13D, 좌측[13:26] 중립 유지
        self.fabric_qd[env_ids].zero_()
        self.fabric_qdd[env_ids].zero_()
        self.object_init_pos[env_ids] = obj_pos_local
        # (goal 은 DEXTRAH식 고정 절대점 — reset 갱신 없음)
        self.in_success_region[env_ids] = False

        # ---- DEXTRAH 관측 노이즈 bias 재샘플 (per-episode, ADR 크기) ----
        self.robot_joint_pos_bias[env_ids] = (
            self._adr("robot_state_noise", "robot_joint_pos_bias")
            * 2.0 * (torch.rand(n, NUM_ARM_DOF + NUM_HAND_DOF, device=self.device) - 0.5)
        )
        self.robot_joint_vel_bias[env_ids] = (
            self._adr("robot_state_noise", "robot_joint_vel_bias")
            * 2.0 * (torch.rand(n, NUM_ARM_DOF + NUM_HAND_DOF, device=self.device) - 0.5)
        )
        self.object_pos_bias[env_ids] = (
            self._adr("object_state_noise", "object_pos_bias")
            * 2.0 * (torch.rand(n, 3, device=self.device) - 0.5)
        )
        self.object_rot_bias[env_ids] = (
            self._adr("object_state_noise", "object_rot_bias")
            * 2.0 * (torch.rand(n, 4, device=self.device) - 0.5)
        )

        # ---- 5. pregrasp / lift-wait-target 버퍼 저장 ----
        self.pregrasp_arm_pos_buf[env_ids] = q_pregrasp[:, :NUM_ARM_DOF]

        # palm_pose_targets를 pregrasp로 동기화 (첫 Fabrics 스텝 타겟 일관성)
        self.palm_pose_targets[env_ids] = pregrasp_palm_pose

        # delta action 기준점: action=0 → pregrasp 위치 유지
        self.pregrasp_palm_pose_buf[env_ids] = pregrasp_palm_pose

        # Fabrics cspace attractor(null-space)를 pregrasp arm pos로 설정
        # default_config가 ARM_START_POSE이면 null-space 항이 계속 팔을 당겨 초기 흔들림 발생
        # pregrasp arm pos로 설정 → 에피소드 시작 시 null-space 항 ≈ 0 → 안정
        self.fabric.default_config[env_ids, :NUM_ARM_DOF] = q_pregrasp[:, :NUM_ARM_DOF]

        prelift_arm = compute_joint7_lift_wait_target(
            q_pregrasp[:, :NUM_ARM_DOF],
            joint7_delta=getattr(self.cfg, "lift_wait_joint7_delta", 0.31),
            joint7_min=self.cfg.warm_j7_min,
            joint7_max=self.cfg.warm_j7_max,
        )
        self.prelift_arm_pos_buf[env_ids] = prelift_arm

        # ---- 7. 컵 spawn ----
        obj_pos_world = obj_pos_local + self.scene.env_origins[env_ids]
        # spawn 회전: ADR 0→1 (DEXTRAH randomize_rotation — x/y축 랜덤 회전 스케일)
        _rot_f = self._adr("object_spawn", "rotation")
        if _rot_f > 0.0:
            _r0 = (torch.rand(n, device=self.device) - 0.5) * 2.0 * math.pi * _rot_f
            _r1 = (torch.rand(n, device=self.device) - 0.5) * 2.0 * math.pi * _rot_f
            _half0, _half1 = _r0 * 0.5, _r1 * 0.5
            _qx = torch.stack([
                torch.cos(_half0), torch.sin(_half0),
                torch.zeros(n, device=self.device), torch.zeros(n, device=self.device),
            ], dim=1)   # x축 회전 (w,x,y,z)
            _qy = torch.stack([
                torch.cos(_half1), torch.zeros(n, device=self.device),
                torch.sin(_half1), torch.zeros(n, device=self.device),
            ], dim=1)   # y축 회전
            spawn_rot = quat_mul(_qx, _qy)
        else:
            spawn_rot = torch.zeros(n, 4, device=self.device)
            spawn_rot[:, 0] = 1.0
        zero_vel = torch.zeros(n, 6, device=self.device)
        cup_root_state = torch.cat([obj_pos_world, spawn_rot, zero_vel], dim=-1)
        self.cup.write_root_state_to_sim(cup_root_state, env_ids=env_ids)

        # ---- 8. 버퍼 리셋 ----
        self.hand_joint_targets[env_ids] = approach_hand
        self.contact_force_raw[env_ids].zero_()
        self.binary_contact_buf[env_ids] = False
        self.num_contacts_buf[env_ids]   = 0
        self.warm_contact_stable_steps_buf[env_ids] = 0
        self.lift_wait_match_hold_steps_buf[env_ids] = 0
        self.distal_contact_force_raw[env_ids].zero_()
        self.distal_binary_contact_buf[env_ids] = False
        self.middle_contact_force_raw[env_ids].zero_()
        self.middle_binary_contact_buf[env_ids] = False
        self.success_flag[env_ids] = False
        self.transfer_entry_grasp_success_buf[env_ids] = False
        self.finger_close_buf[env_ids] = 0.0

        # actions 리셋: delta action 방식 → action=0 = pregrasp 위치
        # (역스케일 불필요: scale(0, delta_mins, delta_maxs) = delta=0 → pregrasp 유지)
        self.actions[env_ids, :6] = 0.0
        self.actions[env_ids, 6:] = -1.0
        self.prev_actions[env_ids, :6] = 0.0
        self.prev_actions[env_ids, 6:] = -1.0

        # DEXTRAH 정렬: reset 직후 obs가 noisy 중간값을 요구 (첫 reset 포함)
        self._compute_intermediate_values()
