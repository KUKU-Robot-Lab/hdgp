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

"""환경 클래스: 5g_grasp_left_v1

right/grasp_v1(grasp_right_env)의 좌우 미러. v7: Fabrics 팔 학습 + per-finger lerp 5D
+ Contact sensor 없는 FK 기반 근접도 리워드

핵심 개선 (v1/v6 대비):
  - v1 문제: fabric_q/qd obs → sim2real 불가, palm_dist 기반 자동 닫힘 → 충돌 충격
  - v6 문제: 팔 고정 → cup 위치 오차 대응 불가, per-finger 5D 협응 학습 부족

Action (11D):
  [0:6]  6D palm pose → Fabrics IK → arm 7 DOF (학습, cup 위치 오차 대응)
  [6:11] 5D 손가락 lerp: -1 → HAND_APPROACH_POSE, +1 → HAND_GRASP_POSE.
         엄지(6)는 독립, 검지~소지(7:11)는 공통닫힘(couple_four_fingers)으로 묶여
         3지 국소최적 차단(grasp_v2 left 0.908 실증).

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
import torch.nn.functional as F

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
from isaaclab.utils.math import quat_apply, quat_conjugate

from openarm.common.grasp_logging import action_policy_scalars, joint_state_scalars
# ★grasp_v1 은 공유 core 를 쓰지 않는다(08.16 지시) — 감쌈 깊이·유지 페널티를
# 계속 손대야 하는데 공유 core 를 건드리면 v2/v7_2/v10_3/adapt 가 전부 영향받는다.
from .grasp_reward import compute_grasp_reward_terms
from openarm.common.grasp_v2_contract import (
    compute_action_delta_norm,
    compute_grasp_v2_stability,
)

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloLeftPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

from .grasp_left_env_cfg import GraspLeftEnvCfg
from .grasp_adr import GraspADR
from .grasp_left_constants import (
    NUM_ARM_DOF,
    NUM_HAND_DOF,
    NUM_FINGERTIPS,
    NUM_PALM_ACTION,
    NUM_FINGER_ACTION,
    NUM_FINGER_CHANNELS,
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
    JOINT_POS_ERR_MAX,
    MIN_CONTACTS_FOR_SUCCESS,
    PREGRASP_FABRICS_STEPS,
    ARM_START_POSE,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
)
from .grasp_left_preset import (
    RIGHT_ARM_REST_JOINT_POS,
    LEFT_ACTUATED_JOINT_NAMES,
    HAND_APPROACH_POSE,
    HAND_GRASP_POSE,
    HAND_FULL_GRIP_POSE,
    OBJECT_GOAL_POS,
)
from .finger_action_utils import compute_grasp_finger_targets, compute_lift_finger_targets
from .grasp_left_utils import (
    compute_joint7_lift_wait_target,
    compute_lift_readiness,
    scale,
    to_torch,
)
from .demo_grasp_reset import DemoGraspResetBank, compute_demo_cup_spawn_local
from .warm_state_cache import GraspWarmStateCache, compute_arm_joint_match


class GraspLeftEnv(DirectRLEnv):
    """OpenArm+Teosllo 왼손 파지 환경 v7.

    Action: 11D
      [0:6]  palm pose (x,y,z,ez,ey,ex), 정규화 [-1,1] → Fabrics IK
      [6:11] per-finger absolute synergy (thumb,index,middle,ring,pinky)
             grasp: APPROACH(-1) to GRASP(+1)
             lift:  GRASP(-1) to FULL_GRIP(+1)

    Episode:
      Grasp phase (step 0~479):  Fabrics arm + 정책 손가락
      Lift-wait phase (step 480~599): scripted joint7-only lift-wait + frozen hand
    """

    cfg: GraspLeftEnvCfg

    def __init__(self, cfg: GraspLeftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ----------------------------------------------------------------
        # DOF index 설정
        # ----------------------------------------------------------------
        self.actuated_dof_indices: list[int] = []
        for name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.robot.joint_names.index(name))

        self.fixed_arm_dof_indices: list[int] = []
        for name in cfg.fixed_arm_joint_names:
            if name in self.robot.joint_names:
                self.fixed_arm_dof_indices.append(self.robot.joint_names.index(name))

        self.arm_dof_indices  = self.actuated_dof_indices[:NUM_ARM_DOF]    # list[int]
        self.hand_dof_indices = self.actuated_dof_indices[NUM_ARM_DOF:]    # list[int]

        # body indices (robot.data.body_pos_w 참조용). 통일 네이밍: l_hl_<finger>_*
        _FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
        _tip_names = [f"l_hl_{fn}_tip" for fn in _FINGERS]
        self.fingertip_body_indices: list[int] = [
            self.robot.data.body_names.index(name) for name in _tip_names
        ]
        _palm_name = "l_hl_palm"
        self.palm_body_index: int = (
            self.robot.data.body_names.index(_palm_name)
            if _palm_name in self.robot.data.body_names
            else -1
        )
        # distal phalanx body indices (l_hl_<finger>_4) — R2 reward용
        _distal4_names = [f"l_hl_{fn}_4" for fn in _FINGERS]
        self.distal4_body_indices: list[int] = [
            self.robot.data.body_names.index(name)
            for name in _distal4_names
            if name in self.robot.data.body_names
        ]

        # ----------------------------------------------------------------
        # Palm pose 절대 workspace (안전 한계 클램프용) — 회전은 P-frame euler_zyx
        # 직접 전달(right 미러, 중심 [-90,0,-90]deg). fabric 에 그대로 넘긴다.
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
        hand_start = to_torch(HAND_APPROACH_POSE, device=self.device)  # (20,)
        robot_start = torch.cat([arm_start, hand_start], dim=0)         # (27,)
        self.robot_start_joint_pos = (
            robot_start.unsqueeze(0).repeat(self.num_envs, 1).contiguous()
        )

        # ----------------------------------------------------------------
        # 오른팔 고정 자세
        # ----------------------------------------------------------------
        fixed_vals = [
            RIGHT_ARM_REST_JOINT_POS.get(self.robot.joint_names[idx], 0.0)
            for idx in self.fixed_arm_dof_indices
        ]
        self.fixed_arm_zero_pos = (
            to_torch(fixed_vals, device=self.device)
            .unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.fixed_arm_zero_vel = torch.zeros(
            self.num_envs, len(self.fixed_arm_dof_indices), device=self.device
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
        # 다물체(MultiAsset 8종) 배정: env_id % N 결정론적 (rh56f1 grasp_v1 이식).
        # ----------------------------------------------------------------
        self._object_names = list(self.cfg.active_object_names)
        self.object_idx = (
            torch.arange(self.num_envs, device=self.device) % len(self._object_names)
        )
        # DEXTRAH/rh56f1 식 onehot 조건화: reset 불변(항상 env_id%N 고정 배정이므로).
        self.multi_object_idx_onehot = F.one_hot(
            self.object_idx, num_classes=len(self._object_names)
        ).float()   # (num_envs, N_obj)

        # per-object bbox → 물리 안착 텐서 (design §per-object 처리, reward 아님).
        # object_spawn_z/cup_radius_approx/cup_grasp_z_offset(cfg, cup_big 기준 스칼라)를
        # object_bbox.json 반높이·반경 비율로 물체별 텐서화한다. 누락 시 즉시 실패(fail loud).
        (
            self.object_spawn_z_buf,
            self.cup_radius_approx_buf,
            self.cup_grasp_z_offset_buf,
        ) = self._load_object_physical_tensors()

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
        self.pregrasp_arm_pos_buf      = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        self.prelift_arm_pos_buf       = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        self.lift_arm_start_buf        = torch.zeros(self.num_envs, NUM_ARM_DOF, device=self.device)
        self.is_lift_phase             = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 접촉 latch 흐름: 잡으면 바로 리프트 (step-480 scripted 대체)
        self.lift_ready_latched_buf    = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.grasp_ready_hold_buf      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.lift_start_step_buf       = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ----------------------------------------------------------------
        # 물체 외란 wrench 버퍼 (08.16 right 이식, cfg 주석 참조). _cup_mass는 mass DR
        # (reset event) 반영 위해 _reset_idx에서 배치 갱신(F=m·a의 m을 현재 실효질량으로).
        # ----------------------------------------------------------------
        self.object_applied_force  = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.object_applied_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # ----------------------------------------------------------------
        # 감쌈 깊이 버퍼 (08.16). wrap_frac = per-finger (middle AND distal) 비율,
        # wrap_at_latch = 래치 순간 스냅샷(유지 페널티 기준선). _pre_physics_step 서 갱신.
        # ----------------------------------------------------------------
        self.wrap_frac_buf     = torch.zeros(self.num_envs, device=self.device)
        self.wrap_at_latch_buf = torch.zeros(self.num_envs, device=self.device)
        self._cup_mass = self.cup.root_physx_view.get_masses().to(self.device).view(-1)

        # ----------------------------------------------------------------
        # Hand joint targets (per-finger lerp 결과)
        # ----------------------------------------------------------------
        self.hand_joint_targets = torch.zeros(self.num_envs, NUM_HAND_DOF, device=self.device)
        # 관절별 접촉-게이트 적응 폐쇄: 관절당 폐쇄 진행도 [0,1] (N,20)
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
        self.reward_contact_hold_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._prev_reward_contacts_buf = torch.zeros(self.num_envs, device=self.device)

        # ----------------------------------------------------------------
        # 기타 버퍼
        # ----------------------------------------------------------------
        self._approach_dir_buf = torch.zeros(self.num_envs, 3, device=self.device)
        self.success_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cup_tipping_cos = math.cos(math.radians(cfg.cup_tipping_max_deg))
        # episode-level 성공 추적 (per-step average 허수 문제 해결)
        self.episode_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.transfer_entry_grasp_success_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._total_episodes: int = 0
        self._successful_episodes: int = 0
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
                # 08.16 물리 DR ADR(grasp_v2 이식): increment 시 질량·마찰 EventTerm
                # 범위를 중립→terminal 로 확장.
                event_manager=getattr(self, "event_manager", None),
                physics_cfg=getattr(cfg, "adr_physics_cfg", None),
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
        self.table = RigidObject(self.cfg.table_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table

        # Actor: fingertip 개별 ContactSensor (Cup-only, real FT sensor 대응).
        # 2026-07-26 MultiAsset 전환: 물체마다 rigid body prim 이 다르다(baseLink 중첩
        # vs Xform 루트 자체) — 두 패턴을 모두 filter 에 걸고 _update_contact_forces 에서
        # filter 축을 합산한다(cfg.object_contact_filter, GPU 검증 필요).
        _OBJECT_FILTER = list(self.cfg.object_contact_filter)
        self._tip_sensors: list[ContactSensor] = []
        for link_name in self.cfg.left_tip_contact_links:
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link_name}",
                filter_prim_paths_expr=_OBJECT_FILTER,
                history_length=1,
                track_air_time=False,
            ))
            self._tip_sensors.append(sensor)
            self.scene.sensors[f"tip_sensor_{link_name}"] = sensor

        # distal/middle 도 tip 처럼 손가락별 개별 Cup-only 센서.
        # 다중 body 단일 센서의 force_matrix_w 는 채워지지 않아(0 반환) Cup 필터가 무력화되므로,
        # l_hl_<finger>_4 / _3 를 개별 ContactSensor 로 만들어 force_matrix_w[:,0,:,:] 로 읽는다.
        _SENSOR_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
        self._distal_sensors: list[ContactSensor] = []
        for i, fn in enumerate(_SENSOR_FINGERS):
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/l_hl_{fn}_4",
                filter_prim_paths_expr=_OBJECT_FILTER,
                history_length=1,
                track_air_time=False,
            ))
            self._distal_sensors.append(sensor)
            self.scene.sensors[f"distal_sensor_{i + 1}"] = sensor

        self._middle_sensors: list[ContactSensor] = []
        for i, fn in enumerate(_SENSOR_FINGERS):
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/l_hl_{fn}_3",
                filter_prim_paths_expr=_OBJECT_FILTER,
                history_length=1,
                track_air_time=False,
            ))
            self._middle_sensors.append(sensor)
            self.scene.sensors[f"middle_sensor_{i + 1}"] = sensor

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # ★다물체 스폰 순서(rh56f1 grasp_v1 07.10 버그 수정 이식): clone → cup(MultiAsset) 생성.
        # RigidObject(cup_cfg)를 clone 이전에 만들면 env_0만 존재하는 시점에 MultiAssetSpawner가
        # 물체[0] 하나만 spawn하고 clone이 그걸 전 env에 복제(전 env 동일 물체 버그)한다.
        # clone을 먼저 해야 spawn 시점에 env prim이 전부 존재해 env_i % N 결정적 배정이 된다.
        self.scene.clone_environments(copy_from_source=True)
        self.cup = RigidObject(self.cfg.cup_cfg)
        self.scene.rigid_objects["cup"] = self.cup

    # ------------------------------------------------------------------
    # per-object 물리 안착 텐서 (design §per-object 처리, reward 아님)
    # ------------------------------------------------------------------
    def _load_object_physical_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """object_bbox.json 기반 물체별 (spawn_z, radius_approx, grasp_z_offset).

        cfg.object_spawn_z/cup_radius_approx/cup_grasp_z_offset 은 원래 단일 cup_big_sdf
        전용 스칼라였다. "cup_big_s100"(cup_big 원본과 동일 bbox)을 캘리브레이션 기준으로
        삼아 테이블 표면 z 와 offset 비율을 역산하고, 물체별 반높이/반경으로 텐서화한다.
        누락 물체는 즉시 실패(fallback 없음 — rh56f1 grasp_v1 _load_object_clearances 동일 원칙).

        assumption(육안/GPU 검증 필요): object_spawn_z=0.297 이 cup_big(반높이 0.0888) 기준
        테이블 안착 높이로 캘리브레이션됐다고 가정 — 다른 형상(shaker_body 등)에서도 바닥
        clearance 가 동일하다고 가정한다(design §검증 3. play 육안 확인 항목).
        """
        path = Path(self.cfg.object_bbox_path)
        if not path.is_file():
            raise FileNotFoundError(f"물체 bbox 파일 없음: {path}")
        table = json.loads(path.read_text(encoding="utf-8"))

        missing = [n for n in self._object_names if n not in table]
        if missing:
            raise KeyError(f"bbox 누락 물체 {len(missing)}종: {missing} — object_bbox.json 등록 필요")
        _REF = "cup_big_s100"
        if _REF not in table:
            raise KeyError(f"캘리브레이션 기준 물체 '{_REF}' bbox 누락")

        half_extents = to_torch(
            [table[n] for n in self._object_names], device=self.device
        )   # (N_obj, 3)
        ref_half_z = float(table[_REF][2])
        table_surface_z = float(self.cfg.object_spawn_z) - ref_half_z
        z_offset_ratio = float(self.cfg.cup_grasp_z_offset) / ref_half_z

        spawn_z_per_obj  = table_surface_z + half_extents[:, 2]
        radius_per_obj   = 0.5 * (half_extents[:, 0] + half_extents[:, 1])
        z_offset_per_obj = z_offset_ratio * half_extents[:, 2]

        return (
            spawn_z_per_obj[self.object_idx],
            radius_per_obj[self.object_idx],
            z_offset_per_obj[self.object_idx],
        )

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
            # right world는 왼팔 영역(y>0)에 반발체를 두므로 left에서 그대로 쓰면 왼손이
            # 자기 workspace에서 밀려난다(left/grasp_v2 lstm_test2 in_success 0 근본원인
            # 이식). left 미러 world 사용.
            world_filename="open_tesollo_left_boxes_no_table",
        )
        self.object_ids, self.object_indicator = self.world_model.get_object_ids()

        self.timestep = self.cfg.fabrics_dt

        # Main fabric (arm 제어용, graph_capturable=False)
        self.open_tesollo_fabric = OpenArmTeoslloLeftPoseFabric(
            self.num_envs, self.device, self.timestep,
            graph_capturable=False,
            use_hand_fabric=False,
            # ★08.17 DG-5FS 전용 fabric URDF(P0b, FK 오차 0.000mm 검증). 기존 URDF 는 불변.
            robot_dir_name="openarm_tesollo_bi_s_left",
            robot_name="openarm_tesollo_bi_s_left",
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
        self._reset_fabric = OpenArmTeoslloLeftPoseFabric(
            self._reset_chunk, self.device, self.timestep,
            graph_capturable=False,
            use_hand_fabric=False,
            # ★08.17 DG-5FS 전용 fabric URDF(P0b, FK 오차 0.000mm 검증). 기존 URDF 는 불변.
            robot_dir_name="openarm_tesollo_bi_s_left",
            robot_name="openarm_tesollo_bi_s_left",
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
            world_filename="open_tesollo_left_boxes_no_table",
        )
        self._reset_obj_ids, self._reset_obj_indicator = self._reset_world.get_object_ids()



        # Pregrasp IK 캐시 사전 계산 (spawn grid 전체)
        if self.cfg.cache_pregrasp_reset and self.demo_grasp_reset_bank is None:
            self._build_pregrasp_cache()

    # ------------------------------------------------------------------
    # Pregrasp grid 캐시 빌드 (startup 1회)
    # ------------------------------------------------------------------
    def _build_pregrasp_cache(self) -> None:
        """spawn 위치 17×17 grid에 대해 Fabrics IK를 startup에서 일괄 계산.

        reset 시 nearest-neighbor lookup → Fabrics rollout 생략 → 대폭 속도 향상.
        1cm 간격 grid이므로 실제 spawn 위치와 최대 ~0.7cm 오차 → Fabrics가 첫 몇 스텝에서 보정.

        2026-07-26: grid 반경을 cfg.pregrasp_cache_xy_range(고정 ±8cm)로 확대 —
        ADR(spawn xy_range 0.02→0.08)가 최종적으로 요구하는 최대범위를 항상 커버해야
        ADR 진행에 따라 lookup이 grid 밖으로 벗어나지 않는다(object_spawn_xy_range는
        ADR 미사용 시 fallback일 뿐이라 캐시 크기 기준으로 쓰지 않는다).
        """
        _cache_range = float(self.cfg.pregrasp_cache_xy_range)
        _N = int(round(2 * _cache_range / 0.01)) + 1  # 1cm 간격 → ±8cm=17
        xs = torch.linspace(
            self.cfg.object_spawn_x_center - _cache_range,
            self.cfg.object_spawn_x_center + _cache_range,
            _N, device=self.device,
        )
        ys = torch.linspace(
            self.cfg.object_spawn_y_center - _cache_range,
            self.cfg.object_spawn_y_center + _cache_range,
            _N, device=self.device,
        )
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        flat_x, flat_y = gx.flatten(), gy.flatten()
        M = flat_x.shape[0]  # _N*_N (17×17=289)

        palm = torch.zeros(M, 6, device=self.device)
        palm[:, 0] = flat_x + self.cfg.pregrasp_offset_x
        palm[:, 1] = flat_y + self.cfg.pregrasp_offset_y
        palm[:, 2] = self.cfg.object_spawn_z + self.cfg.pregrasp_offset_z
        palm[:, 3] = math.radians(-90.0)
        palm[:, 4] = math.radians(0.0)
        palm[:, 5] = math.radians(-90.0)
        palm = torch.max(
            torch.min(palm, self.palm_maxs.unsqueeze(0)),
            self.palm_mins.unsqueeze(0),
        )

        q_init = self.robot_start_joint_pos[0].unsqueeze(0).expand(M, -1).contiguous()
        dummy  = torch.arange(M, device=self.device)
        q_out  = self._run_reset_fabric(dummy, palm, q_init.clone())

        # (_N, _N, 7): arm joints only
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
        # 2026-07-26 MultiAsset: 8종 전부 "Cup/baseLink" 단일 filter(shaker 도 fix_shaker_asset
        # 으로 visdex 표준 이식). force_matrix_w (N, sensor_body=1, filter=1, 3) →
        # [:,0,:,:].sum(dim=1) → (N, 3). filter 축 합산은 단일 filter 에도 안전.
        # Actor: fingertip 개별 센서 (Cup-only)
        tip_xyz = torch.stack([
            s.data.force_matrix_w[:, 0, :, :].sum(dim=1) for s in self._tip_sensors
        ], dim=1)   # (N, 5, 3)
        tip_norms = tip_xyz.norm(dim=-1)   # (N, 5)

        self.contact_force_xyz_raw.copy_(tip_xyz)
        self.contact_force_raw.copy_(tip_norms)
        self.binary_contact_buf.copy_(tip_norms > CONTACT_FORCE_THRESHOLD)
        self.num_contacts_buf.copy_(self.binary_contact_buf.sum(dim=-1).long())

        # Critic: distal (Cup-only, 손가락별 개별 센서 force_matrix_w[:, 0, :, :].sum(dim=1))
        per_distal = torch.stack([
            s.data.force_matrix_w[:, 0, :, :].sum(dim=1) for s in self._distal_sensors
        ], dim=1).norm(dim=-1)   # (N, 5)
        self.distal_contact_force_raw.copy_(per_distal)
        self.distal_binary_contact_buf.copy_(per_distal > CONTACT_FORCE_THRESHOLD)

        # Critic: middle (Cup-only, 손가락별 개별 센서 force_matrix_w[:, 0, :, :].sum(dim=1))
        per_middle = torch.stack([
            s.data.force_matrix_w[:, 0, :, :].sum(dim=1) for s in self._middle_sensors
        ], dim=1).norm(dim=-1)   # (N, 5)
        self.middle_contact_force_raw.copy_(per_middle)
        self.middle_binary_contact_buf.copy_(per_middle > CONTACT_FORCE_THRESHOLD)

    # ------------------------------------------------------------------
    # 파지력 확보: 물체 외란 wrench (08.16 right 이식, DEXTRAH apply_object_wrench)
    # ------------------------------------------------------------------
    def _apply_object_wrench(self) -> None:
        # 게이트: palm이 물체 반경 내면 인가(DEXTRAH 원본) — 접근 후 파지·운반 전 구간에서
        # robust hold 단련. object_pos/palm_center_pos는 직전 스텝 값(둘 다 env-local).
        apply = (
            (self.palm_center_pos - self.object_pos).norm(dim=-1)
            <= float(self.cfg.wrench_hand_dist_threshold)
        ).view(-1, 1, 1)
        # trigger_every step 마다 새 랜덤 wrench (그 사이 유지)
        new_trig = (
            (self.episode_length_buf % int(self.cfg.wrench_trigger_every)) == 0
        ).view(-1, 1, 1)
        max_accel = (
            self.grasp_adr.get_param("object_wrench", "max_linear_accel")
            if self.grasp_adr is not None else float(self.cfg.wrench_max_accel)
        )
        accel = max_accel * torch.rand(self.num_envs, 1, 1, device=self.device)
        fmag = accel * self._cup_mass.view(-1, 1, 1)                       # F = m·a
        tmag = fmag * float(self.cfg.wrench_torsional_radius)              # τ = m·a·r

        def _rand_dir() -> torch.Tensor:
            d = torch.randn(self.num_envs, 1, 3, device=self.device)
            return d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        f = fmag * _rand_dir()
        t = tmag * _rand_dir()
        # 회전 외란(Exp4-A, cfg 주석): lift latch 이후 env에 수평 랜덤축 torque 추가 —
        # pour deep-tilt 회전 하중 재현. latch 판정은 샘플 시점 값(트리거 주기마다 재평가).
        if bool(getattr(self.cfg, "hold_rotation_perturb_enable", False)):
            rot_max = (
                self.grasp_adr.get_param("hold_rotation", "max_accel")
                if self.grasp_adr is not None
                else float(self.cfg.hold_rotation_perturb_max_accel)
            )
            rot_accel = rot_max * torch.rand(self.num_envs, 1, 1, device=self.device)
            rot_tmag = (
                rot_accel * self._cup_mass.view(-1, 1, 1)
                * float(self.cfg.wrench_torsional_radius)
            )
            _ang = torch.rand(self.num_envs, 1, 1, device=self.device) * (2.0 * math.pi)
            rot_axis = torch.cat(
                [torch.cos(_ang), torch.sin(_ang), torch.zeros_like(_ang)], dim=-1
            )  # 수평축(z=0) — 틸트 방향 토크
            rot_t = rot_tmag * rot_axis
            t = t + torch.where(
                self.lift_ready_latched_buf.view(-1, 1, 1), rot_t, torch.zeros_like(rot_t)
            )
        self.object_applied_force = torch.where(new_trig, f, self.object_applied_force)
        self.object_applied_torque = torch.where(new_trig, t, self.object_applied_torque)
        # 게이트 밖(palm 멀리) env 는 wrench 0
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

        palm_action = actions[:, :NUM_PALM_ACTION]                  # (N, 6) ∈ [-1, 1]
        # ★08.16 PIP/DIP 분리: (N,15) → (N,5,3) [손가락, 채널]. 채널 0=_1 외전 / 1=_2 MCP /
        #   2=_3·_4 PIP·DIP 공통. 이제 정책이 **관절 사이의 비율**을 정할 수 있다.
        finger_action = actions[
            :, NUM_PALM_ACTION:NUM_PALM_ACTION + NUM_FINGER_ACTION
        ].view(self.num_envs, NUM_FINGERTIPS, NUM_FINGER_CHANNELS)

        # ---- couple_four_fingers (left-only, 08.02): 3지 국소최적 원천 차단 ----
        # 검지~소지(1:5)를 공통 신호(평균)로 묶어 "특정 손가락만 안 닫힘" action 자체를 표현
        # 불가하게 한다. 엄지(0)는 opposition 회전 위해 독립. 접촉 시 개별 동결(g3/g4)은 그대로라
        # 각 손가락이 닿는 지점서 멈춰 최종 조합은 물체가 결정(형상 적응 유지). grasp_v2 검증법.
        # ★분리 후에도 **채널별로** 평균낸다 — 4지가 같은 자세를 공유하되, 그 자세의
        #   외전/MCP/PIP 비율은 정책이 자유롭게 정한다(3지 방지와 형상 자유도가 양립).
        if bool(getattr(self.cfg, "couple_four_fingers", False)):
            _thumb_a = finger_action[:, 0:1, :]                       # (N,1,3)
            _common4 = finger_action[:, 1:5, :].mean(dim=1, keepdim=True)
            finger_action = torch.cat([_thumb_a, _common4.expand(-1, 4, -1)], dim=1)

        # ---- Phase 판정: 접촉 latch (감싸 잡으면 리프트, step-480 scripted 대체) ----
        # lift 진입 게이트: 손가락별 아무 마디(tip|mid|distal)든 닿은 손가락 수(grip)로 판정.
        # 사용자 설계 의도 = "손가락이 어느 위치든 닿았다면 그대로 lift 진행" + 제어(g3)가 손끝
        # (distal)까지 감아 자연스럽게 인벨롭 유도 → 다양한 크기 컵을 robust 파지.
        # 부실 파지 방지 = ①hold_steps(연속 접촉 유지) ②success의 lifted+stable+tilt 이중 게이트
        #   (과거 any 완화 붕괴 lifted 0.72→0.002는 유지조건이 약했던 것 — 여기선 hold+success로 견고 파지만 성공).
        num_grip_fingers = (
            self.binary_contact_buf
            | self.middle_binary_contact_buf
            | self.distal_binary_contact_buf
        ).sum(dim=-1)
        prev_latched = self.lift_ready_latched_buf.clone()
        self.grasp_ready_hold_buf, _ready_now, lift_latched = compute_lift_readiness(
            num_contacts=num_grip_fingers,
            is_grasp_phase=~self.lift_ready_latched_buf,
            previous_hold_count=self.grasp_ready_hold_buf,
            previous_latched=self.lift_ready_latched_buf,
            min_contacts=int(self.cfg.lift_start_min_grip_fingers),
            hold_steps=int(self.cfg.grasp_ready_hold_steps),
        )
        self.lift_ready_latched_buf.copy_(lift_latched)
        is_lift = self.lift_ready_latched_buf
        self.is_lift_phase.copy_(is_lift)

        # ---- Lift 진입(래치 전환) 시 arm joint pos + 진입 step 캡처 ----
        just_entering_lift = self.lift_ready_latched_buf & (~prev_latched)
        self.lift_start_step_buf = torch.where(
            just_entering_lift,
            self.episode_length_buf,
            self.lift_start_step_buf,
        )

        # ---- 감쌈 깊이(per-finger mid AND distal)와 래치 시점 스냅샷 ----
        # wrap_frac 은 "같은 손가락이 중간마디와 원위마디 둘 다 닿았나" = 진짜 감쌈.
        # envelope_frac(0.5*(mid+dist) 평균)은 서로 다른 손가락이어도 값이 올라 느슨하다.
        # wrap_at_latch 는 유지 페널티의 기준선 — 래치 순간의 깊이를 붙들어 두고,
        # 그보다 얕아진 만큼만 처벌한다(절대 깊이 처벌은 리프트 보상을 억제해 REVISE됨).
        self.wrap_frac_buf.copy_(
            (self.middle_binary_contact_buf & self.distal_binary_contact_buf)
            .float().mean(dim=-1)
        )
        self.wrap_at_latch_buf = torch.where(
            just_entering_lift, self.wrap_frac_buf, self.wrap_at_latch_buf
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
        # Target = actual grasp arm pose with only joint7 moved into lift-wait.
        actual_prelift = compute_joint7_lift_wait_target(
            actual_arm_pos,
            joint7_delta=getattr(self.cfg, "lift_wait_joint7_delta", 0.31),
            joint7_min=self.cfg.warm_j7_min,
            joint7_max=self.cfg.warm_j7_max,
        )
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
        palm_mins = torch.minimum(self.palm_mins.unsqueeze(0), self.pregrasp_palm_pose_buf)
        palm_maxs = torch.maximum(self.palm_maxs.unsqueeze(0), self.pregrasp_palm_pose_buf)
        palm_pose = torch.max(torch.min(palm_pose, palm_maxs), palm_mins)
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

        # ---- 관절별 접촉-게이트 적응 폐쇄 (underactuated wrap) ----
        # 각 curl 관절을 독립 진행도로 폐쇄(APPROACH→FULL_GRIP)하되 자기 마디 센서로 동결:
        #   _1 외전 / _2 MCP: 무게이트(full close, 근위 마디를 컵에 밀착)
        #   _3 PIP: 중간마디(middle) 접촉 시 동결 / _4 DIP: distal|tip 접촉 시 동결
        # → distal→proximal 순차 동결로 컵 형상에 손가락이 드리워짐(envelope).
        # 15D action = 손가락×채널 **절대 폐쇄도**[0,1]. 관절 순서 finger-major [_1,_2,_3,_4]×5.
        cmd_ch = 0.5 * (finger_action.clamp(-1.0, 1.0) + 1.0)       # (N,5,3) ∈ [0,1]
        tip_c  = self.binary_contact_buf.float()                    # (N,5) 끝
        dist_c = self.distal_binary_contact_buf.float()             # (N,5) distal(l_hl_X_4)
        mid_c  = self.middle_binary_contact_buf.float()             # (N,5) middle(l_hl_X_3)
        # 관절별 동결 게이트 (local 0=_1, 1=_2, 2=_3 PIP, 3=_4 DIP)
        g1 = torch.zeros_like(tip_c)                                # _1: 무게이트
        g2 = torch.zeros_like(tip_c)                                # _2 MCP: 무게이트(full close)
        # _3 PIP·_4 DIP 둘 다 distal|tip 접촉 시 동결. (구 g3=mid_c는 중간마디 접촉 순간
        # PIP를 멈춰 손가락이 middle로 컵에 얹힌 채 정지 → 손끝(distal)까지 감아 조이지 못함
        # = 인벨롭 미완성·distal 저조·wrap 미달. 이제 손끝이 닿을 때까지 계속 감아 진짜로 감싼다.)
        g3 = (dist_c + tip_c).clamp(max=1.0)                        # _3 PIP: distal|tip 접촉 시 동결
        g4 = (dist_c + tip_c).clamp(max=1.0)                        # _4 DIP: distal|tip 접촉 시 동결
        gate20 = torch.stack([g1, g2, g3, g4], dim=2).reshape(self.num_envs, -1)  # (N,20)
        # ★08.16 래치 후 재조임 권한(retighten_after_latch).
        # 파지력은 힘 명령이 아니라 `stiffness × (target − actual)` 오버슈트가 전부인데,
        # 동결이 **첫 접촉(0.1N)** 에서 걸리므로 오버슈트가 거의 0 인 채 고정된다 —
        # 즉 "조인 것"이 아니라 "닿은 데서 멈춘 것"이고, 외란이 와도 더 조일 수단이 없다.
        # 래치 후에는 동결을 풀어 정책이 finger_close_buf 를 1.0 쪽으로 더 밀 수 있게 한다.
        # 되풀기(음의 advance)는 넣지 않는다 — 래치 회피 gradient 를 여는 과거 실패 패턴.
        # 래치 전 동결은 그대로 유지(접근→형상적응 감쌈이 이 구조로 만들어짐).
        if bool(getattr(self.cfg, "retighten_after_latch", False)):
            gate20 = gate20 * (~self.lift_ready_latched_buf).float().unsqueeze(1)
        # 채널 → 20관절 전개. [_1, _2, _3, _4] 순서에 [ch0, ch1, ch2, ch2] 를 대응.
        cmd20 = torch.stack(
            [cmd_ch[:, :, 0], cmd_ch[:, :, 1], cmd_ch[:, :, 2], cmd_ch[:, :, 2]], dim=2
        ).reshape(self.num_envs, -1)                                # (N,20)
        # ★08.16 래칫 제거 — 명령을 "속도"에서 "절대 폐쇄도 목표"로 바꾼다.
        # 구: advance = speed × cmd20 ≥ 0 → 단조 증가만 가능. cmd 는 [0,1] 이라 탐색 노이즈
        #   평균(cmd≈0.5)만으로도 스텝당 +0.0125 씩 쌓여 80스텝이면 완전 폐쇄에 도달하고
        #   되돌릴 수 없었다(실증: close_frac_max 가 첫 구간부터 1.0). 즉 정책은 "얼마나
        #   닫을지"를 표현할 수 없었고, **채널을 분리해도 전부 1.0 으로 포화해 비율이 안 생긴다.**
        #   → PIP/DIP 분리가 의미를 가지려면 이 수정이 필수다(둘은 한 묶음).
        # 신: 목표를 향해 finger_close_speed 를 **변화율 상한**으로 삼아 이동. 감소 가능.
        # 동결은 그대로 유지 — 접촉한 관절은 그 자리에 멈춰 컵 형상에 드리워진다(감쌈 생성
        #   메커니즘이자 다형상 적응의 근거). 여기를 건드리면 3지 국소최적으로 회귀한다.
        _rate = float(self.cfg.finger_close_speed)
        delta = (cmd20 - self.finger_close_buf).clamp(-_rate, _rate)
        advance = delta * (1.0 - gate20)
        self.finger_close_buf = (self.finger_close_buf + advance).clamp(0.0, 1.0)  # (N,20)
        hand_target = torch.lerp(
            self.hand_open_pose.unsqueeze(0).expand(self.num_envs, -1),
            self.hand_full_grip_pose.unsqueeze(0).expand(self.num_envs, -1),
            self.finger_close_buf,                                  # (N,20) 관절별 진행도
        )

        hand_target = hand_target.clamp(
            self.hand_joint_lower_limits.unsqueeze(0),
            self.hand_joint_upper_limits.unsqueeze(0),
        )
        self.hand_joint_targets.copy_(hand_target)

        # fabric_q hand 부분 동기화 (FK 계산에 활용)
        self.fabric_q[:, NUM_ARM_DOF:] = hand_target
        self.fabric_qd[:, NUM_ARM_DOF:].zero_()

        # ---- Lift-wait phase: Fabrics arm 상태 동결 ----
        # scripted arm 제어 중 Fabrics integrator 발산 방지
        freeze_mask = self.is_lift_phase
        # util: .any() 동기화 제거 — 마스크 대입은 빈 마스크도 안전(빈 연산)
        self.fabric_q[freeze_mask, :NUM_ARM_DOF] = (
            self.robot.data.joint_pos[freeze_mask][:, self.arm_dof_indices]
        )
        self.fabric_qd[freeze_mask, :NUM_ARM_DOF].zero_()
        self.fabric_qdd[freeze_mask, :NUM_ARM_DOF].zero_()

    def _apply_action(self) -> None:
        is_lift       = self.is_lift_phase        # (N,) bool

        # ---- 왼팔 ----
        # Grasp phase:    Fabrics arm target
        # Lift-wait phase: actual grasp arm → joint7-only lift-wait 선형 보간
        lift_progress = (
            (self.episode_length_buf - self.lift_start_step_buf).clamp(min=0).float()
            / max(1, LIFT_PHASE_STEPS - 1)
        ).clamp(max=1.0).unsqueeze(1)

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

        # ---- 왼손 ----
        # Both phases use policy-controlled absolute synergy targets.
        finger_target = self.hand_joint_targets
        self.robot.set_joint_position_target(finger_target, joint_ids=self.hand_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(finger_target), joint_ids=self.hand_dof_indices
        )

        # ---- 오른팔: 고정 자세 ----
        self.robot.set_joint_position_target(
            self.fixed_arm_zero_pos, joint_ids=self.fixed_arm_dof_indices
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
    # Observations: Actor 106D | Critic 143D
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
        σ_cp = self.cfg.obs_noise_cup_pos

        arm_joint_pos    = arm_joint_pos_clean    + torch.randn_like(arm_joint_pos_clean)    * σ_qp
        arm_joint_vel    = arm_joint_vel_clean    + torch.randn_like(arm_joint_vel_clean)    * σ_qv
        finger_joint_pos = finger_joint_pos_clean + torch.randn_like(finger_joint_pos_clean) * σ_qp
        finger_joint_vel = finger_joint_vel_clean + torch.randn_like(finger_joint_vel_clean) * σ_qv
        palm_center_pos  = palm_center_pos_clean  + torch.randn_like(palm_center_pos_clean)  * σ_bp
        fingertip_pos    = fingertip_pos_clean    + torch.randn_like(fingertip_pos_clean)    * σ_bp
        cup_pos_noisy    = cup_pos_clean          + torch.randn_like(cup_pos_clean)          * σ_cp

        # eval_s2r: cup pose obs 오버라이드 — 평가 하네스(scripts/eval_s2r) 전용.
        # 주입값은 이미 "지각 결과"이므로 obs_noise_cup_pos 를 additionally 얹지 않는다.
        # 학습·기존 play 에서는 속성 부재(getattr→None)로 완전 무동작.
        _eval_cup = getattr(self, "eval_cup_pos_override", None)
        if _eval_cup is not None:
            cup_pos_noisy = _eval_cup.to(cup_pos_clean.device)

        # ==== Actor obs 조합 (106D) ====
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

        # 7. fingertip 접촉력 3축 (15D) — ★08.16 binary(5D) 대체.
        # 왜: 재조임 권한을 열어도 정책이 **자기 파지력을 관측할 수 없으면 조절할 수 없다**.
        # binary 는 0.1N 임계 하나라 "닿음"과 "으스러뜨림"을 구분하지 못한다.
        # 실기 근거: Tesollo DG-5F 손끝은 6축 F/T 내장 —
        #   /dg5f_right/fingertip_{1..5}_broadcaster/wrench 로 실제 발행된다.
        #   6축 중 **force 3축만** 쓴다(torque 3축은 미사용 — 쓰려면 재학습 필요).
        # ★프레임 결정 = tip-local: sim 의 force_matrix_w 는 world frame 이지만 실물 F/T 는
        #   센서(손가락) 로컬 출력이다. world 로 학습하면 배포 시 매 스텝 tip FK 회전으로
        #   변환해야 하고 그 변환이 어긋나면 조용히 잘못된 obs 가 된다(과거 손 obs zeros
        #   사고와 동형). sim 을 tip-local 로 맞춰 **실기 값이 그대로 들어가게** 한다.
        # 정규화는 CONTACT_FORCE_MAX(10N) — 실기 노드도 동일 상수를 써야 한다.
        _tip_quat_w = self.robot.data.body_quat_w[:, self.fingertip_body_indices]  # (N,5,4)
        _tip_f_local = quat_apply(
            quat_conjugate(_tip_quat_w.reshape(-1, 4)),
            self.contact_force_xyz_raw.reshape(-1, 3),
        ).view(self.num_envs, NUM_FINGERTIPS, 3)
        tip_force_local = (_tip_f_local / CONTACT_FORCE_MAX).clamp(-1.0, 1.0).view(
            self.num_envs, -1
        )   # (N,15)

        # 7-b. 손 관절 위치 오차 20D — ★인벨롭 그립의 주 힘 관측(08.16).
        # 힘 ∝ stiffness × (지령 − 실측)이고, **어느 마디가 막히든 오차로 나타난다**.
        # 손끝 F/T 만으로는 부족하다: 인벨롭이 잘 될수록 접촉이 중간·원위마디로 가고
        # 팁은 0 을 읽는다(실측 엄지 팁 0.619 vs 아무 마디 0.844 — 40% 구간 팁 무접촉).
        # 즉 손끝 힘은 "닿을 때만" 유효한 보조 신호이고, 전 마디를 덮는 건 이 오차뿐이다.
        # 실기에서도 우리가 보낸 지령과 /dg5f_right/joint_states 실측으로 그대로 계산된다
        # (추가 센서 불필요) — 배포 가능한 관측.
        # 부호를 보존한다: 어느 방향으로 막혔는지가 정보다.
        _hand_pos_now = self.robot.data.joint_pos[:, self.hand_dof_indices]
        joint_pos_err = (
            (self.hand_joint_targets - _hand_pos_now) / JOINT_POS_ERR_MAX
        ).clamp(-1.0, 1.0)   # (N,20)

        # 8. last actions (11D)
        last_actions = self.actions

        # 9. 물체 onehot (8D, MultiAsset 조건화, 2026-07-26) — reset 불변(env_id%N 고정 배정)
        object_onehot = self.multi_object_idx_onehot

        actor_obs = torch.cat([
            arm_joint_pos,          # 7
            arm_joint_vel,          # 7
            finger_joint_pos,       # 20
            finger_joint_vel,       # 20
            palm_center_pos,        # 3
            fingertip_pos_rel_palm, # 15
            palm_to_cup,            # 3
            cup_to_fingertip,       # 15
            tip_force_local,        # 15  손끝 3축 힘(tip-local·10N) — 보조(팁 무접촉 시 0)
            joint_pos_err,          # 20  ★관절 위치 오차 = 전 마디 힘 관측(주)
            last_actions,           # 13
            object_onehot,          # 8
        ], dim=-1)   # 146D

        if actor_obs.shape[1] != NUM_OBSERVATIONS:
            raise RuntimeError(
                f"[v7] Actor obs dim mismatch: {actor_obs.shape[1]} != {NUM_OBSERVATIONS}"
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

        # fingertip signed dist (5D) — clean positions.
        # 2026-07-26: 전역 CUP_RADIUS_APPROX 상수 → 물체별 텐서(cup_radius_approx_buf).
        tip_to_cup_dist = (
            fingertip_pos_clean - cup_pos_clean.unsqueeze(1)
        ).norm(dim=-1)
        fingertip_signed_dist = tip_to_cup_dist - self.cup_radius_approx_buf.unsqueeze(-1)

        # critic actor_obs_clean (146D) — clean state 재조합
        actor_obs_clean = torch.cat([
            arm_joint_pos_clean,
            arm_joint_vel_clean,
            finger_joint_pos_clean,
            finger_joint_vel_clean,
            palm_center_pos_clean,
            (fingertip_pos_clean - palm_center_pos_clean.unsqueeze(1)).view(self.num_envs, -1),
            cup_pos_clean - palm_center_pos_clean,
            (fingertip_pos_clean - cup_pos_clean.unsqueeze(1)).view(self.num_envs, -1),
            tip_force_local,        # 15 — actor 와 동일(접촉력엔 obs noise 미적용)
            joint_pos_err,          # 20 — actor 와 동일
            last_actions,
            object_onehot,
        ], dim=-1)   # 146D

        critic_obs = torch.cat([
            actor_obs_clean,        # 146
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
        ], dim=-1)   # 151D

        if critic_obs.shape[1] != NUM_CRITIC_OBSERVATIONS:
            raise RuntimeError(
                f"[v7] Critic obs dim mismatch: {critic_obs.shape[1]} != {NUM_CRITIC_OBSERVATIONS}"
            )

        return {"policy": actor_obs, "critic": critic_obs}

    # ------------------------------------------------------------------
    # Rewards: RH56F1 shared grasp-v2 contract
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        cup_height_delta = (
            self.object_pos[:, 2] - self.object_init_pos[:, 2]
        ).clamp(min=0.0)
        grasp_center = self.object_pos.clone()
        # 2026-07-26: cfg.cup_grasp_z_offset(cup_big 기준 스칼라) → 물체별 텐서(bbox 반높이 비율).
        grasp_center[:, 2] += self.cup_grasp_z_offset_buf
        palm_to_cup_dist = (self.palm_center_pos - grasp_center).norm(dim=-1)
        cup_xy_displacement = (
            self.object_pos[:, :2] - self.object_init_pos[:, :2]
        ).norm(dim=-1)

        cup_to_palm_xy = self.palm_center_pos[:, :2] - grasp_center[:, :2]
        approach_dir_xy = cup_to_palm_xy / cup_to_palm_xy.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-6)
        enclosure_axis = torch.zeros(self.num_envs, 3, device=self.device)
        enclosure_axis[:, :2] = torch.stack(
            [-approach_dir_xy[:, 1], approach_dir_xy[:, 0]], dim=1
        )
        # 2026-07-26: cfg.cup_radius_approx(cup_big 기준 스칼라) → 물체별 텐서(bbox 반경 평균).
        radius = self.cup_radius_approx_buf.unsqueeze(-1)
        thumb_target = grasp_center + enclosure_axis * radius
        others_target = grasp_center - enclosure_axis * radius
        thumb_dist = (self.fingertip_pos[:, 0] - thumb_target).norm(dim=-1)
        others_dist = (
            self.fingertip_pos[:, 1:] - others_target.unsqueeze(1)
        ).norm(dim=-1).mean(dim=-1)
        fingertip_side_dist = (
            float(self.cfg.enclosure_thumb_weight) * thumb_dist
            + (1.0 - float(self.cfg.enclosure_thumb_weight)) * others_dist
        )

        full_tip_contact_bool = self.num_contacts_buf >= NUM_FINGERTIPS
        full_tip_contact = full_tip_contact_bool.float()
        tip_contact_frac = (
            self.num_contacts_buf.float() / float(NUM_FINGERTIPS)
        ).clamp(max=1.0)
        persistent_grasp = (
            self.num_contacts_buf >= int(self.cfg.stage0_lift_start_min_contacts)
        )
        self.reward_contact_hold_buf = torch.where(
            persistent_grasp,
            self.reward_contact_hold_buf + 1,
            torch.zeros_like(self.reward_contact_hold_buf),
        )
        contact_persistence_frac = (
            self.reward_contact_hold_buf.float()
            / max(float(self.cfg.grasp_contact_persistence_reward_steps), 1.0)
        ).clamp(max=1.0)
        # envelope wrap 품질: 중간(l_hl_X_3)·원위(l_hl_X_4) 마디 접촉 비율
        # → grasp/lift 보상이 손끝-only가 아닌 진짜 감싸기를 credit하도록 전달
        middle_frac = self.middle_binary_contact_buf.float().mean(dim=-1)
        distal_frac = self.distal_binary_contact_buf.float().mean(dim=-1)
        envelope_frac = 0.5 * (middle_frac + distal_frac)
        # grip_frac: 임의 마디(tip|middle|distal) 접촉 손가락 비율. envelope wrap이 tip을
        # mid/dist로 옮겨도 그립으로 인정 → post_lift 페널티·success가 wrap을 처벌 안 함.
        any_finger_contact = (
            self.binary_contact_buf
            | self.middle_binary_contact_buf
            | self.distal_binary_contact_buf
        )
        num_grip_fingers = any_finger_contact.sum(dim=-1)
        grip_frac = num_grip_fingers.float() / float(NUM_FINGERTIPS)

        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cup_z_world = quat_apply(self.object_rot, z_local)
        cup_tilt_deg = torch.rad2deg(
            torch.acos(cup_z_world[:, 2].clamp(min=-1.0, max=1.0))
        )
        upright_quality = torch.exp(
            -cup_tilt_deg
            / max(float(self.cfg.stabilize_upright_reward_scale_deg), 1e-6)
        )
        action_delta_norm = compute_action_delta_norm(self.actions, self.prev_actions)
        contact_delta = (
            self.num_contacts_buf.float() - self._prev_reward_contacts_buf
        ).abs()
        stability = compute_grasp_v2_stability(
            cup_lin_vel=self.cup.data.root_lin_vel_w,
            cup_ang_vel=self.cup.data.root_ang_vel_w,
            contact_delta=contact_delta,
            action_delta_norm=action_delta_norm,
            cfg=self.cfg,
        )
        # success: 엄지-컵 접촉을 명시 요구 + 나머지 완화(≥success_min_grip_fingers).
        # distal/middle 이 Cup-only 필터가 됐으므로 any_finger_contact[:,0](엄지)는 진짜
        # 컵 접촉만 True. 전손가락 동시(>=5)는 wrap 시 tip 감소로 진동 이력이 있어 완화하되,
        # 엄지 컵 접촉(thumb_cup_grip)을 AND 강제해 "엄지 없는 4지 그립"을 success 에서 배제.
        thumb_cup_grip = any_finger_contact[:, 0]
        full_grip_bool = (
            num_grip_fingers >= int(self.cfg.success_min_grip_fingers)
        ) & thumb_cup_grip
        success_now = (
            self.is_lift_phase
            & (cup_height_delta >= self.cfg.lift_success_height)
            & full_grip_bool
            & (cup_tilt_deg <= self.cfg.stabilize_upright_max_deg)
            & stability.stable
        )

        total, reward_terms, _ = compute_grasp_reward_terms(
            num_tip_contacts=self.num_contacts_buf,
            tip_contact_frac=tip_contact_frac,
            full_tip_contact=full_tip_contact,
            contact_persistence_frac=contact_persistence_frac,
            envelope_frac=envelope_frac,
            grip_frac=grip_frac,
            # 08.16 감쌈 깊이(per-finger mid AND distal)와 래치 기준선. 둘 다 주어져야
            # 유지 페널티가 켜진다 — 미주입이면 core 가 기존 동작 그대로 돈다.
            wrap_frac=self.wrap_frac_buf,
            wrap_at_latch=self.wrap_at_latch_buf,
            palm_to_cup_dist=palm_to_cup_dist,
            fingertip_side_dist=fingertip_side_dist,
            cup_height_delta=cup_height_delta,
            cup_xy_displacement=cup_xy_displacement,
            cup_tilt_deg=cup_tilt_deg,
            upright_quality=upright_quality,
            lift_latched=self.is_lift_phase,
            action_delta_norm=action_delta_norm,
            stabilize_reward_gate=self.is_lift_phase,
            success_now=success_now,
            stable=stability.stable,
            stability_quality=stability.quality,
            cfg=self.cfg,
        )
        self._prev_reward_contacts_buf.copy_(self.num_contacts_buf.float())

        _ep_success_rate = self._successful_episodes / max(self._total_episodes, 1)
        if self.grasp_adr is not None:
            # ★08.16 ADR 트리거를 누적→순간 성공률로 전환(right 이식, grasp_v2 방식).
            # 누적(_ep_success_rate)은 한 번 임계를 넘으면 사실상 내려오지 않아, 난이도가
            # 올라 정책이 무너져도 램프가 멈추지 않는 한방향 래칫이었다. success_flag 는
            # 매 스텝 갱신(_get_dones, DirectRLEnv 가 _get_rewards 보다 먼저 호출 → 동일
            # 스텝 최신값)되는 순간 지표라 성능 저하에 즉시 반응해 램프가 자동 정지한다.
            # (누적 지표는 TB 로깅용으로 계속 사용 — 삭제 금지)
            _adr_metric = self.success_flag.float().mean()
            self.extras["adr/trigger_metric"] = _adr_metric
            # increment 시 확장된 물리 DR 범위(질량·마찰)를 전 env 에 즉시 반영
            # (grasp_v2 동일 — 안 하면 다음 자연 리셋까지 구 범위가 남는다).
            if self.grasp_adr.maybe_increment(_adr_metric):
                _em = getattr(self, "event_manager", None)
                if _em is not None:
                    _em.reset(env_ids=self.robot._ALL_INDICES)
                    _em.apply(
                        env_ids=self.robot._ALL_INDICES,
                        mode="reset",
                        global_env_step_count=0,
                    )

        self.extras["reward/approach"] = reward_terms["approach"].mean()
        self.extras["reward/grasp"] = reward_terms["grasp"].mean()
        self.extras["reward/lift"] = reward_terms["lift"].mean()
        self.extras["reward/stabilize"] = reward_terms["stabilize"].mean()
        self.extras["reward/success_bonus"] = reward_terms["success_bonus"].mean()
        self.extras["reward/post_lift_contact_loss"] = reward_terms["post_lift_contact_loss"].mean()
        self.extras["reward/action_smooth"] = reward_terms["action_smooth"].mean()
        self.extras["reward/stability"] = reward_terms["stability"].mean()
        self.extras["reward/total"] = total.mean()
        self.extras["task/lifted_rate"] = (
            cup_height_delta >= self.cfg.lift_success_height
        ).float().mean()
        # lift 게이트 모니터링: grip=아무 마디(tip|mid|distal) 닿은 손가락 수(신규 lift 기준),
        # wrap=mid&distal 동시(견고 감쌈 관찰용).
        _grip_fingers = (
            self.binary_contact_buf
            | self.middle_binary_contact_buf
            | self.distal_binary_contact_buf
        ).sum(dim=-1)
        _wrap_fingers = (
            self.middle_binary_contact_buf & self.distal_binary_contact_buf
        ).sum(dim=-1)
        self.extras["task/grip_finger_count"] = _grip_fingers.float().mean()
        self.extras["task/wrap_finger_count"] = _wrap_fingers.float().mean()
        self.extras["task/lift_ready_rate"] = self.is_lift_phase.float().mean()
        self.extras["task/five_tip_contact_rate"] = full_tip_contact.mean()
        # util: .any() GPU→CPU 동기화 제거 → nan-safe 마스크 평균(빈 마스크는 clamp로 0)
        _prelift_m = (~self.is_lift_phase).float()
        self.extras["task/prelift_five_tip_contact_rate"] = (
            (full_tip_contact.float() * _prelift_m).sum() / _prelift_m.sum().clamp(min=1.0)
        )
        _lift_m = self.is_lift_phase.float()
        self.extras["task/lift_five_tip_contact_rate"] = (
            (full_tip_contact.float() * _lift_m).sum() / _lift_m.sum().clamp(min=1.0)
        )
        self.extras["cup/height_delta"] = cup_height_delta.mean()
        self.extras["cup/tilt_deg"] = cup_tilt_deg.mean()
        self.extras["contact/count"] = self.num_contacts_buf.float().mean()
        # 인벨롭 진단: 중간마디(_3)/원위(_4) 접촉 + 진짜 인벨롭(팁 AND 중간마디 동시) 측정
        _tip = self.binary_contact_buf
        _mid = self.middle_binary_contact_buf
        _dist = self.distal_binary_contact_buf
        self.extras["contact/middle_count"] = _mid.float().sum(dim=-1).mean()
        self.extras["contact/distal_count"] = _dist.float().sum(dim=-1).mean()
        _envelope_fingers = (_tip & _mid).float().sum(dim=-1)
        self.extras["contact/envelope_finger_count"] = _envelope_fingers.mean()
        self.extras["contact/full_envelope_rate"] = (_envelope_fingers >= 4).float().mean()
        # 엄지(idx0) 컵 접촉 진단: distal/middle 이 Cup-only 필터가 됐으므로 모두 진짜 컵 접촉.
        # cup_any = success 게이트가 요구하는 엄지 grip(tip|mid|distal). 재실험 핵심 검증 지표.
        self.extras["debug/thumb/cup_tip"] = _tip[:, 0].float().mean()
        self.extras["debug/thumb/cup_mid"] = _mid[:, 0].float().mean()
        self.extras["debug/thumb/cup_dist"] = _dist[:, 0].float().mean()
        # 손가락별 컵 접촉율 (any=tip|mid|dist) — tb 진단용(play 없이 tb로 손가락별 확인).
        # 순서 thumb,index,middle,ring,pinky. success 게이트(4지 grip)의 어느 손가락이 병목인지 추적.
        _any_fc = (_tip | _mid | _dist).float()
        for _fi, _fn in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
            self.extras[f"debug/finger/{_fn}_cup"] = _any_fc[:, _fi].mean()
        # 감쌈 깊이 진단(08.16). reward-audit Check5 조건 — 유지 페널티의 회피 경로
        # ("얕게 래치하면 잃을 게 없다")가 실제로 발생하는지 보려면 래치 시점 깊이가 필요하다.
        #   wrap_now      : 현재 per-finger 감쌈 깊이
        #   wrap_at_latch : 래치 순간 깊이(페널티 기준선) — 이게 하락하면 회피가 일어나는 것
        #   wrap_drop     : 실제로 물린 페널티 크기(=relu(latch-now))
        # ★과폐쇄 감시(08.16, retighten_after_latch 도입에 따른 예상 부작용).
        # 유지 페널티는 "잃는 것"만 처벌하고 sim 컵은 rigid 라 **끝까지 조이는 게 공짜**다.
        # 정책이 close_frac→1.0 으로 포화하면 (a) 재조임이 학습이 아니라 상수가 되고
        # (b) 실기에서 손가락·물체 손상 위험. 행동은 안 바꾸고 지표로만 감시한다.
        #   close_frac  : 손가락 폐쇄 진행도 평균(1.0 = 완전 폐쇄)
        #   tip_force   : 손끝 접촉력 평균 [N] — 포화 시 함께 치솟는다
        self.extras["debug/finger/close_frac"] = self.finger_close_buf.mean()
        self.extras["debug/finger/close_frac_max"] = self.finger_close_buf.max()
        # ★08.16 PIP/DIP 분리가 실제로 쓰이는지 보는 지표. 관절 순서 finger-major [_1,_2,_3,_4]×5.
        # 세 값이 서로 붙어 있으면 정책이 채널을 안 쓰는 것(= 구 5D 와 동일) → 분리 실패.
        # 벌어져야 인벨롭 자세(MCP 깊게 / PIP·DIP 얕게)를 실제로 만들고 있다는 뜻이다.
        _cb = self.finger_close_buf.view(self.num_envs, NUM_FINGERTIPS, 4)
        self.extras["debug/finger/close_ab"] = _cb[:, :, 0].mean()      # _1 외전
        self.extras["debug/finger/close_mcp"] = _cb[:, :, 1].mean()     # _2 MCP
        self.extras["debug/finger/close_pip"] = _cb[:, :, 2].mean()     # _3 PIP
        self.extras["debug/finger/close_spread"] = (
            _cb[:, :, 1].mean() - _cb[:, :, 2].mean()                   # MCP − PIP 비율 분화
        )
        self.extras["contact/tip_force_mean"] = self.contact_force_raw.mean()
        self.extras["contact/tip_force_max"] = self.contact_force_raw.max()
        self.extras["contact/wrap_now"] = self.wrap_frac_buf.mean()
        _latched_f = self.lift_ready_latched_buf.float()
        _n_latched = _latched_f.sum().clamp(min=1.0)
        self.extras["contact/wrap_at_latch"] = (
            (self.wrap_at_latch_buf * _latched_f).sum() / _n_latched
        )
        self.extras["contact/wrap_drop"] = (
            ((self.wrap_at_latch_buf - self.wrap_frac_buf).clamp(min=0.0) * _latched_f).sum()
            / _n_latched
        )
        # 외란 진단(08.16, reward-audit Check5 조건): ADR 램프 현재값 + wrench 실인가율.
        # 외란 중 접촉 유지의 대리 지표 = contact/envelope_finger_count·debug/finger/*_cup.
        if self.cfg.wrench_enable:
            _wr_a = (
                self.grasp_adr.get_param("object_wrench", "max_linear_accel")
                if self.grasp_adr is not None else float(self.cfg.wrench_max_accel)
            )
            self.extras["adr/wrench_max_accel"] = torch.tensor(_wr_a, device=self.device)
            _rot_a = (
                self.grasp_adr.get_param("hold_rotation", "max_accel")
                if self.grasp_adr is not None
                else float(self.cfg.hold_rotation_perturb_max_accel)
            )
            self.extras["adr/hold_rotation_max_accel"] = torch.tensor(_rot_a, device=self.device)
            self.extras["debug/wrench/applied_frac"] = (
                self.object_applied_force.view(self.num_envs, 3).norm(dim=-1) > 1e-6
            ).float().mean()
        # 물리 DR 커리큘럼 진행 관측(08.16): 현재 실효 질량 스케일/마찰 상한 + 실측 질량.
        if self.grasp_adr is not None:
            self.extras["adr/increment"] = torch.tensor(
                float(self.grasp_adr.increment_counter), device=self.device
            )
            _em = getattr(self, "event_manager", None)
            if _em is not None and "object_scale_mass" in self.grasp_adr.physics_cfg:
                _mr = _em.get_term_cfg("object_scale_mass").params["mass_distribution_params"]
                self.extras["adr/mass_scale_lo"] = torch.tensor(float(_mr[0]), device=self.device)
                self.extras["adr/mass_scale_hi"] = torch.tensor(float(_mr[1]), device=self.device)
                _fr = _em.get_term_cfg("object_physics_material").params["dynamic_friction_range"]
                self.extras["adr/dyn_friction_lo"] = torch.tensor(float(_fr[0]), device=self.device)
            self.extras["adr/cup_mass_mean"] = self._cup_mass.mean()
        # 물체별 순간 성공률(08.16): ADR 난이도가 오를 때 **특정 물체 계열만** 무너지는지
        # 감지한다. grasp_v2 실측에서 cup_big 계열이 ADR 상승과 함께 0.53→0.1~0.3 으로
        # 단조 붕괴했는데 148종에 묻혀 전체 지표로는 보이지 않았다. v1 은 8종 중 4종이
        # cup_big 이라 같은 일이 생기면 치명적 — 전체 success 보다 먼저 여기서 드러난다.
        # index_add_ 배치 연산(.item()/sync 없음).
        _succ_f = self.success_flag.float()
        _n_obj = len(self._object_names)
        _cnt = torch.bincount(self.object_idx, minlength=_n_obj).float().clamp(min=1.0)
        _sum = torch.zeros(_n_obj, device=self.device).index_add_(0, self.object_idx, _succ_f)
        _per_obj = _sum / _cnt
        for _oi, _on in enumerate(self._object_names):
            self.extras[f"obj_success/{_on}"] = _per_obj[_oi]
        self.extras["debug/thumb/cup_any"] = (
            _tip[:, 0] | _mid[:, 0] | _dist[:, 0]
        ).float().mean()
        # 엄지 자세 진단(play 렌더 "엄지 뒤로 돌아감" 학습 중 추적): 실제 관절각 vs 대향 타깃(+1.57)
        # hand_dof_indices는 finger-major [_1,_2,_3,_4]×5 → index 0=엄지_1(회전), 1=엄지_2(대향)
        _hand_pos = self.robot.data.joint_pos[:, self.hand_dof_indices]
        self.extras["debug/thumb/j1_pos"] = _hand_pos[:, 0].mean()
        self.extras["debug/thumb/j2_pos"] = _hand_pos[:, 1].mean()
        # 대향(+1.57)에서 뒤로(음의 방향=0쪽) 밀린 양. ~0이면 유지, 크면 엄지 뒤로 돌아감.
        self.extras["debug/thumb/j2_backward_gap"] = (
            (1.57 - _hand_pos[:, 1]).clamp(min=0.0).mean()
        )
        # joint state(arm/finger per-joint) + action policy(palm 6D + finger 5D raw) 로깅
        for k, v in joint_state_scalars(
            arm_pos=self.robot.data.joint_pos[:, self.arm_dof_indices],
            arm_vel=self.robot.data.joint_vel[:, self.arm_dof_indices],
            finger_pos=_hand_pos,
            finger_vel=self.robot.data.joint_vel[:, self.hand_dof_indices],
            per_joint=True,
        ).items():
            self.extras[k] = v
        for k, v in action_policy_scalars(
            action=self.actions, prev_action=self.prev_actions, palm_dims=6,
        ).items():
            self.extras[k] = v
        self.extras["task/action_delta_norm"] = action_delta_norm.mean()
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

        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cup_z_world = quat_apply(self.object_rot, z_local)
        tipped = cup_z_world[:, 2] < self._cup_tipping_cos

        # success bookkeeping: 접촉 래치(잡고 리프트) + 그립 유지 + 유효 컵.
        grasped = (self.num_contacts_buf >= MIN_CONTACTS_FOR_SUCCESS)
        valid_cup = ~(out_x | out_y | fallen)
        self.success_flag.copy_(self.lift_ready_latched_buf & grasped & valid_cup)
        self.transfer_entry_grasp_success_buf |= self.success_flag
        self.episode_success_buf |= self.success_flag

        if self.cfg.enable_warm_state_export:
            self._maybe_export_warm_states(cup_z_world[:, 2])

        # scripted lift-wait 중에는 tipped 로 종료하지 않음.
        # joint7 이동으로 cup 이 일시적으로 기울 수 있으나 warm-state 저장은
        # lift-wait 도달과 접촉 조건으로 필터링한다.
        # ★08.16: 억제를 **스크립트 램프 구간(LIFT_PHASE_STEPS)** 으로만 한정한다.
        # 기존엔 래치 이후 전 구간을 억제했는데, 그 구간이 정확히 회전 외란
        # (hold_rotation_perturb)이 걸리는 구간이라 **외란의 유일한 실패 신호가 꺼져 있었다**.
        # 램프가 끝난 hold 구간에서는 틸팅 종료를 되살려 "회전에 놓치면 실패"를 학습시킨다.
        # 램프 중 일시적 기울임은 그대로 면제되므로 warm-state 수집 경로는 영향 없다.
        # (정상 구간 실측 cup_tilt_deg 10.9° vs 임계 60° — 여유 큼)
        is_scripted_phase = self.is_lift_phase
        if bool(getattr(self.cfg, "tipping_active_after_lift_ramp", False)):
            _ramp_left = (
                self.episode_length_buf - self.lift_start_step_buf
            ) < int(LIFT_PHASE_STEPS)
            is_scripted_phase = self.is_lift_phase & _ramp_left
        tipped_active = tipped & ~is_scripted_phase
        terminated = out_x | out_y | fallen | tipped_active
        truncated  = self.episode_length_buf >= self.max_episode_length - 1

        # warm export 진단: scripted phase 중 (tipped 제외) 조기 종료 추적
        if self.cfg.enable_warm_state_export:
            early_term = (out_x | out_y | fallen) & is_scripted_phase
            if early_term.any():
                self._warm_diag_terminated_early += int(early_term.sum().item())

        self.extras["object_z"] = self.object_pos[:, 2].mean()

        return terminated, truncated

    # ------------------------------------------------------------------
    # Warm-state export (grasp 성공 → 디스크 캐시 → pour warmstart)
    # ------------------------------------------------------------------
    def _maybe_export_warm_states(self, _cup_up_z: torch.Tensor) -> None:
        """left-grip lift-wait 상태를 디스크 캐시에 누적, 목표치 도달 시 1회 저장.

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
                    "export_mode": "left_grip_lift_wait_actual_grasp",
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
                "mode=left_grip_lift_wait "
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
        # arm/hand/cup/contact 모두 실제 sim 의 left-grip lift-wait 상태를 저장한다.
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
            # ★08.17 receiver 다양화(both/pour-v1): left warm 도 물체 스펙을 태깅해야
            #   pour 가 receiver 컵 종류별로 필터/페어링할 수 있다(right 와 동일 계약).
            object_spec_idx=self.object_idx[ids],
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
                f"[5g_grasp_left_v1] warm-state progress: "
                f"mode=left_grip_lift_wait {count}/{target} ({percent:.1f}%, +{added})",
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
                f"[5g_grasp_left_v1] warm-state export complete: "
                f"mode=left_grip_lift_wait {len(self._warm_export_cache)} states → {path}",
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
            "mode": "left_grip_lift_wait",
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

        # ---- wrench 질량 캐시 갱신: mass DR(reset event, super 내 적용)이 바꾼 실효질량 반영.
        # 전체 텐서 1회 배치 read(per-env 루프/sync 금지 — isaac-reset-item 교훈).
        if self.cfg.wrench_enable:
            self._cup_mass = self.cup.root_physx_view.get_masses().to(self.device).view(-1)

        # ---- episode 성공 집계 후 클리어 ----
        self._total_episodes += n
        self._successful_episodes += int(self.episode_success_buf[env_ids].sum().item())
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

            # ---- 컵 spawn 위치 계산 ----
            # xy_range: design §위치 ADR — enable_adr=True 면 grasp_adr.get_param("spawn",
            # "xy_range")로 0.02→0.08 점진 확대, 아니면 cfg.object_spawn_xy_range(±6cm) fallback.
            _xy_range = (
                self.grasp_adr.get_param("spawn", "xy_range")
                if self.grasp_adr is not None else float(self.cfg.object_spawn_xy_range)
            )
            obj_x = self.cfg.object_spawn_x_center + (
                torch.rand(n, device=self.device) - 0.5
            ) * 2.0 * _xy_range
            obj_y = self.cfg.object_spawn_y_center + (
                torch.rand(n, device=self.device) - 0.5
            ) * 2.0 * _xy_range
            # z: 물체별 bbox 반높이 텐서(object_spawn_z_buf) — 2026-07-26 MultiAsset 이식,
            # cfg.object_spawn_z(cup_big 기준 스칼라) 고정값 대체.
            obj_pos_local = torch.stack(
                [obj_x, obj_y, self.object_spawn_z_buf[env_ids]], dim=1
            )

            # eval_s2r: 고정 스폰 오버라이드 — 평가 하네스(scripts/eval_s2r) 전용.
            # 학습·기존 play 에서는 속성 부재(getattr→None)로 완전 무동작.
            # obj_x/obj_y 도 동기해야 pregrasp cache lookup 이 오버라이드 위치를 따라간다.
            # z 가 NaN 이면 물체별 테이블 높이(object_spawn_z_buf)를 유지.
            _eval_spawn = getattr(self, "eval_fixed_spawn_local", None)
            if _eval_spawn is not None:
                _ov = _eval_spawn[env_ids].to(self.device).clone()
                _z_nan = torch.isnan(_ov[:, 2])
                _ov[_z_nan, 2] = self.object_spawn_z_buf[env_ids][_z_nan]
                obj_pos_local = _ov
                obj_x = _ov[:, 0]
                obj_y = _ov[:, 1]

            # ---- FABRICS pregrasp rollout/cache lookup ----
            noise = torch.stack([
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_x,
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_y,
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_z,
            ], dim=1)
            pregrasp_pos = obj_pos_local + self.pregrasp_offset.unsqueeze(0) + noise

            pregrasp_palm_pose = torch.zeros(n, 6, device=self.device)
            pregrasp_palm_pose[:, :3] = pregrasp_pos
            pregrasp_palm_pose[:, 3] = math.radians(-90.0)
            pregrasp_palm_pose[:, 4] = math.radians(0.0)
            pregrasp_palm_pose[:, 5] = math.radians(-90.0)
            pregrasp_palm_pose = torch.max(
                torch.min(pregrasp_palm_pose, self.palm_maxs.unsqueeze(0)),
                self.palm_mins.unsqueeze(0),
            )

            if self.cfg.cache_pregrasp_reset:
                # cache lookup: spawn 위치(x,y) → 가장 가까운 grid point arm IK.
                # 2026-07-26 MultiAsset: 캐시는 단일 기준 z(cfg.object_spawn_z)로 빌드돼
                # 물체별 z(object_spawn_z_buf, 최대 편차 ~3cm)를 반영하지 않는다 — 초기
                # arm IK 근사치일 뿐이며, palm_delta 액션(±0.15m) 및 episode 중 Fabrics
                # arm 학습이 이 잔차를 보정한다는 가정(육안 검증 필요, design §검증 3).
                xi = ((obj_x - self._cache_xs[0]) / (self._cache_xs[1] - self._cache_xs[0])).round().long().clamp(0, self._cache_n - 1)
                yi = ((obj_y - self._cache_ys[0]) / (self._cache_ys[1] - self._cache_ys[0])).round().long().clamp(0, self._cache_n - 1)
                q_pregrasp[:, :NUM_ARM_DOF] = self._cache_q_arm[xi, yi]
            else:
                q_pregrasp = self._run_reset_fabric(env_ids, pregrasp_palm_pose, q_pregrasp)

            # hand는 APPROACH_POSE로 강제
            q_pregrasp[:, NUM_ARM_DOF:] = approach_hand

        # ---- 2. 로봇/Fabrics 상태 리셋 ----
        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_pos[:, self.actuated_dof_indices] = q_pregrasp
        full_pos[:, self.fixed_arm_dof_indices] = self.fixed_arm_zero_pos[0]
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        self.fabric_q[env_ids] = q_pregrasp
        self.fabric_qd[env_ids].zero_()
        self.fabric_qdd[env_ids].zero_()
        self.object_init_pos[env_ids] = obj_pos_local

        # ---- 5. pregrasp / lift-wait-target 버퍼 저장 ----
        self.pregrasp_arm_pos_buf[env_ids] = q_pregrasp[:, :NUM_ARM_DOF]

        # palm_pose_targets를 pregrasp로 동기화 (첫 Fabrics 스텝 타겟 일관성)
        self.palm_pose_targets[env_ids] = pregrasp_palm_pose

        # delta action 기준점: action=0 → pregrasp 위치 유지
        self.pregrasp_palm_pose_buf[env_ids] = pregrasp_palm_pose

        # Fabrics cspace attractor(null-space)를 pregrasp arm pos로 설정
        # default_config가 ARM_START_POSE이면 null-space 항이 계속 팔을 당겨 초기 흔들림 발생
        # pregrasp arm pos로 설정 → 에피소드 시작 시 null-space 항 ≈ 0 → 안정
        self.open_tesollo_fabric.default_config[env_ids, :NUM_ARM_DOF] = q_pregrasp[:, :NUM_ARM_DOF]

        prelift_arm = compute_joint7_lift_wait_target(
            q_pregrasp[:, :NUM_ARM_DOF],
            joint7_delta=getattr(self.cfg, "lift_wait_joint7_delta", 0.31),
            joint7_min=self.cfg.warm_j7_min,
            joint7_max=self.cfg.warm_j7_max,
        )
        self.prelift_arm_pos_buf[env_ids] = prelift_arm

        # ---- 7. 컵 spawn ----
        obj_pos_world = obj_pos_local + self.scene.env_origins[env_ids]
        upright_rot = torch.zeros(n, 4, device=self.device)
        upright_rot[:, 0] = 1.0
        zero_vel = torch.zeros(n, 6, device=self.device)
        cup_root_state = torch.cat([obj_pos_world, upright_rot, zero_vel], dim=-1)
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
        self.reward_contact_hold_buf[env_ids] = 0
        self._prev_reward_contacts_buf[env_ids] = 0.0
        self.success_flag[env_ids] = False
        self.transfer_entry_grasp_success_buf[env_ids] = False
        self.lift_ready_latched_buf[env_ids] = False
        self.grasp_ready_hold_buf[env_ids] = 0
        # 감쌈 깊이 버퍼도 리셋 — 안 하면 이전 에피소드의 래치 기준선이 남아
        # 새 에피소드가 시작부터 페널티를 문다.
        self.wrap_frac_buf[env_ids] = 0.0
        self.wrap_at_latch_buf[env_ids] = 0.0
        self.lift_start_step_buf[env_ids] = 0
        self.is_lift_phase[env_ids] = False
        self.finger_close_buf[env_ids] = 0.0

        # actions 리셋: delta action 방식 → action=0 = pregrasp 위치
        # (역스케일 불필요: scale(0, delta_mins, delta_maxs) = delta=0 → pregrasp 유지)
        self.actions[env_ids, :6] = 0.0
        self.actions[env_ids, 6:] = -1.0
        self.prev_actions[env_ids, :6] = 0.0
        self.prev_actions[env_ids, 6:] = -1.0
