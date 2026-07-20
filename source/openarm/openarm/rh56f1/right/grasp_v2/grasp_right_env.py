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
from openarm.distillation.visual_dr import VisualDomainRandomizer

import os as _os

from fabrics_sim.fabrics.openarm_rh56f1_pose_fabric import (
    OpenArmRh56f1PoseFabric,
)
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp, capture_fabric
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
    PREGRASP_EULER_EZ_DEG,
    PREGRASP_EULER_EX_DEG,
    PREGRASP_EULER_EX_TOPDOWN_DEG,
    PREGRASP_TOPDOWN_XY,
    PREGRASP_TOPDOWN_CLEARANCE,
    PREGRASP_SIDE_Z,
    PREGRASP_SIDE_CLEARANCE,
    PREGRASP_R_AJ7_BIAS_TOPDOWN,
    PALM_POS_CENTER_SHIFT_SIDE,
    PALM_POS_CENTER_SHIFT_TOPDOWN,
)
from .grasp_right_utils import (
    compute_palm_pose_id,
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

        # 접근 자세별 회전 경계 (07.13, tesollo 78592a3 이식) — 폭(±max_pose_angle)은
        # 그대로, 중심만 옮긴다. side(ex=90)는 기존 [45°,135°], top-down(ex=180)은
        # [135°,225°]. 두 영역이 겹치지 않아 서로의 탐색을 간섭하지 않는다.
        _mpa = math.radians(cfg.max_pose_angle)
        _ex_top = math.radians(PREGRASP_EULER_EX_TOPDOWN_DEG)
        self.palm_mins_by_pose = torch.stack([self.palm_mins.clone(), self.palm_mins.clone()])
        self.palm_maxs_by_pose = torch.stack([self.palm_maxs.clone(), self.palm_maxs.clone()])
        self.palm_mins_by_pose[1, 5] = _ex_top - _mpa
        self.palm_maxs_by_pose[1, 5] = _ex_top + _mpa
        # 박스 위치 재정렬 (07.14, DEXTRAH 재확인): action=0(박스 중심)이 pregrasp
        # reset 위치와 어긋나 있어 settle 종료 직후 미학습 정책이 잘 계산된 pregrasp
        # 를 버리고 박스 중심으로 끌려가는 문제 수정(폭 유지, 실측 위치로 중심 이동).
        _shift_side = to_torch(PALM_POS_CENTER_SHIFT_SIDE, device=self.device)
        _shift_top  = to_torch(PALM_POS_CENTER_SHIFT_TOPDOWN, device=self.device)
        self.palm_mins_by_pose[0, :3] += _shift_side
        self.palm_maxs_by_pose[0, :3] += _shift_side
        self.palm_mins_by_pose[1, :3] += _shift_top
        self.palm_maxs_by_pose[1, :3] += _shift_top
        # per-env 경계 버퍼 (reset 에서 물체 높이에 따라 채움). 기본 = side.
        self.palm_pose_id  = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.palm_mins_env = self.palm_mins_by_pose[0].unsqueeze(0).repeat(self.num_envs, 1)
        self.palm_maxs_env = self.palm_maxs_by_pose[0].unsqueeze(0).repeat(self.num_envs, 1)

        # pregrasp palm pose 버퍼 (reset 시 palm_pose_targets 초기값 + object_init_pos
        # 스냅샷 타이밍 앵커. palm settle override·rate limit 제거(DEXTRAH 정렬) 후
        # palm 제어에는 미사용 — 아래 _pre_physics_step 참조).
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

        # 손 제어 = 6D per-finger 직접(시너지/PCA 폐기, 07.14). action 6D → 각 관절
        # [lower, upper] URDF limit 직접 lerp. 원위 6은 하드웨어 mimic(drive×mult).
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

        # Pregrasp offset 은 reset 에서 물체 clearance 로 계산한다(_compute_pregrasp_offset,
        # tesollo 9f0e4f7 이식). demo warmstart 경로는 side 로 수집된 고정 자세라 cfg 값을
        # 그대로 쓴다.
        self.pregrasp_offset = to_torch(
            [cfg.pregrasp_offset_x, cfg.pregrasp_offset_y, cfg.pregrasp_offset_z],
            device=self.device,
        )
        self._side_y_sign = -1.0 if cfg.pregrasp_offset_y < 0 else 1.0
        # r_aj_7 bias 도 접근 자세별 (side=cfg 기본값, top-down=전용값). 07.13 이식.
        self.pregrasp_r_aj7_bias_by_pose = to_torch(
            [cfg.pregrasp_r_aj7_bias, PREGRASP_R_AJ7_BIAS_TOPDOWN], device=self.device,
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
        # 접근 자세 분기용 물체 인덱스 (side=cup, 그 외 top-down. tesollo cd29c62 이식)
        _side = [
            self._object_names.index(_n)
            for _n in self.cfg.side_approach_object_names
            if _n in self._object_names
        ]
        self.side_object_idx = to_torch(_side, dtype=torch.long, device=self.device)
        # 물체별 clearance = ‖half_extent‖ (임의 회전 시 중심→표면 최대거리, tesollo
        # 9f0e4f7 이식). pregrasp 를 이 값에 비례시켜 스폰 겹침(→PhysX depenetration
        # 폭주, ADR 회전이 올라야 발현되는 잠복 위험) 을 방지한다. scripts/tools/
        # compute_object_bbox.py 산출물(tesollo와 공유, 동일 153종 visdex 자산).
        # 누락 시 조용한 fallback 없이 즉시 실패.
        self.object_clearance = to_torch(
            self._load_object_clearances(), device=self.device
        )
        # DEXTRAH 물체 조건화: one-hot object id + scale (obs 구조 원본 동일, distillation 대비)
        self.multi_object_idx_onehot = torch.nn.functional.one_hot(
            self.object_idx, num_classes=len(self._object_names)
        ).float()   # (num_envs, N_obj), reset 불변

        # ----------------------------------------------------------------
        # Distillation (Dagger 계약: student="policy", teacher="expert_policy").
        # teacher 학습(distillation=False)에선 use_camera=False → 카메라·student obs 미사용.
        # ----------------------------------------------------------------
        self.num_teacher_observations = cfg.num_teacher_observations
        self.num_observations = (
            cfg.num_student_observations if cfg.distillation
            else cfg.num_teacher_observations
        )
        self.use_camera = cfg.distillation
        # RGB visual DR — 인코더가 RGB 를 보므로 외형 고정 시 단일 장면 과적합. shader prim
        # 은 씬 clone 이후 존재 → super().__init__ 완료 후인 여기서 생성.
        self.visual_dr = (
            VisualDomainRandomizer(
                num_envs=self.num_envs,
                texture_root=cfg.texture_root,
                randomize_dome_light=not cfg.disable_dome_light_randomization,
                randomize_robot=not cfg.disable_robot_randomization,
            )
            if cfg.distillation and cfg.img_aug_type == "rgb" and cfg.enable_visual_dr
            else None
        )
        # 성공 유지 스텝 — distillation rollout 조기 종료용 (DEXTRAH success_timeout)
        self.time_in_success_region = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
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

        # ----------------------------------------------------------------
        # 접촉 상태 버퍼
        # ----------------------------------------------------------------
        self.contact_force_xyz_raw = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.contact_force_raw     = torch.zeros(self.num_envs, NUM_FINGERTIPS, device=self.device)
        self.binary_contact_buf    = torch.zeros(self.num_envs, NUM_FINGERTIPS, dtype=torch.bool, device=self.device)
        self.num_contacts_buf      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # grasp_contact_reward persistence: 연속 접촉(num_contacts>0) step 카운터
        self.contact_persist_buf   = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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
                # DEXTRAH physics DR: increment 시 EventTerm 범위 확장
                event_manager=getattr(self, "event_manager", None),
                physics_cfg=cfg.adr_physics_cfg,
            )
            # distillation: teacher 작동점(만렙)에 ADR 고정 시작.
            if getattr(cfg, "starting_adr_increments", 0) > 0:
                self.grasp_adr.set_increment(int(cfg.starting_adr_increments))
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
        self.table = RigidObject(self.cfg.table_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table

        # Actor: fingertip 개별 ContactSensor + filter[Cup/baseLink].
        # 07.14 전환: filter 대상은 **실제 rigid body prim**(Cup/baseLink)이어야 GPU contact
        # filter 가 작동한다 — Xform 루트(/Cup)를 걸면 MultiAsset 에서 "GPU contact filter not
        # supported" 경고 후 force_matrix_w=0 (rh56f1 실측, 내 오진). tesollo grasp_v2 가
        # Cup/baseLink 로 정상 작동함을 확인해 정합. force_matrix_w[:,0,0,:] = 자기 env Cup
        # 접촉만(테이블·자기 접촉 제외) — net_forces 오염 대체.
        _CUP_FILTER = [f"/World/envs/env_.*/Cup/{self.cfg.cup_rigid_body_name}"]
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

        # distal/middle 도 손가락별 개별 센서 + filter[Cup]. force_matrix_w[:,0,0,:] 로 읽는다.
        # RH56F1 2-마디 손: distal = 말단 링크(tip과 동일), middle = 근위 knuckle 링크(envelope).
        #   distal(말단): thumb_4, {index,middle,ring,pinky}_2
        #   middle(근위): thumb_2, {index,middle,ring,pinky}_1
        _DISTAL_LINKS = ["r_hl_thumb_4", "r_hl_index_2", "r_hl_middle_2", "r_hl_ring_2", "r_hl_pinky_2"]
        _MIDDLE_LINKS = ["r_hl_thumb_2", "r_hl_index_1", "r_hl_middle_1", "r_hl_ring_1", "r_hl_pinky_1"]
        self._distal_sensors: list[ContactSensor] = []
        for i, link in enumerate(_DISTAL_LINKS):
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link}",
                filter_prim_paths_expr=_CUP_FILTER,
                history_length=1,
                track_air_time=False,
            ))
            self._distal_sensors.append(sensor)
            self.scene.sensors[f"distal_sensor_{i + 1}"] = sensor

        self._middle_sensors: list[ContactSensor] = []
        for i, link in enumerate(_MIDDLE_LINKS):
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link}",
                filter_prim_paths_expr=_CUP_FILTER,
                history_length=1,
                track_air_time=False,
            ))
            self._middle_sensors.append(sensor)
            self.scene.sensors[f"middle_sensor_{i + 1}"] = sensor

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # ★다물체 스폰 순서 (07.10 버그 수정): clone → cup(MultiAsset) 생성.
        # RigidObject(cup_cfg)는 생성 시점에 즉시 spawn하는데(asset_base.py — leaf "Cup"은
        # non-regex), clone 이전엔 env_0만 존재해 MultiAssetSpawner가 물체[0] 하나만 spawn
        # → clone(copy_from_source=True)이 그걸 전 env에 복제 = 전 env 동일 물체 버그
        # (probe 실측: 16env 전부 visdex[0]='104738', 물체별 성공률 std 0.005 허구 라벨).
        # clone을 먼저 하면 spawn 시점에 env prim 전부 존재 → env_i % N 결정적 배정 정상화.
        self.scene.clone_environments(copy_from_source=True)
        self.cup = RigidObject(self.cfg.cup_cfg)
        self.scene.rigid_objects["cup"] = self.cup

        # Distillation D435i 카메라(RGB+depth) 또는 occlusion 측정 카메라. teacher 학습 off.
        if getattr(self.cfg, "distillation", False) or getattr(self.cfg, "enable_camera_probe", False):
            from isaaclab.sensors import TiledCamera
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera_cfg)
            self.scene.sensors["tiled_camera"] = self._tiled_camera

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

        # Main fabric — 팔 IK 만 적분(arm). 손은 fabric 밖에서 6D per-finger 직접 PD 제어.
        # (시너지/PCA fabric hand attractor 폐기 07.14 — RH56F1 언더액추라 이미 물리적 시너지.)
        self.fabric = OpenArmRh56f1PoseFabric(
            self.num_envs, self.device, self.timestep,
            graph_capturable=bool(self.cfg.use_cuda_graph),
            use_hand_fabric=False,
        )
        num_joints = self.fabric.num_joints   # 26 (bi-arm: r_arm7,r_hand6,l_arm7,l_hand6)

        self.fabric_integrator = DisplacementIntegrator(self.fabric)

        # Fabric 상태 버퍼 (26D). 우측[0:13]=팔+손 시작자세, 좌측[13:26]=default 중립 유지.
        self.fabric_q   = self.fabric.default_config.clone().contiguous()
        self.fabric_q[:, :NUM_ROBOT_DOF] = self.robot_start_joint_pos
        self.fabric_qd  = torch.zeros(self.num_envs, num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, num_joints, device=self.device)

        # fabric.set_features 의 hand target 인자용 placeholder (use_hand_fabric=False 라 무시됨)
        self.hand_target_dummy = torch.zeros(self.num_envs, 5, device=self.device)
        self.palm_pose_targets = torch.zeros(self.num_envs, 6, device=self.device)
        self.fabric_damping_gain = self.cfg.fabrics_damping_gain * torch.ones(self.num_envs, 1, device=self.device)

        # CUDA Graph 캡처(use_cuda_graph=True): fabric set_features + integrator.step 을
        # 한 번 캡처해두고 매 step replay → 다물체(replicate_physics=False)로 떨어진 GPU
        # 활용을 fabric IK 쪽에서 만회(DEXTRAH env.py 동일 기법). rh56f1 은 palm 을 euler_zyx
        # 고정 버퍼로 직접 넘겨(tesollo 의 quaternion fancy indexing 문제 없음) 바로 캡처 가능.
        if bool(self.cfg.use_cuda_graph):
            self._fabric_inputs = [
                self.hand_target_dummy, self.palm_pose_targets, "euler_zyx",
                self.fabric_q.detach(), self.fabric_qd.detach(),
                self.object_ids, self.object_indicator, self.fabric_damping_gain,
            ]
            self.g, self.fabric_q_new, self.fabric_qd_new, self.fabric_qdd_new = capture_fabric(
                self.fabric,
                self.fabric_q, self.fabric_qd, self.fabric_qdd,
                self.timestep, self.fabric_integrator,
                self._fabric_inputs, self.device,
            )

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



    # ------------------------------------------------------------------
    # 물체 치수 → 접근 자세 분기 (07.13, tesollo cd29c62·9f0e4f7 이식)
    # ------------------------------------------------------------------
    def _load_object_clearances(self) -> list[float]:
        """물체별 clearance = ‖half_extent‖ (m). 누락 물체는 즉시 실패시킨다.

        조용한 fallback(0 채우기)은 pregrasp 를 물체 안에 박아 넣으므로 금지.
        scripts/tools/compute_object_bbox.py 산출물(tesollo와 공유, 동일 153종 visdex).
        """
        path = Path(self.cfg.object_bbox_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"물체 bbox 파일 없음: {path}\n"
                "  python3 scripts/tools/compute_object_bbox.py 로 먼저 생성하세요."
            )
        table = json.loads(path.read_text(encoding="utf-8"))
        missing = [n for n in self._object_names if n not in table]
        if missing:
            raise KeyError(f"bbox 누락 물체 {len(missing)}종: {missing[:5]} … — bbox 재생성 필요")
        return [
            float(sum(float(v) ** 2 for v in table[n]) ** 0.5)
            for n in self._object_names
        ]

    def _compute_palm_pose_id(self, obj_idx: torch.Tensor) -> torch.Tensor:
        """물체 이름 기반 접근 자세: side(cup 등) → 0, 그 외 → 1(top-down)."""
        return compute_palm_pose_id(obj_idx, self.side_object_idx)

    def _sample_spawn_rotation(self, n: int) -> torch.Tensor:
        """물체 spawn 회전 quat (w,x,y,z). ADR 0→1 (DEXTRAH randomize_rotation)."""
        _rot_f = self._adr("object_spawn", "rotation")
        if _rot_f <= 0.0:
            rot = torch.zeros(n, 4, device=self.device)
            rot[:, 0] = 1.0
            return rot

        _r0 = (torch.rand(n, device=self.device) - 0.5) * 2.0 * math.pi * _rot_f
        _r1 = (torch.rand(n, device=self.device) - 0.5) * 2.0 * math.pi * _rot_f
        _half0, _half1 = _r0 * 0.5, _r1 * 0.5
        _zeros = torch.zeros(n, device=self.device)
        _qx = torch.stack([torch.cos(_half0), torch.sin(_half0), _zeros, _zeros], dim=1)
        _qy = torch.stack([torch.cos(_half1), _zeros, torch.sin(_half1), _zeros], dim=1)
        return quat_mul(_qx, _qy)

    def _compute_pregrasp_offset(
        self, obj_idx: torch.Tensor, pose_id: torch.Tensor
    ) -> torch.Tensor:
        """물체 clearance 비례 pregrasp offset (n,3). tesollo 9f0e4f7 이식.

        고정 offset 은 회전 ADR 이 오르면 물체가 palm 을 침범해 PhysX depenetration
        폭주를 일으킨다(tesollo 실증: ADR 36부터 리턴 스파이크, iter 14111 붕괴 -4.9e7).
        top-down 은 물체 위 (clearance + 손끝 여유), side 는 옆 (clearance + palm 여유).
        """
        clr = self.object_clearance[obj_idx]                    # (n,)
        is_top = pose_id == 1
        off = torch.zeros(clr.shape[0], 3, device=self.device)
        off[:, 0] = torch.where(
            is_top,
            torch.full_like(clr, float(PREGRASP_TOPDOWN_XY[0])),
            torch.full_like(clr, float(self.cfg.pregrasp_offset_x)),
        )
        off[:, 1] = torch.where(
            is_top,
            torch.full_like(clr, float(PREGRASP_TOPDOWN_XY[1])),
            self._side_y_sign * (clr + float(PREGRASP_SIDE_CLEARANCE)),
        )
        off[:, 2] = torch.where(
            is_top,
            clr + float(PREGRASP_TOPDOWN_CLEARANCE),
            torch.full_like(clr, float(PREGRASP_SIDE_Z)),
        )
        return off


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
        # force_matrix_w[:, 0, 0, :] = 각 센서(단일 body)가 filter 대상(자기 env Cup)에
        # 대해 받는 접촉력만 (07.14 전환). 개별 센서 + filter[Cup] 라 replicate_physics=False
        # 에서도 작동 검증됨(probe). net_forces(테이블·자기 접촉 오염)를 대체 → 진짜 물체 접촉.
        tip_xyz = torch.stack([
            s.data.force_matrix_w[:, 0, 0, :] for s in self._tip_sensors
        ], dim=1)   # (N, 5, 3)
        tip_norms = tip_xyz.norm(dim=-1)   # (N, 5)

        self.contact_force_xyz_raw.copy_(tip_xyz)
        self.contact_force_raw.copy_(tip_norms)
        self.binary_contact_buf.copy_(tip_norms > CONTACT_FORCE_THRESHOLD)
        self.num_contacts_buf.copy_(self.binary_contact_buf.sum(dim=-1).long())

        # Critic: distal (손가락별 개별 센서, Cup 접촉만)
        per_distal = torch.stack([
            s.data.force_matrix_w[:, 0, 0, :] for s in self._distal_sensors
        ], dim=1).norm(dim=-1)   # (N, 5)
        self.distal_contact_force_raw.copy_(per_distal)
        self.distal_binary_contact_buf.copy_(per_distal > CONTACT_FORCE_THRESHOLD)

        # Critic: middle (손가락별 개별 센서, Cup 접촉만)
        per_middle = torch.stack([
            s.data.force_matrix_w[:, 0, 0, :] for s in self._middle_sensors
        ], dim=1).norm(dim=-1)   # (N, 5)
        self.middle_contact_force_raw.copy_(per_middle)
        self.middle_binary_contact_buf.copy_(per_middle > CONTACT_FORCE_THRESHOLD)

    # ------------------------------------------------------------------
    # 파지력 확보: 물체 외란 wrench (DEXTRAH apply_object_wrench 이식)
    # ------------------------------------------------------------------
    def _apply_object_wrench(self) -> None:
        # DEXTRAH 원본 게이트: 손이 물체 반경(hand_to_object_dist_threshold 0.3m)
        # 내면 외란 인가 — 접근 후 파지·운반 전 구간에서 robust hold 를 단련.
        # (구: in_success_region 게이트 — goal 도달 후만 인가라 원본보다 관대했음)
        # 크기는 ADR 커리큘럼 0→10 (원본 object_wrench.max_linear_accel)
        apply = (
            self.hand_to_object_err <= float(self.cfg.hand_to_object_dist_threshold)
        ).view(-1, 1, 1)
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
        finger_action = actions[:, 6:12]  # (N, 6) ∈ [-1, 1] — 6 액추에이터 직접

        # ---- settle 종료 시 안착 스냅샷 → object_height 로깅 baseline 확정 ----
        # (goal 은 DEXTRAH식 고정 절대점이라 갱신 없음)
        snap = self.episode_length_buf == int(self.cfg.settle_steps)
        if snap.any():
            self.object_init_pos[snap] = self.object_pos[snap]
            # ---- re-anchor (07.12): drop-settle 중 롤링으로 물체가 스폰점을 벗어나
            # anchor 가 빈 곳을 가리킴(probe: xy drift mean 7.3cm/max 37cm → h2o 정체
            # ·grip 0.02 의 기하 원인). 안착 위치 기준으로 anchor xy 재정렬 — slew 가
            # 있어 palm 은 rate 이내로 부드럽게 따라감. offsets 를 당기는 방식은 스폰
            # 충돌(probe A: 손끝이 낙하 지점에 들어가 물체 쳐올림)이라 불가.
            if self.cfg.reanchor_after_settle:
                # per-env offset(07.13): clearance 기반 offset 이 물체·자세별로 달라
                # 배정된 접근 자세를 반영해야 anchor 가 올바른 지점을 가리킨다.
                _off_xy = self._compute_pregrasp_offset(
                    self.object_idx[snap], self.palm_pose_id[snap]
                )[:, :2]
                _new_xy = self.object_pos[snap, :2] + _off_xy
                _new_xy = torch.max(
                    torch.min(_new_xy, self.palm_maxs[:2].unsqueeze(0)),
                    self.palm_mins[:2].unsqueeze(0),
                )
                self.pregrasp_palm_pose_buf[snap, :2] = _new_xy

        # ---- Fabrics arm 제어: palm 절대 pose (DEXTRAH 원본 구조, tesollo 1aa9dcc 이식) ----
        # action[0:6] ∈ [-1,1] 을 palm workspace 박스로 직접 스케일한다.
        # 즉 정책 출력이 곧 "손바닥을 놓을 절대 위치/자세"다.
        #
        # 이전의 anchor+delta 방식(pregrasp ± 0.35m)은 물체까지 20~30cm 를 매 스텝
        # 재적분해야 해서 credit assignment 가 무너졌다 — d9~d15 전체가 "가만히
        # 있기"로 수렴한 근본 원인 중 하나(tesollo 동일 병리 실증: curl 기준 수정
        # 후에도 hand_to_object ep200 0.216 → ep400 0.017 급락). DEXTRAH 는 절대
        # pose 라 "물체 위로 가라"가 1스텝 결정이다.
        #
        # 경계는 per-env(side/top-down 회전 중심이 다르다). scale 결과가 이미
        # 박스 안이므로 별도 clamp 불필요. action=0 → 박스 중심 — 회전 중심은
        # pregrasp 자세와 동일하므로 초기 정책(출력≈0)은 올바른 접근 자세에서 시작.
        # DEXTRAH 원본: 정책 출력을 매 스텝 palm workspace 박스 절대 pose 로 그대로
        # 지령한다 (settle override·rate limit 제거). DEXTRAH 는 t=0 부터 palm 을
        # 자유·절대 제어하며 물체는 자유낙하 중에도 정책 제어하 안착한다
        # (reset_idx: object z=0.5 drop, pre-settle 없음). hand_to_object max-metric
        # 이 물체 안착 위치로 손 전체를 끌어당겨 caging 을 유도. 이전의 settle 중
        # palm 강제 고정 + rate clamp 는 tesollo 계보 scaffolding 으로, 정책이
        # grasp→lift 를 탐색할 자유도를 좁혀 object_height 가 1000ep 내내 0 이던
        # 원인 후보 — DEXTRAH 직접제어로 격리 검증.
        palm_pose = scale(palm_action, self.palm_mins_env, self.palm_maxs_env)  # (N, 6)
        self.palm_pose_targets.copy_(palm_pose)

        # fabric cspace damping: ADR 커리큘럼 10→20 (DEXTRAH fabric_damping.gain)
        if self.grasp_adr is not None:
            self.fabric_damping_gain.fill_(
                self.grasp_adr.get_param("fabric_damping", "gain")
            )

        # 팔 IK 만 fabric 으로 적분 (hand target 인자는 use_hand_fabric=False 라 무시됨).
        if not bool(self.cfg.use_cuda_graph):
            self.fabric.set_features(
                self.hand_target_dummy,
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
        else:
            # CUDA Graph replay: 캡처된 fabric IK 커널을 최신 입력(palm_pose_targets 등 고정
            # 버퍼는 위에서 이미 in-place 갱신됨)으로 재생.
            for _ in range(self.cfg.fabric_decimation):
                self.g.replay()
                self.fabric_q.copy_(self.fabric_q_new)
                self.fabric_qd.copy_(self.fabric_qd_new)
                self.fabric_qdd.copy_(self.fabric_qdd_new)

        # ---- 손: 6D per-finger 직접 제어 (시너지/PCA/래칫 폐기 07.14) ----
        # action 6D ∈[-1,1] → 각 액추에이터 [lower, upper] URDF limit 직접 lerp.
        # 6 액추에이터(thumb_1 외전 opposition 축 포함) 전부 정책이 독립·가역 제어.
        # 시너지(PCA5)는 RH56F1 하드웨어 언더액추(원위 mimic)에 이중으로 씌운 압축이라
        # thumb_1 을 4지 평균에 묶어 opposition 을 죽였음 → 폐기. 래칫/거리게이트도 제거
        # (완전 가역 — 정책이 여닫기 타이밍 자유 학습). 실물 angleSet(6값 각도)과 정합.
        # 원위 6 은 _apply_action 에서 drive×mult mimic(하드웨어 언더액추 재현).
        blend = 0.5 * (finger_action.clamp(-1.0, 1.0) + 1.0)            # (N,6) ∈ [0,1]
        hand_target = torch.lerp(
            self.hand_joint_lower_limits.unsqueeze(0).expand(self.num_envs, -1),
            self.hand_joint_upper_limits.unsqueeze(0).expand(self.num_envs, -1),
            blend,
        )
        # 다물체 drop-settle: episode 초기 settle_steps 동안 손을 approach(열림)로 고정 →
        # 물체(고정 높이 spawn)가 낙하해 테이블에 안착(정지물체 전제). settle 후 정책 제어.
        in_settle = (
            self.episode_length_buf < int(self.cfg.settle_steps)
        ).unsqueeze(-1)
        hand_target = torch.where(
            in_settle,
            self.hand_open_pose.unsqueeze(0).expand(self.num_envs, -1),
            hand_target,
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
        # DEXTRAH pd_targets: velocity feedforward = factor × fabric qd (ADR 1→0).
        # (팔만 — 손 pos target 은 fabric 이 아니라 contact-gated close 라 fabric
        # qd 를 손에 인가하면 pos/vel target 불일치)
        _vel_f = self._adr("pd_targets", "velocity_target_factor")
        self.robot.set_joint_velocity_target(
            _vel_f * self.fabric_qd[:, :NUM_ARM_DOF], joint_ids=self.arm_dof_indices
        )

        # ---- 오른손: 6D per-finger 직접 위치 목표 (정책 절대 제어) ----
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

        # hand↔object 거리 (palm+5tip MAX — reward·wrench 게이트 공용, DEXTRAH
        # hand_to_object_pos_error 대응)
        _hand_points = torch.cat(
            [self.palm_center_pos.unsqueeze(1), self.fingertip_pos], dim=1
        )   # (N, 6, 3)
        self.hand_to_object_err = (
            _hand_points - self.object_pos.unsqueeze(1)
        ).norm(dim=-1).max(dim=-1).values

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

        if not self.use_camera:
            return {"policy": actor_obs, "critic": critic_obs}

        # ==== distillation: student(vision)="policy", teacher="expert_policy" ====
        # RGB 입력 + depth aux 재구성. depth 는 유효 밴드 밖을 0 으로, mask 는 배경(밴드
        # 초과) 픽셀 — aux depth 재구성 손실에서 배경 제외에 쓴다.
        depth = self._tiled_camera.data.output["depth"].clone()          # (N,H,W,1)
        _cf = float(self.cfg.camera_crop_frac)
        if _cf < 0.999:
            _d = depth.permute(0, 3, 1, 2)                               # (N,1,H,W)
            _n, _c, _h, _w = _d.shape
            _ch, _cw = int(_h * _cf), int(_w * _cf)
            _t, _l = (_h - _ch) // 2, (_w - _cw) // 2
            _d = _d[:, :, _t:_t + _ch, _l:_l + _cw]                      # 중앙 crop
            _d = torch.nn.functional.interpolate(_d, size=(_h, _w), mode="nearest")
            depth = _d.permute(0, 2, 3, 1).contiguous()
        mask = depth.permute((0, 3, 1, 2)) > self.cfg.d_max
        depth[depth <= 1e-8] = 10.0        # 렌더 미스(0) → 무효로 밀어냄
        depth[depth > self.cfg.d_max] = 0.0
        depth[depth < self.cfg.d_min] = 0.0

        return {
            "policy": self.compute_student_policy_observations(),
            "expert_policy": actor_obs,
            "critic": critic_obs,
            "img": depth.permute((0, 3, 1, 2)),
            "rgb": self._tiled_camera.data.output["rgb"].clone().permute((0, 3, 1, 2)) / 255.0,
            "aux_info": {"object_pos": self.object_pos},
            "mask": mask,
        }

    def compute_student_policy_observations(self) -> torch.Tensor:
        """student obs (116) — 물체 privileged state 없음.

        teacher obs 에서 object_pos/rot/onehot/scale 을 뺀 것. 물체 정보는 D435i
        RGB 에서 추론(카메라 aux 로 depth/object_pos 재구성). object_goal 은 고정
        절대점이라 실기에서도 알 수 있어 남긴다. (rh56f1 hand 는 촉각 obs 없음.)
        """
        student_obs = torch.cat([
            self.robot_dof_pos_noisy,        # 13
            self.robot_dof_vel_noisy,        # 13
            self.hand_pos_noisy,             # 18
            self.hand_vel_noisy,             # 18
            self.object_goal,                # 3
            self.actions,                    # 12
            self.fabric_q[:, :NUM_ROBOT_DOF],    # 13
            self.fabric_qd[:, :NUM_ROBOT_DOF],   # 13
            self.fabric_qdd[:, :NUM_ROBOT_DOF],  # 13
        ], dim=-1)   # 116
        if student_obs.shape[1] != self.cfg.num_student_observations:
            raise RuntimeError(
                f"Student obs dim mismatch: {student_obs.shape[1]} "
                f"!= {self.cfg.num_student_observations}"
            )
        return student_obs

    # ------------------------------------------------------------------
    # Rewards: DEXTRAH 4항 (dextrah_kuka_allegro compute_rewards 이식)
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # 1) hand_to_object: palm+5손끝 → 물체중심 MAX 거리 (OpenArm 포팅 규약 .max())
        #    거리는 _compute_intermediate_values 공용값(hand_to_object_err — wrench
        #    게이트와 동일 소스) 재사용
        hand_to_object_err = self.hand_to_object_err
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

        # 3) lift: grasp_v1식 접촉 게이팅 × height_delta(4cm 포화). 감쌈→리프트 사다리 연결.
        # 07.14 교체: 기존 goal(60cm) 절대거리 exp 는 초기 리프트 유인 미미 + 접촉 무관
        # baseline(0.05)이 깔려 "감싸고 정지"가 정체. 실접촉으로 게이팅하고 안착점 대비 실제
        # 리프트 높이(4cm 포화)에 비례 → "일단 들기" 유인. goal 운반은 object_to_goal 담당.
        # 07.15 envelope 게이팅(사용자 지적+tesollo grip_frac): tip(말단)만 게이팅하면 "손끝집기
        # (쥐며 이동)"에 머물러 깊은 wrap 을 못 배움 → 손가락별 말단(tip/distal) OR 근위 knuckle
        # (middle) 임의 마디가 닿으면 "감쌈"으로 인정. 여러 마디로 감싸야 lift 보상 → 감싸기 유도.
        envelope_contact = (
            self.binary_contact_buf
            | self.middle_binary_contact_buf
            | self.distal_binary_contact_buf
        )
        graded_contact = (
            envelope_contact.sum(dim=-1).float() / NUM_FINGERTIPS
        ).clamp(0.0, 1.0)
        height_delta = (
            self.object_pos[:, 2] - self.object_init_pos[:, 2]
        ).clamp(min=0.0)
        lift_height_quality = (
            height_delta / max(float(self.cfg.lift_success_height), 1e-6)
        ).clamp(0.0, 1.0)
        lift_reward = _lift_w * graded_contact * lift_height_quality

        # 4) finger_curl_reg (DEXTRAH: -0.01 → -0.005)
        # 07.13 기준 수정(tesollo 77357f0 이식, 치명 버그): DEXTRAH curled_q(=원본
        # index/middle/ring 편 자세+thumb 대향)는 "과도한 말림 억제" 정규화다.
        # FULL_GRIP(주먹) 기준이면 부호가 뒤집힌다 — 빈손으로 주먹 쥐면 거리 0(무패널티),
        # 물체를 잡으면 손가락이 막혀 FULL_GRIP 도달 불가→페널티 발생 → "물체 회피가
        # 최적"이 되는 정확히 반대 유인. hand_open_pose(=HAND_APPROACH_POSE, thumb
        # 대향+나머지 편 자세)가 DEXTRAH curled_q 와 동일 구조라 이걸 기준으로 쓴다.
        finger_pos = self.robot.data.joint_pos[:, self.hand_dof_indices]
        finger_curl_dist = (
            finger_pos - self.hand_open_pose.unsqueeze(0)
        ).norm(p=2, dim=-1).clamp(max=float(self.cfg.finger_curl_dist_max))
        finger_curl_reg = _curl_w * finger_curl_dist ** 2

        # 5) grasp_contact: force_matrix 실접촉(자기 env Cup) 개수 기반 굴곡 유인.
        # grasp_v1 grasp_quality 구조 이식(0.25·tip_frac + 0.35·full_tip + 0.40·persistence).
        # hand_to_object(거리)가 손가락 굴곡 없이 팔 포지셔닝만으로 포화(3000ep 정체)라,
        # 손끝이 실제로 물체를 건드려야만 오르는 항을 신설해 index/middle/pinky 굴곡을 유도.
        tip_frac = (self.num_contacts_buf.float() / NUM_FINGERTIPS).clamp(0.0, 1.0)
        full_tip = (self.num_contacts_buf >= NUM_FINGERTIPS).float()
        _has_contact = self.num_contacts_buf > 0
        self.contact_persist_buf = torch.where(
            _has_contact,
            self.contact_persist_buf + 1,
            torch.zeros_like(self.contact_persist_buf),
        )
        persist_frac = (
            self.contact_persist_buf.float()
            / max(int(self.cfg.grasp_contact_persist_steps), 1)
        ).clamp(0.0, 1.0)
        grasp_quality = 0.25 * tip_frac + 0.35 * full_tip + 0.40 * persist_frac
        grasp_contact_reward = float(self.cfg.grasp_contact_weight) * grasp_quality

        # 6) force_closure(opposition) ← 리프트 주력(07.15 tesollo 이식). 접촉 반력(force_matrix)
        #    은 물체가 손끝을 "미는" 방향 — 엄지와 4지가 물체를 양쪽에서 마주 조이면 두 반력이
        #    대략 반대(−cos>0) → force closure. lstm_test2 실측: 접촉·지속(persist 0.90)은 됐으나
        #    thumb_2(엄지 조임)가 진동(조임 유인 부재)해 리프트 3mm 정체 = 이 대향 조임이 결측.
        #    힘 크기(tanh) 곱하고 엄지·4지 실접촉 AND 게이트로 hacking(한쪽만/빈손) 차단.
        _f = self.contact_force_xyz_raw                       # (N,5,3) Cup-only 접촉력
        _f_thumb = _f[:, 0]                                   # (N,3)
        _f_others = _f[:, 1:].mean(dim=1)                     # (N,3) 4지 평균
        _fc_scale = float(self.cfg.force_closure_force_scale)
        _t_norm = _f_thumb.norm(dim=-1)
        _o_norm = _f_others.norm(dim=-1)
        _cos = (_f_thumb * _f_others).sum(dim=-1) / (_t_norm * _o_norm + 1e-6)
        opposition = (-_cos).clamp(min=0.0)                  # 마주볼 때만(0~1)
        grip_strength = torch.tanh(_t_norm / _fc_scale) * torch.tanh(_o_norm / _fc_scale)
        fc_gate = (
            self.binary_contact_buf[:, 0] & self.binary_contact_buf[:, 1:].any(dim=-1)
        ).float()
        opposition_quality = fc_gate * opposition * grip_strength
        force_closure_reward = float(self.cfg.force_closure_weight) * opposition_quality

        # (palm_orient reward 제거 — DEXTRAH 4항엔 손목 방향 제약이 없고 weight=0 상태로
        #  폐기돼 있었음. 손바닥 법선축 규약은 palm_sensor +Z(손바닥에서 나오는 방향),
        #  +Y=손가락 방향 — 재도입 시 palm_normal = quat_apply(palm_quat, +Z) 사용.)
        total = (
            hand_to_object_reward + object_to_goal_reward + finger_curl_reg
            + lift_reward + grasp_contact_reward + force_closure_reward
        )

        # success: DEXTRAH in_success_region (goal 도달 = 최소 11cm 리프트 내포)
        self.in_success_region = object_to_goal_err < float(self.cfg.object_goal_tol)

        # contact 진단용 grip(tip|mid|distal 접촉 손가락 수).
        # force_matrix_w 전환(07.14): 센서가 이미 자기 env Cup 접촉만 잡으므로 net_forces
        # 시절의 거리 게이트(_finger_near, 테이블 오염 보정용)가 불필요 → 제거.
        # 로깅·진단 전용 — reward 경로 아님.
        _tip_obj    = self.binary_contact_buf
        _mid_obj    = self.middle_binary_contact_buf
        _distal_obj = self.distal_binary_contact_buf
        num_grip_fingers = (_tip_obj | _mid_obj | _distal_obj).sum(dim=-1)

        # 접촉 출처 진단 (GRASP_DEBUG_CONTACT=1 로 play 시): net_forces 센서는 물체/테이블
        # 구분 불가 → tip↔물체 최근접 거리와 grip 동시 출력으로 "grip이 물체 접촉인지" 판별.
        if _os.environ.get("GRASP_DEBUG_CONTACT") and int(self.episode_length_buf[0]) % 30 == 0:
            _tipd = (self.fingertip_pos - self.object_pos.unsqueeze(1)).norm(dim=-1).min(dim=1).values
            _near = (_tipd < 0.06)  # 물체 반경급 근접
            _g = num_grip_fingers.float()
            _pf = self.binary_contact_buf.float().mean(dim=0)  # (5,) 손가락별 tip 접촉률
            print(
                f"DBGC step={int(self.episode_length_buf[0]):3d} "
                f"tipdist mean={_tipd.mean():.3f} min={_tipd.min():.3f} "
                f"grip_all={_g.mean():.2f} grip_near={(_g * _near.float()).sum() / _near.float().sum().clamp(min=1):.2f} "
                f"near_frac={_near.float().mean():.2f} objz={self.object_pos[:, 2].mean():.3f} "
                f"| tip접촉 thumb={_pf[0]:.2f} index={_pf[1]:.2f} middle={_pf[2]:.2f} ring={_pf[3]:.2f} pinky={_pf[4]:.2f}",
                flush=True,
            )

        _ep_success_rate = self._successful_episodes / max(self._total_episodes, 1)
        # ADR increment: DEXTRAH 원본 = in_success_region 순간 평균 > success_for_adr(0.4)
        # increment 시 physics DR 새 범위를 전 env 에 즉시 반영 (원본 동일)
        if self.grasp_adr is not None:
            if self.grasp_adr.maybe_increment(self.in_success_region.float().mean()):
                _em = getattr(self, "event_manager", None)
                if _em is not None:
                    _em.reset(env_ids=self.robot._ALL_INDICES)
                    _em.apply(
                        env_ids=self.robot._ALL_INDICES,
                        mode="reset",
                        global_env_step_count=0,
                    )

        # ── 로깅: DEXTRAH-g 정렬 ──────────────────────────────────────────
        # DEXTRAH 원본 extras: 4 reward 항 + in_success_region (동일 이름)
        self.extras["reward/hand_to_object"] = hand_to_object_reward.mean()
        self.extras["reward/object_to_goal"] = object_to_goal_reward.mean()
        self.extras["reward/finger_curl_reg"] = finger_curl_reg.mean()
        self.extras["reward/lift"] = lift_reward.mean()
        self.extras["in_success_region"] = self.in_success_region.float().mean()
        # 접근 자세 분기 검증용: top-down 으로 배정된 env 비율 (ADR 회전이 커지면 상승, 07.13)
        self.extras["topdown_frac"] = (self.palm_pose_id == 1).float().mean()
        # DEXTRAH 원본 extras: ADR 진행도
        if self.grasp_adr is not None:
            self.extras["num_adr_increases"] = torch.tensor(
                float(self.grasp_adr.increment_counter), device=self.device
            )
        # 해석 보조: 안착점(settle 스냅샷) 대비 실제 리프트 높이 → 렌더와 일치
        self.extras["object_height"] = (
            self.object_pos[:, 2] - self.object_init_pos[:, 2]
        ).mean()
        # contact: 감싸기 노드 그룹별 "물체" 접촉 손가락 수 (0~5, 거리 게이트 정화).
        # contact/raw_grip = 무게이트(테이블 포함) — 구 지표와의 연속성·오염량 관측용.
        self.extras["contact/tip"]    = _tip_obj.float().sum(dim=-1).mean()
        self.extras["contact/middle"] = _mid_obj.float().sum(dim=-1).mean()
        self.extras["contact/distal"] = _distal_obj.float().sum(dim=-1).mean()
        self.extras["contact/grip"]   = num_grip_fingers.float().mean()
        # grasp_contact 보상 진단: 굴곡 유인이 실제 접촉·지속으로 이어지는지
        self.extras["reward/grasp_contact"] = grasp_contact_reward.mean()
        self.extras["contact/tip_frac"]     = tip_frac.mean()
        self.extras["contact/persist_frac"] = persist_frac.mean()
        # force_closure 진단: 엄지-4지 마주조임 학습 여부
        self.extras["reward/force_closure"] = force_closure_reward.mean()
        self.extras["contact/opposition"] = (
            (opposition * fc_gate).sum() / fc_gate.sum().clamp(min=1.0)
        )  # 게이트 통과(양쪽 접촉) env 평균 대향도
        self.extras["contact/fc_gate_frac"] = fc_gate.mean()
        self.extras["contact/envelope_frac"] = graded_contact.mean()  # lift 게이트(감쌈) 값
        # 손가락별 tip 접촉률 (엄지 포함 개별 관측 — tip 센서 순서 thumb_4/index_2/…/pinky_2)
        for _i, _fn in enumerate(("thumb", "index", "middle", "ring", "pinky")):
            self.extras[f"contact/tip_{_fn}"] = self.binary_contact_buf[:, _i].float().mean()
        self.extras["contact/raw_grip"] = (
            self.binary_contact_buf
            | self.middle_binary_contact_buf
            | self.distal_binary_contact_buf
        ).sum(dim=-1).float().mean()
        # action policy(palm 6D + finger 5D raw) 로깅 유지
        for k, v in action_policy_scalars(
            action=self.actions, prev_action=self.prev_actions, palm_dims=6,
        ).items():
            self.extras[k] = v
        # episode_success_rate: 물체별 성공률 + 전체 (누적 집계는 _reset_idx 에서 갱신)
        # 물체별 카운트는 리셋에서만 변하고 TB 는 epoch(=horizon 16 step)에만 쓰므로 매 step
        # 148종 CPU 전송(.tolist() 2 sync)은 낭비 → per_object_log_interval step마다만 갱신.
        # extras dict 은 초기화 안 되므로 값이 유지됨(DEXTRAH: 매 step 집계만, 동기 0). fps
        # sawtooth 의 CPU 구간 제거. 집계 스칼라는 python int 라 sync 없어 매 step 유지.
        if self.common_step_counter % self.cfg.per_object_log_interval == 0:
            _tot_list = self._obj_total_episodes.tolist()
            _suc_list = self._obj_success_episodes.tolist()
            for _i, _name in enumerate(self._object_names):
                _tot = _tot_list[_i]
                if _tot > 0:
                    self.extras[f"episode_success_rate/{_name}"] = torch.tensor(
                        _suc_list[_i] / _tot, device=self.device
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

        # GRASP_DEBUG_DONES=1 (로컬 진단 전용, 07.13 episode-length plateau 원인 분류)
        if _os.environ.get("GRASP_DEBUG_DONES"):
            if not hasattr(self, "_dbg_done_cnt"):
                self._dbg_done_cnt = {"out_x": 0, "out_y": 0, "fallen": 0, "diverged": 0}
            for _k, _m in [("out_x", out_x), ("out_y", out_y), ("fallen", fallen), ("diverged", robot_diverged)]:
                self._dbg_done_cnt[_k] += int(_m.sum().item())

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
            # demo warmstart 는 side-approach 자세로 수집된 데이터 → 분기 대상 아님.
            spawn_rot = self._sample_spawn_rotation(n)
            self.palm_pose_id[env_ids] = 0
            self.palm_mins_env[env_ids] = self.palm_mins_by_pose[0]
            self.palm_maxs_env[env_ids] = self.palm_maxs_by_pose[0]
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
            # 컵 spawn(7단계)이 재사용 — 물리 스폰 회전은 접근 자세와 무관하게 매번 샘플.
            spawn_rot = self._sample_spawn_rotation(n)

            # ---- 접근 자세 결정: 물체 이름 기반 고정 분기 (07.13, tesollo cd29c62 이식) ----
            # 물체 높이 회전 규칙은 ADR 회전이 커지면 스스로 꺼지는 자기모순이 있어 폐기.
            if self.cfg.approach_branch_enable:
                pose_id = self._compute_palm_pose_id(self.object_idx[env_ids])  # (n,) 0=side, 1=top
            else:
                pose_id = torch.zeros(n, dtype=torch.long, device=self.device)
            self.palm_pose_id[env_ids] = pose_id
            self.palm_mins_env[env_ids] = self.palm_mins_by_pose[pose_id]
            self.palm_maxs_env[env_ids] = self.palm_maxs_by_pose[pose_id]

            # ---- FABRICS pregrasp rollout (clearance 기반 offset, IK 캐시 없음) ----
            noise = torch.stack([
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_x,
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_y,
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_z,
            ], dim=1)
            pregrasp_pos = obj_pos_local + self._compute_pregrasp_offset(
                self.object_idx[env_ids], pose_id
            ) + noise

            _ex = torch.where(
                pose_id == 1,
                torch.full((n,), math.radians(PREGRASP_EULER_EX_TOPDOWN_DEG), device=self.device),
                torch.full((n,), math.radians(PREGRASP_EULER_EX_DEG), device=self.device),
            )
            pregrasp_palm_pose = torch.zeros(n, 6, device=self.device)
            pregrasp_palm_pose[:, :3] = pregrasp_pos
            # RH56F1 side grasp: palm_sensor +z(법선)가 물체(-y 접근 → +y)를 향하도록
            # (ez,ey,ex)=(180,0,90). top-down(ex=180)은 법선이 -z(아래보기, 07.13 접근
            # 자세 분기, tesollo cd29c62 이식).
            pregrasp_palm_pose[:, 3] = math.radians(PREGRASP_EULER_EZ_DEG)
            pregrasp_palm_pose[:, 4] = math.radians(0.0)
            pregrasp_palm_pose[:, 5] = _ex
            pregrasp_palm_pose = torch.max(
                torch.min(pregrasp_palm_pose, self.palm_maxs_env[env_ids]),
                self.palm_mins_env[env_ids],
            )

            # IK 캐시 제거(tesollo 9f0e4f7 이식): offset 이 물체별 clearance 로 연속값이
            # 되어 (pose,x,y) grid 캐시 전제가 깨진다. reset 마다 fabrics rollout.
            q_pregrasp = self._run_reset_fabric(env_ids, pregrasp_palm_pose, q_pregrasp)

            # hand는 APPROACH_POSE로 강제
            q_pregrasp[:, NUM_ARM_DOF:] = approach_hand

            # r_aj_7(손목, arm index 6)을 낮춰 palm을 물체 높이로 내림. fabric은 +y 수평
            # 유지 위해 r_aj_7을 높게 잡아 palm이 물체 rim에 뜨므로(probe 확정) bias로
            # 끌어내린다. bias 후 실제 palm(FK)로 anchor를 정합해 정책 시작 시 palm 튐 방지.
            # 접근 자세별 bias(side 0.3 / top-down 0.6, 07.13 이식) — per-env 적용.
            aj7_bias = self.pregrasp_r_aj7_bias_by_pose[pose_id]
            q_pregrasp[:, 6] = q_pregrasp[:, 6] - aj7_bias
            fq_fk = self.fabric.default_config.clone()
            fq_fk[env_ids, :NUM_ROBOT_DOF] = q_pregrasp
            pregrasp_palm_pose = self.fabric.get_palm_pose(fq_fk, "euler_zyx")[env_ids]

        # ---- 1.5 로봇 초기상태 노이즈 (DEXTRAH robot_spawn — ADR 0→0.35 rad / 0→1 rad/s) ----
        _sp_pos = self._adr("robot_spawn", "joint_pos_noise")
        _sp_vel = self._adr("robot_spawn", "joint_vel_noise")
        if _sp_pos > 0.0:
            q_pregrasp = q_pregrasp + _sp_pos * 2.0 * (
                torch.rand_like(q_pregrasp) - 0.5
            )
            _lims = self.robot.data.soft_joint_pos_limits[0, self.actuated_dof_indices]
            q_pregrasp = torch.clamp(q_pregrasp, min=_lims[:, 0], max=_lims[:, 1])

        # ---- 2. 로봇/Fabrics 상태 리셋 ----
        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_pos[:, self.actuated_dof_indices] = q_pregrasp
        full_pos[:, self.left_arm_dof_indices] = self.left_arm_zero_pos[0]
        if _sp_vel > 0.0:
            full_vel[:, self.actuated_dof_indices] = _sp_vel * 2.0 * (
                torch.rand(n, len(self.actuated_dof_indices), device=self.device) - 0.5
            )
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
        # spawn_rot 은 위(1. Reset source)에서 이미 뽑았다 — 접근 자세 분기가 "회전 후
        # 물체 높이"를 필요로 하므로 pregrasp IK 보다 먼저 결정돼야 한다(07.13).
        obj_pos_world = obj_pos_local + self.scene.env_origins[env_ids]
        zero_vel = torch.zeros(n, 6, device=self.device)
        cup_root_state = torch.cat([obj_pos_world, spawn_rot, zero_vel], dim=-1)
        self.cup.write_root_state_to_sim(cup_root_state, env_ids=env_ids)

        # ---- 8. 버퍼 리셋 ----
        self.hand_joint_targets[env_ids] = approach_hand
        self.contact_force_raw[env_ids].zero_()
        self.binary_contact_buf[env_ids] = False
        self.num_contacts_buf[env_ids]   = 0
        self.contact_persist_buf[env_ids] = 0
        self.warm_contact_stable_steps_buf[env_ids] = 0
        self.lift_wait_match_hold_steps_buf[env_ids] = 0
        self.distal_contact_force_raw[env_ids].zero_()
        self.distal_binary_contact_buf[env_ids] = False
        self.middle_contact_force_raw[env_ids].zero_()
        self.middle_binary_contact_buf[env_ids] = False
        self.success_flag[env_ids] = False
        self.transfer_entry_grasp_success_buf[env_ids] = False

        # actions 리셋: palm 은 절대 pose 라 action=0 → 박스 중심(pregrasp 근방). 손가락은
        # action=-1 → lower limit(열림). settle 동안 approach 강제되므로 초기값은 obs 용.
        self.actions[env_ids, :6] = 0.0
        self.actions[env_ids, 6:] = -1.0
        self.prev_actions[env_ids, :6] = 0.0
        self.prev_actions[env_ids, 6:] = -1.0

        # DEXTRAH 정렬: reset 직후 obs가 noisy 중간값을 요구 (첫 reset 포함)
        self._compute_intermediate_values()
