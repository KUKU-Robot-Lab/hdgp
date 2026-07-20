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
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera
from openarm.distillation.visual_dr import VisualDomainRandomizer
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_mul

from openarm.common.grasp_logging import action_policy_scalars

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp, capture_fabric
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

from .grasp_right_env_cfg import GraspRightEnvCfg
from .grasp_adr import GraspADR
from .grasp_right_constants import (
    NUM_ARM_DOF,
    NUM_HAND_DOF,
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
    HAND_APPROACH_POSE,
    HAND_GRASP_POSE,
    HAND_FULL_GRIP_POSE,
    PALM_GRASP_FRAME_ROT,
    PREGRASP_G_EULER_TOPDOWN,
    PREGRASP_G_EULER_SIDE,
    PALM_G_EULER_CENTER_TOPDOWN,
    PALM_G_EULER_CENTER_SIDE,
    PREGRASP_TOPDOWN_XY,
    PREGRASP_TOPDOWN_FINGER_REACH,
    TABLE_TOP_Z,
    PREGRASP_SIDE_Z,
    PREGRASP_SIDE_CLEARANCE,
    PREGRASP_OFFSET,
    HAND_ABDUCTION_LOCAL_INDICES,
    HAND_ABDUCTION_LIMITS_MIN,
    HAND_ABDUCTION_LIMITS_MAX,
    FABRIC_WORLD_FILENAME,
)
from .finger_action_utils import (
    compute_grasp_finger_targets,
    compute_lift_finger_targets,
    compute_per_finger_progress_targets,
    compute_synergy_progress_targets,
)
from .tesollo_hand_synergy import (
    HAND_SYNERGY_BASIS,
    HAND_SYNERGY_ANCHOR,
    HAND_SYNERGY_COEFF_MINS,
    HAND_SYNERGY_COEFF_MAXS,
)
from .grasp_right_utils import (
    compute_abduction_targets,
    compute_joint7_lift_wait_target,
    compute_palm_pose_id,
    compute_rotated_half_z,
    g_pose_to_fabric_quat,
    g_pose_to_fabric_matrix,
    kept_object_names_and_indices,
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

        # body indices (robot.data.body_pos_w 참조용). 통일 네이밍: r_hl_<finger>_*
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
        # DEXTRAH hand bodies (palm + 5 tips) — critic hand_vel/forces 용
        self._hand_body_indices: list[int] = [self.palm_body_index] + self.fingertip_body_indices
        # distal phalanx body indices (r_hl_<finger>_4) — R2 reward용
        _distal4_names = [f"r_hl_{fn}_4" for fn in _FINGERS]
        self.distal4_body_indices: list[int] = [
            self.robot.data.body_names.index(name)
            for name in _distal4_names
            if name in self.robot.data.body_names
        ]

        # ----------------------------------------------------------------
        # Palm pose 절대 workspace (안전 한계 클램프용) — G 규약
        #
        # 회전은 grasp 프레임(G) 에서 명령한다. tesollo palm 로컬축(+X=법선,+Z=손가락)이
        # Allegro/DEXTRAH 규약(+X=손가락,±Z=법선)과 90° 어긋나 있어, P 규약 euler 로는
        # ey≈0 에서 손바닥을 아래로 돌릴 수 없다(lstm_test3 "가짜 top-down" 실증).
        # G 규약에서는 (ez, ey=0, ex=180) 이 ez 와 무관하게 법선을 -Z 로 보낸다
        #   → ez = 손가락 방위각, ey = 법선 기울임 → 기존 ±45° 경계가 그대로 유효.
        # ----------------------------------------------------------------
        self.palm_grasp_frame_rot = to_torch(PALM_GRASP_FRAME_ROT, device=self.device)  # (3,3) C

        # (2, 6): [0]=side(cup), [1]=top-down. 위치 3축 공통, 회전 중심만 다르다.
        self.palm_mins_by_pose = torch.stack([
            to_torch(PALM_POSE_MINS_FUNC(cfg.max_pose_angle, PALM_G_EULER_CENTER_SIDE), device=self.device),
            to_torch(PALM_POSE_MINS_FUNC(cfg.max_pose_angle, PALM_G_EULER_CENTER_TOPDOWN), device=self.device),
        ])
        self.palm_maxs_by_pose = torch.stack([
            to_torch(PALM_POSE_MAXS_FUNC(cfg.max_pose_angle, PALM_G_EULER_CENTER_SIDE), device=self.device),
            to_torch(PALM_POSE_MAXS_FUNC(cfg.max_pose_angle, PALM_G_EULER_CENTER_TOPDOWN), device=self.device),
        ])
        self.palm_mins = self.palm_mins_by_pose[1]
        self.palm_maxs = self.palm_maxs_by_pose[1]

        # per-env 경계 버퍼 (reset 에서 물체 이름에 따라 채움). 기본 = top-down.
        self.palm_pose_id  = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.palm_mins_env = self.palm_mins_by_pose[1].unsqueeze(0).repeat(self.num_envs, 1)
        self.palm_maxs_env = self.palm_maxs_by_pose[1].unsqueeze(0).repeat(self.num_envs, 1)

        # ----------------------------------------------------------------
        # Delta palm action 범위 (pregrasp 기준 상대 오프셋)
        # action=0 → pregrasp 위치 유지, action=±1 → ±delta 이동
        # scale(0) = pregrasp 이므로 초기 정책(출력≈0) = 안정된 pregrasp 위치
        # ----------------------------------------------------------------
        # palm 목표 rate limit (스텝당 최대 변화량 — 접근 밀침·스윙 대책, 기구적 제약)
        _rate_rad = math.radians(cfg.palm_rate_rot_deg_per_step)
        self.palm_rate_limits = to_torch([
            cfg.palm_rate_xyz_per_step, cfg.palm_rate_xyz_per_step, cfg.palm_rate_xyz_per_step,
            _rate_rad, _rate_rad, _rate_rad,
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
        # 시너지(eigengrasp) basis — Allegro PCA 리타겟 (tesollo_hand_synergy.py)
        self.hand_synergy_basis = to_torch(HAND_SYNERGY_BASIS, device=self.device)       # (5,20)
        self.hand_synergy_anchor = to_torch(HAND_SYNERGY_ANCHOR, device=self.device)     # (20,)
        self.hand_synergy_mins = to_torch(HAND_SYNERGY_COEFF_MINS, device=self.device)   # (5,)
        self.hand_synergy_maxs = to_torch(HAND_SYNERGY_COEFF_MAXS, device=self.device)   # (5,)
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

        # Pregrasp offset 은 reset 에서 물체 clearance 로 계산한다(_compute_pregrasp_offset).
        # demo warmstart 경로는 side 로 수집된 고정 자세라 cfg 값을 그대로 쓴다.
        self.pregrasp_offset = to_torch(
            [cfg.pregrasp_offset_x, cfg.pregrasp_offset_y, cfg.pregrasp_offset_z],
            device=self.device,
        )
        self._side_y_sign = -1.0 if PREGRASP_OFFSET[1] < 0 else 1.0

        # ----------------------------------------------------------------
        # 자유화된 abduction 관절 (action[11:15] → 절대 목표)
        # ----------------------------------------------------------------
        self.abduction_local_indices = to_torch(
            HAND_ABDUCTION_LOCAL_INDICES, dtype=torch.long, device=self.device
        )
        self.abduction_limits_min = to_torch(HAND_ABDUCTION_LIMITS_MIN, device=self.device)
        self.abduction_limits_max = to_torch(HAND_ABDUCTION_LIMITS_MAX, device=self.device)
        # 중립값 = HAND_APPROACH_POSE 의 해당 관절 (thumb_2 = -1.57 등).
        # ADR range_scale=0 일 때 이 값에 고정된다 — probe 실증상 파지에 최적인 자세다.
        self.abduction_neutral = self.hand_open_pose[self.abduction_local_indices].clone()
        # 목표 버퍼: reset 시 중립 → rate limit 으로 부드럽게 추종.
        # 자기충돌 검사가 꺼져 있어 순간이동식 abduction 은 인접 손가락을 관통한다.
        self.abduction_targets = self.abduction_neutral.unsqueeze(0).repeat(
            self.num_envs, 1
        ).contiguous()

        # ----------------------------------------------------------------
        # 중간값 버퍼
        # ----------------------------------------------------------------
        self.object_pos      = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_rot      = torch.zeros(self.num_envs, 4, device=self.device)
        self.object_init_pos = torch.zeros(self.num_envs, 3, device=self.device)
        # [FP 배포 검증] eval_pose_hold: settle 시점까지 추종 후 고정되는 pose(=FP lock)
        self.object_pos_held = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_rot_held = torch.zeros(self.num_envs, 4, device=self.device)
        # per-object 로깅: MultiAsset(random_choice=False)는 env_id % N 로 물체 배정.
        self._object_names = list(self.cfg.active_object_names)
        # distillation 실패물체 제외: onehot 은 153 유지, 배정만 kept 로 하되 object_idx 는
        # 원본 슬롯 인덱스로 remap(스포너 kept 순서와 env_id % len(kept) 를 공유해 일치).
        _excluded = getattr(self.cfg, "distill_excluded_object_names", ())
        if getattr(self.cfg, "distillation", False) and _excluded:
            _kept, _orig = kept_object_names_and_indices(self._object_names, _excluded)
            _orig_t = to_torch(_orig, dtype=torch.long, device=self.device)
            self.object_idx = _orig_t[
                torch.arange(self.num_envs, device=self.device) % len(_kept)
            ]
        else:
            self.object_idx = (
                torch.arange(self.num_envs, device=self.device) % len(self._object_names)
            )
        # side 접근을 유지할 물체의 인덱스 (cup). 그 외 전부 top-down.
        _side = [
            self._object_names.index(_n)
            for _n in self.cfg.side_approach_object_names
            if _n in self._object_names
        ]
        self.side_object_idx = to_torch(_side, dtype=torch.long, device=self.device)
        # 물체별 clearance = ‖half_extent‖ (임의 회전 시 중심→표면 최대거리).
        # pregrasp 를 이 값에 비례시켜 스폰 겹침(→ depenetration 폭주)을 없앤다.
        self.object_clearance = to_torch(
            self._load_object_clearances(), device=self.device
        )
        # 물체별 half-extent (n_obj, 3) — top-down palm 높이를 "회전 후 실제 물체
        # 높이(half_z)" 로 잡는 데 쓴다. clearance(대각선)는 직립 물체에서 물체 top 을
        # 최대 3.2cm 과대평가해 손가락이 닿지 않았다.
        self.object_half_extent = to_torch(
            self._load_object_half_extents(), device=self.device
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
        # Distillation (Dagger 가 참조하는 계약. teacher 학습에선 use_camera=False)
        #   num_observations   = 학습 중인 정책의 obs 차원 (student ↔ teacher)
        #   num_teacher_observations = 항상 teacher 차원 (expert_policy)
        # ----------------------------------------------------------------
        self.num_actions = cfg.num_actions
        self.num_observations = (
            cfg.num_student_observations if cfg.distillation
            else cfg.num_teacher_observations
        )
        self.num_teacher_observations = cfg.num_teacher_observations
        self.use_camera = cfg.distillation
        # 시각 도메인 랜덤화 — student 인코더가 RGB 를 보므로 외형이 고정되면
        # 단일 장면에만 맞는 정책이 나온다. shader prim 은 씬 clone 이후에 존재하므로
        # 여기(super().__init__ 완료 후)에서 만든다.
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
        # grasp_contact persistence: 연속 접촉(임의 손끝) step 수 (reset 시 0)
        self.contact_persist_buf   = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.warm_contact_stable_steps_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.lift_wait_match_hold_steps_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self.distal_contact_force_raw  = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, device=self.device)
        self.distal_contact_force_xyz_raw = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, 3, device=self.device)
        self.distal_binary_contact_buf = torch.zeros(self.num_envs, NUM_DISTAL_SENSORS, dtype=torch.bool, device=self.device)

        self.middle_contact_force_raw  = torch.zeros(self.num_envs, NUM_MIDDLE_SENSORS, device=self.device)
        self.middle_contact_force_xyz_raw = torch.zeros(self.num_envs, NUM_MIDDLE_SENSORS, 3, device=self.device)
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
            # DEXTRAH starting_adr_increments (env_cfg.py:339) 이식: distillation 은
            # env 를 teacher 작동점(만렙)에 고정해야 한다 — ADR 0 이면 abduction 이
            # 잠겨 teacher 시연이 왜곡된다(dt1 고원 0.19 의 구조 원인). 기본 0 = 기존 동작.
            _start_adr = int(getattr(cfg, "starting_adr_increments", 0))
            if _start_adr > 0:
                self.grasp_adr.set_increment(_start_adr)
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

        # Actor: 손가락별 개별 ContactSensor + **물체 전용 필터**.
        #
        # 이전 주석은 "MultiAsset(replicate_physics=False)에서 filter_prim_paths_expr는
        # GPU 미지원 → contact 0" 이라며 필터를 지우고 net_forces_w 를 읽었다. **오진이었다.**
        # probe_contact_filter 실측 — 필터를 걸면 force_matrix_w 가 (N,1,1,3) 으로 정상
        # 출력된다. 필터가 적용되는 것은 force_matrix_w 뿐이고 net_forces_w 는 필터와
        # 무관하게 **모든** 접촉을 합산한다(IsaacLab 문서). 즉 필터를 걸어놓고 net_forces_w
        # 를 읽었거나(당시), 필터를 아예 지운 채(현행) 물체·테이블을 구분 못 하고 있었다.
        #
        # 그 대가로 "contact/grip 3.2 인데 object_height 0" 같은 모순된 로그가 나왔다 —
        # 손가락이 물체가 아니라 테이블을 짚고 있었던 것이다.
        _CUP_FILTER = [f"/World/envs/env_.*/Cup/{self.cfg.cup_rigid_body_name}"]

        def _mk_sensor(prim: str) -> ContactSensor:
            return ContactSensor(ContactSensorCfg(
                prim_path=prim,
                filter_prim_paths_expr=_CUP_FILTER,
                history_length=1,
                track_air_time=False,
            ))

        self._tip_sensors: list[ContactSensor] = []
        for link_name in self.cfg.right_tip_contact_links:
            sensor = _mk_sensor(f"/World/envs/env_.*/Robot/{link_name}")
            self._tip_sensors.append(sensor)
            self.scene.sensors[f"tip_sensor_{link_name}"] = sensor

        _SENSOR_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
        self._distal_sensors: list[ContactSensor] = []
        for i, fn in enumerate(_SENSOR_FINGERS):
            sensor = _mk_sensor(f"/World/envs/env_.*/Robot/r_hl_{fn}_4")
            self._distal_sensors.append(sensor)
            self.scene.sensors[f"distal_sensor_{i + 1}"] = sensor

        self._middle_sensors: list[ContactSensor] = []
        for i, fn in enumerate(_SENSOR_FINGERS):
            sensor = _mk_sensor(f"/World/envs/env_.*/Robot/r_hl_{fn}_3")
            self._middle_sensors.append(sensor)
            self.scene.sensors[f"middle_sensor_{i + 1}"] = sensor

        # Distillation: D435i RGB-D. teacher 학습(distillation=False)에선 생성하지 않는다
        # — TiledCamera 는 env 당 렌더 타깃을 잡아 4096 env teacher 학습을 못 돌린다.
        # (self.use_camera 는 아직 없다 — _setup_scene 은 super().__init__ 안에서 돈다)
        if self.cfg.distillation or getattr(self.cfg, "enable_camera_probe", False):
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera_cfg)
            self.scene.sensors["tiled_camera"] = self._tiled_camera

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # ★다물체 스폰 순서 (07.12 버그 수정, rh56f1 d674952와 동일): clone → cup(MultiAsset) 생성.
        # RigidObject(cup_cfg)는 생성 시점에 즉시 spawn하는데(asset_base.py — leaf "Cup"은
        # non-regex), clone 이전엔 env_0만 존재해 MultiAssetSpawner가 물체[0] 하나만 spawn
        # → clone(copy_from_source=True)이 그걸 전 env에 복제 = 전 env 동일 물체 버그
        # (probe 실측 /tmp/probe_tesol2.log: 16env 전부 visdex[0]='104738', unique=1).
        # clone을 먼저 하면 spawn 시점에 env prim 전부 존재 → env_i % N 결정적 배정 정상화.
        self.scene.clone_environments(copy_from_source=True)
        self.cup = RigidObject(self.cfg.cup_cfg)
        self.scene.rigid_objects["cup"] = self.cup

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
            world_filename=FABRIC_WORLD_FILENAME,
        )
        self.object_ids, self.object_indicator = self.world_model.get_object_ids()

        self.timestep = self.cfg.fabrics_dt

        # Main fabric (arm 제어용). use_cuda_graph=True 면 fabric IK 를 CUDA Graph 로
        # 캡처(capture_fabric)해 커널 런치 오버헤드를 없앤다 → graph_capturable 필수.
        self.open_tesollo_fabric = OpenArmTeoslloPoseFabric(
            self.num_envs, self.device, self.timestep,
            graph_capturable=bool(self.cfg.use_cuda_graph),
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
        # palm matrix 고정 버퍼: set_features("matrix") 에 넘길 (B,12)[x,y,z,R_flat9] 를
        # 매 step in-place 로 갱신한다. CUDA Graph 캡처는 입력 텐서 주소 고정 필요(새 텐서
        # 생성 금지) + quaternion 경로의 fancy indexing 회피를 위해 matrix 모드를 쓴다.
        self.palm_matrix_buf = g_pose_to_fabric_matrix(
            self.palm_pose_targets, self.palm_grasp_frame_rot
        ).contiguous()

        # CUDA Graph 캡처(use_cuda_graph=True): fabric set_features + integrator.step 을
        # 한 번 캡처해두고 매 step replay → 다물체(replicate_physics=False)로 떨어진 GPU
        # 활용을 fabric IK 쪽에서 만회(DEXTRAH env.py 동일 기법). 캡처 입력은 전부 고정 버퍼.
        if bool(self.cfg.use_cuda_graph):
            self._fabric_inputs = [
                self.hand_pca_targets, self.palm_matrix_buf, "matrix",
                self.fabric_q.detach(), self.fabric_qd.detach(),
                self.object_ids, self.object_indicator, self.fabric_damping_gain,
            ]
            self.g, self.fabric_q_new, self.fabric_qd_new, self.fabric_qdd_new = capture_fabric(
                self.open_tesollo_fabric,
                self.fabric_q, self.fabric_qd, self.fabric_qdd,
                self.timestep, self.open_tesollo_integrator,
                self._fabric_inputs, self.device,
            )

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
            world_filename=FABRIC_WORLD_FILENAME,
        )
        self._reset_obj_ids, self._reset_obj_indicator = self._reset_world.get_object_ids()

    # ------------------------------------------------------------------
    # 접근 자세 분기
    # ------------------------------------------------------------------
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

    def _compute_palm_pose_id(self, obj_idx: torch.Tensor) -> torch.Tensor:
        """물체 이름 기반 접근 자세: cup → 0(side), 그 외 → 1(top-down)."""
        return compute_palm_pose_id(obj_idx, self.side_object_idx)

    def _load_object_clearances(self) -> list[float]:
        """물체별 clearance = ‖half_extent‖ (m). 누락 물체는 즉시 실패시킨다.

        조용한 fallback(0 채우기)은 pregrasp 를 물체 안에 박아 넣으므로 금지.
        """
        path = Path(self.cfg.object_bbox_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"물체 bbox 파일 없음: {path}\n"
                "  python3 scripts/tools/compute_object_bbox.py 로 먼저 생성하세요."
            )
        table = json.loads(path.read_text(encoding="utf-8"))
        missing = [nm for nm in self._object_names if nm not in table]
        if missing:
            raise KeyError(f"bbox 누락 물체 {len(missing)}종: {missing[:5]} … — bbox 재생성 필요")
        return [
            float(sum(float(v) ** 2 for v in table[nm]) ** 0.5)
            for nm in self._object_names
        ]

    def _load_object_half_extents(self) -> list[list[float]]:
        """active_object_names 순서의 half-extent (m). 누락 시 즉시 실패."""
        path = Path(self.cfg.object_bbox_path)
        table = json.loads(path.read_text(encoding="utf-8"))
        missing = [nm for nm in self._object_names if nm not in table]
        if missing:
            raise KeyError(f"bbox 누락 물체 {len(missing)}종: {missing[:5]} …")
        return [[float(v) for v in table[nm]] for nm in self._object_names]

    def _compute_pregrasp_offset(
        self, obj_idx: torch.Tensor, pose_id: torch.Tensor, spawn_rot: torch.Tensor
    ) -> torch.Tensor:
        """물체 크기 비례 pregrasp offset (n,3).

        top-down: palm z = (회전 후 물체 높이 half_z) + FINGER_REACH.
          clearance(‖half_extent‖ = 대각선)를 쓰면 직립 물체에서 물체 top 을 최대 3.2cm
          과대평가해 palm 이 너무 높이 뜬다 → 손가락(~10cm)을 굽혀도 물체에 닿지 않는다
          (실측: contact/tip 0.00~0.09, object_height 음수 — 한 번도 못 잡음).
          회전 후 half_z 를 쓰면 직립일 때 palm 이 물체 top 바로 위에 오고, ADR 로 물체가
          누우면 half_z 가 커져 palm 도 자동으로 올라간다(겹침 방지 유지).

        side(cup): 옆에서 감싸므로 회전 무관 최대 반경(clearance)을 그대로 쓴다.

        고정 offset 은 회전 ADR 에서 물체가 palm 을 침범해 PhysX depenetration 폭주를
        일으킨다(lstm_test1: ADR 36 부터 스파이크, 14111 에서 -4.9e7 붕괴).
        """
        clr = self.object_clearance[obj_idx]                    # (n,)  회전 무관 최대 반경
        half_z = compute_rotated_half_z(
            self.object_half_extent[obj_idx], matrix_from_quat(spawn_rot)
        )                                                       # (n,)  회전 후 실제 높이
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
            half_z + float(PREGRASP_TOPDOWN_FINGER_REACH),
            torch.full_like(clr, float(PREGRASP_SIDE_Z)),
        )
        return off

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

            # G 규약 euler → fabric quaternion (특이점 없음)
            self._reset_fabric.set_features(
                self._reset_pca,
                g_pose_to_fabric_quat(pp, self.palm_grasp_frame_rot),
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
        # **물체 접촉만** 센다 — force_matrix_w 는 filter_prim_paths_expr(Cup) 에 대한
        # 접촉력이다. net_forces_w 는 필터와 무관하게 테이블·자기 손까지 전부 합산하므로
        # 쓰지 않는다(IsaacLab 문서). 필터 1개 → shape (N, 1, 1, 3).
        def _obj_force(sensors: list[ContactSensor]) -> torch.Tensor:
            return torch.stack([
                s.data.force_matrix_w[:, 0, 0, :] for s in sensors
            ], dim=1)                                        # (N, 5, 3)

        tip_xyz = _obj_force(self._tip_sensors)
        tip_norms = tip_xyz.norm(dim=-1)   # (N, 5)

        self.contact_force_xyz_raw.copy_(tip_xyz)
        self.contact_force_raw.copy_(tip_norms)
        self.binary_contact_buf.copy_(tip_norms > CONTACT_FORCE_THRESHOLD)
        self.num_contacts_buf.copy_(self.binary_contact_buf.sum(dim=-1).long())

        distal_xyz = _obj_force(self._distal_sensors)                # (N, 5, 3)
        per_distal = distal_xyz.norm(dim=-1)                         # (N, 5)
        self.distal_contact_force_xyz_raw.copy_(distal_xyz)
        self.distal_contact_force_raw.copy_(per_distal)
        self.distal_binary_contact_buf.copy_(per_distal > CONTACT_FORCE_THRESHOLD)

        middle_xyz = _obj_force(self._middle_sensors)                # (N, 5, 3)
        per_middle = middle_xyz.norm(dim=-1)                         # (N, 5)
        self.middle_contact_force_xyz_raw.copy_(middle_xyz)
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
        # 게이트(위 apply) 밖의 env 는 wrench 0 — 손이 물체 반경 밖이면 인가하지 않는다.
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

        palm_action      = actions[:, :6]     # (N, 6) ∈ [-1, 1]
        finger_action    = actions[:, 6:11]   # (N, 5) ∈ [-1, 1]  시너지 계수
        abduction_action = actions[:, 11:16]  # (N, 5) ∈ [-1,1] thumb_1/thumb_2/index_1/pinky_1/_2

        # ---- settle 종료 시 안착 스냅샷 → object_height 로깅 baseline 확정 ----
        # (goal 은 DEXTRAH식 고정 절대점이라 갱신 없음)
        snap = self.episode_length_buf == int(self.cfg.settle_steps)
        if snap.any():
            self.object_init_pos[snap] = self.object_pos[snap]

        # [FP 배포 검증] settle 까지는 live pose 추종(물체 낙하 안착 중), 이후 고정.
        # 정적 물체라 settle 후 held == 실제 pose(grasp 로 들어올리기 전까지). = FP lock.
        if bool(self.cfg.eval_pose_hold):
            _track = (self.episode_length_buf <= int(self.cfg.settle_steps))
            self.object_pos_held[_track] = self.object_pos[_track]
            self.object_rot_held[_track] = self.object_rot[_track]

        # ---- Fabrics arm 제어: palm 절대 pose (DEXTRAH 원본 구조) ----
        # action[0:6] ∈ [-1,1] 을 palm workspace 박스로 직접 스케일한다.
        # 즉 정책 출력이 곧 "손바닥을 놓을 절대 위치/자세"다.
        #
        # 이전의 delta 방식(pregrasp ± 0.15m)은 물체까지 20~30cm 를 rate limit
        # (0.04 m/step) 으로 수백 스텝 적분해야 해서 credit assignment 가 무너졌다:
        # hand_to_object 가 오르다가 정책이 "가만히 있기"로 수렴하며 급락했다
        # (curl_fix run: ep200 0.216 → ep400 0.017). DEXTRAH 는 절대 pose 라
        # "물체 위로 가라"가 1스텝 결정이다.
        #
        # 경계는 per-env (side/top-down 회전 중심이 다르다). scale 결과가 이미
        # 박스 안이므로 별도 clamp 불필요.
        # action=0 → 박스 중심. 회전 중심은 G euler pregrasp 자세와 동일하므로
        # 초기 정책(출력≈0)은 올바른 접근 자세에서 시작한다.
        palm_pose = scale(palm_action, self.palm_mins_env, self.palm_maxs_env)  # (N, 6)
        # settle 동안 팔 동결: 물체 낙하-안착 전 정책이 팔로 쫓아가 물체를 쳐내
        # 도달불가 위치로 밀어버리는 것 방지 (finger 억제와 동일 게이트 — 렌더 실증)
        in_settle = (
            self.episode_length_buf < int(self.cfg.settle_steps)
        ).unsqueeze(-1)
        palm_pose = torch.where(
            in_settle, self.pregrasp_palm_pose_buf, palm_pose
        )
        # rate limit: 목표가 이전 목표에서 스텝당 palm_rate_limits 이상 못 움직임
        # (정책의 bang-bang 목표 순간이동 → 접근 밀침·리프트 후 스윙의 기구적 차단.
        #  reset 시 palm_pose_targets=pregrasp 로 앵커됨)
        _step6 = (palm_pose - self.palm_pose_targets).clamp(
            -self.palm_rate_limits, self.palm_rate_limits
        )
        palm_pose = self.palm_pose_targets + _step6
        self.palm_pose_targets.copy_(palm_pose)
        self.hand_pca_targets.zero_()

        # fabric cspace damping: ADR 커리큘럼 10→20 (DEXTRAH fabric_damping.gain)
        if self.grasp_adr is not None:
            self.fabric_damping_gain.fill_(
                self.grasp_adr.get_param("fabric_damping", "gain")
            )

        # palm_pose_targets 는 G 규약 euler(ey≈0 중심, 특이점 없음)로 유지하고,
        # fabric 에는 quaternion 으로 넘긴다 — 최종 행렬은 euler 로 표현 불가(ey=±90).
        # matrix 는 고정 버퍼에 in-place 로 써서 CUDA Graph 캡처 입력 주소를 유지한다.
        self.palm_matrix_buf.copy_(
            g_pose_to_fabric_matrix(self.palm_pose_targets, self.palm_grasp_frame_rot)
        )
        if not bool(self.cfg.use_cuda_graph):
            self.open_tesollo_fabric.set_features(
                self.hand_pca_targets,
                self.palm_matrix_buf,
                "matrix",
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
        else:
            # CUDA Graph replay: 캡처된 fabric IK 커널을 최신 입력(위 in-place 갱신)으로 재생.
            for _ in range(self.cfg.fabric_decimation):
                self.g.replay()
                self.fabric_q.copy_(self.fabric_q_new)
                self.fabric_qd.copy_(self.fabric_qd_new)
                self.fabric_qdd.copy_(self.fabric_qdd_new)

        # ---- 손 제어: per-finger(grasp_v1) 또는 시너지(DEXTRAH PCA) ----
        # cfg.finger_control_mode 로 고른다. 기본은 per_finger — PCA basis 가
        # 20관절을 커플링해 형상 적응이 불가능했다(cfg 주석에 근거 상술).
        if self.cfg.finger_control_mode == "per_finger":
            p_star = compute_per_finger_progress_targets(finger_action)   # (N,20) ∈ [0,1]
        else:
            p_star = compute_synergy_progress_targets(
                finger_action,
                self.hand_synergy_basis, self.hand_synergy_anchor,
                self.hand_synergy_mins, self.hand_synergy_maxs,
                self.hand_open_pose, self.hand_full_grip_pose,
            )                                                            # (N,20) ∈ [0,1]
        # 다물체 drop-settle: episode 초기 settle_steps 동안 손가락 폐쇄 억제 →
        # 물체(DEXTRAH식 고정 높이 spawn)가 낙하해 테이블에 안착(grasp_v1 정지물체 전제).
        # (in_settle 은 위 palm 동결 게이트와 공유)
        p_star = torch.where(in_settle, torch.zeros_like(p_star), p_star)
        tip_c  = self.binary_contact_buf.float()                    # (N,5) 끝
        dist_c = self.distal_binary_contact_buf.float()             # (N,5) distal(rl_dg_X_4)
        mid_c  = self.middle_binary_contact_buf.float()             # (N,5) middle(rl_dg_X_3)
        # 관절별 동결 게이트 (local 0=_1, 1=_2, 2=_3 PIP, 3=_4 DIP)
        g1 = torch.zeros_like(tip_c)                                # _1: 무게이트
        g2 = torch.zeros_like(tip_c)                                # _2 MCP: 무게이트(full close)
        if self.cfg.synergy_freeze_enable:
            g3 = mid_c                                              # _3 PIP: 중간마디 접촉 시 동결
            g4 = (dist_c + tip_c).clamp(max=1.0)                    # _4 DIP: distal|tip 접촉 시 동결
        else:
            # 동결 제거: 손가락이 물체를 계속 조임(물리 collision이 관통/형상적응 담당) → 파지력.
            g3 = torch.zeros_like(tip_c)
            g4 = torch.zeros_like(tip_c)
        gate20 = torch.stack([g1, g2, g3, g4], dim=2).reshape(self.num_envs, -1)  # (N,20)
        # 래칫(전진만): 양방향 추종은 h2o max-거리 보상과 결합해 "손 펴서 손끝을
        # 물체에 붙이기" 국소최적을 만든다(test8 실증: f2~f4 action -0.95 수렴,
        # in_success 0). per-finger 시절의 전진-only semantics 복원 — 정책은
        # "언제/얼마나 감기 시작할지"만 결정, 재개방은 reset에서만.
        _step = float(self.cfg.finger_close_speed)
        delta = (p_star - self.finger_close_buf).clamp(0.0, _step) * (1.0 - gate20)
        self.finger_close_buf = (self.finger_close_buf + delta).clamp(0.0, 1.0)  # (N,20)
        hand_target = torch.lerp(
            self.hand_open_pose.unsqueeze(0).expand(self.num_envs, -1),
            self.hand_full_grip_pose.unsqueeze(0).expand(self.num_envs, -1),
            self.finger_close_buf,                                  # (N,20) 관절별 진행도
        ).clamp(
            self.hand_joint_lower_limits.unsqueeze(0),
            self.hand_joint_upper_limits.unsqueeze(0),
        )

        # ---- abduction (thumb_1/index_1/pinky_1/pinky_2): 시너지 바깥 절대 목표 ----
        # 시너지 basis 열이 0 이라 진행도 경로로는 절대 안 움직인다 → 여기서 덮어쓴다.
        # 양방향(래칫 아님): 납작한 물체는 손끝 핀치, 큰 물체는 감싸기 — 자세를 바꿔야 한다.
        # ADR 커리큘럼: range_scale 0 → 1 로 abduction 자유도를 연다.
        # scale=0 이면 항상 중립(HAND_APPROACH_POSE 값 = thumb_2 -90° 등)이라
        # 실효 action 이 11D(DEXTRAH 원본)가 된다. probe 실증상 그 자세가 파지에 최적이고,
        # abduction 을 벌리면 오히려 못 잡는다 → 기본 파지를 배운 뒤에 연다.
        _abd_scale = self._adr("abduction", "range_scale", default=1.0)
        abd_goal = compute_abduction_targets(
            abduction_action, self.abduction_limits_min, self.abduction_limits_max
        )
        abd_goal = self.abduction_neutral.unsqueeze(0) + _abd_scale * (
            abd_goal - self.abduction_neutral.unsqueeze(0)
        )
        # settle 동안 중립(0) 유지 — 손가락 폐쇄 억제와 동일 게이트.
        abd_goal = torch.where(in_settle, torch.zeros_like(abd_goal), abd_goal)
        # rate limit: 자기충돌 검사가 꺼져 있어 순간이동은 인접 손가락을 관통한다.
        _abd_step = (abd_goal - self.abduction_targets).clamp(
            -float(self.cfg.abduction_rate_limit), float(self.cfg.abduction_rate_limit)
        )
        self.abduction_targets += _abd_step
        hand_target[:, self.abduction_local_indices] = self.abduction_targets
        hand_target = hand_target.clamp(
            self.hand_joint_lower_limits.unsqueeze(0),
            self.hand_joint_upper_limits.unsqueeze(0),
        )
        self.hand_joint_targets.copy_(hand_target)

        # fabric_q hand 부분 동기화 (FK 계산에 활용)
        self.fabric_q[:, NUM_ARM_DOF:] = hand_target
        self.fabric_qd[:, NUM_ARM_DOF:].zero_()

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

        # ---- 오른손 ----
        # Both phases use policy-controlled absolute synergy targets.
        finger_target = self.hand_joint_targets
        self.robot.set_joint_position_target(finger_target, joint_ids=self.hand_dof_indices)
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

        # hand points (palm + 5 tips): fabric FK on noisy q → noisy hand pos/vel
        _palm_pts, _palm_jac = self.open_tesollo_fabric.get_taskmap("palm")(
            self.robot_dof_pos_noisy, None
        )
        _tip_pts, _tip_jac = self.open_tesollo_fabric._fingertip_taskmap(
            self.robot_dof_pos_noisy, None
        )
        self.hand_pos_noisy = torch.cat([_palm_pts[:, :3], _tip_pts], dim=-1)  # (N, 18)
        _vel_palm = torch.bmm(
            _palm_jac[:, :3, :], self.robot_dof_vel_noisy.unsqueeze(2)
        ).squeeze(2)
        _vel_tips = torch.bmm(
            _tip_jac, self.robot_dof_vel_noisy.unsqueeze(2)
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
        # [FP 배포 검증] eval_pose_hold: grasp 단계(물체 정적·폐색)에만 held(settle 고정)
        # pose 사용 = FoundationPose lock. approach/settle 과 lift 는 live(lift 는 물체가
        # 손과 함께 상승 → 운동학 브리지 proxy). 이렇게 분리해야 "정적 grasp 구간에서
        # FP-lock 이 통하는지"를 lift-phase 불일치와 섞이지 않게 격리 측정한다.
        if bool(self.cfg.eval_pose_hold):
            _in_grasp = (
                (self.episode_length_buf >= int(self.cfg.settle_steps))
                & (self.episode_length_buf < LIFT_START_STEP)
            ).unsqueeze(-1)
            _obj_pos_src = torch.where(_in_grasp, self.object_pos_held, self.object_pos)
            _obj_rot_src = torch.where(_in_grasp, self.object_rot_held, self.object_rot)
        else:
            _obj_pos_src = self.object_pos
            _obj_rot_src = self.object_rot
        self.object_pos_noisy = (
            _obj_pos_src
            + _op_w * 2.0 * (torch.rand_like(self.object_pos) - 0.5)
            + self.object_pos_bias
        )
        self.object_rot_noisy = (
            _obj_rot_src
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
            self.robot_dof_pos_noisy,        # 27
            self.robot_dof_vel_noisy,        # 27 (annealing=0 → 상시 0)
            self.hand_pos_noisy,             # 18 (fabric FK: palm+5tip)
            self.hand_vel_noisy,             # 18 (0)
            self.object_pos_noisy,           # 3
            self.object_rot_noisy,           # 4
            self.object_goal,                # 3 (고정 절대점)
            self.multi_object_idx_onehot,    # N_obj
            self.object_scale,               # 1
            self.actions,                    # 16
            self.fabric_q,                   # 27
            self.fabric_qd,                  # 27
            self.fabric_qdd,                 # 27
            # fingertip 접촉력 15D(5tip × xyz, force_matrix Cup-only). 정책이 접촉을
            # "보고" force closure 를 조율하게 한다(실물 FT 센서 대응). CONTACT_FORCE_MAX
            # 로 정규화 — 원시 N 값은 스케일이 커 obs 통계를 교란.
            (self.contact_force_xyz_raw / CONTACT_FORCE_MAX).reshape(self.num_envs, -1),  # 15
        ], dim=-1)   # 208 + N_obj

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
            self.robot_dof_pos,              # 27
            self.robot_dof_vel,              # 27
            hand_pos_clean,                  # 18
            hand_vel_clean,                  # 36
            hand_forces,                     # 3
            measured_joint_torque,           # 27
            self.object_pos,                 # 3
            self.object_rot,                 # 4
            self.cup.data.root_vel_w,        # 6
            self.object_goal,                # 3
            self.multi_object_idx_onehot,    # N_obj
            self.object_scale,               # 1
            self.actions,                    # 16
            self.fabric_q,                   # 27
            self.fabric_qd,                  # 27
            self.fabric_qdd,                 # 27
            # privileged 접촉: tip xyz 15 + distal/middle norm 10 (Cup-only force_matrix)
            (self.contact_force_xyz_raw / CONTACT_FORCE_MAX).reshape(self.num_envs, -1),  # 15
            self.distal_contact_force_raw / CONTACT_FORCE_MAX,   # 5
            self.middle_contact_force_raw / CONTACT_FORCE_MAX,   # 5
        ], dim=-1)   # 272 + N_obj

        if critic_obs.shape[1] != self.cfg.state_space:
            raise RuntimeError(
                f"Critic obs dim mismatch: {critic_obs.shape[1]} != {self.cfg.state_space}"
            )

        if not self.use_camera:
            return {"policy": actor_obs, "critic": critic_obs}

        # ==== distillation: student(vision) 가 "policy", teacher 는 "expert_policy" ====
        # depth 유효 밴드 밖은 0 으로 죽인다. mask 는 배경(=밴드 초과) 픽셀 —
        # aux depth 재구성 손실에서 배경을 제외하는 데 쓴다.
        depth = self._tiled_camera.data.output["depth"].clone()
        # 중앙 crop → 물체 detail 확보(dt1/dt2 고원: 물체 ~19px). 중앙 crop_frac 만큼을
        # 잘라 원 해상도로 업샘플 → 물체 화면 크기 1/crop_frac 배(0.5→2배 ~37px). 유효
        # HFOV 87°→~44°. depth 는 (N,H,W,1) → interpolate 위해 (N,1,H,W) 로 permute.
        _cf = float(self.cfg.camera_crop_frac)
        if _cf < 0.999:
            _d = depth.permute(0, 3, 1, 2)                       # (N,1,H,W)
            _n, _c, _h, _w = _d.shape
            _ch, _cw = int(_h * _cf), int(_w * _cf)
            _t, _l = (_h - _ch) // 2, (_w - _cw) // 2
            _d = _d[:, :, _t:_t + _ch, _l:_l + _cw]              # 중앙 crop
            _d = torch.nn.functional.interpolate(
                _d, size=(_h, _w), mode="nearest"                # depth 는 nearest(값 보존)
            )
            depth = _d.permute(0, 2, 3, 1).contiguous()          # (N,H,W,1) 로 복귀
        mask = depth.permute((0, 3, 1, 2)) > self.cfg.d_max
        depth[depth <= 1e-8] = 10.0        # 렌더 미스(0) → 무효로 밀어냄
        depth[depth > self.cfg.d_max] = 0.0
        depth[depth < self.cfg.d_min] = 0.0

        return {
            "policy": self.compute_student_policy_observations(),
            "expert_policy": actor_obs,
            "critic": critic_obs,
            "img": depth.permute((0, 3, 1, 2)),
            "rgb": self._tiled_camera.data.output["rgb"].clone().permute(
                (0, 3, 1, 2)
            ) / 255.0,
            "aux_info": {"object_pos": self.object_pos},
            "mask": mask,
        }

    def compute_student_policy_observations(self) -> torch.Tensor:
        """student obs (189) — 물체 privileged state 없음.

        teacher obs 에서 object_pos/rot/onehot/scale 을 뺀 것. 물체 정보는
        D435i RGB-D 에서 추론해야 하므로 관측으로 주지 않는다. object_goal 은
        고정 절대점이라 실기에서도 알 수 있으므로 남긴다.
        """
        student_obs = torch.cat([
            self.robot_dof_pos_noisy,        # 27
            self.robot_dof_vel_noisy,        # 27
            self.hand_pos_noisy,             # 18
            self.hand_vel_noisy,             # 18
            self.object_goal,                # 3
            self.actions,                    # 16
            self.fabric_q,                   # 27
            self.fabric_qd,                  # 27
            self.fabric_qdd,                 # 27
            # 접촉력 15D: 실물 student 도 FT 센서로 얻는 값이라 privileged 아님 → 유지.
            (self.contact_force_xyz_raw / CONTACT_FORCE_MAX).reshape(self.num_envs, -1),  # 15
        ], dim=-1)   # 204

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

        # 3) lift: goal 높이 절대거리 (ADR로 5→0 감쇠 — 후반은 object_to_goal 이 담당)
        #    grip_frac 게이트(cfg.lift_grip_gate_enable): 물체를 안 잡으면 lift 보상 0.
        #    DEXTRAH 원본 lift 는 "물체가 올라가면" 무조건 보상이라, 정책이 손을 안 쓰고
        #    물체가 저절로 오르길 기다리는 passive baseline 이 부분최적이 됐다(리프트 0).
        #    grip_frac(임의 마디 실접촉 손가락 비율)을 곱해 "잡아야 lift 신호"로 만든다.
        object_vertical_err = (self.object_goal[:, 2] - self.object_pos[:, 2]).abs()
        lift_reward = _lift_w * torch.exp(
            -float(self.cfg.lift_sharpness) * object_vertical_err
        )
        num_grip_fingers = (
            self.binary_contact_buf
            | self.middle_binary_contact_buf
            | self.distal_binary_contact_buf
        ).sum(dim=-1)
        grip_frac = (num_grip_fingers.float() / NUM_FINGERTIPS).clamp(0.0, 1.0)
        if bool(self.cfg.lift_grip_gate_enable):
            lift_reward = lift_reward * grip_frac

        # 4) finger_curl_reg (DEXTRAH: -0.01 → -0.005)
        #
        # 기준은 HAND_APPROACH_POSE(열린 자세) 다 — FULL_GRIP(완전 주먹)이 아니다.
        # DEXTRAH 원본 curled_q 는 index/middle/ring 전부 0(편 자세) + thumb 만 대향으로,
        # 원본 주석이 의도를 명시한다: "fingers seem to curl in a lot to play with the
        # object. A good strategy is to approach the object with wider set fingers and
        # then encase the object flexing inwards" — 즉 과도한 말림을 막는 정규화다.
        #
        # FULL_GRIP 을 기준으로 쓰면 부호가 정반대가 된다: 빈손으로 주먹을 쥐면 거리 0
        # (페널티 0)이고, 물체를 잡으면 손가락이 물체에 막혀 FULL_GRIP 에 못 가 페널티가
        # 붙는다 → "물체에 가까이 가면 손해" → 정책이 물체에서 도망간다.
        # lstm_test3(G프레임) 실증: hand_to_object 가 epoch 50 에 0.256 까지 올랐다가
        # epoch 100 에 0.028 로 급락(= 거리 34cm). 렌더 관찰의 "제자리 주먹"과 일치.
        # HAND_APPROACH_POSE(thumb 대향 + 나머지 편 자세)는 DEXTRAH curled_q 와 구조가 같다.
        finger_pos = self.robot.data.joint_pos[:, self.hand_dof_indices]
        finger_curl_dist = (
            finger_pos - self.hand_open_pose.unsqueeze(0)
        ).norm(p=2, dim=-1).clamp(max=float(self.cfg.finger_curl_dist_max))
        finger_curl_reg = _curl_w * finger_curl_dist ** 2

        # palm_align: 진단 로깅 전용 (reward 아님).
        #
        # palm_orient reward(weight 1.0, sharp 3.0)는 제거했다. DEXTRAH 원본에 없는
        # 우리 추가분이었고, 거리 무관항이라 손이 물체에서 27cm 떨어져 있어도 0.877 을
        # 준다 → lstm_test3 에서 정책이 접근을 포기하고 정렬 보상만 먹는 orientation
        # hacking 의 직접 동력이었다(in_success 0.000, hand_to_object 0.066).
        # 도입 목적("손바닥이 천장 향하는 것 차단")은 G 규약 palm 경계(ex 중심 ±45°)가
        # 구조적으로 보장하므로 중복이다.
        palm_quat = self.robot.data.body_quat_w[:, self.palm_body_index]   # (N,4) wxyz
        palm_x_local = torch.zeros_like(self.palm_center_pos)
        palm_x_local[:, 0] = 1.0
        palm_normal = quat_apply(palm_quat, palm_x_local)                  # (N,3) world
        palm_to_obj = self.object_pos - self.palm_center_pos
        palm_to_obj = palm_to_obj / (palm_to_obj.norm(dim=-1, keepdim=True) + 1e-8)
        palm_align = (palm_normal * palm_to_obj).sum(dim=-1)               # [-1, 1]

        # 5) grasp_contact (부트스트랩): 손끝 실접촉 유도 (force_matrix Cup-only 개수).
        #    DEXTRAH 4항엔 접촉 신호가 없어 정책이 물체를 건드릴 이유가 없다(passive).
        #    ADR 로 0.3→0.0 감쇠(파지 학습 후 끔). persist 비중 축소(0.40→0.20)로
        #    rh56f1 의 "접촉만 유지·리프트 0" 고착 함정을 완화(reward-audit REVISE).
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
        grasp_quality = 0.45 * tip_frac + 0.35 * full_tip + 0.20 * persist_frac
        _gc_w = (
            self._adr("reward_weights", "grasp_contact_weight")
            if self.grasp_adr is not None else float(self.cfg.grasp_contact_weight)
        )
        grasp_contact_reward = _gc_w * grasp_quality

        # 6) force_closure(opposition) ← 리프트 주력.
        #    접촉 반력(force_matrix_w)은 물체가 손가락을 "미는" 방향이다. 엄지와 4지가
        #    물체를 양쪽에서 마주 조이면 두 반력 벡터가 대략 반대(−cos>0) → force closure.
        #    rh56f1 이 접촉(tip 1.88)해도 못 든 진짜 결측이 이 대향 조임이다. 힘 크기(tanh)
        #    를 곱하고 엄지·4지 실접촉 AND 게이트로 hacking(한쪽만/빈손) 차단.
        #
        #    tip+distal+middle 합산 벡터(fc1 실측 반영): tesollo 긴 손가락의 감싸기는
        #    tip 이 아닌 중간 마디로 조인다 — fc1 에서 middle 0.46~0.82 상승에도 tip 0.10
        #    정체 → tip-only 게이트 fc_gate_frac 0.0005 로 force_closure 질식. 손가락별
        #    총 반력(3마디 합)과 any-segment 게이트(lift grip_frac 과 동일 기준)로 교정.
        _f = (
            self.contact_force_xyz_raw
            + self.distal_contact_force_xyz_raw
            + self.middle_contact_force_xyz_raw
        )                                                     # (N,5,3) 손가락별 총 반력
        _f_thumb = _f[:, 0]                                   # (N,3)
        _f_others = _f[:, 1:].mean(dim=1)                     # (N,3) 4지 평균
        _scale = float(self.cfg.force_closure_force_scale)
        _t_norm = _f_thumb.norm(dim=-1)
        _o_norm = _f_others.norm(dim=-1)
        _cos = (_f_thumb * _f_others).sum(dim=-1) / (_t_norm * _o_norm + 1e-6)
        opposition = (-_cos).clamp(min=0.0)                  # 마주볼 때만(0~1)
        grip_strength = torch.tanh(_t_norm / _scale) * torch.tanh(_o_norm / _scale)
        _any_c = (
            self.binary_contact_buf
            | self.middle_binary_contact_buf
            | self.distal_binary_contact_buf
        )                                                     # (N,5) 임의 마디 접촉
        _thumb_c = _any_c[:, 0]
        # [07-21] "아무거나 1개"→"최소 N개". 헐거운 게이트가 3지(엄지+2)만으로 force_closure
        # 보상을 다 주던 문제 수정(lstm_test2: ring raw action이 학습하며 -0.9까지 붕괴 —
        # 굽힐 이유가 구조적으로 없었음). grasp_contact_weight 와 같은 ADR 커리큘럼(1→3,
        # adr_custom_cfg["reward_weights"])으로 초기 부트스트랩은 느슨하게 유지.
        _min_others = self._adr(
            "reward_weights", "force_closure_min_others",
            default=float(self.cfg.force_closure_min_others),
        )
        _others_c = (_any_c[:, 1:].sum(dim=-1) >= round(_min_others))
        fc_gate = (_thumb_c & _others_c).float()
        opposition_quality = fc_gate * opposition * grip_strength
        force_closure_reward = float(self.cfg.force_closure_weight) * opposition_quality

        total = (
            hand_to_object_reward + object_to_goal_reward + finger_curl_reg
            + lift_reward + grasp_contact_reward + force_closure_reward
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
        # reward 항은 reward/ 그룹으로 묶어 TB에서 접히게 (rl_games 자체의
        # rewards/(에피소드 합)와 구분 — 이쪽은 항별 per-step 평균)
        self.extras["reward/hand_to_object"] = hand_to_object_reward.mean()
        self.extras["reward/object_to_goal"] = object_to_goal_reward.mean()
        self.extras["reward/finger_curl_reg"] = finger_curl_reg.mean()
        self.extras["reward/lift"] = lift_reward.mean()
        self.extras["reward/grasp_contact"] = grasp_contact_reward.mean()
        self.extras["reward/force_closure"] = force_closure_reward.mean()
        # opposition: 게이트 통과(양쪽 접촉) env 에서의 평균 대향도. force closure 학습 진단.
        self.extras["contact/opposition"] = (
            (opposition * fc_gate).sum() / fc_gate.sum().clamp(min=1.0)
        )
        self.extras["contact/fc_gate_frac"] = fc_gate.mean()
        self.extras["palm_align"] = palm_align.mean()
        self.extras["in_success_region"] = self.in_success_region.float().mean()
        # 접근 자세 분기 검증용: top-down 으로 배정된 env 비율 (ADR 회전이 커지면 상승)
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
        # contact: 감싸기 노드 그룹별 접촉 손가락 수 (0~5). palm 센서 없음 → 제외.
        #
        # [정정 07.20] 아래 문단은 낡은 진단(오진)이다 — _setup_scene 참조: probe_contact_filter
        # 실측으로 filter_prim_paths_expr(force_matrix_w)는 GPU에서 정상 동작하며, _update_contact_forces
        # 가 실제로 읽는 값도 force_matrix_w(Cup-only 필터, 52e0fb9)다. net_forces_w 는 쓰지 않는다.
        # 즉 contact/tip·middle·distal 은 현재 **물체 전용** 접촉이다. "grip 3.2인데 object_height 0"
        # 모순은 당시(필터를 net_forces_w로 오독하던 시기) 증상이었고 지금은 재현되지 않는다.
        # 아래 contact/grip_near 는 여전히 유지 — 물체-테이블 혼동이 아니라 손끝-물체 실거리 교차검증용.
        self.extras["contact/tip"]    = self.num_contacts_buf.float().mean()
        self.extras["contact/middle"] = self.middle_binary_contact_buf.float().sum(dim=-1).mean()
        self.extras["contact/distal"] = self.distal_binary_contact_buf.float().sum(dim=-1).mean()
        self.extras["contact/grip"]   = num_grip_fingers.float().mean()

        # ---- 손가락별 분해 (07-20 추가) ----
        # 위 contact/tip·middle·distal 은 5지 합산이라 "어느 손가락이 안 굽는지"를
        # 가린다(예: 중지·약지 tip=0이어도 합산 곡선은 다른 손가락이 메워 정상으로 보임).
        # synergy_freeze_enable 같은 제어 변경이 특정 손가락에 실제로 작동하는지 확인하려면
        # 손가락별 분해가 필수 — finger_close_progress(명령 진행도, 접촉 무관·"굽히려 하는가")
        # 와 손가락별 접촉(결과, "닿았는가")을 나란히 봐야 원인(제어)과 결과(접촉)를 분리할 수 있다.
        _FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
        _close_per_finger = self.finger_close_buf.view(self.num_envs, 5, 4).mean(dim=2)   # (N,5) 관절4개 평균 진행도
        _dip_per_finger = self.finger_close_buf.view(self.num_envs, 5, 4)[:, :, 3]        # (N,5) DIP(_4, tip측) 진행도만
        for _fi, _fn in enumerate(_FINGER_NAMES):
            self.extras[f"close_progress/{_fn}"]     = _close_per_finger[:, _fi].mean()
            self.extras[f"close_progress_dip/{_fn}"] = _dip_per_finger[:, _fi].mean()
            self.extras[f"contact/tip_{_fn}"]    = self.binary_contact_buf[:, _fi].float().mean()
            self.extras[f"contact/distal_{_fn}"] = self.distal_binary_contact_buf[:, _fi].float().mean()
            self.extras[f"contact/middle_{_fn}"] = self.middle_binary_contact_buf[:, _fi].float().mean()

        # ---- 진짜 물체 접촉 / 실거리 (센서의 물체-테이블 혼동을 우회) ----
        # 손끝↔물체 거리로 게이트한다: 접촉 센서가 켜졌고 **그 손끝이 물체 근처**면
        # 물체 접촉으로 본다. 테이블만 짚은 경우는 거리가 멀어 걸러진다.
        _tip_d = (self.fingertip_pos - self.object_pos.unsqueeze(1)).norm(dim=-1)   # (N,5)
        _near = _tip_d < float(self.cfg.contact_near_dist)
        self.extras["contact/grip_near"] = (
            self.binary_contact_buf & _near
        ).float().sum(dim=-1).mean()
        # h2o 보상은 exp(-10·d) 라 꼬리에서 실거리가 안 보인다 — 원거리도 그대로 로깅한다.
        self.extras["dist/tip_min"]  = _tip_d.min(dim=1).values.mean()
        self.extras["dist/tip_max"]  = _tip_d.max(dim=1).values.mean()
        self.extras["dist/palm_obj"] = (
            self.palm_center_pos - self.object_pos
        ).norm(dim=-1).mean()
        # 물체를 건드리기는 하는가 (안착점 대비 수평 변위) — 접근 성공의 직접 증거
        self.extras["engage/obj_disp"] = (
            self.object_pos[:, :2] - self.object_init_pos[:, :2]
        ).norm(dim=-1).mean()

        # ---- palm 자세: 정책이 top-down 을 버리고 side 로 도망치는가 ----
        # LEFT 실측 — 성공 순간 법선이 top-down(0,0,-1)에서 54° 기울어져 있었고 palm
        # 회전 action 3축이 전부 경계(-0.99)에 붙어 있었다. 그 탈출을 로그로 잡는다.
        _tilt = torch.acos(
            (-palm_normal[:, 2]).clamp(-1.0, 1.0)
        ) * (180.0 / math.pi)                                    # 0°=순수 top-down
        self.extras["palm/topdown_deg"] = _tilt.mean()
        # action 포화율: 정책이 action 공간 구석에 몰렸는지 (제약이 병목이라는 신호)
        _a = self.actions
        self.extras["sat/palm_rot"]  = (_a[:, 3:6].abs() > 0.95).float().mean()
        self.extras["sat/palm_pos"]  = (_a[:, 0:3].abs() > 0.95).float().mean()
        self.extras["sat/finger"]    = (_a[:, 6:11].abs() > 0.95).float().mean()
        # abduction 실제 관절 목표 (rad) — 정책이 이 축을 실제로 쓰는지 확인용.
        # 전부 0 근처면 자유화가 무의미한 것이고, 부호가 범위 반대면 미러 버그다.
        for _i, _nm in enumerate(("thumb_1", "thumb_2", "index_1", "pinky_1", "pinky_2")):
            self.extras[f"abduction/{_nm}"] = self.abduction_targets[:, _i].mean()
        # action policy(palm 6D + finger 5D raw) 로깅 유지
        for k, v in action_policy_scalars(
            action=self.actions, prev_action=self.prev_actions, palm_dims=6,
        ).items():
            self.extras[k] = v
        # episode_success_rate: 물체별 성공률 + 전체 (누적 집계는 _reset_idx 에서 갱신)
        # counter 는 리셋에서만 갱신 → GPU 텐서를 .tolist() 로 한 번에 전송(물체당 .item()
        # 2회 = 매 스텝 최대 296 sync → 2 sync). fps sawtooth 의 CPU 구간 축소.
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

        # 종료 원인 분해 — 이게 없어서 "에피소드가 199 스텝에 끝난다"까지만 알고
        # 넷 중 무엇 때문인지 몰랐다. 종료된 env 중 각 원인의 비율로 기록한다
        # (순간 평균은 대부분 0 이라 무의미하다).
        _nt = terminated.sum().clamp(min=1).float()
        self.extras["term/out_x"]          = out_x.float().sum() / _nt
        self.extras["term/out_y"]          = out_y.float().sum() / _nt
        self.extras["term/fallen"]         = fallen.float().sum() / _nt
        self.extras["term/robot_diverged"] = robot_diverged.float().sum() / _nt
        self.extras["term/rate"]           = terminated.float().mean()

        # distillation: 성공을 success_timeout 스텝 유지하면 조기 종료.
        # teacher 학습에선 이 경로를 타지 않는다(에피소드 길이 = reward 스케줄 전제).
        if self.cfg.distillation:
            self.time_in_success_region = torch.where(
                self.success_flag,
                self.time_in_success_region + 1,
                torch.zeros_like(self.time_in_success_region),
            )
            truncated = truncated | (
                self.time_in_success_region >= self.cfg.success_timeout
            )

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

        self.time_in_success_region[env_ids] = 0

        if self.visual_dr is not None:
            self.visual_dr.randomize(env_ids)

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
            # 물체는 DEXTRAH 원본처럼 공중에서 떨어뜨린다(object_start_state[:,2]=0.5).
            # 낙하하며 굴러 위치·자세가 랜덤해지는 것이 의도된 도메인 랜덤화다.
            _spawn_rot_for_z = self._sample_spawn_rotation(n)
            obj_pos_local = torch.stack(
                [obj_x, obj_y, torch.full((n,), self.cfg.object_spawn_z, device=self.device)],
                dim=1,
            )
            # 단, pregrasp 는 "안착 예상 높이" 기준으로 잡는다. 낙하 전 spawn z 를 쓰면
            #   palm~물체 = (spawn_z + half_z + REACH) - (table + half_z) = 0.157
            # 로 half_z 가 소거돼 물체 크기와 무관하게 항상 15.7cm 가 되고, 손가락(~10cm)이
            # 닿지 못한다. 낙하 중 xy 가 굴러 바뀌는 것은 정책이 obs(실시간 물체 위치)로
            # 보정한다 — palm action 이 절대 pose 라 1스텝에 지령할 수 있다.
            _half_z_settled = compute_rotated_half_z(
                self.object_half_extent[self.object_idx[env_ids]],
                matrix_from_quat(_spawn_rot_for_z),
            )
            obj_pos_settled = obj_pos_local.clone()
            obj_pos_settled[:, 2] = TABLE_TOP_Z + _half_z_settled

            # ---- 접근 자세 결정: cup → side, 그 외 → top-down ----
            spawn_rot = _spawn_rot_for_z          # 스폰 높이 계산에 쓴 것과 동일해야 한다
            if self.cfg.approach_branch_enable:
                pose_id = self._compute_palm_pose_id(self.object_idx[env_ids])  # 0=side, 1=top
            else:
                pose_id = torch.zeros(n, dtype=torch.long, device=self.device)
            self.palm_pose_id[env_ids] = pose_id
            self.palm_mins_env[env_ids] = self.palm_mins_by_pose[pose_id]
            self.palm_maxs_env[env_ids] = self.palm_maxs_by_pose[pose_id]

            # ---- FABRICS pregrasp rollout/cache lookup ----
            noise = torch.stack([
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_x,
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_y,
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_z,
            ], dim=1)
            pregrasp_pos = obj_pos_settled + self._compute_pregrasp_offset(
                self.object_idx[env_ids], pose_id, spawn_rot
            ) + noise

            # pregrasp 회전 = G 규약 euler (pose 별 상수). P 규약 하드코드 금지 —
            # 가짜 top-down(손바닥이 옆을 봄)이 여기서 났다.
            _g_top = to_torch(
                [math.radians(v) for v in PREGRASP_G_EULER_TOPDOWN], device=self.device
            )
            _g_side = to_torch(
                [math.radians(v) for v in PREGRASP_G_EULER_SIDE], device=self.device
            )
            _g_euler = torch.where(
                (pose_id == 1).unsqueeze(-1),
                _g_top.unsqueeze(0).expand(n, -1),
                _g_side.unsqueeze(0).expand(n, -1),
            )                                                        # (n,3)
            pregrasp_palm_pose = torch.zeros(n, 6, device=self.device)
            pregrasp_palm_pose[:, :3] = pregrasp_pos
            pregrasp_palm_pose[:, 3:6] = _g_euler
            pregrasp_palm_pose = torch.max(
                torch.min(pregrasp_palm_pose, self.palm_maxs_env[env_ids]),
                self.palm_mins_env[env_ids],
            )

            q_pregrasp = self._run_reset_fabric(env_ids, pregrasp_palm_pose, q_pregrasp)

            # hand는 APPROACH_POSE로 강제
            q_pregrasp[:, NUM_ARM_DOF:] = approach_hand

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

        self.fabric_q[env_ids] = q_pregrasp
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
        self.open_tesollo_fabric.default_config[env_ids, :NUM_ARM_DOF] = q_pregrasp[:, :NUM_ARM_DOF]

        prelift_arm = compute_joint7_lift_wait_target(
            q_pregrasp[:, :NUM_ARM_DOF],
            joint7_delta=getattr(self.cfg, "lift_wait_joint7_delta", 0.31),
            joint7_min=self.cfg.warm_j7_min,
            joint7_max=self.cfg.warm_j7_max,
        )
        self.prelift_arm_pos_buf[env_ids] = prelift_arm

        # ---- 7. 컵 spawn ----
        # spawn_rot 은 위(1. Reset source)에서 이미 뽑았다 — 접근 자세 분기가 "회전 후
        # 물체 높이"를 필요로 하므로 pregrasp IK 보다 먼저 결정돼야 한다.
        obj_pos_world = obj_pos_local + self.scene.env_origins[env_ids]
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
        self.contact_persist_buf[env_ids] = 0
        # abduction 은 중립(HAND_APPROACH_POSE 값)에서 시작.
        self.abduction_targets[env_ids] = self.abduction_neutral

        # actions 리셋: delta action 방식 → action=0 = pregrasp 위치
        # (절대 pose 라 action=0 → 박스 중심. 회전은 pregrasp 자세와 동일)
        self.actions[env_ids, :6] = 0.0
        self.actions[env_ids, 6:] = -1.0
        self.prev_actions[env_ids, :6] = 0.0
        self.prev_actions[env_ids, 6:] = -1.0

        # DEXTRAH 정렬: reset 직후 obs가 noisy 중간값을 요구 (첫 reset 포함)
        self._compute_intermediate_values()
