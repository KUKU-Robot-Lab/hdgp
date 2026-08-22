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

"""gripper/left/grasp_sensor — 왼팔 2지 그리퍼 단일 컵 파지 환경.

Action 7D : TCP 6D delta (Fabrics IK → l_aj_1..7) + 그리퍼 1D
Episode   : 접근·파지 480 step → (양 핑거 접촉 래치) → 수직 리프트 120 step

right/grasp_sensor 와 같은 골격이되 아래가 다르다:
  · Fabrics cspace 가 27 이 아니라 **7**(손을 fixed 로 굳힌 좌팔 전용 URDF)
  · 리셋 홈을 IK 로 풀지 않는다 — preset 에 측정된 관절값이 있다
  · 물체가 단일 종이라 MultiAsset·onehot·per-object bbox 텐서가 없다
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

# Fabrics 경로 설정 (hdgp/source/FABRICS/src 우선)
for _parent in Path(__file__).resolve().parents:
    if _parent.name == "source":
        _vendored = _parent / "FABRICS" / "src"
        if _vendored.exists() and str(_vendored) not in sys.path:
            sys.path.insert(0, str(_vendored))
        break

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply

from openarm.common.grasp_logging import action_policy_scalars
from openarm.common.grasp_v2_contract import (
    compute_action_delta_norm,
    compute_grasp_v2_stability,
)

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmGripperLeftPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

from .grasp_left_constants import (
    CONTACT_FORCE_MAX,
    CONTACT_FORCE_THRESHOLD,
    EPISODE_STEPS,
    GRASP_PHASE_STEPS,
    GRIPPER_POS_ERR_MAX,
    LIFT_PHASE_STEPS,
    NUM_ACTIONS,
    NUM_ARM_DOF,
    NUM_CRITIC_OBSERVATIONS,
    NUM_FINGERS,
    NUM_OBSERVATIONS,
    NUM_PALM_ACTION,
)
from .grasp_left_env_cfg import GraspLeftGripperEnvCfg
from .grasp_left_preset import (
    GRASP_CUP_RADIUS,
    GRASP_DEPTH,
    GRASP_PALM_EULER_ZYX_DEG,
    GRIPPER_BASE_BODY,
    GRIPPER_CLOSED_POS,
    GRIPPER_FINGER_BODIES,
    GRIPPER_OPEN_POS,
    LEFT_ARM_HOME_JOINT_POS,
    LEFT_ARM_JOINT_NAMES,
    LEFT_GRIPPER_JOINT_NAMES,
    PREGRASP_RETREAT,
    RIGHT_REST_JOINT_POS,
    TCP_OFFSET_IN_BASE_Z,
    grasp_axes,
    palm_pose_maxs,
    palm_pose_mins,
)
from .grasp_reward import compute_gripper_grasp_reward_terms


def to_torch(value, device) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32, device=device)


def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


class GraspLeftGripperEnv(DirectRLEnv):
    """왼팔 2지 그리퍼 컵 파지."""

    cfg: GraspLeftGripperEnvCfg

    # ------------------------------------------------------------------
    def __init__(self, cfg: GraspLeftGripperEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        names = self.robot.joint_names
        self.arm_dof_indices = [names.index(n) for n in LEFT_ARM_JOINT_NAMES]
        self.gripper_dof_indices = [names.index(n) for n in LEFT_GRIPPER_JOINT_NAMES]
        # ★목표는 gripper_1 에만 준다 — gripper_2 는 USD PhysX mimic 이라 스스로 따라온다.
        #   둘 다 지령하면 mimic 제약과 드라이브가 싸운다.
        self.gripper_cmd_index = names.index("l_hj_gripper_1")
        self.idle_dof_indices = [names.index(n) for n in RIGHT_REST_JOINT_POS]

        body_names = self.robot.data.body_names
        self.gripper_base_body_index = body_names.index(GRIPPER_BASE_BODY)
        self.finger_body_indices = [body_names.index(n) for n in GRIPPER_FINGER_BODIES]

        # ── 고정 자세 버퍼 ──────────────────────────────────────────
        self.q_home_arm = to_torch(
            [LEFT_ARM_HOME_JOINT_POS[n] for n in LEFT_ARM_JOINT_NAMES], self.device
        )                                                        # (7,)
        self.idle_rest_pos = to_torch(
            [RIGHT_REST_JOINT_POS[names[i]] for i in self.idle_dof_indices], self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        # ── 액션 범위 ───────────────────────────────────────────────
        dxyz = list(self.cfg.palm_delta_xyz)
        drot = math.radians(float(self.cfg.palm_delta_rot_deg))
        self.delta_mins = to_torch([-dxyz[0], -dxyz[1], -dxyz[2], -drot, -drot, -drot], self.device)
        self.delta_maxs = -self.delta_mins
        self.palm_mins = to_torch(palm_pose_mins(), self.device)
        self.palm_maxs = to_torch(palm_pose_maxs(), self.device)

        # ── 파지 기하 ───────────────────────────────────────────────
        _width, _jaw, _approach = grasp_axes()
        self.grasp_approach_axis = to_torch(_approach, self.device)      # (3,)
        self.grasp_jaw_axis = to_torch(_jaw, self.device)                # (3,)
        self.grasp_euler = to_torch(
            [math.radians(v) for v in GRASP_PALM_EULER_ZYX_DEG], self.device
        )                                                                # (3,)
        # 컵 원점 기준 파지 TCP 오프셋 = (파지 높이 − 컵 바닥까지) + 접근축 방향 GRASP_DEPTH
        _grasp_z_rel_cup = (
            float(self.cfg.table_surface_z) + float(self.cfg.grasp_height_above_table)
            - float(self.cfg.object_spawn_z)
        )
        self.grasp_tcp_offset = (
            to_torch([0.0, 0.0, _grasp_z_rel_cup], self.device)
            + GRASP_DEPTH * self.grasp_approach_axis
        )                                                                # (3,)
        # pregrasp = 파지 자세에서 접근축 반대로 후퇴
        self.pregrasp_tcp_offset = self.grasp_tcp_offset - PREGRASP_RETREAT * self.grasp_approach_axis

        # ── 상태 버퍼 ───────────────────────────────────────────────
        N = self.num_envs
        z = lambda *s: torch.zeros(*s, device=self.device)          # noqa: E731
        self.actions = z(N, NUM_ACTIONS)
        self.prev_actions = z(N, NUM_ACTIONS)
        self.pregrasp_palm_pose_buf = z(N, 6)
        self.palm_pose_targets = z(N, 6)
        self.lift_palm_pose_buf = z(N, 6)
        self.gripper_cmd_buf = torch.full((N,), GRIPPER_OPEN_POS, device=self.device)

        self.finger_force_buf = z(N, NUM_FINGERS, 3)
        self.contact_binary_buf = torch.zeros(N, NUM_FINGERS, dtype=torch.bool, device=self.device)
        self.contact_hold_buf = torch.zeros(N, dtype=torch.long, device=self.device)
        self.contact_persistence_buf = z(N)
        self.lift_latched_buf = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.lift_start_step_buf = torch.zeros(N, dtype=torch.long, device=self.device)
        self.cup_spawn_pos = z(N, 3)
        self.prev_contact_count = z(N)
        self.success_buf = torch.zeros(N, dtype=torch.bool, device=self.device)

        self.object_pos = z(N, 3)
        self.object_rot = torch.zeros(N, 4, device=self.device)
        self.object_rot[:, 0] = 1.0
        self.tcp_pos = z(N, 3)
        self.tcp_quat = torch.zeros(N, 4, device=self.device)
        self.tcp_quat[:, 0] = 1.0
        self.finger_pos = z(N, NUM_FINGERS, 3)

        self._setup_geometric_fabrics()
        # _grip 을 여기서 한 번 채워둔다 — _get_observations 가 _get_dones 보다 먼저
        # 불리는 경로(reset 직후)에서도 항상 유효한 사전이 되게.
        self._compute_intermediate_values()
        self.extras["log"] = {}

    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table

        # 핑거 접촉 센서 — **컵만 필터**. 무필터면 핑거가 테이블/자기 몸에 닿아도
        # grip 으로 잡혀 거짓 성공이 생긴다(우측 태스크에서 실증된 버그).
        _filter = list(self.cfg.object_contact_filter)
        self._finger_sensors: list[ContactSensor] = []
        for link_name in self.cfg.gripper_finger_contact_links:
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link_name}",
                filter_prim_paths_expr=_filter,
                history_length=1,
                track_air_time=False,
            ))
            self._finger_sensors.append(sensor)
            self.scene.sensors[f"finger_sensor_{link_name}"] = sensor

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 컵은 clone 이후에 만든다. 단일 자산이라 MultiAsset 배정 버그와는 무관하지만,
        # "clone → 물체 생성" 순서를 저장소 공통 규약으로 유지한다.
        self.scene.clone_environments(copy_from_source=True)
        self.cup = RigidObject(self.cfg.cup_cfg)
        self.scene.rigid_objects["cup"] = self.cup

    # ------------------------------------------------------------------
    def _setup_geometric_fabrics(self) -> None:
        initialize_warp(self.device[-1])
        self.world_model = WorldMeshesModel(
            batch_size=self.num_envs,
            max_objects_per_env=self.cfg.fabrics_max_objects_per_env,
            device=self.device,
            # ★우팔용 월드를 쓰면 안 된다 — 거기엔 `left_arm_body`(좌팔 대역물)와
            #   `left_target_cup`(잡아야 할 컵)이 **장애물로** 들어 있어 좌팔이 자기
            #   자신과 목표물에서 밀려난다. 좌팔 전용 월드 근거는 그 yaml 머리말 참조.
            world_filename="open_gripper_left_boxes_no_table",
        )
        self.object_ids, self.object_indicator = self.world_model.get_object_ids()
        self.timestep = self.cfg.fabrics_dt

        self.fabric = OpenArmGripperLeftPoseFabric(
            self.num_envs, self.device, self.timestep,
            graph_capturable=False,
            robot_dir_name=self.cfg.fabric_robot_dir,
            robot_name=self.cfg.fabric_robot_dir,
        )
        num_joints = self.fabric.num_joints
        if num_joints != NUM_ARM_DOF:
            raise RuntimeError(
                f"fabric cspace 가 {num_joints} 이다 — 팔 {NUM_ARM_DOF} DOF 여야 한다. "
                "손 관절이 fixed 로 굳혀졌는지 확인하라 "
                "(scripts/assets_tools/generate_sensor_left_gripper_fabric_urdf.py)."
            )
        self.integrator = DisplacementIntegrator(self.fabric)

        self.fabric_q = self.q_home_arm.unsqueeze(0).repeat(self.num_envs, 1).contiguous()
        self.fabric_qd = torch.zeros(self.num_envs, num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, num_joints, device=self.device)
        # 손 fabric 미사용이지만 set_features 시그니처가 PCA 인자를 요구한다.
        self._fabric_pca = torch.zeros(self.num_envs, 5, device=self.device)
        self.fabric_damping_gain = self.cfg.fabrics_damping_gain * torch.ones(
            self.num_envs, 1, device=self.device
        )

    # ------------------------------------------------------------------
    # 물리 스텝 전: 액션 해석 + Fabrics 적분
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-1.0, 1.0))

        palm_action = self.actions[:, :NUM_PALM_ACTION]
        gripper_action = self.actions[:, NUM_PALM_ACTION]

        # ── 래치 판정: 양 핑거 접촉을 연속 hold ────────────────────
        # ⚠ 완화 금지. 1지 접촉 래치를 허용하면 부실 파지 국소최적이 생긴다.
        both_contact = self.contact_binary_buf.all(dim=-1)
        self.contact_hold_buf = torch.where(
            both_contact, self.contact_hold_buf + 1, torch.zeros_like(self.contact_hold_buf)
        )
        ready = self.contact_hold_buf >= int(self.cfg.grasp_ready_hold_steps)
        just_latching = ready & (~self.lift_latched_buf)
        self.lift_latched_buf = self.lift_latched_buf | ready
        self.lift_start_step_buf = torch.where(
            just_latching, self.episode_length_buf, self.lift_start_step_buf
        )

        # ── palm(TCP) 목표 ────────────────────────────────────────
        delta = scale(palm_action, self.delta_mins, self.delta_maxs)
        palm_pose = self.pregrasp_palm_pose_buf + delta
        lo = torch.minimum(self.palm_mins.unsqueeze(0), self.pregrasp_palm_pose_buf)
        hi = torch.maximum(self.palm_maxs.unsqueeze(0), self.pregrasp_palm_pose_buf)
        palm_pose = torch.max(torch.min(palm_pose, hi), lo)

        # 래치 시점 palm 을 고정하고 리프트 구간에는 z 만 램프 → 컵이 제자리에서 수직으로 뜬다.
        self.lift_palm_pose_buf = torch.where(
            just_latching.unsqueeze(1), self.palm_pose_targets, self.lift_palm_pose_buf
        )
        lift_prog = (
            (self.episode_length_buf - self.lift_start_step_buf).clamp(min=0).float()
            / max(1, LIFT_PHASE_STEPS - 1)
        ).clamp(max=1.0)
        lift_palm = self.lift_palm_pose_buf.clone()
        lift_palm[:, 2] = lift_palm[:, 2] + float(self.cfg.lift_height_delta) * lift_prog
        lift_palm = torch.max(
            torch.min(lift_palm, self.palm_maxs.unsqueeze(0)), self.palm_mins.unsqueeze(0)
        )
        palm_pose = torch.where(self.lift_latched_buf.unsqueeze(1), lift_palm, palm_pose)
        self.palm_pose_targets.copy_(palm_pose)

        self.fabric.set_features(
            self._fabric_pca,
            self.palm_pose_targets,
            "euler_zyx",
            self.fabric_q.detach(),
            self.fabric_qd.detach(),
            self.object_ids,
            self.object_indicator,
            self.fabric_damping_gain,
        )
        for _ in range(self.cfg.fabric_decimation):
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.integrator.step(
                self.fabric_q.detach(), self.fabric_qd.detach(),
                self.fabric_qdd.detach(), self.timestep,
            )

        # ── 그리퍼 목표 ───────────────────────────────────────────
        # action -1 → 개방(0.044), +1 → 폐쇄(0.0)
        target = GRIPPER_OPEN_POS + 0.5 * (gripper_action + 1.0) * (
            GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS
        )
        # 래치 후에는 잡은 폭을 유지한다(리프트 중 정책이 열어버리는 것을 막는다).
        self.gripper_cmd_buf = torch.where(self.lift_latched_buf, self.gripper_cmd_buf, target)

    # ------------------------------------------------------------------
    def _apply_action(self) -> None:
        arm_target = self.fabric_q[:, :NUM_ARM_DOF]
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_dof_indices)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(arm_target), joint_ids=self.arm_dof_indices
        )
        # 그리퍼: gripper_1 에만 지령 (gripper_2 는 USD mimic)
        self.robot.set_joint_position_target(
            self.gripper_cmd_buf.unsqueeze(1), joint_ids=[self.gripper_cmd_index]
        )
        # 유휴 오른팔·오른손 rest 유지
        self.robot.set_joint_position_target(
            self.idle_rest_pos, joint_ids=self.idle_dof_indices
        )

    # ------------------------------------------------------------------
    def _update_contact_forces(self) -> None:
        # force_matrix_w (N, sensor_body=1, filter, 3) → filter 축 합산 → (N, 3)
        forces = torch.stack([
            s.data.force_matrix_w[:, 0, :, :].sum(dim=1) for s in self._finger_sensors
        ], dim=1)                                                # (N, 2, 3)
        self.finger_force_buf.copy_(forces)
        self.contact_binary_buf.copy_(forces.norm(dim=-1) > CONTACT_FORCE_THRESHOLD)

    def _compute_intermediate_values(self) -> None:
        """물체·TCP·핑거 위치와 접촉을 sim 에서 다시 읽는다.

        ⚠ `_get_dones` **와** `_get_observations` 양쪽에서 부른다.
          Isaac Lab step 순서가 `_get_dones → _get_rewards → _reset_idx → _get_observations`
          이라, dones 앞에서만 갱신하면 **리셋된 env 의 관측이 리셋 전 값**이 된다.
          (이 저장소에서 반복 발생한 "reset 직후 위치버퍼 stale" 함정 — object_pos=0.000)
          두 번 읽는 비용은 텐서 몇 개라 이 버그를 감수할 이유가 없다.
        """
        origins = self.scene.env_origins
        self.object_pos = self.cup.data.root_pos_w - origins
        self.object_rot = self.cup.data.root_quat_w

        base_pos = self.robot.data.body_pos_w[:, self.gripper_base_body_index, :] - origins
        base_quat = self.robot.data.body_quat_w[:, self.gripper_base_body_index, :]
        self.tcp_quat = base_quat
        # TCP 는 body 가 아니라 gripper_base 의 +z 오프셋이다 (physics USD 에 tcp 강체 없음).
        self.tcp_pos = base_pos + quat_apply(
            base_quat,
            to_torch([0.0, 0.0, TCP_OFFSET_IN_BASE_Z], self.device).unsqueeze(0).expand(self.num_envs, -1),
        )
        self.finger_pos = (
            self.robot.data.body_pos_w[:, self.finger_body_indices, :] - origins.unsqueeze(1)
        )
        self._update_contact_forces()
        self._grip = self._grip_metrics()

    # ------------------------------------------------------------------
    # 파지 품질 지표
    # ------------------------------------------------------------------
    def _grip_metrics(self) -> dict[str, torch.Tensor]:
        cup_xy = self.object_pos[:, :2].unsqueeze(1)                     # (N,1,2)
        radial = self.finger_pos[:, :, :2] - cup_xy                      # (N,2,2)
        radial_hat = radial / radial.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # 대향도: 두 핑거의 방사방향이 서로 반대일수록 1 (force-closure 최소조건)
        opposition = (-(radial_hat[:, 0] * radial_hat[:, 1]).sum(dim=-1)).clamp(0.0, 1.0)

        contact_frac = self.contact_binary_buf.float().mean(dim=-1)
        both_contact = self.contact_binary_buf.all(dim=-1)

        gripper_actual = self.robot.data.joint_pos[:, self.gripper_cmd_index]
        gripper_err = (self.gripper_cmd_buf - gripper_actual).clamp(min=0.0)
        squeeze_frac = (gripper_err / GRIPPER_POS_ERR_MAX).clamp(0.0, 1.0)

        # 접근 shaping 용 거리
        tcp_to_cup = (self.object_pos - self.tcp_pos).norm(dim=-1)
        finger_side = (
            radial.norm(dim=-1).mean(dim=-1) - self._cup_radius_at_grasp()
        ).clamp(min=0.0)
        return {
            "contact_frac": contact_frac,
            "both_contact": both_contact,
            "opposition": opposition,
            "squeeze_frac": squeeze_frac,
            "gripper_err": gripper_err,
            "tcp_to_cup": tcp_to_cup,
            "finger_side": finger_side,
        }

    def _cup_radius_at_grasp(self) -> float:
        """파지 높이에서의 컵 단면 반경 [m] (probe_gripper_opening 실측)."""
        return GRASP_CUP_RADIUS

    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        m = self._grip

        arm_pos = self.robot.data.joint_pos[:, self.arm_dof_indices]
        arm_vel = self.robot.data.joint_vel[:, self.arm_dof_indices]
        grip_pos = self.robot.data.joint_pos[:, self.gripper_cmd_index].unsqueeze(1)
        grip_vel = self.robot.data.joint_vel[:, self.gripper_cmd_index].unsqueeze(1)

        finger_rel_tcp = (self.finger_pos - self.tcp_pos.unsqueeze(1)).flatten(1)
        tcp_to_cup_vec = self.object_pos - self.tcp_pos
        cup_to_finger = (self.finger_pos - self.object_pos.unsqueeze(1)).flatten(1)
        finger_force = (self.finger_force_buf / CONTACT_FORCE_MAX).clamp(-1.0, 1.0).flatten(1)

        obs = torch.cat([
            arm_pos, arm_vel, grip_pos, grip_vel,
            self.tcp_pos, finger_rel_tcp, tcp_to_cup_vec, cup_to_finger, finger_force,
            (m["gripper_err"] / GRIPPER_POS_ERR_MAX).clamp(0.0, 1.0).unsqueeze(1),
            self.actions,
        ], dim=-1)
        if obs.shape[-1] != NUM_OBSERVATIONS:
            raise RuntimeError(f"actor obs {obs.shape[-1]} != {NUM_OBSERVATIONS}")

        cup_height_delta = (self.object_pos[:, 2] - self.cup_spawn_pos[:, 2]).unsqueeze(1)
        phase_ratio = (self.episode_length_buf.float() / EPISODE_STEPS).unsqueeze(1)
        finger_signed = (
            (self.finger_pos[:, :, :2] - self.object_pos[:, :2].unsqueeze(1)).norm(dim=-1)
            - self._cup_radius_at_grasp()
        )
        critic = torch.cat([
            obs,
            self.cup.data.root_lin_vel_w, self.cup.data.root_ang_vel_w, self.object_rot,
            cup_height_delta, phase_ratio, finger_signed,
        ], dim=-1)
        if critic.shape[-1] != NUM_CRITIC_OBSERVATIONS:
            raise RuntimeError(f"critic obs {critic.shape[-1]} != {NUM_CRITIC_OBSERVATIONS}")
        return {"policy": obs, "critic": critic}

    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # 중간값은 _get_dones 가 이미 갱신했다 (step 순서: dones → rewards).
        m = self._grip

        cup_height_delta = self.object_pos[:, 2] - self.cup_spawn_pos[:, 2]
        cup_xy_disp = (self.object_pos[:, :2] - self.cup_spawn_pos[:, :2]).norm(dim=-1)
        # 컵 기울기: 로컬 +z 가 월드 +z 에서 벗어난 각도
        up_local = to_torch([0.0, 0.0, 1.0], self.device).unsqueeze(0).expand(self.num_envs, -1)
        cup_up = quat_apply(self.object_rot, up_local)
        cup_tilt_deg = torch.rad2deg(torch.acos(cup_up[:, 2].clamp(-1.0, 1.0)))
        upright_quality = cup_up[:, 2].clamp(0.0, 1.0)

        action_delta_norm = compute_action_delta_norm(self.actions, self.prev_actions)
        contact_count = self.contact_binary_buf.float().sum(dim=-1)
        stab = compute_grasp_v2_stability(
            cup_lin_vel=self.cup.data.root_lin_vel_w,
            cup_ang_vel=self.cup.data.root_ang_vel_w,
            contact_delta=contact_count - self.prev_contact_count,
            action_delta_norm=action_delta_norm,
            cfg=self.cfg,
        )
        self.prev_contact_count = contact_count

        lifted = cup_height_delta >= float(self.cfg.lift_success_height)
        upright_ok = cup_tilt_deg <= float(self.cfg.success_upright_max_deg)
        enough_contact = contact_count >= int(self.cfg.success_min_contacts)
        success_now = self.lift_latched_buf & lifted & upright_ok & enough_contact & stab.stable
        self.success_buf = success_now

        # 접촉 지속률 (연속 hold 를 에피소드 길이로 정규화)
        self.contact_persistence_buf = (
            self.contact_hold_buf.float() / max(1, int(self.cfg.grasp_ready_hold_steps))
        ).clamp(0.0, 1.0)

        total, terms, gates = compute_gripper_grasp_reward_terms(
            contact_frac=m["contact_frac"],
            both_contact=m["both_contact"],
            contact_persistence_frac=self.contact_persistence_buf,
            opposition=m["opposition"],
            squeeze_frac=m["squeeze_frac"],
            tcp_to_cup_dist=m["tcp_to_cup"],
            finger_side_dist=m["finger_side"],
            cup_height_delta=cup_height_delta,
            cup_xy_displacement=cup_xy_disp,
            cup_tilt_deg=cup_tilt_deg,
            upright_quality=upright_quality,
            lift_latched=self.lift_latched_buf,
            action_delta_norm=action_delta_norm,
            success_now=success_now,
            stable=stab.stable,
            stability_quality=stab.quality,
            cfg=self.cfg,
        )

        log = self.extras.setdefault("log", {})
        for name, value in terms.items():
            log[f"reward/{name}"] = value.mean()
        for name, value in gates.items():
            log[f"gate/{name}"] = value.mean()
        log["metric/contact_frac"] = m["contact_frac"].mean()
        log["metric/opposition"] = m["opposition"].mean()
        log["metric/squeeze"] = m["squeeze_frac"].mean()
        log["metric/cup_height_delta"] = cup_height_delta.mean()
        log["metric/cup_xy_disp"] = cup_xy_disp.mean()
        log["metric/latched_rate"] = self.lift_latched_buf.float().mean()
        log["metric/lifted_rate"] = lifted.float().mean()
        log["metric/success_rate"] = success_now.float().mean()
        # keyword-only 시그니처다. palm_dims=6 이면 나머지 1D 가 그리퍼 그룹으로 잡힌다.
        log.update(action_policy_scalars(
            action=self.actions, prev_action=self.prev_actions, palm_dims=NUM_PALM_ACTION
        ))
        return total

    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # step 순서상 여기가 매 스텝 물리값을 처음 읽는 지점이다 (우측 태스크와 동일 배치).
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= EPISODE_STEPS - 1
        # 컵이 테이블 밖으로 떨어지면 조기 종료 (학습 시간 낭비 방지)
        dropped = self.object_pos[:, 2] < (float(self.cfg.table_surface_z) - 0.10)
        return dropped, time_out

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # ── 컵 스폰 ────────────────────────────────────────────────
        rand = torch.rand(n, 2, device=self.device) * 2.0 - 1.0
        cup_local = torch.stack([
            float(self.cfg.object_spawn_x_center) + rand[:, 0] * float(self.cfg.object_spawn_x_range),
            float(self.cfg.object_spawn_y_center) + rand[:, 1] * float(self.cfg.object_spawn_y_range),
            torch.full((n,), float(self.cfg.object_spawn_z), device=self.device),
        ], dim=-1)
        self.cup_spawn_pos[env_ids] = cup_local

        root_state = self.cup.data.default_root_state[env_ids].clone()
        root_state[:, :3] = cup_local + self.scene.env_origins[env_ids]
        root_state[:, 3:7] = to_torch([1.0, 0.0, 0.0, 0.0], self.device).unsqueeze(0).expand(n, -1)
        root_state[:, 7:] = 0.0
        self.cup.write_root_state_to_sim(root_state, env_ids)

        # ── 로봇: 고정 홈 ─────────────────────────────────────────
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:, self.arm_dof_indices] = self.q_home_arm.unsqueeze(0)
        joint_pos[:, self.gripper_dof_indices] = GRIPPER_OPEN_POS
        joint_pos[:, self.idle_dof_indices] = self.idle_rest_pos[:n]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # ── 액션 기준점 = 컵-정준 pregrasp ────────────────────────
        noise = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * to_torch(
            list(self.cfg.pregrasp_pos_noise), self.device
        )
        pregrasp_pos = cup_local + self.pregrasp_tcp_offset.unsqueeze(0) + noise
        self.pregrasp_palm_pose_buf[env_ids] = torch.cat([
            pregrasp_pos, self.grasp_euler.unsqueeze(0).expand(n, -1)
        ], dim=-1)
        self.palm_pose_targets[env_ids] = self.pregrasp_palm_pose_buf[env_ids]
        self.lift_palm_pose_buf[env_ids] = self.pregrasp_palm_pose_buf[env_ids]

        # ── 버퍼 초기화 ───────────────────────────────────────────
        self.fabric_q[env_ids] = self.q_home_arm.unsqueeze(0)
        self.fabric_qd[env_ids] = 0.0
        self.fabric_qdd[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.gripper_cmd_buf[env_ids] = GRIPPER_OPEN_POS
        self.contact_binary_buf[env_ids] = False
        self.contact_hold_buf[env_ids] = 0
        self.contact_persistence_buf[env_ids] = 0.0
        self.lift_latched_buf[env_ids] = False
        self.lift_start_step_buf[env_ids] = GRASP_PHASE_STEPS
        self.prev_contact_count[env_ids] = 0.0
        self.success_buf[env_ids] = False
