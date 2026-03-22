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

"""bi_pouring_v2: incremental joint position control (franka_cabinet 패턴).

v1 대비 주요 변경:
  - 제어 방식: offset-from-home → incremental joint position control
    target += dt * action_speed_scale * action  (clamp to joint limits)
  - Observation: prev_actions → arm_joint_targets (현재 누적 target)
  - 단일 파일: obs/reward/done 모두 인라인 (mdp/ 서브모듈 없음)
  - 내부 버퍼: _raw_actions, _prev_raw_actions (v1의 _actions/_prev_actions 대체)
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

from .bi_pouring_preset import (
    BEAD_SPAWN_POS_SOURCE_CUP_B,
    BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ,
    LEFT_HOLDER_FIXED_JOINT_POS,
    RIGHT_ARM_POUR_READY_POSE,
    RIGHT_HAND_GRASP_JOINT_POS,
    RIGHT_HAND_JOINT_NAMES,
)


def _safe_normalize(vec: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return vec / torch.clamp(torch.norm(vec, dim=dim, keepdim=True), min=1e-6)


class BiPouringEnv(DirectRLEnv):
    cfg: "BiPouringEnvCfg"

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.right_source_cup = RigidObject(self.cfg.right_source_cup_cfg)
        self.left_target_cup = RigidObject(self.cfg.left_target_cup_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.beads: list[RigidObject] = [
            RigidObject(self.cfg.bead_cfg) for _ in range(self.cfg.bead_count)
        ]

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["right_source_cup"] = self.right_source_cup
        self.scene.rigid_objects["left_target_cup"] = self.left_target_cup
        self.scene.rigid_objects["table"] = self.table
        for i, bead in enumerate(self.beads):
            self.scene.rigid_objects[f"bead{i}"] = bead

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def __init__(self, cfg: "BiPouringEnvCfg", render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self._setup_env_refs()

        # GUI 시각화 마커: source pour point (빨강) + target opening (파랑)
        self._vis_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/bi_pouring_markers",
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

        n = self.num_envs
        # incremental 제어: physics_dt = sim.dt * decimation = 1/120 * 2 = 1/60 s
        self._physics_dt: float = self.cfg.sim.dt * self.cfg.decimation

        # raw_actions: policy 출력 (clamp [-1, 1])
        # arm_joint_targets: 누적된 joint position target (incremental 적분)
        self._raw_actions = torch.zeros(n, self.cfg.num_actions, device=self.device)
        self._prev_raw_actions = torch.zeros_like(self._raw_actions)
        self._arm_joint_targets = self._right_arm_home.unsqueeze(0).expand(n, -1).clone()

        self._right_palm_pos_w = torch.zeros(n, 3, device=self.device)
        self._source_pour_point_w = torch.zeros(n, 3, device=self.device)
        self._target_opening_w = torch.zeros(n, 3, device=self.device)

        self._mouth_distance = torch.zeros(n, device=self.device)
        self._mouth_xy_distance = torch.zeros(n, device=self.device)
        self._mouth_z_clearance = torch.zeros(n, device=self.device)
        self._mouth_z_band_error = torch.zeros(n, device=self.device)
        self._source_up_dot_world = torch.zeros(n, device=self.device)
        self._directional_tilt_cos = torch.zeros(n, device=self.device)
        self._mouth_alignment_cos = torch.zeros(n, device=self.device)

        self._r_transport_goal = torch.zeros(n, device=self.device)
        self._r_palm_to_goal = torch.zeros(n, device=self.device)
        self._r_tilt = torch.zeros(n, device=self.device)
        self._r_directional_tilt = torch.zeros(n, device=self.device)
        self._r_height = torch.zeros(n, device=self.device)
        self._r_mouth_alignment = torch.zeros(n, device=self.device)
        self._spill_penalty = torch.zeros(n, device=self.device)
        self._collision_penalty = torch.zeros(n, device=self.device)
        self._smoothness_penalty = torch.zeros(n, device=self.device)

        self._major_spill_flag = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._invalid_state_flag = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._success_flag = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._pre_pour_ready_steps = torch.zeros(n, dtype=torch.long, device=self.device)

        k = self.cfg.bead_count
        self._bead_in_target_flags = torch.zeros(n, k, dtype=torch.bool, device=self.device)
        self._bead_in_source_flags = torch.zeros(n, k, dtype=torch.bool, device=self.device)
        self._bead_spilled_flags = torch.zeros(n, k, dtype=torch.bool, device=self.device)

    def _setup_env_refs(self) -> None:
        cfg = self.cfg

        self._right_arm_joint_ids = [self.robot.joint_names.index(n) for n in cfg.policy_arm_joint_names]
        self._right_hand_joint_ids = [self.robot.joint_names.index(n) for n in RIGHT_HAND_JOINT_NAMES]
        self._left_holder_joint_ids = [self.robot.joint_names.index(n) for n in cfg.left_holder_joint_names]

        # _apply_action에서 매 step write_joint_state_to_sim 호출용 캐시 텐서.
        # 매 step tensor 생성 없이 재사용하여 성능 확보.
        self._hand_ids_t: torch.Tensor | None = None  # lazy init (device 확인 후)
        self._left_ids_t: torch.Tensor | None = None

        if getattr(cfg, "debug_print", False):
            print("\n========== [BiPouring v2] Joint ID Mapping ==========")
            print(f"Total robot joints: {len(self.robot.joint_names)}")
            print(f"All joint names: {self.robot.joint_names}")
            print(f"\nRight arm joint IDs  : {self._right_arm_joint_ids}")
            for gid, name in zip(self._right_arm_joint_ids, cfg.policy_arm_joint_names):
                print(f"  [{gid:2d}] {name}")
            print(f"\nRight hand joint IDs : {self._right_hand_joint_ids[:5]}...({len(self._right_hand_joint_ids)} total)")
            print(f"Left holder joint IDs: {self._left_holder_joint_ids}")
            overlap_arm_hand = set(self._right_arm_joint_ids) & set(self._right_hand_joint_ids)
            overlap_arm_left = set(self._right_arm_joint_ids) & set(self._left_holder_joint_ids)
            print(f"\nOverlap arm vs hand: {overlap_arm_hand}  (should be empty!)")
            print(f"Overlap arm vs left: {overlap_arm_left}  (should be empty!)")
            print("\n========== [BiPouring v2] Actuator Coverage ==========")
            for act_name, actuator in self.robot.actuators.items():
                act_joints = list(actuator.joint_names)
                print(f"  [{act_name}] ({len(act_joints)} joints): {act_joints}")
            all_actuated = set()
            for actuator in self.robot.actuators.values():
                all_actuated.update(actuator.joint_names)
            wrist_names = cfg.policy_arm_joint_names[4:]  # joint5,6,7
            for wn in wrist_names:
                status = "OK" if wn in all_actuated else "*** MISSING from actuator! ***"
                print(f"  {wn}: {status}")
            print("===================================================\n", flush=True)

        self._right_hand_grasp = torch.tensor(
            [RIGHT_HAND_GRASP_JOINT_POS[n] for n in RIGHT_HAND_JOINT_NAMES],
            dtype=torch.float32,
            device=self.device,
        )
        self._left_holder_home = torch.tensor(
            [LEFT_HOLDER_FIXED_JOINT_POS[n] for n in cfg.left_holder_joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self._right_arm_home = torch.tensor(RIGHT_ARM_POUR_READY_POSE, dtype=torch.float32, device=self.device)

        arm_hand_start = torch.cat([self._right_arm_home, self._right_hand_grasp], dim=0)
        self.robot_start_joint_pos = arm_hand_start.unsqueeze(0).repeat(self.num_envs, 1).contiguous()

        self._arm_joint_lower_limits = torch.tensor(cfg.arm_joint_mins, dtype=torch.float32, device=self.device)
        self._arm_joint_upper_limits = torch.tensor(cfg.arm_joint_maxs, dtype=torch.float32, device=self.device)
        self._target_tilt_cos = math.cos(math.radians(cfg.target_transport_tilt_deg))

        attach_pos_r = torch.tensor(cfg.right_source_cup_attach_pos_b, dtype=torch.float32, device=self.device)
        attach_pos_l = torch.tensor(cfg.left_target_cup_attach_pos_b, dtype=torch.float32, device=self.device)
        self._right_source_cup_attach_quat_b = torch.tensor(
            cfg.right_source_cup_attach_quat_wxyz_b, dtype=torch.float32, device=self.device
        )
        self._left_target_cup_attach_quat_b = torch.tensor(
            cfg.left_target_cup_attach_quat_wxyz_b, dtype=torch.float32, device=self.device
        )

        self._right_source_cup_body_id, self._right_source_cup_attach_pos_b = self._resolve_attachment_body(
            cfg.right_source_cup_attach_frame_name, attach_pos_r
        )
        self._left_target_cup_body_id, self._left_target_cup_attach_pos_b = self._resolve_attachment_body(
            cfg.left_target_cup_attach_frame_name, attach_pos_l
        )
        self._right_palm_body_id, _ = self._resolve_attachment_body(
            "palm_ee", torch.zeros(3, dtype=torch.float32, device=self.device)
        )

        self._source_cup_pour_point_pos_b = torch.tensor(
            cfg.source_cup_pour_point_pos_b, dtype=torch.float32, device=self.device
        )
        self._target_cup_opening_pos_b = torch.tensor(
            cfg.target_cup_opening_pos_b, dtype=torch.float32, device=self.device
        )
        self._source_cup_pour_axis_b = torch.tensor(
            cfg.source_cup_pour_axis_b, dtype=torch.float32, device=self.device
        )
        self._source_cup_up_axis_b = torch.tensor(
            cfg.source_cup_up_axis_b, dtype=torch.float32, device=self.device
        )
        self._target_cup_up_axis_b = torch.tensor(
            cfg.target_cup_up_axis_b, dtype=torch.float32, device=self.device
        )
        self._world_up_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).unsqueeze(0)

        self._bead_spawn_pos_source_cup_b = torch.tensor(
            getattr(cfg, "bead_spawn_pos_source_cup_b", BEAD_SPAWN_POS_SOURCE_CUP_B),
            dtype=torch.float32,
            device=self.device,
        )
        self._bead_spawn_quat_source_cup = torch.tensor(
            getattr(cfg, "bead_spawn_quat_source_cup_wxyz", BEAD_SPAWN_QUAT_SOURCE_CUP_WXYZ),
            dtype=torch.float32,
            device=self.device,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_raw_actions.copy_(self._raw_actions)
        self._raw_actions.copy_(torch.clamp(actions, -1.0, 1.0))

        # incremental: target += dt * speed_scale * action
        # max Δ/step = action_speed_scale * physics_dt ≈ 3.0 * (1/60) = 0.05 rad
        delta = self.cfg.action_speed_scale * self._physics_dt * self._raw_actions
        self._arm_joint_targets = torch.clamp(
            self._arm_joint_targets + delta,
            self._arm_joint_lower_limits,
            self._arm_joint_upper_limits,
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._arm_joint_targets, joint_ids=self._right_arm_joint_ids)

        # Hand / left-holder 조인트: PD 대신 매 step write_joint_state_to_sim으로 직접 teleport.
        # 이유:
        #   - rj_dg_1_2 (thumb curl) = -1.57 rad 가 joint limit 경계에 위치 →
        #     PhysX limit constraint가 첫 physics step에서 impulsive force 생성 → vel 100+ rad/s 폭발.
        #   - kinematic cup과 finger 사이 penetration이 잔류할 경우 추가 충격 방지.
        #   - left holder도 고정이므로 동일 방식 적용.
        # 결과: hand/left 조인트는 항상 vel=0, pos=grasp/home 으로 유지 (완전 kinematic).
        if self._hand_ids_t is None:
            self._hand_ids_t = torch.as_tensor(self._right_hand_joint_ids, dtype=torch.long, device=self.device)
            self._left_ids_t = torch.as_tensor(self._left_holder_joint_ids, dtype=torch.long, device=self.device)

        n = self.num_envs
        grasp_pos = self._right_hand_grasp.unsqueeze(0).expand(n, -1).clone()
        left_pos = self._left_holder_home.unsqueeze(0).expand(n, -1).clone()
        zero_hand = torch.zeros_like(grasp_pos)
        zero_left = torch.zeros_like(left_pos)
        self.robot.write_joint_state_to_sim(grasp_pos, zero_hand, joint_ids=self._hand_ids_t)
        self.robot.write_joint_state_to_sim(left_pos, zero_left, joint_ids=self._left_ids_t)

        # PD target도 동기화 (physics buffer의 PD torque가 0이 되도록 pos=target=error=0).
        self.robot.set_joint_position_target(grasp_pos, joint_ids=self._right_hand_joint_ids)
        self.robot.set_joint_position_target(left_pos, joint_ids=self._left_holder_joint_ids)

        # Cup 위치 업데이트 후 bead를 source cup 프레임에 teleport.
        # contact_offset=-0.1로 cup-bead 충돌이 꺼져 있으므로
        # bead가 cup을 통과해 arm 링크에 충돌하는 것을 막기 위해 kinematic으로 고정.
        right_pose = self._compute_attached_root_pose(
            self._right_source_cup_body_id,
            self._right_source_cup_attach_pos_b,
            self._right_source_cup_attach_quat_b,
        )
        left_pose = self._compute_attached_root_pose(
            self._left_target_cup_body_id,
            self._left_target_cup_attach_pos_b,
            self._left_target_cup_attach_quat_b,
        )
        zero_cup_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self.right_source_cup.write_root_pose_to_sim(right_pose)
        self.right_source_cup.write_root_velocity_to_sim(zero_cup_vel)
        self.left_target_cup.write_root_pose_to_sim(left_pose)
        self.left_target_cup.write_root_velocity_to_sim(zero_cup_vel)

        bead_pose = self._sample_bead_pose_inside_source_cup(right_pose)
        zero_bead_vel = torch.zeros(self.num_envs, 6, device=self.device)
        for bead in self.beads:
            bead.write_root_pose_to_sim(bead_pose)
            bead.write_root_velocity_to_sim(zero_bead_vel)

    def _debug_log(self) -> None:
        """env 0의 핵심 수치를 stdout에 출력. cfg.debug_print=True 일 때만 동작."""
        if not getattr(self.cfg, "debug_print", False):
            return

        e = 0  # 모니터링 대상 env index
        step = int(self.episode_length_buf[e].item())

        # step=0~9 는 매번, 이후는 10 step 마다만 출력
        if step >= 10 and step % 10 != 0:
            return

        def _f(t: torch.Tensor) -> str:
            return "[" + " ".join(f"{v:+.3f}" for v in t.cpu().tolist()) + "]"

        joint_pos = self.robot.data.joint_pos[e, self._right_arm_joint_ids]
        joint_vel = self.robot.data.joint_vel[e, self._right_arm_joint_ids]
        target    = self._arm_joint_targets[e]
        # physics buffer에 실제로 설정된 arm target (actuator가 이 값으로 PD 계산)
        arm_tgt_buf = self.robot.data.joint_pos_target[e, self._right_arm_joint_ids]
        raw_action  = self._raw_actions[e]
        home        = self._right_arm_home
        delta_pos   = joint_pos - home   # 현재 위치가 home에서 벗어난 정도
        delta_tgt   = target    - home   # 목표가 home에서 벗어난 정도
        pd_error    = target    - joint_pos  # PD 제어 오차

        # hand joints (first 5 of 20) — rj_dg 마스킹 확인용
        hand_ids_5 = self._right_hand_joint_ids[:5]
        hand_pos5 = self.robot.data.joint_pos[e, hand_ids_5]
        hand_vel5 = self.robot.data.joint_vel[e, hand_ids_5]
        hand_tgt5 = self.robot.data.joint_pos_target[e, hand_ids_5]

        bead_pos_w = self.beads[0].data.root_pos_w[e]
        bead_env   = bead_pos_w - self.scene.env_origins[e]

        cfg = self.cfg
        mouth_xy = float(self._mouth_xy_distance[e].item())
        mouth_z  = float(self._mouth_z_clearance[e].item())
        mouth_d  = float(self._mouth_distance[e].item())
        bead_z   = float(bead_env[2].item())

        # invalid_state 원인 분해
        flag_xy        = mouth_xy >= cfg.invalid_cup_xy_threshold
        flag_z         = abs(mouth_z) >= cfg.invalid_cup_z_threshold
        flag_bead_drop = (
            not bool(self._bead_in_target_flags[e, 0].item())
            and not bool(self._bead_in_source_flags[e, 0].item())
            and (bead_z <= cfg.invalid_bead_drop_z_threshold or bead_z <= cfg.invalid_bead_floor_z_threshold)
        )
        flag_bead_out = float(torch.norm(bead_env[:2]).item()) >= cfg.invalid_bead_xy_threshold

        # wrist PD error 경고 (|error| > 0.3 rad)
        wrist_warn = any(abs(float(pd_error[i].item())) > 0.3 for i in range(4, 7))
        # reward 분해값 (env 0 기준)
        r_transport = float(self._r_transport_goal[e].item()) * cfg.reward_transport_goal_weight
        r_palm     = float(self._r_palm_to_goal[e].item())    * cfg.reward_palm_to_goal_weight
        r_tilt     = float(self._r_tilt[e].item())            * cfg.reward_tilt_weight
        r_dtilt    = float(self._r_directional_tilt[e].item())* cfg.reward_directional_tilt_weight
        r_height   = float(self._r_height[e].item())          * cfg.reward_height_weight
        r_align    = float(self._r_mouth_alignment[e].item()) * cfg.reward_mouth_alignment_weight
        p_spill    = float(self._spill_penalty[e].item())     * cfg.penalty_spill_weight
        p_coll     = float(self._collision_penalty[e].item()) * cfg.penalty_collision_weight
        p_smooth   = float(self._smoothness_penalty[e].item())* cfg.penalty_action_smoothness_weight
        r_total    = r_transport + r_palm + r_tilt + r_dtilt + r_height + r_align - p_spill - p_coll - p_smooth

        # palm → target 벡터 (어느 방향으로 당기는지)
        palm_vec = self._target_opening_w[e] - self._right_palm_pos_w[e]

        hand_max_vel = float(self.robot.data.joint_vel[e, self._right_hand_joint_ids].abs().max().item())
        print(
            f"\n[BiPouring v2 DBG step={step:3d}]"
            f"\n  raw_action  : {_f(raw_action)}"
            f"\n  target(incr): {_f(target)}"
            f"\n  Δtgt(tgt-home): {_f(delta_tgt)}"
            f"\n  arm_tgt_buf : {_f(arm_tgt_buf)}  ← physics buffer 실제값"
            f"\n  joint_pos   : {_f(joint_pos)}"
            f"\n  joint_vel   : {_f(joint_vel)}"
            f"\n  pd_error (tgt-pos): {_f(pd_error)}"
            f"{'  *** WRIST LARGE ERROR ***' if wrist_warn else ''}"
            f"\n  Δpos(pos-home): {_f(delta_pos)}"
            f"\n  hand_pos(rj1-5_1): {_f(hand_pos5)}"
            f"\n  hand_vel(rj1-5_1): {_f(hand_vel5)}  max_abs={hand_max_vel:.2f} rad/s"
            f"\n  hand_tgt(rj1-5_1): {_f(hand_tgt5)}"
            f"\n  mouth dist={mouth_d:.3f}m  xy={mouth_xy:.3f}m  z_clear={mouth_z:.3f}m"
            f"\n  palm→target vec: [{palm_vec[0]:+.3f} {palm_vec[1]:+.3f} {palm_vec[2]:+.3f}]"
            f"\n  bead env-local xyz: [{bead_env[0]:.3f} {bead_env[1]:.3f} {bead_z:.3f}]"
            f"  in_source={bool(self._bead_in_source_flags[e, 0].item())}"
            f"  in_target={bool(self._bead_in_target_flags[e, 0].item())}"
            f"\n  ── reward breakdown ──"
            f"\n  transport_goal={r_transport:+.4f}  palm_to_goal={r_palm:+.4f}  tilt={r_tilt:+.4f}"
            f"\n  dir_tilt={r_dtilt:+.4f}  height={r_height:+.4f}  alignment={r_align:+.4f}"
            f"\n  -spill={p_spill:+.4f}  -collision={p_coll:+.4f}  -smooth={p_smooth:+.4f}"
            f"\n  TOTAL={r_total:+.4f}"
            f"\n  flags: invalid={bool(self._invalid_state_flag[e].item())}"
            f"  spill={bool(self._major_spill_flag[e].item())}"
            f"  success={bool(self._success_flag[e].item())}"
            f"\n  invalid cause: cup_xy={flag_xy}  cup_z={flag_z}"
            f"  bead_drop={flag_bead_drop}  bead_out={flag_bead_out}",
            flush=True,
        )

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        self._debug_log()

        obs = torch.cat(
            [
                self.robot.data.joint_pos[:, self._right_arm_joint_ids],          # 7D
                self.robot.data.joint_vel[:, self._right_arm_joint_ids],           # 7D
                self._right_palm_pos_w - self.scene.env_origins,                   # 3D
                self._target_opening_w - self._source_pour_point_w,               # 3D
                self.right_source_cup.data.root_quat_w,                           # 4D
                self.left_target_cup.data.root_quat_w,                            # 4D
                torch.stack(
                    [
                        self._mouth_distance,
                        self._mouth_xy_distance,
                        self._mouth_z_clearance,
                        self._source_up_dot_world,
                        self._directional_tilt_cos,
                    ],
                    dim=-1,
                ),                                                                 # 5D
                self._arm_joint_targets,                                           # 7D (v1: prev_actions)
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        reward = (
            cfg.reward_transport_goal_weight * self._r_transport_goal
            + cfg.reward_palm_to_goal_weight * self._r_palm_to_goal
            + cfg.reward_tilt_weight * self._r_tilt
            + cfg.reward_directional_tilt_weight * self._r_directional_tilt
            + cfg.reward_height_weight * self._r_height
            + cfg.reward_mouth_alignment_weight * self._r_mouth_alignment
            - cfg.penalty_spill_weight * self._spill_penalty
            - cfg.penalty_collision_weight * self._collision_penalty
            - cfg.penalty_action_smoothness_weight * self._smoothness_penalty
        )
        self.extras["mouth_distance"] = self._mouth_distance.mean()
        self.extras["mouth_xy_distance"] = self._mouth_xy_distance.mean()
        self.extras["mouth_z_clearance"] = self._mouth_z_clearance.mean()
        self.extras["mouth_z_band_error"] = self._mouth_z_band_error.mean()
        self.extras["source_up_dot_world"] = self._source_up_dot_world.mean()
        self.extras["directional_tilt_cos"] = self._directional_tilt_cos.mean()
        self.extras["mouth_alignment_cos"] = self._mouth_alignment_cos.mean()
        self.extras["reward_transport_goal"] = self._r_transport_goal.mean()
        self.extras["reward_palm_to_goal"] = self._r_palm_to_goal.mean()
        self.extras["reward_tilt"] = self._r_tilt.mean()
        self.extras["reward_directional_tilt"] = self._r_directional_tilt.mean()
        self.extras["reward_height"] = self._r_height.mean()
        self.extras["reward_mouth_alignment"] = self._r_mouth_alignment.mean()
        self.extras["penalty_spill"] = self._spill_penalty.mean()
        self.extras["penalty_collision"] = self._collision_penalty.mean()
        self.extras["penalty_action_smoothness"] = self._smoothness_penalty.mean()
        self.extras["pre_pour_ready_steps"] = self._pre_pour_ready_steps.float().mean()
        self.extras["success_rate"] = self._success_flag.float().mean()
        self.extras["invalid_rate"] = self._invalid_state_flag.float().mean()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self._major_spill_flag | self._invalid_state_flag | self._success_flag
        return terminated, time_out

    def _compute_intermediate_values(self) -> None:
        cfg = self.cfg
        n = self.num_envs

        right_cup_pos_w = self.right_source_cup.data.root_pos_w
        right_cup_quat_w = self.right_source_cup.data.root_quat_w
        left_cup_pos_w = self.left_target_cup.data.root_pos_w
        left_cup_quat_w = self.left_target_cup.data.root_quat_w
        self._right_palm_pos_w = self.robot.data.body_pos_w[:, self._right_palm_body_id]

        self._source_pour_point_w = right_cup_pos_w + quat_apply(
            right_cup_quat_w, self._source_cup_pour_point_pos_b.unsqueeze(0).expand(n, -1)
        )
        self._target_opening_w = left_cup_pos_w + quat_apply(
            left_cup_quat_w, self._target_cup_opening_pos_b.unsqueeze(0).expand(n, -1)
        )
        source_up_w = quat_apply(right_cup_quat_w, self._source_cup_up_axis_b.unsqueeze(0).expand(n, -1))
        source_pour_axis_w = quat_apply(
            right_cup_quat_w, self._source_cup_pour_axis_b.unsqueeze(0).expand(n, -1)
        )

        mouth_delta = self._target_opening_w - self._source_pour_point_w
        self._mouth_distance = torch.norm(mouth_delta, dim=-1)
        self._mouth_xy_distance = torch.norm(mouth_delta[:, :2], dim=-1)
        self._mouth_z_clearance = self._source_pour_point_w[:, 2] - self._target_opening_w[:, 2]
        self._source_up_dot_world = torch.sum(source_up_w * self._world_up_axis.expand(n, -1), dim=-1).clamp(-1.0, 1.0)

        target_xy_vec = torch.cat([mouth_delta[:, :2], torch.zeros(n, 1, device=self.device)], dim=-1)
        target_xy_dir = _safe_normalize(target_xy_vec)
        source_pour_dir = _safe_normalize(mouth_delta)
        ref_up = self._target_tilt_cos * self._world_up_axis.expand(n, -1) - math.sin(
            math.radians(cfg.target_transport_tilt_deg)
        ) * target_xy_dir
        ref_up = _safe_normalize(ref_up)
        self._directional_tilt_cos = torch.sum(source_up_w * ref_up, dim=-1).clamp(-1.0, 1.0)
        self._mouth_alignment_cos = torch.sum(source_pour_axis_w * source_pour_dir, dim=-1).clamp(-1.0, 1.0)

        z_low = cfg.target_mouth_z_clearance_min
        z_high = cfg.target_mouth_z_clearance_max
        self._mouth_z_band_error = torch.clamp(z_low - self._mouth_z_clearance, min=0.0) + torch.clamp(
            self._mouth_z_clearance - z_high, min=0.0
        )

        palm_to_goal = self._target_opening_w - self._right_palm_pos_w
        self._r_transport_goal = torch.exp(-cfg.transport_goal_sharpness * self._mouth_distance)
        self._r_palm_to_goal = torch.exp(-cfg.transport_palm_goal_sharpness * torch.norm(palm_to_goal, dim=-1))
        self._r_tilt = torch.exp(
            -cfg.transport_tilt_sharpness * torch.abs(self._source_up_dot_world - self._target_tilt_cos)
        )
        self._r_directional_tilt = 0.5 * (1.0 + self._directional_tilt_cos)
        self._r_height = torch.exp(-cfg.transport_height_sharpness * self._mouth_z_band_error)
        self._r_mouth_alignment = 0.5 * (1.0 + self._mouth_alignment_cos)

        all_bead_pos_w = torch.stack([b.data.root_pos_w for b in self.beads], dim=1)
        bead_pos_in_target_list = []
        bead_pos_in_source_list = []
        for bead in self.beads:
            bead_pos_w = bead.data.root_pos_w
            bead_quat_w = bead.data.root_quat_w
            pos_in_t, _ = subtract_frame_transforms(left_cup_pos_w, left_cup_quat_w, bead_pos_w, bead_quat_w)
            pos_in_s, _ = subtract_frame_transforms(right_cup_pos_w, right_cup_quat_w, bead_pos_w, bead_quat_w)
            bead_pos_in_target_list.append(pos_in_t)
            bead_pos_in_source_list.append(pos_in_s)
        bead_pos_in_target = torch.stack(bead_pos_in_target_list, dim=1)
        bead_pos_in_source = torch.stack(bead_pos_in_source_list, dim=1)

        bead_target_xy = torch.norm(bead_pos_in_target[..., :2], dim=-1)
        bead_source_xy = torch.norm(bead_pos_in_source[..., :2], dim=-1)
        self._bead_in_target_flags = (
            (bead_target_xy <= cfg.target_inner_radius)
            & (bead_pos_in_target[..., 2] >= cfg.target_inside_z_min)
            & (bead_pos_in_target[..., 2] <= cfg.target_inside_z_max)
        )
        self._bead_in_source_flags = (
            (bead_source_xy <= cfg.source_inner_radius)
            & (bead_pos_in_source[..., 2] >= cfg.source_inside_z_min)
            & (bead_pos_in_source[..., 2] <= cfg.source_inside_z_max)
        )
        self._spill_penalty = self._compute_bead_spill(
            all_bead_pos_w,
            self._target_opening_w,
            self._source_pour_point_w,
            cfg,
        )

        right_ee_pos_w = self.robot.data.body_pos_w[:, self._right_source_cup_body_id]
        left_ee_pos_w = self.robot.data.body_pos_w[:, self._left_target_cup_body_id]
        rim_scrape = self._proximity_penalty(self._mouth_xy_distance, cfg.rim_clearance_threshold) * torch.clamp(
            (cfg.success_z_clearance_min - self._mouth_z_clearance) / max(abs(cfg.success_z_clearance_min) + 1e-6, 1e-6),
            min=0.0,
            max=1.0,
        )
        ee_pen = self._proximity_penalty(torch.norm(right_ee_pos_w - left_ee_pos_w, dim=-1), cfg.ee_clearance_threshold)
        cross_pen = torch.maximum(
            self._proximity_penalty(torch.norm(right_cup_pos_w - left_ee_pos_w, dim=-1), cfg.cup_to_opposite_ee_clearance_threshold),
            self._proximity_penalty(torch.norm(left_cup_pos_w - right_ee_pos_w, dim=-1), cfg.cup_to_opposite_ee_clearance_threshold),
        )
        self._collision_penalty = torch.maximum(rim_scrape, torch.maximum(ee_pen, cross_pen))

        # smoothness_penalty: raw_actions 기반 jerk 페널티
        self._smoothness_penalty = torch.mean(torch.square(self._raw_actions - self._prev_raw_actions), dim=-1)

        ready_mask = (
            (self._mouth_xy_distance <= cfg.success_mouth_xy_threshold)
            & (self._mouth_distance <= cfg.success_mouth_dist_threshold)
            & (self._mouth_z_clearance >= cfg.target_mouth_z_clearance_min)
            & (self._mouth_z_clearance <= cfg.target_mouth_z_clearance_max)
            & (torch.abs(self._source_up_dot_world - self._target_tilt_cos) <= cfg.success_tilt_cos_tolerance)
            & (self._directional_tilt_cos >= cfg.success_directional_tilt_cos)
            & (self._mouth_alignment_cos >= cfg.success_alignment_cos)
        )
        self._pre_pour_ready_steps = torch.where(
            ready_mask,
            self._pre_pour_ready_steps + 1,
            torch.zeros_like(self._pre_pour_ready_steps),
        )
        self._success_flag = self._pre_pour_ready_steps >= cfg.success_hold_steps

        bead_pos_env = all_bead_pos_w - self.scene.env_origins.unsqueeze(1)
        bead_dropped = (
            (~self._bead_in_target_flags)
            & (~self._bead_in_source_flags)
            & (
                (bead_pos_env[..., 2] <= cfg.invalid_bead_drop_z_threshold)
                | (bead_pos_env[..., 2] <= cfg.invalid_bead_floor_z_threshold)
            )
        ).any(dim=-1)
        bead_out = (torch.norm(bead_pos_env[..., :2], dim=-1) >= cfg.invalid_bead_xy_threshold).any(dim=-1)
        cup_z = torch.abs(self._mouth_z_clearance)
        self._invalid_state_flag = (
            (self._mouth_xy_distance >= cfg.invalid_cup_xy_threshold)
            | (cup_z >= cfg.invalid_cup_z_threshold)
            | bead_dropped
            | bead_out
            | torch.isnan(self._mouth_distance)
        )

        # GUI 시각화: 빨강 = source pour point, 파랑 = target opening
        _all_pts = torch.cat([self._source_pour_point_w, self._target_opening_w], dim=0)  # (2n, 3)
        _marker_idx = torch.zeros(2 * n, dtype=torch.long, device=self.device)
        _marker_idx[n:] = 1  # 뒤쪽 n개 = target opening (파랑)
        self._vis_markers.visualize(translations=_all_pts, marker_indices=_marker_idx)

    def _compute_bead_spill(
        self,
        all_bead_pos_w: torch.Tensor,
        target_opening_w: torch.Tensor,
        source_pour_point_w: torch.Tensor,
        cfg,
    ) -> torch.Tensor:
        bead_below_target = all_bead_pos_w[..., 2] <= (target_opening_w.unsqueeze(1)[..., 2] + cfg.major_spill_z_margin)
        bead_xy_to_target = torch.norm((all_bead_pos_w - target_opening_w.unsqueeze(1))[..., :2], dim=-1)
        bead_xy_to_source = torch.norm((all_bead_pos_w - source_pour_point_w.unsqueeze(1))[..., :2], dim=-1)
        major_spill_per_bead = (
            (~self._bead_in_target_flags)
            & (~self._bead_in_source_flags)
            & bead_below_target
            & (bead_xy_to_target >= cfg.major_spill_xy_radius)
            & (bead_xy_to_source >= cfg.major_spill_xy_radius)
        )
        self._bead_spilled_flags = major_spill_per_bead | (
            (all_bead_pos_w[..., 2] <= cfg.bead_spill_z_threshold)
            & (~self._bead_in_target_flags)
            & (~self._bead_in_source_flags)
        )
        self._major_spill_flag = major_spill_per_bead.any(dim=-1)
        return self._bead_spilled_flags.sum(dim=-1).float() / float(cfg.bead_count)

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        num_reset = len(env_ids)
        full_pos = self.robot.data.default_joint_pos[env_ids].clone()
        full_vel = torch.zeros(num_reset, self.robot.num_joints, device=self.device)

        full_pos[:, self._right_arm_joint_ids] = self._right_arm_home.unsqueeze(0).expand(num_reset, -1)
        # cup collision_props(contact_offset=-0.1)로 cup-robot collision 비활성화됐으므로
        # 처음부터 grasp 위치로 초기화해도 penetration shock 없음.
        full_pos[:, self._right_hand_joint_ids] = self._right_hand_grasp.unsqueeze(0).expand(num_reset, -1)
        full_pos[:, self._left_holder_joint_ids] = self._left_holder_home.unsqueeze(0).expand(num_reset, -1)
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        # physics target buffer도 명시적으로 초기화.
        _env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        _home_t = self._right_arm_home.unsqueeze(0).expand(num_reset, -1).clone()
        _grasp_t = self._right_hand_grasp.unsqueeze(0).expand(num_reset, -1).clone()
        _left_t = self._left_holder_home.unsqueeze(0).expand(num_reset, -1).clone()
        self.robot.set_joint_position_target(_home_t, joint_ids=self._right_arm_joint_ids, env_ids=_env_ids_t)
        self.robot.set_joint_position_target(_grasp_t, joint_ids=self._right_hand_joint_ids, env_ids=_env_ids_t)
        self.robot.set_joint_position_target(_left_t, joint_ids=self._left_holder_joint_ids, env_ids=_env_ids_t)

        right_pose = self._compute_attached_root_pose(
            self._right_source_cup_body_id,
            self._right_source_cup_attach_pos_b,
            self._right_source_cup_attach_quat_b,
            env_ids=env_ids,
        )
        left_pose = self._compute_attached_root_pose(
            self._left_target_cup_body_id,
            self._left_target_cup_attach_pos_b,
            self._left_target_cup_attach_quat_b,
            env_ids=env_ids,
        )
        zero_vel = torch.zeros(num_reset, 6, device=self.device)
        self.right_source_cup.write_root_pose_to_sim(right_pose, env_ids=env_ids)
        self.right_source_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        self.left_target_cup.write_root_pose_to_sim(left_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        for bead in self.beads:
            jitter = torch.zeros(num_reset, 3, device=self.device)
            if self.cfg.bead_count > 1:
                jitter[:, :2] = (torch.rand(num_reset, 2, device=self.device) - 0.5) * 2.0 * self.cfg.bead_spawn_jitter_xy
            bead_pose = self._sample_bead_pose_inside_source_cup(right_pose, offset=jitter)
            bead.write_root_pose_to_sim(bead_pose, env_ids=env_ids)
            bead.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        # incremental control 버퍼 리셋: home으로 복원
        self._arm_joint_targets[env_ids] = self._right_arm_home.unsqueeze(0).expand(len(env_ids), -1)
        self._raw_actions[env_ids] = 0.0
        self._prev_raw_actions[env_ids] = 0.0

        self._spill_penalty[env_ids] = 0.0
        self._collision_penalty[env_ids] = 0.0
        self._smoothness_penalty[env_ids] = 0.0
        self._major_spill_flag[env_ids] = False
        self._invalid_state_flag[env_ids] = False
        self._success_flag[env_ids] = False
        self._pre_pour_ready_steps[env_ids] = 0
        self._bead_in_target_flags[env_ids] = False
        self._bead_in_source_flags[env_ids] = False
        self._bead_spilled_flags[env_ids] = False

    def _update_attached_cups_from_ee(self, env_ids: Sequence[int] | None = None) -> None:
        right_pose = self._compute_attached_root_pose(
            self._right_source_cup_body_id,
            self._right_source_cup_attach_pos_b,
            self._right_source_cup_attach_quat_b,
            env_ids=env_ids,
        )
        left_pose = self._compute_attached_root_pose(
            self._left_target_cup_body_id,
            self._left_target_cup_attach_pos_b,
            self._left_target_cup_attach_quat_b,
            env_ids=env_ids,
        )
        zero_vel = torch.zeros(right_pose.shape[0], 6, device=self.device)
        self.right_source_cup.write_root_pose_to_sim(right_pose, env_ids=env_ids)
        self.right_source_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        self.left_target_cup.write_root_pose_to_sim(left_pose, env_ids=env_ids)
        self.left_target_cup.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

    def _resolve_attachment_body(self, requested_body_name: str, attach_pos_b: torch.Tensor) -> tuple[int, torch.Tensor]:
        body_names = self.robot.data.body_names
        alias_offsets: dict[str, list[tuple[str, tuple[float, float, float]]]] = {
            "rl_dg_ee": [("rl_dg_ee", (0.0, 0.0, 0.0)), ("palm_ee", (0.0, 0.0, 0.0)), ("rl_dg_palm", (0.028, 0.0, 0.04))],
            "palm_ee": [("palm_ee", (0.0, 0.0, 0.0)), ("rl_dg_ee", (0.0, 0.0, 0.0)), ("rl_dg_palm", (0.028, 0.0, 0.04))],
            "ll_dg_ee": [("ll_dg_ee", (0.0, 0.0, 0.0)), ("openarm_left_hand_tcp", (0.0, 0.0, -0.08)), ("openarm_left_hand", (0.0, 0.0, 0.0))],
            "openarm_left_hand": [("openarm_left_hand", (0.0, 0.0, 0.0)), ("openarm_left_hand_tcp", (0.0, 0.0, -0.08)), ("ll_dg_ee", (0.0, 0.0, -0.08))],
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

    def _sample_bead_pose_inside_source_cup(
        self,
        source_cup_pose: torch.Tensor,
        offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source_cup_pos_w = source_cup_pose[:, :3]
        source_cup_quat_w = source_cup_pose[:, 3:7]
        spawn_offset = self._bead_spawn_pos_source_cup_b.unsqueeze(0).expand_as(source_cup_pos_w)
        if offset is not None:
            spawn_offset = spawn_offset + offset
        bead_pos_w = source_cup_pos_w + quat_apply(source_cup_quat_w, spawn_offset)
        bead_quat_w = quat_mul(
            source_cup_quat_w,
            self._bead_spawn_quat_source_cup.unsqueeze(0).expand(source_cup_quat_w.shape[0], -1),
        )
        return torch.cat([bead_pos_w, bead_quat_w], dim=-1)

    @staticmethod
    def _proximity_penalty(distance: torch.Tensor, threshold: float) -> torch.Tensor:
        return torch.clamp((threshold - distance) / max(threshold, 1e-6), min=0.0, max=1.0)
