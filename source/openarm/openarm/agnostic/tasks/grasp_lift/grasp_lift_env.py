"""robot-agnostic grasp-lift 환경 (direct).

로봇 종속 정보는 전부 RobotProfile 이 공급한다 — 이 파일에 조인트/바디 이름 하드코딩 금지.

제어 스택 (사용자 결정, 2026-08-20):
  팔  = IsaacLab DifferentialIKController(dls, relative pose) — PhysX Jacobian 만 사용,
        사전 자산 0 (Fabrics 폐기: 로봇당 4종 수제 자산 + droop/오픈루프 결함).
  손  = relative joint position (dexsuite 방식) — 동결 게이트·커플링·PCA·latch·
        스크립트 램프 전부 없음. thumb 외전 포함 전 관절이 정책 제어 대상.

커리큘럼: per-env 난이도(0~10, 성공 ±1) → 물체 반중력 보상력(유효 중력 0→g) +
스폰 범위 보간. lift 보상 항이 필요 없는 이유가 이것이다(dexsuite 검증).
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_inv,
    skew_symmetric_matrix,
    subtract_frame_transforms,
)

from .grasp_lift_env_cfg import GraspLiftEnvCfg
from .rewards import compute_grasp_lift_rewards
from .robot_profiles import PROFILES

_GRAVITY = 9.81


class GraspLiftEnv(DirectRLEnv):
    cfg: GraspLiftEnvCfg

    def __init__(self, cfg: GraspLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        p = PROFILES[self.cfg.profile_name]
        self.profile = p

        # ---- 조인트/바디 해석 (fail-loud: 프로필 선언 수와 대조) -----------------
        self.arm_ids, arm_names = self.robot.find_joints(p.arm_joint_regex)
        self.hand_ids, hand_names = self.robot.find_joints(p.hand_joint_regex)
        if len(self.arm_ids) != p.num_arm_joints or len(self.hand_ids) != p.num_hand_joints:
            raise RuntimeError(
                f"[{p.name}] 프로필 조인트 수 불일치: arm {len(self.arm_ids)}!={p.num_arm_joints} "
                f"({arm_names}), hand {len(self.hand_ids)}!={p.num_hand_joints} ({hand_names})"
            )
        self._arm_ids_t = torch.tensor(self.arm_ids, device=self.device, dtype=torch.long)
        self._hand_ids_t = torch.tensor(self.hand_ids, device=self.device, dtype=torch.long)

        palm_ids, _ = self.robot.find_bodies(p.palm_body)
        if len(palm_ids) != 1:
            raise RuntimeError(f"[{p.name}] palm_body '{p.palm_body}' 해석 실패: {palm_ids}")
        self.palm_idx = palm_ids[0]
        self.tip_ids = []
        for n in p.fingertip_bodies:
            ids, _ = self.robot.find_bodies(n)
            if len(ids) != 1:
                raise RuntimeError(f"[{p.name}] fingertip body '{n}' 해석 실패: {ids}")
            self.tip_ids.append(ids[0])
        self._tip_ids_t = torch.tensor(self.tip_ids, device=self.device, dtype=torch.long)

        # fixed-base articulation: jacobian body 인덱스는 base 제외라 -1
        if self.robot.is_fixed_base:
            self._jacobi_body_idx = self.palm_idx - 1
        else:
            self._jacobi_body_idx = self.palm_idx

        # ---- diff IK -------------------------------------------------------------
        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            num_envs=self.num_envs, device=self.device,
        )
        self._tcp_offset = torch.tensor(p.tcp_offset_pos, device=self.device).repeat(self.num_envs, 1)
        self._has_tcp_offset = any(abs(v) > 1e-9 for v in p.tcp_offset_pos)
        self._tcp_offset_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self._tcp_offset_rot[:, 0] = 1.0

        # ---- 목표/버퍼 -------------------------------------------------------------
        jl = self.robot.data.soft_joint_pos_limits  # (N, J, 2)
        self._hand_lo = jl[:, self._hand_ids_t, 0]
        self._hand_hi = jl[:, self._hand_ids_t, 1]
        self._arm_lo = jl[:, self._arm_ids_t, 0]
        self._arm_hi = jl[:, self._arm_ids_t, 1]
        self._default_q = self.robot.data.default_joint_pos.clone()
        self.hand_targets = self._default_q[:, self._hand_ids_t].clone()
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._abnormal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)       # env-local
        self.object_spawn_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # ---- 커리큘럼 ---------------------------------------------------------------
        self.difficulty = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._gate_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_reached_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._object_mass = self.object.root_physx_view.get_masses().to(self.device).view(self.num_envs, 1)

        # ---- 접촉 그룹 인덱스 ---------------------------------------------------------
        fingers = list(p.finger_sensor_bodies.keys())
        self._finger_names = fingers
        self._group_a_idx = torch.tensor(
            [fingers.index(f) for f in p.contact_group_a], device=self.device, dtype=torch.long)
        self._group_b_idx = torch.tensor(
            [fingers.index(f) for f in p.contact_group_b], device=self.device, dtype=torch.long)

        print(f"[grasp_lift] profile={p.name} arm={len(self.arm_ids)} hand={len(self.hand_ids)} "
              f"tips={len(self.tip_ids)} action={self.cfg.action_space} obs={self.cfg.observation_space}",
              flush=True)

    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        # 정적 환경 USD (env.usd: RigidBodyAPI 없음, 전 메시 충돌체) —
        # env_0 에 spawn 하면 clone_environments 가 복제한다.
        tbl = self.cfg.table_cfg
        tbl.spawn.func(
            "/World/envs/env_0/Table", tbl.spawn,
            translation=tuple(tbl.init_state.pos), orientation=tuple(tbl.init_state.rot),
        )

        # 손가락별 접촉 센서 — body 마다 개별 생성(다중 body 단일 센서는 force_matrix_w=0,
        # grasp_sensor 실측 함정). Object-only 필터.
        p = PROFILES[self.cfg.profile_name]
        _filter = list(self.cfg.object_contact_filter)
        self._finger_sensors: dict[str, list[ContactSensor]] = {}
        for finger, bodies in p.finger_sensor_bodies.items():
            sensors = []
            for body in bodies:
                s = ContactSensor(ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{body}",
                    filter_prim_paths_expr=_filter,
                    history_length=1,
                    track_air_time=False,
                ))
                sensors.append(s)
                self.scene.sensors[f"contact_{finger}_{body}"] = s
            self._finger_sensors[finger] = sensors

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=True)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ------------------------------------------------------------------
    # 팔 EE(TCP) pose / Jacobian — root frame (IsaacLab task_space_actions 이식)
    # ------------------------------------------------------------------
    def _ee_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self.robot.data.body_pos_w[:, self.palm_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self.palm_idx]
        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w, ee_pos_w, ee_quat_w)
        if self._has_tcp_offset:
            pos_b, quat_b = combine_frame_transforms(pos_b, quat_b, self._tcp_offset, self._tcp_offset_rot)
        return pos_b, quat_b

    def _ee_jacobian_b(self) -> torch.Tensor:
        jac = self.robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :][:, :, self._arm_ids_t]
        rot = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
        jac = jac.clone()
        jac[:, :3, :] = torch.bmm(rot, jac[:, :3, :])
        jac[:, 3:, :] = torch.bmm(rot, jac[:, 3:, :])
        if self._has_tcp_offset:
            jac[:, 0:3, :] += torch.bmm(-skew_symmetric_matrix(self._tcp_offset), jac[:, 3:, :])
        return jac

    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)
        # 팔: relative pose 명령 (root frame)
        cmd = torch.zeros(self.num_envs, 6, device=self.device)
        cmd[:, :3] = self.actions[:, :3] * float(self.cfg.arm_pos_scale)
        cmd[:, 3:] = self.actions[:, 3:6] * float(self.cfg.arm_rot_scale)
        ee_pos, ee_quat = self._ee_pose_b()
        self._ik.set_command(cmd, ee_pos=ee_pos, ee_quat=ee_quat)
        # 손: 절대 목표를 delta 로 이동 + 관절한계 clamp
        self.hand_targets = (
            self.hand_targets + self.actions[:, 6:] * float(self.cfg.hand_joint_scale)
        ).clamp(self._hand_lo, self._hand_hi)

    def _apply_action(self) -> None:
        # 팔 IK — 물리 스텝마다 최신 상태로 재계산(120 Hz)
        ee_pos, ee_quat = self._ee_pose_b()
        joint_pos = self.robot.data.joint_pos[:, self._arm_ids_t]
        jacobian = self._ee_jacobian_b()
        arm_des = self._ik.compute(ee_pos, ee_quat, jacobian, joint_pos)
        # diff IK 는 관절한계 무방비 — 해를 한계 안으로 잘라서 명령한다(태스크 측 처리).
        arm_des = arm_des.clamp(self._arm_lo + 0.02, self._arm_hi - 0.02)
        self.robot.set_joint_position_target(arm_des, joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(self.hand_targets, joint_ids=self.hand_ids)
        # 유휴 관절(반대팔·헤드 등)은 default 유지 — reset 에서 target 설정됨

        # ---- 커리큘럼: 물체 반중력 보상력 (유효 중력 = g × 난이도/max) -------------
        # ★접촉 성립(대향 게이트 참) 시에만 건다 — 잡기 전엔 정상 중력(컵 안정),
        #   잡으면 가벼움 = 커리큘럼 의도 그대로. 접촉 전부터 걸면 0.15g 컵이 마찰을
        #   잃고 떠돌다 낙하(agn_test1 ep1000: episode 323/480, height_delta -11mm).
        frac = (self.difficulty.float().unsqueeze(1) / float(self.cfg.curriculum_max_level)).clamp(
            min=float(self.cfg.gravity_min_frac))
        comp = self._object_mass * _GRAVITY * (1.0 - frac) * self._gate_buf.float().unsqueeze(1)
        f = torch.zeros(self.num_envs, 1, 3, device=self.device)
        f[:, 0, 2] = comp.squeeze(1)
        self.object.set_external_force_and_torque(
            f, torch.zeros_like(f), body_ids=[0], is_global=True)

    # ------------------------------------------------------------------
    def _contact_forces(self) -> torch.Tensor:
        """손가락별 물체 접촉력 크기 (N, F). body 별 센서 합산, Object-필터."""
        mags = []
        for finger in self._finger_names:
            total = torch.zeros(self.num_envs, device=self.device)
            for s in self._finger_sensors[finger]:
                fm = s.data.force_matrix_w  # (N, B, M, 3)
                total = total + fm.view(self.num_envs, -1, 3).sum(dim=1).norm(dim=-1)
            mags.append(total)
        return torch.stack(mags, dim=1)

    def _env_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return pos_w - self.scene.env_origins

    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel
        joint_pos = torch.cat([q[:, self._arm_ids_t], q[:, self._hand_ids_t]], dim=1)
        joint_vel = torch.cat([qd[:, self._arm_ids_t], qd[:, self._hand_ids_t]], dim=1)
        palm_pos = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        tips = (
            self.robot.data.body_pos_w[:, self._tip_ids_t]
            - self.scene.env_origins[:, None, :]
        ).reshape(self.num_envs, -1)
        obj_pos = self._env_local(self.object.data.root_pos_w)
        obj_quat = self.object.data.root_quat_w
        contact = self._contact_forces().clamp(max=20.0)
        obs = torch.cat([
            joint_pos, joint_vel, palm_pos, palm_quat, tips,
            obj_pos, obj_quat, self.goal_pos, contact, self.actions,
        ], dim=1)
        state = torch.cat([
            obs,
            self.object.data.root_lin_vel_w,
            self.object.data.root_ang_vel_w,
            self.difficulty.float().unsqueeze(1) / float(self.cfg.curriculum_max_level),
        ], dim=1)
        return {"policy": torch.nan_to_num(obs), "critic": torch.nan_to_num(state)}

    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        obj_pos = self._env_local(self.object.data.root_pos_w)
        tips = (
            self.robot.data.body_pos_w[:, self._tip_ids_t]
            - self.scene.env_origins[:, None, :]
        )
        contact = self._contact_forces()
        total, terms, gate = compute_grasp_lift_rewards(
            fingertip_pos=tips,
            object_pos=obj_pos,
            goal_pos=self.goal_pos,
            group_a_force=contact[:, self._group_a_idx],
            group_b_force=contact[:, self._group_b_idx],
            actions=self.actions,
            prev_actions=self.prev_actions,
            cfg=self.cfg,
        )
        self._gate_buf.copy_(gate)   # 반중력 커리큘럼용 (접촉 시에만 보상력)
        # abnormal 종료 페널티(관절한계) — _get_dones 가 같은 스텝에 계산한 플래그 사용
        total = total + float(self.cfg.abnormal_penalty) * self._abnormal_buf.float()
        self.prev_actions.copy_(self.actions)

        # 성공 판정(커리큘럼): goal 근접 상태를 매 스텝 갱신 — 리셋 시 마지막 값 사용
        goal_dist = torch.norm(obj_pos - self.goal_pos, dim=-1)
        self._goal_reached_now = goal_dist < float(self.cfg.success_pos_tolerance)

        # ---- 떨어진 컵 즉시 리스폰 (2026-08-20, agn_test3 ep3000 개입) -------------
        # 스텝의 25%가 "컵이 바닥에 있는 죽은 시간"(fell_rate 0.25, height_delta -76mm)
        # 이라 접근·접촉 연습 밀도가 3/4 로 깎였다. 종료(회피 유인 학습, agn_test2)도
        # 방치(죽은 시간, agn_test3)도 아닌 세 번째 선택지: 컵만 스폰 위치로 되돌린다
        # (로봇·에피소드·goal 유지). 보상은 위에서 떨어진 위치 기준으로 이미 계산됐다.
        _fell = obj_pos[:, 2] < float(self.cfg.object_min_z)
        if _fell.any():
            _ids = _fell.nonzero(as_tuple=False).squeeze(-1)
            _root = torch.zeros(len(_ids), 13, device=self.device)
            _root[:, :3] = self.object_spawn_pos[_ids] + self.scene.env_origins[_ids]
            _root[:, 3] = 1.0
            self.object.write_root_state_to_sim(_root, env_ids=_ids)
        self.extras["task/respawn_rate"] = _fell.float().mean()

        # ---- 로깅 ----
        for k, v in terms.items():
            self.extras[f"reward/{k}"] = v.mean()
        self.extras["reward/total"] = total.mean()
        self.extras["task/contact_gate"] = gate.float().mean()
        self.extras["task/goal_dist"] = goal_dist.mean()
        self.extras["task/goal_reached"] = self._goal_reached_now.float().mean()
        self.extras["task/object_height_delta"] = (obj_pos[:, 2] - self.object_spawn_pos[:, 2]).mean()
        self.extras["curriculum/difficulty_mean"] = self.difficulty.float().mean()
        self.extras["curriculum/gravity_frac"] = self.difficulty.float().mean() / float(
            self.cfg.curriculum_max_level)
        _hand_tau = self.robot.data.applied_torque[:, self._hand_ids_t].abs()
        self.extras["debug/hand/torque_mean"] = _hand_tau.mean()
        self.extras["debug/hand/torque_max"] = _hand_tau.max()
        for gi, gname in ((self._group_a_idx, "group_a"), (self._group_b_idx, "group_b")):
            self.extras[f"contact/{gname}_force"] = contact[:, gi].mean()
        return total

    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_pos = self._env_local(self.object.data.root_pos_w)
        spawn_xy = self.object_spawn_pos[:, :2]
        out_xy = (obj_pos[:, :2] - spawn_xy).norm(dim=-1) > float(self.cfg.object_out_of_bounds_xy)
        fell = obj_pos[:, 2] < float(self.cfg.object_min_z)
        # abnormal = 물리 위반만: 관절이 하드 한계를 실제로 넘었거나 속도 폭주.
        # ★근접도(범위의 99%) 기준은 오판정이었다 — j4 하한(0)은 "팔 뻗기"라는 정상
        #   동작의 종점이고 j2 는 init 자세부터 하한 근처다(probe 실측: +x 계단에서
        #   16/16 전멸). IK 해 clamp(위)가 명령을 한계 안에 가두므로, 실제 초과는
        #   접촉으로 밀렸을 때뿐이다.
        q_arm = self.robot.data.joint_pos[:, self._arm_ids_t]
        qd_arm = self.robot.data.joint_vel[:, self._arm_ids_t]
        beyond = (q_arm < self._arm_lo - 0.05) | (q_arm > self._arm_hi + 0.05)
        runaway = qd_arm.abs() > 20.0
        self._abnormal_buf = (beyond | runaway).any(dim=-1)
        # ★2026-08-20 fell/out 은 **종료하지 않는다** (로깅만). agn_test2 ep1000 실측:
        #   컵 근처에 가면 확률적으로 쳐서 떨어뜨림 → 종료 → discount 된 미래 보상 전체
        #   소실 → 정책이 "접근하지 않고 에피소드를 길게 끄는" 회피를 학습
        #   (reaching 0.056→0.005 붕괴 + episode_lengths 384→389 동반 상승).
        #   떨어진 컵은 reaching/tracking 이 자연히 낮아 보상으로 이미 처벌된다.
        terminated = self._abnormal_buf
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["task/abnormal_rate"] = self._abnormal_buf.float().mean()
        self.extras["task/fell_rate"] = fell.float().mean()
        self.extras["task/out_xy_rate"] = out_xy.float().mean()
        return terminated, truncated

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # ---- 커리큘럼 갱신: 에피소드 종료 시점의 goal 근접 여부로 ±1 ----------------
        succ = self._goal_reached_now[env_ids]
        self.difficulty[env_ids] = (
            self.difficulty[env_ids] + torch.where(succ, 1, -1)
        ).clamp(0, int(self.cfg.curriculum_max_level))
        self._goal_reached_now[env_ids] = False
        self._gate_buf[env_ids] = False

        # ---- 로봇: 프로필 init 자세 ---------------------------------------------------
        q0 = self._default_q[env_ids].clone()
        qd0 = torch.zeros_like(q0)
        self.robot.write_joint_state_to_sim(q0, qd0, env_ids=env_ids)
        self.robot.set_joint_position_target(q0, env_ids=env_ids)
        self.hand_targets[env_ids] = q0[:, self._hand_ids_t]
        self._ik.reset(env_ids)
        self.prev_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0

        # ---- 물체 스폰(난이도 비례 범위) + goal = 스폰 + z offset ----------------------
        p = self.profile
        frac = self.difficulty[env_ids].float() / float(self.cfg.curriculum_max_level)
        rng = float(self.cfg.spawn_range_initial) + frac * (
            float(self.cfg.spawn_range_final) - float(self.cfg.spawn_range_initial))
        offs = (torch.rand(n, 2, device=self.device) - 0.5) * 2.0 * rng.unsqueeze(1)
        spawn = torch.zeros(n, 3, device=self.device)
        spawn[:, 0] = p.object_spawn_center[0] + offs[:, 0]
        spawn[:, 1] = p.object_spawn_center[1] + offs[:, 1]
        # +5mm 패딩: 정확 안착 높이는 스폰 침투 반동으로 컵을 튕긴다(실측 침하 -11mm)
        spawn[:, 2] = p.object_spawn_z + 0.005
        self.object_spawn_pos[env_ids] = spawn
        self.goal_pos[env_ids] = spawn + torch.tensor(
            [0.0, 0.0, float(self.cfg.goal_height_offset)], device=self.device)

        root = torch.zeros(n, 13, device=self.device)
        root[:, :3] = spawn + self.scene.env_origins[env_ids]
        root[:, 3] = 1.0
        self.object.write_root_state_to_sim(root, env_ids=env_ids)
