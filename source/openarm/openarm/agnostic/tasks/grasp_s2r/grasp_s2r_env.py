"""grasp_s2r — 제자리 파지 → 리프트 → 목표 이송 → 정지.

제어 스택은 `grasp_s2r_control.GraspS2RControlMixin`(Fabrics 팔 + 시너지 손),
보상은 `grasp_s2r_rewards`, 로봇 종속 정보는 `robot_profiles` 에 있다.

★액션 규약(grasp_v1 계승): palm 은 **홈 기준 델타**다 — `a=0` 이면 홈을 유지한다.
  절대 매핑(`a=0` = 박스 중심)은 σ=1.0 과 곱해지면 매 스텝 작업공간 전역에서 목표를
  재추첨해 접근이 랜덤워크가 된다(선행 트랙 실측).

★래치는 **보상 단계 표시 전용**이다. grasp_v1 은 래치 후 팔 지령을 z 램프 스크립트로
  대체했는데, 여기서는 그 오버라이드가 없다 — 이송까지 정책이 fabric 으로 제어한다.
"""

from __future__ import annotations

import math

import torch

from isaaclab.envs import DirectRLEnv

from .grasp_s2r_control import GraspS2RControlMixin
from .grasp_s2r_env_cfg import GraspS2REnvCfg
from .grasp_s2r_rewards import GRASP_S2R_REWARD_TERMS, compute_grasp_s2r_rewards
from .robot_profiles import PROFILES


class GraspS2REnv(GraspS2RControlMixin, DirectRLEnv):
    cfg: GraspS2REnvCfg

    def __init__(self, cfg: GraspS2REnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        p = PROFILES[self.cfg.profile_name]
        self.profile = p

        # ---- 조인트/바디 해석 (fail-loud: 프로필 선언 수와 대조) ---------------------
        self.arm_ids, arm_names = self.robot.find_joints(p.arm_joint_regex)
        self.hand_ids, hand_names = self.robot.find_joints(p.hand_joint_regex)
        if len(self.arm_ids) != p.num_arm_joints or len(self.hand_ids) != p.num_hand_joints:
            raise RuntimeError(
                f"[{p.name}] 프로필 조인트 수 불일치: arm {len(self.arm_ids)}"
                f"!={p.num_arm_joints} ({arm_names}), hand {len(self.hand_ids)}"
                f"!={p.num_hand_joints} ({hand_names})")
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

        # ---- 접촉 그룹 (프로필 정의) --------------------------------------------------
        fingers = list(p.finger_sensor_bodies.keys())
        self._finger_names = fingers
        if len(p.fingertip_bodies) != len(fingers):
            raise RuntimeError(
                f"[{p.name}] fingertip_bodies({len(p.fingertip_bodies)}) 와 "
                f"finger_sensor_bodies({len(fingers)}) 의 손가락 수가 달라 "
                "그룹 인덱스를 공유할 수 없다")
        self._group_a_idx = torch.tensor(
            [fingers.index(f) for f in p.contact_group_a],
            device=self.device, dtype=torch.long)
        if not p.envelope_fingers:
            raise RuntimeError(f"[{p.name}] envelope_fingers 미정의 — 감쌈 판정 불가")
        # 감쌈 분모 = 대향 그룹 반대편 ∩ 인벨롭 손가락(프로필이 도달 가능 집합을 정의).
        self._wrap_idx = torch.tensor(
            [i for i, f in enumerate(fingers)
             if f in p.contact_group_b and f in p.envelope_fingers],
            device=self.device, dtype=torch.long)
        if len(self._wrap_idx) < 1:
            raise RuntimeError(f"[{p.name}] contact_group_b ∩ envelope_fingers 가 비었다")

        # ---- 팔·손 제어 배선 ----------------------------------------------------------
        self._policy_dt = float(self.cfg.sim.dt) * int(self.cfg.decimation)
        self._setup_fabrics()

        # ---- 버퍼 ---------------------------------------------------------------------
        jl = self.robot.data.soft_joint_pos_limits           # (N, J, 2)
        self._arm_lo = jl[:, self._arm_ids_t, 0]
        self._arm_hi = jl[:, self._arm_ids_t, 1]
        self._default_q = self.robot.data.default_joint_pos.clone()
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)       # env-local
        self.object_spawn_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # 지령 리미터 상태 — 리셋 직후 첫 지령은 "변화"가 아니라 초기화라 안 건다.
        self._prev_palm_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_palm_cmd_rot = torch.zeros(self.num_envs, 3, device=self.device)
        self._palm_cmd_primed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._palm_cmd_step_raw = torch.zeros(self.num_envs, device=self.device)

        # 닫기 게이트 상태(정렬도) — `_pre_physics_step` 이 매 스텝 갱신한다.
        self._close_gate = torch.ones(self.num_envs, device=self.device)
        self._cage_ctr_dist = torch.zeros(self.num_envs, device=self.device)
        # 케이지 중심의 palm-local 오프셋 — `_report_home_cage` 가 홈에서 실측해 고정한다.
        self._cage_offset_palm = torch.zeros(3, device=self.device)

        # 래치 (보상 단계 표시 전용)
        self._latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._wrap_at_latch = torch.zeros(self.num_envs, device=self.device)
        self._disp_at_latch = torch.zeros(self.num_envs, device=self.device)

        # 판정 버퍼 — `_get_dones` 가 먼저 돌고 `_get_rewards` 가 같은 스텝에 재사용한다.
        self._tilt_deg = torch.zeros(self.num_envs, device=self.device)
        self._abnormal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # ★접촉 지속 카운터. 끊기면 0 이라 "닿았다 뗐다"로는 못 채운다.
        self._persist = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._stay_run = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 단계 도달 플래그 — 에피소드 동안 OR 누적, 리셋에서만 평균 기록(스텝 비용 0).
        self._stage_names = ("grasp", "lift", "transfer", "stay")
        self._stage_hit = torch.zeros(
            self.num_envs, len(self._stage_names), dtype=torch.bool, device=self.device)

        self._init_home_palm()
        self._report_home_cage()
        self._assert_goal_reachable()
        self._setup_cmd_markers()

        # 액션 델타 박스 — palm 은 홈 기준 상대다.
        _d = torch.tensor(self.cfg.palm_delta_xyz, device=self.device)
        _r = math.radians(float(self.cfg.palm_delta_rot_deg))
        self._delta_lo = torch.cat([-_d, torch.full((3,), -_r, device=self.device)])
        self._delta_hi = torch.cat([_d, torch.full((3,), _r, device=self.device)])
        # ★홈은 항상 도달 가능해야 한다(박스가 홈을 잘라내면 a=0 의 의미가 깨진다).
        self._box_lo = torch.minimum(self._palm_lo, self._home_palm)
        self._box_hi = torch.maximum(self._palm_hi, self._home_palm)

        print(f"[grasp_s2r] profile={p.name} arm={len(self.arm_ids)} "
              f"hand={len(self.hand_ids)} tips={len(self.tip_ids)} "
              f"action={self.cfg.action_space} obs={self.cfg.observation_space} "
              f"state={self.cfg.state_space} fabric={p.fabric_robot_dir}", flush=True)

    # ------------------------------------------------------------------
    def _report_home_cage(self) -> None:
        """홈 자세의 **케이지 중심**(엄지 팁 ↔ 4지 팁 중점)이 컵 대비 어디인지 실측·보고.

        ★★이 수치가 접근 난이도를 결정한다. 케이지가 컵보다 **앞(+x)** 에 있으면 정책은
          팔을 **후진**시켰다가 다시 들어가야 하고, 3D 대각선 이동이라 훨씬 어렵다.
          케이지가 컵보다 **위** 에 있으면 지령이 계속 아래로 가면서 엄지가 컵에 걸리고,
          걸린 채 눌리다 풀리면 손이 테이블까지 내려간다(둘 다 사용자 GUI 관찰).
        ★좌팔 그리퍼 트랙 결론: "컵을 앞에 둔다"와 "홈을 뒤로 물린다"는 로봇 기준
          상대 배치가 같아 물리적으로 동등하다 — 그쪽은 스폰을 앞으로 밀어 해결했다.
        """
        p = self.profile
        tips = (self.robot.data.body_pos_w[:, self._tip_ids_t]
                - self.scene.env_origins[:, None, :])[0]
        _a = int(self._group_a_idx[0])
        _others = [i for i in range(len(self.tip_ids)) if i != _a]
        cage = 0.5 * (tips[_a] + tips[_others].mean(dim=0))
        r_cage = 0.5 * float((tips[_a] - tips[_others].mean(dim=0)).norm())
        self._r_cage = r_cage          # 닫기 게이트 임계로도 쓴다

        # ★★케이지 중심을 **palm 에 강체로 붙인다**. 홈 자세에서 한 번 재고 그 뒤로는
        #   손가락이 어떻게 움직이든 이 오프셋이 변하지 않는다.
        #   08.27 실측(s2r_a6): 중심을 실시간 손끝으로 두면 팔이 정지한 구간
        #   (palm_to_cup 0.120~0.140, n=147)에서 corr(syn_close, cage_dist) = −0.974 —
        #   **팔을 안 움직이고 손만 오므려도 중심이 컵 쪽으로 50mm 당겨져** 게이트가
        #   저절로 열렸다(램프 폭 60mm 의 83%). "정렬되면 닫아라"가 아니라
        #   "닫으면 닫아도 된다"는 양의 되먹임이라 게이트가 아무것도 막지 못했다.
        _palm = (self.robot.data.body_pos_w[:, self.palm_idx]
                 - self.scene.env_origins)[0]
        _R = self._palm_ee_R()[0]                       # (3,3), 열 = palm 축
        self._cage_offset_palm = _R.transpose(0, 1) @ (cage - _palm)
        cup = [p.object_spawn_center[0], p.object_spawn_center[1],
               float(self.cfg.table_surface_z) + float(self.cfg.object_origin_offset_z)
               + float(self.cfg.object_grasp_z_offset)]
        d = [round(float(cage[i]) - cup[i], 4) for i in range(3)]
        print(f"[grasp_s2r] 홈 케이지 중심={[round(float(v), 4) for v in cage]} "
              f"· 반경 {r_cage * 1000:.0f}mm | 컵 파지중심={[round(v, 4) for v in cup]} "
              f"| 케이지−컵 = {d} m", flush=True)
        # 전진축 정렬 허용오차 — 케이지 반경의 1/6(20mm) 안이면 후진이 필요 없다.
        if d[0] > 0.02:
            print(f"[grasp_s2r] ⚠ 케이지가 컵보다 {d[0] * 1000:.0f}mm **앞(+x)** 이다 — "
                  "정책이 후진 후 재접근해야 한다(3D 대각선). 컵 스폰을 앞으로 밀거나 "
                  "홈을 뒤로 물릴 것.", flush=True)
        # 접근 간격이 케이지 반경보다 좁으면 리셋 순간 손가락이 컵을 관통한다.
        _gap_xy = float((cage[:2] - torch.tensor(cup[:2], device=cage.device)).norm())
        if _gap_xy < r_cage:
            print(f"[grasp_s2r] ⚠ 홈 케이지↔컵 수평 간격 {_gap_xy * 1000:.0f}mm 가 "
                  f"케이지 반경 {r_cage * 1000:.0f}mm 보다 좁다 — 리셋에서 관통 위험.",
                  flush=True)

    # ------------------------------------------------------------------
    def _assert_goal_reachable(self) -> None:
        """목표가 palm 박스 안인지 부팅에서 확인 — 밖이면 과제가 성립하지 않는다."""
        p = self.profile
        settled_z = float(self.cfg.table_surface_z) + float(self.cfg.object_origin_offset_z)
        goal = [
            p.object_spawn_center[0] + self.cfg.goal_offset_xyz[0],
            p.object_spawn_center[1] + self.cfg.goal_offset_xyz[1],
            settled_z + self.cfg.goal_offset_xyz[2],
        ]
        lo = self._palm_lo[:3].tolist()
        hi = self._palm_hi[:3].tolist()
        if any(g < lo[i] or g > hi[i] for i, g in enumerate(goal)):
            raise RuntimeError(
                f"[{p.name}] 이송 목표 {[round(v, 3) for v in goal]} 가 palm 박스 "
                f"{[round(v, 3) for v in lo]}~{[round(v, 3) for v in hi]} 밖이다 — "
                "goal_offset_xyz 를 줄이거나 프로필 박스를 넓혀라.")
        print(f"[grasp_s2r] 이송 목표 = {[round(v, 3) for v in goal]} "
              f"(정착고 {settled_z:.4f} + offset {list(self.cfg.goal_offset_xyz)})",
              flush=True)

    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)

        # ---- 팔: palm 6D = **홈 + 델타** -------------------------------------------
        # a=0 → 홈. 탐색이 홈 주변 유계 오프셋으로 묶여 절대 매핑의 랜덤워크가 없다.
        delta = 0.5 * (self.actions[:, :6] + 1.0) * (self._delta_hi - self._delta_lo) \
            + self._delta_lo
        self.palm_targets = (self._home_palm.unsqueeze(0) + delta).clamp(
            self._box_lo, self._box_hi)

        # ---- 지령 변화율 리미터 -----------------------------------------------------
        _lim = float(self.cfg.palm_cmd_rate_limit_m)
        _step3 = self.palm_targets[:, :3] - self._prev_palm_cmd
        # 클램프 **전** 원값 로깅 — 상한이 물리는 비율의 유일한 근거다.
        self._palm_cmd_step_raw = torch.where(
            self._palm_cmd_primed, _step3.norm(dim=-1),
            torch.zeros_like(self._palm_cmd_step_raw))
        if _lim > 0.0:
            _scale = (_lim / _step3.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
            self.palm_targets[:, :3] = torch.where(
                self._palm_cmd_primed.unsqueeze(-1),
                self._prev_palm_cmd + _step3 * _scale,
                self.palm_targets[:, :3])
        self._prev_palm_cmd = self.palm_targets[:, :3].clone()

        _lim_r = math.radians(float(self.cfg.palm_cmd_rate_limit_rot_deg))
        if _lim_r > 0.0:
            _dr = self.palm_targets[:, 3:6] - self._prev_palm_cmd_rot
            _sr = (_lim_r / _dr.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
            self.palm_targets[:, 3:6] = torch.where(
                self._palm_cmd_primed.unsqueeze(-1),
                self._prev_palm_cmd_rot + _dr * _sr,
                self.palm_targets[:, 3:6])
        self._prev_palm_cmd_rot = self.palm_targets[:, 3:6].clone()
        self._palm_cmd_primed |= True
        self._update_cmd_markers()          # 시각화 전용 — 물리·보상에 영향 없음

        # ---- 손: 시너지 -------------------------------------------------------------
        _prev = self._syn_target
        # ★★닫기 게이트: 컵이 케이지 안에 들어오기 전에는 오므리지 않는다.
        #   래치로는 못 막는다 — 래치는 보상을 여는 신호일 뿐이고 닫힘은 손 액션이
        #   직접 만든다. 경계에서 끊지 않고 램프를 둬 gradient 를 남긴다.
        # ★케이지 중심은 palm 강체 오프셋(`_cage_offset_palm`, 홈 실측)으로 만든다.
        #   실시간 손끝 평균이면 손을 오므리는 것만으로 게이트가 열린다(위 실측 근거).
        #   이제 게이트는 **팔 지령에만** 의존한다 — 정책이 위치를 맞춰야 열린다.
        # ★거리는 3D. xy 투영은 z 를 못 봐서 palm·검지가 컵보다 내려간 잘못된 자세도
        #   통과시켰다(사용자 GUI 관찰: 엄지가 컵에 걸린 채 접근). 3D 로 바꾸면
        #   상수를 늘리지 않고 높이 조건이 같이 들어온다.
        _obj = self._env_local(self.object.data.root_pos_w)
        _palm = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])
        _cage = _palm + (self._palm_ee_R() @ self._cage_offset_palm)
        self._cage_ctr_dist = (_cage - _obj).norm(dim=-1)
        if bool(self.cfg.close_gate_enabled):
            _ramp = max(float(self.cfg.close_gate_ramp) * self._r_cage, 1e-6)
            _g = ((self._r_cage - self._cage_ctr_dist) / _ramp).clamp(0.0, 1.0)
            # ★래치 후에는 해제한다. 게이트는 **접근 구간 전용**이다 — 들고 가는 중에
            #   컵이 흔들려 게이트가 닫히면 다시 쥘 길이 막힌다(audit Check 3).
            self._close_gate = torch.where(
                self._latched, torch.ones_like(_g), _g)
        else:
            self._close_gate = torch.ones(self.num_envs, device=self.device)
        self._syn_target = self._synergy_targets(self.actions[:, 6:])
        self._syn_vel = (self._syn_target - _prev) / self._policy_dt
        # ★fabric 의 손 **상태**를 실제 손 자세로 동기화한다. 안 그러면 fabric 이
        #   실재하지 않는 손으로 충돌구 FK 를 계산해 없는 자기충돌을 피하려 팔을 민다.
        self.fabric_q[:, self.profile.num_arm_joints:] = self._syn_to_fab(self._syn_target)

        self._step_fabric()

    # ------------------------------------------------------------------
    # 관측
    # ------------------------------------------------------------------
    def _tip_force_local(self) -> torch.Tensor:
        """손끝 접촉력을 **팁 로컬 프레임**으로 회전한 3축 벡터 (N, 3·T).

        실기 `fingertip_*/wrench` 와 직접 대응시키기 위한 표현이다(월드 프레임 힘은
        팔 자세가 바뀌면 같은 접촉이 다른 값으로 읽힌다).
        """
        from isaaclab.utils.math import quat_apply, quat_conjugate
        out = []
        _max = float(self.cfg.contact_force_max)
        for k, finger in enumerate(self._finger_names):
            s = self._finger_sensors[finger][-1]
            f_w = s.data.force_matrix_w.view(self.num_envs, -1, 3).sum(dim=1)
            q = self.robot.data.body_quat_w[:, self.tip_ids[k]]
            out.append(quat_apply(quat_conjugate(q), f_w) / _max)
        return torch.cat(out, dim=1).clamp(-1.0, 1.0)

    def _joint_pos_err(self) -> torch.Tensor:
        """손 관절 목표 − 실측 (N, n_hand), 부호 보존 정규화.

        ★인벨롭이 잘 될수록 팁 F/T 가 0 을 읽는 문제가 있어, 추종 오차가 **주 파지력
          관측**이 된다(잡고 있으면 목표를 못 따라가 오차가 남는다).
        """
        err = self._syn_target - self.robot.data.joint_pos[:, self._syn_ids]
        return (err / float(self.cfg.joint_pos_err_max)).clamp(-1.0, 1.0)

    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel
        n = self.num_envs
        cfgn = self.cfg

        arm_q = q[:, self._arm_ids_t]
        arm_qd = qd[:, self._arm_ids_t]
        hand_q = q[:, self._hand_ids_t]
        hand_qd = qd[:, self._hand_ids_t]
        palm_pos = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])
        _R = self._palm_ee_R()
        # 쿼터니언은 q ≡ −q 부호 이중성이 있어 회전행렬 두 열로 준다.
        palm_ax = torch.cat([_R[:, :, 0], _R[:, :, 1]], dim=1)
        tips_w = self.robot.data.body_pos_w[:, self._tip_ids_t]
        tips_rel_palm = (
            tips_w - self.robot.data.body_pos_w[:, self.palm_idx].unsqueeze(1)
        ).reshape(n, -1)
        obj_pos = self._env_local(self.object.data.root_pos_w)
        palm_to_obj = obj_pos - palm_pos
        obj_to_tips = (tips_w - self.scene.env_origins[:, None, :]
                       - obj_pos.unsqueeze(1)).reshape(n, -1)
        tip_force = self._tip_force_local()
        joint_err = self._joint_pos_err()
        goal_rel = self.goal_pos - obj_pos

        # actor 에만 노이즈 — critic 은 clean state 를 받는다.
        _noisy = torch.cat([
            arm_q + torch.randn_like(arm_q) * cfgn.obs_noise_qpos,
            arm_qd + torch.randn_like(arm_qd) * cfgn.obs_noise_qvel,
            hand_q + torch.randn_like(hand_q) * cfgn.obs_noise_qpos,
            hand_qd + torch.randn_like(hand_qd) * cfgn.obs_noise_qvel,
            palm_pos + torch.randn_like(palm_pos) * cfgn.obs_noise_body,
            palm_ax,
            tips_rel_palm + torch.randn_like(tips_rel_palm) * cfgn.obs_noise_body,
            palm_to_obj + torch.randn_like(palm_to_obj) * cfgn.obs_noise_object,
            obj_to_tips + torch.randn_like(obj_to_tips) * cfgn.obs_noise_object,
            tip_force, joint_err, self.actions, goal_rel,
        ], dim=1)

        clean = torch.cat([
            arm_q, arm_qd, hand_q, hand_qd, palm_pos, palm_ax, tips_rel_palm,
            palm_to_obj, obj_to_tips, tip_force, joint_err, self.actions, goal_rel,
        ], dim=1)

        _mid, _dist = self._contact_forces_split()
        _thr = float(cfgn.contact_force_threshold)
        _max = float(cfgn.contact_force_max)
        state = torch.cat([
            clean,
            self.object.data.root_lin_vel_w,
            self.object.data.root_ang_vel_w,
            self.object.data.root_quat_w,
            (obj_pos[:, 2] - self.object_spawn_pos[:, 2]).unsqueeze(1),
            (_dist > _thr).float(), (_dist / _max).clamp(max=1.0),
            (_mid > _thr).float(), (_mid / _max).clamp(max=1.0),
            (self.episode_length_buf.float()
             / float(self.max_episode_length)).unsqueeze(1),
            (tips_w - self.scene.env_origins[:, None, :] - obj_pos.unsqueeze(1)
             ).norm(dim=-1),
            (self.goal_pos - obj_pos).norm(dim=-1, keepdim=True),
        ], dim=1)
        return {"policy": torch.nan_to_num(_noisy), "critic": torch.nan_to_num(state)}

    # ------------------------------------------------------------------
    # 보상
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        cfgn = self.cfg
        obj_pos = self._env_local(self.object.data.root_pos_w)
        palm_pos = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])
        tips = self.robot.data.body_pos_w[:, self._tip_ids_t] \
            - self.scene.env_origins[:, None, :]

        # ---- 접촉 ------------------------------------------------------------------
        _thr = float(cfgn.contact_force_threshold)
        tip_c = self._tip_contact_forces() > _thr                 # (N, F)
        mid_f, dist_f = self._contact_forces_split()
        mid_c, dist_c = mid_f > _thr, dist_f > _thr
        n_tip = len(self._finger_names)
        tip_frac = tip_c.float().sum(dim=1) / n_tip
        full_tip = tip_c.all(dim=1)
        grip_c = tip_c | mid_c | dist_c
        grip_frac = grip_c.float().sum(dim=1) / n_tip
        # 감쌈 **깊이** = per-finger (중간 AND 원위). 서로 다른 손가락에 얕게 닿는 것을
        # 깊은 감쌈으로 오인하지 않는다.
        wrap_frac = (mid_c & dist_c)[:, self._wrap_idx].float().mean(dim=1)
        n_grip = grip_c.float().sum(dim=1)

        # 접촉 지속 — 끊기면 0 으로 되돌아간다.
        # ★★임계는 래치와 같은 손가락 수다. `>= 1` 이면 **손가락 하나를 컵에 대고
        #   가만히 있는 것**이 만점이 된다 — 08.27 실측(s2r_a6, 202 iter):
        #   contact_persistence 0.918 인데 touch_frac 0.003·wrap_frac 0.000 이었고,
        #   grasp 항 2.07/step 이 전체 보상의 88% 였다(0.75·0.25·0.918·12 = 2.07 로
        #   산술 일치). 정책은 594 스텝 내내 한 손가락만 대고 있는 데 수렴했다.
        #   래치 조건과 묶으면 persist_frac 은 **래치까지의 진척도**가 되어, 주차해도
        #   그 자리가 파지 직전이다.
        _touch = n_grip >= int(cfgn.lift_start_min_grip_fingers)
        self._persist = torch.where(_touch, self._persist + 1,
                                    torch.zeros_like(self._persist))
        persist_frac = (self._persist.float()
                        / float(cfgn.grasp_ready_hold_steps)).clamp(max=1.0)

        # ---- 래치 (보상 단계 표시 전용 — 팔 지령은 건드리지 않는다) ------------------
        _ready = n_grip >= int(cfgn.lift_start_min_grip_fingers)
        self._hold_count = torch.where(
            _ready & ~self._latched, self._hold_count + 1,
            torch.where(self._latched, self._hold_count,
                        torch.zeros_like(self._hold_count)))
        _just = (~self._latched) & (self._hold_count >= int(cfgn.grasp_ready_hold_steps))
        self._latched = self._latched | _just

        # ---- 기하 --------------------------------------------------------------------
        grasp_center = obj_pos.clone()
        grasp_center[:, 2] += float(cfgn.object_grasp_z_offset)
        palm_to_cup = (palm_pos - grasp_center).norm(dim=-1)
        cup_disp = (obj_pos[:, :2] - self.object_spawn_pos[:, :2]).norm(dim=-1)
        height_delta = obj_pos[:, 2] - self.object_spawn_pos[:, 2]
        goal_dist = (obj_pos - self.goal_pos).norm(dim=-1)

        # ---- 케이지 중심 ↔ 컵 -----------------------------------------------------
        # ★★08.27 사용자 GUI 관찰로 확정된 수정: "손바닥·검지~소지는 잘 붙는데 **엄지가
        #   걸려** 인벨롭이 안 된다".
        #   구 수식은 대향축을 `접근방향을 90° 회전`으로 잡았다 —
        #   `axis = (−dir_y, dir_x)` 는 **좌/우 부호가 임의**라, 접근 방향에 따라 엄지
        #   목표가 실제 엄지의 반대편에 놓인다. 그러면 손목을 뒤집어야 도달 가능한
        #   자세를 요구하게 되고, 정책은 그 방향으로 갈 수 없어 엄지가 걸린 채 4지만
        #   붙인다. 실측 귀결: grip_frac 0.20 인데 wrap_frac 이 2,228 iter 내내 0.000.
        # ★수정: 대향 중점을 **손 자신의 기하**에서 뽑는다. 손잡이 방향이 구조적으로
        #   맞고(엄지가 어디 있든 정의가 성립), 물체 반경 상수도 필요 없다.
        #   `opp_mid` 가 컵 중심에 오려면 엄지와 4지가 컵을 **사이에 두어야** 한다 —
        #   두 그룹이 같은 쪽에 있으면 중점이 그쪽으로 쏠려 거리가 남는다.
        _a = int(self._group_a_idx[0])
        _others = [i for i in range(n_tip) if i != _a]
        opp_mid = 0.5 * (tips[:, _a] + tips[:, _others].mean(dim=1))
        cage_dist = (opp_mid - grasp_center).norm(dim=-1)

        # 래치 시점 스냅샷 — 감쌈 유지 기준선과 밀림 감쇠 기준.
        self._wrap_at_latch = torch.where(_just, wrap_frac, self._wrap_at_latch)
        self._disp_at_latch = torch.where(_just, cup_disp, self._disp_at_latch)

        # ---- 자세·안정 ---------------------------------------------------------------
        upright_q = torch.exp(-self._tilt_deg / float(cfgn.upright_sharpness))
        lin_v = self.object.data.root_lin_vel_w.norm(dim=-1)
        ang_v = self.object.data.root_ang_vel_w.norm(dim=-1)
        stable = (lin_v <= float(cfgn.stable_lin_vel)) & (ang_v <= float(cfgn.stable_ang_vel))
        stability_q = torch.exp(-2.0 * lin_v) * torch.exp(-0.5 * ang_v)

        # ---- 성공 · stay 유지 ---------------------------------------------------------
        lifted = height_delta >= float(cfgn.lift_success_height)
        at_goal = goal_dist <= float(cfgn.goal_pos_tolerance)
        holding = (n_grip >= 4) & tip_c[:, _a]        # 4지 이상 + 엄지 접촉
        self._success_now = (
            lifted & at_goal & holding & stable
            & (self._tilt_deg <= float(cfgn.success_tilt_max_deg)))
        _stay_ok = at_goal & stable & (n_grip >= 2)
        self._stay_run = torch.where(_stay_ok, self._stay_run + 1,
                                     torch.zeros_like(self._stay_run))
        stay_frac = (self._stay_run.float()
                     / float(max(int(cfgn.stay_hold_steps), 1))).clamp(max=1.0)

        action_delta = (self.actions - self.prev_actions).pow(2).mean(dim=-1).sqrt()
        self.prev_actions = self.actions.clone()

        total, terms, gates = compute_grasp_s2r_rewards(
            tip_contact_frac=tip_frac,
            full_tip_contact=full_tip,
            contact_persistence_frac=persist_frac,
            wrap_frac=wrap_frac,
            wrap_at_latch=self._wrap_at_latch,
            grip_frac=grip_frac,
            palm_to_cup_dist=palm_to_cup,
            cage_dist=cage_dist,
            cup_height_delta=height_delta,
            cup_xy_disp_now=cup_disp,
            cup_xy_disp_ref=self._disp_at_latch,
            cup_tilt_deg=self._tilt_deg,
            goal_dist=goal_dist,
            upright_quality=upright_q,
            lift_latched=self._latched,
            stay_frac=stay_frac,
            stable=stable,
            stability_quality=stability_q,
            success_now=self._success_now,
            action_delta_norm=action_delta,
            cfg=cfgn,
        )
        total = total + float(cfgn.abnormal_penalty) * self._abnormal.float()

        # 단계 도달 누적 (리셋에서만 평균 기록 — 스텝 비용 0)
        self._stage_hit[:, 0] |= self._latched
        self._stage_hit[:, 1] |= lifted & self._latched
        self._stage_hit[:, 2] |= self._latched & lifted & (goal_dist < 0.10)
        self._stage_hit[:, 3] |= self._success_now

        for k in GRASP_S2R_REWARD_TERMS:
            self.extras[f"reward/{k}"] = terms[k].mean()
        self.extras["reward/total"] = total.mean()
        for k, v in gates.items():
            self.extras[f"gate/{k}"] = v.mean()
        self.extras["task/wrap_frac"] = wrap_frac.mean()
        self.extras["task/grip_frac"] = grip_frac.mean()
        self.extras["task/touch_frac"] = tip_frac.mean()
        self.extras["task/goal_dist"] = goal_dist.mean()
        self.extras["task/height_delta"] = height_delta.mean()
        self.extras["task/cup_disp"] = cup_disp.mean()
        self.extras["task/palm_to_cup"] = palm_to_cup.mean()
        self.extras["task/cage_dist"] = cage_dist.mean()
        self.extras["task/tilt_deg"] = self._tilt_deg.mean()
        self.extras["task/latched"] = self._latched.float().mean()
        self.extras["task/success"] = self._success_now.float().mean()
        self.extras["task/stay_run"] = self._stay_run.float().mean()
        self.extras["task/syn_close"] = self._syn_close.mean()
        # ★손 관절 추종오차 — 액추에이터 포화의 직접 지표. τ = k·err 이므로
        #   err ≥ effort_limit/stiffness 면 토크가 천장에 붙어 힘 제어가 무효가 된다
        #   (5.0/1.5 기준 0.30 rad = 17.2°). 지금까지 팔(fabric/joint_err_*)만 있었다.
        _herr = (self._syn_target
                 - self.robot.data.joint_pos[:, self._syn_ids]).abs()
        self.extras["task/hand_joint_err_mean"] = _herr.mean()
        self.extras["task/hand_joint_err_max"] = _herr.max()
        # ★채널별 폐쇄도 — 전체 평균만 보면 "어느 채널이 안 닫히는지"를 못 본다.
        #   08.27: 평균 0.278 이 채널1(`_2`)만 폐쇄한 예측치 0.250 과 맞아떨어졌고,
        #   GUI 관찰(`_2` 완전굴곡·`_3`/`_4` 정지)과 일치했다. ch2 가 낮은 이유가
        #   "명령이 안 나간다"인지 "명령은 나가는데 동결이 먹는다"인지 가른다.
        for _c in range(self._syn_nch):
            _m = self._syn_ch == _c
            self.extras[f"task/syn_close_ch{_c}"] = self._syn_close[:, _m].mean()
        self.extras["task/close_gate"] = self._close_gate.mean()
        self.extras["task/cage_ctr_dist"] = self._cage_ctr_dist.mean()
        self.extras["task/abnormal_rate"] = self._abnormal.float().mean()
        self.extras["contact/force_max"] = self._contact_forces().max()
        self.extras["fabric/palm_cmd_step_raw"] = self._palm_cmd_step_raw.mean()
        _jerr = (self.fabric_q[:, : self.profile.num_arm_joints]
                 - self.robot.data.joint_pos[:, self._arm_ids_t]).abs()
        self.extras["fabric/joint_err_mean"] = _jerr.mean()
        # ★`fabric_q` 는 오픈루프 적분 plant 라 리셋 전까지 실측으로 되돌아오지 않는다.
        #   에피소드 **안에서** 팔이 막히면(접촉·관절한계) fabric 만 계속 적분해 격차가
        #   벌어지는데, 평균은 그 순간을 묻어버린다. 08.27 실측으로 **누적은 반증**됐지만
        #   (ep_len 16→594 로 37배인데 joint_err 0.040→0.033 로 감소, 전 구간 0.023~0.053)
        #   그건 평균 얘기다 — 최대값을 따로 봐야 막힘 구간을 잡는다.
        self.extras["fabric/joint_err_max"] = _jerr.max()
        self.extras["fabric/palm_err_mean"] = (
            self.palm_targets[:, :3] + self._fab_to_env - palm_pos).norm(dim=-1).mean()
        return total

    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_pos = self._env_local(self.object.data.root_pos_w)
        from isaaclab.utils.math import quat_apply
        _up = quat_apply(
            self.object.data.root_quat_w,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3))
        self._tilt_deg = torch.rad2deg(torch.acos(_up[:, 2].clamp(-1.0, 1.0)))

        out_x = (obj_pos[:, 0] < self.cfg.object_out_x[0]) | \
                (obj_pos[:, 0] > self.cfg.object_out_x[1])
        out_y = (obj_pos[:, 1] < self.cfg.object_out_y[0]) | \
                (obj_pos[:, 1] > self.cfg.object_out_y[1])
        fell = obj_pos[:, 2] < float(self.cfg.object_min_z)
        tipped = self._tilt_deg > float(self.cfg.tilt_reset_deg)

        # abnormal = 물리 위반만(관절 한계 초과 또는 속도 폭주).
        q_arm = self.robot.data.joint_pos[:, self._arm_ids_t]
        qd_arm = self.robot.data.joint_vel[:, self._arm_ids_t]
        beyond = (q_arm < self._arm_lo - 0.05) | (q_arm > self._arm_hi + 0.05)
        runaway = qd_arm.abs() > float(self.cfg.abnormal_qd)
        self._abnormal = (beyond | runaway).any(dim=-1)

        terminated = out_x | out_y | fell | tipped | self._abnormal
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # ★종료 원인별 비율. 없으면 "무엇이 에피소드를 끝냈는가"를 다른 지표로
        #   역산해야 한다(08.27 자살 경로 진단에서 실제로 그랬다).
        self.extras["done/out_xy"] = (out_x | out_y).float().mean()
        self.extras["done/fell"] = fell.float().mean()
        self.extras["done/tipped"] = tipped.float().mean()
        self.extras["done/abnormal"] = self._abnormal.float().mean()
        self.extras["done/truncated"] = truncated.float().mean()
        return terminated, truncated

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # 단계 도달률은 리셋 시점에만 기록한다.
        for i, nm in enumerate(self._stage_names):
            self.extras[f"stage/{nm}"] = self._stage_hit[env_ids, i].float().mean()
        self._stage_hit[env_ids] = False

        # ---- 로봇: 고정 홈 ------------------------------------------------------------
        q0 = self._default_q[env_ids].clone()
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), env_ids=env_ids)
        self.robot.set_joint_position_target(q0, env_ids=env_ids)
        # 손은 완전 개방에서 시작. `_syn_target` 을 안 맞추면 첫 스텝에 거대한 가짜 속도.
        self._syn_close[env_ids] = 0.0
        self._syn_target[env_ids] = q0[:, self._syn_ids]
        self._syn_vel[env_ids] = 0.0

        # ---- fabric 씨딩 (리셋이 fabric 상태를 실측과 맞추는 유일한 지점) -------------
        self.fabric_q[env_ids] = q0[:, self._fab_t]
        self.fabric_qd[env_ids] = 0.0
        self.fabric_qdd[env_ids] = 0.0
        self.palm_targets[env_ids] = self._home_palm.unsqueeze(0)
        self._palm_cmd_primed[env_ids] = False

        # ---- 버퍼 -----------------------------------------------------------------------
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._latched[env_ids] = False
        self._hold_count[env_ids] = 0
        self._wrap_at_latch[env_ids] = 0.0
        self._disp_at_latch[env_ids] = 0.0
        self._persist[env_ids] = 0
        self._stay_run[env_ids] = 0
        self._success_now[env_ids] = False
        self._abnormal[env_ids] = False

        # ---- 물체 스폰 -------------------------------------------------------------------
        p = self.profile
        rng = float(self.cfg.spawn_range)
        offs = (torch.rand(n, 2, device=self.device) - 0.5) * 2.0 * rng
        spawn = torch.zeros(n, 3, device=self.device)
        spawn[:, 0] = p.object_spawn_center[0] + offs[:, 0]
        spawn[:, 1] = p.object_spawn_center[1] + offs[:, 1]
        spawn[:, 2] = float(self.cfg.object_spawn_z)
        # ★기준선은 스폰점이 아니라 **정착고**다(스폰 패드가 리프트 기준에 실리면 안 된다).
        settled = spawn.clone()
        settled[:, 2] = float(self.cfg.table_surface_z) + float(self.cfg.object_origin_offset_z)
        self.object_spawn_pos[env_ids] = settled
        self.goal_pos[env_ids] = settled + torch.tensor(
            self.cfg.goal_offset_xyz, device=self.device)

        root = torch.zeros(n, 13, device=self.device)
        root[:, :3] = spawn + self.scene.env_origins[env_ids]
        root[:, 3] = 1.0
        self.object.write_root_state_to_sim(root, env_ids=env_ids)
