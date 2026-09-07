"""grasp_fj — Track B: 팔 7D 관절 증분(+EMA) + 시너지 15D, **Fabrics 없음**.

`GraspKPEnv`(Track A)를 상속해 팔 액션 어댑터에 해당하는 훅만 덮어쓴다(DESIGN §1 B,
`scratchpad/maps/control.md` §6-7 우회 목록). 목표열·보상·관측·종료·지연·외란은 전부 A 그대로.

팔: `q*_t = clamp(q*_{t-1} + k_arm·a)` → `q*_t = α·q*_t + (1−α)·q*_{t-1}` → 위치 목표만
(`set_joint_velocity_target` 은 팔에 주지 않는다 — 실기 JTC 규약). 손: A 와 동일(시너지 + PD).

★fabric 관련 부모 버퍼(`fabric_q/qd/qdd`, `_fab_t`, `_syn_to_fab_idx`, `palm_targets`,
  `_palm_lo/_hi`, `_home_palm`, `_fab_to_env`)는 부모의 리셋·앵커·박스 부트스트랩이 읽으므로
  **같은 모양으로 할당만** 한다. 제어 경로는 `_arm_q_target` 하나만 읽는다.
"""

from __future__ import annotations

import math

import torch

from ..grasp_kp.grasp_kp_env import GraspKPEnv
from .grasp_fj_env_cfg import GraspFJEnvCfg


class GraspFJEnv(GraspKPEnv):
    cfg: GraspFJEnvCfg

    # ------------------------------------------------------------------
    # 부트스트랩 — fabric 자리에 관절 목표 버퍼
    # ------------------------------------------------------------------
    def _setup_fabrics(self) -> None:
        """mixin `_setup_fabrics` 의 fabric-free 판. 시너지·인덱스·palm 박스 할당은 동일."""
        p = self.profile
        self._setup_synergy()                     # ★`_syn_ids` 가 아래 인덱스보다 먼저(부모 순서 계약)
        self.fabric = None                        # 명시적 OFF — A 의 `_log_fabric_metrics` 가 None 으로 분기
        self._fab_t = self._build_joint_index()
        self._syn_to_fab_idx = self._build_syn_to_fab_idx()
        n, dev = self.num_envs, self.device
        # 부모 `_reset_idx` 가 쓰는 **죽은 버퍼** — 제어 경로는 읽지 않는다(항상 홈·0).
        self.fabric_q = self.robot.data.default_joint_pos[:, self._fab_t].contiguous()
        self.fabric_qd = torch.zeros(n, int(self._fab_t.numel()), device=dev)
        self.fabric_qdd = torch.zeros_like(self.fabric_qd)
        # palm 박스·palm_targets·_home_palm — 부모 앵커/박스 부트스트랩·A 의 floor override 가 읽는다.
        d = math.pi / 180.0
        c = torch.tensor(p.palm_rot_center_deg, device=dev) * d
        h = float(p.palm_rot_half_deg) * d
        self._palm_lo = torch.cat([torch.tensor(p.palm_box_min, device=dev), c - h])
        self._palm_hi = torch.cat([torch.tensor(p.palm_box_max, device=dev), c + h])
        self.palm_targets = torch.zeros(n, 6, device=dev)
        self._home_palm = torch.zeros(6, device=dev)   # _init_home_palm 에서 실측
        # ---- B 의 팔 목표 버퍼 (N, n_arm) — `self.arm_ids` 순서, 클램프는 `_arm_lo/_arm_hi` ----
        self._arm_q_target = self.robot.data.default_joint_pos[:, self._arm_ids_t].clone()
        self._arm_cmd_step_raw = torch.zeros(n, device=dev)     # 클램프 전 |k·a| 평균(진단)
        self._prev_arm_q_target = self._arm_q_target.clone()    # 실현 스텝량 진단용(직전 목표)
        self._arm_limit_sat = torch.zeros(n, device=dev)        # 관절한계 클램프 비율(진단)
        _k, _a = float(self.cfg.k_arm), float(self.cfg.arm_ema)
        # 실효 slew = α·k_arm/dt (EMA 가 누적 목표에 걸려 스텝당 변화가 정확히 α·k·a) — cfg 가 대조했다.
        print(f"[grasp_fj] fabric OFF · 팔 = 관절 증분 k_arm={_k} rad/step · EMA α={_a} → "
              f"실효 포화 slew {_a * _k / self._policy_dt:.3f} rad/s "
              f"(선언 {float(self.cfg.arm_slew_rad_s)}) · 위치 목표만", flush=True)

    def _build_joint_index(self) -> torch.Tensor:
        """프로필 `fabric_joint_order` → articulation 인덱스(없으면 arm+hand 순서).

        왜 따로 만드나: mixin `_build_fabric_index` 는 `self.fabric.num_joints` 를 읽는다.
        여기서는 부모 `_reset_idx` 의 `fabric_q[env_ids] = q0[:, _fab_t]` 한 줄만 이 인덱스를 쓴다.
        """
        p = self.profile
        expect = int(p.num_arm_joints) + int(p.num_hand_joints)
        order = tuple(p.fabric_joint_order)
        if not order:
            return torch.cat([self._arm_ids_t, self._hand_ids_t])
        if len(order) != expect:
            raise RuntimeError(
                f"[{p.name}] fabric_joint_order 길이 {len(order)} != arm+hand {expect}")
        idx = []
        for name in order:
            ids, _ = self.robot.find_joints(name)
            if len(ids) != 1:
                raise RuntimeError(f"[{p.name}] 관절 '{name}' 해석 실패: {ids}")
            idx.append(ids[0])
        return torch.tensor(idx, device=self.device, dtype=torch.long)

    def _build_syn_to_fab_idx(self) -> torch.Tensor:
        """synergy 자세(프로필 순서) → `_fab_t` 손 구간 순서 — mixin 과 같은 이름 기반 매핑."""
        p = self.profile
        _syn_pos = {int(j): k for k, j in enumerate(self._syn_ids)}
        _fab_hand = self._fab_t[int(p.num_arm_joints):].tolist()
        _missing = [int(j) for j in _fab_hand if int(j) not in _syn_pos]
        if _missing:
            raise RuntimeError(
                f"[{p.name}] synergy 자세에 없는 손 관절 {_missing} — hand_joint_names 가 손 관절을 모두 덮어야 한다")
        return torch.tensor([_syn_pos[int(j)] for j in _fab_hand], device=self.device, dtype=torch.long)

    def _init_home_palm(self) -> None:
        """홈 텔레포트 + palm 실측 + 박스 검사. fabric FK 게이트는 없다(fabric 이 없다)."""
        q0 = self.robot.data.default_joint_pos
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        self.robot.set_joint_position_target(q0)
        self.scene.write_data_to_sim()
        for _ in range(2):                        # `__init__` 시점 body_pos_w 는 stale — 2스텝 뒤 읽는다
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
        home = self._palm_pose_6d()[0]
        self._home_palm = home.clone()
        self.palm_targets[:] = home.unsqueeze(0)
        # fabric 프레임이 없다 — palm 부기(앵커·마커)는 env-local 그 자체. 오프셋 0.
        self._fab_to_env = torch.zeros(3, device=self.device)
        out = (home < self._palm_lo) | (home > self._palm_hi)
        if bool(out.any()):
            raise RuntimeError(
                f"[{self.profile.name}] 홈 palm 이 워크스페이스 박스 밖이다: "
                f"home={[round(v, 3) for v in home.tolist()]}")
        print(f"[grasp_fj] 홈 palm={[round(v, 4) for v in home.tolist()]} (env-local · FK 게이트 없음)",
              flush=True)

    # ------------------------------------------------------------------
    # 액션 — 팔 관절 증분 + EMA (위치 목표만)
    # ------------------------------------------------------------------
    def _arm_command(self) -> None:
        """q_raw = clamp(q*_{t-1} + k·a, 한계) → q*_t = clamp(α·q_raw + (1−α)·q*_{t-1}, 한계)."""
        c = self.cfg
        n_arm = int(self.profile.num_arm_joints)
        step = float(c.k_arm) * self.actions[:, :n_arm]
        q_free = self._arm_q_target + step
        q_raw = q_free.clamp(self._arm_lo, self._arm_hi)
        alpha = float(c.arm_ema)
        self._arm_q_target = (alpha * q_raw + (1.0 - alpha) * self._arm_q_target).clamp(
            self._arm_lo, self._arm_hi)
        self._arm_cmd_step_raw = step.abs().mean(dim=1)
        self._arm_limit_sat = (q_raw != q_free).float().mean(dim=1)

    def _post_command(self) -> None:
        """no-op — fabric 이 없으니 손 상태 동기화·적분이 없다."""
        return None

    def _step_fabric(self) -> None:
        """no-op — 부모 훅 자리만 지킨다(누가 불러도 fabric 시간이 흐르지 않는다)."""
        return None

    def _apply_action(self) -> None:
        """decimation 마다. 팔: **위치 목표만**(속도 목표 없음). 손: mixin 412-415 그대로."""
        self.robot.set_joint_position_target(self._arm_q_target, joint_ids=self.arm_ids)
        # 손은 A 와 동일 경로 — A/B 대조에서 손이 변수가 되면 안 된다.
        self.robot.set_joint_position_target(self._syn_target, joint_ids=self._syn_ids)
        self.robot.set_joint_velocity_target(
            float(self.cfg.hand_velocity_ff_scale) * self._syn_vel,
            joint_ids=self._syn_ids)
        self._apply_gravity_compensation()

    # ------------------------------------------------------------------
    # 관측·로그·리셋
    # ------------------------------------------------------------------
    def _cmd_state(self) -> torch.Tensor:
        """정책의 마지막 팔 지령 상태 = q*_{t-1} (N, n_arm)."""
        return self._arm_q_target

    def _log_fabric_metrics(self) -> None:
        """fabric/* 대신 ctrl/* — 목표↔실측 관절 오차(sim2sim 정합 1차 지표)·요청량·한계 포화."""
        _jerr = (self._arm_q_target - self.robot.data.joint_pos[:, self._arm_ids_t]).abs()
        ex = self.extras
        ex["ctrl/joint_err_mean"] = _jerr.mean()
        ex["ctrl/joint_err_max"] = _jerr.max()          # 평균은 막힘 구간을 묻는다
        ex["ctrl/arm_cmd_step_raw"] = self._arm_cmd_step_raw.mean()
        ex["ctrl/arm_limit_sat"] = self._arm_limit_sat.mean()
        # ★09.07 목표가 **스텝당 실제로 얼마나 움직였나**(rad/step). 지령 요청량
        #   `arm_cmd_step_raw`(=k·|a|)는 EMA·클램프 전 값이라, 통과 후 남은 양을 따로 본다.
        #   실효 slew 판정: 이 값 × 60 Hz 가 URDF 한계(최저 5.445 rad/s)·브리지 상한과 비교된다.
        ex["ctrl/arm_target_step"] = (self._arm_q_target - self._prev_arm_q_target).abs().mean()
        self._prev_arm_q_target = self._arm_q_target.clone()

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)       # A: 목표·추적기·큐·외란 / 부모: 홈 텔레포트·시너지
        # 리셋은 홈 텔레포트라 q*_{-1} = 홈 q = 실측 q (DESIGN §1 B).
        self._arm_q_target[env_ids] = self._default_q[env_ids][:, self._arm_ids_t]
        self._prev_arm_q_target[env_ids] = self._arm_q_target[env_ids]   # 리셋 스텝을 큰 이동으로 세지 않는다
        self._arm_cmd_step_raw[env_ids] = 0.0
        self._arm_limit_sat[env_ids] = 0.0
