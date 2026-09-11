"""grasp_fj — Track B: 팔 7D 관절 증분(+EMA) + 손 20관절 full-joint(+EMA), **Fabrics 없음**.

`FJKeypointEnv`(Track A)를 상속해 팔·손 액션 어댑터에 해당하는 훅만 덮어쓴다(DESIGN §1 B,
`scratchpad/maps/control.md` §6-7 우회 목록). 목표열·관측·종료·지연·외란은 A 그대로, 보상은
포크(`fj_reward.py`, goal_bonus 한 항만 다름).

팔: `q*_t = clamp(q*_{t-1} + k_arm·a)` → `q*_t = α·q*_t + (1−α)·q*_{t-1}` → 위치 목표만
(`set_joint_velocity_target` 은 팔에 주지 않는다 — 실기 JTC 규약).
손(`hand_direct`, 09.08 사용자 확정 "SimToolReal 처럼 풀 조인트"): `raw = lo + ½(a+1)(hi−lo)` →
`q*_t = α·raw + (1−α)·q*_{t-1}` → clamp(lo, hi). [lo, hi] = soft limit ∩ 프로필 override
(테솔로 `_3/_4` 하한 0). 폐쇄도·램프·close_gate·blocked 는 없다. `hand_direct` 가 꺼지면 A 의 시너지.

★fabric 관련 부모 버퍼(`fabric_q/qd/qdd`, `_fab_t`, `_syn_to_fab_idx`, `palm_targets`,
  `_palm_lo/_hi`, `_home_palm`, `_fab_to_env`)는 부모의 리셋·앵커·박스 부트스트랩이 읽으므로
  **같은 모양으로 할당만** 한다. 제어 경로는 `_arm_q_target` 하나만 읽는다.
"""

from __future__ import annotations

import math

import torch

from ...modules.keypoint_goal import RisingCurriculum
from .fj_kp_env import FJKeypointEnv
from .fj_reward import compute_fj_reward
from .grasp_fj_env_cfg import GraspFJEnvCfg

# 액션 폭이 이보다 좁은 칸 = **설계상 고정 관절**(프로필이 ±0.01 규약으로 묶은 것).
# `hand_curl` 은 감쌈을 재는 값이라 감쌈에 기여할 수 없는 칸을 넣으면 안 된다 —
# 폭 0.02 짜리 칸은 0.01 rad 흔들림만으로 정규화가 0↔1 을 오가 잡음이 되기도 한다.
_CURL_MIN_SPAN_RAD = 0.05


class GraspFJEnv(FJKeypointEnv):
    cfg: GraspFJEnvCfg

    # ------------------------------------------------------------------
    # 부트스트랩 — fabric 자리에 관절 목표 버퍼
    # ------------------------------------------------------------------
    def _setup_fabrics(self) -> None:
        """mixin `_setup_fabrics` 의 fabric-free 판. 시너지·인덱스·palm 박스 할당은 동일."""
        p = self.profile
        self._setup_synergy()                     # ★`_syn_ids` 가 아래 인덱스보다 먼저(부모 순서 계약)
        # ★★09.11 감쌈 커리큘럼 — 성공의 전제조건(`_grasp_precondition`)이 읽는다.
        #   ★기준을 `hand_curl`(손 **자세**)에서 `wrap_frac`(물체 **포위**)으로 바꿨다.
        #     자세 기준은 물체가 손 **밖**에 있어도 통과한다 — 영상에서 컵이 평평한
        #     손가락 바깥에 얹혀 실려 가던 것이 정확히 그 구멍이다.
        #   start 0 이면 끔.
        _ws = float(self.cfg.grasp_wrap_start)
        self._wrap_cur = (RisingCurriculum(
            start=_ws, ceiling=float(self.cfg.grasp_wrap_max),
            factor=float(self.cfg.grasp_wrap_factor),
            interval=int(self.cfg.grasp_wrap_interval),
            success_threshold=float(self.cfg.grasp_wrap_threshold)) if _ws > 0.0 else None)
        self._wrap_last = torch.zeros(self.num_envs, device=self.device)
        self._wrap_stamp = -1
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
        # ★홈 주변 목표 상자는 여기서 **적용하지 않는다** — `_arm_lo/_arm_hi` 는 부모가
        #   `_setup_fabrics()` **뒤에** 만든다(grasp_s2r_env:231). 첫 `_arm_command` 에서 한 번
        #   교집합을 잡는다(둘 중 좁은 쪽이 이긴다 — 상자가 관절한계를 넓히는 일은 없다).
        self._arm_box_applied = False
        self._arm_cmd_step_raw = torch.zeros(n, device=dev)     # 클램프 전 |k·a| 평균(진단)
        self._prev_arm_q_target = self._arm_q_target.clone()    # 실현 스텝량 진단용(직전 목표)
        self._arm_limit_sat = torch.zeros(n, device=dev)        # 관절한계 클램프 비율(진단)
        self._prev_arm_action = torch.zeros(n, int(p.num_arm_joints), device=dev)   # 액션 1차 차분(벌점 측도)
        self._prev_syn_target = torch.zeros(n, len(self._syn_ids), device=dev)   # 손 목표 스텝량(진단)
        # ★09.08 낙하 sticky 플래그 — `done/fell` 은 판정선 0.15 < 상판 0.205 라 죽어 있다(학습분석 §6).
        #   `diag/drop_frac` 은 스텝 단면이라 "이 에피소드에서 한 번이라도 놓쳤나" 를 못 센다. 리셋에서만 풀린다.
        self._drop_sticky = torch.zeros(n, dtype=torch.bool, device=dev)
        self._start_ft_checked = False       # 시작 거리 부팅 가드(첫 리셋 뒤 한 번만)
        self._start_ft_last = torch.zeros((), device=dev)   # fresh env 없는 스텝의 진단값(직전값 유지)
        self._build_hand_action_range()
        # ★09.10 시작 자세는 **프로필이 소유하는 절대 관절값**이다(`arm_reset_joint_pos`).
        #   구 `cfg.arm_reset_offset_rad`(홈 기준 델타)는 출발 자세에 종속이라 자산 간 이식이
        #   불가능했다 — 사유는 프로필 필드 주석. 빈 튜플이면 홈에서 시작한다.
        _rq = tuple(self.profile.arm_reset_joint_pos)
        if _rq and len(_rq) != self.profile.num_arm_joints:
            raise RuntimeError(
                f"[{self.profile.name}] arm_reset_joint_pos 길이 {len(_rq)} "
                f"≠ num_arm_joints {self.profile.num_arm_joints}")
        self._arm_reset_q = (torch.tensor(_rq, device=dev, dtype=torch.float) if _rq else None)
        if self._arm_reset_q is not None:
            # ★홈은 프로필에서 읽는다 — `_default_q` 는 이 시점에 아직 없다(부모가 나중에 만든다).
            import re as _re
            _pat = _re.compile(str(self.profile.arm_joint_regex))
            _home = [float(v) for k, v in self.profile.init_joint_pos.items() if _pat.fullmatch(k)]
            _dl = ([round(a - b, 4) for a, b in zip(_rq, _home)]
                   if len(_home) == len(_rq) else "(홈 관절 추출 실패)")
            print(f"[grasp_fj] 팔 시작 자세(프로필 절대값) {[round(v, 4) for v in _rq]} rad · "
                  f"홈 대비 델타 {_dl}", flush=True)
        else:
            print(f"[grasp_fj] 팔 시작 자세 = **홈**(arm_reset_joint_pos 미설정) — "
                  f"시작 거리 가드가 대역을 검사한다", flush=True)
        _k, _a = float(self.cfg.k_arm), float(self.cfg.arm_ema)
        # 실효 slew = α·k_arm/dt (EMA 가 누적 목표에 걸려 스텝당 변화가 정확히 α·k·a) — cfg 가 대조했다.
        print(f"[grasp_fj] fabric OFF · 팔 = 관절 증분 k_arm={_k} rad/step · EMA α={_a} → "
              f"실효 포화 slew {_a * _k / self._policy_dt:.3f} rad/s "
              f"(선언 {float(self.cfg.arm_slew_rad_s)}) · 위치 목표만 · "
              f"환산 dofSpeedScale {_k / self._policy_dt:.3f}", flush=True)
        _hd, _hm = bool(self.cfg.hand_direct), float(self.cfg.hand_ema)
        _hlaw = (f"full-joint 선형 [lo,hi] + 관절 EMA α={_hm} (τ≈{self._policy_dt / _hm:.2f}s)" if _hd
                 else f"시너지 램프 {float(self.cfg.synergy_close_speed)}/step")
        _gc = int(self.cfg.goal_clock_restart_step)
        print(f"[grasp_fj] 손 = {_hlaw} · "
              f"vel_ff {float(self.cfg.hand_velocity_ff_scale)} · "
              f"스텝 예산 {'목표당(성공 시 시계→' + str(_gc) + ')' if _gc >= 0 else '에피소드당'} · "
              f"성공 {'연속' if bool(self.cfg.goal_force_consecutive) else '누적'}"
              f"{int(self.cfg.goal_success_steps)}회", flush=True)

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
        if not self._arm_box_applied:
            self._apply_arm_target_box()
        n_arm = int(self.profile.num_arm_joints)
        step = float(c.k_arm) * self.actions[:, :n_arm]
        q_free = self._arm_q_target + step
        q_raw = q_free.clamp(self._arm_lo, self._arm_hi)
        alpha = float(c.arm_ema)
        self._arm_q_target = (alpha * q_raw + (1.0 - alpha) * self._arm_q_target).clamp(
            self._arm_lo, self._arm_hi)
        self._arm_cmd_step_raw = step.abs().mean(dim=1)
        self._arm_limit_sat = (q_raw != q_free).float().mean(dim=1)
        # ★09.07 B-v: 벌점 측도 = 팔 액션 1차 차분 RMS / 2 ∈ [0, 1] (1.0 = 매 스텝 ±1 반전).
        #   B 는 증분+EMA 라 과지령이 없다(스텝당 목표 변화 = α·k·a) — 진동은 a 의 **반전**이다.
        #   리셋 직후는 직전 액션이 0 이라 차분이 |a| 로 튀므로 0.
        _da = self.actions[:, :n_arm] - self._prev_arm_action
        self._cmd_rate = torch.where(
            self.episode_length_buf == 0, torch.zeros_like(self._cmd_rate),
            0.5 * _da.pow(2).mean(dim=1).sqrt())
        self._prev_arm_action = self.actions[:, :n_arm].clone()

    def _apply_arm_target_box(self) -> None:
        """홈 주변 관절 목표 상자를 `_arm_lo/_arm_hi` 에 **한 번** 교집합으로 얹는다.

        ★왜 필요할 수 있나: 증분 매핑은 적분기라 복원력이 없다. 액션 부호가 잠깐 치우치면
          목표가 그 방향으로 계속 쌓여 관절 한계에 붙고, 반사 경계의 정상분포가 **관절 범위
          전체**가 된다 — 손이 물체에서 멀어진 채 돌아오지 못한다(RH56F1 실측 3연속:
          `ctrl/arm_limit_sat` 0 → 0.49 · `ft_dist` 0.13 → 0.69). A 의 palm 박스와 같은 역할이다.
        ★기본값 0 = 끔. tesollo B(fj_b1)는 상자 없이 학습됐으므로 켜는 트랙만 켠다.
        """
        self._arm_box_applied = True
        box = float(self.cfg.arm_target_box_rad)
        if box <= 0.0:
            return
        home = self._default_q[0, self._arm_ids_t]
        self._arm_lo = torch.maximum(self._arm_lo, (home - box).unsqueeze(0))
        self._arm_hi = torch.minimum(self._arm_hi, (home + box).unsqueeze(0))
        print(f"[grasp_fj] 팔 목표 상자 ±{box} rad (홈 기준) — 관절 한계와 교집합 · "
              f"폭 {[round(float(v), 3) for v in (self._arm_hi - self._arm_lo)[0]]}", flush=True)

    def _build_hand_action_range(self) -> None:
        """full-joint 손의 **관절별 액션한계** `[_act_lo, _act_hi]` 를 정한다(프로필 순서 = `_syn_ids`).

        `hand_direct` 가 꺼져 있으면 A 의 시너지 끝점(open→grip)을 그대로 둔다 — `grasp_fj_rh` 가 여기.

        켜져 있으면 SimToolReal(`reset_utils.py:52,59-62` — `joint_pos_limits` = URDF **하드** 한계)과 같은
        출발점에서 프로필 `hand_action_limit_override`(정규식 → (lo, hi)) 와 **교집합**만 취한다. 우리 `_syn_lo/_hi`
        는 soft 한계라 `soft_joint_pos_limit_factor` 1.0 일 때만 같다 — 부팅에서 하드 한계와 대조해 잠근다. 넓히기는 불가 —
        지령이 물리적으로 불가능한 곳을 가리키지 않는다. 왜 override 가 필요한가: SHARPA 는 URDF 에서
        PIP/DIP 하한이 0 이라 원시 한계 매핑이 안전했지만 테솔로 `_3/_4` 는 ±1.571 **대칭**이라 a=−1 이
        손등 −90° 를 지령한다. 접촉 항이 0개인 이 보상에서는 손등 갈고리 파지가 정상 파지와 같은
        점수를 받는다(08.23 실측 exploit). 하한 0 이 유일한 방어선이므로 부팅에서 세 가지를 죽인다:
          · 정규식이 아무 관절도 못 잡음(`_hand_mask`) — 오타가 조용히 "전폭" 으로 돌지 않게
          · 폭 0 액션 칸 — 20칸을 선언했으면 20칸이 다 무언가를 해야 한다
          · 리셋 자세(default_joint_pos)가 범위 밖으로 `hand_reset_clamp_max_rad` 넘게 벗어남 — 그 안이면
            리셋에서 관절 상태·EMA 시드를 한계로 clamp 해 심는다(`_hand_reset_q`, SimToolReal 처럼 EMA 가
            실측 자세에서 출발). 엄지 `_3` 의 −0.5(시너지 open) → 0 이 현재 유일한 사례
        ★관절 **이름**은 프로필이 소유한다 — 여기서는 정규식을 해석만 한다(계약: env 에 관절명 금지).
        """
        lo, hi = self._syn_lo.clone(), self._syn_hi.clone()
        if not bool(self.cfg.hand_direct):
            self._act_lo, self._act_hi = self._syn_open, self._syn_grip
            self._act_span = (self._syn_grip - self._syn_open).abs().clamp(min=1e-6)
            return
        _nm = list(self.profile.hand_joint_names)
        _hard = self.robot.data.joint_pos_limits[0, self._syn_ids, :]
        if not (torch.allclose(_hard[:, 0], lo, atol=1e-6) and torch.allclose(_hard[:, 1], hi, atol=1e-6)):
            raise RuntimeError(f"[{self.profile.name}] soft 한계 ≠ 하드(URDF) 한계 — soft_joint_pos_limit_factor 가 "
                               f"1.0 이 아니다. SimToolReal 은 하드 한계에 매핑한다")
        for regex, (olo, ohi) in dict(self.profile.hand_action_limit_override).items():
            m = self._hand_mask(regex)                     # 못 잡으면 여기서 죽는다
            if olo is not None:
                lo = torch.where(m, torch.maximum(lo, torch.full_like(lo, float(olo))), lo)
            if ohi is not None:
                hi = torch.where(m, torch.minimum(hi, torch.full_like(hi, float(ohi))), hi)
        span = hi - lo
        _dead = [n for n, w in zip(_nm, span.tolist()) if w <= 1e-6]
        if _dead:
            raise RuntimeError(f"[{self.profile.name}] 폭 0 액션 칸: {_dead} — override 가 한계를 뒤집었거나 "
                               f"soft limit 이 잠겨 있다")
        q0 = self.robot.data.default_joint_pos[0, self._syn_ids]
        self._hand_reset_q = q0.clamp(lo, hi)                     # B 리셋이 심는 손 자세(관절 상태 + EMA 시드)
        _mv = (self._hand_reset_q - q0).abs()
        _cl = [f"{n} {q:+.3f}→{c:+.3f}" for n, q, c, m in
               zip(_nm, q0.tolist(), self._hand_reset_q.tolist(), _mv.tolist()) if m > 1e-6]
        if float(_mv.max()) > float(self.cfg.hand_reset_clamp_max_rad):
            raise RuntimeError(f"[{self.profile.name}] 리셋 손 자세가 액션한계에서 {float(_mv.max()):.3f} rad 벗어난다"
                               f"(상한 {self.cfg.hand_reset_clamp_max_rad}): {_cl} — 범위와 다른 자세다")
        self._act_lo, self._act_hi, self._act_span = lo, hi, span
        # 전부 가동 관절이다 — `_hand_blocked`(진단)·`ctrl/hand_joint_err_max` 가 20칸을 다 본다.
        self._syn_movable = torch.ones_like(self._syn_movable)
        _narrow = int((lo > self._syn_lo + 1e-6).sum() + (hi < self._syn_hi - 1e-6).sum())
        print(f"[grasp_fj] 손 액션한계 full-joint — {len(_nm)}관절 · soft limit ∩ override "
              f"{len(self.profile.hand_action_limit_override)}규칙 · 좁힌 끝 {_narrow}개 · "
              f"리셋 clamp {len(_cl)}개 {_cl if _cl else '(없음)'}", flush=True)
        for n, a, b, q in zip(_nm, lo.tolist(), hi.tolist(), self._hand_reset_q.tolist()):
            print(f"           {n:18s} [{a:+.3f}, {b:+.3f}]  reset {q:+.3f}", flush=True)

    def _hand_mask(self, regex: str) -> torch.Tensor:
        """손 관절 정규식 → `_syn_ids` 순서의 bool 마스크 (n_hand,). 빈 문자열이면 전부 False.

        ★관절 **이름**은 프로필이 소유한다 — env 에 박으면 로봇이 바뀔 때 조용히 틀린다(계약).
          여기서는 프로필이 준 정규식을 articulation 으로 해석만 하고, 해석 실패는 시끄럽게 죽인다.
        """
        m = torch.zeros(len(self._syn_ids), dtype=torch.bool, device=self.device)
        if not regex:
            return m
        try:
            # IsaacLab resolve_matching_names 는 **re.fullmatch** 이고, 한 키도 못 잡으면 ValueError 를 던진다.
            ids, names = self.robot.find_joints(regex, preserve_order=False)
        except ValueError as e:
            raise RuntimeError(f"[{self.profile.name}] 손 관절 정규식 '{regex}' 이 아무것도 못 잡았다"
                               f"(fullmatch)") from e
        pos = {int(j): k for k, j in enumerate(self._syn_ids)}
        for j, nm in zip(ids, names):
            if int(j) not in pos:
                raise RuntimeError(f"[{self.profile.name}] '{regex}' 가 손 구간 밖의 관절 '{nm}' 을 잡았다")
            m[pos[int(j)]] = True
        return m

    def _seg_masks(self) -> dict[int, torch.Tensor]:
        """마디 번호(_1.._4) → `_syn_ids` 열 마스크. 첫 호출에 만들고 캐시한다.

        열 순서는 `hand_joint_names` 와 1:1 이다(`fj_core_control.py:475`). 이름 끝의
        `_<n>` 이 마디 번호이고, 없는 마디는 아예 키를 만들지 않는다(2지 그리퍼 등).
        """
        cached = getattr(self, "_seg_mask_cache", None)
        if cached is not None:
            return cached
        names = self.profile.hand_joint_names
        out: dict[int, torch.Tensor] = {}
        for seg in (1, 2, 3, 4):
            m = torch.tensor([n.endswith(f"_{seg}") for n in names], device=self.device)
            if bool(m.any()):
                out[seg] = m
        self._seg_mask_cache = out
        return out

    def _hand_targets(self, a_hand: torch.Tensor) -> torch.Tensor:
        """손 20관절 **full-joint** 한 스텝 — SimToolReal `action_utils.py:61-69` 와 같은 꼴.

            raw  = lo + ½(a+1)(hi−lo)                    관절별 절대 목표(액션한계에 선형)
            q*_t = α·raw + (1−α)·q*_{t-1}                관절 목표 EMA(handMovingAverage)
            q*_t = clamp(q*_t, lo, hi)

        `q*_{t-1}` 은 `_syn_target`(A 의 `_hand_command` 가 이 반환값을 거기 넣는다). 리셋은 부모가
        `_syn_target[env_ids] = q0` 로 심으므로 EMA 는 실측 자세에서 출발한다(SimToolReal
        `reset_utils.py:219` 와 동일). 폐쇄도·램프·close_gate·blocked 는 **없다**(09.08 사용자 확정).
        `_syn_close` 는 진단(`task/syn_close`)용 정규화 목표 (q*−lo)/(hi−lo) 로만 유지한다.

        `a_hand[:, j]` 가 관절 j 의 목표다 — 프로필 순서이고 `_syn_lo/_syn_hi/_syn_target` 과 **같은
        순서**라 이름 매핑이 필요 없다(슬라이스 순서가 어긋나 "기하는 완벽한데 접촉 0"으로 위장되는
        고전적 함정을 구조적으로 피한다).
        """
        if not bool(self.cfg.hand_direct):
            return super()._hand_targets(a_hand)
        lo, hi = self._act_lo.unsqueeze(0), self._act_hi.unsqueeze(0)
        raw = lo + 0.5 * (a_hand.clamp(-1.0, 1.0) + 1.0) * (hi - lo)
        alpha = float(self.cfg.hand_ema)
        tgt = (alpha * raw + (1.0 - alpha) * self._syn_target).clamp(lo, hi)
        self._syn_close = (tgt - lo) / self._act_span.unsqueeze(0)
        return tgt

    def _post_command(self) -> None:
        """no-op — fabric 이 없으니 손 상태 동기화·적분이 없다."""
        return None

    def _step_fabric(self) -> None:
        """no-op — 부모 훅 자리만 지킨다(누가 불러도 fabric 시간이 흐르지 않는다)."""
        return None

    def _apply_action(self) -> None:
        """decimation 마다. 팔: **위치 목표만**(속도 목표 없음). 손: mixin 412-415 그대로."""
        self.robot.set_joint_position_target(self._arm_q_target, joint_ids=self.arm_ids)
        # 손: actuator 경로(set_joint_position_target)만 A 와 공유한다. 법칙은 B 고유(`_hand_targets`),
        #   속도 FF 는 hand_direct 에서 0(검증기).
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

    def _action_obs(self) -> torch.Tensor:
        """관측 액션 블록(27): 팔 7 은 지연 액션 그대로, 손 20 은 **정규화 관절 목표** 2(q*−lo)/(hi−lo)−1.

        SimToolReal 은 raw action 이 아니라 post-EMA `prev_action_targets`(팔+손)를 관측한다
        (obs_utils.py:136-145 · action_utils.py:70-72). 팔은 `cmd_state` 가 이미 q*_{t-1}(7) 을 주므로
        그대로 두고, 손은 EMA 상태(τ≈10스텝)가 a_{t-1}·hand_q(1.5 N·m 약한 PD, 접촉 시 목표≠실측)로
        복원되지 않아 이 칸으로 준다 — 안 주면 MLP 는 구조적 부분관측, LSTM 은 적분을 학습해야 한다
        (09.08 리뷰). 폭 27 불변(계약 136). `hand_direct` 가 꺼지면 A 그대로.
        """
        if not bool(self.cfg.hand_direct):
            return self.actions
        off = self._hand_action_offset
        hand = 2.0 * (self._syn_target - self._act_lo.unsqueeze(0)) / self._act_span.unsqueeze(0) - 1.0
        return torch.cat([self.actions[:, :off], hand], dim=1)

    def _progress_reward(self, **kw):
        """★B 전용 보상(`fj_reward.py`). A 와 제어 방식이 달라 모듈을 공유하지 않는다.

        갈리는 것은 `goal_bonus` 한 항뿐이다 — A 는 near_goal 스텝마다 `goal_bonus/success_steps`,
        B 는 **성공 순간 1회 전액**(SimToolReal env.py:2656-2659 의 forceConsecutive 분기).
        `_get_rewards` 본체는 A 와 공유한다(계약상 덮을 수 없고, 덮을 이유도 없다).

        ★09.09 `hand_curl` 을 여기서 만들어 넘긴다. 부모 `_get_rewards` 는 계약 금지 훅이라
          인자를 추가할 수 없는데, 이 이음매는 `self` 를 갖는다 — 그래서 여기가 유일한 지점이다.
        """
        return compute_fj_reward(wrap_frac=self._wrap_frac_geom(), **kw)

    def _wrap_frac_geom(self) -> torch.Tensor:
        """마디들이 **물체 표면**에 얼마나 붙어 있나 (N,) ∈ (0,1] — 접촉 센서 없이 순수 기하.

        마디(`finger_sensor_bodies` 전체)마다 두 여유를 잰다:
          `e_xy` = 물체 축까지 수평거리 − 파지반경 R   (표면 안이면 0)
          `e_z`  = |마디 z − 물체 z| − 파지 반높이 H    (띠 안이면 0)
        그리고 `exp(-e_xy/τxy) · exp(-e_z/τz)` 의 마디 평균을 낸다.

        왜 중심이 아니라 표면인가: 중심 거리의 기울기는 표면 **법선**(벽을 밀어넣는 쪽)을
        가리킨다. 원통을 감싸는 것은 **접선** 방향이라 중심 거리로는 감쌈을 표현할 수 없다.
        R 을 빼면 표면에서 1.0 으로 포화해 밀어넣을 이득이 사라진다.

        ★왜 z 에는 기울기를 안 주나(09.11 사용자 확정): z 까지 당기면 마디가 전부 물체
          중간 높이로 모여 **인벨롭이 무너진다**. 띠(±H) 안에서 z 항은 정확히 1.0 이고,
          띠 밖으로 나간 양에만 감쇠가 걸린다 — 공중에 뜬 손을 막는 역할만 한다.
        ★exp 를 쓰는 이유: 선형 램프는 도달거리 밖에서 기울기가 **0** 이라 멀리 있는
          마디가 다가올 이유가 없다. exp 는 어디서나 기울기가 산다.
        """
        # ★★스텝당 1회 캐시. `_grasp_precondition` 이 `_progress_reward` **보다 먼저**
        #   돌기 때문에, 캐시가 없으면 전제조건이 한 스텝 **늦은** 값을 읽거나 같은
        #   스텝에 body_pos 를 두 번 읽는다. 스탬프로 둘 다 막는다.
        _now = int(self.common_step_counter)
        if getattr(self, "_wrap_stamp", -1) == _now:
            return self._wrap_last
        rc = self._rw_cfg
        p = (self.robot.data.body_pos_w[:, self._hull_all_t]
             - self.scene.env_origins[:, None, :])                       # (N, L, 3)
        o = self._env_local(self.object.data.root_pos_w).unsqueeze(1)     # (N, 1, 3)
        e_xy = ((p[..., :2] - o[..., :2]).norm(dim=-1)
                - self._obj_grasp_r.unsqueeze(1)).clamp(min=0.0)
        e_z = ((p[..., 2] - o[..., 2]).abs()
               - self._obj_grasp_h.unsqueeze(1)).clamp(min=0.0)
        self._wrap_e_xy, self._wrap_e_z = e_xy, e_z
        w = torch.exp(-e_xy / float(rc.wrap_tau_xy)) * torch.exp(-e_z / float(rc.wrap_tau_z))
        self._wrap_last = w.mean(dim=-1)
        self._wrap_stamp = _now
        return self._wrap_last

    def _hand_curl(self) -> torch.Tensor:
        """감쌈 정도 (N,) ∈ [0,1] — 뿌리 `_2` + 중간 `_3` 의 **실측** 정규화 관절각 평균.

        왜 이 두 마디인가(09.09 ep_3200 계측): 실현율이 `_1` 95% · `_2` 72% · `_3` 67% · `_4` 95%
        로, 끝마디는 시키는 대로 가고 감쌈에 필요한 두 마디만 막힌다. 끝만 굽은 손은 갈고리라
        컵과 손바닥 사이에 끼인다 — 감쌈을 재려면 **이 두 마디**를 봐야 한다.

        ★**실측**(`joint_pos`)이지 지령이 아니다. 지령에 주면 정책이 시키기만 하고 끝난다.
        ★clamp 는 액션한계 기준이다 — 접촉에 밀려 한계 밖으로 나간 관절이 1 을 넘겨
          보상을 부풀리지 않게 한다(09.09 엄지가 한계 밖 3.69 rad 까지 밀린 이력).
        """
        segs = self._seg_masks()
        m = None
        for k in (2, 3):
            if k in segs:
                m = segs[k] if m is None else (m | segs[k])
        if m is None:
            return torch.zeros(self.num_envs, device=self.device)
        # ★★09.11 — 설계상 고정된 칸을 뺀다. `thumb_2`(대향 −1.57)·`pinky_2`(0.0)는
        #   open==grip 이라 감쌈에 기여하지 않는데, 액션한계 정규화가 각각 0.420·0.000 에서
        #   1.000 으로 오를 여지를 줘서 **대향을 푸는 쪽이 curl 을 올리는** 경사를 만들었다.
        m = m & (self._act_span > _CURL_MIN_SPAN_RAD)
        if not bool(m.any()):
            return torch.zeros(self.num_envs, device=self.device)
        q = self.robot.data.joint_pos[:, self._syn_ids].clamp(self._act_lo, self._act_hi)
        closed = (q - self._act_lo.unsqueeze(0)) / self._act_span.unsqueeze(0)
        return closed[:, m].mean(dim=-1)

    def _restart_goal_clock(self) -> None:
        """성공한 env 의 에피소드 시계를 되돌린다 = **목표당** 스텝 예산.

        SimToolReal `env.py:2437-2439` 의 `progress_buf[is_success > 0] = 0` 과 같은 일이다.
        ★왜 하필 로그 훅에서 하나: time-out 술어는 불변 트랙(`grasp_s2r_env:1672`)이 소유하고
          `_get_dones` 는 B 의 계약 금지 훅이다. 호출 순서가
          `_get_rewards`(성공 확정) → `_log_step` → 이 메서드라, 성공이 확정된 **같은 스텝 안**에서
          시계를 되돌릴 수 있는 유일한 허용 지점이다.
        ★분기는 host 상수 하나뿐이라 per-step GPU 동기화가 없다. in-place 라 버퍼 재바인딩도 없다.
        """
        r = int(self.cfg.goal_clock_restart_step)
        if r < 0:
            return
        self.episode_length_buf.masked_fill_(self._success_now, r)
        self.extras["task/goal_clock_restart"] = self._success_now.float().mean()

    def _grasp_precondition(self, is_success):
        """★09.11 **인벨롭 그립을 성공의 전제조건으로** 건다(사용자 확정 + reward-audit).

        성공 = `kp_dist ≤ tol 연속 10회` **AND** `hand_curl ≥ curl_tol`.

        왜: goal_bonus 가 총점의 93.2% 인데 성공 술어에 손 자세가 안 들어가서, 정책이
        손끝을 물체 표면에서 ~50mm 띄운 채 성공을 받는 해에 수렴했다(ft_dist 90mm,
        설계 파지 대비 굴곡 41%). 보상은 epoch 300 이후 평평하다.

        ★임계는 **커리큘럼**이다(고정 아님). 설계 파지 0.83 을 바로 걸면 현재 0.34 라
          성공이 즉시 0 이 되어 총점의 93% 가 사라진다(reward-audit Check 4).
        ★`hand_curl` 은 **실측** 굴곡이다 — 지령에 걸면 시키기만 하고 끝난다.
        """
        if self._wrap_cur is None:
            return is_success
        return is_success & (self._wrap_frac_geom() >= self._wrap_cur.value)

    def _log_joint_limit_violation(self, hand_q, ex) -> None:
        """손 20관절이 **하드 한계 밖으로 밀려난 양**을 관절별로 남긴다.

        왜 필요한가 (09.10). 이탈은 09.08 부터 알려진 문제인데 **TB 에 지표가 없어서**
        판정을 매번 체크포인트 재생으로만 할 수 있었다. 그날 172개 태그를 다 뒤졌지만
        이탈을 답하는 것이 하나도 없었고, 대신 쓴 `ctrl/hand_joint_err_max` 는
        **전 env·전 관절의 최대값**이라 "얼마나 자주"를 못 말한다(중앙값을 비교하면
        전형적 동작이 아니라 최악 outlier 집단을 비교하게 된다).

        그래서 두 축을 나눠 남긴다.
          · `viol/frac`      — 얼마나 **자주** (전 env·전 관절 표본 중 이탈 비율)
          · `viol/max_rad`   — 얼마나 **크게** (최대 이탈량)
          · `viol/frac_<관절>` — **어느 손가락**이 무너지는가 (20칸)
        비율과 크기를 같이 봐야 "드물게 크게"와 "자주 조금"이 구분된다.

        기준은 `joint_pos_limits`(= USD 하드 한계)다. 분석 쪽에 한계를 손으로 박으면
        자산이 바뀔 때 조용히 어긋난다 — `fj_joint_limit_viol.py` 와 같은 규약.
        """
        lim = self.robot.data.joint_pos_limits[0, self._syn_ids, :]
        lo, hi = lim[:, 0].unsqueeze(0), lim[:, 1].unsqueeze(0)
        viol = torch.maximum(torch.maximum(lo - hand_q, hand_q - hi),
                             torch.zeros_like(hand_q))          # (N, 20) rad, 안이면 0
        over = (viol > 1e-3).float()
        ex["viol/frac"] = over.mean()
        ex["viol/max_rad"] = viol.max()
        per = over.mean(dim=0)
        for _n, _v in zip(self.profile.hand_joint_names, per):
            ex[f"viol/frac_{_n}"] = _v

    def _log_fabric_metrics(self) -> None:
        """fabric/* 대신 ctrl/* — 목표↔실측 관절 오차(sim2sim 정합 1차 지표)·요청량·한계 포화."""
        self._restart_goal_clock()      # ★성공 스텝 안에서 — 아래 지표보다 먼저(같은 스텝 의미)
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
        # ★09.08 손 건강 3종 — **진단 전용**, 아무것도 얼리지 않는다(법칙에 게이트·hold 없음). `task/hand_*` 는
        #   불변 트랙의 진단 훅에 사는데 그 훅은 접촉 센서 소비자라 이 트랙에서 계약 금지다. 그래서 여기서 잰다.
        #   `hand_blocked_frac` = 목표↔실측 > 1.0 rad ∧ 한계 밖 아님(접촉 or 추종 실패). 20관절 전부 대상
        #   (외전 포함)이라 b9 의 14관절 값과 직접 비교 불가.
        _hq = self.robot.data.joint_pos[:, self._syn_ids]
        _herr = (self._syn_target - _hq).abs()
        ex["ctrl/hand_joint_err_max"] = _herr[:, self._syn_movable].max()
        ex["ctrl/hand_blocked_frac"] = self._hand_blocked().float().mean()
        self._log_joint_limit_violation(_hq, ex)
        # ★09.11 감쌈 계측 — 커리큘럼이 조여지는지, 정책이 따라오는지 둘 다 봐야 한다
        #   (reward-audit Check 5: 임계를 거는 값은 반드시 직접 로깅한다).
        # `hand_curl` 은 전제조건에서 내려왔지만 **진단으로는 남긴다** — 감쌈이 안 오를 때
        #   "손을 안 굽혀서"인지 "손은 굽었는데 물체가 밖에 있어서"인지 갈라준다.
        ex["task/hand_curl"] = self._hand_curl().mean()
        if self._wrap_cur is not None:
            ex["task/wrap_tol"] = torch.tensor(self._wrap_cur.value, device=self.device)
            ex["task/wrap_pass"] = (self._wrap_last >= self._wrap_cur.value).float().mean()
        # ★09.11 감쌈 기하 3종 — 하나로 뭉치면 "왜 안 감싸는지"를 못 가른다.
        #   `wrap_frac` 이 낮을 때 `wrap_xy_mean` 이 크면 **반경 실패**(손이 물체 곁에
        #   못 간다), `wrap_band_frac` 이 낮으면 **높이 실패**(손이 파지 띠를 벗어나 있다).
        _we = getattr(self, "_wrap_e_xy", None)
        if _we is not None:
            ex["task/wrap_frac"] = self._wrap_last.mean()
            ex["task/wrap_xy_mean"] = _we.mean()
            ex["task/wrap_band_frac"] = (self._wrap_e_z <= 0.0).float().mean()
        # ★09.11 — 고정 칸이 실제로 고정돼 있는가. 액션한계로 묶었으므로 지령은 못 벗어나지만
        #   **접촉은 관절을 밀어낼 수 있다**(thumb_1 이 27배 포화로 밀려난 전례). 설계값은
        #   액션창의 중점이다 — 이 값이 커지면 파지가 엄지 대향을 물리적으로 잃고 있다는 뜻.
        _pin = self._act_span <= _CURL_MIN_SPAN_RAD
        if bool(_pin.any()):
            _mid = 0.5 * (self._act_lo + self._act_hi)
            _dev = (self.robot.data.joint_pos[:, self._syn_ids] - _mid.unsqueeze(0)).abs()
            ex["task/pinned_dev_max"] = _dev[:, _pin].max()
        # ★★09.09 **실측 폐쇄도**. `task/syn_close` 는 (tgt−lo)/span 즉 **지령**이라
        #   "정책이 안 닫는다"와 "손이 못 닫는다"를 3200 epoch 동안 구분하지 못했다.
        #   ep_3200 재생 계측: 지령 0.540 vs 실측 0.452 — 마디별 실현율이
        #   `_1` 95% · `_2` 72% · `_3` 67% · `_4` 95% 로, 감쌈에 필요한 뿌리·중간만 막힌다.
        #   그 격차가 인벨롭 파지 실패의 절반이므로 **로그에 실측을 남긴다**.
        _closed = ((_hq.clamp(self._act_lo, self._act_hi) - self._act_lo.unsqueeze(0))
                   / self._act_span.unsqueeze(0))
        ex["task/syn_close_actual"] = _closed.mean()
        # 실현율 = 실측/지령. 1.0 이면 시킨 대로 다 간 것이고, 낮으면 접촉에 막힌 것이다.
        _cmd = self._syn_close.mean().clamp_min(1e-6)
        ex["task/syn_realized_frac"] = _closed.mean() / _cmd
        # 마디별(_1 벌림 · _2 뿌리 · _3 중간 · _4 끝) — 어느 마디가 막히는지가 처방을 가른다.
        #   ★열 순서는 `_syn_ids = [jn.index(nm) for nm in p.hand_joint_names]` 라
        #     `hand_joint_names` 와 1:1 이다(관절번호-major 인 articulation 순서가 아니다).
        for _seg, _m in self._seg_masks().items():
            ex[f"task/syn_close_actual_seg{_seg}"] = _closed[:, _m].mean()
            ex[f"task/syn_close_cmd_seg{_seg}"] = self._syn_close[:, _m].mean()
        ex["ctrl/hand_target_step"] = (self._syn_target - self._prev_syn_target).abs().mean()
        self._prev_syn_target = self._syn_target.clone()
        # ★09.08 파지 품질 1순위 지표 — 들었다가 놓친 에피소드 비율(sticky). `_latched` 는 `_get_rewards`
        #   가 이 스텝에 갱신한 값이다(호출 순서 _get_rewards → _log_step → 여기).
        _obj = self._env_local(self.object.data.root_pos_w)
        _dz = _obj[:, 2] - self.object_spawn_pos[:, 2]
        self._drop_sticky |= self._latched & (_dz < 0.03)
        ex["ctrl/drop_sticky_frac"] = self._drop_sticky.float().mean()
        # 커리큘럼 게이트 입력 그 자체(mean prev_episode_successes ≥ tol_success_threshold) — `task/successes_mean`
        # 은 에피소드 내 러닝 카운트라 대체가 안 된다. 게이트까지의 거리가 대시보드에 보이게 한다.
        ex["ctrl/prev_ep_successes_mean"] = self._trk.prev_episode_successes.float().mean()
        # ★★09.10 시작 거리 가드 — 기준을 **손바닥 중심**으로 바꿨다(사용자 지시).
        #   구 판본은 손끝 5개의 물체까지 거리 **평균**을 봤다. 그건 팔 위치를 재는 척하면서
        #   실제로는 **손 자세를 잰다** — 같은 팔 자세에서 손가락 굽힘만 0→0.5→0.9 rad 로
        #   바꾸면 98.8 → 67.3 → 82.5 mm 로 요동친다(09.10 FK 실측). 손바닥은 모든 손가락
        #   관절의 **상류**라 같은 조건에서 150.4 mm 로 불변이다.
        #   `arm_reset_joint_pos` 가 이 홈/자산에서 푼 값인지가 이 가드의 질문이므로,
        #   답이 손 자세에 흔들리면 안 된다.
        #   ★스텝당 host 동기화 0 — 마스크 곱·합으로 GPU 에 두고, host 판단은 부팅 직후만.
        _fresh = (self.episode_length_buf <= 1).float()
        _palm = self.robot.data.body_pos_w[:, self.palm_idx] - self.scene.env_origins
        _pd = (_palm - _obj).norm(dim=-1)                       # (N,) 손바닥→물체 중심
        _nf_t = _fresh.sum()
        _start = (_pd * _fresh).sum() / _nf_t.clamp(min=1.0)
        self._start_ft_last = torch.where(_nf_t > 0, _start, self._start_ft_last)
        ex["ctrl/start_palm_dist"] = self._start_ft_last
        # ★09.10 **무조건** 검사한다. 구 판본은 `self._arm_reset_off is not None` 을 전제로 걸어,
        #   시작 자세가 미설정이라 거리가 틀린 **바로 그 경우를 건너뛰었다** — dg5f-m-short 가
        #   243.4 mm 로 부팅해 3 iter 를 돌고 exit 0 으로 끝났다(09.10). 결과를 보는 가드가
        #   수단(오프셋 존재)에 조건을 걸면 안 된다. 원인 구분은 **메시지**가 한다.
        if not self._start_ft_checked and self.common_step_counter <= 4:
            _nf = int(_nf_t)
            if _nf >= min(64, self.num_envs):
                self._start_ft_checked = True
                _lo = float(self.cfg.start_palm_dist_band_m[0])
                _hi = float(self.cfg.start_palm_dist_band_m[1])
                if not (_lo <= float(_start) <= _hi):
                    _why = ("`arm_reset_joint_pos` 가 이 프로필에 **미설정**이라 홈에서 시작한다 — "
                            "이 팔로 IK 를 풀어 채울 것"
                            if self._arm_reset_q is None else
                            "`arm_reset_joint_pos` 가 이 홈/자산에서 푼 값이 아니다")
                    raise RuntimeError(
                        f"[{self.profile.name}] 리셋 직후 **손바닥**→물체 {float(_start) * 1e3:.1f} mm 가 "
                        f"대역 [{_lo * 1e3:.0f}, {_hi * 1e3:.0f}] mm 밖이다 — {_why}")
                print(f"[grasp_fj] 시작 거리 가드 ✓ 손바닥→물체 {float(_start) * 1e3:.1f} mm "
                      f"({_nf} env, 대역 {_lo * 1e3:.0f}~{_hi * 1e3:.0f})", flush=True)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)       # A: 목표·추적기·큐·외란 / 부모: 홈 텔레포트·시너지
        # 리셋은 홈 텔레포트라 q*_{-1} = 홈 q = 실측 q (DESIGN §1 B).
        self._arm_q_target[env_ids] = self._default_q[env_ids][:, self._arm_ids_t]
        self._prev_arm_q_target[env_ids] = self._arm_q_target[env_ids]   # 리셋 스텝을 큰 이동으로 세지 않는다
        self._arm_cmd_step_raw[env_ids] = 0.0
        self._arm_limit_sat[env_ids] = 0.0
        self._prev_arm_action[env_ids] = 0.0
        # ★09.08 고정 리셋 오프셋 — 손을 물체 쪽으로 옮겨 **시작 거리**를 SimToolReal 과 맞춘다.
        #   그들 팔 속도(0.15 rad/s)는 손이 물체 옆(104mm)에서 시작하는 것과 한 묶음이다.
        #   ★컵 상대가 아니라 고정값이다(08.18: 컵 참값 텔레포트는 실기 재현 불가라 폐기).
        #   부모가 홈으로 텔레포트한 **뒤에** 관절 상태를 덮어써야 실측 q 와 지령 q* 가 일치한다.
        if self._arm_reset_q is not None:
            _q = self.robot.data.joint_pos[env_ids].clone()
            _q[:, self._arm_ids_t] = torch.clamp(
                self._arm_reset_q.unsqueeze(0).expand(len(env_ids), -1),
                self._arm_lo[env_ids], self._arm_hi[env_ids])
            self.robot.write_joint_state_to_sim(_q, torch.zeros_like(_q), env_ids=env_ids)
            self._arm_q_target[env_ids] = _q[:, self._arm_ids_t]
            self._prev_arm_q_target[env_ids] = self._arm_q_target[env_ids]
        # ★손: 부모가 `_syn_target[env_ids] = q0`(프로필 리셋 자세)로 심었다. full-joint 는 그 자세를 액션한계로
        #   clamp 한 `_hand_reset_q`(엄지 `_3` −0.5 → 0) 를 **관절 상태와 EMA 시드 둘 다**에 심는다 — SimToolReal
        #   `prev_targets = joint_pos` 처럼 EMA 가 실측 자세에서 출발하고, 첫 스텝에 clamp 로 튀지 않는다.
        if bool(self.cfg.hand_direct):
            _qh = self.robot.data.joint_pos[env_ids].clone()
            _qh[:, self._syn_ids] = self._hand_reset_q.unsqueeze(0)
            self.robot.write_joint_state_to_sim(_qh, torch.zeros_like(_qh), env_ids=env_ids)
            self._syn_target[env_ids] = self._hand_reset_q.unsqueeze(0)
            self._syn_close[env_ids] = ((self._syn_target[env_ids] - self._act_lo.unsqueeze(0))
                                        / self._act_span.unsqueeze(0))
        self._prev_syn_target[env_ids] = self._syn_target[env_ids]
        self._drop_sticky[env_ids] = False
