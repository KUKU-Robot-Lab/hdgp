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
from .fj_reward import compute_fj_reward
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
        # ★홈 주변 목표 상자는 여기서 **적용하지 않는다** — `_arm_lo/_arm_hi` 는 부모가
        #   `_setup_fabrics()` **뒤에** 만든다(grasp_s2r_env:231). 첫 `_arm_command` 에서 한 번
        #   교집합을 잡는다(둘 중 좁은 쪽이 이긴다 — 상자가 관절한계를 넓히는 일은 없다).
        self._arm_box_applied = False
        self._arm_cmd_step_raw = torch.zeros(n, device=dev)     # 클램프 전 |k·a| 평균(진단)
        self._prev_arm_q_target = self._arm_q_target.clone()    # 실현 스텝량 진단용(직전 목표)
        self._arm_limit_sat = torch.zeros(n, device=dev)        # 관절한계 클램프 비율(진단)
        self._prev_arm_action = torch.zeros(n, int(p.num_arm_joints), device=dev)   # 액션 1차 차분(벌점 측도)
        self._prev_syn_target = torch.zeros(n, len(self._syn_ids), device=dev)   # 손 목표 스텝량(진단)
        self._build_hand_action_range()
        _ro = tuple(self.cfg.arm_reset_offset_rad)
        self._arm_reset_off = (torch.tensor(_ro, device=dev, dtype=torch.float) if _ro else None)
        if self._arm_reset_off is not None:
            print(f"[grasp_fj] 팔 리셋 **고정** 오프셋 {[round(v, 4) for v in _ro]} rad "
                  f"(홈 기준 · 컵 상대 아님) — 시작 거리를 SimToolReal 과 맞춘다", flush=True)
        _k, _a = float(self.cfg.k_arm), float(self.cfg.arm_ema)
        # 실효 slew = α·k_arm/dt (EMA 가 누적 목표에 걸려 스텝당 변화가 정확히 α·k·a) — cfg 가 대조했다.
        print(f"[grasp_fj] fabric OFF · 팔 = 관절 증분 k_arm={_k} rad/step · EMA α={_a} → "
              f"실효 포화 slew {_a * _k / self._policy_dt:.3f} rad/s "
              f"(선언 {float(self.cfg.arm_slew_rad_s)}) · 위치 목표만 · "
              f"환산 dofSpeedScale {_k / self._policy_dt:.3f}", flush=True)
        _hd, _hm = bool(self.cfg.hand_direct), float(self.cfg.synergy_close_ema)
        _hlaw = (f"EMA α={_hm} (τ≈{self._policy_dt / _hm:.2f}s)" if _hm > 0.0
                 else f"램프 {float(self.cfg.synergy_close_speed)}/step "
                      f"({1.0 / float(self.cfg.synergy_close_speed) * self._policy_dt:.2f}s 완전폐쇄)")
        _gc = int(self.cfg.goal_clock_restart_step)
        print(f"[grasp_fj] 손 = {'20관절 독립' if _hd else '시너지'} · 폐쇄 {_hlaw} · "
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
        """폐쇄도 [0,1] 이 매핑되는 관절 **끝점**(`_act_lo`/`_act_hi`)을 정한다.

        `synergy`(기본) = open→grip 그대로. A 와 `grasp_fj_rh` 가 여기 머문다.

        `per_role` = **폭 0 으로 묶인 관절만** 관절 한계로 푼다. 판별은 이름이 아니라
        `|grip − open| ≈ 0` 이다 — 로봇 관절명을 env 에 박지 않기 위해서다(계약 테스트).
        tesollo_right 실측(09.08)으로 여섯 개가 걸린다: 다섯 손가락 `_1` 외전 + pinky `_2`.
        하드웨어는 ±0.4~0.6 rad 를 낼 수 있는데 액션 슬롯이 죽어 있었다(20칸 중 가동 14칸).

        ★굴곡 관절은 왜 그대로 두나: open→grip 이 이미 한계와 실질적으로 같다. index_3 은
          [0, 1.8] 을 지령하고 soft limit 1.571 이 흡수하니 도달집합이 [max(0,lo), hi] 와 같다.
          raw 한계로 풀면 하한이 −1.571 이 되어 a=−1 이 **손등 −90°** 를 지령하는데, 접촉 항이
          0개인 이 보상에서는 손등 갈고리 파지가 정상 파지와 같은 점수를 받는다.
        ★푼 관절은 `close_gate` 를 **면제**한다: 게이트는 "정렬 전에 오므리지 마라"는 밸브인데
          외전은 오므림이 아니라 **접근 중에 바꿔야 하는 손 모양**이다. 옆에서 접근할 때
          손을 벌리는 것이 바로 이 판에서 풀어주려는 자유도다. `blocked` hold 는 그대로 건다
          (외전이 물체에 막히면 미는 것을 멈추는 게 맞다).
        ★푼 관절은 폐쇄도 0 이 중립이 **아니다**(한쪽 끝이다). 리셋 폐쇄도를 open 자세에
          대응하는 값(`_close_home`)으로 심는다 — 안 그러면 손이 벌어진 채로 시작한다.
        """
        lo, hi, op, gr = self._syn_lo, self._syn_hi, self._syn_open, self._syn_grip
        frozen = (gr - op).abs() <= 1e-4
        if str(self.cfg.hand_range_mode) != "per_role":
            self._act_lo, self._act_hi = op, gr
            self._shape_mask = torch.zeros_like(frozen)
            self._close_home = torch.zeros_like(op)
            return
        wide = self._hand_mask(self.profile.hand_wide_shape_joint_regex)
        freed = frozen
        span = torch.where(wide, torch.full_like(op, float(self.cfg.hand_wide_span_rad)),
                           torch.full_like(op, float(self.cfg.hand_shape_span_rad)))
        self._act_lo = torch.where(freed, torch.maximum(op - span, lo), op)
        self._act_hi = torch.where(freed, torch.minimum(op + span, hi), gr)
        self._shape_mask = freed
        # ★굴곡은 open→grip 밖으로 절대 안 나간다 — `_3`/`_4` 가 ±90° **대칭**이라
        #   하한을 풀면 a=−1 이 손등 −90° 를 지령한다. SHARPA 는 URDF 하한이 0(PIP·DIP)이라
        #   전 범위 매핑이 안전했지만 테솔로는 기구적으로 대칭이다 — 같은 매핑을 쓰면 안 된다.
        _flex = ~freed
        if not torch.equal(self._act_lo[_flex], op[_flex]) or not torch.equal(self._act_hi[_flex], gr[_flex]):
            raise RuntimeError("굴곡 관절의 액션 범위가 open→grip 을 벗어났다 — 손등 과신전이 열린다")
        _w = (self._act_hi - self._act_lo).clamp(min=1e-6)
        self._close_home = torch.where(freed, (op - self._act_lo) / _w, torch.zeros_like(op))
        # 푼 관절도 blocked 판정 대상이다 — 안 넣으면 외전이 막혀도 계속 민다.
        self._syn_movable = self._syn_movable | freed
        self._syn_close[:] = self._close_home.unsqueeze(0)
        _nm = list(self.profile.hand_joint_names)
        _fr = [n for n, f in zip(_nm, freed.tolist()) if f]
        _wd = [n for n, f in zip(_nm, (freed & wide).tolist()) if f]
        _dead = int((self._act_hi - self._act_lo).abs().le(1e-6).sum())
        _mflex = float((self._act_lo[_flex] - lo[_flex]).min()) if bool(_flex.any()) else 0.0
        print(f"[grasp_fj] 손 범위 per_role — 벌림 {len(_fr)}개 해방(그중 넓게 {len(_wd)}) · "
              f"가동 {int(self._syn_movable.sum())}/{len(op)} · **죽은 액션 칸 {_dead}개** · "
              f"반폭 좁 {self.cfg.hand_shape_span_rad}/넓 {self.cfg.hand_wide_span_rad} rad · "
              f"굴곡 하한 여유 {_mflex:+.3f} rad(과신전 차단) · close_gate 면제\n"
              f"           해방={_fr} · 넓게={_wd}", flush=True)

    def _hand_mask(self, regex: str) -> torch.Tensor:
        """손 관절 정규식 → `_syn_ids` 순서의 bool 마스크 (n_hand,). 빈 문자열이면 전부 False.

        ★관절 **이름**은 프로필이 소유한다 — env 에 박으면 로봇이 바뀔 때 조용히 틀린다(계약).
          여기서는 프로필이 준 정규식을 articulation 으로 해석만 하고, 해석 실패는 시끄럽게 죽인다.
        """
        m = torch.zeros(len(self._syn_ids), dtype=torch.bool, device=self.device)
        if not regex:
            return m
        ids, names = self.robot.find_joints(regex, preserve_order=False)
        if not ids:
            raise RuntimeError(f"[{self.profile.name}] 손 관절 정규식 '{regex}' 이 아무것도 못 잡았다")
        pos = {int(j): k for k, j in enumerate(self._syn_ids)}
        for j, nm in zip(ids, names):
            if int(j) not in pos:
                raise RuntimeError(f"[{self.profile.name}] '{regex}' 가 손 구간 밖의 관절 '{nm}' 을 잡았다")
            m[pos[int(j)]] = True
        return m

    def _closure_to_target(self, c: torch.Tensor) -> torch.Tensor:
        """폐쇄도 (N, n_hand) → 관절 목표. 끝점은 `_build_hand_action_range` 가 정한다."""
        tgt = torch.lerp(self._act_lo.unsqueeze(0), self._act_hi.unsqueeze(0), c)
        return tgt.clamp(self._syn_lo.unsqueeze(0), self._syn_hi.unsqueeze(0))

    def _hand_targets(self, a_hand: torch.Tensor) -> torch.Tensor:
        """폐쇄도 한 스텝(`_hand_step`) → **선택적 EMA** → 관절 목표.

        ★`synergy_close_ema` 가 0 이면 `_hand_step` 결과를 그대로 반환한다 — 오늘과 한 글자도
          다르지 않다. > 0 이면 SimToolReal 의 `handMovingAverage`(α 0.1) 를 재현한다.
        ★왜 관절이 아니라 **폐쇄도**에 거나: `tgt = lerp(open, grip, c)` 가 c 에 아핀이라
          `c ← α·cmd + (1−α)·c` 는 관절공간 EMA 와 **항등**이다. 범위만 원시 관절한계가 아니라
          보정된 open→grip 인데, 그것이 `_hand_step` 이 설명하는 의도한 divergence 다.
        ★그래서 `close_gate`·`blocked` 가 **그대로 산다**: 둘은 증분(delta)에 곱/영치기로 걸리고
          EMA 도 delta 에 곱하는 양수 스칼라라 순서가 교환된다(α·(g·d) = g·(α·d), α·0 = 0).
        ★단, 목표가 최대 20배 빨리 움직이므로 자유공간에서 손 PD 가 못 따라가 `blocked` 임계
          (1.0 rad)를 오발할 수 있다 — `ctrl/hand_blocked_frac`·`ctrl/hand_joint_err_max` 로 본다.
        """
        c0 = self._syn_close.clone()
        tgt = self._hand_step(a_hand)
        m = float(self.cfg.synergy_close_ema)
        if m <= 0.0:
            return tgt
        # ★폐쇄도에 EMA 를 건다. tgt = lerp(open, grip, c) 가 c 에 **아핀**이므로
        #   c ← α·cmd + (1−α)·c 는 관절공간 EMA(SimToolReal action_utils) 와 **항등**이다.
        #   `close_gate`·`blocked` 는 증분(delta)에 곱/영치기로 걸리고 EMA 도 delta 에 곱하는
        #   양수 스칼라라 **순서가 교환된다**(α·(g·d) = g·(α·d), α·0 = 0) — 둘 다 그대로 산다.
        self._syn_close = c0 + m * (self._syn_close - c0)
        return self._closure_to_target(self._syn_close)

    def _hand_step(self, a_hand: torch.Tensor) -> torch.Tensor:
        """손 20관절 **독립** 절대 폐쇄도 한 스텝. 결합만 없애고 안전장치는 그대로 둔다.

        EMA 는 호출자(`_hand_targets`)가 건다 — 여기는 램프·게이트·blocked 까지다.

        `a_hand[:, j]` 가 관절 j 의 폐쇄도 목표다 — 프로필 순서이고 `_syn_close`/`_syn_open`/
        `_syn_grip` 과 **같은 순서**라 이름 매핑이 필요 없다(슬라이스 순서가 어긋나 "기하는 완벽한데
        접촉 0"으로 위장되는 고전적 함정을 구조적으로 피한다).

        ★남기는 것과 이유:
          · `synergy_close_speed` 폐쇄 속도 상한 — 스윕 축이다(fj_c1/c2/c3). EMA 를 켜면 cfg 가
            이 값을 ≥ 1.0 으로 요구해 clamp 가 항등이 된다(램프 무력화 증명).
          · `close_gate` — 정렬 전 폐쇄를 막는다. 푸는 방향은 항상 통과(잘못 오므린 상태 탈출).
          · `blocked` hold — 접촉 항이 **0개**인 이 보상에서 감쌈을 만드는 유일한 장치다.
            "막힐 때까지 민다"라 형상 무관이고 센서가 필요 없다.
        ★★범위가 원시 관절한계가 아니라 보정된 open→grip 인 이유:
          `_3`/`_4` 가 ±90° 대칭이라 원시 한계로 매핑하면 `a=−1` 이 **손등 −90°** 를 지령한다.
          접촉 보상이 0개인 이 트랙에서는 손등 갈고리 파지가 정상 파지와 **같은 점수**를 받고,
          과거에 쓴 대책(`require_palmar_contact`)은 이 트랙에서 계약상 금지다(ContactSensor 미생성).
          grip 자세는 한계를 1.8 까지 넘겨 지령하도록 보정돼 있어 파지력도 그대로 보존된다.
        """
        if not bool(self.cfg.hand_direct):
            return super()._hand_targets(a_hand)
        cmd_j = 0.5 * (a_hand.clamp(-1.0, 1.0) + 1.0)          # 관절별 절대 폐쇄도 [0,1]
        rate = float(self.cfg.synergy_close_speed)
        delta = (cmd_j - self._syn_close).clamp(-rate, rate)
        # ★shape 관절(폭 0 이던 외전)은 게이트 면제 — 게이트는 "정렬 전 오므림" 밸브인데
        #   외전은 오므림이 아니라 접근 중에 바꿔야 하는 손 모양이다(per_role 일 때만 존재).
        _g = self._close_gate.unsqueeze(1).expand_as(delta)
        _g = torch.where(self._shape_mask.unsqueeze(0), torch.ones_like(_g), _g)
        delta = torch.where(delta > 0.0, delta * _g, delta)
        _blk = torch.zeros_like(delta, dtype=torch.bool)
        _blk[:, self._syn_movable] = self._hand_blocked()
        delta = torch.where(_blk & (delta > 0.0), torch.zeros_like(delta), delta)
        self._syn_close = (self._syn_close + delta).clamp(0.0, 1.0)
        return self._closure_to_target(self._syn_close)

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

    def _progress_reward(self, **kw):
        """★B 전용 보상(`fj_reward.py`). A 와 제어 방식이 달라 모듈을 공유하지 않는다.

        갈리는 것은 `goal_bonus` 한 항뿐이다 — A 는 near_goal 스텝마다 `goal_bonus/success_steps`,
        B 는 **성공 순간 1회 전액**(SimToolReal env.py:2656-2659 의 forceConsecutive 분기).
        `_get_rewards` 본체는 A 와 공유한다(계약상 덮을 수 없고, 덮을 이유도 없다).
        """
        return compute_fj_reward(**kw)

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
        ex["ctrl/arm_action_rate_lifted"] = self._lifted_mean(self._cmd_rate)   # A 의 task/cmd_rate_lifted 와 같은 측도
        # ★09.08 손 건강 3종. `task/hand_blocked_frac`·`task/hand_overdrive` 는 불변 트랙의 진단
        #   훅에 사는데 그 훅은 접촉 센서 소비자라 이 트랙에서 계약 금지다. 그래서 여기서 잰다.
        #   용도: 폐쇄가 빨라졌을 때 `blocked` 임계(1.0 rad)를 **자유공간에서** 오발하는지 —
        #   그러면 폐쇄가 중간에 얼어붙고, 겉보기는 "손이 안 닫힌다"와 구분되지 않는다.
        _herr = (self._syn_target - self.robot.data.joint_pos[:, self._syn_ids]).abs()
        ex["ctrl/hand_joint_err_max"] = _herr[:, self._syn_movable].max()
        ex["ctrl/hand_blocked_frac"] = self._hand_blocked().float().mean()
        ex["ctrl/hand_target_step"] = (self._syn_target - self._prev_syn_target).abs().mean()
        self._prev_syn_target = self._syn_target.clone()

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
        if self._arm_reset_off is not None:
            _q = self.robot.data.joint_pos[env_ids].clone()
            _q[:, self._arm_ids_t] = torch.clamp(
                self._default_q[env_ids][:, self._arm_ids_t] + self._arm_reset_off.unsqueeze(0),
                self._arm_lo[env_ids], self._arm_hi[env_ids])
            self.robot.write_joint_state_to_sim(_q, torch.zeros_like(_q), env_ids=env_ids)
            self._arm_q_target[env_ids] = _q[:, self._arm_ids_t]
            self._prev_arm_q_target[env_ids] = self._arm_q_target[env_ids]
        # ★부모는 `_syn_close[env_ids] = 0` 으로 리셋한다. per_role 에서 푼 관절은 0 이
        #   중립이 아니라 **한쪽 끝**이라 그대로 두면 손이 벌어진 채 시작한다(synergy 면 0 그대로).
        self._syn_close[env_ids] = self._close_home.unsqueeze(0)
        self._prev_syn_target[env_ids] = self._closure_to_target(self._syn_close[env_ids])
