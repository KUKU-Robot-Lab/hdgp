"""SideRig — 한쪽 팔의 배선 묶음: Fabrics 팔 + 관절공간 시너지 손 + 접촉 + palm 지령 (mimic 판).

★`pour_fabric/side_rig.py` 사본. 저차원(언더액추) 손·양팔 26 DOF fabric 을 위해 바뀐 곳:
  1. Fabrics: 클래스 생성자가 tesollo 와 다르다(robot_dir·params·tip_per_finger 인자 없음, side 고정).
     fabric cspace 는 **양팔 26** 이고 이 팔은 `bimanual.fabric_slots` 슬라이스만 읽고 쓴다.
     반대팔 슬라이스는 매 스텝 상대 rig 의 상태로 덮는다(`sync_other`) — 안 맞추면 body 반발이
     존재하지 않는 반대팔 자세를 피하려 팔을 민다.
  2. 손 레이아웃 `per_finger`: 프로필 `hand_finger_channels` 가 **구동관절 ↔ 액션 슬롯 1:1** 을
     준다(RH56F1: 채널 접미사가 {1,2} 뿐이라 채널×손가락 격자는 슬롯 4개가 아무 관절에도 안
     걸린다 — grasp_fj_rh 09.07 실측). 4지 결합(`couple_four_fingers`)은 같은 접미사끼리 평균.
  3. 접촉 동결: 접미사 "3" 리터럴 대신 프로필 `hand_freeze_suffixes` 로 굴곡 집합을 정한다
     (RH 4지 구동은 "_1", 종속 "_2" 마디의 접촉이 그 손가락의 구동관절을 멈춘다 = finger 스코프).

grasp_s2r 의 `GraspS2RControlMixin`(한 팔 전용, self 에 묶임)을 **인스턴스 단위**로 옮겼다 —
양팔이 각자 하나씩 갖는다. 로봇 종속 정보는 전부 프로필에서 온다(이름 리터럴 금지 계약).

포팅 출처(수식·규약 동일, 09.13):
  · 시너지 `_setup_synergy` / `_synergy_targets`(contact 모드) / `_close_progress`
    / `_apply_pose_knobs`(oppose delta) — grasp_s2r_control.py
  · fabric 설정·적분·FK 부팅 게이트 — grasp_s2r_control.py `_setup_fabrics`/`_init_home_palm`
  · palm 액션 = 앵커 + 델타 — grasp_s2r_env.py `_pre_physics_step`
"""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat

_DEG = math.pi / 180.0


class SideRig:
    def __init__(self, env, profile, *, role: str, sensors: dict, palm_sensor,
                 delta_lo: tuple, delta_hi: tuple, oppose_sign: float) -> None:
        self.profile = profile
        self.role = role                      # "src" | "rcv"
        self.env = env
        robot, device, N = env.robot, env.device, env.num_envs
        self.robot, self.device, self.N = robot, device, N
        cfg = env.cfg

        # ---- 관절/바디 인덱스 -----------------------------------------------------------
        self.arm_ids, _ = robot.find_joints(profile.arm_joint_regex)
        self.hand_ids, _ = robot.find_joints(profile.hand_joint_regex)
        if (len(self.arm_ids) != profile.num_arm_joints
                or len(self.hand_ids) != profile.num_hand_joints):
            raise RuntimeError(
                f"[{profile.name}] 프로필 조인트 수 불일치: arm {len(self.arm_ids)}"
                f"!={profile.num_arm_joints}, hand {len(self.hand_ids)}!={profile.num_hand_joints}")
        self.arm_t = torch.tensor(self.arm_ids, device=device, dtype=torch.long)
        self.hand_t = torch.tensor(self.hand_ids, device=device, dtype=torch.long)
        self.palm_idx = self._one_body(profile.palm_body)
        self.fingers = list(profile.finger_sensor_bodies.keys())
        self.tip_t = torch.tensor([self._one_body(b) for b in profile.fingertip_bodies],
                                  device=device, dtype=torch.long)
        self.grp_a = torch.tensor([self.fingers.index(f) for f in profile.contact_group_a],
                                  device=device, dtype=torch.long)
        self.grp_b = torch.tensor([self.fingers.index(f) for f in profile.contact_group_b],
                                  device=device, dtype=torch.long)
        self.sensors = sensors                # finger -> [ContactSensor] (프로필 body 순)
        self.palm_sensor = palm_sensor

        # ---- palm 지령 -------------------------------------------------------------------
        self.box_lo = torch.tensor(profile.palm_box_min, device=device)
        self.box_hi = torch.tensor(profile.palm_box_max, device=device)
        lo = [float(v) for v in delta_lo]
        hi = [float(v) for v in delta_hi]
        self.delta_lo = torch.tensor(lo[:3] + [v * _DEG for v in lo[3:]], device=device)
        self.delta_hi = torch.tensor(hi[:3] + [v * _DEG for v in hi[3:]], device=device)
        self.anchor = torch.zeros(N, 6, device=device)       # fabric 프레임
        self.anchor_env = torch.zeros(N, 6, device=device)   # env-local 실측(관측·로깅용)
        self.palm_targets = torch.zeros(N, 6, device=device)
        self.fab_to_env = torch.zeros(3, device=device)

        # ---- 시너지 손 --------------------------------------------------------------------
        self._setup_synergy(oppose_sign)

        # ---- 시작 자세(에피소드 리셋 관절값) --------------------------------------------------
        q_default = robot.data.default_joint_pos[0]
        self.reset_q = q_default.clone()
        if profile.arm_reset_joint_pos:
            if len(profile.arm_reset_joint_pos) != profile.num_arm_joints:
                raise RuntimeError(f"[{profile.name}] arm_reset_joint_pos 길이 오류")
            self.reset_q[self.arm_t] = torch.tensor(profile.arm_reset_joint_pos, device=device)
        self.reset_q[torch.tensor(self.syn_ids, device=device)] = self.syn_open

        self.fabric = None
        self.integrator = None
        self.fab_t = None

    # ==================================================================
    def _one_body(self, name: str) -> int:
        ids, _ = self.robot.find_bodies(name)
        if len(ids) != 1:
            raise RuntimeError(f"[{self.profile.name}] body '{name}' 해석 실패: {ids}")
        return ids[0]

    # ==================================================================
    # 시너지 손 (grasp_s2r `_setup_synergy` 이식)
    # ==================================================================
    def _setup_synergy(self, oppose_sign: float) -> None:
        p, cfg, dev, N = self.profile, self.env.cfg, self.device, self.N
        for _f in ("hand_joint_names", "hand_open_pose", "hand_grip_pose"):
            if not getattr(p, _f):
                raise RuntimeError(f"[{p.name}] 시너지 손 제어에 필요한 프로필 필드 {_f} 가 없다")
        n = len(p.hand_joint_names)
        if len(p.hand_open_pose) != n or len(p.hand_grip_pose) != n:
            raise RuntimeError(f"[{p.name}] 자세 배열 길이 불일치")
        jn = self.robot.data.joint_names
        self.syn_ids = [jn.index(nm) for nm in p.hand_joint_names]
        self.syn_t = torch.tensor(self.syn_ids, device=dev, dtype=torch.long)
        self.syn_open = torch.tensor(p.hand_open_pose, device=dev)
        self.syn_grip = torch.tensor(p.hand_grip_pose, device=dev)

        # ★per_finger 레이아웃: 프로필 `hand_finger_channels[finger][sfx]` = 액션 슬롯. 구동관절과
        #   슬롯이 1:1 이어야 한다(빠지거나 겹치면 부팅 실패 — 슬롯이 죽는 것은 지표에 안 보인다).
        if not p.hand_finger_channels:
            raise RuntimeError(f"[{p.name}] per_finger 레이아웃에 필요한 hand_finger_channels 가 없다")
        ch, fi, slot = [], [], []
        for nm in p.hand_joint_names:
            sfx = nm.rsplit("_", 1)[1]
            if sfx not in p.hand_channel_of_joint:
                raise RuntimeError(f"[{p.name}] hand_channel_of_joint 에 접미사 {sfx} 없음")
            ch.append(int(p.hand_channel_of_joint[sfx]))
            hit = [i for i, f in enumerate(self.fingers) if f"_{f}_" in nm]
            if len(hit) != 1:
                raise RuntimeError(f"[{p.name}] 관절 {nm} 의 손가락을 특정 못함: {hit}")
            fi.append(hit[0])
            fmap = p.hand_finger_channels.get(self.fingers[hit[0]], {})
            if sfx not in fmap:
                raise RuntimeError(f"[{p.name}] hand_finger_channels[{self.fingers[hit[0]]}] 에 접미사 {sfx} 없음")
            slot.append(int(fmap[sfx]))
        if sorted(slot) != list(range(n)):
            raise RuntimeError(f"[{p.name}] per_finger 슬롯이 0..{n-1} 을 정확히 덮지 않는다: {slot}")
        self.syn_ch = torch.tensor(ch, device=dev, dtype=torch.long)
        self.syn_fi = torch.tensor(fi, device=dev, dtype=torch.long)
        self.syn_slot = torch.tensor(slot, device=dev, dtype=torch.long)
        self.syn_nch = len(set(ch))
        self.hand_action_width = n
        # 4지 결합 행렬(n×n): contact_group_a(엄지) 밖 손가락의 같은-접미사 관절은 지령 평균을 쓴다.
        grp_a = set(int(i) for i in self.grp_a.tolist())
        M = torch.eye(n, device=dev)
        sfx_all = [nm.rsplit("_", 1)[1] for nm in p.hand_joint_names]
        for i in range(n):
            if fi[i] in grp_a:
                continue
            group = [j for j in range(n) if fi[j] not in grp_a and sfx_all[j] == sfx_all[i]]
            M[i, :] = 0.0
            M[i, group] = 1.0 / len(group)
        self.syn_couple = M

        # 대향 관절 grip = open + delta (grasp_s2r `_apply_pose_knobs`). 부호는 측별.
        d = float(cfg.oppose_grip_delta_rad) * float(oppose_sign)
        if d != 0.0:
            idx = [i for i, nm in enumerate(p.hand_joint_names)
                   if ch[i] == 1 and any(f"_{f}_" in nm for f in p.contact_group_a)]
            if not idx:
                raise RuntimeError(f"[{p.name}] 대향 손가락 ch1 관절을 못 찾았다")
            for i in idx:
                self.syn_grip[i] = self.syn_open[i] + d

        # 동결 집합은 프로필이 정한다(접미사 리터럴 금지). RH56F1: ("1","2") = 구동 6 전부.
        sfx = [nm.rsplit("_", 1)[1] for nm in p.hand_joint_names]
        if not p.hand_freeze_suffixes:
            raise RuntimeError(f"[{p.name}] hand_freeze_suffixes 가 비어 접촉 동결이 아무 관절도 안 잡는다")
        self.syn_flex = torch.tensor([s in p.hand_freeze_suffixes for s in sfx], device=dev)
        self.syn_movable = (self.syn_grip - self.syn_open).abs() > 1e-4
        if not bool(self.syn_movable.any()):
            raise RuntimeError(f"[{p.name}] 가동 손관절이 없다 — open/grip 자세 확인")
        self.syn_close = torch.zeros(N, n, device=dev)
        self.syn_target = self.syn_open.unsqueeze(0).expand(N, n).clone()
        self.syn_vel = torch.zeros(N, n, device=dev)
        lim = self.robot.data.soft_joint_pos_limits[0, self.syn_t, :]
        self.syn_lo, self.syn_hi = lim[:, 0].contiguous(), lim[:, 1].contiguous()
        _bad_open = ((self.syn_open < self.syn_lo) | (self.syn_open > self.syn_hi)).nonzero()
        if len(_bad_open):
            raise RuntimeError(
                f"[{p.name}] hand_open_pose 가 관절한계 밖: "
                f"{[p.hand_joint_names[int(i)] for i in _bad_open.flatten()]}")
        _grip_c = self.syn_grip.clamp(self.syn_lo, self.syn_hi)
        print(f"[pour_fabric_mimic:{self.role}] synergy {p.name}: 관절 {n} · 채널 {self.syn_nch} · "
              f"가동 {int(self.syn_movable.sum())} · grip 한계clamp "
              f"{int((self.syn_grip != _grip_c).sum())}개 · oppose {d:+.2f}", flush=True)

    def synergy_targets(self, a_hand: torch.Tensor, close_gate: torch.Tensor) -> torch.Tensor:
        """액션(손가락×채널) → 관절 목표 (N, n). grasp_s2r contact 모드 그대로."""
        cfg, N = self.env.cfg, self.N
        a_j = a_hand.clamp(-1.0, 1.0)[:, self.syn_slot]           # (N, n) 관절 순서
        if bool(cfg.couple_four_fingers):
            common = a_j @ self.syn_couple.T
            rs = float(cfg.finger_residual_scale)
            a_j = common if rs == 0.0 else common + rs * (a_j - common)
        cmd_j = 0.5 * (a_j + 1.0)
        rate = float(cfg.synergy_close_speed)
        delta = (cmd_j - self.syn_close).clamp(-rate, rate)
        g = close_gate.unsqueeze(1)
        delta = torch.where(delta > 0.0, delta * g, delta)      # 닫는 방향만 게이트
        if bool(cfg.synergy_contact_freeze):
            mid, dist, _ = self.finger_link_forces()
            thr = float(cfg.contact_force_threshold)
            h_mid = (mid > thr)[:, self.syn_fi]
            h_dist = (dist > thr)[:, self.syn_fi]
            # 손가락당 구동관절이 하나뿐이라 "닿은 마디만" 스코프는 정의되지 않는다 — finger 스코프만.
            hold = (h_mid | h_dist) & self.syn_flex
            delta = torch.where(hold & (delta > 0.0), torch.zeros_like(delta), delta)
        self.syn_close = (self.syn_close + delta).clamp(0.0, 1.0)
        tgt = torch.lerp(self.syn_open.unsqueeze(0), self.syn_grip.unsqueeze(0), self.syn_close)
        return tgt.clamp(self.syn_lo.unsqueeze(0), self.syn_hi.unsqueeze(0))

    def closure(self) -> torch.Tensor:
        """실측 평균 폐쇄도 (N,) [0,1] — 가동 관절만."""
        q = self.robot.data.joint_pos[:, self.syn_t]
        span = (self.syn_grip - self.syn_open).unsqueeze(0)
        prog = ((q - self.syn_open.unsqueeze(0)) / span).clamp(0.0, 1.0)
        return prog[:, self.syn_movable].mean(dim=1)

    def joint_err(self) -> torch.Tensor:
        err = self.syn_target - self.robot.data.joint_pos[:, self.syn_t]
        return (err / float(self.env.cfg.joint_pos_err_max)).clamp(-1.0, 1.0)

    # ==================================================================
    # 접촉
    # ==================================================================
    def _mag(self, sensor) -> torch.Tensor:
        return sensor.data.force_matrix_w.view(self.N, -1, 3).sum(dim=1).norm(dim=-1)

    def finger_forces(self) -> torch.Tensor:
        """손가락별 자기 컵 접촉력 합 (N,F)."""
        out = []
        for f in self.fingers:
            t = torch.zeros(self.N, device=self.device)
            for s in self.sensors[f]:
                t = t + self._mag(s)
            out.append(t)
        return torch.stack(out, dim=1)

    def finger_link_forces(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(중간, 원위, 팁) (N,F). `finger_sensor_bodies` 규약: 마지막=팁, 앞이 (중간, 원위)."""
        mids, dists, tips = [], [], []
        for f in self.fingers:
            ss = self.sensors[f]
            mi, di = (0, 1) if len(ss) >= 3 else (0, 0)
            mids.append(self._mag(ss[mi])); dists.append(self._mag(ss[di])); tips.append(self._mag(ss[-1]))
        return torch.stack(mids, 1), torch.stack(dists, 1), torch.stack(tips, 1)

    def palm_force(self) -> torch.Tensor:
        return self._mag(self.palm_sensor)

    def foreign_force(self) -> torch.Tensor:
        """손(손가락 마디+손바닥)이 **자기 컵 외**의 것에 닿는 힘 (N,) — 09.14 s2r 충돌 신호.

        `net_forces_w` 는 링크가 받는 모든 접촉, `force_matrix_w` 는 자기 컵 필터 접촉이다.
        둘의 크기 차(≥0)가 상대 손·상대 컵·테이블과의 접촉이다(grasp_s2r `_mag_net` 진단 규약).
        """
        tot = torch.zeros(self.N, device=self.device)
        sensors = [s for f in self.fingers for s in self.sensors[f]] + [self.palm_sensor]
        for s in sensors:
            net = s.data.net_forces_w.view(self.N, -1, 3).sum(dim=1).norm(dim=-1)
            tot = tot + (net - self._mag(s)).clamp(min=0.0)
        return tot

    def tip_tactile(self) -> torch.Tensor:
        """손끝 촉각 (N,F) [N] — 실기 RH56F1 TouchData1.finger_forces 대응.

        실기 센서는 닿은 대상과 무관하게 재므로 컵 필터 행렬이 아니라 손끝 링크 전체 접촉력(net)을 쓴다.
        """
        out = [self.sensors[f][-1].data.net_forces_w.view(self.N, -1, 3).sum(dim=1).norm(dim=-1)
               for f in self.fingers]
        return torch.stack(out, dim=1)

    def grasped(self, forces: torch.Tensor) -> torch.Tensor:
        thr = float(self.env.cfg.contact_force_threshold)
        return (forces[:, self.grp_a] > thr).any(dim=1) & (forces[:, self.grp_b] > thr).any(dim=1)

    # ==================================================================
    # 좌표
    # ==================================================================
    def palm_pos(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.palm_idx] - self.env.scene.env_origins

    def palm_R(self) -> torch.Tensor:
        return matrix_from_quat(self.robot.data.body_quat_w[:, self.palm_idx])

    def palm_pose_6d(self) -> torch.Tensor:
        """env-local xyz + [yaw, pitch, roll] — fabric "euler_zyx" 지령과 같은 규약."""
        r, pi, y = euler_xyz_from_quat(self.robot.data.body_quat_w[:, self.palm_idx])
        return torch.cat([self.palm_pos(), torch.stack([y, pi, r], dim=1)], dim=1)

    def tips_pos(self) -> torch.Tensor:
        """(N,F,3) env-local."""
        return self.robot.data.body_pos_w[:, self.tip_t] - self.env.scene.env_origins[:, None, :]

    # ==================================================================
    # Fabrics
    # ==================================================================
    def setup_fabric(self, fabric_class, integrator_cls, arm_slice: slice, hand_slice: slice) -> None:
        """양팔 26 DOF fabric 인스턴스(이 팔의 palm 만 활성). 슬라이스는 `bimanual.fabric_slots`."""
        p, cfg = self.profile, self.env.cfg
        if not p.fabric_class or not p.fabric_robot_dir:
            raise RuntimeError(f"[{p.name}] fabric_class/fabric_robot_dir 가 없다")
        # ★RH56F1 fabric 생성자: (batch_size, device, timestep, graph_capturable, use_hand_fabric,
        #   hand_mode). robot_dir·params 는 클래스가 고정 로드한다 — tesollo 식 kwargs 를 주면 TypeError.
        self.fabric = fabric_class(
            batch_size=self.N, device=self.device, timestep=float(cfg.fabrics_dt),
            graph_capturable=bool(cfg.fabric_use_cuda_graph),
            use_hand_fabric=False, hand_mode="direct")
        self.integrator = integrator_cls(self.fabric)
        self.arm_sl, self.hand_sl = arm_slice, hand_slice
        expect = len(p.fabric_joint_order)
        if self.fabric.num_joints != expect or expect != 2 * (p.num_arm_joints + p.num_hand_joints):
            raise RuntimeError(f"[{p.name}] fabric num_joints={self.fabric.num_joints} != {expect} "
                               f"(양팔 26 DOF 규약)")
        if (arm_slice.stop - arm_slice.start) != p.num_arm_joints or (hand_slice.stop - hand_slice.start) != p.num_hand_joints:
            raise RuntimeError(f"[{p.name}] fabric 슬라이스 폭이 프로필과 다르다: {arm_slice} {hand_slice}")
        idx = []
        for name in p.fabric_joint_order:
            ids, _ = self.robot.find_joints(name)
            if len(ids) != 1:
                raise RuntimeError(f"[{p.name}] fabric 관절 '{name}' 해석 실패: {ids}")
            idx.append(ids[0])
        self.fab_t = torch.tensor(idx, device=self.device, dtype=torch.long)
        # 이 팔의 fabric 팔 슬라이스가 프로필 팔 관절과 같은 관절인지(순서까지) 대조.
        if self.fab_t[self.arm_sl].tolist() != self.arm_ids:
            raise RuntimeError(f"[{p.name}] fabric 팔 슬라이스 {self.fab_t[self.arm_sl].tolist()} "
                               f"≠ 프로필 팔 관절 {self.arm_ids}")
        syn_pos = {int(j): k for k, j in enumerate(self.syn_ids)}
        fab_hand = self.fab_t[self.hand_sl].tolist()
        missing = [int(j) for j in fab_hand if int(j) not in syn_pos]
        if missing:
            raise RuntimeError(f"[{p.name}] synergy 자세에 없는 fabric 손 관절 {missing}")
        self.syn_to_fab_idx = torch.tensor([syn_pos[int(j)] for j in fab_hand],
                                           device=self.device, dtype=torch.long)
        q0 = self.reset_q.unsqueeze(0).expand(self.N, -1)
        self.fabric_q = q0[:, self.fab_t].contiguous()
        self.fabric_qd = torch.zeros(self.N, self.fabric.num_joints, device=self.device)
        self.fabric_qdd = torch.zeros_like(self.fabric_qd)
        self.fabric_hand_cmd = torch.zeros(self.N, p.num_hand_joints, device=self.device)   # direct 계약(무시됨)
        # cspace rest = 시작 자세(널스페이스가 시작 자세로 당긴다) — 양팔 전부.
        self.fabric.default_config.copy_(self.fabric_q)
        self.fabric_damping = float(cfg.fabrics_damping_gain) * torch.ones(self.N, 1, device=self.device)
        if not p.palm_box_verified:
            print(f"[pour_fabric_mimic:{self.role}] ⚠ palm_box 미검증({p.name}) — probe 후 승격할 것",
                  flush=True)

    def init_anchor(self) -> None:
        """시작 자세의 palm 실측 → 앵커 + fabric FK 정합 게이트. 물리 2스텝 뒤에 부를 것."""
        p, cfg = self.profile, self.env.cfg
        home = self.palm_pose_6d()[0]
        q0f = self.reset_q.unsqueeze(0).expand(self.N, -1)[:, self.fab_t].contiguous()
        nt = len(self.fingers)
        tips_fab = self.fabric._fingertip_taskmap(q0f, None)[0].reshape(self.N, nt, 3)[0]
        tips_sim = self.tips_pos()[0]
        delta = tips_sim - tips_fab
        spread = float(delta.std(dim=0).max())
        # ★09.13: 물리 2스텝 중력 처짐이 손끝마다 달라 산포 1~2 mm 가 노이즈 바닥이다(2 mm 로 두면
        #   경계에서 무작위로 죽는다 — 로컬 실측 2.1 mm). 기구학 불일치는 수십 mm 로 나오므로 5 mm.
        if spread > 5e-3:
            raise RuntimeError(
                f"[{p.name}] fabric↔env 프레임이 순수 평행이동이 아니다(산포 {spread*1000:.1f}mm)")
        self.fab_to_env = delta.mean(dim=0)
        fab = self.fabric.get_palm_pose(self.fabric_q.detach(), "euler_zyx")[0]
        dp = float(torch.norm((fab[:3] + self.fab_to_env) - home[:3]))
        # ★각도 차는 wrap 한다 — 좌팔은 yaw 가 ±π 경계에 걸려(−3.127 vs +3.13) 원본 식이면 359° 가 난다.
        _d = torch.remainder(fab[3:] - home[3:] + math.pi, 2 * math.pi) - math.pi
        dr = float(torch.max(torch.abs(_d)))
        print(f"[pour_fabric_mimic:{self.role}] 시작 palm={[round(v, 4) for v in home.tolist()]} | "
              f"fabric FK 정합 pos {dp*1000:.2f}mm rot {math.degrees(dr):.2f}° | "
              f"fab→env {[round(float(v)*1000) for v in self.fab_to_env]}mm", flush=True)
        if dp > float(cfg.fabric_fk_pos_tol) or dr > math.radians(2.0):
            raise RuntimeError(
                f"[{p.name}] fabric FK 가 USD palm 과 어긋난다: {dp*1000:.1f}mm / "
                f"{math.degrees(dr):.1f}° — fabric URDF·joint_order·palm_body 를 본다")
        out = (home[:3] < self.box_lo) | (home[:3] > self.box_hi)
        if bool(out.any()):
            raise RuntimeError(f"[{p.name}] 시작 palm 이 워크스페이스 박스 밖: {home[:3].tolist()}")
        self.anchor_env[:] = home.unsqueeze(0)
        anchor_fab = home.clone()
        anchor_fab[:3] -= self.fab_to_env
        self.anchor[:] = anchor_fab.unsqueeze(0)
        self.palm_targets[:] = self.anchor
        # 델타 박스가 팔 박스에 잘리는 축을 부팅에서 알린다.
        lo_env, hi_env = home[:3] + self.delta_lo[:3], home[:3] + self.delta_hi[:3]
        cut = [ax for i, ax in enumerate("xyz")
               if float(lo_env[i]) < float(self.box_lo[i]) or float(hi_env[i]) > float(self.box_hi[i])]
        if cut:
            print(f"[pour_fabric_mimic:{self.role}] ⚠ 델타 박스가 프로필 palm 박스에 잘리는 축 {cut}",
                  flush=True)

    def compose_palm_target(self, a6: torch.Tensor, active: torch.Tensor) -> None:
        """palm 6D = 앵커 + 델타. hold 중(active=False)은 앵커.

        ★델타 박스가 lo/hi **비대칭**이라 grasp_s2r 의 선형 매핑(0.5(a+1)(hi−lo)+lo)을 쓰면
          a=0 이 앵커가 아니라 박스 중점이 된다(09.13 부팅 실측). a≥0 → a·hi, a<0 → −a·lo
          로 부호별로 매핑해 **a=0 = 앵커**를 지킨다(구 pour `symmetric_action_scale=False`).
        """
        delta = torch.where(a6 >= 0.0, a6 * self.delta_hi, -a6 * self.delta_lo)
        raw = self.anchor + delta
        pos_env = raw[:, :3] + self.fab_to_env
        pos_env = pos_env.clamp(self.box_lo, self.box_hi)
        raw = torch.cat([pos_env - self.fab_to_env, raw[:, 3:]], dim=1)
        self.palm_targets = torch.where(active.unsqueeze(1), raw, self.anchor)

    def step_fabric(self, world_ids, world_indicator) -> None:
        cfg = self.env.cfg
        self.fabric.set_features(
            self.fabric_hand_cmd, self.palm_targets, "euler_zyx",
            self.fabric_q.detach(), self.fabric_qd.detach(),
            world_ids, world_indicator, self.fabric_damping)
        for _ in range(int(cfg.fabric_decimation)):
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.integrator.step(
                self.fabric_q.detach(), self.fabric_qd.detach(),
                self.fabric_qdd.detach(), float(cfg.fabrics_dt))

    def pin_fabric(self, hold: torch.Tensor, q_pin: torch.Tensor) -> None:
        """hold env 는 적분 결과를 시작 자세로 되돌린다(마스크 곱 — GPU sync 없음)."""
        h = hold.unsqueeze(1).float()
        self.fabric_q = h * q_pin + (1.0 - h) * self.fabric_q
        self.fabric_qd = (1.0 - h) * self.fabric_qd
        self.fabric_qdd = (1.0 - h) * self.fabric_qdd

    def apply_action(self) -> None:
        cfg = self.env.cfg
        arm_target = self.fabric_q[:, self.arm_sl]
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_ids)
        self.robot.set_joint_velocity_target(
            float(cfg.fabric_velocity_ff_scale) * self.fabric_qd[:, self.arm_sl], joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(self.syn_target, joint_ids=self.syn_ids)
        self.robot.set_joint_velocity_target(
            float(cfg.hand_velocity_ff_scale) * self.syn_vel, joint_ids=self.syn_ids)

    def sync_fabric_hand(self) -> None:
        """fabric 의 손 상태 = 시너지 목표 — 안 맞추면 fabric 이 없는 자기충돌을 피하려 팔을 민다."""
        self.fabric_q[:, self.hand_sl] = self.syn_target[:, self.syn_to_fab_idx]

    def sync_other(self, other: "SideRig") -> None:
        """반대팔 슬라이스 = 상대 rig 의 fabric 상태(양팔 26 DOF 를 두 인스턴스가 나눠 가진다)."""
        for sl in (other.arm_sl, other.hand_sl):
            self.fabric_q[:, sl] = other.fabric_q[:, sl]
            self.fabric_qd[:, sl] = other.fabric_qd[:, sl]
            self.fabric_qdd[:, sl] = other.fabric_qdd[:, sl]

    # ==================================================================
    def reset(self, env_ids: torch.Tensor) -> None:
        q0 = self.reset_q.unsqueeze(0)
        self.syn_close[env_ids] = 0.0
        self.syn_target[env_ids] = q0[:, self.syn_t]
        self.syn_vel[env_ids] = 0.0
        self.fabric_q[env_ids] = q0[:, self.fab_t]
        self.fabric_qd[env_ids] = 0.0
        self.fabric_qdd[env_ids] = 0.0
        self.palm_targets[env_ids] = self.anchor[env_ids]
