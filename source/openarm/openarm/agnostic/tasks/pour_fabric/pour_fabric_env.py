"""pour_fabric — 양팔 물붓기 (Fabrics ×2, robot-agnostic, direct).

제어 스택 (grasp_lift_fabric 규약 계승):
  팔  = Geometric Fabrics ×2 (source/receiver 각 1 인스턴스).
        정책이 **절대 palm pose** 를 내면(앵커 = warm 측정 pose) fabric 이 관절 목표 생성.
  손  = 양손 전체 **동결** — warm 파지 자세를 관절 PD 로 유지(재조임 없음).

★obs 에 fabric_q / 비드 ground truth 를 넣지 않는다(actor) — 실기에 없는 값이다.
★래치 금지·리셋 시 정책상태 재구성(anchor=측정 FK pose)이 곧 실기 인계 프로토콜이다.

인계 시퀀스 (에피소드 시작):
  reset  : warm 관절+컵+비드 물리 상태 복원, 관절 목표 = warm(그대로 유지)
  hold   : `hold_steps` 동안 fabric_q 를 warm 에 고정(정확한 관절 hold) — 비드 정착
  capture: hold 종료 시점에 palm FK pose 측정 → 앵커·지령·낙하 기준선 확정
           (★리셋 직후 body 버퍼는 stale — 이 저장소에서 3회 재발한 함정)
  ramp   : `handover_ramp_steps` 동안 앵커→정책 목표 선형 보간(+slew 이중 방어)
"""

from __future__ import annotations

import math
import os
import sys

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from openarm.agnostic.modules import adr as _adr

from . import bimanual as _bm
from . import pour_fabric_env_cfg as _cfg
from .bead_flags import BeadGeometry, compute_bead_flags
from .pour_fabric_env_cfg import PourFabricEnvCfg
from .rewards import compute_rewards
from .warm_bank import PourWarmBank

_FABRICS_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..",
                 "FABRICS", "src"))
if os.path.isdir(_FABRICS_SRC) and _FABRICS_SRC not in sys.path:
    sys.path.insert(0, _FABRICS_SRC)

import fabrics_sim.fabrics.openarm_rh56f1_pose_fabric as _fab_rh   # noqa: E402
import fabrics_sim.fabrics.openarm_tesollo_pose_fabric as _fab_tes  # noqa: E402
from fabrics_sim.integrator.integrators import DisplacementIntegrator  # noqa: E402
from fabrics_sim.utils.utils import initialize_warp                   # noqa: E402
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel      # noqa: E402

_DEG = math.pi / 180.0


def _fabric_class(name: str):
    for mod in (_fab_tes, _fab_rh):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise RuntimeError(f"Fabrics 클래스 '{name}' 를 찾을 수 없다")


class _SideRig:
    """한쪽 팔의 배선 묶음(인덱스·fabric·박스·지령 버퍼) — source/receiver 공용.

    로봇 종속 정보는 전부 profile 에서 온다(조인트/바디 이름 하드코딩 금지 계약).
    """

    def __init__(self, env: "PourFabricEnv", profile, *, n_pose: int) -> None:
        self.profile = profile
        self.n_pose = n_pose            # 정책이 제어하는 pose 차원 (source 6 / receiver 3)
        robot, device, N = env.robot, env.device, env.num_envs

        self.arm_ids, arm_names = robot.find_joints(profile.arm_joint_regex)
        self.hand_ids, hand_names = robot.find_joints(profile.hand_joint_regex)
        if (len(self.arm_ids) != profile.num_arm_joints
                or len(self.hand_ids) != profile.num_hand_joints):
            raise RuntimeError(
                f"[{profile.name}] 프로필 조인트 수 불일치: "
                f"arm {len(self.arm_ids)}!={profile.num_arm_joints}, "
                f"hand {len(self.hand_ids)}!={profile.num_hand_joints}")
        self.arm_t = torch.tensor(self.arm_ids, device=device, dtype=torch.long)
        self.hand_t = torch.tensor(self.hand_ids, device=device, dtype=torch.long)

        self.palm_idx = env._one_body(profile, profile.palm_body)
        self.tip_t = torch.tensor(
            [env._one_body(profile, b) for b in profile.fingertip_bodies],
            device=device, dtype=torch.long)

        fingers = list(profile.fingers)
        self.fingers = fingers
        self.grp_a = torch.tensor([fingers.index(f) for f in profile.contact_group_a],
                                  device=device, dtype=torch.long)
        self.grp_b = torch.tensor([fingers.index(f) for f in profile.contact_group_b],
                                  device=device, dtype=torch.long)

        # ---- 지령 버퍼 (앵커 기반 절대 액션 — capture 시점에 확정) ------------------
        self.anchor = torch.zeros(N, 6, device=device)      # warm 측정 palm pose
        self.cmd = torch.zeros(N, 6, device=device)         # slew 지령 (6D 전체 유지)
        self.scale = torch.ones(N, 6, device=device)
        self.lo = torch.zeros(1, 6, device=device)
        self.hi = torch.zeros(1, 6, device=device)
        # 낙하 판정 기준선 (capture 시점)
        self.ref_palm_cup = torch.zeros(N, device=device)
        self.ref_cup_z = torch.zeros(N, device=device)

        # fabric 은 env 가 채운다(_setup_fabrics)
        self.fabric = None
        self.integrator = None
        self.fab_t = None
        self.fabric_q = None
        self.fabric_qd = None
        self.fabric_qdd = None
        self.hand_hold = None            # (N, num_hand_joints) — warm 파지 자세 PD hold
        self.sensors: dict = {}


class PourFabricEnv(DirectRLEnv):
    cfg: PourFabricEnvCfg

    # ==================================================================
    def __init__(self, cfg: PourFabricEnvCfg, render_mode: str | None = None, **kw):
        _cfg.resolve_cfg(cfg)            # hydra 오버라이드 후 파생값 재계산(멱등)
        self.pair = _bm.get_pair(cfg.pair_name)
        self._grav_comp = float(cfg.gravity_compensation) if cfg.enable_gravity else 0.0
        # ★부팅 가드 (grasp_lift_fabric ee74b7e): cfg 필드 vs 파생 robot_cfg 대조.
        _sp = cfg.robot_cfg.spawn
        _gr_off = bool(_sp.rigid_props.disable_gravity)
        _sc_on = bool(_sp.articulation_props.enabled_self_collisions)
        if _gr_off == bool(cfg.enable_gravity) or _sc_on != bool(cfg.enable_self_collisions):
            raise RuntimeError(
                "물리 스위치가 파생 cfg 에 반영되지 않았다 — resolve_cfg 경로 확인.\n"
                f"  enable_gravity={cfg.enable_gravity} vs spawn.disable_gravity={_gr_off}\n"
                f"  enable_self_collisions={cfg.enable_self_collisions} vs "
                f"spawn.enabled_self_collisions={_sc_on}")
        print(f"[pour_fabric] 물리: self_collisions={_sc_on} · gravity={not _gr_off}"
              f" · grav_comp={self._grav_comp}", flush=True)
        super().__init__(cfg, render_mode, **kw)

        N, dev = self.num_envs, self.device
        self.src = _SideRig(self, self.pair.source, n_pose=6)
        self.rcv = _SideRig(self, self.pair.receiver, n_pose=3)
        for rig in (self.src, self.rcv):
            if not rig.profile.palm_box_verified:
                print(f"[pour_fabric] ⚠ 프로필 '{rig.profile.name}' palm 박스 **미실측**"
                      " — probe_workspace_reach 로 확인할 것.", flush=True)

        # ---- palm 박스: 위치 = 프로필 / 자세 = 중심(sign·[90,0,90]) + 비대칭 오프셋 ----
        for rig in (self.src, self.rcv):
            sign = 1.0 if rig.profile.side == "r" else -1.0
            centre = [sign * 90.0, 0.0, sign * 90.0]
            if rig is self.src:
                lo_off, hi_off = cfg.pose_offset_lo_deg, cfg.pose_offset_hi_deg
                if sign < 0:            # 좌팔이 source 면 오프셋 구간을 미러
                    lo_off, hi_off = tuple(-h for h in hi_off), tuple(-l for l in lo_off)
            else:
                # receiver 자세는 액션이 없지만 박스는 sanity 용으로 ±45° 를 둔다.
                lo_off, hi_off = (-45.0,) * 3, (45.0,) * 3
            rig.lo = torch.tensor(
                list(rig.profile.palm_box_min)
                + [(c + o) * _DEG for c, o in zip(centre, lo_off)], device=dev).unsqueeze(0)
            rig.hi = torch.tensor(
                list(rig.profile.palm_box_max)
                + [(c + o) * _DEG for c, o in zip(centre, hi_off)], device=dev).unsqueeze(0)

        # ---- Fabrics ×2 ------------------------------------------------------------
        self._setup_fabrics()

        # ---- 홈 palm 실측 (probe 모드 앵커 + 부팅 sanity) -----------------------------
        # ★__init__ 시점 body_pos_w 는 stale — 홈을 물리로 확정한 뒤 읽는다(grasp 규약).
        _q_home = self.robot.data.default_joint_pos.clone()
        self.robot.write_joint_state_to_sim(_q_home, torch.zeros_like(_q_home))
        self.robot.set_joint_position_target(_q_home)
        self.scene.write_data_to_sim()
        for _ in range(2):
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        self.src.home = self._palm_pose_6d(self.src)
        self.rcv.home = self._palm_pose_6d(self.rcv)
        for rig, label in ((self.src, "source"), (self.rcv, "receiver")):
            in_box = ((rig.home[:, :3] >= rig.lo[:, :3])
                      & (rig.home[:, :3] <= rig.hi[:, :3])).all()
            if not bool(in_box):
                raise RuntimeError(
                    f"[{rig.profile.name}] {label} 홈 palm 위치가 박스 밖:\n"
                    f"  홈 {[round(v, 3) for v in rig.home[0, :3].tolist()]}\n"
                    f"  lo {[round(v, 3) for v in rig.lo[0, :3].tolist()]}\n"
                    f"  hi {[round(v, 3) for v in rig.hi[0, :3].tolist()]}")
            rig.anchor.copy_(rig.home)
            rig.cmd.copy_(rig.home)
            rig.scale.copy_(torch.maximum(rig.hi - rig.home, rig.home - rig.lo))

        # ---- 관절/한계 버퍼 ----------------------------------------------------------
        self._default_q = self.robot.data.default_joint_pos.clone()
        for rig in (self.src, self.rcv):
            rig.hand_hold = self._default_q[:, rig.hand_t].clone()

        # ---- slew ------------------------------------------------------------------
        _sp_ = float(cfg.palm_slew_pos)
        _sr = float(cfg.palm_slew_rot_deg) * _DEG
        self._slew = torch.tensor([_sp_, _sp_, _sp_, _sr, _sr, _sr], device=dev)

        A = cfg.action_space
        self.actions = torch.zeros(N, A, device=dev)
        self.prev_actions = torch.zeros(N, A, device=dev)

        # ---- warm 뱅크 ---------------------------------------------------------------
        self._load_warm_banks()

        # ---- 비드 판정 상태 -----------------------------------------------------------
        k = int(cfg.bead_count)
        geom = BeadGeometry(
            inner_radius=float(cfg.cup_inner_radius),
            inside_z_min=float(cfg.cup_inside_z_min),
            inside_z_max=float(cfg.cup_inside_z_max),
            mouth_z=float(cfg.cup_mouth_z))
        self._geom = geom
        self._prev_tgt_z = torch.full((N, k), -1e6, device=dev)
        self._crossed = torch.zeros(N, k, dtype=torch.bool, device=dev)
        self._prev_in_tgt = torch.zeros(N, device=dev)
        self._prev_in_src = torch.ones(N, device=dev)
        self._prev_spill = torch.zeros(N, device=dev)
        self._flags_fresh = torch.ones(N, dtype=torch.bool, device=dev)

        # ---- 인계/판정 플래그 ---------------------------------------------------------
        self._captured = torch.zeros(N, dtype=torch.bool, device=dev)
        self._dropped_now = torch.zeros(N, dtype=torch.bool, device=dev)
        self._drop_src = torch.zeros(N, dtype=torch.bool, device=dev)
        self._drop_rcv = torch.zeros(N, dtype=torch.bool, device=dev)
        self._success_now = torch.zeros(N, dtype=torch.bool, device=dev)
        self._success_streak = torch.zeros(N, dtype=torch.long, device=dev)

        self.adr = _adr.TaskADR(
            {"success": {"fill_ratio": (cfg.adr_fill_initial, cfg.adr_fill_final)}},
            num_increments=cfg.adr_num_increments,
            increment_interval=cfg.adr_increment_interval,
            trigger_threshold=cfg.adr_trigger_threshold,
            enabled=bool(cfg.enable_adr),
        )

        print(
            f"[pour_fabric] pair={self.pair.name} asset={self.pair.asset.name} "
            f"src={self.src.profile.name} rcv={self.rcv.profile.name} "
            f"receiver_mode={cfg.receiver_control_mode} "
            f"action={A} obs={cfg.observation_space} critic={cfg.state_space} "
            f"warm={'yes' if self._warm_src is not None else 'PROBE-ONLY'} "
            f"beads={k}", flush=True)

    # ------------------------------------------------------------------
    def _one_body(self, profile, name: str) -> int:
        ids, _ = self.robot.find_bodies(name)
        if len(ids) != 1:
            raise RuntimeError(f"[{profile.name}] body '{name}' 해석 실패: {ids}")
        return ids[0]

    def _palm_pose_6d(self, rig: _SideRig) -> torch.Tensor:
        from isaaclab.utils.math import euler_xyz_from_quat
        pos = self.robot.data.body_pos_w[:, rig.palm_idx] - self.scene.env_origins
        r, pi, y = euler_xyz_from_quat(self.robot.data.body_quat_w[:, rig.palm_idx])
        return torch.cat([pos, torch.stack([y, pi, r], dim=1)], dim=1)

    # ==================================================================
    def _load_warm_banks(self) -> None:
        cfg = self.cfg
        self._warm_src = self._warm_rcv = None
        if not cfg.require_warm_bank:
            print("[pour_fabric] ⚠⚠ require_warm_bank=False — **probe 전용** 부팅이다. "
                  "홈+테이블 컵(파지 없음)으로 시작하므로 학습에 쓰면 안 된다.", flush=True)
            return
        usd = self.pair.asset.usd_relpath
        for attr, path, rig in (("_warm_src", cfg.warm_bank_source_path, self.src),
                                ("_warm_rcv", cfg.warm_bank_receiver_path, self.rcv)):
            bank = PourWarmBank.load(
                path,
                expect_robot_usd=usd,
                expect_gravity=bool(cfg.enable_gravity),
                expect_self_collisions=bool(cfg.enable_self_collisions),
                min_states=int(cfg.warm_bank_min_states))
            # 뱅크 관절 이름 → articulation 인덱스 (이름 기반, fail-loud)
            idx = []
            for jn in bank.joint_names:
                ids, _ = self.robot.find_joints(jn)
                if len(ids) != 1:
                    raise RuntimeError(
                        f"{bank.path}: 뱅크 관절 '{jn}' 해석 실패 — 다른 자산의 뱅크다.")
                idx.append(ids[0])
            rig.bank_joint_t = torch.tensor(idx, device=self.device, dtype=torch.long)
            rig.bank_joint_pos = torch.tensor(bank.joint_pos, device=self.device)
            rig.bank_cup_pose = torch.tensor(bank.cup_pose, device=self.device)
            rig.bank_beads = (torch.tensor(bank.bead_state, device=self.device)
                              if bank.bead_state is not None else None)
            setattr(self, attr, bank)
            print(f"[pour_fabric] warm 뱅크 로드: {path} — {len(bank)}개 "
                  f"(beads={'yes' if bank.bead_state is not None else 'no'})", flush=True)
        if self._warm_src is not None and self.src.bank_beads is None:
            print("[pour_fabric] ⚠ source 뱅크에 bead_state 가 없다 — "
                  "리셋마다 컵 내부 격자 스폰으로 대체(정착은 hold 가 담당).", flush=True)

    # ==================================================================
    def _setup_fabrics(self) -> None:
        initialize_warp(str(self.device)[-1])
        self._world = WorldMeshesModel(
            batch_size=self.num_envs, device=self.device,
            max_objects_per_env=int(self.cfg.fabrics_max_objects_per_env))
        self._world_ids, self._world_indicator = self._world.get_object_ids()

        for rig in (self.src, self.rcv):
            p = rig.profile
            cls = _fabric_class(p.fabric_class)
            rig.fabric = cls(
                batch_size=self.num_envs, device=self.device,
                timestep=float(self.cfg.fabrics_dt),
                graph_capturable=bool(self.cfg.fabric_use_cuda_graph),
                use_hand_fabric=False,
                robot_dir_name=p.fabric_robot_dir,
                robot_name=p.fabric_robot_dir)
            rig.integrator = DisplacementIntegrator(rig.fabric)

            n_j = rig.fabric.num_joints
            expect = p.num_arm_joints + p.num_hand_joints
            if n_j != expect:
                raise RuntimeError(
                    f"[{p.name}] fabric num_joints={n_j} != 프로필 {expect}")
            # ★fabric 관절 순서는 finger-major — 프로필 인덱스로 재조립(grasp 규약).
            idx = []
            for name in p.fabric_joint_order:
                ids, _ = self.robot.find_joints(name)
                if len(ids) != 1:
                    raise RuntimeError(f"[{p.name}] fabric 관절 '{name}' 해석 실패: {ids}")
                idx.append(ids[0])
            rig.fab_t = torch.tensor(idx, device=self.device, dtype=torch.long)
            rig.fabric_q = self.robot.data.default_joint_pos[:, rig.fab_t].contiguous()
            rig.fabric_qd = torch.zeros(self.num_envs, n_j, device=self.device)
            rig.fabric_qdd = torch.zeros(self.num_envs, n_j, device=self.device)
            rig.fabric_hand_cmd = torch.zeros(self.num_envs, 5, device=self.device)
            # cspace rest = 프로필 홈(리셋 시 warm 자세로 per-env 재앵커).
            rig.fabric.default_config.copy_(rig.fabric_q)
        self._fabric_damping = float(self.cfg.fabrics_damping_gain) * torch.ones(
            self.num_envs, 1, device=self.device)

    # ==================================================================
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        fix = self.cfg.env_fixture_spawn
        fix.func(_cfg.ENV_FIXTURE_PRIM, fix, translation=(0.0, 0.0, 0.0))

        # 손가락 마디별 접촉 센서 — body 마다 개별, **자기 쪽 컵만** 필터(grasp 규약).
        for rig, flt in ((self.pair.source, list(self.cfg.source_contact_filter)),
                         (self.pair.receiver, list(self.cfg.receiver_contact_filter))):
            store: dict = {}
            for finger in rig.fingers:
                roles = {"tip": [], "wrap": []}
                for role, bodies in (("tip", rig.finger_tip_bodies[finger]),
                                     ("wrap", rig.finger_wrap_bodies.get(finger, ()))):
                    for body in bodies:
                        s = ContactSensor(ContactSensorCfg(
                            prim_path=f"/World/envs/env_.*/Robot/{body}",
                            filter_prim_paths_expr=flt,
                            history_length=1, track_air_time=False))
                        roles[role].append(s)
                        self.scene.sensors[f"contact_{rig.side}_{finger}_{body}"] = s
                store[finger] = roles
            if rig.side == self.pair.source.side:
                self._src_sensor_store = store
            else:
                self._rcv_sensor_store = store

        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, float(self.cfg.ground_plane_z)))
        light = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

        self.scene.clone_environments(copy_from_source=True)
        self.source_cup = RigidObject(self.cfg.source_cup_cfg)
        self.scene.rigid_objects["source_cup"] = self.source_cup
        self.receiver_cup = RigidObject(self.cfg.receiver_cup_cfg)
        self.scene.rigid_objects["receiver_cup"] = self.receiver_cup
        self.beads = RigidObjectCollection(self.cfg.beads_cfg)
        self.scene.rigid_object_collections["beads"] = self.beads

    # ==================================================================
    def _hold_mask(self) -> torch.Tensor:
        return self.episode_length_buf < int(self.cfg.hold_steps)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        cfg = self.cfg
        self.actions = actions.clamp(-1.0, 1.0)
        hold = self._hold_mask()

        # ---- capture: hold 종료 시점(버퍼 fresh)에 앵커·기준선 확정 -------------------
        cap = (self.episode_length_buf == int(cfg.hold_steps)) & (~self._captured)
        cap_f = cap.unsqueeze(1).float()
        for rig, cup in ((self.src, self.source_cup), (self.rcv, self.receiver_cup)):
            pose = self._palm_pose_6d(rig)
            rig.anchor = torch.where(cap.unsqueeze(1), pose, rig.anchor)
            rig.cmd = torch.where(cap.unsqueeze(1), pose, rig.cmd)
            new_scale = torch.maximum(rig.hi - rig.anchor, rig.anchor - rig.lo)
            rig.scale = cap_f * new_scale + (1.0 - cap_f) * rig.scale
            cup_pos = cup.data.root_pos_w - self.scene.env_origins
            d = (pose[:, :3] - cup_pos).norm(dim=-1)
            rig.ref_palm_cup = torch.where(cap, d, rig.ref_palm_cup)
            rig.ref_cup_z = torch.where(cap, cup_pos[:, 2], rig.ref_cup_z)
            # ★cspace rest 를 warm 자세로 재앵커 — nullspace α 채널의 대체물.
            q_fab = self.robot.data.joint_pos[:, rig.fab_t]
            rig.fabric.default_config.copy_(
                torch.where(cap.unsqueeze(1), q_fab, rig.fabric.default_config))
        self._captured |= cap

        # ---- 목표 조립 (절대 + 앵커 + slew, grasp 규약) -------------------------------
        ramp_len = max(int(cfg.handover_ramp_steps), 1)
        t_ramp = ((self.episode_length_buf - int(cfg.hold_steps)).float()
                  / float(ramp_len)).clamp(0.0, 1.0).unsqueeze(1)

        a_src = self.actions[:, :6]
        a_rcv = self.actions[:, 6:9]
        for rig, a in ((self.src, a_src), (self.rcv, a_rcv)):
            if rig is self.rcv and cfg.receiver_control_mode == "frozen":
                desired = rig.anchor.clone()
            else:
                full_a = torch.zeros(self.num_envs, 6, device=self.device)
                full_a[:, : a.shape[1]] = a          # receiver 는 위치 3축만, 자세 0(=앵커)
                if bool(cfg.symmetric_action_scale):
                    desired = (rig.anchor + full_a * rig.scale).clamp(rig.lo, rig.hi)
                else:
                    desired = rig.anchor + torch.where(
                        full_a >= 0.0, full_a * (rig.hi - rig.anchor),
                        full_a * (rig.anchor - rig.lo))
                # receiver 자세는 항상 앵커 고정(직립 유지 — 액션 없음)
                if rig is self.rcv:
                    desired[:, 3:] = rig.anchor[:, 3:]
            # ramp: 앵커 → 정책 목표 선형 보간 (hold 중엔 t=0 → 앵커)
            desired = rig.anchor + t_ramp * (desired - rig.anchor)
            d = (desired - rig.cmd).clamp(-self._slew, self._slew)
            rig.cmd = rig.cmd + d
            rig.fabric.set_features(
                rig.fabric_hand_cmd, rig.cmd, "euler_zyx",
                rig.fabric_q.detach(), rig.fabric_qd.detach(),
                self._world_ids, self._world_indicator, self._fabric_damping)

        self._step_fabric(hold)

    def _step_fabric(self, hold: torch.Tensor) -> None:
        """정책 스텝당 fabric_decimation 회 적분. hold 중인 env 는 warm 자세에 고정.

        ★적분기는 배치 전체를 돌리므로 hold env 는 적분 **후** 원위치로 되돌린다
          (마스크 곱 — `if tensor.any()` 는 GPU sync 를 강제한다, util killer 재발 방지).
        """
        h = hold.unsqueeze(1).float()
        for rig in (self.src, self.rcv):
            q_pin = rig.fabric_q
            for _ in range(int(self.cfg.fabric_decimation)):
                rig.fabric_q, rig.fabric_qd, rig.fabric_qdd = rig.integrator.step(
                    rig.fabric_q.detach(), rig.fabric_qd.detach(),
                    rig.fabric_qdd.detach(), float(self.cfg.fabrics_dt))
            rig.fabric_q = h * q_pin + (1.0 - h) * rig.fabric_q
            rig.fabric_qd = (1.0 - h) * rig.fabric_qd
            rig.fabric_qdd = (1.0 - h) * rig.fabric_qdd

    def _apply_action(self) -> None:
        for rig in (self.src, self.rcv):
            arm_target = rig.fabric_q[:, : rig.profile.num_arm_joints]
            self.robot.set_joint_position_target(arm_target, joint_ids=rig.arm_ids)
            self.robot.set_joint_velocity_target(
                torch.zeros_like(arm_target), joint_ids=rig.arm_ids)
            # 손 전체 동결 — warm 파지 자세 PD hold (재조임 없음)
            self.robot.set_joint_position_target(rig.hand_hold, joint_ids=rig.hand_ids)
        self._apply_gravity_compensation()

    def _apply_gravity_compensation(self) -> None:
        """중력보상 피드포워드 — grasp_lift_fabric 과 동일(양팔+head 전 관절 커버)."""
        if self._grav_comp <= 0.0:
            return
        tau = self.robot.root_physx_view.get_gravity_compensation_forces()
        self.robot.set_joint_effort_target(self._grav_comp * tau[:, : self.robot.num_joints])

    # ==================================================================
    def _contact(self, store: dict, fingers) -> tuple[torch.Tensor, torch.Tensor]:
        """(손가락별 총 접촉력 (N,F), wrap 접촉 여부 (N,F)) — grasp 규약."""
        thr = float(self.cfg.contact_force_threshold)
        tot, wrapped = [], []
        for f in fingers:
            roles = store[f]
            t = torch.zeros(self.num_envs, device=self.device)
            w = torch.zeros(self.num_envs, device=self.device)
            for s in roles["tip"]:
                t = t + s.data.force_matrix_w.view(self.num_envs, -1, 3).sum(1).norm(dim=-1)
            for s in roles["wrap"]:
                m = s.data.force_matrix_w.view(self.num_envs, -1, 3).sum(1).norm(dim=-1)
                t = t + m
                w = torch.maximum(w, m)
            tot.append(t)
            wrapped.append((w > thr).float())
        return torch.stack(tot, 1), torch.stack(wrapped, 1)

    def _local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return pos_w - self.scene.env_origins

    def _cup_up(self, cup: RigidObject) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply
        z = torch.zeros(self.num_envs, 3, device=self.device)
        z[:, 2] = 1.0
        return quat_apply(cup.data.root_quat_w, z)

    def _mouth_w(self, cup: RigidObject) -> torch.Tensor:
        """컵 개구(림) 중심 world 좌표 — FK+컵 pose 파생(실기 재현 가능)."""
        from isaaclab.utils.math import quat_apply
        off = torch.zeros(self.num_envs, 3, device=self.device)
        off[:, 2] = float(self.cfg.cup_mouth_z)
        return cup.data.root_pos_w + quat_apply(cup.data.root_quat_w, off)

    def _aim_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(aim_delta(3), aim_dist, pour_dir_cos)."""
        mouth = self._mouth_w(self.source_cup)
        opening = self._mouth_w(self.receiver_cup)
        opening = opening + torch.tensor(
            [0.0, 0.0, float(self.cfg.aim_height_offset)], device=self.device)
        delta = mouth - opening
        dist = delta.norm(dim=-1)
        up_xy = self._cup_up(self.source_cup)[:, :2]
        to_tgt = (self.receiver_cup.data.root_pos_w
                  - self.source_cup.data.root_pos_w)[:, :2]
        cos = (up_xy * to_tgt).sum(-1) / (
            up_xy.norm(dim=-1) * to_tgt.norm(dim=-1) + 1e-6)
        return delta, dist, cos

    def _tilt_deg(self) -> torch.Tensor:
        up = self._cup_up(self.source_cup)
        return torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))

    # ==================================================================
    def _get_observations(self) -> dict:
        from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        tau = self.robot.data.applied_torque
        idx = torch.cat([self.src.arm_t, self.src.hand_t,
                         self.rcv.arm_t, self.rcv.hand_t])
        joint_pos, joint_vel, joint_eff = q[:, idx], qd[:, idx], tau[:, idx]

        parts = [joint_pos, joint_vel, joint_eff]
        contacts = []
        for rig, store, cup in (
                (self.src, self._src_sensor_store, self.source_cup),
                (self.rcv, self._rcv_sensor_store, self.receiver_cup)):
            c, _ = self._contact(store, rig.fingers)
            contacts.append(c.clamp(max=20.0))
            palm_pos_w = self.robot.data.body_pos_w[:, rig.palm_idx]
            palm_quat_w = self.robot.data.body_quat_w[:, rig.palm_idx]
            p, r = subtract_frame_transforms(
                palm_pos_w, palm_quat_w,
                cup.data.root_pos_w, cup.data.root_quat_w)
            parts += [p, r]
        parts = parts[:3] + contacts + parts[3:]

        aim_delta, _, _ = self._aim_geometry()
        parts.append(aim_delta)
        parts.append(self._cup_up(self.source_cup))
        parts.append(self.actions)
        # slew 지령 상태 — 앵커 기준 정규화(grasp 의 cmd_rel 규약을 앵커로 일반화).
        cmd_rel_src = ((self.src.cmd - self.src.anchor)
                       / self.src.scale.clamp(min=1e-6)).clamp(-2.0, 2.0)
        cmd_rel_rcv = ((self.rcv.cmd[:, :3] - self.rcv.anchor[:, :3])
                       / self.rcv.scale[:, :3].clamp(min=1e-6)).clamp(-2.0, 2.0)
        parts += [cmd_rel_src, cmd_rel_rcv]
        obs = torch.cat(parts, dim=1)

        # ---- critic: 비드 ground truth (actor 금지 — 실기에 없다) --------------------
        bead_fracs = torch.stack(
            [self._prev_in_src, self._prev_in_tgt, self._prev_spill,
             self._crossed.float().mean(dim=-1)], dim=1)
        centroid_rel = self.beads.data.object_pos_w.mean(dim=1) \
            - self.receiver_cup.data.root_pos_w
        centroid_local = quat_apply_inverse(
            self.receiver_cup.data.root_quat_w, centroid_rel)
        state = torch.cat([
            obs, bead_fracs, centroid_local,
            self.source_cup.data.root_lin_vel_w, self.source_cup.data.root_ang_vel_w,
            self.receiver_cup.data.root_lin_vel_w, self.receiver_cup.data.root_ang_vel_w,
        ], dim=1)
        return {"policy": torch.nan_to_num(obs), "critic": torch.nan_to_num(state)}

    # ==================================================================
    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg

        src_c, src_w = self._contact(self._src_sensor_store, self.src.fingers)
        rcv_c, rcv_w = self._contact(self._rcv_sensor_store, self.rcv.fingers)
        thr = float(cfg.contact_force_threshold)
        src_grip = (src_c > thr).float().mean(dim=1)
        rcv_grip = (rcv_c > thr).float().mean(dim=1)
        src_env = (src_w.mean(dim=1) if self.src.profile.has_wrap_sensors else src_grip)
        rcv_env = (rcv_w.mean(dim=1) if self.rcv.profile.has_wrap_sensors else rcv_grip)

        # ---- 비드 분류 (pour_v1 이식, 순수 함수) --------------------------------------
        flags = compute_bead_flags(
            bead_pos_w=self.beads.data.object_pos_w,
            source_pos_w=self.source_cup.data.root_pos_w,
            source_quat_w=self.source_cup.data.root_quat_w,
            target_pos_w=self.receiver_cup.data.root_pos_w,
            target_quat_w=self.receiver_cup.data.root_quat_w,
            geom_source=self._geom, geom_target=self._geom,
            prev_target_local_z=self._prev_tgt_z,
            crossed_mask=self._crossed)
        fresh = self._flags_fresh.float()
        d_in_target = (1.0 - fresh) * (flags.in_target_frac - self._prev_in_tgt)
        d_released = (1.0 - fresh) * (self._prev_in_src - flags.in_source_frac)
        d_spill = (1.0 - fresh) * (flags.spill_frac - self._prev_spill)
        self._prev_in_tgt = flags.in_target_frac
        self._prev_in_src = flags.in_source_frac
        self._prev_spill = flags.spill_frac
        self._prev_tgt_z = flags.target_local_z
        self._crossed = flags.crossed_mask
        self._flags_fresh[:] = False

        aim_delta, aim_dist, dir_cos = self._aim_geometry()
        tilt = self._tilt_deg()

        # ---- 성공/낙하 판정 -----------------------------------------------------------
        fill_thr = float(self.adr.get_param("success", "fill_ratio")) \
            if cfg.enable_adr else float(cfg.success_fill_ratio)
        xy_dist = (self.source_cup.data.root_pos_w[:, :2]
                   - self.receiver_cup.data.root_pos_w[:, :2]).norm(dim=-1)
        self._success_now = (
            (flags.in_target_frac >= fill_thr)
            & (flags.spill_frac <= float(cfg.success_spill_max))
            & (xy_dist < float(cfg.success_xy_thresh)))
        self._success_streak = torch.where(
            self._success_now, self._success_streak + 1,
            torch.zeros_like(self._success_streak))

        # 낙하 판정은 _get_dones(이번 스텝, rewards 보다 먼저 호출됨)가 계산했다 —
        # 같은 스텝에 종료와 일회 페널티가 일치한다.
        dropped = self._dropped_now

        total, terms, gates = compute_rewards(
            src_envelope_frac=src_env, src_grip_frac=src_grip,
            src_group_a_force=src_c[:, self.src.grp_a],
            src_group_b_force=src_c[:, self.src.grp_b],
            rcv_envelope_frac=rcv_env, rcv_grip_frac=rcv_grip,
            rcv_group_a_force=rcv_c[:, self.rcv.grp_a],
            rcv_group_b_force=rcv_c[:, self.rcv.grp_b],
            aim_dist=aim_dist, tilt_deg=tilt, pour_dir_cos=dir_cos,
            d_in_target=d_in_target, d_released=d_released, d_spill=d_spill,
            success_now=self._success_now, dropped_now=self._dropped_now,
            actions=self.actions, prev_actions=self.prev_actions, cfg=cfg)

        # ---- 지터 계측(prev 갱신 전 — grasp 규약) --------------------------------------
        self.extras["action/arm_step_delta"] = (
            self.actions[:, :6] - self.prev_actions[:, :6]).abs().mean()
        self.prev_actions.copy_(self.actions)

        if self.adr.maybe_increment(self._success_now.float().mean()):
            pass

        # ---- 로깅 ----------------------------------------------------------------
        for k, v in terms.items():
            self.extras[f"reward/{k}"] = v.mean()
        self.extras["reward/total"] = total.mean()
        self.extras["task/gate_src"] = gates["gate_src"].float().mean()
        self.extras["task/gate_rcv"] = gates["gate_rcv"].float().mean()
        self.extras["task/src_envelope"] = src_env.mean()
        self.extras["task/rcv_envelope"] = rcv_env.mean()
        # ★접촉력 원값 — grip=0 이 "미접촉"인지 "임계 아래"인지 구분(fab_test1 교훈).
        self.extras["contact/src_best"] = src_c.max(dim=1).values.mean()
        self.extras["contact/rcv_best"] = rcv_c.max(dim=1).values.mean()
        self.extras["contact/src_max"] = src_c.max()
        self.extras["bead/in_source"] = flags.in_source_frac.mean()
        self.extras["bead/in_target"] = flags.in_target_frac.mean()
        self.extras["bead/spill"] = flags.spill_frac.mean()
        self.extras["bead/crossed"] = flags.crossed_frac.mean()
        self.extras["task/aim_dist"] = aim_dist.mean()
        self.extras["task/tilt_deg"] = tilt.mean()
        self.extras["task/tilt_p95"] = torch.quantile(tilt, 0.95)
        self.extras["task/dir_cos"] = dir_cos.mean()
        self.extras["task/success_now"] = self._success_now.float().mean()
        self.extras["task/episode_success"] = (
            self._success_streak >= int(cfg.success_hold_steps)).float().mean()
        self.extras["done/drop_src"] = self._drop_src.float().mean()
        self.extras["done/drop_rcv"] = self._drop_rcv.float().mean()
        # Fabrics 추종 — 지령 대비 실제 palm(78mm 정상상태오차 이력 감시)
        for rig, label in ((self.src, "src"), (self.rcv, "rcv")):
            perr = (rig.cmd[:, :3]
                    - self._local(self.robot.data.body_pos_w[:, rig.palm_idx])).norm(dim=-1)
            self.extras[f"fabric/{label}_palm_err"] = perr.mean()
        self.extras.update(self.adr.log_dict())
        self.extras["adr/trigger_metric"] = self._success_now.float().mean()

        self._log_tick = getattr(self, "_log_tick", 0) + 1
        _every = int(getattr(cfg, "console_log_interval", 600))
        if _every > 0 and self._log_tick % _every == 0:
            print(
                f"[METRICS] step={self._log_tick:>8d}"
                f" rew={total.mean():+.3f}"
                f" gS={gates['gate_src'].float().mean():.2f}"
                f" gR={gates['gate_rcv'].float().mean():.2f}"
                f" aim={aim_dist.mean():.3f}"
                f" tilt={tilt.mean():.1f}"
                f" inT={flags.in_target_frac.mean():.3f}"
                f" inS={flags.in_source_frac.mean():.3f}"
                f" spill={flags.spill_frac.mean():.3f}"
                f" FS={src_c.max(dim=1).values.mean():.1f}N"
                f" FR={rcv_c.max(dim=1).values.mean():.1f}N"
                f" drop={dropped.float().mean():.3f}"
                f" succ={self._success_now.float().mean():.3f}",
                flush=True)
        return total

    # ==================================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """종료 = runaway + 컵 낙하. grasp 와 의도적으로 다르다:
        grasp 의 fell→리스폰은 재파지가 가능해서였고, pour 의 낙하는 회복 불가라
        방치하면 '빈손 조준' farming 이 열린다(pour_v1 문서화 근거). 계약 테스트가 pin.

        ★DirectRLEnv 는 _get_dones 를 _get_rewards **보다 먼저** 부른다(:390-392).
          낙하 판정을 여기서 계산해 두면 같은 스텝에 종료와 일회 페널티가 일치한다
          (rewards 에서 계산하면 종료가 한 스텝 늦는다).
        """
        cfg = self.cfg
        hold = self._hold_mask()
        # 낙하 = 기하 판정(pour_v1 left_cup_drop 이식, 양쪽 적용). capture 전엔 억제.
        for rig, cup, attr in ((self.src, self.source_cup, "_drop_src"),
                               (self.rcv, self.receiver_cup, "_drop_rcv")):
            palm = self._local(self.robot.data.body_pos_w[:, rig.palm_idx])
            cup_p = self._local(cup.data.root_pos_w)
            d = (palm - cup_p).norm(dim=-1)
            far = (d - rig.ref_palm_cup) > float(cfg.drop_dist_m)
            fell = (rig.ref_cup_z - cup_p[:, 2]) > float(cfg.drop_z_m)
            setattr(self, attr, (far | fell) & self._captured & (~hold))
        self._dropped_now = self._drop_src | self._drop_rcv

        qd = torch.cat([self.robot.data.joint_vel[:, self.src.arm_t],
                        self.robot.data.joint_vel[:, self.rcv.arm_t]], dim=1)
        runaway = (qd.abs() > float(self.cfg.runaway_joint_vel)).any(dim=-1)
        terminated = runaway | self._dropped_now
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["task/runaway_rate"] = runaway.float().mean()
        return terminated, truncated

    # ==================================================================
    def _reset_idx(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        dev = self.device

        q0 = self._default_q[env_ids].clone()
        cup_poses = {}
        if self._warm_src is not None:
            for rig, bank_len in ((self.src, len(self._warm_src)),
                                  (self.rcv, len(self._warm_rcv))):
                pick = torch.randint(bank_len, (n,), device=dev)
                q0[:, rig.bank_joint_t] = rig.bank_joint_pos[pick]
                cup_poses[rig] = rig.bank_cup_pose[pick]
                rig._last_pick = pick
        else:
            # probe 전용: 컵을 스폰 중심의 작업면 위에 둔다(파지 없음).
            rest_z = _cfg.POUR_CUP_ORIGIN_OFFSET_Z + float(self.src.profile.surface_z)
            for rig in (self.src, self.rcv):
                pose = torch.zeros(n, 7, device=dev)
                pose[:, 0] = rig.profile.object_spawn_center[0]
                pose[:, 1] = rig.profile.object_spawn_center[1]
                pose[:, 2] = rest_z + 0.002
                pose[:, 3] = 1.0
                cup_poses[rig] = pose

        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), env_ids=env_ids)
        self.robot.set_joint_position_target(q0, env_ids=env_ids)
        for rig in (self.src, self.rcv):
            rig.hand_hold[env_ids] = q0[:, rig.hand_t]
            rig.fabric_q[env_ids] = q0[:, rig.fab_t]
            rig.fabric_qd[env_ids] = 0.0
            rig.fabric_qdd[env_ids] = 0.0
            # 앵커/지령은 capture 전까지 홈 — obs 의 cmd_rel 이 0 근방에 머물게.
            rig.anchor[env_ids] = rig.home[env_ids]
            rig.cmd[env_ids] = rig.home[env_ids]

        for rig, cup in ((self.src, self.source_cup), (self.rcv, self.receiver_cup)):
            root = torch.zeros(n, 13, device=dev)
            root[:, :3] = cup_poses[rig][:, :3] + self.scene.env_origins[env_ids]
            root[:, 3:7] = cup_poses[rig][:, 3:7]
            cup.write_root_state_to_sim(root, env_ids=env_ids)

        # ---- 비드 ----------------------------------------------------------------
        k = int(self.cfg.bead_count)
        if self._warm_src is not None and self.src.bank_beads is not None:
            bead = self.src.bank_beads[self.src._last_pick].clone()   # (n,k,13) env-local
            bead[:, :, :3] += self.scene.env_origins[env_ids].unsqueeze(1)
        else:
            # ★검증된 단일 배치만 사용(bead_assets 08.17 사고 — 새 배치 금지).
            from isaaclab.utils.math import quat_apply
            from openarm.common.bead_assets import bead_offsets_in_cup
            offs = torch.tensor(bead_offsets_in_cup(k), device=dev)     # (k,3)
            src_pose = cup_poses[self.src]
            quat = src_pose[:, 3:7].unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
            off_w = quat_apply(quat, offs.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3))
            bead = torch.zeros(n, k, 13, device=dev)
            bead[:, :, :3] = (src_pose[:, :3].unsqueeze(1)
                              + off_w.view(n, k, 3)
                              + self.scene.env_origins[env_ids].unsqueeze(1))
            bead[:, :, 3] = 1.0
        self.beads.write_object_state_to_sim(bead, env_ids=env_ids)

        # ---- 판정/인계 상태 초기화 --------------------------------------------------
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._captured[env_ids] = False
        self._dropped_now[env_ids] = False
        self._success_now[env_ids] = False
        self._success_streak[env_ids] = 0
        self._crossed[env_ids] = False
        self._prev_tgt_z[env_ids] = -1e6
        self._prev_in_tgt[env_ids] = 0.0
        self._prev_in_src[env_ids] = 1.0
        self._prev_spill[env_ids] = 0.0
        self._flags_fresh[env_ids] = True
