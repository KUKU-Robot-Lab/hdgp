"""pour_fabric_mimic — 양팔 잡기→들기→붓기, **저차원(언더액추·PhysX mimic) 손 전용**.

★`pour_fabric/pour_fabric_env.py` 사본(09.14, 사용자 결정 "원본 냅두고 저차원 핸드 전용 신설").
  다른 점: (1) 양팔 26 DOF fabric 두 인스턴스가 서로의 슬라이스를 매 스텝 동기화(`sync_other`)
  (2) 부팅에서 언더액추 계약 검사 — URDF `<mimic>` 표 ↔ USD `physxMimicJoint` gearing 대조,
      종속관절 한계 확장, open≠grip·grip∈한계 (grasp_fj_rh 09.07 이식)
  (3) `ctrl/mimic_err_max` 등 mimic 건전성 지표.
  보상·관측·성공 판정 구조는 원본과 같다(RewardContext 동일 → 같은 t2r 프롬프트 골격).

(아래는 원본 설명)

★09.13 재작성. 보상은 여기 없다 — `RewardContext` 를 채워 `cfg.reward_code_path` 의
  생성 코드(`modules/t2r`)에 넘긴다. 성공 판정(fill·spill·xy)은 env 가 계산하고 ctx 에
  넣는다: 보상이 바꿀 수 없는 **기준 지표**다.

에피소드: reset(테이블 위 컵 2 + 소스 컵 안 비드) → hold(비드 정착, 팔 고정) → 정책.
제어: 팔 = palm 6D(앵커+델타) → Fabrics → 관절 PD. 손 = 시너지 폐쇄도(접촉 동결).
"""

from __future__ import annotations

import glob
import math
import os
import sys
import xml.etree.ElementTree as ET

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import bind_physics_material, find_matching_prim_paths
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from openarm.agnostic.modules.adr import TaskADR
from openarm.agnostic.modules.object_wrench import WrenchDR
from openarm.agnostic.modules.perception_delay import noisy_pose
from openarm.agnostic.modules.t2r.context import RewardContext
from openarm.agnostic.modules.t2r.loader import call_reward_fn, load_reward_fn
from openarm.common.bead_assets import bead_offsets_in_cup

from . import bimanual as _bm
from . import pour_fabric_env_cfg as _cfg
from .bead_flags import BeadGeometry, compute_bead_flags
from .pour_fabric_env_cfg import PHYSICS_ADR_TERMINAL, PourFabricMimicEnvCfg
from .side_rig import SideRig

_FABRICS_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "FABRICS", "src"))
if os.path.isdir(_FABRICS_SRC) and _FABRICS_SRC not in sys.path:
    sys.path.insert(0, _FABRICS_SRC)

import fabrics_sim.fabrics.openarm_rh56f1_pose_fabric as _fab_rh   # noqa: E402
import fabrics_sim.fabrics.openarm_tesollo_pose_fabric as _fab_tes  # noqa: E402
from fabrics_sim.integrator.integrators import DisplacementIntegrator  # noqa: E402
from fabrics_sim.utils.utils import initialize_warp                   # noqa: E402
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel      # noqa: E402


def _fabric_class(name: str):
    for mod in (_fab_tes, _fab_rh):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise RuntimeError(f"Fabrics 클래스 '{name}' 를 찾을 수 없다")


#: PhysX mimic 제약이 종속관절 prim 에 남기는 속성. 규약 `q_dep + gearing·q_ref + offset = 0`
#: → **gearing = −multiplier**(임포터 정상 동작).
_MIMIC_GEARING_ATTR = "physxMimicJoint:rotZ:gearing"


class PourFabricMimicEnv(DirectRLEnv):
    cfg: PourFabricMimicEnvCfg

    # ==================================================================
    def __init__(self, cfg: PourFabricMimicEnvCfg, render_mode: str | None = None, **kw):
        _cfg.resolve_cfg(cfg)
        self.pair = _bm.get_pair(cfg.pair_name)
        self._grav_comp = float(cfg.gravity_compensation) if cfg.enable_gravity else 0.0
        sp = cfg.robot_cfg.spawn
        if (bool(sp.rigid_props.disable_gravity) == bool(cfg.enable_gravity)
                or bool(sp.articulation_props.enabled_self_collisions) != bool(cfg.enable_self_collisions)):
            raise RuntimeError("물리 스위치가 파생 cfg 에 반영되지 않았다 — resolve_cfg 경로 확인")
        self._reward_fn, self._reward_src = load_reward_fn(cfg.reward_code_path)
        super().__init__(cfg, render_mode, **kw)

        N, dev = self.num_envs, self.device
        # 소스=우(붓기)·리시버=좌. 대향 관절 미러 부호: thumb_2 축 Z → 좌측 −1.
        self.src = SideRig(self, self.pair.source, role="src",
                           sensors=self._sensor_store["src"], palm_sensor=self._palm_store["src"],
                           delta_lo=cfg.src_palm_delta_lo, delta_hi=cfg.src_palm_delta_hi,
                           oppose_sign=1.0)
        self.rcv = SideRig(self, self.pair.receiver, role="rcv",
                           sensors=self._sensor_store["rcv"], palm_sensor=self._palm_store["rcv"],
                           delta_lo=cfg.rcv_palm_delta_lo, delta_hi=cfg.rcv_palm_delta_hi,
                           oppose_sign=-1.0)
        self.rigs = (self.src, self.rcv)
        w_src, w_rcv = self.src.hand_action_width, self.rcv.hand_action_width
        if 6 + w_src != cfg.num_actions_per_side or 6 + w_rcv != cfg.num_actions_per_side:
            raise RuntimeError(f"손 액션 폭 불일치: cfg {cfg.num_actions_per_side - 6} vs "
                               f"src {w_src} / rcv {w_rcv}")

        # ---- 시작 자세 전체(두 팔 + 두 손 + head) ---------------------------------------------
        self._reset_q = self.robot.data.default_joint_pos[0].clone()
        for rig in self.rigs:
            self._reset_q[rig.arm_t] = rig.reset_q[rig.arm_t]
            self._reset_q[rig.syn_t] = rig.reset_q[rig.syn_t]
        for rig in self.rigs:
            rig.reset_q = self._reset_q.clone()

        # ---- Fabrics ×2 --------------------------------------------------------------------
        initialize_warp(str(dev)[-1])
        self._world = WorldMeshesModel(
            batch_size=N, device=dev, max_objects_per_env=int(cfg.fabrics_max_objects_per_env),
            world_dict=self._build_fabric_world())
        self._world_ids, self._world_indicator = self._world.get_object_ids()
        for rig in self.rigs:
            arm_sl, hand_sl = _bm.fabric_slots(rig.profile)
            rig.setup_fabric(_fabric_class(rig.profile.fabric_class), DisplacementIntegrator, arm_sl, hand_sl)
        # ---- 언더액추 계약(부팅 게이트) ----
        self._mimic_pairs = self._load_mimic_pairs()
        self._assert_mimic_constraints_present()
        self._assert_hand_pose_usable()
        self._widen_dependent_joint_limits()

        # ---- 시작 자세 실측(앵커·fabric 게이트) — 리셋 직후 버퍼는 stale, 물리 2스텝 뒤 읽는다 ----
        q0 = self._reset_q.unsqueeze(0).expand(N, -1).contiguous()
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        self.robot.set_joint_position_target(q0)
        self.scene.write_data_to_sim()
        for _ in range(2):
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        for rig in self.rigs:
            rig.init_anchor()

        A = cfg.action_space
        self._half = cfg.num_actions_per_side
        self.actions = torch.zeros(N, A, device=dev)
        self.prev_actions = torch.zeros(N, A, device=dev)
        self._policy_dt = float(cfg.sim.dt) * int(cfg.decimation)

        # ---- 비드 판정 상태 ------------------------------------------------------------------
        k = int(cfg.bead_count)
        self._geom = BeadGeometry(
            inner_radius=float(cfg.cup_inner_radius), inside_z_min=float(cfg.cup_inside_z_min),
            inside_z_max=float(cfg.cup_inside_z_max), mouth_z=float(cfg.cup_mouth_z))
        self._prev_tgt_z = torch.full((N, k), -1e6, device=dev)
        self._crossed = torch.zeros(N, k, dtype=torch.bool, device=dev)
        self._prev_in_tgt = torch.zeros(N, device=dev)
        self._prev_in_src = torch.ones(N, device=dev)
        self._prev_spill = torch.zeros(N, device=dev)
        self._flags_fresh = torch.ones(N, dtype=torch.bool, device=dev)
        # xy 검증 배치 유지 · z 는 바닥 기준 재적층(shaker 0.65: 원본 z 는 림 위로 나간다)
        _offs = torch.tensor(bead_offsets_in_cup(k), device=dev)
        _layer = torch.arange(k, device=dev) // 5
        _offs[:, 2] = -float(cfg.object_origin_offset_z) + float(cfg.bead_z_from_bottom) + float(cfg.bead_layer_dz) * _layer
        self._bead_offs = _offs
        if float(_offs[:, 2].max()) > float(cfg.cup_mouth_z) - 0.01:
            raise RuntimeError(f"[pour_fabric_mimic] 비드 최상층 z {float(_offs[:, 2].max()):.3f} 이 림 {cfg.cup_mouth_z:.3f} 에 닿는다")

        self._success_now = torch.zeros(N, dtype=torch.bool, device=dev)
        self._success_streak = torch.zeros(N, dtype=torch.long, device=dev)
        self._dropped = torch.zeros(N, dtype=torch.bool, device=dev)
        self._src_spawn = torch.zeros(N, 3, device=dev)
        self._rcv_spawn = torch.zeros(N, 3, device=dev)
        self._src_grasped = torch.zeros(N, dtype=torch.bool, device=dev)
        self._rcv_grasped = torch.zeros(N, dtype=torch.bool, device=dev)
        self._cups_nested = torch.zeros(N, dtype=torch.bool, device=dev)
        self._cups_center_dist = torch.zeros(N, device=dev)
        self._last_terms: dict = {}
        self._setup_dr()
        self._log_tick = 0

        print(f"[pour_fabric_mimic] pair={self.pair.name} usd={self.pair.usd_relpath} "
              f"src={self.src.profile.name} rcv={self.rcv.profile.name} "
              f"action={A} obs={cfg.observation_space} critic={cfg.state_space} "
              f"beads={k} reward={self._reward_src}", flush=True)

    # ==================================================================
    # s2r DR (09.14): 지각 지연·노이즈 · 물리 DR(EventTerm, ADR 확장) · 들린 컵 외란
    # ==================================================================
    def _setup_dr(self) -> None:
        cfg, N, dev = self.cfg, self.num_envs, self.device
        em = getattr(self, "event_manager", None)
        phys = {k: v for k, v in PHYSICS_ADR_TERMINAL.items()} if em is not None else {}
        self.adr = TaskADR(
            {"obs": {"object_xyz": (float(cfg.obs_noise_object_xyz), float(cfg.adr_obs_noise_object_xyz_max)),
                     "object_rot_deg": (float(cfg.obs_noise_object_rot_deg), float(cfg.adr_obs_noise_object_rot_max_deg)),
                     "delay_steps": (float(cfg.perception_delay_base_steps), float(cfg.perception_delay_max_steps))},
             "wrench": {"force_scale": (0.0, float(cfg.wrench_force_scale_max)),
                        "torque_scale": (0.0, float(cfg.wrench_torque_scale_max))}},
            num_increments=int(cfg.adr_num_increments), increment_interval=int(cfg.adr_increment_interval),
            trigger_threshold=float(cfg.adr_trigger_threshold), enabled=bool(cfg.enable_adr),
            event_manager=em, physics_cfg=phys)
        # 지각 링버퍼 (N, L, 7): pos3 + quat4. 지연 상한은 ADR 진행도로 0 → max.
        L = int(cfg.perception_delay_max_steps) + 1
        self._perc_buf = {k: torch.zeros(N, L, 7, device=dev) for k in ("src", "rcv")}
        self._perc_flush = torch.ones(N, dtype=torch.bool, device=dev)
        self._perc_arange = torch.arange(N, device=dev)
        self._wrench = {k: WrenchDR(N, dev, force_scale=1.0, torque_scale=1.0,
                                    prob_range=tuple(cfg.wrench_prob_range)) for k in ("src", "rcv")}
        self._cup_mass = {"src": torch.full((N,), _cfg.POUR_CUP_MASS, device=dev),
                          "rcv": torch.full((N,), _cfg.POUR_CUP_MASS, device=dev)}
        print(f"[pour_fabric_mimic] s2r DR: events={'on' if em is not None else 'OFF'} adr={bool(cfg.enable_adr)} "
              f"obs_noise(q {cfg.obs_noise_qpos} qd {cfg.obs_noise_qvel} body {cfg.obs_noise_body}) "
              f"perception delay 0→{cfg.perception_delay_max_steps} step · object noise → "
              f"{cfg.adr_obs_noise_object_xyz_max} m/{cfg.adr_obs_noise_object_rot_max_deg}° · "
              f"wrench → {cfg.wrench_force_scale_max} N/kg · cup friction {cfg.cup_friction_range}", flush=True)

    def _read_cup_mass(self, cup: RigidObject) -> torch.Tensor:
        """실제 질량 (N,) — 질량 DR 뒤 공칭으로 정규화하면 외란 비가 흔들린다(grasp_s2r 09.10)."""
        view = getattr(cup, "root_physx_view", None)
        m = None
        if view is not None:
            try:
                m = view.get_masses()
            except Exception:       # 뷰 미준비 → 공칭
                m = None
        if m is None or m.ndim != 2 or m.shape[0] != self.num_envs:
            return torch.full((self.num_envs,), _cfg.POUR_CUP_MASS, device=self.device)
        return m[:, 0].to(self.device, dtype=torch.float32)

    def _perceive(self, key: str, cup: RigidObject) -> tuple[torch.Tensor, torch.Tensor]:
        """컵 pose 지각: 지연(0..d_max 균등, env 별 매 스텝 재추첨) + 코히런트 노이즈. (pos env-local, quat)."""
        pos = self._local(cup.data.root_pos_w)
        quat = cup.data.root_quat_w
        cur = torch.cat([pos, quat], dim=1)
        buf = self._perc_buf[key]
        fl = self._perc_flush
        buf[fl] = cur[fl].unsqueeze(1)
        buf = torch.roll(buf, shifts=1, dims=1)
        buf[:, 0] = cur
        self._perc_buf[key] = buf
        d_max = int(round(self.adr.get_param("obs", "delay_steps")))
        idx = torch.randint(0, d_max + 1, (self.num_envs,), device=self.device)
        got = buf[self._perc_arange, idx]
        p, q = noisy_pose(got[:, :3], got[:, 3:], self.adr.get_param("obs", "object_xyz"),
                          self.adr.get_param("obs", "object_rot_deg"))
        return p, q

    def _apply_wrench(self) -> None:
        """들린 컵에 질량정규화 외란(매 스텝 재추첨, decay 0). ADR 진행도로 스케일 0 → 종점."""
        fs = self.adr.get_param("wrench", "force_scale")
        ts = self.adr.get_param("wrench", "torque_scale")
        lift_min = float(self.cfg.wrench_lift_min_m)
        fire = 0.0
        for key, cup, spawn in (("src", self.source_cup, self._src_spawn), ("rcv", self.receiver_cup, self._rcv_spawn)):
            w = self._wrench[key]
            w.force_scale, w.torque_scale = fs, ts
            lifted = (self._local(cup.data.root_pos_w)[:, 2] - spawn[:, 2]) > lift_min
            forces, torques = w.step(self._cup_mass[key], lifted)
            cup.set_external_force_and_torque(forces, torques, is_global=True)
            fire += float((forces.view(self.num_envs, -1).norm(dim=-1) > 0.0).float().mean())
        self.extras["dr/wrench_fire_frac"] = fire / 2.0

    # ==================================================================
    # 언더액추 계약 (grasp_fj_rh 이식 — 양손)
    # ==================================================================
    def _load_mimic_pairs(self) -> dict[str, tuple[str, float]]:
        """자산 **폴더**의 URDF `<mimic>` 표에서 종속→(리더, 배율)을 읽는다(양손). 코드에 표를 안 적는다."""
        _dir = os.path.dirname(os.path.join(_cfg._ASSETS_DIR, self.pair.usd_relpath))
        _cands = sorted(glob.glob(os.path.join(_dir, "*.urdf")))
        if len(_cands) != 1:
            raise RuntimeError(f"[pour_fabric_mimic] 자산 URDF 를 특정 못했다({_dir}): {_cands}")
        names = set(self.robot.data.joint_names)
        sides = tuple(rig.profile.hand_joint_names[0].split("_hj_")[0] for rig in self.rigs)
        pairs: dict[str, tuple[str, float]] = {}
        for j in ET.parse(_cands[0]).getroot().iter("joint"):
            m = j.find("mimic")
            dep = j.get("name") or ""
            if m is None or not any(dep.startswith(f"{s}_hj_") for s in sides) or dep not in names:
                continue
            pairs[dep] = (m.get("joint"), float(m.get("multiplier", 1.0)))
        if not pairs:
            raise RuntimeError(f"[pour_fabric_mimic] URDF 에 손 mimic 이 없다 ({_cands[0]}) — 이 트랙은 언더액추 손 전용")
        return pairs

    def _assert_mimic_constraints_present(self) -> None:
        """USD 에 PhysX mimic 제약이 gearing 값까지 맞게 붙어 있는지(09.02: headless 빌드가 12개를 잃었다)."""
        import isaacsim.core.utils.stage as stage_utils

        stage = stage_utils.get_current_stage()
        root = str(self.cfg.robot_cfg.prim_path).replace("env_.*", "env_0")
        found: dict[str, float] = {}
        for p in stage.Traverse():
            if not str(p.GetPath()).startswith(root):
                continue
            a = p.GetAttribute(_MIMIC_GEARING_ATTR)
            if a and a.Get() is not None:
                found[p.GetName()] = float(a.Get())
        bad = []
        for dep, (_lead, mult) in self._mimic_pairs.items():
            got = found.get(dep)
            if got is None:
                bad.append(f"{dep}: mimic 제약 없음")
            elif abs(got + mult) > 1e-3:
                bad.append(f"{dep}: gearing {got:.4f} ≠ −배율 {-mult:.4f}")
        if bad:
            raise RuntimeError("[pour_fabric_mimic] USD 언더액추 결합이 깨졌다 — " + " · ".join(bad))
        jn = self.robot.data.joint_names
        self._mim_dep_t = torch.tensor([jn.index(d) for d in self._mimic_pairs], device=self.device, dtype=torch.long)
        self._mim_lead_t = torch.tensor([jn.index(v[0]) for v in self._mimic_pairs.values()], device=self.device, dtype=torch.long)
        self._mim_mult = torch.tensor([v[1] for v in self._mimic_pairs.values()], device=self.device)
        print(f"[pour_fabric_mimic] 언더액추 확인 — 구동 {sum(len(r.syn_ids) for r in self.rigs)} · "
              f"종속 {len(self._mimic_pairs)}(PhysX mimic, gearing 대조 통과)", flush=True)

    def _assert_hand_pose_usable(self) -> None:
        """grip 이 한계 밖이거나 open==grip 이면 그 손가락은 지표에 흔적 없이 죽는다(09.02)."""
        for rig in self.rigs:
            p = rig.profile
            _c = rig.syn_grip.clamp(rig.syn_lo, rig.syn_hi)
            bad = [f"{n}: grip {float(rig.syn_grip[i]):.3f} → clamp {float(_c[i]):.3f}"
                   for i, n in enumerate(p.hand_joint_names) if abs(float(rig.syn_grip[i] - _c[i])) > 1e-6]
            if bad:
                raise RuntimeError(f"[pour_fabric_mimic:{rig.role}] grip 자세가 관절 한계 밖: " + " · ".join(bad))
            dead = [n for i, n in enumerate(p.hand_joint_names) if not bool(rig.syn_movable[i])]
            if dead:
                raise RuntimeError(f"[pour_fabric_mimic:{rig.role}] open == grip 인 손 관절 {dead}")

    def _widen_dependent_joint_limits(self) -> None:
        """종속관절 한계 확장 — mimic 제약과 한계가 동시에 만족 불가가 되는 것을 막는다(09.07 실측)."""
        m = float(self.cfg.mimic_dep_limit_margin_rad)
        if m <= 0.0:
            print("[pour_fabric_mimic] 종속관절 한계 확장 OFF (자산 값 그대로)", flush=True)
            return
        lim = self.robot.data.joint_pos_limits[:, self._mim_dep_t, :].clone()
        lim[..., 0] -= m
        lim[..., 1] += m
        self.robot.write_joint_limits_to_sim(lim, joint_ids=self._mim_dep_t.tolist())
        print(f"[pour_fabric_mimic] 종속 {int(self._mim_dep_t.numel())}관절 한계를 ±{m} rad 확장", flush=True)

    # ==================================================================
    def _build_fabric_world(self) -> dict | None:
        """fabric 장애물 = 테이블 박스 1개(두 프로필 palm 박스 합집합에서 파생)."""
        cfg = self.cfg
        if not bool(cfg.fabric_table_obstacle):
            print("[pour_fabric_mimic] ⚠fabric 테이블 장애물 OFF", flush=True)
            return None
        los = [p.palm_box_min for p in (self.pair.source, self.pair.receiver)]
        his = [p.palm_box_max for p in (self.pair.source, self.pair.receiver)]
        lo = [min(v[i] for v in los) for i in range(2)]
        hi = [max(v[i] for v in his) for i in range(2)]
        m = float(cfg.fabric_table_margin_xy)
        sx, sy = (hi[0] - lo[0]) + 2 * m, (hi[1] - lo[1]) + 2 * m
        cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        th = float(cfg.fabric_table_thickness)
        cz = float(cfg.table_surface_z) - 0.5 * th
        return {"table": {"env_index": "all", "type": "box", "scaling": f"{sx} {sy} {th}",
                          "transform": f"{cx} {cy} {cz} 0. 0. 0. 1."}}

    # ==================================================================
    def _setup_scene(self) -> None:
        cfg = self.cfg
        self.robot = Articulation(cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # 작업면: env_0 에 정적 프림 → clone 이 복제(grasp_s2r 규약). 마찰 재질 바인딩.
        cfg.table_spawn.func("/World/envs/env_0/Table", cfg.table_spawn, translation=(0.0, 0.0, 0.0))
        mu = float(cfg.surface_friction)
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu, restitution=0.0)
        mat.func("/World/Materials/taskSurface", mat)
        tables = find_matching_prim_paths(_cfg.TABLE_PRIM)
        if not tables:
            raise RuntimeError(f"[pour_fabric_mimic] 테이블 프림이 없다: {_cfg.TABLE_PRIM}")
        for tp in tables:
            bind_physics_material(tp, "/World/Materials/taskSurface")

        # 손가락 마디별 접촉 센서 — body 하나당 하나, **자기 컵만** 필터.
        self._sensor_store: dict = {}
        self._palm_store: dict = {}
        for role, prof, flt in (("src", self.pair.source, list(cfg.source_contact_filter)),
                                ("rcv", self.pair.receiver, list(cfg.receiver_contact_filter))):
            store: dict = {}
            for finger, bodies in prof.finger_sensor_bodies.items():
                ss = []
                for body in bodies:
                    s = ContactSensor(ContactSensorCfg(
                        prim_path=f"/World/envs/env_.*/Robot/{body}",
                        filter_prim_paths_expr=flt, history_length=1, track_air_time=False))
                    ss.append(s)
                    self.scene.sensors[f"contact_{role}_{finger}_{body}"] = s
                store[finger] = ss
            self._sensor_store[role] = store
            ps = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{prof.palm_body}",
                filter_prim_paths_expr=flt, history_length=1, track_air_time=False))
            self.scene.sensors[f"contact_{role}_palm"] = ps
            self._palm_store[role] = ps

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(),
                           translation=(0.0, 0.0, float(cfg.ground_plane_z)))
        light = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

        self.scene.clone_environments(copy_from_source=True)
        self.source_cup = RigidObject(cfg.source_cup_cfg)
        self.scene.rigid_objects["source_cup"] = self.source_cup
        self.receiver_cup = RigidObject(cfg.receiver_cup_cfg)
        self.scene.rigid_objects["receiver_cup"] = self.receiver_cup
        # 컵↔컵 접촉(09.14 s2r 충돌 신호): 소스 컵 센서를 리시버 컵으로 필터.
        self._cup_cup_sensor = ContactSensor(ContactSensorCfg(
            prim_path=_cfg.SOURCE_CUP_BODY, filter_prim_paths_expr=[_cfg.RECEIVER_CUP_BODY],
            history_length=1, track_air_time=False))
        # 부팅 검사: 강체 prim 경로가 env 수만큼 매칭돼야 한다(0 이면 센서·필터가 전부 조용히 0 이 된다).
        for expr in (_cfg.SOURCE_CUP_BODY, _cfg.RECEIVER_CUP_BODY):
            n_hit = len(find_matching_prim_paths(expr))
            if n_hit != self.num_envs:
                raise RuntimeError(f"[pour_fabric_mimic] 물체 강체 prim {expr} 매칭 {n_hit} ≠ env {self.num_envs} — "
                                   f"POUR_CUP_BODY_NAME 이 USD 와 다르다")
        self.scene.sensors["contact_cups"] = self._cup_cup_sensor
        self.beads = RigidObjectCollection(cfg.beads_cfg)
        self.scene.rigid_object_collections["beads"] = self.beads
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ==================================================================
    def _hold_mask(self) -> torch.Tensor:
        return self.episode_length_buf < int(self.cfg.hold_steps)

    def _local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return pos_w - self.scene.env_origins

    def _cup_up(self, cup: RigidObject) -> torch.Tensor:
        z = torch.zeros(self.num_envs, 3, device=self.device)
        z[:, 2] = 1.0
        return quat_apply(cup.data.root_quat_w, z)

    def _mouth(self, cup: RigidObject) -> torch.Tensor:
        off = torch.zeros(self.num_envs, 3, device=self.device)
        off[:, 2] = float(self.cfg.cup_mouth_z)
        return self._local(cup.data.root_pos_w + quat_apply(cup.data.root_quat_w, off))

    def _close_gate(self, rig: SideRig, cup: RigidObject, grasped: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if not bool(cfg.close_gate_enabled):
            return torch.ones(self.num_envs, device=self.device)
        d = (rig.palm_pos() - self._local(cup.data.root_pos_w)).norm(dim=-1)
        r = float(cfg.close_gate_radius)
        ramp = max(float(cfg.close_gate_ramp) * r, 1e-6)
        g = ((r - d) / ramp).clamp(0.0, 1.0)
        return torch.where(grasped, torch.ones_like(g), g)    # 파지 후 해제(들고 갈 때 잠기지 않게)

    # ==================================================================
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)
        active = ~self._hold_mask()
        h = self._half
        for i, (rig, cup, grasped) in enumerate(
                ((self.src, self.source_cup, self._src_grasped),
                 (self.rcv, self.receiver_cup, self._rcv_grasped))):
            a = self.actions[:, i * h:(i + 1) * h]
            rig.compose_palm_target(a[:, :6], active)
            prev = rig.syn_target
            gate = self._close_gate(rig, cup, grasped) * active.float()
            rig.syn_target = rig.synergy_targets(a[:, 6:], gate)
            rig.syn_vel = (rig.syn_target - prev) / self._policy_dt
            rig.sync_fabric_hand()
        # 양팔 26 DOF 를 두 인스턴스가 나눠 가진다 — 적분 전에 반대팔 슬라이스를 상대 상태로 맞춘다.
        self.src.sync_other(self.rcv)
        self.rcv.sync_other(self.src)
        for rig in self.rigs:
            q_pin = rig.fabric_q
            rig.step_fabric(self._world_ids, self._world_indicator)
            rig.pin_fabric(~active, q_pin)
        self._apply_wrench()

    def _apply_action(self) -> None:
        for rig in self.rigs:
            rig.apply_action()
        if self._grav_comp > 0.0:
            tau = self.robot.root_physx_view.get_gravity_compensation_forces()
            self.robot.set_joint_effort_target(self._grav_comp * tau[:, : self.robot.num_joints])

    # ==================================================================
    def _tactile(self, rig: SideRig, noisy: bool) -> torch.Tensor:
        """손끝 촉각 5칸 [N] — 실기 TouchData1.finger_forces 대응(사용자 결정 09.14). 실기 포화 10.24 N 에 맞춰 자른다."""
        f = rig.tip_tactile()
        if noisy:
            f = f + torch.randn_like(f) * float(self.cfg.tactile_obs_noise_n)
        return f.clamp(0.0, float(self.cfg.tactile_obs_clip_n))

    def _side_obs(self, rig: SideRig, cup_p: torch.Tensor, cup_q: torch.Tensor,
                  noisy: bool) -> list[torch.Tensor]:
        """한 팔의 actor 관측. cup_p/cup_q 는 **지각된**(perceived: 지연+노이즈) 컵 pose 다.

        ★09.14 s2r: hand_qd 없음(실기 드라이버 velocity ≠ 관절속도). noisy=True 면 관절·FK 에
          상시 노이즈. critic 은 noisy=False + 참 pose 로 같은 함수를 부른다.
        """
        cfg = self.cfg
        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        palm = rig.palm_pos()
        R = rig.palm_R()
        tips = rig.tips_pos()
        arm_q, arm_qd, hand_q = q[:, rig.arm_t], qd[:, rig.arm_t], q[:, rig.hand_t]
        if noisy:
            arm_q = arm_q + torch.randn_like(arm_q) * float(cfg.obs_noise_qpos)
            arm_qd = arm_qd + torch.randn_like(arm_qd) * float(cfg.obs_noise_qvel)
            hand_q = hand_q + torch.randn_like(hand_q) * float(cfg.obs_noise_qpos)
            palm = palm + torch.randn_like(palm) * float(cfg.obs_noise_body)
            tips = tips + torch.randn_like(tips) * float(cfg.obs_noise_body)
        z = torch.zeros(self.num_envs, 3, device=self.device)
        z[:, 2] = 1.0
        cup_up = quat_apply(cup_q, z)
        return [
            arm_q, arm_qd, hand_q,
            palm, torch.cat([R[:, :, 0], R[:, :, 1]], dim=1),
            (tips - palm.unsqueeze(1)).reshape(self.num_envs, -1),
            cup_p - palm,
            (tips - cup_p.unsqueeze(1)).reshape(self.num_envs, -1),
            rig.joint_err(),
            cup_up,
        ]

    def _mouth_from(self, cup_p: torch.Tensor, cup_q: torch.Tensor) -> torch.Tensor:
        off = torch.zeros(self.num_envs, 3, device=self.device)
        off[:, 2] = float(self.cfg.cup_mouth_z)
        return cup_p + quat_apply(cup_q, off)

    def _get_observations(self) -> dict:
        # ---- actor: 지각된 컵 pose(지연+노이즈) + 노이즈 관절/FK, hand_qd 없음 ----------------
        sp, sq = self._perceive("src", self.source_cup)
        rp, rq = self._perceive("rcv", self.receiver_cup)
        self._perc_flush[:] = False
        parts = self._side_obs(self.src, sp, sq, noisy=True) + self._side_obs(self.rcv, rp, rq, noisy=True)
        parts += [rp - sp, self._mouth_from(rp, rq) - self._mouth_from(sp, sq),
                  self._tactile(self.src, noisy=True), self._tactile(self.rcv, noisy=True), self.prev_actions]
        obs = torch.cat(parts, dim=1)

        # ---- critic: 참값(clean) + hand_qd + 비드 GT + 속도 + 접촉력 -------------------------
        tp_s, tq_s = self._local(self.source_cup.data.root_pos_w), self.source_cup.data.root_quat_w
        tp_r, tq_r = self._local(self.receiver_cup.data.root_pos_w), self.receiver_cup.data.root_quat_w
        clean = self._side_obs(self.src, tp_s, tq_s, noisy=False) + self._side_obs(self.rcv, tp_r, tq_r, noisy=False)
        clean += [tp_r - tp_s, self._mouth(self.receiver_cup) - self._mouth(self.source_cup),
                  self._tactile(self.src, noisy=False), self._tactile(self.rcv, noisy=False), self.prev_actions]
        qd = self.robot.data.joint_vel
        bead_fracs = torch.stack([self._prev_in_src, self._prev_in_tgt, self._prev_spill,
                                  self._crossed.float().mean(dim=-1)], dim=1)
        centroid_rel = self.beads.data.object_pos_w.mean(dim=1) - self.receiver_cup.data.root_pos_w
        centroid_local = quat_apply_inverse(self.receiver_cup.data.root_quat_w, centroid_rel)
        state = torch.cat(clean + [
            qd[:, self.src.hand_t], qd[:, self.rcv.hand_t],
            bead_fracs, centroid_local,
            self.source_cup.data.root_lin_vel_w, self.source_cup.data.root_ang_vel_w,
            self.receiver_cup.data.root_lin_vel_w, self.receiver_cup.data.root_ang_vel_w,
            (self.episode_length_buf.float() / float(self.max_episode_length)).unsqueeze(1),
            self.src.finger_forces().clamp(max=float(self.cfg.contact_obs_clip)),
            self.rcv.finger_forces().clamp(max=float(self.cfg.contact_obs_clip)),
        ], dim=1)
        return {"policy": torch.nan_to_num(obs), "critic": torch.nan_to_num(state)}

    # ==================================================================
    def _ctx_palm_axes(self, rig: SideRig) -> torch.Tensor:
        """보상 입력 palm_axes (N,6) = [손바닥 법선, 두 번째 축]. 계약상 앞 3칸이 법선이다(09.15, cfg 주석 참조)."""
        R = rig.palm_R()
        return torch.cat([R[:, :, int(self.cfg.ctx_palm_normal_col)], R[:, :, int(self.cfg.ctx_palm_second_col)]], dim=1)

    def _build_context(self, flags, d_in_target, d_spill) -> RewardContext:
        cfg = self.cfg
        src_f, rcv_f = self.src.finger_forces(), self.rcv.finger_forces()
        self._src_grasped = self.src.grasped(src_f)
        self._rcv_grasped = self.rcv.grasped(rcv_f)
        scup, rcup = self.source_cup, self.receiver_cup
        s_up, r_up = self._cup_up(scup), self._cup_up(rcup)

        def _tilt(up):
            return torch.acos(up[:, 2].clamp(-1.0, 1.0))
        return RewardContext(
            table_z=float(cfg.table_surface_z), cup_radius=float(cfg.cup_inner_radius),
            cup_mouth_z=float(cfg.cup_mouth_z), cup_bottom_z=-float(cfg.object_origin_offset_z),
            num_beads=int(cfg.bead_count),
            src_palm_pos=self.src.palm_pos(),
            src_palm_axes=self._ctx_palm_axes(self.src),
            src_tips_pos=self.src.tips_pos(), src_hand_closure=self.src.closure(),
            src_finger_force=src_f, src_palm_force=self.src.palm_force(),
            src_grasped=self._src_grasped, src_arm_qd=self.robot.data.joint_vel[:, self.src.arm_t],
            rcv_palm_pos=self.rcv.palm_pos(),
            rcv_palm_axes=self._ctx_palm_axes(self.rcv),
            rcv_tips_pos=self.rcv.tips_pos(), rcv_hand_closure=self.rcv.closure(),
            rcv_finger_force=rcv_f, rcv_palm_force=self.rcv.palm_force(),
            rcv_grasped=self._rcv_grasped, rcv_arm_qd=self.robot.data.joint_vel[:, self.rcv.arm_t],
            src_cup_pos=self._local(scup.data.root_pos_w), src_cup_quat=scup.data.root_quat_w,
            src_cup_up=s_up, src_cup_tilt=_tilt(s_up), src_cup_mouth_pos=self._mouth(scup),
            src_cup_lin_vel=scup.data.root_lin_vel_w, src_cup_ang_vel=scup.data.root_ang_vel_w,
            src_cup_spawn_pos=self._src_spawn,
            rcv_cup_pos=self._local(rcup.data.root_pos_w), rcv_cup_quat=rcup.data.root_quat_w,
            rcv_cup_up=r_up, rcv_cup_tilt=_tilt(r_up), rcv_cup_mouth_pos=self._mouth(rcup),
            rcv_cup_lin_vel=rcup.data.root_lin_vel_w, rcv_cup_ang_vel=rcup.data.root_ang_vel_w,
            rcv_cup_spawn_pos=self._rcv_spawn,
            bead_in_source_frac=flags.in_source_frac, bead_in_target_frac=flags.in_target_frac,
            bead_spill_frac=flags.spill_frac, bead_centroid=self._local(flags.centroid_w),
            d_in_target=d_in_target, d_spill=d_spill,
            cup_cup_force=self._cup_cup_sensor.data.force_matrix_w.view(self.num_envs, -1, 3).sum(dim=1).norm(dim=-1),
            src_hand_foreign_force=self.src.foreign_force(),
            rcv_hand_foreign_force=self.rcv.foreign_force(),
            cups_nested=self._cups_nested, success=self._success_now,
            episode_progress=self.episode_length_buf.float() / float(self.max_episode_length),
            actions=self.actions, prev_actions=self.prev_actions,
        )

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        flags = compute_bead_flags(
            bead_pos_w=self.beads.data.object_pos_w,
            source_pos_w=self.source_cup.data.root_pos_w, source_quat_w=self.source_cup.data.root_quat_w,
            target_pos_w=self.receiver_cup.data.root_pos_w, target_quat_w=self.receiver_cup.data.root_quat_w,
            geom_source=self._geom, geom_target=self._geom,
            prev_target_local_z=self._prev_tgt_z, crossed_mask=self._crossed)
        fresh = self._flags_fresh.float()
        d_in_target = (1.0 - fresh) * (flags.in_target_frac - self._prev_in_tgt)
        d_spill = (1.0 - fresh) * (flags.spill_frac - self._prev_spill)
        self._prev_in_tgt, self._prev_in_src, self._prev_spill = (
            flags.in_target_frac, flags.in_source_frac, flags.spill_frac)
        self._prev_tgt_z, self._crossed = flags.target_local_z, flags.crossed_mask
        self._flags_fresh[:] = False

        xy = (self.source_cup.data.root_pos_w[:, :2] - self.receiver_cup.data.root_pos_w[:, :2]).norm(dim=-1)
        # ★09.13 hacking 차단: 소스 컵을 리시버에 끼워 넣으면(원점 거리 < 9 cm) 성공이 아니다.
        center_d = (self.source_cup.data.root_pos_w - self.receiver_cup.data.root_pos_w).norm(dim=-1)
        self._cups_nested = center_d < float(cfg.cups_nested_dist)
        self._cups_center_dist = center_d
        self._success_now = ((flags.in_target_frac >= float(cfg.success_fill_ratio))
                             & (flags.spill_frac <= float(cfg.success_spill_max))
                             & (xy < float(cfg.success_xy_thresh))
                             & (~self._cups_nested))
        self._success_streak = torch.where(self._success_now, self._success_streak + 1,
                                           torch.zeros_like(self._success_streak))

        ctx = self._build_context(flags, d_in_target, d_spill)
        total, terms = call_reward_fn(self._reward_fn, ctx)
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        # hold 중엔 보상 0(팔이 고정된 구간의 보상은 정책과 무관하다).
        total = total * (~self._hold_mask()).float()
        self._last_terms = terms

        self.extras["action/step_delta"] = (self.actions - self.prev_actions).abs().mean()
        self.prev_actions.copy_(self.actions)
        # ADR: 순간 성공률 트리거(누적 평균은 관성으로 안 오른다 — adr.py 규약)
        if self.adr.maybe_increment(float(self._success_now.float().mean())):
            print(f"[pour_fabric_mimic][ADR] 증분 {self.adr.increment_counter}/{self.adr.num_increments} "
                  f"(progress {self.adr.progress:.2f})", flush=True)
        self._log(total, terms, flags, ctx)
        return total

    def _log_thumb_rim(self, side: str, palm, tips, cup_pos, cup_up) -> None:
        """엄지 입구 걸림 접근 계측 — 보상·관측과 무관, env 가 직접 잰다(09.15 사용자 "지표로깅으로 확인 가능하게").

        near = palm↔컵 원점 < `thumb_rim_near_m`. 걸림 = 엄지 끝(tips 0번)의 컵 축방향 높이 ≥ 입구 − `thumb_rim_band_m`
        이고 반경 < 컵 벽(내경 + 4 mm) + `thumb_rim_radial_margin_m`(= 입구 위). iter_03 `rim_hook` 과 같은 기하.
        """
        cfg = self.cfg
        rel = tips[:, 0, :] - cup_pos
        axial = (rel * cup_up).sum(dim=-1)
        radial = (rel - axial.unsqueeze(-1) * cup_up).norm(dim=-1)
        near = (palm - cup_pos).norm(dim=-1) < float(cfg.thumb_rim_near_m)
        hook = near & (axial >= float(cfg.cup_mouth_z) - float(cfg.thumb_rim_band_m)) \
            & (radial < float(cfg.cup_inner_radius) + 0.004 + float(cfg.thumb_rim_radial_margin_m))
        n_near = near.float().sum().clamp(min=1.0)
        self.extras[f"task/{side}_near_rate"] = near.float().mean()
        self.extras[f"task/{side}_thumb_over_rim_near"] = hook.float().sum() / n_near
        self.extras[f"task/{side}_thumb_above_rim_mm_near"] = \
            1000.0 * torch.where(near, axial - float(cfg.cup_mouth_z), torch.zeros_like(axial)).sum() / n_near

    def _log(self, total, terms, flags, ctx: RewardContext) -> None:
        cfg = self.cfg
        for k, v in terms.items():
            self.extras[f"reward/{k}"] = v.mean()
        self.extras["reward/total"] = total.mean()
        self.extras["task/success_now"] = self._success_now.float().mean()
        self.extras["task/episode_success"] = (
            self._success_streak >= int(cfg.success_hold_steps)).float().mean()
        self.extras["task/src_grasped"] = self._src_grasped.float().mean()
        self.extras["task/rcv_grasped"] = self._rcv_grasped.float().mean()
        self.extras["task/src_closure"] = ctx.src_hand_closure.mean()
        self.extras["task/rcv_closure"] = ctx.rcv_hand_closure.mean()
        self.extras["task/src_cup_lift"] = (ctx.src_cup_pos[:, 2] - self._src_spawn[:, 2]).mean()
        self.extras["task/rcv_cup_lift"] = (ctx.rcv_cup_pos[:, 2] - self._rcv_spawn[:, 2]).mean()
        self.extras["task/src_tilt_deg"] = torch.rad2deg(ctx.src_cup_tilt).mean()
        self.extras["task/rcv_tilt_deg"] = torch.rad2deg(ctx.rcv_cup_tilt).mean()
        self.extras["task/aim_dist"] = (ctx.src_cup_mouth_pos - ctx.rcv_cup_mouth_pos).norm(dim=-1).mean()
        self.extras["task/cups_center_dist"] = self._cups_center_dist.mean()
        self.extras["task/nested_rate"] = self._cups_nested.float().mean()
        thr = float(cfg.collision_force_threshold)
        self.extras["task/cup_collision_rate"] = (ctx.cup_cup_force > thr).float().mean()
        self.extras["task/src_hand_foreign_rate"] = (ctx.src_hand_foreign_force > thr).float().mean()
        self.extras["task/rcv_hand_foreign_rate"] = (ctx.rcv_hand_foreign_force > thr).float().mean()
        self.extras["contact/cup_cup_max"] = ctx.cup_cup_force.max()
        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        _err = (q[:, self._mim_dep_t] - self._mim_mult * q[:, self._mim_lead_t]).abs()
        # 09.07 기준선: 자유 폐쇄 ≤0.05 rad · 파지 ≤0.04 · 10 rad 넘는 epoch 반복이면 종속 한계 부족.
        self.extras["ctrl/mimic_err_max"] = _err.max()
        self.extras["ctrl/hand_dep_qd_max"] = qd[:, self._mim_dep_t].abs().max()
        self.extras.update(self.adr.log_dict())
        self.extras["dr/obs_object_xyz"] = self.adr.get_param("obs", "object_xyz")
        self.extras["dr/delay_steps"] = self.adr.get_param("obs", "delay_steps")
        self.extras["dr/wrench_force_scale"] = self.adr.get_param("wrench", "force_scale")
        self.extras["dr/src_mass_mean"] = self._cup_mass["src"].mean()
        self.extras["task/src_palm_to_cup"] = (ctx.src_palm_pos - ctx.src_cup_pos).norm(dim=-1).mean()
        self.extras["task/rcv_palm_to_cup"] = (ctx.rcv_palm_pos - ctx.rcv_cup_pos).norm(dim=-1).mean()
        self.extras["contact/src_max"] = ctx.src_finger_force.max(dim=1).values.mean()
        self.extras["contact/rcv_max"] = ctx.rcv_finger_force.max(dim=1).values.mean()
        self._log_thumb_rim("src", ctx.src_palm_pos, ctx.src_tips_pos, ctx.src_cup_pos, ctx.src_cup_up)
        self._log_thumb_rim("rcv", ctx.rcv_palm_pos, ctx.rcv_tips_pos, ctx.rcv_cup_pos, ctx.rcv_cup_up)
        self.extras["bead/in_source"] = flags.in_source_frac.mean()
        self.extras["bead/in_target"] = flags.in_target_frac.mean()
        self.extras["bead/spill"] = flags.spill_frac.mean()
        self.extras["bead/crossed"] = flags.crossed_frac.mean()
        self.extras["done/drop"] = self._dropped.float().mean()
        for rig, tag in ((self.src, "src"), (self.rcv, "rcv")):
            perr = (rig.palm_targets[:, :3] + rig.fab_to_env - rig.palm_pos()).norm(dim=-1)
            self.extras[f"fabric/{tag}_palm_err"] = perr.mean()
            # 회전 추종(각 슬롯 wrap 후 최대 절대 오차) — 붓기 tilt 지령이 실제로 도달하는지의 근거
            rerr = rig.palm_targets[:, 3:] - rig.palm_pose_6d()[:, 3:]
            rerr = torch.remainder(rerr + math.pi, 2 * math.pi) - math.pi
            self.extras[f"fabric/{tag}_rot_err_deg"] = torch.rad2deg(rerr.abs().max(dim=1).values).mean()
        self._log_tick += 1
        every = int(cfg.console_log_interval)
        if every > 0 and self._log_tick % every == 0:
            print(f"[METRICS] step={self._log_tick:>8d} rew={total.mean():+.3f} "
                  f"gS={self._src_grasped.float().mean():.2f} gR={self._rcv_grasped.float().mean():.2f} "
                  f"liftS={float(self.extras['task/src_cup_lift']):+.3f} "
                  f"tiltS={float(self.extras['task/src_tilt_deg']):.1f} "
                  f"aim={float(self.extras['task/aim_dist']):.3f} "
                  f"inT={flags.in_target_frac.mean():.3f} inS={flags.in_source_frac.mean():.3f} "
                  f"spill={flags.spill_frac.mean():.3f} succ={self._success_now.float().mean():.3f}",
                  flush=True)

    # ==================================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        floor = float(cfg.table_surface_z) + float(cfg.object_origin_offset_z) - float(cfg.drop_below_table_m)
        s_z = self._local(self.source_cup.data.root_pos_w)[:, 2]
        r_z = self._local(self.receiver_cup.data.root_pos_w)[:, 2]
        self._dropped = ((s_z < floor) | (r_z < floor)) & (~self._hold_mask())
        qd = torch.cat([self.robot.data.joint_vel[:, self.src.arm_t],
                        self.robot.data.joint_vel[:, self.rcv.arm_t]], dim=1)
        runaway = (qd.abs() > float(cfg.runaway_joint_vel)).any(dim=-1)
        # 언더액추 폭주: 종속관절 속도(mimic 제약이 깨진 서명) — 09.14 사용자 결정, 깨진 env 는 즉시 리셋.
        dep_qd = self.robot.data.joint_vel[:, self._mim_dep_t].abs().max(dim=-1).values
        # 09.14 라운드 1: 속도 기준을 빠져나간 느린 폭주 → 결합 오차 자체도 종료(사용자 결정 "2 추가").
        q = self.robot.data.joint_pos
        mim_err = (q[:, self._mim_dep_t] - self._mim_mult * q[:, self._mim_lead_t]).abs().max(dim=-1).values
        mimic_runaway = (dep_qd > float(cfg.mimic_runaway_dep_qd)) | (mim_err > float(cfg.mimic_runaway_err_rad))
        terminated = runaway | mimic_runaway | self._dropped
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["task/runaway_rate"] = runaway.float().mean()
        self.extras["done/mimic_runaway"] = mimic_runaway.float().mean()
        self.extras["done/mimic_err_runaway"] = (mim_err > float(cfg.mimic_runaway_err_rad)).float().mean()
        return terminated, truncated

    # ==================================================================
    def _reset_idx(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n, dev, cfg = len(env_ids), self.device, self.cfg

        q0 = self._reset_q.unsqueeze(0).expand(n, -1).contiguous()
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), env_ids=env_ids)
        self.robot.set_joint_position_target(q0, env_ids=env_ids)
        for rig in self.rigs:
            rig.reset(env_ids)

        rest_z = float(cfg.table_surface_z) + float(cfg.object_origin_offset_z)
        rng = float(cfg.object_spawn_range)
        for rig, cup, store in ((self.src, self.source_cup, self._src_spawn),
                                (self.rcv, self.receiver_cup, self._rcv_spawn)):
            offs = (torch.rand(n, 2, device=dev) - 0.5) * 2.0 * rng
            pos = torch.zeros(n, 3, device=dev)
            pos[:, 0] = rig.profile.object_spawn_center[0] + offs[:, 0]
            pos[:, 1] = rig.profile.object_spawn_center[1] + offs[:, 1]
            pos[:, 2] = rest_z
            store[env_ids] = pos
            root = torch.zeros(n, 13, device=dev)
            root[:, :3] = pos + self.scene.env_origins[env_ids]
            root[:, 2] += float(cfg.object_spawn_pad)
            root[:, 3] = 1.0
            cup.write_root_state_to_sim(root, env_ids=env_ids)

        k = int(cfg.bead_count)
        bead = torch.zeros(n, k, 13, device=dev)
        bead[:, :, :3] = (self._src_spawn[env_ids].unsqueeze(1) + self._bead_offs.unsqueeze(0)
                          + self.scene.env_origins[env_ids].unsqueeze(1))
        bead[:, :, 2] += float(cfg.object_spawn_pad)
        bead[:, :, 3] = 1.0
        self.beads.write_object_state_to_sim(bead, env_ids=env_ids)

        self._perc_flush[env_ids] = True
        for w in self._wrench.values():
            w.reset(env_ids)
        self._cup_mass["src"] = self._read_cup_mass(self.source_cup)
        self._cup_mass["rcv"] = self._read_cup_mass(self.receiver_cup)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._success_now[env_ids] = False
        self._success_streak[env_ids] = 0
        self._dropped[env_ids] = False
        self._src_grasped[env_ids] = False
        self._rcv_grasped[env_ids] = False
        self._crossed[env_ids] = False
        self._prev_tgt_z[env_ids] = -1e6
        self._prev_in_tgt[env_ids] = 0.0
        self._prev_in_src[env_ids] = 1.0
        self._prev_spill[env_ids] = 0.0
        self._flags_fresh[env_ids] = True
