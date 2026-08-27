"""grasp_s2r 제어 스택 — Fabrics 팔 + 관절공간 시너지 손.

`agnostic/tasks/grasp_sensor` 에서 검증된 배선을 그대로 이식했다. 그 트랙은 손 제어
4모드(pd/fabric/tip_cyl/synergy) 분기를 갖고 있었는데, 여기서는 **synergy 하나만**
남긴다(나머지는 전부 기각된 경로다 — 죽은 분기는 나중에 고칠 때 오해만 만든다).

env 본체(`grasp_s2r_env.py`)가 이 믹스인을 상속한다. 여기 있는 것은 전부 "어떻게
움직이는가"이고, "무엇을 보상하는가"는 env 와 `grasp_s2r_rewards.py` 에 있다.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import bind_physics_material
from isaaclab.utils.math import (euler_xyz_from_quat, matrix_from_quat,
                                 quat_from_euler_xyz, quat_mul)

import fabrics_sim.fabrics.openarm_tesollo_pose_fabric as _fab_tesollo
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

from .robot_profiles import PROFILES

_FABRIC_MODULES = (_fab_tesollo,)


def _fabric_class(name: str):
    """프로필의 문자열 이름 → fabric 클래스. env 에 로봇명을 하드코딩하지 않는 계약."""
    for mod in _FABRIC_MODULES:
        if hasattr(mod, name):
            return getattr(mod, name)
    raise RuntimeError(
        f"fabric 클래스 '{name}' 를 찾을 수 없다: {[m.__name__ for m in _FABRIC_MODULES]}")


class GraspS2RControlMixin:
    """씬 구성 · Fabrics · 시너지 손 · 접촉 센서 · 지령 마커."""

    # ------------------------------------------------------------------
    # 씬
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # 정적 환경 USD(env.usd: RigidBodyAPI 없음, 전 메시 충돌체) —
        # env_0 에 spawn 하면 clone_environments 가 복제한다.
        tbl = self.cfg.table_cfg
        tbl.spawn.func(
            "/World/envs/env_0/Table", tbl.spawn,
            translation=tuple(tbl.init_state.pos), orientation=tuple(tbl.init_state.rot),
        )
        # ★테이블은 scene 자산이 아니라 정적 프림이라 EventTerm 이 못 건다. 직접 바인딩한다.
        #   PhysX 결합이 average 라 한쪽만 낮아도 실효 μ 가 중간값이 되고, 컵-테이블
        #   마찰은 접근·안정에 직접 영향을 준다.
        _mu = float(self.cfg.surface_friction)
        _mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=_mu, dynamic_friction=_mu, restitution=0.0)
        _mat.func("/World/Materials/taskSurface", _mat)
        bind_physics_material("/World/envs/env_0/Table", "/World/Materials/taskSurface")

        # ★★손가락별 접촉 센서 — body **하나당 센서 하나**. 다중 body 를 한 센서에
        #   묶으면 `force_matrix_w` 가 무증상 0 을 반환한다(실측 함정).
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

        # env.usd 의 platform 상면이 정확히 z=0 이라 기본 지면과 겹친다 — 지면은 내린다.
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(),
                           translation=(0.0, 0.0, -0.05))
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object
        self.scene.clone_environments(copy_from_source=True)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ------------------------------------------------------------------
    # Fabrics — 팔은 절대 palm pose attractor 로만 움직인다
    # ------------------------------------------------------------------
    def _build_fabric_index(self) -> torch.Tensor:
        """프로필 `fabric_joint_order` → articulation 인덱스.

        ★★articulation 은 관절번호-major(index_1, middle_1, …), fabric URDF 는
          finger-major(thumb_1..4, index_1..4, …)다. 슬라이스로 대응시키면 손 20관절이
          통째로 어긋나 **조용히** 엉뚱한 자세로 움직인다. 순서가 유일한 방어선이다.
        """
        order = self.profile.fabric_joint_order
        if len(order) != self.fabric.num_joints:
            raise RuntimeError(
                f"[{self.profile.name}] fabric_joint_order 길이 {len(order)} != "
                f"fabric num_joints {self.fabric.num_joints}")
        idx = []
        for name in order:
            ids, _ = self.robot.find_joints(name)
            if len(ids) != 1:
                raise RuntimeError(
                    f"[{self.profile.name}] fabric 관절 '{name}' 해석 실패: {ids}")
            idx.append(ids[0])
        return torch.tensor(idx, device=self.device, dtype=torch.long)

    def _setup_fabrics(self) -> None:
        p = self.profile
        # ★`_syn_ids` 가 아래 `_syn_to_fab_idx` 보다 먼저 있어야 한다(순서 계약).
        self._setup_synergy()
        if not p.fabric_class or not p.fabric_robot_dir:
            raise RuntimeError(
                f"[{p.name}] fabric_class/fabric_robot_dir 가 없다. 이 태스크는 Fabrics "
                "로만 돈다 — 자산을 만들거나 다른 프로필을 쓰라.")
        initialize_warp(str(self.device)[-1])          # 멀티 GPU 캐시 분리
        self._world = WorldMeshesModel(
            batch_size=self.num_envs, device=self.device,
            max_objects_per_env=int(self.cfg.fabrics_max_objects_per_env),
        )
        self._world_ids, self._world_indicator = self._world.get_object_ids()

        self.fabric = _fabric_class(p.fabric_class)(
            batch_size=self.num_envs, device=self.device,
            timestep=float(self.cfg.fabrics_dt),
            graph_capturable=bool(self.cfg.fabric_use_cuda_graph),
            # 손은 fabric 밖(관절공간 시너지 + PD)이다. fabric 은 팔 계획 전용이고
            # 손 자세는 **상태 동기화**로만 받아 충돌 모델을 맞춘다.
            use_hand_fabric=False,
            tip_per_finger=False,
            hand_mode="pca",
            use_hand_repulsion=bool(self.cfg.use_hand_repulsion),
            use_body_repulsion_pairs=bool(self.cfg.use_body_repulsion_pairs),
            robot_dir_name=p.fabric_robot_dir,
            robot_name=p.fabric_robot_dir,
            **({"fabric_params_filename": p.fabric_params_filename}
               if p.fabric_params_filename else {}),
        )
        self.integrator = DisplacementIntegrator(self.fabric)

        expect = p.num_arm_joints + p.num_hand_joints
        if self.fabric.num_joints != expect:
            raise RuntimeError(
                f"[{p.name}] fabric num_joints={self.fabric.num_joints} != 프로필 {expect}. "
                "fabric URDF 와 USD 자산이 어긋났다.")
        self._fab_t = self._build_fabric_index()

        # synergy 자세(프로필 finger-major) → fabric 손 구간 순서. 이름 기반 매핑 유지.
        _syn_pos = {int(j): k for k, j in enumerate(self._syn_ids)}
        _fab_hand = self._fab_t[p.num_arm_joints:].tolist()
        _missing = [int(j) for j in _fab_hand if int(j) not in _syn_pos]
        if _missing:
            raise RuntimeError(
                f"[{p.name}] synergy 자세에 없는 fabric 손 관절 {_missing} — "
                "hand_joint_names 가 손 관절을 모두 덮어야 한다")
        self._syn_to_fab_idx = torch.tensor(
            [_syn_pos[int(j)] for j in _fab_hand], device=self.device, dtype=torch.long)

        self.fabric_q = self.robot.data.default_joint_pos[:, self._fab_t].contiguous()
        self.fabric_qd = torch.zeros(self.num_envs, self.fabric.num_joints, device=self.device)
        self.fabric_qdd = torch.zeros_like(self.fabric_qd)
        # use_hand_fabric=False 라 무시되지만 원본 계약(B,5 PCA)은 지킨다.
        self._fabric_hand_cmd = torch.zeros(self.num_envs, 5, device=self.device)
        # cspace attractor(널스페이스) rest 자세를 프로필 홈으로.
        self.fabric.default_config.copy_(self.fabric_q)
        self._fabric_damping = float(self.cfg.fabrics_damping_gain) * torch.ones(
            self.num_envs, 1, device=self.device)

        # palm 목표 박스(env-local 절대) + 회전 박스 — 전부 프로필에서 온다.
        d = math.pi / 180.0
        c = torch.tensor(p.palm_rot_center_deg, device=self.device) * d
        h = float(p.palm_rot_half_deg) * d
        self._palm_lo = torch.cat([torch.tensor(p.palm_box_min, device=self.device), c - h])
        self._palm_hi = torch.cat([torch.tensor(p.palm_box_max, device=self.device), c + h])
        self.palm_targets = torch.zeros(self.num_envs, 6, device=self.device)
        self._home_palm = torch.zeros(6, device=self.device)   # _init_home_palm 에서 실측
        if not p.palm_box_verified:
            print(f"[grasp_s2r] ⚠ palm_box 미검증({p.name}) — 도달성 확인 후 승격할 것",
                  flush=True)

    def _init_home_palm(self) -> None:
        """홈 palm pose 실측 + fabric FK 정합 검사(부팅 게이트 3종).

        ★`__init__` 시점의 `body_pos_w` 는 stale 이다(로봇이 아직 홈에 안 놓임).
          관절을 써넣고 물리를 2스텝 돌린 뒤 읽는다.
        """
        q0 = self.robot.data.default_joint_pos
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        self.robot.set_joint_position_target(q0)
        self.scene.write_data_to_sim()
        for _ in range(2):
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)

        home = self._palm_pose_6d()[0]
        self._home_palm = home.clone()
        self.palm_targets[:] = home.unsqueeze(0)

        # ★fabric FK 프레임과 sim env-local 은 **원점이 다르다**(실측 544mm). 같은
        #   물리점(손끝)을 양쪽에서 읽어 상수 오프셋을 실측한다. 회전까지 다르면
        #   평행이동으로 못 잇으므로 산포를 보고 fail-loud.
        q0f = self.robot.data.default_joint_pos[:, self._fab_t].contiguous()
        _nt = len(self.tip_ids)
        tips_fab = self.fabric._fingertip_taskmap(q0f, None)[0].reshape(
            self.num_envs, _nt, 3)[0]
        tips_sim = (self.robot.data.body_pos_w[:, self._tip_ids_t]
                    - self.scene.env_origins[:, None, :])[0]
        delta = tips_sim - tips_fab
        spread = float(delta.std(dim=0).max())
        if spread > 2e-3:
            raise RuntimeError(
                f"[{self.profile.name}] fabric↔env 프레임이 순수 평행이동이 아니다 "
                f"(손끝 오프셋 산포 {spread * 1000:.1f}mm > 2mm) — 회전 정합 필요")
        self._fab_to_env = delta.mean(dim=0)
        print(f"[grasp_s2r] fabric→env 오프셋 = "
              f"{[round(float(v) * 1000) for v in self._fab_to_env]}mm "
              f"(산포 {spread * 1000:.2f}mm)", flush=True)

        out = (home < self._palm_lo) | (home > self._palm_hi)
        if bool(out.any()):
            raise RuntimeError(
                f"[{self.profile.name}] 홈 palm 이 워크스페이스 박스 밖이다: "
                f"home={[round(v, 3) for v in home.tolist()]}")

        # ★이 한 줄이 (fabric URDF 오선택 / joint_order 오류 / palm_body 오지정)
        #   3대 배선 사고를 부팅에서 전부 잡는다.
        fab = self.fabric.get_palm_pose(self.fabric_q.detach(), "euler_zyx")[0]
        dp = float(torch.norm(fab[:3] - home[:3]))
        dr = float(torch.max(torch.abs(fab[3:] - home[3:])))
        print(f"[grasp_s2r] 홈 palm={[round(v, 4) for v in home.tolist()]} | "
              f"fabric FK 정합 pos {dp * 1000:.2f}mm rot {math.degrees(dr):.2f}°", flush=True)
        if dp > 0.005 or dr > math.radians(2.0):
            raise RuntimeError(
                f"[{self.profile.name}] fabric FK 가 USD palm 과 어긋난다: "
                f"{dp * 1000:.1f}mm / {math.degrees(dr):.1f}° (허용 5mm/2°). "
                "fabric_robot_dir·fabric_joint_order·palm_body 를 확인하라.")

    def _step_fabric(self) -> None:
        """목표 주입 + 적분 — **정책 스텝당 한 번**.

        ★`_apply_action` 은 decimation 만큼 불리므로 거기서 적분하면 fabric 시간이
          2배로 흐른다.
        """
        self.fabric.set_features(
            self._fabric_hand_cmd, self.palm_targets, "euler_zyx",
            self.fabric_q.detach(), self.fabric_qd.detach(),
            self._world_ids, self._world_indicator, self._fabric_damping,
        )
        for _ in range(int(self.cfg.fabric_decimation)):
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.integrator.step(
                self.fabric_q.detach(), self.fabric_qd.detach(),
                self.fabric_qdd.detach(), float(self.cfg.fabrics_dt),
            )

    def _apply_action(self) -> None:
        """decimation 마다 불린다 — **적분은 여기서 하지 않는다**."""
        # fabric_q 는 **오픈루프 plant** — 실측 관절로 되돌려 동기화하면 팔이 명령을
        # 못 따라간다(선행 트랙 사고 2건).
        arm_target = self.fabric_q[:, : self.profile.num_arm_joints]
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_ids)
        # ★속도 피드포워드. 0 을 넣으면 implicit PD 의 감쇠항 kd·(0 − q̇) 이 참조
        #   궤적의 움직임을 반대로 밀어 err ≈ (kd/kp)·q̇ 의 상시 지연이 생긴다.
        self.robot.set_joint_velocity_target(
            float(self.cfg.fabric_velocity_ff_scale)
            * self.fabric_qd[:, : self.profile.num_arm_joints],
            joint_ids=self.arm_ids)
        # 손은 fabric 밖 — 이름으로 찾은 인덱스에 관절 목표를 직접 준다.
        self.robot.set_joint_position_target(self._syn_target, joint_ids=self._syn_ids)
        self.robot.set_joint_velocity_target(
            float(self.cfg.hand_velocity_ff_scale) * self._syn_vel,
            joint_ids=self._syn_ids)

    # ------------------------------------------------------------------
    # 손 — 관절공간 시너지
    # ------------------------------------------------------------------
    def _setup_synergy(self) -> None:
        """시너지 그립 배선 — 관절 목표를 직접 보간해 파워그립을 구조적으로 보장한다.

        ★★관절 순서 함정: 프로필 자세 배열은 finger-major 인데 articulation 은
          관절번호-major 다. 여기서 **이름으로 한 번만** 매핑하고 이후 전부 이 인덱스를 쓴다.
        """
        p = self.profile
        for _f in ("hand_joint_names", "hand_open_pose", "hand_grip_pose"):
            if not getattr(p, _f):
                raise RuntimeError(f"[{p.name}] 시너지 손 제어에 필요한 프로필 필드 {_f} 가 없다")
        n = len(p.hand_joint_names)
        if len(p.hand_open_pose) != n or len(p.hand_grip_pose) != n:
            raise RuntimeError(
                f"[{p.name}] 자세 배열 길이 불일치: names {n} / open "
                f"{len(p.hand_open_pose)} / grip {len(p.hand_grip_pose)}")
        jn = self.robot.data.joint_names
        self._syn_ids = [jn.index(nm) for nm in p.hand_joint_names]
        self._syn_open = torch.tensor(p.hand_open_pose, device=self.device)
        self._syn_grip = torch.tensor(p.hand_grip_pose, device=self.device)

        fingers = list(p.finger_sensor_bodies.keys())
        ch, fi = [], []
        for nm in p.hand_joint_names:
            _sfx = nm.rsplit("_", 1)[1]
            if _sfx not in p.hand_channel_of_joint:
                raise RuntimeError(f"[{p.name}] hand_channel_of_joint 에 접미사 {_sfx} 없음")
            ch.append(int(p.hand_channel_of_joint[_sfx]))
            _hit = [i for i, f in enumerate(fingers) if f"_{f}_" in nm]
            if len(_hit) != 1:
                raise RuntimeError(f"[{p.name}] 관절 {nm} 의 손가락을 특정 못함: {_hit}")
            fi.append(_hit[0])
        self._syn_ch = torch.tensor(ch, device=self.device, dtype=torch.long)
        self._syn_fi = torch.tensor(fi, device=self.device, dtype=torch.long)
        self._syn_nch = len(set(ch))
        self._syn_freeze = torch.tensor(
            [nm.rsplit("_", 1)[1] in p.hand_freeze_suffixes for nm in p.hand_joint_names],
            device=self.device)
        # ★★동결은 **관절별로 자기 링크가 닿았을 때** 걸어야 한다.
        #   구판은 (원위|팁) 접촉 하나로 `_3`·`_4` 를 통째로 얼렸는데, `_2` 가 굽으면
        #   손끝이 가장 먼저 닿으므로 **감쌈이 시작되기 직전에 감쌈 관절을 잠갔다** —
        #   08.27 실측: wrap_frac 이 전 런에서 정확히 0.000, syn_close 0.278 ≈
        #   "채널1(`_2`)만 폐쇄" 예측 0.250.
        #   `_3` → 중간마디 접촉 / `_4` → 원위 또는 팁(팁은 원위 링크에 고정).
        _sfx = [nm.rsplit("_", 1)[1] for nm in p.hand_joint_names]
        self._syn_freeze_mid = torch.tensor(
            [s in p.hand_freeze_suffixes and s == "3" for s in _sfx], device=self.device)
        self._syn_freeze_dist = torch.tensor(
            [s in p.hand_freeze_suffixes and s != "3" for s in _sfx], device=self.device)
        # ★가동 관절 마스크 — open == grip 인 관절은 명령해도 안 움직인다
        #   (실측: r_hj_pinky_2 · r_hj_thumb_2 · 전 `_1` 이 가동폭 0°). 폐쇄 보상의
        #   분모에 넣으면 "못 움직이는 관절을 닫았다"는 공짜 점수가 생긴다.
        self._syn_movable = (self._syn_grip - self._syn_open).abs() > 1e-4
        if not bool(self._syn_movable.any()):
            raise RuntimeError(f"[{p.name}] 가동 손관절이 하나도 없다 — open/grip 자세 확인")
        # 폐쇄도는 **관절별** 독립 진행도다 — 접촉 동결이 관절마다 따로 걸린다.
        self._syn_close = torch.zeros(self.num_envs, n, device=self.device)
        self._syn_target = self.robot.data.joint_pos[:, self._syn_ids].clone()
        self._syn_vel = torch.zeros(self.num_envs, n, device=self.device)
        _lim = self.robot.data.soft_joint_pos_limits[0, self._syn_ids, :]
        self._syn_lo, self._syn_hi = _lim[:, 0].contiguous(), _lim[:, 1].contiguous()
        _grip_clamped = self._syn_grip.clamp(self._syn_lo, self._syn_hi)
        print(f"[grasp_s2r] synergy: 관절 {n}개 · 채널 {self._syn_nch} · "
              f"동결 {int(self._syn_freeze.sum())}개 · "
              f"grip 한계clamp {int((self._syn_grip != _grip_clamped).sum())}개", flush=True)

    def _synergy_targets(self, a_hand: torch.Tensor) -> torch.Tensor:
        """액션(손가락×채널) → 관절 목표. 프로필 순서 (N, n).

        액션은 **절대 폐쇄도 목표**이고 `synergy_close_speed` 는 그 목표를 향한
        변화율 상한이다(속도 명령이 아니다 — 속도로 두면 탐색 노이즈 평균만으로
        완전 폐쇄되고 되돌릴 수 없다).
        """
        p = self.profile
        nf = len(p.finger_sensor_bodies)
        a = a_hand.view(self.num_envs, nf, self._syn_nch)
        if bool(self.cfg.couple_four_fingers):
            # 대향 그룹(엄지)만 독립, 나머지는 채널별 평균 — "특정 손가락만 안 닫힘"을
            # 액션 공간에서 제거한다. 접촉 동결은 관절별로 남아 형상 적응은 유지된다.
            _mask = torch.ones(nf, dtype=torch.bool, device=a.device)
            _mask[self._group_a_idx] = False
            _common = a[:, _mask, :].mean(dim=1, keepdim=True)
            a = torch.where(_mask.view(1, nf, 1), _common.expand(-1, nf, -1), a)
        cmd = 0.5 * (a.clamp(-1.0, 1.0) + 1.0)                    # 절대 폐쇄도 [0,1]
        cmd_j = cmd[:, self._syn_fi, self._syn_ch]                # (N, n) 관절 전개
        rate = float(self.cfg.synergy_close_speed)
        delta = (cmd_j - self._syn_close).clamp(-rate, rate)
        # ★닫는 방향만 정렬 게이트로 스케일한다 — **푸는 방향은 항상 허용**해야
        #   잘못 오므린 상태에서 빠져나올 수 있다.
        _g = self._close_gate.unsqueeze(1)
        delta = torch.where(delta > 0.0, delta * _g, delta)
        if bool(self.cfg.synergy_contact_freeze):
            # ★★감쌈을 만드는 메커니즘: 닿은 마디의 관절만 멈춰 컵 형상에 드리워지게
            #   한다. 끄면 핀치가 된다. 단 **관절마다 자기 링크**를 봐야 한다 —
            #   팁 하나로 `_3`·`_4` 를 같이 얼리면 감쌈 직전에 감쌈을 잠근다.
            _mid, _dist = self._contact_forces_split()
            _thr = float(self.cfg.contact_force_threshold)
            # ★★팁은 동결 트리거가 **아니다**. 팁은 원위와 별개 body·별개 센서이고,
            #   손가락이 말릴 때 팁이 원위 링크보다 먼저 닿는다. 팁으로 `_4` 를 얼리면
            #   원위 링크가 컵에 닿을 기회 자체가 사라져 wrap(중간 AND 원위)이 영원히 0 이다.
            #   08.27 실측(s2r_a8, 817 iter): touch_frac 0.10~0.31 · grip_frac 0.20~0.50
            #   인데 wrap_frac 0.000. ★대향 손가락인 **엄지가 가장 먼저** 닿아 제일 먼저
            #   얼었다 — 사용자 관찰 "4지는 말리는데 엄지 _3/_4 는 홈자세 그대로".
            _h_mid = (_mid > _thr)[:, self._syn_fi]
            _h_dist = (_dist > _thr)[:, self._syn_fi]
            _hold = (_h_mid & self._syn_freeze_mid) | (_h_dist & self._syn_freeze_dist)
            # ★닫는 방향만 얼린다 — 푸는 방향까지 막으면 갇혀서 빠져나올 수 없다
            #   (닫기 게이트와 같은 원칙).
            delta = torch.where(_hold & (delta > 0.0), torch.zeros_like(delta), delta)
        self._syn_close = (self._syn_close + delta).clamp(0.0, 1.0)
        tgt = torch.lerp(self._syn_open.unsqueeze(0), self._syn_grip.unsqueeze(0),
                         self._syn_close)
        return tgt.clamp(self._syn_lo.unsqueeze(0), self._syn_hi.unsqueeze(0))

    def _close_progress(self) -> torch.Tensor:
        """가동 손관절 평균 폐쇄도 (N,) [0,1] — **실측 관절** 기준.

        ★★지령(`_syn_close`)이 아니라 실측이다. 지령을 재면 손이 테이블에 눌려 쫙 펴져도
          "닫으라고 명령했으니" 만점이 나온다 — 08.27 실측(s2r_b1 569 iter):
          hand_joint_err_max 가 최대 3.72 rad(포화 임계 0.30 의 12배)로 손이 물리적으로
          강제 이탈했는데 grasp 는 4.69/step 를 계속 지급했다(사용자 GUI: "손바닥이
          테이블에 쓸리면서 열린다").
        ★실측은 물체에 막히면 스스로 멈춘다 — 그래서 인위적 포화 캡이 필요 없다.
          "닫다가 컵에 막힘"이 곧 접촉이고, 그게 다음 단계다.
        ★가동폭 0° 관절은 제외한다. 안 그러면 못 움직이는 5개(전 `_1` + pinky_2 +
          thumb_2)가 분모에 섞여 공짜 점수를 만든다.
        """
        _q = self.robot.data.joint_pos[:, self._syn_ids]
        _span = (self._syn_grip - self._syn_open).unsqueeze(0)
        _prog = ((_q - self._syn_open.unsqueeze(0)) / _span).clamp(0.0, 1.0)
        return _prog[:, self._syn_movable].mean(dim=1)

    def _syn_to_fab(self, syn_q: torch.Tensor) -> torch.Tensor:
        """synergy 자세(프로필 순서) → fabric 손 구간 순서."""
        return syn_q[:, self._syn_to_fab_idx]

    # ------------------------------------------------------------------
    # 접촉 · 좌표 헬퍼
    # ------------------------------------------------------------------
    def _contact_forces(self) -> torch.Tensor:
        """손가락별 물체 접촉력 크기 (N, F). body 별 센서 합산, Object-필터."""
        mags = []
        for finger in self._finger_names:
            total = torch.zeros(self.num_envs, device=self.device)
            for s in self._finger_sensors[finger]:
                fm = s.data.force_matrix_w                       # (N, B, M, 3)
                total = total + fm.view(self.num_envs, -1, 3).sum(dim=1).norm(dim=-1)
            mags.append(total)
        return torch.stack(mags, dim=1)

    def _contact_forces_split(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(중간, 원위) 마디별 접촉력 (N, F) — 감쌈 판정용.

        `finger_sensor_bodies` 규약: 마지막 원소 = 팁, 그 앞이 (중간, 원위) 순.
        body 가 하나뿐인 손가락은 그 접촉 자체가 감쌈이다(mid=dist=그 body).
        """
        mids, dists = [], []
        for finger in self._finger_names:
            sensors = self._finger_sensors[finger]
            mid_i, dist_i = (0, 1) if len(sensors) >= 3 else (0, 0)

            def _mag(s):
                return s.data.force_matrix_w.view(
                    self.num_envs, -1, 3).sum(dim=1).norm(dim=-1)

            mids.append(_mag(sensors[mid_i]))
            dists.append(_mag(sensors[dist_i]))
        return torch.stack(mids, dim=1), torch.stack(dists, dim=1)

    def _tip_contact_forces(self) -> torch.Tensor:
        """손가락별 **팁만** 접촉력 (N, F)."""
        out = []
        for finger in self._finger_names:
            s = self._finger_sensors[finger][-1]
            out.append(s.data.force_matrix_w.view(
                self.num_envs, -1, 3).sum(dim=1).norm(dim=-1))
        return torch.stack(out, dim=1)

    def _env_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return pos_w - self.scene.env_origins

    def _palm_ee_R(self) -> torch.Tensor:
        """palm 회전행렬 (N,3,3). 열 0 = 손바닥 법선(+x), 열 1 = +y."""
        return matrix_from_quat(self.robot.data.body_quat_w[:, self.palm_idx])

    def _palm_pose_6d(self) -> torch.Tensor:
        """현재 palm pose (env-local xyz + euler_zyx) — fabric 명령과 같은 규약."""
        pos = self.robot.data.body_pos_w[:, self.palm_idx] - self.scene.env_origins
        r, pi, y = euler_xyz_from_quat(self.robot.data.body_quat_w[:, self.palm_idx])
        return torch.cat([pos, torch.stack([y, pi, r], dim=1)], dim=1)

    # ------------------------------------------------------------------
    # 지령 시각화 (env0 · GUI/카메라 렌더일 때만 — headless 비용 0)
    # ------------------------------------------------------------------
    def _setup_cmd_markers(self) -> None:
        self._cmd_markers = None
        if not bool(self.cfg.enable_cmd_markers):
            return
        try:
            import carb
            _cams = bool(carb.settings.get_settings().get("/isaaclab/cameras_enabled"))
        except Exception:
            _cams = False
        if not (self.sim.has_gui() or _cams):
            return

        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        _L = float(self.cfg.cmd_marker_axis_len)
        _r = float(self.cfg.cmd_marker_radius)

        def _axis(color):
            return sim_utils.CylinderCfg(
                radius=_r, height=_L,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))

        self._cmd_markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/GraspS2RCmd",
            markers={
                "cmd": sim_utils.SphereCfg(                      # 지령 원점(흰)
                    radius=_r * 2.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 1.0, 1.0))),
                "ax_x": _axis((0.9, 0.2, 0.2)),
                "ax_y": _axis((0.2, 0.9, 0.2)),
                "ax_z": _axis((0.25, 0.45, 1.0)),
                "palm": sim_utils.SphereCfg(                     # 실제 palm(노랑)
                    radius=_r * 2.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.85, 0.1))),
                "goal": sim_utils.SphereCfg(                     # 이송 목표(하늘)
                    radius=_r * 2.5,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.3, 0.8, 1.0))),
            }))
        self._cmd_marker_idx = torch.arange(6, device=self.device)
        # 원통 기본축은 +z — x/y 로 눕히는 정렬 쿼터니언(상수).
        _s = math.sqrt(0.5)
        self._cmd_axis_align = torch.tensor(
            [[_s, 0.0, _s, 0.0],        # z→x : +90° about y
             [_s, -_s, 0.0, 0.0],       # z→y : −90° about x
             [1.0, 0.0, 0.0, 0.0]],     # z→z : 항등
            device=self.device)
        print(f"[grasp_s2r] 지령 마커 ON — env0 전용 · 축 {_L * 1000:.0f}mm · "
              f"{'GUI' if self.sim.has_gui() else '카메라 녹화'}", flush=True)

        if bool(self.cfg.gui_focus_env0) and self.sim.has_gui():
            _o0 = self.scene.env_origins[0].tolist()
            _eye = [a + b for a, b in zip(self.cfg.gui_camera_eye, _o0)]
            _tgt = [a + b for a, b in zip(self.cfg.gui_camera_target, _o0)]
            self.sim.set_camera_view(eye=_eye, target=_tgt)

    def _update_cmd_markers(self) -> None:
        if self._cmd_markers is None:
            return
        _o0 = self.scene.env_origins[0]
        # palm_targets 는 **fabric 프레임** — env 보정 후 world 로.
        _p = self.palm_targets[0, :3] + self._fab_to_env + _o0
        _e = self.palm_targets[0, 3:6]                    # euler_zyx = (yaw, pitch, roll)
        _q = quat_from_euler_xyz(_e[2:3], _e[1:2], _e[0:1])[0]
        _R = matrix_from_quat(_q.unsqueeze(0))[0]
        _L = float(self.cfg.cmd_marker_axis_len)
        # 원통은 중심 배치 → 축 방향으로 L/2 밀어야 원점에서 뻗어 나간다.
        _tr = torch.stack([
            _p,
            _p + _R[:, 0] * (_L * 0.5),
            _p + _R[:, 1] * (_L * 0.5),
            _p + _R[:, 2] * (_L * 0.5),
            self.robot.data.body_pos_w[0, self.palm_idx],
            self.goal_pos[0] + _o0,
        ], dim=0)
        _qa = quat_mul(_q.unsqueeze(0).expand(3, 4), self._cmd_axis_align)
        _ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0)
        _or = torch.cat([_ident, _qa, _ident, _ident], dim=0)
        self._cmd_markers.visualize(
            translations=_tr, orientations=_or, marker_indices=self._cmd_marker_idx)
