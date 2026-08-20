"""grasp_lift_fabric — Fabrics 팔 + 절대 손 액션 (direct).

로봇 종속 정보는 전부 `modules.robots.RobotProfile` 이 공급한다 — 이 파일에
조인트/바디 이름 하드코딩 금지(계약 테스트가 소스 grep 으로 강제한다).

제어 스택:
  팔  = Geometric Fabrics. 정책이 **절대 palm 6D pose**(워크스페이스 박스 안)를 내면
        Fabrics 가 충돌회피·관절한계·매끄러운 궤적을 보장하며 관절 목표를 만든다.
        같은 Fabrics 를 실기에서도 돌리므로 sim/real 의 팔 거동이 같은 함수가 된다.
  손  = 관절 **절대** 목표(전범위). 적분기·래치·커플링·시너지·PCA 없음.

★obs 에 `fabric_q`/`fabric_qd` 를 넣지 않는다 — 실기에 없는 내부 상태이고,
  grasp_v1 이 s2r 불가였던 직접 원인이다.
"""

from __future__ import annotations

import math
import os
import sys

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from openarm.agnostic.modules import adr as _adr
from openarm.agnostic.modules import object_bank as _ob
from openarm.agnostic.modules import robots as _rb

from . import grasp_lift_fabric_env_cfg as _cfg
from .grasp_lift_fabric_env_cfg import GraspLiftFabricEnvCfg
from .rewards import compute_rewards

# Fabrics 경로 (hdgp/source/FABRICS/src 우선)
_FABRICS_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..",
                 "FABRICS", "src")
)
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


class GraspLiftFabricEnv(DirectRLEnv):
    cfg: GraspLiftFabricEnvCfg

    # ==================================================================
    def __init__(self, cfg: GraspLiftFabricEnvCfg, render_mode: str | None = None, **kw):
        # ★hydra CLI 오버라이드(`env.object_bank=...` 등)는 cfg 필드만 덮어쓰고
        #   __post_init__ 을 다시 돌리지 않는다. 파생값(자산 cfg·차원·replicate_physics)을
        #   여기서 맞추지 않으면 "cup_family 라고 적혀 있는데 컵 하나만 스폰" 같은
        #   조용히 틀린 조합으로 학습이 돈다. resolve_cfg 는 멱등이다.
        _cfg.resolve_cfg(cfg)
        self.profile = _rb.get(cfg.profile_name)
        self.bank = _ob.get(cfg.object_bank, expected_size=cfg.object_bank_expected_size)
        super().__init__(cfg, render_mode, **kw)
        p = self.profile

        # ---- 조인트/바디 해석 (fail-loud) --------------------------------------
        self.arm_ids, arm_names = self.robot.find_joints(p.arm_joint_regex)
        self.hand_ids, hand_names = self.robot.find_joints(p.hand_joint_regex)
        if len(self.arm_ids) != p.num_arm_joints or len(self.hand_ids) != p.num_hand_joints:
            raise RuntimeError(
                f"[{p.name}] 프로필 조인트 수 불일치: "
                f"arm {len(self.arm_ids)}!={p.num_arm_joints} ({arm_names}), "
                f"hand {len(self.hand_ids)}!={p.num_hand_joints} ({hand_names})"
            )
        self._arm_t = torch.tensor(self.arm_ids, device=self.device, dtype=torch.long)
        self._hand_t = torch.tensor(self.hand_ids, device=self.device, dtype=torch.long)

        # ★외전 관절은 정책이 제어하지 않고 init 값에 고정한다 — 손가락 교차를
        #   자유도 수준에서 없앤다(self-collision 을 끈 채 두는 대신).
        _frozen = []
        for jn in p.frozen_hand_joints:
            ids, _ = self.robot.find_joints(jn)
            if len(ids) != 1:
                raise RuntimeError(f"[{p.name}] 고정 관절 '{jn}' 해석 실패: {ids}")
            _frozen.append(ids[0])
        _fset = set(_frozen)
        self._hand_free_t = torch.tensor(
            [i for i in self.hand_ids if i not in _fset], device=self.device, dtype=torch.long)
        n_free = len(self._hand_free_t)
        if 6 + n_free != self.cfg.action_space:
            raise RuntimeError(
                f"[{p.name}] 액션 차원 불일치: 6+{n_free} != {self.cfg.action_space}. "
                "frozen_hand_joints 와 cfg 파생이 어긋났다."
            )
        # ★손가락 상호 겹침을 **학습 중에** 잰다. 인위적 probe(컵과 무관하게 강제 폐합)는
        #   정책 거동이 아니라 100% 겹침으로 나온다 — 실제로 그랬다. in-situ 로 봐야 한다.
        _lnk, _own = [], []
        for _fi, _f in enumerate(p.fingers):
            for _b in p.finger_tip_bodies[_f] + p.finger_wrap_bodies.get(_f, ()):
                _ids, _ = self.robot.find_bodies(_b)
                _lnk.append(_ids[0]); _own.append(_fi)
        self._fl_t = torch.tensor(_lnk, device=self.device, dtype=torch.long)
        _o = torch.tensor(_own, device=self.device)
        self._fl_diff = (_o[:, None] != _o[None, :])      # 다른 손가락 쌍만

        print(f"[grasp_lift_fabric] 손 관절 {len(self.hand_ids)} 중 "
              f"{len(_frozen)}개 고정(외전) → 정책 제어 {n_free}", flush=True)

        self.palm_idx = self._one_body(p.palm_body)
        self._tip_t = torch.tensor(
            [self._one_body(b) for b in p.fingertip_bodies],
            device=self.device, dtype=torch.long,
        )

        # ---- palm 워크스페이스 박스 (액션 [-1,1] → 절대 pose) --------------------
        # euler_zyx 중심: 우팔 [90,0,90]°, 좌팔은 부호 반전. ±max_pose_angle.
        a = float(cfg.max_pose_angle_deg)
        sign = 1.0 if p.side == "r" else -1.0
        centre = (sign * 90.0, 0.0, sign * 90.0)
        # ★박스는 **프로필**이 준다(팔 도달범위 = 로봇 종속 정보).
        #   태스크 cfg 상수로 두면 자산이 바뀔 때 조용히 틀린다 — 실제로 그랬다.
        #   좌우 미러는 프로필이 이미 반영하고 있으므로 여기서 뒤집지 않는다.
        lo = list(p.palm_box_min) + [(c - a) * _DEG for c in centre]
        hi = list(p.palm_box_max) + [(c + a) * _DEG for c in centre]
        if not p.palm_box_verified:
            print(f"[grasp_lift_fabric] ⚠ 프로필 '{p.name}' 의 palm 박스는 **미실측**이다"
                  " — probe_workspace_reach.py 로 확인할 것(다른 로봇 값을 물려받았을 수 있다).",
                  flush=True)
        self.palm_lo = torch.tensor(lo, device=self.device).unsqueeze(0)
        self.palm_hi = torch.tensor(hi, device=self.device).unsqueeze(0)

        # ---- Fabrics -------------------------------------------------------------
        self._setup_fabrics()

        # ---- 홈 palm pose 실측 (액션 매핑의 기준점) --------------------------------
        # ★액션이 절대값이라 "가만히 있기"에 해당하는 값이 필요하다. 박스 중심으로
        #   두면 정책 초기 출력(≈0)이 곧 홈에서 0.29m 떨어진 곳으로의 돌진이 된다
        #   (probe 실측 0.29m). a=0 → 홈 이 되도록 구간별 선형으로 매핑한다.
        # ★★__init__ 시점의 body_pos_w 는 **stale** 이다(로봇이 아직 홈에 배치되기 전).
        #   그대로 읽으면 [0.002, -0.139, 0.187] 같은 미배치 자세가 나온다.
        #   홈을 물리로 확정한 뒤 읽는다 — 이 저장소에서 반복된 버퍼 stale 함정.
        _q_home = self.robot.data.default_joint_pos.clone()
        self.robot.write_joint_state_to_sim(_q_home, torch.zeros_like(_q_home))
        self.robot.set_joint_position_target(_q_home)
        self.scene.write_data_to_sim()
        for _ in range(2):
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        self.home_palm = self._palm_pose_6d()
        _in_box = ((self.home_palm >= self.palm_lo) & (self.home_palm <= self.palm_hi)).all()
        if not bool(_in_box):
            raise RuntimeError(
                f"[{p.name}] 홈 palm pose 가 워크스페이스 박스 밖이다.\n"
                f"  홈   {[round(v, 3) for v in self.home_palm[0].tolist()]}\n"
                f"  박스 lo {[round(v, 3) for v in self.palm_lo[0].tolist()]}\n"
                f"       hi {[round(v, 3) for v in self.palm_hi[0].tolist()]}"
            )

        # ---- 관절 한계 / 버퍼 ------------------------------------------------------
        jl = self.robot.data.soft_joint_pos_limits          # (N, J, 2)
        m = float(cfg.hand_limit_margin)
        self._hand_lo = jl[:, self._hand_free_t, 0] + m
        self._hand_hi = jl[:, self._hand_free_t, 1] - m
        self._arm_lo = jl[:, self._arm_t, 0]
        self._arm_hi = jl[:, self._arm_t, 1]
        self._default_q = self.robot.data.default_joint_pos.clone()

        A = cfg.action_space
        self.actions = torch.zeros(self.num_envs, A, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, A, device=self.device)
        # 자유 관절만 정책이 움직인다. 고정 관절은 default(init) 값 그대로 명령된다.
        self.hand_targets = self._default_q[:, self._hand_free_t].clone()
        self.palm_targets = torch.zeros(self.num_envs, 6, device=self.device)

        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_spawn_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._respawn_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_reached_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ---- 접촉 그룹 인덱스 -------------------------------------------------------
        fingers = list(p.fingers)
        self._fingers = fingers
        self._grp_a = torch.tensor([fingers.index(f) for f in p.contact_group_a],
                                   device=self.device, dtype=torch.long)
        self._grp_b = torch.tensor([fingers.index(f) for f in p.contact_group_b],
                                   device=self.device, dtype=torch.long)

        # ---- 물체 배정 / onehot ------------------------------------------------------
        self.object_idx = torch.tensor(self.bank.assign_indices(self.num_envs),
                                       device=self.device, dtype=torch.long)
        # ★물체별 "작업면에 놓인" z = 작업면 상면 + 그 물체의 원점 오프셋.
        #   합쳐서 상수 하나로 두면 물체가 바뀔 때 조용히 틀린 높이가 되고,
        #   그 값이 곧 lift 보상의 **기준선**이라 보상이 통째로 오염된다
        #   (실측: 구 설정은 스폰 0.302 vs 안착 0.2773 → height_delta 가 -24.7mm 에서 시작).
        _rest = torch.tensor(
            [s.origin_offset_z for s in self.bank.specs], device=self.device
        ) + float(p.surface_z)
        self._object_rest_z = _rest[self.object_idx]
        self._onehot = None
        if cfg.enable_object_onehot:
            self._onehot = torch.nn.functional.one_hot(
                self.object_idx, num_classes=self.bank.onehot_dim).float()

        # ---- ADR (축 하나: 스폰 반경) -------------------------------------------------
        self.adr = _adr.TaskADR(
            {"spawn": {"xy_range": (cfg.spawn_range_initial, cfg.spawn_range_final)}},
            num_increments=cfg.adr_num_increments,
            increment_interval=cfg.adr_increment_interval,
            trigger_threshold=cfg.adr_trigger_threshold,
            enabled=bool(cfg.enable_adr),
            event_manager=getattr(self, "event_manager", None) if cfg.enable_physics_dr else None,
            physics_cfg=self._physics_terminal() if cfg.enable_physics_dr else None,
        )

        # ★★접촉 필터 prim 이 rigid body 를 가리키는지 부팅 시 확인한다.
        #   루트 Xform 을 가리키면 PhysX 가 "GPU contact filter … is not supported" 를
        #   내고 force_matrix_w 가 **항상 0** 이 된다. 보상 7항 중 6항이 접촉 게이트에
        #   걸려 있으므로 그 상태로 학습하면 approach 하나만 남는다 —
        #   fab_test1~4 를 그렇게 날렸다(force_max 가 전 구간 정확히 0.0000).
        _flt = list(self.cfg.object_contact_filter)
        _want = self.bank.rigid_body_name
        if not _flt or not all(f.rstrip("/").endswith("/" + _want) for f in _flt):
            raise RuntimeError(
                f"object_contact_filter 가 rigid body prim('{_want}')을 안 가리킨다: {_flt}\n"
                "  루트 Xform 을 가리키면 PhysX 가 force_matrix_w 를 항상 0 으로 준다 —\n"
                "  보상 7항 중 6항이 접촉 게이트라 학습이 approach 하나로 줄어든다."
            )

        print(
            f"[grasp_lift_fabric] profile={p.name} asset={p.asset.name} "
            f"fabric={p.fabric_robot_dir} bank={self.bank.name}({len(self.bank)}) "
            f"action={A} obs={cfg.observation_space} critic={cfg.state_space} "
            f"dr={cfg.enable_physics_dr} adr={cfg.enable_adr}",
            flush=True,
        )

    # ------------------------------------------------------------------
    def _one_body(self, name: str) -> int:
        ids, _ = self.robot.find_bodies(name)
        if len(ids) != 1:
            raise RuntimeError(f"[{self.profile.name}] body '{name}' 해석 실패: {ids}")
        return ids[0]

    def _physics_terminal(self):
        from openarm.agnostic.modules import physics_dr as _pdr
        return _pdr.PHYSICS_ADR_TERMINAL

    def _palm_pose_6d(self) -> torch.Tensor:
        """현재 palm pose (env-local xyz + euler_zyx) — fabric 명령과 같은 규약."""
        from isaaclab.utils.math import euler_xyz_from_quat

        pos = self.robot.data.body_pos_w[:, self.palm_idx] - self.scene.env_origins
        r, pi, y = euler_xyz_from_quat(self.robot.data.body_quat_w[:, self.palm_idx])
        return torch.cat([pos, torch.stack([y, pi, r], dim=1)], dim=1)

    def _build_fabric_index(self) -> torch.Tensor:
        """프로필의 `fabric_joint_order` → articulation 인덱스 텐서.

        ★articulation 은 depth-major (index_1, middle_1, pinky_1, ring_1, thumb_1, …)
          인데 fabric URDF 는 **finger-major** (thumb_1..4, index_1..4, …) 다.
          `cat([q[:, arm_t], q[:, hand_t]])` 로 만들면 손 20관절이 통째로 어긋나고,
          fabric 이 엉뚱한 손 자세로 충돌구 FK 를 계산해 없는 자기충돌을 피하려 팔을 민다.
        """
        order = self.profile.fabric_joint_order
        if len(order) != self.fabric.num_joints:
            raise RuntimeError(
                f"[{self.profile.name}] fabric_joint_order 길이 {len(order)} != "
                f"fabric num_joints {self.fabric.num_joints}. 프로필을 고쳐라."
            )
        idx = []
        for name in order:
            ids, _ = self.robot.find_joints(name)
            if len(ids) != 1:
                raise RuntimeError(f"[{self.profile.name}] fabric 관절 '{name}' 해석 실패: {ids}")
            idx.append(ids[0])
        return torch.tensor(idx, device=self.device, dtype=torch.long)

    def _fabric_order(self, q: torch.Tensor) -> torch.Tensor:
        """로봇 articulation 순서 → fabric 순서."""
        return q[:, self._fab_t]

    # ==================================================================
    def _setup_fabrics(self) -> None:
        p = self.profile
        initialize_warp(str(self.device)[-1])   # 멀티 GPU 캐시 분리(grasp_v1 규약)
        self._world = WorldMeshesModel(
            batch_size=self.num_envs, device=self.device,
            max_objects_per_env=int(self.cfg.fabrics_max_objects_per_env),
        )
        self._world_ids, self._world_indicator = self._world.get_object_ids()

        cls = _fabric_class(p.fabric_class)
        self.fabric = cls(
            batch_size=self.num_envs, device=self.device,
            timestep=float(self.cfg.fabrics_dt),
            graph_capturable=bool(self.cfg.fabric_use_cuda_graph),
            use_hand_fabric=False,           # 손은 Fabrics 밖(직접 관절 PD)
            robot_dir_name=p.fabric_robot_dir,
            robot_name=p.fabric_robot_dir,
        )
        self.integrator = DisplacementIntegrator(self.fabric)

        # ★fabric 의 관절 순서는 [팔 N, 손 M] 이다. 로봇 articulation 순서와 다르다 —
        #   양팔 로봇에서 `default_joint_pos[:, :27]` 로 자르면 좌우가 섞인 전혀 다른
        #   관절이 들어간다(probe 실측: palm 이 2초에 61mm 밖에 못 움직이면서 관절속도는
        #   20 rad/s 로 포화). 반드시 프로필 인덱스로 재조립한다.
        n_j = self.fabric.num_joints
        n_arm = self.profile.num_arm_joints
        expect = self.profile.num_arm_joints + self.profile.num_hand_joints
        if n_j != expect:
            raise RuntimeError(
                f"[{self.profile.name}] fabric num_joints={n_j} 인데 프로필은 "
                f"팔{self.profile.num_arm_joints}+손{self.profile.num_hand_joints}={expect}. "
                "fabric URDF 와 자산이 어긋났다."
            )
        self._fab_t = self._build_fabric_index()
        self.fabric_q = self._fabric_order(self.robot.data.default_joint_pos).contiguous()
        self.fabric_qd = torch.zeros(self.num_envs, n_j, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, n_j, device=self.device)
        # use_hand_fabric=False 라 hand_target 은 무시되지만, 원본 계약(B,5 PCA)을 지킨다.
        self._fabric_hand_cmd = torch.zeros(self.num_envs, 5, device=self.device)

        # cspace attractor(널스페이스)의 rest 자세를 **프로필 홈**으로 맞춘다.
        #   fabric URDF 내장 기본값은 [1.0, -0.1, -0.6, 0.5, 0, 0, 0] 로 다른 셋업용이다.
        #   robot-agnostic 에서는 rest 자세도 프로필이 정하는 게 맞다.
        #   ※정직한 단서: 이 변경을 "추종 불량의 원인"으로 지목했다가 **철회**했다.
        #     그때 쓴 probe 가 scene.write_data_to_sim() 을 빠뜨려 관절 목표가 PhysX 에
        #     도달하지 않은 상태였다. 이 설정은 설계상 옳아서 남겨둘 뿐,
        #     추종 성능에 대한 근거는 아직 없다.
        _prior = self.fabric.default_config.clone()
        self.fabric.default_config.copy_(self.fabric_q)
        print(
            "[grasp_lift_fabric] fabric.default_config 를 프로필 홈으로 교체\n"
            f"  이전(fabric 내장) 팔 : {[round(v, 3) for v in _prior[0, :n_arm].tolist()]}\n"
            f"  이후(프로필 홈) 팔  : "
            f"{[round(v, 3) for v in self.fabric_q[0, :n_arm].tolist()]}",
            flush=True,
        )
        self._fabric_damping = float(self.cfg.fabrics_damping_gain) * torch.ones(
            self.num_envs, 1, device=self.device)

    # ==================================================================
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # 환경 픽스처: rigid body 없는 **정적** 삼각메시 콜라이더라 RigidObject 로
        # 올리지 않는다. 원점은 로봇 base link 원점에 붙인다(사용자 규약).
        _cfg_fix = self.cfg.env_fixture_spawn
        _cfg_fix.func(_cfg.ENV_FIXTURE_PRIM, _cfg_fix, translation=(0.0, 0.0, 0.0))

        # 손가락 마디별 접촉 센서 — body 마다 **개별** 생성한다.
        # (다중 body 단일 센서는 force_matrix_w 가 0 을 반환한다 — 실측 함정)
        p = self.profile
        flt = list(self.cfg.object_contact_filter)
        self._sensors: dict[str, dict[str, list]] = {}
        for finger in p.fingers:
            roles = {"tip": [], "wrap": []}
            for role, bodies in (("tip", p.finger_tip_bodies[finger]),
                                 ("wrap", p.finger_wrap_bodies.get(finger, ()))):
                for body in bodies:
                    s = ContactSensor(ContactSensorCfg(
                        prim_path=f"/World/envs/env_.*/Robot/{body}",
                        filter_prim_paths_expr=flt,
                        history_length=1, track_air_time=False,
                    ))
                    roles[role].append(s)
                    self.scene.sensors[f"contact_{finger}_{body}"] = s
            self._sensors[finger] = roles

        # ★기본 ground plane 을 z=0 에 두지 않는다 — env.usd 의 platform 상면이 정확히
        #   z=0 이라 지면과 겹쳐 상시 접촉/관통이 생긴다. 바닥은 env.usd 의 base_plate
        #   (z -0.025~-0.015)가 담당하므로, 지면은 그 아래로 내려 시각적 배경으로만 둔다.
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, float(self.cfg.ground_plane_z)),
        )
        light = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

        # ★clone → 물체 순서. 반대로 하면 env_0 만 존재하는 시점에 MultiAssetSpawner 가
        #   assets_cfg[0] 하나만 스폰하고 clone 이 그걸 전 env 에 복제한다(배정 붕괴).
        self.scene.clone_environments(copy_from_source=True)
        _ob.assert_spawned_after_clone(self.bank, cloned=True)
        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object
        if not self.cfg.scene.replicate_physics:
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ==================================================================
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        cfg = self.cfg
        self.actions = actions.clamp(-1.0, 1.0)

        # 팔: **절대** palm 6D pose. a=0 → 홈, a=+1 → 박스 상한, a=-1 → 박스 하한.
        # 구간별 선형이라 박스 전체에 도달하면서도 상태를 추가하지 않는다.
        a_arm = self.actions[:, :6]
        self.palm_targets = self.home_palm + torch.where(
            a_arm >= 0.0,
            a_arm * (self.palm_hi - self.home_palm),
            a_arm * (self.home_palm - self.palm_lo),
        )
        # 손: **절대** 관절 목표 (전범위). 손은 홈이 곧 "펴진 상태"라 대칭 매핑으로 둔다 —
        # a=-1 이 완전 개방, a=+1 이 완전 폐합이 되는 편이 학습에 자연스럽다.
        u_hand = 0.5 * (self.actions[:, 6:] + 1.0)
        self.hand_targets = self._hand_lo + u_hand * (self._hand_hi - self._hand_lo)

        self.fabric.set_features(
            self._fabric_hand_cmd, self.palm_targets, "euler_zyx",
            self.fabric_q.detach(), self.fabric_qd.detach(),
            self._world_ids, self._world_indicator, self._fabric_damping,
        )

        # ★떨어진 컵 리스폰은 여기서 한다(보상 계산 중 sim 상태 변경 금지).
        #   종료(회피 학습, agn_test2)도 방치(죽은 시간, agn_test3)도 아닌 세 번째 선택지.
        # ★`if tensor.any():` 는 매 스텝 GPU→CPU 동기화를 강제한다(util killer, 저장소 재발).
        #   분기 없이 **마스크 곱**으로 처리해 sync 를 없앤다: 리스폰 대상이 아니면
        #   현재 상태를 그대로 다시 써 넣으므로 결과가 같다.
        _m = self._respawn_pending.unsqueeze(1).float()
        _cur = self.object.data.root_state_w
        _resp = torch.zeros_like(_cur)
        _resp[:, :3] = self.object_spawn_pos + self.scene.env_origins
        _resp[:, 3] = 1.0
        self.object.write_root_state_to_sim(_m * _resp + (1.0 - _m) * _cur)
        self._respawn_pending[:] = False

        self._step_fabric()

    def _step_fabric(self) -> None:
        """★정책 스텝마다 **한 번**만 적분한다.

        `_apply_action` 은 decimation(=2) 번 호출되므로 그 안에서 적분하면
        fabric 시간이 2배로 빨리 흐른다(grasp_v1 은 _pre_physics_step 에서 적분한다).
        """
        for _ in range(int(self.cfg.fabric_decimation)):
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.integrator.step(
                self.fabric_q.detach(), self.fabric_qd.detach(),
                self.fabric_qdd.detach(), float(self.cfg.fabrics_dt),
            )

    def _apply_action(self) -> None:
        arm_target = self.fabric_q[:, : self.profile.num_arm_joints]
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_ids)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(arm_target), joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(
            self.hand_targets, joint_ids=self._hand_free_t.tolist())
        # 고정 관절은 init 값 유지 — reset 에서 write 한 뒤 target 도 default 로 둔다.

    # ==================================================================
    def _contact(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(손가락별 총 접촉력 (N,F), 손가락별 wrap 접촉 여부 (N,F))."""
        thr = float(self.cfg.contact_force_threshold)
        tot, wrapped = [], []
        for f in self._fingers:
            roles = self._sensors[f]
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

    def _object_tilt_deg(self) -> torch.Tensor:
        """물체 로컬 +z 와 world +z 사이 각도[deg]."""
        from isaaclab.utils.math import quat_apply
        z = torch.zeros(self.num_envs, 3, device=self.device)
        z[:, 2] = 1.0
        up = quat_apply(self.object.data.root_quat_w, z)
        return torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))

    # ==================================================================
    def _get_observations(self) -> dict:
        from isaaclab.utils.math import subtract_frame_transforms

        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        tau = self.robot.data.applied_torque
        idx = torch.cat([self._arm_t, self._hand_t])
        joint_pos, joint_vel, joint_eff = q[:, idx], qd[:, idx], tau[:, idx]

        # 물체를 **palm 프레임**으로 — palm/손끝 위치는 joint_pos 의 순수 FK 라 중복이고,
        # 이 상대 pose 하나가 관계 정보 전부다. 프레임도 이걸로 통일된다.
        palm_pos_w = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat_w = self.robot.data.body_quat_w[:, self.palm_idx]
        obj_in_palm, obj_quat_in_palm = subtract_frame_transforms(
            palm_pos_w, palm_quat_w,
            self.object.data.root_pos_w, self.object.data.root_quat_w)

        contact, _ = self._contact()
        goal_delta = self.goal_pos - self._local(self.object.data.root_pos_w)

        parts = [joint_pos, joint_vel, joint_eff, contact.clamp(max=20.0),
                 obj_in_palm, obj_quat_in_palm, goal_delta, self.actions]
        if self._onehot is not None:
            parts.append(self._onehot)
        obs = torch.cat(parts, dim=1)
        state = torch.cat([obs, self.object.data.root_lin_vel_w,
                           self.object.data.root_ang_vel_w], dim=1)
        return {"policy": torch.nan_to_num(obs), "critic": torch.nan_to_num(state)}

    # ==================================================================
    def _get_rewards(self) -> torch.Tensor:
        obj_pos = self._local(self.object.data.root_pos_w)
        palm_pos = self._local(self.robot.data.body_pos_w[:, self.palm_idx])
        tips = self.robot.data.body_pos_w[:, self._tip_t] - self.scene.env_origins[:, None, :]

        contact, wrapped = self._contact()
        thr = float(self.cfg.contact_force_threshold)
        grip_frac = (contact > thr).float().mean(dim=1)
        envelope_frac = (wrapped.mean(dim=1) if self.profile.has_wrap_sensors else grip_frac)

        xy_disp = (obj_pos[:, :2] - self.object_spawn_pos[:, :2]).norm(dim=-1)
        tilt = self._object_tilt_deg()
        goal_dist = (obj_pos - self.goal_pos).norm(dim=-1)

        total, terms, gate, up_q = compute_rewards(
            palm_to_object=(palm_pos - obj_pos).norm(dim=-1),
            tip_side_dist=(tips - obj_pos[:, None, :]).norm(dim=-1).mean(dim=1),
            envelope_frac=envelope_frac,
            grip_frac=grip_frac,
            object_height_delta=obj_pos[:, 2] - self.object_spawn_pos[:, 2],
            object_to_goal=goal_dist,
            object_xy_displacement=xy_disp,
            object_tilt_deg=tilt,
            group_a_force=contact[:, self._grp_a],
            group_b_force=contact[:, self._grp_b],
            actions=self.actions,
            prev_actions=self.prev_actions,
            cfg=self.cfg,
        )
        # ★지터는 prev_actions 갱신 **전에** 재야 한다(갱신 후엔 항상 0).
        #   절대 액션이라 정책 노이즈가 곧 목표 순간이동이 된다 — 이 값이 크면
        #   Fabrics 가 저역통과시켜 정책 액션이 실제 palm 위치에 거의 영향을 못 준다.
        self.extras["action/arm_step_delta"] = (
            self.actions[:, :6] - self.prev_actions[:, :6]).abs().mean()
        self.extras["action/hand_step_delta"] = (
            self.actions[:, 6:] - self.prev_actions[:, 6:]).abs().mean()
        self.extras["action/arm_abs_mean"] = self.actions[:, :6].abs().mean()
        self.prev_actions.copy_(self.actions)
        self._goal_reached_now = goal_dist < float(self.cfg.success_pos_tolerance)

        # 떨어진 컵 → 다음 스텝 시작에 리스폰(여기서 sim 을 건드리지 않는다)
        fell = obj_pos[:, 2] < float(self.cfg.object_min_z)
        self._respawn_pending |= fell

        # ---- ADR (트리거 = **순간** 성공률) --------------------------------------
        if self.adr.maybe_increment(self._goal_reached_now.float().mean()):
            em = getattr(self, "event_manager", None)
            if em is not None:
                em.reset(env_ids=self.robot._ALL_INDICES)
                em.apply(env_ids=self.robot._ALL_INDICES, mode="reset",
                         global_env_step_count=0)

        # ---- 로깅 ----------------------------------------------------------------
        for k, v in terms.items():
            self.extras[f"reward/{k}"] = v.mean()
        self.extras["reward/total"] = total.mean()
        self.extras["task/contact_gate"] = gate.float().mean()
        self.extras["task/envelope_frac"] = envelope_frac.mean()
        self.extras["task/grip_frac"] = grip_frac.mean()
        self.extras["task/upright_quality"] = up_q.mean()
        # ★접촉력 **원값**. 이게 없으면 grip_frac=0 일 때 "진짜 미접촉"인지
        #   "임계(contact_force_threshold) 바로 아래"인지 구분할 수단이 없다
        #   — fab_test1 에서 실제로 그 구분이 안 돼 probe 를 따로 돌려야 했다.
        _best = contact.max(dim=1).values
        self.extras["contact/force_max"] = contact.max()
        self.extras["contact/force_best_finger"] = _best.mean()
        self.extras["contact/n_over_thr"] = (contact > thr).float().sum(dim=1).mean()
        self.extras["contact/n_over_tenth"] = (contact > thr * 0.1).float().sum(dim=1).mean()
        self.extras["contact/group_a"] = contact[:, self._grp_a].max(dim=1).values.mean()
        self.extras["contact/group_b"] = contact[:, self._grp_b].max(dim=1).values.mean()
        # 손끝 최소거리 — approach 의 합만으론 "닿기 직전"인지 알 수 없다
        self.extras["task/tip_min_dist"] = (
            (tips - obj_pos[:, None, :]).norm(dim=-1).min(dim=1).values.mean()
        )
        self.extras["task/palm_dist"] = (palm_pos - obj_pos).norm(dim=-1).mean()
        # 손가락 상호 최소거리 — 마디 반경 ~10mm 이므로 20mm 미만이면 겹침이다.
        # 이 값이 작으면 envelope_frac 이 물리적으로 불가능한 감쌈을 세고 있다는 뜻.
        _fp = self.robot.data.body_pos_w[:, self._fl_t]
        _fd = torch.cdist(_fp, _fp).masked_fill(~self._fl_diff.unsqueeze(0), float("inf"))
        _fmin = _fd.view(self.num_envs, -1).min(dim=1).values
        self.extras["hand/finger_min_dist"] = _fmin.mean()
        self.extras["hand/overlap_frac"] = (_fmin < 0.020).float().mean()
        self.extras["task/object_height_delta"] = (
            obj_pos[:, 2] - self.object_spawn_pos[:, 2]).mean()
        self.extras["task/goal_dist"] = goal_dist.mean()
        self.extras["task/goal_reached"] = self._goal_reached_now.float().mean()
        self.extras["task/respawn_rate"] = fell.float().mean()
        # ★측정 공백 해소: grasp_v1 은 이 값을 보상에 쓰면서 로깅하지 않았다.
        self.extras["obj/xy_displacement"] = xy_disp.mean()
        self.extras["obj/xy_disp_p95"] = torch.quantile(xy_disp, 0.95)
        self.extras["obj/tilt_deg"] = tilt.mean()
        # Fabrics 추종 오차 — 정상상태 오차(78mm 이력)를 학습 중에도 본다.
        perr = (self.palm_targets[:, :3] - palm_pos).norm(dim=-1)
        self.extras["fabric/palm_err_mean"] = perr.mean()
        self.extras["fabric/palm_err_p95"] = torch.quantile(perr, 0.95)
        # ★"정책이 컵을 겨냥은 하는가"와 "겨냥은 하는데 못 가는가"를 가른다.
        #   target_to_obj 작고 palm_dist 크면 → 추종/도달 문제
        #   target_to_obj 자체가 크면        → 정책이 컵을 안 겨냥 (탐색 문제)
        self.extras["fabric/target_to_obj"] = (
            self.palm_targets[:, :3] - obj_pos).norm(dim=-1).mean()
        tau = self.robot.data.applied_torque[:, self._hand_t].abs()
        self.extras["debug/hand/torque_mean"] = tau.mean()
        self.extras["debug/hand/torque_max"] = tau.max()
        self.extras.update(self.adr.log_dict())
        self.extras["adr/trigger_metric"] = self._goal_reached_now.float().mean()

        # ── 학습 로그(stdout)에 주기적으로 한 줄 남긴다 ────────────────────────
        # TFEvents 를 파싱하지 않고 `grep METRICS <train.log>` 만으로 추적할 수 있게.
        # ★접촉 관련 원값을 반드시 포함한다 — 0 이 "진짜 0"인지 "임계 아래"인지
        #   구분 못 해 fab_test1~4 를 날린 이력이 있다.
        self._log_tick = getattr(self, "_log_tick", 0) + 1
        _every = int(getattr(self.cfg, "console_log_interval", 600))
        if _every > 0 and self._log_tick % _every == 0:
            print(
                f"[METRICS] step={self._log_tick:>8d}"
                f" rew={total.mean():+.3f}"
                f" approach={terms['approach'].mean():+.3f}"
                f" gate={gate.float().mean():.3f}"
                f" grip={grip_frac.mean():.3f}"
                f" env={envelope_frac.mean():.3f}"
                # ★max 만 찍으면 오독한다 — 2048x5 중 하나의 스파이크가 전형값처럼 보인다.
                #   실제로 그랬다(513N 보고 → 전형은 20N). 전형값을 먼저 놓는다.
                f" F={_best.mean():.1f}N/Fmax={contact.max():.0f}N"
                f" nF>thr={(contact > thr).float().sum(dim=1).mean():.2f}"
                f" dz={(obj_pos[:, 2] - self.object_spawn_pos[:, 2]).mean():+.4f}"
                f" goal={self._goal_reached_now.float().mean():.3f}"
                f" tip={((tips - obj_pos[:, None, :]).norm(dim=-1).min(dim=1).values).mean():.3f}"
                f" tgt2obj={(self.palm_targets[:, :3] - obj_pos).norm(dim=-1).mean():.3f}"
                f" perr={perr.mean():.3f}"
                # ★prev_actions 는 위에서 이미 갱신됐다 — 여기서 다시 빼면 **항상 0** 이다.
                #   extras 에 미리 담아둔(갱신 전 계산) 값을 쓴다. fab_test5 로그의
                #   jit 필드가 그래서 0 으로 찍혔다(TFEvents 쪽은 정상).
                f" jit={float(self.extras['action/arm_step_delta']):.3f}"
                f" fmin={float(self.extras['hand/finger_min_dist'])*1000:.0f}mm"
                f" ovl={float(self.extras['hand/overlap_frac']):.2f}",
                flush=True,
            )
        return total

    # ==================================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Fabrics 의 JointLimitRepulsion 이 관절한계를 담당하므로 한계 초과는 종료 사유가
        # 아니다. 남은 물리 위반은 속도 폭주뿐.
        # ★fell/out 은 종료하지 않는다 — 종료는 "접근 회피"를 가르쳤다(agn_test2 실측:
        #   reaching 0.056→0.005 붕괴 + episode_lengths 동반 상승). 리스폰으로 처리한다.
        qd = self.robot.data.joint_vel[:, self._arm_t]
        runaway = (qd.abs() > float(self.cfg.runaway_joint_vel)).any(dim=-1)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["task/runaway_rate"] = runaway.float().mean()
        return runaway, truncated

    # ==================================================================
    def _reset_idx(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        p = self.profile

        q0 = self._default_q[env_ids].clone()
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), env_ids=env_ids)
        self.robot.set_joint_position_target(q0, env_ids=env_ids)
        self.hand_targets[env_ids] = q0[:, self._hand_free_t]
        self.fabric_q[env_ids] = q0[:, self._fab_t]
        self.fabric_qd[env_ids] = 0.0
        self.fabric_qdd[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._respawn_pending[env_ids] = False
        self._goal_reached_now[env_ids] = False

        rng = self.adr.get_param("spawn", "xy_range")
        offs = (torch.rand(n, 2, device=self.device) - 0.5) * 2.0 * rng
        spawn = torch.zeros(n, 3, device=self.device)
        spawn[:, 0] = p.object_spawn_center[0] + offs[:, 0]
        spawn[:, 1] = p.object_spawn_center[1] + offs[:, 1]
        spawn[:, 2] = self._object_rest_z[env_ids] + float(self.cfg.object_spawn_pad)
        self.object_spawn_pos[env_ids] = spawn
        self.goal_pos[env_ids] = spawn + torch.tensor(
            [0.0, 0.0, float(self.cfg.goal_height_offset)], device=self.device)

        root = torch.zeros(n, 13, device=self.device)
        root[:, :3] = spawn + self.scene.env_origins[env_ids]
        root[:, 3] = 1.0
        self.object.write_root_state_to_sim(root, env_ids=env_ids)
