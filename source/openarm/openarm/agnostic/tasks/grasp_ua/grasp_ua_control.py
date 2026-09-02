"""grasp_ua 제어 스택 — Fabrics 팔 + 관절공간 시너지 손.

`agnostic/tasks/grasp_sensor` 에서 검증된 배선을 그대로 이식했다. 그 트랙은 손 제어
4모드(pd/fabric/tip_cyl/synergy) 분기를 갖고 있었는데, 여기서는 **synergy 하나만**
남긴다(나머지는 전부 기각된 경로다 — 죽은 분기는 나중에 고칠 때 오해만 만든다).

env 본체(`grasp_ua_env.py`)가 이 믹스인을 상속한다. 여기 있는 것은 전부 "어떻게
움직이는가"이고, "무엇을 보상하는가"는 env 와 `grasp_ua_rewards.py` 에 있다.
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

import fabrics_sim.fabrics.openarm_rh56f1_pose_fabric as _fab_rh56f1
import fabrics_sim.fabrics.openarm_tesollo_pose_fabric as _fab_tesollo
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

from .robot_profiles import PROFILES

# ★로봇을 추가하면 그 fabric 모듈을 여기 넣는다 — 이름 해석은 프로필 문자열로만 하고
#   env 코드에는 로봇명을 두지 않는다는 계약을 유지한다.
_FABRIC_MODULES = (_fab_tesollo, _fab_rh56f1)


def _fabric_class(name: str):
    """프로필의 문자열 이름 → fabric 클래스. env 에 로봇명을 하드코딩하지 않는 계약."""
    for mod in _FABRIC_MODULES:
        if hasattr(mod, name):
            return getattr(mod, name)
    raise RuntimeError(
        f"fabric 클래스 '{name}' 를 찾을 수 없다: {[m.__name__ for m in _FABRIC_MODULES]}")


# ★fabric 클래스마다 생성자가 다르다(tesollo 는 이 트랙용으로 확장됐고, RH56F1 은
#   자매 트랙의 시그니처를 그대로 쓴다). 안 받는 인자를 **조용히 버리면** 기능이
#   소리 없이 사라지므로, 중립값일 때만 버리고 아니면 죽인다.
_FABRIC_NEUTRAL = {
    "tip_per_finger": False,
    "use_hand_repulsion": False,
    "use_body_repulsion_pairs": False,
    "palm_position_only": False,
    "use_tip_fabric": False,
}


def _filter_fabric_kwargs(cls, kw: dict, profile) -> dict:
    """생성자가 받는 인자만 남긴다. 버려지는 인자가 기능을 끄는 값이 아니면 fail-loud."""
    import inspect
    accepted = set(inspect.signature(cls.__init__).parameters)
    out, dropped, fatal = {}, [], []
    for k, v in kw.items():
        if k in accepted:
            out[k] = v
        elif k in _FABRIC_NEUTRAL and v != _FABRIC_NEUTRAL[k]:
            fatal.append(f"{k}={v}")
        else:
            dropped.append(k)
    if fatal:
        raise RuntimeError(
            f"[{profile.name}] fabric '{cls.__name__}' 가 {fatal} 를 받지 않는다 — "
            "그 기능을 켠 채로 못 돈다. cfg 에서 끄거나 fabric 클래스를 확장하라.")
    if dropped:
        print(f"[grasp_ua] fabric '{cls.__name__}' 미지원 인자 생략: {sorted(dropped)} "
              "(중립값이거나 클래스가 자체 고정하는 값 — 배선은 부팅 FK 게이트가 검증한다)",
              flush=True)
    return out


class GraspUAControlMixin:
    """씬 구성 · Fabrics · 시너지 손 · 접촉 센서 · 지령 마커."""

    # ------------------------------------------------------------------
    # 씬
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        # ★`_palm_ee_R()` 이 이 뒤 어디서든 불릴 수 있으므로 프레임 리맵을 가장 먼저
        #   잡는다(프로필이 비우면 None = 항등, 기존 로봇 무변화).
        _prof = PROFILES[self.cfg.profile_name]      # ★self.profile 은 아직 없다
        _rm = _prof.palm_frame_remap
        if _rm:
            _M = torch.tensor(_rm, device=self.device, dtype=torch.float32)
            if _M.shape != (3, 3):
                raise RuntimeError(
                    f"[{_prof.name}] palm_frame_remap 은 3x3 이어야 한다: {_M.shape}")
            _err = float((_M @ _M.transpose(0, 1) - torch.eye(3, device=self.device)).abs().max())
            _det = float(torch.det(_M))
            if _err > 1e-4 or abs(_det - 1.0) > 1e-4:
                raise RuntimeError(
                    f"[{_prof.name}] palm_frame_remap 이 정회전이 아니다 "
                    f"(직교 오차 {_err:.2e}, det {_det:.4f}) — 축이 조용히 뒤집힌다")
            self._palm_remap = _M
        else:
            self._palm_remap = None
        self.robot = Articulation(self.cfg.robot_cfg)
        # ★★씬 등록은 `clone_environments` **이후**로 미룬다 — DEXTRAH 원본 규약.
        #   clone 전에 등록하면 env 가 1개인 상태로 자산 뷰가 잡힐 수 있고,
        #   `replicate_physics=True` 는 cloner 의 physics replication 이 뷰를
        #   갱신해 주지만 **False 에서는 갱신되지 않는다**. 08.29 실측 증상:
        #   리셋의 `write_joint_state_to_sim` 이 안 먹어 관절 편차 18.7 rad ·
        #   속도 2,973 rad/s · `episode_lengths` 260 → 1.2(무한 리셋).
        _sensors: dict = {}

        from openarm.agnostic.modules import object_bank as _ob

        _bank = _ob.get(self.cfg.object_bank)
        _multi = _bank.needs_multi_asset

        # ---- 작업면 --------------------------------------------------------------------
        # ★★단일/다물체로 표현이 갈린다. `UsdFileCfg.rigid_props` 로는 못 바꾼다 —
        #   그 경로는 기존 API 를 **수정만** 하지 적용하지 않아 원본 `env.usd`
        #   (`/Env` Xform + 충돌 Mesh 9개, RigidBodyAPI 없음)를 `RigidObject` 로 감싸면
        #   부팅에서 죽는다(`Failed to find a rigid body when resolving '.../Table'`).
        #
        #   단일 물체 : 원시 정적 프림 + `clone_environments` 복제(기존 동작, 불변).
        #   다물체    : `replicate_physics=False` 라 `enable_env_ids` 격리가 없어져
        #               원시 프림이면 전 env 작업면이 한 충돌 그룹에 남고 팔이 물린다
        #               (08.29 분리 실측 abnormal 0.849 · joint_err 0.74 rad).
        #               → kinematic 사본(`env_rigid.usd`)을 **씬 자산**으로 올린다.
        tbl = self.cfg.table_cfg
        if _multi:
            self.table = RigidObject(tbl)
        else:
            tbl.spawn.func(
                "/World/envs/env_0/Table", tbl.spawn,
                translation=tuple(tbl.init_state.pos),
                orientation=tuple(tbl.init_state.rot),
            )
        # ★테이블은 scene 자산이 아니라 정적 프림이라 EventTerm 이 못 건다. 직접 바인딩한다.
        #   PhysX 결합이 average 라 한쪽만 낮아도 실효 μ 가 중간값이 되고, 컵-테이블
        #   마찰은 접근·안정에 직접 영향을 준다.
        _mu = float(self.cfg.surface_friction)
        _mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=_mu, dynamic_friction=_mu, restitution=0.0)
        _mat.func("/World/Materials/taskSurface", _mat)
        # ★`bind_physics_material` 은 regex 가 아니라 **실제 프림**만 받는다
        #   (`Prim at path '.../env_.*/Table' is not valid` 로 fail-loud). 열거해 건다.
        #   replicate_physics=True 면 이 시점에 env_0 만 있고 clone 이 바인딩까지 복제한다.
        from isaaclab.sim.utils import find_matching_prim_paths

        _tables = find_matching_prim_paths(self.cfg.table_cfg.prim_path)
        if not _tables:
            raise RuntimeError(
                f"[grasp_ua] 테이블 프림이 없다: {self.cfg.table_cfg.prim_path}")
        for _tp in _tables:
            bind_physics_material(_tp, "/World/Materials/taskSurface")

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
                _sensors[f"contact_{finger}_{body}"] = s
            self._finger_sensors[finger] = sensors

        # ★★손바닥 접촉 — 08.27 까지 **계측 자체가 없었다**. 손가락 센서만 있어서 컵이
        #   손바닥에 받쳐 있어도 전혀 보이지 않는다. H1 실측(원위 접촉률 0.01·힘 0.05N)
        #   으로 감쌈 정의를 다시 짜야 하는데, 그 후보에 손바닥이 들어가므로 먼저 잰다.
        self._palm_sensor = ContactSensor(ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{p.palm_body}",
            filter_prim_paths_expr=_filter,
            history_length=1,
            track_air_time=False,
        ))
        _sensors["contact_palm"] = self._palm_sensor

        # 진단 카메라 — env 하나당 하나. probe 가 자세를 눈으로 확인할 때만 켠다.
        if bool(self.cfg.debug_camera):
            from isaaclab.sensors import TiledCamera, TiledCameraCfg
            _sensors["debug_cam"] = TiledCamera(TiledCameraCfg(
                prim_path="/World/envs/env_.*/DebugCam",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=tuple(self.cfg.debug_camera_pos),
                    rot=tuple(self.cfg.debug_camera_rot), convention="world"),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=26.0, focus_distance=0.6,
                    horizontal_aperture=20.955, clipping_range=(0.05, 6.0)),
                width=640, height=480,
            ))

        # env.usd 의 platform 상면이 정확히 z=0 이라 기본 지면과 겹친다 — 지면은 내린다.
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(),
                           translation=(0.0, 0.0, -0.05))
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # ★★clone → **씬 등록** → 물체 생성. DEXTRAH 원본(`GRASP/DEXTRAH`)이 이 순서다:
        #   `Articulation`/`RigidObject` 를 만들고, `clone_environments()` 를 부른 **뒤에**
        #   `scene.articulations[...]`/`scene.rigid_objects[...]` 에 등록한다.
        #   clone 전에 등록하면 env 가 1개인 상태로 자산 뷰가 잡히고,
        #   `replicate_physics=True` 는 cloner 의 physics replication 이 뷰를 갱신해 주지만
        #   **False 에서는 갱신되지 않는다** — 08.29 실측: 리셋의 `write_joint_state_to_sim`
        #   이 반영 안 돼 관절 편차 18.7 rad · 속도 2,973 rad/s · `episode_lengths` 1.2.
        # ★다물체는 `InteractiveScene.__init__` 이 이미 env xform 을 복제했으므로
        #   여기서 다시 부르지 않는다(부르면 env_0 내용을 전 env 에 덮어쓴다).
        # ★★`clone_environments` 는 **`replicate_physics=True` 일 때만** 부른다.
        #   False 면 `InteractiveScene.__init__` 이 이미 env xform 을 복제했고, 여기서
        #   또 부르면 env_0 내용을 전 env 에 덮어써 프림이 중복·변형된다. 그것이
        #   리셋 직후 관절 폭발(편차 18~28 rad · 속도 2,500~4,700 rad/s ·
        #   `episode_lengths` 260→1.2)의 남은 유일한 후보다.
        #   ★`filter_collisions` 는 **양쪽 다** 부른다 — True 경로는 clone 의
        #   `enable_env_ids` 가 격리해 주지만 False 경로는 이 호출이 유일한 격리다.
        _replicate = bool(self.cfg.scene.replicate_physics)
        if not _multi:
            # 단일 물체는 clone 이 복제해 줘야 하므로 **clone 전에** 만든다.
            _ob.assert_spawned_after_clone(_bank, cloned=not _replicate)
            self.object = RigidObject(self.cfg.object_cfg)
        if _replicate:
            self.scene.clone_environments(copy_from_source=True)
        if _multi:
            # 다물체는 전 env prim 이 존재해야 `env_id % N` 배정이 성립한다.
            _ob.assert_spawned_after_clone(_bank, cloned=True)
            self.object = RigidObject(self.cfg.object_cfg)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # ---- 씬 등록은 clone **이후** (위 주석의 DEXTRAH 규약) --------------------------
        self.scene.articulations["robot"] = self.robot
        for _k, _v in _sensors.items():
            self.scene.sensors[_k] = _v
        if _multi:
            self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["object"] = self.object
        print(f"[grasp_ua] 물체 뱅크 '{_bank.name}' {len(_bank)}종 · "
              f"replicate_physics={self.cfg.scene.replicate_physics} · "
              f"센서 {len(_sensors)}개 등록", flush=True)

    # ------------------------------------------------------------------
    # Fabrics — 팔은 절대 palm pose attractor 로만 움직인다
    # ------------------------------------------------------------------
    def _build_fabric_world(self) -> dict | None:
        """fabric 장애물 세계 — 테이블 박스 1개. 없으면 None(빈 세계).

        ★★08.27 발견: 여기에 아무것도 안 넘겨서 `object_indicator == 0` 이었고,
          반발 커널이 첫 줄에서 early-out 했다. **fabric 이 테이블을 아예 모르는 상태**로
          계획하고 있었다(사용자 GUI: "아예 테이블을 박히고 간다").
          형제 tesollo 트랙은 전부 `world_filename` 을 넘긴다 — agnostic 트랙만 빠졌었다.
        ★params 의 `body_repulsion.collision_sphere_frames` 에 palm·5지 전 마디
          (소지 `dg_5` 14개 포함)·팔 링크 충돌구가 **이미** 등록돼 있다. 테이블 하나만
          넣으면 손 전체가 한꺼번에 보호되므로 params 는 건드리지 않는다.
        ★박스 크기는 palm 도달영역에서 **파생**한다 — 숫자를 따로 적으면 물리 테이블과
          조용히 어긋난다. 상면 z 는 `table_surface_z` 그 자체다.
        """
        if not bool(self.cfg.fabric_table_obstacle):
            print("[grasp_ua] ⚠fabric 테이블 장애물 OFF — fabric 이 테이블을 모른다",
                  flush=True)
            return None
        p = self.profile
        _lo, _hi = p.palm_box_min, p.palm_box_max
        _m = float(self.cfg.fabric_table_margin_xy)
        _sx = (_hi[0] - _lo[0]) + 2.0 * _m
        _sy = (_hi[1] - _lo[1]) + 2.0 * _m
        _cx = 0.5 * (_lo[0] + _hi[0])
        _cy = 0.5 * (_lo[1] + _hi[1])
        _th = float(self.cfg.fabric_table_thickness)
        _cz = float(self.cfg.table_surface_z) - 0.5 * _th
        print(f"[grasp_ua] fabric 테이블 장애물: 크기 {_sx:.3f}×{_sy:.3f}×{_th:.3f} "
              f"· 중심 ({_cx:.3f}, {_cy:.3f}, {_cz:.3f}) · 상면 z "
              f"{self.cfg.table_surface_z:.3f}", flush=True)
        return {"table": {"env_index": "all", "type": "box",
                          "scaling": f"{_sx} {_sy} {_th}",
                          "transform": f"{_cx} {_cy} {_cz} 0. 0. 0. 1."}}

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
            world_dict=self._build_fabric_world(),
        )
        self._world_ids, self._world_indicator = self._world.get_object_ids()

        _cls = _fabric_class(p.fabric_class)
        _kw = dict(
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
        _kw.update(dict(p.fabric_kwargs))
        self.fabric = _cls(**_filter_fabric_kwargs(_cls, _kw, p))
        self.integrator = DisplacementIntegrator(self.fabric)

        # ★★fabric cspace 가 이 팔만인지(tesollo 27) 양팔 전체인지(RH56F1 26) 는
        #   프로필이 정한다. 슬롯이 비면 구 규약 [0:n_arm] / [n_arm:n_arm+n_hand].
        _n_a, _n_h = p.num_arm_joints, p.num_hand_joints
        _arm_slot = tuple(p.fabric_arm_slot) or tuple(range(_n_a))
        _hand_slot = tuple(p.fabric_hand_slot) or tuple(range(_n_a, _n_a + _n_h))
        if len(_arm_slot) != _n_a or len(_hand_slot) != _n_h:
            raise RuntimeError(
                f"[{p.name}] fabric 슬롯 길이 불일치: 팔 {len(_arm_slot)}!={_n_a} · "
                f"손 {len(_hand_slot)}!={_n_h}")
        _ncs = self.fabric.num_joints
        if not p.fabric_arm_slot and not p.fabric_hand_slot:
            # 구 규약 프로필 — cspace 가 정확히 팔+손이어야 한다.
            if _ncs != _n_a + _n_h:
                raise RuntimeError(
                    f"[{p.name}] fabric num_joints={_ncs} != 프로필 {_n_a + _n_h}. "
                    "fabric URDF 와 USD 자산이 어긋났다. (양팔 cspace 라면 "
                    "fabric_arm_slot/fabric_hand_slot 을 채워라)")
        _bad = [i for i in _arm_slot + _hand_slot if not 0 <= i < _ncs]
        if _bad:
            raise RuntimeError(
                f"[{p.name}] fabric 슬롯 인덱스가 cspace({_ncs}) 밖이다: {_bad}")
        self._fab_t = self._build_fabric_index()
        self._fab_arm_t = torch.tensor(_arm_slot, device=self.device, dtype=torch.long)
        self._fab_hand_t = torch.tensor(_hand_slot, device=self.device, dtype=torch.long)

        # synergy 자세(프로필 finger-major) → fabric 손 구간 순서. 이름 기반 매핑 유지.
        _syn_pos = {int(j): k for k, j in enumerate(self._syn_ids)}
        _fab_hand = self._fab_t[self._fab_hand_t].tolist()
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
            print(f"[grasp_ua] ⚠ palm_box 미검증({p.name}) — 도달성 확인 후 승격할 것",
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
        # ★★09.02 2 → 24 스텝. 게인이 부드러우면(RH56F1 실기 벤더 kp 10~70) 2 스텝으로는
        #   홈에 정착하지 않아 손끝 FK 산포가 게이트(2mm)를 넘는다(실측 2.6mm).
        #   단단한 게인에서는 어차피 즉시 정착하므로 tesollo 거동은 안 바뀐다.
        for _ in range(24):
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
        print(f"[grasp_ua] fabric→env 오프셋 = "
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
        print(f"[grasp_ua] 홈 palm={[round(v, 4) for v in home.tolist()]} | "
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
        arm_target = self.fabric_q[:, self._fab_arm_t]
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_ids)
        # ★속도 피드포워드. 0 을 넣으면 implicit PD 의 감쇠항 kd·(0 − q̇) 이 참조
        #   궤적의 움직임을 반대로 밀어 err ≈ (kd/kp)·q̇ 의 상시 지연이 생긴다.
        self.robot.set_joint_velocity_target(
            float(self.cfg.fabric_velocity_ff_scale)
            * self.fabric_qd[:, self._fab_arm_t],
            joint_ids=self.arm_ids)
        # 손은 fabric 밖 — 이름으로 찾은 인덱스에 관절 목표를 직접 준다.
        self.robot.set_joint_position_target(self._syn_target, joint_ids=self._syn_ids)
        self.robot.set_joint_velocity_target(
            float(self.cfg.hand_velocity_ff_scale) * self._syn_vel,
            joint_ids=self._syn_ids)
        self._apply_mimic_targets()

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
        self._apply_pose_knobs(p)

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
        # ---- per_finger 레이아웃 (08.29 O 라운드) — 관절 → 전역 액션 슬롯(-1=고정) ----
        if str(self.cfg.hand_layout) == "per_finger":
            _map = p.hand_finger_channels
            _act = []
            for k, nm in enumerate(p.hand_joint_names):
                _f = fingers[fi[k]]
                _s = nm.rsplit("_", 1)[1]
                _act.append(int(_map.get(_f, {}).get(_s, -1)))
            self._syn_act = torch.tensor(_act, device=self.device, dtype=torch.long)
            _n_act = int(self._syn_act.max().item()) + 1
            if 6 + _n_act != int(self.cfg.action_space):
                raise RuntimeError(
                    f"[{p.name}] per_finger 슬롯 {_n_act} ≠ action_space-6 "
                    f"{int(self.cfg.action_space) - 6}")
        # 손가락 단위 동결용 — 굴곡 관절(_2/_3/_4) 마스크.
        self._syn_flex = torch.tensor(
            [nm.rsplit("_", 1)[1] in ("2", "3", "4") for nm in p.hand_joint_names],
            device=self.device)
        self._syn_freeze = torch.tensor(
            [nm.rsplit("_", 1)[1] in p.hand_freeze_suffixes for nm in p.hand_joint_names],
            device=self.device)
        # ★★동결은 **관절별로 자기 링크가 닿았을 때** 걸어야 한다.
        #   구판은 (원위|팁) 접촉 하나로 `_3`·`_4` 를 통째로 얼렸는데, `_2` 가 굽으면
        #   손끝이 가장 먼저 닿으므로 **감쌈이 시작되기 직전에 감쌈 관절을 잠갔다** —
        #   08.27 실측: wrap_frac 이 전 런에서 정확히 0.000, syn_close 0.278 ≈
        #   "채널1(`_2`)만 폐쇄" 예측 0.250.
        #   `_3` → 중간마디 접촉 / `_4` → 원위 또는 팁(팁은 원위 링크에 고정).
        #   ★09.02 `"3"` 리터럴 축출 — 접미사는 **로봇 사실**이라 프로필이 들고 있다.
        #     tesollo 는 `("3",)` 로 구 거동과 동일하고, 손가락당 구동관절이 하나뿐인
        #     손(RH56F1)은 비워 두면 전부 원위로 취급된다. 리터럴로 두면 그런 손에서
        #     중간마디 동결이 **아무 관절에도 안 걸린 채 조용히 꺼진다**.
        _sfx = [nm.rsplit("_", 1)[1] for nm in p.hand_joint_names]
        _mid_sfx = tuple(p.hand_freeze_mid_suffixes)
        _unknown = set(_mid_sfx) - set(p.hand_freeze_suffixes)
        if _unknown:
            raise RuntimeError(
                f"[{p.name}] hand_freeze_mid_suffixes {sorted(_unknown)} 가 "
                f"hand_freeze_suffixes {p.hand_freeze_suffixes} 에 없다 — "
                "중간마디는 동결 대상의 부분집합이어야 한다")
        self._syn_freeze_mid = torch.tensor(
            [s in p.hand_freeze_suffixes and s in _mid_sfx for s in _sfx],
            device=self.device)
        self._syn_freeze_dist = torch.tensor(
            [s in p.hand_freeze_suffixes and s not in _mid_sfx for s in _sfx],
            device=self.device)
        # ★가동 관절 마스크 — open == grip 인 관절은 명령해도 안 움직인다
        #   (실측: r_hj_pinky_2 · r_hj_thumb_2 · 전 `_1` 이 가동폭 0°). 폐쇄 보상의
        #   분모에 넣으면 "못 움직이는 관절을 닫았다"는 공짜 점수가 생긴다.
        self._syn_movable = (self._syn_grip - self._syn_open).abs() > 1e-4
        # ★09.02 hot path 동기화 제거 — `_get_rewards` 가 매 스텝
        #   `bool((~_mv).any())` 로 물어 GPU 를 동기화하던 것을 부팅에서 확정한다.
        self._syn_has_fixed = bool((~self._syn_movable).any())
        if not bool(self._syn_movable.any()):
            raise RuntimeError(f"[{p.name}] 가동 손관절이 하나도 없다 — open/grip 자세 확인")
        # 폐쇄도는 **관절별** 독립 진행도다 — 접촉 동결이 관절마다 따로 걸린다.
        self._syn_close = torch.zeros(self.num_envs, n, device=self.device)
        self._syn_target = self.robot.data.joint_pos[:, self._syn_ids].clone()
        self._syn_vel = torch.zeros(self.num_envs, n, device=self.device)
        _lim = self.robot.data.soft_joint_pos_limits[0, self._syn_ids, :]
        self._syn_lo, self._syn_hi = _lim[:, 0].contiguous(), _lim[:, 1].contiguous()
        _grip_clamped = self._syn_grip.clamp(self._syn_lo, self._syn_hi)
        # ★★09.02 신설 — "지령상 움직이는데 한계 때문에 실제로는 못 움직이는" 관절을
        #   부팅에서 죽인다. `_syn_movable` 은 **clamp 전** 값으로 판정하므로 이런
        #   관절을 가동으로 세고, 그러면 폐쇄도 분모에 영원히 0 인 항이 남는다.
        #   실측 계기(RH56F1): `oppose_grip_delta_rad=-0.6` 이 tesollo 에선 엄지
        #   **대향** 관절을 가리키는데 RH56F1 에선 엄지 **굴곡**(한계 [0, 0.475])을
        #   가리켜 grip 이 −0.6 → 0 으로 잘렸다. 엄지가 300스텝 내내 안 움직였고
        #   지표 어디에도 그 이유가 안 나왔다.
        _open_clamped = self._syn_open.clamp(self._syn_lo, self._syn_hi)
        _dead = (((self._syn_grip - self._syn_open).abs() > 1e-4)
                 & ((_grip_clamped - _open_clamped).abs() <= 1e-4))
        if bool(_dead.any()):
            _names = [p.hand_joint_names[i] for i in _dead.nonzero().flatten().tolist()]
            raise RuntimeError(
                f"[{p.name}] 관절 {_names} 의 open/grip 이 관절한계로 같아졌다 — "
                "지령은 닫으라는데 물리적으로 못 움직인다(폐쇄도 분모에 죽은 항이 남는다). "
                "hand_grip_pose 나 자세 노브(oppose_grip_delta_rad 등)를 로봇에 맞춰라.")
        print(f"[grasp_ua] synergy: 관절 {n}개 · 채널 {self._syn_nch} · "
              f"동결 {int(self._syn_freeze.sum())}개 · "
              f"grip 한계clamp {int((self._syn_grip != _grip_clamped).sum())}개", flush=True)
        self._setup_mimic()

    # ------------------------------------------------------------------
    # 언더액추에이션(mimic) — 종속관절 목표
    # ------------------------------------------------------------------
    def _setup_mimic(self) -> None:
        """`hand_mimic` 표를 구동관절까지 전개해 인덱스/계수를 굳힌다.

        ★왜 태스크가 이걸 아는가: 손의 물리 자유도가 제어 자유도보다 많은 손(RH56F1)은
          종속관절이 **어떤 값이어야 하는지**가 로봇의 기구학이지 정책의 선택이 아니다.
          USD 에 PhysxMimicJoint 가 없으면 종속관절은 지령을 못 받아 init 각도에 굳고,
          sim 의 손가락이 실기와 다른 강체 막대가 된다 — s2r 에서 치명적이다.
        """
        p = self.profile
        mode = str(p.hand_mimic_mode)
        if mode not in ("none", "software", "physx"):
            raise RuntimeError(
                f"[{p.name}] hand_mimic_mode='{mode}' 는 없는 값이다 "
                "(none | software | physx)")
        self._mim_mode = mode
        self._mim_dep_ids = None
        if mode == "none":
            if p.hand_mimic:
                raise RuntimeError(
                    f"[{p.name}] hand_mimic 표가 있는데 mode=none 이다 — "
                    "표를 지우거나 mode 를 software/physx 로 하라")
            return
        if not p.hand_mimic:
            raise RuntimeError(
                f"[{p.name}] hand_mimic_mode='{mode}' 인데 hand_mimic 표가 비었다")

        jn = list(self.robot.data.joint_names)
        drive_pos = {nm: k for k, nm in enumerate(p.hand_joint_names)}
        dep_ids, src_local, mults, offs, rows = [], [], [], [], []
        for dep in p.hand_mimic:
            if dep in drive_pos:
                raise RuntimeError(
                    f"[{p.name}] '{dep}' 이 구동관절(hand_joint_names)이면서 "
                    "종속관절이다 — 정책과 결합이 같은 관절을 두고 싸운다")
            # 체인 전개: dep = mult·cur + off 를 유지하며 구동관절까지 내려간다.
            cur, mult, off, seen = dep, 1.0, 0.0, [dep]
            while cur in p.hand_mimic:
                src, m, o = p.hand_mimic[cur]
                off = mult * float(o) + off
                mult = mult * float(m)
                cur = src
                if cur in seen:
                    raise RuntimeError(
                        f"[{p.name}] hand_mimic 이 순환한다: {seen + [cur]}")
                seen.append(cur)
            if cur not in drive_pos:
                raise RuntimeError(
                    f"[{p.name}] '{dep}' 의 뿌리 '{cur}' 가 구동관절이 아니다 — "
                    f"hand_joint_names {p.hand_joint_names} 를 확인하라")
            if dep not in jn:
                raise RuntimeError(f"[{p.name}] mimic 종속관절 '{dep}' 이 자산에 없다")
            dep_ids.append(jn.index(dep))
            src_local.append(drive_pos[cur])
            mults.append(mult)
            offs.append(off)
            rows.append(f"{dep} = {mult:.4f}·{cur}" + (f" + {off:.4f}" if off else ""))

        self._mim_dep_ids = dep_ids
        self._mim_src_local = torch.tensor(src_local, device=self.device, dtype=torch.long)
        self._mim_mult = torch.tensor(mults, device=self.device, dtype=torch.float32)
        self._mim_off = torch.tensor(offs, device=self.device, dtype=torch.float32)
        print(f"[grasp_ua] mimic({mode}) {len(dep_ids)}개: " + " · ".join(rows), flush=True)
        if mode == "physx":
            print("[grasp_ua] ⚠mimic=physx — 이 코드는 종속관절을 **건드리지 않는다**. "
                  "USD 의 PhysxMimicJoint 가 실제로 결합하는지 반드시 probe 로 확인하라 "
                  "(구동관절만 굽혀 종속관절이 배율대로 따라오는지). 결합이 없으면 "
                  "손가락이 init 각도에 굳은 채 조용히 학습된다.", flush=True)

    def _mimic_tracking_err(self) -> torch.Tensor:
        """종속관절이 결합식에서 벗어난 양 (N,) rad — **모드 무관** 진단.

        ★physx 모드의 유일한 감시 수단이다. USD 의 PhysxMimicJoint 가 실제로 안 걸려
          있으면 종속관절이 init 각도에 굳는데, 그 상태는 접촉·보상 어느 지표에도
          안 나타난다 — 여기서만 보인다(구동관절이 굽을수록 값이 커진다).
        """
        if self._mim_dep_ids is None:
            return torch.zeros(self.num_envs, device=self.device)
        _q = self.robot.data.joint_pos
        _dep = _q[:, self._mim_dep_ids]
        _src = _q[:, self._syn_ids][:, self._mim_src_local]
        return (_dep - (_src * self._mim_mult + self._mim_off)).abs().max(dim=1).values

    def _apply_mimic_targets(self) -> None:
        """구동 목표에서 종속 목표를 만들어 써넣는다 — `software` 모드 전용.

        ★`_syn_target` 은 관절한계로 clamp 된 **구동** 목표다. 종속목표를 여기서
          파생해야 결합비가 지령 수준에서 유지된다(실측치에서 만들면 추종 지연이
          결합비에 섞인다).
        """
        if self._mim_dep_ids is None or self._mim_mode != "software":
            return
        _src = self._syn_target[:, self._mim_src_local]
        self.robot.set_joint_position_target(
            _src * self._mim_mult + self._mim_off, joint_ids=self._mim_dep_ids)
        self.robot.set_joint_velocity_target(
            float(self.cfg.hand_velocity_ff_scale)
            * self._syn_vel[:, self._mim_src_local] * self._mim_mult,
            joint_ids=self._mim_dep_ids)

    def _synergy_targets(self, a_hand: torch.Tensor) -> torch.Tensor:
        """액션(손가락×채널) → 관절 목표. 프로필 순서 (N, n).

        액션은 **절대 폐쇄도 목표**이고 `synergy_close_speed` 는 그 목표를 향한
        변화율 상한이다(속도 명령이 아니다 — 속도로 두면 탐색 노이즈 평균만으로
        완전 폐쇄되고 되돌릴 수 없다).
        """
        p = self.profile
        nf = len(p.finger_sensor_bodies)
        if str(self.cfg.hand_layout) == "per_finger":
            # 손가락별 슬롯(엄지 2·검/중/약 각 1·소지 1). 미지정 관절은 폐쇄도 0 고정.
            cmd_flat = 0.5 * (a_hand.clamp(-1.0, 1.0) + 1.0)      # (N, n_act)
            cmd_j = torch.where(
                (self._syn_act >= 0).unsqueeze(0),
                cmd_flat[:, self._syn_act.clamp(min=0)],
                torch.zeros(self.num_envs, len(self._syn_act),
                            device=self.device))
        else:
            a = a_hand.view(self.num_envs, nf, self._syn_nch)
            if bool(self.cfg.couple_four_fingers):
                # 대향 그룹(엄지)만 독립, 나머지는 채널별 평균 — "특정 손가락만 안 닫힘"을
                # 액션 공간에서 제거한다. 접촉 동결은 관절별로 남아 형상 적응은 유지된다.
                _mask = torch.ones(nf, dtype=torch.bool, device=a.device)
                _mask[self._group_a_idx] = False
                _common = a[:, _mask, :].mean(dim=1, keepdim=True).expand(-1, nf, -1)
                # ★공통 + 잔차 — scale 0 이면 평균 대체(구 coupled 항등), 1 이면 개별 지령
                #   그대로(15ch 와 동일). 그 사이가 연속이라 ADR 축으로 열 수 있다.
                _rs = float(getattr(self, "_adr_residual",
                                    self.cfg.finger_residual_scale))
                _blend = _common if _rs == 0.0 else _common + _rs * (a - _common)
                a = torch.where(_mask.view(1, nf, 1), _blend, a)
            cmd = 0.5 * (a.clamp(-1.0, 1.0) + 1.0)                # 절대 폐쇄도 [0,1]
            cmd_j = cmd[:, self._syn_fi, self._syn_ch]            # (N, n) 관절 전개
        rate = float(self.cfg.synergy_close_speed)
        delta = (cmd_j - self._syn_close).clamp(-rate, rate)
        # ★닫는 방향만 정렬 게이트로 스케일한다 — **푸는 방향은 항상 허용**해야
        #   잘못 오므린 상태에서 빠져나올 수 있다.
        _g = self._close_gate.unsqueeze(1)
        delta = torch.where(delta > 0.0, delta * _g, delta)
        if str(self.cfg.synergy_hold_mode) == "blocked":
            # ★★"막힐 때까지 만다" — 접촉 센서를 안 쓰므로 **형상에 무관**하다. 관절이
            #   목표를 못 따라가면(토크 포화) 외부에 막힌 것이고, 그때만 전진을 멈춘다.
            #   닿자마자 멈추는 구판은 감쌈이 시작되기 **전에** 손가락을 세운다.
            _blk = torch.zeros_like(delta, dtype=torch.bool)
            _blk[:, self._syn_movable] = self._hand_blocked()
            delta = torch.where(_blk & (delta > 0.0), torch.zeros_like(delta), delta)
            # ★여는 방향 래칫 차단 — 구판은 닫기만 막고 열기는 통과시켜, 탐색 노이즈만으로
            #   동결된 관절이 완전 개방까지 풀렸다(엄지 `_3` 0.36 → 0.00 단조, 08.27 실측).
            _rdb = float(self.cfg.synergy_release_deadband)
            if _rdb > 0.0:
                delta = torch.where(_blk & (delta < 0.0) & (delta > -_rdb),
                                    torch.zeros_like(delta), delta)
        elif bool(self.cfg.synergy_contact_freeze):
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
            if str(self.cfg.synergy_freeze_scope) == "finger":
                # ★손가락 단위 — (중간∨원위) 접촉이면 그 손가락 굴곡관절 **전부** 정지.
                #   관절별 동결은 언 손끝을 매단 채 근위(_2)가 계속 감겨 큰 컵을
                #   밀어냈다(08.29 s130 영상 + M1 잔여 실패 진단).
                _hold = (_h_mid | _h_dist) & self._syn_flex
            else:
                _hold = ((_h_mid & self._syn_freeze_mid)
                         | (_h_dist & self._syn_freeze_dist))
            # ★닫는 방향만 얼린다 — 푸는 방향까지 막으면 갇혀서 빠져나올 수 없다
            #   (닫기 게이트와 같은 원칙).
            delta = torch.where(_hold & (delta > 0.0), torch.zeros_like(delta), delta)
        # ★★★09.02 닫기 게이트를 **상한**으로도 쓴다(플래그, 기본 꺼짐).
        #   구 코드에서 게이트는 증가율만 스케일했다 — 속도 제한이지 상한이 아니다.
        #   `synergy_close_speed=0.005` × 에피소드 600스텝 = 3.0 이라, 게이트가 g 여도
        #   `3.0·g` 까지 쌓인다: **g > 0.33 이면 정렬과 무관하게 완전히 닫힌다.**
        #   실측(rh_b1 it150): 게이트 0.070 인데 `syn_close` 가 이미 0.060 이었고,
        #   그 상태로 컵을 스치면 래치(엄지+아무 손가락 8스텝)가 걸린다. 래치 후엔
        #   게이트가 강제로 1 이 되어 그 뒤로는 정렬이 아예 불필요해진다 —
        #   "멀리서 쥐고 스쳐서 래치" 가 보상 최적이 되는 고리다.
        #   상한으로 쓰면 정렬 전에는 손이 **펴진 채로 유지**된다(사용자 지적:
        #   "손바닥에 충분히 가까워지기 전까지 손가락을 최대한 피고 있어야 한다").
        #   ★래치 후에는 게이트가 1 이라 상한이 사라진다 — 이송 중 파지 유지에 무영향.
        _hi = torch.ones_like(self._syn_close)
        if bool(getattr(self.cfg, "close_gate_as_ceiling", False)):
            _hi = self._close_gate.unsqueeze(1).expand_as(self._syn_close)
        self._syn_close = (self._syn_close + delta).clamp(0.0, 1.0).minimum(_hi)
        tgt = torch.lerp(self._syn_open.unsqueeze(0), self._syn_grip.unsqueeze(0),
                         self._syn_close)
        return tgt.clamp(self._syn_lo.unsqueeze(0), self._syn_hi.unsqueeze(0))

    def _banded_dist(self, delta: torch.Tensor) -> torch.Tensor:
        """z 데드밴드를 넣은 거리 (N,). `delta` = (N,3) 차이벡터.

        ★★3D 노름은 z 오차를 xy 오차와 **똑같이** 벌한다. 파지 높이는 원래 여유가 있는
          축인데 그 여유가 없어서 palm 이 파지높이 아래로 눌려 내려갔다(08.27 실측:
          palm_above_table mean 0.088 vs 파지중심 0.107). 밴드 안에서는 z 를 0 으로 본다.
        """
        _b = float(self.cfg.grasp_z_deadband)
        _dz = torch.relu(delta[:, 2].abs() - _b)
        return torch.sqrt(delta[:, :2].pow(2).sum(dim=-1) + _dz.pow(2))

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

    def _apply_pose_knobs(self, p) -> None:
        """자세표 실험 노브 — 기본값이면 아무것도 바꾸지 않는다(현행 거동 보존).

        ★프로필의 `hand_open_pose` 와 `hand_grip_pose` 가 **같은 값**인 관절은 시너지가
          아예 안 건드린다. 그건 "물리적으로 못 움직이는 관절"이 아니다 — URDF 실측으로
          `r_hj_thumb_2` 는 가동범위 180° 로 손에서 가장 큰데 −1.57 에 고정돼 있었다.
        ★손가락 이름은 프로필 구조(`contact_group_a`·`hand_channel_of_joint`)에서 끌어와
          로봇 리터럴을 코드에 넣지 않는다.
        """
        def _ch_of(nm: str) -> int | None:
            return p.hand_channel_of_joint.get(nm.rsplit("_", 1)[1])

        _d = float(self.cfg.oppose_grip_delta_rad)
        if _d != 0.0:
            _idx = [i for i, nm in enumerate(p.hand_joint_names)
                    if _ch_of(nm) == 1 and any(f"_{f}_" in nm for f in p.contact_group_a)]
            if not _idx:
                raise RuntimeError(
                    f"[{p.name}] 대향 손가락{p.contact_group_a} 의 ch1 관절을 못 찾았다 "
                    "— oppose_grip_delta_rad 가 조용히 무효가 된다")
            for i in _idx:
                self._syn_grip[i] = self._syn_open[i] + _d
            print(f"[grasp_ua] 대향 관절 {len(_idx)}개 grip = open{_d:+.3f} rad", flush=True)

        _wf, _ws = str(self.cfg.weak_finger), float(self.cfg.weak_finger_curl_scale)
        if _wf and _ws != 1.0:
            _idx = [i for i, nm in enumerate(p.hand_joint_names)
                    if f"_{_wf}_" in nm and _ch_of(nm) == 2]
            if not _idx:
                raise RuntimeError(f"[{p.name}] 손가락 '{_wf}' 의 ch2 관절을 못 찾았다")
            for i in _idx:
                self._syn_grip[i] = self._syn_open[i] + _ws * (
                    self._syn_grip[i] - self._syn_open[i])
            print(f"[grasp_ua] '{_wf}' 굴곡 grip ×{_ws:.2f} ({len(_idx)}개)", flush=True)

    def _hand_blocked(self) -> torch.Tensor:
        """가동 관절이 **외부에 막혀** 있는가 (N, n_movable) bool — 진단 전용.

        판별: 목표를 못 따라가는데(`|target−q| > blocked_err_thr_rad`) 자기 한계에서는
        떨어져 있다(`q` 가 `[lo+eps, hi−eps]` 안). 이 두 조건을 함께 봐야 **한계에 부딪힌
        것**과 **물체에 막힌 것**이 갈린다 — `hand_grip_pose` 가 soft limit 을 넘겨
        과지령이라(1.8 rad vs 1.571) 완전 폐쇄만으로 모든 관절이 "더 못 조임" 상태가
        되기 때문이다. 허공에서 주먹을 쥐어도 오차 조건 하나는 성립한다.
        ★가동폭 0° 관절(전 `_1` + `pinky_2` + `thumb_2`)은 항상 오차 상태라 제외한다.
        """
        _q = self.robot.data.joint_pos[:, self._syn_ids]
        _eps = float(self.cfg.blocked_limit_eps_rad)
        _free = ((_q > self._syn_lo.unsqueeze(0) + _eps)
                 & (_q < self._syn_hi.unsqueeze(0) - _eps))
        _stuck = (self._syn_target - _q).abs() > float(self.cfg.blocked_err_thr_rad)
        return (_stuck & _free)[:, self._syn_movable]

    def _syn_to_fab(self, syn_q: torch.Tensor) -> torch.Tensor:
        """synergy 자세(프로필 순서) → fabric 손 구간 순서."""
        return syn_q[:, self._syn_to_fab_idx]

    # ------------------------------------------------------------------
    # 접촉 · 좌표 헬퍼
    # ------------------------------------------------------------------
    def _contact_step_reset(self) -> None:
        """스텝 캐시 무효화 — `_pre_physics_step` 이 매 스텝 한 번 부른다.

        ★09.02 계측: 접촉 센서 16개를 한 스텝에 **78회** 읽고 있었다(필요한 건 16회).
          `_contact_forces` 3회 · `_palm_contact_force` 3회 · `_tip_contact_forces` 2회
          · `_finger_link_forces` 필터/무필터 2패스. 읽기마다 view→sum→norm 3커널이라
          스텝당 ~230 커널이 접촉 읽기에만 쓰였다. 물리는 스텝 안에서 안 바뀌므로 같은
          값이다 — 캐시로 약 1/5 로 줄인다(거동 완전 동일).
        """
        self._cf_cache = {}

    def _cached(self, key: str, fn):
        _c = getattr(self, "_cf_cache", None)
        if _c is None:
            _c = self._cf_cache = {}
        if key not in _c:
            _c[key] = fn()
        return _c[key]

    def _contact_forces(self) -> torch.Tensor:
        """손가락별 물체 접촉력 크기 (N, F). body 별 센서 합산, Object-필터. 스텝 캐시."""
        return self._cached("cf", self._contact_forces_raw)

    def _contact_forces_raw(self) -> torch.Tensor:
        mags = []
        for finger in self._finger_names:
            total = torch.zeros(self.num_envs, device=self.device)
            for s in self._finger_sensors[finger]:
                fm = s.data.force_matrix_w                       # (N, B, M, 3)
                total = total + fm.view(self.num_envs, -1, 3).sum(dim=1).norm(dim=-1)
            mags.append(total)
        return torch.stack(mags, dim=1)

    def _mag_filtered(self, sensor) -> torch.Tensor:
        """컵-필터 접촉력 크기 (N,). 보상·게이트가 쓰는 정규 경로."""
        return sensor.data.force_matrix_w.view(
            self.num_envs, -1, 3).sum(dim=1).norm(dim=-1)

    def _mag_net(self, sensor) -> torch.Tensor:
        """**필터 없는** 총 접촉력 크기 (N,) — 진단 전용.

        ★`force_matrix_w` 는 `filter_prim_paths_expr`(컵 baseLink)에 걸린 접촉만 담고
          `net_forces_w` 는 그 링크가 받은 **모든** 접촉을 담는다. 둘을 나란히 읽으면
          "링크가 안 닿았다"와 "닿았는데 필터가 못 잡는다"가 갈린다. 08.27 실측에서
          원위(`_4`)만 다섯 손가락 전부·4,553 기록점 내내 정확히 0.000 이었다.
        """
        return sensor.data.net_forces_w.view(
            self.num_envs, -1, 3).sum(dim=1).norm(dim=-1)

    def _finger_link_forces(self, mag) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """손가락별 (중간, 원위, 팁) 접촉력 (N, F). `mag` 가 필터/무필터를 가른다.

        `finger_sensor_bodies` 규약: 마지막 원소 = 팁, 그 앞이 (중간, 원위) 순.
        body 가 하나뿐인 손가락은 그 접촉 자체가 감쌈이다(mid=dist=그 body).
        """
        mids, dists, tips = [], [], []
        for finger in self._finger_names:
            sensors = self._finger_sensors[finger]
            mid_i, dist_i = (0, 1) if len(sensors) >= 3 else (0, 0)
            mids.append(mag(sensors[mid_i]))
            dists.append(mag(sensors[dist_i]))
            tips.append(mag(sensors[-1]))
        return (torch.stack(mids, dim=1), torch.stack(dists, dim=1),
                torch.stack(tips, dim=1))

    def _palm_contact_force(self) -> torch.Tensor:
        """손바닥이 물체에서 받는 접촉력 크기 (N,) — 진단 전용.

        ★감쌈 정의의 후보다. 08.27 H1 실측에서 원위 링크 접촉이 사실상 0(힘 0.053N)인데
          손가락은 컵에 막혀 있었다 — 컵을 실제로 받치는 면이 어디인지 재려면 손바닥을
          빼놓을 수 없는데, 이 트랙은 그동안 손바닥 센서 자체가 없었다.
        """
        return self._cached("palm", lambda: self._mag_filtered(self._palm_sensor))

    def _link_forces_filtered(self):
        """(중간, 원위, 팁) 필터 접촉력 — 스텝 캐시(구 코드는 2회 중복 계산했다)."""
        return self._cached("lf", lambda: self._finger_link_forces(self._mag_filtered))

    def _contact_forces_split(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(중간, 원위) 마디별 접촉력 (N, F) — 감쌈 판정용."""
        _mid, _dist, _ = self._link_forces_filtered()
        return _mid, _dist

    def _tip_contact_forces(self) -> torch.Tensor:
        """손가락별 **팁만** 접촉력 (N, F)."""
        return self._link_forces_filtered()[2]

    def _env_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return pos_w - self.scene.env_origins

    def _palm_ee_R(self) -> torch.Tensor:
        """palm 회전행렬 (N,3,3) — **태스크 표준축**. 열0 = 손바닥 법선, 열1 = 가로,
        열2 = 손가락 방향.

        ★`palm_body` 는 자유롭게 고를 수 없다 — fabric 제어점과 같아야 부팅 FK 게이트를
          통과한다. 그 프레임의 축 배치가 로봇마다 다르므로(RH56F1 `r_hl_palm_sensor` 는
          법선이 국소 +z) `palm_frame_remap` 으로 여기서 한 번만 흡수한다. 안 그러면
          보상의 `palm_normal_dist`/`palm_lateral_dist` 가 축이 바뀐 채 조용히 학습된다.
        """
        _R = matrix_from_quat(self.robot.data.body_quat_w[:, self.palm_idx])
        if self._palm_remap is None:
            return _R
        return _R @ self._palm_remap

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
            prim_path="/Visuals/GraspUACmd",
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
        print(f"[grasp_ua] 지령 마커 ON — env0 전용 · 축 {_L * 1000:.0f}mm · "
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
