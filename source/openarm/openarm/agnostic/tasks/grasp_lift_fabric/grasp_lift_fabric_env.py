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
# ★08.23 보상을 자매 트랙 grasp_sensor 와 **공유**한다(사용자 결정: "리워드 구조는
#   grasp_sensor 쪽으로 모두 바꾸기"). 복사가 아니라 import 다 — 복사하면 두 트랙이
#   다시 갈라지고, 한쪽에서 고친 결함이 다른 쪽에 남는다(lift 분모 0.10 이 그랬다).
from ..grasp_sensor.rewards import compute_grasp_sensor_rewards, contact_gate
from .rewards_tip import compute_tip_rewards

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


def _kuka_absolute(a: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """DEXTRAH `compute_absolute_action` 과 동일한 절대 박스 매핑.

    ``scale(a, lo, hi) = 0.5·(hi−lo)·a + 0.5·(hi+lo)`` 뒤 박스로 clamp.
    a=−1 → lo · a=0 → **박스 중앙** · a=+1 → hi. 방향 대칭이라 같은 액션 크기가
    부호와 무관하게 같은 이동을 낸다(구 "홈 기준 구간별 선형"은 비대칭이었다).
    """
    return (0.5 * (hi - lo) * a + 0.5 * (hi + lo)).clamp(lo, hi)


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
        # ★super().__init__ 안에서 _apply_action 이 불릴 수 있으므로 **먼저** 정한다.
        #   중력이 꺼져 있으면 보상은 무의미하므로 0 으로 잠근다(이중 부정 방지).
        self._grav_comp = float(cfg.gravity_compensation) if cfg.enable_gravity else 0.0
        # ★★부팅 가드: cfg **필드**와 파생 robot_cfg 가 실제로 일치하는지 확인한다.
        #   `params/env.yaml` 덤프는 resolve_cfg **이전** 상태라 필드만 보고 판단하면
        #   조용히 틀린 물리로 학습이 돈다(08.22 실측: probe 가 중력 False 를 찍었는데
        #   USD 는 True 였다. 같은 계열 결함으로 fab_test1~4 를 통째로 날린 적 있다).
        _sp = cfg.robot_cfg.spawn
        _gr_off = bool(_sp.rigid_props.disable_gravity)
        _sc_on = bool(_sp.articulation_props.enabled_self_collisions)
        if _gr_off == bool(cfg.enable_gravity) or _sc_on != bool(cfg.enable_self_collisions):
            raise RuntimeError(
                "물리 스위치가 파생 cfg 에 반영되지 않았다 — resolve_cfg 경로를 확인할 것.\n"
                f"  enable_gravity={cfg.enable_gravity} 인데 spawn.disable_gravity={_gr_off}\n"
                f"  enable_self_collisions={cfg.enable_self_collisions} 인데 "
                f"spawn.enabled_self_collisions={_sc_on}")
        print(f"[grasp_lift_fabric] 물리: self_collisions={_sc_on} · gravity={not _gr_off}"
              f" · grav_comp={self._grav_comp}", flush=True)
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
        self._frozen_t = torch.tensor(_frozen, device=self.device, dtype=torch.long)
        self._hand_free_t = torch.tensor(
            [i for i in self.hand_ids if i not in _fset], device=self.device, dtype=torch.long)
        n_free = len(self._hand_free_t)
        # ★tip IK 모드에서는 손 액션이 관절이 아니라 손끝 5점 × xyz 다. frozen_hand_joints
        #   는 이 모드에서 **적용되지 않는다** — fabric 이 손 20-DOF 를 전부 소유하므로
        #   일부만 얼리면 fabric 이 아는 자세와 실제가 어긋난다(그게 바로 고치려는 결함).
        _hc = str(getattr(self.cfg, "hand_control", "pd"))
        self._tip_ik = (_hc == "tip") or bool(getattr(self.cfg, "use_tip_fabric", False))
        self._tip_delta = (str(getattr(self.cfg, "tip_action_mode", "absolute")) == "delta")
        # 손 20-DOF 를 Fabrics 가 소유하되 **액션 의미는 관절 그대로**.
        # fabric 이 손을 알아야 body_repulsion 에 손가락↔손가락 쌍을 넣을 수 있고,
        # 그래야 PhysX self-collision 을 끌 근거가 생긴다(스텝 시간의 55~64%).
        self._hand_fabric = (_hc == "fabric") and not self._tip_ik
        _n_hand_act = 3 * len(p.fingertip_bodies) if self._tip_ik else n_free
        if 6 + _n_hand_act != self.cfg.action_space:
            raise RuntimeError(
                f"[{p.name}] 액션 차원 불일치: 6+{_n_hand_act} != {self.cfg.action_space}. "
                "use_tip_fabric / frozen_hand_joints 와 cfg 파생이 어긋났다."
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

        if self._tip_ik:
            print(f"[grasp_lift_fabric] 손 관절 {len(self.hand_ids)} 전부 Fabrics 제어 "
                  f"→ 정책 액션 = 손끝 {len(p.fingertip_bodies)}점 × xyz", flush=True)
        else:
            print(f"[grasp_lift_fabric] 손 관절 {len(self.hand_ids)} 중 "
                  f"{len(_frozen)}개 고정(외전) → 정책 제어 {n_free}", flush=True)

        self.palm_idx = self._one_body(p.palm_body)
        self._tip_t = torch.tensor(
            [self._one_body(b) for b in p.fingertip_bodies],
            device=self.device, dtype=torch.long,
        )
        # ★★손의 TCP = `palm_ee`(사용자 확인: **+x 축이 손바닥 법선**). palm 원점은
        #   손목 쪽이라 접근 기준점이 아니다. 자산 URDF 실측 palm+(28,0,40)mm.
        #   ★위치만으로는 정렬을 못 본다 — palm_ee 의 rpy 가 0 이라 회전이 palm 과 같고,
        #     오프셋 방향 (0.028,0,0.04) 은 법선과 다르다. 그래서 obs 에 **자세**를
        #     함께 넣는다(회전행렬 x·z 열 = 6D 연속 표현, y 는 외적으로 복원 가능).
        _ee = p.palm_ee_body or p.palm_body
        self._tcp_idx = self._one_body(_ee)
        if p.palm_ee_body is None:
            print(f"[grasp_lift_fabric] ⚠ 프로필에 palm_ee_body 가 없어 palm 원점을 "
                  f"TCP 로 쓴다 — 접근 정렬 관측이 부정확해진다", flush=True)

        # ---- 감쌈 판정 손바닥면 필터 + grip 항 입력 (08.24) --------------------------
        # wrap 마디 body 인덱스 (F, P). 손가락마다 마디 수가 같아야 텐서가 성립한다.
        _wraps = [p.finger_wrap_bodies.get(f, ()) for f in p.fingers]
        _np = len(_wraps[0])
        if any(len(w) != _np for w in _wraps):
            raise RuntimeError(f"[{p.name}] finger_wrap_bodies 마디 수 불일치: "
                               f"{[len(w) for w in _wraps]}")
        self._wrap_t = torch.tensor(
            [[self._one_body(b) for b in w] for w in _wraps],
            device=self.device, dtype=torch.long,
        )                                                   # (F, P)
        if self._tip_ik and bool(getattr(cfg, "require_palmar_contact", False)):
            # 손바닥면 축(링크 로컬) — 추측 금지, 프로필 실측값만. 미정의 = 부팅 사망.
            _missing = [f for f in p.fingers if f not in p.palmar_axis_local]
            if _missing:
                raise RuntimeError(
                    f"[{p.name}] palmar_axis_local 미정의: {_missing} — 손바닥/손등 "
                    "구분 불가. probe_palmar_sign.py 로 실측해 프로필에 넣거나 "
                    "require_palmar_contact=False 로 끌 것(끄면 손등 접촉을 감쌈으로 센다).")
            _ax = torch.tensor([p.palmar_axis_local[f] for f in p.fingers],
                               device=self.device, dtype=torch.float)     # (F, 3)
            if bool((_ax.norm(dim=-1) < 1e-6).any()):
                raise RuntimeError(f"[{p.name}] palmar_axis_local 에 영벡터가 있다")
            self._palmar_ax = torch.nn.functional.normalize(_ax, dim=-1)
            print(f"[grasp_lift_fabric] 감쌈 판정 = 손바닥면 접촉만 "
                  f"(palmar_axis {p.palmar_axis_local[p.fingers[0]]})", flush=True)
        else:
            self._palmar_ax = None

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

        # ---- D: 방향 대칭 액션 스케일 -------------------------------------------
        # 축마다 하나의 스케일(양쪽 중 **큰 쪽**)을 쓰고 결과를 박스로 clamp 한다.
        # 작은 쪽을 쓰면 y 범위가 40mm 로 줄어 컵(홈에서 +182mm)에 닿지 못한다.
        _up = self.palm_hi - self.home_palm
        _dn = self.home_palm - self.palm_lo
        self._palm_scale = (torch.maximum(_up, _dn) if bool(cfg.symmetric_action_scale)
                            else None)

        # ---- A: 지령 속도 제한 ----------------------------------------------------
        _sp = float(cfg.palm_slew_pos)
        _sr = float(cfg.palm_slew_rot_deg) * math.pi / 180.0
        self._slew = torch.tensor([_sp, _sp, _sp, _sr, _sr, _sr], device=self.device)
        self._slew_on = _sp > 0.0 or _sr > 0.0
        # 지령 버퍼. 리셋 시 홈으로 되돌린다(안 되돌리면 이전 에피소드 지령이 샌다).
        self.palm_cmd = self.home_palm.clone()
        self._prev_cmd = self.home_palm.clone()

        A = cfg.action_space
        self.actions = torch.zeros(self.num_envs, A, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, A, device=self.device)
        # 자유 관절만 정책이 움직인다. 고정 관절은 default(init) 값 그대로 명령된다.
        # tip IK 모드에서는 손 **전체**가 fabric 산출물이라 크기가 다르다.
        self.hand_targets = self._default_q[
            :, (self._hand_t if (self._tip_ik or self._hand_fabric) else self._hand_free_t)].clone()
        self.palm_targets = torch.zeros(self.num_envs, 6, device=self.device)

        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_spawn_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_reached_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ---- 접촉 그룹 인덱스 -------------------------------------------------------
        fingers = list(p.fingers)
        self._fingers = fingers
        # ★대향 파지점 approach 기하는 08.23 부터 grasp_sensor.rewards 안에서 계산한다.
        #   그쪽은 손끝 인덱스로 접촉 그룹 인덱스(_grp_a/_grp_b_env)를 **그대로** 쓰므로
        #   fingertip_bodies 와 fingers 의 순서가 같아야 성립한다 — fail-loud 로 강제한다.
        if len(p.fingertip_bodies) != len(fingers):
            raise RuntimeError(
                f"[{p.name}] fingertip_bodies({len(p.fingertip_bodies)}) 와 "
                f"fingers({len(fingers)}) 길이가 다르다 — 접근 보상의 그룹 인덱스가 어긋난다")
        # ★B. persistence — 대향 게이트 연속 유지 스텝 수
        self._gate_hold = torch.zeros(self.num_envs, device=self.device)
        self._grp_a = torch.tensor([fingers.index(f) for f in p.contact_group_a],
                                   device=self.device, dtype=torch.long)
        self._grp_b = torch.tensor([fingers.index(f) for f in p.contact_group_b],
                                   device=self.device, dtype=torch.long)
        # 인벨롭 손가락(감쌈 분모·d_side 의 wrap 그룹) — 프로필이 정의한다.
        # ★pinky 는 밑동 굴곡이 기본 자세에 없다 — _1 을 60° 로 열어 _2 를 굴곡축으로
        #   돌려놔야 분모 5 가 성립한다(robots.py envelope_fingers 주석).
        if not p.envelope_fingers:
            raise RuntimeError(f"[{p.name}] envelope_fingers 미정의 — 인벨롭 보상 성립 불가")
        self._env_f = torch.tensor([fingers.index(f) for f in p.envelope_fingers],
                                   device=self.device, dtype=torch.long)
        self._grp_b_env = torch.tensor(
            [fingers.index(f) for f in p.contact_group_b if f in p.envelope_fingers],
            device=self.device, dtype=torch.long)
        if len(self._grp_b_env) == 0:
            raise RuntimeError(f"[{p.name}] contact_group_b ∩ envelope_fingers 가 비었다")

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
        # ★★KUKA 고정(08.25) 물체 크기 obs — 원본 teacher 관측의 `object_scale`.
        #   onehot 은 "몇 번 물체인가"만 알려 주고 **크기 자체는 못 알려 준다**.
        #   같은 형상을 스케일만 바꿔 쓰는 뱅크에서는 onehot 이 커질수록 낭비이고,
        #   미학습 크기로 일반화도 안 된다. 스케일은 실기에서도 비전으로 추정 가능하다.
        _scl = torch.tensor([list(s.scale) for s in self.bank.specs],
                            device=self.device, dtype=torch.float32)   # (S,3)
        self._object_scale = _scl[self.object_idx]                     # (N,3)

        # ---- ADR (축 둘: 스폰 반경 · goal 오프셋 반경) --------------------------------
        # goal 축은 initial 0 이라 반경 0 = 구 고정 goal 과 동치로 시작한다.
        # ★★KUKA 고정(08.25): 원본 ADR 축을 전부 연결한다. 시작값만 맞추고 끝으로 가는
        #   경로가 없으면 커리큘럼이 아니라 고정값이다(보상 가중치 축은 지시로 제외).
        self.adr = _adr.TaskADR(
            {"spawn": {"xy_range": (cfg.spawn_range_initial, cfg.spawn_range_final),
                       "rotation": (0.0, cfg.adr_object_rotation_final)},
             "goal": {"xy_radius": (cfg.goal_xy_radius_initial, cfg.goal_xy_radius_final),
                      "z_radius": (cfg.goal_z_radius_initial, cfg.goal_z_radius_final)},
             "pd_targets": {"velocity_target_factor": (
                 cfg.velocity_target_factor, cfg.adr_velocity_target_factor_final)},
             "fabric_damping": {"gain": (
                 cfg.fabrics_damping_gain, cfg.adr_fabric_damping_final)},
             "robot_spawn": {"joint_pos_noise": (0.0, cfg.adr_robot_joint_pos_noise_final),
                             "joint_vel_noise": (0.0, cfg.adr_robot_joint_vel_noise_final)},
             "object_state_noise": {"scale": (
                 cfg.obs_noise_scale, cfg.adr_obs_noise_scale_final)},
             "object_wrench": {"max_linear_accel": (
                 0.0, cfg.adr_object_wrench_accel_final)}},
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
            # PCA(5D)는 감쌈을 제약하므로 쓰지 않는다 — direct 는 20-DOF 관절 그대로다.
            use_hand_fabric=self._hand_fabric,
            hand_mode="direct",
            hand_attractor_gain=getattr(self.cfg, "hand_attractor_gain", None),
            use_hand_repulsion=bool(getattr(self.cfg, "use_hand_repulsion", False)),
            # ★★KUKA식 손↔팔 repulsion(13 쌍 상당). 손가락 쌍은 여기 포함되지 않는다.
            use_body_repulsion_pairs=bool(
                getattr(self.cfg, "use_body_repulsion_pairs", False)),
            use_tip_fabric=self._tip_ik,     # 손끝 5점 위치 attractor(작업공간 IK)
            tip_attractor_gain=float(self.cfg.tip_attractor_gain) if self._tip_ik else None,
            # 손끝 attractor 를 손가락별 5 개로 — 각 항이 그 손가락 4 관절만 움직인다.
            tip_per_finger=self._tip_ik and bool(getattr(self.cfg, "tip_per_finger", False)),
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
        # hand_mode="direct" 면 (B, 손DOF) 관절 목표, 아니면 원본 PCA 계약(B,5).
        _hcmd = p.num_hand_joints if self._hand_fabric else 5
        self._fabric_hand_cmd = torch.zeros(self.num_envs, _hcmd, device=self.device)
        # ★단계형 보상이 파지중심을 쓰고, 그 실측(`_measure_tip_workspace`)이 이
        #   인덱스/한계를 쓴다 — 손 제어 방식과 무관하게 만들어야 pd 대조군이 산다.
        if self._tip_ik or self._hand_fabric or bool(
                getattr(self.cfg, "use_staged_reward", False)):
            # fabric 손 구간의 관절 한계(= robot soft limit 을 fabric 순서로 재배열).
            _jl = self.robot.data.soft_joint_pos_limits[0]          # (J,2)
            _m = float(self.cfg.hand_limit_margin)
            self._fab_hand_lo = _jl[self._fab_t[n_arm:], 0] + _m
            self._fab_hand_hi = _jl[self._fab_t[n_arm:], 1] - _m
            # fabric 손 순서 → robot 손 관절 순서(적용 시 되돌리는 역매핑).
            _fab_hand_ids = self._fab_t[n_arm:].tolist()
            self._hand_from_fab = torch.tensor(
                [_fab_hand_ids.index(int(j)) for j in self._hand_t.tolist()],
                device=self.device, dtype=torch.long)
            # robot 손 관절 순서 → fabric 손 구간 순서(목표를 넘길 때 쓴다).
            _robot_hand_ids = self._hand_t.tolist()
            self._fab_from_hand = torch.tensor(
                [_robot_hand_ids.index(int(j)) for j in _fab_hand_ids],
                device=self.device, dtype=torch.long)

        # cspace attractor(널스페이스)의 rest 자세를 **프로필 홈**으로 맞춘다.
        #   fabric URDF 내장 기본값은 [1.0, -0.1, -0.6, 0.5, 0, 0, 0] 로 다른 셋업용이다.
        #   robot-agnostic 에서는 rest 자세도 프로필이 정하는 게 맞다.
        #   ※정직한 단서: 이 변경을 "추종 불량의 원인"으로 지목했다가 **철회**했다.
        #     그때 쓴 probe 가 scene.write_data_to_sim() 을 빠뜨려 관절 목표가 PhysX 에
        #     도달하지 않은 상태였다. 이 설정은 설계상 옳아서 남겨둘 뿐,
        #     추종 성능에 대한 근거는 아직 없다.
        # ★★palm attractor 오버라이드 — 공유 yaml 을 건드리지 않는다.
        #   Attractor 는 params dict 를 **참조**로 들고(fabric_term.py: self.params =
        #   params) 매 스텝 self.params['conical_gain'] 을 읽으므로, 생성 후 하위
        #   dict 를 덮으면 그대로 반영된다. 사본 yaml 도 임시파일도 필요 없다.
        self._override_palm_attractor()
        _prior = self.fabric.default_config.clone()
        self.fabric.default_config.copy_(self.fabric_q)
        # ★파지중심(손끝 홈 FK 평균)은 **제어 방식과 무관한 기하 상수**인데 산출이
        #   tip 모드에 묶여 있었다 — pd 대조군이 `_grasp_center_local` 없음으로 죽었다.
        #   단계형 보상이 이 값을 쓰므로 손 제어와 무관하게 항상 실측한다.
        if self._tip_ik or bool(getattr(self.cfg, "use_staged_reward", False)):
            self._measure_tip_workspace()
        print(
            "[grasp_lift_fabric] fabric.default_config 를 프로필 홈으로 교체\n"
            f"  이전(fabric 내장) 팔 : {[round(v, 3) for v in _prior[0, :n_arm].tolist()]}\n"
            f"  이후(프로필 홈) 팔  : "
            f"{[round(v, 3) for v in self.fabric_q[0, :n_arm].tolist()]}",
            flush=True,
        )
        self._fabric_damping = float(self.cfg.fabrics_damping_gain) * torch.ones(
            self.num_envs, 1, device=self.device)
        self._wrench_tick = 0        # 외란 렌치 주기 카운터 (KUKA wrench_trigger_every)

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

        # ★★KUKA 고정(08.25): 원본은 매 스텝 ADR 에서 cspace 감쇠를 읽어 갱신한다
        #   (`fabric_damping.gain` 10 → 20). 부팅 시 한 번만 채우면 커리큘럼이 죽는다.
        self._fabric_damping.fill_(float(self.adr.get_param("fabric_damping", "gain")))

        # ★★KUKA 고정(08.25) — 원본 `compute_absolute_action` 과 **같은 식**이다:
        #     scale(a, lo, hi) = 0.5·(hi−lo)·a + 0.5·(hi+lo)   →  a=−1 lo · a=0 **중앙** · a=+1 hi
        #     그다음 tensor_clamp(lo, hi).
        #   구 규약은 a=0 이 **홈**인 구간별 선형이었다 — 홈이 박스 중앙이 아니면 두 반쪽의
        #   기울기가 달라 같은 액션 크기가 방향에 따라 다른 이동을 낸다.
        a_arm = self.actions[:, :6]
        desired = _kuka_absolute(a_arm, self.palm_lo, self.palm_hi)
        if self._slew_on:
            # A: 지령을 목표 쪽으로 **스텝당 상한만큼만** 움직인다.
            d = (desired - self.palm_cmd).clamp(-self._slew, self._slew)
            self.palm_cmd = self.palm_cmd + d
        else:
            self.palm_cmd = desired
        self.palm_targets = self.palm_cmd
        if self._tip_ik:
            # 손: 손끝 5점의 **palm 상대** 목표. a=0 → 홈 자세, a=±1 → 실측 도달 박스 끝.
            # 팔 액션과 같은 구간별 선형 규약이라 두 액션의 의미가 일관된다.
            a_tip = self.actions[:, 6:].view(self.num_envs, -1, 3)
            if self._tip_delta:
                # 누산: 스텝당 delta_scale 만큼만 움직인다. 절대 모드는 정책이 매 스텝
                # 다른 곳을 지시해 fabric(τ≈0.78s)이 못 따라갔다(A 6.7→19.8mm 실측).
                self._tip_cmd = (self._tip_cmd
                                 + a_tip * float(self.cfg.tip_delta_scale)
                                 ).clamp(self._tip_lo, self._tip_hi)
                tip_local = self._tip_cmd
            else:
                # KUKA 고정: 팔과 **같은** 절대 박스 매핑(a=0 → 박스 중앙).
                tip_local = _kuka_absolute(a_tip, self._tip_lo, self._tip_hi)
            # palm 좌표계 → world. ★기준은 **지령** palm(`palm_cmd`)이 아니라 fabric 이
            #   실제로 도달한 palm 이다. 지령을 쓰면 추종오차만큼 손끝 목표가 컵에서
            #   어긋나고, 그 오차는 파지 순간에 가장 크다.
            o, R = self._palm_frame(self.fabric_q.detach())
            tip_world = o[:, None, :] + torch.einsum("bij,bkj->bki", R, tip_local)
            self._tip_target = tip_world.reshape(self.num_envs, -1)
            self.fabric.set_features(
                self._fabric_hand_cmd, self.palm_targets, "euler_zyx",
                self.fabric_q.detach(), self.fabric_qd.detach(),
                self._world_ids, self._world_indicator, self._fabric_damping,
                tip_target=self._tip_target,
            )
        else:
            # 손: **절대** 관절 목표 (전범위). 손은 홈이 곧 "펴진 상태"라 대칭 매핑으로 둔다 —
            # a=-1 이 완전 개방, a=+1 이 완전 폐합이 되는 편이 학습에 자연스럽다.
            u_hand = 0.5 * (self.actions[:, 6:] + 1.0)
            _free_targets = self._hand_lo + u_hand * (self._hand_hi - self._hand_lo)
            if self._hand_fabric:
                # ★손 전체(고정 관절 포함)를 fabric 에 넘긴다. 정책이 안 건드리는 외전은
                #   init 값을 목표로 줘서 fabric 이 그 자세를 **유지**하게 한다 —
                #   일부만 넘기면 fabric 이 아는 손과 실제가 어긋나 body_repulsion 이
                #   틀린 형상으로 회피한다(그게 이 배선으로 고치려는 결함이다).
                _full = self._default_q[:, self._hand_t].clone()
                _free_cols = self._hand_from_fab.new_tensor(
                    [self._hand_t.tolist().index(int(j)) for j in self._hand_free_t.tolist()])
                _full[:, _free_cols] = _free_targets
                self._fabric_hand_cmd = _full[:, self._fab_from_hand]
            else:
                self.hand_targets = _free_targets
            self.fabric.set_features(
                self._fabric_hand_cmd, self.palm_targets, "euler_zyx",
                self.fabric_q.detach(), self.fabric_qd.detach(),
                self._world_ids, self._world_indicator, self._fabric_damping,
            )

        self._step_fabric()
        # ★★KUKA 고정(08.25): 원본은 fabric 적분 **직후** 물체 외란 렌치를 넣는다.
        self._apply_object_wrench()

    # ------------------------------------------------------------------
    def _palm_frame(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """fabric FK 로 (palm 원점 (N,3), palm 회전 (N,3,3) — 열이 x/y/z 축).

        프레임 이름을 여기서 알 필요가 없다: fabric 의 "palm" taskmap 이 원점과 세 축
        보조점을 이미 자기 규약으로 뽑는다(`get_palm_pose` 와 같은 구성). 로봇마다
        다른 링크명을 태스크 코드가 알면 robot-agnostic 이 깨진다.
        """
        pts, _ = self.fabric.get_taskmap("palm")(q, None)
        if pts.shape[1] < 18:
            raise RuntimeError(
                "palm taskmap 이 1점 모드다(palm_position_only) — 손끝 목표를 palm "
                "좌표계로 변환할 회전이 없다. tip IK 는 6-DOF palm 규약을 요구한다.")
        o = pts[:, :3]
        ax = torch.stack([
            torch.nn.functional.normalize(pts[:, 3:6] - o, dim=1),
            torch.nn.functional.normalize(pts[:, 9:12] - o, dim=1),
            torch.nn.functional.normalize(pts[:, 15:18] - o, dim=1),
        ], dim=-1)                                    # (N,3,3) 열 = 축
        return o, ax

    def _override_palm_attractor(self) -> None:
        """cfg 가 지정하면 palm attractor 의 gain/damping 을 덮는다(fail-loud).

        동역학: ``xdd = -gain·tanh(sharpness·|err|)·n̂ - damping·xd`` (forcing policy).
        정상상태에서 목표가 속도 v 로 움직이면 ``gain·tanh(sharp·err) = damping·v`` 라
        추종오차 err 이 **gain 에 반비례 · damping 에 비례**한다. 08.25 실측 A=16.3mm
        가 이 식으로 설명된다(v=0.42m/s, gain 80, damping 50 → 6.6mm 예측).
        ★gain 은 곧 최대 가속도[m/s²] — 실기가 못 내는 값이면 sim 전용 정책이 된다.
        """
        # 0 = "yaml 값 그대로"(sentinel). None 기본값은 configclass 타입 추론을
        # NoneType 으로 굳혀 hydra 오버라이드를 막는다 — cfg 주석 참고.
        _g = float(getattr(self.cfg, "palm_attractor_gain", 0.0) or 0.0)
        _d = float(getattr(self.cfg, "palm_attractor_damping", 0.0) or 0.0)
        if _g <= 0.0 and _d <= 0.0:
            return
        _pp = getattr(self.fabric, "fabric_params", {}).get("palm_attractor")
        if not isinstance(_pp, dict):
            raise RuntimeError(
                "[grasp_lift_fabric] fabric_params['palm_attractor'] 를 못 찾았다 — "
                "오버라이드가 조용히 무시되면 스윕 결과가 전부 같은 값이 된다.")
        _before = (_pp.get("conical_gain"), _pp.get("damping"))
        if _g > 0.0:
            _pp["conical_gain"] = _g
        if _d > 0.0:
            _pp["damping"] = _d
        print(f"[grasp_lift_fabric] palm attractor 오버라이드: "
              f"gain {_before[0]} → {_pp['conical_gain']} · "
              f"damping {_before[1]} → {_pp['damping']}", flush=True)

    def _measure_tip_workspace(self) -> None:
        """손끝 5점의 **palm 상대** 도달 박스를 부팅 시 1회 실측한다.

        하드코딩하지 않는 이유: 자산(손)이 바뀌면 도달 영역도 바뀐다. 랜덤 손 자세를
        FK 로만 돌리므로 물리도 Isaac 도 필요 없다.
        ★팔은 홈에 둔다 — palm 은 팔 관절만의 함수라 손 자세를 바꿔도 안 움직인다.
        """
        n_arm = self.profile.num_arm_joints
        q0 = self.fabric_q.clone()
        lo_j = self._fab_hand_lo.unsqueeze(0)
        hi_j = self._fab_hand_hi.unsqueeze(0)
        _acc: list[torch.Tensor] = []
        rounds = max(1, int(self.cfg.tip_workspace_samples) // max(self.num_envs, 1))
        for r in range(rounds):
            q = q0.clone()
            u = torch.rand(self.num_envs, hi_j.shape[1], device=self.device)
            q[:, n_arm:] = lo_j + u * (hi_j - lo_j)
            if r == 0:
                q[0, n_arm:] = q0[0, n_arm:]          # 표본 하나는 홈 자세
            o, R = self._palm_frame(q)
            tips, _ = self.fabric._fingertip_taskmap(q, None)
            tips = tips.reshape(self.num_envs, -1, 3)
            rel = torch.einsum("bij,bkj->bki", R.transpose(1, 2), tips - o[:, None, :])
            if r == 0:
                self._tip_home = rel[0].clone()       # (T,3) 홈 자세의 손끝 (palm 상대)
                # 파지중심 = 손끝 홈 평균(palm-local 상수). approach 의 기준점 —
                # palm 원점을 쓰면 d=0 이 "손바닥이 컵을 관통"이라 목표가 물리적으로
                # 틀리다(유효 파지에서 palm 원점은 컵에서 ~150mm 뒤).
                # ★대향 두 그룹의 **중점**. 5 점 평균은 thumb 이 손바닥에 붙어 있어
                #   허공을 가리켰다(08.25 FK 실측: 진짜 파지점과 35~60mm 괴리).
                #   ★이 함수는 super().__init__() → _setup_fabrics 흐름에서 불려
                #   self._grp_a 보다 **먼저** 돈다 — 프로필에서 직접 인덱싱한다.
                _fg = list(self.profile.fingers)
                _ia = [_fg.index(f) for f in self.profile.contact_group_a]
                _ib = [_fg.index(f) for f in self.profile.contact_group_b]
                self._grasp_center_local = 0.5 * (
                    rel[0][_ia].mean(dim=0) + rel[0][_ib].mean(dim=0))
            _acc.append(rel)
        # 액션 박스: 홈을 중심으로 실측 범위의 span_frac 만큼만 연다.
        # ★경계는 min/max 가 아니라 **백분위**다. 극값은 표본 하나가 정하는데 그
        #   코너는 다른 축과 동시 달성이 안 되는 자세라, 박스 부피만 키우고 실현
        #   가능한 지시는 늘리지 않는다 — 그 부피가 곧 도달 불가 지시가 된다
        #   (span 0.8 에서 박스 균일 샘플의 5mm 이내 도달 9.7%).
        _rel = torch.cat(_acc, dim=0)                             # (S,T,3)
        _q = float(self.cfg.tip_workspace_quantile)
        rel_min = torch.quantile(_rel, 1.0 - _q, dim=0)
        rel_max = torch.quantile(_rel, _q, dim=0)
        f = float(self.cfg.tip_action_span_frac)
        self._tip_lo = self._tip_home - (self._tip_home - rel_min).clamp(min=0.0) * f
        self._tip_hi = self._tip_home + (rel_max - self._tip_home).clamp(min=0.0) * f
        # 누산 모드의 지령 상태. 리셋 때 홈으로 되돌린다(_reset_idx).
        self._tip_cmd = self._tip_home.unsqueeze(0).expand(
            self.num_envs, *self._tip_home.shape).clone()
        _sp = (self._tip_hi - self._tip_lo) * 1000.0
        _dead = int((_sp < 1.0).sum())            # 폭 1mm 미만 = 그 축 액션이 무효
        _gc = self._grasp_center_local * 1000.0
        print(f"[grasp_lift_fabric] 손끝 워크스페이스 실측(palm 상대, {rounds*self.num_envs}표본) "
              f"span[mm] x{_sp[:, 0].max():.0f} y{_sp[:, 1].max():.0f} z{_sp[:, 2].max():.0f} "
              f"· span_frac={f} · q={_q} · 파지중심 palm+({_gc[0]:.0f},{_gc[1]:.0f},{_gc[2]:.0f})mm",
              flush=True)
        if _dead:
            print(f"[grasp_lift_fabric] ⚠ 손끝 액션 축 {_dead}/15 개가 폭 <1mm — 홈이 백분위 "
                  f"대역 밖이라 그 축 지시가 무효다(tip_workspace_quantile 을 올릴 것).",
                  flush=True)

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

    def _apply_object_wrench(self) -> None:
        """★★KUKA 고정(08.25) — 원본 `apply_object_wrench`.

        1 초(=`wrench_trigger_every` 정책 스텝)마다 물체에 **랜덤 외란 가속도**를 준다.
        파지가 성립한 뒤 흔들려도 놓치지 않게 하는 강건화 축이고, ADR 로 0 → 10 m/s²
        까지 올라간다. 손이 물체에서 멀면(`wrench_hand_distance_threshold`) 넣지 않는다
        — 아직 안 쥔 물체를 흔드는 것은 과제와 무관한 교란이다.

        가속도로 주는 이유: 힘으로 주면 질량 DR 과 곱해져 무거운 물체만 덜 흔들린다.
        토크는 같은 가속도에 `torsional_radius` 를 곱한 모멘트 팔로 환산한다(원본 동일).
        """
        accel = float(self.adr.get_param("object_wrench", "max_linear_accel"))
        if accel <= 0.0:
            return
        self._wrench_tick += 1
        if self._wrench_tick % max(1, int(self.cfg.wrench_trigger_every)) != 0:
            return
        # 손 ↔ 물체 거리 게이트 (원본 hand_to_object_dist_threshold)
        obj_w = self.object.data.root_pos_w
        palm_w = self.robot.data.body_pos_w[:, self.palm_idx]
        near = (obj_w - palm_w).norm(dim=-1) < float(self.cfg.wrench_hand_distance_threshold)
        mass = self.object.data.default_mass.to(self.device).view(-1, 1)
        a = accel * 2.0 * (torch.rand(self.num_envs, 3, device=self.device) - 0.5)
        forces = a * mass * near.unsqueeze(-1)
        torques = forces * float(self.cfg.torsional_radius)
        self.object.set_external_force_and_torque(
            forces.unsqueeze(1), torques.unsqueeze(1))

    def _apply_action(self) -> None:
        """fabric 계획을 관절 PD 목표로 내린다 — **위치와 속도를 모두** 준다.

        ★★08.25 속도 피드포워드를 되살렸다. 이전에는 `fabric_qd` 를 계산해 놓고
        버린 채 0 을 넣었는데, 그러면 PD 의 감쇠항이 움직임 자체를 되민다:

            속도목표 0   : kp·err = kd·v + τ_마찰  →  err ≈ (kd/kp)·v = 0.2·v [rad]
            속도목표 qd  : kp·err = τ_마찰만       →  err ≈ τ_f/kp

        우리 팔은 kp=400 · kd=80 이라 kd/kp = 0.2 — **속도에 정비례해 뒤처지도록**
        배선돼 있었다. DEXTRAH 원본은 `dof_vel_targets = clone(fabric_qd)` 를 그대로
        내리고, `velocity_target_factor` 를 ADR 로 1.0 → 0.0 으로 **줄여 간다**
        (실기의 불완전한 속도 목표에 대비한 커리큘럼). 우리는 그 최종 단계에서
        시작한 셈이었다.
        """
        n_arm = self.profile.num_arm_joints
        # ★★KUKA 고정(08.25): 원본은 이 계수를 ADR 로 1.0 → 0.0 으로 줄여 간다.
        _vf = float(self.adr.get_param("pd_targets", "velocity_target_factor"))
        arm_target = self.fabric_q[:, :n_arm]
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_ids)
        self.robot.set_joint_velocity_target(
            self.fabric_qd[:, :n_arm] * _vf, joint_ids=self.arm_ids)
        if self._tip_ik or self._hand_fabric:
            # ★손 20-DOF 를 **전부** fabric 이 준다. 일부만 얼리면 fabric 이 아는 자세와
            #   실제가 어긋나 body_repulsion 이 틀린 손으로 회피를 계산한다.
            self.hand_targets = self.fabric_q[:, n_arm:][:, self._hand_from_fab]
            self.robot.set_joint_position_target(
                self.hand_targets, joint_ids=self.hand_ids)
            # 손도 fabric 소유면 같은 이유로 속도를 먹인다(손 PD 모드는 정책이 위치만
            # 내므로 속도 목표가 없다 — 그건 fabric 계획이 아니라 지령 자체가 없는 것).
            self.robot.set_joint_velocity_target(
                self.fabric_qd[:, n_arm:][:, self._hand_from_fab] * _vf,
                joint_ids=self.hand_ids)
        else:
            self.robot.set_joint_position_target(
                self.hand_targets, joint_ids=self._hand_free_t.tolist())
        # 고정 관절은 init 값 유지 — reset 에서 write 한 뒤 target 도 default 로 둔다.
        self._apply_gravity_compensation()

    # ------------------------------------------------------------------
    def _apply_gravity_compensation(self) -> None:
        """중력보상 피드포워드 토크.

        Fabrics 는 중력보상을 하지 않는다. 중력을 켠 채 보상이 없으면 관절 PD 의
        정상상태 오차가 그대로 처짐이 된다 — **실측 palm 57.6mm**(자세 의존 37~72mm)로
        `success_pos_std`(50mm)와 같은 규모라 성공 판정 정밀도를 통째로 먹는다.

        ★`gravity_compensation` 은 **계수**다. 1.0 = 완전 보상(중력 OFF 와 물리적으로
          동치), 0.9 = 10% 잔류 처짐, 0.0 = 보상 없음. 실기 컨트롤러도 질량 모델 오차
          때문에 완벽하지 않으므로 이 계수가 s2r 에서 의미 있는 DR 축이 된다.

        ImplicitActuator 는 피드포워드 effort 를 받는다(actuator_pd.py:104
        `computed_effort = stiffness·e_pos + damping·e_vel + joint_efforts`).
        PhysX 의 `get_gravity_compensation_forces()` 는 fixed-base 에서 (N, dof) 의
        **관절공간 보상 토크**를 그대로 준다(`get_generalized_gravity_forces` 는 deprecated).
        """
        if self._grav_comp <= 0.0:
            return
        tau = self.robot.root_physx_view.get_gravity_compensation_forces()
        self.robot.set_joint_effort_target(self._grav_comp * tau[:, : self.robot.num_joints])

    # ==================================================================
    def _contact(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(손가락별 총 접촉력, wrap 전마디 동시접촉 여부, 중간마디 힘, 원위마디 힘) 각 (N,F).

        · tot  — 참여 임계(0.1N) 소비자용 원값.
        · mid/dist — 마디별 원값. 보상의 `envelope_fraction`(grasp_sensor 규약: 마디
          **하나라도** 접촉하면 그 손가락은 감쌈)이 이 둘을 받아 판정한다.
        · wrapped — 우리 트랙이 08.22 에 도입한 **엄격** 판정(전 마디가 동시에 0.5N 초과).
          ★08.23 부터 보상에서 빠지고 **진단 지표**로만 남는다. 사용자 결정으로 보상
            규약을 grasp_sensor 와 통일했기 때문이다. 버리지 않는 이유: 같은 정책을 두
            판정으로 재면 0.503 vs 0.069 로 **7배**가 벌어진 실측이 있다. 느슨한 판정이
            오르는데 엄격 판정이 안 오르면 "받치기"를 감쌈으로 세고 있다는 신호다.
        """
        w_thr = float(getattr(self.cfg, "envelope_force_threshold", 0.5))
        tot, wrapped, mids, dists = [], [], [], []
        zero = torch.zeros(self.num_envs, device=self.device)
        for f in self._fingers:
            roles = self._sensors[f]
            t = torch.zeros(self.num_envs, device=self.device)
            w = None
            mags = []
            for s_ in roles["tip"]:
                t = t + s_.data.force_matrix_w.view(self.num_envs, -1, 3).sum(1).norm(dim=-1)
            for s_ in roles["wrap"]:
                m = s_.data.force_matrix_w.view(self.num_envs, -1, 3).sum(1).norm(dim=-1)
                t = t + m
                mags.append(m)
                w = m if w is None else torch.minimum(w, m)   # ★전 마디 동시 접촉
            if w is None:                                     # wrap 센서가 없는 프로필
                w = zero
            # 프로필 규약: wrap_bodies = (중간 _3, 원위 _4). 마디가 하나뿐이면 둘을 같게
            # 둔다(2지 그리퍼 jaw — 그 접촉 자체가 감쌈). grasp_sensor 와 같은 규약.
            mids.append(mags[0] if mags else zero)
            dists.append(mags[1] if len(mags) > 1 else (mags[0] if mags else zero))
            tot.append(t)
            wrapped.append((w > w_thr).float())
        return (torch.stack(tot, 1), torch.stack(wrapped, 1),
                torch.stack(mids, 1), torch.stack(dists, 1))

    def _local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return pos_w - self.scene.env_origins

    def _palmar_mask(self, obj_pos_local: torch.Tensor) -> torch.Tensor:
        """wrap 마디별 손바닥면 접촉 가능 여부 (N, F, P) bool.

        (마디 palmar 축·world) · (물체 − 마디원점) > 0 — 물체가 손바닥면 쪽에 있어야
        그 마디의 접촉을 감쌈으로 인정한다. 힘 벡터 부호 대신 기하를 쓰는 이유:
        힘 부호 규약은 실측 확정을 못 했고 **뒤집히면 판정이 통째로 반대인데 조용하다**
        (자매 규약 동일). 축 부호는 probe_palmar_sign.py 실측으로 프로필에 박는다.
        """
        F, P = self._wrap_t.shape
        flat = self._wrap_t.reshape(-1)
        pos = self.robot.data.body_pos_w[:, flat].view(self.num_envs, F, P, 3)
        quat = self.robot.data.body_quat_w[:, flat].view(self.num_envs, F, P, 4)
        ax = self._palmar_ax[None, :, None, :].expand(self.num_envs, F, P, 3)
        from isaaclab.utils.math import quat_apply
        ax_w = quat_apply(quat.reshape(-1, 4), ax.reshape(-1, 3)).view(self.num_envs, F, P, 3)
        obj_w = obj_pos_local + self.scene.env_origins
        to_obj = obj_w[:, None, None, :] - pos
        return (ax_w * to_obj).sum(dim=-1) > 0.0

    def _object_axis_w(self) -> torch.Tensor:
        """물체 local +z 를 world 로 (N,3) — 컵 축. 파지 자세 보상의 기준축이다."""
        from isaaclab.utils.math import quat_apply
        z = torch.zeros(self.num_envs, 3, device=self.device)
        z[:, 2] = 1.0
        return quat_apply(self.object.data.root_quat_w, z)

    def _object_up_and_tilt(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(물체 로컬 +z · world +z = cos(기울기), 기울기[deg]).

        ★보상은 cos 을 **직접** 쓴다(자매 트랙 규약). deg 로 바꿨다가 다시 cos 을
          취하면 acos/cos 왕복이라 계산만 늘고 소각에서 수치오차가 붙는다.
          deg 는 종료 판정·로깅용으로만 남긴다.
        """
        from isaaclab.utils.math import quat_apply
        z = torch.zeros(self.num_envs, 3, device=self.device)
        z[:, 2] = 1.0
        up = quat_apply(self.object.data.root_quat_w, z)[:, 2].clamp(-1.0, 1.0)
        return up, torch.rad2deg(torch.acos(up))

    def _object_tilt_deg(self) -> torch.Tensor:
        """물체 로컬 +z 와 world +z 사이 각도[deg]."""
        return self._object_up_and_tilt()[1]

    # ==================================================================
    def _get_observations(self) -> dict:
        from isaaclab.utils.math import matrix_from_quat

        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        idx = torch.cat([self._arm_t, self._hand_t])
        joint_pos, joint_vel = q[:, idx], qd[:, idx]

        # ★★KUKA 고정(08.25) 프레임 규약 — 원본 teacher 관측은 전부 로봇 베이스
        #   (env-local) 기준이다(hand_pos / object_pos / object_goal).
        object_pos = self._local(self.object.data.root_pos_w)          # (N,3)
        object_rot = self.object.data.root_quat_w                      # (N,4) critic 전용
        object_goal = self.goal_pos                                    # (N,3) env-local

        contact, _, _, _ = self._contact()                             # ★critic 전용
        # ★★KUKA 고정(08.25) 관측 노이즈. 원본은 전 항목이 *_noisy 이되 **ADR 로 0 에서
        #   시작**한다 — object_state_noise 축이 그 계수다.
        _ns = float(self.adr.get_param("object_state_noise", "scale"))
        if _ns > 0.0:
            def _u(x, w):        # 균일 ±w
                return x + (_ns * w) * 2.0 * (torch.rand_like(x) - 0.5)
            joint_pos = _u(joint_pos, float(self.cfg.obs_noise_joint_pos))
            joint_vel = _u(joint_vel, float(self.cfg.obs_noise_joint_vel))
            object_pos = _u(object_pos, float(self.cfg.obs_noise_object_pos))

        # ---- 손 TCP(palm_ee) 위치·자세 + 손끝 위치 ----------------------------------
        # ★TCP 는 `palm_ee` — 사용자 확인으로 **+x 축이 손바닥 법선**이다. 정렬(align)을
        #   보려면 위치가 아니라 이 축이 필요하다. 자세는 회전행렬의 x·z 열(6D)로 준다:
        #   quaternion 은 부호 이중성이 있고 euler 는 불연속이라 회귀 입력에 부적합하다.
        tcp_pos = (self.robot.data.body_pos_w[:, self._tcp_idx]
                   - self.scene.env_origins)                            # (N,3)
        _R = matrix_from_quat(self.robot.data.body_quat_w[:, self._tcp_idx])
        tcp_axes = torch.cat([_R[:, :, 0], _R[:, :, 2]], dim=-1)        # (N,6) x축·z축
        tip_pos = (self.robot.data.body_pos_w[:, self._tip_t]
                   - self.scene.env_origins[:, None, :])                # (N,T,3)

        # ★★KUKA 고정(08.25) fabric 계획 상태. 정책이 지시한 목표가 계획으로 어떻게
        #   풀렸는지 보지 못하면 "액션 → 실제 이동"이 부분관측이 된다.
        #   ※원본의 fabric_qd/qdd 는 `observation_annealing` 계수가 (0,0) 이라 **항상 0**
        #     이다(같은 계수가 원본의 joint_vel·hand_vel 까지 죽인다). 상수 0 을 LSTM 에
        #     통과시킬 이유가 없어 policy 에서 빼고 **critic 에만 실값**으로 준다.
        fabric_q = self.fabric_q

        parts = [joint_pos, joint_vel,
                 object_pos, object_goal, self.actions,
                 tcp_pos, tcp_axes, tip_pos.reshape(self.num_envs, -1),
                 self._object_scale, fabric_q]
        if self._onehot is not None:
            parts.append(self._onehot)
        obs = torch.cat(parts, dim=1)
        # ★critic 은 배포하지 않으므로 실기에서 못 얻는 것까지 본다:
        #   물체 회전(추정 불안정) · 접촉력 · 물체 6D 속도 · fabric 속도/가속 원본.
        state = torch.cat([obs, object_rot, contact.clamp(max=20.0),
                           self.object.data.root_lin_vel_w,
                           self.object.data.root_ang_vel_w,
                           self.fabric_qd, self.fabric_qdd], dim=1)
        return {"policy": torch.nan_to_num(obs), "critic": torch.nan_to_num(state)}

    # ==================================================================
    def _get_rewards(self) -> torch.Tensor:
        from isaaclab.utils.math import matrix_from_quat

        obj_pos = self._local(self.object.data.root_pos_w)
        palm_pos = self._local(self.robot.data.body_pos_w[:, self.palm_idx])
        tips = self.robot.data.body_pos_w[:, self._tip_t] - self.scene.env_origins[:, None, :]

        contact, wrapped, mid_f, dist_f = self._contact()
        # ★E. 임계 분리: 게이트(파지 성립)는 1.0N, **참여 판정**은 0.1N (v1/v2 동일).
        p_thr = float(self.cfg.participation_force_threshold)
        grip_frac = (contact > p_thr).float().mean(dim=1)
        # 진단 전용 — 08.22 엄격 감쌈(전 마디 동시 0.5N). 보상은 grasp_sensor 규약
        # (마디 하나라도 접촉)이므로 이 값은 **대조 지표**다: 같은 정책을 두 판정으로
        # 재면 0.503 vs 0.069 로 7배가 벌어졌다. 느슨한 쪽만 오르면 "받치기"다.
        strict_env = (wrapped[:, self._env_f].mean(dim=1)
                      if self.profile.has_wrap_sensors else grip_frac)

        # persistence — 보상에서는 빠졌지만(자매 트랙 규약) 축퇴 진단으로 계속 잰다.
        _gate_now = contact_gate(
            contact[:, self._grp_a], contact[:, self._grp_b],
            float(self.cfg.contact_force_threshold)).float()
        self._gate_hold = (self._gate_hold + 1.0) * _gate_now
        persistence = (self._gate_hold
                       / float(self.cfg.persistence_ref_steps)).clamp(0.0, 1.0)

        xy_disp = (obj_pos[:, :2] - self.object_spawn_pos[:, :2]).norm(dim=-1)
        up_cos, tilt = self._object_up_and_tilt()
        goal_dist = (obj_pos - self.goal_pos).norm(dim=-1)

        # ★★08.23 보상 전면 교체 — 자매 트랙 grasp_sensor 와 **같은 함수**를 쓴다.
        #   구 7항이 실측으로 드러낸 결함 셋(전부 이 교체로 사라진다):
        #     · `envelope_mul_floor` 0.3 → 감쌈 없이도 이송 보상의 30% 가 흘렀다.
        #       좌팔이 감쌈 0.21 로 이송(goal 0.58)만 학습한 직접 원인 — 우선순위 ① 붕괴.
        #     · `grasp_quality` 의 grip/persist 30% 몫 → 감쌈이 아닌 것으로 채워졌다
        #       (우팔 실측 1.3 중 0.43).
        #     · `lift_success_height` 0.10 ≠ goal 0.15 → dz 10cm 에서 포화. 우팔이
        #       한때 18cm 까지 들었다가 6cm 로 되돌아와 고착했다(되돌려도 손해가 없다).
        #   grasp_sensor 는 셋 다 없다: g_eff 하한 없음 · envelope 순수항 · 분모=goal 높이.
        #   대향 파지점 approach 기하도 그쪽 `approach_reward` 가 내부에서 계산한다
        #   (A 그룹 min · B∩envelope mean — 우리 구판은 A 도 mean 이었고 pinky 를
        #    분모에 넣고 있었다).
        if bool(getattr(self.cfg, "use_staged_reward", False)):
            # 단계형 보상 — 손 제어 방식과 **무관**하게 적용한다. "pd" 대조군도 같은
            # 보상으로 돌려야 손 제어 방식만의 차이를 본다(보상까지 바뀌면 원인 분리 불가).
            # 구 10 항은 관절 액션 전제라 세 곳이 어긋난다
            # (rewards_tip.py docstring: action_l2 부호·approach 커널·2 층 게이트).
            # 파지중심(world): 실제 palm 자세로 회전시킨 palm-local 상수.
            #   fabric 계획이 아니라 **실제** 자세를 쓴다 — 추종오차만큼 어긋나면
            #   보상이 계획 기준으로 후하게 나온다.
            from isaaclab.utils.math import quat_apply as _qa
            # ★★08.25 파지중심을 **대향 두 그룹의 중점**으로 바꿨다. 이전 정의(손끝
            #   5 점 평균)는 어느 손가락에도 없는 허공의 점이었다 — thumb 이 대향해
            #   손바닥에 붙어 있어(홈 z=+30) 4 지(z=+190)와 평균하면 중간 빈 공간이
            #   나온다. 그 결과 컵을 두 그룹 사이 정중앙에 완벽히 물려도 approach
            #   거리가 35~60mm 밑으로 못 내려갔다(FK 실측). 실제로 양팔 독립 런이
            #   똑같이 79mm 에서 멈추고 **후퇴**했다 — 접근을 더 해도 보상이 안 오르니
            #   리프트로 갈아탄 것이다. 두 그룹 중점은 컵이 물리는 자리 그 자체라
            #   d→0 이 달성 가능하고, 손을 오므려도 함께 따라와 파지 중에도 유지된다.
            #   ★손끝 위치는 obs 로 계산 가능한 값이라 형상 비의존 원칙을 지킨다.
            _gc_w = 0.5 * (tips[:, self._grp_a].mean(dim=1)
                           + tips[:, self._grp_b].mean(dim=1))
            # ── 손바닥면 필터: 감쌈 **입력만** 거른다. 대향 게이트·obs 는 불변 —
            #    실기 센서도 방향을 모르므로 obs 를 바꾸면 s2r 규약이 흔들린다(자매 규약).
            _F, _P = self._wrap_t.shape
            # ★_local() 은 (N,3) 전용이다 — (N,F*P,3) 을 넘기면 env_origins 와
            #   브로드캐스팅이 깨진다(실측: size 10 vs 512). 축을 명시해 뺀다.
            _wrap_pos = (self.robot.data.body_pos_w[:, self._wrap_t.reshape(-1)]
                         .view(self.num_envs, _F, _P, 3)
                         - self.scene.env_origins[:, None, None, :])
            if self._palmar_ax is not None:
                _pal = self._palmar_mask(obj_pos)                       # (N, F, P)
                mid_use = mid_f * _pal[:, :, 0].float()
                dist_use = dist_f * _pal[:, :, 1].float()
            else:
                _pal = None
                mid_use, dist_use = mid_f, dist_f
            # grip off 신호: 방향 **무관** 접촉(참여 임계 0.1N) — 손등으로 밀어도
            # 당김은 멈추지만(압입 방지) 감쌈 점수는 손바닥면만(의도된 비대칭).
            _touch = contact > float(self.cfg.participation_force_threshold)
            total, terms, gate, envelope_frac = compute_tip_rewards(
                palm_pos=palm_pos,
                grasp_center_pos=_gc_w,
                fingertip_pos=tips,
                object_pos=obj_pos,
                goal_pos=self.goal_pos,
                object_up=up_cos,
                # ★손 TCP(palm_ee) +x 축(손바닥 법선)의 world z 성분. 0 이면 법선이
                #   수평 = 컵 축(≡world +z)과 수직 = 측면 파지 자세다.
                tcp_normal_z=matrix_from_quat(
                    self.robot.data.body_quat_w[:, self._tcp_idx])[:, 2, 0],
                group_a_force=contact[:, self._grp_a],
                group_b_force=contact[:, self._grp_b],
                env_mid_force=mid_use[:, self._env_f],
                env_dist_force=dist_use[:, self._env_f],
                wrap_body_pos=_wrap_pos,
                finger_touch=_touch,
                actions=self.actions,
                prev_actions=self.prev_actions,
                n_arm_actions=6,
                cfg=self.cfg,
            )
            enclosure_dist = terms["_d_grasp"]
            # ── 감쌈 로깅: 손가락별 **손바닥면** 감쌈(원하는 값)과 방향무관 접촉,
            #    그리고 필터 전 값 — raw 와의 차이가 곧 손등 접촉 비중이다.
            from .rewards_tip import envelope_fraction_graded as _efg
            self.extras["task/envelope_frac_raw"] = _efg(
                mid_f[:, self._env_f], dist_f[:, self._env_f],
                float(self.cfg.contact_force_threshold)).mean()
            _thr = float(self.cfg.contact_force_threshold)
            _wrap_pal = 0.5 * ((mid_use > _thr).float() + (dist_use > _thr).float())
            _dwp = terms.pop("_d_wrap_per", None)
            _gpp = terms.pop("_grip_per", None)
            for _i, _fg in enumerate(self._fingers):
                self.extras[f"task/wrap/{_fg}"] = _wrap_pal[:, _i].mean()
                self.extras[f"task/touch/{_fg}"] = _touch[:, _i].float().mean()
                if _dwp is not None:
                    # 손가락별 wrap 마디↔물체 거리[mm] — 큰 값 = 못 갔다(도달/박스),
                    # 작은데 wrap 이 0 = 닿았는데 손등이다.
                    self.extras[f"task/dwrap_mm/{_fg}"] = _dwp[:, _i].mean() * 1000.0
                if _gpp is not None:
                    self.extras[f"task/gripterm/{_fg}"] = _gpp[:, _i].mean()
            # ── 접근 **방향** 진단 (08.24) ────────────────────────────────────────
            # 보상은 거리(`d_grasp`)만 보므로 위에서 덮든 옆에서 오든 점수가 같다.
            # 측면 파지가 성립하려면 ①손바닥이 컵을 향하고 ②palm 접근축이 컵 축에
            # 수직이며 ③손끝이 컵 둘레로 퍼져야 한다. 셋을 상시 로깅해 "접근은
            # 하는데 파지가 안 되는" 구간에서 자세 탓인지 바로 가른다.
            _pq = self.robot.data.body_quat_w[:, self.palm_idx]
            _pp = self.robot.data.body_pos_w[:, self.palm_idx]
            _obj_w = obj_pos + self.scene.env_origins
            _gc_dir = torch.nn.functional.normalize(
                self._grasp_center_local, dim=0).unsqueeze(0).expand(self.num_envs, 3)
            _fwd = _qa(_pq, _gc_dir)                       # palm 접근축(world)
            _to_obj = torch.nn.functional.normalize(_obj_w - _pp, dim=-1)
            # ① 손바닥이 컵을 향하는가 — 접근축과 (컵−palm) 사이 각[deg]. 0=정면.
            self.extras["dir/palm_to_obj_deg"] = torch.rad2deg(torch.acos(
                (_fwd * _to_obj).sum(-1).clamp(-1.0, 1.0))).mean()
            # ② 접근축 ↔ 컵 축 각[deg]. 측면 파지=90° 근처 · 위에서 덮기=0° 근처.
            _cup = self._object_axis_w()
            self.extras["dir/palm_vs_cup_axis_deg"] = torch.rad2deg(torch.acos(
                (_fwd * _cup).sum(-1).abs().clamp(max=1.0))).mean()
            # ③ 손끝의 컵 둘레 방위각 폭[deg] — 감싸면 크고 한쪽에서 밀면 작다.
            _v = (tips + self.scene.env_origins[:, None, :]) - _obj_w[:, None, :]
            _vp = _v - (_v * _cup[:, None, :]).sum(-1, keepdim=True) * _cup[:, None, :]
            _e1 = torch.nn.functional.normalize(_vp[:, 0, :], dim=-1)
            _e2 = torch.cross(_cup, _e1, dim=-1)
            _phi = torch.atan2((_vp * _e2[:, None, :]).sum(-1),
                               (_vp * _e1[:, None, :]).sum(-1))
            self.extras["dir/tip_azimuth_span_deg"] = torch.rad2deg(
                _phi.max(dim=1).values - _phi.min(dim=1).values).mean()
        else:
            total, terms, gate, envelope_frac = compute_grasp_sensor_rewards(
                palm_pos=palm_pos,
                fingertip_pos=tips,
                object_pos=obj_pos,
                goal_pos=self.goal_pos,
                group_a_force=contact[:, self._grp_a],
                group_b_force=contact[:, self._grp_b],
                group_a_tip_idx=self._grp_a,
                group_b_env_tip_idx=self._grp_b_env,
                env_mid_force=mid_f[:, self._env_f],
                env_dist_force=dist_f[:, self._env_f],
                object_tilt_deg=tilt,
                object_up=up_cos,
                height_delta=obj_pos[:, 2] - self.object_spawn_pos[:, 2],
                actions=self.actions,
                prev_actions=self.prev_actions,
                cfg=self.cfg,
            )
            enclosure_dist = terms["_d_side"]
        up_q = up_cos.clamp(0.0, 1.0) ** float(self.cfg.upright_exponent)
        # ★지터는 prev_actions 갱신 **전에** 재야 한다(갱신 후엔 항상 0).
        #   절대 액션이라 정책 노이즈가 곧 목표 순간이동이 된다 — 이 값이 크면
        #   Fabrics 가 저역통과시켜 정책 액션이 실제 palm 위치에 거의 영향을 못 준다.
        self.extras["action/arm_step_delta"] = (
            self.actions[:, :6] - self.prev_actions[:, :6]).abs().mean()
        self.extras["action/hand_step_delta"] = (
            self.actions[:, 6:] - self.prev_actions[:, 6:]).abs().mean()
        self.extras["action/arm_abs_mean"] = self.actions[:, :6].abs().mean()
        # ★슬루를 걸면 "액션 지터"와 "실제 지령 지터"가 갈린다. 액션은 여전히 떨 수
        #   있지만 지령이 안 떨면 팔은 정상 추종한다 — 그 분리를 봐야 판정이 된다.
        self.extras["action/cmd_step_mm"] = (
            self.palm_cmd[:, :3] - self._prev_cmd[:, :3]).norm(dim=-1).mean() * 1000.0
        self._prev_cmd.copy_(self.palm_cmd)
        self.prev_actions.copy_(self.actions)
        # ★★08.23 성공 판정 3조건 AND (자매 트랙 규약): goal 근접 AND 감쌈 AND 직립.
        #   구 판정은 goal 거리 하나뿐이라 **감쌈 0.21 로 이송만 한** 좌팔이 0.58 로
        #   집계됐다 — 지표가 우선순위 ①②를 못 보고 있었다. ADR 승급 기준도 이 값이다.
        _pass_pos = goal_dist < float(self.cfg.success_pos_tolerance)
        _pass_env = envelope_frac >= float(self.cfg.success_envelope_min)
        _pass_tilt = tilt < float(self.cfg.success_tilt_max_deg)
        self._goal_reached_now = _pass_pos & _pass_env & _pass_tilt
        # ★조건별 개별 통과율 — AND 만 보면 어느 조건이 병목인지 알 수 없다.
        self.extras["task/pass_pos"] = _pass_pos.float().mean()
        self.extras["task/pass_envelope"] = _pass_env.float().mean()
        self.extras["task/pass_tilt"] = _pass_tilt.float().mean()
        self.extras["task/goal_reached_loose"] = _pass_pos.float().mean()


        # ---- ADR (트리거 = **순간** 성공률) --------------------------------------
        if self.adr.maybe_increment(self._goal_reached_now.float().mean()):
            em = getattr(self, "event_manager", None)
            if em is not None:
                em.reset(env_ids=self.robot._ALL_INDICES)
                em.apply(env_ids=self.robot._ALL_INDICES, mode="reset",
                         global_env_step_count=0)

        # ---- 로깅 ----------------------------------------------------------------
        for k, v in terms.items():
            # `_` 로 시작하는 키는 보상이 아니라 진단 원값이다(d_palm/d_side/gate_eff).
            self.extras[(f"task/{k[1:]}" if k.startswith("_") else f"reward/{k}")] = v.mean()
        self.extras["reward/total"] = total.mean()
        self.extras["task/contact_gate"] = gate.float().mean()
        self.extras["task/envelope_frac"] = envelope_frac.mean()
        # ★대조: 전 마디 동시접촉(0.5N) 기준. 느슨한 쪽만 오르면 "받치기"다.
        self.extras["task/envelope_strict"] = strict_env.mean()
        self.extras["task/grip_frac"] = grip_frac.mean()
        self.extras["task/persistence"] = persistence.mean()
        self.extras["task/enclosure_dist"] = enclosure_dist.mean()
        self.extras["task/upright_quality"] = up_q.mean()
        # ★접촉력 **원값**. 이게 없으면 grip_frac=0 일 때 "진짜 미접촉"인지
        #   "임계(contact_force_threshold) 바로 아래"인지 구분할 수단이 없다
        #   — fab_test1 에서 실제로 그 구분이 안 돼 probe 를 따로 돌려야 했다.
        _best = contact.max(dim=1).values
        self.extras["contact/force_max"] = contact.max()
        self.extras["contact/force_best_finger"] = _best.mean()
        _g_thr = float(self.cfg.contact_force_threshold)
        self.extras["contact/n_over_thr"] = (contact > _g_thr).float().sum(dim=1).mean()
        self.extras["contact/n_over_tenth"] = (contact > _g_thr * 0.1).float().sum(dim=1).mean()
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
        # ★측정 공백 해소: grasp_v1 은 이 값을 보상에 쓰면서 로깅하지 않았다.
        self.extras["obj/xy_displacement"] = xy_disp.mean()
        self.extras["obj/xy_disp_p95"] = torch.quantile(xy_disp, 0.95)
        self.extras["obj/tilt_deg"] = tilt.mean()
        # Fabrics 추종 오차 — 정상상태 오차(78mm 이력)를 학습 중에도 본다.
        perr = (self.palm_targets[:, :3] - palm_pos).norm(dim=-1)
        self.extras["fabric/palm_err_mean"] = perr.mean()
        self.extras["fabric/palm_err_p95"] = torch.quantile(perr, 0.95)
        # ★★위 perr 은 **계산 오차 + 물리 추종**의 합이다. 둘은 성질이 완전히 달라
        #   섞어 보면 진단이 안 된다 — A 가 크면 fabric 이 목표를 못 푼 것(제어기
        #   설계 문제), B 가 크면 로봇이 계획을 못 따라간 것(관성·게인·토크).
        #   fabric 은 IK 솔버가 아니라 2 차 동역학계라 A 도 0 이 아니다.
        #   ★★_palm_frame(fabric taskmap) 출력은 **이미 로봇 베이스 기준**이다 — world
        #   로 보고 _local() 로 env_origins 를 빼면 43m 어긋난다(실측 43,285mm).
        #   판별 근거: perr(=|palm_targets − palm_pos|)가 26mm 로 성립하므로
        #   palm_targets 와 palm_pos 는 같은 env-local 계이고, 여기에 _local() 을 한 번
        #   더 먹인 _plan 만 튀었다. fabric 은 단일 로봇 모델을 전 env 가 공유한다.
        _plan, _ = self._palm_frame(self.fabric_q)
        self.extras["fabric/plan_err_mean"] = (
            self.palm_targets[:, :3] - _plan).norm(dim=-1).mean()      # A 정책목표→계획
        self.extras["fabric/track_err_mean"] = (
            _plan - palm_pos).norm(dim=-1).mean()                      # B 계획→실제
        # ★"정책이 컵을 겨냥은 하는가"와 "겨냥은 하는데 못 가는가"를 가른다.
        #   target_to_obj 작고 palm_dist 크면 → 추종/도달 문제
        #   target_to_obj 자체가 크면        → 정책이 컵을 안 겨냥 (탐색 문제)
        if self._hand_fabric:
            # ★손 제어를 두 층으로 갈라 잰다 — 어디서 막히는지 이 둘이 가른다.
            #   cmd_err : 정책이 지시한 관절 목표 → fabric attractor 가 실제로 만든
            #             `fabric_q`. 크면 **fabric 내부**가 목표를 못 따라간다
            #             (처방: hand_attractor_gain 상향).
            #   track_err: `fabric_q` → PhysX PD 가 실현한 실제 관절. 크면 **물리 쪽**
            #             (stiffness/effort)이 못 따라간다.
            #   손끝 IK 가 막혔던 곳이 정확히 cmd_err 에 해당한다(추종오차 85mm).
            _fq_hand = self.fabric_q[:, self.profile.num_arm_joints:]
            _cmd = self._fabric_hand_cmd
            self.extras["hand/cmd_err_rad"] = (_fq_hand - _cmd).abs().mean()
            # ★외전 이탈 — hand_control="pd" 는 외전을 init 에 **하드 고정**하지만
            #   "fabric" 은 fabric 이 손 20개를 전부 소유하므로 외전도 움직일 수 있다.
            #   목표는 init 으로 주지만 body_repulsion·joint_limit·cspace attractor 가
            #   함께 작용하면 밀려난다. 손가락이 벌어지면 감쌈이 약해진다
            #   (실측 fmin: fabric 27~28mm vs pd 23~24mm).
            _froz = self._frozen_t
            if _froz.numel() > 0:
                self.extras["hand/abduction_dev_rad"] = (
                    self.robot.data.joint_pos[:, _froz]
                    - self._default_q[:, _froz]).abs().mean()
            self.extras["hand/track_err_rad"] = (
                self.robot.data.joint_pos[:, self._hand_t]
                - _fq_hand[:, self._hand_from_fab]).abs().mean()
        if self._tip_ik:
            # ★손끝 IK 추종오차 — 액션 박스(축별 min/max 직육면체)는 실제 도달 집합보다
            #   크다. 이 값이 크면 정책이 **못 가는 곳**을 계속 지시하고 있다는 뜻이라
            #   `tip_action_span_frac` 을 줄일 근거가 된다. 실측: 도달 가능한 목표는 9mm,
            #   도달 불가 목표는 81~99mm 로 확연히 갈린다.
            _po, _R = self._palm_frame(self.fabric_q.detach())
            _rel = torch.einsum("bij,bkj->bki", _R.transpose(1, 2),
                                tips - _po[:, None, :])
            _want = self._tip_target.view(self.num_envs, -1, 3)
            _want = torch.einsum("bij,bkj->bki", _R.transpose(1, 2),
                                 _want - _po[:, None, :])
            # ★★이 값은 **실제 로봇 손끝** 기준이다(tips = body_pos_w). 정책 목표에서
            #   실제 손끝까지는 세 구간이 있고, 셋을 안 가르면 원인을 fabric 에
            #   잘못 씌운다(08.25 실제로 두 번 그렇게 진단했다):
            #     A fabric  : 정책 목표 → fabric_q 의 손끝 FK   (fabric IK 가 푸는가)
            #     B PD      : fabric_q → 실제 관절              (액추에이터가 따라가는가)
            #     C 접촉    : 나머지 — 물리로 막힌 몫(정상)
            self.extras["tip/target_err_mm"] = (_rel - _want).norm(dim=-1).mean() * 1000.0
            # A: fabric 이 계획한 손끝(같은 palm 좌표계) ↔ 정책 목표
            _tf, _ = self.fabric._fingertip_taskmap(self.fabric_q.detach(), None)
            _tf = _tf.view(self.num_envs, -1, 3)
            _relf = torch.einsum("bij,bkj->bki", _R.transpose(1, 2),
                                 _tf - _po[:, None, :])
            self.extras["tip/fabric_err_mm"] = (_relf - _want).norm(dim=-1).mean() * 1000.0
            # B: 실제 관절 ↔ fabric 계획 관절 [rad]
            _fq_h = self.fabric_q[:, self.profile.num_arm_joints:]
            self.extras["hand/track_err_rad"] = (
                self.robot.data.joint_pos[:, self._hand_t]
                - _fq_h[:, self._hand_from_fab]).abs().mean()
            # C: 손 토크 포화율 — effort_limit_sim=1.5Nm. 상한에 붙어 있으면 PD 가
            #    계획을 못 따라가는 것이 액추에이터 한계 탓이다(파지력 제어도 무효).
            _tau = self.robot.data.applied_torque[:, self._hand_t].abs()
            self.extras["hand/torque_sat_frac"] = (_tau > 1.425).float().mean()
        self.extras["fabric/target_to_obj"] = (
            self.palm_targets[:, :3] - obj_pos).norm(dim=-1).mean()
        tau = self.robot.data.applied_torque[:, self._hand_t].abs()
        self.extras["debug/hand/torque_mean"] = tau.mean()
        self.extras["debug/hand/torque_max"] = tau.max()
        self.extras.update(self.adr.log_dict())
        self.extras["adr/goal_xy_radius"] = self.adr.get_param("goal", "xy_radius")
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
                f" nF>thr={(contact > _g_thr).float().sum(dim=1).mean():.2f}"
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
        # 아니다.
        # ★08.22 리스폰 → **종료** 전환(grasp_v1 `_get_dones` 와 동일 구성:
        #   fallen | out_xy | tipped | 물리위반). 쓰러진/떨어진 컵을 방치하면 회복 불가
        #   상태의 전이가 배치를 희석하고 value 오차가 GAE 로 번진다.
        # ★감시: agn_test2 에서는 같은 종료가 "접근 회피"를 가르쳤다(reaching 0.056→0.005
        #   + episode_lengths 상승). 그 시그니처(approach↓ + eplen↑ 동시)가 보이면
        #   이 결정을 되짚는다. grasp_v1 은 같은 종료로 98% 까지 갔으므로 선례는 있다.
        qd = self.robot.data.joint_vel[:, self._arm_t]
        runaway = (qd.abs() > float(self.cfg.runaway_joint_vel)).any(dim=-1)

        obj = self._local(self.object.data.root_pos_w)
        fallen = obj[:, 2] < float(self.cfg.object_min_z)
        out_xy = (obj[:, :2] - self.object_spawn_pos[:, :2]).norm(dim=-1) > float(
            self.cfg.object_out_of_bounds_xy)
        tipped = self._object_tilt_deg() > float(self.cfg.tipping_termination_deg)

        terminated = runaway | fallen | out_xy | tipped
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["task/runaway_rate"] = runaway.float().mean()
        self.extras["term/fallen"] = fallen.float().mean()
        self.extras["term/out_xy"] = out_xy.float().mean()
        self.extras["term/tipped"] = tipped.float().mean()
        return terminated, truncated

    # ==================================================================
    def _reset_idx(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        p = self.profile

        # ★★KUKA 고정(08.25) `robot_spawn` ADR — 리셋 자세/속도에 노이즈를 준다.
        #   같은 홈에서만 출발하면 정책이 그 궤적 하나에 과적합된다(원본은 0 → 0.35 rad).
        q0 = self._default_q[env_ids].clone()
        _pn = float(self.adr.get_param("robot_spawn", "joint_pos_noise"))
        _vn = float(self.adr.get_param("robot_spawn", "joint_vel_noise"))
        qd0 = torch.zeros_like(q0)
        if _pn > 0.0:
            q0 = q0 + _pn * 2.0 * (torch.rand_like(q0) - 0.5)
            q0 = q0.clamp(self.robot.data.soft_joint_pos_limits[env_ids, :, 0],
                          self.robot.data.soft_joint_pos_limits[env_ids, :, 1])
        if _vn > 0.0:
            qd0 = _vn * 2.0 * (torch.rand_like(qd0) - 0.5)
        self.robot.write_joint_state_to_sim(q0, qd0, env_ids=env_ids)
        self.robot.set_joint_position_target(q0, env_ids=env_ids)
        # ★원본은 속도 목표도 리셋 상태로 되돌린다 — 안 하면 직전 에피소드 목표가 샌다.
        self.robot.set_joint_velocity_target(qd0, env_ids=env_ids)
        self.hand_targets[env_ids] = q0[
            :, (self._hand_t if (self._tip_ik or self._hand_fabric) else self._hand_free_t)]
        if self._tip_ik and getattr(self, "_tip_cmd", None) is not None:
            self._tip_cmd[env_ids] = self._tip_home          # 누산 지령도 홈으로
        # ★원본과 동일: fabric 상태를 리셋된 **실제** 관절 상태로 동기화한다.
        self.fabric_q[env_ids] = q0[:, self._fab_t]
        self.fabric_qd[env_ids] = qd0[:, self._fab_t]
        self.fabric_qdd[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._gate_hold[env_ids] = 0.0
        # ★슬루 지령도 홈으로 되돌린다 — 안 하면 이전 에피소드 지령이 새어 들어온다.
        self.palm_cmd[env_ids] = self.home_palm[env_ids]
        self._goal_reached_now[env_ids] = False

        rng = self.adr.get_param("spawn", "xy_range")
        offs = (torch.rand(n, 2, device=self.device) - 0.5) * 2.0 * rng
        spawn = torch.zeros(n, 3, device=self.device)
        spawn[:, 0] = p.object_spawn_center[0] + offs[:, 0]
        spawn[:, 1] = p.object_spawn_center[1] + offs[:, 1]
        spawn[:, 2] = self._object_rest_z[env_ids] + float(self.cfg.object_spawn_pad)
        self.object_spawn_pos[env_ids] = spawn
        # ---- goal: 스폰과 독립 오프셋 (반경은 ADR 축, 0 이면 구 고정 goal 과 동치) ----
        # ★스폰의 결정론적 함수로 두면 goal obs 가 변별력 없는 입력이 돼 정책이 무시한다
        #   (배포에서 사용자 지정 위치에 반응하지 않음). 이송 학습의 전제 조건.
        g_xy = self.adr.get_param("goal", "xy_radius")
        g_z = self.adr.get_param("goal", "z_radius")
        goal = spawn.clone()
        goal[:, 2] += float(self.cfg.goal_height_offset)
        goal[:, :2] += (torch.rand(n, 2, device=self.device) - 0.5) * 2.0 * g_xy
        goal[:, 2] += (torch.rand(n, device=self.device) - 0.5) * 2.0 * g_z
        # palm 박스 안쪽으로 클램프 — 박스 밖 goal 은 도달 불가(액션 포화 학습 재발 경로).
        _m = float(self.cfg.goal_box_margin)
        # palm_lo/hi 는 (1,6) — env 브로드캐스트로 클램프한다
        goal = torch.max(torch.min(goal, self.palm_hi[0, :3] - _m), self.palm_lo[0, :3] + _m)
        self.goal_pos[env_ids] = goal

        root = torch.zeros(n, 13, device=self.device)
        root[:, :3] = spawn + self.scene.env_origins[env_ids]
        root[:, 3] = 1.0
        # ★★KUKA 고정(08.25) `object_spawn.rotation` ADR — x·y 축 회전 노이즈(0 → 1 rad).
        #   자세가 항상 같으면 정책이 한 접근각에만 맞춘다. z 축은 컵이 회전대칭이라 뺀다.
        _rot = float(self.adr.get_param("spawn", "rotation"))
        if _rot > 0.0:
            _ax = torch.zeros(n, 3, device=self.device)
            _ax[:, :2] = _rot * 2.0 * (torch.rand(n, 2, device=self.device) - 0.5)
            _ang = _ax.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            root[:, 3:7] = torch.cat(
                [torch.cos(0.5 * _ang), (_ax / _ang) * torch.sin(0.5 * _ang)], dim=-1)
        self.object.write_root_state_to_sim(root, env_ids=env_ids)
