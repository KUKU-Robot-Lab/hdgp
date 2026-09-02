"""grasp_ua — 제자리 파지 → 리프트 → 목표 이송 → 정지.

제어 스택은 `grasp_ua_control.GraspUAControlMixin`(Fabrics 팔 + 시너지 손),
보상은 `grasp_ua_rewards`, 로봇 종속 정보는 `robot_profiles` 에 있다.

★액션 규약(grasp_v1 계승): palm 은 **홈 기준 델타**다 — `a=0` 이면 홈을 유지한다.
  절대 매핑(`a=0` = 박스 중심)은 σ=1.0 과 곱해지면 매 스텝 작업공간 전역에서 목표를
  재추첨해 접근이 랜덤워크가 된다(선행 트랙 실측).

★래치는 **보상 단계 표시 전용**이다. grasp_v1 은 래치 후 팔 지령을 z 램프 스크립트로
  대체했는데, 여기서는 그 오버라이드가 없다 — 이송까지 정책이 fabric 으로 제어한다.
"""

from __future__ import annotations

import math

import torch

from isaaclab.envs import DirectRLEnv

from .grasp_ua_control import GraspUAControlMixin
from .grasp_ua_env_cfg import GraspUAEnvCfg
from .grasp_ua_rewards import GRASP_UA_REWARD_TERMS, compute_grasp_ua_rewards
from .robot_profiles import PROFILES


class GraspUAEnv(GraspUAControlMixin, DirectRLEnv):
    cfg: GraspUAEnvCfg

    def __init__(self, cfg: GraspUAEnvCfg, render_mode: str | None = None, **kwargs):
        # ★★hydra 오버라이드는 `__post_init__` **뒤**에 `from_dict` 로 적용되고
        #   `__post_init__` 를 다시 부르지 않는다(IsaacLab `hydra_task_config` 실측).
        #   따라서 `env.object_bank=cup_family` 는 파생 구조(스폰 cfg·replicate_physics·
        #   접촉 필터·스폰고)에 **반영되지 않은 채** 런타임에만 보인다.
        #   그리고 `replicate_physics` 는 `InteractiveScene.__init__` 이 소비하므로
        #   `_setup_scene` 은 이미 늦다 — super() **전에** 재파생해야 한다.
        cfg.finalize_after_overrides()
        super().__init__(cfg, render_mode, **kwargs)
        self._init_task_state()

    def _init_task_state(self) -> None:
        """씬 이후의 태스크 상태 부트스트랩 전부 — `__init__` 꼬리를 그대로 추출했다.

        ★bimanual 폐루프 shim(`scripts/probes/bimanual_chain.py`)이 pour 씬 위에서
          이 메서드를 그대로 불러 같은 사슬(인덱스·fabric·시너지·앵커·박스)을 세운다.
          로직 복제를 금지하기 위한 추출이라, 여기가 바뀌면 shim 도 자동으로 따라간다.
        """
        p = PROFILES[self.cfg.profile_name]
        self.profile = p

        # ---- 조인트/바디 해석 (fail-loud: 프로필 선언 수와 대조) ---------------------
        self.arm_ids, arm_names = self.robot.find_joints(p.arm_joint_regex)
        self.hand_ids, hand_names = self.robot.find_joints(p.hand_joint_regex)
        if len(self.arm_ids) != p.num_arm_joints or len(self.hand_ids) != p.num_hand_joints:
            raise RuntimeError(
                f"[{p.name}] 프로필 조인트 수 불일치: arm {len(self.arm_ids)}"
                f"!={p.num_arm_joints} ({arm_names}), hand {len(self.hand_ids)}"
                f"!={p.num_hand_joints} ({hand_names})")
        self._arm_ids_t = torch.tensor(self.arm_ids, device=self.device, dtype=torch.long)
        self._hand_ids_t = torch.tensor(self.hand_ids, device=self.device, dtype=torch.long)

        palm_ids, _ = self.robot.find_bodies(p.palm_body)
        if len(palm_ids) != 1:
            raise RuntimeError(f"[{p.name}] palm_body '{p.palm_body}' 해석 실패: {palm_ids}")
        self.palm_idx = palm_ids[0]
        self.tip_ids = []
        for n in p.fingertip_bodies:
            ids, _ = self.robot.find_bodies(n)
            if len(ids) != 1:
                raise RuntimeError(f"[{p.name}] fingertip body '{n}' 해석 실패: {ids}")
            self.tip_ids.append(ids[0])
        self._tip_ids_t = torch.tensor(self.tip_ids, device=self.device, dtype=torch.long)
        # ★★손 **전 링크** 인덱스 — 바닥 벌점(`hand_floor`)의 기준이다.
        #   팁만 보면 안 된다: 09.01 실측에서 최하단은 `r_hl_pinky_tip` 이었지만
        #   손을 굽히면 중간마디가 더 내려갈 수 있다. 이름 접두사로 손 전체를 모은다.
        #   (palm 은 뺀다 — palm 원점은 손 최하단보다 5.0~5.7cm 위라 판정을 무디게 한다.)
        _side = p.hand_joint_names[0].split("_hj_")[0]
        _hb = [i for i, nm in enumerate(self.robot.data.body_names)
               if nm.startswith(f"{_side}_hl_") and "palm" not in nm]
        if not _hb:
            raise RuntimeError(f"[{p.name}] 손 링크 해석 실패 — 접두사 '{_side}_hl_'")
        self._hand_body_ids_t = torch.tensor(_hb, device=self.device, dtype=torch.long)
        # ---- 접촉 그룹 (프로필 정의) --------------------------------------------------
        fingers = list(p.finger_sensor_bodies.keys())
        self._finger_names = fingers
        # ★★중간마디(`_3`) 바디 인덱스 — `finger_closure_target="wrap"` 전용.
        #   08.31 실측: 팁 접촉은 0.65~0.85 로 이미 채워졌는데 wrap(중간∧원위)은
        #   전 8종에서 0.000 이다. 즉 손끝만 대고 감아 안지 않는다. 팁 기준 소등
        #   항은 이 지점에서 이미 꺼져 있어 경사를 못 준다 — 중간마디로 재야 한다.
        #   ★`_finger_names` 확정 **뒤**에 만들 것(순서 뒤집으면 부팅에서 죽는다).
        self._mid_ids_t = torch.tensor(
            [self.robot.find_bodies(p.finger_sensor_bodies[f][0])[0][0]
             for f in fingers],
            device=self.device, dtype=torch.long)
        if len(p.fingertip_bodies) != len(fingers):
            raise RuntimeError(
                f"[{p.name}] fingertip_bodies({len(p.fingertip_bodies)}) 와 "
                f"finger_sensor_bodies({len(fingers)}) 의 손가락 수가 달라 "
                "그룹 인덱스를 공유할 수 없다")
        self._group_a_idx = torch.tensor(
            [fingers.index(f) for f in p.contact_group_a],
            device=self.device, dtype=torch.long)
        if not p.envelope_fingers:
            raise RuntimeError(f"[{p.name}] envelope_fingers 미정의 — 감쌈 판정 불가")
        # 감쌈 분모 = 대향 그룹 반대편 ∩ 인벨롭 손가락(프로필이 도달 가능 집합을 정의).
        self._wrap_idx = torch.tensor(
            [i for i, f in enumerate(fingers)
             if f in p.contact_group_b and f in p.envelope_fingers],
            device=self.device, dtype=torch.long)
        if len(self._wrap_idx) < 1:
            raise RuntimeError(f"[{p.name}] contact_group_b ∩ envelope_fingers 가 비었다")
        # ★08.28 `surface_count` 감쌈용 — 대향 그룹의 반대편(4지 / jaw2).
        #   `_wrap_idx` 와 집합은 같지만 의미가 다르다: 저쪽은 "깊이 분모", 이쪽은
        #   "표면 그룹"이다. 신 정의는 여기에 `_group_a_idx`(엄지)와 손바닥을 더해
        #   **여섯 표면 전부**를 분모에 넣는다 — 구 정의는 엄지를 원리적으로 뺐다.
        self._group_b_idx = torch.tensor(
            [i for i, f in enumerate(fingers)
             if f in p.contact_group_b and f in p.envelope_fingers],
            device=self.device, dtype=torch.long)

        # ---- 포위도 키포인트 (08.28 신설) ---------------------------------------------
        # ★이 트랙은 지금까지 마디 링크의 **위치**를 한 번도 읽지 않았다 —
        #   `finger_sensor_bodies` 는 ContactSensor 만 만들고 힘만 읽었다.
        #   포위도는 접촉이 아니라 기하라서 마디 위치가 필요하다.
        # 손가락별로 프로필 튜플 순서를 보존한다(그룹별로 평균 내야 하므로).
        self._hull_ids: dict[str, list[int]] = {}
        for _f in fingers:
            _row = []
            for _b in p.finger_sensor_bodies[_f]:
                _ids, _ = self.robot.find_bodies(_b)
                if len(_ids) != 1:
                    raise RuntimeError(f"[{p.name}] 포위도 body '{_b}' 해석 실패: {_ids}")
                _row.append(_ids[0])
            self._hull_ids[_f] = _row
        # 그룹별 평면 인덱스 — 손가락마다 링크 수가 같다는 보장이 없어 펼친다.
        # 2지 그리퍼(손가락당 body 1개)에서도 그대로 성립한다.
        self._hull_a_t = torch.tensor(
            [i for _f in p.contact_group_a for i in self._hull_ids[_f]],
            device=self.device, dtype=torch.long)
        self._hull_b_t = torch.tensor(
            [i for _f in p.contact_group_b for i in self._hull_ids[_f]],
            device=self.device, dtype=torch.long)
        # ★손가락별 최소참여용 — 위와 같은 body 를 **손가락 축을 살려** (F_b, L) 로 둔다.
        #   평평하게 편 `_hull_b_t` 로는 "어느 손가락이 빠졌는지"를 잴 수 없다.
        _rows = [self._hull_ids[_f] for _f in p.contact_group_b]
        _l = {len(r) for r in _rows}
        if len(_l) != 1:
            raise RuntimeError(
                f"[{p.name}] contact_group_b 손가락별 링크 수가 다르다: "
                f"{[(f, len(self._hull_ids[f])) for f in p.contact_group_b]} — "
                "최소참여 계산이 손가락을 정렬할 수 없다")
        self._hull_part_t = torch.tensor(_rows, device=self.device, dtype=torch.long)

        # ---- 물체 뱅크: env 별 원점 오프셋 (08.29 신설) -------------------------------
        # ★배정은 `env_id % N` 결정론이고 MultiAssetSpawner(random_choice=False)와 같은
        #   규약이다. 이 값은 **스폰 높이·정착고·목표**에만 쓴다 — obs 에는 넣지 않는다
        #   (물체 정체성 금지). 단일 뱅크면 전 env 동일 상수라 현행과 항등이다.
        from openarm.agnostic.modules import object_bank as _ob

        _bank = _ob.get(self.cfg.object_bank)
        _off_of = [s.origin_offset_z for s in _bank.specs]
        self._obj_origin_off = torch.tensor(
            [_off_of[i] for i in _bank.assign_indices(self.num_envs)],
            device=self.device, dtype=torch.float32)
        # ---- 종별 진단 (08.29 신설) — 집계 success 는 종별 실패를 가린다(사용자 지적).
        #   ★진단 전용: obs 금지(정체성 계약)·보상 미사용. 로깅은 extras 로만.
        self._species_ids = torch.tensor(
            _bank.assign_indices(self.num_envs), device=self.device,
            dtype=torch.long)
        self._species_names = [s.id for s in _bank.specs]
        self._n_species = len(_bank.specs)
        self._species_succ_ema = torch.zeros(self._n_species, device=self.device)
        self._species_latch_ema = torch.zeros(self._n_species, device=self.device)

        # ---- 손등 접촉 배제 (08.28 신설 — Hu et al. 의 `p_collision` 대응) -------------
        # ★프로필 미정의는 fail-loud. 기본축을 가정하면 판정이 **조용히 뒤집혀**
        #   손등 파지를 감쌈으로 계속 센다(자매 트랙의 명시 경고).
        # ★`palmar_axis_local` 은 **손가락 마디 링크**의 로컬 축이지 palm body 의
        #   법선이 아니다. 손바닥 법선은 `palm_ee +x`(`_palm_ee_R()` 열 0)로 별개다.
        #   두 규약을 섞으면 조용히 반대 판정이 된다.
        self._palmar_axes = None
        if bool(self.cfg.require_palmar_contact):
            _missing = [f for f in fingers if f not in p.palmar_axis_local]
            if _missing:
                raise RuntimeError(
                    f"[{p.name}] palmar_axis_local 미정의: {_missing} — 손바닥/손등 "
                    "구분 불가. URDF 의 cross(굴곡축, 장축)으로 실측하거나 "
                    "cfg.require_palmar_contact=False 로 구 판정(크기만)을 명시할 것")
            self._palmar_axes = torch.tensor(
                [p.palmar_axis_local[f] for f in fingers],
                device=self.device, dtype=torch.float32)          # (F, 3)
            # 센서 튜플과 **같은 순서**의 마디 body id — 힘 배열과 축을 맞춰야 한다.
            self._palmar_body_ids = torch.tensor(
                [self._hull_ids[f] for f in fingers],
                device=self.device, dtype=torch.long)             # (F, L)

        # ---- 팔·손 제어 배선 ----------------------------------------------------------
        self._policy_dt = float(self.cfg.sim.dt) * int(self.cfg.decimation)
        self._setup_fabrics()

        # ---- 버퍼 ---------------------------------------------------------------------
        jl = self.robot.data.soft_joint_pos_limits           # (N, J, 2)
        self._arm_lo = jl[:, self._arm_ids_t, 0]
        self._arm_hi = jl[:, self._arm_ids_t, 1]
        self._default_q = self.robot.data.default_joint_pos.clone()
        # ★★자산 뷰 정합 검사 — `replicate_physics=False` 에서 리셋의
        #   `write_joint_state_to_sim` 이 반영되지 않는 원인을 좁히기 위한 계측.
        #   뷰 인스턴스 수가 num_envs 와 다르면 `env_ids` 기입이 조용히 어긋난다.
        _ni = int(getattr(self.robot, "num_instances", -1))
        _dq = self._default_q
        print(f"[grasp_ua] 로봇 뷰: num_instances={_ni} (num_envs={self.num_envs}) · "
              f"default_joint_pos {tuple(_dq.shape)} "
              f"min={float(_dq.min()):.4f} max={float(_dq.max()):.4f} "
              f"env간 산포={float(_dq.std(dim=0).max()):.6f} · "
              f"joint_names={len(self.robot.data.joint_names)}", flush=True)
        if _ni != self.num_envs:
            raise RuntimeError(
                f"[grasp_ua] 로봇 아티큘레이션 뷰가 {_ni}개인데 env 는 "
                f"{self.num_envs}개다 — 리셋 기입이 어긋난다")
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)       # env-local
        self.object_spawn_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # 지령 리미터 상태 — 리셋 직후 첫 지령은 "변화"가 아니라 초기화라 안 건다.
        self._prev_palm_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_palm_cmd_rot = torch.zeros(self.num_envs, 3, device=self.device)
        self._palm_cmd_primed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._palm_cmd_step_raw = torch.zeros(self.num_envs, device=self.device)
        # 진단 전용 — 지령이 **박스**에 잘렸는지 / **리미터**에 잘렸는지. 둘은 원인이
        # 다르다: 박스 포화는 도달영역이 부족한 것이고 리미터 포화는 너무 빨리 움직이려는
        # 것이다. 축별로 봐야 어느 축이 부족한지 알 수 있다.
        self._palm_cmd_box_sat = torch.zeros(self.num_envs, 3, device=self.device)
        self._palm_cmd_rate_sat = torch.zeros(self.num_envs, device=self.device)

        # 닫기 게이트 상태(정렬도) — `_pre_physics_step` 이 매 스텝 갱신한다.
        self._close_gate = torch.ones(self.num_envs, device=self.device)
        self._cage_ctr_dist = torch.zeros(self.num_envs, device=self.device)
        # palm 프레임 분해 거리·속도(로깅용) — `_get_rewards` 가 매 스텝 갱신한다.
        self._palm_normal_dist = torch.zeros(self.num_envs, device=self.device)
        self._palm_lateral_dist = torch.zeros(self.num_envs, device=self.device)
        self._palm_speed = torch.zeros(self.num_envs, device=self.device)
        # 케이지 중심의 palm-local 오프셋 — `_report_home_cage` 가 홈에서 실측해 고정한다.
        self._cage_offset_palm = torch.zeros(3, device=self.device)

        # 래치 (보상 단계 표시 전용)
        self._latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._wrap_at_latch = torch.zeros(self.num_envs, device=self.device)
        self._disp_at_latch = torch.zeros(self.num_envs, device=self.device)
        # ★래치 순간의 palm 프레임 물체 위치 `Rᵀ(obj−palm)`. `obs_object_rigid_after_latch`
        #   가 이걸 굴려 "손에 가려 안 보이는 컵"을 추정한다. 위치만 있으면 충분하다 —
        #   obs 의 물체 항 3개(palm_to_obj·obj_to_tips·goal_rel)가 전부 위치다.
        self._obj_off_palm = torch.zeros(self.num_envs, 3, device=self.device)

        # 판정 버퍼 — `_get_dones` 가 먼저 돌고 `_get_rewards` 가 같은 스텝에 재사용한다.
        self._tilt_deg = torch.zeros(self.num_envs, device=self.device)
        self._abnormal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stay_run = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 단계 도달 플래그 — 에피소드 동안 OR 누적, 리셋에서만 평균 기록(스텝 비용 0).
        self._stage_names = ("grasp", "lift", "transfer", "stay")
        self._stage_hit = torch.zeros(
            self.num_envs, len(self._stage_names), dtype=torch.bool, device=self.device)

        self._init_home_palm()
        self._report_home_cage()
        self._assert_goal_reachable()
        self._setup_cmd_markers()

        # 액션 델타 박스 — palm 은 **앵커** 기준 상대다.
        _d = torch.tensor(self.cfg.palm_delta_xyz, device=self.device)
        _r = math.radians(float(self.cfg.palm_delta_rot_deg))
        self._delta_lo = torch.cat([-_d, torch.full((3,), -_r, device=self.device)])
        self._delta_hi = torch.cat([_d, torch.full((3,), _r, device=self.device)])
        # ★앵커는 항상 도달 가능해야 한다(박스가 앵커를 잘라내면 a=0 의 의미가 깨진다).
        self._box_lo = torch.minimum(self._palm_lo, self._home_palm)
        self._box_hi = torch.maximum(self._palm_hi, self._home_palm)
        self._setup_palm_anchor()

        # ---- ADR 상태 — 전역 level 하나. OFF(기본)면 전부 base 값 = 현행 항등. --------
        self._palm_delta_base = _d.clone()
        # env 별 목표 오프셋 — 샘플링 ON 이면 리셋마다 새로 뽑고, OFF 면 전 env 동일.
        self._goal_off_env = torch.zeros(self.num_envs, 3, device=self.device)
        self._adr_level = 0.0
        self._adr_succ = 0
        self._adr_epis = 0
        self._assert_adr_monotonic()
        self._adr_apply()

        print(f"[grasp_ua] profile={p.name} arm={len(self.arm_ids)} "
              f"hand={len(self.hand_ids)} tips={len(self.tip_ids)} "
              f"action={self.cfg.action_space} obs={self.cfg.observation_space} "
              f"state={self.cfg.state_space} fabric={p.fabric_robot_dir}", flush=True)
        # ★★s2r 정합 상태를 **매 학습 로그에 남긴다**. 이 셋은 전부 "조용히 틀릴 수
        #   있는" 축이라(게인은 환경변수, DR 대상·마찰은 cfg 단계에서만 확정된다)
        #   나중에 로그만 보고도 어느 조합으로 돌았는지 알 수 있어야 한다.
        _specs = getattr(p, "actuator_specs", {})
        _real = "right_arm_j1" in _specs
        _hs = next((v for k, v in _specs.items() if "hand" in k), {})
        _hand_g = f"kp{_hs.get('stiffness', '?')}/kd{_hs.get('damping', '?')}"
        _em = getattr(self, "event_manager", None)
        _dr = "(events 꺼짐)"
        if _em is not None:
            try:
                _t = _em.get_term_cfg("robot_joint_stiffness_and_damping")
                _dr = str(_t.params["asset_cfg"].joint_names)
            except Exception:                      # noqa: BLE001
                _dr = "(조회 실패)"
        print(f"[grasp_ua][s2r] 팔게인={'r2s 정합' if _real else 'KUKA 기본'}"
              f"(cfg use_real_gains={bool(self.cfg.use_real_gains)}) · "
              f"손게인={_hand_g} · "
              f"게인DR대상={_dr} · 마찰범위={tuple(self.cfg.object_friction_range)} · "
              f"로봇중력={'ON' if not self.cfg.robot_cfg.spawn.rigid_props.disable_gravity else 'OFF'}",
              flush=True)

    # ------------------------------------------------------------------
    def _report_home_cage(self) -> None:
        """홈 자세의 **케이지 중심**(엄지 팁 ↔ 4지 팁 중점)이 컵 대비 어디인지 실측·보고.

        ★★이 수치가 접근 난이도를 결정한다. 케이지가 컵보다 **앞(+x)** 에 있으면 정책은
          팔을 **후진**시켰다가 다시 들어가야 하고, 3D 대각선 이동이라 훨씬 어렵다.
          케이지가 컵보다 **위** 에 있으면 지령이 계속 아래로 가면서 엄지가 컵에 걸리고,
          걸린 채 눌리다 풀리면 손이 테이블까지 내려간다(둘 다 사용자 GUI 관찰).
        ★좌팔 그리퍼 트랙 결론: "컵을 앞에 둔다"와 "홈을 뒤로 물린다"는 로봇 기준
          상대 배치가 같아 물리적으로 동등하다 — 그쪽은 스폰을 앞으로 밀어 해결했다.
        """
        p = self.profile
        tips = (self.robot.data.body_pos_w[:, self._tip_ids_t]
                - self.scene.env_origins[:, None, :])[0]
        _a = int(self._group_a_idx[0])
        _others = [i for i in range(len(self.tip_ids)) if i != _a]
        cage = 0.5 * (tips[_a] + tips[_others].mean(dim=0))
        r_cage = 0.5 * float((tips[_a] - tips[_others].mean(dim=0)).norm())
        self._r_cage = r_cage          # 닫기 게이트 임계로도 쓴다

        # ★★케이지 중심을 **palm 에 강체로 붙인다**. 홈 자세에서 한 번 재고 그 뒤로는
        #   손가락이 어떻게 움직이든 이 오프셋이 변하지 않는다.
        #   08.27 실측(s2r_a6): 중심을 실시간 손끝으로 두면 팔이 정지한 구간
        #   (palm_to_cup 0.120~0.140, n=147)에서 corr(syn_close, cage_dist) = −0.974 —
        #   **팔을 안 움직이고 손만 오므려도 중심이 컵 쪽으로 50mm 당겨져** 게이트가
        #   저절로 열렸다(램프 폭 60mm 의 83%). "정렬되면 닫아라"가 아니라
        #   "닫으면 닫아도 된다"는 양의 되먹임이라 게이트가 아무것도 막지 못했다.
        _palm = (self.robot.data.body_pos_w[:, self.palm_idx]
                 - self.scene.env_origins)[0]
        _R = self._palm_ee_R()[0]                       # (3,3), 열 = palm 축
        self._cage_offset_palm = _R.transpose(0, 1) @ (cage - _palm)
        cup = [p.object_spawn_center[0], p.object_spawn_center[1],
               float(self.cfg.table_surface_z) + float(self.cfg.object_origin_offset_z)
               + float(self.cfg.object_grasp_z_offset)]
        d = [round(float(cage[i]) - cup[i], 4) for i in range(3)]
        print(f"[grasp_ua] 홈 케이지 중심={[round(float(v), 4) for v in cage]} "
              f"· 반경 {r_cage * 1000:.0f}mm | 컵 파지중심={[round(v, 4) for v in cup]} "
              f"| 케이지−컵 = {d} m", flush=True)
        # 전진축 정렬 허용오차 — 케이지 반경의 1/6(20mm) 안이면 후진이 필요 없다.
        if d[0] > 0.02:
            print(f"[grasp_ua] ⚠ 케이지가 컵보다 {d[0] * 1000:.0f}mm **앞(+x)** 이다 — "
                  "정책이 후진 후 재접근해야 한다(3D 대각선). 컵 스폰을 앞으로 밀거나 "
                  "홈을 뒤로 물릴 것.", flush=True)
        # 접근 간격이 케이지 반경보다 좁으면 리셋 순간 손가락이 컵을 관통한다.
        _gap_xy = float((cage[:2] - torch.tensor(cup[:2], device=cage.device)).norm())
        if _gap_xy < r_cage:
            print(f"[grasp_ua] ⚠ 홈 케이지↔컵 수평 간격 {_gap_xy * 1000:.0f}mm 가 "
                  f"케이지 반경 {r_cage * 1000:.0f}mm 보다 좁다 — 리셋에서 관통 위험.",
                  flush=True)

    # ------------------------------------------------------------------
    def _setup_palm_anchor(self) -> None:
        """액션 원점(`a=0` 이 뜻하는 palm 자세)을 결정하고 부팅에서 검증한다.

        ★`home` 은 팔 **rest 자세**이고 과제 위치가 아니다. 08.29 Phase 0 실측으로
          그 격차가 확인됐다 — 홈(0.280, -0.380, 0.418)은 컵 파지고보다 z +14 cm,
          이송 목표보다 y -27 cm 다. `a=0` 이 "컵에서 도망"을 뜻하니 정책은 액션을
          상시 만재(√6 의 93~98%)로 밀어 저항해야 하고, 출력이 조금만 풀리면 palm 이
          홈으로 튕긴다. 실측 `palm_post_latch_y` -0.399 는 **홈보다도 뒤**다.
        ★`spawn` 은 리셋 시 정착 스냅샷 기준이라 **에피소드 내 상수**다. 실시간 물체
          위치를 앵커에 쓰면 컵이 밀릴 때 액션 원점이 따라가는 되먹임이 된다.
        """
        p = self.profile
        self._anchor_mode = str(self.cfg.palm_anchor_mode)
        if self._anchor_mode not in ("home", "spawn"):
            raise RuntimeError(
                f"[{p.name}] palm_anchor_mode={self._anchor_mode!r} 는 "
                "'home' 또는 'spawn' 이어야 한다")
        self._anchor_off = torch.tensor(
            self.cfg.palm_anchor_offset_xyz, device=self.device, dtype=torch.float32)
        if self._anchor_mode == "home":
            print(f"[grasp_ua] 액션 앵커 = 홈 "
                  f"{[round(v, 4) for v in self._home_palm[:3].tolist()]}", flush=True)
            return
        # 스폰 중심 기준 앵커를 부팅에서 한 번 검증한다 — 오프셋 오타를 여기서 잡는다.
        # ★다물체면 원점 오프셋이 물체마다 다르므로 **최저·최고 둘 다** 검사한다.
        #   최댓값만 보면 가장 낮은 컵에서 앵커가 박스 밖으로 나가는 것을 놓친다.
        _z0 = float(self._obj_origin_off.min())
        _z1 = float(self._obj_origin_off.max())
        _lo, _hi = self._palm_lo[:3], self._palm_hi[:3]
        _a = None
        for _z in (_z0, _z1):
            _spawn = torch.tensor(
                [p.object_spawn_center[0], p.object_spawn_center[1],
                 float(self.cfg.table_surface_z) + _z],
                device=self.device, dtype=torch.float32)
            _a = _spawn + self._anchor_off - self._fab_to_env
            if bool(((_a < _lo) | (_a > _hi)).any()):
                raise RuntimeError(
                    f"[{p.name}] 액션 앵커 {[round(v, 4) for v in _a.tolist()]} "
                    f"(원점 오프셋 {_z:.4f}) 가 palm 박스 "
                    f"{[round(v, 3) for v in _lo.tolist()]}~"
                    f"{[round(v, 3) for v in _hi.tolist()]} 밖이다 — "
                    "palm_anchor_offset_xyz 를 고치거나 프로필 박스를 넓혀라.")
        # 앵커±델타가 박스를 넘으면 그 축은 상시 클램프된다(구 홈 규약의 y 92% 포화가
        # 정확히 그것이었다). 잘리는 축을 부팅에서 이름으로 알려준다.
        _cut = [ax for i, ax in enumerate("xyz")
                if float(_a[i] - self._delta_lo[i].abs()) < float(_lo[i])
                or float(_a[i] + self._delta_hi[i]) > float(_hi[i])]
        print(f"[grasp_ua] 액션 앵커 = 스폰 기준 "
              f"{[round(v, 4) for v in _a.tolist()]} "
              f"(스폰 {[round(v, 4) for v in _spawn.tolist()]} + "
              f"{list(self.cfg.palm_anchor_offset_xyz)}) · "
              f"델타 ±{list(self.cfg.palm_delta_xyz)}"
              + (f" · ⚠박스에 잘리는 축 {_cut}" if _cut else ""), flush=True)

    # ------------------------------------------------------------------
    def _palm_anchor(self) -> torch.Tensor:
        """액션 원점 (N, 6). 회전 성분은 홈 그대로 두고 **위치만** 재중심한다."""
        _home = self._home_palm.unsqueeze(0).expand(self.num_envs, 6)
        if self._anchor_mode == "home":
            return _home
        # `object_spawn_pos` 는 env-local(정착 스냅샷) — palm_targets 는 fabric 프레임.
        _pos = self.object_spawn_pos + self._anchor_off - self._fab_to_env
        # 첫 리셋 전에는 스폰이 0 이다. 그때만 홈으로 대체한다(조용한 0 앵커 방지).
        _unset = (self.object_spawn_pos.abs().sum(dim=1) < 1e-6).unsqueeze(1)
        _pos = torch.where(_unset, _home[:, :3], _pos)
        return torch.cat([_pos, _home[:, 3:]], dim=1)

    # ------------------------------------------------------------------
    def _adr_apply(self) -> None:
        """level → 실효값 재계산. 승급 시에만 불린다(스텝 경로 비용 0).

        ★축③ 제약: 목표 y 확장분만큼 **델타 박스 y 를 같이 키운다** — palm 지령 박스
          (base ±0.10)가 이송 거리를 물리적으로 막는 구조라, 목표만 늘리면 정책이
          도달 불가능한 과제를 받는다.
        ★enable_adr=False 면 level 을 0 으로 강제해 전부 base 값 = 현행 항등이다.
        """
        cfgn = self.cfg
        lvl = float(self._adr_level) if bool(cfgn.enable_adr) else 0.0
        _base_rng = float(cfgn.spawn_range)
        self._adr_spawn_range = _base_rng + lvl * (
            float(cfgn.adr_spawn_range_max) - _base_rng)
        _bx = float(cfgn.goal_offset_xyz[0])
        _by = float(cfgn.goal_offset_xyz[1])
        _bz = float(cfgn.goal_offset_xyz[2])
        _sign = 1.0 if _by >= 0.0 else -1.0
        _y_eff = _sign * (abs(_by) + lvl * (float(cfgn.adr_goal_y_max) - abs(_by)))
        # 3축 확장 — x 는 ±반범위, z 는 [base, max]. 둘 다 기본값이면 0 폭이라 항등.
        _x_eff = lvl * float(getattr(cfgn, "adr_goal_x_max", 0.0))
        _z_eff = _bz + lvl * (float(getattr(cfgn, "adr_goal_z_max", _bz)) - _bz)
        self._adr_goal_base = (_bx, _by, _bz, _sign)
        self._adr_goal_span = (_x_eff, _y_eff, _z_eff)
        # ★`_adr_goal_offset` 은 **최대 코너**다(샘플링 OFF 면 곧 실효값).
        self._adr_goal_offset = torch.tensor(
            [_bx, _y_eff, _bz], device=self.device)
        _d_eff = self._palm_delta_base.clone()
        _d_eff[0] = _d_eff[0] + _x_eff
        _d_eff[1] = _d_eff[1] + (abs(_y_eff) - abs(_by))
        _d_eff[2] = _d_eff[2] + (_z_eff - _bz)
        _r = math.radians(float(cfgn.palm_delta_rot_deg))
        self._delta_lo = torch.cat(
            [-_d_eff, torch.full((3,), -_r, device=self.device)])
        self._delta_hi = torch.cat(
            [_d_eff, torch.full((3,), _r, device=self.device)])
        # ★손 잔차(다섯째 축) — "쥐는 법 먼저, dexterity 나중". max<=base 면 축 꺼짐.
        _rb = float(getattr(cfgn, "finger_residual_scale", 0.0))
        _rm = float(getattr(cfgn, "adr_finger_residual_max", 0.0))
        self._adr_residual = _rb + lvl * max(0.0, _rm - _rb)
        _bn = float(cfgn.obs_noise_object)
        self._adr_obs_noise_object = _bn + lvl * (
            float(cfgn.adr_obs_noise_object_max) - _bn)
        # ---- sim2real 축 (09.01) — 관절 상태 노이즈 ----------------------------------
        _bq = float(cfgn.obs_noise_qpos)
        self._adr_obs_noise_qpos = _bq + lvl * (
            float(getattr(cfgn, "adr_obs_noise_qpos_max", _bq)) - _bq)
        _bv = float(cfgn.obs_noise_qvel)
        self._adr_obs_noise_qvel = _bv + lvl * (
            float(getattr(cfgn, "adr_obs_noise_qvel_max", _bv)) - _bv)
        self._adr_apply_physics(lvl)

    # ------------------------------------------------------------------
    @staticmethod
    def _lerp_range(terminal, lvl: float) -> tuple[float, float]:
        """(1,1) → terminal 을 level 로 선형 보간. 순수 함수 — 시뮬 없이 테스트한다.

        이식 출처: `tesollo/right/grasp_v2/grasp_adr.py:118-135 _expand_physics_ranges`.
        """
        lo, hi = float(terminal[0]), float(terminal[1])
        return (1.0 + lvl * (lo - 1.0), 1.0 + lvl * (hi - 1.0))

    def _adr_apply_physics(self, lvl: float) -> None:
        """물리 DR 범위를 level 로 넓힌다. 종점 (1,1) 이면 전부 항등이라 no-op.

        ★★반드시 `self.event_manager` 를 고친다. `self.cfg.events` 는 ManagerBase 가
          **deepcopy** 해 갔으므로 그쪽을 고치면 조용히 아무 일도 일어나지 않는다.

        ★★마찰(`*_material`)은 **여기 없다**. `randomize_rigid_body_material` 은
          `material_buckets` 를 term 인스턴스 생성 시 1회만 샘플링하고 `__call__` 은
          그 고정 버킷에서 뽑기만 해서 런타임 확장이 **무증상 no-op** 이다(자매
          `grasp_adr.py` 가 재질을 확장하지만 실제 물리는 안 바뀐다). 마찰은
          `object_friction_range` 로 cfg 단계에서 고정 범위를 연다.
        """
        em = getattr(self, "event_manager", None)
        if em is None:                      # enable_events=False 면 속성 자체가 없다
            return
        _m = self._lerp_range(
            getattr(self.cfg, "adr_mass_scale_max", (1.0, 1.0)), lvl)
        em.get_term_cfg("object_scale_mass").params[
            "mass_distribution_params"] = _m
        _g = self._lerp_range(
            getattr(self.cfg, "adr_joint_gain_scale_max", (1.0, 1.0)), lvl)
        _gt = em.get_term_cfg("robot_joint_stiffness_and_damping")
        _gt.params["stiffness_distribution_params"] = _g
        _gt.params["damping_distribution_params"] = _g
        self._adr_mass_range = _m
        self._adr_gain_range = _g

    # ------------------------------------------------------------------
    def _assert_adr_monotonic(self) -> None:
        """ADR 축은 **base → max 로 단조 증가**여야 한다. 아니면 부팅에서 죽인다.

        ★이 가드가 없어서 `adr_goal_z_max=0.08 < base 0.12` 가 조용히 살아 있었다 —
          승급할수록 목표가 **낮아지는**(쉬워지는) 역방향 축이었다. 같은 함정을
          obs 노이즈 축 주석이 이미 경고하고 있었는데도 z 축에서 재발했으므로,
          주석이 아니라 코드로 막는다.
        """
        c = self.cfg
        _bz = abs(float(c.goal_offset_xyz[2]))
        for name, base, mx in (
            ("goal_z", _bz, abs(float(c.adr_goal_z_max))),
            ("obs_noise_object", float(c.obs_noise_object),
             float(c.adr_obs_noise_object_max)),
            ("obs_noise_qpos", float(c.obs_noise_qpos),
             float(getattr(c, "adr_obs_noise_qpos_max", c.obs_noise_qpos))),
            ("obs_noise_qvel", float(c.obs_noise_qvel),
             float(getattr(c, "adr_obs_noise_qvel_max", c.obs_noise_qvel))),
            ("spawn_range", float(c.spawn_range), float(c.adr_spawn_range_max)),
        ):
            if mx < base:
                raise RuntimeError(
                    f"[grasp_ua][ADR] {name}: max({mx}) < base({base}) — "
                    "승급할수록 쉬워지는 역방향 축이다. max 를 base 이상으로 올려라.")

    # ------------------------------------------------------------------
    def _assert_goal_reachable(self) -> None:
        """목표가 palm 박스 안인지 부팅에서 확인 — 밖이면 과제가 성립하지 않는다."""
        p = self.profile
        lo = self._palm_lo[:3].tolist()
        hi = self._palm_hi[:3].tolist()
        # ★다물체면 정착고가 물체마다 다르다 — **양 극단 모두** 도달 가능해야 한다.
        _zs = sorted({float(self._obj_origin_off.min()),
                      float(self._obj_origin_off.max())})
        for _z in _zs:
            settled_z = float(self.cfg.table_surface_z) + _z
            goal = [
                p.object_spawn_center[0] + self.cfg.goal_offset_xyz[0],
                p.object_spawn_center[1] + self.cfg.goal_offset_xyz[1],
                settled_z + self.cfg.goal_offset_xyz[2],
            ]
            if any(g < lo[i] or g > hi[i] for i, g in enumerate(goal)):
                raise RuntimeError(
                    f"[{p.name}] 이송 목표 {[round(v, 3) for v in goal]} "
                    f"(원점 오프셋 {_z:.4f}) 가 palm 박스 "
                    f"{[round(v, 3) for v in lo]}~{[round(v, 3) for v in hi]} 밖이다 — "
                    "goal_offset_xyz 를 줄이거나 프로필 박스를 넓혀라.")
            print(f"[grasp_ua] 이송 목표 = {[round(v, 3) for v in goal]} "
                  f"(정착고 {settled_z:.4f} = 표면 {self.cfg.table_surface_z} + 원점 "
                  f"{_z:.4f} · offset {list(self.cfg.goal_offset_xyz)})", flush=True)
        # ★ADR 이면 **최대 난이도**(goal_y_max + 스폰 코너)도 부팅에서 검증한다 —
        #   런타임 승급 후에 목표가 박스 밖으로 나가면 소리 없이 과제가 죽는다.
        #   ★★기준은 **비확장 프로필 박스**다 — 런타임 최종 클램프(`_box_lo/_box_hi`)는
        #   ADR 로 늘지 않는다(프로필 박스 = 도달영역 실측이라 물리 한계). 델타 확장은
        #   앵커↔목표 span 용이지 도달영역 확장이 아니다.
        if bool(self.cfg.enable_adr):
            _by = float(self.cfg.goal_offset_xyz[1])
            _sign = 1.0 if _by >= 0.0 else -1.0
            _y_max = _sign * float(self.cfg.adr_goal_y_max)
            _rng = float(self.cfg.adr_spawn_range_max)
            # ★3축 확장이면 x·z 극단도 함께 검사한다 — 한 축만 보면 승급 뒤에
            #   목표가 조용히 박스 밖으로 나간다.
            _gx = float(getattr(self.cfg, "adr_goal_x_max", 0.0))
            _gz = float(getattr(self.cfg, "adr_goal_z_max",
                                self.cfg.goal_offset_xyz[2]))
            for _z in _zs:
              for _ox in ((-_gx, _gx) if _gx > 0.0 else (0.0,)):
                for _oz in {float(self.cfg.goal_offset_xyz[2]), _gz}:
                  for _dx in (-_rng, _rng):
                    for _dy in (-_rng, _rng):
                        goal = [
                            p.object_spawn_center[0] + _dx
                            + self.cfg.goal_offset_xyz[0] + _ox,
                            p.object_spawn_center[1] + _dy + _y_max,
                            float(self.cfg.table_surface_z) + _z + _oz,
                        ]
                        if any(g < lo[i] or g > hi[i]
                               for i, g in enumerate(goal)):
                            raise RuntimeError(
                                f"[{p.name}] ADR 최대 목표 "
                                f"{[round(v, 3) for v in goal]} 가 프로필 박스 "
                                f"{[round(v, 3) for v in lo]}~"
                                f"{[round(v, 3) for v in hi]} 밖이다 — "
                                "adr_goal_y_max/adr_spawn_range_max 를 줄여라.")
            print(f"[grasp_ua][ADR] 최대 난이도 검증 통과: goal_y {_y_max:+.3f} · "
                  f"spawn ±{_rng:.3f}", flush=True)

    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)

        # ---- 팔: palm 6D = **앵커 + 델타** -----------------------------------------
        # a=0 → 앵커. 탐색이 앵커 주변 유계 오프셋으로 묶여 절대 매핑의 랜덤워크가 없다.
        # ★앵커는 에피소드 내 상수다(홈 또는 스폰 스냅샷) — 실시간 물체·래치를 쓰면
        #   액션 원점이 에피소드 중 움직여 되먹임/스크립트가 된다. `_palm_anchor` 참조.
        delta = 0.5 * (self.actions[:, :6] + 1.0) * (self._delta_hi - self._delta_lo) \
            + self._delta_lo
        _raw_targets = self._palm_anchor() + delta
        self.palm_targets = _raw_targets.clamp(self._box_lo, self._box_hi)
        # 축별 박스 포화 — 클램프가 값을 바꿨으면 그 축의 도달영역이 부족한 것이다.
        self._palm_cmd_box_sat = (
            self.palm_targets[:, :3] != _raw_targets[:, :3]).float()

        # ---- 지령 변화율 리미터 -----------------------------------------------------
        _lim = float(self.cfg.palm_cmd_rate_limit_m)
        _step3 = self.palm_targets[:, :3] - self._prev_palm_cmd
        # 클램프 **전** 원값 로깅 — 상한이 물리는 비율의 유일한 근거다.
        self._palm_cmd_step_raw = torch.where(
            self._palm_cmd_primed, _step3.norm(dim=-1),
            torch.zeros_like(self._palm_cmd_step_raw))
        if _lim > 0.0:
            _scale = (_lim / _step3.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
            self._palm_cmd_rate_sat = (
                (_scale.squeeze(-1) < 1.0) & self._palm_cmd_primed).float()
            self.palm_targets[:, :3] = torch.where(
                self._palm_cmd_primed.unsqueeze(-1),
                self._prev_palm_cmd + _step3 * _scale,
                self.palm_targets[:, :3])
        self._prev_palm_cmd = self.palm_targets[:, :3].clone()

        _lim_r = math.radians(float(self.cfg.palm_cmd_rate_limit_rot_deg))
        if _lim_r > 0.0:
            _dr = self.palm_targets[:, 3:6] - self._prev_palm_cmd_rot
            _sr = (_lim_r / _dr.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
            self.palm_targets[:, 3:6] = torch.where(
                self._palm_cmd_primed.unsqueeze(-1),
                self._prev_palm_cmd_rot + _dr * _sr,
                self.palm_targets[:, 3:6])
        self._prev_palm_cmd_rot = self.palm_targets[:, 3:6].clone()
        self._palm_cmd_primed |= True
        self._update_cmd_markers()          # 시각화 전용 — 물리·보상에 영향 없음

        # ---- 손: 시너지 -------------------------------------------------------------
        _prev = self._syn_target
        # ★★닫기 게이트: 컵이 케이지 안에 들어오기 전에는 오므리지 않는다.
        #   래치로는 못 막는다 — 래치는 보상을 여는 신호일 뿐이고 닫힘은 손 액션이
        #   직접 만든다. 경계에서 끊지 않고 램프를 둬 gradient 를 남긴다.
        # ★케이지 중심은 palm 강체 오프셋(`_cage_offset_palm`, 홈 실측)으로 만든다.
        #   실시간 손끝 평균이면 손을 오므리는 것만으로 게이트가 열린다(위 실측 근거).
        #   이제 게이트는 **팔 지령에만** 의존한다 — 정책이 위치를 맞춰야 열린다.
        # ★거리는 3D. xy 투영은 z 를 못 봐서 palm·검지가 컵보다 내려간 잘못된 자세도
        #   통과시켰다(사용자 GUI 관찰: 엄지가 컵에 걸린 채 접근). 3D 로 바꾸면
        #   상수를 늘리지 않고 높이 조건이 같이 들어온다.
        _obj = self._env_local(self.object.data.root_pos_w)
        _palm = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])
        _cage = _palm + (self._palm_ee_R() @ self._cage_offset_palm)
        self._cage_ctr_dist = self._banded_dist(_cage - _obj)
        if bool(self.cfg.close_gate_enabled):
            _ramp = max(float(self.cfg.close_gate_ramp) * self._r_cage, 1e-6)
            _g = ((self._r_cage - self._cage_ctr_dist) / _ramp).clamp(0.0, 1.0)
            # ★래치 후에는 해제한다. 게이트는 **접근 구간 전용**이다 — 들고 가는 중에
            #   컵이 흔들려 게이트가 닫히면 다시 쥘 길이 막힌다(audit Check 3).
            self._close_gate = torch.where(
                self._latched, torch.ones_like(_g), _g)
        else:
            self._close_gate = torch.ones(self.num_envs, device=self.device)
        self._syn_target = self._synergy_targets(self.actions[:, 6:])
        self._syn_vel = (self._syn_target - _prev) / self._policy_dt
        # ★fabric 의 손 **상태**를 실제 손 자세로 동기화한다. 안 그러면 fabric 이
        #   실재하지 않는 손으로 충돌구 FK 를 계산해 없는 자기충돌을 피하려 팔을 민다.
        self.fabric_q[:, self._fab_hand_t] = self._syn_to_fab(self._syn_target)

        self._step_fabric()

    # ------------------------------------------------------------------
    # 관측
    # ------------------------------------------------------------------
    def _tip_force_local(self) -> torch.Tensor:
        """손끝 접촉력을 **팁 로컬 프레임**으로 회전한 3축 벡터 (N, 3·T).

        실기 `fingertip_*/wrench` 와 직접 대응시키기 위한 표현이다(월드 프레임 힘은
        팔 자세가 바뀌면 같은 접촉이 다른 값으로 읽힌다).
        """
        from isaaclab.utils.math import quat_apply, quat_conjugate
        out = []
        _max = float(self.cfg.contact_force_max)
        for k, finger in enumerate(self._finger_names):
            s = self._finger_sensors[finger][-1]
            f_w = s.data.force_matrix_w.view(self.num_envs, -1, 3).sum(dim=1)
            q = self.robot.data.body_quat_w[:, self.tip_ids[k]]
            out.append(quat_apply(quat_conjugate(q), f_w) / _max)
        return torch.cat(out, dim=1).clamp(-1.0, 1.0)

    def _joint_pos_err(self) -> torch.Tensor:
        """손 관절 목표 − 실측 (N, n_hand), 부호 보존 정규화.

        ★인벨롭이 잘 될수록 팁 F/T 가 0 을 읽는 문제가 있어, 추종 오차가 **주 파지력
          관측**이 된다(잡고 있으면 목표를 못 따라가 오차가 남는다).
        """
        err = self._syn_target - self.robot.data.joint_pos[:, self._syn_ids]
        return (err / float(self.cfg.joint_pos_err_max)).clamp(-1.0, 1.0)

    def _perceived_object(self, obj_pos: torch.Tensor, palm_pos: torch.Tensor,
                          R: torch.Tensor) -> torch.Tensor:
        """정책이 **믿는** 물체 위치 (N,3). obs 전용 — 보상·판정·critic 은 참값이다.

        두 노브가 각각 독립적으로 실기 정합을 올린다.

        ①`obs_object_rigid_after_latch` — 래치 뒤엔 손이 컵을 가려 비전이 잃는다.
          래치 순간의 palm 상대위치를 현재 palm 자세로 굴려 "잡았으니 손과 같이
          움직인다"를 추정한다. 컵이 손안에서 미끄러져도 정책은 모른다 —
          실기와 동일하게 **접촉력으로만** 알 수 있다.

        ②`obs_object_noise_coherent` — 노이즈를 **한 번만** 뽑는다. 세 항이 같은
          추정값을 쓰므로 평균으로 노이즈를 상쇄하는 우회가 막힌다. 항등 경로에서는
          각 항이 따로 뽑으므로(구 동작) 이 함수의 반환값을 쓰지 않는다.
        """
        obj_est = obj_pos
        if bool(getattr(self.cfg, "obs_object_rigid_after_latch", False)):
            _rigid = palm_pos + torch.einsum("nij,nj->ni", R, self._obj_off_palm)
            obj_est = torch.where(self._latched.unsqueeze(1), _rigid, obj_pos)
            # ★강체 가정의 오차 = 실제 미끄러짐. 이 값이 크면 obs 가 거짓말을 하는 중이다.
            self.extras["perc/obj_est_err"] = float(
                (obj_est - obj_pos).norm(dim=-1).mean())
        return obj_est + torch.randn_like(obj_est) * self._adr_obs_noise_object

    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel
        n = self.num_envs
        cfgn = self.cfg

        arm_q = q[:, self._arm_ids_t]
        arm_qd = qd[:, self._arm_ids_t]
        hand_q = q[:, self._hand_ids_t]
        hand_qd = qd[:, self._hand_ids_t]
        palm_pos = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])
        _R = self._palm_ee_R()
        # 쿼터니언은 q ≡ −q 부호 이중성이 있어 회전행렬 두 열로 준다.
        palm_ax = torch.cat([_R[:, :, 0], _R[:, :, 1]], dim=1)
        tips_w = self.robot.data.body_pos_w[:, self._tip_ids_t]
        tips_rel_palm = (
            tips_w - self.robot.data.body_pos_w[:, self.palm_idx].unsqueeze(1)
        ).reshape(n, -1)
        obj_pos = self._env_local(self.object.data.root_pos_w)
        palm_to_obj = obj_pos - palm_pos
        obj_to_tips = (tips_w - self.scene.env_origins[:, None, :]
                       - obj_pos.unsqueeze(1)).reshape(n, -1)
        tip_force = self._tip_force_local()
        joint_err = self._joint_pos_err()
        goal_rel = self.goal_pos - obj_pos

        # ★★지각 모델 — 물체 파생 obs 를 **하나의 추정값**에서 뽑는다(기본은 항등).
        #   구 경로는 `goal_rel` 에만 노이즈가 없어 `obj = goal_pos − goal_rel` 로
        #   참값이 복원됐다(goal_pos 는 에피소드 상수) — `obs_noise_object` 축이
        #   사실상 무효였던 원인이다. 여기서 세 항을 같은 값으로 묶어 막는다.
        if bool(getattr(cfgn, "obs_object_noise_coherent", False)):
            _obj_obs = self._perceived_object(obj_pos, palm_pos, _R)
            _n_palm_to_obj = _obj_obs - palm_pos
            _n_obj_to_tips = (tips_w - self.scene.env_origins[:, None, :]
                              - _obj_obs.unsqueeze(1)).reshape(n, -1)
            _n_goal_rel = self.goal_pos - _obj_obs
        else:
            _n_palm_to_obj = (palm_to_obj
                              + torch.randn_like(palm_to_obj) * self._adr_obs_noise_object)
            _n_obj_to_tips = (obj_to_tips
                              + torch.randn_like(obj_to_tips) * self._adr_obs_noise_object)
            _n_goal_rel = goal_rel

        # actor 에만 노이즈 — critic 은 clean state 를 받는다.
        _noisy = torch.cat([
            arm_q + torch.randn_like(arm_q) * self._adr_obs_noise_qpos,
            arm_qd + torch.randn_like(arm_qd) * self._adr_obs_noise_qvel,
            hand_q + torch.randn_like(hand_q) * self._adr_obs_noise_qpos,
            hand_qd + torch.randn_like(hand_qd) * self._adr_obs_noise_qvel,
            palm_pos + torch.randn_like(palm_pos) * cfgn.obs_noise_body,
            palm_ax,
            tips_rel_palm + torch.randn_like(tips_rel_palm) * cfgn.obs_noise_body,
            _n_palm_to_obj, _n_obj_to_tips,
            tip_force, joint_err, self.actions, _n_goal_rel,
        ], dim=1)

        clean = torch.cat([
            arm_q, arm_qd, hand_q, hand_qd, palm_pos, palm_ax, tips_rel_palm,
            palm_to_obj, obj_to_tips, tip_force, joint_err, self.actions, goal_rel,
        ], dim=1)

        _mid, _dist = self._contact_forces_split()
        _thr = float(cfgn.contact_force_threshold)
        _max = float(cfgn.contact_force_max)
        state = torch.cat([
            clean,
            self.object.data.root_lin_vel_w,
            self.object.data.root_ang_vel_w,
            self.object.data.root_quat_w,
            (obj_pos[:, 2] - self.object_spawn_pos[:, 2]).unsqueeze(1),
            (_dist > _thr).float(), (_dist / _max).clamp(max=1.0),
            (_mid > _thr).float(), (_mid / _max).clamp(max=1.0),
            (self.episode_length_buf.float()
             / float(self.max_episode_length)).unsqueeze(1),
            (tips_w - self.scene.env_origins[:, None, :] - obj_pos.unsqueeze(1)
             ).norm(dim=-1),
            (self.goal_pos - obj_pos).norm(dim=-1, keepdim=True),
        ], dim=1)
        return {"policy": torch.nan_to_num(_noisy), "critic": torch.nan_to_num(state)}

    # ------------------------------------------------------------------
    # 보상
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        cfgn = self.cfg
        obj_pos = self._env_local(self.object.data.root_pos_w)
        palm_pos = self._env_local(self.robot.data.body_pos_w[:, self.palm_idx])

        # ---- 접촉 ------------------------------------------------------------------
        _thr = float(cfgn.contact_force_threshold)
        tip_f = self._tip_contact_forces()                        # (N, F)
        tip_c = tip_f > _thr
        mid_f, dist_f = self._contact_forces_split()
        mid_c, dist_c = mid_f > _thr, dist_f > _thr
        # ★손등 접촉 배제 — 켜져 있으면 손바닥면이 물체를 향하는 접촉만 남긴다.
        #   센서 튜플 순서가 (mid, dist, tip) 이라 열 0/1/-1 로 맞춘다.
        if self._palmar_axes is not None:
            _pm = self._palmar_mask()                             # (N, F, L)
            mid_c = mid_c & _pm[:, :, 0]
            dist_c = dist_c & _pm[:, :, 1 if _pm.shape[2] >= 3 else 0]
            tip_c = tip_c & _pm[:, :, -1]
        n_tip = len(self._finger_names)
        tip_frac = tip_c.float().sum(dim=1) / n_tip

        # ---- 힘 밴드 (08.29 신설) — `graded_contact` 에 곱할 품질계수 [바닥, 1] --------
        # ★그 env 의 **최대 팁 힘** 하나로 정한다. 사용자 제약이 "팁 센서 정격 0~50 N"
        #   이고 손가락 하나만 과해도 하드웨어가 위험하므로 평균이 아니라 최댓값이다.
        #   `force_band_hi_n` 아래는 정확히 1.0(현행과 동일) → 정상 파지는 손해가 없다.
        # ★게이트가 쓰는 이진 마스크(`tip_c`·`grip_c`)는 **건드리지 않는다** — 여기서
        #   깎으면 케이지·판정까지 조용히 흔들린다. 보상 곱셈 경로에만 넣는다.
        _fb_lo = float(cfgn.force_band_hi_n)
        _fb_hi = float(cfgn.force_sensor_max_n)
        _floor = float(cfgn.force_band_floor)
        _over = ((tip_f.max(dim=1).values - _fb_lo)
                 / max(_fb_hi - _fb_lo, 1e-6)).clamp(0.0, 1.0)
        force_quality = 1.0 - (1.0 - _floor) * _over
        self.extras["gate/force_quality"] = force_quality.mean()
        self.extras["gate/force_quality_min"] = force_quality.min()
        grip_c = tip_c | mid_c | dist_c
        grip_frac = grip_c.float().sum(dim=1) / n_tip
        n_grip = grip_c.float().sum(dim=1)

        # ★이진 케이지 게이트 (DexPoint `r_contact`) — 엄지 ∧ (대향 ≥ n).
        #   프로필의 `contact_group_a/b` 로 일반화해 2지 그리퍼에서도 성립한다.
        #   0 이면 끈다(기본). 접촉 **개수** 그 자체를 보상하지 않는 것이 핵심이다 —
        #   이 저장소에 "손끝을 몰아 개수만 채우는" 실패 이력이 두 건 있다.
        _cg_n = int(cfgn.cage_gate_min_opposing)
        if _cg_n > 0:
            _cage_ok = (grip_c[:, self._group_a_idx].any(dim=1)
                        & (grip_c[:, self._group_b_idx].sum(dim=1) >= _cg_n))
            self.extras["gate/cage_ok"] = _cage_ok.float().mean()
        else:
            _cage_ok = None

        # ---- 감쌈 -------------------------------------------------------------------
        # ★★08.28 재정의. 구 정의(`deep_and`)는 두 가지가 동시에 틀렸다:
        #   ①분모가 `contact_group_b` 뿐이라 **엄지 감쌈이 원리적으로 반영 안 된다**.
        #   ②손바닥은 08.27 에 센서를 붙이고도 진단 로깅에만 쓰였다.
        #   실측(E1 play 900스텝): 다섯 손가락 원위가 전부 0.00 인데 `syn_close` 3·4번
        #   마디는 1.00 완전 굴곡이고 `palm_rate` 0.55~0.82 다 — 손바닥도 손끝도
        #   닿는데 그 사이 마디만 빈다. 즉 `mid ∧ dist` 는 이 손 형상에서 도달 불가다.
        # 사용자 확정 정의: 잡는 방식에 제한을 두지 않되 **다섯 손가락과 손바닥이
        #   유기적으로 감싸는 것**. 작은 물체는 손바닥+4지가 말고 엄지가 위를 얹는
        #   주먹 자세, 큰 물체는 엄지까지 감싸는 자세 — 둘 다 정답이므로 마디 조합을
        #   요구하지 않고 **표면 참여 여부**만 센다.
        # ★2지 그리퍼 호환: group_a=jaw1 · group_b=jaw2 로 그대로 성립하고,
        #   손바닥이 닿지 않는 프로필은 `envelope_palm_weight=0.0` 으로 끄면
        #   가중 합 정규화가 척도를 유지한다.
        _surf_palm = (self._palm_contact_force() > _thr).float()
        _surf_a = grip_c[:, self._group_a_idx].float().mean(dim=1)
        _surf_b = grip_c[:, self._group_b_idx].float().mean(dim=1)
        self._surf_palm, self._surf_a, self._surf_b = _surf_palm, _surf_a, _surf_b
        if str(cfgn.envelope_metric) == "surface_count":
            _wp = float(cfgn.envelope_palm_weight)
            _wa = float(cfgn.envelope_group_a_weight)
            _wb = float(cfgn.envelope_group_b_weight)
            _wsum = max(_wp + _wa + _wb, 1e-6)
            wrap_frac = (_wp * _surf_palm + _wa * _surf_a + _wb * _surf_b) / _wsum
        else:
            # 구 정의 = per-finger (중간 AND 원위). 기본값이라 G0 는 E1 과 항등이다.
            wrap_frac = (mid_c & dist_c)[:, self._wrap_idx].float().mean(dim=1)

        # ---- 래치 (보상 단계 표시 전용 — 팔 지령은 건드리지 않는다) ------------------
        # ★★08.29 "opposition" 신설 — count(기본·현행)는 실측 성공 파지(엄지+palm)에서
        #   n_grip=1 < min 3 이라 **래치가 영원히 0** → lift30·transfer15·stay8·
        #   stabilize10 이 전부 사장되고 postlatch 힘 감시도 0 이었다(K1/M1/N1 공통).
        #   대향 판정 = (그룹A 접촉) AND (그룹B 접촉 OR palm 접촉) — force-closure
        #   독트린 그대로이고, 손바닥은 사용자 정의상 정당한 파지 요소다.
        #   hold 8스텝은 유지(브러시 접촉 필터).
        if str(getattr(cfgn, "latch_mode", "count")) == "opposition":
            _a_c = grip_c[:, self._group_a_idx].any(dim=1)
            _b_c = grip_c[:, self._group_b_idx].any(dim=1)
            _p_c = self._palm_contact_force() > _thr
            _ready = _a_c & (_b_c | _p_c)
        else:
            _ready = n_grip >= int(cfgn.lift_start_min_grip_fingers)
        self._hold_count = torch.where(
            _ready & ~self._latched, self._hold_count + 1,
            torch.where(self._latched, self._hold_count,
                        torch.zeros_like(self._hold_count)))
        _just = (~self._latched) & (self._hold_count >= int(cfgn.grasp_ready_hold_steps))
        self._latched = self._latched | _just
        # ★래치 순간의 palm 상대 물체위치를 스냅샷 — 지각 모델(강체 부착)의 기준.
        #   `_cage_offset_palm` 과 같은 규약이다(Rᵀ(x−palm)). 여기서 뜨는 이유는
        #   `_get_rewards` 가 `_get_observations` **앞**에 돌아 같은 스텝의 래치가
        #   즉시 obs 에 반영되기 때문이다(DirectRLEnv: dones→rewards→reset→obs).
        _R_latch = self._palm_ee_R()
        _off_latch = torch.einsum("nji,nj->ni", _R_latch, obj_pos - palm_pos)
        self._obj_off_palm = torch.where(
            _just.unsqueeze(1), _off_latch, self._obj_off_palm)

        # ---- 기하 --------------------------------------------------------------------
        grasp_center = obj_pos.clone()
        grasp_center[:, 2] += float(cfgn.object_grasp_z_offset)
        palm_to_cup = self._banded_dist(palm_pos - grasp_center)
        # ---- finger_closure — 미접촉 손가락별 컵 접근도(닿으면 소등) ------------------
        #   형상 비의존: 파지중심 하나만 쓴다(반경·표면 없음). 상세 근거는 rewards 참조.
        _tips_l = (self.robot.data.body_pos_w[:, self._tip_ids_t]
                   - self.scene.env_origins[:, None, :])
        _tip_d = (_tips_l - grasp_center.unsqueeze(1)).norm(dim=-1)     # (N, 5)
        if str(getattr(cfgn, "finger_closure_target", "tip")) == "wrap":
            # ★감쌈 기준 — 소등 조건이 "중간∧원위 동시접촉", 거리도 **중간마디** 기준.
            #   팁 기준(구판)은 팁이 닿는 순간 꺼져서 "손끝만 대는" 지점이 종착지가
            #   된다. 감쌈까지 경사를 이으려면 그 다음 마디를 목표로 삼아야 한다.
            _mid_l = (self.robot.data.body_pos_w[:, self._mid_ids_t]
                      - self.scene.env_origins[:, None, :])
            _cl_d = (_mid_l - grasp_center.unsqueeze(1)).norm(dim=-1)   # (N, 5)
            _cl_off = mid_c & dist_c
        else:
            _cl_d, _cl_off = _tip_d, tip_c
        finger_closure = (
            (~_cl_off).float()
            * torch.exp(-float(cfgn.finger_closure_sharpness) * _cl_d)
        ).mean(dim=1)
        self.extras["task/finger_closure"] = finger_closure.mean()
        # ★★palm 프레임 분해 — 법선(palm_ee_x)이 **밀착도**다. `_palm_ee_R()` 열 0 이
        #   손바닥 법선이다. 접근 목표를 케이지가 아니라 palm 이 맡게 하는 핵심 양.
        _d = grasp_center - palm_pos
        _R = self._palm_ee_R()
        _dn = (_d * _R[:, :, 0]).sum(dim=-1)                 # 법선 성분(부호 포함)
        _dy = (_d * _R[:, :, 1]).sum(dim=-1)
        _dz = (_d * _R[:, :, 2]).sum(dim=-1)
        palm_normal_dist = _dn.abs()
        # 손바닥 면 안의 어긋남 — z 는 데드밴드를 통과시킨다(파지 높이는 여유 축).
        _dzb = torch.relu(_dz.abs() - float(self.cfg.grasp_z_deadband))
        palm_lateral_dist = torch.sqrt(_dy.pow(2) + _dzb.pow(2))
        # ★밀착한 채 **정지**해야 시너지 손가락이 말릴 시간이 생긴다.
        _vp = self.robot.data.body_lin_vel_w[:, self.palm_idx].norm(dim=-1)
        palm_still = torch.exp(-float(self.cfg.palm_still_gain) * _vp)
        self._palm_normal_dist, self._palm_lateral_dist, self._palm_speed = (
            palm_normal_dist, palm_lateral_dist, _vp)
        cup_disp = (obj_pos[:, :2] - self.object_spawn_pos[:, :2]).norm(dim=-1)
        height_delta = obj_pos[:, 2] - self.object_spawn_pos[:, 2]
        goal_dist = (obj_pos - self.goal_pos).norm(dim=-1)

        # ---- 케이지 중심 ↔ 컵 -----------------------------------------------------
        # ★★08.27 사용자 GUI 관찰로 확정된 수정: "손바닥·검지~소지는 잘 붙는데 **엄지가
        #   걸려** 인벨롭이 안 된다".
        #   구 수식은 대향축을 `접근방향을 90° 회전`으로 잡았다 —
        #   `axis = (−dir_y, dir_x)` 는 **좌/우 부호가 임의**라, 접근 방향에 따라 엄지
        #   목표가 실제 엄지의 반대편에 놓인다. 그러면 손목을 뒤집어야 도달 가능한
        #   자세를 요구하게 되고, 정책은 그 방향으로 갈 수 없어 엄지가 걸린 채 4지만
        #   붙인다. 실측 귀결: grip_frac 0.20 인데 wrap_frac 이 2,228 iter 내내 0.000.
        # ★수정: 대향 중점을 **손 자신의 기하**에서 뽑는다. 손잡이 방향이 구조적으로
        #   맞고(엄지가 어디 있든 정의가 성립), 물체 반경 상수도 필요 없다.
        #   `opp_mid` 가 컵 중심에 오려면 엄지와 4지가 컵을 **사이에 두어야** 한다 —
        #   두 그룹이 같은 쪽에 있으면 중점이 그쪽으로 쏠려 거리가 남는다.
        # ★★approach 가 쓰는 케이지 거리는 **palm 강체**다(닫기 게이트와 같은 양).
        #   실시간 손끝을 쓰면 "쭉 편 손가락으로 팁을 컵에 모으기"가 이 항의 최적이 되어
        #   파지 예비자세를 정면으로 방해한다 — s2r_a9 실측 corr(ch2, approach) = −0.702.
        cage_dist = self._cage_ctr_dist
        _a = int(self._group_a_idx[0])          # 대향(엄지) 인덱스 — success 판정용

        # 래치 시점 스냅샷 — 감쌈 유지 기준선과 밀림 감쇠 기준.
        self._wrap_at_latch = torch.where(_just, wrap_frac, self._wrap_at_latch)
        self._disp_at_latch = torch.where(_just, cup_disp, self._disp_at_latch)

        # ---- 자세·안정 ---------------------------------------------------------------
        upright_q = torch.exp(-self._tilt_deg / float(cfgn.upright_sharpness))
        lin_v = self.object.data.root_lin_vel_w.norm(dim=-1)
        ang_v = self.object.data.root_ang_vel_w.norm(dim=-1)
        stable = (lin_v <= float(cfgn.stable_lin_vel)) & (ang_v <= float(cfgn.stable_ang_vel))
        stability_q = torch.exp(-2.0 * lin_v) * torch.exp(-0.5 * ang_v)

        # ---- 성공 · stay 유지 ---------------------------------------------------------
        lifted = height_delta >= float(cfgn.lift_success_height)
        at_goal = goal_dist <= float(cfgn.goal_pos_tolerance)
        # ★★08.28 사용자 확정: 과제 목적은 "컵이 목표에 제대로 놓여 멈춰 있는가" 이고
        #   파지 여부는 거기에 **이미 함축**된다. 두 절이 산술로 중복임을 확인했다:
        #   ①`lifted` — 목표 z = 스폰 +0.08, 허용 반경 0.025 라 `at_goal` 이면
        #     컵 z ≥ 스폰 +0.055 로 임계 0.04 를 자동으로 넘는다.
        #   ②`holding` — 테이블에서 8cm 뜬 컵이 `lin_v ≤ 0.04 m/s` 면 무언가가 받치고
        #     있다는 뜻이다(무지지 낙하는 물리 스텝 하나 8.3ms 에 0.08 m/s 초과).
        #   게다가 `n_grip >= 4` 는 코드 리터럴이라 2지 그리퍼에서 절대 성립 불가였다.
        #   → 기본값은 현행 유지, 플래그로 끈다.
        _ok = at_goal & stable & (self._tilt_deg <= float(cfgn.success_tilt_max_deg))
        if bool(cfgn.success_require_lifted):
            _ok = _ok & lifted
        if bool(cfgn.success_require_holding):
            _ok = _ok & (n_grip >= int(cfgn.success_min_grip_fingers)) & tip_c[:, _a]
        self._success_now = _ok
        _stay_ok = at_goal & stable & (n_grip >= 2)
        self._stay_run = torch.where(_stay_ok, self._stay_run + 1,
                                     torch.zeros_like(self._stay_run))
        stay_frac = (self._stay_run.float()
                     / float(max(int(cfgn.stay_hold_steps), 1))).clamp(max=1.0)

        action_delta = (self.actions - self.prev_actions).pow(2).mean(dim=-1).sqrt()
        self.prev_actions = self.actions.clone()

        enclosure = self._enclosure(obj_pos)
        self.extras["task/enclosure"] = enclosure.mean()
        # ★케이지 게이트가 켜져 있으면 포위도에 곱한다 — DexPoint 는 이진 접촉 게이트를
        #   `r_lift` 에 곱하지만, 여기서는 **신설 항에만** 걸어 한 번에 하나의 가설을
        #   지킨다(기존 lift/transfer/stay 의 척도를 건드리지 않는다).
        if _cage_ok is not None:
            enclosure = enclosure * _cage_ok.float()
            self.extras["task/enclosure_gated"] = enclosure.mean()

        # ★과지령 = 가동 손관절이 **도달 불가능한 각도**를 미는 정도 [0,1].
        #   τ = k·err 이므로 err ≥ effort_limit/stiffness 면 토크가 천장이다. 가동폭 0 인
        #   관절(pinky_2·thumb_2·전 `_1`)은 오차가 상수로 깔려 평균을 오염시키므로 뺀다.
        _thr_od = float(cfgn.hand_torque_sat_err_rad)
        _od_err = (self._syn_target
                   - self.robot.data.joint_pos[:, self._syn_ids]).abs()[:, self._syn_movable]
        hand_overdrive = (torch.relu(_od_err - _thr_od)
                          / max(_thr_od, 1e-6)).clamp(max=1.0).mean(dim=1)
        self.extras["task/hand_overdrive"] = hand_overdrive.mean()

        # ---- 손 최저 높이 (바닥 벌점 기준) ----------------------------------------
        # ★env-local 로 변환한다 — 월드 z 를 그대로 쓰면 env 격자 오프셋이 섞인다.
        #   `_env_local` 은 (N,3) 브로드캐스트라 (N,K,3) 에는 못 쓴다 — z 만 직접 뺀다.
        _hand_z_min = (
            self.robot.data.body_pos_w[:, self._hand_body_ids_t, 2]
            - self.scene.env_origins[:, 2].unsqueeze(1)
        ).min(dim=1).values
        self.extras["task/hand_z_min"] = _hand_z_min.mean()
        self.extras["task/hand_z_min_worst"] = _hand_z_min.min()
        # ★09.01 신설 — `hand_floor` 벌점이 `min` 으로 링크 25개를 스칼라 하나로 뭉개
        #   "손 전체를 눕히기"와 "손끝 하나 스치기"가 같은 값이 된다(사용자 지적).
        #   벌점 수식을 고치기 전에 **몇 개 링크가 얼마나 내려가는지**부터 계측한다.
        _z_links = (
            self.robot.data.body_pos_w[:, self._hand_body_ids_t, 2]
            - self.scene.env_origins[:, 2].unsqueeze(1)
        )
        _viol = torch.relu(float(self.cfg.hand_floor_z) - _z_links)     # (N, K)
        self.extras["task/hand_floor_n_links"] = (_viol > 0).float().sum(dim=1).mean()
        self.extras["task/hand_floor_depth_sum"] = _viol.sum(dim=1).mean()
        self.extras["task/hand_floor_depth_max"] = _viol.max()
        # ★09.01 신설 — palm 접근축 자세. 사용자 요구는 "palm_ee_x ⟂ world z"(= 90°)인데
        #   구속은 `palm_rot_half_deg` ±45° 박스뿐이고 중심으로 당기는 보상 항이 없다.
        #   회전이 지금까지 **어디에도 로깅되지 않아** 드리프트를 볼 수 없었다.
        #   ★진단 전용 — 보상·게이트·obs 어디에도 쓰지 않는다.
        from isaaclab.utils.math import matrix_from_quat
        _px = matrix_from_quat(self.robot.data.body_quat_w[:, self.palm_idx])[:, :, 0]
        _ang = torch.rad2deg(torch.arccos(_px[:, 2].clamp(-1.0, 1.0)))  # (N,) 90° = 수직
        self.extras["palm/x_vs_worldz_deg"] = _ang.mean()
        _pre = ~self._latched
        _n_pre = _pre.float().sum().clamp(min=1.0)
        self.extras["palm/x_vs_worldz_deg_prelatch"] = (_ang * _pre.float()).sum() / _n_pre
        _n_post = self._latched.float().sum().clamp(min=1.0)
        self.extras["palm/x_vs_worldz_deg_postlatch"] = (
            (_ang * self._latched.float()).sum() / _n_post)
        self.extras["palm/x_vs_worldz_dev_max"] = (_ang - 90.0).abs().max()

        total, terms, gates = compute_grasp_ua_rewards(
            enclosure=enclosure,
            finger_closure=finger_closure,
            force_quality=force_quality,
            hand_overdrive=hand_overdrive,
            hand_z_min=_hand_z_min,
            tip_contact_frac=tip_frac,
            wrap_frac=wrap_frac,
            wrap_at_latch=self._wrap_at_latch,
            grip_frac=grip_frac,
            # ★★손바닥 포함 "닿기만 하면 된다" 지표 (08.31 사용자 정의).
            #   손가락은 **어느 마디든**(tip|mid|dist) 닿으면 1, 손바닥도 동등한 한 표.
            #   분모 = 손가락 수 + 1. 붓기 과제에서 컵을 놓치지 않으려면 접촉 **개수**가
            #   중요하지 특정 마디 조합이 중요한 게 아니다 — wrap(중간∧원위)은 2~3개만
            #   닿는 현 상태를 못 벗어나게 만든 정의였다.
            anylink_frac=((grip_c.float().sum(dim=1) + self._surf_palm)
                          / (n_tip + 1.0)),
            palm_normal_dist=palm_normal_dist,
            palm_lateral_dist=palm_lateral_dist,
            palm_still=palm_still,
            close_gate=self._close_gate,
            close_progress=self._close_progress(),
            cup_height_delta=height_delta,
            cup_xy_disp_now=cup_disp,
            cup_xy_disp_ref=self._disp_at_latch,
            cup_tilt_deg=self._tilt_deg,
            goal_dist=goal_dist,
            upright_quality=upright_q,
            lift_latched=self._latched,
            stay_frac=stay_frac,
            stable=stable,
            stability_quality=stability_q,
            success_now=self._success_now,
            action_delta_norm=action_delta,
            cfg=cfgn,
        )
        total = total + float(cfgn.abnormal_penalty) * self._abnormal.float()
        # ---- 재소환 벌점 (기본 0 = 항등) — 직전 스텝 재소환 발생 env 에 1회 차감 ----
        _rp = float(getattr(cfgn, "respawn_penalty", 0.0))
        if _rp > 0.0 and hasattr(self, "_respawn_pen_buf"):
            total = total - _rp * self._respawn_pen_buf
            self.extras["reward/respawn_penalty"] = \
                (-_rp * self._respawn_pen_buf).mean()
            self._respawn_pen_buf.zero_()

        # 단계 도달 누적 (리셋에서만 평균 기록 — 스텝 비용 0)
        self._stage_hit[:, 0] |= self._latched
        self._stage_hit[:, 1] |= lifted & self._latched
        self._stage_hit[:, 2] |= self._latched & lifted & (goal_dist < 0.10)
        self._stage_hit[:, 3] |= self._success_now

        for k in GRASP_UA_REWARD_TERMS:
            self.extras[f"reward/{k}"] = terms[k].mean()
        self.extras["reward/total"] = total.mean()
        for k, v in gates.items():
            self.extras[f"gate/{k}"] = v.mean()
        self.extras["task/wrap_frac"] = wrap_frac.mean()
        self.extras["task/grip_frac"] = grip_frac.mean()
        self.extras["task/anylink_frac"] = (
            (grip_c.float().sum(dim=1) + self._surf_palm) / (n_tip + 1.0)).mean()
        self.extras["task/n_contact"] = grip_c.float().sum(dim=1).mean()
        self.extras["task/touch_frac"] = tip_frac.mean()
        # ---- ★★리프트 이후 접촉 구성 (09.01 신설) ---------------------------------
        # 위 지표들은 전부 **에피소드 전체 평균**이라 아무것도 안 닿는 접근 구간이
        # 섞여 희석된다 — "못 감"과 "지나침"을 뭉개는 그 함정이다. pouring 이관 판정은
        # **컵을 든 뒤에 몇 면이 닿고 있는가**로 해야 하므로 리프트로 게이팅해 따로 쓴다.
        # 사용자 기준(09.01): "손바닥과 손끝으로라도 정확히 5개가 지탱해 주면
        # 강체처럼 움직이므로 그게 중요하다."
        _lm = gates["lifted"].float()
        _ln = _lm.sum().clamp(min=1.0)
        def _g(x):                      # 리프트된 env 만의 평균
            return float((x * _lm).sum() / _ln)
        self.extras["lifted/n_tip"] = _g(tip_c.float().sum(dim=1))
        self.extras["lifted/n_mid"] = _g(mid_c.float().sum(dim=1))
        self.extras["lifted/n_dist"] = _g(dist_c.float().sum(dim=1))
        self.extras["lifted/n_finger"] = _g(grip_c.float().sum(dim=1))
        self.extras["lifted/palm"] = _g(self._surf_palm)
        # ★"5점 + 손바닥" 달성률 — 사용자 기준의 직접 판정
        self.extras["lifted/full_support"] = _g(
            ((grip_c.float().sum(dim=1) >= 5.0) & (self._surf_palm > 0.5)).float())
        self.extras["lifted/frac"] = float(_lm.mean())
        self.extras["task/goal_dist"] = goal_dist.mean()
        self.extras["task/height_delta"] = height_delta.mean()
        self.extras["task/cup_disp"] = cup_disp.mean()
        self.extras["task/palm_to_cup"] = palm_to_cup.mean()
        self.extras["task/cage_dist"] = cage_dist.mean()
        self.extras["task/tilt_deg"] = self._tilt_deg.mean()
        self.extras["task/latched"] = self._latched.float().mean()
        self.extras["task/success"] = self._success_now.float().mean()
        self.extras["task/stay_run"] = self._stay_run.float().mean()
        self.extras["task/syn_close"] = self._syn_close.mean()
        self.extras["task/close_credit"] = self._close_progress().mean()
        self.extras["task/palm_normal_dist"] = self._palm_normal_dist.mean()
        self.extras["task/palm_lateral_dist"] = self._palm_lateral_dist.mean()
        self.extras["task/palm_speed"] = self._palm_speed.mean()
        # ★palm 이 테이블에 쓸리는지 — 사용자 GUI 관찰 "손바닥이 테이블에 쓸리면서
        #   열린다". 접촉 센서는 **컵만** 필터링해서 테이블 접촉이 안 보인다. 높이로 잰다.
        # ★기준 body 는 프로필 `palm_body` 의 **원점**이다(palm_ee 가 아니다) — 손바닥
        #   표면·손가락 끝은 이보다 더 내려가므로 이 값은 침범의 **하한**만 말한다.
        _pz = palm_pos[:, 2] - float(self.cfg.table_surface_z)
        self.extras["task/palm_above_table_mean"] = _pz.mean()
        self.extras["task/palm_above_table_min"] = _pz.min()
        # ★손 관절 추종오차 — 액추에이터 포화의 직접 지표. τ = k·err 이므로
        #   err ≥ effort_limit/stiffness 면 토크가 천장에 붙어 힘 제어가 무효가 된다
        #   (5.0/1.5 기준 0.30 rad = 17.2°). 지금까지 팔(fabric/joint_err_*)만 있었다.
        _herr = (self._syn_target
                 - self.robot.data.joint_pos[:, self._syn_ids]).abs()
        self.extras["task/hand_joint_err_mean"] = _herr.mean()
        self.extras["task/hand_joint_err_max"] = _herr.max()
        # ★Phase 0(08.29): 위 평균에는 **가동폭 0인 관절**이 섞여 있다(실측 `_syn_movable`
        #   기준 pinky_2·thumb_2·전 `_1`). 지령이 나가도 안 움직이니 오차가 상수로 깔리고,
        #   그러면 "닿은 뒤에도 계속 더 닫고 있다"를 평균으로 판정할 수 없다. 갈라서 잰다.
        #   τ = k·err 이므로 **가동 관절의** err ≥ effort/stiffness(1.5/5.0 = 0.30 rad)
        #   여야 진짜 토크 포화다.
        _mv = self._syn_movable
        self.extras["task/hand_joint_err_movable_mean"] = _herr[:, _mv].mean()
        self.extras["task/hand_joint_err_movable_max"] = _herr[:, _mv].max()
        self.extras["task/hand_joint_err_fixed_mean"] = (
            _herr[:, ~_mv].mean() if bool((~_mv).any()) else _herr.new_zeros(()))
        self.extras["task/hand_torque_sat_frac"] = (
            _herr[:, _mv] >= float(self.cfg.hand_torque_sat_err_rad)).float().mean()
        # ★채널별 폐쇄도 — 전체 평균만 보면 "어느 채널이 안 닫히는지"를 못 본다.
        #   08.27: 평균 0.278 이 채널1(`_2`)만 폐쇄한 예측치 0.250 과 맞아떨어졌고,
        #   GUI 관찰(`_2` 완전굴곡·`_3`/`_4` 정지)과 일치했다. ch2 가 낮은 이유가
        #   "명령이 안 나간다"인지 "명령은 나가는데 동결이 먹는다"인지 가른다.
        for _c in range(self._syn_nch):
            _m = self._syn_ch == _c
            self.extras[f"task/syn_close_ch{_c}"] = self._syn_close[:, _m].mean()
        self.extras["task/close_gate"] = self._close_gate.mean()
        self.extras["task/cage_ctr_dist"] = self._cage_ctr_dist.mean()
        self.extras["task/abnormal_rate"] = self._abnormal.float().mean()
        # ★Phase 0(08.29): `force_max` 는 손가락별 **3마디 합산**의 최댓값이라 사용자
        #   제약(팁 센서 단독 0~50 N)과 직접 비교가 안 된다. 또 평균(1~2 N)과 60배
        #   어긋나 이 값이 파지력인지 접근 충돌 스파이크인지 알 수 없었다 — 래치로 가른다.
        _cfm = self._contact_forces().max(dim=1).values             # (N,)
        _z = torch.zeros_like(_cfm)
        self.extras["contact/force_max"] = _cfm.max()
        self.extras["contact/force_max_prelatch"] = torch.where(self._latched, _z, _cfm).max()
        self.extras["contact/force_max_postlatch"] = torch.where(self._latched, _cfm, _z).max()
        self.extras["fabric/palm_cmd_step_raw"] = self._palm_cmd_step_raw.mean()
        _jerr = (self.fabric_q[:, self._fab_arm_t]
                 - self.robot.data.joint_pos[:, self._arm_ids_t]).abs()
        self.extras["fabric/joint_err_mean"] = _jerr.mean()
        # ★`fabric_q` 는 오픈루프 적분 plant 라 리셋 전까지 실측으로 되돌아오지 않는다.
        #   에피소드 **안에서** 팔이 막히면(접촉·관절한계) fabric 만 계속 적분해 격차가
        #   벌어지는데, 평균은 그 순간을 묻어버린다. 08.27 실측으로 **누적은 반증**됐지만
        #   (ep_len 16→594 로 37배인데 joint_err 0.040→0.033 로 감소, 전 구간 0.023~0.053)
        #   그건 평균 얘기다 — 최대값을 따로 봐야 막힘 구간을 잡는다.
        self.extras["fabric/joint_err_max"] = _jerr.max()
        self.extras["fabric/palm_err_mean"] = (
            self.palm_targets[:, :3] + self._fab_to_env - palm_pos).norm(dim=-1).mean()
        self._log_diagnostics(_thr, mid_f, dist_f, tip_f, obj_pos, palm_pos)
        return total

    # ------------------------------------------------------------------
    def _palmar_mask(self) -> torch.Tensor:
        """손가락 마디의 **손바닥면**이 물체를 향하는 접촉만 True (N, F, L).

        Hu et al. 2020 의 `p_collision`(손등 접촉 벌점, 가중 −1) 대응이다. 벌점 대신
        **접촉에서 배제**하는 방식을 쓴다 — 자매 트랙에서 검증된 형태이고, 벌점은
        접촉 탐색 자체를 억제한 이력이 있다(s2r_a1: 닿을수록 순증분이 음수).

        ★힘 벡터가 아니라 **기하**로 판정한다. `force_matrix_w` 의 부호 규약이 실측
        확정되지 않아, 뒤집히면 조용히 반대 판정이 된다. 기하 판정은 자매 트랙에서
        probe 로 분리 검증됐다(손바닥 +30.6/+45.0 mm vs 손등 −19.7 mm).
        """
        from isaaclab.utils.math import quat_apply
        _pos = self.robot.data.body_pos_w[:, self._palmar_body_ids]      # (N, F, L, 3)
        _quat = self.robot.data.body_quat_w[:, self._palmar_body_ids]    # (N, F, L, 4)
        _ax = self._palmar_axes[None, :, None, :].expand_as(_pos)
        _palmar_w = quat_apply(_quat.reshape(-1, 4),
                               _ax.reshape(-1, 3)).view_as(_pos)
        _to_obj = self.object.data.root_pos_w[:, None, None, :] - _pos
        return (_palmar_w * _to_obj).sum(dim=-1) > 0.0

    def _enclosure(self, obj_pos: torch.Tensor) -> torch.Tensor:
        """물체를 **둘러싼 정도** [0,1] — 접촉이 아니라 기하다.

        물체 중심에서 손 키포인트로 향하는 단위벡터들이 서로 상쇄되는 정도를 잰다.
        한쪽에서만 잡으면 벡터가 모두 같은 방향이라 합의 크기가 1 에 가깝고(→0),
        둘러싸면 상쇄되어 0 에 가깝다(→1).

        ★★08.28 신설. 근거: Hu et al. 2020(arXiv:2002.04498)은 인벨롭을 **두 항**으로
        나눈다 — `r_topology`(손 키포인트 hull 안에 물체가 들었는가, 가중 **10**)와
        `r_contact`(접촉 개수, 가중 2). 우리는 접촉 쪽만 갖고 있었고, 그래서 팁 파지가
        감쌈 지표를 만점 받았다(G 라운드 실측: `wrap_frac` 0.91 인데 `dist_rate` 0.02).
        접촉 기반 정의를 아무리 고쳐도 같은 자세로 수렴한다.

        ★형상 정보를 **하나도** 쓰지 않는다 — 물체 중심 하나뿐이다. 반경·높이·메시가
        필요 없으므로 컵 종류를 늘려도 그대로 성립한다. hull 대신 방향 분산을 쓰는
        이유는 sim2real 이다: 링크 **위치**(FK)는 실기에서도 정확하지만 접촉점 개수는
        시뮬레이터의 contact discretization·마찰·강성에 민감해 전이되지 않는다.

        ★되먹임 함정 반증 — 이 트랙은 "케이지를 실시간 손끝으로 계산 금지"를 계약으로
        잠가 뒀다(팔 정지 구간 `corr(syn_close, cage_dist) = −0.974`, 손만 오므려도
        게이트가 열렸다). 포위도는 그 함정에 걸리지 않는다: ①게이트가 아니라 **보상**이라
        `close_gate` 에 넣지 않는다 ②물체가 멀면 손을 아무리 오므려도 모든 단위벡터가
        여전히 같은 방향이라 값이 0 이다 — **빈 공간에서 오므려서는 못 올린다**.
        """
        # ★월드 좌표로 계산한다 — 손과 물체의 **차분**이라 env 원점이 상쇄되므로
        #   env-local 변환이 불필요하다(그 변환은 (N,3) 라 (N,B,3) 에 브로드캐스트도 안 된다).
        _p = self.robot.data.body_pos_w                           # (N, B, 3)
        _o = self.object.data.root_pos_w.unsqueeze(1)             # (N, 1, 3)

        def _u(ids) -> torch.Tensor:
            _d = _p[:, ids] - _o
            return _d / _d.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        _wp = float(self.cfg.enclosure_palm_weight)
        _wa = float(self.cfg.enclosure_group_a_weight)
        _wb = float(self.cfg.enclosure_group_b_weight)
        _up = _u([self.palm_idx]).squeeze(1)                       # (N, 3)
        _bar = (_wp * _up
                + _wa * _u(self._hull_a_t).mean(dim=1)
                + _wb * _u(self._hull_b_t).mean(dim=1)) / max(_wp + _wa + _wb, 1e-6)
        _encl = (1.0 - _bar.norm(dim=-1)).clamp(0.0, 1.0)

        # ---- 손가락별 최소참여 (08.29 신설, λ=0 이면 위와 항등) -----------------------
        # ★위 식은 그룹 키포인트를 **평균**한다. 손가락 하나가 빠져도 그 손가락의
        #   단위벡터가 나머지와 비슷한 방향이라 평균이 거의 안 떨어진다 — 즉
        #   **손가락별 최소참여 신호가 없다**. 이것이 `couple_four_fingers` 를 넣게 만든
        #   3지 국소최적의 원인 진단("mean/count 보상엔 손가락별 최소참여 신호 부재")과
        #   같은 결함이고, 커플링을 풀기 전에 반드시 메워야 한다.
        # 손가락 f 의 참여도 = 손바닥과 **반대편**에 있는 정도.
        #   c_f = 1 − ‖(û_palm + û_f)/2‖  ∈ [0,1] (정반대면 1)
        _lam = float(self.cfg.enclosure_participation_lambda)
        if _lam <= 0.0:
            return _encl
        # ★`_u` 는 1D 인덱스 전용이다 — 여기 인덱스는 (F_b, L) 2D 라 물체 축을
        #   한 번 더 펴야 한다((N,F_b,L,3) vs (N,1,3) 은 브로드캐스트가 깨진다).
        _dp = _p[:, self._hull_part_t] - _o.unsqueeze(1)           # (N, F_b, L, 3)
        _uf = (_dp / _dp.norm(dim=-1, keepdim=True).clamp(min=1e-6)).mean(dim=2)
        _uf = _uf / _uf.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (N, F_b, 3)
        _c = 1.0 - (0.5 * (_up.unsqueeze(1) + _uf)).norm(dim=-1)   # (N, F_b)
        _weak = _c.min(dim=1).values.clamp(0.0, 1.0)
        self.extras["task/enclosure_weakest"] = _weak.mean()
        return ((1.0 - _lam) * _encl + _lam * _weak).clamp(0.0, 1.0)

    def _contact_azimuth_spread(self, obj_pos: torch.Tensor,
                                thr: float) -> torch.Tensor:
        """접촉한 손끝이 컵 축 둘레에 퍼진 **각폭**(도) 평균 — 진단 전용.

        `360° − 최대 간극`. 한쪽에 몰리면 작고 감싸면 커진다. 접촉 **개수**가 아니라
        **어디를 눌렀는지**를 보므로 형상에 의존하지 않는다 — force-closure 의 대리량이다.
        ★08.25 실측: 손끝 방위각을 125°→318° 로 몰아 접촉을 끊으면서 grip 보상을 올린
          수법이 있었다. 개수만 세면 그 수법이 안 보인다.
        ★접촉 손끝이 2개 미만이면 각폭이 정의되지 않아 0 을 돌려준다.
        """
        _tips = (self.robot.data.body_pos_w[:, self._tip_ids_t]
                 - self.scene.env_origins[:, None, :])
        _hit = self._tip_contact_forces() > thr                      # (N, F)
        _d = _tips[:, :, :2] - obj_pos[:, None, :2]
        _ang = torch.atan2(_d[:, :, 1], _d[:, :, 0])                 # (N, F) [−π, π]
        # 미접촉 손끝은 정렬에서 뒤로 몰아 무시한다.
        _big = torch.full_like(_ang, 1e3)
        _srt, _ = torch.sort(torch.where(_hit, _ang, _big), dim=1)
        _n = _hit.sum(dim=1)
        # 원형 간극: 이웃 차이 + 마지막→첫 랩어라운드. 접촉분만 유효하다.
        _idx = torch.arange(_hit.shape[1], device=self.device).view(1, -1)
        _valid = _idx < _n.view(-1, 1)
        _nxt = torch.roll(_srt, shifts=-1, dims=1)
        _first = _srt[:, :1]
        _nxt = torch.where(_idx == (_n.view(-1, 1) - 1),
                           _first + 2.0 * math.pi, _nxt)
        _gap = torch.where(_valid, _nxt - _srt, torch.zeros_like(_srt))
        _spread = 2.0 * math.pi - _gap.max(dim=1).values
        return torch.where(_n >= 2, torch.rad2deg(_spread),
                           torch.zeros_like(_spread)).mean()

    # ------------------------------------------------------------------
    def _log_diagnostics(self, thr: float, mid_f: torch.Tensor, dist_f: torch.Tensor,
                         tip_f: torch.Tensor, obj_pos: torch.Tensor,
                         palm_pos: torch.Tensor) -> None:
        """진단 전용 로깅 — 보상·게이트·종료에 **일절 쓰이지 않는다**.

        ★`wrap_frac`(중간 AND 원위)이 s2r_b5 4,553 기록점 내내 정확히 0.000 인데 영상에서는
          감쌈이 성립한다. 세 가설을 한 런에서 가르기 위한 계측이다:
            (a) 필터/집계 결함 — `net > 0` 인데 `matrix = 0`
            (b) 임계 미달      — 낮은 임계에서만 `> 0`
            (c) 진짜 미접촉    — 둘 다 0. 이때 `hand_blocked_frac` 이
                                 "컵에 막힘"인지 "자유롭게 말림"인지 가른다
        """
        _lo = float(self.cfg.diag_contact_threshold_lo)
        _n_mid, _n_dist, _n_tip = self._finger_link_forces(self._mag_net)
        for _nm, _f, _nf in (("mid", mid_f, _n_mid), ("dist", dist_f, _n_dist),
                             ("tip", tip_f, _n_tip)):
            self.extras[f"contact/{_nm}_rate"] = (_f > thr).float().mean()
            self.extras[f"contact/{_nm}_rate_lo"] = (_f > _lo).float().mean()
            self.extras[f"contact/{_nm}_rate_net"] = (_nf > thr).float().mean()
            self.extras[f"contact/{_nm}_f_mean"] = _f.mean()
            self.extras[f"contact/{_nm}_f_net_mean"] = _nf.mean()
        # 원위가 용의자라 손가락별로 따로 본다 — 어느 손가락이 못 닿는지.
        for _i, _fg in enumerate(self._finger_names):
            self.extras[f"contact/dist_rate_{_fg}"] = (dist_f[:, _i] > thr).float().mean()
            self.extras[f"contact/dist_net_{_fg}"] = (_n_dist[:, _i] > thr).float().mean()
        self.extras["task/hand_blocked_frac"] = self._hand_blocked().float().mean()
        # ★손바닥 — 컵을 실제로 받치는 면이 어디인지. 그동안 계측 자체가 없었다.
        _palm_f = self._palm_contact_force()
        self.extras["contact/palm_rate"] = (_palm_f > thr).float().mean()
        self.extras["contact/palm_rate_lo"] = (_palm_f > _lo).float().mean()
        self.extras["contact/palm_f_mean"] = _palm_f.mean()
        # ★신 감쌈의 세 성분을 **활성 metric 과 무관하게 항상** 찍는다 — 구 정의로 도는
        #   갈래에서도 신 지표를 관측해야 사후 비교가 된다.
        self.extras["task/envelope_surf_palm"] = self._surf_palm.mean()
        self.extras["task/envelope_surf_a"] = self._surf_a.mean()
        self.extras["task/envelope_surf_b"] = self._surf_b.mean()

        # ---- Phase 0: 팁 단독 힘 분포 (08.29 신설) ------------------------------------
        # ★실기 팁 센서 정격은 **0~50 N** 이고 그 위는 측정 자체가 안 된다. 그런데
        #   하드웨어는 그것을 넘길 능력이 있다 — URDF effort 7.5 N·m / 원위 모멘트암
        #   25.5 mm(`_4` 축→`_tip`) ⇒ 실기 최대 294 N, sim(effort_limit 1.5) 58.8 N.
        #   즉 지금 팁 힘이 1~2 N 인 것은 보상 설계가 아니라 **정책이 아직 그 자세를
        #   안 만들었기 때문**이고, 보상에는 힘 항이 하나도 없다(grep 0건).
        #   밴드 임계를 추측으로 정하지 않기 위해 분포부터 잰다. 평균만으로는 못 본다.
        _tf = tip_f.reshape(-1)
        self.extras["contact/tip_f_p95"] = torch.quantile(_tf, 0.95)
        self.extras["contact/tip_f_p99"] = torch.quantile(_tf, 0.99)
        self.extras["contact/tip_f_max"] = _tf.max()
        # 접촉 중인 팁만 — 미접촉 0 이 섞인 분위수는 접촉 세기를 과소평가한다.
        _tc = _tf[_tf > thr]
        self.extras["contact/tip_f_c_mean"] = (
            _tc.mean() if _tc.numel() > 0 else _tf.new_zeros(()))
        self.extras["contact/tip_f_c_p95"] = (
            torch.quantile(_tc, 0.95) if _tc.numel() > 0 else _tf.new_zeros(()))
        self.extras["contact/tip_over_band_frac"] = (
            _tf > float(self.cfg.force_band_hi_n)).float().mean()
        self.extras["contact/tip_over_sensor_frac"] = (
            _tf > float(self.cfg.force_sensor_max_n)).float().mean()
        # ★대향성 — 접촉점이 컵 둘레에 **퍼져** 있어야 force-closure 다. 한쪽에 몰리면
        #   팁 개수가 많아도 컵을 밀어낼 뿐이다(08.25 grip 접촉 절벽의 방위각 수법).
        self.extras["contact/azimuth_spread_deg"] = self._contact_azimuth_spread(
            obj_pos, thr)

        # ---- goal 성분 분해 -----------------------------------------------------------
        # `goal_dist` 스칼라만으로는 0.28 이 높이 탓인지 수평 탓인지 알 수 없다.
        _gd = obj_pos - self.goal_pos
        self.extras["task/goal_dz"] = _gd[:, 2].abs().mean()
        self.extras["task/goal_dxy"] = _gd[:, :2].norm(dim=-1).mean()

        # ---- 홈 복귀 확인 -------------------------------------------------------------
        # 액션 규약이 `palm = 홈 + delta(a)` 라 **a=0 이 정확히 홈**이다. 래치 후 정책이
        # 홈으로 이완하면 컵이 목표가 아니라 홈 위로 실려 간다.
        self.extras["task/action_norm_arm"] = self.actions[:, :6].norm(dim=-1).mean()
        self.extras["task/palm_to_home"] = (
            palm_pos - self._home_palm[:3].unsqueeze(0)).norm(dim=-1).mean()

        # ---- palm 지령 포화 -----------------------------------------------------------
        # 박스 포화 = 도달영역 부족, 리미터 포화 = 너무 빨리 움직이려는 것. 원인이 다르다.
        for _i, _ax in enumerate("xyz"):
            self.extras[f"fabric/palm_cmd_box_sat_{_ax}"] = \
                self._palm_cmd_box_sat[:, _i].mean()
        self.extras["fabric/palm_cmd_rate_sat"] = self._palm_cmd_rate_sat.mean()
        self.extras["fabric/palm_cmd_z"] = self.palm_targets[:, 2].mean()
        self.extras["fabric/palm_z_min"] = palm_pos[:, 2].min()

        # ---- Phase 0: 액션 앵커 재설계용 palm 실측 (08.29 신설, 진단 전용) -------------
        # ★앵커 오프셋을 추측하지 않기 위한 계측이다. 지금까지 z 지령 하나만 찍혀 있어
        #   "파지할 때 palm 이 어디에 있고 이송할 때 어디로 가는지"를 답할 수 없었다.
        #   래치로 두 구간을 갈라 각각의 palm 실위치를 잰다(래치는 보상 단계 표시이고
        #   여기서도 **읽기만** 한다 — 액션·게이트·종료 경로에는 들어가지 않는다).
        for _i, _ax in enumerate("xy"):
            self.extras[f"fabric/palm_cmd_{_ax}"] = self.palm_targets[:, _i].mean()
        _lat = self._latched.float()
        _den_l = _lat.sum().clamp(min=1.0)
        _den_n = (1.0 - _lat).sum().clamp(min=1.0)
        for _i, _ax in enumerate("xyz"):
            self.extras[f"fabric/palm_post_latch_{_ax}"] = \
                (palm_pos[:, _i] * _lat).sum() / _den_l
            self.extras[f"fabric/palm_pre_latch_{_ax}"] = \
                (palm_pos[:, _i] * (1.0 - _lat)).sum() / _den_n

    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_pos = self._env_local(self.object.data.root_pos_w)
        from isaaclab.utils.math import quat_apply
        _up = quat_apply(
            self.object.data.root_quat_w,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3))
        self._tilt_deg = torch.rad2deg(torch.acos(_up[:, 2].clamp(-1.0, 1.0)))

        out_x = (obj_pos[:, 0] < self.cfg.object_out_x[0]) | \
                (obj_pos[:, 0] > self.cfg.object_out_x[1])
        out_y = (obj_pos[:, 1] < self.cfg.object_out_y[0]) | \
                (obj_pos[:, 1] > self.cfg.object_out_y[1])
        fell = obj_pos[:, 2] < float(self.cfg.object_min_z)
        tipped = self._tilt_deg > float(self.cfg.tilt_reset_deg)

        # abnormal = 물리 위반만(관절 한계 초과 또는 속도 폭주).
        q_arm = self.robot.data.joint_pos[:, self._arm_ids_t]
        qd_arm = self.robot.data.joint_vel[:, self._arm_ids_t]
        beyond = (q_arm < self._arm_lo - 0.05) | (q_arm > self._arm_hi + 0.05)
        runaway = qd_arm.abs() > float(self.cfg.abnormal_qd)
        self._abnormal = (beyond | runaway).any(dim=-1)
        # ★08.29 신설 — `abnormal` 은 두 원인의 OR 라 합쳐 보면 처방이 갈리지 않는다.
        #   `beyond`(관절 한계 밖) = 초기 자세·리셋 기입 문제.
        #   `runaway`(속도 폭주)   = 스폰 겹침 → depenetration 반발
        #   (`max_depenetration_velocity=1000` 이라 겹치면 엄청나게 튕긴다).
        #   다물체에서 `episode_lengths` 가 260 → **1.2 스텝**으로 붕괴하고
        #   `done/abnormal` 만 0.838 로 발화했다(fell·out_xy 는 0) — 무한 리셋이다.
        self.extras["done/abnormal_beyond"] = beyond.any(dim=-1).float().mean()
        self.extras["done/abnormal_runaway"] = runaway.any(dim=-1).float().mean()
        self.extras["done/arm_qd_max"] = qd_arm.abs().max()
        # 어느 관절인지 — 팔 관절별 위반율(처방이 관절마다 다르다).
        for _i in range(beyond.shape[-1]):
            self.extras[f"done/beyond_j{_i + 1}"] = beyond[:, _i].float().mean()
        # ★★리셋이 실제로 먹었는가 — `replicate_physics=False` 에서 폭주하는 이유의
        #   마지막 미검증 후보다. `_reset_idx` 가 `write_joint_state_to_sim` 으로 홈을
        #   써도 물리가 이전 자세를 유지하면, fabric 은 홈 기준으로 명령하고 실제는
        #   다른 자세라 거대한 토크가 걸린다. 방금 리셋된 env 만 골라 편차를 잰다.
        _fresh = self.episode_length_buf <= 1
        if bool(_fresh.any()):
            _dev = (q_arm[_fresh]
                    - self._default_q[_fresh][:, self._arm_ids_t]).abs()
            self.extras["reset/arm_q_dev_mean"] = _dev.mean()
            self.extras["reset/arm_q_dev_max"] = _dev.max()
            self.extras["reset/arm_qd_max_fresh"] = qd_arm[_fresh].abs().max()
            self.extras["reset/fresh_frac"] = _fresh.float().mean()
        # ★★env 인덱스 정렬 검사 — 로봇 base 는 **자기 env 원점**에 있어야 한다
        #   (`robot_cfg.init_state.pos = [0,0,0]`). `replicate_physics=False` 에서
        #   ArticulationView 가 프림을 모으는 순서가 `env_ids` 와 어긋나면
        #   리셋이 엉뚱한 env 에 쓰이고 원래 env 는 초기화되지 않는다 — 관절이 임의
        #   값이 되고(실측 27.85 rad) 속도가 폭주한다(3,232 rad/s).
        #   어긋나면 이 값이 env_spacing(2.0 m) 배수로 튄다.
        _off = (self.robot.data.root_pos_w[:, :2]
                - self.scene.env_origins[:, :2]).abs()
        self.extras["diag/root_vs_origin_max"] = _off.max()
        self.extras["diag/root_vs_origin_mean"] = _off.mean()

        # ---- 낙하/전도 재소환 (08.30 신설, 기본 OFF) --------------------------------
        # ★★종료가 유일한 실패 처리면 "시도 → 실패 → 미래 보상 전액 상실"이라
        #   무접촉 정체(공짜 enclosure)가 국소최적이 된다(M0·O1 실측 7.2/step 일치).
        #   자매 grip/left/grasp_sensor_v2 가 재소환 도입 직후 성공이 활발해진 실증.
        # ★자매와 다른 점: 우리 앵커·목표가 **스폰 스냅샷**이라 새 자리 샘플링 대신
        #   **이번 에피소드의 원래 스폰점으로 되돌린다** — 에피소드 중 액션 의미 불변.
        # ★palm 여유 게이트 — 손 옆 텔레포트는 depenetration(1000) 폭발 위험.
        #   여유 미달이면 **보류**(컵은 쓰러진 채, 다음 스텝 재시도 — 자매의 검증 규약.
        #   정체 데드락은 없다: 그 자리 보상은 stage 0 뿐이라 gradient 가 밀어낸다).
        if bool(getattr(self.cfg, "respawn_on_fail", False)):
            _need = out_x | out_y | fell | tipped
            if bool(_need.any()):
                _ids = torch.nonzero(_need).squeeze(-1)
                if str(getattr(self.cfg, "respawn_mode", "origin")) == "free":
                    # ★★자매 v2 규약 — 스폰 상자 안에서 **손이 없는 자리**를 리젝션
                    #   샘플링한다. "원래 스폰점 복귀"는 그 지점이 곧 손자리라 보류가
                    #   0.93 까지 갔다(08.30 Q3 실측). 상자는 리셋 때와 같은 범위라
                    #   물체만 새로 리셋하는 것과 의미가 같다.
                    _p = self.profile
                    _rng = float(getattr(self.cfg, "respawn_range", 0.0)) \
                        or float(self._adr_spawn_range)
                    _k = int(getattr(self.cfg, "respawn_tries", 24))
                    _m = _ids.numel()
                    _off2 = (torch.rand(_m, _k, 2, device=self.device) - 0.5) \
                        * 2.0 * _rng
                    _cand = torch.zeros(_m, _k, 3, device=self.device)
                    _cand[:, :, 0] = _p.object_spawn_center[0] + _off2[:, :, 0]
                    _cand[:, :, 1] = _p.object_spawn_center[1] + _off2[:, :, 1]
                    _cand[:, :, 2] = (float(self.cfg.table_surface_z)
                                      + self._obj_origin_off[_ids].unsqueeze(1))
                    _hand = torch.cat([
                        self.robot.data.body_pos_w[:, self.palm_idx].unsqueeze(1),
                        self.robot.data.body_pos_w[:, self._tip_ids_t],
                    ], dim=1)[_ids] - self.scene.env_origins[_ids].unsqueeze(1)
                    # (m, k, hand) → 후보별 손 최소거리
                    _dk = (_cand.unsqueeze(2) - _hand.unsqueeze(1)).norm(dim=-1) \
                        .min(dim=2).values
                    _ok = _dk >= float(self.cfg.respawn_clearance_m)
                    _has = _ok.any(dim=1)
                    _first = torch.argmax(_ok.int(), dim=1)
                    _pick = _cand[torch.arange(_m, device=self.device), _first]
                    # ★후보 전부 실패한 env 는 보류(자매 규약 — 폴백 텔레포트 금지).
                    _clear = _has
                    _tgt = _pick
                    self.extras["done/respawn_cand_dist"] = _dk.max(dim=1).values.mean()
                else:
                    _tgt = self.object_spawn_pos[_ids]
                # ★★여유는 **손 전체**로 재야 한다 — palm 원점만 재면 손끝(palm 에서
                #   10~15cm)이 스폰점에 있어도 통과해 컵이 오므린 손가락 안으로
                #   텔레포트된다. 08.30 실측: 여유 0.15(Q3) postlatch 216N 대비
                #   여유 0.06(R0a/R0b) 479~531N — 여유를 줄일수록 관통이 늘었다.
                    if bool(getattr(self.cfg, "respawn_clearance_uses_tips", False)):
                        _pts = torch.cat([
                            self.robot.data.body_pos_w[:, self.palm_idx].unsqueeze(1),
                            self.robot.data.body_pos_w[:, self._tip_ids_t],
                        ], dim=1)[_ids] - self.scene.env_origins[_ids].unsqueeze(1)
                        _d_hand = (_pts - _tgt.unsqueeze(1)).norm(dim=-1) \
                            .min(dim=1).values
                    else:
                        _d_hand = (self._env_local(
                            self.robot.data.body_pos_w[:, self.palm_idx])[_ids]
                            - _tgt).norm(dim=-1)
                    _clear = _d_hand >= float(self.cfg.respawn_clearance_m)
                _go = _ids[_clear]
                _tgt_go = _tgt[_clear]
                self.extras["done/respawn_rate"] = _need.float().mean()
                self.extras["done/respawn_defer"] = (~_clear).float().mean()
                # 연속 보류 카운트 — 예산 초과 env 는 아래에서 종료로 폴백한다.
                if not hasattr(self, "_defer_count"):
                    self._defer_count = torch.zeros(
                        self.num_envs, dtype=torch.long, device=self.device)
                _defer_ids = _ids[~_clear]
                self._defer_count += 1
                _keep = torch.zeros_like(self._defer_count, dtype=torch.bool)
                _keep[_defer_ids] = True
                self._defer_count = torch.where(
                    _keep, self._defer_count, torch.zeros_like(self._defer_count))
                self.extras["done/defer_streak_max"] = self._defer_count.max()
                if _go.numel() > 0:
                    _n = _go.numel()
                    # ★free 모드는 새 자리로 가므로 **스폰 기준·목표를 같이 옮긴다** —
                    #   안 옮기면 `cup_xy_disp` 가 즉시 큰 값이 되어 approach 가
                    #   순벌점이 된다(08.30 Q3 −0.35 의 정체). 물체만 새로 리셋하는
                    #   것과 의미가 같고, 실시간 추종이 아니라 **이산 사건**이라
                    #   앵커 되먹임 계약(실시간 물체 추종 금지)에 걸리지 않는다.
                    if str(getattr(self.cfg, "respawn_mode", "origin")) == "free":
                        #   앵커는 `_palm_anchor()` 가 `object_spawn_pos` 를 그때그때
                        #   읽으므로 이 대입만으로 같이 따라간다(재검증 불필요).
                        self.object_spawn_pos[_go] = _tgt_go
                        # 목표 오프셋은 **이 에피소드에 뽑힌 값**을 유지한다.
                        self.goal_pos[_go] = _tgt_go + self._goal_off_env[_go]
                    _pose = torch.zeros(_n, 7, device=self.device)
                    _pose[:, :3] = (self.object_spawn_pos[_go]
                                    + self.scene.env_origins[_go])
                    _pose[:, 2] += float(self.cfg.object_spawn_pad)
                    _pose[:, 3] = 1.0                       # 직립 (w,x,y,z)
                    self.object.write_root_pose_to_sim(_pose, env_ids=_go)
                    self.object.write_root_velocity_to_sim(
                        torch.zeros(_n, 6, device=self.device), env_ids=_go)
                    # 파지 단계 상태 되감기 — 컵이 시작점으로 돌아갔다.
                    if not hasattr(self, "_respawn_pen_buf"):
                        self._respawn_pen_buf = torch.zeros(
                            self.num_envs, device=self.device)
                    self._respawn_pen_buf[_go] = 1.0
                    self._latched[_go] = False
                    self._hold_count[_go] = 0
                    self._wrap_at_latch[_go] = 0.0
                    self._disp_at_latch[_go] = 0.0
                    self._obj_off_palm[_go] = 0.0
                    self._stay_run[_go] = 0
            else:
                self.extras["done/respawn_rate"] = torch.zeros(
                    (), device=self.device)
            # ★보류 예산 초과 = 재소환으로 구제 불가 → 옛 규약(종료)으로 폴백.
            _budget = int(getattr(self.cfg, "respawn_defer_budget", 0))
            if _budget > 0 and hasattr(self, "_defer_count"):
                _stuck = self._defer_count >= _budget
                self.extras["done/respawn_stuck"] = _stuck.float().mean()
                terminated = self._abnormal | _stuck
                self._defer_count = torch.where(
                    _stuck, torch.zeros_like(self._defer_count), self._defer_count)
            else:
                terminated = self._abnormal.clone()
        else:
            terminated = out_x | out_y | fell | tipped | self._abnormal
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # ★종료 원인별 비율. 없으면 "무엇이 에피소드를 끝냈는가"를 다른 지표로
        #   역산해야 한다(08.27 자살 경로 진단에서 실제로 그랬다).
        self.extras["done/out_xy"] = (out_x | out_y).float().mean()
        self.extras["done/fell"] = fell.float().mean()
        self.extras["done/tipped"] = tipped.float().mean()
        self.extras["done/abnormal"] = self._abnormal.float().mean()
        self.extras["done/truncated"] = truncated.float().mean()
        return terminated, truncated

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # 단계 도달률은 리셋 시점에만 기록한다.
        for i, nm in enumerate(self._stage_names):
            self.extras[f"stage/{nm}"] = self._stage_hit[env_ids, i].float().mean()
        self._stage_hit[env_ids] = False

        # ---- 로봇: 고정 홈 ------------------------------------------------------------
        q0 = self._default_q[env_ids].clone()
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), env_ids=env_ids)
        self.robot.set_joint_position_target(q0, env_ids=env_ids)
        # 손은 완전 개방에서 시작. `_syn_target` 을 안 맞추면 첫 스텝에 거대한 가짜 속도.
        self._syn_close[env_ids] = 0.0
        self._syn_target[env_ids] = q0[:, self._syn_ids]
        self._syn_vel[env_ids] = 0.0

        # ---- fabric 씨딩 (리셋이 fabric 상태를 실측과 맞추는 유일한 지점) -------------
        self.fabric_q[env_ids] = q0[:, self._fab_t]
        self.fabric_qd[env_ids] = 0.0
        self.fabric_qdd[env_ids] = 0.0
        self.palm_targets[env_ids] = self._home_palm.unsqueeze(0)
        self._palm_cmd_primed[env_ids] = False

        # ---- 버퍼 -----------------------------------------------------------------------
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._latched[env_ids] = False
        self._hold_count[env_ids] = 0
        # ★래치 스냅샷은 래치 상태와 **항상 짝지어** 지운다(계약 테스트가 잠근다).
        self._obj_off_palm[env_ids] = 0.0
        # ---- ADR 승급 판정 — 종료 에피소드의 success 집계(전역 level·단조 상승) -----
        #   ★`_success_now` 를 0 으로 되돌리기 **전에** 읽어야 한다.
        if bool(self.cfg.enable_adr):
            self._adr_succ += int(self._success_now[env_ids].sum())
            self._adr_epis += n
            if self._adr_epis >= int(self.cfg.adr_eval_episodes):
                _rate = self._adr_succ / max(self._adr_epis, 1)
                if (_rate >= float(self.cfg.adr_success_threshold)
                        and self._adr_level < 1.0):
                    self._adr_level = min(
                        1.0, self._adr_level + float(self.cfg.adr_step))
                    self._adr_apply()
                    _mr = getattr(self, "_adr_mass_range", (1.0, 1.0))
                    _gr = getattr(self, "_adr_gain_range", (1.0, 1.0))
                    print(f"[grasp_ua][ADR] 승급 level={self._adr_level:.2f} "
                          f"(창 성공률 {_rate:.3f} · spawn_range "
                          f"{self._adr_spawn_range:.3f} · goal_y "
                          f"{float(self._adr_goal_offset[1]):.3f} · obs_noise "
                          f"{self._adr_obs_noise_object:.4f} · qpos "
                          f"{self._adr_obs_noise_qpos:.4f} · qvel "
                          f"{self._adr_obs_noise_qvel:.3f} · mass "
                          f"[{_mr[0]:.2f},{_mr[1]:.2f}] · gain "
                          f"[{_gr[0]:.2f},{_gr[1]:.2f}])", flush=True)
                self._adr_succ = 0
                self._adr_epis = 0
        self.extras["adr/level"] = self._adr_level
        self.extras["adr/spawn_range"] = self._adr_spawn_range
        self.extras["adr/goal_y"] = float(self._adr_goal_offset[1])
        self.extras["adr/finger_residual"] = float(self._adr_residual)
        self.extras["adr/goal_x_span"] = float(self._adr_goal_span[0])
        self.extras["adr/goal_z_max"] = float(self._adr_goal_span[2])
        # 실제로 뽑힌 목표 분포 — 샘플링이 살아있는지 여기서 본다(폭 0 이면 OFF).
        self.extras["adr/goal_y_sampled_mean"] = self._goal_off_env[:, 1].abs().mean()
        self.extras["adr/goal_dist_mean"] = self._goal_off_env.norm(dim=1).mean()
        self.extras["adr/obs_noise_object"] = self._adr_obs_noise_object
        # ---- sim2real 축 (09.01) — 종점이 base 면 전부 상수라 판독 비용만 든다 --------
        self.extras["adr/obs_noise_qpos"] = self._adr_obs_noise_qpos
        self.extras["adr/obs_noise_qvel"] = self._adr_obs_noise_qvel
        _mr = getattr(self, "_adr_mass_range", (1.0, 1.0))
        self.extras["adr/mass_lo"] = _mr[0]
        self.extras["adr/mass_hi"] = _mr[1]
        _gr = getattr(self, "_adr_gain_range", (1.0, 1.0))
        self.extras["adr/gain_lo"] = _gr[0]
        self.extras["adr/gain_hi"] = _gr[1]

        # ---- 종별 success/latched EMA — 집계가 가리는 종별 실패를 드러낸다 ----------
        #   ★무동기 집계(index_add) — per-step 리셋 경로라 .any()/.item() 루프 금지
        #   (Isaac reset 동기화 = util killer 이력). `_latched` 는 1262 에서 지워지므로
        #   여기(그 전)서 읽어야 종료 에피소드의 값이다.
        if self._n_species > 1:
            _sp = self._species_ids[env_ids]
            _one = torch.ones(n, device=self.device)
            _cnt = torch.zeros(
                self._n_species, device=self.device).index_add_(0, _sp, _one)
            _s_sum = torch.zeros(
                self._n_species, device=self.device).index_add_(
                0, _sp, self._success_now[env_ids].float())
            _l_sum = torch.zeros(
                self._n_species, device=self.device).index_add_(
                0, _sp, self._latched[env_ids].float())
            _has = _cnt > 0
            _a = 0.05
            self._species_succ_ema = torch.where(
                _has, (1 - _a) * self._species_succ_ema
                + _a * (_s_sum / _cnt.clamp(min=1.0)), self._species_succ_ema)
            self._species_latch_ema = torch.where(
                _has, (1 - _a) * self._species_latch_ema
                + _a * (_l_sum / _cnt.clamp(min=1.0)), self._species_latch_ema)
            for _k, _nm in enumerate(self._species_names):
                self.extras[f"species/success_{_nm}"] = self._species_succ_ema[_k]
                self.extras[f"species/latched_{_nm}"] = self._species_latch_ema[_k]
            self.extras["species/success_min"] = self._species_succ_ema.min()
            self.extras["species/success_spread"] = (
                self._species_succ_ema.max() - self._species_succ_ema.min())

        self._wrap_at_latch[env_ids] = 0.0
        self._disp_at_latch[env_ids] = 0.0
        self._stay_run[env_ids] = 0
        self._success_now[env_ids] = False
        self._abnormal[env_ids] = False

        # ---- 물체 스폰 -------------------------------------------------------------------
        p = self.profile
        rng = float(self._adr_spawn_range)
        offs = (torch.rand(n, 2, device=self.device) - 0.5) * 2.0 * rng
        spawn = torch.zeros(n, 3, device=self.device)
        spawn[:, 0] = p.object_spawn_center[0] + offs[:, 0]
        spawn[:, 1] = p.object_spawn_center[1] + offs[:, 1]
        # ★다물체면 원점 오프셋이 물체마다 다르다 — env 별 값을 쓴다. 스칼라를 쓰면
        #   작은 컵은 공중에서 떨어지고 큰 컵은 테이블을 뚫은 채 스폰된다.
        _off = self._obj_origin_off[env_ids]
        spawn[:, 2] = float(self.cfg.table_surface_z) + _off + float(self.cfg.object_spawn_pad)
        # ★기준선은 스폰점이 아니라 **정착고**다(스폰 패드가 리프트 기준에 실리면 안 된다).
        settled = spawn.clone()
        settled[:, 2] = float(self.cfg.table_surface_z) + _off
        self.object_spawn_pos[env_ids] = settled
        # ---- 목표 오프셋 — 샘플링 ON 이면 env 별로 [base, 현재레벨] 안에서 뽑는다 ----
        #   ★단조 상승만 하면 짧은 이송을 잊는다(성공률 지도 실측 y 0.05 에서 0.000).
        _bx, _by, _bz, _sgn = self._adr_goal_base
        _xs, _ys, _zs = self._adr_goal_span
        if bool(getattr(self.cfg, "adr_goal_sample", False)):
            _u = torch.rand(n, 3, device=self.device)
            _goff = torch.empty(n, 3, device=self.device)
            _goff[:, 0] = _bx + (_u[:, 0] * 2.0 - 1.0) * _xs
            _goff[:, 1] = _sgn * (abs(_by) + _u[:, 1] * (abs(_ys) - abs(_by)))
            _goff[:, 2] = _bz + _u[:, 2] * (_zs - _bz)
        else:
            _goff = self._adr_goal_offset.unsqueeze(0).expand(n, 3)
        self._goal_off_env[env_ids] = _goff
        self.goal_pos[env_ids] = settled + _goff

        root = torch.zeros(n, 13, device=self.device)
        root[:, :3] = spawn + self.scene.env_origins[env_ids]
        root[:, 3] = 1.0
        self.object.write_root_state_to_sim(root, env_ids=env_ids)
