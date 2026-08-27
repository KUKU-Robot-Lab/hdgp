"""grasp_lift_fabric — `grasp_s2r` 전면 상속, **손 제어만** 이 트랙 고유.

사용자 지시(08.27): "현재 핸드 제어 부분 빼고, grasp-s2r 세팅으로 변경"
(범위 = 전체 동기: obs·스폰·종료까지).

## 이 트랙이 자매와 다른 단 하나

| | 자매 `grasp_s2r` | 여기 |
|---|---|---|
| 손 액션 | 손가락 5 × 시너지 채널 3 = 15D, **절대 폐쇄도**를 `open→grip` 자세로 lerp | 자유 관절 13D, **절대 관절 목표** |
| 닫힘 경로 | 변화율 상한 + 닫기 게이트 + 접촉 동결 | 없음 — 정책이 관절을 직접 지시 |
| a = −1 | 완전 개방(폐쇄도 0) | **홈(펴짐)** |
| a = +1 | grip 자세 | **굴곡 한계**(부팅 FK 실측 부호) |

팔(Fabrics·홈+델타·slew)·보상·obs·스폰·goal·종료·ADR·마커는 **전부 상속**이다.
자매 파일은 다른 세션 소유 — 읽기·상속만 하고 수정하지 않는다.

## 왜 절대 관절 목표인가

인벨롭 그립은 손가락마다 컵 형상에 맞춰 **다른 각도**로 멈춰야 성립한다. 시너지는
그걸 접촉 동결로 만들지만, 관절을 직접 지시하면 정책이 형상을 직접 학습한다.
대신 역굴곡을 액션 공간에서 **구조적으로 제거**해야 한다 — `_measure_flex_signs`
참조(08.27 "손가락이 난리"의 직접 원인이 이 결함이었다).
"""

from __future__ import annotations

import torch

from ..grasp_s2r.grasp_s2r_env import GraspS2REnv
from .grasp_lift_fabric_env_cfg import GraspLiftFabricEnvCfg, resolve_frozen
from .robot_profiles import PALM_NORMAL_COL


class GraspLiftFabricEnv(GraspS2REnv):
    cfg: GraspLiftFabricEnvCfg

    def __init__(self, cfg: GraspLiftFabricEnvCfg, render_mode: str | None = None, **kw):
        super().__init__(cfg, render_mode, **kw)
        # ★감쌈 분모에서 **가용하지 않은 손가락**을 뺀다. 소지는 `_1`/`_2` 가 둘 다
        #   고정이라 굴곡축이 없다 — 분모에 두면 wrap_frac 상한이 0.75 로 깎이고,
        #   "벌어진 채 굳은 손가락이 컵에 걸린 것"이 접촉으로 세어진다(08.27 실측).
        _unusable = tuple(getattr(self.cfg, "hand_unusable_fingers", ()) or ())
        if _unusable:
            _keep = [int(i) for i in self._wrap_idx.tolist()
                     if self._finger_names[int(i)] not in _unusable]
            if not _keep:
                raise RuntimeError(
                    f"[{self.profile.name}] hand_unusable_fingers={_unusable} 를 빼면 "
                    "감쌈 분모가 비어버린다 — 고정 관절 목록을 확인하라")
            self._wrap_idx = torch.tensor(_keep, device=self.device, dtype=torch.long)
            print(f"[grasp_lift_fabric] 감쌈 분모 = "
                  f"{[self._finger_names[i] for i in _keep]} "
                  f"(제외 {list(_unusable)})", flush=True)

    # ==================================================================
    # 로봇 종속 — 손바닥 법선축
    # ==================================================================
    def _palm_ee_R(self):
        """palm 회전행렬 (N,3,3) — **열 0 이 손바닥 법선**이 되도록 재정렬한다.

        ★★자매 코드는 열 0 = 법선(+x)을 가정한다(approach 의 `palm_normal_dist =
          |d_local.x|`, 케이지 오프셋, obs `palm_ax`). 그런데 **법선축은 자산마다
          다르다** — sensor 자산은 +x, bi_s 자산은 +y 다(`robot_profiles.py` 상단
          근거: URDF 굴곡축×장축 유도 + probe_palmar_sign 실측).
          여기서 한 번 재정렬하면 그 accessor 를 쓰는 downstream 이 전부 맞는다.
        ★순환 치환만 쓴다 — 열을 임의로 바꾸면 det=−1(왼손계)이 되어 회전이 아니게 된다.
        """
        R = super()._palm_ee_R()
        # ★`self.profile` 은 부모 `__init__` 후반에야 생긴다. `_report_home_cage` 가
        #   그보다 먼저 이 accessor 를 쓰므로 **cfg** 에서 읽는다.
        k = PALM_NORMAL_COL[self.cfg.profile_name]
        if k == 0:
            return R
        return R[:, :, [k % 3, (k + 1) % 3, (k + 2) % 3]]

    # ==================================================================
    # 부팅 — 홈 오버라이드 → fabric → 굴곡 부호 실측
    # ==================================================================
    def _setup_fabrics(self) -> None:
        """★순서 계약: 홈 오버라이드는 **`super()._setup_fabrics()` 이전**이어야 한다.

        fabric 의 cspace attractor rest 자세(`default_config`)와 리셋 q0 가 전부
        `robot.data.default_joint_pos` 에서 나오므로, 그 뒤에 덮으면 셋이 갈린다.
        """
        self._apply_hand_home_override()
        super()._setup_fabrics()
        # fabric 이 생긴 뒤에야 손끝 FK 를 쓸 수 있다.
        self._measure_flex_signs()

    def _apply_hand_home_override(self) -> None:
        _ovr = tuple(getattr(self.cfg, "hand_home_override", ()) or ())
        if not _ovr:
            return
        _side = self.profile.name.split("_")[-1][0]
        _jn = list(self.robot.data.joint_names)
        _dj = self.robot.data.default_joint_pos
        _done = []
        for _nm_t, _val in _ovr:
            _nm = _nm_t.replace("{side}", _side)
            if _nm not in _jn:
                raise RuntimeError(
                    f"[{self.profile.name}] hand_home_override 관절 '{_nm}' 가 없다")
            _dj[:, _jn.index(_nm)] = float(_val)
            _done.append(f"{_nm.split('hj_')[-1]}={_val}")
        print(f"[grasp_lift_fabric] 홈 오버라이드 {_done}", flush=True)

    def _setup_synergy(self) -> None:
        """자매 배선(이름 매핑·버퍼)을 그대로 쓰고, **자유/고정 관절 분해만** 얹는다."""
        super()._setup_synergy()
        p = self.profile
        _frozen_names = resolve_frozen(
            p.name, tuple(getattr(self.cfg, "frozen_hand_joints_override", ()) or ()))
        _jn = list(self.robot.data.joint_names)
        _frozen = {_jn.index(nm) for nm in _frozen_names}
        # 자유 관절 — **articulation 관절 인덱스 순서**. 액션 순서의 단일 정의다.
        self._hand_free_t = torch.tensor(
            [i for i in self.hand_ids if i not in _frozen],
            device=self.device, dtype=torch.long)
        n_free = int(self._hand_free_t.numel())
        if 6 + n_free != int(self.cfg.action_space):
            raise RuntimeError(
                f"[{p.name}] 액션 차원 불일치: 6+{n_free} != {self.cfg.action_space} — "
                "cfg 파생(__post_init__)과 고정 관절 목록이 어긋났다")
        # 자유 관절이 synergy 순서(`_syn_ids`) 어디에 있는지 — 목표 조립용.
        _pos = {int(j): k for k, j in enumerate(self._syn_ids)}
        self._free_syn_idx = torch.tensor(
            [_pos[int(j)] for j in self._hand_free_t.tolist()],
            device=self.device, dtype=torch.long)
        # 홈 자세(synergy 순서) — 고정 관절은 여기서 얼어붙는다.
        self._syn_home = self.robot.data.default_joint_pos[0, self._syn_ids].clone()
        # ★`_close_progress`(자매, **실측 관절** 기준)의 분모를 우리 자유 관절로
        #   좁힌다. 고정 관절이 분모에 섞이면 "닫았다"는 공짜 점수가 생긴다.
        #   ★자매 조건(|grip−open| > 0)도 **함께** 남긴다 — 폐쇄도는 open→grip 구간의
        #   비율이라 그 구간이 0 인 관절은 실측으로도 정의되지 않는다.
        _free_mask = torch.zeros_like(self._syn_movable)
        _free_mask[self._free_syn_idx] = True
        self._syn_movable = _free_mask & self._syn_movable
        if not bool(self._syn_movable.any()):
            raise RuntimeError(
                f"[{p.name}] 폐쇄도 분모가 비었다 — 자유 관절 중 open≠grip 인 것이 없다")
        print(f"[grasp_lift_fabric] 손 자유 관절 {n_free}개 · 고정 "
              f"{[n.split('hj_')[-1] for n in _frozen_names] or '없음'} · "
              f"액션 {self.cfg.action_space}D", flush=True)

    def _measure_flex_signs(self) -> None:
        """부팅 1회 FK — 자유 손관절마다 **굴곡(말림) 방향 부호**를 실측한다.

        ★왜 필요한가(08.27 실측): 액션이 `a∈[-1,1] → [관절 lo, hi]` 선형이었는데
          `_3`/`_4` 한계가 **좌우 모두 대칭 ±90°** 라 홈(0)이 한계 **중앙**이다.
          즉 액션 범위의 절반이 손등 쪽 **역굴곡**을 지시하고 있었다("손가락이
          난리"의 직접 원인). 게다가 엄지 `_3`/`_4` 는 우 `+q`·좌 `−q` 가 굴곡인데
          (URDF `thumb_3` origin rpy 가 좌우 뒤집힘, axis 는 둘 다 (0,0,1) 이라
          한계에는 안 드러난다) 액션 매핑에 미러가 없어 좌손 엄지는 `a=+1` 이
          완전 개방이었다.

        판정 기준: **대향 그룹 쪽으로 가면 굴곡**이다. palm 로컬 +x(법선) 성분으로
        재는 `probe_curl_local` 규약은 4지에는 맞지만 **엄지는 대향 운동이라 −x 로
        움직여** 오판한다(FK 실측). 대향 그룹 거리는 프레임 불변이고 로봇 종속
        이름도 안 쓴다.
        """
        p = self.profile
        n_arm = p.num_arm_joints
        _fg = list(self._finger_names)
        _ia = [_fg.index(f) for f in p.contact_group_a]
        _ib = [_fg.index(f) for f in p.contact_group_b]
        _jn = list(self.robot.data.joint_names)
        _fab_hand = self._fab_t[n_arm:]
        _free_fab, _free_fing = [], []
        for _j in self._hand_free_t.tolist():
            _hit_slot = (_fab_hand == _j).nonzero()
            if _hit_slot.numel() != 1:
                raise RuntimeError(
                    f"[{p.name}] 자유 관절 '{_jn[_j]}' 가 fabric 손 구간에 없다")
            _nm = _jn[_j]
            _hit = [k for k, f in enumerate(_fg) if f"_{f}_" in _nm]
            if len(_hit) != 1:
                raise RuntimeError(
                    f"관절 '{_nm}' 의 손가락을 특정할 수 없다(매칭 {_hit}) "
                    f"— fingers={_fg}. 프로필 이름 규약을 확인할 것")
            _free_fab.append(int(_hit_slot[0, 0]))
            _free_fing.append(_hit[0])

        def _tips_of(q: torch.Tensor) -> torch.Tensor:
            return self.fabric._fingertip_taskmap(q, None)[0].reshape(q.shape[0], -1, 3)

        _q0 = self.fabric_q[:1].repeat(self.num_envs, 1)
        _tips0 = _tips_of(_q0)[0]                        # (F,3) 홈 손끝
        _delta = 0.30                                    # [rad] 부호만 보므로 크게
        signs, worst = [], 1e9
        _n = len(_free_fab)
        for _s in range(0, _n, self.num_envs):
            _chunk = list(range(_s, min(_s + self.num_envs, _n)))
            _qb = _q0.clone()
            for _r, _i in enumerate(_chunk):
                _qb[_r, n_arm + _free_fab[_i]] += _delta
            _tips = _tips_of(_qb)
            for _r, _i in enumerate(_chunk):
                _f = _free_fing[_i]
                _opp = _ib if _f in _ia else _ia
                _d0 = float(torch.norm(_tips0[_f] - _tips0[_opp].mean(dim=0)))
                _d1 = float(torch.norm(_tips[_r][_f] - _tips[_r][_opp].mean(dim=0)))
                _chg = _d0 - _d1                          # >0 = 가까워짐 = 굴곡
                worst = min(worst, abs(_chg))
                signs.append(1.0 if _chg > 0.0 else -1.0)
        if worst < 1e-4:
            raise RuntimeError(
                f"굴곡 부호 실측 실패: 대향거리 변화 최소 {worst * 1000:.3f}mm 로 "
                "판별 불가 — taskmap/한계/자산을 확인할 것")
        self._flex_sign = torch.tensor(signs, device=self.device)      # (n_free,)
        # 굴곡 쪽 한계. a=+1 이 여기로 간다. 역굴곡은 액션 공간 **밖**이라
        # 클램프도 벌점도 필요 없다.
        _jl0 = self.robot.data.soft_joint_pos_limits[0]
        _mg = float(self.cfg.hand_limit_margin)
        _lo = _jl0[self._hand_free_t, 0] + _mg
        _hi = _jl0[self._hand_free_t, 1] - _mg
        self._flex_limit = torch.where(self._flex_sign > 0, _hi, _lo)  # (n_free,)
        self._hand_home_free = self.robot.data.default_joint_pos[
            0, self._hand_free_t].clone()
        _span = (self._flex_limit - self._hand_home_free).abs()
        if float(_span.min()) < 1e-3:
            _dead = [_jn[int(j)].split("hj_")[-1]
                     for j, s in zip(self._hand_free_t.tolist(), _span.tolist())
                     if s < 1e-3]
            raise RuntimeError(
                f"[{p.name}] 가동폭 0 인 자유 관절 {_dead} — 액션을 줘도 안 움직인다. "
                "고정 목록에 넣거나 한계를 확인하라")
        _neg = [_jn[int(j)].split("hj_")[-1]
                for j, s in zip(self._hand_free_t.tolist(), signs) if s < 0]
        print(f"[grasp_lift_fabric] 굴곡 부호 실측 {len(signs)}관절 · "
              f"음수(−q 가 굴곡) {_neg if _neg else '없음'} · "
              f"최소 판별폭 {worst * 1000:.1f}mm · "
              f"가동폭 {float(_span.min()):.2f}~{float(_span.max()):.2f} rad", flush=True)

    def _report_home_cage(self) -> None:
        """자매 보고 + **관통 위험은 경고가 아니라 정지**로 올린다.

        ★자매의 `object_spawn_center` 는 **자매 홈 케이지**에서 역산된 값이다.
          우리는 `pinky_1 = 0` 홈 오버라이드로 손끝 배치가 달라지므로 그 상수가
          그대로 유효한지 부팅에서 다시 검산해야 한다.
        """
        super()._report_home_cage()
        p = self.profile
        tips = (self.robot.data.body_pos_w[:, self._tip_ids_t]
                - self.scene.env_origins[:, None, :])[0]
        _a = int(self._group_a_idx[0])
        _others = [i for i in range(len(self.tip_ids)) if i != _a]
        cage_xy = 0.5 * (tips[_a] + tips[_others].mean(dim=0))[:2]
        cup_xy = torch.tensor(p.object_spawn_center[:2], device=cage_xy.device)
        gap = float((cage_xy - cup_xy).norm())
        if gap < self._r_cage:
            raise RuntimeError(
                f"[{p.name}] 홈 케이지↔컵 수평 간격 {gap * 1000:.0f}mm 가 케이지 반경 "
                f"{self._r_cage * 1000:.0f}mm 보다 좁다 — 리셋에서 손가락이 컵을 "
                "관통한다. object_spawn_center 또는 홈을 조정하라.")
        print(f"[grasp_lift_fabric] 케이지↔컵 수평 간격 {gap * 1000:.0f}mm > "
              f"반경 {self._r_cage * 1000:.0f}mm ✓", flush=True)

    # ==================================================================
    # 손 — 자유 관절 절대 목표 (시너지 자리를 대신한다)
    # ==================================================================
    def _synergy_targets(self, a_hand: torch.Tensor) -> torch.Tensor:
        """액션(자유 관절 13D) → 손 관절 목표 (N, n_hand) · **synergy 순서**.

        `target = 홈 + 0.5(a+1)·(굴곡한계 − 홈)`
        · a=−1 → 홈(펴짐) · a=+1 → 굴곡 한계. 고정 관절은 홈에 머문다.

        ★자매의 변화율 상한·닫기 게이트·접촉 동결은 여기 없다 — 셋 다 **누산 delta**
          위에서만 뜻이 있는 시너지 기구다. 우리 액션은 속도가 아니라 절대 각도라
          정책이 언제든 되돌릴 수 있고, 접근 중 손을 열어 두는 것도 a=−1 로 표현된다.
          (보상 쪽 `close_gate` 는 자매와 동일하게 살아 있다 — cfg 주석 참조.)
        """
        u = 0.5 * (a_hand.clamp(-1.0, 1.0) + 1.0)                       # (N, n_free)
        tgt = self._syn_home.unsqueeze(0).repeat(self.num_envs, 1)
        tgt[:, self._free_syn_idx] = (
            self._hand_home_free + u * (self._flex_limit - self._hand_home_free))
        # ★`_syn_close` 는 **지령** 진행도다(로깅·리셋 정합용). 보상이 쓰는 폐쇄도는
        #   자매 `_close_progress` 의 **실측 관절** 값이다(72ac912) — 지령을 재면 손이
        #   테이블에 눌려 펴져도 만점이 나온다.
        self._syn_close = torch.zeros_like(self._syn_close)
        self._syn_close[:, self._free_syn_idx] = u
        return tgt.clamp(self._syn_lo.unsqueeze(0), self._syn_hi.unsqueeze(0))
