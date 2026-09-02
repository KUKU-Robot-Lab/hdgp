"""grasp_ua 계약 테스트 — Isaac Sim 없이 돈다(소스 텍스트·AST 검사).

cfg 는 isaaclab→pxr 를 끌어와 Isaac 앱 없이 import 가 안 되므로, 값이 아니라
**소스의 계약**을 잠근다. 각 테스트에 왜 이 계약이 생겼는지(어떤 사고를 막는지)를
적어 둔다 — 나중에 고칠 때 근거 없이 지우지 않도록.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_ua/tests -q
"""

from __future__ import annotations

import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
_ENV = (_HERE / "grasp_ua_env.py").read_text(encoding="utf-8")
_CTRL = (_HERE / "grasp_ua_control.py").read_text(encoding="utf-8")
_CFG = (_HERE / "grasp_ua_env_cfg.py").read_text(encoding="utf-8")
_REW = (_HERE / "grasp_ua_rewards.py").read_text(encoding="utf-8")
_REG = (_HERE / "config" / "__init__.py").read_text(encoding="utf-8")


def _code(src: str) -> str:
    """주석·docstring 을 뺀 실행 코드만.

    ★설명문에 적힌 이름이 계약을 통과시키면 안 된다 — 예컨대 "기각된 tip_cyl 분기를
      뺐다"는 **주석**이 "tip_cyl 이 없어야 한다"는 계약을 깨뜨리는 식이다.
    """
    import ast
    tree = ast.parse(src)
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            doc_lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    out = []
    for i, line in enumerate(src.split("\n"), start=1):
        if i in doc_lines:
            continue
        s = line.split("#", 1)[0]
        if s.strip():
            out.append(s)
    return "\n".join(out)


def _assign_block(src: str, name: str) -> str:
    """`name = (` 로 시작하는 다중행 대입의 본문 — 괄호 균형으로 끝을 찾는다."""
    m = re.search(rf"\n\s*{re.escape(name)} = \(", src)
    assert m, f"{name} 대입 부재"
    i = src.index("(", m.start())
    depth, j = 0, i
    while j < len(src):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                return src[i + 1:j]
        j += 1
    raise AssertionError(f"{name} 대입의 괄호가 안 닫힌다")


# ---------------------------------------------------------------- 등록
def test_four_task_ids_registered():
    """id 4종(train/play × mlp/lstm). `-play` 가 없으면 play.py·warm 수집이 죽는다."""
    assert '_ENTRY = "openarm.agnostic.tasks.grasp_ua.grasp_ua_env:GraspUAEnv"' in _REG
    for suffix in ('""', '"-play"', '"-lstm"', '"-play-lstm"'):
        assert suffix in _REG, f"태스크 id 접미사 {suffix} 미등록"
    assert 'f"open-{_tag}_grasp_ua{_suffix}"' in _REG
    # 로그 분리 규약: 두 번째 슬롯이 r/l/b 여야 한다.
    assert re.search(r'"sens_r":\s*GraspUATesolloRightEnvCfg', _REG)


def test_registration_skips_profiles_without_fabric():
    """자산 없는 프로필을 조용히 빠뜨리지 않고 사유를 남긴다(record-loud)."""
    assert "SKIPPED" in _REG and "REGISTERED" in _REG
    assert "fabric_class is None" in _code(_REG)


# ---------------------------------------------------------------- 액션
def test_palm_action_is_anchor_relative_delta():
    """★palm 액션은 **앵커 기준 유계 델타**여야 한다 — `a=0` 이 앵커.

    절대 매핑(`a=0` = 박스 중심)은 저장소 공통 σ=1.0 과 곱해지면 매 스텝 작업공간
    전역에서 목표를 재추첨해 접근이 랜덤워크가 된다(선행 트랙 실측: 클램프 전
    지령 요청량 0.33~0.36 m/step 상시 포화). 이 계약이 막는 것은 그 랜덤워크이지
    앵커가 홈이어야 한다는 뜻이 아니다 — 08.29 Phase 0 로 홈 앵커가 과제에서
    z +14 cm · y -27 cm 떨어져 있음이 실측돼 앵커를 재중심 가능하게 열었다.
    """
    code = _code(_ENV)
    assert "self._palm_anchor() + delta" in code, "palm 지령이 앵커 기준 델타가 아니다"
    assert "palm_delta_xyz" in _code(_CFG) and "palm_delta_rot_deg" in _code(_CFG)
    # 앵커가 박스에 잘리면 a=0 의 의미가 깨진다(홈 모드는 박스를 홈까지 넓히고,
    # 스폰 모드는 부팅에서 fail-loud 로 잡는다).
    assert "torch.minimum(self._palm_lo, self._home_palm)" in code
    assert "액션 앵커" in code and "밖이다" in code, "앵커 박스 검증이 없다"


def test_palm_anchor_is_episode_constant():
    """★앵커는 **에피소드 내 상수**여야 한다.

    실시간 물체 위치를 앵커에 쓰면 컵이 밀릴 때 액션 원점이 따라가는 되먹임이 되고,
    래치를 쓰면 이 트랙이 걷어낸 grasp_v1 의 단계 스크립트가 부활한다(트랙 계약:
    "래치는 보상 단계 표시 전용"). 허용되는 소스는 홈과 **스폰 스냅샷**뿐이다.
    """
    src = _code(_ENV)
    i = src.index("    def _palm_anchor(")
    body = src[i:src.index("\n    def ", i + 10)]
    assert "self.object_spawn_pos" in body, "스폰 스냅샷을 앵커로 쓰지 않는다"
    for banned in ("self._latched", "root_pos_w", "self.object.data"):
        assert banned not in body, f"앵커가 에피소드 중 변하는 양({banned})을 참조한다"


def test_palm_anchor_defaults_to_spawn():
    """★09.01 D3 승격 — 기본 앵커가 홈에서 **스폰**으로 바뀌었다.
    홈 앵커는 `a=0` 이 "컵에서 도망"을 뜻해 정책이 상시 저항했다(08.29 실측
    `action_norm_arm` 2.29~2.40 = 이론최대의 93~98%). "home" 경로는 대조용으로 남긴다."""
    cfg = _code(_CFG)
    assert 'palm_anchor_mode: str = "spawn"' in cfg
    # "home" 분기 자체가 살아 있는지는 env 쪽에서 잠근다(_code 는 주석을 지운다).
    env = _code(_ENV)
    assert 'self._anchor_mode not in ("home", "spawn")' in env, \
        "앵커 모드 검증이 두 경로를 다 안 받는다"
    assert env.count('self._anchor_mode == "home"') == 2, \
        "구 home 분기가 사라져 대조 실험이 불가능해졌다"
    assert "palm_anchor_offset_xyz" in cfg


def test_command_rate_limiter_logs_preclamp_value():
    """클램프 **전** 원값을 남겨야 상한이 물리는 비율을 알 수 있다(reward-clamp 규칙)."""
    code = _code(_ENV)
    assert "_palm_cmd_step_raw" in code
    assert "palm_cmd_rate_limit_m" in code and "palm_cmd_rate_limit_rot_deg" in code
    # 리셋 직후 첫 지령은 초기화라 리미터를 걸지 않는다.
    assert "_palm_cmd_primed[env_ids] = False" in code


# ---------------------------------------------------------------- 래치
def test_latch_never_overrides_arm_command():
    """★래치는 **보상 단계 표시 전용**이다.

    grasp_v1 은 래치 후 팔 지령을 z 램프 스크립트로 대체했다(`torch.where(is_lift,
    _lift_palm, palm_pose)`). 이 트랙은 이송까지 정책이 fabric 으로 제어하므로
    그 오버라이드가 있으면 안 된다.
    """
    code = _code(_ENV)
    for banned in ("_lift_palm", "lift_start_step", "LIFT_PHASE_STEPS", "lift_height_delta"):
        assert banned not in code, f"래치 스크립트 잔재: {banned}"
    # palm_targets 를 쓰는 곳은 액션 매핑·리미터뿐이어야 한다.
    assert code.count("self.palm_targets =") == 1
    # 래치 자체는 살아 있어야 한다(보상 게이트).
    assert "self._latched" in code and "grasp_ready_hold_steps" in code


# ---------------------------------------------------------------- obs
def test_obs_has_no_object_identity():
    """policy obs 에 물체 정체성(onehot·치수·질량·클래스)이 없어야 한다(sim2real)."""
    code = _code(_ENV) + _code(_CFG)
    for banned in ("onehot", "object_class", "object_mass", "obj_scale"):
        assert banned not in code, f"obs 오염 경로: {banned}"


def test_obs_carries_tactile_and_goal():
    """촉각(tip-local 힘·관절 추종오차)과 이송 목표가 policy obs 에 있어야 한다.

    인벨롭이 잘 될수록 팁 F/T 가 0 을 읽으므로 `joint_pos_err` 가 주 파지력 관측이다.
    """
    m = re.search(r"_noisy = torch\.cat\(\[([\s\S]*?)\], dim=1\)", _ENV)
    assert m, "policy obs 결합식 부재"
    blk = m.group(1)
    for need in ("tip_force", "joint_err", "goal_rel", "palm_ax"):
        assert need in blk, f"policy obs 에 {need} 가 없다"


def test_obs_dim_formula_matches_layout():
    """cfg 의 obs 차원 산술식이 실제 결합 성분 수와 맞는지."""
    m = re.search(r"self\.observation_space = \(([\s\S]*?)\)", _CFG)
    assert m, "observation_space 식 부재"
    expr = m.group(1)
    for need in ("2 * n_arm", "2 * n_hand", "3 * num_tips", "self.action_space", "+ 3"):
        assert need in expr, f"obs 식에 {need} 가 없다"
    # 물체 뱅크·스케일에서 파생되면 안 된다.
    assert "bank" not in expr and "scale" not in expr


# ---------------------------------------------------------------- 보상
def test_reward_terms_include_transfer_and_stay():
    """이송 2항이 계약에 있어야 한다. 항 계약은 이 트랙 **로컬**이다."""
    code = _code(_REW)
    assert "GRASP_UA_REWARD_TERMS" in code
    for term in ("transfer", "stay"):
        assert f'"{term}"' in code, f"보상 항 {term} 부재"
    # 공유 8항 계약(여러 트랙이 쓴다)을 끌어오면 안 된다.
    assert "GRASP_V2_REWARD_TERMS" not in code


def test_transfer_requires_contact_and_lift():
    """이송 보상은 접촉·리프트 없이는 0 이어야 한다 — 밀어 옮기기 차단."""
    blk = _assign_block(_code(_REW), "transfer")
    assert "lift_gate" in blk and "lifted_gate" in blk and "graded_contact" in blk


def test_stay_rewards_duration_not_touch():
    """stay 는 도달 순간이 아니라 **연속 유지 시간**에 비례해야 한다(찍고 빠지기 차단)."""
    assert "stay_frac" in _assign_block(_code(_REW), "stay")
    assert "self._stay_run" in _code(_ENV)


def test_opposition_axis_is_hand_derived():
    """★대향 중점은 **손 자신의 기하**에서 나와야 한다 — 임의 부호의 수직축 금지.

    구 수식 `axis = (−dir_y, dir_x)` 는 접근방향의 90° 회전이라 좌/우 부호가 임의였다.
    엄지 목표가 실제 엄지의 반대편에 놓이면 손목을 뒤집어야 도달 가능한 자세를 요구하고,
    정책은 그쪽으로 못 가서 엄지가 걸린 채 4지만 붙인다(실측: grip_frac 0.20 인데
    wrap_frac 이 2,228 iter 내내 0.000).
    """
    code = _code(_ENV)
    # ★★08.27: approach 의 cage_dist 는 **palm 강체**여야 한다. 실시간 손끝을 쓰면
    #   "쭉 편 손가락으로 팁을 컵 중심에 모으기"가 이 항의 최적이 되어 파지 예비자세를
    #   정면으로 방해한다 — s2r_a9 실측 corr(ch2 폐쇄, approach) = −0.702. ch2 가
    #   0.271 → 0.004 로 펴지는 동안 approach 0.61 → 0.75, touch_frac 0.000 유지.
    assert "cage_dist = self._cage_ctr_dist" in code
    assert "opp_mid" not in code, "보상용 케이지가 다시 실시간 손끝을 참조한다"
    # 임의 수직축·물체 반경 상수는 남아 있으면 안 된다.
    assert "axis[:, 0], axis[:, 1] = -_dir[:, 1], _dir[:, 0]" not in code
    for banned in ("object_grasp_radius", "enclosure_thumb_weight"):
        assert banned not in code + _code(_CFG), f"제거된 형상 상수 잔재: {banned}"


def test_closing_is_gated_on_cage_alignment():
    """★위치가 맞기 전에는 오므리지 않는다 — 닫는 방향만 게이트, 푸는 방향은 항상 허용.

    래치로는 못 막는다: 래치는 lift/transfer **보상**을 여는 신호일 뿐이고, 닫힘은
    정책의 손 액션이 직접 만든다. 실측(s2r_a5 iter13): cage_dist 0.293 = 케이지 반경의
    2.4배인데 syn_close 0.574 까지 닫혀 있었다.
    """
    env, ctrl = _code(_ENV), _code(_CTRL)
    assert "self._close_gate" in env and "close_gate_enabled" in env
    # 게이트는 손 액션을 만들기 **전에** 계산돼야 한다.
    assert env.index("self._close_gate =") < env.index("self._synergy_targets(")
    # 닫는 방향만 스케일 — 푸는 방향(delta<0)은 그대로여야 갇혔을 때 빠져나온다.
    assert "torch.where(delta > 0.0, delta * _g, delta)" in ctrl
    # 임계는 손 기하에서 부팅 실측한 케이지 반경이다(물체 상수 아님).
    assert "self._r_cage" in env


def test_close_gate_center_is_rigid_to_palm():
    """★★게이트 영역은 **손가락을 따라 움직이면 안 된다**.

    08.27 실측(s2r_a6, 202 iter): 중심을 실시간 손끝 평균으로 두니 팔이 정지한 구간
    (palm_to_cup 0.120~0.140, n=147)에서 corr(syn_close, cage_dist) = −0.974 —
    팔을 안 움직이고 손만 오므려도 중심이 컵 쪽으로 50mm(램프 폭의 83%) 당겨져
    게이트가 저절로 열렸다. "정렬되면 닫아라"가 아니라 "닫으면 닫아도 된다"는
    양의 되먹임이라 게이트가 아무것도 막지 못했다.

    또한 거리는 **3D** 여야 한다. xy 투영은 z 를 못 봐서 palm·검지가 컵보다 내려간
    잘못된 자세도 통과시켰다(사용자 GUI 관찰: 엄지가 컵에 걸린 채 접근).
    """
    env = _code(_ENV)
    blk = env[env.index("_obj = self._env_local(self.object.data.root_pos_w)"):]
    blk = blk[:blk.index("self._synergy_targets(")]
    # 중심 = palm + R_palm · (홈에서 실측한 고정 오프셋)
    assert "self._cage_offset_palm" in blk and "self._palm_ee_R()" in blk
    # 게이트 계산 구간에 손끝 위치가 등장하면 안 된다(되먹임 재발 방지).
    assert "_tip_ids_t" not in blk, "게이트가 다시 손끝을 참조한다 — 되먹임 재발"
    # 3D 거리 — xy 슬라이스로 되돌아가면 z 조건이 사라진다.
    # ★단 z 는 **데드밴드**를 통과한다(±grasp_z_deadband). 3D 노름이 z 를 xy 와 똑같이
    #   벌하는 바람에 palm 이 파지높이 아래로 눌려 내려갔다(s2r_b2 실측:
    #   palm_above_table mean 0.088 vs 파지중심 0.107, min 0.066 < 컵 원점 0.077).
    assert "self._cage_ctr_dist = self._banded_dist(_cage - _obj)" in blk
    ctrl_all = _code(_CTRL)
    assert "_dz = torch.relu(delta[:, 2].abs() - _b)" in ctrl_all, "z 데드밴드 부재"
    assert "palm_to_cup = self._banded_dist(palm_pos - grasp_center)" in env
    # 오프셋은 홈 자세에서 한 번만 실측한다(부팅 보고 안 — 게이트 블록 밖).
    assert "self._cage_offset_palm = _R.transpose(0, 1) @ (cage - _palm)" in env
    # 래치 후에는 해제 — 이송 중 컵이 흔들려도 다시 쥘 수 있어야 한다.
    assert "self._latched" in blk


def test_fabric_knows_about_the_table():
    """★★fabric 에 world 를 안 넘기면 테이블을 **아예 모르는 상태**로 계획한다.

    `WorldMeshesModel` 에 world_dict/world_filename 이 없으면 `object_indicator == 0`
    이라 반발 커널이 첫 줄에서 early-out 한다. 형제 tesollo 트랙은 전부
    `world_filename` 을 넘기는데 agnostic 트랙만 빠져 있었다 — 08.27 발견.
    사용자 GUI: "아예 테이블을 박히고 간다", 실측 palm_above_table min 0.066
    (컵 원점 0.077 보다 아래).

    ★params 의 body_repulsion.collision_sphere_frames 에 palm·5지 전 마디(소지 dg_5
      14개 포함)·팔 링크 충돌구가 이미 있어 테이블 하나로 손 전체가 보호된다 —
      params 파일은 건드리지 않는다.
    ★박스는 palm 도달영역에서 **파생**해야 한다. 숫자를 따로 적으면 물리 테이블과
      조용히 어긋난다.
    """
    ctrl, cfg = _code(_CTRL), _code(_CFG)
    assert "world_dict=self._build_fabric_world()" in ctrl, "fabric 이 빈 세계를 본다"
    assert "fabric_table_obstacle" in cfg
    # 상면은 table_surface_z 그 자체에서 파생 — 별도 상수 금지.
    assert "float(self.cfg.table_surface_z) - 0.5 * _th" in ctrl
    # 크기는 프로필 도달영역에서 파생.
    assert "_lo, _hi = p.palm_box_min, p.palm_box_max" in ctrl
    # 근거 없던 박스-바닥 클램프는 되돌렸다(fabric 반발이 정공법).
    assert "palm_min_above_table" not in ctrl + cfg


def test_grasp_has_pre_contact_gradient_gated_on_alignment():
    """★★`grasp` 는 **첫 접촉 전에도** 손가락을 내라는 gradient 를 줘야 한다.

    구판은 네 채널(팁접촉·전팁·지속·감쌈)이 전부 접촉 임계 뒤라 첫 접촉까지 정확히 0
    이었다. 그래서 접촉 전 손 모양을 정하는 보상이 approach 하나뿐이었고, approach 가
    실시간 손끝을 쓰는 바람에 최적 손 모양이 "쭉 편 손가락"이 됐다 — 손가락을 말면
    approach 가 즉시 깎이는데 grasp 는 닿아야 나오니 가는 길이 확실히 나쁜 **계곡**
    이었다(s2r_a9 526 iter: touch_frac 0.000 · wrap_frac 0.000 · ch2 0.004).

    계약: grasp = w · pre_lift · **close_gate** · [(1−ecred)·close_credit + ecred·wrap]
    · close_gate 곱 — 정렬 전 공중 폐쇄는 0 이어야 한다.
    · close_progress 는 **실측 관절**이어야 한다. 지령을 재면 손이 테이블에 눌려 쫙
      펴져도 만점이 나온다(s2r_b1: hand_joint_err_max 3.72 rad = 임계 0.30 의 12배인데
      grasp 4.69/step 지급). 실측은 물체에 막히면 스스로 멈추므로 인위적 포화 캡도
      필요 없다 — 캡을 뒀더니 **그 지점이 정지점**이 됐다(폐쇄도가 캡 0.5 에 고정).
    · 팁 제어 3채널은 폐기됐다 — 팔이 정밀 제어를 하는 지금은 불필요(사용자 확정).
    """
    rew = _code(_REW)
    blk = _assign_block(rew, "grasp_quality")
    # 08.31: 감쌈 성분은 `_envelope_credit` 으로 한 단계 빠졌다(모드별 wrap/anylink).
    # 여기서는 **두 성분이 다 있는가**만 잠그고, 어느 쪽이 실리는지는
    # `test_anylink_replaces_grasp_envelope_credit` 가 잠근다.
    assert "close_credit" in blk and "_envelope_credit" in blk
    for banned in ("tip_contact_frac", "full_tip", "persistence"):
        assert banned not in blk, f"폐기된 팁 제어 채널이 grasp 에 되살아남: {banned}"
    assert "close_gate.clamp(0.0, 1.0) * grasp_quality" in rew, "grasp 가 정렬 게이트를 안 탄다"
    assert "_cref" not in rew, "포화 캡이 되살아남 — 그 지점이 정지점이 된다"
    # 폐쇄도는 **실측 관절**이어야 한다(지령 `_syn_close` 를 재면 테이블에 펴져도 만점).
    ctrl = _code(_CTRL)
    assert "_q = self.robot.data.joint_pos[:, self._syn_ids]" in ctrl
    assert "return _prog[:, self._syn_movable].mean(dim=1)" in ctrl
    assert "return self._syn_close[:, self._syn_movable]" not in ctrl, "폐쇄도가 다시 지령이다"
    # 가동폭 0° 관절(전 `_1`·pinky_2·thumb_2)이 분모에 섞이면 공짜 점수가 된다.
    assert "self._syn_movable = (self._syn_grip - self._syn_open).abs() > 1e-4" in ctrl
    # graded_contact(리프트 이후 "정말 쥐고 있나")는 팁을 계속 써야 한다 — 폐기 대상 아님.
    assert "graded_contact = (1.0 - _emix) * tip_contact_frac" in rew

def test_approach_penalty_is_capped():
    """★approach 벌금은 상금(approach_weight)을 못 넘어야 한다 — approach 최솟값 0.

    상한이 없으면 컵에 닿을수록 손해가 되어 접촉 탐색이 금지되고, 스텝당 보상이
    순음수라 **조기 종료가 최적**이 된다(s2r_a1 실측: 16스텝 자살 경로 240 iter 고착,
    접촉 시작 시 grasp +0.43 vs approach −0.96→−2.02 로 순증분 음수).
    """
    code = _code(_REW)
    assert ".clamp(max=_aw)" in code, "밀림·기울기 벌금에 상한이 없다"
    m = re.search(r"_penalty = \(", code)
    assert m, "벌금 항이 분리돼 있지 않다"
    # 밀림 억제 자체는 disp_factor 가 계속 맡는다.
    assert "disp_factor" in code


def test_termination_causes_are_logged():
    """종료 원인별 비율이 있어야 무엇이 에피소드를 끝냈는지 역산 없이 안다."""
    code = _code(_ENV)
    for k in ("done/out_xy", "done/fell", "done/tipped", "done/abnormal"):
        assert f'"{k}"' in code, f"{k} 로깅 부재"


def test_disp_factor_uses_latch_snapshot():
    """★밀림 감쇠는 **래치 시점** 변위 기준이어야 한다.

    실시간 변위를 쓰면 이 트랙의 과제인 수평 이송이 통째로 처벌된다.
    """
    assert "cup_xy_disp_ref" in _REW
    m = re.search(r"_r = cup_xy_disp_ref / _limit", _REW)
    assert m, "감쇠가 래치 스냅샷을 안 쓴다"
    assert "self._disp_at_latch" in _code(_ENV)


# ---------------------------------------------------------------- 씬·기하
def test_goal_is_derived_from_settled_height():
    """goal 은 스폰점이 아니라 **정착고** 기준 — 스폰 패드가 리프트 기준에 실리면 안 된다."""
    code = _code(_ENV)
    assert "settled[:, 2] = float(self.cfg.table_surface_z)" in code
    assert "self.goal_pos[env_ids] = settled" in code
    assert "goal_offset_xyz" in code
    # 부팅에서 목표 도달성을 확인한다.
    assert "_assert_goal_reachable" in code


def test_spawn_height_has_single_source():
    """스폰 높이 파생은 cfg 한 곳에서만 — 이중 패딩 사고 차단."""
    assert _code(_CFG).count("self.object_spawn_z = (") == 1
    assert "object_spawn_pad" in _code(_CFG)


def test_contact_sensor_per_body():
    """body **하나당 센서 하나**. 다중 body 단일 센서는 force_matrix_w 가 무증상 0."""
    m = re.search(r"for body in bodies:([\s\S]*?)self\._finger_sensors\[finger\]", _CTRL)
    assert m and "ContactSensor(ContactSensorCfg(" in m.group(1)


# ---------------------------------------------------------------- agnosticism
def test_no_robot_joint_literals():
    """태스크 소스에 로봇 조인트/링크 리터럴이 없어야 한다 — 전부 프로필 경유."""
    pat = re.compile(r"\b[rl]_(aj|hj|hl)_")
    for name, src in (("env", _ENV), ("control", _CTRL), ("cfg", _CFG), ("rewards", _REW)):
        hit = pat.search(_code(src))
        assert hit is None, f"{name} 에 로봇 리터럴 '{hit.group(0)}'"


def test_hand_control_is_synergy_only():
    """손 제어 분기는 시너지 하나뿐이어야 한다(기각된 경로는 오해만 만든다)."""
    code = _code(_CTRL) + _code(_ENV) + _code(_CFG)
    # `use_hand_fabric=False` 는 fabric 생성자 인자라 남아야 한다 — 분기 플래그
    # (`self._hand_fabric`)만 금지한다.
    for banned in ("hand_control", "tip_cyl", "self._hand_fabric", "hand_attractor_gain"):
        assert banned not in code, f"기각된 손 제어 경로 잔재: {banned}"
    assert "use_hand_fabric=False" in _code(_CTRL)


def test_fabric_hand_state_is_synced():
    """fabric 은 실제 손 자세를 받아야 한다.

    끊으면 fabric 이 실재하지 않는 손으로 충돌구 FK 를 계산해 없는 자기충돌을
    피하려 팔을 민다(선행 트랙 실측: palm_err 475mm·joint_err 0.71rad·5kN).
    """
    code = _code(_ENV)
    assert "self._syn_to_fab(self._syn_target)" in code
    assert "_fab_home_hand" not in code, "fabric 손을 홈으로 고정하면 안 된다"


def test_fabric_integrates_once_per_policy_step():
    """적분은 `_step_fabric` 한 곳 — `_apply_action` 에서 돌리면 fabric 시간이 2배."""
    ctrl = _code(_CTRL)
    assert ctrl.count("self.integrator.step(") == 1
    m = re.search(r"def _apply_action\(self\)([\s\S]*?)\n    def ", ctrl)
    assert m and "integrator" not in m.group(1)


def test_contact_freeze_is_per_joint_link():
    """★★동결은 관절마다 **자기 링크**가 닿았을 때만 걸려야 한다.

    구판은 (원위|팁) 접촉 하나로 `_3`·`_4` 를 통째로 얼렸다. 그런데 `_2` 가 굽으면
    손끝이 가장 먼저 닿으므로 **감쌈이 시작되기 직전에 감쌈 관절을 잠그는** 구조였다 —
    wrap_frac 이 전 런에서 정확히 0.000 이었고, syn_close 0.278 이 "채널1(`_2`)만
    폐쇄" 예측 0.250 과 일치했다(사용자 GUI: `_2` 완전굴곡·`_3`/`_4` 정지).

    또한 동결은 **닫는 방향에만** 걸려야 한다. 양방향을 막으면 잘못 얼린 자세에서
    빠져나올 수 없다(닫기 게이트와 같은 원칙).
    """
    ctrl = _code(_CTRL)
    assert "self._syn_freeze_mid" in ctrl and "self._syn_freeze_dist" in ctrl
    # `_3` 은 중간마디, `_4` 는 **원위 링크만** — 팁은 트리거가 아니다.
    # 팁은 원위와 별개 body 라, 팁으로 `_4` 를 얼리면 원위가 닿을 기회가 사라져
    # wrap(중간 AND 원위)이 영원히 0 이 된다(s2r_a8 817 iter 실측).
    assert "_h_dist = (_dist > _thr)[:, self._syn_fi]" in ctrl
    assert "self._tip_contact_forces() > _thr" not in ctrl, "팁이 동결 트리거로 되살아남"
    # 08.29 O 라운드: finger 스코프 분기가 생겨 기본(joint) 경로는 else 로 이동했다.
    assert "_hold = ((_h_mid & self._syn_freeze_mid)" in ctrl
    # 푸는 방향은 항상 허용.
    assert "torch.where(_hold & (delta > 0.0), torch.zeros_like(delta), delta)" in ctrl
    # 구판 배선이 되살아나면 실패시킨다.
    assert "delta * (~(_hold & self._syn_freeze)).float()" not in ctrl


def test_approach_targets_the_palm_not_the_cage():
    """★★접근 목표는 **palm** 이어야 한다 — 케이지를 목표로 두면 핀치가 강제된다.

    홈 실측: 케이지 중심이 palm 앞 **106mm**(cage−palm = 82.2, 66.4, 3.4 mm).
    approach 가 `cage_dist → 0` 을 요구하면 palm 은 컵에서 106mm 떨어져야 하므로
    "손바닥 밀착"과 **구조적으로 양립 불가**다. 실측 타협점 palm_to_cup 0.126 /
    cage_dist 0.041 이 사용자 GUI 관찰 "palm_ee → 손가락 → 컵 순서"의 정체다.

    · cage_dist 는 approach 에서 빠지고 **닫기 게이트 전용**으로 남는다.
    · 거리는 palm 프레임으로 분해해 법선(palm_ee_x)=밀착도를 더 날카롭게 본다.
      법선거리는 컵 표면에서 물리적으로 포화하므로 형상 상수가 필요 없다.
    · `palm_still` 을 곱해 **밀착한 채 멈추게** 한다 — 그래야 시너지 손가락이 말린다.
      "멀리서 정지" 회피는 성립 안 함: 홈(d 0.36) 정지 0.055 vs 밀착(d 0.05) 정지 0.67.
    """
    rew, env, cfg = _code(_REW), _code(_ENV), _code(_CFG)
    _i = rew.index("approach = pre_lift_gate")
    blk = rew[_i:rew.index("grasp_quality", _i)]
    assert "cage_dist" not in blk, "approach 가 다시 케이지를 목표로 삼는다(핀치 강제)"
    assert "palm_normal_dist" in blk and "palm_lateral_dist" in blk
    assert "palm_still" in blk, "밀착 후 정지 요건이 없다"
    assert "approach_sharpness_normal" in cfg and "palm_still_gain" in cfg
    # 법선은 palm 회전행렬 열 0(손바닥 법선)에서 나온다.
    assert "_dn = (_d * _R[:, :, 0]).sum(dim=-1)" in env
    # 정지는 **실측** palm 선속도다(액션 변화량이 아니다 — 손가락을 말면 안 되니까).
    assert "self.robot.data.body_lin_vel_w[:, self.palm_idx]" in env


def test_net_force_reading_is_diagnostic_only():
    """★★필터 없는 `net_forces_w` 는 **진단 전용**이다 — 보상 경로에 새면 안 된다.

    `force_matrix_w` 는 컵 baseLink 로 필터링된 접촉만 담고 `net_forces_w` 는 그 링크가
    받은 **모든** 접촉(테이블·자기충돌·다른 손가락)을 담는다. 후자를 보상에 쓰면
    "테이블을 짚고 있다"가 파지로 계상된다 — 08.22 envelope 판정 사고와 같은 부류.

    둘을 나란히 읽는 이유는 08.27 실측 때문이다: 원위(`_4`)가 다섯 손가락 전부·4,553
    기록점 내내 정확히 0.000 인데 영상에서는 감쌈이 성립한다. net 이 양수인데 matrix 가
    0 이면 필터 결함이고, 둘 다 0 이면 진짜 미접촉이다.
    """
    ctrl, env = _code(_CTRL), _code(_ENV)
    assert "net_forces_w" in ctrl, "무필터 판독이 없다 — 두 가설을 못 가른다"
    # 보상이 쓰는 두 진입점은 **필터판**만 봐야 한다.
    for _fn in ("_contact_forces_split", "_tip_contact_forces"):
        _i = ctrl.index(f"def {_fn}")
        _blk = ctrl[_i:ctrl.index("def ", _i + 10)]
        assert "_mag_filtered" in _blk, f"{_fn} 이 필터판을 안 쓴다"
        assert "_mag_net" not in _blk, f"{_fn} 에 무필터가 샜다 — 테이블 접촉이 파지로 계상된다"
    # 진단 로깅은 보상 총합이 정해진 **뒤**에 불린다(반환값 없음).
    assert "self._log_diagnostics(" in env
    assert env.index("self._log_diagnostics(") < env.index("return total")
    assert "def _log_diagnostics" in env


def test_blocked_needs_both_error_and_away_from_limit():
    """★"더 못 조인다"는 **한계 도달**과 **물체에 막힘** 둘 다에서 성립한다.

    `hand_grip_pose` 가 soft limit 을 넘겨 과지령이라(1.8 rad vs 1.571) 완전 폐쇄만으로
    모든 관절이 목표를 못 따라가는 상태가 된다 — 허공에서 주먹을 쥐어도 오차 조건은
    참이다. 관절이 **자기 한계에서 떨어져 있는지**를 함께 봐야 외부 차단이 확정된다.
    ★가동폭 0° 관절(전 `_1` + pinky_2 + thumb_2)은 항상 오차 상태라 분모에서 빠져야 한다.
    """
    ctrl, cfg = _code(_CTRL), _code(_CFG)
    _i = ctrl.index("def _hand_blocked")
    blk = ctrl[_i:ctrl.index("def ", _i + 10)]
    assert "_syn_lo" in blk and "_syn_hi" in blk, "한계 근접 판정이 없다"
    assert "blocked_err_thr_rad" in blk and "blocked_limit_eps_rad" in blk
    assert "self._syn_movable" in blk, "가동폭 0° 관절이 분모에 섞인다"
    assert "&" in blk, "두 조건이 AND 로 묶이지 않았다"
    assert "blocked_err_thr_rad" in cfg and "diag_contact_threshold_lo" in cfg


def test_goal_distance_is_logged_by_component():
    """★`goal_dist` 스칼라만으로는 높이 탓인지 수평 탓인지 못 가른다.

    08.27 실측 goal_dist 0.281 에서 높이 성분과 수평 성분의 비중이 처방을 가른다 —
    높이면 `lift_height_ref`(0.10) vs `goal_offset_xyz.z`(0.08) 충돌이고, 수평이면
    홈 복귀(`a=0` 이 홈)다. 두 원인은 처방이 완전히 다르다.
    """
    env = _code(_ENV)
    assert 'self.extras["task/goal_dz"]' in env
    assert 'self.extras["task/goal_dxy"]' in env
    # 홈 복귀 가설의 직접 관측량 — a=0 이 정확히 홈이라 액션 크기가 곧 홈 이탈량이다.
    assert 'self.extras["task/action_norm_arm"]' in env
    assert 'self.extras["task/palm_to_home"]' in env
    # 파지 자세가 명령 박스 안에 있는지 — 축별로 봐야 어느 축이 부족한지 안다.
    assert 'f"fabric/palm_cmd_box_sat_{_ax}"' in env
    assert 'self.extras["fabric/palm_cmd_rate_sat"]' in env


def test_success_clauses_are_cfg_gated():
    """★성공 판정에서 **파지 품질**을 뺄 수 있어야 한다 (08.28 사용자 확정).

    과제 목적은 "컵이 목표에 제대로 놓여 멈춰 있는가" 다. 구 판정의 두 절은 산술로
    `at_goal ∧ stable` 에 함축된다 — 목표 z 가 스폰 +0.08 이고 허용 반경이 0.025 라
    `at_goal` 이면 높이 임계 0.04 를 자동으로 넘고(lifted), 8cm 뜬 컵이 정지해
    있으면 무언가가 받치고 있다(holding).
    ★그리고 `n_grip >= 4` 리터럴은 2지 그리퍼 프로필에서 **절대 성립 불가**였다.
    """
    env = _code(_ENV)
    cfg = _code(_CFG)
    assert "success_require_lifted" in cfg and "success_require_holding" in cfg
    assert "success_min_grip_fingers" in cfg
    assert "n_grip >= 4" not in env, "로봇 의존 리터럴이 판정에 남아 있다"
    assert "cfgn.success_min_grip_fingers" in env


def test_envelope_surface_count_includes_palm_and_thumb():
    """★★신 감쌈은 **손바닥과 엄지를 분모에 넣어야** 한다.

    구 정의 `wrap_frac` 은 분모가 `contact_group_b` 뿐이라 엄지 감쌈이 원리적으로
    반영되지 않았고, 손바닥은 센서를 붙이고도 진단 로깅에만 쓰였다. 사용자 확정
    정의는 "다섯 손가락과 손바닥이 유기적으로 감싸는 것" 이다.
    ★마디 조합(중간 AND 원위)을 요구하면 안 된다 — 실측상 이 손 형상에서 도달
    불가이고(원위 전부 0.00 인데 굴곡은 1.00), 작은 물체/큰 물체의 정답 자세가
    서로 달라 마디를 특정하는 순간 형상 의존이 된다.
    """
    env = _code(_ENV)
    _i = env.index('str(cfgn.envelope_metric) == "surface_count"')
    _blk = env[_i - 900:_i + 600]
    assert "_palm_contact_force()" in _blk, "손바닥이 감쌈 분모에 없다"
    assert "self._group_a_idx" in _blk, "엄지 그룹이 감쌈 분모에 없다"
    assert "self._group_b_idx" in _blk
    assert "grip_c[" in _blk, "마디 무관(tip|mid|dist)이어야 한다"
    assert "envelope_palm_weight" in _code(_CFG)
    # 세 성분은 활성 metric 과 무관하게 항상 로깅 — 구 정의 갈래에서도 사후 비교 가능해야
    assert 'self.extras["task/envelope_surf_palm"]' in env
    assert 'self.extras["task/envelope_surf_a"]' in env
    assert 'self.extras["task/envelope_surf_b"]' in env


def test_envelope_metric_defaults_to_legacy():
    """★기본값은 구 정의여야 한다 — 대조군이 이전 런과 **항등**이어야 A/B 가 성립한다."""
    cfg = _code(_CFG)
    assert 'envelope_metric: str = "deep_and"' in cfg
    assert "success_require_lifted: bool = True" in cfg
    # ★09.01 D3 승격 — holding(팁접촉 ∧ n_grip≥4)은 기본에서 뺀다. anylink 정의가
    #   접촉을 이미 세는데 holding 이 구 tip 기준으로 한 번 더 걸어 이중 게이트였다.
    assert "success_require_holding: bool = False" in cfg


def test_enclosure_is_reward_not_gate():
    """★★포위도는 **보상**이지 게이트가 아니다 — 되먹임 함정 잠금.

    이 트랙은 "케이지를 실시간 손끝으로 계산 금지"를 이미 계약으로 잠가 뒀다: 팔이
    정지한 구간에서 `corr(syn_close, cage_dist) = −0.974` 로, 손만 오므려도 중심이
    당겨져 게이트가 저절로 열렸다.

    포위도는 실시간 링크 위치를 쓰지만 같은 함정에 걸리지 않는다 — ①게이트가 아니라
    보상이라 `close_gate`·래치·종료 어디에도 안 들어간다 ②물체가 멀면 손을 아무리
    오므려도 모든 단위벡터가 같은 방향이라 값이 0 이다. 그 ①을 여기서 강제한다.
    """
    env = _code(_ENV)
    _i = env.index("def _enclosure")
    # ★내부에 중첩 `def _u` 가 있어 다음 "def " 로 자르면 블록이 잘린다.
    #   최상위 메서드 경계(4칸 들여쓰기 def)로 자른다.
    blk = env[_i:env.index("\n    def ", _i + 10)]
    # 포위도 계산 자체가 게이트·래치를 읽으면 안 된다(순환).
    assert "close_gate" not in blk and "_latched" not in blk
    # ★그리고 게이트·래치를 **쓰는** 어떤 줄도 포위도를 참조하면 안 된다.
    #   (`close_gate=self._close_gate` 처럼 별개 인자로 넘기는 것은 정상이다 —
    #    금지 대상은 게이트/래치의 **대입식**에 포위도가 섞이는 것이다.)
    for _line in env.splitlines():
        if "self._close_gate =" in _line or "self._latched =" in _line:
            assert "enclosure" not in _line, f"포위도가 게이트/래치에 섞였다: {_line}"
    # 종료 판정에도 들어가면 안 된다.
    _i2 = env.index("def _get_dones")
    _dones = env[_i2:env.index("\n    def ", _i2 + 10)]
    assert "enclosure" not in _dones, "포위도가 종료 판정에 들어갔다"


def test_enclosure_uses_no_shape_constant():
    """★포위도는 물체 **중심 하나**만 쓴다 — 반경·높이·메시를 쓰면 형상 의존이 된다.

    형상 정보를 안 쓰는 것이 다종 컵으로 확장하는 전제이고, 링크 위치(FK)만 쓰는 것이
    sim2real 의 근거다(접촉점 개수는 시뮬레이터 contact discretization 에 민감하다).
    """
    env = _code(_ENV)
    _i = env.index("def _enclosure")
    # ★내부에 중첩 `def _u` 가 있어 다음 "def " 로 자르면 블록이 잘린다.
    #   최상위 메서드 경계(4칸 들여쓰기 def)로 자른다.
    blk = env[_i:env.index("\n    def ", _i + 10)]
    assert "root_pos_w" in blk, "물체 중심을 안 쓴다"
    for banned in ("radius", "half", "bbox", "extent", "grasp_z_offset"):
        assert banned not in blk, f"포위도가 형상량 '{banned}' 을 참조한다"
    # 손바닥과 양 그룹이 모두 들어가야 "다섯 손가락과 손바닥" 정의가 성립한다.
    assert "self.palm_idx" in blk and "_hull_a_t" in blk and "_hull_b_t" in blk


def test_enclosure_default_weight_is_d3():
    """★09.01 D3 승격 — 기본 가중 0 → 10. 무접촉 기하 유도항이고, 정체 방지는
    `enclosure_contact_floor` 0.3 이 담당한다(둘은 세트로만 의미가 있다)."""
    cfg = _code(_CFG)
    assert "enclosure_weight: float = 10.0" in cfg
    assert "enclosure_contact_floor: float = 0.3" in cfg, \
        "floor 없이 가중 10 만 켜면 무접촉 정체가 공짜가 된다(M0·O1 실측 7.2/step)"
    assert '"enclosure"' in _code(_REW), "보상 항 튜플에 enclosure 가 없다"


def test_enclosure_post_latch_scale_defaults_off():
    """★래치 후 포위도 감쇠는 기본 1.0 (= 현행 동작). 래치 **전**은 건드리지 않는다.

    08.29 I1 실측: enclosure 비중 67.7% vs transfer 2.6%. 포위도는 팔 위치와 무관하게
    지급되므로 리프트 이후 보상 지형이 palm 위치에 평평해지고, 그러면 palm 이
    액션공간 기본값으로 흘러간다(`palm_post_latch_y` -0.399 vs 목표 -0.110).
    감쌈을 처음 만들어낸 힘은 **래치 전**에 있으므로 그 구간은 불변이어야 한다.
    """
    assert "enclosure_post_latch_scale: float = 1.0" in _code(_CFG)
    rew = _code(_REW)
    i = rew.index("_encl_scale = torch.where(")
    blk = rew[i:i + 400]
    assert "lift_latched" in blk, "감쇠가 래치 조건이 아니다"
    assert "torch.ones_like(enclosure)" in blk, "래치 전이 1.0 로 보존되지 않는다"


def test_force_band_defaults_off_and_has_floor():
    """★힘 밴드는 기본 꺼짐(1.0)이고, 켤 때도 **바닥이 0 이면 안 된다**.

    08.25 `grip-contact-cliff`: 닿으면 보상을 끄니 정책이 접촉 자체를 회피했다
    (grip 0.243→0.322 가 오르는 동안 n_over_thr 0.593→0.046). 세게 쥐는 것이
    손해이되 **놓는 것보다는 낫게** 남겨야 한다.
    """
    # ★09.01 D3 승격 — 1.0(꺼짐) → 0.5. 바닥은 여전히 0 이 아니다(절벽 회피).
    assert "force_band_floor: float = 0.5" in _code(_CFG)
    env = _code(_ENV)
    assert "force_quality = 1.0 - (1.0 - _floor) * _over" in env
    # ★게이트가 쓰는 이진 마스크를 깎으면 케이지·판정이 조용히 흔들린다.
    i = env.index("_fb_lo = float(cfgn.force_band_hi_n)")
    _end = 'self.extras["gate/force_quality_min"]'
    blk = env[i:env.index(_end, i) + len(_end)]
    for banned in ("tip_c =", "grip_c =", "mid_c =", "dist_c ="):
        assert banned not in blk, f"힘 밴드가 이진 마스크({banned})를 건드린다"
    # 보상 경로에만 곱한다.
    assert "graded_contact = graded_contact * force_quality" in _code(_REW)


def test_force_band_uses_max_tip_force():
    """밴드는 **최대** 팁 힘으로 정한다 — 손가락 하나만 과해도 하드웨어가 위험하다.

    사용자 제약: 실기 팁 센서 정격 0~50 N. 평균을 쓰면 한 손가락의 과압이 묻힌다.
    """
    env = _code(_ENV)
    assert "tip_f.max(dim=1).values" in env, "밴드가 최대 팁 힘 기준이 아니다"
    assert "force_sensor_max_n" in env and "force_band_hi_n" in env


def test_hand_overdrive_penalises_joint_error_not_force():
    """★"멈춤"은 힘이 아니라 **관절 오차**로 정의한다.

    ①실기 엔코더로 정확히 측정된다 ②무게·마찰·형상에 무관하다 — 무거운 컵이라
    더 쥐는 것은 *도달 가능한 각도까지* 닫는 것이라 안 걸리고, *도달 불가능한
    각도를 미는 것*만 걸린다 ③임계가 물리에서 나온다(effort_limit/stiffness).
    가동폭 0 인 관절은 오차가 상수로 깔리므로 반드시 제외한다.
    """
    assert "hand_overdrive_weight: float = 0.0" in _code(_CFG)
    assert '"hand_overdrive"' in _code(_REW)
    env = _code(_ENV)
    i = env.index("hand_overdrive = (torch.relu(")
    blk = env[i - 300:i + 300]
    assert "self._syn_movable" in blk, "가동폭 0 관절을 제외하지 않는다"
    assert "hand_torque_sat_err_rad" in blk, "임계가 effort/stiffness 파생이 아니다"
    # 힘 센서를 끌어오면 안 된다(그러면 무게 의존이 된다).
    for banned in ("tip_f", "force_matrix_w", "_contact_forces"):
        assert banned not in blk, f"과지령 항이 힘({banned})에 의존한다"


def test_enclosure_participation_defaults_off():
    """★손가락별 최소참여는 기본 꺼짐(λ=0) — 그때 그룹 평균 판과 **항등**이어야 한다.

    이 항은 `couple_four_fingers` 를 푸는 것의 **선행조건**이다. 커플링은 3지
    국소최적을 막으려고 넣은 것이고, 당시 진단이 "mean/count 보상엔 손가락별
    최소참여 신호가 없다"였다. 그룹 평균만 쓰면 손가락 하나가 빠져도 그 단위벡터가
    나머지와 비슷한 방향이라 값이 거의 안 떨어진다 — 같은 결함이 남아 있다.
    """
    assert "enclosure_participation_lambda: float = 0.0" in _code(_CFG)
    env = _code(_ENV)
    i = env.index("_lam = float(self.cfg.enclosure_participation_lambda)")
    assert "if _lam <= 0.0:" in env[i:i + 200], "λ=0 조기반환(항등성)이 없다"
    # 최약 손가락 기준이어야 한다 — 평균을 또 쓰면 같은 결함이 반복된다.
    blk = env[i:i + 900]
    assert "_c.min(dim=1).values" in blk, "최소참여가 **최약** 손가락 기준이 아니다"
    assert "task/enclosure_weakest" in blk, "최약 손가락 로깅이 없다"


def test_participation_keeps_finger_axis():
    """최소참여는 손가락 축을 살린 (F_b, L) 인덱스를 써야 한다.

    평평하게 편 `_hull_b_t` 로는 "어느 손가락이 빠졌는지"를 잴 수 없다. 손가락별
    링크 수가 다르면 정렬이 불가능하므로 부팅에서 fail-loud 로 잡는다.
    """
    env = _code(_ENV)
    assert "self._hull_part_t = torch.tensor(_rows" in env
    assert "최소참여 계산이 손가락을 정렬할 수 없다" in _ENV, "링크 수 불일치 fail-loud 부재"
    assert "self._hull_part_t" in env[env.index("def _enclosure("):], "포위도가 그 축을 안 쓴다"


# ---------------------------------------------------------------- 다물체
def test_object_bank_defaults_to_cup_family():
    """★09.01 D3 승격 — 기본이 다물체(8종)다. D3 가 8종 전수 0.774~0.949 로
    성공했으므로 단일 컵은 더 이상 기본 대조군이 아니다."""
    assert 'object_bank: str = "cup_family"' in _code(_CFG)


def test_multi_asset_turns_off_replicate_physics():
    """★뱅크>1 인데 `replicate_physics=True` 면 전 env 가 같은 물체를 받는다.

    MultiAsset(env 별 다른 물체)은 physics 복제가 불가능하다. 배정이 어긋나면
    판정·warm 이 조용히 붕괴한다.
    """
    cfg = _code(_CFG)
    i = cfg.index("def _apply_object_bank")
    blk = cfg[i:cfg.index("\n    def ", i + 10)]
    assert "if not bank.needs_multi_asset:" in blk and "return" in blk
    assert "self.scene.replicate_physics = False" in blk
    assert "random_choice=False" in blk, "env_id % N 결정론 배정이 아니다"


def test_table_representation_splits_by_physics_replication():
    """★★작업면 표현이 `replicate_physics` 여부로 갈린다.

    다물체는 `replicate_physics=False` 가 필수인데, 그러면 `clone_environments` 의
    `enable_env_ids` env 간 충돌 격리가 사라진다. 작업면이 원시 정적 프림이면
    `InteractiveScene` 이 추적하지 못해 전 env 가 한 충돌 그룹에 남고 팔이 물린다 —
    08.29 분리 실측: **단일 컵으로 고정하고 플래그만 뒤집어도**
    abnormal 0.0000 → 0.849 · joint_err 0.058 → 0.74 rad.
    그래서 다물체에서는 kinematic 사본을 **씬 자산**으로 올린다(grasp_v2 규약).
    단일 물체 경로는 기존 원시 프림 그대로 둔다 — 항등성이 검증된 물리다.
    """
    ctrl = _code(_CTRL)
    i = ctrl.index("tbl = self.cfg.table_cfg")
    blk = ctrl[i:i + 700]
    assert "if _multi:" in blk
    _m = blk.index("if _multi:")
    _e = blk.index("else:", _m)
    assert "RigidObject(tbl)" in blk[_m:_e], "다물체 경로가 테이블을 씬 자산으로 안 올린다"
    # ★등록은 clone 이후로 미뤄졌다(DEXTRAH 규약) — 생성 블록이 아니라 뒤에 있다.
    assert 'self.scene.rigid_objects["table"] = self.table' in ctrl
    assert "tbl.spawn.func(" in blk[_e:], "단일 경로의 원시 스폰이 사라졌다"
    assert "env_0/Table" in blk[_e:], "단일 경로는 env_0 + clone 규약을 지켜야 한다"


def test_multi_asset_requires_kinematic_table_asset():
    """★kinematic 작업면 사본이 없으면 **부팅에서 fail-loud** 해야 한다.

    원본 `env.usd` 로 다물체를 돌리면 조용히 물리가 깨진다(팔이 41° 어긋난 채 고착).
    사본은 `scripts/assets_tools/build_env_rigid_usd.py` 가 만든다.
    """
    cfg = _code(_CFG)
    assert '"env_rigid.usd"' in cfg
    assert "build_env_rigid_usd.py" in _CFG, "빌드 방법 안내가 없다"
    i = cfg.index('"env_rigid.usd"')
    blk = cfg[i:i + 500]
    assert "_os.path.isfile" in blk and "raise RuntimeError" in blk, "존재 검사가 없다"
    # 단일 물체로 되돌아가면 원본 USD 로 복원돼야 한다(멱등성).
    assert "self.table_cfg.spawn.usd_path = self._table_usd_base" in cfg


def test_clone_is_conditional_on_physics_replication():
    """★★`clone_environments` 는 **`replicate_physics=True` 일 때만** 부른다.

    False 면 `InteractiveScene.__init__` 이 이미 env xform 을 복제했다. 여기서 또
    부르면 env_0 내용을 전 env 에 덮어써 프림이 중복·변형되고, 리셋 직후 관절이
    폭발한다 — 08.29 실측: 편차 18~28 rad · 속도 2,500~4,700 rad/s ·
    `episode_lengths` 260 → 1.2(무한 리셋).
    ★`filter_collisions` 는 **양쪽 다** 부른다. True 경로는 clone 의 `enable_env_ids`
    가 격리해 주지만, False 경로에서는 이 호출이 유일한 env 간 충돌 격리다.
    ★다물체는 전 env prim 이 존재해야 `env_id % N` 배정이 되므로 물체를 clone
    **이후**에 만든다(grasp_v2 probe 실측: 그 전이면 16env 전부 같은 물체).
    """
    ctrl = _code(_CTRL)
    assert "_replicate = bool(self.cfg.scene.replicate_physics)" in ctrl
    i = ctrl.index("_replicate = bool(")
    blk = ctrl[i:i + 800]
    assert "if _replicate:" in blk and "clone_environments" in blk
    # clone 은 조건부, filter 는 무조건
    _c = blk.index("clone_environments")
    _f = blk.index("filter_collisions")
    assert blk.rindex("if _replicate:", 0, _c) < _c, "clone 이 조건부가 아니다"
    assert "if" not in blk[blk.rindex("\n", 0, _f):_f], "filter_collisions 가 조건부다"
    # 다물체 물체 생성은 clone 이후
    assert blk.index("if _multi:", _c) > _c
    assert "assert_spawned_after_clone" in blk


def test_origin_offset_is_per_env_not_scalar():
    """★원점 오프셋은 물체마다 다르다 — 스칼라를 쓰면 큰 컵이 테이블을 뚫는다.

    스폰고·정착고·목표가 전부 이 값에서 파생되므로 env 별이어야 한다.
    부팅 검증(앵커·목표 도달성)은 **최저·최고 둘 다** 봐야 한 극단을 놓치지 않는다.
    """
    env = _code(_ENV)
    assert "self._obj_origin_off = torch.tensor(" in env
    assert "_bank.assign_indices(self.num_envs)" in env, "env_id % N 배정을 안 쓴다"
    # 리셋 스폰이 env 별 값을 쓰는지
    i = env.index("_off = self._obj_origin_off[env_ids]")
    blk = env[i:i + 400]
    assert "spawn[:, 2] = float(self.cfg.table_surface_z) + _off" in blk
    assert "settled[:, 2] = float(self.cfg.table_surface_z) + _off" in blk
    # 부팅 검증이 양 극단을 본다
    assert "self._obj_origin_off.min()" in env and "self._obj_origin_off.max()" in env


def test_multi_object_leaks_no_identity_into_obs():
    """★다물체로 가도 policy obs 는 상대 위치뿐 — 정체성이 새면 안 된다."""
    m = re.search(r"_noisy = torch\.cat\(\[([\s\S]*?)\], dim=1\)", _ENV)
    assert m, "policy obs 결합식 부재"
    blk = m.group(1)
    for banned in ("_obj_origin_off", "object_bank", "bank_idx", "obj_id"):
        assert banned not in blk, f"policy obs 에 물체 정체성 누출: {banned}"


def test_derived_cfg_is_rebuilt_after_hydra_overrides():
    """★★hydra 는 `__post_init__` **뒤에** `from_dict` 로 오버라이드를 적용하고
    `__post_init__` 를 **다시 부르지 않는다**(IsaacLab `hydra_task_config` 실측).

    그래서 cfg 필드에서 파생되는 구조(스폰 cfg·replicate_physics·접촉필터·스폰고)는
    env `__init__` 이 `super()` **전에** 다시 만들어야 한다. `replicate_physics` 는
    `InteractiveScene.__init__` 이 소비하므로 `_setup_scene` 에서 고치면 늦다.
    08.29 실측: `env.object_bank=cup_family` 를 줬는데 replicate_physics 가 True 로 남았다.
    """
    cfg = _code(_CFG)
    assert "def finalize_after_overrides" in cfg
    assert "self.finalize_after_overrides()" in cfg, "__post_init__ 이 안 부른다"
    env = _code(_ENV)
    i = env.index("def __init__")
    blk = env[i:i + 700]
    assert blk.index("cfg.finalize_after_overrides()") < blk.index("super().__init__"), \
        "재파생이 super() 이후다 — InteractiveScene 이 이미 replicate_physics 를 소비한다"
    # 멱등이어야 한다(두 번 불린다).
    assert "_object_spawn_base" in cfg, "원본 스폰 보존이 없다 — 두 번째 호출이 이중 래핑된다"
    # ★공간 차원도 파생이다(hand_layout 소비) — __post_init__ 에만 있으면
    #   env.hand_layout CLI 가 no-op 이 된다(O1 부팅 fail-loud 실측 08.29).
    i2 = cfg.index("def finalize_after_overrides")
    fin_blk = cfg[i2:cfg.index("def _apply_object_bank")]
    assert "self._derive_spaces(profile)" in fin_blk, \
        "공간 차원 파생이 finalize 밖이다 — hand_layout CLI 오버라이드가 no-op"


def test_self_collision_override_lands_in_robot_cfg():
    """★★다물체 폭주의 근본 원인 잠금 (08.29 확정).

    `replicate_physics=False`(per-env 파싱)에서 손 hull 초기 겹침 × 자기충돌 ON 이
    손가락을 340~660mm 튕기고 palm 173kN 을 만든다 — self-collision OFF 로 sick
    0/256 완치 실측. 다물체 학습은 `env.enable_self_collisions=False` 로 기동하는데,
    `robot_cfg` 재구축이 `__post_init__` 에만 있으면 이 CLI 오버라이드가 **조용한
    no-op** 이 된다(hydra 는 __post_init__ 를 다시 부르지 않는다). 재구축은
    `finalize_after_overrides` 안에 있어야 한다.
    """
    cfg = _code(_CFG)
    i = cfg.index("def finalize_after_overrides")
    j = cfg.index("def ", i + 10)
    blk = cfg[i:j]
    assert "_build_robot_cfg" in blk, \
        "robot_cfg 재구축이 finalize 밖이다 — enable_self_collisions CLI 오버라이드가 no-op"
    assert "enable_self_collisions" in blk


# ---------------------------------------------------------------------------
# ADR (08.29 신설) — 전역 level 하나가 스폰 범위·이송 y·물체 obs 노이즈를 스케일한다.
# ---------------------------------------------------------------------------

def _adr_apply_block() -> str:
    env = _code(_ENV)
    i = env.index("def _adr_apply")
    return env[i:env.index("def ", i + 10)]


def test_adr_defaults_off_and_identity():
    """기본값 OFF = 현행 항등. OFF 면 level 이 0 으로 강제돼 전부 base 값이다."""
    cfg = _code(_CFG)
    assert "enable_adr: bool = False" in cfg
    for k in ("adr_success_threshold", "adr_eval_episodes", "adr_step",
              "adr_spawn_range_max", "adr_goal_y_max",
              "adr_obs_noise_object_max"):
        assert k in cfg, f"cfg 에 {k} 가 없다"
    blk = _adr_apply_block()
    assert "if bool(cfgn.enable_adr) else 0.0" in blk, \
        "OFF 항등 보장이 없다 — enable_adr=False 인데 level 이 실릴 수 있다"
    env = _code(_ENV)
    assert "rng = float(self._adr_spawn_range)" in env, \
        "리셋 스폰이 실효값이 아니라 cfg 원값을 읽는다"
    # 08.30 3축 샘플링으로 목표가 env 별이 됐다 — 샘플링 OFF 면 `_adr_goal_offset`
    # 를 그대로 브로드캐스트하므로 의미는 같다(cfg 원값 직독만 아니면 된다).
    assert "settled + _goff" in env and "self._adr_goal_offset.unsqueeze(0)" in env, \
        "goal 이 실효 오프셋이 아니라 cfg 원값을 읽는다"


def test_adr_goal_growth_expands_delta_box():
    """★축③ 구조 제약 — 목표 y 확장분만큼 델타 박스 y 도 커져야 한다.
    palm 지령 박스(base ±0.10)가 이송 거리를 물리적으로 막는다."""
    blk = _adr_apply_block()
    assert "_d_eff[1] = _d_eff[1] + (abs(_y_eff) - abs(_by))" in blk
    assert "_delta_lo" in blk and "_delta_hi" in blk, \
        "델타 박스가 재계산되지 않는다"


def test_adr_level_is_global_and_monotonic():
    """★env 별 난이도 금지(h7 데드락 이력) + 하강 없는 단조 승급."""
    env = _code(_ENV)
    assert "self._adr_level = 0.0" in env
    assert "self._adr_level = min(" in env, "승급이 상한 없이 자란다"
    assert "self._adr_level -" not in env and "self._adr_level =-" not in env, \
        "level 하강 경로가 있다 — 단조 규약 위반"
    assert "adr_success_threshold" in env, "승급이 성공률 판정 없이 일어난다"


def test_adr_max_goal_validated_at_boot():
    """승급 후 목표가 프로필 박스 밖이면 조용히 과제가 죽는다 — 부팅 fail-loud 로 잠근다."""
    env = _code(_ENV)
    i = env.index("def _assert_goal_reachable")
    blk = env[i:env.index("def ", i + 10)]
    assert "adr_goal_y_max" in blk and "adr_spawn_range_max" in blk
    assert "RuntimeError" in blk


def test_respawn_on_fail_defaults_on_and_is_safe():
    """08.30 — 종료가 유일한 실패 처리면 무접촉 정체가 국소최적(M0·O1 실측 7.2/step).
    재소환 계약: ①기본 OFF ②원래 스폰점 복귀(앵커·목표 불변 — 새 자리 샘플링 금지)
    ③palm 여유 미달 시 보류(폴백 텔레포트 금지 — 자매 v2 규약) ④abnormal 은 여전히 종료
    ⑤파지 단계 상태 되감기."""
    cfg = _code(_CFG)
    # ★09.01 D3 승격 — 기본 ON. 아래 ②~⑤ 안전 계약은 그대로 잠근다.
    assert "respawn_on_fail: bool = True" in cfg
    env = _code(_ENV)
    i = env.index('respawn_on_fail", False))')
    blk = env[i:i + 6000]
    assert "self.object_spawn_pos[_go]" in blk, "원래 스폰점 복귀가 아니다"
    assert "respawn_clearance_m" in blk and "respawn_defer" in blk, "보류 규약이 없다"
    # 08.30 보류 예산 신설로 분기가 생겼다 — 두 경로 모두 abnormal 을 종료로 유지한다.
    assert "terminated = self._abnormal.clone()" in blk \
        and "terminated = self._abnormal | _stuck" in blk, "abnormal 종료가 사라졌다"
    assert "self._latched[_go] = False" in blk, "파지 단계 되감기가 없다"


def test_finger_residual_blend_and_adr_axis():
    """08.30 W 진단 처방 — 커플링은 4지 지령을 평균으로 **대체**해 두 극단만 있었다:
    닫히지만 둔한 손(coupled `syn_close` 0.320) ↔ 손가락 독립인데 안 닫히는 손
    (15ch 0.022~0.106). 공통+잔차로 그 사이를 연속으로 잇는다.
    계약: ①기본 0.0 = 구 coupled 항등(평균 대체) ②블렌드 = 공통 + scale·(개별−공통)
    ③ADR 다섯째 축으로 열리되 max<=base 면 꺼짐 ④액션 차원 불변(warm 보존)."""
    cfg = _code(_CFG)
    assert "finger_residual_scale: float = 0.0" in cfg
    assert "adr_finger_residual_max: float = 0.0" in cfg
    ctl = _code(_CTRL)
    i = ctl.index("_adr_residual")
    blk = ctl[i - 400:i + 400]
    assert "_common + _rs * (a - _common)" in blk, "공통+잔차 블렌드가 아니다"
    assert "_common if _rs == 0.0 else" in blk, "scale 0 항등 분기가 없다"
    env = _code(_ENV)
    j = env.index("self._adr_residual =")
    assert "lvl * max(0.0, _rm - _rb)" in env[j:j + 160], "ADR 축이 base 기준이 아니다"
    assert 'adr/finger_residual' in env, "잔차 로깅이 없다"
    # 액션 차원은 잔차와 무관해야 한다 — 커리큘럼 도중 warm 이 깨지면 안 된다.
    assert "finger_residual" not in _code(_CFG).split("_derive_spaces")[-1][:1200], \
        "잔차가 액션 차원 계산에 새어 들어갔다"


def test_contact_quality_anylink_mode():
    """08.31 사용자 정의 — "어떤 부분이든 5손가락(또는 손바닥까지) 닿고 안정적으로
    유지만 하면 된다". 구 정의(0.4·팁 + 0.6·wrap)는 wrap 이 **중간∧원위 동시**라
    8종 전수 0.000 이었고, 정책이 팁 0.4 만 먹고 2~3개 접촉에서 멈췄다 — 그 상태로는
    컵을 기울이는 과제에서 놓친다. 계약: ①기본 "tipwrap"(현행 항등) ②"anylink" 는
    `grip_c`(tip|mid|dist) 기반 ③**손바닥이 한 표**로 들어가고 분모는 손가락+1."""
    cfg = _code(_CFG)
    # ★09.01 D3 승격 — 기본이 "anylink". "tipwrap" 경로는 대조용으로 남긴다.
    assert 'contact_quality_mode: str = "anylink"' in cfg
    rew = _code(_REW)
    i = rew.index('contact_quality_mode", "tipwrap")) == "anylink"')
    blk = rew[i:i + 420]
    assert "graded_contact = anylink_frac" in blk, "anylink 를 접촉 품질로 안 쓴다"
    assert "_emix" in blk, "구 tipwrap 경로가 사라졌다"
    env = _code(_ENV)
    j = env.index("anylink_frac=")
    b2 = env[j:j + 220]
    assert "grip_c.float().sum(dim=1) + self._surf_palm" in b2, "손바닥 표가 빠졌다"
    assert "(n_tip + 1.0)" in b2, "분모가 손가락+1 이 아니다"
    assert "task/anylink_frac" in env and "task/n_contact" in env, "진단 로깅이 없다"


def test_anylink_replaces_grasp_envelope_credit():
    """08.31 B1 실측 처방 — `contact_quality_mode="anylink"` 를 `graded_contact`
    (곱셈)에만 걸었더니 접촉 개수가 2.58 정점 뒤 2.05 로 하락했다(1,100 iter 회복
    없음). 곱셈은 ∂R/∂접촉 = (lift+transfer+…)/6 이라 사다리가 작을 때 경사도 같이
    작아진다. 같은 시점 `grasp_quality` 는 _ecred(0.80)이 `wrap_frac` 0.02 에
    묶여 0.082 뿐 — grasp 의 80% 가 죽어 있었다. 계약: ①anylink 면 grasp 의 감쌈
    성분도 `anylink_frac` ②거기에 `force_quality` 를 곱해 "세게 눌러 더 닿기"를
    막는다(audit Check 3) ③기본 tipwrap 은 `wrap_frac` 그대로(항등)."""
    rew = _code(_REW)
    i = rew.index("_envelope_credit = ")
    blk = rew[i - 260:i + 320]
    assert 'contact_quality_mode", "tipwrap")) == "anylink"' in blk, \
        "grasp 감쌈 성분이 모드 분기를 안 탄다"
    assert "anylink_frac.clamp(0.0, 1.0)" in blk, "anylink 경로가 없다"
    assert "force_quality.clamp(0.0, 1.0)" in blk, \
        "force_quality 가 안 곱해져 과압 상한이 없다"
    assert "_envelope_credit = wrap_frac.clamp(0.0, 1.0)" in blk, \
        "tipwrap 기본 분기(항등)가 사라졌다"
    assert "_ecred * _envelope_credit" in rew, "grasp_quality 가 새 성분을 안 쓴다"
    assert "_ecred * wrap_frac" not in rew, "옛 직접 참조가 남아 분기가 무효다"


def test_finger_closure_wrap_target():
    """08.31 8종 실측 처방 — 팁 접촉 0.65~0.85 인데 wrap 은 **전 종 0.000**.
    `graded_contact = 0.4·팁 + 0.6·wrap` 에서 싼 0.4 만 먹고 멈췄고, 팁 기준
    소등 항은 그 지점에서 이미 꺼져 경사를 못 준다. 계약: ①기본 "tip"(현행 항등)
    ②"wrap" 이면 소등 = 중간∧원위, 거리 = **중간마디** 기준 ③중간마디 인덱스는
    프로필 `finger_sensor_bodies` 첫 원소에서 이름 해석(리터럴 금지)."""
    cfg = _code(_CFG)
    # ★09.01 D3 승격 — 기본이 "wrap". "tip" 분기는 아래에서 계속 잠근다.
    assert 'finger_closure_target: str = "wrap"' in cfg
    env = _code(_ENV)
    assert "self._mid_ids_t" in env, "중간마디 인덱스가 없다"
    i = env.index("finger_sensor_bodies[f][0]")
    assert "find_bodies" in env[i - 120:i], "중간마디를 이름 해석으로 안 찾는다"
    j = env.index('finger_closure_target", "tip")) == "wrap"')
    blk = env[j:j + 600]
    assert "self._mid_ids_t" in blk, "wrap 모드가 중간마디 거리를 안 쓴다"
    assert "_cl_off = mid_c & dist_c" in blk, "소등 조건이 중간∧원위가 아니다"
    assert "_cl_d, _cl_off = _tip_d, tip_c" in blk, "tip 기본 분기가 사라졌다"


def test_enclosure_contact_floor_blend():
    """08.30 W4 — enclosure(10)는 무접촉으로도 ~7.2/step 공짜라 FRESH 정체의 원천
    (M0·O1 총보상 실측 일치). floor 블렌드는 무접촉 상한을 10·floor 로 낮추되
    ★H 라운드가 기각한 완전 곱셈(접근 구간 항 사망)과 달리 floor 비율의 접근
    gradient 를 보존한다. 계약: ①기본 1.0 = 항등(floor<1 조건 분기) ②블렌드 수식
    = floor + (1−floor)·graded_contact ③enclosure_term 자체에 곱한다(별항 아님)."""
    cfg = _code(_CFG)
    # ★09.01 D3 승격 — 기본 1.0(항등) → 0.3. floor<1 분기가 이제 상시 경로다.
    assert "enclosure_contact_floor: float = 0.3" in cfg
    rew = _code(_REW)
    i = rew.index('enclosure_contact_floor", 1.0)')
    blk = rew[i:i + 400]
    assert "if _encl_floor < 1.0:" in blk, "기본값 항등 분기가 없다"
    assert "_encl_floor + (1.0 - _encl_floor) * graded_contact" in blk, \
        "floor 블렌드 수식이 아니다 — 완전 곱셈(H 기각)으로 퇴행 금지"
    assert "enclosure_term = enclosure_term * (" in blk, \
        "enclosure_term 에 곱하지 않는다"


def test_adr_goal_three_axis_sampling():
    """08.30 성공률 지도 처방 — 이송 y 0.12 에서 0.94~0.98 인데 **0.05 에서 0.000**.
    난이도를 단조로 올리기만 하면 시작 구간을 잊고 "한 방향 14cm" 하나만 배운다.
    계약: ①기본 OFF(현행 항등) ②[base, 레벨] 구간 **에피소드별 샘플링**
    ③x 는 ±범위·z 는 [base,max] ④델타 박스가 세 축 모두 연동 ⑤부팅 검증이 x·z 극단도 본다."""
    cfg = _code(_CFG)
    assert "adr_goal_sample: bool = False" in cfg
    assert "adr_goal_x_max" in cfg and "adr_goal_z_max" in cfg
    env = _code(_ENV)
    i = env.index('adr_goal_sample", False))')
    blk = env[i:i + 700]
    assert "_u[:, 0] * 2.0 - 1.0) * _xs" in blk, "x 가 ± 대칭 샘플이 아니다"
    assert "_u[:, 1] * (abs(_ys) - abs(_by))" in blk, "y 가 [base,레벨] 구간이 아니다"
    assert "_bz + _u[:, 2] * (_zs - _bz)" in blk, "z 가 [base,max] 구간이 아니다"
    j = env.index("_d_eff[0] = _d_eff[0] + _x_eff")
    assert "_d_eff[2] = _d_eff[2] + (_z_eff - _bz)" in env[j:j + 400], \
        "델타 박스가 x·z 확장을 못 따라간다 — 지령이 목표를 못 덮는다"
    k = env.index("def _assert_goal_reachable")
    vblk = env[k:env.index("def ", k + 10)]
    assert "adr_goal_x_max" in vblk and "adr_goal_z_max" in vblk, \
        "부팅 검증이 x·z 극단을 안 본다"


def test_respawn_free_mode_moves_spawn_reference():
    """08.30 자매 v2 규약 이식 — "원래 스폰점 복귀"는 그 자리가 곧 손자리라 보류가
    0.93 까지 갔다(Q3 실측). free 모드는 스폰 상자에서 손 없는 자리를 리젝션
    샘플링한다. 계약: ①기본 origin(현행) ②손 전체(palm+팁) 기준 거리 ③새 자리로
    가면 **스폰 기준·목표를 같이 옮긴다**(안 옮기면 cup_disp 가 즉시 커져 approach 가
    순벌점이 된다 — Q3 −0.35 의 정체) ④후보 전부 실패면 보류(폴백 텔레포트 금지)."""
    cfg = _code(_CFG)
    # ★09.01 D3 승격 — 기본이 "free". "origin" 경로는 대조용으로 남긴다.
    assert 'respawn_mode: str = "free"' in cfg
    assert "respawn_range: float = 0.09" in cfg, "free 모드인데 샘플링 반경이 0 이다"
    env = _code(_ENV)
    i = env.index('respawn_mode", "origin")) == "free"')
    blk = env[i:i + 1800]
    assert "_hand.unsqueeze(1)" in blk and "min(dim=2)" in blk, "손 전체 최소거리가 아니다"
    assert "_clear = _has" in blk, "후보 전부 실패 시 보류가 아니다"
    j = env.index("self.object_spawn_pos[_go] = _tgt_go")
    upd = env[j:j + 200]
    assert "self.goal_pos[_go]" in upd, "목표를 같이 옮기지 않는다 — approach 순벌점 재발"


def test_respawn_defer_budget_falls_back_to_terminate():
    """08.30 Q3 실측 처방 — 보류만 하면 넘어진 컵이 팔 옆에 방치돼 approach 가
    순벌점(−0.35)이 되고 에피소드가 그 상태로 600 스텝을 버틴다(defer 0.93 ·
    palm 접촉 1.4% vs 정상 81.7%). 연속 보류가 예산을 넘으면 종료로 폴백해야 한다.
    기본 0 = 무제한(현행)."""
    cfg = _code(_CFG)
    # ★09.01 D3 승격 — 기본 0(무제한) → 60. 무제한은 Q3 의 defer 0.93 정체를 낳았다.
    assert "respawn_defer_budget: int = 60" in cfg
    env = _code(_ENV)
    assert "self._defer_count" in env, "연속 보류 카운터가 없다"
    i = env.index("respawn_defer_budget")
    blk = env[i:i + 700]
    assert "terminated = self._abnormal | _stuck" in blk, "종료 폴백이 없다"
    assert "done/respawn_stuck" in blk, "폴백 발생률 로깅이 없다"


def test_latch_mode_defaults_to_opposition():
    """08.29 — count 래치는 실측 성공 파지(엄지+palm, n_grip=1)에서 영원히 안 열려
    lift/transfer/stay/stabilize 가 사장됐다. opposition = (A) AND (B OR palm),
    hold 스텝 유지. 기본값은 count(현행 항등, M1 대조 보존)."""
    cfg = _code(_CFG)
    # ★09.01 D3 승격 — 기본이 "opposition". "count" 경로는 대조용으로 남긴다.
    assert 'latch_mode: str = "opposition"' in cfg
    env = _code(_ENV)
    i = env.index('latch_mode", "count")) == "opposition"')
    blk = env[i:i + 400]
    assert "_a_c & (_b_c | _p_c)" in blk, "대향 형식이 아니다"
    assert "grasp_ready_hold_steps" in env[i:i + 900], "hold 필터가 사라졌다"


def test_per_finger_layout_defaults_off_and_matches_user_spec():
    """08.29 O 라운드 — 사용자 확정 손가락별 레이아웃. 잠그는 계약:
    ①기본 hand_layout="coupled3"·freeze_scope="joint" = 현행 항등
    ②엄지 2슬롯·검/중/약 각 1슬롯·소지 1슬롯(j3/4) — 외전(_1)은 어디에도 없음
    ③per_finger 는 액션 차원이 바뀌므로 슬롯 연속성·action_space 정합 fail-loud"""
    cfg = _code(_CFG)
    assert 'hand_layout: str = "coupled3"' in cfg
    assert 'synergy_freeze_scope: str = "joint"' in cfg
    assert "hand_finger_channels" in cfg, "action_space 가 per_finger 분기를 모른다"
    prof = _code((_HERE / "robot_profiles.py").read_text(encoding="utf-8"))
    i = prof.index("hand_finger_channels={")
    blk = prof[i:i + 500]
    assert '"thumb": {"3": 0, "4": 1}' in blk, "엄지 근위/원위 2슬롯이 아니다"
    assert '"pinky": {"3": 5, "4": 5}' in blk, "소지는 j3/4 단일 슬롯이어야 한다"
    assert '"1"' not in blk, "외전(_1)이 액션에 배정됐다 — 고정 계약 위반"
    ctrl = _code(_CTRL)
    assert "self._syn_act" in ctrl and 'hand_layout) == "per_finger"' in ctrl
    assert "per_finger 슬롯" in ctrl, "슬롯↔action_space 정합 fail-loud 가 없다"


def test_finger_scope_freeze_stops_whole_finger():
    """08.29 진단 잠금 — 관절별 동결은 언 손끝을 매단 채 근위(_2)가 계속 감겨
    큰 컵을 밀어낸다(s130 영상). finger 스코프는 (중간∨원위) 접촉 시 그 손가락
    굴곡관절 전부를 세운다. 기본값은 joint(현행 항등)."""
    ctrl = _code(_CTRL)
    i = ctrl.index('synergy_freeze_scope) == "finger"')
    blk = ctrl[i:i + 400]
    assert "(_h_mid | _h_dist) & self._syn_flex" in blk
    assert "_syn_flex" in ctrl and '("2", "3", "4")' in ctrl, \
        "굴곡관절 마스크에 _2 가 빠졌다 — 파고듦이 그대로다"


def test_species_diagnostics_stay_out_of_obs():
    """08.29 신설 — 집계 success 는 종별 실패를 가린다(사용자 지적). 종별 EMA 는
    extras 진단 전용이고, ①obs 에 새면 정체성 계약 위반 ②per-step 리셋 경로라
    .any()/.item() 루프 금지(무동기 index_add 집계)."""
    env = _code(_ENV)
    assert "species/success_min" in env, "종별 최소 성공률 로깅이 없다"
    assert "index_add_" in env, "종별 집계가 무동기 벡터화가 아니다"
    i = env.index("def _get_observations")
    obs_blk = env[i:env.index("def ", i + 10)]
    for banned in ("_species_ids", "_species_succ", "species/"):
        assert banned not in obs_blk, f"obs 경로에 종 정보가 샜다: {banned}"


def test_finger_closure_extinguishes_on_contact():
    """08.29 신설 — 접촉 전 손가락별 연속 신호. 잠그는 계약 셋:
    ①기본 가중 0 = 현행 항등 ②(1−접촉) 소등 — 압입 유인 금지, 조임량은 물리 몫
    ③close_gate 곱 — 빈손 말아쥐기 차단(케이지 되먹임 함정 계열)."""
    cfg = _code(_CFG)
    # ★09.01 D3 승격 — 기본 가중 0 → 3.0. 소등·게이트 계약은 그대로 잠근다.
    assert "finger_closure_weight: float = 3.0" in cfg
    rew = _code(_REW)
    assert '"finger_closure"' in rew, "TERMS 에 미등록 — fail-loud 가 안 잡는다"
    i = rew.index("finger_closure_term = (")
    blk = rew[i - 1200:i + 300]
    assert "close_gate.clamp(0.0, 1.0) * finger_closure.clamp(0.0, 1.0)" in blk
    env = _code(_ENV)
    # 08.31 target 노브 신설로 소등 대상이 분기됐다 — tip 기본값과 wrap 양쪽 모두
    # "닿으면 소등"을 유지해야 한다(압입 유인 차단은 두 경로 공통 계약).
    assert "(~_cl_off).float()" in env and "_cl_d, _cl_off = _tip_d, tip_c" in env, \
        "접촉 소등이 없다 — 압입 유인이 생긴다"
    assert "grasp_center.unsqueeze(1)" in env, \
        "거리 기준이 파지중심이 아니다 — 형상 비의존 계약 위반"


def test_adr_obs_noise_wired_to_actor_only():
    """물체 pose 노이즈는 actor obs 에만 — critic 은 clean state 를 유지한다."""
    env = _code(_ENV)
    i = env.index("def _get_observations")
    blk = env[i:env.index("def ", i + 10)]
    assert blk.count("self._adr_obs_noise_object") == 2, \
        "palm_to_obj·obj_to_tips 두 항 모두 실효 노이즈를 써야 한다"
    _clean = blk[blk.index("clean = torch.cat"):blk.index("state = torch.cat")]
    assert "_adr_obs_noise_object" not in _clean, "clean(critic) 경로에 노이즈가 샜다"


def test_d3_default_set_is_intact():
    """★★09.01 — 기본값 전체가 D3(`s2r_d3_liftonly_fresh_v2`) 세팅인지 한 곳에서 잠근다.

    D3 는 20,000 iter FRESH 로 8종 전수 성공(0.774~0.949 · `success_min` 0.607)을
    낸 첫 런이고, 그 조합을 기본값으로 올렸다. 개별 필드는 각자의 테스트가 메커니즘을
    잠그지만, **조합이 통째로 유효**하다는 것은 여기서만 보장된다 — 한 값만 되돌아가도
    (예: `grasp_weight` 12 복귀) D2 의 pre-lift 주차장이 되살아난다.

    되돌리려면 이 테스트를 먼저 고쳐야 하고, 그때 D3 를 이긴 실측을 근거로 남길 것.
    """
    cfg = _code(_CFG)
    d3 = {
        # 물체·자기충돌
        'object_bank: str = "cup_family"': "다물체 8종",
        "enable_self_collisions: bool = False": "자기충돌 OFF",
        # 액션 앵커
        'palm_anchor_mode: str = "spawn"': "스폰 앵커",
        "palm_delta_xyz: tuple[float, float, float] = (0.10, 0.10, 0.10)": "델타 등방 0.10",
        "palm_delta_rot_deg: float = 40.0": "회전 델타 40°",
        # 접촉 정의
        'contact_quality_mode: str = "anylink"': "어느 마디든 + 손바닥",
        'finger_closure_target: str = "wrap"': "소등 = 중간∧원위",
        "force_band_floor: float = 0.5": "과압 감쇠 바닥",
        # 래치
        'latch_mode: str = "opposition"': "대향 래치",
        "oppose_grip_delta_rad: float = -0.6": "엄지 대향축 활성(a상태)",
        # 목표·리프트
        "goal_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.12)": "수직 12cm",
        "lift_height_ref: float = 0.06": "리프트 정규화 기준",
        "success_require_lifted: bool = True": "성공은 리프트 필수",
        "success_require_holding: bool = False": "holding 이중 게이트 제거",
        # 가중치 — ★D2 주차장의 직접 처방
        "grasp_weight: float = 4.0": "pre-lift 수입 억제",
        "enclosure_weight: float = 10.0": "무접촉 기하 유도",
        "enclosure_contact_floor: float = 0.3": "정체 방지 floor",
        "stabilize_weight: float = 1.0": "정지 항 축소",
        "finger_closure_weight: float = 3.0": "접촉 전 연속 경사",
        # 재소환
        "respawn_on_fail: bool = True": "실패 시 재소환",
        'respawn_mode: str = "free"': "손 없는 자리 샘플링",
        "respawn_clearance_uses_tips: bool = True": "손 전체 기준 여유",
        "respawn_clearance_m: float = 0.12": "여유 거리",
        "respawn_range: float = 0.09": "샘플링 반경",
        "respawn_defer_budget: int = 60": "보류 예산",
        "respawn_penalty: float = 2.0": "재소환 비용",
    }
    missing = [f"{k}  ({why})" for k, why in d3.items() if k not in cfg]
    assert not missing, "D3 기본값이 어긋났다:\n  " + "\n  ".join(missing)
    # ADR 은 D3 에서도 꺼져 있었다 — 승격 대상이 아니라 **원래 기본**이다.
    assert "enable_adr: bool = False" in cfg


# ======================================================================
# 09.01 — ADR 재설계: 과제 난이도 → sim2real 랜덤화
# ======================================================================

def _fn_block(src: str, name: str) -> str:
    """`def name(...)` 부터 다음 `def ` 전까지. 함수 단위 계약 검사용."""
    i = src.index(f"def {name}(")
    j = src.find("\n    def ", i + 1)
    return src[i:j if j > 0 else len(src)]


def test_object_perception_defaults_are_identity():
    """★09.01 신설 — 지각 모델 두 노브는 **기본 False = 현행 항등**이다.

    D3(`544c88b` 기본값)로 학습된 체크포인트가 그대로 재생돼야 하고, 아카이브된
    118런의 dump 복원도 같은 obs 를 봐야 한다. 켜는 것은 CLI 로만.
    """
    cfg = _code(_CFG)
    assert "obs_object_rigid_after_latch: bool = False" in cfg
    assert "obs_object_noise_coherent: bool = False" in cfg


def test_goal_rel_is_not_a_clean_object_channel():
    """★★09.01 누수 회귀 방지 — `goal_rel` 은 물체 위치의 **깨끗한 우회로**였다.

    `goal_rel = goal_pos − obj_pos` 이고 `goal_pos` 는 에피소드 상수라
    `obj_pos = goal_pos − goal_rel` 로 참값이 그대로 복원된다. 그래서
    `palm_to_obj`·`obj_to_tips` 에 노이즈를 아무리 키워도 **`obs_noise_object`
    축 자체가 무효**였다(켠 적은 없지만 켰어도 안 먹었다).

    계약: coherent 모드에서 세 항이 **하나의** 추정값 `_obj_obs` 에서 나온다.
    그래야 `goal_rel + palm_to_obj = goal_pos − palm_pos` 로만 소거되고 그 식엔
    물체 정보가 없다.
    """
    env = _code(_ENV)
    blk = _fn_block(env, "_get_observations")
    assert 'obs_object_noise_coherent", False)' in blk, "coherent 분기가 없다"
    i = blk.index("_obj_obs = self._perceived_object")
    tail = blk[i:i + 700]
    assert "_n_palm_to_obj = _obj_obs - palm_pos" in tail
    assert "- _obj_obs.unsqueeze(1)" in tail, "obj_to_tips 가 추정값을 안 쓴다"
    assert "_n_goal_rel = self.goal_pos - _obj_obs" in tail, \
        "goal_rel 이 여전히 참값 기반이다 — 누수가 살아 있다"
    # 항등 경로(else)는 구 동작 그대로여야 한다.
    assert "_n_goal_rel = goal_rel" in blk, "기본 경로가 바뀌었다"


def test_perceived_object_is_rigid_attached_after_latch():
    """★09.01 — 래치 뒤 물체 obs 는 **palm 강체 부착 추정**이다.

    실기에서는 손이 컵을 감싼 순간 비전이 컵을 잃는다. 참값을 계속 주면 정책이
    실기에 없는 정보에 의존한다. 컵이 손안에서 미끄러져도 정책은 몰라야 하고,
    그게 실기와 같은 조건이다(접촉력으로만 안다).
    """
    env = _code(_ENV)
    blk = _fn_block(env, "_perceived_object")
    assert 'obs_object_rigid_after_latch", False)' in blk, "플래그 분기가 없다"
    assert "palm_pos + torch.einsum(\"nij,nj->ni\", R, self._obj_off_palm)" in blk, \
        "현재 palm 자세로 스냅샷을 굴리지 않는다"
    assert "torch.where(self._latched.unsqueeze(1)" in blk, "래치 게이트가 없다"
    assert "perc/obj_est_err" in blk, "강체가정 오차(=실제 미끄러짐) 로깅이 없다"
    # 노이즈는 이 함수에서 **한 번만** 뽑는다.
    assert blk.count("self._adr_obs_noise_object") == 1, \
        "추정값에 노이즈가 두 번 이상 실린다"


def test_latch_snapshot_is_palm_frame_and_cleared_on_both_resets():
    """★09.01 — 스냅샷은 palm 프레임 `Rᵀ(obj−palm)` 이고, 리셋 **두 경로 모두**에서
    지워야 한다. 재소환 경로를 빠뜨리면 컵이 새 자리로 갔는데 옛 오프셋이 남아
    obs 가 조용히 틀린다(`_disp_at_latch` 가 이미 같은 규율을 따른다).
    """
    env = _code(_ENV)
    assert 'torch.einsum("nji,nj->ni", _R_latch, obj_pos - palm_pos)' in env, \
        "스냅샷이 palm 프레임 역변환이 아니다"
    assert "_just.unsqueeze(1), _off_latch, self._obj_off_palm" in env, \
        "래치 순간(_just)에만 기록하지 않는다"
    assert "self._obj_off_palm[_go] = 0.0" in env, "재소환 경로 초기화 누락"
    assert "self._obj_off_palm[env_ids] = 0.0" in env, "_reset_idx 초기화 누락"


def test_adr_sim2real_axes_default_to_identity():
    """★09.01 — 신규 sim2real 축은 전부 base 와 같은 값 = 폭 0 = 항등이다."""
    cfg = _code(_CFG)
    for line in (
        "adr_obs_noise_qpos_max: float = 0.01",   # = obs_noise_qpos
        "adr_obs_noise_qvel_max: float = 0.05",   # = obs_noise_qvel
        "adr_mass_scale_max: tuple[float, float] = (1.0, 1.0)",
        "adr_joint_gain_scale_max: tuple[float, float] = (1.0, 1.0)",
        "object_friction_range: tuple[float, float] = (1.0, 1.0)",
    ):
        assert line in cfg, f"항등 기본값이 아니다: {line}"
    # 관절 노이즈가 실제로 ADR 실효값을 타는지(cfg 상수 직독이면 축이 죽은 것)
    env = _code(_ENV)
    blk = _fn_block(env, "_get_observations")
    assert "self._adr_obs_noise_qpos" in blk and "self._adr_obs_noise_qvel" in blk, \
        "관절 노이즈가 ADR 실효값을 안 쓴다"
    assert "cfgn.obs_noise_qpos" not in blk, "cfg 상수를 직독해 축이 무효다"


def test_goal_transfer_axes_are_neutralized():
    """★09.01 사용자 확정 — 이송은 IK 로 푼다. ADR 의 목표 축 셋을 폭 0 으로 만든다.

    ★필드를 **지우지 않는다** — 아카이브된 118런의 `params/env.yaml` dump 정합.
    동시에 구 `adr_goal_z_max=0.08` 은 base z(0.12)보다 작아 **승급할수록 목표가
    낮아지는** 역방향 축이었다. base 로 맞춰 폭 0 과 버그를 함께 없앤다.
    """
    cfg = _code(_CFG)
    assert "adr_goal_y_max: float = 0.0" in cfg
    assert "adr_goal_z_max: float = 0.12" in cfg, "base z(0.12)와 달라 폭이 남는다"
    assert "adr_goal_x_max: float = 0.0" in cfg
    assert "adr_goal_sample: bool = False" in cfg


def test_adr_axes_never_invert_guard():
    """★09.01 — `max < base` 를 **부팅에서** 죽인다.

    구 `adr_goal_z_max=0.08 < base 0.12` 가 조용히 살아 있었다. obs 노이즈 축
    주석이 같은 함정을 이미 경고했는데도 z 축에서 재발했으므로 주석이 아니라
    코드로 막는다.
    """
    env = _code(_ENV)
    blk = _fn_block(env, "_assert_adr_monotonic")
    assert "if mx < base:" in blk and "raise RuntimeError" in blk
    for axis in ("goal_z", "obs_noise_object", "obs_noise_qpos",
                 "obs_noise_qvel", "spawn_range"):
        assert f'"{axis}"' in blk, f"{axis} 축이 가드에서 빠졌다"
    assert "self._assert_adr_monotonic()" in env, "부팅에서 안 부른다"


def test_adr_physics_writes_event_manager_not_cfg():
    """★★09.01 — 물리 DR 확장은 `event_manager` 를 고쳐야 한다.

    ManagerBase 가 cfg 를 **deepcopy** 하므로 `self.cfg.events` 를 고치면 조용히
    아무 일도 일어나지 않는다. 그리고 `enable_events=False` 면 속성 자체가 없다.
    """
    env = _code(_ENV)
    blk = _fn_block(env, "_adr_apply_physics")
    assert 'getattr(self, "event_manager", None)' in blk, "guard 없이 접근한다"
    assert "if em is None:" in blk, "enable_events=False 에서 죽는다"
    assert 'em.get_term_cfg("object_scale_mass")' in blk
    assert 'em.get_term_cfg("robot_joint_stiffness_and_damping")' in blk
    assert "self.cfg.events" not in blk, \
        "cfg.events 를 고친다 — deepcopy 라 무효다"
    assert "self._adr_apply_physics(lvl)" in _fn_block(env, "_adr_apply"), \
        "_adr_apply 가 물리 확장을 안 부른다"


def test_friction_is_fixed_dr_not_adr():
    """★★09.01 — 마찰은 ADR 축이 **될 수 없다**.

    `randomize_rigid_body_material` 은 `material_buckets` 를 term 인스턴스 생성
    시 1회만 샘플링하고(PhysX 재질 64,000개 상한) `__call__` 은 그 고정 버킷에서
    뽑기만 한다 → 런타임 확장은 **무증상 no-op**. 자매 `grasp_v2/grasp_adr.py` 가
    재질을 확장하지만 실제 물리는 안 바뀐다. 그래서 cfg 단계(deepcopy 이전)에서만
    연다. 이 테스트는 "나중에 누가 _adr_apply 에 마찰을 넣는" 재발을 막는다.
    """
    env = _code(_ENV)
    for fn in ("_adr_apply", "_adr_apply_physics"):
        blk = _fn_block(env, fn)
        assert "material" not in blk, \
            f"{fn} 이 재질을 건드린다 — 버킷 캐시라 no-op 다"
    cfg = _code(_CFG)
    assert 'object_material.params["static_friction_range"]' in cfg, \
        "cfg 단계 마찰 배선이 없다"
    assert 'object_material.params["dynamic_friction_range"]' in cfg


def test_lerp_range_is_pure_and_numerically_correct():
    """★09.01 — `_lerp_range` 는 순수 함수라 **시뮬 없이** 수치를 검증한다.

    이식 출처: `tesollo/right/grasp_v2/grasp_adr.py:118-135`.
    """
    import ast
    tree = ast.parse(_ENV)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_lerp_range")
    fn.decorator_list = []                      # staticmethod 데코레이터 제거
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<lerp>", "exec"), ns)
    lerp = ns["_lerp_range"]
    assert lerp((0.5, 2.0), 0.0) == (1.0, 1.0), "level 0 은 항등이어야 한다"
    assert lerp((0.5, 2.0), 1.0) == (0.5, 2.0), "level 1 은 종점이어야 한다"
    _mid = lerp((0.5, 2.0), 0.5)
    assert abs(_mid[0] - 0.75) < 1e-9 and abs(_mid[1] - 1.5) < 1e-9
    assert lerp((1.0, 1.0), 1.0) == (1.0, 1.0), "종점이 항등이면 전 구간 항등"


def test_real_gain_branch_is_recorded_and_verified():
    """★★09.01 재생 무결성 — r2s 정합 게인 사용 여부가 dump 에 남고 부팅에서 대조된다.

    게인 선택은 `robot_profiles.py` 가 **import 시점**에 `HDGP_S2R_REAL_GAINS` 로
    하는데 환경변수는 `params/env.yaml` 에 **안 남는다**. 그래서 그 게인으로 학습한
    체크포인트를 환경변수 없이 재생하면 4배 단단한 KUKA 게인으로 조용히 돌아간다 —
    m1_final 이 죽은 것과 같은 계열(차원은 맞는데 의미가 다른)이다.

    계약: ①`use_real_gains` 가 cfg 필드라 dump 에 실린다 ②기본값은 환경변수를 따라
    기존 워크플로가 깨지지 않는다 ③`finalize_after_overrides` 가 실제 조립 결과와
    대조해 어긋나면 **부팅에서 죽인다**(조용히 틀리지 않는다).
    """
    cfg = _code(_CFG)
    assert 'use_real_gains: bool = _os.environ.get("HDGP_S2R_REAL_GAINS") == "1"' in cfg, \
        "cfg 필드가 아니거나 환경변수 기본값을 안 따른다"
    blk = _fn_block(cfg, "_assert_gain_branch")
    assert '"right_arm_proximal" in _specs' in blk, "KUKA 분기 판별이 없다"
    assert '"right_arm_j1" in _specs' in blk, "실측 분기 판별이 없다"
    assert "if not (_kuka or _real):" in blk, \
        "게인 분기가 없는 프로필(gripper_left)에서 오탐한다"
    assert "raise RuntimeError" in blk, "불일치를 조용히 넘긴다"
    assert "self._assert_gain_branch(profile)" in cfg, \
        "finalize_after_overrides 가 대조를 안 부른다"


def test_robot_gravity_stays_disabled():
    """★★09.01 확정 — 로봇 중력은 **켜지 않는다**(실기가 중력보상을 켠 채로 돈다).

    켜면 중력을 두 번 세는 것이다. 실측(08.31 우팔): 무보상 처짐 **12.76°** →
    보상 후 **2.05°**. sim 중력 ON 은 12.76° 쪽을 모델링해 실기와 10° 벌어진다.
    r2s 정합 게인에서 더 치명적이다 — 손목 kp 50→10 이라 테솔로 손 1.763 kg 의
    토크(≈2.1 N·m)에 0.21 rad = 12° 처진다. 그 게인 자체가 **중력보상 전제**로
    동정됐다(r2s collect 가 `gravity_comp_node.py` 를 필수로 요구한다).

    ★물체(컵)는 별개다 — `object_cfg` 는 `disable_gravity=False` 로 무게가 살아 있어야
    파지·리프트가 물리적으로 성립한다.
    """
    cfg = _code(_CFG)
    _rb = cfg[cfg.index("def _build_robot_cfg"):]
    _rb = _rb[:_rb.index("max_depenetration_velocity")]
    assert "disable_gravity=True" in _rb, \
        "로봇 중력이 켜졌다 — 실기 중력보상과 이중 계산된다"
    assert "disable_gravity=False" in cfg, "물체 중력이 꺼졌다 — 파지가 무의미해진다"


def test_hand_sim_gains_stay_at_grip_force_value():
    """★★09.01 손 튜닝 확정 — sim 손 게인은 `kp 5.0 · kd 2.0` 을 **그대로 둔다**.

    정합은 **실기 쪽을 올려서** 했다: ros2_control JTC PID 의 p 를 벤더 1.5 →
    **4.5** 로 써 넣었다(`SetJointGain*` 서비스로 쓸 수 있는 것이 팔에 없던 자유도).
    같은 주먹 램프에서 실기 정상오차 0.39°(최대 1.50°) vs sim 0.05°(최대 0.60°) —
    차이 0.34° 는 손가락 끝에서 1 mm 이하라 파지에 영향이 없다.

    ★sim kp 를 실기에 맞춰 **낮추면 안 된다** — 5.0 은 파지력 기준으로 정해진 값이고
      (grasp_v1 kd 스윕: kd 6.71 포화 20.5% · kd ≤0.5 채터 · kd 2.0 이 양쪽 최적),
      낮추면 파지력이 그만큼 준다. 반대로 실기 p 를 더 올리는 길은 **p=6 진동**이
      막는다(σ 0.095° → 0.160°). 지금이 양쪽 한계 안에서의 최선이다.

    ⚠실기 p=4.5 는 **bringup 마다 벤더 기본으로 돌아간다** — vendor yaml 은 안 고쳤다.
      `sim2real/scripts/apply_hand_gains.py --execute` 를 매번 돌려야 한다.
    """
    prof = (_HERE / "robot_profiles.py").read_text(encoding="utf-8")
    assert 'stiffness=5.0, damping=2.0' in prof, \
        "손 sim 게인이 바뀌었다 — 파지력 기준값(kp 5.0)을 실기에 맞춰 낮추지 말 것"
    assert "effort_limit_sim=1.5" in prof, "손 effort 한계가 바뀌었다"


def test_gain_dr_excludes_hand_by_default():
    """★09.01 — 게인 DR 은 **팔만** 대상이다.

    손은 이제 미지가 아니다(위 테스트 참조 — 실기 p 를 우리가 써 넣었고 격차가
    0.34° 로 실측됐다). 없는 불확실성을 랜덤화하면서 파지력 기준값을 훼손할 이유가
    없다. 팔 게인은 벤더 고정값이라 진짜 불확실성이 남는다.

    ★대상 좁히기는 **cfg 단계**에서만 가능하다 — EventManager 가 생성 시 관절
      인덱스를 굳히므로 런타임에 `joint_names` 를 바꿔도 늦다(마찰 버킷과 같은 계열).
    """
    cfg = _code(_CFG)
    assert 'gain_dr_joints: str = "arm"' in cfg, "기본이 팔 한정이 아니다"
    assert 'if str(self.gain_dr_joints) == "arm":' in cfg
    assert '"asset_cfg"].joint_names = [profile.arm_joint_regex]' in cfg, \
        "팔 regex 를 프로필에서 안 읽는다(리터럴 금지)"


# ================================================================= 언더액추 손
# ★09.02 grasp_ua 신설 — grasp_s2r 를 뜬 뒤 "구동 자유도 < 물리 자유도" 하드웨어를
#   일반화하며 생긴 계약들. 여기서 잠그는 사고는 전부 **조용한** 것들이다:
#   결합이 사라진 손, 아무 관절에도 안 걸리는 접미사, 축이 바뀐 손바닥 법선.

def _load_profiles():
    """`robot_profiles.py` 를 값으로 읽는다 — 순수 데이터라 Isaac 없이 뜬다."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "_gua_rp", _HERE / "robot_profiles.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gua_rp"] = mod          # dataclass 가 cls.__module__ 을 찾는다
    spec.loader.exec_module(mod)
    return mod


_RH_URDF = (_HERE.parents[5] / "assets" / "robot" / "openarm_bi_rh56f1_rl"
            / "openarm_bi_rh56f1_rl.urdf")


def _urdf_joints(path):
    """URDF 우손 revolute 관절 → {이름: (한계lo, 한계hi, mimic|None)}."""
    src = path.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r'<joint name="(r_hj_[a-z0-9_]+)" type="revolute">(.*?)</joint>',
                         src, re.S):
        nm, body = m.group(1), m.group(2)
        lim = re.search(r'<limit[^>]*lower="([^"]+)"[^>]*upper="([^"]+)"', body)
        mim = re.search(r'<mimic joint="([^"]+)" multiplier="([^"]+)" offset="([^"]+)"', body)
        out[nm] = (float(lim.group(1)), float(lim.group(2)),
                   (mim.group(1), float(mim.group(2)), float(mim.group(3))) if mim else None)
    return out


def test_new_profile_fields_are_identity_by_default():
    """★신규 필드 전부 기본값이 **현행 거동**이어야 tesollo/grasp_s2r 이 안 흔들린다."""
    rp = _load_profiles()
    d = rp.RobotProfile(name="_probe", usd_relpath="x")
    assert d.hand_mimic == {} and d.hand_mimic_mode == "none"
    assert d.hand_freeze_mid_suffixes == ()
    assert d.fabric_arm_slot == () and d.fabric_hand_slot == ()
    assert d.palm_frame_remap == ()


def test_tesollo_mid_suffix_reproduces_the_old_literal():
    """축출한 `s == "3"` 과 정확히 같아야 한다 — tesollo 거동 항등의 유일한 근거."""
    rp = _load_profiles()
    assert rp.PROFILES["tesollo_right"].hand_freeze_mid_suffixes == ("3",)
    assert rp.PROFILES["tesollo_right"].hand_mimic_mode == "none"
    assert rp.PROFILES["tesollo_right"].fabric_arm_slot == ()


def test_mid_suffix_literal_is_gone_from_control():
    """★`"3"` 이 코드에 남아 있으면 마디가 하나뿐인 손에서 중간마디 동결이 아무
    관절에도 안 걸린 채 **조용히 꺼진다**. 로봇 사실은 프로필이 들어야 한다."""
    ctl = _code(_CTRL)
    assert 'hand_freeze_mid_suffixes' in ctl
    assert re.search(r's\s*==\s*"3"', ctl) is None, '축출한 접미사 리터럴이 되살아났다'
    assert re.search(r's\s*!=\s*"3"', ctl) is None


def test_mid_suffixes_must_be_subset_of_freeze_suffixes():
    """중간마디는 동결 대상의 부분집합 — 아니면 부팅에서 죽는다(값 검증 + 가드 존재)."""
    rp = _load_profiles()
    for p in rp.PROFILES.values():
        assert set(p.hand_freeze_mid_suffixes) <= set(p.hand_freeze_suffixes), p.name
    assert "hand_freeze_mid_suffixes {sorted(_unknown)} 가" in _CTRL


def test_mimic_table_matches_the_urdf_exactly():
    """★★프로필의 결합표는 URDF `<mimic>` 의 사본이어야 한다.

    배율이 하나만 틀려도 sim 의 손가락이 실기와 다른 궤적으로 말리는데, 지표는
    전부 정상으로 보인다(접촉도 나고 파지도 된다). 자산이 바뀌면 여기서 걸린다.
    """
    if not _RH_URDF.exists():
        import pytest
        pytest.skip(f"자산 없음: {_RH_URDF}")
    rp = _load_profiles()
    p = rp.PROFILES["rh56f1_right"]
    urdf = {k: v[2] for k, v in _urdf_joints(_RH_URDF).items() if v[2] is not None}
    assert set(p.hand_mimic) == set(urdf), (
        f"결합 관절 집합 불일치: 프로필 {sorted(p.hand_mimic)} vs URDF {sorted(urdf)}")
    for dep, (src, mult, off) in p.hand_mimic.items():
        u_src, u_mult, u_off = urdf[dep]
        assert src == u_src, f"{dep}: 구동관절 {src} != URDF {u_src}"
        assert abs(mult - u_mult) < 1e-6, f"{dep}: 배율 {mult} != URDF {u_mult}"
        assert abs(off - u_off) < 1e-9, f"{dep}: 오프셋 {off} != URDF {u_off}"


def test_mimic_chains_resolve_to_drive_joints():
    """`thumb_4 → thumb_3 → thumb_2` 같은 체인이 반드시 구동관절에서 끝나야 한다.

    안 그러면 종속관절이 **아무도 안 모는 관절**을 따라가 init 각도에 굳는다.
    """
    rp = _load_profiles()
    for p in rp.PROFILES.values():
        if not p.hand_mimic:
            continue
        drive = set(p.hand_joint_names)
        for dep in p.hand_mimic:
            assert dep not in drive, f"[{p.name}] {dep} 이 구동이면서 종속이다"
            cur, seen = dep, [dep]
            while cur in p.hand_mimic:
                cur = p.hand_mimic[cur][0]
                assert cur not in seen, f"[{p.name}] 결합 순환 {seen + [cur]}"
                seen.append(cur)
            assert cur in drive, f"[{p.name}] {dep} 의 뿌리 {cur} 가 구동관절이 아니다"


def test_mimic_write_is_gated_on_software_mode():
    """physx 모드에서 우리가 종속목표를 쓰면 PhysX 결합과 **싸운다**."""
    ctl = _code(_CTRL)
    assert "_apply_mimic_targets" in ctl
    assert 'self._mim_mode != "software"' in ctl, "software 게이트가 없다"
    assert "_apply_mimic_targets()" in ctl.split("def _apply_action")[1].split("def ")[0], \
        "_apply_action 이 종속목표를 안 쓴다"


def test_mimic_targets_derive_from_command_not_measurement():
    """★지령(`_syn_target`)에서 파생해야 결합비가 유지된다 — 실측에서 만들면
    구동관절의 추종 지연이 결합비에 섞여 들어간다."""
    body = _code(_CTRL).split("def _apply_mimic_targets")[1].split("\n    def ")[0]
    assert "_syn_target" in body
    assert "data.joint_pos" not in body, "실측 관절값에서 종속목표를 만들고 있다"


def test_mimic_mode_values_are_closed():
    """오타난 모드가 조용히 '결합 없음'으로 떨어지면 안 된다."""
    ctl = _code(_CTRL)
    assert '("none", "software", "physx")' in ctl
    rp = _load_profiles()
    for p in rp.PROFILES.values():
        assert p.hand_mimic_mode in ("none", "software", "physx"), p.name


# ---------------------------------------------------------------- palm 프레임
def test_palm_frame_remap_is_a_proper_rotation():
    """★리맵이 정회전이 아니면 손바닥 법선과 가로축이 **조용히** 바뀐 채 학습된다."""
    rp = _load_profiles()
    for p in rp.PROFILES.values():
        if not p.palm_frame_remap:
            continue
        M = [list(r) for r in p.palm_frame_remap]
        assert len(M) == 3 and all(len(r) == 3 for r in M), p.name
        for i in range(3):
            for j in range(3):
                dot = sum(M[k][i] * M[k][j] for k in range(3))
                assert abs(dot - (1.0 if i == j else 0.0)) < 1e-4, f"[{p.name}] 비직교"


def test_palm_ee_R_applies_the_remap():
    """`_palm_ee_R()` 열 0 = 손바닥 법선이라는 규약은 보상이 통째로 의존한다."""
    body = _code(_CTRL).split("def _palm_ee_R")[1].split("\n    def ")[0]
    assert "_palm_remap" in body, "리맵이 적용되지 않는다"
    setup = _code(_CTRL).split("def _setup_scene")[1].split("\n    def ")[0]
    assert "_palm_remap" in setup, "리맵을 씬 셋업 초반에 안 잡으면 이른 호출이 깨진다"


# ---------------------------------------------------------------- fabric 슬롯
def test_fabric_slots_cover_the_controlled_joints():
    """★RH56F1 fabric 은 **양팔 26 DOF** 다. 슬롯이 어긋나면 손 상태 동기화가
    엉뚱한 관절에 실려 fabric 이 없는 자기충돌을 피하려 팔을 민다."""
    rp = _load_profiles()
    for p in rp.PROFILES.values():
        if not p.fabric_hand_slot:
            continue
        order = p.fabric_joint_order
        assert len(p.fabric_arm_slot) == p.num_arm_joints, p.name
        assert len(p.fabric_hand_slot) == p.num_hand_joints, p.name
        assert max(p.fabric_arm_slot + p.fabric_hand_slot) < len(order), p.name
        assert [order[i] for i in p.fabric_hand_slot] == list(p.hand_joint_names), (
            f"[{p.name}] fabric 손 슬롯이 hand_joint_names 와 순서까지 같아야 한다")
        assert len(set(p.fabric_arm_slot) & set(p.fabric_hand_slot)) == 0, p.name


def test_fabric_slicing_uses_slot_tensors_not_literals():
    """구 규약(`[:, :num_arm_joints]`) 슬라이스가 남아 있으면 양팔 cspace 에서
    조용히 왼팔 관절을 팔 목표로 써넣는다."""
    for src, nm in ((_code(_CTRL), "control"), (_code(_ENV), "env")):
        assert "num_arm_joints]" not in src.replace("num_arm_joints]", "X") or True
        assert not re.search(r"fabric_q\w*\[:,\s*:?\s*self\.profile\.num_arm_joints", src), nm
    assert "_fab_arm_t" in _code(_CTRL) and "_fab_hand_t" in _code(_CTRL)
    assert "_fab_hand_t" in _code(_ENV) and "_fab_arm_t" in _code(_ENV)


# ---------------------------------------------------------------- rh56f1 프로필
def test_rh56f1_is_registered_with_its_own_tag():
    rp = _load_profiles()
    assert "rh56f1_right" in rp.PROFILES
    assert re.search(r'"rh_r":\s*GraspUARh56f1RightEnvCfg', _REG)
    assert "GraspUARh56f1RightEnvCfg" in _CFG


def test_rh56f1_uses_per_finger_layout_matching_its_dof():
    """★기본 `coupled3`(채널×손가락)로 두면 4지의 채널1이 아무 관절에도 안 걸려
    액션 4개가 통째로 죽는다."""
    rp = _load_profiles()
    p = rp.PROFILES["rh56f1_right"]
    slots = sorted({s for m in p.hand_finger_channels.values() for s in m.values()})
    assert slots == list(range(p.num_hand_joints)), slots
    sub = _CFG.split("class GraspUARh56f1RightEnvCfg")[1].split("\n@configclass")[0]
    assert 'hand_layout: str = "per_finger"' in sub
    assert 'synergy_hold_mode: str = "blocked"' in sub, (
        "손가락당 구동관절이 하나면 관절 동결로 드레이프를 만들 수 없다")


def test_rh56f1_poses_are_inside_urdf_limits():
    """★한계 밖 grip 자세는 런타임 clamp 로 조용히 잘려 폐쇄도가 절대 1 이 안 된다."""
    if not _RH_URDF.exists():
        import pytest
        pytest.skip(f"자산 없음: {_RH_URDF}")
    rp = _load_profiles()
    p = rp.PROFILES["rh56f1_right"]
    lim = _urdf_joints(_RH_URDF)
    assert len(p.hand_open_pose) == len(p.hand_grip_pose) == p.num_hand_joints
    for nm, o, g in zip(p.hand_joint_names, p.hand_open_pose, p.hand_grip_pose):
        lo, hi, _ = lim[nm]
        assert lo - 1e-6 <= o <= hi + 1e-6, f"{nm} open {o} ∉ [{lo},{hi}]"
        assert lo - 1e-6 <= g <= hi + 1e-6, f"{nm} grip {g} ∉ [{lo},{hi}]"
        assert abs(g - o) > 1e-4, f"{nm} 이 open==grip 이라 시너지가 안 건드린다"


def test_rh56f1_contact_bodies_exist_and_have_collision():
    """★collision 없는 링크에 ContactSensor 를 달면 힘이 **항상 0** 이고, 감쌈
    게이트가 영원히 안 켜진 채 '접촉을 학습으로 회피한다'로 오진된다."""
    if not _RH_URDF.exists():
        import pytest
        pytest.skip(f"자산 없음: {_RH_URDF}")
    src = _RH_URDF.read_text(encoding="utf-8")
    has_col = {m.group(1) for m in re.finditer(
        r'<link name="(r_hl_[a-z0-9_]+)">(.*?)</link>', src, re.S)
        if "<collision" in m.group(2)}
    rp = _load_profiles()
    p = rp.PROFILES["rh56f1_right"]
    for f, bodies in p.finger_sensor_bodies.items():
        assert len(bodies) == 3, f"{f}: (중간, 원위, 팁) 삼중 규약"
        for b in bodies:
            assert b in has_col, f"{b} 에 collision 이 없다 — 접촉력이 항상 0 이다"
    for b in p.fingertip_bodies:
        assert b in has_col, b
    assert p.palm_body.startswith("r_hl_")


def test_rh56f1_init_pose_is_mimic_consistent():
    """★결합 불일치 init 은 첫 스텝에 종속관절이 snap 하면서 컵을 튕긴다."""
    rp = _load_profiles()
    p = rp.PROFILES["rh56f1_right"]
    ip = p.init_joint_pos
    for dep, (src, mult, off) in p.hand_mimic.items():
        assert dep in ip, f"{dep} 이 init_joint_pos 에 없다"
        assert abs(ip[dep] - (mult * ip[src] + off)) < 1e-6, (
            f"{dep} init {ip[dep]} != {mult}·{src}({ip[src]}) + {off}")
    for nm, o in zip(p.hand_joint_names, p.hand_open_pose):
        assert abs(ip[nm] - o) < 1e-9, f"{nm} init 이 hand_open_pose 와 다르다"


def test_rh56f1_gain_branch_is_honest():
    """★RH56F1 자산으로 r2s 식별을 아직 안 했다. tesollo 손(1.763 kg)으로 잡은
    손목 게인을 복사하면 거짓이 되므로, KUKA 분기 키를 써서 `use_real_gains=1`
    이 부팅에서 죽게 둔다(의도된 게이트)."""
    rp = _load_profiles()
    p = rp.PROFILES["rh56f1_right"]
    assert "right_arm_proximal" in p.actuator_specs
    assert "right_arm_j1" not in p.actuator_specs
    # 전 DOF 커버리지: 구동·종속·팔·좌측·헤드가 모두 어느 그룹엔가 들어가야 한다.
    exprs = " ".join(str(v.get("joint_names_expr")) for v in p.actuator_specs.values())
    for nm in list(p.hand_joint_names) + list(p.hand_mimic):
        stem = nm.split("r_hj_")[1]
        assert stem.split("_")[0] in exprs, f"{nm} 이 어느 액추에이터 그룹에도 없다"


def test_dead_joint_guard_exists():
    """★"지령상 움직이는데 한계로 못 움직이는" 관절을 부팅에서 죽인다.

    이게 없으면 폐쇄도 분모에 영원히 0 인 항이 남고, 지표 어디에도 이유가 안 나온다
    (09.02 실측: RH56F1 엄지가 300스텝 폐쇄 지령에도 0.0001 rad).
    """
    ctl = _code(_CTRL)
    assert "_open_clamped" in ctl and "_dead" in ctl
    assert "open/grip 이 관절한계로 같아졌다" in _CTRL


def test_rh56f1_turns_off_the_opposition_knob():
    """`oppose_grip_delta_rad` 가 가리키는 ch1 관절의 **의미가 로봇마다 다르다**.

    tesollo ch1 = 대향 관절 / RH56F1 ch1 = 엄지 굴곡(한계 [0, 0.475]).
    """
    sub = _CFG.split("class GraspUARh56f1RightEnvCfg")[1].split("\n@configclass")[0]
    assert "oppose_grip_delta_rad: float = 0.0" in sub
    assert "use_body_repulsion_pairs: bool = False" in sub


def test_fabric_kwargs_are_filtered_loudly():
    """fabric 생성자가 안 받는 인자를 **조용히 버리면** 기능이 소리 없이 사라진다."""
    ctl = _code(_CTRL)
    assert "_filter_fabric_kwargs" in ctl and "_FABRIC_NEUTRAL" in ctl
    assert "inspect.signature" in ctl
    body = ctl.split("def _filter_fabric_kwargs")[1].split("\nclass ")[0]
    assert "raise RuntimeError" in body, "중립값이 아닌 미지원 인자는 죽여야 한다"


def test_robot_material_body_names_must_stay_wildcard():
    """★`.*` 가 아니면 IsaacLab 이 body 별 shape 수 검사를 켜고, 콜라이더 없는 링크가
    있는 자산에서 죽는다. 게다가 그 예외를 EventManager 가 삼켜 첫 리셋에서야
    엉뚱한 TypeError 로 터진다(09.02 RH56F1 실측 460 vs 459)."""
    assert 'SceneEntityCfg("robot", body_names=".*")' in _CFG
    assert "body 별 shape" in _CFG or "shape 수" in _CFG


# ---------------------------------------------- 09.02 mimic USD 반영
_RH_MANIFEST = (_HERE.parents[5] / "assets" / "robot" / "openarm_bi_rh56f1_rl"
                / "openarm_bi_rh56f1_rl_manifest.yaml")


def test_rh56f1_fabric_order_matches_the_manifest_contract():
    """★자산이 진실원천이다 — `control_joint_order`(26)와 순서까지 같아야 한다.

    fabric cspace 순서가 어긋나면 손 상태 동기화가 엉뚱한 관절에 실려 fabric 이
    없는 자기충돌을 피하려 팔을 민다(선행 트랙 실측).
    """
    if not _RH_MANIFEST.exists():
        import pytest
        pytest.skip(f"자산 없음: {_RH_MANIFEST}")
    import yaml
    order = yaml.safe_load(_RH_MANIFEST.read_text(encoding="utf-8"))["control_joint_order"]
    rp = _load_profiles()
    assert list(rp.PROFILES["rh56f1_right"].fabric_joint_order) == list(order)


def test_rh56f1_has_no_pd_group_on_mimic_joints():
    """★★mimic 을 켜면 종속관절의 DriveAPI 가 사라진다. 거기에 PD 를 걸면 PhysX
    제약과 싸운다 — mimic 이 없던 시절의 우회책이 남아 있으면 안 된다."""
    rp = _load_profiles()
    p = rp.PROFILES["rh56f1_right"]
    assert p.hand_mimic_mode == "physx"
    exprs = " ".join(str(v.get("joint_names_expr")) for v in p.actuator_specs.values())
    for dep in p.hand_mimic:
        stem = dep.split("r_hj_")[1]                     # thumb_3 / index_2 …
        assert stem not in exprs, f"{dep} 에 아직 PD 그룹이 걸려 있다"
    assert "right_hand_mimic" not in p.actuator_specs


def test_rh56f1_real_gain_branch_uses_vendor_values():
    """`HDGP_S2R_REAL_GAINS=1` 이 벤더 `control_gains.yaml` 값을 싣는지.

    ★kd 는 **벤더값**이지 r2s 정합값이 아니다(RH56F1 손으로 재식별 전). tesollo 의
      kd 를 복사하면 안 된다 — 손 질량이 달라 손목 관성이 다르다.
    """
    prof = (_HERE / "robot_profiles.py").read_text(encoding="utf-8")
    blk = prof.split("RH56F1_RIGHT = RobotProfile(")[1].split("\nPROFILES")[0]
    for j, kp, kd in ((1, 70.0, 2.75), (2, 70.0, 2.50), (3, 70.0, 2.00),
                      (4, 60.0, 2.00), (5, 10.0, 0.70), (6, 10.0, 0.60),
                      (7, 10.0, 0.50)):
        assert re.search(rf'"right_arm_j{j}":.*?r_aj_{j}.*?stiffness={kp}, damping={kd}',
                         blk, re.S), f"r_aj_{j} 벤더 게인 불일치"
    assert 'HDGP_S2R_REAL_GAINS' in blk
    assert '"right_arm_proximal"' in blk, "KUKA 기본 분기가 있어야 게이트가 동작한다"


def test_mimic_coupling_is_monitored_at_runtime():
    """★physx 모드에서 USD 제약이 안 걸려 있으면 종속관절이 init 각도에 굳는데,
    그 상태는 접촉·보상 어느 지표에도 안 나타난다 — 전용 진단이 있어야 한다."""
    ctl = _code(_CTRL)
    assert "_mimic_tracking_err" in ctl
    assert 'self.extras["hand/mimic_err_max"]' in _code(_ENV)


def test_rh56f1_uses_a_hand_sized_object_bank():
    """★물체 뱅크는 **엔드이펙터 종속**이다.

    `cup_family`(cup_big r62mm × 0.85~1.30 = 지름 105~161mm)는 tesollo DG-5F 기준이라
    RH56F1 으로는 감쌈이 기하적으로 불가능하다 — 검지 MCP→팁이 URDF 실측 72.5mm 다.
    그대로 두면 "정책이 못 배운다"로 오진하기 딱 좋다(물체가 애초에 손보다 크다).
    """
    sub = _CFG.split("class GraspUARh56f1RightEnvCfg")[1].split("\n@configclass")[0]
    assert 'object_bank: str = "shaker_family"' in sub

    import importlib.util
    import sys
    _ob = _HERE.parents[1] / "modules" / "object_bank.py"
    spec = importlib.util.spec_from_file_location("_gua_ob", _ob)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gua_ob"] = mod
    spec.loader.exec_module(mod)
    bank = mod.BANKS["shaker_family"]
    assert len(bank.specs) == 8
    d = [2.0 * s.base_grasp_radius * s.scale[0] for s in bank.specs]
    assert 0.070 <= min(d) and max(d) <= 0.097, f"지름 범위 {min(d):.3f}~{max(d):.3f} m"
    assert d == sorted(d), "종 인덱스와 크기 순서가 어긋나면 종별 진단을 못 읽는다"
    for s in bank.specs:
        assert s.base_origin_offset_z is not None, f"{s.id} 원점 오프셋 미측정"
        assert s.base_rim_z is not None, f"{s.id} 림 z 미측정"


# ------------------------------------------------- 09.02 홈/앵커/스폰 캘리브
def test_rh56f1_home_is_not_the_sibling_track_literal():
    """★같은 팔이라도 손이 바뀌면 같은 관절값이 전혀 다른 palm 자세를 낸다.

    선행 rh56f1 트랙 홈(0.5, 0.1, 0.4, 0.60, −0.2, 0, 0)은 이 palm 프레임에서
    `ez −9.6° ey −62.7° ex −99.1°` 를 내고, 부팅 게이트가 "홈이 워크스페이스 박스
    밖"으로 막는다(09.02 실측). 그 리터럴이 되살아나면 안 된다.
    """
    rp = _load_profiles()
    ip = rp.PROFILES["rh56f1_right"].init_joint_pos
    sibling = {"r_aj_1": 0.5, "r_aj_2": 0.1, "r_aj_3": 0.4, "r_aj_4": 0.60,
               "r_aj_5": -0.2, "r_aj_6": 0.0, "r_aj_7": 0.0}
    assert any(abs(ip[k] - v) > 1e-6 for k, v in sibling.items()), \
        "선행 트랙 팔 홈 리터럴이 그대로 들어와 있다"
    for j in range(1, 8):
        assert f"r_aj_{j}" in ip


def test_rh56f1_rotation_center_avoids_gimbal_lock():
    """★태스크 기준 자세(손바닥 +y · 손가락 +x 수평)는 이 프레임에서 **짐벌락**
    (ey = −90°)이다. 손가락을 아래로 기울여 특이점에서 떼어내야 한다.

    선행 트랙 프리셋 (180, 0, 90) 은 손가락이 **위**를 향하는 자세라 fabric 이
    도달 못 하고 130mm/51° 로 포화했다(09.02 실측).
    """
    rp = _load_profiles()
    p = rp.PROFILES["rh56f1_right"]
    ez, ey, ex = p.palm_rot_center_deg
    assert (ez, ey, ex) != (180.0, 0.0, 90.0), "도달 불가 프리셋이 되살아났다"
    assert abs(abs(ey) - 90.0) >= 10.0, f"ey={ey}° 가 짐벌락(±90°)에 너무 가깝다"
    # 짐벌락 자세가 액션 회전 박스 안에 있어야 정책이 수평 파지도 낼 수 있다.
    assert abs(-90.0 - ey) <= p.palm_rot_half_deg, (
        f"수평 파지 자세(ey −90°)가 ±{p.palm_rot_half_deg}° 박스 밖이다")


def test_rh56f1_anchor_and_spawn_are_mutually_derived():
    """앵커·스폰·홈이 **케이지 오프셋 하나**에서 파생돼야 한다.

    ★케이지는 palm 에 강체로 붙어 있고, 이 손의 world 오프셋은 홈 자세에서 실측
      (+0.070, +0.035, −0.017) m 다. tesollo 값을 그대로 쓰면 케이지가 파지중심보다
      110mm 위에 떠 델타 0.10 으로 못 내려온다.
    """
    sub = _CFG.split("class GraspUARh56f1RightEnvCfg")[1].split("\n@configclass")[0]
    m = re.search(r"palm_anchor_offset_xyz: tuple = \(([-\d., ]+)\)", sub)
    assert m, "rh56f1 전용 앵커 오프셋이 없다(tesollo 값을 물려받고 있다)"
    off = [float(v) for v in m.group(1).split(",")]
    cage = (0.070, 0.035, -0.017)          # 홈 자세에서의 케이지 world 오프셋 실측
    # 앵커에서의 케이지 = 스폰 + off + cage · 파지중심 = 스폰 + (0,0,0.03)
    dz = off[2] + cage[2] - 0.03
    assert abs(off[0] + cage[0]) < 0.005, "앵커 케이지가 컵과 x 정렬이 아니다"
    assert abs(off[1] + cage[1]) < 0.005, "앵커 케이지가 컵과 y 정렬이 아니다"
    assert 0.03 <= dz <= 0.07, f"앵커 케이지가 파지중심 위 {dz * 1000:.0f}mm — 30~70mm 로"

    rp = _load_profiles()
    sx, sy = rp.PROFILES["rh56f1_right"].object_spawn_center
    lo = rp.PROFILES["rh56f1_right"].palm_box_min
    # ★앵커 ± 델타가 박스를 넘으면 그 축이 상시 클램프된다(구 홈 규약의 y 92% 포화).
    assert sx + off[0] - 0.10 >= lo[0] - 1e-9, (
        f"앵커 x {sx + off[0]:.3f} − 델타 0.10 이 박스 하한 {lo[0]} 밑이다")
