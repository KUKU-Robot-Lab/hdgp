"""grasp_s2r 계약 테스트 — Isaac Sim 없이 돈다(소스 텍스트·AST 검사).

cfg 는 isaaclab→pxr 를 끌어와 Isaac 앱 없이 import 가 안 되므로, 값이 아니라
**소스의 계약**을 잠근다. 각 테스트에 왜 이 계약이 생겼는지(어떤 사고를 막는지)를
적어 둔다 — 나중에 고칠 때 근거 없이 지우지 않도록.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_s2r/tests -q
"""

from __future__ import annotations

import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
_SHARED_PROFILES = _HERE.parent.parent / "modules" / "robot_profiles.py"
_ENV = (_HERE / "grasp_s2r_env.py").read_text(encoding="utf-8")
_CTL = (_HERE / "grasp_s2r_control.py").read_text(encoding="utf-8")
_CTRL = (_HERE / "grasp_s2r_control.py").read_text(encoding="utf-8")
_CFG = (_HERE / "grasp_s2r_env_cfg.py").read_text(encoding="utf-8")
_REW = (_HERE / "grasp_s2r_rewards.py").read_text(encoding="utf-8")
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
    assert '_ENTRY = "openarm.agnostic.tasks.grasp_s2r.grasp_s2r_env:GraspS2REnv"' in _REG
    for suffix in ('""', '"-play"', '"-lstm"', '"-play-lstm"'):
        assert suffix in _REG, f"태스크 id 접미사 {suffix} 미등록"
    assert 'f"open-{_tag}_grasp_s2r{_suffix}"' in _REG
    # 로그 분리 규약: 두 번째 슬롯이 r/l/b 여야 한다.
    assert re.search(r'"sens_r":\s*GraspS2RTesolloRightEnvCfg', _REG)


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


def test_obs_is_proprioceptive_only_no_tactile():
    """촉각은 actor·critic **어디에도** 없다(사용자 확정 09.10). 목표·자세는 있어야 한다.

    파지 폐쇄 신호는 `joint_err`(엔코더 기반 추종오차)가 단독으로 진다 — 인벨롭이 잘 될수록
    팁 F/T 가 0 을 읽는 역상관 신호이기 때문이다(`_joint_pos_err` docstring).

    ★이 단언을 **금지 방향으로** 쓰는 이유: 그냥 지우면 다음 라운드에 조용히 되살아난다
      (이 저장소의 재발 유형). 무엇이 바뀌면 이 계약이 거짓이 되는지 —
      **사용자가 촉각 사용 금지를 철회하면** 이다. 그때는 이 테스트를 되뒤집는다.
    ★제거는 **관측 계층 한정**이다. 보상(`graded_contact`·`force_quality`)·래치·
      손 제어(`synergy_contact_freeze`)는 여전히 접촉을 쓰고, 그건 이 테스트 밖이다.
    """
    m = re.search(r"_noisy = torch\.cat\(\[([\s\S]*?)\], dim=1\)", _ENV)
    assert m, "policy obs 결합식 부재"
    blk = m.group(1)
    for need in ("joint_err", "goal_rel", "palm_ax"):
        assert need in blk, f"policy obs 에 {need} 가 없다"
    assert "tip_force" not in blk, "policy obs 에 촉각이 되살아났다"

    m = re.search(r"clean = torch\.cat\(\[([\s\S]*?)\], dim=1\)", _ENV)
    assert m, "critic clean 결합식 부재"
    assert "tip_force" not in m.group(1), "critic clean obs 에 촉각이 되살아났다"

    m = re.search(r"state = torch\.cat\(\[([\s\S]*?)\], dim=1\)", _ENV)
    assert m, "critic state 결합식 부재"
    st = m.group(1)
    for banned in ("tip_force", "_contact_forces_split", "_dist >", "_mid >"):
        assert banned not in st, f"critic state 에 촉각이 되살아났다: {banned}"


def test_obs_dim_formula_matches_layout():
    """cfg 의 obs 차원 산술식이 실제 결합 성분 수와 맞는지."""
    m = re.search(r"self\.observation_space = \(([\s\S]*?)\)", _CFG)
    assert m, "observation_space 식 부재"
    expr = m.group(1)
    for need in ("2 * n_arm", "2 * n_hand", "3 * num_tips", "self.action_space", "+ 3"):
        assert need in expr, f"obs 식에 {need} 가 없다"
    # 물체 뱅크·스케일에서 파생되면 안 된다.
    assert "bank" not in expr and "scale" not in expr
    # ★★09.10 **개수까지** 잠근다. 구 단언(`"3 * num_tips" in expr`)은 항이 3개에서
    #   2개로 줄어도 참이라, 촉각 제거를 검출하지 못했다 — 존재만 보는 단언은 크기 변화에
    #   눈이 멀다. tips 파생 항은 `tips_rel_palm` 과 `obj_to_tips` **둘뿐**이다.
    assert expr.count("3 * num_tips") == 2, (
        f"tips 파생 항은 2개(tips_rel_palm·obj_to_tips)여야 한다: {expr.count('3 * num_tips')}개")

    # critic 식도 검사한다 — 지금까지 **어떤 테스트도 state_space 식을 안 봤다**.
    m = re.search(r"self\.state_space = \(([\s\S]*?)\)", _CFG)
    assert m, "state_space 식 부재"
    st = m.group(1)
    assert "self.observation_space" in st, "critic 은 policy obs 위에 얹혀야 한다"
    assert "num_fingers" not in st, "critic 식에 마디 접촉(4·num_fingers)이 되살아났다"
    assert "num_tips" in st, "critic 의 fingertip_signed_dist(nt)가 사라졌다"


# ---------------------------------------------------------------- 보상
def test_reward_terms_include_transfer_and_stay():
    """이송 2항이 계약에 있어야 한다. 항 계약은 이 트랙 **로컬**이다."""
    code = _code(_REW)
    assert "GRASP_S2R_REWARD_TERMS" in code
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
    """적분은 `_step_fabric` 한 곳 — `_apply_action` 에서 돌리면 fabric 시간이 2배.

    ★09.10: CUDA Graph 배선이 `_setup_fabric_graph` 에서 워밍업·캡처로 적분을 부른다.
      그건 **부팅 1회**라 정책 스텝당 적분 횟수와 무관하다. 그래서 전역 개수가 아니라
      '어느 함수 안에 있는가'를 잠근다 — 개수 프록시는 올바른 구현까지 막는다.
    """
    ctrl = _code(_CTRL)
    _allowed = ("_step_fabric", "_setup_fabric_graph")
    total = ctrl.count("self.integrator.step(")
    inside = sum(_fn_block(ctrl, fn).count("self.integrator.step(") for fn in _allowed)
    assert total == inside, f"허용되지 않은 함수에서 적분한다 (전역 {total} vs 허용 {inside})"
    assert _fn_block(ctrl, "_step_fabric").count("self.integrator.step(") == 1, \
        "정책 스텝당 적분 호출 지점은 하나여야 한다"
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
    # ★09.10 — 단언을 **리터럴에서 조건으로** 바꿨다. 구판은 `palm_still` 이
    #   approach 식에 **직접** 적혀 있을 것을 요구했는데, E1 개편이 그것을
    #   블렌드(`_still_eff = floor + (1−floor)·palm_still`)로 한 줄 위로 올리자
    #   요건은 그대로인데 단언만 죽었다(관측 0179 의 7번째 재현).
    #   계약은 "approach 가 **palm 속도에서 유래한 인자**로 스케일된다" 이다.
    _sf = re.search(r"approach = pre_lift_gate \* (\w+)", rew)
    assert _sf, "approach 가 pre_lift_gate 로 시작하지 않는다"
    _fac = _sf.group(1)
    assert _fac == "palm_still" or re.search(
        rf"{_fac}\s*=[^\n]*palm_still", rew), (
        f"approach 의 스케일 인자 {_fac!r} 가 palm_still 에서 유래하지 않는다 "
        "— 밀착 후 정지 요건이 사라졌다")
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
    assert "simulation_setting/env_v1/usd/env_v1.usda" in cfg
    # ★09.05: env_v1 은 루트에 kinematic RigidBodyAPI 가 저작돼 있다 — 사본 대신 저작 검사.
    i = cfg.index("physics:kinematicEnabled = 1")
    blk = cfg[max(0, i - 800):i + 300]
    assert "_os.path.isfile" in blk and "raise RuntimeError" in blk, "존재·kinematic 검사가 없다"
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
    # ★09.10 실체는 `agnostic/modules/robot_profiles.py` 로 옮겼다(s2r 밑은 재수출 shim).
    prof = _code(_SHARED_PROFILES.read_text(encoding="utf-8"))
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

        "success_require_lifted: bool = True": "성공은 리프트 필수",
        "success_require_holding: bool = False": "holding 이중 게이트 제거",
        # 가중치 — ★D2 주차장의 직접 처방
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
    # ★★09.10 E1 개편 — 아래 항목은 D3 값을 **의도적으로 승계**했다. 지우지 않고
    #   여기로 옮겨 잠근다: 무엇이 언제 왜 바뀌었는지가 테스트에 남아야 다음 라운드에
    #   조용히 되살아나지 않는다(관측 0179 의 부수 규칙).
    #   근거 문서: `repo/reports/grasp_s2r_보상개편_계획_0910.md`
    #   ⚠D3 를 이긴 **실측은 아직 없다** — 측정 기반 설계 판정이다. E1 이 D3 에 지면
    #     이 사전을 되돌리고 위 d3 사전으로 두 항목을 복귀시킨다.
    e1 = {
        "lift_height_ref: float = 0.12": "구 0.06 은 목표(0.095)보다 작아 평지를 만들었다",
        "lift_weight: float = 10.0": "구 30.0 = 위치무관 상수수입 20.05/step",
        "transfer_weight: float = 30.0": "lift 에서 회수한 몫, 마지막 3.5cm 소유",
        "transfer_sharpness: float = 10.0": "그 구간 Δexp 0.295 → 0.392, 증분 4.43 → 11.76",
        "success_weight: float = 8.0": "구 20.0 = 목표 진입 절벽 ~28/step",
        "stay_weight: float = 12.0": "실현/명목 20.2% 로 전 항 최저",
        "approach_weight: float = 6.0": "접근·파지 실현 0.18% → 명목 재분배",
        "approach_still_floor: float = 0.5": "palm_still 을 곱셈에서 블렌드로",
        "grasp_weight: float = 8.0": "실현 1.9% (D3 는 4.0)",
    }
    missing = [f"{k}  ({why})" for k, why in d3.items() if k not in cfg]
    missing += [f"[E1] {k}  ({why})" for k, why in e1.items() if k not in cfg]
    assert not missing, "D3/E1 기본값이 어긋났다:\n  " + "\n  ".join(missing)
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


def test_object_perception_knobs_are_pinned():
    """지각 모델 두 노브의 **현재 계약**. 09.10 에 하나가 뒤집혔고, 사유를 여기 남긴다.

    ★구 계약(09.01)은 둘 다 False 였고 사유는 "D3 체크포인트 재생·아카이브 dump 복원이
      같은 obs 를 봐야 한다" 였다. 09.10 에 **같은 커밋에서 촉각 15칸을 뺐으므로**
      obs 폭이 155→140 으로 바뀌었다 — 그 전제가 스스로 무효가 됐다(체크포인트 재생은
      어차피 불가능하다). 전제가 사라진 가드를 그대로 두면 올바른 수정을 막는다.

    ①`obs_object_noise_coherent = True` — 촉각을 뺀 상태에서 물체 위치가 sim 참값이면
      정책이 실기에 없는 정보에 의존한다. 구 경로는 `obj = goal_pos − goal_rel` 로
      참값 복원이 가능해 `obs_noise_object` 축이 사실상 무효였다.
    ②`obs_object_rigid_after_latch = False` — **유지**. 촉각을 뺀 판에서 이걸 켜면
      래치 후 물체 pose 가 palm 강체 가정으로 대체돼, 컵이 손안에서 미끄러져도 정책이
      알 길이 전혀 없어진다(알 유일한 수단이 촉각인데 방금 뺐다). 둘의 동시 적용은
      정책에서 파지 상태 관측을 **전부** 없애는 것과 같다 — 그래서 여기서 잠근다.

    무엇이 이 계약을 거짓으로 만드는가: ②는 촉각을 되살리거나 기하 기반 파지 상태 관측을
    신설한 뒤에만 열 수 있다. 그때 이 테스트를 갱신한다.
    """
    cfg = _code(_CFG)
    assert "obs_object_rigid_after_latch: bool = False" in cfg, (
        "촉각을 뺀 판에서 강체 가정까지 켜면 파지 상태 관측이 전멸한다")
    assert "obs_object_noise_coherent: bool = True" in cfg, (
        "촉각 제거와 물체 참값 누수 차단은 같은 판에 간다")


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

    def _scalar(name: str) -> float:
        m = re.search(rf"{name}:\s*float\s*=\s*([\d.]+)", cfg)
        assert m, f"{name} 를 못 찾았다"
        return float(m.group(1))

    def _pair(name: str) -> tuple[float, float]:
        m = re.search(
            rf"{name}:\s*tuple\[float, float\]\s*=\s*\(([\d.]+),\s*([\d.]+)\)", cfg)
        assert m, f"{name} 를 못 찾았다"
        return float(m.group(1)), float(m.group(2))

    def _event_pair(key: str) -> tuple[float, float]:
        m = re.search(rf'"{key}":\s*\(([\d.]+),\s*([\d.]+)\)', cfg)
        assert m, f"EventCfg {key} 를 못 찾았다"
        return float(m.group(1)), float(m.group(2))

    # ★★09.10 — 단언을 **리터럴에서 조건으로** 바꿨다. 계약의 내용은 위 docstring 이
    #   이미 정확히 말하고 있다: "**base 와 같은 값 = 폭 0**". 그런데 단언은 그 개념이
    #   아니라 당시의 base 리터럴 `(1.0, 1.0)` 을 적어 놓았고, E1 이 질량 base 를
    #   (0.5, 2.5) 로 열자 **폭은 여전히 0 인데** 단언만 죽었다.
    #   ⚠질량·게인축의 base 는 cfg 필드가 아니라 `EventCfg` 의 분포 파라미터다.
    for name, terminal, base in (
        ("obs_noise_qpos", _scalar("adr_obs_noise_qpos_max"), _scalar("obs_noise_qpos")),
        ("obs_noise_qvel", _scalar("adr_obs_noise_qvel_max"), _scalar("obs_noise_qvel")),
        ("mass_scale", _pair("adr_mass_scale_max"),
         _event_pair("mass_distribution_params")),
        ("joint_gain_scale", _pair("adr_joint_gain_scale_max"),
         _event_pair("stiffness_distribution_params")),
    ):
        assert terminal == base, (
            f"ADR 축 {name} 의 폭이 0 이 아니다: terminal={terminal} vs base={base} — "
            "축을 열려면 그 근거를 남기고 이 테스트를 먼저 고칠 것")
    # 마찰은 ADR 축이 아니라 cfg 고정 범위다(런타임 확장이 무증상 no-op).
    assert _pair("object_friction_range") == (1.0, 1.0), "마찰 기본이 항등이 아니다"
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


def test_arm_gains_are_vendor_values_and_checked_at_boot():
    """★★2026-09-06 확정 — 팔 PD 게인은 **벤더 `control_gains.yaml` 하나뿐**이다.

    구 계약(`use_real_gains` + `HDGP_S2R_REAL_GAINS`)은 "KUKA 냐 r2s 냐"를 고르는
    스위치였는데, 선택지가 벤더값 하나로 줄어든 뒤에는 판별식이 **항상 참**이 되어
    기본 False 인 환경변수와 어긋났다. 그래서 태스크가 부팅에서 무조건 죽었다.
    이제 스위치는 없고, 대신 조립된 게인이 정말 벤더값인지 대조한다.

    다른 게인으로 학습한 정책은 **다른 로봇에서 배운 것**이라 배포할 수 없다
    (09.03 우팔 d3 = KUKA kp 300 학습 → 배포 불가).
    """
    from openarm.agnostic.modules import vendor_gains as VG

    cfg = _code(_CFG)
    assert "use_real_gains" not in cfg.replace("구 `use_real_gains`", ""), \
        "죽은 게인 스위치가 되살아났다"
    assert "HDGP_S2R_REAL_GAINS" not in cfg.replace("`HDGP_S2R_REAL_GAINS`", ""), \
        "죽은 환경변수 분기가 되살아났다"
    blk = _fn_block(cfg, "_assert_vendor_gains")
    assert "_vg.load()" in blk, "벤더 yaml 을 안 읽는다"
    assert "raise RuntimeError" in blk, "불일치를 조용히 넘긴다"
    assert "self._assert_vendor_gains(profile)" in cfg, \
        "finalize_after_overrides 가 대조를 안 부른다"

    # 프로필이 실제로 벤더값으로 조립되는가(해석 결과 대조)
    from openarm.agnostic.tasks.grasp_s2r import robot_profiles as RP
    table = VG.load()
    for prof in RP.PROFILES.values():
        for spec in prof.actuator_specs.values():
            exprs = spec.get("joint_names_expr", ())
            if len(exprs) != 1:
                continue
            for side in VG.SIDES:
                for idx in VG.ARM_JOINTS:
                    if exprs[0] == VG.joint_name(side, idx):
                        assert (spec["stiffness"], spec["damping"]) == table[idx], \
                            f"{prof.name}/{exprs[0]} 게인이 벤더값이 아니다"


def test_gravity_compensation_matches_the_real_controller():
    """★★2026-09-06 — sim 도 실기 pd 노드와 **같은 자리에 τ_ff** 를 넣는다.

    실기는 `gravity.mode: model_tau_ff` 로 팔에 중력 피드포워드를 얹는다. 학습이 그걸
    안 하면 다른 로봇에서 배운 것이 된다.

    실측 근거(홈 자세를 PD 로만 유지, 600 스텝):
        보상 0.0 → 손 최저 z 0.3685 → **0.1505** (상판 0.205 아래 54.5mm), 처짐 최대 13.81°
        보상 1.0 → 손 최저 z **0.3620 유지**, 처짐 최대 **0.34°**
    보상이 없으면 정책이 무엇을 하기 전에 손이 테이블에 박힌 채로 에피소드가 시작된다.

    계약 셋: ①팔 관절에만 건다(실기 DG-5F 드라이버에 중력보상이 없다) ②중력이 꺼졌는데
    보상이 켜져 있으면 부팅에서 죽인다(중력을 두 번 지운다) ③`enable_gravity` 가
    spawn 속성으로 실제 파생된다.
    """
    cfg, env, ctl = _code(_CFG), _code(_ENV), _code(_CTL)
    assert "enable_gravity: bool = True" in cfg, "중력 스위치가 cfg 필드가 아니다"
    assert "gravity_compensation: float = 1.0" in cfg, "중력보상 배율이 cfg 필드가 아니다"
    assert "disable_gravity=not enable_gravity" in cfg, "spawn 속성이 cfg 에서 파생되지 않는다"

    blk = _fn_block(ctl, "_apply_gravity_compensation")
    assert "get_gravity_compensation_forces()" in blk, "중력 토크를 안 읽는다"
    assert "set_joint_effort_target" in blk, "τ_ff 를 안 보낸다"
    assert "joint_ids=self._grav_ids" in blk, "전 관절에 걸면 손·머리까지 보상해 실기와 갈린다"
    assert "self._apply_gravity_compensation()" in _fn_block(ctl, "_apply_action"), \
        "_apply_action 이 보상을 안 부른다"

    assert '"[rl]_aj_[1-7]"' in env, "중력보상 대상이 양팔 7관절이 아니다"
    assert "중력을 두 번 지운다" in env, "중력 OFF + 보상 ON 조합을 막는 가드가 없다"


def test_robot_gravity_is_enabled():
    """★★2026-09-06 사용자 확정 — 로봇 자체 중력을 **켠다**. 보상은 어디에도 없다.

    실기도 `gravity.mode: off` 로 가므로 양쪽 PD 가 같은 중력을 그대로 맞는다.
    근거는 09.02 유령질량 수정 후의 처짐 실측이다(벤더 게인, 단위 deg):
        관절        j1     j2     j3     j4     j7
        sim        2.11   2.78   0.67  -2.53   4.77
        실기       1.92   2.66   0.59  -2.33   4.24
    전 관절 0.5° 이내 = 같은 물리다. 구 09.01 규약("켜지 않는다")은 ①실기가 보상을
    켠 채 돈다 ②게인이 보상 전제로 동정됐다 ③중력 기여분이 2.9~4.5° 다 라는 세 전제가
    모두 무너져 폐기했다(③은 유령질량 자산에서 잰 과대값이었다).

    ★물체(컵)도 `disable_gravity=False` 라야 파지·리프트가 물리적으로 성립한다.
    """
    cfg = _code(_CFG)
    _rb = cfg[cfg.index("def _build_robot_cfg"):]
    # ★시그니처를 건너뛰고 **본문**만 본다. 09.09 에 `max_depenetration_velocity` 가
    #   인자로도 생기면서 "첫 등장까지 자르기"가 시그니처 안에서 끊겨 오탐이 났다.
    _rb = _rb[_rb.index("-> ArticulationCfg:"):]
    _rb = _rb[:_rb.index("articulation_props")]
    assert "disable_gravity=not enable_gravity" in _rb, \
        "로봇 중력이 cfg 에서 파생되지 않는다 — 하드코딩하면 오버라이드가 조용히 죽는다"
    assert "enable_gravity: bool = True" in cfg, "로봇 중력 기본값이 ON 이 아니다"
    # 물체는 별개 — 무게가 살아 있어야 파지·리프트가 성립한다.
    _ob = cfg[cfg.index("object_cfg: RigidObjectCfg"):]
    assert "disable_gravity=False" in _ob[:_ob.index("object_spawn_base")
                                          if "object_spawn_base" in _ob else 4000], \
        "물체 중력이 꺼졌다 — 파지가 무의미해진다"


def test_hand_sim_gains_are_the_vendor_driver_pid():
    """★★2026-09-06 사용자 확정 — 손도 **벤더 드라이버 PID(p 1.5 · d 0)** 만 쓴다.

    이 테스트는 정반대의 계약을 잠그던 자리다. 구 09.01 규약은 sim 을 `kp 5.0 · kd 2.0`
    에 두고 **실기를 4.5 로 올려** 맞췄다(bringup 마다 `apply_hand_gains.py` 재적용).
    09.06 결정으로 그 방향이 뒤집혔다 — 양쪽 다 벤더값으로 내린다.

    ⚠따라오는 실측 결과: 벤더 p=1.5 는 4 s 주먹 램프에서 지령의 **82 %** 까지만 간다
      (4.5 는 98~101 %). sim 도 같은 1.5 라 sim↔실기 정합은 오히려 좋아지지만,
      **파지력·도달률은 재확인 대상**이다. effort 한계 1.5 N·m 는 게인이 아니라 그대로다.
    """
    from openarm.agnostic.modules import vendor_gains as VG

    prof = _SHARED_PROFILES.read_text(encoding="utf-8")   # ★09.10 modules/ 로 이동
    assert "_vg.hand_actuator(" in prof, "손 게인이 벤더 모듈을 거치지 않는다"
    assert "effort_limit_sim=1.5" in prof, "손 effort 한계가 바뀌었다"
    assert VG.hand_gains() == (1.5, 0.0)
    # ★구 손 게인(5.0/2.0) 금지는 **DG-5F 프로필 구간에만** 건다. 이 파일은 로봇
    #   레지스트리라 벤더 PD 가 없는 손(RH56F1: RS-485 위치 서보 — `VG.NO_VENDOR_PD`)도
    #   같이 산다. 파일 전체 텍스트 금지는 그 예외까지 잡아 규칙과 무관한 실패를 낸다.
    _dg5f = prof[prof.index("TESOLLO_RIGHT = RobotProfile("):prof.index("RH56F1_RIGHT = RobotProfile(")]
    assert "stiffness=5.0, damping=2.0" not in _dg5f, "구 DG-5F 손 게인(5.0/2.0)이 되살아났다"
    assert "rh56f1_hand" in VG.NO_VENDOR_PD, "벤더 PD 없는 손의 예외 사유가 사라졌다"


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


def test_table_top_matches_the_env_v1_asset_geometry():
    """★★테이블 상면 0.205 를 **자산 메시와 수치로** 대조한다(09.06 신설).

    지금까지 계약 테스트는 파생 **수식**만 봤다(`table_surface_z + origin + pad`).
    그래서 상수 자체가 자산과 어긋나도 전부 통과했다. 실제로 구 `env` 자산은 0.200,
    `env_v1` 은 0.205 이고 그 5 mm 가 배포 계약(`fabric.table_z`)까지 따라간다.

    datum: sim z=0 = 마운트 플레이트 상면. 실기 줄자·Fusion CAD 둘 다 0.205
    (`sim2real/docs/TABLE_DATUM_2026-09-05.md`).
    """
    m = re.search(r"table_surface_z:\s*float\s*=\s*([0-9.]+)", _CFG)
    assert m, "table_surface_z 선언을 못 찾았다"
    declared = float(m.group(1))
    assert declared == 0.205, f"table_surface_z 가 {declared} — 09.05 확정값은 0.205"

    usda = _HERE.parents[5] / "assets/simulation_setting/env_v1/usd/env_v1.usda"
    if not usda.is_file():          # 학습 서버에는 USD 만 배포된다
        return
    text = usda.read_text(encoding="utf-8", errors="replace")
    block = re.search(r'def Mesh "Collision".*?point3f\[\] points = \[(.*?)\]', text, re.S)
    assert block, "env_v1.usda 에 Collision 메시 points 가 없다"
    zs = [float(z) for z in re.findall(
        r"\(\s*[-\d.e+]+,\s*[-\d.e+]+,\s*([-\d.e+]+)\s*\)", block.group(1))]
    assert zs, "Collision points 를 못 읽었다"
    top = max(zs)
    assert abs(top - declared) < 1e-6, (
        f"자산 상면 {top:.6f} vs cfg {declared} — {abs(top - declared) * 1000:.1f} mm 어긋난다")
    # 마운트 플레이트 상면이 곧 원점이어야 한다(로봇 spawn z=0 의 전제).
    assert abs(min(z for z in zs if z >= -1e-9)) < 1e-6, \
        "z=0 평면이 없다 — 마운트 플레이트 상면이 원점이 아니다"


def test_profile_poses_are_inside_asset_joint_limits():
    """프로필의 기본/개방/파지 자세가 **자산 URDF 한계 안**이어야 한다.

    ★09.09 사고: 자산 한계를 좁히면서(`thumb_3` → [0, 1.05]) 프로필 `init_joint_pos` 의
      −0.5 를 같이 안 고쳤다. 런타임 clamp 는 있었지만 IsaacLab 은 **그 앞에서** 기본자세를
      한계와 대조해 거부한다 — 서버 3런이 동시에 죽고 나서야 발견했다. 여기서 막는다.

    `init_joint_pos` 는 하드 실패(위 사고)이고, `hand_open_pose` 도 리셋 경로가 그대로
    쓰므로 같은 기준으로 본다. `hand_grip_pose` 는 **지령**이라 런타임 clamp 대상이므로
    한계 초과가 설계상 허용된다(주석에 명시돼 있다) — 검사하지 않는다.
    """
    import re
    from pathlib import Path

    from openarm.agnostic.tasks.grasp_s2r.robot_profiles import PROFILES

    profile = PROFILES["tesollo_right"]
    # ★env_cfg 를 import 하지 않는다 — 이 스위트는 Isaac 없이 도는 소스 검사다.
    #   _HERE = .../source/openarm/openarm/agnostic/tasks/grasp_s2r → parents[5] = hdgp 루트
    urdf = (_HERE.parents[5] / "assets" / profile.usd_relpath).with_suffix(".urdf")
    if not urdf.is_file():
        import pytest
        pytest.skip(f"자산 URDF 없음: {urdf}")
    text = urdf.read_text()
    limits = {}
    for m in re.finditer(r'<joint name="([^"]+)" type="revolute"(.*?)</joint>', text, re.S):
        block = re.search(r"<limit([^/>]*)", m.group(2)).group(1)
        limits[m.group(1)] = (float(re.search(r'lower="([^"]+)"', block).group(1)),
                              float(re.search(r'upper="([^"]+)"', block).group(1)))

    bad = []
    for name, value in dict(profile.init_joint_pos).items():
        if name not in limits:
            continue
        lo, hi = limits[name]
        if not (lo - 1e-6 <= value <= hi + 1e-6):
            bad.append(f"init_joint_pos {name}={value:+.4f} 한계 [{lo:+.4f},{hi:+.4f}]")
    if profile.hand_open_pose:
        for name, value in zip(profile.hand_joint_names, profile.hand_open_pose):
            if name not in limits:
                continue
            lo, hi = limits[name]
            if not (lo - 1e-6 <= value <= hi + 1e-6):
                bad.append(f"hand_open_pose {name}={value:+.4f} 한계 [{lo:+.4f},{hi:+.4f}]")
    assert not bad, "자산 한계 밖 자세 — 부팅에서 죽는다:\n  " + "\n  ".join(bad)



# ══════════════════════════════════════════════════════════════════════════════
# 09.09 손 PD/armature override — **재조립 뒤에도 살아남는가** + **로그가 진실인가**
#   isaaclab 은 SimulationApp 기동 전 import 가 안 돼 cfg 를 실제로 조립할 수 없다.
#   그래서 이 파일의 규약대로 소스 텍스트로 배선을 잠근다. 실제 값은 부팅 로그
#   `[grasp_s2r][s2r] … 손게인=kp…` 가 **조립본**을 찍으므로 거기서 확인한다.
# ══════════════════════════════════════════════════════════════════════════════
_CFG_SRC = (_HERE / "grasp_s2r_env_cfg.py").read_text(encoding="utf-8")
_ENV_SRC = (_HERE / "grasp_s2r_env.py").read_text(encoding="utf-8")


def test_hand_override_is_a_cfg_field_not_a_post_hoc_patch():
    """override 는 cfg 필드여야 한다 — robot_cfg 에 나중에 얹으면 재조립이 지운다.

    09.06·09.09 에 같은 함정으로 네 번 당했다(gravity·tol_eval·object USD·solver knob):
    A/B 두 팔이 **비트 동일**한 씬을 돌아 스윕이 통째로 no-op 이 됐다.
    """
    for field in ("hand_stiffness_override", "hand_damping_override", "hand_armature"):
        assert re.search(rf"^    {field}: float = ", _CFG_SRC, re.M), \
            f"{field} 가 cfg 필드가 아니다"


def test_hand_override_is_consumed_by_the_rebuild():
    """`finalize_after_overrides` 의 `_build_robot_cfg` 호출이 세 필드를 넘겨야 한다."""
    call = re.search(r"self\.robot_cfg = _build_robot_cfg\((.*?)\)\n", _CFG_SRC, re.S)
    assert call, "_build_robot_cfg 재조립 호출을 못 찾았다"
    blk = call.group(1)
    for field in ("hand_stiffness_override", "hand_damping_override", "hand_armature"):
        assert field in blk, f"재조립이 {field} 를 안 읽는다 — override 가 조용히 지워진다"


def test_hand_override_targets_hand_actuators_only():
    """팔·머리는 손대지 않는다 — 다른 게인으로 학습한 팔 정책은 배포 불가다(09.03 d3)."""
    fn = re.search(r"def _hand_gain_override\(.*?\n\n\n", _CFG_SRC, re.S)
    assert fn, "_hand_gain_override 헬퍼가 없다"
    body = fn.group(0)
    assert 'name.endswith("hand")' in body, "손 actuator 만 고르는 조건이 없다"
    assert "return dict(spec)" in body, "원본을 변형한다 — 새 dict 를 돌려줘야 한다(불변)"


def test_hand_override_default_is_vendor():
    """기본값은 벤더값 유지여야 한다 — 예외를 안 켠 런은 현행과 비트 동일해야 한다."""
    assert re.search(r"^    hand_stiffness_override: float = 0\.0", _CFG_SRC, re.M)
    assert re.search(r"^    hand_damping_override: float = -1\.0", _CFG_SRC, re.M)
    assert re.search(r"^    hand_armature: float = 0\.0", _CFG_SRC, re.M)
    fn = re.search(r"def _hand_gain_override\(.*?\n\n\n", _CFG_SRC, re.S).group(0)
    assert "if stiffness > 0.0:" in fn and "if damping >= 0.0:" in fn \
        and "if armature > 0.0:" in fn, "기본값에서 원본을 건드리지 않는다는 보장이 없다"


def test_boot_log_reports_assembled_gains_not_the_profile():
    """부팅 로그가 **조립본**을 찍어야 한다.

    프로필(`actuator_specs`)은 override **이전**의 입력이다. 그걸 찍으면 kp 5.0 으로 돌면서
    로그에는 벤더 1.5 가 남아, 실험이 사후에 검증 불가가 된다(09.09 실제로 그랬다).
    """
    blk = re.search(r"\n        _a\w* = .{0,400}?robot_cfg.{0,2000}?손게인=", _ENV_SRC, re.S)
    assert blk, "손게인 로그 블록이 robot_cfg 를 안 읽는다(또는 블록을 못 찾았다)"
    assert "robot_cfg" in blk.group(0), "로그가 조립된 robot_cfg 를 안 읽는다(프로필을 읽는다)"
    assert 'getattr(p, "actuator_specs"' not in blk.group(0), \
        "로그가 아직 프로필을 읽는다 — override 가 로그에 안 보인다"


# ---------------------------------------------------------------- CUDA Graph (09.10)
def test_graph_on_off_share_one_set_features_call():
    """ON/OFF 가 **같은** `set_features` 호출을 쓴다 = 두 경로의 수치가 동일하다.

    원본(`capture_fabric`)은 set_features + step 을 한 덩어리로 캡처해 재생마다
    set_features 를 다시 돌린다. 그러면 의미가 달라지고(부분스텝마다 특징 재계산)
    일도 더 한다 — 09.10 실측 1024env fabric 12.30 → 17.15 ms. 호출을 분기 **밖**에
    한 번만 두는 것이 그 두 문제를 동시에 없앤 계약이다.
    """
    blk = _fn_block(_code(_CTL), "_step_fabric")
    assert blk.count("self._set_fabric_features()") == 1, "set_features 호출은 정확히 한 번"
    assert blk.index("self._set_fabric_features()") < blk.index("self._fabric_graph is None"), \
        "set_features 가 ON/OFF 분기보다 앞(= 공통 경로)에 있어야 한다"
    assert "set_features(" not in blk, "분기 안에서 fabric.set_features 를 직접 부르면 안 된다"


def test_only_integrator_is_captured():
    """캡처 대상은 `integrator.step` 뿐 — set_features 가 그래프 안에 들어가면 안 된다."""
    blk = _fn_block(_code(_CTL), "_setup_fabric_graph")
    assert "torch.cuda.graph(" in blk, "캡처 블록이 없다 — 플래그만 있고 이득이 0인 반쪽 배선"
    # ★캡처 **블록만** 자른다. 함수 끝까지 자르면 뒤따르는 print 의 설명 문자열이
    #   금지어 검사에 걸린다(설명문이 계약을 깨뜨리는 전형).
    cap = blk[blk.index("torch.cuda.graph("):blk.index("self._graph_state_ptrs")]
    assert "self.integrator.step(" in cap, "적분을 캡처하지 않는다"
    assert "set_features" not in cap, "set_features 가 캡처 블록 안에 있다(재생마다 중복 실행)"
    # 캡처 전 1회 호출로 fabric 내부 특징 버퍼 주소를 확정시켜야 한다.
    assert blk.index("self._set_fabric_features()") < blk.index("torch.cuda.graph("), \
        "캡처 전에 set_features 를 한 번 불러 내부 버퍼 주소를 확정해야 한다"


def test_replay_copies_state_back_in_place():
    """재생 후 상태는 `copy_` 로 되받는다 — 재대입하면 다음 재생이 옛 메모리를 읽는다."""
    blk = _fn_block(_code(_CTL), "_step_fabric")
    on = blk.split("else:")[1]
    for name in ("fabric_q", "fabric_qd", "fabric_qdd"):
        assert f"self.{name}.copy_(self.{name}_new)" in on, f"{name} 를 copy_ 로 안 받는다"
    assert "self.fabric_q, self.fabric_qd, self.fabric_qdd =" not in on, \
        "ON 경로에서 상태를 재대입한다"


def test_graph_address_guard_exists_and_is_on_by_default():
    """조용한 실패를 시끄러운 실패로. 이 버그의 유일한 증상은 '에러 없이 팔이 고정'이다."""
    assert "fabric_graph_guard: bool = True" in _code(_CFG), "주소 가드 기본값이 True 가 아니다"
    guard = _fn_block(_CTL, "_assert_graph_addresses")
    assert "data_ptr()" in guard and "raise RuntimeError" in guard
    assert "self._assert_graph_addresses()" in _fn_block(_CTL, "_step_fabric"), \
        "가드가 스텝에서 안 불린다"


def test_fabric_world_is_static_so_graph_survives_multi_object():
    """다물체 전제. fabric 세계는 테이블 하나뿐이고 setup 이후 갱신되지 않아야 한다.

    갱신되면 `_world_ids`/`_world_indicator` 가 재대입되어 캡처된 그래프가 옛 주소를
    읽는다. 파지 대상 물체는 fabric 입력이 아니므로 뱅크가 몇 종이든 무관해야 한다.
    """
    code = _code(_CTL)
    assert code.count("self._world_ids, self._world_indicator =") == 1, \
        "world id 를 두 번 이상 대입한다 — 갱신되면 그래프가 깨진다"
    world = _fn_block(_CTL, "_build_fabric_world")
    assert '"env_index": "all"' in world, "fabric 장애물이 env 별로 갈리면 다물체에서 흔들린다"


def test_fabric_profile_timer_is_opt_in_and_avoids_sync():
    """fabric 몫을 재는 유일한 수단. 단 같은 스텝에서 읽으면 동기화가 측정을 오염시킨다."""
    assert "fabric_profile: bool = False" in _code(_CFG), "프로파일 기본값은 OFF 여야 한다"
    stop = _fn_block(_CTL, "_fabric_timer_stop")
    assert "_fabric_ev_prev" in stop and ".query()" in stop, \
        "직전 스텝 쌍을 query() 로 확인하고 읽어야 한다(동기화 금지)"


# ---------------------------------------------------------------- 외란 DR (09.10 신설)
def test_wrench_gate_is_height_latch_not_contact_latch():
    """★외란 게이트는 **높이 래치**(`_lifted`)여야 한다. 접촉 래치(`_latched`)면 안 된다.

    이식원 `grasp_kp` 는 같은 호출을 `self._latched` 로 쓰는데, **kp 의 그 이름은 높이
    래치**이고 s2r 의 `_latched` 는 접촉 래치다(이름만 같고 뜻이 반대다). 그대로 베끼면
    컵이 아직 테이블에 있는 접근 구간에 외란이 발화해 `cup_disp` 가 approach 순벌점을
    만들고, `disp_at_latch` 가 오염돼 lift·transfer·success_bonus 에 곱해지는 감쇠
    계수를 통째로 망친다. 이 테스트가 그 이식 사고를 잠근다.
    """
    # ★조각을 `_code()` 에 넘기지 않는다 — 그 헬퍼는 ast.parse 라 함수 조각에서 죽는다.
    #   대신 호출식 한 줄을 정확히 잡는다(조각 파싱 없이 조건을 검사한다).
    m = re.search(r"self\._wrench\.step\(([^)]*)\)", _ENV)
    assert m, "`_wrench.step` 호출 부재"
    args = m.group(1)
    assert "self._lifted" in args, (
        f"외란 게이트가 `self._lifted`(높이 래치)가 아니다: step({args})")
    assert "_latched" not in args, (
        f"외란이 접촉 래치를 게이트로 쓰고 있다 — kp 이식 사고 재발: step({args})")


def test_wrench_state_is_rewound_on_reset_and_respawn():
    """외란 상태(env 별 발화확률)와 높이 래치는 **리셋과 재소환 양쪽**에서 되감긴다.

    재소환은 컵을 스폰점으로 되돌리는 단계 되감기인데, 여기서 외란을 안 끄면 테이블에
    놓인 컵에 외란이 계속 걸린다(단계 되감기가 불완전해진다).
    """
    code = _code(_ENV)
    assert code.count("self._wrench.reset(") >= 2, (
        "`_wrench.reset` 이 리셋·재소환 양쪽에 있어야 한다")
    assert "self._lifted[env_ids] = False" in code, "리셋에서 높이 래치를 안 지운다"
    assert "self._lifted[_go] = False" in code, "재소환에서 높이 래치를 안 지운다"


def test_wrench_is_w1_and_validated():
    """외란은 **W1(5.0 / 0.5)** 로 승격됐고, 부팅 검증이 no-op·반쪽 설정을 막는다.

    ★★09.10 — 구판은 "기본값은 항등(끔)"을 리터럴 `0.0` 두 개로 잠갔다. 그 규칙은
      **기능을 처음 넣을 때** 맞는 것이지 영구 계약이 아니다(현행 동작 보존이 목적이었다).
      E1 이 W1 을 기본으로 승격하면서 그 단언이 올바른 변경을 막는 유일한 장애물이
      됐다 — 지우지 않고 **뒤집어서** 승격을 사유와 함께 잠근다(관측 0179 부수 규칙).

    승격 사유: 과제 정의가 "다양한 무게, **흔들림**에서 파지를 잃지 않는 것"이고,
      질량 DR 만으로는 흔들림이 생기지 않는다. W1 은 질량 정규화(N/kg)라 질량 DR
      (0.5~2.5배)과 간섭하지 않는다 — 힘/무게 비가 전 구간 0.81 로 일정하다.
    ⚠W2(20.0 / 2.0) 로 올릴 때는 래치 전후 수입비를 **다시** 계산할 것.
      W1 의 원 audit 은 D3 의 5.3배를 근거로 했는데 E1 명목은 2.30배다(1.72배 축소).

    무엇이 이 계약을 거짓으로 만드는가: 외란이 파지 품질을 강제하지 않는다는 실측.
      그때는 W1 을 끄는 근거를 남기고 이 단언을 다시 뒤집을 것.
    """
    assert "wrench_force_scale: float = 5.0" in _CFG, "외란이 W1(5.0 N/kg)이 아니다"
    assert "wrench_torque_scale: float = 0.5" in _CFG, "외란이 W1(0.5 N·m/kg)이 아니다"
    code = _code(_CFG)
    assert "_assert_wrench_sane" in code, "외란 부팅 검증이 없다"
    # 한쪽만 켜는 것은 CLI 오타의 전형이다 — 반드시 fail-loud.
    assert "(_f > 0.0) != (_t > 0.0)" in code, "force/torque 반쪽 설정 가드가 없다"


def test_wrench_firing_is_observable_in_tb():
    """조용한 no-op 방지 3중 확인 — 부팅 로그·런 dump·TB 태그.

    cfg 오버라이드가 실렸다는 것과 **실제로 발화했다**는 것은 다른 주장이라, 발화 자체를
    남긴다(관측 [0023]: dump 는 값이 설정된 것을 증명하지 그것이 존중된 것을 증명하지 않는다).
    """
    code = _code(_ENV)
    for tag in ('"dr/wrench_fire_frac"', '"dr/wrench_f_norm_max"'):
        assert tag in code, f"외란 발화 지표 {tag} 가 없다"


def test_stay_break_cause_is_split_into_two_tags():
    """`stay_run` 이 끊긴 원인을 `stable` 과 `n_grip` 으로 갈라 남긴다.

    D3 실측 stay_run 14.46 / 60 (24%) 의 원인이 손 미세 진동인지 접촉 들락거림인지에 따라
    처방이 갈린다. 후자면 그것이 곧 power grip 결손이다. 이 두 태그 없이
    `stay_weight` 를 만지는 것은 금지(reward-audit Check 5).
    """
    code = _code(_ENV)
    for tag in ('"gate/stay_break_by_stable"', '"gate/stay_break_by_grip"'):
        assert tag in code, f"stay 끊김 원인 지표 {tag} 가 없다"


# ---------------------------------------------------------------- 보상 회계 (09.10 Phase 0)
def test_reward_term_set_is_closed_both_ways():
    """`terms` 딕셔너리 키와 `GRASP_S2R_REWARD_TERMS` 튜플이 **양방향으로** 같아야 한다.

    ★기존 `rewards.py` 의 검사는 `set(TERMS) - set(terms)` 뿐이라 **누락만** 잡는다.
      새 항을 `terms` 에 넣고 튜플에 안 넣으면 통과하고, 그 항은 `reward/*` 로그에서
      조용히 빠진다(로깅 루프가 튜플을 돈다). 반대 방향을 여기서 잠근다.
    """
    body = re.search(r"terms = \{([\s\S]*?)\n    \}", _REW)
    assert body, "terms 딕셔너리 부재"
    keys = set(re.findall(r'"([a-z_]+)":', body.group(1)))
    tup = re.search(r"GRASP_S2R_REWARD_TERMS[^=]*=\s*\(([\s\S]*?)\)", _REW)
    assert tup, "GRASP_S2R_REWARD_TERMS 부재"
    names = set(re.findall(r'"([a-z_]+)"', tup.group(1)))
    assert keys == names, (
        f"terms 키와 계약 튜플이 다르다 — 튜플에만: {names - keys} · terms 에만: {keys - names}")


def test_penalties_outside_terms_are_still_logged():
    """`terms` 밖에서 `total` 에 더해지는 벌점은 **전부 자기 태그를 가져야** 한다.

    `abnormal_penalty`·`respawn_penalty` 는 `compute_grasp_s2r_rewards` 반환값 **뒤에**
    더해지므로 `GRASP_S2R_REWARD_TERMS` 루프를 안 탄다. 태그가 없으면 `total` 에는
    들어가는데 어떤 그래프에도 안 보인다 — 09.10 이전 `abnormal_penalty` 가 그 상태였다.
    무엇이 이 계약을 깨는가: `total` 에 더해지는 새 항을 만들면서 태그를 안 붙일 때.
    """
    # ★단언을 **코드 모양**(`total = total + …`)이 아니라 **조건**으로 쓴다.
    #   09.10 에 이 테스트 자신이 그 함정에 걸렸다: `_abn = …; total = total + _abn` 로
    #   두 줄로 나누자 정규식이 죽었다. 벌점을 한 줄에 쓰든 변수로 빼든 계약은 같아야 한다.
    #   조건: env.py 가 소비하는 모든 `*_penalty` cfg 노브는 자기 태그를 가진다.
    code = _code(_ENV)
    knobs = set(re.findall(r"cfgn?\.([a-z_]+_penalty)\b", code)) | set(
        re.findall(r'getattr\(\s*cfgn?\s*,\s*"([a-z_]+_penalty)"', code))
    assert knobs, "env.py 가 소비하는 *_penalty 노브를 못 찾았다 — 명명 규약이 바뀌었나?"
    for name in sorted(knobs):
        assert f'"reward/{name}"' in code, (
            f"`{name}` 을 env.py 가 소비하는데 `reward/{name}` 태그가 없다 — "
            "total 에는 들어가고 어떤 그래프에도 안 보이는 보상이 된다")


def test_conditional_tags_divide_only_by_binary_gates():
    """조건부 태그(`reward_cond/*`)는 **이진** 마스크로만 나눈다.

    `Σ항 / Σ마스크` 가 조건부 평균과 같으려면 마스크가 0/1 이어야 한다. 연속 게이트
    (`graded_contact`·`close_gate` 등)로 나누면 E[XY]/E[Y] ≠ E[X|Y] 라 틀린 값이 나온다.
    `pre_lift = 1 − lift_latched.float()` · `lift = lift_latched.float()` 만 이진이다.
    """
    code = _code(_ENV)
    m = re.search(r'_pre = gates\["(\w+)"\]; _lat = gates\["(\w+)"\]', code)
    assert m, "조건부 태그의 마스크 출처를 못 찾았다"
    assert set(m.groups()) == {"pre_lift", "lift"}, (
        f"이진이 아닌 게이트로 나누고 있다: {m.groups()}")


def test_lift_plateau_is_instrumented():
    """`lift` 높이 품질의 **포화**를 측정하는 태그가 있어야 한다.

    `lift_height_quality = clamp(dz / lift_height_ref)` 는 목표 반경 안 최소 높이차
    (goal_z − tolerance)가 `lift_height_ref` 보다 크면 목표까지 내내 1.0 이다 —
    가중 30 짜리 최대 항이 그 구간에서 경사 0 이라는 뜻이고, Phase 2(절단)의 전제다.
    산술로 확정되지만 **런이 그 구간에 얼마나 앉아 있는지**는 측정해야 안다.
    """
    code = _code(_ENV)
    for tag in ('"task/lift_quality_sat_frac"', '"task/height_delta_p10"'):
        assert tag in code, f"lift 평지 계측 태그 {tag} 가 없다"


def test_lift_ramp_reaches_the_goal_height():
    """`lift` 의 높이 램프는 **목표 반경 안 최소 높이차까지 단조**여야 한다.

    `lift_height_quality = clamp(dz / lift_height_ref)` 이므로
    `lift_height_ref < goal_z − goal_pos_tolerance` 이면 그 사이 구간에서 항이 상수가
    되고, 가중치만큼이 **위치 무관 수입**이 된다(D3 실측: 20.049/step = 총보상 36.6%).

    무엇이 이 계약을 거짓으로 만드는가: `goal_offset_xyz.z` 를 키우거나
    `goal_pos_tolerance` 를 줄이면서 `lift_height_ref` 를 그대로 두는 것.
    ★리터럴이 아니라 **부등식**을 잠근다 — 세 값이 같이 움직여도 살아남는다.
    """
    cfg = _code(_CFG)
    ref = float(re.search(r"lift_height_ref:\s*float\s*=\s*([\d.]+)", cfg).group(1))
    gz = float(re.search(
        r"goal_offset_xyz:\s*tuple\[float, float, float\]\s*=\s*\([^)]*?,\s*([\d.]+)\)",
        cfg).group(1))
    tol = float(re.search(r"goal_pos_tolerance:\s*float\s*=\s*([\d.]+)", cfg).group(1))
    assert ref >= gz - tol - 1e-9, (
        f"lift 램프가 목표 전에 포화한다: ref={ref} < goal_z({gz}) − tol({tol}) = {gz - tol:.4f} "
        "— 그 구간에서 lift_weight 전액이 위치 무관 상수 수입이 된다")


def test_approach_still_is_blended_not_a_bare_product():
    """`palm_still` 은 approach 에 **블렌드**로 들어가야 한다(순수 곱셈 금지).

    순수 곱셈은 구조적 긴장을 만든다 — 다가가려면 움직여야 하는데 움직이면 깎인다
    (0.05 m/s 에서 ×0.61). D3 실측에서 이 항의 실집행이 명목의 1.1% 였다.
    floor 는 리터럴이 아니라 cfg 필드에서 와야 한다.

    무엇이 이 계약을 거짓으로 만드는가: `palm_still` 자체를 폐기하는 결정.
    그때는 이 단언을 지우지 말고 "approach 에 속도 항이 없어야 한다"로 뒤집을 것.
    """
    rw, cfg = _code(_REW), _code(_CFG)
    i = rw.index("approach = pre_lift_gate")
    blk = rw[i:i + 400]
    assert "_still_eff" in blk, "approach 가 palm_still 을 그대로 곱하고 있다"
    assert 'approach_still_floor' in rw and "approach_still_floor: float" in cfg, \
        "still floor 가 cfg 필드가 아니다"
    # 블렌드 형태: floor + (1 − floor)·palm_still
    assert re.search(r"_still_eff\s*=\s*_sfl\s*\+\s*\(1\.0\s*-\s*_sfl\)\s*\*\s*palm_still",
                     rw), "블렌드 식이 floor + (1−floor)·still 형태가 아니다"


def test_adr_physics_base_comes_from_eventcfg_not_a_literal():
    """물리 DR 의 level-0 값은 **cfg 가 적은 범위**여야 한다.

    `_adr_apply_physics` 는 `enable_adr` 와 무관하게 `__init__` 에서 한 번 돈다.
    구판 `_lerp_range` 는 base 를 (1,1) 로 하드코딩해서, `EventCfg` 에 적어둔 질량·게인
    범위를 부팅 시점에 (1,1) 로 **조용히 덮어썼다** — ADR 을 안 쓰면서 고정 물리 DR 을
    여는 것이 원리적으로 불가능했다.

    무엇이 이 계약을 거짓으로 만드는가: ADR 이 물리축을 `EventCfg` 가 아닌 다른 경로로
    쓰게 되는 것. 그때는 그 경로에 같은 검사를 옮길 것.
    """
    env = _code(_ENV)
    assert "def _lerp_range(terminal, lvl: float, base=(1.0, 1.0))" in env, \
        "_lerp_range 가 base 를 인자로 받지 않는다"
    i = env.index("def _adr_apply_physics")
    blk = env[i:env.index("\n    def ", i + 10)]
    assert "_adr_phys_base" in blk, "부팅 시점 EventCfg 값을 base 로 캡처하지 않는다"
    assert blk.count("base=self._adr_phys_base") == 2, \
        "질량·게인 두 축 모두 캡처한 base 에서 보간해야 한다"
    # 순수함수 수치 검증 — base 를 주면 그 값에서 출발한다.
    import ast as _ast
    tree = _ast.parse(env)
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == "_lerp_range")
    ns: dict = {}
    exec(compile(_ast.Module(body=[fn], type_ignores=[]), "<t>", "exec"), ns)
    lerp = ns["_lerp_range"]
    assert lerp((2.0, 3.0), 0.0) == (1.0, 1.0), "기본 base 가 (1,1) 이 아니다(하위호환 깨짐)"
    assert lerp((0.5, 2.5), 0.0, base=(0.5, 2.5)) == (0.5, 2.5), \
        "level 0 에서 cfg 범위가 보존되지 않는다"


def test_physics_dr_effective_range_is_logged_from_the_consumer():
    """물리 DR 의 **실효 범위**는 `event_manager` 에서 읽어 로깅해야 한다.

    `params/env.yaml` dump 는 `self.cfg` 를 찍는데, 실제로 물리에 쓰이는 것은
    `ManagerBase` 가 deepcopy 해 간 `event_manager` 쪽 사본이다. 09.10 이전
    `_adr_apply_physics` 는 그 사본만 (1,1) 로 덮어썼으므로 **dump 는 버그가 있어도
    옳아 보였다** — cfg 에 적은 (0.5, 2.5) 가 그대로 찍히고 물리는 (1,1) 로 돌았다.
    그래서 소스가 아니라 **소비되는 자리**에서 읽은 값을 태그로 남긴다.

    무엇이 이 계약을 거짓으로 만드는가: 물리 DR 이 EventManager 를 떠나는 것.
    """
    env = _code(_ENV)
    for tag in ('"dr/mass_scale_lo"', '"dr/mass_scale_hi"',
                '"dr/gain_scale_lo"', '"dr/gain_scale_hi"'):
        assert tag in env, f"실효 DR 범위 태그 {tag} 가 없다"
    i = env.index('self.extras["dr/mass_scale_lo"]')
    blk = env[max(0, i - 700):i]
    assert "get_term_cfg(\"object_scale_mass\")" in blk, \
        "질량 실효범위를 cfg 가 아니라 event_manager 에서 읽어야 한다"


def test_wrench_mass_is_refreshed_when_mass_dr_is_open():
    """질량 DR 이 열려 있으면 외란의 질량 기준은 **리셋마다 실제 질량**이어야 한다.

    외란은 질량 정규화(N/kg)다. 공칭 질량(`default_mass`)을 `__init__` 에서 한 번만 읽으면
    scale (0.5, 2.5) 구간에서 힘/무게 비가 0.32~1.62 로 흔들려 **두 DR 축이 서로를
    상쇄한다** — 무거운 개체일수록 상대적으로 약한 외란을 받는다.
    질량 DR 은 reset 모드 이벤트라 `super()._reset_idx` 안에서 적용되므로, 그 직후에
    다시 읽어야 한다.

    ★이 계약은 **조건부**다 — 질량 DR 이 항등이면 공칭값으로 충분하고 갱신은 불필요하다.
      그래서 cfg 를 먼저 보고, 열려 있을 때만 갱신을 요구한다. (구판 docstring 이
      이 조건을 정확히 적어 두었는데 코드가 아니라 산문이라, DR 을 여는 사람이
      그 문장을 읽지 않으면 조용히 깨졌다 — 그래서 테스트로 옮긴다.)
    """
    cfg, env = _code(_CFG), _code(_ENV)
    m = re.search(r'"mass_distribution_params":\s*\(([\d.]+),\s*([\d.]+)\)', cfg)
    assert m, "질량 DR 항을 못 찾았다"
    if float(m.group(1)) == 1.0 and float(m.group(2)) == 1.0:
        return                                    # 질량 DR 꺼짐 — 갱신 불필요
    blk = _fn_block(env, "_reset_idx")
    assert "self._obj_mass = self._read_obj_mass()" in blk, (
        "질량 DR 이 열려 있는데 _reset_idx 가 외란 질량 기준을 갱신하지 않는다")
    assert blk.index("super()._reset_idx") < blk.index("self._obj_mass ="), (
        "질량 갱신이 reset 이벤트 적용(super()._reset_idx) 보다 먼저다 — 옛 값을 읽는다")
    rd = _fn_block(env, "_read_obj_mass")
    assert "root_physx_view" in rd and "get_masses" in rd, (
        "실제 질량이 아니라 공칭(default_mass)만 읽는다")
    # 실효 질량이 로깅돼야 DR 이 물리에 닿았는지 확인할 수 있다.
    for tag in ('"dr/obj_mass_min"', '"dr/obj_mass_max"'):
        assert tag in env, f"외란 정규화에 쓰인 실효 질량 태그 {tag} 가 없다"


def test_mass_dr_is_open_and_adr_never_narrows_it():
    """무게 DR 이 열려 있어야 한다 — power grip 과제의 정의다(사용자 확정).

    "다양한 무게, 흔들림에서 그 파지를 잃지 않는 것"이 이 트랙의 목적인데 질량 축이
    (1.0, 1.0) 으로 꺼져 있었고, `cup_family` 8종도 전부 같은 0.134 kg 이었다.
    그리고 ADR 종점이 base 보다 **좁으면** 승급할수록 쉬워지는 역방향 축이 된다 —
    질량축은 `_assert_adr_monotonic` 대상이 아니라 여기서만 잡힌다.
    """
    cfg = _code(_CFG)
    m = re.search(r'"mass_distribution_params":\s*\(([\d.]+),\s*([\d.]+)\)', cfg)
    assert m, "질량 DR 항을 못 찾았다"
    lo, hi = float(m.group(1)), float(m.group(2))
    assert not (lo == 1.0 and hi == 1.0), "질량 DR 이 꺼져 있다 — power grip 이 강제되지 않는다"
    assert lo < hi, f"질량 범위가 뒤집혔다: ({lo}, {hi})"
    a = re.search(r"adr_mass_scale_max:\s*tuple\[float, float\]\s*=\s*\(([\d.]+),\s*([\d.]+)\)", cfg)
    assert a, "adr_mass_scale_max 를 못 찾았다"
    alo, ahi = float(a.group(1)), float(a.group(2))
    assert alo <= lo and ahi >= hi, (
        f"ADR 종점 ({alo}, {ahi}) 이 base ({lo}, {hi}) 보다 좁다 — 승급이 과제를 쉽게 만든다")


def test_fk_gate_subtracts_frame_offset():
    """palm FK 게이트는 **좌표계 오프셋을 뺀 뒤** 비교해야 한다.

    `_fab_to_env` 는 손끝 5개로 바로 위에서 측정한 상수다(env = fab + offset).
    그걸 빼지 않으면 게이트가 두 오차를 섞어 잰다 — ①좌표계 원점 차이(기구학과 무관,
    이미 측정됨) ②진짜 기구학 불일치(잡고 싶은 것). 섞으면 ⓐ상수가 허용 예산을 잠식하고
    ⓑ오프셋이 크면 기구학이 완벽해도 죽고 ⓒ오프셋이 기구학 오차를 상쇄해 가릴 수 있다.

    무엇이 이 계약을 거짓으로 만드는가: 두 프레임이 순수 평행이동이 아니게 되는 것.
    그때는 회전 정합이 필요하고 평행이동 보정만으로는 부족하다 — 바로 위 산포 검사
    (`spread > 2e-3` 이면 부팅 거부)가 그 조건을 지킨다. 그 검사를 지우면 이 보정도 무효다.
    """
    code = _code(_CTL)
    m = re.search(r"dp\s*=\s*float\(torch\.norm\(([^\n]*)\)\)", code)
    assert m, "palm FK 게이트의 위치 오차 계산식을 못 찾았다"
    expr = m.group(1)
    assert "_fab_to_env" in expr, (
        f"FK 게이트가 좌표계 오프셋을 빼지 않는다: dp = norm({expr})")
    # 산포 검사(평행이동 가정의 근거)가 살아 있어야 이 보정이 정당하다.
    assert "spread > 2e-3" in code or "spread &gt; 2e-3" in code, (
        "평행이동 가정을 지키는 산포 검사가 사라졌다 — 그러면 오프셋 보정도 무효다")
    # 허용치는 리터럴이 아니라 cfg 에서 온다.
    assert "self.cfg.fabric_fk_pos_tol" in code, "FK 허용치가 cfg 필드가 아니다"
